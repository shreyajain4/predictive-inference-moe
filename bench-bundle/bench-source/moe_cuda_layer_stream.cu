// moe_cuda_layer_stream.cu
// ------------------------
// CUDA port of moe_layer_stream.mm. Same C ABI (moe_layer_stream.h) — N
// circular CUDA device buffers, each holding one layer's expert payload.
// Round-robin: buf[L % N] holds layer L's data. Background worker thread
// issues cudaMemcpyAsync(host_src -> device buffer) over a dedicated stream
// and waits for completion (via cudaEvent) before marking the slot READY.
//
// Why CUDA stream + worker thread (rather than just enqueueing copies on
// any stream): keeps PCIe transfer truly overlapped with the compute stream
// the bench's ggml graph executes on. The worker thread blocks on
// cudaEventSynchronize so we can mark slot state atomically without polling.
//
// Memory: each buffer = layer_bytes of device memory (cudaMalloc). For
// Qwen3-30B-A3B Q8 full-layer: 612 MiB × N=2 = 1.2 GiB. T4 has 15 GiB total
// VRAM, ~1.5 GiB for non-expert weights → up to ~13 GiB / 612 MiB ≈ N=22
// possible.

#include "moe_layer_stream.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <vector>

namespace {

enum slot_state : int {
    SLOT_EMPTY   = 0,
    SLOT_LOADING = 1,
    SLOT_READY   = 2,
};

struct slot {
    std::atomic<int>  state{SLOT_EMPTY};
    std::atomic<int>  current_layer{-1};
    void *            device_ptr = nullptr;   // cudaMalloc'd device memory
    size_t            bytes_in_use = 0;
};

struct pending_segment {
    const void * src = nullptr;
    size_t       dst_offset = 0;
    size_t       nbytes = 0;
};
struct pending_request {
    std::vector<pending_segment> segments;
    bool has_request() const { return !segments.empty(); }
    void clear() { segments.clear(); }
};

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[moe_cuda_layer_stream] cuda error %s at %s:%d (%s)\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__, #call); \
    } \
} while (0)

}  // namespace

struct moe_layer_stream_impl {
    size_t                       n_buffers = 0;
    size_t                       layer_bytes = 0;
    std::vector<slot>            slots;
    std::vector<pending_request> pending;     // size = n_buffers
    std::mutex                   queue_mu;
    std::condition_variable      queue_cv;
    std::condition_variable      done_cv;
    std::thread                  worker;
    std::atomic<bool>            shutdown{false};
    cudaStream_t                 xfer_stream = nullptr;   // dedicated transfer stream

    // Stats
    std::atomic<int64_t> prefetches{0};
    std::atomic<int64_t> lookups_hit{0};
    std::atomic<int64_t> lookups_miss{0};
    std::atomic<int64_t> lookups_in_flight{0};
    std::atomic<int64_t> overwrites{0};
    std::atomic<int64_t> bytes_copied{0};
    std::atomic<int64_t> worker_busy_us{0};
    std::atomic<int64_t> wait_blocked_us{0};
};

namespace {

void worker_loop(moe_layer_stream_impl * c) {
    while (true) {
        size_t slot_idx = 0;
        pending_request req;

        {
            std::unique_lock<std::mutex> lk(c->queue_mu);
            c->queue_cv.wait(lk, [c, &slot_idx]() {
                if (c->shutdown.load(std::memory_order_relaxed)) return true;
                for (size_t i = 0; i < c->pending.size(); ++i) {
                    if (c->pending[i].has_request()) { slot_idx = i; return true; }
                }
                return false;
            });
            if (c->shutdown.load(std::memory_order_relaxed)) return;

            req = std::move(c->pending[slot_idx]);
            c->pending[slot_idx].clear();
        }

        // Issue async H2D copies, then synchronize the stream to know when
        // the slot is fully populated. Synchronizing here (worker thread)
        // keeps the main compute thread unblocked.
        auto t0 = std::chrono::steady_clock::now();
        slot & s = c->slots[slot_idx];
        size_t total_copied = 0;
        for (const auto & seg : req.segments) {
            if (seg.src == nullptr || seg.nbytes == 0) continue;
            size_t end = seg.dst_offset + seg.nbytes;
            size_t nbytes = seg.nbytes;
            if (end > c->layer_bytes) {
                if (seg.dst_offset >= c->layer_bytes) continue;
                nbytes = c->layer_bytes - seg.dst_offset;
            }
            CUDA_CHECK(cudaMemcpyAsync(
                (char *)s.device_ptr + seg.dst_offset,
                seg.src,
                nbytes,
                cudaMemcpyHostToDevice,
                c->xfer_stream));
            total_copied += nbytes;
        }
        CUDA_CHECK(cudaStreamSynchronize(c->xfer_stream));

        s.bytes_in_use = total_copied;
        c->bytes_copied.fetch_add((int64_t)total_copied, std::memory_order_relaxed);
        s.state.store(SLOT_READY, std::memory_order_release);

        auto t1 = std::chrono::steady_clock::now();
        c->worker_busy_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
            std::memory_order_relaxed);

        c->done_cv.notify_all();
    }
}

}  // namespace

// ── Construction / teardown ─────────────────────────────────────────────

extern "C" moe_layer_stream_impl * moe_layer_stream_create(size_t n_buffers, size_t layer_bytes) {
    if (n_buffers == 0 || layer_bytes == 0) return nullptr;

    auto * c = new moe_layer_stream_impl();
    c->n_buffers = n_buffers;
    c->layer_bytes = layer_bytes;
    c->slots = std::vector<slot>(n_buffers);
    c->pending.resize(n_buffers);

    // Use the highest-priority non-default stream to avoid waiting on the
    // bench's compute stream (the ggml-cuda backend will use stream 0).
    CUDA_CHECK(cudaStreamCreateWithFlags(&c->xfer_stream, cudaStreamNonBlocking));

    for (size_t i = 0; i < n_buffers; ++i) {
        void * ptr = nullptr;
        cudaError_t e = cudaMalloc(&ptr, layer_bytes);
        if (e != cudaSuccess) {
            fprintf(stderr,
                "[moe_cuda_layer_stream] cudaMalloc %zu MiB failed: %s\n",
                layer_bytes / (1024 * 1024), cudaGetErrorString(e));
            for (size_t j = 0; j < i; ++j) cudaFree(c->slots[j].device_ptr);
            cudaStreamDestroy(c->xfer_stream);
            delete c;
            return nullptr;
        }
        c->slots[i].device_ptr = ptr;
    }

    c->worker = std::thread(worker_loop, c);

    int dev = -1;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr,
        "[moe_cuda_layer_stream] %zu buffers × %zu bytes (total %.1f MiB device-resident, async worker) on %s (cc %d.%d)\n",
        n_buffers, layer_bytes,
        (n_buffers * layer_bytes) / (1024.0 * 1024.0),
        prop.name, prop.major, prop.minor);
    return c;
}

extern "C" void moe_layer_stream_destroy(moe_layer_stream_impl * c) {
    if (!c) return;

    {
        std::lock_guard<std::mutex> lk(c->queue_mu);
        c->shutdown.store(true, std::memory_order_relaxed);
    }
    c->queue_cv.notify_all();
    if (c->worker.joinable()) c->worker.join();

    for (auto & s : c->slots) {
        if (s.device_ptr) cudaFree(s.device_ptr);
        s.device_ptr = nullptr;
    }
    if (c->xfer_stream) cudaStreamDestroy(c->xfer_stream);
    delete c;
}

// ── Lookup / prefetch ───────────────────────────────────────────────────

extern "C" void * moe_layer_stream_get(moe_layer_stream_impl * c, int layer) {
    if (!c || c->n_buffers == 0) return nullptr;
    if (layer < 0) return nullptr;
    size_t slot_idx = (size_t)layer % c->n_buffers;
    slot & s = c->slots[slot_idx];

    int st = s.state.load(std::memory_order_acquire);
    int cl = s.current_layer.load(std::memory_order_acquire);

    if (st == SLOT_READY && cl == layer) {
        c->lookups_hit.fetch_add(1, std::memory_order_relaxed);
        return s.device_ptr;
    }
    if (st == SLOT_LOADING && cl == layer) {
        c->lookups_in_flight.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    c->lookups_miss.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

extern "C" void * moe_layer_stream_prefetch(
    moe_layer_stream_impl * c,
    int layer,
    const void * src,
    size_t nbytes)
{
    moe_layer_segment seg{ src, 0, nbytes };
    return moe_layer_stream_prefetch_segments(c, layer, &seg, 1);
}

extern "C" void * moe_layer_stream_prefetch_segments(
    moe_layer_stream_impl * c,
    int layer,
    const moe_layer_segment * segments,
    size_t n_segments)
{
    if (!c || !segments || n_segments == 0 || layer < 0) return nullptr;

    size_t slot_idx = (size_t)layer % c->n_buffers;
    slot & s = c->slots[slot_idx];

    if (s.state.load(std::memory_order_acquire) == SLOT_READY &&
        s.current_layer.load(std::memory_order_acquire) == layer) {
        return s.device_ptr;
    }

    {
        std::lock_guard<std::mutex> lk(c->queue_mu);
        int prev_layer = s.current_layer.load(std::memory_order_relaxed);
        if (prev_layer >= 0 && prev_layer != layer) {
            c->overwrites.fetch_add(1, std::memory_order_relaxed);
        }
        s.state.store(SLOT_LOADING, std::memory_order_release);
        s.current_layer.store(layer, std::memory_order_release);
        c->pending[slot_idx].clear();
        c->pending[slot_idx].segments.reserve(n_segments);
        for (size_t i = 0; i < n_segments; ++i) {
            c->pending[slot_idx].segments.push_back(pending_segment{
                segments[i].src, segments[i].dst_offset, segments[i].nbytes });
        }
        c->prefetches.fetch_add(1, std::memory_order_relaxed);
    }
    c->queue_cv.notify_one();
    return s.device_ptr;
}

extern "C" void * moe_layer_stream_wait(moe_layer_stream_impl * c, int layer) {
    if (!c || layer < 0) return nullptr;
    size_t slot_idx = (size_t)layer % c->n_buffers;
    slot & s = c->slots[slot_idx];

    if (s.state.load(std::memory_order_acquire) == SLOT_READY &&
        s.current_layer.load(std::memory_order_acquire) == layer) {
        c->lookups_hit.fetch_add(1, std::memory_order_relaxed);
        return s.device_ptr;
    }

    auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(c->queue_mu);
        c->done_cv.wait(lk, [&]() {
            int st = s.state.load(std::memory_order_acquire);
            int cl = s.current_layer.load(std::memory_order_acquire);
            if (st == SLOT_READY && cl == layer) return true;
            if (cl != layer) return true;
            return false;
        });
    }
    auto t1 = std::chrono::steady_clock::now();
    c->wait_blocked_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
        std::memory_order_relaxed);

    if (s.state.load(std::memory_order_acquire) == SLOT_READY &&
        s.current_layer.load(std::memory_order_acquire) == layer) {
        c->lookups_hit.fetch_add(1, std::memory_order_relaxed);
        return s.device_ptr;
    }
    c->lookups_miss.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

// ── Stats ───────────────────────────────────────────────────────────────

extern "C" void moe_layer_stream_get_stats(const moe_layer_stream_impl * c, moe_layer_stream_stats * out) {
    if (!c || !out) return;
    out->prefetches        = c->prefetches.load(std::memory_order_relaxed);
    out->lookups_hit       = c->lookups_hit.load(std::memory_order_relaxed);
    out->lookups_miss      = c->lookups_miss.load(std::memory_order_relaxed);
    out->lookups_in_flight = c->lookups_in_flight.load(std::memory_order_relaxed);
    out->overwrites        = c->overwrites.load(std::memory_order_relaxed);
    out->bytes_copied      = c->bytes_copied.load(std::memory_order_relaxed);
    out->worker_busy_us    = c->worker_busy_us.load(std::memory_order_relaxed);
    out->wait_blocked_us   = c->wait_blocked_us.load(std::memory_order_relaxed);
}

extern "C" void moe_layer_stream_reset_stats(moe_layer_stream_impl * c) {
    if (!c) return;
    c->prefetches.store(0, std::memory_order_relaxed);
    c->lookups_hit.store(0, std::memory_order_relaxed);
    c->lookups_miss.store(0, std::memory_order_relaxed);
    c->lookups_in_flight.store(0, std::memory_order_relaxed);
    c->overwrites.store(0, std::memory_order_relaxed);
    c->bytes_copied.store(0, std::memory_order_relaxed);
    c->worker_busy_us.store(0, std::memory_order_relaxed);
    c->wait_blocked_us.store(0, std::memory_order_relaxed);
}

extern "C" void moe_layer_stream_print_stats(const moe_layer_stream_impl * c, const char * tag) {
    if (!c) return;
    int64_t hits = c->lookups_hit.load(std::memory_order_relaxed);
    int64_t miss = c->lookups_miss.load(std::memory_order_relaxed);
    int64_t infl = c->lookups_in_flight.load(std::memory_order_relaxed);
    const double total_lookups = (double)(hits + miss + infl);
    const double hr = total_lookups > 0 ? 100.0 * (double)hits / total_lookups : 0.0;
    fprintf(stderr,
        "[moe_cuda_layer_stream %s] prefetches=%lld hits=%lld misses=%lld in_flight=%lld hit_rate=%.2f%% "
        "overwrites=%lld bytes_copied=%.2f MiB worker_busy=%.1f ms wait_blocked=%.1f ms\n",
        tag ? tag : "",
        (long long)c->prefetches.load(std::memory_order_relaxed),
        (long long)hits,
        (long long)miss,
        (long long)infl, hr,
        (long long)c->overwrites.load(std::memory_order_relaxed),
        c->bytes_copied.load(std::memory_order_relaxed) / (1024.0 * 1024.0),
        c->worker_busy_us.load(std::memory_order_relaxed) / 1000.0,
        c->wait_blocked_us.load(std::memory_order_relaxed) / 1000.0);
}
