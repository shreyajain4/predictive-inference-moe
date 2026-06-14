// moe_cuda_expert_cache.cu
// -------------------------
// Implementation. Pool of N slots × max_expert_size, LRU eviction, hash map.
// Three "kinds" of experts (gate/up/down) — each slot stores ONE kind for
// one (layer, expert) pair. So total entries = 3 × n_layers × n_experts.

#include "moe_cuda_expert_cache.h"

#include "ggml-backend.h"   // for ggml_set_moe_expert_cache_hook

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>
#include <list>
#include <unordered_map>
#include <vector>

namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[moe_cuda_expert_cache] CUDA error %s at %s:%d (%s)\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__, #call); \
    } \
} while(0)

// Key: (layer, expert, kind) packed into one uint64.
// 16 bits layer, 16 bits expert, 8 bits kind → fits easily.
struct expert_key {
    int layer;
    int expert;
    int kind;     // 0=gate, 1=up, 2=down

    bool operator==(const expert_key & o) const {
        return layer==o.layer && expert==o.expert && kind==o.kind;
    }
};
struct expert_key_hash {
    size_t operator()(const expert_key & k) const noexcept {
        return (size_t)k.layer * 13 + (size_t)k.expert * 257 + (size_t)k.kind * 65537;
    }
};

}  // namespace

struct moe_cuda_expert_cache {
    int n_layers = 0;
    int n_experts_per_layer = 0;
    size_t expert_size[3] = {0, 0, 0};   // gate, up, down
    size_t slot_size = 0;                // = max(expert_size[0..2])
    int n_slots = 0;

    // Device memory: contiguous slot pool
    uint8_t * pool = nullptr;             // device ptr, n_slots × slot_size

    // Slot bookkeeping (host-side)
    struct slot_info {
        bool occupied = false;
        expert_key key{-1, -1, -1};
        size_t bytes_used = 0;
        // LRU list iterator (only valid when occupied)
        std::list<int>::iterator lru_it;
    };
    std::vector<slot_info> slots;
    std::list<int> lru_order;             // front = most-recently-used, back = LRU
    std::unordered_map<expert_key, int, expert_key_hash> index;   // (key → slot_idx)

    std::mutex mu;                        // protects index / lru_order / slot_info

    // Streams
    cudaStream_t prefetch_stream = nullptr;

    // Stats (atomic)
    std::atomic<int64_t> stat_prefetches{0};
    std::atomic<int64_t> stat_prefetches_dedup{0};
    std::atomic<int64_t> stat_d2d_hits{0};
    std::atomic<int64_t> stat_d2d_partial_miss{0};
    std::atomic<int64_t> stat_evictions{0};
    std::atomic<int64_t> stat_bytes_prefetched{0};
    std::atomic<int64_t> stat_bytes_d2d_served{0};
};

// Global handle so the C hook function can find the cache.
// (Single cache per process — simplest design for the bench.)
static moe_cuda_expert_cache * g_cache = nullptr;

// ── Tensor name parser ──────────────────────────────────────────────────
// "blk.5.ffn_gate_exps.weight" → layer=5, kind=0 (gate)
// "blk.5.ffn_up_exps.weight"   → layer=5, kind=1
// "blk.5.ffn_down_exps.weight" → layer=5, kind=2
// Returns true on success.
static bool parse_tensor_name(const char * name, int & out_layer, int & out_kind) {
    if (!name) return false;
    if (strncmp(name, "blk.", 4) != 0) return false;
    const char * p = name + 4;
    int layer = 0;
    while (*p >= '0' && *p <= '9') { layer = layer*10 + (*p++ - '0'); }
    if (*p != '.') return false;
    p++;
    if      (strncmp(p, "ffn_gate_exps.weight", 20) == 0) { out_layer = layer; out_kind = 0; return true; }
    else if (strncmp(p, "ffn_up_exps.weight",   18) == 0) { out_layer = layer; out_kind = 1; return true; }
    else if (strncmp(p, "ffn_down_exps.weight", 20) == 0) { out_layer = layer; out_kind = 2; return true; }
    return false;
}

// ── Lifecycle ───────────────────────────────────────────────────────────

extern "C" moe_cuda_expert_cache * moe_cuda_expert_cache_create(
    int     n_layers,
    int     n_experts_per_layer,
    size_t  expert_size_gate,
    size_t  expert_size_up,
    size_t  expert_size_down,
    int     n_slots)
{
    if (n_layers <= 0 || n_experts_per_layer <= 0 || n_slots <= 0) return nullptr;

    auto * c = new moe_cuda_expert_cache();
    c->n_layers = n_layers;
    c->n_experts_per_layer = n_experts_per_layer;
    c->expert_size[0] = expert_size_gate;
    c->expert_size[1] = expert_size_up;
    c->expert_size[2] = expert_size_down;
    c->slot_size = std::max(std::max(expert_size_gate, expert_size_up), expert_size_down);
    c->n_slots = n_slots;

    size_t pool_bytes = (size_t)n_slots * c->slot_size;
    cudaError_t e = cudaMalloc((void**)&c->pool, pool_bytes);
    if (e != cudaSuccess) {
        fprintf(stderr,
            "[moe_cuda_expert_cache] cudaMalloc %.1f MiB pool failed: %s\n",
            pool_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
        delete c;
        return nullptr;
    }

    c->slots.resize(n_slots);
    c->index.reserve((size_t)n_layers * n_experts_per_layer * 3);

    CUDA_CHECK(cudaStreamCreateWithFlags(&c->prefetch_stream, cudaStreamNonBlocking));

    int dev = -1;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr,
        "[moe_cuda_expert_cache] pool %.1f MiB, %d slots × %zu bytes "
        "(gate=%zu up=%zu down=%zu) on %s (cc %d.%d)\n",
        pool_bytes / (1024.0*1024.0), n_slots, c->slot_size,
        expert_size_gate, expert_size_up, expert_size_down,
        prop.name, prop.major, prop.minor);

    return c;
}

extern "C" void moe_cuda_expert_cache_destroy(moe_cuda_expert_cache * c) {
    if (!c) return;
    if (c->prefetch_stream) cudaStreamSynchronize(c->prefetch_stream);
    if (c->prefetch_stream) cudaStreamDestroy(c->prefetch_stream);
    if (c->pool) cudaFree(c->pool);
    if (g_cache == c) g_cache = nullptr;
    delete c;
}

// ── Slot allocation + LRU ───────────────────────────────────────────────

// Find a free slot, or evict LRU. Caller must hold c->mu.
static int alloc_or_evict_slot(moe_cuda_expert_cache * c) {
    // First scan for a free slot
    for (int i = 0; i < c->n_slots; ++i) {
        if (!c->slots[i].occupied) return i;
    }
    // Else evict LRU
    if (c->lru_order.empty()) return -1;
    int victim_slot = c->lru_order.back();
    c->lru_order.pop_back();
    auto & vs = c->slots[victim_slot];
    if (vs.occupied) {
        c->index.erase(vs.key);
        vs.occupied = false;
        c->stat_evictions.fetch_add(1, std::memory_order_relaxed);
    }
    return victim_slot;
}

// Mark slot most-recently-used. Caller holds mu.
static void touch_slot(moe_cuda_expert_cache * c, int slot_idx) {
    auto & s = c->slots[slot_idx];
    if (s.occupied) {
        c->lru_order.erase(s.lru_it);
        c->lru_order.push_front(slot_idx);
        s.lru_it = c->lru_order.begin();
    }
}

// ── Prefetch ────────────────────────────────────────────────────────────

extern "C" int moe_cuda_expert_cache_touch(
    moe_cuda_expert_cache * c,
    int layer_id, int expert_id, int kind)
{
    if (!c || kind < 0 || kind > 2) return -1;
    expert_key key{layer_id, expert_id, kind};
    std::lock_guard<std::mutex> lk(c->mu);
    auto it = c->index.find(key);
    if (it == c->index.end()) return 0;   // not in cache, can't touch
    touch_slot(c, it->second);
    c->stat_prefetches_dedup.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int moe_cuda_expert_cache_prefetch(
    moe_cuda_expert_cache * c,
    int layer_id, int expert_id, int kind,
    const void * host_src, size_t nbytes)
{
    if (!c || kind < 0 || kind > 2 || !host_src) return 0;
    if (nbytes != c->expert_size[kind]) return 0;

    expert_key key{layer_id, expert_id, kind};
    int slot_idx = -1;
    {
        std::lock_guard<std::mutex> lk(c->mu);
        auto it = c->index.find(key);
        if (it != c->index.end()) {
            touch_slot(c, it->second);
            c->stat_prefetches_dedup.fetch_add(1, std::memory_order_relaxed);
            return 1;
        }
        slot_idx = alloc_or_evict_slot(c);
        if (slot_idx < 0) return 0;
        auto & s = c->slots[slot_idx];
        s.occupied = true;
        s.key = key;
        s.bytes_used = nbytes;
        c->lru_order.push_front(slot_idx);
        s.lru_it = c->lru_order.begin();
        c->index[key] = slot_idx;
    }
    void * dst = c->pool + (size_t)slot_idx * c->slot_size;
    CUDA_CHECK(cudaMemcpyAsync(dst, host_src, nbytes,
                                cudaMemcpyHostToDevice, c->prefetch_stream));
    c->stat_prefetches.fetch_add(1, std::memory_order_relaxed);
    c->stat_bytes_prefetched.fetch_add((int64_t)nbytes, std::memory_order_relaxed);
    return 1;
}

extern "C" void moe_cuda_expert_cache_drain(moe_cuda_expert_cache * c) {
    if (!c || !c->prefetch_stream) return;
    cudaStreamSynchronize(c->prefetch_stream);
}

// ── d2d hook (called by ggml-backend.cpp copy_experts) ──────────────────

extern "C" int moe_cuda_expert_cache_try_d2d(
    const char * tensor_name,
    int32_t first_expert_id, int32_t n_experts_in_run,
    void *  input_cpy_data,  size_t dst_offset,
    size_t  expert_size,     size_t total_bytes,
    uint64_t * out_uncached_bits)
{
    (void)total_bytes;
    if (!g_cache) return 0;
    if (!tensor_name || !input_cpy_data) return 0;
    if (n_experts_in_run <= 0 || n_experts_in_run > 64) return 0;  // partial-hit limited to uint64 mask
    if (!out_uncached_bits) return 0;

    // DEBUG override: skip cache entirely.
    if (const char * dbg = getenv("MOE_CACHE_FORCE_FALLTHROUGH")) {
        if (dbg[0] == '1') {
            g_cache->stat_d2d_partial_miss.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
    }

    int layer = -1, kind = -1;
    if (!parse_tensor_name(tensor_name, layer, kind)) return 0;

    moe_cuda_expert_cache * c = g_cache;
    const size_t per_expert = c->expert_size[kind];

    // If ggml's expert_size disagrees with what we stored when populating
    // the slot, the bytes won't line up. Bail entirely.
    if (expert_size != per_expert) {
        c->stat_d2d_partial_miss.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Look up every expert; collect slot indices for cached ones.
    int slot_indices[64];
    for (int i = 0; i < 64; ++i) slot_indices[i] = -1;
    int n_cached = 0;
    {
        std::lock_guard<std::mutex> lk(c->mu);
        for (int32_t i = 0; i < n_experts_in_run; ++i) {
            expert_key key{layer, first_expert_id + i, kind};
            auto it = c->index.find(key);
            if (it == c->index.end()) continue;   // bit i remains set in caller's mask
            slot_indices[i] = it->second;
            touch_slot(c, it->second);
            n_cached++;
        }
    }

    if (n_cached == 0) {
        c->stat_d2d_partial_miss.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    // Wait for any in-flight prefetches to land before we D→D from the slots.
    cudaStreamSynchronize(c->prefetch_stream);

    uint8_t * dst_base = (uint8_t *)input_cpy_data + dst_offset;
    size_t   bytes_served = 0;
    uint64_t bits_cleared = 0;
    for (int32_t i = 0; i < n_experts_in_run; ++i) {
        if (slot_indices[i] < 0) continue;
        const uint8_t * src = c->pool + (size_t)slot_indices[i] * c->slot_size;
        uint8_t *       dst = dst_base + (size_t)i * expert_size;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, expert_size,
                                    cudaMemcpyDeviceToDevice, /*default stream*/ 0));
        bits_cleared |= (uint64_t(1) << i);
        bytes_served += expert_size;
    }
    cudaDeviceSynchronize();

    *out_uncached_bits &= ~bits_cleared;
    c->stat_bytes_d2d_served.fetch_add((int64_t)bytes_served, std::memory_order_relaxed);

    const bool full = (n_cached == n_experts_in_run);
    if (full) {
        c->stat_d2d_hits.fetch_add(1, std::memory_order_relaxed);
        return 1;
    } else {
        // Partial: count this run as a partial-miss for stats purposes, but
        // we DID save some PCIe.
        c->stat_d2d_partial_miss.fetch_add(1, std::memory_order_relaxed);
        return 2;
    }
}

// Snapshot post-PCIe bytes from input_cpy into our cache slots (D→D). Called
// by ggml-backend.cpp AFTER ggml_backend_tensor_set_async writes input_cpy.
// We do a cudaDeviceSynchronize first to ensure the PCIe write completed
// (its on backend's stream, our snapshot is on default stream — without
// device sync the read might see stale bytes).
extern "C" void moe_cuda_expert_cache_snapshot(
    const char * tensor_name,
    int32_t first_expert_id, int32_t n_experts_in_run,
    void *  input_cpy_data,  size_t dst_offset,
    size_t  expert_size,     size_t total_bytes)
{
    (void)total_bytes;  // unused — we snapshot only the expert bytes, not padding
    if (!g_cache || !tensor_name || !input_cpy_data) return;

    int layer = -1, kind = -1;
    if (!parse_tensor_name(tensor_name, layer, kind)) return;
    moe_cuda_expert_cache * c = g_cache;
    const size_t per_expert = c->expert_size[kind];
    if (expert_size != per_expert) return;   // slot size mismatch, skip caching

    // Wait for the PCIe write on the backend's stream to complete. Coarse
    // but correct. Optimization later: use cudaStreamWaitEvent on a specific
    // event from the backend.
    cudaDeviceSynchronize();

    // Allocate slots for each expert in the run (or reuse if already cached).
    const uint8_t * src_base = (const uint8_t *)input_cpy_data + dst_offset;
    for (int32_t i = 0; i < n_experts_in_run; ++i) {
        expert_key key{layer, first_expert_id + i, kind};
        int slot_idx = -1;
        {
            std::lock_guard<std::mutex> lk(c->mu);
            auto it = c->index.find(key);
            if (it != c->index.end()) {
                // Already cached — refresh LRU; bytes should be identical
                // (PCIe wrote the same converted format the slot already has).
                touch_slot(c, it->second);
                c->stat_prefetches_dedup.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            slot_idx = alloc_or_evict_slot(c);
            if (slot_idx < 0) continue;
            auto & s = c->slots[slot_idx];
            s.occupied = true;
            s.key = key;
            s.bytes_used = expert_size;
            c->lru_order.push_front(slot_idx);
            s.lru_it = c->lru_order.begin();
            c->index[key] = slot_idx;
        }
        // D→D from input_cpy (already on GPU) to our pool slot.
        void *       dst = c->pool + (size_t)slot_idx * c->slot_size;
        const void * src = src_base + (size_t)i * expert_size;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, expert_size,
                                    cudaMemcpyDeviceToDevice, c->prefetch_stream));
        c->stat_prefetches.fetch_add(1, std::memory_order_relaxed);
        c->stat_bytes_prefetched.fetch_add((int64_t)expert_size, std::memory_order_relaxed);
    }
}

extern "C" void moe_cuda_expert_cache_install_hook(moe_cuda_expert_cache * c) {
    g_cache = c;
    ggml_set_moe_expert_cache_hook(moe_cuda_expert_cache_try_d2d);
    // Note: snapshot hook intentionally NOT installed. Cache is populated
    // exclusively via moe_cuda_expert_cache_prefetch (H→D from CPU mmap).
    // The L2 snapshot path was a workaround for a misdiagnosed bug.
}

// ── Stats ───────────────────────────────────────────────────────────────

extern "C" void moe_cuda_expert_cache_get_stats(
    const moe_cuda_expert_cache * c, moe_cuda_expert_cache_stats * out)
{
    if (!c || !out) return;
    out->prefetches         = c->stat_prefetches.load(std::memory_order_relaxed);
    out->prefetches_dedup   = c->stat_prefetches_dedup.load(std::memory_order_relaxed);
    out->d2d_hits           = c->stat_d2d_hits.load(std::memory_order_relaxed);
    out->d2d_partial_miss   = c->stat_d2d_partial_miss.load(std::memory_order_relaxed);
    out->evictions          = c->stat_evictions.load(std::memory_order_relaxed);
    out->bytes_prefetched   = c->stat_bytes_prefetched.load(std::memory_order_relaxed);
    out->bytes_d2d_served   = c->stat_bytes_d2d_served.load(std::memory_order_relaxed);
}

extern "C" void moe_cuda_expert_cache_print_stats(
    const moe_cuda_expert_cache * c, const char * tag)
{
    if (!c) return;
    moe_cuda_expert_cache_stats s{};
    moe_cuda_expert_cache_get_stats(c, &s);
    const int64_t total = s.d2d_hits + s.d2d_partial_miss;
    const double hr = total > 0 ? 100.0 * (double)s.d2d_hits / (double)total : 0.0;
    fprintf(stderr,
        "[moe_cuda_expert_cache %s] prefetches=%lld dedup=%lld evictions=%lld "
        "| d2d_hits=%lld d2d_misses=%lld d2d_hit_rate=%.2f%% "
        "| bytes_prefetched=%.2f MiB bytes_d2d_served=%.2f MiB\n",
        tag ? tag : "",
        (long long)s.prefetches, (long long)s.prefetches_dedup, (long long)s.evictions,
        (long long)s.d2d_hits, (long long)s.d2d_partial_miss, hr,
        s.bytes_prefetched / (1024.0*1024.0),
        s.bytes_d2d_served / (1024.0*1024.0));
}
