// moe-predictor-bench.cpp
// -----------------------
// Mirrors examples/moe-trace/moe-trace.cpp but instead of writing trace CSV,
// runs the LR-hidden predictor in lockstep with real decode and measures:
//
//   - predictor accuracy (recall@k) on real-model decode
//   - simulated cache hit rate (what fraction of actually-fired experts
//     would have been resident if we'd prefetched the top-k each step)
//   - decode latency (with vs without the callback installed)
//
// This is INSTRUMENTATION ONLY — no actual prefetch DMA happens. The purpose
// is to validate predictor behavior in a real llama.cpp environment before
// wiring in the real prefetch path.
//
// All dimensions (n_layers, num_experts, hidden_dim, top_k, first_layer) are
// read from the .bin header at runtime — no model-specific compile-time consts.
//
// Build: drops into examples/moe-predictor-bench/ in the llama.cpp tree.
// See companion CMakeLists.txt.
//
// Usage:
//   ./llama-moe-predictor-bench -m model.gguf -p "prompt" -n 64 \
//       --predictor-weights ./qwen3_predictor_weights.bin \
//       --prefetch-k 12
//   ./llama-moe-predictor-bench -m model.gguf -p "prompt" -n 64 --no-callback

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include "moe_predictor.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fcntl.h>     // open
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/mman.h>  // mmap, posix_madvise, MADV_*
#include <sys/stat.h>  // fstat
#include <thread>
#include <unistd.h>    // close, pread
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task.h>
#include <mach/task_info.h>
#endif

// log_memory_state - logs the bench process's virtual size + RSS, plus
// system-wide page state (free/active/inactive/wired) to stderr.
// Called at key points to debug Metal OOM failures and memory pressure.
static void log_memory_state(const char * tag) {
#ifdef __APPLE__
    // Process-level: mach_task_basic_info
    struct mach_task_basic_info ti;
    mach_msg_type_number_t ti_count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                 (task_info_t)&ti, &ti_count);
    if (kr == KERN_SUCCESS) {
        fprintf(stderr,
            "[MEM %-20s] proc_virtual=%llu MiB  proc_rss=%llu MiB  rss_peak=%llu MiB\n",
            tag,
            (unsigned long long)(ti.virtual_size  / (1024ull * 1024ull)),
            (unsigned long long)(ti.resident_size / (1024ull * 1024ull)),
            (unsigned long long)(ti.resident_size_max / (1024ull * 1024ull)));
    }

    // System-level: host_statistics64 (vm_stat equivalent)
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t vm_count = HOST_VM_INFO64_COUNT;
    kr = host_statistics64(mach_host_self(), HOST_VM_INFO64,
                            (host_info64_t)&vmstat, &vm_count);
    if (kr == KERN_SUCCESS) {
        const uint64_t page_kb = 16; // M1 Pro page = 16 KiB
        fprintf(stderr,
            "[MEM %-20s] sys: free=%llu MiB active=%llu MiB inactive=%llu MiB "
            "wired=%llu MiB speculative=%llu MiB compressor=%llu MiB swapouts=%llu\n",
            tag,
            (unsigned long long)(vmstat.free_count        * page_kb / 1024ull),
            (unsigned long long)(vmstat.active_count      * page_kb / 1024ull),
            (unsigned long long)(vmstat.inactive_count    * page_kb / 1024ull),
            (unsigned long long)(vmstat.wire_count        * page_kb / 1024ull),
            (unsigned long long)(vmstat.speculative_count * page_kb / 1024ull),
            (unsigned long long)(vmstat.compressor_page_count * page_kb / 1024ull),
            (unsigned long long)(vmstat.swapouts));
    }
#else
    (void)tag;
#endif
}

// ───────────────────────────────────────────────────────────────────────
// State carried by the eval callback. One per llama_context.
// All array sizes are runtime, set after the predictor weights are loaded.
// ───────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────
// PrefetchWorker — background thread that issues pread() against the GGUF
// file to warm the kernel page cache for predicted expert ranges.
//
// Rationale: POSIX_MADV_WILLNEED on macOS appears to be synchronous-ish
// (measured: more hints -> slower decode). pread() in a dedicated thread is
// genuinely async w.r.t. the decode thread — the SSD read happens on the
// worker's CPU/I/O time, populating the page cache transparently. When the
// main thread's mmap touches the same offsets later, the pages are already
// resident.
//
// The scratch buffer's only purpose is to satisfy pread's "destination"
// argument; the read's BYPRODUCT (page-cache fill) is what we want.
// ────────────────────────────────────────────────────────────────────────
struct prefetch_task {
    off_t   offset;       // for pread mode
    size_t  length;
    char *  mmap_addr;    // for mmap-touch mode (may be null in pread mode)
};

struct prefetch_worker {
    int fd = -1;
    std::vector<char>          scratch;     // shared, reused per pread
    std::mutex                 mu;
    std::condition_variable    cv;
    std::deque<prefetch_task>  queue;
    std::atomic<bool>          shutdown{false};
    std::thread                t;
    int64_t                    pread_calls   = 0;
    int64_t                    pread_bytes   = 0;
    int64_t                    queue_drops   = 0;
    int                        max_queue_depth = 256;
    // mode: 0 = pread into scratch, 1 = touch mmap pages directly (active list)
    int                        mode = 0;
};

static void prefetch_worker_loop(prefetch_worker * w) {
    auto worker_start = std::chrono::steady_clock::now();
    int64_t total_pread_us = 0;
    int64_t max_pread_us   = 0;
    while (true) {
        std::unique_lock<std::mutex> lk(w->mu);
        w->cv.wait(lk, [w] { return !w->queue.empty() || w->shutdown.load(); });
        if (w->shutdown.load() && w->queue.empty()) {
            auto worker_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - worker_start).count();
            fprintf(stderr, "[pread_worker] total %lld pread calls, %.2f GB, "
                    "sum_pread_time=%.1f ms, max_pread_us=%lld, worker_lifetime=%lld ms\n",
                    (long long)w->pread_calls,
                    w->pread_bytes / (1024.0*1024.0*1024.0),
                    total_pread_us / 1000.0,
                    (long long)max_pread_us,
                    (long long)worker_total_ms);
            return;
        }
        prefetch_task task = w->queue.front();
        w->queue.pop_front();
        const int queue_depth_after_pop = (int)w->queue.size();
        lk.unlock();

        auto t0 = std::chrono::steady_clock::now();
        ssize_t n = 0;
        if (w->mode == 2 && task.mmap_addr != nullptr) {
            // mlock mode: page-fault + lock in one call. mlock pulls pages
            // from SSD if not resident, then PINS them so kernel can't evict.
            // Synchronous w.r.t. SSD read, but happens on worker thread, so
            // main decode thread is unblocked. After compute uses them, main
            // thread calls munlock() to release the lock budget.
            if (mlock(task.mmap_addr, task.length) == 0) {
                n = (ssize_t)task.length;
            } else {
                n = -1;
            }
        } else if (w->mode == 1 && task.mmap_addr != nullptr) {
            // mmap-touch mode: read one byte per page from the mmap'd file.
            const long page_size = sysconf(_SC_PAGESIZE);
            volatile char x = 0;
            for (size_t off = 0; off < task.length; off += page_size) {
                x += task.mmap_addr[off];
            }
            (void)x;
            n = (ssize_t)task.length;
        } else {
            const size_t to_read = std::min(task.length, w->scratch.size());
            n = pread(w->fd, w->scratch.data(), to_read, task.offset);
        }
        auto t1 = std::chrono::steady_clock::now();
        int64_t dt_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (n > 0) {
            w->pread_calls += 1;
            w->pread_bytes += n;
            total_pread_us += dt_us;
            if (dt_us > max_pread_us) max_pread_us = dt_us;
            // Log slow ops (>5 ms) to detect SSD-bound situations
            if (dt_us > 5000) {
                fprintf(stderr, "[pf_worker] slow op (mode=%d): %lld us, %zu bytes, "
                        "queue_depth_after=%d\n",
                        w->mode, (long long)dt_us, (size_t)n, queue_depth_after_pop);
            }
        }
    }
}

static inline void prefetch_enqueue(prefetch_worker * w, off_t off, size_t len,
                                     char * mmap_addr = nullptr) {
    {
        std::lock_guard<std::mutex> lk(w->mu);
        if ((int)w->queue.size() >= w->max_queue_depth) {
            w->queue_drops += 1;
            return;
        }
        w->queue.push_back({off, len, mmap_addr});
    }
    w->cv.notify_one();
}

// Per-projection byte range for one layer's expert tensor in the mmap'd GGUF.
//   base_addr_voidp + e * per_expert_bytes is the start of expert e's slice.
struct expert_proj_range {
    char *  base_addr        = nullptr;   // mmap pointer + base_offset
    size_t  per_expert_bytes = 0;
    size_t  total_bytes      = 0;
};
struct expert_layer_ranges {
    expert_proj_range gate;
    expert_proj_range up;
    expert_proj_range down;
};

struct bench_ctx {
    moe_predictor_weights weights;
    int32_t prefetch_k = 12;
    int32_t prefetch_horizon = 1;   // # layers ahead to predict+prefetch via chained inference
    int32_t cache_size = 0;   // experts/layer in sim cache; 0 = no sim
    bool    cache_size_is_dynamic = false;

    // ── madvise prefetch state ────────────────────────────────────────
    // If gguf_mmap_addr is non-null, we'll posix_madvise(MADV_WILLNEED) the
    // predicted experts' byte ranges. macOS / Linux page cache is shared
    // across mmaps of the same file, so warming our mmap warms llama.cpp's
    // mmap of the same file too. NO COPY happens — just kernel readahead.
    char *  gguf_mmap_addr   = nullptr;
    size_t  gguf_mmap_size   = 0;
    std::vector<expert_layer_ranges> expert_ranges;   // indexed by absolute layer id
    int64_t madvise_calls    = 0;
    int64_t madvise_bytes    = 0;

    // ── mlock tracking ──
    // For each absolute layer, the list of (addr, len) we've mlocked for its
    // predicted experts. After that layer fires (compute is done), we munlock
    // them to free the per-process lock budget for the next layer.
    std::vector<std::vector<std::pair<char *, size_t>>> locked_ranges;
    int64_t mlock_calls    = 0;
    int64_t mlock_failures = 0;
    int64_t mlock_bytes    = 0;
    int64_t munlock_calls  = 0;

    // ── Diagnostic counters (for mincore residency sampling) ──
    int64_t mincore_total_pages    = 0;
    int64_t mincore_resident_pages = 0;
    int64_t mincore_samples        = 0;

    // pread-based async prefetch (alternative to MADV_WILLNEED on macOS).
    // When set, callbacks enqueue tasks into worker instead of issuing
    // posix_madvise(WILLNEED).
    prefetch_worker * pf_worker = nullptr;

    // ── Dynamic memory-budgeted cache sizing ──────────────────────────
    // If memory_budget_mb > 0, cache_size is recomputed at each decode step:
    //   available_for_experts_mb = memory_budget_mb - kv_bytes_so_far / 1024^2
    //   cache_size = max(min_cache_size, available / per_expert_mb / n_layers)
    // This models the real edge constraint: total RAM budget for cache +
    // KV must stay under what's available after the model. As decode
    // proceeds, KV grows and the expert cache shrinks accordingly.
    double  memory_budget_mb     = 0.0;   // 0 = use static cache_size
    double  per_expert_mb        = 0.0;   // size of one expert at deploy quant
    double  kv_bytes_per_token   = 0.0;   // computed from model dims at init
    int32_t min_cache_size       = 1;     // floor; never evict below this

    // Runtime stats on dynamic cache size
    int32_t observed_min_cache_size = INT32_MAX;
    int32_t observed_max_cache_size = 0;
    int64_t cache_size_sum_samples  = 0;
    int64_t cache_size_n_samples    = 0;

    // Per-decode state — reset between tokens (sized [n_layers] after init)
    // Indexed by ABSOLUTE layer id, so we allocate [last_layer+1] slots.
    std::vector<std::vector<int32_t>> curr_token_layer_experts;     // [L][n_expert_used]
    std::vector<bool>                  curr_token_layer_seen;       // [L]
    std::vector<std::vector<int32_t>> last_prediction_for_layer;    // [L][prefetch_k]
    std::vector<int32_t>               last_prediction_k;           // [L]

    // Cross-decode state — carries from prev token
    std::vector<std::vector<int32_t>> prev_token_layer_experts;     // [L][n_expert_used]
    std::vector<bool>                  prev_token_layer_seen;       // [L]
    std::vector<float>                 prev_token_hidden;
    bool     prev_token_hidden_valid = false;

    // Scratch: result_norm grabs the last-token hidden state here
    std::vector<float> staging_hidden;

    // ── Per-layer LRU cache SIMULATOR (no actual memory backing) ──────
    //   - Tracks recency order of experts kept "warm" for this layer.
    //   - When predictor predicts top-k for next layer, insert those experts
    //     (mark as MRU, evict oldest if over capacity).
    //   - When model fires an expert at this layer, check membership: hit if
    //     present (then touch to MRU); miss if not (insert, evict if needed).
    //   - Reports the operational metric the paper needs: fraction of routed
    //     expert reads that would be served from cache rather than a cold load.
    //   - Front of deque = MRU, back = LRU.
    std::vector<std::vector<int32_t>> sim_cache;   // [L][≤cache_size] expert IDs in recency order
    int64_t sim_cache_hits   = 0;
    int64_t sim_cache_misses = 0;
    std::vector<int64_t> per_layer_sim_hits;
    std::vector<int64_t> per_layer_sim_total;

    // Aggregated stats
    int64_t total_predictions = 0;
    int64_t total_hit_count   = 0;
    int64_t total_actual_n    = 0;
    std::vector<int64_t> per_layer_pred;   // [last_layer+1]
    std::vector<int64_t> per_layer_hits;   // [last_layer+1]
    int64_t madvise_free_calls = 0;
    int64_t madvise_free_bytes = 0;
    bool    enable_madv_free   = true;   // false = disable MADV_FREE on eviction

    void init_from_weights() {
        const int slots = weights.last_layer() + 1;   // 0-indexed absolute
        curr_token_layer_experts.assign(slots, std::vector<int32_t>(weights.n_expert_used, -1));
        curr_token_layer_seen.assign(slots, false);
        last_prediction_for_layer.assign(slots, std::vector<int32_t>(prefetch_k, -1));
        last_prediction_k.assign(slots, 0);
        prev_token_layer_experts.assign(slots, std::vector<int32_t>(weights.n_expert_used, -1));
        prev_token_layer_seen.assign(slots, false);
        per_layer_pred.assign(slots, 0);
        per_layer_hits.assign(slots, 0);
        sim_cache.assign(slots, {});
        per_layer_sim_hits.assign(slots, 0);
        per_layer_sim_total.assign(slots, 0);
        locked_ranges.assign(slots, {});
    }
};

// ────────────────────────────────────────────────────────────────────────
// Minimal JSON-fragment parser for the qwen3_expert_offsets.json file
// (avoids pulling in nlohmann/json as a dependency).
// Supports the exact shape extract_expert_offsets.py emits.
// ────────────────────────────────────────────────────────────────────────
static bool find_int_after_key(const std::string & s, size_t from, const char * key,
                                 int64_t & out) {
    std::string k = std::string("\"") + key + "\"";
    size_t pos = s.find(k, from);
    if (pos == std::string::npos) return false;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n')) pos++;
    char * end = nullptr;
    out = std::strtoll(s.c_str() + pos, &end, 10);
    if (end == s.c_str() + pos) return false;
    return true;
}

static bool load_expert_offsets_json(const std::string & path,
                                      std::vector<expert_layer_ranges> & ranges_by_layer,
                                      std::string & out_gguf_path,
                                      int & out_num_experts) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[madvise] cannot open offsets json: %s\n", path.c_str());
        return false;
    }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    if (fread(&buf[0], 1, sz, f) != sz) { fclose(f); return false; }
    fclose(f);

    // gguf_path: "gguf_path": "<value>"
    size_t k = buf.find("\"gguf_path\"");
    if (k != std::string::npos) {
        size_t colon = buf.find(':', k);
        if (colon != std::string::npos) {
            size_t open  = buf.find('"', colon + 1);
            if (open != std::string::npos) {
                size_t close = buf.find('"', open + 1);
                if (close != std::string::npos) {
                    out_gguf_path = buf.substr(open + 1, close - open - 1);
                }
            }
        }
    }
    int64_t nexp = 128;
    find_int_after_key(buf, 0, "num_experts", nexp);
    out_num_experts = (int)nexp;

    int64_t nlayers = 48;
    find_int_after_key(buf, 0, "num_layers", nlayers);

    ranges_by_layer.assign((size_t)nlayers, expert_layer_ranges{});

    // Iterate per-layer entries. Each layer starts with "layer": N.
    size_t pos = buf.find("\"layers\"");
    if (pos == std::string::npos) return false;
    while (true) {
        size_t lkey = buf.find("\"layer\"", pos);
        if (lkey == std::string::npos) break;
        int64_t L;
        if (!find_int_after_key(buf, lkey, "layer", L)) break;

        auto parse_proj = [&](const char * proj_name, expert_proj_range & out_r) {
            std::string needle = std::string("\"") + proj_name + "\"";
            size_t pkey = buf.find(needle, lkey);
            if (pkey == std::string::npos) return;
            int64_t base_off = 0, per_e = 0, total = 0;
            find_int_after_key(buf, pkey, "base_offset", base_off);
            find_int_after_key(buf, pkey, "per_expert_bytes", per_e);
            find_int_after_key(buf, pkey, "total_bytes", total);
            out_r.base_addr        = (char *)(intptr_t)base_off;   // tag with offset; will rebase after mmap
            out_r.per_expert_bytes = (size_t)per_e;
            out_r.total_bytes      = (size_t)total;
        };

        if (L >= 0 && L < (int64_t)ranges_by_layer.size()) {
            parse_proj("gate", ranges_by_layer[L].gate);
            parse_proj("up",   ranges_by_layer[L].up);
            parse_proj("down", ranges_by_layer[L].down);
        }
        pos = lkey + 1;
    }
    return true;
}

// Check page residency using mincore(). Returns (resident_pages, total_pages).
// Called when an expert FIRES — tells us whether the pages our predictor told
// the kernel to bring in are actually in RAM, vs evicted before use.
static inline void mincore_check_expert(bench_ctx & b, int32_t layer, int32_t expert) {
    if (b.gguf_mmap_addr == nullptr) return;
    if (layer < 0 || layer >= (int)b.expert_ranges.size()) return;
    const auto & r = b.expert_ranges[layer];
    const long page_size = sysconf(_SC_PAGESIZE);
    auto check = [&](const expert_proj_range & p) {
        if (p.base_addr == nullptr || p.per_expert_bytes == 0) return;
        char * addr = p.base_addr + (size_t)expert * p.per_expert_bytes;
        // Round down to page boundary for mincore
        uintptr_t start = (uintptr_t)addr;
        uintptr_t aligned = start & ~(uintptr_t)(page_size - 1);
        size_t off = start - aligned;
        size_t len = p.per_expert_bytes + off;
        size_t npages = (len + page_size - 1) / page_size;
        if (npages == 0 || npages > 2048) return;  // sanity cap
        std::vector<char> vec(npages, 0);
        if (mincore((void *)aligned, npages * page_size, vec.data()) == 0) {
            int resident = 0;
            for (size_t i = 0; i < npages; ++i) if (vec[i] & 0x1) resident++;
            b.mincore_resident_pages += resident;
            b.mincore_total_pages    += npages;
            b.mincore_samples        += 1;
        }
    };
    check(r.gate);
    check(r.up);
    check(r.down);
}

// On expert eviction from our LRU cache, tell the kernel it can release those
// pages immediately. MADV_FREE is the macOS+Linux portable spelling that
// allows the kernel to reclaim the pages without (necessarily) writing them
// back — perfect for mmap'd read-only model weights.
static inline void evict_expert_pages(bench_ctx & b, int32_t layer, int32_t expert) {
    if (!b.enable_madv_free) return;     // explicitly disabled via CLI
    if (b.gguf_mmap_addr == nullptr) return;
    if (layer < 0 || layer >= (int)b.expert_ranges.size()) return;
    const auto & r = b.expert_ranges[layer];
    auto free_proj = [&](const expert_proj_range & p) {
        if (p.base_addr == nullptr || p.per_expert_bytes == 0) return;
        void * addr = p.base_addr + (size_t)expert * p.per_expert_bytes;
#ifdef MADV_FREE
        if (madvise(addr, p.per_expert_bytes, MADV_FREE) == 0) {
            b.madvise_free_calls += 1;
            b.madvise_free_bytes += p.per_expert_bytes;
        }
#else
        if (posix_madvise(addr, p.per_expert_bytes, POSIX_MADV_DONTNEED) == 0) {
            b.madvise_free_calls += 1;
            b.madvise_free_bytes += p.per_expert_bytes;
        }
#endif
    };
    free_proj(r.gate);
    free_proj(r.up);
    free_proj(r.down);
}

// LRU helpers: touch moves expert to front (MRU); insert with cap evicts the back (LRU).
static inline void sim_cache_touch(std::vector<int32_t> & cache, int32_t expert) {
    auto it = std::find(cache.begin(), cache.end(), expert);
    if (it != cache.end()) cache.erase(it);
    cache.insert(cache.begin(), expert);
}

// When eviction happens, callback is invoked for each evicted expert id.
template <class EvictFn>
static inline void sim_cache_warm_with_evict(std::vector<int32_t> & cache,
                                             int32_t expert, int32_t cap,
                                             EvictFn && on_evict) {
    if (cap <= 0) return;
    sim_cache_touch(cache, expert);
    while ((int32_t)cache.size() > cap) {
        int32_t evicted = cache.back();
        cache.pop_back();
        on_evict(evicted);
    }
}

// Returns true if expert was in cache (hit); on miss it loads (which may evict).
template <class EvictFn>
static inline bool sim_cache_check_and_load_with_evict(std::vector<int32_t> & cache,
                                                       int32_t expert, int32_t cap,
                                                       EvictFn && on_evict) {
    if (cap <= 0) return false;
    auto it = std::find(cache.begin(), cache.end(), expert);
    const bool hit = it != cache.end();
    if (hit) cache.erase(it);
    cache.insert(cache.begin(), expert);
    while ((int32_t)cache.size() > cap) {
        int32_t evicted = cache.back();
        cache.pop_back();
        on_evict(evicted);
    }
    return hit;
}

// Backwards-compatible thin wrappers (no eviction action — kept for callers
// that don't have a real cache, e.g. cache_size==0 paths).
static inline void sim_cache_warm(std::vector<int32_t> & cache, int32_t expert,
                                  int32_t cap) {
    sim_cache_warm_with_evict(cache, expert, cap, [](int32_t){});
}

// Pre-fill sim_cache from a warm-from-history profile produced by
// scripts/npy_to_warm_profile.py. Each line:
//   <layer> <expert> <hits>
// Lines starting with '#' are comments. Per layer, the entries are emitted in
// hits-desc order, so we take the first min(per_layer_entries, cache_size).
// The MRU position holds the highest-hit expert, so the first LRU eviction
// touches the lowest-hit warm expert.
//
// Returns the total number of (layer, expert) slots actually pre-filled.
static size_t prefill_sim_cache_from_warm_profile(bench_ctx & bc,
                                                  const std::string & path) {
    if (path.empty()) return 0;
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("[warm-profile] cannot open %s\n", path.c_str());
        return 0;
    }
    if (bc.cache_size <= 0) {
        LOG_INF("[warm-profile] cache_size=0, skipping preload (sim cache is disabled)\n");
        return 0;
    }

    std::vector<std::vector<int32_t>> per_layer(bc.sim_cache.size());
    size_t parsed = 0, bad = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int layer = -1, expert = -1;
        int64_t hits = 0;
        if (!(iss >> layer >> expert >> hits) || layer < 0 || expert < 0) {
            ++bad;
            continue;
        }
        if (layer >= (int)per_layer.size()) continue;
        per_layer[layer].push_back(expert);
        ++parsed;
    }

    size_t preloaded = 0;
    for (size_t L = 0; L < per_layer.size(); ++L) {
        const auto & src = per_layer[L];
        const size_t take = std::min((size_t)bc.cache_size, src.size());
        bc.sim_cache[L].assign(src.begin(), src.begin() + take);
        preloaded += take;
    }
    LOG_INF("[warm-profile] preloaded %zu experts from %s (%zu lines parsed, %zu bad)\n",
            preloaded, path.c_str(), parsed, bad);
    return preloaded;
}
static inline bool sim_cache_check_and_load(std::vector<int32_t> & cache,
                                            int32_t expert, int32_t cap) {
    return sim_cache_check_and_load_with_evict(cache, expert, cap, [](int32_t){});
}

static void rotate_token_state(bench_ctx & b) {
    if (!b.staging_hidden.empty()) {
        b.prev_token_hidden = b.staging_hidden;
        b.prev_token_hidden_valid = true;
        b.staging_hidden.clear();
    }
    for (int L = b.weights.first_layer; L <= b.weights.last_layer(); ++L) {
        if (b.curr_token_layer_seen[L]) {
            b.prev_token_layer_experts[L] = b.curr_token_layer_experts[L];
            b.prev_token_layer_seen[L] = true;
        }
        b.curr_token_layer_seen[L] = false;
    }
}

// ───────────────────────────────────────────────────────────────────────
// Eval callback
// ───────────────────────────────────────────────────────────────────────

static bool bench_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return strcmp(t->name, "result_norm") == 0 ||
               strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    }
    auto * b = static_cast<bench_ctx *>(user_data);

    // Capture post-norm hidden state of the LAST token in this batch.
    if (strcmp(t->name, "result_norm") == 0) {
        const int64_t n_embd   = t->ne[0];
        const int64_t n_tokens = t->ne[1];
        if (n_embd != b->weights.hidden_dim) return true;
        b->staging_hidden.resize(b->weights.hidden_dim);
        const size_t last_col_offset = (n_tokens - 1) * t->nb[1];
        ggml_backend_tensor_get(t, b->staging_hidden.data(),
                                last_col_offset,
                                b->weights.hidden_dim * sizeof(float));
        return true;
    }

    // ffn_moe_topk-{L}: read actuals, score last prediction, predict next layer.
    int curr_layer = -1;
    if (sscanf(t->name + 13, "%d", &curr_layer) != 1) return true;
    if (curr_layer < b->weights.first_layer ||
        curr_layer > b->weights.last_layer()) return true;

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];
    if (n_expert_used != b->weights.n_expert_used) return true;
    if (n_tokens != 1) return true;   // decode only

    std::vector<int32_t> observed(n_expert_used);
    ggml_backend_tensor_get(t, observed.data(), 0, n_expert_used * sizeof(int32_t));

    b->curr_token_layer_experts[curr_layer] = observed;
    b->curr_token_layer_seen[curr_layer] = true;

    // Simulated cache + (optional) real eviction via MADV_FREE.
    // For each fired expert, check membership AND check residency BEFORE we
    // munlock. (If we munlocked first, the kernel might have already started
    // DIAGNOSTIC: residency check at OBSERVE time (compute is about to use
    // these experts). Done BEFORE munlock so we measure while pages are pinned.
    // Unconditional so we get the data even without sim cache enabled.
    for (int a = 0; a < (int)observed.size(); ++a) {
        const int32_t e = observed[a];
        if (e < 0 || e >= b->weights.num_experts) continue;
        mincore_check_expert(*b, curr_layer, e);
    }
    if (b->cache_size > 0) {
        auto on_evict_curr = [&](int32_t evicted_e) {
            evict_expert_pages(*b, curr_layer, evicted_e);
        };
        for (int a = 0; a < (int)observed.size(); ++a) {
            const int32_t e = observed[a];
            if (e < 0 || e >= b->weights.num_experts) continue;
            const bool hit = sim_cache_check_and_load_with_evict(
                b->sim_cache[curr_layer], e, b->cache_size, on_evict_curr);
            b->per_layer_sim_total[curr_layer]++;
            if (hit) {
                b->sim_cache_hits++;
                b->per_layer_sim_hits[curr_layer]++;
            } else {
                b->sim_cache_misses++;
            }
        }
    }

    // ── munlock predicted experts for THIS layer (AFTER mincore measured) ──
    // Compute has now used the predictions we mlock'd. Free the lock budget
    // for the NEXT layer's predictions.
    if (b->pf_worker != nullptr && b->pf_worker->mode == 2 &&
        curr_layer < (int)b->locked_ranges.size()) {
        for (const auto & range : b->locked_ranges[curr_layer]) {
            if (munlock(range.first, range.second) == 0) {
                b->munlock_calls += 1;
            }
        }
        b->locked_ranges[curr_layer].clear();
    }

    // Score the prediction WE made the last time we predicted curr_layer.
    if (b->last_prediction_k[curr_layer] > 0) {
        int hits = 0;
        for (int p = 0; p < b->last_prediction_k[curr_layer]; ++p) {
            for (int a = 0; a < b->weights.n_expert_used; ++a) {
                if (b->last_prediction_for_layer[curr_layer][p] == observed[a]) {
                    ++hits;
                    break;
                }
            }
        }
        b->total_predictions++;
        b->total_hit_count   += hits;
        b->total_actual_n    += b->weights.n_expert_used;
        b->per_layer_pred[curr_layer]++;
        b->per_layer_hits[curr_layer] += hits;
    }

    // ── Corrective cache update at observe-time ──
    // Now that we know the actual experts that fired at curr_layer, we can:
    //   (a) evict experts we PREDICTED but were WRONG — they wasted SSD bandwidth
    //       on prefetch and are pure cache pollution. MADV_FREE their pages so
    //       the OS can reclaim them for upcoming prefetches.
    //   (b) record actuals as resident in our sim_cache — they were just read
    //       by the MoE op via mmap, so they're in the OS page cache. Tracking
    //       them lets future prefetches for the same experts skip redundant work
    //       (lookup in cache, return early).
    // Only do (a) / (b) when cache_size > 0 (sim cache is being used).
    if (b->cache_size > 0 && b->last_prediction_k[curr_layer] > 0) {
        // Build a fast lookup of actuals.
        bool is_actual[1024] = {false};   // small ceiling for num_experts
        for (int a = 0; a < b->weights.n_expert_used; ++a) {
            int32_t e = observed[a];
            if (e >= 0 && e < (int32_t)(sizeof(is_actual)/sizeof(is_actual[0]))) {
                is_actual[e] = true;
            }
        }

        // (a) Wrong predictions: MADV_FREE + remove from sim_cache.
        auto & cache = b->sim_cache[curr_layer];
        for (int p = 0; p < b->last_prediction_k[curr_layer]; ++p) {
            int32_t e = b->last_prediction_for_layer[curr_layer][p];
            if (e < 0 || e >= b->weights.num_experts) continue;
            if (!is_actual[e]) {
                // Wrong prediction — free its pages.
                evict_expert_pages(*b, curr_layer, e);
                auto it = std::find(cache.begin(), cache.end(), e);
                if (it != cache.end()) cache.erase(it);
            }
        }

        // (b) Missed actuals: add to sim_cache (they're already in OS page cache).
        auto on_evict_curr = [&](int32_t evicted_e) {
            evict_expert_pages(*b, curr_layer, evicted_e);
        };
        for (int a = 0; a < b->weights.n_expert_used; ++a) {
            int32_t e = observed[a];
            if (e < 0 || e >= b->weights.num_experts) continue;
            sim_cache_warm_with_evict(cache, e, b->cache_size, on_evict_curr);
        }
    }

    // Multi-horizon prediction: predict and prefetch for layers curr_layer + 1
    // .. curr_layer + prefetch_horizon. h=1 uses curr_layer's actual experts as
    // feat_B. h>1 chains: feat_B = top n_expert_used of the prediction we just
    // made at h-1 (so deeper-horizon predictions feed on each other).
    if (curr_layer + 1 > b->weights.last_layer()) {
        rotate_token_state(*b);
        return true;
    }

    // --prefetch-k 0: skip prediction + prefetch entirely. Useful for a
    // baseline-with-active-cache run (observe + LRU evict + MADV_FREE), no
    // predictor noise. The cache_size > 0 branch above populates sim_cache
    // with observed actuals; eviction triggers MADV_FREE.
    if (b->prefetch_k <= 0) {
        return true;
    }

    const int32_t * prev_tok_last = b->prev_token_layer_seen[b->weights.last_layer()]
                                     ? b->prev_token_layer_experts[b->weights.last_layer()].data()
                                     : nullptr;
    const float   * prev_hid = b->prev_token_hidden_valid
                                ? b->prev_token_hidden.data() : nullptr;

    // Per-iteration feat_B. h=1 starts from observed actuals.
    std::vector<int32_t> chain_feat_B(b->weights.n_expert_used, -1);
    const int32_t * curr_tok_Lm1 = observed.data();

    int32_t k = b->prefetch_k;
    if (k > b->weights.num_experts) k = b->weights.num_experts;

    for (int h = 1; h <= b->prefetch_horizon; ++h) {
        int target_layer = curr_layer + h;
        if (target_layer > b->weights.last_layer()) break;

        const int32_t * prev_tok_L  = b->prev_token_layer_seen[target_layer]
                                       ? b->prev_token_layer_experts[target_layer].data() : nullptr;

        std::vector<float> features(b->weights.feature_dim);
        moe_predictor_build_features(&b->weights,
                                      prev_tok_L, curr_tok_Lm1, prev_tok_last, prev_hid,
                                      target_layer, features.data());

        moe_predictor_predict(&b->weights, target_layer, features.data(), k,
                               b->last_prediction_for_layer[target_layer].data());
        b->last_prediction_k[target_layer] = k;

        // ── REAL prefetch: enqueue predicted experts' byte ranges ──
        if (b->gguf_mmap_addr != nullptr &&
            target_layer < (int)b->expert_ranges.size()) {
            const auto & ranges = b->expert_ranges[target_layer];
            const auto & cache  = b->sim_cache[target_layer];   // pre-warm snapshot
            if (ranges.gate.base_addr != nullptr) {
                for (int p = 0; p < k; ++p) {
                    const int32_t e = b->last_prediction_for_layer[target_layer][p];
                    if (e < 0 || e >= b->weights.num_experts) continue;
                    // Skip if already in cache (presumed page-resident already).
                    if (std::find(cache.begin(), cache.end(), e) != cache.end()) {
                        continue;
                    }
                    int hint_layer = target_layer;
                    auto hint = [&](const expert_proj_range & r) {
                        if (r.base_addr == nullptr || r.per_expert_bytes == 0) return;
                        if (b->pf_worker != nullptr) {
                            char * mmap_addr = r.base_addr + (size_t)e * r.per_expert_bytes;
                            off_t file_off = (off_t)(mmap_addr - b->gguf_mmap_addr);
                            prefetch_enqueue(b->pf_worker, file_off, r.per_expert_bytes, mmap_addr);
                            b->madvise_calls += 1;
                            b->madvise_bytes += r.per_expert_bytes;
                            if (b->pf_worker->mode == 2) {
                                b->locked_ranges[hint_layer].push_back({mmap_addr, r.per_expert_bytes});
                                b->mlock_bytes += r.per_expert_bytes;
                            }
                        } else {
                            void * addr = r.base_addr + (size_t)e * r.per_expert_bytes;
                            if (posix_madvise(addr, r.per_expert_bytes, POSIX_MADV_WILLNEED) == 0) {
                                b->madvise_calls += 1;
                                b->madvise_bytes += r.per_expert_bytes;
                            }
                        }
                    };
                    hint(ranges.gate);
                    hint(ranges.up);
                    hint(ranges.down);
                }
            }
        }

        // Sim cache warm for target_layer's predictions.
        if (b->cache_size > 0) {
            int evict_layer = target_layer;
            auto on_evict = [&](int32_t evicted_e) {
                evict_expert_pages(*b, evict_layer, evicted_e);
            };
            for (int p = 0; p < k; ++p) {
                const int32_t e = b->last_prediction_for_layer[target_layer][p];
                if (e < 0 || e >= b->weights.num_experts) continue;
                sim_cache_warm_with_evict(b->sim_cache[target_layer], e, b->cache_size,
                                           on_evict);
            }
        }

        // Prepare feat_B for next iteration: take top n_expert_used predictions.
        int chain_k = std::min<int>(b->weights.n_expert_used, k);
        for (int i = 0; i < chain_k; ++i) {
            chain_feat_B[i] = b->last_prediction_for_layer[target_layer][i];
        }
        for (int i = chain_k; i < b->weights.n_expert_used; ++i) {
            chain_feat_B[i] = -1;
        }
        curr_tok_Lm1 = chain_feat_B.data();
    }

    return true;
}

// ───────────────────────────────────────────────────────────────────────
// main
// ───────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    // Custom flags
    std::string predictor_weights_path = "moe_predictor_weights.bin";
    std::string warm_profile_path;          // path to flat text from npy_to_warm_profile.py; empty = no warm-up
    int32_t prefetch_k = 12;
    int32_t prefetch_horizon = 1;
    int32_t cache_size = 0;
    double  memory_budget_mb       = 0.0;   // 0 = static cache_size
    double  per_expert_mb          = 0.0;
    int32_t min_cache_size         = 1;
    int32_t cache_resize_interval  = 1;
    double  bytes_per_token_kv     = 0.0;
    std::string expert_offsets_json;        // path to qwen3_expert_offsets.json
    bool use_pread_prefetch = false;        // use background pread() instead of madvise
    bool use_touch_prefetch = false;        // background mmap-touch (active list)
    bool use_mlock_prefetch = false;        // background mlock (page-in + pin)
    bool no_madv_free       = false;        // disable MADV_FREE on cache eviction
    bool no_callback = false;  // baseline mode: skip the predictor entirely
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--predictor-weights") == 0 && i + 1 < argc) {
            predictor_weights_path = argv[i + 1];
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--prefetch-k") == 0 && i + 1 < argc) {
            prefetch_k = std::atoi(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--prefetch-horizon") == 0 && i + 1 < argc) {
            // Number of layers to look ahead and prefetch in a chained manner.
            // 1 = predict L+1 only (default). 3 = predict L+1, L+2, L+3 with
            // L+h's prediction feeding feat_B for predicting L+(h+1). Gives the
            // background worker more lead time at the cost of lower recall on
            // the deeper-chain predictions.
            prefetch_horizon = std::atoi(argv[i + 1]);
            if (prefetch_horizon < 1) prefetch_horizon = 1;
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--cache-size") == 0 && i + 1 < argc) {
            // experts/layer in simulated LRU cache. 0 = disable simulation.
            cache_size = std::atoi(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--warm-profile") == 0 && i + 1 < argc) {
            // Flat-text warm-from-history profile produced by
            // scripts/npy_to_warm_profile.py. Each line: "<layer> <expert> <hits>".
            // After init_from_weights(), sim_cache[L] is pre-populated with the
            // top min(K_in_file, cache_size) experts per layer, in hits-desc order,
            // before any decode runs. Stats from that point reflect first-token
            // benefit on top of the warmed state.
            warm_profile_path = argv[i + 1];
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--memory-budget-mb") == 0 && i + 1 < argc) {
            // Enable dynamic cache sizing. Total MB usable for (expert cache
            // + KV cache) — model itself is assumed mmap'd outside this budget.
            // As decode proceeds, KV grows and the per-layer cache shrinks.
            // Recommended on 16 GB Mac: 3000-5000 MB. Requires --per-expert-mb.
            memory_budget_mb = std::atof(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--per-expert-mb") == 0 && i + 1 < argc) {
            per_expert_mb = std::atof(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--min-cache-size") == 0 && i + 1 < argc) {
            min_cache_size = std::atoi(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--bytes-per-token-kv") == 0 && i + 1 < argc) {
            // KV bytes per token (model-specific):
            //   bytes/tok = n_layers × n_head_kv × head_dim × 2 (K+V) × kv_dtype_bytes
            // For Qwen3-30B-A3B fp16 KV: 98304
            // For DeepSeek-V2-Lite MLA fp16: ~26624
            bytes_per_token_kv = std::atof(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--no-madv-free") == 0) {
            // Skip MADV_FREE on cache eviction. Otherwise we mark pages as
            // "kernel may discard" — but those pages might be needed in 2-3
            // tokens when predictor cycles back, contradicting kernel's
            // natural LRU. (Diagnostic showed 27% residency at fire time
            // when MADV_FREE was enabled.)
            no_madv_free = true;
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            argc -= 1; --i;
        } else if (strcmp(argv[i], "--prefetch-mlock") == 0) {
            // Background worker mlock()s predicted experts: forces page-in AND
            // pins them so kernel can't evict before compute touches them.
            // Main thread munlocks them after the layer fires (compute done).
            // Per-layer locked footprint: 12 experts × 3 projections × 1.6 MB
            // = ~59 MB, well under macOS default mlock limit.
            use_mlock_prefetch = true;
            use_pread_prefetch = true;   // shares worker infra
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            argc -= 1; --i;
        } else if (strcmp(argv[i], "--prefetch-touch") == 0) {
            // Background worker thread touches mmap pages (one byte per page)
            // to force page faults that put pages on the ACTIVE list (vs
            // pread's inactive-list placement). Better resistance to eviction.
            use_touch_prefetch = true;
            use_pread_prefetch = true;  // implies pread mode is on (shares worker)
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            argc -= 1; --i;
        } else if (strcmp(argv[i], "--prefetch-pread") == 0) {
            // Use background-thread pread() to populate page cache instead of
            // posix_madvise(MADV_WILLNEED). On macOS, madvise(WILLNEED) appears
            // to be synchronous and hurts decode (measured); pread() in a
            // worker thread is genuinely async.
            use_pread_prefetch = true;
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            argc -= 1; --i;
        } else if (strcmp(argv[i], "--expert-offsets") == 0 && i + 1 < argc) {
            // Path to qwen3_expert_offsets.json (output of extract_expert_offsets.py).
            // Enables actual posix_madvise(WILLNEED) prefetch of predicted expert
            // pages in the mmap'd GGUF file. The bench will open the same file in
            // its own mmap, share page cache with llama.cpp's mmap, and hint pages.
            expert_offsets_json = argv[i + 1];
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--cache-resize-interval") == 0 && i + 1 < argc) {
            // How often (in decoded tokens) to recompute cache_size from the
            // memory budget. 1 = every token (default). 8-16 reduces variance.
            // 0 = compute once at start (static, ignores KV growth).
            // Not used unless --memory-budget-mb is set.
            // (Stored as a local for the decode loop below.)
            cache_resize_interval = std::atoi(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
            argc -= 2; --i;
        } else if (strcmp(argv[i], "--no-callback") == 0) {
            no_callback = true;
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            argc -= 1; --i;
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    bench_ctx bc;
    if (!no_callback) {
        if (!moe_predictor_load(predictor_weights_path.c_str(), &bc.weights)) {
            LOG_ERR("failed to load predictor weights: %s\n", predictor_weights_path.c_str());
            return 1;
        }
        bc.prefetch_k = prefetch_k;
        bc.prefetch_horizon = prefetch_horizon;
        bc.cache_size = cache_size;
        bc.memory_budget_mb   = memory_budget_mb;
        bc.per_expert_mb      = per_expert_mb;
        bc.min_cache_size     = min_cache_size;
        bc.kv_bytes_per_token = bytes_per_token_kv;
        bc.enable_madv_free   = !no_madv_free;
        if (memory_budget_mb > 0.0) {
            if (per_expert_mb <= 0.0) {
                LOG_ERR("--memory-budget-mb requires --per-expert-mb (size of one expert)\n");
                return 1;
            }
            bc.cache_size_is_dynamic = true;
            // Initialize cache_size as if KV were empty (pre-decode).
            double initial = bc.memory_budget_mb / bc.per_expert_mb / (double)bc.weights.n_layers;
            bc.cache_size = std::max(bc.min_cache_size, (int32_t)initial);
            LOG_INF("dynamic cache sizing: budget=%.0f MB, per_expert=%.3f MB, "
                    "initial_cache_size=%d experts/layer\n",
                    memory_budget_mb, per_expert_mb, bc.cache_size);
        }
        bc.init_from_weights();
        LOG_INF("predictor: %d layers × %d experts × %d feat_dim, "
                "top_k_actual=%d, prefetch_k=%d, sim_cache_size=%d\n",
                bc.weights.n_layers, bc.weights.num_experts,
                bc.weights.feature_dim, bc.weights.n_expert_used,
                prefetch_k, cache_size);

        // ── Optional warm-from-history preload ─────────────────────────
        // Must run after init_from_weights() (sim_cache is sized then) and
        // before any decode starts, so the first prompt's misses get the
        // benefit of the pre-filled cache.
        prefill_sim_cache_from_warm_profile(bc, warm_profile_path);

        // ── Optional madvise prefetch setup ────────────────────────────
        if (!expert_offsets_json.empty()) {
            std::string gguf_path;
            int json_num_experts = 0;
            if (!load_expert_offsets_json(expert_offsets_json,
                                          bc.expert_ranges, gguf_path, json_num_experts)) {
                LOG_ERR("failed to load expert offsets json: %s\n",
                        expert_offsets_json.c_str());
                return 1;
            }
            if (json_num_experts != bc.weights.num_experts) {
                LOG_ERR("num_experts mismatch: json=%d, predictor=%d\n",
                        json_num_experts, bc.weights.num_experts);
                return 1;
            }
            int fd = open(gguf_path.c_str(), O_RDONLY);
            if (fd < 0) {
                LOG_ERR("madvise: cannot open GGUF %s\n", gguf_path.c_str());
                return 1;
            }
            struct stat st;
            if (fstat(fd, &st) != 0) { close(fd); LOG_ERR("fstat\n"); return 1; }
            void * addr = mmap(nullptr, (size_t)st.st_size, PROT_READ,
                                MAP_SHARED, fd, 0);
            if (addr == MAP_FAILED) {
                close(fd);
                LOG_ERR("mmap of GGUF failed for madvise prefetch\n");
                return 1;
            }
            close(fd);
            bc.gguf_mmap_addr = (char *)addr;
            bc.gguf_mmap_size = (size_t)st.st_size;
            // Rebase per-layer ranges: replace stored file-offset with actual virtual addr.
            for (auto & r : bc.expert_ranges) {
                if (r.gate.per_expert_bytes > 0)
                    r.gate.base_addr = bc.gguf_mmap_addr + (intptr_t)r.gate.base_addr;
                if (r.up.per_expert_bytes > 0)
                    r.up.base_addr = bc.gguf_mmap_addr + (intptr_t)r.up.base_addr;
                if (r.down.per_expert_bytes > 0)
                    r.down.base_addr = bc.gguf_mmap_addr + (intptr_t)r.down.base_addr;
            }
            // (Previously we set POSIX_MADV_RANDOM on the whole mmap, but this
            //  prevented helpful sequential readahead for hot non-MoE tensors
            //  like attention weights. We now let the kernel use its default
            //  policy and only intervene via per-expert WILLNEED/FREE hints.)
            LOG_INF("madvise prefetch enabled: GGUF mmap'd at %p (%.2f GB), "
                    "%d layers indexed\n",
                    addr, st.st_size / (1024.0*1024.0*1024.0),
                    (int)bc.expert_ranges.size());

            // Optional: spawn background pread() worker.
            if (use_pread_prefetch) {
                int worker_fd = open(gguf_path.c_str(), O_RDONLY);
                if (worker_fd < 0) {
                    LOG_ERR("pread prefetch: cannot reopen GGUF for worker\n");
                    return 1;
                }
                bc.pf_worker = new prefetch_worker;
                bc.pf_worker->fd = worker_fd;
                // 4 MiB scratch — big enough for any single expert projection (1.6 MiB)
                bc.pf_worker->scratch.resize(4 * 1024 * 1024);
                bc.pf_worker->mode = use_mlock_prefetch ? 2 :
                                     use_touch_prefetch ? 1 : 0;
                bc.pf_worker->t = std::thread(prefetch_worker_loop, bc.pf_worker);
                const char * mode_name = bc.pf_worker->mode == 2 ? "mlock (pin)"
                                       : bc.pf_worker->mode == 1 ? "mmap-touch (active list)"
                                       : "pread";
                LOG_INF("prefetch worker enabled (mode=%s): scratch=%zu MiB, "
                        "max_queue=%d\n",
                        mode_name,
                        bc.pf_worker->scratch.size() / (1024*1024),
                        bc.pf_worker->max_queue_depth);
            }
        }

        params.cb_eval           = bench_cb;
        params.cb_eval_user_data = &bc;
    } else {
        LOG_INF("--no-callback: running BASELINE (no predictor, no callback)\n");
    }
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    log_memory_state("before_model_load");

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * ctx      = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load model\n");
        return 1;
    }

    log_memory_state("after_model_load");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    auto tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("empty prompt\n");
        return 1;
    }

    log_memory_state("before_prefill");

    // Prefill
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("prefill failed\n");
        log_memory_state("prefill_FAILED");
        return 1;
    }

    log_memory_state("after_prefill");

    // KV bytes/token must be passed via --bytes-per-token-kv. Compute formula:
    //   bytes/tok = n_layers × n_head_kv × head_dim × 2 (K+V) × kv_dtype_bytes
    // For Qwen3-30B-A3B fp16 KV: 48 × 4 × 128 × 2 × 2 = 98,304 ≈ 96 KB
    // For DeepSeek-V2-Lite MLA fp16: 26 × 512 × 2 ≈ 26 KB (latent compressed)
    if (bc.cache_size_is_dynamic && bc.kv_bytes_per_token > 0.0) {
        LOG_INF("kv_bytes_per_token = %.0f (user-provided). "
                "Prompt of %zu tokens already costs %.1f MB KV.\n",
                bc.kv_bytes_per_token, tokens.size(),
                (bc.kv_bytes_per_token * tokens.size()) / (1024.0*1024.0));
    } else if (bc.cache_size_is_dynamic) {
        LOG_WRN("--memory-budget-mb set but --bytes-per-token-kv missing; "
                "treating KV as 0 (cache_size will be static at initial value)\n");
    }

    // Decode n_predict tokens
    auto t0 = std::chrono::steady_clock::now();
    int n_predict = params.n_predict > 0 ? params.n_predict : 64;
    llama_token tok_id = tokens.back();
    const size_t prefill_tokens = tokens.size();
    for (int i = 0; i < n_predict; ++i) {
        // Dynamic cache resize: recompute every N decode steps based on
        // current sequence length (prompt + decoded tokens so far). When KV
        // grows, expert cache shrinks.
        if (bc.cache_size_is_dynamic && cache_resize_interval > 0
                && (i % cache_resize_interval) == 0) {
            const double kv_mb = (bc.kv_bytes_per_token
                                  * (double)(prefill_tokens + i))
                                  / (1024.0 * 1024.0);
            const double avail_mb = bc.memory_budget_mb - kv_mb;
            int new_cache_size = bc.min_cache_size;
            if (avail_mb > 0.0) {
                new_cache_size = std::max(
                    bc.min_cache_size,
                    (int32_t)(avail_mb / bc.per_expert_mb
                              / (double)bc.weights.n_layers));
            }
            bc.cache_size = new_cache_size;
            if (new_cache_size < bc.observed_min_cache_size)
                bc.observed_min_cache_size = new_cache_size;
            if (new_cache_size > bc.observed_max_cache_size)
                bc.observed_max_cache_size = new_cache_size;
            bc.cache_size_sum_samples += new_cache_size;
            bc.cache_size_n_samples   += 1;
        }

        // Greedy: pick argmax of logits
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        float best_v = -1e30;
        for (int v = 0; v < n_vocab; ++v) {
            if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        }
        tok_id = best;
        if (llama_vocab_is_eog(vocab, tok_id)) break;
        // Log memory state at step 0 (first decode) — Metal OOM happens here.
        // Also every 16 steps to track resident set growth.
        if (i == 0 || (i % 16) == 0) {
            char tag[64];
            snprintf(tag, sizeof(tag), "before_decode_step_%d", i);
            log_memory_state(tag);
        }
        if (llama_decode(ctx, llama_batch_get_one(&tok_id, 1))) {
            LOG_ERR("decode failed at step %d\n", i);
            log_memory_state("decode_FAILED");
            break;
        }
    }
    log_memory_state("after_decode_loop");
    auto t1 = std::chrono::steady_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int decoded = n_predict;
    LOG_INF("decode: %d tokens in %.1f ms (%.2f tok/s)\n",
            decoded, decode_ms, decoded * 1000.0 / decode_ms);

    // ── Report ─────────────────────────────────────────────────────────
    if (!no_callback) {
        printf("\n=== predictor accuracy on %lld real decode events ===\n",
               (long long)bc.total_predictions);
        if (bc.total_actual_n > 0) {
            double overall_recall = (double)bc.total_hit_count / (double)bc.total_actual_n;
            printf("micro_recall@%d  : %.2f%%\n", prefetch_k, 100.0 * overall_recall);
        }
        printf("\n  layer | predictions | recall@%d\n", prefetch_k);
        for (int L = bc.weights.first_layer; L <= bc.weights.last_layer(); ++L) {
            if (bc.per_layer_pred[L] == 0) continue;
            double r = (double)bc.per_layer_hits[L]
                     / (double)(bc.per_layer_pred[L] * bc.weights.n_expert_used);
            printf("  L%-3d  | %11lld | %.2f%%\n",
                   L, (long long)bc.per_layer_pred[L], 100.0 * r);
        }

        // ── Cache simulator report ─────────────────────────────────────
        if (bc.cache_size > 0 || bc.cache_size_is_dynamic) {
            const int64_t total = bc.sim_cache_hits + bc.sim_cache_misses;
            if (bc.cache_size_is_dynamic) {
                const double avg = bc.cache_size_n_samples > 0
                    ? (double)bc.cache_size_sum_samples / bc.cache_size_n_samples
                    : (double)bc.cache_size;
                printf("\n=== simulated LRU cache (DYNAMIC, budget=%.0f MB, "
                       "per_expert=%.3f MB, warmed via predicted top-%d) ===\n",
                       bc.memory_budget_mb, bc.per_expert_mb, prefetch_k);
                printf("  cache_size/layer: min=%d max=%d avg=%.1f over %lld samples\n",
                       bc.observed_min_cache_size, bc.observed_max_cache_size,
                       avg, (long long)bc.cache_size_n_samples);
            } else {
                printf("\n=== simulated LRU cache (cap=%d experts/layer, "
                       "warmed via predicted top-%d) ===\n",
                       bc.cache_size, prefetch_k);
            }
            if (total > 0) {
                printf("  total expert reads : %lld\n", (long long)total);
                printf("  cache hits         : %lld  (%.2f%% hit rate)\n",
                       (long long)bc.sim_cache_hits,
                       100.0 * bc.sim_cache_hits / (double)total);
                printf("  cache misses       : %lld  (%.2f%% miss rate)\n",
                       (long long)bc.sim_cache_misses,
                       100.0 * bc.sim_cache_misses / (double)total);
                printf("  expert bytes saved : %lld × per-expert  (each "
                       "hit = no cold load)\n",
                       (long long)bc.sim_cache_hits);
            }
            printf("\n  layer | reads | hit rate\n");
            for (int L = bc.weights.first_layer; L <= bc.weights.last_layer(); ++L) {
                if (bc.per_layer_sim_total[L] == 0) continue;
                double r = (double)bc.per_layer_sim_hits[L]
                         / (double)bc.per_layer_sim_total[L];
                printf("  L%-3d  | %5lld | %.2f%%\n",
                       L, (long long)bc.per_layer_sim_total[L], 100.0 * r);
            }
        }
    } else {
        printf("\n=== BASELINE: no callback, predictor disabled ===\n");
    }

    // ── madvise stats ───────────────────────────────────────────────
    if (!no_callback && bc.gguf_mmap_addr != nullptr) {
        printf("\n=== prefetch stats ===\n");
        if (bc.pf_worker != nullptr) {
            const char * mode_name = bc.pf_worker->mode == 2 ? "mlock (pin)"
                                   : bc.pf_worker->mode == 1 ? "mmap-touch"
                                   : "pread";
            printf("  mode             : %s\n", mode_name);
            printf("  worker ops       : %lld\n", (long long)bc.pf_worker->pread_calls);
            printf("  worker bytes     : %.2f GB\n",
                   bc.pf_worker->pread_bytes / (1024.0*1024.0*1024.0));
            printf("  queue drops      : %lld\n", (long long)bc.pf_worker->queue_drops);
            if (bc.pf_worker->mode == 2) {
                printf("  mlock bytes      : %.2f GB total (cycled per layer)\n",
                       bc.mlock_bytes / (1024.0*1024.0*1024.0));
                printf("  munlock calls    : %lld\n", (long long)bc.munlock_calls);
            }
        } else {
            printf("  mode             : posix_madvise(WILLNEED)\n");
            printf("  WILLNEED calls   : %lld\n", (long long)bc.madvise_calls);
            printf("  WILLNEED bytes   : %.2f GB\n",
                   bc.madvise_bytes / (1024.0*1024.0*1024.0));
        }
        printf("  FREE (eviction) calls : %lld\n", (long long)bc.madvise_free_calls);
        printf("  FREE bytes            : %.2f GB\n",
               bc.madvise_free_bytes / (1024.0*1024.0*1024.0));
        // Diagnostic: page residency at compute time.
        // If our prefetch was working, residency should be ~100%.
        // If pages were evicted before use, residency drops.
        if (bc.mincore_samples > 0) {
            printf("  ── mincore (residency at fire time) ──\n");
            printf("  samples           : %lld experts fired\n",
                   (long long)bc.mincore_samples);
            printf("  resident pages    : %lld / %lld total = %.2f%% resident\n",
                   (long long)bc.mincore_resident_pages,
                   (long long)bc.mincore_total_pages,
                   100.0 * bc.mincore_resident_pages / (double)bc.mincore_total_pages);
        }
    }

    // Shut down background prefetch worker if active.
    if (bc.pf_worker != nullptr) {
        bc.pf_worker->shutdown.store(true);
        bc.pf_worker->cv.notify_all();
        if (bc.pf_worker->t.joinable()) bc.pf_worker->t.join();
        close(bc.pf_worker->fd);
        delete bc.pf_worker;
        bc.pf_worker = nullptr;
    }

    if (bc.gguf_mmap_addr != nullptr) {
        munmap(bc.gguf_mmap_addr, bc.gguf_mmap_size);
    }

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
