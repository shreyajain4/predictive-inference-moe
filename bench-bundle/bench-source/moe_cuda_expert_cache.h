// moe_cuda_expert_cache.h
// ------------------------
// Persistent per-expert CUDA cache for MoE inference. Built specifically
// for the 3070 (gen4 PCIe) experiment.
//
// Pool: one cudaMalloc'd region (e.g. 6 GB) on the GPU, divided into N
// slots of expert_size bytes each. LRU eviction.
// Index: hash map (layer_id, expert_id) → slot.
//
// Producer side (bench's eval callback):
//   - Predictor predicts top-K experts for layer L+1
//   - For each predicted expert e: moe_cuda_expert_cache_prefetch(L+1, e, src_host, nbytes)
//     - Looks up cache. If hit, no-op. If miss, picks an LRU slot, evicts,
//       cudaMemcpyAsync(H→D) on the prefetch stream.
//
// Consumer side (ggml-backend.cpp copy_experts):
//   - For each run [first_id..last_id] at layer L, calls
//     moe_cuda_expert_cache_try_d2d(tensor_name, first_id, n, dst_ptr, dst_off, total_bytes)
//   - tensor_name encodes layer (e.g. "blk.5.ffn_gate_exps.weight" → L=5)
//     and projection (gate/up/down).
//   - If all n experts in run are cached AND the kind matches, copy D→D
//     into dst_ptr + dst_off and return 1. Else return 0 (PCIe fall-through).
//
// Stream model: dedicated prefetch stream (non-blocking) for cache fills.
// The d2d copy uses the DEFAULT cuda stream (stream 0) + cudaStreamSynchronize
// so the data is visible globally before ggml's compute stream reads it.
// (Suboptimal — future optimization: route d2d on the same compute stream.)

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct moe_cuda_expert_cache;

// Cache layout: hold per-expert data for each of the 3 projection KINDS
// (gate, up, down). Pass expert_size_bytes_per_kind[3] at create time —
// these are the per-expert byte sizes for gate, up, and down respectively
// (q4_k_m differs per projection because output dims differ).
//
// Total pool size = n_slots × max(expert_size_bytes_per_kind).
//
// Returns nullptr on cudaMalloc failure or invalid args.
moe_cuda_expert_cache * moe_cuda_expert_cache_create(
    int     n_layers,
    int     n_experts_per_layer,
    size_t  expert_size_gate,
    size_t  expert_size_up,
    size_t  expert_size_down,
    int     n_slots);     // total slots in the pool; e.g. 2400 for 6 GB / 2.5 MB

void moe_cuda_expert_cache_destroy(moe_cuda_expert_cache * c);

// LRU "touch" — if (layer, expert, kind) is in the cache, promote to
// most-recently-used so it's less likely to be evicted. No-op on miss.
int moe_cuda_expert_cache_touch(
    moe_cuda_expert_cache * c,
    int layer_id, int expert_id, int kind);

// H→D prefetch from CPU mmap into a pool slot. If already cached, just
// touches LRU and returns 1 (dedup). On miss, allocates a slot (evicting
// LRU if needed) and issues cudaMemcpyAsync on the prefetch stream.
// Returns 1 on success (hit or queued copy), 0 if size/kind invalid.
int moe_cuda_expert_cache_prefetch(
    moe_cuda_expert_cache * c,
    int          layer_id,
    int          expert_id,
    int          kind,           // 0=gate, 1=up, 2=down
    const void * host_src,       // CPU mmap pointer to first byte
    size_t       nbytes);        // must equal expert_size[kind]

// Wait for all in-flight cache operations to finish (typically called before bench exits).
void moe_cuda_expert_cache_drain(moe_cuda_expert_cache * c);

// Hook function used by ggml-backend.cpp's copy_experts.
// Tensor name format: "blk.<L>.ffn_<gate|up|down>_exps.weight"
// On hit, performs D→D from cache slot(s) to (input_cpy_data + dst_offset)
// using default stream + synchronize, then returns 1. On miss, returns 0.
int moe_cuda_expert_cache_try_d2d(
    const char * tensor_name,
    int32_t first_expert_id, int32_t n_experts_in_run,
    void *  input_cpy_data, size_t dst_offset,
    size_t  expert_size,
    size_t  total_bytes);

// Hook called by ggml-backend.cpp AFTER a successful PCIe write. Captures
// the post-conversion bytes into a cache slot (D→D from input_cpy to pool).
// Allocates a new slot via LRU eviction if necessary.
void moe_cuda_expert_cache_snapshot(
    const char * tensor_name,
    int32_t first_expert_id, int32_t n_experts_in_run,
    void *  input_cpy_data,  size_t dst_offset,
    size_t  expert_size,     size_t total_bytes);

// Install both hooks (try_d2d on miss-lookup, snapshot on PCIe-completion).
void moe_cuda_expert_cache_install_hook(moe_cuda_expert_cache * c);

// ── Stats ────────────────────────────────────────────────────────────────
typedef struct {
    int64_t prefetches;           // total prefetch calls
    int64_t prefetches_dedup;     // prefetches that hit an already-cached entry (no copy)
    int64_t d2d_hits;             // copy_experts runs satisfied entirely by cache
    int64_t d2d_partial_miss;     // copy_experts runs where some experts missed (fall-through)
    int64_t evictions;            // slot reuses via LRU
    int64_t bytes_prefetched;     // total H→D copied via prefetch
    int64_t bytes_d2d_served;     // total D→D bytes served on cache hits
} moe_cuda_expert_cache_stats;

void moe_cuda_expert_cache_get_stats(const moe_cuda_expert_cache * c, moe_cuda_expert_cache_stats * out);
void moe_cuda_expert_cache_print_stats(const moe_cuda_expert_cache * c, const char * tag);

#ifdef __cplusplus
}
#endif
