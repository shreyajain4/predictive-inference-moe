// moe_layer_stream.h
// -------------------
// Layered streaming MoE expert cache — a SLIDING WINDOW of layer-sized
// Metal buffers cycling through the inference layers.
//
// Concept (matches your "L getting processed → bring L+1 to Metal → evict L"
// model):
//
//   N circular Metal buffers, each sized to hold ALL 128 experts (gate+up+down)
//   of one layer ≈ 612 MB for Qwen3-30B-A3B Q8.
//
//   buf[L % N] holds layer L's expert weights when prefetched.
//   While layer L's MoE op fires, layer L+1 is being DMA'd into buf[(L+1)%N].
//   Layer L+2 starts filling buf[(L+2)%N] etc. Old layers' buffers get
//   transparently overwritten by their (L + N)-th successor.
//
// Memory: N × ~612 MB. N=3 → 1.8 GB Metal-wired. N=4 → 2.4 GB.
//   Well under the ~12 GB Metal working set we have on M1 Pro AND under the
//   ~3 GB safe ceiling we identified empirically (no auto-wired-region trigger
//   because we manage these buffers ourselves).
//
// This is distinct from moe_metal_expert_cache.h/.mm (per-expert LRU). That
// design caches individual experts with LRU eviction; this one caches whole
// layers with explicit round-robin slot reuse. Whole-layer is simpler if you
// want every expert of layer L+1 ready when its router fires (which is what
// you described).
//
// Bench integration model (observational at first):
//   - At callback for layer L's router fire: kick prefetch_layer(L+1)
//   - DMA happens in parallel with layer L's MoE compute on whatever backend
//   - When the actual MoE op fires for L+1, its experts are (hopefully) ready
//     in buf[(L+1) % N].
//
// API is C-callable so the bench (C++) can include + link the .mm file.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque impl handle (defined in .mm).
struct moe_layer_stream_impl;

// Create a layer streaming cache.
//   n_buffers:    sliding window depth (3-4 typical).
//   layer_bytes:  size of one layer's expert payload to keep resident. For
//                 Qwen3-30B-A3B Q8: 128 experts × 3 projs × 1.59 MB ≈ 612 MB.
//                 Pass a slightly-larger ceiling (e.g. 640 MB) for safety.
// Returns nullptr on Metal device unavailable / allocation failure.
moe_layer_stream_impl * moe_layer_stream_create(size_t n_buffers, size_t layer_bytes);

// Free resources. After this, all previously-returned buffer pointers are
// invalid.
void moe_layer_stream_destroy(moe_layer_stream_impl * c);

// Returns the GPU-visible pointer for layer L's buffer slot IF currently
// holding layer L's data. nullptr otherwise.
//
// On Apple Silicon's unified memory the returned pointer is also CPU-readable
// via the shared-storage MTLBuffer's contents pointer.
void * moe_layer_stream_get(moe_layer_stream_impl * c, int layer);

// Kick off DMA of `nbytes` from `src` into the slot reserved for `layer`.
// The slot is buf[layer % n_buffers] — any previous resident (an older layer)
// is implicitly overwritten.
//
// ASYNC: the actual memcpy runs on a background worker thread. This call
// enqueues the request and returns immediately. While the copy is in flight,
// moe_layer_stream_get(layer) returns nullptr (state=LOADING). When the
// worker finishes, the slot's current_layer flips to `layer` and get() returns
// the buffer pointer.
//
// Returns the slot's GPU-visible pointer (the same pointer regardless of
// whether the copy is in flight or done — caller should use _get() to test
// readiness).
void * moe_layer_stream_prefetch(
    moe_layer_stream_impl * c,
    int layer,
    const void * src,
    size_t nbytes);

// Block until any in-flight prefetch for `layer` completes (or returns
// immediately if not pending). Useful if compute is about to use the slot
// and we MUST have it ready. Returns the buffer ptr (or nullptr if the
// requested layer was never prefetched / got overwritten).
void * moe_layer_stream_wait(moe_layer_stream_impl * c, int layer);

// ── Multi-segment prefetch (for full-layer mode) ─────────────────────────
//
// A real MoE layer has 3 projection tensors (gate, up, down) at 3 separate
// file offsets in the GGUF. To stage the FULL layer (~610 MiB) we need to
// memcpy 3 contiguous ranges into the slot. This is one atomic prefetch
// request: the worker processes all segments before marking the slot READY.
//
// Total bytes copied = sum(segments[i].nbytes), must fit within layer_bytes.
struct moe_layer_segment {
    const void * src;          // mmap source address
    size_t       dst_offset;   // offset within the Metal slot
    size_t       nbytes;       // payload size for this segment
};

void * moe_layer_stream_prefetch_segments(
    moe_layer_stream_impl * c,
    int layer,
    const struct moe_layer_segment * segments,
    size_t n_segments);

// ── Stats ────────────────────────────────────────────────────────────────

typedef struct {
    int64_t prefetches;         // total prefetch calls
    int64_t lookups_hit;        // layer was resident when get() called
    int64_t lookups_miss;       // layer NOT resident when get() called
    int64_t lookups_in_flight;  // layer matched slot but copy still in progress (counted as miss)
    int64_t overwrites;         // prefetch displaced an older layer in the slot
    int64_t bytes_copied;       // total CPU→Metal bytes
    int64_t worker_busy_us;     // total microseconds the worker thread was actively copying
    int64_t wait_blocked_us;    // total microseconds get_wait blocked waiting for a copy
} moe_layer_stream_stats;

void moe_layer_stream_get_stats(const moe_layer_stream_impl * c, moe_layer_stream_stats * out);
void moe_layer_stream_reset_stats(moe_layer_stream_impl * c);
void moe_layer_stream_print_stats(const moe_layer_stream_impl * c, const char * tag);

#ifdef __cplusplus
}
#endif
