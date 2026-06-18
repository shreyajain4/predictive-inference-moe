// moe_predictor_hook.h
// --------------------
// Public API for the LR-hidden predictor integration into llama.cpp.
//
// Lifecycle:
//   1. moe_predictor_hook_init(...) at llama startup, after model load.
//      - Loads predictor weights, allocates the expert cache, creates the
//        prefetch CUDA stream, pins shared experts.
//   2. moe_predictor_hook_install(ctx) wires the callback into the
//      llama_context. Subsequent llama_decode() calls invoke the hook.
//   3. moe_predictor_hook_destroy() at shutdown.
//
// The hook is a no-op during prefill (n_tokens != 1) — predictor's value
// is in the decode phase.

#pragma once

#include <cstddef>
#include <cstdint>

#include "ggml.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct moe_predictor_hook_config {
    const char * weights_path;     // path to moe_predictor_weights.bin
    size_t n_cache_slots;          // total VRAM slots (pinned + managed)
    int32_t prefetch_k;            // experts to prefetch per layer transition
    int32_t protect_next_layers;   // 0=plain LRU; 2 recommended
    bool    pin_shared_experts;    // true to pin all 52 shared experts at init

    // Optional warm-from-history. If non-null, the file is loaded after
    // shared-expert pinning and each (layer, expert) entry is pinned via the
    // same llama_get_expert_ptr_host() that routed experts use. Produced by
    // scripts/npy_to_warm_profile.py from a build_user_history_profile.py
    // count matrix. Pinned entries come out of the n_cache_slots budget.
    const char * warm_profile_path;
};

// Initialize the predictor + cache + stream. Returns 0 on success.
int moe_predictor_hook_init(
    struct llama_model * model,
    const struct moe_predictor_hook_config * cfg
);

// Install the eval callback on a context so the predictor fires during decode.
void moe_predictor_hook_install(struct llama_context * ctx);

// Print + reset accumulated stats (cache hits, misses, evictions, prefetches).
void moe_predictor_hook_print_stats();
void moe_predictor_hook_reset_stats();

// Tear down. Frees cache, stream, predictor weights.
void moe_predictor_hook_destroy();

#ifdef __cplusplus
}
#endif
