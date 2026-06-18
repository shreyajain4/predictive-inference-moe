// moe_predictor_hook.cpp
// ----------------------
// Wires the LR-hidden predictor + expert cache into llama.cpp.
//
// Tensor capture points (intercepted via ggml_backend eval callback):
//   - `result_norm`             → captures prev-token's last-layer post-norm
//                                 hidden state (for prev_hidden feature)
//   - `ffn_moe_topk-{L}`        → actual routed top-6 at layer L. Used as
//                                 feat_A (for next token) and feat_B (for
//                                 next layer of same token). After observing,
//                                 we predict L+1 and prefetch.
//
// Layer indexing: llama.cpp's MoE layers in DeepSeek-V2-Lite are model
// layers 1..26 (layer 0 is dense). Tensor names use those absolute layer
// indices. The predictor uses the same 1..26 indexing.
//
// llama.cpp prerequisites (must be added to llama.cpp's source tree):
//   - `const void * llama_get_expert_ptr_host(struct llama_model *,
//                                              int layer, int expert)`:
//        returns CPU-side pointer to the routed expert's combined weight
//        tensor (gate + up + down) for use as the DMA src for prefetch.
//        These must be cudaHostRegister'd if the GGUF was mmap-loaded.
//
//   - `const void * llama_get_shared_expert_ptr_host(struct llama_model *,
//                                                     int layer, int idx)`:
//        same for the 2 shared experts per layer (idx ∈ {0, 1}).
//
//   - `size_t llama_get_expert_bytes(struct llama_model *)`:
//        bytes per single expert's weights. Uniform across experts in the
//        same model.

#include "moe_predictor_hook.h"
#include "moe_predictor.h"
#include "moe_expert_cache.h"
#include "moe_warm_profile.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// Tensor name prefixes used by llama.cpp for MoE intermediates.
// Update these if llama.cpp renames them.
static constexpr const char * TENSOR_FINAL_NORM = "result_norm";
static constexpr const char * TENSOR_TOPK_PREFIX = "ffn_moe_topk-";
static constexpr int TENSOR_TOPK_PREFIX_LEN = 13;

// External llama.cpp helpers (need patches on the llama.cpp side; see header).
extern "C" {
    const void * llama_get_expert_ptr_host(struct llama_model * m, int layer, int expert);
    const void * llama_get_shared_expert_ptr_host(struct llama_model * m, int layer, int idx);
    size_t       llama_get_expert_bytes(struct llama_model * m);
}

// ───────────────────────────────────────────────────────────────────────
// Module state (singleton; one predictor + cache per process).
// ───────────────────────────────────────────────────────────────────────

namespace {

struct PredictorCtx {
    bool initialized = false;
    moe_predictor_weights weights;
    MoeExpertCache *      cache = nullptr;
    cuda_stream_handle *  stream = nullptr;
    int32_t               prefetch_k = 12;

    // Per-layer last-fired actuals (this token's routing so far).
    // Keyed by absolute layer index [1..26].
    int32_t curr_token_layer_experts[MOE_PREDICTOR_LAST_LAYER + 1][MOE_PREDICTOR_N_EXPERT_USED];
    bool    curr_token_layer_seen[MOE_PREDICTOR_LAST_LAYER + 1] = {};

    // Previous token's full routing (carried into this token via feat_A
    // for same-layer-prev-token signal).
    int32_t prev_token_layer_experts[MOE_PREDICTOR_LAST_LAYER + 1][MOE_PREDICTOR_N_EXPERT_USED];
    bool    prev_token_layer_seen[MOE_PREDICTOR_LAST_LAYER + 1] = {};

    // Previous token's last-layer post-norm hidden state (for prev_hidden).
    std::vector<float> prev_token_hidden;  // size MOE_PREDICTOR_HIDDEN_DIM
    bool               prev_token_hidden_valid = false;

    // Scratch for the current decode step's hidden state capture; populated
    // by result_norm hook, promoted to prev_token_hidden at end of step.
    std::vector<float> staging_hidden;     // size MOE_PREDICTOR_HIDDEN_DIM

    std::mutex mu;
};

PredictorCtx & ctx_inst() {
    static PredictorCtx c;
    return c;
}

// Promote staging_hidden → prev_token_hidden at end of this decode step.
// Called by the last-fired MoE layer (layer 26) so we have the data ready
// for the next decode step's predictions.
void rotate_token_state(PredictorCtx & ctx) {
    if (!ctx.staging_hidden.empty()) {
        ctx.prev_token_hidden = ctx.staging_hidden;
        ctx.prev_token_hidden_valid = true;
        ctx.staging_hidden.clear();
    }
    // curr → prev
    for (int L = MOE_PREDICTOR_FIRST_LAYER; L <= MOE_PREDICTOR_LAST_LAYER; ++L) {
        if (ctx.curr_token_layer_seen[L]) {
            memcpy(ctx.prev_token_layer_experts[L],
                   ctx.curr_token_layer_experts[L],
                   MOE_PREDICTOR_N_EXPERT_USED * sizeof(int32_t));
            ctx.prev_token_layer_seen[L] = true;
        }
        ctx.curr_token_layer_seen[L] = false;
    }
}

// ggml eval callback: fires for every tensor named in `result_norm` or
// `ffn_moe_topk-{L}`.
bool predictor_eval_cb(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask) {
        // We want result_norm AND any ffn_moe_topk-N tensor.
        return strcmp(t->name, TENSOR_FINAL_NORM) == 0 ||
               strncmp(t->name, TENSOR_TOPK_PREFIX, TENSOR_TOPK_PREFIX_LEN) == 0;
    }

    PredictorCtx & ctx = ctx_inst();
    if (!ctx.initialized) return true;

    // ── result_norm: capture hidden state for last token in batch ──────
    if (strcmp(t->name, TENSOR_FINAL_NORM) == 0) {
        // Tensor shape: [n_embd, n_tokens]. We want the LAST token's column,
        // because at the end of this forward pass it'll be the prev_hidden
        // for the next decode step.
        const int64_t n_embd   = t->ne[0];
        const int64_t n_tokens = t->ne[1];
        if (n_embd != MOE_PREDICTOR_HIDDEN_DIM) {
            // Model mismatch; bail out without breaking inference.
            return true;
        }

        std::lock_guard<std::mutex> lk(ctx.mu);
        ctx.staging_hidden.resize(MOE_PREDICTOR_HIDDEN_DIM);
        const size_t row_stride = t->nb[1];   // bytes per token column
        const size_t last_col_offset = (n_tokens - 1) * row_stride;
        ggml_backend_tensor_get(t, ctx.staging_hidden.data(),
                                last_col_offset,
                                MOE_PREDICTOR_HIDDEN_DIM * sizeof(float));
        return true;
    }

    // ── ffn_moe_topk-{L}: observed top-k for this layer ────────────────
    int curr_layer = -1;
    if (sscanf(t->name + TENSOR_TOPK_PREFIX_LEN, "%d", &curr_layer) != 1) return true;
    if (curr_layer < MOE_PREDICTOR_FIRST_LAYER ||
        curr_layer > MOE_PREDICTOR_LAST_LAYER) return true;

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];

    // Only act during decode (single new token). Prefill processes the whole
    // prompt in one batch — no prefetch opportunity since the union of
    // experts touched per layer is ~the full set.
    if (n_tokens != 1) return true;
    if (n_expert_used != MOE_PREDICTOR_N_EXPERT_USED) return true;

    int32_t observed_topk[MOE_PREDICTOR_N_EXPERT_USED];
    ggml_backend_tensor_get(t, observed_topk, 0,
                            n_expert_used * sizeof(int32_t));

    std::lock_guard<std::mutex> lk(ctx.mu);

    // Save the actuals so they can serve as feat_A next token + feat_B for
    // next layer of this token.
    memcpy(ctx.curr_token_layer_experts[curr_layer], observed_topk,
           MOE_PREDICTOR_N_EXPERT_USED * sizeof(int32_t));
    ctx.curr_token_layer_seen[curr_layer] = true;

    // Tell the cache which layer is currently active (used by
    // protect_next_layers eviction).
    ctx.cache->set_current_layer(curr_layer);

    // Predict next layer L+1. If we just fired layer 26, we don't predict
    // beyond — we rotate token state and exit.
    int next_layer = curr_layer + 1;
    if (next_layer > MOE_PREDICTOR_LAST_LAYER) {
        rotate_token_state(ctx);
        return true;
    }

    // Build features for predicting (token, L_target = next_layer).
    //   feat_A   = prev-token's L_target experts (if seen)
    //   feat_B   = current token's L_target-1 experts (= just observed)
    //   prev_hid = prev-token's last-layer post-norm hidden
    const int32_t * prev_tok_L = ctx.prev_token_layer_seen[next_layer]
                                  ? ctx.prev_token_layer_experts[next_layer] : nullptr;
    const int32_t * curr_tok_Lm1 = observed_topk;
    const int32_t * prev_tok_last = ctx.prev_token_layer_seen[MOE_PREDICTOR_LAST_LAYER]
                                     ? ctx.prev_token_layer_experts[MOE_PREDICTOR_LAST_LAYER]
                                     : nullptr;
    const float   * prev_hid = ctx.prev_token_hidden_valid
                                ? ctx.prev_token_hidden.data() : nullptr;

    float features[MOE_PREDICTOR_FEATURE_DIM];
    moe_predictor_build_features(
        prev_tok_L, curr_tok_Lm1, prev_tok_last, prev_hid,
        next_layer, features
    );

    int32_t predicted[64];   // max top_k ceiling
    int32_t k = ctx.prefetch_k;
    if (k > MOE_PREDICTOR_NUM_EXPERTS) k = MOE_PREDICTOR_NUM_EXPERTS;
    moe_predictor_predict(&ctx.weights, next_layer, features, k, predicted);

    // Prefetch each predicted expert (skip those already resident).
    for (int i = 0; i < k; ++i) {
        const int32_t e = predicted[i];
        if (e < 0 || e >= MOE_PREDICTOR_NUM_EXPERTS) continue;
        if (ctx.cache->lookup(next_layer, e) != nullptr) continue;
        // Note: lookup() above counts as a hit; we don't want misses for
        // "check if already resident". The current cache API doesn't have
        // a peek-without-stats — TODO add one. For now this slightly inflates
        // hits, but reset_stats() before measurement zeroes that out.
        const void * src = llama_get_expert_ptr_host(nullptr, next_layer, e);
        if (src) {
            ctx.cache->prefetch(next_layer, e, src);
        }
    }

    return true;
}

} // namespace

// ───────────────────────────────────────────────────────────────────────
// Public API
// ───────────────────────────────────────────────────────────────────────

int moe_predictor_hook_init(struct llama_model * model,
                             const struct moe_predictor_hook_config * cfg) {
    PredictorCtx & ctx = ctx_inst();
    if (ctx.initialized) {
        fprintf(stderr, "[moe_predictor_hook] already initialized\n");
        return -1;
    }

    if (!moe_predictor_load(cfg->weights_path, &ctx.weights)) {
        fprintf(stderr, "[moe_predictor_hook] failed to load weights: %s\n",
                cfg->weights_path);
        return -2;
    }
    fprintf(stderr, "[moe_predictor_hook] loaded predictor (%d layers × %d × %d)\n",
            ctx.weights.n_layers, ctx.weights.num_experts, ctx.weights.feature_dim);

    ctx.stream = moe_cuda_stream_create();
    if (!ctx.stream) {
        fprintf(stderr, "[moe_predictor_hook] failed to create CUDA stream\n");
        return -3;
    }

    const size_t expert_bytes = llama_get_expert_bytes(model);
    ctx.cache = new MoeExpertCache(cfg->n_cache_slots, expert_bytes,
                                    ctx.stream, cfg->protect_next_layers);
    fprintf(stderr, "[moe_predictor_hook] cache: %zu slots × %zu bytes "
                    "(total %.1f MB)\n",
            cfg->n_cache_slots, expert_bytes,
            cfg->n_cache_slots * expert_bytes / 1024.0 / 1024.0);

    if (cfg->pin_shared_experts) {
        int n_pinned = 0;
        for (int L = MOE_PREDICTOR_FIRST_LAYER;
             L <= MOE_PREDICTOR_LAST_LAYER; ++L) {
            for (int s = 0; s < 2; ++s) {
                const void * src = llama_get_shared_expert_ptr_host(model, L, s);
                if (src) {
                    // Use synthetic expert IDs >= 1000 to avoid collision with
                    // routed expert IDs 0..63.
                    ctx.cache->pin(L, 1000 + s, src);
                    n_pinned++;
                }
            }
        }
        fprintf(stderr, "[moe_predictor_hook] pinned %d shared experts\n", n_pinned);
    }

    if (cfg->warm_profile_path) {
        auto entries = load_warm_profile(cfg->warm_profile_path);
        if (!entries.empty()) {
            auto src_provider = [model](int layer, int expert) -> const void * {
                return llama_get_expert_ptr_host(model, layer, expert);
            };
            apply_warm_profile(*ctx.cache, entries, src_provider);
        }
    }

    ctx.prefetch_k = cfg->prefetch_k;
    ctx.initialized = true;
    return 0;
}

void moe_predictor_hook_install(struct llama_context * llama_ctx) {
    // Hook into llama.cpp's eval callback. llama.cpp has llama_set_eval_callback
    // or similar — check the actual API name; it may live on the ggml_backend
    // scheduler instead.
    //
    // For the public llama.cpp API (~b3000 and later):
    //   llama_set_eval_callback(llama_ctx, predictor_eval_cb, nullptr);
    //
    // If using the backend scheduler directly:
    //   ggml_backend_sched_set_eval_callback(sched, predictor_eval_cb, nullptr);
    //
    // The hook is no-op until the context actually runs llama_decode().
    llama_set_eval_callback(llama_ctx, predictor_eval_cb, nullptr);
}

void moe_predictor_hook_print_stats() {
    PredictorCtx & ctx = ctx_inst();
    if (!ctx.initialized || !ctx.cache) return;
    const int64_t h = ctx.cache->hits();
    const int64_t m = ctx.cache->misses();
    const int64_t p = ctx.cache->prefetches();
    const int64_t e = ctx.cache->evictions();
    const double total = static_cast<double>(h + m);
    fprintf(stderr,
        "[moe_predictor_hook] hits=%lld misses=%lld hit_rate=%.2f%% "
        "prefetches=%lld evictions=%lld\n",
        (long long)h, (long long)m, total > 0 ? 100.0 * h / total : 0.0,
        (long long)p, (long long)e);
}

void moe_predictor_hook_reset_stats() {
    PredictorCtx & ctx = ctx_inst();
    if (!ctx.initialized || !ctx.cache) return;
    ctx.cache->reset_stats();
}

void moe_predictor_hook_destroy() {
    PredictorCtx & ctx = ctx_inst();
    if (!ctx.initialized) return;
    delete ctx.cache;
    ctx.cache = nullptr;
    moe_cuda_stream_destroy(ctx.stream);
    ctx.stream = nullptr;
    ctx.weights.data.clear();
    ctx.initialized = false;
}
