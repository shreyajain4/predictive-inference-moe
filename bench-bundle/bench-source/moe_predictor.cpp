// moe_predictor.cpp
// -----------------
// Implementation of the LR-hidden predictor. See moe_predictor.h for API docs.
//
// NOTE: the predictor is purely about routed experts. Shared experts (always-
// active) are NOT predicted; they must be pinned in the expert cache at
// startup. See docs/llama_cpp_integration_plan.md "Shared experts" section.

#include "moe_predictor.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

inline const float * weights_for_layer_idx(const moe_predictor_weights * w, int L_idx) {
    return w->data.data() + (size_t)L_idx * w->per_layer_floats();
}

inline const float * bias_for_layer_idx(const moe_predictor_weights * w, int L_idx) {
    return weights_for_layer_idx(w, L_idx)
         + (size_t)w->num_experts * w->feature_dim;
}

} // namespace

bool moe_predictor_load(const char * path, moe_predictor_weights * out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[moe_predictor] failed to open %s\n", path);
        return false;
    }

    uint32_t header[8] = {0};
    if (fread(header, 4, 8, f) != 8) {
        fprintf(stderr, "[moe_predictor] header read failed in %s\n", path);
        fclose(f);
        return false;
    }
    uint32_t magic         = header[0];
    uint32_t version       = header[1];
    uint32_t n_layers      = header[2];
    uint32_t feature_dim   = header[3];
    uint32_t num_experts   = header[4];
    uint32_t hidden_dim    = header[5];
    uint32_t first_layer   = header[6];
    uint32_t n_expert_used = header[7];

    if (magic != MOE_PREDICTOR_MAGIC) {
        fprintf(stderr, "[moe_predictor] bad magic in %s: got 0x%X, expected 0x%X\n",
                path, magic, MOE_PREDICTOR_MAGIC);
        fclose(f);
        return false;
    }
    if (version != MOE_PREDICTOR_VERSION) {
        fprintf(stderr, "[moe_predictor] unsupported version %u (expected %u)\n",
                version, MOE_PREDICTOR_VERSION);
        fclose(f);
        return false;
    }
    if (feature_dim != 2 * num_experts + hidden_dim) {
        fprintf(stderr, "[moe_predictor] inconsistent header: "
                "feature_dim=%u, expected 2*num_experts+hidden_dim = 2*%u+%u = %u\n",
                feature_dim, num_experts, hidden_dim,
                2 * num_experts + hidden_dim);
        fclose(f);
        return false;
    }

    out->n_layers      = (int32_t)n_layers;
    out->feature_dim   = (int32_t)feature_dim;
    out->num_experts   = (int32_t)num_experts;
    out->hidden_dim    = (int32_t)hidden_dim;
    out->first_layer   = (int32_t)first_layer;
    out->n_expert_used = (int32_t)n_expert_used;

    const size_t n_floats = (size_t)n_layers * out->per_layer_floats();
    out->data.resize(n_floats);
    const size_t n_read = fread(out->data.data(), sizeof(float), n_floats, f);
    fclose(f);
    if (n_read != n_floats) {
        fprintf(stderr, "[moe_predictor] short read: got %zu floats, expected %zu\n",
                n_read, n_floats);
        return false;
    }
    fprintf(stderr, "[moe_predictor] loaded %s: %d layers, %d experts, "
            "feature_dim=%d (hidden=%d), top_k_actual=%d, first_L=%d\n",
            path, out->n_layers, out->num_experts, out->feature_dim,
            out->hidden_dim, out->n_expert_used, out->first_layer);
    return true;
}

void moe_predictor_build_features(
    const moe_predictor_weights * w,
    const int32_t * prev_tok_L_experts,
    const int32_t * curr_tok_Lm1_experts,
    const int32_t * prev_tok_last_experts,
    const float   * prev_tok_hidden,
    int32_t target_layer,
    float * out_features
) {
    const int32_t E = w->num_experts;
    const int32_t K = w->n_expert_used;
    const int32_t H = w->hidden_dim;
    const int32_t D = w->feature_dim;

    // Zero everything first.
    memset(out_features, 0, D * sizeof(float));

    // feat_A [0..E-1]: experts@(i-1, target_layer)
    if (prev_tok_L_experts != nullptr) {
        for (int k = 0; k < K; ++k) {
            const int32_t e = prev_tok_L_experts[k];
            if (e >= 0 && e < E) {
                out_features[e] = 1.0f;
            }
        }
    }

    // feat_B [E..2E-1]:
    //   target_layer > first_layer → experts@(i, target_layer - 1)
    //   target_layer == first_layer → experts@(i-1, last_layer)  (cross-token fallback)
    const int32_t * feat_B_src = (target_layer > w->first_layer)
        ? curr_tok_Lm1_experts
        : prev_tok_last_experts;
    if (feat_B_src != nullptr) {
        for (int k = 0; k < K; ++k) {
            const int32_t e = feat_B_src[k];
            if (e >= 0 && e < E) {
                out_features[E + e] = 1.0f;
            }
        }
    }

    // prev_hidden [2E..D-1]: hidden_state[i-1, last_layer, post-norm]
    if (prev_tok_hidden != nullptr) {
        memcpy(out_features + 2 * E,
               prev_tok_hidden,
               H * sizeof(float));
    }
}

void moe_predictor_predict(
    const moe_predictor_weights * w,
    int32_t target_layer,
    const float * features,
    int32_t top_k,
    int32_t * out_top_experts
) {
    if (target_layer < w->first_layer || target_layer > w->last_layer()) {
        for (int k = 0; k < top_k; ++k) out_top_experts[k] = -1;
        return;
    }
    const int32_t E = w->num_experts;
    if (top_k > E) top_k = E;

    const int   L_idx = target_layer - w->first_layer;
    const float * W   = weights_for_layer_idx(w, L_idx);
    const float * b   = bias_for_layer_idx(w, L_idx);

    // logits[e] = b[e] + sum_d W[e, d] * features[d]
    std::vector<float> logits(E);
    const int32_t D = w->feature_dim;
    for (int e = 0; e < E; ++e) {
        const float * Wrow = W + (size_t)e * D;
        float s = b[e];
        for (int d = 0; d < D; ++d) s += Wrow[d] * features[d];
        logits[e] = s;
    }

    // Partial top-k. E is small enough (≤128) that std::partial_sort is fine.
    std::vector<std::pair<float, int32_t>> ranked(E);
    for (int e = 0; e < E; ++e) {
        ranked[e] = std::make_pair(logits[e], e);
    }
    std::partial_sort(
        ranked.begin(), ranked.begin() + top_k, ranked.end(),
        [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
            return a.first > b.first;
        });
    for (int k = 0; k < top_k; ++k) out_top_experts[k] = ranked[k].second;
}
