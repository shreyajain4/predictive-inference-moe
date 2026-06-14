// moe_predictor.h
// ---------------
// LR-hidden predictor for MoE expert prefetching in llama.cpp.
// Per-layer Linear(feature_dim → num_experts), BCE-trained on
// prev-token-hidden-state + prev-expert features. Dimensions are now
// runtime-configurable; previously hardcoded for DeepSeek-V2-Lite, now
// extensible to Qwen3-30B-A3B etc.
//
// Companion training/eval: experiments/lr_hidden_train.py, experiments/lr_hidden_eval.py.
// Binary weights format v2: see scripts/convert_predictor_pt_to_bin.py.
//   header: magic (u32) + version (u32) + n_layers (u32) + feature_dim (u32)
//           + num_experts (u32) + hidden_dim (u32) + first_layer (u32) + n_expert_used (u32)
//   floats: per layer L_idx ∈ [0..n_layers):
//             W[L_idx]: (num_experts, feature_dim) row-major
//             b[L_idx]: (num_experts,)
//           total per layer = num_experts * (feature_dim + 1) floats

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

constexpr uint32_t MOE_PREDICTOR_MAGIC   = 0x4D4F4550;   // "MOEP" little-endian
constexpr uint32_t MOE_PREDICTOR_VERSION = 2;            // v2 = runtime dims

struct moe_predictor_weights {
    int32_t n_layers       = 0;
    int32_t feature_dim    = 0;
    int32_t num_experts    = 0;
    int32_t hidden_dim     = 0;
    int32_t first_layer    = 0;   // absolute layer index of L_idx=0
    int32_t n_expert_used  = 0;   // top-k actually fired by the gate (not predict-k)
    // Flat float array. For layer L_idx ∈ [0..n_layers):
    //   W[L_idx] = &data[L_idx * per_layer]                           shape (num_experts, feature_dim) row-major
    //   b[L_idx] = &data[L_idx * per_layer + num_experts*feature_dim] shape (num_experts,)
    // where per_layer = num_experts * (feature_dim + 1).
    std::vector<float> data;

    inline int32_t last_layer() const { return first_layer + n_layers - 1; }
    inline size_t  per_layer_floats() const {
        return (size_t)num_experts * (feature_dim + 1);
    }
};

// Load weights from the .bin produced by convert_predictor_pt_to_bin.py (v2).
// Returns true on success.
bool moe_predictor_load(const char * path, moe_predictor_weights * out);

// Build the feature vector for predicting experts at (token_i, target_layer L).
// Length of out_features must be at least w.feature_dim.
//
// Inputs:
//   w:                      loaded weights (used for dims; not modified).
//   prev_tok_L_experts:     length n_expert_used — multihot positions for feat_A. nullptr -> all zero.
//   curr_tok_Lm1_experts:   length n_expert_used — multihot positions for feat_B (used when L > first_layer).
//   prev_tok_last_experts:  length n_expert_used — multihot positions for feat_B fallback when L == first_layer.
//   prev_tok_hidden:        length w.hidden_dim — prev token's post-norm hidden state. nullptr -> all zero.
//   target_layer:           absolute layer (w.first_layer .. w.last_layer()).
//
// Output:
//   out_features: length w.feature_dim, layout
//     [feat_A(num_experts) | feat_B(num_experts) | prev_hidden(hidden_dim)].
//
// Mirrors experiments/lr_hidden_train.py::build_lr_examples_hidden() byte-for-byte.
void moe_predictor_build_features(
    const moe_predictor_weights * w,
    const int32_t * prev_tok_L_experts,
    const int32_t * curr_tok_Lm1_experts,
    const int32_t * prev_tok_last_experts,
    const float   * prev_tok_hidden,
    int32_t target_layer,
    float * out_features
);

// Forward pass + top-k for a single (target_layer, features) prediction.
// Writes top_k expert IDs (sorted descending by logit) into out_top_experts.
// out_top_experts buffer must hold at least top_k int32_t.
void moe_predictor_predict(
    const moe_predictor_weights * w,
    int32_t target_layer,
    const float * features,
    int32_t top_k,
    int32_t * out_top_experts
);

#ifdef __cplusplus
}
#endif
