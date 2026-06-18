# Warm-from-history on RTX 3070 + Qwen3-30B-A3B Q4: clean negative result

**Date:** 2026-06-18
**Hardware:** Legion RTX 3070 mobile, 8 GB VRAM, gen4 PCIe ~13 GB/s
**Model:** Qwen3-30B-A3B-Instruct-2507-Q4_K_M (~17 GB, 48 layers, 128 experts × top-8)

## Setup

User-history-based pre-population of the snapshot expert cache. 100 prompts from the user's Claude Code transcripts, 80/20 train/test split.

- **Pipeline:** `llama-moe-trace` collects routing → `build_user_history_profile.py` → top-K experts/layer → `npy_to_warm_profile.py` → flat txt → `--warm-snapshot-profile` flag in `moe-predictor-bench` calls `moe_cuda_expert_cache_prefetch` per (layer, expert) × 3 kinds at startup.
- **Cache budget:** `--expert-cache-mb 3500` → 2844 slots × 1.29 MB.
- **Warm profile:** K=32 per layer × 48 layers = 1536 entries × 3 kinds = 4608 slot loads.
- **Forcing PCIe traffic:** `-ngl 99 --override-tensor exps=CPU GGML_OP_OFFLOAD_MIN_BATCH=1`. Without `MIN_BATCH=1` the MoE op stays on CPU and `copy_experts` never fires.

## Result — single-prompt smoke (32-token decode, n=1 prompt)

| Config | tok/s | d2d_hit_rate | bytes via cache |
|---|---|---|---|
| ngl=12 (partial offload, no cache) | **22.9** | n/a | 0 |
| CPU MoE (no forced offload, override CPU) | 17.2 | 0% | 0 |
| Snapshot + warm-from-history (K=32) | 14.21 | 61.5% | 22.8 GB |
| Snapshot alone | 13.81 | 61.5% | 22.8 GB |
| Forced offload, no cache (memory) | ~9.0 | — | 36.6 GB PCIe |

## Result — aggregate (29 paired prompts, 8-token decode each)

| Config | mean tok/s | median | min | max | hit rate |
|---|---|---|---|---|---|
| **ngl=auto** | **26.70** | 26.72 | 26.44 | 26.79 | n/a |
| ngl=12 (manual) | 21.31 | 21.38 | 19.37 | 21.56 | n/a |
| CPU MoE | 16.77 | 16.76 | 16.65 | 16.87 | 0% |
| Snapshot + warm-from-history (K=32) | 11.59 | 11.67 | 9.73 | 13.34 | 35.78% |
| Snapshot alone | 11.06 | 11.10 | 9.31 | 12.95 | 35.75% |

`ngl=auto` = pass no `-ngl` flag at all; llama.cpp's `common_fit_params` chooses `n_gpu_layers` to fit available VRAM. On 3070+Qwen3-30B-A3B Q4 with c=4096 it picks more than 12 layers and is +25% over manual ngl=12. Range is also much tighter (26.44–26.79 vs 19.37–21.56 for ngl=12). **This is the bar to beat, not ngl=12.**

**Warm-from-history adds +0.53 t/s (+4.8%) over snapshot alone, paired across 29 prompts.** Real but small effect. Hit-rate is nearly identical between snap and snap+warm (35.78 vs 35.75%) — the warm-loaded experts don't show up as d2d_hits, yet per-prompt tok/s is consistently higher. The warm-up's `drain()` finishes before the decode timer starts, so H→D overhead is amortized and the cache state at decode-start is slightly different (different evictions during warm-up → different cold-start configuration). Single-prompt smoke showed +0.4 t/s at the edge of noise; the paired 29-prompt result confirms the direction with much tighter variance.

**Snapshot is real per-prompt** (+4.8 t/s over raw forced-offload in single-prompt smoke). Aggregate hit rate is lower (35.75% vs 61.5%) because each fresh prompt routes to new experts — snapshot has to refill from scratch every iteration.

**But the entire snapshot regime loses on this hardware.** Mean snap+warm (11.59) < CPU MoE (16.77) < ngl=12 (21.31). Forcing experts to GPU and then caching them is *slower* than just leaving MoE on CPU. Partial offload (ngl=12) dominates by ~10 t/s.

## Root cause for the null warm result

Cache slot accounting. 1536 entries × 3 kinds = 4608 slot loads into a 2844-slot pool. The warm-up phase itself evicts ~1764 of its own entries. By the time the first decode token fires, snapshot then fills the cache organically with the experts that actually route — overwriting whatever warm entries survived.

The sim_cache prediction from `predict_warm_history_hit_rate.py` (58.3% hit rate upper bound at K=32) was **observational only** — it counted hits in a per-(layer, expert) array, not in a slot-budgeted physical cache. The real pool runs out of room.

## What would change the answer

1. **K=16 retry:** 768 × 3 = 2304 slots fits in 2844 with ~540-slot headroom for snapshot's organic fill. Untested but the math allows warm entries to survive to first decode.
2. **Regime pivot:** ngl=12 crushes this comparison because Qwen3-30B-A3B Q4 fits 12 layers in 8 GB cleanly. On a model/hardware combo where no `-ngl` value fits (e.g., Mixtral 8x22B Q4 ~80 GB on 24 GB), forced-offload is the only path and snapshot+warm vs snapshot becomes the real comparison.

## Honest framing for the talk / paper

Don't pitch warm-from-history vs snapshot on the 3070+Qwen3 combo. It's the losing comparison. Pitch it as: *"Sim_cache shows +21 pp first-decision hit rate from user-history pre-population. Wall-clock translation requires a regime where (a) cache budget fits the warm profile, and (b) forced offload is the only inference option. On 3070+Qwen3-30B-A3B Q4, partial offload wins both; on Mixtral 8x22B Q4 on 24 GB A100, forced offload is mandatory and the comparison becomes meaningful."*

Same shape as the [DS Q8 substitution result](../../memory/ds_q8_substitution_result.md): clean positive mechanism, clean negative wall-clock on the hardware where the mechanism isn't needed.

## Artifacts

- Bench source with `--warm-snapshot-profile`: `bench-bundle/bench-source/moe-predictor-bench.cpp` (commit 6a3d5e1)
- Profile generator: `moe-advisor/scripts/npy_to_warm_profile.py`
- Train profile: `moe-advisor/data/user_history/shreya_train_profile.{npy,json}` (commit 0318fda)
- Test prompts: `moe-advisor/data/user_history/shreya_prompts_test.tsv` (commit 2f7fd96)
- Sim_cache hit-rate predictor: `moe-advisor/scripts/predict_warm_history_hit_rate.py` (commit 6da95b5)
- Jaccard@K comparator (train vs test profile generalization): `moe-advisor/scripts/compare_user_history_profiles.py` (commit 5462753)
