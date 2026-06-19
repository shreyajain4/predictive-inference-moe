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

## Result — context sweep (single prompt, 70 chars)

Tests the hypothesis "ngl_auto's lead shrinks as KV grows and displaces experts on GPU."

| Config | c=4096 | c=8192 | c=16384 | c=32768 | c=65536 |
|---|---|---|---|---|---|
| **ngl_auto** | **26.27** | **25.86** | **24.94** | **23.00** | **20.30** |
| snap | 10.47 | 10.13 | 9.55 | OOM | OOM |
| snap_pred | 7.41 | 6.75 | 6.53 | OOM | OOM |
| snap_warm | 10.91 | 10.49 | 9.96 | OOM | OOM |
| snap_warm_pred | 7.67 | 6.97 | 6.72 | OOM | OOM |

Cache hit rates (snap variants only):

| Config | c=4096 | c=8192 | c=16384 |
|---|---|---|---|
| snap | 31.9% | 29.6% | 26.4% |
| snap_pred | 44.3% | 42.1% | 41.6% |
| snap_warm | **31.9%** | **29.6%** | **26.4%** |
| snap_warm_pred | **44.3%** | **42.1%** | **41.6%** |

**Hypothesis falsified.** ngl_auto loses only 23% throughput across c=4k→65k (26.27 → 20.30) and remains ~2× the snapshot regime at every tested context. KV growth on 3070+Qwen3-30B-A3B Q4 doesn't push auto-fit into a regime where forced-offload-with-cache can catch up.

**Predictor hurts by ~3 t/s** despite raising cache hit rate by ~12 pp. Extra H→D prefetches cost more than they save. Same pattern as the original snapshot bench's predictor-driven prefetch result (9.04 vs 12.58 t/s).

**Warm-from-history's hit rate is identical to snap-only at every context** (31.9 / 29.6 / 26.4 — digit-for-digit). Warm-loaded experts self-evict during warm-up before decode begins; snap then organically fills the cache with the same experts warm would have hit. The +0.5 t/s tok/s edge (snap_warm vs snap) is post-warm-up state perturbation, not cache serving.

Snap variants OOM at c=32768 and c=65536 with cache_mb=1500 because graph activation memory grows with `n_batch=ctx`. Dropping cache_mb to 600 MiB at c=32k lets snap run (~9.5 t/s), still well behind ngl_auto at 23. `offload_no_cache` (forced-offload regime *without* the VRAM pool) runs flat ~8.6 t/s across c=4k–32k; snap adds +1.84 t/s on top of that — so snapshot itself is doing real work in its regime, but the entire regime sits at half of ngl_auto's performance.

## Why expert-level hybrid within a layer doesn't help at batch=1

A tempting fallback for cache misses would be: route the *uncached* experts to CPU compute (no PCIe) while letting cached ones run on GPU. At batch=1 this doesn't speed anything up.

| Path | Time per layer |
|---|---|
| All GPU (cache hit) | ~0.05 ms |
| All CPU compute | ~0.30 ms (DRAM-bound) |
| Mixed 4 GPU + 4 CPU | max(0.05, 0.30) = **0.30 ms** |

Layer L+1's input is layer L's full output → can't start until ALL active experts at L finish for the current token. The slowest expert in the mix gates the layer transition. Mixed gives no speedup over pure CPU.

This is why ngl_auto's *layer-level* split works (each layer runs entirely on one side, transitions cost ~80 KB activation transfer between layers) while *expert-level* split within a layer doesn't. Expert-level hybrid only helps at batch ≫ 1 (different tokens take different paths in parallel), in speculative decoding, or in pipelined prefill.

## Regime classification — when this whole class of mechanism could win

The cache mechanism is a workaround for one specific bottleneck: per-token PCIe between CPU-resident experts and GPU compute. The mechanism is structurally relevant only when:

1. **PCIe ≥ DRAM bandwidth** (true on H100/PCIe5 + DDR5, false on RTX 3070 + DDR4/5), OR
2. **The model doesn't fit any productive partial offload** (ngl_auto degenerates, forced-offload is mandatory)

Regimes where snap/warm-from-history could win:
- Model ≫ VRAM (e.g. Mixtral 8x22B Q8 ~140 GB on 40 GB A100, DeepSeek-V3, …)
- Batch ≫ 1 serving (PCIe amortizes across many tokens reusing same expert)
- NVMe-backed inference (snapshot amortizes disk reads)
- TTFT (first-token latency) optimization in cold-start serving

Regimes where it can't:
- Mac (unified memory, no PCIe between CPU and GPU)
- Consumer GPU + model that fits via ngl_auto (this measurement)
- Single-stream batch=1 decode with PCIe slower than DRAM

The 40 GB A100 + Mixtral 8x22B Q4 case (~80 GB model) is borderline — ngl_auto fits ~half, snap+cache competes with the CPU half's traffic. Worth measuring once if we want a tighter answer; otherwise the negative result here generalizes.

## How does this square with MoE-Infinity and related work showing 4-7× speedups?

Natural question: if cache+prefetch is structurally dominated by `ngl_auto` here, how do MoE-Infinity (Xue et al. 2024), FluxMoE, DuoServe-MoE, OD-MoE etc. report large speedups? They're in different regimes AND compare against weaker baselines.

**1. Different hardware.** Server A100/H100 with PCIe gen4-5 (32-64 GB/s) and DDR5 (50+ GB/s). The per-expert math reverses — PCIe + GPU compute is faster than CPU compute. On consumer gen4 + DDR4/5 (RTX 3070), it's the opposite. OD-MoE uses 10-node distributed setups where the PCIe-to-host-CPU bottleneck doesn't apply at all.

**2. Different model regimes.** They target Switch-XL, GLaM, Mixtral 8x22B Q8 — models that don't fit in any productive partial offload. The auto-fit baseline degenerates because there's no productive `ngl`. Forced-offload+cache becomes the only meaningful comparison.

**3. Different baselines** (the key one). They compare against DeepSpeed-MII or static-partition (≈ our `offload_no_cache`) — both pay full PCIe per token with no cache. They don't compare against llama.cpp's `common_fit_params` auto-fit. **Our `snap` IS measurably better than `offload_no_cache` (+29%) — same shape of win as their reported numbers, just smaller magnitude.** We just additionally measured the auto-fit baseline that dominates both.

**4. Different metrics.** Many MoE serving papers report (a) prefill throughput, (b) batched serving latency at batch=8/16/32+, or (c) TTFT — not single-stream batch=1 decode tok/s. At higher batch, PCIe amortizes across many tokens reusing the same expert. Batch=1 decode is the most pessimistic metric for cache mechanisms.

**The honest takeaway.** This negative result doesn't refute MoE-Infinity — it identifies a regime boundary they didn't explore. The publishable framing:

> "MoE expert caching mechanisms (snap, predictor prefetch, warm-from-history) deliver real benefits in server-class regimes where partial offload doesn't fit (per MoE-Infinity, FluxMoE et al.). On consumer GPUs with adaptive partial-offload (llama.cpp's `common_fit_params`), the layer-level placement strategy structurally dominates at batch=1 decode. We characterize the regime boundary and quantify the ordering: ngl_auto > cpu_moe > snap+cache > offload_no_cache > raw forced-offload."

If you wanted to reproduce one of those papers' positive results in our infrastructure, the closest reachable regime would be: 40 GB A100 + Mixtral 8x22B Q8 (~140 GB, ngl_auto can fit only ~28% of layers) + batch ≥ 4. That's where the math flips back toward favoring snap+cache.

## Snapshot vs ahead-of-time prefetch (2026-06-19)

A separate diagnostic isolates the predictor's standalone contribution from snapshot via a new `--no-snapshot-fill` flag (commit `b1f373b`). Single-prompt smoke, c=4096:

| Config | t/s | hit_rate | bytes_prefetched | bytes_d2d_served |
|---|---|---|---|---|
| **A: snap only** | **10.48** | 31.9% | 8.0 GB (snap fills) | 4.1 GB |
| B: snap + predictor | 7.43 | 44.4% | 10.5 GB | 5.7 GB |
| C: **predictor only** (no snap) | 7.26 | 43.1% | 4.6 GB | 5.5 GB |
| D: nothing | 7.55 | 0.0% | 0 | 0 |

**Findings:**

1. **Snapshot alone is the only winner.** A is +39% over baseline D. Snapshot fills the cache with bytes that were crossing PCIe anyway → zero net PCIe overhead.
2. **Predictor is net-negative even ALONE.** C (predictor-only) at 7.26 t/s is WORSE than D (nothing) at 7.55 — so predictor's harm isn't from snapshot redundancy. Removing snapshot competition (B → C) barely changed the answer (-0.17 t/s). The predictor's H→D prefetches themselves are the cost: they compete with `copy_experts` for PCIe bandwidth, add `try_d2d` hook latency on every cache lookup, and incur stream-0 sync stalls on hits.
3. **The predictor hook is implemented correctly.** Recall is real (43% hit rate in C from purely predictor-driven fills, 60% dedup in B), the prefetches do land. The mechanism is just structurally inferior to snapshot on PCIe-bound consumer hardware.

**Generalized claim:**

> On PCIe-bound consumer GPUs, snapshot's "free fill" mechanism (post-PCIe capture) is structurally superior to ANY ahead-of-time prefetch mechanism (predictor OR warm-from-history) — even when prefetch has high recall, even when no other mechanism competes for the same cache slots. Snapshot pays zero net PCIe overhead while prefetch always adds H→D that has to overlap with compute (which it doesn't on gen4 PCIe + short layer compute time).

This collapses two previously-separate findings (predictor-driven prefetch hurts, warm-from-history barely helps) into one structural rule. They're in the same losing class: ahead-of-time H→D into cache. **Substitution** ([sensitivity_substitution_result](../../memory)) is in a different class — it doesn't add PCIe, it replaces a missed expert with an approximation. That's why substitution gets a positive result in this regime where prefetch can't.

**Where ahead-of-time prefetch could still win:**

- Compute time per layer >> per-expert PCIe time (workstation gen5, bigger experts) → prefetches actually overlap with compute
- No reactive PCIe to capture (batched serving where one PCIe serves many tokens, or models so big that snapshot churns too fast to retain anything)

Otherwise: invest in snapshot, not prefetch.

## Final: Mixtral 8x22B SSD-regime test (2026-06-19)

The last regime to test was "model genuinely doesn't fit in CPU+VRAM and forces SSD spillage." On Legion (45 GB RAM + 8 GB VRAM = 53 GB fast memory), Mixtral 8x22B IQ4_XS (~76 GB) is the smallest model that triggers this regime.

**Setup:**
- 30 MoE layers × 8 experts × top-2 routing
- Per-expert at IQ4_XS: 170 MB (cache slot 69 MB for max kind = down projection)
- Non-expert weights on GPU: 3.3 GB (fixed by model)
- Compute graph buffers: ~500 MB minimum even with `-c 512 -ub 128`
- Max cache budget on 8 GB 3070: ~3.5 GB → 50 slots → 20% expert coverage ceiling
- Achievable cache: 1500 MB → 22 slots → 9% coverage

**Result table:**

| Config | tok/s |
|---|---|
| ngl_auto (llama-cli, no cache, no override) | **0.40** |
| Forced offload no cache | 0.20 |
| Forced offload + snap+cache (cache_mb=1500) | **0.19** |

**Cache delivered zero d2d hits.** Per-token working set is 30 layers × 2 active × 3 kinds = 180 expert reads; cache has 22 slots. Working set is 8× larger than cache → every layer evicts the previous layer's snap fills before reuse. 8490 snapshot fills and 440 GB of D→D traffic, all wasted.

## Complete scoreboard across all 4 tested regimes

| Regime | ngl_auto | snap+cache | ngl_auto wins by |
|---|---|---|---|
| Qwen3-30B-A3B Q4 (fits RAM, 3070) | 26.70 | 11.06 | 2.4× |
| DeepSeek-V2-Lite Q8 (fits RAM, 3070) | 13.58 | 7.14 | 1.9× |
| Cgroup-pressured DS Q8 (simulated SSD) | ~13.6 | ~7.1 | ~1.9× |
| Mixtral 8x22B IQ4_XS (true SSD spill, 3070) | **0.40** | **0.19** | **2.1×** |

**Definitive conclusion:** on RTX 3070 with gen4 PCIe, snap+cache is structurally dominated by ngl_auto across every accessible regime, regardless of model size or memory pressure. The 2× gap is consistent because the root cause is the same — PCIe gen4 is slower than DRAM, so reactive CPU compute (ngl_auto) beats per-token PCIe transfer (forced offload + cache).

**Path to positive result:** workstation gen5 PCIe hardware (RTX PRO 6000 Blackwell, ~₹179/hr on Jarvis cloud) where PCIe is competitive with DRAM AND the cache pool is large enough to hold the per-token working set. Not reachable on this consumer hardware.

## CPU-side prewarm test (also negative)

Last mechanism tested: instead of GPU VRAM caching, pre-load expert pages into OS page cache via Python `pread()`. ngl_auto's CPU MoE compute then reads from RAM not SSD. No PCIe involved.

**Setup:** prewarm_experts.py pread()'s top-K=4 of 8 experts per layer × 30 MoE layers × 3 kinds = 22 GB into OS page cache. Bench in ngl_auto mode before vs after.

**Result:**

| Config | gen t/s | Cached after step |
|---|---|---|
| A: cold ngl_auto | **0.39** | 46.3 GB |
| (prewarm: read 19.7 GB) | — | **37.0 GB** ← dropped 9 GB |
| B: ngl_auto after prewarm | **0.35** | 45.8 GB |

**The prewarm hurt performance, not helped.** The OS evicted 9 GB of useful pages to make room for our prewarmed pages. After B ran, Cached returned to ~46 GB because the OS re-cached what was actually being used.

**Root cause:** without a real Mixtral routing profile, "first K experts per layer" is arbitrary selection. The OS page cache's natural LRU had already cached the experts that previous ngl_auto runs touched — which are the experts actually used. Our prewarm replaced correct cache contents with incorrect ones.

**Sharpest final claim across the project:**

*ngl_auto + Linux's reactive LRU page cache is structurally complete on consumer hardware. The OS already implements warm-from-history "for free" based on observed routing. To beat this, a predictor must anticipate future routing more accurately than recent past routing predicts. Without such a predictor, the entire MoE expert caching mechanism class (GPU-side OR CPU-side) adds no value over ngl_auto + the OS's LRU.*

The complete landscape across 5 tested mechanism classes on RTX 3070:

| Mechanism | Result |
|---|---|
| GPU snap (forced offload + cache) | LOSES (-2× across all regimes) |
| GPU snap + warm-from-history | LOSES (+0.5 t/s noise) |
| GPU predictor prefetch | LOSES (hurts even standalone, no snap competition) |
| GPU snap + warm + predictor | LOSES (worst combination) |
| **CPU prewarm into OS page cache** | **LOSES (HURTS by ~10%)** |

Every "ahead-of-time pre-position" mechanism is dominated by the OS's reactive LRU. The only positive mechanism in the project is sensitivity-aware substitution (per `sensitivity_substitution_result`) — it doesn't add or pre-position traffic, it *skips* traffic by accepting quality loss.

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
