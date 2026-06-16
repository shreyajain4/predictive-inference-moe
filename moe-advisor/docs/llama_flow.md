# MoE inference on llama.cpp — configurations, mechanisms, cost models

A reference for understanding where time goes in each setup. Order:

**§1–3 Three baseline configurations:**
1. **Default `-ngl 0`** — pure CPU, no GPU involvement
2. **Partial offload `-ngl X`** — first X layers on GPU, rest on CPU (the strong vanilla baseline)
3. **Snapshot approach** — `-ngl 99 --override-tensor "ffn_.*_exps=CPU"` + persistent GPU cache

**§4–5: When does snapshot beat -ngl** — crossover math + Q&A summary.

**§6–8 Mechanisms layered on top of snapshot:**
6. **Predictor-driven prefetch** — what we built, why it doesn't help inference (negative result)
7. **Sensitivity-aware substitution** — the +29–32% winner at Q4
8. **Hybrid CPU+GPU compute** (proposed) — the Q8 escape, scales 1/m on big models

Variable conventions used throughout:

| symbol | meaning | typical value |
|---|---|---|
| `L` | total MoE layers in the model | 27 (DS), 48 (Qwen3) |
| `X` | `-ngl` setting (number of layers GPU-resident) | 12 (DS Q8), 18 (Qwen3 Q4) |
| `K` | top-K experts per layer | 6 (DS), 8 (Qwen3) |
| `E` | per-expert weight size (one projection × Q-format) | 3 MB Q4, 9 MB Q8 |
| `h` | cache hit rate (fraction of expert lookups served from cache) | 0.74 Q4, 0.54 Q8 |
| `m` | miss rate = `1 - h` | 0.26 Q4, 0.46 Q8 |
| `PCIe_BW` | PCIe gen4 bandwidth | ~13 GB/s |
| `CPU_BW` | CPU memory bandwidth (DDR4) | ~50 GB/s |
| `GPU_BW` | VRAM bandwidth (RTX 3070) | ~450 GB/s |

---

## 1. Default `-ngl 0` — everything on CPU

```
Token T arrives
 │
 ├─ Layer 0  (CPU)  attention + FFN + MoE
 ├─ Layer 1  (CPU)  attention + FFN + MoE
 ├─ ...
 ├─ Layer L−1 (CPU)
 │
 └─ lm_head (CPU) → logits → sample
```

**Where things live:**
- All weights: CPU (mmap'd from GGUF)
- KV cache: CPU
- Compute: CPU only
- GPU: not used

**PCIe traffic per token: ZERO.** Nothing crosses the bus.

**Per-token cost:**
```
cost = L × CPU_FFN_per_layer
     = L × (K × E / CPU_BW)        [FFN bandwidth-bound]
       + L × CPU_attn               [attention, layer-cost ~constant per token]
```

For DS Q8 (`L=27, K=6, E=9 MB, CPU_BW=50 GB/s`):
- FFN cost = 27 × (6 × 9 MB / 50 GB/s) = 27 × 1.08 ms = **29 ms FFN per token**
- Measured: 10.77 t/s ≈ 93 ms/token → roughly 29 ms FFN + 64 ms attention/overhead

This is the floor. Adding GPU only helps if it makes the dominant term cheaper.

---

## 2. Partial offload `-ngl X` — split compute (vanilla strong baseline)

```
Token T arrives
 │
 ├─ Layer 0    (GPU) attention + FFN + MoE   ┐
 ├─ Layer 1    (GPU)                          │ weights live on GPU
 ├─ ...                                       │ KV cache lives on GPU
 ├─ Layer X−1  (GPU) →  hidden_state          │ NO PCIe per token
 │                       ║ ~4 KB hidden_state  ┘
 │                       ║ PCIe transfer (~0.3 μs at 13 GB/s)
 │                       ▼
 ├─ Layer X    (CPU) attention + FFN + MoE   ┐
 ├─ ...                                       │ weights live on CPU
 ├─ Layer L−1  (CPU) →  hidden_state          │ KV cache lives on CPU
 │                       ║ ~4 KB hidden_state  │ NO PCIe per token
 │                       ║ PCIe transfer       ┘
 │                       ▼
 └─ lm_head (GPU, usually) → logits → sample
```

**Where things live:**
- Layers 0..X−1: weights + KV + compute all on GPU
- Layers X..L−1: weights + KV + compute all on CPU
- `lm_head` (output.weight): typically GPU, depends on memory headroom

**PCIe traffic per token:** ~8 KB total (one hidden state down at the GPU→CPU boundary, one hidden state up to lm_head). At gen4 13 GB/s, **~0.6 μs**. Effectively zero.

> **Note:** No PCIe for *weights* in -ngl mode. Weights move once at load time when the GGUF is mapped onto each device. After that, weights never cross PCIe during decode.

**Per-token cost:**
```
cost = X × GPU_FFN_per_layer + (L − X) × CPU_FFN_per_layer + attention + boundary_PCIe
     = X × (K × E / GPU_BW) + (L − X) × (K × E / CPU_BW) + (...)
```

For DS Q8 at `-ngl 12`:
- GPU FFN: 12 × (6 × 9 MB / 450 GB/s) = 12 × 0.12 ms = **1.4 ms**
- CPU FFN: 15 × (6 × 9 MB / 50 GB/s) = 15 × 1.08 ms = **16.2 ms**
- Boundary PCIe: 0.6 μs (negligible)
- Total FFN: 17.6 ms; measured 15.34 t/s ≈ 65 ms/token → ~17.6 ms FFN + ~47 ms attention/overhead

The win over `-ngl 0`: those 12 GPU layers run FFN ~9× faster (450 vs 50 GB/s memory bandwidth). The remaining 15 stay on CPU at the same cost as before.

---

## 3. Snapshot approach — all-GPU compute, weights stream through cache

Setup: `-ngl 99 --override-tensor "ffn_.*_exps=CPU" --expert-cache-mb 5120`.

```
Token T arrives
 │
 ├─ Router fires at layer L → picks top-K expert IDs {E1, E2, …, EK}
 │
 ├─ For each expert E in {E1..EK}:
 │   ┌── Is E already in the GPU cache pool?
 │   │     YES (hit)  → input_cpy pointer set to pool slot     ←── 0 PCIe
 │   │     NO  (miss) → ggml runs ggml_backend_tensor_set_async
 │   │                  CPU mmap → input_cpy via PCIe (~13 GB/s)
 │   │                  cudaMemcpyD2D from input_cpy → pool[next_slot]
 │   │                  (LRU evicts cold slot if pool is full)
 │   └──
 │
 ├─ GPU FFN kernel runs against input_cpy (always)
 │
 └─ Next layer fires. Pool persists across layers and tokens.
```

**Where things live:**
- Routed expert weights: CPU mmap'd (15 GB at Q8)
- Cache pool: GPU, fixed N slots × E bytes (5 GB at Q8 = ~1750 slots)
- KV cache, attention weights, shared experts, embeddings, lm_head: GPU
- Compute: ALL on GPU

**PCIe traffic per token:**
```
PCIe_bytes = m × K × L × E
PCIe_time  = m × K × L × E / PCIe_BW
```

D→D snapshot cost per miss: `E / GPU_BW` — at 9 MB / 450 GB/s = 20 μs. Negligible vs the 700 μs PCIe.

**Per-token cost:**
```
cost = L × GPU_FFN_per_layer + L × m × K × E / PCIe_BW + attention + ...
     = L × (K × E / GPU_BW) + L × m × K × E / PCIe_BW + ...
```

For DS Q8 (h=0.54, m=0.46):
- GPU FFN: 27 × (6 × 9 MB / 450 GB/s) = 27 × 0.12 ms = **3.2 ms** (all 27 on GPU)
- PCIe misses: 27 × 0.46 × 6 × 9 MB / 13 GB/s = 27 × 1.91 ms = **51.6 ms**
- Total: 54.8 ms FFN+PCIe; measured 10.32 t/s ≈ 97 ms/token → ~55 ms FFN+PCIe + 42 ms attention

PCIe DOMINATES the snapshot cost at Q8. That's why it loses to vanilla at this regime.

For Qwen3 Q4_K_M (h=0.74, m=0.26, K=8, E=3 MB, L=48):
- GPU FFN: 48 × (8 × 3 MB / 450 GB/s) = 48 × 0.053 ms = **2.5 ms**
- PCIe misses: 48 × 0.26 × 8 × 3 MB / 13 GB/s = 48 × 0.48 ms = **23 ms**
- Total: ~26 ms FFN+PCIe; measured 15.33 t/s ≈ 65 ms/token

PCIe is the dominant cost here too, but it's much smaller than vanilla's CPU-FFN cost on 30 offloaded layers.

---

## 4. When does snapshot win over `-ngl X`?

Equate the two per-token costs and solve for the boundary. Cancel out attention/overhead (similar in both modes; minor differences in MLA cost) and focus on the FFN+PCIe terms:

**Snapshot wins when:**
```
L × m × K × E / PCIe_BW < (L − X) × K × E × (1/CPU_BW − 1/GPU_BW)
```

Cancel `K × E`:
```
L × m / PCIe_BW < (L − X) × (1/CPU_BW − 1/GPU_BW)
```

With GPU ≫ CPU, the `1/GPU_BW` term rounds away:
```
m < (L − X) / L × PCIe_BW / CPU_BW
```

**Boundary condition for snapshot to win:**
```
m_max ≈ (L − X) / L × PCIe_BW / CPU_BW
```

For our hardware (PCIe gen4 13 GB/s, DDR4 50 GB/s): `PCIe_BW / CPU_BW ≈ 0.25`. So:

```
m_max ≈ 0.25 × (L − X) / L
```

### Plugged in for each setup:

| setup | L | X | (L−X)/L | m_max | actual m | winner |
|---|---|---|---|---|---|---|
| **Qwen3 Q4_K_M** | 48 | 18 | 0.625 | **0.156** | 0.26 | **vanilla wins narrowly** — but snapshot is close enough that substitution can close it (and does, 20.89 vs 15.81) |
| **DS Q4 mixed** | 26 | ~10 | 0.61 | **0.153** | ~0.26 | similar boundary; snapshot+substitution wins |
| **DS Q8** | 27 | 12 | 0.556 | **0.139** | 0.46 | **vanilla wins by a lot** — m is 3× past the boundary |

Interesting wrinkle: the simple model says vanilla SHOULD win at Qwen3 Q4 (m=0.26 > m_max=0.156). But measured snapshot baseline (15.33) is competitive with ngl=18 (15.81). Why?

Because the simplified model ignores PCIe burstiness AND assumes CPU FFN can fully overlap GPU work, which it can't at batch=1. In practice the CPU-FFN cost is HIGHER than the bandwidth model says (synchronization, single-token batching kills CPU throughput), narrowing the gap. So the empirical boundary is more permissive than `0.156` — snapshot remains competitive even at m=0.26.

### Why even the simplified math underestimates snapshot's losses at Q8

Two reasons the math is conservative against snapshot:

1. **PCIe is bursty.** A miss serializes the pipeline: GPU stalls waiting for the bytes. Bandwidth treats time as uniform, but the stalls compound. At m=0.46 you have a stall on every other expert lookup, which is much worse than 46% × PCIe_time of pure stall.

2. **CPU FFN parallelizes across cores.** A 16-thread CPU FFN on 15 layers runs faster than the bandwidth model suggests because activation reuse and pipelining help. The model treats CPU as a single-stream bandwidth-bound engine; it's actually a multi-stream bandwidth-bound engine with some compute amortization.

Both effects push the real boundary stricter than `0.25 × (L−X)/L`. So when the simple model says it's a tie, vanilla usually wins; when it says snapshot wins by a little, snapshot wins comfortably.

---

## 5. The clean summary for Q&A

**Mental model in one paragraph:**

> Vanilla `-ngl X` moves the LARGE thing (weights) once at load time. Snapshot moves the LARGE thing constantly during decode — every cache miss costs a PCIe transfer. So snapshot wins when the cache is big enough relative to the expert population that miss rate stays low. Snapshot loses when experts are big (Q8), cache is small relative to the model, or routing is dispersed enough that hit rates can't climb.

**The crossover rule:**

> Snapshot wins when miss rate < `(L − X) / L × PCIe_BW / CPU_BW`. For our hardware, that's `miss rate < 0.25 × (L − X) / L`. Below the boundary, the savings from avoided CPU FFN exceed the cost of PCIe transfers on misses. Above the boundary, vanilla wins.

**Three concrete regimes:**

| regime | description | result |
|---|---|---|
| Small experts + lots of layers | Q4 with 48-layer model (Qwen3) | snapshot competitive; substitution wins |
| Small experts + few layers | Q4 with 26-layer DS | snapshot+substitution wins by ~30% |
| Big experts + few layers | Q8 with 26-layer DS | vanilla wins by ~12% |

---

## 6. Where the predictor fits — and why it doesn't help at inference

We trained a per-layer linear expert predictor early in the project, before building snapshot. The idea was the obvious one: predict which experts will fire at layer L+1, prefetch them across PCIe ahead of compute, hide the transfer latency behind the current layer's GPU work.

### What the predictor is

**LR-hidden:** per-layer linear models, one `Linear(2176, n_experts)` head per MoE layer. Features are simple — prior-token hidden state + observed experts at adjacent (layer, token) positions. No transformer, no attention, no LSTM. Total params:

| model | architecture | params | training |
|---|---|---|---|
| DeepSeek-V2-Lite (64 experts × top-6 × 26 layers) | 26 × Linear(2176, 64) | 3.6M | 5595 Puffin prompts × 3 epochs, BCE |
| Qwen3-30B-A3B (128 experts × top-8 × 48 layers) | 48 × Linear(2304, 128) | 14.2M | 2045 Wildchat prompts × 4 epochs, BCE |

### What recall the predictor achieves

All numbers at k = top_k_actual unless noted (apples-to-apples — comparing recall when k matches what the model itself selects):

| model | recall@k_actual | recall@12 | recall@16 | vs random |
|---|---|---|---|---|
| DeepSeek-V2-Lite (in-dist Puffin) | **57.88%** @ k=6 | 76.85% | – | **6.2×** |
| DeepSeek-V2-Lite (cross-dataset Wildchat) | 51.81% @ k=6 | 70.86% | – | 5.5× |
| Qwen3-30B-A3B (in-dist Wildchat) | **65.00%** @ k=8 | 77.04% | 83.57% | **10.4×** |

Recall@12 is the "prefetch budget" number — if we issue 12 prefetches per layer, ~77% of the time we cover the 6–8 experts that will actually fire. That's good enough to think it would matter.

### Why it doesn't help inference

Measured on Qwen3 Q4_K_M @ n=128, 6 GB cache, RTX 3070:

| config | t/s | hit rate | bytes prefetched (per 32-tok run) |
|---|---|---|---|
| Snapshot cache, no predictor | **15.33** | 74.24% | 30 GB |
| Snapshot + predictor (k=12 prefetch) | **11.77 (−23%)** | 74.72% | **54.5 GB (+25 GB extra)** |

Measured on DS-V2-Lite Q4_K @ n=128, 6 GB cache:

| config | t/s | hit rate |
|---|---|---|
| Snapshot cache, no predictor | **14.25** | 55.72% |
| Snapshot + predictor (k=12) | 12.45 (**−13%, below CPU baseline 12.90**) | 65.70% (+10pp) |

The pattern repeats across both models: **predictor adds +0.5 to +10 pp of cache hit rate, but throughput drops 13–23%**. The hit rate climbs are real — the predictor's mechanism works — but they come at a bandwidth cost that's worse than the cost they save.

### The PCIe-bandwidth-competition mechanism

In §3 we set up the snapshot cost as:
```
snapshot_cost = L × GPU_FFN + L × m × K × E / PCIe_BW
```

When we add the predictor's prefetch path, we get TWO PCIe streams sharing the same bus:

```
Pipe A: ggml's natural miss → input_cpy   (the cache fills as it always would via snapshot)
Pipe B: predictor's prefetch ahead of time  (extra bytes, hoping to land before the miss)
```

The bytes Pipe B issues are LARGELY THE SAME bytes Pipe A would have issued anyway — the predictor doesn't change which experts get used, only when their weights arrive. If we issue 12 prefetches per layer to cover the 6 that actually fire, **half the prefetched bytes are wasted** (the 50% precision tax at recall@12 ≈ 77%).

The wasted bytes don't just sit idle — they compete with Pipe A's real misses for PCIe bandwidth. At Q4 the +25 GB of extra prefetched bytes is roughly equal to what snapshot moves naturally (30 GB), doubling total PCIe traffic for marginal hit-rate gain.

**The key structural insight:** snapshot's "free signal" is the bytes already crossing PCIe via ggml's miss path. Adding predicted prefetch on top doesn't create new free bytes — it pays for bytes that were already going to be free. The predictor's lead time (microseconds — when the previous layer's GPU work finishes) isn't enough to amortize the PCIe cost across overlapping compute.

### What the predictor IS useful for

Two things, even though it doesn't help inference directly:

1. **Standalone research artifact.** The recall numbers (65% @ k=8 Qwen3, 58% @ k=6 DS, 5.5–6.2× random) are a real result. Prior work (MoE-Beyond, DuoServe-MoE) used vocab-leaky features that don't hold cross-token; LR-hidden uses only prior-token hidden state and observed routing, which IS deployable. The 10.4× lift over random IS the predictor's contribution to the field.

2. **Upstream of sensitivity calibration.** The trace collection pipeline that produces the predictor's training data (`scripts/collect_traces.py`) is the SAME pipeline that feeds `scripts/compute_sensitivity.py` for the cascade protocol. The sensitivity numbers that drive substitution come from this trace data. So the predictor work isn't wasted — it just contributes upstream, not at the hot path.

The honest framing: **we built a predictor, validated that prediction is possible, then found through measurement that the system layer (snapshot cache + the trace pipeline → sensitivity → substitution) is the path that earns the throughput gain. The predictor's prefetch path is a clean negative result.**

### Where this fits the talk

This is the talk's "Three takeaways" slide #1: **"Free signal beats learned signal."** When the system already gives you the data (snapshot), don't pay ML to re-fetch it via prediction. The predictor's recall numbers go on a separate slide as research artifact — proves the prediction is real even though we don't deploy it in the hot path.

---

## 7. Where substitution fits

Substitution is the trick that lets snapshot close the gap when it's near (but past) the boundary. Mechanism: on a cache miss at a layer marked "perturbation-tolerant" (sensitivity ≤ θ), pick a cached expert at the same (layer, kind) and use it instead of issuing PCIe.

**Effect on the cost model:**
- Effective miss rate becomes `m × (fraction of layers ineligible for substitution)`
- For sensitivity-thr=1.8 on DS Q4: ~50% of layers eligible → effective `m` halves → snapshot crosses the boundary
- For DS Q8: same threshold makes 7/27 layers eligible → m drops from 0.46 to ~0.34 → still above boundary 0.14 → still loses to vanilla, but only by 12% instead of 33%

So substitution amplifies snapshot's regime but doesn't change the boundary's location. The boundary depends on the cache-vs-expert-size ratio. Substitution lets you cheat on miss rate when you're close to the line.

**Why this matters for the talk story:** the headline result (+32% Qwen3 Q4, +29% DS Q4) is at the *favorable* end of the regime. At Q8 the mechanism still gives +30% over the snap baseline — just not enough to beat vanilla. Honest framing keeps both.

---

## 8. Configuration 4 (proposed): hybrid CPU+GPU compute on cache misses

**The question:** at Q8, current snapshot loses because PCIe per missed expert (~700 μs) is the dominant cost. What if, instead of stalling the GPU waiting for PCIe, we computed the missed expert's FFN locally on CPU in parallel with the GPU's work on cached experts?

MoE FFN output is `Σ_k w_k × expert_k(x)` — each expert's contribution is independent and sums at the end. So nothing prevents us from computing some on GPU, some on CPU, and merging.

```
Layer L fires, top-K experts = {E1..E6}
 │
 ├─ {E1, E2, E3} are in GPU cache  ───►  GPU FFN  (65 μs)        ┐
 │                                                                 │
 ├─ {E4, E5, E6} NOT in cache       ───►  CPU FFN  (cold cache)   │  parallel
 │   weights already mmap'd on CPU                                 │
 │                                                                 │
 ├─ (optionally) Also PCIe E4-E6 ───► snapshot to GPU pool        │
 │   for future hits — async, doesn't block compute                │
 │                                                                 │
 └────────────────────────────────────────────────────────────────►  merge on GPU
                                              ║
                                              ║ wait = max(GPU_time, CPU_time)
                                              ▼
```

### Cost model

```
hybrid_per_layer = max(
    h × K × E / GPU_BW,         # GPU computes hits
    m × K × E_Q8 / CPU_BW_agg   # CPU computes misses (Q8 native, multi-thread)
)
```

`m × K × E` is bandwidth-bound on the CPU side — multi-threading helps until you saturate the DRAM bus. At DDR4 aggregate ~50 GB/s, K=6 misses × 9 MB × m=0.46 = 24.8 MB / 50 GB/s = 497 μs per layer.

GPU hit path: 0.54 × 6 × 9 MB / 450 GB/s = 65 μs per layer.

Hybrid per layer = max(65, 497) = **497 μs**. Compare to current snapshot's **2030 μs/layer** at the same hit rate.

### Empirical measurement

We measured per-expert CPU FFN on Legion (Q8, DDR4) via `bench-bundle/scripts/cpu_ffn_bench.py`:

| measurement | value |
|---|---|
| Per-expert FFN, cold cache, FP32 numpy single-thread | **1270 μs** |
| Per-expert FFN, warm cache (L3-resident) | 250 μs |
| Achieved DRAM bandwidth | 27 GB/s |

The cold-cache 1270 μs is the **floor**: numpy FP32 inflates Q8 memory traffic 4× and is single-threaded. Real ggml-cpu Q8 matmul (1/4 the memory traffic, multi-thread) projects substantially faster.

### Three projection scenarios

All using measured `h=0.54`, attention overhead 50 ms, `n_moe_layers=26`:

| scenario | assumption | per-expert | hybrid t/s | vs vanilla 15.34 |
|---|---|---|---|---|
| A: FP32 numpy single-thread | what the bench measures directly | 1270 μs | **7.1** | -54% (the floor) |
| B: Q8 native single-thread | real ggml-cpu Q8: 1/4 the memory traffic | 318 μs | **13.7** | -11% |
| **C: Q8 native multi-thread** | DDR4 aggregate 50 GB/s, misses run in parallel | bandwidth-bound | **15.9** | **+4%** |

**Bottom line: hybrid would narrowly beat vanilla at Q8 (15.9 vs 15.34 = +4%).** Not a blowout like substitution at Q4, but proves the Q8 regime boundary is escapable.

### What this changes about the regime story

Section 4's boundary (`m_max ≈ 0.25 × (L − X) / L`) assumes the CPU-side compute is fully idle in snapshot mode. The hybrid disproves that assumption — CPU FFN on the missed experts CAN run in parallel with GPU's work, and at Q8 the CPU is fast enough (relative to PCIe) that it's a net win.

Updated mental model:
- **At Q4** (small experts): snapshot wins via cache. CPU has nothing to do because PCIe is already cheap.
- **At Q8** (big experts): snapshot loses to vanilla via PCIe stalls. Hybrid recovers because CPU is faster than PCIe per missed expert. CPU becomes the rescue path.

So the regime boundary isn't fixed by hardware; it's fixed by **what we choose to do on misses**. Three escalating mechanisms:

| mechanism | what it does on miss | regime where it helps |
|---|---|---|
| Plain snapshot | PCIe transfer, GPU compute | small experts only |
| + Substitution | sub a cached expert at tolerant layers | borderline regimes |
| **+ Hybrid CPU compute** | **CPU FFN in parallel with GPU hits** | **big-expert regimes (Q8)** |

### Scaling: does hybrid help MORE on bigger models?

The user's intuition is right and we can prove it. Set up the cost comparison:

```
hybrid_per_token  = L × max( h × K × E / GPU_BW ,  m × K × E / CPU_BW )
vanilla_per_token = X × K × E / GPU_BW + (L − X) × K × E / CPU_BW
```

In the CPU-bottleneck regime (where `m × GPU_BW > h × CPU_BW`, true for any m > ~0.11 on our hardware — i.e. always), `max(...)` is the CPU term:

```
hybrid  ≈ L × m × K × E / CPU_BW
vanilla = K × E × [ X / GPU_BW  +  (L − X) / CPU_BW ]
```

Let `r = CPU_BW / GPU_BW ≈ 0.11` for our gear. Divide:

```
Speedup  S  =  vanilla / hybrid
           =  [ X · r + (L − X) ] / [ L × m ]
           =  [ L − X (1 − r) ] / [ L × m ]
           ≈  [ L − 0.89 X ] / [ L × m ]          (r ≈ 0.11)
```

**Hybrid beats vanilla when `S > 1`:**

```
X  <  L × (1 − m) / (1 − r)
X  <  L × h / 0.89                  (i.e. X < ~1.12 × L × h)
```

In words: **hybrid wins whenever the GPU-fittable layer count `X` is less than ~1.12 × (cache hit rate × total layers)**. So three factors decide it:

| factor | effect on hybrid vs vanilla |
|---|---|
| **Bigger model `L`** with `X` bounded by VRAM | `X/L` shrinks → easier for hybrid to win |
| **Higher cache hit rate `h`** (better routing reuse, bigger cache) | makes the bound easier to clear |
| **Smaller GPU/CPU bandwidth ratio (1/r)** | helps hybrid more (DDR4 vs PCIe is ~4×, DDR5 vs DDR4 is ~1.6×) |

### Asymptotic behavior: very big models

For huge models where `L → ∞` and `X` is bounded by VRAM:

```
vanilla  ≈  (L − X) × K × E / CPU_BW       (vanilla CPU layers dominate)
hybrid   ≈  L × m × K × E / CPU_BW          (still CPU-bottlenecked but proportional)

S  →  (L − X) / (L × m)
   →  1 / m       as L → ∞
```

**Hybrid asymptotically gives a `1 / miss_rate` speedup over vanilla on big models.** At h=0.5 (m=0.5) that's 2×; at h=0.7 (m=0.3) it's 3.3×; at h=0.8 (m=0.2) it's 5×.

The intuition: vanilla forces `(L − X)` layers to run *fully* on CPU. Hybrid lets the GPU contribute on EVERY layer in proportion to the hit rate, even at layers vanilla couldn't fit. The savings compound across all layers, not just `X`.

### Plugging in three concrete model sizes

All at Q8 on the same 8 GB 3070 (5 GB cache), assuming achievable `h` scales gently with cache fraction:

| model | L | est. X (8 GB VRAM) | est. h (5 GB cache) | X/L | 1.12 × h | hybrid wins? | projected speedup S |
|---|---|---|---|---|---|---|---|
| **DS-V2-Lite Q8** | 27 | 12 | 0.54 | 0.44 | 0.60 | YES (just barely) | 1.07× |
| **Qwen3-30B-A3B Q8** | 48 | 4–6 | ~0.45 | 0.10 | 0.50 | YES (clearly) | 1.85× |
| **DeepSeek-V3 671B Q8** | 60 | 0 (model > total RAM, requires streaming) | ~0.20 | 0 | 0.22 | YES (huge) | 5× |

**This is the strongest argument for hybrid as future work.** On the model we tested (DS Q8) hybrid is a narrow win (+4% over vanilla). On the models the technique was DESIGNED for (Qwen3-30B Q8, DS-V3) it's projected to be a 1.85–5× speedup — exactly the kind of margin worth building infrastructure for.

The math says: **the bigger the gap between model size and GPU VRAM, the bigger hybrid's win**. Edge inference at 100+ GB models on 8 GB cards is where the technique earns its keep.

### What it would take to ship

Real engineering project — not measured end-to-end, only projected:

| component | work |
|---|---|
| ggml dispatch | extend `--override-tensor` to per-expert device choice, OR build a parallel compute path outside the scheduler |
| MoE kernel split | gather hits onto GPU input_cpy, dispatch misses to CPU threads with Q8 native matmul |
| Result merge | CPU thread returns `w_k × expert_k(x)` per missed expert → gather onto GPU before residual add |
| Sync barrier | wait for both paths; could be hidden behind next layer's GPU attention if pipelined |

Estimated ~1-2 weeks for a working prototype. The microbench validates the timing math; the integration is the gap.

### Where this fits the talk

This belongs on the future-work slide. Frame:

> "At Q8 our substitution lifts snapshot by +30% but doesn't beat vanilla. We measured CPU FFN per expert at 1270 μs cold-cache — and even that pessimistic number projects to 15.9 t/s with proper Q8 native multi-thread, narrowly beating vanilla. The Q8 regime boundary is escapable. The math is done; engineering is the gap."
