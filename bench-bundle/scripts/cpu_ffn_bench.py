#!/usr/bin/env python3
"""
cpu_ffn_bench.py
----------------
Standalone benchmark: how long does CPU FFN take for ONE MoE expert?

Why this matters: in our current snapshot path on Q8, a cache miss triggers a
PCIe transfer (~700 μs per expert at gen4) followed by GPU FFN compute. The
hypothesis we want to test is: could we INSTEAD compute the missed expert's
FFN locally on CPU in parallel with GPU work on cached experts? That would
avoid the PCIe stall entirely.

The mmap'd expert weights are already in CPU memory (that's how llama.cpp's
copy_experts mechanism loads them on demand). All we need is a numerical
verification that CPU FFN per expert is faster than the PCIe-then-GPU path.

Theoretical prediction (back-of-envelope):
  - Q8 per-expert size = 9 MB (gate 3 MB + up 3 MB + down 3 MB)
  - DDR4 bandwidth = ~50 GB/s single-channel saturated
  - Single-expert CPU FFN ≈ 9 MB / 50 GB/s = 180 μs

  vs PCIe gen4 per expert: 9 MB / 13 GB/s = 700 μs

If measured ≈ predicted, we have green-light for hybrid compute as future work.

Usage:
  python3 cpu_ffn_bench.py --gguf path/to/DeepSeek-V2-Lite-Chat.Q8_0.gguf
  python3 cpu_ffn_bench.py --gguf path/to/DeepSeek-V2-Lite-Chat.Q8_0.gguf --layer 5 --expert 7
"""

from __future__ import annotations
import argparse
import sys
import time
from typing import Optional

import numpy as np

try:
    import gguf
    from gguf import GGUFReader
except ImportError:
    sys.exit("gguf-py not installed. Run: pip install --user --break-system-packages gguf")


def dequant_q8_0(raw_bytes: np.ndarray, n_elements: int) -> np.ndarray:
    """
    Dequantize Q8_0 GGUF blob into FP32.

    Q8_0 block layout: 32 elements packed into 34 bytes:
      bytes 0..1:  FP16 scale
      bytes 2..33: 32 INT8 values
    """
    assert n_elements % 32 == 0, f"Q8_0 needs n_elements divisible by 32, got {n_elements}"
    n_blocks = n_elements // 32
    expected_bytes = n_blocks * 34
    assert raw_bytes.nbytes >= expected_bytes, f"Not enough bytes: need {expected_bytes}, got {raw_bytes.nbytes}"

    blob = raw_bytes.view(np.uint8)[:expected_bytes].reshape(n_blocks, 34)
    scales = np.frombuffer(blob[:, :2].tobytes(), dtype=np.float16).astype(np.float32)  # (n_blocks,)
    ints = blob[:, 2:].view(np.int8)  # (n_blocks, 32)
    return (scales[:, None] * ints.astype(np.float32)).reshape(-1)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def find_expert_tensors(reader: GGUFReader, layer: int):
    """Find (gate_exps, up_exps, down_exps) tensors for a given layer."""
    gate = up = down = None
    for t in reader.tensors:
        n = t.name
        if f"blk.{layer}.ffn_gate_exps" in n:
            gate = t
        elif f"blk.{layer}.ffn_up_exps" in n:
            up = t
        elif f"blk.{layer}.ffn_down_exps" in n:
            down = t
    if gate is None or up is None or down is None:
        # Try without the .weight suffix variants
        for t in reader.tensors:
            n = t.name
            if gate is None and f"blk.{layer}.ffn_gate_exps" in n: gate = t
            if up is None and f"blk.{layer}.ffn_up_exps" in n: up = t
            if down is None and f"blk.{layer}.ffn_down_exps" in n: down = t
    return gate, up, down


def get_expert_weights(t, expert_id: int) -> np.ndarray:
    """
    Extract one expert's weights from a stacked expert tensor.

    GGUF stores ffn_*_exps with n_experts as the LAST dimension. Example for
    DeepSeek-V2-Lite Q8_0:
      blk.L.ffn_gate_exps  shape = (2048, 1408, 64)  = (hidden_dim, interm_dim, n_experts)
      blk.L.ffn_down_exps  shape = (1408, 2048, 64)  = (interm_dim, hidden_dim, n_experts)

    gguf-py's .data auto-dequantizes Q8_0 to float32 when supported; if it
    returns raw bytes (older versions), we dequantize manually.
    """
    data = t.data
    if data.dtype == np.uint8:
        shape = tuple(int(s) for s in t.shape)
        n_total = 1
        for s in shape:
            n_total *= s
        dequantized = dequant_q8_0(data, n_total)
        full = dequantized.reshape(shape)
    else:
        full = data

    # Detect which axis is n_experts. We assume it's the smallest dim and
    # equal across all three projection tensors (gate/up/down). For DS-V2-Lite
    # it's 64; for Qwen3 it's 128. The last dim is the convention in GGUF.
    if full.ndim == 3:
        # Slice the last axis (n_experts is the last dim by GGUF convention)
        return full[..., expert_id]
    elif full.ndim == 2:
        # Single-expert tensor (not stacked) — return as-is
        return full
    else:
        raise ValueError(f"Unexpected expert-tensor ndim={full.ndim}, shape={full.shape}")


def main():
    ap = argparse.ArgumentParser(description="Time CPU FFN for one MoE expert from a GGUF.")
    ap.add_argument("--gguf", required=True, help="Path to GGUF (Q8_0 expected)")
    ap.add_argument("--layer", type=int, default=1, help="MoE layer index (default 1)")
    ap.add_argument("--expert", type=int, default=0, help="Expert ID within the layer (default 0)")
    ap.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dim (DS-V2-Lite = 2048)")
    ap.add_argument("--top-k", type=int, default=6, help="Top-K active experts (DS = 6)")
    ap.add_argument("--n-trials", type=int, default=200, help="Trials for timing")
    ap.add_argument("--n-threads", type=int, default=None, help="Threads (default: numpy default)")
    args = ap.parse_args()

    if args.n_threads:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.n_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.n_threads)

    print(f"Reading {args.gguf} (this may take a moment for large GGUFs)...")
    reader = GGUFReader(args.gguf)
    print(f"Loaded. {len(reader.tensors)} tensors, {len(reader.fields)} metadata fields.")

    gate_t, up_t, down_t = find_expert_tensors(reader, args.layer)
    if gate_t is None:
        # Print available expert-related tensors to help debug
        candidates = [t.name for t in reader.tensors if "ffn_" in t.name and "exps" in t.name]
        print(f"\nCould not find layer {args.layer} expert tensors.")
        print("Available expert tensors (first 20):")
        for c in candidates[:20]:
            print(f"  {c}")
        sys.exit(1)

    print(f"\nUsing layer {args.layer}, expert {args.expert}:")
    print(f"  gate: {gate_t.name} shape={tuple(gate_t.shape)} type={gate_t.tensor_type.name}")
    print(f"  up:   {up_t.name} shape={tuple(up_t.shape)} type={up_t.tensor_type.name}")
    print(f"  down: {down_t.name} shape={tuple(down_t.shape)} type={down_t.tensor_type.name}")

    print(f"\nExtracting expert {args.expert} weights (this dequantizes Q8_0 → FP32, one-time cost)...")
    t0 = time.perf_counter()
    gate_w = get_expert_weights(gate_t, args.expert)
    up_w = get_expert_weights(up_t, args.expert)
    down_w = get_expert_weights(down_t, args.expert)
    t_extract = time.perf_counter() - t0
    print(f"Extraction (dequant) took {t_extract*1000:.1f} ms")
    print(f"  gate_w shape: {gate_w.shape}, dtype: {gate_w.dtype}, contiguous: {gate_w.flags['C_CONTIGUOUS']}")
    print(f"  up_w   shape: {up_w.shape}, contiguous: {up_w.flags['C_CONTIGUOUS']}")
    print(f"  down_w shape: {down_w.shape}, contiguous: {down_w.flags['C_CONTIGUOUS']}")

    # CRITICAL: slicing full[..., expert_id] from a 3D tensor gives a non-contiguous
    # view (stride 64 between columns because n_experts is interleaved). matmul on
    # non-contiguous data either silently copies per call or thrashes cache. Force
    # contiguous now so the timed matmul reflects real bandwidth, not stride access.
    print(f"\nForcing contiguous layout (was strided due to last-axis slice)...")
    t0 = time.perf_counter()
    gate_w = np.ascontiguousarray(gate_w)
    up_w = np.ascontiguousarray(up_w)
    down_w = np.ascontiguousarray(down_w)
    t_contig = time.perf_counter() - t0
    print(f"Contiguous copy took {t_contig*1000:.1f} ms")
    print(f"  gate_w contiguous now: {gate_w.flags['C_CONTIGUOUS']}, {gate_w.nbytes/1e6:.2f} MB")

    # Print numpy's BLAS info so we know what we're benchmarking
    try:
        info = np.show_config(mode="dicts")
        blas = info.get("Build Dependencies", {}).get("blas", {})
        print(f"\nNumPy BLAS: name={blas.get('name','?')} version={blas.get('version','?')}")
    except Exception:
        pass

    # Validate dims: gate and up should produce same intermediate dim from hidden_dim input
    H = args.hidden_dim
    # gate_w should multiply hidden (H,) to produce (interm_dim,)
    # So gate_w should be (interm_dim, H) or (H, interm_dim) — check
    if gate_w.shape[1] == H:
        # (interm_dim, H) — matmul gate_w @ hidden
        gate_orient = "standard"
        interm_dim = gate_w.shape[0]
    elif gate_w.shape[0] == H:
        # (H, interm_dim) — matmul hidden @ gate_w
        gate_orient = "transposed"
        interm_dim = gate_w.shape[1]
    else:
        print(f"\nWARNING: gate_w shape {gate_w.shape} doesn't match hidden_dim {H} on either axis.")
        print("Best guess: gate_w shape's smaller dim is hidden_dim. Try --hidden-dim override.")
        sys.exit(1)
    print(f"  inferred orient: {gate_orient}, interm_dim = {interm_dim}")

    # Random hidden state input — represents one token's hidden representation
    rng = np.random.default_rng(42)
    hidden = rng.standard_normal(H, dtype=np.float32)

    def ffn_step(g_w, u_w, d_w):
        """One FFN pass: silu(g@x) * (u@x) → d @ result. Returns the output."""
        if gate_orient == "standard":
            g = g_w @ hidden
            u = u_w @ hidden
        else:
            g = hidden @ g_w
            u = hidden @ u_w
        f = silu(g) * u
        return d_w @ f if d_w.shape[0] == H else f @ d_w

    # --- Mode 1: WARM cache (same expert every trial) ---
    print(f"\n=== Mode 1: WARM cache (same expert repeated, hits L3) ===")
    print(f"Warming up...")
    for _ in range(10):
        out = ffn_step(gate_w, up_w, down_w)
    _ = out.sum()

    print(f"Timing single-expert FFN over {args.n_trials} trials (warm)...")
    start = time.perf_counter()
    for _ in range(args.n_trials):
        out = ffn_step(gate_w, up_w, down_w)
    elapsed_warm = time.perf_counter() - start
    per_expert_warm_us = elapsed_warm / args.n_trials * 1e6

    expert_bytes_fp32 = gate_w.nbytes + up_w.nbytes + down_w.nbytes
    expert_bytes_q8 = expert_bytes_fp32 // 4
    # Correct GB/s: bytes / μs / 1e3 (since 1 GB = 1e9 B and 1 s = 1e6 μs)
    bw_warm_fp32 = expert_bytes_fp32 / per_expert_warm_us / 1e3
    bw_warm_q8 = expert_bytes_q8 / per_expert_warm_us / 1e3
    print(f"  per-expert time (warm):  {per_expert_warm_us:.1f} μs")
    print(f"  FP32 weight bytes:       {expert_bytes_fp32/1e6:.2f} MB")
    print(f"  achieved FP32 bw (warm): {bw_warm_fp32:.1f} GB/s   (hitting L3 — overestimates)")
    print(f"  equivalent Q8 bw:        {bw_warm_q8:.1f} GB/s")

    # --- Mode 2: COLD cache (rotate through all experts → forces DRAM reads) ---
    n_experts = int(gate_t.shape[-1])
    print(f"\n=== Mode 2: COLD cache (rotate {n_experts} experts → DRAM-bound) ===")
    print(f"Loading all {n_experts} experts' weights to RAM (this takes ~{n_experts*3*0.7:.0f}s for dequant)...")
    t0 = time.perf_counter()
    all_gate = [np.ascontiguousarray(get_expert_weights(gate_t, e)) for e in range(n_experts)]
    all_up   = [np.ascontiguousarray(get_expert_weights(up_t,   e)) for e in range(n_experts)]
    all_down = [np.ascontiguousarray(get_expert_weights(down_t, e)) for e in range(n_experts)]
    total_gb = sum(w.nbytes for w in all_gate + all_up + all_down) / 1e9
    print(f"Loaded all experts in {time.perf_counter()-t0:.1f}s. Total FP32 working set: {total_gb:.2f} GB (way bigger than L3)")

    # Warm up by touching first few
    for e in range(min(5, n_experts)):
        _ = ffn_step(all_gate[e], all_up[e], all_down[e])

    print(f"Timing FFN rotating through {n_experts} experts over {args.n_trials} trials (cold)...")
    start = time.perf_counter()
    for i in range(args.n_trials):
        e = i % n_experts
        out = ffn_step(all_gate[e], all_up[e], all_down[e])
    elapsed_cold = time.perf_counter() - start
    per_expert_cold_us = elapsed_cold / args.n_trials * 1e6

    bw_cold_fp32 = expert_bytes_fp32 / per_expert_cold_us / 1e3
    bw_cold_q8 = expert_bytes_q8 / per_expert_cold_us / 1e3
    print(f"  per-expert time (cold):  {per_expert_cold_us:.1f} μs")
    print(f"  achieved FP32 bw (cold): {bw_cold_fp32:.1f} GB/s   (DRAM-bound, this is the real number)")
    print(f"  equivalent Q8 bw:        {bw_cold_q8:.1f} GB/s")

    # For projections downstream, use the COLD measurement
    per_expert_us = per_expert_cold_us

    # --- Three hybrid scenarios with explicit assumptions ---
    # Scenario A: FP32 numpy single-thread (what THIS bench measures directly)
    # Scenario B: Q8 native single-thread (real ggml-cpu Q8 matmul: 4× less memory traffic)
    # Scenario C: Q8 native multi-thread bandwidth-limited (parallel misses share DRAM)
    print(f"\n=== Three hybrid projection scenarios ===")
    print(f"All assume cache hit rate h=0.54 (measured DS Q8 at 5 GB cache).")
    print(f"Numbers in parens: per-token throughput including ~50 ms attention overhead.\n")

    hit_rate = 0.54
    miss_rate = 1 - hit_rate
    gpu_per_layer_us = 65          # measured: K × E_Q8 / GPU_BW @ 450 GB/s
    n_moe_layers = 26
    attn_ms = 50

    def project_hybrid(cpu_per_expert_us: float, label: str, assumption: str):
        cpu_layer_us = cpu_per_expert_us * args.top_k * miss_rate  # sequential miss processing
        hybrid_layer_us = max(gpu_per_layer_us, cpu_layer_us)
        hybrid_ms = hybrid_layer_us * n_moe_layers / 1000
        tok_per_sec = 1000 / (hybrid_ms + attn_ms)
        print(f"  [{label}]  per-expert {cpu_per_expert_us:>6.0f} μs  →  "
              f"hybrid {hybrid_layer_us:>5.0f} μs/layer  ({tok_per_sec:>4.1f} t/s)")
        print(f"    └─ {assumption}")
        return tok_per_sec

    # A — measured FP32 numpy, single-thread (THIS bench)
    project_hybrid(
        per_expert_us, "A: FP32 numpy",
        "what this script measured. Floor: FP32 inflates Q8 memory traffic 4×.")

    # B — Q8 native single-thread (theoretical, 4× faster on memory)
    q8_native_per_expert = per_expert_us / 4
    project_hybrid(
        q8_native_per_expert, "B: Q8 native",
        "real ggml-cpu Q8 matmul reads 1/4 the bytes (3 MB vs 11.5 MB per projection).")

    # C — Q8 native + multi-thread bandwidth-bound (aggregate DRAM 50 GB/s, top-K parallel)
    # Each missed expert handled by its own thread; total bytes = m × K × 9 MB; bandwidth shared.
    DRAM_AGG_GBS = 50.0
    cpu_layer_us_C = miss_rate * args.top_k * 9.0 / DRAM_AGG_GBS * 1000  # ms then μs
    hybrid_layer_us_C = max(gpu_per_layer_us, cpu_layer_us_C)
    hybrid_ms_C = hybrid_layer_us_C * n_moe_layers / 1000
    tok_C = 1000 / (hybrid_ms_C + attn_ms)
    print(f"  [C: Q8 multi-thread]  CPU bytes/layer = m×K×9MB = {miss_rate * args.top_k * 9:.1f} MB  →  "
          f"hybrid {hybrid_layer_us_C:>5.0f} μs/layer  ({tok_C:>4.1f} t/s)")
    print(f"    └─ {DRAM_AGG_GBS:.0f} GB/s aggregate DRAM (DDR4 dual-channel), parallel across cores.")

    print(f"\n  baselines for comparison:")
    print(f"    vanilla -ngl 12 (measured):  15.34 t/s")
    print(f"    current snapshot (measured): 10.32 t/s")
    print(f"\n  bottom line: integration of Q8 native CPU FFN would be needed to test scenario B/C.")
    print(f"  scenario A is the floor and isn't competitive. Scenario C would beat vanilla at Q8.")


if __name__ == "__main__":
    main()
