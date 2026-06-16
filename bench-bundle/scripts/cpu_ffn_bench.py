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
    gate_w = get_expert_weights(gate_t, args.expert)  # likely (interm_dim, hidden_dim)
    up_w = get_expert_weights(up_t, args.expert)
    down_w = get_expert_weights(down_t, args.expert)  # likely (hidden_dim, interm_dim)
    t_extract = time.perf_counter() - t0
    print(f"Extraction (dequant) took {t_extract*1000:.1f} ms")

    print(f"  gate_w shape: {gate_w.shape}, dtype: {gate_w.dtype}, size: {gate_w.nbytes/1e6:.2f} MB")
    print(f"  up_w   shape: {up_w.shape}")
    print(f"  down_w shape: {down_w.shape}")

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

    # --- Benchmark single-expert FFN ---
    print(f"\nWarming up...")
    for _ in range(10):
        if gate_orient == "standard":
            g = gate_w @ hidden
            u = up_w @ hidden
        else:
            g = hidden @ gate_w
            u = hidden @ up_w
        f = silu(g) * u
        if down_w.shape[0] == H:
            out = down_w @ f
        else:
            out = f @ down_w
    _ = out.sum()  # force completion

    print(f"Timing single-expert FFN over {args.n_trials} trials...")
    start = time.perf_counter()
    for _ in range(args.n_trials):
        if gate_orient == "standard":
            g = gate_w @ hidden
            u = up_w @ hidden
        else:
            g = hidden @ gate_w
            u = hidden @ up_w
        f = silu(g) * u
        if down_w.shape[0] == H:
            out = down_w @ f
        else:
            out = f @ down_w
    elapsed = time.perf_counter() - start
    per_expert_us = elapsed / args.n_trials * 1e6

    expert_bytes_fp32 = gate_w.nbytes + up_w.nbytes + down_w.nbytes
    # Note: we time FP32 matmul on dequantized weights. Real bench-aligned cost
    # would be Q8 dequant + INT8/FP32 matmul. For bandwidth comparison, the FP32
    # number gives us the upper bound (more memory traffic per op).
    # In real impl, you'd use ggml's Q8 CPU matmul which reads Q8 directly.
    expert_bytes_q8 = expert_bytes_fp32 // 4  # Q8 is ~4× more compact than FP32
    bw_fp32_gbs = expert_bytes_fp32 / per_expert_us * 1e3 / 1e9
    bw_q8_proj_gbs = expert_bytes_q8 / per_expert_us * 1e3 / 1e9

    print(f"\n=== Single-expert FFN result ===")
    print(f"  per-expert time:        {per_expert_us:.1f} μs")
    print(f"  FP32 weight bytes:      {expert_bytes_fp32/1e6:.2f} MB")
    print(f"  achieved FP32 bw:       {bw_fp32_gbs:.1f} GB/s")
    print(f"  equivalent Q8 bw:       {bw_q8_proj_gbs:.1f} GB/s  (what real bench would touch)")

    # --- Project to top-K case ---
    topk_us = per_expert_us * args.top_k
    print(f"\n=== Projected top-{args.top_k} layer cost ===")
    print(f"  Sequential CPU FFN:     {topk_us:.1f} μs ({topk_us/1000:.2f} ms) per layer")
    print(f"  vs PCIe gen4 (9 MB ea): {9000 / 13 * args.top_k:.0f} μs (one-way transfer, K experts)")
    print(f"  vs measured snap PCIe:  {0.46 * args.top_k * 9000 / 13:.0f} μs (at m=0.46)")

    # --- Project to whole-model ---
    # DS-V2-Lite has 26 MoE layers (layer 0 is dense)
    n_moe_layers = 26
    full_layer_us = per_expert_us * args.top_k
    total_per_token_ms = full_layer_us * n_moe_layers / 1000
    print(f"\n=== Projected whole-model FFN cost (if all CPU, no overlap) ===")
    print(f"  {n_moe_layers} MoE layers × top-{args.top_k}: {total_per_token_ms:.1f} ms FFN per token")
    print(f"  Measured CPU baseline:  ~93 ms/token total (incl. attention)")
    print(f"  → FFN share of total:   {total_per_token_ms / 93 * 100:.0f}%")

    # --- The hybrid hypothesis ---
    print(f"\n=== Hybrid (snap-hit on GPU || miss on CPU) projection ===")
    hit_rate = 0.54  # measured DS Q8 at 5 GB cache
    miss_rate = 1 - hit_rate
    gpu_per_layer_us = 65  # measured: K × E / 450 GB/s
    cpu_miss_us = per_expert_us * args.top_k * miss_rate
    hybrid_per_layer_us = max(gpu_per_layer_us, cpu_miss_us)
    hybrid_per_token_ms = hybrid_per_layer_us * n_moe_layers / 1000
    print(f"  GPU hit path (parallel):   {gpu_per_layer_us:.0f} μs/layer")
    print(f"  CPU miss path (parallel):  {cpu_miss_us:.0f} μs/layer  (at m={miss_rate:.2f})")
    print(f"  Hybrid per layer:          {hybrid_per_layer_us:.0f} μs (max of the two)")
    print(f"  Hybrid per-token FFN:      {hybrid_per_token_ms:.1f} ms")
    print(f"  + attention overhead:      ~50 ms (measured)")
    print(f"  → projected throughput:    ~{1000 / (hybrid_per_token_ms + 50):.1f} t/s")
    print(f"  vs vanilla -ngl 12:        15.34 t/s (measured)")
    print(f"  vs current snapshot:       10.32 t/s (measured)")


if __name__ == "__main__":
    main()
