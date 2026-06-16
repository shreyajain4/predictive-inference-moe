#!/usr/bin/env python3
"""
cpu_ffn_load.py
---------------
Continuous CPU FFN load generator. Designed to run CONCURRENTLY with the
snapshot+substitution bench on the same machine to measure whether running
CPU FFN matmul in parallel with GPU work causes measurable contention.

The hypothesis the hybrid CPU+GPU compute proposal rests on: while GPU is
busy on cached experts, the CPU can compute missed-expert FFN in parallel
without slowing the GPU down. This test validates that claim empirically
by saturating the CPU side and observing the GPU throughput.

Usage:
  # Terminal 1 (baseline) — snapshot bench, no CPU load
  ./llama.cpp/build/bin/llama-moe-predictor-bench [args...]
  # Note the t/s

  # Terminal 2 — run snapshot bench while THIS script burns CPU
  python3 cpu_ffn_load.py --gguf <path> --threads 12 &
  ./llama.cpp/build/bin/llama-moe-predictor-bench [args...]
  # Compare t/s to baseline

  # If snapshot bench's t/s is unchanged → CPU FFN runs in parallel without contention
  # If snapshot bench's t/s drops → CPU/GPU contention is real, hybrid claim is weaker

Reports total FFN ops/sec and per-second throughput so you can see if
the CPU load is sustained.
"""

from __future__ import annotations
import argparse
import sys
import time
import threading
from typing import List

import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("gguf-py not installed. Run: pip install --user --break-system-packages gguf")


def dequant_q8_0(raw_bytes: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 GGUF blob to FP32. 32-element blocks: FP16 scale + 32 int8."""
    assert n_elements % 32 == 0, f"Q8_0 needs n_elements % 32 == 0, got {n_elements}"
    n_blocks = n_elements // 32
    expected = n_blocks * 34
    blob = raw_bytes.view(np.uint8)[:expected].reshape(n_blocks, 34)
    scales = np.frombuffer(blob[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    ints = blob[:, 2:].view(np.int8)
    return (scales[:, None] * ints.astype(np.float32)).reshape(-1)


def get_expert_2d(t, expert_id: int) -> np.ndarray:
    """Extract one expert's 2D weight slice from a stacked GGUF tensor.

    Handles both auto-dequantized (newer gguf-py → float32) and raw-bytes
    (older gguf-py → uint8) cases. Slices the LAST axis since GGUF stacks
    experts with n_experts as the last dim.
    """
    data = t.data
    if data.dtype == np.uint8:
        # Raw bytes — dequant manually using the full tensor shape
        shape = tuple(int(s) for s in t.shape)
        n_total = 1
        for s in shape:
            n_total *= s
        full = dequant_q8_0(data, n_total).reshape(shape)
    else:
        full = data
    if full.ndim == 3:
        return np.ascontiguousarray(full[..., expert_id])
    elif full.ndim == 2:
        return np.ascontiguousarray(full)
    raise ValueError(f"Unexpected ndim={full.ndim}, shape={full.shape}")


def find_and_load_all_experts(gguf_path: str, layer: int, verbose: bool = True):
    """Load all routed expert weights for one layer into a list of (gate, up, down) tuples (FP32)."""
    if verbose:
        print(f"Reading {gguf_path}...")
    reader = GGUFReader(gguf_path)

    gate_t = up_t = down_t = None
    for t in reader.tensors:
        if f"blk.{layer}.ffn_gate_exps" in t.name: gate_t = t
        elif f"blk.{layer}.ffn_up_exps" in t.name:   up_t = t
        elif f"blk.{layer}.ffn_down_exps" in t.name: down_t = t
    if gate_t is None:
        sys.exit(f"Could not find expert tensors for layer {layer}")

    n_experts = int(gate_t.shape[-1])
    if verbose:
        print(f"Layer {layer}: {n_experts} experts, gate shape {tuple(gate_t.shape)} ({gate_t.tensor_type.name})")
        print(f"Loading + dequantizing all {n_experts} experts to FP32 (one-time cost ~{n_experts*3*0.7:.0f}s)...")

    t0 = time.perf_counter()
    experts = []
    for e in range(n_experts):
        g = get_expert_2d(gate_t, e)
        u = get_expert_2d(up_t, e)
        d = get_expert_2d(down_t, e)
        experts.append((g, u, d))
        if verbose and e == 0:
            print(f"  expert 0 shapes: gate {g.shape}, up {u.shape}, down {d.shape} (sanity check)")
    if verbose:
        total_gb = sum(g.nbytes + u.nbytes + d.nbytes for g, u, d in experts) / 1e9
        print(f"Loaded {n_experts} experts ({total_gb:.2f} GB FP32) in {time.perf_counter()-t0:.1f}s")
    return experts


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


# Shared state for the worker threads
STOP_FLAG = threading.Event()
GLOBAL_OP_COUNTER = 0
COUNTER_LOCK = threading.Lock()


def _ffn_step(hidden: np.ndarray, g: np.ndarray, u: np.ndarray, d: np.ndarray, H: int) -> np.ndarray:
    """One expert FFN. Handles both orientations of gate/up/down weight matrices."""
    # gate / up: produce intermediate from hidden. One axis matches H.
    gate_out = hidden @ g if g.shape[0] == H else g @ hidden
    up_out   = hidden @ u if u.shape[0] == H else u @ hidden
    fused = silu(gate_out) * up_out
    # down: produce hidden from intermediate. The H-axis is the output side.
    return d @ fused if d.shape[0] == H else fused @ d


def worker_loop(experts, hidden_dim: int, thread_id: int):
    """One worker thread: loops forever, rotating through experts and computing FFN."""
    global GLOBAL_OP_COUNTER
    n_experts = len(experts)
    # Each thread has its own input vector to avoid cache aliasing
    hidden = np.random.default_rng(thread_id).standard_normal(hidden_dim, dtype=np.float32)

    local_count = 0
    BATCH = 64

    try:
        while not STOP_FLAG.is_set():
            e = (thread_id * 7919 + local_count) % n_experts
            g, u, d = experts[e]
            _ = _ffn_step(hidden, g, u, d, hidden_dim)
            local_count += 1
            if local_count % BATCH == 0:
                with COUNTER_LOCK:
                    GLOBAL_OP_COUNTER += BATCH
    except Exception as ex:
        # Print loudly so we know if all threads silently died
        print(f"!!! Thread {thread_id} died: {type(ex).__name__}: {ex}", file=sys.stderr)
        STOP_FLAG.set()
        raise


def main():
    ap = argparse.ArgumentParser(description="Continuous CPU FFN load — run alongside the snapshot bench.")
    ap.add_argument("--gguf", required=True, help="Path to Q8 GGUF")
    ap.add_argument("--layer", type=int, default=1, help="MoE layer to load expert weights from")
    ap.add_argument("--threads", type=int, default=12, help="Worker threads (matches Legion 16-thread default minus overhead)")
    ap.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dim (DS-V2-Lite = 2048)")
    ap.add_argument("--report-secs", type=float, default=2.0, help="Print throughput every N seconds")
    ap.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = run forever until SIGINT)")
    args = ap.parse_args()

    experts = find_and_load_all_experts(args.gguf, args.layer)
    print(f"\nStarting {args.threads} CPU FFN worker threads. Each iteration: 1 expert FFN at hidden_dim={args.hidden_dim}.")
    print(f"This generator burns CPU continuously. Run the snapshot bench in another terminal and compare its t/s.")
    print(f"Ctrl+C to stop.\n")

    threads = []
    for tid in range(args.threads):
        t = threading.Thread(target=worker_loop, args=(experts, args.hidden_dim, tid), daemon=True)
        t.start()
        threads.append(t)

    start = time.perf_counter()
    last_report = start
    last_count = 0
    try:
        while True:
            time.sleep(args.report_secs)
            now = time.perf_counter()
            with COUNTER_LOCK:
                c = GLOBAL_OP_COUNTER
            ops_in_window = c - last_count
            elapsed = now - last_report
            total_elapsed = now - start
            ops_per_sec = ops_in_window / elapsed
            # Each "op" = one expert FFN (gate + up + silu + mul + down)
            # Memory traffic per op = ~34 MB FP32 (3 projections × 11.5 MB)
            bytes_per_sec_gb = ops_per_sec * 34.6e6 / 1e9
            print(f"[t={total_elapsed:5.1f}s] {ops_per_sec:.1f} FFN ops/s/total  "
                  f"({ops_per_sec/args.threads:.1f}/thread)  "
                  f"~{bytes_per_sec_gb:.1f} GB/s DRAM read")
            last_report = now
            last_count = c
            if args.duration > 0 and total_elapsed > args.duration:
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        STOP_FLAG.set()
        for t in threads:
            t.join(timeout=2.0)
        with COUNTER_LOCK:
            total_ops = GLOBAL_OP_COUNTER
        total_time = time.perf_counter() - start
        print(f"\n=== Summary ===")
        print(f"Total time:        {total_time:.1f}s")
        print(f"Total FFN ops:     {total_ops:,}")
        print(f"Average ops/sec:   {total_ops / total_time:.1f}")
        print(f"Average per thread: {total_ops / total_time / args.threads:.1f} ops/s")
        print(f"Sustained DRAM bw: {total_ops * 34.6e6 / total_time / 1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
