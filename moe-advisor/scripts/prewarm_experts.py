#!/usr/bin/env python3
"""Pre-warm expert pages into OS page cache so subsequent mmap reads hit RAM
instead of SSD. Used to test the hypothesis that CPU-side warm-from-history
beats SSD-spilled ngl_auto on consumer hardware.

Reads the expert-offsets JSON produced by extract_expert_offsets.py, picks
the first K experts per MoE layer, and pread()'s their byte ranges from the
GGUF file(s). Handles split-GGUF (multi-part) files automatically.

Usage:
  python3 prewarm_experts.py \\
    --offsets /tmp/mixtral_iq4xs_offsets.json \\
    --k 4 \\
    [--parts /path/to/part2.gguf]
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", required=True, type=Path)
    ap.add_argument("--k", type=int, default=4,
                    help="experts per layer to warm (first K of N)")
    ap.add_argument("--parts", type=Path, nargs="*", default=[],
                    help="additional split-GGUF parts (in order). The primary file "
                         "comes from offsets JSON's gguf_path field.")
    args = ap.parse_args()

    with args.offsets.open() as f:
        offsets = json.load(f)

    primary = Path(offsets["gguf_path"])
    files = [primary] + [Path(p) for p in args.parts]

    # Compute logical file boundaries (concatenated split GGUF).
    part_sizes = [p.stat().st_size for p in files]
    cumulative = [0]
    for sz in part_sizes:
        cumulative.append(cumulative[-1] + sz)
    print(f"primary: {primary} ({part_sizes[0]/1e9:.1f} GB)")
    for i, p in enumerate(files[1:], 1):
        print(f"part {i+1}:  {p} ({part_sizes[i]/1e9:.1f} GB)")
    print(f"total:   {cumulative[-1]/1e9:.1f} GB across {len(files)} part(s)")
    print()

    fds = [os.open(str(p), os.O_RDONLY) for p in files]

    n_experts = offsets["num_experts"]
    k = min(args.k, n_experts)
    print(f"warming first {k} of {n_experts} experts per layer across {len(offsets['layers'])} MoE layers")
    print()

    bytes_read = 0
    pages_touched = 0
    layers_warmed = 0
    start = time.time()

    READ_CHUNK = 64 * 1024 * 1024  # 64 MB chunks for pread

    for layer_info in offsets["layers"]:
        L = layer_info["layer"]
        for kind in ("gate", "up", "down"):
            info = layer_info[kind]
            base = info["base_offset"]
            per_expert = info["per_expert_bytes"]
            for e in range(k):
                start_byte = base + e * per_expert
                end_byte = start_byte + per_expert
                # Figure out which file part this range is in
                file_idx = 0
                for i in range(len(cumulative) - 1):
                    if start_byte >= cumulative[i] and start_byte < cumulative[i + 1]:
                        file_idx = i
                        break
                local_offset = start_byte - cumulative[file_idx]
                # Read in chunks to avoid huge single allocs
                remaining = per_expert
                cur = local_offset
                while remaining > 0:
                    n = min(remaining, READ_CHUNK)
                    try:
                        data = os.pread(fds[file_idx], n, cur)
                    except OSError as ex:
                        print(f"  [warn] pread failed at part={file_idx} off={cur}: {ex}")
                        break
                    bytes_read += len(data)
                    # Touch the data so it's actually paged in (pread does this
                    # for us, but a checksum forces compiler/python not to
                    # optimize away the read).
                    if data:
                        _ = data[0] | data[-1]
                    cur += n
                    remaining -= n
                    if len(data) < n:
                        break
                pages_touched += (per_expert + 4095) // 4096
        layers_warmed += 1
        if layers_warmed % 10 == 0:
            elapsed = time.time() - start
            print(f"  ...{layers_warmed} layers, {bytes_read/1e9:.1f} GB read, {elapsed:.1f}s")

    for fd in fds:
        os.close(fd)

    elapsed = time.time() - start
    print()
    print(f"warmed {layers_warmed} layers × {k} experts × 3 kinds = "
          f"{layers_warmed * k * 3} (layer, expert, kind) ranges")
    print(f"bytes read: {bytes_read/1e9:.2f} GB")
    print(f"pages: {pages_touched:,} (~{pages_touched * 4 / 1024:.0f} MB worth)")
    print(f"time: {elapsed:.1f} s  ({bytes_read/1e9/elapsed:.2f} GB/s)")


if __name__ == "__main__":
    main()
