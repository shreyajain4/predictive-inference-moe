#!/usr/bin/env python3
"""Convert per-(layer, expert) count .npy to a flat text warm profile.

Output format (one entry per line, sorted by layer asc, then by hits desc):
  <layer> <expert_id> <hits>
Lines starting with '#' are comments and ignored by the C++ loader.

C++ side reads this file at startup, then calls cache.pin() for each line.

Usage:
  python3 npy_to_warm_profile.py \\
      --profile data/user_history/shreya_train_profile.npy \\
      --k 32 \\
      --out data/user_history/shreya_warm_k32.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, type=Path,
                    help=".npy count matrix from build_user_history_profile.py")
    ap.add_argument("--k", type=int, required=True,
                    help="experts per layer to include in the warm profile")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    counts = np.load(args.profile)
    n_layers, n_experts = counts.shape
    if args.k > n_experts:
        raise SystemExit(f"k={args.k} exceeds n_experts={n_experts}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    skipped_empty = 0
    with args.out.open("w") as f:
        f.write(f"# warm-from-history profile  layers={n_layers}  k_per_layer={args.k}\n")
        f.write(f"# source={args.profile.name}\n")
        for L in range(n_layers):
            if int(counts[L].sum()) == 0:
                skipped_empty += 1
                continue
            top = np.argsort(counts[L])[::-1][:args.k]
            for e in top:
                hits = int(counts[L, e])
                if hits == 0:
                    continue
                f.write(f"{L} {int(e)} {hits}\n")
                n_written += 1

    print(f"wrote {n_written} entries to {args.out}")
    if skipped_empty:
        print(f"  skipped {skipped_empty} layers with zero routing data (likely dense layers)")
    print(f"  cache must have >= {n_written} slots to fit all pinned entries")


if __name__ == "__main__":
    main()
