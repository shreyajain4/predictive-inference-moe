#!/usr/bin/env python3
"""Compare two per-(layer, expert) frequency profiles built by
build_user_history_profile.py.

Reports per-layer Jaccard overlap of the top-K most-popular experts at
K=8,16,24,32. This is the gate for warm-from-history: if Jaccard@24 is high,
preloading train's top-24 hits roughly the same experts test prompts will
need. If it's low, the train profile doesn't predict test usage and warming
the cache from history is a dead end.

Usage:
  python3 compare_user_history_profiles.py \\
      --train data/user_history/shreya_train_profile.npy \\
      --test  data/user_history/shreya_test_profile.npy
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def topk_set(counts_row: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(counts_row)[::-1][:k].tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--ks", default="8,16,24,32,48",
                    help="comma-separated K values for Jaccard@K")
    args = ap.parse_args()

    train = np.load(args.train)
    test = np.load(args.test)
    if train.shape != test.shape:
        raise SystemExit(f"shape mismatch: train={train.shape} test={test.shape}")

    n_layers, n_experts = train.shape
    ks = [int(x) for x in args.ks.split(",")]

    print(f"train: {args.train.name}   shape {train.shape}")
    print(f"test:  {args.test.name}   shape {test.shape}")
    print(f"layers: {n_layers}  experts/layer: {n_experts}")
    print()

    header = "  " + f"{'layer':>5}  " + "  ".join(f"J@{k:<3}" for k in ks)
    print(header)
    per_k_means = {k: [] for k in ks}
    for L in range(n_layers):
        if train[L].sum() == 0 and test[L].sum() == 0:
            continue
        row = f"  {L:>5}  "
        for k in ks:
            j = jaccard(topk_set(train[L], k), topk_set(test[L], k))
            per_k_means[k].append(j)
            row += f"{j:>5.2f}  "
        if L == 0 or L == n_layers - 1 or L % 4 == 0:
            print(row)

    print()
    print("Mean Jaccard@K across all layers:")
    for k in ks:
        vals = per_k_means[k]
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        mn, mx = min(vals), max(vals)
        print(f"  K={k:>3}:  mean {mean:.2f}  median {med:.2f}  range {mn:.2f}..{mx:.2f}")

    print()
    j24_mean = float(np.mean(per_k_means[24])) if 24 in per_k_means else None
    if j24_mean is not None:
        if j24_mean >= 0.7:
            verdict = "strong overlap — warm-from-history should help"
        elif j24_mean >= 0.5:
            verdict = "moderate overlap — likely some benefit, worth measuring"
        elif j24_mean >= 0.3:
            verdict = "weak overlap — marginal benefit at best"
        else:
            verdict = "low overlap — train history does not predict test usage"
        print(f"verdict (using Jaccard@24): {verdict}")


if __name__ == "__main__":
    main()
