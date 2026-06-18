#!/usr/bin/env python3
"""Predict the cache hit rate of warm-from-history.

For each routing decision in the test trace (token x layer x active expert),
check whether that expert is in the train top-K per-layer set. Reports the
fraction that would hit a cache preloaded with train's top-K, per layer and
overall. This is the strict upper bound the warm-from-history wiring can
deliver (real cache will be lower once eviction kicks in mid-decode).

Usage:
  python3 predict_warm_history_hit_rate.py \\
      --test-csv      data/user_history/shreya_test_routing.csv \\
      --train-profile data/user_history/shreya_train_profile.npy
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--train-profile", required=True, type=Path,
                    help=".npy from build_user_history_profile.py")
    ap.add_argument("--ks", default="8,16,24,32,48,64")
    args = ap.parse_args()

    train_counts = np.load(args.train_profile)
    n_layers, n_experts = train_counts.shape
    ks = [int(x) for x in args.ks.split(",")]

    top_at_k = {
        k: [set(np.argsort(train_counts[L])[::-1][:k].tolist())
            for L in range(n_layers)]
        for k in ks
    }

    hits = {k: np.zeros(n_layers, dtype=np.int64) for k in ks}
    totals = np.zeros(n_layers, dtype=np.int64)

    with args.test_csv.open("r", newline="") as f:
        for row in csv.reader(f):
            try:
                layer = int(row[2])
            except (ValueError, IndexError):
                continue
            experts = []
            for x in row[4:]:
                s = x.lstrip("-")
                if s.isdigit():
                    e = int(x)
                    if 0 <= e < n_experts:
                        experts.append(e)
            if not experts:
                continue
            totals[layer] += len(experts)
            for k in ks:
                in_top = top_at_k[k][layer]
                hits[k][layer] += sum(1 for e in experts if e in in_top)

    total_decisions = int(totals.sum())
    active_layers = int((totals > 0).sum())
    print(f"test routing decisions: {total_decisions:,} across {active_layers} layers")
    print()

    header = f"  {'layer':>5}  {'#dec':>8}  " + "  ".join(f"h@{k:<3}" for k in ks)
    print(header)
    for L in range(n_layers):
        if totals[L] == 0:
            continue
        line = f"  {L:>5}  {int(totals[L]):>8}  "
        for k in ks:
            r = hits[k][L] / totals[L]
            line += f"{r:>5.2f}  "
        if L == 0 or L == n_layers - 1 or L % 4 == 0:
            print(line)

    print()
    print("Overall predicted hit rate (hits / total decisions):")
    for k in ks:
        r = int(hits[k].sum()) / total_decisions
        cache_size = k * active_layers
        print(f"  K={k:>3}:  {r:.3f}   (cache holds {cache_size} expert slots)")


if __name__ == "__main__":
    main()
