#!/usr/bin/env python3
"""Build per-(layer, expert) frequency profile from a llama-moe-trace CSV.

Counts expert selections per layer across the user's prompt history, then emits:
  <out>.npy   (n_layers, n_experts) int64 count matrix
  <out>.json  per-layer ranked list of (expert_id, hits)

Prints a concentration summary so you can decide if warm-from-history is worth
wiring up: if the median layer needs ~20/128 experts to cover 80% of routing,
the win is real; if it's near-uniform, it isn't.

Schema assumed: user_id, query_id, layer, token_pos, expert_0..expert_{K-1}
(matches the moe-trace build currently emitting 12 cols for Qwen3 top-8).
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output stem (no extension)")
    ap.add_argument("--n-experts", type=int, default=128)
    args = ap.parse_args()

    counts: dict[int, Counter] = {}
    total = 0

    with args.csv.open("r", newline="") as f:
        for row in csv.reader(f):
            try:
                layer = int(row[2])
            except (ValueError, IndexError):
                continue
            experts = [int(x) for x in row[4:] if x.lstrip("-").isdigit()]
            counts.setdefault(layer, Counter()).update(experts)
            total += 1

    layers = sorted(counts)
    n_layers = layers[-1] + 1
    mat = np.zeros((n_layers, args.n_experts), dtype=np.int64)
    for L in layers:
        for e, c in counts[L].items():
            if 0 <= e < args.n_experts:
                mat[L, e] = c

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.with_suffix(".npy"), mat)
    json_out = {str(L): [(int(e), int(c)) for e, c in counts[L].most_common()] for L in layers}
    with args.out.with_suffix(".json").open("w") as f:
        json.dump(json_out, f, indent=2)

    print(f"rows: {total:,}  layers present: {len(layers)} ({layers[0]}..{layers[-1]})  experts/layer: {args.n_experts}")
    print(f"wrote {args.out.with_suffix('.npy')}")
    print(f"wrote {args.out.with_suffix('.json')}")
    print()
    print("Per-layer concentration (every 4th layer + endpoints):")
    print(f"  {'layer':>5}  {'top1':>6}  {'#@50%':>5}  {'#@80%':>5}  {'#@90%':>5}")
    cov80_all, cov50_all = [], []
    for L in layers:
        c = mat[L]
        s = int(c.sum())
        if s == 0:
            continue
        sc = np.sort(c)[::-1]
        cum = np.cumsum(sc) / s
        top1 = sc[0] / s * 100
        c50 = int((cum < 0.5).sum()) + 1
        c80 = int((cum < 0.8).sum()) + 1
        c90 = int((cum < 0.9).sum()) + 1
        cov50_all.append(c50)
        cov80_all.append(c80)
        if L == layers[0] or L == layers[-1] or L % 4 == 0:
            print(f"  {L:>5}  {top1:>5.1f}%  {c50:>5}  {c80:>5}  {c90:>5}")

    print()
    print(f"Across {len(cov80_all)} layers:")
    print(f"  experts needed to cover 50% of routing: median {int(np.median(cov50_all))}, range {min(cov50_all)}..{max(cov50_all)}")
    print(f"  experts needed to cover 80% of routing: median {int(np.median(cov80_all))}, range {min(cov80_all)}..{max(cov80_all)}")
    uniform_80 = int(0.8 * args.n_experts)
    med80 = int(np.median(cov80_all))
    if med80 < uniform_80 * 0.5:
        verdict = "concentrated — warm-from-history likely helps"
    elif med80 < uniform_80 * 0.8:
        verdict = "moderately concentrated — marginal win"
    else:
        verdict = "near-uniform — warm-from-history unlikely to help"
    print(f"  verdict: {verdict} (uniform would need {uniform_80}/{args.n_experts} for 80%)")


if __name__ == "__main__":
    main()
