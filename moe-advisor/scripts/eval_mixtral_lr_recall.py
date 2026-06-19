#!/usr/bin/env python3
"""Evaluate the recall@k of the trained Mixtral LR predictor on held-out tokens.

Loads the .pt checkpoint from lr_governor_sweep.py + the parquet trace, re-does
the 85/15 token-level split, and for each held-out (prompt, token, layer)
position computes:
  predicted_top_k = top-k argmax of LR(features)
  recall@k = |predicted_top_k ∩ actual_top_K_used| / K_used

Reports overall + per-layer for k in {2, 4, 6, 8}.

Random baseline for Mixtral (8 experts, top-2 actual):
  recall@2 = 0.250   recall@6 = 0.750
  recall@4 = 0.500   recall@8 = 1.000

Usage:
  python3 eval_mixtral_lr_recall.py \\
    --ckpt /tmp/mixtral_lr_train_XXX/mixtral_lr.pt \\
    --trace /tmp/mixtral_lr_train_XXX/routing.parquet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--train-frac", type=float, default=0.85)
    ap.add_argument("--ks", default="2,4,6,8")
    args = ap.parse_args()

    print(f"loading checkpoint: {args.ckpt}", flush=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    predictors = state["predictors"]
    config = state.get("config", {})
    n_experts = config.get("num_experts", config.get("n_experts", 8))
    n_used = config.get("n_expert_used", config.get("top_k", 2))
    print(f"  config: n_experts={n_experts}, n_used={n_used}, predictors={len(predictors)} layers")

    print(f"loading parquet: {args.trace}", flush=True)
    df = pd.read_parquet(args.trace, columns=["p", "t", "l", "e"])
    print(f"  {len(df):,} rows")

    df = df.sort_values(["p", "t", "l"]).reset_index(drop=True)
    layers = sorted(df["l"].unique().tolist())
    print(f"  layers: {layers[0]}..{layers[-1]} ({len(layers)} unique)")

    # Group per prompt
    prompts = sorted(df["p"].unique().tolist())
    n_train = int(args.train_frac * 1000) / 1000.0

    ks = [int(x) for x in args.ks.split(",")]
    # Per-layer accumulators: hits / total per k
    per_layer_hits = {k: {L: 0 for L in layers} for k in ks}
    per_layer_total = {L: 0 for L in layers}

    n_skipped = 0
    n_evaluated = 0

    # Build a lookup: per (prompt, layer) → ordered list of (token, experts) for prev-token feature
    for p in prompts:
        sub = df[df["p"] == p]
        tokens_in_p = sorted(sub["t"].unique().tolist())
        if len(tokens_in_p) < 2:
            continue
        cut = int(args.train_frac * len(tokens_in_p))
        eval_tokens = set(tokens_in_p[cut:])

        # Build (t, L) → expert list
        routing = {}
        for _, row in sub.iterrows():
            routing[(int(row["t"]), int(row["l"]))] = list(row["e"])

        for ti_idx, t in enumerate(tokens_in_p):
            if t not in eval_tokens:
                continue
            if ti_idx == 0:
                continue  # no prev-token to build feature from
            t_prev = tokens_in_p[ti_idx - 1]

            for L in layers:
                actual = routing.get((t, L))
                if actual is None:
                    continue

                # feat_A = multihot of prev-token's same-layer experts
                feat_A = np.zeros(n_experts, dtype=np.float32)
                prev_same = routing.get((t_prev, L))
                if prev_same is not None:
                    for e in prev_same:
                        if 0 <= int(e) < n_experts:
                            feat_A[int(e)] = 1.0

                # feat_B = multihot of current-token's prev-layer experts (or prev-token last-layer if L=first)
                feat_B = np.zeros(n_experts, dtype=np.float32)
                if L == layers[0]:
                    last_L = layers[-1]
                    src = routing.get((t_prev, last_L))
                else:
                    L_prev = layers[layers.index(L) - 1]
                    src = routing.get((t, L_prev))
                if src is not None:
                    for e in src:
                        if 0 <= int(e) < n_experts:
                            feat_B[int(e)] = 1.0

                if L not in predictors:
                    n_skipped += 1
                    continue
                pred_model = predictors[L]
                features = np.concatenate([feat_A, feat_B])
                with torch.no_grad():
                    x = torch.from_numpy(features).unsqueeze(0)
                    logits = pred_model(x).squeeze(0).numpy()

                actual_set = {int(e) for e in actual if int(e) >= 0}
                for k in ks:
                    top_k_preds = set(np.argsort(logits)[-k:].tolist())
                    hits = len(top_k_preds & actual_set)
                    per_layer_hits[k][L] += hits

                per_layer_total[L] += len(actual_set)
                n_evaluated += 1

    print(f"\nevaluated {n_evaluated:,} (token, layer) positions; {n_skipped} skipped (no predictor for layer)")
    print(f"\nOverall recall@k (Mixtral 8 experts × top-2; random@k = k/8):")
    print(f"  {'k':>3}  {'recall':>7}  {'random':>7}  {'lift':>6}")
    total_hits = {k: sum(per_layer_hits[k].values()) for k in ks}
    total_actuals = sum(per_layer_total.values())
    for k in ks:
        recall = total_hits[k] / total_actuals if total_actuals > 0 else 0.0
        rand = k / n_experts
        lift = recall / rand if rand > 0 else float("inf")
        print(f"  {k:>3}  {recall:>6.3f}   {rand:>6.3f}   {lift:>5.2f}×")

    print(f"\nPer-layer recall@4 (every 8th layer):")
    print(f"  {'layer':>5}  {'total':>6}  {'recall@4':>9}")
    for L in layers:
        if L % 8 != 0 and L != layers[-1]:
            continue
        tot = per_layer_total[L]
        if tot == 0:
            continue
        r = per_layer_hits[4][L] / tot
        print(f"  {L:>5}  {tot:>6}  {r:>8.3f}")


if __name__ == "__main__":
    main()
