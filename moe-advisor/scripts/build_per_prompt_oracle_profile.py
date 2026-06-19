#!/usr/bin/env python3
"""Build a per-prompt oracle warm profile from a routing parquet.

For a specific prompt (--prompt-id), aggregate the top-K most-frequent
experts per MoE layer that actually fired during that prompt. This is
the BEST POSSIBLE static prewarm — we're literally using ground-truth
top-K experts for the test prompt. If even this doesn't beat ngl_auto,
no realistic predictor can.

Output: flat warm profile (layer expert hits per line) compatible with
prewarm_experts.py.

Usage:
  python3 build_per_prompt_oracle_profile.py \\
    --trace /tmp/mixtral_lr_train_XXX/routing.parquet \\
    --prompt-id 0 \\
    --k 4 \\
    --out /tmp/mixtral_oracle_prompt0.txt
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--prompt-id", type=int, default=0,
                    help="which prompt's routing to oracle (default: first)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.trace, columns=["p", "t", "l", "e"])
    print(f"trace: {len(df):,} rows total")

    sub = df[df["p"] == args.prompt_id]
    if len(sub) == 0:
        sys.exit(f"prompt-id={args.prompt_id} not in parquet; have {sorted(df['p'].unique())[:10]}")
    print(f"prompt {args.prompt_id}: {len(sub):,} routing rows, "
          f"{sub['t'].nunique()} tokens, {sub['l'].nunique()} layers")

    # Group by layer; aggregate expert hit counts
    layers = sorted(sub["l"].unique().tolist())
    n_written = 0
    with args.out.open("w") as f:
        f.write(f"# oracle profile for prompt {args.prompt_id}, top-{args.k} per layer\n")
        f.write(f"# trace: {args.trace}\n")
        for L in layers:
            counts: Counter = Counter()
            for _, row in sub[sub["l"] == L].iterrows():
                for e in row["e"]:
                    if int(e) >= 0:
                        counts[int(e)] += 1
            for expert, hits in counts.most_common(args.k):
                f.write(f"{L} {expert} {hits}\n")
                n_written += 1

    print(f"wrote {args.out}: {n_written} (layer, expert) entries "
          f"({len(layers)} layers × up to {args.k})")


if __name__ == "__main__":
    main()
