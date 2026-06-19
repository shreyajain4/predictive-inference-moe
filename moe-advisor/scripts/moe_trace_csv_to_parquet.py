#!/usr/bin/env python3
"""Convert llama-moe-trace CSV to lr_governor_sweep parquet format.

llama-moe-trace CSV columns:
  user_id, query_id, layer, token_pos, token_id, expert_0..expert_{K-1}, logit_*, weight_*

Parquet schema lr_governor_sweep.load_packed() expects:
  p (int), t (int), l (int), e (list[int]), tid (int)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output parquet path")
    args = ap.parse_args()

    print(f"reading {args.csv} ...", flush=True)
    df = pd.read_csv(args.csv)
    print(f"  {len(df):,} rows; cols={list(df.columns)[:8]}...", flush=True)

    # Identify expert_* columns
    expert_cols = sorted([c for c in df.columns if c.startswith("expert_")],
                         key=lambda c: int(c.split("_")[1]))
    if not expert_cols:
        sys.exit("no expert_* columns found")
    print(f"  top_k = {len(expert_cols)} (cols {expert_cols[0]}..{expert_cols[-1]})", flush=True)

    # Build list[int] for `e`
    e_arr = df[expert_cols].to_numpy(dtype=np.int32)
    e_list = [row.tolist() for row in e_arr]

    out_df = pd.DataFrame({
        "p": df["query_id"].astype(np.int32),
        "t": df["token_pos"].astype(np.int32),
        "l": df["layer"].astype(np.int32),
        "e": e_list,
        "tid": df["token_id"].astype(np.int32),
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, compression="zstd", index=False)
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB, {len(out_df):,} rows)", flush=True)
    print(f"  prompts: {out_df['p'].nunique():,}", flush=True)
    print(f"  layers : {out_df['l'].min()}..{out_df['l'].max()}", flush=True)
    print(f"  tokens : avg {out_df.groupby('p')['t'].nunique().mean():.1f} per prompt", flush=True)


if __name__ == "__main__":
    main()
