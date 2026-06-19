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
    # Try with header first, fall back to no-header (positional)
    df = pd.read_csv(args.csv, low_memory=False)
    cols0 = list(df.columns)
    print(f"  {len(df):,} rows; cols={cols0[:10]}...", flush=True)

    # If no recognized header (e.g., bench dumped no header line), reload with positional schema.
    if not any(str(c).startswith("expert") or c in ("query_id", "layer", "token_pos") for c in cols0):
        print("  no header detected — reloading with positional schema "
              "(user_id, query_id, layer, token_pos, expert_0..K-1)", flush=True)
        df = pd.read_csv(args.csv, header=None, low_memory=False)
        n_meta = 4  # user_id, query_id, layer, token_pos
        n_experts_csv = len(df.columns) - n_meta
        df.columns = ["user_id", "query_id", "layer", "token_pos"] + [f"expert_{i}" for i in range(n_experts_csv)]

    # Identify expert_* columns
    expert_cols = sorted([c for c in df.columns if str(c).startswith("expert_")],
                         key=lambda c: int(str(c).split("_")[1]))
    if not expert_cols:
        sys.exit(f"no expert_* columns found. Header was: {list(df.columns)}")
    print(f"  top_k = {len(expert_cols)} (cols {expert_cols[0]}..{expert_cols[-1]})", flush=True)

    # Build list[int] for `e`
    e_arr = df[expert_cols].to_numpy(dtype=np.int32)
    e_list = [row.tolist() for row in e_arr]

    # Tolerant column lookup
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    qcol = pick("query_id", "query", "qid")
    tcol = pick("token_pos", "tok", "pos")
    lcol = pick("layer", "l")
    tidcol = pick("token_id", "tid")
    if qcol is None or tcol is None or lcol is None:
        sys.exit(f"CSV missing required columns. Have: {list(df.columns)}")

    out_cols = {
        "p": df[qcol].astype(np.int32),
        "t": df[tcol].astype(np.int32),
        "l": df[lcol].astype(np.int32),
        "e": e_list,
    }
    if tidcol is not None:
        out_cols["tid"] = df[tidcol].astype(np.int32)
    else:
        print(f"  no token-id column; skipping tid (lr_governor_sweep treats it as optional unless --use-embeddings)", flush=True)
    out_df = pd.DataFrame(out_cols)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, compression="zstd", index=False)
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB, {len(out_df):,} rows)", flush=True)
    print(f"  prompts: {out_df['p'].nunique():,}", flush=True)
    print(f"  layers : {out_df['l'].min()}..{out_df['l'].max()}", flush=True)
    print(f"  tokens : avg {out_df.groupby('p')['t'].nunique().mean():.1f} per prompt", flush=True)


if __name__ == "__main__":
    main()
