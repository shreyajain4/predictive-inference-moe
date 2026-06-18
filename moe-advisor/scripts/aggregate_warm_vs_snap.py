#!/usr/bin/env python3
"""Aggregate the per-config TSV files produced by bench_warm_vs_snap_loop.sh.

Run after the bench loop finishes — point it at the outdir printed by the
script (e.g., /tmp/warm_vs_snap_NNN).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


PAT_HITS = re.compile(r"d2d_hits=(\d+)")
PAT_MISS = re.compile(r"d2d_misses=(\d+)")
PAT_D2D_MB = re.compile(r"bytes_d2d_served=([\d.]+)")
PAT_PRE_MB = re.compile(r"bytes_prefetched=([\d.]+)")
PAT_TPS = re.compile(r"\(([\d.]+) tok/s")


def aggregate(tsv: Path) -> None:
    if not tsv.exists():
        print(f"  {tsv.name}: missing")
        return
    runs_cache = 0
    runs_tps = 0
    h = m = 0
    d2d_mb = pre_mb = 0.0
    tps_vals = []
    for line in tsv.read_text().splitlines():
        if "d2d_hits=" in line:
            runs_cache += 1
            if (mat := PAT_HITS.search(line)): h += int(mat.group(1))
            if (mat := PAT_MISS.search(line)): m += int(mat.group(1))
            if (mat := PAT_D2D_MB.search(line)): d2d_mb += float(mat.group(1))
            if (mat := PAT_PRE_MB.search(line)): pre_mb += float(mat.group(1))
        if "tok/s" in line:
            if (mat := PAT_TPS.search(line)):
                tps_vals.append(float(mat.group(1)))
                runs_tps += 1

    if runs_cache:
        hit_rate = 100.0 * h / (h + m) if (h + m) else 0.0
        print(f"  cache: runs={runs_cache}  hits={h:,}  miss={m:,}  hit_rate={hit_rate:.2f}%")
        print(f"         bytes_prefetched={pre_mb:.0f} MiB  bytes_d2d_served={d2d_mb:.0f} MiB")
    else:
        print(f"  cache: (not used in this config)")
    if tps_vals:
        mean = sum(tps_vals) / len(tps_vals)
        tps_vals_sorted = sorted(tps_vals)
        median = tps_vals_sorted[len(tps_vals_sorted) // 2]
        print(f"  tok/s: runs={runs_tps}  mean={mean:.2f}  median={median:.2f}  "
              f"min={min(tps_vals):.2f}  max={max(tps_vals):.2f}")
    else:
        print(f"  tok/s: no runs parsed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path,
                    help="Directory containing per-config .tsv files (printed by the bench loop)")
    args = ap.parse_args()
    if not args.outdir.is_dir():
        sys.exit(f"not a directory: {args.outdir}")

    for cfg in ("snap_only", "snap_warm", "ngl12", "cpu_moe"):
        print(f"--- {cfg} ---")
        aggregate(args.outdir / f"{cfg}.tsv")


if __name__ == "__main__":
    main()
