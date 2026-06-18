"""
split_user_history.py
---------------------
Deterministic train/test split of a user-history TSV (one query per line,
user_id<TAB>text). Random shuffle with --seed; first (1 - test_frac) → train,
rest → test.

Train set is used to build the per-(layer, expert) frequency profile that
warms the cache. Test set is the held-out demo queries where we measure
TTFT / short-decode latency with and without warm-from-profile.

Usage:
    python scripts/split_user_history.py \\
        --in  data/user_history/shreya_prompts.tsv \\
        --train-out data/user_history/shreya_prompts_train.tsv \\
        --test-out  data/user_history/shreya_prompts_test.tsv \\
        --test-frac 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--train-out", required=True, type=Path)
    ap.add_argument("--test-out", required=True, type=Path)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [l.rstrip("\n") for l in open(args.inp, encoding="utf-8") if l.strip()]
    n = len(rows)
    if n < 10:
        print(f"too few rows ({n}) to split", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    perm = list(range(n))
    rng.shuffle(perm)

    n_test = max(1, int(round(args.test_frac * n)))
    n_train = n - n_test
    test_idx = set(perm[:n_test])

    train_rows = [rows[i] for i in range(n) if i not in test_idx]
    test_rows  = [rows[i] for i in range(n) if i in test_idx]

    for out, body in [(args.train_out, train_rows), (args.test_out, test_rows)]:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for r in body:
                fh.write(r + "\n")

    print(f"seed={args.seed} test_frac={args.test_frac}", file=sys.stderr)
    print(f"  total : {n}", file=sys.stderr)
    print(f"  train : {n_train}  → {args.train_out}", file=sys.stderr)
    print(f"  test  : {n_test}  → {args.test_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
