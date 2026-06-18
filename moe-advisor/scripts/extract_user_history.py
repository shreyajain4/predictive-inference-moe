"""
extract_user_history.py
-----------------------
Extract user-typed prompts from Claude Code .jsonl transcripts and emit a
TSV (user_id<TAB>query) in the format llama-moe-trace -f consumes.

This produces the "user history" corpus for the profile-warmed cache demo.

Usage:
    python scripts/extract_user_history.py \\
        --transcripts-glob "/Users/shreyajain/.claude/projects/-Users-shreyajain-Documents-predictive-inference-moe*/*.jsonl" \\
        --out data/user_history/shreya_prompts.tsv

Filters applied (in order):
  - keep only msg.type == "user" with msg.message.role == "user"
  - drop messages whose content starts with "<" (system reminders / tags)
  - drop "[Request interrupted by user]" and similar control lines
  - drop messages shorter than --min-chars
  - dedup exact-duplicate text within a transcript
  - dedup across transcripts (keep first occurrence)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


JUNK_PREFIXES = (
    "<",                                  # system reminders, tool tags
    "[Request interrupted by user]",
    "tool_use_id",
)


def extract_user_text(msg: dict) -> str | None:
    if msg.get("type") != "user":
        return None
    m = msg.get("message")
    if not isinstance(m, dict):
        return None
    if m.get("role") != "user":
        return None
    content = m.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            t = blk.get("type")
            if t == "text":
                txt = blk.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
            # tool_result blocks are NOT user-typed text — skip
        return "\n".join(parts) if parts else None
    return None


def clean(text: str) -> str:
    """Collapse internal newlines/tabs so the line survives TSV write+read."""
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def is_junk(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return True
    if any(text.startswith(p) for p in JUNK_PREFIXES):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts-glob", required=True,
                    help="glob pattern matching .jsonl transcript files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-chars", type=int, default=10,
                    help="drop prompts shorter than this (after clean)")
    ap.add_argument("--user-id", default="shreya",
                    help="value to write in the first TSV column")
    args = ap.parse_args()

    files = sorted(glob.glob(args.transcripts_glob))
    if not files:
        print(f"no files matched: {args.transcripts_glob}", file=sys.stderr)
        return 1

    seen: set[str] = set()
    rows: list[str] = []
    n_total = 0
    n_kept = 0
    n_dup = 0
    n_junk = 0
    chars_kept = 0

    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                text = extract_user_text(msg)
                if text is None:
                    continue
                n_total += 1
                text = clean(text)
                if is_junk(text, args.min_chars):
                    n_junk += 1
                    continue
                if text in seen:
                    n_dup += 1
                    continue
                seen.add(text)
                rows.append(text)
                n_kept += 1
                chars_kept += len(text)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(f"{args.user_id}\t{r}\n")

    print(f"transcripts scanned: {len(files)}", file=sys.stderr)
    print(f"user messages seen : {n_total:,}", file=sys.stderr)
    print(f"  dropped junk     : {n_junk:,}", file=sys.stderr)
    print(f"  dropped dup      : {n_dup:,}", file=sys.stderr)
    print(f"kept                : {n_kept:,}", file=sys.stderr)
    print(f"chars kept          : {chars_kept/1024:.0f} KB", file=sys.stderr)
    print(f"avg prompt length   : {chars_kept/max(n_kept,1):.0f} chars", file=sys.stderr)
    print(f"output              : {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
