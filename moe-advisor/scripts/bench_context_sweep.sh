#!/bin/bash
# bench_context_sweep.sh
# Sweep n_ctx across mechanism variants on RTX 3070 + Qwen3-30B-A3B Q4.
# Tests the hypothesis: ngl=auto's lead shrinks as KV grows (KV displaces
# resident experts on GPU). At long context, snap variants may catch up.
# At small context (1024, 2048) ngl=auto fits nearly everything → should
# dominate. Cache size adapts per-context: bigger c → smaller cache (KV
# eats VRAM), smaller c → larger cache.
#
# Five configs per (prompt, context), paired:
#   A. ngl_auto         no -ngl, no cache
#   B. snap             snapshot cache only
#   C. snap_pred        snap + predictor prefetch-k=12
#   D. snap_warm        snap + warm-from-history (K=32)
#   E. snap_warm_pred   snap + warm + predictor
#
# Cache MB per context — baselined at 3500 MiB (proven safe with c=4096
# from earlier runs), shrunk linearly with extra KV beyond that. Floor 1500.
#   c=4096:  3500 MiB    c=16384:  2324 MiB
#   c=8192:  3108 MiB    c=32768:  1500 MiB (floor)
#                        c=65536:  1500 MiB (floor)
#
# Usage:
#   bash scripts/bench_context_sweep.sh [N_PROMPTS] "ctx1 ctx2 ..."
# Defaults: N_PROMPTS=1  contexts="4096 8192 16384 32768 65536"
#
# Single-prompt mode (default): 5 configs × 5 contexts = 25 runs ≈ 12 min.
# Noisy but quick — use for sanity / direction-finding.
# For real measurement, pass N=10 (≈ 2 hours).

set -e

N_PROMPTS=${1:-1}
CONTEXTS=${2:-"4096 8192 16384 32768 65536"}

: "${MODEL:=/home/shreya/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
: "${WEIGHTS:=/home/shreya/predictive-inference-moe/bench-bundle/predictor-weights/qwen3_predictor_weights.bin}"
: "${OFFSETS:=$HOME/qwen3_q4_expert_offsets.json}"
: "${WARM:=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_warm_k32.txt}"
: "${TESTSET:=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_prompts_test.tsv}"
: "${BENCH:=$HOME/llama.cpp/build/bin/llama-moe-predictor-bench}"
: "${N_TOKENS:=8}"

# Cache size baselined at 3500 MiB (known-safe with c=4096 from earlier
# runs), reduced linearly with extra KV beyond c=4096. Floored at 1500.
# Tried going larger at small c before (5500) — OOM'd because lm_head +
# embed + per-layer attention + driver overhead ate more than I assumed.
cache_mb_for_ctx() {
  local ctx=$1
  local kv_mb=$(( ctx * 98 / 1024 ))
  local baseline_kv=$(( 4096 * 98 / 1024 ))   # 392 MiB at c=4096
  local cache=$(( 3500 - (kv_mb - baseline_kv) ))
  if [ $cache -lt 1500 ]; then cache=1500; fi
  if [ $cache -gt 3500 ]; then cache=3500; fi
  echo $cache
}

OUTDIR=/tmp/ctx_sweep_$(date +%s)
mkdir -p "$OUTDIR"
echo "outdir: $OUTDIR  contexts: $CONTEXTS  prompts: $N_PROMPTS"
echo "cache MB per context:"
for ctx in $CONTEXTS; do
  echo "  c=$ctx  cache_mb=$(cache_mb_for_ctx $ctx)"
done

run_one() {
  local cfg="$1" ctx="$2" prompt="$3" log="$4"
  local cache_mb=$(cache_mb_for_ctx "$ctx")
  case "$cfg" in
    ngl_auto)
      "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        --predictor-weights "$WEIGHTS" --prefetch-k 0 \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
    snap)
      GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        -ngl 99 --override-tensor exps=CPU \
        --expert-offsets "$OFFSETS" --expert-cache-mb "$cache_mb" \
        --predictor-weights "$WEIGHTS" --prefetch-k 0 \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
    snap_pred)
      GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        -ngl 99 --override-tensor exps=CPU \
        --expert-offsets "$OFFSETS" --expert-cache-mb "$cache_mb" \
        --predictor-weights "$WEIGHTS" --prefetch-k 12 \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
    snap_warm)
      GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        -ngl 99 --override-tensor exps=CPU \
        --expert-offsets "$OFFSETS" --expert-cache-mb "$cache_mb" \
        --predictor-weights "$WEIGHTS" --prefetch-k 0 \
        --warm-snapshot-profile "$WARM" \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
    snap_warm_pred)
      GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        -ngl 99 --override-tensor exps=CPU \
        --expert-offsets "$OFFSETS" --expert-cache-mb "$cache_mb" \
        --predictor-weights "$WEIGHTS" --prefetch-k 12 \
        --warm-snapshot-profile "$WARM" \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
  esac
  stats=$(grep -m1 "moe_cuda_expert_cache final" "$log" 2>/dev/null || echo "no-cache")
  tps=$(grep -oE "decode:.*tok/s" "$log" 2>/dev/null | tail -1 || echo "no-tps")
  printf "%s\t%s\t%s\n" "${cfg}_c${ctx}_mb${cache_mb}" "$stats" "$tps"
}

CONFIGS="ngl_auto snap snap_pred snap_warm snap_warm_pred"

for ctx in $CONTEXTS; do
  for cfg in $CONFIGS; do
    : > "$OUTDIR/${cfg}_c${ctx}.tsv"
  done
done

# Pick prompt(s):
#   PROMPT_TEXT="..."       → use this literal text (highest priority)
#   PROMPT_LINES="3 7 12"   → use these TSV line numbers
#   N_PROMPTS=1 (default)   → longest prompt that fits in CTX_MIN's prefill
#                             budget (75% of n_ctx × 3.5 chars/tok)
#   N_PROMPTS=N             → first N lines (paired aggregate)
if [ -n "$PROMPT_TEXT" ]; then
  echo "using PROMPT_TEXT (len=${#PROMPT_TEXT} chars)"
  i=1
  for ctx in $CONTEXTS; do
    for cfg in $CONFIGS; do
      log="$OUTDIR/${cfg}_c${ctx}_${i}.log"
      run_one "$cfg" "$ctx" "$PROMPT_TEXT" "$log" >> "$OUTDIR/${cfg}_c${ctx}.tsv"
    done
  done
else
  if [ -n "$PROMPT_LINES" ]; then
    LINES="$PROMPT_LINES"
    echo "using PROMPT_LINES=$LINES"
  elif [ "$N_PROMPTS" = "1" ]; then
    CTX_MIN=$(echo "$CONTEXTS" | tr ' ' '\n' | sort -n | head -1)
    CHAR_LIMIT=$(( CTX_MIN * 75 / 100 * 35 / 10 ))
    LINES=$(awk -F'\t' -v lim="$CHAR_LIMIT" '
      length($2) <= lim { print length($2)"\t"NR }
    ' "$TESTSET" | sort -rn | head -1 | cut -f2)
    if [ -z "$LINES" ]; then
      echo "ERROR: no prompt fits in CHAR_LIMIT=$CHAR_LIMIT (CTX_MIN=$CTX_MIN)"
      exit 1
    fi
    PLEN=$(awk -F'\t' -v ln=$LINES 'NR==ln {print length($2)}' "$TESTSET")
    echo "single-prompt mode: picked line $LINES (length=$PLEN chars, limit=$CHAR_LIMIT for CTX_MIN=$CTX_MIN)"
  else
    LINES=$(seq 1 "$N_PROMPTS")
  fi

  i=0
  for line_no in $LINES; do
    prompt=$(awk -F'\t' -v ln=$line_no 'NR==ln {print $2}' "$TESTSET")
    [ -z "$prompt" ] && continue
    i=$((i+1))
    echo "[$i] line=$line_no len=${#prompt}"
    for ctx in $CONTEXTS; do
      for cfg in $CONFIGS; do
        log="$OUTDIR/${cfg}_c${ctx}_${i}.log"
        run_one "$cfg" "$ctx" "$prompt" "$log" >> "$OUTDIR/${cfg}_c${ctx}.tsv"
      done
    done
  done
fi

echo ""
echo "=== AGGREGATES ==="
python3 - << PYEOF
import re
from pathlib import Path
outdir = Path("$OUTDIR")
contexts = "$CONTEXTS".split()
configs = "$CONFIGS".split()

PAT_TPS = re.compile(r"\(([\d.]+) tok/s")
PAT_HITS = re.compile(r"d2d_hits=(\d+)")
PAT_MISS = re.compile(r"d2d_misses=(\d+)")

print()
print("mean tok/s")
print(f"{'config':<18} " + "  ".join(f"c={c:>5}" for c in contexts))
print("-" * (18 + len(contexts) * 11))
for cfg in configs:
    row = f"{cfg:<18} "
    for ctx in contexts:
        tsv = outdir / f"{cfg}_c{ctx}.tsv"
        if not tsv.exists():
            row += "  -      "; continue
        tps = [float(m.group(1)) for line in tsv.read_text().splitlines()
               for m in [PAT_TPS.search(line)] if m]
        if tps:
            row += f"{sum(tps)/len(tps):>6.2f}   "
        else:
            row += "  -      "
    print(row)

print()
print("cache d2d hit rate (snap variants only)")
print(f"{'config':<18} " + "  ".join(f"c={c:>5}" for c in contexts))
print("-" * (18 + len(contexts) * 11))
for cfg in configs:
    if cfg == "ngl_auto": continue
    row = f"{cfg:<18} "
    for ctx in contexts:
        tsv = outdir / f"{cfg}_c{ctx}.tsv"
        if not tsv.exists():
            row += "  -      "; continue
        h = m = 0
        for line in tsv.read_text().splitlines():
            hm = PAT_HITS.search(line); mm = PAT_MISS.search(line)
            if hm: h += int(hm.group(1))
            if mm: m += int(mm.group(1))
        if h + m > 0:
            row += f"{100*h/(h+m):>5.1f}%   "
        else:
            row += "  -      "
    print(row)
PYEOF
echo ""
echo "raw logs under: $OUTDIR"
