#!/bin/bash
# bench_context_sweep.sh
# Sweep n_ctx ∈ {4096, 8192, 16384, 32768} to test the hypothesis:
# ngl=auto's lead shrinks as KV grows (because KV displaces resident
# experts on GPU). At long context, snap+warm and predictor variants
# may catch up.
#
# Three configs per prompt per context, paired:
#   A. ngl_auto      no -ngl flag (common_fit_params picks)
#   B. snap_warm     forced offload + snapshot cache + warm-from-history
#   C. snap_warm_pred  B + predictor prefetch-k=12
#
# Usage:
#   bash scripts/bench_context_sweep.sh [N_PROMPTS] "ctx1 ctx2 ..."
# Defaults: N_PROMPTS=15  contexts="4096 8192 16384 32768"
#
# Time: 3 configs × N prompts × 4 contexts × ~25 s/launch
# 15 × 12 = 180 runs ≈ 75 min.

set -e

N_PROMPTS=${1:-15}
CONTEXTS=${2:-"4096 8192 16384 32768"}

: "${MODEL:=/home/shreya/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
: "${WEIGHTS:=/home/shreya/predictive-inference-moe/bench-bundle/predictor-weights/qwen3_predictor_weights.bin}"
: "${OFFSETS:=$HOME/qwen3_q4_expert_offsets.json}"
: "${WARM:=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_warm_k32.txt}"
: "${TESTSET:=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_prompts_test.tsv}"
: "${BENCH:=$HOME/llama.cpp/build/bin/llama-moe-predictor-bench}"
: "${N_TOKENS:=8}"

OUTDIR=/tmp/ctx_sweep_$(date +%s)
mkdir -p "$OUTDIR"
echo "outdir: $OUTDIR  contexts: $CONTEXTS  prompts: $N_PROMPTS"

run_one() {
  local cfg="$1" ctx="$2" prompt="$3" log="$4"
  case "$cfg" in
    ngl_auto)
      "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        --predictor-weights "$WEIGHTS" --prefetch-k 0 \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
    snap_warm)
      GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        -ngl 99 --override-tensor exps=CPU \
        --predictor-weights "$WEIGHTS" --prefetch-k 0 \
        --expert-offsets "$OFFSETS" --expert-cache-mb 3500 \
        --warm-snapshot-profile "$WARM" \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
    snap_warm_pred)
      GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" -m "$MODEL" -c "$ctx" -b "$ctx" -ub 1024 -n "$N_TOKENS" \
        -ngl 99 --override-tensor exps=CPU \
        --predictor-weights "$WEIGHTS" --prefetch-k 12 \
        --expert-offsets "$OFFSETS" --expert-cache-mb 3500 \
        --warm-snapshot-profile "$WARM" \
        -p "$prompt" > "$log" 2>&1 || true
      ;;
  esac
  stats=$(grep -m1 "moe_cuda_expert_cache final" "$log" 2>/dev/null || echo "no-cache")
  tps=$(grep -oE "decode:.*tok/s" "$log" 2>/dev/null | tail -1 || echo "no-tps")
  printf "%s\t%s\t%s\n" "${cfg}_c${ctx}" "$stats" "$tps"
}

for ctx in $CONTEXTS; do
  for cfg in ngl_auto snap_warm snap_warm_pred; do
    : > "$OUTDIR/${cfg}_c${ctx}.tsv"
  done
done

i=0
while IFS=$'\t' read -r user prompt; do
  [ -z "$prompt" ] && continue
  i=$((i+1))
  [ $i -gt "$N_PROMPTS" ] && break
  echo "[$i/$N_PROMPTS]"
  for ctx in $CONTEXTS; do
    for cfg in ngl_auto snap_warm snap_warm_pred; do
      log="$OUTDIR/${cfg}_c${ctx}_${i}.log"
      run_one "$cfg" "$ctx" "$prompt" "$log" >> "$OUTDIR/${cfg}_c${ctx}.tsv"
    done
  done
done < "$TESTSET"

echo ""
echo "=== AGGREGATES ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 - << PYEOF
import re, os
from pathlib import Path
outdir = Path("$OUTDIR")
contexts = "$CONTEXTS".split()
configs = ["ngl_auto", "snap_warm", "snap_warm_pred"]

PAT_TPS = re.compile(r"\(([\d.]+) tok/s")
PAT_HITS = re.compile(r"d2d_hits=(\d+)")
PAT_MISS = re.compile(r"d2d_misses=(\d+)")

print(f"{'config':<18} " + "  ".join(f"c={c:>5}" for c in contexts))
print("-" * (18 + len(contexts) * 11))
for cfg in configs:
    row = f"{cfg:<18} "
    for ctx in contexts:
        tsv = outdir / f"{cfg}_c{ctx}.tsv"
        if not tsv.exists():
            row += f"{'-':>8}  "
            continue
        tps_vals = []
        for line in tsv.read_text().splitlines():
            m = PAT_TPS.search(line)
            if m:
                tps_vals.append(float(m.group(1)))
        if tps_vals:
            mean = sum(tps_vals) / len(tps_vals)
            row += f"{mean:>6.2f}    "
        else:
            row += f"{'-':>8}  "
    print(row)

print()
print("cache hit rates (snap_warm variants only):")
for cfg in ["snap_warm", "snap_warm_pred"]:
    row = f"{cfg:<18} "
    for ctx in contexts:
        tsv = outdir / f"{cfg}_c{ctx}.tsv"
        if not tsv.exists():
            row += f"{'-':>8}  "
            continue
        h = m = 0
        for line in tsv.read_text().splitlines():
            hm = PAT_HITS.search(line); mm = PAT_MISS.search(line)
            if hm: h += int(hm.group(1))
            if mm: m += int(mm.group(1))
        if h + m > 0:
            hr = 100 * h / (h + m)
            row += f"{hr:>5.1f}%    "
        else:
            row += f"{'-':>8}  "
    print(row)
PYEOF
echo ""
echo "raw logs under: $OUTDIR"
