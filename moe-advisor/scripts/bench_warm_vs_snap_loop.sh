#!/bin/bash
# bench_warm_vs_snap_loop.sh
# Aggregate warm-from-history measurement on RTX 3070 + Qwen3-30B-A3B Q4.
# Runs four configs per prompt over N test prompts, paired (same prompt
# under each config so per-prompt variance is controlled):
#
#   A. snap_only       forced-offload, snapshot cache only
#   B. snap_warm       forced-offload, snapshot + warm-from-history
#   C. ngl12           partial offload (12 layers on GPU), no cache (= the bar to beat)
#   D. cpu_moe         MoE stays on CPU (no GGML_OP_OFFLOAD_MIN_BATCH=1, no cache)
#
# Each launch reloads the model (~5 s). With -n 8 and 30 prompts: ~30 min.
#
# Usage:
#   bash scripts/bench_warm_vs_snap_loop.sh [N_PROMPTS] [N_TOKENS]
# Defaults: N_PROMPTS=30, N_TOKENS=8

set -e

N_PROMPTS=${1:-30}
N_TOKENS=${2:-8}

: "${MODEL:=/home/shreya/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
: "${WEIGHTS:=/home/shreya/predictive-inference-moe/bench-bundle/predictor-weights/qwen3_predictor_weights.bin}"
: "${OFFSETS:=$HOME/qwen3_q4_expert_offsets.json}"
: "${WARM:=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_warm_k32.txt}"
: "${TESTSET:=$HOME/predictive-inference-moe/moe-advisor/data/user_history/shreya_prompts_test.tsv}"
: "${BENCH:=$HOME/llama.cpp/build/bin/llama-moe-predictor-bench}"

OUTDIR=/tmp/warm_vs_snap_$(date +%s)
mkdir -p "$OUTDIR"
echo "outdir: $OUTDIR"
echo "prompts: $N_PROMPTS  tokens: $N_TOKENS"

# Shared base flags
BASE=(
  -m "$MODEL"
  -c 4096 -b 4096 -ub 1024
  -n "$N_TOKENS"
)

# Forced-offload flags (A and B share)
FORCED=(
  -ngl 99 --override-tensor exps=CPU
  --predictor-weights "$WEIGHTS"
  --prefetch-k 0
  --expert-offsets "$OFFSETS"
  --expert-cache-mb 3500
)

run_snap_only() {  # forced offload, cache, no warm
  GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" "${BASE[@]}" "${FORCED[@]}" -p "$2" > "$1" 2>&1 || true
}
run_snap_warm() {  # forced offload, cache, warm-from-history
  GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" "${BASE[@]}" "${FORCED[@]}" \
    --warm-snapshot-profile "$WARM" -p "$2" > "$1" 2>&1 || true
}
run_ngl12() {  # partial offload, no cache machinery
  "$BENCH" "${BASE[@]}" -ngl 12 \
    --predictor-weights "$WEIGHTS" --prefetch-k 0 \
    -p "$2" > "$1" 2>&1 || true
}
run_cpu_moe() {  # forced-offload flags but NO env var → MoE stays on CPU
  "$BENCH" "${BASE[@]}" "${FORCED[@]}" -p "$2" > "$1" 2>&1 || true
}

for cfg in snap_only snap_warm ngl12 cpu_moe; do
  : > "$OUTDIR/${cfg}.tsv"
done

i=0
while IFS=$'\t' read -r user prompt; do
  [ -z "$prompt" ] && continue
  i=$((i+1))
  [ $i -gt "$N_PROMPTS" ] && break
  echo "[$i/$N_PROMPTS] running 4 configs..."
  for cfg in snap_only snap_warm ngl12 cpu_moe; do
    log="$OUTDIR/${cfg}_${i}.log"
    case "$cfg" in
      snap_only) run_snap_only "$log" "$prompt" ;;
      snap_warm) run_snap_warm "$log" "$prompt" ;;
      ngl12)     run_ngl12     "$log" "$prompt" ;;
      cpu_moe)   run_cpu_moe   "$log" "$prompt" ;;
    esac
    stats=$(grep -m1 "moe_cuda_expert_cache final" "$log" 2>/dev/null || echo "no-cache")
    tps=$(grep -oE "decode:.*tok/s" "$log" 2>/dev/null | tail -1 || echo "no-tps")
    printf "%s\t%s\t%s\n" "${cfg}_${i}" "$stats" "$tps" >> "$OUTDIR/${cfg}.tsv"
  done
done < "$TESTSET"

echo ""
echo "=== AGGREGATES ==="
for cfg in snap_only snap_warm ngl12 cpu_moe; do
  echo "--- $cfg ---"
  awk -F'\t' '
    /d2d_hits=/ {
      match($2, /d2d_hits=([0-9]+)/, a); h += a[1]
      match($2, /d2d_misses=([0-9]+)/, b); m += b[1]
      match($2, /bytes_d2d_served=([0-9.]+)/, c); d2d_mb += c[1]
      match($2, /bytes_prefetched=([0-9.]+)/, e); pre_mb += e[1]
      cache_runs++
    }
    /decode:/ {
      match($3, /\(([0-9.]+) tok/, t); tps_sum += t[1]; tps_n++
    }
    END {
      if (cache_runs > 0) {
        printf "cache: runs=%d  hits=%d  miss=%d  hit_rate=%.2f%%\n", cache_runs, h, m, (h+m>0?100*h/(h+m):0)
        printf "       bytes_prefetched=%.0f MiB  bytes_d2d_served=%.0f MiB\n", pre_mb, d2d_mb
      } else {
        printf "cache: (not used in this config)\n"
      }
      if (tps_n > 0) printf "mean tok/s = %.2f  (over %d runs)\n", tps_sum/tps_n, tps_n
    }
  ' "$OUTDIR/${cfg}.tsv"
done
echo ""
echo "raw logs and per-run rows are under: $OUTDIR"
