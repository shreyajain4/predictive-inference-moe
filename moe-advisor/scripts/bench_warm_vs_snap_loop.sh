#!/bin/bash
# bench_warm_vs_snap_loop.sh
# Runs the snapshot bench in forced-offload mode (-ngl 99 + override exps=CPU
# + GGML_OP_OFFLOAD_MIN_BATCH=1) over many test prompts, twice: once with
# --warm-snapshot-profile, once without. Aggregates cache stats + tok/s.
#
# Each bench launch reloads the model (~5 s). With -n 8 and 30 prompts that's
# ~20 min per config.
#
# Usage:
#   bash scripts/bench_warm_vs_snap_loop.sh [N_PROMPTS] [N_TOKENS]
# Defaults: N_PROMPTS=30, N_TOKENS=8

set -e

N_PROMPTS=${1:-30}
N_TOKENS=${2:-8}

# Required paths (set as env vars OR edit defaults below)
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

COMMON_FLAGS=(
  -m "$MODEL"
  -c 4096 -b 4096 -ub 1024
  -n "$N_TOKENS"
  -ngl 99 --override-tensor exps=CPU
  --predictor-weights "$WEIGHTS"
  --prefetch-k 0
  --expert-offsets "$OFFSETS"
  --expert-cache-mb 3500
)

run_one() {
  local label="$1"
  local prompt="$2"
  local extra="$3"     # "" or "--warm-snapshot-profile $WARM"
  local logfile="$OUTDIR/${label}.log"
  GGML_OP_OFFLOAD_MIN_BATCH=1 "$BENCH" "${COMMON_FLAGS[@]}" \
    $extra -p "$prompt" > "$logfile" 2>&1 || true
  # Parse cache stats line and tok/s
  local stats tps
  stats=$(grep -m1 "moe_cuda_expert_cache final" "$logfile" || echo "MISSING")
  tps=$(grep -oE "decode:.*tok/s" "$logfile" | tail -1 || echo "MISSING")
  printf "%s\t%s\t%s\n" "$label" "$stats" "$tps"
}

echo "=== snap-only ===" > "$OUTDIR/snap_only.tsv"
echo "=== snap+warm ===" > "$OUTDIR/snap_warm.tsv"

i=0
while IFS=$'\t' read -r user prompt; do
  [ -z "$prompt" ] && continue
  i=$((i+1))
  [ $i -gt "$N_PROMPTS" ] && break
  echo "[$i/$N_PROMPTS] running both configs..."
  run_one "snap_only_$i" "$prompt" "" >> "$OUTDIR/snap_only.tsv"
  run_one "snap_warm_$i" "$prompt" "--warm-snapshot-profile $WARM" >> "$OUTDIR/snap_warm.tsv"
done < "$TESTSET"

echo ""
echo "=== AGGREGATES ==="
for f in "$OUTDIR/snap_only.tsv" "$OUTDIR/snap_warm.tsv"; do
  echo "--- $f ---"
  awk -F'\t' '
    /d2d_hits=/ {
      match($2, /d2d_hits=([0-9]+)/, a); h += a[1]
      match($2, /d2d_misses=([0-9]+)/, b); m += b[1]
      match($2, /bytes_d2d_served=([0-9.]+)/, c); d2d_mb += c[1]
      match($2, /bytes_prefetched=([0-9.]+)/, e); pre_mb += e[1]
      n++
    }
    /decode:/ {
      match($3, /\(([0-9.]+) tok/, t); tps_sum += t[1]; tps_n++
    }
    END {
      printf "runs=%d  hits=%d  miss=%d  hit_rate=%.2f%%\n", n, h, m, (h+m>0?100*h/(h+m):0)
      printf "bytes_prefetched=%.0f MiB   bytes_d2d_served=%.0f MiB\n", pre_mb, d2d_mb
      if (tps_n > 0) printf "mean tok/s = %.2f  (over %d runs)\n", tps_sum/tps_n, tps_n
    }
  ' "$f"
done
echo ""
echo "raw logs and per-run rows are under: $OUTDIR"
