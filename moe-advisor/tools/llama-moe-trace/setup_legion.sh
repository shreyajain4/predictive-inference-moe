#!/usr/bin/env bash
# Drops the moe-trace example into Legion's llama.cpp tree and builds it.
# Assumes llama.cpp lives at ~/llama.cpp with a working build/ already configured.
#
# Idempotent: copying + adding the subdirectory line are safe to re-run.

set -e

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
EXAMPLE_DIR="$LLAMA_DIR/examples/moe-trace"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> source: $SRC_DIR"
echo "==> llama.cpp tree: $LLAMA_DIR"

if [ ! -d "$LLAMA_DIR/build" ]; then
  echo "ERROR: $LLAMA_DIR/build doesn't exist. Configure llama.cpp first."
  exit 1
fi

mkdir -p "$EXAMPLE_DIR"
cp "$SRC_DIR/moe-trace.cpp"     "$EXAMPLE_DIR/moe-trace.cpp"
cp "$SRC_DIR/CMakeLists.txt"    "$EXAMPLE_DIR/CMakeLists.txt"
echo "==> copied moe-trace.cpp + CMakeLists.txt"

# Register subdirectory in examples/CMakeLists.txt if not already there
EXAMPLES_CMAKE="$LLAMA_DIR/examples/CMakeLists.txt"
if ! grep -q "add_subdirectory(moe-trace)" "$EXAMPLES_CMAKE"; then
  # Insert next to another example to keep the file tidy
  if grep -q "add_subdirectory(eval-callback)" "$EXAMPLES_CMAKE"; then
    sed -i '/add_subdirectory(eval-callback)/a\    add_subdirectory(moe-trace)' "$EXAMPLES_CMAKE"
  else
    echo "    add_subdirectory(moe-trace)" >> "$EXAMPLES_CMAKE"
  fi
  echo "==> registered add_subdirectory(moe-trace) in examples/CMakeLists.txt"
else
  echo "==> add_subdirectory(moe-trace) already registered"
fi

# Reconfigure (cheap if cmake cache is healthy) + build the new target
cd "$LLAMA_DIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release > /tmp/moe_trace_cmake.log 2>&1 || {
  echo "ERROR: cmake reconfigure failed; see /tmp/moe_trace_cmake.log"
  tail -30 /tmp/moe_trace_cmake.log
  exit 1
}
echo "==> cmake configure ok"

echo "==> building llama-moe-trace ..."
cmake --build . --target llama-moe-trace -j 8

BIN="$LLAMA_DIR/build/bin/llama-moe-trace"
if [ -x "$BIN" ]; then
  echo ""
  echo "==> success: $BIN"
  echo ""
  echo "Run:"
  echo "  cd ~/predictive-inference-moe/moe-advisor"
  echo "  nohup $BIN \\"
  echo "    -m /home/shreya/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \\"
  echo "    -f data/user_history/shreya_prompts_train.tsv \\"
  echo "    > data/user_history/shreya_train_routing.csv \\"
  echo "    2> data/user_history/shreya_train_routing.log &"
  echo "  echo \"PID=\$!\""
else
  echo "ERROR: build did not produce $BIN"
  exit 1
fi
