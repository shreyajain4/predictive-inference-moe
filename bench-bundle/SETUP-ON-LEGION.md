# Setup on Legion (3070 box)

This bundle contains the moe-predictor-bench source files, the trained LR-hidden predictor weights for Qwen3-30B-A3B, and helpers to generate Q4-specific expert offsets.

## Prerequisites (already done if you got this far)

- llama.cpp cloned at `~/llama.cpp`, checked out at commit `6b80c74` (or compatible)
- Build directory exists with CUDA enabled: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release`
- Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf at `~/models/`

## Step 1: Drop bench source into llama.cpp

```bash
mkdir -p ~/llama.cpp/examples/moe-predictor-bench
cp ~/predictive-inference-moe/bench-bundle/bench-source/* \
   ~/llama.cpp/examples/moe-predictor-bench/

# Register in the examples build (add line under simple)
grep -q "moe-predictor-bench" ~/llama.cpp/examples/CMakeLists.txt || \
  sed -i "/add_subdirectory(simple)/a\\    add_subdirectory(moe-predictor-bench)" ~/llama.cpp/examples/CMakeLists.txt
```

## Step 2: Build the bench

```bash
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
cmake --build build --target llama-moe-predictor-bench -j$(nproc) 2>&1 | tail -10
```

Successful build ends with: `Built target llama-moe-predictor-bench`

## Step 3: Generate Q4-specific expert offsets JSON

The bundled `qwen3_expert_offsets.json` is from the Q8 GGUF. Q4 has different file offsets and per-expert byte counts, so we need to regenerate:

```bash
cd ~/predictive-inference-moe/bench-bundle/predictor-weights
python3 extract_expert_offsets.py \
    --gguf ~/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    --output ~/qwen3_q4_expert_offsets.json
```

If that script needs deps:
```bash
pip3 install --user --break-system-packages numpy gguf
```

## Step 4: Phase 0 — predictor recall sanity check on Q4

The critical question: does our Q8-trained predictor still produce ≥55% recall@8 on Q4 traces?

```bash
~/llama.cpp/build/bin/llama-moe-predictor-bench \
    -m ~/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    -p "The role of attention in modern neural networks is" \
    -n 64 --seed 42 --temp 0 -ngl 0 -fit off \
    --predictor-weights ~/predictive-inference-moe/bench-bundle/predictor-weights/qwen3_predictor_weights.bin \
    --prefetch-k 12 --prefetch-horizon 1 \
    --expert-offsets ~/qwen3_q4_expert_offsets.json \
    2>&1 | tail -60
```

What to look for in the output:
- A table showing per-layer recall@k
- A `micro_recall@12` line near the end with a percentage

Decision:
- **recall@12 ≥ 55%**: proceed with cache experiment using existing predictor
- **recall@12 < 50%**: pause, retrain predictor on Q4 traces (separate task)

## Step 5: Forced-GPU baseline with predictor enabled (already-measured baseline 8.99 t/s)

If Phase 0 passes:

```bash
GGML_OP_OFFLOAD_MIN_BATCH=1 ~/llama.cpp/build/bin/llama-moe-predictor-bench \
    -m ~/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    -p "The role of attention in modern neural networks is" \
    -n 64 --seed 42 --temp 0 \
    -ngl 99 --override-tensor "ffn_.*_exps=CPU" -fit off \
    --predictor-weights ~/predictive-inference-moe/bench-bundle/predictor-weights/qwen3_predictor_weights.bin \
    --prefetch-k 12 --prefetch-horizon 1 \
    --expert-offsets ~/qwen3_q4_expert_offsets.json \
    2>&1 | tail -40
```

This is the same setup as the doc's `copy_experts_pcie_measurement` but on PCIe gen4. Look for:
- A `decode: N tokens in M ms (X tok/s)` line — should be 8-10 t/s (matches our earlier baseline)
- The `[MOE-COPY-STATS] copy_experts calls=...` line on exit
- Predictor recall numbers

## Step 6+ (after Phase 0 passes): persistent HBM cache

This is the engineering work. The bench currently includes:
- The Metal-side `moe_layer_stream` (used on Mac, not relevant here)
- The CUDA-side `moe_cuda_layer_stream.cu` (port of the Metal worker; built when GGML_CUDA is on)

We need to add:
- A persistent VRAM expert pool (`cudaMalloc` of 5 GB at startup)
- LRU eviction tracking
- Intercept in `ggml-backend.cpp:1648` (`copy_experts` lambda) to check cache before PCIe transfer
- Predictor-driven prefetch via background `cudaMemcpyAsync` on a separate stream

This is ~2-3 days of focused work. Once you confirm Phase 0 passes and the forced-GPU baseline reproduces 8.99 t/s, we'll iterate on the cache code together.
