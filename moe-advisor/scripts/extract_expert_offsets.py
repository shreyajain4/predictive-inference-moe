"""
extract_expert_offsets.py
-------------------------
For a Qwen3-style MoE GGUF, write a JSON file listing each MoE layer's expert-
tensor byte ranges (gate/up/down) so the C++ bench can call posix_madvise on
specific expert slices without doing GGUF parsing in C++.

Output JSON shape:
{
  "gguf_path": "/abs/path/to/file.gguf",
  "tensor_data_start": <int>,   # absolute file offset where tensor data section begins
  "num_layers": 48,
  "num_experts": 128,
  "layers": [
    {
      "layer": 0,
      "gate":  {"base_offset": <int>, "per_expert_bytes": <int>, "total_bytes": <int>},
      "up":    {...},
      "down":  {...}
    },
    ...
  ]
}

Notes
-----
- For Qwen3 GGUF v3, each ffn_*_exps tensor is shape (e0, e1, num_experts).
  Per-expert slice = e0 * e1 * type_size bytes (after accounting for Q8 block packing).
- "base_offset" is the file offset of the FIRST expert (expert 0) within that tensor.
- To get expert e's byte range: base + e * per_expert_bytes, len = per_expert_bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("install gguf-py first: pip install gguf  (or use llama.cpp's gguf-py path)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf",   required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    # In current gguf-py, tensor.data_offset is ALREADY the absolute file offset.
    # (Earlier we added r.data_offset on top, which was a silent off-by-section-start
    # bug — for Qwen3 Q4_K_M r.data_offset ≈ 5.7 MB, so we read ~6 MB ahead of where
    # each tensor actually lives. Verified 2026-06-14 against raw-file bytes.)
    tensor_data_start = int(r.data_offset)  # still recorded in the JSON for reference

    # Group expert tensors per layer.
    layers: dict[int, dict] = {}
    num_experts = None
    for t in r.tensors:
        name = t.name
        if not name.startswith("blk."):
            continue
        if "ffn_gate_exps.weight" in name:
            proj = "gate"
        elif "ffn_up_exps.weight" in name:
            proj = "up"
        elif "ffn_down_exps.weight" in name:
            proj = "down"
        else:
            continue
        # Parse layer index from "blk.{L}.ffn_..."
        layer = int(name.split(".")[1])

        # Shape is (e0, e1, num_experts) — last dim varies expert.
        shape = list(t.shape)
        if len(shape) != 3:
            print(f"WARN: {name} has shape {shape}, expected 3D", file=sys.stderr)
            continue
        ne = shape[-1]   # num_experts
        if num_experts is None:
            num_experts = int(ne)
        elif int(ne) != num_experts:
            print(f"WARN: inconsistent num_experts: {ne} vs {num_experts} for {name}",
                  file=sys.stderr)

        # Byte size per expert slice = total tensor bytes / num_experts.
        # (Quant block alignment is handled by ggml so we trust ggml's reported size.)
        total = int(t.n_bytes)
        per_expert = total // int(ne)
        if total != per_expert * int(ne):
            print(f"WARN: {name} bytes {total} not evenly divisible by {ne} experts",
                  file=sys.stderr)

        abs_base = int(t.data_offset)   # gguf-py's data_offset is already absolute

        layers.setdefault(layer, {})[proj] = {
            "base_offset":      int(abs_base),
            "per_expert_bytes": int(per_expert),
            "total_bytes":      int(total),
            "dtype":            t.tensor_type.name,
            "shape":            [int(x) for x in shape],
        }

    if num_experts is None:
        sys.exit("found no MoE expert tensors in the GGUF")

    sorted_layers = []
    for L in sorted(layers.keys()):
        entry = {"layer": L}
        entry.update(layers[L])
        sorted_layers.append(entry)

    out = {
        "gguf_path":         str(args.gguf.resolve()),
        "tensor_data_start": tensor_data_start,
        "num_layers":        len(sorted_layers),
        "num_experts":       num_experts,
        "layers":            sorted_layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}: {len(sorted_layers)} MoE layers × {num_experts} experts each",
          flush=True)
    if sorted_layers:
        ex = sorted_layers[0]
        for p in ("gate", "up", "down"):
            if p in ex:
                d = ex[p]
                print(f"  L0 {p}: base={d['base_offset']}, "
                      f"per_expert={d['per_expert_bytes']} bytes "
                      f"({d['per_expert_bytes']/1024:.1f} KiB), "
                      f"total={d['total_bytes']/1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
