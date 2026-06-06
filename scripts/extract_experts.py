"""
Per-expert weight extraction for a single Granite MoE layer.

Given the layer-group shard that contains layer L, slice the layer's
block_sparse_moe tensors along the expert dimension and write one
.safetensors file per expert.

Granite MoE weight layout (from HF GraniteMoeHybridParallelExperts):
  input_linear.weight  shape = [num_experts, 2 * intermediate_size, hidden_size]
  output_linear.weight shape = [num_experts, hidden_size, intermediate_size]
  router.layer.weight  shape = [num_experts, hidden_size]   (stays with client)

Each expert's *input_linear* slice contains BOTH the gate and up projections,
fused along dim 1 — chunk(2, dim=-1) at inference time splits them.

Output:
  output/granite-tiny-q4g64/experts_layer_<L>/expert_<i>.safetensors
    keys:
      input_linear.weight   shape = [2 * intermediate_size, hidden_size]
      output_linear.weight  shape = [hidden_size, intermediate_size]

  output/granite-tiny-q4g64/experts_layer_<L>/router.safetensors
    keys:
      router.weight         shape = [num_experts, hidden_size]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path,
                        default=Path("./output/granite-tiny-q4g64"))
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer index whose MoE block we extract per-expert")
    parser.add_argument("--out-subdir", type=str, default=None,
                        help="Optional override for output subdirectory name")
    args = parser.parse_args()

    # Find which layer-group shard contains the requested layer.
    manifest = json.loads((args.shard_dir / "manifest.json").read_text())
    target_shard = None
    for s in manifest["shards"]:
        lr = s.get("layer_range")
        if lr and lr[0] <= args.layer <= lr[1]:
            target_shard = s
            break
    if target_shard is None:
        print(f"ERROR: layer {args.layer} not found in any shard")
        return 1
    shard_path = args.shard_dir / target_shard["filename"]
    print(f"layer {args.layer} lives in {shard_path.name}")

    prefix = f"model.layers.{args.layer}.block_sparse_moe."
    input_key = prefix + "input_linear.weight"
    output_key = prefix + "output_linear.weight"
    router_key = prefix + "router.layer.weight"

    out_dir = args.shard_dir / (args.out_subdir or f"experts_layer_{args.layer}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"output: {out_dir}")
    t_total = time.perf_counter()

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for k in (input_key, output_key, router_key):
            if k not in keys:
                print(f"ERROR: expected key not in shard: {k}")
                return 1

        input_linear = f.get_tensor(input_key)
        output_linear = f.get_tensor(output_key)
        router = f.get_tensor(router_key)

    num_experts = input_linear.shape[0]
    hidden_size = input_linear.shape[2]
    intermediate_x2 = input_linear.shape[1]
    print(f"  input_linear  : {list(input_linear.shape)} {input_linear.dtype}")
    print(f"  output_linear : {list(output_linear.shape)} {output_linear.dtype}")
    print(f"  router        : {list(router.shape)} {router.dtype}")
    print(f"  num_experts={num_experts}  hidden_size={hidden_size}  intermediate*2={intermediate_x2}")

    # Save the router as a separate file (stays with the client/coordinator).
    save_file({"router.weight": router.contiguous()}, str(out_dir / "router.safetensors"))
    print(f"  wrote router.safetensors ({router.numel() * router.element_size() / 1e6:.2f} MB)")

    # Per-expert slices.
    total_bytes = 0
    for i in range(num_experts):
        ein = input_linear[i].contiguous()       # [intermediate_x2, hidden_size]
        eout = output_linear[i].contiguous()      # [hidden_size, intermediate]
        save_file(
            {"input_linear.weight": ein, "output_linear.weight": eout},
            str(out_dir / f"expert_{i:02d}.safetensors"),
        )
        sz = (ein.numel() * ein.element_size()
              + eout.numel() * eout.element_size())
        total_bytes += sz

    # Manifest for the per-expert set.
    expert_manifest = {
        "layer": args.layer,
        "source_shard": target_shard["filename"],
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_x2 // 2,
        "per_expert_bytes": total_bytes // num_experts,
        "total_bytes": total_bytes,
        "router_file": "router.safetensors",
        "expert_files": [f"expert_{i:02d}.safetensors" for i in range(num_experts)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(expert_manifest, indent=2))

    elapsed = time.perf_counter() - t_total
    print(f"\nwrote {num_experts} expert shards (~{total_bytes / 1e6 / num_experts:.2f} MB each, "
          f"{total_bytes / 1e6:.1f} MB total) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
