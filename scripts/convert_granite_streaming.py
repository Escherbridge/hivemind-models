"""
Streaming Granite sharder.

Walks the HuggingFace safetensors files for ibm-granite/granite-4.0-h-tiny
using safetensors.safe_open (memory-mapped, copy-on-demand). Builds one
output shard at a time and frees its tensors before moving on, so peak RAM
stays under ~2GB even though the full model is ~14GB.

This sidesteps the OOM that killed AutoModelForCausalLM.from_pretrained when
it tried to materialize all 586 tensors into Python objects at once.

Usage:
    python scripts/convert_granite_streaming.py \
        --config configs/granite-tiny.yaml
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("convert_granite_streaming")


# ---------- HF cache helpers ------------------------------------------------


def fetch_index(model_id: str) -> dict[str, str]:
    """Return {tensor_key: shard_filename} mapping from the safetensors index."""
    p = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
    return json.load(open(p))["weight_map"]


def cached_shard_path(model_id: str, shard_filename: str) -> Path:
    """Resolve cached path for a single safetensors shard file."""
    return Path(hf_hub_download(repo_id=model_id, filename=shard_filename))


# ---------- streaming tensor iterator --------------------------------------


def iter_tensors_for_keys(
    model_id: str,
    weight_map: dict[str, str],
    keys: list[str],
    dtype: torch.dtype,
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (key, tensor) pairs for the given keys, opening each shard file
    only as needed. Tensors are copied into the requested dtype.
    """
    # Group keys by which shard file they live in, so we open each file once.
    by_file: dict[str, list[str]] = {}
    for k in keys:
        f = weight_map[k]
        by_file.setdefault(f, []).append(k)

    for shard_filename, file_keys in by_file.items():
        path = cached_shard_path(model_id, shard_filename)
        logger.debug("opening %s for %d keys", shard_filename, len(file_keys))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in file_keys:
                t = f.get_tensor(k)
                if t.dtype != dtype:
                    t = t.to(dtype)
                yield k, t


# ---------- quantization passthrough ---------------------------------------


def maybe_quantize(
    tensors: dict[str, torch.Tensor],
    quantize: bool,
    bits: int,
    group_size: int | None,
) -> dict[str, torch.Tensor]:
    """Apply group-wise INT4 / INT8 quant to weight tensors (>=2D, name contains 'weight')."""
    if not quantize:
        return tensors

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.convert.quantize import quantize_tensor, QuantizationConfig

    cfg = QuantizationConfig(bits=bits, group_size=group_size)
    out = {}
    for name, w in tensors.items():
        if "weight" in name and w.dim() >= 2:
            out[name] = quantize_tensor(w, cfg)
        else:
            out[name] = w
    return out


# ---------- shard plan ------------------------------------------------------


def plan_shards(
    weight_map: dict[str, str],
    layer_groups: list[tuple[int, int]],
) -> list[dict]:
    """
    Return a list of shard descriptors:
      {"name": "embed", "filename": "shard_embed.safetensors", "keys": [...]}
      {"name": "layers_0_9", ..., "layer_range": [0, 9]}
      ...
      {"name": "head",  ...}

    Layer keys are matched by `model.layers.{idx}.` prefix.
    """
    all_keys = list(weight_map.keys())
    plans: list[dict] = []

    embed_keys = [k for k in all_keys if k.startswith("model.embed_tokens.")]
    plans.append({
        "name": "embed",
        "filename": "shard_embed.safetensors",
        "keys": embed_keys,
        "layer_range": None,
    })

    for (start, end) in layer_groups:
        layer_keys: list[str] = []
        for idx in range(start, end + 1):
            prefix = f"model.layers.{idx}."
            layer_keys.extend(k for k in all_keys if k.startswith(prefix))
        plans.append({
            "name": f"layers_{start}_{end}",
            "filename": f"shard_layers_{start}_{end}.safetensors",
            "keys": layer_keys,
            "layer_range": [start, end],
        })

    # Granite-hybrid has model.norm.weight + tied lm_head (we include embed too
    # so the head shard can serve as a stand-alone unit; matches the handler).
    head_keys = [k for k in all_keys if k.startswith("model.norm.")]
    # lm_head.weight may or may not exist explicitly (tied); include embed as backup
    head_keys.extend(k for k in all_keys if k.startswith("lm_head."))
    head_keys.extend(k for k in all_keys if k.startswith("model.embed_tokens."))
    # de-dupe while preserving order
    seen = set()
    head_keys = [k for k in head_keys if not (k in seen or seen.add(k))]

    plans.append({
        "name": "head",
        "filename": "shard_head.safetensors",
        "keys": head_keys,
        "layer_range": None,
    })

    return plans


# ---------- main ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output_dir from YAML")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_id = cfg["model_id"]
    output_dir = args.output or Path(cfg["output_dir"])
    layer_groups = [tuple(g) for g in cfg["layer_groups"]]
    quantize = bool(cfg.get("quantize", False))
    quant_bits = int(cfg.get("quant_bits", 4))
    quant_group_size = cfg.get("quant_group_size")
    dtype_str = cfg.get("dtype", "float16")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("model_id=%s", model_id)
    logger.info("output_dir=%s", output_dir)
    logger.info("layer_groups=%s", layer_groups)
    logger.info("quantize=%s bits=%s group=%s", quantize, quant_bits, quant_group_size)

    # 1. Resolve weight map (forces a fetch of the index file).
    logger.info("fetching weight index ...")
    weight_map = fetch_index(model_id)
    logger.info("  %d tensors across %d shard files",
                len(weight_map), len(set(weight_map.values())))

    # 2. Plan the output shards.
    plans = plan_shards(weight_map, layer_groups)
    logger.info("planned %d output shards:", len(plans))
    for p in plans:
        logger.info("  %-16s keys=%d  range=%s", p["name"], len(p["keys"]), p["layer_range"])

    # 3. Build each output shard, streaming its tensors and freeing them when done.
    manifest_shards = []
    total_bytes = 0
    t_start = time.perf_counter()

    for p in plans:
        logger.info("=== building %s ===", p["name"])
        t0 = time.perf_counter()
        tensors: dict[str, torch.Tensor] = {}
        for k, t in iter_tensors_for_keys(model_id, weight_map, p["keys"], dtype):
            tensors[k] = t.contiguous()

        tensors = maybe_quantize(tensors, quantize, quant_bits, quant_group_size)
        # Move to CPU + contiguous (already CPU from safe_open, but be safe)
        cpu = {k: v.cpu().contiguous() for k, v in tensors.items()}

        out_path = output_dir / p["filename"]
        save_file(cpu, str(out_path))
        size_bytes = out_path.stat().st_size
        total_bytes += size_bytes
        elapsed = time.perf_counter() - t0
        logger.info("  wrote %s (%d tensors, %.2f MB) in %.1fs",
                    out_path.name, len(cpu), size_bytes / 1e6, elapsed)

        # Checksum
        hasher = hashlib.sha256()
        with open(out_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        checksum = hasher.hexdigest()

        manifest_shards.append({
            "filename": p["filename"],
            "size_bytes": size_bytes,
            "checksum_sha256": checksum,
            "tensor_count": len(cpu),
            "layer_range": p["layer_range"],
        })

        # Free
        del tensors, cpu
        gc.collect()

    # 4. Manifest.
    manifest = {
        "model_id": model_id,
        "model_type": "granitemoehybrid",
        "version": "1.0",
        "total_size_bytes": total_bytes,
        "quantized": quantize,
        "quant_bits": quant_bits if quantize else None,
        "quant_group_size": quant_group_size,
        "dtype": dtype_str,
        "layer_groups": [list(g) for g in layer_groups],
        "shards": manifest_shards,
        "tokenizer": {"path": "tokenizer/", "type": "AutoTokenizer"},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed_total = time.perf_counter() - t_start
    logger.info("=== done in %.1fs, %.2f GB total ===", elapsed_total, total_bytes / 1e9)
    logger.info("manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
