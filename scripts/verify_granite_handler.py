"""
In-process correctness check for the Granite PyTorch handler.

Loads each shard via the GraniteHandler that pi wrote, runs a single forward
pass through the full pipeline (embed -> 4 layer groups -> head), and
compares the top-1 prediction against the llama.cpp GGUF reference
(should be ' Paris' for 'The capital of France is').

No HTTP. No swarm. Just the handler talking to itself, so we can prove the
handler logic and the streamed shards are individually correct before we
spread them across processes.

Usage:
    python scripts/verify_granite_handler.py \
        --shard-dir ./output/granite-tiny-q4g64 \
        --reference output/granite_reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.architectures import LayerShardSpec, get_handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path,
                        default=Path("./output/granite-tiny-q4g64"))
    parser.add_argument("--reference", type=Path,
                        default=Path("output/granite_reference.json"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads((args.shard_dir / "manifest.json").read_text())
    model_id = manifest["model_id"]
    layer_groups = [tuple(g) for g in manifest["layer_groups"]]
    print(f"model_id: {model_id}")
    print(f"layer_groups: {layer_groups}")

    device = torch.device(args.device)

    # We need the HF config to construct the handler's modules. We CAN'T load
    # the full HF model on this machine (that's what OOM-ed last time), but
    # AutoConfig is tiny — just JSON.
    print("\nLoading HF config (no weights)...")
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"

    handler = get_handler("granitemoehybrid")
    print(f"Using handler: {handler.name}")

    # ---- Embed ----
    print("\n[embed] loading shard...")
    t0 = time.perf_counter()
    embed_state = load_file(str(args.shard_dir / "shard_embed.safetensors"))
    embed_mod = handler.build_embed(config, embed_state, device)
    del embed_state
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ---- Tokenize ----
    tok = AutoTokenizer.from_pretrained(model_id)
    input_ids = tok.encode(args.prompt, return_tensors="pt").to(device)
    print(f"prompt: {args.prompt!r}")
    print(f"tokens ({input_ids.shape[1]}): {input_ids.tolist()}")

    t0 = time.perf_counter()
    hidden = handler.forward_embed(embed_mod, input_ids)
    print(f"[embed]  -> {list(hidden.shape)} dtype={hidden.dtype}  ({(time.perf_counter()-t0)*1000:.0f}ms)")
    del embed_mod

    # ---- Layer groups (one at a time, free after use) ----
    for (start, end) in layer_groups:
        shard_path = args.shard_dir / f"shard_layers_{start}_{end}.safetensors"
        print(f"\n[layers {start}..{end}] loading shard ({shard_path.stat().st_size / 1e6:.0f} MB)...")
        t0 = time.perf_counter()
        layer_state = load_file(str(shard_path))
        layer_mod = handler.build_layers(
            config, layer_state, LayerShardSpec(layer_start=start, layer_end=end), device,
        )
        del layer_state
        load_s = time.perf_counter() - t0
        print(f"  loaded in {load_s:.1f}s")

        t0 = time.perf_counter()
        hidden = handler.forward_layers(layer_mod, hidden)
        print(f"  -> {list(hidden.shape)} dtype={hidden.dtype}  ({(time.perf_counter()-t0)*1000:.0f}ms)")
        del layer_mod

    # ---- Head ----
    print("\n[head] loading shard...")
    t0 = time.perf_counter()
    head_state = load_file(str(args.shard_dir / "shard_head.safetensors"))
    head_mod = handler.build_head(config, head_state, device)
    del head_state
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    logits = handler.forward_head(head_mod, hidden)
    print(f"[head]   -> {list(logits.shape)} dtype={logits.dtype}  ({(time.perf_counter()-t0)*1000:.0f}ms)")

    # ---- Top-k ----
    last = logits[0, -1].float()
    topk = torch.topk(last, k=5)
    print("\nTop-5 next-token predictions:")
    for tok_id, lv in zip(topk.indices.tolist(), topk.values.tolist()):
        tok_str = tok.decode([tok_id])
        safe = tok_str.encode("ascii", errors="replace").decode("ascii")
        print(f"  {tok_id:6d}  {lv:8.3f}  {safe!r}")

    top1_id = int(topk.indices[0])
    top1_str = tok.decode([top1_id])
    print(f"\nPyTorch handler top-1: {top1_str!r}")

    # ---- Reference comparison ----
    if args.reference.exists():
        ref = json.loads(args.reference.read_text(encoding="utf-8"))
        ref_top1 = ref["top_k_tokens"][0]["token"]
        # GGUF reference includes the leading space; HF tokenizer may differ
        print(f"GGUF reference top-1: {ref_top1!r}")
        if top1_str.strip().lower() == ref_top1.strip().lower():
            print("\nPASS: top-1 tokens match the reference (modulo whitespace)")
            return 0
        else:
            print("\nFAIL: top-1 tokens differ")
            return 1
    else:
        print(f"\n(no reference at {args.reference} -- skipping parity check)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
