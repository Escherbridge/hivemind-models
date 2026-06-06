"""
Debug correctness of the sharded pipeline.

Runs a single forward pass through TWO paths on the SAME input:

  PATH A (reference): Load the full unsharded HF model and run model(input_ids)
                      capturing hidden_states after each layer via hooks.
  PATH B (sharded):   Replicate exactly what shard_server.py does, in-process,
                      layer-by-layer using the shard safetensors files.

Compare hidden states at each layer boundary (max abs diff, mean abs diff,
cosine similarity). Print where divergence starts.

Usage:
    python scripts/debug_shard_correctness.py --shard-dir ./output/tinyllama-1b-q4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare two tensors of the same shape."""
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {"shape_mismatch": True, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0),
        b.flatten().unsqueeze(0),
    ).item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "cosine_sim": cos,
        "a_norm": a.norm().item(),
        "b_norm": b.norm().item(),
    }


def reference_path(model_id: str, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the full HF model, capture hidden states at each layer + final logits."""
    print("\n[REFERENCE] Loading full HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).eval()

    captures: dict[str, torch.Tensor] = {}

    # output_hidden_states=True returns one tensor per layer (including embeddings)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)

    # hidden_states is a tuple of (n_layers + 1) tensors
    # [0] = embedding output, [i] = output of layer i-1
    hidden_states = out.hidden_states
    captures["embed"] = hidden_states[0].clone()
    for i in range(1, len(hidden_states)):
        captures[f"after_layer_{i-1}"] = hidden_states[i].clone()
    captures["logits"] = out.logits.clone()
    print(f"[REFERENCE] Captured {len(captures)} tensors. Embed shape: {captures['embed'].shape}")
    return captures


def sharded_path(
    shard_dir: Path,
    model_id: str,
    input_ids: torch.Tensor,
    layer_groups: list[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Run sharded inference EXACTLY as shard_server.py does. Capture intermediates."""
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
        create_causal_mask,
    )

    print("\n[SHARDED] Loading config + shards...")
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"

    captures: dict[str, torch.Tensor] = {}
    device = torch.device("cpu")

    # --- Embed shard ---
    embed_state = load_file(str(shard_dir / "shard_embed.safetensors"))
    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size).to(device)
    embed.weight.data = embed_state["model.embed_tokens.weight"].to(device)
    embed.eval()
    with torch.no_grad():
        hidden = embed(input_ids)
    captures["embed"] = hidden.clone()
    print(f"[SHARDED] embed -> {hidden.shape}, dtype={hidden.dtype}")

    # ---- Simulate the over-the-wire trip: float32 -> reload as torch ----
    # shard_server packs as float32, unpacks back to torch via from_numpy.
    # Then the layer handler does hidden = hidden.half().
    hidden = hidden.float()  # pack step
    hidden = torch.from_numpy(hidden.cpu().numpy().copy())  # unpack step

    # --- Layer groups ---
    for (layer_start, layer_end) in layer_groups:
        shard_file = shard_dir / f"shard_layers_{layer_start}_{layer_end}.safetensors"
        state = load_file(str(shard_file))

        layers = torch.nn.ModuleList()
        for idx in range(layer_start, layer_end + 1):
            layer = LlamaDecoderLayer(config, layer_idx=idx).to(device)
            prefix = f"model.layers.{idx}."
            layer_state = {
                k.removeprefix(prefix): v.to(device)
                for k, v in state.items()
                if k.startswith(prefix)
            }
            layer.load_state_dict(layer_state, strict=False)
            layer.eval()
            layers.append(layer)

        rotary_emb = LlamaRotaryEmbedding(config=config).to(device)
        rotary_emb.eval()

        # --- This block is the EXACT code path of shard_server.py /forward ---
        hidden = hidden.half()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        position_embeddings = rotary_emb(hidden, position_ids)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        with torch.no_grad():
            for layer in layers:
                result = layer(
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden = result[0]
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

        captures[f"after_layer_{layer_end}"] = hidden.clone()
        print(f"[SHARDED] layers {layer_start}-{layer_end} -> {hidden.shape}, dtype={hidden.dtype}")

        # Round-trip simulation between shards
        hidden = hidden.float()
        hidden = torch.from_numpy(hidden.cpu().numpy().copy())

    # --- Head shard ---
    head_state = load_file(str(shard_dir / "shard_head.safetensors"))
    norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)
    norm.weight.data = head_state["model.norm.weight"].to(device)
    head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device)
    head.weight.data = head_state["lm_head.weight"].to(device)
    norm.eval(); head.eval()

    hidden = hidden.half()
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    with torch.no_grad():
        normed = norm(hidden)
        logits = head(normed)
    captures["logits"] = logits.clone()
    print(f"[SHARDED] head -> {logits.shape}")

    return captures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="./output/tinyllama-1b-q4")
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    model_id = manifest["model_id"]
    layer_groups = [tuple(g) for g in manifest["layer_groups"]]

    print(f"Model: {model_id}")
    print(f"Layer groups: {layer_groups}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    print(f"Prompt: '{args.prompt}'")
    print(f"Tokens ({input_ids.shape[1]}): {input_ids.tolist()}")

    # Run both paths
    ref = reference_path(model_id, input_ids)
    shd = sharded_path(shard_dir, model_id, input_ids, layer_groups)

    # Compare at every captured key
    print("\n" + "=" * 72)
    print("DIVERGENCE ANALYSIS (REFERENCE vs SHARDED)")
    print("=" * 72)
    print(f"{'stage':<22s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'cos_sim':>10s}  {'a_norm':>10s}  {'b_norm':>10s}")
    print("-" * 80)

    ref_keys = list(ref.keys())
    for k in ref_keys:
        if k not in shd:
            print(f"{k:<22s}  (missing in sharded)")
            continue
        d = tensor_diff(ref[k], shd[k])
        if d.get("shape_mismatch"):
            print(f"{k:<22s}  SHAPE MISMATCH  ref={d['a_shape']}  shd={d['b_shape']}")
            continue
        flag = ""
        if d["cosine_sim"] < 0.99:
            flag = "  <-- DIVERGES"
        if d["cosine_sim"] < 0.5:
            flag = "  <-- GARBAGE"
        print(f"{k:<22s}  {d['max_abs']:>10.4f}  {d['mean_abs']:>10.4f}  "
              f"{d['cosine_sim']:>10.4f}  {d['a_norm']:>10.2f}  {d['b_norm']:>10.2f}{flag}")

    # Final logits top-5 comparison
    print("\n--- Top-5 next token predictions ---")
    ref_logits = ref["logits"][0, -1].float()
    shd_logits = shd["logits"][0, -1].float()
    ref_top5 = torch.topk(ref_logits, 5)
    shd_top5 = torch.topk(shd_logits, 5)

    def safe(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")
    print(f"  REFERENCE: {[(safe(tokenizer.decode([int(i)])), float(v)) for i, v in zip(ref_top5.indices, ref_top5.values)]}")
    print(f"  SHARDED:   {[(safe(tokenizer.decode([int(i)])), float(v)) for i, v in zip(shd_top5.indices, shd_top5.values)]}")


if __name__ == "__main__":
    main()
