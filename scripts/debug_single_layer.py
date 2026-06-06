"""
Minimal repro: run a SINGLE layer (layer 0) two ways:
  (a) Via the full HF model with output_hidden_states
  (b) Standalone, exactly like shard_server does (with causal mask now)
  (c) Standalone, but pulling layer 0's WEIGHTS DIRECTLY from the loaded HF model
      (sanity check: is it the shard weights, or the calling convention?)

Goal: isolate whether the divergence is:
  - in the shard weights themselves (load_file issue)
  - in how we call LlamaDecoderLayer (missing kwarg)
  - in the rotary embedding usage
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    create_causal_mask,
)


def main():
    shard_dir = Path("./output/tinyllama-1b-q4")
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    model_id = manifest["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")

    # Load full model
    print("Loading full HF model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).eval()
    config = model.config

    # (a) Get layer 0 input and output from full model
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    layer_0_input = out.hidden_states[0]   # embedding output
    layer_0_output_ref = out.hidden_states[1]  # after layer 0

    print(f"\nLayer 0 input:  shape={list(layer_0_input.shape)}, norm={layer_0_input.norm():.3f}")
    print(f"Layer 0 output (REF): norm={layer_0_output_ref.norm():.3f}")

    # (b) Standalone with shard weights
    shard_state = load_file(str(shard_dir / "shard_layers_0_3.safetensors"))
    standalone_layer = LlamaDecoderLayer(config, layer_idx=0)
    layer_state = {
        k.removeprefix("model.layers.0."): v
        for k, v in shard_state.items()
        if k.startswith("model.layers.0.")
    }
    standalone_layer.load_state_dict(layer_state, strict=False)
    standalone_layer.eval()
    standalone_layer = standalone_layer.half()

    rotary = LlamaRotaryEmbedding(config=config).eval()
    seq_len = layer_0_input.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    pos_emb = rotary(layer_0_input.half(), position_ids)
    mask = create_causal_mask(
        config=config, inputs_embeds=layer_0_input.half(),
        attention_mask=None, past_key_values=None, position_ids=position_ids,
    )

    with torch.no_grad():
        result_b = standalone_layer(
            layer_0_input.half(),
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
        )[0]

    print(f"Layer 0 output (B - shard weights): norm={result_b.norm():.3f}")
    diff_b = (layer_0_output_ref.float() - result_b.float()).abs()
    cos_b = torch.nn.functional.cosine_similarity(
        layer_0_output_ref.flatten().unsqueeze(0).float(),
        result_b.flatten().unsqueeze(0).float()).item()
    print(f"  max_abs vs ref={diff_b.max():.4f}  mean_abs={diff_b.mean():.4f}  cos={cos_b:.4f}")

    # (c) Standalone with weights pulled DIRECTLY from the loaded HF model
    hf_layer_0 = model.model.layers[0]
    print("\n--- Sanity check: same standalone setup but with weights cloned from HF model ---")

    fresh_layer = LlamaDecoderLayer(config, layer_idx=0)
    fresh_layer.load_state_dict(hf_layer_0.state_dict(), strict=True)
    fresh_layer.eval().half()

    with torch.no_grad():
        result_c = fresh_layer(
            layer_0_input.half(),
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
        )[0]
    print(f"Layer 0 output (C - direct HF weights): norm={result_c.norm():.3f}")
    diff_c = (layer_0_output_ref.float() - result_c.float()).abs()
    cos_c = torch.nn.functional.cosine_similarity(
        layer_0_output_ref.flatten().unsqueeze(0).float(),
        result_c.flatten().unsqueeze(0).float()).item()
    print(f"  max_abs vs ref={diff_c.max():.4f}  mean_abs={diff_c.mean():.4f}  cos={cos_c:.4f}")

    # Compare shard weights vs HF model weights for layer 0
    print("\n--- Weight comparison: shard file vs HF model (layer 0) ---")
    hf_state = hf_layer_0.state_dict()
    shard_keys = list(layer_state.keys())
    for k in sorted(shard_keys)[:5]:
        if k in hf_state:
            shard_w = layer_state[k]
            hf_w = hf_state[k]
            print(f"  {k}: shard.shape={list(shard_w.shape)} hf.shape={list(hf_w.shape)}")
            if shard_w.shape == hf_w.shape:
                d = (shard_w.float() - hf_w.float()).abs()
                print(f"      max_diff={d.max():.6f}  mean_diff={d.mean():.6f}  match={'YES' if d.max() < 1e-3 else 'NO'}")
        else:
            print(f"  {k}: MISSING in hf model state_dict")

    # And keys that exist in HF but not in shard
    print(f"\n  shard has {len(layer_state)} keys for layer 0")
    print(f"  hf model layer 0 has {len(hf_state)} keys")
    hf_only = set(hf_state.keys()) - set(layer_state.keys())
    shard_only = set(layer_state.keys()) - set(hf_state.keys())
    if hf_only:
        print(f"  Keys in HF but not shard: {sorted(hf_only)}")
    if shard_only:
        print(f"  Keys in shard but not HF: {sorted(shard_only)}")


if __name__ == "__main__":
    main()
