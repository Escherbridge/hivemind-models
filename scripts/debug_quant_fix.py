"""
Quick verification: does GROUP-WISE INT4 (group_size=64) fix the accuracy
problem we saw with per-tensor INT4?

Approach:
  1. Load TinyLlama full HF model in fp16.
  2. For layer 0, take the original weights and run 3 variants:
       (a) Original fp16 (baseline)
       (b) Per-tensor INT4 quantized + dequantized (current bug)
       (c) Group-wise INT4 quantized + dequantized, group_size=64 (proposed fix)
  3. Compare layer 0 output cos sim vs reference for each variant.

If (c) gives cos sim > 0.99, the fix is correct and we regenerate shards.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    create_causal_mask,
)

from src.convert.quantize import quantize_tensor, QuantizationConfig


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0).float(),
        b.flatten().unsqueeze(0).float(),
    ).item()


def run_layer_with_weights(
    config, layer_idx: int, weights: dict, hidden: torch.Tensor,
) -> torch.Tensor:
    """Build a LlamaDecoderLayer with given weights and run it on hidden."""
    layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
    layer.load_state_dict(weights, strict=True)
    layer.eval().half()

    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rotary = LlamaRotaryEmbedding(config=config).eval()
    pos_emb = rotary(hidden.half(), position_ids)
    mask = create_causal_mask(
        config=config, inputs_embeds=hidden.half(),
        attention_mask=None, past_key_values=None, position_ids=position_ids,
    )
    with torch.no_grad():
        return layer(
            hidden.half(),
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
        )[0]


def quantize_weights(weights: dict, config: QuantizationConfig) -> dict:
    """Quantize+dequantize only 2D 'weight' tensors (skip layernorms, biases)."""
    out = {}
    for name, w in weights.items():
        if "weight" in name and w.dim() >= 2:
            out[name] = quantize_tensor(w, config)
        else:
            out[name] = w
    return out


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")

    print("Loading full HF model (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).eval()
    config = model.config

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)

    layer_input = out.hidden_states[0]
    layer0_ref = out.hidden_states[1]   # after layer 0
    final_logits_ref = out.logits

    # Original layer 0 weights (in fp16)
    layer0_weights_fp16 = {k: v.clone() for k, v in model.model.layers[0].state_dict().items()}

    print(f"\nReference: layer 0 output norm={layer0_ref.norm():.3f}")

    # --- (a) Baseline: same weights, our calling code ---
    result_a = run_layer_with_weights(config, 0, layer0_weights_fp16, layer_input)
    print(f"\n(a) fp16 baseline:")
    print(f"     norm={result_a.norm():.3f}  cos={cos_sim(layer0_ref, result_a):.6f}")

    # --- (b) Per-tensor INT4 (current bug) ---
    qcfg_b = QuantizationConfig(bits=4, group_size=None)
    w_b = quantize_weights(layer0_weights_fp16, qcfg_b)
    result_b = run_layer_with_weights(config, 0, w_b, layer_input)
    print(f"\n(b) Per-tensor INT4 (current):")
    print(f"     norm={result_b.norm():.3f}  cos={cos_sim(layer0_ref, result_b):.6f}")

    # --- (c) Group-wise INT4 (proposed fix), group_size=64 ---
    qcfg_c = QuantizationConfig(bits=4, group_size=64)
    w_c = quantize_weights(layer0_weights_fp16, qcfg_c)
    result_c = run_layer_with_weights(config, 0, w_c, layer_input)
    print(f"\n(c) Group-wise INT4 (group=64):")
    print(f"     norm={result_c.norm():.3f}  cos={cos_sim(layer0_ref, result_c):.6f}")

    # --- (d) Group-wise INT4, group_size=128 ---
    qcfg_d = QuantizationConfig(bits=4, group_size=128)
    w_d = quantize_weights(layer0_weights_fp16, qcfg_d)
    result_d = run_layer_with_weights(config, 0, w_d, layer_input)
    print(f"\n(d) Group-wise INT4 (group=128):")
    print(f"     norm={result_d.norm():.3f}  cos={cos_sim(layer0_ref, result_d):.6f}")

    # --- (e) Per-tensor INT8 for comparison ---
    qcfg_e = QuantizationConfig(bits=8, group_size=None)
    w_e = quantize_weights(layer0_weights_fp16, qcfg_e)
    result_e = run_layer_with_weights(config, 0, w_e, layer_input)
    print(f"\n(e) Per-tensor INT8:")
    print(f"     norm={result_e.norm():.3f}  cos={cos_sim(layer0_ref, result_e):.6f}")

    # --- Per-weight error fingerprint comparison ---
    print("\n--- mlp.down_proj.weight quant error per scheme ---")
    orig = layer0_weights_fp16["mlp.down_proj.weight"]
    for label, w in [("b-pertensor-int4", w_b["mlp.down_proj.weight"]),
                     ("c-group64-int4",   w_c["mlp.down_proj.weight"]),
                     ("d-group128-int4",  w_d["mlp.down_proj.weight"]),
                     ("e-pertensor-int8", w_e["mlp.down_proj.weight"])]:
        d = (orig.float() - w.float()).abs()
        print(f"  {label:>20s}: max={d.max():.5f}  mean={d.mean():.5f}")


if __name__ == "__main__":
    main()
