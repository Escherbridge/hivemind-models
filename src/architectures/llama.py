"""
Llama-family handler (Llama 1/2/3, Mistral, TinyLlama — anything that uses
`transformers.models.llama.modeling_llama.LlamaDecoderLayer`).

This is the reference implementation of the ArchitectureHandler interface.
It mirrors the verified-correct logic from the original shard_server.py
that produces "Paris" for "The capital of France is" through the swarm.

Two non-obvious requirements that earlier bugs taught us:

  1. The decoder layer needs a causal attention_mask. Passing None gives
     bidirectional attention and produces garbage. See create_causal_mask.
  2. Position ids restart at 0 in every layer-group shard. RoPE is computed
     fresh each layer, so this is correct (a layer group has no notion of
     its absolute position in the model).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from src.architectures.base import ArchitectureHandler, LayerShardSpec, ShardModule
from src.architectures.registry import register_handler


logger = logging.getLogger("architectures.llama")


@register_handler(
    "llama",
    aliases=["mistral", "tinyllama"],  # share the decoder block shape
)
class LlamaHandler:
    name = "llama"

    # ---- builders ----------------------------------------------------------

    def build_embed(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
    ) -> ShardModule:
        key = "model.embed_tokens.weight"
        if key not in state_dict:
            raise RuntimeError(f"Embedding weight {key!r} not found in shard")
        embed = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
        embed.weight.data = state_dict[key].to(device)
        embed.eval()
        logger.info("Loaded embedding layer")
        return embed

    def build_layers(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        spec: LayerShardSpec,
        device: torch.device,
    ) -> ShardModule:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRotaryEmbedding,
        )

        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"

        layers = nn.ModuleList()
        for idx in range(spec.layer_start, spec.layer_end + 1):
            layer = LlamaDecoderLayer(config, layer_idx=idx).to(device)
            prefix = f"model.layers.{idx}."
            layer_state = {
                k.removeprefix(prefix): v.to(device)
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if not layer_state:
                raise RuntimeError(f"No weights found for layer {idx}")
            layer.load_state_dict(layer_state, strict=False)
            layer.eval()
            layers.append(layer)
            logger.info(f"Loaded layer {idx} ({len(layer_state)} tensors)")

        rotary_emb = LlamaRotaryEmbedding(config=config).to(device)
        rotary_emb.eval()

        # Return as a dict so we can pass `config` along to forward_layers
        # without storing it elsewhere.
        return {
            "layers": layers,
            "rotary_emb": rotary_emb,
            "config": config,
        }

    def build_head(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
    ) -> ShardModule:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)
        if "model.norm.weight" in state_dict:
            norm.weight.data = state_dict["model.norm.weight"].to(device)
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device)
        if "lm_head.weight" in state_dict:
            head.weight.data = state_dict["lm_head.weight"].to(device)
        norm.eval()
        head.eval()
        logger.info("Loaded head (norm + lm_head)")
        return {"norm": norm, "head": head}

    # ---- forward passes ----------------------------------------------------

    def forward_embed(self, module: ShardModule, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype in (torch.float16, torch.float32, torch.bfloat16):
            token_ids = token_ids.long()
        with torch.no_grad():
            return module(token_ids)

    def forward_layers(self, module: ShardModule, hidden: torch.Tensor) -> torch.Tensor:
        from transformers.models.llama.modeling_llama import create_causal_mask

        layers = module["layers"]
        rotary_emb = module["rotary_emb"]
        config = module["config"]

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
        return hidden

    def forward_head(self, module: ShardModule, hidden: torch.Tensor) -> torch.Tensor:
        norm = module["norm"]
        head = module["head"]
        hidden = hidden.half()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        with torch.no_grad():
            normed = norm(hidden)
            return head(normed)
