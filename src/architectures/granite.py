"""
Granite-MoE-Hybrid handler (Granite 4.0-H series).

Handles the ibm-granite/granite-4.0-h-tiny model with mixed Mamba/Attention
decoder layers and MoE routing. The model uses:
  - No rotary position embeddings (position_embedding_type = "nope")
  - embedding_multiplier on input embeddings
  - logits_scaling on output logits
  - residual_multiplier inside each decoder layer (handled by the HF layer)
  - Tied embedding/lm_head weights

Two non-obvious requirements from earlier LlamaHandler bugs:
  1. Attention layers REQUIRE a causal attention_mask. Passing None gives
     bidirectional attention and produces garbage.
  2. INT4 quantization must be group-wise (group_size=64), not per-tensor.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from src.architectures.base import ArchitectureHandler, LayerShardSpec, ShardModule
from src.architectures.registry import register_handler


logger = logging.getLogger("architectures.granite")


@register_handler(
    "granitemoehybrid",
    aliases=["granitemoe", "granite"],
)
class GraniteHandler:
    name = "granitemoehybrid"

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
        embedding_multiplier = getattr(config, "embedding_multiplier", 1.0)
        logger.info(
            "Loaded embedding layer (embedding_multiplier=%.4f)", embedding_multiplier
        )
        return {"embed": embed, "embedding_multiplier": embedding_multiplier}

    def build_layers(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        spec: LayerShardSpec,
        device: torch.device,
    ) -> ShardModule:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridDecoderLayer,
        )

        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"

        layers = nn.ModuleList()
        layer_types: list[str] = []
        for idx in range(spec.layer_start, spec.layer_end + 1):
            layer = GraniteMoeHybridDecoderLayer(config, layer_idx=idx).to(device)
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
            layer_types.append(config.layers_block_type[idx])
            logger.info(
                "Loaded layer %d (%s, %d tensors)",
                idx,
                config.layers_block_type[idx],
                len(layer_state),
            )

        return {
            "layers": layers,
            "layer_types": layer_types,
            "config": config,
        }

    def build_head(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
    ) -> ShardModule:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridRMSNorm,
        )

        norm = GraniteMoeHybridRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        ).to(device)
        if "model.norm.weight" in state_dict:
            norm.weight.data = state_dict["model.norm.weight"].to(device)

        # lm_head is tied to embed_tokens
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device)
        if "lm_head.weight" in state_dict:
            head.weight.data = state_dict["lm_head.weight"].to(device)
        elif "model.embed_tokens.weight" in state_dict:
            head.weight.data = state_dict["model.embed_tokens.weight"].to(device)
        else:
            raise RuntimeError("Neither lm_head.weight nor model.embed_tokens.weight found in head shard")

        logits_scaling = getattr(config, "logits_scaling", 1.0)
        norm.eval()
        head.eval()
        logger.info("Loaded head (norm + lm_head, logits_scaling=%.4f)", logits_scaling)
        return {"norm": norm, "head": head, "logits_scaling": logits_scaling}

    # ---- forward passes ----------------------------------------------------

    def forward_embed(
        self, module: ShardModule, token_ids: torch.Tensor
    ) -> torch.Tensor:
        embed = module["embed"]
        embedding_multiplier = module["embedding_multiplier"]
        if token_ids.dtype in (torch.float16, torch.float32, torch.bfloat16):
            token_ids = token_ids.long()
        with torch.no_grad():
            hidden = embed(token_ids)
            if embedding_multiplier != 1.0:
                hidden = hidden * embedding_multiplier
        return hidden

    def forward_layers(
        self, module: ShardModule, hidden: torch.Tensor
    ) -> torch.Tensor:
        from transformers.masking_utils import create_causal_mask

        layers = module["layers"]
        layer_types = module["layer_types"]
        config = module["config"]

        hidden = hidden.half()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # Build per-layer-type causal masks, mirroring GraniteMoeHybridModel.forward
        position_ids = torch.arange(
            hidden.shape[1], device=hidden.device
        ).unsqueeze(0)

        # Pre-compute masks for each unique layer type
        causal_mask_mapping: dict[str, torch.Tensor | None] = {}
        unique_types = set(layer_types)
        for layer_type in unique_types:
            if "mamba" in layer_type:
                # For mamba layers in single-sequence non-cached inference,
                # the mask is None (no attention mask needed)
                causal_mask_mapping[layer_type] = None
            else:
                causal_mask_mapping[layer_type] = create_causal_mask(
                    config=config,
                    inputs_embeds=hidden,
                    attention_mask=None,
                    past_key_values=None,
                )

        # No rotary embeddings for Granite (position_embedding_type = "nope")
        position_embeddings = None

        with torch.no_grad():
            for layer, layer_type in zip(layers, layer_types):
                mask = causal_mask_mapping[layer_type]
                result = layer(
                    hidden,
                    attention_mask=mask,
                    position_embeddings=position_embeddings,
                )
                hidden = result
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
        return hidden

    def forward_head(
        self, module: ShardModule, hidden: torch.Tensor
    ) -> torch.Tensor:
        norm = module["norm"]
        head = module["head"]
        logits_scaling = module["logits_scaling"]
        hidden = hidden.half()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        with torch.no_grad():
            normed = norm(hidden)
            logits = head(normed)
            if logits_scaling != 1.0:
                logits = logits / logits_scaling
        return logits