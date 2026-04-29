"""
MoE upcycling: convert a dense HuggingFace model into a Mixture-of-Experts
model by duplicating MLP blocks and adding gating networks.

The upcycled model retains the original attention layers unmodified and
replaces selected MLP layers with ``MoELayer`` modules that contain *N*
expert copies plus a lightweight gating network.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn

from src.moe.config import MoEConfig
from src.moe.gating import ExpertChoiceGating, TopKGating

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MoE layer wrapper
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer that wraps *N* expert MLP copies and a gating
    network.

    During forward:
      1. Gating network selects top-k experts per token.
      2. Each selected expert processes the token.
      3. Outputs are combined using the gating weights.
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        gating: TopKGating | ExpertChoiceGating,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.gating = gating
        self.num_experts = len(experts)
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, S, H)

        Returns:
            output: (B, S, H)
            aux_loss: scalar load-balancing loss
        """
        B, S, H = hidden_states.shape

        if isinstance(self.gating, TopKGating):
            return self._forward_topk(hidden_states, B, S, H)
        else:
            return self._forward_expert_choice(hidden_states, B, S, H)

    # ------------------------------------------------------------------
    # Top-k routing
    # ------------------------------------------------------------------

    def _forward_topk(
        self,
        hidden_states: torch.Tensor,
        B: int,
        S: int,
        H: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_values, expert_indices, aux_loss = self.gating(hidden_states)
        # gate_values: (B, S, top_k)   expert_indices: (B, S, top_k)

        output = torch.zeros_like(hidden_states)

        # Flatten to (B*S, H)
        flat_hidden = hidden_states.view(-1, H)
        flat_output = output.view(-1, H)
        flat_gate = gate_values.view(-1, self.top_k)
        flat_indices = expert_indices.view(-1, self.top_k)

        for k in range(self.top_k):
            expert_ids = flat_indices[:, k]    # (B*S,)
            weights = flat_gate[:, k]           # (B*S,)

            for expert_idx in range(self.num_experts):
                mask = expert_ids == expert_idx
                if not mask.any():
                    continue
                tokens = flat_hidden[mask]
                expert_out = self.experts[expert_idx](tokens)
                flat_output[mask] += weights[mask].unsqueeze(-1) * expert_out

        return flat_output.view(B, S, H), aux_loss

    # ------------------------------------------------------------------
    # Expert-choice routing
    # ------------------------------------------------------------------

    def _forward_expert_choice(
        self,
        hidden_states: torch.Tensor,
        B: int,
        S: int,
        H: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_hidden = hidden_states.view(-1, H)  # (N, H)
        N = flat_hidden.shape[0]

        dispatch_mask, combine_weights, aux_loss = self.gating(flat_hidden)
        # dispatch_mask: (E, C), combine_weights: (E, C)

        output = torch.zeros(N, H, device=hidden_states.device, dtype=hidden_states.dtype)

        for expert_idx in range(self.num_experts):
            token_ids = dispatch_mask[expert_idx]   # (C,)
            weights = combine_weights[expert_idx]   # (C,)
            tokens = flat_hidden[token_ids]         # (C, H)
            expert_out = self.experts[expert_idx](tokens)
            output.index_add_(
                0,
                token_ids,
                weights.unsqueeze(-1) * expert_out,
            )

        return output.view(B, S, H), aux_loss


# ---------------------------------------------------------------------------
# MoE upcycler
# ---------------------------------------------------------------------------

class MoEUpcycler:
    """
    Convert a dense HuggingFace transformer into a MoE transformer.

    Steps:
      1. Identify MLP sub-modules in the target layers.
      2. For each target layer, duplicate the MLP *num_experts* times.
      3. Add a small gating network.
      4. Optionally inject Gaussian noise to break symmetry.
      5. Replace the original MLP with an ``MoELayer``.

    Usage::

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        config = MoEConfig(num_experts=8, top_k=2, moe_layers="all_mlp")
        upcycler = MoEUpcycler(model, config)
        moe_model = upcycler.upcycle()
    """

    # Common MLP attribute names across architectures
    _MLP_ATTR_NAMES = ("mlp", "feed_forward", "ff")

    def __init__(self, model: nn.Module, config: MoEConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upcycle(self) -> nn.Module:
        """
        Perform the full upcycling and return the modified model (in-place).
        """
        layer_indices = self._resolve_layers()
        logger.info(
            "Upcycling %d layers to MoE (%d experts, top-%d, gating=%s)",
            len(layer_indices),
            self.config.num_experts,
            self.config.top_k,
            self.config.gating_type,
        )

        transformer_layers = self._get_transformer_layers()
        if transformer_layers is None:
            raise RuntimeError(
                "Could not locate the transformer layer list in the model. "
                "Ensure the model follows a standard HuggingFace architecture."
            )

        for idx in layer_indices:
            layer = transformer_layers[idx]
            mlp_attr, mlp_module = self._find_mlp(layer)
            if mlp_module is None:
                logger.warning("Layer %d: no MLP sub-module found, skipping.", idx)
                continue

            hidden_size = self._infer_hidden_size(mlp_module)
            moe_layer = self._build_moe_layer(mlp_module, hidden_size)
            setattr(layer, mlp_attr, moe_layer)
            logger.debug("Layer %d: replaced '%s' with MoELayer.", idx, mlp_attr)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Upcycling complete. Total parameters: %s", f"{total_params:,}")
        return self.model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_layers(self) -> list[int]:
        """Determine which layer indices to upcycle."""
        transformer_layers = self._get_transformer_layers()
        num_layers = len(transformer_layers) if transformer_layers else 0
        return self.config.resolve_moe_layer_indices(num_layers)

    def _get_transformer_layers(self) -> nn.ModuleList | None:
        """
        Locate the ``nn.ModuleList`` of transformer blocks.

        Tries common HuggingFace patterns:
          - model.model.layers
          - model.transformer.h
          - model.model.decoder.layers
        """
        candidates = [
            ("model", "layers"),
            ("transformer", "h"),
            ("model", "decoder", "layers"),
        ]

        root = self.model
        for path in candidates:
            obj = root
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and isinstance(obj, nn.ModuleList):
                return obj

        # Last resort: walk children
        for child in root.children():
            if isinstance(child, nn.ModuleList):
                return child
            for grandchild in child.children():
                if isinstance(grandchild, nn.ModuleList):
                    return grandchild

        return None

    def _find_mlp(self, layer: nn.Module) -> tuple[str, nn.Module | None]:
        """Find the MLP sub-module inside a transformer layer."""
        for attr_name in self._MLP_ATTR_NAMES:
            mlp = getattr(layer, attr_name, None)
            if mlp is not None:
                return attr_name, mlp
        # Fallback: look for a child whose class name contains "MLP"
        for name, child in layer.named_children():
            if "mlp" in name.lower() or "MLP" in type(child).__name__:
                return name, child
        return "mlp", None

    @staticmethod
    def _infer_hidden_size(mlp: nn.Module) -> int:
        """
        Infer the hidden_size (input dimension) of the MLP.

        Tries common weight names: gate_proj, up_proj, fc1, w1, c_fc.
        """
        for attr in ("gate_proj", "up_proj", "fc1", "w1", "c_fc"):
            sub = getattr(mlp, attr, None)
            if sub is not None and hasattr(sub, "in_features"):
                return sub.in_features
            if sub is not None and hasattr(sub, "weight"):
                return sub.weight.shape[1]

        # Walk parameters and use the first 2-D weight's second dim
        for p in mlp.parameters():
            if p.dim() == 2:
                return p.shape[1]

        raise RuntimeError("Cannot infer hidden_size from MLP module.")

    def _build_moe_layer(
        self,
        original_mlp: nn.Module,
        hidden_size: int,
    ) -> MoELayer:
        """Create an MoELayer from the original MLP."""
        experts = nn.ModuleList()
        for i in range(self.config.num_experts):
            expert = copy.deepcopy(original_mlp)
            if self.config.noise_std > 0:
                self._inject_noise(expert, self.config.noise_std)
            experts.append(expert)

        # Build gating
        if self.config.gating_type == "expert_choice":
            gating: TopKGating | ExpertChoiceGating = ExpertChoiceGating(
                hidden_size=hidden_size,
                num_experts=self.config.num_experts,
                capacity_factor=self.config.expert_capacity_factor,
                top_k=self.config.top_k,
            )
        else:
            gating = TopKGating(
                hidden_size=hidden_size,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
                load_balance_weight=self.config.load_balance_weight,
            )

        return MoELayer(experts=experts, gating=gating, top_k=self.config.top_k)

    @staticmethod
    def _inject_noise(module: nn.Module, std: float) -> None:
        """Add Gaussian noise to every parameter to break symmetry."""
        with torch.no_grad():
            for param in module.parameters():
                param.add_(torch.randn_like(param) * std)
