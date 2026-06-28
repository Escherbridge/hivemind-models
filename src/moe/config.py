"""
Mixture-of-Experts configuration.

Defines the core `MoEConfig` dataclass used throughout the MoE upcycling,
gating, and training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import yaml


@dataclass
class MoEConfig:
    """
    Configuration for converting a dense model into a Mixture-of-Experts model.

    These are generic MoE compute knobs (upcycling, gating, load-balancing).
    The former ``tool_categories`` field \u2014 which mapped category names to expert
    indices \u2014 has been removed; that domain-specialisation premise is broken
    (learned routers route by token syntax, not domain) and is archived at
    hivemind-archive/domain-moe-sdk/.

    Attributes:
        num_experts: Number of expert copies per MoE layer.
        top_k: Number of experts activated per token.
        moe_layers: Which layers to convert.  Accepts a list of integer
                     layer indices OR the string ``"all_mlp"`` to convert
                     every MLP layer.
        expert_capacity_factor: Multiplicative factor applied to the
                                theoretical per-expert token budget (used
                                for load-balancing).
        gating_type: Gating algorithm \u2014 ``"top_k"`` (tokens choose experts)
                     or ``"expert_choice"`` (experts choose tokens).
        base_model_id: HuggingFace model identifier for the dense base.
        noise_std: Standard deviation for Gaussian noise injected into
                   duplicated expert weights to break symmetry.
        load_balance_weight: Coefficient for the auxiliary load-balancing loss.
    """

    num_experts: int = 8
    top_k: int = 2
    moe_layers: list[int] | Literal["all_mlp"] = "all_mlp"
    expert_capacity_factor: float = 1.25
    gating_type: Literal["top_k", "expert_choice"] = "top_k"
    base_model_id: str = ""
    noise_std: float = 0.01
    load_balance_weight: float = 0.01

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "moe_layers": self.moe_layers,
            "expert_capacity_factor": self.expert_capacity_factor,
            "gating_type": self.gating_type,
            "base_model_id": self.base_model_id,
            "noise_std": self.noise_std,
            "load_balance_weight": self.load_balance_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoEConfig":
        moe_layers_raw = data.get("moe_layers", "all_mlp")
        if isinstance(moe_layers_raw, str):
            moe_layers: list[int] | Literal["all_mlp"] = "all_mlp"
        else:
            moe_layers = [int(x) for x in moe_layers_raw]

        return cls(
            num_experts=int(data.get("num_experts", 8)),
            top_k=int(data.get("top_k", 2)),
            moe_layers=moe_layers,
            expert_capacity_factor=float(data.get("expert_capacity_factor", 1.25)),
            gating_type=data.get("gating_type", "top_k"),
            base_model_id=data.get("base_model_id", ""),
            noise_std=float(data.get("noise_std", 0.01)),
            load_balance_weight=float(data.get("load_balance_weight", 0.01)),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MoEConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw.get("moe", raw))

    def save_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump({"moe": self.to_dict()}, fh, default_flow_style=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def resolve_moe_layer_indices(self, num_hidden_layers: int) -> list[int]:
        """
        Resolve ``moe_layers`` into concrete layer indices.

        If ``moe_layers == "all_mlp"`` every layer index ``[0, num_hidden_layers)``
        is returned.  Otherwise the stored list is returned as-is.
        """
        if self.moe_layers == "all_mlp":
            return list(range(num_hidden_layers))
        return list(self.moe_layers)
