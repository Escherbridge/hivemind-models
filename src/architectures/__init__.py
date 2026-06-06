"""
Architecture handlers for the shard pipeline.

Each handler knows how to:
  - Reconstruct an embedding / decoder-layer / head module from raw safetensors
    weights for a specific model family (Llama, Granite-MoE-Hybrid, etc).
  - Run a forward pass through a layer-group shard, including any per-family
    setup (causal mask, rotary embeddings, MoE gating, Mamba state).

The shard server resolves the right handler from the loaded model's
`config.model_type` and delegates all architecture-specific work to it.
"""

from src.architectures.base import ArchitectureHandler, LayerShardSpec, ShardModule
from src.architectures.registry import get_handler, register_handler, available_handlers

# Importing the concrete handlers registers them via @register_handler decorator.
from src.architectures import llama  # noqa: F401  side effect: register
from src.architectures import granite  # noqa: F401  side effect: register

__all__ = [
    "ArchitectureHandler",
    "LayerShardSpec",
    "ShardModule",
    "get_handler",
    "register_handler",
    "available_handlers",
]
