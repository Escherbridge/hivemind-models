"""
Base interface for architecture handlers.

A handler is the *only* place that imports model-family-specific classes
(e.g. LlamaDecoderLayer, GraniteMoeHybridDecoderLayer). The shard server
talks to handlers through this interface and stays architecture-agnostic.

A handler reconstructs three kinds of shards:
  - `embed`  : input token embedding (rare extras like learned position embs
               are handled internally by the handler).
  - `layers` : a contiguous range of decoder layers (e.g. 4..10 inclusive).
  - `head`   : final layernorm + lm_head projection.

For each shard kind it builds the torch module(s), and provides a forward
function that takes the hidden states (or token IDs, for embed) and returns
the next-stage tensor. The forward function is responsible for any
per-family setup such as the causal attention mask, rotary embeddings,
MoE routing, or Mamba state initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch


# A ShardModule is whatever the handler wants to stash in server state:
# a single nn.Module, a tuple, a dict — opaque to the shard server.
ShardModule = Any


@dataclass
class LayerShardSpec:
    """What the shard server hands to build_layers."""

    layer_start: int  # inclusive
    layer_end: int    # inclusive


class ArchitectureHandler(Protocol):
    """
    Per-architecture handler. Concrete implementations live in sibling modules
    (llama.py, granite.py, …) and self-register via register_handler().

    Method contracts:
      - build_* receives the loaded HF config, the shard's state_dict (already
        loaded from safetensors), and the target device. Returns whatever
        the handler wants to keep around for forward(); the server stores it
        opaquely in server_state['module'].
      - forward_* receives the unpacked input tensor (already on `device`)
        and the previously-built module. Returns the output tensor that the
        server will pack and ship to the next shard. Must run under
        torch.no_grad() (the server handles that).
    """

    # Identity ------------------------------------------------------------

    name: str
    """Short identifier matching the value used in registry.register_handler()."""

    # Builders ------------------------------------------------------------

    def build_embed(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
    ) -> ShardModule:
        """Reconstruct the input-embedding shard module."""
        ...

    def build_layers(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        spec: LayerShardSpec,
        device: torch.device,
    ) -> ShardModule:
        """Reconstruct a decoder-layer-range shard module."""
        ...

    def build_head(
        self,
        config: Any,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
    ) -> ShardModule:
        """Reconstruct the final-norm + lm_head shard module."""
        ...

    # Forward passes ------------------------------------------------------

    def forward_embed(self, module: ShardModule, token_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs (int64) → embedded hidden states (float16/32)."""
        ...

    def forward_layers(self, module: ShardModule, hidden: torch.Tensor) -> torch.Tensor:
        """Hidden states → hidden states, through every layer in the shard."""
        ...

    def forward_head(self, module: ShardModule, hidden: torch.Tensor) -> torch.Tensor:
        """Hidden states → vocab logits."""
        ...


# Convenience: a tiny helper so server code can dispatch by shard type
# without an if/elif ladder in the hot path.
ShardForward = Callable[[ShardModule, torch.Tensor], torch.Tensor]


def get_forward_fn(handler: ArchitectureHandler, shard_type: str) -> ShardForward:
    """Return the right forward_* method on `handler` for `shard_type`."""
    if shard_type == "embed":
        return handler.forward_embed
    if shard_type == "layers":
        return handler.forward_layers
    if shard_type == "head":
        return handler.forward_head
    raise ValueError(f"Unknown shard_type: {shard_type!r}")
