"""
Deterministic Checksum Layer for HiveMind Model Integrity

Computes a deterministic checksum from intermediate activations during inference.
This checksum is used by the relay to verify peers run correct, unmodified weights.

Design decisions:
- SHA-256 of quantized (rounded) activations for cross-platform determinism
- Rounds to 4 decimal places to handle floating-point variance
- Captures activations at configurable layer indices
- Incremental computation minimizes memory overhead
- Thread-safe: each inference gets its own hasher state
"""

import hashlib
import struct
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ChecksumConfig:
    """Configuration for checksum computation."""
    checkpoint_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24, 28, 31])
    round_decimals: int = 4
    max_samples_per_layer: int = 4096
    sample_seed: int = 42
    algorithm: str = "sha256"


class ChecksumComputer:
    """
    Incremental checksum computation over model activations.
    Thread-safe: each instance maintains independent state.
    """

    def __init__(self, config: Optional[ChecksumConfig] = None):
        self.config = config or ChecksumConfig()
        self._hasher = hashlib.new(self.config.algorithm)
        self._layers_seen: List[int] = []
        self._finalized = False
        self._rng = np.random.RandomState(self.config.sample_seed)

    def update(self, layer_idx: int, activations: Union[np.ndarray, "torch.Tensor"]) -> None:
        if self._finalized:
            raise ValueError("Checksum already finalized. Create a new ChecksumComputer.")
        if layer_idx not in self.config.checkpoint_layers:
            return
        if layer_idx in self._layers_seen:
            raise ValueError(f"Layer {layer_idx} already recorded in this checksum")

        act_np = self._to_numpy(activations)
        flat = act_np.flatten().astype(np.float32)

        if len(flat) > self.config.max_samples_per_layer:
            indices = self._rng.choice(len(flat), size=self.config.max_samples_per_layer, replace=False)
            indices.sort()
            flat = flat[indices]

        rounded = np.round(flat, decimals=self.config.round_decimals)
        header = struct.pack('<II', layer_idx, len(rounded))
        self._hasher.update(header)
        self._hasher.update(rounded.tobytes())
        self._layers_seen.append(layer_idx)

    def finalize(self) -> str:
        if self._finalized:
            raise ValueError("Checksum already finalized.")
        if not self._layers_seen:
            raise ValueError("No layers recorded. Call update() before finalize().")

        footer = struct.pack(f'<I{len(self._layers_seen)}I', len(self._layers_seen), *self._layers_seen)
        self._hasher.update(footer)
        self._finalized = True
        return self._hasher.hexdigest()

    @property
    def layers_recorded(self) -> List[int]:
        return list(self._layers_seen)

    @property
    def is_complete(self) -> bool:
        return set(self.config.checkpoint_layers).issubset(set(self._layers_seen))

    def _to_numpy(self, tensor: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        if isinstance(tensor, np.ndarray):
            return tensor
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().float().numpy()
        except ImportError:
            pass
        return np.array(tensor, dtype=np.float32)


def compute_checksum(activations_by_layer: dict, config: Optional[ChecksumConfig] = None) -> str:
    """Compute checksum from a dict of {layer_idx: activations}."""
    computer = ChecksumComputer(config)
    for layer_idx in sorted(activations_by_layer.keys()):
        computer.update(layer_idx, activations_by_layer[layer_idx])
    return computer.finalize()


def verify_checksum(expected_checksum: str, activations_by_layer: dict, config: Optional[ChecksumConfig] = None) -> bool:
    """Verify that activations produce the expected checksum."""
    actual = compute_checksum(activations_by_layer, config)
    import hmac as hmac_module
    return hmac_module.compare_digest(actual, expected_checksum)
