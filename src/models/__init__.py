"""
Models module.

Model architecture detection and handling is done generically via
transformers.AutoModelForCausalLM. The MoE upcycler (src/moe/upcycle.py)
auto-detects layer structure, MLP location, and hidden sizes.

Utilities:
    - checksum_layer: Layer-level checksum computation for shard verification.
"""

from src.models.checksum_layer import ChecksumComputer, verify_checksum

__all__ = [
    "ChecksumComputer",
    "verify_checksum",
]
