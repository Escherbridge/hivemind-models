"""
BitLLM quantization utilities.

Implements 1-bit (binary) and 2-bit (ternary) weight quantization
for efficient storage and WebGPU dequantization.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BitLLMConfig:
    """Configuration for BitLLM quantization."""
    bit_width: Literal[1, 2] = 1  # 1-bit binary or 2-bit ternary
    group_size: int = 64  # Number of weights per scale factor


@dataclass
class BitLLMTensor:
    """Packed BitLLM tensor representation."""
    packed_weights: torch.Tensor  # uint8, packed bits
    scales: torch.Tensor  # float16, per-group scales
    zeros: torch.Tensor | None  # float16, for ternary only
    original_shape: tuple[int, ...]
    bit_width: int
    group_size: int


def quantize_to_1bit(
    tensor: torch.Tensor,
    group_size: int = 64
) -> BitLLMTensor:
    """
    Quantize tensor to 1-bit (binary: -1 or +1).

    Binary quantization: w_q = sign(w) * scale
    where scale = mean(|w|) per group

    Args:
        tensor: Input tensor to quantize
        group_size: Number of weights per scale factor

    Returns:
        BitLLMTensor with packed weights and scales
    """
    original_shape = tensor.shape
    tensor = tensor.float()

    # Reshape for grouping
    flat = tensor.view(-1)
    original_numel = flat.numel()

    if original_numel % group_size != 0:
        # Pad to multiple of group_size
        pad_size = group_size - (original_numel % group_size)
        flat = torch.nn.functional.pad(flat, (0, pad_size))

    grouped = flat.view(-1, group_size)
    num_groups = grouped.shape[0]

    # Compute scales (mean absolute value per group)
    scales = grouped.abs().mean(dim=1).half()

    # Binarize: 1 if positive or zero, 0 if negative
    binary = (grouped >= 0).to(torch.uint8)

    # Pack 8 bits into each uint8
    packed = pack_bits(binary.view(-1))

    logger.debug(
        f"1-bit quantization: {original_shape} -> "
        f"packed={packed.shape}, scales={scales.shape}"
    )

    return BitLLMTensor(
        packed_weights=packed,
        scales=scales,
        zeros=None,
        original_shape=original_shape,
        bit_width=1,
        group_size=group_size
    )


def quantize_to_ternary(
    tensor: torch.Tensor,
    group_size: int = 64
) -> BitLLMTensor:
    """
    Quantize tensor to ternary (-1, 0, +1).

    Ternary quantization with threshold:
    w_q = -1 if w < -threshold, 0 if |w| < threshold, +1 if w > threshold

    Args:
        tensor: Input tensor to quantize
        group_size: Number of weights per scale factor

    Returns:
        BitLLMTensor with packed weights, scales, and thresholds
    """
    original_shape = tensor.shape
    tensor = tensor.float()

    flat = tensor.view(-1)
    original_numel = flat.numel()

    if original_numel % group_size != 0:
        pad_size = group_size - (original_numel % group_size)
        flat = torch.nn.functional.pad(flat, (0, pad_size))

    grouped = flat.view(-1, group_size)
    num_groups = grouped.shape[0]

    # Compute threshold (0.7 * mean absolute value per group)
    abs_mean = grouped.abs().mean(dim=1, keepdim=True)
    threshold = 0.7 * abs_mean

    # Compute scales
    scales = abs_mean.squeeze().half()

    # Ternarize: 0 = zero, 1 = positive, 2 = negative (as 2-bit values)
    ternary = torch.zeros_like(grouped, dtype=torch.uint8)
    ternary[grouped > threshold] = 1
    ternary[grouped < -threshold] = 2

    # Pack 4 ternary values (2 bits each) into each uint8
    packed = pack_ternary(ternary.view(-1))

    logger.debug(
        f"Ternary quantization: {original_shape} -> "
        f"packed={packed.shape}, scales={scales.shape}"
    )

    return BitLLMTensor(
        packed_weights=packed,
        scales=scales,
        zeros=threshold.squeeze().half(),
        original_shape=original_shape,
        bit_width=2,
        group_size=group_size
    )


def pack_bits(binary: torch.Tensor) -> torch.Tensor:
    """
    Pack binary tensor (0/1 values) into uint8, 8 bits per byte.

    Args:
        binary: Tensor with 0/1 values

    Returns:
        Packed uint8 tensor with 1/8 the elements
    """
    binary = binary.view(-1)
    # Pad to multiple of 8
    if binary.numel() % 8 != 0:
        pad_size = 8 - (binary.numel() % 8)
        binary = torch.nn.functional.pad(binary, (0, pad_size))

    binary = binary.view(-1, 8)
    # Pack bits: bit 0 is LSB
    packed = torch.zeros(binary.shape[0], dtype=torch.uint8, device=binary.device)
    for i in range(8):
        packed |= (binary[:, i].to(torch.uint8) << i)

    return packed


def pack_ternary(ternary: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary tensor (0/1/2 values) into uint8, 4 values per byte.

    Args:
        ternary: Tensor with 0/1/2 values

    Returns:
        Packed uint8 tensor with 1/4 the elements
    """
    ternary = ternary.view(-1)
    # Pad to multiple of 4
    if ternary.numel() % 4 != 0:
        pad_size = 4 - (ternary.numel() % 4)
        ternary = torch.nn.functional.pad(ternary, (0, pad_size))

    ternary = ternary.view(-1, 4)
    # Pack: 2 bits per value
    packed = torch.zeros(ternary.shape[0], dtype=torch.uint8, device=ternary.device)
    for i in range(4):
        packed |= (ternary[:, i].to(torch.uint8) << (i * 2))

    return packed


def unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 to binary tensor.

    Args:
        packed: Packed uint8 tensor

    Returns:
        Binary tensor with 8x the elements
    """
    packed = packed.view(-1)
    result = torch.zeros(packed.numel() * 8, dtype=torch.uint8, device=packed.device)
    for i in range(8):
        result[i::8] = (packed >> i) & 1
    return result


def unpack_ternary(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack uint8 to ternary tensor.

    Args:
        packed: Packed uint8 tensor

    Returns:
        Ternary tensor with 4x the elements
    """
    packed = packed.view(-1)
    result = torch.zeros(packed.numel() * 4, dtype=torch.uint8, device=packed.device)
    for i in range(4):
        result[i::4] = (packed >> (i * 2)) & 3
    return result


def dequantize_1bit(bitllm: BitLLMTensor) -> torch.Tensor:
    """
    Dequantize 1-bit tensor back to float16.

    Args:
        bitllm: BitLLMTensor to dequantize

    Returns:
        Dequantized float16 tensor in original shape
    """
    # Unpack bits
    unpacked = unpack_bits(bitllm.packed_weights)

    # Convert 0/1 to -1/+1
    weights = unpacked.float() * 2 - 1

    # Reshape for group scaling
    weights = weights.view(-1, bitllm.group_size)

    # Apply scales
    weights = weights * bitllm.scales.float().unsqueeze(1)

    # Reshape to original, trimming padding
    weights = weights.view(-1)
    original_numel = int(np.prod(bitllm.original_shape))
    weights = weights[:original_numel]

    return weights.view(bitllm.original_shape).half()


def dequantize_ternary(bitllm: BitLLMTensor) -> torch.Tensor:
    """
    Dequantize ternary tensor back to float16.

    Args:
        bitllm: BitLLMTensor to dequantize

    Returns:
        Dequantized float16 tensor in original shape
    """
    # Unpack ternary values
    unpacked = unpack_ternary(bitllm.packed_weights)

    # Convert 0/1/2 to 0/+1/-1
    weights = torch.zeros_like(unpacked, dtype=torch.float32)
    weights[unpacked == 1] = 1.0
    weights[unpacked == 2] = -1.0

    # Reshape for group scaling
    weights = weights.view(-1, bitllm.group_size)

    # Apply scales
    weights = weights * bitllm.scales.float().unsqueeze(1)

    # Reshape to original, trimming padding
    weights = weights.view(-1)
    original_numel = int(np.prod(bitllm.original_shape))
    weights = weights[:original_numel]

    return weights.view(bitllm.original_shape).half()


def dequantize(bitllm: BitLLMTensor) -> torch.Tensor:
    """
    Dequantize BitLLM tensor (auto-detects bit width).

    Args:
        bitllm: BitLLMTensor to dequantize

    Returns:
        Dequantized float16 tensor in original shape
    """
    if bitllm.bit_width == 1:
        return dequantize_1bit(bitllm)
    elif bitllm.bit_width == 2:
        return dequantize_ternary(bitllm)
    else:
        raise ValueError(f"Unsupported bit width: {bitllm.bit_width}")


def tensor_to_safetensors_dict(
    bitllm: BitLLMTensor,
    base_name: str
) -> dict[str, torch.Tensor]:
    """
    Convert BitLLMTensor to dictionary for safetensors storage.

    Returns tensors with naming convention:
    - {base_name}.packed -> packed weights
    - {base_name}.scales -> per-group scales
    - {base_name}.zeros -> zeros/thresholds (ternary only)
    - {base_name}.meta -> metadata tensor (shape, bit_width, group_size)

    Args:
        bitllm: BitLLMTensor to convert
        base_name: Base name for the tensor keys

    Returns:
        Dictionary of tensor name to tensor for safetensors storage
    """
    # Store shape info as int32 tensor
    shape_tensor = torch.tensor(
        list(bitllm.original_shape) + [bitllm.bit_width, bitllm.group_size],
        dtype=torch.int32
    )

    result = {
        f"{base_name}.packed": bitllm.packed_weights,
        f"{base_name}.scales": bitllm.scales,
        f"{base_name}.meta": shape_tensor,
    }

    if bitllm.zeros is not None:
        result[f"{base_name}.zeros"] = bitllm.zeros

    return result


def safetensors_dict_to_tensor(
    tensors: dict[str, torch.Tensor],
    base_name: str
) -> BitLLMTensor:
    """
    Reconstruct BitLLMTensor from safetensors dictionary.

    Args:
        tensors: Dictionary loaded from safetensors
        base_name: Base name for the tensor keys

    Returns:
        Reconstructed BitLLMTensor
    """
    packed = tensors[f"{base_name}.packed"]
    scales = tensors[f"{base_name}.scales"]
    meta = tensors[f"{base_name}.meta"].tolist()

    # Meta format: [*shape, bit_width, group_size]
    group_size = meta[-1]
    bit_width = meta[-2]
    original_shape = tuple(meta[:-2])

    zeros = tensors.get(f"{base_name}.zeros")

    return BitLLMTensor(
        packed_weights=packed,
        scales=scales,
        zeros=zeros,
        original_shape=original_shape,
        bit_width=bit_width,
        group_size=group_size
    )


def quantize_tensor_bitllm(
    tensor: torch.Tensor,
    config: BitLLMConfig
) -> BitLLMTensor:
    """
    Quantize tensor using BitLLM method.

    Args:
        tensor: Input tensor to quantize
        config: BitLLM configuration

    Returns:
        Quantized BitLLMTensor
    """
    if config.bit_width == 1:
        return quantize_to_1bit(tensor, config.group_size)
    elif config.bit_width == 2:
        return quantize_to_ternary(tensor, config.group_size)
    else:
        raise ValueError(f"Unsupported bit width: {config.bit_width}")


def estimate_bitllm_size(
    tensor: torch.Tensor,
    bit_width: int = 1,
    group_size: int = 64
) -> dict[str, int]:
    """
    Estimate the size of a BitLLM-quantized tensor.

    Args:
        tensor: Input tensor
        bit_width: Quantization bit width (1 or 2)
        group_size: Quantization group size

    Returns:
        Dictionary with size estimates in bytes
    """
    numel = tensor.numel()
    original_bytes = numel * tensor.element_size()

    if bit_width == 1:
        # 1 bit per weight, packed into uint8
        packed_bytes = (numel + 7) // 8
    else:
        # 2 bits per weight, packed into uint8
        packed_bytes = (numel + 3) // 4

    # Scales: float16 per group
    n_groups = (numel + group_size - 1) // group_size
    scale_bytes = n_groups * 2

    # Zeros (for ternary only): float16 per group
    zeros_bytes = n_groups * 2 if bit_width == 2 else 0

    total_bytes = packed_bytes + scale_bytes + zeros_bytes
    compression_ratio = original_bytes / total_bytes if total_bytes > 0 else 0

    return {
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "zeros_bytes": zeros_bytes,
        "total_bytes": total_bytes,
        "compression_ratio": compression_ratio,
        "bit_width": bit_width,
        "group_size": group_size,
    }
