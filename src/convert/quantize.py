"""
Quantization utilities for model weights.

Supports INT8 and INT4 quantization using absmax method.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class QuantMethod(Enum):
    """Quantization method."""

    ABSMAX = "absmax"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    BITLLM_1BIT = "bitllm_1bit"
    BITLLM_TERNARY = "bitllm_ternary"


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    bits: int = 4
    method: QuantMethod = QuantMethod.ABSMAX
    group_size: int | None = None  # None means per-tensor quantization

    def __post_init__(self):
        # BitLLM methods have different valid bit widths
        if self.method in (QuantMethod.BITLLM_1BIT, QuantMethod.BITLLM_TERNARY):
            valid_bits = {QuantMethod.BITLLM_1BIT: 1, QuantMethod.BITLLM_TERNARY: 2}
            expected = valid_bits[self.method]
            if self.bits != expected:
                raise ValueError(
                    f"BitLLM method {self.method.value} requires bits={expected}, got {self.bits}"
                )
            if self.group_size is None:
                self.group_size = 64  # Default group size for BitLLM
        elif self.bits not in (4, 8):
            raise ValueError(f"Quantization bits must be 4 or 8, got {self.bits}")


@dataclass
class QuantizedTensor:
    """Wrapper for quantized tensor data."""

    quantized: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]
    bits: int


def compute_absmax_scales(
    tensor: torch.Tensor,
    bits: int,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scales for absmax quantization.

    Args:
        tensor: Input tensor to quantize
        bits: Number of bits (4 or 8)
        group_size: Group size for group-wise quantization

    Returns:
        Tuple of (scales, reshaped_tensor)
    """
    qmax = 2 ** (bits - 1) - 1

    if group_size is None:
        # Per-tensor quantization
        absmax = tensor.abs().max()
        scales = absmax / qmax
        return scales.unsqueeze(0), tensor
    else:
        # Group-wise quantization
        original_shape = tensor.shape
        tensor_flat = tensor.reshape(-1)

        # Pad to multiple of group_size
        numel = tensor_flat.numel()
        padded_size = ((numel + group_size - 1) // group_size) * group_size
        if padded_size > numel:
            tensor_flat = torch.nn.functional.pad(
                tensor_flat, (0, padded_size - numel), value=0
            )

        # Reshape into groups
        tensor_grouped = tensor_flat.reshape(-1, group_size)

        # Compute per-group scales
        absmax = tensor_grouped.abs().max(dim=1).values
        scales = absmax / qmax

        return scales, tensor_grouped


def quantize_absmax(
    tensor: torch.Tensor,
    bits: int,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor using absmax method.

    Args:
        tensor: Input tensor
        bits: Number of bits (4 or 8)
        group_size: Group size for group-wise quantization

    Returns:
        Tuple of (quantized_tensor, scales)
    """
    scales, tensor_reshaped = compute_absmax_scales(tensor, bits, group_size)
    qmax = 2 ** (bits - 1) - 1

    if group_size is None:
        # Per-tensor quantization
        if scales.item() == 0:
            quantized = torch.zeros_like(tensor, dtype=torch.int8)
        else:
            quantized = torch.clamp(
                torch.round(tensor / scales.item()), -qmax - 1, qmax
            ).to(torch.int8)
    else:
        # Group-wise quantization
        scales_expanded = scales.unsqueeze(1)
        safe_scales = torch.where(
            scales_expanded == 0,
            torch.ones_like(scales_expanded),
            scales_expanded,
        )
        quantized = torch.clamp(
            torch.round(tensor_reshaped / safe_scales), -qmax - 1, qmax
        ).to(torch.int8)

    return quantized, scales


def dequantize_absmax(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple[int, ...],
    group_size: int | None = None,
) -> torch.Tensor:
    """
    Dequantize tensor from absmax quantization.

    Args:
        quantized: Quantized tensor
        scales: Scale factors
        original_shape: Original tensor shape
        group_size: Group size used for quantization

    Returns:
        Dequantized tensor
    """
    if group_size is None:
        # Per-tensor dequantization
        dequantized = quantized.float() * scales.item()
    else:
        # Group-wise dequantization
        scales_expanded = scales.unsqueeze(1)
        dequantized = (quantized.float() * scales_expanded).reshape(-1)

        # Remove padding
        numel = 1
        for dim in original_shape:
            numel *= dim
        dequantized = dequantized[:numel]

    return dequantized.reshape(original_shape)


def quantize_tensor(
    tensor: torch.Tensor,
    config: QuantizationConfig,
) -> torch.Tensor:
    """
    Quantize a tensor and return the quantized version.

    For simplicity in safetensors storage, this returns a float16 tensor
    that has been quantized and dequantized. This reduces precision but
    maintains compatibility with standard safetensors format.

    For true INT4/INT8 storage, use quantize_tensor_full which returns
    the QuantizedTensor with separate scales.

    For BitLLM methods, this returns the dequantized float16 result.
    Use quantize_tensor_bitllm_full for packed storage.

    Args:
        tensor: Input tensor to quantize
        config: Quantization configuration

    Returns:
        Dequantized tensor in float16 (precision-reduced)
    """
    # Handle BitLLM methods
    if config.method == QuantMethod.BITLLM_1BIT:
        from src.convert.bitllm import quantize_to_1bit, dequantize_1bit
        bitllm_tensor = quantize_to_1bit(tensor, config.group_size or 64)
        return dequantize_1bit(bitllm_tensor)

    if config.method == QuantMethod.BITLLM_TERNARY:
        from src.convert.bitllm import quantize_to_ternary, dequantize_ternary
        bitllm_tensor = quantize_to_ternary(tensor, config.group_size or 64)
        return dequantize_ternary(bitllm_tensor)

    if config.method != QuantMethod.ABSMAX:
        raise NotImplementedError(f"Method {config.method} not implemented")

    tensor_f32 = tensor.float()

    quantized, scales = quantize_absmax(tensor_f32, config.bits, config.group_size)
    dequantized = dequantize_absmax(quantized, scales, tensor.shape, config.group_size)

    # Return in float16 for storage efficiency
    return dequantized.to(torch.float16)


def quantize_tensor_full(
    tensor: torch.Tensor,
    config: QuantizationConfig,
) -> QuantizedTensor:
    """
    Fully quantize a tensor, returning quantized data and metadata.

    This is useful for custom storage formats that can handle
    separate quantized values and scales.

    Args:
        tensor: Input tensor to quantize
        config: Quantization configuration

    Returns:
        QuantizedTensor with quantized data and scales
    """
    if config.method != QuantMethod.ABSMAX:
        raise NotImplementedError(f"Method {config.method} not implemented")

    original_dtype = tensor.dtype
    original_shape = tensor.shape
    tensor_f32 = tensor.float()

    quantized, scales = quantize_absmax(tensor_f32, config.bits, config.group_size)

    return QuantizedTensor(
        quantized=quantized,
        scales=scales,
        zero_points=None,  # absmax is symmetric, no zero points
        original_dtype=original_dtype,
        original_shape=original_shape,
        bits=config.bits,
    )


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack INT4 values into INT8 tensor (2 values per byte).

    Args:
        tensor: INT8 tensor with values in [-8, 7] range

    Returns:
        Packed tensor with half the elements
    """
    assert tensor.dtype == torch.int8
    assert tensor.numel() % 2 == 0, "Tensor must have even number of elements"

    flat = tensor.flatten()
    # Convert to unsigned [0, 15] range
    unsigned = (flat + 8).to(torch.uint8)

    # Pack two values per byte
    even = unsigned[0::2]
    odd = unsigned[1::2]
    packed = (even << 4) | odd

    return packed.to(torch.uint8)


def unpack_int4(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    """
    Unpack INT4 values from INT8 packed tensor.

    Args:
        packed: Packed tensor
        original_numel: Original number of elements

    Returns:
        Unpacked INT8 tensor
    """
    flat = packed.flatten()

    # Unpack
    high = (flat >> 4).to(torch.int8)
    low = (flat & 0x0F).to(torch.int8)

    # Interleave
    unpacked = torch.empty(flat.numel() * 2, dtype=torch.int8)
    unpacked[0::2] = high
    unpacked[1::2] = low

    # Convert back to signed range
    unpacked = unpacked - 8

    return unpacked[:original_numel]


def estimate_quantized_size(
    tensor: torch.Tensor,
    bits: int,
    group_size: int | None = None,
) -> dict[str, Any]:
    """
    Estimate the size of a quantized tensor.

    Args:
        tensor: Input tensor
        bits: Quantization bits
        group_size: Group size for group-wise quantization

    Returns:
        Dictionary with size estimates
    """
    numel = tensor.numel()
    original_bytes = numel * tensor.element_size()

    if bits == 8:
        quantized_bytes = numel  # 1 byte per value
    else:  # bits == 4
        quantized_bytes = (numel + 1) // 2  # 0.5 bytes per value

    if group_size is None:
        scale_bytes = 4  # Single float32 scale
    else:
        n_groups = (numel + group_size - 1) // group_size
        scale_bytes = n_groups * 4  # float32 per group

    total_bytes = quantized_bytes + scale_bytes
    compression_ratio = original_bytes / total_bytes

    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total_bytes,
        "compression_ratio": compression_ratio,
        "bits": bits,
        "group_size": group_size,
    }


def quantize_tensor_bitllm_full(
    tensor: torch.Tensor,
    config: QuantizationConfig,
) -> "BitLLMTensor":
    """
    Fully quantize a tensor using BitLLM method.

    Returns the packed BitLLMTensor for efficient storage.

    Args:
        tensor: Input tensor to quantize
        config: Quantization configuration (must use BITLLM_1BIT or BITLLM_TERNARY)

    Returns:
        BitLLMTensor with packed weights and scales

    Raises:
        ValueError: If config method is not a BitLLM method
    """
    from src.convert.bitllm import (
        quantize_to_1bit,
        quantize_to_ternary,
        BitLLMTensor,
    )

    group_size = config.group_size or 64

    if config.method == QuantMethod.BITLLM_1BIT:
        return quantize_to_1bit(tensor, group_size)
    elif config.method == QuantMethod.BITLLM_TERNARY:
        return quantize_to_ternary(tensor, group_size)
    else:
        raise ValueError(
            f"quantize_tensor_bitllm_full requires BITLLM_1BIT or BITLLM_TERNARY method, "
            f"got {config.method}"
        )
