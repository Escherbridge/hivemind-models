"""
Convert module - Model sharding and quantization utilities.
"""

from src.convert.sharder import ModelSharder, ShardConfig
from src.convert.quantize import quantize_tensor, QuantizationConfig
from src.convert.safetensors_utils import save_safetensors, load_safetensors
from src.convert.validation import validate_shards, verify_tensor_shapes
from src.convert.bitllm import (
    quantize_to_1bit,
    quantize_to_ternary,
    BitLLMConfig,
    BitLLMTensor,
    quantize_tensor_bitllm,
    dequantize,
    tensor_to_safetensors_dict,
    safetensors_dict_to_tensor,
    estimate_bitllm_size,
)

__all__ = [
    "ModelSharder",
    "ShardConfig",
    "quantize_tensor",
    "QuantizationConfig",
    "save_safetensors",
    "load_safetensors",
    "validate_shards",
    "verify_tensor_shapes",
    # BitLLM exports
    "quantize_to_1bit",
    "quantize_to_ternary",
    "BitLLMConfig",
    "BitLLMTensor",
    "quantize_tensor_bitllm",
    "dequantize",
    "tensor_to_safetensors_dict",
    "safetensors_dict_to_tensor",
    "estimate_bitllm_size",
]
