"""
ModelSharder - Main sharding logic for HuggingFace models.

This module handles:
- Loading HuggingFace models
- Extracting embedding weights -> shard_embed.safetensors
- Extracting layer groups -> shard_layers_X_Y.safetensors
- Extracting LM head -> shard_head.safetensors
- Generating manifest.json
- Exporting tokenizer
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Configuration for model sharding."""

    model_id: str
    output_dir: Path
    layer_groups: list[tuple[int, int]]
    quantize: bool = False
    quant_bits: int = 4
    dtype: str = "float16"
    trust_remote_code: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ShardConfig":
        """Load shard configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Convert layer_groups to tuples
        layer_groups = [tuple(group) for group in data.get("layer_groups", [])]

        return cls(
            model_id=data["model_id"],
            output_dir=Path(data.get("output_dir", "./output")),
            layer_groups=layer_groups,
            quantize=data.get("quantize", False),
            quant_bits=data.get("quant_bits", 4),
            dtype=data.get("dtype", "float16"),
            trust_remote_code=data.get("trust_remote_code", False),
        )


@dataclass
class ShardInfo:
    """Information about a generated shard."""

    filename: str
    size_bytes: int
    checksum_sha256: str
    tensor_count: int
    tensor_names: list[str]
    layer_range: tuple[int, int] | None = None


@dataclass
class ShardingResult:
    """Result of model sharding operation."""

    model_id: str
    shards: list[ShardInfo]
    total_size_bytes: int
    config: dict[str, Any]
    manifest_path: Path | None = None


class ModelSharder:
    """
    Shard HuggingFace models into separate safetensors files.

    This class handles the conversion of large language models into
    smaller shard files suitable for distributed inference.
    """

    # Mapping of model architectures to their layer key patterns
    LAYER_PATTERNS = {
        "llama": "model.layers.{layer_idx}",
        "mistral": "model.layers.{layer_idx}",
        "phi": "model.layers.{layer_idx}",
        "gpt2": "transformer.h.{layer_idx}",
        "gpt_neo": "transformer.h.{layer_idx}",
        "opt": "model.decoder.layers.{layer_idx}",
    }

    EMBED_KEYS = {
        "llama": ["model.embed_tokens"],
        "mistral": ["model.embed_tokens"],
        "phi": ["model.embed_tokens"],
        "gpt2": ["transformer.wte", "transformer.wpe"],
        "opt": ["model.decoder.embed_tokens", "model.decoder.embed_positions"],
    }

    HEAD_KEYS = {
        "llama": ["lm_head", "model.norm"],
        "mistral": ["lm_head", "model.norm"],
        "phi": ["lm_head", "model.norm"],
        "gpt2": ["lm_head", "transformer.ln_f"],
        "opt": ["lm_head", "model.decoder.final_layer_norm"],
    }

    def __init__(self, config: ShardConfig):
        """
        Initialize the ModelSharder.

        Args:
            config: Sharding configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_type: str | None = None
        self._state_dict: dict[str, torch.Tensor] | None = None

    def _detect_model_type(self, config_dict: dict) -> str:
        """Detect model architecture type from config."""
        model_type = config_dict.get("model_type", "").lower()

        if "llama" in model_type:
            return "llama"
        elif "mistral" in model_type:
            return "mistral"
        elif "phi" in model_type:
            return "phi"
        elif "gpt2" in model_type:
            return "gpt2"
        elif "opt" in model_type:
            return "opt"
        else:
            # Default to llama pattern for most modern models
            logger.warning(f"Unknown model type '{model_type}', defaulting to llama pattern")
            return "llama"

    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float16)

    def load_model(self) -> None:
        """
        Load the HuggingFace model and tokenizer.

        This loads the model in the specified dtype and prepares it for sharding.
        """
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")

        logger.info(f"Loading model: {self.config.model_id}")

        # Load config first to detect model type
        model_config = AutoConfig.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model_type = self._detect_model_type(model_config.to_dict())
        logger.info(f"Detected model type: {self.model_type}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Load model
        logger.info("Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=self._get_dtype(),
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        )

        # Get state dict
        self._state_dict = self.model.state_dict()
        logger.info(f"Model loaded with {len(self._state_dict)} tensors")

    def _compute_checksum(self, tensors: dict[str, torch.Tensor]) -> str:
        """Compute SHA256 checksum of tensor data."""
        hasher = hashlib.sha256()
        for name in sorted(tensors.keys()):
            tensor = tensors[name]
            hasher.update(name.encode())
            hasher.update(tensor.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def _get_tensor_size(self, tensors: dict[str, torch.Tensor]) -> int:
        """Calculate total size of tensors in bytes."""
        total = 0
        for tensor in tensors.values():
            total += tensor.numel() * tensor.element_size()
        return total

    def _extract_embedding_tensors(self) -> dict[str, torch.Tensor]:
        """Extract embedding layer tensors."""
        if self._state_dict is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        embed_keys = self.EMBED_KEYS.get(self.model_type, self.EMBED_KEYS["llama"])
        tensors = {}

        for key in self._state_dict:
            for embed_key in embed_keys:
                if key.startswith(embed_key):
                    tensors[key] = self._state_dict[key]
                    break

        logger.info(f"Extracted {len(tensors)} embedding tensors")
        return tensors

    def _extract_layer_tensors(
        self, start_layer: int, end_layer: int
    ) -> dict[str, torch.Tensor]:
        """Extract tensors for a range of layers (inclusive)."""
        if self._state_dict is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        layer_pattern = self.LAYER_PATTERNS.get(self.model_type, self.LAYER_PATTERNS["llama"])
        tensors = {}

        for layer_idx in range(start_layer, end_layer + 1):
            prefix = layer_pattern.format(layer_idx=layer_idx)
            for key in self._state_dict:
                if key.startswith(prefix):
                    tensors[key] = self._state_dict[key]

        logger.info(f"Extracted {len(tensors)} tensors for layers {start_layer}-{end_layer}")
        return tensors

    def _extract_head_tensors(self) -> dict[str, torch.Tensor]:
        """Extract LM head and final normalization tensors."""
        if self._state_dict is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        head_keys = self.HEAD_KEYS.get(self.model_type, self.HEAD_KEYS["llama"])
        tensors = {}

        for key in self._state_dict:
            for head_key in head_keys:
                if key.startswith(head_key):
                    tensors[key] = self._state_dict[key]
                    break

        logger.info(f"Extracted {len(tensors)} head tensors")
        return tensors

    def _save_shard(
        self,
        tensors: dict[str, torch.Tensor],
        filename: str,
        layer_range: tuple[int, int] | None = None,
    ) -> ShardInfo:
        """Save tensors to a safetensors shard file."""
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("safetensors library required. Install with: pip install safetensors")

        from src.convert.quantize import quantize_tensor, QuantizationConfig

        output_path = self.config.output_dir / filename

        # Apply quantization if enabled
        if self.config.quantize:
            quant_config = QuantizationConfig(bits=self.config.quant_bits)
            quantized_tensors = {}
            for name, tensor in tensors.items():
                # Only quantize weight tensors, not biases or norms
                if "weight" in name and tensor.dim() >= 2:
                    quantized_tensors[name] = quantize_tensor(tensor, quant_config)
                else:
                    quantized_tensors[name] = tensor
            tensors = quantized_tensors

        # Ensure all tensors are on CPU and contiguous
        cpu_tensors = {
            name: tensor.cpu().contiguous() for name, tensor in tensors.items()
        }

        # Save to safetensors
        save_file(cpu_tensors, str(output_path))

        # Compute checksum and size
        checksum = self._compute_checksum(cpu_tensors)
        size_bytes = output_path.stat().st_size

        return ShardInfo(
            filename=filename,
            size_bytes=size_bytes,
            checksum_sha256=checksum,
            tensor_count=len(tensors),
            tensor_names=list(tensors.keys()),
            layer_range=layer_range,
        )

    def shard(self) -> ShardingResult:
        """
        Perform the full sharding operation.

        Returns:
            ShardingResult with information about generated shards
        """
        if self._state_dict is None:
            self.load_model()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        shards: list[ShardInfo] = []
        total_size = 0

        # 1. Extract and save embedding shard
        logger.info("Creating embedding shard...")
        embed_tensors = self._extract_embedding_tensors()
        if embed_tensors:
            shard_info = self._save_shard(embed_tensors, "shard_embed.safetensors")
            shards.append(shard_info)
            total_size += shard_info.size_bytes

        # 2. Extract and save layer group shards
        for i, (start_layer, end_layer) in enumerate(tqdm(self.config.layer_groups, desc="Sharding layers")):
            filename = f"shard_layers_{start_layer}_{end_layer}.safetensors"
            layer_tensors = self._extract_layer_tensors(start_layer, end_layer)
            if layer_tensors:
                shard_info = self._save_shard(
                    layer_tensors, filename, layer_range=(start_layer, end_layer)
                )
                shards.append(shard_info)
                total_size += shard_info.size_bytes

        # 3. Extract and save head shard
        logger.info("Creating head shard...")
        head_tensors = self._extract_head_tensors()
        if head_tensors:
            shard_info = self._save_shard(head_tensors, "shard_head.safetensors")
            shards.append(shard_info)
            total_size += shard_info.size_bytes

        # 4. Export tokenizer
        self._export_tokenizer()

        # 5. Generate manifest
        result = ShardingResult(
            model_id=self.config.model_id,
            shards=shards,
            total_size_bytes=total_size,
            config={
                "model_id": self.config.model_id,
                "layer_groups": self.config.layer_groups,
                "quantize": self.config.quantize,
                "quant_bits": self.config.quant_bits if self.config.quantize else None,
                "dtype": self.config.dtype,
                "model_type": self.model_type,
            },
        )

        manifest_path = self._generate_manifest(result)
        result.manifest_path = manifest_path

        logger.info(f"Sharding complete. Total size: {total_size / (1024**2):.2f} MB")
        return result

    def _export_tokenizer(self) -> None:
        """Export tokenizer files to output directory."""
        if self.tokenizer is None:
            logger.warning("Tokenizer not loaded, skipping export")
            return

        tokenizer_dir = self.config.output_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)

        logger.info("Exporting tokenizer...")
        self.tokenizer.save_pretrained(str(tokenizer_dir))

        # Also save tokenizer.json if using fast tokenizer
        if hasattr(self.tokenizer, "backend_tokenizer"):
            self.tokenizer.save_pretrained(str(tokenizer_dir))

    def _generate_manifest(self, result: ShardingResult) -> Path:
        """Generate manifest.json for the sharded model."""
        manifest = {
            "model_id": result.model_id,
            "model_type": self.model_type,
            "version": "1.0",
            "total_size_bytes": result.total_size_bytes,
            "quantized": self.config.quantize,
            "quant_bits": self.config.quant_bits if self.config.quantize else None,
            "dtype": self.config.dtype,
            "layer_groups": self.config.layer_groups,
            "shards": [
                {
                    "filename": shard.filename,
                    "size_bytes": shard.size_bytes,
                    "checksum_sha256": shard.checksum_sha256,
                    "tensor_count": shard.tensor_count,
                    "layer_range": shard.layer_range,
                }
                for shard in result.shards
            ],
            "tokenizer": {
                "path": "tokenizer/",
                "type": type(self.tokenizer).__name__ if self.tokenizer else None,
            },
        }

        manifest_path = self.config.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Manifest written to {manifest_path}")
        return manifest_path

    def dry_run(self) -> dict[str, Any]:
        """
        Perform a dry run showing what sharding would produce.

        Returns:
            Dictionary describing the planned sharding operation
        """
        plan = {
            "model_id": self.config.model_id,
            "output_dir": str(self.config.output_dir),
            "planned_shards": [],
            "quantization": {
                "enabled": self.config.quantize,
                "bits": self.config.quant_bits if self.config.quantize else None,
            },
            "dtype": self.config.dtype,
        }

        # Plan embedding shard
        plan["planned_shards"].append({
            "filename": "shard_embed.safetensors",
            "description": "Embedding layer weights",
        })

        # Plan layer shards
        for start_layer, end_layer in self.config.layer_groups:
            plan["planned_shards"].append({
                "filename": f"shard_layers_{start_layer}_{end_layer}.safetensors",
                "description": f"Transformer layers {start_layer}-{end_layer}",
                "layer_range": [start_layer, end_layer],
            })

        # Plan head shard
        plan["planned_shards"].append({
            "filename": "shard_head.safetensors",
            "description": "LM head and final normalization",
        })

        return plan


def create_sharder_from_config(config_path: str | Path) -> ModelSharder:
    """
    Create a ModelSharder from a YAML config file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured ModelSharder instance
    """
    config = ShardConfig.from_yaml(config_path)
    return ModelSharder(config)
