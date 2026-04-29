"""
Manifest generation for sharded models.

Creates manifest.json files for client consumption.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ShardEntry:
    """Entry for a single shard in the manifest."""

    filename: str
    size_bytes: int
    checksum_sha256: str
    tensor_count: int
    layer_range: tuple[int, int] | None = None
    cdn_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "tensor_count": self.tensor_count,
        }
        if self.layer_range:
            d["layer_range"] = list(self.layer_range)
        if self.cdn_url:
            d["cdn_url"] = self.cdn_url
        return d


@dataclass
class TokenizerEntry:
    """Entry for tokenizer in the manifest."""

    path: str
    type: str
    vocab_size: int
    files: list[str] = field(default_factory=list)
    cdn_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "path": self.path,
            "type": self.type,
            "vocab_size": self.vocab_size,
            "files": self.files,
        }
        if self.cdn_url:
            d["cdn_url"] = self.cdn_url
        return d


@dataclass
class ModelManifest:
    """Complete manifest for a sharded model."""

    model_id: str
    model_type: str
    version: str
    shards: list[ShardEntry]
    tokenizer: TokenizerEntry | None
    layer_groups: list[tuple[int, int]]
    total_size_bytes: int
    quantized: bool = False
    quant_bits: int | None = None
    dtype: str = "float16"
    cdn_base_url: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at,
            "total_size_bytes": self.total_size_bytes,
            "quantized": self.quantized,
            "quant_bits": self.quant_bits,
            "dtype": self.dtype,
            "layer_groups": [list(g) for g in self.layer_groups],
            "shards": [s.to_dict() for s in self.shards],
            "tokenizer": self.tokenizer.to_dict() if self.tokenizer else None,
            "cdn_base_url": self.cdn_base_url,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save manifest to file."""
        path = Path(path)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info(f"Manifest saved to {path}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelManifest":
        """Create manifest from dictionary."""
        shards = [
            ShardEntry(
                filename=s["filename"],
                size_bytes=s["size_bytes"],
                checksum_sha256=s["checksum_sha256"],
                tensor_count=s.get("tensor_count", 0),
                layer_range=tuple(s["layer_range"]) if s.get("layer_range") else None,
                cdn_url=s.get("cdn_url"),
            )
            for s in data.get("shards", [])
        ]

        tokenizer = None
        if data.get("tokenizer"):
            t = data["tokenizer"]
            tokenizer = TokenizerEntry(
                path=t.get("path", "tokenizer/"),
                type=t.get("type", "unknown"),
                vocab_size=t.get("vocab_size", 0),
                files=t.get("files", []),
                cdn_url=t.get("cdn_url"),
            )

        return cls(
            model_id=data["model_id"],
            model_type=data.get("model_type", "unknown"),
            version=data.get("version", "1.0"),
            shards=shards,
            tokenizer=tokenizer,
            layer_groups=[tuple(g) for g in data.get("layer_groups", [])],
            total_size_bytes=data.get("total_size_bytes", 0),
            quantized=data.get("quantized", False),
            quant_bits=data.get("quant_bits"),
            dtype=data.get("dtype", "float16"),
            cdn_base_url=data.get("cdn_base_url"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        """Load manifest from file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class ManifestGenerator:
    """
    Generate manifests for sharded models.

    Scans a directory and creates a complete manifest.
    """

    def __init__(self, model_dir: str | Path):
        """
        Initialize the generator.

        Args:
            model_dir: Directory containing sharded model
        """
        self.model_dir = Path(model_dir)

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _count_tensors(self, shard_path: Path) -> int:
        """Count tensors in a safetensors file."""
        try:
            from safetensors import safe_open
            with safe_open(str(shard_path), framework="pt") as f:
                return len(f.keys())
        except Exception:
            return 0

    def _parse_layer_range(self, filename: str) -> tuple[int, int] | None:
        """Parse layer range from filename like shard_layers_0_3.safetensors."""
        if "layers_" not in filename:
            return None

        try:
            # Extract the layer range part
            parts = filename.replace(".safetensors", "").split("_")
            # Find "layers" and get the next two numbers
            layers_idx = parts.index("layers")
            start = int(parts[layers_idx + 1])
            end = int(parts[layers_idx + 2])
            return (start, end)
        except (ValueError, IndexError):
            return None

    def generate(
        self,
        model_id: str,
        model_type: str = "llama",
        quantized: bool = False,
        quant_bits: int | None = None,
        dtype: str = "float16",
        cdn_base_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelManifest:
        """
        Generate manifest for the model directory.

        Args:
            model_id: Model identifier
            model_type: Model architecture type
            quantized: Whether model is quantized
            quant_bits: Quantization bits if quantized
            dtype: Data type
            cdn_base_url: Base URL for CDN access
            metadata: Additional metadata

        Returns:
            ModelManifest
        """
        shards: list[ShardEntry] = []
        layer_groups: list[tuple[int, int]] = []
        total_size = 0

        # Scan for shard files
        for shard_path in sorted(self.model_dir.glob("*.safetensors")):
            size_bytes = shard_path.stat().st_size
            checksum = self._compute_checksum(shard_path)
            tensor_count = self._count_tensors(shard_path)
            layer_range = self._parse_layer_range(shard_path.name)

            cdn_url = None
            if cdn_base_url:
                cdn_url = f"{cdn_base_url.rstrip('/')}/{shard_path.name}"

            shard = ShardEntry(
                filename=shard_path.name,
                size_bytes=size_bytes,
                checksum_sha256=checksum,
                tensor_count=tensor_count,
                layer_range=layer_range,
                cdn_url=cdn_url,
            )
            shards.append(shard)
            total_size += size_bytes

            if layer_range:
                layer_groups.append(layer_range)

        # Sort layer groups
        layer_groups.sort(key=lambda x: x[0])

        # Scan tokenizer
        tokenizer = self._scan_tokenizer(cdn_base_url)

        return ModelManifest(
            model_id=model_id,
            model_type=model_type,
            version="1.0",
            shards=shards,
            tokenizer=tokenizer,
            layer_groups=layer_groups,
            total_size_bytes=total_size,
            quantized=quantized,
            quant_bits=quant_bits,
            dtype=dtype,
            cdn_base_url=cdn_base_url,
            metadata=metadata or {},
        )

    def _scan_tokenizer(self, cdn_base_url: str | None) -> TokenizerEntry | None:
        """Scan for tokenizer files."""
        tokenizer_dir = self.model_dir / "tokenizer"

        if not tokenizer_dir.exists():
            return None

        files = [f.name for f in tokenizer_dir.iterdir() if f.is_file()]

        if not files:
            return None

        # Try to determine vocab size
        vocab_size = 0
        tokenizer_type = "unknown"

        config_path = tokenizer_dir / "tokenizer_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                tokenizer_type = config.get("tokenizer_class", "unknown")
            except Exception:
                pass

        # Try to get vocab size from tokenizer.json
        tokenizer_json = tokenizer_dir / "tokenizer.json"
        if tokenizer_json.exists():
            try:
                with open(tokenizer_json) as f:
                    data = json.load(f)
                vocab = data.get("model", {}).get("vocab", {})
                vocab_size = len(vocab)
            except Exception:
                pass

        cdn_url = None
        if cdn_base_url:
            cdn_url = f"{cdn_base_url.rstrip('/')}/tokenizer"

        return TokenizerEntry(
            path="tokenizer/",
            type=tokenizer_type,
            vocab_size=vocab_size,
            files=files,
            cdn_url=cdn_url,
        )


def update_manifest_cdn_urls(
    manifest_path: str | Path,
    cdn_base_url: str,
) -> ModelManifest:
    """
    Update CDN URLs in an existing manifest.

    Args:
        manifest_path: Path to manifest.json
        cdn_base_url: New CDN base URL

    Returns:
        Updated ModelManifest
    """
    manifest = ModelManifest.load(manifest_path)
    manifest.cdn_base_url = cdn_base_url

    # Update shard URLs
    for shard in manifest.shards:
        shard.cdn_url = f"{cdn_base_url.rstrip('/')}/{shard.filename}"

    # Update tokenizer URL
    if manifest.tokenizer:
        manifest.tokenizer.cdn_url = f"{cdn_base_url.rstrip('/')}/tokenizer"

    # Save updated manifest
    manifest.save(manifest_path)
    return manifest


def create_cdn_manifest(
    manifest: ModelManifest,
    output_path: str | Path,
) -> None:
    """
    Create a CDN-optimized manifest with only necessary fields.

    Args:
        manifest: Source manifest
        output_path: Output path for CDN manifest
    """
    cdn_manifest = {
        "model_id": manifest.model_id,
        "model_type": manifest.model_type,
        "version": manifest.version,
        "total_size_bytes": manifest.total_size_bytes,
        "quantized": manifest.quantized,
        "quant_bits": manifest.quant_bits,
        "dtype": manifest.dtype,
        "shards": [
            {
                "filename": s.filename,
                "size_bytes": s.size_bytes,
                "checksum": s.checksum_sha256,
                "layer_range": list(s.layer_range) if s.layer_range else None,
            }
            for s in manifest.shards
        ],
        "layer_groups": [list(g) for g in manifest.layer_groups],
    }

    if manifest.tokenizer:
        cdn_manifest["tokenizer"] = {
            "path": manifest.tokenizer.path,
            "vocab_size": manifest.tokenizer.vocab_size,
        }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(cdn_manifest, f, indent=2)

    logger.info(f"CDN manifest created at {output_path}")
