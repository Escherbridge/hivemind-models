"""
PipelineResult — The output of a pipeline build.

Holds the trained model, shards, manifest, and knows how to export
to multiple formats (HiveMind, HuggingFace, local files).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExpertResult:
    """Result for a single trained expert."""

    name: str
    expert_index: int
    tool_names: list[str]
    shard_path: str | None = None
    shard_size_bytes: int = 0
    accuracy: float | None = None
    training_loss: float | None = None
    parameter_count: int = 0


@dataclass
class PipelineResult:
    """
    Complete result of a pipeline build.

    Contains all artifacts needed to deploy the model
    to HiveMind or use it locally.
    """

    model_id: str
    output_dir: str
    experts: list[ExpertResult] = field(default_factory=list)
    shards: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    # Model stats
    total_params: int = 0
    active_params_per_token: int = 0
    total_size_bytes: int = 0
    bitnet_size_bytes: int = 0
    compression_ratio: float = 0.0

    # Pipeline metadata
    stages_completed: list[str] = field(default_factory=list)
    checkpoint_dir: str | None = None

    def export(self, path: str | Path, format: str = "hivemind") -> Path:
        """
        Export all artifacts to a directory.

        Args:
            path: Output directory
            format: Export format — "hivemind", "huggingface", or "raw"

        Returns:
            Path to the export directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if format == "hivemind":
            return self._export_hivemind(path)
        elif format == "huggingface":
            return self._export_huggingface(path)
        elif format == "raw":
            return self._export_raw(path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'hivemind', 'huggingface', or 'raw'.")

    def _export_hivemind(self, path: Path) -> Path:
        """Export in HiveMind format (shards + manifest + tool schemas)."""
        # Write manifest
        manifest = {
            **self.manifest,
            "format": "hivemind",
            "experts": [
                {
                    "name": e.name,
                    "index": e.expert_index,
                    "tools": e.tool_names,
                    "shard": e.shard_path,
                    "accuracy": e.accuracy,
                }
                for e in self.experts
            ],
            "stats": {
                "total_params": self.total_params,
                "active_params_per_token": self.active_params_per_token,
                "total_size_bytes": self.total_size_bytes,
                "bitnet_size_bytes": self.bitnet_size_bytes,
                "compression_ratio": self.compression_ratio,
            },
        }

        manifest_path = path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Exported HiveMind artifacts to {path}")
        logger.info(f"  Shards: {len(self.shards)}")
        logger.info(f"  Experts: {len(self.experts)}")
        logger.info(f"  Total size: {self.total_size_bytes / 1e6:.1f} MB")
        logger.info(f"  BitNet size: {self.bitnet_size_bytes / 1e6:.1f} MB")

        return path

    def _export_huggingface(self, path: Path) -> Path:
        """Export in HuggingFace format (for Hub upload)."""
        # Write model card
        card = (
            f"# {self.model_id} — HiveMind MoE\n\n"
            f"BitNet MoE model with {len(self.experts)} tool-calling experts.\n\n"
            f"## Experts\n"
        )
        for e in self.experts:
            card += f"- **{e.name}**: {', '.join(e.tool_names)}\n"

        with open(path / "README.md", "w") as f:
            f.write(card)

        logger.info(f"Exported HuggingFace artifacts to {path}")
        return path

    def _export_raw(self, path: Path) -> Path:
        """Export raw shards and metadata."""
        meta = {
            "model_id": self.model_id,
            "experts": [e.__dict__ for e in self.experts],
            "stages_completed": self.stages_completed,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Exported raw artifacts to {path}")
        return path

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            f"Pipeline Result: {self.model_id}",
            f"  Stages completed: {', '.join(self.stages_completed)}",
            f"  Total params: {self.total_params:,}",
            f"  Active params/token: {self.active_params_per_token:,}",
            f"  Experts: {len(self.experts)}",
        ]
        for e in self.experts:
            acc = f", accuracy={e.accuracy:.1%}" if e.accuracy else ""
            lines.append(f"    [{e.expert_index}] {e.name}: {', '.join(e.tool_names)}{acc}")
        lines.extend([
            f"  Shards: {len(self.shards)}",
            f"  Total size: {self.total_size_bytes / 1e6:.1f} MB",
            f"  BitNet size: {self.bitnet_size_bytes / 1e6:.1f} MB",
            f"  Compression: {self.compression_ratio:.1f}x",
        ])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PipelineResult(model={self.model_id!r}, "
            f"experts={len(self.experts)}, "
            f"shards={len(self.shards)}, "
            f"stages={self.stages_completed})"
        )
