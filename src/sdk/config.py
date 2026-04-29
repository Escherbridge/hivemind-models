"""
Configuration objects for the SDK pipeline.

Re-exports canonical config types from src.moe where they exist,
and defines SDK-specific lightweight wrappers only where needed.
YAML is supported as an export format; the primary API is Python code.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Canonical config types — single source of truth
from src.moe.config import MoEConfig
from src.moe.pipeline_config import QuantizeConfig as BitNetConfig


@dataclass
class TrainingConfig:
    """Configuration for expert fine-tuning (SDK-facing subset)."""

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_seq_length: int = 2048
    samples_per_expert: int = 500
    fp16: bool = True


@dataclass
class TeacherConfig:
    """Configuration for the teacher model used in dataset generation."""

    provider: Literal["claude", "openai", "local", "dry_run"] = "dry_run"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None  # If None, reads from env / Colab secrets


@dataclass
class PipelineConfig:
    """
    Top-level SDK configuration.

    Can be created from code or loaded from YAML.
    The model_id must be provided — there is no default.
    """

    model_id: str = ""
    output_dir: str = "./output"
    moe: MoEConfig = field(default_factory=MoEConfig)
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    # Pipeline control
    stages: list[str] = field(default_factory=lambda: [
        "ingest", "upcycle", "generate_data", "train", "quantize", "shard", "export"
    ])

    def save(self, path: str | Path) -> None:
        """Export to YAML."""
        import dataclasses
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            model_id=data.get("model_id", ""),
            output_dir=data.get("output_dir", "./output"),
            moe=MoEConfig.from_dict(data.get("moe", {})),
            bitnet=BitNetConfig.from_dict(data.get("bitnet", {})),
            training=TrainingConfig(**{
                k: v for k, v in data.get("training", {}).items()
                if k in TrainingConfig.__dataclass_fields__
            }),
            teacher=TeacherConfig(**{
                k: v for k, v in data.get("teacher", {}).items()
                if k in TeacherConfig.__dataclass_fields__
            }),
            stages=data.get("stages", cls.stages),
        )
