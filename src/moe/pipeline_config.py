"""
Pipeline configuration loaded from YAML.

A single ``PipelineConfig`` object drives every stage of the MoE pipeline:
ingest, upcycle, dataset generation, expert training, quantization, sharding,
and export.

Example YAML::

    pipeline:
      name: "my-moe-pipeline"
      output_dir: "./output/my-moe"
      resume_from: null          # path to checkpoint dir, or null

      ingest:
        model_id: "meta-llama/Llama-3.2-1B"
        dtype: "float16"
        trust_remote_code: false

      moe:
        num_experts: 8
        top_k: 2
        moe_layers: "all_mlp"
        gating_type: "top_k"
        noise_std: 0.01
        load_balance_weight: 0.01
        expert_capacity_factor: 1.25
        tool_categories:
          web_search: 0
          code_execution: 1
          database: 2
          api_calls: 3
          math_compute: 4
          file_io: 5
          communication: 6
          knowledge: 7

      dataset:
        teacher_provider: "dry_run"    # claude | openai | dry_run
        teacher_model: null            # override model name
        teacher_api_key: null          # override (else env var)
        n_samples_per_category: 200
        negative_ratio: 0.2
        output_dir: "./datasets"

      training:
        learning_rate: 2e-4
        num_steps: 500
        batch_size: 4
        max_seq_len: 512
        lora_rank: 16
        lora_alpha: 32
        lora_dropout: 0.05
        warmup_steps: 50
        weight_decay: 0.01
        fp16: true
        per_expert: {}

      quantize:
        enabled: true
        bit_width: 2           # 1 (binary) or 2 (ternary)
        group_size: 64

      shard:
        layer_groups: "auto"   # "auto" or list of [start, end] pairs
        target_shard_mb: 256

      export:
        formats: ["safetensors"]
        include_tokenizer: true
        include_tool_schemas: true
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.moe.config import MoEConfig
from src.moe.expert_trainer import ExpertTrainingConfig


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class IngestConfig:
    model_id: str = ""
    dtype: str = "float16"
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestConfig":
        return cls(
            model_id=data.get("model_id", ""),
            dtype=data.get("dtype", "float16"),
            trust_remote_code=data.get("trust_remote_code", False),
        )


@dataclass
class DatasetConfig:
    teacher_provider: str = "dry_run"
    teacher_model: str | None = None
    teacher_api_key: str | None = None
    n_samples_per_category: int = 200
    negative_ratio: float = 0.2
    output_dir: str = "./datasets"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        return cls(
            teacher_provider=data.get("teacher_provider", "dry_run"),
            teacher_model=data.get("teacher_model"),
            teacher_api_key=data.get("teacher_api_key"),
            n_samples_per_category=int(data.get("n_samples_per_category", 200)),
            negative_ratio=float(data.get("negative_ratio", 0.2)),
            output_dir=data.get("output_dir", "./datasets"),
        )


@dataclass
class QuantizeConfig:
    enabled: bool = True
    bit_width: Literal[1, 2] = 2
    group_size: int = 64

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizeConfig":
        return cls(
            enabled=data.get("enabled", True),
            bit_width=int(data.get("bit_width", 2)),
            group_size=int(data.get("group_size", 64)),
        )


@dataclass
class ShardExportConfig:
    layer_groups: list[list[int]] | Literal["auto"] = "auto"
    target_shard_mb: int = 256

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShardExportConfig":
        lg_raw = data.get("layer_groups", "auto")
        if isinstance(lg_raw, str):
            layer_groups: list[list[int]] | Literal["auto"] = "auto"
        else:
            layer_groups = [list(g) for g in lg_raw]
        return cls(
            layer_groups=layer_groups,
            target_shard_mb=int(data.get("target_shard_mb", 256)),
        )


@dataclass
class ExportConfig:
    formats: list[str] = field(default_factory=lambda: ["safetensors"])
    include_tokenizer: bool = True
    include_tool_schemas: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportConfig":
        return cls(
            formats=data.get("formats", ["safetensors"]),
            include_tokenizer=data.get("include_tokenizer", True),
            include_tool_schemas=data.get("include_tool_schemas", True),
        )


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Master configuration for the full MoE pipeline.

    Load from YAML with ``PipelineConfig.from_yaml(path)``.
    """

    name: str = "hivemind-moe-pipeline"
    output_dir: str = "./output"
    resume_from: str | None = None

    ingest: IngestConfig = field(default_factory=IngestConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: ExpertTrainingConfig = field(default_factory=ExpertTrainingConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    shard: ShardExportConfig = field(default_factory=ShardExportConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # ------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        data = raw.get("pipeline", raw)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        training_raw = data.get("training", {})
        # Convert per_expert keys to int
        per_expert = {}
        for k, v in training_raw.get("per_expert", {}).items():
            per_expert[int(k)] = v
        training_raw_clean = {k: v for k, v in training_raw.items() if k != "per_expert"}

        training_cfg = ExpertTrainingConfig(
            **{
                k: v
                for k, v in training_raw_clean.items()
                if k in ExpertTrainingConfig.__dataclass_fields__
            },
            per_expert=per_expert,
        )

        return cls(
            name=data.get("name", "hivemind-moe-pipeline"),
            output_dir=data.get("output_dir", "./output"),
            resume_from=data.get("resume_from"),
            ingest=IngestConfig.from_dict(data.get("ingest", {})),
            moe=MoEConfig.from_dict(data.get("moe", {})),
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
            training=training_cfg,
            quantize=QuantizeConfig.from_dict(data.get("quantize", {})),
            shard=ShardExportConfig.from_dict(data.get("shard", {})),
            export=ExportConfig.from_dict(data.get("export", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": {
                "name": self.name,
                "output_dir": self.output_dir,
                "resume_from": self.resume_from,
                "ingest": {
                    "model_id": self.ingest.model_id,
                    "dtype": self.ingest.dtype,
                    "trust_remote_code": self.ingest.trust_remote_code,
                },
                "moe": self.moe.to_dict(),
                "dataset": {
                    "teacher_provider": self.dataset.teacher_provider,
                    "teacher_model": self.dataset.teacher_model,
                    "n_samples_per_category": self.dataset.n_samples_per_category,
                    "negative_ratio": self.dataset.negative_ratio,
                    "output_dir": self.dataset.output_dir,
                },
                "training": {
                    "learning_rate": self.training.learning_rate,
                    "num_steps": self.training.num_steps,
                    "batch_size": self.training.batch_size,
                    "max_seq_len": self.training.max_seq_len,
                    "lora_rank": self.training.lora_rank,
                    "lora_alpha": self.training.lora_alpha,
                    "lora_dropout": self.training.lora_dropout,
                    "warmup_steps": self.training.warmup_steps,
                    "weight_decay": self.training.weight_decay,
                    "fp16": self.training.fp16,
                },
                "quantize": {
                    "enabled": self.quantize.enabled,
                    "bit_width": self.quantize.bit_width,
                    "group_size": self.quantize.group_size,
                },
                "shard": {
                    "layer_groups": self.shard.layer_groups,
                    "target_shard_mb": self.shard.target_shard_mb,
                },
                "export": {
                    "formats": self.export.formats,
                    "include_tokenizer": self.export.include_tokenizer,
                    "include_tool_schemas": self.export.include_tool_schemas,
                },
            }
        }

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints"

    @property
    def datasets_dir(self) -> Path:
        return Path(self.dataset.output_dir)

    @property
    def export_dir(self) -> Path:
        return Path(self.output_dir) / "export"
