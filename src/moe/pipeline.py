"""
MoE pipeline orchestrator.

``MoEPipeline`` drives the full lifecycle:
  ingest -> upcycle -> generate_datasets -> train_experts -> quantize -> shard -> export

Each step persists a checkpoint so that the pipeline can be resumed from any
previously completed stage.
"""

from __future__ import annotations

import json
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from src.convert.bitllm import BitLLMConfig, quantize_tensor_bitllm, tensor_to_safetensors_dict
from src.convert.sharder import ModelSharder, ShardConfig
from src.moe.config import MoEConfig
from src.moe.expert_trainer import ExpertTrainer, ExpertTrainingConfig
from src.moe.pipeline_config import PipelineConfig
from src.moe.upcycle import MoEUpcycler
from src.tools.categories import DEFAULT_CATEGORIES, build_default_registry
from src.tools.dataset_generator import ToolCallingDatasetGenerator, create_teacher
from src.tools.schema import ToolCategory, ToolRegistry

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Ordered pipeline stages."""

    INGEST = "ingest"
    UPCYCLE = "upcycle"
    GENERATE_DATASETS = "generate_datasets"
    TRAIN_EXPERTS = "train_experts"
    QUANTIZE = "quantize"
    SHARD = "shard"
    EXPORT = "export"

    @classmethod
    def ordered(cls) -> list["PipelineStage"]:
        return [
            cls.INGEST,
            cls.UPCYCLE,
            cls.GENERATE_DATASETS,
            cls.TRAIN_EXPERTS,
            cls.QUANTIZE,
            cls.SHARD,
            cls.EXPORT,
        ]


class MoEPipeline:
    """
    End-to-end orchestrator for the BitNet MoE tool-calling pipeline.

    Usage::

        config = PipelineConfig.from_yaml("pipeline.yaml")
        pipeline = MoEPipeline(config)
        pipeline.run()      # runs all stages
        # or run individual stages:
        pipeline.ingest()
        pipeline.upcycle()
        ...
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._output_dir = Path(config.output_dir)
        self._checkpoint_dir = config.checkpoint_dir
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Populated across stages
        self.model: Any = None
        self.tokenizer: Any = None
        self.tool_registry: ToolRegistry | None = None
        self.dataset_paths: dict[int, Path] = {}
        self.loss_curves: dict[int, list[float]] = {}

        self._completed: set[PipelineStage] = set()
        self._load_progress()

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    @property
    def _progress_path(self) -> Path:
        return self._checkpoint_dir / "progress.json"

    def _load_progress(self) -> None:
        if self._progress_path.exists():
            with open(self._progress_path, "r") as fh:
                data = json.load(fh)
            self._completed = {PipelineStage(s) for s in data.get("completed", [])}
            self.dataset_paths = {
                int(k): Path(v)
                for k, v in data.get("dataset_paths", {}).items()
            }
            logger.info("Resumed pipeline. Completed stages: %s", self._completed)

    def _save_progress(self) -> None:
        data = {
            "completed": [s.value for s in self._completed],
            "dataset_paths": {str(k): str(v) for k, v in self.dataset_paths.items()},
        }
        with open(self._progress_path, "w") as fh:
            json.dump(data, fh, indent=2)

    def _mark_done(self, stage: PipelineStage) -> None:
        self._completed.add(stage)
        self._save_progress()
        logger.info("Stage '%s' complete.", stage.value)

    def _is_done(self, stage: PipelineStage) -> bool:
        return stage in self._completed

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self, start_from: PipelineStage | None = None) -> None:
        """
        Execute every pipeline stage in order.

        Stages that are already marked complete (from a previous run or
        checkpoint) are skipped unless *start_from* is specified.
        """
        stages = PipelineStage.ordered()
        if start_from is not None:
            idx = stages.index(start_from)
            stages = stages[idx:]

        dispatch = {
            PipelineStage.INGEST: self.ingest,
            PipelineStage.UPCYCLE: self.upcycle,
            PipelineStage.GENERATE_DATASETS: self.generate_datasets,
            PipelineStage.TRAIN_EXPERTS: self.train_experts,
            PipelineStage.QUANTIZE: self.quantize,
            PipelineStage.SHARD: self.shard,
            PipelineStage.EXPORT: self.export,
        }

        for stage in stages:
            if self._is_done(stage) and start_from is None:
                logger.info("Skipping already-completed stage: %s", stage.value)
                continue
            logger.info("========== %s ==========", stage.value.upper())
            dispatch[stage]()

        logger.info("Pipeline complete. Artifacts in: %s", self._output_dir)

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------

    def ingest(self, model_id: str | None = None) -> None:
        """Load a HuggingFace model and tokenizer."""
        model_id = model_id or self.config.ingest.model_id
        if not model_id:
            raise ValueError("No model_id specified in config or argument.")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. pip install transformers"
            )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.ingest.dtype, torch.float16)

        logger.info("Loading model: %s (dtype=%s)", model_id, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=self.config.ingest.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=self.config.ingest.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        logger.info(
            "Model loaded. Parameters: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        # Save checkpoint
        ckpt_path = self._checkpoint_dir / "ingest"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(ckpt_path / "tokenizer"))
        # Save config only (not full weights) to keep checkpoints light
        self.model.config.save_pretrained(str(ckpt_path / "model_config"))

        self._mark_done(PipelineStage.INGEST)

    # ------------------------------------------------------------------
    # Stage 2: Upcycle
    # ------------------------------------------------------------------

    def upcycle(self, moe_config: MoEConfig | None = None) -> None:
        """Convert the dense model into MoE."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Run ingest() first.")

        cfg = moe_config or self.config.moe
        upcycler = MoEUpcycler(self.model, cfg)
        self.model = upcycler.upcycle()

        # Checkpoint: save state dict
        ckpt_path = self._checkpoint_dir / "upcycle"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_path / "moe_state_dict.pt")
        logger.info("MoE state dict saved to %s", ckpt_path)

        self._mark_done(PipelineStage.UPCYCLE)

    # ------------------------------------------------------------------
    # Stage 3: Generate datasets
    # ------------------------------------------------------------------

    def generate_datasets(
        self,
        tool_categories: list[ToolCategory] | None = None,
        n_samples_per_category: int | None = None,
    ) -> dict[int, Path]:
        """
        Generate per-expert training datasets using a teacher model.

        Returns:
            Mapping from expert index to the JSONL file path.
        """
        categories = tool_categories or self._resolve_categories()
        n_samples = n_samples_per_category or self.config.dataset.n_samples_per_category

        teacher = create_teacher(
            provider=self.config.dataset.teacher_provider,
            model=self.config.dataset.teacher_model,
            api_key=self.config.dataset.teacher_api_key,
        )

        dataset_dir = Path(self.config.dataset.output_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        generator = ToolCallingDatasetGenerator(teacher, output_dir=dataset_dir)

        cat_to_expert = self.config.moe.tool_categories
        self.dataset_paths = {}

        for cat in tqdm(categories, desc="Generating datasets"):
            expert_idx = cat_to_expert.get(cat.name)
            if expert_idx is None:
                logger.warning(
                    "Category '%s' has no expert mapping, skipping.", cat.name
                )
                continue

            path = generator.generate_and_save(
                n_samples=n_samples,
                category=cat,
                negative_ratio=self.config.dataset.negative_ratio,
                filename=f"expert_{expert_idx}_{cat.name}.jsonl",
            )
            self.dataset_paths[expert_idx] = path

        self._mark_done(PipelineStage.GENERATE_DATASETS)
        return dict(self.dataset_paths)

    # ------------------------------------------------------------------
    # Stage 4: Train experts
    # ------------------------------------------------------------------

    def train_experts(
        self,
        training_config: ExpertTrainingConfig | None = None,
    ) -> dict[int, list[float]]:
        """Fine-tune each expert on its dataset using LoRA."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not self.dataset_paths:
            raise RuntimeError("No datasets generated. Run generate_datasets() first.")

        cfg = training_config or self.config.training

        trainer = ExpertTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=cfg,
        )

        self.loss_curves = trainer.train_all_experts(self.dataset_paths)

        # Save artifacts
        ckpt_path = self._checkpoint_dir / "train"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_path / "trained_state_dict.pt")
        trainer.save_loss_curves(ckpt_path / "loss_curves.json")

        self._mark_done(PipelineStage.TRAIN_EXPERTS)
        return self.loss_curves

    # ------------------------------------------------------------------
    # Stage 5: Quantize
    # ------------------------------------------------------------------

    def quantize(self, bitnet_config: BitLLMConfig | None = None) -> None:
        """
        Quantize model weights to BitNet format.

        Reuses the existing ``src.convert.bitllm`` infrastructure.
        """
        if not self.config.quantize.enabled:
            logger.info("Quantization disabled in config. Skipping.")
            self._mark_done(PipelineStage.QUANTIZE)
            return

        if self.model is None:
            raise RuntimeError("Model not loaded.")

        cfg = bitnet_config or BitLLMConfig(
            bit_width=self.config.quantize.bit_width,
            group_size=self.config.quantize.group_size,
        )

        logger.info("Quantizing to %d-bit (group_size=%d)", cfg.bit_width, cfg.group_size)
        state_dict = self.model.state_dict()
        quantized_tensors: dict[str, torch.Tensor] = {}

        for name, tensor in tqdm(state_dict.items(), desc="Quantizing"):
            if tensor.dim() >= 2 and "weight" in name:
                bitllm = quantize_tensor_bitllm(tensor, cfg)
                for k, v in tensor_to_safetensors_dict(bitllm, name).items():
                    quantized_tensors[k] = v
            else:
                quantized_tensors[name] = tensor

        # Persist quantized state dict
        ckpt_path = self._checkpoint_dir / "quantize"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_tensors, ckpt_path / "quantized_state_dict.pt")
        logger.info(
            "Quantized %d tensors. Saved to %s", len(quantized_tensors), ckpt_path
        )

        self._mark_done(PipelineStage.QUANTIZE)

    # ------------------------------------------------------------------
    # Stage 6: Shard
    # ------------------------------------------------------------------

    def shard(self, shard_config: ShardConfig | None = None) -> None:
        """
        Shard the model for HiveMind distributed inference.

        Reuses the existing ``src.convert.sharder.ModelSharder``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        # Determine layer groups
        layer_groups = self._resolve_layer_groups()

        cfg = shard_config or ShardConfig(
            model_id=self.config.ingest.model_id,
            output_dir=Path(self.config.output_dir) / "shards",
            layer_groups=layer_groups,
            quantize=False,  # We already quantized in previous stage
            dtype=self.config.ingest.dtype,
            trust_remote_code=self.config.ingest.trust_remote_code,
        )

        sharder = ModelSharder(cfg)
        # Inject already-loaded model so sharder doesn't re-download
        sharder.model = self.model
        sharder.tokenizer = self.tokenizer
        sharder._state_dict = self.model.state_dict()
        sharder.model_type = self._detect_model_type()

        result = sharder.shard()
        logger.info(
            "Sharding complete: %d shards, %.2f MB total",
            len(result.shards),
            result.total_size_bytes / (1024**2),
        )

        self._mark_done(PipelineStage.SHARD)

    # ------------------------------------------------------------------
    # Stage 7: Export
    # ------------------------------------------------------------------

    def export(self, output_dir: str | Path | None = None) -> Path:
        """
        Export all final artifacts to a clean output directory.

        Exports:
        - Shards (from previous stage)
        - Manifest
        - Tokenizer
        - Tool schemas (YAML + rendered prompt)
        - Pipeline config snapshot
        """
        export_dir = Path(output_dir) if output_dir else self.config.export_dir
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy shards
        shard_src = Path(self.config.output_dir) / "shards"
        if shard_src.exists():
            shard_dst = export_dir / "shards"
            if shard_dst.exists():
                shutil.rmtree(shard_dst)
            shutil.copytree(shard_src, shard_dst)
            logger.info("Copied shards to %s", shard_dst)

        # Export tokenizer
        if self.config.export.include_tokenizer and self.tokenizer is not None:
            tok_dir = export_dir / "tokenizer"
            tok_dir.mkdir(exist_ok=True)
            self.tokenizer.save_pretrained(str(tok_dir))
            logger.info("Tokenizer exported to %s", tok_dir)

        # Export tool schemas
        if self.config.export.include_tool_schemas:
            registry = self.tool_registry or build_default_registry()
            schemas_data = {
                "categories": [cat.to_dict() for cat in registry.categories]
            }
            schema_path = export_dir / "tool_schemas.yaml"
            with open(schema_path, "w", encoding="utf-8") as fh:
                yaml.dump(schemas_data, fh, default_flow_style=False, sort_keys=False)

            prompt_path = export_dir / "tool_prompt.txt"
            with open(prompt_path, "w", encoding="utf-8") as fh:
                fh.write(registry.to_prompt_format())
            logger.info("Tool schemas exported to %s", export_dir)

        # Save pipeline config snapshot
        config_path = export_dir / "pipeline_config.yaml"
        self.config.save_yaml(config_path)

        # Save loss curves if available
        if self.loss_curves:
            curves_path = export_dir / "loss_curves.json"
            with open(curves_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {str(k): v for k, v in self.loss_curves.items()}, fh, indent=2
                )

        self._mark_done(PipelineStage.EXPORT)
        logger.info("Export complete. All artifacts in: %s", export_dir)
        return export_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_categories(self) -> list[ToolCategory]:
        """Get tool categories from the registry or defaults."""
        if self.tool_registry:
            return self.tool_registry.categories
        cat_names = set(self.config.moe.tool_categories.keys())
        if cat_names:
            return [c for c in DEFAULT_CATEGORIES if c.name in cat_names]
        return list(DEFAULT_CATEGORIES)

    def _resolve_layer_groups(self) -> list[tuple[int, int]]:
        """Determine shard layer groups, auto-computing if needed."""
        raw = self.config.shard.layer_groups
        if raw != "auto":
            return [(g[0], g[1]) for g in raw]

        # Auto: determine number of layers and create even groups
        num_layers = self._count_layers()
        # Estimate: aim for ~4 shards as baseline
        n_shards = max(1, num_layers // 8)
        layers_per_shard = max(1, num_layers // n_shards)

        groups: list[tuple[int, int]] = []
        for i in range(0, num_layers, layers_per_shard):
            end = min(i + layers_per_shard - 1, num_layers - 1)
            groups.append((i, end))

        logger.info("Auto layer groups: %s", groups)
        return groups

    def _count_layers(self) -> int:
        """Count transformer layers in the model."""
        if self.model is None:
            return 32  # fallback
        # Try HF config
        hf_cfg = getattr(self.model, "config", None)
        if hf_cfg and hasattr(hf_cfg, "num_hidden_layers"):
            return hf_cfg.num_hidden_layers
        return 32

    def _detect_model_type(self) -> str:
        """Detect model architecture type."""
        if self.model is None:
            return "llama"
        hf_cfg = getattr(self.model, "config", None)
        if hf_cfg:
            mt = getattr(hf_cfg, "model_type", "llama")
            return mt.lower()
        return "llama"


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_pipeline_from_yaml(config_path: str | Path) -> MoEPipeline:
    """Load config from YAML and run the full pipeline."""
    config = PipelineConfig.from_yaml(config_path)
    pipeline = MoEPipeline(config)
    pipeline.run()
    return pipeline
