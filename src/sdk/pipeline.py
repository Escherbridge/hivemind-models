"""
Pipeline — The high-level SDK orchestrator.

Thin facade over ``src.moe.pipeline.MoEPipeline``, providing a fluent
builder API.  All real stage logic lives in the MoE pipeline; this module
only translates the declarative Expert/Tool API into config and delegates.

Usage:
    from hivemind_models import Pipeline, Expert, Tool

    pipeline = Pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    pipeline.add_expert(Expert("web_search", tools=[
        Tool("search", description="Search the web", params={"query": str}),
    ]))

    result = pipeline.build(teacher="dry_run", samples=100)
    result.export("./output")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.sdk.expert import Expert
from src.sdk.config import PipelineConfig
from src.sdk.result import PipelineResult, ExpertResult

logger = logging.getLogger(__name__)


class Pipeline:
    """
    High-level pipeline for building distributed MoE experts.

    Takes any HuggingFace model ID and a set of Expert definitions,
    then orchestrates: ingest -> upcycle -> generate data -> train -> quantize -> shard -> export.

    The model is loaded generically via ``transformers.AutoModelForCausalLM``
    and ``AutoTokenizer`` — no model-specific handlers needed. Architecture
    detection (layer structure, MLP location, hidden size) happens
    automatically in the upcycler.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str = "./output",
        config: PipelineConfig | None = None,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.config = config or PipelineConfig(model_id=model_id, output_dir=output_dir)
        self.experts: list[Expert] = []
        self._moe_pipeline = None  # Lazy-init MoEPipeline

    # ---- Builder methods (return self for chaining) ----

    def add_expert(self, expert: Expert) -> "Pipeline":
        """Add an expert to the pipeline. Returns self for chaining."""
        expert.expert_index = len(self.experts)
        self.experts.append(expert)
        return self

    def add_experts(self, *experts: Expert) -> "Pipeline":
        """Add multiple experts. Returns self for chaining."""
        for expert in experts:
            self.add_expert(expert)
        return self

    def configure_moe(self, **kwargs: Any) -> "Pipeline":
        """Configure MoE parameters. Returns self for chaining."""
        for k, v in kwargs.items():
            if hasattr(self.config.moe, k):
                setattr(self.config.moe, k, v)
        return self

    def configure_training(self, **kwargs: Any) -> "Pipeline":
        """Configure training parameters. Returns self for chaining."""
        for k, v in kwargs.items():
            if hasattr(self.config.training, k):
                setattr(self.config.training, k, v)
        return self

    def configure_quantization(self, **kwargs: Any) -> "Pipeline":
        """Configure BitNet quantization. Returns self for chaining."""
        for k, v in kwargs.items():
            if hasattr(self.config.bitnet, k):
                setattr(self.config.bitnet, k, v)
        return self

    def configure_teacher(self, **kwargs: Any) -> "Pipeline":
        """Configure the teacher model for dataset generation. Returns self for chaining."""
        for k, v in kwargs.items():
            if hasattr(self.config.teacher, k):
                setattr(self.config.teacher, k, v)
        return self

    # ---- Build (the main entry point) ----

    def build(
        self,
        teacher: str | None = None,
        samples: int | None = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline.

        Args:
            teacher: Shorthand for teacher model ("claude-sonnet-4-20250514", "dry_run", etc.)
            samples: Shorthand for samples_per_expert

        Returns:
            PipelineResult with all artifacts
        """
        if not self.experts:
            raise ValueError(
                "No experts defined. Add experts with pipeline.add_expert() before building."
            )

        # Apply shorthands
        if teacher:
            if teacher == "dry_run":
                self.config.teacher.provider = "dry_run"
            elif teacher.startswith("claude"):
                self.config.teacher.provider = "claude"
                self.config.teacher.model = teacher
            elif teacher.startswith("gpt"):
                self.config.teacher.provider = "openai"
                self.config.teacher.model = teacher
            else:
                self.config.teacher.provider = teacher

        if samples:
            self.config.training.samples_per_expert = samples

        # Auto-set num_experts from expert count
        self.config.moe.num_experts = max(self.config.moe.num_experts, len(self.experts))

        # Map experts to tool_categories for the MoE config
        for expert in self.experts:
            if expert.expert_index is not None:
                self.config.moe.tool_categories[expert.name] = expert.expert_index

        # Delegate to MoEPipeline if available, otherwise run lightweight version
        start = time.time()
        result = self._run_pipeline()
        elapsed = time.time() - start

        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(result.summary())
        return result

    def _run_pipeline(self) -> PipelineResult:
        """
        Delegate to MoEPipeline for real execution, or run a lightweight
        version when heavy dependencies (torch, transformers) aren't available.
        """
        try:
            from src.moe.pipeline import MoEPipeline

            # Translate SDK config -> MoE pipeline config
            moe_config = self._to_moe_config()
            inner = MoEPipeline(moe_config)
            inner.run()

            # Translate results back
            return self._build_result_from_inner(inner)

        except ImportError:
            logger.info("Running lightweight pipeline (torch/transformers not available)")
            return self._run_lightweight()

    def _to_moe_config(self) -> Any:
        """Translate the SDK PipelineConfig to a MoE PipelineConfig."""
        from src.moe.pipeline_config import (
            PipelineConfig as MoEPipelineConfig,
            IngestConfig,
            DatasetConfig,
            QuantizeConfig,
        )
        from src.moe.config import MoEConfig as InternalMoEConfig
        from src.moe.expert_trainer import ExpertTrainingConfig

        return MoEPipelineConfig(
            name=f"sdk-{self.model_id.split('/')[-1]}",
            output_dir=str(self.output_dir),
            ingest=IngestConfig(model_id=self.model_id),
            moe=InternalMoEConfig(
                num_experts=self.config.moe.num_experts,
                top_k=self.config.moe.top_k,
                moe_layers=self.config.moe.moe_layers,
                gating_type=self.config.moe.gating_type,
                noise_std=self.config.moe.noise_std,
                load_balance_weight=self.config.moe.load_balance_weight,
                tool_categories={
                    e.name: e.expert_index
                    for e in self.experts
                    if e.expert_index is not None
                },
            ),
            dataset=DatasetConfig(
                teacher_provider=self.config.teacher.provider,
                teacher_model=self.config.teacher.model,
                teacher_api_key=self.config.teacher.api_key,
                n_samples_per_category=self.config.training.samples_per_expert,
            ),
            training=ExpertTrainingConfig(
                learning_rate=self.config.training.learning_rate,
                batch_size=self.config.training.batch_size,
                lora_rank=self.config.training.lora_rank,
                lora_alpha=self.config.training.lora_alpha,
                lora_dropout=self.config.training.lora_dropout,
                fp16=self.config.training.fp16,
            ),
            quantize=QuantizeConfig(
                bit_width=self.config.bitnet.bit_width,
                group_size=self.config.bitnet.group_size,
            ),
        )

    def _build_result_from_inner(self, inner: Any) -> PipelineResult:
        """Extract a PipelineResult from the completed MoEPipeline."""
        result = PipelineResult(
            model_id=self.model_id,
            output_dir=str(self.output_dir),
        )

        if inner.model is not None:
            result.total_params = sum(p.numel() for p in inner.model.parameters())

        for expert in self.experts:
            result.experts.append(ExpertResult(
                name=expert.name,
                expert_index=expert.expert_index or 0,
                tool_names=expert.tool_names,
                accuracy=expert.accuracy,
            ))

        result.stages_completed = [s.value for s in inner._completed]
        return result

    def _run_lightweight(self) -> PipelineResult:
        """
        Lightweight pipeline for environments without torch/transformers.
        Handles dataset generation and export only.
        """
        result = PipelineResult(
            model_id=self.model_id,
            output_dir=str(self.output_dir),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Export expert definitions (always works)
        experts_dir = self.output_dir / "experts"
        experts_dir.mkdir(exist_ok=True)
        for expert in self.experts:
            expert.save(experts_dir / f"{expert.name}.yaml")
            result.experts.append(ExpertResult(
                name=expert.name,
                expert_index=expert.expert_index or 0,
                tool_names=expert.tool_names,
            ))

        self.config.save(self.output_dir / "pipeline_config.yaml")
        result.stages_completed = ["export"]
        return result

    # ---- Class methods ----

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Pipeline:
        """Create a Pipeline from a PipelineConfig."""
        return cls(
            model_id=config.model_id,
            output_dir=config.output_dir,
            config=config,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pipeline:
        """Create a Pipeline from a YAML config file."""
        config = PipelineConfig.load(path)
        return cls.from_config(config)

    def __repr__(self) -> str:
        return (
            f"Pipeline({self.model_id!r}, "
            f"experts={len(self.experts)}, "
            f"stages={self.config.stages})"
        )
