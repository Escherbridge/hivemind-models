"""
HiveMind Models SDK — Build distributed MoE experts with tool calling.

Quick start:
    from hivemind_models import Pipeline, Tool, Expert

    expert = Expert("web_search", tools=[
        Tool("search", description="Search the web", params={"query": str}),
        Tool("fetch", description="Fetch a URL", params={"url": str}),
    ])

    pipeline = Pipeline("any-huggingface/model-id")
    pipeline.add_expert(expert)
    result = pipeline.build(teacher="dry_run", samples=100)
    result.export("./output")

Three API layers:
    - High-level: Pipeline, Expert, Tool (declarative, minimal code)
    - Stage-level: MoEUpcycler, ExpertTrainer, DatasetGenerator (composable)
    - Primitives: BitNet codec, MoE gating, LoRA, tensor ops, HMTF export
"""

from src.sdk.tool import Tool
from src.sdk.expert import Expert
from src.sdk.pipeline import Pipeline
from src.sdk.result import PipelineResult
from src.sdk.config import PipelineConfig, TrainingConfig, TeacherConfig

# Re-export canonical types for convenience
from src.moe.config import MoEConfig
from src.moe.pipeline_config import QuantizeConfig as BitNetConfig

__all__ = [
    "Tool",
    "Expert",
    "Pipeline",
    "PipelineResult",
    "PipelineConfig",
    "MoEConfig",
    "BitNetConfig",
    "TrainingConfig",
    "TeacherConfig",
]
