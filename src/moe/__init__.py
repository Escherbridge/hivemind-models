"""
MoE module - Mixture-of-Experts upcycling, gating, training, and pipeline.
"""

from src.moe.config import MoEConfig
from src.moe.gating import TopKGating, ExpertChoiceGating
from src.moe.upcycle import MoEUpcycler, MoELayer
from src.moe.expert_trainer import (
    ExpertTrainer,
    ExpertTrainingConfig,
    LoRALinear,
    ToolCallingDataset,
)
from src.moe.pipeline_config import PipelineConfig
from src.moe.pipeline import MoEPipeline, PipelineStage, run_pipeline_from_yaml

__all__ = [
    # Config
    "MoEConfig",
    # Gating
    "TopKGating",
    "ExpertChoiceGating",
    # Upcycling
    "MoEUpcycler",
    "MoELayer",
    # Training
    "ExpertTrainer",
    "ExpertTrainingConfig",
    "LoRALinear",
    "ToolCallingDataset",
    # Pipeline
    "PipelineConfig",
    "MoEPipeline",
    "PipelineStage",
    "run_pipeline_from_yaml",
]
