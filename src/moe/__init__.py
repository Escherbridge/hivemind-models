"""
MoE module - Mixture-of-Experts upcycling and gating primitives.

Domain-specialisation classes (ExpertTrainer, MoEPipeline, SDK Expert/Pipeline)
have been archived to hivemind-archive/domain-moe-sdk/ — see that repo's README.md
for rationale (broken domain-MoE premise; replaced by Branch-Train-Merge, Axis 3).
"""

from src.moe.config import MoEConfig
from src.moe.gating import TopKGating, ExpertChoiceGating
from src.moe.upcycle import MoEUpcycler, MoELayer
from src.moe.pipeline_config import PipelineConfig

__all__ = [
    # Config
    "MoEConfig",
    # Gating
    "TopKGating",
    "ExpertChoiceGating",
    # Upcycling
    "MoEUpcycler",
    "MoELayer",
    # Pipeline config (generic, no domain claim)
    "PipelineConfig",
]
