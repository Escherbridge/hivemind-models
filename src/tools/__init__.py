"""
Tools module - Tool schema definitions, categories, and dataset generation.
"""

from src.tools.schema import ToolSchema, ToolCategory, ToolRegistry
from src.tools.categories import (
    DEFAULT_CATEGORIES,
    WEB_SEARCH,
    CODE_EXECUTION,
    DATABASE,
    API_CALLS,
    MATH_COMPUTE,
    FILE_IO,
    COMMUNICATION,
    KNOWLEDGE,
    build_default_registry,
)
from src.tools.dataset_generator import (
    TeacherModel,
    ClaudeTeacher,
    OpenAITeacher,
    DryRunTeacher,
    ToolCallingDatasetGenerator,
    GeneratedSample,
    create_teacher,
)

__all__ = [
    # Schema
    "ToolSchema",
    "ToolCategory",
    "ToolRegistry",
    # Categories
    "DEFAULT_CATEGORIES",
    "WEB_SEARCH",
    "CODE_EXECUTION",
    "DATABASE",
    "API_CALLS",
    "MATH_COMPUTE",
    "FILE_IO",
    "COMMUNICATION",
    "KNOWLEDGE",
    "build_default_registry",
    # Dataset generation
    "TeacherModel",
    "ClaudeTeacher",
    "OpenAITeacher",
    "DryRunTeacher",
    "ToolCallingDatasetGenerator",
    "GeneratedSample",
    "create_teacher",
]
