"""
Tool schema system for defining, registering, and rendering tool definitions.

Provides dataclasses for tool schemas and categories, plus a registry that
loads definitions from YAML and renders them in the prompt format LLMs
expect for tool-calling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ToolSchema:
    """A single tool definition with JSON-Schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    category: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolSchema":
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            category=data.get("category", ""),
        )

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def to_prompt_format(self) -> str:
        """Render in the OpenAI-style function-calling JSON block."""
        spec = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        return json.dumps(spec, indent=2)


@dataclass
class ToolCategory:
    """A named group of related tools."""

    name: str
    description: str
    tools: list[ToolSchema] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tools": [t.to_dict() for t in self.tools],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCategory":
        tools = [ToolSchema.from_dict(t) for t in data.get("tools", [])]
        # Propagate category name to each tool
        for tool in tools:
            if not tool.category:
                tool.category = data["name"]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tools=tools,
        )

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def to_prompt_format(self) -> str:
        """Render every tool in the category as a combined prompt block."""
        lines = [f"## Tool Category: {self.name}", f"{self.description}", ""]
        lines.append("Available tools:")
        for tool in self.tools:
            lines.append(tool.to_prompt_format())
        return "\n".join(lines)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]


class ToolRegistry:
    """
    Central registry that loads tool definitions from YAML files and provides
    lookup / rendering helpers.

    Expected YAML structure::

        categories:
          - name: web_search
            description: Tools for searching the web
            tools:
              - name: web_search
                description: Search the web
                parameters:
                  type: object
                  properties:
                    query:
                      type: string
                  required: [query]
    """

    def __init__(self) -> None:
        self._categories: dict[str, ToolCategory] = {}
        self._tools: dict[str, ToolSchema] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_yaml(self, path: str | Path) -> None:
        """Load tool definitions from a YAML file and merge into the registry."""
        path = Path(path)
        logger.info("Loading tool definitions from %s", path)
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        for cat_data in data.get("categories", []):
            category = ToolCategory.from_dict(cat_data)
            self.register_category(category)

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load tool definitions from an already-parsed dictionary."""
        for cat_data in data.get("categories", []):
            category = ToolCategory.from_dict(cat_data)
            self.register_category(category)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_category(self, category: ToolCategory) -> None:
        """Register a full category (overwrites if name already present)."""
        self._categories[category.name] = category
        for tool in category.tools:
            self._tools[tool.name] = tool
        logger.debug(
            "Registered category '%s' with %d tools",
            category.name,
            len(category.tools),
        )

    def register_tool(self, tool: ToolSchema, category_name: str | None = None) -> None:
        """Register a single tool, optionally attaching it to a category."""
        self._tools[tool.name] = tool
        if category_name and category_name in self._categories:
            self._categories[category_name].tools.append(tool)
            tool.category = category_name

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> ToolSchema | None:
        return self._tools.get(name)

    def get_category(self, name: str) -> ToolCategory | None:
        return self._categories.get(name)

    @property
    def categories(self) -> list[ToolCategory]:
        return list(self._categories.values())

    @property
    def category_names(self) -> list[str]:
        return list(self._categories.keys())

    @property
    def all_tools(self) -> list[ToolSchema]:
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def to_prompt_format(self, category_names: list[str] | None = None) -> str:
        """
        Render tool definitions for injection into a prompt.

        Args:
            category_names: If given, only include these categories.
                            Otherwise include everything.
        """
        cats = (
            [self._categories[n] for n in category_names if n in self._categories]
            if category_names
            else list(self._categories.values())
        )
        return "\n\n".join(cat.to_prompt_format() for cat in cats)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __repr__(self) -> str:
        return (
            f"ToolRegistry(categories={len(self._categories)}, "
            f"tools={len(self._tools)})"
        )
