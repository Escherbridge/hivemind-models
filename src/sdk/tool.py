"""
Tool — The atomic unit of the SDK.

A Tool is a function signature that an expert can call.
Tools are composable, serializable, and map directly to
the function-calling format that LLMs expect.

Usage:
    # Minimal
    tool = Tool("search", description="Search the web", params={"query": str})

    # With full schema
    tool = Tool(
        name="search",
        description="Search the web for information",
        params={
            "query": str,
            "max_results": int,
            "language": str,
        },
        required=["query"],
        returns="list[dict]",
        examples=[
            {"query": "python tutorials", "max_results": 5},
        ],
    )

    # Render as prompt
    print(tool.to_schema())   # JSON Schema format
    print(tool.to_prompt())   # LLM-friendly format
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# Python type → JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _python_type_to_json_schema(t: Any) -> str:
    """Convert a Python type hint to a JSON Schema type string."""
    if t in _TYPE_MAP:
        return _TYPE_MAP[t]
    if isinstance(t, str):
        return t  # Already a string type name
    return "string"  # Default fallback


@dataclass
class Tool:
    """
    A callable tool that an Expert can invoke.

    This is the atomic unit of the SDK. Tools define what actions
    an expert can take, with typed parameters and descriptions.
    """

    name: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    required: list[str] | None = None
    returns: str = "string"
    examples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # If required is not set, all params are required by default
        if self.required is None:
            self.required = [
                k for k in self.params
                if not k.endswith("?")
            ]
        # Strip ? suffix from optional param names
        cleaned = {}
        for k, v in self.params.items():
            clean_key = k.rstrip("?")
            cleaned[clean_key] = v
        self.params = cleaned

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema (OpenAI function calling format)."""
        properties = {}
        for name, type_hint in self.params.items():
            properties[name] = {
                "type": _python_type_to_json_schema(type_hint),
            }

        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required or [],
                },
            },
        }
        return schema

    def to_prompt(self) -> str:
        """Render as a human-readable prompt description."""
        params_str = ", ".join(
            f"{name}: {_python_type_to_json_schema(t)}"
            + ("" if name in (self.required or []) else " (optional)")
            for name, t in self.params.items()
        )
        lines = [f"  {self.name}({params_str})"]
        if self.description:
            lines.append(f"    {self.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for YAML/JSON export."""
        return {
            "name": self.name,
            "description": self.description,
            "params": {
                k: _python_type_to_json_schema(v)
                for k, v in self.params.items()
            },
            "required": self.required,
            "returns": self.returns,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tool:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            params=data.get("params", {}),
            required=data.get("required"),
            returns=data.get("returns", "string"),
            examples=data.get("examples", []),
        )

    def __repr__(self) -> str:
        return f"Tool({self.name!r}, params={list(self.params.keys())})"
