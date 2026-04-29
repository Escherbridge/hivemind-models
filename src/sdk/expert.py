"""
Expert — A collection of tools that forms one MoE expert.

An Expert defines a capability boundary: the set of tools it can call,
its training data requirements, and its identity in the MoE routing system.

Usage:
    from hivemind_models import Expert, Tool

    # Simple: just name and tools
    search = Expert("web_search", tools=[
        Tool("search", description="Search the web", params={"query": str}),
        Tool("fetch_url", description="Fetch a URL", params={"url": str}),
    ])

    # With training hints
    code = Expert(
        name="code_execution",
        description="Execute code and manage files",
        tools=[...],
        system_prompt="You are a code execution assistant...",
        negative_domains=["cooking", "travel"],  # Don't route these here
    )

    # Experts know how to generate their own training data
    dataset = search.generate_dataset(teacher="claude-sonnet-4-20250514", n=500)

    # Experts are serializable
    search.save("experts/web_search.yaml")
    loaded = Expert.load("experts/web_search.yaml")
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.sdk.tool import Tool


@dataclass
class Expert:
    """
    A tool-calling expert that maps to one MoE expert slot.

    Experts define:
    - What tools they can invoke
    - How to generate training data for specialization
    - Their routing identity (for the gating network)
    """

    name: str
    tools: list[Tool] = field(default_factory=list)
    description: str = ""
    system_prompt: str = ""
    negative_domains: list[str] = field(default_factory=list)
    expert_index: int | None = None  # Assigned during pipeline.build()

    # Training state (populated during pipeline execution)
    dataset_path: str | None = None
    lora_path: str | None = None
    trained: bool = False
    accuracy: float | None = None

    def __post_init__(self):
        if not self.description:
            tool_names = ", ".join(t.name for t in self.tools)
            self.description = f"Expert for: {tool_names}"
        if not self.system_prompt:
            self.system_prompt = self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Generate a default system prompt from the tool definitions."""
        tool_block = "\n".join(t.to_prompt() for t in self.tools)
        return (
            f"You are a specialized assistant for {self.name.replace('_', ' ')}.\n"
            f"You have access to the following tools:\n{tool_block}\n\n"
            f"When a user's request requires one of these tools, respond with a "
            f"tool call in the format:\n"
            f"<tool_call>{{\"name\": \"tool_name\", \"args\": {{...}}}}</tool_call>\n\n"
            f"If the request does not require any of your tools, respond normally."
        )

    @property
    def tool_names(self) -> list[str]:
        """List of tool names this expert handles."""
        return [t.name for t in self.tools]

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """JSON Schema representations of all tools."""
        return [t.to_schema() for t in self.tools]

    def add_tool(self, tool: Tool) -> "Expert":
        """Add a tool to this expert. Returns self for chaining."""
        self.tools.append(tool)
        return self

    def get_dataset_generation_prompt(self, n_samples: int = 10) -> str:
        """
        Generate the prompt for a teacher model to create training data.

        This is the core of dynamic dataset creation: we describe the tools
        and ask the teacher to generate diverse (query, tool_call) pairs.
        """
        tool_schemas = json.dumps(self.tool_schemas, indent=2)
        negative_hint = ""
        if self.negative_domains:
            domains = ", ".join(self.negative_domains)
            negative_hint = (
                f"\n\nAlso generate {n_samples // 5} NEGATIVE examples — queries about "
                f"{domains} that should NOT use these tools. For negatives, "
                f"respond with a normal text answer, not a tool call."
            )

        return (
            f"Generate {n_samples} diverse training examples for a tool-calling AI assistant.\n\n"
            f"The assistant has access to these tools:\n{tool_schemas}\n\n"
            f"For each example, provide:\n"
            f"1. A realistic user message (diverse phrasing, varying complexity)\n"
            f"2. The correct assistant response using <tool_call> tags\n\n"
            f"Format each example as JSON:\n"
            f'{{"messages": [\n'
            f'  {{"role": "system", "content": "<system prompt>"}},\n'
            f'  {{"role": "user", "content": "<user query>"}},\n'
            f'  {{"role": "assistant", "content": "<tool_call>{{...}}</tool_call>"}}\n'
            f"]}}\n\n"
            f"Output {n_samples} examples as a JSON array. Make queries diverse: "
            f"simple, complex, multi-step, edge cases, ambiguous."
            f"{negative_hint}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": [t.to_dict() for t in self.tools],
            "system_prompt": self.system_prompt,
            "negative_domains": self.negative_domains,
            "expert_index": self.expert_index,
            "trained": self.trained,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Expert:
        """Deserialize from dictionary."""
        tools = [Tool.from_dict(t) for t in data.get("tools", [])]
        return cls(
            name=data["name"],
            tools=tools,
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            negative_domains=data.get("negative_domains", []),
            expert_index=data.get("expert_index"),
            trained=data.get("trained", False),
            accuracy=data.get("accuracy"),
        )

    def save(self, path: str | Path) -> None:
        """Save expert definition to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> Expert:
        """Load expert definition from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        tools_str = ", ".join(self.tool_names)
        status = "trained" if self.trained else "untrained"
        return f"Expert({self.name!r}, tools=[{tools_str}], {status})"
