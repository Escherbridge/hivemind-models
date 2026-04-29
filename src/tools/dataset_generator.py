"""
Synthetic tool-calling dataset generator.

Uses a teacher model (Claude, OpenAI, or dry-run placeholder) to produce
diverse (user_query, tool_call_response) training pairs for each tool
category, formatted with special tokens for fine-tuning.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.tools.schema import ToolCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special tokens used in the chat-completion training format
# ---------------------------------------------------------------------------
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"


# ---------------------------------------------------------------------------
# Teacher model abstraction
# ---------------------------------------------------------------------------

@dataclass
class TeacherResponse:
    """Container for a teacher-model generation."""

    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class TeacherModel(abc.ABC):
    """Abstract interface for a teacher LLM used to generate training pairs."""

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> TeacherResponse:
        """Send a prompt pair and return the model's text response."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""


# ---------------------------------------------------------------------------
# Claude (Anthropic) teacher
# ---------------------------------------------------------------------------

class ClaudeTeacher(TeacherModel):
    """Teacher backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError("anthropic SDK required. Install with: pip install anthropic")

    def generate(self, system_prompt: str, user_prompt: str) -> TeacherResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text
        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        return TeacherResponse(text=text, model=self._model, usage=usage)

    @property
    def name(self) -> str:
        return f"Claude ({self._model})"


# ---------------------------------------------------------------------------
# OpenAI teacher
# ---------------------------------------------------------------------------

class OpenAITeacher(TeacherModel):
    """Teacher backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY."
            )

        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError("openai SDK required. Install with: pip install openai")

    def generate(self, system_prompt: str, user_prompt: str) -> TeacherResponse:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        completion = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            }
        return TeacherResponse(text=text, model=self._model, usage=usage)

    @property
    def name(self) -> str:
        return f"OpenAI ({self._model})"


# ---------------------------------------------------------------------------
# Dry-run teacher (no API calls)
# ---------------------------------------------------------------------------

class DryRunTeacher(TeacherModel):
    """Placeholder teacher that returns deterministic dummy data."""

    def generate(self, system_prompt: str, user_prompt: str) -> TeacherResponse:
        # Return a minimal but structurally valid placeholder
        placeholder = (
            "QUERY: What is the capital of France?\n"
            "TOOL_CALL: {\"name\": \"web_search\", "
            "\"arguments\": {\"query\": \"capital of France\"}}\n"
            "TOOL_RESULT: {\"answer\": \"Paris\"}\n"
            "RESPONSE: The capital of France is Paris."
        )
        return TeacherResponse(text=placeholder, model="dry-run", usage={})

    @property
    def name(self) -> str:
        return "DryRun"


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

@dataclass
class GeneratedSample:
    """A single training sample in chat-completion format."""

    id: str
    category: str
    messages: list[dict[str, str]]
    is_positive: bool  # True = should use tools, False = should NOT
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "messages": self.messages,
            "is_positive": self.is_positive,
            "metadata": self.metadata,
        }


class ToolCallingDatasetGenerator:
    """
    Generate synthetic tool-calling training data for a given tool category.

    Workflow:
        1. Build a system prompt describing the available tools.
        2. Ask the teacher model to produce diverse (query, tool_call) pairs.
        3. Parse the teacher's output into structured chat-completion samples.
        4. Persist as JSONL.
    """

    # Number of samples requested per teacher call (batch efficiency)
    _SAMPLES_PER_CALL: int = 5

    def __init__(
        self,
        teacher: TeacherModel,
        output_dir: str | Path = "./datasets",
    ) -> None:
        self.teacher = teacher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dataset generator using teacher: %s", teacher.name)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, category: ToolCategory) -> str:
        tools_json = json.dumps(
            [t.to_dict() for t in category.tools], indent=2
        )
        return (
            "You are a training-data generator for a tool-calling language model.\n"
            "You will be given a set of tool definitions and asked to produce "
            "realistic (user_query, assistant_response) pairs.\n\n"
            f"## Tool Category: {category.name}\n"
            f"{category.description}\n\n"
            f"## Available Tools\n```json\n{tools_json}\n```\n\n"
            "## Output Format\n"
            "For EACH sample, produce exactly this structure (no extra keys):\n"
            "```\n"
            "SAMPLE_START\n"
            "USER: <natural language query>\n"
            "ASSISTANT: <reasoning, then a tool call block>\n"
            f"{TOOL_CALL_OPEN}\n"
            '{"name": "<tool_name>", "arguments": {<args>}}\n'
            f"{TOOL_CALL_CLOSE}\n"
            f"{TOOL_RESULT_OPEN}\n"
            "<simulated tool output as JSON>\n"
            f"{TOOL_RESULT_CLOSE}\n"
            "ASSISTANT: <final answer incorporating the tool result>\n"
            "SAMPLE_END\n"
            "```\n"
            "Vary the queries in style, complexity, and specificity.\n"
        )

    def _build_positive_prompt(self, category: ToolCategory, n: int) -> str:
        tool_names = ", ".join(t.name for t in category.tools)
        return (
            f"Generate {n} diverse POSITIVE examples where the user's query "
            f"requires using one of these tools: [{tool_names}].\n"
            "Make the queries realistic and varied \u2014 some simple, some multi-step.\n"
            "Each query MUST result in a tool call."
        )

    def _build_negative_prompt(self, category: ToolCategory, n: int) -> str:
        return (
            f"Generate {n} NEGATIVE examples where the user's query does NOT "
            f"require any tool from the {category.name} category.\n"
            "The assistant should answer directly without using any tools.\n"
            "For negative samples, do NOT include <tool_call> blocks \u2014 "
            "just USER and ASSISTANT turns."
        )

    # ------------------------------------------------------------------
    # Parsing teacher output
    # ------------------------------------------------------------------

    def _parse_teacher_output(
        self,
        raw: str,
        category: ToolCategory,
        is_positive: bool,
    ) -> list[GeneratedSample]:
        """Best-effort parse of the teacher's structured output."""
        samples: list[GeneratedSample] = []
        blocks = raw.split("SAMPLE_START")

        for block in blocks:
            block = block.strip()
            if not block or "USER:" not in block:
                continue

            # Trim everything after SAMPLE_END if present
            if "SAMPLE_END" in block:
                block = block[: block.index("SAMPLE_END")]

            messages: list[dict[str, str]] = []

            # Extract USER turn
            user_start = block.find("USER:")
            if user_start == -1:
                continue
            after_user = block[user_start + len("USER:") :]
            # Find next ASSISTANT:
            asst_start = after_user.find("ASSISTANT:")
            user_text = (
                after_user[:asst_start].strip() if asst_start != -1 else after_user.strip()
            )
            messages.append({"role": "user", "content": user_text})

            # Everything after first ASSISTANT: is the assistant content
            if asst_start != -1:
                assistant_raw = after_user[asst_start + len("ASSISTANT:") :]

                # Check for a second ASSISTANT: turn (after tool_result)
                second_asst = assistant_raw.find("ASSISTANT:")
                if second_asst != -1:
                    first_part = assistant_raw[:second_asst].strip()
                    second_part = assistant_raw[second_asst + len("ASSISTANT:") :].strip()
                    messages.append({"role": "assistant", "content": first_part})
                    messages.append({"role": "assistant", "content": second_part})
                else:
                    messages.append(
                        {"role": "assistant", "content": assistant_raw.strip()}
                    )

            if messages:
                samples.append(
                    GeneratedSample(
                        id=str(uuid.uuid4()),
                        category=category.name,
                        messages=messages,
                        is_positive=is_positive,
                        metadata={"teacher": self.teacher.name},
                    )
                )

        return samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        n_samples: int,
        category: ToolCategory,
        *,
        negative_ratio: float = 0.2,
    ) -> list[GeneratedSample]:
        """
        Generate *n_samples* training examples for *category*.

        Args:
            n_samples: Total samples to produce.
            category: The tool category to target.
            negative_ratio: Fraction of samples that should be negative
                            (no tool call expected).

        Returns:
            List of GeneratedSample objects.
        """
        n_negative = max(1, int(n_samples * negative_ratio))
        n_positive = n_samples - n_negative

        all_samples: list[GeneratedSample] = []
        system_prompt = self._build_system_prompt(category)

        # --- positive samples ---
        remaining = n_positive
        with tqdm(total=n_positive, desc=f"Positive ({category.name})") as pbar:
            while remaining > 0:
                batch_size = min(remaining, self._SAMPLES_PER_CALL)
                prompt = self._build_positive_prompt(category, batch_size)
                resp = self.teacher.generate(system_prompt, prompt)
                parsed = self._parse_teacher_output(resp.text, category, is_positive=True)
                all_samples.extend(parsed)
                got = len(parsed)
                pbar.update(got)
                remaining -= got
                if got == 0:
                    logger.warning("Teacher returned 0 parseable positive samples; breaking.")
                    break

        # --- negative samples ---
        remaining = n_negative
        with tqdm(total=n_negative, desc=f"Negative ({category.name})") as pbar:
            while remaining > 0:
                batch_size = min(remaining, self._SAMPLES_PER_CALL)
                prompt = self._build_negative_prompt(category, batch_size)
                resp = self.teacher.generate(system_prompt, prompt)
                parsed = self._parse_teacher_output(resp.text, category, is_positive=False)
                all_samples.extend(parsed)
                got = len(parsed)
                pbar.update(got)
                remaining -= got
                if got == 0:
                    logger.warning("Teacher returned 0 parseable negative samples; breaking.")
                    break

        logger.info(
            "Generated %d samples for category '%s' (%d positive, %d negative)",
            len(all_samples),
            category.name,
            sum(1 for s in all_samples if s.is_positive),
            sum(1 for s in all_samples if not s.is_positive),
        )
        return all_samples

    def save_jsonl(
        self,
        samples: list[GeneratedSample],
        filename: str | None = None,
    ) -> Path:
        """
        Persist samples to a JSONL file.

        Args:
            samples: Samples to save.
            filename: Filename (default: auto-generated from category + timestamp).

        Returns:
            Path to the written file.
        """
        if not filename:
            ts = int(time.time())
            cat = samples[0].category if samples else "unknown"
            filename = f"{cat}_{ts}.jsonl"

        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample.to_jsonl_dict(), ensure_ascii=False) + "\n")

        logger.info("Saved %d samples to %s", len(samples), path)
        return path

    def generate_and_save(
        self,
        n_samples: int,
        category: ToolCategory,
        *,
        negative_ratio: float = 0.2,
        filename: str | None = None,
    ) -> Path:
        """Convenience: generate + save in one call."""
        samples = self.generate_batch(
            n_samples, category, negative_ratio=negative_ratio
        )
        return self.save_jsonl(samples, filename=filename)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_teacher(
    provider: str = "dry_run",
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> TeacherModel:
    """
    Factory to instantiate the right TeacherModel implementation.

    Args:
        provider: One of "claude", "openai", "dry_run".
        model: Model name override.
        api_key: API key override (otherwise taken from env).
    """
    provider = provider.lower().replace("-", "_")

    if provider in ("claude", "anthropic"):
        kw: dict[str, Any] = {}
        if model:
            kw["model"] = model
        if api_key:
            kw["api_key"] = api_key
        kw.update(kwargs)
        return ClaudeTeacher(**kw)

    if provider == "openai":
        kw = {}
        if model:
            kw["model"] = model
        if api_key:
            kw["api_key"] = api_key
        kw.update(kwargs)
        return OpenAITeacher(**kw)

    if provider in ("dry_run", "dryrun", "dry"):
        return DryRunTeacher()

    raise ValueError(f"Unknown teacher provider: {provider!r}")
