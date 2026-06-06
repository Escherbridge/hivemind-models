"""
Registry of architecture handlers, keyed by the lowercase model_type that
appears in HuggingFace configs (e.g. "llama", "granitemoehybrid").

Concrete handlers register themselves at import time via @register_handler.
The shard server (and CLI tools that need to peek at supported archs) calls
get_handler(model_type) to resolve.
"""

from __future__ import annotations

from typing import Callable

from src.architectures.base import ArchitectureHandler


_HANDLERS: dict[str, ArchitectureHandler] = {}
_ALIASES: dict[str, str] = {}  # alternate model_type strings -> canonical name


def register_handler(
    name: str,
    *,
    aliases: list[str] | None = None,
) -> Callable[[type[ArchitectureHandler]], type[ArchitectureHandler]]:
    """
    Class decorator. Registers an ArchitectureHandler class under `name` and
    any extra aliases. Aliases let one handler match multiple HF model_type
    strings (e.g. "llama" and "tinyllama" if HF ever differentiated them).
    """

    def _decorator(cls: type[ArchitectureHandler]) -> type[ArchitectureHandler]:
        instance = cls()
        _HANDLERS[name.lower()] = instance
        if aliases:
            for alias in aliases:
                _ALIASES[alias.lower()] = name.lower()
        return cls

    return _decorator


def get_handler(model_type: str) -> ArchitectureHandler:
    """Resolve a handler by HF model_type (or alias). Raises if unsupported."""
    key = model_type.lower()
    if key in _ALIASES:
        key = _ALIASES[key]
    if key not in _HANDLERS:
        raise ValueError(
            f"No architecture handler registered for model_type={model_type!r}. "
            f"Registered handlers: {sorted(_HANDLERS.keys())}. "
            f"Aliases: {sorted(_ALIASES.keys())}."
        )
    return _HANDLERS[key]


def available_handlers() -> list[str]:
    """All canonical handler names that the registry knows about."""
    return sorted(_HANDLERS.keys())
