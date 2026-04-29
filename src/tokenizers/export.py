"""
Tokenizer export utilities for web-based inference.

Exports tokenizers in formats suitable for browser-based use.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenizerExportConfig:
    """Configuration for tokenizer export."""

    output_dir: Path
    include_special_tokens: bool = True
    include_added_tokens: bool = True
    export_vocab_json: bool = True
    export_merges: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class TokenizerExportResult:
    """Result of tokenizer export operation."""

    success: bool
    output_dir: Path
    files: list[str]
    vocab_size: int
    tokenizer_type: str
    message: str


def export_tokenizer(
    tokenizer: Any,
    config: TokenizerExportConfig,
) -> TokenizerExportResult:
    """
    Export a tokenizer for web-based inference.

    Supports HuggingFace tokenizers (fast and slow).

    Args:
        tokenizer: HuggingFace tokenizer object
        config: Export configuration

    Returns:
        TokenizerExportResult with export details
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    exported_files: list[str] = []

    # Determine tokenizer type
    tokenizer_class = type(tokenizer).__name__
    is_fast = hasattr(tokenizer, "backend_tokenizer")

    logger.info(f"Exporting {tokenizer_class} (fast={is_fast})")

    try:
        # Save using HuggingFace's built-in method
        tokenizer.save_pretrained(str(config.output_dir))

        # List exported files
        for file_path in config.output_dir.iterdir():
            if file_path.is_file():
                exported_files.append(file_path.name)

        # Create additional web-friendly exports if requested
        if config.export_vocab_json and is_fast:
            _export_vocab_json(tokenizer, config.output_dir)
            if "vocab.json" not in exported_files:
                exported_files.append("vocab.json")

        # Export special tokens config
        if config.include_special_tokens:
            _export_special_tokens(tokenizer, config.output_dir)
            if "special_tokens.json" not in exported_files:
                exported_files.append("special_tokens.json")

        vocab_size = len(tokenizer)

        return TokenizerExportResult(
            success=True,
            output_dir=config.output_dir,
            files=exported_files,
            vocab_size=vocab_size,
            tokenizer_type=tokenizer_class,
            message=f"Successfully exported {len(exported_files)} files",
        )

    except Exception as e:
        logger.error(f"Failed to export tokenizer: {e}")
        return TokenizerExportResult(
            success=False,
            output_dir=config.output_dir,
            files=exported_files,
            vocab_size=0,
            tokenizer_type=tokenizer_class,
            message=f"Export failed: {e}",
        )


def _export_vocab_json(tokenizer: Any, output_dir: Path) -> None:
    """Export vocabulary as JSON for web use."""
    try:
        vocab = tokenizer.get_vocab()
        vocab_path = output_dir / "vocab.json"

        # Sort by token ID for consistency
        sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)

        logger.debug(f"Exported vocab.json with {len(vocab)} tokens")
    except Exception as e:
        logger.warning(f"Failed to export vocab.json: {e}")


def _export_special_tokens(tokenizer: Any, output_dir: Path) -> None:
    """Export special tokens configuration."""
    special_tokens = {
        "bos_token": _get_token_info(tokenizer, "bos_token"),
        "eos_token": _get_token_info(tokenizer, "eos_token"),
        "pad_token": _get_token_info(tokenizer, "pad_token"),
        "unk_token": _get_token_info(tokenizer, "unk_token"),
        "sep_token": _get_token_info(tokenizer, "sep_token"),
        "cls_token": _get_token_info(tokenizer, "cls_token"),
        "mask_token": _get_token_info(tokenizer, "mask_token"),
    }

    # Get additional special tokens
    if hasattr(tokenizer, "additional_special_tokens"):
        special_tokens["additional_special_tokens"] = tokenizer.additional_special_tokens

    # Filter out None values
    special_tokens = {k: v for k, v in special_tokens.items() if v is not None}

    special_path = output_dir / "special_tokens.json"
    with open(special_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)

    logger.debug(f"Exported special_tokens.json")


def _get_token_info(tokenizer: Any, token_name: str) -> dict[str, Any] | None:
    """Get information about a special token."""
    token = getattr(tokenizer, token_name, None)
    if token is None:
        return None

    token_id = getattr(tokenizer, f"{token_name}_id", None)
    if token_id is None:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            pass

    return {
        "token": str(token),
        "id": token_id,
    }


def export_tokenizer_from_model_id(
    model_id: str,
    output_dir: str | Path,
    trust_remote_code: bool = False,
) -> TokenizerExportResult:
    """
    Export tokenizer directly from a HuggingFace model ID.

    Args:
        model_id: HuggingFace model ID
        output_dir: Output directory
        trust_remote_code: Whether to trust remote code

    Returns:
        TokenizerExportResult
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return TokenizerExportResult(
            success=False,
            output_dir=Path(output_dir),
            files=[],
            vocab_size=0,
            tokenizer_type="unknown",
            message="transformers library not installed",
        )

    logger.info(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    config = TokenizerExportConfig(output_dir=Path(output_dir))
    return export_tokenizer(tokenizer, config)


def create_tokenizer_manifest(
    tokenizer_dir: str | Path,
) -> dict[str, Any]:
    """
    Create a manifest for the exported tokenizer.

    Args:
        tokenizer_dir: Directory containing exported tokenizer

    Returns:
        Manifest dictionary
    """
    tokenizer_dir = Path(tokenizer_dir)

    manifest = {
        "type": "tokenizer",
        "files": [],
        "config": {},
    }

    # List all files with sizes
    for file_path in tokenizer_dir.iterdir():
        if file_path.is_file():
            manifest["files"].append({
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
            })

    # Try to load tokenizer config
    config_path = tokenizer_dir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            manifest["config"] = json.load(f)

    # Try to load special tokens
    special_path = tokenizer_dir / "special_tokens.json"
    if special_path.exists():
        with open(special_path) as f:
            manifest["special_tokens"] = json.load(f)

    return manifest
