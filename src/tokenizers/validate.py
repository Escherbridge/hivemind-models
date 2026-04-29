"""
Tokenizer validation utilities.

Validate exported tokenizers for correctness and web compatibility.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenizerValidationResult:
    """Result of tokenizer validation."""

    valid: bool
    tokenizer_type: str
    vocab_size: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def validate_tokenizer(
    tokenizer_dir: str | Path,
    test_strings: list[str] | None = None,
) -> TokenizerValidationResult:
    """
    Validate an exported tokenizer.

    Args:
        tokenizer_dir: Directory containing exported tokenizer
        test_strings: Optional strings to test tokenization

    Returns:
        TokenizerValidationResult
    """
    tokenizer_dir = Path(tokenizer_dir)
    issues: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    # Check directory exists
    if not tokenizer_dir.exists():
        return TokenizerValidationResult(
            valid=False,
            tokenizer_type="unknown",
            vocab_size=0,
            issues=[f"Tokenizer directory not found: {tokenizer_dir}"],
        )

    # Check for required files
    required_files = ["tokenizer.json", "tokenizer_config.json"]
    optional_files = ["vocab.json", "merges.txt", "special_tokens_map.json"]

    found_files = [f.name for f in tokenizer_dir.iterdir() if f.is_file()]
    details["found_files"] = found_files

    # Check for tokenizer.json (fast tokenizer)
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    has_fast_tokenizer = tokenizer_json_path.exists()

    if not has_fast_tokenizer:
        # Check for vocab.json + merges.txt (slow tokenizer)
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"

        if not vocab_path.exists():
            issues.append("Missing tokenizer.json or vocab.json")
        else:
            warnings.append("Using slow tokenizer format (vocab.json)")

    # Load and validate tokenizer config
    config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_type = "unknown"
    vocab_size = 0

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)

            tokenizer_type = config.get("tokenizer_class", "unknown")
            details["config"] = config
        except json.JSONDecodeError as e:
            issues.append(f"Invalid tokenizer_config.json: {e}")
    else:
        warnings.append("Missing tokenizer_config.json")

    # Try to get vocab size
    if has_fast_tokenizer:
        try:
            with open(tokenizer_json_path) as f:
                tokenizer_data = json.load(f)

            model_data = tokenizer_data.get("model", {})
            vocab = model_data.get("vocab", {})
            vocab_size = len(vocab)
            details["vocab_size"] = vocab_size
        except Exception as e:
            warnings.append(f"Could not parse tokenizer.json: {e}")
    else:
        vocab_path = tokenizer_dir / "vocab.json"
        if vocab_path.exists():
            try:
                with open(vocab_path) as f:
                    vocab = json.load(f)
                vocab_size = len(vocab)
                details["vocab_size"] = vocab_size
            except Exception as e:
                warnings.append(f"Could not parse vocab.json: {e}")

    # Validate special tokens
    special_tokens_result = _validate_special_tokens(tokenizer_dir)
    if special_tokens_result["issues"]:
        issues.extend(special_tokens_result["issues"])
    if special_tokens_result["warnings"]:
        warnings.extend(special_tokens_result["warnings"])
    details["special_tokens"] = special_tokens_result.get("tokens", {})

    # Test tokenization if tokenizer can be loaded
    if test_strings:
        tokenization_result = _test_tokenization(tokenizer_dir, test_strings)
        if tokenization_result["issues"]:
            issues.extend(tokenization_result["issues"])
        details["tokenization_tests"] = tokenization_result.get("results", [])

    # Determine validity
    valid = len(issues) == 0

    return TokenizerValidationResult(
        valid=valid,
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        issues=issues,
        warnings=warnings,
        details=details,
    )


def _validate_special_tokens(tokenizer_dir: Path) -> dict[str, Any]:
    """Validate special tokens configuration."""
    result: dict[str, Any] = {
        "issues": [],
        "warnings": [],
        "tokens": {},
    }

    # Check special_tokens_map.json
    special_map_path = tokenizer_dir / "special_tokens_map.json"
    if special_map_path.exists():
        try:
            with open(special_map_path) as f:
                special_map = json.load(f)
            result["tokens"] = special_map
        except json.JSONDecodeError:
            result["issues"].append("Invalid special_tokens_map.json")
            return result

        # Check required special tokens
        required_tokens = ["bos_token", "eos_token"]
        for token_name in required_tokens:
            if token_name not in special_map:
                result["warnings"].append(f"Missing recommended token: {token_name}")

    else:
        # Try to get from tokenizer_config.json
        config_path = tokenizer_dir / "tokenizer_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                for key in ["bos_token", "eos_token", "pad_token", "unk_token"]:
                    if key in config:
                        result["tokens"][key] = config[key]
            except Exception:
                pass

        if not result["tokens"]:
            result["warnings"].append("No special tokens configuration found")

    return result


def _test_tokenization(
    tokenizer_dir: Path,
    test_strings: list[str],
) -> dict[str, Any]:
    """Test tokenization with sample strings."""
    result: dict[str, Any] = {
        "issues": [],
        "results": [],
    }

    try:
        from transformers import AutoTokenizer
    except ImportError:
        result["issues"].append("transformers not installed, skipping tokenization tests")
        return result

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    except Exception as e:
        result["issues"].append(f"Failed to load tokenizer: {e}")
        return result

    for test_string in test_strings:
        try:
            tokens = tokenizer.encode(test_string)
            decoded = tokenizer.decode(tokens)

            result["results"].append({
                "input": test_string,
                "tokens": tokens,
                "token_count": len(tokens),
                "decoded": decoded,
                "roundtrip_match": decoded.strip() == test_string.strip(),
            })
        except Exception as e:
            result["issues"].append(f"Tokenization failed for '{test_string[:20]}...': {e}")

    return result


def validate_tokenizer_for_web(
    tokenizer_dir: str | Path,
) -> TokenizerValidationResult:
    """
    Validate tokenizer specifically for web browser use.

    Checks for web-specific requirements.

    Args:
        tokenizer_dir: Directory containing exported tokenizer

    Returns:
        TokenizerValidationResult
    """
    tokenizer_dir = Path(tokenizer_dir)

    # Run standard validation first
    result = validate_tokenizer(tokenizer_dir)

    # Additional web-specific checks
    web_issues: list[str] = []
    web_warnings: list[str] = []

    # Check for tokenizer.json (required for web)
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        web_issues.append(
            "tokenizer.json required for web use. "
            "Use a fast tokenizer or convert the tokenizer."
        )

    # Check file sizes (large files may be slow to load in browser)
    max_size_mb = 50
    for file_path in tokenizer_dir.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                web_warnings.append(
                    f"Large file {file_path.name} ({size_mb:.1f} MB) "
                    f"may be slow to load in browser"
                )

    # Update result with web-specific issues
    result.issues.extend(web_issues)
    result.warnings.extend(web_warnings)
    result.valid = len(result.issues) == 0

    return result


def print_validation_result(result: TokenizerValidationResult) -> None:
    """Print a formatted validation result."""
    print(f"\n{'=' * 50}")
    print(f"Tokenizer Validation: {'PASSED' if result.valid else 'FAILED'}")
    print(f"{'=' * 50}")
    print(f"Type: {result.tokenizer_type}")
    print(f"Vocab Size: {result.vocab_size}")

    if result.issues:
        print(f"\nIssues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"  [ERROR] {issue}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  [WARN] {warning}")

    print(f"{'=' * 50}\n")
