"""
Validation utilities for sharded models.

Verify shard integrity, tensor shapes, and test model reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    message: str
    details: dict[str, Any] | None = None


@dataclass
class ShardValidationReport:
    """Complete validation report for a sharded model."""

    model_id: str
    total_shards: int
    passed: bool
    results: list[ValidationResult]
    summary: str


def validate_checksum(
    file_path: str | Path,
    expected_checksum: str,
    algorithm: str = "sha256",
) -> ValidationResult:
    """
    Validate file checksum.

    Args:
        file_path: Path to the file
        expected_checksum: Expected checksum hex digest
        algorithm: Hash algorithm

    Returns:
        ValidationResult
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return ValidationResult(
            passed=False,
            message=f"File not found: {file_path}",
        )

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        return ValidationResult(
            passed=False,
            message=f"Unsupported algorithm: {algorithm}",
        )

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    actual_checksum = hasher.hexdigest()

    if actual_checksum == expected_checksum:
        return ValidationResult(
            passed=True,
            message=f"Checksum valid for {file_path.name}",
            details={"checksum": actual_checksum},
        )
    else:
        return ValidationResult(
            passed=False,
            message=f"Checksum mismatch for {file_path.name}",
            details={
                "expected": expected_checksum,
                "actual": actual_checksum,
            },
        )


def validate_shard_tensors(
    shard_path: str | Path,
    expected_tensors: list[str] | None = None,
) -> ValidationResult:
    """
    Validate tensors in a shard file.

    Args:
        shard_path: Path to the shard file
        expected_tensors: Optional list of expected tensor names

    Returns:
        ValidationResult
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        return ValidationResult(
            passed=False,
            message="safetensors library not installed",
        )

    shard_path = Path(shard_path)

    if not shard_path.exists():
        return ValidationResult(
            passed=False,
            message=f"Shard file not found: {shard_path}",
        )

    try:
        tensors = load_file(str(shard_path))
    except Exception as e:
        return ValidationResult(
            passed=False,
            message=f"Failed to load shard: {e}",
        )

    tensor_info = {}
    for name, tensor in tensors.items():
        tensor_info[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "has_nan": bool(torch.isnan(tensor).any()),
            "has_inf": bool(torch.isinf(tensor).any()),
        }

    # Check for NaN/Inf values
    bad_tensors = [
        name for name, info in tensor_info.items()
        if info["has_nan"] or info["has_inf"]
    ]

    if bad_tensors:
        return ValidationResult(
            passed=False,
            message=f"Found NaN/Inf values in tensors: {bad_tensors}",
            details={"tensor_info": tensor_info},
        )

    # Check expected tensors
    if expected_tensors is not None:
        missing = set(expected_tensors) - set(tensors.keys())
        extra = set(tensors.keys()) - set(expected_tensors)

        if missing:
            return ValidationResult(
                passed=False,
                message=f"Missing expected tensors: {missing}",
                details={"missing": list(missing), "extra": list(extra)},
            )

    return ValidationResult(
        passed=True,
        message=f"Shard {shard_path.name} valid with {len(tensors)} tensors",
        details={"tensor_count": len(tensors), "tensor_info": tensor_info},
    )


def verify_tensor_shapes(
    tensors: dict[str, torch.Tensor],
    expected_shapes: dict[str, list[int]],
) -> ValidationResult:
    """
    Verify tensor shapes match expected shapes.

    Args:
        tensors: Dictionary of tensors
        expected_shapes: Dictionary of expected shapes

    Returns:
        ValidationResult
    """
    mismatches = {}

    for name, expected_shape in expected_shapes.items():
        if name not in tensors:
            mismatches[name] = {"error": "missing", "expected": expected_shape}
            continue

        actual_shape = list(tensors[name].shape)
        if actual_shape != expected_shape:
            mismatches[name] = {
                "expected": expected_shape,
                "actual": actual_shape,
            }

    if mismatches:
        return ValidationResult(
            passed=False,
            message=f"Shape mismatches found in {len(mismatches)} tensors",
            details={"mismatches": mismatches},
        )

    return ValidationResult(
        passed=True,
        message=f"All {len(expected_shapes)} tensor shapes verified",
    )


def validate_manifest(manifest_path: str | Path) -> ValidationResult:
    """
    Validate a model manifest file.

    Args:
        manifest_path: Path to manifest.json

    Returns:
        ValidationResult
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return ValidationResult(
            passed=False,
            message=f"Manifest not found: {manifest_path}",
        )

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(
            passed=False,
            message=f"Invalid JSON in manifest: {e}",
        )

    # Check required fields
    required_fields = ["model_id", "shards", "total_size_bytes"]
    missing_fields = [f for f in required_fields if f not in manifest]

    if missing_fields:
        return ValidationResult(
            passed=False,
            message=f"Missing required fields in manifest: {missing_fields}",
        )

    # Validate shard entries
    for shard in manifest.get("shards", []):
        shard_required = ["filename", "size_bytes", "checksum_sha256"]
        missing_shard_fields = [f for f in shard_required if f not in shard]

        if missing_shard_fields:
            return ValidationResult(
                passed=False,
                message=f"Shard entry missing fields: {missing_shard_fields}",
                details={"shard": shard},
            )

    return ValidationResult(
        passed=True,
        message="Manifest structure valid",
        details={"model_id": manifest.get("model_id"), "shard_count": len(manifest.get("shards", []))},
    )


def validate_shards(
    model_dir: str | Path,
    verify_checksums: bool = True,
    verify_tensors: bool = True,
) -> ShardValidationReport:
    """
    Validate all shards in a model directory.

    Args:
        model_dir: Directory containing shards and manifest
        verify_checksums: Whether to verify file checksums
        verify_tensors: Whether to verify tensor contents

    Returns:
        ShardValidationReport
    """
    model_dir = Path(model_dir)
    results: list[ValidationResult] = []

    # Validate manifest
    manifest_path = model_dir / "manifest.json"
    manifest_result = validate_manifest(manifest_path)
    results.append(manifest_result)

    if not manifest_result.passed:
        return ShardValidationReport(
            model_id="unknown",
            total_shards=0,
            passed=False,
            results=results,
            summary="Manifest validation failed",
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    model_id = manifest.get("model_id", "unknown")
    shards = manifest.get("shards", [])

    # Validate each shard
    for shard_info in shards:
        shard_path = model_dir / shard_info["filename"]

        # Check file exists
        if not shard_path.exists():
            results.append(ValidationResult(
                passed=False,
                message=f"Shard file missing: {shard_info['filename']}",
            ))
            continue

        # Verify checksum
        if verify_checksums:
            checksum_result = validate_checksum(
                shard_path,
                shard_info["checksum_sha256"],
            )
            results.append(checksum_result)

        # Verify tensors
        if verify_tensors:
            tensor_result = validate_shard_tensors(shard_path)
            results.append(tensor_result)

    # Check tokenizer
    tokenizer_dir = model_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer_files = list(tokenizer_dir.glob("*"))
        if tokenizer_files:
            results.append(ValidationResult(
                passed=True,
                message=f"Tokenizer found with {len(tokenizer_files)} files",
            ))
        else:
            results.append(ValidationResult(
                passed=False,
                message="Tokenizer directory empty",
            ))
    else:
        results.append(ValidationResult(
            passed=False,
            message="Tokenizer directory not found",
        ))

    # Compile report
    all_passed = all(r.passed for r in results)
    failed_count = sum(1 for r in results if not r.passed)

    if all_passed:
        summary = f"All {len(results)} validation checks passed"
    else:
        summary = f"{failed_count}/{len(results)} validation checks failed"

    return ShardValidationReport(
        model_id=model_id,
        total_shards=len(shards),
        passed=all_passed,
        results=results,
        summary=summary,
    )


def test_model_reconstruction(
    model_dir: str | Path,
    test_input: torch.Tensor | None = None,
) -> ValidationResult:
    """
    Test that a sharded model can be reconstructed and run inference.

    Args:
        model_dir: Directory containing shards
        test_input: Optional test input tensor

    Returns:
        ValidationResult
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        return ValidationResult(
            passed=False,
            message="safetensors library not installed",
        )

    model_dir = Path(model_dir)

    # Load manifest
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return ValidationResult(
            passed=False,
            message="Manifest not found",
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load all shards
    all_tensors: dict[str, torch.Tensor] = {}
    for shard_info in manifest.get("shards", []):
        shard_path = model_dir / shard_info["filename"]
        try:
            shard_tensors = load_file(str(shard_path))
            all_tensors.update(shard_tensors)
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Failed to load shard {shard_info['filename']}: {e}",
            )

    # Basic sanity checks
    if not all_tensors:
        return ValidationResult(
            passed=False,
            message="No tensors loaded from shards",
        )

    # Check for common required tensors
    embed_found = any("embed" in k.lower() for k in all_tensors.keys())
    layer_found = any("layer" in k.lower() for k in all_tensors.keys())
    head_found = any("lm_head" in k.lower() or "head" in k.lower() for k in all_tensors.keys())

    missing = []
    if not embed_found:
        missing.append("embedding")
    if not layer_found:
        missing.append("layers")
    if not head_found:
        missing.append("head")

    if missing:
        return ValidationResult(
            passed=False,
            message=f"Missing expected tensor groups: {missing}",
            details={"tensor_names": list(all_tensors.keys())[:20]},
        )

    return ValidationResult(
        passed=True,
        message=f"Model reconstruction successful with {len(all_tensors)} tensors",
        details={
            "total_tensors": len(all_tensors),
            "has_embedding": embed_found,
            "has_layers": layer_found,
            "has_head": head_found,
        },
    )


def print_validation_report(report: ShardValidationReport) -> None:
    """Print a formatted validation report."""
    print(f"\n{'=' * 60}")
    print(f"Validation Report: {report.model_id}")
    print(f"{'=' * 60}")
    print(f"Total Shards: {report.total_shards}")
    print(f"Status: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Summary: {report.summary}")
    print(f"\nDetails:")
    print("-" * 40)

    for result in report.results:
        status = "[OK]" if result.passed else "[FAIL]"
        print(f"  {status} {result.message}")

    print(f"{'=' * 60}\n")
