"""
Safetensors I/O helpers.

Utilities for reading and writing safetensors files.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_safetensors(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> int:
    """
    Save tensors to a safetensors file.

    Args:
        tensors: Dictionary of tensor name -> tensor
        path: Output file path
        metadata: Optional metadata to include in the file

    Returns:
        Size of the saved file in bytes
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("safetensors library required. Install with: pip install safetensors")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure tensors are on CPU and contiguous
    cpu_tensors = {}
    for name, tensor in tensors.items():
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        cpu_tensors[name] = tensor

    # Save with metadata if provided
    if metadata:
        save_file(cpu_tensors, str(path), metadata=metadata)
    else:
        save_file(cpu_tensors, str(path))

    file_size = path.stat().st_size
    logger.debug(f"Saved {len(tensors)} tensors to {path} ({file_size} bytes)")
    return file_size


def load_safetensors(
    path: str | Path,
    device: str | torch.device = "cpu",
    tensors: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Load tensors from a safetensors file.

    Args:
        path: Input file path
        device: Device to load tensors to
        tensors: Optional list of specific tensors to load (loads all if None)

    Returns:
        Dictionary of tensor name -> tensor
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors library required. Install with: pip install safetensors")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    loaded = load_file(str(path), device=str(device))

    if tensors is not None:
        loaded = {k: v for k, v in loaded.items() if k in tensors}

    logger.debug(f"Loaded {len(loaded)} tensors from {path}")
    return loaded


def get_safetensors_metadata(path: str | Path) -> dict[str, Any]:
    """
    Get metadata from a safetensors file without loading tensors.

    Args:
        path: Input file path

    Returns:
        Dictionary with file metadata and tensor information
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors library required. Install with: pip install safetensors")

    path = Path(path)

    with safe_open(str(path), framework="pt") as f:
        tensor_names = f.keys()
        metadata = f.metadata() or {}

        tensor_info = {}
        for name in tensor_names:
            tensor = f.get_tensor(name)
            tensor_info[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "size_bytes": tensor.numel() * tensor.element_size(),
            }

    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "metadata": metadata,
        "tensor_count": len(tensor_names),
        "tensors": tensor_info,
    }


def compute_file_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute checksum of a file.

    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hex digest of the checksum
    """
    path = Path(path)

    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    with open(path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def merge_safetensors(
    input_paths: list[str | Path],
    output_path: str | Path,
    check_duplicates: bool = True,
) -> int:
    """
    Merge multiple safetensors files into one.

    Args:
        input_paths: List of input file paths
        output_path: Output file path
        check_duplicates: Whether to check for duplicate tensor names

    Returns:
        Size of the merged file in bytes
    """
    merged_tensors: dict[str, torch.Tensor] = {}

    for path in input_paths:
        tensors = load_safetensors(path)

        if check_duplicates:
            duplicates = set(merged_tensors.keys()) & set(tensors.keys())
            if duplicates:
                raise ValueError(f"Duplicate tensor names found: {duplicates}")

        merged_tensors.update(tensors)

    return save_safetensors(merged_tensors, output_path)


def split_safetensors(
    input_path: str | Path,
    output_dir: str | Path,
    max_size_bytes: int | None = None,
    tensor_groups: dict[str, list[str]] | None = None,
) -> list[Path]:
    """
    Split a safetensors file into multiple files.

    Args:
        input_path: Input file path
        output_dir: Output directory
        max_size_bytes: Maximum size per output file (optional)
        tensor_groups: Named groups of tensors to split into files (optional)

    Returns:
        List of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tensors = load_safetensors(input_path)
    output_paths: list[Path] = []

    if tensor_groups is not None:
        # Split by named groups
        for group_name, tensor_names in tensor_groups.items():
            group_tensors = {
                name: all_tensors[name]
                for name in tensor_names
                if name in all_tensors
            }
            if group_tensors:
                output_path = output_dir / f"{group_name}.safetensors"
                save_safetensors(group_tensors, output_path)
                output_paths.append(output_path)

    elif max_size_bytes is not None:
        # Split by size
        current_tensors: dict[str, torch.Tensor] = {}
        current_size = 0
        part_idx = 0

        for name, tensor in all_tensors.items():
            tensor_size = tensor.numel() * tensor.element_size()

            if current_size + tensor_size > max_size_bytes and current_tensors:
                # Save current batch
                output_path = output_dir / f"part_{part_idx:04d}.safetensors"
                save_safetensors(current_tensors, output_path)
                output_paths.append(output_path)
                current_tensors = {}
                current_size = 0
                part_idx += 1

            current_tensors[name] = tensor
            current_size += tensor_size

        # Save remaining tensors
        if current_tensors:
            output_path = output_dir / f"part_{part_idx:04d}.safetensors"
            save_safetensors(current_tensors, output_path)
            output_paths.append(output_path)

    else:
        # No splitting criteria, just copy
        output_path = output_dir / Path(input_path).name
        save_safetensors(all_tensors, output_path)
        output_paths.append(output_path)

    return output_paths


def convert_dtype(
    tensors: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
    exclude_patterns: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convert tensors to a different dtype.

    Args:
        tensors: Input tensors
        target_dtype: Target dtype
        exclude_patterns: Patterns for tensor names to exclude from conversion

    Returns:
        Converted tensors
    """
    exclude_patterns = exclude_patterns or []

    converted = {}
    for name, tensor in tensors.items():
        should_exclude = any(pattern in name for pattern in exclude_patterns)

        if should_exclude:
            converted[name] = tensor
        else:
            converted[name] = tensor.to(target_dtype)

    return converted
