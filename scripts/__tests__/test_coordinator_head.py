"""Tests for the head-shard loader and HeadModule forward pass.

Strategy: write a small synthetic safetensors file with a known
``lm_head.weight`` and (optionally) ``model.final_layernorm.weight``, load it
through ``load_head_from_safetensors``, and assert the forward produces the
expected logits for a known input.

This does not exercise the S3 download path; that is covered separately by
``test_ensure_head_shard_skips_when_cached`` and is impossible to mock
end-to-end without the real S3 endpoint. The S3 *contract* is documented in
the runbook and verified manually by the operator before deploy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from scripts.expert_coordinator import (
    CoordinatorConfig,
    HeadModule,
    _ensure_head_shard,
    load_head_from_safetensors,
)


HIDDEN = 1536
VOCAB = 64


def _make_synthetic_head_shard(path: Path, *, include_layernorm: bool = True) -> dict:
    torch.manual_seed(0)
    lm_head_w = torch.randn(VOCAB, HIDDEN, dtype=torch.float16)
    state: dict[str, torch.Tensor] = {"lm_head.weight": lm_head_w}
    if include_layernorm:
        # Use a non-uniform layernorm gain so the layernorm has a measurable
        # effect.
        state["model.final_layernorm.weight"] = torch.full(
            (HIDDEN,), 0.7, dtype=torch.float16,
        )
    save_file(state, str(path))
    return state


def test_load_head_lm_head_only(tmp_path: Path) -> None:
    path = tmp_path / "shard_head.safetensors"
    state = _make_synthetic_head_shard(path, include_layernorm=False)
    head = load_head_from_safetensors(path)
    assert head.vocab == VOCAB
    assert head.hidden == HIDDEN
    assert head.final_layernorm_weight is None
    # forward(hidden) == hidden @ lm_head_w.T
    hidden = torch.randn(2, HIDDEN, dtype=torch.float32)
    logits = head.forward(hidden)
    expected = hidden @ state["lm_head.weight"].to(torch.float32).T
    assert torch.allclose(logits, expected, atol=1e-3, rtol=1e-3)


def test_load_head_with_layernorm(tmp_path: Path) -> None:
    path = tmp_path / "shard_head.safetensors"
    state = _make_synthetic_head_shard(path, include_layernorm=True)
    head = load_head_from_safetensors(path)
    assert head.final_layernorm_weight is not None
    hidden = torch.randn(1, HIDDEN, dtype=torch.float32)
    logits = head.forward(hidden)
    # Replicate the reference RMSNorm + matmul.
    eps = head.layernorm_eps
    var = hidden.pow(2).mean(dim=-1, keepdim=True)
    normed = hidden * torch.rsqrt(var + eps)
    normed = normed * state["model.final_layernorm.weight"].to(torch.float32)
    expected = normed @ state["lm_head.weight"].to(torch.float32).T
    assert torch.allclose(logits, expected, atol=1e-3, rtol=1e-3)


def test_load_head_missing_lm_head_raises(tmp_path: Path) -> None:
    path = tmp_path / "shard_head.safetensors"
    # Save a file without lm_head.weight.
    save_file({"foo": torch.zeros(4)}, str(path))
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        load_head_from_safetensors(path)


def test_head_module_forward_is_deterministic(tmp_path: Path) -> None:
    """Same input must produce identical logits on re-invocation."""

    path = tmp_path / "shard_head.safetensors"
    _make_synthetic_head_shard(path, include_layernorm=True)
    head = load_head_from_safetensors(path)
    hidden = torch.randn(1, HIDDEN, dtype=torch.float32)
    a = head.forward(hidden)
    b = head.forward(hidden)
    assert torch.equal(a, b)


def test_ensure_head_shard_skips_when_cached(tmp_path: Path) -> None:
    path = tmp_path / "shard_head.safetensors"
    path.write_bytes(b"already cached")
    cfg = CoordinatorConfig()
    cfg.head_local_path = path
    # Should NOT raise, even with no S3 creds, because the file exists.
    _ensure_head_shard(cfg)
    # File contents untouched.
    assert path.read_bytes() == b"already cached"


def test_ensure_head_shard_no_s3_raises(tmp_path: Path) -> None:
    cfg = CoordinatorConfig()
    cfg.head_local_path = tmp_path / "absent_head.safetensors"
    cfg.head_s3_bucket = None
    with pytest.raises(SystemExit):
        _ensure_head_shard(cfg)


def test_ensure_head_shard_partial_s3_raises(tmp_path: Path) -> None:
    cfg = CoordinatorConfig()
    cfg.head_local_path = tmp_path / "absent_head.safetensors"
    cfg.head_s3_bucket = "some-bucket"
    cfg.head_s3_access_key = "ak"
    # secret key + key missing
    with pytest.raises(SystemExit):
        _ensure_head_shard(cfg)
