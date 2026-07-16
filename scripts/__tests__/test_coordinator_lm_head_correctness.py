"""End-to-end correctness test for the coordinator's lm_head path.

The fixture format and capture procedure are documented in
``scripts/capture_lm_head_reference.py``. Briefly:

- ``scripts/__tests__/fixtures/lm_head_reference.npz`` carries a known
  ``(hidden, lm_head_weight, final_layernorm_weight)`` triple together with
  the top-5 ``(token_id, logit)`` pairs that triple should produce.

- For CI / developer machines we use a SYNTHETIC fixture generated from a
  fixed RNG seed -- the math is identical to the real Granite path
  (RMSNorm then matmul) but the weights are random. This gives a
  structurally valid correctness check; semantic validity comes from the
  operator re-capturing against the real Granite shards before deploy.

- For operator validation we use a GRANITE_REAL fixture; the test path is
  the same.

If the synthetic fixture is missing, the test auto-bootstraps it from the
seed in ``DEFAULT_SEED`` so a fresh checkout does not require running the
capture script before tests pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from scripts.expert_coordinator import (
    Coordinator,
    CoordinatorConfig,
    HeadModule,
    RegionConfig,
    build_app,
    score_lm_head,
)
from scripts.capture_lm_head_reference import DEFAULT_SEED, capture_synthetic


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "lm_head_reference.npz"


class _StubTokenizer:
    def decode(self, ids: list[int]) -> str:
        return "".join(f"<tok {i}>" for i in ids)


def _ensure_synthetic_fixture() -> None:
    """If the .npz is missing, regenerate it from the well-known seed.

    This keeps a fresh checkout green without depending on the operator
    having pre-run the capture script.
    """

    if FIXTURE_PATH.exists():
        return
    bundle = capture_synthetic(seed=DEFAULT_SEED)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE_PATH, **bundle)


@pytest.fixture
def coord_lm_head_fixture() -> dict[str, Any]:
    _ensure_synthetic_fixture()
    data = np.load(FIXTURE_PATH, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return {
        "hidden": data["hidden"],
        "lm_head_weight": data["lm_head_weight"],
        "final_layernorm_weight": data["final_layernorm_weight"],
        "top5_token_ids": data["top5_token_ids"],
        "top5_logits": data["top5_logits"],
        "meta": meta,
    }


def _build_coord_with_head(fixture: dict[str, Any]) -> Coordinator:
    cfg = CoordinatorConfig()
    cfg.regions = [RegionConfig("us-west2", "ws://uw.local", [0, 1, 2])]
    coord = Coordinator(cfg)
    head = HeadModule(
        torch.from_numpy(fixture["lm_head_weight"]),
        torch.from_numpy(fixture["final_layernorm_weight"]),
    )
    coord.head_module = head
    coord.tokenizer = _StubTokenizer()
    return coord


def test_coordinator_lm_head_matches_fixture_top5_ids(
    coord_lm_head_fixture: dict[str, Any],
) -> None:
    """Top-5 token ids must match the fixture exactly."""

    fx = coord_lm_head_fixture
    coord = _build_coord_with_head(fx)

    payload = fx["hidden"].astype(np.float16).tobytes()
    shape = (fx["hidden"].shape[-2], fx["hidden"].shape[-1])
    top_5, ms = score_lm_head(coord, payload, shape, "fp16")
    assert ms >= 0

    got_ids = [e["token_id"] for e in top_5]
    expected_ids = [int(x) for x in fx["top5_token_ids"]]
    assert got_ids == expected_ids, f"expected {expected_ids}, got {got_ids}"


def test_coordinator_lm_head_matches_fixture_logits(
    coord_lm_head_fixture: dict[str, Any],
) -> None:
    """Logit values must match the fixture to 1e-3 absolute tolerance."""

    fx = coord_lm_head_fixture
    coord = _build_coord_with_head(fx)
    payload = fx["hidden"].astype(np.float16).tobytes()
    shape = (fx["hidden"].shape[-2], fx["hidden"].shape[-1])
    top_5, _ = score_lm_head(coord, payload, shape, "fp16")
    got_logits = np.array([e["logit"] for e in top_5], dtype=np.float32)
    expected_logits = fx["top5_logits"].astype(np.float32)
    diff = np.abs(got_logits - expected_logits)
    assert np.all(diff < 1e-3), (
        f"logit drift {diff} > 1e-3 (got={got_logits} expected={expected_logits})"
    )


def test_fixture_metadata_recorded(coord_lm_head_fixture: dict[str, Any]) -> None:
    """The fixture meta must declare its source so the operator can tell at a
    glance whether it's the synthetic dev fixture or the real captured one."""

    meta = coord_lm_head_fixture["meta"]
    assert meta["source"] in ("synthetic", "granite_real")
    assert meta["hidden"] == 1536
    if meta["source"] == "synthetic":
        assert meta["synthetic_seed"] is not None
