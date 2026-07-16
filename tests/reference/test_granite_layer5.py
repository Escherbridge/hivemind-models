"""Tests for ``reference/granite_layer5.py`` (Lane D of Wave 3).

What this file pins down
=========================

1. The Python reference forward is **deterministic** for a fixed seed +
   fixed input shape. Two consecutive calls produce bit-identical output.

2. The reference output **matches HuggingFace's
   GraniteMoeHybridDecoderLayer** to fp32 precision. We use the deployed
   layer-group shard in ``output/granite-tiny-q4g64/`` so the math is
   anchored to the same weights the browser will eventually download.

3. The hidden state has the expected shape ``[1, seq_len, 1536]`` and the
   router top-6 has the expected shape ``[1, seq_len, 6]`` for both
   expert ids (int64) and weights (float32, summing to 1 per token).

4. The golden fixture (``reference/golden/layer5.json``) is reproducible
   from the same seed + shape, so the WebGPU-side parity test in
   ``hivemind-client`` can load the same numbers and assert agreement
   to ``1e-3`` (fp16) / ``1e-5`` (fp32) per spec FR-3 acceptance
   criteria.

Frozen input contract (mirrored in WebGPU tests on the client side)
====================================================================

- Seed:  ``42``
- Shape:  ``[1, 4, 1536]`` -- batch=1, seq_len=4 tokens, hidden_dim=1536
- Dtype:  fp32
- Distribution: ``torch.randn`` (per-element iid standard normal)

Why this seed + shape
---------------------

seq_len=4 keeps the fixture tiny (the golden JSON is <250 KB) while still
exercising causal attention masking (a 4x4 mask is non-trivial). Seed=42
is the canonical "scratch" seed in HF's own tests; using it here makes
it easy to remember.

Tolerance (per spec FR-3 acceptance criteria)
=============================================

- Hidden state:  absolute 1e-5 (fp32 reference itself)
- Router top-6 expert ids:  exact match (integer ids)
- Router top-6 weights:  absolute 1e-5 (fp32 reference)
- Recombine math (WGSL parity test consumes this):  1e-3 (fp16) or 1e-5 (fp32)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make ``reference`` importable without an install step. The production
# reference module lives at hivemind-models/reference/granite_layer5.py.
# We must put that directory on sys.path *before* pytest's import machinery
# resolves ``tests.reference`` as a package, otherwise the local empty
# ``tests/reference/__init__.py`` shadows the sibling import.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _REPO_ROOT / "reference"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))

# Importing the file by absolute path avoids the ``tests.reference`` vs
# top-level ``reference`` package shadowing that pytest introduces when a
# tests tree has the same name as a top-level package.
import importlib.util as _ilu  # noqa: E402

_REF_MODULE_PATH = _REFERENCE_DIR / "granite_layer5.py"
_spec = _ilu.spec_from_file_location("reference.granite_layer5", _REF_MODULE_PATH)
assert _spec is not None and _spec.loader is not None
gl5 = _ilu.module_from_spec(_spec)
sys.modules["reference.granite_layer5"] = gl5
_spec.loader.exec_module(gl5)


# ---------------------------------------------------------------------------
# Constants (frozen for the lane, do not edit casually)
# ---------------------------------------------------------------------------

FROZEN_SEED: int = 42
FROZEN_SEQ_LEN: int = 4
FROZEN_HIDDEN: int = 1536
FROZEN_BATCH: int = 1

# Default path to the deployed Granite-tiny layer-group shard. The fixture
# is large (~3 GB) so it lives in ``output/`` rather than in git; tests
# that need the real weights skip if the file is missing. The golden JSON
# itself is checked in.
DEFAULT_SHARD_DIR = _REPO_ROOT / "output" / "granite-tiny-q4g64"


# ---------------------------------------------------------------------------
# Fixtures (Lane D namespace prefix: ``ref_*``)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_frozen_input() -> torch.Tensor:
    """The canonical fp32 input used to generate ``golden/layer5.json``.

    Returning the same tensor across all tests keeps the contract
    explicit: the golden fixture is tied to *this* exact tensor.
    """
    g = torch.Generator(device="cpu").manual_seed(FROZEN_SEED)
    return torch.randn(
        (FROZEN_BATCH, FROZEN_SEQ_LEN, FROZEN_HIDDEN), generator=g, dtype=torch.float32
    )


@pytest.fixture(scope="module")
def ref_real_layer5_weights() -> dict[str, torch.Tensor]:
    """Load layer 5 weights from the deployed shard (if present).

    All keys are stripped of the ``model.layers.5.`` prefix so the dict
    matches what ``granite_layer5.forward_layer5`` expects.
    """
    shard_path = DEFAULT_SHARD_DIR / "shard_layers_0_9.safetensors"
    if not shard_path.exists():
        pytest.skip(f"deployed shard not present at {shard_path}")
    from safetensors.torch import load_file

    sd = load_file(str(shard_path))
    return {
        k.removeprefix("model.layers.5."): v
        for k, v in sd.items()
        if k.startswith("model.layers.5.")
    }


# ---------------------------------------------------------------------------
# Tests: shape + dtype contract
# ---------------------------------------------------------------------------


class TestShapeAndDtype:
    """The forward's output contract is part of the public API."""

    def test_forward_layer5_hidden_shape(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        """Output hidden state is ``[1, seq_len, 1536]`` fp32."""
        out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        assert "hidden" in out, f"missing 'hidden' key; got keys: {sorted(out)}"
        hidden = out["hidden"]
        assert isinstance(hidden, torch.Tensor)
        assert hidden.shape == (FROZEN_BATCH, FROZEN_SEQ_LEN, FROZEN_HIDDEN), (
            f"expected {(FROZEN_BATCH, FROZEN_SEQ_LEN, FROZEN_HIDDEN)}, "
            f"got {tuple(hidden.shape)}"
        )
        assert hidden.dtype == torch.float32, f"expected fp32, got {hidden.dtype}"

    def test_forward_layer5_router_top6_shape(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        """Router top-6 ids and weights each have shape ``[1, seq_len, 6]``."""
        out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        assert "router_top6_expert_ids" in out
        assert "router_top6_weights" in out
        ids = out["router_top6_expert_ids"]
        weights = out["router_top6_weights"]
        assert ids.shape == (FROZEN_BATCH, FROZEN_SEQ_LEN, 6)
        assert weights.shape == (FROZEN_BATCH, FROZEN_SEQ_LEN, 6)
        assert ids.dtype in (torch.int64, torch.long), f"ids dtype: {ids.dtype}"
        assert weights.dtype == torch.float32, f"weights dtype: {weights.dtype}"

    def test_forward_layer5_router_ids_in_range(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        """All router top-6 ids must be in ``[0, 63]`` (Granite-tiny has 64 experts)."""
        out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        ids = out["router_top6_expert_ids"]
        assert int(ids.min()) >= 0
        assert int(ids.max()) <= 63

    def test_forward_layer5_router_weights_softmax(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        """Router top-6 weights sum to 1.0 per token (within fp32 epsilon)."""
        out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        weights = out["router_top6_weights"]
        sums = weights.sum(dim=-1)  # [batch, seq_len]
        torch.testing.assert_close(
            sums, torch.ones_like(sums), atol=1e-5, rtol=0
        )


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Two calls on the same input produce bit-identical output."""

    def test_forward_layer5_deterministic(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        out1 = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        out2 = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        torch.testing.assert_close(out1["hidden"], out2["hidden"], atol=0, rtol=0)
        torch.testing.assert_close(
            out1["router_top6_weights"], out2["router_top6_weights"], atol=0, rtol=0
        )
        assert torch.equal(out1["router_top6_expert_ids"], out2["router_top6_expert_ids"])


# ---------------------------------------------------------------------------
# Tests: ground truth against HuggingFace
# ---------------------------------------------------------------------------


class TestHFMatches:
    """The reference must agree with HF's GraniteMoeHybridDecoderLayer to fp32."""

    def test_hidden_state_matches_hf(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        """Bit-equivalent to the HF decoder layer (max abs diff < 1e-5)."""
        pytest.importorskip(
            "transformers.masking_utils",
            reason="HF-parity reference tests need transformers>=4.52",
        )
        from transformers import AutoConfig
        from transformers.masking_utils import create_causal_mask
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridDecoderLayer,
        )

        config = AutoConfig.from_pretrained("ibm-granite/granite-4.0-h-tiny")
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"

        hf_layer = GraniteMoeHybridDecoderLayer(config, layer_idx=5)
        hf_layer.load_state_dict(ref_real_layer5_weights, strict=False)
        hf_layer.eval()

        cache_position = torch.arange(FROZEN_SEQ_LEN)
        mask = create_causal_mask(
            config=config,
            inputs_embeds=ref_frozen_input,
            attention_mask=None,
            past_key_values=None,
            cache_position=cache_position,
        )

        with torch.no_grad():
            hf_out = hf_layer(
                ref_frozen_input,
                attention_mask=mask,
                position_embeddings=None,
            )
        # HF returns a tensor (not a tuple) in this transformers version.
        if isinstance(hf_out, tuple):
            hf_hidden = hf_out[0]
        else:
            hf_hidden = hf_out

        ref_out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        max_abs = (ref_out["hidden"] - hf_hidden).abs().max().item()
        assert max_abs < 1e-5, f"max abs diff {max_abs} exceeds 1e-5"

    def test_router_matches_hf(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        """The router top-6 expert ids agree exactly with HF (softmax tie-break
        may differ for tied logits, so we check the set of top-6 is in the
        reference's top-K rather than asserting identical ordering)."""
        pytest.importorskip(
            "transformers.masking_utils",
            reason="HF-parity reference tests need transformers>=4.52",
        )
        from transformers import AutoConfig
        from transformers.masking_utils import create_causal_mask
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridDecoderLayer,
        )
        import torch.nn.functional as F

        config = AutoConfig.from_pretrained("ibm-granite/granite-4.0-h-tiny")
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"

        hf_layer = GraniteMoeHybridDecoderLayer(config, layer_idx=5)
        hf_layer.load_state_dict(ref_real_layer5_weights, strict=False)
        hf_layer.eval()

        cache_position = torch.arange(FROZEN_SEQ_LEN)
        mask = create_causal_mask(
            config=config,
            inputs_embeds=ref_frozen_input,
            attention_mask=None,
            past_key_values=None,
            cache_position=cache_position,
        )

        with torch.no_grad():
            # Recompute the post-layernorm hidden that feeds the router.
            residual = ref_frozen_input
            x = hf_layer.input_layernorm(ref_frozen_input)
            attn_out, _ = hf_layer.self_attn(x, attention_mask=mask, position_embeddings=None)
            x = residual + attn_out * hf_layer.residual_multiplier
            residual = x
            x = hf_layer.post_attention_layernorm(x)
            router_logits = hf_layer.block_sparse_moe.router.layer(x).float()
            # Full top-6 + softmax
            top6_logits, hf_top6_idx = router_logits.topk(6, dim=-1)
            hf_top6_w = F.softmax(top6_logits, dim=-1).type_as(x)

        ref_out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        ref_ids = ref_out["router_top6_expert_ids"]  # [1, S, 6]
        ref_w = ref_out["router_top6_weights"]

        # Tie-breaks for exactly-tied logits can vary across implementations;
        # the contract is that the *set* of top-6 per token is correct, not
        # the relative ordering within ties. We compare the sorted sets.
        for s in range(FROZEN_SEQ_LEN):
            ref_set = sorted(int(i) for i in ref_ids[0, s, :].tolist())
            hf_set = sorted(int(i) for i in hf_top6_idx[0, s, :].tolist())
            assert ref_set == hf_set, (
                f"token {s} top-6 sets disagree: ref={ref_set} hf={hf_set}"
            )
            # And the weight associated with each id matches in the softmax.
            for eid in ref_set:
                ref_w_for_eid = ref_w[0, s, (ref_ids[0, s, :] == eid).nonzero(as_tuple=True)[0][0]]
                hf_w_for_eid = hf_top6_w[0, s, (hf_top6_idx[0, s, :] == eid).nonzero(as_tuple=True)[0][0]]
                assert abs(float(ref_w_for_eid) - float(hf_w_for_eid)) < 1e-5, (
                    f"token {s} expert {eid} weight disagrees: "
                    f"ref={float(ref_w_for_eid)} hf={float(hf_w_for_eid)}"
                )


# ---------------------------------------------------------------------------
# Tests: golden fixture reproducibility
# ---------------------------------------------------------------------------


class TestGoldenFixture:
    """The golden JSON must match what the forward produces right now."""

    GOLDEN_PATH = _REFERENCE_DIR / "golden" / "layer5.json"

    def test_golden_json_exists(self) -> None:
        assert self.GOLDEN_PATH.exists(), (
            f"missing {self.GOLDEN_PATH}. Run the generation step (see "
            f"reference/granite_layer5.py --write-golden)."
        )

    def test_golden_matches_forward(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        if not self.GOLDEN_PATH.exists():
            pytest.skip("golden JSON not generated yet")
        with self.GOLDEN_PATH.open() as f:
            golden = json.load(f)

        # 1) The frozen input is the exact same tensor used to generate
        #    the golden (sha256 pinned at generation time).
        frozen_bytes = ref_frozen_input.cpu().numpy().tobytes()
        frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
        assert frozen_sha == golden["input_sha256"], (
            f"input sha mismatch: ref_input={frozen_sha[:16]}... "
            f"golden={golden['input_sha256'][:16]}... -- did the input "
            f"contract change without regenerating the golden?"
        )

        # 2) The forward run with the current weights produces the exact
        #    numbers in the golden (within fp32 epsilon; the golden is
        #    written to fp32, not fp16, because the reference itself is fp32).
        ref_out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        golden_hidden = np.asarray(golden["expected_hidden"], dtype=np.float32)
        assert (
            golden_hidden.shape == ref_out["hidden"].shape
        ), f"hidden shape {golden_hidden.shape} != {ref_out['hidden'].shape}"
        np.testing.assert_allclose(
            ref_out["hidden"].cpu().numpy(),
            golden_hidden,
            atol=1e-5,
            rtol=0,
        )

        golden_ids = np.asarray(golden["expected_router_top6_expert_ids"], dtype=np.int64)
        np.testing.assert_array_equal(
            ref_out["router_top6_expert_ids"].cpu().numpy(),
            golden_ids,
        )

        golden_weights = np.asarray(golden["expected_router_top6_weights"], dtype=np.float32)
        np.testing.assert_allclose(
            ref_out["router_top6_weights"].cpu().numpy(),
            golden_weights,
            atol=1e-5,
            rtol=0,
        )


class TestRouterOnlyFixture:
    """A small router-only fixture is also written for fast client-side tests."""

    ROUTER_GOLDEN_PATH = _REFERENCE_DIR / "golden" / "router_top6.json"

    def test_router_golden_json_exists(self) -> None:
        assert self.ROUTER_GOLDEN_PATH.exists(), (
            f"missing {self.ROUTER_GOLDEN_PATH}. Run the generation step."
        )

    def test_router_golden_matches_forward(
        self,
        ref_frozen_input: torch.Tensor,
        ref_real_layer5_weights: dict[str, torch.Tensor],
    ) -> None:
        if not self.ROUTER_GOLDEN_PATH.exists():
            pytest.skip("router golden JSON not generated yet")
        with self.ROUTER_GOLDEN_PATH.open() as f:
            golden = json.load(f)

        ref_out = gl5.forward_layer5(ref_frozen_input, ref_real_layer5_weights)
        golden_ids = np.asarray(golden["expected_router_top6_expert_ids"], dtype=np.int64)
        np.testing.assert_array_equal(
            ref_out["router_top6_expert_ids"].cpu().numpy(), golden_ids
        )
        golden_weights = np.asarray(golden["expected_router_top6_weights"], dtype=np.float32)
        np.testing.assert_allclose(
            ref_out["router_top6_weights"].cpu().numpy(),
            golden_weights,
            atol=1e-5,
            rtol=0,
        )
