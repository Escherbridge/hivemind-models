"""Tests for the JSON dispatch / lm_head frame validators.

Each error code from the closed set in wire-frames.md §1.9 that maps to a
validation failure is exercised. The validators do not touch network; they
operate on dicts.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from scripts.expert_coordinator import (
    ERROR_CODES,
    FrameError,
    HIDDEN_DIM,
    validate_dispatch_frame,
    validate_lm_head_frame,
)


# ---------------------------------------------------------------------------
# ERROR_CODES is canonical and stable
# ---------------------------------------------------------------------------


def test_error_codes_is_closed_set() -> None:
    expected = {
        # Wave-3:
        "expected_hello",
        "bad_json",
        "frame_too_large",
        "unknown_type",
        "bad_mode",
        "bad_expert_id",
        "bad_shape",
        "payload_size_mismatch",
        "lm_head_not_ready",
        "upstream_unavailable",
        "all_regions_failed",
        "internal_error",
        # Wave-4 desktop-peer (wire-frames.md §1.9):
        "dispatch_moved",
        "peer_offline",
        "unknown_peer",
        "peer_id_collision",
    }
    assert ERROR_CODES == expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_call(
    expert_id: int = 0,
    n_tokens: int = 1,
    dtype: int = 1,
) -> dict:
    nbytes = n_tokens * HIDDEN_DIM * (2 if dtype == 1 else 4)
    payload = b"\x00" * nbytes
    return {
        "expert_id": expert_id,
        "n_tokens": n_tokens,
        "hidden": HIDDEN_DIM,
        "dtype": dtype,
        "payload_b64": base64.b64encode(payload).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_dispatch_happy_path() -> None:
    msg = {
        "type": "dispatch",
        "request_id": "abc-123",
        "mode": "wait_all",
        "k": 2,
        "calls": [_valid_call(0), _valid_call(7, n_tokens=4)],
    }
    out = validate_dispatch_frame(msg)
    assert out["request_id"] == "abc-123"
    assert out["mode"] == "wait_all"
    assert out["k"] == 2
    assert len(out["calls"]) == 2
    assert out["calls"][0]["expert_id"] == 0
    assert len(out["calls"][0]["payload_bytes"]) == HIDDEN_DIM * 2
    assert len(out["calls"][1]["payload_bytes"]) == 4 * HIDDEN_DIM * 2


def test_dispatch_first_of_k_mode_accepted() -> None:
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "first_of_k",
        "k": 1, "calls": [_valid_call(0)],
    }
    out = validate_dispatch_frame(msg)
    assert out["mode"] == "first_of_k"


def test_dispatch_bad_mode_raises_bad_mode() -> None:
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "nope",
        "k": 1, "calls": [_valid_call(0)],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_mode"
    assert ei.value.request_id == "x"


def test_dispatch_bad_expert_id_low_raises() -> None:
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [_valid_call(-1)],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_expert_id"


def test_dispatch_bad_expert_id_high_raises() -> None:
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [_valid_call(64)],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_expert_id"


def test_dispatch_bad_hidden_raises_bad_shape() -> None:
    call = _valid_call(0)
    call["hidden"] = 999
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [call],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_shape"


@pytest.mark.parametrize("n_tokens", [0, -1, 513])
def test_dispatch_bad_n_tokens_raises_bad_shape(n_tokens: int) -> None:
    call = _valid_call(0)
    call["n_tokens"] = n_tokens
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [call],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_shape"


@pytest.mark.parametrize("dtype", [2, -1, "fp16"])
def test_dispatch_bad_dtype_raises_bad_shape(dtype) -> None:
    call = _valid_call(0)
    call["dtype"] = dtype
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [call],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_shape"


def test_dispatch_payload_size_mismatch_raises() -> None:
    # Declare 8 tokens but ship 1 token's worth of payload.
    payload = b"\x00" * (1 * HIDDEN_DIM * 2)
    call = {
        "expert_id": 0,
        "n_tokens": 8,
        "hidden": HIDDEN_DIM,
        "dtype": 1,
        "payload_b64": base64.b64encode(payload).decode("ascii"),
    }
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [call],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "payload_size_mismatch"


def test_dispatch_k_mismatch_raises_bad_shape() -> None:
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 2, "calls": [_valid_call(0)],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_shape"


def test_dispatch_k_out_of_range_raises_bad_shape() -> None:
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 33, "calls": [_valid_call(0)] * 33,
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_shape"


def test_dispatch_non_list_calls_raises_bad_shape() -> None:
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": "not a list",
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_shape"


def test_dispatch_non_string_payload_raises_bad_shape() -> None:
    call = _valid_call(0)
    call["payload_b64"] = 1234
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [call],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    assert ei.value.code == "bad_shape"


def test_dispatch_non_base64_payload_surfaces_as_size_mismatch() -> None:
    call = _valid_call(0)
    call["payload_b64"] = "not!valid!base64!"
    msg = {
        "type": "dispatch", "request_id": "x", "mode": "wait_all",
        "k": 1, "calls": [call],
    }
    with pytest.raises(FrameError) as ei:
        validate_dispatch_frame(msg)
    # Either the base64 decode fails (-> payload_size_mismatch) or the
    # decoded size != expected. Both map to the same closed-set code.
    assert ei.value.code == "payload_size_mismatch"


# ---------------------------------------------------------------------------
# lm_head
# ---------------------------------------------------------------------------


def test_lm_head_happy_path() -> None:
    payload = np.zeros(HIDDEN_DIM, dtype=np.float16).tobytes()
    msg = {
        "type": "lm_head",
        "request_id": "rid",
        "hidden_b64": base64.b64encode(payload).decode("ascii"),
        "shape": [1, HIDDEN_DIM],
        "dtype": "fp16",
    }
    out = validate_lm_head_frame(msg)
    assert out["shape"] == (1, HIDDEN_DIM)
    assert out["dtype"] == "fp16"
    assert out["payload_bytes"] == payload


def test_lm_head_bad_hidden_dim_raises_bad_shape() -> None:
    payload = np.zeros(8, dtype=np.float16).tobytes()
    msg = {
        "type": "lm_head", "request_id": "rid",
        "hidden_b64": base64.b64encode(payload).decode("ascii"),
        "shape": [1, 8], "dtype": "fp16",
    }
    with pytest.raises(FrameError) as ei:
        validate_lm_head_frame(msg)
    assert ei.value.code == "bad_shape"


def test_lm_head_bad_dtype_raises_bad_shape() -> None:
    payload = np.zeros(HIDDEN_DIM, dtype=np.float32).tobytes()
    msg = {
        "type": "lm_head", "request_id": "rid",
        "hidden_b64": base64.b64encode(payload).decode("ascii"),
        "shape": [1, HIDDEN_DIM], "dtype": "bf16",
    }
    with pytest.raises(FrameError) as ei:
        validate_lm_head_frame(msg)
    assert ei.value.code == "bad_shape"


def test_lm_head_payload_size_mismatch_raises() -> None:
    payload = b"\x00" * 16  # way too small
    msg = {
        "type": "lm_head", "request_id": "rid",
        "hidden_b64": base64.b64encode(payload).decode("ascii"),
        "shape": [1, HIDDEN_DIM], "dtype": "fp16",
    }
    with pytest.raises(FrameError) as ei:
        validate_lm_head_frame(msg)
    assert ei.value.code == "payload_size_mismatch"


def test_lm_head_missing_payload_raises_bad_shape() -> None:
    msg = {
        "type": "lm_head", "request_id": "rid",
        "shape": [1, HIDDEN_DIM], "dtype": "fp16",
    }
    with pytest.raises(FrameError) as ei:
        validate_lm_head_frame(msg)
    assert ei.value.code == "bad_shape"


def test_lm_head_seq_too_big_raises_bad_shape() -> None:
    msg = {
        "type": "lm_head", "request_id": "rid",
        "hidden_b64": "",
        "shape": [513, HIDDEN_DIM], "dtype": "fp16",
    }
    with pytest.raises(FrameError) as ei:
        validate_lm_head_frame(msg)
    assert ei.value.code == "bad_shape"
