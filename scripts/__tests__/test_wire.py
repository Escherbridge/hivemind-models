"""Tests for ``scripts/_wire.py``: round-trip the binary protocol.

The contract under test is the same struct definition shipped by the deployed
expert servers. Breaking these tests means breaking three live services, so
the bar is "byte-identical".
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from scripts._wire import (
    DTYPE_MAP,
    DTYPE_REVERSE,
    OP_ERROR,
    OP_FORWARD,
    _HEADER,
    expected_payload_size,
    pack_forward_frame,
    pack_tensor,
    parse_expert_ids,
    unpack_request,
    unpack_response,
)


# ---------------------------------------------------------------------------
# Header layout
# ---------------------------------------------------------------------------


def test_header_layout_matches_canon() -> None:
    """``<BHIIB`` is wire canon. Any change here breaks deployed services."""

    assert _HEADER.format == "<BHIIB"
    # Wire size: op(1) + expert_id(2) + n_tokens(4) + hidden(4) + dtype(1) = 12.
    assert _HEADER.size == 12


def test_dtype_codes_canonical() -> None:
    assert DTYPE_MAP == {0: torch.float32, 1: torch.float16}
    assert DTYPE_REVERSE[torch.float32] == 0
    assert DTYPE_REVERSE[torch.float16] == 1


def test_op_codes() -> None:
    assert OP_FORWARD == 1
    assert OP_ERROR == 0xFF


# ---------------------------------------------------------------------------
# parse_expert_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("0-3", [0, 1, 2, 3]),
        ("0,2,4", [0, 2, 4]),
        ("0-15,32-63", list(range(0, 16)) + list(range(32, 64))),
        ("", []),
        ("5", [5]),
        (" 0 - 2 , 5 ", [0, 1, 2, 5]),
        (" , 0-1 , , 3 , ", [0, 1, 3]),
    ],
)
def test_parse_expert_ids_table(spec: str, expected: list[int]) -> None:
    assert parse_expert_ids(spec) == expected


def test_parse_expert_ids_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        parse_expert_ids("abc")


# ---------------------------------------------------------------------------
# pack/unpack round-trip
# ---------------------------------------------------------------------------


def test_round_trip_fp16_is_bit_identical() -> None:
    """Phase 1 demo runs at fp16. Bytes must round-trip exact."""

    torch.manual_seed(0)
    t = torch.randn(8, 1536, dtype=torch.float16)
    frame = pack_tensor(expert_id=7, t=t)

    parsed = unpack_request(frame)
    assert parsed is not None
    expert_id, out = parsed
    assert expert_id == 7
    assert out.dtype == torch.float16
    assert out.shape == (8, 1536)
    assert torch.equal(out, t)


def test_round_trip_fp32_preserves_payload() -> None:
    torch.manual_seed(1)
    t = torch.randn(4, 1536, dtype=torch.float32)
    frame = pack_tensor(expert_id=42, t=t)

    parsed = unpack_request(frame)
    assert parsed is not None
    expert_id, out = parsed
    assert expert_id == 42
    assert out.dtype == torch.float32
    assert out.shape == (4, 1536)
    assert torch.equal(out, t)


def test_unsupported_dtype_is_upcast_to_fp32() -> None:
    """bfloat16 is not in DTYPE_MAP. It must up-cast to fp32 silently."""

    t = torch.zeros(2, 4, dtype=torch.bfloat16)
    frame = pack_tensor(expert_id=0, t=t)
    # Header should advertise fp32 (dtype_code 0).
    _, _, _, _, dtype_code = _HEADER.unpack_from(frame, 0)
    assert dtype_code == 0


def test_unpack_request_rejects_short_frame() -> None:
    assert unpack_request(b"") is None
    assert unpack_request(b"\x01" + b"\x00" * 5) is None


def test_unpack_request_rejects_wrong_op() -> None:
    bad = _HEADER.pack(2, 0, 1, 1536, 1) + b"\x00\x00"
    assert unpack_request(bad) is None


def test_unpack_request_rejects_payload_size_mismatch() -> None:
    # Declare 8 tokens but only ship 1 token's worth of payload.
    header = _HEADER.pack(1, 0, 8, 1536, 1)
    body = (np.zeros(1 * 1536, dtype=np.float16).tobytes())
    assert unpack_request(header + body) is None


def test_pack_forward_frame_matches_pack_tensor() -> None:
    """The raw-bytes packer must produce frames byte-identical to pack_tensor."""

    torch.manual_seed(2)
    t = torch.randn(8, 1536, dtype=torch.float16)
    via_tensor = pack_tensor(expert_id=11, t=t)
    via_bytes = pack_forward_frame(
        expert_id=11,
        n_tokens=8,
        hidden=1536,
        dtype_code=1,
        payload=t.contiguous().numpy().tobytes(),
    )
    assert via_tensor == via_bytes


def test_pack_forward_frame_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError):
        pack_forward_frame(0, 1, 1, dtype_code=7, payload=b"")


# ---------------------------------------------------------------------------
# unpack_response error path
# ---------------------------------------------------------------------------


def test_unpack_response_raises_on_error_op() -> None:
    err_frame = bytes([OP_ERROR]) + b"expert offline"
    with pytest.raises(RuntimeError, match="expert offline"):
        unpack_response(err_frame)


def test_unpack_response_raises_on_unknown_op() -> None:
    bad = bytes([0x42])
    with pytest.raises(RuntimeError, match="unexpected op byte"):
        unpack_response(bad)


def test_unpack_response_raises_on_empty_frame() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        unpack_response(b"")


def test_unpack_response_round_trip_success() -> None:
    t = torch.randn(1, 1536, dtype=torch.float16)
    frame = pack_tensor(expert_id=3, t=t)
    expert_id, out = unpack_response(frame)
    assert expert_id == 3
    assert torch.equal(out, t)


def test_unpack_response_raises_on_malformed_forward() -> None:
    """A forward op byte with too-short payload must surface as RuntimeError."""

    bad = _HEADER.pack(1, 0, 8, 1536, 1)  # declares payload but ships none
    with pytest.raises(RuntimeError, match="malformed"):
        unpack_response(bad)


# ---------------------------------------------------------------------------
# expected_payload_size
# ---------------------------------------------------------------------------


def test_expected_payload_size_fp16() -> None:
    assert expected_payload_size(8, 1536, 1) == 8 * 1536 * 2


def test_expected_payload_size_fp32() -> None:
    assert expected_payload_size(8, 1536, 0) == 8 * 1536 * 4


def test_expected_payload_size_unknown_raises() -> None:
    with pytest.raises(ValueError):
        expected_payload_size(1, 1, 9)


# ---------------------------------------------------------------------------
# Cross-check against the deployed expert server's own struct
# ---------------------------------------------------------------------------


def test_struct_size_arithmetic() -> None:
    """Sanity: 1 + 2 + 4 + 4 + 1 = 12. Repeated here so a typo screams loud."""

    assert struct.calcsize("<BHIIB") == 12
