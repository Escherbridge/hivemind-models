"""Shared wire-protocol helpers for the Granite expert MoE family of services.

This module is the single source of truth for the binary coordinator <-> expert
wire format used between ``expert_railway_server.py`` (the expert pool) and the
new ``expert_coordinator.py`` (the browser-facing fan-out).

The binary frame layout is:

    | op:u8 | expert_id:u16 | n_tokens:u32 | hidden:u32 | dtype:u8 | payload |

with all numeric fields little-endian and ``payload`` of
``n_tokens * hidden * sizeof(dtype)`` bytes.

The struct definition here is identical to the one historically embedded in
``expert_railway_server.py``; changing it here is a wire-breaking change and
must be coordinated across all three deployed Railway expert services plus the
coordinator. See ``conductor/tracks/_shared/wire-frames.md`` Section 2 for the
canonical contract.

Re-exports are deliberately small and stable so that callers can write::

    from scripts._wire import DTYPE_MAP, _HEADER, parse_expert_ids, pack_tensor, unpack_request

without pulling the full server module into a coordinator process.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Binary header
# ---------------------------------------------------------------------------

# Wire header: op:u8, expert_id:u16, n_tokens:u32, hidden:u32, dtype:u8
# This must stay byte-identical to the deployed expert services.
_HEADER = struct.Struct("<BHIIB")

# Dtype codes used in the binary frame.
DTYPE_MAP = {0: torch.float32, 1: torch.float16}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}

# Op codes
OP_FORWARD = 1
OP_ERROR = 0xFF


def parse_expert_ids(spec: str) -> list[int]:
    """Parse a string like ``"0-15,32-63"`` into the list ``[0..15, 32..63]``.

    Accepts comma-separated ids and/or hyphen ranges. Empty chunks are
    skipped. Whitespace inside a chunk is stripped but the syntax does not
    otherwise tolerate junk -- non-numeric ids raise ``ValueError``.

    Examples
    --------
    >>> parse_expert_ids("0-3")
    [0, 1, 2, 3]
    >>> parse_expert_ids("0,2,4-6")
    [0, 2, 4, 5, 6]
    >>> parse_expert_ids("")
    []
    >>> parse_expert_ids(" 0 - 2 , 5 ")
    [0, 1, 2, 5]
    """

    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo.strip()), int(hi.strip()) + 1))
        else:
            out.append(int(chunk))
    return out


def pack_tensor(expert_id: int, t: torch.Tensor) -> bytes:
    """Pack a 2D ``[n_tokens, hidden]`` tensor into a binary forward frame.

    Tensors with an unsupported dtype (anything other than fp32 or fp16) are
    silently up-cast to fp32 to keep the wire valid. Callers that care about
    fidelity should up-cast deliberately before calling.

    Parameters
    ----------
    expert_id:
        The expert id to address. Must fit in u16.
    t:
        A 2D tensor of shape ``[n_tokens, hidden]``.

    Returns
    -------
    bytes
        The serialized ``[header || payload]`` frame.
    """

    n_tokens, hidden = t.shape
    dtype_code = DTYPE_REVERSE.get(t.dtype, 0)
    if t.dtype not in DTYPE_REVERSE:
        t = t.float()
        dtype_code = 0
    header = _HEADER.pack(OP_FORWARD, expert_id, n_tokens, hidden, dtype_code)
    return header + t.contiguous().numpy().tobytes()


def pack_forward_frame(
    expert_id: int,
    n_tokens: int,
    hidden: int,
    dtype_code: int,
    payload: bytes,
) -> bytes:
    """Pack a forward frame directly from raw payload bytes.

    Used by the coordinator on the hot path so it doesn't need to materialize a
    torch tensor just to re-serialize the bytes it already received from the
    browser.
    """

    if dtype_code not in DTYPE_MAP:
        raise ValueError(f"unknown dtype code {dtype_code}")
    header = _HEADER.pack(OP_FORWARD, expert_id, n_tokens, hidden, dtype_code)
    return header + payload


def unpack_request(buf: bytes) -> Optional[Tuple[int, torch.Tensor]]:
    """Parse a forward request frame into ``(expert_id, tensor)`` or ``None``.

    Returns ``None`` for any frame that is too short, has the wrong op byte,
    or whose payload length doesn't match the declared shape. Callers that
    need a structured error should re-raise as appropriate.
    """

    if len(buf) < _HEADER.size or buf[0] != OP_FORWARD:
        return None
    op, expert_id, n_tokens, hidden, dtype_code = _HEADER.unpack_from(buf, 0)
    dtype = DTYPE_MAP.get(dtype_code, torch.float32)
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    payload_size = n_tokens * hidden * np.dtype(np_dtype).itemsize
    if len(buf) - _HEADER.size != payload_size:
        return None
    arr = np.frombuffer(
        buf,
        dtype=np_dtype,
        count=n_tokens * hidden,
        offset=_HEADER.size,
    ).reshape(n_tokens, hidden).copy()
    return expert_id, torch.from_numpy(arr)


def unpack_response(buf: bytes) -> Tuple[int, torch.Tensor]:
    """Parse a forward response frame, raising ``RuntimeError`` on remote error.

    The coordinator uses this when reading a binary frame back from an upstream
    expert socket. On the error op (``0xFF``) the remaining bytes are decoded
    as UTF-8 (lossy) and raised; on the success op we return ``(expert_id,
    tensor)`` like ``unpack_request``.
    """

    if len(buf) < 1:
        raise RuntimeError("empty response frame")
    op = buf[0]
    if op == OP_ERROR:
        raise RuntimeError(
            f"upstream error: {buf[1:].decode('utf-8', 'replace')}"
        )
    if op != OP_FORWARD:
        raise RuntimeError(f"unexpected op byte 0x{op:02x}")
    parsed = unpack_request(buf)
    if parsed is None:
        raise RuntimeError("malformed forward response frame")
    return parsed


def expected_payload_size(n_tokens: int, hidden: int, dtype_code: int) -> int:
    """Bytes a payload of ``(n_tokens, hidden, dtype)`` is expected to occupy."""

    if dtype_code == 0:
        return n_tokens * hidden * 4
    if dtype_code == 1:
        return n_tokens * hidden * 2
    raise ValueError(f"unknown dtype code {dtype_code}")


__all__ = [
    "DTYPE_MAP",
    "DTYPE_REVERSE",
    "OP_FORWARD",
    "OP_ERROR",
    "_HEADER",
    "parse_expert_ids",
    "pack_tensor",
    "pack_forward_frame",
    "unpack_request",
    "unpack_response",
    "expected_payload_size",
]
