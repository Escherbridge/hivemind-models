"""
Wire-protocol smoke test for expert_ws_server.

Sends a fixed-seed random input to a single expert over WebSocket and compares
the result to an in-process forward through the same expert weights.

PASS means the binary frame layout, dtype handling, and matmul all work over
the wire. FAIL points at exactly which dimension is wrong.

Usage:
    python scripts/test_expert_wire.py \
        --weights output/granite-tiny-q4g64/experts_layer_0/expert_00.safetensors \
        --uri ws://127.0.0.1:9800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import websockets
from safetensors.torch import load_file


DTYPE_MAP = {0: torch.float32, 1: torch.float16}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}
_HEADER = struct.Struct("<BHIIB")  # op, expert_id, n_tokens, hidden, dtype


def pack_request(expert_id: int, t: torch.Tensor) -> bytes:
    n_tokens, hidden = t.shape
    dtype_code = DTYPE_REVERSE.get(t.dtype, 0)
    if t.dtype not in DTYPE_REVERSE:
        t = t.float()
        dtype_code = 0
    header = _HEADER.pack(1, expert_id, n_tokens, hidden, dtype_code)
    return header + t.contiguous().numpy().tobytes()


def unpack_response(buf: bytes) -> tuple[int, torch.Tensor]:
    op = buf[0]
    if op == 0xFF:
        raise RuntimeError(f"server error: {buf[1:].decode('utf-8', errors='replace')}")
    if op != 1:
        raise RuntimeError(f"unknown op {op}")
    _, expert_id, n_tokens, hidden, dtype_code = _HEADER.unpack_from(buf, 0)
    dtype = DTYPE_MAP[dtype_code]
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    arr = np.frombuffer(buf, dtype=np_dtype, count=n_tokens * hidden,
                        offset=_HEADER.size).reshape(n_tokens, hidden).copy()
    return expert_id, torch.from_numpy(arr)


def in_process_forward(weights: dict, tokens: torch.Tensor) -> torch.Tensor:
    w_in = weights["input_linear.weight"]
    w_out = weights["output_linear.weight"]
    x = tokens.to(w_in.dtype)
    h = F.linear(x, w_in)
    gate, up = h.chunk(2, dim=-1)
    h = F.silu(gate) * up
    return F.linear(h, w_out)


async def run(uri: str, weights_path: Path, expert_id: int) -> int:
    state = load_file(str(weights_path))
    hidden_size = state["input_linear.weight"].shape[1]
    print(f"loaded weights: hidden_size={hidden_size}, expert_id={expert_id}")

    # Synthetic input: 3 tokens, fixed seed for reproducibility
    torch.manual_seed(0)
    x = torch.randn(3, hidden_size, dtype=torch.float16)

    # In-process reference
    ref = in_process_forward(state, x).float()
    print(f"in-process output: shape={list(ref.shape)} norm={ref.norm():.3f}")

    # Over the wire
    print(f"\nconnecting to {uri} ...")
    async with websockets.connect(uri, max_size=2**24, open_timeout=5) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        print(f"  hello: {hello}")
        if hello.get("type") != "ready":
            print(f"  FAIL: unexpected hello")
            return 2

        t0 = time.perf_counter()
        await ws.send(pack_request(expert_id, x))
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        got_eid, out = unpack_response(raw)
        out = out.float()
        print(f"  ws response: expert_id={got_eid} shape={list(out.shape)} norm={out.norm():.3f} "
              f"round-trip={elapsed_ms:.1f}ms")

        # Compare
        diff = (ref - out).abs()
        cos = F.cosine_similarity(ref.flatten().unsqueeze(0),
                                  out.flatten().unsqueeze(0)).item()
        print(f"\nmax_abs_diff={diff.max():.5f}  mean_abs_diff={diff.mean():.5f}  cos_sim={cos:.6f}")

        if cos > 0.9999 and diff.max() < 1e-2:
            print("PASS: wire result matches in-process within fp16 noise")
            return 0
        print("FAIL: wire result diverges from in-process")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="ws://127.0.0.1:9800")
    parser.add_argument("--weights", type=Path,
                        default=Path("output/granite-tiny-q4g64/experts_layer_0/expert_00.safetensors"))
    parser.add_argument("--expert-id", type=int, default=0,
                        help="Which expert hosted at this URI to call (must match weights)")
    args = parser.parse_args()
    code = asyncio.run(run(args.uri, args.weights, args.expert_id))
    sys.exit(code)


if __name__ == "__main__":
    main()
