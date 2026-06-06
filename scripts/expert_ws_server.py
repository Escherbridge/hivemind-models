"""
WebSocket server hosting one or more MoE experts.

A single server process can hold N experts (one nn-module each); the
incoming request frame tells it which expert to run. Co-locating multiple
experts saves the per-process Python overhead at the cost of less parallel
compute inside that one process (Python GIL means matmuls run effectively
serially per server). The dispatcher (moe_coordinator.py) still talks to
each expert independently via a shared client-side connection.

Wire protocol (binary frames, little-endian)
============================================

  client -> server:
    [op:u8=1][expert_id:u16][n_tokens:u32][hidden_size:u32][dtype:u8][payload:bytes]
        op=1   : forward
        dtype  : 0=float32, 1=float16
        payload: n_tokens * hidden_size * sizeof(dtype)

  server -> client (success):
    [op:u8=1][expert_id:u16][n_tokens:u32][hidden_size:u32][dtype:u8][payload:bytes]

  server -> client (error):
    [op:u8=0xff][reason:utf8 string remainder]

Hello frame (JSON, sent on connect):
    {"type": "ready",
     "experts": [{"expert_id": 0, "layer": 0, "hidden": 1536, "intermediate": 512},
                 {"expert_id": 1, "layer": 0, "hidden": 1536, "intermediate": 512},
                 ...]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import websockets
from safetensors.torch import load_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("expert_ws")


DTYPE_MAP = {0: torch.float32, 1: torch.float16}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


# Wire protocol --------------------------------------------------------------

# Header layout for a forward request/response:
#   op:u8, expert_id:u16, n_tokens:u32, hidden_size:u32, dtype:u8
_HEADER = struct.Struct("<BHIIB")


def pack_tensor(expert_id: int, t: torch.Tensor) -> bytes:
    n_tokens, hidden = t.shape
    dtype_code = DTYPE_REVERSE.get(t.dtype, 0)
    if t.dtype not in DTYPE_REVERSE:
        t = t.float()
        dtype_code = 0
    header = _HEADER.pack(1, expert_id, n_tokens, hidden, dtype_code)
    return header + t.contiguous().numpy().tobytes()


def unpack_request(buf: bytes) -> tuple[int, torch.Tensor] | None:
    if len(buf) < _HEADER.size or buf[0] != 1:
        return None
    op, expert_id, n_tokens, hidden, dtype_code = _HEADER.unpack_from(buf, 0)
    dtype = DTYPE_MAP.get(dtype_code, torch.float32)
    import numpy as np
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    payload_size = n_tokens * hidden * np_dtype().itemsize
    if len(buf) - _HEADER.size != payload_size:
        return None
    arr = np.frombuffer(buf, dtype=np_dtype, count=n_tokens * hidden, offset=_HEADER.size).reshape(n_tokens, hidden).copy()
    return expert_id, torch.from_numpy(arr)


# Expert -------------------------------------------------------------------


class Expert:
    """One GLU FFN with its own weight tensors."""

    def __init__(self, expert_id: int, layer: int, weights_path: Path) -> None:
        self.expert_id = expert_id
        self.layer = layer
        state = load_file(str(weights_path))
        self.input_linear = state["input_linear.weight"]
        self.output_linear = state["output_linear.weight"]
        self.hidden_size = self.input_linear.shape[1]
        self.intermediate_size = self.output_linear.shape[1]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.to(self.input_linear.dtype)
        h = F.linear(x, self.input_linear)
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up
        return F.linear(h, self.output_linear)

    def info(self) -> dict:
        return {
            "expert_id": self.expert_id,
            "layer": self.layer,
            "hidden": self.hidden_size,
            "intermediate": self.intermediate_size,
        }


# Server -------------------------------------------------------------------


class MultiExpertServer:
    def __init__(self, experts: list[Expert]) -> None:
        self.experts: dict[int, Expert] = {e.expert_id: e for e in experts}
        self.order = sorted(self.experts.keys())
        logger.info(
            "hosting %d experts (ids=%s) for layer %d",
            len(self.experts), self.order,
            experts[0].layer if experts else -1,
        )

    async def handle(self, ws: websockets.ServerConnection) -> None:
        peer = ws.remote_address
        logger.debug("client connected: %s", peer)
        await ws.send(json.dumps({
            "type": "ready",
            "experts": [self.experts[i].info() for i in self.order],
        }))
        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    await self._handle_binary(ws, bytes(raw))
                else:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        await ws.send(json.dumps({"type": "error", "message": "bad frame"}))
                        continue
                    if msg.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                    else:
                        await ws.send(json.dumps({"type": "error", "message": "unknown json type"}))
        except websockets.ConnectionClosed:
            logger.debug("client disconnected: %s", peer)

    async def _handle_binary(self, ws: websockets.ServerConnection, buf: bytes) -> None:
        parsed = unpack_request(buf)
        if parsed is None:
            await ws.send(bytes([0xFF]) + b"bad binary frame")
            return
        expert_id, x = parsed
        if expert_id not in self.experts:
            await ws.send(bytes([0xFF]) + f"expert {expert_id} not on this server".encode())
            return
        e = self.experts[expert_id]
        if x.shape[1] != e.hidden_size:
            await ws.send(bytes([0xFF]) + f"shape mismatch: got hidden={x.shape[1]}, expert expects {e.hidden_size}".encode())
            return
        t0 = time.perf_counter()
        with torch.no_grad():
            out = e.forward(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        await ws.send(pack_tensor(expert_id, out))
        logger.debug("expert %d forward [%dx%d] in %.1fms",
                     expert_id, x.shape[0], x.shape[1], elapsed_ms)


# Argument parsing ---------------------------------------------------------


_FILE_PATTERN = re.compile(r"expert_(\d+)\.safetensors$")


def parse_expert_arg(arg: str) -> tuple[int, Path]:
    """
    Accept either:
      - 'path/to/expert_07.safetensors'  (id inferred from filename)
      - '7:path/to/anything.safetensors' (explicit id)
    """
    if ":" in arg and not arg[1:3] == ":\\":  # be careful of windows drive letters
        idx_str, _, path_str = arg.partition(":")
        return int(idx_str), Path(path_str)
    path = Path(arg)
    m = _FILE_PATTERN.search(path.name)
    if not m:
        raise SystemExit(
            f"cannot infer expert id from {path}; use 'N:{path}' to specify"
        )
    return int(m.group(1)), path


async def main_async(args: argparse.Namespace) -> None:
    experts: list[Expert] = []
    for spec in args.weights:
        eid, p = parse_expert_arg(spec)
        if not p.exists():
            raise SystemExit(f"weights not found: {p}")
        experts.append(Expert(eid, args.layer, p))

    server = MultiExpertServer(experts)
    logger.info("listening on %s:%d", args.host, args.port)
    async with websockets.serve(
        server.handle,
        args.host,
        args.port,
        max_size=2**24,
        ping_interval=None,
    ):
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True,
                        help="One or more expert_<i>.safetensors paths (or N:path)")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
