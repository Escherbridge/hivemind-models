"""
Railway-friendly expert server.

Combines the existing expert WebSocket protocol from expert_ws_server.py with
an HTTP listener on the SAME port for Railway healthchecks. Railway exposes a
single public port per service, and the platform's HTTP healthcheck must hit
that port before the service is marked live.

Environment variables:
    PORT            -- port to bind (Railway sets this)
    EXPERT_IDS      -- comma-separated list of expert ids to host, e.g. "0,1,2,3"
                       or a range "0-31". Loads matching expert_NN.safetensors
                       files from EXPERTS_DIR.
    EXPERTS_DIR     -- directory containing expert_NN.safetensors (default
                       /app/experts_layer_0)
    LAYER           -- layer index these experts belong to (default 0)
    REGION          -- optional human-readable region tag (e.g. "us-west2"),
                       returned by /health and /info for measurement scripts.

Wire protocol (binary) is identical to expert_ws_server.py:
    [op:u8=1][expert_id:u16][n_tokens:u32][hidden_size:u32][dtype:u8][payload]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import websockets
from safetensors.torch import load_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("expert_railway")


DTYPE_MAP = {0: torch.float32, 1: torch.float16}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}
_HEADER = struct.Struct("<BHIIB")


def parse_expert_ids(spec: str) -> list[int]:
    """Accept comma-separated ids and/or hyphen ranges, e.g. '0-15,32-63'."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


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
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    payload_size = n_tokens * hidden * np.dtype(np_dtype).itemsize
    if len(buf) - _HEADER.size != payload_size:
        return None
    arr = np.frombuffer(buf, dtype=np_dtype, count=n_tokens * hidden,
                        offset=_HEADER.size).reshape(n_tokens, hidden).copy()
    return expert_id, torch.from_numpy(arr)


class Expert:
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


class ExpertHost:
    def __init__(self, experts: list[Expert], region: str, layer: int) -> None:
        self.experts: dict[int, Expert] = {e.expert_id: e for e in experts}
        self.order = sorted(self.experts.keys())
        self.region = region
        self.layer = layer
        self.started_at = time.time()
        self.forward_count = 0

    def info_dict(self) -> dict:
        return {
            "region": self.region,
            "layer": self.layer,
            "expert_ids": self.order,
            "experts": [self.experts[i].info() for i in self.order],
            "uptime_s": time.time() - self.started_at,
            "forward_count": self.forward_count,
        }

    async def handle_ws(self, ws: websockets.ServerConnection) -> None:
        peer = ws.remote_address
        logger.info("ws client connected: %s", peer)
        await ws.send(json.dumps({
            "type": "ready",
            "region": self.region,
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
                        await ws.send(json.dumps({"type": "error",
                                                   "message": "bad frame"}))
                        continue
                    mtype = msg.get("type")
                    if mtype == "ping":
                        # Echo server-side timestamp so clients can also use
                        # JSON pings for latency probing.
                        await ws.send(json.dumps({
                            "type": "pong",
                            "server_ts": time.time(),
                            "region": self.region,
                            "echo": msg.get("nonce"),
                        }))
                    else:
                        await ws.send(json.dumps({"type": "error",
                                                   "message": "unknown json type"}))
        except websockets.ConnectionClosed:
            logger.info("ws client disconnected: %s", peer)

    async def _handle_binary(self, ws, buf: bytes) -> None:
        parsed = unpack_request(buf)
        if parsed is None:
            await ws.send(bytes([0xFF]) + b"bad binary frame")
            return
        expert_id, x = parsed
        if expert_id not in self.experts:
            await ws.send(bytes([0xFF])
                          + f"expert {expert_id} not on this region {self.region}".encode())
            return
        e = self.experts[expert_id]
        if x.shape[1] != e.hidden_size:
            await ws.send(bytes([0xFF])
                          + f"shape mismatch: got hidden={x.shape[1]}, expert expects {e.hidden_size}".encode())
            return
        with torch.no_grad():
            out = e.forward(x)
        self.forward_count += 1
        await ws.send(pack_tensor(expert_id, out))


def _ensure_s3_experts(experts_dir: Path, ids: list[int]) -> None:
    """If any expert weight file is missing on the local volume, download it
    from the configured S3 bucket. Cached on the volume, so the next boot is
    instant.
    """
    bucket = os.environ.get("EXPERTS_S3_BUCKET")
    if not bucket:
        return  # purely local mode (e.g. Dockerfile bakes weights)

    endpoint = os.environ.get("EXPERTS_S3_ENDPOINT", "https://t3.storageapi.dev")
    access_key = os.environ.get("EXPERTS_S3_ACCESS_KEY")
    secret_key = os.environ.get("EXPERTS_S3_SECRET_KEY")
    prefix = os.environ.get("EXPERTS_S3_PREFIX", "layer_0/")
    region = os.environ.get("EXPERTS_S3_REGION", "auto")
    if not (access_key and secret_key):
        raise SystemExit("EXPERTS_S3_BUCKET set but EXPERTS_S3_ACCESS_KEY/"
                         "EXPERTS_S3_SECRET_KEY missing")

    experts_dir.mkdir(parents=True, exist_ok=True)

    missing = [eid for eid in ids
               if not (experts_dir / f"expert_{eid:02d}.safetensors").exists()]
    if not missing:
        logger.info("all %d experts already cached at %s", len(ids), experts_dir)
        return

    import boto3
    from botocore.config import Config

    logger.info("fetching %d experts from s3://%s/%s (cache miss)",
                len(missing), bucket, prefix)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(s3={"addressing_style": "virtual"},
                      retries={"max_attempts": 5}),
    )

    t0 = time.time()
    for eid in missing:
        fname = f"expert_{eid:02d}.safetensors"
        dest = experts_dir / fname
        tmp = dest.with_suffix(".safetensors.partial")
        s3.download_file(bucket, prefix + fname, str(tmp))
        tmp.rename(dest)
    elapsed = time.time() - t0
    logger.info("downloaded %d experts in %.1fs", len(missing), elapsed)


def load_experts(experts_dir: Path, ids: list[int], layer: int) -> list[Expert]:
    _ensure_s3_experts(experts_dir, ids)
    out: list[Expert] = []
    for eid in ids:
        p = experts_dir / f"expert_{eid:02d}.safetensors"
        if not p.exists():
            raise SystemExit(f"weights not found: {p}")
        out.append(Expert(eid, layer, p))
        logger.info("loaded expert %d from %s", eid, p.name)
    return out


def _json_response(status: int, payload: dict):
    """Legacy websockets process_request return shape:
        (HTTPStatus_or_int, list[tuple[str,str]] headers, bytes body)
    Returning None means "proceed with WebSocket handshake".
    """
    import http
    body = json.dumps(payload).encode()
    return (
        http.HTTPStatus(status),
        [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            ("Connection", "close"),
        ],
        body,
    )


async def run(port: int, experts_dir: Path, expert_ids: list[int],
              layer: int, region: str) -> None:
    experts = load_experts(experts_dir, expert_ids, layer)
    host_state = ExpertHost(experts, region=region, layer=layer)

    async def handler(ws) -> None:
        await host_state.handle_ws(ws)

    async def http_fallback(path: str, request_headers):
        # Legacy `websockets` signature: (path, headers) -> Optional[response].
        # request_headers is a websockets Headers/Multimap; `.get(name)` works.
        upgrade_hdr = request_headers.get("Upgrade", "") or ""
        if "websocket" in upgrade_hdr.lower():
            return None  # proceed with WS handshake
        clean_path = path.split("?", 1)[0]
        if clean_path in ("/", "/health", "/ready"):
            return _json_response(200, {
                "status": "ok",
                "region": host_state.region,
                "layer": host_state.layer,
                "n_experts": len(host_state.experts),
                "uptime_s": time.time() - host_state.started_at,
            })
        if clean_path == "/info":
            return _json_response(200, host_state.info_dict())
        return _json_response(404, {"error": "not found"})

    logger.info("expert host: region=%s layer=%d experts=%s port=%d",
                region, layer, host_state.order, port)

    async with websockets.serve(
        handler,
        host="0.0.0.0",
        port=port,
        max_size=2**24,
        ping_interval=20,    # keep connection warm; Railway proxy idles at 60s
        ping_timeout=20,
        process_request=http_fallback,
    ):
        logger.info("listening on 0.0.0.0:%d", port)
        await asyncio.Future()


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    experts_dir = Path(os.environ.get("EXPERTS_DIR", "/app/experts_layer_0"))
    expert_ids_spec = os.environ.get("EXPERT_IDS")
    layer = int(os.environ.get("LAYER", "0"))
    region = os.environ.get("REGION") or os.environ.get("RAILWAY_REGION") or "unknown"

    if not expert_ids_spec:
        raise SystemExit("EXPERT_IDS env var is required (e.g. '0-31' or '0,1,2')")

    expert_ids = parse_expert_ids(expert_ids_spec)
    asyncio.run(run(port, experts_dir, expert_ids, layer, region))


if __name__ == "__main__":
    main()
