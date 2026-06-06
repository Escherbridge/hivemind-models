"""
Shard Server — loads a safetensors shard and serves layer forward passes over HTTP.

Usage:
    python scripts/shard_server.py \
        --shard-dir ./output/tinyllama-1b-q4 \
        --layers 4 10 \
        --port 9001 \
        --relay http://localhost:8787

This loads shard_layers_4_10.safetensors + the model config, reconstructs
layers 4-10 as LlamaDecoderLayer modules, and exposes:

    POST /forward   — accept hidden_states tensor, return transformed tensor
    GET  /health    — liveness + shard metadata
    POST /register  — re-announce to relay
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import time
import uuid
from pathlib import Path

# Load .env if present (for HF_TOKEN etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from safetensors.torch import load_file
from transformers import AutoConfig

# Make src/ importable so we can pull in the architecture handlers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.architectures import (  # noqa: E402  (after path manipulation)
    LayerShardSpec,
    available_handlers,
    get_handler,
)
from src.architectures.base import get_forward_fn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("shard_server")

# ---------------------------------------------------------------------------
# Tensor packing — matches relay's packTensor/unpackTensor (lib/tensor.ts)
# Wire format: [shape_len:u32][shape...:u32[]][dtype_len:u32][dtype:utf8][data:bytes]
# ---------------------------------------------------------------------------

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


def pack_tensor(t: torch.Tensor) -> bytes:
    t_cont = t.contiguous()
    dtype_str = DTYPE_REVERSE.get(t.dtype, "float32")
    if t.dtype not in DTYPE_REVERSE:
        t_cont = t_cont.float()
        dtype_str = "float32"

    shape = list(t_cont.shape)
    dtype_bytes = dtype_str.encode("utf-8")
    data_bytes = t_cont.numpy().tobytes()

    buf = struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<I", d)
    buf += struct.pack("<I", len(dtype_bytes))
    buf += dtype_bytes
    buf += data_bytes
    return buf


def unpack_tensor(raw: bytes) -> torch.Tensor:
    off = 0
    (ndim,) = struct.unpack_from("<I", raw, off); off += 4
    shape = []
    for _ in range(ndim):
        (d,) = struct.unpack_from("<I", raw, off); off += 4
        shape.append(d)
    (dtype_len,) = struct.unpack_from("<I", raw, off); off += 4
    dtype_str = raw[off : off + dtype_len].decode("utf-8"); off += dtype_len
    data = raw[off:]

    import numpy as np
    np_dtype = {"float32": np.float32, "float16": np.float16, "bfloat16": np.float32}[dtype_str]
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(arr.copy())
    if dtype_str == "bfloat16":
        tensor = tensor.bfloat16()
    return tensor


# ---------------------------------------------------------------------------
# Module construction is delegated to per-architecture handlers under
# src/architectures/. The shard server only knows about shard *kinds*
# (embed / layers / head); the handler resolves family-specific details.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

app = FastAPI(title="HiveMind Shard Server")
server_state: dict = {}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "peer_id": server_state.get("peer_id"),
        "model_id": server_state.get("model_id"),
        "shard_type": server_state.get("shard_type"),
        "layer_start": server_state.get("layer_start"),
        "layer_end": server_state.get("layer_end"),
        "device": str(server_state.get("device")),
    }


@app.post("/forward")
async def forward(request: Request):
    raw = await request.body()
    t0 = time.perf_counter()

    hidden = unpack_tensor(raw).to(server_state["device"])
    shard_type = server_state["shard_type"]
    handler = server_state["handler"]
    module = server_state["module"]

    try:
        forward_fn = get_forward_fn(handler, shard_type)
    except ValueError as e:
        return Response(status_code=400, content=str(e))

    output = forward_fn(module, hidden)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Forward pass ({handler.name}/{shard_type}): "
        f"{list(hidden.shape)} -> {list(output.shape)} in {elapsed_ms:.1f}ms"
    )

    packed = pack_tensor(output.cpu().float())
    return Response(content=packed, media_type="application/octet-stream")


def detect_shard_type(shard_dir: Path, layer_start: int | None, layer_end: int | None) -> tuple[str, str]:
    """Return (shard_type, shard_filename)."""
    if layer_start is not None and layer_end is not None:
        return "layers", f"shard_layers_{layer_start}_{layer_end}.safetensors"
    # Auto-detect from available files
    for f in shard_dir.iterdir():
        if f.name == "shard_embed.safetensors":
            return "embed", f.name
        if f.name == "shard_head.safetensors":
            return "head", f.name
    raise RuntimeError("Cannot detect shard type. Specify --layers or ensure shard files exist.")


def register_with_relay(relay_url: str, peer_data: dict):
    """POST peer registration to relay."""
    import urllib.request
    import urllib.error

    url = f"{relay_url}/v1/peers/register"
    body = json.dumps(peer_data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        logger.info(f"Registered with relay at {relay_url}")
    except urllib.error.URLError as e:
        logger.warning(f"Could not register with relay: {e} (relay may not support /v1/peers/register yet)")


def main():
    parser = argparse.ArgumentParser(description="HiveMind Shard Server")
    parser.add_argument("--shard-dir", type=str, required=True, help="Directory with sharded model output")
    parser.add_argument("--layers", type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="Layer range to serve (e.g. --layers 4 10)")
    parser.add_argument("--shard-type", type=str, choices=["embed", "layers", "head"], default=None,
                        help="Override shard type detection")
    parser.add_argument("--port", type=int, default=9001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--relay", type=str, default="http://localhost:8787", help="Relay URL for registration")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, cuda:0)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="HuggingFace model ID (auto-detected from manifest if omitted)")
    parser.add_argument("--architecture", type=str, default=None,
                        help=(
                            "Architecture handler name "
                            "(overrides config.model_type detection)."
                        ))
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    if not shard_dir.exists():
        logger.error(f"Shard directory not found: {shard_dir}")
        sys.exit(1)

    # Load manifest for model config
    manifest_path = shard_dir / "manifest.json"
    model_id = args.model_id
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        model_id = model_id or manifest.get("model_id")
        logger.info(f"Loaded manifest: {model_id}")
    if not model_id:
        logger.error("Cannot determine model_id. Provide --model-id or ensure manifest.json exists.")
        sys.exit(1)

    # Detect shard type
    layer_start, layer_end = (args.layers if args.layers else (None, None))
    if args.shard_type:
        shard_type = args.shard_type
        if shard_type == "layers" and args.layers:
            shard_file = f"shard_layers_{layer_start}_{layer_end}.safetensors"
        elif shard_type == "embed":
            shard_file = "shard_embed.safetensors"
        elif shard_type == "head":
            shard_file = "shard_head.safetensors"
        else:
            shard_type, shard_file = detect_shard_type(shard_dir, layer_start, layer_end)
    else:
        shard_type, shard_file = detect_shard_type(shard_dir, layer_start, layer_end)

    shard_path = shard_dir / shard_file
    if not shard_path.exists():
        logger.error(f"Shard file not found: {shard_path}")
        sys.exit(1)

    # Load model config from HuggingFace
    device = torch.device(args.device)
    logger.info(f"Loading model config: {model_id}")
    config = AutoConfig.from_pretrained(model_id)

    # Resolve the architecture handler by model_type. Allow CLI override for
    # tricky cases (e.g. an architecture whose HF model_type string doesn't
    # match anything we've registered yet).
    handler_name = args.architecture or getattr(config, "model_type", None)
    if not handler_name:
        logger.error("Could not determine architecture from config; pass --architecture")
        sys.exit(1)
    try:
        handler = get_handler(handler_name)
    except ValueError as e:
        logger.error(str(e))
        logger.error(f"Available handlers: {available_handlers()}")
        sys.exit(1)
    logger.info(f"Using architecture handler: {handler.name} (model_type={handler_name!r})")

    # Load shard weights
    logger.info(f"Loading shard: {shard_path}")
    state_dict = load_file(str(shard_path))
    logger.info(f"Loaded {len(state_dict)} tensors from {shard_file}")

    # Build module via the handler
    if shard_type == "layers":
        spec = LayerShardSpec(layer_start=layer_start, layer_end=layer_end)
        module = handler.build_layers(config, state_dict, spec, device)
    elif shard_type == "embed":
        module = handler.build_embed(config, state_dict, device)
    elif shard_type == "head":
        module = handler.build_head(config, state_dict, device)
    else:
        raise ValueError(f"Unknown shard type: {shard_type}")

    # Set server state
    peer_id = f"peer_{shard_type}_{uuid.uuid4().hex[:8]}"
    server_state.update({
        "peer_id": peer_id,
        "model_id": model_id,
        "shard_type": shard_type,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "device": device,
        "module": module,
        "config": config,
        "handler": handler,
    })

    # Register with relay
    peer_data = {
        "id": peer_id,
        "endpoint": f"http://{args.host}:{args.port}",
        "modelId": model_id,
        "layerStart": layer_start or 0,
        "layerEnd": layer_end or 0,
        "status": "online",
        "lastSeen": int(time.time() * 1000),
        "maxBatchSize": 1,
        "shardType": shard_type,
    }
    register_with_relay(args.relay, peer_data)

    logger.info(f"Starting shard server: {shard_type} on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
