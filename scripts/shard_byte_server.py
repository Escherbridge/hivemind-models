"""
Shard byte server for the browser-as-peer demo.

Serves safetensors files to the WebGPU client so the browser can run
the embedding + selected transformer layers locally.

  GET /embed.safetensors
        Full embedding shard (~295 MB). Streamed from disk.
        Browser caches in IndexedDB.

  GET /head.safetensors
        Full head shard (~295 MB) -- only needed if the browser ever
        wants to compute lm_head locally. Phase 1 keeps this on the
        coordinator instead.

  GET /layer/{idx}.safetensors
        A *new* safetensors file containing only the named layer's
        tensors, sliced server-side out of the relevant layer-group
        shard. For one layer this is ~80 MB instead of the 3.2 GB
        layer-group shard.

  GET /manifest.json
        Browser-friendly manifest: hidden_size, num_layers, vocab
        size, layer types, endpoint paths.

The server reads shards from disk (Railway volume mount) and falls
back to fetching from the same S3 bucket the expert servers use if
the file is missing on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("shard_byte_server")


SHARDS_DIR = Path(os.environ.get("SHARDS_DIR", "/data/shards"))
LOCAL_FALLBACK = Path(os.environ.get("SHARDS_LOCAL", "output/granite-tiny-q4g64"))
PORT = int(os.environ.get("PORT", "8080"))

# Granite-tiny layer-group layout (must match convert_granite_streaming.py)
LAYER_GROUPS = [(0, 9), (10, 19), (20, 29), (30, 39)]


def shard_for_layer(layer_idx: int) -> tuple[Path, tuple[int, int]]:
    for (lo, hi) in LAYER_GROUPS:
        if lo <= layer_idx <= hi:
            name = f"shard_layers_{lo}_{hi}.safetensors"
            for base in (SHARDS_DIR, LOCAL_FALLBACK):
                p = base / name
                if p.exists():
                    return p, (lo, hi)
            return SHARDS_DIR / name, (lo, hi)
    raise ValueError(f"layer {layer_idx} out of range 0..39")


def named_shard(name: str) -> Path:
    for base in (SHARDS_DIR, LOCAL_FALLBACK):
        p = base / name
        if p.exists():
            return p
    return SHARDS_DIR / name


# ---- safetensors layout ----------------------------------------------------
#
#   [u64 header_len]
#   [header_bytes: utf-8 JSON]
#       { "tensor_name": {"dtype": "...", "shape": [...], "data_offsets": [start, end]} ,
#         "__metadata__": {...} }
#   [data_bytes ...]
#
# data_offsets are relative to the start of the data section.


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode("utf-8"))
    return header, 8 + hlen


_EXPERT_TENSOR_NAMES = {
    "block_sparse_moe.input_linear.weight",
    "block_sparse_moe.output_linear.weight",
}


def slice_layer_safetensors(src: Path, layer_idx: int,
                            include_experts: bool = True) -> bytes:
    """Build a NEW safetensors file with only tensors whose names start
    with 'model.layers.{layer_idx}.'. If include_experts is False, the
    large stacked-expert tensors are omitted (browser hosts attention +
    router + shared MLP, but dispatches expert calls remotely)."""
    header, data_start = read_safetensors_header(src)
    prefix = f"model.layers.{layer_idx}."
    keep_keys = []
    for k in header.keys():
        if k == "__metadata__":
            continue
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix):]
        if (not include_experts) and suffix in _EXPERT_TENSOR_NAMES:
            continue
        keep_keys.append(k)
    if not keep_keys:
        raise ValueError(f"no tensors for layer {layer_idx} in {src.name}")

    new_header: dict = {}
    new_data = io.BytesIO()
    cursor = 0
    with src.open("rb") as f:
        for k in keep_keys:
            meta = header[k]
            start, end = meta["data_offsets"]
            size = end - start
            f.seek(data_start + start)
            buf = f.read(size)
            new_data.write(buf)
            new_header[k] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [cursor, cursor + size],
            }
            cursor += size

    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    out = io.BytesIO()
    out.write(struct.pack("<Q", len(header_bytes)))
    out.write(header_bytes)
    out.write(new_data.getvalue())
    return out.getvalue()


_LAYER_CACHE: dict[tuple[int, bool], bytes] = {}


def get_layer_bytes(layer_idx: int, include_experts: bool = True) -> bytes:
    key = (layer_idx, include_experts)
    if key in _LAYER_CACHE:
        return _LAYER_CACHE[key]
    src, _ = shard_for_layer(layer_idx)
    if not src.exists():
        raise FileNotFoundError(f"shard not on disk: {src}")
    t0 = time.time()
    payload = slice_layer_safetensors(src, layer_idx, include_experts=include_experts)
    elapsed = time.time() - t0
    logger.info("sliced layer %d (experts=%s): %d bytes in %.2fs",
                layer_idx, include_experts, len(payload), elapsed)
    _LAYER_CACHE[key] = payload
    return payload


# ---- S3 hydration ----------------------------------------------------------


def hydrate_from_bucket():
    bucket = os.environ.get("EXPERTS_S3_BUCKET")
    if not bucket:
        logger.info("no EXPERTS_S3_BUCKET; assuming local shards in %s",
                    SHARDS_DIR if SHARDS_DIR.exists() else LOCAL_FALLBACK)
        return

    # Phase 1: browser needs embed + layer-0..9 (for layer 5).
    needed = [
        "shard_embed.safetensors",
        "shard_layers_0_9.safetensors",
        # head.safetensors is large and only the coordinator service needs it,
        # but include it so curious browsers can fetch it for offline modes.
        "shard_head.safetensors",
    ]
    missing = [n for n in needed if not (SHARDS_DIR / n).exists()
               and not (LOCAL_FALLBACK / n).exists()]
    if not missing:
        logger.info("all needed shards already present locally")
        return

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("EXPERTS_S3_ENDPOINT", "https://t3.storageapi.dev")
    access_key = os.environ["EXPERTS_S3_ACCESS_KEY"]
    secret_key = os.environ["EXPERTS_S3_SECRET_KEY"]
    prefix = os.environ.get("SHARDS_S3_PREFIX", "shards/")
    s3 = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(s3={"addressing_style": "virtual"},
                      retries={"max_attempts": 5}),
    )

    def _fetch(name: str):
        key = prefix + name
        dest = SHARDS_DIR / name
        tmp = dest.with_suffix(dest.suffix + ".partial")
        logger.info("downloading s3://%s/%s -> %s", bucket, key, dest)
        t0 = time.time()
        s3.download_file(bucket, key, str(tmp))
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        logger.info("%s done: %.1f MB in %.1fs",
                    name, size_mb, time.time() - t0)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_fetch, missing))


# ---- HTTP routes -----------------------------------------------------------


def cors_headers() -> dict:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Cache-Control": "public, max-age=86400",
    }


async def handle_named_shard(_req, name: str):
    p = named_shard(name)
    if not p.exists():
        return web.Response(status=503, text=f"{name} not yet available")
    return web.FileResponse(p, headers={
        **cors_headers(),
        "Content-Type": "application/octet-stream",
    })


async def handle_embed(req):
    return await handle_named_shard(req, "shard_embed.safetensors")


async def handle_head(req):
    return await handle_named_shard(req, "shard_head.safetensors")


async def handle_layer(req: web.Request):
    try:
        layer_idx = int(req.match_info["idx"])
    except (KeyError, ValueError):
        return web.Response(status=400, text="bad layer index")
    if not (0 <= layer_idx <= 39):
        return web.Response(status=400, text="layer out of range 0..39")
    # ?experts=0 (default 1): omit the stacked-expert tensors so the
    # browser only carries attention + router + shared FFN + norms,
    # and dispatches expert calls to remote peers.
    include_experts = req.query.get("experts", "1") not in ("0", "false", "no")
    try:
        payload = await asyncio.get_event_loop().run_in_executor(
            None, get_layer_bytes, layer_idx, include_experts)
    except FileNotFoundError as e:
        return web.Response(status=503, text=str(e))
    except ValueError as e:
        return web.Response(status=404, text=str(e))
    return web.Response(body=payload, headers={
        **cors_headers(),
        "Content-Type": "application/octet-stream",
    })


async def handle_manifest(_req):
    return web.json_response({
        "model": "ibm-granite/granite-4.0-h-tiny",
        "hidden_size": 1536,
        "num_layers": 40,
        "layer_groups": LAYER_GROUPS,
        "num_local_experts": 64,
        "num_experts_per_tok": 6,
        "intermediate_size": 512,
        "vocab_size": 100352,
        "embedding_multiplier": 12.0,
        "logits_scaling": 6.0,
        "rms_norm_eps": 1e-5,
        "endpoints": {
            "embed": "/embed.safetensors",
            "layer": "/layer/{idx}.safetensors",
            "head": "/head.safetensors",
        },
    }, headers=cors_headers())


# -----------------------------------------------------------------------------
# manifest_v2: artifacts-array shape consumed by hivemind-client's
# loadGraniteManifest (browser-as-peer-phase1).
# Schema: wire-frames.md §3.2.
# -----------------------------------------------------------------------------

_SHA256_CACHE: dict[str, str] = {}


def _sha256_of(path: Path) -> str:
    """Return cached lowercase-hex sha256 of file; computes once per process."""
    key = str(path.resolve())
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA256_CACHE[key] = digest
    return digest


def _artifact_entry(uri: str, semantic_tag: str, file_path: Path,
                    body_bytes: bytes | None = None) -> dict:
    """Build a single artifact descriptor."""
    if body_bytes is not None:
        h = hashlib.sha256()
        h.update(body_bytes)
        return {
            "uri": uri,
            "sha256": h.hexdigest(),
            "byte_length": len(body_bytes),
            "semantic_tag": semantic_tag,
        }
    return {
        "uri": uri,
        "sha256": _sha256_of(file_path),
        "byte_length": file_path.stat().st_size,
        "semantic_tag": semantic_tag,
    }


def _manifest_v2_version() -> str:
    """Implicit version derived from the on-disk shards.

    Combines mtime of the three primary shards so a redeploy that pulls
    new bytes from S3 produces a new version string and the browser cache
    is invalidated.
    """
    parts: list[str] = []
    for name in ("shard_embed.safetensors", "shard_layers_0_9.safetensors",
                 "shard_head.safetensors"):
        p = named_shard(name)
        if p.exists():
            parts.append(f"{int(p.stat().st_mtime)}")
        else:
            parts.append("missing")
    return "-".join(parts)


async def handle_manifest_v2(_req):
    """Browser-friendly artifacts-array manifest (wire-frames §3.2).

    Used by hivemind-client's loadGraniteManifest. sha256 + byte_length
    are computed lazily (sha256 cached per-process), so the first call
    after a deploy is slow (~3 s on a 295 MB shard) and subsequent calls
    are instant.

    Layer slicing is done at GET time, so the layer-5-without-experts
    artifact's sha256 + byte_length cannot be precomputed from on-disk
    files; we slice once and hash the result the same way the layer
    endpoint will produce it on a real request.
    """
    embed_p = named_shard("shard_embed.safetensors")
    head_p = named_shard("shard_head.safetensors")
    layer5_src, _ = shard_for_layer(5)

    if not (embed_p.exists() and head_p.exists() and layer5_src.exists()):
        return web.Response(status=503, text="shards not yet hydrated")

    # Compute layer-5-without-experts bytes (cached after first call).
    layer5_bytes = await asyncio.get_event_loop().run_in_executor(
        None, get_layer_bytes, 5, False)

    artifacts = [
        _artifact_entry("/embed.safetensors", "embed", embed_p),
        _artifact_entry("/layer/5.safetensors?experts=0",
                        "layer-5-without-experts", layer5_src,
                        body_bytes=layer5_bytes),
        _artifact_entry("/head.safetensors", "head", head_p),
    ]

    return web.json_response({
        "model": "granite-tiny",
        "version": _manifest_v2_version(),
        "artifacts": artifacts,
        "tokenizer_uri": "/tokenizer.json",
    }, headers=cors_headers())


async def handle_health(_req):
    embed_p = named_shard("shard_embed.safetensors")
    layer5_src, _ = shard_for_layer(5)
    head_p = named_shard("shard_head.safetensors")
    return web.json_response({
        "status": "ok",
        "embed_ready": embed_p.exists(),
        "layer_0_9_shard_ready": layer5_src.exists(),
        "head_ready": head_p.exists(),
        "cached_layers": sorted(_LAYER_CACHE.keys()),
        "shards_dir": str(SHARDS_DIR),
    }, headers=cors_headers())


async def handle_options(_req):
    """Catch-all CORS preflight handler.

    Browsers issue OPTIONS preflight for cross-origin GETs that include
    non-simple headers (e.g. `Range`). Without an explicit handler aiohttp
    returns 405, which Chrome interprets as a CORS failure and aborts the
    follow-up GET. Return 204 + the standard CORS headers for every path.
    """
    return web.Response(status=204, headers=cors_headers())


def build_app() -> web.Application:
    app = web.Application(client_max_size=2**30)
    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/manifest.json", handle_manifest)
    app.router.add_get("/manifest_v2.json", handle_manifest_v2)
    app.router.add_get("/embed.safetensors", handle_embed)
    app.router.add_get("/head.safetensors", handle_head)
    app.router.add_get("/layer/{idx}.safetensors", handle_layer)
    # CORS preflight catch-all (must come after the GET routes so it does
    # not shadow them).
    app.router.add_route("OPTIONS", "/{tail:.*}", handle_options)
    return app


async def amain():
    await asyncio.get_event_loop().run_in_executor(None, hydrate_from_bucket)
    runner = web.AppRunner(build_app())
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    logger.info("shard_byte_server listening on 0.0.0.0:%d", PORT)
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(amain())
