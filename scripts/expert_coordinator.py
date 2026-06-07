"""Granite-tiny expert coordinator -- Railway-friendly aiohttp service.

This service sits between the browser (running the Granite layer-5 MoE
router on WebGPU) and the 3-region expert pool (already deployed; see
``expert_railway_server.py``). It owns three responsibilities and no more:

1.  Accept one persistent WSS connection per browser at ``/ws``.
2.  Fan ``dispatch`` frames (browser-chosen ``(expert_id, hidden_state)``
    pairs) out to the appropriate expert services over a coordinator-owned
    WSS pool, in either ``wait_all`` or ``first_of_k`` mode.
3.  Run the real Granite ``lm_head`` over a 1536-dim hidden state and return
    top-5 token candidates so the demo renders real tokens.

The browser <-> coordinator wire is JSON over WSS (Section 1 of
``conductor/tracks/_shared/wire-frames.md``). The coordinator <-> expert
wire is the existing binary protocol from ``_wire.py`` (Section 2 of the
same doc).

This file is intentionally a single module -- Railway services should be
trivial to inspect, deploy, and roll back. Subdivisions live in ``_wire.py``
(protocol primitives) only.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import os
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from aiohttp import WSMsgType, web

# Add repo root to sys.path so `from scripts._wire import ...` works when run
# as `python scripts/expert_coordinator.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._wire import (  # noqa: E402  (post sys.path mutation)
    DTYPE_MAP,
    DTYPE_REVERSE,
    OP_ERROR,
    OP_FORWARD,
    _HEADER,
    expected_payload_size,
    pack_forward_frame,
    parse_expert_ids,
    unpack_response,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("expert_coordinator")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 1
HIDDEN_DIM = 1536  # Granite-tiny hidden state size
N_EXPERTS_TOTAL = 64

INBOUND_FRAME_MAX_BYTES = 8 * 1024 * 1024  # 8 MB hard cap, per wire-frames §1.1

# Per-call timeout to a single upstream socket. Pass-4 max wall-clock was 118 ms;
# 500 ms gives ~4x headroom without making a stuck upstream block forever.
UPSTREAM_CALL_TIMEOUT_S = 0.5

# Default ping cadence (aiohttp `heartbeat`) for both inbound browser WS and
# outbound upstream WS. Matches the existing expert services.
WS_HEARTBEAT_S = 20.0

# Connection backoff schedule and circuit-breaker constants.
RECONNECT_BACKOFF_S = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
CIRCUIT_THRESHOLD = 3      # consecutive failures before opening
CIRCUIT_COOLDOWN_S = 5.0   # how long to fast-fail after opening

# Default fixed wave-3 regions (informational; the actual mapping is built
# from env vars at boot).
DEFAULT_REGIONS = ("us-west2", "us-east4", "europe-west4")

REGION_ENV_VARS: dict[str, tuple[str, str]] = {
    # region_label: (URL env var, IDS env var)
    "us-west2": ("EXPERT_US_WEST_URL", "EXPERT_US_WEST_IDS"),
    "us-east4": ("EXPERT_US_EAST_URL", "EXPERT_US_EAST_IDS"),
    "europe-west4": ("EXPERT_EU_URL", "EXPERT_EU_IDS"),
}

# Closed set of error codes the coordinator may emit on the browser surface.
# Tests import this. Adding a new code is a wire-breaking change.
ERROR_CODES: set[str] = {
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
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RegionConfig:
    """Per-region URL + the set of expert ids that region hosts."""

    label: str
    url: str
    expert_ids: list[int]


@dataclass
class CoordinatorConfig:
    """Static configuration computed from environment at boot."""

    port: int = 8080
    regions: list[RegionConfig] = field(default_factory=list)
    head_local_path: Path = Path("/app/cache/shard_head.safetensors")
    head_s3_bucket: Optional[str] = None
    head_s3_endpoint: str = "https://t3.storageapi.dev"
    head_s3_access_key: Optional[str] = None
    head_s3_secret_key: Optional[str] = None
    head_s3_key: Optional[str] = None
    head_s3_region: str = "auto"
    model_id: str = "ibm-granite/granite-3.0-1b-a400m-instruct"
    upstream_call_timeout_s: float = UPSTREAM_CALL_TIMEOUT_S

    @property
    def expert_to_regions(self) -> dict[int, list[str]]:
        """Map ``expert_id -> [region_label, ...]`` in config order."""

        out: dict[int, list[str]] = {}
        for region in self.regions:
            for eid in region.expert_ids:
                out.setdefault(eid, []).append(region.label)
        return out

    @property
    def region_labels(self) -> list[str]:
        return [r.label for r in self.regions]


def load_config_from_env(env: Optional[dict[str, str]] = None) -> CoordinatorConfig:
    """Build a ``CoordinatorConfig`` from process env.

    Zero configured regions is a startup failure (``SystemExit``). Any single
    missing region warns and continues -- a degraded coordinator can still
    serve experts hosted on the regions that did come up.
    """

    env = env if env is not None else os.environ

    cfg = CoordinatorConfig()
    cfg.port = int(env.get("PORT", "8080"))

    regions: list[RegionConfig] = []
    for label, (url_var, ids_var) in REGION_ENV_VARS.items():
        url = env.get(url_var, "").strip()
        ids_spec = env.get(ids_var, "").strip()
        if not url and not ids_spec:
            continue
        if not url:
            logger.warning(
                "region %s: %s set but %s missing -- skipping",
                label, ids_var, url_var,
            )
            continue
        if not ids_spec:
            logger.warning(
                "region %s: %s set but %s missing -- skipping",
                label, url_var, ids_var,
            )
            continue
        ids = parse_expert_ids(ids_spec)
        if not ids:
            logger.warning(
                "region %s: %s parsed to empty id list -- skipping", label, ids_var,
            )
            continue
        regions.append(RegionConfig(label=label, url=url, expert_ids=ids))

    if not regions:
        raise SystemExit(
            "No expert regions configured. Set at least one of "
            "EXPERT_US_WEST_URL/IDS, EXPERT_US_EAST_URL/IDS, EXPERT_EU_URL/IDS."
        )
    cfg.regions = regions

    cfg.head_local_path = Path(env.get("HEAD_LOCAL_PATH", str(cfg.head_local_path)))
    cfg.head_s3_bucket = env.get("HEAD_S3_BUCKET") or None
    cfg.head_s3_endpoint = env.get("HEAD_S3_ENDPOINT", cfg.head_s3_endpoint)
    cfg.head_s3_access_key = env.get("HEAD_S3_ACCESS_KEY") or None
    cfg.head_s3_secret_key = env.get("HEAD_S3_SECRET_KEY") or None
    cfg.head_s3_key = env.get("HEAD_S3_KEY") or None
    cfg.head_s3_region = env.get("HEAD_S3_REGION", cfg.head_s3_region)
    cfg.model_id = env.get("MODEL_ID", cfg.model_id)
    cfg.upstream_call_timeout_s = float(
        env.get("UPSTREAM_CALL_TIMEOUT_S", str(cfg.upstream_call_timeout_s))
    )
    return cfg


# ---------------------------------------------------------------------------
# Upstream pool
# ---------------------------------------------------------------------------


@dataclass
class UpstreamState:
    """Bookkeeping for one upstream region connection."""

    region: RegionConfig
    ws: Any = None              # ClientWebSocketResponse or None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    connected: bool = False
    advertised_expert_ids: list[int] = field(default_factory=list)
    last_error: Optional[str] = None
    reconnects: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    degraded: bool = False
    _reader_task: Optional[asyncio.Task] = None


class UpstreamPool:
    """Holds one persistent WSS per upstream region.

    The existing binary protocol has no correlation id, so this pool serializes
    calls per upstream socket behind an ``asyncio.Lock``. Parallelism comes
    from having multiple upstreams, not from concurrent calls into one.
    """

    def __init__(
        self,
        regions: list[RegionConfig],
        *,
        ws_connect: Optional[Callable[..., Awaitable[Any]]] = None,
        call_timeout_s: float = UPSTREAM_CALL_TIMEOUT_S,
    ) -> None:
        self.regions = regions
        self.call_timeout_s = call_timeout_s
        # Injectable for tests; default uses aiohttp's client session.
        self._ws_connect = ws_connect
        self._session: Any = None
        self.states: dict[str, UpstreamState] = {
            r.label: UpstreamState(region=r) for r in regions
        }
        self._closing = False

    def info(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for label, st in self.states.items():
            out[label] = {
                "url": st.region.url,
                "connected": st.connected,
                "degraded": st.degraded,
                "advertised_expert_ids": list(st.advertised_expert_ids),
                "expected_expert_ids": list(st.region.expert_ids),
                "last_error": st.last_error,
                "reconnects": st.reconnects,
                "circuit_open": time.monotonic() < st.circuit_open_until,
            }
        return out

    async def start(self) -> None:
        """Spawn the reconnect supervisor for each region."""

        if self._ws_connect is None:
            # Lazy aiohttp client session for production use.
            import aiohttp
            self._session = aiohttp.ClientSession()

            async def _connect(url: str) -> Any:
                return await self._session.ws_connect(
                    url, max_msg_size=2**24, heartbeat=WS_HEARTBEAT_S,
                )

            self._ws_connect = _connect

        for st in self.states.values():
            st._reader_task = asyncio.create_task(self._supervise(st))

    async def close(self) -> None:
        self._closing = True
        tasks = [st._reader_task for st in self.states.values() if st._reader_task]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        for st in self.states.values():
            if st.ws is not None:
                try:
                    await st.ws.close()
                except Exception:
                    pass
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass

    async def _supervise(self, st: UpstreamState) -> None:
        """Backoff-reconnect loop for one upstream."""

        attempt = 0
        while not self._closing:
            try:
                logger.info("upstream %s: connecting %s", st.region.label, st.region.url)
                ws = await self._ws_connect(st.region.url)  # type: ignore[misc]
                hello_raw = await asyncio.wait_for(
                    self._recv_text(ws), timeout=10.0,
                )
                hello = json.loads(hello_raw)
                advertised = sorted(
                    e.get("expert_id", -1) for e in hello.get("experts", [])
                )
                # If the upstream advertises a different region or a different
                # expert-id slice than we expected, we still keep the socket but
                # mark the region degraded.
                st.advertised_expert_ids = advertised
                expected = sorted(st.region.expert_ids)
                if advertised and advertised != expected:
                    st.degraded = True
                    logger.warning(
                        "upstream %s: advertised expert ids %s != expected %s; "
                        "marking degraded",
                        st.region.label, advertised, expected,
                    )
                else:
                    st.degraded = False
                st.ws = ws
                st.connected = True
                st.last_error = None
                st.consecutive_failures = 0
                st.circuit_open_until = 0.0
                attempt = 0
                logger.info("upstream %s: connected", st.region.label)
                # Block until the connection drops; the WS object itself drives
                # disconnection.
                await self._wait_for_close(ws)
                logger.info("upstream %s: connection closed", st.region.label)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                st.last_error = str(e)
                logger.warning("upstream %s: %s", st.region.label, e)
            finally:
                if st.connected:
                    st.reconnects += 1
                st.connected = False
                st.ws = None

            if self._closing:
                return

            st.consecutive_failures += 1
            if st.consecutive_failures >= CIRCUIT_THRESHOLD:
                st.circuit_open_until = time.monotonic() + CIRCUIT_COOLDOWN_S
                logger.warning(
                    "upstream %s: circuit open for %.1fs",
                    st.region.label, CIRCUIT_COOLDOWN_S,
                )
            backoff = RECONNECT_BACKOFF_S[
                min(attempt, len(RECONNECT_BACKOFF_S) - 1)
            ]
            attempt += 1
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                return

    async def _recv_text(self, ws: Any) -> str:
        """Await one text frame and return its string body.

        Works with aiohttp ClientWebSocketResponse and with stub WS objects in
        tests that expose either ``receive_str`` or ``recv``.
        """

        if hasattr(ws, "receive_str"):
            return await ws.receive_str()
        if hasattr(ws, "recv"):
            v = await ws.recv()
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8")
            return v
        raise RuntimeError("upstream ws has no recv method")

    async def _recv_bytes(self, ws: Any) -> bytes:
        if hasattr(ws, "receive_bytes"):
            return await ws.receive_bytes()
        if hasattr(ws, "recv"):
            v = await ws.recv()
            if isinstance(v, str):
                return v.encode("utf-8")
            return bytes(v)
        raise RuntimeError("upstream ws has no recv method")

    async def _send_bytes(self, ws: Any, payload: bytes) -> None:
        if hasattr(ws, "send_bytes"):
            await ws.send_bytes(payload)
            return
        if hasattr(ws, "send"):
            await ws.send(payload)
            return
        raise RuntimeError("upstream ws has no send method")

    async def _wait_for_close(self, ws: Any) -> None:
        """Block until the upstream socket closes.

        For aiohttp ClientWebSocketResponse we await ``closed`` via
        ``receive()`` returning a CLOSE/CLOSED/ERROR frame. For test stubs we
        use a sentinel future on the stub if present, else best-effort.
        """

        if hasattr(ws, "wait_closed"):
            await ws.wait_closed()
            return
        # Default: poll closed flag.
        while True:
            closed = getattr(ws, "closed", False)
            if closed:
                return
            await asyncio.sleep(1.0)

    async def forward(
        self,
        region_label: str,
        expert_id: int,
        n_tokens: int,
        hidden: int,
        dtype_code: int,
        payload: bytes,
    ) -> tuple[bytes, float]:
        """Send one forward request to the named region; return ``(response_bytes, ms)``.

        Raises ``UpstreamUnavailable`` if the region is currently disconnected
        or its circuit breaker is open. Raises ``RuntimeError`` if the upstream
        replies with the binary error op.
        """

        st = self.states.get(region_label)
        if st is None:
            raise UpstreamUnavailable(f"unknown region {region_label!r}")
        if time.monotonic() < st.circuit_open_until:
            raise UpstreamUnavailable(f"region {region_label} circuit open")
        if not st.connected or st.ws is None:
            raise UpstreamUnavailable(f"region {region_label} not connected")

        frame = pack_forward_frame(expert_id, n_tokens, hidden, dtype_code, payload)

        async with st.lock:
            ws = st.ws
            if ws is None:
                raise UpstreamUnavailable(f"region {region_label} disconnected")
            t0 = time.perf_counter()
            try:
                await self._send_bytes(ws, frame)
                resp = await asyncio.wait_for(
                    self._recv_bytes(ws), timeout=self.call_timeout_s,
                )
            except asyncio.TimeoutError as e:
                raise UpstreamUnavailable(
                    f"region {region_label} timeout"
                ) from e
            ms = (time.perf_counter() - t0) * 1000.0

        # unpack_response raises RuntimeError on error op.
        unpack_response(resp)
        return resp, ms


class UpstreamUnavailable(RuntimeError):
    """Raised when a region is offline, its circuit is open, or timed out."""


# ---------------------------------------------------------------------------
# lm_head
# ---------------------------------------------------------------------------


class HeadModule:
    """Holds the ``lm_head.weight`` and optional ``final_layernorm.weight``.

    The matmul runs on CPU under ``torch.no_grad``. The forward expects a
    ``[seq, hidden]`` tensor and returns ``[seq, vocab]`` logits. The caller
    is responsible for slicing the last token before topk.
    """

    def __init__(
        self,
        lm_head_weight: "Any",
        final_layernorm_weight: Optional["Any"] = None,
        layernorm_eps: float = 1e-5,
    ) -> None:
        import torch  # local import keeps cold-boot fast for non-torch tests

        if not isinstance(lm_head_weight, torch.Tensor):
            lm_head_weight = torch.as_tensor(lm_head_weight)
        self.lm_head_weight = lm_head_weight.to(torch.float32)
        self.final_layernorm_weight = (
            torch.as_tensor(final_layernorm_weight).to(torch.float32)
            if final_layernorm_weight is not None
            else None
        )
        self.layernorm_eps = layernorm_eps
        self.vocab, self.hidden = self.lm_head_weight.shape

    def forward(self, hidden: "Any") -> "Any":
        import torch

        with torch.no_grad():
            x = hidden if isinstance(hidden, torch.Tensor) else torch.as_tensor(hidden)
            x = x.to(torch.float32)
            if self.final_layernorm_weight is not None:
                # RMSNorm: scale by rsqrt(mean(x^2) + eps), then * weight.
                # Matches GraniteRMSNorm in transformers.
                var = x.pow(2).mean(dim=-1, keepdim=True)
                x = x * torch.rsqrt(var + self.layernorm_eps)
                x = x * self.final_layernorm_weight
            logits = x @ self.lm_head_weight.T
            return logits


def load_head_from_safetensors(path: Path) -> HeadModule:
    """Load the head matmul weight (+ optional final RMSNorm weight) from a file.

    Granite-tiny uses **weight-tied** embeddings: there is no explicit
    ``lm_head.weight`` in the head shard. Instead ``model.embed_tokens.weight``
    is reused as the lm_head projection (this matches the conversion in
    ``scripts/convert_granite_streaming.py`` which packs `model.norm.*`,
    any `lm_head.*` (typically none on Granite-tiny), and
    `model.embed_tokens.*` into the head shard).

    The final RMSNorm tensor on Granite is ``model.norm.weight``; older
    HF layouts use ``model.final_layernorm.weight``. We try both.

    Lookup order:
      1. ``lm_head.weight`` (explicit, if the model was un-tied)
      2. ``model.lm_head.weight`` (older HF layouts)
      3. ``model.embed_tokens.weight`` (Granite-tiny tied case)

    Raises RuntimeError if none of the above are present.
    """

    from safetensors.torch import load_file

    state = load_file(str(path))
    head_w = (
        state.get("lm_head.weight")
        or state.get("model.lm_head.weight")
        or state.get("model.embed_tokens.weight")
    )
    if head_w is None:
        raise RuntimeError(
            f"{path} does not contain a head weight under any of: "
            "lm_head.weight, model.lm_head.weight, model.embed_tokens.weight"
        )
    ln_w = (
        state.get("model.norm.weight")
        or state.get("model.final_layernorm.weight")
        or state.get("final_layernorm.weight")
    )
    return HeadModule(head_w, ln_w)


def _ensure_head_shard(cfg: CoordinatorConfig) -> None:
    """Download ``shard_head.safetensors`` from S3 if not on the local volume.

    Mirrors ``_ensure_s3_experts`` in ``expert_railway_server.py``. If
    ``HEAD_S3_BUCKET`` is not set, this is a no-op (local-bake mode).
    """

    if cfg.head_local_path.exists():
        logger.info("head shard cached at %s", cfg.head_local_path)
        return
    if not cfg.head_s3_bucket:
        raise SystemExit(
            f"head shard not present at {cfg.head_local_path} and HEAD_S3_BUCKET "
            f"not set"
        )
    if not (cfg.head_s3_access_key and cfg.head_s3_secret_key and cfg.head_s3_key):
        raise SystemExit(
            "HEAD_S3_BUCKET set but HEAD_S3_ACCESS_KEY/SECRET_KEY/KEY missing"
        )

    import boto3
    from botocore.config import Config

    cfg.head_local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "downloading head shard from s3://%s/%s -> %s",
        cfg.head_s3_bucket, cfg.head_s3_key, cfg.head_local_path,
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg.head_s3_endpoint,
        aws_access_key_id=cfg.head_s3_access_key,
        aws_secret_access_key=cfg.head_s3_secret_key,
        region_name=cfg.head_s3_region,
        config=Config(
            s3={"addressing_style": "virtual"}, retries={"max_attempts": 5},
        ),
    )
    tmp = cfg.head_local_path.with_suffix(".partial")
    t0 = time.time()
    s3.download_file(cfg.head_s3_bucket, cfg.head_s3_key, str(tmp))
    tmp.rename(cfg.head_local_path)
    logger.info(
        "head shard download complete (%.1fs, %d MB)",
        time.time() - t0, cfg.head_local_path.stat().st_size // (1 << 20),
    )


def load_tokenizer(model_id: str):
    """Load the HF tokenizer for ``model_id``.

    Bake-in vs. boot-time download is a deployment choice; this function
    accepts either (``AutoTokenizer.from_pretrained`` picks the cached one if
    HF_HOME is set inside the image).
    """

    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(model_id)


# ---------------------------------------------------------------------------
# Coordinator app
# ---------------------------------------------------------------------------


@dataclass
class CoordinatorStats:
    """Rolling stats for /metrics."""

    started_at: float = field(default_factory=time.time)
    dispatch_count: int = 0
    lm_head_count: int = 0
    errors_by_code: dict[str, int] = field(default_factory=dict)
    last_100_dispatch_ms: list[float] = field(default_factory=list)
    last_100_lm_head_ms: list[float] = field(default_factory=list)

    def record_dispatch(self, ms: float) -> None:
        self.dispatch_count += 1
        self.last_100_dispatch_ms.append(ms)
        if len(self.last_100_dispatch_ms) > 100:
            self.last_100_dispatch_ms.pop(0)

    def record_lm_head(self, ms: float) -> None:
        self.lm_head_count += 1
        self.last_100_lm_head_ms.append(ms)
        if len(self.last_100_lm_head_ms) > 100:
            self.last_100_lm_head_ms.pop(0)

    def record_error(self, code: str) -> None:
        self.errors_by_code[code] = self.errors_by_code.get(code, 0) + 1

    def summary(self) -> dict[str, Any]:
        def pct(values: list[float], p: float) -> Optional[float]:
            if not values:
                return None
            s = sorted(values)
            k = (len(s) - 1) * p
            f = int(k)
            if f + 1 < len(s):
                return s[f] + (s[f + 1] - s[f]) * (k - f)
            return s[f]

        return {
            "uptime_s": time.time() - self.started_at,
            "dispatch_count": self.dispatch_count,
            "lm_head_count": self.lm_head_count,
            "errors_by_code": dict(self.errors_by_code),
            "dispatch_ms": {
                "p50": pct(self.last_100_dispatch_ms, 0.50),
                "p95": pct(self.last_100_dispatch_ms, 0.95),
                "p99": pct(self.last_100_dispatch_ms, 0.99),
                "n": len(self.last_100_dispatch_ms),
            },
            "lm_head_ms": {
                "p50": pct(self.last_100_lm_head_ms, 0.50),
                "p95": pct(self.last_100_lm_head_ms, 0.95),
                "p99": pct(self.last_100_lm_head_ms, 0.99),
                "n": len(self.last_100_lm_head_ms),
            },
        }


class Coordinator:
    """Runtime state shared across the aiohttp app handlers."""

    def __init__(
        self,
        config: CoordinatorConfig,
        *,
        upstream_pool: Optional[UpstreamPool] = None,
        head_module: Optional[HeadModule] = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = config
        self.pool = upstream_pool or UpstreamPool(
            config.regions, call_timeout_s=config.upstream_call_timeout_s,
        )
        self.head_module: Optional[HeadModule] = head_module
        self.tokenizer = tokenizer
        self.head_load_error: Optional[str] = None
        self.stats = CoordinatorStats()

    @property
    def lm_head_ready(self) -> bool:
        return self.head_module is not None and self.tokenizer is not None

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": time.time() - self.stats.started_at,
            "n_experts": N_EXPERTS_TOTAL,
            "regions": [r.label for r in self.config.regions],
            "lm_head_ready": self.lm_head_ready,
            "head_load_error": self.head_load_error,
        }

    def info_payload(self) -> dict[str, Any]:
        expert_map = {
            str(eid): regs
            for eid, regs in sorted(self.config.expert_to_regions.items())
        }
        return {
            "regions": [
                {
                    "label": r.label,
                    "url": r.url,
                    "expert_ids": list(r.expert_ids),
                }
                for r in self.config.regions
            ],
            "expert_to_regions": expert_map,
            "head_local_path": str(self.config.head_local_path),
            "head_s3_bucket": self.config.head_s3_bucket,
            "lm_head_ready": self.lm_head_ready,
            "upstream_status": self.pool.info(),
            "protocol_version": PROTOCOL_VERSION,
            "model_id": self.config.model_id,
        }

    def metrics_payload(self) -> dict[str, Any]:
        return self.stats.summary()

    def best_region_for(self, expert_id: int) -> Optional[str]:
        """Pick the first connected region that hosts the expert.

        Returns ``None`` if no region is connected for this expert id.
        """

        for label in self.config.expert_to_regions.get(expert_id, []):
            st = self.pool.states.get(label)
            if st is not None and st.connected:
                return label
        # Fallback: even if not connected, return the first configured region so
        # the caller can report ``upstream_unavailable`` with a region attached.
        return next(iter(self.config.expert_to_regions.get(expert_id, [])), None)

    def all_regions_for(self, expert_id: int) -> list[str]:
        return list(self.config.expert_to_regions.get(expert_id, []))


# ---------------------------------------------------------------------------
# Frame parsing / payload codec
# ---------------------------------------------------------------------------


class FrameError(Exception):
    """Raised when an inbound JSON frame fails validation.

    Attributes
    ----------
    code:
        One of ``ERROR_CODES``.
    message:
        Human-readable detail. Goes verbatim into the ``error`` text frame.
    request_id:
        The frame's request id if known; ``None`` otherwise.
    fatal:
        ``True`` if the connection must close after sending the error. See
        wire-frames.md §1.9.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: Optional[str] = None,
        fatal: bool = False,
    ) -> None:
        super().__init__(message)
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        self.code = code
        self.message = message
        self.request_id = request_id
        self.fatal = fatal


def _decode_b64(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise FrameError("payload_size_mismatch", f"base64 decode failed: {e}")


def _encode_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def validate_dispatch_frame(msg: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``dispatch`` JSON frame, returning the normalized dict.

    Raises ``FrameError`` for any wire-frames.md §1.4 / §1.9 violation. All
    error codes returned here are recoverable (the connection stays open).
    """

    request_id = msg.get("request_id")
    mode = msg.get("mode")
    if mode not in ("wait_all", "first_of_k"):
        raise FrameError(
            "bad_mode",
            f"mode must be 'wait_all' or 'first_of_k'; got {mode!r}",
            request_id=request_id,
        )
    calls = msg.get("calls")
    if not isinstance(calls, list):
        raise FrameError(
            "bad_shape", "calls must be a list", request_id=request_id,
        )
    k = msg.get("k")
    if not isinstance(k, int) or k < 1 or k > 32:
        raise FrameError(
            "bad_shape",
            f"k must be int 1..32; got {k!r}",
            request_id=request_id,
        )
    if len(calls) != k:
        raise FrameError(
            "bad_shape",
            f"k={k} but calls has {len(calls)} entries",
            request_id=request_id,
        )

    normalized_calls: list[dict[str, Any]] = []
    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            raise FrameError(
                "bad_shape",
                f"calls[{i}] must be object",
                request_id=request_id,
            )
        eid = call.get("expert_id")
        if not isinstance(eid, int) or eid < 0 or eid >= N_EXPERTS_TOTAL:
            raise FrameError(
                "bad_expert_id",
                f"calls[{i}].expert_id must be 0..63; got {eid!r}",
                request_id=request_id,
            )
        n_tokens = call.get("n_tokens")
        if not isinstance(n_tokens, int) or n_tokens < 1 or n_tokens > 512:
            raise FrameError(
                "bad_shape",
                f"calls[{i}].n_tokens must be 1..512; got {n_tokens!r}",
                request_id=request_id,
            )
        hidden = call.get("hidden")
        if hidden != HIDDEN_DIM:
            raise FrameError(
                "bad_shape",
                f"calls[{i}].hidden must equal {HIDDEN_DIM}; got {hidden!r}",
                request_id=request_id,
            )
        dtype = call.get("dtype")
        if dtype not in (0, 1):
            raise FrameError(
                "bad_shape",
                f"calls[{i}].dtype must be 0 (fp32) or 1 (fp16); got {dtype!r}",
                request_id=request_id,
            )
        payload_b64 = call.get("payload_b64")
        if not isinstance(payload_b64, str):
            raise FrameError(
                "bad_shape",
                f"calls[{i}].payload_b64 must be string",
                request_id=request_id,
            )
        payload_bytes = _decode_b64(payload_b64)
        try:
            expected = expected_payload_size(n_tokens, hidden, dtype)
        except ValueError as e:
            raise FrameError("bad_shape", str(e), request_id=request_id)
        if len(payload_bytes) != expected:
            raise FrameError(
                "payload_size_mismatch",
                (
                    f"calls[{i}].payload_b64 decoded to {len(payload_bytes)} "
                    f"bytes; expected {expected} "
                    f"(n_tokens={n_tokens}, hidden={hidden}, dtype={dtype})"
                ),
                request_id=request_id,
            )
        normalized_calls.append({
            "expert_id": eid,
            "n_tokens": n_tokens,
            "hidden": hidden,
            "dtype": dtype,
            "payload_bytes": payload_bytes,
        })

    return {
        "request_id": request_id,
        "mode": mode,
        "k": k,
        "calls": normalized_calls,
    }


def validate_lm_head_frame(msg: dict[str, Any]) -> dict[str, Any]:
    """Validate a WSS ``lm_head`` frame, returning normalized dict."""

    request_id = msg.get("request_id")
    shape = msg.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(x, int) and x > 0 for x in shape)
        or shape[1] != HIDDEN_DIM
    ):
        raise FrameError(
            "bad_shape",
            f"shape must be [seq, {HIDDEN_DIM}] with seq>=1; got {shape!r}",
            request_id=request_id,
        )
    seq, hidden = shape
    if seq > 512:
        raise FrameError(
            "bad_shape", f"shape[0] (seq) must be <=512; got {seq}",
            request_id=request_id,
        )
    dtype = msg.get("dtype", "fp16")
    if dtype not in ("fp16", "fp32"):
        raise FrameError(
            "bad_shape", f"dtype must be 'fp16' or 'fp32'; got {dtype!r}",
            request_id=request_id,
        )
    hidden_b64 = msg.get("hidden_b64")
    if not isinstance(hidden_b64, str):
        raise FrameError(
            "bad_shape", "hidden_b64 must be string", request_id=request_id,
        )
    payload = _decode_b64(hidden_b64)
    dtype_code = 1 if dtype == "fp16" else 0
    expected = expected_payload_size(seq, hidden, dtype_code)
    if len(payload) != expected:
        raise FrameError(
            "payload_size_mismatch",
            (
                f"hidden_b64 decoded to {len(payload)} bytes; expected "
                f"{expected} (seq={seq}, hidden={hidden}, dtype={dtype})"
            ),
            request_id=request_id,
        )
    return {
        "request_id": request_id,
        "shape": (seq, hidden),
        "dtype": dtype,
        "payload_bytes": payload,
    }


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


async def _do_one_call(
    pool: UpstreamPool,
    region: str,
    call: dict[str, Any],
) -> dict[str, Any]:
    """Single upstream call. Returns the per-call result dict."""

    try:
        resp_bytes, ms = await pool.forward(
            region_label=region,
            expert_id=call["expert_id"],
            n_tokens=call["n_tokens"],
            hidden=call["hidden"],
            dtype_code=call["dtype"],
            payload=call["payload_bytes"],
        )
        # Strip the binary header to get just the payload bytes for the response.
        payload_bytes = resp_bytes[_HEADER.size:]
        return {
            "expert_id": call["expert_id"],
            "region": region,
            "ms": ms,
            "payload_b64": _encode_b64(payload_bytes),
        }
    except UpstreamUnavailable as e:
        return {
            "expert_id": call["expert_id"],
            "region": region,
            "error": "upstream_unavailable",
            "message": str(e),
        }
    except RuntimeError as e:
        return {
            "expert_id": call["expert_id"],
            "region": region,
            "error": "upstream_unavailable",
            "message": str(e),
        }


async def dispatch_wait_all(
    coord: Coordinator,
    frame: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fan calls out; pick first region per expert; await all."""

    tasks: list[asyncio.Task] = []
    for call in frame["calls"]:
        eid = call["expert_id"]
        region = coord.best_region_for(eid)
        if region is None:
            tasks.append(
                asyncio.create_task(_unavail_result(eid, None))
            )
            continue
        tasks.append(asyncio.create_task(_do_one_call(coord.pool, region, call)))
    results = await asyncio.gather(*tasks)
    return list(results)


async def _unavail_result(expert_id: int, region: Optional[str]) -> dict[str, Any]:
    return {
        "expert_id": expert_id,
        "region": region,
        "error": "upstream_unavailable",
        "message": "no region hosts this expert",
    }


async def dispatch_first_of_k(
    coord: Coordinator,
    frame: dict[str, Any],
) -> list[dict[str, Any]]:
    """Race redundant regions for each call; return the winner per call."""

    out: list[Optional[dict[str, Any]]] = [None] * len(frame["calls"])

    async def race_one(i: int, call: dict[str, Any]) -> dict[str, Any]:
        eid = call["expert_id"]
        regions = coord.all_regions_for(eid)
        if not regions:
            return {
                "expert_id": eid,
                "error": "upstream_unavailable",
                "message": "no region hosts this expert",
            }
        tasks: list[asyncio.Task] = [
            asyncio.create_task(_do_one_call(coord.pool, r, call)) for r in regions
        ]
        errors: list[dict[str, Any]] = []
        pending = set(tasks)
        winner: Optional[dict[str, Any]] = None
        try:
            while pending and winner is None:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for d in done:
                    try:
                        r = d.result()
                    except Exception as e:  # noqa: BLE001
                        errors.append({"region": None, "message": str(e)})
                        continue
                    if "error" in r:
                        errors.append({
                            "region": r.get("region"),
                            "message": r.get("message", r["error"]),
                        })
                        continue
                    winner = r
                    break
        finally:
            # Loser tasks: cancel, but await so the per-upstream lock that
            # they hold (if any) is released after the upstream response is
            # drained. The lock is held only during the active send/recv,
            # so awaiting after cancel() is safe.
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

        if winner is not None:
            return winner
        return {
            "expert_id": eid,
            "error": "all_regions_failed",
            "errors": errors,
        }

    coros = [race_one(i, call) for i, call in enumerate(frame["calls"])]
    finished = await asyncio.gather(*coros)
    for i, r in enumerate(finished):
        out[i] = r
    return [r for r in out if r is not None]


# ---------------------------------------------------------------------------
# lm_head scoring
# ---------------------------------------------------------------------------


def _decode_hidden(payload: bytes, shape: tuple[int, int], dtype: str) -> "Any":
    import torch

    np_dtype = np.float16 if dtype == "fp16" else np.float32
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape).copy()
    t = torch.from_numpy(arr)
    return t


def score_lm_head(
    coord: Coordinator,
    payload: bytes,
    shape: tuple[int, int],
    dtype: str,
) -> tuple[list[dict[str, Any]], float]:
    """Run the lm_head on ``payload`` and return ``(top_5, ms)``.

    Raises ``FrameError("lm_head_not_ready", ...)`` if the head shard has not
    finished loading yet.
    """

    if not coord.lm_head_ready or coord.head_module is None:
        raise FrameError(
            "lm_head_not_ready",
            "head shard not loaded yet; retry shortly",
        )
    import torch

    t0 = time.perf_counter()
    hidden = _decode_hidden(payload, shape, dtype)
    logits = coord.head_module.forward(hidden)
    last = logits[-1] if logits.dim() == 2 else logits[0, -1]
    last = last.float()
    topk = torch.topk(last, k=5)
    top_5: list[dict[str, Any]] = []
    for tok_id, lv in zip(topk.indices.tolist(), topk.values.tolist()):
        try:
            tok_str = coord.tokenizer.decode([int(tok_id)])
        except Exception:  # noqa: BLE001
            tok_str = ""
        top_5.append({
            "token_id": int(tok_id),
            "token_str": tok_str,
            "logit": float(lv),
        })
    ms = (time.perf_counter() - t0) * 1000.0
    return top_5, ms


# ---------------------------------------------------------------------------
# aiohttp handlers
# ---------------------------------------------------------------------------


def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


async def handle_root(request: web.Request) -> web.Response:
    coord: Coordinator = request.app["coord"]
    return web.json_response(coord.health_payload(), headers=_cors_headers())


async def handle_health(request: web.Request) -> web.Response:
    coord: Coordinator = request.app["coord"]
    return web.json_response(coord.health_payload(), headers=_cors_headers())


async def handle_info(request: web.Request) -> web.Response:
    coord: Coordinator = request.app["coord"]
    return web.json_response(coord.info_payload(), headers=_cors_headers())


async def handle_metrics(request: web.Request) -> web.Response:
    coord: Coordinator = request.app["coord"]
    return web.json_response(coord.metrics_payload(), headers=_cors_headers())


async def handle_options(request: web.Request) -> web.Response:
    return web.Response(status=204, headers=_cors_headers())


async def handle_lm_head_http(request: web.Request) -> web.Response:
    coord: Coordinator = request.app["coord"]
    try:
        msg = await request.json()
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "bad_json", "message": "JSON parse failed"},
            status=400, headers=_cors_headers(),
        )
    # Re-shape the HTTP body into the WSS frame schema so validate_lm_head_frame
    # is the single source of truth.
    try:
        normalized = validate_lm_head_frame({
            "request_id": msg.get("request_id"),
            "shape": msg.get("shape"),
            "dtype": msg.get("dtype", "fp16"),
            "hidden_b64": msg.get("hidden_b64"),
        })
    except FrameError as e:
        coord.stats.record_error(e.code)
        return web.json_response(
            {"error": e.code, "message": e.message},
            status=400, headers=_cors_headers(),
        )
    try:
        top_5, ms = await asyncio.get_event_loop().run_in_executor(
            None,
            score_lm_head,
            coord, normalized["payload_bytes"], normalized["shape"], normalized["dtype"],
        )
    except FrameError as e:
        coord.stats.record_error(e.code)
        status = 503 if e.code == "lm_head_not_ready" else 400
        return web.json_response(
            {"error": e.code, "message": e.message},
            status=status, headers=_cors_headers(),
        )
    coord.stats.record_lm_head(ms)
    return web.json_response(
        {"top_5": top_5, "ms": ms}, headers=_cors_headers(),
    )


def _error_text_frame(
    code: str, message: str, request_id: Optional[str] = None,
) -> str:
    return json.dumps({
        "type": "error",
        "request_id": request_id,
        "code": code,
        "message": message,
    })


async def _send_error_frame(
    ws: web.WebSocketResponse,
    code: str,
    message: str,
    request_id: Optional[str] = None,
) -> None:
    await ws.send_str(_error_text_frame(code, message, request_id))


async def _handle_ws_message(
    coord: Coordinator,
    ws: web.WebSocketResponse,
    msg_data: str,
) -> None:
    """Dispatch a single inbound JSON text frame to the right handler."""

    if len(msg_data.encode("utf-8")) > INBOUND_FRAME_MAX_BYTES:
        coord.stats.record_error("frame_too_large")
        await _send_error_frame(
            ws, "frame_too_large", "inbound frame > 8 MB",
        )
        return
    try:
        parsed = json.loads(msg_data)
    except json.JSONDecodeError as e:
        coord.stats.record_error("bad_json")
        await _send_error_frame(ws, "bad_json", f"JSON parse failed: {e}")
        return
    if not isinstance(parsed, dict):
        coord.stats.record_error("bad_json")
        await _send_error_frame(ws, "bad_json", "frame must be a JSON object")
        return

    ftype = parsed.get("type")
    if ftype == "dispatch":
        await _handle_dispatch(coord, ws, parsed)
    elif ftype == "lm_head":
        await _handle_lm_head_ws(coord, ws, parsed)
    else:
        coord.stats.record_error("unknown_type")
        await _send_error_frame(
            ws,
            "unknown_type",
            f"unsupported frame type {ftype!r}",
            request_id=parsed.get("request_id"),
        )


async def _handle_dispatch(
    coord: Coordinator,
    ws: web.WebSocketResponse,
    parsed: dict[str, Any],
) -> None:
    try:
        frame = validate_dispatch_frame(parsed)
    except FrameError as e:
        coord.stats.record_error(e.code)
        await _send_error_frame(ws, e.code, e.message, e.request_id)
        return

    t0 = time.perf_counter()
    if frame["mode"] == "wait_all":
        results = await dispatch_wait_all(coord, frame)
    else:
        results = await dispatch_first_of_k(coord, frame)
    total_ms = (time.perf_counter() - t0) * 1000.0
    coord.stats.record_dispatch(total_ms)
    await ws.send_str(json.dumps({
        "type": "result",
        "request_id": frame["request_id"],
        "results": results,
        "ms": total_ms,
    }))


async def _handle_lm_head_ws(
    coord: Coordinator,
    ws: web.WebSocketResponse,
    parsed: dict[str, Any],
) -> None:
    try:
        normalized = validate_lm_head_frame(parsed)
    except FrameError as e:
        coord.stats.record_error(e.code)
        await _send_error_frame(ws, e.code, e.message, e.request_id)
        return
    try:
        top_5, ms = await asyncio.get_event_loop().run_in_executor(
            None,
            score_lm_head,
            coord,
            normalized["payload_bytes"],
            normalized["shape"],
            normalized["dtype"],
        )
    except FrameError as e:
        coord.stats.record_error(e.code)
        await _send_error_frame(ws, e.code, e.message, normalized["request_id"])
        return
    coord.stats.record_lm_head(ms)
    await ws.send_str(json.dumps({
        "type": "lm_head_result",
        "request_id": normalized["request_id"],
        "top_5": top_5,
        "ms": ms,
    }))


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    """The single WSS upgrade endpoint."""

    coord: Coordinator = request.app["coord"]

    # Reject non-WebSocket GETs to /ws with a structured 400.
    upgrade = request.headers.get("Upgrade", "").lower()
    if "websocket" not in upgrade:
        return web.json_response(
            {"error": "expected_websocket_upgrade"},
            status=400, headers=_cors_headers(),
        )

    origin = request.headers.get("Origin", "<missing>")
    logger.info("ws upgrade from origin=%s peer=%s", origin, request.remote)

    ws = web.WebSocketResponse(
        heartbeat=WS_HEARTBEAT_S,
        max_msg_size=INBOUND_FRAME_MAX_BYTES,
    )
    await ws.prepare(request)

    # ---- Phase 1: await hello -------------------------------------------
    try:
        first = await ws.receive(timeout=30.0)
    except asyncio.TimeoutError:
        coord.stats.record_error("expected_hello")
        await _send_error_frame(ws, "expected_hello", "no frame within 30s")
        await ws.close(code=1002)
        return ws

    if first.type == WSMsgType.BINARY:
        coord.stats.record_error("expected_hello")
        await _send_error_frame(ws, "expected_hello", "first frame must be text hello")
        await ws.close(code=1002)
        return ws
    if first.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
        return ws
    if first.type != WSMsgType.TEXT:
        coord.stats.record_error("expected_hello")
        await _send_error_frame(ws, "expected_hello", "first frame must be text hello")
        await ws.close(code=1002)
        return ws

    # Frame-size check on the hello frame itself.
    if len(first.data.encode("utf-8")) > INBOUND_FRAME_MAX_BYTES:
        coord.stats.record_error("frame_too_large")
        await _send_error_frame(ws, "frame_too_large", "hello frame > 8 MB")
        await ws.close(code=1002)
        return ws

    try:
        hello = json.loads(first.data)
    except json.JSONDecodeError as e:
        coord.stats.record_error("bad_json")
        await _send_error_frame(ws, "bad_json", f"hello JSON parse failed: {e}")
        await ws.close(code=1002)
        return ws

    if not isinstance(hello, dict) or hello.get("type") != "hello":
        coord.stats.record_error("expected_hello")
        await _send_error_frame(
            ws, "expected_hello", "first frame must be type=hello",
        )
        await ws.close(code=1002)
        return ws

    proto_v = hello.get("protocol_version", PROTOCOL_VERSION)
    if proto_v != PROTOCOL_VERSION:
        logger.warning(
            "ws client requested protocol_version=%r; coordinator is v%d",
            proto_v, PROTOCOL_VERSION,
        )

    await ws.send_str(json.dumps({
        "type": "ready",
        "n_experts": N_EXPERTS_TOTAL,
        "regions": [r.label for r in coord.config.regions],
        "lm_head_ready": coord.lm_head_ready,
        "protocol_version": PROTOCOL_VERSION,
    }))

    # ---- Phase 2: dispatch / lm_head loop --------------------------------
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                await _handle_ws_message(coord, ws, msg.data)
            except Exception as e:  # noqa: BLE001
                logger.exception("ws handler crashed: %s", e)
                coord.stats.record_error("internal_error")
                try:
                    await _send_error_frame(
                        ws, "internal_error", f"handler crashed: {e}",
                    )
                except Exception:
                    pass
        elif msg.type == WSMsgType.BINARY:
            coord.stats.record_error("unknown_type")
            await _send_error_frame(
                ws, "unknown_type", "binary frames are not supported after hello",
            )
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
            break
    return ws


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


async def build_app(config: CoordinatorConfig) -> web.Application:
    """Build the aiohttp Application but do NOT start the upstream pool yet.

    Tests call this and then inject stubs into ``app["coord"]`` before starting
    the runner. Production calls ``run_forever`` which spawns the pool.
    """

    coord = Coordinator(config)
    app = web.Application(
        client_max_size=INBOUND_FRAME_MAX_BYTES,
    )
    app["coord"] = coord

    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/info", handle_info)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_post("/lm_head", handle_lm_head_http)
    app.router.add_options("/lm_head", handle_options)
    app.router.add_get("/ws", handle_ws)

    return app


async def _load_head_background(coord: Coordinator) -> None:
    """Run the head-shard download + load + tokenizer load off the event loop."""

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _ensure_head_shard, coord.config)
        head_mod = await loop.run_in_executor(
            None, load_head_from_safetensors, coord.config.head_local_path,
        )
        tok = await loop.run_in_executor(
            None, load_tokenizer, coord.config.model_id,
        )
        coord.head_module = head_mod
        coord.tokenizer = tok
        logger.info(
            "lm_head ready: vocab=%d hidden=%d",
            head_mod.vocab, head_mod.hidden,
        )
    except Exception as e:  # noqa: BLE001
        coord.head_load_error = f"{type(e).__name__}: {e}"
        logger.exception("head shard load failed: %s", e)


async def run_forever(config: CoordinatorConfig) -> None:
    """Production entry point: bind ``$PORT`` and serve forever."""

    app = await build_app(config)
    coord: Coordinator = app["coord"]
    await coord.pool.start()
    head_task = asyncio.create_task(_load_head_background(coord))

    logger.info(
        "expert coordinator: regions=%s port=%d head=%s",
        [r.label for r in config.regions], config.port, config.head_local_path,
    )
    logger.warning(
        "STARTUP BANNER: this coordinator is unauthenticated. Do not run in "
        "production without the security track applied."
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=config.port)
    await site.start()
    try:
        await asyncio.Future()
    finally:
        head_task.cancel()
        await coord.pool.close()
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config_from_env()
    if args.port is not None:
        config.port = args.port
    asyncio.run(run_forever(config))


if __name__ == "__main__":
    main()
