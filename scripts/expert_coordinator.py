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
from collections import deque
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
# outbound upstream WS.
# Default raised to 50s (was 20s) to sit just under Railway's 60s edge proxy
# idle timeout without thrashing. Override via WS_HEARTBEAT_S env var (e.g.,
# set to 0 to disable WS heartbeat entirely).
WS_HEARTBEAT_S = float(os.environ.get("WS_HEARTBEAT_S", "50.0"))

# Connection backoff schedule and circuit-breaker constants.
# Backoff extended at the tail so persistent failures stop hammering the
# upstream (and Railway's edge) once the circuit has been tripping.
RECONNECT_BACKOFF_S = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 300.0]
CIRCUIT_THRESHOLD = 3      # consecutive failures before opening
CIRCUIT_COOLDOWN_S = 30.0  # how long to fast-fail after opening (was 5s)

# Default fixed wave-3 regions (informational; the actual mapping is built
# from env vars at boot).
DEFAULT_REGIONS = ("us-west2", "us-east4", "europe-west4")

REGION_ENV_VARS: dict[str, tuple[str, str]] = {
    # region_label: (URL env var, IDS env var)
    "us-west2": ("EXPERT_US_WEST_URL", "EXPERT_US_WEST_IDS"),
    "us-east4": ("EXPERT_US_EAST_URL", "EXPERT_US_EAST_IDS"),
    "europe-west4": ("EXPERT_EU_URL", "EXPERT_EU_IDS"),
}

# Wave-4 desktop-peer: dispatch moved off the coordinator to direct
# browser <-> peer WebRTC DataChannels (see conductor/tracks/_shared/
# wire-frames.md §6). When LEGACY_DISPATCH is "0" or unset (the default
# in Wave 4), the upstream region pool is not started, the `dispatch`
# frame on `/ws` returns a `dispatch_moved` error, and the lack of
# EXPERT_*_URL env vars is not a startup failure. Set LEGACY_DISPATCH=1
# to restore the Wave-3 behavior (still required for old test fixtures
# that exercise the dispatch surface).
#
# The constant is the module-import-time snapshot for callers that don't
# pass an explicit env dict (e.g. the production main() path). Tests and
# any code path that takes an env override should call ``_legacy_dispatch_enabled``
# below with the same env dict they pass to ``load_config_from_env``,
# so the flag stays consistent with the rest of the config.
LEGACY_DISPATCH = os.environ.get("LEGACY_DISPATCH", "0") == "1"


def _legacy_dispatch_enabled(env: Optional[dict[str, str]] = None) -> bool:
    """Return whether the legacy Wave-3 dispatch surface is enabled.

    Reads ``LEGACY_DISPATCH`` from ``env`` (or ``os.environ`` if None). The
    module-level ``LEGACY_DISPATCH`` constant is the snapshot used by
    runtime code paths (``_handle_ws_message``, ``UpstreamPool.start``)
    that don't have an env dict at hand; for unit tests and other paths
    that DO have an env dict, this helper guarantees the same answer.
    """
    e = env if env is not None else os.environ
    return e.get("LEGACY_DISPATCH", "0") == "1"


# Wave-4 desktop-peer: in-memory peer registry. Peer TTL defaults to 90 s
# per wire-frames.md §1.10.1 (announce extends, sweeper drops expired).
# The sweeper interval is half the TTL by default so a missed announce is
# detected within one sweep cycle.
PEER_TTL_S = float(os.environ.get("PEER_TTL_S", "90.0"))
PEER_SWEEP_INTERVAL_S = float(os.environ.get("PEER_SWEEP_INTERVAL_S", "30.0"))
PEER_LIST_MAX = 64

# Wave-4 desktop-peer: /peers/socket inbound frame cap. SDP blobs are at
# most a few KB so 256 KB is plenty (vs. the 8 MB cap on /ws for dispatch
# payloads). See wire-frames.md §5.1.
PEER_SOCKET_FRAME_MAX_BYTES = 256 * 1024

# Wave-4 desktop-peer: /peers/socket heartbeat cadence. Coordinator pings
# the peer every 30 s per wire-frames.md §5.6 / §5.1 step 7. aiohttp's
# native WS keepalive handles control-frame pong replies; no JSON-level
# ping is needed.
PEER_SOCKET_HEARTBEAT_S = float(os.environ.get("PEER_SOCKET_HEARTBEAT_S", "30.0"))

# Wave-4 desktop-peer: inbox TTL for /peers/signal GET drains. Per
# wire-frames.md §1.10.4, entries TTL out after 10 s if the browser
# never polls them. The inbox keys are (peer_id, nonce); the deque is
# trimmed on every push and on every GET.
PEER_SIGNAL_INBOX_TTL_S = float(os.environ.get("PEER_SIGNAL_INBOX_TTL_S", "10.0"))

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
    # Wave-4 desktop-peer additions (see wire-frames.md §1.9):
    "dispatch_moved",
    "peer_offline",
    "unknown_peer",
    "peer_id_collision",
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

    if not regions and _legacy_dispatch_enabled(env):
        raise SystemExit(
            "No expert regions configured. Set at least one of "
            "EXPERT_US_WEST_URL/IDS, EXPERT_US_EAST_URL/IDS, EXPERT_EU_URL/IDS, "
            "or unset LEGACY_DISPATCH to run in Wave-4 signaling-only mode."
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
        """Spawn the reconnect supervisor for each region.

        Wave-4 desktop-peer: when ``self.regions`` is empty (the new
        default when ``LEGACY_DISPATCH`` is unset), this is a no-op. No
        aiohttp client session is opened, no supervisor task is spawned,
        and the coordinator stops thrashing reconnects against the dead
        Wave-3 ``expert-*-production`` URLs.
        """

        if not self.regions:
            logger.info(
                "upstream pool: no regions configured (LEGACY_DISPATCH=%s); "
                "dispatch is browser <-> peer over WebRTC in Wave-4",
                LEGACY_DISPATCH,
            )
            return

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
# Wave-4 desktop-peer: peer registry
# ---------------------------------------------------------------------------


@dataclass
class PeerEntry:
    """One entry in the in-memory peer registry.

    Mirrors the wire-frames.md §1.10.1 / §1.10.2 contract:

    - ``peer_id`` is the v4 UUID the peer generated and persisted.
    - ``capabilities`` is the most recent announce body's capabilities dict.
    - ``expires_at_s`` is wall-clock seconds since epoch; entries older than
      this are dropped by ``PeerRegistry.sweep_expired``.
    - ``last_seen_s`` tracks the most recent announce timestamp; the list
      endpoint uses this to compute ``last_seen_ms_ago``.
    - ``ws`` is the live aiohttp WebSocketResponse for ``/peers/socket/
      <peer_id>``, set by Task 1.5. ``None`` until the peer opens its
      socket; the list endpoint filters by this being non-None.
    """

    peer_id: str
    capabilities: dict
    expires_at_s: float
    last_seen_s: float
    ws: Any = None  # web.WebSocketResponse | None, late-bound to dodge import cycle


class PeerRegistry:
    """Thread-safe (single event loop) in-memory peer registry.

    Lifecycle:

    1. Peer POSTs ``/peers/announce`` -> ``register(peer_id, capabilities)``.
    2. Peer opens WSS to ``/peers/socket/<peer_id>`` -> ``attach_ws``.
    3. Peer re-announces every 60 s -> ``register`` extends TTL.
    4. Heartbeat or sweeper finds entry expired -> ``sweep_expired``.

    The registry is held by ``Coordinator`` and read by the HTTP / WSS
    handlers. No locks: aiohttp dispatches inside one asyncio event loop,
    so all access is single-threaded.
    """

    def __init__(self, ttl_s: float = PEER_TTL_S) -> None:
        self.ttl_s = ttl_s
        self._entries: dict[str, PeerEntry] = {}

    # -- mutation ---------------------------------------------------------

    def register(self, peer_id: str, capabilities: dict) -> PeerEntry:
        """Create or refresh the entry for ``peer_id``.

        Idempotent: re-announce with the same id reuses the entry object,
        updates ``capabilities``, and pushes ``expires_at_s`` forward.
        """
        now = time.time()
        entry = self._entries.get(peer_id)
        if entry is None:
            entry = PeerEntry(
                peer_id=peer_id,
                capabilities=dict(capabilities),
                expires_at_s=now + self.ttl_s,
                last_seen_s=now,
            )
            self._entries[peer_id] = entry
        else:
            entry.capabilities = dict(capabilities)
            entry.expires_at_s = now + self.ttl_s
            entry.last_seen_s = now
        return entry

    def attach_ws(self, peer_id: str, ws: Any) -> Optional[PeerEntry]:
        """Bind the live WSS to an existing entry. Returns None if no entry."""
        entry = self._entries.get(peer_id)
        if entry is None:
            return None
        entry.ws = ws
        entry.last_seen_s = time.time()
        return entry

    def detach_ws(self, peer_id: str) -> None:
        """Clear the WSS pointer (e.g. on disconnect). Entry stays for TTL."""
        entry = self._entries.get(peer_id)
        if entry is not None:
            entry.ws = None

    def sweep_expired(self, now: Optional[float] = None) -> list[str]:
        """Drop entries whose ``expires_at_s`` is in the past. Returns dropped ids."""
        now = now if now is not None else time.time()
        dropped: list[str] = []
        for pid, entry in list(self._entries.items()):
            if entry.expires_at_s <= now:
                dropped.append(pid)
                del self._entries[pid]
        return dropped

    # -- read -------------------------------------------------------------

    def get(self, peer_id: str) -> Optional[PeerEntry]:
        return self._entries.get(peer_id)

    def list_active(
        self,
        now: Optional[float] = None,
        require_ws: bool = False,
        limit: int = PEER_LIST_MAX,
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` non-expired entries as JSON-shaped dicts.

        Sorted by ``last_seen_ms_ago`` ascending (most-recently-seen first).
        If ``require_ws`` is True, only entries with a live ``/peers/socket``
        connection are returned — this is what ``GET /peers/list`` uses, so
        the browser only learns about peers it can actually signal to.
        """
        now = now if now is not None else time.time()
        active: list[PeerEntry] = []
        for entry in self._entries.values():
            if entry.expires_at_s <= now:
                continue
            if require_ws and entry.ws is None:
                continue
            active.append(entry)
        active.sort(key=lambda e: e.last_seen_s, reverse=True)
        out: list[dict[str, Any]] = []
        for entry in active[:limit]:
            out.append({
                "peer_id": entry.peer_id,
                "capabilities": dict(entry.capabilities),
                "last_seen_ms_ago": max(0, int((now - entry.last_seen_s) * 1000)),
            })
        return out


class SignalInbox:
    """Per-(peer_id, browser-nonce) inbox of peer->browser signal frames.

    The browser does not hold a socket to the coordinator for signaling
    answers — it polls ``GET /peers/signal/<peer_id>?inbox=<nonce>``
    (wire-frames.md §1.10.4). Each peer-side ``signal`` frame whose
    ``to`` field matches a known nonce is stashed here; the next GET
    drains it.

    Entries TTL out after ``PEER_SIGNAL_INBOX_TTL_S`` (default 10 s)
    so an abandoned browser tab doesn't leak the entries forever. The
    TTL is enforced lazily on every push and drain — no background
    sweeper task; the inbox surface is rarely-used relative to the
    registry sweeper, so lazy cleanup is enough.
    """

    def __init__(self, ttl_s: float = PEER_SIGNAL_INBOX_TTL_S) -> None:
        self.ttl_s = ttl_s
        # key: (peer_id, nonce) -> deque[(enqueued_at_s, frame_dict)]
        self._inboxes: dict[tuple[str, str], deque[tuple[float, dict]]] = {}

    def _prune(self, now: float, key: tuple[str, str]) -> None:
        dq = self._inboxes.get(key)
        if dq is None:
            return
        while dq and (now - dq[0][0]) > self.ttl_s:
            dq.popleft()
        if not dq:
            self._inboxes.pop(key, None)

    def push(self, peer_id: str, nonce: str, frame: dict) -> None:
        """Stash one peer->browser frame for the (peer_id, nonce) inbox."""
        now = time.time()
        key = (peer_id, nonce)
        dq = self._inboxes.setdefault(key, deque())
        dq.append((now, dict(frame)))
        self._prune(now, key)

    def drain(self, peer_id: str, nonce: str) -> list[dict]:
        """Return and clear all non-expired frames for ``(peer_id, nonce)``."""
        now = time.time()
        key = (peer_id, nonce)
        self._prune(now, key)
        dq = self._inboxes.pop(key, None)
        if dq is None:
            return []
        return [frame for (_ts, frame) in dq]

    def sweep_expired(self, now: Optional[float] = None) -> int:
        """Drop all expired entries; return the number of inboxes pruned."""
        now = now if now is not None else time.time()
        dropped = 0
        for key in list(self._inboxes.keys()):
            self._prune(now, key)
            if key not in self._inboxes:
                dropped += 1
        return dropped


def _is_valid_uuid_v4(s: object) -> bool:
    """Per wire-frames.md §4.4: lowercase hyphenated UUID v4, 36 chars."""
    if not isinstance(s, str) or len(s) != 36:
        return False
    try:
        u = uuid.UUID(s)
    except (ValueError, AttributeError, TypeError):
        return False
    if u.version != 4:
        return False
    # Reject UUIDs with uppercase hex (the canonical form is lowercase).
    return s == str(u)


def validate_announce_body(body: object) -> dict[str, Any]:
    """Validate a ``POST /peers/announce`` body. Raises ``FrameError`` on bad input.

    Returns a normalized dict ``{"peer_id": str, "capabilities": dict}``.
    Capabilities are validated per wire-frames.md §1.10.1:

    - ``expert_ids``: non-empty sorted list of ints in [0, 63].
    - ``compute_mode``: ``"real"`` or ``"echo"``.
    - ``dtype``: ``"fp16"`` (Wave-4 has one model).
    - ``hidden``: must equal ``HIDDEN_DIM`` (1536).
    """
    if not isinstance(body, dict):
        raise FrameError("bad_json", "announce body must be a JSON object")

    peer_id = body.get("peer_id")
    if not _is_valid_uuid_v4(peer_id):
        raise FrameError("bad_json", "peer_id must be a lowercase v4 UUID string")

    caps = body.get("capabilities")
    if not isinstance(caps, dict):
        raise FrameError("bad_shape", "capabilities must be a JSON object")

    expert_ids = caps.get("expert_ids")
    if not isinstance(expert_ids, list) or not expert_ids:
        raise FrameError("bad_shape", "capabilities.expert_ids must be a non-empty list")
    for eid in expert_ids:
        if not isinstance(eid, int) or isinstance(eid, bool):
            raise FrameError("bad_expert_id", f"expert id {eid!r} must be an integer")
        if not (0 <= eid < N_EXPERTS_TOTAL):
            raise FrameError(
                "bad_expert_id",
                f"expert id {eid} outside [0, {N_EXPERTS_TOTAL - 1}]",
            )

    compute_mode = caps.get("compute_mode")
    if compute_mode not in ("real", "echo"):
        raise FrameError(
            "bad_shape",
            f"compute_mode {compute_mode!r} must be 'real' or 'echo'",
        )

    dtype = caps.get("dtype")
    if dtype != "fp16":
        raise FrameError("bad_shape", f"dtype {dtype!r} must be 'fp16'")

    hidden = caps.get("hidden")
    if hidden != HIDDEN_DIM:
        raise FrameError("bad_shape", f"hidden {hidden!r} must equal {HIDDEN_DIM}")

    return {
        "peer_id": peer_id,
        "capabilities": {
            "expert_ids": list(expert_ids),
            "compute_mode": compute_mode,
            "dtype": dtype,
            "hidden": hidden,
        },
    }


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
        # KEEP the weight in its source dtype (fp16 for Granite). Upcasting
        # to fp32 here doubles permanent resident memory (300MB -> 600MB on
        # Granite's [100352, 1536] head). The matmul in forward() handles
        # dtype promotion by upcasting the small hidden state instead.
        self.lm_head_weight = lm_head_weight
        if final_layernorm_weight is not None:
            ln_w = torch.as_tensor(final_layernorm_weight)
        else:
            ln_w = None
        self.final_layernorm_weight = ln_w
        self.layernorm_eps = layernorm_eps
        self.vocab, self.hidden = self.lm_head_weight.shape

    def forward(self, hidden: "Any") -> "Any":
        import torch

        with torch.no_grad():
            x = hidden if isinstance(hidden, torch.Tensor) else torch.as_tensor(hidden)
            # Match the weight's dtype so the matmul does not allocate a
            # full-precision copy. fp16 matmul on CPU is fine for Granite-tiny
            # at single-token lm_head scale.
            target_dtype = self.lm_head_weight.dtype
            x = x.to(target_dtype)
            if self.final_layernorm_weight is not None:
                ln_w = self.final_layernorm_weight.to(target_dtype)
                # RMSNorm: scale by rsqrt(mean(x^2) + eps), then * weight.
                # Matches GraniteRMSNorm in transformers.
                # Promote just the variance computation to fp32 to avoid
                # half-precision overflow on the squared sum, then return
                # to target dtype before applying weight.
                var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
                x = x * torch.rsqrt(var + self.layernorm_eps).to(target_dtype)
                x = x * ln_w
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
    # Explicit None checks — `tensor or other` triggers torch.Tensor.__bool__
    # which raises "Boolean value of Tensor with more than one value is
    # ambiguous" on any non-scalar tensor.
    head_w = None
    for key in (
        "lm_head.weight",
        "model.lm_head.weight",
        "model.embed_tokens.weight",
    ):
        candidate = state.get(key)
        if candidate is not None:
            head_w = candidate
            break
    if head_w is None:
        raise RuntimeError(
            f"{path} does not contain a head weight under any of: "
            "lm_head.weight, model.lm_head.weight, model.embed_tokens.weight"
        )
    ln_w = None
    for key in (
        "model.norm.weight",
        "model.final_layernorm.weight",
        "final_layernorm.weight",
    ):
        candidate = state.get(key)
        if candidate is not None:
            ln_w = candidate
            break
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
        peer_registry: Optional[PeerRegistry] = None,
        signal_inbox: Optional[SignalInbox] = None,
    ) -> None:
        self.config = config
        self.pool = upstream_pool or UpstreamPool(
            config.regions, call_timeout_s=config.upstream_call_timeout_s,
        )
        self.head_module: Optional[HeadModule] = head_module
        self.tokenizer = tokenizer
        self.head_load_error: Optional[str] = None
        self.stats = CoordinatorStats()
        # Wave-4 desktop-peer: in-memory peer registry. The sweeper task is
        # spawned by ``run_forever``; tests that don't need TTL sweeping
        # just construct the registry and skip the sweeper.
        self.peer_registry: PeerRegistry = peer_registry or PeerRegistry()
        self._peer_sweeper_task: Optional[asyncio.Task] = None
        # Wave-4 desktop-peer: per-(peer_id, browser-nonce) inbox of
        # peer->browser signal frames. Drained by GET /peers/signal.
        self.signal_inbox: SignalInbox = signal_inbox or SignalInbox()

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


# ---------------------------------------------------------------------------
# Wave-4 desktop-peer: /peers/* handlers
# ---------------------------------------------------------------------------


async def handle_peers_list(request: web.Request) -> web.Response:
    """GET /peers/list. See wire-frames.md §1.10.2.

    Returns up to ``PEER_LIST_MAX`` (64) non-expired peers that have a
    live ``/peers/socket/<peer_id>`` WSS attached. Sorted by
    ``last_seen_ms_ago`` ascending (most-recently-seen first).
    """
    coord: Coordinator = request.app["coord"]
    peers = coord.peer_registry.list_active(require_ws=True)
    return web.json_response({"peers": peers}, headers=_cors_headers())


async def handle_peers_announce(request: web.Request) -> web.Response:
    """POST /peers/announce. See wire-frames.md §1.10.1.

    Validates the announce body, upserts the peer registry entry, returns
    ``{"status": "announced", "peer_id": ..., "expires_at_s": ...,
    "ttl_s": ...}``. Idempotent on ``peer_id``: re-announce extends TTL
    and updates capabilities.
    """
    coord: Coordinator = request.app["coord"]
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response(
            {"error": "bad_json", "message": "JSON parse failed"},
            status=400, headers=_cors_headers(),
        )
    try:
        normalized = validate_announce_body(body)
    except FrameError as e:
        coord.stats.record_error(e.code)
        return web.json_response(
            {"error": e.code, "message": e.message},
            status=400, headers=_cors_headers(),
        )
    entry = coord.peer_registry.register(
        normalized["peer_id"], normalized["capabilities"],
    )
    return web.json_response(
        {
            "status": "announced",
            "peer_id": entry.peer_id,
            "expires_at_s": entry.expires_at_s,
            "ttl_s": int(coord.peer_registry.ttl_s),
        },
        headers=_cors_headers(),
    )


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
        # Wave-4 desktop-peer: when no upstream regions are configured (the
        # default outside the Wave-3 fixtures), dispatch is now browser
        # <-> peer over WebRTC DataChannels. The /ws endpoint still serves
        # lm_head; the dispatch frame is a soft-rejected legacy surface
        # and the connection stays open per wire-frames.md §1.9.
        #
        # The runtime gate is "config.regions is non-empty" rather than a
        # raw env-var check, so unit-test fixtures that set up regions
        # explicitly (test_coordinator_ws.py) continue to exercise the
        # legacy dispatch path without needing to monkeypatch env vars.
        if coord.config.regions:
            await _handle_dispatch(coord, ws, parsed)
        else:
            coord.stats.record_error("dispatch_moved")
            await _send_error_frame(
                ws,
                "dispatch_moved",
                "Dispatch is now browser <-> peer over WebRTC; "
                "see /peers/list and wire-frames.md §6.",
                request_id=parsed.get("request_id"),
            )
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


# ---------------------------------------------------------------------------
# Wave-4 desktop-peer: /peers/socket WSS + /peers/signal HTTP relay
# ---------------------------------------------------------------------------


def _validate_peer_hello(body: object, path_peer_id: str) -> dict[str, Any]:
    """Validate the first peer->coord frame on ``/peers/socket/<peer_id>``.

    Per wire-frames.md §5.2: ``type == "hello"``, ``peer_id`` matches the
    path, and ``capabilities`` matches the most recent ``/peers/announce``
    body for that peer_id. Raises FrameError on any mismatch.
    """
    if not isinstance(body, dict):
        raise FrameError("bad_json", "hello must be a JSON object")
    if body.get("type") != "hello":
        raise FrameError("expected_hello", "first frame must be type=hello")
    body_pid = body.get("peer_id")
    if not _is_valid_uuid_v4(body_pid):
        raise FrameError(
            "unknown_peer", "hello.peer_id must be a lowercase v4 UUID",
        )
    if body_pid != path_peer_id:
        raise FrameError(
            "unknown_peer",
            f"hello.peer_id {body_pid!r} does not match path {path_peer_id!r}",
        )
    caps = body.get("capabilities")
    if not isinstance(caps, dict):
        raise FrameError("bad_shape", "hello.capabilities must be a JSON object")
    return {"peer_id": body_pid, "capabilities": caps}


def _capabilities_equal(a: dict, b: dict) -> bool:
    """Per §5.2: hello.capabilities must match the announce capabilities.

    Compare on the four required keys; extra fields are ignored so future
    capability additions don't trip collision-detection on stale clients.
    """
    keys = ("expert_ids", "compute_mode", "dtype", "hidden")
    for k in keys:
        if a.get(k) != b.get(k):
            return False
    return True


async def _route_peer_signal_frame(
    coord: Coordinator, peer_id: str, parsed: dict[str, Any],
) -> Optional[FrameError]:
    """Stash a peer->browser ``signal`` frame in the addressed inbox.

    Per wire-frames.md §5.4: exactly one of ``sdp`` or ``candidate`` must
    be non-null; ``to`` is the browser nonce. Unknown ``to`` (no inbox)
    is silently dropped — the browser may have given up and the spec
    says this is not an error.
    """
    to = parsed.get("to")
    if not isinstance(to, str) or not to:
        return FrameError("bad_shape", "signal.to must be a non-empty string")
    sdp = parsed.get("sdp")
    cand = parsed.get("candidate")
    if (sdp is None) == (cand is None):
        return FrameError(
            "bad_shape",
            "exactly one of signal.sdp or signal.candidate must be non-null",
        )
    coord.signal_inbox.push(peer_id, to, parsed)
    return None


async def handle_peers_socket(request: web.Request) -> web.WebSocketResponse:
    """``WSS /peers/socket/<peer_id>``. See wire-frames.md §5.

    Lifecycle:

    1. Peer opens the WSS.
    2. First text frame MUST be ``type=hello`` with matching ``peer_id``;
       any mismatch closes 1002 with the appropriate error code.
    3. Coordinator replies ``ready`` and binds the live WS to the
       registry entry via ``attach_ws``.
    4. Peer pushes ``signal`` frames addressed to a browser nonce; the
       coordinator stashes them in ``SignalInbox`` for GET drainage.
    5. Coordinator pushes ``signal`` frames the other direction when
       browsers POST to ``/peers/signal/<peer_id>`` (see
       ``handle_peers_signal_post``).
    6. On close: ``detach_ws`` clears the pointer; the registry entry
       stays alive for the TTL window so a quick reconnect doesn't
       require a fresh announce.

    aiohttp's native ``heartbeat=PEER_SOCKET_HEARTBEAT_S`` handles
    control-frame pings; missed pongs close the socket automatically.
    """

    coord: Coordinator = request.app["coord"]
    peer_id = request.match_info.get("peer_id", "")

    # Reject obviously-malformed path early so we don't accept the WS just
    # to immediately close it (and so test clients without an Upgrade
    # header still see the structured ``unknown_peer`` error).
    if not _is_valid_uuid_v4(peer_id):
        return web.json_response(
            {"error": "unknown_peer",
             "message": "peer_id path param must be a lowercase v4 UUID"},
            status=400, headers=_cors_headers(),
        )

    upgrade = request.headers.get("Upgrade", "").lower()
    if "websocket" not in upgrade:
        return web.json_response(
            {"error": "expected_websocket_upgrade"},
            status=400, headers=_cors_headers(),
        )

    origin = request.headers.get("Origin", "<missing>")
    logger.info(
        "peer-socket upgrade peer=%s origin=%s remote=%s",
        peer_id, origin, request.remote,
    )

    ws = web.WebSocketResponse(
        heartbeat=PEER_SOCKET_HEARTBEAT_S,
        max_msg_size=PEER_SOCKET_FRAME_MAX_BYTES,
    )
    await ws.prepare(request)

    # ---- Phase 1: await hello -------------------------------------------
    try:
        first = await ws.receive(timeout=30.0)
    except asyncio.TimeoutError:
        coord.stats.record_error("expected_hello")
        await _send_error_frame(ws, "expected_hello", "no hello within 30s")
        await ws.close(code=1002)
        return ws

    if first.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
        return ws
    if first.type == WSMsgType.BINARY:
        coord.stats.record_error("expected_hello")
        await _send_error_frame(
            ws, "expected_hello", "first frame must be text hello",
        )
        await ws.close(code=1002)
        return ws
    if first.type != WSMsgType.TEXT:
        coord.stats.record_error("expected_hello")
        await _send_error_frame(
            ws, "expected_hello", "first frame must be text hello",
        )
        await ws.close(code=1002)
        return ws

    if len(first.data.encode("utf-8")) > PEER_SOCKET_FRAME_MAX_BYTES:
        coord.stats.record_error("frame_too_large")
        await _send_error_frame(
            ws, "frame_too_large",
            f"hello frame > {PEER_SOCKET_FRAME_MAX_BYTES // 1024} KB",
        )
        await ws.close(code=1002)
        return ws

    try:
        hello = json.loads(first.data)
    except json.JSONDecodeError as e:
        coord.stats.record_error("bad_json")
        await _send_error_frame(ws, "bad_json", f"hello JSON parse failed: {e}")
        await ws.close(code=1002)
        return ws

    try:
        normalized = _validate_peer_hello(hello, peer_id)
    except FrameError as e:
        coord.stats.record_error(e.code)
        await _send_error_frame(ws, e.code, e.message)
        await ws.close(code=1002)
        return ws

    entry = coord.peer_registry.get(peer_id)
    if entry is None:
        # Peer connected before announce. wire-frames.md §5.1 requires the
        # announce-first ordering; reject without leaking an entry.
        coord.stats.record_error("unknown_peer")
        await _send_error_frame(
            ws, "unknown_peer",
            "no registry entry for peer_id; POST /peers/announce first",
        )
        await ws.close(code=1002)
        return ws

    if not _capabilities_equal(normalized["capabilities"], entry.capabilities):
        # A second peer process is trying to hijack the id (or the user
        # restarted with different flags without re-announcing).
        coord.stats.record_error("peer_id_collision")
        await _send_error_frame(
            ws, "peer_id_collision",
            "hello.capabilities does not match the most recent announce",
        )
        await ws.close(code=1002)
        return ws

    if entry.ws is not None:
        # Another live WSS already holds this peer_id; refuse the new one.
        coord.stats.record_error("peer_id_collision")
        await _send_error_frame(
            ws, "peer_id_collision",
            "another /peers/socket connection is already live for this peer_id",
        )
        await ws.close(code=1002)
        return ws

    coord.peer_registry.attach_ws(peer_id, ws)
    await ws.send_str(json.dumps({
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "ping_interval_s": int(PEER_SOCKET_HEARTBEAT_S),
    }))

    # ---- Phase 2: signal-routing loop -----------------------------------
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if len(msg.data.encode("utf-8")) > PEER_SOCKET_FRAME_MAX_BYTES:
                    coord.stats.record_error("frame_too_large")
                    await _send_error_frame(
                        ws, "frame_too_large",
                        f"frame > {PEER_SOCKET_FRAME_MAX_BYTES // 1024} KB",
                    )
                    continue
                try:
                    parsed = json.loads(msg.data)
                except json.JSONDecodeError as e:
                    coord.stats.record_error("bad_json")
                    await _send_error_frame(
                        ws, "bad_json", f"JSON parse failed: {e}",
                    )
                    continue
                if not isinstance(parsed, dict):
                    coord.stats.record_error("bad_json")
                    await _send_error_frame(
                        ws, "bad_json", "frame must be a JSON object",
                    )
                    continue
                ftype = parsed.get("type")
                if ftype == "signal":
                    err = await _route_peer_signal_frame(coord, peer_id, parsed)
                    if err is not None:
                        coord.stats.record_error(err.code)
                        await _send_error_frame(ws, err.code, err.message)
                else:
                    coord.stats.record_error("unknown_type")
                    await _send_error_frame(
                        ws, "unknown_type",
                        f"unsupported frame type {ftype!r}",
                    )
            elif msg.type == WSMsgType.BINARY:
                coord.stats.record_error("unknown_type")
                await _send_error_frame(
                    ws, "unknown_type",
                    "binary frames not supported on /peers/socket",
                )
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
    finally:
        coord.peer_registry.detach_ws(peer_id)
    return ws


async def handle_peers_signal_post(request: web.Request) -> web.Response:
    """``POST /peers/signal/<peer_id>``. See wire-frames.md §1.10.3.

    Relays an SDP offer or ICE candidate from a browser to the peer's
    live WSS, rewrapped as ``{"type": "signal", ...body}``.

    Status codes:

    - **202** ``{"status": "relayed"}`` on successful forward.
    - **404** ``{"error": "unknown_peer"}`` if no registry entry.
    - **410** ``{"error": "peer_offline"}`` if no live WSS attached.
    - **400** on bad body shape (``bad_json``, ``bad_shape``).
    """
    coord: Coordinator = request.app["coord"]
    peer_id = request.match_info.get("peer_id", "")

    if not _is_valid_uuid_v4(peer_id):
        coord.stats.record_error("unknown_peer")
        return web.json_response(
            {"error": "unknown_peer",
             "message": "peer_id path param must be a lowercase v4 UUID"},
            status=404, headers=_cors_headers(),
        )

    try:
        body = await request.json()
    except json.JSONDecodeError:
        coord.stats.record_error("bad_json")
        return web.json_response(
            {"error": "bad_json", "message": "JSON parse failed"},
            status=400, headers=_cors_headers(),
        )
    if not isinstance(body, dict):
        coord.stats.record_error("bad_json")
        return web.json_response(
            {"error": "bad_json", "message": "body must be a JSON object"},
            status=400, headers=_cors_headers(),
        )

    src = body.get("from")
    if not isinstance(src, str) or not src.startswith("browser-"):
        coord.stats.record_error("bad_shape")
        return web.json_response(
            {"error": "bad_shape",
             "message": "from must be a string prefixed 'browser-'"},
            status=400, headers=_cors_headers(),
        )

    sdp = body.get("sdp")
    cand = body.get("candidate")
    if (sdp is None) == (cand is None):
        coord.stats.record_error("bad_shape")
        return web.json_response(
            {"error": "bad_shape",
             "message": "exactly one of sdp or candidate must be non-null"},
            status=400, headers=_cors_headers(),
        )

    entry = coord.peer_registry.get(peer_id)
    if entry is None:
        coord.stats.record_error("unknown_peer")
        return web.json_response(
            {"error": "unknown_peer",
             "message": f"no registry entry for peer_id {peer_id}"},
            status=404, headers=_cors_headers(),
        )
    if entry.ws is None:
        coord.stats.record_error("peer_offline")
        return web.json_response(
            {"error": "peer_offline",
             "message": "peer has no live /peers/socket connection"},
            status=410, headers=_cors_headers(),
        )

    frame = {
        "type": "signal",
        "from": src,
        "sdp": sdp,
        "candidate": cand,
    }
    try:
        await entry.ws.send_str(json.dumps(frame))
    except (ConnectionResetError, RuntimeError) as e:
        # Peer disconnected between the entry check and the send. Clear
        # the pointer and surface the same 410 the caller would have
        # seen if it had retried.
        logger.info("peer-socket send failed mid-relay peer=%s: %s", peer_id, e)
        coord.peer_registry.detach_ws(peer_id)
        coord.stats.record_error("peer_offline")
        return web.json_response(
            {"error": "peer_offline",
             "message": "peer socket closed mid-relay"},
            status=410, headers=_cors_headers(),
        )

    return web.json_response(
        {"status": "relayed"}, status=202, headers=_cors_headers(),
    )


async def handle_peers_signal_get(request: web.Request) -> web.Response:
    """``GET /peers/signal/<peer_id>?inbox=<nonce>``. See wire-frames.md §1.10.4.

    Drains and returns all non-expired peer->browser ``signal`` frames
    that the peer addressed to ``nonce``. Returns ``{"signals": [...]}``;
    second GET with the same nonce returns ``{"signals": []}`` until
    fresh frames arrive (inbox is drained on read).

    - **200** with signals on success (possibly empty).
    - **404** ``{"error": "unknown_peer"}`` if no registry entry for ``peer_id``.
    - **400** ``{"error": "bad_shape"}`` if ``inbox`` query param missing.
    """
    coord: Coordinator = request.app["coord"]
    peer_id = request.match_info.get("peer_id", "")

    if not _is_valid_uuid_v4(peer_id):
        coord.stats.record_error("unknown_peer")
        return web.json_response(
            {"error": "unknown_peer",
             "message": "peer_id path param must be a lowercase v4 UUID"},
            status=404, headers=_cors_headers(),
        )

    nonce = request.query.get("inbox", "")
    if not nonce:
        coord.stats.record_error("bad_shape")
        return web.json_response(
            {"error": "bad_shape", "message": "inbox query param is required"},
            status=400, headers=_cors_headers(),
        )

    if coord.peer_registry.get(peer_id) is None:
        coord.stats.record_error("unknown_peer")
        return web.json_response(
            {"error": "unknown_peer",
             "message": f"no registry entry for peer_id {peer_id}"},
            status=404, headers=_cors_headers(),
        )

    signals = coord.signal_inbox.drain(peer_id, nonce)
    return web.json_response(
        {"signals": signals}, headers=_cors_headers(),
    )


# ---------------------------------------------------------------------------
# /ws (Wave-3 surface, kept for lm_head and dispatch_moved soft-reject)
# ---------------------------------------------------------------------------


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

    # Wave-4 desktop-peer: peer signaling surface (see wire-frames.md §1.10).
    app.router.add_post("/peers/announce", handle_peers_announce)
    app.router.add_options("/peers/announce", handle_options)
    app.router.add_get("/peers/list", handle_peers_list)
    app.router.add_options("/peers/list", handle_options)
    app.router.add_get("/peers/socket/{peer_id}", handle_peers_socket)
    app.router.add_post("/peers/signal/{peer_id}", handle_peers_signal_post)
    app.router.add_get("/peers/signal/{peer_id}", handle_peers_signal_get)
    app.router.add_options("/peers/signal/{peer_id}", handle_options)

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
