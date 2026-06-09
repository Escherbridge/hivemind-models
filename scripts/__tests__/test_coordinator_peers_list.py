"""Tests for ``GET /peers/list``. Pins the wire-frames.md §1.10.2 contract.

The list endpoint returns ONLY peers that:

1. Have a future ``expires_at_s`` (registry sweeper hasn't dropped them).
2. Have a live ``/peers/socket/<peer_id>`` WSS attached (so the browser
   only learns about peers it can actually signal to).

In these tests we don't yet exercise the real WSS (that's Task 1.5).
We use ``coord.peer_registry.attach_ws(peer_id, <sentinel>)`` to mark
an entry as having a live socket. A non-None sentinel object is enough
for the ``list_active(require_ws=True)`` filter.
"""

from __future__ import annotations

import time
import uuid

import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.expert_coordinator import CoordinatorConfig, build_app


def _wave4_config() -> CoordinatorConfig:
    cfg = CoordinatorConfig()
    cfg.regions = []
    return cfg


def _valid_capabilities() -> dict:
    return {
        "expert_ids": list(range(0, 64)),
        "compute_mode": "real",
        "dtype": "fp16",
        "hidden": 1536,
    }


# Sentinel "fake WSS" object for attach_ws. Just needs to be non-None.
class _FakeWS:
    pass


@pytest.mark.asyncio
async def test_list_empty_when_no_peers_registered() -> None:
    app = await build_app(_wave4_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/list")
        assert r.status == 200
        b = await r.json()
        assert b == {"peers": []}


@pytest.mark.asyncio
async def test_list_omits_peers_without_live_ws() -> None:
    """A peer that announced but hasn't opened its WSS yet must NOT appear.
    The browser can't signal to a peer with no inbound socket."""
    app = await build_app(_wave4_config())
    coord = app["coord"]
    coord.peer_registry.register(str(uuid.uuid4()), _valid_capabilities())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/list")
        b = await r.json()
        assert b == {"peers": []}


@pytest.mark.asyncio
async def test_list_returns_peer_with_attached_ws() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    coord.peer_registry.attach_ws(pid, _FakeWS())

    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/list")
        b = await r.json()
        assert len(b["peers"]) == 1
        entry = b["peers"][0]
        assert entry["peer_id"] == pid
        assert entry["capabilities"] == _valid_capabilities()
        assert isinstance(entry["last_seen_ms_ago"], int)
        assert entry["last_seen_ms_ago"] >= 0


@pytest.mark.asyncio
async def test_list_sorted_by_recency_most_recent_first() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    p1, p2, p3 = (str(uuid.uuid4()) for _ in range(3))

    coord.peer_registry.register(p1, _valid_capabilities())
    coord.peer_registry.attach_ws(p1, _FakeWS())
    time.sleep(0.02)
    coord.peer_registry.register(p2, _valid_capabilities())
    coord.peer_registry.attach_ws(p2, _FakeWS())
    time.sleep(0.02)
    coord.peer_registry.register(p3, _valid_capabilities())
    coord.peer_registry.attach_ws(p3, _FakeWS())

    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/list")
        b = await r.json()
        ids = [p["peer_id"] for p in b["peers"]]
        assert ids == [p3, p2, p1]


@pytest.mark.asyncio
async def test_list_capped_at_64_entries() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    for _ in range(80):
        pid = str(uuid.uuid4())
        coord.peer_registry.register(pid, _valid_capabilities())
        coord.peer_registry.attach_ws(pid, _FakeWS())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/list")
        b = await r.json()
        assert len(b["peers"]) == 64


@pytest.mark.asyncio
async def test_list_filters_expired_entries() -> None:
    """Expired entries don't appear even if a sweeper hasn't run yet —
    ``list_active`` filters by expires_at_s on every read."""
    from scripts.expert_coordinator import PeerRegistry

    app = await build_app(_wave4_config())
    coord = app["coord"]
    # Replace the registry with one that has a 50 ms TTL so we can expire
    # entries within the test without sleeping forever.
    short_reg = PeerRegistry(ttl_s=0.05)
    coord.peer_registry = short_reg

    pid = str(uuid.uuid4())
    short_reg.register(pid, _valid_capabilities())
    short_reg.attach_ws(pid, _FakeWS())

    # Immediately listing returns the peer.
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/list")
        b = await r.json()
        assert len(b["peers"]) == 1

        # Wait for the entry to expire; list reflects the filter on the next
        # request without the sweeper having run.
        time.sleep(0.1)
        r2 = await client.get("/peers/list")
        b2 = await r2.json()
        assert b2 == {"peers": []}


@pytest.mark.asyncio
async def test_list_cors_headers() -> None:
    app = await build_app(_wave4_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/list")
        assert r.headers.get("Access-Control-Allow-Origin") == "*"
