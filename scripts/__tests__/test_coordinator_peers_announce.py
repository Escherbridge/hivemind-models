"""Tests for ``POST /peers/announce`` and the in-memory peer registry.

Pins the wire-frames.md §1.10.1 contract:

- Happy-path announce returns ``{"status": "announced", "peer_id": ...,
  "expires_at_s": <float>, "ttl_s": <int>}``.
- Re-announce extends the TTL.
- Malformed ``peer_id`` (non-UUID) returns ``bad_json``.
- Missing required capability fields returns ``bad_shape``.
- Idempotent on ``peer_id``: same id, second announce updates capabilities.

The registry sweeper task is exercised in ``test_coordinator_peers_list.py``
(Task 1.4) where expired-entry filtering is observable.
"""

from __future__ import annotations

import time
import uuid

import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.expert_coordinator import (
    CoordinatorConfig,
    PeerEntry,
    PeerRegistry,
    build_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wave4_config() -> CoordinatorConfig:
    """A regions-empty config — the Wave-4 signaling-only default."""
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


def _valid_announce_body(peer_id: str | None = None) -> dict:
    return {
        "peer_id": peer_id or str(uuid.uuid4()),
        "capabilities": _valid_capabilities(),
    }


# ---------------------------------------------------------------------------
# PeerEntry / PeerRegistry direct unit tests
# ---------------------------------------------------------------------------


def test_registry_register_creates_entry() -> None:
    reg = PeerRegistry(ttl_s=90)
    pid = str(uuid.uuid4())
    entry = reg.register(pid, _valid_capabilities())
    assert isinstance(entry, PeerEntry)
    assert entry.peer_id == pid
    assert entry.capabilities == _valid_capabilities()
    assert entry.expires_at_s > time.time()
    assert reg.get(pid) is entry


def test_registry_register_extends_ttl_on_reannounce() -> None:
    reg = PeerRegistry(ttl_s=90)
    pid = str(uuid.uuid4())
    e1 = reg.register(pid, _valid_capabilities())
    first_expiry = e1.expires_at_s

    time.sleep(0.05)  # advance monotonic + wall clock a hair
    caps2 = {**_valid_capabilities(), "compute_mode": "echo"}
    e2 = reg.register(pid, caps2)
    assert e2 is e1  # same entry object
    assert e2.expires_at_s > first_expiry
    assert e2.capabilities["compute_mode"] == "echo"


def test_registry_sweep_drops_expired() -> None:
    reg = PeerRegistry(ttl_s=0.05)  # 50 ms TTL
    pid = str(uuid.uuid4())
    reg.register(pid, _valid_capabilities())
    assert reg.get(pid) is not None
    time.sleep(0.1)
    dropped = reg.sweep_expired()
    assert pid in dropped
    assert reg.get(pid) is None


def test_registry_list_active_filters_expired_and_sorts_by_recency() -> None:
    reg = PeerRegistry(ttl_s=90)
    p1 = str(uuid.uuid4())
    p2 = str(uuid.uuid4())
    p3 = str(uuid.uuid4())
    reg.register(p1, _valid_capabilities())
    time.sleep(0.01)
    reg.register(p2, _valid_capabilities())
    time.sleep(0.01)
    reg.register(p3, _valid_capabilities())

    active = reg.list_active(now=time.time())
    # Sorted by most-recently-seen first
    ids = [p["peer_id"] for p in active]
    assert ids == [p3, p2, p1]
    # All last_seen_ms_ago are non-negative ints
    for p in active:
        assert isinstance(p["last_seen_ms_ago"], int)
        assert p["last_seen_ms_ago"] >= 0


def test_registry_list_active_caps_at_64() -> None:
    reg = PeerRegistry(ttl_s=90)
    for _ in range(80):
        reg.register(str(uuid.uuid4()), _valid_capabilities())
    active = reg.list_active(now=time.time())
    assert len(active) == 64


# ---------------------------------------------------------------------------
# POST /peers/announce HTTP surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_announce_happy_path_returns_expected_shape() -> None:
    app = await build_app(_wave4_config())
    body = _valid_announce_body()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.status == 200, await r.text()
        b = await r.json()
        assert b["status"] == "announced"
        assert b["peer_id"] == body["peer_id"]
        assert isinstance(b["expires_at_s"], float)
        assert b["expires_at_s"] > time.time()
        assert b["ttl_s"] == 90


@pytest.mark.asyncio
async def test_announce_is_idempotent_extends_ttl() -> None:
    app = await build_app(_wave4_config())
    body = _valid_announce_body()
    async with TestClient(TestServer(app)) as client:
        r1 = await client.post("/peers/announce", json=body)
        b1 = await r1.json()
        first_exp = b1["expires_at_s"]

        time.sleep(0.05)
        r2 = await client.post("/peers/announce", json=body)
        assert r2.status == 200
        b2 = await r2.json()
        assert b2["peer_id"] == body["peer_id"]
        assert b2["expires_at_s"] > first_exp


@pytest.mark.asyncio
async def test_announce_updates_capabilities_on_reannounce() -> None:
    app = await build_app(_wave4_config())
    pid = str(uuid.uuid4())
    body1 = {"peer_id": pid, "capabilities": _valid_capabilities()}
    body2 = {
        "peer_id": pid,
        "capabilities": {**_valid_capabilities(), "compute_mode": "echo"},
    }
    async with TestClient(TestServer(app)) as client:
        await client.post("/peers/announce", json=body1)
        r2 = await client.post("/peers/announce", json=body2)
        assert r2.status == 200
        # The registry has the new capabilities
        coord = app["coord"]
        entry = coord.peer_registry.get(pid)
        assert entry is not None
        assert entry.capabilities["compute_mode"] == "echo"


@pytest.mark.asyncio
async def test_announce_rejects_non_uuid_peer_id() -> None:
    app = await build_app(_wave4_config())
    body = _valid_announce_body(peer_id="not-a-uuid")
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_json"


@pytest.mark.asyncio
async def test_announce_rejects_missing_capabilities() -> None:
    app = await build_app(_wave4_config())
    body = {"peer_id": str(uuid.uuid4())}  # no capabilities key
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_announce_rejects_empty_expert_ids() -> None:
    app = await build_app(_wave4_config())
    caps = {**_valid_capabilities(), "expert_ids": []}
    body = {"peer_id": str(uuid.uuid4()), "capabilities": caps}
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_announce_rejects_expert_id_out_of_range() -> None:
    app = await build_app(_wave4_config())
    caps = {**_valid_capabilities(), "expert_ids": [0, 1, 64]}  # 64 is out
    body = {"peer_id": str(uuid.uuid4()), "capabilities": caps}
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_expert_id"


@pytest.mark.asyncio
async def test_announce_rejects_bad_compute_mode() -> None:
    app = await build_app(_wave4_config())
    caps = {**_valid_capabilities(), "compute_mode": "neither"}
    body = {"peer_id": str(uuid.uuid4()), "capabilities": caps}
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_announce_rejects_wrong_hidden_dim() -> None:
    app = await build_app(_wave4_config())
    caps = {**_valid_capabilities(), "hidden": 2048}
    body = {"peer_id": str(uuid.uuid4()), "capabilities": caps}
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_announce_rejects_malformed_json() -> None:
    app = await build_app(_wave4_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.post(
            "/peers/announce",
            data=b"this is not json{",
            headers={"Content-Type": "application/json"},
        )
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_json"


@pytest.mark.asyncio
async def test_announce_cors_headers() -> None:
    app = await build_app(_wave4_config())
    body = _valid_announce_body()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/announce", json=body)
        assert r.headers.get("Access-Control-Allow-Origin") == "*"
