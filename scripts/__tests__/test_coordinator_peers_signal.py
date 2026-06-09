"""Tests for ``/peers/signal/<peer_id>``. Pins wire-frames.md §1.10.3-4.

Covers:

- POST happy-path: browser offer is relayed verbatim (rewrapped) to
  the peer's live WSS, returns 202 ``{"status": "relayed"}``.
- POST when no registry entry returns **404 unknown_peer**.
- POST when registry entry exists but no live WSS returns
  **410 peer_offline**.
- POST with malformed JSON / shape returns 400 ``bad_json`` / ``bad_shape``.
- POST with both or neither of ``sdp`` / ``candidate`` returns
  ``bad_shape``.
- POST with missing or unprefixed ``from`` returns ``bad_shape``.
- GET drains the inbox: first GET returns stashed signals, second GET
  with the same nonce returns ``{"signals": []}``.
- GET without ``inbox`` query param returns ``bad_shape``.
- GET with no registry entry returns 404.
- Inbox TTL expires entries after ``PEER_SIGNAL_INBOX_TTL_S``.
- CORS header is present on both surfaces.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.expert_coordinator import (
    CoordinatorConfig,
    SignalInbox,
    build_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _hello(peer_id: str) -> dict:
    return {
        "type": "hello",
        "peer_id": peer_id,
        "capabilities": _valid_capabilities(),
    }


def _offer_body(nonce: str | None = None) -> dict:
    return {
        "from": nonce or ("browser-" + str(uuid.uuid4())),
        "sdp": {"type": "offer", "sdp": "v=0\r\noffer-body"},
        "candidate": None,
    }


# ---------------------------------------------------------------------------
# SignalInbox unit tests
# ---------------------------------------------------------------------------


def test_inbox_push_drain_roundtrip() -> None:
    inbox = SignalInbox(ttl_s=10.0)
    pid, nonce = "p1", "n1"
    inbox.push(pid, nonce, {"to": nonce, "sdp": {"type": "answer", "sdp": "v"}})
    inbox.push(pid, nonce, {"to": nonce, "candidate": {"candidate": "x"}})
    drained = inbox.drain(pid, nonce)
    assert len(drained) == 2
    # Drained: second drain is empty.
    assert inbox.drain(pid, nonce) == []


def test_inbox_ttl_drops_old_entries() -> None:
    inbox = SignalInbox(ttl_s=0.05)
    inbox.push("p", "n", {"to": "n", "sdp": {"type": "answer", "sdp": "v"}})
    time.sleep(0.1)
    assert inbox.drain("p", "n") == []


def test_inbox_isolates_by_peer_and_nonce() -> None:
    inbox = SignalInbox(ttl_s=10.0)
    inbox.push("p1", "n1", {"to": "n1", "sdp": {"type": "answer", "sdp": "a"}})
    inbox.push("p1", "n2", {"to": "n2", "sdp": {"type": "answer", "sdp": "b"}})
    inbox.push("p2", "n1", {"to": "n1", "sdp": {"type": "answer", "sdp": "c"}})
    assert len(inbox.drain("p1", "n1")) == 1
    assert len(inbox.drain("p1", "n2")) == 1
    assert len(inbox.drain("p2", "n1")) == 1
    assert inbox.drain("p1", "n1") == []


# ---------------------------------------------------------------------------
# POST /peers/signal/<peer_id>
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_signal_relays_to_live_peer_socket() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid)))
            await ws.receive_json()  # ready

            body = _offer_body()
            r = await client.post(f"/peers/signal/{pid}", json=body)
            assert r.status == 202, await r.text()
            j = await r.json()
            assert j == {"status": "relayed"}

            # The peer's ws now receives the rewrapped frame.
            frame = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
            assert frame["type"] == "signal"
            assert frame["from"] == body["from"]
            assert frame["sdp"] == body["sdp"]
            assert frame["candidate"] is None


@pytest.mark.asyncio
async def test_post_signal_unknown_peer_returns_404() -> None:
    app = await build_app(_wave4_config())
    pid = str(uuid.uuid4())
    async with TestClient(TestServer(app)) as client:
        r = await client.post(f"/peers/signal/{pid}", json=_offer_body())
        assert r.status == 404
        b = await r.json()
        assert b["error"] == "unknown_peer"


@pytest.mark.asyncio
async def test_post_signal_offline_peer_returns_410() -> None:
    """Peer announced but its /peers/socket WSS isn't connected."""
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    # No attach_ws; the registry entry has ws=None.

    async with TestClient(TestServer(app)) as client:
        r = await client.post(f"/peers/signal/{pid}", json=_offer_body())
        assert r.status == 410
        b = await r.json()
        assert b["error"] == "peer_offline"


@pytest.mark.asyncio
async def test_post_signal_bad_path_uuid_returns_404() -> None:
    app = await build_app(_wave4_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/peers/signal/not-a-uuid", json=_offer_body())
        assert r.status == 404
        b = await r.json()
        assert b["error"] == "unknown_peer"


@pytest.mark.asyncio
async def test_post_signal_malformed_json_returns_bad_json() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    async with TestClient(TestServer(app)) as client:
        r = await client.post(
            f"/peers/signal/{pid}",
            data=b"not json{",
            headers={"Content-Type": "application/json"},
        )
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_json"


@pytest.mark.asyncio
async def test_post_signal_both_sdp_and_candidate_rejected() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    body = {
        "from": "browser-" + str(uuid.uuid4()),
        "sdp": {"type": "offer", "sdp": "v=0\r\n"},
        "candidate": {"candidate": "candidate:0 1 UDP 0 1.2.3.4 5 typ host",
                      "sdpMid": "0", "sdpMLineIndex": 0},
    }
    async with TestClient(TestServer(app)) as client:
        r = await client.post(f"/peers/signal/{pid}", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_post_signal_neither_sdp_nor_candidate_rejected() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    body = {
        "from": "browser-" + str(uuid.uuid4()),
        "sdp": None,
        "candidate": None,
    }
    async with TestClient(TestServer(app)) as client:
        r = await client.post(f"/peers/signal/{pid}", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_post_signal_unprefixed_from_rejected() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    body = {**_offer_body(), "from": "not-a-browser-nonce"}
    async with TestClient(TestServer(app)) as client:
        r = await client.post(f"/peers/signal/{pid}", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


# ---------------------------------------------------------------------------
# GET /peers/signal/<peer_id>?inbox=<nonce>
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_signal_drains_inbox_then_returns_empty() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    nonce = "browser-" + str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    # Pre-seed inbox as if the peer had pushed an answer.
    coord.signal_inbox.push(pid, nonce, {
        "to": nonce, "sdp": {"type": "answer", "sdp": "v=0\r\n"},
        "candidate": None,
    })

    async with TestClient(TestServer(app)) as client:
        r1 = await client.get(f"/peers/signal/{pid}?inbox={nonce}")
        assert r1.status == 200
        b1 = await r1.json()
        assert len(b1["signals"]) == 1
        assert b1["signals"][0]["to"] == nonce

        # Second GET drains to empty.
        r2 = await client.get(f"/peers/signal/{pid}?inbox={nonce}")
        assert r2.status == 200
        b2 = await r2.json()
        assert b2 == {"signals": []}


@pytest.mark.asyncio
async def test_get_signal_missing_inbox_param_returns_bad_shape() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    async with TestClient(TestServer(app)) as client:
        r = await client.get(f"/peers/signal/{pid}")
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_get_signal_unknown_peer_returns_404() -> None:
    app = await build_app(_wave4_config())
    pid = str(uuid.uuid4())
    async with TestClient(TestServer(app)) as client:
        r = await client.get(f"/peers/signal/{pid}?inbox=browser-x")
        assert r.status == 404
        b = await r.json()
        assert b["error"] == "unknown_peer"


@pytest.mark.asyncio
async def test_get_signal_inbox_ttl_expires_entries() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    nonce = "browser-" + str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    # Swap in a tight-TTL inbox so we don't sleep 10s.
    coord.signal_inbox = SignalInbox(ttl_s=0.05)
    coord.signal_inbox.push(pid, nonce, {
        "to": nonce, "sdp": {"type": "answer", "sdp": "v"}, "candidate": None,
    })

    async with TestClient(TestServer(app)) as client:
        time.sleep(0.1)  # let entries TTL out
        r = await client.get(f"/peers/signal/{pid}?inbox={nonce}")
        assert r.status == 200
        b = await r.json()
        assert b == {"signals": []}


# ---------------------------------------------------------------------------
# Round-trip end-to-end (POST → WSS → ack from peer → GET drains)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_roundtrip_browser_offer_to_peer_answer() -> None:
    """Mirror of wire-frames.md §1.10.5: browser POSTs offer, peer answers,
    browser polls and gets the answer."""
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    nonce = "browser-" + str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as peer_ws:
            await peer_ws.send_str(json.dumps(_hello(pid)))
            await peer_ws.receive_json()  # ready

            # Browser → peer (offer).
            r = await client.post(f"/peers/signal/{pid}", json={
                "from": nonce,
                "sdp": {"type": "offer", "sdp": "v=0\r\noffer"},
                "candidate": None,
            })
            assert r.status == 202

            relayed = await asyncio.wait_for(peer_ws.receive_json(), timeout=2.0)
            assert relayed["type"] == "signal"
            assert relayed["from"] == nonce
            assert relayed["sdp"]["type"] == "offer"

            # Peer → browser (answer) over the WSS.
            await peer_ws.send_str(json.dumps({
                "type": "signal",
                "to": nonce,
                "sdp": {"type": "answer", "sdp": "v=0\r\nanswer"},
                "candidate": None,
            }))
            await asyncio.sleep(0.05)

            # Browser drains its inbox.
            poll = await client.get(f"/peers/signal/{pid}?inbox={nonce}")
            assert poll.status == 200
            j = await poll.json()
            assert len(j["signals"]) == 1
            assert j["signals"][0]["sdp"]["type"] == "answer"


@pytest.mark.asyncio
async def test_post_signal_cors_header() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    async with TestClient(TestServer(app)) as client:
        # Use the offline path so we don't need a peer WSS — CORS still
        # has to be on the error responses too.
        r = await client.post(f"/peers/signal/{pid}", json=_offer_body())
        assert r.headers.get("Access-Control-Allow-Origin") == "*"


@pytest.mark.asyncio
async def test_get_signal_cors_header() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    coord.peer_registry.register(pid, _valid_capabilities())
    async with TestClient(TestServer(app)) as client:
        r = await client.get(f"/peers/signal/{pid}?inbox=browser-x")
        assert r.headers.get("Access-Control-Allow-Origin") == "*"
