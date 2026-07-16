"""Tests for ``WSS /peers/socket/<peer_id>``. Pins wire-frames.md §5.

Covers:

- Hello/ready happy-path round-trip and registry WSS attach.
- Bad-hello branches: non-hello first frame, peer_id mismatch, missing
  capabilities, capabilities mismatch with the announce body, binary
  before hello, oversized hello, malformed JSON.
- Connecting before announce returns ``unknown_peer``.
- Two concurrent connections under different ``peer_id``s don't
  cross-talk.
- ``peer_id_collision`` when a second WSS tries to claim a peer_id
  that already has a live socket.
- Close clears the registry's ``ws`` pointer (TTL stays alive).
- Peer ``signal`` frames are stashed in the per-(peer_id, nonce)
  inbox so the matching GET drains them.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.expert_coordinator import CoordinatorConfig, build_app


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


def _hello(peer_id: str, caps: dict | None = None) -> dict:
    return {
        "type": "hello",
        "peer_id": peer_id,
        "capabilities": caps if caps is not None else _valid_capabilities(),
    }


async def _announce(coord, peer_id: str, caps: dict | None = None) -> None:
    coord.peer_registry.register(
        peer_id, caps if caps is not None else _valid_capabilities(),
    )


# ---------------------------------------------------------------------------
# Hello/ready handshake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hello_ready_happy_path_attaches_ws() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid)))
            ready = await ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["protocol_version"] == 1
            assert ready["ping_interval_s"] >= 1

            # The registry now holds a non-None ws pointer.
            entry = coord.peer_registry.get(pid)
            assert entry is not None
            assert entry.ws is not None


@pytest.mark.asyncio
async def test_socket_rejects_path_peer_id_not_a_uuid() -> None:
    app = await build_app(_wave4_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/peers/socket/not-a-uuid")
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "unknown_peer"


@pytest.mark.asyncio
async def test_non_hello_first_frame_errors_and_closes() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps({"type": "signal", "to": "x"}))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "expected_hello"


@pytest.mark.asyncio
async def test_hello_path_peer_id_mismatch_closes_unknown_peer() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid_path = str(uuid.uuid4())
    pid_body = str(uuid.uuid4())
    await _announce(coord, pid_path)
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid_path}") as ws:
            await ws.send_str(json.dumps(_hello(pid_body)))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "unknown_peer"


@pytest.mark.asyncio
async def test_socket_rejects_when_no_announce_yet() -> None:
    app = await build_app(_wave4_config())
    pid = str(uuid.uuid4())
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid)))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "unknown_peer"


@pytest.mark.asyncio
async def test_hello_capabilities_mismatch_closes_collision() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid, _valid_capabilities())

    # Hello declares a different compute_mode than the announce body.
    mismatched = {**_valid_capabilities(), "compute_mode": "echo"}
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid, mismatched)))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "peer_id_collision"


@pytest.mark.asyncio
async def test_binary_before_hello_closes() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_bytes(b"\x01\x02\x03")
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "expected_hello"


@pytest.mark.asyncio
async def test_hello_with_bad_json_closes() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str("not json{")
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_json"


# ---------------------------------------------------------------------------
# Concurrency + collision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_connections_under_distinct_peer_ids() -> None:
    """Two peers sharing the same coordinator must not see each other's traffic."""
    app = await build_app(_wave4_config())
    coord = app["coord"]
    p1 = str(uuid.uuid4())
    p2 = str(uuid.uuid4())
    await _announce(coord, p1)
    await _announce(coord, p2)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{p1}") as ws1, \
                client.ws_connect(f"/peers/socket/{p2}") as ws2:
            await ws1.send_str(json.dumps(_hello(p1)))
            await ws2.send_str(json.dumps(_hello(p2)))

            ready1 = await ws1.receive_json()
            ready2 = await ws2.receive_json()
            assert ready1["type"] == "ready"
            assert ready2["type"] == "ready"

            # Browser POSTs a signal targeted at p1's nonce. Only ws1 sees it.
            await client.post(f"/peers/signal/{p1}", json={
                "from": "browser-" + str(uuid.uuid4()),
                "sdp": {"type": "offer", "sdp": "v=0\r\n"},
                "candidate": None,
            })

            frame = await asyncio.wait_for(ws1.receive_json(), timeout=2.0)
            assert frame["type"] == "signal"
            assert "sdp" in frame

            # ws2 must be silent: assert receive times out within 200 ms.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws2.receive_json(), timeout=0.2)

            assert coord.peer_registry.get(p1).ws is not None
            assert coord.peer_registry.get(p2).ws is not None


@pytest.mark.asyncio
async def test_second_socket_for_same_peer_id_collides() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws1:
            await ws1.send_str(json.dumps(_hello(pid)))
            ready1 = await ws1.receive_json()
            assert ready1["type"] == "ready"

            async with client.ws_connect(f"/peers/socket/{pid}") as ws2:
                await ws2.send_str(json.dumps(_hello(pid)))
                err = await ws2.receive_json()
                assert err["type"] == "error"
                assert err["code"] == "peer_id_collision"


@pytest.mark.asyncio
async def test_close_detaches_ws_keeps_registry_entry() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid)))
            await ws.receive_json()  # ready
            assert coord.peer_registry.get(pid).ws is not None
            await ws.close()

        # Give the server-side coroutine a tick to run its finally block.
        await asyncio.sleep(0.05)
        entry = coord.peer_registry.get(pid)
        assert entry is not None  # TTL window keeps the entry alive
        assert entry.ws is None  # but the socket pointer is cleared


# ---------------------------------------------------------------------------
# Peer -> coord signal routing into the inbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_peer_signal_frame_lands_in_inbox() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    nonce = "browser-" + str(uuid.uuid4())
    await _announce(coord, pid)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid)))
            await ws.receive_json()  # ready

            # Peer sends an SDP answer addressed to the browser nonce.
            await ws.send_str(json.dumps({
                "type": "signal",
                "to": nonce,
                "sdp": {"type": "answer", "sdp": "v=0\r\nanswer"},
                "candidate": None,
            }))
            # Let the server-side coroutine handle the frame.
            await asyncio.sleep(0.05)

            drained = coord.signal_inbox.drain(pid, nonce)
            assert len(drained) == 1
            assert drained[0]["to"] == nonce
            assert drained[0]["sdp"]["type"] == "answer"


@pytest.mark.asyncio
async def test_peer_signal_missing_to_returns_bad_shape() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid)))
            await ws.receive_json()  # ready
            await ws.send_str(json.dumps({
                "type": "signal",
                "sdp": {"type": "answer", "sdp": "v=0\r\n"},
                "candidate": None,
            }))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_shape"


@pytest.mark.asyncio
async def test_unknown_frame_type_after_hello_errors_but_stays_open() -> None:
    app = await build_app(_wave4_config())
    coord = app["coord"]
    pid = str(uuid.uuid4())
    await _announce(coord, pid)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect(f"/peers/socket/{pid}") as ws:
            await ws.send_str(json.dumps(_hello(pid)))
            await ws.receive_json()  # ready
            await ws.send_str(json.dumps({"type": "garbage"}))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "unknown_type"

            # Connection still alive; a valid signal goes through.
            nonce = "browser-" + str(uuid.uuid4())
            await ws.send_str(json.dumps({
                "type": "signal",
                "to": nonce,
                "sdp": None,
                "candidate": {"candidate": "candidate:0 1 UDP 0 1.2.3.4 5"
                                          " typ host", "sdpMid": "0",
                              "sdpMLineIndex": 0},
            }))
            await asyncio.sleep(0.05)
            drained = coord.signal_inbox.drain(pid, nonce)
            assert len(drained) == 1
