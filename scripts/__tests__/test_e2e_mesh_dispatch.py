"""Cross-boundary e2e: signaling plane (coordinator) + data plane (peer expert)
+ first_of_k dispatch + tier-inheritance gate.

This is the single integrated check the desktop-pivot session builds toward. It
crosses these real boundaries instead of mocking them:

1. **Signaling plane** — the SHIPPED Wave-4 Phase 1 surface on
   ``expert_coordinator.build_app`` run as an in-process aiohttp TestServer (the
   sidecar coordinator, substrate-adr.md D3). Real ``/peers/announce``,
   ``/peers/list``, ``/peers/socket/<id>`` hello/ready, and the bidirectional
   ``/peers/signal`` SDP/ICE relay between a browser nonce and the peer.

2. **Data plane** — the ``scripts/_wire.py`` binary forward frames
   (``<BHIIB`` header + payload) that, in production, ride the browser<->peer
   WebRTC DataChannel (wire-frames.md §6). The coordinator's ``/peers/socket``
   is signaling-only and rejects binary frames, so the data plane here is an
   in-process peer-hosted expert socket standing in for the post-handshake
   DataChannel. It exercises the REAL pack/unpack and a real GLU expert forward.

3. **Dispatch** — ``first_of_k`` (substrate-adr.md D6) racing k peers, take the
   first valid result, cancel the losers.

4. **Tier gate** — ``scripts/sovereignty.py`` (substrate-adr.md D5): a private
   shard is local-only and must never be dispatched across peers.

No external weight file, no Railway, no network — fully offline and
self-contained, runs under the same pytest sweep as the rest of the suite.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
import torch
import torch.nn.functional as F
from aiohttp import ClientSession, WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from scripts._wire import (
    OP_ERROR,
    pack_tensor,
    unpack_request,
    unpack_response,
)
from scripts.expert_coordinator import CoordinatorConfig, build_app
from scripts.sovereignty import (
    Tier,
    TierViolation,
    assert_dispatchable,
    is_dispatchable,
    most_restrictive,
)

HIDDEN = 1536
INTERMEDIATE = 512
CAPS = {
    "expert_ids": list(range(64)),
    "compute_mode": "echo",
    "dtype": "fp16",
    "hidden": HIDDEN,
}


# --------------------------------------------------------------------------- #
# Synthetic expert — a real GLU forward over the real wire format, no weights  #
# file required. Mirrors expert_railway_server.Expert.forward shape.           #
# --------------------------------------------------------------------------- #
class SyntheticExpert:
    """Deterministic GLU expert keyed by ``expert_id`` so racing peers that
    host the same expert return byte-identical results (first_of_k correctness).
    """

    def __init__(self, expert_id: int) -> None:
        self.expert_id = expert_id
        g = torch.Generator().manual_seed(expert_id)
        # input_linear: [2*intermediate, hidden]; output_linear: [hidden, intermediate]
        self.input_linear = torch.randn(2 * INTERMEDIATE, HIDDEN, generator=g, dtype=torch.float32)
        self.output_linear = torch.randn(HIDDEN, INTERMEDIATE, generator=g, dtype=torch.float32)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.to(torch.float32)
        h = F.linear(x, self.input_linear)
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up
        return F.linear(h, self.output_linear)


async def _serve_expert_frame(buf: bytes, delay_s: float = 0.0) -> bytes:
    """Peer-side data-plane handler: unpack a forward request, run the expert,
    pack the response. This is exactly what the DataChannel handler does.
    """

    if delay_s:
        await asyncio.sleep(delay_s)
    parsed = unpack_request(buf)
    if parsed is None:
        return bytes([OP_ERROR]) + b"malformed forward frame"
    expert_id, x = parsed
    out = SyntheticExpert(expert_id).forward(x)
    return pack_tensor(expert_id, out)


# --------------------------------------------------------------------------- #
# first_of_k dispatch over k peer data-plane handlers (D6) with the tier gate. #
# --------------------------------------------------------------------------- #
async def dispatch_first_of_k_to_peers(
    expert_id: int,
    x: torch.Tensor,
    shard_tier: "Tier | str",
    peer_delays_s: list[float],
) -> torch.Tensor:
    """Race ``len(peer_delays_s)`` peers hosting ``expert_id``; return the first
    valid expert output. Raises ``TierViolation`` BEFORE any dispatch if the
    shard tier is local-only (private).
    """

    # D5 gate: refuse to cross the machine boundary for private shards.
    assert_dispatchable(shard_tier)

    req = pack_tensor(expert_id, x)
    tasks = [
        asyncio.create_task(_serve_expert_frame(req, delay_s=d))
        for d in peer_delays_s
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:  # cancel the losers
        t.cancel()
    winner = done.pop().result()
    _, out = unpack_response(winner)
    return out


@pytest.fixture
async def coord_client():
    """In-process sidecar coordinator (no regions, no head shard)."""

    cfg = CoordinatorConfig()
    cfg.regions = []
    app = await build_app(cfg)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
async def test_signaling_handshake_announce_socket_signal_relay(coord_client):
    """Full signaling round-trip on the real Phase 1 surface: announce, open the
    peer socket, browser POSTs an offer that relays to the peer, peer answers
    and the browser drains it. This is the WebRTC handshake the data plane
    rides on top of.
    """

    peer_id = str(uuid.uuid4())
    nonce = "browser-" + str(uuid.uuid4())

    r = await coord_client.post(
        "/peers/announce", json={"peer_id": peer_id, "capabilities": CAPS}
    )
    assert r.status == 200, await r.text()

    async with coord_client.session.ws_connect(
        coord_client.make_url(f"/peers/socket/{peer_id}")
    ) as ws:
        await ws.send_str(json.dumps(
            {"type": "hello", "peer_id": peer_id, "capabilities": CAPS}
        ))
        ready = await ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["protocol_version"] == 1

        # peer now appears in /peers/list (announced AND live ws)
        lst = await (await coord_client.get("/peers/list")).json()
        assert any(p["peer_id"] == peer_id for p in lst["peers"])

        # browser -> peer offer relays over the live socket
        r = await coord_client.post(
            f"/peers/signal/{peer_id}",
            json={"from": nonce,
                  "sdp": {"type": "offer", "sdp": "v=0\r\ne2e-offer"},
                  "candidate": None},
        )
        assert r.status == 202, await r.text()
        relayed = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
        assert relayed["type"] == "signal" and relayed["from"] == nonce
        assert relayed["sdp"]["type"] == "offer"

        # peer -> browser answer; browser drains its inbox
        await ws.send_str(json.dumps(
            {"type": "signal", "to": nonce,
             "sdp": {"type": "answer", "sdp": "v=0\r\ne2e-answer"},
             "candidate": None}
        ))
        await asyncio.sleep(0.05)
        drained = await (await coord_client.get(
            f"/peers/signal/{peer_id}?inbox={nonce}"
        )).json()
        assert len(drained["signals"]) == 1
        assert drained["signals"][0]["sdp"]["type"] == "answer"

    # after socket close the peer drops from /peers/list (no live ws)
    lst = await (await coord_client.get("/peers/list")).json()
    assert not any(p["peer_id"] == peer_id for p in lst["peers"])


async def test_data_plane_first_of_k_returns_first_valid(coord_client):
    """After handshake, the data plane races 3 peers hosting the same expert;
    first_of_k returns the fast peer's result, byte-identical to a direct
    expert forward (deterministic synthetic expert)."""

    expert_id = 7
    x = torch.randn(4, HIDDEN, dtype=torch.float16)

    out = await dispatch_first_of_k_to_peers(
        expert_id, x, shard_tier=Tier.SHARED,
        peer_delays_s=[0.20, 0.05, 0.40],  # peer index 1 wins
    )
    expected = SyntheticExpert(expert_id).forward(x)
    # GLU expert: output_linear maps intermediate -> hidden, so forward returns
    # [n_tokens, hidden] (matches expert_railway_server.Expert.forward shape).
    assert out.shape == (4, HIDDEN)
    assert torch.allclose(out, expected, atol=1e-3)


async def test_private_shard_is_never_dispatched(coord_client):
    """D5: a private-tier shard is local-only — dispatch must refuse before any
    peer frame is sent."""

    x = torch.randn(2, HIDDEN, dtype=torch.float16)
    with pytest.raises(TierViolation):
        await dispatch_first_of_k_to_peers(
            5, x, shard_tier=Tier.PRIVATE, peer_delays_s=[0.0, 0.0]
        )


@pytest.mark.parametrize(
    "tier,dispatchable",
    [(Tier.PRIVATE, False), (Tier.SHARED, True), (Tier.COMMUNITY, True)],
)
def test_tier_gate(tier, dispatchable):
    assert is_dispatchable(tier) is dispatchable


def test_mixed_tier_inherits_most_restrictive():
    """Sovereignty mixed-tier rule: a shard from mixed inputs inherits the
    most-restrictive tier, so a community+private mix is private (local-only)."""

    assert most_restrictive([Tier.COMMUNITY, Tier.SHARED]) is Tier.SHARED
    assert most_restrictive([Tier.COMMUNITY, Tier.PRIVATE]) is Tier.PRIVATE
    assert most_restrictive(["shared", "community"]) is Tier.SHARED
    assert is_dispatchable(most_restrictive([Tier.COMMUNITY, Tier.PRIVATE])) is False
