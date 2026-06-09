"""Tests for the coordinator WSS endpoint at ``/ws``.

Covers:

- The hello/ready handshake happy path and every error branch listed in
  wire-frames.md §1.9 that touches the handshake (``expected_hello``,
  ``bad_json``, binary-before-hello, oversized hello).
- The ``dispatch`` text-frame round-trip against a stub UpstreamPool that
  echoes input bytes back as output.
- The WSS ``lm_head`` text-frame round-trip against a stub HeadModule.

The coordinator's UpstreamPool is replaced with ``_StubPool`` so we never
need a real WS upstream in unit tests.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts._wire import _HEADER, pack_forward_frame
from scripts.expert_coordinator import (
    Coordinator,
    CoordinatorConfig,
    HeadModule,
    RegionConfig,
    UpstreamPool,
    UpstreamUnavailable,
    build_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _three_region_config() -> CoordinatorConfig:
    cfg = CoordinatorConfig()
    cfg.regions = [
        RegionConfig("us-west2", "ws://stub-uw.local", list(range(0, 32))),
        RegionConfig("us-east4", "ws://stub-ue.local", list(range(16, 48))),
        RegionConfig("europe-west4", "ws://stub-eu.local",
                     list(range(0, 16)) + list(range(32, 64))),
    ]
    return cfg


class _StubPool:
    """An UpstreamPool replacement that echoes its input back as the response.

    Each region can be marked offline via ``offline_regions``. Per-region delay
    in ms is configurable to make ``first_of_k`` tests deterministic.
    """

    def __init__(
        self,
        regions: list[RegionConfig],
        *,
        offline_regions: set[str] | None = None,
        delays_ms: dict[str, float] | None = None,
        error_regions: set[str] | None = None,
    ) -> None:
        self.regions = regions
        self.offline_regions = offline_regions or set()
        self.delays_ms = delays_ms or {}
        self.error_regions = error_regions or set()
        # Mirror UpstreamPool's `states` shape so info_payload works.
        class _S:
            def __init__(self, region: RegionConfig, connected: bool) -> None:
                self.region = region
                self.connected = connected
                self.advertised_expert_ids: list[int] = list(region.expert_ids)
                self.last_error = None
                self.reconnects = 0
                self.degraded = False
                self.circuit_open_until = 0.0
                self.ws = None

        self.states: dict[str, Any] = {
            r.label: _S(r, connected=(r.label not in self.offline_regions))
            for r in regions
        }

    def info(self) -> dict[str, Any]:
        return {
            label: {
                "connected": st.connected,
                "url": st.region.url,
                "expert_ids": list(st.region.expert_ids),
                "advertised_expert_ids": list(st.advertised_expert_ids),
                "last_error": None,
                "reconnects": 0,
                "degraded": False,
                "circuit_open": False,
                "expected_expert_ids": list(st.region.expert_ids),
            }
            for label, st in self.states.items()
        }

    async def forward(
        self,
        region_label: str,
        expert_id: int,
        n_tokens: int,
        hidden: int,
        dtype_code: int,
        payload: bytes,
    ) -> tuple[bytes, float]:
        if region_label in self.offline_regions:
            raise UpstreamUnavailable(f"region {region_label} offline (stub)")
        delay = self.delays_ms.get(region_label, 0.0)
        if delay > 0:
            await asyncio.sleep(delay / 1000.0)
        if region_label in self.error_regions:
            raise RuntimeError(f"upstream error from stub {region_label}")
        # Echo: return the same payload framed as a response.
        resp = pack_forward_frame(expert_id, n_tokens, hidden, dtype_code, payload)
        return resp, delay

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _attach_stub_pool(app, stub: _StubPool) -> None:
    coord: Coordinator = app["coord"]
    coord.pool = stub  # type: ignore[assignment]


def _random_payload(n_tokens: int = 4) -> tuple[bytes, str]:
    rng = np.random.default_rng(seed=0)
    arr = rng.standard_normal((n_tokens, 1536)).astype(np.float16)
    return arr.tobytes(), base64.b64encode(arr.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Hello/ready handshake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hello_ready_happy_path() -> None:
    app = await build_app(_three_region_config())
    _attach_stub_pool(app, _StubPool(_three_region_config().regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({
                "type": "hello",
                "region": "browser-test",
                "protocol_version": 1,
            }))
            ready = await ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["n_experts"] == 64
            assert ready["regions"] == ["us-west2", "us-east4", "europe-west4"]
            assert ready["lm_head_ready"] is False
            assert ready["protocol_version"] == 1


@pytest.mark.asyncio
async def test_non_hello_first_frame_errors_and_closes() -> None:
    app = await build_app(_three_region_config())
    _attach_stub_pool(app, _StubPool(_three_region_config().regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "dispatch"}))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "expected_hello"


@pytest.mark.asyncio
async def test_bad_json_first_frame_errors_and_closes() -> None:
    app = await build_app(_three_region_config())
    _attach_stub_pool(app, _StubPool(_three_region_config().regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str("this is not json{")
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_json"


@pytest.mark.asyncio
async def test_binary_before_hello_errors_and_closes() -> None:
    app = await build_app(_three_region_config())
    _attach_stub_pool(app, _StubPool(_three_region_config().regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_bytes(b"\x01\x02\x03")
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "expected_hello"


# ---------------------------------------------------------------------------
# Wave-4 desktop-peer: dispatch_moved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_returns_dispatch_moved_when_no_regions_configured() -> None:
    """Wave-4 default: when ``config.regions`` is empty, the coordinator no
    longer routes ``dispatch`` frames to an upstream pool. Instead, it
    returns a ``dispatch_moved`` error frame and keeps the WS open so the
    client can switch to /peers/list + WebRTC. See wire-frames.md §1.9.
    """

    cfg = CoordinatorConfig()
    cfg.regions = []  # Wave-4 signaling-only mode
    app = await build_app(cfg)
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            ready = await ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["regions"] == []

            req_id = str(uuid.uuid4())
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": req_id,
                "mode": "wait_all",
                "k": 1,
                "calls": [{
                    "expert_id": 0,
                    "n_tokens": 1,
                    "hidden": 1536,
                    "dtype": 1,
                    "payload_b64": base64.b64encode(b"\x00" * 3072).decode("ascii"),
                }],
            }))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "dispatch_moved"
            assert err["request_id"] == req_id

            # Connection stays open per §1.9 (recoverable error). Verify
            # by sending an lm_head frame and seeing the regular not-ready
            # response rather than a closed socket.
            await ws.send_str(json.dumps({
                "type": "lm_head",
                "request_id": str(uuid.uuid4()),
                "hidden_b64": base64.b64encode(b"\x00" * 3072).decode("ascii"),
                "shape": [1, 1536],
                "dtype": "fp16",
            }))
            lm = await ws.receive_json()
            assert lm["type"] == "error"
            assert lm["code"] == "lm_head_not_ready"


# ---------------------------------------------------------------------------
# Dispatch wait_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_wait_all_happy_path_echoes_payloads() -> None:
    cfg = _three_region_config()
    app = await build_app(cfg)
    _attach_stub_pool(app, _StubPool(cfg.regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()

            calls = []
            expected_payloads: dict[int, bytes] = {}
            for eid in (3, 12, 18, 27, 40, 55):
                p, p_b64 = _random_payload(n_tokens=4)
                expected_payloads[eid] = p
                calls.append({
                    "expert_id": eid,
                    "n_tokens": 4,
                    "hidden": 1536,
                    "dtype": 1,
                    "payload_b64": p_b64,
                })
            req_id = str(uuid.uuid4())
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": req_id,
                "mode": "wait_all",
                "k": 6,
                "calls": calls,
            }))
            result = await ws.receive_json()
            assert result["type"] == "result"
            assert result["request_id"] == req_id
            assert len(result["results"]) == 6
            assert result["ms"] >= 0
            for entry in result["results"]:
                assert "error" not in entry, entry
                # Bit-identical echo round-trip.
                got = base64.b64decode(entry["payload_b64"])
                assert got == expected_payloads[entry["expert_id"]]
                # Region must be one of the regions hosting this expert.
                assert entry["region"] in cfg.expert_to_regions[entry["expert_id"]]


@pytest.mark.asyncio
async def test_dispatch_wait_all_with_offline_region() -> None:
    """Calls to experts hosted only on the offline region get an error entry."""

    cfg = _three_region_config()
    # Make eu-only experts (none in default) and uw-only experts (1..15) impossible
    # by taking us-west2 offline. Experts 0-15 live on us-west2 + europe-west4,
    # and 16-31 live on us-west2 + us-east4, so taking us-west2 down means:
    #   expert 0: still served by europe-west4
    #   expert 30: still served by us-east4
    # All experts have a fallback. Better: take ALL regions offline that host
    # expert 5 except us-west2 (we'll force it). Actually with the canonical
    # ring topology there's no expert that lives ONLY on us-west2. Instead,
    # use a custom config: region A has expert 0 alone.
    cfg2 = CoordinatorConfig()
    cfg2.regions = [
        RegionConfig("us-west2", "ws://stub-uw.local", [0]),
        RegionConfig("us-east4", "ws://stub-ue.local", [1, 2]),
        RegionConfig("europe-west4", "ws://stub-eu.local", [3]),
    ]
    app = await build_app(cfg2)
    _attach_stub_pool(app, _StubPool(cfg2.regions, offline_regions={"us-west2"}))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            p, p_b64 = _random_payload(n_tokens=1)
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": "abc",
                "mode": "wait_all",
                "k": 2,
                "calls": [
                    {"expert_id": 0, "n_tokens": 1, "hidden": 1536,
                     "dtype": 1, "payload_b64": p_b64},
                    {"expert_id": 1, "n_tokens": 1, "hidden": 1536,
                     "dtype": 1, "payload_b64": p_b64},
                ],
            }))
            result = await ws.receive_json()
            assert result["type"] == "result"
            results_by_eid = {r["expert_id"]: r for r in result["results"]}
            # expert 0 lives only on us-west2 (offline) -> error
            assert results_by_eid[0].get("error") == "upstream_unavailable"
            # expert 1 lives on us-east4 (online) -> success
            assert "error" not in results_by_eid[1]
            assert results_by_eid[1]["region"] == "us-east4"


@pytest.mark.asyncio
async def test_dispatch_bad_mode_returns_error_frame_stays_open() -> None:
    app = await build_app(_three_region_config())
    _attach_stub_pool(app, _StubPool(_three_region_config().regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": "1",
                "mode": "best_of_world",  # bad
                "k": 1,
                "calls": [{
                    "expert_id": 0, "n_tokens": 1, "hidden": 1536,
                    "dtype": 1, "payload_b64": "",
                }],
            }))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_mode"
            assert err["request_id"] == "1"


# ---------------------------------------------------------------------------
# Dispatch first_of_k
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_first_of_k_picks_fast_region() -> None:
    cfg = _three_region_config()
    app = await build_app(cfg)
    _attach_stub_pool(app, _StubPool(
        cfg.regions,
        delays_ms={"us-west2": 30.0, "us-east4": 5.0, "europe-west4": 30.0},
    ))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            # Expert 20 lives on us-west2 and us-east4. With us-east4 at 5ms
            # the winner should be us-east4.
            _, p_b64 = _random_payload(n_tokens=1)
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": "id-1",
                "mode": "first_of_k",
                "k": 1,
                "calls": [{
                    "expert_id": 20, "n_tokens": 1, "hidden": 1536,
                    "dtype": 1, "payload_b64": p_b64,
                }],
            }))
            result = await ws.receive_json()
            assert result["type"] == "result"
            entry = result["results"][0]
            assert "error" not in entry
            assert entry["region"] == "us-east4"


@pytest.mark.asyncio
async def test_dispatch_first_of_k_one_region_fails_other_wins() -> None:
    cfg = _three_region_config()
    app = await build_app(cfg)
    _attach_stub_pool(app, _StubPool(
        cfg.regions,
        error_regions={"us-west2"},
    ))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            _, p_b64 = _random_payload(n_tokens=1)
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": "id-2",
                "mode": "first_of_k",
                "k": 1,
                "calls": [{
                    "expert_id": 20, "n_tokens": 1, "hidden": 1536,
                    "dtype": 1, "payload_b64": p_b64,
                }],
            }))
            result = await ws.receive_json()
            entry = result["results"][0]
            assert "error" not in entry, entry
            assert entry["region"] == "us-east4"


@pytest.mark.asyncio
async def test_dispatch_first_of_k_all_fail_returns_all_regions_failed() -> None:
    cfg = _three_region_config()
    app = await build_app(cfg)
    _attach_stub_pool(app, _StubPool(
        cfg.regions, error_regions={"us-west2", "us-east4"},
    ))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            _, p_b64 = _random_payload(n_tokens=1)
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": "id-3",
                "mode": "first_of_k",
                "k": 1,
                "calls": [{
                    # Expert 20 lives on us-west2 + us-east4 only.
                    "expert_id": 20, "n_tokens": 1, "hidden": 1536,
                    "dtype": 1, "payload_b64": p_b64,
                }],
            }))
            result = await ws.receive_json()
            entry = result["results"][0]
            assert entry["error"] == "all_regions_failed"
            assert len(entry["errors"]) >= 1


@pytest.mark.asyncio
async def test_dispatch_first_of_k_single_region_expert() -> None:
    """An expert with only one configured region still works in first_of_k."""

    cfg = CoordinatorConfig()
    cfg.regions = [
        RegionConfig("us-west2", "ws://uw.local", [0]),
    ]
    app = await build_app(cfg)
    _attach_stub_pool(app, _StubPool(cfg.regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            _, p_b64 = _random_payload(n_tokens=1)
            await ws.send_str(json.dumps({
                "type": "dispatch",
                "request_id": "id-4",
                "mode": "first_of_k",
                "k": 1,
                "calls": [{
                    "expert_id": 0, "n_tokens": 1, "hidden": 1536,
                    "dtype": 1, "payload_b64": p_b64,
                }],
            }))
            result = await ws.receive_json()
            entry = result["results"][0]
            assert "error" not in entry
            assert entry["region"] == "us-west2"


# ---------------------------------------------------------------------------
# Unknown frame type and binary after hello
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_frame_type_returns_error_stays_open() -> None:
    app = await build_app(_three_region_config())
    _attach_stub_pool(app, _StubPool(_three_region_config().regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            await ws.send_str(json.dumps({"type": "frobnicate"}))
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "unknown_type"


@pytest.mark.asyncio
async def test_binary_after_hello_returns_unknown_type() -> None:
    app = await build_app(_three_region_config())
    _attach_stub_pool(app, _StubPool(_three_region_config().regions))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            await ws.send_bytes(b"\x00\x01\x02")
            err = await ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "unknown_type"
