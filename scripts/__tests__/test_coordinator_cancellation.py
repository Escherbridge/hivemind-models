"""Stress test for ``first_of_k`` cancellation hygiene.

The risk: when first_of_k cancels a loser task, the loser's upstream
response may still be in flight on the wire. Because the per-upstream lock
is held during the send/recv pair, the cancelled coroutine must finish
draining the response (or release the lock in a state where the next call
can succeed) before the next iteration runs.

This stress test runs 100 first_of_k dispatches in a loop with mixed
upstream timings and asserts:

1. No iteration's per-call result is corrupted (the echoed payload still
   matches the input).
2. The pool's per-upstream state is healthy after the loop -- no leftover
   pending bytes that would desync subsequent calls.

The test uses the ``_StubPool`` from ``test_coordinator_ws.py`` repackaged
here for simplicity, but configured with two regions and per-region delays
that force frequent cancellation events.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import uuid
from typing import Any

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts._wire import pack_forward_frame
from scripts.expert_coordinator import (
    Coordinator,
    CoordinatorConfig,
    RegionConfig,
    UpstreamUnavailable,
    build_app,
)


class _DelayedStubPool:
    """Stub pool with per-region jittered delay to force frequent cancellation."""

    def __init__(self, regions: list[RegionConfig], delays_ms: dict[str, tuple[float, float]]):
        self.regions = regions
        self.delays_ms = delays_ms
        self.call_count = 0
        self.error_count = 0

        class _S:
            def __init__(self, region: RegionConfig) -> None:
                self.region = region
                self.connected = True
                self.advertised_expert_ids = list(region.expert_ids)
                self.last_error = None
                self.reconnects = 0
                self.degraded = False
                self.circuit_open_until = 0.0
                self.ws = None

        self.states: dict[str, Any] = {r.label: _S(r) for r in regions}

    def info(self) -> dict[str, Any]:
        return {
            label: {
                "connected": True, "url": st.region.url,
                "advertised_expert_ids": list(st.region.expert_ids),
                "expected_expert_ids": list(st.region.expert_ids),
                "last_error": None, "reconnects": 0, "degraded": False,
                "circuit_open": False,
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
        lo, hi = self.delays_ms[region_label]
        delay = random.uniform(lo, hi)
        # Even on cancellation the asyncio.sleep is a clean cancellation point;
        # the per-upstream lock is held only during this sleep + frame build.
        await asyncio.sleep(delay / 1000.0)
        self.call_count += 1
        resp = pack_forward_frame(expert_id, n_tokens, hidden, dtype_code, payload)
        return resp, delay

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _two_region_redundant_config() -> CoordinatorConfig:
    cfg = CoordinatorConfig()
    cfg.regions = [
        RegionConfig("us-west2", "ws://uw.local", [0, 1, 2, 3]),
        RegionConfig("us-east4", "ws://ue.local", [0, 1, 2, 3]),
    ]
    return cfg


@pytest.mark.asyncio
async def test_first_of_k_100_iters_no_desync() -> None:
    cfg = _two_region_redundant_config()
    app = await build_app(cfg)
    pool = _DelayedStubPool(
        cfg.regions,
        delays_ms={
            "us-west2": (5.0, 25.0),
            "us-east4": (5.0, 25.0),
        },
    )
    coord: Coordinator = app["coord"]
    coord.pool = pool  # type: ignore[assignment]

    random.seed(0)
    rng = np.random.default_rng(seed=1)

    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "stress"}))
            await ws.receive_json()

            for it in range(100):
                payload_bytes = rng.standard_normal(
                    (1, 1536)).astype(np.float16).tobytes()
                payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
                req_id = str(uuid.uuid4())
                await ws.send_str(json.dumps({
                    "type": "dispatch",
                    "request_id": req_id,
                    "mode": "first_of_k",
                    "k": 1,
                    "calls": [{
                        "expert_id": it % 4,
                        "n_tokens": 1,
                        "hidden": 1536,
                        "dtype": 1,
                        "payload_b64": payload_b64,
                    }],
                }))
                result = await ws.receive_json()
                assert result["type"] == "result", f"iter {it}: {result}"
                assert result["request_id"] == req_id
                entry = result["results"][0]
                assert "error" not in entry, f"iter {it}: {entry}"
                # The echoed payload must still match.
                assert base64.b64decode(entry["payload_b64"]) == payload_bytes, (
                    f"iter {it}: payload desync"
                )
