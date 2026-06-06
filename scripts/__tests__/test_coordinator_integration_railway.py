"""Integration tests against the live Railway expert services.

These tests are GATED. They only run when ``RUN_RAILWAY_INTEGRATION=1`` is set
in the environment. Otherwise pytest skips them. The reason: the live
services bill on cold boot and have a ~30 s wake-up, and the test relies on
the operator having put all three regions into ``COMPUTE_MODE=echo`` first.

The tests are also marked ``@pytest.mark.integration`` so any future CI lane
can include or exclude them by mark alone.

Run locally with::

    RUN_RAILWAY_INTEGRATION=1 \
    EXPERT_US_WEST_URL=wss://expert-us-west.up.railway.app \
    EXPERT_US_WEST_IDS=0-31 \
    EXPERT_US_EAST_URL=wss://expert-us-east.up.railway.app \
    EXPERT_US_EAST_IDS=16-47 \
    EXPERT_EU_URL=wss://expert-eu.up.railway.app \
    EXPERT_EU_IDS=0-15,32-63 \
    pytest scripts/__tests__/test_coordinator_integration_railway.py -v

The numerical targets (110 ms wait_all p99, 40 ms first_of_k p99) are a touch
over the Railway backbone numbers from ``output/region_latency_findings.md``
to allow for the local-developer-to-coordinator hop on top of the backbone.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import statistics
import time
import uuid
from typing import Any

import aiohttp
import numpy as np
import pytest

from scripts.expert_coordinator import (
    CoordinatorConfig,
    build_app,
    load_config_from_env,
    run_forever,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_RAILWAY_INTEGRATION") != "1",
        reason="set RUN_RAILWAY_INTEGRATION=1 to run live Railway integration tests",
    ),
]


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    if f + 1 < len(s):
        return s[f] + (s[f + 1] - s[f]) * (k - f)
    return s[f]


@pytest.fixture
async def coord_running(coord_free_port: int):
    """Boot the real coordinator in-process pointed at live upstreams."""

    cfg = load_config_from_env()
    cfg.port = coord_free_port

    task = asyncio.create_task(run_forever(cfg))
    # Wait for the healthcheck to come up.
    for _ in range(200):
        try:
            r = socket.create_connection(("127.0.0.1", cfg.port), timeout=0.2)
            r.close()
            break
        except OSError:
            await asyncio.sleep(0.05)
    yield cfg
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def _random_payload(n_tokens: int = 8) -> str:
    rng = np.random.default_rng(seed=42)
    return base64.b64encode(
        rng.standard_normal((n_tokens, 1536)).astype(np.float16).tobytes()
    ).decode("ascii")


async def _run_dispatch_loop(
    coord_url: str,
    mode: str,
    iters: int,
    k: int = 6,
    n_tokens: int = 8,
) -> dict[str, Any]:
    """Open one WSS to the local coordinator and run `iters` dispatches."""

    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(f"{coord_url}/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "int-test"}))
            await ws.receive_json()

            payload_b64 = _random_payload(n_tokens=n_tokens)
            per_iter_ms: list[float] = []
            winners: dict[str, int] = {}

            for _ in range(iters):
                # Six experts spanning the ring so every region is touched.
                calls = [
                    {
                        "expert_id": eid,
                        "n_tokens": n_tokens,
                        "hidden": 1536,
                        "dtype": 1,
                        "payload_b64": payload_b64,
                    }
                    for eid in (3, 12, 18, 27, 40, 55)[:k]
                ]
                req_id = str(uuid.uuid4())
                t0 = time.perf_counter()
                await ws.send_str(json.dumps({
                    "type": "dispatch",
                    "request_id": req_id,
                    "mode": mode,
                    "k": k,
                    "calls": calls,
                }))
                resp = await ws.receive_json()
                per_iter_ms.append((time.perf_counter() - t0) * 1000.0)
                assert resp["type"] == "result", resp
                assert resp["request_id"] == req_id
                # Echo round-trip: every payload must be byte-identical to input.
                for entry in resp["results"]:
                    if "error" in entry:
                        continue
                    assert entry["payload_b64"] == payload_b64, (
                        f"payload mismatch for expert {entry['expert_id']}"
                    )
                    winners[entry["region"]] = winners.get(entry["region"], 0) + 1
                await asyncio.sleep(0.02)
        return {
            "per_iter_ms": per_iter_ms,
            "winners": winners,
        }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_wait_all_against_live_railway(coord_running) -> None:
    cfg: CoordinatorConfig = coord_running
    result = await _run_dispatch_loop(
        f"http://127.0.0.1:{cfg.port}", mode="wait_all", iters=40,
    )
    p99 = _percentile(result["per_iter_ms"], 0.99)
    p50 = _percentile(result["per_iter_ms"], 0.50)
    print(f"\nwait_all   p50={p50:.1f}ms  p99={p99:.1f}ms  n={len(result['per_iter_ms'])}")
    assert p99 < 110.0, f"wait_all p99 {p99:.1f}ms exceeds 110 ms target"


@pytest.mark.asyncio
async def test_first_of_k_against_live_railway(coord_running) -> None:
    cfg: CoordinatorConfig = coord_running
    result = await _run_dispatch_loop(
        f"http://127.0.0.1:{cfg.port}", mode="first_of_k", iters=40,
    )
    p99 = _percentile(result["per_iter_ms"], 0.99)
    p50 = _percentile(result["per_iter_ms"], 0.50)
    print(
        f"\nfirst_of_k p50={p50:.1f}ms  p99={p99:.1f}ms  n={len(result['per_iter_ms'])}"
        f"  winners={result['winners']}"
    )
    assert p99 < 40.0, f"first_of_k p99 {p99:.1f}ms exceeds 40 ms target"
    # Expect non-trivial distribution across at least 2 regions.
    assert len(result["winners"]) >= 2, (
        f"expected winners across multiple regions; got {result['winners']}"
    )
