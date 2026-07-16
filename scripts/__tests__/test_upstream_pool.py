"""Tests for the coordinator's UpstreamPool.

Strategy: instead of standing up a real aiohttp WSS server we inject a
stub ``ws_connect`` callable that returns a fake WS object. The fake supports
the small surface UpstreamPool actually uses: ``send_bytes``, ``receive_str``,
``receive_bytes``, ``close``, ``wait_closed``.

This keeps the test deterministic and fast (no event-loop port contention)
while still exercising the real serialization, reconnect, and circuit-breaker
logic.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import numpy as np
import pytest
import torch

from scripts._wire import _HEADER, pack_forward_frame, pack_tensor
from scripts.expert_coordinator import (
    CIRCUIT_COOLDOWN_S,
    CIRCUIT_THRESHOLD,
    RegionConfig,
    UpstreamPool,
    UpstreamUnavailable,
)


# ---------------------------------------------------------------------------
# Fake WS
# ---------------------------------------------------------------------------


class _FakeUpstreamWS:
    """Stub for an aiohttp ClientWebSocketResponse.

    Behaves like an echo expert: on ``send_bytes`` it queues the matching
    response frame; ``receive_bytes`` pops the front.
    """

    def __init__(
        self,
        region_label: str,
        expert_ids: list[int],
        *,
        delay_ms: float = 0.0,
        echo: bool = True,
        respond_with_error: bool = False,
    ) -> None:
        self.region_label = region_label
        self.expert_ids = expert_ids
        self.delay_ms = delay_ms
        self.echo = echo
        self.respond_with_error = respond_with_error
        self.closed = False
        self._hello_sent = False
        self._send_log: list[bytes] = []
        self._pending_responses: asyncio.Queue[bytes] = asyncio.Queue()
        self._close_event = asyncio.Event()

    async def receive_str(self) -> str:
        """First call returns the hello frame; later calls block forever."""

        if not self._hello_sent:
            self._hello_sent = True
            return json.dumps({
                "type": "ready",
                "region": self.region_label,
                "experts": [{"expert_id": e, "layer": 0,
                             "hidden": 1536, "intermediate": 512}
                            for e in self.expert_ids],
            })
        # In practice the pool reads hello once then only sends binary;
        # texts arriving later are not consumed. Block.
        await self._close_event.wait()
        raise asyncio.CancelledError

    async def send_bytes(self, payload: bytes) -> None:
        self._send_log.append(payload)
        # Parse the inbound header so we can echo with matching shape.
        op, expert_id, n_tokens, hidden, dtype_code = _HEADER.unpack_from(payload, 0)
        body = payload[_HEADER.size:]
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)
        if self.respond_with_error:
            resp = bytes([0xFF]) + f"region {self.region_label} error".encode()
        elif self.echo:
            resp = pack_forward_frame(expert_id, n_tokens, hidden, dtype_code, body)
        else:
            # Send the wrong body length, which unpack_response should
            # surface as RuntimeError.
            resp = bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) + b"\x00"
        await self._pending_responses.put(resp)

    async def receive_bytes(self) -> bytes:
        return await self._pending_responses.get()

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._close_event.set()

    async def wait_closed(self) -> None:
        await self._close_event.wait()


@pytest.fixture
def coord_region_one() -> RegionConfig:
    return RegionConfig("us-west2", "ws://stub.local", [0, 1, 2])


@pytest.fixture
def coord_region_two() -> RegionConfig:
    return RegionConfig("us-east4", "ws://stub.local", [16, 17])


# ---------------------------------------------------------------------------
# Cold connect + happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_connects_and_forwards(coord_region_one: RegionConfig) -> None:
    fake = _FakeUpstreamWS("us-west2", [0, 1, 2])

    async def connect(url: str) -> Any:
        return fake

    pool = UpstreamPool([coord_region_one], ws_connect=connect)
    await pool.start()
    try:
        # Wait until the supervisor connects.
        for _ in range(50):
            if pool.states["us-west2"].connected:
                break
            await asyncio.sleep(0.02)
        assert pool.states["us-west2"].connected

        torch.manual_seed(0)
        t = torch.randn(2, 1536, dtype=torch.float16)
        payload = t.contiguous().numpy().tobytes()
        resp, ms = await pool.forward(
            region_label="us-west2",
            expert_id=1,
            n_tokens=2,
            hidden=1536,
            dtype_code=1,
            payload=payload,
        )
        # Echo: stripped payload must equal what we sent.
        body = resp[_HEADER.size:]
        assert body == payload
        assert ms >= 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pool_forward_unknown_region_raises(
    coord_region_one: RegionConfig,
) -> None:
    pool = UpstreamPool([coord_region_one], ws_connect=lambda url: _never())  # type: ignore[arg-type,return-value]
    with pytest.raises(UpstreamUnavailable):
        await pool.forward(
            region_label="atlantis",
            expert_id=0,
            n_tokens=1,
            hidden=1536,
            dtype_code=1,
            payload=b"\x00" * (1 * 1536 * 2),
        )


async def _never() -> Any:
    await asyncio.Future()


# ---------------------------------------------------------------------------
# Serialization under per-upstream lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_serializes_concurrent_forwards_per_upstream(
    coord_region_one: RegionConfig,
) -> None:
    """Two concurrent forwards to one upstream must serialize behind the lock.

    With the stub delay_ms=20 ms, two parallel forwards must take ~40 ms
    total (serialized) rather than ~20 ms (true parallel).
    """

    fake = _FakeUpstreamWS("us-west2", [0, 1, 2], delay_ms=20.0)

    async def connect(url: str) -> Any:
        return fake

    pool = UpstreamPool([coord_region_one], ws_connect=connect)
    await pool.start()
    try:
        for _ in range(50):
            if pool.states["us-west2"].connected:
                break
            await asyncio.sleep(0.02)

        payload = (np.zeros((1, 1536), dtype=np.float16).tobytes())
        t0 = time.perf_counter()
        a, b = await asyncio.gather(
            pool.forward("us-west2", 0, 1, 1536, 1, payload),
            pool.forward("us-west2", 1, 1, 1536, 1, payload),
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        # Serialized: ~40 ms; parallel would be ~20 ms.
        assert total_ms >= 35.0, f"expected serialized timing; got {total_ms:.1f}ms"
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Reconnect after mid-session drop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_reconnects_after_drop(coord_region_one: RegionConfig) -> None:
    fakes: list[_FakeUpstreamWS] = []

    async def connect(url: str) -> Any:
        fake = _FakeUpstreamWS("us-west2", [0, 1, 2])
        fakes.append(fake)
        return fake

    # Make the very first backoff tiny so the test is fast.
    from scripts import expert_coordinator as ec
    original_backoff = ec.RECONNECT_BACKOFF_S
    ec.RECONNECT_BACKOFF_S = [0.05]
    try:
        pool = UpstreamPool([coord_region_one], ws_connect=connect)
        await pool.start()
        try:
            for _ in range(100):
                if pool.states["us-west2"].connected:
                    break
                await asyncio.sleep(0.02)
            assert pool.states["us-west2"].connected
            assert len(fakes) == 1

            # Drop the first connection.
            await fakes[0].close()

            # Wait for the supervisor to reconnect.
            for _ in range(200):
                if pool.states["us-west2"].connected and len(fakes) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert pool.states["us-west2"].connected
            assert len(fakes) >= 2
            assert pool.states["us-west2"].reconnects >= 1
        finally:
            await pool.close()
    finally:
        ec.RECONNECT_BACKOFF_S = original_backoff


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_circuit_breaker_opens_after_failures(
    coord_region_one: RegionConfig,
) -> None:
    """Three consecutive connect failures open the circuit."""

    fail_count = 0

    async def connect(url: str) -> Any:
        nonlocal fail_count
        fail_count += 1
        raise RuntimeError(f"connect refused (attempt {fail_count})")

    from scripts import expert_coordinator as ec
    original_backoff = ec.RECONNECT_BACKOFF_S
    ec.RECONNECT_BACKOFF_S = [0.02]
    try:
        pool = UpstreamPool([coord_region_one], ws_connect=connect)
        await pool.start()
        try:
            # Wait until at least CIRCUIT_THRESHOLD failures have piled up.
            for _ in range(500):
                if pool.states["us-west2"].consecutive_failures >= CIRCUIT_THRESHOLD:
                    break
                await asyncio.sleep(0.02)
            assert pool.states["us-west2"].consecutive_failures >= CIRCUIT_THRESHOLD
            assert pool.states["us-west2"].circuit_open_until > time.monotonic()

            # forward() must fast-fail with UpstreamUnavailable.
            with pytest.raises(UpstreamUnavailable, match="circuit open"):
                await pool.forward(
                    "us-west2", 0, 1, 1536, 1,
                    b"\x00" * (1 * 1536 * 2),
                )
        finally:
            await pool.close()
    finally:
        ec.RECONNECT_BACKOFF_S = original_backoff


# ---------------------------------------------------------------------------
# Mismatched hello / degraded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_marks_degraded_on_hello_mismatch(
    coord_region_one: RegionConfig,
) -> None:
    """If the upstream advertises a different expert set, we keep the socket
    but mark the region degraded."""

    fake = _FakeUpstreamWS("us-west2", [99, 100])  # NOT matching expected [0,1,2]

    async def connect(url: str) -> Any:
        return fake

    pool = UpstreamPool([coord_region_one], ws_connect=connect)
    await pool.start()
    try:
        for _ in range(50):
            if pool.states["us-west2"].connected:
                break
            await asyncio.sleep(0.02)
        assert pool.states["us-west2"].connected
        assert pool.states["us-west2"].degraded is True
        assert pool.states["us-west2"].advertised_expert_ids == [99, 100]
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_info_shape(
    coord_region_one: RegionConfig, coord_region_two: RegionConfig,
) -> None:
    fakes: dict[str, _FakeUpstreamWS] = {}

    async def connect(url: str) -> Any:
        label = "us-west2" if "us-west" in url else "us-east4"
        fake = _FakeUpstreamWS(
            label,
            [0, 1, 2] if label == "us-west2" else [16, 17],
        )
        fakes[label] = fake
        return fake

    pool = UpstreamPool(
        [coord_region_one, coord_region_two], ws_connect=connect,
    )
    await pool.start()
    try:
        for _ in range(100):
            if all(s.connected for s in pool.states.values()):
                break
            await asyncio.sleep(0.02)
        info = pool.info()
        assert set(info) == {"us-west2", "us-east4"}
        assert info["us-west2"]["connected"] is True
        assert info["us-east4"]["connected"] is True
        assert info["us-west2"]["last_error"] is None
        assert info["us-west2"]["reconnects"] == 0
    finally:
        await pool.close()
