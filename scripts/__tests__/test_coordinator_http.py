"""Tests for the coordinator's HTTP surface: /, /health, /info, /metrics, CORS.

These tests exercise the aiohttp app via ``aiohttp.test_utils.TestClient`` so
the full request/response pipeline is hit, including the CORS middleware
behavior.
"""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.expert_coordinator import (
    CoordinatorConfig,
    RegionConfig,
    build_app,
    load_config_from_env,
)


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------


def _full_env() -> dict[str, str]:
    return {
        "EXPERT_US_WEST_URL": "ws://us-west.local",
        "EXPERT_US_WEST_IDS": "0-31",
        "EXPERT_US_EAST_URL": "ws://us-east.local",
        "EXPERT_US_EAST_IDS": "16-47",
        "EXPERT_EU_URL": "ws://eu.local",
        "EXPERT_EU_IDS": "0-15,32-63",
    }


def test_load_config_full_ring_overlap() -> None:
    cfg = load_config_from_env(_full_env())
    assert len(cfg.regions) == 3

    labels = {r.label: r for r in cfg.regions}
    assert sorted(labels) == ["europe-west4", "us-east4", "us-west2"]
    assert labels["us-west2"].expert_ids == list(range(0, 32))
    assert labels["us-east4"].expert_ids == list(range(16, 48))
    assert labels["europe-west4"].expert_ids == list(range(0, 16)) + list(range(32, 64))

    e2r = cfg.expert_to_regions
    # Ring overlap: every expert id lives on at least 2 regions.
    for eid in range(64):
        assert len(e2r[eid]) >= 1
    # Specific overlap check
    assert set(e2r[5]) == {"us-west2", "europe-west4"}
    assert set(e2r[20]) == {"us-west2", "us-east4"}
    assert set(e2r[40]) == {"us-east4", "europe-west4"}


def test_load_config_partial_region_warns_but_continues() -> None:
    env = {
        "EXPERT_US_WEST_URL": "ws://us-west.local",
        "EXPERT_US_WEST_IDS": "0-31",
    }
    cfg = load_config_from_env(env)
    assert len(cfg.regions) == 1
    assert cfg.regions[0].label == "us-west2"


def test_load_config_no_regions_raises() -> None:
    with pytest.raises(SystemExit):
        load_config_from_env({})


def test_load_config_url_without_ids_is_skipped() -> None:
    env = {
        "EXPERT_US_WEST_URL": "ws://us-west.local",
        # missing EXPERT_US_WEST_IDS
        "EXPERT_EU_URL": "ws://eu.local",
        "EXPERT_EU_IDS": "0-63",
    }
    cfg = load_config_from_env(env)
    labels = [r.label for r in cfg.regions]
    assert labels == ["europe-west4"]


# ---------------------------------------------------------------------------
# HTTP endpoint shape
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


@pytest.mark.asyncio
async def test_health_returns_expected_shape() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/health")
        assert r.status == 200
        body = await r.json()
        assert body["status"] == "ok"
        assert body["n_experts"] == 64
        assert body["regions"] == ["us-west2", "us-east4", "europe-west4"]
        assert body["lm_head_ready"] is False
        assert isinstance(body["uptime_s"], float)


@pytest.mark.asyncio
async def test_root_alias_for_health() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/")
        assert r.status == 200
        body = await r.json()
        assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_info_exposes_expert_map() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/info")
        assert r.status == 200
        body = await r.json()
        assert body["protocol_version"] == 1
        assert body["lm_head_ready"] is False
        assert "expert_to_regions" in body
        # expert id 20 hosted on us-west2 and us-east4
        assert sorted(body["expert_to_regions"]["20"]) == ["us-east4", "us-west2"]
        assert sorted(body["expert_to_regions"]["5"]) == ["europe-west4", "us-west2"]
        assert len(body["regions"]) == 3


@pytest.mark.asyncio
async def test_metrics_endpoint() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/metrics")
        assert r.status == 200
        body = await r.json()
        assert body["dispatch_count"] == 0
        assert body["lm_head_count"] == 0
        assert body["errors_by_code"] == {}


@pytest.mark.asyncio
async def test_cors_headers_on_all_endpoints() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        for path in ("/", "/health", "/info", "/metrics"):
            r = await client.get(path)
            assert r.headers.get("Access-Control-Allow-Origin") == "*", path


@pytest.mark.asyncio
async def test_cors_preflight_on_lm_head() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.options("/lm_head")
        assert r.status == 204
        assert r.headers["Access-Control-Allow-Origin"] == "*"


@pytest.mark.asyncio
async def test_lm_head_returns_503_when_not_ready() -> None:
    """POST /lm_head must surface ``lm_head_not_ready`` until head loads."""

    import base64

    import numpy as np

    app = await build_app(_three_region_config())
    payload = np.zeros((1, 1536), dtype=np.float16).tobytes()
    body = {
        "hidden_b64": base64.b64encode(payload).decode("ascii"),
        "shape": [1, 1536],
        "dtype": "fp16",
    }
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/lm_head", json=body)
        assert r.status == 503
        b = await r.json()
        assert b["error"] == "lm_head_not_ready"


@pytest.mark.asyncio
async def test_lm_head_returns_400_on_bad_shape() -> None:
    import base64

    app = await build_app(_three_region_config())
    body = {
        "hidden_b64": base64.b64encode(b"\x00" * 4).decode("ascii"),
        "shape": [1, 999],  # wrong hidden
        "dtype": "fp16",
    }
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/lm_head", json=body)
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_lm_head_returns_400_on_bad_json() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.post(
            "/lm_head", data=b"this is not json{",
            headers={"Content-Type": "application/json"},
        )
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_json"


@pytest.mark.asyncio
async def test_ws_endpoint_rejects_non_websocket_get() -> None:
    app = await build_app(_three_region_config())
    async with TestClient(TestServer(app)) as client:
        r = await client.get("/ws")
        assert r.status == 400
        b = await r.json()
        assert "error" in b
