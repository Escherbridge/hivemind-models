"""Tests for the coordinator's HTTP and WSS ``lm_head`` endpoints.

These tests inject a small ``HeadModule`` and a stub tokenizer so the test
is fast and deterministic. The HTTP path and the WSS path share the same
underlying ``score_lm_head`` function; both paths are exercised here.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer
from safetensors.torch import save_file

from scripts.expert_coordinator import (
    Coordinator,
    CoordinatorConfig,
    HeadModule,
    RegionConfig,
    build_app,
    load_head_from_safetensors,
)


HIDDEN = 1536
VOCAB = 64


class _StubTokenizer:
    """Minimal tokenizer stub that returns ``"<tok N>"`` for any id."""

    def decode(self, ids: list[int]) -> str:
        return "".join(f"<tok {i}>" for i in ids)


def _three_region_config() -> CoordinatorConfig:
    cfg = CoordinatorConfig()
    cfg.regions = [
        RegionConfig("us-west2", "ws://uw.local", list(range(0, 32))),
    ]
    return cfg


def _make_head_module(tmp_path: Path) -> HeadModule:
    torch.manual_seed(0)
    state = {
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, dtype=torch.float16),
        "model.final_layernorm.weight": torch.ones(HIDDEN, dtype=torch.float16),
    }
    path = tmp_path / "shard_head.safetensors"
    save_file(state, str(path))
    return load_head_from_safetensors(path)


def _attach_head(app, head: HeadModule) -> None:
    coord: Coordinator = app["coord"]
    coord.head_module = head
    coord.tokenizer = _StubTokenizer()


@pytest.mark.asyncio
async def test_lm_head_http_happy_path(tmp_path: Path) -> None:
    app = await build_app(_three_region_config())
    _attach_head(app, _make_head_module(tmp_path))
    hidden = torch.randn(1, HIDDEN, dtype=torch.float16)
    body = {
        "hidden_b64": base64.b64encode(
            hidden.contiguous().numpy().tobytes()
        ).decode("ascii"),
        "shape": [1, HIDDEN],
        "dtype": "fp16",
    }
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/lm_head", json=body)
        assert r.status == 200
        data = await r.json()
        assert len(data["top_5"]) == 5
        # Logits must be descending.
        logits = [e["logit"] for e in data["top_5"]]
        assert logits == sorted(logits, reverse=True)
        # token_str is non-empty.
        assert all(e["token_str"] for e in data["top_5"])
        assert data["ms"] >= 0


@pytest.mark.asyncio
async def test_lm_head_http_missing_field_returns_bad_shape(tmp_path: Path) -> None:
    app = await build_app(_three_region_config())
    _attach_head(app, _make_head_module(tmp_path))
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/lm_head", json={
            "hidden_b64": "AA==", "dtype": "fp16",
        })
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "bad_shape"


@pytest.mark.asyncio
async def test_lm_head_http_payload_size_mismatch(tmp_path: Path) -> None:
    app = await build_app(_three_region_config())
    _attach_head(app, _make_head_module(tmp_path))
    async with TestClient(TestServer(app)) as client:
        # Declare [1, 1536] fp16 but ship a single byte.
        r = await client.post("/lm_head", json={
            "hidden_b64": base64.b64encode(b"\x00").decode("ascii"),
            "shape": [1, HIDDEN], "dtype": "fp16",
        })
        assert r.status == 400
        b = await r.json()
        assert b["error"] == "payload_size_mismatch"


@pytest.mark.asyncio
async def test_lm_head_ws_happy_path(tmp_path: Path) -> None:
    app = await build_app(_three_region_config())
    _attach_head(app, _make_head_module(tmp_path))
    hidden = torch.randn(1, HIDDEN, dtype=torch.float16)
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            ready = await ws.receive_json()
            assert ready["lm_head_ready"] is True
            await ws.send_str(json.dumps({
                "type": "lm_head",
                "request_id": "r1",
                "hidden_b64": base64.b64encode(
                    hidden.contiguous().numpy().tobytes()
                ).decode("ascii"),
                "shape": [1, HIDDEN],
                "dtype": "fp16",
            }))
            resp = await ws.receive_json()
            assert resp["type"] == "lm_head_result"
            assert resp["request_id"] == "r1"
            assert len(resp["top_5"]) == 5
            assert resp["ms"] >= 0


@pytest.mark.asyncio
async def test_lm_head_ws_not_ready_returns_error_frame() -> None:
    """Until head_module is set, lm_head WSS frames return lm_head_not_ready."""

    app = await build_app(_three_region_config())
    # Do NOT attach head.
    hidden = np.zeros(HIDDEN, dtype=np.float16).tobytes()
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            ready = await ws.receive_json()
            assert ready["lm_head_ready"] is False
            await ws.send_str(json.dumps({
                "type": "lm_head",
                "request_id": "r2",
                "hidden_b64": base64.b64encode(hidden).decode("ascii"),
                "shape": [1, HIDDEN],
                "dtype": "fp16",
            }))
            resp = await ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "lm_head_not_ready"
            assert resp["request_id"] == "r2"


@pytest.mark.asyncio
async def test_lm_head_ws_bad_shape_returns_error_frame(tmp_path: Path) -> None:
    app = await build_app(_three_region_config())
    _attach_head(app, _make_head_module(tmp_path))
    async with TestClient(TestServer(app)) as client:
        async with client.ws_connect("/ws") as ws:
            await ws.send_str(json.dumps({"type": "hello", "region": "t"}))
            await ws.receive_json()
            await ws.send_str(json.dumps({
                "type": "lm_head",
                "request_id": "r3",
                "hidden_b64": "AAA=",
                "shape": [1, 4],  # wrong hidden
                "dtype": "fp16",
            }))
            resp = await ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "bad_shape"
