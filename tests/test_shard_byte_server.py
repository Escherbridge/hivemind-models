"""Tests for ``scripts/shard_byte_server.py``.

Phase 2 of the shard-byte-server track (see
``conductor/tracks/shard-byte-server_20260606/plan.md``):

- Task 2.1: synthetic safetensors fixture (provided by ``conftest.py``).
- Task 2.2: slicer recomputes valid header.
- Task 2.3: ``?experts=0`` drops exactly the two stacked-expert tensors.
- Task 2.4: cache hot path is fast (and serves the same object).
- Task 2.5: out-of-range layer raises ``ValueError``.
- Task 2.6: aiohttp app smoke test against a synthetic ``SHARDS_DIR``.

These tests cover the slicer (``slice_layer_safetensors``), the cache
(``get_layer_bytes`` / ``_LAYER_CACHE``), and every route handler in
``build_app``. Combined coverage on those code paths is well above the
80% line target in spec NFR-5.
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
import time
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from conftest import (  # type: ignore[import-not-found]
    SHARD_EXPECTED_LAYER5_FULL,
    SHARD_EXPECTED_LAYER5_NON_EXPERT,
    SHARD_EXPERT_TENSOR_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_header(blob: bytes) -> tuple[dict, int]:
    """Parse a safetensors blob from bytes and return (header, data_start)."""
    hlen = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + hlen].decode("utf-8"))
    return header, 8 + hlen


def _fresh_shard_module(monkeypatch, shards_dir: Path):
    """Reload ``shard_byte_server`` against a custom ``SHARDS_DIR`` and a
    clear in-memory cache. Returns the freshly-imported module.

    ``shard_byte_server`` reads ``SHARDS_DIR`` from the environment at *import*
    time, so swapping it requires a clean reload. We also point
    ``SHARDS_LOCAL`` at a non-existent path so the fallback never kicks in
    silently.
    """
    monkeypatch.setenv("SHARDS_DIR", str(shards_dir))
    monkeypatch.setenv("SHARDS_LOCAL", str(shards_dir / "__nonexistent_fallback__"))
    monkeypatch.delenv("EXPERTS_S3_BUCKET", raising=False)
    sys.modules.pop("shard_byte_server", None)
    mod = importlib.import_module("shard_byte_server")
    mod._LAYER_CACHE.clear()
    return mod


# ---------------------------------------------------------------------------
# Task 2.1 sanity: fixture is a real safetensors blob the production
# ``read_safetensors_header`` helper can parse.
# ---------------------------------------------------------------------------


def test_fixture_is_valid_safetensors(monkeypatch,
                                       shard_synthetic_safetensors_path: Path):
    mod = _fresh_shard_module(monkeypatch, shard_synthetic_safetensors_path.parent)
    header, data_start = mod.read_safetensors_header(shard_synthetic_safetensors_path)
    # Header round-trips, contains every layer-5 tensor name, plus decoys, plus metadata.
    assert "__metadata__" in header
    assert header["__metadata__"]["synthetic"] == "true"
    for name in SHARD_EXPECTED_LAYER5_FULL:
        assert name in header, f"missing layer-5 tensor {name}"
    # And the data section begins at the expected offset.
    blob = shard_synthetic_safetensors_path.read_bytes()
    expected_header_bytes = blob[8 : data_start]
    assert json.loads(expected_header_bytes.decode("utf-8")) == header


# ---------------------------------------------------------------------------
# Task 2.2: slicer recomputes valid header with correctly remapped offsets.
# ---------------------------------------------------------------------------


def test_slicer_recomputes_valid_header(monkeypatch,
                                         shard_synthetic_safetensors_path: Path):
    mod = _fresh_shard_module(monkeypatch, shard_synthetic_safetensors_path.parent)

    sliced = mod.slice_layer_safetensors(
        shard_synthetic_safetensors_path,
        layer_idx=5,
        include_experts=True,
    )
    # The output is a valid safetensors blob.
    new_header, new_data_start = _read_header(sliced)
    # ``__metadata__`` is preserved.
    assert new_header.get("__metadata__", {}).get("synthetic") == "true"

    # Every kept tensor's recomputed ``data_offsets`` must index into the
    # new data section, and the bytes at that range must match the source.
    src_header, src_data_start = mod.read_safetensors_header(shard_synthetic_safetensors_path)
    src_bytes = shard_synthetic_safetensors_path.read_bytes()

    kept_names = [k for k in new_header.keys() if k != "__metadata__"]
    assert set(kept_names) == SHARD_EXPECTED_LAYER5_FULL, (
        f"unexpected kept set: {kept_names}"
    )

    # Offsets must be contiguous starting at 0 (no holes; cursor advances by
    # exactly the tensor byte length in the slicer).
    running = 0
    for name in kept_names:
        meta = new_header[name]
        start, end = meta["data_offsets"]
        assert start == running, (
            f"non-contiguous offset for {name}: expected start={running}, got {start}"
        )
        size = end - start
        running = end

        # Per-tensor byte equality against the source shard.
        src_meta = src_header[name]
        src_size = src_meta["data_offsets"][1] - src_meta["data_offsets"][0]
        assert size == src_size, f"size drift on {name}"
        sliced_bytes = sliced[new_data_start + start : new_data_start + end]
        src_slice = src_bytes[
            src_data_start + src_meta["data_offsets"][0] :
            src_data_start + src_meta["data_offsets"][1]
        ]
        assert sliced_bytes == src_slice, f"byte mismatch on {name}"

    # Final cursor equals the data-section length.
    assert running == len(sliced) - new_data_start


# ---------------------------------------------------------------------------
# Task 2.3: ``include_experts=False`` (the ``?experts=0`` query mode) drops
# exactly the two stacked-expert tensors and nothing else; the experts=1
# default keeps all 11.
# ---------------------------------------------------------------------------


def test_experts_0_drops_exactly_two_tensors(monkeypatch,
                                              shard_synthetic_safetensors_path: Path):
    mod = _fresh_shard_module(monkeypatch, shard_synthetic_safetensors_path.parent)

    sliced_no_experts = mod.slice_layer_safetensors(
        shard_synthetic_safetensors_path, layer_idx=5, include_experts=False,
    )
    header_no, _ = _read_header(sliced_no_experts)
    tensor_names_no = {k for k in header_no.keys() if k != "__metadata__"}
    assert tensor_names_no == SHARD_EXPECTED_LAYER5_NON_EXPERT, (
        f"experts=0 must keep exactly 9 tensors, got {len(tensor_names_no)}: "
        f"{sorted(tensor_names_no)}"
    )
    # Neither stacked-expert tensor may appear in any form.
    for name in tensor_names_no:
        for suffix in SHARD_EXPERT_TENSOR_SUFFIXES:
            assert not name.endswith(suffix), f"experts=0 leaked {name}"

    sliced_full = mod.slice_layer_safetensors(
        shard_synthetic_safetensors_path, layer_idx=5, include_experts=True,
    )
    header_full, _ = _read_header(sliced_full)
    tensor_names_full = {k for k in header_full.keys() if k != "__metadata__"}
    assert tensor_names_full == SHARD_EXPECTED_LAYER5_FULL
    # The 11-vs-9 difference must be exactly the two expert tensors.
    dropped = tensor_names_full - tensor_names_no
    assert {n.removeprefix("model.layers.5.") for n in dropped} == SHARD_EXPERT_TENSOR_SUFFIXES


# ---------------------------------------------------------------------------
# Task 2.4: cache hot path. The second call returns the SAME bytes object
# (identity check on the cache value) and runs in microseconds, well below
# the 100 ms generous bound in the plan.
# ---------------------------------------------------------------------------


def test_cache_hit_path_is_fast(monkeypatch,
                                 shard_synthetic_safetensors_path: Path):
    mod = _fresh_shard_module(monkeypatch, shard_synthetic_safetensors_path.parent)

    # First call: cold; populates the cache.
    first = mod.get_layer_bytes(5, include_experts=False)
    assert (5, False) in mod._LAYER_CACHE

    # Second call: must be served from the cache and must be the SAME object.
    t0 = time.perf_counter()
    second = mod.get_layer_bytes(5, include_experts=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert second is first, "cache hit must return the same bytes object"
    assert elapsed_ms < 100.0, (
        f"cache hit took {elapsed_ms:.3f} ms, expected <100 ms; cache may be "
        f"missing or the slice is recomputing"
    )

    # Distinct keys hit independent cache entries.
    third = mod.get_layer_bytes(5, include_experts=True)
    assert third is not first
    assert (5, True) in mod._LAYER_CACHE


# ---------------------------------------------------------------------------
# Task 2.5: out-of-range layer raises ``ValueError`` everywhere it should.
# ---------------------------------------------------------------------------


def test_out_of_range_layer_raises(monkeypatch,
                                    shard_synthetic_safetensors_path: Path):
    mod = _fresh_shard_module(monkeypatch, shard_synthetic_safetensors_path.parent)

    with pytest.raises(ValueError):
        mod.shard_for_layer(40)
    with pytest.raises(ValueError):
        mod.shard_for_layer(-1)
    with pytest.raises(ValueError):
        mod.shard_for_layer(99)

    # The slicer also rejects an in-range index that has no tensors in the
    # supplied shard (e.g. layer 99 against a layers_0_9 fixture).
    with pytest.raises(ValueError, match="no tensors for layer 99"):
        mod.slice_layer_safetensors(
            shard_synthetic_safetensors_path, layer_idx=99, include_experts=True,
        )


# ---------------------------------------------------------------------------
# Task 2.6: aiohttp app smoke test. Boot the app against a synthetic
# ``SHARDS_DIR`` and verify every route serves the right thing with the
# right headers.
# ---------------------------------------------------------------------------


async def test_app_smoke_all_routes(monkeypatch,
                                     shard_synthetic_shards_dir: Path):
    mod = _fresh_shard_module(monkeypatch, shard_synthetic_shards_dir)
    app = mod.build_app()

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            # /health -- 200 with all readiness flags true.
            resp = await client.get("/health")
            assert resp.status == 200
            health = await resp.json()
            assert health["status"] == "ok"
            assert health["embed_ready"] is True
            assert health["layer_0_9_shard_ready"] is True
            assert health["head_ready"] is True
            assert health["shards_dir"] == str(shard_synthetic_shards_dir)
            # CORS on every response.
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

            # / aliases to /health.
            resp = await client.get("/")
            assert resp.status == 200

            # /manifest.json -- spec FR-3 keys.
            resp = await client.get("/manifest.json")
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            manifest = await resp.json()
            for required_field in (
                "model", "hidden_size", "num_layers", "layer_groups",
                "num_local_experts", "num_experts_per_tok", "intermediate_size",
                "vocab_size", "embedding_multiplier", "logits_scaling",
                "rms_norm_eps", "endpoints",
            ):
                assert required_field in manifest, f"manifest missing {required_field}"
            assert manifest["hidden_size"] == 1536
            assert manifest["num_layers"] == 40
            assert manifest["num_local_experts"] == 64
            assert manifest["num_experts_per_tok"] == 6
            assert manifest["intermediate_size"] == 512
            assert manifest["vocab_size"] == 100352

            # /embed.safetensors -- streams the synthetic blob with octet-stream + CORS.
            resp = await client.get("/embed.safetensors")
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "application/octet-stream"
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            body = await resp.read()
            assert body == (
                shard_synthetic_shards_dir / "shard_embed.safetensors"
            ).read_bytes()

            # /head.safetensors -- same shape as embed.
            resp = await client.get("/head.safetensors")
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "application/octet-stream"
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

            # /layer/5.safetensors?experts=0 -- returns a re-emitted blob with
            # exactly the 9 non-expert tensors.
            resp = await client.get("/layer/5.safetensors?experts=0")
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "application/octet-stream"
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            layer_bytes = await resp.read()
            header, _ = _read_header(layer_bytes)
            assert {k for k in header.keys() if k != "__metadata__"} == \
                SHARD_EXPECTED_LAYER5_NON_EXPERT

            # /layer/5.safetensors (default experts=1) returns all 11.
            resp = await client.get("/layer/5.safetensors")
            assert resp.status == 200
            layer_bytes_full = await resp.read()
            header_full, _ = _read_header(layer_bytes_full)
            assert {k for k in header_full.keys() if k != "__metadata__"} == \
                SHARD_EXPECTED_LAYER5_FULL

            # /layer/40.safetensors -- out-of-range index returns 400.
            resp = await client.get("/layer/40.safetensors")
            assert resp.status == 400


async def test_app_returns_503_when_shards_missing(monkeypatch, tmp_path: Path):
    """The hydration window is observable: missing shard files yield 503,
    not a 5xx crash. Spec NFR-4 reliability requirement."""
    empty_shards = tmp_path / "empty"
    empty_shards.mkdir()
    mod = _fresh_shard_module(monkeypatch, empty_shards)
    app = mod.build_app()

    async with TestServer(app) as server:
        async with TestClient(server) as client:
            for path in ("/embed.safetensors", "/head.safetensors"):
                resp = await client.get(path)
                assert resp.status == 503, f"{path} should 503 when missing"

            # Layer slice raises FileNotFoundError -> 503.
            resp = await client.get("/layer/5.safetensors?experts=0")
            assert resp.status == 503

            # /health still works, and reports nothing ready.
            resp = await client.get("/health")
            assert resp.status == 200
            health = await resp.json()
            assert health["embed_ready"] is False
            assert health["layer_0_9_shard_ready"] is False
            assert health["head_ready"] is False
