"""Shared pytest fixtures for hivemind-models tests.

Fixture naming convention (per Wave 3 collision rules in
conductor/tracks/_shared/ownership.md):

- Agent A (shard-byte-server) namespaces fixtures with prefix ``shard_*``.
- Agent B (expert-coordinator)   namespaces fixtures with prefix ``coord_*``.
- pi    (browser-shaders / reference) namespaces fixtures with prefix ``ref_*``.

Only ``shard_*`` fixtures live in this file.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Make the in-repo ``scripts/`` directory importable so the production module
# ``shard_byte_server`` resolves without an install step.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Synthetic safetensors fixture (Phase 2 Task 2.1).
#
# The slicer in ``shard_byte_server.slice_layer_safetensors`` operates on the
# safetensors binary layout:
#
#     [u64 header_len]
#     [header_bytes: utf-8 JSON]
#         {
#           "tensor_name": {"dtype": "...", "shape": [...], "data_offsets": [s, e]},
#           "__metadata__": {...}
#         }
#     [data_bytes ...]
#
# We synthesize a tiny shard that mimics the Granite-tiny layer-group layout
# for layers 0..9, with layer 5 fully populated (the canonical test target).
# Every tensor's bytes are unique so post-slice byte-for-byte comparison can
# catch any header / offset mistake.
# ---------------------------------------------------------------------------


def _bytes_for(label: str, size: int) -> bytes:
    """Generate ``size`` deterministic bytes derived from ``label`` so each
    tensor's payload is distinguishable on inspection."""
    seed = (sum(label.encode("utf-8")) % 251) or 1
    return bytes(((seed + i) % 256 for i in range(size)))


# Tiny dtype size table. fp16 is 2 bytes; fp32 is 4; we only need fp16 here.
_DTYPE_SIZE = {"F16": 2, "F32": 4}


def _build_safetensors(tensor_specs: list[tuple[str, str, list[int]]],
                       extra_metadata: dict | None = None) -> bytes:
    """Construct a valid safetensors blob from ``(name, dtype, shape)`` tuples.

    Returns the raw bytes (header_len + header_json + data) so callers can
    write them to disk for ``shard_byte_server`` to ingest.
    """
    header: dict = {}
    data_buf = io.BytesIO()
    cursor = 0
    for name, dtype, shape in tensor_specs:
        nelem = 1
        for d in shape:
            nelem *= d
        nbytes = nelem * _DTYPE_SIZE[dtype]
        payload = _bytes_for(name, nbytes)
        data_buf.write(payload)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    if extra_metadata is not None:
        header["__metadata__"] = extra_metadata

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out = io.BytesIO()
    out.write(struct.pack("<Q", len(header_json)))
    out.write(header_json)
    out.write(data_buf.getvalue())
    return out.getvalue()


# Layer 5's full tensor name set (per wire-frames.md §3.3 + spec FR-2).
# Tensor shapes are small but proportionally correct so the slice is still
# a meaningful exercise. The two stacked-expert tensors must be omitted by
# the ``experts=0`` mode of the slicer; all others must be kept.
_LAYER_5_TENSORS: list[tuple[str, str, list[int]]] = [
    # 9 non-expert tensors -- these must remain in experts=0 mode.
    ("model.layers.5.self_attn.q_proj.weight",                                  "F16", [8, 8]),
    ("model.layers.5.self_attn.k_proj.weight",                                  "F16", [4, 8]),
    ("model.layers.5.self_attn.v_proj.weight",                                  "F16", [4, 8]),
    ("model.layers.5.self_attn.o_proj.weight",                                  "F16", [8, 8]),
    ("model.layers.5.input_layernorm.weight",                                   "F16", [8]),
    ("model.layers.5.post_attention_layernorm.weight",                          "F16", [8]),
    ("model.layers.5.block_sparse_moe.router.weight",                           "F16", [4, 8]),
    ("model.layers.5.block_sparse_moe.shared_mlp.gate_proj.weight",             "F16", [4, 8]),
    ("model.layers.5.block_sparse_moe.shared_mlp.down_proj.weight",             "F16", [8, 4]),
    # 2 stacked expert tensors -- these are the only ones experts=0 should drop.
    ("model.layers.5.block_sparse_moe.input_linear.weight",                     "F16", [4, 4, 8]),
    ("model.layers.5.block_sparse_moe.output_linear.weight",                    "F16", [4, 8, 4]),
]

# Decoy tensors that share the same shard but belong to *other* layers.
# The slicer must NOT include them in any layer-5 slice.
_DECOY_TENSORS: list[tuple[str, str, list[int]]] = [
    ("model.layers.0.self_attn.q_proj.weight", "F16", [8, 8]),
    ("model.layers.3.input_layernorm.weight",  "F16", [8]),
    ("model.layers.7.block_sparse_moe.router.weight", "F16", [4, 8]),
    ("model.embed_tokens.weight",              "F16", [16, 8]),
]


# The 9-tensor expected set for layer 5 with experts=0 (per FR-2 + spec line 64).
SHARD_EXPECTED_LAYER5_NON_EXPERT = frozenset(name for name, *_ in _LAYER_5_TENSORS[:9])

# The 11-tensor expected set for layer 5 with experts=1 (all of _LAYER_5_TENSORS).
SHARD_EXPECTED_LAYER5_FULL = frozenset(name for name, *_ in _LAYER_5_TENSORS)

# The two tensors experts=0 must drop, by exact name (must match the
# ``_EXPERT_TENSOR_NAMES`` blacklist in ``shard_byte_server``).
SHARD_EXPERT_TENSOR_SUFFIXES = frozenset({
    "block_sparse_moe.input_linear.weight",
    "block_sparse_moe.output_linear.weight",
})


@pytest.fixture
def shard_synthetic_safetensors_bytes() -> bytes:
    """Return raw bytes of a synthetic safetensors file containing layer 5's
    tensors plus a few decoys (other layers + embed)."""
    specs = _LAYER_5_TENSORS + _DECOY_TENSORS
    return _build_safetensors(specs, extra_metadata={"format": "pt", "synthetic": "true"})


@pytest.fixture
def shard_synthetic_safetensors_path(tmp_path: Path,
                                     shard_synthetic_safetensors_bytes: bytes) -> Path:
    """Write the synthetic safetensors blob to a temp path mimicking the
    real layer-group shard filename so ``shard_for_layer`` would find it."""
    path = tmp_path / "shard_layers_0_9.safetensors"
    path.write_bytes(shard_synthetic_safetensors_bytes)
    return path


@pytest.fixture
def shard_synthetic_shards_dir(tmp_path: Path,
                               shard_synthetic_safetensors_bytes: bytes) -> Path:
    """Build a temp shards directory layout with the 3 required filenames,
    each backed by the same tiny synthetic blob. The ``embed`` and ``head``
    files are intentionally identical to the layer-0..9 blob for fixture
    economy; tests should not depend on their contents being distinct."""
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    for name in (
        "shard_embed.safetensors",
        "shard_layers_0_9.safetensors",
        "shard_head.safetensors",
    ):
        (shards_dir / name).write_bytes(shard_synthetic_safetensors_bytes)
    return shards_dir


@pytest.fixture
def shard_tensor_specs_layer5_full() -> list[tuple[str, str, list[int]]]:
    """Expose layer 5's full tensor spec list to tests that want to recompute
    expected byte slices against the source fixture."""
    return list(_LAYER_5_TENSORS)
