"""Pure-PyTorch reference forward for Granite-tiny layer 5.

Why this file exists
====================

The browser-as-peer Phase 1 demo runs Granite-tiny's first attention layer
(layer 5) on the GPU via WebGPU. WebGPU WGSL shaders are validated against
a Python reference; this file is that reference. Its output must match
HuggingFace's ``GraniteMoeHybridDecoderLayer`` to fp32 precision, and the
golden JSON at ``reference/golden/layer5.json`` is generated from a single
run of this function on a frozen input tensor.

Granite-tiny layer 5 is the first *attention* layer; layers 0-4 are Mamba
(selective scan) and out of scope for the browser demo. The forward we
implement is the canonical decoder layer forward in eager attention mode,
which is the same mode used by ``hivemind-models/scripts/verify_granite_handler.py``.

Math summary (the contract the WGSL shaders reproduce)
======================================================

Given a hidden state ``x`` of shape ``[batch, seq, 1536]`` and the layer-5
weight dict (keys stripped of the ``model.layers.5.`` prefix), compute::

    residual1 = x
    x = RMSNorm(x, weight=input_layernorm.weight, eps=1e-5)
    attn_out = GQA_attention(
        x,
        q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight,
        scale=attention_multiplier,         # 0.0078125 for granite-4.0-h-tiny
        num_heads=12, num_kv_heads=4, head_dim=128,
    )  # GQA: K and V are repeated 3x to match Q's 12 heads
    x = residual1 + attn_out * residual_multiplier   # 0.22

    residual2 = x
    x = RMSNorm(x, weight=post_attention_layernorm.weight, eps=1e-5)

    # Router: pick top-6 experts per token.
    router_logits = x @ router.layer.weight.T          # [batch, seq, 64]
    top6_logits, top6_ids = topk(router_logits, k=6)
    top6_weights = softmax(top6_logits, dim=-1).type_as(x)

    # MoE output: weighted sum of the 6 routed expert outputs.
    # The browser receives the 6 expert outputs from the coordinator and
    # recombines them on the GPU. We compute the same sum here in fp32.
    moe_out = zeros_like(x)
    for token in range(seq_len):
        for k in range(6):
            eid = top6_ids[token, k]
            w   = top6_weights[token, k]
            h   = input_linear.weight[eid] @ x[token]      # [1024]
            gate, up = chunk(h, 2)
            h = silu(gate) * up
            moe_out[token] += w * (output_linear.weight[eid] @ h)

    # Shared MLP (always on, regardless of MoE).
    h = shared_mlp.input_linear.weight @ x              # [2048]
    gate, up = chunk(h, 2)
    h = silu(gate) * up                                  # [1024]
    shared_out = shared_mlp.output_linear.weight @ h    # [1536]

    out = residual2 + (moe_out + shared_out) * residual_multiplier

    return {
        "hidden":                      out,
        "router_top6_expert_ids":      top6_ids,        # int64
        "router_top6_weights":         top6_weights,    # float32
    }

The reference does not depend on ``transformers`` for the math itself; we
*do* import the HF layer in the tests to assert bit-equivalence, but the
production path here is dependency-light and CPU-only.

Input convention
================

The reference takes the layer-5 weight dict with **keys already stripped**
of the ``model.layers.5.`` prefix, so the keys are exactly::

    input_layernorm.weight
    post_attention_layernorm.weight
    self_attn.q_proj.weight
    self_attn.k_proj.weight
    self_attn.v_proj.weight
    self_attn.o_proj.weight
    block_sparse_moe.router.layer.weight
    block_sparse_moe.input_linear.weight
    block_sparse_moe.output_linear.weight
    shared_mlp.input_linear.weight
    shared_mlp.output_linear.weight

This matches the convention used by the HF decoder layer's
``load_state_dict``.

Generating the golden fixture
=============================

Run this module directly to regenerate ``reference/golden/layer5.json``
and ``reference/golden/router_top6.json``::

    python reference/granite_layer5.py --write-golden \
        --shard-dir ./output/granite-tiny-q4g64

Use the same ``--seed`` and ``--seq-len`` (defaults: 42 and 4) for the
golden to be reproducible; if you change them you must also update
``tests/reference/test_granite_layer5.py``'s frozen constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public output contract (the keys callers may rely on)
# ---------------------------------------------------------------------------


class Layer5Output(TypedDict):
    """Output of :func:`forward_layer5`."""

    hidden: torch.Tensor  # [batch, seq_len, 1536] float32
    router_top6_expert_ids: torch.Tensor  # [batch, seq_len, 6] int64
    router_top6_weights: torch.Tensor  # [batch, seq_len, 6] float32


# ---------------------------------------------------------------------------
# Granite-tiny config (frozen for the lane; do not change without a
# corresponding change to the WebGPU shaders and the golden fixture)
# ---------------------------------------------------------------------------

GRANITE_TINY_HIDDEN: int = 1536
GRANITE_TINY_NUM_HEADS: int = 12
GRANITE_TINY_NUM_KV_HEADS: int = 4
GRANITE_TINY_HEAD_DIM: int = 128  # 1536 / 12
GRANITE_TINY_GQA_RATIO: int = 3   # num_heads / num_kv_heads
GRANITE_TINY_INTERMEDIATE: int = 512          # expert intermediate
GRANITE_TINY_SHARED_INTERMEDIATE: int = 1024  # shared MLP intermediate
GRANITE_TINY_NUM_EXPERTS: int = 64
GRANITE_TINY_TOP_K: int = 6
GRANITE_TINY_RMS_EPS: float = 1e-5
GRANITE_TINY_RESIDUAL_MULTIPLIER: float = 0.22
GRANITE_TINY_ATTENTION_MULTIPLIER: float = 0.0078125  # = 1 / 128

# Frozen seed/shape for the golden fixture. If you change these, regenerate
# the golden and update tests/reference/test_granite_layer5.py's constants.
FROZEN_SEED: int = 42
FROZEN_BATCH: int = 1
FROZEN_SEQ_LEN: int = 4


# ---------------------------------------------------------------------------
# Primitive ops (kept tiny and self-contained so the WGSL parity test can
# point at them by name)
# ---------------------------------------------------------------------------


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm, fp32 in, ``x.dtype`` out.

    Granite's RMSNorm is the standard Llama-style one::

        y = x * rsqrt(mean(x^2) + eps) * weight

    We upcast to fp32 internally to match HF's reference behavior, then
    cast back to ``x.dtype`` (fp32 here, but written this way so the
    same function is used by an fp16 caller if one ever shows up).
    """
    in_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    y = x_f32 * torch.rsqrt(var + eps)
    return (y * weight.to(torch.float32)).to(in_dtype)


def gqa_attention(
    x: torch.Tensor,
    q_proj_w: torch.Tensor,
    k_proj_w: torch.Tensor,
    v_proj_w: torch.Tensor,
    o_proj_w: torch.Tensor,
    *,
    scale: float,
    num_heads: int = GRANITE_TINY_NUM_HEADS,
    num_kv_heads: int = GRANITE_TINY_NUM_KV_HEADS,
    head_dim: int = GRANITE_TINY_HEAD_DIM,
) -> torch.Tensor:
    """Granite-tiny's GQA attention forward (eager, no KV cache, no RoPE).

    Steps
    -----
    1. ``Q = x @ q_proj_w.T`` shape ``[B, S, num_heads * head_dim]``
    2. ``K = x @ k_proj_w.T`` shape ``[B, S, num_kv_heads * head_dim]``
    3. ``V = x @ v_proj_w.T`` shape ``[B, S, num_kv_heads * head_dim]``
    4. Reshape to ``[B, num_heads, S, head_dim]`` (Q) and
       ``[B, num_kv_heads, S, head_dim]`` (K, V).
    5. Repeat K and V along the head dimension ``num_heads / num_kv_heads``
       times so the matmul is uniform.
    6. Causal mask + softmax with the given scale.
    7. ``attn_out = softmax(...) @ V`` then ``o_proj(attn_out)``.

    Output shape: ``[B, S, hidden_size]`` (1536 for Granite-tiny).
    """
    bsz, seq, _ = x.shape
    q = F.linear(x, q_proj_w).view(bsz, seq, num_heads, head_dim).transpose(1, 2)
    k = F.linear(x, k_proj_w).view(bsz, seq, num_kv_heads, head_dim).transpose(1, 2)
    v = F.linear(x, v_proj_w).view(bsz, seq, num_kv_heads, head_dim).transpose(1, 2)
    # GQA: repeat K and V along the head dim. ``repeat_interleave`` is
    # exactly what HF's ``repeat_kv`` does (named differently here so the
    # WGSL parity test can match by literal symbol).
    gqa_ratio = num_heads // num_kv_heads
    k = k.repeat_interleave(gqa_ratio, dim=1)
    v = v.repeat_interleave(gqa_ratio, dim=1)

    # Scaled dot product with causal mask.
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(
        torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
    )
    attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
    # Softmax in fp32 for stability; HF does this too.
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
    attn_out = torch.matmul(attn_weights, v)  # [B, num_heads, S, head_dim]
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq, num_heads * head_dim)
    return F.linear(attn_out, o_proj_w)


def moe_router(
    x: torch.Tensor, router_weight: torch.Tensor, top_k: int = GRANITE_TINY_TOP_K
) -> tuple[torch.Tensor, torch.Tensor]:
    """Granite-tiny's top-K router.

    Returns
    -------
    top_k_ids:
        ``[batch*seq, top_k]`` int64
    top_k_weights:
        ``[batch*seq, top_k]`` float (same dtype as ``x``), softmax-normalized
        across the top-K dimension (so they sum to 1.0 per token).
    """
    # The HF router is a single ``nn.Linear`` whose weight is stored as
    # ``[num_experts, hidden]``. Its forward computes ``x @ weight.T``.
    router_logits = F.linear(x, router_weight).float()  # [B*S, E]
    top_k_logits, top_k_ids = router_logits.topk(top_k, dim=1)
    top_k_weights = F.softmax(top_k_logits, dim=1).type_as(x)
    return top_k_ids, top_k_weights


def single_expert_forward(
    x: torch.Tensor,
    input_linear_w: torch.Tensor,  # [intermediate * 2, hidden]
    output_linear_w: torch.Tensor,  # [hidden, intermediate]
) -> torch.Tensor:
    """One expert's forward: silu(gate) * up, then down.

    Matches the ``GraniteMoeHybridParallelExperts`` per-expert call:

    - ``input_linear_w[e] @ x`` → ``[..., intermediate * 2]``
    - chunk into ``gate`` and ``up`` along the last dim
    - ``silu(gate) * up`` → ``[..., intermediate]``
    - ``output_linear_w[e] @ h`` → ``[..., hidden]``
    """
    h = F.linear(x, input_linear_w)  # [..., 2*intermediate]
    gate, up = h.chunk(2, dim=-1)
    h = F.silu(gate) * up
    return F.linear(h, output_linear_w)


def shared_mlp_forward(
    x: torch.Tensor,
    input_linear_w: torch.Tensor,  # [2*shared_intermediate, hidden]
    output_linear_w: torch.Tensor,  # [hidden, shared_intermediate]
) -> torch.Tensor:
    """Granite's shared MLP (always-on, in addition to the MoE experts)."""
    h = F.linear(x, input_linear_w)
    gate, up = h.chunk(2, dim=-1)
    h = F.silu(gate) * up
    return F.linear(h, output_linear_w)


# ---------------------------------------------------------------------------
# The full layer-5 forward
# ---------------------------------------------------------------------------


def forward_layer5(
    hidden_in: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    rms_norm_eps: float = GRANITE_TINY_RMS_EPS,
    residual_multiplier: float = GRANITE_TINY_RESIDUAL_MULTIPLIER,
    attention_multiplier: float = GRANITE_TINY_ATTENTION_MULTIPLIER,
    num_heads: int = GRANITE_TINY_NUM_HEADS,
    num_kv_heads: int = GRANITE_TINY_NUM_KV_HEADS,
    head_dim: int = GRANITE_TINY_HEAD_DIM,
    top_k: int = GRANITE_TINY_TOP_K,
) -> Layer5Output:
    """Pure-PyTorch forward for Granite-tiny layer 5.

    Parameters
    ----------
    hidden_in:
        ``[batch, seq_len, 1536]`` fp32 input hidden state. The function
        will upcast to fp32 internally (no-op here) and return fp32
        output, so the caller can keep everything in one dtype.
    weights:
        Layer-5 weight dict, with ``model.layers.5.`` already stripped.
        See the module docstring for the exact key set.
    rms_norm_eps, residual_multiplier, attention_multiplier:
        Granite-tiny hyperparameters. Defaults match the deployed config;
        pass overrides only for testing.

    Returns
    -------
    dict with keys
        ``hidden`` (fp32 ``[B, S, 1536]``),
        ``router_top6_expert_ids`` (int64 ``[B, S, 6]``),
        ``router_top6_weights`` (fp32 ``[B, S, 6]``).
    """
    bsz, seq, hidden = hidden_in.shape
    assert hidden == GRANITE_TINY_HIDDEN, (
        f"hidden dim {hidden} != Granite-tiny {GRANITE_TINY_HIDDEN}"
    )
    assert bsz == 1, (
        f"this reference assumes batch=1 (Phase 1 demo is single-stream); "
        f"got batch={bsz}"
    )
    x = hidden_in.to(torch.float32)

    # All matmul ops in this forward require matching dtypes. The shard
    # weights are stored as fp16 in safetensors (matching the deployed
    # Granite checkpoint); we upcast to fp32 here so the reference does
    # its math at full precision and so the WGSL parity test can use the
    # fp32 tolerance (1e-5) instead of the fp16 tolerance (1e-3). The
    # WGSL side runs in fp16 on the GPU and is judged against the
    # golden JSON with the spec's 1e-3 / 1e-5 tolerances per spec FR-3.
    weights = {k: v.to(torch.float32) for k, v in weights.items()}

    # 1. Pre-attention RMSNorm.
    x_norm = rmsnorm(x, weights["input_layernorm.weight"], rms_norm_eps)

    # 2. GQA attention (eager, no RoPE, no KV cache).
    attn_out = gqa_attention(
        x_norm,
        weights["self_attn.q_proj.weight"],
        weights["self_attn.k_proj.weight"],
        weights["self_attn.v_proj.weight"],
        weights["self_attn.o_proj.weight"],
        scale=attention_multiplier,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # 3. First residual with multiplier.
    residual1 = x
    x = residual1 + attn_out * residual_multiplier

    # 4. Post-attention RMSNorm.
    x_norm2 = rmsnorm(x, weights["post_attention_layernorm.weight"], rms_norm_eps)

    # 5. Router: top-K ids and softmax weights.
    # The router is a single Linear over the hidden dim; it does not see
    # the sequence axis separately.
    router_w = weights["block_sparse_moe.router.layer.weight"]
    router_logits = F.linear(x_norm2, router_w)  # [B, S, E]
    bs = bsz * seq
    flat_logits = router_logits.view(bs, -1).float()
    topk_logits, topk_ids = flat_logits.topk(top_k, dim=1)  # [BS, K], [BS, K]
    topk_weights = F.softmax(topk_logits, dim=1).to(torch.float32)  # [BS, K]
    # Reshape back to [B, S, K] for the public output.
    topk_ids = topk_ids.view(bsz, seq, top_k)
    topk_weights = topk_weights.view(bsz, seq, top_k)

    # 6. MoE output: weighted sum of the 6 routed expert outputs.
    #    The browser receives the 6 expert outputs from the coordinator and
    #    does the same weighted sum on the GPU. We compute it here as the
    #    reference for the WGSL parity test.
    moe_out = torch.zeros_like(x_norm2)
    expert_input_w = weights["block_sparse_moe.input_linear.weight"]  # [E, 2I, H]
    expert_output_w = weights["block_sparse_moe.output_linear.weight"]  # [E, H, I]
    # Naive per-token loop. seq_len is small in the demo (1-4 tokens) and
    # we only need this for the reference / parity test, not the hot path.
    for s in range(seq):
        for k in range(top_k):
            eid = int(topk_ids[0, s, k])
            w = float(topk_weights[0, s, k])
            x_in = x_norm2[0, s, :]  # [H]
            h = single_expert_forward(
                x_in, expert_input_w[eid], expert_output_w[eid]
            )  # [H]
            moe_out[0, s, :] = moe_out[0, s, :] + w * h

    # 7. Shared MLP.
    shared_out = shared_mlp_forward(
        x_norm2,
        weights["shared_mlp.input_linear.weight"],
        weights["shared_mlp.output_linear.weight"],
    )

    # 8. Final residual with multiplier.
    residual2 = x
    out = residual2 + (moe_out + shared_out) * residual_multiplier

    return Layer5Output(
        hidden=out,
        router_top6_expert_ids=topk_ids,
        router_top6_weights=topk_weights,
    )


# ---------------------------------------------------------------------------
# Golden-fixture writer
# ---------------------------------------------------------------------------


def write_golden(
    shard_dir: Path,
    out_dir: Path,
    *,
    seed: int = FROZEN_SEED,
    seq_len: int = FROZEN_SEQ_LEN,
) -> tuple[Path, Path]:
    """Run the reference once on a frozen input and dump the golden JSONs.

    Returns the paths to the two written files
    (``golden/layer5.json`` and ``golden/router_top6.json``).
    """
    from safetensors.torch import load_file  # local import to keep top of file clean

    shard_path = shard_dir / "shard_layers_0_9.safetensors"
    if not shard_path.exists():
        raise FileNotFoundError(f"shard not found at {shard_path}")
    sd = load_file(str(shard_path))
    weights = {
        k.removeprefix("model.layers.5."): v
        for k, v in sd.items()
        if k.startswith("model.layers.5.")
    }

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((FROZEN_BATCH, seq_len, GRANITE_TINY_HIDDEN), generator=g, dtype=torch.float32)
    out = forward_layer5(x, weights)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Full golden: hidden + router
    full_payload = {
        "model": "ibm-granite/granite-4.0-h-tiny",
        "layer": 5,
        "frozen_input": {
            "seed": seed,
            "shape": list(x.shape),
            "dtype": "float32",
            "sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        },
        "input_sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        "expected_hidden": out["hidden"].cpu().numpy().tolist(),
        "expected_router_top6_expert_ids": out["router_top6_expert_ids"].cpu().numpy().tolist(),
        "expected_router_top6_weights": out["router_top6_weights"].cpu().numpy().tolist(),
        "tolerances": {
            "hidden_abs": 1e-3,            # fp16
            "hidden_abs_fp32": 1e-5,       # fp32 fallback
            "router_ids": "exact",
            "router_weights_abs": 1e-5,
        },
    }
    full_path = out_dir / "layer5.json"
    with full_path.open("w") as f:
        json.dump(full_payload, f)

    # Router-only golden: smaller, for the parity test that doesn't need
    # the full hidden state (the client-side dispatch only depends on
    # which 6 experts + their weights).
    router_payload = {
        "model": "ibm-granite/granite-4.0-h-tiny",
        "layer": 5,
        "frozen_input": full_payload["frozen_input"],
        "input_sha256": full_payload["input_sha256"],
        "expected_router_top6_expert_ids": full_payload["expected_router_top6_expert_ids"],
        "expected_router_top6_weights": full_payload["expected_router_top6_weights"],
    }
    router_path = out_dir / "router_top6.json"
    with router_path.open("w") as f:
        json.dump(router_payload, f)

    return full_path, router_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("./output/granite-tiny-q4g64"),
        help="Path to the deployed granite-tiny layer-group shard directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "golden",
        help="Directory to write the golden JSON files into.",
    )
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--seq-len", type=int, default=FROZEN_SEQ_LEN)
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="Run the forward once on a frozen input and write the golden JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.write_golden:
        print(
            "No action specified. Pass --write-golden to regenerate the golden JSONs.",
            file=sys.stderr,
        )
        return 1
    full, router = write_golden(
        args.shard_dir, args.out_dir, seed=args.seed, seq_len=args.seq_len
    )
    print(f"wrote {full}")
    print(f"wrote {router}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
