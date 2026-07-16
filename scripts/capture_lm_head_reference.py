"""Capture an ``lm_head`` reference fixture for the correctness test.

WHAT THIS SCRIPT PRODUCES
-------------------------

A single ``.npz`` file at::

    scripts/__tests__/fixtures/lm_head_reference.npz

containing four arrays:

  - ``hidden`` -- the input ``[1, 1536]`` fp16 tensor.
  - ``lm_head_weight`` -- the ``[vocab, 1536]`` fp16 weight that produced the
    fixture's logits.
  - ``final_layernorm_weight`` -- the ``[1536]`` fp16 layernorm gain, or all
    ones if no layernorm.
  - ``top5_token_ids`` -- the top-5 token ids by logit (highest first).
  - ``top5_logits`` -- the matching logit values.
  - ``meta`` -- a small JSON blob with at least the keys::
        {
          "model_id": str,
          "synthetic_seed": int | null,
          "prompt": str | null,
          "vocab": int,
          "hidden": int,
          "captured_at": str (ISO-8601),
          "source": "synthetic" | "granite_real"
        }

USAGE -- SYNTHETIC (works on any machine; this is what CI uses)
---------------------------------------------------------------

    python scripts/capture_lm_head_reference.py --synthetic \
        --out scripts/__tests__/fixtures/lm_head_reference.npz

The synthetic mode uses a deterministic RNG seed so the fixture is
reproducible. The correctness test exercises the *shape* and the *bytes-clean
matmul path*, not Granite semantics. The fixture's ``source`` field will read
``"synthetic"``.

USAGE -- REAL GRANITE-TINY (operator only; needs S3 creds + ~30 GB disk)
------------------------------------------------------------------------

    HEAD_S3_BUCKET=... HEAD_S3_ACCESS_KEY=... HEAD_S3_SECRET_KEY=... \
    HEAD_S3_KEY=shard_head.safetensors HEAD_LOCAL_PATH=/tmp/shard_head.safetensors \
    python scripts/capture_lm_head_reference.py --granite \
        --shard-dir ./output/granite-tiny-q4g64 \
        --prompt "The capital of France is" \
        --out scripts/__tests__/fixtures/lm_head_reference.npz

The Granite mode runs ``verify_granite_handler.py``-style end-to-end up to
the lm_head boundary, captures the (hidden, top_5_logits) pair, and writes a
fixture whose ``source`` field reads ``"granite_real"``. After capture, run
the correctness test once to confirm everything is wired.

Re-running this script overwrites the fixture in place. Commit the
.npz to git so CI does not need to re-capture.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import torch


HIDDEN = 1536
DEFAULT_SYNTHETIC_VOCAB = 64
DEFAULT_GRANITE_VOCAB = 49152  # Granite-tiny actual vocab size
DEFAULT_SEED = 20260606  # easy to spot in fixture meta


def _topk_from_logits(logits: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 2:
        last = logits[-1]
    else:
        last = logits[0, -1] if logits.dim() == 3 else logits
    last = last.float()
    topk = torch.topk(last, k=k)
    return topk.indices, topk.values


def capture_synthetic(seed: int = DEFAULT_SEED) -> dict:
    """Generate a fully deterministic reference using a small random head."""

    g = torch.Generator()
    g.manual_seed(seed)
    lm_head_w = torch.randn(
        DEFAULT_SYNTHETIC_VOCAB, HIDDEN, generator=g, dtype=torch.float32,
    ).to(torch.float16)
    final_ln_w = torch.full((HIDDEN,), 0.7, dtype=torch.float16)
    hidden = torch.randn(1, HIDDEN, generator=g, dtype=torch.float32).to(torch.float16)

    # Capture logits through the REAL HeadModule so the fixture reflects exactly
    # what the runtime produces. HeadModule.forward keeps the lm_head weight in
    # fp16 (a deliberate memory optimization) and does the matmul in fp16, so a
    # reference computed in fp32 here would drift ~1e-2 from the runtime at
    # these logit magnitudes and make the 1e-3 correctness tolerance unusable.
    # Computing via HeadModule keeps that tolerance a meaningful regression
    # guard. See HeadModule.forward in expert_coordinator.py.
    try:
        from scripts.expert_coordinator import HeadModule
    except ModuleNotFoundError:
        # Allow running this file as a standalone script (cwd-relative import)
        # in addition to the package import pytest's conftest enables.
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
        from scripts.expert_coordinator import HeadModule

    head = HeadModule(lm_head_w, final_layernorm_weight=final_ln_w, layernorm_eps=1e-5)
    logits = head.forward(hidden)
    top5_ids, top5_vals = _topk_from_logits(logits.float())

    return {
        "hidden": hidden.numpy(),
        "lm_head_weight": lm_head_w.numpy(),
        "final_layernorm_weight": final_ln_w.numpy(),
        "top5_token_ids": top5_ids.numpy(),
        "top5_logits": top5_vals.numpy(),
        "meta": json.dumps({
            "source": "synthetic",
            "synthetic_seed": seed,
            "model_id": None,
            "prompt": None,
            "vocab": DEFAULT_SYNTHETIC_VOCAB,
            "hidden": HIDDEN,
            "captured_at": _dt.datetime.utcnow().isoformat() + "Z",
        }),
    }


def capture_granite_real(
    shard_dir: Path,
    prompt: str,
    head_local_path: Path,
) -> dict:
    """Capture against the real Granite-tiny shards. Operator-only path.

    Requires the same env vars as the coordinator's HEAD_S3_* download.
    """

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer

    # Lazy import so synthetic mode does not need this code path on PATH.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.architectures import LayerShardSpec, get_handler  # type: ignore[import-not-found]
    from scripts.expert_coordinator import (  # type: ignore[no-redef]
        CoordinatorConfig, _ensure_head_shard, load_head_from_safetensors,
    )

    manifest = json.loads((shard_dir / "manifest.json").read_text())
    model_id = manifest["model_id"]
    layer_groups = [tuple(g) for g in manifest["layer_groups"]]

    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"
    handler = get_handler("granitemoehybrid")
    tok = AutoTokenizer.from_pretrained(model_id)
    input_ids = tok.encode(prompt, return_tensors="pt")
    device = torch.device("cpu")

    embed_state = load_file(str(shard_dir / "shard_embed.safetensors"))
    embed_mod = handler.build_embed(config, embed_state, device)
    del embed_state
    hidden = handler.forward_embed(embed_mod, input_ids)
    del embed_mod

    for (start, end) in layer_groups:
        shard_state = load_file(str(shard_dir / f"shard_layers_{start}_{end}.safetensors"))
        layer_mod = handler.build_layers(
            config, shard_state, LayerShardSpec(layer_start=start, layer_end=end), device,
        )
        del shard_state
        hidden = handler.forward_layers(layer_mod, hidden)
        del layer_mod

    # Pull head shard via coordinator's path so the fixture matches deploy.
    head_cfg = CoordinatorConfig()
    head_cfg.head_local_path = head_local_path
    _ensure_head_shard(head_cfg)
    head_mod = load_head_from_safetensors(head_local_path)

    last_hidden = hidden[0, -1:, :].to(torch.float16)  # [1, hidden]
    logits = head_mod.forward(last_hidden)
    top5_ids, top5_vals = _topk_from_logits(logits)

    ln_w = (
        head_mod.final_layernorm_weight.to(torch.float16).numpy()
        if head_mod.final_layernorm_weight is not None
        else np.ones(HIDDEN, dtype=np.float16)
    )
    return {
        "hidden": last_hidden.numpy(),
        "lm_head_weight": head_mod.lm_head_weight.to(torch.float16).numpy(),
        "final_layernorm_weight": ln_w,
        "top5_token_ids": top5_ids.numpy(),
        "top5_logits": top5_vals.numpy(),
        "meta": json.dumps({
            "source": "granite_real",
            "synthetic_seed": None,
            "model_id": model_id,
            "prompt": prompt,
            "vocab": int(head_mod.vocab),
            "hidden": int(head_mod.hidden),
            "captured_at": _dt.datetime.utcnow().isoformat() + "Z",
        }),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="output .npz path")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate a synthetic reference (no shards)")
    parser.add_argument("--granite", action="store_true",
                        help="capture against real Granite-tiny shards")
    parser.add_argument("--shard-dir", type=Path,
                        default=Path("./output/granite-tiny-q4g64"))
    parser.add_argument("--head-local-path", type=Path,
                        default=Path("/tmp/shard_head.safetensors"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.synthetic == args.granite:
        parser.error("specify exactly one of --synthetic or --granite")

    if args.synthetic:
        bundle = capture_synthetic(seed=args.seed)
    else:
        bundle = capture_granite_real(
            shard_dir=args.shard_dir,
            prompt=args.prompt,
            head_local_path=args.head_local_path,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bundle)
    print(f"wrote fixture: {args.out}")
    print("meta:", bundle["meta"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
