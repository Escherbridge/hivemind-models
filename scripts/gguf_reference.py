"""
Run reference inference against the local Granite GGUF.

This gives us the ground-truth top-k predictions that our sharded pipeline
must match (within reason — Q4_K_M is lossy on its own, but we want the
sharded run to agree with single-process llama.cpp on the same GGUF).

Usage:
    python scripts/gguf_reference.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llama_cpp import Llama


DEFAULT_GGUF = Path(
    "C:/Users/atooz/.lmstudio/models/lmstudio-community/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--n-tokens", type=int, default=8,
                        help="number of tokens to generate after the prompt")
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5,
                        help="top-k logits to show for the *first* generated token")
    parser.add_argument("--verbose", action="store_true",
                        help="show llama.cpp's own loading + inference logs")
    args = parser.parse_args()

    if not args.gguf.exists():
        raise SystemExit(f"GGUF not found at {args.gguf}")

    print(f"Loading GGUF: {args.gguf}")
    llm = Llama(
        model_path=str(args.gguf),
        n_ctx=args.n_ctx,
        n_threads=None,           # auto
        logits_all=True,          # we need per-step logits for top-k probe
        verbose=args.verbose,
    )

    # Metadata sanity
    print("\n--- model metadata ---")
    md = llm.metadata
    keys_of_interest = [
        k for k in md
        if any(s in k.lower() for s in (
            "general.architecture", "general.name", "general.parameter",
            "block_count", "embedding_length", "expert", "moe",
            "head_count", "layer", "context_length",
        ))
    ]
    for k in sorted(keys_of_interest):
        print(f"  {k}: {md[k]}")

    print("\n--- reference inference ---")
    print(f"prompt: {args.prompt!r}")

    # Use the high-level completion API to drive the model — driving llama.cpp's
    # low-level eval() directly gave us garbage indices off the end of the vocab,
    # so we let llama-cpp-python orchestrate the run and then pull logits from
    # its scores array after the fact.
    ids = llm.tokenize(args.prompt.encode("utf-8"))
    print(f"prompt tokens ({len(ids)}): {ids}")

    out = llm(
        args.prompt,
        max_tokens=args.n_tokens,
        temperature=0.0,
        top_k=1,
        logprobs=args.top_k,
        echo=False,
    )

    import numpy as np

    # llama-cpp-python returns the per-token top logprobs in choices[0].logprobs
    first_logprobs = out["choices"][0]["logprobs"]["top_logprobs"][0]
    print(f"\nTop-{args.top_k} predictions for the NEXT token (after the prompt):")
    # first_logprobs is dict {token_string: logprob}, ordered by rank
    for tok_str, lp in list(first_logprobs.items())[: args.top_k]:
        safe = tok_str.encode("ascii", errors="replace").decode("ascii")
        print(f"  {float(lp):10.4f}  {safe!r}")

    completion = out["choices"][0]["text"]
    safe = completion.encode("ascii", errors="replace").decode("ascii")
    print("\nGreedy continuation:")
    print(f"  {args.prompt}{safe}")

    # Persist the reference for the diff harness
    out_path = Path("output/granite_reference.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "gguf": str(args.gguf),
        "prompt": args.prompt,
        "prompt_token_ids": list(map(int, ids)),
        "top_k": int(args.top_k),
        "top_k_tokens": [
            {"token": tok, "logprob": float(lp)}
            for tok, lp in list(first_logprobs.items())[: args.top_k]
        ],
        "greedy_completion_text": completion,
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReference saved to {out_path}")


if __name__ == "__main__":
    main()
