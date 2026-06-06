"""
Performance + capability test for the running shard swarm.

Assumes scripts/run_local_swarm.py is already running on ports 9000-9005.

Tests:
  1. Multiple prompts -> generate N tokens each, report tok/s
  2. Context length probe -> push longer inputs until failure or slowdown
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


SERVER_PORTS = [9000, 9001, 9002, 9003, 9004, 9005]
SERVER_NAMES = ["embed", "layers_0_3", "layers_4_10", "layers_11_17", "layers_18_21", "head"]


def pack_tensor(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    shape = list(arr.shape)
    dtype_bytes = b"float32"
    buf = struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<I", d)
    buf += struct.pack("<I", len(dtype_bytes))
    buf += dtype_bytes
    buf += arr.tobytes()
    return buf


def unpack_tensor(raw: bytes) -> np.ndarray:
    off = 0
    (ndim,) = struct.unpack_from("<I", raw, off); off += 4
    shape = []
    for _ in range(ndim):
        (d,) = struct.unpack_from("<I", raw, off); off += 4
        shape.append(d)
    (dl,) = struct.unpack_from("<I", raw, off); off += 4
    dtype_str = raw[off:off + dl].decode(); off += dl
    np_dtype = {"float32": np.float32, "float16": np.float16}[dtype_str]
    return np.frombuffer(raw[off:], dtype=np_dtype).reshape(shape).copy()


def forward_through_pipeline(input_ids: np.ndarray, timeout: float = 120.0) -> tuple[np.ndarray, dict]:
    """Run a single forward pass through all shards. Returns (logits, per_stage_ms)."""
    current = input_ids.astype(np.float32)
    timings: dict[str, float] = {}

    for port, name in zip(SERVER_PORTS, SERVER_NAMES):
        url = f"http://localhost:{port}/forward"
        body = pack_tensor(current)
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        t0 = time.perf_counter()
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read()
        elapsed = (time.perf_counter() - t0) * 1000
        timings[name] = elapsed
        current = unpack_tensor(raw)

    return current, timings


def sample_next_token(logits: np.ndarray, temperature: float = 0.0) -> int:
    """Greedy or temperature sample from the last position's logits."""
    last_logits = logits[0, -1, :]
    if temperature <= 0:
        return int(np.argmax(last_logits))
    probs = np.exp(last_logits / temperature)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate(tokenizer, prompt: str, n_tokens: int, temperature: float = 0.0) -> dict:
    """Generate n_tokens autoregressively. Returns timing + output dict."""
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    prompt_len = input_ids.shape[1]

    prefill_t0 = time.perf_counter()
    logits, prefill_timings = forward_through_pipeline(input_ids)
    prefill_ms = (time.perf_counter() - prefill_t0) * 1000

    generated_ids = []
    decode_times = []
    current_ids = input_ids.copy()

    for i in range(n_tokens):
        next_tok = sample_next_token(logits, temperature)
        generated_ids.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break

        # Append + re-run full prefix (no KV cache in this swarm yet)
        current_ids = np.concatenate(
            [current_ids, np.array([[next_tok]], dtype=current_ids.dtype)],
            axis=1,
        )
        t0 = time.perf_counter()
        logits, _ = forward_through_pipeline(current_ids)
        decode_times.append((time.perf_counter() - t0) * 1000)

    total_tokens = len(generated_ids)
    decode_total_ms = sum(decode_times)
    decode_avg_ms = decode_total_ms / max(1, len(decode_times))
    tok_per_sec = (total_tokens * 1000) / (prefill_ms + decode_total_ms) if total_tokens > 0 else 0.0

    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    safe_completion = completion.encode("ascii", errors="replace").decode("ascii")

    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "completion": safe_completion,
        "n_generated": total_tokens,
        "prefill_ms": prefill_ms,
        "prefill_stages": prefill_timings,
        "decode_avg_ms": decode_avg_ms,
        "decode_total_ms": decode_total_ms,
        "tok_per_sec": tok_per_sec,
    }


def context_probe(tokenizer, lengths: list[int], timeout: float = 120.0) -> list[dict]:
    """Probe how long an input the pipeline can handle in a single forward."""
    results = []
    # Use a repeating filler that tokenizes predictably
    filler = "The quick brown fox jumps over the lazy dog. "

    for target_len in lengths:
        # Repeat filler until we exceed target_len tokens, then truncate
        text = (filler * ((target_len // 10) + 2))
        ids = tokenizer.encode(text, return_tensors="np")
        if ids.shape[1] > target_len:
            ids = ids[:, :target_len]
        actual_len = ids.shape[1]

        print(f"  [ctx {target_len:>5}]  actual_tokens={actual_len:>5}  ", end="", flush=True)
        try:
            t0 = time.perf_counter()
            logits, stage_timings = forward_through_pipeline(ids, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"OK  total={elapsed_ms:>7.0f}ms  out_shape={list(logits.shape)}")
            results.append({
                "target_len": target_len,
                "actual_len": actual_len,
                "total_ms": elapsed_ms,
                "stages": stage_timings,
                "ok": True,
            })
        except Exception as e:
            err_str = str(e).splitlines()[0][:120]
            print(f"FAIL  {err_str}")
            results.append({
                "target_len": target_len,
                "actual_len": actual_len,
                "error": err_str,
                "ok": False,
            })
            break  # stop probing once we hit a failure

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--gen-tokens", type=int, default=20,
                        help="tokens to generate per prompt")
    parser.add_argument("--skip-context-probe", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("HiveMind Local Swarm — Performance Test")
    print("=" * 72)
    print(f"Model: {args.model_id}")
    print(f"Servers: {len(SERVER_PORTS)} shards on ports {SERVER_PORTS[0]}-{SERVER_PORTS[-1]}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print(f"Tokenizer ready (vocab_size={tokenizer.vocab_size})")
    print()

    # --- Test 1: generation prompts ---
    prompts = [
        "The capital of France is",
        "Once upon a time in a small village,",
        "def fibonacci(n):",
        "Q: What is 2 + 2?\nA:",
        "The three primary colors are",
    ]

    print(f"--- Test 1: Generate {args.gen_tokens} tokens per prompt ({len(prompts)} prompts) ---")
    print()
    all_results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] '{prompt}'")
        r = generate(tokenizer, prompt, args.gen_tokens)
        all_results.append(r)
        print(f"  -> '{r['completion']}'")
        print(f"  prompt_tokens={r['prompt_len']}  generated={r['n_generated']}")
        print(f"  prefill={r['prefill_ms']:.0f}ms  decode_avg={r['decode_avg_ms']:.0f}ms/tok  "
              f"throughput={r['tok_per_sec']:.2f} tok/s")
        # Show per-stage prefill breakdown for the first prompt
        if i == 1:
            stages = r["prefill_stages"]
            print(f"  prefill stages: " + "  ".join(
                f"{name}={ms:.0f}ms" for name, ms in stages.items()
            ))
        print()

    # --- Aggregate stats ---
    print("--- Aggregate ---")
    avg_throughput = sum(r["tok_per_sec"] for r in all_results) / len(all_results)
    avg_decode = sum(r["decode_avg_ms"] for r in all_results) / len(all_results)
    avg_prefill = sum(r["prefill_ms"] for r in all_results) / len(all_results)
    print(f"  Avg prefill:    {avg_prefill:.0f} ms")
    print(f"  Avg decode/tok: {avg_decode:.0f} ms")
    print(f"  Avg throughput: {avg_throughput:.2f} tok/s")
    print()

    # --- Test 2: max context probe ---
    if not args.skip_context_probe:
        print("--- Test 2: Max context probe (single forward pass) ---")
        # TinyLlama's trained context is 2048; we'll probe up to and past it
        probe_lengths = [128, 256, 512, 1024, 2048, 3072, 4096]
        ctx_results = context_probe(tokenizer, probe_lengths)
        print()
        max_ok = max((r["actual_len"] for r in ctx_results if r.get("ok")), default=0)
        print(f"  Max successful context: {max_ok} tokens")

    print()
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
