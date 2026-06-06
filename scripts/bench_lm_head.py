"""Microbenchmark for the coordinator's ``POST /lm_head`` endpoint.

Issues ``--iters`` (default 100) requests at ``[1, 1536]`` fp16 against a
running coordinator and prints p50 / p95 / p99 latency plus a pass/fail
against NFR-1 (< 50 ms p50, < 60 ms p99).

This is an operational tool, not a CI test. Run after deploy to confirm the
service hits its budget; rerun after any change to the lm_head path.

Example::

    python scripts/bench_lm_head.py --url http://localhost:8080 --iters 100

Against the deployed service::

    python scripts/bench_lm_head.py \
        --url https://expert-coordinator.up.railway.app --iters 200
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
from typing import Any

import aiohttp
import numpy as np


HIDDEN = 1536


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    if f + 1 < len(s):
        return s[f] + (s[f + 1] - s[f]) * (k - f)
    return s[f]


async def run_bench(url: str, iters: int, warmup: int, n_tokens: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed=0)
    hidden = rng.standard_normal((n_tokens, HIDDEN)).astype(np.float16)
    body = {
        "hidden_b64": base64.b64encode(hidden.tobytes()).decode("ascii"),
        "shape": [n_tokens, HIDDEN],
        "dtype": "fp16",
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup
        for _ in range(warmup):
            async with session.post(f"{url}/lm_head", json=body) as r:
                await r.read()

        client_ms: list[float] = []
        server_ms: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            async with session.post(f"{url}/lm_head", json=body) as r:
                data = await r.json()
            client_ms.append((time.perf_counter() - t0) * 1000.0)
            server_ms.append(float(data.get("ms", -1)))
    return {
        "client_ms": client_ms,
        "server_ms": server_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True,
                        help="base URL of the coordinator (no trailing slash)")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--n-tokens", type=int, default=1)
    args = parser.parse_args()

    print(f"benching POST {args.url}/lm_head, "
          f"iters={args.iters}, warmup={args.warmup}, n_tokens={args.n_tokens}")
    result = asyncio.run(
        run_bench(args.url, args.iters, args.warmup, args.n_tokens)
    )

    for label, samples in (("client", result["client_ms"]),
                            ("server", result["server_ms"])):
        if any(s < 0 for s in samples):
            print(f"  {label}: incomplete samples")
            continue
        p50 = _percentile(samples, 0.50)
        p95 = _percentile(samples, 0.95)
        p99 = _percentile(samples, 0.99)
        mn = min(samples)
        mx = max(samples)
        mean = statistics.mean(samples)
        print(
            f"  {label}: n={len(samples)} "
            f"min={mn:.1f}ms mean={mean:.1f}ms p50={p50:.1f}ms "
            f"p95={p95:.1f}ms p99={p99:.1f}ms max={mx:.1f}ms"
        )

    # NFR-1 check against server-reported latency.
    server = result["server_ms"]
    if all(s >= 0 for s in server):
        p50 = _percentile(server, 0.50)
        p99 = _percentile(server, 0.99)
        pass_p50 = p50 < 50.0
        pass_p99 = p99 < 60.0
        verdict = "PASS" if (pass_p50 and pass_p99) else "FAIL"
        print(
            f"\nNFR-1: p50<50ms? {pass_p50}  p99<60ms? {pass_p99}  -> {verdict}"
        )
        return 0 if pass_p50 and pass_p99 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
