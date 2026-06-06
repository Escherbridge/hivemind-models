"""
Measure latency from this machine to each Railway-hosted expert region.

Holds a persistent WebSocket connection per region (warm pool, no reconnect
between samples) and runs three measurement passes:

  1. JSON ping/pong  -- pure round-trip with ~no payload
  2. Binary forward  -- realistic MoE expert dispatch on small / medium / large
                        token batches
  3. Sustained ping  -- 30s of pings at 200ms cadence to verify the connection
                        stays warm without being reset by intermediate proxies

Output: output/region_latency.json with per-region p50/p95/stdev for each pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import struct
import time
from pathlib import Path

import numpy as np
import websockets


# Match the wire protocol from expert_railway_server.py
_HEADER = struct.Struct("<BHIIB")
HIDDEN = 1536  # Granite-tiny hidden size


def pack_forward(expert_id: int, tokens: np.ndarray) -> bytes:
    n, h = tokens.shape
    return _HEADER.pack(1, expert_id, n, h, 1) + tokens.tobytes()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    if f + 1 < len(values):
        return values[f] + (values[f + 1] - values[f]) * (k - f)
    return values[f]


def summarize(label: str, values: list[float]) -> dict:
    return {
        "label": label,
        "n": len(values),
        "min_ms": min(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
        "mean_ms": statistics.mean(values) if values else None,
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


async def open_connection(url: str):
    """Open WS and read the 'ready' frame; return (ws, ready_payload)."""
    ws = await websockets.connect(
        url,
        max_size=2**24,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=15,
    )
    hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
    return ws, hello


async def measure_json_ping(ws, n: int) -> list[float]:
    samples: list[float] = []
    for i in range(n):
        nonce = f"p{i}"
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "ping", "nonce": nonce}))
        resp = await ws.recv()
        t1 = time.perf_counter()
        msg = json.loads(resp)
        if msg.get("echo") != nonce:
            raise RuntimeError(f"pong nonce mismatch: {msg}")
        samples.append((t1 - t0) * 1000)
        await asyncio.sleep(0.02)  # back off so we measure RTT, not throughput
    return samples


async def measure_forward(ws, expert_id: int, n_tokens: int, n_iters: int) -> list[float]:
    samples: list[float] = []
    for _ in range(n_iters):
        x = np.random.randn(n_tokens, HIDDEN).astype(np.float16)
        frame = pack_forward(expert_id, x)
        t0 = time.perf_counter()
        await ws.send(frame)
        resp = await ws.recv()
        t1 = time.perf_counter()
        if not isinstance(resp, (bytes, bytearray)) or resp[0] != 1:
            raise RuntimeError(f"forward error: {resp[:200]}")
        samples.append((t1 - t0) * 1000)
        await asyncio.sleep(0.02)
    return samples


async def measure_sustained_ping(ws, duration_s: float, interval_s: float) -> list[float]:
    """Verify the connection stays warm. Returns RTT samples; should not spike
    upward if intermediate proxies are not resetting us."""
    samples: list[float] = []
    deadline = time.time() + duration_s
    i = 0
    while time.time() < deadline:
        nonce = f"k{i}"
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "ping", "nonce": nonce}))
        resp = await ws.recv()
        t1 = time.perf_counter()
        if json.loads(resp).get("echo") != nonce:
            raise RuntimeError("nonce mismatch in sustained ping")
        samples.append((t1 - t0) * 1000)
        i += 1
        # Sleep so cadence is roughly interval_s, accounting for RTT.
        slack = interval_s - (t1 - t0)
        if slack > 0:
            await asyncio.sleep(slack)
    return samples


async def measure_region(name: str, url: str, args) -> dict:
    print(f"\n=== {name} ({url}) ===")
    try:
        ws, hello = await open_connection(url)
    except Exception as e:
        print(f"  CONNECT FAILED: {e}")
        return {"name": name, "url": url, "error": str(e)}

    available = sorted(e["expert_id"] for e in hello.get("experts", []))
    region_tag = hello.get("region", "unknown")
    print(f"  region={region_tag} hosting {len(available)} experts "
          f"(min={available[0] if available else '-'} max={available[-1] if available else '-'})")
    target_expert = available[0] if available else 0

    result: dict = {
        "name": name,
        "url": url,
        "region_tag": region_tag,
        "n_experts_available": len(available),
        "target_expert_id": target_expert,
        "open_connection_ms": None,
    }

    # Warmup: do a few of each to amortize JIT / TCP window
    print("  warmup ...")
    await measure_json_ping(ws, args.warmup)
    await measure_forward(ws, target_expert, 1, args.warmup)

    print(f"  json-ping x{args.ping_iters} ...")
    ping_samples = await measure_json_ping(ws, args.ping_iters)
    result["json_ping"] = summarize("json_ping", ping_samples)
    print(f"    p50={result['json_ping']['p50_ms']:.1f}ms  "
          f"p95={result['json_ping']['p95_ms']:.1f}ms  "
          f"min={result['json_ping']['min_ms']:.1f}ms")

    forward_results = {}
    for n_tokens in args.token_batches:
        print(f"  forward [{n_tokens}x{HIDDEN}] x{args.forward_iters} ...")
        s = await measure_forward(ws, target_expert, n_tokens, args.forward_iters)
        forward_results[f"tokens_{n_tokens}"] = summarize(f"forward_{n_tokens}", s)
        print(f"    p50={forward_results[f'tokens_{n_tokens}']['p50_ms']:.1f}ms  "
              f"p95={forward_results[f'tokens_{n_tokens}']['p95_ms']:.1f}ms  "
              f"min={forward_results[f'tokens_{n_tokens}']['min_ms']:.1f}ms")
    result["forward"] = forward_results

    if args.sustained_s > 0:
        print(f"  sustained ping {args.sustained_s}s ...")
        s = await measure_sustained_ping(ws, args.sustained_s, 0.2)
        result["sustained_ping"] = summarize("sustained_ping", s)
        print(f"    p50={result['sustained_ping']['p50_ms']:.1f}ms  "
              f"p95={result['sustained_ping']['p95_ms']:.1f}ms  "
              f"max={result['sustained_ping']['max_ms']:.1f}ms")

    await ws.close()
    return result


async def main_async(args):
    regions = json.loads(args.regions_json)
    # Run regions serially to keep your home upload from contending with
    # multiple WS streams when the forward payload gets large; latency is
    # the metric, throughput is not.
    results = []
    for name, url in regions.items():
        results.append(await measure_region(name, url, args))

    out = {
        "measured_at": time.time(),
        "client_host": os.environ.get("COMPUTERNAME") or os.uname().nodename,
        "config": {
            "warmup": args.warmup,
            "ping_iters": args.ping_iters,
            "forward_iters": args.forward_iters,
            "token_batches": args.token_batches,
            "sustained_s": args.sustained_s,
        },
        "regions": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


def default_regions() -> str:
    return json.dumps({
        "us-west2":      "wss://expert-us-west-production.up.railway.app",
        "us-east4":      "wss://expert-us-east-production.up.railway.app",
        "europe-west4":  "wss://expert-eu-production.up.railway.app",
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", default=default_regions(),
                   help='JSON mapping of region label -> wss URL')
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--ping-iters", type=int, default=50)
    p.add_argument("--forward-iters", type=int, default=20)
    p.add_argument("--token-batches", type=int, nargs="+",
                   default=[1, 8, 32])
    p.add_argument("--sustained-s", type=float, default=30.0,
                   help="Sustained ping duration. 0 to skip.")
    p.add_argument("--output", default="output/region_latency.json")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
