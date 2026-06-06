"""
WebSocket client smoke test for the Granite shard server.

Connects, exchanges a ping, requests a completion, and prints the result.
Compares the top-1 prediction against the previously-saved reference from
gguf_reference.py.

Usage:
    python scripts/granite_ws_client.py --uri ws://127.0.0.1:9700
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets


async def run(uri: str, prompt: str, max_tokens: int) -> int:
    print(f"connecting to {uri} ...")
    try:
        ws = await websockets.connect(uri, max_size=2**24, open_timeout=5)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 2

    async with ws:
        # 1) read the server's hello
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if hello.get("type") != "ready":
            print(f"  FAIL: first message wasn't 'ready', got {hello!r}")
            return 2
        print(f"  ready: arch={hello['model']['arch']}")
        for k, v in (hello.get("metadata") or {}).items():
            print(f"    {k}: {v}")

        # 2) ping
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await ws.recv())
        ping_ms = (time.perf_counter() - t0) * 1000
        if pong.get("type") != "pong":
            print(f"  FAIL: ping response wasn't 'pong', got {pong!r}")
            return 2
        print(f"  ping: {ping_ms:.1f}ms round-trip")

        # 3) completion request
        cid = "verify-1"
        req = {
            "type": "complete",
            "id": cid,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_k_logprobs": 5,
        }
        print(f"  -> complete: {prompt!r}")
        await ws.send(json.dumps(req))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))

        if resp.get("type") == "error":
            print(f"  SERVER ERROR: {resp.get('message')}")
            return 2
        if resp.get("type") != "complete_result":
            print(f"  FAIL: unexpected response type {resp.get('type')!r}")
            return 2

        completion = resp["completion"]
        elapsed = resp["elapsed_ms"]
        safe = completion.encode("ascii", errors="replace").decode("ascii")
        print(f"  <- '{prompt}{safe}'")
        print(f"     tokens={resp['tokens_generated']}  elapsed={elapsed:.0f}ms")

        top_k = resp.get("first_token_top_k")
        if top_k:
            print("  first-token top-5:")
            for entry in top_k:
                safe = entry["token"].encode("ascii", errors="replace").decode("ascii")
                print(f"    {entry['logprob']:8.4f}  {safe!r}")
            top1 = top_k[0]["token"]
        else:
            top1 = None

        # 4) compare against reference
        ref_path = Path("output/granite_reference.json")
        if ref_path.exists() and top1 is not None:
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
            if ref["prompt"] == prompt:
                ref_top1 = ref["top_k_tokens"][0]["token"]
                if ref_top1 == top1:
                    print(f"\n  PARITY: top-1 matches reference ({top1!r})")
                else:
                    print(f"\n  DIVERGENCE: ws_top1={top1!r}  ref_top1={ref_top1!r}")
                    return 1
            else:
                print(f"\n  (reference prompt was {ref['prompt']!r}, not comparing top-1)")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="ws://127.0.0.1:9700")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    code = asyncio.run(run(args.uri, args.prompt, args.max_tokens))
    sys.exit(code)


if __name__ == "__main__":
    main()
