"""
One-shot verification that the refactored shard_server (with architecture
handlers) still produces "Paris" through the live swarm for the canonical
"The capital of France is" prompt. No generation loop, just one forward pass.
"""

from __future__ import annotations

import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


PORTS = [9000, 9001, 9002, 9003, 9004, 9005]
NAMES = ["embed", "layers_0_3", "layers_4_10", "layers_11_17", "layers_18_21", "head"]


def pack(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    shape = list(arr.shape)
    dtype = b"float32"
    buf = struct.pack("<I", len(shape))
    for d in shape:
        buf += struct.pack("<I", d)
    buf += struct.pack("<I", len(dtype)) + dtype + arr.tobytes()
    return buf


def unpack(raw: bytes) -> np.ndarray:
    off = 0
    (n,) = struct.unpack_from("<I", raw, off); off += 4
    shape = []
    for _ in range(n):
        (d,) = struct.unpack_from("<I", raw, off); off += 4
        shape.append(d)
    (dl,) = struct.unpack_from("<I", raw, off); off += 4
    dt = raw[off:off + dl].decode(); off += dl
    return np.frombuffer(raw[off:], dtype=np.float32 if dt == "float32" else np.float16).reshape(shape).copy()


def main() -> int:
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompt = "The capital of France is"
    ids = tok.encode(prompt, return_tensors="np").astype(np.float32)
    print(f"Prompt: {prompt!r}")
    print(f"Tokens ({ids.shape[1]}): {ids.tolist()}")

    cur = ids
    for port, name in zip(PORTS, NAMES):
        url = f"http://localhost:{port}/forward"
        t0 = time.perf_counter()
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    data=pack(cur),
                    headers={"Content-Type": "application/octet-stream"},
                ),
                timeout=60,
            )
            raw = resp.read()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            return 1
        elapsed = (time.perf_counter() - t0) * 1000
        cur = unpack(raw)
        print(f"  [OK] {name:14s} -> shape={list(cur.shape)}  ({elapsed:.0f}ms)")

    logits = cur[0, -1]
    top5_idx = np.argsort(logits)[-5:][::-1]
    print("\nTop-5 next-token predictions:")
    for idx in top5_idx:
        s = tok.decode([int(idx)]).encode("ascii", errors="replace").decode("ascii")
        print(f"  {int(idx):6d}  {float(logits[idx]):8.3f}  {s!r}")

    top1 = tok.decode([int(top5_idx[0])]).strip()
    if top1.lower() == "paris":
        print(f"\nPASS: top-1 prediction is {top1!r}")
        return 0
    print(f"\nFAIL: top-1 prediction is {top1!r}, expected 'Paris'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
