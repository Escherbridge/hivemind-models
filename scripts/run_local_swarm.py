"""
Launch a local swarm of shard servers for a sharded model.

Usage:
    python scripts/run_local_swarm.py --shard-dir ./output/tinyllama-1b-q4

Starts one shard_server.py per shard file (embed, layer groups, head),
each on a different port. Then runs a quick end-to-end inference test
through the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

PORT_BASE = 9000


def main():
    parser = argparse.ArgumentParser(description="Launch local shard swarm")
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--relay", type=str, default="http://localhost:8787")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test", action="store_true", help="Run end-to-end test after launch")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No manifest.json in {shard_dir}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    model_id = manifest["model_id"]
    layer_groups = manifest.get("layer_groups", [])

    # Plan which servers to launch
    servers = []
    port = PORT_BASE

    # Embed server
    if (shard_dir / "shard_embed.safetensors").exists():
        servers.append({
            "name": "embed",
            "port": port,
            "args": ["--shard-type", "embed"],
        })
        port += 1

    # Layer group servers
    for start, end in layer_groups:
        shard_file = shard_dir / f"shard_layers_{start}_{end}.safetensors"
        if shard_file.exists():
            servers.append({
                "name": f"layers_{start}_{end}",
                "port": port,
                "args": ["--layers", str(start), str(end)],
            })
            port += 1

    # Head server
    if (shard_dir / "shard_head.safetensors").exists():
        servers.append({
            "name": "head",
            "port": port,
            "args": ["--shard-type", "head"],
        })
        port += 1

    print(f"Launching {len(servers)} shard servers for {model_id}")
    print(f"  Shard dir: {shard_dir}")
    print(f"  Device: {args.device}")
    print()

    processes = []
    for srv in servers:
        cmd = [
            sys.executable, "scripts/shard_server.py",
            "--shard-dir", str(shard_dir),
            "--port", str(srv["port"]),
            "--relay", args.relay,
            "--device", args.device,
            "--model-id", model_id,
            *srv["args"],
        ]
        print(f"  [{srv['name']}] port {srv['port']} -> {' '.join(cmd[-4:])}")
        proc = subprocess.Popen(cmd, cwd=str(shard_dir.parent.parent))
        processes.append((srv, proc))

    print()
    print("All servers launching. Waiting for startup...")

    # Wait for health checks
    import urllib.request
    import urllib.error

    healthy = set()
    for attempt in range(30):
        time.sleep(1)
        for srv, proc in processes:
            if srv["name"] in healthy:
                continue
            try:
                url = f"http://localhost:{srv['port']}/health"
                resp = urllib.request.urlopen(url, timeout=2)
                if resp.status == 200:
                    healthy.add(srv["name"])
                    print(f"  [OK] {srv['name']} ready on port {srv['port']}")
            except (urllib.error.URLError, ConnectionError):
                pass
        if len(healthy) == len(servers):
            break

    if len(healthy) < len(servers):
        missing = [s["name"] for s in servers if s["name"] not in healthy]
        print(f"\nWARNING: {len(missing)} servers failed to start: {missing}")
    else:
        print(f"\nAll {len(servers)} servers healthy!")

    # Print server map
    print("\n--- Server Map ---")
    for srv, _ in processes:
        print(f"  {srv['name']:20s}  http://localhost:{srv['port']}")

    if args.test:
        print("\n--- Running end-to-end test ---")
        run_e2e_test(servers, model_id, args.device)

    # Wait for Ctrl+C
    print("\nPress Ctrl+C to stop all servers...")

    def shutdown(sig, frame):
        print("\nShutting down...")
        for srv, proc in processes:
            proc.terminate()
        for srv, proc in processes:
            proc.wait(timeout=5)
        print("All servers stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep main process alive
    while True:
        time.sleep(1)
        for srv, proc in processes:
            if proc.poll() is not None:
                print(f"WARNING: {srv['name']} exited with code {proc.returncode}")


def run_e2e_test(servers: list[dict], model_id: str, device: str):
    """Send a prompt through the full shard pipeline."""
    import urllib.request
    import struct
    import numpy as np

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  transformers not installed, skipping test")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    print(f"  Prompt: '{prompt}'")
    print(f"  Tokens: {input_ids.shape} = {input_ids.tolist()}")

    # Pack token IDs as float32 tensor (embed server will cast to long)
    current = input_ids.astype(np.float32)

    for srv in servers:
        url = f"http://localhost:{srv['port']}/forward"

        # Pack current tensor
        shape = list(current.shape)
        dtype_str = "float32"
        dtype_bytes = dtype_str.encode("utf-8")
        buf = struct.pack("<I", len(shape))
        for d in shape:
            buf += struct.pack("<I", d)
        buf += struct.pack("<I", len(dtype_bytes))
        buf += dtype_bytes
        buf += current.tobytes()

        req = urllib.request.Request(
            url, data=buf,
            headers={"Content-Type": "application/octet-stream"},
        )

        t0 = time.perf_counter()
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            raw = resp.read()
        except Exception as e:
            print(f"  [FAIL] {srv['name']}: {e}")
            return
        elapsed = (time.perf_counter() - t0) * 1000

        # Unpack response tensor
        off = 0
        (ndim,) = struct.unpack_from("<I", raw, off); off += 4
        out_shape = []
        for _ in range(ndim):
            (d,) = struct.unpack_from("<I", raw, off); off += 4
            out_shape.append(d)
        (dl,) = struct.unpack_from("<I", raw, off); off += 4
        out_dtype = raw[off:off+dl].decode(); off += dl
        np_dtype = {"float32": np.float32, "float16": np.float16}[out_dtype]
        current = np.frombuffer(raw[off:], dtype=np_dtype).reshape(out_shape)

        print(f"  [OK] {srv['name']:20s} -> shape {out_shape}  ({elapsed:.0f}ms)")

    # Final output: logits -> sample token
    logits = current[0, -1, :]  # last position
    top5 = np.argsort(logits)[-5:][::-1]
    print("\n  Top-5 predictions:")
    for idx in top5:
        token_str = tokenizer.decode([int(idx)])
        # Sanitize for Windows console encoding
        safe_str = token_str.encode("ascii", errors="replace").decode("ascii")
        print(f"    {idx:6d}  {logits[idx]:8.3f}  '{safe_str}'")

    next_token = int(top5[0])
    next_str = tokenizer.decode([next_token]).encode("ascii", errors="replace").decode("ascii")
    print(f"\n  Generated: '{prompt}{next_str}'")


if __name__ == "__main__":
    main()
