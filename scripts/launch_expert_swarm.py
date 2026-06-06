"""
Launch a swarm of expert WebSocket servers with configurable density.

`--experts-per-server N` groups consecutive experts together: N=1 spawns
one process per expert (64 processes for layer 0), N=8 spawns 8 processes
each holding 8 experts. The wire protocol carries an `expert_id` byte, so
the coordinator's view is unchanged regardless of grouping.

Usage:
    python scripts/launch_expert_swarm.py \
        --experts-dir output/granite-tiny-q4g64/experts_layer_0 \
        --experts-per-server 8 \
        --base-port 9800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


PYTHON = sys.executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts-dir", type=Path, required=True)
    parser.add_argument("--experts-per-server", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=9800)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    manifest = json.loads((args.experts_dir / "manifest.json").read_text())
    layer = int(manifest["layer"])
    n_experts = int(manifest["num_experts"])
    eps = max(1, args.experts_per_server)
    if n_experts % eps != 0:
        print(f"WARNING: {n_experts} experts not divisible by {eps}; last server will hold fewer")

    n_servers = (n_experts + eps - 1) // eps
    print(f"layer {layer}: {n_experts} experts, {eps} per server, {n_servers} server processes")

    server_script = str(Path(__file__).resolve().parent / "expert_ws_server.py")
    procs: list[tuple[int, list[int], subprocess.Popen]] = []

    for s_idx in range(n_servers):
        port = args.base_port + s_idx
        start = s_idx * eps
        end = min(start + eps, n_experts)
        weight_args = []
        for e in range(start, end):
            weight_args.append(str(args.experts_dir / manifest["expert_files"][e]))
        cmd = [
            PYTHON, server_script,
            "--layer", str(layer),
            "--host", args.host,
            "--port", str(port),
            "--weights", *weight_args,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        procs.append((port, list(range(start, end)), proc))

    print(f"spawned {n_servers} processes; waiting for readiness ...")

    import websockets

    async def check(port: int) -> bool:
        try:
            async with websockets.connect(
                f"ws://{args.host}:{port}", open_timeout=1,
            ) as ws:
                hello = await asyncio.wait_for(ws.recv(), timeout=2)
                obj = json.loads(hello)
                return obj.get("type") == "ready"
        except Exception:
            return False

    async def wait_all() -> int:
        ready: set[int] = set()
        deadline = time.time() + 60
        while time.time() < deadline and len(ready) < n_servers:
            to_check = [p for (p, _, _) in procs if p not in ready]
            results = await asyncio.gather(*(check(p) for p in to_check),
                                           return_exceptions=True)
            for p, ok in zip(to_check, results):
                if ok is True:
                    ready.add(p)
            if len(ready) < n_servers:
                await asyncio.sleep(0.5)
        return len(ready)

    ready = asyncio.run(wait_all())
    print(f"  {ready}/{n_servers} servers healthy")

    if ready < n_servers:
        print("WARNING: not all servers came up; continuing anyway.")

    # Build a port -> expert_ids map so the coordinator can discover topology.
    topo = {p: ids for (p, ids, _) in procs}
    topo_file = args.experts_dir / "topology.json"
    topo_file.write_text(json.dumps({
        "host": args.host,
        "layer": layer,
        "n_experts": n_experts,
        "experts_per_server": eps,
        "servers": [{"port": p, "expert_ids": ids} for (p, ids, _) in procs],
    }, indent=2))
    print(f"wrote topology to {topo_file}")

    def shutdown(*_):
        print("\nshutting down ...")
        # Kill children FIRST so they release their ports before we exit.
        # On Windows, SIGTERM doesn't propagate to children automatically.
        for _, _, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for _, _, p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                    p.wait(timeout=2)
                except Exception:
                    pass
        print("all children stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    # Windows specific: catch CTRL_BREAK_EVENT too (sent by Popen.terminate
    # on Windows when no handle is shared)
    try:
        signal.signal(signal.SIGBREAK, shutdown)
    except (AttributeError, ValueError):
        pass

    print(f"\nexpert swarm ready. ports {args.base_port}..{args.base_port + n_servers - 1}")
    print("press Ctrl+C to stop")

    while True:
        time.sleep(2)
        dead = [(port, ids, proc) for (port, ids, proc) in procs if proc.poll() is not None]
        if dead:
            for port, ids, proc in dead:
                print(f"WARNING: server port={port} ids={ids} exited code {proc.returncode}")
            procs[:] = [(p, ids, pr) for (p, ids, pr) in procs if pr.poll() is None]
            if not procs:
                return 1


if __name__ == "__main__":
    sys.exit(main())
