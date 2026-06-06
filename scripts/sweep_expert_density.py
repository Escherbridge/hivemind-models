"""
Latency sweep over expert-server density.

For each value of experts-per-server in {64, 32, 16, 8, 4, 2, 1},
  1) launch the swarm
  2) wait for readiness
  3) run moe_coordinator.py and capture telemetry
  4) tear down the swarm

Writes a JSON summary so we can drop the numbers straight into chapter 10.

Usage:
    python scripts/sweep_expert_density.py \
        --experts-dir output/granite-tiny-q4g64/experts_layer_0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PYTHON = sys.executable


def run_swarm(experts_dir: Path, eps: int, base_port: int) -> subprocess.Popen:
    # -u forces unbuffered stdout/stderr so our readline() loop sees lines.
    cmd = [
        PYTHON, "-u", "scripts/launch_expert_swarm.py",
        "--experts-dir", str(experts_dir),
        "--experts-per-server", str(eps),
        "--base-port", str(base_port),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    # Wait for "swarm ready"
    deadline = time.time() + 120
    saw_lines = []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"launcher exited prematurely (code {proc.returncode}). "
                    f"output so far:\n" + "".join(saw_lines)
                )
            time.sleep(0.05)
            continue
        saw_lines.append(line)
        if "swarm ready" in line:
            return proc
    raise TimeoutError(
        "swarm did not become ready in 120s. output so far:\n" + "".join(saw_lines)
    )


def stop_swarm(proc: subprocess.Popen, base_port: int, max_servers: int) -> None:
    """
    Terminate the launcher AND any expert servers it spawned. Windows signal
    propagation to child processes is unreliable, so we explicitly find any
    processes listening on the expected ports and kill them.
    """
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    # Belt and suspenders: kill anything still listening on our port range.
    end_port = base_port + max_servers + 5  # +5 slack
    kill_listeners_on_ports(base_port, end_port)


_NETSTAT_LISTEN_RE = re.compile(
    r"^\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$"
)


def _listening_pids_by_port() -> dict[int, int]:
    """Return {local_port: pid} for all current TCP listeners (Windows)."""
    if sys.platform != "win32":
        return {}
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return {}
    result: dict[int, int] = {}
    for line in out.stdout.splitlines():
        m = _NETSTAT_LISTEN_RE.match(line)
        if m:
            port, pid = int(m.group(1)), int(m.group(2))
            result[port] = pid
    return result


def kill_listeners_on_ports(start: int, end: int) -> None:
    """Find PIDs listening on any port in [start, end) and kill them."""
    if sys.platform != "win32":
        return
    listeners = _listening_pids_by_port()
    pids = {pid for port, pid in listeners.items() if start <= port < end}
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=5)
        except Exception:
            pass


def wait_ports_free(start: int, end: int, deadline_s: float = 15.0) -> bool:
    """Block until all ports in [start, end) are unbound, or timeout."""
    if sys.platform != "win32":
        time.sleep(2)
        return True
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        listeners = _listening_pids_by_port()
        held = [p for p in listeners if start <= p < end]
        if not held:
            return True
        time.sleep(0.5)
    return False


_ITER_TELEMETRY = re.compile(
    r"iter #(\d+):\s+router=([\d.]+)ms\s+dispatch=([\d.]+)ms\s+recombine=([\d.]+)ms\s+"
    r"experts_hit=(\d+)/(\d+)\s+calls=(\d+)\s+max_size=(\d+)"
)
_AGG_STATS = re.compile(
    r"(\w+)\s*:\s*mean=([\d.]+)ms\s+median=([\d.]+)ms\s+stdev=([\d.]+)ms\s+"
    r"min=([\d.]+)ms\s+max=([\d.]+)ms"
)


def run_coordinator(warmup: int, repeats: int) -> dict:
    proc = subprocess.run(
        [PYTHON, "-u", "scripts/moe_coordinator.py",
         "--warmup", str(warmup), "--repeats", str(repeats)],
        capture_output=True, text=True, timeout=900,
    )
    output = proc.stdout + proc.stderr
    pass_line = "PASS: top-1 matches reference" in output

    # Per-iter telemetry
    iters = []
    for m in _ITER_TELEMETRY.finditer(output):
        iters.append({
            "iter": int(m.group(1)),
            "router_ms": float(m.group(2)),
            "dispatch_ms": float(m.group(3)),
            "recombine_ms": float(m.group(4)),
            "experts_hit": int(m.group(5)),
            "n_experts": int(m.group(6)),
            "calls": int(m.group(7)),
            "max_size": int(m.group(8)),
        })

    # Aggregate stats (only printed when >=2 iters)
    agg: dict[str, dict] = {}
    for m in _AGG_STATS.finditer(output):
        agg[m.group(1)] = {
            "mean": float(m.group(2)),
            "median": float(m.group(3)),
            "stdev": float(m.group(4)),
            "min": float(m.group(5)),
            "max": float(m.group(6)),
        }

    return {
        "exit_code": proc.returncode,
        "pass": pass_line,
        "iters": iters,
        "aggregate": agg,
        "stdout_tail": output.splitlines()[-50:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts-dir", type=Path,
                        default=Path("output/granite-tiny-q4g64/experts_layer_0"))
    parser.add_argument("--densities", nargs="+", type=int,
                        default=[64, 32, 16, 8, 4, 2, 1])
    parser.add_argument("--base-port", type=int, default=9800)
    parser.add_argument("--repeats", type=int, default=1,
                        help="How many coordinator runs per density (each run does its own warmup+iters)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="MoE warmup dispatches per coordinator run")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed MoE dispatches per coordinator run")
    parser.add_argument("--output", type=Path,
                        default=Path("output/expert_density_sweep.json"))
    args = parser.parse_args()

    # Best-effort pre-cleanup: in case a prior sweep left zombies behind,
    # nuke everything in our port range before starting density #1.
    max_n_servers = max(64 // eps if eps else 1 for eps in args.densities)
    print(f"pre-sweep cleanup: clearing ports {args.base_port}..{args.base_port + max_n_servers - 1}")
    kill_listeners_on_ports(args.base_port, args.base_port + max_n_servers + 5)
    if not wait_ports_free(args.base_port, args.base_port + max_n_servers + 5):
        print("WARNING: not all ports cleared after pre-sweep cleanup")

    results: list[dict] = []
    for eps in args.densities:
        n_srv = (64 + eps - 1) // eps
        print(f"\n{'='*72}\n=== experts-per-server = {eps} (=> {n_srv} servers) ===\n{'='*72}")
        swarm = run_swarm(args.experts_dir, eps, args.base_port)
        time.sleep(2)  # let things settle
        runs: list[dict] = []
        try:
            for r in range(args.repeats):
                print(f"  -- run {r+1}/{args.repeats} (warmup={args.warmup}, iters={args.iters}) --")
                res = run_coordinator(args.warmup, args.iters)
                if res["aggregate"].get("dispatch"):
                    a = res["aggregate"]["dispatch"]
                    print(f"    dispatch: mean={a['mean']:.1f}  median={a['median']:.1f}  "
                          f"stdev={a['stdev']:.1f}  min={a['min']:.1f}  max={a['max']:.1f}  "
                          f"(n={len(res['iters'])})  pass={res['pass']}")
                else:
                    print(f"    NO TELEMETRY; exit={res['exit_code']}")
                    tail = res.get("stdout_tail", [])
                    for line in tail[-20:]:
                        print(f"      | {line.rstrip()}")
                runs.append(res)
        finally:
            stop_swarm(swarm, args.base_port, n_srv)
            if not wait_ports_free(args.base_port, args.base_port + n_srv + 2):
                print(f"WARNING: ports {args.base_port}..{args.base_port + n_srv - 1} still held after kill")

        results.append({
            "experts_per_server": eps,
            "n_servers": n_srv,
            "runs": runs,
        })

    # Compact summary
    summary = {
        "config": {
            "warmup": args.warmup,
            "iters_per_run": args.iters,
            "runs_per_density": args.repeats,
        },
        "results": [
            {
                "experts_per_server": r["experts_per_server"],
                "n_servers": r["n_servers"],
                "pass_count": sum(1 for run in r["runs"] if run["pass"]),
                "total_runs": len(r["runs"]),
                "runs": [
                    {
                        "pass": run["pass"],
                        "iters": run["iters"],
                        "aggregate": run["aggregate"],
                    }
                    for run in r["runs"]
                ],
            }
            for r in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary to {args.output}")

    print("\n=== SUMMARY (median over all timed iters per density) ===")
    print(f"{'experts/srv':>12s} {'servers':>8s} "
          f"{'median':>9s} {'mean':>9s} {'stdev':>9s} {'min':>9s} {'max':>9s} "
          f"{'iters':>6s} {'PASS':>6s}")
    import statistics as st
    for r in summary["results"]:
        # Flatten all dispatch_ms across all runs of this density
        all_dispatch: list[float] = []
        for run in r["runs"]:
            for it in run["iters"]:
                all_dispatch.append(it["dispatch_ms"])
        if not all_dispatch:
            print(f"{r['experts_per_server']:>12d} {r['n_servers']:>8d}  (no data)")
            continue
        med = st.median(all_dispatch)
        mn = st.mean(all_dispatch)
        sd = st.stdev(all_dispatch) if len(all_dispatch) > 1 else 0.0
        print(f"{r['experts_per_server']:>12d} {r['n_servers']:>8d} "
              f"{med:>9.1f} {mn:>9.1f} {sd:>9.1f} {min(all_dispatch):>9.1f} {max(all_dispatch):>9.1f} "
              f"{len(all_dispatch):>6d} "
              f"{r['pass_count']:>2d}/{r['total_runs']:<2d} ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
