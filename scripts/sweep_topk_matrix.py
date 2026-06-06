"""
Drive /probe_topk on each Railway region, fanning out to all three
regions (round-robin). Two modes:

  --mode wait_all     time-to-Kth-response   (production dispatch)
  --mode first_of_k   time-to-first-response (best-of-K redundancy)

Output: output/region_topk_<mode>.json
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


REGIONS = {
    "us-west2":     ("https://expert-us-west-production.up.railway.app",
                     "wss://expert-us-west-production.up.railway.app"),
    "us-east4":     ("https://expert-us-east-production.up.railway.app",
                     "wss://expert-us-east-production.up.railway.app"),
    "europe-west4": ("https://expert-eu-production.up.railway.app",
                     "wss://expert-eu-production.up.railway.app"),
}


def call_probe_topk(source_http: str, targets_csv: str, iters: int,
                    n_tokens: int, k: int, mode: str, warmup: int,
                    timeout_s: int = 300) -> dict:
    qs = urllib.parse.urlencode({
        "targets": targets_csv,
        "iters": iters,
        "n_tokens": n_tokens,
        "k": k,
        "mode": mode,
        "warmup": warmup,
    })
    url = f"{source_http}/probe_topk?{qs}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "hivemind-topk-matrix/1.0",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = r.read()
        elapsed = time.time() - t0
        return {"http_status": r.status, "client_elapsed_s": elapsed,
                **json.loads(body)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("wait_all", "first_of_k"),
                    default="wait_all")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--targets-from-source",
                    choices=("all_others", "all_three"),
                    default="all_others",
                    help="'all_others' = fan out to the other 2 regions; "
                         "'all_three' = include the source region itself")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = Path(args.output or f"output/region_topk_{args.mode}.json")

    results = []
    for src, (src_http, _) in REGIONS.items():
        if args.targets_from_source == "all_others":
            tgts_ws = [ws for name, (_, ws) in REGIONS.items() if name != src]
            tgts_labels = [name for name in REGIONS if name != src]
        else:
            tgts_ws = [ws for _, (_, ws) in REGIONS.items()]
            tgts_labels = list(REGIONS.keys())
        targets_csv = ",".join(tgts_ws)
        print(f"\n>>> {src}  fan-out k={args.k} mode={args.mode} -> {tgts_labels}")
        try:
            r = call_probe_topk(src_http, targets_csv,
                                args.iters, args.n_tokens, args.k,
                                args.mode, args.warmup, args.timeout_s)
            res = r.get("result") or {}
            il = res.get("iter_latency", {})
            pcl = res.get("per_call_latency", {})
            print(f"  iter (time-to-{('Kth' if args.mode == 'wait_all' else '1st')} resp): "
                  f"p50={il.get('p50_ms', 0):.1f}ms  "
                  f"p95={il.get('p95_ms', 0):.1f}ms  "
                  f"p99={il.get('p99_ms', 0):.1f}ms  "
                  f"max={il.get('max_ms', 0):.1f}ms")
            print(f"  per-call: p50={pcl.get('p50_ms', 0):.1f}ms  "
                  f"p95={pcl.get('p95_ms', 0):.1f}ms")
            if args.mode == "first_of_k":
                wins = res.get("winning_target_counts") or {}
                print(f"  winner distribution: {wins}")
        except Exception as e:
            r = {"error": str(e)}
            print(f"  FAIL: {e}")
        results.append({
            "source": src,
            "targets": tgts_labels,
            "mode": args.mode,
            "k": args.k,
            "n_tokens": args.n_tokens,
            "data": r,
        })

    out = {
        "measured_at": time.time(),
        "config": vars(args),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
