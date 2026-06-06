"""
Drive the /probe endpoint on each Railway region against the other two,
giving us the full 3x2 region-to-region latency matrix measured inside
Railway's network (independent of the home-uplink path to the client).

Output: output/region_matrix.json with all 6 directed pairs.
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


def probe(source_http: str, target_ws: str, ping_iters: int,
          forward_iters: int, tokens: str, warmup: int,
          timeout_s: int = 240) -> dict:
    qs = urllib.parse.urlencode({
        "target": target_ws,
        "ping_iters": ping_iters,
        "forward_iters": forward_iters,
        "tokens": tokens,
        "warmup": warmup,
    })
    url = f"{source_http}/probe?{qs}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "hivemind-region-matrix/1.0",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        body = r.read()
        elapsed = time.time() - t0
        return {"http_status": r.status, "client_elapsed_s": elapsed,
                **json.loads(body)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping-iters", type=int, default=50)
    ap.add_argument("--forward-iters", type=int, default=20)
    ap.add_argument("--tokens", default="1,8,32")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout-s", type=int, default=240)
    ap.add_argument("--output", default="output/region_matrix.json")
    args = ap.parse_args()

    results = []
    for src, (src_http, _) in REGIONS.items():
        for dst, (_, dst_ws) in REGIONS.items():
            if src == dst:
                continue
            print(f"\n>>> {src} -> {dst}")
            try:
                r = probe(src_http, dst_ws, args.ping_iters,
                          args.forward_iters, args.tokens, args.warmup,
                          args.timeout_s)
                jp = (r.get("result") or {}).get("json_ping") or {}
                fw = (r.get("result") or {}).get("forward") or {}
                print(f"  open={(r.get('result') or {}).get('open_ms', 0):.1f}ms "
                      f"ping p50={jp.get('p50_ms', 0):.1f}ms  "
                      f"p95={jp.get('p95_ms', 0):.1f}ms")
                for tk, summ in fw.items():
                    if "p50_ms" in summ:
                        print(f"  fwd {tk}: p50={summ['p50_ms']:.1f}ms  "
                              f"p95={summ['p95_ms']:.1f}ms")
                    else:
                        print(f"  fwd {tk}: {summ}")
            except Exception as e:
                r = {"error": str(e)}
                print(f"  FAIL: {e}")
            results.append({"source": src, "target": dst, "data": r})

    out = {
        "measured_at": time.time(),
        "config": vars(args),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
