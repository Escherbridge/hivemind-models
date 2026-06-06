"""Quick verifier: hit /health and /info on each region and report shape."""
import json
import sys
import urllib.request
import urllib.error

REGIONS = {
    "us-west2":     "https://expert-us-west-production.up.railway.app",
    "us-east4":     "https://expert-us-east-production.up.railway.app",
    "europe-west4": "https://expert-eu-production.up.railway.app",
}


def get(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "hivemind-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:200]}
    except Exception as e:
        return None, {"error": str(e)}


all_ok = True
for label, base in REGIONS.items():
    print(f"\n== {label} ({base}) ==")
    s, h = get(f"{base}/health")
    print(f"  /health -> {s} {json.dumps(h)[:200]}")
    s, info = get(f"{base}/info")
    print(f"  /info   -> {s}", end=" ")
    if isinstance(info, dict) and "experts" in info:
        print(f"n_experts={info.get('n_experts', len(info.get('experts', [])))} "
              f"region_tag={info.get('region')} layer={info.get('layer')}")
        ids = info.get("expert_ids", [])
        print(f"  expert_ids={ids[:6]}{'...' if len(ids) > 6 else ''} (total {len(ids)})")
    else:
        print(json.dumps(info)[:200])
        all_ok = False

sys.exit(0 if all_ok else 1)
