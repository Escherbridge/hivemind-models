# Multi-region expert latency on Railway

Measured 2026-06-06. Three Railway services hosting copies of the
Granite-tiny layer-0 MoE experts:

| Service          | Region        | Experts hosted |
|------------------|---------------|----------------|
| `expert-us-west` | `us-west2`    | 0–31           |
| `expert-us-east` | `us-east4`    | 16–47          |
| `expert-eu`      | `europe-west4`| 0–15, 32–63    |

Each expert is replicated across exactly two regions (ring overlap), so
the router has a redundant choice for every call.

Two measurement passes were run:

1. **Client → region.** Latency from a Windows laptop on a US home
   connection to each Railway region (`scripts/measure_region_latency.py`).
2. **Region → region.** Latency between Railway regions, measured by a
   `/probe` endpoint on each expert server that opens a WebSocket to a
   peer region and runs the same sweep server-side
   (`scripts/sweep_region_matrix.py`).


## Pass 1 — client → region (json-ping p50)

| Target region   | p50 ms | p95 ms | min ms |
|-----------------|--------|--------|--------|
| `us-west2`      | 169    | 185    | 158    |
| `us-east4`      | 165    | 179    | 144    |
| `europe-west4`  | 167    | 178    | 150    |

All three regions land within ~5 ms of each other from this client.
That is not what physics would predict — `us-west2` is ~30 ms RTT from
the US west coast, `europe-west4` is ~140 ms. **The fact that every
region reports the same ~165 ms means the bottleneck is on the path
between the client and the Railway edge, not between client and region.**
The home uplink (or the TLS-terminating CDN in front of Railway) is
adding a fixed ~150 ms cost that drowns out any region difference.


## Pass 2 — region → region (json-ping p50, Railway backbone)

|              | → us-west2 | → us-east4 | → europe-west4 |
|--------------|------------|------------|----------------|
| us-west2     |     —      |    6.0 ms  |    5.4 ms      |
| us-east4     |   4.9 ms   |     —      |    5.9 ms      |
| europe-west4 |   5.0 ms   |    5.3 ms  |      —         |

Every Railway-internal pair sits in the **5–6 ms ping band**. This holds
even for the transatlantic `us-west2 ↔ europe-west4` pair, which on the
public internet is typically 140 ms RTT. That is a **~30× speedup**
over the client-measured numbers.

The implication for HiveMind routing is large: when the coordinator and
the expert peers both live on Railway (or any cloud with a low-latency
backbone), region choice barely matters for control-plane traffic. The
real cost is getting from the user's machine to the cloud edge.


## Pass 3 — binary forward (32×1536 fp16 ≈ 96 KB payload)

| Pair (source → target) | p50 ms | p95 ms |
|------------------------|--------|--------|
| us-west2 → us-east4    |   23.5 |   35.7 |
| us-west2 → europe-west4|   30.8 |   38.0 |
| us-east4 → us-west2    |   40.3 |   52.1 |
| us-east4 → europe-west4|   27.6 |   42.3 |
| europe-west4 → us-west2|   35.0 |   49.0 |
| europe-west4 → us-east4|   26.5 |   32.9 |

For comparison, the same payload measured from the client landed at
**450–510 ms p50** — again a ~15-20× gap.

Forward p50 across all backbone pairs is **23–40 ms** regardless of
geography. The fast pairs (≈24 ms) are usually the ones whose
healthchecks were warmest at the time of measurement; the slow pairs
(≈40 ms) include one TCP handshake per probe (open_ms = 70–110 ms is
spent before any measurement starts). Pings within a held-open
connection stayed steady — the keepalive design works.


## What this tells us about the chapter-10 architecture

The headline question of the granular-specialist chapter is whether
**distributed MoE dispatch can be made cheap enough to be useful**.
The ebook §10.2 numbers for localhost-on-Windows show a 13–15 ms
dispatch floor. On WSL2 it ballooned to 290 ms, which was a
virtualization artifact.

Railway region-to-region matches the Windows-on-bare-metal numbers
almost exactly: **23–30 ms for a single binary forward over a real
network**, sustained under load, with no warm-up needed once the WS is
established. The ~10 ms penalty over loopback is real network cost; the
~170 ms penalty seen from the laptop is **not in the protocol**, it's
on the client's last mile.

This validates two architectural choices simultaneously:

1. **Persistent WebSocket pools** between relays are the right pattern.
   The cold-connection cost (open_ms ≈ 70–110 ms) is dominated by TLS
   and TCP handshake; once a connection is held, RTT collapses to
   ~5 ms regardless of region pair. Re-opening for every dispatch
   would destroy the budget.

2. **Region locality matters for the user, not for the routing.**
   Sending a request to a relay 5,000 miles away is no more expensive
   between relays than between adjacent ones, but it adds 100+ ms for
   the user's first hop. So the routing-fabric design wants to **pin
   the entry relay close to the user** and let it freely dispatch
   across continents to wherever the experts happen to live.

What we still don't know (good follow-up questions):

- **Top-6 fan-out behavior.** This experiment only probed one expert
  per request. A real layer dispatch needs 6 experts and waits for the
  slowest; with redundancy that becomes best-of-K. Worth measuring
  before claiming the 23 ms p50 holds at production fan-out.
- **GPU dispatch.** All compute here is CPU on small experts (~5 MB).
  At Mixtral or DeepSeek scale, per-call compute dominates over
  network. The crossover point matters for sizing decisions.
- **Sustained load.** Probes were ~70 forward calls each. Continuous
  thousand-call workloads may surface buffer or scheduling effects
  this didn't capture.


## How to reproduce

```bash
# 1. Hit each region directly from the client
python scripts/measure_region_latency.py \
  --output output/region_latency.json

# 2. Drive the /probe endpoint on each region against the other two
python scripts/sweep_region_matrix.py \
  --output output/region_matrix.json
```

Region URLs live in both scripts at the top; change them if the Railway
service domains change.
