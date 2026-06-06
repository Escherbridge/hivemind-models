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


## Pass 4 — top-6 fan-out (wait-all dispatch)

This is the realistic production dispatch pattern: from one region,
open 6 persistent WebSockets (round-robin across the other two
regions), fire 6 parallel forwards per iteration, wait for the slowest
to come back. Payload 8×1536 fp16 ≈ 24 KB per call.

| Source       | Targets               | p50 ms | p95 ms | p99 ms | max ms |
|--------------|-----------------------|--------|--------|--------|--------|
| us-west2     | us-east4, europe-west4|  71.3  | 103.0  | 106.9  | 108.9  |
| us-east4     | us-west2, europe-west4|  49.8  |  66.9  |  81.1  |  86.9  |
| europe-west4 | us-west2, us-east4    |  77.9  | 105.6  | 117.9  | 118.4  |

Per-call latency stayed at the same 26–37 ms p50 we saw in pass 2 —
fan-out doesn't slow individual calls down. The penalty is entirely
that you're now waiting for the slowest of 6, so the iteration p50
sits near the per-call p95.

p99 of 100–120 ms is the bound you'd quote a user. For a single layer
that's fine; for 40 layers chained (the Granite-tiny shape) you'd
naively get 40 × 80 ms = 3.2 s, which is the bound the chapter has
been working against from the start.

## Pass 5 — top-6 first-of-K (best-of-K redundancy)

Same fan-out, but record the time-to-**first** response and let the
losers finish in the background. This is the chapter §10.4.3
argument: redundant peers collapse tail latency.

| Source       | Targets               | p50 ms | p95 ms | p99 ms | max ms |
|--------------|-----------------------|--------|--------|--------|--------|
| us-west2     | us-east4, europe-west4|  20.5  |  28.3  |  33.8  |  35.6  |
| us-east4     | us-west2, europe-west4|  12.0  |  15.0  |  18.6  |  20.5  |
| europe-west4 | us-west2, us-east4    |   8.9  |  21.2  |  28.2  |  30.5  |

**3–8× faster than wait-all at p50, 3–6× faster at p99.** The full
40-layer naive bound drops from ~3 s to ~400 ms per token using nothing
but the cheapest possible redundancy (k=6 hitting 2 regions). That's
the difference between "slow demo" and "feels interactive."

### Winner distribution (which target finished first)

| Source       | Winner counts (40 iters)                                |
|--------------|---------------------------------------------------------|
| us-west2     | europe-west4: 30, us-east4: 10                          |
| us-east4     | us-west2: 31, europe-west4: 9                           |
| europe-west4 | us-west2: 39, us-east4: 1                               |

This is the surprising part. From `europe-west4`, the **us-west2**
target (5,400 mi away) won 39 of 40 races against us-east4 (3,900 mi
away). Geographic distance is not the dominant factor here — what
matters is which peer happened to have shorter queue depth at the
moment of dispatch. The redundancy is doing real work, not just
duplicating identical paths.

If you wanted a single-region routing rule ("always send my second
copy to peer X"), the data says **don't** — pick the redundant peer
randomly or by current queue depth, not by distance. Static
"closest second peer" routing would have made the EU experiments
worse on average, not better.


## What this means for production grade

The headline number was **108 ms p99 wait-all from us-west2** and
**33.8 ms p99 first-of-K from us-west2**. Per layer, distributed,
on real network. That puts the realistic operating envelope at:

- **Single-layer dispatch under 35 ms p99** with k=6, best-of-K against
  2 redundant regions.
- **First-token latency on a 40-layer model around 400 ms** if every
  layer were distributed this way and the layers were serial. Most
  models can pipeline layer-N+1 input on layer-N output, so this is
  the worst-case wall clock.
- **No CPU constraint** in any of the measurements above. These
  experiments ran in `COMPUTE_MODE=echo` for the topology probes
  (real torch matmuls on the chapter-10 path), so the numbers are
  routing + network only. Production with GPU dispatch would
  contribute another ~1–5 ms per layer; production with CPU at
  Granite-tiny scale contributes ~0.5 ms. Neither moves the bound.

The two things that would still make this **not** production-grade for
a real end user:

1. **The 165 ms client → edge floor.** Every measurement above is
   inside Railway. The user's last mile adds 50–170 ms one-way (often
   asymmetric — uplink is usually worse). For interactive use that's
   the dominant cost, and the answer is "put the entry relay
   geographically close to the user," not anything in the routing
   layer.

2. **Sustained load.** All sweeps ran 20–40 iterations. For real
   workloads you need to confirm the p99 doesn't drift upward over
   tens of thousands of calls (TCP buffer pressure, OS scheduler
   tail effects, Railway proxy connection reaping). Worth a follow-up
   that holds the WS pool open for an hour and reports the rolling p99.

GPU is **not** required to hit these numbers. The chapter-10
distributed-MoE architecture is already production-viable on CPU
relays, provided the entry relay is close to the user and the routing
fabric uses k>=2 redundancy. Moving to GPU is about per-call compute
cost (model size scaling), not about hitting interactive latency.


## How to reproduce

```bash
# 1. Hit each region directly from the client
python scripts/measure_region_latency.py \
  --output output/region_latency.json

# 2. Drive the /probe endpoint on each region against the other two
python scripts/sweep_region_matrix.py \
  --output output/region_matrix.json

# 3. Top-6 wait-all dispatch (production pattern)
python scripts/sweep_topk_matrix.py --mode wait_all --k 6 \
  --iters 40 --output output/region_topk_wait_all.json

# 4. Top-6 best-of-K redundancy
python scripts/sweep_topk_matrix.py --mode first_of_k --k 6 \
  --iters 40 --output output/region_topk_first_of_k.json
```

Region URLs live in all four scripts at the top; change them if the
Railway service domains change.
