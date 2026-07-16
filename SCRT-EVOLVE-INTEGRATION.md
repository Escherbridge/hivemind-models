# scrt-evolve ↔ hivemind — Branch-Train-Merge Integration Brief

> **Cross-repo coordination artifact.** Authored 2026-06-25 from the
> scrt-evolve side, for the hivemind coding agent. Companion to `HANDOFF.md` (which
> covers the distributed-dispatch *experiments*; this covers the *product topology*
> and the contract between the two repos). Lived as git-ignored scratch until
> 2026-07-16, when it was committed: the release docs cite it as the authoritative
> integration contract, so it belongs in history. Co-owned with the scrt-evolve
> side — coordinate edits.

> **Product framing note (2026-07-16, hivemind side).** The shipping product is
> `hivemind-desktop` — a desktop app wrapping the evolve engine, with opt-in P2P
> export of inference capacity and monetization on the mesh (attribution-only
> logging v1 per `inference-trading-demo_20260627`, DAO settlement v2 per ADR D2).
> The "evolve start" UX design (captured in
> `conductor/tracks/_shared/evolve-start-design.md`) is the complete product
> target. scrt-evolve-side gaps — the `start` verb, the `[corpus]` config block,
> live-serve — are in flight in the scrt-evolve repo.

---

## 0. The one thing to internalize: the topology pivoted

`HANDOFF.md`'s design — distribute **one** Granite model's MoE experts across peers
(`expert_id`-routed frames) — is being **superseded for the product path**, because
it does not survive how MoE routing actually works:

- Mixtral's own paper: experts route on **token-surface / syntax, NOT semantic
  domain** — *"we do not observe obvious patterns ... based on the topic."* [arXiv 2401.04088]
- Upcycled identical experts **collapse onto ~2 experts** (load imbalance). [arXiv 2502.19261]
- ⇒ "one expert = one domain = one peer" yields **all-to-all PER-TOKEN traffic** and
  hot/idle peer imbalance. It breaks.

**New topology = Branch-Train-Merge** [Li & Gururangan, arXiv 2208.03306; c-BTM 2303.14177]:
each peer hosts a **standalone domain-specialized small model (a "branch" / ELM)**, and
requests route **per-request** (by an explicit domain classifier), **not per-token**.
The distributed-dispatch transport you built is still the substrate — but the **unit
hosted per peer changes** from "an FFN expert sub-block" to "a whole small branch model."

---

## 0.5 RESOLVED ARCHITECTURE — fully decentralized (decided 2026-06-25)

The architecture is **fully peer-to-peer, no central component**:
- **No central routing / no central "core" model.** Central routing was considered and **rejected**. Orchestration is an **app-level** concern: apps orchestrate + route through multiple models using the branch fleet + `BranchRouter` + linker as PRIMITIVES (via the SDK). A "frontier core" is one kind of app, NOT infrastructure.
- **Desktop client = peer + relay.** Since it already talks out + receives data, it also **hosts the linker-relays** (the Mamba-shard relays, `branch-linker-transport` FR-5). No separate relay infrastructure; full P2P self-management on fully-local models.
- **Federated linker-relays do the routing** (not a central router). "How smart" each relay is (forward / route / orchestrate) and broker-E2E vs trusted-decrypt are **per-relay knobs**.
- **Trust is a phased MENU a client opts into (NOT a launch blocker).** Modes: (a) **private/trusted-group** (trust members) — now; (b) **certified relay** — route through a Lexame-/operator-self-hosted certified relay (ebook `relay-trust_20260425` institutional framework, **Wave-1 complete**) — now; (c) **trusted-peer relay** — opt into a specific peer's relay you personally trust — now; (d) **P2P reputation / trustless** (`p2p-trust-scoring_20260625`, track 19) — the **incremental** path matured over time. **Certified relays are OPT-IN, not a mandatory hub** — the network runs with none present (federated desktop-relays + reputation); no re-centralization. So PUBLIC networks work *now* via (b)/(c); track 19 progressively earns *trustless* public P2P. PRIVATE/personal nets need none of this.
- **Sharing economy:** peers share produced **branches**, and optionally **training recipes, constitution recipes, datasets** (ties to `dataset_sovereignty` / `dao_gating` + scrt-evolve constitution/taste).

Net: BTM branch fleet (Axis 3) + federated linker-relays (routing + NaCl-box crypto + trust anchor) on peer/desktop hosts; orchestration + sharing as app/SDK layers; Petals/layer-sharding (Axis 1) only as the fallback for branches too big for one peer.

---

## 1. What a "branch" is (confirmed with the scrt-evolve owner)

A branch ≡ a **BTM Expert LM (ELM)** [arXiv 2208.03306]: a standalone, domain-specialized
model, **independently (embarrassingly-parallel) trained**, and **ensemble-able /
parameter-merge-able**. scrt-evolve creates one in either mode:

```
base model               → smaller domain expert                 (specialize / carve)
base model + domain data → smaller domain expert (trained + enhanced)   (corpus →
                           teacher-QA → distill; the data ENHANCES, not just compresses)
```

"Smaller" = a smaller model than the source/teacher. v1 = **small base + specialize**;
true distill-to-smaller (teacher → smaller student) is a later mode.

---

## 2. Branch-Train-Merge maps onto the two-repo split

| BTM leg | Repo | Owns |
| :-- | :-- | :-- |
| **Branch + Train** | **scrt-evolve** (Rust CLI + Python ML) — the **branch factory** | corpus → `discover` → **teacher QA** → distill smaller domain expert → **eval-gate** → GGUF + manifest → **branch registry**; the local `BranchRouter`; synthesizes the **Mamba linker** head |
| **Merge** | **hivemind** (this repo) — the **serve + ensemble fabric** | discover/host branches on peers; **route per-request**; **ensemble** top-k branches (BTM merge = weighted output average by domain posterior) and/or parameter-merge to collapse a fleet; carry the **wire protocol** for cross-peer handoff (incl. the linker state) |

---

## 3. The integration contract (what scrt-evolve hands you)

### 3a. Branch artifact = `{ GGUF + manifest.json }` — **manifest SCHEMA v2** (RESOLVED 2026-06-27)

> Owned by track `evolve-branch-contract_20260627` (FR-4). Schema v2 is additive over v1:
> the three new fields are optional and a v1 manifest upgrades to v2 defaults (all null). A
> v2 manifest is refused by a v1-only reader (no silent guess).

- **GGUF**: Q4_K_M, llama.cpp-servable (handles the hybrid Mamba SSM state the HF
  forward OOMs on).
- **manifest.json** (v2 — the binding schema; code on both sides asserts against this):
  ```jsonc
  {
    "name": "legal-tools",
    "base_model": "granite-eval-0.5b",
    "domain": "legal/tool-calling",
    "corpus_descriptor": "…what corpus produced it…",
    "router_signature": { "kind": "simhash|embedding|tfidf", "vector": [/* centroid */] },
    "eval_report": { "…gates that admitted it…": 0.0 },
    "lineage": { "parent": "…branch it forked from, if any…" },

    // --- v2 additions (evolve-branch-contract FR-4) ---
    "stateBasisId": "basis:granite-eval-0.5b:v1" /* | null */,  // FR-2; same base_model ⇒ same id
    "distill_descriptor": { /* seam map + student-block spec, or null if not --distill */ },
    "engine_hint": "llama.cpp@<pinned-version>",                // FR-3; bundle ships the match

    "version": "2", "gguf_sha": "…", "created": "ISO-8601"
  }
  ```
  `router_signature` is the **domain descriptor for routing** (c-BTM-style centroid /
  tf-idf / simhash). **This is what your router matches a request against.**
  `stateBasisId` is the **STATE-TRANSFER eligibility key** (§4); `null` ⇒ SWITCH-only.

### 3b. Branch registry = `branches/registry.json`
```jsonc
{ "schema_version": 2, "branches": [ /* v2 manifest, … */ ] }
```
Your peers read this to know what branches exist + their `router_signature`s + `stateBasisId`s.

### 3c. `BranchRouter` — the shared seam (DON'T fork it)
A request → branch resolution. scrt-evolve ships **v1 LOCAL** (classify request vs each
branch's `router_signature` → top-k **local** branches). hivemind extends the **same**
abstraction: resolution returns **`(peer, branch)`** instead of a local branch, plus the
cross-peer handoff. **One routing model, two resolvers (local / remote).** This is the
agreement that keeps the local router and the distributed linker from diverging.

> **RESOLVED (§6) 2026-06-27:** remote resolution lives **per-peer + linker-relay broker**
> (`branch-linker-transport` FR-5 / §0.5), not a central router. Ensemble policy =
> `single_best` default, `average_topk` opt-in (`branch-substrate` FR-7). Fan-out when K peers
> host the same branch = **`first_of_k`** (take first valid, cancel the rest; ADR D6).

### 3d. `BranchServer` engine interface — **v1 backend = scrt-evolve native candle (SUPERSEDED 2026-07-04, ADR D8)**

> **[SUPERSEDED 2026-07-04 — ADR D8]** The original decision (below, struck) was
> "bundle + spawn `llama.cpp`." scrt-evolve **track 39** (native candle inference,
> spec 2026-07-03) retired the llama.cpp sidecar: *"This is the ONLY serving path —
> the llama.cpp sidecar is retired, not wrapped."* So the v1 `BranchServer` backend
> is **scrt-evolve's native candle engine, linked into `hivemind-desktop` as a Rust
> crate** (in-process; no subprocess, no bundled llama.cpp). The interface stays
> engine-agnostic. `engine_hint` in the manifest still records the producing engine
> for interop, but hivemind serves via the linked evolve crate, not a spawned binary.
> **GGUF remains an EXPORT / interop format** (lexame sharing, other runtimes;
> candle's quantized loader consumes it) — not the hivemind serving mechanism.
> `branch-substrate` FR-2 implements the crate-linked serving on the hivemind side.

~~The whole-GGUF serving unit. **v1 backend = bundle + spawn `llama.cpp`** (a pinned server
binary run as a child process, Unsloth-v1 style, over stdio/HTTP), serving a whole branch GGUF
(handles the hybrid Mamba SSM state the HF forward OOMs on). The interface is engine-agnostic
(a future first-party engine can implement it without changing the routing/serving contract),
but **only llama.cpp ships in v1**. The pinned binary version is recorded in `engine_hint` so
hivemind's desktop bundle ships the matching build. This is what `branch-substrate` FR-2
implements on the hivemind side.~~

### 3e. `tier` propagation (RESOLVED 2026-06-27, ADR D5) — sovereignty inherits to the branch

A branch built from `private`-tier `TrainingRecord`s (see `dataset_generator_contract` /
`dataset_sovereignty`) **inherits `private`** in its manifest/capability. hivemind dispatch
**refuses to route a `private`-tier branch** to other peers — private = local-only. Mixed-tier
training data inherits the **most-restrictive** tier. The dispatch tier-gate is **new code** on
the hivemind side (`branch-p2p-serve-consume` FR-3; coordinator + dispatch are tier-blind today).

---

## 4. The Mamba linker (your cross-peer handoff mechanism)

- **Why Mamba for the linker**: the SSM state is a **constant-size** summary of the
  sequence → a **bounded cross-peer handoff payload** (vs a KV-cache that grows with
  context). This is the bandwidth mechanism for "take the bandwidth hit for expanded
  domain" — the hit is *bounded*, not context-proportional.
- **De-risk status (DONE, scrt-evolve side)**: a from-scratch Mamba2 block learns a
  transformer layer's map via seam distillation on an 8 GB GPU — **capability confirmed,
  data-limited** (val delta-cos 0.51 → 0.74 as calibration data scaled 16→512 seqs;
  reconstructed-output cosine 0.95). So evolve **can** synthesize the linker head.
- **The fork (RESOLVED 2026-06-27 — BOTH paths, basis-gated):**
  - **SWITCH** — route the request, ship prompt / partial text. Trivial, c-BTM-clean,
    per-request. The **always-safe default**; used wherever a shared basis is absent.
  - **STATE-TRANSFER** — ship the bounded SSM state so peer B continues peer A's context.
    **Now in scope** (was "research bet / reserve"). evolve builds the **shared-state-basis
    mode** (`evolve-branch-contract` FR-2): **branches off the same `base_model` share a basis**
    (deterministic `stateBasisId` derived from `base_model`) and are STATE-TRANSFER-eligible
    with each other. `stateBasisId` mismatch or `null` ⇒ falls back to SWITCH. The grouping rule
    is mechanical (same base ⇒ same basis), no coordination needed. The linker head + its
    `stateBasisId` are exported per branch (FR-5); `branch-linker-transport` FR-2/FR-3 ship the
    `purpose:"ssm_state"` HMTF frame and gate on the basis match.

---

## 5. What this changes in your current code

- The expert-dispatch wire protocol (`expert_id`-routed frames) **generalizes to BRANCH
  dispatch** (`branch_id` / peer-routed).
- **Redundant hosting + first-response resolution** (`HANDOFF.md` "High value" #3) now
  applies **per-branch** (host branch X on K peers, resolve on first response). Reuse the
  expert-swarm machinery.
- **"Distribute all 40 MoE blocks"** (`HANDOFF.md` open #2) is **moot for the product
  path** — you host whole branches, not one model's sharded experts. Keep it only if you
  still want single-large-model distribution as a separate mode.
- **Routing is per-request** (domain classifier over `router_signature`s), **not
  per-token.** This is the load-bearing change that makes traffic sparse + local.

---

## 6. Open decisions — **ALL RESOLVED 2026-06-27** (track `evolve-branch-contract_20260627`)

- ~~`manifest.json` + `registry.json` schema~~ → **RESOLVED: schema v2** (§3a/§3b). Adds
  `stateBasisId`, `distill_descriptor`, `engine_hint`; version bumped 1→2; additive + back-compat.
- ~~Wire protocol: SWITCH vs STATE-TRANSFER~~ → **RESOLVED: both, basis-gated** (§4). SWITCH is the
  always-safe default; STATE-TRANSFER between same-`base_model` branches via `stateBasisId`.
- ~~Where remote `BranchRouter` resolution lives~~ → **RESOLVED: per-peer + linker-relay broker**
  (§3c note; `branch-linker-transport` FR-5), no central router.
- ~~Merge/ensemble policy~~ → **RESOLVED: `single_best` default, `average_topk` opt-in** (§3c note;
  `branch-substrate` FR-7). Fan-out across K hosting peers = `first_of_k` (ADR D6).
- **NEW, RESOLVED:** inference engine = ~~llama.cpp bundle+spawn~~ **scrt-evolve native candle,
  crate-linked** (§3d, **SUPERSEDED 2026-07-04 ADR D8**); `tier` inherits to the branch and gates
  dispatch (§3e, ADR D5); shared-basis grouping = **same `base_model`** (§4).

### Binding decision records cited above
ADR `conductor/tracks/_shared/substrate-adr.md` — D1 (Tauri+Svelte shell), D2 (relay owns
DAO/rewards), D3 (sidecar coordinator), D4 (Elixir scaffold deleted), D5 (private = local-only,
tier inherits to shard), D6 (`first_of_k` fan-out). Session 2026-06-27 locks: ~~llama.cpp engine~~
(superseded by D8, 2026-07-04 — crate-linked candle engine), both-paths-basis-gated, same-base basis
grouping, daemon = sole VRAM authority, rewards deferred to v2.

---

## 7. Source pointers (scrt-evolve repo — may not be on this filesystem)

- Feasibility research report: `scrt-evolve/.omc/research/hybrid-mamba-moe-synthesis-2026-06-25.md`
- Linker de-risk + results: `scrt-evolve/bench/seam_distill/{seam_distill_tinyllama.py, RESULTS.md}`
- Canonical papers: **BTM 2208.03306** · **c-BTM 2303.14177** · Mixtral 2401.04088 ·
  MOHAWK 2408.10189 · Mamba-in-Llama 2408.15237 · Drop-Upcycling 2502.19261

---

## 8. Feature-request queue (hivemind → evolve)

This brief flows evolve → hivemind. The reverse channel — engine capabilities the
hivemind desktop needs FROM scrt-evolve (dataset store/provenance API, human-curation
verdicts, compute-authority hooks, runtime corpus registry) — lives in
`conductor/tracks/_shared/evolve-cli-feature-requests.md`. It is the request queue;
the evolve side responds by editing that file in place with status markers, the same
way §6 resolutions were recorded here. First batch R1–R4 REQUESTED 2026-07-16.
