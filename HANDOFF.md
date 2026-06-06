# Handoff: Granite MoE Distributed-Expert Pipeline

This document is the catch-up brief for the next session. It assumes you
are stepping into a fresh Claude context with no memory of what came
before.

## What this project is

We're building HiveMind / Lexame's distributed LLM inference: a network
of peers and relays that share the work of running large models, with
the model split (sharded) into pieces hosted on different machines and
the wire protocol carrying tensor activations between them. The ebook at
`ebook/10-granular-specialist-architecture.md` is the canonical design
document — read it before doing any new architecture work in this area.

The hivemind-models repo (where you are now) is the experimental
playground for **the model side of the protocol**: sharding a real
model, validating numerical correctness, and measuring the cost of
distribution.

## Current status (as of this handoff)

We have an empirically validated, numerically correct pipeline that:

1. **Loads IBM Granite 4.0-H Tiny** (40-layer hybrid Mamba/Transformer
   MoE, 64 experts per layer, top-6 routing, ~14 GB fp16). Granite is
   loaded via a custom streaming sharder that avoids the HF
   `from_pretrained` OOM by reading safetensors files memory-mapped one
   tensor at a time. Shards are written to
   `output/granite-tiny-q4g64/` as embed + 4 layer groups + head.

2. **Validates against a llama.cpp GGUF reference** (top-1 = `' Paris'`
   for "The capital of France is"). Every distributed configuration
   we've tested produces bit-identical logits to this reference.

3. **Distributes one MoE block (layer 0) over WebSocket** to a
   configurable number of expert servers (1 to 64). Each server hosts
   N consecutive experts, routed by an `expert_id` field in the binary
   wire frame. The coordinator runs the gate locally, dispatches the
   expert calls in parallel via a per-server thread pool, and
   recombines the weighted outputs.

4. **Has been rigorously benchmarked on Windows and WSL2** with a clean
   methodology: 3 warmup dispatches + 10 timed dispatches per density,
   median/stdev reported. The data and the cross-OS finding live in
   chapter 10 §10.2 of the ebook and in
   `output/expert_density_findings.md`.

The Windows numbers are great (13-15 ms dispatch floor at 4-16 servers).
The WSL2 numbers are bad (290-330 ms at every multi-server config).
**The WSL2 result is a virtualization artifact**, not a Linux
characteristic — WSL2 routes localhost through Hyper-V virtual
networking. Native Linux on bare metal is unmeasured and expected to
look more like Windows.

## What is open

Listed roughly by importance and tractability.

### High value, well-scoped

1. **Native Linux measurement.** The chapter currently has Windows and
   WSL2. The most-likely-to-be-used environment (bare-metal Linux relays)
   is the missing third data point. Set up a Linux box (cloud VM with
   real network access, not a VM-inside-VM), pull this repo, run
   `scripts/sweep_expert_density.py --warmup 3 --iters 10 --repeats 1
   --output output/expert_density_sweep_linux.json`. The expected result
   is "looks like Windows" but we shouldn't claim that without measuring.
   Add a §10.2.7 to chapter 10 with the data.

2. **Multi-layer distribution.** All measurements distribute exactly
   one MoE block (layer 0). The chapter projects that distributing all
   40 MoE blocks would cost ~560 ms per forward on Windows but does not
   actually measure it. Modify the coordinator to patch multiple layers,
   not just one. Measure with 1, 2, 5, 10, 20, 40 layers distributed.
   Find out whether the cost is truly additive or whether warm-cache
   amortization across layers helps. This is probably the most
   important unanswered question.

3. **Redundant expert hosting with first-response resolution.** The
   chapter argues (in §10.2.5 and §10.4.3) that the WSL2 penalty
   matters less in production because redundant peers + first-response
   resolution turn worst-case-peer latency into best-of-N. **This is
   currently unmeasured.** Extend `launch_expert_swarm.py` to spawn K
   identical copies of each expert on different ports. Extend the
   coordinator's `ExpertPool.forward_many` to dispatch each call to K
   peers and resolve on first response (cancel the others). Re-run the
   sweep. The hypothesis is that the WSL2 tail latency collapses to
   the median, validating the architectural argument.

### Medium value

4. **Speculative routing at the expert layer.** Chapter 13 of the ebook
   develops speculative routing at the semantic-specialist layer. The
   same mechanism (dispatch to top-N most-likely experts before the
   router finalizes, cancel losers) should apply at the expert layer.
   This is the "most likely path to push the dispatch floor below
   10 ms" mentioned in §10.9. Build a small prototype: run the router
   speculatively in parallel with optimistic dispatch to e.g. the
   top-12 most-frequently-hit experts based on prior calls, finalize
   when the router lands. Measure dispatch_ms vs the current
   serialized router-then-dispatch path.

5. **Real multi-machine network test.** Everything has been localhost
   (Windows loopback or WSL2 virtualized loopback). Real production
   latencies depend on the actual network between coordinator and
   peers. Provision two cloud VMs in the same region, run the
   coordinator on one and the expert swarm on the other, sweep
   densities. The hypothesis: per-call RTT goes up but the U-shape
   shifts only upward, not changing fundamentally.

6. **Larger models.** Granite-tiny has 4.7 MB per expert. Mixtral-8x7B
   has ~700 MB per expert; DeepSeek-V3 has ~3 GB per expert. The
   dispatch overhead is approximately constant per call, but the
   compute time per expert call scales with expert size. The current
   sweep is dispatch-overhead-dominated; at Mixtral scale the per-call
   compute would matter more. Measure where the crossover happens.

### Lower priority but interesting

7. **GPU dispatch.** Everything is CPU. With GPU, the per-call expert
   compute drops by 10-50x (the actual matmul becomes near-free) but
   the dispatch overhead doesn't change. The relative cost of
   distribution would worsen. This characterizes the GPU operating
   regime, which is what production inference will actually use.

8. **Energy cost.** The original chapter 10 estimated 30-50% energy
   reduction for granular specialists vs monolithic models. We measured
   latency, not energy. Energy cost specifically for distributed MoE
   (which chains thousands of small messages) needs its own analysis.

9. **Wire-protocol optimization.** Current binary frames are simple
   but unoptimized: one expert call per frame, fixed-size header,
   numpy.tobytes() serialization. Multiplexing multiple expert calls
   into one frame, using shared memory for same-machine deployments,
   and reducing Python-side bookkeeping could push the Windows floor
   below 10 ms.

## How to get oriented in the codebase

Required reading, in order:

1. **`ebook/10-granular-specialist-architecture.md`** — the design
   document. Read §10.1-§10.5 at minimum. §10.5.5 is the scientific-
   process log including all the dead-ends; reading it will save you
   from making the same mistakes.

2. **`output/expert_density_findings.md`** — the parallel
   reproducibility doc with both OS results side-by-side.

3. **`scripts/moe_coordinator.py`** — the end-to-end test harness.
   `make_distributed_moe_layer()` is where the layer-0 MoE block
   gets monkey-patched to dispatch over WebSocket. `ExpertPool` is
   the connection manager.

4. **`scripts/expert_ws_server.py`** — the WebSocket server. Wire
   protocol is documented in the docstring.

5. **`scripts/sweep_expert_density.py`** — the benchmark harness.
   The `_listening_pids_by_port` + `kill_listeners_on_ports` +
   `wait_ports_free` block is the cleanup machinery you must
   preserve — without it, port collisions silently kill the sweep.

6. **`src/architectures/granite.py`** — the PyTorch handler for
   Granite. Implements the abstract `ArchitectureHandler` protocol
   defined in `src/architectures/base.py`. The handler is what runs
   when the coordinator forwards through a layer-group shard.

7. **`scripts/convert_granite_streaming.py`** — the memory-bounded
   sharder. If you need to re-shard the model (changed quantization,
   different layer groups, etc.), this is the entry point.

## How to verify your changes are correct

After any modification touching the model, handler, sharder, or
coordinator:

1. **Reproduce the reference**: `python scripts/gguf_reference.py`
   should still print top-1 = `' Paris'`. (Only do this if you have
   the GGUF locally; it's not strictly needed if you trust the
   existing `output/granite_reference.json`.)

2. **In-process correctness**: `python scripts/verify_granite_handler.py`
   should exit 0 with "PASS: top-1 tokens match the reference". This
   runs the full pipeline in one process, no networking. If this
   fails, the bug is in the handler or sharder, not the wire protocol.

3. **Wire-protocol correctness**: spawn one expert server, run
   `python scripts/test_expert_wire.py`. Cosine similarity must be
   `1.000000`. If this fails, the bug is in the binary frame
   serialization on either client or server.

4. **End-to-end distributed correctness**: spawn a swarm at any
   density, run `python scripts/moe_coordinator.py --warmup 0 --repeats 1`.
   Top-1 must be `' Paris'`. If this fails after the previous three
   pass, the bug is in the coordinator's gate / dispatch / recombine
   logic.

Never skip steps 1-3 before claiming step 4 means anything. A wrong
output that "happens to match" can mask a bug; only sequential validation
proves the distribution is doing what you think.

## Environment notes

- **Windows venv**: `.venv313/` (Python 3.13, torch 2.7.1+cu118)
- **WSL2 venv**: `.venv_wsl/` mounted from Windows at
  `/mnt/c/Users/atooz/Documents/Escherbridge/laxame-hivemind/hivemind-models/.venv_wsl/`
  (Python 3.12, torch 2.12.0+cpu).
  - Activate with `.venv_wsl/bin/python ...` from inside WSL.
- The shards on disk are accessible from both OSes via the Windows
  mount, no need to re-shard for cross-OS tests.

## Memory and process discipline (important)

The sweep harness now does proper port cleanup, but the model is
memory-heavy. Two operational rules learned the hard way:

1. **Close LM Studio / other model hosts before running the sweep.**
   They hold multi-GB resident weight state that competes with the
   coordinator's 3.4 GB layer-group load.

2. **Verify ports 9800-9863 are free before launching a swarm.** The
   sweep does this automatically; manual coordinator invocations
   don't. Use `Get-NetTCPConnection -LocalPort 9800..9863 -State Listen`
   (PowerShell) or `netstat -ano | findstr LISTENING | findstr 980`
   to check.

## What success on the next session looks like

If you complete #1 (native Linux measurement), the chapter gains a
critical third data point and the "OS-dependent shape" claim becomes
properly supported.

If you complete #2 (multi-layer distribution), the chapter answers its
most important open question and the "full-model distribution is
infeasible" claim moves from projected to measured.

If you complete #3 (redundant expert hosting), the chapter's
architectural argument that "redundancy fixes WSL2-style tail latency"
becomes measured rather than asserted.

Any one of these is a meaningful session. All three is a publishable
result.

## What success means for the project

The combination of:
- distributed MoE expert dispatch (this session, done)
- multi-relay routing with redundant peers (open)
- speculative routing at the expert layer (open)
- token-incentive participation (protocol exists; economics not done)

is the architectural floor on which the rest of the HiveMind /
Lexame stack rests. If these layers are fast, correct, and
fault-tolerant, the higher layers (semantic specialist routing,
multi-hop chains, etc.) have something solid to build on. If these
layers are slow or wrong, nothing above them works.

The single-session win to keep aiming at: **show that, given enough
redundant peers, the system can serve a token at sub-300 ms
end-to-end on a real network**. That's the threshold for interactive
use, and everything in chapters 10-13 is in service of it.

Good luck.
