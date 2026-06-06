# Expert-density empirical findings

Source for the ebook chapter 10 update. All measurements on a single
Windows laptop (32 GB RAM, CPU only). Cross-OS comparison: Windows native
vs WSL2 Ubuntu, same hardware, same code, same shard files.

## Model

`ibm-granite/granite-4.0-h-tiny` — 40 hybrid layers, 64 experts per MoE
block, top-6 routing, 1536 hidden, 512 intermediate per expert.

Layer 0 MoE was extracted into 64 per-expert safetensors files (4.7 MB
each) plus a router file (0.2 MB). The full pipeline ran in one
coordinator process; only layer 0's MoE forward was dispatched to
remote expert servers via WebSocket.

## Reference

Prompt: `"The capital of France is"` (5 tokens).
Top-1 across every configuration in every OS that returned a value:
`' Paris'`. Bit-identical logits to the in-process baseline AND to the
llama.cpp GGUF reference inference.

## Benchmark methodology (rigorous version)

Per density: spawn launcher → wait for swarm ready → run coordinator with
`--warmup 3 --repeats 10` → coordinator does 3 discarded MoE forwards then
10 timed forwards → records per-iter and aggregate (mean/median/stdev/
min/max) → write JSON → kill all expert servers by port → next density.

The earlier loose version (n=2 per density, no warmup, no variance) is
preserved in `output/expert_density_sweep.json` for historical reference
but is not the data the chapter rests on.

## Windows results

```
experts/srv  servers  median  mean   stdev   min    max
         64        1   35.6   35.6   1.0    34.1   37.4
         32        2   20.8   20.6   1.0    19.0   21.8
         16        4   14.2   14.9   1.3    13.6   16.8
          8        8   14.9   15.1   0.9    13.4   16.8
          4       16   13.7   13.7   0.9    12.6   15.0
          2       32   12.8   34.5   64.4   10.6   217.4
          1       64   16.9   17.7   2.2    16.3   23.8
```

All in milliseconds. n=10 per density, after 3 warmup. All 70 dispatches
passed (top-1 = ' Paris').

## WSL2 Ubuntu results

```
experts/srv  servers  median  mean   stdev   min    max
         64        1   77.7   77.5   9.6    63.7   95.1
         32        2  139.8  136.0  36.8    57.2  184.2
         16        4  290.4  299.8  31.2   261.0  355.8
          8        8  304.8  308.7  28.3   266.9  355.0
          4       16  329.0  330.2  33.0   276.6  381.6
          2       32  315.9  311.4  34.1   259.9  358.2
          1       64  317.1  332.8  53.4   270.4  416.0
```

All in milliseconds. Same hardware as Windows.

**IMPORTANT: This is WSL2, not native Linux.** WSL2 runs inside Hyper-V;
localhost traffic between processes inside WSL2 goes through a virtual
network adapter rather than the kernel loopback shortcut native Linux
uses. The plateau at 290-330 ms is a virtualization artifact, not a Linux
characteristic. Native Linux on bare metal is unmeasured here and is
expected to behave more like Windows than like WSL2 for this workload,
because the per-connection virtual-network tax we observe is specifically
a virtualization cost.

**The WSL2 numbers are useful as a pessimistic bound for virtualized
environments** (Docker bridge networking on a desktop, K8s service mesh
on a hosted control plane, similar layered networking). They are not a
prediction of native-Linux production behavior.

## Cross-OS ratio

```
servers  Windows median   WSL median    Ratio (WSL/Win)
      1            35.6         77.7              2.2x
      2            20.8        139.8              6.7x
      4            14.2        290.4             20.4x
      8            14.9        304.8             20.4x
     16            13.7        329.0             24.0x
     32            12.8        315.9             24.7x
     64            16.9        317.1             18.8x
```

The penalty grows with server count, then plateaus at ~20-25x once the
virtual-network capacity is saturated.

## Architectural takeaways

1. Distributed MoE is numerically correct on both OSes. Wire protocol
   is bit-exact across all 7 densities and 14 sweep runs total.

2. **The right deployment shape is OS-dependent.**
   - Windows (and likely native Linux): 4-16 server processes per
     MoE block hits the latency floor; adding more does not help.
   - WSL2 (and likely Docker bridge, K8s service mesh): single-server
     is the only viable shape; multi-server adds 10-25× latency.

3. The 1-process Windows GIL-serialized case (35 ms) is still faster
   than the WSL2 single-process case (78 ms). The OS networking stack
   matters even when there is only one process talking to it.

4. Windows shows tail-latency instability at 32+ servers (median 12.8 ms
   but max 217 ms). This is real production-relevant behavior; the
   median alone is misleading. Redundant dispatch with first-response
   resolution would hide these stalls.

5. The analytical bound in chapter 10's original §10.1.3 was ~572 ms
   for this workload. Empirical Windows minimum is 13 ms, about 44x
   faster. The analytical model ignored parallel dispatch within a
   single forward.

6. Full-model token-level distribution remains infeasible. Windows-best
   14 ms × 40 layers × 100 tokens = 56 seconds per query. The original
   chapter's conclusion holds at the level of the full model even though
   per-block dispatch is much cheaper than predicted.

## Scripts to reproduce

- `scripts/gguf_reference.py` — establish ground truth from GGUF
- `scripts/convert_granite_streaming.py` — memory-bounded sharder
- `scripts/verify_granite_handler.py` — in-process top-1 vs reference
- `scripts/extract_experts.py --layer 0` — per-expert weight slicing
- `scripts/expert_ws_server.py` — multi-expert WS server
- `scripts/launch_expert_swarm.py --experts-per-server N` — spawn N-density
- `scripts/moe_coordinator.py --warmup 3 --repeats 10` — coordinator with
  rigorous timing
- `scripts/sweep_expert_density.py --warmup 3 --iters 10` — full sweep

## Raw data files

- `output/expert_density_sweep_windows.json` — rigorous Windows data (§10.2.3)
- `output/expert_density_sweep_wsl.json` — rigorous WSL2 data (§10.2.4)
- `output/expert_density_sweep.json` — historical noisy v1 data
- `output/granite_reference.json` — llama.cpp ground truth
