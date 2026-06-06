"""
MoE coordinator: runs the full Granite pipeline in one process but dispatches
*only* layer L's MoE expert computation to 64 remote WebSocket expert servers.

This is the chapter-10 latency-trap test from §10.1.3, applied empirically
to one MoE layer. Goal: prove the wire protocol gives bit-identical output
to a co-located run, and measure the round-trip cost.

How the swap works
==================

Granite's GraniteMoeHybridMoE.forward (the in-process version):

    h = layer_input.reshape(-1, hidden)
    _, batch_index, batch_gates, expert_size, _ = self.router(h)
    expert_inputs = h[batch_index]
    out = self.input_linear(expert_inputs, expert_size)
    g, u = out.chunk(2, -1)
    out = silu(g) * u
    out = self.output_linear(out, expert_size)
    out = out * batch_gates[:, None]
    result = zeros.index_add(0, batch_index, out)
    return result.view(bsz, length, hidden)

We replace the input_linear/activation/output_linear chunk: instead of
running input_list[i] through the local experts, we ship those tokens to
remote experts over WS, then resume from the gates multiplication.

The router (a single linear) STAYS LOCAL, as the architecture doc requires.

Usage:
    python scripts/moe_coordinator.py \
        --shard-dir output/granite-tiny-q4g64 \
        --experts-dir output/granite-tiny-q4g64/experts_layer_0 \
        --moe-layer 0 \
        --base-port 9800
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from websockets.sync.client import connect as ws_connect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.architectures import LayerShardSpec, get_handler


DTYPE_MAP = {0: torch.float32, 1: torch.float16}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}

# Wire header: op:u8, expert_id:u16, n_tokens:u32, hidden:u32, dtype:u8
_HEADER = struct.Struct("<BHIIB")


def pack_request(expert_id: int, t: torch.Tensor) -> bytes:
    n_tokens, hidden = t.shape
    dtype_code = DTYPE_REVERSE.get(t.dtype, 0)
    if t.dtype not in DTYPE_REVERSE:
        t = t.float()
        dtype_code = 0
    header = _HEADER.pack(1, expert_id, n_tokens, hidden, dtype_code)
    return header + t.contiguous().numpy().tobytes()


def unpack_response(buf: bytes) -> tuple[int, torch.Tensor]:
    op = buf[0]
    if op == 0xFF:
        raise RuntimeError(f"server error: {buf[1:].decode('utf-8', 'replace')}")
    _, expert_id, n_tokens, hidden, dtype_code = _HEADER.unpack_from(buf, 0)
    dtype = DTYPE_MAP[dtype_code]
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    arr = np.frombuffer(buf, dtype=np_dtype, count=n_tokens * hidden,
                        offset=_HEADER.size).reshape(n_tokens, hidden).copy()
    return expert_id, torch.from_numpy(arr)


# ---------- WS expert client pool ------------------------------------------


import threading


class ExpertPool:
    """
    One persistent (sync) WS connection per *server*. Many experts may live
    on one server. Per-server connections are mutexed so two threads don't
    interleave frames on the same socket; parallelism comes from having
    multiple servers, not from concurrent calls into one server.
    """

    def __init__(self, topology: dict) -> None:
        self.host = topology["host"]
        self.n_experts = int(topology["n_experts"])
        self.experts_per_server = int(topology.get("experts_per_server", 1))
        # expert_id -> (port, connection, lock)
        self.servers: list[dict] = []  # one entry per server
        self.expert_to_server: dict[int, int] = {}  # expert_id -> index into self.servers
        for s in topology["servers"]:
            s_idx = len(self.servers)
            self.servers.append({
                "port": int(s["port"]),
                "expert_ids": [int(i) for i in s["expert_ids"]],
                "ws": None,
                "lock": threading.Lock(),
            })
            for eid in self.servers[-1]["expert_ids"]:
                self.expert_to_server[eid] = s_idx
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(len(self.servers), 1)
        )

    def connect_all(self) -> None:
        def one(s: dict) -> None:
            ws = ws_connect(
                f"ws://{self.host}:{s['port']}", max_size=2**24, open_timeout=5,
            )
            hello = ws.recv(timeout=5)
            json.loads(hello)  # verify
            s["ws"] = ws

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.servers)) as ex:
            list(ex.map(one, self.servers))
        print(f"  connected to {len(self.servers)} expert server(s) "
              f"hosting {self.n_experts} expert(s), {self.experts_per_server}/server")

    def close_all(self) -> None:
        for s in self.servers:
            if s["ws"] is not None:
                try:
                    s["ws"].close()
                except Exception:
                    pass
        self.pool.shutdown(wait=False)

    def forward(self, expert_id: int, tokens: torch.Tensor) -> torch.Tensor:
        s = self.servers[self.expert_to_server[expert_id]]
        with s["lock"]:
            s["ws"].send(pack_request(expert_id, tokens))
            raw = s["ws"].recv()
        _, out = unpack_response(raw)
        return out

    def forward_many(
        self, calls: list[tuple[int, torch.Tensor]],
    ) -> list[tuple[int, torch.Tensor]]:
        """
        Dispatch calls in parallel across servers. Calls to experts on the same
        server are serialized by the per-server lock; calls to different
        servers run in parallel via the thread pool.
        """
        # Group by server to give each thread a sequential batch (avoid
        # contending on the same lock from multiple threads at once).
        by_server: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for (e_id, t) in calls:
            s_idx = self.expert_to_server[e_id]
            by_server.setdefault(s_idx, []).append((e_id, t))

        def do_server(items: list[tuple[int, torch.Tensor]]) -> list[tuple[int, torch.Tensor]]:
            out: list[tuple[int, torch.Tensor]] = []
            for (e_id, t) in items:
                out.append((e_id, self.forward(e_id, t)))
            return out

        futures = [self.pool.submit(do_server, items) for items in by_server.values()]
        results: list[tuple[int, torch.Tensor]] = []
        for f in concurrent.futures.as_completed(futures):
            results.extend(f.result())
        return results


# ---------- distributed MoE forward -----------------------------------------


def distributed_moe_forward(
    layer_input: torch.Tensor,
    router: torch.nn.Linear,
    pool: ExpertPool,
    top_k: int,
    hidden_size: int,
) -> tuple[torch.Tensor, dict]:
    """
    Implements GraniteMoeHybridMoE.forward but with remote experts.
    Synchronous; uses ExpertPool.forward_many (thread pool) for parallel dispatch.
    Returns (output, telemetry).
    """
    bsz, length, emb = layer_input.shape
    h = layer_input.reshape(-1, emb)  # [N, hidden]

    # ---- 1. router (local) ------------------------------------------------
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = router(h.float())   # [N, num_experts]
        top_k_logits, top_k_indices = logits.topk(top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(h)

    n_tokens = h.shape[0]
    flat_expert_ids = top_k_indices.reshape(-1)
    flat_gates = top_k_gates.reshape(-1)
    flat_token_ids = (
        torch.arange(n_tokens, device=h.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    )
    sort_idx = torch.argsort(flat_expert_ids, stable=True)
    sorted_expert_ids = flat_expert_ids[sort_idx]
    sorted_token_ids = flat_token_ids[sort_idx]
    sorted_gates = flat_gates[sort_idx]
    expert_size = torch.bincount(sorted_expert_ids, minlength=pool.n_experts)
    router_ms = (time.perf_counter() - t0) * 1000

    # ---- 2. dispatch in parallel to remote experts -----------------------
    t0 = time.perf_counter()
    calls: list[tuple[int, torch.Tensor]] = []
    cursor = 0
    for e_id in range(pool.n_experts):
        n = int(expert_size[e_id].item())
        if n == 0:
            continue
        rows = sorted_token_ids[cursor : cursor + n]
        calls.append((e_id, h[rows]))
        cursor += n

    results = pool.forward_many(calls)
    dispatch_ms = (time.perf_counter() - t0) * 1000

    # ---- 3. recombine outputs --------------------------------------------
    t0 = time.perf_counter()
    # Build a {expert_id: output} map so we can use the original (sorted) cursor order
    out_by_expert = {e: o for e, o in results}
    out = torch.zeros((n_tokens, hidden_size), dtype=h.dtype)
    cursor = 0
    for e_id in range(pool.n_experts):
        n = int(expert_size[e_id].item())
        if n == 0:
            cursor += n
            continue
        rows = sorted_token_ids[cursor : cursor + n]
        gates = sorted_gates[cursor : cursor + n]
        expert_out = out_by_expert[e_id]
        weighted = expert_out.to(out.dtype) * gates[:, None]
        out.index_add_(0, rows, weighted)
        cursor += n
    recombine_ms = (time.perf_counter() - t0) * 1000

    out = out.view(bsz, length, hidden_size)
    return out, {
        "router_ms": router_ms,
        "dispatch_ms": dispatch_ms,
        "recombine_ms": recombine_ms,
        "experts_hit": int((expert_size > 0).sum().item()),
        "total_expert_calls": len(calls),
        "expert_size_max": int(expert_size.max().item()),
    }


# ---------- patching the handler --------------------------------------------


def make_distributed_moe_layer(
    original_layer: torch.nn.Module,
    pool: ExpertPool,
    telemetry: list,
    warmup_iters: int = 0,
    timed_iters: int = 1,
):
    """
    Monkey-patch a GraniteMoeHybridDecoderLayer's MoE block to run remotely.

    For benchmarking, the patched forward can do `warmup_iters` discarded
    dispatches (results thrown away) followed by `timed_iters` measured
    dispatches whose telemetry is appended to `telemetry`. Only the final
    timed dispatch's output is returned so downstream layers see a
    well-formed tensor.
    """
    moe = original_layer.block_sparse_moe
    router = moe.router.layer
    top_k = moe.router.top_k
    hidden = moe.input_size

    def new_moe_forward(layer_input: torch.Tensor) -> torch.Tensor:
        # Warmup: discard results
        for _ in range(warmup_iters):
            _ = distributed_moe_forward(layer_input, router, pool, top_k, hidden)

        # Timed: record telemetry, keep last output
        last_out = None
        for i in range(timed_iters):
            out, tel = distributed_moe_forward(layer_input, router, pool, top_k, hidden)
            tel["iter"] = i
            telemetry.append(tel)
            last_out = out
        return last_out

    moe.forward = new_moe_forward
    return moe


# ---------- main ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path,
                        default=Path("./output/granite-tiny-q4g64"))
    parser.add_argument("--experts-dir", type=Path,
                        default=Path("./output/granite-tiny-q4g64/experts_layer_0"))
    parser.add_argument("--moe-layer", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=9800)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Discarded MoE dispatch iterations before timing")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Timed MoE dispatch iterations to record")
    args = parser.parse_args()

    manifest = json.loads((args.shard_dir / "manifest.json").read_text())
    model_id = manifest["model_id"]
    layer_groups = [tuple(g) for g in manifest["layer_groups"]]

    # Connect to expert servers using the topology written by the launcher.
    topo_path = args.experts_dir / "topology.json"
    if not topo_path.exists():
        raise SystemExit(
            f"no topology at {topo_path} -- run launch_expert_swarm.py first"
        )
    topology = json.loads(topo_path.read_text())
    n_experts = int(topology["n_experts"])
    eps = int(topology.get("experts_per_server", 1))
    ports = [s["port"] for s in topology["servers"]]
    print(f"connecting to {len(ports)} expert server(s) "
          f"(layer={topology['layer']}, {eps} experts/server) on ports "
          f"{min(ports)}..{max(ports)}")
    pool = ExpertPool(topology)
    pool.connect_all()

    # Build the pipeline (same as verify_granite_handler.py).
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"
    handler = get_handler("granitemoehybrid")
    tok = AutoTokenizer.from_pretrained(model_id)
    input_ids = tok.encode(args.prompt, return_tensors="pt")

    device = torch.device("cpu")
    telemetry: list = []

    print(f"\nprompt: {args.prompt!r}")
    print(f"tokens: {input_ids.tolist()}")

    # Embed shard.
    embed_state = load_file(str(args.shard_dir / "shard_embed.safetensors"))
    embed_mod = handler.build_embed(config, embed_state, device)
    del embed_state
    hidden = handler.forward_embed(embed_mod, input_ids)
    del embed_mod

    # Run layer groups, patching the MoE in layer args.moe_layer.
    t_pipeline = time.perf_counter()
    for (start, end) in layer_groups:
        print(f"\n[layers {start}..{end}] loading shard ...")
        shard_state = load_file(str(args.shard_dir / f"shard_layers_{start}_{end}.safetensors"))
        layer_mod = handler.build_layers(
            config, shard_state, LayerShardSpec(layer_start=start, layer_end=end), device,
        )
        del shard_state

        if start <= args.moe_layer <= end:
            offset = args.moe_layer - start
            target_layer = layer_mod["layers"][offset]
            print(f"  patching layer {args.moe_layer} MoE to dispatch to {n_experts} remote experts "
                  f"(warmup={args.warmup}, timed={args.repeats})")
            make_distributed_moe_layer(
                target_layer, pool, telemetry,
                warmup_iters=args.warmup,
                timed_iters=args.repeats,
            )

        t0 = time.perf_counter()
        hidden = handler.forward_layers(layer_mod, hidden)
        print(f"  -> {list(hidden.shape)} in {(time.perf_counter() - t0) * 1000:.0f}ms")
        del layer_mod

    # Head.
    head_state = load_file(str(args.shard_dir / "shard_head.safetensors"))
    head_mod = handler.build_head(config, head_state, device)
    del head_state
    logits = handler.forward_head(head_mod, hidden)
    pipeline_ms = (time.perf_counter() - t_pipeline) * 1000

    # Top-5.
    last = logits[0, -1].float()
    topk = torch.topk(last, k=5)
    print("\nTop-5 predictions:")
    for tok_id, lv in zip(topk.indices.tolist(), topk.values.tolist()):
        tok_str = tok.decode([tok_id])
        safe = tok_str.encode("ascii", errors="replace").decode("ascii")
        print(f"  {tok_id:6d}  {lv:8.3f}  {safe!r}")

    top1 = tok.decode([int(topk.indices[0])])
    print(f"\nDistributed-MoE top-1: {top1!r}")

    match = True
    ref_path = Path("output/granite_reference.json")
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        ref_top1 = ref["top_k_tokens"][0]["token"]
        print(f"GGUF reference top-1: {ref_top1!r}")
        match = top1.strip().lower() == ref_top1.strip().lower()
        print("\n" + ("PASS" if match else "FAIL") + ": "
              f"top-1 {'matches' if match else 'differs from'} reference")

    if telemetry:
        print(f"\n--- MoE dispatch telemetry ({len(telemetry)} timed call(s) on layer {args.moe_layer}) ---")
        for i, t in enumerate(telemetry):
            print(f"  iter #{i}: router={t['router_ms']:.1f}ms  "
                  f"dispatch={t['dispatch_ms']:.1f}ms  "
                  f"recombine={t['recombine_ms']:.1f}ms  "
                  f"experts_hit={t['experts_hit']}/{n_experts}  "
                  f"calls={t['total_expert_calls']}  max_size={t['expert_size_max']}")

        # Aggregate stats
        if len(telemetry) >= 2:
            import statistics as st

            def stats(name: str) -> str:
                vs = [t[name] for t in telemetry]
                return (f"mean={st.mean(vs):.2f}ms  "
                        f"median={st.median(vs):.2f}ms  "
                        f"stdev={st.stdev(vs):.2f}ms  "
                        f"min={min(vs):.2f}ms  "
                        f"max={max(vs):.2f}ms")

            print(f"\n--- aggregate stats over {len(telemetry)} timed iters "
                  f"(warmup={args.warmup} discarded) ---")
            print(f"  router    : {stats('router_ms')}")
            print(f"  dispatch  : {stats('dispatch_ms')}")
            print(f"  recombine : {stats('recombine_ms')}")

    print(f"\nTotal pipeline (excl. embed/head load + tokenize): {pipeline_ms:.0f}ms")

    pool.close_all()
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
