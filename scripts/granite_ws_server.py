"""
WebSocket Granite shard server.

For the first iteration this server holds the *full* Granite GGUF and answers
generic completion requests over a persistent WebSocket connection. Once the
wire protocol is proven, we'll add layer-range and per-expert variants.

Wire protocol (JSON over text frames — keeps it debuggable for now; can
switch to msgpack/binary once tensor shipping is real):

  client -> server:
    {"type": "ping"}
    {"type": "complete",
     "id": "<arbitrary correlation id>",
     "prompt": "<text>",
     "max_tokens": <int>,
     "temperature": <float>,
     "top_k_logprobs": <int|null>}        # report top-k of FIRST token only

  server -> client:
    {"type": "pong"}
    {"type": "ready", "model": {"path": "...", "arch": "..."},
                       "metadata": {<a few hand-picked fields>}}
    {"type": "complete_result",
     "id": "<corr id>",
     "completion": "<text>",
     "tokens_generated": <int>,
     "first_token_top_k": [{"token": "...", "logprob": <float>}, ...] or null,
     "elapsed_ms": <float>}
    {"type": "error", "id": "<corr id>", "message": "..."}

Usage:
    python scripts/granite_ws_server.py \
        --gguf C:/path/to/granite-4.0-h-tiny-Q4_K_M.gguf \
        --port 9700
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import websockets
from llama_cpp import Llama


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("granite_ws_server")


DEFAULT_GGUF = Path(
    "C:/Users/atooz/.lmstudio/models/lmstudio-community/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf"
)


def metadata_subset(md: dict) -> dict:
    """Pick a handful of useful fields from llama.cpp's metadata dict."""
    keys = [
        "general.architecture",
        "general.name",
        "granitehybrid.block_count",
        "granitehybrid.embedding_length",
        "granitehybrid.expert_count",
        "granitehybrid.expert_used_count",
        "granitehybrid.context_length",
    ]
    return {k: md[k] for k in keys if k in md}


class GraniteWSServer:
    def __init__(self, gguf_path: Path, n_ctx: int) -> None:
        self.gguf_path = gguf_path
        self.n_ctx = n_ctx
        logger.info("Loading GGUF %s (n_ctx=%d)", gguf_path, n_ctx)
        self.llm = Llama(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            verbose=False,
            logits_all=True,  # required so we can serve top-k logprobs
        )
        self.arch = self.llm.metadata.get("general.architecture", "unknown")
        self.meta = metadata_subset(self.llm.metadata)
        logger.info("Loaded %s — %s", self.arch, self.meta)
        # llama_cpp.Llama is not thread-safe; serialize completion calls.
        self._lock = asyncio.Lock()

    async def handle(self, ws: websockets.ServerConnection) -> None:
        peer = ws.remote_address
        logger.info("client connected: %s", peer)
        # Greet the client with model identity so it can confirm it's talking
        # to the right shard before sending tensors.
        await ws.send(json.dumps({
            "type": "ready",
            "model": {"path": str(self.gguf_path), "arch": self.arch},
            "metadata": self.meta,
        }))

        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as e:
                    await ws.send(json.dumps({"type": "error", "message": f"bad json: {e}"}))
                    continue

                t = msg.get("type")
                if t == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                elif t == "complete":
                    await self._handle_complete(ws, msg)
                else:
                    await ws.send(json.dumps(
                        {"type": "error", "id": msg.get("id"), "message": f"unknown type: {t!r}"}
                    ))
        except websockets.ConnectionClosed:
            logger.info("client disconnected: %s", peer)

    async def _handle_complete(self, ws: websockets.ServerConnection, msg: dict) -> None:
        cid = msg.get("id")
        prompt = msg.get("prompt", "")
        max_tokens = int(msg.get("max_tokens", 16))
        temperature = float(msg.get("temperature", 0.0))
        top_k_logprobs = msg.get("top_k_logprobs")

        if not prompt:
            await ws.send(json.dumps({
                "type": "error", "id": cid, "message": "prompt is required",
            }))
            return

        t0 = time.perf_counter()
        try:
            # Serialize against other concurrent completion requests on the
            # same llama instance — Llama is not re-entrant.
            async with self._lock:
                # Run the (blocking) inference in a default executor so the
                # event loop stays responsive to ping frames + new clients.
                out = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=1 if temperature == 0.0 else 50,
                        logprobs=top_k_logprobs,
                        echo=False,
                    ),
                )
        except Exception as e:
            logger.exception("completion failed")
            await ws.send(json.dumps(
                {"type": "error", "id": cid, "message": f"{type(e).__name__}: {e}"}
            ))
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000
        choice = out["choices"][0]
        completion = choice["text"]
        tokens_generated = (out.get("usage") or {}).get("completion_tokens") or 0

        first_token_top_k = None
        if top_k_logprobs:
            try:
                top = choice["logprobs"]["top_logprobs"][0]
                first_token_top_k = [
                    {"token": tok, "logprob": float(lp)}
                    for tok, lp in list(top.items())[: int(top_k_logprobs)]
                ]
            except (KeyError, IndexError, TypeError):
                first_token_top_k = None

        await ws.send(json.dumps({
            "type": "complete_result",
            "id": cid,
            "completion": completion,
            "tokens_generated": tokens_generated,
            "first_token_top_k": first_token_top_k,
            "elapsed_ms": elapsed_ms,
        }))
        # Keep the log line short — `completion` can be long.
        safe = completion.encode("ascii", errors="replace").decode("ascii")[:80]
        logger.info(
            "complete id=%s tokens=%s elapsed=%.0fms text=%r",
            cid, tokens_generated, elapsed_ms, safe,
        )


async def main_async(args: argparse.Namespace) -> None:
    if not args.gguf.exists():
        raise SystemExit(f"GGUF not found at {args.gguf}")

    server = GraniteWSServer(args.gguf, args.n_ctx)
    logger.info("Starting WebSocket server on %s:%d", args.host, args.port)
    async with websockets.serve(
        server.handle,
        args.host,
        args.port,
        max_size=2**24,  # 16MB — plenty for tensor blobs once we add those
    ):
        await asyncio.Future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser(description="Granite WebSocket shard server")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9700)
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
