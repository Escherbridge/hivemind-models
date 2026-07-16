# Expert Coordinator -- Operations Runbook

Service: `expert-coordinator` (Railway, region `us-east4`)
Code: `hivemind-models/scripts/expert_coordinator.py` (+ `_wire.py`)
Spec: `conductor/tracks/expert-coordinator_20260606/spec.md`
Plan: `conductor/tracks/expert-coordinator_20260606/plan.md`
Wire canon: `conductor/tracks/_shared/wire-frames.md` Sections 1, 2, 4.

## 1. What this service does

The expert coordinator is the seam between WebGPU browsers running the
Granite-tiny layer-5 MoE router and the 3-region expert pool (
`expert-us-west`, `expert-us-east`, `expert-eu`). It owns exactly three
responsibilities:

1.  One persistent WSS connection per browser at `/ws`.
2.  Fan-out of browser-chosen `(expert_id, hidden_state)` pairs to the
    correct expert region(s) in either `wait_all` or `first_of_k` mode.
3.  Running the real Granite `lm_head` over a 1536-dim hidden state to
    return top-5 token candidates so the demo renders human tokens.

The service does **not** run the router (browser owns it), host experts
(expert services own them), or do streaming, KV cache, or auth.

The browser <-> coordinator wire is JSON over WSS (Sec. 1 of wire-frames).
The coordinator <-> expert wire is the existing binary protocol from
`_wire.py` (Sec. 2). The browser never speaks the binary protocol.

## 2. Deployment

### 2.1 First-time deploy

1.  `git checkout track/expert-coordinator_20260606` in `hivemind-models`.
2.  Push the branch to Railway via the dashboard (or `railway up`).
3.  In the Railway service settings:
    -   Build config: `Dockerfile.coordinator`
    -   Deploy region: `us-east4` (matches the iad-adjacent S3 bucket).
    -   Healthcheck path: `/health`
    -   Healthcheck timeout: 240 s (head shard download dominates cold boot).
    -   `sleepApplication: true` (controlled via `railway.coordinator.json`).
    -   Attach a Railway volume mounted at `/app/cache` so the head shard
        survives redeploys.
4.  Set the env vars listed in Section 3.
5.  First boot will download `shard_head.safetensors` (~295 MB) from S3 to
    `/app/cache/shard_head.safetensors`; subsequent boots are ~30 s.

### 2.2 Redeploy (no S3 change)

`git push` to the track branch; Railway auto-redeploys. The head shard is
re-used from the volume, so cold boot is ~30 s. Healthcheck must return
green within 240 s.

### 2.3 Rollback

Railway's deploy history allows one-click revert. The previous deploy's
container image is retained for 24 h. For older rollbacks: `git revert
<sha>` on the track branch and push.

For protocol-level rollback (incompatible wire change): never roll the
expert services back -- the coordinator is the only thing that knows the
binary protocol; if it goes stale, redeploy the coordinator first.

## 3. Environment variables

| Variable                  | Default                                                  | Notes |
|---------------------------|----------------------------------------------------------|-------|
| `PORT`                    | `8080`                                                   | Railway sets this. |
| `EXPERT_US_WEST_URL`      | -- (required for that region)                            | `wss://expert-us-west.up.railway.app` |
| `EXPERT_US_WEST_IDS`      | --                                                       | e.g. `0-31` |
| `EXPERT_US_EAST_URL`      | --                                                       | `wss://expert-us-east.up.railway.app` |
| `EXPERT_US_EAST_IDS`      | --                                                       | e.g. `16-47` |
| `EXPERT_EU_URL`           | --                                                       | `wss://expert-eu.up.railway.app` |
| `EXPERT_EU_IDS`           | --                                                       | e.g. `0-15,32-63` |
| `HEAD_LOCAL_PATH`         | `/app/cache/shard_head.safetensors`                       | volume-backed cache path |
| `HEAD_S3_BUCKET`          | --                                                       | leave unset for local-bake mode |
| `HEAD_S3_ENDPOINT`        | `https://t3.storageapi.dev`                               | matches expert services |
| `HEAD_S3_ACCESS_KEY`      | --                                                       | required if `HEAD_S3_BUCKET` set |
| `HEAD_S3_SECRET_KEY`      | --                                                       | required if `HEAD_S3_BUCKET` set |
| `HEAD_S3_KEY`             | --                                                       | object key, e.g. `shard_head.safetensors` |
| `HEAD_S3_REGION`          | `auto`                                                   | passes through to boto3 |
| `MODEL_ID`                | `ibm-granite/granite-3.0-1b-a400m-instruct`               | HF tokenizer source |
| `UPSTREAM_CALL_TIMEOUT_S` | `0.5`                                                    | per-upstream call timeout |

Missing any single region's pair degrades the service (warn at boot, keep
running with the remaining regions). Missing **all** regions is a
`SystemExit` at boot.

## 4. Verifying a fresh deploy

1.  `curl https://<service>/health` -- expect 200 with `lm_head_ready`
    eventually flipping to `true`.
2.  `curl https://<service>/info` -- verify the expert-to-region map
    matches the ring overlap (every expert id appears on >= 1 region;
    overlapping ids appear on 2).
3.  `wscat -c wss://<service>/ws`
    -   send `{"type":"hello","region":"local-test"}`
    -   expect `{"type":"ready", "n_experts": 64, "regions": [...], ...}`.
4.  Open `scripts/browser_smoke_test.html` in Chrome, point it at the
    deployed `wss://<service>/ws`, and run all four buttons. Every
    `dispatch` response must echo the input payloads bit-identical (the
    upstream services run `COMPUTE_MODE=echo` in dev). Every `lm_head`
    response must list 5 tokens, all with non-empty `token_str`.
5.  Run the microbenchmark:
    ```
    python scripts/bench_lm_head.py --url https://<service> --iters 100
    ```
    Confirm `NFR-1 PASS` (`p50 < 50 ms`, `p99 < 60 ms`).

## 5. Common failures and fixes

### Symptom: `/health` reports `lm_head_ready: false` for > 60 s after green healthcheck

Look for `head shard load failed` in the logs. Causes:

- **S3 credentials wrong.** `head_load_error` field in `/health` will name
  the boto3 exception type. Fix the env var, redeploy.
- **Shard file present but malformed.** `RuntimeError: ... does not contain
  lm_head.weight`. The shard at `HEAD_LOCAL_PATH` is from a different
  model; delete it on the volume and let the next boot re-download.
- **HF tokenizer download failed.** `head_load_error` will reference
  `transformers` or HF Hub. The Dockerfile pre-warms the tokenizer cache,
  but if the operator changed `MODEL_ID` post-deploy, the tokenizer won't
  be cached. Set `MODEL_ID` back to the default and redeploy, or wait
  for the runtime download to complete.

### Symptom: `dispatch` results contain `"error": "upstream_unavailable"` for some expert ids

Hit `/info`. The offending region will show `connected: false` and a
`last_error` field. Fixes:

- **Expert service asleep.** Railway sleep-on-idle hibernates the expert
  service. The first call wakes it (~30 s); the coordinator's circuit
  breaker means the next call after wake-up succeeds. Wait and retry.
- **Expert service crashed.** Check the expert service's own logs in the
  Railway dashboard; restart from there.
- **Mismatched hello.** `degraded: true` on the region means the expert
  service is advertising a different expert-id slice than the coordinator
  expects (the `EXPERT_*_IDS` env var doesn't match what the expert service
  advertises). Reconcile: either fix the coordinator env var or the expert
  service's `EXPERT_IDS`.

### Symptom: all `dispatch` calls fail with `all_regions_failed`

All three upstreams are unreachable. Likely causes:

- **Coordinator network partition.** Check Railway status.
- **All expert services down.** Hit each region's `/health` directly.
- **Coordinator booted before any upstream was reachable.** Restart the
  coordinator -- the supervisor will reconnect on schedule.

### Symptom: `lm_head` responses are slower than 60 ms p99

- Verify the matmul is on CPU (no GPU drift). Logs show `lm_head ready
  vocab=... hidden=...` at boot.
- Memory pressure can cause torch to spill. Check Railway memory metrics;
  bump the instance plan if resident usage approaches 1.5 GB.
- The lm_head path is single-flight (GIL-bound). Concurrent requests
  serialize on the coordinator. For load testing, fan out client side.

### Symptom: `frame_too_large` errors flooding the logs

The browser is sending JSON > 8 MB. This shouldn't happen for the Phase 1
demo shape (k=6, 8 tokens, 1536 hidden, fp16 ~ 24 KB per call). Inspect
the client; either it is shipping too many tokens or its base64 encoder
is broken.

### Symptom: `bad_mode`, `bad_expert_id`, or `bad_shape` from the browser

The browser is talking the wrong wire version. Look at
`expert-coordinator/info` for `protocol_version` and confirm the
hivemind-client uses the same. If they diverge, the browser is on a stale
deploy -- update `hivemind-client` and re-deploy.

## 6. Cost discipline

- `sleepApplication: true` means the service idles to zero billing after
  ~5 min of no inbound WS. Cold-boot wake-up (with the head shard cached
  on the volume) is ~30 s.
- Idle resident memory with the head shard loaded is < 1.5 GB. Don't
  upgrade plan tier unless metrics show sustained > 1.2 GB.
- Egress: each `dispatch` response is `6 * n_tokens * hidden * 2 bytes` of
  fp16 in base64 (~ 32 KB for the demo shape). Trivial against Railway's
  egress allowance.

## 7. Where to look in logs

| Line prefix                               | What it tells you                  |
|-------------------------------------------|-----------------------------------|
| `expert coordinator: regions=[...]`        | boot, env was parsed              |
| `upstream <region>: connecting <url>`      | reconnect supervisor starting     |
| `upstream <region>: connected`             | reconnect supervisor success      |
| `upstream <region>: circuit open for N s`  | failover state engaged            |
| `head shard download complete`             | S3 fetch finished, load next      |
| `lm_head ready: vocab=N hidden=1536`       | head module loaded successfully   |
| `head shard load failed`                   | look at `head_load_error` field   |
| `ws upgrade from origin=... peer=...`      | a browser connected               |
| `ws handler crashed`                       | INTERNAL_ERROR; bug in this code  |
| `STARTUP BANNER: unauthenticated`          | reminder; ignore unless prod      |

## 8. Closed-set error codes (browser-facing)

These are the only codes the coordinator emits on the `/ws` text frame
surface. Adding or removing one is a wire-breaking change.

`expected_hello`, `bad_json`, `frame_too_large`, `unknown_type`,
`bad_mode`, `bad_expert_id`, `bad_shape`, `payload_size_mismatch`,
`lm_head_not_ready`, `upstream_unavailable`, `all_regions_failed`,
`internal_error`.

See `scripts/expert_coordinator.py:ERROR_CODES` for the canonical set and
wire-frames Sec. 1.9 for the recoverability table.

## 9. Out of scope -- explicit non-goals

- Auth, bearer tokens, HMAC, rate limiting (deferred to security track).
- Token streaming (one dispatch -> one result).
- KV cache (browser owns its own).
- Wire-protocol v2 with multiplexed binary frames (deferred).
- Multi-coordinator coordination (one Railway service, one process).

A future security track must layer auth on top before this service goes
public. The startup banner reminds the operator on every boot.
