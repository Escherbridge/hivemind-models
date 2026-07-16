# `__tests__/fixtures/`

This directory holds test fixtures for the expert-coordinator correctness
tests.

## `lm_head_reference.npz`

The lm_head correctness fixture. See
`scripts/capture_lm_head_reference.py` for the schema and the two ways to
(re)generate it:

- **Synthetic** (default; works on any machine, no shards):
  ```
  python scripts/capture_lm_head_reference.py --synthetic \
      --out scripts/__tests__/fixtures/lm_head_reference.npz
  ```

- **Granite real** (operator only; needs S3 creds + the local shard dir):
  ```
  python scripts/capture_lm_head_reference.py --granite \
      --shard-dir ./output/granite-tiny-q4g64 \
      --prompt "The capital of France is" \
      --out scripts/__tests__/fixtures/lm_head_reference.npz
  ```

If `lm_head_reference.npz` is missing,
`scripts/__tests__/test_coordinator_lm_head_correctness.py` auto-bootstraps
it from the well-known seed in
`scripts/capture_lm_head_reference.DEFAULT_SEED` so a fresh checkout still
runs green. Commit the bootstrapped synthetic fixture so CI does not have
to regenerate it every run.

Re-capture against the real Granite shard before deploy to validate
semantic correctness; the synthetic fixture only validates the matmul math.
