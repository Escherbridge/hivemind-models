# Wire format note

The coordinator <-> expert data plane uses a 12-byte little-endian header
(`<BHIIB`: op, expert_id, n_tokens, hidden, dtype) followed by a contiguous
payload of `n_tokens * hidden * sizeof(dtype)` bytes. See `scripts/_wire.py`.
