"""Reference test package (Lane D of Wave 3, browser-as-peer-phase1).

This package owns all tests for the Python forward reference at
``hivemind-models/reference/granite_layer5.py`` and the WGSL numerical
parity fixtures at ``hivemind-models/reference/golden/``.

Fixture namespacing per ``conductor/tracks/_shared/ownership.md``:

- Agent A (shard-byte-server) prefixes fixtures with ``shard_*``.
- Agent B (expert-coordinator) prefixes fixtures with ``coord_*``.
- pi (browser-shader / reference) prefixes fixtures with ``ref_*``.
"""
