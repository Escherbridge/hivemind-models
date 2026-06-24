# `workspace_1/` — synthetic fixture workspace

A minimal, dependency-free workspace used by the substrate tests
(`dataset_sovereignty`, `dataset_generator_contract`, and the mesh e2e). It
spans all three sovereignty tiers (`private` / `shared` / `community`) so tier
enforcement can be exercised on real data. See
`conductor/tracks/_shared/substrate-adr.md` D5 for the tier-dispatch rule and
`conductor/tracks/_handoff/desktop-pivot.md` §4.4 for the original layout spec.

## Layout

```
workspace_1/
  .mpg/mind-palace.json     # 3 stashes, one per tier, with category tags
  .sift/index.json          # 2 index entries (private, shared); JSON not .db
                            #   to stay dependency-free + inspectable
  sample_files/note1.md     # source doc the private stash/sift entry point at
  sample_files/note2.md     # source doc the shared stash/sift entry point at
  training_records/
    record_001.json         # tier: private   (engine: mpg)
    record_002.json         # tier: shared    (engine: combined)
    record_003.json         # tier: community (engine: sift)
```

## Tier coverage

| Artifact | private | shared | community |
|----------|:-------:|:------:|:---------:|
| mpg stash       | stash-001 | stash-002 | stash-003 |
| sift entry      | sift-001  | sift-002  | —         |
| TrainingRecord  | record_001 | record_002 | record_003 |

`tier` is a first-class string field on every artifact, matching
`scripts.sovereignty.Tier`. `record_001` is `private`, so per D5 a shard built
from it must never be dispatched across peers.

Loaded by the `fixture_workspace_dir` fixture in `conftest.py`; validated by
`test_fixture_workspace.py`.
