"""Validate the synthetic fixture workspace and that its tiers round-trip
through the sovereignty substrate.

This is the round-trip-validation requirement from
``dataset_generator_contract_20260613/spec.md`` §4 reduced to what exists
today: every artifact carries a first-class ``tier`` that parses to
``scripts.sovereignty.Tier``, and the private-tier records are correctly
identified as non-dispatchable per substrate-adr.md D5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sovereignty import Tier, coerce_tier, is_dispatchable

REQUIRED_TR_FIELDS = {
    "schema_version", "workspace_id", "tier", "tool_category",
    "query", "context_nodes", "positive_response", "negative_examples",
    "provenance",
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_workspace_layout_present(fixture_workspace_dir: Path) -> None:
    ws = fixture_workspace_dir
    assert (ws / ".mpg" / "mind-palace.json").exists()
    assert (ws / ".sift" / "index.json").exists()
    assert (ws / "sample_files" / "note1.md").exists()
    assert (ws / "sample_files" / "note2.md").exists()
    records = sorted((ws / "training_records").glob("record_*.json"))
    assert len(records) == 3


def test_mpg_stashes_have_valid_tiers(fixture_workspace_dir: Path) -> None:
    palace = _load(fixture_workspace_dir / ".mpg" / "mind-palace.json")
    tiers = {coerce_tier(s["tier"]) for s in palace["palace"]}
    # all three tiers represented
    assert tiers == {Tier.PRIVATE, Tier.SHARED, Tier.COMMUNITY}


def test_sift_entries_have_valid_tiers(fixture_workspace_dir: Path) -> None:
    index = _load(fixture_workspace_dir / ".sift" / "index.json")
    for entry in index["entries"]:
        coerce_tier(entry["tier"])  # raises if invalid


def test_training_records_well_formed_and_tiered(fixture_workspace_dir: Path) -> None:
    ws = fixture_workspace_dir
    found_tiers = set()
    for p in sorted((ws / "training_records").glob("record_*.json")):
        rec = _load(p)
        assert REQUIRED_TR_FIELDS <= set(rec), f"{p.name} missing fields"
        assert rec["schema_version"] == "1.0"
        found_tiers.add(coerce_tier(rec["tier"]))
    assert found_tiers == {Tier.PRIVATE, Tier.SHARED, Tier.COMMUNITY}


def test_private_record_is_not_dispatchable(fixture_workspace_dir: Path) -> None:
    """D5: the private TrainingRecord's tier must gate dispatch off."""

    rec = _load(fixture_workspace_dir / "training_records" / "record_001.json")
    assert coerce_tier(rec["tier"]) is Tier.PRIVATE
    assert is_dispatchable(rec["tier"]) is False
