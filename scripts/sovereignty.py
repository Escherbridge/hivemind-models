"""Dataset / shard sovereignty tiers and the dispatch gate.

Single source of truth for the three sovereignty tiers defined in
``conductor/tracks/dataset_sovereignty_20260613/spec.md`` and the dispatch
behaviour mandated by ``conductor/tracks/_shared/substrate-adr.md`` D5:

    A shard built from ``private``-tier records inherits ``private`` and is
    NEVER dispatched to other peers. Inference over private shards is
    local-only. ``shared`` / ``community`` shards may be sharded across peers.

This module is intentionally dependency-free (stdlib only) so both the
coordinator (signaling-only, must stay light) and any peer/desktop engine can
import it without pulling torch/aiohttp.

Tier ordering, from most to least restrictive::

    private  >  shared  >  community

``most_restrictive`` implements the sovereignty spec's rule that a mixed-tier
adapter / shard inherits the most-restrictive tier of its inputs
(``tier_lineage``). ``is_dispatchable`` is the single gate dispatch code calls
before routing any expert call across peers.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class Tier(str, Enum):
    """Sovereignty tier. ``str`` mixin so JSON round-trips as the bare string."""

    PRIVATE = "private"
    SHARED = "shared"
    COMMUNITY = "community"


# Restrictiveness rank: higher = more restrictive. private is the most
# restrictive (and the default for every new artifact per the spec).
_RANK = {Tier.PRIVATE: 2, Tier.SHARED: 1, Tier.COMMUNITY: 0}

# The default tier on every new artifact (sovereignty spec requirement).
DEFAULT_TIER = Tier.PRIVATE


def coerce_tier(value: "Tier | str") -> Tier:
    """Normalize a tier given as a ``Tier`` or a raw string.

    Raises ``ValueError`` on an unknown string so bad data fails loud rather
    than silently downgrading a privacy guarantee.
    """

    if isinstance(value, Tier):
        return value
    try:
        return Tier(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown tier {value!r}; expected one of "
            f"{[t.value for t in Tier]}"
        ) from exc


def most_restrictive(tiers: Iterable["Tier | str"]) -> Tier:
    """Return the most-restrictive tier among ``tiers``.

    Implements the sovereignty spec's mixed-tier inheritance rule: a shard /
    adapter built from inputs of several tiers inherits the most restrictive
    one. An empty input defaults to ``private`` (fail closed).
    """

    coerced = [coerce_tier(t) for t in tiers]
    if not coerced:
        return DEFAULT_TIER
    return max(coerced, key=lambda t: _RANK[t])


def is_dispatchable(tier: "Tier | str") -> bool:
    """Whether a shard of ``tier`` may be dispatched to OTHER peers.

    ADR D5: ``private`` shards are local-only and never cross the machine
    boundary. ``shared`` and ``community`` may be sharded across peers.
    """

    return coerce_tier(tier) is not Tier.PRIVATE


class TierViolation(Exception):
    """Raised when dispatch is attempted for a non-dispatchable (private) shard."""

    def __init__(self, tier: Tier) -> None:
        self.tier = tier
        super().__init__(
            f"shard tier {tier.value!r} is local-only and must not be "
            f"dispatched across peers (substrate-adr.md D5)"
        )


def assert_dispatchable(tier: "Tier | str") -> Tier:
    """Return the coerced tier if dispatchable, else raise ``TierViolation``.

    The single enforcement point dispatch code calls before routing an expert
    call to a remote peer.
    """

    coerced = coerce_tier(tier)
    if not is_dispatchable(coerced):
        raise TierViolation(coerced)
    return coerced


__all__ = [
    "Tier",
    "DEFAULT_TIER",
    "TierViolation",
    "coerce_tier",
    "most_restrictive",
    "is_dispatchable",
    "assert_dispatchable",
]
