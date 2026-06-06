"""Pytest conftest for the expert-coordinator track's test suite.

This conftest only ships fixtures used by the ``__tests__`` package; it does
not collide with the project-wide ``tests/conftest.py`` (which is Agent A's
file). To make sure pytest can discover the scripts package, we prepend the
repo root to ``sys.path`` so that ``import scripts._wire`` works regardless
of cwd.

Per the ownership map, all coordinator-owned fixtures are namespaced
``coord_*`` so they cannot collide with Agent A's ``shard_*`` namespace or pi's
``ref_*`` namespace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Add the hivemind-models repo root to sys.path so `import scripts._wire` works
# when pytest is invoked from any cwd inside the repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def coord_free_port() -> int:
    """Return a probably-free TCP port for an in-process aiohttp test server."""

    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
