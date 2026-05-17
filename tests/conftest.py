"""Pytest and Hypothesis configuration for the Python test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Helpers such as `_step_logs` live next to this file; ensure imports resolve.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from hypothesis import settings

# ---------------------------------------------------------------------------
# Hypothesis profiles
#
# - dev: fast local runs (higher default than legacy per-test 100 when unset).
# - ci:  respects HYPOTHESIS_MAX_EXAMPLES (GitHub nightly sets 5000); no deadline
#        so Rust-backed examples are not flaky under load.
#
# Override explicitly: HYPOTHESIS_PROFILE=dev pytest ...
# Reproduce a failure: use the printed @reproduce_failure blob or
#   pytest --hypothesis-seed=<seed> path/to/test.py
# ---------------------------------------------------------------------------

_CI_DEFAULT_EXAMPLES = 5000
_DEV_DEFAULT_EXAMPLES = 200


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


settings.register_profile(
    "dev",
    max_examples=_int_env("HYPOTHESIS_MAX_EXAMPLES", _DEV_DEFAULT_EXAMPLES),
    deadline=None,
)
settings.register_profile(
    "ci",
    max_examples=_int_env("HYPOTHESIS_MAX_EXAMPLES", _CI_DEFAULT_EXAMPLES),
    deadline=None,
)

_profile = os.environ.get("HYPOTHESIS_PROFILE")
if _profile is None:
    _ci_markers = ("true", "1", "yes")
    _profile = (
        "ci" if os.environ.get("CI", "").lower() in _ci_markers else "dev"
    )

settings.load_profile(_profile)
