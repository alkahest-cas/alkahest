"""Pytest and Hypothesis configuration for the Python test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Helpers such as `_step_logs` live next to this file; ensure imports resolve.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from hypothesis import settings  # noqa: E402 — must follow sys.path bootstrap for `_step_logs`

# ---------------------------------------------------------------------------
# glibc malloc arenas
#
# glibc gives a thread that allocates its own malloc arena, reserves 64 MiB of
# address space for each one, never destroys an arena, and allows up to
# 8 x (number of cores) of them. The deep-recursion tests keep a chain of
# stack-segment threads alive at once (`simplify/stack.rs`), each of which gets
# a fresh arena: on a 32-core machine `tests/test_expression_depth_limit.py`
# alone takes this process from 5.1 GB to 15.7 GB of *virtual* address space
# while its resident set stays under 0.5 GB.
#
# That is invisible until the suite runs under `ulimit -v` (as the repository's
# agent guidance recommends for heavy runs) — which is exactly when Alkahest's
# address-space guard (`budget/memory.rs`) is active. CI sets no `ulimit`, so
# the guard is inert there. Under a 16 GB cap this bloat made the q-Zeilberger
# tests hang, which was a library bug and is fixed in the library: a stale
# address-space sample no longer reads as growth (`budget::GROWTH_WINDOW`), and
# q_zeilberger stops on a budget refusal instead of computing on unreduced
# fractions. With those fixes the suite no longer hangs even with this cap
# disabled — but the process really is near its limit then, and a
# depth-limit test cannot spawn its stack threads (EAGAIN) on a many-core
# machine.
#
# So this cap is a property of the *test process*, not a library workaround:
# it keeps the suite's address-space footprint independent of the core count,
# so that a run under `ulimit -v` tests the library rather than the machine.
# Eight arenas still give the `parallel` feature's worker threads mostly
# uncontended malloc while bounding the arenas at ~512 MiB of address space.
# The MALLOC_ARENA_MAX environment variable is read only when the process
# starts, so the cap is applied with mallopt(M_ARENA_MAX) at conftest import
# time instead, before the tests start any thread. An explicit
# MALLOC_ARENA_MAX in the environment wins (MALLOC_ARENA_MAX=256 reproduces
# the uncapped behaviour on a 32-core machine).
# ---------------------------------------------------------------------------

_TEST_MALLOC_ARENA_MAX = 8


def _cap_glibc_malloc_arenas(limit: int) -> None:
    if not sys.platform.startswith("linux") or "MALLOC_ARENA_MAX" in os.environ:
        return
    import ctypes

    try:
        libc = ctypes.CDLL(None)
        # Present only in glibc; musl has neither arenas nor this symbol.
        libc.gnu_get_libc_version  # noqa: B018
        m_arena_max = -8  # <malloc.h>: #define M_ARENA_MAX -8
        libc.mallopt(m_arena_max, limit)
    except (OSError, AttributeError):
        pass


_cap_glibc_malloc_arenas(_TEST_MALLOC_ARENA_MAX)

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
    _profile = "ci" if os.environ.get("CI", "").lower() in _ci_markers else "dev"

settings.load_profile(_profile)
