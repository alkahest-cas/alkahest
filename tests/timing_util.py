"""Shared helpers for the few tests that assert on wall-clock time.

A wall-clock assertion in a test checks an *order of magnitude* ("linear, not
exponential"; "a budget trip returns, not hangs"), never a deadline. Keep the
bound generous, scale it with :func:`scale` on slow runners, and use
:func:`best_of` so one descheduled run cannot fail a healthy build. Where the
property can be asserted without a clock (a result, an error code, a step
count) assert that instead.
"""

from __future__ import annotations

import os
import time


def scale() -> float:
    """Multiplier for time bounds: ``ALKAHEST_TEST_TIME_SCALE`` (default 1)."""
    try:
        value = float(os.environ.get("ALKAHEST_TEST_TIME_SCALE", "1"))
    except ValueError:
        return 1.0
    return value if value > 0 else 1.0


def limit(seconds: float) -> float:
    """``seconds`` stretched by :func:`scale`."""
    return seconds * scale()


def best_of(fn, runs: int = 3):
    """Run ``fn`` up to ``runs`` times; return ``(result, fastest_seconds)``.

    Stops early nothing: every run executes, so ``fn`` must be repeatable.
    """
    best = float("inf")
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return result, best
