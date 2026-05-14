"""Depth / workload profiles for cross-CAS and Python benchmarks.

Profiles control (1) how many times each timed block runs and (2) which
``size`` values from each :class:`BenchTask` are exercised.  Use ``stress`` to
append optional larger sizes declared on tasks as ``stress_size_params``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Public depth names (CLI choices)
DEPTH_CHOICES = ("smoke", "quick", "standard", "deep", "stress")


@dataclass(frozen=True)
class DepthProfile:
    """Timing and size-selection parameters for one run."""

    repeat: int
    number: int
    key: str


_PROFILES: dict[str, DepthProfile] = {
    # One small size, minimal repeats — CI / sanity checks
    "smoke": DepthProfile(repeat=1, number=1, key="smoke"),
    # Smallest + largest default size per task, few repeats
    "quick": DepthProfile(repeat=2, number=1, key="quick"),
    # Default: matches historical cas_comparison behaviour
    "standard": DepthProfile(repeat=3, number=1, key="standard"),
    # More statistical stability, same sizes as standard
    "deep": DepthProfile(repeat=5, number=2, key="deep"),
    # Longer timings + optional stress sizes from tasks
    "stress": DepthProfile(repeat=7, number=3, key="stress"),
}


def get_profile(name: str) -> DepthProfile:
    key = name.strip().lower()
    if key not in _PROFILES:
        allowed = ", ".join(DEPTH_CHOICES)
        raise ValueError(f"unknown depth {name!r}; use one of: {allowed}")
    return _PROFILES[key]


def sizes_for_task(task: Any, profile_key: str) -> list[int]:
    """Return the size list to run for *task* under the given profile key."""
    base = list(task.size_params)
    if not base:
        return [1]

    stress_extra = list(task.stress_size_params)

    if profile_key == "smoke":
        return [base[0]]
    if profile_key == "quick":
        if len(base) == 1:
            return base[:]
        return [base[0], base[-1]]
    if profile_key in ("standard", "deep"):
        return base
    if profile_key == "stress":
        merged = sorted(set(base + stress_extra))
        return merged if merged else base[:]
    return base
