"""Helpers for comparing derivation logs independent of step order.

Parallel simplification merges child logs in a nondeterministic order; tests should
compare multisets of rule names (or sorted projections), not raw list equality.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def rule_counter(steps: list[Mapping[str, Any]]) -> Counter[str]:
    """Multiset of ``rule`` names recorded in ``steps``."""
    return Counter(str(s["rule"]) for s in steps)


def assert_same_step_rules(
    sequential_steps: list[Mapping[str, Any]],
    parallel_steps: list[Mapping[str, Any]],
    *,
    context: str = "",
) -> None:
    """Assert parallel and sequential simplification recorded the same rules."""
    a = rule_counter(sequential_steps)
    b = rule_counter(parallel_steps)
    assert a == b, (
        f"derivation rule multiset mismatch {a!r} vs {b!r}"
        + (f" ({context})" if context else "")
    )
