"""E-graph post-extraction constant folding (report7-20)."""

from __future__ import annotations

import alkahest as ak
import pytest

pytestmark = pytest.mark.skipif(
    not getattr(ak, "HAS_EGRAPH", False),
    reason="egraph feature not enabled in this build",
)


@pytest.fixture
def pool():
    return ak.ExprPool()


def test_simplify_egraph_double_over_two(pool):
    """``(x + x) / 2`` must fold to ``x`` (was ``((x * 2) * 1/2)``)."""
    x = pool.symbol("x")
    result = ak.simplify_egraph((x + x) / 2)
    assert result.value == x


def test_simplify_egraph_nested_two_x_times_half(pool):
    """Flattened coefficient fold: ``(2·x)·(1/2)`` → ``x``."""
    x = pool.symbol("x")
    two_x = x * 2
    half = pool.integer(1) / 2
    result = ak.simplify_egraph(two_x * half)
    assert result.value == x


def test_simplify_egraph_still_folds_plain_constants(pool):
    result = ak.simplify_egraph(pool.integer(3) + pool.integer(4))
    assert result.value == pool.integer(7)
