"""Tests for V2-4: Real root isolation (Vincent–Akritas–Strzeboński).

Test plan (from ROADMAP.md acceptance criteria):
- Correctness: all roots of (x-1)(x-2)…(x-5) are isolated.
- Disjointness: returned intervals are pairwise non-overlapping.
- Enclosure: each known root is contained in its interval.
- Negative roots: x²-1, x³+x²-x-1 (roots at -1, -1, 1).
- Chebyshev T₄: 4 roots in (-1, 1).
- Zero polynomial: raises RealRootError.
- Constant non-zero: returns empty list.
- refine_root: ball contains the root.
- Symbolic entry point: real_roots(x²-4, x).
- RootInterval attributes: lo, hi, lo_exact, hi_exact.
"""

import math

import pytest
from alkahest import ExprPool, RealRootError, real_roots, refine_root

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pool():
    return ExprPool()


def roots_of(pool, expr, var):
    """Return sorted list of RootIntervals for ``expr`` in ``var``."""
    ivs = real_roots(expr, var)
    return sorted(ivs, key=lambda iv: iv.lo)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_linear_positive_root():
    p = make_pool()
    x = p.symbol("x")
    expr = x + p.integer(-3)
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 1
    assert ivs[0].lo <= 3.0 <= ivs[0].hi


def test_x_squared_minus_1():
    p = make_pool()
    x = p.symbol("x")
    expr = x**2 + p.integer(-1)
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 2
    # Roots at -1 and 1.
    assert ivs[0].lo <= -1.0 <= ivs[0].hi
    assert ivs[1].lo <= 1.0 <= ivs[1].hi


def test_x_squared_minus_4():
    p = make_pool()
    x = p.symbol("x")
    expr = x**2 + p.integer(-4)
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 2
    assert ivs[0].lo <= -2.0 <= ivs[0].hi
    assert ivs[1].lo <= 2.0 <= ivs[1].hi


def test_five_integer_roots():
    """(x-1)(x-2)(x-3)(x-4)(x-5) — 5 distinct roots."""
    p = make_pool()
    x = p.symbol("x")
    # x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120
    expr = (
        x**5
        + p.integer(-15) * x**4
        + p.integer(85) * x**3
        + p.integer(-225) * x**2
        + p.integer(274) * x
        + p.integer(-120)
    )
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 5, f"Expected 5 roots, got {len(ivs)}: {ivs}"
    for k in range(1, 6):
        assert any(iv.lo <= k <= iv.hi for iv in ivs), f"Root {k} not enclosed"


# ---------------------------------------------------------------------------
# Disjointness invariant
# ---------------------------------------------------------------------------


def test_disjoint_intervals_five_roots():
    p = make_pool()
    x = p.symbol("x")
    expr = (
        x**5
        + p.integer(-15) * x**4
        + p.integer(85) * x**3
        + p.integer(-225) * x**2
        + p.integer(274) * x
        + p.integer(-120)
    )
    ivs = roots_of(p, expr, x)
    for i in range(len(ivs) - 1):
        assert ivs[i].hi <= ivs[i + 1].lo, (
            f"Intervals overlap: [{ivs[i].lo},{ivs[i].hi}] and [{ivs[i + 1].lo},{ivs[i + 1].hi}]"
        )


def test_disjoint_three_roots():
    p = make_pool()
    x = p.symbol("x")
    # (x-1)(x-2)(x-3) = x³-6x²+11x-6
    expr = x**3 + p.integer(-6) * x**2 + p.integer(11) * x + p.integer(-6)
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 3
    for i in range(2):
        assert ivs[i].hi <= ivs[i + 1].lo


# ---------------------------------------------------------------------------
# Squarefree stripping
# ---------------------------------------------------------------------------


def test_repeated_root_counted_once():
    """(x-1)² has one distinct root at x=1."""
    p = make_pool()
    x = p.symbol("x")
    # (x-1)^2 = x²-2x+1
    expr = x**2 + p.integer(-2) * x + p.integer(1)
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 1
    assert ivs[0].lo <= 1.0 <= ivs[0].hi


# ---------------------------------------------------------------------------
# Chebyshev polynomials
# ---------------------------------------------------------------------------


def test_chebyshev_t4_four_roots():
    """T₄(x) = 8x⁴-8x²+1 — 4 real roots in (-1, 1)."""
    p = make_pool()
    x = p.symbol("x")
    expr = p.integer(8) * x**4 + p.integer(-8) * x**2 + p.integer(1)
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 4, f"T₄ should have 4 roots, got {len(ivs)}"
    for iv in ivs:
        assert iv.lo >= -1.0 - 1e-12
        assert iv.hi <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# No real roots
# ---------------------------------------------------------------------------


def test_no_real_roots():
    """x²+1 has no real roots."""
    p = make_pool()
    x = p.symbol("x")
    expr = x**2 + p.integer(1)
    ivs = real_roots(expr, x)
    assert ivs == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_zero_polynomial_raises():
    p = make_pool()
    x = p.symbol("x")
    with pytest.raises(RealRootError):
        real_roots(p.integer(0), x)


def test_constant_non_zero_returns_empty():
    p = make_pool()
    x = p.symbol("x")
    ivs = real_roots(p.integer(5), x)
    assert ivs == []


# ---------------------------------------------------------------------------
# refine_root
# ---------------------------------------------------------------------------


def test_refine_root_sqrt2():
    """Refine x²-2 root near √2."""
    p = make_pool()
    x = p.symbol("x")
    expr = x**2 + p.integer(-2)
    ivs = roots_of(p, expr, x)
    assert len(ivs) == 2
    pos = next(iv for iv in ivs if iv.lo >= 0)
    ball = refine_root(expr, pos, x)
    sqrt2 = math.sqrt(2)
    eps = 1e-10  # tolerance for f64 boundary effects
    assert ball.mid - ball.rad - eps <= sqrt2 <= ball.mid + ball.rad + eps


def test_refine_root_exact():
    """refine_root on exact root x=3 returns a narrow ball."""
    p = make_pool()
    x = p.symbol("x")
    expr = x + p.integer(-3)
    ivs = real_roots(expr, x)
    assert len(ivs) == 1
    ball = refine_root(expr, ivs[0], x)
    assert abs(ball.mid - 3.0) <= ball.rad + 1e-12


# ---------------------------------------------------------------------------
# RootInterval attributes
# ---------------------------------------------------------------------------


def test_root_interval_exact_attributes():
    p = make_pool()
    x = p.symbol("x")
    # x - 1 has root at x=1 (exact rational).
    expr = x + p.integer(-1)
    ivs = real_roots(expr, x)
    assert len(ivs) == 1
    iv = ivs[0]
    assert iv.lo == iv.hi == 1.0
    lo_n, lo_d = iv.lo_exact()
    hi_n, hi_d = iv.hi_exact()
    assert lo_n == hi_n
    assert lo_d == hi_d


def test_root_interval_repr():
    p = make_pool()
    x = p.symbol("x")
    expr = x + p.integer(-2)
    ivs = real_roots(expr, x)
    r = repr(ivs[0])
    assert "RootInterval" in r
    assert "2" in r
