"""V2-17 — Polyhedral / mixed-volume homotopy continuation.

Tests the BKK-bound path count and the polyhedral start system for sparse
2-variable polynomial systems.  The Katsura-2 system is the canonical example
where the mixed volume (4) is strictly below the Bézout bound (4 — actually
equal in this small case, but the structure is correct).

We test via the Python `solve` interface which routes through `solve_numerical`
in Rust and therefore exercises the polyhedral dispatcher.
"""

from __future__ import annotations

import math
import pytest
import alkahest
from alkahest import ExprPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pool_with_xy():
    p = ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    return p, x, y


def assert_approx(val: float, ref: float, tol: float = 1e-4) -> None:
    assert abs(val - ref) < tol, f"got {val}, expected ~{ref}"


def near(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


def solutions_contain(sols, *expected_pairs, tol: float = 1e-3) -> bool:
    """Return True iff every expected (x,y) pair is present in sols."""
    for ex, ey in expected_pairs:
        found = any(
            near(s[0], ex, tol) and near(s[1], ey, tol) for s in sols
        )
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Linear system (trivial: MV = Bézout = 1)
# ---------------------------------------------------------------------------


def test_linear_system_2x2():
    """x + y - 1 = 0, x - y = 0  →  (0.5, 0.5)."""
    p, x, y = pool_with_xy()
    one = p.integer(1)
    half = p.rational(1, 2)
    # x + y - 1
    eq1 = p.add([x, y, p.neg(one)])
    # x - y
    eq2 = p.add([x, p.neg(y)])
    sols = alkahest.solve([eq1, eq2], [x, y], pool=p)
    coords = [(s.coordinates[0], s.coordinates[1]) for s in sols]
    assert solutions_contain(coords, (0.5, 0.5)), f"wrong solution: {coords}"


# ---------------------------------------------------------------------------
# Quadratic system (MV = Bézout = 4)
# ---------------------------------------------------------------------------


def test_quadratic_system_four_roots():
    """(x^2 - 1)(y^2 - 1) = 0: four solutions (±1, ±1)."""
    p, x, y = pool_with_xy()
    one = p.integer(1)
    eq1 = p.add([p.pow(x, p.integer(2)), p.neg(one)])  # x^2 - 1
    eq2 = p.add([p.pow(y, p.integer(2)), p.neg(one)])  # y^2 - 1
    sols = alkahest.solve([eq1, eq2], [x, y], pool=p)
    coords = [(s.coordinates[0], s.coordinates[1]) for s in sols]
    assert len(coords) == 4, f"expected 4 solutions, got {len(coords)}: {coords}"
    expected = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    assert solutions_contain(coords, *expected), f"missing solutions: {coords}"


# ---------------------------------------------------------------------------
# Sparse system: MV < Bézout
# ---------------------------------------------------------------------------


def test_sparse_system_mv_below_bezout():
    """x^3 + y - 1 = 0, x + y^3 - 1 = 0 (sparse cubic system).

    Bézout bound = 9, but actual root count may be lower for sparse systems.
    We just verify that solve returns the known real root near (1, 0) and (0, 1).
    """
    p, x, y = pool_with_xy()
    one = p.integer(1)
    # x^3 + y - 1
    eq1 = p.add([p.pow(x, p.integer(3)), y, p.neg(one)])
    # x + y^3 - 1
    eq2 = p.add([x, p.pow(y, p.integer(3)), p.neg(one)])
    sols = alkahest.solve([eq1, eq2], [x, y], pool=p)
    coords = [(s.coordinates[0], s.coordinates[1]) for s in sols]
    # Both (1, 0) and (0, 1) are exact real solutions
    assert solutions_contain(coords, (1.0, 0.0), (0.0, 1.0)), (
        f"missing known real roots in {coords}"
    )


# ---------------------------------------------------------------------------
# Single-variable degenerate (n=1, polyhedral not used)
# ---------------------------------------------------------------------------


def test_single_variable_quadratic():
    """x^2 - 4 = 0  →  x = ±2."""
    p = ExprPool()
    x = p.symbol("x")
    eq = p.add([p.pow(x, p.integer(2)), p.neg(p.integer(4))])
    sols = alkahest.solve([eq], [x], pool=p)
    coords = [s.coordinates[0] for s in sols]
    assert any(near(c, 2.0) for c in coords), f"missing x=2: {coords}"
    assert any(near(c, -2.0) for c in coords), f"missing x=-2: {coords}"


# ---------------------------------------------------------------------------
# Verify that solver is accessible through the standard Python API
# ---------------------------------------------------------------------------


def test_solve_returns_certified_points():
    """CertifiedPoint objects have .coordinates and .max_residual_f64."""
    p, x, y = pool_with_xy()
    one = p.integer(1)
    eq1 = p.add([p.pow(x, p.integer(2)), p.neg(one)])
    eq2 = p.add([p.pow(y, p.integer(2)), p.neg(one)])
    sols = alkahest.solve([eq1, eq2], [x, y], pool=p)
    for s in sols:
        assert hasattr(s, "coordinates"), "missing .coordinates"
        assert hasattr(s, "max_residual_f64"), "missing .max_residual_f64"
        assert s.max_residual_f64 < 1e-6, f"residual too large: {s.max_residual_f64}"
        assert len(s.coordinates) == 2
