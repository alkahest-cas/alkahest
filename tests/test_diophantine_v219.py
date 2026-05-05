"""V2-19 — Diophantine equations (`diophantine`)."""

import pytest

import alkahest

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "diophantine"),
    reason="native module built without groebner feature",
)


def test_diophantine_linear_parametric():
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    eq = p.integer(3) * x + p.integer(5) * y - p.integer(1)
    sol = alkahest.diophantine(eq, [x, y])
    assert sol.kind == "parametric_linear"
    assert sol.parameter is not None
    assert sol.parametric is not None and len(sol.parametric) == 2


def test_diophantine_pell_unit():
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    eq = x**2 - p.integer(2) * y**2 - p.integer(1)
    sol = alkahest.diophantine(eq, [x, y])
    assert sol.kind == "pell_fundamental"
    fx, fy = sol.fundamental
    assert int(str(fx)) == 3
    assert int(str(fy)) == 2
    assert int(str(sol.pell_d)) == 2


def test_diophantine_sum_two_squares():
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    eq = x**2 + y**2 - p.integer(5)
    sol = alkahest.diophantine(eq, [x, y])
    assert sol.kind == "finite"
    pairs = {(int(str(a)), int(str(b))) for a, b in sol.points}
    assert pairs == {(1, 2), (2, 1)}


def test_diophantine_sum_two_squares_65_two_orbits():
    """65 = 1² + 8² = 4² + 7² (distinct representations)."""
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sol = alkahest.diophantine(x**2 + y**2 - p.integer(65), [x, y])
    assert sol.kind == "finite"
    pairs = {(int(str(a)), int(str(b))) for a, b in sol.points}
    assert {(1, 8), (8, 1), (4, 7), (7, 4)}.issubset(pairs)


def test_diophantine_pell_generalized_x2_minus_2y2_eq_minus1():
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    eq = x**2 - p.integer(2) * y**2 + p.integer(1)
    sol = alkahest.diophantine(eq, [x, y])
    assert sol.kind in ("pell_generalized", "pell_fundamental")
    if sol.kind == "pell_generalized":
        assert sol.pell_n is not None and int(str(sol.pell_n)) == -1
        assert sol.pell_particular is not None
        assert sol.pell_unit is not None
        x0, y0 = sol.pell_particular
        assert int(str(x0)) ** 2 - 2 * int(str(y0)) ** 2 == -1


def test_diophantine_no_solution_linear():
    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    eq = p.integer(2) * x + p.integer(4) * y + p.integer(1)
    sol = alkahest.diophantine(eq, [x, y])
    assert sol.kind == "no_solution"


def test_diophantine_exactly_two_vars():
    p = alkahest.ExprPool()
    x, y, z = p.symbol("x"), p.symbol("y"), p.symbol("z")
    eq = x + y - p.integer(1)
    with pytest.raises(Exception) as exc:
        alkahest.diophantine(eq, [x, y, z])
    assert "two" in str(exc.value).lower()


def test_diophantine_sympy_finite_subset():
    sympy = pytest.importorskip("sympy")

    sx, sy = sympy.symbols("x y")
    py_sol = sympy.diophantine(sx**2 + sy**2 - 5)
    sp_abs = {tuple(sorted((abs(int(t[0])), abs(int(t[1]))))) for t in py_sol}

    p = alkahest.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    sol = alkahest.diophantine(x**2 + y**2 - p.integer(5), [x, y])
    ap = {tuple(sorted((int(str(a)), int(str(b))))) for a, b in sol.points}
    assert ap == {(1, 2)}
    assert ap <= sp_abs

