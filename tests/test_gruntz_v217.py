"""V2-17 — Gruntz algorithm limits (exp-log combinations, comparability graph).

These tests cover limits that are beyond the reach of L'Hôpital or series
expansion at 0 because the expressions involve essential singularities (e.g.
exp(-1/t) after an infinity substitution).  All cases are cross-checked with
SymPy as the oracle where available.
"""

from __future__ import annotations

import pytest
import alkahest
from alkahest import ExprPool, limit, pos_infinity


sympy = pytest.importorskip("sympy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pool_with_x():
    p = ExprPool()
    x = p.symbol("x")
    return p, x


def oo(p: ExprPool):
    return p.pos_infinity()


def neg_oo(p: ExprPool):
    return -p.pos_infinity()


def sp_limit(expr_str: str, *, direction: str | None = None):
    """SymPy limit of expr_str(x) as x → +∞."""
    x = sympy.Symbol("x")
    e = sympy.sympify(expr_str)
    if direction == "+":
        return sympy.limit(e, x, sympy.oo, dir="+")
    return sympy.limit(e, x, sympy.oo)


def alk_limit(expr, var, pt, direction="+-"):
    return limit(expr, var, pt, direction)


def equiv(a, b) -> bool:
    if a == b:
        return True
    try:
        diff = sympy.simplify(a - b)
        return diff == 0
    except Exception:
        return False


def to_sympy(e):
    INF = "∞"
    n = e.node()
    tag = n[0]
    if tag == "symbol":
        return sympy.oo if n[1] == INF else sympy.Symbol(n[1])
    if tag == "integer":
        return sympy.Integer(int(n[1]))
    if tag == "rational":
        return sympy.Rational(int(n[1]), int(n[2]))
    if tag == "add":
        return sympy.Add(*[to_sympy(c) for c in n[1]])
    if tag == "mul":
        return sympy.Mul(*[to_sympy(c) for c in n[1]])
    if tag == "pow":
        return sympy.Pow(to_sympy(n[1]), to_sympy(n[2]))
    if tag == "func":
        name, args = n[1], n[2]
        av = [to_sympy(a) for a in args]
        return getattr(sympy, name)(*av)
    raise AssertionError(f"unknown tag: {tag}")


# ---------------------------------------------------------------------------
# Basic exp-dominance cases
# ---------------------------------------------------------------------------


def test_exp_neg_x_to_zero():
    """lim_{x→+∞} exp(-x) = 0."""
    p, x = pool_with_x()
    expr = alkahest.exp(-x)
    r = alk_limit(expr, x, oo(p))
    assert r == p.integer(0), f"got {r}"


def test_x_exp_neg_x_to_zero():
    """lim_{x→+∞} x * exp(-x) = 0."""
    p, x = pool_with_x()
    expr = x * alkahest.exp(-x)
    r = alk_limit(expr, x, oo(p))
    assert r == p.integer(0), f"got {r}"


def test_x_sq_exp_neg_x_to_zero():
    """lim_{x→+∞} x² * exp(-x) = 0."""
    p, x = pool_with_x()
    expr = x**2 * alkahest.exp(-x)
    r = alk_limit(expr, x, oo(p))
    assert r == p.integer(0), f"got {r}"


def test_exp_x_over_x_sq_is_inf():
    """lim_{x→+∞} exp(x) / x² = +∞."""
    p, x = pool_with_x()
    expr = alkahest.exp(x) / x**2
    r = alk_limit(expr, x, oo(p))
    assert r == oo(p), f"got {r}"


def test_exp_x_over_x_pow_100_is_inf():
    """lim_{x→+∞} exp(x) / x^100 = +∞ (exp grows faster than any power)."""
    p, x = pool_with_x()
    expr = alkahest.exp(x) / x**100
    r = alk_limit(expr, x, oo(p))
    assert r == oo(p), f"got {r}"


def test_exp_neg_x_over_x_neg_100_is_zero():
    """lim_{x→+∞} x^100 * exp(-x) = 0."""
    p, x = pool_with_x()
    expr = x**100 * alkahest.exp(-x)
    r = alk_limit(expr, x, oo(p))
    assert r == p.integer(0), f"got {r}"


# ---------------------------------------------------------------------------
# Ratio of exponentials
# ---------------------------------------------------------------------------


def test_ratio_exp2x_over_exp3x():
    """lim_{x→+∞} exp(2x) / exp(3x) = 0."""
    p, x = pool_with_x()
    expr = alkahest.exp(p.integer(2) * x) / alkahest.exp(p.integer(3) * x)
    r = alk_limit(expr, x, oo(p))
    assert r == p.integer(0), f"got {r}"


def test_ratio_exp3x_over_exp2x():
    """lim_{x→+∞} exp(3x) / exp(2x) = +∞."""
    p, x = pool_with_x()
    expr = alkahest.exp(p.integer(3) * x) / alkahest.exp(p.integer(2) * x)
    r = alk_limit(expr, x, oo(p))
    assert r == oo(p), f"got {r}"


def test_exp_x_plus_one_over_exp_x():
    """lim_{x→+∞} exp(x+1) / exp(x) = exp(1)."""
    p, x = pool_with_x()
    expr = alkahest.exp(x + p.integer(1)) / alkahest.exp(x)
    r = alk_limit(expr, x, oo(p))
    expected = alkahest.exp(p.integer(1))
    # Compare symbolically
    r_sp = to_sympy(r)
    ex_sp = sympy.E
    assert equiv(r_sp, ex_sp), f"got {r} (sympy: {r_sp}), expected e"


def test_exp_x_minus_one_over_exp_x():
    """lim_{x→+∞} exp(x-1) / exp(x) = exp(-1) = 1/e."""
    p, x = pool_with_x()
    expr = alkahest.exp(x - p.integer(1)) / alkahest.exp(x)
    r = alk_limit(expr, x, oo(p))
    r_sp = to_sympy(r)
    assert equiv(r_sp, sympy.exp(-1)), f"got {r_sp}"


# ---------------------------------------------------------------------------
# Nested exp
# ---------------------------------------------------------------------------


def test_nested_exp_exp_x_is_inf():
    """lim_{x→+∞} exp(exp(x)) = +∞."""
    p, x = pool_with_x()
    expr = alkahest.exp(alkahest.exp(x))
    r = alk_limit(expr, x, oo(p))
    assert r == oo(p), f"got {r}"


def test_nested_exp_ratio_simplifies():
    """lim_{x→+∞} exp(exp(x) + x) / exp(exp(x)) = +∞ (since exp(x)→∞)."""
    p, x = pool_with_x()
    # exp(exp(x) + x) / exp(exp(x)) = exp(x) → ∞
    ee = alkahest.exp(x)
    expr = alkahest.exp(ee + x) / alkahest.exp(ee)
    r = alk_limit(expr, x, oo(p))
    assert r == oo(p), f"got {r}"


# ---------------------------------------------------------------------------
# SymPy oracle cross-checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr_str", "sp_ref"),
    [
        ("exp(-x)", 0),
        ("x*exp(-x)", 0),
        ("x**2*exp(-x)", 0),
        ("exp(x)/x**2", sympy.oo),
        ("exp(2*x)/exp(3*x)", 0),
        ("exp(3*x)/exp(2*x)", sympy.oo),
    ],
)
def test_gruntz_sympy_oracle(expr_str, sp_ref):
    """Cross-check Gruntz results against SymPy for simple exp expressions."""
    x_sp = sympy.Symbol("x")
    e_sp = sympy.sympify(expr_str)
    sp_actual = sympy.limit(e_sp, x_sp, sympy.oo)
    if isinstance(sp_ref, int):
        sp_ref = sympy.Integer(sp_ref)
    assert sp_actual == sp_ref, f"SymPy sanity check failed: {expr_str}"

    p, x = pool_with_x()
    e = alkahest.parse(expr_str.replace("**", "^"), p, {"x": x})
    r = alk_limit(e, x, oo(p))
    r_sp = to_sympy(r)
    assert equiv(r_sp, sp_ref), f"{expr_str}: got {r_sp}, expected {sp_ref}"
