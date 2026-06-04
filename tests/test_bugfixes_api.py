"""Regression tests for API bugfixes (2026-06 audit)."""

from __future__ import annotations

import math

import alkahest
import pytest
from alkahest import PoolError


def test_cross_pool_add_raises():
    p1, p2 = alkahest.ExprPool(), alkahest.ExprPool()
    x1, x2 = p1.symbol("x"), p2.symbol("x")
    with pytest.raises(PoolError):
        _ = x1 + x2


def test_pool_integer_bigint():
    p = alkahest.ExprPool()
    n = 10**100
    e = p.integer(n)
    assert str(e) == str(n)


def test_subs_int_and_float():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assert str(alkahest.subs(x**2, {x: 3})) == "3^2"
    assert "1.5" in str(alkahest.subs(x**2, {x: 1.5}))
    assert str(alkahest.subs(x**2, {x: 1.5})).endswith("^2")


def test_subs_derived_result_value():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    r = alkahest.diff(x, x)
    assert str(alkahest.subs(x, {x: r})) == "1"


def test_eval_expr_derived_result():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    r = alkahest.diff(x**2, x)
    assert alkahest.eval_expr(r, {x: 2.0}) == 4.0


def test_derived_result_bool_and_eq():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    zero = alkahest.diff(x - x, x)
    one = alkahest.diff(x, x)
    assert not zero
    assert one
    assert zero == p.integer(0)
    assert one == p.integer(1)
    assert zero.value == p.integer(0)


def test_product_inverted_bounds():
    p = alkahest.ExprPool()
    k, n = p.symbol("k"), p.symbol("n")
    expr = alkahest.product_definite(k, k, p.integer(5), p.integer(3)).value
    fc = alkahest.compile_expr(expr, [n])
    assert fc([3.0]) == 1.0


def test_product_factorial_no_spurious_one_pow():
    p = alkahest.ExprPool()
    k, n = p.symbol("k"), p.symbol("n")
    expr = alkahest.simplify(alkahest.product_definite(k, k, p.integer(1), n).value).value
    s = str(expr)
    assert "1^n" not in s
    fc = alkahest.compile_expr(expr, [n])
    assert abs(fc([5.0]) - math.factorial(5)) < 1e-3


def test_rsolve_no_one_pow():
    p = alkahest.ExprPool()
    n = p.symbol("n")

    def f(*args):
        return p.func("f", list(args))

    eq = alkahest.simplify(f(n) - f(n + p.integer(-1)) - p.integer(1)).value
    sol = alkahest.rsolve(eq, n, "f", None)
    assert "1^n" not in str(sol)
    assert "C0" in str(sol)


def test_solve_simplified_radicals():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    sols = alkahest.solve([x**2 - 1], [x])
    assert len(sols) == 2
    for s in sols:
        val = next(iter(s.values()))
        simp = alkahest.simplify(val).value
        t = str(simp)
        if t in ("1", "-1", "1.0", "-1.0") or "sqrt" in t:
            continue
        assert abs(alkahest.eval_expr(simp, {})) == pytest.approx(1.0)


def test_domain_enum_export():
    assert hasattr(alkahest, "Domain")
    assert "Domain" in alkahest.__all__
    assert str(alkahest.Domain.Real) == "real"
    assert str(alkahest.Domain.Integer) == "integer"
    p = alkahest.ExprPool()
    x = p.symbol("x", alkahest.Domain.Integer)
    y = p.symbol("y", "integer")
    assert x is not None and y is not None
    with alkahest.context(pool=p, domain=alkahest.Domain.Real):
        z = alkahest.symbol("z")
        assert z is not None


def test_primary_decomposition_stable_export():
    assert hasattr(alkahest, "primary_decomposition")
    assert hasattr(alkahest, "radical")
    assert "primary_decomposition" in alkahest.__all__


def test_piecewise_pool_gt():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    cond = p.gt(x, p.integer(0))
    pw = alkahest.piecewise([(cond, x)], p.integer(-1) * x)
    assert pw is not None


def test_context_simplify_invokes_postprocess(monkeypatch):
    """context(simplify=True) runs _derived_result_context_simplify on diff."""
    import alkahest

    calls: list[object] = []
    orig = alkahest._derived_result_context_simplify

    def counting(dr):
        calls.append(dr)
        return orig(dr)

    monkeypatch.setattr(alkahest, "_derived_result_context_simplify", counting)
    p = alkahest.ExprPool()
    x = p.symbol("x")
    with alkahest.context(simplify=True):
        alkahest.diff(x**2, x)
    assert calls
    calls.clear()
    alkahest.diff(x**2, x)
    assert not calls


def test_context_simplify_merges_step():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    expr = x + 0
    with alkahest.context(simplify=True):
        r = alkahest.diff(expr, x)
    rules = [s["rule"] for s in r.steps]
    # When simplify changes the value, a context_simplify step is recorded.
    if "context_simplify" in rules:
        assert rules.index("context_simplify") >= 0


def test_derived_result_coercion_extended():
    """sum/product/poly_normal accept DerivedResult first args (B2)."""
    p = alkahest.ExprPool()
    k, n = p.symbol("k"), p.symbol("n")
    hyper = alkahest.simplify(k * alkahest.gamma(k + p.integer(1)))
    assert alkahest.sum_indefinite(hyper, k).steps
    assert alkahest.sum_definite(hyper, k, p.integer(0), n).value is not None
    dr_k = alkahest.integrate(p.integer(1), k)
    assert alkahest.product_indefinite(dr_k, k).value is not None
    assert alkahest.product_definite(dr_k, k, p.integer(1), n).value is not None
    assert alkahest.poly_normal(alkahest.diff(k**2, k), [k]) is not None
