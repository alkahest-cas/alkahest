"""Regression tests for the 2026-09-29 core audit, sections A3, A6 and A7.

* A3 — pattern matching compared integer literals through ``to_i64()``, so
  every integer past i64 matched every other one.
* A6 — entry points that take several expressions read foreign-pool ids as
  local ones and returned nonsense (``diff(x*y, a_other)`` gave ``y``).
* A7 — ``subs`` captured variables under binders, rewrote bound variables
  through compound keys, and skipped ``RootSum`` bodies.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from alkahest import ExprPool, PoolError

# ---------------------------------------------------------------------------
# A7 — subs under binders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binder", ["forall", "exists"])
def test_subs_does_not_capture_a_free_variable(binder):
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    f = getattr(p, binder)(y, p.gt(x + y, p.integer(0)))
    out = ak.subs(f, {x: y})
    s = str(out)
    # The free y that replaced x and the bound variable must stay distinct.
    assert "(y + y)" not in s, s
    assert "y_1" in s, s
    # Renaming back gives the same formula as substituting into a binder
    # over a different name to begin with.
    y1 = p.symbol("y_1")
    assert out == getattr(p, binder)(y1, p.gt(y + y1, p.integer(0)))


def test_subs_without_capture_keeps_the_binder_name():
    p = ExprPool()
    x, y, z = p.symbol("x"), p.symbol("y"), p.symbol("z")
    f = p.forall(y, p.gt(x + y, p.integer(0)))
    assert ak.subs(f, {x: z}) == p.forall(y, p.gt(z + y, p.integer(0)))


def test_subs_compound_key_does_not_rewrite_the_bound_variable():
    p = ExprPool()
    y = p.symbol("y")
    e = p.exists(y, p.gt(y**2, p.integer(1)))
    assert ak.subs(e, {y**2: p.integer(0)}) == e


def test_subs_descends_into_root_sum():
    p = ExprPool()
    x = p.symbol("x")
    F = ak.integrate(1 / (x**3 + x + 1), x).value
    assert "RootSum" in str(F)
    G = ak.subs(F, {x: p.integer(2)})
    assert G != F
    want = ak.eval_expr(F, {x: 2.0})
    assert math.isclose(ak.eval_expr(G, {}), want, rel_tol=1e-9, abs_tol=1e-12)


def test_subs_integer_key_matches_a_whole_rational():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    assert ak.subs(x + p.rational(2, 1), {p.integer(2): y}) == x + y


# ---------------------------------------------------------------------------
# A3 — big-integer literals in patterns and rules
# ---------------------------------------------------------------------------


def test_match_pattern_compares_big_integers_exactly():
    p = ExprPool()
    X = p.symbol("X")
    pat = X * p.integer(10**20)
    assert ak.match_pattern(pat, X * p.integer(10**21)) == []
    assert ak.match_pattern(pat, X * p.integer(-(2**63))) == []
    assert ak.match_pattern(pat, X * p.integer(10**20)) == [{}]
    assert ak.match_pattern(p.integer(2**64), p.integer(2**65)) == []
    assert ak.match_pattern(p.integer(2**64), p.integer(2**64)) == [{}]


def test_rule_on_big_integer_does_not_fire_on_another():
    p = ExprPool()
    rule = ak.make_rule(p.func("f", [p.integer(2**64)]), p.integer(0))
    other = p.func("f", [p.integer(2**70)])
    assert ak.simplify_with(other, [rule]).value == other
    assert ak.simplify_with(p.func("f", [p.integer(2**64)]), [rule]).value == p.integer(0)


# ---------------------------------------------------------------------------
# A6 — cross-pool arguments
# ---------------------------------------------------------------------------


def _pools():
    p, q = ExprPool(), ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    # Pad q so its ids overlap p's: a foreign id then names *some* local node.
    a = q.symbol("a")
    for i in range(4):
        q.symbol(f"pad{i}")
    return p, q, x, y, a


def _cases():
    p, q, x, y, a = _pools()
    b = q.symbol("b")
    cases = {
        "pool.add": lambda: q.add([x, b]),
        "pool.mul": lambda: q.mul([x, b]),
        "pool.func": lambda: p.func("f", [a]),
        "pool.forall": lambda: p.forall(a, p.gt(x, p.integer(0))),
        "pool.exists": lambda: p.exists(x, q.gt(a, q.integer(0))),
        "pool.gt": lambda: p.gt(x, a),
        "pool.pred_and": lambda: p.pred_and([p.gt(x, p.integer(0)), q.gt(a, q.integer(0))]),
        "pool.big_o": lambda: p.big_o(a),
        "pow_expr": lambda: x.pow_expr(a),
        "diff": lambda: ak.diff(x * y, a),
        "integrate": lambda: ak.integrate(x, a),
        "eval_expr": lambda: ak.eval_expr(x, {a: 2.0}),
        "evaluate": lambda: ak.evaluate(x, {a: 2}),
        "subs": lambda: ak.subs(x, {a: 1}),
        "match_pattern": lambda: ak.match_pattern(x, a),
        "make_rule": lambda: ak.make_rule(x, a),
        "simplify_with": lambda: ak.simplify_with(a, [ak.make_rule(x, y)]),
        "limit": lambda: ak.limit(x, a, p.integer(0)),
        "series": lambda: ak.series(x, a, p.integer(0), 3),
        "sum_definite": lambda: ak.sum_definite(x, a, p.integer(0), p.integer(3)),
        "poly_normal": lambda: ak.poly_normal(x * y, [x, a]),
        "resultant": lambda: ak.resultant(x, a, x),
        "horner": lambda: ak.horner(x**2 + x, a),
        "atan2": lambda: ak.atan2(x, a),
        "compile_expr": lambda: ak.compile_expr(x, [a]),
        "interval_eval": lambda: ak.interval_eval(x, {a: ak.ArbBall(1.0)}),
        "matrix_add": lambda: ak.Matrix([[x]]) + ak.Matrix([[a]]),
        "And": lambda: ak.And(p.gt(x, p.integer(0)), q.gt(a, q.integer(0))),
    }
    return cases


@pytest.mark.parametrize("name", sorted(_cases()))
def test_cross_pool_argument_is_refused(name):
    fn = _cases()[name]
    with pytest.raises(PoolError) as ei:
        fn()
    assert getattr(ei.value, "code", None) == "E-POOL-001", ei.value
    assert "E-POOL-001" in str(ei.value)


def test_cross_pool_error_names_the_arguments():
    _p, _q, x, y, a = _pools()
    with pytest.raises(
        PoolError, match=r"diff\(\): `var` belongs to a different ExprPool than `expr`"
    ):
        ak.diff(x * y, a)


def test_same_pool_calls_still_work():
    p, _q, x, y, _a = _pools()
    assert ak.diff(x * y, x).value == y
    assert ak.eval_expr(x, {x: 2.0}) == 2.0
    assert x.pow_expr(y) == x**y
    assert ak.match_pattern(p.symbol("X"), p.symbol("X")) == [{}]


def test_compile_cache_does_not_serve_another_pools_function():
    p, q = ExprPool(), ExprPool()
    x = p.symbol("x")
    u = q.symbol("u")
    cache = ak.CompileCache()
    f = cache.compile(x**2, [x])
    assert f([3.0]) == 9.0
    # Built in the same order, q's `u**2` has the same ids as p's `x**2`;
    # the cached entry must not answer for it.
    assert not cache.contains(u**2, [u])
    assert cache.contains(x**2, [x])
    with pytest.raises(PoolError):
        cache.compile(u**2, [u])
    cache.clear()
    g = cache.compile(u + 1, [u])
    assert g([3.0]) == 4.0
