"""Regression tests for second audit (report6-2-more-bugs.md)."""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from alkahest import NumberTheoryError
from alkahest.number_theory import factorint, isprime


def test_multipoly_display_uses_symbol_names() -> None:
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    f = ak.MultiPoly.from_symbolic((x + y) * (x - y), [x, y])
    s = str(f)
    assert "x" in s
    assert "y" in s
    assert "x0" not in s
    assert "x1" not in s


def test_sqrt_integer_simplifies() -> None:
    pool = ak.ExprPool()
    e = ak.sqrt(pool.integer(4))
    assert str(ak.simplify(e).value) == "2"


def test_compile_piecewise_not_nan() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), x)], pool.integer(-1))
    f = ak.compile_expr(pw, [x])
    assert f([1.0]) == pytest.approx(1.0)
    assert f([-1.0]) == pytest.approx(-1.0)
    assert not math.isnan(f([1.0]))


def test_eval_piecewise_and_predicate() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), x)], pool.integer(-1))
    assert ak.eval_expr(pw, {x: 1.0}) == pytest.approx(1.0)
    assert ak.eval_expr(pool.gt(x, pool.integer(0)), {x: 1.0}) == pytest.approx(1.0)


def test_interval_eval_piecewise() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), x)], pool.integer(-1))
    ball = ak.interval_eval(pw, {x: ak.ArbBall(1.0, 1e-6)})
    assert ball.contains(1.0)


def test_compile_matches_eval_expr_piecewise() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), x)], pool.integer(-1))
    f = ak.compile_expr(pw, [x])
    for xv in (-2.0, 0.5, 3.0):
        assert ak.eval_expr(pw, {x: xv}) == pytest.approx(f([xv]))


def test_match_pattern_literal_symbols() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    wild = ak.match_pattern(x, x**2)
    assert any(m.get("x") == x**2 for m in wild)
    literal = ak.match_pattern(x, x**2, wildcards=False)
    assert not any(m.get("x") == x**2 for m in literal)


def test_subs_folds_numeric_predicates() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(pool.integer(2), pool.integer(0)), x)], pool.integer(0))
    out = ak.subs(pw, {x: pool.integer(1)})
    assert ">" not in str(out)


def test_simplify_x_pow_zero() -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    assert str(ak.simplify(x**0).value) == "1"


def test_factorint_zero_raises() -> None:
    with pytest.raises(NumberTheoryError):
        factorint(0)


def test_isprime_negative_raises() -> None:
    with pytest.raises(NumberTheoryError):
        isprime(-5)
