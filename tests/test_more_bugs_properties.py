"""Hypothesis property tests for second-audit bugfix areas."""

from __future__ import annotations

import math

import alkahest as ak
import hypothesis.strategies as st
import pytest
from alkahest import NumberTheoryError
from alkahest.number_theory import factorint, isprime
from hypothesis import given

perfect_square_n = st.integers(min_value=1, max_value=50)


@given(n=perfect_square_n)
def test_simplify_sqrt_perfect_square(n: int) -> None:
    pool = ak.ExprPool()
    n_sq = n * n
    e = ak.sqrt(pool.integer(n_sq))
    assert str(ak.simplify(e).value) == str(n)


@given(xv=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_piecewise_interp_matches_compile(xv: float) -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(x, pool.integer(0)), x)], pool.integer(-1))
    f = ak.compile_expr(pw, [x])
    direct = ak.eval_expr(pw, {x: xv})
    compiled = f([xv])
    assert direct == pytest.approx(compiled)
    assert not math.isnan(compiled)


@given(a=st.integers(min_value=-20, max_value=20), b=st.integers(min_value=-20, max_value=20))
def test_match_pattern_literal_never_binds_expr_to_whole_pow(a: int, b: int) -> None:
    if a == 0:
        return
    pool = ak.ExprPool()
    x = pool.symbol("x")
    expr = x**a
    # pattern x must not wildcard-bind to x**a when wildcards=False
    literal = ak.match_pattern(x, expr, wildcards=False)
    assert not any(m.get("x") == expr for m in literal)


@given(n=st.integers(min_value=1, max_value=100))
def test_fold_predicates_via_subs(n: int) -> None:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pw = ak.piecewise([(pool.gt(pool.integer(n), pool.integer(0)), x)], pool.integer(0))
    out = ak.subs(pw, {x: pool.integer(1)})
    assert ">" not in str(out)


def test_factorint_zero_property() -> None:
    with pytest.raises(NumberTheoryError):
        factorint(0)


def test_isprime_negative_property() -> None:
    with pytest.raises(NumberTheoryError):
        isprime(-1)
