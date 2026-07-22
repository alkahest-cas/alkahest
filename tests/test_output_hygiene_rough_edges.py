"""Output hygiene rough edges: display, cancel, gamma(1), division by zero."""

from __future__ import annotations

import alkahest as ak
import pytest


@pytest.fixture
def pool_x():
    pool = ak.ExprPool()
    return pool, pool.symbol("x")


def test_cancel_omits_unit_coeff(pool_x):
    _pool, x = pool_x
    out = ak.cancel((x**2 - 1) / (x - 1), [x])
    assert str(out) in {"(x + 1)", "(1 + x)", "x + 1"}


def test_nested_pow_display_is_parenthesized(pool_x):
    pool, x = pool_x
    half = pool.integer(1) / pool.integer(2)
    nested = (x**half) ** 3
    s = str(nested)
    assert ")^" in s or s.startswith("(")
    assert s != "x^1/2^3"


def test_sqrt_integral_latex_parens(pool_x):
    _pool, x = pool_x
    v = ak.integrate(ak.sqrt(x), x).value
    tex = ak.latex(v)
    # Nested power of a root must not render as ambiguous \sqrt{x}^3
    assert r"\sqrt{x}^3" not in tex


def test_gamma_one_simplifies(pool_x):
    pool, _x = pool_x
    g = ak.gamma(pool.integer(1))
    assert str(ak.simplify(g).value) == "1"


def test_div_by_zero_raises(pool_x):
    pool, _x = pool_x
    with pytest.raises(ZeroDivisionError):
        _ = pool.integer(1) / pool.integer(0)
