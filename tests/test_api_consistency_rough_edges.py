"""API consistency rough edges: ints in constructors, optional vars, error codes."""

from __future__ import annotations

import pytest

import alkahest as ak


@pytest.fixture
def pool_x():
    pool = ak.ExprPool()
    return pool, pool.symbol("x")


def test_unipoly_from_coefficients_accepts_python_ints(pool_x):
    pool, x = pool_x
    poly = ak.UniPoly.from_coefficients([-1, 0, 1], x)
    assert poly.degree() == 2
    assert poly.coefficients() == [-1, 0, 1]


def test_cancel_infers_vars(pool_x):
    _pool, x = pool_x
    out = ak.cancel((x**2 - 1) / (x - 1))
    simplified = ak.simplify(out).value
    assert simplified == x + 1 or simplified == 1 + x


def test_together_infers_vars(pool_x):
    _pool, x = pool_x
    out = ak.together(1 / x + 1 / (x + 1))
    assert out is not None


def test_multipoly_from_symbolic_infers_vars(pool_x):
    pool, x = pool_x
    y = pool.symbol("y")
    mp = ak.MultiPoly.from_symbolic(x**2 * y + x)
    assert not mp.is_zero()


def test_error_message_includes_stable_code(pool_x):
    _pool, x = pool_x
    with pytest.raises(ak.IntegrationError) as ei:
        ak.integrate(ak.exp(-(x**2)), x)
    assert ei.value.code == "E-INT-004"
    assert "[E-INT-004]" in str(ei.value)
