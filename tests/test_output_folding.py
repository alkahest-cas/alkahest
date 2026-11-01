"""Printed results carry no unfolded literal arithmetic (launch report #10, #12).

The printer is unchanged; these pin the *producers* that used to hand it
`(1 * 1)`, `(x * 1)`, `(1/3 + 1/6)`, `0^(1/2)` and `sqrt(0)`, and
``simplify_expanded`` silently declining a small expansion.
"""

from __future__ import annotations

import math
import warnings

import alkahest as ak
import pytest

P = ak.ExprPool()
x, y, z = P.symbol("x"), P.symbol("y"), P.symbol("z")
I = P.integer  # noqa: E741


def _no_unit_products(s: str) -> None:
    for junk in ("* 1)", "(1 *", "* 0)", "(0 *", "0^(1/2)", "sqrt(0)"):
        assert junk not in s, f"{junk!r} in {s}"


def test_series_terms_drop_unit_factors():
    s = str(ak.series(ak.exp(x), x, I(0), 6))
    _no_unit_products(s)
    assert "O(x^6)" in s


def test_matrix_product_folds_literals():
    m = ak.Matrix([[I(1), I(2)], [I(3), I(4)]])
    eye = ak.Matrix([[I(1), I(0)], [I(0), I(1)]])
    prod = m @ eye
    assert str(prod) == str(m)
    sym = ak.Matrix([[x, I(0)], [I(0), y]]) @ ak.Matrix([[I(1), I(2)], [I(3), I(1)]])
    _no_unit_products(str(sym))
    # Values unchanged.
    for r in range(2):
        for c in range(2):
            a = ak.eval_expr(sym.get(r, c), {x: 1.5, y: -0.25})
            want = [[1.5 * 1, 1.5 * 2], [-0.25 * 3, -0.25 * 1]][r][c]
            assert math.isclose(a, want)


def test_rational_literals_add_exactly():
    assert P.rational(1, 3) + P.rational(1, 6) == P.rational(1, 2)
    assert I(2) + I(3) == I(5)
    assert P.rational(1, 2) + P.rational(1, 2) == I(1)


@pytest.mark.parametrize(
    ("integrand", "lo", "hi", "value"),
    [
        (1 / ak.sqrt(x), I(0), I(1), 2.0),
        (ak.sqrt(1 - x**2), I(0), I(1), math.pi / 4),
        (ak.sqrt(x), I(0), I(4), 16.0 / 3.0),
    ],
)
def test_definite_integral_endpoints_are_folded(integrand, lo, hi, value):
    r = ak.integrate(integrand, x, lo, hi)
    _no_unit_products(str(r.value))
    assert math.isclose(ak.eval_expr(r.value, {}), value, rel_tol=1e-12)


def test_simplify_expanded_expands_small_high_powers():
    r = ak.simplify_expanded((x + 1) ** 13)
    s = str(r.value)
    assert "^13" in s
    assert "(x + 1)^13" not in s
    assert math.isclose(ak.eval_expr(r.value, {x: 0.3}), 1.3**13, rel_tol=1e-12)
    r = ak.simplify_expanded((x + y + z) ** 8)
    assert "^8" in str(r.value)
    assert "(x + y + z)^8" not in str(r.value)
    env = {x: 0.3, y: -0.7, z: 1.1}
    assert math.isclose(ak.eval_expr(r.value, env), (0.3 - 0.7 + 1.1) ** 8, rel_tol=1e-9)


def test_simplify_expanded_says_so_when_it_declines():
    big = sum((P.symbol(f"v{i}") for i in range(10)), I(0)) ** 9
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = ak.simplify_expanded(big)
    assert any("unexpanded" in str(m.message) for m in w)
    assert any(st["rule"] == "expand_pow_limit_reached" for st in r.steps)
