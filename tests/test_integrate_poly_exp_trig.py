"""∫ P(x)·e^{ax+b}·Π sin/cos(cx+d) dx — polynomial × exponential × trig (B3).

The 3.11.0 wheel refused ``∫ x⁶·e^{2x}·sin(x) dx`` with E-INT-001; products
and powers of the trig factor (``eˣ·sin²x``, ``x²eˣ·sin(x)·cos(2x)``) were
still refused after the single-factor case was closed.  Every result here is
checked by differentiation — the symbolic residual ``F′ − f`` must simplify
to 0, and a central difference must agree at random points — and must be
real (no ``I``) for real inputs.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import alkahest as ak
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _residual_is_zero(F, f, x):
    d = ak.diff(F, x)
    d = getattr(d, "value", d)
    # Exact: some simplifier must reduce F' − f to the literal 0.  Products of
    # trig factors with different arguments need the expanded / trig normal
    # forms (product-to-sum is not in the default rule set).
    for simp in (ak.simplify, ak.simplify_expanded, ak.simplify_trig_normal_form):
        z = simp(d - f)
        if str(getattr(z, "value", z)) == "0":
            return True
    return False


def _numeric_check(F, f, x, xs):
    """F'(x) == f(x) at the sample points, with F' from symbolic differentiation."""
    d = ak.diff(F, x)
    d = getattr(d, "value", d)
    for xv in xs:
        dv = ak.eval_expr(d, {x: xv})
        fv = ak.eval_expr(f, {x: xv})
        assert abs(dv - fv) <= 1e-7 * (1 + abs(fv)), (str(F), xv, dv, fv)


def _check(src, rng=None):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.parse(src, pool)
    r = ak.integrate(f, x)
    F = r.value
    assert "I" not in str(F).replace("Integral", ""), str(F)
    assert _residual_is_zero(F, f, x), (src, str(F))
    rng = rng or random.Random(src)
    _numeric_check(F, f, x, [rng.uniform(-1.5, 1.5) for _ in range(4)])
    return r


def test_repro_x6_exp2x_sinx():
    r = _check("x**6*exp(2*x)*sin(x)")
    assert r.verification["status"] == "exactly_verified"


@pytest.mark.parametrize(
    "src",
    [
        "exp(x)*sin(x)**2",
        "exp(-x)*cos(x)**3",
        "x**2*exp(x)*sin(x)*cos(2*x)",
        "x*exp(x)*sin(x)*sin(x+1)",
        "x**3*exp(-2*x+1)*sin(x)**3*cos(x+2)",
        "x**8*exp(x/2)*cos(3*x)**2",
        "(x+1)**2*exp(3*x)*sin(2*x-1)",
        "x**4*exp(2*x)*sin(x)**2 + x*exp(x)*cos(x)",
        # a = 0: polynomial × trig product.
        "x**2*sin(x)*cos(3*x)",
        # c = 0 after product-to-sum: sin(x)·sin(x) has a constant half.
        "x**5*exp(x/3)*sin(2*x)**2",
    ],
)
def test_products_and_powers_of_trig(src):
    _check(src)


def test_symbolic_rates():
    pool = ak.ExprPool()
    x, a, c = pool.symbol("x"), pool.symbol("a"), pool.symbol("c")
    f = ak.parse("x*exp(a*x)*sin(c*x)**2", pool)
    r = ak.integrate(f, x)
    rng = random.Random(7)
    for _ in range(3):
        env = {a: rng.uniform(0.4, 2.0), c: rng.uniform(0.4, 2.0)}
        for xv in (0.3, 0.9, 1.4):
            h = 1e-6
            d = (
                ak.eval_expr(r.value, {**env, x: xv + h})
                - ak.eval_expr(r.value, {**env, x: xv - h})
            ) / (2 * h)
            fv = ak.eval_expr(f, {**env, x: xv})
            assert abs(d - fv) <= 1e-5 * (1 + abs(fv))


def _frac(fr: Fraction) -> str:
    if fr.denominator == 1:
        return f"({fr.numerator})"
    return f"({fr.numerator}/{fr.denominator})"


def _lin(c: Fraction, d: Fraction) -> str:
    if d == 0:
        return f"{_frac(c)}*x"
    return f"{_frac(c)}*x + {_frac(d)}"


_rat = st.fractions(min_value=-3, max_value=3, max_denominator=3)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    coeffs=st.lists(st.integers(-4, 4), min_size=1, max_size=9),
    a=_rat,
    b=_rat,
    c=_rat,
    d=_rat,
    kinds=st.lists(st.sampled_from(["sin", "cos"]), min_size=1, max_size=2),
    c2=_rat,
)
def test_random_poly_exp_trig(coeffs, a, b, c, d, kinds, c2):
    if all(k == 0 for k in coeffs):
        coeffs = [1]
    poly = " + ".join(f"({k})*x**{i}" for i, k in enumerate(coeffs) if k != 0)
    trig = f"{kinds[0]}({_lin(c, d)})"
    if len(kinds) == 2:
        trig += f"*{kinds[1]}({_lin(c2, Fraction(0))})"
    src = f"({poly})*exp({_lin(a, b)})*{trig}"
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.parse(src, pool)
    r = ak.integrate(f, x)
    F = r.value
    assert "I" not in str(F), str(F)
    rng = random.Random(src)
    _numeric_check(F, f, x, [rng.uniform(-1.2, 1.2) for _ in range(4)])
    assert _residual_is_zero(F, f, x), (src, str(F))


def test_repro_is_fast():
    import time

    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.parse("x**6*exp(2*x)*sin(x)", pool)
    ak.integrate(f, x)  # warm
    best = math.inf
    for _ in range(3):
        t = time.perf_counter()
        ak.integrate(f, x)
        best = min(best, time.perf_counter() - t)
    # SymPy takes ~4 s; generous bound for sanitizer / loaded CI runners.
    assert best < 2.0, best
