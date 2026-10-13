"""
Antiderivatives that stay valid across branch cuts (F4).

A substitution antiderivative — ``t = tan(x/2)``, ``t = tan x``, ``u = √cot x``
— is an antiderivative between consecutive poles of the substitution and
nowhere else: its ``atan`` terms jump by ``±π`` where the substituted tangent
runs off to infinity, although the integrand is perfectly regular there.  Then
``F(b) − F(a)`` is not the integral over an interval containing the pole, and
an indefinite result is only correct on a window around the origin.

Pinned here:

* ``∫_{1/5}^{1} cot(x)^{3/2} dx = 2.17202966873…`` again (regressed to a
  refusal once better simplification steered it through a route whose answer
  was wrong on the complex branch; it is now integrated by the reflection
  ``cot u = tan(π/2 − u)``);
* ~25 definite integrals over intervals that cross the poles of the
  substitution (rational trig functions, ``cot^{p/q}``, ``1/(a + b sin x)``,
  integrands invariant under ``x ↦ x + π`` that need ``t = tan x``) agree with
  quadrature to 1e-12;
* indefinite results are continuous on a dense grid over [−20, 20] wherever the
  integrand is finite, and differentiate back to the integrand there.

Oracle: ``mpmath.quad`` when mpmath is installed, otherwise composite
tanh–sinh in pure Python (≈1e-14 here, endpoint singularities included), so
the test runs in every CI shard.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest

try:  # pragma: no cover - exercised in the extras shard
    import mpmath as _mp
except ImportError:  # pragma: no cover
    _mp = None


def _quad(f, a, b):
    """``∫_a^b f`` to ~1e-13, including an algebraic endpoint singularity (``√x``)."""
    if _mp is not None:
        _mp.mp.dps = 30
        pts = [a + (b - a) * k / 40 for k in range(41)]
        return float(_mp.quad(lambda t: f(float(t)), pts))
    # Composite tanh–sinh: nodes cluster at every panel end, so an integrable
    # endpoint singularity costs no accuracy (a Gauss rule loses ~7 digits on
    # √x at 0).
    total, panels, step = 0.0, 40, 1.0 / 32
    width = (b - a) / panels
    for k in range(panels):
        lo = a + k * width
        hi = b if k == panels - 1 else lo + width
        r = 0.5 * (hi - lo)
        for j in range(-128, 129):
            t = j * step
            u = 0.5 * math.pi * math.sinh(t)
            gap = r * math.exp(-abs(u)) / math.cosh(u)
            x = hi - gap if t >= 0 else lo + gap
            if not lo < x < hi:
                continue
            w = step * r * 0.5 * math.pi * math.cosh(t) / math.cosh(u) ** 2
            total += w * f(x)
    return total


def _f(src):
    """Numeric integrand from the same source string (``math`` namespace)."""
    ns = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "cot": lambda u: 1 / math.tan(u),
        "pi": math.pi,
    }
    code = compile(src.replace("^", "**"), "<integrand>", "eval")
    return lambda x: eval(code, dict(ns), {"x": x})


# (integrand, lower, upper) — every interval crosses at least one pole of the
# substitution the integrator uses, or (for the cot/tan radicals) a zero of the
# denominator of an atan argument.
DEFINITE = [
    ("cot(x)^(3/2)", "1/5", "1"),
    ("cot(x)^(1/2)", "1/5", "3/2"),
    ("cot(x)^(5/2)", "1/2", "3/2"),
    ("tan(x)^(1/2)", "1/5", "3/2"),
    ("tan(x)^(3/2)", "1/5", "3/2"),
    ("1/(2+cos(x))", "0", "10"),
    ("1/(3+2*sin(x))", "-7", "9"),
    ("1/(5+3*cos(x))", "0", "20"),
    ("1/(5+3*sin(x))", "-11", "7"),
    ("1/(2+cos(x))^2", "0", "10"),
    ("1/(5-4*cos(x))^2", "1", "12"),
    ("1/(2+sin(x))^3", "0", "10"),
    ("1/(2+sin(x)+cos(x))", "0", "8"),
    ("sin(x)/(2+sin(x))", "0", "9"),
    ("cos(x)/(2+cos(x))", "-10", "10"),
    ("1/(1+sin(x)^2)", "0", "5"),
    ("1/(1+cos(x)^2)", "-4", "5"),
    ("1/(1+3*sin(x)^2)", "0", "4"),
    ("1/(2+cos(x)^2)", "-6", "6"),
    ("sin(x)^2/(1+sin(x)^2)", "0", "6"),
    ("1/(3+sin(x)*cos(x))", "0", "7"),
    ("1/(1+cos(x)^2)^2", "0", "7"),
    ("1/(2+tan(x)^2)", "0", "3"),
    ("1/(4+cos(2*x))", "0", "4"),
    ("1/(3+tan(x)^2)", "-2", "5"),
    # Rationalising substitutions: `u = √x`, `u = x^{1/3}`, `u = tan(x)^{1/3}`
    # leave `atan(N/D)` terms whose denominator vanishes inside the interval.
    ("sqrt(x)/(1+x^2)", "0", "4"),
    ("x^(1/3)/(1+x)", "0", "5"),
    ("tan(x)^(1/3)", "1/10", "3/2"),
    ("cot(x)^(1/3)", "1/10", "3/2"),
]


@pytest.mark.parametrize(("src", "a", "b"), DEFINITE, ids=[c[0] for c in DEFINITE])
def test_definite_across_branch_cuts_matches_quadrature(src, a, b):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    expr = ak.parse(src, pool)
    r = ak.integrate(expr, x, ak.parse(a, pool), ak.parse(b, pool))
    got = ak.eval_expr(r.value, {})
    lo, hi = float(ak.eval_expr(ak.parse(a, pool), {})), float(ak.eval_expr(ak.parse(b, pool), {}))
    want = _quad(_f(src), lo, hi)
    assert abs(got - want) <= 1e-12 * (1 + abs(want)), (src, a, b, got, want, str(r.value))


def test_cot_three_halves_repro():
    """Report 10-9 follow-up: the value is back, and exact (no float-lifted constants)."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.integrate(ak.parse("cot(x)^(3/2)", pool), x, pool.rational(1, 5), pool.integer(1))
    assert abs(ak.eval_expr(r.value, {}) - 2.172029668733499) < 1e-12
    assert "/1125899906842624" not in str(r.value)
    assert "/9007199254740992" not in str(r.value)


# Integrands whose indefinite antiderivative must be continuous on all of ℝ.
INDEFINITE = [
    "1/(2+cos(x))",
    "1/(3+2*sin(x))",
    "1/(2+cos(x))^2",
    "1/(5-4*cos(x))^2",
    "1/(2+sin(x))^3",
    "1/(1+sin(x)^2)",
    "1/(1+cos(x)^2)",
    "1/(1+3*sin(x)^2)",
    "1/(2+cos(x)^2)",
    "1/(3+sin(x)*cos(x))",
    "1/(1+cos(x)^2)^2",
    "1/(3+tan(x)^2)",
    "1/(2+tan(x)^2)",
    "sin(x)^2/(1+sin(x)^2)",
]

GRID = [-20.0 + 0.0137 * i for i in range(2920)]  # dense, avoids lattice points


def _ev(expr, sym, v):
    try:
        return ak.eval_expr(expr, {sym: v})
    except ValueError:
        return math.nan


@pytest.mark.parametrize("src", INDEFINITE)
def test_indefinite_is_continuous_and_differentiates_back(src):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    expr = ak.parse(src, pool)
    big_f = ak.integrate(expr, x).value
    f = _f(src)
    # Continuity: consecutive grid values move by at most what the integrand
    # allows (mean value theorem, with slack for sampling sup|f|).
    prev = None
    for xv in GRID:
        fv = _ev(big_f, x, xv)
        assert math.isfinite(fv), (src, xv, str(big_f))
        if prev is not None:
            px, pv = prev
            bound = max(abs(f(px)), abs(f(xv)), abs(f(0.5 * (px + xv)))) * (xv - px)
            assert abs(fv - pv) <= 4 * bound + 1e-9, (src, px, xv, pv, fv, str(big_f))
        prev = (xv, fv)
    # Derivative: d/dx F = f on a spread of points across [−20, 20].
    d = ak.diff(big_f, x).value
    for xv in GRID[::97]:
        dv = _ev(d, x, xv)
        assert abs(dv - f(xv)) <= 1e-8 * (1 + abs(f(xv))), (src, xv, dv, f(xv))


def test_global_correction_is_exact_over_many_periods():
    """F(b) − F(a) of the corrected indefinite result is the integral itself."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    for src, a, b in [("1/(1+sin(x)^2)", -17.3, 19.1), ("1/(2+tan(x)^2)", -9.0, 14.5)]:
        big_f = ak.integrate(ak.parse(src, pool), x).value
        got = _ev(big_f, x, b) - _ev(big_f, x, a)
        want = _quad(_f(src), a, b)
        assert abs(got - want) <= 1e-10 * (1 + abs(want)), (src, got, want)
