"""Standard-calculus gaps found by the 2026-11-21 CAS comparison: parser
spellings, Gamma-ratio limits at infinity, rational sums through digamma, and
the Airy / Bessel / reduction-of-order classes of ``dsolve``."""

from __future__ import annotations

import math

import alkahest as ak
import alkahest.experimental as ex
import pytest


@pytest.fixture
def pool():
    return ak.ExprPool()


def _ev(e, env=None):
    return complex(ak.eval_expr(e, env or {})).real


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "x0", "want"),
    [
        ("factorial(x)", 3.5, math.gamma(4.5)),
        ("binomial(x, 2)", 5.5, 5.5 * 4.5 / 2),
        ("harmonic(x)", 4.0, 1 + 1 / 2 + 1 / 3 + 1 / 4),
        ("acot(x)", 2.0, math.atan(0.5)),
        ("asec(x)", 2.0, math.acos(0.5)),
        ("acsc(x)", 2.0, math.asin(0.5)),
        ("acoth(x)", 2.0, math.atanh(0.5)),
        ("asech(x)", 0.5, math.acosh(2.0)),
        ("acsch(x)", 2.0, math.asinh(0.5)),
        ("Abs(x)", -2.5, 2.5),
        ("ln(x)", 2.0, math.log(2.0)),
        ("max(x, 1)", 0.5, 1.0),
        ("sec(x)", 0.3, 1 / math.cos(0.3)),
    ],
)
def test_parse_function_spellings(pool, src, x0, want):
    x = pool.symbol("x")
    e = ak.parse(src, pool, {"x": x})
    assert abs(_ev(e, {x: x0}) - want) < 1e-9


def test_parse_exact_literals(pool):
    assert str(ak.parse("factorial(5)", pool)) == "120"
    assert str(ak.parse("binomial(6, 2)", pool)) == "15"
    assert str(ak.parse("harmonic(4)", pool)) == "25/12"


def test_parse_wrong_arity_is_a_parse_error(pool):
    with pytest.raises(ak.ParseError):
        ak.parse("binomial(3)", pool)
    with pytest.raises(ak.ParseError):
        ak.parse("acot(1, 2)", pool)


def test_harmonic_constructor(pool):
    n = pool.symbol("n")
    assert str(ak.harmonic(pool.integer(3))) == "11/6"
    h = ak.harmonic(n)
    assert abs(_ev(h, {n: 9.0}) - sum(1 / j for j in range(1, 10))) < 1e-12
    h2 = ak.harmonic(n, 2)
    assert abs(_ev(h2, {n: 5.0}) - sum(1 / j**2 for j in range(1, 6))) < 1e-12


# ---------------------------------------------------------------------------
# Limits at +oo of gamma-family expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "want"),
    [
        ("gamma(x+1/2)/(gamma(x)*sqrt(x))", 1.0),
        ("gamma(x+1)/(x*gamma(x))", 1.0),
        ("gamma(x+3)/(gamma(x)*x^3)", 1.0),
        ("loggamma(x)/(x*log(x))", 1.0),
        ("digamma(x)/log(x)", 1.0),
        ("factorial(x)/gamma(x+1)", 1.0),
        ("gamma(x)/gamma(x+1)", 0.0),
    ],
)
def test_gamma_ratio_limits(pool, src, want):
    x = pool.symbol("x")
    r = ak.limit(ak.parse(src, pool, {"x": x}), x, pool.pos_infinity())
    assert abs(_ev(r.value) - want) < 1e-12


def test_gamma_growth_is_infinite(pool):
    x = pool.symbol("x")
    r = ak.limit(ak.parse("gamma(x)/exp(x)", pool, {"x": x}), x, pool.pos_infinity())
    assert str(r.value) in ("oo", "∞")


def test_gamma_inside_a_sum_is_not_rewritten_wrongly(pool):
    # Γ(x+1)/Γ(x) − x ≡ 0, but the leading Stirling term alone gives a
    # non-zero constant; the rewrite must not apply to a sum term.
    x = pool.symbol("x")
    e = ak.parse("gamma(x+1)/gamma(x) - x", pool, {"x": x})
    try:
        r = ak.limit(e, x, pool.pos_infinity())
    except ak.AlkahestError:
        return
    assert abs(_ev(r.value)) < 1e-12


# ---------------------------------------------------------------------------
# Rational sums via digamma
# ---------------------------------------------------------------------------


def _check_sum(pool, src, lo, n0, f):
    k, n = pool.symbol("k"), pool.symbol("n")
    term = ak.parse(src, pool, {"k": k})
    r = ak.sum_definite(term, k, pool.integer(lo), n)
    want = sum(f(j) for j in range(lo, n0 + 1))
    assert abs(_ev(r.value, {n: float(n0)}) - want) < 1e-9 * max(1, abs(want))
    return r


def test_harmonic_sum(pool):
    r = _check_sum(pool, "1/k", 1, 9, lambda j: 1 / j)
    s = str(r.value)
    assert "digamma" in s
    assert "EulerGamma" in s


@pytest.mark.parametrize(
    ("src", "lo", "f"),
    [
        ("1/k^2", 1, lambda j: 1 / j**2),
        ("1/k^3", 1, lambda j: 1 / j**3),
        ("1/(2*k+1)", 0, lambda j: 1 / (2 * j + 1)),
        ("1/(k*(k+2))", 1, lambda j: 1 / (j * (j + 2))),
        ("(k^2+1)/(k+1)", 0, lambda j: (j * j + 1) / (j + 1)),
        ("1/(k+1/3)", 0, lambda j: 1 / (j + 1 / 3)),
    ],
)
def test_rational_sums(pool, src, lo, f):
    _check_sum(pool, src, lo, 8, f)


def test_sum_with_a_pole_in_range_is_refused(pool):
    k, n = pool.symbol("k"), pool.symbol("n")
    with pytest.raises(ak.AlkahestError):
        ak.sum_definite(ak.parse("1/k", pool, {"k": k}), k, pool.integer(0), n)


# ---------------------------------------------------------------------------
# dsolve: Airy, Bessel, y-absent nonlinear
# ---------------------------------------------------------------------------


def _solve(pool, src, order):
    x, y = pool.symbol("x"), pool.symbol("y")
    ds = [pool.symbol(s) for s in ("yp", "ypp", "yppp")][:order]
    syms = {"x": x, "y": y, "yp": ds[0]}
    if order > 1:
        syms["ypp"] = ds[1]
    eq = ak.parse(src, pool, syms)
    sols = ex.dsolve(eq, x, y, ds)
    return sols[0], x


@pytest.mark.parametrize(
    ("src", "fn"),
    [
        ("ypp - x*y", "airy_ai"),
        ("ypp - (2*x+1)*y", "airy_ai"),
        ("ypp + x*y", "airy_ai"),
        ("ypp + 2*yp - x*y", "airy_ai"),
        ("x^2*ypp + x*yp + x^2*y", "bessel_j"),
        ("x^2*ypp + x*yp + (x^2 - 1/4)*y", "bessel_j"),
        ("x^2*ypp + x*yp + (4*x^2 - 9)*y", "bessel_j"),
        ("x^2*ypp + x*yp - (x^2 + 1)*y", "bessel_i"),
        ("x*ypp + yp + x*y", "bessel_j"),
        ("x*ypp + 2*yp + x*y", "bessel_j"),
    ],
)
def test_dsolve_special_function_classes(pool, src, fn):
    s0, _ = _solve(pool, src, 2)
    assert fn in str(s0["y_of_x"])
    assert len(s0["constants"]) == 2


def test_dsolve_y_absent_nonlinear(pool):
    s0, _x = _solve(pool, "ypp - yp^2", 2)
    assert "log" in str(s0["y_of_x"])
    assert len(s0["constants"]) == 2
