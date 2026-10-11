"""Integrator pre-normalisation and symbolic parameters (report 10-9, G5).

Each case is checked against an independent oracle: indefinite results are
differentiated back numerically (central differences) at random positive
parameter values, and definite results are compared with the closed form or
mpmath quadrature.  Where SymPy is available the antiderivative is also
compared with SymPy's at the same points.
"""

from __future__ import annotations

import math
import random

import alkahest as ak
import pytest
from alkahest import Domain

mpmath = pytest.importorskip("mpmath")
mpmath.mp.dps = 30


def _eval(expr, env):
    return ak.eval_expr(expr, env)


def _check_antiderivative(F, f, var, params, rng, xs=(0.37, 0.91, 1.42, 2.17)):
    """F'(x) == f(x) by a central difference at random parameter values."""
    for _ in range(3):
        env = {p: rng.uniform(0.4, 2.6) for p in params}
        for xv in xs:
            h = 1e-6
            e1 = dict(env)
            e1[var] = xv + h
            e0 = dict(env)
            e0[var] = xv - h
            ex = dict(env)
            ex[var] = xv
            d = (_eval(F, e1) - _eval(F, e0)) / (2 * h)
            fv = _eval(f, ex)
            assert abs(d - fv) <= 1e-5 * (1 + abs(fv)), (str(F), xv, env, d, fv)


@pytest.fixture
def P():
    return ak.ExprPool()


def test_reciprocal_sum_of_squares_symbolic(P):
    k, m = P.symbol("k"), P.symbol("m")
    r = ak.integrate(1 / (k**2 + m**2), k)
    _check_antiderivative(r.value, 1 / (k**2 + m**2), k, [m], random.Random(1))
    assert "atan" in str(r.value)


def test_reciprocal_quadratic_unknown_sign_states_condition(P):
    x, a, b = P.symbol("x"), P.symbol("a"), P.symbol("b")
    f = 1 / (a * x**2 + b)
    r = ak.integrate(f, x)
    assert "atan" in str(r.value)
    # 4ab > 0 is the stated side condition; at a, b > 0 the result is right.
    conds = r.verification["side_conditions"]
    assert conds, "a sign-dependent arctangent must state its side condition"
    _check_antiderivative(r.value, f, x, [a, b], random.Random(2))


def test_reciprocal_quadratic_positive_parameters_no_condition(P):
    x = P.symbol("x")
    a, b = P.symbol("a", Domain.Positive), P.symbol("b", Domain.Positive)
    f = 1 / (a * x**2 + b)
    r = ak.integrate(f, x)
    assert "atan" in str(r.value)
    _check_antiderivative(r.value, f, x, [a, b], random.Random(3))


def test_reciprocal_quadratic_negative_discriminant_gives_logs(P):
    x = P.symbol("x")
    m = P.symbol("m", Domain.Positive)
    f = 1 / (x**2 - m**2)
    r = ak.integrate(f, x)
    assert "log" in str(r.value)
    _check_antiderivative(r.value, f, x, [m], random.Random(4), xs=(3.1, 4.7, 5.3))


@pytest.mark.parametrize("sign", [1, -1])
def test_poly_times_exp_symbolic_rate(P, sign):
    x, k = P.symbol("x"), P.symbol("k")
    for n in range(0, 4):
        f = x**n * ak.exp(sign * k * x) if n else ak.exp(sign * k * x)
        r = ak.integrate(f, x)
        _check_antiderivative(r.value, f, x, [k], random.Random(10 + n))


def test_poly_exp_trig_tabular(P):
    x = P.symbol("x")
    for n in (0, 1, 2, 3):
        for trig in (ak.sin, ak.cos):
            f = x**n * ak.exp(2 * x) * trig(x) if n else ak.exp(2 * x) * trig(x)
            r = ak.integrate(f, x)
            _check_antiderivative(r.value, f, x, [], random.Random(20 + n))


def test_poly_exp_trig_symbolic(P):
    x, k, w = P.symbol("x"), P.symbol("k"), P.symbol("w")
    f = x * ak.exp(-k * x) * ak.cos(w * x)
    r = ak.integrate(f, x)
    _check_antiderivative(r.value, f, x, [k, w], random.Random(30))


def test_fisher_information_unexpanded(P):
    z, s = P.symbol("z"), P.symbol("s")
    f = (s * z) ** 2 * ak.exp(-(z**2) / 2)
    r = ak.integrate(f, z)
    _check_antiderivative(r.value, f, z, [s], random.Random(40))
    d = ak.integrate(f, z, P.neg_infinity(), P.pos_infinity())
    sv = 1.7
    assert abs(_eval(d.value, {s: sv}) - sv**2 * math.sqrt(2 * math.pi)) < 1e-9


def test_density_convolution_merges_exponentials(P):
    tau, t, a, b = P.symbol("tau"), P.symbol("t"), P.symbol("a"), P.symbol("b")
    f = ak.exp(-a * tau) * ak.exp(-b * (t - tau))
    r = ak.integrate(f, tau)
    _check_antiderivative(r.value, f, tau, [a, b, t], random.Random(50))
    d = ak.integrate(f, tau, P.integer(0), t)
    av, bv, tv = 0.7, 1.9, 1.3
    want = mpmath.quad(lambda u: mpmath.e ** (-av * u - bv * (tv - u)), [0, tv])
    assert abs(_eval(d.value, {a: av, b: bv, t: tv}) - float(want)) < 1e-10


def test_order_statistic_combines_powers(P):
    x = P.symbol("x")
    n = P.symbol("n", Domain.Positive)
    d = ak.integrate(x * n * x ** (n - 1), x, P.integer(0), P.integer(1))
    for nv in (0.5, 3.0, 7.25):
        assert abs(_eval(d.value, {n: nv}) - nv / (nv + 1)) < 1e-12


def test_order_statistic_unknown_exponent_sign_refuses(P):
    # 0^(n+1) at the lower bound is 0 only for n > -1: with nothing known about
    # n the endpoint value is not established, so the integral is refused.
    x, n = P.symbol("x"), P.symbol("n")
    with pytest.raises(ak.IntegrationError) as e:
        ak.integrate(x * n * x ** (n - 1), x, P.integer(0), P.integer(1))
    assert "E-INT-001" in str(e.value)


def test_fem_mass_power_of_sum(P):
    x, L = P.symbol("x"), P.symbol("L")
    f = (1 + x**2 / L**2) ** 2
    r = ak.integrate(f, x)
    _check_antiderivative(r.value, f, x, [L], random.Random(60))
    d = ak.integrate(f, x, P.integer(0), L)
    for Lv in (0.8, 2.5):
        assert abs(_eval(d.value, {L: Lv}) - 28 * Lv / 15) < 1e-12


def test_zoh_cancels_rational_factor(P):
    tau, a = P.symbol("tau"), P.symbol("a")
    r = ak.integrate(tau * (a * tau) ** -1, tau)
    _check_antiderivative(r.value, tau * (a * tau) ** -1, tau, [a], random.Random(70))


def test_bateman_auc_and_aumc(P):
    t = P.symbol("t", Domain.Positive)
    ke, ka = P.symbol("ke", Domain.Positive), P.symbol("ka", Domain.Positive)
    bat = ak.exp(-ke * t) - ak.exp(-ka * t)
    auc = ak.integrate(bat, t, P.integer(0), P.pos_infinity())
    aumc = ak.integrate(t * bat, t, P.integer(0), P.pos_infinity())
    for kev, kav in ((0.3, 1.7), (2.2, 0.9)):
        env = {ke: kev, ka: kav}
        assert abs(_eval(auc.value, env) - (1 / kev - 1 / kav)) < 1e-12
        assert abs(_eval(aumc.value, env) - (1 / kev**2 - 1 / kav**2)) < 1e-12


def test_exponential_moments_positive_rate(P):
    x = P.symbol("x")
    a = P.symbol("a", Domain.Positive)
    for n in range(0, 4):
        f = x**n * ak.exp(-a * x) if n else ak.exp(-a * x)
        d = ak.integrate(f, x, P.integer(0), P.pos_infinity())
        av = 1.3
        assert abs(_eval(d.value, {a: av}) - math.factorial(n) / av ** (n + 1)) < 1e-12


def test_chi_square_normaliser_gamma_family(P):
    y = P.symbol("y")
    d = ak.integrate(ak.exp(-y / 2) / ak.sqrt(y), y, P.integer(0), P.pos_infinity())
    assert abs(_eval(d.value, {}) - math.sqrt(2 * math.pi)) < 1e-12
    x = P.symbol("x")
    a = P.symbol("a", Domain.Positive)
    d2 = ak.integrate(x ** P.rational(3, 2) * ak.exp(-a * x), x, P.integer(0), P.pos_infinity())
    av = 0.8
    want = float(mpmath.gamma(2.5) / mpmath.mpf(av) ** 2.5)
    assert abs(_eval(d2.value, {a: av}) - want) < 1e-12


def test_unknown_rate_sign_still_refuses(P):
    x, a = P.symbol("x"), P.symbol("a")
    with pytest.raises(ak.IntegrationError) as e:
        ak.integrate(ak.exp(-a * x), x, P.integer(0), P.pos_infinity())
    assert "E-INT-001" in str(e.value)
    with pytest.raises(ak.IntegrationError):
        ak.integrate(x * ak.exp(-a * x), x, P.integer(0), P.pos_infinity())


def test_gaussian_unchanged(P):
    x = P.symbol("x")
    r = ak.integrate(ak.exp(-(x**2)), x)
    assert "erf" in str(r.value)


@pytest.mark.parametrize(
    "case",
    ["x*exp(-k*x)", "1/(k**2+m**2)", "x**2*exp(2*x)*sin(x)"],
)
def test_against_sympy(P, case):
    sympy = pytest.importorskip("sympy")
    syms = {n: P.symbol(n) for n in ("x", "k", "m")}
    var = syms["x"] if "x" in case else syms["k"]
    expr = eval(case, {"exp": ak.exp, "sin": ak.sin, **syms})
    F = ak.integrate(expr, var).value
    sx = {n: sympy.Symbol(n, positive=True) for n in syms}
    sF = sympy.integrate(sympy.sympify(case, locals=sx), sx[str(var)])
    rng = random.Random(99)
    for _ in range(3):
        env = {n: rng.uniform(0.5, 2.5) for n in syms}
        p0 = {syms[n]: v for n, v in env.items()}
        p1 = dict(p0)
        p1[var] = env[str(var)] + 0.5
        ours = _eval(F, p1) - _eval(F, p0)
        s0 = {sx[n]: v for n, v in env.items()}
        s1 = dict(s0)
        s1[sx[str(var)]] = env[str(var)] + 0.5
        theirs = float(sF.subs(s1) - sF.subs(s0))
        assert abs(ours - theirs) < 1e-9 * (1 + abs(theirs))
