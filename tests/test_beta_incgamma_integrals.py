"""Beta / incomplete-gamma primitives and the definite integrals that need them (F11).

Report 10-9, G5 follow-ups: Feynman-parameter / Beta-family integrals
(``∫₀¹ xᵃ(1−x)ᵇ dx`` with symbolic exponents, ``∫₀¹ log(m² − x(1−x)p²) dx``,
``∫₀¹ dx/(m² − x(1−x)p²)ⁿ``), improper integrals from a nonzero lower bound
(``∫_a^∞ x^{s−1}e^{−x} dx = Γ(s, a)``) and readable definite results.

Every value is checked against an independent oracle: mpmath ``beta``,
``gammainc`` and ``quad`` at random parameter values, and SymPy where it
evaluates the same integral.
"""

from __future__ import annotations

import random

import alkahest as ak
import pytest
from alkahest import Domain

mpmath = pytest.importorskip("mpmath")
mpmath.mp.dps = 30

P_ = Domain.Positive


@pytest.fixture
def P():
    return ak.ExprPool()


def _ev(expr, env):
    return ak.eval_expr(expr, env)


def _conds(r):
    return r.verification["side_conditions"]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_beta_values_and_numerics(P):
    assert ak.simplify(ak.beta(P.integer(2), P.integer(3))).value == P.rational(1, 12)
    half = P.rational(1, 2)
    assert abs(_ev(ak.beta(half, half), {}) - float(mpmath.pi)) < 1e-14
    a = P.symbol("a")
    assert str(ak.beta(a, P.integer(1))) == str(a**-1)
    rng = random.Random(11)
    b = P.symbol("b")
    node = ak.beta(a, b)
    for _ in range(5):
        av, bv = rng.uniform(0.1, 4.0), rng.uniform(0.1, 4.0)
        assert abs(_ev(node, {a: av, b: bv}) - float(mpmath.beta(av, bv))) < 1e-12
    with pytest.raises(ak.SpecialFunctionError) as e:
        ak.beta(P.integer(0), a)
    assert e.value.code == "E-SPECFN-006"


def test_beta_derivative_matches_finite_difference(P):
    a, b = P.symbol("a"), P.symbol("b")
    d = ak.diff(ak.beta(a, b), a).value
    av, bv, h = 1.7, 0.8, 1e-6
    fd = (mpmath.beta(av + h, bv) - mpmath.beta(av - h, bv)) / (2 * h)
    assert abs(_ev(d, {a: av, b: bv}) - float(fd)) < 1e-7


def test_incomplete_gamma_against_mpmath(P):
    s, z = P.symbol("s"), P.symbol("z")
    up, lo = ak.uppergamma(s, z), ak.lowergamma(s, z)
    rng = random.Random(7)
    for _ in range(6):
        sv, zv = rng.uniform(0.2, 5.0), rng.uniform(0.05, 6.0)
        assert abs(_ev(up, {s: sv, z: zv}) - float(mpmath.gammainc(sv, zv))) < 1e-12
        assert abs(_ev(lo, {s: sv, z: zv}) - float(mpmath.gammainc(sv, 0, zv))) < 1e-12


def test_incomplete_gamma_special_values(P):
    z = P.symbol("z")
    assert str(ak.uppergamma(P.integer(1), z)) == str(ak.exp(-z))
    assert "erfc" in str(ak.uppergamma(P.rational(1, 2), z))
    assert "erf" in str(ak.lowergamma(P.rational(1, 2), z))
    for s in (P.integer(3), P.rational(5, 2), P.rational(1, 2)):
        sv = float(_ev(s, {}))
        for zv in (0.3, 1.7, 4.2):
            assert abs(_ev(ak.uppergamma(s, z), {z: zv}) - float(mpmath.gammainc(sv, zv))) < 1e-12
            assert (
                abs(_ev(ak.lowergamma(s, z), {z: zv}) - float(mpmath.gammainc(sv, 0, zv))) < 1e-12
            )
    # Γ(s, 0) = Γ(s)
    g = ak.uppergamma(P.rational(7, 3), P.integer(0))
    assert abs(_ev(g, {}) - float(mpmath.gamma(mpmath.mpf(7) / 3))) < 1e-13
    with pytest.raises(ak.SpecialFunctionError):
        ak.uppergamma(P.integer(0), P.integer(0))


def test_incomplete_gamma_derivatives(P):
    s, z = P.symbol("s"), P.symbol("z")
    d = ak.diff(ak.uppergamma(s, z), z).value
    assert abs(_ev(d, {s: 2.5, z: 1.3}) + 1.3**1.5 * float(mpmath.exp(-1.3))) < 1e-13
    d2 = ak.diff(ak.lowergamma(s, z), z).value
    assert abs(_ev(d2, {s: 2.5, z: 1.3}) - 1.3**1.5 * float(mpmath.exp(-1.3))) < 1e-13
    with pytest.raises(ak.DiffError) as e:
        ak.diff(ak.uppergamma(s, z), s)
    assert e.value.code == "E-DIFF-001"


def test_parse_and_latex_round_trip(P):
    s, z = P.symbol("s"), P.symbol("z")
    for e in (ak.uppergamma(s, z), ak.lowergamma(s, z), ak.beta(s, z)):
        assert str(ak.parse(str(e), P, {"s": s, "z": z})) == str(e)
    assert r"\Gamma" in ak.latex(ak.uppergamma(s, z))
    assert r"\gamma" in ak.latex(ak.lowergamma(s, z))
    assert r"\mathrm{B}" in ak.latex(ak.beta(s, z))


# ---------------------------------------------------------------------------
# Beta-family definite integrals
# ---------------------------------------------------------------------------


def test_beta_integral_symbolic_exponents_state_conditions(P):
    x, a, b = P.symbol("x"), P.symbol("a"), P.symbol("b")
    r = ak.integrate(x**a * (1 - x) ** b, x, P.integer(0), P.integer(1))
    conds = _conds(r)
    assert len(conds) == 2, conds  # a + 1 > 0 and b + 1 > 0
    rng = random.Random(3)
    for _ in range(5):
        av, bv = rng.uniform(-0.9, 3.0), rng.uniform(-0.9, 3.0)
        assert abs(_ev(r.value, {a: av, b: bv}) - float(mpmath.beta(av + 1, bv + 1))) < 1e-11


def test_beta_integral_positive_exponents_no_conditions(P):
    x = P.symbol("x")
    a, b = P.symbol("a", P_), P.symbol("b", P_)
    r = ak.integrate(x ** (a - 1) * (1 - x) ** (b - 1), x, P.integer(0), P.integer(1))
    # a > 0 is not a − 1 > −1 by syntax but by sign algebra: established.
    assert not _conds(r)
    assert abs(_ev(r.value, {a: 2.5, b: 0.7}) - float(mpmath.beta(2.5, 0.7))) < 1e-12


def test_beta_integral_exact_half_integers(P):
    x = P.symbol("x")
    r = ak.integrate(ak.sqrt(x) * (1 - x) ** P.rational(3, 2), x, P.integer(0), P.integer(1))
    # B(3/2, 5/2) = π/16
    assert abs(_ev(r.value, {}) - float(mpmath.pi / 16)) < 1e-14
    assert "pi" in str(r.value)


def test_beta_integral_on_scaled_interval(P):
    x = P.symbol("x")
    L = P.symbol("L", P_)
    f = x ** P.rational(1, 3) * (L - x) ** P.rational(2, 5)
    r = ak.integrate(f, x, P.integer(0), L)
    for Lv in (0.7, 2.3):
        want = mpmath.quad(lambda t, Lv=Lv: t ** (mpmath.mpf(1) / 3) * (Lv - t) ** 0.4, [0, Lv])
        assert abs(_ev(r.value, {L: Lv}) - float(want)) < 1e-11


def test_beta_integral_divergent_refused(P):
    x = P.symbol("x")
    with pytest.raises(ak.IntegrationError) as e:
        ak.integrate(x ** P.rational(-3, 2) * ak.sqrt(1 - x), x, P.integer(0), P.integer(1))
    assert "diverg" in str(e.value)


def test_beta_half_line(P):
    x = P.symbol("x")
    s, t = P.symbol("s", P_), P.symbol("t", P_)
    r = ak.integrate(x ** (s - 1) / (1 + x) ** (s + t), x, P.integer(0), P.pos_infinity())
    assert not _conds(r)
    for sv, tv in ((0.4, 1.3), (2.2, 0.6)):
        assert abs(_ev(r.value, {s: sv, t: tv}) - float(mpmath.beta(sv, tv))) < 1e-11
    # Scaled: ∫₀^∞ √x/(2 + 3x)² dx
    r2 = ak.integrate(ak.sqrt(x) / (2 + 3 * x) ** 2, x, P.integer(0), P.pos_infinity())
    want = mpmath.quad(lambda u: mpmath.sqrt(u) / (2 + 3 * u) ** 2, [0, mpmath.inf])
    assert abs(_ev(r2.value, {}) - float(want)) < 1e-12


def test_wallis_beta(P):
    x = P.symbol("x")
    p_, q_ = P.symbol("p"), P.symbol("q")
    half_pi = P.rational(1, 2) * ak.parse("pi", P)
    r = ak.integrate(ak.sin(x) ** p_ * ak.cos(x) ** q_, x, P.integer(0), half_pi)
    assert len(_conds(r)) == 2
    for pv, qv in ((0.5, 1.5), (2.3, -0.4)):
        want = mpmath.quad(
            lambda u, pv=pv, qv=qv: mpmath.sin(u) ** pv * mpmath.cos(u) ** qv, [0, mpmath.pi / 2]
        )
        assert abs(_ev(r.value, {p_: pv, q_: qv}) - float(want)) < 1e-10
    r2 = ak.integrate(ak.sqrt(ak.sin(x)), x, P.integer(0), half_pi)
    want2 = mpmath.quad(lambda u: mpmath.sqrt(mpmath.sin(u)), [0, mpmath.pi / 2])
    assert abs(_ev(r2.value, {}) - float(want2)) < 1e-12


# ---------------------------------------------------------------------------
# Incomplete gamma from a nonzero bound
# ---------------------------------------------------------------------------


def test_upper_incomplete_gamma_integral(P):
    x = P.symbol("x")
    s = P.symbol("s")
    a = P.symbol("a", P_)
    r = ak.integrate(x ** (s - 1) * ak.exp(-x), x, a, P.pos_infinity())
    assert "uppergamma" in str(r.value)
    assert not _conds(r)
    rng = random.Random(5)
    for _ in range(5):
        sv, av = rng.uniform(-1.5, 4.0), rng.uniform(0.2, 3.0)
        assert abs(_ev(r.value, {s: sv, a: av}) - float(mpmath.gammainc(sv, av))) < 1e-11


def test_rate_and_finite_interval(P):
    x = P.symbol("x")
    b = P.symbol("b", P_)
    f = x ** P.rational(3, 2) * ak.exp(-b * x)
    r = ak.integrate(f, x, P.integer(1), P.integer(3))
    for bv in (0.5, 2.0):
        want = mpmath.quad(lambda u, bv=bv: u**1.5 * mpmath.exp(-bv * u), [1, 3])
        assert abs(_ev(r.value, {b: bv}) - float(want)) < 1e-12
    # Lower incomplete gamma: ∫₀^c x^{s-1} e^{-x} with s > 0.
    c = P.symbol("c", P_)
    r2 = ak.integrate(x ** P.rational(-1, 3) * ak.exp(-x), x, P.integer(0), c)
    assert "lowergamma" in str(r2.value)
    assert abs(_ev(r2.value, {c: 1.4}) - float(mpmath.gammainc(mpmath.mpf(2) / 3, 0, 1.4))) < 1e-12


def test_gaussian_tail_is_erfc(P):
    x = P.symbol("x")
    a, b = P.symbol("a", P_), P.symbol("b", P_)
    r = ak.integrate(ak.exp(-b * x**2), x, a, P.pos_infinity())
    assert "erfc" in str(r.value)
    for av, bv in ((0.3, 1.0), (1.2, 0.4)):
        want = mpmath.quad(lambda u, bv=bv: mpmath.exp(-bv * u**2), [av, mpmath.inf])
        assert abs(_ev(r.value, {a: av, b: bv}) - float(want)) < 1e-12
    # x²·e^{−x²} on [1, ∞): Γ(3/2, 1)/2, half-integer → erfc + elementary.
    r2 = ak.integrate(x**2 * ak.exp(-(x**2)), x, P.integer(1), P.pos_infinity())
    want2 = mpmath.quad(lambda u: u**2 * mpmath.exp(-(u**2)), [1, mpmath.inf])
    assert abs(_ev(r2.value, {}) - float(want2)) < 1e-13
    assert "uppergamma" not in str(r2.value)


# ---------------------------------------------------------------------------
# Feynman-parameter integrals
# ---------------------------------------------------------------------------


def _feynman(P):
    x = P.symbol("x")
    m, p = P.symbol("m", P_), P.symbol("p", P_)
    return x, m, p, m**2 - x * (1 - x) * p**2


def _at(P, expr, env):
    """Substitute exact rationals and evaluate (the Piecewise resolves)."""
    sub = {k: P.rational(v.numerator, v.denominator) for k, v in env.items()}
    val = ak.simplify(ak.subs(expr, sub)).value
    return complex(ak.evaluate(val, {}, mode="complex").value)


def test_feynman_log_three_regions(P):
    from fractions import Fraction as F

    x, m, p, Q = _feynman(P)
    r = ak.integrate(ak.log(Q), x, P.integer(0), P.integer(1))
    assert "Piecewise" in str(r.value)
    # p² < 4m² (atan branch) and p² = 4m² (threshold): real values.
    for mv, pv in ((F(1), F(1, 2)), (F(13, 10), F(21, 10)), (F(7, 10), F(7, 5)), (F(1, 2), F(1))):
        mf, pf = float(mv), float(pv)
        want = mpmath.quad(
            lambda u, mf=mf, pf=pf: mpmath.log(mf**2 - u * (1 - u) * pf**2), [0, 0.5, 1]
        )
        if 4 * mv**2 == pv**2:
            # At the threshold f64 cannot decide κ = 0; the value is continuous.
            near = _ev(r.value, {m: mf, p: pf * (1 - 1e-12)})
            assert abs(near - (2 * float(mpmath.log(mf)) - 2)) < 1e-5, (mv, pv)
            continue
        assert abs(_ev(r.value, {m: mf, p: pf}) - float(want)) < 1e-10, (mv, pv)
        assert abs(_at(P, r.value, {m: mv, p: pv}) - complex(want)) < 1e-10, (mv, pv)
    # p² > 4m²: the principal-branch log has imaginary part π·β, β = √(1 − 4m²/p²).
    for mv, pv in ((F(1, 2), F(3, 2)), (F(3, 10), F(2))):
        mf, pf = float(mv), float(pv)
        beta = mpmath.sqrt(1 - 4 * mf**2 / pf**2)
        x1, x2 = (1 - beta) / 2, (1 + beta) / 2

        def f(u, mf=mf, pf=pf):
            return mpmath.log(mpmath.mpc(mf**2 - u * (1 - u) * pf**2, 0))

        want = mpmath.quad(f, [0, x1, x2, 1])
        got = _at(P, r.value, {m: mv, p: pv})
        assert abs(got - complex(want)) < 1e-9, (mv, pv, got, want)
        assert abs(got.imag - float(mpmath.pi * beta)) < 1e-12


def test_report_q2_unknown_sign_momentum(P):
    """Report 10-9 Q2 verbatim: ``p2`` is a plain symbol (either sign)."""
    from fractions import Fraction as F

    x, p2 = P.symbol("x"), P.symbol("p2")
    m = P.symbol("m", P_)
    r = ak.integrate(ak.log(m**2 - x * (1 - x) * p2), x, P.integer(0), P.integer(1))
    for mv, pv in ((F(1), F(-3)), (F(1), F(0)), (F(1), F(2)), (F(1, 2), F(9, 4))):
        mf, pf = float(mv), float(pv)

        def f(u, mf=mf, pf=pf):
            return mpmath.log(mpmath.mpc(mf**2 - u * (1 - u) * pf, 0))

        pts = [0, 0.5, 1]
        if pf > 4 * mf**2:  # the roots of Q, where log Q is singular
            b = mpmath.sqrt(1 - 4 * mf**2 / pf)
            pts = [0, (1 - b) / 2, 0.5, (1 + b) / 2, 1]
        want = mpmath.quad(f, pts)
        got = _at(P, r.value, {m: mv, p2: pv})
        assert abs(got - complex(want)) < 1e-8, (mv, pv, got, want)
    # The reciprocal: finite exactly when Q(½) = m² − p2/4 > 0, which is stated.
    r2 = ak.integrate(1 / (m**2 - x * (1 - x) * p2), x, P.integer(0), P.integer(1))
    assert _conds(r2)
    for mf, pf in ((1.0, -3.0), (1.0, 2.0), (0.8, 0.0)):
        want = mpmath.quad(lambda u, mf=mf, pf=pf: 1 / (mf**2 - u * (1 - u) * pf), [0, 1])
        assert abs(_ev(r2.value, {m: mf, p2: pf}) - float(want)) < 1e-10


def test_quadratic_log_double_root_branch(P):
    # κ = 0 with the root outside [1, 2]: log((x − 1/2)²).
    x = P.symbol("x")
    r = ak.integrate(ak.log(x**2 - x + P.rational(1, 4)), x, P.integer(1), P.integer(2))
    want = mpmath.quad(lambda u: mpmath.log((u - 0.5) ** 2), [1, 2])
    assert abs(_ev(r.value, {}) - float(want)) < 1e-13


def test_feynman_log_spacelike_is_real(P):
    # p² → −k² (spacelike): Q = m² + x(1−x)k² > 0 on [0, 1]; the log branch.
    x = P.symbol("x")
    m, k = P.symbol("m", P_), P.symbol("k", P_)
    r = ak.integrate(ak.log(m**2 + x * (1 - x) * k**2), x, P.integer(0), P.integer(1))
    for mv, kv in ((1.0, 0.5), (0.4, 3.0)):
        want = mpmath.quad(lambda u, mv=mv, kv=kv: mpmath.log(mv**2 + u * (1 - u) * kv**2), [0, 1])
        assert abs(_ev(r.value, {m: mv, k: kv}) - float(want)) < 1e-11


@pytest.mark.parametrize("n", [1, 2, 3])
def test_feynman_reciprocal_powers(P, n):
    x, m, p, Q = _feynman(P)
    r = ak.integrate(Q ** (-n), x, P.integer(0), P.integer(1))
    assert _conds(r), "the threshold p² < 4m² must be stated"
    for mv, pv in ((1.0, 0.5), (1.3, 2.1), (0.9, 1.7)):
        want = mpmath.quad(lambda u, mv=mv, pv=pv: (mv**2 - u * (1 - u) * pv**2) ** (-n), [0, 1])
        assert abs(_ev(r.value, {m: mv, p: pv}) - float(want)) < 1e-10 * (1 + abs(want))


def test_feynman_reciprocal_above_threshold_diverges(P):
    x = P.symbol("x")
    Q = 1 - x * (1 - x) * 9  # m = 1, p = 3: Q vanishes inside (0, 1)
    with pytest.raises(ak.IntegrationError):
        ak.integrate(1 / Q, x, P.integer(0), P.integer(1))


# ---------------------------------------------------------------------------
# Display of definite results
# ---------------------------------------------------------------------------


def test_convolution_result_is_readable(P):
    tau, t = P.symbol("tau"), P.symbol("t")
    a, b = P.symbol("a", P_), P.symbol("b", P_)
    f = ak.exp(-a * tau) * ak.exp(-b * (t - tau))
    r = ak.integrate(f, tau, P.integer(0), t)
    s = str(r.value)
    # Each exponent is a monomial (e^{−at}, e^{−bt}), not e^{−bt + t(b−a)}.
    assert "exp(((" not in s, s
    av, bv, tv = 0.7, 1.9, 1.3
    want = mpmath.quad(lambda u: mpmath.e ** (-av * u - bv * (tv - u)), [0, tv])
    assert abs(_ev(r.value, {a: av, b: bv, t: tv}) - float(want)) < 1e-12


# ---------------------------------------------------------------------------
# SymPy agreement
# ---------------------------------------------------------------------------


def test_against_sympy(P):
    sp = pytest.importorskip("sympy")
    X = sp.Symbol("x", positive=True)
    x = P.symbol("x")
    cases = [
        (ak.sqrt(x) * (1 - x) ** P.rational(3, 2), sp.sqrt(X) * (1 - X) ** sp.Rational(3, 2), 0, 1),
        (x ** P.rational(3, 2) * ak.exp(-x), X ** sp.Rational(3, 2) * sp.exp(-X), 2, sp.oo),
        (ak.exp(-(x**2)), sp.exp(-(X**2)), 1, sp.oo),
    ]
    for ours, theirs, lo, hi in cases:
        hi_ours = P.pos_infinity() if hi is sp.oo else P.integer(hi)
        r = ak.integrate(ours, x, P.integer(lo), hi_ours)
        want = sp.N(sp.integrate(theirs, (X, lo, hi)), 20)
        assert abs(_ev(r.value, {}) - float(want)) < 1e-12
