"""Γ/ψ special values, Laurent series at the poles, hypergeometric pFq (G11c).

The report's repro was ``series(gamma(eps), eps, 0, 2)`` → ``E-SERIES-004``,
which made dimensional-regularisation ε-expansions impossible; ``ψ(1)`` and
``simplify(digamma(n))`` folded the Euler–Mascheroni constant into the float
``-0.5772…``; ``trigamma(1)`` stayed unevaluated.

Oracles:

* **SymPy** — ``series(gamma(x), x, 0, n)`` and at the negative integers
  (exact, with ``EulerGamma``), ``m**(-2*eps)*gamma(eps)``, ``polygamma`` at
  rationals, ``hyperexpand``;
* **mpmath** — ``gamma``, ``digamma``, ``polygamma``, ``euler``, ``catalan``,
  ``hyp0f1/hyp1f1/hyp2f1/hyper`` at random points (``f64``) and at 200 bits
  (balls).
"""

from __future__ import annotations

import random
import re
from fractions import Fraction as F

import alkahest as ak
import pytest

mp = pytest.importorskip("mpmath")
sp = pytest.importorskip("sympy")


@pytest.fixture
def pool():
    return ak.ExprPool()


def to_sympy(e):
    """``crosscheck.to_sympy`` with ``pi`` read as ``sympy.pi`` and the
    big-O remainder dropped."""
    from alkahest.crosscheck import to_sympy as _to_sympy

    return _to_sympy(e).subs(sp.Symbol("pi"), sp.pi)


def has_float(e) -> bool:
    """Does the printed expression carry a float literal?"""
    return re.search(r"\d\.\d|\de-\d", str(e)) is not None


def series_terms(e, var, point, order):
    """The SymPy polynomial of ``ak.series`` without its ``O(...)`` term."""
    s = ak.series(e, var, point, order)
    assert not has_float(s), str(s)
    text = str(s).replace("^", "**")
    names = {"EulerGamma": sp.EulerGamma, "pi": sp.pi, "zeta": sp.zeta, "log": sp.log}
    names["O"] = lambda *a: 0
    names[str(var)] = sp.Symbol(str(var))
    for sym in ("m",):
        names[sym] = sp.Symbol(sym, positive=True)
    return sp.sympify(text, locals=names)


# ---------------------------------------------------------------------------
# The repro and the Laurent series
# ---------------------------------------------------------------------------


def test_report_repro_gamma_eps_expands(pool):
    eps = pool.symbol("eps")
    got = series_terms(ak.gamma(eps), eps, pool.integer(0), 2)
    e = sp.Symbol("eps")
    want = sp.series(sp.gamma(e), e, 0, 2).removeO()
    assert sp.simplify(got - want) == 0, (got, want)
    assert "EulerGamma" in str(ak.series(ak.gamma(eps), eps, pool.integer(0), 2))


@pytest.mark.parametrize("order", [1, 2, 4])
def test_gamma_laurent_at_zero_matches_sympy_exactly(pool, order):
    x = pool.symbol("x")
    got = series_terms(ak.gamma(x), x, pool.integer(0), order)
    s = sp.Symbol("x")
    want = sp.series(sp.gamma(s), s, 0, order).removeO()
    assert sp.expand(sp.simplify(got - want)) == 0, (got, want)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_gamma_laurent_at_negative_integers(pool, k):
    """Simple poles with residue (−1)^k/k!, and the exact constant terms."""
    x = pool.symbol("x")
    got = series_terms(ak.gamma(x), x, pool.integer(-k), 2)
    s = sp.Symbol("x")
    want = sp.series(sp.gamma(s), s, -k, 2).removeO()
    assert sp.expand(sp.simplify(got - want)) == 0, (got, want)


def test_digamma_and_trigamma_at_poles(pool):
    x = pool.symbol("x")
    s = sp.Symbol("x")
    # SymPy cannot expand ψ at its pole directly (PoleError); it can expand
    # the shifted, analytic form ψ(s) = ψ(s+1) − 1/s.
    got = series_terms(ak.digamma(x), x, pool.integer(0), 3)
    want = (sp.series(sp.digamma(s + 1), s, 0, 3) - 1 / s).removeO()
    assert sp.expand(sp.simplify(got - want)) == 0, (got, want)
    # ψ₁(s) = ψ₁(s + 2) + 1/s² + 1/(s + 1)², about s = −1 (h = s + 1).
    h = sp.Symbol("h")
    got = series_terms(ak.trigamma(x), x, pool.integer(-1), 1).subs(s, h - 1)
    want = (sp.series(sp.trigamma(h + 1) + 1 / (h - 1) ** 2, h, 0, 1) + 1 / h**2).removeO()
    assert sp.expand(sp.simplify(got - want)) == 0, (got, want)


def test_dimensional_regularisation(pool):
    """``m^{-2ε} Γ(ε) = 1/ε − γ − 2 log m + O(ε)``, exactly."""
    eps = pool.symbol("eps")
    m = pool.symbol("m")
    got = series_terms(m ** (-2 * eps) * ak.gamma(eps), eps, pool.integer(0), 2)
    e = sp.Symbol("eps")
    ms = sp.Symbol("m", positive=True)
    want = sp.series(ms ** (-2 * e) * sp.gamma(e), e, 0, 2).removeO()
    assert sp.simplify(sp.expand_log(got - want, force=True)) == 0, (got, want)


def test_gamma_one_plus_eps_has_no_float(pool):
    eps = pool.symbol("eps")
    s = ak.series(ak.gamma(eps + 1), eps, pool.integer(0), 4)
    assert not has_float(s)
    assert "zeta(3)" in str(s)


# ---------------------------------------------------------------------------
# Exact special values
# ---------------------------------------------------------------------------


def _sympy_value(e):
    return to_sympy(ak.simplify(e).value)


@pytest.mark.parametrize(
    "q",
    [
        F(1),
        F(2),
        F(7),
        F(1, 2),
        F(5, 2),
        F(-3, 2),
        F(1, 3),
        F(2, 3),
        F(1, 4),
        F(3, 4),
        F(1, 6),
        F(5, 6),
        F(-7, 4),
        F(10, 3),
    ],
)
def test_digamma_exact_values(pool, q):
    v = ak.simplify(ak.digamma(pool.rational(q.numerator, q.denominator))).value
    assert not has_float(v), str(v)
    assert "digamma" not in str(v)
    mp.mp.dps = 50
    got = sp.N(to_sympy(v), 45)
    want = mp.digamma(mp.mpf(q.numerator) / q.denominator)
    assert abs(mp.mpf(str(got)) - want) < mp.mpf(10) ** -40, (q, v)


def test_digamma_at_integers_is_exact_in_euler_gamma(pool):
    """The float-in-exact-path defect: ψ(n) folded γ to −0.5772…"""
    for n in range(1, 30):
        v = ak.simplify(ak.digamma(pool.integer(n))).value
        assert not has_float(v), str(v)
        assert sp.simplify(to_sympy(v) - sp.digamma(n)) == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("q", [F(1), F(3), F(1, 2), F(7, 2), F(-5, 2)])
def test_polygamma_exact_values_match_sympy(pool, n, q):
    v = ak.polygamma(n, pool.rational(q.numerator, q.denominator))
    assert not has_float(v), str(v)
    assert "polygamma" not in str(v)
    assert "trigamma" not in str(v)
    if q.denominator == 1 or q == F(1, 2):
        # SymPy evaluates these exactly.
        want = sp.polygamma(n, sp.Rational(q.numerator, q.denominator))
        assert sp.simplify(to_sympy(v) - sp.expand_func(want)) == 0, (n, q, v)
    # SymPy leaves ψ⁽ⁿ⁾(7/2) unevaluated: an exact closed form must still
    # agree with mpmath to 45 digits.
    mp.mp.dps = 50
    got = mp.mpf(str(sp.N(to_sympy(v), 50)))
    want = mp.polygamma(n, mp.mpf(q.numerator) / q.denominator)
    assert abs(got - want) < mp.mpf(10) ** -45 * max(1, abs(want)), (n, q, v)


def test_trigamma_quarter_uses_catalan(pool):
    v = ak.simplify(ak.trigamma(pool.rational(1, 4))).value
    assert "Catalan" in str(v)
    mp.mp.dps = 30
    assert abs(ak.eval_expr(v, {}) - float(mp.psi(1, 0.25))) < 1e-13


def test_report_values(pool):
    one = pool.integer(1)
    assert to_sympy(ak.simplify(ak.digamma(one)).value) == -sp.EulerGamma
    assert sp.simplify(to_sympy(ak.simplify(ak.trigamma(one)).value) - sp.pi**2 / 6) == 0
    assert sp.simplify(to_sympy(ak.polygamma(2, one)) + 2 * sp.zeta(3)) == 0
    assert str(ak.simplify(ak.gamma(pool.integer(25))).value) == str(sp.factorial(24))


def test_polygamma_spellings_and_refusals(pool):
    x = pool.symbol("x")
    assert str(ak.polygamma(0, x)) == "digamma(x)"
    assert str(ak.polygamma(1, x)) == "trigamma(x)"
    assert str(ak.polygamma(4, x)) == "polygamma(4, x)"
    for bad in (F(1, 2), -1):
        with pytest.raises(ak.SpecialFunctionError) as info:
            ak.polygamma(bad, x)
        assert info.value.code == "E-SPECFN-008"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.polygamma(2, pool.integer(-3))
    assert info.value.code == "E-SPECFN-006"


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def test_named_constants_numerics_and_printing(pool):
    g, c, a = ak.euler_gamma(pool), ak.catalan(pool), ak.glaisher(pool)
    assert str(g) == "EulerGamma"
    assert ak.latex(g) == r"\gamma"
    assert ak.parse("EulerGamma + Catalan", pool) == g + c
    mp.mp.prec = 400
    assert ak.eval_expr(g, {}) == float(mp.euler)
    assert ak.eval_expr(c, {}) == float(mp.catalan)
    assert abs(ak.eval_expr(a, {}) - float(mp.glaisher)) < 1e-15
    for const, want in ((g, mp.euler), (c, mp.catalan)):
        b = ak.interval_eval(const, {}, 200)
        assert b.rad < 1e-55
        assert b.lo <= float(want) <= b.hi
    x = pool.symbol("x")
    f = ak.compile_expr(g * x, [x])
    assert abs(f([2.0]) - 2 * float(mp.euler)) < 1e-15
    assert "0.5772156649015" in ak.emit_c_expr(g * x, [x])


def test_polygamma_f64_matches_mpmath(pool):
    mp.mp.dps = 40
    x = pool.symbol("x")
    n = pool.symbol("n")
    e = pool.func("polygamma", [n, x])
    rng = random.Random(11)
    for _ in range(40):
        k = rng.randint(0, 6)
        xv = rng.uniform(-6.0, 8.0)
        if abs(xv - round(xv)) < 1e-3:
            continue
        got = ak.eval_expr(e, {n: float(k), x: xv})
        want = float(mp.polygamma(k, xv))
        assert abs(got - want) <= 1e-13 * abs(want) + 1e-300, (k, xv, got, want)


def test_polygamma_balls_at_200_bits(pool):
    mp.mp.prec = 400
    x = pool.symbol("x")
    n = pool.symbol("n")
    e = pool.func("polygamma", [n, x])
    for k, xv in [(2, 0.75), (3, 2.5), (1, -1.25), (5, 4.0)]:
        b = ak.interval_eval(e, {n: ak.ArbBall(float(k), 0.0, 200), x: ak.ArbBall(xv, 0.0, 200)})
        want = mp.polygamma(k, xv)
        assert b.rad < 1e-50 * max(1.0, abs(float(want)))
        assert abs(b.mid - float(want)) <= 2.3e-16 * abs(float(want))


HYPER_CASES = [
    # (a, b, z)
    ([], [F(5, 2)], 1.3),
    ([], [F(1, 3)], -2.2),
    ([F(1, 2)], [F(3, 2)], -0.8),
    ([F(2, 3)], [F(9, 4)], 3.5),
    ([F(1, 2), 3], [F(7, 3)], 0.3),
    ([F(1, 3), F(2, 5)], [F(3, 2)], -4.0),
    ([1, 1, 1], [2, 2], 0.5),
    ([F(1, 2), F(1, 3), F(1, 4)], [F(5, 3), F(7, 4)], -0.9),
    ([F(3, 2)], [], 0.25),
]


@pytest.mark.parametrize(("a", "b", "zv"), HYPER_CASES)
def test_hyper_f64_matches_mpmath(pool, a, b, zv):
    mp.mp.dps = 40
    z = pool.symbol("z")
    e = ak.hyper(a, b, z)
    got = ak.eval_expr(e, {z: zv})
    want = float(
        mp.hyper(
            [mp.mpf(v.numerator) / v.denominator for v in map(F, a)],
            [mp.mpf(v.numerator) / v.denominator for v in map(F, b)],
            zv,
        )
    )
    assert abs(got - want) <= 1e-13 * abs(want), (a, b, zv, got, want)


@pytest.mark.parametrize(("a", "b", "zv"), HYPER_CASES[:6])
def test_hyper_balls_at_200_bits(pool, a, b, zv):
    mp.mp.prec = 400
    z = pool.symbol("z")
    e = ak.hyper(a, b, z)
    bb = ak.interval_eval(e, {z: ak.ArbBall(zv, 0.0, 200)}, 200)
    want = mp.hyper(
        [mp.mpf(v.numerator) / v.denominator for v in map(F, a)],
        [mp.mpf(v.numerator) / v.denominator for v in map(F, b)],
        zv,
    )
    assert bb.rad < 1e-50 * max(1.0, abs(float(want)))
    assert abs(bb.mid - float(want)) <= 2.3e-16 * abs(float(want))


def test_hyper_past_the_branch_cut_declines(pool):
    z = pool.symbol("z")
    with pytest.raises((ak.AlkahestError, ValueError)):
        ak.eval_expr(ak.hyper([F(1, 2), F(1, 2)], [F(3, 2)], z), {z: 2.0})


def test_hyper_derivative_by_finite_differences(pool):
    z = pool.symbol("z")
    e = ak.hyper([F(1, 3), F(2, 5)], [F(3, 2)], z)
    d = ak.diff(e, z).value
    assert "hyper(2, 1, 4/3, 7/5, 5/2, z)" in str(d)
    h = 1e-6
    fd = (ak.eval_expr(e, {z: 0.4 + h}) - ak.eval_expr(e, {z: 0.4 - h})) / (2 * h)
    assert abs(ak.eval_expr(d, {z: 0.4}) - fd) < 1e-7
    a = pool.symbol("a")
    with pytest.raises(ak.DiffError):
        ak.diff(ak.hyper([a], [F(3, 2)], z), a)


# ---------------------------------------------------------------------------
# Structure and expand_hyper
# ---------------------------------------------------------------------------


def test_hyper_node_prints_parses_and_cancels(pool):
    z = pool.symbol("z")
    a = pool.symbol("a")
    e = ak.hyper([a, 3], [F(7, 3)], z)
    assert str(e) == "hyper(2, 1, a, 3, 7/3, z)"
    assert ak.latex(e) == r"{}_{2}F_{1}\!\left(a, 3; \frac{7}{3}; z\right)"
    assert ak.parse(str(e), pool) is not None
    assert str(ak.hyper([a, 2], [a], z)) == "hyper(1, 0, 2, z)"
    assert str(ak.hyper([0, a], [2], z)) == "1"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.hyper([a], [-2], z)
    assert info.value.code == "E-SPECFN-006"


EXPAND_CASES = [
    ([], [], "z"),
    ([F(3, 2)], [], "z"),
    ([1, 1], [2], "z"),
    ([F(1, 2), 1], [F(3, 2)], "z"),
    ([F(1, 2), F(1, 2)], [F(3, 2)], "z"),
    ([1], [2], "z"),
    ([F(1, 2)], [F(3, 2)], "-z"),
    ([], [F(1, 2)], "z"),
    ([], [F(3, 2)], "z"),
    ([-3, F(1, 3)], [F(5, 2)], "z"),
    ([-2, F(1, 2), 4], [F(2, 3), F(7, 2)], "z"),
]


@pytest.mark.parametrize(("a", "b", "arg"), EXPAND_CASES)
def test_expand_hyper_matches_hyperexpand(pool, a, b, arg):
    z = pool.symbol("z")
    zz = -z if arg == "-z" else z
    got = ak.expand_hyper(ak.hyper(a, b, zz))
    assert "hyper" not in str(got), str(got)
    s = sp.Symbol("z")
    want = sp.hyperexpand(
        sp.hyper(
            [sp.Rational(v.numerator, v.denominator) for v in map(F, a)],
            [sp.Rational(v.numerator, v.denominator) for v in map(F, b)],
            -s if arg == "-z" else s,
        )
    )
    for zv in (0.15, 0.4, 0.7):
        g = ak.eval_expr(got, {z: zv})
        w = complex(sp.N(want.subs(s, zv), 30))
        assert abs(w.imag) < 1e-20
        assert abs(g - w.real) < 1e-12 * (1 + abs(w.real)), (a, b, zv, g, w)


def test_gauss_summation(pool):
    one = pool.integer(1)
    e = ak.expand_hyper(ak.hyper([F(1, 2), F(1, 3)], [2], one))
    assert "hyper" not in str(e)
    want = (
        sp.gamma(2)
        * sp.gamma(sp.Rational(7, 6))
        / (sp.gamma(sp.Rational(3, 2)) * sp.gamma(sp.Rational(5, 3)))
    )
    assert sp.simplify(to_sympy(e) - want) == 0
    # c − a − b ≤ 0: the series diverges at 1 and the node is kept.
    assert "hyper" in str(ak.expand_hyper(ak.hyper([1, 1], [2], one)))


def test_terminating_series_cap(pool):
    z = pool.symbol("z")
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.expand_hyper(ak.hyper([-1000], [3], z))
    assert info.value.code == "E-SPECFN-004"


def test_exported():
    for name in ("euler_gamma", "catalan", "glaisher", "polygamma", "hyper", "expand_hyper"):
        assert name in ak.__all__, name
        assert hasattr(ak, name), name
