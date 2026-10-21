"""Hypergeometric follow-ups (F26).

G11c left complex ``hyper`` values refused, no 2F1 transformations, and
``digamma`` / ``polygamma`` exact only at integer shifts of 1/2, 1/3, 1/4,
1/6.  Now:

* ``evaluate(..., mode="complex")`` gives ``pFq`` at complex points, with
  ``2F1`` continued analytically past ``|z| = 1`` (on the cut ``[1, oo)`` the
  limit from below, mpmath's convention);
* ``hyper_transform`` applies Euler, both Pfaff, the four two-term connection
  formulas, the quadratic transformations and Kummer's;
* ``expand_hyper`` reaches new closed forms (the ``a, a + 1/2`` families,
  ``1F1(a; 2a; z)`` as Bessel ``I``, and through Euler/Pfaff/Kummer);
* ``digamma(p/q)`` is exact for every ``q <= 100`` (Gauss's digamma theorem)
  and ``polygamma(n, p/q)`` for every rational (classical forms at
  denominators 3, 4, 6 for even ``n``, the exact Hurwitz value otherwise).

Oracles: mpmath ``hyp2f1`` / ``hyp1f1`` / ``hyper`` / ``digamma`` /
``polygamma`` at 30 digits, SymPy ``hyperexpand``.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import alkahest as ak
import pytest

mp = pytest.importorskip("mpmath")
sp = pytest.importorskip("sympy")


@pytest.fixture
def pool():
    return ak.ExprPool()


def cval(expr, env):
    r = ak.evaluate(expr, env, mode="complex")
    assert r.status == "ok", r.reason
    return complex(r.value)


def mpf(q):
    q = F(q)
    return mp.mpf(q.numerator) / q.denominator


def rand_param(rng, lo, hi):
    d = rng.randint(3, 11)
    return F(round(rng.uniform(lo, hi) * d), d)


# ---------------------------------------------------------------------------
# Complex values
# ---------------------------------------------------------------------------


def test_complex_hyper_repro(pool):
    """The report's gap: a complex hyper value used to be refused."""
    z = pool.symbol("z")
    e = ak.hyper([1, 1], [2], z)
    got = cval(e, {z: 3 + 2j})
    want = complex(mp.hyp2f1(1, 1, 2, mp.mpc(3, 2)))
    assert abs(got - want) < 1e-14 * abs(want)


def test_complex_2f1_matches_mpmath_inside_and_outside_the_disk(pool):
    mp.mp.dps = 30
    z = pool.symbol("z")
    rng = random.Random("f26-2f1")
    for _ in range(30):
        a, b = rand_param(rng, -2.5, 2.5), rand_param(rng, -2.5, 2.5)
        c = rand_param(rng, 0.2, 3.0)
        e = ak.hyper([a, b], [c], z)
        zv = complex(rng.uniform(-4, 4), rng.uniform(-4, 4))
        if abs(zv.imag) < 0.05:
            continue
        got = cval(e, {z: zv})
        want = complex(mp.hyp2f1(mpf(a), mpf(b), mpf(c), mp.mpc(zv)))
        assert abs(got - want) <= 1e-12 * (1 + abs(want)), (a, b, c, zv, got, want)


def test_complex_2f1_on_the_cut_is_the_limit_from_below(pool):
    mp.mp.dps = 30
    z = pool.symbol("z")
    for a, b, c, x in [(1, 1, 2, 2.0), (F(1, 3), F(1, 2), F(5, 4), 3.5)]:
        got = cval(ak.hyper([a, b], [c], z), {z: complex(x, 0.0)})
        want = complex(mp.hyp2f1(mpf(a), mpf(b), mpf(c), mp.mpf(x)))
        below = complex(mp.hyp2f1(mpf(a), mpf(b), mpf(c), mp.mpc(x, -1e-25)))
        assert abs(got - want) < 1e-13 * abs(want), (got, want)
        assert abs(got - below) < 1e-13 * abs(want), (got, below)


def test_complex_1f1_0f1_and_pfq_match_mpmath(pool):
    mp.mp.dps = 30
    z = pool.symbol("z")
    rng = random.Random("f26-pfq")
    for _ in range(20):
        a = rand_param(rng, -2.5, 2.5)
        b = rand_param(rng, 0.2, 3.0)
        zv = complex(rng.uniform(-6, 6), rng.uniform(-6, 6))
        got = cval(ak.hyper([a], [b], z), {z: zv})
        want = complex(mp.hyp1f1(mpf(a), mpf(b), mp.mpc(zv)))
        assert abs(got - want) <= 1e-12 * (1 + abs(want)), (a, b, zv)
        got = cval(ak.hyper([], [b], z), {z: zv})
        want = complex(mp.hyp0f1(mpf(b), mp.mpc(zv)))
        assert abs(got - want) <= 1e-12 * (1 + abs(want)), (b, zv)
        # 3F2 inside the unit disk
        a2 = [a, rand_param(rng, 0.1, 2), rand_param(rng, 0.1, 2)]
        b2 = [b, rand_param(rng, 0.3, 3)]
        w = complex(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6))
        got = cval(ak.hyper(a2, b2, z), {z: w})
        want = complex(mp.hyper([mpf(x) for x in a2], [mpf(x) for x in b2], mp.mpc(w)))
        assert abs(got - want) <= 1e-12 * (1 + abs(want)), (a2, b2, w)


def test_3f2_outside_the_disk_declines_rather_than_guesses(pool):
    z = pool.symbol("z")
    r = ak.evaluate(ak.hyper([1, 1, 1], [2, 2], z), {z: 3 + 1j}, mode="complex")
    assert r.status != "ok"


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

LINEAR = ["euler", "pfaff", "pfaff_b", "z/(z-1)", "1-z", "1/z", "1/(1-z)", "1-1/z"]


@pytest.mark.parametrize("kind", LINEAR)
def test_linear_transformations_hold_at_random_parameters(pool, kind):
    mp.mp.dps = 30
    z = pool.symbol("z")
    rng = random.Random("f26-" + kind)
    done = 0
    while done < 12:
        a, b = rand_param(rng, -2.5, 2.5), rand_param(rng, -2.5, 2.5)
        c = rand_param(rng, 0.2, 3.0)
        if (a - b).denominator == 1 or (c - a - b).denominator == 1:
            continue
        if c in (a, b) or 0 in (a, b):
            continue  # the node cancels to 1F0 or 1
        rhs = ak.hyper_transform(ak.hyper([a, b], [c], z), kind)
        zv = complex(rng.uniform(-3, 3), rng.choice([-1, 1]) * rng.uniform(0.2, 3))
        got = cval(rhs, {z: zv})
        want = complex(mp.hyp2f1(mpf(a), mpf(b), mpf(c), mp.mpc(zv)))
        assert abs(got - want) <= 1e-9 * (1 + abs(want)), (kind, a, b, c, zv, str(rhs))
        done += 1


def test_quadratic_transformations(pool):
    mp.mp.dps = 30
    z = pool.symbol("z")
    rng = random.Random("f26-quad")
    for _ in range(10):
        a, b = rand_param(rng, -2, 2), rand_param(rng, 0.2, 2)
        for params in (
            ([a, b], 2 * b),
            ([a, b], a - b + 1),
            ([a, b], (a + b + 1) / 2),
        ):
            c = params[1]
            if (c <= 0 and c.denominator == 1) or c in (a, b) or a == 0:
                continue
            rhs = ak.hyper_transform(ak.hyper(params[0], [c], z), "quadratic")
            zv = complex(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
            got = cval(rhs, {z: zv})
            want = complex(mp.hyp2f1(mpf(a), mpf(b), mpf(c), mp.mpc(zv)))
            assert abs(got - want) <= 1e-10 * (1 + abs(want)), (a, b, c, zv)


def test_kummer_transformation(pool):
    z = pool.symbol("z")
    rhs = ak.hyper_transform(ak.hyper([F(2, 3)], [F(7, 4)], z), "kummer")
    for zv in (1.5 - 0.5j, -4 + 2j):
        want = complex(mp.hyp1f1(mp.mpf(2) / 3, mp.mpf(7) / 4, mp.mpc(zv)))
        assert abs(cval(rhs, {z: zv}) - want) < 1e-12 * abs(want)


def test_transformation_refusals(pool):
    z = pool.symbol("z")
    log_case = ak.hyper([1, 1], [2], z)
    for kind in ("1-z", "1/z", "1/(1-z)", "1-1/z"):
        with pytest.raises(ak.SpecialFunctionError) as info:
            ak.hyper_transform(log_case, kind)
        assert info.value.code == "E-SPECFN-009"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.hyper_transform(ak.hyper([F(1, 3), F(1, 7)], [F(9, 5)], z), "quadratic")
    assert info.value.code == "E-SPECFN-009"
    with pytest.raises(ak.SpecialFunctionError) as info:
        ak.hyper_transform(ak.hyper([F(1, 3)], [F(9, 5)], z), "euler")
    assert info.value.code == "E-SPECFN-009"
    with pytest.raises(ValueError):
        ak.hyper_transform(log_case, "nope")


# ---------------------------------------------------------------------------
# expand_hyper
# ---------------------------------------------------------------------------

EXPAND_CASES = [
    ([F(1, 3), F(5, 6)], [F(1, 2)]),
    ([F(1, 3), F(5, 6)], [F(3, 2)]),
    ([F(1, 3), F(5, 6)], [F(5, 3)]),
    ([F(1, 3), F(5, 6)], [F(2, 3)]),
    ([F(1, 2), 1], [2]),
    ([1, 1], [F(3, 2)]),
    ([F(7, 2), F(1, 3)], [F(3, 2)]),
    ([F(1, 2)], [1]),
    ([F(3, 4)], [F(3, 2)]),
    ([1], [F(3, 2)]),
]


def test_expand_hyper_lowergamma(pool):
    """1F1(a; a+1; -w) = a w^-a lowergamma(a, w); 1F1(1; a+1; w) via Kummer."""
    mp.mp.dps = 30
    w = pool.symbol("w")
    for a, b, sign in [(F(1, 3), F(4, 3), -1), (F(5, 2), F(7, 2), -1), (1, F(4, 3), 1)]:
        got = ak.expand_hyper(ak.hyper([a], [b], sign * w))
        assert "hyper" not in str(got), str(got)
        assert "lowergamma" in str(got) or "erf" in str(got), str(got)
        for wv in (0.3, 1.7, 4.0):
            want = float(mp.hyp1f1(mpf(a), mpf(b), sign * wv))
            assert abs(ak.eval_expr(got, {w: wv}) - want) < 1e-12 * abs(want), (a, b, wv)


@pytest.mark.parametrize(("a", "b"), EXPAND_CASES)
def test_expand_hyper_new_closed_forms(pool, a, b):
    mp.mp.dps = 30
    z = pool.symbol("z")
    got = ak.expand_hyper(ak.hyper(a, b, z))
    assert "hyper" not in str(got), str(got)
    for zv in (0.15, 0.4, 0.7):
        g = ak.eval_expr(got, {z: zv})
        want = mp.hyper([mpf(x) for x in a], [mpf(x) for x in b], zv)
        assert abs(g - float(want)) < 1e-12 * (1 + abs(float(want))), (a, b, zv, str(got))
    # SymPy's hyperexpand, where it finds a closed form, agrees.
    s = sp.Symbol("z")
    rat = lambda v: sp.Rational(F(v).numerator, F(v).denominator)  # noqa: E731
    want = sp.hyperexpand(sp.hyper([rat(v) for v in a], [rat(v) for v in b], s))
    if not want.has(sp.hyper):
        for zv in (0.15, 0.4):
            w = complex(sp.N(want.subs(s, zv), 30))
            assert abs(ak.eval_expr(got, {z: zv}) - w.real) < 1e-12 * (1 + abs(w))


# ---------------------------------------------------------------------------
# Digamma / polygamma at rationals
# ---------------------------------------------------------------------------


def test_digamma_gauss_theorem_matches_mpmath(pool):
    mp.mp.dps = 30
    rng = random.Random("f26-psi")
    seen = 0
    for q in range(5, 30):
        for p in range(1, q):
            if F(p, q).denominator != q or rng.random() > 0.3:
                continue
            x = F(p, q) + rng.choice([0, 1, -1, 3])
            v = ak.simplify(ak.digamma(pool.rational(x.numerator, x.denominator))).value
            assert "digamma" not in str(v), (x, str(v))
            got = ak.eval_expr(v, {})
            want = float(mp.digamma(mpf(x)))
            assert abs(got - want) < 1e-12 * (1 + abs(want)), (x, got, want)
            seen += 1
    assert seen > 30


def test_digamma_one_fifth_agrees_with_sympy(pool):
    v = ak.simplify(ak.digamma(pool.rational(1, 5))).value
    assert "digamma" not in str(v)
    want = sp.digamma(sp.Rational(1, 5))
    assert abs(ak.eval_expr(v, {}) - float(sp.N(want, 30))) < 1e-13


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_polygamma_at_rationals_matches_mpmath(pool, n):
    mp.mp.dps = 30
    for p, q in [(1, 3), (2, 3), (1, 4), (3, 4), (1, 6), (5, 6), (2, 5), (3, 8), (7, 10)]:
        x = F(p, q) + (1 if n % 2 else 0)
        v = ak.polygamma(n, pool.rational(x.numerator, x.denominator))
        s = str(v)
        assert "polygamma" not in s, s
        assert "trigamma" not in s, s
        classical = (n % 2 == 0 and q in (3, 4, 6)) or (n == 1 and q == 4)
        assert ("hurwitz_zeta" not in s) == classical, (n, x, s)
        # A shifted Hurwitz value is a difference of nearby numbers in f64
        # (hurwitz_zeta(6, 1/3) - 729), so check the exact value at 200 bits.
        got = ak.interval_eval(v, {})
        want = mp.polygamma(n, mpf(x))
        assert abs(got.mid - float(want)) < 1e-14 * abs(float(want)), (n, x, s)


def test_polygamma_quarter_classical_forms(pool):
    # psi''(1/4) = -2 pi^3 - 56 zeta(3)
    v = ak.polygamma(2, pool.rational(1, 4))
    e = sp.sympify(str(v).replace("^", "**"), locals={"zeta": sp.zeta, "pi": sp.pi})
    assert sp.simplify(e - (-2 * sp.pi**3 - 56 * sp.zeta(3))) == 0
    # psi''(1/3) = -4 sqrt(3) pi^3 / 9 - 26 zeta(3)
    v = ak.polygamma(2, pool.rational(1, 3))
    e = sp.sympify(str(v).replace("^", "**"), locals={"zeta": sp.zeta, "pi": sp.pi})
    want = -4 * sp.sqrt(3) * sp.pi**3 / 9 - 26 * sp.zeta(3)
    assert sp.simplify(e - want) == 0
