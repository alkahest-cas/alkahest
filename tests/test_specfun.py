"""Bessel, Airy, Riemann/Hurwitz zeta and polylogarithm functions (G11b).

Oracles:

* **mpmath** — ``besselj/bessely/besseli/besselk``, ``airyai/airybi``,
  ``zeta``, ``polylog`` at random real points (``f64`` to ~1e-14 relative),
  at random complex points (``evaluate`` with complex bindings), and rigorous
  balls at 200 bits;
* **SymPy** — ``expand_func(besselj(n + 1/2, x))`` for the half-integer
  closed forms, ``zeta(2n)`` / ``zeta(-n)``, ``expand_func(polylog(-n, z))``
  and the Airy values at 0, compared exactly;
* **finite differences** — the derivative identities.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import alkahest as ak
import pytest

mp = pytest.importorskip("mpmath")

NEW = [
    "bessel_j",
    "bessel_y",
    "bessel_i",
    "bessel_k",
    "spherical_jn",
    "spherical_yn",
    "airy_ai",
    "airy_bi",
    "airy_ai_prime",
    "airy_bi_prime",
    "zeta",
    "hurwitz_zeta",
    "polylog",
]
BINARY = {"bessel_j", "bessel_y", "bessel_i", "bessel_k", "spherical_jn", "spherical_yn"}
BINARY |= {"hurwitz_zeta", "polylog"}


@pytest.fixture
def pool():
    return ak.ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def to_sympy(e):
    """``alkahest.crosscheck.to_sympy`` with the ``pi`` symbol read as ``sympy.pi``."""
    sp = pytest.importorskip("sympy")
    from alkahest.crosscheck import to_sympy as _to_sympy

    return _to_sympy(e).subs(sp.Symbol("pi"), sp.pi)


def has_arb(pool, x):
    try:
        ak.eval_expr(ak.airy_ai(x), {x: 0.5})
        return True
    except (ak.AlkahestError, ValueError):
        return False


@pytest.fixture
def arb(pool, x):
    if not has_arb(pool, x):
        pytest.skip("numeric kernels need FLINT's Arb layer in this build")


def node(name, pool, order, x):
    """The unexpanded node ``name(order, x)`` (or ``name(x)``)."""
    fn = getattr(ak, name)
    if name in {"bessel_j", "bessel_y", "bessel_i", "bessel_k", "spherical_jn", "spherical_yn"}:
        return fn(order, x, expand=False)
    if name in BINARY:
        return fn(order, x)
    return fn(x)


def oracle(name, order, xv):
    if name == "bessel_j":
        return mp.besselj(order, xv)
    if name == "bessel_y":
        return mp.bessely(order, xv)
    if name == "bessel_i":
        return mp.besseli(order, xv)
    if name == "bessel_k":
        return mp.besselk(order, xv)
    if name == "spherical_jn":
        return mp.sqrt(mp.pi / (2 * xv)) * mp.besselj(order + mp.mpf(1) / 2, xv)
    if name == "spherical_yn":
        return mp.sqrt(mp.pi / (2 * xv)) * mp.bessely(order + mp.mpf(1) / 2, xv)
    if name == "airy_ai":
        return mp.airyai(xv)
    if name == "airy_bi":
        return mp.airybi(xv)
    if name == "airy_ai_prime":
        return mp.airyai(xv, 1)
    if name == "airy_bi_prime":
        return mp.airybi(xv, 1)
    if name == "zeta":
        return mp.zeta(xv)
    if name == "hurwitz_zeta":
        return mp.zeta(order, xv)
    if name == "polylog":
        return mp.polylog(order, xv)
    raise AssertionError(name)


def sample(name, rng):
    """A random ``(order, x)`` inside the real domain of ``name``."""
    if name in {"bessel_j", "bessel_i"}:
        return rng.choice([0, 1, 2, 5, rng.uniform(-3.5, 6.0)]), rng.uniform(0.05, 25.0)
    if name in {"bessel_y", "bessel_k"}:
        return rng.uniform(-4.0, 6.0), rng.uniform(0.05, 25.0)
    if name in {"spherical_jn", "spherical_yn"}:
        return rng.randint(0, 8), rng.uniform(0.1, 20.0)
    if name.startswith("airy"):
        return None, rng.uniform(-12.0, 6.0)
    if name == "zeta":
        return None, rng.choice([rng.uniform(-15.0, 0.9), rng.uniform(1.1, 30.0)])
    if name == "hurwitz_zeta":
        return rng.choice([rng.uniform(1.2, 9.0), rng.uniform(-6.0, 0.8)]), rng.uniform(0.1, 5.0)
    if name == "polylog":
        return rng.choice([2, 3, 4, rng.uniform(-3.0, 6.0)]), rng.uniform(-8.0, 0.95)
    raise AssertionError(name)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_exported():
    for name in [*NEW, "expand_special", "bessel_j0", "bessel_j1"]:
        assert name in ak.__all__, name
        assert hasattr(ak, name), name


def test_capability_flags_are_truthful(pool, x):
    rows = {r["name"]: r for r in ak.capabilities()["primitives"]}
    n = pool.symbol("n")
    for name in NEW:
        row = rows[name]
        # ζ′ has no closed form: zeta must not claim a derivative.
        assert row["diff_forward"] == (name != "zeta"), name
        for flag in ("diff_reverse", "lean_theorem", "taylor_model", "lower_llvm"):
            assert not row[flag], (name, flag)
        e = node(name, pool, n, x)
        env_f = {n: 2.0, x: 0.75} if name in BINARY else {x: 0.75}
        try:
            ak.eval_expr(e, env_f)
            works = True
        except (ak.AlkahestError, ValueError):
            works = False
        assert row["numeric_f64"] == works, name
        env_b = {k: ak.ArbBall(v) for k, v in env_f.items()}
        try:
            ak.interval_eval(e, env_b)
            works = True
        except ValueError:
            works = False
        assert row["numeric_ball"] == works, name


# ---------------------------------------------------------------------------
# Numeric oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NEW)
def test_f64_matches_mpmath_at_random_real_points(pool, x, arb, name):
    mp.mp.dps = 40
    n = pool.symbol("n")
    e = node(name, pool, n, x)
    rng = random.Random(name)
    for _ in range(25):
        order, xv = sample(name, rng)
        env = {n: float(order), x: xv} if order is not None else {x: xv}
        got = ak.eval_expr(e, env)
        want = oracle(name, mp.mpf(order) if order is not None else None, mp.mpf(xv))
        assert mp.im(want) == 0 or abs(mp.im(want)) < 1e-30 * abs(want)
        want = float(mp.re(want))
        assert abs(got - want) <= 1e-14 * abs(want) + 1e-300, (name, order, xv, got, want)


@pytest.mark.parametrize("name", NEW)
def test_complex_values_match_mpmath(pool, x, arb, name):
    mp.mp.dps = 40
    n = pool.symbol("n")
    rng = random.Random("c" + name)
    for _ in range(12):
        order, _ = sample(name, rng)
        if name in {"spherical_jn", "spherical_yn"}:
            order = rng.randint(0, 5)
        zv = complex(rng.uniform(-4.0, 4.0), rng.uniform(0.2, 4.0))
        if name == "zeta":
            zv = complex(rng.uniform(-3.0, 4.0), rng.uniform(0.5, 20.0))
        if name == "polylog":
            zv = complex(rng.uniform(-3.0, 3.0), rng.uniform(0.2, 3.0))
        e = node(name, pool, pool.integer(order) if name.startswith("spherical") else n, x)
        env = {x: zv}
        if order is not None and not name.startswith("spherical"):
            env[n] = complex(order, 0.0)
        r = ak.evaluate(e, env, mode="complex")
        assert r.status == "ok", (name, order, zv, r.reason)
        got = complex(r.value)
        want = complex(oracle(name, mp.mpf(order) if order is not None else None, mp.mpc(zv)))
        assert abs(got - want) <= 1e-13 * abs(want) + 1e-300, (name, order, zv, got, want)


@pytest.mark.parametrize("name", NEW)
def test_balls_at_200_bits(pool, x, arb, name):
    mp.mp.prec = 400
    n = pool.symbol("n")
    e = node(name, pool, n, x)
    rng = random.Random("b" + name)
    for _ in range(6):
        order, xv = sample(name, rng)
        env = {x: ak.ArbBall(xv, 0.0, 200)}
        if order is not None:
            env[n] = ak.ArbBall(float(order), 0.0, 200)
        b = ak.interval_eval(e, env)
        want = mp.re(oracle(name, mp.mpf(order) if order is not None else None, mp.mpf(xv)))
        scale = max(1.0, abs(float(want)))
        # 200 bits ≈ 60 digits: the enclosure is tight, and its midpoint (rounded
        # to a double for Python) is the double nearest the true value.
        assert b.rad < 1e-50 * scale, (name, order, xv, b.rad)
        assert abs(b.mid - float(want)) <= 2.3e-16 * abs(float(want)) + 1e-300, (name, b.mid)
        assert b.lo <= float(want) <= b.hi or abs(b.mid - float(want)) < 1e-300


def test_out_of_domain_values_decline_rather_than_return_a_real_part(pool, x, arb):
    n = pool.symbol("n")
    for e, env in [
        (ak.bessel_y(n, x, expand=False), {n: 0.0, x: -1.0}),
        (ak.bessel_k(n, x, expand=False), {n: 1.0, x: -2.0}),
        (ak.bessel_j(n, x, expand=False), {n: 0.5, x: -1.0}),
        (ak.polylog(n, x), {n: 3.0, x: 2.0}),
        (ak.zeta(x), {x: 1.0}),
    ]:
        with pytest.raises((ak.AlkahestError, ValueError)):
            ak.eval_expr(e, env)
    # The complex evaluator has them.
    r = ak.evaluate(ak.polylog(n, x), {n: 3 + 0j, x: 2 + 0j}, mode="complex")
    want = complex(mp.polylog(3, 2))
    assert abs(complex(r.value) - want) < 1e-13 * abs(want)


# ---------------------------------------------------------------------------
# Exact values
# ---------------------------------------------------------------------------


def test_bessel_special_values_and_aliases(pool, x):
    z = pool.integer(0)
    assert ak.bessel_j(0, x) == ak.bessel_j0(x)
    assert ak.bessel_j(1, x) == ak.bessel_j1(x)
    assert ak.bessel_j(0, z) == pool.integer(1)
    assert ak.simplify(ak.bessel_j0(z)).value == pool.integer(1)
    assert ak.simplify(ak.bessel_j1(z)).value == pool.integer(0)
    for nn in (2, 3, 7):
        assert ak.bessel_j(nn, z) == pool.integer(0)
        assert ak.bessel_i(nn, z) == pool.integer(0)
    assert ak.bessel_i(0, z) == pool.integer(1)
    # Reflection in the order.
    assert ak.bessel_j(-3, x) == -ak.bessel_j(3, x)
    assert ak.bessel_j(-2, x) == ak.bessel_j(2, x)
    assert ak.bessel_y(-1, x) == -ak.bessel_y(1, x)
    assert ak.bessel_i(-5, x) == ak.bessel_i(5, x)
    assert ak.bessel_k(F(-7, 3), x) == ak.bessel_k(F(7, 3), x)
    # The singular ones stay nodes.
    assert "bessel_y" in str(ak.bessel_y(0, z))


def test_zeta_exact_values_match_sympy(pool):
    sp = pytest.importorskip("sympy")

    for s in list(range(-12, 1)) + list(range(2, 31, 2)):
        got = to_sympy(ak.zeta(pool.integer(s)))
        assert sp.simplify(got - sp.zeta(s)) == 0, s
    assert str(ak.zeta(pool.integer(3))) == "zeta(3)"
    with pytest.raises(ak.SpecialFunctionError) as ei:
        ak.zeta(pool.integer(1))
    assert ei.value.code == "E-SPECFN-006"


def test_polylog_closed_forms_match_sympy(pool, x):
    sp = pytest.importorskip("sympy")

    z = sp.Symbol("x")
    for s in range(-8, 2):
        got = to_sympy(ak.polylog(pool.integer(s), x))
        want = sp.expand_func(sp.polylog(s, z))
        assert sp.simplify(got - want) == 0, s
    assert ak.polylog(pool.integer(2), x) == ak.dilog(x)
    pi = pool.symbol("pi")
    assert ak.polylog(pool.integer(4), pool.integer(1)) == pool.rational(1, 90) * pi**4
    li3m1 = to_sympy(ak.polylog(pool.integer(3), pool.integer(-1)))
    assert sp.simplify(li3m1 - sp.polylog(3, -1).rewrite(sp.zeta)) == 0
    with pytest.raises(ak.SpecialFunctionError) as ei:
        ak.polylog(pool.integer(1), pool.integer(1))
    assert ei.value.code == "E-SPECFN-006"


def test_hurwitz_zeta_reductions(pool):
    sp = pytest.importorskip("sympy")

    a = pool.symbol("a")
    sa = sp.Symbol("a")
    for nn in range(0, 7):
        got = to_sympy(ak.hurwitz_zeta(pool.integer(-nn), a))
        want = -sp.bernoulli(nn + 1, sa) / (nn + 1)
        assert sp.expand(got - want) == 0, nn
    assert ak.hurwitz_zeta(a, pool.integer(1)) == ak.zeta(a)
    got = to_sympy(ak.hurwitz_zeta(pool.integer(3), pool.rational(1, 2)))
    assert sp.simplify(got - 7 * sp.zeta(3)) == 0


def test_airy_values_at_zero_match_sympy(pool):
    sp = pytest.importorskip("sympy")

    z = pool.integer(0)
    for name, want in [
        ("airy_ai", sp.airyai(0)),
        ("airy_bi", sp.airybi(0)),
        ("airy_ai_prime", sp.airyaiprime(0)),
        ("airy_bi_prime", sp.airybiprime(0)),
    ]:
        got = to_sympy(getattr(ak, name)(z))
        assert abs(complex(sp.N(got - want, 30))) < 1e-25, name


@pytest.mark.parametrize("name", ["bessel_j", "bessel_y", "bessel_i", "bessel_k"])
def test_half_integer_closed_forms_match_sympy(pool, x, name):
    sp = pytest.importorskip("sympy")

    sx = sp.Symbol("x", positive=True)
    sfn = {"bessel_j": sp.besselj, "bessel_y": sp.bessely}
    sfn |= {"bessel_i": sp.besseli, "bessel_k": sp.besselk}
    for num in (-7, -5, -3, -1, 1, 3, 5, 7):
        e = getattr(ak, name)(F(num, 2), x)
        assert name not in str(e), (name, num, str(e))
        got = to_sympy(e).subs(sp.Symbol("x"), sx)
        want = sp.expand_func(sfn[name](sp.Rational(num, 2), sx))
        diff = sp.simplify(got - want)
        if diff != 0:
            # SymPy cannot always close it; then 40-digit agreement at 3 points.
            for v in (sp.Rational(3, 10), sp.Rational(17, 7), sp.Integer(9)):
                assert abs(sp.N(diff.subs(sx, v), 40)) < 1e-30, (name, num, v)


@pytest.mark.parametrize("name", ["spherical_jn", "spherical_yn"])
def test_spherical_closed_forms_match_sympy(pool, x, name):
    sp = pytest.importorskip("sympy")

    sx = sp.Symbol("x")
    sfn = sp.jn if name == "spherical_jn" else sp.yn
    for nn in range(0, 7):
        got = to_sympy(getattr(ak, name)(nn, x))
        want = sp.expand_func(sfn(nn, sx))
        assert sp.simplify(got - want) == 0, (name, nn)
    with pytest.raises(ak.SpecialFunctionError) as ei:
        ak.spherical_jn(F(1, 2), x)
    assert ei.value.code == "E-SPECFN-007"


def test_expand_special_after_subs(pool, x):
    nu = pool.symbol("nu")
    e = ak.bessel_j(nu, x)
    assert str(e) == "bessel_j(nu, x)"
    e2 = ak.expand_special(ak.subs(e, {nu: pool.rational(1, 2)}))
    assert "bessel" not in str(e2)
    assert "sin" in str(e2)


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


CASES = [
    ("bessel_j", F(7, 3), 1.3),
    ("bessel_j", 0, 2.2),
    ("bessel_y", 2, 2.1),
    ("bessel_i", F(1, 3), 0.8),
    ("bessel_k", F(5, 4), 1.6),
    ("spherical_jn", 3, 1.9),
    ("spherical_yn", 2, 2.4),
    ("airy_ai", None, 0.7),
    ("airy_bi", None, -1.1),
    ("airy_ai_prime", None, 0.3),
    ("airy_bi_prime", None, -0.9),
    ("hurwitz_zeta", F(5, 2), 0.6),
    ("polylog", F(5, 2), 0.35),
    ("polylog", 3, -0.6),
]


@pytest.mark.parametrize(("name", "order", "xv"), CASES)
def test_derivative_identities_vs_finite_differences(pool, x, arb, name, order, xv):
    mp.mp.dps = 30
    fn = getattr(ak, name)
    if order is None:
        e = fn(x)
    elif name.startswith(("bessel", "spherical")):
        e = fn(order, x, expand=False)
    else:
        e = fn(order, x)
    d = ak.diff(e, x).value
    got = ak.eval_expr(d, {x: xv})
    o = mp.mpf(order.numerator) / order.denominator if isinstance(order, F) else order
    want = float(mp.diff(lambda t: oracle(name, o, t), mp.mpf(xv)))
    assert abs(got - want) <= 1e-12 * (1 + abs(want)), (name, order, got, want)


def test_derivative_in_the_order_is_refused_not_dropped(pool, x):
    n = pool.symbol("n")
    for e in (
        ak.bessel_j(x, n, expand=False),
        ak.bessel_k(x, pool.integer(2), expand=False),
        ak.polylog(x, pool.rational(1, 2)),
        ak.hurwitz_zeta(x, pool.integer(3)),
        ak.zeta(x),
    ):
        with pytest.raises(ak.DiffError) as ei:
            ak.diff(e, x)
        assert ei.value.code == "E-DIFF-001"
    # A constant zeta value differentiates to 0.
    assert ak.diff(ak.zeta(pool.integer(3)), x).value == pool.integer(0)


def test_symbolic_derivative_shapes(pool, x):
    nu = pool.symbol("nu")
    d = ak.diff(ak.bessel_j(nu, x), x).value
    s = str(d)
    lo = ak.bessel_j(nu - 1, x)
    hi = ak.bessel_j(nu + 1, x)
    assert ak.simplify(d - (lo - hi) / 2).value == pool.integer(0), s
    # d/dx J0 = -J1 still holds through the alias.
    assert ak.simplify(ak.diff(ak.bessel_j(0, x), x).value).value == -ak.bessel_j1(x)
    # Ai'' = x Ai
    d2 = ak.diff(ak.diff(ak.airy_ai(x), x).value, x).value
    assert ak.simplify(d2 - x * ak.airy_ai(x)).value == pool.integer(0)


# ---------------------------------------------------------------------------
# Sums, integrals, series, parsing
# ---------------------------------------------------------------------------


def test_odd_zeta_sums_are_named(pool):
    k = pool.symbol("k")
    inf = pool.pos_infinity()
    v = ak.sum_definite(k**-3, k, pool.integer(1), inf).value
    assert v == ak.zeta(pool.integer(3)), str(v)
    v = ak.sum_definite((2 * k + 1) ** -3, k, pool.integer(0), inf).value
    assert ak.simplify(v - pool.rational(7, 8) * ak.zeta(pool.integer(3))).value == pool.integer(0)
    v = ak.sum_definite(k**-5, k, pool.integer(2), inf).value
    assert ak.simplify(v - (ak.zeta(pool.integer(5)) - 1)).value == pool.integer(0)


def test_bessel_integrals(pool, x):
    r = ak.integrate(x * ak.bessel_j0(x), x)
    assert ak.simplify(r.value - x * ak.bessel_j1(x)).value == pool.integer(0), str(r.value)
    r = ak.integrate(ak.bessel_j1(x), x)
    assert ak.simplify(r.value + ak.bessel_j0(x)).value == pool.integer(0), str(r.value)


def test_parse_round_trip(pool, x):
    n = pool.symbol("n")
    for e in (
        ak.bessel_j(n, x),
        ak.bessel_k(pool.integer(3), x),
        ak.spherical_yn(n, x),
        ak.airy_bi_prime(x),
        ak.zeta(x),
        ak.hurwitz_zeta(n, x),
        ak.polylog(n, x),
    ):
        back = ak.parse(str(e), pool, {"x": x, "n": n})
        assert back == e, (str(e), str(back))


def test_taylor_series_at_zero(pool, x):
    sp = pytest.importorskip("sympy")
    sx = sp.Symbol("x")
    for name, nn, sfn in [
        ("bessel_j", 0, sp.besselj),
        ("bessel_j", 1, sp.besselj),
        ("bessel_j", 3, sp.besselj),
        ("bessel_i", 2, sp.besseli),
    ]:
        e = getattr(ak, name)(nn, x)
        s = ak.series(e, x, pool.integer(0), 10)
        got = sp.expand(to_sympy(s.truncated()))
        want = sp.series(sfn(nn, sx), sx, 0, 10).removeO()
        assert sp.expand(got - want) == 0, (name, nn, got, want)
