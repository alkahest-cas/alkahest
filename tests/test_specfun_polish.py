"""F24 — special-function polish.

* Taylor-model rules (``taylor_model`` capability) for the orthogonal
  polynomials, integer-order Bessel ``J/Y/I/K``, Airy, ``zeta``, ``polylog``
  and ``polygamma``: ``bound_on_box`` encloses mpmath values on random boxes.
* ``real_spherical_harmonic_y``: orthonormal real basis.
* ``hankel1`` / ``hankel2`` against mpmath.
* ``zeta_prime`` against ``mpmath.zeta(s, derivative=1)``; ``diff(zeta)``.
* LaTeX notation of the whole family.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import alkahest as ak
import pytest

mp = pytest.importorskip("mpmath")

mp.mp.dps = 40


@pytest.fixture
def pool():
    p = ak.ExprPool()
    with ak.context(pool=p):
        yield p


@pytest.fixture
def x(pool):
    return ak.symbol("x", pool=pool)


SLACK = 1e-12


def _ev(expr, env):
    """Numeric value of ``expr`` (a float, or a complex for complex bindings)."""
    if any(isinstance(v, complex) for v in env.values()):
        return ak.evaluate(expr, env).value
    return ak.eval_expr(expr, env)


def _contains(r, values):
    lo, hi = float(r.lower), float(r.upper)
    for v in values:
        v = float(v)
        assert lo - SLACK * (1 + abs(v)) <= v <= hi + SLACK * (1 + abs(v)), (lo, hi, v)


def _samples(lo, hi, n=40):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


# (builder, mpmath oracle, box generator range)
def _cases():
    F = Fraction
    return [
        ("legendre_p(5)", lambda x: ak.legendre_p(5, x), lambda t: mp.legendre(5, t), (-1.0, 1.0)),
        ("chebyshev_t(7)", lambda x: ak.chebyshev_t(7, x), lambda t: mp.chebyt(7, t), (-1.0, 1.0)),
        ("chebyshev_u(4)", lambda x: ak.chebyshev_u(4, x), lambda t: mp.chebyu(4, t), (-1.0, 1.0)),
        ("hermite_h(4)", lambda x: ak.hermite_h(4, x), lambda t: mp.hermite(4, t), (-2.0, 2.0)),
        (
            "hermite_he(5)",
            lambda x: ak.hermite_he(5, x),
            lambda t: mp.mpf(2) ** (-mp.mpf(5) / 2) * mp.hermite(5, t / mp.sqrt(2)),
            (-2.0, 2.0),
        ),
        (
            "laguerre_l(4)",
            lambda x: ak.laguerre_l(4, x),
            lambda t: mp.laguerre(4, 0, t),
            (0.0, 6.0),
        ),
        (
            "assoc_laguerre_l(3, 1/2)",
            lambda x: ak.assoc_laguerre_l(3, F(1, 2), x),
            lambda t: mp.laguerre(3, mp.mpf(1) / 2, t),
            (0.0, 6.0),
        ),
        (
            "gegenbauer_c(4, 3/2)",
            lambda x: ak.gegenbauer_c(4, F(3, 2), x),
            lambda t: mp.gegenbauer(4, mp.mpf(3) / 2, t),
            (-1.0, 1.0),
        ),
        (
            "jacobi_p(3, 1/2, -1/3)",
            lambda x: ak.jacobi_p(3, F(1, 2), F(-1, 3), x),
            lambda t: mp.jacobi(3, mp.mpf(1) / 2, -mp.mpf(1) / 3, t),
            (-0.95, 0.95),
        ),
        ("bessel_j(2)", lambda x: ak.bessel_j(2, x), lambda t: mp.besselj(2, t), (0.0, 8.0)),
        ("bessel_j(-3)", lambda x: ak.bessel_j(-3, x), lambda t: mp.besselj(-3, t), (0.0, 8.0)),
        ("bessel_y(0)", lambda x: ak.bessel_y(0, x), lambda t: mp.bessely(0, t), (0.3, 9.0)),
        ("bessel_y(2)", lambda x: ak.bessel_y(2, x), lambda t: mp.bessely(2, t), (0.8, 9.0)),
        ("bessel_y(-1)", lambda x: ak.bessel_y(-1, x), lambda t: mp.bessely(-1, t), (0.8, 9.0)),
        ("bessel_i(1)", lambda x: ak.bessel_i(1, x), lambda t: mp.besseli(1, t), (-3.0, 3.0)),
        ("bessel_k(0)", lambda x: ak.bessel_k(0, x), lambda t: mp.besselk(0, t), (0.3, 6.0)),
        ("bessel_k(3)", lambda x: ak.bessel_k(3, x), lambda t: mp.besselk(3, t), (0.8, 6.0)),
        ("airy_ai", lambda x: ak.airy_ai(x), lambda t: mp.airyai(t), (-6.0, 3.0)),
        ("airy_bi", lambda x: ak.airy_bi(x), lambda t: mp.airybi(t), (-6.0, 3.0)),
        (
            "airy_ai_prime",
            lambda x: ak.airy_ai_prime(x),
            lambda t: mp.airyai(t, derivative=1),
            (-6.0, 3.0),
        ),
        (
            "airy_bi_prime",
            lambda x: ak.airy_bi_prime(x),
            lambda t: mp.airybi(t, derivative=1),
            (-6.0, 3.0),
        ),
        ("zeta (s > 1)", lambda x: ak.zeta(x), lambda t: mp.zeta(t), (1.2, 6.0)),
        ("zeta (s < 1)", lambda x: ak.zeta(x), lambda t: mp.zeta(t), (-5.0, 0.8)),
        ("polygamma(0)", lambda x: ak.polygamma(0, x), lambda t: mp.polygamma(0, t), (0.3, 8.0)),
        ("polygamma(2)", lambda x: ak.polygamma(2, x), lambda t: mp.polygamma(2, t), (0.3, 8.0)),
        ("polylog(3)", lambda x: ak.polylog(3, x), lambda t: mp.polylog(3, t), (-0.95, 0.95)),
        (
            "polylog(5/2)",
            lambda x: ak.polylog(F(5, 2), x),
            lambda t: mp.polylog(mp.mpf(5) / 2, t),
            (-0.9, 0.9),
        ),
        ("polylog(-1)", lambda x: ak.polylog(-1, x), lambda t: mp.polylog(-1, t), (-0.9, 0.9)),
    ]


def _bound_or_none(e, x, a, b):
    """``bound_on_box`` on ``[a, b]``; ``None`` for a domain/overflow refusal."""
    try:
        return ak.bound_on_box(e, [(x, a, b)], order=8, prec=128, tol=1e-2, max_subdivisions=6)
    except ak.ValidatedError as exc:
        if exc.code not in {"E-VALIDATED-003", "E-VALIDATED-004"}:
            raise
        return None


@pytest.mark.parametrize(("name", "build", "oracle", "rng"), _cases(), ids=[c[0] for c in _cases()])
def test_bound_on_box_encloses_mpmath_on_random_boxes(name, build, oracle, rng, x):
    gen = random.Random(hash(name) & 0xFFFF)
    e = build(x)
    lo_all, hi_all = rng
    for _ in range(6):
        a, b = sorted(gen.uniform(lo_all, hi_all) for _ in range(2))
        if b - a < 1e-3:
            b = min(a + 0.05, hi_all)
        # A box that reaches a singular point is a legitimate coded refusal
        # (zeta's pole, polylog at 0 or 1, ...): skip those boxes.
        r = _bound_or_none(e, x, a, b)
        if r is None:
            continue
        _contains(r, (oracle(mp.mpf(t)) for t in _samples(a, b)))


def test_bound_on_box_on_composite_expressions(x, pool):
    y = ak.symbol("y", pool=pool)
    e = (
        ak.bessel_y(1, x) * ak.airy_ai(x)
        + ak.polygamma(1, y)
        - ak.zeta(y + 2) * ak.legendre_p(3, x)
    )
    box = [(x, 0.5, 1.5), (y, 0.5, 1.0)]
    r = ak.bound_on_box(e, box, order=8, prec=128, tol=1e-2, max_subdivisions=8)
    gen = random.Random(3)
    vals = []
    for _ in range(200):
        xv, yv = gen.uniform(0.5, 1.5), gen.uniform(0.5, 1.0)
        vals.append(
            mp.bessely(1, xv) * mp.airyai(xv)
            + mp.polygamma(1, yv)
            - mp.zeta(yv + 2) * mp.legendre(3, xv)
        )
    _contains(r, vals)
    # the enclosure is informative, not just correct
    assert float(r.upper) - float(r.lower) < 50.0


def test_taylor_model_tight_on_small_boxes(x):
    for e, oracle in [
        (ak.bessel_k(2, x), lambda t: mp.besselk(2, t)),
        (ak.airy_bi(x), lambda t: mp.airybi(t)),
        (ak.zeta(x), lambda t: mp.zeta(t)),
        (ak.polylog(2, x), lambda t: mp.polylog(2, t)),
    ]:
        r = ak.bound_on_box(e, [(x, 0.6, 0.61)], order=6, prec=128)
        assert float(r.upper) - float(r.lower) < 0.2, e
        _contains(r, (oracle(mp.mpf(t)) for t in _samples(0.6, 0.61, 11)))


def test_capability_flags_are_truthful():
    rows = {r["name"]: r for r in ak.capabilities()["primitives"]}
    supported = [
        "legendre_p",
        "assoc_legendre_p",
        "hermite_h",
        "hermite_he",
        "laguerre_l",
        "assoc_laguerre_l",
        "chebyshev_t",
        "chebyshev_u",
        "gegenbauer_c",
        "jacobi_p",
        "bessel_j",
        "bessel_y",
        "bessel_i",
        "bessel_k",
        "airy_ai",
        "airy_bi",
        "airy_ai_prime",
        "airy_bi_prime",
        "zeta",
        "polylog",
        "polygamma",
    ]
    for name in supported:
        assert rows[name]["taylor_model"] is True, name
    for name in [
        "spherical_jn",
        "spherical_yn",
        "hurwitz_zeta",
        "hyper",
        "zeta_prime",
        "hankel1",
        "hankel2",
    ]:
        assert rows[name]["taylor_model"] is False, name
    # complex-valued: no real numeric slots
    assert rows["hankel1"]["numeric_f64"] is False
    assert rows["zeta_prime"]["numeric_f64"] is True


def test_taylor_model_refusals_are_coded(x, pool):
    nu = ak.symbol("nu", pool=pool)
    n = ak.symbol("n", pool=pool)

    def code(expr, lo, hi, extra=()):
        with pytest.raises(ak.ValidatedError) as err:
            ak.bound_on_box(expr, [(x, lo, hi), *extra], order=4, prec=64, max_subdivisions=2)
        return err.value.code

    # (`bound_on_box` reports a domain refusal that survives subdivision as
    # E-VALIDATED-003 or, once the box has been split, E-VALIDATED-004.)
    domain = {"E-VALIDATED-003", "E-VALIDATED-004"}
    # y/k are singular at 0 and not real below it
    assert code(ak.bessel_y(1, x), -1.0, 1.0) in domain
    assert code(ak.bessel_k(0, x), 0.0, 1.0) in domain
    # pole of zeta, polygamma poles, past the branch point of Li_s
    assert code(ak.zeta(x), 0.5, 1.5) in domain
    assert code(ak.polygamma(1, x), -1.0, 1.0) in domain
    assert code(ak.polylog(2, x), 0.5, 1.0) in domain
    # a symbolic order or degree, and a non-integer Bessel order, have no rule
    assert code(ak.bessel_j(nu, x), 1.0, 2.0, extra=[(nu, 1.0, 1.0)]) == "E-VALIDATED-005"
    assert code(ak.legendre_p(n, x), 0.1, 0.2, extra=[(n, 2.0, 2.0)]) == "E-VALIDATED-005"
    assert code(ak.bessel_y(Fraction(1, 2), x, expand=False), 1.0, 2.0) == "E-VALIDATED-001"
    assert code(ak.hankel1(0, x), 1.0, 2.0) == "E-VALIDATED-001"
    assert code(ak.zeta_prime(x), 2.0, 3.0) == "E-VALIDATED-001"


# ---------------------------------------------------------------------------
# Real spherical harmonics
# ---------------------------------------------------------------------------


def _gauss_sphere(n_theta=14, n_phi=40):
    np = pytest.importorskip("numpy")

    xs, ws = np.polynomial.legendre.leggauss(n_theta)
    thetas = np.arccos(xs)
    phis = 2 * np.pi * np.arange(n_phi) / n_phi
    return thetas, ws, phis, 2 * np.pi / n_phi


def test_real_spherical_harmonics_are_orthonormal(pool):
    np = pytest.importorskip("numpy")

    t = ak.symbol("t", pool=pool)
    f = ak.symbol("f", pool=pool)
    lmax = 3
    basis = [(ell, m) for ell in range(lmax + 1) for m in range(-ell, ell + 1)]
    exprs = [ak.real_spherical_harmonic_y(ell, m, t, f) for ell, m in basis]
    thetas, ws, phis, dphi = _gauss_sphere()
    vals = np.zeros((len(basis), len(thetas), len(phis)))
    for k, e in enumerate(exprs):
        for i, th in enumerate(thetas):
            for j, ph in enumerate(phis):
                vals[k, i, j] = _ev(e, {t: float(th), f: float(ph)})
    gram = np.einsum("kij,lij,i->kl", vals, vals, ws) * dphi
    assert np.allclose(gram, np.eye(len(basis)), atol=1e-10)


def test_real_spherical_harmonics_convention(pool):
    t = ak.symbol("t", pool=pool)
    f = ak.symbol("f", pool=pool)
    c = math.sqrt(3 / (4 * math.pi))
    for (ell, m), want in {
        (1, 1): lambda th, ph: c * math.sin(th) * math.cos(ph),  # x/r
        (1, -1): lambda th, ph: c * math.sin(th) * math.sin(ph),  # y/r
        (1, 0): lambda th, ph: c * math.cos(th),  # z/r
        (2, 0): lambda th, ph: math.sqrt(5 / (16 * math.pi)) * (3 * math.cos(th) ** 2 - 1),
        (2, 1): lambda th, ph: (
            math.sqrt(15 / (4 * math.pi)) * math.sin(th) * math.cos(th) * math.cos(ph)
        ),
        (2, -2): lambda th, ph: (
            math.sqrt(15 / (16 * math.pi)) * math.sin(th) ** 2 * math.sin(2 * ph)
        ),
    }.items():
        e = ak.real_spherical_harmonic_y(ell, m, t, f)
        for th, ph in [(0.7, 0.3), (2.0, -1.2), (1.1, 4.0)]:
            got = _ev(e, {t: th, f: ph})
            assert abs(got - want(th, ph)) < 1e-13, (ell, m, th, ph)


def test_real_spherical_harmonics_match_sympy_ynm(pool):
    sp = pytest.importorskip("sympy")
    t = ak.symbol("t", pool=pool)
    f = ak.symbol("f", pool=pool)
    for ell in range(0, 4):
        for m in range(-ell, ell + 1):
            e = ak.real_spherical_harmonic_y(ell, m, t, f)
            for th, ph in [(0.9, 0.4), (2.3, -2.0)]:
                y = complex(sp.N(sp.Ynm(ell, abs(m), th, ph).expand(func=True)))
                if m == 0:
                    want = y.real
                elif m > 0:
                    want = math.sqrt(2) * (-1) ** m * y.real
                else:
                    want = math.sqrt(2) * (-1) ** m * y.imag
                assert abs(_ev(e, {t: th, f: ph}) - want) < 1e-12, (ell, m)


def test_real_spherical_harmonic_errors(pool):
    t = ak.symbol("t", pool=pool)
    f = ak.symbol("f", pool=pool)
    assert str(ak.real_spherical_harmonic_y(1, 2, t, f)) == "0"
    with pytest.raises(ak.SpecialFunctionError) as err:
        ak.real_spherical_harmonic_y(Fraction(1, 2), 0, t, f)
    assert err.value.code == "E-SPECFN-003"
    with pytest.raises(ak.SpecialFunctionError) as err:
        ak.real_spherical_harmonic_y(-1, 0, t, f)
    assert err.value.code == "E-SPECFN-002"


# ---------------------------------------------------------------------------
# Hankel functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [0, 1, 2, -2, Fraction(3, 10), Fraction(11, 4)])
@pytest.mark.parametrize("xv", [0.4, 2.5, 11.0])
def test_hankel_matches_mpmath(nu, xv, x):
    nu_mp = mp.mpf(nu.numerator) / nu.denominator if isinstance(nu, Fraction) else nu
    for fn, mfn in [(ak.hankel1, mp.hankel1), (ak.hankel2, mp.hankel2)]:
        got = _ev(fn(nu, x, expand=False), {x: xv + 0j})
        want = complex(mfn(nu_mp, xv))
        assert abs(got - want) < 1e-11 * (1 + abs(want)), (fn.__name__, nu, xv, got, want)


@pytest.mark.parametrize("twice", [1, 3, 5, 9, -1, -3, -7])
def test_hankel_half_integer_closed_forms(twice, x):
    nu = Fraction(twice, 2)
    for fn, mfn in [(ak.hankel1, mp.hankel1), (ak.hankel2, mp.hankel2)]:
        e = fn(nu, x)  # expand=True: elementary
        s = str(e)
        assert "hankel" not in s
        for xv in (0.4, 1.7, 6.3):
            got = _ev(e, {x: xv + 0j})
            want = complex(mfn(mp.mpf(twice) / 2, xv))
            assert abs(got - want) < 1e-11 * (1 + abs(want)), (fn.__name__, nu, xv)


def test_hankel_known_closed_form(x):
    # H1_{1/2}(x) = -i sqrt(2/(pi x)) exp(ix)
    e = ak.hankel1(Fraction(1, 2), x)
    for xv in (0.5, 3.0):
        want = -1j * math.sqrt(2 / (math.pi * xv)) * complex(math.cos(xv), math.sin(xv))
        assert abs(_ev(e, {x: xv + 0j}) - want) < 1e-13


def test_hankel_is_j_plus_i_y(x):
    for nu in (0, 1, 3):
        for xv in (0.7, 4.2):
            h1 = _ev(ak.hankel1(nu, x), {x: xv + 0j})
            h2 = _ev(ak.hankel2(nu, x), {x: xv + 0j})
            j = float(mp.besselj(nu, xv))
            y = float(mp.bessely(nu, xv))
            assert abs(h1 - complex(j, y)) < 1e-12
            assert abs(h2 - complex(j, -y)) < 1e-12


def test_hankel_derivative_parse_roundtrip_and_reflection(x, pool):
    d = ak.diff(ak.hankel1(2, x), x).value
    xv = 1.3
    want = 0.5 * (complex(mp.hankel1(1, xv)) - complex(mp.hankel1(3, xv)))
    assert abs(_ev(d, {x: xv + 0j}) - want) < 1e-12
    # H_{-n} = (-1)^n H_n
    assert ak.simplify(ak.hankel1(-3, x, expand=False)) is not None
    got = _ev(ak.hankel2(-3, x, expand=False), {x: xv + 0j})
    assert abs(got - complex(mp.hankel2(-3, xv))) < 1e-12
    # derivative in the order is refused, never dropped
    nu = ak.symbol("nu", pool=pool)
    with pytest.raises(ak.DiffError) as err:
        ak.diff(ak.hankel1(nu, x), nu)
    assert err.value.code == "E-DIFF-001"
    # str/parse round trip
    e = ak.hankel2(2, x, expand=False)
    assert ak.parse(str(e), pool) == e


# ---------------------------------------------------------------------------
# zeta'(s)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", [-7.5, -2.5, -1.0, 0.0, 0.5, 2.0, 2.5, 3.0, 10.0])
def test_zeta_prime_matches_mpmath(s, x):
    e = ak.zeta_prime(x)
    want = float(mp.zeta(mp.mpf(s), derivative=1))
    got = _ev(e, {x: s})
    assert abs(got - want) < 1e-12 * (1 + abs(want))
    ball = ak.interval_eval(e, {x: ak.ArbBall(s, 0)}) if hasattr(ak, "ArbBall") else None
    if ball is not None:
        assert float(ball.lo) <= want <= float(ball.hi)


def test_zeta_prime_complex_argument(x):
    s = 0.5 + 14.134725141734693j  # near the first zero of zeta
    got = _ev(ak.zeta_prime(x), {x: s})
    want = complex(mp.zeta(mp.mpc(s.real, s.imag), derivative=1))
    assert abs(got - want) < 1e-9 * (1 + abs(want))


def test_zeta_prime_exact_values(pool):
    for n in (0, -1, 2, -2, -4, -6):
        e = ak.zeta_prime(pool.integer(n))
        assert "zeta_prime" not in str(e), n
        want = float(mp.zeta(n, derivative=1))
        assert abs(_ev(e, {}) - want) < 1e-12 * (1 + abs(want)), n
    # no closed form known: the node stays
    assert "zeta_prime" in str(ak.zeta_prime(pool.integer(3)))
    with pytest.raises(ak.SpecialFunctionError) as err:
        ak.zeta_prime(pool.integer(1))
    assert err.value.code == "E-SPECFN-006"


def test_diff_zeta_is_zeta_prime(x):
    d = ak.diff(ak.zeta(x), x).value
    assert d == ak.zeta_prime(x)
    d2 = ak.diff(ak.zeta(2 * x), x).value
    assert d2 == 2 * ak.zeta_prime(2 * x)
    # chain through the argument numerically
    xv = 1.7
    want = float(2 * mp.zeta(2 * mp.mpf(xv), derivative=1))
    assert abs(_ev(d2, {x: xv}) - want) < 1e-12
    # the second derivative has no node
    with pytest.raises(ak.DiffError) as err:
        ak.diff(ak.zeta_prime(x), x)
    assert err.value.code == "E-DIFF-001"


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def test_latex_notation(pool):
    x = ak.symbol("x", pool=pool)
    n = ak.symbol("n", pool=pool)
    a = ak.symbol("a", pool=pool)
    b = ak.symbol("b", pool=pool)
    cases = [
        (ak.jacobi_p(n, a, b, x), r"P_{n}^{(a,b)}\!\left(x\right)"),
        (ak.legendre_p(n, x), r"P_{n}\!\left(x\right)"),
        (ak.assoc_legendre_p(n, 2, x), r"P_{n}^{2}\!\left(x\right)"),
        (ak.hermite_h(n, x), r"H_{n}\!\left(x\right)"),
        (ak.laguerre_l(n, x), r"L_{n}\!\left(x\right)"),
        (ak.assoc_laguerre_l(n, a, x), r"L_{n}^{(a)}\!\left(x\right)"),
        (ak.chebyshev_t(n, x), r"T_{n}\!\left(x\right)"),
        (ak.gegenbauer_c(n, a, x), r"C_{n}^{(a)}\!\left(x\right)"),
        (ak.bessel_j(n, x), r"J_{n}\!\left(x\right)"),
        (ak.bessel_y(2, x), r"Y_{2}\!\left(x\right)"),
        (ak.bessel_i(n, x), r"I_{n}\!\left(x\right)"),
        (ak.bessel_k(n, x), r"K_{n}\!\left(x\right)"),
        (ak.hankel1(n, x), r"H^{(1)}_{n}\!\left(x\right)"),
        (ak.hankel2(n, x), r"H^{(2)}_{n}\!\left(x\right)"),
        (ak.airy_ai(x), r"\operatorname{Ai}\!\left(x\right)"),
        (ak.airy_bi_prime(x), r"\operatorname{Bi}'\!\left(x\right)"),
        (ak.zeta(x), r"\zeta\!\left(x\right)"),
        (ak.zeta_prime(x), r"\zeta'\!\left(x\right)"),
        (ak.hurwitz_zeta(n, x), r"\zeta\!\left(n, x\right)"),
        (ak.polylog(n, x), r"\operatorname{Li}_{n}\!\left(x\right)"),
        (ak.polygamma(n, x), r"\psi^{(n)}\!\left(x\right)"),
    ]
    for expr, want in cases:
        assert ak.latex(expr) == want


def test_new_names_are_exported():
    for name in ["hankel1", "hankel2", "zeta_prime", "real_spherical_harmonic_y"]:
        assert name in ak.__all__
