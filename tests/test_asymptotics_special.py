"""Stirling, Hankel, erfc/Ei/Airy asymptotics; ``loggamma``; ``gammasimp`` (F25).

Report 10-9 (G11): ``asymptotic_expand`` refused Stirling's series, ``J0`` and
``erfc`` (``E-ASYMPT-005``), there was no ``loggamma``, and products of
``gamma`` could not be rewritten by the reflection / duplication / recurrence
identities.

Oracles, computed inside the tests:

* SymPy ``series(f, x, oo, n)`` where it has the expansion (``loggamma``,
  ``digamma``, ``polygamma``, ``erfc``, ``besselk``, ``besseli``, ``Ei``):
  the difference between our partial sum and SymPy's must be ``o(last term)``,
  so every returned coefficient is SymPy's;
* mpmath for the rest (``J``, ``Y``, Airy, ``gamma``, ``factorial``,
  ``binomial``): at large ``x`` the truncation error of the ``n``-term sum is
  ``O(next term)`` and shrinks like it.
"""

import cmath
import math

import alkahest as ak
import pytest
from alkahest.experimental import asymptotic_expand

sp = pytest.importorskip("sympy")
mpmath = pytest.importorskip("mpmath")
from alkahest.crosscheck import to_sympy  # noqa: E402

mpmath.mp.dps = 60
X = sp.Symbol("x", positive=True)
N = sp.Symbol("n", positive=True)


@pytest.fixture
def px():
    p = ak.ExprPool()
    return p, p.symbol("x", "positive")


def _sym(expr):
    """SymPy form with every symbol named ``x``/``n`` replaced by one plain
    symbol, and the constant ``pi`` as SymPy's ``pi``."""
    s = to_sympy(expr)
    sub = {}
    for v in s.free_symbols:
        if v.name == "pi":
            sub[v] = sp.pi
        elif v.name in ("x", "n"):
            sub[v] = sp.Symbol(v.name, positive=True)
    return s.xreplace(sub)


def _mp(expr, x, xv):
    """Value of an Alkahest expression at ``x = xv`` (SymPy evalf, 60 digits)."""
    s = _sym(expr)
    return mpmath.mpf(sp.N(s.subs(dict.fromkeys(s.free_symbols, xv)), 60))


def _check_against_sympy(f_ak, f_sp, x, n, order):
    """Our ``n`` terms equal SymPy's: ``(ours - sympy)/last_term -> 0``."""
    terms = asymptotic_expand(f_ak, x, n)
    assert len(terms) == n, [str(t) for t in terms]
    xs = X
    ours = sum(_sym(t) for t in terms)
    # SymPy nests the O() inside a product for erfc/Bessel; drop it anywhere.
    theirs = sp.series(f_sp(xs), xs, sp.oo, order).replace(
        lambda e: isinstance(e, sp.Order), lambda e: sp.S.Zero
    )
    ratio = (ours - theirs) / _sym(terms[-1])
    for xv in (10**4, 10**6):
        r = sp.N(ratio.subs(xs, xv), 40)
        assert abs(r) < 1e-3, (xv, r, [str(t) for t in terms])
    return terms


def _check_numeric(f_ak, x, n, points=(200, 2000), min_terms=None):
    """``|f - Σ_{k<n}|`` is bounded by the first omitted term (computed with
    ``n + 1`` terms) and that bound is tight to a factor 10."""
    terms = asymptotic_expand(f_ak, x, n + 1)
    assert len(terms) == n + 1, [str(t) for t in terms]
    for xv in points:
        fv = _mp(f_ak, x, xv)
        partial = sum(_mp(t, x, xv) for t in terms[:n])
        nxt = abs(_mp(terms[n], x, xv))
        assert abs(fv - partial) <= 10 * nxt + mpmath.mpf(10) ** -50 * abs(fv), (xv, n)
    return terms[:n]


# ---------------------------------------------------------------------------
# The report's repros
# ---------------------------------------------------------------------------


def test_stirling_loggamma_matches_sympy(px):
    _p, x = px
    terms = _check_against_sympy(ak.loggamma(x), sp.loggamma, x, 6, 8)
    xs = X
    assert sp.simplify(_sym(terms[0]) - xs * sp.log(xs)) == 0


def test_gamma_stirling_coefficients(px):
    _p, x = px
    terms = _check_numeric(ak.gamma(x), x, 5)
    # Γ(x) ~ √(2π) x^{x−1/2} e^{−x} (1 + 1/(12x) + 1/(288x²) − 139/(51840x³) …)
    scale = sp.sqrt(2 * sp.pi) * X ** (X - sp.Rational(1, 2))
    scale *= sp.exp(-X)
    want = [1, sp.Rational(1, 12), sp.Rational(1, 288), sp.Rational(-139, 51840)]
    for k, w in enumerate(want):
        got = sp.simplify(_sym(terms[k]) / scale * X**k)
        assert got == w, (k, got)


def test_bessel_j0_hankel(px):
    _p, x = px
    terms = _check_numeric(ak.bessel_j(0, x), x, 4)
    xs = X
    lead = sp.sqrt(2) * sp.cos(xs - sp.pi / 4) / (sp.sqrt(sp.pi) * sp.sqrt(xs))
    assert sp.simplify(_sym(terms[0]) - lead) == 0


def test_erfc_matches_sympy(px):
    _p, x = px
    _check_against_sympy(ak.erfc(x), sp.erfc, x, 4, 9)


# ---------------------------------------------------------------------------
# Siblings — SymPy oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "sym", "n", "order"),
    [
        (lambda p, x: ak.digamma(x), sp.digamma, 5, 9),
        (lambda p, x: ak.polygamma(2, x), lambda t: sp.polygamma(2, t), 4, 8),
        (lambda p, x: ak.trigamma(x), lambda t: sp.polygamma(1, t), 4, 8),
        (
            lambda p, x: ak.loggamma(2 * x + 1),
            lambda t: sp.loggamma(2 * t + 1),
            6,
            6,
        ),
        (
            lambda p, x: ak.digamma(x + p.rational(1, 3)),
            lambda t: sp.digamma(t + sp.Rational(1, 3)),
            5,
            6,
        ),
        (lambda p, x: ak.exp_integral_ei(x), sp.Ei, 5, 6),
    ],
    ids=["digamma", "polygamma2", "trigamma", "loggamma_2x+1", "digamma_x+1/3", "Ei"],
)
def test_matches_sympy_series(px, build, sym, n, order):
    p, x = px
    _check_against_sympy(build(p, x), sym, x, n, order)


# ---------------------------------------------------------------------------
# Siblings — mpmath oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda p, x: ak.bessel_k(0, x),
        lambda p, x: ak.bessel_i(1, x),
        lambda p, x: ak.bessel_j(1, x),
        lambda p, x: ak.bessel_j(p.rational(1, 3), x),
        lambda p, x: ak.bessel_y(0, x),
        lambda p, x: ak.bessel_y(2, 3 * x),
        lambda p, x: ak.bessel_k(p.rational(2, 3), x),
        lambda p, x: ak.airy_ai(x),
        lambda p, x: ak.airy_bi(x),
        lambda p, x: ak.airy_ai_prime(x),
        lambda p, x: ak.airy_bi_prime(2 * x),
        lambda p, x: ak.erfc(3 * x),
        lambda p, x: ak.gamma(x + p.rational(1, 2)),
        lambda p, x: ak.gamma(2 * x) / ak.gamma(x) ** 2,
        lambda p, x: ak.sqrt(x) * ak.bessel_j(0, x),
        lambda p, x: ak.loggamma(x) - x * ak.log(x),
    ],
    ids=[
        "K0",
        "I1",
        "J1",
        "J_1/3",
        "Y0",
        "Y2(3x)",
        "K_2/3",
        "Ai",
        "Bi",
        "Ai'",
        "Bi'(2x)",
        "erfc(3x)",
        "gamma(x+1/2)",
        "gamma(2x)/gamma(x)^2",
        "sqrt(x)J0",
        "loggamma-xlogx",
    ],
)
def test_numeric_truncation_error(px, build):
    p, x = px
    _check_numeric(build(p, x), x, 3)


def test_modified_bessel_coefficients_are_dlmf(px):
    # SymPy 1.14's series(besselk(0, x), x, oo) has the signs of the 1/x
    # terms flipped (+1/(8x) where DLMF 10.40.2 has −1/(8x); mpmath agrees
    # with DLMF), so these are pinned to DLMF and checked numerically above.
    _p, x = px
    k0 = asymptotic_expand(ak.bessel_k(0, x), x, 4)
    lead = sp.sqrt(sp.pi / (2 * X)) * sp.exp(-X)
    for k, w in enumerate([1, sp.Rational(-1, 8), sp.Rational(9, 128), sp.Rational(-75, 1024)]):
        assert sp.simplify(_sym(k0[k]) / lead * X**k) == w, k
    i1 = asymptotic_expand(ak.bessel_i(1, x), x, 4)
    lead = sp.exp(X) / sp.sqrt(2 * sp.pi * X)
    for k, w in enumerate([1, sp.Rational(-3, 8), sp.Rational(-15, 128), sp.Rational(-105, 1024)]):
        assert sp.simplify(_sym(i1[k]) / lead * X**k) == w, k


def test_upper_incomplete_gamma(px):
    p, x = px
    # Γ(s, x) ~ x^{s-1} e^{-x} (1 + (s-1)/x + (s-1)(s-2)/x² + …)  (DLMF 8.11.2)
    _check_numeric(ak.uppergamma(p.rational(1, 3), x), x, 4)
    _check_numeric(ak.uppergamma(p.rational(7, 3), 2 * x), x, 3)
    # Half-integer s simplifies to e^{-x}·(…) + sqrt(pi)·erfc(sqrt(x)) pieces.
    _check_numeric(ak.uppergamma(p.rational(5, 2), x), x, 3)
    _check_numeric(ak.erfc(ak.sqrt(4 * x)), x, 3)
    # uppergamma(1/2, x) simplifies to sqrt(pi)*erfc(sqrt(x)): argument x^(1/2).
    _check_numeric(ak.uppergamma(p.rational(1, 2), x), x, 3)
    _check_numeric(ak.bessel_j(0, ak.sqrt(x)), x, 2)
    # A positive integer s terminates: Γ(3, x) = e^{-x}(x² + 2x + 2).
    terms = asymptotic_expand(ak.uppergamma(3, x), x, 6)
    assert len(terms) == 3
    total = sum(_sym(t) for t in terms)
    assert sp.simplify(total - sp.exp(-X) * (X**2 + 2 * X + 2)) == 0


def test_gammasimp_beta():
    p = ak.ExprPool()
    x, y = p.symbol("x", "positive"), p.symbol("y", "positive")
    assert ak.is_zero(ak.gammasimp(ak.beta(x, 1)) - 1 / x) is True
    lhs = ak.beta(x, y)
    rhs = ak.gamma(x) * ak.gamma(y) / ak.gamma(x + y)
    assert ak.is_zero(lhs - rhs) is True
    assert ak.gammasimp(ak.uppergamma(x, y)) == ak.uppergamma(x, y)


def test_factorial_and_central_binomial(px):
    p = ak.ExprPool()
    n = p.symbol("n", "positive")
    fact = p.func("factorial", [n])
    terms = asymptotic_expand(fact, n, 3)
    ns = N
    lead = sp.sqrt(2 * sp.pi) * ns ** (ns + sp.Rational(1, 2)) * sp.exp(-ns)
    assert sp.simplify(_sym(terms[0]) / lead) == 1
    assert sp.simplify(_sym(terms[1]) / lead * ns) == sp.Rational(1, 12)
    binom = p.func("binomial", [2 * n, n])
    terms = asymptotic_expand(binom, n, 3)
    # C(2n, n) ~ 4^n/√(πn) (1 − 1/(8n) + 1/(128n²) …)
    lead = 4**ns / sp.sqrt(sp.pi * ns)
    want = [1, sp.Rational(-1, 8), sp.Rational(1, 128)]
    for k, w in enumerate(want):
        assert sp.simplify(_sym(terms[k]) / lead * ns**k) == w, k
    for nv in (50, 400):
        exact = math.comb(2 * nv, nv)
        approx = sum(_mp(t, n, nv) for t in terms)
        assert abs(approx / exact - 1) < 2 / nv**3


def test_half_integer_bessel_is_exact(px):
    p, x = px
    # K_{1/2} has the finite expansion √(π/(2x)) e^{−x}: one term, exact.
    f = ak.bessel_k(p.rational(1, 2), x, expand=False)
    terms = asymptotic_expand(f, x, 4)
    assert len(terms) == 1
    xs = X
    assert sp.simplify(_sym(terms[0]) - sp.sqrt(sp.pi / (2 * xs)) * sp.exp(-xs)) == 0


def test_still_refuses_what_it_cannot_expand(px):
    p, x = px
    # Γ at −∞ (poles accumulate), a non-linear argument, a symbolic order.
    nu = p.symbol("nu")
    for f in (ak.gamma(-x), ak.gamma(x**2), ak.bessel_j(nu, x), 1 / ak.erfc(x)):
        with pytest.raises(ak.AsymptoticError):
            asymptotic_expand(f, x, 3)


# ---------------------------------------------------------------------------
# loggamma
# ---------------------------------------------------------------------------


def test_loggamma_values_vs_mpmath():
    p = ak.ExprPool()
    x = p.symbol("x")
    f = ak.loggamma(x)
    for v in (0.1, 0.5, 1.5, 3.7, 25.0, 170.5, 1e5):
        got = ak.eval_expr(f, {x: v})
        want = float(mpmath.loggamma(v))
        assert got == pytest.approx(want, rel=1e-14, abs=1e-15), v
    # Principal branch off the positive axis — not log(gamma(x)).
    for z in (-0.5 + 0j, -2.5 + 0j, 1 + 2j, -3.2 - 0.7j):
        r = ak.evaluate(f, {x: z}, mode="complex")
        assert r.status == "ok", (z, r.reason)
        got = complex(r.value)
        want = complex(mpmath.loggamma(z))
        assert cmath.isclose(got, want, rel_tol=1e-13), z
    # Real evaluation of a negative argument declines instead of returning
    # log|Γ(x)|.
    with pytest.raises(Exception):
        ak.eval_expr(f, {x: -0.5})


def test_loggamma_exact_values_derivative_and_refusal():
    p = ak.ExprPool()
    x = p.symbol("x")
    assert ak.loggamma(p.integer(1)) == p.integer(0)
    assert ak.loggamma(p.integer(2)) == p.integer(0)
    assert ak.loggamma(p.integer(5)) == ak.log(p.integer(24))
    half = ak.loggamma(p.rational(1, 2))
    assert ak.eval_expr(half, {}) == pytest.approx(0.5 * math.log(math.pi), rel=1e-15)
    assert ak.diff(ak.loggamma(x), x).value == ak.digamma(x)
    with pytest.raises(ak.SpecialFunctionError) as ei:
        ak.loggamma(p.integer(-2))
    assert ei.value.code == "E-SPECFN-006"
    # Parses and round-trips; LaTeX.
    assert ak.parse("loggamma(x)", p) == ak.loggamma(x)
    assert "\\log\\Gamma" in ak.latex(ak.loggamma(x))


def test_loggamma_ball_contains_mpmath():
    p = ak.ExprPool()
    x = p.symbol("x")
    b = ak.interval_eval(ak.loggamma(x), {x: ak.ArbBall(7.25, 0.0)})
    want = mpmath.loggamma(mpmath.mpf("7.25"))
    assert b.lo <= want <= b.hi


# ---------------------------------------------------------------------------
# gammasimp
# ---------------------------------------------------------------------------


def _num_equal(a, b, x, pts=(0.3, 1.7, 2.45, 4.1)):
    for v in pts:
        assert ak.eval_expr(a, {x: v}) == pytest.approx(ak.eval_expr(b, {x: v}), rel=1e-11)


def test_gammasimp_reflection_duplication_recurrence():
    p = ak.ExprPool()
    x = p.symbol("x")
    pi = p.symbol("pi")
    one = p.integer(1)
    refl = ak.gammasimp(ak.gamma(x) * ak.gamma(one - x))
    assert ak.is_zero(refl - pi / ak.sin(pi * x)) is True
    dup = ak.gamma(x) * ak.gamma(x + p.rational(1, 2))
    g = ak.gammasimp(dup)
    _num_equal(g, dup, x)
    assert str(g).count("gamma") == 1
    rec = ak.gammasimp(ak.gamma(x + 3) / ak.gamma(x))
    assert ak.is_zero(rec - x * (x + 1) * (x + 2)) is True
    inv = ak.gammasimp(ak.gamma(x) / ak.gamma(x + 2))
    assert ak.is_zero(inv - 1 / (x * (x + 1))) is True
    # Shifted reflection Γ(x)Γ(3 − x) = (1−x)(2−x) π/sin(πx).
    sh = ak.gamma(x) * ak.gamma(3 - x)
    _num_equal(ak.gammasimp(sh), sh, x)
    # Nothing to do: unchanged.
    assert ak.gammasimp(ak.gamma(x) * ak.gamma(x + 1)) == ak.gamma(x) * ak.gamma(x + 1)


def test_is_zero_proves_gamma_identities():
    p = ak.ExprPool()
    x = p.symbol("x")
    pi = p.symbol("pi")
    half = p.rational(1, 2)
    assert ak.is_zero(ak.gamma(x) * ak.gamma(1 - x) - pi / ak.sin(pi * x)) is True
    lhs = ak.gamma(x) * ak.gamma(x + half)
    rhs = p.integer(2) ** (1 - 2 * x) * ak.sqrt(pi) * ak.gamma(2 * x)
    assert ak.is_zero(lhs - rhs) is True
    assert ak.is_zero(ak.gamma(x + 1) - x * ak.gamma(x)) is True
    # And a false identity is not proved.
    assert ak.is_zero(ak.gamma(x) * ak.gamma(1 - x) - pi / ak.cos(pi * x)) is not True
