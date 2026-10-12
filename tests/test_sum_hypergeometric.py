"""F12: infinite sums of hypergeometric terms, polylogarithms, quadratic poles.

G6 (``tests/test_sum_infinite_parametric.py``) summed polynomial × geometric
and rational series with linear poles; these were still refused with
``E-SUM-002``:

* ``Σ_{k≥0} qᵏ/k!`` (→ ``exp(q)``), ``Σ_{k≥1} rᵏ/k`` (→ ``−log(1 − r)``),
  ``Σ_{k≥1} rᵏ/k²`` (→ ``Li₂(r)``), and hypergeometric terms in general
  (→ ``t₀·ₚFq(a; b; z)``, expanded when a closed form is known);
* ``Σ 1/(k² + a²)`` and other rational series with irreducible quadratic
  denominators (→ ``coth``/``tanh`` closed forms).

Oracles: mpmath ``nsum`` / ``hyper`` at random parameter values inside the
convergence region, SymPy ``summation`` where it finds a closed form.
Divergent series keep their ``E-SUM-003`` refusal; conditional results carry
their convergence hypothesis.
"""

import math
import random

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")

pool = ak.ExprPool()
k = pool.symbol("k")
r = pool.symbol("r")
q = pool.symbol("q")
x = pool.symbol("x")
a = pool.symbol("a")
I = pool.integer  # noqa: E741
OO = pool.pos_infinity()
RNG = random.Random(20261012)


def _ev(expr, env=None):
    return ak.eval_expr(expr, env or {})


def _conds(res):
    return [c for st in res.steps for c in st.get("side_conditions", [])]


def _close(got, want, tol=1e-9):
    return abs(got - want) <= tol * max(1.0, abs(want))


def fact(e):
    return ak.gamma(e + 1)


def _nsum(f, lo):
    return float(mpmath.nsum(f, [lo, mpmath.inf]))


# ---------------------------------------------------------------------------
# The report's repros
# ---------------------------------------------------------------------------


def test_exponential_series_is_exp():
    s = ak.sum_definite(q**k / fact(k), k, I(0), OO)
    assert str(s.value) == "exp(q)"
    assert _conds(s) == []  # entire: no hypothesis


def test_log_series():
    s = ak.sum_definite(r**k / k, k, I(1), OO)
    assert "log" in str(s.value)
    assert any("abs(r)" in c for c in _conds(s))
    for _ in range(5):
        rv = RNG.uniform(-0.95, 0.95)
        assert _close(_ev(s.value, {r: rv}), -math.log(1 - rv))


@pytest.mark.parametrize("s_ord", [2, 3, 4])
def test_polylog_series(s_ord):
    s = ak.sum_definite(r**k / k**s_ord, k, I(1), OO)
    assert _conds(s), "|r| < 1 must be stated"
    for _ in range(4):
        rv = RNG.uniform(-0.9, 0.9)
        want = float(mpmath.polylog(s_ord, rv))
        assert _close(_ev(s.value, {r: rv}), want), (s_ord, rv)


def test_dilog_matches_sympy():
    sp = pytest.importorskip("sympy")
    K, R = sp.symbols("k r")
    exact = sp.summation(R**K / K**2, (K, 1, sp.oo))
    s = ak.sum_definite(r**k / k**2, k, I(1), OO)
    for rv in (sp.Rational(1, 3), sp.Rational(-1, 2), sp.Rational(4, 5)):
        want = float(sp.N(exact.subs(R, rv).rewrite(sp.polylog), 30))
        assert _close(_ev(s.value, {r: float(rv)}), want, 1e-12)


def test_alternating_harmonic_at_the_boundary():
    s = ak.sum_definite(I(-1) ** k / k, k, I(1), OO)
    assert _close(_ev(s.value), -math.log(2), 1e-14)
    assert _conds(s) == []


# ---------------------------------------------------------------------------
# Rational × geometric (polylog family)
# ---------------------------------------------------------------------------


RATGEO = [
    ("r^k/(k(k+1))", lambda: r**k / (k * (k + 1)), lambda j, rv: rv**j / (j * (j + 1)), 1),
    ("k r^k/(k+2)^2", lambda: k * r**k / (k + 2) ** 2, lambda j, rv: j * rv**j / (j + 2) ** 2, 0),
    ("(k^2+1) r^k/k", lambda: (k**2 + 1) * r**k / k, lambda j, rv: (j**2 + 1) * rv**j / j, 1),
    ("r^k/(k+3)^3", lambda: r**k / (k + 3) ** 3, lambda j, rv: rv**j / (j + 3) ** 3, 0),
]


@pytest.mark.parametrize(("name", "term", "f", "lo"), RATGEO, ids=[c[0] for c in RATGEO])
def test_rational_times_geometric_matches_nsum(name, term, f, lo):
    s = ak.sum_definite(term(), k, I(lo), OO)
    assert _conds(s), name
    for _ in range(4):
        rv = RNG.uniform(-0.85, 0.85)
        want = _nsum(lambda j, rv=rv: f(j, rv), lo)
        assert _close(_ev(s.value, {r: rv}), want, 1e-8), (name, rv)


def test_numeric_ratio_needs_no_condition():
    s = ak.sum_definite(pool.rational(1, 2) ** k / k, k, I(1), OO)
    assert _conds(s) == []
    assert _close(_ev(s.value), math.log(2), 1e-14)


# ---------------------------------------------------------------------------
# General hypergeometric terms → pFq
# ---------------------------------------------------------------------------


def test_cosh_series():
    s = ak.sum_definite(x ** (2 * k) / fact(2 * k), k, I(0), OO)
    assert "hyper" not in str(s.value)
    for xv in (0.4, 1.3, -2.1):
        assert _close(_ev(s.value, {x: xv}), math.cosh(xv), 1e-12)


def test_leibniz_series_is_pi_over_4():
    s = ak.sum_definite(I(-1) ** k / (2 * k + 1), k, I(0), OO)
    assert _close(_ev(s.value), math.pi / 4, 1e-14)


def test_central_binomial_series():
    # binomial(2k, k) = Γ(2k+1)/Γ(k+1)²
    t = fact(2 * k) / fact(k) ** 2 * x**k
    s = ak.sum_definite(t, k, I(0), OO)
    assert _conds(s), "|4x| < 1 must be stated"
    for xv in (0.05, -0.2, 0.2):
        assert _close(_ev(s.value, {x: xv}), 1 / math.sqrt(1 - 4 * xv), 1e-12)


def test_k_times_exponential_series():
    s = ak.sum_definite(k * x**k / fact(k), k, I(0), OO)
    for xv in (0.5, -1.5, 2.0):
        assert _close(_ev(s.value, {x: xv}), xv * math.exp(xv), 1e-12)


def test_hyper_node_when_no_closed_form():
    # Σ (k!)³/(3k)! = ₃F₂(1, 1, 1; 1/3, 2/3; 1/27): kept as the exact node.
    s = ak.sum_definite(fact(k) ** 3 / fact(3 * k), k, I(0), OO)
    assert "hyper" in str(s.value)
    want = _nsum(lambda j: mpmath.factorial(j) ** 3 / mpmath.factorial(3 * j), 0)
    assert _close(_ev(s.value), want, 1e-12)


def test_bessel_like_series_matches_mpmath():
    # I₀(2√x): exact for every x, real-evaluable for x > 0.
    s = ak.sum_definite(x**k / fact(k) ** 2, k, I(0), OO)
    for xv in (0.3, 2.0, 5.5):
        want = float(mpmath.hyp0f1(1, xv))
        assert _close(_ev(s.value, {x: xv}), want, 1e-12)
    # A negative literal argument comes back through J₀.
    s = ak.sum_definite(pool.rational(-6, 5) ** k / fact(k) ** 2, k, I(0), OO)
    assert _close(_ev(s.value), float(mpmath.hyp0f1(1, -1.2)), 1e-12)


def test_random_hypergeometric_terms_vs_nsum():
    # t_k = Γ(k + α)/Γ(k + β) · zᵏ / k!  (₁F₁-type, entire) at random α, β, z.
    for _ in range(4):
        al = RNG.choice([pool.rational(1, 3), pool.rational(3, 2), I(2)])
        be = RNG.choice([pool.rational(5, 2), I(3), pool.rational(7, 4)])
        zv = RNG.uniform(-2, 2)
        z = pool.rational(round(zv * 8), 8)
        t = ak.gamma(k + al) / ak.gamma(k + be) * z**k / fact(k)
        s = ak.sum_definite(t, k, I(0), OO)
        alf, bef, zf = _ev(al), _ev(be), _ev(z)
        want = _nsum(
            lambda j, alf=alf, bef=bef, zf=zf: (
                mpmath.gamma(j + alf)
                / mpmath.gamma(j + bef)
                * mpmath.mpf(zf) ** j
                / mpmath.factorial(j)
            ),
            0,
        )
        assert _close(_ev(s.value), want, 1e-10), (str(t), str(s.value))


def test_gauss_boundary_convergence():
    # Σ Γ(k+1/2)/Γ(k+5/2) is ₂F₁(1/2, 1; 5/2; 1) up to t₀: z = 1 and
    # c − a − b = 1 > 0, so it converges (Gauss).
    t = ak.gamma(k + pool.rational(1, 2)) / ak.gamma(k + pool.rational(5, 2))
    s = ak.sum_definite(t, k, I(0), OO)
    want = _nsum(lambda j: mpmath.gamma(j + 0.5) / mpmath.gamma(j + 2.5), 0)
    assert _close(_ev(s.value), want, 1e-10)


# ---------------------------------------------------------------------------
# Quadratic denominators
# ---------------------------------------------------------------------------


def test_one_over_k2_plus_a2():
    s = ak.sum_definite(1 / (k**2 + a**2), k, I(1), OO)
    assert "tanh" in str(s.value)
    assert _conds(s), "a^2 > 0 must be stated"
    for _ in range(5):
        av = RNG.uniform(0.2, 3.0) * RNG.choice([1, -1])
        want = (math.pi * av / math.tanh(math.pi * av) - 1) / (2 * av * av)
        assert _close(_ev(s.value, {a: av}), want, 1e-12), av


def test_one_over_k2_plus_a2_positive_parameter_has_no_condition():
    ap = pool.symbol("ap", ak.Domain.Positive)
    s = ak.sum_definite(1 / (k**2 + ap**2), k, I(0), OO)
    assert _conds(s) == []
    want = (1 + math.pi * 1.7 / math.tanh(math.pi * 1.7)) / (2 * 1.7**2)
    assert _close(_ev(s.value, {ap: 1.7}), want, 1e-12)


def test_one_over_k2_plus_1_matches_sympy():
    sp = pytest.importorskip("sympy")
    K = sp.Symbol("k")
    want = float(sp.N(sp.summation(1 / (K**2 + 1), (K, 0, sp.oo)), 30))
    s = ak.sum_definite(1 / (k**2 + 1), k, I(0), OO)
    assert _close(_ev(s.value), want, 1e-14)


QUAD = [
    ("1/(k^2+k+1)", lambda: 1 / (k**2 + k + 1), lambda j: 1 / (j**2 + j + 1), 0),
    ("1/((k+1)^2+4)", lambda: 1 / ((k + 1) ** 2 + 4), lambda j: 1 / ((j + 1) ** 2 + 4), 0),
    ("k/(k^4+4)", lambda: k / (k**4 + 4), lambda j: j / (j**4 + 4), 1),
    ("1/(4k^2+1)", lambda: 1 / (4 * k**2 + 1), lambda j: 1 / (4 * j**2 + 1), 0),
    ("1/(k^2(k^2+1))", lambda: 1 / (k**2 * (k**2 + 1)), lambda j: 1 / (j**2 * (j**2 + 1)), 1),
]


@pytest.mark.parametrize(("name", "term", "f", "lo"), QUAD, ids=[c[0] for c in QUAD])
def test_quadratic_denominators_match_nsum(name, term, f, lo):
    s = ak.sum_definite(term(), k, I(lo), OO)
    assert "digamma" not in str(s.value), str(s.value)
    want = _nsum(f, lo)
    assert _close(_ev(s.value), want, 1e-12), (name, str(s.value))


# ---------------------------------------------------------------------------
# Refusals keep their codes
# ---------------------------------------------------------------------------


REFUSED = [
    ("2^k/k", lambda: I(2) ** k / k, 1, "E-SUM-003"),
    ("r^k/k at r=1 is the harmonic series", lambda: I(1) / k, 1, "E-SUM-003"),
    ("k! x^k (2F0)", lambda: fact(k) * x**k, 0, "E-SUM-003"),
    ("(-1)^k k/(k+1)", lambda: I(-1) ** k * k / (k + 1), 1, "E-SUM-003"),
    ("(1/2)^k/(k-3): pole at k=3", lambda: pool.rational(1, 2) ** k / (k - 3), 1, "E-SUM-003"),
    ("k/(k^2+1) ~ 1/k", lambda: k / (k**2 + 1), 1, "E-SUM-003"),
    # Re ψ(1 + i) has no elementary form: not guessed.
    ("1/((k+1)(k^2+1))", lambda: 1 / ((k + 1) * (k**2 + 1)), 0, "E-SUM-002"),
    ("1/(k^4+1)", lambda: 1 / (k**4 + 1), 0, "E-SUM-002"),
]


@pytest.mark.parametrize(("name", "term", "lo", "code"), REFUSED, ids=[c[0] for c in REFUSED])
def test_refusals_keep_their_codes(name, term, lo, code):
    with pytest.raises(ak.SumError) as ei:
        ak.sum_definite(term(), k, I(lo), OO)
    assert code in str(ei.value), (name, str(ei.value))


def test_symbolic_ratio_outside_unit_disc_is_refused():
    bad = ak.Assumptions(pool)
    bad.refine(pool.gt(r, I(1)))
    with ak.context(assumptions=bad), pytest.raises(ak.SumError) as ei:
        ak.sum_definite(r**k / k**2, k, I(1), OO)
    assert "E-SUM-003" in str(ei.value)
    good = ak.Assumptions(pool)
    good.refine(pool.gt(r, I(0)))
    good.refine(pool.lt(r, I(1)))
    with ak.context(assumptions=good):
        s = ak.sum_definite(r**k / k**2, k, I(1), OO)
    assert _conds(s) == []


def test_closed_forms_carry_no_false_lean_rewrite():
    # `qᵏ/k! = exp(q)` equates a summand to its sum: not a rewrite to certify.
    for s in (
        ak.sum_definite(q**k / fact(k), k, I(0), OO),
        ak.sum_definite(r**k / k**2, k, I(1), OO),
        ak.sum_definite(fact(k) ** 3 / fact(3 * k), k, I(0), OO),
        ak.sum_definite(1 / (k**2 + 1), k, I(0), OO),
    ):
        assert s.certificate is None
