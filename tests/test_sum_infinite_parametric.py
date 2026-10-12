"""G6: infinite and parametric sums.

Repros from report 10-9 §3 G6 (multiple dosing, perpetuities, partition
functions, ζ-values): each was refused with ``E-SUM-001``/``E-SUM-002`` or —
for a parsed ``oo`` — silently returned an expression containing ``oo``.

Every closed form is checked numerically against an independent oracle
(mpmath ``nsum`` / direct partial sums, and SymPy ``summation`` where it
applies) at random parameter values inside the convergence region; divergent
series must be refused, and a conditional result must carry its ``|r| < 1``
hypothesis.
"""

import math
import random

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")

pool = ak.ExprPool()
P = ak.Domain.Positive
k = pool.symbol("k")
n = pool.symbol("n")
r = pool.symbol("r")
a = pool.symbol("a")
b = pool.symbol("b")
ke = pool.symbol("ke", P)
tau = pool.symbol("tau", P)
beta = pool.symbol("beta", P)
w = pool.symbol("w", P)
I = pool.integer  # noqa: E741
OO = pool.pos_infinity()
NEG = I(-1)
RNG = random.Random(20261011)


def _ev(expr, env=None):
    return ak.eval_expr(expr, env or {})


def _conds(res):
    return [c for st in res.steps for c in st.get("side_conditions", [])]


def _close(got, want, tol=1e-9):
    return abs(got - want) <= tol * max(1.0, abs(want))


# ---------------------------------------------------------------------------
# Geometric family, symbolic ratio
# ---------------------------------------------------------------------------


def test_geometric_to_infinity_is_conditional_on_abs_r_below_one():
    s = ak.sum_definite(r**k, k, I(0), OO)
    conds = _conds(s)
    assert conds
    assert any("abs(r)" in c for c in conds), conds
    for _ in range(5):
        rv = RNG.uniform(-0.95, 0.95)
        assert _close(_ev(s.value, {r: rv}), 1 / (1 - rv))
        want = float(mpmath.nsum(lambda j, rv=rv: rv**j, [0, mpmath.inf]))
        assert _close(_ev(s.value, {r: rv}), want)


def test_geometric_condition_discharged_or_refuted_by_assumptions():
    ctx = ak.Assumptions(pool)
    ctx.refine(pool.gt(r, I(0)))
    ctx.refine(pool.lt(r, I(1)))
    with ak.context(assumptions=ctx):
        s = ak.sum_definite(r**k, k, I(0), OO)
    assert _conds(s) == []
    bad = ak.Assumptions(pool)
    bad.refine(pool.gt(r, I(1)))
    with ak.context(assumptions=bad), pytest.raises(ak.SumError) as ei:
        ak.sum_definite(r**k, k, I(0), OO)
    assert "E-SUM-003" in str(ei.value)


@pytest.mark.parametrize("lo", [0, 1, 3])
def test_symbolic_lower_bound_and_shifted_start(lo):
    s = ak.sum_definite(r**k, k, I(lo), OO)
    rv = 0.37
    assert _close(_ev(s.value, {r: rv}), rv**lo / (1 - rv))
    m = pool.symbol("m")
    s = ak.sum_definite(r**k, k, m, OO)
    assert _close(_ev(s.value, {r: rv, m: 4.0}), rv**4 / (1 - rv))


def test_q_series_partition_like():
    q = pool.symbol("q")
    s = ak.sum_definite(q**k, k, I(0), OO)
    assert _close(_ev(s.value, {q: 0.5}), 2.0)


# ---------------------------------------------------------------------------
# Arithmetico-geometric Σ k^p r^k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_arithmetico_geometric_finite_matches_partial_sums(p):
    s = ak.sum_definite(k**p * r**k, k, I(0), n)
    for rv in (0.3, -0.8, 2.5):
        for nv in range(0, 8):
            want = sum(j**p * rv**j for j in range(nv + 1))
            assert _close(_ev(s.value, {r: rv, n: nv}), want), (p, rv, nv)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_arithmetico_geometric_infinite_matches_nsum(p):
    s = ak.sum_definite(k**p * r**k, k, I(1), OO)
    assert _conds(s), "|r| < 1 must be stated"
    for _ in range(4):
        rv = RNG.uniform(-0.9, 0.9)
        want = float(mpmath.nsum(lambda j, rv=rv: j**p * rv**j, [1, mpmath.inf]))
        assert _close(_ev(s.value, {r: rv}), want, 1e-8), (p, rv)


def test_arithmetico_geometric_against_sympy_exact():
    sp = pytest.importorskip("sympy")
    K, R = sp.symbols("k r")
    exact = sp.summation(K**2 * R**K, (K, 1, sp.oo)).args[0][0]  # the |r|<1 branch
    s = ak.sum_definite(k**2 * r**k, k, I(1), OO)
    for rv in (sp.Rational(1, 3), sp.Rational(-2, 5), sp.Rational(7, 9)):
        assert _close(_ev(s.value, {r: float(rv)}), float(exact.subs(R, rv)))


def test_numeric_ratio_is_exact():
    s = ak.sum_definite(k * pool.rational(1, 2) ** k, k, I(0), OO)
    assert s.value == I(2)
    s = ak.sum_definite((k**2 + 1) * pool.rational(1, 3) ** k, k, I(0), OO)
    # Σ k² 3^-k = 3/2, Σ 3^-k = 3/2
    assert s.value == I(3)


# ---------------------------------------------------------------------------
# Multiple dosing / steady state (report §4.1)
# ---------------------------------------------------------------------------


def test_multiple_dosing_superposition_and_steady_state():
    term = ak.exp(NEG * ke * k * tau)
    fin = ak.sum_definite(term, k, I(0), n - I(1))
    env = {ke: 0.3, tau: 8.0, n: 5}
    assert _close(_ev(fin.value, env), sum(math.exp(-0.3 * 8 * j) for j in range(5)))
    ss = ak.sum_definite(term, k, I(0), OO)
    assert _conds(ss) == [], "ke, tau > 0 decide convergence"
    assert _close(_ev(ss.value, {ke: 0.3, tau: 8.0}), 1 / (1 - math.exp(-2.4)))
    # Concentration after many doses: D/V·e^{-ke t}·Σ e^{-ke k τ}
    t = pool.symbol("t", P)
    conc = ak.sum_definite(ak.exp(NEG * ke * (t + k * tau)), k, I(0), n - I(1))
    env = {ke: 0.2, tau: 6.0, t: 1.5, n: 7}
    want = sum(math.exp(-0.2 * (1.5 + j * 6.0)) for j in range(7))
    assert _close(_ev(conc.value, env), want)


def test_steady_state_as_limit_of_partial_sums():
    ctx = ak.Assumptions(pool)
    ctx.refine(pool.gt(r, I(0)))
    ctx.refine(pool.lt(r, I(1)))
    with ak.context(assumptions=ctx):
        lim = ak.limit((r**n - 1) / (r - 1), n, OO).value
        assert _close(_ev(lim, {r: 0.25}), 1 / 0.75)
        assert ak.limit(r**n, n, OO).value == I(0)


# ---------------------------------------------------------------------------
# Partition function (report §4.3)
# ---------------------------------------------------------------------------


def test_harmonic_oscillator_partition_function():
    term = ak.exp(NEG * beta * w * (k + pool.rational(1, 2)))
    z = ak.sum_definite(term, k, I(0), OO)
    assert _conds(z) == []
    for bv, wv in ((0.5, 1.0), (2.0, 0.3), (1.3, 1.7)):
        want = 1 / (2 * math.sinh(bv * wv / 2))
        assert _close(_ev(z.value, {beta: bv, w: wv}), want)


# ---------------------------------------------------------------------------
# Rational summands
# ---------------------------------------------------------------------------


def test_telescoping_rational_series_are_exact():
    assert ak.sum_definite(1 / (k**2 + k), k, I(1), OO).value == I(1)
    assert ak.sum_definite(1 / (4 * k**2 - 1), k, I(1), OO).value == pool.rational(1, 2)
    s = ak.sum_definite(1 / (k * (k + 1) * (k + 2)), k, I(1), OO)
    assert s.value == pool.rational(1, 4)


@pytest.mark.parametrize("spelling", ["1/k**2", "k**-2", "(k**2)**-1"])
def test_basel_independent_of_spelling(spelling):
    term = {"1/k**2": 1 / k**2, "k**-2": k**-2, "(k**2)**-1": (k**2) ** -1}[spelling]
    s = ak.sum_definite(term, k, I(1), OO)
    assert _close(_ev(s.value), math.pi**2 / 6)
    assert "pi" in str(s.value)


def test_even_zeta_values_and_shifted_starts():
    assert _close(_ev(ak.sum_definite(1 / k**4, k, I(1), OO).value), math.pi**4 / 90)
    assert _close(_ev(ak.sum_definite(1 / k**6, k, I(1), OO).value), math.pi**6 / 945)
    s = ak.sum_definite(1 / k**2, k, I(4), OO)
    assert _close(_ev(s.value), float(mpmath.nsum(lambda j: 1 / j**2, [4, mpmath.inf])))
    s = ak.sum_definite(1 / (2 * k + 1) ** 2, k, I(0), OO)
    assert _close(_ev(s.value), math.pi**2 / 8)


def test_non_telescoping_rational_uses_log2_exactly():
    s = ak.sum_definite(1 / (k * (k + pool.rational(1, 2))), k, I(1), OO)
    assert "log" in str(s.value)
    assert "e-" not in str(s.value)
    assert _close(_ev(s.value), 4 - 4 * math.log(2))


def test_symbolic_shift_pair():
    s = ak.sum_definite(1 / ((k + a) * (k + b)), k, I(0), OO)
    conds = _conds(s)
    assert conds, "a ≠ b and the positivity of the shifts must be stated"
    for av, bv in ((0.3, 1.7), (2.0, 0.5)):
        want = float(
            mpmath.nsum(lambda j, av=av, bv=bv: 1 / ((j + av) * (j + bv)), [0, mpmath.inf])
        )
        assert _close(_ev(s.value, {a: av, b: bv}), want, 1e-8)
    s = ak.sum_definite(1 / ((k + a) * (k + a + 2)), k, I(0), OO)
    for av in (0.5, 3.25):
        assert _close(_ev(s.value, {a: av}), 0.5 * (1 / av + 1 / (av + 1)))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("term_fn", "lo", "code"),
    [
        (lambda: 1 / k, 1, "E-SUM-003"),  # harmonic: diverges
        (lambda: I(2) ** k, 0, "E-SUM-003"),  # |r| = 2
        (lambda: NEG**k, 0, "E-SUM-003"),  # Grandi
        (lambda: k / (k + 1), 0, None),  # terms do not tend to 0
        (lambda: 1 / (k * (k + 1)), 0, "E-SUM-003"),  # k = 0 term undefined
    ],
)
def test_divergent_or_unsupported_sums_are_refused(term_fn, lo, code):
    with pytest.raises(ak.SumError) as ei:
        ak.sum_definite(term_fn(), k, I(lo), OO)
    if code:
        assert code in str(ei.value)


def test_parsed_oo_is_infinity_not_a_symbol():
    oo = ak.parse("oo", pool)
    assert oo == OO
    s = ak.sum_definite(r**k, k, I(0), oo)
    assert "oo" not in str(s.value)
    with pytest.raises(ak.SumError):
        ak.sum_definite(r**k, k, ak.parse("-oo", pool), I(0))


def test_closed_forms_carry_no_false_lean_rewrite():
    # A summand equated to its sum (`r^k = 1/(1 − r)`) is not a rewrite; the
    # generic step emitter must withhold rather than certify it.
    for s in (
        ak.sum_definite(r**k, k, I(0), OO),
        ak.sum_definite(1 / (k**2 + k), k, I(1), OO),
        ak.sum_definite(k * r**k, k, I(0), n),
    ):
        assert s.certificate is None
