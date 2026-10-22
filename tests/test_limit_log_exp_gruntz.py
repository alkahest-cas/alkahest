"""Gruntz leading terms through ``log`` and ``exp`` (report10-21 N3, C1, N12).

* **N3** — ``limit(log(exp(-2*t))/t, t, oo)`` returned ``0`` (true ``-2``); so
  did ``log(e^{-2t} + e^{-3t})/t``, the Bateman terminal slope, and
  ``(e^{-2t} + e^{-3t})^{1/t}`` came back as ``1`` (true ``e^{-2}``). The MRV
  rewrite turned ``log(e^{-2t})`` into ``log ω`` and handed it to a Taylor
  expansion in ω at 0, where ``log`` has a branch point.
* **C1** — ``limit(log(exp(t))/t, t, oo)`` aborted the interpreter on a 14 GB
  allocation: the same Taylor loop on ``log(ω⁻¹)`` differentiated without bound.
* The rewrite also dropped the lower-class factor ``exp(h_e − c·h_ω)``:
  ``e^{1−3t}/(e^{−3t} + e^{−4t})`` came back as ``0`` (true ``e``).
* **N12** — the Black–Scholes price at ``T → 0`` with symbolic ``S, K`` came
  back as a "value" containing ``(σ·√0)⁻¹``.

Every answer is checked against an oracle: the closed form, SymPy, or a
high-precision mpmath evaluation far out on the curve.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from fractions import Fraction

import alkahest as ak
import pytest
from limit_at_infinity_corpus import CORPUS

TIMEOUT = 120


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def t(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("t", "real")


def _lim(expr: ak.Expr, var: ak.Expr, pool: ak.ExprPool) -> ak.Expr:
    return ak.limit(expr, var, pool.pos_infinity()).value


def _num(e: ak.Expr) -> float:
    return float(ak.eval_expr(e, {}))


def _q(pool: ak.ExprPool, f: Fraction) -> ak.Expr:
    return (
        pool.rational(f.numerator, f.denominator)
        if f.denominator != 1
        else pool.integer(f.numerator)
    )


# ---------------------------------------------------------------------------
# The report's repros
# ---------------------------------------------------------------------------


@pytest.mark.timeout(TIMEOUT)
def test_log_of_exp_over_t_is_the_rate(pool, t):
    """N3: λz = −2, not 0. Oracle: log(e^{−100})/50 = −2."""
    got = _lim(ak.log(ak.exp(-2 * t)) / t, t, pool)
    assert got == pool.integer(-2), f"got {got}"


@pytest.mark.timeout(TIMEOUT)
def test_log_of_exponential_sum_over_t_is_the_slowest_rate(pool, t):
    got = _lim(ak.log(ak.exp(-2 * t) + ak.exp(-3 * t)) / t, t, pool)
    assert got == pool.integer(-2), f"got {got}"


@pytest.mark.timeout(TIMEOUT)
def test_exponential_sum_to_the_one_over_t(pool, t):
    got = _lim((ak.exp(-2 * t) + ak.exp(-3 * t)) ** (1 / t), t, pool)
    assert math.isclose(_num(got), math.exp(-2), rel_tol=1e-12), f"got {got}"


@pytest.mark.timeout(TIMEOUT)
def test_mrv_factor_is_not_dropped(pool, t):
    """``e^{1−3t}`` is ``e·ω^{3/2}`` for ω = e^{−2t}: the ``e`` is the answer."""
    got = _lim(ak.exp(1 - 3 * t) / (ak.exp(-3 * t) + ak.exp(-4 * t)), t, pool)
    assert math.isclose(_num(got), math.e, rel_tol=1e-12), f"got {got}"


@pytest.mark.timeout(TIMEOUT)
def test_bateman_terminal_slope_numeric_rates(pool, t):
    """``log C(t)/t → −k_e`` for the one-compartment oral-dose curve; the
    mpmath oracle at t = 2000 is −0.39982 for k_a = 13/10, k_e = 2/5."""
    ka, ke = pool.rational(13, 10), pool.rational(2, 5)
    c = ka / (ka - ke) * (ak.exp(-ke * t) - ak.exp(-ka * t))
    got = _lim(ak.log(c) / t, t, pool)
    assert got == pool.rational(-2, 5), f"got {got}"


@pytest.mark.timeout(TIMEOUT)
def test_bateman_terminal_slope_symbolic_rates_needs_their_order(pool, t):
    """With symbolic rates the slope is −min(k_a, k_e): refused until the
    order is stated, then −k_e under k_a > k_e."""
    ka = pool.symbol("ka", "positive")
    ke = pool.symbol("ke", "positive")
    c = ka / (ka - ke) * (ak.exp(-1 * ke * t) - ak.exp(-1 * ka * t))
    expr = ak.log(c) / t
    with pytest.raises(ak.AlkahestError):
        _lim(expr, t, pool)
    ctx = ak.Assumptions(pool)
    ctx.refine(pool.gt(ka, ke))
    with ak.context(assumptions=ctx):
        got = _lim(expr, t, pool)
    assert got == -1 * ke or str(got) == str(-1 * ke), f"got {got}"


# ---------------------------------------------------------------------------
# C1: these used to abort the interpreter or run for minutes. Each runs in a
# child process so a regression fails this test instead of killing pytest.
# ---------------------------------------------------------------------------

_CHILD = r"""
import alkahest as ak
p = ak.ExprPool(); t = p.symbol("t", "real")
e = {expr}
print(ak.limit(e, t, p.pos_infinity()).value)
"""


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("ak.log(ak.exp(t))/t", "1"),
        ("ak.log(ak.exp(2*t))/t", "2"),
        ("ak.log(ak.exp(t)+1)/t", "1"),
        ("ak.log(ak.exp(2*t)+1)/t", "2"),
        ("ak.log(ak.exp(-t)-ak.exp(-2*t))/t", "-1"),
    ],
)
def test_crash_cases_finish_quickly_and_correctly(expr, expected):
    r = subprocess.run(
        [sys.executable, "-c", _CHILD.format(expr=expr)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"{expr}: rc={r.returncode} {r.stderr[-400:]}"
    assert r.stdout.strip() == expected, f"{expr}: got {r.stdout.strip()}"


# ---------------------------------------------------------------------------
# Randomised oracle: log(Σ cᵢ e^{aᵢt})/t → max aᵢ and (Σ cᵢ e^{aᵢt})^{1/t} → e^{max aᵢ}
# ---------------------------------------------------------------------------


def _random_exp_sum(pool, t, rng):
    n = rng.randint(1, 4)
    rates = rng.sample([Fraction(k, d) for k in range(-6, 7) for d in (1, 2, 3)], n)
    rates = sorted(set(rates), reverse=True)
    terms = []
    for i, a in enumerate(rates):
        # The leading coefficient is positive so the sum is eventually positive;
        # the others may have either sign.
        c = Fraction(rng.randint(1, 9), rng.randint(1, 4))
        if i > 0 and rng.random() < 0.4:
            c = -c
        terms.append(_q(pool, c) * ak.exp(_q(pool, a) * t))
    s = terms[0]
    for term in terms[1:]:
        s = s + term
    return s, rates[0]


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("seed", range(12))
def test_random_log_exponential_sum_over_t(pool, t, seed):
    rng = random.Random(seed)
    s, top = _random_exp_sum(pool, t, rng)
    got = _lim(ak.log(s) / t, t, pool)
    assert got == _q(pool, top), f"seed {seed}: log({s})/t -> {got}, expected {top}"


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("seed", range(12, 20))
def test_random_exponential_sum_to_the_one_over_t(pool, t, seed):
    rng = random.Random(seed)
    s, top = _random_exp_sum(pool, t, rng)
    got = _lim(s ** (1 / t), t, pool)
    assert math.isclose(_num(got), math.exp(top), rel_tol=1e-12), (
        f"seed {seed}: ({s})^(1/t) -> {got}, expected e^{top}"
    )


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("seed", range(20, 28))
def test_random_shifted_exponent_coefficient(pool, t, seed):
    """``e^{a t + b}/(Σ cᵢ e^{aᵢ t})`` with ``a`` the top rate: the limit is
    ``e^b / c_top`` — exactly the factor the rewrite used to drop."""
    rng = random.Random(seed)
    s, top = _random_exp_sum(pool, t, rng)
    b = Fraction(rng.randint(-5, 5), rng.randint(1, 3))
    num = ak.exp(_q(pool, top) * t + _q(pool, b))
    got = _lim(num / s, t, pool)
    # leading coefficient of s: the coefficient of e^{top t}
    lead = _lim(s * ak.exp(-1 * _q(pool, top) * t), t, pool)
    assert math.isclose(_num(got), math.exp(b) / _num(lead), rel_tol=1e-10), f"seed {seed}: {got}"


# ---------------------------------------------------------------------------
# N12: no unevaluable node is ever returned as a value
# ---------------------------------------------------------------------------


def _black_scholes(pool):
    S, K, s, T = (pool.symbol(n, "positive") for n in ("S", "K", "sigma", "T"))

    def nc(z):
        return pool.rational(1, 2) * (1 + ak.erf(z / ak.sqrt(pool.integer(2))))

    d1 = (ak.log(S / K) + (s**2 / 2) * T) / (s * ak.sqrt(T))
    d2 = d1 - s * ak.sqrt(T)
    return S * nc(d1) - K * nc(d2), (S, K, s, T)


@pytest.mark.timeout(TIMEOUT)
def test_black_scholes_expiry_limit_is_never_an_unevaluable_value(pool):
    c, (S, K, s, T) = _black_scholes(pool)
    try:
        got = ak.limit(c, T, pool.integer(0)).value
    except ak.AlkahestError:
        return  # an honest refusal: the answer depends on the sign of log(S/K)
    text = str(got)
    assert "sqrt(0)" not in text, text
    assert "0^-1" not in text, text
    # Whatever came back must be max(S − K, 0) at both moneyness sides.
    for sv, kv, want in ((100, 95, 5.0), (90, 95, 0.0)):
        v = _num(ak.subs(got, {S: pool.integer(sv), K: pool.integer(kv), s: pool.rational(1, 4)}))
        assert math.isclose(v, want, abs_tol=1e-9), (sv, kv, text)


@pytest.mark.timeout(TIMEOUT)
def test_black_scholes_numeric_strikes_still_resolve(pool):
    c, (S, K, s, T) = _black_scholes(pool)
    for sv, want in ((100, 5), (90, 0)):
        cn = ak.subs(c, {S: pool.integer(sv), K: pool.integer(95), s: pool.rational(1, 4)})
        assert ak.limit(cn, T, pool.integer(0)).value == pool.integer(want)


@pytest.mark.timeout(TIMEOUT)
def test_zero_inside_a_product_to_a_negative_power_is_refused(pool):
    """The general shape behind N12: a parameter times a vanishing factor,
    inverted, is not a value whatever the parameter is."""
    x = pool.symbol("x", "real")
    a = pool.symbol("a", "positive")
    with pytest.raises(ak.AlkahestError):
        ak.limit(ak.erf(1 / (a * ak.sqrt(x))), x, pool.integer(0))


# ---------------------------------------------------------------------------
# Corpus at +∞ against SymPy
# ---------------------------------------------------------------------------


def _to_sympy(e, sympy):
    n = e.node()
    tag = n[0]
    if tag == "symbol":
        return sympy.oo if n[1] == "∞" else sympy.Symbol(n[1], positive=True)
    if tag == "integer":
        return sympy.Integer(int(n[1]))
    if tag == "rational":
        return sympy.Rational(int(n[1]), int(n[2]))
    if tag == "add":
        return sympy.Add(*[_to_sympy(c, sympy) for c in n[1]])
    if tag == "mul":
        return sympy.Mul(*[_to_sympy(c, sympy) for c in n[1]])
    if tag == "pow":
        return sympy.Pow(_to_sympy(n[1], sympy), _to_sympy(n[2], sympy))
    if tag == "func":
        return getattr(sympy, n[1])(*[_to_sympy(a, sympy) for a in n[2]])
    raise AssertionError(f"unknown node {tag}")


@pytest.mark.timeout(600)
def test_limit_corpus_at_infinity_against_sympy():
    """No wrong answer anywhere in the corpus; a refusal is allowed, and the
    number of them is pinned so the engine cannot quietly regress to refusing."""
    sympy = pytest.importorskip("sympy")
    x_sp = sympy.Symbol("x", positive=True)
    p = ak.ExprPool()
    x = p.symbol("x")
    wrong, refused = [], []
    for src in CORPUS:
        e = ak.parse(src.replace("**", "^"), p, {"x": x})
        want = sympy.limit(sympy.sympify(src, locals={"x": x_sp}), x_sp, sympy.oo)
        try:
            got = _to_sympy(ak.limit(e, x, p.pos_infinity()).value, sympy)
        except ak.AlkahestError:
            refused.append(src)
            continue
        if got == want:
            ok = True
        elif any(v.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan) for v in (got, want)):
            ok = False
        else:
            ok = abs(complex(sympy.N(got - want, 30))) < 1e-9
        if not ok:
            wrong.append(f"{src}: alkahest {got}, sympy {want}")
    assert not wrong, "\n".join(wrong)
    assert len(refused) <= MAX_REFUSED, refused


# Measured on this change; lower it when a refusal is fixed.
MAX_REFUSED = 3
