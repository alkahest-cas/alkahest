"""Classic inputs that 3.13.0 refused, each checked against an independent oracle.

1. ``∫₀¹ atan(x)/x = Catalan`` and its relatives — mpmath ``quad`` to 30 digits.
2. ``Σ_{k=0}^{n} C(n,k) = 2ⁿ`` and the other Zeilberger sums — exact
   ``math.comb`` sums for ``n = 0..30``.
3. ``series(sqrt(x), x, 0, 3)`` — a branch-point refusal that names
   ``puiseux_series`` and shows the expansion (checked against SymPy's series).
4. ``diff(diff(zeta(x)))`` — ``zeta_deriv(2, x)``, numerically against mpmath's
   ``zeta(s, 1, n)``.
5. ``number_theory.discrete_log`` for composite moduli — brute force and SymPy.
"""

from __future__ import annotations

import math
import random

import pytest

import alkahest as ak
from alkahest import number_theory as nt

mpmath = pytest.importorskip("mpmath")
sympy = pytest.importorskip("sympy")
from alkahest.crosscheck import to_sympy  # noqa: E402


@pytest.fixture
def pool():
    return ak.ExprPool()


def _num(expr, dps=40):
    """High-precision value of an Alkahest constant expression, via SymPy."""
    return mpmath.mpf(str(sympy.N(_sym(expr), dps)))


_CONSTS = {"pi": sympy.pi, "Catalan": sympy.Catalan, "E": sympy.E, "EulerGamma": sympy.EulerGamma}


def _sym(expr):
    """``to_sympy`` with the named constants as SymPy's own."""
    s = to_sympy(expr)
    return s.subs({f: _CONSTS[f.name] for f in s.free_symbols if f.name in _CONSTS})


# ---------------------------------------------------------------------------
# 1. Catalan's constant and relatives
# ---------------------------------------------------------------------------

_M = mpmath


def _integrals(p):
    x = p.symbol("x")
    pi = ak.parse("pi", p) if not hasattr(p, "pi") else p.pi()
    zero, one, inf = p.integer(0), p.integer(1), p.pos_infinity()
    return x, [
        # (integrand, a, b, mpmath integrand, mp a, mp b, mentions Catalan)
        (ak.atan(x) / x, zero, one, lambda t: _M.atan(t) / t, 0, 1, True),
        (ak.log(x) / (1 + x**2), zero, one, lambda t: _M.log(t) / (1 + t**2), 0, 1, True),
        (ak.log(x) / (1 + x**2), one, inf, lambda t: _M.log(t) / (1 + t**2), 1, _M.inf, True),
        (
            ak.atan(x) / (x * (1 + x**2)),
            zero,
            inf,
            lambda t: _M.atan(t) / (t * (1 + t**2)),
            0,
            _M.inf,
            False,
        ),
        (
            ak.log(1 + x**2) / (1 + x**2),
            zero,
            one,
            lambda t: _M.log(1 + t**2) / (1 + t**2),
            0,
            1,
            True,
        ),
        (
            ak.log(1 + x**2) / (1 + x**2),
            zero,
            inf,
            lambda t: _M.log(1 + t**2) / (1 + t**2),
            0,
            _M.inf,
            False,
        ),
        (ak.log(1 + x) / (1 + x**2), zero, one, lambda t: _M.log(1 + t) / (1 + t**2), 0, 1, False),
        (ak.atan(x) / (1 + x), zero, one, lambda t: _M.atan(t) / (1 + t), 0, 1, False),
        (ak.log(ak.tan(x)), zero, pi / 4, lambda t: _M.log(_M.tan(t)), 0, _M.pi / 4, True),
        (ak.log(1 + ak.tan(x)), zero, pi / 4, lambda t: _M.log(1 + _M.tan(t)), 0, _M.pi / 4, False),
        (x / ak.sin(x), zero, pi / 2, lambda t: t / _M.sin(t), 0, _M.pi / 2, True),
        (ak.log(ak.sin(x)), zero, pi / 2, lambda t: _M.log(_M.sin(t)), 0, _M.pi / 2, False),
        (ak.log(ak.cos(x)), zero, pi / 2, lambda t: _M.log(_M.cos(t)), 0, _M.pi / 2, False),
        (ak.log(ak.sin(x)), zero, pi / 4, lambda t: _M.log(_M.sin(t)), 0, _M.pi / 4, True),
        (ak.log(ak.cos(x)), zero, pi / 4, lambda t: _M.log(_M.cos(t)), 0, _M.pi / 4, True),
        (x / ak.cosh(x), zero, inf, lambda t: t / _M.cosh(t), 0, _M.inf, True),
        (
            ak.atan(x) / (x * ak.sqrt(1 - x**2)),
            zero,
            one,
            # x = sin t removes the endpoint singularity the quadrature cannot resolve.
            lambda t: _M.atan(_M.sin(t)) / _M.sin(t),
            0,
            _M.pi / 2,
            False,
        ),
    ]


def test_catalan_integrals_match_mpmath(pool):
    with mpmath.workdps(40):
        # Built at 40 digits: the mpmath bounds (pi/4, …) are evaluated here.
        x, rows = _integrals(pool)
        for e, a, b, f, ma, mb, catalan in rows:
            r = ak.integrate(e, x, a, b)
            got = _num(r.value)
            want = mpmath.quad(f, [ma, mb], method="tanh-sinh")
            assert abs(got - want) < mpmath.mpf(10) ** -28, (str(e), str(r.value), got, want)
            assert ("Catalan" in str(r.value)) == catalan, str(r.value)


def test_log_over_one_plus_x2_half_line_is_zero(pool):
    # The [0, 1] and [1, oo) pieces cancel: x -> 1/x maps one to minus the other.
    x = pool.symbol("x")
    e = ak.log(x) / (1 + x**2)
    assert str(ak.integrate(e, x, pool.integer(0), pool.pos_infinity()).value) == "0"
    f = lambda t: _M.log(t) / (1 + t**2)
    with mpmath.workdps(30):
        assert abs(mpmath.quad(f, [0, 1, _M.inf])) < mpmath.mpf(10) ** -25


def test_the_reported_repro_is_catalan(pool):
    x = pool.symbol("x")
    r = ak.integrate(ak.atan(x) / x, x, pool.integer(0), pool.integer(1))
    assert str(r.value) == "Catalan"
    # ∫₀¹ log(x)/(1+x²) = −G.
    r = ak.integrate(ak.log(x) / (1 + x**2), x, pool.integer(0), pool.integer(1))
    assert _sym(r.value) == -sympy.Catalan


def test_catalan_constant_multiples_reversal_and_sums(pool):
    x = pool.symbol("x")
    zero, one = pool.integer(0), pool.integer(1)
    a = pool.symbol("a")
    r = ak.integrate(3 * ak.atan(x) / x, x, zero, one)
    assert _sym(r.value) == 3 * sympy.Catalan
    r = ak.integrate(a * ak.atan(x) / x, x, zero, one)
    assert sympy.simplify(_sym(r.value) - sympy.Symbol("a") * sympy.Catalan) == 0
    r = ak.integrate(ak.atan(x) / x, x, one, zero)
    assert _sym(r.value) == -sympy.Catalan
    # Linearity: a table term plus an elementary one.
    r = ak.integrate(ak.atan(x) / x + x**2, x, zero, one)
    assert sympy.simplify(_sym(r.value) - (sympy.Catalan + sympy.Rational(1, 3))) == 0


def test_non_table_interval_is_still_refused(pool):
    # ∫₀² atan(x)/x = Ti₂(2) has no closed form in these constants.
    x = pool.symbol("x")
    with pytest.raises(ak.IntegrationError):
        ak.integrate(ak.atan(x) / x, x, pool.integer(0), pool.integer(2))


# ---------------------------------------------------------------------------
# 2. Zeilberger sums with a symbolic upper limit
# ---------------------------------------------------------------------------


def _check_sum(pool, term_fn, oracle, *, n_from=0, lo=None, hi=None):
    n = pool.symbol("n")
    k = pool.symbol("k")
    lo = pool.integer(0) if lo is None else lo(n)
    hi = n if hi is None else hi(n)
    r = ak.sum_definite(term_fn(n, k), k, lo, hi)
    s = _sym(r.value)
    ns = sympy.Symbol("n")
    for m in range(n_from, 31):
        got = sympy.nsimplify(sympy.simplify(s.subs(ns, m)))
        assert got == oracle(m), (m, str(r.value), got, oracle(m))
    return r


def test_sum_binomial_is_two_to_the_n(pool):
    r = _check_sum(pool, lambda n, k: ak.binomial(n, k), lambda m: 2**m)
    assert sympy.simplify(_sym(r.value) - 2 ** sympy.Symbol("n")) == 0


def test_sum_k_binomial(pool):
    _check_sum(
        pool,
        lambda n, k: k * ak.binomial(n, k),
        lambda m: sum(j * math.comb(m, j) for j in range(m + 1)),
    )


def test_sum_binomial_squared_is_central_binomial(pool):
    _check_sum(pool, lambda n, k: ak.binomial(n, k) ** 2, lambda m: math.comb(2 * m, m))


def test_vandermonde(pool):
    # Σ_k C(n,k)·C(2n, n−k) = C(3n, n).
    _check_sum(
        pool,
        lambda n, k: ak.binomial(n, k) * ak.binomial(2 * n, n - k),
        lambda m: math.comb(3 * m, m),
    )


def test_inhomogeneous_recurrence(pool):
    # Σ C(n,k)/(k+1) = (2^{n+1} − 1)/(n + 1): Zeilberger's boundary is nonzero.
    from fractions import Fraction

    def oracle(m):
        return sympy.Rational(sum(Fraction(math.comb(m, j), j + 1) for j in range(m + 1)))

    _check_sum(pool, lambda n, k: ak.binomial(n, k) / (k + 1), oracle)


def test_alternating_binomial_states_its_domain(pool):
    # Σ (−1)^k C(n,k) = [n = 0]: the closed form 0 holds from n = 1, and the
    # result says so instead of claiming it at n = 0.
    n = pool.symbol("n")
    k = pool.symbol("k")
    r = ak.sum_definite((-1) ** k * ak.binomial(n, k), k, pool.integer(0), n)
    assert str(r.value) == "0"
    conds = [c for s in r.steps for c in s.get("side_conditions", [])]
    assert any("n" in c for c in conds), conds


def test_binomial_past_its_natural_boundary_is_zero(pool):
    # The k = n term of C(n, k+1) and the k = n+1 term of C(n, k) are 1/Γ(0) = 0.
    _check_sum(
        pool,
        lambda n, k: ak.binomial(n, k) * ak.binomial(n, k + 1),
        lambda m: sum(math.comb(m, j) * math.comb(m, j + 1) for j in range(m + 1)),
        n_from=1,
    )
    _check_sum(pool, lambda n, k: ak.binomial(n, k), lambda m: 2**m, hi=lambda n: n + 1, n_from=1)


def test_order_two_sum_is_refused_with_a_reason(pool):
    # Franel numbers Σ C(n,k)³ satisfy an order-2 recurrence only.
    n = pool.symbol("n")
    k = pool.symbol("k")
    with pytest.raises(ak.SumError) as ei:
        ak.sum_definite(ak.binomial(n, k) ** 3, k, pool.integer(0), n)
    assert "Zeilberger" in str(ei.value)
    assert "gamma argument" not in str(ei.value)


# ---------------------------------------------------------------------------
# 3. Branch points in series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, sym_build",
    [
        (lambda p, x: ak.sqrt(x), lambda t: sympy.sqrt(t)),
        (lambda p, x: ak.sqrt(x) * ak.exp(x), lambda t: sympy.sqrt(t) * sympy.exp(t)),
        (
            lambda p, x: x ** p.rational(1, 3) * ak.sin(x),
            lambda t: t ** sympy.Rational(1, 3) * sympy.sin(t),
        ),
    ],
)
def test_series_at_a_branch_point_says_so(build, sym_build):
    p = ak.ExprPool()
    x = p.symbol("x")
    e = build(p, x)
    with pytest.raises(ak.SeriesError) as ei:
        ak.series(e, x, p.integer(0), 3)
    msg = str(ei.value)
    assert ei.value.code == "E-SERIES-004"
    assert "branch point" in msg and "puiseux_series" in msg, msg
    assert "indeterminate" not in msg, msg
    # The expansion in the message is SymPy's, term for term.
    shown = msg.rsplit("): ", 1)[1].split(" + O(")[0]
    t = sympy.Symbol("x", positive=True)
    want = sympy.series(sym_build(t), t, 0, 3).removeO()
    got = sympy.sympify(shown.replace("^", "**"), locals={"x": t})
    assert sympy.simplify(got - want) == 0, (shown, want)


def test_series_log_is_still_indeterminate(pool):
    x = pool.symbol("x")
    with pytest.raises(ak.SeriesError) as ei:
        ak.series(ak.log(x), x, pool.integer(0), 3)
    msg = str(ei.value)
    assert "not a power series" not in msg and "indeterminate form" in msg, msg


# ---------------------------------------------------------------------------
# 4. Higher derivatives of zeta
# ---------------------------------------------------------------------------


def test_zeta_higher_derivatives(pool):
    s = pool.symbol("s")
    d1 = ak.diff(ak.zeta(s), s).value
    d2 = ak.diff(d1, s).value
    d3 = ak.diff(d2, s).value
    assert str(d1) == "zeta_prime(s)"
    assert str(d2) == "zeta_deriv(2, s)"
    assert str(d3) == "zeta_deriv(3, s)"
    assert ak.zeta_deriv(1, s) == d1
    assert ak.zeta_deriv(0, s) == ak.zeta(s)


@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("sv", [-2.5, 0.5, 2.0, 3.75])
def test_zeta_deriv_numeric_matches_mpmath(pool, order, sv):
    s = pool.symbol("s")
    e = ak.zeta_deriv(order, s)
    got = ak.eval_expr(e, {s: sv})
    want = float(mpmath.zeta(sv, 1, order))
    assert got == pytest.approx(want, rel=1e-13, abs=1e-15)


def test_zeta_deriv_chain_rule_numeric(pool):
    s = pool.symbol("s")
    e = ak.diff(ak.diff(ak.zeta(2 * s), s).value, s).value
    got = ak.eval_expr(e, {s: 1.3})
    want = 4 * float(mpmath.zeta(2.6, 1, 2))
    assert got == pytest.approx(want, rel=1e-13)


def test_zeta_deriv_refusals(pool):
    s = pool.symbol("s")
    with pytest.raises(ak.SpecialFunctionError):
        ak.zeta_deriv(-1, s)
    with pytest.raises(ak.SpecialFunctionError):
        ak.zeta_deriv(2, pool.integer(1))


# ---------------------------------------------------------------------------
# 5. Discrete logarithms modulo a composite
# ---------------------------------------------------------------------------


def _brute(r, b, m):
    cur = 1 % m
    for e in range(m + 1):
        if cur == r % m:
            return e
        cur = cur * b % m
    return None


def test_reported_repro_has_no_solution():
    # 13 generates the index-2 subgroup of (Z/100)^x; 3 is not in it.
    assert _brute(3, 13, 100) is None
    with pytest.raises(nt.NumberTheoryError) as ei:
        nt.discrete_log(3, 13, 100)
    assert ei.value.code == "E-NT-003"
    assert nt.discrete_log(17, 13, 100) == 7


def test_composite_discrete_log_brute_force():
    rng = random.Random(20261108)
    for m in range(2, 2001):
        bases = {2, 3, 6, 10, m - 1, m // 2, rng.randrange(m), rng.randrange(m)}
        for b in bases:
            rs = {0, 1, rng.randrange(m), rng.randrange(m), pow(b, rng.randrange(1, 50), m)}
            for r in rs:
                want = _brute(r, b, m)
                if want is None:
                    with pytest.raises(nt.NumberTheoryError) as ei:
                        nt.discrete_log(r, b, m)
                    assert ei.value.code == "E-NT-003", (r, b, m)
                else:
                    assert nt.discrete_log(r, b, m) == want, (r, b, m)


def test_composite_discrete_log_against_sympy():
    from sympy.ntheory import discrete_log as sp_dlog

    rng = random.Random(7)
    for _ in range(150):
        m = rng.randrange(10**6, 10**12)
        b = rng.randrange(2, m)
        if math.gcd(b, m) != 1:
            continue
        e = rng.randrange(0, m)
        r = pow(b, e, m)
        try:
            want = sp_dlog(m, r, b)
        except ValueError:
            continue
        got = nt.discrete_log(r, b, m)
        assert pow(b, got, m) == r
        assert got == want, (r, b, m)


def test_composite_discrete_log_honours_budget():
    p = 4611686018427394499  # safe prime: ord(4) is a 61-bit prime
    m = 15 * p
    r = pow(4, 987654321987654321, m)
    with pytest.raises(ak.BudgetExceededError) as ei, ak.context(budget=ak.Budget(wall_ms=300)):
        nt.discrete_log(r, 4, m)
    assert ei.value.code.startswith("E-BUDGET-")
