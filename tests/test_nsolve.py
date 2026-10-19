"""Certified numeric root-finding (F5): ``nsolve``, ``solve(numeric=True,
interval=...)``, the lower Lambert branch ``W₋₁`` and the numeric
free-parameter fallback.

Every root is compared against an independent oracle — ``scipy.optimize.brentq``
on the same function written in plain Python, a dense sign-change scan, or
``mpmath.lambertw`` — and every *verified* enclosure must contain the oracle's
root.
"""

import math
import random

import alkahest as ak
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "nsolve"),
    reason="native module built without groebner feature",
)


# --------------------------------------------------------------------------
# Implied volatility (the report's repro)
# --------------------------------------------------------------------------


def _bs_call_py(s, k, r, t, sigma):
    d1 = (math.log(s / k) + (r + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
    return s * phi(d1) - k * math.exp(-r * t) * phi(d2)


def _bs_call_expr(sig, s, k, r, t):
    rt = math.sqrt(t)
    d1 = (math.log(s / k) + (r + sig**2 / 2) * t) / (sig * rt)
    d2 = d1 - sig * rt
    inv_sqrt2 = 1 / math.sqrt(2)
    phi1 = (1 + ak.erf(d1 * inv_sqrt2)) / 2
    phi2 = (1 + ak.erf(d2 * inv_sqrt2)) / 2
    return s * phi1 - k * math.exp(-r * t) * phi2


def test_implied_volatility_matches_brentq_at_random_parameters():
    brentq = pytest.importorskip("scipy.optimize").brentq
    rng = random.Random(20261012)
    p = ak.ExprPool()
    sig = p.symbol("sigma")
    for _ in range(12):
        s = rng.uniform(50, 150)
        k = s * rng.uniform(0.7, 1.3)
        r = rng.uniform(0.0, 0.08)
        t = rng.uniform(0.1, 3.0)
        true_sigma = rng.uniform(0.05, 0.9)
        price = _bs_call_py(s, k, r, t, true_sigma)
        eq = _bs_call_expr(sig, s, k, r, t) - price
        root = ak.nsolve(eq, sig, bracket=(1e-3, 5.0))
        want = brentq(
            lambda v, s=s, k=k, r=r, t=t, price=price: _bs_call_py(s, k, r, t, v) - price,
            1e-3,
            5.0,
            xtol=1e-15,
            rtol=1e-15,
        )
        assert root.status == "verified", root
        assert root.verified
        # The enclosure is proved; brentq's root is within its own tolerance.
        slack = 1e-12 * max(1.0, abs(want))
        assert root.lo - slack <= want <= root.hi + slack, (root, want)
        assert root.hi - root.lo < 1e-9
        assert abs(root.value - true_sigma) < 1e-6
        assert float(root) == root.value


def test_implied_volatility_through_solve_with_an_interval():
    p = ak.ExprPool()
    sig = p.symbol("sigma")
    s, k, r, t = 100.0, 105.0, 0.03, 0.75
    price = _bs_call_py(s, k, r, t, 0.27)
    eq = _bs_call_expr(sig, s, k, r, t) - price
    sols = ak.solve([eq], [sig], numeric=True, interval=(0.01, 3.0))
    assert len(sols) == 1
    assert abs(sols[0][sig] - 0.27) < 1e-9
    notes = ak.solve_side_conditions()
    assert any("complete" in n for n in notes)
    assert any(n.startswith("sigma = ") for n in notes)


def test_numeric_solve_without_interval_points_at_the_numeric_route():
    p = ak.ExprPool()
    sig = p.symbol("sigma")
    eq = _bs_call_expr(sig, 100.0, 105.0, 0.03, 0.75) - 7.0
    with pytest.raises(ak.SolverError) as ei:
        ak.solve([eq], [sig], numeric=True)
    assert ei.value.code == "E-SOLVE-001"
    assert "interval=" in str(ei.value)


# --------------------------------------------------------------------------
# nsolve: bracket / x0 / all_in
# --------------------------------------------------------------------------


def test_bracket_and_newton_agree_on_dottie_number():
    p = ak.ExprPool()
    x = p.symbol("x")
    dottie = 0.7390851332151607
    for root in (
        ak.nsolve(ak.cos(x) - x, x, bracket=(0, 1)),
        ak.nsolve(ak.cos(x) - x, x, x0=1.0),
    ):
        assert root.status == "verified"
        assert root.lo <= dottie <= root.hi
        assert root.enclosure == (root.lo, root.hi)


def _sign_change_roots(f, a, b, n):
    xs = [a + (b - a) * i / n for i in range(n + 1)]
    out = []
    for u, v in zip(xs, xs[1:]):
        fu, fv = f(u), f(v)
        if fu == 0.0:
            out.append(u)
        elif fu * fv < 0:
            lo, hi = u, v
            for _ in range(100):
                m = 0.5 * (lo + hi)
                if (f(m) < 0) == (f(lo) < 0):
                    lo = m
                else:
                    hi = m
            out.append(0.5 * (lo + hi))
    return out


def test_all_in_sin_equals_x_over_10_matches_dense_sampling():
    p = ak.ExprPool()
    x = p.symbol("x")
    iso = ak.nsolve(ak.sin(x) - x / 10, x, all_in=(-12.3, 11.7))
    assert iso.complete
    assert iso.unresolved == []
    want = _sign_change_roots(lambda v: math.sin(v) - v / 10, -12.3, 11.7, 20000)
    assert len(iso) == len(want) == 7
    for r, w in zip(iso, want):
        assert r.status == "verified"
        assert r.lo - 1e-13 <= w <= r.hi + 1e-13, (r, w)
    # Enclosures are disjoint and increasing.
    assert all(a.hi < b.lo for a, b in zip(iso.roots, iso.roots[1:]))


def test_all_in_reports_a_double_root_instead_of_guessing():
    p = ak.ExprPool()
    x = p.symbol("x")
    iso = ak.nsolve((x - 1) ** 2 * (x + 2), x, all_in=(-3, 3))
    assert not iso.complete
    assert len(iso) == 1
    assert abs(iso[0].value + 2) < 1e-15
    ((lo, hi),) = iso.unresolved
    assert lo <= 1 <= hi
    # solve(interval=...) promises a whole solution set, so it refuses.
    with pytest.raises(ak.SolverError) as ei:
        ak.solve([(x - 1) ** 2 * (x + 2)], [x], numeric=True, interval=(-3, 3))
    assert ei.value.code == "E-SOLVE-012"


def test_nsolve_refusals():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(ak.SolverError) as ei:
        ak.nsolve(x**2 + 1, x, bracket=(-1, 1))
    assert ei.value.code == "E-SOLVE-009"
    with pytest.raises(ak.SolverError) as ei:
        ak.nsolve(x - y, x, x0=0.0)
    assert ei.value.code == "E-SOLVE-009"
    assert "y" in str(ei.value)
    with pytest.raises(ak.SolverError) as ei:
        ak.nsolve(ak.abs(x) - 1, x, all_in=(-2, 2))
    assert ei.value.code == "E-SOLVE-009"
    with pytest.raises(ValueError):
        ak.nsolve(x - 1, x)
    with pytest.raises(ValueError):
        ak.nsolve(x - 1, x, x0=0.0, bracket=(0, 2))


def test_nsolve_honours_the_budget():
    p = ak.ExprPool()
    x = p.symbol("x")
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=5)):
        ak.nsolve(ak.sin(50 * x) - p.rational(1, 3), x, all_in=(-10, 10))


# --------------------------------------------------------------------------
# Lambert W₋₁
# --------------------------------------------------------------------------


def test_lambert_wm1_against_mpmath():
    mpmath = pytest.importorskip("mpmath")
    p = ak.ExprPool()
    for c in (-0.3678, -0.36, -0.3, -0.25, -0.1, -1e-3, -1e-9):
        want = float(mpmath.lambertw(c, -1).real)
        got = ak.eval_expr(ak.lambert_wm1(p.float(c)), {})
        assert abs(got - want) <= 1e-13 * abs(want), (c, got, want)
        assert got <= -1
    x = p.symbol("x")
    assert ak.lambert_w(x, k=-1) == ak.lambert_wm1(x)
    assert ak.lambert_w(x, k=0) == ak.lambert_w(x)
    with pytest.raises(ValueError):
        ak.lambert_w(x, k=1)


def test_lambert_wm1_derivative_and_special_values():
    mpmath = pytest.importorskip("mpmath")
    p = ak.ExprPool()
    x = p.symbol("x")
    d = ak.diff(ak.lambert_wm1(x), x).value
    for c in (-0.3, -0.2, -0.05):
        want = float(mpmath.diff(lambda t: mpmath.lambertw(t, -1).real, c))
        got = ak.eval_expr(d, {x: c})
        assert abs(got - want) < 1e-9 * abs(want)
    two = p.integer(-2)
    assert ak.simplify(ak.lambert_wm1(two * ak.exp(two))).value == two
    assert "lambert_wm1" in str(ak.parse("lambert_wm1(x)", p))


def test_both_lambert_branches_substitute_back():
    mpmath = pytest.importorskip("mpmath")
    p = ak.ExprPool()
    x = p.symbol("x")
    # x·eˣ = −1/4: W₀(−1/4) and W₋₁(−1/4).
    sols = ak.solve([x * ak.exp(x) + p.rational(1, 4)], [x], numeric=True)
    vals = sorted(s[x] for s in sols)
    want = sorted(float(mpmath.lambertw(-0.25, k).real) for k in (0, -1))
    assert vals == pytest.approx(want, abs=1e-14)
    for v in vals:
        assert abs(v * math.exp(v) + 0.25) < 1e-15
    # x**x = 4/5: exp(W₀(log 4/5)) and exp(W₋₁(log 4/5)).
    sols = ak.solve([x**x - p.rational(4, 5)], [x], numeric=True)
    assert len(sols) == 2
    for s in sols:
        assert abs(s[x] ** s[x] - 0.8) < 1e-14
    # y − 2·log y − 1 = 0 (A/B < 0): 1 and the W₋₁ root, vs brentq-free bisection.
    sols = ak.solve([x - 2 * ak.log(x) - 1], [x], numeric=True)
    vals = sorted(s[x] for s in sols)
    assert len(vals) == 2
    for v in vals:
        assert abs(v - 2 * math.log(v) - 1) < 1e-13
    iso = ak.nsolve(x - 2 * ak.log(x) - 1, x, all_in=(0.1, 10))
    assert iso.complete
    assert len(iso) == 2
    for r, v in zip(iso, vals):
        assert r.lo - 1e-14 <= v <= r.hi + 1e-14
    # Below −1/e there is no real root at all.
    assert ak.solve([x * ak.exp(x) + p.rational(1, 2)], [x]) == []


def test_negative_symbolic_lambert_argument_records_the_branch_condition():
    p = ak.ExprPool()
    x = p.symbol("x")
    c = p.symbol("c", "positive")
    sols = ak.solve([x * ak.exp(x) + c], [x])
    assert len(sols) == 2
    names = sorted(str(s[x]) for s in sols)
    assert any("lambert_wm1" in n for n in names)
    conds = ak.solve_side_conditions()
    assert len(conds) == 1, conds
    assert "> 0" in conds[0], conds
    for s in sols:
        v = ak.eval_expr(s[x], {c: 0.2})
        assert abs(v * math.exp(v) + 0.2) < 1e-14


# --------------------------------------------------------------------------
# W2: numeric=True with a free parameter falls back to the exact solution
# --------------------------------------------------------------------------


def test_free_parameter_numeric_falls_back_to_the_symbolic_solution():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    (sol,) = ak.solve([x - y], [x], numeric=True)
    assert sol[x] == y
    (note,) = ak.solve_side_conditions()
    assert "free parameter" in note
    assert "y" in note
    # Mixed: a parametric and a numeric component in one system.
    sols = ak.solve([x - y, x + y - 2], [x, y], numeric=True)
    assert sols == [{x: 1.0, y: 1.0}]
    # No symbolic solution to fall back to: still a coded refusal.
    with pytest.raises(ak.SolverError) as ei:
        ak.solve([x**3 - y], [x], numeric=True)
    assert ei.value.code == "E-SOLVE-006"
