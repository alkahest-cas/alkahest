"""Parametric ``solve``: every returned tuple is a solution, none is missing (N2),
and the unknown's declared ``Domain`` is honoured (N4).

Before the fix, back-substitution treated every leading coefficient that
mentions a parameter as non-zero, so ``solve([a*x - b*x*y, -c*y + d*x*y])``
divided by ``b*y - a`` at ``y = a/b`` and returned ``(0, (ac+|a||c|)/(2bc))`` —
not a solution — while losing ``(c/d, a/b)``; with positive parameters the same
division surfaced as ``x = -1*0*0^-1``.  Parametric tuples were never
substituted back, and the hypotheses of discarded branches (``(d*0) ≠ 0``) were
reported as side conditions.

The randomised test builds products of linear and quadratic factors with
symbolic coefficients and checks each answer against SymPy at random rational
parameter values: every returned tuple must satisfy the specialised system, and
the set must equal SymPy's.
"""

from __future__ import annotations

import random

import alkahest as ak
import pytest

sp = pytest.importorskip("sympy")

pytestmark = pytest.mark.skipif(not hasattr(ak, "solve_side_conditions"), reason="groebner")


def _points(sols, unknowns, env):
    out = []
    for sol in sols:
        point = []
        for v in unknowns:
            r = ak.evaluate(sol[v], env, mode="complex")
            assert r.status == "ok", f"{sol[v]} does not evaluate at {env}: {r.status}"
            point.append(complex(r.value))
        out.append(tuple(point))
    return out


def _same_set(got, want, tol=1e-9):
    def close(p, q):
        return all(abs(a - b) <= tol * (1 + abs(b)) for a, b in zip(p, q))

    return (
        len(got) == len(want)
        and all(any(close(p, q) for q in want) for p in got)
        and all(any(close(p, q) for p in got) for q in want)
    )


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# The report's repros
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", [ak.Domain.Real, ak.Domain.Positive])
def test_lotka_volterra_equilibria(pool, domain):
    x, y = pool.symbol("x"), pool.symbol("y")
    a, b, c, d = (pool.symbol(n, domain) for n in "abcd")
    sols = ak.solve([a * x - b * x * y, -c * y + d * x * y], [x, y])
    env = {a: 2.0, b: 1.0, c: 3.0, d: 1.5}
    assert _same_set(_points(sols, [x, y], env), [(0, 0), (2, 2)])
    for sol in sols:
        assert "0^-1" not in str(sol[x]) + str(sol[y])
    conds = ak.solve_side_conditions()
    # Only the divisors of the kept solution, never a condition with a literal 0.
    assert not any("0)" in c or "* 0" in c for c in conds), conds
    if domain == ak.Domain.Positive:
        assert conds == []
    else:
        assert sorted(conds) == ["b ≠ 0", "d ≠ 0"]


def test_variable_order_does_not_change_the_answer(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    al, g = pool.symbol("alpha"), pool.symbol("g")
    env = {al: 0.7, g: 1.3}
    for order in ([x, y], [y, x]):
        sols = ak.solve([x * (al - y), y * (x - g)], order)
        assert _same_set(_points(sols, [x, y], env), [(0, 0), (1.3, 0.7)])


def test_competition_model_has_four_equilibria(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    r1, r2, k1, k2, a12, a21 = (
        pool.symbol(n, ak.Domain.Positive) for n in ("r1", "r2", "K1", "K2", "a12", "a21")
    )
    f1 = r1 * x * (1 - (x + a12 * y) / k1)
    f2 = r2 * y * (1 - (y + a21 * x) / k2)
    sols = ak.solve([f1, f2], [x, y])
    env = {r1: 1.0, r2: 0.5, k1: 10.0, k2: 8.0, a12: 0.5, a21: 0.25}
    # Coexistence: x + 0.5 y = 10, y + 0.25 x = 8 → x = 48/7, y = 44/7.
    want = [(0, 0), (10, 0), (0, 8), (48 / 7, 44 / 7)]
    assert _same_set(_points(sols, [x, y], env), want)


def test_two_hyperbolas(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    a, b, c, d = (pool.symbol(n) for n in "abcd")
    sols = ak.solve([(x - a) * (y - b), (x - c) * (y - d)], [x, y])
    env = {a: 1.0, b: 2.0, c: 3.0, d: 5.0}
    assert _same_set(_points(sols, [x, y], env), [(1, 5), (3, 2)])


def test_a_parametric_line_is_returned_as_the_ideal(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    a, b = pool.symbol("a"), pool.symbol("b")
    out = ak.solve([x * (y - a), x * (y - b)], [x, y])
    # x = 0 with y free: positive-dimensional, never a single 0/0 point.
    assert not isinstance(out, list)


def test_seir_disease_free_equilibrium(pool):
    beta, gamma, sigma, mu, n = (
        pool.symbol(s, ak.Domain.Positive) for s in ("beta", "gamma", "sigma", "mu", "N")
    )
    s, e, i = pool.symbol("S"), pool.symbol("E"), pool.symbol("I")
    fs = mu * n - beta * s * i / n - mu * s
    fe = beta * s * i / n - (sigma + mu) * e
    fi = sigma * e - (gamma + mu) * i
    sols = ak.solve([fs, fe, fi], [s, e, i])
    env = {beta: 0.5, sigma: 0.2, gamma: 0.1, mu: 0.01, n: 1000.0}
    pts = _points(sols, [s, e, i], env)
    assert len(pts) == 2
    # The disease-free equilibrium is S = N, and every point is an equilibrium.
    assert any(abs(p[0] - 1000) < 1e-9 and abs(p[1]) < 1e-12 and abs(p[2]) < 1e-12 for p in pts)
    for p in pts:
        point = dict(zip([s, e, i], p))
        point.update(env)
        for f in (fs, fe, fi):
            assert abs(complex(ak.evaluate(f, point, mode="complex").value)) < 1e-9


def test_generically_inconsistent_system_has_no_solution(pool):
    x = pool.symbol("x")
    a, b = pool.symbol("a"), pool.symbol("b")
    assert ak.solve([x - a, x - b], [x]) == []


# ---------------------------------------------------------------------------
# Randomised oracle: products of linear / quadratic factors vs SymPy
# ---------------------------------------------------------------------------


def _random_system(rng, pool, x, y, params, sx, sy, sparams):
    def coef():
        k = rng.choice([1, 1, 1, 2, -1, -2, 3])
        if rng.random() < 0.6:
            j = rng.randrange(len(params))
            return k * params[j], k * sparams[j]
        return pool.integer(k), sp.Integer(k)

    def linear():
        terms, sterms = [], []
        for v, sv in ((x, sx), (y, sy)):
            if rng.random() < 0.75:
                c, sc = coef()
                terms.append(c * v)
                sterms.append(sc * sv)
        c, sc = coef()
        terms.append(c)
        sterms.append(sc)
        if len(terms) == 1:
            terms.append(x)
            sterms.append(sx)
        e = terms[0]
        for t in terms[1:]:
            e = e + t
        return e, sum(sterms)

    def factor():
        r = rng.random()
        if r < 0.2:
            return rng.choice([(x, sx), (y, sy)])
        if r < 0.36:
            lin, slin = linear()
            c, sc = coef()
            v, sv = rng.choice([(x, sx), (y, sy)])
            return v * v + c * lin, sv * sv + sc * slin
        return linear()

    eqs = []
    for _ in range(2):
        e, s = factor()
        if rng.random() < 0.6:
            e2, s2 = factor()
            e, s = e * e2, s * s2
        eqs.append((e, s))
    return eqs


@pytest.mark.parametrize("seed", range(4))
def test_random_parametric_equilibria_match_sympy(seed):
    rng = random.Random(1000 + seed)
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    params = [pool.symbol(n, ak.Domain.Positive) for n in "abc"]
    sx, sy = sp.symbols("x y")
    sparams = sp.symbols("a b c", positive=True)
    checked = 0
    for _ in range(12):
        eqs = _random_system(rng, pool, x, y, params, sx, sy, sparams)
        try:
            sols = ak.solve([e for e, _ in eqs], [x, y])
        except ak.SolverError:
            continue  # a coded refusal is an honest answer
        if not isinstance(sols, list):
            continue  # a positive-dimensional ideal, returned as such
        # Random primes over primes: off every proper subvariety a generic
        # statement can fail on, with overwhelming probability.
        vals = [
            sp.Rational(rng.choice([7, 11, 13, 17, 19, 23]), rng.choice([3, 5, 29]))
            for _ in range(3)
        ]
        env = dict(zip(params, (float(v) for v in vals)))
        system = [sp.expand(s.subs(dict(zip(sparams, vals)))) for _, s in eqs]
        ref = sp.solve(system, [sx, sy], dict=True)
        if any(len(r) < 2 for r in ref):
            continue  # positive-dimensional at this point
        want = [(complex(sp.N(r[sx], 30)), complex(sp.N(r[sy], 30))) for r in ref]
        got = _points(sols, [x, y], env)
        # Every returned tuple is a solution of the specialised system ...
        for gx, gy in got:
            for s in system:
                res = complex(sp.N(s.subs({sx: gx, sy: gy})))
                assert abs(res) < 1e-7 * (1 + abs(gx) + abs(gy)) ** 4, (system, (gx, gy))
        # ... and the set is SymPy's.
        assert _same_set(got, want, tol=1e-7), (system, got, want)
        checked += 1
    assert checked >= 6


def test_random_parametric_solutions_are_identities(pool):
    """Each returned tuple substitutes back to the zero function (is_zero)."""
    x, y = pool.symbol("x"), pool.symbol("y")
    a, b, c, d = (pool.symbol(n, ak.Domain.Positive) for n in "abcd")
    systems = [
        [a * x - b * x * y, -c * y + d * x * y],
        [x * (a - x - b * y), y * (c - y - d * x)],
        [(x - a) * (y - b), (x - c) * (y - d)],
    ]
    for eqs in systems:
        sols = ak.solve(eqs, [x, y])
        assert sols
        for sol in sols:
            for eq in eqs:
                residual = ak.subs(eq, {x: sol[x], y: sol[y]})
                assert ak.is_zero(residual) is True, (eq, sol)


# ---------------------------------------------------------------------------
# N4: the declared domain of the unknown
# ---------------------------------------------------------------------------


def test_positive_unknown(pool):
    xp = pool.symbol("xp", ak.Domain.Positive)
    assert ak.solve([xp + 3], [xp]) == []
    sols = ak.solve([xp**2 - 4], [xp])
    assert [ak.eval_expr(s[xp], {}) for s in sols] == [2.0]


def test_integer_unknown(pool):
    n = pool.symbol("n", ak.Domain.Integer)
    assert ak.solve([2 * n - 3], [n]) == []
    sols = ak.solve([n**2 - 4], [n])
    assert sorted(ak.eval_expr(s[n], {}) for s in sols) == [-2.0, 2.0]
    assert ak.solve_side_conditions() == []


def test_nonnegative_and_nonzero_unknowns(pool):
    xn = pool.symbol("xn", ak.Domain.NonNegative)
    assert [ak.eval_expr(s[xn], {}) for s in ak.solve([xn * (xn + 1)], [xn])] == [0.0]
    nz = pool.symbol("nz", ak.Domain.NonZero)
    assert [ak.eval_expr(s[nz], {}) for s in ak.solve([nz * (nz - 1)], [nz])] == [1.0]


def test_positive_unknown_drops_complex_roots(pool):
    xp = pool.symbol("xp", ak.Domain.Positive)
    assert ak.solve([xp**2 + 4], [xp]) == []


def test_undecided_membership_is_stated(pool):
    xp = pool.symbol("xp", ak.Domain.Positive)
    a = pool.symbol("a")
    sols = ak.solve([xp - a], [xp])
    assert len(sols) == 1
    assert ak.solve_side_conditions() == ["a > 0"]
    # A parameter declared positive settles it.
    ap = pool.symbol("ap", ak.Domain.Positive)
    assert len(ak.solve([xp - ap], [xp])) == 1
    assert ak.solve_side_conditions() == []


def test_domain_agrees_with_assumptions_keyword(pool):
    """Declaring the domain and passing ``assumptions=`` give the same answer."""
    xp = pool.symbol("xp", ak.Domain.Positive)
    x = pool.symbol("x")
    asm = ak.Assumptions(pool)
    asm.refine(pool.gt(x, pool.integer(0)))
    via_domain = [ak.eval_expr(s[xp], {}) for s in ak.solve([xp**2 - xp - 2], [xp])]
    via_assumptions = [
        ak.eval_expr(s[x], {}) for s in ak.solve([x**2 - x - 2], [x], assumptions=asm)
    ]
    assert via_domain == via_assumptions == [2.0]
