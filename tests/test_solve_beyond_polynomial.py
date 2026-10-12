"""``solve`` beyond polynomials (G2).

* Systems that are polynomial in the unknowns but carry non-polynomial
  coefficients — multibody equations of motion solved for q̈, where ``sin(q1)``
  and ``cos(q1 - q2)`` are coefficients, not unknowns.
* Log-reducible exponential equations (time of peak concentration, ``a**x = b``).
* Lambert-W shapes beyond ``x·eˣ = a``: ``x·e^{a·x} = b``, ``a·x + b·log x = c``,
  ``x**x = c``, integrated Michaelis–Menten.
* Trigonometric equations, returned as whole periodic families.

Every returned solution is substituted back into the equation it came from;
where an independent oracle exists (a linear solve of the numerically evaluated
mass matrix, a bracketing root finder, SymPy) it is compared against that too.
"""

import math
import random

import alkahest as ak
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(ak, "solve_numerical"),
    reason="native module built without groebner feature",
)


def _ev(expr, env):
    return ak.eval_expr(expr, env)


def _gauss(a, b):
    """Plain Gaussian elimination with partial pivoting (an oracle that shares
    no code with the library)."""
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for j in range(c, n + 1):
                    m[r][j] -= f * m[c][j]
    return [m[i][n] / m[i][i] for i in range(n)]


def _bisect(f, lo, hi, tol=1e-13):
    flo = f(lo)
    assert flo * f(hi) < 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Multibody: n-link pendulum EOM solved for the accelerations
# ---------------------------------------------------------------------------


def _pendulum_eom(n):
    """Euler–Lagrange equations of an n-link planar pendulum, with q, q̇, q̈ as
    plain symbols (the same construction as the report's probe)."""
    p = ak.ExprPool()
    m = [p.symbol(f"m{i + 1}") for i in range(n)]
    ln = [p.symbol(f"L{i + 1}") for i in range(n)]
    g = p.symbol("g")
    q = [p.symbol(f"q{i + 1}") for i in range(n)]
    qd = [p.symbol(f"qd{i + 1}") for i in range(n)]
    qdd = [p.symbol(f"qdd{i + 1}") for i in range(n)]
    zero = p.integer(0)

    def d(e, v):
        return ak.diff(e, v).value

    def ddt(e):
        out = zero
        for j in range(n):
            out = out + d(e, q[j]) * qd[j] + d(e, qd[j]) * qdd[j]
        return out

    x, y = zero, zero
    kin, pot = zero, zero
    for i in range(n):
        x = x + ln[i] * ak.sin(q[i])
        y = y - ln[i] * ak.cos(q[i])
        kin = kin + m[i] * (ddt(x) ** 2 + ddt(y) ** 2) / 2
        pot = pot + m[i] * g * y
    lag = kin - pot
    eom = [ak.simplify(ddt(d(lag, qd[i])) - d(lag, q[i])).value for i in range(n)]
    params = [*m, *ln, g]
    return p, eom, params, q, qd, qdd


def _check_pendulum(n, points):
    _, eom, params, q, qd, qdd = _pendulum_eom(n)
    sols = ak.solve(eom, qdd)
    assert len(sols) == 1
    sol = sols[0]
    rng = random.Random(1234 + n)
    for _ in range(points):
        env = {s: rng.uniform(0.5, 2.0) for s in params}
        env.update({s: rng.uniform(-math.pi, math.pi) for s in q})
        env.update({s: rng.uniform(-2.0, 2.0) for s in qd})
        got = [_ev(sol[a], env) for a in qdd]
        # Oracle: the EOM is affine in q̈, e(q̈) = M·q̈ + c. Read M and c off the
        # equations numerically and solve the linear system independently.
        base = dict(env)
        base.update(dict.fromkeys(qdd, 0.0))
        c = [_ev(e, base) for e in eom]
        mass = [[0.0] * n for _ in range(n)]
        for j in range(n):
            unit = dict(base)
            unit[qdd[j]] = 1.0
            for i, e in enumerate(eom):
                mass[i][j] = _ev(e, unit) - c[i]
        want = _gauss(mass, [-ci for ci in c])
        for gv, wv in zip(got, want):
            assert abs(gv - wv) <= 1e-9 * (1 + abs(wv)), (got, want)
        # Substitute back into the equations as written.
        full = dict(env)
        full.update(dict(zip(qdd, got)))
        for e in eom:
            assert abs(_ev(e, full)) < 1e-8


def test_double_pendulum_qdd():
    # Report §4.2 repro: E-SOLVE-001 "sin is not polynomial" on main.
    _check_pendulum(2, points=6)


def test_triple_pendulum_qdd():
    _check_pendulum(3, points=4)


def test_double_pendulum_matches_numpy():
    np = pytest.importorskip("numpy")
    _, eom, params, q, qd, qdd = _pendulum_eom(2)
    sol = ak.solve(eom, qdd)[0]
    vals = {"m1": 1.3, "m2": 0.7, "L1": 1.1, "L2": 0.9, "g": 9.81}
    env = {s: vals[str(s)] for s in params}
    env.update({q[0]: 0.4, q[1]: -1.2, qd[0]: 0.5, qd[1]: -0.3})
    m1, m2, l1, l2, g = (vals[k] for k in ("m1", "m2", "L1", "L2", "g"))
    t1, t2, w1, w2 = 0.4, -1.2, 0.5, -0.3
    mass = np.array(
        [
            [(m1 + m2) * l1**2, m2 * l1 * l2 * math.cos(t1 - t2)],
            [m2 * l1 * l2 * math.cos(t1 - t2), m2 * l2**2],
        ]
    )
    f = np.array(
        [
            -m2 * l1 * l2 * w2**2 * math.sin(t1 - t2) - (m1 + m2) * g * l1 * math.sin(t1),
            m2 * l1 * l2 * w1**2 * math.sin(t1 - t2) - m2 * g * l2 * math.sin(t2),
        ]
    )
    want = np.linalg.solve(mass, f)
    got = [_ev(sol[a], env) for a in qdd]
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


def test_opaque_coefficient_linear_states_its_hypothesis():
    # sin(a)·x = b  →  x = b/sin(a), for sin(a) ≠ 0.
    p = ak.ExprPool()
    x, a, b = p.symbol("x"), p.symbol("a"), p.symbol("b")
    sols = ak.solve([ak.sin(a) * x - b], [x])
    assert len(sols) == 1
    env = {a: 0.7, b: 1.9}
    assert abs(_ev(sols[0][x], env) - 1.9 / math.sin(0.7)) < 1e-12
    assert any("sin" in c for c in ak.solve_side_conditions())


def test_opaque_identity_is_not_divided_by():
    # (sin²a + cos²a − 1)·x = 1 has no solution: the coefficient is identically
    # zero. Frozen as independent atoms it is not, so the generic answer would
    # divide by zero. It must not be returned.
    p = ak.ExprPool()
    x, a = p.symbol("x"), p.symbol("a")
    coeff = ak.sin(a) ** 2 + ak.cos(a) ** 2 - 1
    try:
        sols = ak.solve([coeff * x - 1], [x])
    except ak.SolverError:
        return
    assert sols == []


def test_unknown_inside_a_function_still_refuses():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(ak.SolverError) as ei:
        ak.solve([ak.sin(x) + y, x - y], [x, y])
    assert ei.value.code == "E-SOLVE-001"


def test_parameter_free_opaque_roots_are_verified():
    # x² = sin(1)  →  ±√sin(1), checked in ball arithmetic.
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([x**2 - ak.sin(p.integer(1))], [x], numeric=True)
    vals = sorted(s[x] for s in sols)
    r = math.sqrt(math.sin(1.0))
    assert len(vals) == 2
    assert abs(vals[0] + r) < 1e-12
    assert abs(vals[1] - r) < 1e-12


# ---------------------------------------------------------------------------
# Log-reducible exponential equations
# ---------------------------------------------------------------------------


def _tmax_residual(kav, kev):
    return lambda tt: kav * math.exp(-kav * tt) - kev * math.exp(-kev * tt)


def test_tmax_oral_one_compartment():
    # Report §4.1: Tmax solves ka·e^{-ka·t} = ke·e^{-ke·t}.
    p = ak.ExprPool()
    t, ka, ke = p.symbol("t"), p.symbol("ka"), p.symbol("ke")
    eq = ka * ak.exp(-ka * t) - ke * ak.exp(-ke * t)
    sols = ak.solve([eq], [t])
    assert len(sols) == 1
    tmax = sols[0][t]
    assert "log" in str(tmax)
    conds = ak.solve_side_conditions()
    assert any("> 0" in c for c in conds), conds
    assert any("≠ 0" in c for c in conds), conds
    for kav, kev in [(1.5, 0.2), (0.3, 2.0), (0.9, 0.11)]:
        env = {ka: kav, ke: kev}
        got = _ev(tmax, env)
        want = _bisect(_tmax_residual(kav, kev), 1e-9, 60)
        assert abs(got - want) < 1e-9
        assert abs(got - math.log(kav / kev) / (kav - kev)) < 1e-12


def test_tmax_with_positive_rates_drops_the_sign_hypothesis():
    p = ak.ExprPool()
    t = p.symbol("t")
    ka, ke = p.symbol("ka", "positive"), p.symbol("ke", "positive")
    sols = ak.solve([ka * ak.exp(-ka * t) - ke * ak.exp(-ke * t)], [t])
    assert len(sols) == 1
    conds = ak.solve_side_conditions()
    assert not any("> 0" in c for c in conds), conds
    assert len(conds) == 1
    assert "≠ 0" in conds[0]


def test_tmax_matches_scipy_brentq():
    optimize = pytest.importorskip("scipy.optimize")
    p = ak.ExprPool()
    t, ka, ke = p.symbol("t"), p.symbol("ka"), p.symbol("ke")
    tmax = ak.solve([ka * ak.exp(-ka * t) - ke * ak.exp(-ke * t)], [t])[0][t]
    for kav, kev in [(1.2, 0.15), (0.05, 0.7)]:
        root = optimize.brentq(_tmax_residual(kav, kev), 1e-9, 200)
        assert abs(_ev(tmax, {ka: kav, ke: kev}) - root) < 1e-9


def test_power_with_variable_exponent():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([p.integer(2) ** x - 5], [x], numeric=True)
    assert len(sols) == 1
    assert abs(sols[0][x] - math.log(5) / math.log(2)) < 1e-12


def test_exp_equals_symbolic_constant():
    p = ak.ExprPool()
    x, c = p.symbol("x"), p.symbol("c")
    sols = ak.solve([ak.exp(3 * x) - c], [x])
    assert len(sols) == 1
    assert abs(_ev(sols[0][x], {c: 7.0}) - math.log(7.0) / 3) < 1e-12
    assert any("c > 0" in s for s in ak.solve_side_conditions())


def test_two_exponentials_no_real_root():
    p = ak.ExprPool()
    x = p.symbol("x")
    assert ak.solve([ak.exp(2 * x) + 3 * ak.exp(x)], [x]) == []


def test_log_reducible_vs_sympy():
    sympy = pytest.importorskip("sympy")
    p = ak.ExprPool()
    x = p.symbol("x")
    sx = sympy.Symbol("x", real=True)
    cases = [
        (ak.exp(2 * x) - 3 * ak.exp(x), sympy.exp(2 * sx) - 3 * sympy.exp(sx)),
        (p.integer(3) ** x - 7 * p.integer(2) ** x, 3**sx - 7 * 2**sx),
        (5 * ak.exp(-x / 2) - 2, 5 * sympy.exp(-sx / 2) - 2),
    ]
    for ours, theirs in cases:
        got = sorted(s[x] for s in ak.solve([ours], [x], numeric=True))
        want = sorted(float(r) for r in sympy.solve(theirs, sx))
        assert len(got) == len(want)
        for g_, w_ in zip(got, want):
            assert abs(g_ - w_) < 1e-10


# ---------------------------------------------------------------------------
# Lambert W
# ---------------------------------------------------------------------------


def test_lambert_x_exp_ax_symbolic_a():
    p = ak.ExprPool()
    x, a, b = p.symbol("x"), p.symbol("a"), p.symbol("b")
    sols = ak.solve([x * ak.exp(a * x) - b], [x])
    assert len(sols) == 1
    v = sols[0][x]
    for av, bv in [(0.7, 2.0), (1.9, 0.4)]:
        xv = _ev(v, {a: av, b: bv})
        assert abs(xv * math.exp(av * xv) - bv) < 1e-10


def test_lambert_linear_plus_log():
    # a·x + b·log(x) = c  →  x = (b/a)·W((a/b)·e^{c/b})
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([2 * x + 3 * ak.log(x) - 5], [x], numeric=True)
    assert len(sols) == 1
    xv = sols[0][x]
    assert abs(2 * xv + 3 * math.log(xv) - 5) < 1e-10


def test_michaelis_menten_integrated():
    p = ak.ExprPool()
    s, s0, km, vmax, t = (p.symbol(n, "positive") for n in ("S", "S0", "Km", "Vmax", "t"))
    eq = km * ak.log(s / s0) + s - s0 + vmax * t
    sols = ak.solve([eq], [s])
    assert len(sols) == 1
    assert "W" in str(sols[0][s]) or "lambert" in str(sols[0][s]).lower()
    env = {s0: 10.0, km: 2.5, vmax: 1.3, t: 3.0}
    sv = _ev(sols[0][s], env)
    assert abs(2.5 * math.log(sv / 10.0) + sv - 10.0 + 1.3 * 3.0) < 1e-10


def test_self_power():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([x**x - 5], [x], numeric=True)
    assert len(sols) == 1
    xv = sols[0][x]
    assert abs(xv**xv - 5) < 1e-10


def test_self_power_two_real_roots_returns_both():
    # x**x = 0.8 has two real roots, on W0 and W-1; both are returned (F5).
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([x**x - p.rational(4, 5)], [x])
    assert len(sols) == 2
    assert sum("lambert_wm1" in str(s[x]) for s in sols) == 1
    for s in sols:
        v = _ev(s[x], {})
        assert abs(v**v - 0.8) < 1e-13


def test_lambert_second_branch_is_returned():
    # x·eˣ = −1/4 has two real roots; W0 alone would be half the solution set.
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([x * ak.exp(x) + p.rational(1, 4)], [x])
    assert sorted(str(s[x]) for s in sols) == ["lambert_w(-1/4)", "lambert_wm1(-1/4)"]
    for s in sols:
        v = _ev(s[x], {})
        assert abs(v * math.exp(v) + 0.25) < 1e-15


def test_lambert_vs_sympy():
    sympy = pytest.importorskip("sympy")
    p = ak.ExprPool()
    x = p.symbol("x")
    sx = sympy.Symbol("x")
    cases = [
        (x * ak.exp(3 * x) - 2, sx * sympy.exp(3 * sx) - 2),
        (x + ak.log(x) - 4, sx + sympy.log(sx) - 4),
        (x**x - 10, sx**sx - 10),
    ]
    for ours, theirs in cases:
        got = [s[x] for s in ak.solve([ours], [x], numeric=True)]
        want = [complex(sympy.N(r)) for r in sympy.solve(theirs, sx)]
        want = [w.real for w in want if abs(w.imag) < 1e-12]
        assert len(got) == len(want) == 1
        assert abs(got[0] - want[0]) < 1e-10


# ---------------------------------------------------------------------------
# Trigonometric equations: whole periodic families
# ---------------------------------------------------------------------------


def _family_index(sol):
    names = [c.split(" ")[0] for c in ak.solve_side_conditions() if "Integer" in c]
    assert len(names) == 1, ak.solve_side_conditions()
    return names[0]


@pytest.mark.parametrize(
    ("make", "f", "rhs", "count"),
    [
        (lambda x: ak.sin(x), math.sin, 0.5, 2),
        (lambda x: ak.cos(2 * x), lambda v: math.cos(2 * v), -0.3, 2),
        (lambda x: ak.tan(x), math.tan, 1.7, 1),
        (lambda x: ak.sin(x), math.sin, 1.0, 1),
    ],
)
def test_trig_family_is_complete(make, f, rhs, count):
    p = ak.ExprPool()
    x = p.symbol("x")
    rhs_e = p.rational(*float(rhs).as_integer_ratio())
    sols = ak.solve([make(x) - rhs_e], [x])
    assert len(sols) == count
    name = _family_index(sols)
    k = p.symbol(name, "integer")
    reps = set()
    for s in sols:
        for kv in range(-3, 4):
            v = _ev(s[x], {k: float(kv)})
            assert abs(f(v) - rhs) < 1e-12
            reps.add(round(v, 9))
    # Completeness: every root in [-10, 10], found by a sign-change scan of
    # f - rhs (or of its derivative's sign for the tangent double root at ±1),
    # is a member of the returned families.
    grid = [-10 + i * 1e-3 for i in range(20001)]
    for lo, hi in zip(grid, grid[1:]):
        a, b = f(lo) - rhs, f(hi) - rhs
        if a == 0 or (a < 0) != (b < 0):
            if abs(a) > 1 or abs(b) > 1:
                continue  # tangent pole, not a root
            r = _bisect(lambda v: f(v) - rhs, lo, hi) if a != 0 else lo
            assert any(abs(r - m) < 1e-6 for m in reps), r


def test_trig_numeric_reports_one_member_per_family():
    p = ak.ExprPool()
    x = p.symbol("x")
    sols = ak.solve([ak.sin(x) - p.rational(1, 2)], [x], numeric=True)
    vals = sorted(s[x] for s in sols)
    assert len(vals) == 2
    assert abs(vals[0] - math.pi / 6) < 1e-12
    assert abs(vals[1] - 5 * math.pi / 6) < 1e-12
    assert any("periodic" in c for c in ak.solve_side_conditions())


def test_trig_vs_sympy_principal_values():
    sympy = pytest.importorskip("sympy")
    p = ak.ExprPool()
    x = p.symbol("x")
    sx = sympy.Symbol("x")
    for ours, theirs, period in [
        (ak.sin(x) - p.rational(1, 3), sympy.sin(sx) - sympy.Rational(1, 3), 2 * math.pi),
        (
            ak.cos(3 * x) - p.rational(1, 5),
            sympy.cos(3 * sx) - sympy.Rational(1, 5),
            2 * math.pi / 3,
        ),
    ]:
        got = sorted(s[x] for s in ak.solve([ours], [x], numeric=True))
        want = sorted(float(r) for r in sympy.solve(theirs, sx))
        # SymPy's `solve` lists one period's representatives; every one of them
        # must be in a returned family (mod the period), and vice versa.
        for w in want:
            assert any(abs((w - g_) / period - round((w - g_) / period)) < 1e-9 for g_ in got)
        for g_ in got:
            assert any(abs((w - g_) / period - round((w - g_) / period)) < 1e-9 for w in want)


def test_trig_family_index_avoids_user_symbols():
    p = ak.ExprPool()
    x, k = p.symbol("x"), p.symbol("k")
    sols = ak.solve([ak.sin(x) - k], [x])
    assert len(sols) == 2
    assert _family_index(sols) != "k"
