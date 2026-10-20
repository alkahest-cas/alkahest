"""F28 — validated ODE integration (`alkahest.verified_ode`).

report10-9 §4.3 (SC6): "Validated ODE integration: none exists;
`ode_integrate_rk45` on Lorenz matches scipy to 1e-9" — a floating-point
integrator gives an *estimate*, not a statement.  `verified_ode` returns boxes
that provably contain the solution, or refuses with a coded error.

Every containment check below is against an independent oracle: a closed
form evaluated by mpmath at 40 digits, mpmath's own Taylor-series ODE solver
(`mpmath.odefun`) at 30+ digits, or scipy's DOP853 at rtol 1e-12/1e-13.
"""

import math

import alkahest as ak
import mpmath as mp
import pytest

mp.mp.dps = 40


def _inside(lo, hi, v, slack=0.0):
    return mp.mpf(lo) - slack <= v <= mp.mpf(hi) + slack


@pytest.fixture
def pool():
    return ak.ExprPool()


# ---------------------------------------------------------------------------
# Linear systems with closed forms
# ---------------------------------------------------------------------------


def test_exponential_decay_contains_exp_everywhere(pool):
    y = pool.symbol("y")
    sol = ak.verified_ode([-y], [y], None, [1.0], (0.0, 5.0))
    assert sol.dim == 1
    assert len(sol) == sol.steps + 1
    for t, box in zip(sol.times, sol.enclosures):
        lo, hi = box[0]
        assert _inside(lo, hi, mp.exp(-mp.mpf(t))), t
    # Dense output between grid points.
    for k in range(101):
        t = 0.05 * k
        lo, hi = sol.enclosure(t)[0]
        assert _inside(lo, hi, mp.exp(-mp.mpf(t))), t
        assert hi - lo < 1e-14
    assert sol.max_width < 1e-14


def test_rotation_contains_cos_sin_and_stays_tight(pool):
    # x' = -y, y' = x.  A naive interval method wraps the rotating box at
    # every step and the width grows like 2^(steps); the QR frame does not.
    x, y = pool.symbol("x"), pool.symbol("y")
    sol = ak.verified_ode([-y, x], [x, y], None, [1.0, 0.0], (0.0, 20.0))
    for t, (bx, by) in zip(sol.times, sol.enclosures):
        tt = mp.mpf(t)
        assert _inside(*bx, mp.cos(tt)), t
        assert _inside(*by, mp.sin(tt)), t
    for k in range(200):
        t = 0.1 * k + 0.013
        bx, by = sol.enclosure(t)
        assert _inside(*bx, mp.cos(mp.mpf(t))), t
        assert _inside(*by, mp.sin(mp.mpf(t))), t
    assert sol.max_width < 1e-12


def test_damped_oscillator_matches_matrix_exponential(pool):
    # x' = v, v' = -x - 0.25 v  ⇒  [x, v](t) = expm(t A) [1, 0].
    x, v = pool.symbol("x"), pool.symbol("v")
    sol = ak.verified_ode([v, -x - pool.rational(1, 4) * v], [x, v], None, [1.0, 0.0], (0.0, 10.0))
    a = mp.matrix([[0, 1], [-1, mp.mpf(-1) / 4]])
    for k in range(51):
        t = 0.2 * k
        exact = mp.expm(a * mp.mpf(t)) * mp.matrix([1, 0])
        bx, bv = sol.enclosure(t)
        assert _inside(*bx, exact[0]), t
        assert _inside(*bv, exact[1]), t
    assert sol.max_width < 1e-12


def test_logistic_matches_closed_form(pool):
    # y' = y (1 - y), y(0) = 1/2  ⇒  y = 1 / (1 + e^{-t}).
    y = pool.symbol("y")
    sol = ak.verified_ode([y * (1 - y)], [y], None, [0.5], (0.0, 10.0))
    for k in range(101):
        t = 0.1 * k
        lo, hi = sol.enclosure(t)[0]
        assert _inside(lo, hi, 1 / (1 + mp.exp(-mp.mpf(t)))), t
        assert hi - lo < 1e-13


def test_time_dependent_rhs(pool):
    # y' = cos(t) y  ⇒  y = exp(sin t).
    y, t = pool.symbol("y"), pool.symbol("t")
    sol = ak.verified_ode([ak.cos(t) * y], [y], t, [1.0], (0.0, 6.0))
    for k in range(61):
        tv = 0.1 * k
        lo, hi = sol.enclosure(tv)[0]
        assert _inside(lo, hi, mp.exp(mp.sin(mp.mpf(tv)))), tv
    assert sol.max_width < 1e-12


# ---------------------------------------------------------------------------
# Nonlinear: mpmath.odefun at high precision, scipy at rtol 1e-12
# ---------------------------------------------------------------------------


def test_nonlinear_against_mpmath_odefun(pool):
    y, t = pool.symbol("y"), pool.symbol("t")
    rhs = ak.sin(t) * y - y**3 / 10 + ak.exp(-y) / 5
    sol = ak.verified_ode([rhs], [y], t, [1.0], (0.0, 3.0))
    mp.mp.dps = 30
    f = mp.odefun(lambda tt, yy: mp.sin(tt) * yy - yy**3 / 10 + mp.exp(-yy) / 5, 0, mp.mpf(1))
    for tv in (0.25, 0.5, 1.0, 1.7, 2.2, 3.0):
        lo, hi = sol.enclosure(tv)[0]
        assert _inside(lo, hi, f(mp.mpf(tv))), tv
        assert hi - lo < 1e-12
    mp.mp.dps = 40


def test_interval_initial_condition_encloses_every_solution(pool):
    y, t = pool.symbol("y"), pool.symbol("t")
    sol = ak.verified_ode([ak.sin(t) * y - y**3 / 10], [y], t, [(0.9, 1.1)], (0.0, 3.0))
    mp.mp.dps = 25
    for y0 in (0.9, 0.95, 1.0, 1.05, 1.1):
        f = mp.odefun(lambda tt, yy: mp.sin(tt) * yy - yy**3 / 10, 0, mp.mpf(y0))
        for tv in (1.0, 2.0, 3.0):
            lo, hi = sol.enclosure(tv)[0]
            assert _inside(lo, hi, f(mp.mpf(tv))), (y0, tv)
    mp.mp.dps = 40
    # The enclosure is not wildly pessimistic: the true spread at t = 3 is
    # about 0.06.
    lo, hi = sol.enclosure(3.0)[0]
    assert hi - lo < 0.2


def _lorenz(pool):
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    rhs = [10 * (y - x), x * (28 - z) - y, x * y - pool.rational(8, 3) * z]
    return rhs, [x, y, z]


def test_lorenz_contains_scipy_high_accuracy_solution(pool):
    scipy_integrate = pytest.importorskip("scipy.integrate")
    rhs, xs = _lorenz(pool)
    sol = ak.verified_ode(rhs, xs, None, [1.0, 1.0, 1.0], (0.0, 2.0))

    def f(_t, v):
        return [10 * (v[1] - v[0]), v[0] * (28 - v[2]) - v[1], v[0] * v[1] - 8 / 3 * v[2]]

    ref = scipy_integrate.solve_ivp(
        f, (0.0, 2.0), [1.0, 1.0, 1.0], method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True
    )
    for k in range(201):
        t = 0.01 * k
        r = ref.sol(t)
        box = sol.enclosure(t)
        for i in range(3):
            lo, hi = box[i]
            # scipy is itself only accurate to ~1e-10 here; the slack is for
            # scipy's error, the enclosure's own width is ~1e-13.
            slack = 1e-8 * max(1.0, abs(r[i]))
            assert lo - slack <= r[i] <= hi + slack, (t, i, box[i], r[i])
    # Widths reported (and bounded): Lorenz has a positive Lyapunov exponent,
    # yet at this horizon the enclosure keeps ~12 significant digits.
    widths = {t: max(hi - lo for lo, hi in sol.enclosure(t)) for t in (0.5, 1.0, 1.5, 2.0)}
    print("lorenz enclosure widths:", widths)
    assert sol.max_width < 1e-10


def test_lorenz_contains_mpmath_solution(pool):
    rhs, xs = _lorenz(pool)
    sol = ak.verified_ode(rhs, xs, None, [1.0, 1.0, 1.0], (0.0, 0.5))
    mp.mp.dps = 30
    f = mp.odefun(
        lambda _t, v: [
            10 * (v[1] - v[0]),
            v[0] * (28 - v[2]) - v[1],
            v[0] * v[1] - mp.mpf(8) / 3 * v[2],
        ],
        0,
        [mp.mpf(1), mp.mpf(1), mp.mpf(1)],
    )
    for tv in (0.1, 0.25, 0.5):
        exact = f(mp.mpf(tv))
        box = sol.enclosure(tv)
        for i in range(3):
            assert _inside(*box[i], exact[i]), (tv, i)
    mp.mp.dps = 40


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_blow_up_is_refused_not_enclosed(pool):
    # y' = y², y(0) = 1: y = 1/(1-t) blows up at t = 1.
    y = pool.symbol("y")
    with pytest.raises(ak.ValidatedError) as info:
        ak.verified_ode([y**2], [y], None, [1.0], (0.0, 2.0))
    assert info.value.code == "E-VALIDATED-006"
    assert "blow up" in str(info.value)
    # Up to t = 0.9 everything is enclosed correctly.
    sol = ak.verified_ode([y**2], [y], None, [1.0], (0.0, 0.9))
    for t, box in zip(sol.times, sol.enclosures):
        assert _inside(*box[0], 1 / (1 - mp.mpf(t))), t


def test_stiff_problem_hits_step_limit(pool):
    y = pool.symbol("y")
    with pytest.raises(ak.ValidatedError) as info:
        ak.verified_ode([-(10**8) * y], [y], None, [1.0], (0.0, 1.0), max_steps=100)
    assert info.value.code == "E-VALIDATED-008"


def test_non_lipschitz_start_is_refused(pool):
    # y' = sqrt(y), y(0) = 0 has infinitely many solutions (0 and t²/4 …):
    # there is no "the" solution to enclose with a Taylor method.
    y = pool.symbol("y")
    with pytest.raises(ak.ValidatedError) as info:
        ak.verified_ode([ak.sqrt(y)], [y], None, [0.0], (0.0, 1.0))
    assert info.value.code in ("E-VALIDATED-003", "E-VALIDATED-006")


def test_coded_refusals(pool):
    y, z = pool.symbol("y"), pool.symbol("z")
    cases = [
        (([ak.gamma(y)], [y], None, [1.0], (0.0, 1.0)), "E-VALIDATED-001"),
        (([z], [y], None, [1.0], (0.0, 1.0)), "E-VALIDATED-002"),
        (([ak.log(y)], [y], None, [-1.0], (0.0, 1.0)), "E-VALIDATED-003"),
        (([y], [y], None, [(2.0, 1.0)], (0.0, 1.0)), "E-VALIDATED-005"),
        (([y], [y], None, [1.0], (1.0, 0.0)), "E-VALIDATED-005"),
        (([y, y], [y], None, [1.0], (0.0, 1.0)), "E-VALIDATED-005"),
    ]
    for args, code in cases:
        with pytest.raises(ak.ValidatedError) as info:
            ak.verified_ode(*args)
        assert info.value.code == code, (args, info.value)
    with pytest.raises(TypeError):
        ak.verified_ode([y], [y], None, ["one"], (0.0, 1.0))


def test_enclosure_outside_span_is_refused(pool):
    y = pool.symbol("y")
    sol = ak.verified_ode([-y], [y], None, [1.0], (0.0, 1.0))
    with pytest.raises(ak.ValidatedError) as info:
        sol.enclosure(1.5)
    assert info.value.code == "E-VALIDATED-005"


def test_budget_is_honoured(pool):
    y = pool.symbol("y")
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=3)):
        ak.verified_ode([-y], [y], None, [1.0], (0.0, 100.0))


def test_surface(pool):
    assert "verified_ode" in ak.__all__
    assert "VerifiedOdeSolution" in ak.__all__
    y = pool.symbol("y")
    sol = ak.verified_ode([-y], [y], None, [1.0], (0.0, 1.0))
    assert isinstance(sol, ak.VerifiedOdeSolution)
    assert "VerifiedOdeSolution(" in repr(sol)
    assert sol.times[0] == 0.0
    assert sol.times[-1] == 1.0
    lo, hi = sol.enclosures[-1][0]
    assert lo <= math.exp(-1) <= hi
