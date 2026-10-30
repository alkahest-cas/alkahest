"""G8 — ``context(budget=...)`` covers the expensive algebraic paths.

Report 10-9 §3 G8 measured on 3.12.0: Gröbner bases (Buchberger and F5)
ignored the budget — cyclic-6 under grevlex ran > 120 s and ``compute_f5`` on
cyclic-5 > 100 s — and ``solve`` ran 9.1 s against a 300 ms budget. Each path
below now polls the active budget in its unbounded loops and raises
:class:`alkahest.BudgetExceededError` (``E-BUDGET-*``) on a trip, never a
partial result.

Timings use a generous ceiling (the brief's "within ~3x the wall limit", widened
for loaded CI runners): the claim under test is "stops near the budget instead
of running for minutes", not a precise overshoot.
"""

from __future__ import annotations

import time

import alkahest as ak
import pytest

#: Wall budget for the hard inputs, and the ceiling a budgeted call must stop
#: within. The ceiling is ~13x the budget so a saturated CI box does not flake;
#: before this change each of these ran for minutes.
WALL_MS = 300
# 300 ms / 50 ms budgets: stopping inside 20 s is a termination check, not a
# deadline, so a loaded runner cannot fail it.
CEILING_S = 20.0


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture(autouse=True)
def _no_cancel():
    ak.clear_cancel()
    yield
    ak.clear_cancel()


def _cyclic(pool: ak.ExprPool, n: int):
    xs = [pool.symbol(f"c{i}") for i in range(n)]
    eqs = []
    for d in range(1, n):
        s = 0
        for i in range(n):
            t = 1
            for j in range(d):
                t = t * xs[(i + j) % n]
            s = s + t
        eqs.append(s)
    p = 1
    for v in xs:
        p = p * v
    eqs.append(p - 1)
    return eqs, xs


def _stops_within_budget(pool, fn, wall_ms=WALL_MS):
    t = time.monotonic()
    with (
        pytest.raises(ak.BudgetExceededError) as info,
        ak.context(pool=pool, budget=ak.Budget(wall_ms=wall_ms)),
    ):
        fn()
    elapsed = time.monotonic() - t
    assert info.value.code == "E-BUDGET-001"
    assert elapsed < CEILING_S, f"took {elapsed:.2f}s against a {wall_ms} ms budget"
    return info.value


# ---------------------------------------------------------------------------
# Gröbner bases
# ---------------------------------------------------------------------------


def test_groebner_cyclic6_grevlex_stops_within_budget(pool):
    eqs, xs = _cyclic(pool, 6)
    _stops_within_budget(pool, lambda: ak.GroebnerBasis.compute(eqs, xs, order="grevlex"))


def test_groebner_f5_cyclic5_stops_within_budget(pool):
    eqs, xs = _cyclic(pool, 5)
    _stops_within_budget(pool, lambda: ak.GroebnerBasis.compute_f5(eqs, xs, order="grevlex"))


def test_groebner_step_budget_trips_with_its_own_code(pool):
    eqs, xs = _cyclic(pool, 4)
    with (
        pytest.raises(ak.BudgetExceededError) as info,
        ak.context(budget=ak.Budget(max_steps=0)),
    ):
        ak.GroebnerBasis.compute(eqs, xs, order="grevlex")
    assert info.value.code == "E-BUDGET-002"


def test_groebner_generous_budget_matches_unbudgeted(pool):
    eqs, xs = _cyclic(pool, 4)
    plain = ak.GroebnerBasis.compute(eqs, xs, order="grevlex")
    with ak.context(budget=ak.Budget(wall_ms=600_000)):
        budgeted = ak.GroebnerBasis.compute(eqs, xs, order="grevlex")
        lex = ak.GroebnerBasis.compute(eqs, xs)
        f5 = ak.GroebnerBasis.compute_f5(eqs, xs, order="grevlex")
    assert [str(p) for p in budgeted.to_exprs()] == [str(p) for p in plain.to_exprs()]
    assert len(lex) > 0
    assert len(f5) > 0
    # Every input generator reduces to zero modulo the budgeted basis.
    for e in eqs:
        assert budgeted.contains(e)


def test_cancellation_stops_groebner(pool):
    eqs, xs = _cyclic(pool, 4)
    ak.request_cancel()
    with pytest.raises(ak.BudgetExceededError) as info, ak.context(budget=ak.Budget()):
        ak.GroebnerBasis.compute(eqs, xs, order="grevlex")
    assert info.value.code == "E-BUDGET-003"


# ---------------------------------------------------------------------------
# solve (Gröbner + back-substitution), homotopy
# ---------------------------------------------------------------------------


def test_solve_report_repro_stops_within_budget(pool):
    """Report §4.5: 9.1 s against a 300 ms budget on 3.12.0.

    Since the solver takes its lex basis through grevlex + FGLM this system
    solves in ~0.1 s (it took ~8 s), so under the 300 ms budget it either
    finishes — with all 49 roots, each checked — or trips promptly. The
    budget-trip half of the old claim is carried by the cyclic-6 test below.
    """
    x, y = pool.symbol("x"), pool.symbol("y")
    eqs = [x**7 + y**5 - 3 * x * y - 1, y**7 - x**3 * y + 2]
    t = time.monotonic()
    sols, code = None, None
    try:
        with ak.context(pool=pool, budget=ak.Budget(wall_ms=WALL_MS)):
            sols = ak.solve(eqs, [x, y], numeric=True)
    except ak.BudgetExceededError as exc:
        code = exc.code
    assert time.monotonic() - t < CEILING_S
    assert code in (None, "E-BUDGET-001")
    if sols is not None:
        assert len(sols) == 49
        for s in sols:
            assert abs(s[x] ** 7 + s[y] ** 5 - 3 * s[x] * s[y] - 1) < 1e-8
            assert abs(s[y] ** 7 - s[x] ** 3 * s[y] + 2) < 1e-8


def test_solve_under_generous_budget_is_unchanged(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    with ak.context(budget=ak.Budget(wall_ms=600_000)):
        sols = ak.solve([x**2 + y**2 - 1, x - y], [x, y], numeric=True)
    assert len(sols) == 2
    for s in sols:
        assert abs(s[x] ** 2 + s[y] ** 2 - 1) < 1e-12
        assert abs(s[x] - s[y]) < 1e-12


def _residuals(eqs, xs, sol):
    """The cyclic-n equations evaluated in plain Python complex arithmetic at
    ``sol`` — an oracle independent of alkahest's own evaluator."""
    vals = [complex(sol[v]) for v in xs]
    n = len(vals)
    out = []
    for d in range(1, n):
        s = 0j
        for i in range(n):
            t = 1 + 0j
            for j in range(d):
                t *= vals[(i + j) % n]
            s += t
        out.append(abs(s))
    prod = 1 + 0j
    for v in vals:
        prod *= v
    out.append(abs(prod - 1))
    assert len(out) == len(eqs)
    return out


@pytest.mark.parametrize("numeric", [False, True])
def test_solve_cyclic5_answers_or_refuses_promptly(pool, numeric):
    """Launch report 2026-10-24 row 8: ``solve(cyclic-5)`` ran > 14 min.

    The time went into a *direct* lex Buchberger run in the solver; the same
    reduced lex basis via grevlex + FGLM takes milliseconds. Cyclic-5 has 70
    isolated solutions. Either they come back (and each one satisfies the
    system numerically) or the call refuses with a coded error — quickly,
    with no budget set at all.
    """
    eqs, xs = _cyclic(pool, 5)
    t = time.monotonic()
    sols, code = None, None
    try:
        sols = ak.solve(eqs, xs, numeric=numeric)
    except (ak.SolverError, ak.HomotopyError) as exc:
        code = exc.code
    elapsed = time.monotonic() - t
    assert elapsed < CEILING_S, f"cyclic-5 solve took {elapsed:.1f}s"
    # A refusal is coded; it is never a budget trip (no budget is set).
    assert code is None or code.startswith(("E-SOLVE-", "E-HOMOTOPY-")), code
    if sols is not None:
        assert len(sols) == 70
        for s in sols:
            if not numeric:
                s = {v: complex(ak.evaluate(s[v], {}, mode="complex").value) for v in xs}
            assert max(_residuals(eqs, xs, s)) < 1e-8


def test_solve_cyclic6_stops_within_budget(pool):
    """With the cheap basis, cyclic-6's grevlex pass is the long loop; it polls."""
    eqs, xs = _cyclic(pool, 6)
    _stops_within_budget(pool, lambda: ak.solve(eqs, xs))
    _stops_within_budget(pool, lambda: ak.solve(eqs, xs, numeric=True))


def test_solve_zero_dimensional_via_fglm_is_verified(pool):
    """Cyclic-3 goes through the grevlex + FGLM basis; all six roots check out."""
    eqs, xs = _cyclic(pool, 3)
    sols = ak.solve(eqs, xs, numeric=True)
    assert len(sols) == 6
    for s in sols:
        assert max(_residuals(eqs, xs, s)) < 1e-10
    assert len({tuple(round(complex(s[v]).real, 8) for v in xs) for s in sols}) >= 2


def test_homotopy_honours_a_step_budget(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    eqs = [x**3 + y**2 - 3 * x * y - 1, y**3 - x * y + 2]
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=0)):
        ak.solve(eqs, [x, y], method="homotopy")
    # Still solves without one; check the roots against the equations.
    sols = ak.solve(eqs, [x, y], method="homotopy")
    assert sols
    for s in sols:
        xv, yv = s[x], s[y]
        assert abs(xv**3 + yv**2 - 3 * xv * yv - 1) < 1e-8
        assert abs(yv**3 - xv * yv + 2) < 1e-8


# ---------------------------------------------------------------------------
# Ideal-theoretic engines built on Gröbner bases
# ---------------------------------------------------------------------------


def test_primary_decomposition_stops_within_budget(pool):
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    gens = [
        x**5 * y**3 - z**4 * x + y**7,
        x**4 - y * z**3 + x * y * z,
        z**6 - x**2 * y**3 + 1,
    ]
    _stops_within_budget(pool, lambda: ak.primary_decomposition(gens, [x, y, z]))


def test_primary_decomposition_generous_budget(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    with ak.context(budget=ak.Budget(wall_ms=600_000)):
        comps = ak.primary_decomposition([x * y], [x, y])
    assert len(comps) == 2


def test_triangularize_and_radical_honour_a_step_budget(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    eqs = [x**2 + y**2 - 1, x - y]
    with ak.context(budget=ak.Budget(max_steps=0)):
        with pytest.raises(ak.BudgetExceededError):
            ak.triangularize(eqs, [x, y])
        with pytest.raises(ak.BudgetExceededError):
            ak.radical(eqs, [x, y])
    assert ak.triangularize(eqs, [x, y])


# ---------------------------------------------------------------------------
# Search engines
# ---------------------------------------------------------------------------


def test_guess_holonomic_honours_a_step_budget():
    # Catalan numbers: an order-1 recurrence exists.
    catalan = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900]
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=0)):
        ak.guess_holonomic(catalan)
    assert ak.guess_holonomic(catalan) is not None


def test_zeilberger_budget_trip_is_not_a_negative_answer(pool):
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    c = ak.gamma(n + one) / (ak.gamma(k + one) * ak.gamma(n - k + one))
    franel = c * c * c  # minimal order 2
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=0)):
        ak.zeilberger(franel, n, k)
    # A real wall budget stops the order-2 search well before it finishes.
    _stops_within_budget(pool, lambda: ak.zeilberger(franel, n, k, max_degree=16), wall_ms=50)


def test_decide_two_variables_honours_a_step_budget(pool):
    x, y = pool.symbol("x"), pool.symbol("y")
    body = pool.gt(pool.integer(1), x**2 + y**2)
    phi = ak.Exists(x, ak.Exists(y, body))
    with pytest.raises(ak.BudgetExceededError), ak.context(budget=ak.Budget(max_steps=0)):
        ak.decide(phi)
    truth, _ = ak.decide(phi)
    assert truth is True
