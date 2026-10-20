"""Exact linear and mixed-integer programming: ``linprog``, ``milp``, ``linprog_expr``.

Report 10-9 §3 G10 / §4.4 (finance / OR): "LP / ILP / simplex: nothing present".

Every result is checked twice and independently of the solver:

* its certificate (dual multipliers, Farkas vector or ray) is re-verified here
  in pure-Python ``Fraction`` arithmetic — a proof of the status, not a
  comparison;
* its optimum is compared with an oracle — exact vertex enumeration in
  ``Fraction`` arithmetic always, SciPy/HiGHS when it is installed, and
  brute-force enumeration of the integer points for the integer programmes.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import alkahest as ak
import pytest

F = Fraction


# ---------------------------------------------------------------------------
# Independent certificate checks (pure Python, exact)
# ---------------------------------------------------------------------------


def _norm(c, A_ub, b_ub, A_eq, b_eq, bounds):
    n = len(c)
    A_ub = [list(map(F, r)) for r in (A_ub or [])]
    A_eq = [list(map(F, r)) for r in (A_eq or [])]
    b_ub = list(map(F, b_ub or []))
    b_eq = list(map(F, b_eq or []))
    if bounds is None:
        bounds = [(0, None)] * n
    elif len(bounds) == 2 and not isinstance(bounds[0], (tuple, list)):
        bounds = [tuple(bounds)] * n
    bounds = [(None if lo is None else F(lo), None if hi is None else F(hi)) for lo, hi in bounds]
    return list(map(F, c)), A_ub, b_ub, A_eq, b_eq, bounds


def _dot(a, b):
    return sum((F(x) * F(y) for x, y in zip(a, b)), F(0))


def _feasible(x, A_ub, b_ub, A_eq, b_eq, bounds):
    return (
        all(_dot(r, x) <= b for r, b in zip(A_ub, b_ub))
        and all(_dot(r, x) == b for r, b in zip(A_eq, b_eq))
        and all(
            (lo is None or v >= lo) and (hi is None or v <= hi) for v, (lo, hi) in zip(x, bounds)
        )
    )


def check_certificate(res, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, sense="min"):
    """Re-prove ``res``'s status from its certificate, in Fractions."""
    c, A_ub, b_ub, A_eq, b_eq, bounds = _norm(c, A_ub, b_ub, A_eq, b_eq, bounds)
    n = len(c)
    sgn = 1 if sense == "min" else -1  # multipliers are sign-flipped for max
    if res.status == "optimal":
        x = list(map(F, res.x))
        assert _feasible(x, A_ub, b_ub, A_eq, b_eq, bounds)
        assert F(res.fun) == _dot(c, x)
        y_ub, y_eq = list(map(F, res.ineqlin)), list(map(F, res.eqlin))
        zl, zu = list(map(F, res.lower)), list(map(F, res.upper))
        assert all(sgn * y <= 0 for y in y_ub)
        assert all(sgn * z >= 0 for z in zl)
        assert all(sgn * z <= 0 for z in zu)
        for j in range(n):
            g = zl[j] + zu[j] + sum(A_ub[i][j] * y_ub[i] for i in range(len(A_ub)))
            g += sum(A_eq[i][j] * y_eq[i] for i in range(len(A_eq)))
            assert g == c[j], "stationarity"
        dual = _dot(b_ub, y_ub) + _dot(b_eq, y_eq)
        for j, (lo, hi) in enumerate(bounds):
            if lo is None:
                assert zl[j] == 0
            else:
                dual += lo * zl[j]
                assert zl[j] == 0 or x[j] == lo
            if hi is None:
                assert zu[j] == 0
            else:
                dual += hi * zu[j]
                assert zu[j] == 0 or x[j] == hi
        for i, r in enumerate(A_ub):
            assert y_ub[i] == 0 or _dot(r, x) == b_ub[i], "complementary slackness"
        assert dual == F(res.fun), "strong duality"
    elif res.status == "infeasible":
        f = res.farkas
        y_ub, y_eq = list(map(F, f["ineqlin"])), list(map(F, f["eqlin"]))
        wl, wu = list(map(F, f["lower"])), list(map(F, f["upper"]))
        assert all(v >= 0 for v in y_ub + wl + wu)
        rhs = _dot(b_ub, y_ub) + _dot(b_eq, y_eq)
        for j, (lo, hi) in enumerate(bounds):
            comb = wu[j] - wl[j] + sum(A_ub[i][j] * y_ub[i] for i in range(len(A_ub)))
            comb += sum(A_eq[i][j] * y_eq[i] for i in range(len(A_eq)))
            assert comb == 0
            assert lo is not None or wl[j] == 0
            assert hi is not None or wu[j] == 0
            if lo is not None:
                rhs -= lo * wl[j]
            if hi is not None:
                rhs += hi * wu[j]
        assert rhs < 0, "Farkas: 0 <= negative"
    else:
        assert res.status == "unbounded"
        x, d = list(map(F, res.x)), list(map(F, res.ray))
        assert _feasible(x, A_ub, b_ub, A_eq, b_eq, bounds)
        assert all(_dot(r, d) <= 0 for r in A_ub)
        assert all(_dot(r, d) == 0 for r in A_eq)
        for v, (lo, hi) in zip(d, bounds):
            assert lo is None or v >= 0
            assert hi is None or v <= 0
        assert sgn * _dot(c, d) < 0
    assert res.verify()


def _exact(values):
    for v in values:
        assert isinstance(v, (int, Fraction)), f"{v!r} is not exact"


# ---------------------------------------------------------------------------
# Report repro and basic behaviour
# ---------------------------------------------------------------------------


def test_report_repro_production_planning():
    """§4.4: a production-planning LP solved exactly, with its dual prices."""
    # max 3a + 5b  s.t. a ≤ 4, 2b ≤ 12, 3a + 2b ≤ 18  (Hillier–Lieberman).
    args = {"A_ub": [[1, 0], [0, 2], [3, 2]], "b_ub": [4, 12, 18]}
    res = ak.linprog([3, 5], **args, sense="max")
    assert res.status == "optimal"
    assert res.success
    assert res.x == [2, 6]
    assert res.fun == 36
    # Shadow prices of the three resources: 0, 3/2, 1.
    assert res.ineqlin == [0, F(3, 2), 1]
    _exact(res.x + res.ineqlin + [res.fun])
    check_certificate(res, [3, 5], **args, sense="max")


def test_textbook_fractional_optimum():
    res = ak.linprog([1, 1], A_ub=[[1, 2], [3, 1]], b_ub=[4, 6], sense="max")
    assert res.x == [F(8, 5), F(6, 5)]
    assert res.fun == F(14, 5)
    assert "optimal" in repr(res)


def test_inputs_may_be_fractions_strings_and_exprs():
    p = ak.ExprPool()
    res = ak.linprog(
        [F(1, 2), "1/3"], A_ub=[[1, p.integer(1)]], b_ub=[p.rational(7, 2)], sense="max"
    )
    assert res.fun == F(7, 4)
    check_certificate(res, [F(1, 2), F(1, 3)], A_ub=[[1, 1]], b_ub=[F(7, 2)], sense="max")


def test_floats_are_refused():
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog([1.5, 1], A_ub=[[1, 1]], b_ub=[1])
    assert e.value.code == "E-LP-002"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog([1, 1], A_ub=[[1, 1]], b_ub=[0.1])
    assert e.value.code == "E-LP-002"
    # ±inf in bounds means "no bound" and is accepted.
    res = ak.linprog([1], bounds=[(float("-inf"), float("inf"))], A_ub=[[-1]], b_ub=[3])
    assert res.fun == -3


def test_shape_errors():
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog([1, 1], A_ub=[[1]], b_ub=[1])
    assert e.value.code == "E-LP-001"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog([1, 1], A_ub=[[1, 1]], b_ub=[1, 2])
    assert e.value.code == "E-LP-001"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog([1], sense="sideways")
    assert e.value.code == "E-LP-001"


def test_infeasible_farkas_certificate():
    args = {"A_ub": [[1, 1], [-1, -1]], "b_ub": [1, -3]}
    res = ak.linprog([1, 1], **args)
    assert res.status == "infeasible"
    assert not res.success
    assert res.x is None
    assert res.fun is None
    check_certificate(res, [1, 1], **args)


def test_infeasible_by_bounds_and_equalities():
    args = {"A_eq": [[1, 1]], "b_eq": [5], "bounds": [(0, 2), (None, 2)]}
    res = ak.linprog([0, 0], **args)
    assert res.status == "infeasible"
    check_certificate(res, [0, 0], **args)


def test_unbounded_ray():
    args = {"A_ub": [[1, -1]], "b_ub": [1]}
    res = ak.linprog([-1, -1], **args)
    assert res.status == "unbounded"
    check_certificate(res, [-1, -1], **args)


def test_beale_cycling_example_terminates():
    """Beale (1955) cycles under Dantzig's rule; Bland's rule terminates."""
    c = [F(-3, 4), 20, F(-1, 2), 6]
    args = {
        "A_ub": [[F(1, 4), -8, -1, 9], [F(1, 2), -12, F(-1, 2), 3], [0, 0, 1, 0]],
        "b_ub": [0, 0, 1],
    }
    res = ak.linprog(c, **args)
    assert res.fun == F(-5, 4)
    assert res.x == [1, 0, 1, 0]
    check_certificate(res, c, **args)


def test_degenerate_redundant_equalities():
    args = {
        "A_eq": [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
        "b_eq": [1, 2, 1],
        "A_ub": [[1, 0, 0], [1, 0, 0]],
        "b_ub": [1, 1],
    }
    res = ak.linprog([0, 0, -1], **args)
    assert res.fun == -1
    check_certificate(res, [0, 0, -1], **args)


def test_budget_is_honoured():
    c = [F(-3, 4), 20, F(-1, 2), 6]
    args = {
        "A_ub": [[F(1, 4), -8, -1, 9], [F(1, 2), -12, F(-1, 2), 3], [0, 0, 1, 0]],
        "b_ub": [0, 0, 1],
    }
    with ak.context(budget=ak.Budget(max_steps=2)), pytest.raises(ak.BudgetExceededError) as e:
        ak.linprog(c, **args)
    assert e.value.code == "E-BUDGET-002"


# ---------------------------------------------------------------------------
# Oracles: SymPy (exact, always) and SciPy (when installed)
# ---------------------------------------------------------------------------


def _random_problem(rng, *, free_bounds=True, max_vars=5):
    n = rng.randint(1, max_vars)
    c = [rng.randint(-5, 5) for _ in range(n)]
    A_ub = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(rng.randint(0, 5))]
    b_ub = [rng.randint(-3, 10) for _ in A_ub]
    A_eq = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(rng.randint(0, 2))]
    b_eq = [rng.randint(-4, 4) for _ in A_eq]
    bounds = []
    for _ in range(n):
        k = rng.randint(0, 4) if free_bounds else 3
        if k == 0:
            bounds.append((None, None))
        elif k == 1:
            bounds.append((rng.randint(-3, 0), rng.randint(1, 4)))
        elif k == 2:
            bounds.append((None, rng.randint(-2, 3)))
        else:
            bounds.append((0, None))
    return {"c": c, "A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq, "bounds": bounds}


def _solve_square(rows, rhs):
    """Exact Gauss-Jordan; the unique solution of a square system, or None."""
    n = len(rows)
    m = [[*map(F, r), F(b)] for r, b in zip(rows, rhs)]
    for col in range(n):
        piv = next((r for r in range(col, n) if m[r][col] != 0), None)
        if piv is None:
            return None
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[r][n] for r in range(n)]


def _vertex_oracle(c, A_ub, b_ub, A_eq, b_eq):
    """min c·x over x ≥ 0 by enumerating every vertex, in Fractions.

    Independent of any simplex: with x ≥ 0 the polyhedron is pointed, so it is
    empty iff it has no vertex, and a bounded LP attains its optimum at one.
    Returns ``None`` when there is no vertex (infeasible).
    """
    n = len(c)
    rows = [(list(r), b) for r, b in zip(A_ub, b_ub)] + [
        ([1 if k == j else 0 for k in range(n)], 0) for j in range(n)
    ]
    eqs = [(list(r), b) for r, b in zip(A_eq, b_eq)]
    best = None
    for k in range(n - len(eqs), n + 1):
        if k < 0:
            continue
        for active in itertools.combinations(rows, k):
            for eq_sub in itertools.combinations(eqs, n - k):
                system = list(active) + list(eq_sub)
                x = _solve_square([r for r, _ in system], [b for _, b in system])
                if x is None:
                    continue
                if not _feasible(x, A_ub, b_ub, A_eq, b_eq, [(0, None)] * n):
                    continue
                v = _dot(c, x)
                if best is None or v < best:
                    best = v
    return best


def test_random_problems_against_vertex_enumeration():
    """Exact oracle: every vertex of the feasible region, in Fractions.

    (SymPy's ``sympy.solvers.simplex.linprog`` was tried as the exact oracle
    and rejected: on this very family it returns points that violate the
    equality constraints and wrong optima with bounds, so it is not one.)
    """
    rng = random.Random(20261011)
    seen = set()
    for _ in range(60):
        prob = _random_problem(rng, free_bounds=False, max_vars=4)
        res = ak.linprog(**prob)
        check_certificate(res, **prob)
        seen.add(res.status)
        best = _vertex_oracle(prob["c"], prob["A_ub"], prob["b_ub"], prob["A_eq"], prob["b_eq"])
        if res.status == "infeasible":
            assert best is None, prob
        else:
            assert best is not None, prob
            if res.status == "optimal":
                assert F(res.fun) == best, prob
    assert seen == {"optimal", "infeasible", "unbounded"}


def test_random_problems_against_scipy():
    opt = pytest.importorskip("scipy.optimize")
    rng = random.Random(7)
    codes = {0: "optimal", 2: "infeasible", 3: "unbounded"}
    for _ in range(50):
        prob = _random_problem(rng)
        for s in ("min", "max"):
            res = ak.linprog(**prob, sense=s)
            check_certificate(res, **prob, sense=s)
            sign = 1 if s == "min" else -1
            ref = opt.linprog(
                [sign * v for v in prob["c"]],
                A_ub=prob["A_ub"] or None,
                b_ub=prob["b_ub"] or None,
                A_eq=prob["A_eq"] or None,
                b_eq=prob["b_eq"] or None,
                bounds=prob["bounds"],
                method="highs",
            )
            assert res.status == codes[ref.status], (prob, s, ref.message)
            if res.status == "optimal":
                assert abs(float(res.fun) - sign * ref.fun) <= 1e-9


# ---------------------------------------------------------------------------
# Mixed-integer programming
# ---------------------------------------------------------------------------


def _brute_force(c, A_ub, b_ub, A_eq, b_eq, box, sense="min"):
    best = None
    for x in itertools.product(range(box[0], box[1] + 1), repeat=len(c)):
        if _feasible(list(x), A_ub, b_ub, A_eq, b_eq, [box] * len(c)):
            v = _dot(c, x)
            if best is None or (v < best if sense == "min" else v > best):
                best = v
    return best


def test_knapsack():
    c, w = [10, 13, 7, 8], [[4, 6, 3, 5]]
    res = ak.milp(c, A_ub=w, b_ub=[12], bounds=(0, 1), sense="max")
    assert res.status == "optimal"
    assert res.gap == 0
    assert res.bound == res.fun
    assert res.fun == 25 == _brute_force(c, w, [12], [], [], (0, 1), "max")
    assert res.x == [1, 0, 1, 1]
    assert res.verify()


def test_assignment():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    c = [v for row in cost for v in row]
    A_eq, b_eq = [], []
    for i in range(3):
        A_eq.append([1 if k // 3 == i else 0 for k in range(9)])
        A_eq.append([1 if k % 3 == i else 0 for k in range(9)])
        b_eq += [1, 1]
    res = ak.milp(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
    assert res.fun == 5 == _brute_force(c, [], [], A_eq, b_eq, (0, 1))


def test_integrality_gap_and_mixed_variables():
    args = {"A_ub": [[-1, 1], [3, 2], [2, 3]], "b_ub": [1, 12, 12]}
    assert ak.linprog([0, 1], **args, sense="max").fun == F(14, 5)
    assert ak.milp([0, 1], **args, sense="max").fun == 2
    # x continuous, y integer: 2x + 2y = 3, min y.
    res = ak.milp([0, 1], A_eq=[[2, 2]], b_eq=[3], integrality=[0, 1])
    assert res.x == [F(3, 2), 0]


def test_milp_infeasible_and_unbounded():
    res = ak.milp([1], A_eq=[[2]], b_eq=[1])
    assert res.status == "infeasible"
    assert res.x is None
    res = ak.milp([-1, 0], A_ub=[[1, -2]], b_ub=[F(1, 2)])
    assert res.status == "unbounded"
    assert all(isinstance(v, int) for v in res.x + res.ray)
    assert res.verify()


def test_milp_node_limit_reports_incumbent():
    c, w = [10, 13, 7, 8, 9, 11], [[4, 6, 3, 5, 4, 5]]
    with pytest.raises(ak.LinprogError) as e:
        ak.milp(c, A_ub=w, b_ub=[13], bounds=(0, 1), sense="max", max_nodes=3, cut_rounds=0)
    err = e.value
    assert err.code == "E-LP-005"
    assert err.nodes == 3
    if err.incumbent is not None:
        assert _feasible(err.incumbent, [[F(v) for v in w[0]]], [F(13)], [], [], [(0, 1)] * 6)
        assert err.bound >= err.incumbent_fun
    full = ak.milp(c, A_ub=w, b_ub=[13], bounds=(0, 1), sense="max")
    assert full.fun == _brute_force(c, w, [13], [], [], (0, 1), "max")


def test_milp_budget_carries_incumbent():
    c, w = [10, 13, 7, 8, 9, 11], [[4, 6, 3, 5, 4, 5]]
    with ak.context(budget=ak.Budget(max_steps=40)), pytest.raises(ak.BudgetExceededError) as e:
        ak.milp(c, A_ub=w, b_ub=[13], bounds=(0, 1), sense="max")
    assert e.value.code == "E-BUDGET-002"
    assert hasattr(e.value, "incumbent")


def test_random_milps_against_brute_force():
    rng = random.Random(99)
    for _ in range(40):
        n = rng.randint(1, 3)
        c = [rng.randint(-6, 6) for _ in range(n)]
        A_ub = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(rng.randint(1, 4))]
        b_ub = [rng.randint(-2, 9) for _ in A_ub]
        sense = rng.choice(["min", "max"])
        res = ak.milp(c, A_ub=A_ub, b_ub=b_ub, bounds=(-3, 3), sense=sense)
        expect = _brute_force(c, A_ub, b_ub, [], [], (-3, 3), sense)
        if expect is None:
            assert res.status == "infeasible"
        else:
            assert res.status == "optimal"
            assert res.fun == expect
            assert res.verify()


def test_random_milps_against_scipy():
    opt = pytest.importorskip("scipy.optimize")
    np = pytest.importorskip("numpy")
    rng = random.Random(5)
    for _ in range(30):
        n = rng.randint(2, 5)
        c = [rng.randint(-6, 6) for _ in range(n)]
        A = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(rng.randint(1, 4))]
        b = [rng.randint(0, 12) for _ in A]
        integ = [rng.randint(0, 1) for _ in range(n)]
        res = ak.milp(c, A_ub=A, b_ub=b, bounds=(0, 4), integrality=integ)
        ref = opt.milp(
            c,
            constraints=opt.LinearConstraint(np.array(A), -np.inf, np.array(b)),
            integrality=np.array(integ),
            bounds=opt.Bounds(0, 4),
        )
        assert ref.status == 0
        assert res.status == "optimal"
        assert abs(float(res.fun) - ref.fun) <= 1e-9


# ---------------------------------------------------------------------------
# Symbolic front end
# ---------------------------------------------------------------------------


@pytest.fixture
def xy():
    p = ak.ExprPool()
    return p, p.symbol("x"), p.symbol("y")


def test_linprog_expr(xy):
    p, x, y = xy
    zero = p.integer(0)
    cons = [
        p.le(x + y, p.integer(4)),
        p.le(x + 3 * y, p.integer(6)),
        p.ge(x, zero),
        p.ge(y, zero),
        p.le(x, p.integer(3)),
    ]
    res = ak.linprog_expr(3 * x + 2 * y, cons, [x, y], sense="max")
    assert res.fun == 11
    assert res.solution == {x: 3, y: 1}
    prog = res.program()
    check_certificate(
        res,
        prog["c"],
        prog["A_ub"],
        prog["b_ub"],
        prog["A_eq"],
        prog["b_eq"],
        prog["bounds"],
        sense="max",
    )


def test_linprog_expr_rational_and_equality(xy):
    p, x, y = xy
    half = p.rational(1, 2)
    cons = [p.pred_eq(x + y, p.integer(1)), p.ge(x, half * y), p.ge(y, p.integer(0))]
    res = ak.linprog_expr(y, cons, [x, y], sense="max")
    assert res.solution == {x: F(1, 3), y: F(2, 3)}


def test_linprog_expr_integer(xy):
    p, x, y = xy
    cons = [
        p.le(-x + y, p.integer(1)),
        p.le(3 * x + 2 * y, p.integer(12)),
        p.le(2 * x + 3 * y, p.integer(12)),
        p.ge(x, p.integer(0)),
        p.ge(y, p.integer(0)),
    ]
    res = ak.linprog_expr(y, cons, [x, y], sense="max", integer=True)
    assert isinstance(res, ak.MILPResult)
    assert res.fun == 2


def test_linprog_expr_refusals(xy):
    p, x, y = xy
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog_expr(x * y, [], [x, y])
    assert e.value.code == "E-LP-003"
    a = p.symbol("a")
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog_expr(a * x, [], [x])
    assert e.value.code == "E-LP-003"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog_expr(x, [p.lt(x, p.integer(1))], [x])
    assert e.value.code == "E-LP-004"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog_expr(ak.sqrt(p.integer(2)) * x, [], [x])
    assert e.value.code == "E-LP-002"
    # Free variables: min x with no constraint is unbounded, not 0.
    assert ak.linprog_expr(x, [], [x]).status == "unbounded"
