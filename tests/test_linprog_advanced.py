"""Revised simplex with warm starts, Gomory cuts in ``milp``, and parametric LP.

Follow-ups to the exact ``linprog`` / ``milp`` of report 10-9 §3 G10. Every
result is checked independently of the solver, as in ``test_linprog.py``:
certificates are re-proved in pure-Python ``Fraction`` arithmetic, optima are
compared with exact vertex enumeration (and SciPy/HiGHS when installed), cuts
are checked against every integer point of a box by brute force, and
parametric results against independent solves at sampled parameter values.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import alkahest as ak
import pytest
from test_linprog import _feasible, _random_problem, _vertex_oracle, check_certificate

F = Fraction


def _sparse_problem(rng, m, n, per_row=4):
    """max c·x, sparse rows with positive entries, 0 ≤ x ≤ 10, feasible at x = 1."""
    c = [rng.randint(-9, 9) for _ in range(n)]
    A_ub, b_ub = [], []
    for _ in range(m):
        row = [0] * n
        for _ in range(per_row):
            row[rng.randrange(n)] = rng.randint(1, 9)
        A_ub.append(row)
        b_ub.append(sum(row) + rng.randint(0, 20))
    return {"c": c, "A_ub": A_ub, "b_ub": b_ub, "bounds": (0, 10)}


# ---------------------------------------------------------------------------
# Revised simplex
# ---------------------------------------------------------------------------


def test_revised_matches_dense_and_vertex_enumeration():
    rng = random.Random(1912)
    seen = set()
    for _ in range(80):
        prob = _random_problem(rng, free_bounds=False, max_vars=4)
        dense = ak.linprog(**prob, method="dense")
        rev = ak.linprog(**prob, method="revised")
        assert (dense.method, rev.method) == ("dense", "revised")
        check_certificate(rev, **prob)
        assert rev.status == dense.status, prob
        assert rev.fun == dense.fun, prob
        best = _vertex_oracle(prob["c"], prob["A_ub"], prob["b_ub"], prob["A_eq"], prob["b_eq"])
        if rev.status == "optimal":
            assert F(rev.fun) == best
        seen.add(rev.status)
    assert seen == {"optimal", "infeasible", "unbounded"}


def test_revised_with_general_bounds():
    rng = random.Random(77)
    for _ in range(80):
        prob = _random_problem(rng)
        for sense in ("min", "max"):
            dense = ak.linprog(**prob, sense=sense, method="dense")
            rev = ak.linprog(**prob, sense=sense, method="revised")
            check_certificate(rev, **prob, sense=sense)
            assert (rev.status, rev.fun) == (dense.status, dense.fun), prob


def test_medium_sparse_problem_auto_uses_revised():
    rng = random.Random(5)
    prob = _sparse_problem(rng, 80, 80)
    res = ak.linprog(**prob, sense="max")
    assert res.method == "revised"
    assert res.status == "optimal"
    check_certificate(res, **prob, sense="max")
    opt = pytest.importorskip("scipy.optimize")
    ref = opt.linprog([-v for v in prob["c"]], A_ub=prob["A_ub"], b_ub=prob["b_ub"], bounds=(0, 10))
    assert abs(float(res.fun) + ref.fun) <= 1e-7


def test_bad_method_and_basis_are_refused():
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog([1], method="interior")
    assert e.value.code == "E-LP-001"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog([1], basis=[("nonsense", 0)])
    assert e.value.code == "E-LP-001"


# ---------------------------------------------------------------------------
# Warm starts
# ---------------------------------------------------------------------------


def test_warm_start_after_rhs_change():
    rng = random.Random(42)
    warmed = 0
    for _ in range(25):
        prob = _sparse_problem(rng, 14, 12, per_row=3)
        first = ak.linprog(**prob, sense="max", method="revised")
        b2 = [b - (rng.randint(1, 12) if rng.random() < 0.4 else 0) for b in prob["b_ub"]]
        prob2 = {**prob, "b_ub": b2}
        warm = ak.linprog(**prob2, sense="max", basis=first.basis)
        cold = ak.linprog(**prob2, sense="max", method="dense")
        check_certificate(warm, **prob2, sense="max")
        assert (warm.status, warm.fun) == (cold.status, cold.fun)
        warmed += warm.warm_started
    assert warmed >= 22


def test_warm_start_from_result_and_added_row():
    args = {"A_ub": [[1, 1], [1, 0], [0, 1]], "b_ub": [4, 3, 3]}
    first = ak.linprog([1, 1], **args, sense="max", method="revised")
    assert first.fun == 4
    assert all(isinstance(k, str) and isinstance(i, int) for k, i in first.basis)
    # A cutting row x + y >= 5 makes it infeasible; the dual simplex proves it.
    args2 = {"A_ub": args["A_ub"] + [[-1, -1]], "b_ub": args["b_ub"] + [-5]}
    res = ak.linprog([1, 1], **args2, sense="max", basis=first)
    assert res.warm_started
    assert res.status == "infeasible"
    check_certificate(res, [1, 1], **args2, sense="max")
    # And x + y <= 7/2 keeps it feasible.
    args3 = {"A_ub": args["A_ub"] + [[1, 1]], "b_ub": args["b_ub"] + [F(7, 2)]}
    res = ak.linprog([1, 1], **args3, sense="max", basis=first.basis)
    assert res.warm_started
    assert res.fun == F(7, 2)
    check_certificate(res, [1, 1], **args3, sense="max")


# ---------------------------------------------------------------------------
# Gomory cuts
# ---------------------------------------------------------------------------


def _integer_points(A_ub, b_ub, A_eq, b_eq, bounds, box=4):
    n = len(bounds)
    for x in itertools.product(range(-box, box + 1), repeat=n):
        if _feasible(list(x), A_ub, b_ub, A_eq, b_eq, bounds):
            yield list(x)


def test_gomory_cut_closes_simple_gap():
    res = ak.milp([1, 1], A_ub=[[2, 2]], b_ub=[3], sense="max")
    assert res.fun == 1
    assert res.cuts, "the root should have produced a cut"
    assert res.nodes == 1
    plain = ak.milp([1, 1], A_ub=[[2, 2]], b_ub=[3], sense="max", cut_rounds=0)
    assert plain.fun == 1
    assert plain.cuts == []


def test_gomory_cuts_never_remove_integer_points():
    """Brute force: every cut holds at every integer-feasible point of the box."""
    rng = random.Random(606)
    with_cuts = 0
    for _ in range(120):
        n = rng.randint(2, 3)
        c = [rng.randint(-6, 6) for _ in range(n)]
        A_ub = [
            [F(rng.randint(-7, 7), rng.randint(1, 3)) for _ in range(n)]
            for _ in range(rng.randint(1, 4))
        ]
        b_ub = [F(rng.randint(-2, 13), rng.randint(1, 2)) for _ in A_ub]
        A_eq, b_eq = [], []
        if rng.random() < 0.3:
            A_eq = [[rng.randint(-2, 2) for _ in range(n)]]
            b_eq = [rng.randint(-2, 2)]
        bounds = [rng.choice([(F(-7, 2), F(10, 3)), (-3, 3), (F(-5, 2), 3)]) for _ in range(n)]
        sense = rng.choice(["min", "max"])
        pts = list(
            _integer_points(A_ub, b_ub, A_eq, b_eq, [(F(lo), F(hi)) for lo, hi in bounds], box=4)
        )
        res = ak.milp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, sense=sense)
        if not pts:
            assert res.status == "infeasible"
            continue
        vals = [sum(F(ci) * xi for ci, xi in zip(c, x)) for x in pts]
        assert res.fun == (min(vals) if sense == "min" else max(vals))
        assert res.verify()
        with_cuts += bool(res.cuts)
        for row, rhs in res.cuts:
            for x in pts:
                assert sum(F(a) * v for a, v in zip(row, x)) <= rhs, (row, rhs, x)
    assert with_cuts > 20


def test_mixed_integer_cuts_agree_with_plain_branch_and_bound():
    rng = random.Random(3131)
    for _ in range(50):
        n = rng.randint(2, 4)
        c = [rng.randint(-6, 6) for _ in range(n)]
        A_ub = [
            [F(rng.randint(-7, 7), rng.randint(1, 2)) for _ in range(n)]
            for _ in range(rng.randint(1, 4))
        ]
        b_ub = [F(rng.randint(-2, 13), rng.randint(1, 3)) for _ in A_ub]
        bounds = [(F(-5, 2), 4)] * n
        integrality = [rng.randint(0, 2) > 0 for _ in range(n)]
        sense = rng.choice(["min", "max"])
        args = {"A_ub": A_ub, "b_ub": b_ub, "bounds": bounds, "sense": sense}
        cut = ak.milp(c, **args, integrality=integrality)
        plain = ak.milp(c, **args, integrality=integrality, cut_rounds=0, warm_start=False)
        assert (cut.status, cut.fun) == (plain.status, plain.fun)
        if plain.status == "optimal":
            for row, rhs in cut.cuts:
                assert sum(F(a) * F(v) for a, v in zip(row, plain.x)) <= rhs


def test_milp_with_cuts_matches_scipy():
    opt = pytest.importorskip("scipy.optimize")
    import numpy as np

    rng = random.Random(99)
    for _ in range(20):
        n = rng.randint(2, 5)
        c = [rng.randint(-8, 8) for _ in range(n)]
        A = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(rng.randint(1, 4))]
        b = [rng.randint(0, 15) for _ in A]
        integ = [rng.randint(0, 1) for _ in range(n)]
        res = ak.milp(c, A_ub=A, b_ub=b, bounds=(0, 6), integrality=integ)
        ref = opt.milp(
            c,
            constraints=opt.LinearConstraint(np.array(A), -np.inf, np.array(b)),
            bounds=opt.Bounds(0, 6),
            integrality=np.array(integ),
        )
        assert res.status == "optimal"
        assert ref.status == 0
        assert abs(float(res.fun) - ref.fun) <= 1e-7


# ---------------------------------------------------------------------------
# Parametric LP
# ---------------------------------------------------------------------------


def _check_parametric(res, prob, sense, *, c_theta=None, b_ub_theta=None, b_eq_theta=None):
    """Compare the parametric result with independent solves at sample θ."""
    samples = set()
    for b in res.breakpoints:
        samples |= {F(b), F(b) - F(1, 7), F(b) + F(1, 7)}
    bps = [F(b) for b in res.breakpoints]
    samples |= {(u + v) / 2 for u, v in zip(bps, bps[1:])}
    samples |= {F(t) for t in (-40, -5, -1, 0, 1, 4, 40)}
    regs = res.regions
    for r1, r2 in zip(regs, regs[1:]):
        assert r1["theta"][1] == r2["theta"][0]
    for t in sorted(samples):
        status = res.status_at(t)
        if status is None:
            continue
        p = dict(prob)
        if c_theta is not None:
            p["c"] = [F(a) + t * F(d) for a, d in zip(prob["c"], c_theta)]
        if b_ub_theta is not None:
            p["b_ub"] = [F(a) + t * F(d) for a, d in zip(prob["b_ub"], b_ub_theta)]
        if b_eq_theta is not None:
            p["b_eq"] = [F(a) + t * F(d) for a, d in zip(prob["b_eq"], b_eq_theta)]
        ref = ak.linprog(**p, sense=sense, method="dense")
        assert status == ref.status, (t, res)
        assert res.value_at(t) == ref.fun, t
        if status == "optimal":
            x = res.x_at(t)
            norm_bounds = p.get("bounds")
            n = len(p["c"])
            if norm_bounds is None:
                norm_bounds = [(0, None)] * n
            elif len(norm_bounds) == 2 and not isinstance(norm_bounds[0], (tuple, list)):
                norm_bounds = [tuple(norm_bounds)] * n
            nb = [
                (None if lo is None else F(lo), None if hi is None else F(hi))
                for lo, hi in norm_bounds
            ]
            assert _feasible(
                list(map(F, x)),
                p.get("A_ub") or [],
                p.get("b_ub") or [],
                p.get("A_eq") or [],
                p.get("b_eq") or [],
                nb,
            )
    assert res.verify()


def test_parametric_rhs_textbook():
    prob = {"c": [1, 1], "A_ub": [[1, 2], [3, 1]], "b_ub": [4, 6]}
    res = ak.linprog_parametric(**prob, b_ub_theta=[1, 0], sense="max")
    assert res.kind == "rhs"
    assert res.breakpoints == [-4, -2, 8]
    statuses = [r["status"] for r in res.regions]
    assert statuses == ["infeasible", "optimal", "optimal", "optimal"]
    assert [r["value"] for r in res.regions[1:]] == [(4, 1), (F(14, 5), F(2, 5)), (6, 0)]
    assert res.value_at(-5) is None
    assert res.status_at(-5) == "infeasible"
    assert res.value_at(0) == F(14, 5)
    _check_parametric(res, prob, "max", b_ub_theta=[1, 0])


def test_parametric_objective_textbook():
    prob = {"c": [1, 0], "A_ub": [[1, 2], [3, 1]], "b_ub": [4, 6]}
    res = ak.linprog_parametric(**prob, c_theta=[0, 1], sense="max")
    assert res.kind == "objective"
    assert res.breakpoints == [F(1, 3), 2]
    assert res.x_at(1) == [F(8, 5), F(6, 5)]
    _check_parametric(res, prob, "max", c_theta=[0, 1])


def test_parametric_random_against_pointwise_solves():
    rng = random.Random(2468)
    for k in range(40):
        prob = _random_problem(rng, max_vars=4)
        sense = rng.choice(["min", "max"])
        rng_range = rng.choice([None, (-6, 6), (rng.randint(-3, 0), None)])
        if k % 2:
            ct = [rng.randint(-3, 3) for _ in prob["c"]]
            res = ak.linprog_parametric(**prob, sense=sense, c_theta=ct, theta_range=rng_range)
            _check_parametric(res, prob, sense, c_theta=ct)
        else:
            bu = [rng.randint(-3, 3) for _ in prob["b_ub"]]
            be = [rng.randint(-2, 2) for _ in prob["b_eq"]]
            res = ak.linprog_parametric(
                **prob, sense=sense, b_ub_theta=bu, b_eq_theta=be, theta_range=rng_range
            )
            _check_parametric(res, prob, sense, b_ub_theta=bu, b_eq_theta=be)


def test_parametric_expressions_and_refusals():
    p = ak.ExprPool()
    x, y, t = p.symbol("x"), p.symbol("y"), p.symbol("t")
    zero = p.integer(0)
    cons = [
        p.le(x + 2 * y, p.integer(4) + t),
        p.le(3 * x + y, p.integer(6)),
        p.ge(x, zero),
        p.ge(y, zero),
    ]
    res = ak.linprog_expr(x + y, cons, [x, y], sense="max", parameter=t)
    assert isinstance(res, ak.ParametricLPResult)
    assert res.breakpoints == [-4, -2, 8]
    assert res.solution_at(0) == {x: F(8, 5), y: F(6, 5)}
    assert res.program()["b_ub_theta"] == [1, 0, 0, 0]
    # Objective (2 + t)·x + y over the second family.
    res = ak.linprog_expr(
        (p.integer(2) + t) * x + y,
        cons[1:],
        [x, y],
        sense="max",
        parameter=t,
        parameter_range=(-5, 5),
    )
    assert res.kind == "objective"
    assert res.verify()
    # Refusals: parameter in the matrix, in both places, squared, with integers.
    for obj, cs in [
        (x + y, [p.le(t * x, p.integer(1))]),
        ((p.integer(2) + t) * x, cons),
        (t**2 * x, cons[1:]),
    ]:
        with pytest.raises(ak.LinprogError) as e:
            ak.linprog_expr(obj, cs, [x, y], parameter=t)
        assert e.value.code == "E-LP-007"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog_expr(x + y, cons, [x, y], parameter=t, integer=True)
    assert e.value.code == "E-LP-007"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog_parametric([1, 1], A_ub=[[1, 1]], b_ub=[1], c_theta=[1, 0], b_ub_theta=[1])
    assert e.value.code == "E-LP-007"
    with pytest.raises(ak.LinprogError) as e:
        ak.linprog_parametric([1, 1], c_theta=[1])
    assert e.value.code == "E-LP-001"
