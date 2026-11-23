"""Milestone M2: second-order linear PDEs in two variables.

Canonical forms, factored constant-coefficient operators, d'Alembert, the
Laplace cascade, and the corpus in ``tests/pde_corpus/second_order.py``.
Every returned solution must be ``exactly_verified``; anything else must be a
refusal with an ``E-PDE-1xx`` code.
"""

from __future__ import annotations

import os
import sys

import alkahest as ak
import pytest

akx = pytest.importorskip("alkahest.experimental")
if not hasattr(akx, "solve_second_order"):
    pytest.skip("alkahest built without the PDE M2 layer", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pde_corpus"))

from second_order import CORPUS  # noqa: E402
from translate import to_alkahest  # noqa: E402

D = akx.D


@pytest.fixture
def P():
    pool = ak.ExprPool()

    class NS:
        pass

    ns = NS()
    ns.pool = pool
    ns.x, ns.y, ns.t, ns.c = (pool.symbol(n) for n in ("x", "y", "t", "c"))
    ns.u = akx.Function("u")
    return ns


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", CORPUS, ids=[e["id"] for e in CORPUS])
def test_corpus_entry(entry):
    pool = ak.ExprPool()
    eq, u, ics = to_alkahest(entry, pool)
    expect = entry["expect"]
    if expect.startswith("E-PDE-"):
        with pytest.raises(akx.PdeError) as ei:
            akx.solve_second_order(eq, u, ics=ics or None)
        assert expect in str(ei.value)
        return
    s = akx.solve_second_order(eq, u, ics=ics or None)
    assert s.kind == expect, (s.method, str(s.solution))
    # the invariant: nothing unverified is ever returned
    assert s.verification["status"] == "exactly_verified"
    assert s.verification["verified"] is True
    # and an independent re-check through the public M0 entry point
    pde = akx.PDE(eq, u)
    for lhs, rhs in ics:
        pde = pde.ic(lhs, rhs)
    assert akx.verify_pde_solution(pde, s.solution)["status"] == "exactly_verified"
    if s.kind == "general":
        assert len(s.arbitrary_functions) == 2, str(s.solution)


def test_corpus_score_floor():
    """The verified-solve count may only go up."""
    solved = 0
    for entry in CORPUS:
        pool = ak.ExprPool()
        eq, u, ics = to_alkahest(entry, pool)
        try:
            s = akx.solve_second_order(eq, u, ics=ics or None)
        except akx.PdeError:
            continue
        if s.verification["status"] == "exactly_verified":
            solved += 1
    assert solved >= sum(1 for e in CORPUS if not e["expect"].startswith("E-"))


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------


def test_canonical_form_hyperbolic_constant(P):
    u = P.u(P.x, P.y)
    cf = akx.canonical_form(D(u, P.x, 2) - 4 * D(u, P.y, 2), u)
    assert cf.type == "hyperbolic"
    xi, eta = cf.new_variables
    w = cf.unknown
    # w_ξη = 0
    assert cf.equation == D(w, xi, eta)
    assert cf.in_new_variables


def test_canonical_form_parabolic_variable(P):
    x, y = P.x, P.y
    u = P.u(x, y)
    cf = akx.canonical_form(x**2 * D(u, x, 2) + 2 * x * y * D(u, x, y) + y**2 * D(u, y, 2), u)
    assert cf.type == "parabolic"
    assert len(cf.new_variables) == 2
    assert cf.in_new_variables
    # the principal part is w_ηη
    assert cf.coefficients[2] == P.pool.integer(1)
    assert cf.coefficients[0] == P.pool.integer(0)
    assert cf.coefficients[1] == P.pool.integer(0)


def test_canonical_form_elliptic_is_real(P):
    u = P.u(P.x, P.y)
    cf = akx.canonical_form(D(u, P.x, 2) + 4 * D(u, P.x, P.y) + 5 * D(u, P.y, 2), u)
    assert cf.type == "elliptic"
    a, b = cf.transformation
    assert "I" not in str(a)
    assert "I" not in str(b)
    assert cf.coefficients[0] == cf.coefficients[2]


def test_canonical_forms_by_region_tricomi(P):
    x, y = P.x, P.y
    u = P.u(x, y)
    with pytest.raises(akx.PdeError, match="E-PDE-102"):
        akx.canonical_form(y * D(u, x, 2) + D(u, y, 2), u)
    forms = akx.canonical_forms(y * D(u, x, 2) + D(u, y, 2), u)
    types = sorted(f.type for f in forms)
    assert types == ["elliptic", "hyperbolic"]
    for f in forms:
        assert f.region is not None


def test_canonical_form_abbasi_403(P):
    x, y = P.x, P.y
    u = P.u(x, y)
    eq = D(u, x, 2) - 2 * ak.sin(x) * D(u, x, y) - ak.cos(x) ** 2 * D(u, y, 2) - ak.cos(x) * D(u, y)
    cf = akx.canonical_form(eq, u)
    assert cf.type == "hyperbolic"
    xi, eta = cf.new_variables
    assert cf.equation == D(cf.unknown, xi, eta)


# ---------------------------------------------------------------------------
# Laplace cascade
# ---------------------------------------------------------------------------


def test_laplace_invariants(P):
    x, y = P.x, P.y
    u = P.u(x, y)
    h, k = akx.laplace_invariants(D(u, x, y) + y * D(u, x) + x * D(u, y) + x * y * u, u)
    assert str(h) == "0"
    assert str(k) == "0"


def test_laplace_cascade_steps(P):
    x, y = P.x, P.y
    u = P.u(x, y)
    for n, steps in ((2, 1), (6, 2), (12, 3)):
        s = akx.laplace_cascade(D(u, x, y) - n * u / (x + y) ** 2, u)
        assert s.kind == "general"
        assert s.verification["status"] == "exactly_verified"
        assert f"({steps} step(s))" in s.method


def test_laplace_cascade_budget(P):
    x, y = P.x, P.y
    u = P.u(x, y)
    eq = D(u, x, y) - 12 * u / (x + y) ** 2
    with pytest.raises(akx.PdeError, match="E-PDE-103"):
        akx.laplace_cascade(eq, u, max_steps=2)
    with pytest.raises(akx.PdeError, match="E-PDE-103"):
        akx.laplace_cascade(D(u, x, y) - u, u, max_steps=6)


def test_laplace_cascade_needs_uxy_form(P):
    u = P.u(P.x, P.y)
    with pytest.raises(akx.PdeError, match="E-PDE-108"):
        akx.laplace_cascade(D(u, P.x, 2) - D(u, P.y, 2), u)


# ---------------------------------------------------------------------------
# Results, refusals, the PDE object
# ---------------------------------------------------------------------------


def test_solution_object(P):
    u = P.u(P.x, P.t)
    s = akx.solve_second_order(D(u, P.t, 2) - P.c**2 * D(u, P.x, 2), u)
    assert s.value == s.solution
    assert s.kind == "general"
    assert s.arbitrary_functions == ["F1", "F2"]
    assert s.steps
    d = s.to_dict()
    assert d["verification"]["pde"][0]["status"] == "proved_zero"
    assert "SecondOrderSolution" in repr(s)


def test_pde_object_with_cauchy_data(P):
    x, t = P.x, P.t
    u = P.u(x, t)
    pde = (
        akx.PDE(D(u, t, 2) - D(u, x, 2), u)
        .ic(P.u(x, 0), ak.sin(x))
        .ic(ak.subs(D(u, t), {t: P.pool.integer(0)}), ak.cos(x))
    )
    s = akx.solve_second_order(pde)
    assert s.method == "dalembert"
    assert s.verification["status"] == "exactly_verified"
    assert len(s.verification["ic"]) == 2


def test_boundary_value_problem_is_refused(P):
    x, t = P.x, P.t
    u = P.u(x, t)
    pde = akx.PDE(D(u, t, 2) - D(u, x, 2), u).bc(P.u(0, t), 0).bc(P.u(1, t), 0)
    with pytest.raises(akx.PdeError, match="E-PDE-107"):
        akx.solve_second_order(pde)


def test_not_second_order(P):
    u = P.u(P.x, P.y)
    with pytest.raises(akx.PdeError, match="E-PDE-100"):
        akx.solve_second_order(D(u, P.x) + D(u, P.y), u)
    v = akx.Function("v")(P.x, P.y, P.t)
    with pytest.raises(akx.PdeError, match="E-PDE-100"):
        akx.solve_second_order(D(v, P.x, 2) + D(v, P.y, 2) + D(v, P.t, 2), v)


def test_particular_family_can_be_disabled(P):
    u = P.u(P.x, P.t)
    heat = D(u, P.t) - D(u, P.x, 2)
    assert akx.solve_second_order(heat, u).kind == "particular"
    with pytest.raises(akx.PdeError, match="E-PDE-104"):
        akx.solve_second_order(heat, u, allow_particular=False)


def test_prefix(P):
    u = P.u(P.x, P.y)
    s = akx.solve_second_order(D(u, P.x, P.y), u, prefix="G")
    assert s.arbitrary_functions == ["G1", "G2"]
