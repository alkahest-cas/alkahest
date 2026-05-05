"""V2-13 — Differential algebra / Rosenfeld–Gröbner (groebner feature)."""

import alkahest
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "rosenfeld_groebner"),
    reason="native module built without groebner feature",
)


def test_rosenfeld_groebner_inconsistent_linear():
    pool = alkahest.ExprPool()
    t = pool.symbol("t")
    y = pool.symbol("y")
    dy = pool.symbol("dy/dt")
    eq1 = dy - y
    eq2 = dy - y - pool.integer(1)
    dae = alkahest.DAE.new([eq1, eq2], [y], [dy], t)
    r = alkahest.rosenfeld_groebner(dae, max_prolong_rounds=1)
    assert r.consistent is False


def test_rosenfeld_groebner_ode_consistent_smoke():
    pool = alkahest.ExprPool()
    t = pool.symbol("t")
    x = pool.symbol("x")
    dx = pool.symbol("dx/dt")
    eq = dx - x
    dae = alkahest.DAE.new([eq], [x], [dx], t)
    r = alkahest.rosenfeld_groebner(dae, max_prolong_rounds=0)
    assert r.consistent is True
    assert r.truncated is True


def test_dae_index_reduce_prefers_pantelides():
    pool = alkahest.ExprPool()
    t = pool.symbol("t")
    x = pool.symbol("x")
    dx = pool.symbol("dx/dt")
    eq = dx - x
    dae = alkahest.DAE.new([eq], [x], [dx], t)
    out = alkahest.dae_index_reduce(dae)
    assert out.used_pantelides is True
    assert out.used_rosenfeld_groebner is False
    reduced = alkahest.pantelides(dae)
    assert out.dae().n_equations() == reduced.n_equations()
