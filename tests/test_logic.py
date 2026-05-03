"""V3-3: First-order logic / FOFormula (satisfiability + quantifiers)."""

from __future__ import annotations

import alkahest
from alkahest import ExprPool


def test_satisfiable_and_contradiction():
    p = ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    f = alkahest.And(p.gt(x, z), p.lt(x, z))
    assert alkahest.satisfiable(f) is False


def test_satisfiable_or_cover():
    p = ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    f = alkahest.Or(p.gt(x, z), p.le(x, z))
    r = alkahest.satisfiable(f)
    assert r is True or (isinstance(r, dict) and "x" in r)


def test_forall_exists_pool_api():
    p = ExprPool()
    x = p.symbol("x")
    body = p.gt(x, p.integer(0))
    ex = p.exists(x, body)
    fa = p.forall(x, body)
    assert alkahest.satisfiable(fa) is None  # unsupported fragment
    n = ex.node()
    assert n[0] == "exists"


def test_top_level_forall_exists():
    p = ExprPool()
    x = p.symbol("x")
    body = p.gt(x, p.integer(0))
    e = alkahest.Exists(x, body)
    assert e.node()[0] == "exists"
