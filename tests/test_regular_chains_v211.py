"""V2-11 — Regular chains / triangular decomposition."""

import alkahest
import pytest
from alkahest import ExprPool

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "triangularize"),
    reason="native module built without groebner feature",
)


def test_triangularize_linear():
    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    neg_one = pool.integer(-1)
    eq1 = x + y + neg_one
    eq2 = x + neg_one * y
    chains = alkahest.triangularize([eq1, eq2], [x, y])
    assert len(chains) == 1
    c0 = chains[0]
    assert c0.n_vars == 2
    assert len(c0) >= 1
    assert len(c0.polys()) >= 1


def test_triangularize_splits_x2_minus_1():
    pool = ExprPool()
    x = pool.symbol("x")
    eq = x**2 - pool.integer(1)
    chains = alkahest.triangularize([eq], [x])
    assert len(chains) == 2
    for ch in chains:
        assert len(ch.polys()) == 1


def test_groebner_basis_compute_still_importable():
    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    from alkahest import GroebnerBasis

    _ = GroebnerBasis.compute([x - y, x**2 - pool.integer(1)], [x, y])
