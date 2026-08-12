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


def test_triangularize_refusal_carries_e_solve_004():
    """A chain that would under-determine the system refuses, with its own code."""
    pool = ExprPool()
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    # <xy, xz> needs a splitting decomposition into [x] and [y, z]; extraction
    # alone would return a chain cutting out a larger variety than the input.
    with pytest.raises(ValueError) as ei:
        alkahest.triangularize([x * y, x * z], [x, y, z])
    assert getattr(ei.value, "code", None) == "E-SOLVE-004"


def test_genuine_non_polynomial_is_not_reattributed_to_the_refusal():
    """The out-of-band code must not leak onto an unrelated `NotPolynomial`.

    Both travel through the same enum variant, so a stale or unconditionally
    read refusal would relabel this as E-SOLVE-004.
    """
    pool = ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    with pytest.raises(ValueError) as ei:
        alkahest.triangularize([alkahest.sin(x) - y, y - pool.integer(1)], [x, y])
    assert getattr(ei.value, "code", None) == "E-SOLVE-001"


def test_triangularize_keeps_both_generators():
    """Regression: the main-variable pick was inverted and discarded generators."""
    pool = ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    chains = alkahest.triangularize([x - y - pool.integer(1), y**2 - pool.integer(2)], [x, y])
    assert len(chains) == 1
    assert len(chains[0].polys()) == 2
