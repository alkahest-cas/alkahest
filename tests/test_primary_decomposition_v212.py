"""V2-12 — primary decomposition (Gianni–Trager–Zacharias fragment)."""

import alkahest
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(alkahest, "primary_decomposition"),
    reason="native module built without groebner feature",
)


def test_primary_decomposition_xy_xz():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    z = pool.symbol("z")
    dec = alkahest.primary_decomposition([x * y, x * z], [x, y, z])
    assert len(dec) == 2
    for c in dec:
        p = c.primary()
        ap = c.associated_prime()
        assert len(p) >= 1
        assert len(ap) >= 1


def test_radical_x2_xy():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    r = alkahest.radical([x**2, x * y], [x, y])
    assert r.contains(x)
