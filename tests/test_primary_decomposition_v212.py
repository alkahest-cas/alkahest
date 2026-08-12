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


# ---------------------------------------------------------------------------
# Refusals must reach Python with their own stable code.
#
# `radical` / `primary_decomposition` report "I cannot certify this" through
# `PrimaryDecompositionError::Factorization` and record the real reason out of
# band (the enum is public and exhaustive, so it cannot grow a variant without a
# major semver break). If the binding forgets to consult `take_ideal_refusal`,
# the refusal still *happens* but arrives as an uncoded `ValueError` — honest,
# but not machine-readable, which is what an autoresearch loop branches on.
# ---------------------------------------------------------------------------


def _codes_of(exc):
    return getattr(exc, "code", None)


def test_radical_refusal_carries_e_ideal_005():
    pool = alkahest.ExprPool()
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    # The twisted cubic is prime and *is* its own radical — the old code returned
    # the input unchanged and was right here by accident. Refusing is correct;
    # the point of this test is that the refusal is coded.
    with pytest.raises(ValueError) as ei:
        alkahest.radical([y - x**2, z - x**3], [x, y, z])
    assert _codes_of(ei.value) == "E-IDEAL-005"
    assert ei.value.remediation


def test_primary_decomposition_refusal_carries_e_ideal_006():
    pool = alkahest.ExprPool()
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    with pytest.raises(ValueError) as ei:
        alkahest.primary_decomposition([y - x**2, z - x**3], [x, y, z])
    assert _codes_of(ei.value) == "E-IDEAL-006"


def test_certified_cases_still_answer_and_are_not_refused():
    """The refusal path must not swallow the cases that *are* certifiable."""
    pool = alkahest.ExprPool()
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    assert len(alkahest.primary_decomposition([x**2 - y**2], [x, y, z])) == 2
    assert len(alkahest.primary_decomposition([x * z, y * z], [x, y, z])) == 2
    assert alkahest.radical([(x - y) ** 2], [x, y, z]).contains(x - y)
