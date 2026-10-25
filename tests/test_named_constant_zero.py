"""Named constants are numbers, not indeterminates, in the zero test.

Launch report 2026-10-24, row 3: with ``phi = parse("GoldenRatio", pool)``,
``is_zero(phi**2 - phi - 1)`` was ``False`` and ``equals`` reported
``proved_nonzero`` by ``rational_normal_form``: the normal form treated ``φ`` as
an independent symbol, for which ``t² − t − 1`` is a non-zero polynomial. ``φ``
is algebraic (``(1 + √5)/2``) and the identity is now proved.

The same "independent symbol" argument was applied to ``EulerGamma``,
``Catalan``, ``Glaisher`` and the Stieltjes constants, for which no such
theorem is known (whether ``γ`` is irrational is open). A non-zero verdict for
them must now come from a rigorous Arb ball, never from the normal form.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from alkahest import ExprPool, parse
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

mpmath = pytest.importorskip("mpmath")


@pytest.fixture
def pool():
    return ExprPool()


def test_report_repro_golden_ratio_identity(pool):
    phi = parse("GoldenRatio", pool)
    assert ak.is_zero(phi**2 - phi - 1) is True
    t = ak.equals(phi**2, phi + 1)
    assert t.status == "proved_zero"
    assert ak.is_zero(parse("GoldenRatio^2-GoldenRatio-1", pool)) is True


def test_golden_ratio_against_radicals(pool):
    phi = parse("GoldenRatio", pool)
    assert ak.is_zero(phi - (1 + ak.sqrt(pool.integer(5))) / 2) is True
    assert ak.is_zero(1 / phi - (phi - 1)) is True
    assert ak.is_zero(phi**2 - phi) is False


def test_simplify_reduces_golden_ratio_identity(pool):
    phi = parse("GoldenRatio", pool)
    assert ak.simplify(phi**2 - phi - 1).value == pool.integer(0)


def _fib(k):
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(-6, 6), min_size=1, max_size=7), st.integers(-3, 3))
def test_random_golden_ratio_identities_are_never_proved_nonzero(coeffs, tweak):
    """Σ cₖ φᵏ − (b φ + a), with φᵏ = Fₖ φ + Fₖ₋₁, is identically zero."""
    pool = ExprPool()
    phi = parse("GoldenRatio", pool)
    lhs = pool.integer(0)
    a = b = 0
    for k, c in enumerate(coeffs):
        lhs = lhs + pool.integer(c) * phi**k
        b += c * _fib(k)
        a += c * (_fib(k - 1) if k >= 1 else 1)
    expr = lhs - (pool.integer(b) * phi + pool.integer(a)) + pool.integer(tweak)
    t = ak.equals(expr, pool.integer(0))
    if tweak == 0:
        assert t.status != "proved_nonzero", (coeffs, t)
        assert t.status == "proved_zero", (coeffs, t)
    else:
        assert t.status != "proved_zero", (coeffs, t)


@pytest.mark.parametrize("name", ["EulerGamma", "Catalan", "Glaisher", "StieltjesGamma1"])
def test_other_constants_are_not_independent_indeterminates(pool, name):
    c = pool.symbol(name)
    # Non-zero, but the proof has to be numeric.
    t = ak.equals(c**2 - c - 1, pool.integer(0))
    assert t.status == "proved_nonzero"
    assert t.method != "rational_normal_form", t
    # A polynomial relation between two named constants is not refuted by
    # treating them as independent symbols either.
    g = parse("EulerGamma", pool)
    t = ak.equals(c * g, g * g + c)
    assert t.method != "rational_normal_form", t


def test_negative_stieltjes_constant_is_not_positive(pool):
    # γ₁ = −0.0728…: log(γ₁²) = 2·log|γ₁| but 2·log(γ₁) = 2·log|γ₁| + 2πi.
    g1 = pool.symbol("StieltjesGamma1")
    assert mpmath.stieltjes(1) < 0
    t = ak.equals(ak.log(g1 * g1), 2 * ak.log(g1))
    assert t.status != "proved_zero", t


def test_pi_identities_still_decided(pool):
    pi = parse("pi", pool)
    x = pool.symbol("x")
    assert ak.is_zero((pi + x) ** 2 - pi**2 - 2 * pi * x - x**2) is True
    assert ak.is_zero(pi**2 - pi - 1) is False
