"""Textbook gate — polynomial algebra.

First-course polynomial operations: factoring over Z (univariate and
multivariate), GCD, real root isolation, and total degree. See
``tests/textbook_gate/README.md`` for the verification philosophy — this
file never compares against alkahest's printed normal form. Instead:

- ``UniPoly``: coefficient *lists* (ascending, exact integers) are compared
  directly against a hand-computed expected polynomial — that's real data,
  not a normal-form artifact (mirrors ``tests/test_oracle.py``).
- ``UniPoly.factor_z()``: factor degrees/multiplicities are checked against
  the expected factorization, and the product of the returned factors is
  reconstructed (via plain Python convolution of the coefficient lists) and
  compared back to the original polynomial's coefficients.
- ``MultiPoly``: has no coefficient accessor, so equality is checked via
  arithmetic: build a *reference* ``MultiPoly`` from a known expression,
  subtract, and assert ``.is_zero()`` — this only depends on ``MultiPoly``'s
  own (canonical, content-addressed) arithmetic, not on any printed form.
- ``real_roots``: each analytic root is checked against exactly one returned
  bracketing interval (``lo``/``hi``), with a small epsilon tolerance.

As of the 2026-07-20 usage eval (report7-20.md), no bugs were reported in
this subsystem specifically — every case below is expected to pass unless
noted otherwise.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


@pytest.fixture
def y(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("y")


@pytest.fixture
def z(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("z")


def _unipoly(pool: ak.ExprPool, var: ak.Expr, coeffs: list[int]) -> ak.UniPoly:
    """Build a UniPoly from plain Python ints (ascending order)."""
    return ak.UniPoly.from_coefficients([pool.integer(c) for c in coeffs], var)


def _convolve(a: list[int], b: list[int]) -> list[int]:
    """Plain Python polynomial multiplication of two ascending coefficient lists."""
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def _strip_trailing_zeros(coeffs: list[int]) -> list[int]:
    coeffs = list(coeffs)
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs


def _assert_factorization_reconstructs(poly: ak.UniPoly, expected_coeffs: list[int]) -> None:
    """Multiply out ``poly.factor_z()``'s factors (with multiplicity and unit)
    and check the convolution matches ``expected_coeffs`` exactly."""
    fz = poly.factor_z()
    product = [int(fz.unit)]
    for factor, multiplicity in fz.factor_list():
        for _ in range(multiplicity):
            product = _convolve(product, factor.coefficients())
    assert _strip_trailing_zeros(product) == _strip_trailing_zeros(expected_coeffs)


# --- UniPoly.factor_z() --------------------------------------------------


def test_unipoly_factor_x4_minus_1(pool, x):
    """x^4 - 1 = (x+1)(x-1)(x^2+1) -- one linear pair plus an irreducible quadratic."""
    p = _unipoly(pool, x, [-1, 0, 0, 0, 1])
    fz = p.factor_z()
    factors = fz.factor_list()
    assert fz.unit == "1"
    assert len(factors) == 3
    degrees = sorted(f.degree() for f, _ in factors)
    assert degrees == [1, 1, 2]
    assert all(m == 1 for _, m in factors)
    _assert_factorization_reconstructs(p, [-1, 0, 0, 0, 1])


def test_unipoly_factor_x3_minus_x_three_linear_factors(pool, x):
    """x^3 - x = x(x-1)(x+1) -- three distinct real linear factors."""
    p = _unipoly(pool, x, [0, -1, 0, 1])
    fz = p.factor_z()
    factors = fz.factor_list()
    assert fz.unit == "1"
    assert len(factors) == 3
    assert all(f.degree() == 1 and m == 1 for f, m in factors)
    _assert_factorization_reconstructs(p, [0, -1, 0, 1])


def test_unipoly_factor_x2_plus_1_irreducible_over_z(pool, x):
    """x^2 + 1 has no real roots and is irreducible over Z -- factor_z should
    return it unchanged as a single degree-2 factor."""
    p = _unipoly(pool, x, [1, 0, 1])
    fz = p.factor_z()
    factors = fz.factor_list()
    assert fz.unit == "1"
    assert len(factors) == 1
    factor, multiplicity = factors[0]
    assert factor.degree() == 2
    assert multiplicity == 1
    assert factor.coefficients() == [1, 0, 1]


def test_unipoly_factor_cubic_rational_root_plus_irreducible_quadratic(pool, x):
    """(x-2)(x^2+1) = x^3 - 2x^2 + x - 2 -- one rational root, one irreducible quadratic."""
    p = _unipoly(pool, x, [-2, 1, -2, 1])
    fz = p.factor_z()
    factors = fz.factor_list()
    assert fz.unit == "1"
    assert len(factors) == 2
    degrees = sorted(f.degree() for f, _ in factors)
    assert degrees == [1, 2]
    assert all(m == 1 for _, m in factors)
    _assert_factorization_reconstructs(p, [-2, 1, -2, 1])


def test_unipoly_factor_repeated_linear_factor(pool, x):
    """(x-1)^2 (x+3) = x^3 + x^2 - 5x + 3 -- a linear factor with multiplicity 2."""
    p = _unipoly(pool, x, [3, -5, 1, 1])
    fz = p.factor_z()
    factors = fz.factor_list()
    assert fz.unit == "1"
    assert len(factors) == 2
    assert all(f.degree() == 1 for f, _ in factors)
    multiplicities = sorted(m for _, m in factors)
    assert multiplicities == [1, 2]
    _assert_factorization_reconstructs(p, [3, -5, 1, 1])


def test_unipoly_factor_quartic_four_linear_factors(pool, x):
    """x^4 - 5x^2 + 4 = (x-1)(x+1)(x-2)(x+2) -- fully factors into four
    distinct linear pieces."""
    p = _unipoly(pool, x, [4, 0, -5, 0, 1])
    fz = p.factor_z()
    factors = fz.factor_list()
    assert fz.unit == "1"
    assert len(factors) == 4
    assert all(f.degree() == 1 and m == 1 for f, m in factors)
    _assert_factorization_reconstructs(p, [4, 0, -5, 0, 1])


def test_unipoly_degree_and_coefficients_roundtrip(pool, x):
    """Sanity check: from_coefficients/.degree()/.coefficients() round-trip
    exactly for a simple polynomial (2x^3 - 7 = 2x^3 + 0x^2 + 0x - 7)."""
    p = _unipoly(pool, x, [-7, 0, 0, 2])
    assert p.degree() == 3
    assert p.coefficients() == [-7, 0, 0, 2]


# --- UniPoly.gcd() --------------------------------------------------------


def test_unipoly_gcd_shared_linear_factor(pool, x):
    """gcd(x^2-1, x^2-3x+2) = x-1 -- both share the root x=1."""
    a = _unipoly(pool, x, [-1, 0, 1])  # x^2 - 1 = (x-1)(x+1)
    b = _unipoly(pool, x, [2, -3, 1])  # x^2 - 3x + 2 = (x-1)(x-2)
    g = a.gcd(b)
    assert g.degree() == 1
    # coefficients are only defined up to a unit; normalize so the constant
    # term's sign matches (x - 1) has coefficients [-1, 1]
    coeffs = g.coefficients()
    if coeffs[-1] < 0:
        coeffs = [-c for c in coeffs]
    assert coeffs == [-1, 1]


def test_unipoly_gcd_shared_quadratic_factor(pool, x):
    """gcd((x^2+1)(x-5), (x^2+1)(x+7)) = x^2+1."""
    common = [1, 0, 1]  # x^2 + 1
    a = _unipoly(pool, x, _convolve(common, [-5, 1]))
    b = _unipoly(pool, x, _convolve(common, [7, 1]))
    g = a.gcd(b)
    assert g.degree() == 2
    coeffs = g.coefficients()
    if coeffs[-1] < 0:
        coeffs = [-c for c in coeffs]
    assert coeffs == [1, 0, 1]


def test_unipoly_gcd_coprime_polys_is_constant(pool, x):
    """gcd(x-1, x-2) share no roots -- the gcd is a nonzero constant (degree 0)."""
    a = _unipoly(pool, x, [-1, 1])
    b = _unipoly(pool, x, [-2, 1])
    g = a.gcd(b)
    assert g.degree() == 0


# --- ak.real_roots ---------------------------------------------------------


def _assert_root_bracketed(roots, expected_root: float, eps: float = 1e-6) -> None:
    matches = [r for r in roots if r.lo - eps <= expected_root <= r.hi + eps]
    assert len(matches) == 1, (
        f"expected exactly one interval bracketing {expected_root}, "
        f"got {[(r.lo, r.hi) for r in matches]} among {[(r.lo, r.hi) for r in roots]}"
    )


def test_real_roots_x2_minus_4(pool, x):
    """x^2 - 4 has roots -2 and 2."""
    roots = ak.real_roots(x**2 - pool.integer(4), x)
    assert len(roots) == 2
    _assert_root_bracketed(roots, -2.0)
    _assert_root_bracketed(roots, 2.0)


def test_real_roots_x3_minus_2x(pool, x):
    """x^3 - 2x = x(x^2-2) has roots -sqrt(2), 0, sqrt(2)."""
    roots = ak.real_roots(x**3 - pool.integer(2) * x, x)
    assert len(roots) == 3
    _assert_root_bracketed(roots, -math.sqrt(2))
    _assert_root_bracketed(roots, 0.0)
    _assert_root_bracketed(roots, math.sqrt(2))


def test_real_roots_repeated_root_counted_once(pool, x):
    """(x-1)^2 has a repeated root at x=1, which should appear as exactly one
    bracketing interval for the squarefree part, not two."""
    roots = ak.real_roots((x - pool.integer(1)) ** 2, x)
    assert len(roots) == 1
    _assert_root_bracketed(roots, 1.0)


def test_real_roots_cubic_three_distinct_roots(pool, x):
    """x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3) has three distinct real roots."""
    poly = x**3 - pool.integer(6) * x**2 + pool.integer(11) * x - pool.integer(6)
    roots = ak.real_roots(poly, x)
    assert len(roots) == 3
    _assert_root_bracketed(roots, 1.0)
    _assert_root_bracketed(roots, 2.0)
    _assert_root_bracketed(roots, 3.0)


def test_real_roots_no_real_roots(pool, x):
    """x^2 + 1 has no real roots -- real_roots should return an empty list."""
    roots = ak.real_roots(x**2 + pool.integer(1), x)
    assert len(roots) == 0


# --- MultiPoly.total_degree() ---------------------------------------------


def test_multipoly_total_degree_binomial_cube(pool, x, y):
    """(x+y)^3 has total degree 3."""
    mp = ak.MultiPoly.from_symbolic((x + y) ** 3, [x, y])
    assert mp.total_degree() == 3


def test_multipoly_total_degree_trinomial_square(pool, x, y, z):
    """(x+y+z)^2 has total degree 2."""
    mp = ak.MultiPoly.from_symbolic((x + y + z) ** 2, [x, y, z])
    assert mp.total_degree() == 2


def test_multipoly_total_degree_mixed(pool, x, y):
    """x^2*y + y^3 -- both monomials have total degree 3."""
    mp = ak.MultiPoly.from_symbolic(x**2 * y + y**3, [x, y])
    assert mp.total_degree() == 3


def test_multipoly_total_degree_linear(pool, x, y):
    """x + 2y + 1 has total degree 1."""
    mp = ak.MultiPoly.from_symbolic(x + pool.integer(2) * y + pool.integer(1), [x, y])
    assert mp.total_degree() == 1


# --- MultiPoly.gcd() --------------------------------------------------------


def _assert_multipoly_equal(a: ak.MultiPoly, b: ak.MultiPoly) -> None:
    """Compare two MultiPoly values via arithmetic (no coefficient accessor
    exists): subtract and check the result is the zero polynomial."""
    assert (a - b).is_zero()


def test_multipoly_gcd_shared_linear_factor(pool, x, y):
    """gcd((x+y)(x-y), (x+y)(x+2y)) = x+y."""
    a = ak.MultiPoly.from_symbolic((x + y) * (x - y), [x, y])
    b = ak.MultiPoly.from_symbolic((x + y) * (x + pool.integer(2) * y), [x, y])
    g = a.gcd(b)
    expected = ak.MultiPoly.from_symbolic(x + y, [x, y])
    assert g.total_degree() == 1
    _assert_multipoly_equal(g, expected)


def test_multipoly_gcd_shared_trivariate_factor(pool, x, y, z):
    """gcd((x+y+z)(x-y), (x+y+z)(y+z)) = x+y+z."""
    a = ak.MultiPoly.from_symbolic((x + y + z) * (x - y), [x, y, z])
    b = ak.MultiPoly.from_symbolic((x + y + z) * (y + z), [x, y, z])
    g = a.gcd(b)
    expected = ak.MultiPoly.from_symbolic(x + y + z, [x, y, z])
    assert g.total_degree() == 1
    _assert_multipoly_equal(g, expected)


def test_multipoly_gcd_coprime_is_constant(pool, x, y):
    """gcd(x, y) share no common factor -- the gcd has total degree 0."""
    a = ak.MultiPoly.from_symbolic(x, [x, y])
    b = ak.MultiPoly.from_symbolic(y, [x, y])
    g = a.gcd(b)
    assert g.total_degree() == 0


# --- MultiPoly.factor_z() ---------------------------------------------------


def test_multipoly_factor_z_difference_of_squares(pool, x, y):
    """(x+y)(x-y) = x^2 - y^2 -- two linear factors."""
    mp = ak.MultiPoly.from_symbolic((x + y) * (x - y), [x, y])
    fz = mp.factor_z()
    factors = fz.factor_list()
    assert len(factors) == 2
    assert all(f.total_degree() == 1 and m == 1 for f, m in factors)

    # reconstruct the product and check it matches the original (up to unit)
    product = None
    for factor, multiplicity in factors:
        for _ in range(multiplicity):
            product = factor if product is None else product * factor
    assert fz.unit == "1"
    _assert_multipoly_equal(product, mp)


def test_multipoly_factor_z_repeated_factor(pool, x, y):
    """(x+y)^2 * (x-y) has one factor with multiplicity 2."""
    mp = ak.MultiPoly.from_symbolic((x + y) ** 2 * (x - y), [x, y])
    fz = mp.factor_z()
    factors = fz.factor_list()
    multiplicities = sorted(m for _, m in factors)
    assert multiplicities == [1, 2]

    product = None
    for factor, multiplicity in factors:
        for _ in range(multiplicity):
            product = factor if product is None else product * factor
    assert fz.unit == "1"
    _assert_multipoly_equal(product, mp)
