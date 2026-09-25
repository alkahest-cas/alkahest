"""Algebraic number fields and the classical arithmetic functions.

The number-field surface sits on FLINT's ``nf``/``nf_elem``, whose element
type is a C union over the degree of the field — Antic special-cases degree 1
and degree 2. Degrees 1, 2, 3 and 4 are therefore each exercised here on
purpose; a suite that only built cubic fields would leave two union arms
untouched.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from alkahest.experimental import (
    ArithmeticError,
    NumberField,
    NumberFieldError,
    bernoulli_number,
    cyclotomic_polynomial_coeffs,
    divisor_sigma,
    euler_number,
    harmonic_number,
    moebius_mu,
    partition_number,
    stirling_first,
    stirling_first_unsigned,
    stirling_second,
    sum_of_squares,
)

# ---------------------------------------------------------------------------
# Number fields
# ---------------------------------------------------------------------------


def test_degree_one_field_is_the_rationals():
    k = NumberField([-3, 1])  # x - 3
    assert k.degree == 1
    g = k.generator()
    assert g.norm() == Fraction(3)
    assert g.trace() == Fraction(3)
    assert g.coefficients() == [Fraction(3)]
    assert (g * g.inverse()).is_one()


def test_sqrt_two():
    k = NumberField([-2, 0, 1])
    assert k.degree == 2
    assert k.defining_polynomial == [-2, 0, 1]
    assert k.polynomial_discriminant == 8

    root = k.generator()
    assert root.norm() == Fraction(-2)
    assert root.trace() == Fraction(0)

    assert k.element([]).is_zero(), "no coordinates is the zero element"

    unit = k.one() + root  # the fundamental unit 1 + sqrt 2
    assert unit.norm() == Fraction(-1)
    assert unit.trace() == Fraction(2)
    assert str(unit) == "a + 1"

    # N(a + b sqrt2) = a^2 - 2 b^2.
    for a in range(-3, 4):
        for b in range(-3, 4):
            assert k.element([a, b]).norm() == Fraction(a * a - 2 * b * b)


def test_golden_ratio_field():
    k = NumberField([-1, -1, 1])
    phi = k.generator()
    assert phi.norm() == Fraction(-1)
    assert phi.trace() == Fraction(1)
    assert phi.minimal_polynomial() == [Fraction(-1), Fraction(-1), Fraction(1)]
    assert phi**2 == phi + k.one()


def test_cube_root_of_two():
    k = NumberField([-2, 0, 0, 1])
    c = k.generator()
    assert k.degree == 3
    assert c.norm() == Fraction(2)
    assert c.minimal_polynomial() == [Fraction(-2), Fraction(0), Fraction(0), Fraction(1)]
    assert c**3 == k.element([2])
    assert c**-1 == c.inverse()


def test_cyclotomic_field_of_fifth_roots():
    k = NumberField.cyclotomic(5)
    assert k.degree == 4
    assert k.cyclotomic_order == 5
    assert k.defining_polynomial == [1, 1, 1, 1, 1]

    z = k.generator()
    assert z.norm() == Fraction(1)
    assert z.trace() == Fraction(-1)
    assert z**5 == k.one()

    total = k.zero()
    for e in range(5):
        total = total + z**e
    assert total.is_zero(), "1 + z + z^2 + z^3 + z^4 must vanish"


def test_ring_lwe_sized_cyclotomic_field():
    # Q(zeta_256) = Q[x]/(x^128 + 1) — the shape lattice cryptography uses.
    k = NumberField.cyclotomic(256)
    assert k.degree == 128
    z = k.generator()
    assert z**256 == k.one()
    assert z**128 == -k.one()


def test_cyclotomic_polynomial_anchors():
    assert cyclotomic_polynomial_coeffs(1) == [-1, 1]
    assert cyclotomic_polynomial_coeffs(2) == [1, 1]
    assert cyclotomic_polynomial_coeffs(6) == [1, -1, 1]
    assert cyclotomic_polynomial_coeffs(12) == [1, 0, -1, 0, 1]
    # deg Phi_n == phi(n)
    from alkahest.number_theory import totient

    for n in range(1, 40):
        assert len(cyclotomic_polynomial_coeffs(n)) - 1 == totient(n)


def test_rational_coefficients_cross_the_boundary_exactly():
    k = NumberField(["1/3", 0, 1])  # x^2 + 1/3, cleared to 3x^2 + 1
    assert k.defining_polynomial == [1, 0, 3]
    e = k.element([Fraction(1, 2), "-2/7"])
    assert e.coefficients() == [Fraction(1, 2), Fraction(-2, 7)]


def test_big_coefficients_are_not_truncated():
    big = 10**40 + 7
    k = NumberField([-big, 0, 1])
    assert k.defining_polynomial == [-big, 0, 1]
    assert k.generator().norm() == Fraction(-big)


def test_reducible_defining_polynomial_is_refused():
    with pytest.raises(NumberFieldError) as excinfo:
        NumberField([-1, 0, 1])  # x^2 - 1 = (x-1)(x+1)
    assert excinfo.value.code == "E-NUMF-003"


def test_zero_has_no_inverse():
    k = NumberField([-2, 0, 1])
    with pytest.raises(NumberFieldError) as excinfo:
        k.zero().inverse()
    assert excinfo.value.code == "E-NUMF-005"
    with pytest.raises(NumberFieldError):
        k.one() / k.zero()


def test_elements_of_different_fields_do_not_mix():
    a = NumberField([-2, 0, 1]).generator()
    b = NumberField([-3, 0, 1]).generator()
    with pytest.raises(NumberFieldError) as excinfo:
        _ = a + b
    assert excinfo.value.code == "E-NUMF-006"


def test_too_many_coordinates_is_refused():
    k = NumberField([-2, 0, 0, 1])
    with pytest.raises(NumberFieldError) as excinfo:
        k.element([1, 0, 0, 1])
    assert excinfo.value.code == "E-NUMF-007"


def test_float_coefficients_are_refused_not_rounded():
    with pytest.raises(NumberFieldError) as excinfo:
        NumberField([0.5, 0, 1])
    assert excinfo.value.code == "E-NUMF-004"


def test_cyclotomic_order_out_of_range():
    with pytest.raises(NumberFieldError) as excinfo:
        NumberField.cyclotomic(0)
    assert excinfo.value.code == "E-NUMF-008"


def test_norm_is_multiplicative_and_trace_additive():
    k = NumberField.cyclotomic(7)
    x = k.element([1, -2, 3, 0, 1, -1])
    y = k.element([2, 1, 0, -3, 1, 2])
    assert (x * y).norm() == x.norm() * y.norm()
    assert (x + y).trace() == x.trace() + y.trace()
    assert (x * x.inverse()).is_one()


# ---------------------------------------------------------------------------
# Arithmetic functions
# ---------------------------------------------------------------------------


def test_partition_function():
    assert partition_number(0) == 1
    assert partition_number(100) == 190569292
    assert partition_number(1000) == 24061467864032622473692149727991


def test_bernoulli_convention_is_minus_one_half():
    """FLINT uses B_1 = -1/2. Pinned, because the two conventions differ in
    exactly this one value and nowhere else."""
    assert bernoulli_number(1) == Fraction(-1, 2)


def test_bernoulli_anchors():
    assert bernoulli_number(0) == Fraction(1)
    assert bernoulli_number(2) == Fraction(1, 6)
    assert bernoulli_number(4) == Fraction(-1, 30)
    assert bernoulli_number(12) == Fraction(-691, 2730)
    assert all(bernoulli_number(n) == 0 for n in range(3, 20, 2))


def test_euler_numbers():
    assert [euler_number(n) for n in range(0, 9, 2)] == [1, -1, 5, -61, 1385]
    assert all(euler_number(n) == 0 for n in range(1, 10, 2))


def test_harmonic_numbers():
    assert harmonic_number(0) == 0
    assert harmonic_number(4) == Fraction(25, 12)
    assert harmonic_number(5) == Fraction(137, 60)
    assert harmonic_number(10) - harmonic_number(9) == Fraction(1, 10)


def test_stirling_numbers():
    assert stirling_first(4, 2) == 11
    assert stirling_first_unsigned(4, 2) == 11
    assert stirling_first(3, 2) == -3
    assert stirling_first_unsigned(3, 2) == 3
    assert stirling_second(4, 2) == 7
    for n in range(1, 8):
        assert stirling_second(n, 1) == 1
        assert stirling_second(n, n) == 1
    # sum_k S(n, k) is the Bell number.
    assert [sum(stirling_second(n, k) for k in range(1, n + 1)) for n in range(1, 8)] == [
        1,
        2,
        5,
        15,
        52,
        203,
        877,
    ]


def test_moebius_and_divisor_sigma():
    assert moebius_mu(1) == 1
    assert moebius_mu(30) == -1
    assert moebius_mu(12) == 0
    assert divisor_sigma(0, 12) == 6
    assert divisor_sigma(1, 12) == 28
    assert divisor_sigma(2, 12) == 210
    assert divisor_sigma(1, 6) == 12  # 6 is perfect
    for n in range(1, 30):
        assert divisor_sigma(1, n) == sum(d for d in range(1, n + 1) if n % d == 0)


def test_sums_of_squares():
    assert sum_of_squares(2, 0) == 1
    assert sum_of_squares(2, 5) == 8
    assert sum_of_squares(4, 1) == 8
    # Jacobi's four-square theorem.
    for n in range(1, 25):
        want = 8 * sum(d for d in range(1, n + 1) if n % d == 0 and d % 4)
        assert sum_of_squares(4, n) == want


def test_work_caps_refuse_rather_than_hang():
    from alkahest.number_theory import NumberTheoryError

    with pytest.raises(NumberTheoryError) as excinfo:
        partition_number(10**9)
    assert excinfo.value.code == "E-NT-006"


def test_arithmetic_errors_are_catchable_as_number_theory_errors():
    """The arithmetic functions raise their own exception type for versioning
    reasons on the Rust side (``NumberTheoryError`` is an exhaustive enum in the
    stable surface). That must not leak into Python: one ``except
    NumberTheoryError`` has to keep catching everything this surface raises, and
    the code must still read ``E-NT-NNN``.
    """
    from alkahest.number_theory import NumberTheoryError

    assert issubclass(ArithmeticError, NumberTheoryError)

    with pytest.raises(NumberTheoryError) as excinfo:
        partition_number(10**9)
    assert isinstance(excinfo.value, ArithmeticError)
    assert excinfo.value.code == "E-NT-006"

    # A domain violation underneath an arithmetic function keeps its own code
    # rather than being re-labelled as the work cap.
    with pytest.raises(NumberTheoryError) as excinfo:
        moebius_mu(0)
    assert excinfo.value.code == "E-NT-002"
    with pytest.raises(NumberTheoryError) as excinfo:
        divisor_sigma(1, -4)
    assert excinfo.value.code == "E-NT-002"
