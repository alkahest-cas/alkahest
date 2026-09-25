"""Linear algebra over GF(q) — ``alkahest.experimental.FiniteField`` / ``GfMatrix``.

The Rust side is covered by ``alkahest-core/src/ffield/tests.rs``; what is
checked here is the *boundary*: that elements cross it in the documented form
(an ``int`` over GF(p), a ``list[int]`` of ``k`` coordinates over GF(p**k)),
that every refusal arrives as a ``FiniteFieldError`` carrying its stable
``E-GFQ-NNN`` code, and that the code path a linear-code user actually walks —
parity check matrix, nullspace, generator matrix — produces the right answer
end to end.
"""

from __future__ import annotations

import pytest
from alkahest.experimental import FiniteField, FiniteFieldError, GfMatrix

# The [7,4] Hamming code's parity-check matrix, columns in binary-count order.
HAMMING_74 = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]


@pytest.fixture
def gf2():
    return FiniteField(2)


@pytest.fixture
def gf8():
    """GF(2**3) from x**3 + x + 1, pinned so the tests state their own basis."""
    return FiniteField.with_defining_polynomial(2, [1, 1, 0, 1])


def code_of(exc_info):
    return exc_info.value.code


# ---------------------------------------------------------------------------
# Field construction
# ---------------------------------------------------------------------------


def test_prime_field_attributes(gf2):
    assert gf2.characteristic == 2
    assert gf2.degree == 1
    assert gf2.order == 2
    assert gf2.is_prime_field is True
    assert gf2.defining_polynomial() == []
    assert gf2.generator() is None
    assert str(gf2) == "GF(2)"
    assert gf2 == FiniteField(2)
    assert gf2 != FiniteField(3)
    assert gf2 != "not a field"
    assert hash(gf2) == hash(FiniteField(2))


def test_extension_field_attributes(gf8):
    assert gf8.characteristic == 2
    assert gf8.degree == 3
    assert gf8.order == 8
    assert gf8.is_prime_field is False
    assert gf8.defining_polynomial() == [1, 1, 0, 1]
    assert gf8.generator() == [0, 1, 0]
    assert gf8.render([1, 1, 0]) == "a + 1"
    assert str(gf8) == "GF(2^3)"


def test_degree_one_extension_is_the_prime_field():
    assert FiniteField(7, 1) == FiniteField(7)
    assert FiniteField(7, 1).is_prime_field


def test_two_presentations_of_gf8_are_not_interchangeable(gf8):
    other = FiniteField.with_defining_polynomial(2, [1, 0, 1, 1])
    assert gf8 != other
    with pytest.raises(FiniteFieldError) as e:
        GfMatrix(gf8, [[1]]).add(GfMatrix(other, [[1]]))
    assert code_of(e) == "E-GFQ-006"


@pytest.mark.parametrize("p", [0, 1, 4, 6, 9, 100])
def test_a_composite_characteristic_is_refused(p):
    with pytest.raises(FiniteFieldError) as e:
        FiniteField(p)
    assert code_of(e) == "E-GFQ-001"
    assert e.value.remediation


def test_a_characteristic_past_a_machine_word_is_refused_by_code():
    # A Python int has no width. The refusal must name the value and carry
    # E-GFQ-002, not surface as a bare OverflowError from the conversion.
    huge = 10**120 + 7
    with pytest.raises(FiniteFieldError) as e:
        FiniteField(huge)
    assert code_of(e) == "E-GFQ-002"
    assert str(huge) in str(e.value)

    with pytest.raises(FiniteFieldError) as e:
        FiniteField(-7)
    assert code_of(e) == "E-GFQ-001", "a negative modulus is not 'too large'"


def test_an_out_of_range_degree_is_refused():
    with pytest.raises(FiniteFieldError) as e:
        FiniteField(2, 0)
    assert code_of(e) == "E-GFQ-003"
    with pytest.raises(FiniteFieldError) as e:
        FiniteField(2, 65)
    assert code_of(e) == "E-GFQ-003"


def test_a_reducible_defining_polynomial_is_refused():
    # x**2 + 1 = (x + 1)**2 over GF(2).
    with pytest.raises(FiniteFieldError) as e:
        FiniteField.with_defining_polynomial(2, [1, 0, 1])
    assert code_of(e) == "E-GFQ-004"


def test_a_non_monic_defining_polynomial_is_refused():
    with pytest.raises(FiniteFieldError) as e:
        FiniteField.with_defining_polynomial(3, [1, 1, 2])
    assert code_of(e) == "E-GFQ-004"
    # Its monic scaling is accepted.
    assert FiniteField.with_defining_polynomial(3, [2, 2, 1]).degree == 2


# ---------------------------------------------------------------------------
# Elements across the boundary
# ---------------------------------------------------------------------------


def test_prime_field_entries_are_ints(gf2):
    m = GfMatrix(gf2, [[1, 0], [1, 1]])
    assert m.entry(0, 0) == 1
    assert isinstance(m.entry(0, 0), int)
    assert m.to_list() == [[1, 0], [1, 1]]


def test_extension_entries_are_coordinate_lists(gf8):
    a = gf8.generator()
    m = GfMatrix(gf8, [[a, 1], [0, [1, 1, 0]]])
    assert m.entry(0, 0) == [0, 1, 0]
    assert m.entry(0, 1) == [1, 0, 0], "an int is read as the prime subfield"
    assert m.entry(1, 0) == [0, 0, 0]
    assert m.entry(1, 1) == [1, 1, 0]
    assert m.to_list() == [[[0, 1, 0], [1, 0, 0]], [[0, 0, 0], [1, 1, 0]]]


def test_entries_are_reduced_mod_p():
    f = FiniteField(5)
    m = GfMatrix(f, [[7, -1], [10, 12]])
    assert m.to_list() == [[2, 4], [0, 2]]


def test_an_over_long_element_is_refused(gf8):
    with pytest.raises(FiniteFieldError) as e:
        GfMatrix(gf8, [[[1, 0, 0, 1]]])
    assert code_of(e) == "E-GFQ-005"


def test_a_non_numeric_entry_is_a_type_error(gf2):
    with pytest.raises(TypeError):
        GfMatrix(gf2, [["x"]])


def test_ragged_rows_are_refused(gf2):
    with pytest.raises(FiniteFieldError) as e:
        GfMatrix(gf2, [[1, 0], [1]])
    assert code_of(e) == "E-GFQ-007"


# ---------------------------------------------------------------------------
# The operation this module exists for
# ---------------------------------------------------------------------------


def test_hamming_74_nullspace_is_a_generator_matrix(gf2):
    h = GfMatrix(gf2, HAMMING_74)
    assert h.shape() == (3, 7)
    assert h.rank() == 3

    g_t = h.nullspace()
    assert g_t.shape() == (7, 4), "a [7,4] code"
    assert h.rank() + g_t.ncols == h.ncols, "rank + nullity == ncols"
    assert (h @ g_t).is_zero(), "H @ G.T == 0"
    assert g_t.rank() == 4, "the kernel basis is independent"

    g = g_t.transpose()
    assert (g @ h.transpose()).is_zero()


def test_syndrome_decoding_round_trip(gf2):
    h = GfMatrix(gf2, HAMMING_74)
    for bit in range(7):
        err = GfMatrix.from_flat(gf2, 7, 1, [1 if i == bit else 0 for i in range(7)])
        syndrome = h @ err
        # The syndrome of a single-bit error is that column of H, and it is
        # non-zero and distinct for every bit — that is what makes the code
        # single-error-correcting.
        assert not syndrome.is_zero()
        recovered = h.solve(syndrome)
        assert (h @ recovered) == syndrome


def test_rref_transform_actually_transforms(gf2):
    h = GfMatrix(gf2, HAMMING_74)
    r = h.rref()
    assert r.rank == 3
    assert r.pivots() == sorted(r.pivots())
    assert len(r.pivots()) == r.rank
    assert (r.transform @ h) == r.matrix
    assert r.transform.inverse() is not None  # invertible, else it would raise
    assert r.matrix.shape() == (3, 7)
    assert r.transform.shape() == (3, 3)


def test_rank_nullity_over_a_wide_rank_deficient_matrix():
    f = FiniteField(3)
    a = GfMatrix(f, [[1, 2, 0, 1, 1], [2, 1, 0, 2, 2]])
    assert a.rank() == 1
    n = a.nullspace()
    assert a.rank() + n.ncols == a.ncols == 5
    assert (a @ n).is_zero()


# ---------------------------------------------------------------------------
# Arithmetic and square-matrix operations
# ---------------------------------------------------------------------------


def test_arithmetic_matches_hand_computation():
    f = FiniteField(7)
    a = GfMatrix(f, [[1, 2, 3], [4, 5, 6]])
    b = GfMatrix(f, [[6, 5, 4], [3, 2, 1]])
    assert (a + b).is_zero()
    assert (a - b).to_list() == [[2, 4, 6], [1, 3, 5]]
    assert (-a).to_list() == [[6, 5, 4], [3, 2, 1]]
    assert a.scalar_mul(3).to_list() == [[3, 6, 2], [5, 1, 4]]
    assert (a @ b.transpose()).to_list() == [[0, 3], [3, 0]]
    assert a.transpose().to_list() == [[1, 4], [2, 5], [3, 6]]


def test_determinant_inverse_and_charpoly():
    f = FiniteField(7)
    a = GfMatrix(f, [[1, 2], [3, 4]])
    assert a.determinant() == 5  # 4 - 6 = -2 = 5 mod 7
    assert a.charpoly() == [5, 2, 1]  # x**2 - 5x - 2 = x**2 + 2x + 5
    inv = a.inverse()
    assert (a @ inv) == GfMatrix.identity(f, 2)
    assert (inv @ a) == GfMatrix.identity(f, 2)


def test_gf4_determinant_is_the_generator():
    f = FiniteField.with_defining_polynomial(2, [1, 1, 1])  # x**2 + x + 1
    a = f.generator()
    m = GfMatrix(f, [[a, 1], [1, a]])
    # det = a*a - 1 = a**2 + 1 = (a + 1) + 1 = a
    assert m.determinant() == a
    assert m.charpoly() == [a, [0, 0], [1, 0]]
    assert (m @ m.inverse()) == GfMatrix.identity(f, 2)


def test_singular_and_non_square_refusals():
    f = FiniteField(7)
    singular = GfMatrix(f, [[1, 2], [2, 4]])
    assert singular.determinant() == 0
    with pytest.raises(FiniteFieldError) as e:
        singular.inverse()
    assert code_of(e) == "E-GFQ-009"

    rect = GfMatrix(f, [[1, 2, 3], [4, 5, 6]])
    for op in (rect.determinant, rect.charpoly, rect.inverse):
        with pytest.raises(FiniteFieldError) as e:
            op()
        assert code_of(e) == "E-GFQ-008"


def test_an_inconsistent_system_is_refused():
    f = FiniteField(5)
    a = GfMatrix(f, [[1, 2], [2, 4], [3, 1]])
    b = GfMatrix.from_flat(f, 3, 1, [1, 0, 0])
    with pytest.raises(FiniteFieldError) as e:
        a.solve(b)
    assert code_of(e) == "E-GFQ-010"
    assert "column space" in str(e.value)


def test_shape_and_index_refusals(gf2):
    a = GfMatrix(gf2, [[1, 0, 1], [0, 1, 1]])
    with pytest.raises(FiniteFieldError) as e:
        a.mul(a)
    assert code_of(e) == "E-GFQ-007"
    with pytest.raises(FiniteFieldError) as e:
        a.entry(5, 0)
    assert code_of(e) == "E-GFQ-011"
    with pytest.raises(FiniteFieldError) as e:
        GfMatrix.zeros(gf2, 1 << 40, 1 << 40)
    assert code_of(e) == "E-GFQ-012"


def test_every_refusal_is_an_alkahest_error(gf2):
    import alkahest

    with pytest.raises(alkahest.AlkahestError):
        FiniteField(4)


def test_set_entry_and_copy(gf2):
    import copy

    a = GfMatrix(gf2, [[1, 0], [0, 1]])
    b = copy.copy(a)
    b.set_entry(0, 1, 1)
    assert a.to_list() == [[1, 0], [0, 1]]
    assert b.to_list() == [[1, 1], [0, 1]]
    assert a != b
    assert a != "not a matrix"


def test_a_word_sized_prime_near_the_top_of_the_range():
    p = 2**64 - 2**32 + 1  # Goldilocks
    f = FiniteField(p)
    assert f.characteristic == p
    a = GfMatrix(f, [[1, 2], [3, 4]])
    assert a.determinant() == p - 2
    assert (a @ a.inverse()) == GfMatrix.identity(f, 2)
