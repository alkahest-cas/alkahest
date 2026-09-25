"""Conjugacy classes and exact character tables through the Python binding.

The Rust side carries the exhaustive suite. These tests exist to pin the things
a binding can get wrong on its own:

* that a character value crosses the boundary as an **exact** field element with
  working arithmetic, not as a float or a string — the assertions on ``A_4``'s
  cube roots of unity and ``A_5``'s golden ratio are done *in Python*, with
  ``**`` and ``*`` on the returned objects;
* that group and centraliser orders arrive as arbitrary-precision ``int`` and
  inner products as exact ``Fraction``;
* the refusals, which are the part a caller is most likely to meet and least
  likely to have read about, each with its stable ``.code``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from alkahest.experimental import (
    CHARACTER_DEFAULT_CLASS_CAP,
    CHARACTER_DEFAULT_TABLE_CAP,
    CHARACTER_MAX_CLASS_CAP,
    CHARACTER_MAX_EXPONENT_FIELD_DEGREE,
    CharacterError,
    CharacterTable,
    ConjugacyClasses,
    NumberField,
    Permutation,
    PermutationGroup,
)


def symmetric(n: int) -> PermutationGroup:
    return PermutationGroup.symmetric(n)


def alternating(n: int) -> PermutationGroup:
    return PermutationGroup.alternating(n)


def cyclic(n: int) -> PermutationGroup:
    return PermutationGroup.cyclic(n)


def dihedral(n: int) -> PermutationGroup:
    return PermutationGroup.dihedral(n)


def quaternion_group() -> PermutationGroup:
    """Q_8 as its right regular representation on 1, -1, i, -i, j, -j, k, -k."""
    by_i = Permutation([2, 3, 1, 0, 7, 6, 4, 5])
    by_j = Permutation([4, 5, 6, 7, 1, 0, 3, 2])
    return PermutationGroup(8, [by_i, by_j])


def integer_row(table: CharacterTable, i: int) -> list[int]:
    """Row ``i`` as integers; fails loudly if any value is irrational."""
    out = []
    for value in table.character(i):
        coefficients = value.coefficients()
        assert all(c == 0 for c in coefficients[1:]), f"{value} is not rational"
        assert coefficients[0].denominator == 1
        out.append(int(coefficients[0]))
    return out


# ---------------------------------------------------------------------------
# Conjugacy classes
# ---------------------------------------------------------------------------


def test_s3_classes():
    classes = ConjugacyClasses(symmetric(3))
    assert len(classes) == 3
    assert classes.group_order == 6
    assert classes.exponent == 6
    assert classes.degree == 3
    assert classes.sizes() == [1, 3, 2]
    assert classes.element_orders() == [1, 2, 3]
    # Class 0 is always the identity class.
    assert classes[0].representative.is_identity()
    assert classes[0].size == 1
    # |C_G(g)| = |G| / |K|, as an int.
    assert [c.centraliser_order for c in classes.classes()] == [6, 2, 3]
    assert all(isinstance(c.centraliser_order, int) for c in classes.classes())


def test_class_sizes_divide_and_sum_to_the_order():
    for group in (symmetric(4), alternating(5), dihedral(6), quaternion_group()):
        classes = ConjugacyClasses(group)
        order = classes.group_order
        assert sum(classes.sizes()) == order
        assert all(order % size == 0 for size in classes.sizes())
        assert order % classes.exponent == 0


def test_class_of_and_inverse_class():
    group = symmetric(4)
    classes = ConjugacyClasses(group)
    transposition = Permutation.from_cycles(4, [[0, 1]])
    index = classes.class_of(transposition)
    assert classes[index].element_order == 2
    assert classes[index].size == 6
    assert transposition in classes.class_elements(index)
    # An involution is its own inverse, so its class is inverse-closed.
    assert classes.inverse_class(index) == index
    # inverse_class is an involution on indices.
    for i in range(len(classes)):
        assert classes.inverse_class(classes.inverse_class(i)) == i


def test_class_multiplication_coefficients_agree_with_the_matrix():
    classes = ConjugacyClasses(alternating(4))
    r = len(classes)
    for k in range(r):
        matrix = classes.multiplication_matrix(k)
        assert len(matrix) == r
        for i in range(r):
            for j in range(r):
                assert matrix[i][j] == classes.multiplication_coefficient(k, i, j)
        # Σ_j a_kij |K_j| = |K_k| |K_i|: every pair is accounted for once.
        for i in range(r):
            total = sum(matrix[i][j] * classes[j].size for j in range(r))
            assert total == classes[k].size * classes[i].size


# ---------------------------------------------------------------------------
# Rational character tables
# ---------------------------------------------------------------------------


def test_s3_character_table():
    table = CharacterTable(symmetric(3))
    assert len(table) == 3
    assert table.degrees() == [1, 1, 2]
    assert integer_row(table, 0) == [1, 1, 1]
    assert integer_row(table, 1) == [1, -1, 1]
    assert integer_row(table, 2) == [2, 0, -1]


def test_s4_character_table():
    table = CharacterTable(symmetric(4))
    assert len(table) == 5
    assert table.degrees() == [1, 1, 2, 3, 3]
    assert table.classes().sizes() == [1, 3, 6, 8, 6]
    assert table.classes().element_orders() == [1, 2, 2, 3, 4]
    assert integer_row(table, 0) == [1, 1, 1, 1, 1]
    assert integer_row(table, 1) == [1, 1, -1, 1, -1]
    assert integer_row(table, 2) == [2, 2, 0, -1, 0]
    assert integer_row(table, 3) == [3, -1, -1, 0, 1]
    assert integer_row(table, 4) == [3, -1, 1, 0, -1]
    # Σ χ(1)² = |G|, and the count matches the classes.
    assert sum(d * d for d in table.degrees()) == 24
    assert len(table) == len(table.classes())


def test_the_number_of_characters_equals_the_number_of_classes():
    for group in (symmetric(5), alternating(5), dihedral(6), cyclic(7)):
        table = CharacterTable(group)
        assert len(table) == len(table.classes())
        order = table.classes().group_order
        assert sum(d * d for d in table.degrees()) == order
        assert all(order % d == 0 for d in table.degrees())


# ---------------------------------------------------------------------------
# The irrationalities — checked with arithmetic on the returned objects
# ---------------------------------------------------------------------------


def test_a4_values_are_exact_cube_roots_of_unity():
    table = CharacterTable(alternating(4))
    assert table.degrees() == [1, 1, 1, 3]
    assert table.exponent == 6
    field = table.field()
    assert isinstance(field, NumberField)
    assert field.degree == 2
    one = field.one()

    for i in (1, 2):
        for c in (2, 3):
            value = table.value(i, c)
            # A cube root of unity that is not 1 — and the check is done here,
            # in Python, on the object that crossed the boundary.
            assert value**3 == one
            assert value != one
            # Genuinely irrational: the zeta_6 coordinate is non-zero. This is
            # the assertion a rationals-only implementation cannot pass.
            assert value.coefficients()[1] != 0
    # And the two rows are complex conjugates: chi_2(g) == chi_1(g**-1).
    classes = table.classes()
    for c in range(len(classes)):
        assert table.value(2, c) == table.value(1, classes.inverse_class(c))


def test_a5_degree_three_values_are_the_golden_ratio():
    table = CharacterTable(alternating(5))
    assert table.degrees() == [1, 3, 3, 4, 5]
    assert table.exponent == 30
    assert table.field().degree == 8
    one = table.field().one()
    # The two values on each order-5 class are the roots of x² − x − 1.
    for c in (3, 4):
        u = table.value(1, c)
        v = table.value(2, c)
        assert u + v == one
        assert u * v == -one
        assert u * u == u + one
        assert v * v == v + one
        assert any(coefficient != 0 for coefficient in u.coefficients()[1:])
    # Still rational on the classes of order 1, 2 and 3.
    assert integer_row(table, 3) == [4, 0, 1, -1, -1]
    assert integer_row(table, 4) == [5, 1, -1, 0, 0]


def test_cyclic_group_values_are_roots_of_unity():
    for n in (5, 6, 8):
        table = CharacterTable(cyclic(n))
        assert len(table) == n
        assert table.degrees() == [1] * n
        assert table.exponent == n
        assert table.classes().sizes() == [1] * n
        one = table.field().one()
        for i in range(n):
            for c in range(n):
                assert table.value(i, c) ** n == one
        # Some value is a *primitive* n-th root: the group has a faithful
        # character.
        assert any(
            all(table.value(i, c) ** d != one for d in range(1, n) if n % d == 0)
            for i in range(n)
            for c in range(n)
        )


def test_d4_and_q8_share_a_table_but_not_their_class_data():
    d4 = CharacterTable(dihedral(4))
    q8 = CharacterTable(quaternion_group())
    assert d4.degrees() == [1, 1, 1, 1, 2]
    assert q8.degrees() == [1, 1, 1, 1, 2]
    assert d4.classes().sizes() == [1, 1, 2, 2, 2]
    assert q8.classes().sizes() == [1, 1, 2, 2, 2]

    # D_4 and Q_8 are the textbook pair of non-isomorphic groups with the
    # **same** character table, so equality is the correct assertion here.
    def fingerprint(table):
        rows = []
        for i in range(len(table)):
            rows.append(
                sorted(tuple(value.coefficients()) for value in table.character(i))
            )
        return sorted(rows)

    assert fingerprint(d4) == fingerprint(q8)

    # What tells them apart is the class data, not the table: Q_8 has six
    # elements of order 4 and D_4 has two.
    assert d4.classes().element_orders() == [1, 2, 2, 2, 4]
    assert q8.classes().element_orders() == [1, 2, 4, 4, 4]

    def of_order_four(table):
        return sum(c.size for c in table.classes().classes() if c.element_order == 4)

    assert of_order_four(d4) == 2
    assert of_order_four(q8) == 6


# ---------------------------------------------------------------------------
# Orthogonality
# ---------------------------------------------------------------------------


def test_orthogonality_relations():
    for group in (symmetric(4), alternating(4), alternating(5), quaternion_group()):
        table = CharacterTable(group)
        # `verify` re-runs the checks the constructor already made; calling it
        # is how a test tells "they pass" from "they were skipped".
        table.verify()
        r = len(table)
        for i in range(r):
            for j in range(r):
                product = table.inner_product(i, j)
                assert isinstance(product, Fraction)
                assert product == Fraction(1 if i == j else 0)
        # Column orthogonality, spelled out.
        classes = table.classes()
        field = table.field()
        for c in range(r):
            for d in range(r):
                inverse = classes.inverse_class(d)
                total = field.zero()
                for i in range(r):
                    total = total + table.value(i, c) * table.value(i, inverse)
                expected = classes[c].centraliser_order if c == d else 0
                assert total == field.element([expected])


def test_trivial_character_is_row_zero_and_str_renders_a_grid():
    table = CharacterTable(symmetric(3))
    one = table.field().one()
    assert all(value == one for value in table.character(0))
    text = str(table)
    lines = text.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("Q(zeta_6)")
    assert lines[1].startswith("chi_0")
    assert "CharacterTable(" in repr(table)
    assert "ConjugacyClasses(" in repr(table.classes())
    assert "ConjugacyClass(" in repr(table.classes()[0])


def test_from_classes_reuses_the_partition():
    classes = ConjugacyClasses(symmetric(4))
    table = CharacterTable.from_classes(classes)
    assert table.degrees() == [1, 1, 2, 3, 3]
    assert table.classes().sizes() == classes.sizes()
    # The whole table is available as a list of rows.
    rows = table.table()
    assert len(rows) == 5
    assert all(len(row) == 5 for row in rows)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_group_past_the_cap_is_refused_and_the_order_is_still_exact():
    # |S_7| = 5040, just past the default table cap of 5000.
    big = symmetric(7)
    with pytest.raises(CharacterError) as info:
        CharacterTable(big)
    assert info.value.code == "E-CHAR-001"
    assert "5040" in str(info.value)
    assert info.value.remediation
    # The order is not refused — only the class partition is.
    assert big.order() == 5040
    # Raising the cap gets past it. S_7 has p(7) = 15 classes.
    table = CharacterTable(big, cap=10_000)
    assert sum(d * d for d in table.degrees()) == 5040
    assert len(table) == len(table.classes()) == 15


def test_a_cap_past_the_hard_ceiling_is_refused():
    with pytest.raises(CharacterError) as info:
        ConjugacyClasses(symmetric(3), cap=CHARACTER_MAX_CLASS_CAP + 1)
    assert info.value.code == "E-CHAR-002"


def test_a_large_exponent_is_refused_by_field_degree():
    # |G| = 9·11·13 = 1287 on 33 points: inside every order and degree cap, but
    # phi(1287) = 720 is past the field-degree ceiling.
    generator = Permutation.from_cycles(
        33, [list(range(0, 9)), list(range(9, 20)), list(range(20, 33))]
    )
    group = PermutationGroup(33, [generator])
    assert group.order() == 1287
    with pytest.raises(CharacterError) as info:
        CharacterTable(group)
    assert info.value.code == "E-CHAR-003"
    assert "720" in str(info.value)
    # The classes themselves are fine; it is the table that is refused.
    assert len(ConjugacyClasses(group)) == 1287


def test_an_element_outside_the_group_is_refused():
    classes = ConjugacyClasses(alternating(4))
    odd = Permutation.from_cycles(4, [[0, 1]])
    with pytest.raises(CharacterError) as info:
        classes.class_of(odd)
    assert info.value.code == "E-CHAR-005"
    # A degree mismatch keeps the permutation-group layer's own code.
    with pytest.raises(CharacterError) as info:
        classes.class_of(Permutation.identity(5))
    assert info.value.code == "E-GRP-002"


def test_out_of_range_indices_are_refused():
    classes = ConjugacyClasses(symmetric(3))
    table = CharacterTable(symmetric(3))
    for call in (
        lambda: classes[3],
        lambda: classes.inverse_class(9),
        lambda: classes.class_elements(3),
        lambda: classes.multiplication_matrix(3),
        lambda: table.character(3),
        lambda: table.value(0, 3),
        lambda: table.inner_product(0, 3),
    ):
        with pytest.raises(CharacterError) as info:
            call()
        assert info.value.code == "E-CHAR-006"


def test_caps_are_exported_and_ordered():
    assert CHARACTER_DEFAULT_TABLE_CAP <= CHARACTER_DEFAULT_CLASS_CAP
    assert CHARACTER_DEFAULT_CLASS_CAP <= CHARACTER_MAX_CLASS_CAP
    assert CHARACTER_MAX_EXPONENT_FIELD_DEGREE >= 2
