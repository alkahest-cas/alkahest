"""Permutation groups through the Python binding.

The Rust side carries the exhaustive suite; these tests exist to pin the three
things a binding can get wrong on its own — the **0-based** point convention,
the **left-to-right** composition convention, and the exactness of an order
that does not fit in 64 bits — plus the refusals, which are the part a caller
is most likely to meet and least likely to have read about.
"""

from __future__ import annotations

import math

import pytest
from alkahest.experimental import (
    GROUP_DEFAULT_ELEMENT_CAP,
    GroupError,
    Permutation,
    PermutationGroup,
)

# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_points_are_zero_based():
    p = Permutation([1, 0, 2])
    assert p.degree == 3
    assert p.apply(0) == 1
    assert p.apply(1) == 0
    assert p.cycles() == [[0, 1]]
    assert str(p) == "(0 1)"


def test_one_based_constructor_shifts_the_input_only():
    zero = Permutation.from_cycles(5, [[0, 2, 4]])
    one = Permutation.from_cycles_one_based(5, [[1, 3, 5]])
    assert zero == one
    assert one.images() == [2, 1, 4, 3, 0]


def test_composition_applies_the_left_factor_first():
    p = Permutation.from_cycles(3, [[0, 1]])
    q = Permutation.from_cycles(3, [[1, 2]])
    pq = p.compose(q)
    assert pq == p * q
    for i in range(3):
        assert pq.apply(i) == q.apply(p.apply(i))
    # 0 -p-> 1 -q-> 2
    assert pq.images() == [2, 0, 1]
    # And the reverse order is genuinely different.
    assert (q * p).images() == [1, 2, 0]
    assert pq != q * p


def test_inverse_and_powers():
    p = Permutation([1, 2, 3, 4, 0])
    assert (p * ~p).is_identity()
    assert (~p * p).is_identity()
    assert p.pow(5).is_identity()
    assert p.pow(-1) == p.inverse()
    assert p.pow(0) == Permutation.identity(5)


def test_cycle_structure_sign_and_order():
    p = Permutation([1, 0, 2, 4, 5, 3])  # (0 1)(3 4 5)
    assert p.cycles() == [[0, 1], [3, 4, 5]]
    assert p.cycle_type() == [3, 2, 1]
    assert sum(p.cycle_type()) == p.degree
    assert p.order() == 6
    assert p.sign() == -1
    assert not p.is_even()
    assert p.support() == [0, 1, 3, 4, 5]


def test_permutation_is_hashable_and_comparable():
    a = Permutation([1, 0, 2])
    b = Permutation.from_cycles(3, [[0, 1]])
    assert a == b
    assert len({a, b}) == 1
    assert a != Permutation.identity(3)
    assert a != "not a permutation"


# ---------------------------------------------------------------------------
# Orders of the standard families, and of the Mathieu groups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 8))
def test_symmetric_and_alternating_orders(n):
    assert PermutationGroup.symmetric(n).order() == math.factorial(n)
    assert PermutationGroup.alternating(n).order() == max(math.factorial(n) // 2, 1)


@pytest.mark.parametrize("n", range(3, 10))
def test_dihedral_and_cyclic_orders(n):
    assert PermutationGroup.dihedral(n).order() == 2 * n
    assert PermutationGroup.cyclic(n).order() == n


def test_large_symmetric_order_is_exact_beyond_64_bits():
    order = PermutationGroup.symmetric(30).order()
    assert order == math.factorial(30)
    assert order > 2**64
    assert isinstance(order, int)


def _mathieu_11() -> PermutationGroup:
    a = Permutation.from_cycles_one_based(11, [list(range(1, 12))])
    b = Permutation.from_cycles_one_based(11, [[3, 7, 11, 8], [4, 10, 5, 6]])
    return PermutationGroup(11, [a, b])


def _mathieu_12() -> PermutationGroup:
    a = Permutation.from_cycles_one_based(12, [list(range(1, 12))])
    b = Permutation.from_cycles_one_based(12, [[3, 7, 11, 8], [4, 10, 5, 6]])
    c = Permutation.from_cycles_one_based(12, [[1, 12], [2, 11], [3, 6], [4, 8], [5, 9], [7, 10]])
    return PermutationGroup(12, [a, b, c])


def test_mathieu_group_orders():
    assert _mathieu_11().order() == 7920
    assert _mathieu_12().order() == 95040


def test_mathieu_12_membership():
    g = _mathieu_12()
    assert not g.contains(Permutation.from_cycles(12, [[0, 1]]))
    for generator in g.generators():
        assert generator in g
    assert g.is_transitive()


# ---------------------------------------------------------------------------
# Orbits, Schreier vectors, the chain
# ---------------------------------------------------------------------------


def test_orbits_partition_the_points():
    a = Permutation.from_cycles(6, [[0, 1, 2]])
    b = Permutation.from_cycles(6, [[3, 4]])
    g = PermutationGroup(6, [a, b])
    assert g.orbits() == [[0, 1, 2], [3, 4], [5]]
    assert g.orbit(0) == [0, 1, 2]
    assert not g.is_transitive()
    assert g.order() == 6


def test_schreier_vector_and_transversal_agree():
    g = PermutationGroup.symmetric(5)
    vector = g.schreier_vector(0)
    assert vector[0] is None  # the base point is the root of the walk
    for beta in g.orbit(0):
        u = g.transversal_element(0, beta)
        assert u is not None
        assert u.apply(0) == beta
    assert g.transversal_element(0, 4) is not None


def test_stabilizer_chain_is_exposed_and_consistent():
    g = _mathieu_11()
    base = g.base()
    orbits = g.basic_orbits()
    assert len(base) == len(orbits)
    product = 1
    for orbit in orbits:
        product *= len(orbit)
    assert product == g.order()
    for level, point in enumerate(base):
        for generator in g.stabilizer_generators(level):
            for earlier in base[:level]:
                assert generator.apply(earlier) == earlier
            assert generator in g
        assert point in orbits[level]
    assert g.strong_generators()
    with pytest.raises(IndexError):
        g.stabilizer_generators(len(base))


def test_sift_reports_membership_and_residue():
    g = PermutationGroup.alternating(5)
    even = Permutation.from_cycles(5, [[0, 1, 2]])
    odd = Permutation.from_cycles(5, [[0, 1]])

    hit = g.sift(even)
    assert hit.is_member
    assert hit.residue().is_identity()

    miss = g.sift(odd)
    assert not miss.is_member
    assert not miss.residue().is_identity()


def test_random_elements_are_members_and_reproducible():
    g = _mathieu_12()
    first = g.random_element(7)
    assert g.random_element(7) == first
    for seed in range(20):
        p = g.random_element(seed)
        assert p in g
        assert g.sift(p).is_member


# ---------------------------------------------------------------------------
# Enumeration, and the refusals
# ---------------------------------------------------------------------------


def test_enumeration_is_the_whole_group_exactly_once():
    g = PermutationGroup.symmetric(4)
    elements = g.elements()
    assert len(elements) == 24 == g.order()
    assert len(set(elements)) == 24
    assert all(p in g for p in elements)


def test_enumeration_refuses_above_the_cap_but_order_still_works():
    g = PermutationGroup.symmetric(9)
    assert g.order() == math.factorial(9)
    with pytest.raises(GroupError) as excinfo:
        g.elements()
    assert excinfo.value.code == "E-GRP-004"
    assert str(GROUP_DEFAULT_ELEMENT_CAP) in str(excinfo.value)
    assert len(g.elements(cap=400_000)) == math.factorial(9)


def test_non_bijective_images_are_refused():
    with pytest.raises(GroupError) as excinfo:
        Permutation([0, 0, 2])
    assert excinfo.value.code == "E-GRP-001"


def test_degree_mismatch_is_refused_not_padded():
    p = Permutation.identity(3)
    q = Permutation.identity(4)
    with pytest.raises(GroupError) as excinfo:
        p.compose(q)
    assert excinfo.value.code == "E-GRP-002"
    # The explicit embedding is how you say what you meant.
    assert p.extend_degree(4).compose(q) == q.compose(p.extend_degree(4))


def test_point_out_of_range_is_refused():
    with pytest.raises(GroupError) as excinfo:
        Permutation.identity(3).apply(3)
    assert excinfo.value.code == "E-GRP-003"
    with pytest.raises(GroupError):
        PermutationGroup.symmetric(4).orbit(4)


def test_dihedral_below_three_is_refused_rather_than_wrong():
    with pytest.raises(GroupError) as excinfo:
        PermutationGroup.dihedral(2)
    assert excinfo.value.code == "E-GRP-006"
    with pytest.raises(GroupError):
        PermutationGroup.cyclic(0)


def test_group_error_carries_the_structured_attributes():
    with pytest.raises(GroupError) as excinfo:
        Permutation([5, 1, 2])
    err = excinfo.value
    assert err.code.startswith("E-GRP-")
    assert err.remediation
    assert err.code in str(err)
