"""Finitely presented groups through the Python binding.

The Rust side carries the exhaustive suite; these tests pin the things a binding
can get wrong on its own — the relator syntax, the signed 1-based letters, the
0-based cosets, an order that has to come back as an exact ``int`` — and above
all the **two refusals**, which a caller is most likely to meet and least likely
to have read about:

* ``E-FPGRP-004`` — the coset enumeration hit its cap, which says *nothing*
  about whether the group is finite.
* ``E-FPGRP-005`` — the group **is** infinite, proved from its abelianisation.

Conflating those two is the failure this module is arranged to prevent, so the
distinction is asserted here and not only in Rust.
"""

from __future__ import annotations

import pytest
from alkahest.experimental import (
    FPGROUP_DEFAULT_MAX_COSETS,
    FPGROUP_MAX_COHOMOLOGY_DEGREE,
    AbelianInvariants,
    CosetTable,
    FpGroup,
    FpGroupError,
    PermutationGroup,
    SubgroupPresentation,
    Word,
)


def a5() -> FpGroup:
    return FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^5"])


# ---------------------------------------------------------------------------
# Presentations and words
# ---------------------------------------------------------------------------


def test_the_classical_orders():
    assert a5().order() == 60
    assert FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^4"]).order() == 24
    assert FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^3"]).order() == 12
    assert FpGroup(["a"], ["a^7"]).order() == 7
    for n in range(1, 8):
        assert FpGroup(["a", "b"], ["a^2", "b^2", f"(a*b)^{n}"]).order() == 2 * n


def test_relator_syntax():
    g = FpGroup(["a", "b"], ["a^2", "b^3", "ababababab"])
    assert g.order() == 60
    assert g.generators == ["a", "b"]
    assert g.rank == 2
    assert [len(r) for r in g.relators] == [2, 3, 10]
    assert FpGroup(["a"]).relators == []
    assert FpGroup.free(2).generators == ["x1", "x2"]


def test_words_are_signed_one_based_and_freely_reduced():
    w = Word([1, 2, -2, -1, 3])
    assert w.letters == [3]
    assert len(w) == 1
    assert Word().is_identity()
    assert (Word([1]) * Word([-1])).is_identity()
    assert Word([1, 2]).inverse().letters == [-2, -1]
    assert (~Word([1, 2])).letters == [-2, -1]
    assert (Word([1]) ** 3).letters == [1, 1, 1]
    assert (Word([1]) ** -2).letters == [-1, -1]
    assert Word([1, 2]) == Word([1, 2, 2, -2])
    assert hash(Word([1, 2])) == hash(Word([1, 2]))
    assert Word([1, 1]).exponent_sums(2) == [2, 0]
    with pytest.raises(FpGroupError) as exc:
        Word([1, 0])
    assert exc.value.code == "E-FPGRP-001"


def test_parse_and_repr():
    g = a5()
    assert g.parse("(a*b)^5").letters == [1, 2] * 5
    assert g.parse("1").is_identity()
    assert g.generator(0) == Word([1])
    assert "a" in repr(g)
    with pytest.raises(FpGroupError) as exc:
        g.parse("c")
    assert exc.value.code == "E-FPGRP-003"


# ---------------------------------------------------------------------------
# The two refusals
# ---------------------------------------------------------------------------


def test_an_incomplete_enumeration_is_not_a_claim_of_infiniteness():
    # The (2,3,7) triangle group is infinite, and no cap completes it.
    g = FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^7"])
    assert g.abelian_invariants().is_trivial()
    with pytest.raises(FpGroupError) as exc:
        g.order(max_cosets=2000)
    assert exc.value.code == "E-FPGRP-004"
    assert "NOT a claim" in str(exc.value)


def test_a_provably_infinite_group_has_its_own_code():
    z2 = FpGroup(["a", "b"], ["a*b*a^-1*b^-1"])
    with pytest.raises(FpGroupError) as exc:
        z2.order()
    assert exc.value.code == "E-FPGRP-005"
    assert exc.value.remediation
    # ... and the abelianisation, which always terminates, says why.
    inv = z2.abelian_invariants()
    assert inv.free_rank == 2
    assert inv.order() is None
    assert str(inv) == "Z^2"


def test_a_bare_string_subgroup_is_refused_rather_than_misread():
    # Iterating "ab" would give the two generators a and b, not the one word ab.
    with pytest.raises(TypeError):
        a5().index("ab")


def test_out_of_range_arguments_refuse():
    g = a5()
    with pytest.raises(FpGroupError) as exc:
        g.order(max_cosets=0)
    assert exc.value.code == "E-FPGRP-006"
    table = g.coset_table()
    with pytest.raises(FpGroupError) as exc:
        table.image(60, 1)
    assert exc.value.code == "E-FPGRP-007"
    with pytest.raises(FpGroupError) as exc:
        table.image(0, 3)
    assert exc.value.code == "E-FPGRP-001"


# ---------------------------------------------------------------------------
# Coset tables and the permutation representation
# ---------------------------------------------------------------------------


def test_the_coset_table_is_zero_based_with_coset_zero_the_subgroup():
    g = a5()
    table = g.coset_table(["a"])
    assert isinstance(table, CosetTable)
    assert table.index == 30
    assert table.rank == 2
    assert table.trace(0, g.parse("a")) == 0, "H fixes its own coset"
    rows = table.rows()
    assert len(rows) == 30
    assert all(len(r) == 4 for r in rows)
    for c, row in enumerate(rows):
        for x, d in enumerate(row):
            assert rows[d][x ^ 1] == c
    assert table.max_cosets == FPGROUP_DEFAULT_MAX_COSETS
    assert table.strategy == "HLT with lookahead"
    assert table.cosets_defined >= table.index
    assert len(table.transversal()) == 30


def test_the_permutation_representation_is_an_independent_check():
    g = a5()
    perm = g.permutation_group()
    assert isinstance(perm, PermutationGroup)
    assert perm.degree == 60
    # Schreier-Sims, in a module that knows nothing about presentations.
    assert perm.order() == 60
    perms = g.coset_table().permutations()
    assert [p.order() for p in perms] == [2, 3]
    assert (perms[0] * perms[1]).order() == 5
    # On the cosets of <b>, degree 20 and still faithful.
    small = g.permutation_group(["b"])
    assert small.degree == 20
    assert small.order() == 60


def test_the_multiplication_table_is_a_group():
    g = FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^3"])
    mult = g.multiplication_table()
    assert len(mult) == 12
    assert mult[0] == list(range(12))
    for row in mult:
        assert sorted(row) == list(range(12))


# ---------------------------------------------------------------------------
# Abelianisation
# ---------------------------------------------------------------------------


def test_abelian_invariants():
    assert a5().abelian_invariants().is_trivial(), "A5 is perfect"
    inv = FpGroup(["a"], ["a^6"]).abelian_invariants()
    assert inv.torsion == [6]
    assert inv.order() == 6
    assert inv.is_finite()
    assert str(inv) == "Z/6"
    assert isinstance(inv, AbelianInvariants)
    s4 = FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^4"]).abelian_invariants()
    assert s4.torsion == [2]
    assert FpGroup.free(3).abelian_invariants().free_rank == 3
    assert a5().relation_matrix() == [[2, 0], [0, 3], [5, 5]]


# ---------------------------------------------------------------------------
# Reidemeister-Schreier
# ---------------------------------------------------------------------------


def test_subgroup_presentation_satisfies_lagrange():
    g = a5()
    sub = g.subgroup_presentation(["a"])
    assert isinstance(sub, SubgroupPresentation)
    assert sub.index == 30
    assert sub.rank == 30 * 2 - 29
    # |H| from a second enumeration, on the rewritten presentation.
    assert sub.presentation().order() == 2
    assert sub.index * sub.presentation().order() == 60
    table = g.coset_table(["a"])
    for w in sub.generator_words():
        assert table.trace(0, w) == 0, "a Schreier generator must lie in H"


def test_a_finite_index_subgroup_of_z_squared_is_z_squared():
    z2 = FpGroup(["a", "b"], ["a*b*a^-1*b^-1"])
    sub = z2.subgroup_presentation(["a", "b^2"])
    assert sub.index == 2
    assert sub.presentation().abelian_invariants().free_rank == 2


# ---------------------------------------------------------------------------
# Cohomology
# ---------------------------------------------------------------------------


def test_cohomology_of_cyclic_groups_with_trivial_coefficients():
    for n in (2, 3, 4, 6):
        g = FpGroup(["a"], [f"a^{n}"])
        # H^0(G, Z) = Z, H^1(G, Z) = Hom(G, Z) = 0, H^2(G, Z) = Z/n.
        assert str(g.cohomology(0, [0])) == "Z"
        assert g.cohomology(1, [0]).is_trivial()
        assert str(g.cohomology(2, [0])) == f"Z/{n}"
        assert g.cohomology(2, [0]).order() == n


def test_h2_of_the_klein_four_group_with_f2_coefficients_has_order_eight():
    g = FpGroup(["a", "b"], ["a^2", "b^2", "a*b*a^-1*b^-1"])
    assert g.cohomology(2, [2]).order() == 8
    assert g.cohomology(1, [2]).order() == 4, "Hom((Z/2)^2, Z/2)"
    assert g.cohomology(2, [0]).order() == 4, "the dual of the abelianisation"


def test_cohomology_with_a_non_trivial_action():
    # Z/2 acting on Z by -1: H^0 = 0, H^1 = Z/2, H^2 = 0.
    g = FpGroup(["a"], ["a^2"])
    action = [[[-1]]]
    assert g.cohomology(0, [0], action=action).is_trivial()
    assert str(g.cohomology(1, [0], action=action)) == "Z/2"
    assert g.cohomology(2, [0], action=action).is_trivial()


def test_cohomology_refuses_loudly():
    g = FpGroup(["a"], ["a^4"])
    with pytest.raises(FpGroupError) as exc:
        g.cohomology(FPGROUP_MAX_COHOMOLOGY_DEGREE + 1, [0])
    assert exc.value.code == "E-FPGRP-008"
    # A module whose "action" does not satisfy the relators is refused, never
    # used: multiplication by 2 does not have order 4.
    with pytest.raises(FpGroupError) as exc:
        g.cohomology(1, [0], action=[[[2]]])
    assert exc.value.code == "E-FPGRP-011"
    # Wrong number of action matrices.
    with pytest.raises(FpGroupError) as exc:
        g.cohomology(1, [0], action=[[[1]], [[1]]])
    assert exc.value.code == "E-FPGRP-010"
    # A group too large for H^2.
    with pytest.raises(FpGroupError) as exc:
        FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^5"]).cohomology(2, [0])
    assert exc.value.code == "E-FPGRP-009"


def test_refusals_carry_a_stable_code_and_are_value_errors():
    # The native exception hierarchy hangs off ValueError, which is what keeps
    # `except ValueError` working for callers written before the codes existed.
    assert issubclass(FpGroupError, ValueError)
    with pytest.raises(FpGroupError) as exc:
        FpGroup(["a", "a"], [])
    assert exc.value.code == "E-FPGRP-002"
    assert exc.value.remediation
    assert "E-FPGRP-002" in str(exc.value)
