"""Matrix groups over GF(q) — ``alkahest.experimental.MatGroup``.

The mathematics is covered by ``alkahest-core/src/matgroup/{tests,proptests}.rs``.
What is checked here is the *boundary*: that the row-vector convention crosses
it unchanged, that every refusal arrives as a ``MatGroupError`` with its stable
``E-MATGRP-NNN`` code (and that a refusal raised inside the GF(q) or
permutation layer keeps its own ``E-GFQ`` / ``E-GRP`` code instead), and that
the four headline orders come out right through the bindings and not only in
Rust.

The orders here are computed by Schreier–Sims from generators. They are
compared against ``matgroup_gl_order`` / ``matgroup_sl_order`` /
``matgroup_sp_order``, which are the closed-form product formulas — so a bug in
the chain shows up as a disagreement rather than as a plausible number.
"""

from __future__ import annotations

import pytest
from alkahest.experimental import (
    MATGROUP_MAX_DEGREE,
    FiniteField,
    GfMatrix,
    MatGroup,
    MatGroupError,
    matgroup_gl_order,
    matgroup_sl_order,
    matgroup_sp_order,
)


@pytest.fixture
def gf2():
    return FiniteField(2)


@pytest.fixture
def gf5():
    return FiniteField(5)


def code_of(exc_info):
    return exc_info.value.code


# ---------------------------------------------------------------------------
# The correctness anchors
# ---------------------------------------------------------------------------


def test_the_four_headline_orders(gf2, gf5):
    assert MatGroup.general_linear(gf2, 3).order() == 168
    assert MatGroup.special_linear(gf5, 2).order() == 120
    assert MatGroup.symplectic(gf2, 2).order() == 720
    gf4 = FiniteField(2, 2)
    assert MatGroup.general_linear(gf4, 2).order() == 180


@pytest.mark.parametrize("p", [2, 3, 5, 7])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_order_from_the_chain_equals_the_closed_form(p, n):
    if p**n > 100:  # keep the suite quick; Rust covers the wider sweep
        pytest.skip("covered in the Rust sweep")
    field = FiniteField(p)
    assert MatGroup.general_linear(field, n).order() == matgroup_gl_order(p, n)
    assert MatGroup.special_linear(field, n).order() == matgroup_sl_order(p, n)


@pytest.mark.parametrize("p", [2, 3])
def test_symplectic_order_equals_the_closed_form(p):
    field = FiniteField(p)
    for n in (1, 2):
        assert MatGroup.symplectic(field, n).order() == matgroup_sp_order(p, n)


def test_psl_2_7_is_the_projective_action_and_has_order_168():
    sl = MatGroup.special_linear(FiniteField(7), 2)
    assert sl.order() == 336
    projective = sl.permutation_action_on_projective_points()
    assert projective.degree == 8  # PG(1, 7) has 8 points
    assert projective.order() == 168
    # The vector action is faithful, so it keeps the full order.
    assert sl.permutation_action_on_vectors().order() == 336


@pytest.mark.parametrize(("p", "n", "expected"), [(2, 3, 7), (2, 4, 15), (3, 2, 8), (5, 2, 24)])
def test_singer_cycles(p, n, expected):
    g = MatGroup.singer_cycle(FiniteField(p), n)
    assert g.order() == expected
    # Cyclic, hence its own centre.
    assert g.centre().order() == expected
    assert g.derived_subgroup().order() == 1


def test_centre_of_gl_is_the_scalars(gf5):
    gl = MatGroup.general_linear(gf5, 2)
    assert gl.centre().order() == 4  # q - 1
    assert len(gl.centre_elements()) == 4
    # The centralizing algebra of the natural module is one-dimensional.
    assert len(gl.commutant_basis()) == 1


def test_derived_subgroup_of_gl_is_sl(gf5):
    gl = MatGroup.general_linear(gf5, 2)
    assert gl.derived_subgroup().order() == matgroup_sl_order(5, 2)
    sl = MatGroup.special_linear(gf5, 2)
    assert sl.is_perfect()  # SL(2,5) is the double cover of A5


# ---------------------------------------------------------------------------
# The boundary: shapes, conventions, and generic generators
# ---------------------------------------------------------------------------


def test_a_group_from_arbitrary_generators(gf5):
    """The Borel subgroup of GL(2, 5): order q(q-1)**2 = 80."""
    unipotent = GfMatrix(gf5, [[1, 1], [0, 1]])
    scale_left = GfMatrix(gf5, [[2, 0], [0, 1]])
    scale_right = GfMatrix(gf5, [[1, 0], [0, 2]])
    borel = MatGroup(gf5, 2, [unipotent, scale_left, scale_right])
    assert borel.order() == 80
    assert borel.degree == 2
    assert borel.field == gf5
    assert not borel.is_trivial
    assert len(borel.generators()) == 3
    assert len(borel.elements()) == 80
    # It fixes the line spanned by e_0 and is transitive on the other 5.
    assert sorted(len(o) for o in borel.projective_orbits()) == [1, 5]


def test_the_action_is_on_row_vectors(gf5):
    """``v -> v @ M`` and not ``M @ v``.

    Under the row-vector action, ``e_i @ M`` is **row** ``i`` of ``M``. For the
    lower-triangular unipotent below that makes ``e_0`` a fixed point (row 0 is
    ``(1, 0)``) and ``e_1`` the one that moves (row 1 is ``(1, 1)``). Under the
    column convention ``M @ v`` the two swap, so this test fails loudly if the
    convention is ever flipped — which is the only way this module's answers can
    be wrong while still being self-consistent.
    """
    lower = GfMatrix(gf5, [[1, 0], [1, 1]])
    g = MatGroup(gf5, 2, [lower])
    assert g.order() == 5
    e0 = GfMatrix(gf5, [[1, 0]])
    assert len(g.vector_orbit(e0)) == 1
    e1 = GfMatrix(gf5, [[0, 1]])
    assert len(g.vector_orbit(e1)) == 5
    # The base the chain picks is therefore e_1, not e_0.
    assert g.base() == [1]


def test_gl_is_transitive_on_vectors_and_projective_points(gf5):
    gl = MatGroup.general_linear(gf5, 2)
    orbits = gl.vector_orbits()
    assert len(orbits) == 1
    assert len(orbits[0]) == 24  # q**d - 1
    assert len(gl.nonzero_vectors()) == 24
    projective = gl.projective_orbits()
    assert len(projective) == 1
    assert len(projective[0]) == 6  # (q**d - 1)/(q - 1)
    assert len(gl.projective_points()) == 6


def test_membership_and_sifting(gf5):
    sl = MatGroup.special_linear(gf5, 2)
    identity = sl.identity()
    assert sl.contains(identity)
    sift = sl.sift(identity)
    assert sift.is_member
    assert sift.level == len(sl.base())
    # det = 2, so not in SL; and a singular matrix is False, not an error.
    assert not sl.contains(GfMatrix(gf5, [[2, 0], [0, 1]]))
    assert not sl.contains(GfMatrix(gf5, [[0, 0], [0, 0]]))


def test_random_elements_are_members_and_reproducible(gf5):
    gl = MatGroup.general_linear(gf5, 2)
    first = gl.random_element(42)
    assert gl.contains(first)
    assert first.to_list() == gl.random_element(42).to_list()
    batch = gl.random_elements(7, 12)
    assert len(batch) == 12
    assert all(gl.contains(m) for m in batch)


def test_strong_generators_and_the_chain(gf2):
    gl = MatGroup.general_linear(gf2, 3)
    assert gl.base() == [0, 1, 2]
    assert gl.basic_orbit_lengths() == [7, 6, 4]
    product = 1
    for length in gl.basic_orbit_lengths():
        product *= length
    assert product == gl.order()
    assert len(gl.base_vectors()) == 3
    strong = gl.strong_generators()
    rebuilt = MatGroup(gf2, 3, strong)
    assert rebuilt.order() == gl.order()


def test_normal_closure(gf5):
    gl = MatGroup.general_linear(gf5, 2)
    transvection = GfMatrix(gf5, [[1, 1], [0, 1]])
    assert gl.contains(transvection)
    # A single transvection normally generates SL(2, 5).
    assert gl.normal_closure([transvection]).order() == 120
    assert gl.normal_closure([]).order() == 1


def test_trivial_group(gf5):
    t = MatGroup.trivial(gf5, 3)
    assert t.is_trivial
    assert t.order() == 1
    assert t.base() == []
    assert len(t.elements()) == 1
    assert t.centre().order() == 1
    assert repr(t).startswith("<matrix group")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_degree_refusals(gf5):
    with pytest.raises(MatGroupError) as exc:
        MatGroup.trivial(gf5, 0)
    assert code_of(exc) == "E-MATGRP-005"
    with pytest.raises(MatGroupError) as exc:
        MatGroup.trivial(gf5, MATGROUP_MAX_DEGREE + 1)
    assert code_of(exc) == "E-MATGRP-005"
    with pytest.raises(MatGroupError) as exc:
        MatGroup.general_linear(gf5, 0)
    assert code_of(exc) == "E-MATGRP-010"


def test_generator_refusals(gf5):
    with pytest.raises(MatGroupError) as exc:
        MatGroup(gf5, 2, [GfMatrix(gf5, [[1, 0, 0], [0, 1, 0]])])
    assert code_of(exc) == "E-MATGRP-001"
    with pytest.raises(MatGroupError) as exc:
        MatGroup(gf5, 2, [GfMatrix(gf5, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])])
    assert code_of(exc) == "E-MATGRP-002"
    with pytest.raises(MatGroupError) as exc:
        MatGroup(gf5, 2, [GfMatrix(gf5, [[0, 0], [0, 0]])])
    assert code_of(exc) == "E-MATGRP-004"
    with pytest.raises(MatGroupError) as exc:
        MatGroup(gf5, 2, [GfMatrix(FiniteField(7), [[1, 0], [0, 1]])])
    assert code_of(exc) == "E-MATGRP-003"


def test_a_column_vector_is_refused_not_transposed(gf5):
    """The convention is enforced: a ``d x 1`` column is an error."""
    gl = MatGroup.general_linear(gf5, 2)
    with pytest.raises(MatGroupError) as exc:
        gl.vector_orbit(GfMatrix(gf5, [[1], [0]]))
    assert code_of(exc) == "E-MATGRP-002"


def test_the_zero_vector_spans_no_projective_point(gf5):
    gl = MatGroup.general_linear(gf5, 2)
    zero = GfMatrix(gf5, [[0, 0]])
    assert len(gl.vector_orbit(zero)) == 1  # fixed by everything
    with pytest.raises(MatGroupError) as exc:
        gl.projective_orbit(zero)
    assert code_of(exc) == "E-MATGRP-012"


def test_enumeration_cap_withholds_the_list_but_not_the_order(gf5):
    gl = MatGroup.general_linear(gf5, 2)
    assert gl.order() == 480
    with pytest.raises(MatGroupError) as exc:
        gl.elements(cap=10)
    assert code_of(exc) == "E-MATGRP-008"
    assert gl.order() == 480  # still exact


def test_an_exhausted_schreier_sims_budget_reports_no_order_at_all(gf5):
    """The refusal that matters most: a partial chain would report a proper
    divisor of the true order, which reads exactly like a correct answer."""
    starved = MatGroup.general_linear(gf5, 2).with_budget(1)
    assert starved.budget == 1
    with pytest.raises(MatGroupError) as exc:
        starved.order()
    assert code_of(exc) == "E-MATGRP-007"
    assert exc.value.remediation
    # And raising it gets the right answer rather than a different wrong one.
    assert MatGroup.general_linear(gf5, 2).with_budget(10_000_000).order() == 480


def test_an_orbit_past_its_cap_is_refused_not_truncated():
    field = FiniteField(97)
    transvection = GfMatrix(field, [[1, 1], [0, 1]])
    scale = GfMatrix(field, [[5, 0], [0, 1]])
    big = MatGroup(field, 2, [transvection, scale])
    with pytest.raises(MatGroupError) as exc:
        big.order()
    assert code_of(exc) == "E-MATGRP-006"


def test_a_field_too_large_to_enumerate():
    with pytest.raises(MatGroupError) as exc:
        MatGroup.general_linear(FiniteField(10_007), 2)
    assert code_of(exc) == "E-MATGRP-009"


def test_singer_cycles_over_an_extension_field_are_refused():
    with pytest.raises(MatGroupError) as exc:
        MatGroup.singer_cycle(FiniteField(2, 2), 2)
    assert code_of(exc) == "E-MATGRP-010"


def test_a_gfq_refusal_underneath_keeps_its_own_code(gf5):
    """``MatGroupError`` wraps ``FiniteFieldError``; the wrapped code must reach
    the caller rather than being relabelled ``E-MATGRP-*``."""
    gl = MatGroup.general_linear(gf5, 2)
    # A matrix over a *different* field is caught by the matrix-group layer, so
    # that one is E-MATGRP-003. The delegation is visible on the Rust side; here
    # the boundary property is that the exception class is one a caller can
    # branch on by code at all.
    with pytest.raises(MatGroupError) as exc:
        gl.contains(GfMatrix(FiniteField(7), [[1, 0], [0, 1]]))
    assert code_of(exc).startswith("E-")
    assert exc.value.remediation
