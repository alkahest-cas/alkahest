"""Stabilizer codes, the binary symplectic form, and matrix groups over GF(q),
through the Python binding.

The Rust side carries the exhaustive suite. These tests exist to pin the three
things a binding can get wrong on its own — the **`(x | z)`** layout, the
**Hermitian-letter** string convention (`"Y"` is `Y`, not the raw `XZ`), and
the fact that a distance and a *bound* on a distance are different objects —
plus the refusals, which are the part a caller is most likely to meet.
"""

from __future__ import annotations

import pytest
from alkahest.experimental import (
    STABILIZER_MAX_DISTANCE_SEARCH_DIM,
    CssCode,
    FiniteField,
    GfMatrix,
    MatrixGroup,
    PauliOperator,
    StabilizerCode,
    StabilizerError,
    StabilizerGroup,
    is_symplectic,
    symplectic_complement,
    symplectic_form,
    symplectic_gram_matrix,
    symplectic_gram_schmidt,
)

GF2 = FiniteField(2)

# The [7,4,3] Hamming parity-check matrix, used as both H_X and H_Z.
HAMMING = [
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
]

SHOR_HX = [
    [1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1],
]
SHOR_HZ = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
]


def mat(rows):
    return GfMatrix(GF2, rows)


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_the_form_pins_the_x_z_layout():
    # One qubit: X = (1 | 0), Z = (0 | 1).
    assert symplectic_form([1, 0], [0, 1]) == 1
    assert symplectic_form([1, 0], [1, 0]) == 0
    # The form is alternating in every dimension.
    for bits in range(16):
        v = [(bits >> i) & 1 for i in range(4)]
        assert symplectic_form(v, v) == 0


def test_omega_squares_to_the_identity_and_is_symplectic():
    omega = symplectic_gram_matrix(2)
    assert omega.shape() == (4, 4)
    assert is_symplectic(omega)


def test_y_is_the_hermitian_y():
    y = PauliOperator("Y")
    assert str(y) == "+Y"
    assert y.is_hermitian()
    # The stored phase exponent is 1, because Y = i * XZ.
    assert y.phase == 1
    # The raw symplectic product XZ has phase 0 and is not Hermitian.
    xz = PauliOperator.from_xz([1], [1], 0)
    assert not xz.is_hermitian()
    assert str(xz) == "-iY"


@pytest.mark.parametrize("s", ["+XIZY", "-XZZXI", "+iYYY", "-iZZ", "+IIII"])
def test_pauli_strings_round_trip(s):
    assert str(PauliOperator(s)) == s


def test_pauli_multiplication_and_commutation():
    x, z = PauliOperator("X"), PauliOperator("Z")
    assert (x * z) == (z * x).negate()
    assert not x.commutes_with(z)
    assert PauliOperator("XX").commutes_with(PauliOperator("ZZ"))
    assert PauliOperator("XIZY").weight() == 3
    assert (x * x.inverse()).is_identity()


def test_complement_dimension():
    comp = symplectic_complement([[1, 0, 0, 0]], 2)
    assert len(comp) == 3
    for w in comp:
        assert symplectic_form([1, 0, 0, 0], w) == 0


def test_gram_schmidt_returns_pairs_and_a_radical():
    basis = [[1 if i == j else 0 for i in range(4)] for j in range(4)]
    pairs, radical = symplectic_gram_schmidt(basis, 2)
    assert len(pairs) == 2
    assert radical == []
    for u, v in pairs:
        assert symplectic_form(u, v) == 1


# ---------------------------------------------------------------------------
# The anchor codes
# ---------------------------------------------------------------------------


def assert_logicals_are_well_formed(code):
    for lx in code.logical_x:
        for g in code.stabilizer.generators:
            assert g.commutes_with(lx)
    for lz in code.logical_z:
        for g in code.stabilizer.generators:
            assert g.commutes_with(lz)
    for i in range(code.k):
        for j in range(code.k):
            want = 1 if i == j else 0
            assert code.logical_x[i].symplectic_product(code.logical_z[j]) == want


def test_steane_is_7_1_3():
    h = mat(HAMMING)
    code = CssCode(h, h)
    assert (code.n, code.k) == (7, 1)
    assert (code.x_rank, code.z_rank) == (3, 3)
    d = code.minimum_distance()
    assert d.exact
    assert d.value == 3
    assert repr(d) == "Distance(exact=3)"


def test_steane_agrees_with_the_general_stabilizer_path():
    h = mat(HAMMING)
    css = CssCode(h, h)
    general = css.to_stabilizer_code()
    assert (general.n, general.k) == (7, 1)
    assert general.minimum_distance() == css.minimum_distance()
    assert_logicals_are_well_formed(general)


def test_shor_is_9_1_3():
    code = CssCode(mat(SHOR_HX), mat(SHOR_HZ))
    assert (code.n, code.k) == (9, 1)
    assert code.minimum_distance().value == 3
    assert_logicals_are_well_formed(code.to_stabilizer_code())


def test_five_qubit_perfect_code_is_5_1_3():
    gens = [PauliOperator(s) for s in ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")]
    code = StabilizerCode(gens)
    assert (code.n, code.k) == (5, 1)
    assert code.minimum_distance().value == 3
    assert_logicals_are_well_formed(code)

    # Perfect: the 15 weight-one errors have 15 distinct non-zero syndromes.
    seen = set()
    for q in range(5):
        for letter in "XYZ":
            s = ["I"] * 5
            s[q] = letter
            syn = tuple(code.syndrome(PauliOperator("".join(s))))
            assert any(syn)
            seen.add(syn)
    assert len(seen) == 15


def test_four_two_two_code_has_two_logical_pairs():
    code = StabilizerCode([PauliOperator("XXXX"), PauliOperator("ZZZZ")])
    assert (code.n, code.k) == (4, 2)
    assert code.minimum_distance().value == 2
    assert_logicals_are_well_formed(code)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_broken_css_condition_is_refused():
    bad = [row[:] for row in HAMMING]
    bad[0][0] ^= 1
    with pytest.raises(StabilizerError) as exc:
        CssCode(mat(bad), mat(HAMMING))
    assert exc.value.code == "E-STAB-005"


def test_anticommuting_generators_are_refused():
    with pytest.raises(StabilizerError) as exc:
        StabilizerGroup([PauliOperator("XI"), PauliOperator("ZI")])
    assert exc.value.code == "E-STAB-004"


def test_minus_identity_in_the_group_is_refused():
    with pytest.raises(StabilizerError) as exc:
        StabilizerGroup([PauliOperator("ZZ"), PauliOperator("-ZZ")])
    assert exc.value.code == "E-STAB-006"


def test_an_empty_generator_list_needs_an_explicit_qubit_count():
    with pytest.raises(StabilizerError) as exc:
        StabilizerGroup([])
    assert exc.value.code == "E-STAB-002"
    code = StabilizerCode([], qubits=3)
    assert (code.n, code.k) == (3, 3)


def test_a_k_zero_code_has_no_distance():
    code = StabilizerCode([PauliOperator("Z")])
    assert code.k == 0
    with pytest.raises(StabilizerError) as exc:
        code.minimum_distance()
    assert exc.value.code == "E-STAB-007"


def test_a_capped_distance_search_refuses_and_offers_a_bound():
    code = CssCode(mat(HAMMING), mat(HAMMING)).to_stabilizer_code()
    with pytest.raises(StabilizerError) as exc:
        code.minimum_distance(cap=4)
    assert exc.value.code == "E-STAB-008"
    bound = code.distance_upper_bound()
    assert not bound.exact
    assert bound.value >= 3
    assert repr(bound).startswith("Distance(at_most=")
    # A bound is not equal to the exact distance of the same value.
    assert bound != code.minimum_distance()


def test_a_non_binary_field_is_refused():
    gf3 = FiniteField(3)
    h = GfMatrix(gf3, [[1, 1, 1]])
    with pytest.raises(StabilizerError) as exc:
        CssCode(h, h)
    assert exc.value.code == "E-STAB-001"


def test_a_gfq_failure_keeps_its_own_code():
    # 2n is odd -> the symplectic layer refuses with its own code.
    with pytest.raises(StabilizerError) as exc:
        symplectic_form([1, 0], [1, 0, 0, 0])
    assert exc.value.code == "E-STAB-002"


def test_the_distance_cap_is_exposed():
    assert STABILIZER_MAX_DISTANCE_SEARCH_DIM >= 20


# ---------------------------------------------------------------------------
# Matrix groups
# ---------------------------------------------------------------------------


def test_classical_group_orders():
    assert MatrixGroup.general_linear(GF2, 2).order == 6
    assert MatrixGroup.general_linear(GF2, 3).order == 168
    assert MatrixGroup.symplectic(GF2, 1).order == 6
    assert MatrixGroup.symplectic(GF2, 2).order == 720
    gf3 = FiniteField(3)
    assert MatrixGroup.general_linear(gf3, 2).order == 48
    assert MatrixGroup.special_linear(gf3, 2).order == 24


def test_the_order_is_exact_beyond_64_bits():
    big = FiniteField(2147483647)
    order = MatrixGroup.general_linear(big, 10).order
    assert isinstance(order, int)
    assert order > 2**64


def test_enumeration_agrees_with_the_formula():
    g = MatrixGroup.general_linear(GF2, 3)
    assert len(g.elements()) == g.order
    s = MatrixGroup.symplectic(GF2, 2)
    assert len(s.elements()) == s.order


def test_enumeration_is_capped_but_the_order_is_not():
    g = MatrixGroup.symplectic(FiniteField(3), 2)
    with pytest.raises(StabilizerError) as exc:
        g.elements()
    assert exc.value.code == "E-STAB-011"
    assert g.order == 51840


def test_permutation_action_reproduces_the_order():
    g = MatrixGroup.general_linear(GF2, 3)
    perm = g.permutation_action()
    assert perm.degree == 7
    assert perm.order() == 168
    s = MatrixGroup.symplectic(GF2, 2)
    assert s.permutation_action().order() == 720


def test_membership():
    gl = MatrixGroup.general_linear(GF2, 2)
    assert gl.contains(mat([[1, 0], [0, 1]]))
    assert not gl.contains(mat([[1, 1], [1, 1]]))
    sp = MatrixGroup.symplectic(GF2, 1)
    assert sp.contains(mat([[1, 1], [0, 1]]))
    assert repr(gl) == "MatrixGroup(GL(2, 2))"
