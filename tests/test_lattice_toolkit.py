"""The lattice toolkit — ``alkahest.experimental.Lattice``.

The mathematics is covered on the Rust side
(``alkahest-core/src/lattice/tests.rs``, which is where ``E_8``'s 240 roots and
the Leech lattice's 196560 minimal vectors are pinned). What is checked here is
the *boundary*: that exact values cross it as ``int`` / ``Fraction`` and never
as ``float``, that a ``float`` going the other way is refused rather than
rounded, and that every refusal arrives as a ``LatticeError`` carrying its
stable ``E-LAT-NNN`` code — in particular that asking for a shortest vector
above the enumeration ceiling raises instead of approximating.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from alkahest import LatticeError
from alkahest.experimental import (
    LATTICE_MAX_ENUM_RANK,
    Lattice,
)


def test_standard_lattices_have_their_textbook_invariants():
    z3 = Lattice.zn(3)
    assert z3.rank == 3
    assert z3.determinant() == 1
    assert z3.minimum() == 1
    assert z3.kissing_number() == 6

    a2 = Lattice.a_n(2)
    assert a2.determinant() == 3
    assert a2.kissing_number() == 6

    d4 = Lattice.d_n(4)
    assert d4.determinant() == 4
    assert d4.kissing_number() == 24
    assert d4.is_even

    e8 = Lattice.e8()
    assert e8.determinant() == 1
    assert e8.minimum() == 2
    assert e8.kissing_number() == 240
    assert len(e8.minimal_vectors()) == 240


def test_e8_center_density_is_exactly_one_sixteenth():
    e8 = Lattice.e8()
    assert e8.center_density_exact() == Fraction(1, 16)
    assert e8.center_density() == pytest.approx(1 / 16)
    assert e8.hermite_invariant() == pytest.approx(2.0)
    # Δ = π⁴/384
    assert e8.packing_density() == pytest.approx(0.2536695079, rel=1e-8)


def test_theta_series_matches_the_sum_of_squares_counts():
    # r_2(n): 1, 4, 4, 0, 4, 8, 0, 0, 4, 4, …
    assert Lattice.zn(2).theta_series(9) == [1, 4, 4, 0, 4, 8, 0, 0, 4, 4]
    # Jacobi: r_4(n) = 8·σ(n) for odd n.
    theta4 = Lattice.zn(4).theta_series(9)
    for n, sigma in [(1, 1), (3, 4), (5, 6), (7, 8), (9, 13)]:
        assert theta4[n] == 8 * sigma
    # E_8's theta series is the Eisenstein series E_4 = 1 + 240 q + 2160 q² + …
    assert Lattice.e8().theta_series(4) == [1, 0, 240, 0, 2160]


def test_exact_values_never_arrive_as_floats():
    dual = Lattice.a_n(3).dual()
    det = dual.determinant()
    assert det == Fraction(1, 4)
    assert isinstance(det, Fraction)
    assert not isinstance(det, float)
    # Integral values come back as plain ints, not Fraction(n, 1).
    assert isinstance(Lattice.zn(2).determinant(), int)
    gram = Lattice.a_n(2).gram_matrix()
    assert gram == [[2, -1], [-1, 2]]


def test_float_input_is_refused_rather_than_rounded():
    with pytest.raises(TypeError, match="exact"):
        Lattice.from_basis([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError, match="exact"):
        Lattice.zn(2).closest_vector([0.5, 0.5])
    # …but strings and Fractions are fine.
    lat = Lattice.from_basis([["1/2", 0], [0, "1/2"]])
    assert lat.determinant() == Fraction(1, 16)
    v = Lattice.zn(2).closest_vector([Fraction(2, 5), Fraction(3, 5)])
    assert v.coordinates() == [0, 1]
    assert v.norm() == Fraction(2, 5) ** 2 + Fraction(2, 5) ** 2


def test_gram_only_lattice_has_no_ambient_coordinates():
    e8 = Lattice.e8()
    gram_only = Lattice.from_gram(e8.gram_matrix())
    assert gram_only.basis() is None
    assert gram_only.determinant() == 1
    assert gram_only.shortest_vector().coordinates() is None
    with pytest.raises(LatticeError) as exc:
        gram_only.closest_vector([0] * 8)
    assert exc.value.code == "E-LAT-013"


def test_enumeration_refuses_above_the_rank_ceiling():
    n = LATTICE_MAX_ENUM_RANK + 1
    big = Lattice.zn(n)
    assert big.rank == n
    with pytest.raises(LatticeError) as exc:
        big.shortest_vector()
    assert exc.value.code == "E-LAT-008"
    # The refusal says what to do instead rather than handing back a guess.
    assert exc.value.remediation


def test_node_budget_is_a_refusal_not_a_truncation():
    with pytest.raises(LatticeError) as exc:
        Lattice.zn(6).theta_series(400, budget=50)
    assert exc.value.code == "E-LAT-009"


def test_bad_gram_matrices_are_refused_with_codes():
    with pytest.raises(LatticeError) as exc:
        Lattice.from_gram([[1, 0]])
    assert exc.value.code == "E-LAT-005"

    with pytest.raises(LatticeError) as exc:
        Lattice.from_gram([[1, 2], [3, 1]])
    assert exc.value.code == "E-LAT-006"

    with pytest.raises(LatticeError) as exc:
        Lattice.from_gram([[1, 2], [2, 1]])
    assert exc.value.code == "E-LAT-007"

    with pytest.raises(LatticeError) as exc:
        Lattice.from_basis([[1, 2], [2, 4]])
    assert exc.value.code == "E-LAT-007"

    non_integral = Lattice.from_gram([["1/3", 0], [0, 1]])
    with pytest.raises(LatticeError) as exc:
        non_integral.theta_series(4)
    assert exc.value.code == "E-LAT-010"

    with pytest.raises(LatticeError) as exc:
        Lattice.zn(2).closest_vector([1])
    assert exc.value.code == "E-LAT-011"

    with pytest.raises(LatticeError) as exc:
        Lattice.a_n(0)
    assert exc.value.code == "E-LAT-012"


def test_dual_of_dual_is_the_original():
    for lat in [Lattice.zn(3), Lattice.a_n(3), Lattice.d_n(4), Lattice.e8()]:
        assert lat.dual().dual().gram_matrix() == lat.gram_matrix()


def test_lll_reduced_keeps_the_lattice():
    lat = Lattice.from_basis([[1, 0, 0], [0, 1, 0], [9001, 8999, 1]])
    reduced = lat.lll_reduced()
    assert reduced.determinant() == lat.determinant()
    assert reduced.minimum() == 1
    # Big integers survive the round trip exactly.
    huge = 10**30
    big = Lattice.from_basis([[1, 0, huge], [0, 1, huge + 1]])
    assert big.rank == 2
    assert big.lll_reduced().determinant() == big.determinant()


def test_closest_vector_is_exact_on_a_skewed_basis():
    lat = Lattice.from_basis([[1, 0], [10, 1]])
    target = [Fraction(11, 2), Fraction(1, 2)]
    v = lat.closest_vector(target)
    x, y = v.coordinates()
    assert (x - target[0]) ** 2 + (y - target[1]) ** 2 == v.norm()
    for a in range(-6, 7):
        for b in range(-3, 4):
            px, py = a + 10 * b, b
            d = (px - target[0]) ** 2 + (py - target[1]) ** 2
            assert d >= v.norm()
