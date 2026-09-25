"""Classical linear codes, MacWilliams, Krawtchouk and the Delsarte LP bound.

The anchors are textbook codes whose parameters are rigid — Hamming [7,4,3]
and its simplex dual, the extended Hamming [8,4,4], and both binary Golay
codes — so a wrong answer cannot hide behind a plausible-looking number.

The LP assertions are one-sided on purpose in the places where the exact value
is not classical: a bound that came out *too small* would rule out a code that
exists, and that is the only unacceptable failure here.
"""

from fractions import Fraction

import alkahest as ak
import pytest
from alkahest.experimental import (
    CodingError,
    DelsarteBound,
    FiniteField,
    GfMatrix,
    LinearCode,
    WeightEnumerator,
    delsarte_lp_bound,
    hamming_bound,
    krawtchouk,
    krawtchouk_poly,
    singleton_bound,
)

GF2 = FiniteField(2)


def test_surface_is_exported():
    for name in (
        "LinearCode",
        "WeightEnumerator",
        "DelsarteBound",
        "CodingError",
        "krawtchouk",
        "krawtchouk_poly",
        "delsarte_lp_bound",
        "singleton_bound",
        "hamming_bound",
    ):
        assert name in ak.experimental.__all__
        assert hasattr(ak.experimental, name)


def test_hamming_7_4_3_from_its_parity_check():
    h = GfMatrix(
        GF2,
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
    )
    c = LinearCode.from_parity_check(h)
    assert (c.length, c.dimension, c.redundancy) == (7, 4, 3)
    assert c.size() == 16
    assert c.minimum_distance() == 3
    assert c.weight_distribution() == [1, 0, 0, 7, 7, 0, 0, 1]


def test_hamming_constructor_and_its_simplex_dual():
    c = LinearCode.hamming(GF2, 3)
    assert c.weight_distribution() == [1, 0, 0, 7, 7, 0, 0, 1]
    d = c.dual()
    assert (d.length, d.dimension) == (7, 3)
    assert d.weight_distribution() == [1, 0, 0, 0, 7, 0, 0, 0]
    assert d.minimum_distance() == 4


def test_extended_hamming_is_self_dual():
    c = LinearCode.hamming(GF2, 3).extend()
    assert (c.length, c.dimension) == (8, 4)
    assert c.weight_distribution() == [1, 0, 0, 0, 14, 0, 0, 0, 1]
    assert c.minimum_distance() == 4
    assert c.is_self_dual()
    assert c.is_self_orthogonal()


def test_binary_golay_codes():
    g23 = LinearCode.golay_binary()
    assert (g23.length, g23.dimension) == (23, 12)
    assert g23.minimum_distance() == 7
    a = g23.weight_distribution()
    assert sum(a) == 4096
    assert a[7] == 253
    assert a[8] == 506
    assert a[11] == 1288
    assert a[12] == 1288

    g24 = g23.extend()
    assert (g24.length, g24.dimension) == (24, 12)
    assert g24.minimum_distance() == 8
    b = g24.weight_distribution()
    assert b[8] == 759
    assert b[12] == 2576
    assert b[16] == 759
    assert b[24] == 1
    assert g24.is_self_dual()


def test_round_trip_through_the_generator_matrix():
    c = LinearCode.hamming(GF2, 3)
    again = LinearCode.from_generator(c.generator())
    assert again.weight_distribution() == c.weight_distribution()
    assert LinearCode(c.generator()).dimension == 4


def test_membership():
    c = LinearCode.hamming(GF2, 3)
    assert c.contains(GfMatrix(GF2, [[0] * 7]))
    assert not c.contains(GfMatrix(GF2, [[1] + [0] * 6]))


def test_macwilliams_both_directions():
    c = LinearCode.hamming(GF2, 3)
    wc = c.weight_enumerator()
    wd = c.dual().weight_enumerator()
    assert wc.macwilliams() == wd
    assert wd.macwilliams() == wc
    assert wc.macwilliams().macwilliams() == wc
    assert str(wc) == "x^7 + 7*x^4*y^3 + 7*x^3*y^4 + y^7"
    assert wc.size() == 16
    assert wc.evaluate(1, 1) == 16
    assert wc.is_dual_feasible()


def test_macwilliams_fixes_the_self_dual_golay():
    w = LinearCode.golay_binary().extend().weight_enumerator()
    assert w.macwilliams() == w


def test_weight_enumerator_validates_its_input():
    with pytest.raises(CodingError) as exc:
        WeightEnumerator(2, [2, 0, 0])
    assert exc.value.code == "E-CODE-006"

    with pytest.raises(CodingError) as exc:
        WeightEnumerator(2, [1, 1, 1]).macwilliams()
    assert exc.value.code == "E-CODE-006"


def test_krawtchouk_identities():
    # K_k(0) = C(n, k) (q-1)^k
    from math import comb

    for n, q in [(7, 2), (9, 3), (6, 4)]:
        for k in range(n + 1):
            assert krawtchouk(k, 0, n, q) == comb(n, k) * (q - 1) ** k
        for x in range(n + 1):
            assert krawtchouk(0, x, n, q) == 1
            assert krawtchouk(1, x, n, q) == (q - 1) * n - q * x


def test_krawtchouk_polynomial_matches_pointwise_values():
    for n, q in [(7, 2), (5, 3)]:
        for k in range(n + 1):
            coeffs = krawtchouk_poly(k, n, q)
            assert len(coeffs) == k + 1
            for x in range(-2, n + 3):
                value = sum(Fraction(c) * x**i for i, c in enumerate(coeffs))
                assert value == krawtchouk(k, x, n, q)


def test_delsarte_bound_on_the_perfect_codes():
    b = delsarte_lp_bound(7, 3, 2)
    assert isinstance(b, DelsarteBound)
    assert b.bound == 16
    assert b.optimum() == 16
    assert b.verify_certificate()
    assert b.verify_distribution()
    assert delsarte_lp_bound(8, 4, 2).bound == 16


def test_delsarte_bound_is_tight_for_the_golay_codes():
    assert delsarte_lp_bound(23, 7, 2).bound == 4096
    b = delsarte_lp_bound(24, 8, 2)
    assert b.bound == 4096
    assert b.verify_certificate()
    assert LinearCode.golay_binary().extend().size() == 4096


def test_delsarte_optimum_is_exact_and_fractional_when_it_should_be():
    b = delsarte_lp_bound(11, 3, 2)
    assert b.optimum() == Fraction(512, 3)
    assert isinstance(b.optimum(), Fraction)
    assert b.bound == 170
    # It coincides with the sphere-packing bound here.
    assert hamming_bound(11, 3, 2) == 170


def test_the_certificate_is_a_self_contained_proof():
    b = delsarte_lp_bound(7, 3, 2)
    y = b.certificate()
    assert len(y) == 7
    assert all(v >= 0 for v in y)
    # Σ_k y_k K_k(i) ≤ −1 for d ≤ i ≤ n — re-derived here, not taken on trust.
    for i in range(b.distance, b.length + 1):
        acc = sum(
            Fraction(y[k - 1]) * krawtchouk(k, i, b.length, b.alphabet_size)
            for k in range(1, b.length + 1)
        )
        assert acc <= -1
    value = 1 + sum(
        Fraction(y[k - 1]) * krawtchouk(k, 0, b.length, b.alphabet_size)
        for k in range(1, b.length + 1)
    )
    assert value == b.optimum()


def test_delsarte_never_beats_singleton_or_hamming_and_never_rules_out_a_code():
    for n in range(2, 11):
        for d in range(1, n + 1):
            for q in (2, 3):
                b = delsarte_lp_bound(n, d, q)
                assert b.bound <= singleton_bound(n, d, q)
                assert b.bound <= hamming_bound(n, d, q)
                assert b.verify_certificate()

    for code in (
        LinearCode.hamming(GF2, 3),
        LinearCode.hamming(GF2, 3).extend(),
        LinearCode.repetition(GF2, 7),
        LinearCode.golay_binary(),
    ):
        d = code.minimum_distance()
        assert delsarte_lp_bound(code.length, d, 2).bound >= code.size()


def test_enumeration_past_the_cap_refuses_rather_than_truncating():
    # A [64, 40] code: 2**40 codewords.
    rows = [[0] * 64 for _ in range(40)]
    for i in range(40):
        rows[i][i] = 1
    g = GfMatrix(GF2, rows)
    c = LinearCode.from_generator(g)
    with pytest.raises(CodingError) as exc:
        c.weight_distribution()
    assert exc.value.code == "E-CODE-004"
    with pytest.raises(CodingError) as exc:
        c.minimum_distance()
    assert exc.value.code == "E-CODE-004"


def test_degenerate_arguments_carry_stable_codes():
    for args, code in [
        ((0, 1, 2), "E-CODE-001"),
        ((7, 0, 2), "E-CODE-002"),
        ((7, 8, 2), "E-CODE-002"),
        ((7, 3, 1), "E-CODE-008"),
        ((1000, 3, 2), "E-CODE-005"),
    ]:
        with pytest.raises(CodingError) as exc:
            delsarte_lp_bound(*args)
        assert exc.value.code == code, args


def test_the_zero_code_has_no_minimum_distance_and_says_so():
    whole = LinearCode.from_generator(GfMatrix.identity(GF2, 5))
    zero = whole.dual()
    assert zero.dimension == 0
    assert zero.minimum_distance() is None
    assert whole.minimum_distance() == 1


def test_non_binary_codes():
    gf3 = FiniteField(3)
    c = LinearCode.hamming(gf3, 2)
    assert (c.length, c.dimension) == (4, 2)
    assert c.size() == 9
    assert c.weight_distribution() == [1, 0, 0, 8, 0]

    gf4 = FiniteField(2, 2)
    r = LinearCode.repetition(gf4, 3)
    assert r.size() == 4
    assert r.weight_distribution() == [1, 0, 0, 3]
    assert r.weight_enumerator().macwilliams() == r.dual().weight_enumerator()


def test_repr_is_informative():
    assert repr(LinearCode.hamming(GF2, 3)) == "LinearCode(n=7, k=4, q=2)"
    assert repr(delsarte_lp_bound(7, 3, 2)) == "DelsarteBound(n=7, d=3, q=2, bound=16)"
