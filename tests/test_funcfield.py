"""The ``alkahest.experimental`` function-field surface.

Divisors, the divisor class group ``Pic⁰``, torsion order and Riemann–Roch on
imaginary hyperelliptic curves.  Every expected value here is derived by hand
from the definitions (the derivation is in the test's docstring or an inline
comment), never read off the implementation's own output.

The refusal tests matter at least as much as the value tests: the implemented
class is narrow on purpose, and a boundary that silently returns something
plausible is the failure mode this module exists to avoid.
"""

from __future__ import annotations

from fractions import Fraction

import alkahest.experimental as ex
import pytest

# ---------------------------------------------------------------------------
# Fixtures: the three curves used throughout
# ---------------------------------------------------------------------------


@pytest.fixture
def e1():
    """``y² = x³ − x``, genus 1.  E(ℚ) = {∞, (0,0), (±1,0)} ≅ (ℤ/2)²."""
    return ex.FunctionField.hyperelliptic([0, -1, 0, 1])


@pytest.fixture
def e_plus1():
    """``y² = x³ + 1``, genus 1, with E(ℚ)_tors ≅ ℤ/6 (Cremona 36a1)."""
    return ex.FunctionField.hyperelliptic([1, 0, 0, 1])


@pytest.fixture
def c2():
    """``y² = x⁵ + 1``, genus 2."""
    return ex.FunctionField.hyperelliptic([1, 0, 0, 0, 0, 1])


def _divisor_or_none(u):
    """``div(u)``, or ``None`` when its support is not representable.

    A basis element of ``L(D)`` can legitimately have zeros at a place of
    degree > 1 even though ``u`` is in the space; that is a limitation of the
    *representation* (``E-FFLD-003``), not of the space, so the caller skips it
    rather than weakening the check where it does apply.
    """
    try:
        return u.divisor()
    except ex.FunctionFieldError as exc:
        if exc.code != "E-FFLD-003":
            raise
        return None


def inf_divisor(field, n):
    return ex.Divisor(field, [(ex.Place.infinity(), n)])


def point_class_divisor(field, x, y):
    """``(x, y) − ∞``, a degree-zero divisor."""
    return ex.Divisor(field, [(ex.Place.finite(x, y), 1), (ex.Place.infinity(), -1)])


# ---------------------------------------------------------------------------
# Genus, and which models are accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "genus", "imaginary"),
    [
        ([0, 1], 0, True),  # y² = x,        deg 1 = 2·0+1
        ([1, 0, 1], 0, False),  # y² = x²+1,     deg 2 = 2·0+2
        ([0, -1, 0, 1], 1, True),  # y² = x³−x,     deg 3 = 2·1+1
        ([1, 0, 0, 0, 1], 1, False),  # y² = x⁴+1,     deg 4 = 2·1+2
        ([1, 0, 0, 0, 0, 1], 2, True),  # y² = x⁵+1,     deg 5 = 2·2+1
        ([1, 1, 0, 0, 0, 0, 1], 2, False),  # y² = x⁶+x+1,   deg 6 = 2·2+2
        ([1, 0, 0, 0, 0, 0, 0, 1], 3, True),  # y² = x⁷+1,     deg 7 = 2·3+1
    ],
)
def test_genus_of_squarefree_hyperelliptic_models(a, genus, imaginary):
    """``g`` is ``⌊(deg a − 1)/2⌋`` for squarefree ``a``, odd degree or even."""
    f = ex.FunctionField.hyperelliptic(a)
    assert f.genus == genus
    assert f.is_imaginary is imaginary
    assert f.curve_degree == len(a) - 1


def test_genus_is_reported_even_where_divisors_are_refused():
    """The even-degree model has no representable place at infinity, but the
    genus does not depend on the model, so refusing it would be gratuitous."""
    f = ex.FunctionField.hyperelliptic([1, 0, 0, 0, 1])
    assert f.genus == 1
    with pytest.raises(ex.FunctionFieldError) as exc:
        f.canonical_divisor()
    assert exc.value.code == "E-FFLD-002"


def test_general_quadratic_is_normalised_by_completing_the_square():
    """``y² + 2xy + (x² − x³) = 0`` is ``(2y + 2x)² = 4x³``; peeling ``x²`` and
    the square constant 4 leaves ``Z² = x``, a genus-0 field."""
    f = ex.FunctionField([[0, 0, 1, -1], [0, 2], [1]])
    assert f.curve() == [0, 1]
    assert f.genus == 0
    assert not f.is_normalised
    assert "Y =" in f.normalisation()


def test_a_square_scalar_multiple_is_the_same_field(e1):
    """``2y² = 2x³ − 2x`` is ``y² = x³ − x`` after ``Y = y/4``; the class group
    and everything else must agree."""
    same = ex.FunctionField([[0, 2, 0, -2], [], [2]])
    assert same == e1
    assert same.curve() == e1.curve()


def test_a_non_square_scalar_is_a_genuine_twist(e1):
    """``y² = 2(x³ − x)`` is a quadratic twist, not the same curve, and must
    not be normalised into it."""
    twist = ex.FunctionField.hyperelliptic([0, -2, 0, 2])
    assert twist != e1
    assert twist.curve() == [0, -2, 0, 2]
    assert twist.genus == 1


@pytest.mark.parametrize(
    ("coeffs", "why"),
    [
        ([[1, 0, 0, 0, -1], [], [], [1]], "y³ = x⁴ − 1 is not quadratic in y"),
        ([[1, 0, 0, -1], [], [0, 1]], "x·y² has a non-constant leading coefficient"),
        ([[0, 0, 1], [0, 2], [1]], "(y + x)² is a perfect square"),
    ],
)
def test_unsupported_models_are_refused_not_guessed(coeffs, why):
    with pytest.raises(ex.FunctionFieldError) as exc:
        ex.FunctionField(coeffs)
    assert exc.value.code == "E-FFLD-001", why


def test_a_constant_discriminant_is_not_a_function_field():
    """``y² = 2`` defines ℚ(√2)(x), a number field over ℚ(x), not a curve."""
    with pytest.raises(ex.FunctionFieldError) as exc:
        ex.FunctionField.hyperelliptic([2])
    assert exc.value.code == "E-FFLD-001"


# ---------------------------------------------------------------------------
# Places and divisors
# ---------------------------------------------------------------------------


def test_place_accessors(e1):
    p = ex.Place.finite(0, 0)
    assert p.x == 0
    assert p.y == 0
    assert p.degree == 1
    assert p.is_ramified  # a(0) = 0
    assert not p.is_infinite
    q = ex.Place.infinity()
    assert q.is_infinite
    assert q.is_ramified
    assert q.x is None
    assert q.y is None


def test_exact_rational_coordinates_round_trip():
    """Coordinates are exact.  ``y² = x³ + 1`` at ``x = −1/4`` is not rational,
    so use a curve where it is: ``y² = x`` at ``x = 9/4`` gives ``y = 3/2``."""
    f = ex.FunctionField.hyperelliptic([0, 1])
    p = ex.Place.finite(Fraction(9, 4), "3/2")
    assert p.x == Fraction(9, 4)
    assert p.y == Fraction(3, 2)
    d = ex.Divisor(f, [(p, 1), (ex.Place.infinity(), -1)])
    assert d.degree == 0


def test_floats_are_refused_rather_than_rounded():
    """``0.1`` is not ``1/10``; a place that lands on the curve by rounding is
    exactly the silent wrong answer this surface refuses."""
    with pytest.raises(TypeError, match="exact"):
        ex.Place.finite(0.5, 0.0)


def test_place_involution(e_plus1):
    p = ex.Place.finite(2, 3)
    assert p.involution() == ex.Place.finite(2, -3)
    # Branch points and infinity are fixed.
    assert ex.Place.finite(-1, 0).involution() == ex.Place.finite(-1, 0)
    assert ex.Place.infinity().involution() == ex.Place.infinity()


def test_divisor_arithmetic_and_degree(e1):
    p0 = ex.Place.finite(0, 0)
    p1 = ex.Place.finite(1, 0)
    d = ex.Divisor(e1, [(p0, 3), (p1, -1)])
    e = ex.Divisor(e1, [(p1, 4), (ex.Place.infinity(), 2)])
    assert d.degree == 2
    assert e.degree == 6
    assert (d + e).degree == 8
    assert (d - e).degree == -4
    assert (-d).degree == -2
    assert (3 * d).degree == 6
    assert (d * 0).is_zero
    assert d.coefficient(p0) == 3
    assert len(d) == 2
    assert set(d.support()) == {p0, p1}


def test_divisor_partial_order(e1):
    p0 = ex.Place.finite(0, 0)
    p1 = ex.Place.finite(1, 0)
    a = ex.Divisor(e1, [(p0, 1)])
    b = ex.Divisor(e1, [(p1, 1)])
    both = a + b
    assert a <= both
    assert a < both
    assert both >= b
    assert both > b
    # Distinct single places are *incomparable*: every comparison is False.
    for comparison in (a < b, a > b, a <= b, a >= b, b < a, b > a):
        assert not comparison
    assert a != b


def test_a_place_off_the_curve_is_refused(e1):
    """``2³ − 2 = 6`` is not a rational square, so no rational place lies over
    ``x = 2``."""
    with pytest.raises(ex.FunctionFieldError) as exc:
        ex.Divisor(e1, [(ex.Place.finite(2, 0), 1)])
    assert exc.value.code == "E-FFLD-004"


def test_divisors_from_different_curves_do_not_mix(e1, c2):
    a = ex.Divisor.zero(e1)
    b = ex.Divisor.zero(c2)
    with pytest.raises(ex.FunctionFieldError) as exc:
        a + b
    assert exc.value.code == "E-FFLD-009"


def test_real_models_have_no_divisors():
    f = ex.FunctionField.hyperelliptic([1, 0, 0, 0, 1])
    with pytest.raises(ex.FunctionFieldError) as exc:
        ex.Divisor.zero(f)
    assert exc.value.code == "E-FFLD-002"


# ---------------------------------------------------------------------------
# The divisor of a function
# ---------------------------------------------------------------------------


def test_divisor_of_x_at_a_branch_point(e1):
    """``x = 0`` is a branch point of ``y² = x³ − x``, so ``v_P(x) = 2`` and
    ``div(x) = 2·(0,0) − 2·∞``."""
    d = ex.FunctionFieldElement.x(e1).divisor()
    assert d.coefficient(ex.Place.finite(0, 0)) == 2
    assert d.coefficient(ex.Place.infinity()) == -2
    assert d.degree == 0


def test_divisor_of_y_is_the_branch_locus(e1):
    """``div(y) = (−1,0) + (0,0) + (1,0) − 3·∞``, since ``x³ − x`` splits."""
    d = ex.FunctionFieldElement.y(e1).divisor()
    for x in (-1, 0, 1):
        assert d.coefficient(ex.Place.finite(x, 0)) == 1
    assert d.coefficient(ex.Place.infinity()) == -3
    assert d.degree == 0


def test_divisor_of_a_quotient(e1):
    """``div(y/x) = (−1,0) − (0,0) + (1,0) − ∞``."""
    u = ex.FunctionFieldElement(e1, [0], [1], [0, 1])
    d = u.divisor()
    assert d.coefficient(ex.Place.finite(-1, 0)) == 1
    assert d.coefficient(ex.Place.finite(0, 0)) == -1
    assert d.coefficient(ex.Place.finite(1, 0)) == 1
    assert d.coefficient(ex.Place.infinity()) == -1
    assert d.degree == 0


def test_div_is_additive_on_products(e1):
    u = ex.FunctionFieldElement.x(e1)
    v = ex.FunctionFieldElement.y(e1)
    assert (u * v).divisor() == u.divisor() + v.divisor()


def test_divisor_of_a_constant_is_zero(e1):
    assert ex.FunctionFieldElement(e1, ["7/3"]).divisor().is_zero


def test_the_zero_function_has_no_divisor(e1):
    with pytest.raises(ex.FunctionFieldError) as exc:
        ex.FunctionFieldElement(e1, [0]).divisor()
    assert exc.value.code == "E-FFLD-008"


def test_div_refuses_when_the_support_is_not_rational(c2):
    """``x⁵ + 1 = (x+1)·Φ₁₀(x)`` and ``Φ₁₀`` is irreducible over ℚ, so four of
    the five branch points have degree > 1.  Reporting only ``(−1,0) − 5·∞``
    would be a 'divisor' of degree −4 for a function."""
    with pytest.raises(ex.FunctionFieldError) as exc:
        ex.FunctionFieldElement.y(c2).divisor()
    assert exc.value.code == "E-FFLD-003"


# ---------------------------------------------------------------------------
# The divisor class group
# ---------------------------------------------------------------------------


def test_the_zero_divisor_is_the_identity_class(e1):
    c = ex.Divisor.zero(e1).divisor_class()
    assert c.is_identity
    assert c == ex.DivisorClass.identity(e1)
    assert c.weight == 0


def test_a_class_plus_its_inverse_is_the_identity(e1, e_plus1, c2):
    for f, x, y in [(e1, 0, 0), (e_plus1, 2, 3), (c2, 0, 1)]:
        c = point_class_divisor(f, x, y).divisor_class()
        assert (c + (-c)).is_identity
        assert (c - c).is_identity


@pytest.mark.parametrize("x", [-1, 0, 1])
def test_branch_points_are_two_torsion(e1, x):
    """``2((α,0) − ∞) = div(x − α)`` for a branch point, so the class has order
    exactly 2 (it is non-trivial because ``(α,0) ≠ ∞``)."""
    d = point_class_divisor(e1, x, 0)
    c = d.divisor_class()
    assert not c.is_identity
    assert c.order() == 2
    assert e1.is_principal(2 * d)
    assert not e1.is_principal(d)


@pytest.mark.parametrize(
    ("x", "y", "order"),
    [
        (0, 1, 3),  # 2P = (0,−1) = −P by the duplication formula ⇒ order 3
        (0, -1, 3),
        (2, 3, 6),  # Cremona 36a1: E(ℚ)_tors ≅ ℤ/6, generated by (2, ±3)
        (2, -3, 6),
        (-1, 0, 2),  # a branch point
    ],
)
def test_torsion_orders_on_y2_equals_x3_plus_1(e_plus1, x, y, order):
    c = point_class_divisor(e_plus1, x, y).divisor_class()
    assert c.order() == order
    assert (order * c).is_identity
    for k in range(1, order):
        assert not (k * c).is_identity, f"{k} already kills a class of claimed order {order}"


def test_n_delta_principal_iff_the_order_divides_n(e_plus1):
    d = point_class_divisor(e_plus1, 0, 1)
    order = d.divisor_class().order()
    assert order == 3
    for n in range(1, 13):
        assert e_plus1.is_principal(n * d) == (n % order == 0)


def test_non_torsion_is_a_verdict_not_a_refusal():
    """``y² = x³ − 2`` has rank 1 with generator ``(3, 5)`` and trivial torsion
    (Mordell curve k = −2; Silverman AEC X.§6), so the class is of infinite
    order.  ``E-FFLD-007`` is a verdict: a wider search will not change it."""
    f = ex.FunctionField.hyperelliptic([-2, 0, 0, 1])
    assert f.contains_point(3, 5)
    c = point_class_divisor(f, 3, 5).divisor_class()
    with pytest.raises(ex.FunctionFieldError) as exc:
        c.order()
    assert exc.value.code == "E-FFLD-007"
    assert c.is_torsion() is False


def test_the_class_map_is_a_homomorphism(e_plus1):
    d1 = point_class_divisor(e_plus1, 0, 1)
    d2 = point_class_divisor(e_plus1, 2, 3)
    assert (d1 + d2).divisor_class() == d1.divisor_class() + d2.divisor_class()
    assert (-d1).divisor_class() == -d1.divisor_class()
    for k in (-3, -1, 0, 1, 2, 5):
        assert (k * d1).divisor_class() == k * d1.divisor_class()


def test_class_reduction_is_idempotent(e_plus1):
    d = 7 * point_class_divisor(e_plus1, 2, 3)
    c = d.divisor_class()
    rep = c.reduced_divisor()
    assert rep.degree == 0
    assert rep.divisor_class() == c
    assert rep.divisor_class().reduced_divisor() == rep
    assert c.weight <= e_plus1.genus


def test_a_nonzero_degree_divisor_has_no_class(e1):
    d = ex.Divisor(e1, [(ex.Place.finite(0, 0), 1)])
    with pytest.raises(ex.FunctionFieldError) as exc:
        d.divisor_class()
    assert exc.value.code == "E-FFLD-005"


def test_mumford_representation_is_exposed(e1):
    """``(0,0) − ∞`` has Mumford pair ``u = x``, ``v = 0``."""
    c = point_class_divisor(e1, 0, 0).divisor_class()
    assert c.mumford_u() == [0, 1]
    assert c.mumford_v() == []
    assert c.weight == 1


# ---------------------------------------------------------------------------
# Riemann–Roch
# ---------------------------------------------------------------------------


def test_dim_l_of_zero_is_one(e1, c2):
    """``L(0)`` is the constants on any projective curve."""
    for f in (e1, c2):
        s = ex.riemann_roch(ex.Divisor.zero(f))
        assert s.dimension == 1
        assert len(s.basis()) == 1


def test_canonical_divisor_degree_and_dimension(e1, c2):
    """``deg K = 2g − 2`` and ``dim L(K) = g`` — the defining properties."""
    for f in (e1, c2):
        g = f.genus
        k = f.canonical_divisor()
        assert k.degree == 2 * g - 2
        assert ex.riemann_roch(k).dimension == g


def test_genus_zero_canonical_divisor_has_no_sections():
    f = ex.FunctionField.hyperelliptic([0, 1])  # y² = x, g = 0
    k = f.canonical_divisor()
    assert k.degree == -2
    assert ex.riemann_roch(k).dimension == 0


@pytest.mark.parametrize("a", [[0, -1, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1]])
def test_riemann_roch_equality_above_the_canonical_degree(a):
    """``dim L(D) = deg D + 1 − g`` whenever ``deg D > 2g − 2``."""
    f = ex.FunctionField.hyperelliptic(a)
    g = f.genus
    for n in range(2 * g - 1, 2 * g + 8):
        assert ex.riemann_roch(inf_divisor(f, n)).dimension == n + 1 - g


@pytest.mark.parametrize("n", [-1, -2, -5])
def test_negative_degree_divisors_have_no_sections(e1, n):
    assert ex.riemann_roch(inf_divisor(e1, n)).dimension == 0


def test_elliptic_basis_of_l_3_infinity(e1):
    """``L(3·∞)`` on an elliptic curve is spanned by ``1, x, y`` — and the
    basis returned really is that, checked by recomputing each divisor."""
    d = inf_divisor(e1, 3)
    s = ex.riemann_roch(d)
    assert s.dimension == 3
    assert len(s) == 3
    divisors = []
    for u in s.basis():
        du = u.divisor()
        assert (du + d).is_effective
        divisors.append(du)
    # Exactly one basis element involves y (its divisor has odd pole order).
    odd_pole = [dd for dd in divisors if int(dd.coefficient(ex.Place.infinity())) % 2 != 0]
    assert len(odd_pole) == 1


def test_riemann_roch_basis_lies_in_the_space(e1, c2):
    """The strongest available check: recompute ``div(u)`` independently and
    confirm ``div(u) + D ≥ 0``."""
    cases = [
        (e1, [(ex.Place.infinity(), 5)]),
        (e1, [(ex.Place.finite(0, 0), 2)]),
        (e1, [(ex.Place.infinity(), 6), (ex.Place.finite(1, 0), -2)]),
        (c2, [(ex.Place.infinity(), 7)]),
    ]
    for f, terms in cases:
        d = ex.Divisor(f, terms)
        s = ex.riemann_roch(d)
        assert s.dimension > 0
        for u in s.basis():
            # A basis element's own divisor can legitimately be supported at a
            # place of degree > 1 (E-FFLD-003) even though u ∈ L(D); that is a
            # limitation of the *representation*, not of the space, so skip it
            # rather than weaken the check where it does apply.
            du = _divisor_or_none(u)
            if du is None:
                continue
            assert (du + d).is_effective


def test_dimension_is_a_linear_equivalence_invariant(e1):
    """``dim L(D)`` depends only on the class, so adding ``div(x)`` — a
    principal divisor — must not change it."""
    dx = ex.FunctionFieldElement.x(e1).divisor()
    for n in range(-2, 7):
        d = inf_divisor(e1, n)
        assert ex.riemann_roch(d).dimension == ex.riemann_roch(d + dx).dimension


def test_riemann_roch_below_the_canonical_degree_is_not_the_formula(c2):
    """On genus 2, ``deg D = 0`` and ``deg D = 2`` sit at or below ``2g − 2``,
    where the formula is a strict inequality.  ``dim L(0) = 1 ≠ −1`` and
    ``dim L(K) = 2 ≠ 1``."""
    assert ex.riemann_roch(ex.Divisor.zero(c2)).dimension == 1
    assert ex.riemann_roch(c2.canonical_divisor()).dimension == 2
    # Riemann's inequality, which does always hold.
    for n in range(0, 3):
        d = inf_divisor(c2, n)
        assert ex.riemann_roch(d).dimension >= max(0, n + 1 - c2.genus)


def test_riemann_roch_space_reports_its_divisor(e1):
    d = inf_divisor(e1, 4)
    s = ex.riemann_roch(d)
    assert s.divisor() == d
    assert "dim=" in repr(s)
