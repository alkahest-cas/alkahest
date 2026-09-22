"""The ``alkahest.experimental`` theta / modular surface.

Riemann theta functions, the classical modular functions and the Weierstrass
family, returned as **balls** rather than floats.

Every expected value here is a classical normalisation checkpoint derived from
the definitions, not read off the implementation's own output:

* ``j(i) = 1728`` and ``j(rho) = 0`` for ``rho = exp(2 pi i / 3)``;
* ``eta(i) = Gamma(1/4) / (2 pi^(3/4)) = 0.7682254223...``;
* ``theta_3(0, i) = pi^(1/4) / Gamma(3/4) = 1.0864348112...``;
* Jacobi's quartic identity ``theta_3^4 = theta_2^4 + theta_4^4``.

The identity checks are asserted with ``overlaps``, which is true exactly when
the identity is consistent with both computed error bounds — so they test the
values and the radii at the same time.
"""

from __future__ import annotations

import alkahest.experimental as ex
import pytest

PREC = 256

# The whole surface needs a FLINT carrying Arb (>= 3.1) for the modular
# functions and >= 3.2 for genus-g Riemann theta.  On an older FLINT the module
# still imports and every evaluator refuses with ``E-THETA-001`` -- which is
# checked once below rather than by 30 failing tests.
HAVE_ARB = ex.arb_backend_available()
HAVE_THETA = ex.riemann_theta_available()

pytestmark = pytest.mark.skipif(
    not HAVE_ARB,
    reason="this FLINT has no Arb layer (needs >= 3.1); see ThetaError E-THETA-001",
)


@pytest.fixture
def tau_i():
    """``tau = i``, the fixed point of ``tau -> -1/tau``."""
    return ex.ComplexBall(0.0, 1.0, PREC)


# ---------------------------------------------------------------------------
# Availability and the ball type itself
# ---------------------------------------------------------------------------


def test_backend_is_available():
    assert HAVE_ARB
    assert HAVE_THETA


def test_an_exact_input_has_zero_radius(tau_i):
    assert tau_i.is_exact
    assert not tau_i.is_indeterminate
    assert tau_i.radius_real == 0.0
    assert tau_i.contains(0.0, 1.0)
    assert not tau_i.contains(0.0, 1.000001)


def test_value_refuses_when_the_enclosure_is_too_wide(tau_i):
    j = ex.j_invariant(tau_i, prec=PREC)
    assert abs(j.value(100) - 1728) < 1e-9
    with pytest.raises(ex.ThetaError) as exc:
        j.value(10_000)
    assert exc.value.code == "E-THETA-010"


def test_an_uncertain_input_widens_the_output():
    """A ball input is carried through, not discarded."""
    fuzzy = ex.ComplexBall.from_midpoint_radius(0.0, 0.0, 1.0, 1e-6, prec=PREC)
    j = ex.j_invariant(fuzzy, prec=PREC)
    assert j.contains(1728.0, 0.0)
    # A 1e-6 input radius costs most of the output accuracy, and says so.
    assert j.accuracy_bits < 40
    with pytest.raises(ex.ThetaError):
        j.value(100)


# ---------------------------------------------------------------------------
# j-invariant
# ---------------------------------------------------------------------------


def test_j_of_i_is_1728(tau_i):
    j = ex.j_invariant(tau_i, prec=PREC)
    assert j.contains(1728.0, 0.0)
    assert j.accuracy_bits > 200


def test_j_of_i_with_an_accuracy_target(tau_i):
    j = ex.j_invariant(tau_i, accurate_to=200)
    assert j.contains(1728.0, 0.0)
    assert j.accuracy_bits >= 200


def test_j_of_rho_is_zero():
    """``rho`` is built as a ball; a decimal literal would not be ``rho``."""
    rho = ex.ComplexBall.root_of_unity(3, prec=PREC)
    assert rho.midpoint_real < 0
    j = ex.j_invariant(rho, prec=PREC)
    assert j.contains_zero()
    assert j.radius_real < 1e-50


def test_accurate_to_refuses_on_an_exactly_zero_value():
    """Relative accuracy of zero is not a thing, and the refusal says so."""
    rho = ex.ComplexBall.root_of_unity(3, prec=128)
    with pytest.raises(ex.ThetaError) as exc:
        ex.j_invariant(rho, accurate_to=64)
    assert exc.value.code == "E-THETA-010"


def test_j_is_invariant_under_the_modular_group():
    tau = ex.ComplexBall(0.17, 1.31, PREC)
    j = ex.j_invariant(tau, prec=PREC)
    assert j.overlaps(ex.j_invariant(tau + 1, prec=PREC))
    assert j.overlaps(ex.j_invariant(ex.ComplexBall(-1.0, 0.0, PREC) / tau, prec=PREC))


def test_outside_the_upper_half_plane_is_refused():
    with pytest.raises(ex.ThetaError) as exc:
        ex.j_invariant(ex.ComplexBall(1.0, 0.0, 64), prec=64)
    assert exc.value.code == "E-THETA-009"


# ---------------------------------------------------------------------------
# eta, Delta, Eisenstein
# ---------------------------------------------------------------------------


def test_eta_of_i(tau_i):
    eta = ex.dedekind_eta(tau_i, prec=PREC)
    # Gamma(1/4) / (2 pi^(3/4)).
    assert abs(eta.midpoint_real - 0.7682254223260566) < 1e-15
    assert eta.imag_interval()[0] <= 0.0 <= eta.imag_interval()[1]
    assert eta.accuracy_bits > 200


def test_delta_is_eta_to_the_twenty_fourth():
    tau = ex.ComplexBall(0.25, 1.75, PREC)
    delta = ex.modular_discriminant(tau, prec=PREC)
    eta = ex.dedekind_eta(tau, prec=PREC)
    p = eta
    for _ in range(23):
        p = p * eta
    assert delta.overlaps(p)


def test_eisenstein_series_shape(tau_i):
    es = ex.eisenstein_series(tau_i, 2, prec=PREC)
    assert len(es) == 2
    # E_6 vanishes at tau = i.
    assert es[1].contains_zero()
    assert not es[0].contains_zero()
    assert ex.eisenstein_series(tau_i, 0, prec=PREC) == []


# ---------------------------------------------------------------------------
# Jacobi theta
# ---------------------------------------------------------------------------


def test_theta3_null_at_i(tau_i):
    th = ex.jacobi_theta_null(tau_i, prec=PREC)
    assert len(th) == 4
    # pi^(1/4) / Gamma(3/4).
    assert abs(th[2].midpoint_real - 1.0864348112133080) < 1e-15


def test_theta1_null_vanishes(tau_i):
    th = ex.jacobi_theta_null(tau_i, prec=PREC)
    assert th[0].contains_zero()
    assert th[0].radius_real < 1e-50


def test_jacobi_quartic_identity_holds_inside_the_radii():
    for re, im in [(0.0, 1.0), (0.3, 1.4), (-0.45, 0.6)]:
        th = ex.jacobi_theta_null(ex.ComplexBall(re, im, PREC), prec=PREC)

        def fourth(x):
            s = x * x
            return s * s

        lhs = fourth(th[2])
        rhs = fourth(th[1]) + fourth(th[3])
        assert lhs.overlaps(rhs), f"failed at tau = {re}+{im}j"
        assert lhs.accuracy_bits > 100


# ---------------------------------------------------------------------------
# Weierstrass
# ---------------------------------------------------------------------------


def test_weierstrass_differential_equation():
    tau = ex.ComplexBall(0.2, 1.3, PREC)
    z = ex.ComplexBall(0.31, 0.17, PREC)
    p = ex.weierstrass_p(z, tau, prec=PREC)
    pp = ex.weierstrass_p_prime(z, tau, prec=PREC)
    g2, g3 = ex.weierstrass_invariants(tau, prec=PREC)
    lhs = pp * pp
    rhs = p * p * p * 4.0 - g2 * p - g3
    assert lhs.overlaps(rhs)


def test_weierstrass_p_at_a_lattice_point_is_indeterminate():
    tau = ex.ComplexBall(0.0, 1.0, PREC)
    p = ex.weierstrass_p(ex.ComplexBall(0.0, 0.0, PREC), tau, prec=PREC)
    assert p.is_indeterminate
    with pytest.raises(ex.ThetaError) as exc:
        p.value(1)
    assert exc.value.code == "E-THETA-011"


def test_weierstrass_roots_sum_to_zero():
    tau = ex.ComplexBall(0.1, 1.9, PREC)
    e1, e2, e3 = ex.weierstrass_roots(tau, prec=PREC)
    assert (e1 + e2 + e3).contains_zero()


# ---------------------------------------------------------------------------
# Characteristics
# ---------------------------------------------------------------------------


def test_characteristic_index_matches_flints_example():
    # FLINT's own worked example: a = (1, 0), b = (0, 0) in genus 2 is 8.
    assert ex.theta_characteristic_index([1, 0], [0, 0]) == 8
    assert ex.theta_characteristic_bits(8, 2) == ([1, 0], [0, 0])


def test_characteristic_parity_counts():
    for g in (1, 2, 3):
        even = sum(ex.theta_characteristic_is_even(ab, g) for ab in range(4**g))
        assert even == 2 ** (g - 1) * (2**g + 1)


def test_characteristic_out_of_range():
    with pytest.raises(ex.ThetaError) as exc:
        ex.theta_characteristic_is_even(4, 1)
    assert exc.value.code == "E-THETA-008"


# ---------------------------------------------------------------------------
# Riemann theta
# ---------------------------------------------------------------------------


def test_genus_one_matches_the_classical_jacobi_route():
    """Two independent FLINT code paths must agree, sign and all.

    FLINT's genus-1 dictionary is
    ``(t1, t2, t3, t4) = (-th[3], th[2], th[0], th[1])``.
    """
    z = ex.ComplexBall(0.3, 0.1, PREC)
    tau = ex.ComplexBall(0.2, 1.3, PREC)
    classical = ex.jacobi_theta(z, tau, prec=PREC)
    sm = ex.SiegelMatrix.genus_one(tau)
    all_vals = ex.riemann_theta([z], sm, prec=PREC)
    assert len(all_vals) == 4
    assert classical[0].overlaps(-all_vals[3])
    assert classical[1].overlaps(all_vals[2])
    assert classical[2].overlaps(all_vals[0])
    assert classical[3].overlaps(all_vals[1])


def test_genus_two_diagonal_factorises():
    """``tau = diag(t1, t2)`` splits the lattice, so theta factorises."""
    t1 = ex.ComplexBall(0.0, 1.0, PREC)
    t2 = ex.ComplexBall(0.25, 1.6, PREC)
    zero = ex.ComplexBall(0.0, 0.0, PREC)
    tau = ex.SiegelMatrix.from_upper_triangle(2, [t1, zero, t2])
    vals = ex.riemann_theta([zero, zero], tau, prec=PREC)
    assert len(vals) == 16
    assert vals.genus == 2
    assert not vals.is_squared

    v1 = ex.riemann_theta([zero], ex.SiegelMatrix.genus_one(t1), prec=PREC)
    v2 = ex.riemann_theta([zero], ex.SiegelMatrix.genus_one(t2), prec=PREC)
    for a1 in (0, 1):
        for a2 in (0, 1):
            for b1 in (0, 1):
                for b2 in (0, 1):
                    ab = ex.theta_characteristic_index([a1, a2], [b1, b2])
                    i1 = ex.theta_characteristic_index([a1], [b1])
                    i2 = ex.theta_characteristic_index([a2], [b2])
                    assert vals[ab].overlaps(v1[i1] * v2[i2])


def test_odd_characteristics_vanish_at_z_zero():
    zero = ex.ComplexBall(0.0, 0.0, PREC)
    tau = ex.SiegelMatrix.from_upper_triangle(
        2,
        [
            ex.ComplexBall(0.1, 1.4, PREC),
            ex.ComplexBall(0.05, 0.3, PREC),
            ex.ComplexBall(-0.2, 1.7, PREC),
        ],
    )
    vals = ex.riemann_theta([zero, zero], tau, prec=PREC)
    for ab in range(16):
        if ex.theta_characteristic_is_even(ab, 2):
            assert not vals[ab].contains_zero()
        else:
            assert vals[ab].contains_zero()


def test_squares_match_the_squares_of_the_values():
    zero = ex.ComplexBall(0.0, 0.0, PREC)
    tau = ex.SiegelMatrix.from_upper_triangle(
        2,
        [
            ex.ComplexBall(0.0, 1.2, PREC),
            ex.ComplexBall(0.1, 0.25, PREC),
            ex.ComplexBall(0.0, 1.5, PREC),
        ],
    )
    z = [ex.ComplexBall(0.11, 0.02, PREC), ex.ComplexBall(-0.07, 0.13, PREC)]
    vals = ex.riemann_theta(z, tau, prec=PREC)
    sqrs = ex.riemann_theta_squared(z, tau, prec=PREC)
    assert sqrs.is_squared
    for ab in range(16):
        assert (vals[ab] * vals[ab]).overlaps(sqrs[ab])
    assert vals[0].overlaps(ex.riemann_theta_characteristic(z, tau, 0, prec=PREC))
    del zero


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_non_symmetric_period_matrix_is_refused():
    with pytest.raises(ex.ThetaError) as exc:
        ex.SiegelMatrix(
            2,
            [
                ex.ComplexBall(0.0, 1.0, PREC),
                ex.ComplexBall(0.1, 0.2, PREC),
                ex.ComplexBall(0.3, 0.2, PREC),
                ex.ComplexBall(0.0, 1.0, PREC),
            ],
        )
    assert exc.value.code == "E-THETA-006"


def test_outside_the_siegel_upper_half_space_is_refused():
    zero = ex.ComplexBall(0.0, 0.0, PREC)
    tau = ex.SiegelMatrix.from_upper_triangle(
        2,
        [ex.ComplexBall(0.0, 1.0, PREC), zero, ex.ComplexBall(0.0, -1.0, PREC)],
    )
    assert not tau.in_siegel_upper_half_space(PREC)
    with pytest.raises(ex.ThetaError) as exc:
        ex.riemann_theta([zero, zero], tau, prec=PREC)
    assert exc.value.code == "E-THETA-007"


def test_a_z_of_the_wrong_length_is_refused(tau_i):
    zero = ex.ComplexBall(0.0, 0.0, PREC)
    with pytest.raises(ex.ThetaError) as exc:
        ex.riemann_theta([zero, zero], ex.SiegelMatrix.genus_one(tau_i), prec=PREC)
    assert exc.value.code == "E-THETA-005"


def test_genus_and_precision_limits(tau_i):
    zero = ex.ComplexBall(0.0, 0.0, PREC)
    sm = ex.SiegelMatrix.genus_one(tau_i)
    with pytest.raises(ex.ThetaError) as exc:
        ex.riemann_theta([zero], sm, prec=1)
    assert exc.value.code == "E-THETA-003"
    with pytest.raises(ex.ThetaError) as exc:
        ex.SiegelMatrix(9, [])
    assert exc.value.code == "E-THETA-004"


# ---------------------------------------------------------------------------
# Siegel reduction
# ---------------------------------------------------------------------------


def test_siegel_reduce_is_symplectic_and_preserves_j():
    tau = ex.SiegelMatrix.genus_one(ex.ComplexBall(3.5, 0.02, PREC))
    assert not ex.siegel_is_reduced(tau, prec=PREC)
    red = ex.siegel_reduce(tau, prec=PREC)
    m = red.symplectic()
    assert len(m) == 2
    assert len(m[0]) == 2
    assert m[0][0] * m[1][1] - m[0][1] * m[1][0] == 1
    assert ex.siegel_is_reduced(red.reduced, prec=PREC)
    j0 = ex.j_invariant(tau.entry(0, 0), prec=PREC)
    j1 = ex.j_invariant(red.reduced.entry(0, 0), prec=PREC)
    assert j0.overlaps(j1)


@pytest.mark.skipif(HAVE_ARB, reason="this FLINT does carry the Arb layer")
def test_without_the_backend_everything_refuses_with_a_code():
    """The case the rest of this file cannot reach.

    On a FLINT with no Arb layer the names still exist -- ``import alkahest``
    and ``dir(alkahest.experimental)`` are unchanged -- and every evaluator
    raises ``ThetaError`` with ``E-THETA-001`` rather than ``AttributeError``.
    """
    tau = ex.ComplexBall(0.0, 1.0, 64)
    for call in (
        lambda: ex.j_invariant(tau, prec=64),
        lambda: ex.dedekind_eta(tau, prec=64),
        lambda: ex.jacobi_theta_null(tau, prec=64),
        lambda: ex.weierstrass_p(tau, tau, prec=64),
    ):
        with pytest.raises(ex.ThetaError) as exc:
            call()
        assert exc.value.code == "E-THETA-001"
    # Pure combinatorics keeps working.
    assert ex.theta_characteristic_index([1, 0], [0, 0]) == 8
