//! Tests for the theta / modular surface.
//!
//! The classical checkpoints (`j(i) = 1728`, `j(rho) = 0`,
//! `eta(i) = Gamma(1/4) / (2 pi^{3/4})`, `theta_3(0, i) = pi^{1/4} / Gamma(3/4)`)
//! are here because they catch normalisation errors that nothing else does: a
//! wrapper can be wired to the wrong FLINT entry point, or to the right one
//! with the wrong convention, and still produce plausible numbers everywhere
//! except at these points.
//!
//! Where a reference value is irrational it is built as a ball out of `pi` and
//! `Gamma`, and the assertion is that the two balls **overlap** — which tests
//! the value and the error bound at the same time. A decimal literal would
//! test only the value, and only to the digits typed.

use super::*;

const P: u32 = 300;

fn bits(n: u32) -> Precision {
    Precision::Bits(n)
}

/// `tau = i`.
fn tau_i() -> ComplexBall {
    ComplexBall::exact_i64(0, 1, P)
}

// ---------------------------------------------------------------------------
// FFI foundations
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn arb_backend_is_available_and_abi_agrees() {
    assert!(
        arb_backend_available(),
        "the Arb layer must be usable in this build"
    );
    assert!(
        riemann_theta_available(),
        "acb_theta must be usable in this build"
    );
}

#[test]
fn exact_input_balls_have_zero_radius() {
    let t = tau_i();
    assert!(t.is_exact());
    assert_eq!(t.accuracy_bits(), i64::MAX);
    assert!(t.contains_f64(0.0, 1.0));
    assert!(!t.contains_f64(0.0, 1.0000001));
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn ball_round_trips_through_flint_without_widening() {
    // Exact dyadic input, an exact operation: the radius must still be zero
    // coming back. Anything else means the mantissa/exponent interchange is
    // rounding, and a rounding conversion would quietly inflate every result.
    let a = ComplexBall::exact_f64(1.5, -0.25, 128);
    let b = ComplexBall::exact_f64(0.5, 0.75, 128);
    let s = a.add(&b, 128).unwrap();
    assert!(s.is_exact(), "exact + exact must stay exact: {s}");
    assert!(s.contains_f64(2.0, 0.5));
    let p = a.mul(&b, 128).unwrap();
    // (1.5 - 0.25i)(0.5 + 0.75i) = 0.75 + 1.125i + 0.1875 - 0.125i
    assert!(p.is_exact(), "exact * exact must stay exact: {p}");
    assert!(p.contains_f64(0.9375, 1.0));
}

#[test]
fn indeterminate_balls_are_reported_as_such() {
    let x = ComplexBall::indeterminate(64);
    assert!(x.is_indeterminate());
    assert_eq!(x.accuracy_bits(), i64::MIN);
    let err = x.value_if_accurate("test", 1).unwrap_err();
    assert!(matches!(err, ThetaError::Indeterminate { .. }));
}

// ---------------------------------------------------------------------------
// j-invariant — the normalisation checkpoints
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn j_of_i_is_1728() {
    let j = j_invariant(&tau_i(), bits(P)).unwrap();
    assert!(j.contains_f64(1728.0, 0.0), "j(i) = {j}");
    assert!(
        j.accuracy_bits() > 200,
        "j(i) should be sharp, got {} bits",
        j.accuracy_bits()
    );
    // And the accuracy-driven mode reaches it.
    let j2 = j_invariant(&tau_i(), Precision::AccurateTo(200)).unwrap();
    assert!(j2.contains_f64(1728.0, 0.0));
    assert!(j2.accuracy_bits() >= 200);
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn j_of_rho_is_zero() {
    // rho = exp(2 pi i / 3), built as a ball rather than as a decimal literal:
    // the whole point of this checkpoint is that j vanishes *at* rho.
    let rho = ComplexBall::root_of_unity(3, P).unwrap();
    assert!(
        rho.real().midpoint_f64() < 0.0,
        "rho should be in Q2: {rho}"
    );
    let j = j_invariant(&rho, bits(P)).unwrap();
    assert!(
        j.contains_zero(),
        "j(rho) must enclose 0, got {j} (radius {:e})",
        j.radius_re().to_f64()
    );
    // The enclosure must also be *tight*: containing zero inside a huge ball
    // would prove nothing.
    assert!(
        j.radius_re().to_f64() < 1e-60,
        "j(rho) enclosure is too loose: {j}"
    );
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn accurate_to_refuses_on_an_exactly_zero_value() {
    // j(rho) = 0 has no relative accuracy at any precision. The refusal is the
    // documented behaviour, and it must be the typed one rather than a hang or
    // a midpoint with nothing behind it.
    let rho = ComplexBall::root_of_unity(3, 128).unwrap();
    let err = j_invariant(&rho, Precision::AccurateTo(64)).unwrap_err();
    match err {
        ThetaError::InsufficientAccuracy {
            requested_bits,
            at_precision,
            ..
        } => {
            assert_eq!(requested_bits, 64);
            assert!(at_precision >= 64);
        }
        other => panic!("expected InsufficientAccuracy, got {other}"),
    }
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn modular_functions_refuse_outside_the_upper_half_plane() {
    let real_tau = ComplexBall::exact_i64(1, 0, 64);
    let err = j_invariant(&real_tau, bits(64)).unwrap_err();
    assert!(matches!(err, ThetaError::NotInUpperHalfPlane { .. }));
    let lower = ComplexBall::exact_i64(0, -1, 64);
    let err = dedekind_eta(&lower, bits(64)).unwrap_err();
    assert!(matches!(err, ThetaError::NotInUpperHalfPlane { .. }));
}

// ---------------------------------------------------------------------------
// Dedekind eta
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn eta_of_i_is_gamma_quarter_over_two_pi_three_quarters() {
    let eta = dedekind_eta(&tau_i(), bits(P)).unwrap();
    // Gamma(1/4) / (2 * pi^{3/4}), as a ball.
    let quarter = RealBall::exact_i64(1, P)
        .div(&RealBall::exact_i64(4, P), P)
        .unwrap();
    let g = quarter.gamma(P).unwrap();
    let pi = RealBall::pi(P).unwrap();
    let three_quarters = RealBall::exact_i64(3, P)
        .div(&RealBall::exact_i64(4, P), P)
        .unwrap();
    let pi_pow = pi.pow(&three_quarters, P).unwrap();
    let denom = pi_pow.mul(&RealBall::exact_i64(2, P), P).unwrap();
    let reference = g.div(&denom, P).unwrap();

    assert!(
        eta.real().overlaps(&reference),
        "eta(i) = {eta} does not overlap Gamma(1/4)/(2 pi^(3/4)) = {reference}"
    );
    assert!(eta.imag().contains_zero(), "eta(i) must be real, got {eta}");
    // The enclosure is far tighter than an `f64` decimal literal, so the
    // literal is *outside* it — which is the right way round. Compare the
    // rounded midpoint instead; the rigorous statement is the overlap above.
    assert!((eta.real().midpoint_f64() - 0.768_225_422_326_056_6).abs() < 1e-15);
    assert!(eta.accuracy_bits() > 200);
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn delta_is_eta_to_the_twenty_fourth() {
    // Delta(tau) = eta(tau)^24 — an identity between two independent FLINT
    // entry points, asserted inside the computed radii.
    let tau = ComplexBall::exact_f64(0.25, 1.75, P);
    let delta = modular_discriminant(&tau, bits(P)).unwrap();
    let eta = dedekind_eta(&tau, bits(P)).unwrap();
    let mut p = eta.clone();
    for _ in 0..23 {
        p = p.mul(&eta, P).unwrap();
    }
    assert!(
        delta.overlaps(&p),
        "Delta = {delta} does not overlap eta^24 = {p}"
    );
}

// ---------------------------------------------------------------------------
// Jacobi theta functions
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn theta3_null_at_i_is_pi_fourth_root_over_gamma_three_quarters() {
    let th = jacobi_theta_null(&tau_i(), bits(P)).unwrap();
    let theta3 = &th[2];
    let pi = RealBall::pi(P).unwrap();
    let quarter = RealBall::exact_i64(1, P)
        .div(&RealBall::exact_i64(4, P), P)
        .unwrap();
    let num = pi.pow(&quarter, P).unwrap();
    let three_quarters = RealBall::exact_i64(3, P)
        .div(&RealBall::exact_i64(4, P), P)
        .unwrap();
    let den = three_quarters.gamma(P).unwrap();
    let reference = num.div(&den, P).unwrap();
    assert!(
        theta3.real().overlaps(&reference),
        "theta_3(0, i) = {theta3} does not overlap pi^(1/4)/Gamma(3/4) = {reference}"
    );
    // As in the eta test: compare the rounded midpoint, because the ball is
    // narrower than the `f64` literal is accurate.
    assert!((theta3.real().midpoint_f64() - 1.086_434_811_213_308).abs() < 1e-15);
    assert!(theta3.imag().contains_zero());
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn theta1_null_is_identically_zero() {
    let th = jacobi_theta_null(&ComplexBall::exact_f64(0.3, 1.1, P), bits(P)).unwrap();
    assert!(th[0].contains_zero(), "theta_1(0, tau) = {}", th[0]);
    assert!(
        th[0].radius_re().to_f64() < 1e-60,
        "theta_1(0, tau) enclosure too loose: {}",
        th[0]
    );
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn theta3_tends_to_one_as_tau_goes_up_the_imaginary_axis() {
    // q = exp(pi i tau) -> 0, and theta_3(0, tau) = 1 + 2 sum q^{n^2} -> 1.
    // Stands in for the "theta_3(0, 0) = 1" checkpoint, which is stated at
    // q = 0 — a point that is not in the upper half-plane and is refused here.
    //
    // How far up the axis matters, and is a good illustration of what a ball
    // buys you: at tau = 40i the difference from 1 is 2 exp(-40 pi) ~ 7e-55,
    // and at 300 bits the enclosure is narrow enough to *exclude* 1 — correctly,
    // because theta_3(0, 40i) is not 1. At tau = 100i the difference is ~1e-136,
    // comfortably inside the radius, so the ball does contain 1.
    let near = jacobi_theta_null(&ComplexBall::exact_i64(0, 40, P), bits(P)).unwrap();
    assert!(
        !near[2].contains_f64(1.0, 0.0),
        "theta_3(0, 40i) should be resolvably different from 1: {}",
        near[2]
    );
    let tau = ComplexBall::exact_i64(0, 100, P);
    let th = jacobi_theta_null(&tau, bits(P)).unwrap();
    assert!(th[2].contains_f64(1.0, 0.0), "theta_3(0, 100i) = {}", th[2]);
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn jacobi_quartic_identity_holds_inside_the_computed_radii() {
    // theta_3^4 = theta_2^4 + theta_4^4. Asserted as an overlap of the two
    // computed balls, which tests the values and the error bounds together.
    for (re, im) in [(0.0, 1.0), (0.3, 1.4), (-0.45, 0.6), (0.125, 2.5)] {
        let tau = ComplexBall::exact_f64(re, im, P);
        let th = jacobi_theta_null(&tau, bits(P)).unwrap();
        let p4 = |x: &ComplexBall| {
            let s = x.mul(x, P).unwrap();
            s.mul(&s, P).unwrap()
        };
        let lhs = p4(&th[2]);
        let rhs = p4(&th[1]).add(&p4(&th[3]), P).unwrap();
        assert!(
            lhs.overlaps(&rhs),
            "Jacobi identity failed at tau = {re} + {im}i: {lhs} vs {rhs}"
        );
        assert!(
            lhs.accuracy_bits() > 100,
            "identity asserted on a ball with only {} bits",
            lhs.accuracy_bits()
        );
    }
}

// ---------------------------------------------------------------------------
// Weierstrass
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn weierstrass_differential_equation_holds() {
    // p'^2 = 4 p^3 - g2 p - g3, inside the computed radii.
    let tau = ComplexBall::exact_f64(0.2, 1.3, P);
    let z = ComplexBall::exact_f64(0.31, 0.17, P);
    let p = weierstrass_p(&z, &tau, bits(P)).unwrap();
    let pp = weierstrass_p_prime(&z, &tau, bits(P)).unwrap();
    let (g2, g3) = weierstrass_invariants(&tau, bits(P)).unwrap();

    let lhs = pp.mul(&pp, P).unwrap();
    let p3 = p.mul(&p, P).unwrap().mul(&p, P).unwrap();
    let four = ComplexBall::exact_i64(4, 0, P);
    let rhs = four
        .mul(&p3, P)
        .unwrap()
        .sub(&g2.mul(&p, P).unwrap(), P)
        .unwrap()
        .sub(&g3, P)
        .unwrap();
    assert!(lhs.overlaps(&rhs), "Weierstrass ODE failed: {lhs} vs {rhs}");
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn weierstrass_p_has_periods_exactly_one_and_tau() {
    // This is the test that pins the *lattice scaling*, and it is the only one
    // here that can. The differential equation, `e1 + e2 + e3 = 0` and the
    // relation to `j` are all invariant under rescaling the lattice by a
    // constant — if FLINT's lattice were `2Z + 2 tau Z` rather than
    // `Z + tau Z`, every one of them would still pass. Periodicity with period
    // exactly 1 does not.
    let tau = ComplexBall::exact_f64(0.2, 1.3, P);
    let z = ComplexBall::exact_f64(0.31, 0.17, P);
    let p0 = weierstrass_p(&z, &tau, bits(P)).unwrap();

    let z1 = z.add(&ComplexBall::exact_i64(1, 0, P), P).unwrap();
    let p1 = weierstrass_p(&z1, &tau, bits(P)).unwrap();
    assert!(p0.overlaps(&p1), "p is not 1-periodic: {p0} vs {p1}");

    let z2 = z.add(&tau, P).unwrap();
    let p2 = weierstrass_p(&z2, &tau, bits(P)).unwrap();
    assert!(p0.overlaps(&p2), "p is not tau-periodic: {p0} vs {p2}");

    // And a half-period is *not* a period, so the check above is not vacuous.
    let zh = z.add(&ComplexBall::exact_f64(0.5, 0.0, P), P).unwrap();
    let ph = weierstrass_p(&zh, &tau, bits(P)).unwrap();
    assert!(
        !p0.overlaps(&ph),
        "p(z + 1/2) agreed with p(z); the periodicity test proves nothing"
    );
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn jacobi_theta_is_periodic_in_z_with_period_one() {
    // Pins the `z` normalisation: FLINT's theta_3 is
    // `1 + 2 sum q^{n^2} cos(2 n pi z)`, whose period in `z` is 1. The variant
    // that appears in half the literature writes `cos(2 n z)` and has period
    // `pi`. Every value test in this file would pass under either, because they
    // all sit at a single `z`.
    let tau = ComplexBall::exact_f64(0.15, 1.1, P);
    let z = ComplexBall::exact_f64(0.23, 0.08, P);
    let th = jacobi_theta(&z, &tau, bits(P)).unwrap();

    let shifted = jacobi_theta(
        &z.add(&ComplexBall::exact_i64(1, 0, P), P).unwrap(),
        &tau,
        bits(P),
    )
    .unwrap();
    // theta_1 and theta_2 are anti-periodic, theta_3 and theta_4 periodic.
    assert!(th[0].neg(P).unwrap().overlaps(&shifted[0]));
    assert!(th[1].neg(P).unwrap().overlaps(&shifted[1]));
    assert!(th[2].overlaps(&shifted[2]));
    assert!(th[3].overlaps(&shifted[3]));

    // A shift by 1/2 is not a period, so the test is not vacuous.
    let half = jacobi_theta(
        &z.add(&ComplexBall::exact_f64(0.5, 0.0, P), P).unwrap(),
        &tau,
        bits(P),
    )
    .unwrap();
    assert!(!th[2].overlaps(&half[2]));
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn weierstrass_roots_sum_to_zero() {
    let tau = ComplexBall::exact_f64(0.1, 1.9, P);
    let [e1, e2, e3] = weierstrass_roots(&tau, bits(P)).unwrap();
    let s = e1.add(&e2, P).unwrap().add(&e3, P).unwrap();
    assert!(s.contains_zero(), "e1 + e2 + e3 = {s}");
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn weierstrass_p_is_indeterminate_at_a_lattice_point() {
    let tau = ComplexBall::exact_f64(0.0, 1.0, P);
    let z = ComplexBall::exact_i64(0, 0, P);
    let p = weierstrass_p(&z, &tau, bits(P)).unwrap();
    assert!(
        p.is_indeterminate(),
        "p has a double pole at z = 0; got {p}"
    );
    assert!(p.value_if_accurate("weierstrass_p", 1).is_err());
}

// ---------------------------------------------------------------------------
// Characteristics
// ---------------------------------------------------------------------------

#[test]
fn characteristic_index_matches_flints_documented_example() {
    // FLINT's own worked example: a = (1, 0), b = (0, 0) in genus 2 is index 8.
    assert_eq!(theta_characteristic_index(&[1, 0], &[0, 0]).unwrap(), 8);
    assert_eq!(theta_characteristic_index(&[0], &[0]).unwrap(), 0);
    assert_eq!(theta_characteristic_index(&[0], &[1]).unwrap(), 1);
    assert_eq!(theta_characteristic_index(&[1], &[0]).unwrap(), 2);
    assert_eq!(theta_characteristic_index(&[1], &[1]).unwrap(), 3);
}

#[test]
fn characteristic_bits_round_trip() {
    for g in 1..=3usize {
        for ab in 0..(1u64 << (2 * g)) {
            let (a, b) = theta_characteristic_bits(ab, g).unwrap();
            assert_eq!(theta_characteristic_index(&a, &b).unwrap(), ab);
        }
    }
}

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn char_parity_matches_flint() {
    // The pure-Rust parity must agree with FLINT's acb_theta_char_dot for every
    // characteristic up to genus 3, and the even/odd counts must be the
    // classical 2^{g-1}(2^g +/- 1).
    for g in 1..=3usize {
        let mask = (1u64 << g) - 1;
        let mut even = 0;
        for ab in 0..(1u64 << (2 * g)) {
            let a = (ab >> g) & mask;
            let b = ab & mask;
            let flint = super::backend::char_dot(a, b, g).unwrap();
            let ours = theta_characteristic_is_even(ab, g).unwrap();
            assert_eq!(
                ours,
                flint % 2 == 0,
                "parity disagreement at g={g}, ab={ab}: flint dot = {flint}"
            );
            if ours {
                even += 1;
            }
        }
        let expect = 1u64 << (g - 1);
        assert_eq!(even, expect * ((1u64 << g) + 1));
    }
}

#[test]
fn characteristic_out_of_range_is_refused() {
    let err = theta_characteristic_is_even(4, 1).unwrap_err();
    assert!(matches!(err, ThetaError::CharacteristicOutOfRange { .. }));
    let err = theta_characteristic_bits(0, 0).unwrap_err();
    assert!(matches!(err, ThetaError::GenusOutOfRange { .. }));
}

// ---------------------------------------------------------------------------
// Genus 1 Riemann theta vs the classical Jacobi route
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn genus_one_agrees_with_classical_jacobi() {
    // Two independent FLINT code paths — acb_theta (genus-g quasi-linear
    // duplication) and acb_modular (classical q-series with a modular
    // transformation) — must produce overlapping balls, under FLINT's
    // documented genus-1 dictionary
    //   (theta_1, theta_2, theta_3, theta_4) = (-th[3], th[2], th[0], th[1]).
    for (zr, zi, tr, ti) in [
        (0.0, 0.0, 0.0, 1.0),
        (0.3, 0.1, 0.2, 1.3),
        (-0.21, 0.44, 0.45, 0.8),
    ] {
        let z = ComplexBall::exact_f64(zr, zi, P);
        let tau = ComplexBall::exact_f64(tr, ti, P);
        let classical = jacobi_theta(&z, &tau, bits(P)).unwrap();
        let sm = SiegelMatrix::genus_one(tau.clone());
        let all = riemann_theta(std::slice::from_ref(&z), &sm, bits(P)).unwrap();
        assert_eq!(all.len(), 4);

        let neg_th3 = all.get(3).unwrap().neg(P).unwrap();
        let pairs: [(&ComplexBall, &ComplexBall); 4] = [
            (&classical[0], &neg_th3),
            (&classical[1], all.get(2).unwrap()),
            (&classical[2], all.get(0).unwrap()),
            (&classical[3], all.get(1).unwrap()),
        ];
        for (k, (a, b)) in pairs.iter().enumerate() {
            assert!(
                a.overlaps(b),
                "theta_{} disagreement at z={zr}+{zi}i, tau={tr}+{ti}i: {a} vs {b}",
                k + 1
            );
        }
        // And the sign really is load-bearing: theta_1 and +th[3] must *not*
        // agree away from the zero locus.
        assert!(
            !classical[0].overlaps(all.get(3).unwrap()) || classical[0].contains_zero(),
            "the theta_1 sign convention is not being tested at z={zr}+{zi}i"
        );
    }
}

// ---------------------------------------------------------------------------
// Genus 2 and above
// ---------------------------------------------------------------------------

/// A diagonal genus-2 period matrix `diag(tau1, tau2)`.
fn diagonal_g2(t1: ComplexBall, t2: ComplexBall) -> SiegelMatrix {
    let zero = ComplexBall::exact_i64(0, 0, P);
    SiegelMatrix::from_upper_triangle(2, vec![t1, zero, t2]).unwrap()
}

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn genus_two_diagonal_factorises_into_genus_one() {
    // For tau = diag(t1, t2) the lattice splits, and
    //   theta_{(a1,a2),(b1,b2)}(0, tau) = theta_{a1,b1}(0, t1) * theta_{a2,b2}(0, t2).
    // Independent evidence that the genus-2 path and the characteristic
    // ordering are both right.
    let t1 = ComplexBall::exact_f64(0.0, 1.0, P);
    let t2 = ComplexBall::exact_f64(0.25, 1.6, P);
    let tau = diagonal_g2(t1.clone(), t2.clone());
    let z = vec![ComplexBall::exact_i64(0, 0, P); 2];
    let all = riemann_theta(&z, &tau, bits(P)).unwrap();
    assert_eq!(all.len(), 16);

    let s1 = SiegelMatrix::genus_one(t1);
    let s2 = SiegelMatrix::genus_one(t2);
    let v1 = riemann_theta(&[ComplexBall::exact_i64(0, 0, P)], &s1, bits(P)).unwrap();
    let v2 = riemann_theta(&[ComplexBall::exact_i64(0, 0, P)], &s2, bits(P)).unwrap();

    for a1 in 0..2u8 {
        for a2 in 0..2u8 {
            for b1 in 0..2u8 {
                for b2 in 0..2u8 {
                    let ab = theta_characteristic_index(&[a1, a2], &[b1, b2]).unwrap();
                    let lhs = all.get(ab).unwrap();
                    let i1 = theta_characteristic_index(&[a1], &[b1]).unwrap();
                    let i2 = theta_characteristic_index(&[a2], &[b2]).unwrap();
                    let rhs = v1.get(i1).unwrap().mul(v2.get(i2).unwrap(), P).unwrap();
                    assert!(
                        lhs.overlaps(&rhs),
                        "genus-2 splitting failed at a=({a1},{a2}) b=({b1},{b2}): {lhs} vs {rhs}"
                    );
                }
            }
        }
    }
}

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn genus_two_odd_characteristics_vanish_at_z_zero() {
    // theta_{a,b}(-z, tau) = (-1)^{a.b} theta_{a,b}(z, tau), so the six odd
    // characteristics in genus 2 must all enclose zero at z = 0.
    let tau = SiegelMatrix::from_upper_triangle(
        2,
        vec![
            ComplexBall::exact_f64(0.1, 1.4, P),
            ComplexBall::exact_f64(0.05, 0.3, P),
            ComplexBall::exact_f64(-0.2, 1.7, P),
        ],
    )
    .unwrap();
    let z = vec![ComplexBall::exact_i64(0, 0, P); 2];
    let all = riemann_theta(&z, &tau, bits(P)).unwrap();
    let mut odd_seen = 0;
    for ab in 0..16u64 {
        let v = all.get(ab).unwrap();
        if theta_characteristic_is_even(ab, 2).unwrap() {
            assert!(
                !v.contains_zero(),
                "even characteristic {ab} unexpectedly encloses zero: {v}"
            );
        } else {
            odd_seen += 1;
            assert!(
                v.contains_zero() && v.radius_re().to_f64() < 1e-30,
                "odd characteristic {ab} should vanish at z = 0: {v}"
            );
        }
    }
    assert_eq!(odd_seen, 6);
}

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn genus_two_squares_match_the_squares_of_the_values() {
    let tau = SiegelMatrix::from_upper_triangle(
        2,
        vec![
            ComplexBall::exact_f64(0.0, 1.2, P),
            ComplexBall::exact_f64(0.1, 0.25, P),
            ComplexBall::exact_f64(0.0, 1.5, P),
        ],
    )
    .unwrap();
    let z = vec![
        ComplexBall::exact_f64(0.11, 0.02, P),
        ComplexBall::exact_f64(-0.07, 0.13, P),
    ];
    let vals = riemann_theta(&z, &tau, bits(P)).unwrap();
    let sqrs = riemann_theta_squared(&z, &tau, bits(P)).unwrap();
    assert!(!vals.is_squared());
    assert!(sqrs.is_squared());
    for ab in 0..16u64 {
        let v = vals.get(ab).unwrap();
        let s = sqrs.get(ab).unwrap();
        let v2 = v.mul(v, P).unwrap();
        assert!(v2.overlaps(s), "square mismatch at {ab}: {v2} vs {s}");
    }
}

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn theta_one_agrees_with_theta_all() {
    let tau = SiegelMatrix::from_upper_triangle(
        2,
        vec![
            ComplexBall::exact_f64(0.0, 1.1, P),
            ComplexBall::exact_f64(0.2, 0.4, P),
            ComplexBall::exact_f64(0.0, 1.3, P),
        ],
    )
    .unwrap();
    let z = vec![
        ComplexBall::exact_f64(0.05, 0.0, P),
        ComplexBall::exact_f64(0.0, 0.09, P),
    ];
    let all = riemann_theta(&z, &tau, bits(P)).unwrap();
    for ab in 0..16u64 {
        let one = riemann_theta_characteristic(&z, &tau, ab, bits(P)).unwrap();
        assert!(
            one.overlaps(all.get(ab).unwrap()),
            "acb_theta_one and acb_theta_all disagree at {ab}"
        );
    }
}

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn genus_three_diagonal_still_factorises() {
    // Genus 3 is the last genus with a test. 64 values; slower but tractable.
    let p = 128;
    let ts = [
        ComplexBall::exact_f64(0.0, 1.0, p),
        ComplexBall::exact_f64(0.1, 1.3, p),
        ComplexBall::exact_f64(-0.2, 1.7, p),
    ];
    let zero = ComplexBall::exact_i64(0, 0, p);
    let upper = vec![
        ts[0].clone(),
        zero.clone(),
        zero.clone(),
        ts[1].clone(),
        zero.clone(),
        ts[2].clone(),
    ];
    let tau = SiegelMatrix::from_upper_triangle(3, upper).unwrap();
    let z = vec![zero.clone(); 3];
    let all = riemann_theta(&z, &tau, Precision::Bits(p)).unwrap();
    assert_eq!(all.len(), 64);

    let singles: Vec<_> = ts
        .iter()
        .map(|t| {
            riemann_theta(
                std::slice::from_ref(&zero),
                &SiegelMatrix::genus_one(t.clone()),
                Precision::Bits(p),
            )
            .unwrap()
        })
        .collect();

    for ab in 0..64u64 {
        let (a, b) = theta_characteristic_bits(ab, 3).unwrap();
        let lhs = all.get(ab).unwrap();
        let mut rhs = ComplexBall::exact_i64(1, 0, p);
        for k in 0..3 {
            let i = theta_characteristic_index(&[a[k]], &[b[k]]).unwrap();
            rhs = rhs.mul(singles[k].get(i).unwrap(), p).unwrap();
        }
        assert!(
            lhs.overlaps(&rhs),
            "genus-3 splitting failed at ab={ab}: {lhs} vs {rhs}"
        );
    }
}

// ---------------------------------------------------------------------------
// Siegel domain checks and refusals
// ---------------------------------------------------------------------------

#[test]
fn a_non_symmetric_period_matrix_is_refused() {
    let entries = vec![
        ComplexBall::exact_f64(0.0, 1.0, P),
        ComplexBall::exact_f64(0.1, 0.2, P),
        ComplexBall::exact_f64(0.3, 0.2, P),
        ComplexBall::exact_f64(0.0, 1.0, P),
    ];
    let err = SiegelMatrix::new(2, entries).unwrap_err();
    assert!(matches!(err, ThetaError::NotSymmetric { i: 0, j: 1 }));
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn a_matrix_outside_the_siegel_upper_half_space_is_refused() {
    // Im(tau) = diag(1, -1) is not positive definite.
    let tau = SiegelMatrix::from_upper_triangle(
        2,
        vec![
            ComplexBall::exact_i64(0, 1, P),
            ComplexBall::exact_i64(0, 0, P),
            ComplexBall::exact_i64(0, -1, P),
        ],
    )
    .unwrap();
    assert!(!tau.is_certainly_in_siegel_upper_half_space(P).unwrap());
    let z = vec![ComplexBall::exact_i64(0, 0, P); 2];
    let err = riemann_theta(&z, &tau, bits(P)).unwrap_err();
    assert!(matches!(
        err,
        ThetaError::NotInSiegelUpperHalfSpace { genus: 2, .. }
    ));
}

#[test]
fn a_z_of_the_wrong_length_is_refused() {
    let tau = SiegelMatrix::genus_one(tau_i());
    let z = vec![ComplexBall::exact_i64(0, 0, P); 2];
    let err = riemann_theta(&z, &tau, bits(P)).unwrap_err();
    assert!(matches!(err, ThetaError::DimensionMismatch { .. }));
}

#[test]
fn genus_and_precision_limits_are_enforced() {
    let tau = SiegelMatrix::genus_one(tau_i());
    let z = vec![ComplexBall::exact_i64(0, 0, P)];
    let err = riemann_theta(&z, &tau, Precision::Bits(1)).unwrap_err();
    assert!(matches!(err, ThetaError::PrecisionOutOfRange { .. }));
    let err = riemann_theta(&z, &tau, Precision::Bits(MAX_PRECISION_BITS + 1)).unwrap_err();
    assert!(matches!(err, ThetaError::PrecisionOutOfRange { .. }));
    let err = SiegelMatrix::new(MAX_GENUS + 1, Vec::new()).unwrap_err();
    assert!(matches!(err, ThetaError::GenusOutOfRange { .. }));
}

#[test]
#[cfg_attr(
    not(flint_acb_theta),
    ignore = "needs FLINT >= 3.2's acb_theta interface; see build.rs's flint_acb_theta probe"
)]
fn siegel_reduce_returns_a_symplectic_matrix_and_a_reduced_point() {
    // tau = 3.5 + 0.02i is far from the fundamental domain in both directions.
    let tau = SiegelMatrix::genus_one(ComplexBall::exact_f64(3.5, 0.02, P));
    assert!(!siegel_is_reduced(&tau, -10, P).unwrap());
    let red = siegel_reduce(&tau, P).unwrap();
    assert_eq!(red.symplectic().len(), 4);
    // det of the 2x2 symplectic matrix is 1.
    let m = red.symplectic();
    let det = rug::Integer::from(&m[0] * &m[3]) - rug::Integer::from(&m[1] * &m[2]);
    assert_eq!(det, 1, "Siegel reduction returned a non-symplectic matrix");
    assert!(
        siegel_is_reduced(red.reduced(), -10, P).unwrap(),
        "reduced tau = {:?} is not in the reduced domain",
        red.reduced().entry(0, 0).map(ToString::to_string)
    );
    // And j is invariant under the action, which is the check that the
    // transformed matrix really is SL2-equivalent to the original.
    let j0 = j_invariant(tau.entry(0, 0).unwrap(), bits(P)).unwrap();
    let j1 = j_invariant(red.reduced().entry(0, 0).unwrap(), bits(P)).unwrap();
    assert!(j0.overlaps(&j1), "j is not invariant: {j0} vs {j1}");
}

// ---------------------------------------------------------------------------
// Modular transformation checkpoints
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn j_is_invariant_under_tau_plus_one_and_minus_one_over_tau() {
    let tau = ComplexBall::exact_f64(0.17, 1.31, P);
    let j = j_invariant(&tau, bits(P)).unwrap();

    let shifted = tau.add(&ComplexBall::exact_i64(1, 0, P), P).unwrap();
    let j_shift = j_invariant(&shifted, bits(P)).unwrap();
    assert!(j.overlaps(&j_shift), "j(tau+1) != j(tau): {j} vs {j_shift}");

    let inverted = ComplexBall::exact_i64(-1, 0, P).div(&tau, P).unwrap();
    let j_inv = j_invariant(&inverted, bits(P)).unwrap();
    assert!(j.overlaps(&j_inv), "j(-1/tau) != j(tau): {j} vs {j_inv}");
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn eta_transforms_by_the_twenty_fourth_root_of_unity_under_tau_plus_one() {
    // eta(tau + 1) = exp(pi i / 12) eta(tau).
    let tau = ComplexBall::exact_f64(0.3, 1.1, P);
    let e0 = dedekind_eta(&tau, bits(P)).unwrap();
    let e1 = dedekind_eta(
        &tau.add(&ComplexBall::exact_i64(1, 0, P), P).unwrap(),
        bits(P),
    )
    .unwrap();
    // exp(pi i / 12) = zeta_24
    let zeta24 = ComplexBall::root_of_unity(24, P).unwrap();
    let rhs = zeta24.mul(&e0, P).unwrap();
    assert!(
        e1.overlaps(&rhs),
        "eta(tau+1) = {e1} does not match zeta_24 * eta(tau) = {rhs}"
    );
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn theta3_transforms_under_the_modular_inversion() {
    // theta_3(0, -1/tau) = sqrt(-i tau) theta_3(0, tau).
    let tau = ComplexBall::exact_f64(0.0, 1.7, P);
    let th = jacobi_theta_null(&tau, bits(P)).unwrap();
    let inv = ComplexBall::exact_i64(-1, 0, P).div(&tau, P).unwrap();
    let th_inv = jacobi_theta_null(&inv, bits(P)).unwrap();
    let minus_i = ComplexBall::exact_i64(0, -1, P);
    let factor = minus_i.mul(&tau, P).unwrap().sqrt(P).unwrap();
    let rhs = factor.mul(&th[2], P).unwrap();
    assert!(
        th_inv[2].overlaps(&rhs),
        "theta_3 inversion failed: {} vs {rhs}",
        th_inv[2]
    );
}

// ---------------------------------------------------------------------------
// Precision behaviour
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn higher_precision_gives_a_tighter_enclosure() {
    let tau = ComplexBall::exact_f64(0.11, 1.23, 512);
    let lo = j_invariant(&tau, Precision::Bits(64)).unwrap();
    let hi = j_invariant(&tau, Precision::Bits(512)).unwrap();
    assert!(lo.overlaps(&hi));
    assert!(
        hi.accuracy_bits() > lo.accuracy_bits() + 300,
        "accuracy did not improve with precision: {} -> {}",
        lo.accuracy_bits(),
        hi.accuracy_bits()
    );
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn a_wide_input_ball_produces_a_wide_output_ball() {
    // The point of the ball model: uncertainty in the input is carried, not
    // discarded.
    let sharp = ComplexBall::exact_f64(0.0, 1.0, P);
    let fuzzy = ComplexBall::from_parts(
        RealBall::exact_f64(0.0, P),
        RealBall::from_mid_rad(
            rug::Float::with_val(P, 1.0),
            rug::Float::with_val(P, 1e-6),
            P,
        ),
    );
    let j_sharp = j_invariant(&sharp, bits(P)).unwrap();
    let j_fuzzy = j_invariant(&fuzzy, bits(P)).unwrap();
    assert!(j_sharp.overlaps(&j_fuzzy));
    assert!(
        j_fuzzy.accuracy_bits() < 40,
        "a 1e-6 input radius should cost most of the output accuracy, got {}",
        j_fuzzy.accuracy_bits()
    );
    assert!(j_fuzzy.contains_f64(1728.0, 0.0));
}

#[test]
#[cfg_attr(
    not(flint_arb),
    ignore = "needs a FLINT carrying the Arb layer; see build.rs's flint_arb probe"
)]
fn value_if_accurate_gates_on_the_radius() {
    let j = j_invariant(&tau_i(), bits(P)).unwrap();
    let (re, im) = j.value_if_accurate("j", 100).unwrap();
    assert!((re - 1728.0).abs() < 1e-9);
    assert!(im.abs() < 1e-9);
    let err = j.value_if_accurate("j", 10_000).unwrap_err();
    assert!(matches!(err, ThetaError::InsufficientAccuracy { .. }));
}

#[test]
fn error_codes_are_stable_and_registered() {
    use crate::errors::AlkahestError;
    let cases: Vec<ThetaError> = vec![
        ThetaError::BackendUnavailable { capability: "arb" },
        ThetaError::AbiMismatch {
            detail: String::new(),
        },
        ThetaError::PrecisionOutOfRange { bits: 0, max: 1 },
        ThetaError::GenusOutOfRange { genus: 9, max: 6 },
        ThetaError::DimensionMismatch {
            what: "x",
            expected: 1,
            got: 2,
        },
        ThetaError::NotSymmetric { i: 0, j: 1 },
        ThetaError::NotInSiegelUpperHalfSpace { genus: 2, prec: 64 },
        ThetaError::CharacteristicOutOfRange { ab: 9, genus: 1 },
        ThetaError::NotInUpperHalfPlane { function: "j" },
        ThetaError::InsufficientAccuracy {
            function: "j",
            requested_bits: 1,
            achieved_bits: None,
            at_precision: 2,
        },
        ThetaError::Indeterminate { function: "j" },
        ThetaError::NotRepresentable { what: "midpoint" },
    ];
    for (k, e) in cases.iter().enumerate() {
        assert_eq!(e.code(), format!("E-THETA-{:03}", k + 1));
        assert!(e.remediation().is_some(), "{} has no remediation", e.code());
        assert!(!e.to_string().is_empty());
    }
}

/// On a build whose FLINT carries no Arb layer, every evaluator must refuse
/// with `E-THETA-001` — not panic, and not fail to exist.
///
/// This is the case the rest of this file cannot cover, because the rest of
/// this file needs the backend. It compiles only on such a build, which is what
/// keeps the stub in `backend_stub.rs` from rotting unnoticed.
#[cfg(not(flint_arb))]
#[test]
fn every_evaluator_refuses_without_the_backend() {
    use crate::errors::AlkahestError;
    let tau = ComplexBall::exact_i64(0, 1, 64);
    let expect = |e: ThetaError| {
        assert_eq!(e.code(), "E-THETA-001", "unexpected refusal: {e}");
    };
    expect(j_invariant(&tau, bits(64)).unwrap_err());
    expect(dedekind_eta(&tau, bits(64)).unwrap_err());
    expect(jacobi_theta_null(&tau, bits(64)).unwrap_err());
    expect(weierstrass_p(&tau, &tau, bits(64)).unwrap_err());
    expect(weierstrass_invariants(&tau, bits(64)).unwrap_err());
    let sm = SiegelMatrix::genus_one(tau.clone());
    expect(riemann_theta(std::slice::from_ref(&tau), &sm, bits(64)).unwrap_err());
    expect(riemann_theta_characteristic(std::slice::from_ref(&tau), &sm, 0, bits(64)).unwrap_err());
    expect(siegel_reduce(&sm, 64).unwrap_err());
    expect(tau.mul(&tau, 64).unwrap_err());
    assert!(!arb_backend_available());
    assert!(!riemann_theta_available());
    // The parts that are pure Rust keep working, because they never needed
    // FLINT: characteristic packing and parity are combinatorics.
    assert_eq!(theta_characteristic_index(&[1, 0], &[0, 0]).unwrap(), 8);
    assert!(theta_characteristic_is_even(0, 2).unwrap());
}
