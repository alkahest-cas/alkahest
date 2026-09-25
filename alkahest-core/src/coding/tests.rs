//! Unit tests for `coding`, anchored on classical codes whose parameters are
//! in every textbook.
//!
//! The anchors are chosen so that a wrong answer cannot hide: the Hamming and
//! simplex codes are duals of each other and MacWilliams must map each onto the
//! other's *independently enumerated* distribution; the Golay weight
//! distributions are famous and rigid; and every LP bound is checked against a
//! code that actually achieves it, so a bound that came out too small — the
//! only dangerous direction — fails loudly.

use rug::ops::Pow;
use rug::{Integer, Rational};

use crate::errors::AlkahestError;
use crate::ffield::{FiniteField, GfMatrix};

use super::*;

fn gf2() -> FiniteField {
    FiniteField::prime(2).unwrap()
}

fn dist(code: &LinearCode) -> Vec<i64> {
    code.weight_distribution()
        .unwrap()
        .iter()
        .map(|c| c.to_i64().unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// Construction and duality
// ---------------------------------------------------------------------------

#[test]
fn hamming_7_4_from_the_textbook_parity_check() {
    let f = gf2();
    let h = GfMatrix::from_u64(
        &f,
        3,
        7,
        &[
            1, 0, 1, 0, 1, 0, 1, //
            0, 1, 1, 0, 0, 1, 1, //
            0, 0, 0, 1, 1, 1, 1,
        ],
    )
    .unwrap();
    let c = LinearCode::from_parity_check(&h).unwrap();
    assert_eq!((c.length(), c.dimension()), (7, 4));
    assert_eq!(c.redundancy(), 3);
    assert_eq!(c.size(), Integer::from(16));
    assert_eq!(dist(&c), vec![1, 0, 0, 7, 7, 0, 0, 1]);
    assert_eq!(c.minimum_distance().unwrap(), Some(3));
}

#[test]
fn hamming_constructor_agrees_with_the_textbook_matrix() {
    let f = gf2();
    let c = LinearCode::hamming(&f, 3).unwrap();
    assert_eq!((c.length(), c.dimension()), (7, 4));
    assert_eq!(dist(&c), vec![1, 0, 0, 7, 7, 0, 0, 1]);
}

#[test]
fn generator_and_parity_check_determine_the_same_code() {
    let f = gf2();
    let c = LinearCode::hamming(&f, 3).unwrap();
    let from_g = LinearCode::from_generator(c.generator()).unwrap();
    let from_h = LinearCode::from_parity_check(c.parity_check()).unwrap();
    assert!(from_g.generator().equals(c.generator()));
    assert!(from_h.generator().equals(c.generator()));
    // H · Gᵀ = 0 — the defining relation.
    let prod = c.parity_check().mul(&c.generator().transpose()).unwrap();
    assert!(prod.is_zero());
}

#[test]
fn the_dual_of_the_hamming_code_is_the_simplex_code() {
    let f = gf2();
    let c = LinearCode::hamming(&f, 3).unwrap();
    let d = c.dual();
    assert_eq!((d.length(), d.dimension()), (7, 3));
    assert_eq!(dist(&d), vec![1, 0, 0, 0, 7, 0, 0, 0]);
    assert_eq!(d.minimum_distance().unwrap(), Some(4));
    // Dual of the dual is the original code, on the nose after rref.
    let dd = d.dual();
    assert!(dd.generator().equals(c.generator()));
}

#[test]
fn dependent_generator_rows_are_reduced_to_a_basis() {
    let f = gf2();
    // Three rows spanning a 2-dimensional space.
    let g = GfMatrix::from_u64(
        &f,
        3,
        4,
        &[
            1, 1, 0, 0, //
            0, 0, 1, 1, //
            1, 1, 1, 1,
        ],
    )
    .unwrap();
    let c = LinearCode::from_generator(&g).unwrap();
    assert_eq!(c.dimension(), 2);
    assert_eq!(c.size(), Integer::from(4));
    assert_eq!(dist(&c), vec![1, 0, 2, 0, 1]);
}

#[test]
fn the_whole_space_and_the_zero_code_are_duals() {
    let f = gf2();
    let whole = LinearCode::from_generator(&GfMatrix::identity(&f, 5).unwrap()).unwrap();
    assert_eq!(whole.dimension(), 5);
    assert_eq!(whole.redundancy(), 0);
    assert_eq!(whole.size(), Integer::from(32));
    assert_eq!(dist(&whole), vec![1, 5, 10, 10, 5, 1]);

    let zero = whole.dual();
    assert_eq!(zero.dimension(), 0);
    assert_eq!(dist(&zero), vec![1, 0, 0, 0, 0, 0]);
    // The zero code has no minimum distance, and says so rather than guessing.
    assert_eq!(zero.minimum_distance().unwrap(), None);
    assert!(zero.contains(&GfMatrix::zeros(&f, 1, 5).unwrap()).unwrap());
}

#[test]
fn membership_is_decided_by_the_parity_check() {
    let f = gf2();
    let c = LinearCode::hamming(&f, 3).unwrap();
    let zero = GfMatrix::zeros(&f, 1, 7).unwrap();
    assert!(c.contains(&zero).unwrap());
    let g0 = GfMatrix::from_u64(&f, 1, 7, &c.generator().to_u64().unwrap()[0..7]).unwrap();
    assert!(c.contains(&g0).unwrap());
    let bad = GfMatrix::from_u64(&f, 1, 7, &[1, 0, 0, 0, 0, 0, 0]).unwrap();
    assert!(!c.contains(&bad).unwrap());
}

#[test]
fn extended_hamming_is_the_self_dual_8_4_4_code() {
    let f = gf2();
    let c = LinearCode::hamming(&f, 3).unwrap().extend().unwrap();
    assert_eq!((c.length(), c.dimension()), (8, 4));
    assert_eq!(dist(&c), vec![1, 0, 0, 0, 14, 0, 0, 0, 1]);
    assert_eq!(c.minimum_distance().unwrap(), Some(4));
    assert!(c.is_self_orthogonal().unwrap());
    assert!(c.is_self_dual().unwrap());
    // Self-dual really means the dual has the same weight distribution.
    assert_eq!(dist(&c.dual()), dist(&c));
}

#[test]
fn ternary_hamming_4_2_3() {
    let f = FiniteField::prime(3).unwrap();
    let c = LinearCode::hamming(&f, 2).unwrap();
    assert_eq!((c.length(), c.dimension()), (4, 2));
    assert_eq!(c.size(), Integer::from(9));
    assert_eq!(dist(&c), vec![1, 0, 0, 8, 0]);
    assert_eq!(c.minimum_distance().unwrap(), Some(3));
}

#[test]
fn a_code_over_gf4_has_the_right_size_and_weights() {
    let f = FiniteField::extension(2, 2).unwrap();
    // The [3,1,3] repetition code over GF(4): 4 words, 3 of weight 3.
    let c = LinearCode::repetition(&f, 3).unwrap();
    assert_eq!(c.size(), Integer::from(4));
    assert_eq!(dist(&c), vec![1, 0, 0, 3]);
    // Its dual is the [3,2,2] code.
    let d = c.dual();
    assert_eq!(d.dimension(), 2);
    assert_eq!(d.minimum_distance().unwrap(), Some(2));
}

#[test]
fn repetition_and_parity_are_duals_over_gf2() {
    let f = gf2();
    let rep = LinearCode::repetition(&f, 6).unwrap();
    assert_eq!(dist(&rep), vec![1, 0, 0, 0, 0, 0, 1]);
    let parity = rep.dual();
    assert_eq!(parity.dimension(), 5);
    assert_eq!(dist(&parity), vec![1, 0, 15, 0, 15, 0, 1]);
    assert_eq!(parity.minimum_distance().unwrap(), Some(2));
}

// ---------------------------------------------------------------------------
// Golay
// ---------------------------------------------------------------------------

#[test]
fn binary_golay_23_12_7() {
    let c = LinearCode::golay_binary().unwrap();
    assert_eq!((c.length(), c.dimension()), (23, 12));
    let a = dist(&c);
    let mut expected = vec![0i64; 24];
    expected[0] = 1;
    expected[7] = 253;
    expected[8] = 506;
    expected[11] = 1288;
    expected[12] = 1288;
    expected[15] = 506;
    expected[16] = 253;
    expected[23] = 1;
    assert_eq!(a, expected);
    assert_eq!(a.iter().sum::<i64>(), 4096);
    assert_eq!(c.minimum_distance().unwrap(), Some(7));
    // Perfect: the spheres of radius 3 tile GF(2)^23.
    assert_eq!(
        hamming_bound(23, 7, 2).unwrap(),
        Integer::from(4096),
        "the binary Golay code is perfect"
    );
}

#[test]
fn extended_binary_golay_24_12_8_is_self_dual() {
    let c = LinearCode::golay_binary().unwrap().extend().unwrap();
    assert_eq!((c.length(), c.dimension()), (24, 12));
    let a = dist(&c);
    let mut expected = vec![0i64; 25];
    expected[0] = 1;
    expected[8] = 759;
    expected[12] = 2576;
    expected[16] = 759;
    expected[24] = 1;
    assert_eq!(a, expected);
    assert_eq!(c.minimum_distance().unwrap(), Some(8));
    assert!(c.is_self_dual().unwrap());
    assert_eq!(dist(&c.dual()), a);
}

// ---------------------------------------------------------------------------
// MacWilliams
// ---------------------------------------------------------------------------

#[test]
fn macwilliams_maps_hamming_to_simplex_and_back() {
    let f = gf2();
    let c = LinearCode::hamming(&f, 3).unwrap();
    let d = c.dual();

    let wc = c.weight_enumerator().unwrap();
    let wd = d.weight_enumerator().unwrap();

    // Forwards: transform of the primal equals the *enumerated* dual.
    assert_eq!(wc.macwilliams().unwrap(), wd);
    // Backwards: transform of the dual equals the enumerated primal.
    assert_eq!(wd.macwilliams().unwrap(), wc);
    // Which makes it an involution.
    assert_eq!(wc.macwilliams().unwrap().macwilliams().unwrap(), wc);
}

#[test]
fn macwilliams_on_the_extended_golay_reproduces_it() {
    let c = LinearCode::golay_binary().unwrap().extend().unwrap();
    let w = c.weight_enumerator().unwrap();
    // Self-dual, so MacWilliams fixes it.
    assert_eq!(w.macwilliams().unwrap(), w);
}

#[test]
fn macwilliams_over_gf3_and_gf4() {
    for f in [
        FiniteField::prime(3).unwrap(),
        FiniteField::extension(2, 2).unwrap(),
    ] {
        let c = LinearCode::hamming(&f, 2).unwrap();
        let d = c.dual();
        let wc = c.weight_enumerator().unwrap();
        let wd = d.weight_enumerator().unwrap();
        assert_eq!(wc.macwilliams().unwrap(), wd, "field {f:?}");
        assert_eq!(wd.macwilliams().unwrap(), wc, "field {f:?}");
    }
}

#[test]
fn weight_enumerator_renders_and_evaluates() {
    let f = gf2();
    let c = LinearCode::hamming(&f, 3).unwrap();
    let w = c.weight_enumerator().unwrap();
    assert_eq!(w.to_string(), "x^7 + 7*x^4*y^3 + 7*x^3*y^4 + y^7");
    // W(1, 1) = |C|.
    assert_eq!(
        w.evaluate(&Integer::from(1), &Integer::from(1)),
        Integer::from(16)
    );
    assert_eq!(w.size(), Integer::from(16));
    assert_eq!(w.minimum_distance(), Some(3));
    assert!(w.is_dual_feasible());
}

#[test]
fn a_distribution_that_is_not_a_linear_code_is_refused() {
    // A_0 must be 1.
    let e = WeightEnumerator::new(2, vec![Integer::from(2), Integer::from(0)]).unwrap_err();
    assert_eq!(e.code(), "E-CODE-006");
    // Negative multiplicities are refused.
    let e = WeightEnumerator::new(2, vec![Integer::from(1), Integer::from(-1)]).unwrap_err();
    assert_eq!(e.code(), "E-CODE-006");
    // A shape that is not divisible by |C| under MacWilliams.
    let w = WeightEnumerator::new(
        2,
        vec![Integer::from(1), Integer::from(1), Integer::from(1)],
    )
    .unwrap();
    let e = w.macwilliams().unwrap_err();
    assert_eq!(e.code(), "E-CODE-006");
}

// ---------------------------------------------------------------------------
// Krawtchouk
// ---------------------------------------------------------------------------

#[test]
fn krawtchouk_at_zero_is_the_binomial_times_q_minus_one() {
    for &(n, q) in &[(7usize, 2u64), (11, 3), (8, 4), (5, 5)] {
        for k in 0..=n {
            assert_eq!(
                krawtchouk(k, 0, n, q),
                binomial(n, k) * Integer::from(q - 1).pow(k as u32),
                "K_{k}(0; {n}, {q})"
            );
        }
    }
}

#[test]
fn krawtchouk_low_orders_match_the_closed_forms() {
    for &(n, q) in &[(7usize, 2u64), (9, 3), (6, 4)] {
        for x in 0..=n as i64 {
            assert_eq!(krawtchouk(0, x, n, q), Integer::from(1));
            // K_1(x) = (q-1)n - qx
            assert_eq!(
                krawtchouk(1, x, n, q),
                Integer::from((q - 1) * n as u64) - Integer::from(q * x as u64)
            );
        }
    }
}

#[test]
fn krawtchouk_three_term_recurrence_holds() {
    for &(n, q) in &[(7usize, 2u64), (10, 3), (6, 5)] {
        for x in -2..=(n as i64 + 2) {
            for k in 0..n {
                let lhs = Integer::from(k as u32 + 1) * krawtchouk(k + 1, x, n, q);
                let coeff = Integer::from(k as u64) + Integer::from((q - 1) * (n - k) as u64)
                    - Integer::from(q) * Integer::from(x);
                let mut rhs = coeff * krawtchouk(k, x, n, q);
                if k > 0 {
                    rhs -= Integer::from((q - 1) * (n - k + 1) as u64) * krawtchouk(k - 1, x, n, q);
                }
                assert_eq!(lhs, rhs, "recurrence at k={k}, x={x}, n={n}, q={q}");
            }
        }
    }
}

#[test]
fn krawtchouk_polynomial_agrees_with_the_closed_form() {
    for &(n, q) in &[(7usize, 2u64), (9, 3), (5, 4)] {
        for k in 0..=n {
            let poly = krawtchouk_poly(k, n, q);
            assert_eq!(poly.len(), k + 1, "degree of K_{k}");
            for x in -3..=(n as i64 + 3) {
                let mut acc = Rational::from(0);
                let mut xp = Rational::from(1);
                for c in poly.iter() {
                    acc += Rational::from(c * &xp);
                    xp *= Rational::from(x);
                }
                assert_eq!(
                    acc,
                    Rational::from(krawtchouk(k, x, n, q)),
                    "K_{k}({x}; {n}, {q}) by recurrence vs closed form"
                );
            }
        }
    }
}

#[test]
fn krawtchouk_orthogonality() {
    // Σ_i C(n,i)(q-1)^i K_k(i) K_l(i) = q^n C(n,k) (q-1)^k δ_{kl}
    for &(n, q) in &[(6usize, 2u64), (5, 3), (4, 4)] {
        for k in 0..=n {
            for l in 0..=n {
                let mut acc = Integer::from(0);
                for i in 0..=n {
                    let w = binomial(n, i) * Integer::from(q - 1).pow(i as u32);
                    acc += w * krawtchouk(k, i as i64, n, q) * krawtchouk(l, i as i64, n, q);
                }
                let expected = if k == l {
                    Integer::from(q).pow(n as u32)
                        * binomial(n, k)
                        * Integer::from(q - 1).pow(k as u32)
                } else {
                    Integer::from(0)
                };
                assert_eq!(acc, expected, "orthogonality at k={k}, l={l}, n={n}, q={q}");
            }
        }
    }
}

#[test]
fn krawtchouk_reciprocity_and_the_boundedness_identity() {
    for &(n, q) in &[(7usize, 2u64), (5, 3)] {
        for i in 0..=n {
            for k in 0..=n {
                // C(n,i)(q-1)^i K_k(i) = C(n,k)(q-1)^k K_i(k)
                let lhs = binomial(n, i)
                    * Integer::from(q - 1).pow(i as u32)
                    * krawtchouk(k, i as i64, n, q);
                let rhs = binomial(n, k)
                    * Integer::from(q - 1).pow(k as u32)
                    * krawtchouk(i, k as i64, n, q);
                assert_eq!(lhs, rhs, "reciprocity at i={i}, k={k}");
            }
            // Σ_k K_k(i) = q^n [i == 0] — the identity that bounds the LP.
            let total: Integer =
                (0..=n).fold(Integer::from(0), |a, k| a + krawtchouk(k, i as i64, n, q));
            let expected = if i == 0 {
                Integer::from(q).pow(n as u32)
            } else {
                Integer::from(0)
            };
            assert_eq!(total, expected, "Σ_k K_k({i})");
        }
    }
}

#[test]
fn krawtchouk_pairing_is_nonnegative_on_real_codes() {
    let f = gf2();
    for c in [
        LinearCode::hamming(&f, 3).unwrap(),
        LinearCode::hamming(&f, 3).unwrap().extend().unwrap(),
        LinearCode::golay_binary().unwrap(),
    ] {
        let a = c.weight_distribution().unwrap();
        let n = c.length();
        for k in 0..=n {
            assert!(
                krawtchouk_pairing(&a, k, n, 2) >= 0,
                "dual feasibility fails at k={k} for {c:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Delsarte LP
// ---------------------------------------------------------------------------

#[test]
fn delsarte_reproduces_a_2_7_3_equals_16() {
    let b = delsarte_lp_bound(7, 3, 2).unwrap();
    assert_eq!(*b.bound(), Integer::from(16));
    assert_eq!(*b.optimum(), Rational::from(16));
    assert!(b.verify_certificate());
    assert!(b.verify_distribution());
    // The Hamming code achieves it, so the bound is exactly right.
    let f = gf2();
    assert_eq!(
        LinearCode::hamming(&f, 3).unwrap().size(),
        Integer::from(16)
    );
}

#[test]
fn delsarte_reproduces_a_2_8_4_equals_16() {
    let b = delsarte_lp_bound(8, 4, 2).unwrap();
    assert_eq!(*b.bound(), Integer::from(16));
    assert!(b.verify_certificate());
    let f = gf2();
    assert_eq!(
        LinearCode::hamming(&f, 3).unwrap().extend().unwrap().size(),
        Integer::from(16)
    );
}

#[test]
fn delsarte_reproduces_a_2_24_8_equals_4096() {
    // The famous tight case: the LP bound for (24, 8) is exactly 4096, met by
    // the extended binary Golay code.
    let b = delsarte_lp_bound(24, 8, 2).unwrap();
    assert_eq!(*b.bound(), Integer::from(4096));
    assert!(b.verify_certificate());
    assert!(b.verify_distribution());
    let golay = LinearCode::golay_binary().unwrap().extend().unwrap();
    assert_eq!(golay.size(), Integer::from(4096));
}

#[test]
fn delsarte_reproduces_a_2_23_7_equals_4096() {
    let b = delsarte_lp_bound(23, 7, 2).unwrap();
    assert_eq!(*b.bound(), Integer::from(4096));
    assert!(b.verify_certificate());
    assert_eq!(
        LinearCode::golay_binary().unwrap().size(),
        Integer::from(4096)
    );
}

#[test]
fn delsarte_at_distance_one_is_the_whole_space() {
    for &(n, q) in &[(5usize, 2u64), (4, 3), (3, 5)] {
        let b = delsarte_lp_bound(n, 1, q).unwrap();
        assert_eq!(*b.bound(), Integer::from(q).pow(n as u32));
        assert!(b.verify_certificate());
    }
}

#[test]
fn delsarte_at_distance_n_is_q() {
    // A_q(n, n) = q: the repetition code is optimal.
    for &(n, q) in &[(5usize, 2u64), (6, 3), (4, 4)] {
        let b = delsarte_lp_bound(n, n, q).unwrap();
        assert_eq!(*b.bound(), Integer::from(q), "A_{q}({n}, {n})");
    }
}

#[test]
fn delsarte_matches_known_small_binary_values() {
    // (n, d, A_2(n,d)) — the classical table. The LP is tight at all of these.
    for &(n, d, expected) in &[
        (5usize, 3usize, 4i64),
        (6, 3, 8),
        (7, 3, 16),
        (8, 3, 20),
        (9, 3, 40),
        (6, 4, 4),
        (7, 4, 8),
        (8, 4, 16),
        (9, 4, 20),
        (10, 4, 40),
        (9, 5, 6),
        (10, 5, 12),
        (11, 5, 24),
        (12, 5, 32),
    ] {
        let b = delsarte_lp_bound(n, d, 2).unwrap();
        assert!(
            *b.bound() >= expected,
            "A_2({n},{d}) = {expected} but the LP bound came out {} — a bound below a \
             code that exists is the one unacceptable failure",
            b.bound()
        );
        assert!(b.verify_certificate(), "certificate for ({n}, {d})");
    }
}

#[test]
fn delsarte_is_tight_for_the_perfect_and_mds_cases() {
    // Cases where the LP bound is known to equal A_2(n, d) exactly.
    for &(n, d, expected) in &[
        (5usize, 3usize, 4i64),
        (6, 3, 8),
        (7, 3, 16),
        (6, 4, 4),
        (7, 4, 8),
        (8, 4, 16),
        (23, 7, 4096),
        (24, 8, 4096),
    ] {
        let b = delsarte_lp_bound(n, d, 2).unwrap();
        assert_eq!(*b.bound(), Integer::from(expected), "A_2({n},{d})");
    }
}

#[test]
fn delsarte_never_exceeds_singleton_or_hamming() {
    for n in 2usize..=12 {
        for d in 1..=n {
            for q in [2u64, 3, 4] {
                let lp = delsarte_lp_bound(n, d, q).unwrap();
                let s = singleton_bound(n, d, q).unwrap();
                let h = hamming_bound(n, d, q).unwrap();
                assert!(
                    *lp.bound() <= s,
                    "LP {} beat Singleton {s} at ({n}, {d}, {q})",
                    lp.bound()
                );
                assert!(
                    *lp.bound() <= h,
                    "LP {} beat Hamming {h} at ({n}, {d}, {q})",
                    lp.bound()
                );
                assert!(lp.verify_certificate());
            }
        }
    }
}

#[test]
fn delsarte_never_falls_below_a_linear_code_that_exists() {
    // Enumerate a few real codes and check the LP admits each of them.
    let f = gf2();
    let codes = [
        LinearCode::hamming(&f, 3).unwrap(),
        LinearCode::hamming(&f, 3).unwrap().extend().unwrap(),
        LinearCode::hamming(&f, 4).unwrap(),
        LinearCode::repetition(&f, 7).unwrap(),
        LinearCode::golay_binary().unwrap(),
        LinearCode::golay_binary().unwrap().extend().unwrap(),
    ];
    for c in codes.iter() {
        let d = c.minimum_distance().unwrap().expect("non-zero code");
        let b = delsarte_lp_bound(c.length(), d, 2).unwrap();
        assert!(
            *b.bound() >= c.size(),
            "LP bound {} is below the size {} of {c:?}",
            b.bound(),
            c.size()
        );
    }
}

#[test]
fn delsarte_optimum_can_be_fractional_and_is_floored() {
    // A_2(11, 3): the LP optimum is 512/3 = 170.666…, so the certified bound
    // is 170. It coincides with the sphere-packing bound 2^11 / 12 here, which
    // is a useful cross-check that the fractional value is not an artefact.
    let b = delsarte_lp_bound(11, 3, 2).unwrap();
    assert_eq!(*b.optimum(), Rational::from((512, 3)));
    assert_eq!(*b.bound(), Integer::from(170));
    assert_eq!(hamming_bound(11, 3, 2).unwrap(), Integer::from(170));
    assert!(b.verify_certificate());
}

#[test]
fn singleton_and_hamming_bounds_are_the_textbook_values() {
    assert_eq!(singleton_bound(7, 3, 2).unwrap(), Integer::from(32));
    assert_eq!(hamming_bound(7, 3, 2).unwrap(), Integer::from(16));
    assert_eq!(hamming_bound(23, 7, 2).unwrap(), Integer::from(4096));
    assert_eq!(singleton_bound(4, 3, 3).unwrap(), Integer::from(9));
    // Reed–Solomon meets Singleton over GF(4): [4, 2, 3] has 16 words.
    assert_eq!(singleton_bound(4, 3, 4).unwrap(), Integer::from(16));
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn enumeration_past_the_cap_is_refused_not_truncated() {
    let f = gf2();
    // A [64, 40] code: 2^40 codewords, far past the cap.
    let g = GfMatrix::identity(&f, 40).unwrap();
    let mut wide = GfMatrix::zeros(&f, 40, 64).unwrap();
    for i in 0..40 {
        for j in 0..40 {
            wide.set_entry(i, j, &g.entry(i, j).unwrap()).unwrap();
        }
    }
    let c = LinearCode::from_generator(&wide).unwrap();
    let e = c.weight_distribution().unwrap_err();
    assert_eq!(e.code(), "E-CODE-004");
    assert!(e.to_string().contains("truncated"));
    // And the same refusal comes out of minimum_distance rather than a guess.
    assert_eq!(c.minimum_distance().unwrap_err().code(), "E-CODE-004");
}

#[test]
fn raising_the_cap_explicitly_lets_a_larger_code_through() {
    let c = LinearCode::golay_binary().unwrap();
    // Well within the default cap already, but the API exists and works.
    let a = c.weight_distribution_with_cap(1 << 13).unwrap();
    assert_eq!(a[7], Integer::from(253));
    // Below q^k = 4096 it refuses.
    let e = c.weight_distribution_with_cap(100).unwrap_err();
    assert_eq!(e.code(), "E-CODE-004");
}

#[test]
fn degenerate_arguments_are_refused_with_stable_codes() {
    let f = gf2();
    let empty = GfMatrix::zeros(&f, 2, 0).unwrap();
    assert_eq!(
        LinearCode::from_generator(&empty).unwrap_err().code(),
        "E-CODE-001"
    );
    assert_eq!(delsarte_lp_bound(0, 1, 2).unwrap_err().code(), "E-CODE-001");
    assert_eq!(delsarte_lp_bound(7, 0, 2).unwrap_err().code(), "E-CODE-002");
    assert_eq!(delsarte_lp_bound(7, 8, 2).unwrap_err().code(), "E-CODE-002");
    assert_eq!(delsarte_lp_bound(7, 3, 1).unwrap_err().code(), "E-CODE-008");
    assert_eq!(
        delsarte_lp_bound(MAX_LP_LENGTH + 1, 3, 2)
            .unwrap_err()
            .code(),
        "E-CODE-005"
    );
    assert_eq!(LinearCode::hamming(&f, 1).unwrap_err().code(), "E-CODE-002");
    assert_eq!(
        LinearCode::repetition(&f, 0).unwrap_err().code(),
        "E-CODE-001"
    );
}

#[test]
fn every_coding_error_code_is_registered() {
    let codes = [
        "E-CODE-001",
        "E-CODE-002",
        "E-CODE-003",
        "E-CODE-004",
        "E-CODE-005",
        "E-CODE-006",
        "E-CODE-007",
        "E-CODE-008",
    ];
    for code in codes {
        assert!(
            crate::errors::codes::REGISTRY
                .iter()
                .any(|s| s.code == code && s.class == "CodingError"),
            "{code} missing from REGISTRY"
        );
    }
}

#[test]
fn a_finite_field_refusal_keeps_its_own_code_in_the_message() {
    let f2 = gf2();
    let f3 = FiniteField::prime(3).unwrap();
    let c = LinearCode::repetition(&f2, 4).unwrap();
    let wrong = GfMatrix::zeros(&f3, 1, 4).unwrap();
    let e = c.contains(&wrong).unwrap_err();
    assert_eq!(e.code(), "E-CODE-003");
    assert!(
        e.to_string().contains("E-GFQ-006"),
        "the underlying code must stay visible: {e}"
    );
}

/// The measurements behind [`MAX_LP_LENGTH`]'s cost table, re-runnable.
///
/// Ignored by default because the tail of it is minutes long; `n = 96` (the
/// cap) is deliberately left out of the default list for the same reason.
#[test]
#[ignore = "timing probe for MAX_LP_LENGTH; minutes long, run explicitly"]
fn delsarte_timing_probe() {
    for n in [24usize, 32, 48, 64] {
        let t = std::time::Instant::now();
        let b = delsarte_lp_bound(n, n / 3, 2).unwrap();
        println!("n={n} d={} bound={} in {:?}", n / 3, b.bound(), t.elapsed());
        assert!(b.verify_certificate());
    }
}

/// Independent anchor: the Plotkin regime `A_2(2d, d) = 4d` for even `d`.
///
/// This is a *theorem*, not a table lookup, so it checks the LP in a range no
/// literature value was consulted for. The Delsarte LP is known to be tight
/// here, so the bound must land exactly on `4d` - too high means the LP is
/// weak, too low means it is unsound and would rule out codes that exist.
///
/// `d = 4` overlaps the existing `A_2(8,4) = 16` anchor on purpose, as a
/// control; `d = 6, 8, 10` are new.
#[test]
fn delsarte_matches_plotkin_on_a_2_2d_d() {
    for d in [4u32, 6, 8, 10] {
        let n = 2 * d;
        let expected = Integer::from(4 * d);
        let b = delsarte_lp_bound(n as usize, d as usize, 2).unwrap();
        assert_eq!(
            *b.bound(),
            expected,
            "A_2({n}, {d}) should be 4d = {expected} by Plotkin"
        );
        assert!(
            b.verify_certificate(),
            "dual certificate failed at ({n}, {d})"
        );
    }
}
