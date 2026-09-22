//! Correctness anchors for the lattice toolkit.
//!
//! The constructors are checked against counts that only the right lattice
//! produces — 240 vectors of norm 2 for `E_8`, 196560 of norm 4 for the Leech
//! lattice, the classical `r_2` / `r_4` sum-of-squares formulae for `ℤⁿ`.
//! A construction that is nearly right does not hit any of them.

use super::*;
use rug::{Integer, Rational};

fn rows_i64(v: &[&[i64]]) -> Vec<Vec<Integer>> {
    v.iter()
        .map(|r| r.iter().map(|&x| Integer::from(x)).collect())
        .collect()
}

/// `r_2(n)` — representations of `n` as an ordered sum of two signed squares.
/// Jacobi: `4·(d_1(n) − d_3(n))` over divisors `≡ 1, 3 (mod 4)`.
fn r2(n: u64) -> u64 {
    if n == 0 {
        return 1;
    }
    let mut acc: i64 = 0;
    for d in 1..=n {
        if n % d == 0 {
            match d % 4 {
                1 => acc += 1,
                3 => acc -= 1,
                _ => {}
            }
        }
    }
    (4 * acc) as u64
}

/// `r_4(n)` — Jacobi's four-square theorem: `8 · Σ_{d | n, 4 ∤ d} d`.
fn r4(n: u64) -> u64 {
    if n == 0 {
        return 1;
    }
    let mut acc = 0u64;
    for d in 1..=n {
        if n % d == 0 && d % 4 != 0 {
            acc += d;
        }
    }
    8 * acc
}

// ---------------------------------------------------------------------------
// Z^n
// ---------------------------------------------------------------------------

#[test]
fn zn_theta_series_is_the_sum_of_squares_count() {
    let z2 = zn(2).unwrap();
    let theta = z2.theta_series(30).unwrap();
    for (n, c) in theta.iter().enumerate() {
        assert_eq!(*c, Integer::from(r2(n as u64)), "r_2({n})");
    }

    let z4 = zn(4).unwrap();
    let theta = z4.theta_series(20).unwrap();
    for (n, c) in theta.iter().enumerate() {
        assert_eq!(*c, Integer::from(r4(n as u64)), "r_4({n})");
    }
    // The Jacobi closed form for odd n, stated separately because it is the
    // headline identity: r_4(n) = 8·σ(n).
    for n in [1u64, 3, 5, 7, 9, 15] {
        let sigma: u64 = (1..=n).filter(|d| n % d == 0).sum();
        assert_eq!(theta[n as usize], Integer::from(8 * sigma), "8·σ({n})");
    }
}

#[test]
fn zn_basic_invariants() {
    for n in 1..=5usize {
        let l = zn(n).unwrap();
        assert_eq!(l.rank(), n);
        assert_eq!(l.determinant(), Rational::from(1));
        assert_eq!(l.minimum().unwrap(), Rational::from(1));
        assert_eq!(l.kissing_number().unwrap(), 2 * n as u64);
        assert!(l.is_integral());
    }
    // Z^n is integral but odd.
    assert!(!zn(3).unwrap().is_even());
}

// ---------------------------------------------------------------------------
// Root lattices
// ---------------------------------------------------------------------------

#[test]
fn a2_is_hexagonal() {
    let a2 = a_n(2).unwrap();
    assert_eq!(a2.rank(), 2);
    assert_eq!(a2.determinant(), Rational::from(3));
    assert_eq!(a2.minimum().unwrap(), Rational::from(2));
    assert_eq!(a2.kissing_number().unwrap(), 6);
    assert_eq!(a2.minimal_vectors().unwrap().len(), 6);
    let theta = a2.theta_series(6).unwrap();
    assert_eq!(theta[0], Integer::from(1));
    assert_eq!(theta[2], Integer::from(6));
    assert_eq!(theta[6], Integer::from(6));
}

#[test]
fn a_n_family_determinant_and_kissing_number() {
    for n in 1..=5usize {
        let l = a_n(n).unwrap();
        assert_eq!(l.determinant(), Rational::from(n as u32 + 1), "det A_{n}");
        assert_eq!(l.minimum().unwrap(), Rational::from(2));
        assert_eq!(
            l.kissing_number().unwrap(),
            (n * (n + 1)) as u64,
            "kissing A_{n}"
        );
        assert!(l.is_even());
    }
}

#[test]
fn d4_has_twenty_four_minimal_vectors() {
    let d4 = d_n(4).unwrap();
    assert_eq!(d4.determinant(), Rational::from(4));
    assert_eq!(d4.minimum().unwrap(), Rational::from(2));
    assert_eq!(d4.kissing_number().unwrap(), 24);
    assert_eq!(d4.minimal_vectors().unwrap().len(), 24);
    // D_4 is even and its theta series starts 1 + 24q² + 24q⁴ + 96q⁶ + …
    assert!(d4.is_even());
    let theta = d4.theta_series(6).unwrap();
    assert_eq!(theta[2], Integer::from(24));
    assert_eq!(theta[4], Integer::from(24));
    assert_eq!(theta[6], Integer::from(96));
}

#[test]
fn d_n_family_determinant_and_kissing_number() {
    for n in 3..=6usize {
        let l = d_n(n).unwrap();
        assert_eq!(l.determinant(), Rational::from(4), "det D_{n}");
        assert_eq!(l.minimum().unwrap(), Rational::from(2));
        assert_eq!(
            l.kissing_number().unwrap(),
            (2 * n * (n - 1)) as u64,
            "kissing D_{n}"
        );
    }
}

// ---------------------------------------------------------------------------
// E_8
// ---------------------------------------------------------------------------

#[test]
fn e8_is_even_unimodular_with_240_roots() {
    let e = e8().unwrap();
    assert_eq!(e.rank(), 8);
    assert_eq!(e.determinant(), Rational::from(1), "E_8 is unimodular");
    assert!(e.is_even(), "E_8 is an even lattice");
    assert_eq!(e.minimum().unwrap(), Rational::from(2));
    assert_eq!(e.kissing_number().unwrap(), 240);
    assert_eq!(e.minimal_vectors().unwrap().len(), 240);
}

#[test]
fn e8_theta_series_is_the_eisenstein_series() {
    // θ_{E_8} = E_4 = 1 + 240 Σ σ_3(n) qⁿ.
    let e = e8().unwrap();
    let theta = e.theta_series(8).unwrap();
    for n in 1..=4u64 {
        // norm 2n ↔ the qⁿ coefficient of E_4
        let sigma3: u64 = (1..=n).filter(|d| n % d == 0).map(|d| d * d * d).sum();
        assert_eq!(
            theta[(2 * n) as usize],
            Integer::from(240 * sigma3),
            "E_4 coefficient at q^{n}"
        );
    }
    // Every odd norm is empty in an even lattice.
    for n in (1..=7).step_by(2) {
        assert_eq!(theta[n], Integer::new());
    }
}

#[test]
fn e8_packing_invariants() {
    let e = e8().unwrap();
    assert_eq!(
        e.center_density_exact().unwrap(),
        Some(Rational::from((1, 16))),
        "centre density of E_8 is exactly 1/16"
    );
    assert!((e.center_density().unwrap() - 1.0 / 16.0).abs() < 1e-12);
    assert!((e.hermite_invariant().unwrap() - 2.0).abs() < 1e-12);
    // Δ = δ·V_8 = (1/16)·π⁴/24 = π⁴/384 ≈ 0.25367
    assert!((e.packing_density().unwrap() - 0.253_669_5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Leech
// ---------------------------------------------------------------------------

#[test]
fn leech_is_even_unimodular_of_minimum_four() {
    let l = leech().unwrap();
    assert_eq!(l.rank(), 24);
    assert_eq!(l.determinant(), Rational::from(1));
    assert!(l.is_even());
    // One enumeration pass; `center_density_exact` and `hermite_invariant`
    // would each run another, and at rank 24 that is minutes rather than
    // seconds. Both are exercised against `E_8` above, and with det = 1 and
    // λ₁² = 4 their values here are arithmetic: δ = (4/4)^12 = 1, γ = 4/1 = 4.
    assert_eq!(l.minimum().unwrap(), Rational::from(4), "no roots at all");
}

/// The 196560 test. Nothing but the Leech lattice produces this number.
#[test]
fn leech_has_196560_minimal_vectors() {
    let l = leech().unwrap();
    assert_eq!(l.kissing_number().unwrap(), 196_560);
}

// ---------------------------------------------------------------------------
// Duals
// ---------------------------------------------------------------------------

#[test]
fn dual_of_dual_is_the_original() {
    for l in [
        zn(3).unwrap(),
        a_n(3).unwrap(),
        d_n(4).unwrap(),
        e8().unwrap(),
    ] {
        let dd = l.dual().unwrap().dual().unwrap();
        assert_eq!(dd.gram_matrix(), l.gram_matrix());
    }
}

#[test]
fn dual_determinant_is_the_reciprocal() {
    let a3 = a_n(3).unwrap();
    assert_eq!(a3.determinant(), Rational::from(4));
    assert_eq!(a3.dual().unwrap().determinant(), Rational::from((1, 4)));
    // E_8 and Λ₂₄ are self-dual up to isometry, so their Gram matrices have the
    // same determinant.
    let e = e8().unwrap();
    assert_eq!(e.dual().unwrap().determinant(), Rational::from(1));
}

// ---------------------------------------------------------------------------
// SVP / CVP
// ---------------------------------------------------------------------------

#[test]
fn shortest_vector_attains_the_minimum() {
    let basis = rows_i64(&[&[1, 0, 0], &[0, 1, 0], &[9001, 8999, 1]]);
    let l = Lattice::from_basis(&basis).unwrap();
    let v = l.shortest_vector().unwrap();
    assert_eq!(v.norm, l.minimum().unwrap());
    assert_eq!(v.norm, Rational::from(1));
    let coords = v.coordinates.expect("basis was supplied");
    let n: Rational = coords.iter().map(|c| Rational::from(c * c)).sum();
    assert_eq!(n, v.norm, "reported norm must match the coordinates");
}

#[test]
fn closest_vector_on_z2() {
    let l = zn(2).unwrap();
    let target = vec![Rational::from((2, 5)), Rational::from((3, 5))];
    let v = l.closest_vector(&target).unwrap();
    assert_eq!(
        v.coordinates.unwrap(),
        vec![Rational::new(), Rational::from(1)]
    );
    assert_eq!(
        v.norm,
        Rational::from((2, 5)) * Rational::from((2, 5))
            + Rational::from((2, 5)) * Rational::from((2, 5))
    );
}

#[test]
fn closest_vector_beats_naive_rounding_on_a_skewed_basis() {
    // A basis of Z^2 that rounding gets wrong: b1 = (1, 0), b2 = (10, 1) …
    // asked for the point (5, 1/2), plain coefficient rounding lands further
    // away than the true closest vector.
    let basis = rows_i64(&[&[1, 0], &[10, 1]]);
    let l = Lattice::from_basis(&basis).unwrap();
    let target = vec![Rational::from((11, 2)), Rational::from((1, 2))];
    let v = l.closest_vector(&target).unwrap();
    let coords = v.coordinates.clone().unwrap();
    let dist: Rational = coords
        .iter()
        .zip(target.iter())
        .map(|(c, t)| {
            let d = Rational::from(c - t);
            Rational::from(&d * &d)
        })
        .sum();
    assert_eq!(dist, v.norm);
    // Every lattice point in a generous window must be at least as far away.
    for a in -6i64..=6 {
        for b in -3i64..=3 {
            let x = Rational::from(a) + Rational::from(10 * b);
            let y = Rational::from(b);
            let dx = Rational::from(&x - &target[0]);
            let dy = Rational::from(&y - &target[1]);
            let d = Rational::from(&dx * &dx) + Rational::from(&dy * &dy);
            assert!(
                d >= v.norm,
                "({a},{b}) is closer than the reported CVP answer"
            );
        }
    }
}

#[test]
fn closest_vector_needs_ambient_coordinates() {
    let e = e8().unwrap();
    let gram_only = Lattice::from_gram(e.gram_matrix()).unwrap();
    let t = vec![Rational::from(0); 8];
    assert!(matches!(
        gram_only.closest_vector(&t),
        Err(LatticeError::NoBasis)
    ));
    let z = zn(2).unwrap();
    assert!(matches!(
        z.closest_vector(&[Rational::from(1)]),
        Err(LatticeError::DimensionMismatch { .. })
    ));
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn refusals_are_typed_and_coded() {
    use crate::errors::AlkahestError;

    // Rank above the enumeration ceiling.
    let big: Vec<Vec<Integer>> = (0..MAX_ENUM_RANK + 1)
        .map(|i| {
            (0..MAX_ENUM_RANK + 1)
                .map(|j| Integer::from(i64::from(i == j)))
                .collect()
        })
        .collect();
    let l = Lattice::from_basis(&big).unwrap();
    let err = l.minimum().unwrap_err();
    assert!(matches!(err, LatticeError::RankTooLarge { .. }));
    assert_eq!(err.code(), "E-LAT-008");

    // Node budget.
    let z = zn(6).unwrap();
    let err = z.theta_series_with_budget(400, 50).unwrap_err();
    assert!(matches!(err, LatticeError::EnumerationBudget { .. }));
    assert_eq!(err.code(), "E-LAT-009");

    // Non-integral Gram matrix.
    let g = vec![
        vec![Rational::from((1, 3)), Rational::new()],
        vec![Rational::new(), Rational::from(1)],
    ];
    let l = Lattice::from_gram(&g).unwrap();
    let err = l.theta_series(4).unwrap_err();
    assert!(matches!(err, LatticeError::NonIntegralGram { .. }));
    assert_eq!(err.code(), "E-LAT-010");

    // Degenerate forms.
    let dependent = rows_i64(&[&[1, 2], &[2, 4]]);
    let err = Lattice::from_basis(&dependent).unwrap_err();
    assert!(matches!(err, LatticeError::NotPositiveDefinite { .. }));
    assert_eq!(err.code(), "E-LAT-007");

    let asym = vec![
        vec![Rational::from(1), Rational::from(2)],
        vec![Rational::from(3), Rational::from(1)],
    ];
    assert_eq!(Lattice::from_gram(&asym).unwrap_err().code(), "E-LAT-006");

    let nonsquare = vec![vec![Rational::from(1), Rational::from(0)]];
    assert_eq!(
        Lattice::from_gram(&nonsquare).unwrap_err().code(),
        "E-LAT-005"
    );

    assert_eq!(zn(0).unwrap_err().code(), "E-LAT-012");
    assert_eq!(a_n(0).unwrap_err().code(), "E-LAT-012");
    assert_eq!(d_n(1).unwrap_err().code(), "E-LAT-012");
    assert_eq!(
        zn(2)
            .unwrap()
            .theta_series(MAX_THETA_NORM + 1)
            .unwrap_err()
            .code(),
        "E-LAT-012"
    );
}

// ---------------------------------------------------------------------------
// LLL: defining properties, not a fixed matrix
// ---------------------------------------------------------------------------

/// Determinant of a square integer matrix, by exact fraction-free elimination.
fn det_abs(m: &[Vec<Integer>]) -> Rational {
    let n = m.len();
    let mut a: Vec<Vec<Rational>> = m
        .iter()
        .map(|r| r.iter().map(Rational::from).collect())
        .collect();
    let mut det = Rational::from(1);
    for i in 0..n {
        let Some(p) = (i..n).find(|&r| a[r][i] != 0) else {
            return Rational::new();
        };
        if p != i {
            a.swap(i, p);
            det = -det;
        }
        det *= &a[i][i].clone();
        let inv = a[i][i].clone();
        for r in (i + 1)..n {
            let f = Rational::from(&a[r][i] / &inv);
            let pivot_row: Vec<Rational> = a[i][i..n].to_vec();
            for (offset, ai) in pivot_row.iter().enumerate() {
                let sub = Rational::from(&f * ai);
                a[r][i + offset] -= sub;
            }
        }
    }
    det.abs()
}

#[test]
fn flint_lll_output_satisfies_the_exact_lll_inequalities() {
    // Bases where a naive reduction historically misbehaved, plus a genuinely
    // nasty one (a knapsack-style lattice with a huge last column).
    let cases = vec![
        rows_i64(&[&[1, 1], &[0, 1]]),
        rows_i64(&[&[2, 15], &[1, 21]]),
        rows_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]),
        rows_i64(&[&[1, 0, 0, 12345], &[0, 1, 0, 23456], &[0, 0, 1, 34567]]),
        rows_i64(&[
            &[1, 0, 0, 0, 1000000],
            &[0, 1, 0, 0, 1414213],
            &[0, 0, 1, 0, 1732050],
            &[0, 0, 0, 1, 2236067],
        ]),
    ];
    for basis in cases {
        for (num, den) in [(3u32, 4u32), (51, 100), (99, 100)] {
            let delta = Rational::from((num, den));
            let reduced = lattice_reduce_rows_with_delta(&basis, delta.clone()).unwrap();
            assert_eq!(reduced.len(), basis.len());
            validate_lll_rows(&reduced, &delta)
                .unwrap_or_else(|e| panic!("δ = {num}/{den} produced a non-reduced basis: {e}"));
            assert_eq!(
                det_abs(&reduced),
                det_abs(&basis),
                "reduction changed the lattice"
            );
        }
    }
}

/// `|det|` is a weak witness that a reduction preserved the lattice — two
/// different lattices can share one. The Hermite normal form is a *canonical*
/// basis, so equal HNFs is the real statement "these rows generate the same
/// lattice". A mis-declared `fmpz_mat` layout, which is the headline FFI hazard
/// here, would fail this long before it failed a determinant check.
#[test]
fn reduction_preserves_the_lattice_itself_not_merely_its_determinant() {
    let cases = vec![
        rows_i64(&[&[1, 1], &[0, 1]]),
        rows_i64(&[&[2, 15], &[1, 21]]),
        rows_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]),
        rows_i64(&[&[1, 0, 0, 12345], &[0, 1, 0, 23456], &[0, 0, 1, 34567]]),
        rows_i64(&[&[-7, 31, 0, 5], &[13, -2, 44, 1], &[0, 9, -3, 28]]),
    ];
    for basis in cases {
        let cols = basis[0].len();
        let reduced = lattice_reduce_rows(&basis).unwrap();
        let exact = lattice_reduce_rows_exact(&basis, Rational::from((3, 4))).unwrap();
        let hnf = |r: &[Vec<Integer>]| super::flint_backend::hnf_basis(r, cols);
        assert_eq!(
            hnf(&reduced),
            hnf(&basis),
            "FLINT reduction changed the lattice"
        );
        assert_eq!(
            hnf(&exact),
            hnf(&basis),
            "exact reduction changed the lattice"
        );
    }
}

#[test]
fn flint_and_exact_backends_agree_on_the_lattice() {
    let basis = rows_i64(&[&[1, 0, 0, 12345], &[0, 1, 0, 23456], &[0, 0, 1, 34567]]);
    let delta = Rational::from((3, 4));
    let via_flint = lattice_reduce_rows(&basis).unwrap();
    let via_exact = lattice_reduce_rows_exact(&basis, delta.clone()).unwrap();
    validate_lll_rows(&via_flint, &delta).unwrap();
    validate_lll_rows(&via_exact, &delta).unwrap();
    // Both span the same lattice as the input …
    let gram_det = |r: &[Vec<Integer>]| {
        Lattice::from_basis(r)
            .map(|l| l.determinant())
            .unwrap_or_else(|_| Rational::new())
    };
    assert_eq!(gram_det(&via_flint), gram_det(&basis));
    assert_eq!(gram_det(&via_exact), gram_det(&basis));
    // … and both reach the same (unique, here) shortest length.
    let m1 = Lattice::from_basis(&via_flint).unwrap().minimum().unwrap();
    let m2 = Lattice::from_basis(&via_exact).unwrap().minimum().unwrap();
    assert_eq!(m1, m2);
}

#[test]
fn lll_reduced_lattice_is_the_same_lattice() {
    let basis = rows_i64(&[&[11, 3, 0], &[7, 19, 0], &[0, 0, 5]]);
    let l = Lattice::from_basis(&basis).unwrap();
    let r = l.lll_reduced().unwrap();
    assert_eq!(r.determinant(), l.determinant());
    assert_eq!(r.minimum().unwrap(), l.minimum().unwrap());
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// LLL never changes the lattice, so it never changes |det|.
        #[test]
        fn reduction_preserves_the_determinant(
            entries in prop::collection::vec(-40i64..=40, 9)
        ) {
            let basis = rows_i64(&[
                &entries[0..3],
                &entries[3..6],
                &entries[6..9],
            ]);
            let before = det_abs(&basis);
            prop_assume!(before != 0);
            let reduced = lattice_reduce_rows(&basis).unwrap();
            prop_assert_eq!(det_abs(&reduced), before);
        }

        /// … and the output really is reduced, at the default δ = ¾.
        #[test]
        fn reduction_output_is_reduced(
            entries in prop::collection::vec(-40i64..=40, 9)
        ) {
            let basis = rows_i64(&[
                &entries[0..3],
                &entries[3..6],
                &entries[6..9],
            ]);
            prop_assume!(det_abs(&basis) != 0);
            let reduced = lattice_reduce_rows(&basis).unwrap();
            prop_assert!(validate_lll_rows(&reduced, &Rational::from((3, 4))).is_ok());
        }

        /// `(L*)* = L`.
        #[test]
        fn dual_is_an_involution(entries in prop::collection::vec(-12i64..=12, 4)) {
            let basis = rows_i64(&[&entries[0..2], &entries[2..4]]);
            prop_assume!(det_abs(&basis) != 0);
            let l = Lattice::from_basis(&basis).unwrap();
            let dd = l.dual().unwrap().dual().unwrap();
            prop_assert_eq!(dd.gram_matrix(), l.gram_matrix());
        }

        /// The zeroth theta coefficient counts the origin, and nothing else.
        #[test]
        fn theta_zero_is_one(entries in prop::collection::vec(-6i64..=6, 4)) {
            let basis = rows_i64(&[&entries[0..2], &entries[2..4]]);
            prop_assume!(det_abs(&basis) != 0);
            let l = Lattice::from_basis(&basis).unwrap();
            let theta = l.theta_series(4).unwrap();
            prop_assert_eq!(theta[0].clone(), Integer::from(1));
            // Every shell is centrally symmetric, so every later coefficient is
            // even.
            for c in theta.iter().skip(1) {
                prop_assert!(c.is_divisible_u(2));
            }
        }

        /// The shortest vector is at most as long as any basis vector, and the
        /// enumeration agrees with a brute-force scan of a small box.
        #[test]
        fn svp_matches_brute_force_in_the_plane(
            entries in prop::collection::vec(-9i64..=9, 4)
        ) {
            let basis = rows_i64(&[&entries[0..2], &entries[2..4]]);
            prop_assume!(det_abs(&basis) != 0);
            let l = Lattice::from_basis(&basis).unwrap();
            let min = l.minimum().unwrap();
            let mut brute: Option<Rational> = None;
            for a in -30i64..=30 {
                for b in -30i64..=30 {
                    if a == 0 && b == 0 {
                        continue;
                    }
                    let x = Rational::from(a * entries[0] + b * entries[2]);
                    let y = Rational::from(a * entries[1] + b * entries[3]);
                    let n = Rational::from(&x * &x) + Rational::from(&y * &y);
                    if brute.as_ref().is_none_or(|m| n < *m) {
                        brute = Some(n);
                    }
                }
            }
            prop_assert_eq!(min, brute.unwrap());
        }
    }
}
