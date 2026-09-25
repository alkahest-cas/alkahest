//! Property tests for `coding`.
//!
//! The invariants here are the cheap ones to state and the ones a subtly wrong
//! enumeration or transform would break first: `A_0 = 1`, `Σ A_i = q^k`,
//! `k + (n−k) = n`, MacWilliams is an involution that really does send a code's
//! enumerator to its *independently enumerated* dual's, and the Delsarte bound
//! never sits below a code that exists.
//!
//! The codes are random subspaces rather than named families on purpose: the
//! named families are all highly structured, and a bug that only shows up on an
//! unstructured code would hide behind them.

use proptest::prelude::*;
use rug::ops::Pow;
use rug::Integer;

use super::*;
use crate::ffield::{FiniteField, GfMatrix};

/// A random `rows × cols` matrix over GF(p), `p` prime and small.
fn random_matrix(p: u64, rows: usize, cols: usize) -> impl Strategy<Value = GfMatrix> {
    prop::collection::vec(0u64..p, rows * cols).prop_map(move |entries| {
        let f = FiniteField::prime(p).unwrap();
        GfMatrix::from_u64(&f, rows, cols, &entries).unwrap()
    })
}

/// A random binary code of length `4..=9` and at most 4 generators.
fn random_binary_code() -> impl Strategy<Value = LinearCode> {
    (4usize..=9, 1usize..=4).prop_flat_map(|(n, k)| {
        random_matrix(2, k, n).prop_map(|g| LinearCode::from_generator(&g).unwrap())
    })
}

/// A random code over GF(2), GF(3) or GF(5), kept small enough to enumerate
/// both it and its dual.
fn random_small_code() -> impl Strategy<Value = LinearCode> {
    (
        prop::sample::select(vec![2u64, 3, 5]),
        3usize..=6,
        1usize..=3,
    )
        .prop_flat_map(|(p, n, k)| {
            random_matrix(p, k.min(n), n).prop_map(|g| LinearCode::from_generator(&g).unwrap())
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// `A_0 = 1`, every `A_i ≥ 0`, and `Σ A_i = q^k`.
    #[test]
    fn weight_distribution_sums_to_the_code_size(code in random_small_code()) {
        let a = code.weight_distribution().unwrap();
        prop_assert_eq!(a.len(), code.length() + 1);
        prop_assert_eq!(a[0].clone(), Integer::from(1));
        prop_assert!(a.iter().all(|c| *c >= 0));
        let total = a.iter().fold(Integer::from(0), |acc, c| acc + c);
        prop_assert_eq!(total, code.size());
    }

    /// Rank–nullity, carried into the code: `k + (n − k) = n`, and `H·Gᵀ = 0`.
    #[test]
    fn dimensions_and_the_defining_relation_agree(code in random_small_code()) {
        prop_assert_eq!(code.dimension() + code.redundancy(), code.length());
        let prod = code.parity_check().mul(&code.generator().transpose()).unwrap();
        prop_assert!(prod.is_zero());
    }

    /// `(C⊥)⊥ = C`, on the nose once both are in reduced row echelon form.
    #[test]
    fn dual_is_an_involution(code in random_small_code()) {
        let back = code.dual().dual();
        prop_assert!(back.generator().equals(code.generator()));
    }

    /// MacWilliams sends a code's enumerator to its dual's, both ways, and is
    /// therefore an involution.
    #[test]
    fn macwilliams_matches_the_enumerated_dual(code in random_small_code()) {
        let wc = code.weight_enumerator().unwrap();
        let wd = code.dual().weight_enumerator().unwrap();
        prop_assert_eq!(wc.macwilliams().unwrap(), wd.clone());
        prop_assert_eq!(wd.macwilliams().unwrap(), wc.clone());
        prop_assert_eq!(wc.macwilliams().unwrap().macwilliams().unwrap(), wc);
    }

    /// The Delsarte dual-feasibility constraints hold for every real code —
    /// they are the hypothesis the linear programme optimises under.
    #[test]
    fn real_codes_are_dual_feasible(code in random_small_code()) {
        let w = code.weight_enumerator().unwrap();
        prop_assert!(w.is_dual_feasible());
    }

    /// The minimum distance is the least index with a non-zero multiplicity,
    /// and every codeword of that weight really is a codeword.
    #[test]
    fn minimum_distance_is_the_first_non_zero_weight(code in random_binary_code()) {
        let a = code.weight_distribution().unwrap();
        let expected = a.iter().skip(1).position(|c| *c > 0).map(|i| i + 1);
        prop_assert_eq!(code.minimum_distance().unwrap(), expected);
    }

    /// Extending a binary code adds one coordinate, keeps the dimension, and
    /// never lowers the minimum distance.
    #[test]
    fn extension_preserves_dimension_and_cannot_shorten_the_distance(
        code in random_binary_code()
    ) {
        let ext = code.extend().unwrap();
        prop_assert_eq!(ext.length(), code.length() + 1);
        prop_assert_eq!(ext.dimension(), code.dimension());
        if let (Some(d0), Some(d1)) = (
            code.minimum_distance().unwrap(),
            ext.minimum_distance().unwrap(),
        ) {
            prop_assert!(d1 >= d0);
            prop_assert!(d1 <= d0 + 1);
        }
    }

    /// The whole point: the certified bound must never rule out a code that
    /// exists, and must never beat Singleton or Hamming.
    #[test]
    fn delsarte_admits_every_code_and_respects_the_elementary_bounds(
        code in random_binary_code()
    ) {
        let Some(d) = code.minimum_distance().unwrap() else {
            return Ok(());
        };
        let n = code.length();
        let b = delsarte_lp_bound(n, d, 2).unwrap();
        prop_assert!(b.verify_certificate());
        prop_assert!(b.verify_distribution());
        prop_assert!(*b.bound() >= code.size());
        prop_assert!(*b.bound() <= singleton_bound(n, d, 2).unwrap());
        prop_assert!(*b.bound() <= hamming_bound(n, d, 2).unwrap());
    }

    /// The Krawtchouk recurrence, over a wider range than the unit test walks.
    #[test]
    fn krawtchouk_recurrence(n in 1usize..=14, k in 0usize..14, x in -4i64..20, q in 2u64..=6) {
        prop_assume!(k < n);
        let lhs = Integer::from(k as u32 + 1) * krawtchouk(k + 1, x, n, q);
        let coeff = Integer::from(k as u64)
            + Integer::from((q - 1) * (n - k) as u64)
            - Integer::from(q) * Integer::from(x);
        let mut rhs = coeff * krawtchouk(k, x, n, q);
        if k > 0 {
            rhs -= Integer::from((q - 1) * (n - k + 1) as u64) * krawtchouk(k - 1, x, n, q);
        }
        prop_assert_eq!(lhs, rhs);
    }

    /// `Σ_k K_k(i) = q^n [i = 0]` — the identity that makes the programme
    /// bounded, and hence the identity that makes a refusal-free run possible.
    #[test]
    fn krawtchouk_column_sums(n in 1usize..=12, q in 2u64..=6) {
        for i in 0..=n {
            let total = (0..=n).fold(Integer::from(0), |a, k| a + krawtchouk(k, i as i64, n, q));
            let expected = if i == 0 {
                Integer::from(q).pow(n as u32)
            } else {
                Integer::from(0)
            };
            prop_assert_eq!(total, expected);
        }
    }
}
