//! Property tests for conjugacy classes and character tables.
//!
//! The orthogonality relations are the strongest cheap invariant available
//! here: they are implied by the group axioms, so they hold for *every* group
//! and a random group either satisfies them or has found a bug. The
//! implementation already checks them before returning a table, so what these
//! tests really establish is that the checks are reachable and that no random
//! small group takes a path around them — including the paths where the answer
//! is a refusal, which must still be a typed one rather than a panic.

use super::*;
use crate::group::{Permutation, PermutationGroup};
use proptest::prelude::*;
use rug::Integer;

/// A permutation of `degree` points from a Lehmer code, so every generated
/// value is a bijection by construction.
fn arb_permutation(degree: usize) -> impl Strategy<Value = Permutation> {
    proptest::collection::vec(0usize..1_000, degree).prop_map(move |codes| {
        let mut pool: Vec<usize> = (0..degree).collect();
        let mut images = Vec::with_capacity(degree);
        for (i, code) in codes.iter().enumerate() {
            let index = code % (degree - i);
            images.push(pool.remove(index));
        }
        Permutation::from_images(images).expect("a Lehmer code always yields a bijection")
    })
}

/// A small group: degree 2–5 (so `|G| ≤ 120`), one to three generators.
fn arb_small_group() -> impl Strategy<Value = PermutationGroup> {
    (2usize..=5)
        .prop_flat_map(|n| {
            (
                Just(n),
                proptest::collection::vec(arb_permutation(n), 1..=3),
            )
        })
        .prop_map(|(n, generators)| {
            PermutationGroup::new(n, generators).expect("generators built at the group's degree")
        })
}

proptest! {
    /// Class sizes divide `|G|` and sum to it, and the centraliser orders are
    /// the matching cofactors.
    #[test]
    fn class_sizes_partition_the_group(group in arb_small_group()) {
        let classes = ConjugacyClasses::of(&group).unwrap();
        let order = classes.group_order().clone();
        let mut total = Integer::new();
        for class in classes.classes() {
            prop_assert!(order.is_divisible(&Integer::from(class.size())));
            prop_assert_eq!(
                class.centraliser_order().clone() * Integer::from(class.size()),
                order.clone()
            );
            prop_assert_eq!(classes.exponent() % class.element_order(), 0);
            total += Integer::from(class.size());
        }
        prop_assert_eq!(total, order.clone());
        prop_assert!(order.is_divisible(&Integer::from(classes.exponent())));
        prop_assert!(classes.classes()[0].representative().is_identity());
    }

    /// Conjugate elements land in one class, and every element of `G` is found.
    #[test]
    fn class_membership_is_conjugation_invariant(group in arb_small_group(), seed in 0u64..1_000) {
        let classes = ConjugacyClasses::of(&group).unwrap();
        let x = group.random_element(seed).unwrap();
        let y = group.random_element(seed.wrapping_mul(2_654_435_761).wrapping_add(1)).unwrap();
        let index = classes.class_of(&x).unwrap();
        let conjugate = y.inverse().compose(&x).unwrap().compose(&y).unwrap();
        prop_assert_eq!(classes.class_of(&conjugate).unwrap(), index);
        // And inverses really do land in the recorded inverse class.
        prop_assert_eq!(
            classes.class_of(&x.inverse()).unwrap(),
            classes.inverse_class(index).unwrap()
        );
    }

    /// `Σ_j a_{kij} |K_j| = |K_k| |K_i|`: the class multiplication matrix
    /// accounts for every pair exactly once.
    #[test]
    fn class_multiplication_conserves_mass(group in arb_small_group()) {
        let classes = ConjugacyClasses::of(&group).unwrap();
        let r = classes.len();
        for k in 0..r {
            let matrix = classes.multiplication_matrix(k).unwrap();
            prop_assert_eq!(matrix.len(), r);
            for (i, row) in matrix.iter().enumerate() {
                let total: u64 = row
                    .iter()
                    .enumerate()
                    .map(|(j, &a)| a * classes.class(j).unwrap().size())
                    .sum();
                prop_assert_eq!(
                    total,
                    classes.class(k).unwrap().size() * classes.class(i).unwrap().size()
                );
            }
        }
    }
}

proptest! {
    // The table is the expensive half — `O(r·|G|)` group products and `O(r³)`
    // multiplications in a cyclotomic field — so this block runs fewer cases.
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Every invariant a character table has to satisfy, on a random group.
    ///
    /// A refusal is an acceptable outcome and a panic is not, so the refusal
    /// branch asserts that the error is one of this module's typed caps.
    #[test]
    fn character_table_satisfies_the_orthogonality_relations(group in arb_small_group()) {
        match CharacterTable::of(&group) {
            Ok(table) => {
                let r = table.classes().len();
                prop_assert_eq!(table.len(), r);
                table.verify().unwrap();

                let order = table.classes().group_order().clone();
                let mut squares = Integer::new();
                for &d in table.degrees() {
                    prop_assert!(order.is_divisible(&Integer::from(d)));
                    squares += Integer::from(d) * Integer::from(d);
                }
                prop_assert_eq!(squares, order);

                // Degrees ascend, and the trivial character is row 0.
                prop_assert_eq!(table.degrees()[0], 1);
                let one = table.field().one();
                prop_assert!(table.character(0).unwrap().iter().all(|v| *v == one));
                for window in table.degrees().windows(2) {
                    prop_assert!(window[0] <= window[1]);
                }

                // Row orthogonality through the public inner product.
                for i in 0..r {
                    for j in 0..r {
                        let expected = rug::Rational::from(i32::from(i == j));
                        prop_assert_eq!(table.inner_product(i, j).unwrap(), expected);
                    }
                }

                // Each value is an algebraic integer of absolute value at most
                // the degree, which shows up as |χ(g)| ≤ χ(1) on the trace of
                // the class sum; the cheap shadow of it is that a degree-1
                // character has every value a root of unity.
                for i in 0..r {
                    if table.degrees()[i] == 1 {
                        for c in 0..r {
                            let m = table.classes().class(c).unwrap().element_order();
                            prop_assert_eq!(table.value(i, c).unwrap().pow(m), one.clone());
                        }
                    }
                }
            }
            Err(e) => {
                let code = crate::errors::AlkahestError::code(&e);
                prop_assert!(
                    ["E-CHAR-001", "E-CHAR-002", "E-CHAR-003"].contains(&code),
                    "unexpected refusal {code}: {e}"
                );
            }
        }
    }
}
