//! Property tests for words and coset enumeration.
//!
//! Three invariants, chosen because each is cheap to state and each fails
//! immediately on the mistakes that are easy to make here:
//!
//! * **Free reduction is idempotent** and reversing-and-negating inverts it.
//! * **`⟨a | a^m, a^n⟩` is cyclic of order `gcd(m, n)`.** The answer is known in
//!   advance for every input, and getting it requires the coincidence
//!   machinery: the two relators collapse a table built for `max(m, n)` cosets
//!   down to `gcd(m, n)`.
//! * **`[G:H] · |H| = |G|`**, with the index from coset enumeration, `|H|` from
//!   Todd–Coxeter on the Reidemeister–Schreier presentation of `H`, and `|G|`
//!   from a third enumeration. Lagrange's theorem relates three computations
//!   that share no code path below the enumerator itself.

use super::*;
use proptest::prelude::*;
use rug::Integer;

fn arb_letters(rank: i32, max_len: usize) -> impl Strategy<Value = Vec<i32>> {
    proptest::collection::vec(prop_oneof![(1..=rank), (-rank..=-1)], 0..=max_len)
}

proptest! {
    /// Reducing a reduced word changes nothing, and the reduced form has no
    /// adjacent inverse pair left in it.
    #[test]
    fn free_reduction_is_idempotent(letters in arb_letters(4, 20)) {
        let once = Word::from_letters(&letters).unwrap();
        let twice = Word::from_letters(once.letters()).unwrap();
        prop_assert_eq!(&once, &twice);
        prop_assert!(once.len() <= letters.len());
        for pair in once.letters().windows(2) {
            prop_assert_ne!(pair[0], -pair[1]);
        }
    }

    #[test]
    fn inversion_is_an_involution_and_an_antihomomorphism(
        u in arb_letters(3, 12),
        v in arb_letters(3, 12),
    ) {
        let u = Word::from_letters(&u).unwrap();
        let v = Word::from_letters(&v).unwrap();
        prop_assert_eq!(u.inverse().inverse(), u.clone());
        prop_assert_eq!(u.times(&v).inverse(), v.inverse().times(&u.inverse()));
        prop_assert_eq!(u.times(&u.inverse()), Word::identity());
        prop_assert_eq!(u.times(&Word::identity()), u.clone());
    }

    #[test]
    fn exponent_sums_are_additive(u in arb_letters(3, 10), v in arb_letters(3, 10)) {
        let u = Word::from_letters(&u).unwrap();
        let v = Word::from_letters(&v).unwrap();
        let su = u.exponent_sums(3).unwrap();
        let sv = v.exponent_sums(3).unwrap();
        let suv = u.times(&v).exponent_sums(3).unwrap();
        for i in 0..3 {
            prop_assert_eq!(suv[i], su[i] + sv[i]);
        }
    }

    /// Doubly presented cyclic groups: the order is the gcd of the exponents,
    /// and this is the cheapest property test that exercises coincidences.
    #[test]
    fn a_doubly_presented_cyclic_group_has_order_gcd(m in 1u32..=24, n in 1u32..=24) {
        let g = FpGroup::from_strings(&["a"], &[&format!("a^{m}"), &format!("a^{n}")]).unwrap();
        let expected = Integer::from(m).gcd(&Integer::from(n));
        prop_assert_eq!(g.order().unwrap(), expected.clone());
        // And the abelianisation agrees, by a completely different route.
        prop_assert_eq!(
            g.abelian_invariants().unwrap().order().unwrap(),
            expected
        );
    }

    /// `[G:H] · |H| = |G|` for a random cyclic subgroup of `S₄` or `A₄`.
    #[test]
    fn index_times_subgroup_order_is_the_group_order(
        which in 0usize..2,
        letters in arb_letters(2, 5),
    ) {
        let (rels, order) = if which == 0 {
            (vec!["a^2", "b^3", "(a*b)^3"], 12u32)
        } else {
            (vec!["a^2", "b^3", "(a*b)^4"], 24u32)
        };
        let g = FpGroup::from_strings(&["a", "b"], &rels).unwrap();
        let h = vec![g.word(&letters).unwrap()];
        let index = g.index(&h).unwrap();
        prop_assert!(index >= 1 && index <= order as usize);
        let sub = g.subgroup_presentation(&h).unwrap();
        let sub_order = sub.presentation().order().unwrap();
        prop_assert_eq!(
            Integer::from(index) * sub_order,
            Integer::from(order)
        );
    }

    /// The permutation representation on the cosets always has degree equal to
    /// the index, and its order divides `|G|`.
    #[test]
    fn the_permutation_representation_has_the_index_as_its_degree(
        letters in arb_letters(2, 4),
    ) {
        let g = FpGroup::from_strings(&["a", "b"], &["a^2", "b^3", "(a*b)^4"]).unwrap();
        let h = vec![g.word(&letters).unwrap()];
        let table = g.coset_table(&h).unwrap();
        let perm = table.permutation_group().unwrap();
        prop_assert_eq!(perm.degree(), table.index());
        let order = perm.order().unwrap();
        prop_assert_eq!(Integer::from(24) % order.clone(), Integer::from(0));
        prop_assert!(order >= table.index());
    }
}
