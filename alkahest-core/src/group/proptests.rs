//! Property tests for permutations and permutation groups.
//!
//! Each invariant here is cheap to state and expensive to get wrong silently:
//! `p·p⁻¹ = 1` pins the composition order against the inverse, the sign
//! homomorphism pins parity against composition, and "every orbit length
//! divides `|G|`" is an orbit-stabilizer consequence that a mis-built
//! stabilizer chain fails almost immediately.

use super::*;
use proptest::prelude::*;
use rug::Integer;
use std::collections::HashSet;

/// A uniform-ish permutation of `degree` points, built from a Lehmer code so
/// that every generated value is a genuine bijection by construction.
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

fn arb_degree_and_permutation() -> impl Strategy<Value = Permutation> {
    (1usize..=8).prop_flat_map(arb_permutation)
}

fn arb_pair() -> impl Strategy<Value = (Permutation, Permutation)> {
    (1usize..=8).prop_flat_map(|n| (arb_permutation(n), arb_permutation(n)))
}

fn arb_triple() -> impl Strategy<Value = (Permutation, Permutation, Permutation)> {
    (1usize..=7).prop_flat_map(|n| (arb_permutation(n), arb_permutation(n), arb_permutation(n)))
}

/// A small group: degree 3–6, one to three generators.
fn arb_small_group() -> impl Strategy<Value = PermutationGroup> {
    (3usize..=6)
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
    #[test]
    fn inverse_is_a_two_sided_inverse(p in arb_degree_and_permutation()) {
        let identity = Permutation::identity(p.degree());
        prop_assert_eq!(p.compose(&p.inverse()).unwrap(), identity.clone());
        prop_assert_eq!(p.inverse().compose(&p).unwrap(), identity);
        prop_assert_eq!(p.inverse().inverse(), p);
    }

    #[test]
    fn sign_is_a_homomorphism((p, q) in arb_pair()) {
        prop_assert_eq!(p.compose(&q).unwrap().sign(), p.sign() * q.sign());
        prop_assert_eq!(p.inverse().sign(), p.sign());
    }

    #[test]
    fn composition_is_associative((p, q, r) in arb_triple()) {
        let left = p.compose(&q).unwrap().compose(&r).unwrap();
        let right = p.compose(&q.compose(&r).unwrap()).unwrap();
        prop_assert_eq!(left, right);
    }

    #[test]
    fn inverse_of_a_product_reverses_it((p, q) in arb_pair()) {
        let lhs = p.compose(&q).unwrap().inverse();
        let rhs = q.inverse().compose(&p.inverse()).unwrap();
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn composition_applies_the_left_factor_first((p, q) in arb_pair()) {
        let pq = p.compose(&q).unwrap();
        for i in 0..p.degree() {
            prop_assert_eq!(pq.apply(i).unwrap(), q.apply(p.apply(i).unwrap()).unwrap());
        }
    }

    #[test]
    fn cycle_decomposition_reconstructs_the_permutation(p in arb_degree_and_permutation()) {
        let rebuilt = Permutation::from_cycles(p.degree(), &p.cycles()).unwrap();
        prop_assert_eq!(rebuilt, p.clone());
        prop_assert_eq!(p.cycle_type().iter().sum::<usize>(), p.degree());
    }

    #[test]
    fn element_order_is_minimal_and_annihilating(p in arb_degree_and_permutation()) {
        let order = p.order();
        let order_i64 = order.to_i64().expect("degree <= 8, so the order is tiny");
        prop_assert!(p.pow(order_i64).is_identity());
        for k in 1..order_i64 {
            prop_assert!(!p.pow(k).is_identity(), "{} had a smaller order than {}", p, order);
        }
    }

    #[test]
    fn powers_add_exponents(p in arb_degree_and_permutation(), a in -6i64..=6, b in -6i64..=6) {
        prop_assert_eq!(p.pow(a).compose(&p.pow(b)).unwrap(), p.pow(a + b));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn orbit_lengths_divide_the_group_order(g in arb_small_group()) {
        let order = g.order().unwrap();
        for point in 0..g.degree() {
            let orbit = g.orbit(point).unwrap();
            prop_assert!(order.is_divisible(&Integer::from(orbit.len())));
            // Every transversal element really carries the base point where it says.
            for &beta in orbit.points() {
                let u = orbit.transversal_element(beta).unwrap();
                prop_assert_eq!(u.apply(point).unwrap(), beta);
            }
        }
    }

    #[test]
    fn orbits_partition_the_point_set(g in arb_small_group()) {
        let orbits = g.orbits();
        let mut all: Vec<usize> = orbits.iter().flatten().copied().collect();
        all.sort_unstable();
        prop_assert_eq!(all, (0..g.degree()).collect::<Vec<_>>());
    }

    #[test]
    fn generators_and_their_products_are_members(g in arb_small_group()) {
        for generator in g.generators() {
            prop_assert!(g.contains(generator).unwrap());
            prop_assert!(g.contains(&generator.inverse()).unwrap());
        }
        let mut product = Permutation::identity(g.degree());
        for (i, generator) in g.generators().iter().cycle().take(7).enumerate() {
            product = product.compose(generator).unwrap();
            prop_assert!(g.contains(&product).unwrap(), "product of {} generators escaped", i + 1);
        }
    }

    #[test]
    fn random_elements_sift_to_the_identity(g in arb_small_group(), seed in any::<u64>()) {
        let p = g.random_element(seed).unwrap();
        let sift = g.sift(&p).unwrap();
        prop_assert!(sift.is_member());
        prop_assert!(sift.residue().is_identity());
    }

    #[test]
    fn enumeration_matches_the_order_and_is_closed(g in arb_small_group()) {
        let elements = g.elements().unwrap();
        prop_assert_eq!(Integer::from(elements.len()), g.order().unwrap());
        let set: HashSet<Permutation> = elements.iter().cloned().collect();
        prop_assert_eq!(set.len(), elements.len());
        // Closed under composition with a generator, and under inverses.
        for p in &elements {
            prop_assert!(set.contains(&p.inverse()));
            for generator in g.generators() {
                prop_assert!(set.contains(&p.compose(generator).unwrap()));
            }
        }
    }

    #[test]
    fn membership_is_exactly_membership_in_the_enumeration(
        g in arb_small_group(),
        p in (3usize..=6).prop_flat_map(arb_permutation),
    ) {
        prop_assume!(p.degree() == g.degree());
        let set: HashSet<Permutation> = g.elements().unwrap().into_iter().collect();
        prop_assert_eq!(g.contains(&p).unwrap(), set.contains(&p));
    }

    #[test]
    fn schreier_vectors_agree_with_the_stored_transversal(g in arb_small_group()) {
        for point in 0..g.degree() {
            let orbit = g.orbit(point).unwrap();
            for &beta in orbit.points() {
                let walked = orbit
                    .transversal_from_schreier_vector(beta, g.generators())
                    .unwrap();
                prop_assert_eq!(orbit.transversal_element(beta).unwrap(), &walked);
            }
        }
    }
}
