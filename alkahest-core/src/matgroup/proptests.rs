//! Property tests for matrix groups over GF(q).
//!
//! Each invariant is cheap to state and expensive to get wrong silently.
//!
//! * `|G|` is a multiple of the order of every element — a stabilizer chain
//!   that dropped a level reports a proper divisor of `|G|`, and an element of
//!   full order in the dropped part fails this immediately.
//! * Every orbit length divides `|G|` (orbit–stabilizer).
//! * Sifting any element of the group — including a product-replacement random
//!   one — leaves the identity, and sifting factorises the element exactly.
//! * Membership agrees between the chain and a rebuild from the strong
//!   generators.
//!
//! The groups are generated from **random invertible matrices**, so nothing in
//! the classical constructors can mask a bug in the chain.

use super::element::{entry_key, is_identity};
use super::*;
use crate::ffield::{FiniteField, GfMatrix};
use proptest::prelude::*;
use rug::Integer;

/// A random matrix over GF(p) built from a list of entry codes, retried by
/// elementary row operations until it is invertible.
///
/// Rather than rejecting singular draws (which would bias towards small
/// degrees), the matrix is forced invertible by construction: start from the
/// identity and apply a sequence of transvections and row scalings, each of
/// which is invertible, so the product is.
fn arb_invertible(p: u64, degree: usize) -> impl Strategy<Value = GfMatrix> {
    proptest::collection::vec((0usize..degree, 0usize..degree, 1u64..p), 1..=6).prop_map(
        move |ops| {
            let field = FiniteField::prime(p).expect("p is prime in the test parameters");
            let mut m = GfMatrix::identity(&field, degree).expect("identity exists");
            for (i, j, c) in ops {
                let lambda = field.scalar(c);
                let mut entries = vec![field.zero(); degree * degree];
                for d in 0..degree {
                    entries[d * degree + d] = field.one();
                }
                if i == j {
                    // A row scaling by a non-zero c.
                    entries[i * degree + i] = lambda;
                } else {
                    entries[i * degree + j] = lambda;
                }
                let factor = GfMatrix::from_elements(&field, degree, degree, &entries)
                    .expect("shape is degree x degree");
                m = m.mul(&factor).expect("same shape and field");
            }
            m
        },
    )
}

/// A group from one to three random invertible matrices over GF(p), `p` in
/// `{2, 3, 5}` and degree in `2..=3` — small enough that `order()` and
/// `elements()` both finish, large enough to have non-trivial chains.
fn arb_group() -> impl Strategy<Value = MatGroup> {
    (prop_oneof![Just(2u64), Just(3u64), Just(5u64)], 2usize..=3)
        .prop_flat_map(|(p, degree)| {
            (
                Just(p),
                Just(degree),
                proptest::collection::vec(arb_invertible(p, degree), 1..=3),
            )
        })
        .prop_map(|(p, degree, generators)| {
            let field = FiniteField::prime(p).expect("p is prime");
            MatGroup::new(&field, degree, generators)
                .expect("every generator is invertible by construction")
        })
}

/// The multiplicative order of `m`, by repeated multiplication. Bounded by
/// `limit` so a bug cannot hang the test.
fn element_order(group: &MatGroup, m: &GfMatrix, limit: u64) -> Option<u64> {
    let field = group.field();
    let mut x = m.clone();
    let mut order = 1u64;
    while !is_identity(field, &x) {
        x = x.mul(m).expect("same shape and field");
        order += 1;
        if order > limit {
            return None;
        }
    }
    Some(order)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// `|G|` is a multiple of the order of every element of `G`.
    #[test]
    fn order_is_a_multiple_of_every_generator_order(group in arb_group()) {
        let order = group.order().expect("small group");
        for g in group.generators() {
            if let Some(n) = element_order(&group, g, 10_000) {
                prop_assert!(
                    order.is_divisible(&Integer::from(n)),
                    "element order {n} does not divide |G| = {order}"
                );
            }
        }
    }

    /// The same, for product-replacement random elements — which also checks
    /// that they really are in the group.
    #[test]
    fn random_elements_are_members_of_known_order(group in arb_group(), seed in any::<u64>()) {
        let order = group.order().expect("small group");
        for m in group.random_elements(seed, 3).expect("matrix arithmetic") {
            prop_assert!(group.contains(&m).expect("shape is right by construction"));
            let sift = group.sift(&m).expect("shape is right");
            prop_assert!(sift.is_member());
            prop_assert!(is_identity(group.field(), sift.residue()));
            if let Some(n) = element_order(&group, &m, 10_000) {
                prop_assert!(order.is_divisible(&Integer::from(n)));
            }
        }
    }

    /// Every orbit length divides `|G|`, on vectors and on projective points.
    #[test]
    fn orbit_lengths_divide_the_order(group in arb_group()) {
        let order = group.order().expect("small group");
        for orbit in group.vector_orbits().expect("small field and degree") {
            prop_assert!(order.is_divisible(&Integer::from(orbit.len())));
        }
        for orbit in group.projective_orbits().expect("small field and degree") {
            prop_assert!(order.is_divisible(&Integer::from(orbit.len())));
        }
    }

    /// Orbits partition the non-zero vectors: every vector is in exactly one.
    #[test]
    fn vector_orbits_partition_the_non_zero_vectors(group in arb_group()) {
        let all = group.nonzero_vectors().expect("small field and degree");
        let orbits = group.vector_orbits().expect("small field and degree");
        let total: usize = orbits.iter().map(|o| o.len()).sum();
        prop_assert_eq!(total, all.len());
        let mut seen = std::collections::HashSet::new();
        for orbit in &orbits {
            for v in orbit {
                prop_assert!(seen.insert(entry_key(group.field(), v)), "a vector in two orbits");
            }
        }
        prop_assert_eq!(seen.len(), all.len());
    }

    /// The chain's element list is closed under multiplication and inverses,
    /// has exactly `|G|` distinct members, and every member is recognised.
    #[test]
    fn the_element_list_is_a_group(group in arb_group()) {
        let order = group.order().expect("small group");
        prop_assume!(order <= 2_000);
        let elements = group.elements().expect("below the cap");
        prop_assert_eq!(Integer::from(elements.len()), order.clone());
        let keys: std::collections::HashSet<Vec<u64>> =
            elements.iter().map(|m| entry_key(group.field(), m)).collect();
        prop_assert_eq!(keys.len(), elements.len());
        // Closure, on a sample rather than the whole multiplication table.
        for (i, a) in elements.iter().enumerate().take(6) {
            prop_assert!(keys.contains(&entry_key(group.field(), &a.inverse().expect("invertible"))));
            for b in elements.iter().skip(i).take(6) {
                let product = a.mul(b).expect("same shape and field");
                prop_assert!(keys.contains(&entry_key(group.field(), &product)));
            }
        }
    }

    /// Rebuilding from the strong generators gives the same group.
    #[test]
    fn strong_generators_regenerate_the_group(group in arb_group()) {
        let order = group.order().expect("small group");
        let strong = group.strong_generators().expect("small group");
        let rebuilt = MatGroup::new(group.field(), group.degree(), strong)
            .expect("strong generators are invertible");
        prop_assert_eq!(rebuilt.order().expect("small group"), order);
        for g in group.generators() {
            prop_assert!(rebuilt.contains(g).expect("same shape"));
        }
    }

    /// The derived subgroup is a subgroup, its order divides `|G|`, and the
    /// abelianisation is abelian — i.e. every commutator is in `G'`.
    #[test]
    fn the_derived_subgroup_contains_every_commutator(group in arb_group()) {
        let order = group.order().expect("small group");
        prop_assume!(order <= 2_000);
        let derived = group.derived_subgroup().expect("small group");
        let derived_order = derived.order().expect("small group");
        prop_assert!(order.is_divisible(&derived_order));
        for a in group.generators() {
            for b in group.generators() {
                let c = super::element::commutator(a, b).expect("invertible");
                prop_assert!(derived.contains(&c).expect("same shape"));
            }
        }
    }

    /// The centre is central, abelian, and its order divides `|G|`.
    #[test]
    fn the_centre_is_central(group in arb_group()) {
        let order = group.order().expect("small group");
        prop_assume!(order <= 2_000);
        let central = group.centre_elements().expect("small group");
        prop_assert!(!central.is_empty(), "the centre always contains the identity");
        prop_assert!(order.is_divisible(&Integer::from(central.len())));
        for z in &central {
            prop_assert!(group.contains(z).expect("same shape"));
            for g in group.generators() {
                prop_assert!(
                    super::element::commutes(z, g).expect("same shape"),
                    "a claimed central element does not commute with a generator"
                );
            }
        }
    }

    /// The induced action on non-zero vectors is faithful, so the permutation
    /// group's order — computed by a completely separate Schreier–Sims — equals
    /// the matrix group's.
    #[test]
    fn the_permutation_action_reproduces_the_order(group in arb_group()) {
        let order = group.order().expect("small group");
        let action = group.permutation_action_on_vectors().expect("small field and degree");
        prop_assume!(action.degree() <= crate::group::MAX_BSGS_DEGREE);
        prop_assert_eq!(action.order().expect("degree is under the cap"), order);
    }

    /// The normal closure of the generators is the group itself, and the normal
    /// closure of anything is a normal subgroup whose order divides `|G|`.
    #[test]
    fn normal_closure_is_normal(group in arb_group()) {
        let order = group.order().expect("small group");
        prop_assume!(order <= 2_000);
        let whole = group.normal_closure(group.generators()).expect("small group");
        prop_assert_eq!(whole.order().expect("small group"), order.clone());

        let seed_element = group.generators()[0].clone();
        let closure = group.normal_closure(&[seed_element]).expect("small group");
        let closure_order = closure.order().expect("small group");
        prop_assert!(order.is_divisible(&closure_order));
        for g in group.generators() {
            for h in closure.generators() {
                let c = super::element::conjugate(g, h).expect("invertible");
                prop_assert!(closure.contains(&c).expect("same shape"), "closure is not normal");
            }
        }
    }
}
