//! Unit tests for the permutation-group module.
//!
//! The order tests are the load-bearing ones. `|S_n| = n!` and `|A_n| = n!/2`
//! catch a chain that is too short; the Mathieu groups catch a chain that is
//! subtly *wrong* in a way the symmetric groups never surface, because `M₁₁`
//! and `M₁₂` are sharply transitive in a way that makes the second and third
//! levels of the chain non-obvious.

use super::*;
use rug::Integer;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Every element of `g`, by breadth-first closure over the generators. Used to
/// cross-check the stabilizer chain on groups small enough to list twice.
fn brute_force_elements(g: &PermutationGroup) -> HashSet<Permutation> {
    let identity = Permutation::identity(g.degree());
    let mut seen = HashSet::new();
    seen.insert(identity.clone());
    let mut frontier = vec![identity];
    while let Some(p) = frontier.pop() {
        for generator in g.generators() {
            let q = p.compose(generator).expect("degrees match by construction");
            if seen.insert(q.clone()) {
                frontier.push(q);
            }
        }
    }
    seen
}

fn factorial(n: u32) -> Integer {
    let mut f = Integer::from(1);
    for k in 2..=n {
        f *= k;
    }
    f
}

/// `M₁₁` from the standard degree-11 generators (ATLAS / GAP, 1-based).
fn mathieu_11() -> PermutationGroup {
    let a = Permutation::from_cycles_one_based(11, &[(1..=11).collect()]).unwrap();
    let b =
        Permutation::from_cycles_one_based(11, &[vec![3, 7, 11, 8], vec![4, 10, 5, 6]]).unwrap();
    PermutationGroup::new(11, vec![a, b]).unwrap()
}

/// `M₁₂` from the standard degree-12 generators: `M₁₁`'s two, read on 12
/// points, plus the involution that fuses them.
fn mathieu_12() -> PermutationGroup {
    let a = Permutation::from_cycles_one_based(12, &[(1..=11).collect()]).unwrap();
    let b =
        Permutation::from_cycles_one_based(12, &[vec![3, 7, 11, 8], vec![4, 10, 5, 6]]).unwrap();
    let c = Permutation::from_cycles_one_based(
        12,
        &[
            vec![1, 12],
            vec![2, 11],
            vec![3, 6],
            vec![4, 8],
            vec![5, 9],
            vec![7, 10],
        ],
    )
    .unwrap();
    PermutationGroup::new(12, vec![a, b, c]).unwrap()
}

// ---------------------------------------------------------------------------
// Permutation basics
// ---------------------------------------------------------------------------

#[test]
fn identity_is_identity() {
    let e = Permutation::identity(5);
    assert!(e.is_identity());
    assert_eq!(e.degree(), 5);
    assert_eq!(e.images(), &[0, 1, 2, 3, 4]);
    assert_eq!(e.cycles(), Vec::<Vec<usize>>::new());
    assert_eq!(e.cycle_type(), vec![1, 1, 1, 1, 1]);
    assert_eq!(e.sign(), 1);
    assert_eq!(e.order(), Integer::from(1));
    assert_eq!(e.to_string(), "()");
}

#[test]
fn from_images_rejects_non_bijections() {
    let repeated = Permutation::from_images(vec![0, 0, 2]).unwrap_err();
    assert!(matches!(repeated, GroupError::NotAPermutation { .. }));
    let out_of_range = Permutation::from_images(vec![0, 3, 2]).unwrap_err();
    assert!(matches!(out_of_range, GroupError::NotAPermutation { .. }));
    use crate::errors::AlkahestError;
    assert_eq!(repeated.code(), "E-GRP-001");
}

#[test]
fn from_cycles_rejects_overlapping_cycles() {
    let err = Permutation::from_cycles(4, &[vec![0, 1], vec![1, 2]]).unwrap_err();
    assert!(matches!(err, GroupError::NotAPermutation { .. }));
    let err = Permutation::from_cycles(3, &[vec![0, 5]]).unwrap_err();
    assert!(matches!(err, GroupError::NotAPermutation { .. }));
}

#[test]
fn one_based_cycles_shift_by_one() {
    let zero = Permutation::from_cycles(5, &[vec![0, 2, 4]]).unwrap();
    let one = Permutation::from_cycles_one_based(5, &[vec![1, 3, 5]]).unwrap();
    assert_eq!(zero, one);
    assert!(Permutation::from_cycles_one_based(5, &[vec![0, 1]]).is_err());
    assert!(Permutation::from_cycles_one_based(5, &[vec![1, 6]]).is_err());
}

/// The single most important assertion in this module: composition is
/// left-to-right, `(p·q)(i) = q(p(i))`.
#[test]
fn composition_is_left_to_right() {
    let p = Permutation::from_cycles(3, &[vec![0, 1]]).unwrap();
    let q = Permutation::from_cycles(3, &[vec![1, 2]]).unwrap();

    let pq = p.compose(&q).unwrap();
    for i in 0..3 {
        assert_eq!(
            pq.apply(i).unwrap(),
            q.apply(p.apply(i).unwrap()).unwrap(),
            "(p·q)(i) must be q(p(i)) at i = {i}"
        );
    }
    // Spelled out: 0 -^p-> 1 -^q-> 2.
    assert_eq!(pq.images(), &[2, 0, 1]);
    // And the other order really is different, so the test above is not vacuous.
    assert_eq!(q.compose(&p).unwrap().images(), &[1, 2, 0]);
    assert_ne!(pq, q.compose(&p).unwrap());
}

#[test]
fn compose_refuses_degree_mismatch() {
    let p = Permutation::identity(3);
    let q = Permutation::identity(4);
    let err = p.compose(&q).unwrap_err();
    assert_eq!(err, GroupError::DegreeMismatch { left: 3, right: 4 });
    use crate::errors::AlkahestError;
    assert_eq!(err.code(), "E-GRP-002");
}

#[test]
fn inverse_and_powers() {
    let p = Permutation::from_images(vec![1, 2, 3, 4, 0]).unwrap();
    let i = p.inverse();
    assert!(p.compose(&i).unwrap().is_identity());
    assert!(i.compose(&p).unwrap().is_identity());
    assert_eq!(p.pow(0), Permutation::identity(5));
    assert_eq!(p.pow(1), p);
    assert_eq!(p.pow(-1), i);
    assert_eq!(p.pow(5), Permutation::identity(5));
    assert_eq!(p.pow(7), p.pow(2));
    assert_eq!(p.pow(-3), p.pow(2));
}

#[test]
fn cycles_and_cycle_type() {
    // (0 1)(3 4 5), with 2 fixed.
    let p = Permutation::from_images(vec![1, 0, 2, 4, 5, 3]).unwrap();
    assert_eq!(p.cycles(), vec![vec![0, 1], vec![3, 4, 5]]);
    assert_eq!(p.cycle_type(), vec![3, 2, 1]);
    assert_eq!(p.cycle_type().iter().sum::<usize>(), p.degree());
    assert_eq!(p.support(), vec![0, 1, 3, 4, 5]);
    assert_eq!(p.first_moved_point(), Some(0));
    assert_eq!(p.to_string(), "(0 1)(3 4 5)");
    // order = lcm(2, 3)
    assert_eq!(p.order(), Integer::from(6));
    assert_eq!(p.sign(), -1); // (2-1) + (3-1) = 3, odd
    assert!(!p.is_even());
}

#[test]
fn element_order_is_lcm_of_cycle_lengths() {
    // Disjoint cycles of lengths 2, 3, 5, 7, 11 on 28 of 30 points.
    let mut cycles = Vec::new();
    let mut next = 0usize;
    for len in [2usize, 3, 5, 7, 11] {
        cycles.push((next..next + len).collect::<Vec<_>>());
        next += len;
    }
    let p = Permutation::from_cycles(30, &cycles).unwrap();
    assert_eq!(p.order(), Integer::from(2 * 3 * 5 * 7 * 11));
    assert!(p.pow(2310).is_identity());
    assert!(!p.pow(1155).is_identity());
}

#[test]
fn sign_is_multiplicative_on_a_hand_example() {
    let p = Permutation::from_cycles(5, &[vec![0, 1]]).unwrap(); // odd
    let q = Permutation::from_cycles(5, &[vec![2, 3, 4]]).unwrap(); // even
    assert_eq!(p.sign(), -1);
    assert_eq!(q.sign(), 1);
    assert_eq!(p.compose(&q).unwrap().sign(), -1);
    assert_eq!(p.compose(&p).unwrap().sign(), 1);
}

#[test]
fn extend_degree_embeds_but_never_restricts() {
    let p = Permutation::from_cycles(3, &[vec![0, 1, 2]]).unwrap();
    let big = p.extend_degree(5).unwrap();
    assert_eq!(big.images(), &[1, 2, 0, 3, 4]);
    assert_ne!(big.degree(), p.degree());
    assert!(p.extend_degree(2).is_err());
}

#[test]
fn apply_refuses_out_of_range_points() {
    let p = Permutation::identity(3);
    let err = p.apply(3).unwrap_err();
    assert_eq!(
        err,
        GroupError::PointOutOfRange {
            point: 3,
            degree: 3
        }
    );
    use crate::errors::AlkahestError;
    assert_eq!(err.code(), "E-GRP-003");
}

// ---------------------------------------------------------------------------
// Orbits and Schreier vectors
// ---------------------------------------------------------------------------

#[test]
fn orbit_of_a_cyclic_group_is_everything() {
    let g = cyclic(7).unwrap();
    let orbit = g.orbit(0).unwrap();
    assert_eq!(orbit.len(), 7);
    assert_eq!(orbit.sorted_points(), (0..7).collect::<Vec<_>>());
    assert!(g.is_transitive());
}

#[test]
fn orbits_partition_the_points() {
    // (0 1 2) on the first three points, (3 4) on the next two, 5 fixed.
    let a = Permutation::from_cycles(6, &[vec![0, 1, 2]]).unwrap();
    let b = Permutation::from_cycles(6, &[vec![3, 4]]).unwrap();
    let g = PermutationGroup::new(6, vec![a, b]).unwrap();
    assert_eq!(g.orbits(), vec![vec![0, 1, 2], vec![3, 4], vec![5]]);
    assert!(!g.is_transitive());
    assert_eq!(g.order().unwrap(), Integer::from(6));
}

#[test]
fn schreier_vector_reproduces_the_stored_transversal() {
    for g in [
        symmetric(6).unwrap(),
        alternating(6).unwrap(),
        dihedral(7).unwrap(),
        mathieu_11(),
    ] {
        for point in 0..g.degree() {
            let orbit = g.orbit(point).unwrap();
            for &beta in orbit.points() {
                let stored = orbit.transversal_element(beta).unwrap();
                let walked = orbit
                    .transversal_from_schreier_vector(beta, g.generators())
                    .unwrap();
                assert_eq!(
                    stored, &walked,
                    "Schreier-vector walk disagreed with the stored transversal at {beta}"
                );
                // And it is a transversal element at all.
                assert_eq!(stored.apply(point).unwrap(), beta);
            }
            // Points outside the orbit have neither.
            for outside in 0..g.degree() {
                if !orbit.contains(outside) {
                    assert!(orbit.transversal_element(outside).is_none());
                    assert!(orbit
                        .transversal_from_schreier_vector(outside, g.generators())
                        .is_none());
                }
            }
        }
    }
}

#[test]
fn orbit_refuses_points_outside_the_degree() {
    let g = symmetric(4).unwrap();
    assert!(matches!(
        g.orbit(4).unwrap_err(),
        GroupError::PointOutOfRange { .. }
    ));
}

// ---------------------------------------------------------------------------
// Orders of the standard families
// ---------------------------------------------------------------------------

#[test]
fn symmetric_group_orders_are_factorials() {
    for n in 0..=8usize {
        let g = symmetric(n).unwrap();
        assert_eq!(
            g.order().unwrap(),
            factorial(n as u32),
            "|S_{n}| should be {n}!"
        );
        assert_eq!(g.degree(), n);
    }
}

#[test]
fn symmetric_group_order_needs_arbitrary_precision() {
    // 30! = 265252859812191058636308480000000 — nowhere near a u64.
    let g = symmetric(30).unwrap();
    assert_eq!(g.order().unwrap(), factorial(30));
    assert!(g.order().unwrap() > u64::MAX);
}

#[test]
fn alternating_group_orders_are_half_factorials() {
    for n in 2..=8usize {
        let g = alternating(n).unwrap();
        assert_eq!(
            g.order().unwrap(),
            factorial(n as u32) / 2u32,
            "|A_{n}| should be {n}!/2"
        );
    }
    assert_eq!(alternating(0).unwrap().order().unwrap(), Integer::from(1));
    assert_eq!(alternating(1).unwrap().order().unwrap(), Integer::from(1));
}

#[test]
fn alternating_group_contains_only_even_permutations() {
    let g = alternating(5).unwrap();
    let elements = g.elements().unwrap();
    assert_eq!(elements.len(), 60);
    assert!(elements.iter().all(|p| p.is_even()));
}

#[test]
fn cyclic_group_orders() {
    for n in 1..=12usize {
        let g = cyclic(n).unwrap();
        assert_eq!(g.order().unwrap(), Integer::from(n));
    }
    assert!(matches!(
        cyclic(0).unwrap_err(),
        GroupError::UnsupportedDegree { .. }
    ));
}

#[test]
fn dihedral_group_orders_are_twice_n() {
    for n in 3..=12usize {
        let g = dihedral(n).unwrap();
        assert_eq!(
            g.order().unwrap(),
            Integer::from(2 * n),
            "|D_{n}| should be {}",
            2 * n
        );
    }
    // D_3 is S_3.
    assert_eq!(
        dihedral(3).unwrap().order().unwrap(),
        symmetric(3).unwrap().order().unwrap()
    );
    let err = dihedral(2).unwrap_err();
    assert!(matches!(err, GroupError::UnsupportedDegree { .. }));
    use crate::errors::AlkahestError;
    assert_eq!(err.code(), "E-GRP-006");
}

#[test]
fn trivial_group_has_order_one() {
    let g = trivial(5);
    assert_eq!(g.order().unwrap(), Integer::from(1));
    assert!(g.is_trivial());
    assert_eq!(g.elements().unwrap(), vec![Permutation::identity(5)]);
    assert_eq!(g.stabilizer_chain().unwrap().base(), Vec::<usize>::new());
}

// ---------------------------------------------------------------------------
// The Mathieu groups — the tests that catch a wrong Schreier–Sims
// ---------------------------------------------------------------------------

#[test]
fn mathieu_11_has_order_7920() {
    let g = mathieu_11();
    assert_eq!(g.order().unwrap(), Integer::from(7920));
    assert!(g.is_transitive());
    // Sharply 4-transitive: 11·10·9·8 = 7920, so the first four basic orbits
    // of a chain based at four points multiply to the whole order.
    let chain = g.stabilizer_chain().unwrap();
    let product: usize = chain.levels().iter().map(|l| l.orbit().len()).product();
    assert_eq!(product, 7920);
}

#[test]
fn mathieu_11_agrees_with_brute_force_enumeration() {
    let g = mathieu_11();
    let brute = brute_force_elements(&g);
    assert_eq!(brute.len(), 7920);
    let listed = g.elements().unwrap();
    assert_eq!(listed.len(), 7920);
    let listed: HashSet<Permutation> = listed.into_iter().collect();
    assert_eq!(listed, brute, "chain enumeration differs from the closure");
}

#[test]
fn mathieu_12_has_order_95040() {
    let g = mathieu_12();
    assert_eq!(g.order().unwrap(), Integer::from(95_040));
    assert!(g.is_transitive());
    assert_eq!(g.degree(), 12);
}

#[test]
fn mathieu_12_membership_is_not_all_of_s12() {
    let g = mathieu_12();
    // |S_12| = 479001600, so M_12 is a very thin subgroup: a transposition is
    // not in it (M_12 is contained in A_12).
    let transposition = Permutation::from_cycles(12, &[vec![0, 1]]).unwrap();
    assert!(!g.contains(&transposition).unwrap());
    // Every generator is, and so is every product of a few of them.
    for generator in g.generators() {
        assert!(g.contains(generator).unwrap());
    }
    let product = g.generators()[0]
        .compose(&g.generators()[2])
        .unwrap()
        .compose(&g.generators()[1])
        .unwrap();
    assert!(g.contains(&product).unwrap());
}

// ---------------------------------------------------------------------------
// Membership, sifting, enumeration
// ---------------------------------------------------------------------------

#[test]
fn membership_agrees_with_brute_force_on_small_groups() {
    let ambient = symmetric(5).unwrap();
    let all_of_s5 = ambient.elements().unwrap();
    assert_eq!(all_of_s5.len(), 120);

    for g in [
        alternating(5).unwrap(),
        dihedral(5).unwrap(),
        cyclic(5).unwrap(),
        PermutationGroup::new(
            5,
            vec![Permutation::from_cycles(5, &[vec![0, 1], vec![2, 3]]).unwrap()],
        )
        .unwrap(),
    ] {
        let brute = brute_force_elements(&g);
        assert_eq!(
            Integer::from(brute.len()),
            g.order().unwrap(),
            "closure size and |G| disagree"
        );
        for p in &all_of_s5 {
            assert_eq!(
                g.contains(p).unwrap(),
                brute.contains(p),
                "membership of {p} disagreed with the closure"
            );
        }
    }
}

#[test]
fn a_random_element_sifts_to_the_identity() {
    for g in [
        symmetric(7).unwrap(),
        alternating(7).unwrap(),
        dihedral(9).unwrap(),
        mathieu_11(),
        mathieu_12(),
    ] {
        for seed in 0..40u64 {
            let p = g.random_element(seed).unwrap();
            let sift = g.sift(&p).unwrap();
            assert!(
                sift.is_member(),
                "a random element of the group failed its own membership test"
            );
            assert!(sift.residue().is_identity());
            assert_eq!(sift.level(), g.stabilizer_chain().unwrap().levels().len());
            assert!(g.contains(&p).unwrap());
        }
    }
}

#[test]
fn random_elements_are_spread_over_the_group() {
    // Not a distribution test — just enough to catch a generator that always
    // returns the identity or always the same coset.
    let g = symmetric(5).unwrap();
    let sample: HashSet<Permutation> = (0..400u64).map(|s| g.random_element(s).unwrap()).collect();
    assert!(
        sample.len() > 60,
        "only {} distinct elements out of 400 draws from S_5",
        sample.len()
    );
}

#[test]
fn a_non_member_sifts_to_a_non_identity_residue() {
    let g = alternating(5).unwrap();
    let odd = Permutation::from_cycles(5, &[vec![0, 1]]).unwrap();
    let sift = g.sift(&odd).unwrap();
    assert!(!sift.is_member());
    assert!(!sift.residue().is_identity());
    assert!(!g.contains(&odd).unwrap());
}

#[test]
fn sift_refuses_a_permutation_of_the_wrong_degree() {
    let g = symmetric(4).unwrap();
    let p = Permutation::identity(5);
    assert!(matches!(
        g.sift(&p).unwrap_err(),
        GroupError::DegreeMismatch { .. }
    ));
}

#[test]
fn enumeration_is_exactly_the_group_without_repeats() {
    for g in [
        symmetric(5).unwrap(),
        alternating(6).unwrap(),
        dihedral(8).unwrap(),
        cyclic(12).unwrap(),
    ] {
        let elements = g.elements().unwrap();
        let order = g.order().unwrap();
        assert_eq!(Integer::from(elements.len()), order);
        let unique: HashSet<Permutation> = elements.iter().cloned().collect();
        assert_eq!(unique.len(), elements.len(), "enumeration repeated itself");
        assert_eq!(unique, brute_force_elements(&g));
        for p in &elements {
            assert!(g.contains(p).unwrap());
        }
    }
}

#[test]
fn enumeration_refuses_above_the_cap_but_the_order_is_still_exact() {
    let g = symmetric(9).unwrap();
    assert_eq!(g.order().unwrap(), factorial(9)); // 362880
    let err = g.elements().unwrap_err();
    match &err {
        GroupError::EnumerationTooLarge { order, cap } => {
            assert_eq!(order, "362880");
            assert_eq!(*cap, DEFAULT_ELEMENT_CAP);
        }
        other => panic!("expected EnumerationTooLarge, got {other:?}"),
    }
    use crate::errors::AlkahestError;
    assert_eq!(err.code(), "E-GRP-004");
    // With a big enough cap it succeeds.
    assert_eq!(g.elements_with_cap(400_000).unwrap().len(), 362_880);
}

#[test]
fn enumeration_cap_is_itself_capped() {
    // 20! is astronomically above MAX_ELEMENT_CAP; asking for u64::MAX must not
    // start allocating.
    let g = symmetric(20).unwrap();
    let err = g.elements_with_cap(u64::MAX).unwrap_err();
    match err {
        GroupError::EnumerationTooLarge { cap, .. } => assert_eq!(cap, MAX_ELEMENT_CAP),
        other => panic!("expected EnumerationTooLarge, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The stabilizer chain itself
// ---------------------------------------------------------------------------

#[test]
fn chain_levels_are_stabilizers_of_the_preceding_base_points() {
    for g in [
        symmetric(6).unwrap(),
        alternating(7).unwrap(),
        mathieu_11(),
        mathieu_12(),
    ] {
        let chain = g.stabilizer_chain().unwrap();
        let base = chain.base();
        for (index, level) in chain.levels().iter().enumerate() {
            assert_eq!(level.base_point(), base[index]);
            for generator in level.generators() {
                for &b in &base[..index] {
                    assert_eq!(
                        generator.apply(b).unwrap(),
                        b,
                        "a level-{index} strong generator moves base point {b}"
                    );
                }
                assert!(g.contains(generator).unwrap());
            }
            assert_eq!(level.orbit().base_point(), level.base_point());
        }
        // |G| is the product of the basic orbit lengths.
        let mut product = Integer::from(1);
        for level in chain.levels() {
            product *= level.orbit().len() as u64;
        }
        assert_eq!(product, g.order().unwrap());
        assert!(!chain.strong_generators().is_empty());
    }
}

#[test]
fn orbit_stabilizer_theorem_on_s5() {
    let g = symmetric(5).unwrap();
    let order = g.order().unwrap();
    let orbit = g.orbit(0).unwrap();
    assert_eq!(orbit.len(), 5);
    let fixing_zero = g
        .elements()
        .unwrap()
        .into_iter()
        .filter(|p| p.apply(0).unwrap() == 0)
        .count();
    assert_eq!(
        Integer::from(fixing_zero) * Integer::from(orbit.len()),
        order
    );
}

#[test]
fn orbit_lengths_divide_the_group_order() {
    for g in [
        symmetric(6).unwrap(),
        alternating(6).unwrap(),
        dihedral(10).unwrap(),
        mathieu_11(),
        mathieu_12(),
    ] {
        let order = g.order().unwrap();
        for point in 0..g.degree() {
            let len = Integer::from(g.orbit(point).unwrap().len());
            assert!(
                order.is_divisible(&len),
                "orbit length {len} does not divide |G| = {order}"
            );
        }
    }
}

#[test]
fn the_chain_is_cached_and_stable() {
    let g = mathieu_11();
    let first = g.stabilizer_chain().unwrap().base();
    let second = g.stabilizer_chain().unwrap().base();
    assert_eq!(first, second);
    assert_eq!(g.order().unwrap(), Integer::from(7920));
}

#[test]
fn degree_above_the_bsgs_limit_is_refused_not_attempted() {
    let g = symmetric(MAX_BSGS_DEGREE + 1).unwrap();
    // Orbits still work at any degree.
    assert_eq!(g.orbit(0).unwrap().len(), MAX_BSGS_DEGREE + 1);
    let err = g.order().unwrap_err();
    assert!(matches!(err, GroupError::DegreeTooLargeForBsgs { .. }));
    use crate::errors::AlkahestError;
    assert_eq!(err.code(), "E-GRP-005");
    // And the refusal is cached rather than recomputed into a different answer.
    assert_eq!(g.order().unwrap_err(), err);
}

#[test]
fn group_construction_refuses_a_generator_of_the_wrong_degree() {
    let err = PermutationGroup::new(4, vec![Permutation::identity(5)]).unwrap_err();
    assert!(matches!(err, GroupError::DegreeMismatch { .. }));
    assert!(PermutationGroup::from_generators(Vec::new()).is_err());
    assert_eq!(
        PermutationGroup::from_generators(vec![Permutation::identity(6)])
            .unwrap()
            .degree(),
        6
    );
}

#[test]
fn a_group_generated_by_identities_is_trivial() {
    let g = PermutationGroup::new(4, vec![Permutation::identity(4)]).unwrap();
    assert!(g.is_trivial());
    assert_eq!(g.order().unwrap(), Integer::from(1));
    assert_eq!(g.stabilizer_chain().unwrap().levels().len(), 0);
}

#[test]
fn intransitive_subgroup_order_is_the_product_of_its_parts() {
    // S_3 on {0,1,2} times S_3 on {3,4,5}: order 36, and not transitive.
    let mut generators = Vec::new();
    for block in [[0usize, 1, 2], [3, 4, 5]] {
        generators.push(Permutation::from_cycles(6, &[vec![block[0], block[1]]]).unwrap());
        generators.push(Permutation::from_cycles(6, &[block.to_vec()]).unwrap());
    }
    let g = PermutationGroup::new(6, generators).unwrap();
    assert_eq!(g.order().unwrap(), Integer::from(36));
    assert!(!g.is_transitive());
    assert_eq!(g.orbits(), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    assert_eq!(
        HashSet::<Permutation>::from_iter(g.elements().unwrap()),
        brute_force_elements(&g)
    );
}

#[test]
fn a_subgroup_generated_by_one_element_is_its_cyclic_group() {
    let p = Permutation::from_cycles(10, &[vec![0, 1, 2, 3], vec![4, 5, 6]]).unwrap();
    let g = PermutationGroup::from_generators(vec![p.clone()]).unwrap();
    assert_eq!(g.order().unwrap(), p.order());
    assert_eq!(g.order().unwrap(), Integer::from(12));
    for k in 0..12i64 {
        assert!(g.contains(&p.pow(k)).unwrap());
    }
}

#[test]
fn mathieu_24_has_order_244823040() {
    // A third, larger Mathieu group, from the standard degree-24 generators.
    // Its order is 2^10·3^3·5·7·11·23, and no part of the chain for it is
    // guessable from the S_n cases: for these generators the base has seven
    // points and the basic orbits are 24, 23, 22, 21, 20, 16, 3 — whose
    // product is exactly the order asserted below.
    let a = Permutation::from_cycles_one_based(24, &[(1..=23).collect()]).unwrap();
    let b = Permutation::from_cycles_one_based(
        24,
        &[
            vec![3, 17, 10, 7, 9],
            vec![4, 13, 14, 19, 5],
            vec![8, 18, 11, 12, 23],
            vec![15, 20, 22, 21, 16],
        ],
    )
    .unwrap();
    let c = Permutation::from_cycles_one_based(
        24,
        &[
            vec![1, 24],
            vec![2, 23],
            vec![3, 12],
            vec![4, 16],
            vec![5, 18],
            vec![6, 10],
            vec![7, 20],
            vec![8, 14],
            vec![9, 21],
            vec![11, 17],
            vec![13, 22],
            vec![15, 19],
        ],
    )
    .unwrap();
    let g = PermutationGroup::new(24, vec![a, b, c]).unwrap();
    assert_eq!(g.order().unwrap(), Integer::from(244_823_040u64));
    assert!(g.is_transitive());
    // Far too big to list, and it says so rather than trying.
    assert!(matches!(
        g.elements().unwrap_err(),
        GroupError::EnumerationTooLarge { .. }
    ));
    for seed in 0..10u64 {
        assert!(g.contains(&g.random_element(seed).unwrap()).unwrap());
    }
}

