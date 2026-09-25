//! Unit tests for [`super::MatGroup`].
//!
//! The spine of this file is one assertion repeated across families and fields:
//! **the order computed by Schreier–Sims from the generators equals the closed
//! form**. Neither side is allowed to be derived from the other. The generating
//! sets in [`super::classical`] are theory-backed and complete rather than
//! pruned until the order matched, so the comparison is a real check on the
//! chain; and the formulas are independently cross-checked against
//! [`crate::stabilizer::MatrixGroup`], which computes them in different code.

use super::element::{field_elements, is_identity};
use super::*;
use crate::errors::AlkahestError;
use crate::ffield::{FiniteField, GfMatrix};
use crate::group::MAX_BSGS_DEGREE;
use crate::stabilizer::{is_symplectic, MatrixGroup};
use rug::ops::Pow;
use rug::Integer;

fn gf(p: u64) -> FiniteField {
    FiniteField::prime(p).unwrap()
}

fn gfk(p: u64, k: u32) -> FiniteField {
    FiniteField::extension(p, k).unwrap()
}

fn q_of(field: &FiniteField) -> Integer {
    Integer::from(field.order().unwrap())
}

/// Every `n × n` matrix over GF(q), for the brute-force cross-checks.
fn all_matrices(field: &FiniteField, n: usize) -> Vec<GfMatrix> {
    let alphabet = field_elements(field).unwrap();
    let q = alphabet.len();
    let cells = n * n;
    let total = q.pow(cells as u32);
    let mut out = Vec::with_capacity(total);
    for code in 0..total {
        let mut rest = code;
        let mut entries = Vec::with_capacity(cells);
        for _ in 0..cells {
            entries.push(alphabet[rest % q].clone());
            rest /= q;
        }
        out.push(GfMatrix::from_elements(field, n, n, &entries).unwrap());
    }
    out
}

// ---------------------------------------------------------------------------
// The correctness anchors
// ---------------------------------------------------------------------------

#[test]
fn the_four_headline_orders() {
    assert_eq!(
        MatGroup::general_linear(&gf(2), 3)
            .unwrap()
            .order()
            .unwrap(),
        168
    );
    assert_eq!(
        MatGroup::special_linear(&gf(5), 2)
            .unwrap()
            .order()
            .unwrap(),
        120
    );
    assert_eq!(
        MatGroup::symplectic(&gf(2), 2).unwrap().order().unwrap(),
        720
    );
    assert_eq!(
        MatGroup::general_linear(&gfk(2, 2), 2)
            .unwrap()
            .order()
            .unwrap(),
        180
    );
}

#[test]
fn gl_order_from_the_chain_matches_the_product_formula() {
    for field in [gf(2), gf(3), gf(5), gf(7), gfk(2, 2), gfk(3, 2), gfk(2, 3)] {
        let q = q_of(&field);
        for n in 1..=3usize {
            // Skip what the orbit cap refuses rather than asserting a refusal
            // here; `refusals` covers that side.
            if q.clone().pow(n as u32) > MAX_MATGROUP_ORBIT as u64 {
                continue;
            }
            let g = MatGroup::general_linear(&field, n).unwrap();
            assert_eq!(
                g.order().unwrap(),
                gl_order(&q, n),
                "GL({n}, {q}) order disagreed with the formula"
            );
        }
    }
}

#[test]
fn sl_order_from_the_chain_matches_the_product_formula() {
    for field in [gf(2), gf(3), gf(5), gf(7), gfk(2, 2), gfk(3, 2)] {
        let q = q_of(&field);
        for n in 1..=3usize {
            if q.clone().pow(n as u32) > MAX_MATGROUP_ORBIT as u64 {
                continue;
            }
            let g = MatGroup::special_linear(&field, n).unwrap();
            assert_eq!(
                g.order().unwrap(),
                sl_order(&q, n),
                "SL({n}, {q}) order disagreed with the formula"
            );
        }
    }
}

#[test]
fn sp_order_from_the_chain_matches_the_product_formula() {
    for field in [gf(2), gf(3), gf(5), gfk(2, 2)] {
        let q = q_of(&field);
        for n in 1..=2usize {
            if q.clone().pow((2 * n) as u32) > MAX_MATGROUP_ORBIT as u64 {
                continue;
            }
            let g = MatGroup::symplectic(&field, n).unwrap();
            assert_eq!(
                g.order().unwrap(),
                sp_order(&q, n),
                "Sp({}, {q}) order disagreed with the formula",
                2 * n
            );
        }
    }
    // Sp(2, q) = SL(2, q) is the identity the formulas must also satisfy.
    for field in [gf(2), gf(3), gf(5), gf(7)] {
        let q = q_of(&field);
        assert_eq!(sp_order(&q, 1), sl_order(&q, 2));
        assert_eq!(
            MatGroup::symplectic(&field, 1).unwrap().order().unwrap(),
            MatGroup::special_linear(&field, 2)
                .unwrap()
                .order()
                .unwrap()
        );
    }
}

/// The closed forms in this module and the ones in `stabilizer::matrix_group`
/// are separate code; a typo in either is invisible until they are compared.
#[test]
fn the_formulas_agree_with_the_older_classical_surface() {
    for field in [gf(2), gf(3), gf(5), gfk(2, 2)] {
        let q = q_of(&field);
        for n in 1..=4usize {
            assert_eq!(
                MatrixGroup::general_linear(&field, n).unwrap().order(),
                gl_order(&q, n)
            );
            assert_eq!(
                MatrixGroup::special_linear(&field, n).unwrap().order(),
                sl_order(&q, n)
            );
            assert_eq!(
                MatrixGroup::symplectic(&field, n).unwrap().order(),
                sp_order(&q, n)
            );
        }
    }
}

/// `PSL(2, 7) ≅ GL(3, 2)`, both of order 168 — and the first is reached through
/// the projective action, the second directly.
#[test]
fn psl_2_7_and_gl_3_2_both_have_order_168() {
    let sl = MatGroup::special_linear(&gf(7), 2).unwrap();
    assert_eq!(sl.order().unwrap(), 336);
    let psl = sl.permutation_action_on_projective_points().unwrap();
    assert_eq!(psl.degree(), 8, "PG(1, 7) has 8 points");
    assert_eq!(psl.order().unwrap(), 168);
    assert_eq!(
        MatGroup::general_linear(&gf(2), 3)
            .unwrap()
            .order()
            .unwrap(),
        168
    );
    // Both are simple non-abelian of order 168, hence perfect.
    assert!(MatGroup::general_linear(&gf(2), 3)
        .unwrap()
        .is_perfect()
        .unwrap());
}

/// A Singer cycle: nothing about `qⁿ − 1` resembles the classical formulas, so
/// this exercises the chain on a group the formulas cannot have leaked into.
#[test]
fn singer_cycles_are_cyclic_of_order_q_to_the_n_minus_one() {
    for (p, n, expected) in [
        (2u64, 1usize, 1u64),
        (2, 2, 3),
        (2, 3, 7),
        (2, 4, 15),
        (2, 5, 31),
        (2, 6, 63),
        (3, 2, 8),
        (3, 3, 26),
        (5, 2, 24),
        (7, 2, 48),
        (11, 2, 120),
    ] {
        let field = gf(p);
        let g = MatGroup::singer_cycle(&field, n).unwrap();
        assert_eq!(
            g.order().unwrap(),
            expected,
            "Singer cycle in GL({n}, {p}) had the wrong order"
        );
        // Cyclic, hence abelian, hence its own centre and trivially derived.
        assert_eq!(g.centre().unwrap().order().unwrap(), expected);
        assert_eq!(g.derived_subgroup().unwrap().order().unwrap(), 1);
        // And it really is inside GL(n, p).
        if n > 1 {
            assert_eq!(g.generators().len(), 1);
        }
        for m in g.elements().unwrap() {
            assert!(!m.determinant().unwrap().is_zero());
        }
    }
}

/// The centre of `GL(n, q)` is the scalars, of order `q − 1`.
#[test]
fn the_centre_of_gl_is_the_scalars() {
    for (field, n) in [
        (gf(2), 2),
        (gf(2), 3),
        (gf(3), 2),
        (gf(3), 3),
        (gf(5), 2),
        (gf(7), 2),
        (gfk(2, 2), 2),
        (gfk(3, 2), 2),
    ] {
        let q = q_of(&field);
        let gl = MatGroup::general_linear(&field, n).unwrap();
        let centre = gl.centre().unwrap();
        assert_eq!(
            centre.order().unwrap(),
            q.clone() - Integer::from(1),
            "Z(GL({n}, {q})) should be the q-1 scalars"
        );
        // Every central element is a scalar matrix.
        for z in centre.centre_elements().unwrap() {
            let entries = z.to_elements();
            let d = z.nrows();
            let diagonal = entries[0].clone();
            for i in 0..d {
                for j in 0..d {
                    let e = &entries[i * d + j];
                    if i == j {
                        assert_eq!(*e, diagonal);
                    } else {
                        assert!(e.is_zero());
                    }
                }
            }
        }
        // The centralizing algebra of the natural GL-module is the scalars.
        assert_eq!(gl.commutant_basis().unwrap().len(), 1);
    }
}

/// `GL(n, q)' = SL(n, q)` for most `(n, q)`, and `GL(2, 2)` is the exception.
#[test]
fn derived_subgroups() {
    for (field, n) in [
        (gf(2), 3),
        (gf(3), 2),
        (gf(3), 3),
        (gf(5), 2),
        (gf(7), 2),
        (gfk(2, 2), 2),
    ] {
        let q = q_of(&field);
        let gl = MatGroup::general_linear(&field, n).unwrap();
        assert_eq!(
            gl.derived_subgroup().unwrap().order().unwrap(),
            sl_order(&q, n),
            "GL({n}, {q})' should be SL({n}, {q})"
        );
    }
    // GL(2, 2) = S_3 has derived subgroup A_3, of order 3, not SL(2, 2) = S_3.
    let gl22 = MatGroup::general_linear(&gf(2), 2).unwrap();
    assert_eq!(gl22.order().unwrap(), 6);
    assert_eq!(gl22.derived_subgroup().unwrap().order().unwrap(), 3);

    // SL(2, 5) is the double cover of A_5 and is perfect.
    let sl25 = MatGroup::special_linear(&gf(5), 2).unwrap();
    assert_eq!(sl25.order().unwrap(), 120);
    assert!(sl25.is_perfect().unwrap());
    assert_eq!(sl25.derived_subgroup().unwrap().order().unwrap(), 120);
    // Its centre is {+-I}.
    assert_eq!(sl25.centre().unwrap().order().unwrap(), 2);
    // SL(2, 3) is *not* perfect: its derived subgroup is the quaternion group.
    let sl23 = MatGroup::special_linear(&gf(3), 2).unwrap();
    assert_eq!(sl23.order().unwrap(), 24);
    assert_eq!(sl23.derived_subgroup().unwrap().order().unwrap(), 8);
}

/// Membership against brute-force enumeration, over every matrix of the right
/// shape — including the singular ones, which must be `false` and not an error.
#[test]
fn membership_matches_brute_force() {
    for (field, n) in [(gf(2), 2), (gf(3), 2), (gf(2), 3), (gfk(2, 2), 2)] {
        let gl = MatGroup::general_linear(&field, n).unwrap();
        let sl = MatGroup::special_linear(&field, n).unwrap();
        let one = field.one();
        let mut gl_count = 0u64;
        let mut sl_count = 0u64;
        for m in all_matrices(&field, n) {
            let det = m.determinant().unwrap();
            let in_gl = !det.is_zero();
            let in_sl = det == one;
            assert_eq!(gl.contains(&m).unwrap(), in_gl, "GL membership disagreed");
            assert_eq!(sl.contains(&m).unwrap(), in_sl, "SL membership disagreed");
            gl_count += u64::from(in_gl);
            sl_count += u64::from(in_sl);
        }
        assert_eq!(gl.order().unwrap(), gl_count);
        assert_eq!(sl.order().unwrap(), sl_count);
    }
}

/// `Sp(2n, q)` built from symplectic transvections here and the predicate in
/// `stabilizer::symplectic` must describe the same subgroup. The generators are
/// checked one by one, and then the whole group by counting.
#[test]
fn symplectic_agrees_with_the_stabilizer_predicate() {
    for (field, n) in [(gf(2), 1), (gf(2), 2), (gf(3), 1), (gf(3), 2), (gf(5), 1)] {
        let sp = MatGroup::symplectic(&field, n).unwrap();
        for g in sp.generators() {
            assert!(
                is_symplectic(g).unwrap(),
                "a generator of Sp({}, {}) failed the stabilizer predicate",
                2 * n,
                q_of(&field)
            );
        }
        let order = sp.order().unwrap();
        assert_eq!(order, sp_order(&q_of(&field), n));
        // Every element is symplectic, and there are as many of them as the
        // formula says, so the two sets coincide.
        if order <= 1000 {
            for m in sp.elements().unwrap() {
                assert!(is_symplectic(&m).unwrap());
            }
        }
    }
    // Sp(4, 2) exhaustively, both ways round.
    let field = gf(2);
    let sp = MatGroup::symplectic(&field, 2).unwrap();
    let mut count = 0u64;
    for m in all_matrices(&field, 4) {
        let predicate = is_symplectic(&m).unwrap();
        assert_eq!(sp.contains(&m).unwrap(), predicate);
        count += u64::from(predicate);
    }
    assert_eq!(count, 720);
}

// ---------------------------------------------------------------------------
// The chain itself
// ---------------------------------------------------------------------------

#[test]
fn the_base_lives_in_the_standard_basis_and_orbit_lengths_multiply_to_the_order() {
    for group in [
        MatGroup::general_linear(&gf(2), 3).unwrap(),
        MatGroup::special_linear(&gf(3), 3).unwrap(),
        MatGroup::symplectic(&gf(3), 2).unwrap(),
        MatGroup::general_linear(&gfk(2, 2), 2).unwrap(),
        MatGroup::singer_cycle(&gf(2), 4).unwrap(),
    ] {
        let chain = group.stabilizer_chain().unwrap();
        let base = chain.base();
        assert!(base.len() <= group.degree(), "at most d levels");
        let mut seen = std::collections::HashSet::new();
        for b in &base {
            assert!(*b < group.degree());
            assert!(seen.insert(*b), "a base point appeared twice");
        }
        let mut product = Integer::from(1);
        for level in chain.levels() {
            product *= level.orbit().len();
            assert_eq!(level.base_index(), level.orbit().base_index());
            // The transversal really is a transversal.
            for (i, point) in level.orbit().points().iter().enumerate() {
                let u = level.orbit().transversal_element(i).unwrap();
                let base_vector =
                    super::element::basis_vector(group.field(), group.degree(), level.base_index())
                        .unwrap();
                assert!(base_vector.mul(u).unwrap().equals(point));
            }
        }
        assert_eq!(product, group.order().unwrap());
    }
}

#[test]
fn strong_generators_generate_the_same_group() {
    for group in [
        MatGroup::general_linear(&gf(3), 2).unwrap(),
        MatGroup::symplectic(&gf(2), 2).unwrap(),
        MatGroup::special_linear(&gf(5), 2).unwrap(),
    ] {
        let strong = group.strong_generators().unwrap();
        let rebuilt = MatGroup::new(group.field(), group.degree(), strong).unwrap();
        assert_eq!(rebuilt.order().unwrap(), group.order().unwrap());
        for g in group.generators() {
            assert!(rebuilt.contains(g).unwrap());
        }
    }
}

#[test]
fn sifting_an_element_of_the_group_gives_the_identity() {
    let group = MatGroup::special_linear(&gf(5), 2).unwrap();
    for m in group.elements().unwrap() {
        let sift = group.sift(&m).unwrap();
        assert!(sift.is_member());
        assert!(is_identity(group.field(), sift.residue()));
        assert_eq!(
            sift.level(),
            group.stabilizer_chain().unwrap().levels().len()
        );
    }
    // And an element outside it does not.
    let gl = MatGroup::general_linear(&gf(5), 2).unwrap();
    let outside = gl
        .elements()
        .unwrap()
        .into_iter()
        .find(|m| m.determinant().unwrap() != gf(5).one())
        .unwrap();
    assert!(!group.contains(&outside).unwrap());
    assert!(!group.sift(&outside).unwrap().is_member());
}

#[test]
fn element_enumeration_has_exactly_order_distinct_matrices() {
    for group in [
        MatGroup::general_linear(&gf(3), 2).unwrap(),
        MatGroup::symplectic(&gf(2), 2).unwrap(),
        MatGroup::singer_cycle(&gf(3), 3).unwrap(),
        MatGroup::trivial(&gf(7), 3).unwrap(),
    ] {
        let elements = group.elements().unwrap();
        let order = group.order().unwrap();
        assert_eq!(Integer::from(elements.len()), order);
        let distinct: std::collections::HashSet<Vec<u64>> = elements
            .iter()
            .map(|m| super::element::entry_key(group.field(), m))
            .collect();
        assert_eq!(distinct.len(), elements.len(), "duplicate elements listed");
        for m in &elements {
            assert!(group.contains(m).unwrap());
        }
    }
}

#[test]
fn the_trivial_group_is_a_group() {
    let t = MatGroup::trivial(&gf(5), 3).unwrap();
    assert!(t.is_trivial());
    assert_eq!(t.order().unwrap(), 1);
    assert_eq!(t.base().unwrap(), Vec::<usize>::new());
    assert_eq!(t.elements().unwrap().len(), 1);
    assert!(t.contains(&t.identity().unwrap()).unwrap());
    assert_eq!(t.centre().unwrap().order().unwrap(), 1);
    assert_eq!(t.derived_subgroup().unwrap().order().unwrap(), 1);
    // The commutant of no generators is all of M_d(q).
    assert_eq!(t.commutant_basis().unwrap().len(), 9);
    // A group generated by the identity alone is the same thing.
    let explicit = MatGroup::new(&gf(5), 3, vec![t.identity().unwrap()]).unwrap();
    assert!(explicit.is_trivial());
    assert_eq!(explicit.order().unwrap(), 1);
}

// ---------------------------------------------------------------------------
// Orbits
// ---------------------------------------------------------------------------

#[test]
fn gl_is_transitive_on_non_zero_vectors_and_on_projective_points() {
    for (field, n) in [(gf(2), 3), (gf(3), 2), (gf(5), 2), (gfk(2, 2), 2)] {
        let q = field.order().unwrap() as usize;
        let gl = MatGroup::general_linear(&field, n).unwrap();
        let vector_orbits = gl.vector_orbits().unwrap();
        assert_eq!(
            vector_orbits.len(),
            1,
            "GL is transitive on non-zero vectors"
        );
        assert_eq!(vector_orbits[0].len(), q.pow(n as u32) - 1);

        let projective_orbits = gl.projective_orbits().unwrap();
        assert_eq!(projective_orbits.len(), 1);
        assert_eq!(
            projective_orbits[0].len(),
            (q.pow(n as u32) - 1) / (q - 1),
            "PG({}, {q}) point count",
            n - 1
        );
        assert_eq!(
            gl.projective_points().unwrap().len(),
            projective_orbits[0].len()
        );
    }
}

#[test]
fn orbit_lengths_divide_the_group_order() {
    for group in [
        MatGroup::singer_cycle(&gf(3), 3).unwrap(),
        MatGroup::special_linear(&gf(3), 2).unwrap(),
        MatGroup::symplectic(&gf(2), 2).unwrap(),
        borel_subgroup_of_gl2(&gf(5)),
    ] {
        let order = group.order().unwrap();
        for orbit in group.vector_orbits().unwrap() {
            assert!(
                order.is_divisible(&Integer::from(orbit.len())),
                "orbit length {} does not divide |G| = {order}",
                orbit.len()
            );
        }
        for orbit in group.projective_orbits().unwrap() {
            assert!(order.is_divisible(&Integer::from(orbit.len())));
        }
    }
}

#[test]
fn the_zero_vector_is_fixed_and_has_no_projective_point() {
    let group = MatGroup::general_linear(&gf(3), 2).unwrap();
    let zero = GfMatrix::zeros(&gf(3), 1, 2).unwrap();
    assert_eq!(group.vector_orbit(&zero).unwrap().len(), 1);
    let err = group.projective_orbit(&zero).unwrap_err();
    assert_eq!(err.code(), "E-MATGRP-012");
}

/// The Borel subgroup of `GL(2, q)` — upper triangular invertible matrices — is
/// the standard example of an *arbitrary* generated subgroup: order
/// `q·(q−1)²`, and nothing about it comes from a classical formula.
fn borel_subgroup_of_gl2(field: &FiniteField) -> MatGroup {
    let one = field.one();
    let zero = field.zero();
    let a = super::element::primitive_element(field).unwrap();
    let unipotent = GfMatrix::from_elements(
        field,
        2,
        2,
        &[one.clone(), one.clone(), zero.clone(), one.clone()],
    )
    .unwrap();
    let diag_left = GfMatrix::from_elements(
        field,
        2,
        2,
        &[a.clone(), zero.clone(), zero.clone(), one.clone()],
    )
    .unwrap();
    let diag_right =
        GfMatrix::from_elements(field, 2, 2, &[one.clone(), zero.clone(), zero, a]).unwrap();
    MatGroup::new(field, 2, vec![unipotent, diag_left, diag_right]).unwrap()
}

#[test]
fn an_arbitrary_generated_subgroup_the_borel() {
    for p in [3u64, 5, 7, 11] {
        let field = gf(p);
        let borel = borel_subgroup_of_gl2(&field);
        let expected = Integer::from(p) * Integer::from((p - 1) * (p - 1));
        assert_eq!(
            borel.order().unwrap(),
            expected,
            "|B| should be q(q-1)^2 for q = {p}"
        );
        // Every element is upper triangular, and the count says that is all of
        // them.
        for m in borel.elements().unwrap() {
            assert!(m.entry(1, 0).unwrap().is_zero());
        }
        // B fixes the line spanned by e_0 and is transitive on the other q
        // projective points of PG(1, q).
        let mut lengths: Vec<usize> = borel
            .projective_orbits()
            .unwrap()
            .iter()
            .map(|o| o.len())
            .collect();
        lengths.sort_unstable();
        assert_eq!(lengths, vec![1, p as usize]);
        // Its derived subgroup is the unipotent radical, of order q.
        assert_eq!(borel.derived_subgroup().unwrap().order().unwrap(), p);
    }
}

// ---------------------------------------------------------------------------
// Induced permutation actions — the reuse of `crate::group`
// ---------------------------------------------------------------------------

#[test]
fn the_action_on_vectors_is_faithful_and_reproduces_the_order() {
    for group in [
        MatGroup::general_linear(&gf(2), 3).unwrap(),
        MatGroup::special_linear(&gf(3), 2).unwrap(),
        MatGroup::symplectic(&gf(2), 2).unwrap(),
        MatGroup::general_linear(&gfk(2, 2), 2).unwrap(),
        MatGroup::singer_cycle(&gf(3), 3).unwrap(),
    ] {
        let action = group.permutation_action_on_vectors().unwrap();
        assert!(
            action.degree() <= MAX_BSGS_DEGREE,
            "this test only covers degrees the permutation chain accepts"
        );
        assert_eq!(
            action.order().unwrap(),
            group.order().unwrap(),
            "the action on vectors must be faithful"
        );
    }
}

#[test]
fn the_projective_action_quotients_by_the_scalars_it_contains() {
    for (field, n) in [(gf(5), 2), (gf(7), 2), (gf(3), 3), (gfk(2, 2), 2)] {
        for group in [
            MatGroup::general_linear(&field, n).unwrap(),
            MatGroup::special_linear(&field, n).unwrap(),
        ] {
            let scalars = group.centre_elements().unwrap().len();
            let action = group.permutation_action_on_projective_points().unwrap();
            assert_eq!(
                action.order().unwrap() * Integer::from(scalars),
                group.order().unwrap(),
                "|G/(G cap scalars)| for degree {n} over {field}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Random elements
// ---------------------------------------------------------------------------

#[test]
fn random_elements_are_in_the_group_and_reproducible() {
    for group in [
        MatGroup::general_linear(&gf(3), 3).unwrap(),
        MatGroup::symplectic(&gf(3), 2).unwrap(),
        MatGroup::singer_cycle(&gf(2), 5).unwrap(),
        MatGroup::trivial(&gf(5), 2).unwrap(),
    ] {
        for seed in [0u64, 1, 42, u64::MAX] {
            let a = group.random_element(seed).unwrap();
            let b = group.random_element(seed).unwrap();
            assert!(a.equals(&b), "the same seed must give the same element");
            assert!(group.contains(&a).unwrap());
        }
        let batch = group.random_elements(7, 20).unwrap();
        assert_eq!(batch.len(), 20);
        for m in &batch {
            assert!(group.contains(m).unwrap());
        }
    }
}

/// Not a uniformity test — product replacement's distribution is a heuristic —
/// but a non-degeneracy one: on a group of order 480 a run of 40 samples that
/// returned one matrix would mean the accumulator was not moving.
#[test]
fn random_elements_are_not_all_the_same() {
    let group = MatGroup::general_linear(&gf(5), 2).unwrap();
    let distinct: std::collections::HashSet<Vec<u64>> = group
        .random_elements(11, 40)
        .unwrap()
        .iter()
        .map(|m| super::element::entry_key(group.field(), m))
        .collect();
    assert!(
        distinct.len() > 20,
        "only {} distinct samples",
        distinct.len()
    );
}

// ---------------------------------------------------------------------------
// Normal closure
// ---------------------------------------------------------------------------

#[test]
fn normal_closure_of_a_single_transvection_is_sl() {
    // A single elementary transvection normally generates SL(n, q) inside
    // GL(n, q) for n >= 2, q > 3 -- the standard fact behind SL's simplicity
    // modulo its centre.
    for (field, n) in [(gf(5), 2), (gf(7), 2), (gf(3), 3), (gf(2), 3)] {
        let q = q_of(&field);
        let gl = MatGroup::general_linear(&field, n).unwrap();
        let one = field.one();
        let mut entries = vec![field.zero(); n * n];
        for d in 0..n {
            entries[d * n + d] = one.clone();
        }
        entries[1] = one;
        let t = GfMatrix::from_elements(&field, n, n, &entries).unwrap();
        assert!(gl.contains(&t).unwrap());
        let closure = gl.normal_closure(&[t]).unwrap();
        assert_eq!(
            closure.order().unwrap(),
            sl_order(&q, n),
            "the normal closure of a transvection in GL({n}, {q}) should be SL"
        );
    }
}

#[test]
fn normal_closure_of_the_centre_is_the_centre_and_of_nothing_is_trivial() {
    let gl = MatGroup::general_linear(&gf(5), 2).unwrap();
    let centre = gl.centre_elements().unwrap();
    let closure = gl.normal_closure(&centre).unwrap();
    assert_eq!(closure.order().unwrap(), 4);
    assert_eq!(gl.normal_closure(&[]).unwrap().order().unwrap(), 1);
    // The normal closure of the whole group is the whole group.
    let whole = gl.normal_closure(gl.generators()).unwrap();
    assert_eq!(whole.order().unwrap(), gl.order().unwrap());
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn refusals() {
    let field = gf(3);

    // Degree 0 and degree past the ceiling.
    assert_eq!(
        MatGroup::trivial(&field, 0).unwrap_err().code(),
        "E-MATGRP-005"
    );
    assert_eq!(
        MatGroup::trivial(&field, MAX_MATGROUP_DEGREE + 1)
            .unwrap_err()
            .code(),
        "E-MATGRP-005"
    );
    assert_eq!(
        MatGroup::general_linear(&field, 0).unwrap_err().code(),
        "E-MATGRP-010"
    );
    assert_eq!(
        MatGroup::symplectic(&field, 0).unwrap_err().code(),
        "E-MATGRP-010"
    );
    assert_eq!(
        MatGroup::symplectic(&field, MAX_MATGROUP_DEGREE)
            .unwrap_err()
            .code(),
        "E-MATGRP-005"
    );

    // A non-square generator, a generator of the wrong degree, a singular one,
    // and one over another field.
    let oblong = GfMatrix::zeros(&field, 2, 3).unwrap();
    assert_eq!(
        MatGroup::new(&field, 2, vec![oblong]).unwrap_err().code(),
        "E-MATGRP-001"
    );
    let wrong_degree = GfMatrix::identity(&field, 3).unwrap();
    assert_eq!(
        MatGroup::new(&field, 2, vec![wrong_degree])
            .unwrap_err()
            .code(),
        "E-MATGRP-002"
    );
    let singular = GfMatrix::zeros(&field, 2, 2).unwrap();
    assert_eq!(
        MatGroup::new(&field, 2, vec![singular]).unwrap_err().code(),
        "E-MATGRP-004"
    );
    let elsewhere = GfMatrix::identity(&gf(5), 2).unwrap();
    assert_eq!(
        MatGroup::new(&field, 2, vec![elsewhere.clone()])
            .unwrap_err()
            .code(),
        "E-MATGRP-003"
    );

    // `from_generators` refuses an empty list rather than guessing a degree.
    assert_eq!(
        MatGroup::from_generators(&field, vec![])
            .unwrap_err()
            .code(),
        "E-MATGRP-002"
    );

    // Membership of a matrix from another field or shape is an error, not
    // `false`.
    let gl = MatGroup::general_linear(&field, 2).unwrap();
    assert_eq!(gl.contains(&elsewhere).unwrap_err().code(), "E-MATGRP-003");
    assert_eq!(
        gl.contains(&GfMatrix::identity(&field, 3).unwrap())
            .unwrap_err()
            .code(),
        "E-MATGRP-002"
    );
    // But a singular matrix is simply not a member.
    assert!(!gl
        .contains(&GfMatrix::zeros(&field, 2, 2).unwrap())
        .unwrap());

    // Element enumeration refuses above its cap; the order still works.
    assert_eq!(gl.order().unwrap(), 48);
    assert_eq!(gl.elements_with_cap(10).unwrap_err().code(), "E-MATGRP-008");

    // The Schreier-Sims budget is a refusal, not a partial chain.
    let starved = gl.with_budget(1);
    assert_eq!(starved.order().unwrap_err().code(), "E-MATGRP-007");
    // And the refusal is cached rather than recomputed into a different answer.
    assert_eq!(starved.order().unwrap_err().code(), "E-MATGRP-007");
    // Raising it again gets the right answer.
    assert_eq!(
        gl.with_budget(MAX_MATGROUP_SCHREIER_WORK).order().unwrap(),
        48
    );

    // A field too large to enumerate, for the operations that enumerate it.
    let big = FiniteField::prime(10_007).unwrap();
    assert_eq!(
        MatGroup::general_linear(&big, 2).unwrap_err().code(),
        "E-MATGRP-009"
    );
    // ... and a degree/field combination whose orbit is past the cap. Note the
    // *constructor* succeeds -- building generators costs nothing -- and it is
    // the chain that refuses, which is the split this module wants: a group can
    // exist without its order being computable here.
    let medium = FiniteField::prime(97).unwrap();
    let gl97 = MatGroup::general_linear(&medium, 3).unwrap();
    assert_eq!(gl97.order().unwrap_err().code(), "E-MATGRP-006");

    // Singer cycles over a proper extension field are refused, not guessed.
    assert_eq!(
        MatGroup::singer_cycle(&gfk(2, 2), 2).unwrap_err().code(),
        "E-MATGRP-010"
    );

    // A vector of the wrong shape.
    let row = GfMatrix::zeros(&field, 1, 3).unwrap();
    assert_eq!(gl.vector_orbit(&row).unwrap_err().code(), "E-MATGRP-002");
    let column = GfMatrix::zeros(&field, 2, 1).unwrap();
    assert_eq!(gl.vector_orbit(&column).unwrap_err().code(), "E-MATGRP-002");
}

#[test]
fn the_orbit_cap_is_a_refusal_not_a_truncation() {
    // GL(2, 97) has a 9408-point orbit on non-zero vectors, past the cap.
    let field = FiniteField::prime(97).unwrap();
    let one = field.one();
    let zero = field.zero();
    let a = super::element::primitive_element(&field).unwrap();
    let m = GfMatrix::from_elements(&field, 2, 2, &[a, zero.clone(), zero, one]).unwrap();
    // A single diagonal generator has a small orbit and is fine.
    let small = MatGroup::new(&field, 2, vec![m]).unwrap();
    assert_eq!(small.order().unwrap(), 96);
    // Adding a transvection makes the orbit the whole of GF(97)^2 minus zero.
    let t = GfMatrix::from_elements(
        &field,
        2,
        2,
        &[field.one(), field.one(), field.zero(), field.one()],
    )
    .unwrap();
    let big = MatGroup::new(&field, 2, vec![small.generators()[0].clone(), t]).unwrap();
    let err = big.order().unwrap_err();
    assert_eq!(err.code(), "E-MATGRP-006");
    assert!(err.remediation().is_some());
    // The orbit of a *given* vector refuses the same way rather than truncating.
    let v = GfMatrix::from_elements(&field, 1, 2, &[field.one(), field.zero()]).unwrap();
    assert_eq!(big.vector_orbit(&v).unwrap_err().code(), "E-MATGRP-006");
}

#[test]
fn display_and_accessors() {
    let gl = MatGroup::general_linear(&gfk(2, 2), 2).unwrap();
    let rendered = format!("{gl}");
    assert!(rendered.contains("degree 2"), "{rendered}");
    assert_eq!(gl.degree(), 2);
    assert!(gl.budget() > 0);
    assert!(!gl.generators().is_empty());
    assert!(gl.stabilizer_chain().unwrap().work() > 0);
    assert_eq!(gl.stabilizer_chain().unwrap().degree(), 2);
    assert_eq!(gl.stabilizer_chain().unwrap().field(), gl.field());
    assert_eq!(
        gl.stabilizer_chain().unwrap().base_vectors().unwrap().len(),
        2
    );
    let orbit = gl.stabilizer_chain().unwrap().levels()[0].orbit();
    assert!(!orbit.is_empty());
    assert!(orbit.contains(gl.field(), &orbit.points()[0]));
    assert!(orbit.schreier_vector()[0].is_none());
    assert!(!gl.stabilizer_chain().unwrap().levels()[0]
        .generators()
        .is_empty());
}

/// The centre has **two** routes and the cheap one hides the other.
///
/// Below [`DEFAULT_MATGROUP_ELEMENT_CAP`] the centre is found by listing `G`
/// and filtering; above it, by intersecting `G` with the algebra of matrices
/// commuting with the generators. Every other centre test in this file stays
/// under the cap, so this one exists to exercise the algebra route on a group
/// where listing the elements is not an option: `|GL(4, 3)| = 24 261 120`.
#[test]
fn the_centre_above_the_element_cap_comes_from_the_commutant_algebra() {
    let field = gf(3);
    let gl = MatGroup::general_linear(&field, 4).unwrap();
    let order = gl.order().unwrap();
    assert_eq!(order, gl_order(&q_of(&field), 4));
    assert!(
        order > DEFAULT_MATGROUP_ELEMENT_CAP,
        "this test is pointless if the element route can be taken"
    );
    assert_eq!(gl.elements().unwrap_err().code(), "E-MATGRP-008");

    // The natural GL-module is absolutely irreducible, so its endomorphism
    // algebra is the scalars: one basis element, and q - 1 invertible ones.
    assert_eq!(gl.commutant_basis().unwrap().len(), 1);
    let centre = gl.centre_elements().unwrap();
    assert_eq!(centre.len(), 2, "Z(GL(4,3)) is {{I, -I}}");
    for z in &centre {
        assert!(gl.contains(z).unwrap());
        for g in gl.generators() {
            assert!(super::element::commutes(z, g).unwrap());
        }
    }
    assert_eq!(gl.centre().unwrap().order().unwrap(), 2);

    // SL(4, 3) has the same centre, and its index in GL is q - 1.
    let sl = MatGroup::special_linear(&field, 4).unwrap();
    assert_eq!(sl.order().unwrap(), sl_order(&q_of(&field), 4));
    assert_eq!(sl.centre_elements().unwrap().len(), 2);
}
