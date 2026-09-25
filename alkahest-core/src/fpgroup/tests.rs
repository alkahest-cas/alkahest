//! Tests for finitely presented groups.
//!
//! The orders asserted here are the classical ones: the von Dyck groups
//! `⟨a, b | a², b³, (ab)ⁿ⟩` are `A₄`, `S₄` and `A₅` for `n = 3, 4, 5` and
//! infinite for `n = 7`; `⟨a, b | a², b², (ab)ⁿ⟩` is dihedral of order `2n`;
//! `⟨a | aⁿ⟩` is cyclic. Every finite order is **also** recomputed through
//! [`PermutationGroup::order`] — Schreier–Sims on the regular representation —
//! so a self-consistent but wrong coset table cannot pass.

use super::*;
use crate::errors::AlkahestError;
use rug::Integer;
use std::collections::HashSet;

fn group(gens: &[&str], rels: &[&str]) -> FpGroup {
    FpGroup::from_strings(gens, rels).expect("presentation should be well formed")
}

fn order_of(g: &FpGroup) -> Integer {
    g.order().expect("enumeration should complete")
}

// ---------------------------------------------------------------------------
// Words and free groups
// ---------------------------------------------------------------------------

#[test]
fn free_reduction_cancels_inverse_pairs() {
    let w = Word::from_letters(&[1, 2, -2, -1, 3]).unwrap();
    assert_eq!(w.letters(), &[3]);
    let w = Word::from_letters(&[1, -1]).unwrap();
    assert!(w.is_empty());
    assert_eq!(w, Word::identity());
}

#[test]
fn zero_is_not_a_letter() {
    let e = Word::from_letters(&[1, 0]).unwrap_err();
    assert_eq!(e.code(), "E-FPGRP-001");
}

#[test]
fn word_arithmetic_respects_the_free_group() {
    let a = Word::generator(0).unwrap();
    let b = Word::generator(1).unwrap();
    assert_eq!(a.times(&a.inverse()), Word::identity());
    assert_eq!(a.pow(3).letters(), &[1, 1, 1]);
    assert_eq!(a.pow(-2).letters(), &[-1, -1]);
    assert_eq!(a.pow(0), Word::identity());
    // (uv)^-1 = v^-1 u^-1
    let uv = a.times(&b);
    assert_eq!(uv.inverse(), b.inverse().times(&a.inverse()));
    assert_eq!(Word::commutator(&a, &b).letters(), &[1, 2, -1, -2]);
}

#[test]
fn exponent_sums_abelianise_the_word() {
    let free = FreeGroup::with_names(&["a", "b"]).unwrap();
    let w = free.parse("a^3*b^-2*a^-1").unwrap();
    assert_eq!(w.exponent_sums(2).unwrap(), vec![2, -2]);
    let c = free.parse("a*b*a^-1*b^-1").unwrap();
    assert_eq!(c.exponent_sums(2).unwrap(), vec![0, 0]);
}

#[test]
fn the_parser_handles_powers_products_and_parentheses() {
    let free = FreeGroup::with_names(&["a", "b"]).unwrap();
    assert_eq!(free.parse("a^2").unwrap().letters(), &[1, 1]);
    assert_eq!(free.parse("(a*b)^2").unwrap().letters(), &[1, 2, 1, 2]);
    assert_eq!(free.parse("ab").unwrap().letters(), &[1, 2]);
    assert_eq!(free.parse("a*b^-1").unwrap().letters(), &[1, -2]);
    assert_eq!(free.parse("(a*b)^-1").unwrap().letters(), &[-2, -1]);
    assert!(free.parse("1").unwrap().is_empty());
    assert!(free.parse("  a * b  ").unwrap().letters() == [1, 2]);
    // Cancellation happens during parsing too.
    assert!(free.parse("a*a^-1").unwrap().is_empty());
}

#[test]
fn the_parser_matches_the_longest_generator_name() {
    let free = FreeGroup::with_names(&["a", "ab"]).unwrap();
    assert_eq!(free.parse("ab").unwrap().letters(), &[2]);
    assert_eq!(free.parse("a*b").unwrap_err().code(), "E-FPGRP-003");
}

#[test]
fn parse_failures_are_typed_and_located() {
    let free = FreeGroup::with_names(&["a", "b"]).unwrap();
    for bad in ["c", "a^", "(a*b", "a**b", "a^999999999999"] {
        let e = free.parse(bad).unwrap_err();
        assert_eq!(e.code(), "E-FPGRP-003", "{bad} should be a syntax error");
    }
}

#[test]
fn malformed_alphabets_are_refused() {
    assert_eq!(
        FreeGroup::with_names(&["a", "a"]).unwrap_err().code(),
        "E-FPGRP-002"
    );
    assert_eq!(
        FreeGroup::with_names(&["a", ""]).unwrap_err().code(),
        "E-FPGRP-002"
    );
    assert_eq!(
        FreeGroup::with_names(&["a", "b^2"]).unwrap_err().code(),
        "E-FPGRP-002"
    );
    assert_eq!(
        FreeGroup::with_names(&["a", "2b"]).unwrap_err().code(),
        "E-FPGRP-002"
    );
}

#[test]
fn a_relator_outside_the_alphabet_is_refused() {
    let free = FreeGroup::with_names(&["a"]).unwrap();
    let w = Word::from_letters(&[2]).unwrap();
    assert_eq!(
        FpGroup::new(free, vec![w]).unwrap_err().code(),
        "E-FPGRP-001"
    );
}

// ---------------------------------------------------------------------------
// Todd-Coxeter: the classical orders
// ---------------------------------------------------------------------------

#[test]
fn cyclic_groups_have_the_order_of_their_relator() {
    for n in 1usize..=12 {
        let g = group(&["a"], &[&format!("a^{n}")]);
        assert_eq!(
            order_of(&g),
            Integer::from(n),
            "|<a | a^{n}>| should be {n}"
        );
    }
}

#[test]
fn a_cyclic_group_presented_twice_over_collapses_to_the_gcd() {
    // <a | a^m, a^n> is Z/gcd(m,n); getting this right needs coincidences.
    for (m, n, expected) in [(6, 9, 3), (4, 6, 2), (12, 18, 6), (5, 7, 1)] {
        let g = group(&["a"], &[&format!("a^{m}"), &format!("a^{n}")]);
        assert_eq!(order_of(&g), Integer::from(expected), "gcd({m},{n})");
    }
}

#[test]
fn dihedral_groups_have_order_twice_n() {
    for n in 1usize..=10 {
        let g = group(&["a", "b"], &["a^2", "b^2", &format!("(a*b)^{n}")]);
        assert_eq!(
            order_of(&g),
            Integer::from(2 * n),
            "dihedral of order {}",
            2 * n
        );
    }
}

#[test]
fn the_finite_von_dyck_groups_are_a4_s4_and_a5() {
    for (n, expected) in [(3usize, 12u32), (4, 24), (5, 60)] {
        let g = group(&["a", "b"], &["a^2", "b^3", &format!("(a*b)^{n}")]);
        assert_eq!(
            order_of(&g),
            Integer::from(expected),
            "<a,b | a^2, b^3, (ab)^{n}> should have order {expected}"
        );
    }
}

#[test]
fn quaternion_and_coxeter_presentations() {
    // Q8 = <a, b | a^4, b^2 a^-2, b a b^-1 a>
    let q8 = group(&["a", "b"], &["a^4", "b^2*a^-2", "b*a*b^-1*a"]);
    assert_eq!(order_of(&q8), Integer::from(8));
    // The Coxeter group H3, order 120 — three generators rather than two.
    let h3 = group(
        &["a", "b", "c"],
        &["a^2", "b^2", "c^2", "(a*b)^3", "(b*c)^5", "(a*c)^2"],
    );
    assert_eq!(order_of(&h3), Integer::from(120));
    // S4 as a Coxeter group, a second presentation of a group we also build as
    // a von Dyck group above.
    let s4 = group(
        &["a", "b", "c"],
        &["a^2", "b^2", "c^2", "(a*b)^3", "(b*c)^3", "(a*c)^2"],
    );
    assert_eq!(order_of(&s4), Integer::from(24));
}

#[test]
fn the_trivial_group_survives_being_presented_awkwardly() {
    assert_eq!(order_of(&group(&["a", "b"], &["a", "b"])), Integer::from(1));
    assert_eq!(
        order_of(&group(&["a", "b"], &["a", "b", "(a*b)^7"])),
        Integer::from(1)
    );
    // (ab)^2 and (ab)^3 force ab = 1, so b = a and a^2 = 1.
    assert_eq!(
        order_of(&group(&["a", "b"], &["a^2", "b^2", "(a*b)^2", "(a*b)^3"])),
        Integer::from(2)
    );
    assert_eq!(order_of(&FpGroup::free(0).unwrap()), Integer::from(1));
}

// ---------------------------------------------------------------------------
// Todd-Coxeter: coincidence handling
// ---------------------------------------------------------------------------

/// Enumerating the cosets of the *whole group* must give index 1. The subgroup
/// generators alone force it, so this is the shortest path through the table and
/// the answer is known without any computation.
#[test]
fn enumerating_the_whole_group_gives_one_coset() {
    let cases: Vec<(FpGroup, Vec<&str>)> = vec![
        (
            group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]),
            vec!["a", "b"],
        ),
        (
            group(&["a", "b"], &["a^2", "b^3", "(a*b)^4"]),
            vec!["a", "b"],
        ),
        (
            group(
                &["a", "b", "c"],
                &["a^2", "b^2", "c^2", "(a*b)^3", "(b*c)^5", "(a*c)^2"],
            ),
            vec!["a", "b", "c"],
        ),
        // A generating set that is not the generators: <a, ab> = <a, b>.
        (
            group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]),
            vec!["a", "a*b"],
        ),
    ];
    for (g, subgens) in cases {
        let words = g.parse_words(&subgens).unwrap();
        let table = g.coset_table(&words).unwrap();
        assert_eq!(table.index(), 1, "{g} on {subgens:?}");
        let perm = table.permutation_group().unwrap();
        assert_eq!(perm.degree(), 1);
    }
}

/// Coincidences, deliberately provoked, with the answer known in advance.
///
/// Every case here collapses a table that grew larger than its answer: the
/// counts are asserted so that a future change which stops exercising the
/// cascade — say by pre-reducing the relators — fails instead of quietly
/// testing nothing. `cosets_defined > index` is the cascade; the order is
/// whether it was handled right.
#[test]
fn coincidence_cascades_are_provoked_and_handled() {
    struct Case {
        gens: Vec<&'static str>,
        rels: Vec<&'static str>,
        order: u32,
        least_coincidences: usize,
    }
    let cases = vec![
        // <a | a^6, a^9> = Z/3: the second relator collapses six cosets to three.
        Case {
            gens: vec!["a"],
            rels: vec!["a^6", "a^9"],
            order: 3,
            least_coincidences: 3,
        },
        // (ab)^2 and (ab)^3 together force ab = 1.
        Case {
            gens: vec!["a", "b"],
            rels: vec!["a^2", "b^2", "(a*b)^2", "(a*b)^3"],
            order: 2,
            least_coincidences: 2,
        },
        Case {
            gens: vec!["a", "b"],
            rels: vec!["a^2", "b^3", "(a*b)^5"],
            order: 60,
            least_coincidences: 1,
        },
        // The Coxeter group H3: 200 cosets defined for an answer of 120.
        Case {
            gens: vec!["a", "b", "c"],
            rels: vec!["a^2", "b^2", "c^2", "(a*b)^3", "(b*c)^5", "(a*c)^2"],
            order: 120,
            least_coincidences: 10,
        },
        Case {
            gens: vec!["a", "b"],
            rels: vec!["a^4", "b^2*a^-2", "b*a*b^-1*a"],
            order: 8,
            least_coincidences: 1,
        },
    ];
    for case in cases {
        let g = group(&case.gens, &case.rels);
        let table = g.coset_table(&[]).unwrap();
        assert_eq!(table.index(), case.order as usize, "index of {g}");
        assert!(
            table.coincidences() >= case.least_coincidences,
            "{g} processed only {} coincidences; this case exists to exercise them",
            table.coincidences()
        );
        assert!(
            table.cosets_defined() > table.index(),
            "{g} defined no more cosets than it kept, so nothing collapsed"
        );
        // And the collapsed table is still a group of the right order, checked
        // through Schreier-Sims rather than through itself.
        assert_eq!(
            table.permutation_group().unwrap().order().unwrap(),
            Integer::from(case.order)
        );
    }
}

/// A coincidence cascade whose result is checked two independent ways: the
/// index, and the order of the subgroup's own Reidemeister–Schreier
/// presentation. `[G:H] · |H| = |G|` is Lagrange, and neither side is computed
/// from the other.
#[test]
fn lagrange_holds_for_subgroups_of_a5_and_s4() {
    let a5 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    let s4 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^4"]);
    let cases = [
        (&a5, "a", 30usize, 2u32, 60u32),
        (&a5, "b", 20, 3, 60),
        (&a5, "a*b", 12, 5, 60),
        (&s4, "a", 12, 2, 24),
        (&s4, "b", 8, 3, 24),
        (&s4, "a*b", 6, 4, 24),
    ];
    for (g, gen, index, sub_order, order) in cases {
        let h = vec![g.parse_word(gen).unwrap()];
        assert_eq!(g.index(&h).unwrap(), index, "[G:<{gen}>]");
        let sub = g.subgroup_presentation(&h).unwrap();
        assert_eq!(
            sub.presentation().order().unwrap(),
            Integer::from(sub_order),
            "|<{gen}>| from its own presentation"
        );
        assert_eq!(
            Integer::from(index) * Integer::from(sub_order),
            Integer::from(order),
            "Lagrange for <{gen}>"
        );
    }
}

/// Redundant relators and redundant subgroup generators change the amount of
/// collapsing without changing the answer.
#[test]
fn redundant_input_does_not_change_the_index() {
    let g = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    let plain = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    let padded = group(
        &["a", "b"],
        &[
            "a^2",
            "b^3",
            "(a*b)^5",
            "a^4",
            "b^6",
            "(a*b)^10",
            "a*a*a*a*a*a",
        ],
    );
    assert_eq!(order_of(&plain), order_of(&padded));
    let h1 = g.parse_words(&["a"]).unwrap();
    let h2 = g.parse_words(&["a", "a^-1", "a^3", "1"]).unwrap();
    assert_eq!(g.index(&h1).unwrap(), g.index(&h2).unwrap());
}

// ---------------------------------------------------------------------------
// Todd-Coxeter: the refusal
// ---------------------------------------------------------------------------

/// The `(2,3,7)` triangle group is infinite. The enumerator must refuse — not
/// hang, and above all not report a finite order.
#[test]
fn the_237_triangle_group_refuses_rather_than_answering() {
    let g = group(&["a", "b"], &["a^2", "b^3", "(a*b)^7"]);
    // Its abelianisation is trivial, so the infinite-ness is *not* detectable
    // the cheap way: this really does have to hit the cap.
    assert!(g.abelian_invariants().unwrap().is_trivial());
    for cap in [100usize, 1_000, 20_000] {
        let e = g.order_with_cap(cap).unwrap_err();
        assert_eq!(
            e.code(),
            "E-FPGRP-004",
            "cap {cap} should give the did-not-complete refusal"
        );
        let text = e.to_string();
        assert!(
            text.contains("NOT a claim"),
            "the refusal must not read as a claim of infiniteness: {text}"
        );
    }
    // And at the default cap, which is the path a caller takes by accident.
    assert_eq!(g.order().unwrap_err().code(), "E-FPGRP-004");
}

/// A subgroup of infinite index in a finite-index-free setting: `⟨a⟩` in the
/// free group of rank 2 has infinite index, so the enumeration cannot complete.
#[test]
fn infinite_index_subgroups_refuse_at_the_cap() {
    let f2 = FpGroup::free(2).unwrap();
    let h = vec![f2.word(&[1]).unwrap()];
    assert_eq!(
        f2.index_with_cap(&h, 5_000).unwrap_err().code(),
        "E-FPGRP-004"
    );
}

#[test]
fn a_provably_infinite_group_says_so_with_a_different_code() {
    // Z^2: the abelianisation is Z^2, so this never reaches the enumerator.
    let z2 = group(&["a", "b"], &["a*b*a^-1*b^-1"]);
    let e = z2.order().unwrap_err();
    assert_eq!(e.code(), "E-FPGRP-005");
    assert!(e.to_string().contains("infinite"));
    // The free group of rank 1 is Z.
    assert_eq!(
        FpGroup::free(1).unwrap().order().unwrap_err().code(),
        "E-FPGRP-005"
    );
    // The two refusals are different codes for different facts.
    let undecided = group(&["a", "b"], &["a^2", "b^3", "(a*b)^7"])
        .order_with_cap(500)
        .unwrap_err();
    assert_ne!(e.code(), undecided.code());
}

#[test]
fn an_absurd_coset_cap_is_refused_before_allocating() {
    let g = group(&["a", "b"], &["a^2", "b^3"]);
    assert_eq!(
        g.order_with_cap(usize::MAX / 8).unwrap_err().code(),
        "E-FPGRP-006"
    );
    assert_eq!(g.order_with_cap(0).unwrap_err().code(), "E-FPGRP-006");
}

// ---------------------------------------------------------------------------
// The coset table itself
// ---------------------------------------------------------------------------

#[test]
fn the_coset_table_is_complete_and_involutive() {
    let g = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    let table = g.coset_table(&[]).unwrap();
    assert_eq!(table.index(), 60);
    assert_eq!(table.rank(), 2);
    for c in 0..table.index() {
        let row = table.row(c).unwrap();
        assert_eq!(row.len(), 4);
        for (x, &d) in row.iter().enumerate() {
            assert!(d < table.index());
            assert_eq!(table.row(d).unwrap()[x ^ 1], c);
        }
    }
    // Every relator closes at every coset — the defining property.
    for r in g.relators() {
        for c in 0..table.index() {
            assert_eq!(table.trace(c, r).unwrap(), c);
        }
    }
    assert_eq!(table.image(0, 1).unwrap(), table.row(0).unwrap()[0]);
    assert_eq!(table.image(0, -1).unwrap(), table.row(0).unwrap()[1]);
}

#[test]
fn out_of_range_cosets_and_letters_are_refused() {
    let g = group(&["a", "b"], &["a^2", "b^3", "(a*b)^3"]);
    let table = g.coset_table(&[]).unwrap();
    assert_eq!(table.row(table.index()).unwrap_err().code(), "E-FPGRP-007");
    assert_eq!(table.image(0, 3).unwrap_err().code(), "E-FPGRP-001");
    assert_eq!(table.image(0, 0).unwrap_err().code(), "E-FPGRP-001");
}

#[test]
fn the_transversal_reaches_every_coset() {
    let g = group(&["a", "b"], &["a^2", "b^3", "(a*b)^4"]);
    let table = g.coset_table(&[]).unwrap();
    let transversal = table.transversal().unwrap();
    assert_eq!(transversal.len(), table.index());
    assert!(transversal[0].is_empty());
    for (c, u) in transversal.iter().enumerate() {
        assert_eq!(
            table.trace(0, u).unwrap(),
            c,
            "u_{c} should reach coset {c}"
        );
    }
}

#[test]
fn the_multiplication_table_is_a_group() {
    let g = group(&["a", "b"], &["a^2", "b^3", "(a*b)^3"]);
    let mult = g.multiplication_table().unwrap();
    let n = mult.len();
    assert_eq!(n, 12);
    // Identity.
    for (a, row) in mult.iter().enumerate() {
        assert_eq!(mult[0][a], a);
        assert_eq!(row[0], a);
    }
    // Latin square, by rows and then by columns.
    for row in &mult {
        assert_eq!(row.iter().copied().collect::<HashSet<_>>().len(), n);
    }
    for b in 0..n {
        let column: HashSet<usize> = mult.iter().map(|row| row[b]).collect();
        assert_eq!(column.len(), n);
    }
    // Associativity.
    for (a, row) in mult.iter().enumerate() {
        for (b, &ab) in row.iter().enumerate() {
            for (c, &bc) in mult[b].iter().enumerate() {
                assert_eq!(mult[ab][c], mult[a][bc]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The permutation representation — the independent cross-check
// ---------------------------------------------------------------------------

/// The regular representation's order is recomputed by Schreier–Sims in
/// [`crate::group`], a module this one did not write. If the coset table were
/// wrong in a way that preserved its own invariants, this is where it would
/// show.
#[test]
fn the_regular_representation_recomputes_the_order() {
    for (rels, expected) in [
        (vec!["a^2", "b^3", "(a*b)^3"], 12u32),
        (vec!["a^2", "b^3", "(a*b)^4"], 24),
        (vec!["a^2", "b^3", "(a*b)^5"], 60),
        (vec!["a^2", "b^2", "(a*b)^6"], 12),
        (vec!["a^4", "b^2*a^-2", "b*a*b^-1*a"], 8),
    ] {
        let g = group(&["a", "b"], &rels);
        let perm = g.regular_representation().unwrap();
        assert_eq!(perm.degree(), expected as usize);
        assert_eq!(
            perm.order().unwrap(),
            Integer::from(expected),
            "Schreier-Sims disagrees with the coset index for {g}"
        );
        assert!(perm.is_transitive());
    }
}

/// On the cosets of a subgroup with trivial core the action is still faithful,
/// so the permutation group has the full order at a smaller degree.
#[test]
fn the_action_on_cosets_of_a_subgroup_has_the_right_degree_and_order() {
    let a5 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    let h = vec![a5.parse_word("b").unwrap()];
    let perm = a5.permutation_group(&h).unwrap();
    assert_eq!(perm.degree(), 20);
    assert_eq!(perm.order().unwrap(), Integer::from(60));

    let s4 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^4"]);
    let h = vec![s4.parse_word("b").unwrap()];
    let perm = s4.permutation_group(&h).unwrap();
    assert_eq!(perm.degree(), 8);
    assert_eq!(perm.order().unwrap(), Integer::from(24));

    // A subgroup whose core is the whole group: the action is trivial.
    let perm = a5
        .permutation_group(&a5.parse_words(&["a", "b"]).unwrap())
        .unwrap();
    assert_eq!(perm.degree(), 1);
    assert_eq!(perm.order().unwrap(), Integer::from(1));
}

#[test]
fn the_generator_permutations_have_the_orders_the_relators_demand() {
    let a5 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    let table = a5.coset_table(&[]).unwrap();
    let perms = table.permutations().unwrap();
    assert_eq!(perms.len(), 2);
    assert_eq!(perms[0].order(), Integer::from(2));
    assert_eq!(perms[1].order(), Integer::from(3));
    let ab = perms[0].compose(&perms[1]).unwrap();
    assert_eq!(ab.order(), Integer::from(5));
    // All 60 elements are even permutations of the 60 cosets? No — but the
    // regular representation of A5 is a subgroup of S60 of order 60.
    assert_eq!(
        table.generator_permutation(2).unwrap_err().code(),
        "E-FPGRP-001"
    );
}

// ---------------------------------------------------------------------------
// Abelian invariants
// ---------------------------------------------------------------------------

#[test]
fn abelian_invariants_of_the_standard_examples() {
    struct Case {
        gens: Vec<&'static str>,
        rels: Vec<&'static str>,
        free_rank: usize,
        torsion: Vec<u32>,
    }
    fn case(
        gens: Vec<&'static str>,
        rels: Vec<&'static str>,
        free_rank: usize,
        torsion: Vec<u32>,
    ) -> Case {
        Case {
            gens,
            rels,
            free_rank,
            torsion,
        }
    }
    let cases = vec![
        // <a, b | [a,b]> = Z^2
        case(vec!["a", "b"], vec!["a*b*a^-1*b^-1"], 2, vec![]),
        // A5 is perfect
        case(vec!["a", "b"], vec!["a^2", "b^3", "(a*b)^5"], 0, vec![]),
        // <a | a^6> = Z/6
        case(vec!["a"], vec!["a^6"], 0, vec![6]),
        // S4^ab = Z/2
        case(vec!["a", "b"], vec!["a^2", "b^3", "(a*b)^4"], 0, vec![2]),
        // A4^ab = Z/3
        case(vec!["a", "b"], vec!["a^2", "b^3", "(a*b)^3"], 0, vec![3]),
        // dihedral of order 12: Z/2 + Z/2
        case(vec!["a", "b"], vec!["a^2", "b^2", "(a*b)^6"], 0, vec![2, 2]),
        // Z/2 x Z/2
        case(
            vec!["a", "b"],
            vec!["a^2", "b^2", "a*b*a^-1*b^-1"],
            0,
            vec![2, 2],
        ),
        // Z x Z/2
        case(vec!["a", "b"], vec!["b^2", "a*b*a^-1*b^-1"], 1, vec![2]),
        // Q8^ab = Z/2 + Z/2
        case(
            vec!["a", "b"],
            vec!["a^4", "b^2*a^-2", "b*a*b^-1*a"],
            0,
            vec![2, 2],
        ),
    ];
    for c in cases {
        let g = group(&c.gens, &c.rels);
        let inv = g.abelian_invariants().unwrap();
        assert_eq!(inv.free_rank(), c.free_rank, "free rank of {g}");
        let expected: Vec<Integer> = c.torsion.iter().map(|&d| Integer::from(d)).collect();
        assert_eq!(inv.torsion(), expected.as_slice(), "torsion of {g}");
    }
}

#[test]
fn a_free_group_abelianises_to_a_free_abelian_group() {
    for rank in 0usize..=4 {
        let inv = FpGroup::free(rank).unwrap().abelian_invariants().unwrap();
        assert_eq!(inv.free_rank(), rank);
        assert!(inv.torsion().is_empty());
    }
}

#[test]
fn abelian_invariants_agree_with_the_order_when_the_group_is_abelian() {
    // <a, b | a^4, b^6, [a,b]> is Z/4 x Z/6 = Z/2 + Z/12, of order 24.
    let g = group(&["a", "b"], &["a^4", "b^6", "a*b*a^-1*b^-1"]);
    let inv = g.abelian_invariants().unwrap();
    assert_eq!(inv.to_string(), "Z/2 + Z/12");
    assert_eq!(inv.order().unwrap(), Integer::from(24));
    assert_eq!(order_of(&g), Integer::from(24));
}

#[test]
fn abelian_invariants_render_readably() {
    assert_eq!(AbelianInvariants::trivial().to_string(), "0");
    assert_eq!(
        AbelianInvariants::new(2, vec![Integer::from(6)]).to_string(),
        "Z^2 + Z/6"
    );
    assert_eq!(AbelianInvariants::new(1, vec![]).to_string(), "Z");
    // Units are not torsion factors, and neither is zero.
    let inv = AbelianInvariants::new(0, vec![Integer::from(1), Integer::from(4)]);
    assert_eq!(inv.torsion(), [Integer::from(4)]);
    assert_eq!(inv.order().unwrap(), Integer::from(4));
    assert!(AbelianInvariants::new(1, vec![]).order().is_none());
}

// ---------------------------------------------------------------------------
// Reidemeister-Schreier
// ---------------------------------------------------------------------------

#[test]
fn schreier_generator_counts_match_nielsen_schreier() {
    // A subgroup of index m in a group with n generators gets m*n - m + 1
    // Schreier generators, and for a *free* group that is its rank.
    let f2 = FpGroup::free(2).unwrap();
    // The kernel of F2 -> Z/2, a -> 1, b -> 0 has index 2 and is free of rank 3.
    let h = f2.parse_words(&["x1^2", "x2", "x1*x2*x1^-1"]).unwrap();
    let sub = f2.subgroup_presentation(&h).unwrap();
    assert_eq!(sub.index(), 2);
    assert_eq!(sub.rank(), 3);
    assert!(sub.presentation().relators().is_empty());
    assert_eq!(
        sub.presentation().abelian_invariants().unwrap().free_rank(),
        3
    );
    for (k, w) in sub.generator_words().iter().enumerate() {
        assert!(
            !w.is_empty(),
            "Schreier generator {k} should be non-trivial"
        );
    }
}

#[test]
fn subgroup_presentations_have_the_order_lagrange_demands() {
    let s3 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^2"]);
    let h = s3.parse_words(&["b"]).unwrap();
    let sub = s3.subgroup_presentation(&h).unwrap();
    assert_eq!(sub.index(), 2);
    assert_eq!(sub.rank(), 2 * 2 - 1);
    assert_eq!(sub.presentation().order().unwrap(), Integer::from(3));

    let a4 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^3"]);
    let h = a4.parse_words(&["a"]).unwrap();
    let sub = a4.subgroup_presentation(&h).unwrap();
    assert_eq!(sub.index(), 6);
    assert_eq!(sub.rank(), 6 * 2 - 5);
    assert_eq!(sub.presentation().order().unwrap(), Integer::from(2));
}

#[test]
fn a_subgroup_of_z_squared_is_z_squared() {
    let z2 = group(&["a", "b"], &["a*b*a^-1*b^-1"]);
    let h = z2.parse_words(&["a", "b^2"]).unwrap();
    let sub = z2.subgroup_presentation(&h).unwrap();
    assert_eq!(sub.index(), 2);
    let inv = sub.presentation().abelian_invariants().unwrap();
    assert_eq!(inv.free_rank(), 2, "a finite-index subgroup of Z^2 is Z^2");
    assert!(inv.torsion().is_empty());
}

#[test]
fn schreier_generators_are_words_in_the_parent_and_lie_in_the_subgroup() {
    // Every Schreier generator must fix the coset of H, which is exactly the
    // statement that it lies in H.
    let a5 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    let h = a5.parse_words(&["a*b"]).unwrap();
    let table = a5.coset_table(&h).unwrap();
    let sub = a5.subgroup_presentation(&h).unwrap();
    for (k, w) in sub.generator_words().iter().enumerate() {
        assert_eq!(
            table.trace(0, w).unwrap(),
            0,
            "Schreier generator {k} ({w}) should lie in H"
        );
    }
}

// ---------------------------------------------------------------------------
// Cohomology
// ---------------------------------------------------------------------------

fn cyclic(n: usize) -> FpGroup {
    group(&["a"], &[&format!("a^{n}")])
}

fn klein() -> FpGroup {
    group(&["a", "b"], &["a^2", "b^2", "a*b*a^-1*b^-1"])
}

fn trivial_module(invariants: &[u32], group_rank: usize) -> GModule {
    GModule::trivial(
        invariants.iter().map(|&d| Integer::from(d)).collect(),
        group_rank,
    )
    .unwrap()
}

/// `H⁰(G, M) = M^G`, which for a trivial action is `M` itself. Verified by hand
/// from the definition.
#[test]
fn h0_of_a_trivial_action_is_the_whole_module() {
    let g = cyclic(6);
    let m = trivial_module(&[0], 1);
    let h0 = g.cohomology(0, &m).unwrap();
    assert_eq!(h0.free_rank(), 1);
    assert!(h0.torsion().is_empty());

    let m = trivial_module(&[6], 1);
    assert_eq!(g.cohomology(0, &m).unwrap().to_string(), "Z/6");

    let m = trivial_module(&[2, 0], 1);
    assert_eq!(g.cohomology(0, &m).unwrap().to_string(), "Z + Z/2");
}

/// `H¹(G, ℤ) = Hom(G, ℤ) = 0` for finite `G`: a homomorphism from a finite group
/// to a torsion-free group is trivial. Verified by hand.
#[test]
fn h1_with_trivial_integer_coefficients_vanishes_for_finite_groups() {
    for g in [cyclic(2), cyclic(5), klein()] {
        let m = trivial_module(&[0], g.rank());
        let h1 = g.cohomology(1, &m).unwrap();
        assert!(h1.is_trivial(), "H^1({g}, Z) should be 0, got {h1}");
    }
    let s3 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^2"]);
    let m = trivial_module(&[0], 2);
    assert!(s3.cohomology(1, &m).unwrap().is_trivial());
}

/// `H²(ℤ/n, ℤ) = ℤ/n` with the trivial action — the cyclic-group formula
/// `H²(C, M) = M^C / N·M` with `N = n`. Verified by hand.
#[test]
fn h2_of_a_cyclic_group_with_integer_coefficients_is_the_group() {
    for n in [2usize, 3, 4, 5, 6] {
        let g = cyclic(n);
        let m = trivial_module(&[0], 1);
        let h2 = g.cohomology(2, &m).unwrap();
        assert_eq!(
            h2.to_string(),
            format!("Z/{n}"),
            "H^2(Z/{n}, Z) should be Z/{n}"
        );
        assert_eq!(h2.order().unwrap(), Integer::from(n));
    }
}

/// `H¹(ℤ/n, ℤ/n) = ℤ/n` and `H²(ℤ/n, ℤ/n) = ℤ/n` for the trivial action — again
/// the cyclic formula, with `N = n = 0` on `ℤ/n`. Verified by hand.
#[test]
fn cohomology_of_a_cyclic_group_with_coefficients_in_itself() {
    for n in [2usize, 3, 4] {
        let g = cyclic(n);
        let m = trivial_module(&[n as u32], 1);
        assert_eq!(g.cohomology(1, &m).unwrap().to_string(), format!("Z/{n}"));
        assert_eq!(g.cohomology(2, &m).unwrap().to_string(), format!("Z/{n}"));
    }
}

/// `H¹(G, M) = Hom(G^ab, M)` for a trivial action. Checked against
/// [`FpGroup::abelian_invariants`], which is computed a completely different
/// way.
#[test]
fn h1_with_a_trivial_action_is_hom_from_the_abelianisation() {
    let cases: Vec<(FpGroup, u32, &str)> = vec![
        (klein(), 2, "Z/2 + Z/2"),
        (cyclic(4), 2, "Z/2"),
        (cyclic(4), 4, "Z/4"),
        (cyclic(3), 2, "0"),
        // A4^ab = Z/3, so Hom(A4, Z/2) = 0 and Hom(A4, Z/3) = Z/3.
        (group(&["a", "b"], &["a^2", "b^3", "(a*b)^3"]), 2, "0"),
        (group(&["a", "b"], &["a^2", "b^3", "(a*b)^3"]), 3, "Z/3"),
    ];
    for (g, m, expected) in cases {
        let module = trivial_module(&[m], g.rank());
        assert_eq!(
            g.cohomology(1, &module).unwrap().to_string(),
            expected,
            "H^1({g}, Z/{m})"
        );
    }
}

/// `H²(ℤ/2 × ℤ/2, ℤ/2)` has order 8 — the degree-2 part of `F₂[x, y]`, spanned
/// by `x²`, `xy`, `y²`. Taken from the literature rather than computed by hand;
/// the group structure `(ℤ/2)³` is what this implementation reports.
#[test]
fn h2_of_the_klein_four_group_with_f2_coefficients_has_order_eight() {
    let g = klein();
    let m = trivial_module(&[2], 2);
    let h2 = g.cohomology(2, &m).unwrap();
    assert_eq!(h2.order().unwrap(), Integer::from(8));
    assert_eq!(h2.to_string(), "Z/2 + Z/2 + Z/2");
    // And H^1 is Hom((Z/2)^2, Z/2), of order 4.
    assert_eq!(
        g.cohomology(1, &m).unwrap().order().unwrap(),
        Integer::from(4)
    );
}

/// `H²(G, ℤ) ≅ Hom(G, ℚ/ℤ) ≅ (G^ab)^∨` for finite `G`, from the long exact
/// sequence of `0 → ℤ → ℚ → ℚ/ℤ → 0` and `H^n(G, ℚ) = 0` for `n > 0`. That is a
/// derivation, not a hand computation, so it is labelled as such.
#[test]
fn h2_with_integer_coefficients_is_the_dual_of_the_abelianisation() {
    let cases: Vec<(FpGroup, &str)> = vec![
        (klein(), "Z/2 + Z/2"),
        (group(&["a", "b"], &["a^2", "b^3", "(a*b)^2"]), "Z/2"),
        (group(&["a", "b"], &["a^2", "b^3", "(a*b)^3"]), "Z/3"),
    ];
    for (g, expected) in cases {
        let m = trivial_module(&[0], g.rank());
        assert_eq!(
            g.cohomology(2, &m).unwrap().to_string(),
            expected,
            "H^2({g}, Z)"
        );
        // Same answer as the abelianisation, which is a different computation.
        assert_eq!(
            g.cohomology(2, &m).unwrap(),
            g.abelian_invariants().unwrap()
        );
    }
}

/// A **non-trivial** action: `ℤ/2` acting on `ℤ` by `−1`. The cyclic formulas
/// give `H⁰ = M^G = 0`, `H¹ = ker(1 + t)/im(t − 1) = ℤ/2ℤ` and
/// `H² = M^G/N·M = 0`. All three by hand.
#[test]
fn cohomology_with_a_non_trivial_action() {
    let g = cyclic(2);
    let m = GModule::new(vec![Integer::from(0)], vec![vec![vec![Integer::from(-1)]]]).unwrap();
    assert!(!m.is_trivial_action());
    assert_eq!(g.cohomology(0, &m).unwrap().to_string(), "0");
    assert_eq!(g.cohomology(1, &m).unwrap().to_string(), "Z/2");
    assert_eq!(g.cohomology(2, &m).unwrap().to_string(), "0");
}

/// `ℤ/2` acting on `ℤ²` by swapping the summands. `H⁰ = (ℤ²)^G = ℤ` (the
/// diagonal), and `H¹ = 0` because the module is induced (`ℤ² = ℤ[G]` is free
/// over the group ring, so its higher cohomology vanishes). Both by hand.
#[test]
fn a_permutation_module_is_cohomologically_trivial() {
    let g = cyclic(2);
    let swap = vec![
        vec![Integer::from(0), Integer::from(1)],
        vec![Integer::from(1), Integer::from(0)],
    ];
    let m = GModule::new(vec![Integer::from(0), Integer::from(0)], vec![swap]).unwrap();
    assert_eq!(g.cohomology(0, &m).unwrap().to_string(), "Z");
    assert_eq!(g.cohomology(1, &m).unwrap().to_string(), "0");
    assert_eq!(g.cohomology(2, &m).unwrap().to_string(), "0");
}

#[test]
fn cohomology_refuses_what_it_cannot_do() {
    let g = cyclic(4);
    let m = trivial_module(&[0], 1);
    // Degree.
    assert_eq!(g.cohomology(3, &m).unwrap_err().code(), "E-FPGRP-008");
    // Wrong number of action matrices.
    let wrong = trivial_module(&[0], 2);
    assert_eq!(g.cohomology(1, &wrong).unwrap_err().code(), "E-FPGRP-010");
    // A module with no generators.
    assert_eq!(
        GModule::trivial(vec![], 1).unwrap_err().code(),
        "E-FPGRP-010"
    );
    // A negative invariant.
    assert_eq!(
        GModule::trivial(vec![Integer::from(-2)], 1)
            .unwrap_err()
            .code(),
        "E-FPGRP-010"
    );
    // Rank above the ceiling.
    assert_eq!(
        GModule::trivial(vec![Integer::from(0); MAX_MODULE_RANK + 1], 1)
            .unwrap_err()
            .code(),
        "E-FPGRP-010"
    );
    // A group too big for H^2: |G| = 24 needs 13824 cochain coordinates.
    let s4 = group(&["a", "b"], &["a^2", "b^3", "(a*b)^4"]);
    let m2 = trivial_module(&[0], 2);
    assert_eq!(s4.cohomology(2, &m2).unwrap_err().code(), "E-FPGRP-009");
    // An infinite group.
    let z2 = group(&["a", "b"], &["a*b*a^-1*b^-1"]);
    assert_eq!(z2.cohomology(1, &m2).unwrap_err().code(), "E-FPGRP-005");
}

#[test]
fn an_action_that_is_not_one_is_refused_rather_than_answered() {
    // `a` has order 2 in the group but the matrix has infinite order, so no
    // action exists and there is no cohomology to report.
    let g = cyclic(2);
    let m = GModule::new(vec![Integer::from(0)], vec![vec![vec![Integer::from(2)]]]).unwrap();
    assert_eq!(g.cohomology(1, &m).unwrap_err().code(), "E-FPGRP-011");
    // A matrix that does not respect the module's own relations is rejected at
    // construction. On Z/4 + Z/2 the map (x, y) -> (x + y, y) is not defined:
    // y is only known modulo 2, and the first coordinate is read modulo 4.
    let bad = GModule::new(
        vec![Integer::from(4), Integer::from(2)],
        vec![vec![
            vec![Integer::from(1), Integer::from(1)],
            vec![Integer::from(0), Integer::from(1)],
        ]],
    );
    assert_eq!(bad.unwrap_err().code(), "E-FPGRP-011");
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

#[test]
fn presentations_render_as_they_were_written() {
    let g = group(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]);
    assert_eq!(g.to_string(), "⟨a, b | a^2, b^3, a*b*a*b*a*b*a*b*a*b⟩");
    assert_eq!(FpGroup::free(2).unwrap().to_string(), "⟨x1, x2 | ⟩");
    let table = g.coset_table(&[]).unwrap();
    assert_eq!(table.strategy(), "HLT with lookahead");
    assert_eq!(table.max_cosets(), DEFAULT_MAX_COSETS);
    assert!(table.subgroup_generators().is_empty());
    assert_eq!(table.rows().len(), 60);
}

/// `smith_invariants` is the FLINT-backed shortcut this module's quotients rest
/// on, and it must agree entry for entry with the transform-tracking
/// `smith_form` it replaces — including on the **rectangular** shapes the
/// cohomology produces, which the existing normal-form tests only cover square.
/// The test lives here because this module is what would be wrong if it drifted.
#[test]
fn smith_invariants_agrees_with_the_full_smith_form() {
    use crate::matrix::normal_form::{smith_form, smith_invariants, IntegerMatrix};
    let cases: Vec<Vec<Vec<i64>>> = vec![
        vec![vec![2, 0], vec![0, 3], vec![5, 5]],
        vec![vec![12, 6, 4], vec![3, 9, 6], vec![2, 16, 14]],
        vec![vec![6, 0, 0, 1], vec![0, 0, 4, -2]],
        vec![vec![0, 0], vec![0, 0]],
        vec![vec![-4, 8, 12]],
        vec![vec![1], vec![-1], vec![7]],
    ];
    for rows in cases {
        let m = IntegerMatrix::from_nested(rows.clone()).unwrap();
        let (s, _u, _v) = smith_form(&m).unwrap();
        let diag: Vec<Integer> = (0..m.rows.min(m.cols))
            .map(|i| s.get(i, i).clone())
            .collect();
        assert_eq!(smith_invariants(&m), diag, "disagreement on {rows:?}");
    }
}
