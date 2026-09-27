//! Unit tests for conjugacy classes and character tables.
//!
//! The anchors are the smallest groups on which a plausible-looking wrong
//! implementation is caught:
//!
//! * `A_4` and `A_5` for the irrationalities — a rationals-only implementation
//!   passes `S_3` and `S_4` and fails both of these, in two *different* fields
//!   (`ℚ(ζ_3)` and `ℚ(√5)`).
//! * `D_4` and `Q_8` for reading off degrees instead of computing values. Note
//!   what the right assertion is: these two are the textbook pair of
//!   non-isomorphic groups with **identical** character tables, so a test that
//!   demanded different tables would be demanding a wrong answer. What differs
//!   is the class data — the element orders attached to the columns — and that
//!   is what is asserted here.
//! * Brute-force conjugation by *every* element, against the generator-only
//!   orbit search the implementation uses.

use super::*;
use crate::errors::AlkahestError;
use crate::group::{alternating, cyclic, dihedral, symmetric, Permutation, PermutationGroup};
use crate::numfield::NumberFieldElement;
use rug::{Integer, Rational};
use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The quaternion group of order 8, as its right regular representation on the
/// eight elements `1, −1, i, −i, j, −j, k, −k` (points `0 … 7`).
fn quaternion_group() -> PermutationGroup {
    // Right multiplication by i, then by j.
    let by_i = Permutation::from_images(vec![2, 3, 1, 0, 7, 6, 4, 5]).unwrap();
    let by_j = Permutation::from_images(vec![4, 5, 6, 7, 1, 0, 3, 2]).unwrap();
    PermutationGroup::new(8, vec![by_i, by_j]).unwrap()
}

/// A character value that is an integer, as an `i64`; panics otherwise, which
/// is what a test wants.
fn integer_value(value: &NumberFieldElement) -> i64 {
    let coefficients = value.coefficients();
    assert!(
        coefficients.iter().skip(1).all(|c| *c == 0),
        "expected a rational value, got {value}"
    );
    let constant = coefficients.first().cloned().unwrap_or_else(Rational::new);
    assert_eq!(*constant.denom(), 1, "expected an integer, got {constant}");
    constant.numer().to_i64().expect("a small character value")
}

/// Assert a whole row of the table against integers.
fn assert_row(table: &CharacterTable, i: usize, expected: &[i64]) {
    let row = table.character(i).unwrap();
    let actual: Vec<i64> = row.iter().map(integer_value).collect();
    assert_eq!(actual, expected, "row {i} of\n{table}");
}

/// The class sizes, in table-column order.
fn class_sizes(table: &CharacterTable) -> Vec<u64> {
    table.classes().classes().iter().map(|c| c.size()).collect()
}

/// The element orders of the class representatives, in table-column order.
fn class_element_orders(table: &CharacterTable) -> Vec<u64> {
    table
        .classes()
        .classes()
        .iter()
        .map(|c| c.element_order())
        .collect()
}

/// Every character value, as the sorted list of its coordinate vectors — a
/// table fingerprint that does not depend on row or column order.
fn table_fingerprint(table: &CharacterTable) -> Vec<Vec<Vec<Rational>>> {
    let width = table.field().degree();
    let mut rows: Vec<Vec<Vec<Rational>>> = (0..table.len())
        .map(|i| {
            let row = table.character(i).unwrap();
            let mut cells: Vec<Vec<Rational>> = row
                .iter()
                .map(|v| {
                    let mut c = v.coefficients();
                    c.resize(width.max(c.len()), Rational::new());
                    c
                })
                .collect();
            cells.sort();
            cells
        })
        .collect();
    rows.sort();
    rows
}

/// Conjugacy classes the slow, obviously-correct way: conjugate every element
/// by **every** element.
fn brute_force_class_sizes(group: &PermutationGroup) -> BTreeMap<Vec<usize>, u64> {
    let elements = group.elements().unwrap();
    let mut class_of: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for x in &elements {
        // The lexicographically smallest conjugate labels the class.
        let mut smallest = x.images().to_vec();
        for g in &elements {
            let conjugate = g.inverse().compose(x).unwrap().compose(g).unwrap();
            if conjugate.images() < smallest.as_slice() {
                smallest = conjugate.images().to_vec();
            }
        }
        class_of.insert(x.images().to_vec(), smallest);
    }
    let mut sizes: BTreeMap<Vec<usize>, u64> = BTreeMap::new();
    for label in class_of.values() {
        *sizes.entry(label.clone()).or_insert(0) += 1;
    }
    sizes
}

// ---------------------------------------------------------------------------
// Conjugacy classes
// ---------------------------------------------------------------------------

#[test]
fn s3_classes_have_the_expected_shape() {
    let classes = ConjugacyClasses::of(&symmetric(3).unwrap()).unwrap();
    assert_eq!(classes.len(), 3);
    assert_eq!(*classes.group_order(), Integer::from(6));
    assert_eq!(classes.exponent(), 6);

    let sizes: Vec<u64> = classes.classes().iter().map(|c| c.size()).collect();
    assert_eq!(sizes, vec![1, 3, 2]);
    let orders: Vec<u64> = classes
        .classes()
        .iter()
        .map(|c| c.element_order())
        .collect();
    assert_eq!(orders, vec![1, 2, 3]);
    let centralisers: Vec<Integer> = classes
        .classes()
        .iter()
        .map(|c| c.centraliser_order().clone())
        .collect();
    assert_eq!(
        centralisers,
        vec![Integer::from(6), Integer::from(2), Integer::from(3)]
    );

    // Class 0 is always the identity class.
    assert!(classes.classes()[0].representative().is_identity());
}

#[test]
fn class_sizes_divide_and_sum_to_the_group_order() {
    for group in [
        symmetric(4).unwrap(),
        alternating(5).unwrap(),
        dihedral(6).unwrap(),
        quaternion_group(),
        cyclic(7).unwrap(),
    ] {
        let classes = ConjugacyClasses::of(&group).unwrap();
        let order = classes.group_order().clone();
        let mut total = Integer::new();
        for class in classes.classes() {
            assert!(
                order.is_divisible(&Integer::from(class.size())),
                "class size {} does not divide |G| = {order}",
                class.size()
            );
            total += Integer::from(class.size());
        }
        assert_eq!(total, order);
    }
}

#[test]
fn class_sizes_match_brute_force_conjugation() {
    for group in [
        symmetric(4).unwrap(),
        alternating(4).unwrap(),
        dihedral(5).unwrap(),
        quaternion_group(),
    ] {
        let expected: Vec<u64> = {
            let mut v: Vec<u64> = brute_force_class_sizes(&group).into_values().collect();
            v.sort_unstable();
            v
        };
        let classes = ConjugacyClasses::of(&group).unwrap();
        let mut actual: Vec<u64> = classes.classes().iter().map(|c| c.size()).collect();
        actual.sort_unstable();
        assert_eq!(actual, expected, "degree {}", group.degree());

        // And the representatives really are the lexicographically smallest
        // members, which is what makes the ordering reproducible.
        for (i, class) in classes.classes().iter().enumerate() {
            let members = classes.class_elements(i).unwrap();
            assert_eq!(members.len() as u64, class.size());
            assert_eq!(members.first().unwrap(), class.representative());
        }
    }
}

#[test]
fn class_of_locates_every_element() {
    let group = symmetric(4).unwrap();
    let classes = ConjugacyClasses::of(&group).unwrap();
    for element in group.elements().unwrap() {
        let index = classes.class_of(&element).unwrap();
        // The element is genuinely in the class it was assigned to.
        assert!(classes.class_elements(index).unwrap().contains(&element));
        // Conjugate elements share a class.
        for g in group.generators() {
            let conjugate = g.inverse().compose(&element).unwrap().compose(g).unwrap();
            assert_eq!(classes.class_of(&conjugate).unwrap(), index);
        }
    }
}

#[test]
fn inverse_class_is_an_involution() {
    for group in [
        symmetric(4).unwrap(),
        alternating(5).unwrap(),
        cyclic(7).unwrap(),
    ] {
        let classes = ConjugacyClasses::of(&group).unwrap();
        for i in 0..classes.len() {
            let j = classes.inverse_class(i).unwrap();
            assert_eq!(classes.inverse_class(j).unwrap(), i);
            assert_eq!(
                classes.class(i).unwrap().size(),
                classes.class(j).unwrap().size()
            );
            assert_eq!(
                classes.class(i).unwrap().element_order(),
                classes.class(j).unwrap().element_order()
            );
        }
        // Class 0 is its own inverse: 1⁻¹ = 1.
        assert_eq!(classes.inverse_class(0).unwrap(), 0);
    }
}

#[test]
fn exponent_divides_the_group_order() {
    for group in [
        symmetric(4).unwrap(),
        alternating(5).unwrap(),
        dihedral(6).unwrap(),
        quaternion_group(),
    ] {
        let classes = ConjugacyClasses::of(&group).unwrap();
        assert!(classes
            .group_order()
            .is_divisible(&Integer::from(classes.exponent())));
        for class in classes.classes() {
            assert_eq!(classes.exponent() % class.element_order(), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Class multiplication coefficients
// ---------------------------------------------------------------------------

#[test]
fn multiplication_matrix_agrees_with_the_coefficients() {
    for group in [symmetric(4).unwrap(), alternating(4).unwrap()] {
        let classes = ConjugacyClasses::of(&group).unwrap();
        let r = classes.len();
        for k in 0..r {
            let matrix = classes.multiplication_matrix(k).unwrap();
            for (i, row) in matrix.iter().enumerate() {
                for (j, &entry) in row.iter().enumerate() {
                    assert_eq!(
                        entry,
                        classes.multiplication_coefficient(k, i, j).unwrap(),
                        "M_{k}[{i}][{j}]"
                    );
                    // The class algebra is commutative, so the coefficients are
                    // symmetric in their first two indices — which is also why
                    // nothing this module returns depends on the left-to-right
                    // composition convention.
                    assert_eq!(
                        entry,
                        classes.multiplication_coefficient(i, k, j).unwrap(),
                        "a_{{{k},{i},{j}}} != a_{{{i},{k},{j}}}"
                    );
                }
            }
        }
    }
}

#[test]
fn class_algebra_products_have_the_right_total_mass() {
    // Counting the pairs (x, y) ∈ K_k × K_i two ways: |K_k|·|K_i| products in
    // all, and Σ_j a_{kij}·|K_j| once sorted by the class of the product.
    let classes = ConjugacyClasses::of(&symmetric(4).unwrap()).unwrap();
    let r = classes.len();
    for k in 0..r {
        let matrix = classes.multiplication_matrix(k).unwrap();
        for (i, row) in matrix.iter().enumerate() {
            let total: u64 = row
                .iter()
                .enumerate()
                .map(|(j, &a)| a * classes.class(j).unwrap().size())
                .sum();
            assert_eq!(
                total,
                classes.class(k).unwrap().size() * classes.class(i).unwrap().size(),
                "class sum K_{k} · K_{i}"
            );
        }
    }
}

#[test]
fn class_multiplication_matrices_commute() {
    // The class sums span the *centre* of the group algebra, so the M_k
    // commute. This is the hypothesis Dixon's simultaneous diagonalisation
    // rests on, and it is cheap to check directly.
    let classes = ConjugacyClasses::of(&alternating(4).unwrap()).unwrap();
    let r = classes.len();
    let matrices: Vec<Vec<Vec<u64>>> = (0..r)
        .map(|k| classes.multiplication_matrix(k).unwrap())
        .collect();
    let product = |a: &[Vec<u64>], b: &[Vec<u64>]| -> Vec<Vec<u64>> {
        a.iter()
            .map(|row| {
                (0..r)
                    .map(|j| row.iter().zip(b.iter()).map(|(x, br)| x * br[j]).sum())
                    .collect()
            })
            .collect()
    };
    for k in 0..r {
        for l in 0..r {
            assert_eq!(
                product(&matrices[k], &matrices[l]),
                product(&matrices[l], &matrices[k]),
                "M_{k} and M_{l}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Character tables: the rational anchors
// ---------------------------------------------------------------------------

#[test]
fn trivial_group_has_one_character() {
    let table = CharacterTable::of(&crate::group::trivial(3)).unwrap();
    assert_eq!(table.len(), 1);
    assert_eq!(table.classes().len(), 1);
    assert_eq!(table.degrees(), &[1]);
    assert_eq!(table.exponent(), 1);
    assert_eq!(table.field().degree(), 1);
    assert_row(&table, 0, &[1]);
}

#[test]
fn s3_character_table() {
    let table = CharacterTable::of(&symmetric(3).unwrap()).unwrap();
    assert_eq!(table.classes().len(), 3);
    assert_eq!(table.degrees(), &[1, 1, 2]);
    assert_eq!(class_sizes(&table), vec![1, 3, 2]);
    assert_row(&table, 0, &[1, 1, 1]);
    assert_row(&table, 1, &[1, -1, 1]);
    assert_row(&table, 2, &[2, 0, -1]);
}

#[test]
fn s4_character_table() {
    let table = CharacterTable::of(&symmetric(4).unwrap()).unwrap();
    assert_eq!(table.classes().len(), 5);
    assert_eq!(table.degrees(), &[1, 1, 2, 3, 3]);
    // Columns: 1, (ab)(cd) ×3, (ab) ×6, (abc) ×8, (abcd) ×6.
    assert_eq!(class_sizes(&table), vec![1, 3, 6, 8, 6]);
    assert_eq!(class_element_orders(&table), vec![1, 2, 2, 3, 4]);
    assert_row(&table, 0, &[1, 1, 1, 1, 1]);
    assert_row(&table, 1, &[1, 1, -1, 1, -1]);
    assert_row(&table, 2, &[2, 2, 0, -1, 0]);
    assert_row(&table, 3, &[3, -1, -1, 0, 1]);
    assert_row(&table, 4, &[3, -1, 1, 0, -1]);
}

#[test]
fn s5_character_table() {
    let table = CharacterTable::of(&symmetric(5).unwrap()).unwrap();
    assert_eq!(table.classes().len(), 7);
    assert_eq!(table.degrees(), &[1, 1, 4, 4, 5, 5, 6]);
    let squares: u64 = table.degrees().iter().map(|d| d * d).sum();
    assert_eq!(squares, 120);
    // Every value of S_n is a rational integer.
    for i in 0..table.len() {
        for c in 0..table.classes().len() {
            integer_value(table.value(i, c).unwrap());
        }
    }
}

// ---------------------------------------------------------------------------
// Character tables: the irrational anchors
// ---------------------------------------------------------------------------

#[test]
fn a4_has_two_genuinely_irrational_characters() {
    let table = CharacterTable::of(&alternating(4).unwrap()).unwrap();
    assert_eq!(table.classes().len(), 4);
    assert_eq!(table.degrees(), &[1, 1, 1, 3]);
    assert_eq!(class_sizes(&table), vec![1, 3, 4, 4]);
    assert_row(&table, 0, &[1, 1, 1, 1]);
    assert_row(&table, 3, &[3, -1, 0, 0]);

    // Values live in Q(zeta_6), which contains the cube roots of unity.
    assert_eq!(table.exponent(), 6);
    assert_eq!(table.field().degree(), 2);

    let one = table.field().one();
    for i in [1usize, 2] {
        // Linear characters: χ is a homomorphism into the cube roots of unity.
        assert_eq!(*table.value(i, 1).unwrap(), one, "row {i} on the 2-2 class");
        for c in [2usize, 3] {
            let value = table.value(i, c).unwrap();
            assert_ne!(*value, one, "row {i} column {c} should not be 1");
            assert_eq!(value.pow(3), one, "row {i} column {c} is not a cube root");
            // And it is genuinely not rational: the coefficient of zeta_6 is
            // non-zero. This is the assertion a rationals-only implementation
            // cannot pass.
            assert_ne!(
                value.coefficients()[1],
                Rational::new(),
                "row {i} column {c} came out rational"
            );
        }
    }

    // The two are complex conjugates of each other: χ_2(g) = χ_1(g⁻¹).
    for c in 0..table.classes().len() {
        let inverse = table.classes().inverse_class(c).unwrap();
        assert_eq!(table.value(2, c).unwrap(), table.value(1, inverse).unwrap());
    }
}

#[test]
fn a5_degree_three_characters_carry_the_golden_ratio() {
    let table = CharacterTable::of(&alternating(5).unwrap()).unwrap();
    assert_eq!(table.classes().len(), 5);
    assert_eq!(table.degrees(), &[1, 3, 3, 4, 5]);
    assert_eq!(class_sizes(&table), vec![1, 15, 20, 12, 12]);
    assert_eq!(class_element_orders(&table), vec![1, 2, 3, 5, 5]);
    assert_row(&table, 0, &[1, 1, 1, 1, 1]);
    assert_row(&table, 3, &[4, 0, 1, -1, -1]);
    assert_row(&table, 4, &[5, 1, -1, 0, 0]);

    // Q(zeta_30) has degree 8 and contains sqrt(5) — a different field from
    // A_4's, which is the point of testing both.
    assert_eq!(table.exponent(), 30);
    assert_eq!(table.field().degree(), 8);

    let field = table.field();
    let one = field.one();
    // The two values on each order-5 class are the roots of x² − x − 1: sum 1,
    // product −1. That pins them to (1 ± √5)/2 exactly.
    for c in [3usize, 4] {
        let u = table.value(1, c).unwrap();
        let v = table.value(2, c).unwrap();
        assert_eq!(u.add(v).unwrap(), one, "sum on class {c}");
        assert_eq!(u.mul(v).unwrap(), one.neg(), "product on class {c}");
        // Each individually satisfies x² = x + 1.
        assert_eq!(u.mul(u).unwrap(), u.add(&one).unwrap());
        assert_eq!(v.mul(v).unwrap(), v.add(&one).unwrap());
        // And is irrational.
        assert!(
            u.coefficients().iter().skip(1).any(|c| *c != 0),
            "the golden ratio came out rational"
        );
    }
    // Both are still rational on the classes of order 1, 2 and 3.
    assert_eq!(integer_value(table.value(1, 0).unwrap()), 3);
    assert_eq!(integer_value(table.value(1, 1).unwrap()), -1);
    assert_eq!(integer_value(table.value(1, 2).unwrap()), 0);
}

#[test]
fn d5_degree_two_characters_carry_a_different_quadratic_irrationality() {
    // χ(r) = 2cos(2πk/5) for the two 2-dimensional characters: the roots of
    // x² + x − 1, i.e. (−1 ± √5)/2. Same √5 as A_5 and a different field:
    // Q(zeta_10) has degree 4, not 8.
    let table = CharacterTable::of(&dihedral(5).unwrap()).unwrap();
    assert_eq!(table.classes().len(), 4);
    assert_eq!(table.degrees(), &[1, 1, 2, 2]);
    assert_eq!(table.exponent(), 10);
    assert_eq!(table.field().degree(), 4);

    let field = table.field();
    let one = field.one();
    let rotation_classes: Vec<usize> = (0..4)
        .filter(|&c| table.classes().class(c).unwrap().element_order() == 5)
        .collect();
    assert_eq!(rotation_classes.len(), 2);
    for i in [2usize, 3] {
        let c = rotation_classes[0];
        let value = table.value(i, c).unwrap();
        // x² + x − 1 = 0.
        assert_eq!(
            value.mul(value).unwrap().add(value).unwrap(),
            one,
            "row {i} on class {c} of\n{table}"
        );
        assert!(value.coefficients().iter().skip(1).any(|c| *c != 0));
    }
}

#[test]
fn cyclic_group_characters_are_the_roots_of_unity() {
    for n in [2u64, 5, 6, 8] {
        let table = CharacterTable::of(&cyclic(n as usize).unwrap()).unwrap();
        assert_eq!(table.classes().len(), n as usize);
        assert_eq!(table.degrees(), vec![1; n as usize].as_slice());
        assert_eq!(table.exponent(), n);
        // Every class is a singleton: an abelian group is its own centre.
        assert!(table.classes().classes().iter().all(|c| c.size() == 1));
        let one = table.field().one();
        for i in 0..table.len() {
            for c in 0..table.classes().len() {
                let value = table.value(i, c).unwrap();
                assert_eq!(
                    value.pow(n),
                    one,
                    "chi_{i} on class {c} is not an n-th root"
                );
            }
        }
        // Some character is faithful, so some value is a *primitive* n-th root.
        let primitive = (0..table.len()).any(|i| {
            (0..table.classes().len()).any(|c| {
                let value = table.value(i, c).unwrap();
                (1..n).all(|d| n % d != 0 || value.pow(d) != one)
            })
        });
        assert!(primitive, "no primitive {n}-th root of unity in the table");
    }
}

// ---------------------------------------------------------------------------
// D4 and Q8: the same table, different classes
// ---------------------------------------------------------------------------

#[test]
fn d4_and_q8_share_a_character_table_but_not_their_class_data() {
    let d4 = CharacterTable::of(&dihedral(4).unwrap()).unwrap();
    let q8_group = quaternion_group();
    assert_eq!(q8_group.order().unwrap(), Integer::from(8));
    let q8 = CharacterTable::of(&q8_group).unwrap();

    // Both: order 8, five classes, degrees 1,1,1,1,2.
    assert_eq!(d4.degrees(), &[1, 1, 1, 1, 2]);
    assert_eq!(q8.degrees(), &[1, 1, 1, 1, 2]);
    assert_eq!(class_sizes(&d4), vec![1, 1, 2, 2, 2]);
    assert_eq!(class_sizes(&q8), vec![1, 1, 2, 2, 2]);

    // The character *tables* agree. D_4 and Q_8 are the textbook pair of
    // non-isomorphic groups with the same table, so this is the correct
    // assertion — demanding different tables would be demanding a wrong
    // answer.
    assert_eq!(table_fingerprint(&d4), table_fingerprint(&q8));

    // What distinguishes them is the class data: Q_8 has six elements of
    // order 4 and D_4 has two. A table read off degrees alone cannot see this.
    assert_eq!(class_element_orders(&d4), vec![1, 2, 2, 2, 4]);
    assert_eq!(class_element_orders(&q8), vec![1, 2, 4, 4, 4]);
    let order_four = |table: &CharacterTable| -> u64 {
        table
            .classes()
            .classes()
            .iter()
            .filter(|c| c.element_order() == 4)
            .map(|c| c.size())
            .sum()
    };
    assert_eq!(order_four(&d4), 2);
    assert_eq!(order_four(&q8), 6);
}

// ---------------------------------------------------------------------------
// The orthogonality relations, as assertions in their own right
// ---------------------------------------------------------------------------

#[test]
fn orthogonality_holds_for_every_tested_group() {
    let groups = [
        crate::group::trivial(2),
        cyclic(6).unwrap(),
        symmetric(3).unwrap(),
        symmetric(4).unwrap(),
        alternating(4).unwrap(),
        alternating(5).unwrap(),
        dihedral(4).unwrap(),
        dihedral(5).unwrap(),
        dihedral(6).unwrap(),
        quaternion_group(),
    ];
    for group in groups {
        let table = CharacterTable::of(&group).unwrap();
        // `verify` is what the constructor already ran; running it again is how
        // a test tells "the checks pass" from "the checks were skipped".
        table.verify().unwrap();

        let r = table.classes().len();
        assert_eq!(table.len(), r, "one character per class");

        // Row orthogonality, through the public inner product.
        for i in 0..r {
            for j in 0..r {
                let expected = Rational::from(i32::from(i == j));
                assert_eq!(table.inner_product(i, j).unwrap(), expected, "<{i},{j}>");
            }
        }

        // Σ_i χ_i(1)² = |G|, and every degree divides |G|.
        let order = table.classes().group_order().clone();
        let mut squares = Integer::new();
        for &d in table.degrees() {
            assert!(
                order.is_divisible(&Integer::from(d)),
                "{d} does not divide {order}"
            );
            squares += Integer::from(d) * Integer::from(d);
        }
        assert_eq!(squares, order);

        // Column orthogonality, spelled out here rather than delegated.
        let field = table.field();
        for c in 0..r {
            for d in 0..r {
                let inverse = table.classes().inverse_class(d).unwrap();
                let mut total = field.zero();
                for i in 0..r {
                    total = total
                        .add(
                            &table
                                .value(i, c)
                                .unwrap()
                                .mul(table.value(i, inverse).unwrap())
                                .unwrap(),
                        )
                        .unwrap();
                }
                let expected = if c == d {
                    Rational::from(
                        table
                            .classes()
                            .class(c)
                            .unwrap()
                            .centraliser_order()
                            .clone(),
                    )
                } else {
                    Rational::new()
                };
                assert_eq!(total, field.rational(&expected), "column pair ({c}, {d})");
            }
        }
    }
}

#[test]
fn the_trivial_character_is_row_zero() {
    for group in [
        symmetric(4).unwrap(),
        alternating(5).unwrap(),
        quaternion_group(),
        cyclic(6).unwrap(),
    ] {
        let table = CharacterTable::of(&group).unwrap();
        let one = table.field().one();
        assert!(table.character(0).unwrap().iter().all(|v| *v == one));
        assert_eq!(table.degrees()[0], 1);
    }
}

#[test]
fn the_table_is_reproducible() {
    // Two independent runs agree entry for entry. The eigenvector search and
    // the choice of primitive root are deterministic, and the row order is
    // canonical, so this is not a tautology — it is what pins them down.
    let group = alternating(5).unwrap();
    let first = CharacterTable::of(&group).unwrap();
    let second = CharacterTable::of(&group).unwrap();
    assert_eq!(first.degrees(), second.degrees());
    assert_eq!(first.working_prime(), second.working_prime());
    for i in 0..first.len() {
        for c in 0..first.classes().len() {
            assert_eq!(first.value(i, c).unwrap(), second.value(i, c).unwrap());
        }
    }
    assert_eq!(first.render(), second.render());
}

#[test]
fn render_lays_out_one_line_per_character_plus_a_header() {
    let table = CharacterTable::of(&symmetric(3).unwrap()).unwrap();
    let text = table.render();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4);
    assert!(lines[0].starts_with("Q(zeta_6)"), "{text}");
    assert!(lines[1].starts_with("chi_0"), "{text}");
    assert_eq!(text, table.to_string());
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn a_group_past_the_cap_is_refused_with_its_order() {
    // |S_8| = 40320, past the default character-table cap of 5000.
    let err = CharacterTable::of(&symmetric(8).unwrap()).unwrap_err();
    assert_eq!(err.code(), "E-CHAR-001");
    match &err {
        CharacterError::OrderTooLarge { order, cap } => {
            assert_eq!(order, "40320");
            assert_eq!(*cap, DEFAULT_CHARACTER_TABLE_CAP);
        }
        other => panic!("expected OrderTooLarge, got {other:?}"),
    }
    // The order itself is still exact and still available.
    assert_eq!(
        symmetric(8).unwrap().order().unwrap(),
        Integer::from(40320u32)
    );
    // And raising the cap gets past *this* refusal (the class cap is separate).
    let err = ConjugacyClasses::of_with_cap(&symmetric(8).unwrap(), 10_000).unwrap_err();
    assert_eq!(err.code(), "E-CHAR-001");
    assert!(ConjugacyClasses::of_with_cap(&symmetric(8).unwrap(), 50_000).is_ok());
}

#[test]
fn a_cap_past_the_hard_ceiling_is_refused() {
    let err = ConjugacyClasses::of_with_cap(&symmetric(3).unwrap(), MAX_CLASS_ENUMERATION_CAP + 1)
        .unwrap_err();
    assert_eq!(err.code(), "E-CHAR-002");
    match err {
        CharacterError::CapTooLarge { cap, max } => {
            assert_eq!(cap, MAX_CLASS_ENUMERATION_CAP + 1);
            assert_eq!(max, MAX_CLASS_ENUMERATION_CAP);
        }
        other => panic!("expected CapTooLarge, got {other:?}"),
    }
}

#[test]
fn a_large_exponent_is_refused_by_field_degree_not_by_order() {
    // A cyclic group of order 9·11·13 = 1287, realised on 33 points as a
    // product of disjoint cycles of coprime length. |G| = 1287 is well inside
    // the order cap and the degree is well inside the Schreier–Sims one, but
    // phi(1287) = 720 is past the field-degree ceiling — which is the point:
    // this ceiling is on the exponent, not on the order.
    let generator = Permutation::from_cycles(
        33,
        &[
            (0..9).collect::<Vec<usize>>(),
            (9..20).collect::<Vec<usize>>(),
            (20..33).collect::<Vec<usize>>(),
        ],
    )
    .unwrap();
    let group = PermutationGroup::new(33, vec![generator]).unwrap();
    assert_eq!(group.order().unwrap(), Integer::from(1287));

    let err = CharacterTable::of(&group).unwrap_err();
    assert_eq!(err.code(), "E-CHAR-003");
    match err {
        CharacterError::ExponentFieldTooLarge {
            exponent,
            degree,
            max,
        } => {
            assert_eq!(exponent, 1287);
            assert_eq!(degree, 720);
            assert_eq!(max, MAX_EXPONENT_FIELD_DEGREE);
        }
        other => panic!("expected ExponentFieldTooLarge, got {other:?}"),
    }
    // The classes themselves are fine — it is the *table* that is refused.
    assert_eq!(ConjugacyClasses::of(&group).unwrap().len(), 1287);
}

#[test]
fn an_element_outside_the_group_is_refused() {
    let classes = ConjugacyClasses::of(&alternating(4).unwrap()).unwrap();
    let odd = Permutation::from_cycles(4, &[vec![0, 1]]).unwrap();
    let err = classes.class_of(&odd).unwrap_err();
    assert_eq!(err.code(), "E-CHAR-005");
    assert!(matches!(err, CharacterError::NotAMember { .. }));

    // A degree mismatch keeps the permutation-group layer's own code.
    let wrong_degree = Permutation::identity(5);
    let err = classes.class_of(&wrong_degree).unwrap_err();
    assert_eq!(err.code(), "E-GRP-002");
    assert!(matches!(err, CharacterError::Group(_)));
    assert!(err.remediation().unwrap().contains("extend_degree"));
}

#[test]
fn out_of_range_indices_are_refused() {
    let classes = ConjugacyClasses::of(&symmetric(3).unwrap()).unwrap();
    for err in [
        classes.class(3).unwrap_err(),
        classes.inverse_class(9).unwrap_err(),
        classes.class_elements(3).unwrap_err(),
        classes.multiplication_matrix(3).unwrap_err(),
        classes.multiplication_coefficient(0, 0, 7).unwrap_err(),
    ] {
        assert_eq!(err.code(), "E-CHAR-006");
    }
    let table = CharacterTable::of(&symmetric(3).unwrap()).unwrap();
    assert_eq!(table.character(3).unwrap_err().code(), "E-CHAR-006");
    assert_eq!(table.value(0, 3).unwrap_err().code(), "E-CHAR-006");
    assert_eq!(table.value(3, 0).unwrap_err().code(), "E-CHAR-006");
    assert_eq!(table.inner_product(0, 3).unwrap_err().code(), "E-CHAR-006");
}

#[test]
fn every_error_code_is_registered_and_has_remediation() {
    use crate::errors::codes::REGISTRY;
    let registered: Vec<&str> = REGISTRY
        .iter()
        .map(|spec| spec.code)
        .filter(|code| code.starts_with("E-CHAR-"))
        .collect();
    assert_eq!(
        registered,
        vec![
            "E-CHAR-001",
            "E-CHAR-002",
            "E-CHAR-003",
            "E-CHAR-004",
            "E-CHAR-005",
            "E-CHAR-006",
            "E-CHAR-007",
            "E-CHAR-008",
            "E-CHAR-009",
        ]
    );

    // Every variant's own code is in that list, and every one carries
    // remediation text.
    let samples: Vec<CharacterError> = vec![
        CharacterError::OrderTooLarge {
            order: "1".into(),
            cap: 1,
        },
        CharacterError::CapTooLarge { cap: 1, max: 1 },
        CharacterError::ExponentFieldTooLarge {
            exponent: 1,
            degree: 1,
            max: 1,
        },
        CharacterError::NoSuitablePrime {
            exponent: 1,
            order: "1".into(),
            searched_to: 1,
        },
        CharacterError::NotAMember {
            element: "()".into(),
        },
        CharacterError::ClassIndexOutOfRange {
            index: 1,
            classes: 1,
        },
        CharacterError::SelfCheckFailed {
            check: "row orthogonality",
            detail: "x".into(),
        },
        CharacterError::SplittingIncomplete {
            found: 1,
            classes: 2,
        },
        CharacterError::Internal { detail: "x".into() },
    ];
    for error in &samples {
        assert!(registered.contains(&error.code()), "{}", error.code());
        assert!(error.remediation().is_some(), "{}", error.code());
        assert!(!error.to_string().is_empty());
    }

    // The delegating variants forward rather than relabel.
    let delegated = CharacterError::Field(crate::numfield::NumberFieldError::NotInvertible);
    assert!(delegated.code().starts_with("E-NUMF-"));
}

// ---------------------------------------------------------------------------
// The modular scaffolding, tested on its own
// ---------------------------------------------------------------------------

#[test]
fn the_working_prime_satisfies_both_conditions_it_is_chosen_for() {
    use super::dixon::working_prime;
    for (exponent, order) in [
        (1u64, 1u64),
        (2, 2),
        (6, 6),
        (12, 24),
        (30, 60),
        (840, 40_320),
    ] {
        let p = working_prime(exponent, order).unwrap();
        // p ≡ 1 (mod exp G) puts a full set of exp G-th roots of unity in
        // GF(p); p > |G| makes every lift in `table` unambiguous and forces
        // p ∤ |G|, without which the class algebra is not semisimple mod p.
        assert_eq!((p - 1) % exponent, 0, "p = {p} for exponent {exponent}");
        assert!(p > order, "p = {p} is not above |G| = {order}");
        assert!(crate::modular::is_prime(p), "p = {p} is not prime");
        assert_ne!(order % p, 0, "p = {p} divides |G| = {order}");
    }
}

#[test]
fn the_root_of_unity_has_exactly_the_order_it_claims() {
    // This is the check the whole lift rests on. An element of order a proper
    // *divisor* of e would be a silent wrong answer — the inverse DFT would
    // reconstruct multiplicities against the wrong roots of unity — so the
    // order is computed here by brute-force repeated multiplication rather
    // than inferred from how the element was built.
    use super::dixon::{mul_mod, root_of_unity, working_prime};
    for exponent in [1u64, 2, 3, 4, 6, 8, 10, 12, 30, 60, 100] {
        let p = working_prime(exponent, exponent).unwrap();
        let w = root_of_unity(p, exponent).unwrap();
        let mut power = 1u64 % p;
        let mut order = 0u64;
        for k in 1..=exponent {
            power = mul_mod(power, w, p);
            if power == 1 {
                order = k;
                break;
            }
        }
        assert_eq!(
            order, exponent,
            "w = {w} in GF({p}) has multiplicative order {order}, not {exponent}"
        );
    }
}

#[test]
fn the_root_of_unity_is_the_same_one_every_time() {
    // The table's correctness needs *one* ring homomorphism Z[zeta_e] -> GF(p)
    // used for every class and every character. A different choice of w gives a
    // Galois-conjugate table, which is equally valid but must not be mixed
    // within one table, so the choice is deterministic.
    use super::dixon::root_of_unity;
    for (p, e) in [(61u64, 30u64), (13, 12), (101, 100)] {
        let first = root_of_unity(p, e).unwrap();
        for _ in 0..4 {
            assert_eq!(root_of_unity(p, e).unwrap(), first);
        }
    }
}
