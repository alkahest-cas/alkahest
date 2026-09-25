//! Property tests for the function-field layer.
//!
//! The invariants here are the cheap ones to state and the expensive ones to
//! get wrong: divisor degree is additive, the class map is a group
//! homomorphism, class reduction is idempotent, `div` is additive on products,
//! and `dim L(D)` depends only on the linear equivalence class.

use proptest::prelude::*;
use rug::{Integer, Rational};

use super::{Divisor, DivisorClass, FunctionField, FunctionFieldElement, Place};
use crate::funcfield::riemann_roch::riemann_roch;

/// Three curves with enough rational places to build interesting divisors.
fn curves() -> Vec<(FunctionField, Vec<Place>)> {
    let fin = |x: i64, y: i64| Place::finite(Rational::from(x), Rational::from(y));
    vec![
        (
            // y² = x³ − x, g = 1.  E(ℚ) = {∞, (0,0), (±1,0)}.
            FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap(),
            vec![fin(-1, 0), fin(0, 0), fin(1, 0), Place::Infinity],
        ),
        (
            // y² = x³ + 1, g = 1.  Torsion ℤ/6.
            FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 1]).unwrap(),
            vec![
                fin(-1, 0),
                fin(0, 1),
                fin(0, -1),
                fin(2, 3),
                fin(2, -3),
                Place::Infinity,
            ],
        ),
        (
            // y² = x⁵ + 1, g = 2.
            FunctionField::hyperelliptic_from_i64(&[1, 0, 0, 0, 0, 1]).unwrap(),
            vec![fin(-1, 0), fin(0, 1), fin(0, -1), Place::Infinity],
        ),
    ]
}

/// A divisor on curve `idx`, from multiplicities drawn for each rational place.
fn build(idx: usize, coeffs: &[i32]) -> Divisor {
    let all = curves();
    let (field, places) = &all[idx % all.len()];
    let terms: Vec<(Place, Integer)> = places
        .iter()
        .zip(coeffs.iter().cycle())
        .map(|(p, c)| (p.clone(), Integer::from(*c)))
        .collect();
    Divisor::from_terms(field.clone(), terms).expect("places are on their curve")
}

/// The same, forced to degree zero by adjusting the multiplicity at infinity.
fn build_deg0(idx: usize, coeffs: &[i32]) -> Divisor {
    let d = build(idx, coeffs);
    let deg = d.degree();
    let fix = Divisor::from_terms(d.field().clone(), [(Place::Infinity, -deg)])
        .expect("infinity is a place of every imaginary model");
    d.add(&fix).expect("same curve")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// `deg(D + E) = deg D + deg E`, and `deg(kD) = k·deg D`.
    #[test]
    fn divisor_degree_is_additive(
        idx in 0usize..3,
        a in prop::collection::vec(-6i32..7, 1..7),
        b in prop::collection::vec(-6i32..7, 1..7),
        k in -5i64..6,
    ) {
        let d = build(idx, &a);
        let e = build(idx, &b);
        let sum = d.add(&e).unwrap();
        prop_assert_eq!(sum.degree(), d.degree() + e.degree());
        prop_assert_eq!(d.sub(&e).unwrap().degree(), d.degree() - e.degree());
        prop_assert_eq!(d.neg().degree(), -d.degree());
        prop_assert_eq!(
            d.scale(&Integer::from(k)).degree(),
            d.degree() * Integer::from(k)
        );
    }

    /// The divisor group is abelian, and the canonical form has no zeroes.
    #[test]
    fn divisor_addition_is_an_abelian_group(
        idx in 0usize..3,
        a in prop::collection::vec(-6i32..7, 1..7),
        b in prop::collection::vec(-6i32..7, 1..7),
        c in prop::collection::vec(-6i32..7, 1..7),
    ) {
        let (d, e, f) = (build(idx, &a), build(idx, &b), build(idx, &c));
        prop_assert_eq!(d.add(&e).unwrap(), e.add(&d).unwrap());
        prop_assert_eq!(
            d.add(&e).unwrap().add(&f).unwrap(),
            d.add(&e.add(&f).unwrap()).unwrap()
        );
        prop_assert!(d.add(&d.neg()).unwrap().is_zero());
        // No place in the canonical form carries a zero multiplicity.
        for p in d.support() {
            prop_assert_ne!(d.coefficient(&p), Integer::from(0));
        }
    }

    /// `D ≤ E` iff `E − D` is effective, and the order is antisymmetric.
    #[test]
    fn the_divisor_order_matches_effectivity(
        idx in 0usize..3,
        a in prop::collection::vec(-4i32..5, 1..6),
        b in prop::collection::vec(-4i32..5, 1..6),
    ) {
        let (d, e) = (build(idx, &a), build(idx, &b));
        prop_assert_eq!(d.leq(&e).unwrap(), e.sub(&d).unwrap().is_effective());
        if d.leq(&e).unwrap() && e.leq(&d).unwrap() {
            prop_assert_eq!(d, e);
        }
    }

    /// `[·]` is a homomorphism from degree-zero divisors to `Pic⁰`.
    #[test]
    fn the_class_map_is_a_homomorphism(
        idx in 0usize..3,
        a in prop::collection::vec(-4i32..5, 1..6),
        b in prop::collection::vec(-4i32..5, 1..6),
    ) {
        let (d, e) = (build_deg0(idx, &a), build_deg0(idx, &b));
        let (cd, ce) = (DivisorClass::of(&d).unwrap(), DivisorClass::of(&e).unwrap());
        prop_assert_eq!(
            DivisorClass::of(&d.add(&e).unwrap()).unwrap(),
            cd.add(&ce).unwrap()
        );
        prop_assert_eq!(DivisorClass::of(&d.neg()).unwrap(), cd.neg());
        prop_assert!(cd.add(&cd.neg()).unwrap().is_identity());
    }

    /// Reduction is idempotent, and the reduced representative has the same
    /// class and weight ≤ g.
    #[test]
    fn class_reduction_is_idempotent(
        idx in 0usize..3,
        a in prop::collection::vec(-5i32..6, 1..6),
    ) {
        let d = build_deg0(idx, &a);
        let c = DivisorClass::of(&d).unwrap();
        prop_assert!(c.weight() <= d.field().genus());
        // `reduced_divisor` can refuse when u does not split over ℚ; when it
        // succeeds, re-reducing must be a no-op.
        if let Ok(rep) = c.reduced_divisor() {
            prop_assert_eq!(rep.degree(), Integer::from(0));
            let c2 = DivisorClass::of(&rep).unwrap();
            prop_assert_eq!(&c, &c2);
            prop_assert_eq!(c2.reduced_divisor().unwrap(), rep);
        }
    }

    /// `k·[D]` agrees with `[k·D]`, for positive and negative `k`.
    #[test]
    fn scalar_multiplication_commutes_with_the_class_map(
        idx in 0usize..3,
        a in prop::collection::vec(-4i32..5, 1..6),
        k in -8i64..9,
    ) {
        let d = build_deg0(idx, &a);
        let c = DivisorClass::of(&d).unwrap();
        prop_assert_eq!(
            c.scalar_mul(&Integer::from(k)).unwrap(),
            DivisorClass::of(&d.scale(&Integer::from(k))).unwrap()
        );
    }

    /// `N·δ` is principal exactly when the order of `[δ]` divides `N`.
    #[test]
    fn n_delta_principal_iff_the_order_divides_n(
        idx in 0usize..3,
        a in prop::collection::vec(-3i32..4, 1..5),
        n in 1u64..13,
    ) {
        let d = build_deg0(idx, &a);
        let field = d.field().clone();
        let c = DivisorClass::of(&d).unwrap();
        if let Ok(order) = c.order() {
            let scaled = d.scale(&Integer::from(n));
            prop_assert_eq!(
                field.is_principal(&scaled).unwrap(),
                n % order == 0,
                "order = {}, n = {}", order, n
            );
        }
    }

    /// `div` is a homomorphism: `div(u·v) = div(u) + div(v)`.
    #[test]
    fn div_is_additive_on_products(
        p1 in prop::collection::vec(-4i64..5, 1..4),
        q1 in prop::collection::vec(-4i64..5, 1..3),
        p2 in prop::collection::vec(-4i64..5, 1..4),
    ) {
        let f = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        let to_q = |v: &[i64]| -> Vec<Rational> { v.iter().map(|&c| Rational::from(c)).collect() };
        let u = FunctionFieldElement::polynomial(f.clone(), to_q(&p1), to_q(&q1));
        let v = FunctionFieldElement::polynomial(f, to_q(&p2), Vec::new());
        if u.is_zero() || v.is_zero() {
            return Ok(());
        }
        let uv = u.mul(&v).unwrap();
        if let (Ok(du), Ok(dv), Ok(duv)) = (u.divisor(), v.divisor(), uv.divisor()) {
            prop_assert_eq!(duv, du.add(&dv).unwrap());
        }
    }

    /// Every divisor of a function has degree zero.
    #[test]
    fn principal_divisors_have_degree_zero(
        p in prop::collection::vec(-5i64..6, 1..5),
        q in prop::collection::vec(-5i64..6, 1..4),
    ) {
        let f = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        let to_q = |v: &[i64]| -> Vec<Rational> { v.iter().map(|&c| Rational::from(c)).collect() };
        let u = FunctionFieldElement::polynomial(f.clone(), to_q(&p), to_q(&q));
        if u.is_zero() {
            return Ok(());
        }
        if let Ok(d) = u.divisor() {
            prop_assert_eq!(d.degree(), Integer::from(0));
            // A principal divisor is, by definition, in the identity class.
            prop_assert!(f.is_principal(&d).unwrap());
        }
    }

    /// Riemann–Roch above the canonical degree: `dim L(D) = deg D + 1 − g`.
    #[test]
    fn riemann_roch_equality_holds_above_the_canonical_degree(
        idx in 0usize..3,
        a in prop::collection::vec(-3i32..5, 1..6),
    ) {
        let d = build(idx, &a);
        let g = d.field().genus() as i64;
        let deg = d.degree().to_i64().unwrap();
        let space = riemann_roch(&d).unwrap();
        if deg > 2 * g - 2 {
            prop_assert_eq!(space.dimension() as i64, deg + 1 - g);
        } else {
            // Riemann's inequality always holds.
            prop_assert!(space.dimension() as i64 >= (deg + 1 - g).max(0));
        }
        if deg < 0 {
            prop_assert_eq!(space.dimension(), 0);
        }
    }

    /// Every basis element really lies in `L(D)`, checked by recomputing its
    /// divisor from scratch.
    #[test]
    fn riemann_roch_basis_elements_lie_in_the_space(
        idx in 0usize..3,
        a in prop::collection::vec(-2i32..4, 1..5),
    ) {
        let d = build(idx, &a);
        let space = riemann_roch(&d).unwrap();
        for u in space.basis() {
            if let Ok(du) = u.divisor() {
                prop_assert!(
                    du.add(&d).unwrap().is_effective(),
                    "div(u) + D = {} is not effective", du.add(&d).unwrap()
                );
            }
        }
    }

    /// `dim L(D)` is a linear-equivalence invariant: adding `div(x − α)`
    /// must not change it.
    #[test]
    fn riemann_roch_dimension_is_a_class_invariant(
        n in -3i64..8,
        alpha in prop::sample::select(vec![-1i64, 0, 1]),
    ) {
        // y² = x³ − x; x − α is principal for every branch point α.
        let f = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
        let d = Divisor::from_terms(f.clone(), [(Place::Infinity, Integer::from(n))]).unwrap();
        let to_q = |v: &[i64]| -> Vec<Rational> { v.iter().map(|&c| Rational::from(c)).collect() };
        let u = FunctionFieldElement::polynomial(f, to_q(&[-alpha, 1]), Vec::new());
        let du = u.divisor().unwrap();
        let shifted = d.add(&du).unwrap();
        prop_assert_eq!(shifted.degree(), d.degree());
        prop_assert_eq!(
            riemann_roch(&d).unwrap().dimension(),
            riemann_roch(&shifted).unwrap().dimension()
        );
    }
}
