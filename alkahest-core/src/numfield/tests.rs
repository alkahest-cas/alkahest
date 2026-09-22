//! Tests for [`crate::numfield`].
//!
//! Degrees 1, 2, 3 and 4 are each exercised deliberately, because
//! `nf_elem_t` is a union over the degree: Antic special-cases degree 1 and
//! degree 2, so a suite that only ever built cubic fields would leave two of
//! the three union arms untested and a wrong arm would corrupt memory
//! silently rather than fail to compile.

use super::*;
use crate::errors::AlkahestError;
use crate::flint::ffi;
use proptest::prelude::*;
use rug::{Integer, Rational};

fn q(n: i64) -> Rational {
    Rational::from(n)
}

fn field(coeffs: &[i64]) -> NumberField {
    let c: Vec<Rational> = coeffs.iter().copied().map(q).collect();
    NumberField::new(&c).expect("irreducible defining polynomial")
}

fn ints(coeffs: &[i64]) -> Vec<Integer> {
    coeffs.iter().map(|&c| Integer::from(c)).collect()
}

fn rats(coeffs: &[i64]) -> Vec<Rational> {
    coeffs.iter().copied().map(q).collect()
}

// ---------------------------------------------------------------------------
// FFI layout — the checks that catch a wrong struct before anything else does
// ---------------------------------------------------------------------------

#[test]
fn fmpq_layout_matches_flint() {
    // `Fmpq` is the one struct in the number-field FFI that is mirrored rather
    // than held opaquely, because `fmpq_numref`/`fmpq_denref` are C macros with
    // no exported symbol to route through. Cross-check the mirrored fields
    // against FLINT's own printer: B_12 = -691/2730.
    assert_eq!(std::mem::size_of::<ffi::Fmpq>(), 16);
    let b12 = crate::number_theory::bernoulli_number(12).unwrap();
    assert_eq!(*b12.numer(), Integer::from(-691));
    assert_eq!(*b12.denom(), Integer::from(2730));
}

#[test]
fn nf_buffers_have_generous_slack() {
    // Measured on FLINT 3.5.0: sizeof(nf_struct) = 112 of the 512 reserved,
    // and an nf_elem_struct is 16 bytes in degree 1 and 32 otherwise, of 128.
    // If a future FLINT grows either struct, this fails before the heap does.
    for coeffs in [
        vec![-3i64, 1],
        vec![-2, 0, 1],
        vec![-2, 0, 0, 1],
        vec![1, 1, 1, 1, 1],
    ] {
        let k = field(&coeffs);
        assert!(
            k.untouched_tail() >= 256,
            "nf_t buffer slack shrank to {} bytes for degree {}",
            k.untouched_tail(),
            k.degree()
        );
        let g = k.generator();
        assert!(
            g.untouched_tail() >= 64,
            "nf_elem_t buffer slack shrank to {} bytes for degree {}",
            g.untouched_tail(),
            k.degree()
        );
    }
}

// ---------------------------------------------------------------------------
// Degree 1 — the linear arm of the union
// ---------------------------------------------------------------------------

#[test]
fn degree_one_field_is_the_rationals() {
    let k = field(&[-3, 1]); // x - 3
    assert_eq!(k.degree(), 1);
    assert_eq!(k.defining_polynomial(), ints(&[-3, 1]).as_slice());

    let g = k.generator();
    assert_eq!(g.norm(), q(3));
    assert_eq!(g.trace(), q(3));
    assert_eq!(g.coefficients(), rats(&[3]));
    assert_eq!(g.to_string(), "3");

    let inv = g.inverse().unwrap();
    assert_eq!(inv.coefficients(), vec![Rational::from((1, 3))]);
    assert!(g.mul(&inv).unwrap().is_one());

    // The minimal polynomial of 3 over Q is x - 3.
    assert_eq!(g.minimal_polynomial(), rats(&[-3, 1]));
}

#[test]
fn degree_one_arithmetic_round_trips() {
    let k = field(&[-3, 1]);
    let a = k.rational(&Rational::from((5, 7)));
    let b = k.rational(&Rational::from((-2, 3)));
    assert_eq!(
        a.add(&b).unwrap().coefficients(),
        vec![Rational::from((1, 21))]
    );
    assert_eq!(
        a.mul(&b).unwrap().coefficients(),
        vec![Rational::from((-10, 21))]
    );
    assert_eq!(
        a.div(&b).unwrap().coefficients(),
        vec![Rational::from((-15, 14))]
    );
    assert_eq!(a.norm(), Rational::from((5, 7)));
    assert_eq!(a.trace(), Rational::from((5, 7)));
}

// ---------------------------------------------------------------------------
// Degree 2 — the quadratic arm of the union
// ---------------------------------------------------------------------------

#[test]
fn sqrt_two_norm_and_trace() {
    let k = field(&[-2, 0, 1]);
    assert_eq!(k.degree(), 2);
    let root = k.generator();
    assert_eq!(root.norm(), q(-2));
    assert_eq!(root.trace(), q(0));

    // N(a + b sqrt2) = a^2 - 2b^2 and Tr = 2a, over a spread of (a, b).
    for a in -4i64..=4 {
        for b in -4i64..=4 {
            let e = k.element(&rats(&[a, b])).unwrap();
            assert_eq!(e.norm(), q(a * a - 2 * b * b), "norm at ({a}, {b})");
            assert_eq!(e.trace(), q(2 * a), "trace at ({a}, {b})");
        }
    }

    // No coordinates at all is the zero element, not a malformed one.
    assert!(k.element(&[]).unwrap().is_zero());

    // The fundamental unit.
    let u = k.one().add(&root).unwrap();
    assert_eq!(u.norm(), q(-1));
    assert_eq!(u.to_string(), "a + 1");
}

#[test]
fn golden_ratio_field() {
    let k = field(&[-1, -1, 1]); // x^2 - x - 1
    let phi = k.generator();
    assert_eq!(phi.norm(), q(-1));
    assert_eq!(phi.trace(), q(1));
    assert_eq!(phi.minimal_polynomial(), rats(&[-1, -1, 1]));
    // phi^2 = phi + 1.
    assert_eq!(phi.pow(2), phi.add(&k.one()).unwrap());
    assert_eq!(k.polynomial_discriminant(), Integer::from(5));
}

// ---------------------------------------------------------------------------
// Degree 3 and 4 — the generic arm
// ---------------------------------------------------------------------------

#[test]
fn cube_root_of_two() {
    let k = field(&[-2, 0, 0, 1]);
    assert_eq!(k.degree(), 3);
    let c = k.generator();
    assert_eq!(c.norm(), q(2));
    assert_eq!(c.trace(), q(0));
    assert_eq!(c.minimal_polynomial(), rats(&[-2, 0, 0, 1]));
    assert_eq!(c.pow(3), k.rational(&q(2)));
    assert_eq!(k.polynomial_discriminant(), Integer::from(-108));
}

#[test]
fn zeta_five() {
    let k = NumberField::cyclotomic(5).unwrap();
    assert_eq!(k.degree(), 4);
    assert_eq!(k.cyclotomic_order(), Some(5));
    assert_eq!(k.defining_polynomial(), ints(&[1, 1, 1, 1, 1]).as_slice());

    let z = k.generator();
    assert_eq!(z.norm(), q(1));
    assert_eq!(z.trace(), q(-1));

    // 1 + z + z^2 + z^3 + z^4 = 0.
    let mut sum = k.zero();
    for e in 0..5u64 {
        sum = sum.add(&z.pow(e)).unwrap();
    }
    assert!(sum.is_zero(), "sum of the fifth roots of unity is {sum}");
    assert_eq!(z.pow(5), k.one());

    // z + z^4 = 2cos(72 degrees) generates the real quadratic subfield.
    let s = z.add(&z.pow(4)).unwrap();
    assert_eq!(s.minimal_polynomial(), rats(&[-1, 1, 1]));
    assert_eq!(k.polynomial_discriminant(), Integer::from(125));
}

#[test]
fn zeta_twelve_is_degree_four() {
    let k = NumberField::cyclotomic(12).unwrap();
    assert_eq!(k.degree(), 4);
    assert_eq!(k.defining_polynomial(), ints(&[1, 0, -1, 0, 1]).as_slice());
    let z = k.generator();
    assert_eq!(z.norm(), q(1));
    assert_eq!(z.trace(), q(0));
    assert_eq!(z.pow(12), k.one());
    assert_ne!(z.pow(6), k.one());
}

// ---------------------------------------------------------------------------
// Cyclotomic polynomials
// ---------------------------------------------------------------------------

#[test]
fn cyclotomic_polynomial_anchors() {
    assert_eq!(cyclotomic_polynomial(1), ints(&[-1, 1]));
    assert_eq!(cyclotomic_polynomial(2), ints(&[1, 1]));
    assert_eq!(cyclotomic_polynomial(6), ints(&[1, -1, 1]));
    assert_eq!(cyclotomic_polynomial(12), ints(&[1, 0, -1, 0, 1]));
    // Phi_105 is the smallest with a coefficient outside {-1, 0, 1}.
    assert!(cyclotomic_polynomial(105).iter().any(|c| *c == -2));
}

#[test]
fn cyclotomic_degree_is_totient() {
    for n in 1u64..=60 {
        let deg = cyclotomic_polynomial(n).len() - 1;
        let phi: u64 = crate::number_theory::totient(&n.to_string())
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(deg as u64, phi, "deg Phi_{n}");
        assert_eq!(NumberField::cyclotomic(n).unwrap().degree() as u64, phi);
    }
}

#[test]
fn cyclotomic_one_and_two_are_the_rationals() {
    for n in [1u64, 2] {
        let k = NumberField::cyclotomic(n).unwrap();
        assert_eq!(k.degree(), 1);
        let z = k.generator();
        assert_eq!(z.norm(), if n == 1 { q(1) } else { q(-1) });
    }
}

#[test]
fn large_cyclotomic_field_is_constructible() {
    // The ring-LWE shape: Q(zeta_256) = Q[x]/(x^128 + 1). Building it must not
    // run an irreducibility check — that is what makes it fast enough to be a
    // test rather than a benchmark.
    let k = NumberField::cyclotomic(256).unwrap();
    assert_eq!(k.degree(), 128);
    let z = k.generator();
    assert_eq!(z.pow(256), k.one());
    assert_eq!(z.pow(128), k.one().neg());
    assert_eq!(z.norm(), q(1));
}

#[test]
fn cyclotomic_order_out_of_range_refuses() {
    let e = NumberField::cyclotomic(0).unwrap_err();
    assert_eq!(e.code(), "E-NUMF-008");
    // phi(3^9) = 2 * 3^8 = 13122, past MAX_FIELD_DEGREE.
    let e = NumberField::cyclotomic(19683).unwrap_err();
    assert_eq!(e.code(), "E-NUMF-008");
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn reducible_defining_polynomial_refuses() {
    // x^2 - 1 = (x-1)(x+1): Q[x]/(x^2-1) has zero divisors.
    let e = NumberField::new(&rats(&[-1, 0, 1])).unwrap_err();
    assert_eq!(e.code(), "E-NUMF-003");
    assert!(matches!(e, NumberFieldError::Reducible { .. }), "{e:?}");
    // A repeated factor is reducible too.
    let e = NumberField::new(&rats(&[1, 2, 1])).unwrap_err();
    assert_eq!(e.code(), "E-NUMF-003");
}

#[test]
fn degenerate_defining_polynomials_refuse() {
    assert_eq!(
        NumberField::new(&[]).unwrap_err().code(),
        "E-NUMF-001",
        "the empty polynomial"
    );
    assert_eq!(
        NumberField::new(&rats(&[0, 0, 0])).unwrap_err().code(),
        "E-NUMF-001",
        "the zero polynomial"
    );
    assert_eq!(
        NumberField::new(&rats(&[7])).unwrap_err().code(),
        "E-NUMF-002",
        "a non-zero constant"
    );
}

#[test]
fn malformed_coefficient_refuses() {
    let e = NumberField::from_strings(&["1".into(), "not a number".into()]).unwrap_err();
    assert_eq!(e.code(), "E-NUMF-004");
}

#[test]
fn zero_has_no_inverse() {
    let k = field(&[-2, 0, 1]);
    assert_eq!(k.zero().inverse().unwrap_err().code(), "E-NUMF-005");
    assert_eq!(k.one().div(&k.zero()).unwrap_err().code(), "E-NUMF-005");
}

#[test]
fn elements_of_different_fields_do_not_mix() {
    let a = field(&[-2, 0, 1]).generator();
    let b = field(&[-3, 0, 1]).generator();
    let e = a.add(&b).unwrap_err();
    assert_eq!(e.code(), "E-NUMF-006");
    assert!(a.mul(&b).is_err());
    assert!(a != b);
}

#[test]
fn too_many_coordinates_refuses() {
    let k = field(&[-2, 0, 0, 1]);
    let e = k.element(&rats(&[1, 0, 0, 1])).unwrap_err();
    assert_eq!(e.code(), "E-NUMF-007");
    assert!(matches!(
        e,
        NumberFieldError::CoefficientCountMismatch { got: 4, degree: 3 }
    ));
}

#[test]
fn every_error_code_is_registered() {
    for e in [
        NumberFieldError::EmptyDefiningPolynomial,
        NumberFieldError::DegreeOutOfRange { degree: 0, max: 1 },
        NumberFieldError::Reducible {
            poly: String::new(),
            factor: String::new(),
        },
        NumberFieldError::MalformedCoefficient {
            text: String::new(),
        },
        NumberFieldError::NotInvertible,
        NumberFieldError::FieldMismatch {
            lhs: String::new(),
            rhs: String::new(),
        },
        NumberFieldError::CoefficientCountMismatch { got: 0, degree: 0 },
        NumberFieldError::CyclotomicOrderOutOfRange { n: 0, max: 0 },
    ] {
        assert!(
            crate::errors::codes::REGISTRY
                .iter()
                .any(|s| s.code == e.code()),
            "{} is not in REGISTRY",
            e.code()
        );
        assert!(e.remediation().is_some(), "{} has no remediation", e.code());
        assert!(!e.to_string().is_empty());
    }
}

// ---------------------------------------------------------------------------
// Normalisation, discriminant, rendering
// ---------------------------------------------------------------------------

#[test]
fn defining_polynomial_is_canonicalised() {
    // Scaling by a rational, and flipping the sign, name the same field.
    let a = field(&[-2, 0, 1]);
    let b = field(&[-4, 0, 2]);
    let c = NumberField::new(&[q(2), q(0), q(-1)]).unwrap();
    let d = NumberField::new(&[Rational::from((-2, 3)), q(0), Rational::from((1, 3))]).unwrap();
    for other in [&b, &c, &d] {
        assert_eq!(&a, other);
        assert_eq!(other.defining_polynomial(), ints(&[-2, 0, 1]).as_slice());
    }
    // Elements built in the two handles interoperate.
    assert_eq!(a.generator().add(&b.one()).unwrap().norm(), q(-1));
}

#[test]
fn polynomial_discriminant_is_not_the_field_discriminant() {
    // Q(sqrt 5): disc(x^2 - 5) = 20, while d_K = 5. The accessor is named for
    // what it computes and the module documents the difference; this test pins
    // the value so nobody "fixes" it into the field discriminant by accident.
    let k = field(&[-5, 0, 1]);
    assert_eq!(k.polynomial_discriminant(), Integer::from(20));
    assert_eq!(
        field(&[-2, 0, 1]).polynomial_discriminant(),
        Integer::from(8)
    );
    // x^3 + x^2 - 1: the cubic field of discriminant -23.
    assert_eq!(
        field(&[-1, 0, 1, 1]).polynomial_discriminant(),
        Integer::from(-23)
    );
}

#[test]
fn display_is_readable() {
    let k = field(&[-2, 0, 1]);
    assert_eq!(k.to_string(), "Q[x]/(x^2 - 2)");
    assert_eq!(k.defining_polynomial_string("t"), "t^2 - 2");
    assert_eq!(k.zero().to_string(), "0");
    assert_eq!(k.one().to_string(), "1");
    assert_eq!(k.generator().to_string(), "a");
    assert_eq!(k.generator().neg().to_string(), "-a");
    assert_eq!(
        k.generator().inverse().unwrap().to_string(),
        "1/2*a",
        "a rational coefficient"
    );
    let named = k.with_variable("s");
    assert_eq!(named.generator().to_string(), "s");
    assert_eq!(named, k);

    let z5 = NumberField::cyclotomic(5).unwrap();
    assert_eq!(
        z5.generator().inverse().unwrap().to_string(),
        "-a^3 - a^2 - a - 1"
    );
}

#[test]
fn flint_rendering_agrees_on_the_terms() {
    // Cross-check this module's renderer against FLINT's own. They differ only
    // in that FLINT writes a redundant `1*` on unit coefficients; if the terms
    // themselves ever disagree, one of the two is reading the wrong union arm.
    let z5 = NumberField::cyclotomic(5).unwrap();
    let inv = z5.generator().inverse().unwrap();
    assert_eq!(inv.flint_string(), "-a^3 - 1*a^2 - 1*a - 1");
    assert_eq!(inv.to_string(), "-a^3 - a^2 - a - 1");
    assert_eq!(field(&[-2, 0, 1]).generator().flint_string(), "a");
}

// ---------------------------------------------------------------------------
// Minimal polynomials
// ---------------------------------------------------------------------------

#[test]
fn minimal_polynomial_of_rationals_and_generators() {
    let k = field(&[-2, 0, 0, 1]);
    assert_eq!(
        k.rational(&Rational::from((3, 4))).minimal_polynomial(),
        vec![Rational::from((-3, 4)), q(1)],
        "x - 3/4"
    );
    assert_eq!(k.zero().minimal_polynomial(), rats(&[0, 1]));
    assert_eq!(k.one().minimal_polynomial(), rats(&[-1, 1]));
    // 1 + cbrt2 has minimal polynomial (x-1)^3 - 2 = x^3 - 3x^2 + 3x - 3.
    let e = k.one().add(&k.generator()).unwrap();
    assert_eq!(e.minimal_polynomial(), rats(&[-3, 3, -3, 1]));
}

#[test]
fn minimal_polynomial_degree_divides_the_field_degree() {
    let k = NumberField::cyclotomic(12).unwrap();
    for e in [
        k.generator(),
        k.generator().pow(2),
        k.generator().pow(3),
        k.generator().add(&k.one()).unwrap(),
        k.rational(&q(7)),
    ] {
        let m = e.minimal_polynomial();
        let deg = m.len() - 1;
        assert!(deg >= 1 && k.degree() % deg == 0, "degree {deg}");
        assert_eq!(m[deg], q(1), "minimal polynomials are monic");
        // Evaluating the minimal polynomial at the element gives zero.
        let mut acc = k.zero();
        for (i, c) in m.iter().enumerate() {
            acc = acc
                .add(&k.rational(c).mul(&e.pow(i as u64)).unwrap())
                .unwrap();
        }
        assert!(acc.is_zero(), "m(e) = {acc} for e = {e}");
    }
}

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

fn small_rational(s: (i64, i64)) -> Rational {
    Rational::from((s.0, if s.1 == 0 { 1 } else { s.1 }))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The norm is multiplicative, in every degree the union distinguishes.
    #[test]
    fn norm_is_multiplicative(
        deg_pick in 0usize..4,
        a in prop::collection::vec((-9i64..9, -5i64..5), 4),
        b in prop::collection::vec((-9i64..9, -5i64..5), 4),
    ) {
        let defs: [&[i64]; 4] = [&[-3, 1], &[-2, 0, 1], &[-2, 0, 0, 1], &[1, 1, 1, 1, 1]];
        let k = field(defs[deg_pick]);
        let d = k.degree();
        let x = k.element(&a[..d].iter().copied().map(small_rational).collect::<Vec<_>>()).unwrap();
        let y = k.element(&b[..d].iter().copied().map(small_rational).collect::<Vec<_>>()).unwrap();
        prop_assert_eq!(x.mul(&y).unwrap().norm(), x.norm() * y.norm());
    }

    /// The trace is additive.
    #[test]
    fn trace_is_additive(
        deg_pick in 0usize..4,
        a in prop::collection::vec((-9i64..9, -5i64..5), 4),
        b in prop::collection::vec((-9i64..9, -5i64..5), 4),
    ) {
        let defs: [&[i64]; 4] = [&[-3, 1], &[-2, 0, 1], &[-2, 0, 0, 1], &[1, 1, 1, 1, 1]];
        let k = field(defs[deg_pick]);
        let d = k.degree();
        let x = k.element(&a[..d].iter().copied().map(small_rational).collect::<Vec<_>>()).unwrap();
        let y = k.element(&b[..d].iter().copied().map(small_rational).collect::<Vec<_>>()).unwrap();
        prop_assert_eq!(x.add(&y).unwrap().trace(), x.trace() + y.trace());
    }

    /// `a * a^-1 == 1`, and `norm(a^-1) == norm(a)^-1`.
    #[test]
    fn inverse_round_trips(
        deg_pick in 0usize..4,
        a in prop::collection::vec((-9i64..9, -5i64..5), 4),
    ) {
        let defs: [&[i64]; 4] = [&[-3, 1], &[-2, 0, 1], &[-2, 0, 0, 1], &[1, 1, 1, 1, 1]];
        let k = field(defs[deg_pick]);
        let d = k.degree();
        let x = k.element(&a[..d].iter().copied().map(small_rational).collect::<Vec<_>>()).unwrap();
        if x.is_zero() {
            prop_assert!(x.inverse().is_err());
        } else {
            let inv = x.inverse().unwrap();
            prop_assert!(x.mul(&inv).unwrap().is_one());
            prop_assert_eq!(inv.norm(), Rational::from(x.norm().recip_ref()));
            prop_assert_eq!(x.div(&x).unwrap(), k.one());
        }
    }

    /// Every element is a root of its own minimal polynomial, whose degree
    /// divides the degree of the field.
    #[test]
    fn minimal_polynomial_annihilates(
        a in prop::collection::vec((-6i64..6, -3i64..3), 4),
    ) {
        let k = NumberField::cyclotomic(5).unwrap();
        let x = k.element(&a.iter().copied().map(small_rational).collect::<Vec<_>>()).unwrap();
        let m = x.minimal_polynomial();
        let deg = m.len() - 1;
        prop_assert_eq!(k.degree() % deg, 0);
        prop_assert_eq!(m[deg].clone(), q(1));
        let mut acc = k.zero();
        for (i, c) in m.iter().enumerate() {
            acc = acc.add(&k.rational(c).mul(&x.pow(i as u64)).unwrap()).unwrap();
        }
        prop_assert!(acc.is_zero());
    }
}

#[test]
fn non_monic_defining_polynomials_are_a_different_antic_path() {
    // A primitive integral polynomial need not be monic, and Antic takes a
    // *different* branch for those (its `nf->flag` loses the MONIC bit). The
    // union arms are chosen by degree and the monic bit independently, so the
    // non-monic path needs its own coverage.
    //
    // For f = 3x^2 - 2 the generator is sqrt(2/3): the product of the roots is
    // -c/a = -2/3 and their sum is 0.
    let k = field(&[-2, 0, 3]);
    assert_eq!(k.degree(), 2);
    assert_eq!(k.defining_polynomial(), ints(&[-2, 0, 3]).as_slice());
    let g = k.generator();
    assert_eq!(g.norm(), Rational::from((-2, 3)));
    assert_eq!(g.trace(), q(0));
    // prod(1 + a_i) = f(-1)/lc = (3 - 2)/3.
    assert_eq!(k.one().add(&g).unwrap().norm(), Rational::from((1, 3)));
    assert_eq!(g.pow(2), k.rational(&Rational::from((2, 3))));
    assert_eq!(
        g.minimal_polynomial(),
        vec![Rational::from((-2, 3)), q(0), q(1)]
    );
    assert!(g.mul(&g.inverse().unwrap()).unwrap().is_one());
    assert_eq!(k.polynomial_discriminant(), Integer::from(24));

    // And the generic (degree > 2) non-monic arm: 2x^3 - 3, root cbrt(3/2).
    let c = field(&[-3, 0, 0, 2]);
    assert_eq!(c.degree(), 3);
    let g = c.generator();
    assert_eq!(g.norm(), Rational::from((3, 2)));
    assert_eq!(g.trace(), q(0));
    assert_eq!(g.pow(3), c.rational(&Rational::from((3, 2))));
    assert!(g.mul(&g.inverse().unwrap()).unwrap().is_one());
}

#[test]
fn elements_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NumberField>();
    assert_send_sync::<NumberFieldElement>();
}

#[test]
fn clone_is_independent() {
    let k = field(&[-2, 0, 1]);
    let a = k.generator();
    let b = a.clone();
    let c = a.add(&k.one()).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    drop(a);
    assert_eq!(b.norm(), q(-2));
}
