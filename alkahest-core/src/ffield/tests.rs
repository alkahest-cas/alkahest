//! Unit and property tests for GF(q) linear algebra.
//!
//! The FFI round-trip tests at the top are load-bearing, not ceremony.
//! `nmod_mat_struct` / `fq_nmod_mat_struct` changed layout in FLINT 3.1 and a
//! wrong guess is silent memory corruption, not a compile error — so every
//! entry of several *non-square* shapes is written and read back with a value
//! that identifies its own position. A square-only or 1-column test passes for
//! the wrong stride; these do not.

use super::*;
use crate::errors::AlkahestError;
use proptest::prelude::*;

fn gf(p: u64) -> FiniteField {
    FiniteField::prime(p).expect("test prime")
}

fn gf2() -> FiniteField {
    gf(2)
}

/// GF(2³) from x³ + x + 1, pinned rather than taken from Conway so the tests
/// state their own coordinates.
fn gf8() -> FiniteField {
    FiniteField::with_defining_polynomial(2, &[1, 1, 0, 1]).expect("x^3+x+1 is irreducible")
}

/// GF(4) from x² + x + 1, the only irreducible quadratic over GF(2).
fn gf4() -> FiniteField {
    FiniteField::with_defining_polynomial(2, &[1, 1, 1]).expect("x^2+x+1 is irreducible")
}

fn err_code<T: std::fmt::Debug>(r: Result<T, FiniteFieldError>) -> &'static str {
    r.expect_err("expected a refusal").code()
}

/// Shapes chosen so a wrong row stride cannot survive: wide, tall, prime
/// dimensions, and a degenerate single row / single column.
const SHAPES: &[(usize, usize)] = &[
    (1, 7),
    (7, 1),
    (3, 5),
    (5, 3),
    (2, 13),
    (13, 2),
    (4, 4),
    (1, 1),
    (6, 11),
];

// ---------------------------------------------------------------------------
// FFI layout: entry round-trips
// ---------------------------------------------------------------------------

#[test]
fn prime_entry_round_trip_survives_every_shape() {
    // A modulus large enough that every entry gets a *distinct* value: with
    // p = 2 a transposed or mis-strided read has a 50% chance of matching.
    let f = gf(1_000_003);
    for &(r, c) in SHAPES {
        let entries: Vec<u64> = (0..r * c).map(|t| (t as u64) + 1).collect();
        let m = GfMatrix::from_u64(&f, r, c, &entries).unwrap();
        assert_eq!(m.shape(), (r, c), "shape {r}x{c}");
        for i in 0..r {
            for j in 0..c {
                let want = (i * c + j) as u64 + 1;
                assert_eq!(
                    m.entry(i, j).unwrap().as_u64(),
                    Some(want),
                    "GF(1000003) entry ({i},{j}) of a {r}x{c} matrix"
                );
            }
        }
        assert_eq!(m.to_u64().unwrap(), entries, "row-major readback {r}x{c}");
    }
}

#[test]
fn extension_entry_round_trip_survives_every_shape() {
    // GF(3^4): 81 distinct elements, enough to label every entry of every
    // shape above uniquely.
    let f = FiniteField::extension(3, 4).unwrap();
    assert_eq!(f.degree(), 4);
    for &(r, c) in SHAPES {
        let entries: Vec<FieldElement> = (0..r * c)
            .map(|t| {
                let t = t as u64;
                f.element(&[t % 3, (t / 3) % 3, (t / 9) % 3, (t / 27) % 3])
                    .unwrap()
            })
            .collect();
        let m = GfMatrix::from_elements(&f, r, c, &entries).unwrap();
        assert_eq!(m.shape(), (r, c));
        for i in 0..r {
            for j in 0..c {
                assert_eq!(
                    m.entry(i, j).unwrap(),
                    entries[i * c + j],
                    "GF(3^4) entry ({i},{j}) of a {r}x{c} matrix"
                );
            }
        }
        assert_eq!(m.to_elements(), entries);
    }
}

#[test]
fn set_entry_writes_where_it_says_it_does() {
    // A wrong stride most often shows up as a write landing in the wrong row.
    let f = gf(101);
    let (r, c) = (5, 3);
    let mut m = GfMatrix::zeros(&f, r, c).unwrap();
    for i in 0..r {
        for j in 0..c {
            m.set_entry(i, j, &f.scalar((7 * i + j) as u64 + 1))
                .unwrap();
            // Every previously written entry must still hold its own value.
            for a in 0..=i {
                for b in 0..c {
                    if a * c + b > i * c + j {
                        continue;
                    }
                    assert_eq!(
                        m.entry(a, b).unwrap().as_u64(),
                        Some((7 * a + b) as u64 + 1),
                        "write to ({i},{j}) disturbed ({a},{b})"
                    );
                }
            }
        }
    }
}

#[test]
fn fq_ctx_buffer_has_generous_slack() {
    // The opaque `fq_nmod_ctx_t` buffer is deliberately over-sized. If a future
    // FLINT grows the struct past it the failure is heap corruption, so the
    // margin is asserted rather than assumed.
    for (p, k) in [(2u64, 3u32), (3, 4), (5, 2), (7, 6)] {
        let f = FiniteField::extension(p, k).unwrap();
        let slack = f
            .ctx_untouched_tail()
            .expect("an extension field has a ctx");
        assert!(
            slack >= 128,
            "fq_nmod_ctx_t for GF({p}^{k}) left only {slack} spare bytes of 512"
        );
    }
}

#[test]
fn a_word_sized_prime_near_the_top_of_the_range_works() {
    // The Goldilocks prime 2^64 - 2^32 + 1, which ZK work actually uses.
    let p = 0xFFFF_FFFF_0000_0001u64;
    let f = FiniteField::prime(p).expect("Goldilocks is prime");
    assert_eq!(f.characteristic(), p);
    let a = GfMatrix::from_u64(&f, 2, 2, &[1, 2, 3, 4]).unwrap();
    assert_eq!(a.determinant().unwrap().as_u64(), Some(p - 2)); // 4 - 6 = -2
    let inv = a.inverse().unwrap();
    assert!(a
        .mul(&inv)
        .unwrap()
        .equals(&GfMatrix::identity(&f, 2).unwrap()));
}

// ---------------------------------------------------------------------------
// Field construction and refusals
// ---------------------------------------------------------------------------

#[test]
fn a_composite_characteristic_is_refused() {
    for n in [0u64, 1, 4, 6, 9, 100, 1_000_000] {
        assert_eq!(err_code(FiniteField::prime(n)), "E-GFQ-001", "n = {n}");
    }
    assert!(FiniteField::prime(2).is_ok());
    assert!(FiniteField::prime(65_537).is_ok());
}

#[test]
fn a_characteristic_past_a_machine_word_is_refused_separately() {
    let too_big = u128::from(u64::MAX) + 1;
    assert_eq!(err_code(FiniteField::prime_wide(too_big)), "E-GFQ-002");
    // …and a value that does fit still goes through the primality test.
    assert_eq!(err_code(FiniteField::prime_wide(12)), "E-GFQ-001");
    assert!(FiniteField::prime_wide(13).is_ok());

    // The arbitrary-precision entry point names the value it refused, which is
    // the whole reason it takes a string: an overflow in the conversion would
    // name nothing.
    let huge = "1".to_string() + &"0".repeat(120) + "7";
    let err = FiniteField::prime_from_decimal(&huge).expect_err("120 digits will not fit");
    assert_eq!(err.code(), "E-GFQ-002");
    assert!(err.to_string().contains(&huge), "{err}");

    // Negative and non-numeric are "not a prime characteristic", not "too big".
    assert_eq!(err_code(FiniteField::prime_from_decimal("-7")), "E-GFQ-001");
    assert_eq!(
        err_code(FiniteField::prime_from_decimal("seven")),
        "E-GFQ-001"
    );
    assert!(FiniteField::prime_from_decimal(" 13 ").is_ok());
}

#[test]
fn an_out_of_range_extension_degree_is_refused() {
    assert_eq!(err_code(FiniteField::extension(2, 0)), "E-GFQ-003");
    assert_eq!(
        err_code(FiniteField::extension(2, MAX_EXTENSION_DEGREE as u32 + 1)),
        "E-GFQ-003"
    );
    // Degree 1 is the prime field, not an error.
    let f = FiniteField::extension(7, 1).unwrap();
    assert!(f.is_prime_field());
    assert_eq!(f, gf(7));
}

#[test]
fn a_reducible_defining_polynomial_is_refused() {
    // x^2 + 1 = (x + 1)^2 over GF(2).
    assert_eq!(
        err_code(FiniteField::with_defining_polynomial(2, &[1, 0, 1])),
        "E-GFQ-004"
    );
    // x^2 + x = x(x + 1) over GF(2).
    assert_eq!(
        err_code(FiniteField::with_defining_polynomial(2, &[0, 1, 1])),
        "E-GFQ-004"
    );
    // x^3 - 2x = x(x^2 - 2) over GF(5), and 2 is a QR mod 5? (3^2 = 4, 2^2 = 4)
    // — irrelevant: the factor x already makes it reducible.
    assert_eq!(
        err_code(FiniteField::with_defining_polynomial(5, &[0, 3, 0, 1])),
        "E-GFQ-004"
    );
    // A constant has no quotient field at all.
    assert_eq!(
        err_code(FiniteField::with_defining_polynomial(2, &[1])),
        "E-GFQ-004"
    );
    // Non-monic: 2x^2 + x + 1 over GF(3) is irreducible but is refused, because
    // normalising it would silently return a polynomial the caller did not write.
    assert_eq!(
        err_code(FiniteField::with_defining_polynomial(3, &[1, 1, 2])),
        "E-GFQ-004"
    );
    // The monic scaling of the same polynomial (multiply by 2^-1 = 2 mod 3:
    // x^2 + 2x + 2) is accepted.
    assert!(FiniteField::with_defining_polynomial(3, &[2, 2, 1]).is_ok());
}

#[test]
fn an_irreducible_defining_polynomial_builds_the_field() {
    let f = gf8();
    assert_eq!(f.characteristic(), 2);
    assert_eq!(f.degree(), 3);
    assert_eq!(f.order(), Some(8));
    assert_eq!(f.defining_polynomial(), &[1, 1, 0, 1]);
    assert_eq!(f.to_string(), "GF(2^3)");
    assert_eq!(f.render(&f.generator().unwrap()), "a");
}

#[test]
fn two_presentations_of_the_same_order_are_different_fields() {
    // x^3 + x + 1 and x^3 + x^2 + 1 are both irreducible over GF(2). The fields
    // are isomorphic; the coordinates are not interchangeable.
    let a = gf8();
    let b = FiniteField::with_defining_polynomial(2, &[1, 0, 1, 1]).unwrap();
    assert_ne!(a, b);

    let ma = GfMatrix::from_u64(&a, 1, 1, &[1]).unwrap();
    let mb = GfMatrix::from_u64(&b, 1, 1, &[1]).unwrap();
    assert_eq!(err_code(ma.add(&mb)), "E-GFQ-006");
    assert!(!ma.equals(&mb));
}

#[test]
fn an_element_with_too_many_coordinates_is_refused() {
    let f = gf8();
    assert_eq!(err_code(f.element(&[1, 0, 0, 1])), "E-GFQ-005");
    assert!(f.element(&[1, 0, 1]).is_ok());
    // Coefficients *are* reduced mod p — that is what "mod p" means.
    assert_eq!(
        f.element(&[3, 2, 5]).unwrap(),
        f.element(&[1, 0, 1]).unwrap()
    );
}

#[test]
fn field_mismatch_is_caught_before_the_shapes_are_looked_at() {
    let a = GfMatrix::from_u64(&gf2(), 2, 2, &[1, 0, 0, 1]).unwrap();
    let b = GfMatrix::from_u64(&gf(3), 3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
    assert_eq!(err_code(a.add(&b)), "E-GFQ-006");
    assert_eq!(err_code(a.mul(&b)), "E-GFQ-006");
    assert_eq!(err_code(a.solve(&b)), "E-GFQ-006");
}

#[test]
fn shape_mismatches_are_refused() {
    let f = gf(5);
    let a = GfMatrix::from_u64(&f, 2, 3, &[1, 2, 3, 4, 0, 1]).unwrap();
    let b = GfMatrix::from_u64(&f, 3, 2, &[1, 0, 0, 1, 1, 1]).unwrap();
    assert_eq!(err_code(a.add(&b)), "E-GFQ-007");
    assert_eq!(err_code(a.sub(&b)), "E-GFQ-007");
    assert!(a.mul(&b).is_ok());
    assert_eq!(err_code(a.mul(&a)), "E-GFQ-007");
    assert_eq!(err_code(a.determinant()), "E-GFQ-008");
    assert_eq!(err_code(a.charpoly()), "E-GFQ-008");
    assert_eq!(err_code(a.inverse()), "E-GFQ-008");
    assert_eq!(err_code(a.entry(2, 0)), "E-GFQ-011");
    assert_eq!(err_code(a.entry(0, 3)), "E-GFQ-011");
    assert_eq!(
        err_code(GfMatrix::from_u64(&f, 2, 3, &[1, 2, 3])),
        "E-GFQ-007"
    );
}

#[test]
fn every_registered_gfq_code_is_reachable() {
    // Codes 001..012, each raised by an operation in this file's reach.
    let seen: Vec<&'static str> = vec![
        FiniteFieldError::NotPrime {
            modulus: "4".to_string(),
        }
        .code(),
        FiniteFieldError::ModulusTooLarge {
            modulus: "0".to_string(),
        }
        .code(),
        FiniteFieldError::DegreeOutOfRange { degree: 0 }.code(),
        FiniteFieldError::BadDefiningPolynomial {
            reason: String::new(),
        }
        .code(),
        FiniteFieldError::MalformedElement { got: 4, degree: 3 }.code(),
        FiniteFieldError::FieldMismatch {
            lhs: String::new(),
            rhs: String::new(),
        }
        .code(),
        FiniteFieldError::DimensionMismatch {
            op: "add",
            lhs: (1, 1),
            rhs: (2, 2),
        }
        .code(),
        FiniteFieldError::NotSquare {
            op: "det",
            shape: (1, 2),
        }
        .code(),
        FiniteFieldError::Singular { shape: (2, 2) }.code(),
        FiniteFieldError::Inconsistent.code(),
        FiniteFieldError::IndexOutOfBounds {
            i: 0,
            j: 0,
            rows: 0,
            cols: 0,
        }
        .code(),
        FiniteFieldError::DimensionTooLarge {
            rows: 1 << 40,
            cols: 1 << 40,
        }
        .code(),
    ];
    let mut expected: Vec<String> = (1..=12).map(|n| format!("E-GFQ-{n:03}")).collect();
    expected.sort();
    let mut got: Vec<String> = seen.iter().map(|s| s.to_string()).collect();
    got.sort();
    assert_eq!(got, expected);
    for spec in crate::errors::codes::REGISTRY {
        if spec.code.starts_with("E-GFQ-") {
            assert_eq!(spec.class, "FiniteFieldError");
            assert!(
                spec.remediation.is_some(),
                "{} has no remediation",
                spec.code
            );
        }
    }
    // The remediation really is wired through to the error value.
    assert!(FiniteFieldError::Inconsistent.remediation().is_some());
}

#[test]
fn an_absurd_shape_is_refused_rather_than_allocated() {
    let f = gf2();
    assert_eq!(err_code(GfMatrix::zeros(&f, 1 << 40, 1 << 40)), "E-GFQ-012");
    assert_eq!(
        err_code(GfMatrix::zeros(&f, usize::MAX, usize::MAX)),
        "E-GFQ-012"
    );
}

// ---------------------------------------------------------------------------
// Arithmetic against values computed by hand
// ---------------------------------------------------------------------------

#[test]
fn addition_and_multiplication_match_hand_computation() {
    let f = gf(7);
    // A = [[1,2,3],[4,5,6]]  B = [[6,5,4],[3,2,1]]
    let a = GfMatrix::from_u64(&f, 2, 3, &[1, 2, 3, 4, 5, 6]).unwrap();
    let b = GfMatrix::from_u64(&f, 2, 3, &[6, 5, 4, 3, 2, 1]).unwrap();
    // A + B = [[7,7,7],[7,7,7]] = 0 mod 7
    assert!(a.add(&b).unwrap().is_zero());
    // A - B = [[-5,-3,-1],[1,3,5]] = [[2,4,6],[1,3,5]]
    assert_eq!(a.sub(&b).unwrap().to_u64().unwrap(), vec![2, 4, 6, 1, 3, 5]);
    // A * Bᵀ where Bᵀ is 3x2: row0·col0 = 1*6+2*5+3*4 = 28 = 0 mod 7
    //                        row0·col1 = 1*3+2*2+3*1 = 10 = 3
    //                        row1·col0 = 4*6+5*5+6*4 = 73 = 3
    //                        row1·col1 = 4*3+5*2+6*1 = 28 = 0
    let prod = a.mul(&b.transpose()).unwrap();
    assert_eq!(prod.shape(), (2, 2));
    assert_eq!(prod.to_u64().unwrap(), vec![0, 3, 3, 0]);
    // 3·A = [[3,6,9],[12,15,18]] = [[3,6,2],[5,1,4]]
    assert_eq!(
        a.scalar_mul(&f.scalar(3)).unwrap().to_u64().unwrap(),
        vec![3, 6, 2, 5, 1, 4]
    );
    // -A
    assert!(a.add(&a.neg()).unwrap().is_zero());
}

#[test]
fn transpose_of_a_rectangle_is_the_rectangle_transposed() {
    let f = gf(11);
    let a = GfMatrix::from_u64(&f, 2, 3, &[1, 2, 3, 4, 5, 6]).unwrap();
    let t = a.transpose();
    assert_eq!(t.shape(), (3, 2));
    assert_eq!(t.to_u64().unwrap(), vec![1, 4, 2, 5, 3, 6]);
    assert!(t.transpose().equals(&a));
}

#[test]
fn gf4_arithmetic_matches_the_multiplication_table() {
    let f = gf4();
    let one = f.one();
    let a = f.generator().unwrap(); // a, with a² = a + 1
    let a1 = f.element(&[1, 1]).unwrap(); // a + 1

    // [[a, 1], [1, a]] · [[a, 1], [1, a]] = [[a²+1, a+a],[a+a, 1+a²]]
    //                                     = [[a, 0], [0, a]]
    let m = GfMatrix::from_elements(&f, 2, 2, &[a.clone(), one.clone(), one.clone(), a.clone()])
        .unwrap();
    let sq = m.mul(&m).unwrap();
    assert_eq!(
        sq.to_elements(),
        vec![a.clone(), f.zero(), f.zero(), a.clone()]
    );

    // det = a·a - 1 = a² + 1 = a
    assert_eq!(m.determinant().unwrap(), a);
    // charpoly = x² - tr·x + det = x² + a   (trace a + a = 0 in char 2)
    assert_eq!(
        m.charpoly().unwrap(),
        vec![a.clone(), f.zero(), one.clone()]
    );
    // A·A⁻¹ = I
    let inv = m.inverse().unwrap();
    assert!(m
        .mul(&inv)
        .unwrap()
        .equals(&GfMatrix::identity(&f, 2).unwrap()));

    // (a + 1)·a = a² + a = 1, so scaling by a+1 then by a is the identity.
    let scaled = m.scalar_mul(&a1).unwrap().scalar_mul(&a).unwrap();
    assert!(scaled.equals(&m));
}

// ---------------------------------------------------------------------------
// rank / rref / nullspace
// ---------------------------------------------------------------------------

/// The [7,4] Hamming code's parity-check matrix.
fn hamming74() -> GfMatrix {
    GfMatrix::from_u64(
        &gf2(),
        3,
        7,
        &[
            1, 0, 1, 0, 1, 0, 1, //
            0, 1, 1, 0, 0, 1, 1, //
            0, 0, 0, 1, 1, 1, 1,
        ],
    )
    .unwrap()
}

#[test]
fn the_hamming_parity_check_matrix_behaves_like_a_74_code() {
    let h = hamming74();
    assert_eq!(h.rank(), 3);

    let g_t = h.nullspace().unwrap();
    assert_eq!(
        g_t.shape(),
        (7, 4),
        "a [7,4] code has a 4-dimensional kernel"
    );
    assert_eq!(h.rank() + g_t.ncols(), h.ncols(), "rank + nullity = ncols");

    // H · Gᵀ = 0 — every basis vector is a codeword.
    assert!(h.mul(&g_t).unwrap().is_zero());

    // The kernel basis is itself independent.
    assert_eq!(g_t.rank(), 4);

    // …and the transpose is a generator matrix: G · Hᵀ = 0.
    let g = g_t.transpose();
    assert!(g.mul(&h.transpose()).unwrap().is_zero());
}

#[test]
fn rref_returns_a_transform_that_actually_transforms() {
    let f = gf(5);
    // Tall, wide, square, rank-deficient, and zero.
    let cases: Vec<GfMatrix> = vec![
        GfMatrix::from_u64(&f, 5, 3, &[1, 2, 3, 2, 4, 6, 0, 1, 1, 3, 0, 2, 1, 1, 1]).unwrap(),
        GfMatrix::from_u64(&f, 3, 5, &[1, 2, 3, 4, 0, 2, 4, 6, 8, 0, 1, 0, 1, 0, 1]).unwrap(),
        GfMatrix::from_u64(&f, 4, 4, &[1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4, 1, 1, 1, 4]).unwrap(),
        GfMatrix::zeros(&f, 3, 4).unwrap(),
        hamming74(),
    ];
    for a in cases {
        let (m, n) = a.shape();
        let r = a.rref().unwrap();
        assert_eq!(r.matrix.shape(), (m, n));
        assert_eq!(r.transform.shape(), (m, m));

        // U · A = R.
        assert!(
            r.transform.mul(&a).unwrap().equals(&r.matrix),
            "U*A != R for {a:?}"
        );
        // U is invertible — a row transform must not lose information.
        assert!(r.transform.inverse().is_ok());
        // The rank agrees with the independent computation.
        assert_eq!(
            r.rank,
            a.rank(),
            "rref rank disagrees with rank() for {a:?}"
        );
        assert_eq!(r.pivots.len(), r.rank);
        // Pivot columns strictly ascend, and each pivot is a leading 1 whose
        // column is otherwise zero — the defining property of *reduced* echelon.
        for w in r.pivots.windows(2) {
            assert!(
                w[0] < w[1],
                "pivots not strictly increasing: {:?}",
                r.pivots
            );
        }
        for (i, &j) in r.pivots.iter().enumerate() {
            assert_eq!(r.matrix.entry(i, j).unwrap(), f.one());
            for t in 0..m {
                if t != i {
                    assert!(r.matrix.entry(t, j).unwrap().is_zero());
                }
            }
        }
        // Rows past the rank are zero.
        for i in r.rank..m {
            for j in 0..n {
                assert!(r.matrix.entry(i, j).unwrap().is_zero());
            }
        }
    }
}

#[test]
fn rref_matches_a_textbook_reduction() {
    // Over GF(5):  [[2, 4, 1], [1, 3, 2]]
    // R1 <- 3·R1 (3 = 2⁻¹) -> [1, 2, 3]; R2 <- R2 - R1 -> [0, 1, 4]
    // R1 <- R1 - 2·R2 -> [1, 0, 0]
    // rref = [[1,0,0],[0,1,4]]
    let f = gf(5);
    let a = GfMatrix::from_u64(&f, 2, 3, &[2, 4, 1, 1, 3, 2]).unwrap();
    let r = a.rref().unwrap();
    assert_eq!(r.rank, 2);
    assert_eq!(r.pivots, vec![0, 1]);
    assert_eq!(r.matrix.to_u64().unwrap(), vec![1, 0, 0, 0, 1, 4]);
    assert!(r.transform.mul(&a).unwrap().equals(&r.matrix));
}

#[test]
fn nullspace_of_a_nonsingular_square_matrix_is_empty() {
    let f = gf(13);
    let a = GfMatrix::from_u64(&f, 3, 3, &[2, 1, 0, 0, 3, 1, 1, 0, 4]).unwrap();
    assert_eq!(a.rank(), 3);
    let n = a.nullspace().unwrap();
    assert_eq!(n.shape(), (3, 0));
    assert_eq!(a.rank() + n.ncols(), 3);
}

#[test]
fn nullspace_of_the_zero_matrix_is_everything() {
    let f = gf(3);
    let z = GfMatrix::zeros(&f, 4, 6).unwrap();
    assert_eq!(z.rank(), 0);
    let n = z.nullspace().unwrap();
    assert_eq!(n.shape(), (6, 6));
    assert_eq!(n.rank(), 6);
    assert!(z.mul(&n).unwrap().is_zero());
}

#[test]
fn nullspace_handles_a_wide_rank_deficient_matrix() {
    // 2x5 over GF(3), second row a multiple of the first: rank 1, nullity 4.
    let f = gf(3);
    let a = GfMatrix::from_u64(&f, 2, 5, &[1, 2, 0, 1, 1, 2, 1, 0, 2, 2]).unwrap();
    assert_eq!(a.rank(), 1);
    let n = a.nullspace().unwrap();
    assert_eq!(n.ncols(), 4);
    assert_eq!(a.rank() + n.ncols(), a.ncols());
    assert!(a.mul(&n).unwrap().is_zero());
    assert_eq!(n.rank(), 4);
}

#[test]
fn nullspace_works_over_an_extension_field() {
    let f = gf4();
    let a = f.generator().unwrap();
    let one = f.one();
    let zero = f.zero();
    // Careful: [[1, a, a+1], [a, a+1, 1]] looks like two independent rows and is
    // not — the second is a times the first (a*a = a+1, a*(a+1) = 1). Use a row
    // that genuinely escapes the span.
    let a1 = f.element(&[1, 1]).unwrap();
    let dependent = GfMatrix::from_elements(
        &f,
        2,
        3,
        &[
            one.clone(),
            a.clone(),
            a1.clone(),
            a.clone(),
            a1.clone(),
            one.clone(),
        ],
    )
    .unwrap();
    assert_eq!(dependent.rank(), 1, "row 2 = a * row 1 over GF(4)");
    assert_eq!(dependent.nullspace().unwrap().ncols(), 2);

    // H = [[1, a, a+1], [a, 1, 0]]: rank 2, nullity 1.
    let h = GfMatrix::from_elements(
        &f,
        2,
        3,
        &[
            one.clone(),
            a.clone(),
            a1.clone(),
            a.clone(),
            one.clone(),
            f.zero(),
        ],
    )
    .unwrap();
    assert_eq!(h.rank(), 2);
    let n = h.nullspace().unwrap();
    assert_eq!(h.rank() + n.ncols(), h.ncols());
    assert_eq!(n.ncols(), 1);
    assert!(h.mul(&n).unwrap().is_zero());
    assert!(!n.is_zero(), "a nullspace basis vector is never zero");
    assert_ne!(zero, one);
}

#[test]
fn rref_works_over_an_extension_field() {
    let f = gf8();
    let one = f.one();
    let a = f.generator().unwrap();
    let m = GfMatrix::from_elements(
        &f,
        3,
        4,
        &[
            a.clone(),
            one.clone(),
            f.zero(),
            a.clone(),
            one.clone(),
            a.clone(),
            one.clone(),
            f.zero(),
            f.zero(),
            f.zero(),
            one.clone(),
            one.clone(),
        ],
    )
    .unwrap();
    let r = m.rref().unwrap();
    assert_eq!(r.rank, m.rank());
    assert!(r.transform.mul(&m).unwrap().equals(&r.matrix));
    assert!(r.transform.inverse().is_ok());
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

#[test]
fn solve_recovers_a_known_solution() {
    let f = gf(7);
    let a = GfMatrix::from_u64(&f, 3, 3, &[2, 1, 1, 1, 3, 2, 1, 0, 4]).unwrap();
    let x = GfMatrix::from_u64(&f, 3, 1, &[5, 2, 6]).unwrap();
    let b = a.mul(&x).unwrap();
    let got = a.solve(&b).unwrap();
    assert_eq!(got.shape(), (3, 1));
    assert!(got.equals(&x), "a nonsingular system has a unique solution");
}

#[test]
fn solve_returns_a_particular_solution_for_an_underdetermined_system() {
    let f = gf2();
    let h = hamming74();
    // A syndrome that is reachable: take the third column of H.
    let e = GfMatrix::from_u64(&f, 7, 1, &[0, 0, 1, 0, 0, 0, 0]).unwrap();
    let s = h.mul(&e).unwrap();
    let x = h.solve(&s).unwrap();
    assert_eq!(x.shape(), (7, 1));
    assert!(h.mul(&x).unwrap().equals(&s), "A·x must reproduce b");
}

#[test]
fn an_inconsistent_system_is_refused_rather_than_approximated() {
    let f = gf(5);
    // A has rank 1; b is outside its column space.
    let a = GfMatrix::from_u64(&f, 3, 2, &[1, 2, 2, 4, 3, 1]).unwrap();
    let b = GfMatrix::from_u64(&f, 3, 1, &[1, 0, 0]).unwrap();
    assert_eq!(err_code(a.solve(&b)), "E-GFQ-010");

    // Row-count mismatch is a different refusal.
    let wrong = GfMatrix::from_u64(&f, 2, 1, &[1, 0]).unwrap();
    assert_eq!(err_code(a.solve(&wrong)), "E-GFQ-007");
}

#[test]
fn solve_works_over_an_extension_field() {
    let f = gf8();
    let a = f.generator().unwrap();
    let one = f.one();
    let m = GfMatrix::from_elements(&f, 2, 2, &[a.clone(), one.clone(), one.clone(), a.clone()])
        .unwrap();
    let x = GfMatrix::from_elements(&f, 2, 1, &[f.element(&[1, 1]).unwrap(), a.clone()]).unwrap();
    let b = m.mul(&x).unwrap();
    assert!(m.solve(&b).unwrap().equals(&x));
}

// ---------------------------------------------------------------------------
// inverse / determinant / charpoly
// ---------------------------------------------------------------------------

#[test]
fn determinant_matches_hand_computation() {
    // [[1,2],[3,4]] has det -2.
    for p in [5u64, 7, 11, 65_537] {
        let f = gf(p);
        let a = GfMatrix::from_u64(&f, 2, 2, &[1, 2, 3, 4]).unwrap();
        assert_eq!(a.determinant().unwrap().as_u64(), Some(p - 2), "p = {p}");
    }
    // A singular 3x3 over GF(3): rows 1 and 3 are equal.
    let f = gf(3);
    let s = GfMatrix::from_u64(&f, 3, 3, &[1, 2, 0, 0, 1, 1, 1, 2, 0]).unwrap();
    assert!(s.determinant().unwrap().is_zero());
    assert_eq!(err_code(s.inverse()), "E-GFQ-009");
    // Upper triangular: det is the product of the diagonal.
    let t = GfMatrix::from_u64(&gf(11), 3, 3, &[2, 5, 7, 0, 3, 1, 0, 0, 4]).unwrap();
    assert_eq!(t.determinant().unwrap().as_u64(), Some(24 % 11));
}

#[test]
fn determinant_over_an_extension_field_matches_the_diagonal_product() {
    let f = gf8();
    let a = f.generator().unwrap();
    let a2 = f.element(&[0, 0, 1]).unwrap();
    // Upper triangular with diagonal (a, a², 1): det = a·a²·1 = a³ = a + 1.
    let m = GfMatrix::from_elements(
        &f,
        3,
        3,
        &[
            a.clone(),
            f.one(),
            f.one(),
            f.zero(),
            a2.clone(),
            f.one(),
            f.zero(),
            f.zero(),
            f.one(),
        ],
    )
    .unwrap();
    assert_eq!(m.determinant().unwrap(), f.element(&[1, 1]).unwrap());

    // Odd dimension exercises the (-1)^n branch on an odd characteristic too.
    let g = FiniteField::extension(3, 2).unwrap();
    let diag = GfMatrix::from_u64(&g, 3, 3, &[2, 0, 0, 0, 2, 0, 0, 0, 2]).unwrap();
    // det = 2³ = 8 = 2 mod 3.
    assert_eq!(diag.determinant().unwrap(), g.scalar(2));
}

#[test]
fn charpoly_matches_hand_computation() {
    // [[0,1],[1,0]] over GF(2): det(xI - A) = x² - 1 = x² + 1.
    let f = gf2();
    let a = GfMatrix::from_u64(&f, 2, 2, &[0, 1, 1, 0]).unwrap();
    assert_eq!(
        a.charpoly()
            .unwrap()
            .iter()
            .map(FieldElement::as_u64)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(0), Some(1)]
    );

    // [[1,2],[3,4]] over GF(7): x² - 5x - 2 = x² + 2x + 5.
    let g = gf(7);
    let b = GfMatrix::from_u64(&g, 2, 2, &[1, 2, 3, 4]).unwrap();
    assert_eq!(
        b.charpoly()
            .unwrap()
            .iter()
            .map(FieldElement::as_u64)
            .collect::<Vec<_>>(),
        vec![Some(5), Some(2), Some(1)]
    );

    // The identity: (x - 1)^n. For n = 3 over GF(5): x³ - 3x² + 3x - 1
    //                                             = x³ + 2x² + 3x + 4.
    let h = gf(5);
    let i3 = GfMatrix::identity(&h, 3).unwrap();
    assert_eq!(
        i3.charpoly()
            .unwrap()
            .iter()
            .map(FieldElement::as_u64)
            .collect::<Vec<_>>(),
        vec![Some(4), Some(3), Some(2), Some(1)]
    );
}

/// Evaluate `Σ cᵢ Aⁱ`.
fn eval_poly_at(coeffs: &[FieldElement], a: &GfMatrix) -> GfMatrix {
    let f = a.field().clone();
    let n = a.nrows();
    let mut acc = GfMatrix::zeros(&f, n, n).unwrap();
    let mut power = GfMatrix::identity(&f, n).unwrap();
    for c in coeffs {
        acc = acc.add(&power.scalar_mul(c).unwrap()).unwrap();
        power = power.mul(a).unwrap();
    }
    acc
}

#[test]
fn cayley_hamilton_holds_for_the_computed_charpoly() {
    // The strongest available cross-check on charpoly that does not restate it:
    // a matrix must annihilate its own characteristic polynomial.
    let cases: Vec<GfMatrix> = vec![
        GfMatrix::from_u64(&gf(5), 3, 3, &[1, 2, 3, 0, 4, 1, 2, 0, 3]).unwrap(),
        GfMatrix::from_u64(
            &gf2(),
            4,
            4,
            &[1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        )
        .unwrap(),
        GfMatrix::from_u64(&gf(7), 2, 2, &[1, 2, 3, 4]).unwrap(),
        GfMatrix::zeros(&gf(3), 3, 3).unwrap(),
    ];
    for a in cases {
        let cp = a.charpoly().unwrap();
        assert_eq!(cp.len(), a.nrows() + 1, "charpoly degree must equal n");
        assert_eq!(cp.last().unwrap(), &a.field().one(), "charpoly is monic");
        assert!(
            eval_poly_at(&cp, &a).is_zero(),
            "Cayley–Hamilton failed for {a:?}"
        );
    }

    // …and over an extension field.
    let f = gf4();
    let g = f.generator().unwrap();
    let m = GfMatrix::from_elements(&f, 2, 2, &[g.clone(), f.one(), f.one(), g.clone()]).unwrap();
    let cp = m.charpoly().unwrap();
    assert!(eval_poly_at(&cp, &m).is_zero());
}

#[test]
fn inverse_matches_hand_computation() {
    // [[1,2],[3,4]] over GF(5): det = -2 = 3, 3⁻¹ = 2 mod 5.
    // adj = [[4,-2],[-3,1]] = [[4,3],[2,1]], inv = 2·adj = [[3,1],[4,2]]
    let f = gf(5);
    let a = GfMatrix::from_u64(&f, 2, 2, &[1, 2, 3, 4]).unwrap();
    let inv = a.inverse().unwrap();
    assert_eq!(inv.to_u64().unwrap(), vec![3, 1, 4, 2]);
    assert!(a
        .mul(&inv)
        .unwrap()
        .equals(&GfMatrix::identity(&f, 2).unwrap()));
    assert!(inv
        .mul(&a)
        .unwrap()
        .equals(&GfMatrix::identity(&f, 2).unwrap()));
}

#[test]
fn clone_is_independent_of_the_original() {
    let f = gf(7);
    let a = GfMatrix::from_u64(&f, 2, 3, &[1, 2, 3, 4, 5, 6]).unwrap();
    let mut b = a.clone();
    assert!(a.equals(&b));
    b.set_entry(0, 0, &f.scalar(6)).unwrap();
    assert_eq!(a.entry(0, 0).unwrap().as_u64(), Some(1));
    assert_eq!(b.entry(0, 0).unwrap().as_u64(), Some(6));

    let g = gf8();
    let c = GfMatrix::from_u64(&g, 2, 3, &[1, 0, 1, 0, 1, 1]).unwrap();
    let mut d = c.clone();
    d.set_entry(1, 2, &g.generator().unwrap()).unwrap();
    assert_eq!(c.entry(1, 2).unwrap(), g.one());
    assert_eq!(d.entry(1, 2).unwrap(), g.generator().unwrap());
}

#[test]
fn degenerate_shapes_do_not_panic() {
    // A zero-row or zero-column matrix is degenerate, not invalid, and a
    // caller building one from a filter can hit it. FLINT handles empty
    // matrices; the wrappers must not index into them.
    let f = gf(5);

    let empty_rows = GfMatrix::from_u64(&f, 0, 3, &[]).unwrap();
    assert_eq!(empty_rows.shape(), (0, 3));
    assert_eq!(empty_rows.rank(), 0);
    assert!(empty_rows.is_zero());
    let n = empty_rows.nullspace().unwrap();
    assert_eq!(
        n.shape(),
        (3, 3),
        "nothing constrains x, so the kernel is all of it"
    );
    assert_eq!(empty_rows.rank() + n.ncols(), 3);
    let r = empty_rows.rref().unwrap();
    assert_eq!(r.rank, 0);
    assert_eq!(r.matrix.shape(), (0, 3));
    assert_eq!(r.transform.shape(), (0, 0));

    let empty_cols = GfMatrix::from_u64(&f, 3, 0, &[]).unwrap();
    assert_eq!(empty_cols.shape(), (3, 0));
    assert_eq!(empty_cols.rank(), 0);
    assert_eq!(empty_cols.nullspace().unwrap().shape(), (0, 0));
    assert_eq!(empty_cols.transpose().shape(), (0, 3));
    assert!(empty_cols.rref().is_ok());

    // The empty matrix: det of a 0x0 is the empty product, 1.
    let nothing = GfMatrix::zeros(&f, 0, 0).unwrap();
    assert_eq!(nothing.determinant().unwrap(), f.one());
    assert_eq!(nothing.charpoly().unwrap(), vec![f.one()]);
    assert!(nothing.inverse().is_ok());

    // The same degenerate shapes over an extension field, where the entry type
    // is a polynomial rather than a word.
    let e = gf4();
    let ext_empty = GfMatrix::zeros(&e, 0, 0).unwrap();
    assert_eq!(ext_empty.determinant().unwrap(), e.one());
    assert_eq!(ext_empty.charpoly().unwrap(), vec![e.one()]);
    let ext_wide = GfMatrix::zeros(&e, 0, 3).unwrap();
    assert_eq!(ext_wide.rank(), 0);
    assert_eq!(ext_wide.nullspace().unwrap().shape(), (3, 3));
    assert!(ext_wide.rref().is_ok());

    // A 1x1 zero matrix is singular; a 1x1 non-zero one is not.
    let z1 = GfMatrix::zeros(&f, 1, 1).unwrap();
    assert_eq!(err_code(z1.inverse()), "E-GFQ-009");
    assert_eq!(z1.nullspace().unwrap().shape(), (1, 1));
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

fn small_prime() -> impl Strategy<Value = u64> {
    prop::sample::select(vec![2u64, 3, 5, 7, 11, 13, 251])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Every entry comes back exactly as it went in, for every shape.
    #[test]
    fn prop_entries_round_trip(
        p in small_prime(),
        rows in 1usize..7,
        cols in 1usize..7,
        seed in any::<u64>(),
    ) {
        let f = gf(p);
        let entries: Vec<u64> =
            (0..rows * cols).map(|t| seed.wrapping_mul(t as u64 + 1) % p).collect();
        let m = GfMatrix::from_u64(&f, rows, cols, &entries).unwrap();
        for i in 0..rows {
            for j in 0..cols {
                prop_assert_eq!(m.entry(i, j).unwrap().as_u64(), Some(entries[i * cols + j]));
            }
        }
    }

    /// rank + nullity = ncols, for any shape.
    #[test]
    fn prop_rank_nullity(
        p in small_prime(),
        rows in 1usize..6,
        cols in 1usize..6,
        v in prop::collection::vec(0u64..251, 25),
    ) {
        let f = gf(p);
        let entries: Vec<u64> = v.into_iter().take(rows * cols).map(|x| x % p).collect();
        prop_assume!(entries.len() == rows * cols);
        let a = GfMatrix::from_u64(&f, rows, cols, &entries).unwrap();
        let n = a.nullspace().unwrap();
        prop_assert_eq!(a.rank() + n.ncols(), cols);
        prop_assert_eq!(n.nrows(), cols);
        // A · N = 0, and the basis is independent.
        prop_assert!(a.mul(&n).unwrap().is_zero());
        prop_assert_eq!(n.rank(), n.ncols());
    }

    /// A·A⁻¹ = A⁻¹·A = I whenever the inverse exists, and the inverse exists
    /// exactly when the determinant is non-zero.
    #[test]
    fn prop_inverse_round_trip(p in small_prime(), n in 1usize..5, seed in any::<u64>()) {
        let f = gf(p);
        let entries: Vec<u64> = (0..n * n)
            .map(|t| seed.rotate_left(t as u32 * 7).wrapping_add(t as u64) % p)
            .collect();
        let a = GfMatrix::from_u64(&f, n, n, &entries).unwrap();
        let singular = a.determinant().unwrap().is_zero();
        match a.inverse() {
            Ok(inv) => {
                prop_assert!(!singular, "a singular matrix must not invert");
                let id = GfMatrix::identity(&f, n).unwrap();
                prop_assert!(a.mul(&inv).unwrap().equals(&id));
                prop_assert!(inv.mul(&a).unwrap().equals(&id));
                prop_assert_eq!(a.rank(), n);
            }
            Err(e) => {
                prop_assert!(singular, "a nonsingular matrix must invert");
                prop_assert_eq!(e.code(), "E-GFQ-009");
                prop_assert!(a.rank() < n);
            }
        }
    }

    /// Rank is invariant under an invertible row transform — and `rref` hands
    /// one back, so this also checks that `U` really is invertible.
    #[test]
    fn prop_rank_invariant_under_row_operations(
        p in small_prime(),
        rows in 1usize..5,
        cols in 1usize..5,
        seed in any::<u64>(),
    ) {
        let f = gf(p);
        let entries: Vec<u64> =
            (0..rows * cols).map(|t| seed.rotate_right(t as u32 * 5) % p).collect();
        let a = GfMatrix::from_u64(&f, rows, cols, &entries).unwrap();
        let r = a.rref().unwrap();
        prop_assert_eq!(r.rank, a.rank());
        prop_assert!(r.transform.mul(&a).unwrap().equals(&r.matrix));
        prop_assert_eq!(r.matrix.rank(), a.rank());
        // The reduced row echelon form is *unique*, so asserting its defining
        // properties pins R exactly: ascending pivots, a leading 1 whose column
        // is otherwise zero, and nothing below the rank.
        prop_assert_eq!(r.pivots.len(), r.rank);
        for w in r.pivots.windows(2) {
            prop_assert!(w[0] < w[1]);
        }
        for (i, &j) in r.pivots.iter().enumerate() {
            prop_assert_eq!(r.matrix.entry(i, j).unwrap(), f.one());
            for t in 0..rows {
                if t != i {
                    prop_assert!(r.matrix.entry(t, j).unwrap().is_zero());
                }
            }
        }
        for i in r.rank..rows {
            for j in 0..cols {
                prop_assert!(r.matrix.entry(i, j).unwrap().is_zero());
            }
        }
        // Rank is also invariant under transposition.
        prop_assert_eq!(a.transpose().rank(), a.rank());
    }

    /// Whenever `solve` returns, its answer satisfies the system; whenever it
    /// refuses, no solution exists (checked via the rank criterion).
    #[test]
    fn prop_solve_is_sound(
        p in small_prime(),
        rows in 1usize..5,
        cols in 1usize..5,
        seed in any::<u64>(),
    ) {
        let f = gf(p);
        let entries: Vec<u64> =
            (0..rows * cols).map(|t| seed.wrapping_mul(t as u64 + 3) % p).collect();
        let a = GfMatrix::from_u64(&f, rows, cols, &entries).unwrap();
        let rhs: Vec<u64> = (0..rows).map(|t| seed.rotate_left(t as u32 * 11) % p).collect();
        let b = GfMatrix::from_u64(&f, rows, 1, &rhs).unwrap();
        match a.solve(&b) {
            Ok(x) => {
                prop_assert_eq!(x.shape(), (cols, 1));
                prop_assert!(a.mul(&x).unwrap().equals(&b));
            }
            Err(e) => {
                prop_assert_eq!(e.code(), "E-GFQ-010");
                // No solution means b is outside the column space, i.e. the
                // augmented matrix has strictly greater rank.
                let mut aug = GfMatrix::zeros(&f, rows, cols + 1).unwrap();
                for i in 0..rows {
                    for j in 0..cols {
                        aug.set_entry(i, j, &a.entry(i, j).unwrap()).unwrap();
                    }
                    aug.set_entry(i, cols, &b.entry(i, 0).unwrap()).unwrap();
                }
                prop_assert!(aug.rank() > a.rank());
            }
        }
    }

    /// Matrix multiplication is associative and distributes over addition —
    /// a cheap end-to-end check on the FLINT product for rectangular shapes.
    #[test]
    fn prop_mul_is_associative(p in small_prime(), seed in any::<u64>()) {
        let f = gf(p);
        let mk = |off: u64, r: usize, c: usize| {
            let v: Vec<u64> =
                (0..r * c).map(|t| seed.wrapping_mul(t as u64 + off) % p).collect();
            GfMatrix::from_u64(&f, r, c, &v).unwrap()
        };
        let a = mk(1, 2, 3);
        let b = mk(5, 3, 4);
        let c = mk(9, 4, 2);
        prop_assert!(a.mul(&b).unwrap().mul(&c).unwrap().equals(&a.mul(&b.mul(&c).unwrap()).unwrap()));
        let b2 = mk(13, 3, 4);
        prop_assert!(
            a.mul(&b.add(&b2).unwrap()).unwrap()
                .equals(&a.mul(&b).unwrap().add(&a.mul(&b2).unwrap()).unwrap())
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// The same invariants over an extension field, where the entry layout is
    /// `fq_nmod_struct`-strided rather than word-strided.
    #[test]
    fn prop_extension_rank_nullity(
        rows in 1usize..5,
        cols in 1usize..5,
        seed in any::<u64>(),
    ) {
        let f = gf4();
        let entries: Vec<FieldElement> = (0..rows * cols)
            .map(|t| {
                let v = seed.rotate_left(t as u32 * 3) % 4;
                f.element(&[v % 2, v / 2]).unwrap()
            })
            .collect();
        let a = GfMatrix::from_elements(&f, rows, cols, &entries).unwrap();
        for i in 0..rows {
            for j in 0..cols {
                prop_assert_eq!(a.entry(i, j).unwrap(), entries[i * cols + j].clone());
            }
        }
        let n = a.nullspace().unwrap();
        prop_assert_eq!(a.rank() + n.ncols(), cols);
        prop_assert!(a.mul(&n).unwrap().is_zero());
    }
}

/// Independent integration check: the binary Hamming [7,4,3] code.
///
/// H has the non-zero binary 3-vectors as columns, so rank 3, nullity 4, and
/// the nullspace *is* the Hamming code. Minimum distance is verified by brute
/// force over all 16 codewords, which pins the code rather than just its shape.
#[test]
fn hamming_7_4_3_code_over_gf2() {
    let f = FiniteField::prime(2).unwrap();
    let h = GfMatrix::from_u64(
        &f,
        3,
        7,
        &[
            0, 0, 0, 1, 1, 1, 1, //
            0, 1, 1, 0, 0, 1, 1, //
            1, 0, 1, 0, 1, 0, 1,
        ],
    )
    .unwrap();

    assert_eq!(h.rank(), 3, "parity-check matrix must have full row rank");
    let n = h.nullspace().unwrap();
    assert_eq!(n.ncols(), 4, "k = n - rank = 7 - 3 = 4");
    assert_eq!(n.nrows(), 7);
    assert_eq!(h.rank() + n.ncols(), h.ncols(), "rank-nullity");

    // Every basis vector is a codeword: H * N == 0.
    assert!(h.mul(&n).unwrap().is_zero(), "H * N must vanish");

    // Brute-force the 16 codewords c = N * m and take the minimum weight.
    let mut min_w = usize::MAX;
    for mask in 0u32..16 {
        let m = GfMatrix::from_u64(
            &f,
            4,
            1,
            &[
                (mask & 1) as u64,
                ((mask >> 1) & 1) as u64,
                ((mask >> 2) & 1) as u64,
                ((mask >> 3) & 1) as u64,
            ],
        )
        .unwrap();
        let c = n.mul(&m).unwrap();
        assert!(
            h.mul(&c).unwrap().is_zero(),
            "every codeword satisfies H c = 0"
        );
        let w = c.to_u64().unwrap().iter().filter(|&&b| b != 0).count();
        if mask != 0 && w < min_w {
            min_w = w;
        }
    }
    assert_eq!(min_w, 3, "Hamming [7,4] has minimum distance 3");
}
