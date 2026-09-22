//! Unit tests for the symplectic / stabilizer layer.
//!
//! The four codes at the centre of this file — Steane `[[7,1,3]]`, Shor
//! `[[9,1,3]]`, the five-qubit `[[5,1,3]]` perfect code and the `[[4,2,2]]`
//! code — are the correctness anchors. Two of them are chosen to exercise
//! paths the others do not: the five-qubit code is **not** CSS, so it goes
//! through the general symplectic centralizer computation rather than the
//! nullspace/quotient shortcut, and `[[4,2,2]]` has `k = 2`, so the logical
//! pairing (`⟨X̄_i, Z̄_j⟩ = δ_ij`) has something to get wrong.
//!
//! Where a claim can be checked two ways, it is: the CSS distance
//! `min(d_X, d_Z)` is cross-checked against the general
//! `2^(n+k)`-element centralizer search, and every matrix-group order is
//! checked against both the product formula and (where small enough) brute-force
//! enumeration and Schreier–Sims on the induced permutation action.

use super::*;
use crate::errors::AlkahestError;
use crate::ffield::FiniteField;
use std::str::FromStr;

fn gf2f() -> FiniteField {
    FiniteField::prime(2).expect("2 is prime")
}

fn mat(rows: usize, cols: usize, data: &[u64]) -> GfMatrix {
    GfMatrix::from_u64(&gf2f(), rows, cols, data).expect("well-formed test matrix")
}

fn p(s: &str) -> PauliOperator {
    PauliOperator::from_str(s).expect("well-formed Pauli string")
}

// ---------------------------------------------------------------------------
// The symplectic form
// ---------------------------------------------------------------------------

#[test]
fn form_pins_the_x_z_convention() {
    // One qubit: X = (1 | 0), Z = (0 | 1), Y = (1 | 1).
    assert_eq!(
        symplectic_form(&[1, 0], &[0, 1]).unwrap(),
        1,
        "X and Z anticommute"
    );
    assert_eq!(
        symplectic_form(&[1, 0], &[1, 1]).unwrap(),
        1,
        "X and Y anticommute"
    );
    assert_eq!(
        symplectic_form(&[0, 1], &[1, 1]).unwrap(),
        1,
        "Z and Y anticommute"
    );
    assert_eq!(
        symplectic_form(&[1, 0], &[1, 0]).unwrap(),
        0,
        "X commutes with itself"
    );
}

#[test]
fn form_is_alternating_and_symmetric() {
    // Every vector of F_2^4 is orthogonal to itself, and the form is symmetric.
    for a in 0u8..16 {
        let u: Vec<u8> = (0..4).map(|i| (a >> i) & 1).collect();
        assert_eq!(symplectic_form(&u, &u).unwrap(), 0);
        for b in 0u8..16 {
            let v: Vec<u8> = (0..4).map(|i| (b >> i) & 1).collect();
            assert_eq!(
                symplectic_form(&u, &v).unwrap(),
                symplectic_form(&v, &u).unwrap()
            );
        }
    }
}

#[test]
fn form_refuses_mismatched_lengths() {
    let e = symplectic_form(&[1, 0], &[1, 0, 0, 0]).unwrap_err();
    assert_eq!(e.code(), "E-STAB-002");
}

#[test]
fn omega_is_its_own_inverse_and_is_symplectic() {
    let omega = symplectic_gram_matrix(3).unwrap();
    let id = GfMatrix::identity(&gf2f(), 6).unwrap();
    assert!(omega.mul(&omega).unwrap().equals(&id), "Ω² = I over GF(2)");
    assert!(is_symplectic(&omega).unwrap());
    assert!(is_symplectic(&id).unwrap());
}

#[test]
fn symplectic_membership_agrees_in_both_conventions() {
    // MᵀΩM = Ω and MΩMᵀ = Ω are the same condition over GF(2). Checked on
    // every one of the 2^16 four-by-four GF(2) matrices would be slow in a
    // debug build; all 2^4 two-by-two ones plus a deterministic sample of
    // 4×4 ones is enough to catch a transposed implementation.
    let omega2 = symplectic_gram_matrix(1).unwrap();
    for code in 0u64..16 {
        let data: Vec<u64> = (0..4).map(|i| (code >> i) & 1).collect();
        let m = mat(2, 2, &data);
        let lhs = m.transpose().mul(&omega2).unwrap().mul(&m).unwrap();
        let rhs = m.mul(&omega2).unwrap().mul(&m.transpose()).unwrap();
        assert_eq!(
            lhs.equals(&omega2),
            rhs.equals(&omega2),
            "conventions disagree on {data:?}"
        );
        assert_eq!(is_symplectic(&m).unwrap(), lhs.equals(&omega2));
    }

    let omega4 = symplectic_gram_matrix(2).unwrap();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..400 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let data: Vec<u64> = (0..16).map(|i| (state >> i) & 1).collect();
        let m = mat(4, 4, &data);
        let lhs = m.transpose().mul(&omega4).unwrap().mul(&m).unwrap();
        let rhs = m.mul(&omega4).unwrap().mul(&m.transpose()).unwrap();
        assert_eq!(lhs.equals(&omega4), rhs.equals(&omega4));
    }
}

#[test]
fn symplectic_membership_refuses_a_non_square_matrix() {
    let e = is_symplectic(&mat(2, 3, &[1, 0, 0, 0, 1, 0])).unwrap_err();
    assert_eq!(e.code(), "E-STAB-002");
}

#[test]
fn complement_has_the_complementary_dimension() {
    // A single non-zero vector in F_2^4 has a 3-dimensional complement.
    let v = vec![vec![1u8, 0, 0, 0]];
    let c = symplectic_complement(&v, 2).unwrap();
    assert_eq!(c.len(), 3);
    for w in &c {
        assert_eq!(symplectic_form(&v[0], w).unwrap(), 0);
    }
    // The complement of {0} is everything.
    assert_eq!(symplectic_complement(&[], 2).unwrap().len(), 4);
}

#[test]
fn gram_schmidt_splits_a_full_space_into_hyperbolic_pairs() {
    let n = 3;
    let all: Vec<Vec<u8>> = (0..2 * n)
        .map(|i| {
            let mut e = vec![0u8; 2 * n];
            e[i] = 1;
            e
        })
        .collect();
    let hb = symplectic_gram_schmidt(&all, n).unwrap();
    assert_eq!(hb.pairs().len(), n);
    assert!(
        hb.radical().is_empty(),
        "a non-degenerate form has no radical"
    );
    assert_eq!(hb.dimension(), 2 * n);
}

#[test]
fn gram_schmidt_finds_the_radical_of_an_isotropic_space() {
    // Span of Z_1 and Z_2 in F_2^4 is totally isotropic: the whole space is
    // its own radical, and there are no hyperbolic pairs.
    let vs = vec![vec![0u8, 0, 1, 0], vec![0u8, 0, 0, 1]];
    let hb = symplectic_gram_schmidt(&vs, 2).unwrap();
    assert!(hb.pairs().is_empty());
    assert_eq!(hb.radical().len(), 2);
}

// ---------------------------------------------------------------------------
// Pauli operators
// ---------------------------------------------------------------------------

#[test]
fn pauli_strings_round_trip() {
    for s in ["+XIZY", "-XZZXI", "+iYYY", "-iZZ", "+IIII"] {
        let op = p(s);
        assert_eq!(op.to_string(), s, "round trip failed for {s}");
    }
    // A bare string is read as +.
    assert_eq!(p("XZZXI").to_string(), "+XZZXI");
}

#[test]
fn y_is_the_hermitian_y_not_the_raw_xz_product() {
    let y = p("Y");
    assert_eq!(y.phase(), 1, "Y = i·XZ, so the stored phase exponent is 1");
    assert!(y.is_hermitian());
    // The raw symplectic product XZ has phase 0 and is *not* Hermitian.
    let xz = PauliOperator::from_xz(&[1], &[1], 0).unwrap();
    assert!(!xz.is_hermitian());
    assert_eq!(xz.to_string(), "-iY");
}

#[test]
fn pauli_multiplication_carries_the_phase() {
    let x = p("X");
    let z = p("Z");
    let xz = x.mul(&z).unwrap();
    let zx = z.mul(&x).unwrap();
    assert_eq!(xz.x_bits(), &[1]);
    assert_eq!(xz.z_bits(), &[1]);
    // XZ and ZX differ by exactly −1.
    assert_eq!(zx, xz.negate());
    assert_eq!(xz.phase(), 0);
    assert_eq!(zx.phase(), 2);
    // Y = iXZ.
    assert_eq!(x.mul(&z).unwrap().mul(&p("+iI")).unwrap(), p("Y"));
}

#[test]
fn pauli_inverse_and_squares() {
    for s in ["+XIZY", "-XZZXI", "+iYYY", "-iZZ"] {
        let op = p(s);
        assert!(op.mul(&op.inverse()).unwrap().is_identity(), "{s}");
        assert!(op.inverse().mul(&op).unwrap().is_identity(), "{s}");
    }
    // A Hermitian Pauli squares to the identity.
    for s in ["+XIZY", "-XZZXI", "+YYY"] {
        let op = p(s);
        assert!(op.is_hermitian());
        assert!(op.mul(&op).unwrap().is_identity(), "{s}");
    }
}

#[test]
fn weight_ignores_the_phase() {
    assert_eq!(p("+XIZY").weight(), 3);
    assert_eq!(p("-XIZY").weight(), 3);
    assert_eq!(p("-IIII").weight(), 0);
}

#[test]
fn commutation_matches_the_symplectic_product() {
    assert!(!p("XI").commutes_with(&p("ZI")).unwrap());
    assert!(p("XI").commutes_with(&p("IZ")).unwrap());
    assert!(p("XX").commutes_with(&p("ZZ")).unwrap());
    assert!(p("XZZXI").commutes_with(&p("IXZZX")).unwrap());
}

#[test]
fn pauli_refuses_a_qubit_count_mismatch() {
    let e = p("XX").mul(&p("X")).unwrap_err();
    assert_eq!(e.code(), "E-STAB-003");
}

#[test]
fn pauli_refuses_a_bad_letter() {
    let e = PauliOperator::from_str("XQZ").unwrap_err();
    assert_eq!(e.code(), "E-STAB-010");
}

// ---------------------------------------------------------------------------
// Stabilizer groups
// ---------------------------------------------------------------------------

#[test]
fn anticommuting_generators_are_refused() {
    let e = StabilizerGroup::new(vec![p("XI"), p("ZI")]).unwrap_err();
    assert_eq!(e.code(), "E-STAB-004");
    match e {
        StabilizerError::NotCommuting { i, j } => assert_eq!((i, j), (0, 1)),
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn minus_identity_in_the_group_is_refused() {
    // Two dependent generators whose product is −I.
    let e = StabilizerGroup::new(vec![p("ZZ"), p("-ZZ")]).unwrap_err();
    assert_eq!(e.code(), "E-STAB-006");
}

#[test]
fn a_dependent_but_consistent_generating_list_is_accepted() {
    let g = StabilizerGroup::new(vec![p("ZZ"), p("ZZ")]).unwrap();
    assert_eq!(g.rank(), 1);
    assert!(!g.is_independent());
    assert_eq!(StabilizerCode::new(g).unwrap().k(), 1);
}

#[test]
fn non_hermitian_generators_are_refused() {
    let ix = PauliOperator::from_xz(&[1], &[0], 1).unwrap(); // i·X
    let e = StabilizerGroup::new(vec![ix]).unwrap_err();
    assert_eq!(e.code(), "E-STAB-010");
}

#[test]
fn membership_respects_the_sign() {
    let g = StabilizerGroup::new(vec![p("ZZI"), p("IZZ")]).unwrap();
    assert!(g.contains(&p("ZZI")).unwrap());
    assert!(g.contains(&p("ZIZ")).unwrap(), "ZZI · IZZ = ZIZ");
    assert!(
        !g.contains(&p("-ZZI")).unwrap(),
        "the sign is part of the element"
    );
    assert!(g.contains(&PauliOperator::identity(3).unwrap()).unwrap());
}

#[test]
fn syndrome_names_the_anticommuting_generators() {
    let g = StabilizerGroup::new(vec![p("ZZI"), p("IZZ")]).unwrap();
    assert_eq!(g.syndrome(&p("XII")).unwrap(), vec![1, 0]);
    assert_eq!(g.syndrome(&p("IXI")).unwrap(), vec![1, 1]);
    assert_eq!(g.syndrome(&p("IIX")).unwrap(), vec![0, 1]);
    assert_eq!(g.syndrome(&p("ZZZ")).unwrap(), vec![0, 0]);
}

// ---------------------------------------------------------------------------
// The anchor codes
// ---------------------------------------------------------------------------

/// The `[7,4,3]` Hamming parity-check matrix.
fn hamming743() -> GfMatrix {
    mat(
        3,
        7,
        &[
            0, 0, 0, 1, 1, 1, 1, //
            0, 1, 1, 0, 0, 1, 1, //
            1, 0, 1, 0, 1, 0, 1,
        ],
    )
}

/// Shor's `[[9,1,3]]` code: `X`-type checks of weight six, `Z`-type of weight two.
fn shor() -> (GfMatrix, GfMatrix) {
    let hx = mat(
        2,
        9,
        &[
            1, 1, 1, 1, 1, 1, 0, 0, 0, //
            0, 0, 0, 1, 1, 1, 1, 1, 1,
        ],
    );
    let hz = mat(
        6,
        9,
        &[
            1, 1, 0, 0, 0, 0, 0, 0, 0, //
            0, 1, 1, 0, 0, 0, 0, 0, 0, //
            0, 0, 0, 1, 1, 0, 0, 0, 0, //
            0, 0, 0, 0, 1, 1, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, 1, 1, 0, //
            0, 0, 0, 0, 0, 0, 0, 1, 1,
        ],
    );
    (hx, hz)
}

fn assert_logicals_are_well_formed(code: &StabilizerCode) {
    for (i, lx) in code.logical_x().iter().enumerate() {
        for (g, gen) in code.stabilizer().generators().iter().enumerate() {
            assert!(
                gen.commutes_with(lx).unwrap(),
                "logical X_{i} anticommutes with generator {g}"
            );
        }
    }
    for (i, lz) in code.logical_z().iter().enumerate() {
        for (g, gen) in code.stabilizer().generators().iter().enumerate() {
            assert!(
                gen.commutes_with(lz).unwrap(),
                "logical Z_{i} anticommutes with generator {g}"
            );
        }
    }
    for i in 0..code.k() {
        for j in 0..code.k() {
            let want = u8::from(i == j);
            assert_eq!(
                code.logical_x()[i]
                    .symplectic_product(&code.logical_z()[j])
                    .unwrap(),
                want,
                "⟨X_{i}, Z_{j}⟩"
            );
            if i != j {
                assert_eq!(
                    code.logical_x()[i]
                        .symplectic_product(&code.logical_x()[j])
                        .unwrap(),
                    0
                );
                assert_eq!(
                    code.logical_z()[i]
                        .symplectic_product(&code.logical_z()[j])
                        .unwrap(),
                    0
                );
            }
        }
    }
}

#[test]
fn steane_is_7_1_3() {
    let h = hamming743();
    let code = CssCode::new(&h, &h).unwrap();
    assert_eq!(code.n(), 7);
    assert_eq!(code.k(), 1);
    assert_eq!(code.x_rank(), 3);
    assert_eq!(code.z_rank(), 3);
    assert_eq!(code.minimum_distance().unwrap(), Distance::Exact(3));
    assert_eq!(code.to_string(), "CSS[[7, 1]]");
}

#[test]
fn steane_agrees_with_the_general_stabilizer_path() {
    let h = hamming743();
    let css = CssCode::new(&h, &h).unwrap();
    let general = css.to_stabilizer_code().unwrap();
    assert_eq!(general.n(), 7);
    assert_eq!(general.k(), 1);
    assert_eq!(
        general.minimum_distance().unwrap(),
        css.minimum_distance().unwrap(),
        "min(d_X, d_Z) must equal the full centralizer search"
    );
    assert_logicals_are_well_formed(&general);
}

#[test]
fn steane_satisfies_the_css_condition_and_a_perturbation_does_not() {
    let h = hamming743();
    assert!(h.mul(&h.transpose()).unwrap().is_zero());

    // Flip one entry of H_X. The pair is no longer CSS and construction refuses.
    let mut data: Vec<u64> = h.to_u64().unwrap();
    data[0] ^= 1;
    let bad = mat(3, 7, &data);
    assert!(!bad.mul(&h.transpose()).unwrap().is_zero());
    let e = CssCode::new(&bad, &h).unwrap_err();
    assert_eq!(e.code(), "E-STAB-005");
    match e {
        StabilizerError::CssConditionViolated { row, .. } => assert_eq!(row, 0),
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn shor_is_9_1_3() {
    let (hx, hz) = shor();
    let code = CssCode::new(&hx, &hz).unwrap();
    assert_eq!(code.n(), 9);
    assert_eq!(code.k(), 1);
    assert_eq!(code.x_rank(), 2);
    assert_eq!(code.z_rank(), 6);
    assert_eq!(code.minimum_distance().unwrap(), Distance::Exact(3));

    let general = code.to_stabilizer_code().unwrap();
    assert_eq!(general.k(), 1);
    assert_eq!(general.minimum_distance().unwrap(), Distance::Exact(3));
    assert_logicals_are_well_formed(&general);
}

#[test]
fn five_qubit_perfect_code_is_5_1_3() {
    // Not CSS: this exercises the general symplectic centralizer path.
    let gens = vec![p("XZZXI"), p("IXZZX"), p("XIXZZ"), p("ZXIXZ")];
    let code = StabilizerCode::from_generators(gens).unwrap();
    assert_eq!(code.n(), 5);
    assert_eq!(code.k(), 1);
    assert_eq!(code.stabilizer().rank(), 4);
    assert!(code.stabilizer().is_independent());
    assert_eq!(code.minimum_distance().unwrap(), Distance::Exact(3));
    assert_eq!(code.to_string(), "[[5, 1]]");
    assert_logicals_are_well_formed(&code);

    // The five-qubit code is perfect: the 3·5 = 15 weight-one errors have 15
    // distinct non-zero syndromes, one for each of the 2^4 − 1 possibilities.
    let mut seen = std::collections::HashSet::new();
    for q in 0..5usize {
        for letter in ['X', 'Y', 'Z'] {
            let mut s: Vec<char> = vec!['I'; 5];
            s[q] = letter;
            let err = p(&s.iter().collect::<String>());
            let syn = code.syndrome(&err).unwrap();
            assert_ne!(syn, vec![0, 0, 0, 0], "a weight-one error must be detected");
            assert!(
                seen.insert(syn),
                "syndromes must be distinct: {letter} on {q}"
            );
        }
    }
    assert_eq!(seen.len(), 15);
}

#[test]
fn four_two_two_code_has_two_logical_pairs() {
    // The [[4,2,2]] code: S = ⟨XXXX, ZZZZ⟩.
    let code = StabilizerCode::from_generators(vec![p("XXXX"), p("ZZZZ")]).unwrap();
    assert_eq!(code.n(), 4);
    assert_eq!(code.k(), 2);
    assert_eq!(code.minimum_distance().unwrap(), Distance::Exact(2));
    assert_logicals_are_well_formed(&code);
}

#[test]
fn an_empty_generator_list_is_refused_and_the_trivial_group_must_state_n() {
    let e = StabilizerGroup::new(vec![]).unwrap_err();
    assert_eq!(e.code(), "E-STAB-002");
    // The trivial group on 3 qubits is the [[3, 3]] code.
    let code = StabilizerCode::from_generators_on(3, vec![]).unwrap();
    assert_eq!((code.n(), code.k()), (3, 3));
    assert_eq!(code.stabilizer().rank(), 0);
    assert_logicals_are_well_formed(&code);
    assert_eq!(code.minimum_distance().unwrap(), Distance::Exact(1));
}

#[test]
fn a_k_zero_code_has_no_distance() {
    let code = StabilizerCode::from_generators(vec![p("Z")]).unwrap();
    assert_eq!(code.k(), 0);
    let e = code.minimum_distance().unwrap_err();
    assert_eq!(e.code(), "E-STAB-007");
}

#[test]
fn distance_search_is_capped() {
    let h = hamming743();
    let code = CssCode::new(&h, &h).unwrap().to_stabilizer_code().unwrap();
    // n + k = 8, so a cap of 4 refuses.
    let e = code.minimum_distance_with_cap(4).unwrap_err();
    assert_eq!(e.code(), "E-STAB-008");
    // The bound is still available and is clearly labelled as a bound.
    let b = code.distance_upper_bound().unwrap();
    assert!(!b.is_exact());
    assert!(
        b.value() >= 3,
        "an upper bound cannot be below the true distance"
    );
    assert!(b.to_string().starts_with('≤'));
}

#[test]
fn logical_errors_are_exactly_the_undetectable_ones() {
    let code =
        StabilizerCode::from_generators(vec![p("XZZXI"), p("IXZZX"), p("XIXZZ"), p("ZXIXZ")])
            .unwrap();
    let lx = &code.logical_x()[0];
    assert!(code.is_logical_error(lx).unwrap());
    assert!(code.syndrome(lx).unwrap().iter().all(|&b| b == 0));
    // A stabilizer element is not a logical error.
    assert!(!code.is_logical_error(&p("XZZXI")).unwrap());
    // A weight-one error is detected, so it is not in N(S) at all.
    assert!(!code.is_logical_error(&p("XIIII")).unwrap());
}

#[test]
fn a_css_pair_with_mismatched_widths_is_refused() {
    let e = CssCode::new(&hamming743(), &mat(1, 5, &[1, 1, 1, 1, 1])).unwrap_err();
    assert_eq!(e.code(), "E-STAB-002");
}

#[test]
fn a_non_binary_field_is_refused() {
    let gf3 = FiniteField::prime(3).unwrap();
    let h = GfMatrix::from_u64(&gf3, 1, 3, &[1, 1, 1]).unwrap();
    let e = CssCode::new(&h, &h).unwrap_err();
    assert_eq!(e.code(), "E-STAB-001");
}

// ---------------------------------------------------------------------------
// Matrix groups
// ---------------------------------------------------------------------------

#[test]
fn classical_group_orders_match_the_product_formulas() {
    let gf2 = gf2f();
    let gf3 = FiniteField::prime(3).unwrap();
    let gf4 = FiniteField::extension(2, 2).unwrap();

    assert_eq!(MatrixGroup::general_linear(&gf2, 2).unwrap().order(), 6);
    assert_eq!(MatrixGroup::general_linear(&gf2, 3).unwrap().order(), 168);
    assert_eq!(MatrixGroup::general_linear(&gf2, 4).unwrap().order(), 20160);
    assert_eq!(MatrixGroup::symplectic(&gf2, 1).unwrap().order(), 6);
    assert_eq!(MatrixGroup::symplectic(&gf2, 2).unwrap().order(), 720);
    assert_eq!(MatrixGroup::symplectic(&gf2, 3).unwrap().order(), 1451520);

    // |GL(2,3)| = (9−1)(9−3) = 48, |SL(2,3)| = 48/2 = 24.
    assert_eq!(MatrixGroup::general_linear(&gf3, 2).unwrap().order(), 48);
    assert_eq!(MatrixGroup::special_linear(&gf3, 2).unwrap().order(), 24);
    // |Sp(2,q)| = |SL(2,q)|.
    assert_eq!(
        MatrixGroup::symplectic(&gf3, 1).unwrap().order(),
        MatrixGroup::special_linear(&gf3, 2).unwrap().order()
    );
    // Over GF(4): |GL(2,4)| = (16−1)(16−4) = 180.
    assert_eq!(MatrixGroup::general_linear(&gf4, 2).unwrap().order(), 180);
    assert_eq!(MatrixGroup::special_linear(&gf4, 2).unwrap().order(), 60);

    // Arbitrary precision, well past u64: |GL(10, 2^31−1)| has 300-odd digits.
    let big = FiniteField::prime(2147483647).unwrap();
    assert!(MatrixGroup::general_linear(&big, 10).unwrap().order() > u64::MAX);
}

#[test]
fn enumeration_agrees_with_the_order_formula() {
    let gf2 = gf2f();
    let gf3 = FiniteField::prime(3).unwrap();
    for g in [
        MatrixGroup::general_linear(&gf2, 2).unwrap(),
        MatrixGroup::general_linear(&gf2, 3).unwrap(),
        MatrixGroup::symplectic(&gf2, 1).unwrap(),
        MatrixGroup::symplectic(&gf2, 2).unwrap(),
        MatrixGroup::general_linear(&gf3, 2).unwrap(),
        MatrixGroup::special_linear(&gf3, 2).unwrap(),
    ] {
        let elems = g.elements().unwrap();
        assert_eq!(
            rug::Integer::from(elems.len()),
            g.order(),
            "brute force disagrees with the formula for {g}"
        );
    }
}

#[test]
fn enumeration_is_capped_but_the_order_is_not() {
    let gf3 = FiniteField::prime(3).unwrap();
    let g = MatrixGroup::symplectic(&gf3, 2).unwrap();
    let e = g.elements().unwrap_err();
    assert_eq!(e.code(), "E-STAB-011");
    // |Sp(4,3)| = 3^4 · (3^2−1)(3^4−1) = 81 · 8 · 80 = 51840.
    assert_eq!(g.order(), 51840);
}

#[test]
fn permutation_action_reproduces_the_order_via_schreier_sims() {
    let gf2 = gf2f();
    // GL(3,2) ≅ PSL(2,7) acts faithfully on the 7 non-zero vectors of F_2^3.
    let g = MatrixGroup::general_linear(&gf2, 3).unwrap();
    let perm = g.permutation_action().unwrap();
    assert_eq!(perm.degree(), 7);
    assert_eq!(perm.order().unwrap(), 168);
    assert!(perm.is_transitive());

    // Sp(4,2) ≅ S_6 acts faithfully on the 15 non-zero vectors of F_2^4.
    let s = MatrixGroup::symplectic(&gf2, 2).unwrap();
    let sperm = s.permutation_action().unwrap();
    assert_eq!(sperm.degree(), 15);
    assert_eq!(sperm.order().unwrap(), 720);
}

#[test]
fn membership_matches_the_defining_condition() {
    let gf2 = gf2f();
    let gl = MatrixGroup::general_linear(&gf2, 2).unwrap();
    let sl = MatrixGroup::special_linear(&gf2, 2).unwrap();
    let id = GfMatrix::identity(&gf2, 2).unwrap();
    assert!(gl.contains(&id).unwrap());
    assert!(sl.contains(&id).unwrap());
    // Over GF(2), det ≠ 0 means det = 1, so GL(n,2) = SL(n,2).
    assert_eq!(gl.order(), sl.order());
    let singular = mat(2, 2, &[1, 1, 1, 1]);
    assert!(!gl.contains(&singular).unwrap());

    let sp = MatrixGroup::symplectic(&gf2, 1).unwrap();
    assert!(sp.contains(&id).unwrap());
    // [[1,1],[0,1]] is a transvection and lies in Sp(2,2) = SL(2,2).
    assert!(sp.contains(&mat(2, 2, &[1, 1, 0, 1])).unwrap());
}

#[test]
fn membership_refuses_the_wrong_field_and_the_wrong_shape() {
    let gf2 = gf2f();
    let gf3 = FiniteField::prime(3).unwrap();
    let gl = MatrixGroup::general_linear(&gf2, 2).unwrap();
    let wrong_field = GfMatrix::identity(&gf3, 2).unwrap();
    assert_eq!(gl.contains(&wrong_field).unwrap_err().code(), "E-STAB-012");
    let wrong_shape = GfMatrix::identity(&gf2, 3).unwrap();
    assert_eq!(gl.contains(&wrong_shape).unwrap_err().code(), "E-STAB-002");
}

#[test]
fn every_enumerated_symplectic_matrix_preserves_the_form() {
    let gf2 = gf2f();
    let sp = MatrixGroup::symplectic(&gf2, 2).unwrap();
    let omega = symplectic_gram_matrix(2).unwrap();
    for m in sp.elements().unwrap() {
        assert!(m
            .transpose()
            .mul(&omega)
            .unwrap()
            .mul(&m)
            .unwrap()
            .equals(&omega));
    }
}

#[test]
fn matrix_group_display_and_degree() {
    let gf2 = gf2f();
    assert_eq!(
        MatrixGroup::general_linear(&gf2, 3).unwrap().to_string(),
        "GL(3, 2)"
    );
    assert_eq!(
        MatrixGroup::symplectic(&gf2, 2).unwrap().to_string(),
        "Sp(4, 2)"
    );
    assert_eq!(MatrixGroup::symplectic(&gf2, 2).unwrap().degree(), 4);
    assert_eq!(
        MatrixGroup::general_linear(&gf2, 0).unwrap_err().code(),
        "E-STAB-002"
    );
}

// ---------------------------------------------------------------------------
// Error-code coverage
// ---------------------------------------------------------------------------

#[test]
fn every_stab_code_has_a_registry_entry() {
    use crate::errors::codes::REGISTRY;
    for n in 1..=13 {
        let code = format!("E-STAB-{n:03}");
        assert!(
            REGISTRY.iter().any(|s| s.code == code),
            "{code} is missing from the registry"
        );
    }
}

#[test]
fn a_gfq_failure_keeps_its_own_code_across_the_boundary() {
    let e: StabilizerError = crate::ffield::FiniteFieldError::Inconsistent.into();
    assert_eq!(e.code(), "E-GFQ-010");
}
