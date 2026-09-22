//! Property tests for the symplectic / stabilizer layer.
//!
//! Each invariant here is cheap to state and the kind of thing that fails
//! silently when the `(x | z)` layout is transposed or an index is off by `n`:
//! the form is alternating and bilinear, `dim V + dim V^⊥ = 2n`, the two
//! `Sp(2n, 2)` membership conventions agree, `k = n − rank(S)`, and every
//! logical operator commutes with every stabilizer generator.

use super::*;
use proptest::prelude::*;

/// A bit vector of the given length.
fn bits(len: usize) -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(0u8..2, len)
}

/// `n` qubits (1..=4) and two `(x | z)` vectors on them.
fn arb_pair() -> impl Strategy<Value = (usize, Vec<u8>, Vec<u8>)> {
    (1usize..=4).prop_flat_map(|n| (Just(n), bits(2 * n), bits(2 * n)))
}

/// `n` qubits and a list of 0..=4 vectors on them.
fn arb_subspace() -> impl Strategy<Value = (usize, Vec<Vec<u8>>)> {
    (1usize..=4).prop_flat_map(|n| (Just(n), proptest::collection::vec(bits(2 * n), 0..=4usize)))
}

/// `n` columns and an `r × n` GF(2) matrix, read as `X`-type checks.
fn arb_check_matrix() -> impl Strategy<Value = (usize, Vec<Vec<u8>>)> {
    (2usize..=6).prop_flat_map(|n| (Just(n), proptest::collection::vec(bits(n), 0..=3usize)))
}

proptest! {
    /// The form is alternating: `⟨u, u⟩ = 0` for every `u`, in every dimension.
    #[test]
    fn form_is_alternating((_n, u, _v) in arb_pair()) {
        prop_assert_eq!(symplectic_form(&u, &u).unwrap(), 0);
    }

    /// Over GF(2) the form is symmetric, because `−1 = 1`.
    #[test]
    fn form_is_symmetric((_n, u, v) in arb_pair()) {
        prop_assert_eq!(
            symplectic_form(&u, &v).unwrap(),
            symplectic_form(&v, &u).unwrap()
        );
    }

    /// Bilinearity in the first slot.
    #[test]
    fn form_is_bilinear((n, u, v) in arb_pair(), w in bits(8)) {
        let w: Vec<u8> = w.into_iter().take(2 * n).collect();
        prop_assume!(w.len() == 2 * n);
        let sum: Vec<u8> = u.iter().zip(&v).map(|(a, b)| a ^ b).collect();
        prop_assert_eq!(
            symplectic_form(&sum, &w).unwrap(),
            symplectic_form(&u, &w).unwrap() ^ symplectic_form(&v, &w).unwrap()
        );
    }

    /// `dim V + dim V^⊥ = 2n`, and every complement vector really is orthogonal
    /// to every generator of `V`.
    #[test]
    fn complement_has_complementary_dimension((n, vs) in arb_subspace()) {
        let dim = bit_rank(&vs, 2 * n).unwrap();
        let comp = symplectic_complement(&vs, n).unwrap();
        prop_assert_eq!(comp.len(), 2 * n - dim);
        for c in &comp {
            for v in &vs {
                prop_assert_eq!(symplectic_form(v, c).unwrap(), 0);
            }
        }
    }

    /// `V^⊥⊥ = V` for the span of any set of vectors.
    #[test]
    fn double_complement_is_the_span((n, vs) in arb_subspace()) {
        let dim = bit_rank(&vs, 2 * n).unwrap();
        let comp = symplectic_complement(&vs, n).unwrap();
        let double = symplectic_complement(&comp, n).unwrap();
        prop_assert_eq!(bit_rank(&double, 2 * n).unwrap(), dim);
        let mut joint = double.clone();
        joint.extend(vs.iter().cloned());
        prop_assert_eq!(bit_rank(&joint, 2 * n).unwrap(), dim);
    }

    /// Gram–Schmidt preserves the dimension and splits it as
    /// `2·(pairs) + (radical)`.
    #[test]
    fn gram_schmidt_preserves_dimension((n, vs) in arb_subspace()) {
        let dim = bit_rank(&vs, 2 * n).unwrap();
        let hb = symplectic_gram_schmidt(&vs, n).unwrap();
        prop_assert_eq!(hb.dimension(), dim);
        prop_assert_eq!(2 * hb.pairs().len() + hb.radical().len(), dim);
    }

    /// `MᵀΩM = Ω` and `MΩMᵀ = Ω` are the same condition over GF(2).
    #[test]
    fn symplectic_conventions_agree(n in 1usize..=2, data in proptest::collection::vec(0u64..2, 16)) {
        let d = 2 * n;
        let entries: Vec<u64> = data.into_iter().take(d * d).collect();
        prop_assume!(entries.len() == d * d);
        let f = FiniteField::prime(2).unwrap();
        let m = GfMatrix::from_u64(&f, d, d, &entries).unwrap();
        let omega = symplectic_gram_matrix(n).unwrap();
        let by_columns = m.transpose().mul(&omega).unwrap().mul(&m).unwrap().equals(&omega);
        let by_rows = m.mul(&omega).unwrap().mul(&m.transpose()).unwrap().equals(&omega);
        prop_assert_eq!(by_columns, by_rows);
        prop_assert_eq!(is_symplectic(&m).unwrap(), by_columns);
    }

    /// A Pauli times its inverse is the identity, phase included.
    #[test]
    fn pauli_inverse_round_trips((n, u, _v) in arb_pair(), phase in 0u8..4) {
        let op = PauliOperator::from_symplectic(&u, phase).unwrap();
        prop_assert!(op.mul(&op.inverse()).unwrap().is_identity());
        prop_assert_eq!(op.qubits(), n);
    }

    /// Multiplication agrees with the symplectic sum on the bits, and the
    /// commutator is exactly the symplectic form: `PQ = (−1)^⟨P,Q⟩ QP`.
    #[test]
    fn commutator_is_the_symplectic_form((_n, u, v) in arb_pair(), a in 0u8..4, b in 0u8..4) {
        let pu = PauliOperator::from_symplectic(&u, a).unwrap();
        let pv = PauliOperator::from_symplectic(&v, b).unwrap();
        let uv = pu.mul(&pv).unwrap();
        let vu = pv.mul(&pu).unwrap();
        prop_assert_eq!(uv.x_bits(), vu.x_bits());
        prop_assert_eq!(uv.z_bits(), vu.z_bits());
        let expected = if symplectic_form(&u, &v).unwrap() == 0 { vu.clone() } else { vu.negate() };
        prop_assert_eq!(uv, expected);
    }

    /// A Hermitian Pauli squares to `+I`.
    #[test]
    fn hermitian_paulis_are_involutions((n, u, _v) in arb_pair(), negative in any::<bool>()) {
        let op = PauliOperator::hermitian(&u[..n], &u[n..], negative).unwrap();
        prop_assert!(op.is_hermitian());
        prop_assert!(op.mul(&op).unwrap().is_identity());
    }

    /// For an `X`-type stabilizer group from an arbitrary GF(2) check matrix:
    /// `k = n − rank(S)`, every logical commutes with every generator, and the
    /// logical pairs anticommute exactly on the diagonal.
    #[test]
    fn x_type_codes_have_k_equal_n_minus_rank((n, rows) in arb_check_matrix()) {
        let zero = vec![0u8; n];
        let gens: Vec<PauliOperator> = rows
            .iter()
            .map(|r| PauliOperator::hermitian(r, &zero, false).unwrap())
            .collect();
        let group = StabilizerGroup::with_qubits(n, gens).unwrap();
        let rank = bit_rank(&rows, n).unwrap();
        prop_assert_eq!(group.rank(), rank);

        let code = StabilizerCode::new(group).unwrap();
        prop_assert_eq!(code.k(), n - rank);
        prop_assert_eq!(code.logical_x().len(), n - rank);

        for lx in code.logical_x() {
            for g in code.stabilizer().generators() {
                prop_assert!(g.commutes_with(lx).unwrap());
            }
        }
        for lz in code.logical_z() {
            for g in code.stabilizer().generators() {
                prop_assert!(g.commutes_with(lz).unwrap());
            }
        }
        for i in 0..code.k() {
            for j in 0..code.k() {
                prop_assert_eq!(
                    code.logical_x()[i].symplectic_product(&code.logical_z()[j]).unwrap(),
                    u8::from(i == j)
                );
            }
        }
    }

    /// A random CSS pair built as `(A, a subset of ker A)` is accepted, and its
    /// `k` from `n − rank(H_X) − rank(H_Z)` agrees with `n − rank(S)` computed
    /// on the symplectic side.
    #[test]
    fn random_css_pairs_agree_with_the_general_path((n, rows) in arb_check_matrix(), take in 0usize..=3) {
        let f = FiniteField::prime(2).unwrap();
        let hx = if rows.is_empty() {
            GfMatrix::zeros(&f, 0, n).unwrap()
        } else {
            matrix_from_rows(&rows, n).unwrap()
        };
        // ker(H_X) as rows; any subset of it is a valid H_Z.
        let kernel = if rows.is_empty() {
            (0..n).map(|i| { let mut e = vec![0u8; n]; e[i] = 1; e }).collect::<Vec<_>>()
        } else {
            matrix_cols(&hx.nullspace().unwrap()).unwrap()
        };
        let chosen: Vec<Vec<u8>> = kernel.into_iter().take(take).collect();
        let hz = if chosen.is_empty() {
            GfMatrix::zeros(&f, 0, n).unwrap()
        } else {
            matrix_from_rows(&chosen, n).unwrap()
        };

        let css = CssCode::new(&hx, &hz).unwrap();
        prop_assert_eq!(css.k(), n - css.x_rank() - css.z_rank());
        let general = css.to_stabilizer_code().unwrap();
        prop_assert_eq!(general.k(), css.k());
    }

    /// Perturbing a valid CSS pair so that `H_X · H_Zᵀ ≠ 0` is always refused,
    /// and always with `E-STAB-005`.
    #[test]
    fn a_broken_css_condition_is_always_refused(n in 2usize..=5, i in 0usize..5, j in 0usize..5) {
        use crate::errors::AlkahestError;
        prop_assume!(i < n && j < n && i != j);
        let f = FiniteField::prime(2).unwrap();
        // H_X = e_i, H_Z = e_j: orthogonal, so CSS. Then flip H_Z to e_i.
        let mut ex = vec![0u8; n];
        ex[i] = 1;
        let mut ez = vec![0u8; n];
        ez[j] = 1;
        let hx = matrix_from_rows(&[ex.clone()], n).unwrap();
        prop_assert!(CssCode::new(&hx, &matrix_from_rows(&[ez], n).unwrap()).is_ok());
        let bad = matrix_from_rows(&[ex], n).unwrap();
        let e = CssCode::new(&hx, &bad).unwrap_err();
        prop_assert_eq!(e.code(), "E-STAB-005");
        let _ = f;
    }
}
