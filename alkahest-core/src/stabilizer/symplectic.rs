//! The binary symplectic form on `F_2^{2n}`, and the linear algebra that goes
//! with it.
//!
//! # Convention
//!
//! Vectors are in **`(x | z)` layout**: `2n` bits, the first `n` being the `X`
//! exponents and the last `n` the `Z` exponents. The form is
//!
//! ```text
//!     ⟨(x₁ | z₁), (x₂ | z₂)⟩  =  x₁·z₂  +  z₁·x₂          (mod 2)
//! ```
//!
//! equivalently `u Ω vᵀ` with `Ω = [[0, I], [I, 0]]`. See the
//! [module docs](crate::stabilizer) for why this matters more than it looks
//! like it should.
//!
//! Three facts about this form are used throughout and are worth having in
//! front of you:
//!
//! * It is **alternating**: `⟨u, u⟩ = x·z + z·x = 0` for every `u`, over GF(2)
//!   as over any field. There are therefore no "unit vectors" to normalise, and
//!   the Gram–Schmidt here builds *hyperbolic pairs* rather than an orthonormal
//!   basis.
//! * Over GF(2) it is **symmetric** as well (`⟨u, v⟩ = ⟨v, u⟩`), because
//!   `−1 = 1`. Several of the projection formulas below are one term shorter
//!   than their characteristic-zero counterparts for that reason, and are
//!   *wrong* in odd characteristic.
//! * `Ω² = I`, so `Ω⁻¹ = Ω`. This is why `MᵀΩM = Ω` and `MΩMᵀ = Ω` are the
//!   same condition here (see [`is_symplectic`]).

use super::{
    bit_add, bit_rank, check_qubits, gf2, matrix_cols, matrix_from_rows, matrix_rows,
    StabilizerError,
};
use crate::ffield::{FiniteField, GfMatrix};

/// The symplectic form `⟨u, v⟩ = u_x·v_z + u_z·v_x` over GF(2).
///
/// Both arguments are in `(x | z)` layout and must have the same **even**
/// length. The result is `0` (the two Paulis commute) or `1` (they
/// anticommute).
///
/// # Errors
///
/// `E-STAB-002` when the lengths differ or are odd.
///
/// ```
/// use alkahest_cas::experimental::symplectic_form;
///
/// // X on one qubit is (1 | 0); Z is (0 | 1). They anticommute.
/// assert_eq!(symplectic_form(&[1, 0], &[0, 1]).unwrap(), 1);
/// // The form is alternating: every vector is orthogonal to itself.
/// assert_eq!(symplectic_form(&[1, 1], &[1, 1]).unwrap(), 0);
/// ```
pub fn symplectic_form(u: &[u8], v: &[u8]) -> Result<u8, StabilizerError> {
    if u.len() != v.len() || u.len() % 2 != 0 {
        return Err(StabilizerError::ShapeMismatch {
            op: "symplectic form",
            expected: "two vectors of the same even length 2n".to_string(),
            got: format!("lengths {} and {}", u.len(), v.len()),
        });
    }
    let n = u.len() / 2;
    let mut acc = 0u8;
    for i in 0..n {
        acc ^= (u[i] & v[n + i]) ^ (u[n + i] & v[i]);
    }
    Ok(acc & 1)
}

/// Same as [`symplectic_form`] but without the length check.
///
/// Used on the hot paths inside this module, where lengths were validated once
/// at the entry point.
#[inline]
pub(crate) fn form_unchecked(u: &[u8], v: &[u8]) -> u8 {
    let n = u.len() / 2;
    let mut acc = 0u8;
    for i in 0..n {
        acc ^= (u[i] & v[n + i]) ^ (u[n + i] & v[i]);
    }
    acc & 1
}

/// The Gram matrix `Ω = [[0, I], [I, 0]]` of the form over GF(2), `2n × 2n`.
///
/// # Errors
///
/// `E-STAB-009` above [`MAX_QUBITS`](super::MAX_QUBITS); `E-GFQ-012` for a
/// shape FLINT cannot allocate.
pub fn symplectic_gram_matrix(n: usize) -> Result<GfMatrix, StabilizerError> {
    symplectic_gram_matrix_over(gf2(), n)
}

/// The Gram matrix `Ω = [[0, I], [−I, 0]]` over an arbitrary GF(q), `2n × 2n`.
///
/// Over GF(2) this is [`symplectic_gram_matrix`], because `−1 = 1`. In odd
/// characteristic the sign is real and the form is genuinely alternating rather
/// than symmetric.
///
/// # Errors
///
/// `E-STAB-009` above [`MAX_QUBITS`](super::MAX_QUBITS); `E-GFQ-012` for a
/// shape FLINT cannot allocate.
pub fn symplectic_gram_matrix_over(
    field: &FiniteField,
    n: usize,
) -> Result<GfMatrix, StabilizerError> {
    check_qubits(n)?;
    let d = 2 * n;
    let minus_one = field.characteristic() - 1;
    let mut entries = vec![0u64; d * d];
    for i in 0..n {
        entries[i * d + (n + i)] = 1;
        entries[(n + i) * d + i] = minus_one;
    }
    Ok(GfMatrix::from_u64(field, d, d, &entries)?)
}

/// Is `M` in `Sp(2n, q)`, i.e. does it preserve the symplectic form?
///
/// The test performed is
///
/// ```text
///     Mᵀ Ω M  =  Ω,        Ω = [[0, I], [−I, 0]]
/// ```
///
/// the **column-vector** convention: `M` acts on column vectors `v ↦ Mv`, and
/// `(Mu)ᵀ Ω (Mv) = uᵀ (MᵀΩM) v`.
///
/// The row-vector convention gives `M Ω Mᵀ = Ω` instead, and the two are
/// equivalent: from `MΩMᵀ = Ω` one gets `Mᵀ = Ω⁻¹M⁻¹Ω`, hence
/// `MᵀΩM = Ω⁻¹M⁻¹ΩΩM = Ω⁻¹M⁻¹M Ω²… = Ω`. `stabilizer::tests` checks both
/// forms agree on every matrix of `Sp(4, 2)` rather than leaving it as an
/// argument on paper.
///
/// # Errors
///
/// `E-STAB-002` when `M` is not square or has odd size.
pub fn is_symplectic(m: &GfMatrix) -> Result<bool, StabilizerError> {
    let (r, c) = m.shape();
    if r != c || r % 2 != 0 {
        return Err(StabilizerError::ShapeMismatch {
            op: "Sp(2n, q) membership",
            expected: "a square matrix of even size 2n".to_string(),
            got: format!("{r}x{c}"),
        });
    }
    if r == 0 {
        // Sp(0, q) is the trivial group; the empty matrix is its identity.
        return Ok(true);
    }
    let omega = symplectic_gram_matrix_over(m.field(), r / 2)?;
    let lhs = m.transpose().mul(&omega)?.mul(m)?;
    Ok(lhs.equals(&omega))
}

/// A hyperbolic decomposition of a subspace under the symplectic form.
///
/// The subspace `V` splits as
///
/// ```text
///     V  =  ⟨u₁, v₁⟩ ⊥ ⟨u₂, v₂⟩ ⊥ … ⊥ ⟨u_m, v_m⟩ ⊥ rad(V)
/// ```
///
/// where `⟨u_i, v_i⟩ = 1`, every other pairing among the `u`'s and `v`'s is
/// `0`, and `rad(V) = V ∩ V^⊥` is the radical — the part of `V` on which the
/// restricted form is identically zero.
///
/// For a stabilizer code this is exactly the structure that produces logical
/// operators: run it on the centralizer `S^⊥` and the radical comes out as the
/// stabilizer `S` itself, while the `m = k` hyperbolic pairs are the logical
/// `X_i` / `Z_i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperbolicBasis {
    n: usize,
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    radical: Vec<Vec<u8>>,
}

impl HyperbolicBasis {
    /// Number of qubits, i.e. the vectors have length `2n`.
    pub fn qubits(&self) -> usize {
        self.n
    }

    /// The hyperbolic pairs `(u_i, v_i)` with `⟨u_i, v_i⟩ = 1`.
    pub fn pairs(&self) -> &[(Vec<u8>, Vec<u8>)] {
        &self.pairs
    }

    /// A basis of the radical `V ∩ V^⊥`.
    pub fn radical(&self) -> &[Vec<u8>] {
        &self.radical
    }

    /// `dim V = 2·|pairs| + |radical|`.
    pub fn dimension(&self) -> usize {
        2 * self.pairs.len() + self.radical.len()
    }

    /// Every basis vector: the pairs flattened, then the radical.
    pub fn vectors(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::with_capacity(self.dimension());
        for (u, v) in &self.pairs {
            out.push(u.clone());
            out.push(v.clone());
        }
        out.extend(self.radical.iter().cloned());
        out
    }
}

/// An independent spanning subset of `rows`, in reduced row echelon form.
pub(crate) fn independent_rows(
    rows: &[Vec<u8>],
    width: usize,
) -> Result<Vec<Vec<u8>>, StabilizerError> {
    if rows.is_empty() || width == 0 {
        return Ok(Vec::new());
    }
    let m = matrix_from_rows(rows, width)?;
    let rref = m.rref()?;
    let all = matrix_rows(&rref.matrix)?;
    Ok(all.into_iter().take(rref.rank).collect())
}

/// **Symplectic Gram–Schmidt**: a hyperbolic basis for the span of `vectors`.
///
/// The input need not be independent — it is row-reduced first — and need not
/// be isotropic. The output is a [`HyperbolicBasis`] whose pairs and radical
/// together span the same subspace.
///
/// # The algorithm, and why it terminates with the right radical
///
/// Take the first remaining basis vector `u`. If some remaining `w` has
/// `⟨u, w⟩ = 1`, call it `v`, remove both, and replace every other remaining
/// `w` by
///
/// ```text
///     w'  =  w  +  ⟨w, v⟩·u  +  ⟨w, u⟩·v
/// ```
///
/// which is orthogonal to both `u` and `v` (this is where the GF(2) symmetry
/// `⟨u, v⟩ = ⟨v, u⟩` is used). If no remaining `w` pairs non-trivially with
/// `u`, then `u` is orthogonal to everything left *and*, by the projection step,
/// to every pair already extracted — so `u` lies in the radical, and every
/// vector produced later is a combination of vectors `u` is orthogonal to.
///
/// The result is verified before it is returned: all the pairings are checked,
/// and the radical is checked to be orthogonal to the whole span. A failure
/// there is [`StabilizerError::InconsistentResult`] — withheld, not returned.
///
/// # Errors
///
/// `E-STAB-002` for a vector whose length is not `2n`; `E-STAB-009` above
/// [`MAX_QUBITS`](super::MAX_QUBITS); `E-STAB-013` if the result fails its own
/// check.
pub fn symplectic_gram_schmidt(
    vectors: &[Vec<u8>],
    n: usize,
) -> Result<HyperbolicBasis, StabilizerError> {
    check_qubits(n)?;
    let width = 2 * n;
    for v in vectors {
        if v.len() != width {
            return Err(StabilizerError::ShapeMismatch {
                op: "symplectic Gram–Schmidt",
                expected: format!("vectors of length 2n = {width}"),
                got: format!("a vector of length {}", v.len()),
            });
        }
    }

    let mut basis = independent_rows(vectors, width)?;
    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut radical: Vec<Vec<u8>> = Vec::new();

    while !basis.is_empty() {
        let u = basis.remove(0);
        let partner = basis.iter().position(|w| form_unchecked(&u, w) == 1);
        match partner {
            None => radical.push(u),
            Some(idx) => {
                let v = basis.remove(idx);
                for w in basis.iter_mut() {
                    let a = form_unchecked(w, &v);
                    let b = form_unchecked(w, &u);
                    if a == 1 {
                        *w = bit_add(w, &u);
                    }
                    if b == 1 {
                        *w = bit_add(w, &v);
                    }
                }
                pairs.push((u, v));
            }
        }
    }

    let out = HyperbolicBasis { n, pairs, radical };
    verify_hyperbolic(&out, vectors, width)?;
    Ok(out)
}

/// The checks `symplectic_gram_schmidt` runs on itself before returning.
fn verify_hyperbolic(
    b: &HyperbolicBasis,
    original: &[Vec<u8>],
    width: usize,
) -> Result<(), StabilizerError> {
    let all = b.vectors();
    let m = b.pairs.len();
    for i in 0..m {
        for j in 0..m {
            let want_uv = u8::from(i == j);
            if form_unchecked(&b.pairs[i].0, &b.pairs[j].1) != want_uv {
                return Err(StabilizerError::InconsistentResult {
                    reason: format!("⟨u_{i}, v_{j}⟩ is not {want_uv}"),
                });
            }
            if i != j
                && (form_unchecked(&b.pairs[i].0, &b.pairs[j].0) != 0
                    || form_unchecked(&b.pairs[i].1, &b.pairs[j].1) != 0)
            {
                return Err(StabilizerError::InconsistentResult {
                    reason: format!("hyperbolic pairs {i} and {j} are not orthogonal"),
                });
            }
        }
    }
    for (r, rv) in b.radical.iter().enumerate() {
        for w in &all {
            if form_unchecked(rv, w) != 0 {
                return Err(StabilizerError::InconsistentResult {
                    reason: format!("radical vector {r} is not orthogonal to the whole span"),
                });
            }
        }
    }
    // Same span as the input, checked by rank rather than by equality of sets.
    let span_rank = bit_rank(original, width)?;
    if span_rank != b.dimension() {
        return Err(StabilizerError::InconsistentResult {
            reason: format!(
                "the hyperbolic basis has dimension {} but the input spans dimension {span_rank}",
                b.dimension()
            ),
        });
    }
    let mut joint = all.clone();
    joint.extend(original.iter().cloned());
    if bit_rank(&joint, width)? != span_rank {
        return Err(StabilizerError::InconsistentResult {
            reason: "the hyperbolic basis does not span the same subspace as the input".to_string(),
        });
    }
    Ok(())
}

/// The **symplectic complement** `V^⊥ = { w : ⟨w, v⟩ = 0 for all v ∈ V }`,
/// returned as a basis.
///
/// `V` is the span of `vectors`. The returned basis has
/// `dim V^⊥ = 2n − dim V`, and `V ⊆ V^⊥` exactly when `V` is isotropic — which
/// is the case that matters, because a stabilizer group's symplectic span is
/// isotropic and its complement is the centralizer.
///
/// # Errors
///
/// `E-STAB-002` for a vector whose length is not `2n`; `E-STAB-009` above
/// [`MAX_QUBITS`](super::MAX_QUBITS); `E-GFQ-012` for a shape FLINT cannot
/// allocate.
pub fn symplectic_complement(
    vectors: &[Vec<u8>],
    n: usize,
) -> Result<Vec<Vec<u8>>, StabilizerError> {
    check_qubits(n)?;
    let width = 2 * n;
    for v in vectors {
        if v.len() != width {
            return Err(StabilizerError::ShapeMismatch {
                op: "symplectic complement",
                expected: format!("vectors of length 2n = {width}"),
                got: format!("a vector of length {}", v.len()),
            });
        }
    }
    if width == 0 {
        return Ok(Vec::new());
    }
    if vectors.is_empty() {
        // The complement of {0} is everything.
        return Ok((0..width)
            .map(|i| {
                let mut e = vec![0u8; width];
                e[i] = 1;
                e
            })
            .collect());
    }
    // ⟨a, w⟩ = (a Ω) · w, and a Ω just swaps a's x and z halves.
    let swapped: Vec<Vec<u8>> = vectors
        .iter()
        .map(|a| {
            let mut s = vec![0u8; width];
            s[..n].copy_from_slice(&a[n..]);
            s[n..].copy_from_slice(&a[..n]);
            s
        })
        .collect();
    let m = matrix_from_rows(&swapped, width)?;
    let ns = m.nullspace()?;
    matrix_cols(&ns)
}
