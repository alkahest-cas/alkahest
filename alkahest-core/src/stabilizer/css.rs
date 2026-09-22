//! [`CssCode`] — Calderbank–Shor–Steane codes from a pair of GF(2) check
//! matrices.
//!
//! A CSS code takes two classical parity-check matrices `H_X` (`r_X × n`) and
//! `H_Z` (`r_Z × n`) over GF(2) and builds the stabilizer group whose
//! generators are
//!
//! ```text
//!     X-type:  row i of H_X, as the Pauli (x | z) = (H_X[i] | 0)
//!     Z-type:  row j of H_Z, as the Pauli (x | z) = (0 | H_Z[j])
//! ```
//!
//! The symplectic product of `(a | 0)` and `(0 | b)` is `a·b`, so those
//! generators commute **exactly when** `H_X · H_Zᵀ = 0`. That is the CSS
//! condition, and [`CssCode::new`] checks it: a violated entry `(i, j)` comes
//! back as [`StabilizerError::CssConditionViolated`] naming the two rows,
//! never as a code built from generators that do not commute.
//!
//! # Parameters
//!
//! `k = n − rank(H_X) − rank(H_Z)`. The `X`-type logical operators are a basis
//! of `ker(H_Z) / rowspace(H_X)` and the `Z`-type ones of
//! `ker(H_X) / rowspace(H_Z)`; the CSS condition is exactly the statement
//! `rowspace(H_X) ⊆ ker(H_Z)`, which is what makes those quotients meaningful.
//! The two bases are then paired so that `⟨X̄_i, Z̄_j⟩ = δ_ij`, by inverting the
//! `k × k` pairing matrix (which is invertible precisely because the form is
//! non-degenerate on the quotient).
//!
//! # Distance
//!
//! For a CSS code the minimum distance is `min(d_X, d_Z)`, where `d_X` is the
//! least weight in `ker(H_Z) \ rowspace(H_X)` and `d_Z` the least weight in
//! `ker(H_X) \ rowspace(H_Z)`. The argument is one line and worth stating,
//! because "the lightest logical operator might be a mixed `Y`-type one" is a
//! reasonable worry: if `L = X^a Z^b` lies in `N(S) \ S`, then `a ∉ rowspace(H_X)`
//! or `b ∉ rowspace(H_Z)`; in the first case `X^a` alone is already in
//! `N(S) \ S`, and `wt(X^a) = wt(a) ≤ wt(L)`. So a mixed operator is never
//! strictly lighter than the best pure one.
//!
//! This is a genuine saving — the two searches have dimension `n − r_Z` and
//! `n − r_X` rather than `n + k` — and `stabilizer::tests` cross-checks it
//! against the general [`StabilizerCode`] path on Steane and Shor rather than
//! resting on the argument alone.

use super::code::{min_weight_outside, Distance, StabilizerCode, MAX_DISTANCE_SEARCH_DIM};
use super::pauli::PauliOperator;
use super::symplectic::independent_rows;
use super::{bit_rank, matrix_cols, matrix_from_rows, matrix_rows, require_gf2, StabilizerError};
use crate::ffield::GfMatrix;
use std::fmt;

/// A CSS code built from `(H_X, H_Z)` over GF(2).
#[derive(Clone, Debug)]
pub struct CssCode {
    n: usize,
    hx: GfMatrix,
    hz: GfMatrix,
    rx: usize,
    rz: usize,
    k: usize,
    /// `X`-type logical operators, as `n`-bit vectors (the `z` half is zero).
    logical_x: Vec<Vec<u8>>,
    /// `Z`-type logical operators, as `n`-bit vectors (the `x` half is zero).
    logical_z: Vec<Vec<u8>>,
}

/// Extend `sub` to a basis of the span of `sub ∪ candidates`, returning only
/// the candidates that were needed — a basis of the quotient.
fn quotient_basis(
    sub: &[Vec<u8>],
    candidates: &[Vec<u8>],
    width: usize,
) -> Result<Vec<Vec<u8>>, StabilizerError> {
    let mut rows: Vec<Vec<u8>> = sub.to_vec();
    let mut r = bit_rank(&rows, width)?;
    let mut picked = Vec::new();
    for c in candidates {
        rows.push(c.clone());
        let r2 = bit_rank(&rows, width)?;
        if r2 > r {
            r = r2;
            picked.push(c.clone());
        } else {
            rows.pop();
        }
    }
    Ok(picked)
}

/// A basis of `ker(H)` as `n`-bit row vectors.
fn kernel_basis(h: &GfMatrix) -> Result<Vec<Vec<u8>>, StabilizerError> {
    let (rows, n) = h.shape();
    if rows == 0 {
        return Ok((0..n)
            .map(|i| {
                let mut e = vec![0u8; n];
                e[i] = 1;
                e
            })
            .collect());
    }
    matrix_cols(&h.nullspace()?)
}

impl CssCode {
    /// Build the CSS code for the pair `(H_X, H_Z)`.
    ///
    /// # Errors
    ///
    /// `E-STAB-001` when either matrix is not over GF(2); `E-STAB-002` when
    /// they have different column counts; `E-STAB-005` when
    /// `H_X · H_Zᵀ ≠ 0`; `E-STAB-009` when `n` exceeds
    /// [`MAX_QUBITS`](super::MAX_QUBITS); `E-STAB-013` when the logical
    /// pairing fails its own check; `E-GFQ-*` from the GF(2) linear algebra.
    pub fn new(hx: &GfMatrix, hz: &GfMatrix) -> Result<Self, StabilizerError> {
        require_gf2(hx)?;
        require_gf2(hz)?;
        let (_, n) = hx.shape();
        let (_, nz) = hz.shape();
        if n != nz {
            return Err(StabilizerError::ShapeMismatch {
                op: "build a CSS code",
                expected: format!("H_Z with {n} columns, matching H_X"),
                got: format!("{nz} columns"),
            });
        }
        super::check_qubits(n)?;

        // The CSS condition, checked rather than assumed.
        let product = hx.mul(&hz.transpose())?;
        if !product.is_zero() {
            let pr = matrix_rows(&product)?;
            for (i, row) in pr.iter().enumerate() {
                if let Some(j) = row.iter().position(|&b| b == 1) {
                    return Err(StabilizerError::CssConditionViolated { row: i, col: j });
                }
            }
            return Err(StabilizerError::InconsistentResult {
                reason: "H_X · H_Zᵀ is non-zero but no non-zero entry was found".to_string(),
            });
        }

        let rx = hx.rank();
        let rz = hz.rank();
        let k = n
            .checked_sub(rx + rz)
            .ok_or_else(|| StabilizerError::InconsistentResult {
                reason: format!(
                    "rank(H_X) + rank(H_Z) = {} exceeds n = {n}, which the CSS condition forbids",
                    rx + rz
                ),
            })?;

        let hx_rows = matrix_rows(hx)?;
        let hz_rows = matrix_rows(hz)?;
        let logical_x = quotient_basis(&hx_rows, &kernel_basis(hz)?, n)?;
        let logical_z = quotient_basis(&hz_rows, &kernel_basis(hx)?, n)?;
        if logical_x.len() != k || logical_z.len() != k {
            return Err(StabilizerError::InconsistentResult {
                reason: format!(
                    "expected {k} logical operators of each type, got {} X-type and {} Z-type",
                    logical_x.len(),
                    logical_z.len()
                ),
            });
        }

        let logical_z = pair_up(&logical_x, &logical_z, n, k)?;

        let code = Self {
            n,
            hx: hx.clone(),
            hz: hz.clone(),
            rx,
            rz,
            k,
            logical_x,
            logical_z,
        };
        code.verify()?;
        Ok(code)
    }

    /// Block length.
    pub fn n(&self) -> usize {
        self.n
    }

    /// `k = n − rank(H_X) − rank(H_Z)`.
    pub fn k(&self) -> usize {
        self.k
    }

    /// `rank(H_X)`.
    pub fn x_rank(&self) -> usize {
        self.rx
    }

    /// `rank(H_Z)`.
    pub fn z_rank(&self) -> usize {
        self.rz
    }

    /// The `X`-type check matrix, as supplied.
    pub fn hx(&self) -> &GfMatrix {
        &self.hx
    }

    /// The `Z`-type check matrix, as supplied.
    pub fn hz(&self) -> &GfMatrix {
        &self.hz
    }

    /// The `X`-type logical operators, as `n`-bit supports.
    pub fn logical_x_bits(&self) -> &[Vec<u8>] {
        &self.logical_x
    }

    /// The `Z`-type logical operators, as `n`-bit supports.
    pub fn logical_z_bits(&self) -> &[Vec<u8>] {
        &self.logical_z
    }

    /// The `X`-type logical operators as Hermitian Pauli operators.
    ///
    /// # Errors
    ///
    /// `E-STAB-009` above [`MAX_QUBITS`](super::MAX_QUBITS).
    pub fn logical_x(&self) -> Result<Vec<PauliOperator>, StabilizerError> {
        let zero = vec![0u8; self.n];
        self.logical_x
            .iter()
            .map(|v| PauliOperator::hermitian(v, &zero, false))
            .collect()
    }

    /// The `Z`-type logical operators as Hermitian Pauli operators.
    ///
    /// # Errors
    ///
    /// `E-STAB-009` above [`MAX_QUBITS`](super::MAX_QUBITS).
    pub fn logical_z(&self) -> Result<Vec<PauliOperator>, StabilizerError> {
        let zero = vec![0u8; self.n];
        self.logical_z
            .iter()
            .map(|v| PauliOperator::hermitian(&zero, v, false))
            .collect()
    }

    /// The generators of the corresponding stabilizer group, `X`-type first.
    ///
    /// # Errors
    ///
    /// `E-STAB-009` above [`MAX_QUBITS`](super::MAX_QUBITS).
    pub fn stabilizer_generators(&self) -> Result<Vec<PauliOperator>, StabilizerError> {
        let zero = vec![0u8; self.n];
        let mut gens = Vec::new();
        for r in matrix_rows(&self.hx)? {
            gens.push(PauliOperator::hermitian(&r, &zero, false)?);
        }
        for r in matrix_rows(&self.hz)? {
            gens.push(PauliOperator::hermitian(&zero, &r, false)?);
        }
        Ok(gens)
    }

    /// The same code seen through the general [`StabilizerCode`] machinery.
    ///
    /// Useful as a cross-check: the `k` computed from `n − rank(S)` on the
    /// symplectic side must match the `k` computed here from
    /// `n − rank(H_X) − rank(H_Z)`, and a mismatch is
    /// [`StabilizerError::InconsistentResult`].
    ///
    /// # Errors
    ///
    /// As [`StabilizerCode::from_generators`], plus `E-STAB-013` on a `k`
    /// mismatch.
    pub fn to_stabilizer_code(&self) -> Result<StabilizerCode, StabilizerError> {
        let code = StabilizerCode::from_generators_on(self.n, self.stabilizer_generators()?)?;
        if code.k() != self.k {
            return Err(StabilizerError::InconsistentResult {
                reason: format!(
                    "the CSS parameters give k = {}, but n − rank(S) = {}",
                    self.k,
                    code.k()
                ),
            });
        }
        Ok(code)
    }

    /// The **exact** minimum distance, as `min(d_X, d_Z)` — see the
    /// [module docs](self) for why that is the true distance and not just a
    /// bound.
    ///
    /// # Errors
    ///
    /// `E-STAB-007` when `k = 0`; `E-STAB-008` when either search exceeds
    /// [`MAX_DISTANCE_SEARCH_DIM`].
    pub fn minimum_distance(&self) -> Result<Distance, StabilizerError> {
        self.minimum_distance_with_cap(MAX_DISTANCE_SEARCH_DIM)
    }

    /// [`CssCode::minimum_distance`] with an explicit cap on each search
    /// dimension (`n − r_Z` for the `X` side, `n − r_X` for the `Z` side).
    ///
    /// # Errors
    ///
    /// As [`CssCode::minimum_distance`].
    pub fn minimum_distance_with_cap(&self, cap: usize) -> Result<Distance, StabilizerError> {
        if self.k == 0 {
            return Err(StabilizerError::NoLogicalQubits { n: self.n });
        }
        let zero = vec![0u8; self.n];
        let embed_x = |v: &Vec<u8>| {
            let mut w = v.clone();
            w.extend_from_slice(&zero);
            w
        };
        let embed_z = |v: &Vec<u8>| {
            let mut w = zero.clone();
            w.extend_from_slice(v);
            w
        };

        let hx_basis = independent_rows(&matrix_rows(&self.hx)?, self.n)?;
        let hz_basis = independent_rows(&matrix_rows(&self.hz)?, self.n)?;

        let lx: Vec<Vec<u8>> = self.logical_x.iter().map(&embed_x).collect();
        let sx: Vec<Vec<u8>> = hx_basis.iter().map(&embed_x).collect();
        let dx = min_weight_outside(&lx, &sx, self.n, cap)?;

        let lz: Vec<Vec<u8>> = self.logical_z.iter().map(&embed_z).collect();
        let sz: Vec<Vec<u8>> = hz_basis.iter().map(&embed_z).collect();
        let dz = min_weight_outside(&lz, &sz, self.n, cap)?;

        Ok(Distance::Exact(dx.min(dz)))
    }

    /// An **upper bound** on the distance: the least weight among the logical
    /// generators themselves, with no stabilizer multiplication.
    ///
    /// # Errors
    ///
    /// `E-STAB-007` when `k = 0`.
    pub fn distance_upper_bound(&self) -> Result<Distance, StabilizerError> {
        if self.k == 0 {
            return Err(StabilizerError::NoLogicalQubits { n: self.n });
        }
        let best = self
            .logical_x
            .iter()
            .chain(self.logical_z.iter())
            .map(|v| v.iter().filter(|&&b| b == 1).count())
            .min()
            .unwrap_or(0);
        Ok(Distance::UpperBound(best))
    }

    /// The checks that run before a CSS code is handed back.
    fn verify(&self) -> Result<(), StabilizerError> {
        // X logicals must be in ker(H_Z), Z logicals in ker(H_X).
        for (i, v) in self.logical_x.iter().enumerate() {
            let prod = self
                .hz
                .mul(&matrix_from_rows(std::slice::from_ref(v), self.n)?.transpose())?;
            if !prod.is_zero() {
                return Err(StabilizerError::InconsistentResult {
                    reason: format!("X-type logical {i} does not commute with every Z stabilizer"),
                });
            }
        }
        for (i, v) in self.logical_z.iter().enumerate() {
            let prod = self
                .hx
                .mul(&matrix_from_rows(std::slice::from_ref(v), self.n)?.transpose())?;
            if !prod.is_zero() {
                return Err(StabilizerError::InconsistentResult {
                    reason: format!("Z-type logical {i} does not commute with every X stabilizer"),
                });
            }
        }
        // ⟨X̄_i, Z̄_j⟩ = δ_ij.
        for i in 0..self.k {
            for j in 0..self.k {
                let dot: u8 = self.logical_x[i]
                    .iter()
                    .zip(&self.logical_z[j])
                    .fold(0u8, |acc, (&a, &b)| acc ^ (a & b));
                if dot != u8::from(i == j) {
                    return Err(StabilizerError::InconsistentResult {
                        reason: format!("⟨X̄_{i}, Z̄_{j}⟩ is not {}", u8::from(i == j)),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Replace the `Z` logical basis by `(M⁻¹)ᵀ Z` so that `⟨X̄_i, Z̄_j⟩ = δ_ij`.
fn pair_up(
    lx: &[Vec<u8>],
    lz: &[Vec<u8>],
    n: usize,
    k: usize,
) -> Result<Vec<Vec<u8>>, StabilizerError> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let mut m = vec![vec![0u8; k]; k];
    for i in 0..k {
        for j in 0..k {
            m[i][j] = lx[i]
                .iter()
                .zip(&lz[j])
                .fold(0u8, |acc, (&a, &b)| acc ^ (a & b));
        }
    }
    let gm = matrix_from_rows(&m, k)?;
    let inv = gm.inverse().map_err(|e| match e {
        crate::ffield::FiniteFieldError::Singular { .. } => StabilizerError::InconsistentResult {
            reason: "the pairing matrix ⟨X̄_i, Z̄_j⟩ is singular; the symplectic form is \
                     degenerate on the logical quotient, which cannot happen for a valid \
                     CSS pair"
                .to_string(),
        },
        other => StabilizerError::FiniteField(other),
    })?;
    let a = matrix_rows(&inv.transpose())?;
    let mut out = Vec::with_capacity(k);
    for row in a.iter().take(k) {
        let mut v = vec![0u8; n];
        for (j, &c) in row.iter().enumerate() {
            if c == 1 {
                for t in 0..n {
                    v[t] ^= lz[j][t];
                }
            }
        }
        out.push(v);
    }
    Ok(out)
}

impl fmt::Display for CssCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CSS[[{}, {}]]", self.n, self.k)
    }
}
