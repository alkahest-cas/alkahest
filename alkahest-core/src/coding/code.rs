//! Linear `[n, k]` codes over GF(q).
//!
//! A [`LinearCode`] is a `k`-dimensional subspace of `GF(q)^n`, held as **both**
//! a generator matrix (a `k × n` basis, rows) and a parity-check matrix (an
//! `(n-k) × n` basis of the dual), each derived from the other by
//! [`GfMatrix::nullspace`] and [`GfMatrix::rref`]. Holding both is deliberate:
//! every question this module answers is natural from exactly one of them, and
//! recomputing the other on demand would put an `O(n^3)` elimination inside
//! `dual()`.
//!
//! # Minimum distance
//!
//! [`LinearCode::minimum_distance`] enumerates all `q^k` codewords. There is no
//! cleverer algorithm here — no Brouwer–Zimmermann, no branch-and-bound — and
//! that is a deliberate choice: minimum distance is NP-hard in general, the
//! published speed-ups are intricate, and a subtly wrong `d` is worse than no
//! `d`. Enumeration is obviously correct and is capped
//! ([`MAX_ENUMERATED_CODEWORDS`]); past the cap it refuses (`E-CODE-004`)
//! rather than truncating the search, because the minimum weight of a *prefix*
//! of the codewords is an upper bound on `d` wearing `d`'s name.
//!
//! In practice the cap admits everything up to about `q^k = 4 · 10^6`, which
//! covers the binary Golay codes (`2^12 = 4096` words) comfortably and stops
//! well short of, say, a `[63, 36]` BCH code.

use rug::ops::Pow;
use rug::Integer;

use crate::ffield::{FieldElement, FiniteField, GfMatrix};

use super::error::{wrap_ff, CodingError};
use super::weight::WeightEnumerator;

/// Hard cap on `q^k` for exhaustive codeword enumeration.
///
/// Not a mathematical limit — a refusal point. `q^k` is the only quantity in
/// this module that grows exponentially in the input, and the cap is where an
/// interactive call stops being interactive.
pub const MAX_ENUMERATED_CODEWORDS: u128 = 1 << 22;

/// Cap on the number of field coordinates touched, so that a *long* code with
/// few codewords cannot slip past [`MAX_ENUMERATED_CODEWORDS`].
///
/// Both the enumeration itself (`q^k · n · deg` coordinate updates) and the
/// table of scalar multiples it walks (`q · k · n · deg` words) are measured
/// against this.
pub const MAX_ENUMERATION_CELLS: u128 = 1 << 26;

/// A linear `[n, k]` code over GF(q).
///
/// Cloning is cheap-ish: it clones two dense FLINT matrices.
#[derive(Clone)]
pub struct LinearCode {
    field: FiniteField,
    n: usize,
    /// `k × n`, rows a basis of the code. Full row rank by construction.
    generator: GfMatrix,
    /// `(n-k) × n`, rows a basis of the dual. Full row rank by construction.
    parity: GfMatrix,
}

impl LinearCode {
    /// The code spanned by the **rows** of `generator`.
    ///
    /// The rows need not be independent: the code is the row space, and the
    /// stored generator is the reduced row echelon basis of it, so `k` is the
    /// rank rather than the row count.
    ///
    /// # Errors
    ///
    /// `E-CODE-001` when the matrix has no columns; `E-CODE-003` when the GF(q)
    /// backend refuses the elimination.
    pub fn from_generator(generator: &GfMatrix) -> Result<Self, CodingError> {
        let n = generator.ncols();
        if n == 0 {
            return Err(CodingError::InvalidLength { n });
        }
        let g = canonical_row_basis(generator)?;
        let parity = dual_basis(&g, n)?;
        Ok(Self {
            field: generator.field().clone(),
            n,
            generator: g,
            parity,
        })
    }

    /// The code `{x : H · xᵀ = 0}`, the null space of the parity-check matrix
    /// `H`.
    ///
    /// The rows of `H` need not be independent; `k = n − rank(H)`.
    ///
    /// # Errors
    ///
    /// `E-CODE-001` when the matrix has no columns; `E-CODE-003` when the GF(q)
    /// backend refuses the elimination.
    pub fn from_parity_check(parity: &GfMatrix) -> Result<Self, CodingError> {
        let n = parity.ncols();
        if n == 0 {
            return Err(CodingError::InvalidLength { n });
        }
        let h = canonical_row_basis(parity)?;
        let generator = dual_basis(&h, n)?;
        Ok(Self {
            field: parity.field().clone(),
            n,
            generator,
            parity: h,
        })
    }

    /// The field the code is defined over.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// The length `n`.
    pub fn length(&self) -> usize {
        self.n
    }

    /// The dimension `k`.
    pub fn dimension(&self) -> usize {
        self.generator.nrows()
    }

    /// The redundancy `n − k`, i.e. the number of independent parity checks.
    pub fn redundancy(&self) -> usize {
        self.parity.nrows()
    }

    /// The number of codewords, `q^k`, exactly.
    pub fn size(&self) -> Integer {
        let q = Integer::from(self.field.characteristic()).pow(self.field.degree() as u32);
        q.pow(self.dimension() as u32)
    }

    /// The generator matrix: `k × n`, rows a basis, in reduced row echelon
    /// form.
    pub fn generator(&self) -> &GfMatrix {
        &self.generator
    }

    /// The parity-check matrix: `(n−k) × n`, rows a basis of the dual, in
    /// reduced row echelon form.
    pub fn parity_check(&self) -> &GfMatrix {
        &self.parity
    }

    /// The dual code `C⊥ = {y : y · xᵀ = 0 for all x ∈ C}`, an `[n, n−k]` code.
    ///
    /// Free: the two matrices simply swap roles. `C.dual().dual()` spans the
    /// same subspace as `C` (and is equal to it on the nose once both have been
    /// put in reduced row echelon form).
    pub fn dual(&self) -> Self {
        Self {
            field: self.field.clone(),
            n: self.n,
            generator: self.parity.clone(),
            parity: self.generator.clone(),
        }
    }

    /// Whether `C ⊆ C⊥`, i.e. `G · Gᵀ = 0`.
    ///
    /// # Errors
    ///
    /// `E-CODE-003` when the GF(q) backend refuses the product.
    pub fn is_self_orthogonal(&self) -> Result<bool, CodingError> {
        let gt = self.generator.transpose();
        let prod = self
            .generator
            .mul(&gt)
            .map_err(|e| wrap_ff("multiply G by its transpose", e))?;
        Ok(prod.is_zero())
    }

    /// Whether `C = C⊥`: self-orthogonal **and** `2k = n`.
    ///
    /// # Errors
    ///
    /// `E-CODE-003` when the GF(q) backend refuses the product.
    pub fn is_self_dual(&self) -> Result<bool, CodingError> {
        Ok(2 * self.dimension() == self.n && self.is_self_orthogonal()?)
    }

    /// Whether the `1 × n` row vector `word` is a codeword, i.e. `H · wordᵀ = 0`.
    ///
    /// # Errors
    ///
    /// `E-CODE-001` when `word` is not `1 × n`; `E-CODE-003` when the fields
    /// differ or the product is refused.
    pub fn contains(&self, word: &GfMatrix) -> Result<bool, CodingError> {
        if word.nrows() != 1 || word.ncols() != self.n {
            return Err(CodingError::InvalidLength { n: word.ncols() });
        }
        if self.parity.nrows() == 0 {
            return Ok(true);
        }
        let s = self
            .parity
            .mul(&word.transpose())
            .map_err(|e| wrap_ff("apply the parity check", e))?;
        Ok(s.is_zero())
    }

    /// The extended code: one overall parity symbol appended, so that every
    /// codeword's coordinates sum to zero.
    ///
    /// `[n, k] → [n+1, k]`. Over GF(2) this is what turns the `[7,4,3]` Hamming
    /// code into the `[8,4,4]` extended Hamming code and the `[23,12,7]` Golay
    /// code into the `[24,12,8]` extended Golay code.
    ///
    /// # Errors
    ///
    /// `E-CODE-003` when the GF(q) backend refuses a step.
    pub fn extend(&self) -> Result<Self, CodingError> {
        let k = self.dimension();
        let ones = {
            let mut m = GfMatrix::zeros(&self.field, self.n, 1)
                .map_err(|e| wrap_ff("allocate the all-ones column", e))?;
            let one = self.field.one();
            for i in 0..self.n {
                m.set_entry(i, 0, &one)
                    .map_err(|e| wrap_ff("fill the all-ones column", e))?;
            }
            m
        };
        let sums = self
            .generator
            .mul(&ones)
            .map_err(|e| wrap_ff("compute row sums", e))?
            .neg();

        let mut g = GfMatrix::zeros(&self.field, k, self.n + 1)
            .map_err(|e| wrap_ff("allocate the extended generator", e))?;
        for i in 0..k {
            for j in 0..self.n {
                let v = self
                    .generator
                    .entry(i, j)
                    .map_err(|e| wrap_ff("read the generator", e))?;
                g.set_entry(i, j, &v)
                    .map_err(|e| wrap_ff("write the extended generator", e))?;
            }
            let v = sums.entry(i, 0).map_err(|e| wrap_ff("read a row sum", e))?;
            g.set_entry(i, self.n, &v)
                .map_err(|e| wrap_ff("write the overall parity symbol", e))?;
        }
        LinearCode::from_generator(&g)
    }

    /// The weight distribution `A_0 … A_n`, by exhaustive enumeration of all
    /// `q^k` codewords.
    ///
    /// `A_0 = 1` and `Σ A_i = q^k` always; both are asserted by property tests.
    ///
    /// # Errors
    ///
    /// `E-CODE-004` when `q^k` — or the total coordinate work — is past the cap
    /// ([`MAX_ENUMERATED_CODEWORDS`], [`MAX_ENUMERATION_CELLS`]). `E-CODE-008`
    /// when `q` itself is beyond what a `u128` can hold. `E-CODE-003` when the
    /// GF(q) backend refuses a scalar multiple.
    pub fn weight_distribution(&self) -> Result<Vec<Integer>, CodingError> {
        self.weight_distribution_with_cap(MAX_ENUMERATED_CODEWORDS)
    }

    /// [`LinearCode::weight_distribution`] with an explicit cap on `q^k`.
    ///
    /// For callers who know they are asking for a long wait and want it anyway.
    /// The secondary cap [`MAX_ENUMERATION_CELLS`] still applies and is *not*
    /// raised by this: it guards memory, not patience.
    pub fn weight_distribution_with_cap(&self, cap: u128) -> Result<Vec<Integer>, CodingError> {
        let counts = self.enumerate_weights(cap)?;
        Ok(counts.into_iter().map(Integer::from).collect())
    }

    /// The weight enumerator polynomial `W_C(x, y) = Σ_i A_i x^(n-i) y^i`.
    ///
    /// # Errors
    ///
    /// As [`LinearCode::weight_distribution`].
    pub fn weight_enumerator(&self) -> Result<WeightEnumerator, CodingError> {
        let q = self.alphabet_size()?;
        let q = u64::try_from(q).map_err(|_| CodingError::InvalidAlphabet {
            q: q.to_string(),
            reason: "the weight enumerator's Krawtchouk transform needs q to fit a u64".to_string(),
        })?;
        WeightEnumerator::new(q, self.weight_distribution()?)
    }

    /// The minimum distance, by exhaustive enumeration.
    ///
    /// `None` for the zero code: it has no non-zero codeword, so it has no
    /// minimum distance. Conventions differ on whether to call that `n` or `∞`,
    /// and picking one silently would be a guess.
    ///
    /// # Errors
    ///
    /// As [`LinearCode::weight_distribution`].
    pub fn minimum_distance(&self) -> Result<Option<usize>, CodingError> {
        let counts = self.enumerate_weights(MAX_ENUMERATED_CODEWORDS)?;
        Ok(counts.iter().skip(1).position(|c| *c > 0).map(|i| i + 1))
    }

    /// `q = p^deg`, the size of the alphabet.
    fn alphabet_size(&self) -> Result<u128, CodingError> {
        self.field
            .order()
            .ok_or_else(|| CodingError::InvalidAlphabet {
                q: format!("{}^{}", self.field.characteristic(), self.field.degree()),
                reason: "the field order overflows a u128".to_string(),
            })
    }

    /// Count codewords by weight. Returns `A_0 … A_n` as machine integers —
    /// safe, because the cap keeps `q^k` far below `u64::MAX`.
    fn enumerate_weights(&self, cap: u128) -> Result<Vec<u64>, CodingError> {
        let k = self.dimension();
        let n = self.n;
        let deg = self.field.degree();
        let p = self.field.characteristic();
        let q = self.alphabet_size()?;

        let codewords = q
            .checked_pow(u32::try_from(k).unwrap_or(u32::MAX))
            .unwrap_or(u128::MAX);
        if codewords > cap {
            return Err(CodingError::EnumerationTooLarge {
                codewords: if codewords == u128::MAX {
                    format!("q^k with q = {q}, k = {k}")
                } else {
                    codewords.to_string()
                },
                n,
                limit: format!("at most {cap} codewords"),
            });
        }
        let cells = codewords
            .saturating_mul(n as u128)
            .saturating_mul(deg as u128);
        if cells > MAX_ENUMERATION_CELLS {
            return Err(CodingError::EnumerationTooLarge {
                codewords: codewords.to_string(),
                n,
                limit: format!(
                    "at most {MAX_ENUMERATION_CELLS} coordinate updates; this needs {cells}"
                ),
            });
        }
        let table_cells = q
            .saturating_mul(k as u128)
            .saturating_mul(n as u128)
            .saturating_mul(deg as u128);
        if table_cells > MAX_ENUMERATION_CELLS {
            return Err(CodingError::EnumerationTooLarge {
                codewords: codewords.to_string(),
                n,
                limit: format!(
                    "the table of scalar multiples would need {table_cells} words, \
                     past {MAX_ENUMERATION_CELLS}"
                ),
            });
        }

        let mut counts = vec![0u64; n + 1];
        if k == 0 {
            counts[0] = 1;
            return Ok(counts);
        }

        // `multiples[c * k + i]` is the coordinate vector of `elem_c · g_i`,
        // flattened as `n` blocks of `deg` coordinates over GF(p).
        let q_usize = q as usize;
        let elems = field_elements(&self.field)?;
        let mut multiples: Vec<Vec<u64>> = Vec::with_capacity(q_usize * k);
        for c in elems.iter() {
            let scaled = self
                .generator
                .scalar_mul(c)
                .map_err(|e| wrap_ff("scale the generator", e))?;
            let entries = scaled.to_elements();
            for i in 0..k {
                let mut row = Vec::with_capacity(n * deg);
                for j in 0..n {
                    row.extend_from_slice(&entries[i * n + j].coefficients_padded(deg));
                }
                multiples.push(row);
            }
        }

        let mut cur = vec![0u64; n * deg];
        enumerate(0, k, q_usize, p, n, deg, &multiples, &mut cur, &mut counts);
        Ok(counts)
    }
}

/// Depth-first walk over all `q^k` coefficient vectors, maintaining the partial
/// codeword in `cur`.
#[allow(clippy::too_many_arguments)]
fn enumerate(
    level: usize,
    k: usize,
    q: usize,
    p: u64,
    n: usize,
    deg: usize,
    multiples: &[Vec<u64>],
    cur: &mut [u64],
    counts: &mut [u64],
) {
    if level == k {
        let mut w = 0usize;
        for j in 0..n {
            if cur[j * deg..(j + 1) * deg].iter().any(|v| *v != 0) {
                w += 1;
            }
        }
        counts[w] += 1;
        return;
    }
    // c = 0 contributes nothing, so it costs one recursion and no arithmetic.
    enumerate(level + 1, k, q, p, n, deg, multiples, cur, counts);
    for c in 1..q {
        let add = &multiples[c * k + level];
        for (t, v) in cur.iter_mut().zip(add.iter()) {
            *t += *v;
            if *t >= p {
                *t -= p;
            }
        }
        enumerate(level + 1, k, q, p, n, deg, multiples, cur, counts);
        for (t, v) in cur.iter_mut().zip(add.iter()) {
            if *t < *v {
                *t += p;
            }
            *t -= *v;
        }
    }
}

/// Every element of GF(q), indexed so that `0` is zero and `1` is one.
///
/// For an extension the index is the mixed-radix encoding of the coordinate
/// vector, least significant first, which puts `1 = (1, 0, …, 0)` at index 1.
pub(crate) fn field_elements(field: &FiniteField) -> Result<Vec<FieldElement>, CodingError> {
    let p = field.characteristic();
    let deg = field.degree();
    let q = field.order().ok_or_else(|| CodingError::InvalidAlphabet {
        q: format!("{p}^{deg}"),
        reason: "the field order overflows a u128".to_string(),
    })?;
    let q = usize::try_from(q).map_err(|_| CodingError::InvalidAlphabet {
        q: q.to_string(),
        reason: "the field is far too large to enumerate element by element".to_string(),
    })?;
    if deg == 1 {
        return Ok((0..p).map(|x| field.scalar(x)).collect());
    }
    let mut out = Vec::with_capacity(q);
    for m in 0..q {
        let mut coeffs = Vec::with_capacity(deg);
        let mut rest = m as u128;
        for _ in 0..deg {
            coeffs.push((rest % p as u128) as u64);
            rest /= p as u128;
        }
        out.push(
            field
                .element(&coeffs)
                .map_err(|e| wrap_ff("build a field element", e))?,
        );
    }
    Ok(out)
}

/// The reduced row echelon basis of the row space, with the zero rows dropped.
fn canonical_row_basis(m: &GfMatrix) -> Result<GfMatrix, CodingError> {
    if m.nrows() == 0 {
        return Ok(m.clone());
    }
    let rref = m.rref().map_err(|e| wrap_ff("rref", e))?;
    if rref.rank == m.nrows() {
        return Ok(rref.matrix);
    }
    let mut out = GfMatrix::zeros(m.field(), rref.rank, m.ncols())
        .map_err(|e| wrap_ff("allocate the row basis", e))?;
    for i in 0..rref.rank {
        for j in 0..m.ncols() {
            let v = rref
                .matrix
                .entry(i, j)
                .map_err(|e| wrap_ff("read the rref", e))?;
            out.set_entry(i, j, &v)
                .map_err(|e| wrap_ff("write the row basis", e))?;
        }
    }
    Ok(out)
}

/// A full-row-rank basis of `{y : m · yᵀ = 0}`, as an `(n − rank) × n` matrix in
/// reduced row echelon form.
///
/// FLINT's nullspace basis is *a* basis, not a canonical one, so it is reduced
/// before being stored. Without that, `generator()` would be canonical when the
/// code was built from a generator and arbitrary when it was built from a
/// parity check — and `LinearCode::from_generator(c.generator())` would not
/// round-trip to an equal matrix, only to an equal subspace.
fn dual_basis(m: &GfMatrix, n: usize) -> Result<GfMatrix, CodingError> {
    if m.nrows() == 0 {
        // The empty system: every vector is a solution.
        return GfMatrix::identity(m.field(), n).map_err(|e| wrap_ff("identity", e));
    }
    let ns = m.nullspace().map_err(|e| wrap_ff("nullspace", e))?;
    canonical_row_basis(&ns.transpose())
}

// ---------------------------------------------------------------------------
// Named families
// ---------------------------------------------------------------------------

impl LinearCode {
    /// The Hamming code over GF(q) with `r` parity checks:
    /// `[n, n−r, 3]` with `n = (q^r − 1)/(q − 1)`.
    ///
    /// The parity-check matrix has one column per point of the projective space
    /// `PG(r−1, q)` — every non-zero vector whose first non-zero coordinate is
    /// `1`. No two columns are proportional, so no codeword has weight 1 or 2,
    /// and some triple is dependent, so `d = 3` exactly. Over GF(2) with
    /// `r = 3` this is the `[7, 4, 3]` code.
    ///
    /// # Errors
    ///
    /// `E-CODE-002` when `r < 2` (`r = 1` gives the zero-length or the whole
    /// space, neither of which has distance 3); `E-CODE-004` when `q^r` is too
    /// large to enumerate the columns; `E-CODE-003` from the backend.
    pub fn hamming(field: &FiniteField, r: u32) -> Result<Self, CodingError> {
        if r < 2 {
            return Err(CodingError::InvalidDistance {
                d: 3,
                n: r as usize,
            });
        }
        let elems = field_elements(field)?;
        let q = elems.len();
        let total = (q as u128)
            .checked_pow(r)
            .ok_or_else(|| CodingError::EnumerationTooLarge {
                codewords: format!("q^r with q = {q}, r = {r}"),
                n: 0,
                limit: format!("at most {MAX_ENUMERATED_CODEWORDS}"),
            })?;
        if total > MAX_ENUMERATED_CODEWORDS {
            return Err(CodingError::EnumerationTooLarge {
                codewords: total.to_string(),
                n: 0,
                limit: format!(
                    "at most {MAX_ENUMERATED_CODEWORDS} vectors to sieve for projective points"
                ),
            });
        }

        // One column per projective point, in lexicographic order of the
        // mixed-radix encoding with coordinate 0 most significant.
        let mut columns: Vec<Vec<usize>> = Vec::new();
        for m in 1..total {
            let mut digits = vec![0usize; r as usize];
            let mut rest = m;
            for t in (0..r as usize).rev() {
                digits[t] = (rest % q as u128) as usize;
                rest /= q as u128;
            }
            let lead = digits.iter().position(|d| *d != 0).expect("m != 0");
            if digits[lead] == 1 {
                columns.push(digits);
            }
        }

        let n = columns.len();
        let mut h = GfMatrix::zeros(field, r as usize, n)
            .map_err(|e| wrap_ff("allocate the Hamming parity check", e))?;
        for (j, col) in columns.iter().enumerate() {
            for (i, d) in col.iter().enumerate() {
                h.set_entry(i, j, &elems[*d])
                    .map_err(|e| wrap_ff("fill the Hamming parity check", e))?;
            }
        }
        LinearCode::from_parity_check(&h)
    }

    /// The binary Golay code `[23, 12, 7]`.
    ///
    /// Built as the cyclic code generated by
    /// `g(x) = 1 + x² + x⁴ + x⁵ + x⁶ + x¹⁰ + x¹¹`, one of the two degree-11
    /// factors of `x²³ − 1` over GF(2). (They are the quadratic-residue and
    /// non-residue factors; the two codes they generate are equivalent, so the
    /// choice is arbitrary and the weight distribution is the same either way.)
    ///
    /// Its `[24, 12, 8]` extension — the self-dual extended Golay code — is
    /// `LinearCode::golay_binary()?.extend()?`.
    ///
    /// # Errors
    ///
    /// `E-CODE-003` from the GF(2) backend.
    pub fn golay_binary() -> Result<Self, CodingError> {
        let field = FiniteField::prime(2).map_err(|e| wrap_ff("build GF(2)", e))?;
        // 1 + x^2 + x^4 + x^5 + x^6 + x^10 + x^11, ascending.
        const G: [u64; 12] = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1];
        let n = 23;
        let k = 12;
        let mut gen = GfMatrix::zeros(&field, k, n)
            .map_err(|e| wrap_ff("allocate the Golay generator", e))?;
        let one = field.one();
        for i in 0..k {
            for (t, c) in G.iter().enumerate() {
                if *c == 1 {
                    gen.set_entry(i, i + t, &one)
                        .map_err(|e| wrap_ff("fill the Golay generator", e))?;
                }
            }
        }
        LinearCode::from_generator(&gen)
    }

    /// The repetition code `[n, 1, n]` over GF(q): the span of the all-ones
    /// word.
    ///
    /// # Errors
    ///
    /// `E-CODE-001` for `n = 0`; `E-CODE-003` from the backend.
    pub fn repetition(field: &FiniteField, n: usize) -> Result<Self, CodingError> {
        if n == 0 {
            return Err(CodingError::InvalidLength { n });
        }
        let mut g =
            GfMatrix::zeros(field, 1, n).map_err(|e| wrap_ff("allocate the repetition code", e))?;
        let one = field.one();
        for j in 0..n {
            g.set_entry(0, j, &one)
                .map_err(|e| wrap_ff("fill the repetition code", e))?;
        }
        LinearCode::from_generator(&g)
    }
}

impl std::fmt::Debug for LinearCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LinearCode[n={}, k={}] over {:?}",
            self.n,
            self.dimension(),
            self.field
        )
    }
}
