//! [`StabilizerCode`] — an `[[n, k, d]]` code from an abelian Pauli subgroup.
//!
//! # What is computed, and how
//!
//! * `n` is the qubit count and `k = n − rank(S)`. The rank is the GF(2) rank
//!   of the `(x | z)` check matrix, so a dependent generating list gives the
//!   same `k` as an independent one.
//! * The **logical operators** come from symplectic Gram–Schmidt on the
//!   centralizer. `S` is isotropic, so `S ⊆ S^⊥`, and
//!   `dim S^⊥ = 2n − (n − k) = n + k`. The radical of the form restricted to
//!   `S^⊥` is `S^⊥ ∩ S^⊥⊥ = S^⊥ ∩ S = S` itself, of dimension `n − k`, which
//!   leaves `((n + k) − (n − k))/2 = k` hyperbolic pairs. Those pairs *are* the
//!   `k` logical `X_i`/`Z_i`: each commutes with every stabilizer generator
//!   (being in `S^⊥`), `⟨X_i, Z_i⟩ = 1`, and every other pairing vanishes.
//!   [`StabilizerCode::new`] re-checks all of that on the result rather than
//!   trusting the derivation.
//! * The **syndrome** of an error `E` is the vector of symplectic products
//!   `⟨g_i, E⟩` over the generators — one bit per generator, `1` where `E`
//!   anticommutes.
//! * The **distance** is the least weight in `N(S) \ S`, by exhaustive search
//!   over the `2^(n+k)` elements of the centralizer. That is the only algorithm
//!   here; see [`MAX_DISTANCE_SEARCH_DIM`].
//!
//! # Phases of the logical operators
//!
//! The logicals are returned as **Hermitian** Paulis with a `+` sign — the
//! phase is chosen by [`PauliOperator::hermitian`]. Any of the `4^k` sign
//! choices is an equally valid set of logicals; this one is fixed so that
//! repeated runs agree.

use super::pauli::{PauliOperator, StabilizerGroup};
use super::symplectic::{symplectic_complement, symplectic_gram_schmidt};
use super::StabilizerError;
use std::fmt;

/// The largest `n + k` for which an exhaustive minimum-distance search runs.
///
/// The search visits `2^(n+k)` centralizer elements. At the cap that is about
/// four million, which takes well under a second; the next few doublings do
/// not. Minimum distance of a stabilizer code is `NP`-hard and no
/// polynomial-time shortcut is implemented here, so above the cap the answer is
/// [`StabilizerError::DistanceSearchTooLarge`] and the only thing on offer is
/// [`StabilizerCode::distance_upper_bound`].
pub const MAX_DISTANCE_SEARCH_DIM: usize = 22;

/// A minimum distance, or a bound on one — kept apart at the type level.
///
/// The distinction is the point. A caller that stores a
/// [`Distance::UpperBound`] in a field typed "distance" has thrown away the
/// only fact that made it safe to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Distance {
    /// Verified by exhaustive search over the whole centralizer: no element of
    /// `N(S) \ S` has smaller weight.
    Exact(usize),
    /// A witness of this weight was exhibited in `N(S) \ S`, so `d ≤` this.
    /// **Nothing** is claimed below it.
    UpperBound(usize),
}

impl Distance {
    /// The numeric value, whatever its status. Use with care: the whole reason
    /// this enum exists is that the two are not interchangeable.
    pub fn value(&self) -> usize {
        match self {
            Distance::Exact(d) | Distance::UpperBound(d) => *d,
        }
    }

    /// Is this an exhaustively verified distance?
    pub fn is_exact(&self) -> bool {
        matches!(self, Distance::Exact(_))
    }
}

impl fmt::Display for Distance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Distance::Exact(d) => write!(f, "{d}"),
            Distance::UpperBound(d) => write!(f, "≤ {d}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Packed bit vectors, for the distance search only
// ---------------------------------------------------------------------------

/// A `(x | z)` vector packed into 64-bit words, so that the inner loop of the
/// distance search is `xor` and `popcount` rather than a byte at a time.
#[derive(Clone)]
struct Packed {
    xw: Vec<u64>,
    zw: Vec<u64>,
}

impl Packed {
    fn zeros(words: usize) -> Self {
        Self {
            xw: vec![0; words],
            zw: vec![0; words],
        }
    }

    fn pack(v: &[u8], n: usize) -> Self {
        let words = n.div_ceil(64).max(1);
        let mut p = Self::zeros(words);
        for i in 0..n {
            if v[i] & 1 == 1 {
                p.xw[i / 64] |= 1u64 << (i % 64);
            }
            if v[n + i] & 1 == 1 {
                p.zw[i / 64] |= 1u64 << (i % 64);
            }
        }
        p
    }

    fn xor_assign(&mut self, other: &Self) {
        for i in 0..self.xw.len() {
            self.xw[i] ^= other.xw[i];
            self.zw[i] ^= other.zw[i];
        }
    }

    fn weight(&self) -> usize {
        self.xw
            .iter()
            .zip(&self.zw)
            .map(|(a, b)| (a | b).count_ones() as usize)
            .sum()
    }
}

/// The least weight of `L + s` over non-trivial logical classes `L` and all
/// `s` in the span of `stabs`.
///
/// `logicals` and `stabs` are `(x | z)` vectors of length `2n`; the search
/// enumerates the `2^(|logicals| + |stabs|)` combinations by Gray code and
/// skips those whose logical part is zero (those lie in the stabilizer and are
/// not errors at all).
///
/// # Errors
///
/// `E-STAB-008` when `|logicals| + |stabs|` exceeds `cap`.
pub(crate) fn min_weight_outside(
    logicals: &[Vec<u8>],
    stabs: &[Vec<u8>],
    n: usize,
    cap: usize,
) -> Result<usize, StabilizerError> {
    let dim = logicals.len() + stabs.len();
    if dim > cap {
        return Err(StabilizerError::DistanceSearchTooLarge { dim, cap });
    }
    if logicals.is_empty() {
        return Err(StabilizerError::NoLogicalQubits { n });
    }
    let basis: Vec<Packed> = logicals
        .iter()
        .chain(stabs.iter())
        .map(|v| Packed::pack(v, n))
        .collect();
    let low_mask: u64 = if logicals.len() >= 64 {
        u64::MAX
    } else {
        (1u64 << logicals.len()) - 1
    };

    let words = n.div_ceil(64).max(1);
    let mut cur = Packed::zeros(words);
    let mut gray: u64 = 0;
    let mut best = usize::MAX;
    let total: u64 = 1u64 << dim;
    for i in 1..total {
        let b = i.trailing_zeros() as usize;
        cur.xor_assign(&basis[b]);
        gray ^= 1u64 << b;
        if gray & low_mask != 0 {
            let w = cur.weight();
            if w < best {
                best = w;
            }
        }
    }
    if best == usize::MAX || best == 0 {
        // Unreachable on a well-formed code: a non-trivial logical class
        // cannot contain the identity. Withheld rather than reported as 0.
        return Err(StabilizerError::InconsistentResult {
            reason: "the distance search found no non-trivial logical operator, or found one \
                     of weight 0"
                .to_string(),
        });
    }
    Ok(best)
}

// ---------------------------------------------------------------------------
// StabilizerCode
// ---------------------------------------------------------------------------

/// An `[[n, k]]` stabilizer code: a stabilizer group, its logical operators and
/// its syndrome map.
#[derive(Clone, Debug)]
pub struct StabilizerCode {
    group: StabilizerGroup,
    n: usize,
    k: usize,
    logical_x: Vec<PauliOperator>,
    logical_z: Vec<PauliOperator>,
    /// An independent basis of `S`'s symplectic span, `n − k` vectors.
    stabilizer_basis: Vec<Vec<u8>>,
}

impl StabilizerCode {
    /// Build the code stabilized by `group`.
    ///
    /// # Errors
    ///
    /// `E-STAB-013` if the computed logical operators fail the commutation
    /// pattern they are supposed to satisfy — withheld rather than returned;
    /// `E-STAB-002` / `E-STAB-009` from the symplectic layer; `E-GFQ-*` from
    /// the GF(2) linear algebra.
    pub fn new(group: StabilizerGroup) -> Result<Self, StabilizerError> {
        let n = group.qubits();
        let k = n - group.rank();
        let stab_rows: Vec<Vec<u8>> = group.generators().iter().map(|g| g.symplectic()).collect();
        let centralizer = symplectic_complement(&stab_rows, n)?;
        let hb = symplectic_gram_schmidt(&centralizer, n)?;

        if hb.pairs().len() != k {
            return Err(StabilizerError::InconsistentResult {
                reason: format!(
                    "symplectic Gram–Schmidt on the centralizer produced {} hyperbolic pairs, \
                     but n − rank(S) = {k}",
                    hb.pairs().len()
                ),
            });
        }
        if hb.radical().len() != n - k {
            return Err(StabilizerError::InconsistentResult {
                reason: format!(
                    "the radical of the centralizer has dimension {}, but rank(S) = {}",
                    hb.radical().len(),
                    n - k
                ),
            });
        }

        let mut logical_x = Vec::with_capacity(k);
        let mut logical_z = Vec::with_capacity(k);
        for (u, v) in hb.pairs() {
            logical_x.push(PauliOperator::hermitian(&u[..n], &u[n..], false)?);
            logical_z.push(PauliOperator::hermitian(&v[..n], &v[n..], false)?);
        }

        let code = Self {
            group,
            n,
            k,
            logical_x,
            logical_z,
            stabilizer_basis: hb.radical().to_vec(),
        };
        code.verify_logicals()?;
        Ok(code)
    }

    /// Build from generators directly — [`StabilizerGroup::new`] then
    /// [`StabilizerCode::new`].
    ///
    /// # Errors
    ///
    /// As [`StabilizerGroup::new`] and [`StabilizerCode::new`].
    pub fn from_generators(generators: Vec<PauliOperator>) -> Result<Self, StabilizerError> {
        Self::new(StabilizerGroup::new(generators)?)
    }

    /// Build from generators on an explicitly stated number of qubits — the
    /// only way to build the `[[n, n]]` code with no stabilizer at all.
    ///
    /// # Errors
    ///
    /// As [`StabilizerGroup::with_qubits`] and [`StabilizerCode::new`].
    pub fn from_generators_on(
        n: usize,
        generators: Vec<PauliOperator>,
    ) -> Result<Self, StabilizerError> {
        Self::new(StabilizerGroup::with_qubits(n, generators)?)
    }

    /// The checks that run before a code is handed back.
    ///
    /// Every logical must commute with every stabilizer generator, `X_i` must
    /// anticommute with `Z_i` and commute with everything else, and the
    /// stabilizer basis must have the right rank.
    fn verify_logicals(&self) -> Result<(), StabilizerError> {
        for (i, lx) in self.logical_x.iter().enumerate() {
            for (g, gen) in self.group.generators().iter().enumerate() {
                if gen.symplectic_product(lx)? != 0 {
                    return Err(StabilizerError::InconsistentResult {
                        reason: format!("logical X_{i} anticommutes with stabilizer generator {g}"),
                    });
                }
            }
        }
        for (i, lz) in self.logical_z.iter().enumerate() {
            for (g, gen) in self.group.generators().iter().enumerate() {
                if gen.symplectic_product(lz)? != 0 {
                    return Err(StabilizerError::InconsistentResult {
                        reason: format!("logical Z_{i} anticommutes with stabilizer generator {g}"),
                    });
                }
            }
        }
        for i in 0..self.k {
            for j in 0..self.k {
                let want = u8::from(i == j);
                if self.logical_x[i].symplectic_product(&self.logical_z[j])? != want {
                    return Err(StabilizerError::InconsistentResult {
                        reason: format!("⟨X_{i}, Z_{j}⟩ is not {want}"),
                    });
                }
                if i != j
                    && (self.logical_x[i].symplectic_product(&self.logical_x[j])? != 0
                        || self.logical_z[i].symplectic_product(&self.logical_z[j])? != 0)
                {
                    return Err(StabilizerError::InconsistentResult {
                        reason: format!("logical pairs {i} and {j} are not independent"),
                    });
                }
            }
        }
        Ok(())
    }

    /// The block length `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of logical qubits, `k = n − rank(S)`.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The stabilizer group.
    pub fn stabilizer(&self) -> &StabilizerGroup {
        &self.group
    }

    /// The `k` logical `X` operators.
    pub fn logical_x(&self) -> &[PauliOperator] {
        &self.logical_x
    }

    /// The `k` logical `Z` operators.
    pub fn logical_z(&self) -> &[PauliOperator] {
        &self.logical_z
    }

    /// The syndrome of `error` — one bit per stabilizer generator.
    ///
    /// # Errors
    ///
    /// `E-STAB-003` for an error on a different number of qubits.
    pub fn syndrome(&self, error: &PauliOperator) -> Result<Vec<u8>, StabilizerError> {
        self.group.syndrome(error)
    }

    /// Is `error` a **logical** error — in `N(S)` but not in `S`?
    ///
    /// These are exactly the errors the syndrome cannot see.
    ///
    /// # Errors
    ///
    /// `E-STAB-003` for a mismatched qubit count; `E-GFQ-*` from the solve.
    pub fn is_logical_error(&self, error: &PauliOperator) -> Result<bool, StabilizerError> {
        if !self.group.centralizes(error)? {
            return Ok(false);
        }
        // Compare symplectic parts only: a sign difference from S is a
        // stabilizer times −I, which acts as the identity on the code space.
        let mut rows = self.stabilizer_basis.clone();
        let r = super::bit_rank(&rows, 2 * self.n)?;
        rows.push(error.symplectic());
        Ok(super::bit_rank(&rows, 2 * self.n)? > r)
    }

    /// The **exact** minimum distance, by exhaustive search over `N(S)`.
    ///
    /// Returns [`Distance::Exact`] or refuses; it never guesses.
    ///
    /// # Errors
    ///
    /// `E-STAB-007` when `k = 0` (there is no `N(S) \ S` to minimise over);
    /// `E-STAB-008` when `n + k` exceeds [`MAX_DISTANCE_SEARCH_DIM`].
    pub fn minimum_distance(&self) -> Result<Distance, StabilizerError> {
        self.minimum_distance_with_cap(MAX_DISTANCE_SEARCH_DIM)
    }

    /// [`StabilizerCode::minimum_distance`] with an explicit cap on `n + k`.
    ///
    /// Raising the cap raises the running time by a factor of two per step, and
    /// nothing else changes. There is no cleverer algorithm behind it.
    ///
    /// # Errors
    ///
    /// As [`StabilizerCode::minimum_distance`].
    pub fn minimum_distance_with_cap(&self, cap: usize) -> Result<Distance, StabilizerError> {
        if self.k == 0 {
            return Err(StabilizerError::NoLogicalQubits { n: self.n });
        }
        let logicals: Vec<Vec<u8>> = self
            .logical_x
            .iter()
            .chain(self.logical_z.iter())
            .map(|p| p.symplectic())
            .collect();
        let d = min_weight_outside(&logicals, &self.stabilizer_basis, self.n, cap)?;
        Ok(Distance::Exact(d))
    }

    /// An **upper bound** on the distance, available at any size.
    ///
    /// The bound is the least weight over the non-trivial logical classes
    /// *without* multiplying by stabilizer elements (and, when `2k` is itself
    /// too large to enumerate, over the `2k` logical generators alone). It is a
    /// witness, not a search: the true distance can be strictly smaller, which
    /// is why it comes back as [`Distance::UpperBound`].
    ///
    /// # Errors
    ///
    /// `E-STAB-007` when `k = 0`.
    pub fn distance_upper_bound(&self) -> Result<Distance, StabilizerError> {
        if self.k == 0 {
            return Err(StabilizerError::NoLogicalQubits { n: self.n });
        }
        let logicals: Vec<Vec<u8>> = self
            .logical_x
            .iter()
            .chain(self.logical_z.iter())
            .map(|p| p.symplectic())
            .collect();
        let best = if logicals.len() <= 20 {
            min_weight_outside(&logicals, &[], self.n, logicals.len())?
        } else {
            logicals
                .iter()
                .map(|v| Packed::pack(v, self.n).weight())
                .min()
                .unwrap_or(0)
        };
        Ok(Distance::UpperBound(best))
    }
}

impl fmt::Display for StabilizerCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[[{}, {}]]", self.n, self.k)
    }
}
