//! Classical matrix groups over GF(q): `GL(n, q)`, `SL(n, q)` and `Sp(2n, q)`.
//!
//! This is a deliberately **modest** surface. Three things are on offer and
//! nothing else:
//!
//! * **Exact order**, from the standard product formulas, as an
//!   arbitrary-precision [`rug::Integer`]. This is closed-form and has no size
//!   limit at all — `|GL(100, 2^31−1)|` is a perfectly ordinary call.
//! * **Membership testing**, from the defining condition:
//!   `det ≠ 0`, `det = 1`, or `MᵀΩM = Ω`.
//! * **The induced permutation action** on the non-zero vectors of `F_q^d`,
//!   handed to [`crate::group::PermutationGroup`] so that orbits, a base and
//!   strong generating set, and membership by sifting all come from the
//!   existing Schreier–Sims rather than a second implementation.
//!
//! # The formulas
//!
//! ```text
//!     |GL(n, q)|   = ∏_{i=0}^{n−1} (qⁿ − qⁱ)
//!     |SL(n, q)|   = |GL(n, q)| / (q − 1)
//!     |Sp(2n, q)|  = q^{n²} · ∏_{i=1}^{n} (q^{2i} − 1)
//! ```
//!
//! so `|GL(2,2)| = 6`, `|GL(3,2)| = 168`, `|Sp(2,2)| = 6` and `|Sp(4,2)| = 720`
//! — all four checked in `stabilizer::tests`, the last two twice over, once
//! from the formula and once by brute-force enumeration.
//!
//! # Scope limits
//!
//! * **No generating set from theory.** Small generating sets for these groups
//!   exist (a transvection and a Singer cycle, say), but getting them right for
//!   every `(n, q)` — including the degenerate `SL(2, 2)` and `SL(2, 3)` — is a
//!   separate piece of work. [`MatrixGroup::elements`] brute-forces over all
//!   `q^(d²)` matrices of the right size and filters, which is honest, easy to
//!   check, and capped at [`MAX_MATRIX_ENUMERATION`] candidates.
//! * **No Schreier–Sims on matrices.** [`MatrixGroup::permutation_action`]
//!   converts to permutations and reuses the existing implementation; there is
//!   no matrix-group stabilizer chain here.
//! * **No orthogonal or unitary groups**, no projective quotients
//!   (`PGL`, `PSL`, `PSp`), no twisted types.

use rug::ops::Pow;
use rug::Integer;

use super::symplectic::is_symplectic;
use super::StabilizerError;
use crate::ffield::{FieldElement, FiniteField, GfMatrix};
use crate::group::{Permutation, PermutationGroup};

/// The largest number of candidate matrices [`MatrixGroup::elements`] will scan.
///
/// Enumeration is brute force over all `q^(d²)` matrices of size `d`, so the
/// cap is on `q^(d²)` and not on `|G|`. `Sp(4, 2)` scans `2^16 = 65 536`
/// candidates to find its 720 elements; `Sp(4, 3)` would scan `3^16 ≈ 4.3 × 10⁷`
/// and is refused. The **order** is available in either case: it comes from a
/// formula, not from the list.
pub const MAX_MATRIX_ENUMERATION: u64 = 1 << 20;

/// Which classical family a [`MatrixGroup`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatrixGroupKind {
    /// `GL(n, q)` — all invertible `n × n` matrices.
    GeneralLinear,
    /// `SL(n, q)` — determinant `1`.
    SpecialLinear,
    /// `Sp(2n, q)` — preserving `Ω = [[0, I], [−I, 0]]`; the matrices are
    /// `2n × 2n`.
    Symplectic,
}

impl MatrixGroupKind {
    fn name(&self) -> &'static str {
        match self {
            MatrixGroupKind::GeneralLinear => "GL",
            MatrixGroupKind::SpecialLinear => "SL",
            MatrixGroupKind::Symplectic => "Sp",
        }
    }
}

/// A classical matrix group over GF(q).
#[derive(Clone, Debug)]
pub struct MatrixGroup {
    kind: MatrixGroupKind,
    field: FiniteField,
    /// The size of the matrices: `n` for `GL`/`SL`, `2n` for `Sp`.
    degree: usize,
}

impl MatrixGroup {
    /// `GL(n, q)`.
    ///
    /// # Errors
    ///
    /// `E-STAB-002` for `n = 0`.
    pub fn general_linear(field: &FiniteField, n: usize) -> Result<Self, StabilizerError> {
        Self::linear(MatrixGroupKind::GeneralLinear, field, n)
    }

    /// `SL(n, q)`.
    ///
    /// # Errors
    ///
    /// `E-STAB-002` for `n = 0`.
    pub fn special_linear(field: &FiniteField, n: usize) -> Result<Self, StabilizerError> {
        Self::linear(MatrixGroupKind::SpecialLinear, field, n)
    }

    fn linear(
        kind: MatrixGroupKind,
        field: &FiniteField,
        n: usize,
    ) -> Result<Self, StabilizerError> {
        if n == 0 {
            return Err(StabilizerError::ShapeMismatch {
                op: "build a linear group",
                expected: "n ≥ 1".to_string(),
                got: "n = 0".to_string(),
            });
        }
        Ok(Self {
            kind,
            field: field.clone(),
            degree: n,
        })
    }

    /// `Sp(2n, q)` — note the argument is `n`, so the matrices are `2n × 2n`.
    ///
    /// # Errors
    ///
    /// `E-STAB-002` for `n = 0`.
    pub fn symplectic(field: &FiniteField, n: usize) -> Result<Self, StabilizerError> {
        if n == 0 {
            return Err(StabilizerError::ShapeMismatch {
                op: "build a symplectic group",
                expected: "n ≥ 1, giving 2n × 2n matrices".to_string(),
                got: "n = 0".to_string(),
            });
        }
        Ok(Self {
            kind: MatrixGroupKind::Symplectic,
            field: field.clone(),
            degree: 2 * n,
        })
    }

    /// Which family this is.
    pub fn kind(&self) -> MatrixGroupKind {
        self.kind
    }

    /// The field.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// The size of the matrices — `n` for `GL`/`SL`, `2n` for `Sp`.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// `q`, the field order, as an arbitrary-precision integer.
    pub fn field_order(&self) -> Integer {
        Integer::from(self.field.characteristic()).pow(self.field.degree() as u32)
    }

    /// The exact group order, from the product formula.
    ///
    /// No enumeration and no cap: this is closed form.
    ///
    /// ```
    /// use alkahest_cas::experimental::{FiniteField, MatrixGroup};
    ///
    /// let gf2 = FiniteField::prime(2).unwrap();
    /// assert_eq!(MatrixGroup::general_linear(&gf2, 3).unwrap().order(), 168);
    /// assert_eq!(MatrixGroup::symplectic(&gf2, 2).unwrap().order(), 720);
    /// ```
    pub fn order(&self) -> Integer {
        let q = self.field_order();
        match self.kind {
            MatrixGroupKind::GeneralLinear => gl_order(&q, self.degree),
            MatrixGroupKind::SpecialLinear => {
                let full = gl_order(&q, self.degree);
                full / (q - Integer::from(1))
            }
            MatrixGroupKind::Symplectic => {
                let n = self.degree / 2;
                // q^{n²} · ∏_{i=1}^{n} (q^{2i} − 1)
                let mut acc = q.clone().pow((n * n) as u32);
                for i in 1..=n {
                    acc *= q.clone().pow((2 * i) as u32) - Integer::from(1);
                }
                acc
            }
        }
    }

    /// Is `m` an element of this group?
    ///
    /// # Errors
    ///
    /// `E-STAB-012` when `m` is over a different field; `E-STAB-002` when `m`
    /// has the wrong shape; `E-GFQ-*` from the determinant.
    pub fn contains(&self, m: &GfMatrix) -> Result<bool, StabilizerError> {
        if m.field() != &self.field {
            return Err(StabilizerError::FieldMismatch {
                lhs: format!("{}", self.field),
                rhs: format!("{}", m.field()),
            });
        }
        let (r, c) = m.shape();
        if r != self.degree || c != self.degree {
            return Err(StabilizerError::ShapeMismatch {
                op: "matrix-group membership",
                expected: format!("a {}x{} matrix", self.degree, self.degree),
                got: format!("{r}x{c}"),
            });
        }
        match self.kind {
            MatrixGroupKind::GeneralLinear => Ok(!m.determinant()?.is_zero()),
            MatrixGroupKind::SpecialLinear => Ok(m.determinant()? == self.field.one()),
            MatrixGroupKind::Symplectic => is_symplectic(m),
        }
    }

    /// Every element of the group, by brute force over all `q^(d²)` matrices.
    ///
    /// # Errors
    ///
    /// `E-STAB-011` when `q^(d²)` exceeds [`MAX_MATRIX_ENUMERATION`]. The
    /// [`MatrixGroup::order`] is still exact in that case; it is the *list*
    /// that is refused.
    pub fn elements(&self) -> Result<Vec<GfMatrix>, StabilizerError> {
        self.elements_with_cap(MAX_MATRIX_ENUMERATION)
    }

    /// [`MatrixGroup::elements`] with an explicit candidate cap.
    ///
    /// # Errors
    ///
    /// `E-STAB-011` above the cap.
    pub fn elements_with_cap(&self, cap: u64) -> Result<Vec<GfMatrix>, StabilizerError> {
        let q = self.field_order();
        let cells = self.degree * self.degree;
        let candidates = q.clone().pow(cells as u32);
        if candidates > cap {
            return Err(StabilizerError::EnumerationTooLarge {
                candidates: candidates.to_string(),
                cap,
            });
        }
        let total = candidates
            .to_u64()
            .ok_or(StabilizerError::EnumerationTooLarge {
                candidates: candidates.to_string(),
                cap,
            })?;
        let alphabet = field_elements(&self.field)?;
        let qs = alphabet.len() as u64;

        let mut out = Vec::new();
        let mut entries = vec![self.field.zero(); cells];
        for code in 0..total {
            let mut rest = code;
            for entry in entries.iter_mut() {
                *entry = alphabet[(rest % qs) as usize].clone();
                rest /= qs;
            }
            let m = GfMatrix::from_elements(&self.field, self.degree, self.degree, &entries)?;
            if self.contains(&m)? {
                out.push(m);
            }
        }
        Ok(out)
    }

    /// The induced action on the `q^d − 1` non-zero vectors of `F_q^d`, as a
    /// [`PermutationGroup`].
    ///
    /// The action is on **row** vectors, `v ↦ v·M`, so that it composes the
    /// same way round as [`crate::group`]'s left-to-right
    /// [`Permutation::compose`]. It is faithful for every subgroup of
    /// `GL(d, q)` — if `vM = v` for every `v` then `M = I` — so the returned
    /// group has exactly [`MatrixGroup::order`] elements, which is what
    /// `stabilizer::tests` asserts against Schreier–Sims.
    ///
    /// Points are numbered by reading each vector's coordinates as base-`q`
    /// digits (least significant first, each coordinate itself a base-`p`
    /// number over the polynomial basis), then subtracting one to skip the
    /// zero vector.
    ///
    /// # Errors
    ///
    /// `E-STAB-011` when the element enumeration is capped out;
    /// `E-GRP-005` when `q^d − 1` exceeds
    /// [`MAX_BSGS_DEGREE`](crate::group::MAX_BSGS_DEGREE) and the chain cannot
    /// be built.
    pub fn permutation_action(&self) -> Result<PermutationGroup, StabilizerError> {
        let elements = self.elements()?;
        let alphabet = field_elements(&self.field)?;
        let q = alphabet.len();
        let points = q.pow(self.degree as u32) - 1;

        let vectors = nonzero_vectors(&self.field, self.degree, &alphabet)?;
        let target = self.order();

        let mut gens: Vec<Permutation> = Vec::new();
        let mut group = perm_group(points, &gens)?;
        for m in &elements {
            let p = induced_permutation(&self.field, m, &vectors, q, self.degree)?;
            if !gens.is_empty() && group.contains(&p).map_err(group_err)? {
                continue;
            }
            if gens.is_empty() && p.is_identity() {
                continue;
            }
            gens.push(p);
            group = perm_group(points, &gens)?;
            if group.order().map_err(group_err)? == target {
                break;
            }
        }
        Ok(group)
    }
}

fn group_err(e: crate::group::GroupError) -> StabilizerError {
    StabilizerError::InconsistentResult {
        reason: format!("the permutation layer refused: {e}"),
    }
}

fn perm_group(points: usize, gens: &[Permutation]) -> Result<PermutationGroup, StabilizerError> {
    PermutationGroup::new(points, gens.to_vec()).map_err(group_err)
}

fn gl_order(q: &Integer, n: usize) -> Integer {
    let qn = q.clone().pow(n as u32);
    let mut acc = Integer::from(1);
    for i in 0..n {
        acc *= qn.clone() - q.clone().pow(i as u32);
    }
    acc
}

/// Every element of GF(q), in the order used to number vectors.
fn field_elements(field: &FiniteField) -> Result<Vec<FieldElement>, StabilizerError> {
    let p = field.characteristic();
    let k = field.degree();
    let q = field
        .order()
        .and_then(|o| usize::try_from(o).ok())
        .ok_or_else(|| StabilizerError::EnumerationTooLarge {
            candidates: format!("GF({p}^{k})"),
            cap: MAX_MATRIX_ENUMERATION,
        })?;
    let mut out = Vec::with_capacity(q);
    for code in 0..q {
        let mut rest = code as u64;
        let mut coeffs = Vec::with_capacity(k);
        for _ in 0..k {
            coeffs.push(rest % p);
            rest /= p;
        }
        out.push(field.element(&coeffs)?);
    }
    Ok(out)
}

/// The non-zero vectors of `F_q^d`, in point order.
fn nonzero_vectors(
    field: &FiniteField,
    d: usize,
    alphabet: &[FieldElement],
) -> Result<Vec<GfMatrix>, StabilizerError> {
    let q = alphabet.len();
    let total = q.pow(d as u32);
    let mut out = Vec::with_capacity(total - 1);
    for code in 1..total {
        let mut rest = code;
        let mut entries = Vec::with_capacity(d);
        for _ in 0..d {
            entries.push(alphabet[rest % q].clone());
            rest /= q;
        }
        out.push(GfMatrix::from_elements(field, 1, d, &entries)?);
    }
    Ok(out)
}

/// The point index of a `1 × d` row vector: base-`q` digits, least significant
/// first, minus one for the skipped zero vector.
fn vector_index(field: &FiniteField, v: &GfMatrix, q: usize, d: usize) -> usize {
    let k = field.degree();
    let p = field.characteristic() as usize;
    let mut idx = 0usize;
    let mut scale = 1usize;
    for j in 0..d {
        let e = v.entry(0, j).unwrap_or_else(|_| field.zero());
        let digits = e.coefficients_padded(k);
        let mut val = 0usize;
        let mut ps = 1usize;
        for c in digits {
            val += (c as usize) * ps;
            ps *= p;
        }
        idx += val * scale;
        scale *= q;
    }
    idx - 1
}

fn induced_permutation(
    field: &FiniteField,
    m: &GfMatrix,
    vectors: &[GfMatrix],
    q: usize,
    d: usize,
) -> Result<Permutation, StabilizerError> {
    let mut images = Vec::with_capacity(vectors.len());
    for v in vectors {
        let w = v.mul(m)?;
        images.push(vector_index(field, &w, q, d));
    }
    Permutation::from_images(images).map_err(group_err)
}

impl std::fmt::Display for MatrixGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let q = self.field_order();
        match self.kind {
            MatrixGroupKind::Symplectic => {
                write!(f, "Sp({}, {})", self.degree, q)
            }
            k => write!(f, "{}({}, {})", k.name(), self.degree, q),
        }
    }
}
