//! Dense matrices over GF(q), backed by FLINT's `nmod_mat` and `fq_nmod_mat`.
//!
//! See the module docs in [`super`] for what is and is not in scope.

use std::fmt;

use super::error::FiniteFieldError;
use super::field::{FieldElement, FiniteField, NmodPoly};
use crate::flint::ffi;

/// Largest number of entries this module will ask FLINT to allocate.
///
/// This is a sanity bound, **not** a memory budget: 2³² entries is already 32 GB
/// for a prime field, so on most machines FLINT's allocator gives out well
/// before it. What it does catch is the shape that is obviously not meant —
/// a transposed pair of dimensions, a `usize` that came from a subtraction, the
/// `ncols × ncols` scratch [`GfMatrix::nullspace`] needs for a very wide
/// matrix — where the alternative is an abort inside FLINT that no `Result` can
/// intercept. Sizing the real ceiling would mean asking the OS how much memory
/// is left, which `crate::budget` does for the caller at a higher level.
const MAX_ENTRIES: usize = 1 << 32;

// ---------------------------------------------------------------------------
// Owned FLINT matrices
// ---------------------------------------------------------------------------

/// Owned `nmod_mat_t`.
struct NmodMat {
    inner: ffi::NmodMatStruct,
}

// SAFETY: owns its FLINT allocation; no thread-local state.
unsafe impl Send for NmodMat {}
unsafe impl Sync for NmodMat {}

impl NmodMat {
    fn new(rows: usize, cols: usize, p: u64) -> Self {
        // SAFETY: `zeroed` is a valid starting point; `nmod_mat_init` writes
        // every field before any read, and the dimensions were bounds-checked
        // by the caller (`check_shape`).
        let mut inner: ffi::NmodMatStruct = unsafe { std::mem::zeroed() };
        unsafe { ffi::nmod_mat_init(&mut inner, rows as ffi::slong, cols as ffi::slong, p) };
        Self { inner }
    }

    fn rows(&self) -> usize {
        self.inner.r as usize
    }

    fn cols(&self) -> usize {
        self.inner.c as usize
    }

    /// Entry read. Delegated to FLINT so the row offset is computed by the
    /// installed library rather than from a mirrored struct layout.
    fn get(&self, i: usize, j: usize) -> u64 {
        debug_assert!(i < self.rows() && j < self.cols());
        // SAFETY: indices are in bounds; `inner` is initialised.
        unsafe { ffi::nmod_mat_get_entry(&self.inner, i as ffi::slong, j as ffi::slong) }
    }

    fn set(&mut self, i: usize, j: usize, v: u64) {
        debug_assert!(i < self.rows() && j < self.cols());
        // SAFETY: indices are in bounds; `inner` is initialised.
        unsafe { ffi::nmod_mat_set_entry(&mut self.inner, i as ffi::slong, j as ffi::slong, v) };
    }
}

impl Drop for NmodMat {
    fn drop(&mut self) {
        // SAFETY: initialised in `new`.
        unsafe { ffi::nmod_mat_clear(&mut self.inner) };
    }
}

/// Owned `fq_nmod_mat_t`, together with the field whose context it borrows.
///
/// The field is held here rather than passed to `drop` because FLINT's
/// `fq_nmod_mat_clear` needs the context, and a `Drop` impl takes no arguments.
struct FqMat {
    inner: ffi::FqNmodMatStruct,
    field: FiniteField,
}

// SAFETY: owns its FLINT allocation; the borrowed context is immutable and is
// kept alive by the `FiniteField` handle stored alongside it.
unsafe impl Send for FqMat {}
unsafe impl Sync for FqMat {}

impl FqMat {
    fn new(rows: usize, cols: usize, field: &FiniteField) -> Self {
        // SAFETY: as `NmodMat::new`; `field` is an extension field, so
        // `ctx_ptr` is live for at least as long as this matrix.
        let mut inner: ffi::FqNmodMatStruct = unsafe { std::mem::zeroed() };
        unsafe {
            ffi::fq_nmod_mat_init(
                &mut inner,
                rows as ffi::slong,
                cols as ffi::slong,
                field.ctx_ptr(),
            )
        };
        Self {
            inner,
            field: field.clone(),
        }
    }

    fn rows(&self) -> usize {
        self.inner.r as usize
    }

    fn cols(&self) -> usize {
        self.inner.c as usize
    }

    fn ctx(&self) -> *const ffi::FqNmodCtxBuf {
        self.field.ctx_ptr()
    }

    fn get(&self, i: usize, j: usize) -> FieldElement {
        debug_assert!(i < self.rows() && j < self.cols());
        let k = self.field.degree();
        // SAFETY: indices in bounds. `fq_nmod_mat_entry` returns a borrowed
        // pointer to a live `fq_nmod_struct` (= `nmod_poly_struct`); we only
        // read coefficients from it.
        unsafe {
            let e = ffi::fq_nmod_mat_entry(&self.inner, i as ffi::slong, j as ffi::slong);
            let coeffs: Vec<u64> = (0..k)
                .map(|t| ffi::nmod_poly_get_coeff_ui(e, t as ffi::slong))
                .collect();
            self.field
                .element(&coeffs)
                .expect("k coefficients always fit a degree-k field")
        }
    }

    fn set(&mut self, i: usize, j: usize, scratch: &mut NmodPoly, v: &FieldElement) {
        debug_assert!(i < self.rows() && j < self.cols());
        // SAFETY: `scratch` is a live `nmod_poly_t` over the same characteristic,
        // which is what FLINT means by `fq_nmod_t`. Coefficients below the
        // extension degree need no reduction, and `FieldElement` is canonical.
        unsafe {
            ffi::fq_nmod_zero(scratch.as_mut_ptr(), self.ctx());
            for (t, &c) in v.coefficients().iter().enumerate() {
                ffi::nmod_poly_set_coeff_ui(scratch.as_mut_ptr(), t as ffi::slong, c);
            }
            ffi::fq_nmod_mat_entry_set(
                &mut self.inner,
                i as ffi::slong,
                j as ffi::slong,
                scratch.as_ptr(),
                self.ctx(),
            );
        }
    }
}

impl Drop for FqMat {
    fn drop(&mut self) {
        // SAFETY: initialised in `new`; the context outlives it via `self.field`.
        unsafe { ffi::fq_nmod_mat_clear(&mut self.inner, self.field.ctx_ptr()) };
    }
}

enum Repr {
    Prime(NmodMat),
    Ext(FqMat),
}

// ---------------------------------------------------------------------------
// GfMatrix
// ---------------------------------------------------------------------------

/// A dense `rows × cols` matrix over GF(q).
///
/// Rectangular shapes are first-class: a parity-check matrix is never square,
/// and every operation here that can accept a non-square argument does.
pub struct GfMatrix {
    field: FiniteField,
    repr: Repr,
}

/// The output of [`GfMatrix::rref`]: a reduced row echelon form together with
/// the invertible row transform that produced it.
pub struct Rref {
    /// `R`, the reduced row echelon form of `A` — same shape as `A`.
    pub matrix: GfMatrix,
    /// `U`, invertible and `rows × rows`, with `U · A = R`.
    pub transform: GfMatrix,
    /// The rank of `A`, i.e. the number of non-zero rows of `R`.
    pub rank: usize,
    /// Pivot column index of each of the first `rank` rows, ascending.
    pub pivots: Vec<usize>,
}

fn check_shape(rows: usize, cols: usize) -> Result<(), FiniteFieldError> {
    let too_big = rows > i64::MAX as usize
        || cols > i64::MAX as usize
        || rows
            .checked_mul(cols)
            .map(|n| n > MAX_ENTRIES)
            .unwrap_or(true);
    if too_big {
        return Err(FiniteFieldError::DimensionTooLarge { rows, cols });
    }
    Ok(())
}

impl GfMatrix {
    /// The all-zero `rows × cols` matrix.
    ///
    /// # Errors
    ///
    /// `E-GFQ-012` when the shape exceeds what FLINT can allocate.
    pub fn zeros(field: &FiniteField, rows: usize, cols: usize) -> Result<Self, FiniteFieldError> {
        check_shape(rows, cols)?;
        Ok(Self::zeros_unchecked(field, rows, cols))
    }

    fn zeros_unchecked(field: &FiniteField, rows: usize, cols: usize) -> Self {
        let repr = if field.is_prime_field() {
            Repr::Prime(NmodMat::new(rows, cols, field.characteristic()))
        } else {
            Repr::Ext(FqMat::new(rows, cols, field))
        };
        Self {
            field: field.clone(),
            repr,
        }
    }

    /// The `n × n` identity.
    ///
    /// # Errors
    ///
    /// `E-GFQ-012` when the shape exceeds what FLINT can allocate.
    pub fn identity(field: &FiniteField, n: usize) -> Result<Self, FiniteFieldError> {
        let mut m = Self::zeros(field, n, n)?;
        let one = field.one();
        let mut scratch = NmodPoly::new(field.characteristic());
        for i in 0..n {
            m.put(i, i, &one, &mut scratch);
        }
        Ok(m)
    }

    /// Build from row-major integer entries, each read as an element of the
    /// **prime subfield** GF(p) and reduced modulo `p`.
    ///
    /// This is the constructor for parity-check and generator matrices over
    /// GF(2): `from_u64(&gf2, 3, 7, &[1,1,1,0,1,0,0, …])`.
    ///
    /// # Errors
    ///
    /// `E-GFQ-007` when `entries.len() != rows * cols`; `E-GFQ-012` for a shape
    /// FLINT cannot allocate.
    pub fn from_u64(
        field: &FiniteField,
        rows: usize,
        cols: usize,
        entries: &[u64],
    ) -> Result<Self, FiniteFieldError> {
        check_shape(rows, cols)?;
        if entries.len() != rows * cols {
            return Err(FiniteFieldError::DimensionMismatch {
                op: "fill",
                lhs: (rows, cols),
                rhs: (entries.len(), 1),
            });
        }
        let mut m = Self::zeros_unchecked(field, rows, cols);
        let mut scratch = NmodPoly::new(field.characteristic());
        for i in 0..rows {
            for j in 0..cols {
                let e = field.scalar(entries[i * cols + j]);
                m.put(i, j, &e, &mut scratch);
            }
        }
        Ok(m)
    }

    /// Build from row-major field elements.
    ///
    /// # Errors
    ///
    /// `E-GFQ-007` when `entries.len() != rows * cols`; `E-GFQ-005` when an
    /// element carries more coefficients than the extension degree;
    /// `E-GFQ-012` for a shape FLINT cannot allocate.
    pub fn from_elements(
        field: &FiniteField,
        rows: usize,
        cols: usize,
        entries: &[FieldElement],
    ) -> Result<Self, FiniteFieldError> {
        check_shape(rows, cols)?;
        if entries.len() != rows * cols {
            return Err(FiniteFieldError::DimensionMismatch {
                op: "fill",
                lhs: (rows, cols),
                rhs: (entries.len(), 1),
            });
        }
        for e in entries {
            if e.coefficients().len() > field.degree() {
                return Err(FiniteFieldError::MalformedElement {
                    got: e.coefficients().len(),
                    degree: field.degree(),
                });
            }
        }
        let mut m = Self::zeros_unchecked(field, rows, cols);
        let mut scratch = NmodPoly::new(field.characteristic());
        for i in 0..rows {
            for j in 0..cols {
                m.put(i, j, &entries[i * cols + j], &mut scratch);
            }
        }
        Ok(m)
    }

    /// The field this matrix lives over.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        match &self.repr {
            Repr::Prime(m) => m.rows(),
            Repr::Ext(m) => m.rows(),
        }
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        match &self.repr {
            Repr::Prime(m) => m.cols(),
            Repr::Ext(m) => m.cols(),
        }
    }

    /// Shape as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Entry `(i, j)`.
    ///
    /// # Errors
    ///
    /// `E-GFQ-011` when the index is outside the matrix.
    pub fn entry(&self, i: usize, j: usize) -> Result<FieldElement, FiniteFieldError> {
        self.check_index(i, j)?;
        Ok(self.get(i, j))
    }

    /// Overwrite entry `(i, j)`.
    ///
    /// # Errors
    ///
    /// `E-GFQ-011` for an out-of-range index, `E-GFQ-005` for an element with
    /// more coefficients than the extension degree.
    pub fn set_entry(
        &mut self,
        i: usize,
        j: usize,
        value: &FieldElement,
    ) -> Result<(), FiniteFieldError> {
        self.check_index(i, j)?;
        if value.coefficients().len() > self.field.degree() {
            return Err(FiniteFieldError::MalformedElement {
                got: value.coefficients().len(),
                degree: self.field.degree(),
            });
        }
        let mut scratch = NmodPoly::new(self.field.characteristic());
        self.put(i, j, value, &mut scratch);
        Ok(())
    }

    /// All entries, row-major.
    pub fn to_elements(&self) -> Vec<FieldElement> {
        let (r, c) = self.shape();
        let mut out = Vec::with_capacity(r * c);
        for i in 0..r {
            for j in 0..c {
                out.push(self.get(i, j));
            }
        }
        out
    }

    /// All entries as integers, row-major — `None` when any entry lies outside
    /// the prime subfield.
    pub fn to_u64(&self) -> Option<Vec<u64>> {
        self.to_elements()
            .iter()
            .map(FieldElement::as_u64)
            .collect()
    }

    /// `true` when every entry is zero.
    pub fn is_zero(&self) -> bool {
        match &self.repr {
            Repr::Prime(m) => (0..m.rows()).all(|i| (0..m.cols()).all(|j| m.get(i, j) == 0)),
            Repr::Ext(m) => (0..m.rows()).all(|i| (0..m.cols()).all(|j| m.get(i, j).is_zero())),
        }
    }

    // -- internal accessors --------------------------------------------------

    fn check_index(&self, i: usize, j: usize) -> Result<(), FiniteFieldError> {
        let (rows, cols) = self.shape();
        if i >= rows || j >= cols {
            return Err(FiniteFieldError::IndexOutOfBounds { i, j, rows, cols });
        }
        Ok(())
    }

    fn get(&self, i: usize, j: usize) -> FieldElement {
        match &self.repr {
            Repr::Prime(m) => self.field.scalar(m.get(i, j)),
            Repr::Ext(m) => m.get(i, j),
        }
    }

    fn put(&mut self, i: usize, j: usize, v: &FieldElement, scratch: &mut NmodPoly) {
        match &mut self.repr {
            Repr::Prime(m) => m.set(i, j, v.as_u64().unwrap_or(0)),
            Repr::Ext(m) => m.set(i, j, scratch, v),
        }
    }

    fn prime_mut(&mut self) -> &mut NmodMat {
        match &mut self.repr {
            Repr::Prime(m) => m,
            Repr::Ext(_) => unreachable!("prime_mut() on an extension-field matrix"),
        }
    }

    fn ext_mut(&mut self) -> &mut FqMat {
        match &mut self.repr {
            Repr::Ext(m) => m,
            Repr::Prime(_) => unreachable!("ext_mut() on a prime-field matrix"),
        }
    }

    fn same_field(&self, other: &Self) -> Result<(), FiniteFieldError> {
        if self.field != other.field {
            return Err(FiniteFieldError::FieldMismatch {
                lhs: self.field.to_string(),
                rhs: other.field.to_string(),
            });
        }
        Ok(())
    }

    fn require_square(&self, op: &'static str) -> Result<usize, FiniteFieldError> {
        let (r, c) = self.shape();
        if r != c {
            return Err(FiniteFieldError::NotSquare { op, shape: (r, c) });
        }
        Ok(r)
    }

    // -- arithmetic ----------------------------------------------------------

    /// Entrywise sum.
    ///
    /// # Errors
    ///
    /// `E-GFQ-006` for different fields, `E-GFQ-007` for different shapes.
    pub fn add(&self, other: &Self) -> Result<Self, FiniteFieldError> {
        self.same_field(other)?;
        if self.shape() != other.shape() {
            return Err(FiniteFieldError::DimensionMismatch {
                op: "add",
                lhs: self.shape(),
                rhs: other.shape(),
            });
        }
        let (r, c) = self.shape();
        let mut out = Self::zeros_unchecked(&self.field, r, c);
        match (&self.repr, &other.repr) {
            // SAFETY: shapes and moduli agree (checked above); all three
            // matrices are live and distinct.
            (Repr::Prime(a), Repr::Prime(b)) => unsafe {
                ffi::nmod_mat_add(&mut out.prime_mut().inner, &a.inner, &b.inner)
            },
            (Repr::Ext(a), Repr::Ext(b)) => unsafe {
                ffi::fq_nmod_mat_add(
                    &mut out.ext_mut().inner,
                    &a.inner,
                    &b.inner,
                    self.field.ctx_ptr(),
                )
            },
            _ => unreachable!("equal fields imply equal representations"),
        }
        Ok(out)
    }

    /// Entrywise difference.
    ///
    /// # Errors
    ///
    /// `E-GFQ-006` for different fields, `E-GFQ-007` for different shapes.
    pub fn sub(&self, other: &Self) -> Result<Self, FiniteFieldError> {
        self.same_field(other)?;
        if self.shape() != other.shape() {
            return Err(FiniteFieldError::DimensionMismatch {
                op: "subtract",
                lhs: self.shape(),
                rhs: other.shape(),
            });
        }
        let (r, c) = self.shape();
        let mut out = Self::zeros_unchecked(&self.field, r, c);
        match (&self.repr, &other.repr) {
            // SAFETY: as `add`.
            (Repr::Prime(a), Repr::Prime(b)) => unsafe {
                ffi::nmod_mat_sub(&mut out.prime_mut().inner, &a.inner, &b.inner)
            },
            (Repr::Ext(a), Repr::Ext(b)) => unsafe {
                ffi::fq_nmod_mat_sub(
                    &mut out.ext_mut().inner,
                    &a.inner,
                    &b.inner,
                    self.field.ctx_ptr(),
                )
            },
            _ => unreachable!("equal fields imply equal representations"),
        }
        Ok(out)
    }

    /// Additive inverse.
    pub fn neg(&self) -> Self {
        let (r, c) = self.shape();
        let mut out = Self::zeros_unchecked(&self.field, r, c);
        match &self.repr {
            // SAFETY: shapes agree by construction.
            Repr::Prime(a) => unsafe { ffi::nmod_mat_neg(&mut out.prime_mut().inner, &a.inner) },
            Repr::Ext(a) => unsafe {
                ffi::fq_nmod_mat_neg(&mut out.ext_mut().inner, &a.inner, self.field.ctx_ptr())
            },
        }
        out
    }

    /// Multiply every entry by `c`.
    ///
    /// # Errors
    ///
    /// `E-GFQ-005` when `c` has more coefficients than the extension degree.
    pub fn scalar_mul(&self, c: &FieldElement) -> Result<Self, FiniteFieldError> {
        if c.coefficients().len() > self.field.degree() {
            return Err(FiniteFieldError::MalformedElement {
                got: c.coefficients().len(),
                degree: self.field.degree(),
            });
        }
        let (r, cols) = self.shape();
        let mut out = Self::zeros_unchecked(&self.field, r, cols);
        match &self.repr {
            Repr::Prime(a) => {
                // SAFETY: shapes and moduli agree by construction.
                unsafe {
                    ffi::nmod_mat_scalar_mul(
                        &mut out.prime_mut().inner,
                        &a.inner,
                        c.as_u64().unwrap_or(0),
                    )
                }
            }
            Repr::Ext(a) => {
                let ctx = self.field.ctx_ptr();
                let p = self.field.characteristic();
                let mut coeff = NmodPoly::new(p);
                let mut acc = NmodPoly::new(p);
                let mut scratch = NmodPoly::new(p);
                // SAFETY: `coeff` holds the scalar as an `fq_nmod_t`; `acc`
                // receives each product. All three are live for the loop.
                unsafe {
                    ffi::fq_nmod_zero(coeff.as_mut_ptr(), ctx);
                    for (t, &v) in c.coefficients().iter().enumerate() {
                        ffi::nmod_poly_set_coeff_ui(coeff.as_mut_ptr(), t as ffi::slong, v);
                    }
                }
                for i in 0..r {
                    for j in 0..cols {
                        // SAFETY: indices in bounds; entry pointer is borrowed
                        // from a live matrix.
                        let e = unsafe {
                            ffi::fq_nmod_mat_entry(&a.inner, i as ffi::slong, j as ffi::slong)
                        };
                        unsafe {
                            ffi::fq_nmod_mul(acc.as_mut_ptr(), e, coeff.as_ptr(), ctx);
                        }
                        let v = self
                            .field
                            .element(&acc.to_coefficients())
                            .expect("a reduced product has degree < k");
                        out.put(i, j, &v, &mut scratch);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Matrix product `self · other`.
    ///
    /// # Errors
    ///
    /// `E-GFQ-006` for different fields, `E-GFQ-007` when the inner dimensions
    /// disagree.
    pub fn mul(&self, other: &Self) -> Result<Self, FiniteFieldError> {
        self.same_field(other)?;
        if self.ncols() != other.nrows() {
            return Err(FiniteFieldError::DimensionMismatch {
                op: "multiply",
                lhs: self.shape(),
                rhs: other.shape(),
            });
        }
        let mut out = Self::zeros_unchecked(&self.field, self.nrows(), other.ncols());
        match (&self.repr, &other.repr) {
            // SAFETY: inner dimensions agree (checked above); the destination
            // was allocated with the product shape.
            (Repr::Prime(a), Repr::Prime(b)) => unsafe {
                ffi::nmod_mat_mul(&mut out.prime_mut().inner, &a.inner, &b.inner)
            },
            (Repr::Ext(a), Repr::Ext(b)) => unsafe {
                ffi::fq_nmod_mat_mul(
                    &mut out.ext_mut().inner,
                    &a.inner,
                    &b.inner,
                    self.field.ctx_ptr(),
                )
            },
            _ => unreachable!("equal fields imply equal representations"),
        }
        Ok(out)
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let (r, c) = self.shape();
        let mut out = Self::zeros_unchecked(&self.field, c, r);
        match &self.repr {
            // SAFETY: the destination was allocated with the transposed shape.
            Repr::Prime(a) => unsafe {
                ffi::nmod_mat_transpose(&mut out.prime_mut().inner, &a.inner)
            },
            Repr::Ext(a) => unsafe {
                ffi::fq_nmod_mat_transpose(&mut out.ext_mut().inner, &a.inner, self.field.ctx_ptr())
            },
        }
        out
    }

    /// Entrywise equality. `false` when the fields or the shapes differ.
    pub fn equals(&self, other: &Self) -> bool {
        if self.field != other.field || self.shape() != other.shape() {
            return false;
        }
        match (&self.repr, &other.repr) {
            // SAFETY: shapes and fields agree.
            (Repr::Prime(a), Repr::Prime(b)) => unsafe {
                ffi::nmod_mat_equal(&a.inner, &b.inner) != 0
            },
            (Repr::Ext(a), Repr::Ext(b)) => unsafe {
                ffi::fq_nmod_mat_equal(&a.inner, &b.inner, self.field.ctx_ptr()) != 0
            },
            _ => false,
        }
    }

    // -- linear algebra ------------------------------------------------------

    /// Rank over GF(q). Defined for any shape.
    pub fn rank(&self) -> usize {
        match &self.repr {
            // SAFETY: `inner` is a live matrix over a prime modulus.
            Repr::Prime(a) => unsafe { ffi::nmod_mat_rank(&a.inner).max(0) as usize },
            Repr::Ext(a) => unsafe {
                ffi::fq_nmod_mat_rank(&a.inner, self.field.ctx_ptr()).max(0) as usize
            },
        }
    }

    /// Reduced row echelon form together with the transform `U` and the pivots.
    ///
    /// `U` is `rows × rows`, invertible, and satisfies `U · A = R`. It is
    /// obtained by reducing the augmented matrix `[A | I]`: every operation is a
    /// row operation on the whole of `[A | I]`, so the result is `[E·A | E]`,
    /// and the pivots that the elimination finds in the `I` block only ever
    /// combine rows whose `A` part is already zero — which is why the left block
    /// really is `rref(A)` and not something further reduced.
    ///
    /// # Errors
    ///
    /// `E-GFQ-012` when the augmented shape exceeds what FLINT can allocate.
    pub fn rref(&self) -> Result<Rref, FiniteFieldError> {
        let (m, n) = self.shape();
        let wide = n
            .checked_add(m)
            .ok_or(FiniteFieldError::DimensionTooLarge { rows: m, cols: n })?;
        check_shape(m, wide)?;

        let mut aug = Self::zeros_unchecked(&self.field, m, n + m);
        let mut scratch = NmodPoly::new(self.field.characteristic());
        let one = self.field.one();
        for i in 0..m {
            for j in 0..n {
                let v = self.get(i, j);
                aug.put(i, j, &v, &mut scratch);
            }
            aug.put(i, n + i, &one, &mut scratch);
        }

        let reduced = match &mut aug.repr {
            Repr::Prime(a) => {
                // SAFETY: in-place rref of a live matrix over a prime modulus.
                unsafe { ffi::nmod_mat_rref(&mut a.inner) };
                None
            }
            Repr::Ext(a) => {
                let mut dst = FqMat::new(m, n + m, &self.field);
                // SAFETY: `dst` has the same shape as `a`; the context is live.
                unsafe { ffi::fq_nmod_mat_rref(&mut dst.inner, &a.inner, self.field.ctx_ptr()) };
                Some(dst)
            }
        };
        if let Some(dst) = reduced {
            aug.repr = Repr::Ext(dst);
        }

        let mut r_mat = Self::zeros_unchecked(&self.field, m, n);
        let mut u_mat = Self::zeros_unchecked(&self.field, m, m);
        for i in 0..m {
            for j in 0..n {
                let v = aug.get(i, j);
                r_mat.put(i, j, &v, &mut scratch);
            }
            for j in 0..m {
                let v = aug.get(i, n + j);
                u_mat.put(i, j, &v, &mut scratch);
            }
        }

        let mut pivots = Vec::new();
        for i in 0..m {
            if let Some(j) = (0..n).find(|&j| !r_mat.get(i, j).is_zero()) {
                pivots.push(j);
            }
        }
        let rank = pivots.len();

        Ok(Rref {
            matrix: r_mat,
            transform: u_mat,
            rank,
            pivots,
        })
    }

    /// A basis for the right nullspace `{x : A·x = 0}`, as the **columns** of an
    /// `ncols × nullity` matrix.
    ///
    /// For coding theory the usual next step is `.transpose()`, which turns the
    /// kernel basis of a parity-check matrix `H` into a generator matrix `G`
    /// with `H · Gᵀ = 0`.
    ///
    /// `rank() + nullspace().ncols() == ncols()` always holds; a unit test and a
    /// proptest both assert it.
    ///
    /// # Cost, and the shape this refuses
    ///
    /// FLINT's `nmod_mat_nullspace` writes the basis into a caller-supplied
    /// matrix that must have `ncols` rows and at least `nullity` columns, and
    /// the nullity is not known until it has run. This wrapper therefore sizes
    /// the scratch at `ncols × ncols`, so the peak memory is quadratic in the
    /// **column** count regardless of how few rows there are. A 70 000-column
    /// GF(2) parity-check matrix is refused with `E-GFQ-012` naming a
    /// `70000 × 70000` shape — that is the scratch, not the input. Sizing it
    /// exactly would cost a second full elimination to learn the rank first,
    /// which for the wide matrices this module exists to serve saves nothing:
    /// there `nullity ≈ ncols` anyway.
    ///
    /// # Errors
    ///
    /// `E-GFQ-012` when `ncols × ncols` exceeds what FLINT can allocate.
    pub fn nullspace(&self) -> Result<Self, FiniteFieldError> {
        let (_, n) = self.shape();
        check_shape(n, n)?;
        if n == 0 {
            return Self::zeros(&self.field, 0, 0);
        }
        let mut basis = Self::zeros_unchecked(&self.field, n, n);
        let nullity = match (&self.repr, &mut basis.repr) {
            // SAFETY: `x` is n×n where n = a->c, which is the space FLINT
            // documents as sufficient for the basis.
            (Repr::Prime(a), Repr::Prime(x)) => unsafe {
                ffi::nmod_mat_nullspace(&mut x.inner, &a.inner)
            },
            (Repr::Ext(a), Repr::Ext(x)) => unsafe {
                ffi::fq_nmod_mat_nullspace(&mut x.inner, &a.inner, self.field.ctx_ptr())
            },
            _ => unreachable!("one matrix, one representation"),
        };
        let d = nullity.max(0) as usize;

        let mut out = Self::zeros_unchecked(&self.field, n, d);
        let mut scratch = NmodPoly::new(self.field.characteristic());
        for i in 0..n {
            for j in 0..d {
                let v = basis.get(i, j);
                out.put(i, j, &v, &mut scratch);
            }
        }
        Ok(out)
    }

    /// One solution `X` of `A · X = B`, for any shape of `A`.
    ///
    /// The full solution set is `X + N·y` for arbitrary `y`, where `N` is
    /// [`GfMatrix::nullspace`]. When `A` is square and non-singular the solution
    /// is unique and `N` is empty.
    ///
    /// # Errors
    ///
    /// `E-GFQ-006` for different fields; `E-GFQ-007` when `B` has a different
    /// row count from `A`; `E-GFQ-010` when no solution exists.
    pub fn solve(&self, rhs: &Self) -> Result<Self, FiniteFieldError> {
        self.same_field(rhs)?;
        if self.nrows() != rhs.nrows() {
            return Err(FiniteFieldError::DimensionMismatch {
                op: "solve",
                lhs: self.shape(),
                rhs: rhs.shape(),
            });
        }
        let mut out = Self::zeros_unchecked(&self.field, self.ncols(), rhs.ncols());
        let ok = match (&self.repr, &rhs.repr, &mut out.repr) {
            // SAFETY: `x` is (a->c) × (b->c), the shape FLINT requires.
            (Repr::Prime(a), Repr::Prime(b), Repr::Prime(x)) => unsafe {
                ffi::nmod_mat_can_solve(&mut x.inner, &a.inner, &b.inner)
            },
            (Repr::Ext(a), Repr::Ext(b), Repr::Ext(x)) => unsafe {
                ffi::fq_nmod_mat_can_solve(&mut x.inner, &a.inner, &b.inner, self.field.ctx_ptr())
            },
            _ => unreachable!("equal fields imply equal representations"),
        };
        if ok == 0 {
            return Err(FiniteFieldError::Inconsistent);
        }
        Ok(out)
    }

    /// The multiplicative inverse.
    ///
    /// # Errors
    ///
    /// `E-GFQ-008` for a non-square matrix, `E-GFQ-009` when it is singular.
    pub fn inverse(&self) -> Result<Self, FiniteFieldError> {
        let n = self.require_square("inverse")?;
        let mut out = Self::zeros_unchecked(&self.field, n, n);
        let ok = match (&self.repr, &mut out.repr) {
            // SAFETY: both matrices are n×n over the same field.
            (Repr::Prime(a), Repr::Prime(b)) => unsafe {
                ffi::nmod_mat_inv(&mut b.inner, &a.inner)
            },
            (Repr::Ext(a), Repr::Ext(b)) => unsafe {
                ffi::fq_nmod_mat_inv(&mut b.inner, &a.inner, self.field.ctx_ptr())
            },
            _ => unreachable!("one matrix, one representation"),
        };
        if ok == 0 {
            return Err(FiniteFieldError::Singular { shape: (n, n) });
        }
        Ok(out)
    }

    /// The determinant.
    ///
    /// Over an extension field FLINT exports no `fq_nmod_mat_det`, so this is
    /// read off the characteristic polynomial: `p(x) = det(xI − A)` gives
    /// `p(0) = (−1)ⁿ det A`. Correct, and `O(n⁴)` rather than `O(n³)` — a
    /// deliberate trade of speed for a result that is checked against a
    /// textbook value in the tests rather than open-coded here.
    ///
    /// # Errors
    ///
    /// `E-GFQ-008` for a non-square matrix.
    pub fn determinant(&self) -> Result<FieldElement, FiniteFieldError> {
        let n = self.require_square("determinant")?;
        match &self.repr {
            // SAFETY: square matrix over a prime modulus.
            Repr::Prime(a) => Ok(self.field.scalar(unsafe { ffi::nmod_mat_det(&a.inner) })),
            Repr::Ext(_) => {
                let cp = self.charpoly()?;
                let c0 = cp.first().cloned().unwrap_or_else(|| self.field.zero());
                Ok(if n % 2 == 0 {
                    c0
                } else {
                    self.negate_elem(&c0)
                })
            }
        }
    }

    /// Coefficients of the characteristic polynomial `det(xI − A)`, ascending,
    /// so the last entry is the leading `1`.
    ///
    /// # Errors
    ///
    /// `E-GFQ-008` for a non-square matrix.
    pub fn charpoly(&self) -> Result<Vec<FieldElement>, FiniteFieldError> {
        self.require_square("charpoly")?;
        match &self.repr {
            Repr::Prime(a) => {
                let mut poly = NmodPoly::new(self.field.characteristic());
                // SAFETY: `poly` is a live `nmod_poly_t` over the same modulus;
                // `a` is square (checked above).
                unsafe { ffi::nmod_mat_charpoly(poly.as_mut_ptr(), &a.inner) };
                Ok(poly
                    .to_coefficients()
                    .into_iter()
                    .map(|c| self.field.scalar(c))
                    .collect())
            }
            Repr::Ext(a) => {
                let ctx = self.field.ctx_ptr();
                let k = self.field.degree();
                let mut buf = ffi::FqNmodPolyBuf([0u8; 128]);
                let mut coeff = NmodPoly::new(self.field.characteristic());
                // SAFETY: `buf` is an over-sized aligned allocation for an
                // `fq_nmod_poly_t` (24 bytes on FLINT 3.5.0); `a` is square.
                let out = unsafe {
                    ffi::fq_nmod_poly_init(&mut buf, ctx);
                    ffi::fq_nmod_mat_charpoly(&mut buf, &a.inner, ctx);
                    let len = ffi::fq_nmod_poly_length(&buf, ctx).max(0);
                    let mut v = Vec::with_capacity(len as usize);
                    for t in 0..len {
                        ffi::fq_nmod_poly_get_coeff(coeff.as_mut_ptr(), &buf, t, ctx);
                        let coeffs: Vec<u64> = (0..k)
                            .map(|s| ffi::nmod_poly_get_coeff_ui(coeff.as_ptr(), s as ffi::slong))
                            .collect();
                        v.push(
                            self.field
                                .element(&coeffs)
                                .expect("a reduced element has degree < k"),
                        );
                    }
                    ffi::fq_nmod_poly_clear(&mut buf, ctx);
                    v
                };
                Ok(out)
            }
        }
    }

    fn negate_elem(&self, e: &FieldElement) -> FieldElement {
        let p = self.field.characteristic();
        let v: Vec<u64> = e
            .coefficients()
            .iter()
            .map(|&c| if c == 0 { 0 } else { p - c })
            .collect();
        self.field
            .element(&v)
            .expect("negation preserves the coefficient count")
    }
}

impl Clone for GfMatrix {
    fn clone(&self) -> Self {
        let (r, c) = self.shape();
        let mut out = Self::zeros_unchecked(&self.field, r, c);
        match (&self.repr, &mut out.repr) {
            // SAFETY: identical shapes and fields.
            (Repr::Prime(a), Repr::Prime(b)) => unsafe {
                ffi::nmod_mat_set(&mut b.inner, &a.inner)
            },
            (Repr::Ext(a), Repr::Ext(b)) => unsafe {
                ffi::fq_nmod_mat_set(&mut b.inner, &a.inner, self.field.ctx_ptr())
            },
            _ => unreachable!("one matrix, one representation"),
        }
        out
    }
}

impl fmt::Debug for GfMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, c) = self.shape();
        writeln!(f, "GfMatrix {r}x{c} over {}", self.field)?;
        for i in 0..r {
            let row: Vec<String> = (0..c).map(|j| self.field.render(&self.get(i, j))).collect();
            writeln!(f, "  [{}]", row.join(", "))?;
        }
        Ok(())
    }
}
