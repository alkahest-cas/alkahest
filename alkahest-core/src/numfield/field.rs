//! [`NumberField`] and [`NumberFieldElement`], on FLINT's `nf` / `nf_elem`.
//!
//! A [`NumberField`] is a cheap, cloneable handle (`Arc` inside) around an
//! owned FLINT `nf_t`. Elements hold one and compare it before every binary
//! operation, so an element of `ℚ(√2)` and an element of `ℚ(∛2)` cannot be
//! added by accident.
//!
//! # Where the FFI danger is
//!
//! `nf_elem_t` is a C union whose active arm depends on the degree of the
//! field — degree 1 and degree 2 are special-cased by Antic and do not have
//! the layout of degree 3. Nothing in this file knows or cares: the buffer is
//! opaque, and every operation on it passes the `nf_t` to FLINT, which picks
//! the arm. See the notes in [`crate::flint::ffi`].

#[cfg(test)]
use std::ffi::{CStr, CString};
use std::fmt;
use std::sync::Arc;

use rug::{Integer, Rational};

use super::error::NumberFieldError;
use crate::flint::ffi;
use crate::flint::poly::{FlintPoly, FlintPolyFactor};
use crate::flint::rational::FlintRational as Fq;
use crate::flint::FlintInteger;

/// Largest field degree this module will build.
///
/// Not a mathematical limit — a guard on memory and on the irreducibility
/// check, whose cost grows quickly with the degree and the coefficient size.
/// It is set high enough for the cyclotomic fields ring-LWE and lattice-based
/// KEMs use: `ℚ(ζ_2048)` has degree 1024.
pub const MAX_FIELD_DEGREE: usize = 1024;

// ---------------------------------------------------------------------------
// Thin RAII wrappers over the three FLINT types this module touches
// ---------------------------------------------------------------------------

/// Owned `fmpq_poly_t`, held as an opaque buffer.
struct FqPoly {
    buf: Box<ffi::FmpqPolyBuf>,
}

impl FqPoly {
    fn new() -> Self {
        let mut buf = Box::new(ffi::FmpqPolyBuf([0u8; 64]));
        // SAFETY: a 64-byte aligned allocation for a 32-byte struct.
        unsafe { ffi::fmpq_poly_init(&mut *buf) };
        Self { buf }
    }

    /// Exact copy of an integer polynomial (denominator 1).
    fn from_integer_poly(p: &FlintPoly) -> Self {
        let mut q = Self::new();
        // SAFETY: both operands are live and initialised.
        unsafe { ffi::fmpq_poly_set_fmpz_poly(&mut *q.buf, p.inner_ptr()) };
        q
    }

    fn set_coeff(&mut self, i: usize, c: &Fq) {
        // SAFETY: `self.buf` is initialised; FLINT grows the coefficient array.
        unsafe { ffi::fmpq_poly_set_coeff_fmpq(&mut *self.buf, i as ffi::slong, c.as_ptr()) };
    }

    fn as_ptr(&self) -> *const ffi::FmpqPolyBuf {
        &*self.buf
    }
}

impl Drop for FqPoly {
    fn drop(&mut self) {
        // SAFETY: initialised in `new`.
        unsafe { ffi::fmpq_poly_clear(&mut *self.buf) };
    }
}

// ---------------------------------------------------------------------------
// NumberField
// ---------------------------------------------------------------------------

/// Fill byte written into an `nf_t` / `nf_elem_t` buffer before FLINT sees it,
/// so a test can measure how much slack is left.
const SENTINEL: u8 = 0xAA;

struct NfInner {
    /// Owned `nf_t`, boxed so its address is stable across moves of the
    /// [`NumberField`] that holds it — every element keeps a borrowed pointer
    /// to it for its whole lifetime.
    nf: Box<ffi::NfBuf>,
    /// Canonical defining polynomial: primitive, integral, ascending,
    /// positive leading coefficient.
    defining: Vec<Integer>,
    degree: usize,
    /// Discriminant of `defining` — of the **polynomial**, not of the field.
    disc: Integer,
    var: String,
    /// `Some(n)` when this field was built by [`NumberField::cyclotomic`].
    cyclotomic_order: Option<u64>,
}

// SAFETY: an `nf_t` is written once by `nf_init` and is thereafter read-only.
// Every `nf_elem_*` entry point takes it as `const nf_t`, and the precomputed
// data it caches — the preinverse, the powers of the defining polynomial, and
// the trace vector — is built **eagerly** inside `nf_init`, not lazily on
// first use: disassembly of `nf_init` on FLINT 3.5.0 shows it calling
// `fmpz_preinvn_init`, `_fmpz_poly_powers_precompute` and
// `_fmpq_poly_powers_precompute` up front. So there is no interior mutation
// for two threads to race on. It owns its FLINT allocations and touches no
// thread-local state.
unsafe impl Send for NfInner {}
unsafe impl Sync for NfInner {}

impl Drop for NfInner {
    fn drop(&mut self) {
        // SAFETY: initialised by `nf_init` in every constructor.
        unsafe { ffi::nf_clear(&mut *self.nf) };
    }
}

/// An algebraic number field `ℚ[x]/(f)` for an irreducible `f ∈ ℚ[x]`.
///
/// Cloning is `Arc`-cheap. Two fields are equal when their canonical defining
/// polynomials agree; see [`NumberFieldError::FieldMismatch`] for what that
/// does and does not identify.
#[derive(Clone)]
pub struct NumberField {
    inner: Arc<NfInner>,
}

impl NumberField {
    /// `ℚ[x]/(f)` from the coefficients of `f`, **ascending** in degree.
    ///
    /// `[-2, 0, 1]` is `x² − 2`. The polynomial is normalised to a primitive
    /// integral one with a positive leading coefficient — which changes
    /// neither the field nor the generator, only the representative — and then
    /// **checked for irreducibility over ℚ**.
    ///
    /// # Errors
    ///
    /// `E-NUMF-001` for the zero polynomial, `E-NUMF-002` for a degree outside
    /// `1..=`[`MAX_FIELD_DEGREE`], `E-NUMF-003` when `f` is reducible.
    pub fn new(coeffs: &[Rational]) -> Result<Self, NumberFieldError> {
        Self::build(coeffs, None, true)
    }

    /// `ℚ[x]/(f)` from integer coefficients, ascending in degree.
    ///
    /// # Errors
    ///
    /// As [`NumberField::new`].
    pub fn from_integer_coeffs(coeffs: &[Integer]) -> Result<Self, NumberFieldError> {
        let rats: Vec<Rational> = coeffs.iter().map(Rational::from).collect();
        Self::new(&rats)
    }

    /// `ℚ[x]/(f)` from coefficients written as decimal strings, ascending.
    ///
    /// Each entry is `"p"` or `"p/q"`. This is the entry point an
    /// arbitrary-precision front end should call: a 300-digit coefficient
    /// survives it, where any fixed-width Rust integer type would not.
    ///
    /// # Errors
    ///
    /// `E-NUMF-004` for a coefficient that is not a rational number, otherwise
    /// as [`NumberField::new`].
    pub fn from_strings(coeffs: &[String]) -> Result<Self, NumberFieldError> {
        let mut rats = Vec::with_capacity(coeffs.len());
        for c in coeffs {
            rats.push(parse_rational(c)?);
        }
        Self::new(&rats)
    }

    /// The cyclotomic field `ℚ(ζ_n)`, defined by the cyclotomic polynomial
    /// `Φ_n`, of degree `φ(n)`.
    ///
    /// `Φ_n` is irreducible over `ℚ` — that is a theorem, not an observation —
    /// so this constructor skips the factorisation that [`NumberField::new`]
    /// runs. That is deliberate and is what makes `ℚ(ζ_2048)` constructible
    /// here: factoring `x^1024 + 1` to rediscover a known result would be the
    /// dominant cost of building the field.
    ///
    /// `n = 1` gives `Φ_1 = x − 1`, i.e. `ℚ` itself, and `n = 2` gives
    /// `Φ_2 = x + 1`, also `ℚ`. Both are degree 1 and both are legal.
    ///
    /// # Errors
    ///
    /// `E-NUMF-008` when `n == 0` or `φ(n) > `[`MAX_FIELD_DEGREE`].
    pub fn cyclotomic(n: u64) -> Result<Self, NumberFieldError> {
        if n == 0 {
            return Err(NumberFieldError::CyclotomicOrderOutOfRange {
                n,
                max: MAX_FIELD_DEGREE,
            });
        }
        let phi = euler_phi_u64(n);
        if phi > MAX_FIELD_DEGREE as u64 {
            return Err(NumberFieldError::CyclotomicOrderOutOfRange {
                n,
                max: MAX_FIELD_DEGREE,
            });
        }
        let coeffs: Vec<Rational> = cyclotomic_polynomial(n)
            .iter()
            .map(Rational::from)
            .collect();
        Self::build(&coeffs, Some(n), false)
    }

    fn build(
        coeffs: &[Rational],
        cyclotomic_order: Option<u64>,
        check_irreducible: bool,
    ) -> Result<Self, NumberFieldError> {
        let integral = canonical_integral_poly(coeffs)?;
        let degree = integral.len() - 1;
        if !(1..=MAX_FIELD_DEGREE).contains(&degree) {
            return Err(NumberFieldError::DegreeOutOfRange {
                degree: degree as i64,
                max: MAX_FIELD_DEGREE,
            });
        }

        let poly = FlintPoly::from_rug_coefficients(&integral);
        if check_irreducible {
            if let Some(factor) = proper_factor(&poly) {
                return Err(NumberFieldError::Reducible {
                    poly: render_integer_poly(&integral, "x"),
                    factor,
                });
            }
        }
        let disc = poly.discriminant().to_rug();

        let fq_poly = FqPoly::from_integer_poly(&poly);
        let mut nf = Box::new(ffi::NfBuf([SENTINEL; 512]));
        // SAFETY: a 512-byte aligned allocation for a 112-byte struct;
        // `fq_poly` is a live `fmpq_poly_t` of degree >= 1, which is what
        // `nf_init` requires, and FLINT copies it into the context.
        unsafe { ffi::nf_init(&mut *nf, fq_poly.as_ptr()) };

        Ok(Self {
            inner: Arc::new(NfInner {
                nf,
                defining: integral,
                degree,
                disc,
                var: "a".to_string(),
                cyclotomic_order,
            }),
        })
    }

    /// Rename the generator used when rendering elements. Purely cosmetic.
    #[must_use]
    pub fn with_variable(&self, var: &str) -> Self {
        let inner = &*self.inner;
        // The `nf_t` cannot be cloned cheaply, so a rename rebuilds the field.
        // It is the only operation that does, and it is not on any hot path.
        let coeffs: Vec<Rational> = inner.defining.iter().map(Rational::from).collect();
        let mut rebuilt = Self::build(&coeffs, inner.cyclotomic_order, false)
            .expect("a field that exists can be rebuilt from its own polynomial");
        Arc::get_mut(&mut rebuilt.inner)
            .expect("freshly built, not yet shared")
            .var = var.to_string();
        rebuilt
    }

    /// Degree `[K : ℚ]`.
    pub fn degree(&self) -> usize {
        self.inner.degree
    }

    /// The canonical defining polynomial: primitive, integral, ascending in
    /// degree, positive leading coefficient.
    pub fn defining_polynomial(&self) -> &[Integer] {
        &self.inner.defining
    }

    /// The defining polynomial rendered in the given variable.
    pub fn defining_polynomial_string(&self, var: &str) -> String {
        render_integer_poly(&self.inner.defining, var)
    }

    /// Discriminant of the **defining polynomial**.
    ///
    /// This is *not* the discriminant of the field. For a monic integral `f`
    /// with root `a`, `disc(f) = [O_K : ℤ[a]]² · d_K`, so the two agree only
    /// when `ℤ[a]` is the full ring of integers. `ℚ[x]/(x²−5)` has
    /// `disc(f) = 20` while `d_K = 5`, because `(1+√5)/2` is an algebraic
    /// integer that `ℤ[√5]` misses. Computing `d_K` needs a maximal-order
    /// algorithm (Round 2 / Zassenhaus), which this module does not implement
    /// and does not pretend to; see the module docs.
    pub fn polynomial_discriminant(&self) -> Integer {
        self.inner.disc.clone()
    }

    /// `Some(n)` when this field was built as `ℚ(ζ_n)`.
    pub fn cyclotomic_order(&self) -> Option<u64> {
        self.inner.cyclotomic_order
    }

    /// Name of the generator, as used by [`NumberFieldElement`]'s `Display`.
    pub fn variable(&self) -> &str {
        &self.inner.var
    }

    /// The generator `a`, a root of the defining polynomial.
    pub fn generator(&self) -> NumberFieldElement {
        let mut e = NumberFieldElement::blank(self);
        // SAFETY: `e.buf` was initialised against this same `nf`.
        unsafe { ffi::nf_elem_gen(e.as_mut_ptr(), self.nf_ptr()) };
        e
    }

    /// The additive identity.
    pub fn zero(&self) -> NumberFieldElement {
        let mut e = NumberFieldElement::blank(self);
        // SAFETY: as `generator`.
        unsafe { ffi::nf_elem_zero(e.as_mut_ptr(), self.nf_ptr()) };
        e
    }

    /// The multiplicative identity.
    pub fn one(&self) -> NumberFieldElement {
        let mut e = NumberFieldElement::blank(self);
        // SAFETY: as `generator`.
        unsafe { ffi::nf_elem_one(e.as_mut_ptr(), self.nf_ptr()) };
        e
    }

    /// The image of a rational number in the field.
    pub fn rational(&self, r: &Rational) -> NumberFieldElement {
        self.element(std::slice::from_ref(r))
            .expect("one coefficient always fits a field of degree >= 1")
    }

    /// An element from its coordinates in the power basis `1, a, a², …`,
    /// **ascending**, with at most [`NumberField::degree`] of them.
    ///
    /// # Errors
    ///
    /// `E-NUMF-007` when more coordinates are supplied than the degree. They
    /// are not silently reduced modulo the defining polynomial: a caller who
    /// passes four coordinates to a cubic field has made a mistake, and
    /// reducing would hide it.
    pub fn element(&self, coeffs: &[Rational]) -> Result<NumberFieldElement, NumberFieldError> {
        if coeffs.len() > self.degree() {
            return Err(NumberFieldError::CoefficientCountMismatch {
                got: coeffs.len(),
                degree: self.degree(),
            });
        }
        let mut poly = FqPoly::new();
        for (i, c) in coeffs.iter().enumerate() {
            poly.set_coeff(i, &Fq::from_rug(c));
        }
        let mut e = NumberFieldElement::blank(self);
        // SAFETY: `poly` has degree < the degree of the field, which is what
        // `nf_elem_set_fmpq_poly` is defined for, and both are live.
        unsafe { ffi::nf_elem_set_fmpq_poly(e.as_mut_ptr(), poly.as_ptr(), self.nf_ptr()) };
        Ok(e)
    }

    /// As [`NumberField::element`], with coordinates as decimal strings.
    ///
    /// # Errors
    ///
    /// `E-NUMF-004` for a coordinate that is not a rational number, otherwise
    /// as [`NumberField::element`].
    pub fn element_from_strings(
        &self,
        coeffs: &[String],
    ) -> Result<NumberFieldElement, NumberFieldError> {
        let mut rats = Vec::with_capacity(coeffs.len());
        for c in coeffs {
            rats.push(parse_rational(c)?);
        }
        self.element(&rats)
    }

    fn nf_ptr(&self) -> *const ffi::NfBuf {
        &*self.inner.nf
    }

    /// How many trailing bytes of the `nf_t` buffer FLINT never wrote.
    #[cfg(test)]
    pub(crate) fn untouched_tail(&self) -> usize {
        self.inner
            .nf
            .0
            .iter()
            .rev()
            .take_while(|&&b| b == SENTINEL)
            .count()
    }
}

impl PartialEq for NumberField {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || self.inner.defining == other.inner.defining
    }
}
impl Eq for NumberField {}

impl fmt::Display for NumberField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Q[x]/({})",
            render_integer_poly(&self.inner.defining, "x")
        )
    }
}

impl fmt::Debug for NumberField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NumberField({}, degree {})", self, self.inner.degree)
    }
}

// ---------------------------------------------------------------------------
// NumberFieldElement
// ---------------------------------------------------------------------------

/// An element of a [`NumberField`], held in the power basis `1, a, a², …`.
pub struct NumberFieldElement {
    field: NumberField,
    buf: Box<ffi::NfElemBuf>,
}

// SAFETY: an `nf_elem_t` owns its `fmpz`/`fmpq_poly` allocations and shares
// nothing but the read-only `nf_t`, which is itself `Sync`.
unsafe impl Send for NumberFieldElement {}
unsafe impl Sync for NumberFieldElement {}

impl NumberFieldElement {
    fn blank(field: &NumberField) -> Self {
        let mut buf = Box::new(ffi::NfElemBuf([SENTINEL; 128]));
        // SAFETY: a 128-byte aligned allocation for a struct of at most 32
        // bytes. `nf_elem_init` reads the field's degree out of the `nf_t` and
        // initialises whichever union arm goes with it.
        unsafe { ffi::nf_elem_init(&mut *buf, field.nf_ptr()) };
        Self {
            field: field.clone(),
            buf,
        }
    }

    fn as_ptr(&self) -> *const ffi::NfElemBuf {
        &*self.buf
    }

    fn as_mut_ptr(&mut self) -> *mut ffi::NfElemBuf {
        &mut *self.buf
    }

    /// The field this element lives in.
    pub fn field(&self) -> &NumberField {
        &self.field
    }

    fn same_field(&self, other: &Self) -> Result<(), NumberFieldError> {
        if self.field == other.field {
            Ok(())
        } else {
            Err(NumberFieldError::FieldMismatch {
                lhs: self.field.defining_polynomial_string("x"),
                rhs: other.field.defining_polynomial_string("x"),
            })
        }
    }

    /// Coordinates in the power basis `1, a, a², …`, ascending, always exactly
    /// `degree` of them.
    pub fn coefficients(&self) -> Vec<Rational> {
        (0..self.field.degree())
            .map(|i| {
                let mut c = Fq::new();
                // SAFETY: `i < degree`, which is the range
                // `nf_elem_get_coeff_fmpq` is defined on; it also receives the
                // `nf_t` and so reads the right union arm.
                unsafe {
                    ffi::nf_elem_get_coeff_fmpq(
                        c.as_mut_ptr(),
                        self.as_ptr(),
                        i as ffi::slong,
                        self.field.nf_ptr(),
                    );
                }
                c.to_rug()
            })
            .collect()
    }

    /// `true` for the zero element.
    pub fn is_zero(&self) -> bool {
        // SAFETY: live element, live field.
        unsafe { ffi::nf_elem_is_zero(self.as_ptr(), self.field.nf_ptr()) != 0 }
    }

    /// `true` for the multiplicative identity.
    pub fn is_one(&self) -> bool {
        // SAFETY: live element, live field.
        unsafe { ffi::nf_elem_is_one(self.as_ptr(), self.field.nf_ptr()) != 0 }
    }

    /// `self + other`.
    ///
    /// # Errors
    ///
    /// `E-NUMF-006` when the operands are from different fields.
    pub fn add(&self, other: &Self) -> Result<Self, NumberFieldError> {
        self.binop(other, ffi::nf_elem_add)
    }

    /// `self − other`.
    ///
    /// # Errors
    ///
    /// `E-NUMF-006` when the operands are from different fields.
    pub fn sub(&self, other: &Self) -> Result<Self, NumberFieldError> {
        self.binop(other, ffi::nf_elem_sub)
    }

    /// `self · other`.
    ///
    /// # Errors
    ///
    /// `E-NUMF-006` when the operands are from different fields.
    pub fn mul(&self, other: &Self) -> Result<Self, NumberFieldError> {
        self.binop(other, ffi::nf_elem_mul)
    }

    /// `self / other`.
    ///
    /// # Errors
    ///
    /// `E-NUMF-005` when `other` is zero, `E-NUMF-006` when the operands are
    /// from different fields.
    pub fn div(&self, other: &Self) -> Result<Self, NumberFieldError> {
        self.same_field(other)?;
        if other.is_zero() {
            return Err(NumberFieldError::NotInvertible);
        }
        self.binop(other, ffi::nf_elem_div)
    }

    fn binop(
        &self,
        other: &Self,
        f: unsafe extern "C" fn(
            *mut ffi::NfElemBuf,
            *const ffi::NfElemBuf,
            *const ffi::NfElemBuf,
            *const ffi::NfBuf,
        ),
    ) -> Result<Self, NumberFieldError> {
        self.same_field(other)?;
        let mut out = Self::blank(&self.field);
        // SAFETY: three live elements over one live field; `out` is distinct
        // from both inputs, so no aliasing rule of FLINT's is at stake.
        unsafe {
            f(
                out.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                self.field.nf_ptr(),
            )
        };
        Ok(out)
    }

    /// `−self`.
    #[must_use]
    pub fn neg(&self) -> Self {
        let mut out = Self::blank(&self.field);
        // SAFETY: live element, live field, distinct destination.
        unsafe { ffi::nf_elem_neg(out.as_mut_ptr(), self.as_ptr(), self.field.nf_ptr()) };
        out
    }

    /// `self⁻¹`.
    ///
    /// # Errors
    ///
    /// `E-NUMF-005` for zero — the only non-invertible element of a field, and
    /// the reason the defining polynomial is checked for irreducibility.
    pub fn inverse(&self) -> Result<Self, NumberFieldError> {
        if self.is_zero() {
            return Err(NumberFieldError::NotInvertible);
        }
        let mut out = Self::blank(&self.field);
        // SAFETY: non-zero element over a genuine field, so the inversion
        // cannot divide by zero; distinct destination.
        unsafe { ffi::nf_elem_inv(out.as_mut_ptr(), self.as_ptr(), self.field.nf_ptr()) };
        Ok(out)
    }

    /// `self^exp` for a non-negative exponent. `0^0` is 1.
    #[must_use]
    pub fn pow(&self, exp: u64) -> Self {
        let mut out = Self::blank(&self.field);
        // SAFETY: live element, live field, distinct destination.
        unsafe { ffi::nf_elem_pow(out.as_mut_ptr(), self.as_ptr(), exp, self.field.nf_ptr()) };
        out
    }

    /// The field norm `N_{K/ℚ}(self)` — the product of the conjugates, and the
    /// determinant of multiplication-by-`self` as a ℚ-linear map.
    pub fn norm(&self) -> Rational {
        let mut out = Fq::new();
        // SAFETY: live element, live field.
        unsafe { ffi::nf_elem_norm(out.as_mut_ptr(), self.as_ptr(), self.field.nf_ptr()) };
        out.to_rug()
    }

    /// The field trace `Tr_{K/ℚ}(self)` — the sum of the conjugates.
    pub fn trace(&self) -> Rational {
        let mut out = Fq::new();
        // SAFETY: live element, live field.
        unsafe { ffi::nf_elem_trace(out.as_mut_ptr(), self.as_ptr(), self.field.nf_ptr()) };
        out.to_rug()
    }

    /// The monic minimal polynomial of this element over `ℚ`, ascending in
    /// degree, with a trailing `1`.
    ///
    /// Its degree divides `[K : ℚ]` and equals it exactly when the element
    /// generates the whole field; for a rational element it is `x − c`.
    ///
    /// Computed by finding the first linear dependency among
    /// `1, a, a², …, a^d` over `ℚ` — exact rational elimination, no rounding
    /// and no factorisation, so the answer is the minimal polynomial rather
    /// than some multiple of it.
    pub fn minimal_polynomial(&self) -> Vec<Rational> {
        let d = self.field.degree();
        // `basis` holds, for each pivot found so far, the eliminated vector and
        // the combination of powers that produced it.
        let mut basis: Vec<(usize, Vec<Rational>, Vec<Rational>)> = Vec::new();
        let mut power = self.field.one();

        for k in 0..=d {
            let mut w = power.coefficients();
            let mut combo = vec![Rational::new(); k + 1];
            combo[k] = Rational::from(1);

            for (pivot, bv, bc) in &basis {
                let factor = w[*pivot].clone();
                if factor == 0 {
                    continue;
                }
                for (wi, bi) in w.iter_mut().zip(bv.iter()) {
                    *wi -= Rational::from(&factor * bi);
                }
                for (ci, bi) in combo.iter_mut().zip(bc.iter()) {
                    *ci -= Rational::from(&factor * bi);
                }
            }

            match w.iter().position(|c| *c != 0) {
                None => {
                    // combo[k] is still 1: every basis entry has zero in that
                    // slot, so the dependency is monic of degree k.
                    combo.truncate(k + 1);
                    return combo;
                }
                Some(pivot) => {
                    let lead = w[pivot].clone();
                    for wi in w.iter_mut() {
                        *wi /= &lead;
                    }
                    for ci in combo.iter_mut() {
                        *ci /= &lead;
                    }
                    combo.resize(d + 1, Rational::new());
                    basis.push((pivot, w, combo));
                }
            }

            if k < d {
                power = power.mul(self).expect("same field by construction");
            }
        }
        unreachable!("the powers 1..a^d of a degree-d element are always dependent")
    }

    /// FLINT's own rendering of the element, for cross-checking this module's.
    #[cfg(test)]
    pub(crate) fn flint_string(&self) -> String {
        let var = CString::new(self.field.variable()).expect("variable has no NUL");
        // SAFETY: `nf_elem_get_str_pretty` allocates a NUL-terminated string
        // the caller owns; `var` outlives the call.
        unsafe {
            let raw = ffi::nf_elem_get_str_pretty(self.as_ptr(), var.as_ptr(), self.field.nf_ptr());
            let owned = CStr::from_ptr(raw).to_string_lossy().into_owned();
            ffi::flint_free(raw.cast());
            owned
        }
    }

    /// How many trailing bytes of the `nf_elem_t` buffer FLINT never wrote.
    #[cfg(test)]
    pub(crate) fn untouched_tail(&self) -> usize {
        self.buf
            .0
            .iter()
            .rev()
            .take_while(|&&b| b == SENTINEL)
            .count()
    }
}

impl Clone for NumberFieldElement {
    fn clone(&self) -> Self {
        let mut out = Self::blank(&self.field);
        // SAFETY: live source and destination over one live field.
        unsafe { ffi::nf_elem_set(out.as_mut_ptr(), self.as_ptr(), self.field.nf_ptr()) };
        out
    }
}

impl Drop for NumberFieldElement {
    fn drop(&mut self) {
        // SAFETY: initialised by `nf_elem_init` against this same field, which
        // is still alive because the element holds an `Arc` to it.
        unsafe { ffi::nf_elem_clear(&mut *self.buf, self.field.nf_ptr()) };
    }
}

impl PartialEq for NumberFieldElement {
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: both live; the comparison is meaningless across fields, so
        // it is refused rather than answered.
        self.field == other.field
            && unsafe {
                ffi::nf_elem_equal(self.as_ptr(), other.as_ptr(), self.field.nf_ptr()) != 0
            }
    }
}
impl Eq for NumberFieldElement {}

impl fmt::Display for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            render_rational_poly(&self.coefficients(), self.field.variable())
        )
    }
}

impl fmt::Debug for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NumberFieldElement({} in {})", self, self.field)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// The cyclotomic polynomial `Φ_n`, ascending in degree (`fmpz_poly_cyclotomic`).
///
/// `Φ_1 = x − 1`, `Φ_2 = x + 1`, `Φ_6 = x² − x + 1`. Its degree is `φ(n)`.
/// `n = 0` has no cyclotomic polynomial and returns the constant `1`.
pub fn cyclotomic_polynomial(n: u64) -> Vec<Integer> {
    if n == 0 {
        return vec![Integer::from(1)];
    }
    let p = FlintPoly::cyclotomic(n);
    let len = p.length();
    (0..len).map(|i| p.get_coeff_flint(i).to_rug()).collect()
}

/// Euler's totient `φ(n)` for a machine-word `n`, via FLINT's `fmpz_euler_phi`.
fn euler_phi_u64(n: u64) -> u64 {
    let z = FlintInteger::from_rug(&Integer::from(n));
    let mut out = FlintInteger::new();
    // SAFETY: both operands are live `fmpz`.
    unsafe { ffi::fmpz_euler_phi(out.inner_mut_ptr(), z.inner_ptr()) };
    out.to_rug().to_u64().unwrap_or(u64::MAX)
}

fn parse_rational(text: &str) -> Result<Rational, NumberFieldError> {
    Rational::from_str_radix(text.trim(), 10).map_err(|_| NumberFieldError::MalformedCoefficient {
        text: text.to_string(),
    })
}

/// Normalise rational coefficients to a primitive integral polynomial with a
/// positive leading coefficient. Trailing zeros are stripped first.
fn canonical_integral_poly(coeffs: &[Rational]) -> Result<Vec<Integer>, NumberFieldError> {
    let last = coeffs.iter().rposition(|c| *c != 0);
    let Some(last) = last else {
        return Err(NumberFieldError::EmptyDefiningPolynomial);
    };
    let trimmed = &coeffs[..=last];

    let mut lcm = Integer::from(1);
    for c in trimmed {
        lcm = lcm.lcm(c.denom());
    }
    let mut ints: Vec<Integer> = trimmed
        .iter()
        .map(|c| {
            (Rational::from(c) * Rational::from(&lcm))
                .into_numer_denom()
                .0
        })
        .collect();

    let mut content = Integer::new();
    for c in &ints {
        content = content.gcd(c);
    }
    if content == 0 {
        return Err(NumberFieldError::EmptyDefiningPolynomial);
    }
    let negate = ints[ints.len() - 1] < 0;
    for c in ints.iter_mut() {
        *c /= &content;
        if negate {
            *c *= -1;
        }
    }
    Ok(ints)
}

/// One proper factor of `poly` over `ℤ`, rendered, or `None` when it is
/// irreducible. Degree-1 polynomials are irreducible by definition.
fn proper_factor(poly: &FlintPoly) -> Option<String> {
    if poly.degree() <= 1 {
        return None;
    }
    let mut fac = FlintPolyFactor::new();
    fac.factor(poly);
    if fac.len() == 1 && fac.exp_at(0) == 1 {
        return None;
    }
    let first = fac.poly_at(0);
    let coeffs: Vec<Integer> = (0..first.length())
        .map(|i| first.get_coeff_flint(i).to_rug())
        .collect();
    Some(render_integer_poly(&coeffs, "x"))
}

fn render_integer_poly(coeffs: &[Integer], var: &str) -> String {
    let rats: Vec<Rational> = coeffs.iter().map(Rational::from).collect();
    render_rational_poly(&rats, var)
}

/// Render `c₀ + c₁·v + c₂·v² + …` descending, skipping zero terms and the
/// redundant `1*`.
fn render_rational_poly(coeffs: &[Rational], var: &str) -> String {
    let mut out = String::new();
    for (i, c) in coeffs.iter().enumerate().rev() {
        if *c == 0 {
            continue;
        }
        let negative = *c < 0;
        let mag = Rational::from(c.abs_ref());
        if out.is_empty() {
            if negative {
                out.push('-');
            }
        } else if negative {
            out.push_str(" - ");
        } else {
            out.push_str(" + ");
        }
        let unit = mag == 1;
        if !unit || i == 0 {
            out.push_str(&mag.to_string());
            if i > 0 {
                out.push('*');
            }
        }
        if i == 1 {
            out.push_str(var);
        } else if i > 1 {
            out.push_str(var);
            out.push('^');
            out.push_str(&i.to_string());
        }
    }
    if out.is_empty() {
        out.push('0');
    }
    out
}
