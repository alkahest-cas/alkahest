//! The field descriptor: GF(p) and GF(p^k), and the elements that live in them.
//!
//! A [`FiniteField`] is a cheap, cloneable handle (`Arc` inside) around either a
//! word-sized prime modulus or an owned FLINT `fq_nmod_ctx_t`. Matrices hold one
//! and compare it before every binary operation, so a GF(2) matrix and a GF(3)
//! matrix cannot be added by accident.

use std::ffi::CString;
use std::fmt;
use std::sync::Arc;

use super::error::FiniteFieldError;
use crate::flint::ffi;

/// Largest extension degree this module will build.
///
/// Not a mathematical limit — a guard. FLINT's Conway table thins out well
/// before this, and past it the fallback minimal-weight search is the dominant
/// cost of constructing the field rather than of using it.
pub const MAX_EXTENSION_DEGREE: i64 = 64;

// ---------------------------------------------------------------------------
// Owned fq_nmod_ctx_t
// ---------------------------------------------------------------------------

/// Owned `fq_nmod_ctx_t`, boxed so its address is stable across moves of the
/// `FiniteField` that holds it (FLINT matrices keep a borrowed pointer to it
/// for their whole lifetime).
struct FqCtx {
    buf: Box<ffi::FqNmodCtxBuf>,
}

// SAFETY: `fq_nmod_ctx_t` is written once by `fq_nmod_ctx_init*` and is
// thereafter read-only — every `fq_nmod_*` entry point takes it as
// `const fq_nmod_ctx_t`. It owns its own FLINT allocations and touches no
// thread-local state, so sharing a `&FqCtx` across threads is sound.
unsafe impl Send for FqCtx {}
unsafe impl Sync for FqCtx {}

impl FqCtx {
    /// Fill pattern written before handing the buffer to FLINT, so that
    /// [`FqCtx::untouched_tail`] can measure how much slack is left.
    const SENTINEL: u8 = 0xAA;

    fn blank() -> Box<ffi::FqNmodCtxBuf> {
        Box::new(ffi::FqNmodCtxBuf([Self::SENTINEL; 512]))
    }

    /// Conway polynomial for `(p, k)` when FLINT has one, otherwise FLINT's
    /// deterministic minimal-weight irreducible.
    fn conway(p: u64, k: i64, var: &CString) -> Self {
        let mut buf = Self::blank();
        // SAFETY: `buf` is a 512-byte aligned allocation; FLINT 3.5.0 writes
        // 160 bytes of `fq_nmod_ctx_struct` into it. `var` outlives the call
        // (FLINT copies the string).
        unsafe { ffi::fq_nmod_ctx_init_ui(&mut *buf, p, k, var.as_ptr()) };
        Self { buf }
    }

    /// Field defined by a caller-supplied irreducible polynomial over ℤ/pℤ.
    fn from_modulus(modulus: &ffi::NmodPolyStruct, var: &CString) -> Self {
        let mut buf = Self::blank();
        // SAFETY: as `conway`; `modulus` is a live `nmod_poly_struct` and FLINT
        // copies it into the context.
        unsafe { ffi::fq_nmod_ctx_init_modulus(&mut *buf, modulus, var.as_ptr()) };
        Self { buf }
    }

    fn as_ptr(&self) -> *const ffi::FqNmodCtxBuf {
        &*self.buf
    }

    /// How many trailing bytes of the buffer FLINT never wrote.
    ///
    /// Used by a unit test to assert the opaque buffer is still comfortably
    /// larger than the struct a future FLINT might grow into.
    #[cfg(test)]
    fn untouched_tail(&self) -> usize {
        self.buf
            .0
            .iter()
            .rev()
            .take_while(|&&b| b == Self::SENTINEL)
            .count()
    }
}

impl Drop for FqCtx {
    fn drop(&mut self) {
        // SAFETY: initialised by one of the `fq_nmod_ctx_init*` calls above.
        unsafe { ffi::fq_nmod_ctx_clear(&mut *self.buf) };
    }
}

// ---------------------------------------------------------------------------
// FiniteField
// ---------------------------------------------------------------------------

struct FieldInner {
    p: u64,
    k: usize,
    /// `None` for a prime field, where FLINT's `nmod_mat` is used directly.
    ctx: Option<FqCtx>,
    /// Coefficients of the defining polynomial, ascending, length `k + 1`.
    /// Empty for a prime field.
    modulus: Vec<u64>,
    /// Name of the generator, for rendering elements.
    var: String,
}

/// A finite field GF(q) with `q = p^k`, `p` prime and word-sized.
///
/// Cloning is `Arc`-cheap. Two fields are equal when they have the same
/// characteristic, the same degree, **and** the same defining polynomial —
/// GF(2³) built from `x³ + x + 1` and GF(2³) built from `x³ + x² + 1` are
/// isomorphic but not interchangeable, because the coordinates of an element
/// mean different things in each.
#[derive(Clone)]
pub struct FiniteField {
    inner: Arc<FieldInner>,
}

impl FiniteField {
    /// GF(p) for a prime `p` that fits in a machine word.
    ///
    /// # Errors
    ///
    /// `E-GFQ-001` when `p` is not prime (including `p < 2`).
    pub fn prime(p: u64) -> Result<Self, FiniteFieldError> {
        // SAFETY: `n_is_prime` is a pure function of a machine word.
        if p < 2 || unsafe { ffi::n_is_prime(p) } == 0 {
            return Err(FiniteFieldError::NotPrime {
                modulus: p.to_string(),
            });
        }
        Ok(Self {
            inner: Arc::new(FieldInner {
                p,
                k: 1,
                ctx: None,
                modulus: Vec::new(),
                var: "a".to_string(),
            }),
        })
    }

    /// GF(p) from a possibly-too-large characteristic.
    ///
    /// This is the entry point a front-end with arbitrary-precision integers
    /// should call: it separates "not prime" from "will not fit", which are
    /// different problems with different fixes.
    ///
    /// # Errors
    ///
    /// `E-GFQ-002` when `p` exceeds `u64::MAX`, otherwise as [`Self::prime`].
    pub fn prime_wide(p: u128) -> Result<Self, FiniteFieldError> {
        Self::prime_from_decimal(&p.to_string())
    }

    /// GF(p) from a characteristic that may not fit any Rust integer type.
    ///
    /// `decimal` is the modulus written in base ten. This is what a PyO3 or
    /// other arbitrary-precision front end should call: a 300-digit modulus is
    /// refused by `E-GFQ-002` **naming the value the caller passed**, rather
    /// than by an overflow in the conversion, which names nothing.
    ///
    /// # Errors
    ///
    /// `E-GFQ-002` when the value does not fit in a `u64`; `E-GFQ-001` when it
    /// is not prime; `E-GFQ-001` for a value that is not a non-negative
    /// decimal integer at all.
    pub fn prime_from_decimal(decimal: &str) -> Result<Self, FiniteFieldError> {
        let trimmed = decimal.trim();
        match trimmed.parse::<u64>() {
            Ok(p) => Self::prime(p),
            // Parsing fails for three different reasons and only one of them is
            // "too large": a negative value and a non-numeric string are both
            // "not a prime characteristic", not "will not fit".
            Err(_) if trimmed.starts_with('-') || !trimmed.bytes().all(|b| b.is_ascii_digit()) => {
                Err(FiniteFieldError::NotPrime {
                    modulus: trimmed.to_string(),
                })
            }
            Err(_) => Err(FiniteFieldError::ModulusTooLarge {
                modulus: trimmed.to_string(),
            }),
        }
    }

    /// GF(p^k), using FLINT's Conway polynomial for `(p, k)` when one is
    /// tabulated and its deterministic minimal-weight irreducible otherwise.
    ///
    /// `k == 1` returns the prime field, so callers do not have to special-case
    /// the degree-one extension.
    ///
    /// # Errors
    ///
    /// `E-GFQ-001` for a non-prime `p`, `E-GFQ-003` for a degree outside
    /// `1..=`[`MAX_EXTENSION_DEGREE`].
    pub fn extension(p: u64, k: u32) -> Result<Self, FiniteFieldError> {
        let kd = i64::from(k);
        if !(1..=MAX_EXTENSION_DEGREE).contains(&kd) {
            return Err(FiniteFieldError::DegreeOutOfRange { degree: kd });
        }
        let base = Self::prime(p)?;
        if kd == 1 {
            return Ok(base);
        }
        let var = CString::new("a").expect("literal has no NUL");
        let ctx = FqCtx::conway(p, kd, &var);
        let modulus = read_ctx_modulus(&ctx);
        Ok(Self {
            inner: Arc::new(FieldInner {
                p,
                k: k as usize,
                ctx: Some(ctx),
                modulus,
                var: "a".to_string(),
            }),
        })
    }

    /// GF(p^k) defined by a caller-supplied polynomial.
    ///
    /// `modulus` lists coefficients in **ascending** degree order, so
    /// `x³ + x + 1` over GF(2) is `[1, 1, 0, 1]`. Its degree fixes `k`.
    ///
    /// The polynomial is checked for irreducibility over GF(p) before the field
    /// is built. A reducible one would produce a ring with zero divisors that
    /// FLINT would happily do arithmetic in — the classic silent wrong answer.
    ///
    /// # Errors
    ///
    /// `E-GFQ-001` for a non-prime `p`; `E-GFQ-003` when the implied degree is
    /// out of range; `E-GFQ-004` when the polynomial is constant after
    /// reduction mod `p`, is not monic, or is reducible over GF(p).
    pub fn with_defining_polynomial(p: u64, modulus: &[u64]) -> Result<Self, FiniteFieldError> {
        let base = Self::prime(p)?;

        // Reduce, then trim trailing zeros so the degree is the true one.
        let mut coeffs: Vec<u64> = modulus.iter().map(|c| c % p).collect();
        while coeffs.last() == Some(&0) {
            coeffs.pop();
        }
        if coeffs.len() < 2 {
            return Err(FiniteFieldError::BadDefiningPolynomial {
                reason: format!(
                    "degree {} after reduction mod {p}; a defining polynomial must have degree >= 1",
                    coeffs.len() as i64 - 1
                ),
            });
        }
        let deg = (coeffs.len() - 1) as i64;
        if deg > MAX_EXTENSION_DEGREE {
            return Err(FiniteFieldError::DegreeOutOfRange { degree: deg });
        }
        if deg == 1 {
            // GF(p^1) — the quotient is the prime field itself. Accepting it
            // keeps `with_defining_polynomial(p, &[c, 1])` from being a special
            // case for the caller.
            return Ok(base);
        }

        // A non-monic modulus generates the same ideal, so normalising it would
        // be sound — but it would also mean `defining_polynomial()` returns
        // something the caller did not write. Refuse and name the one-line fix.
        let lead = *coeffs.last().expect("length >= 2 was checked");
        if lead != 1 {
            return Err(FiniteFieldError::BadDefiningPolynomial {
                reason: format!(
                    "leading coefficient is {lead}, not 1; scale the polynomial by the \
                     inverse of {lead} mod {p} to make it monic"
                ),
            });
        }

        let mut poly = NmodPoly::new(p);
        for (i, &c) in coeffs.iter().enumerate() {
            poly.set_coeff(i, c);
        }
        // SAFETY: `poly` is an initialised `nmod_poly_t`.
        if unsafe { ffi::nmod_poly_is_irreducible(poly.as_ptr()) } == 0 {
            return Err(FiniteFieldError::BadDefiningPolynomial {
                reason: format!(
                    "{} is reducible over GF({p}); the quotient ring would have zero divisors",
                    render_poly(&coeffs, "x")
                ),
            });
        }

        let var = CString::new("a").expect("literal has no NUL");
        let ctx = FqCtx::from_modulus(poly.as_ptr_ref(), &var);
        let stored = read_ctx_modulus(&ctx);
        Ok(Self {
            inner: Arc::new(FieldInner {
                p,
                k: deg as usize,
                ctx: Some(ctx),
                modulus: stored,
                var: "a".to_string(),
            }),
        })
    }

    /// The characteristic `p`.
    pub fn characteristic(&self) -> u64 {
        self.inner.p
    }

    /// The extension degree `k`, so that `q = p^k`. `1` for a prime field.
    pub fn degree(&self) -> usize {
        self.inner.k
    }

    /// The order `q = p^k`, or `None` when it overflows `u128`.
    pub fn order(&self) -> Option<u128> {
        u128::from(self.inner.p).checked_pow(self.inner.k as u32)
    }

    /// `true` when this is GF(p) rather than a proper extension.
    pub fn is_prime_field(&self) -> bool {
        self.inner.k == 1
    }

    /// Coefficients of the defining polynomial, ascending. Empty for GF(p).
    pub fn defining_polynomial(&self) -> &[u64] {
        &self.inner.modulus
    }

    pub(crate) fn ctx_ptr(&self) -> *const ffi::FqNmodCtxBuf {
        self.inner
            .ctx
            .as_ref()
            .expect("ctx_ptr called on a prime field")
            .as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn ctx_untouched_tail(&self) -> Option<usize> {
        self.inner.ctx.as_ref().map(FqCtx::untouched_tail)
    }

    /// Build an element from coefficients over the prime subfield, ascending in
    /// the polynomial basis `1, a, a², …`.
    ///
    /// Coefficients are reduced modulo `p` (that is what "mod p" means), but a
    /// vector *longer* than the degree is refused rather than reduced modulo the
    /// defining polynomial — see [`FiniteFieldError::MalformedElement`].
    ///
    /// # Errors
    ///
    /// `E-GFQ-005` when `coeffs.len() > k`.
    pub fn element(&self, coeffs: &[u64]) -> Result<FieldElement, FiniteFieldError> {
        if coeffs.len() > self.inner.k {
            return Err(FiniteFieldError::MalformedElement {
                got: coeffs.len(),
                degree: self.inner.k,
            });
        }
        let mut v: Vec<u64> = coeffs.iter().map(|c| c % self.inner.p).collect();
        while v.last() == Some(&0) {
            v.pop();
        }
        Ok(FieldElement { coeffs: v })
    }

    /// The element of the prime subfield represented by `x` (reduced mod `p`).
    pub fn scalar(&self, x: u64) -> FieldElement {
        let c = x % self.inner.p;
        FieldElement {
            coeffs: if c == 0 { Vec::new() } else { vec![c] },
        }
    }

    /// The additive identity.
    pub fn zero(&self) -> FieldElement {
        FieldElement { coeffs: Vec::new() }
    }

    /// The multiplicative identity.
    pub fn one(&self) -> FieldElement {
        FieldElement { coeffs: vec![1] }
    }

    /// The generator `a` of the extension, or `None` for a prime field.
    pub fn generator(&self) -> Option<FieldElement> {
        (self.inner.k > 1).then(|| FieldElement { coeffs: vec![0, 1] })
    }

    /// Render an element using this field's generator name.
    pub fn render(&self, e: &FieldElement) -> String {
        if self.inner.k == 1 {
            return e.coeffs.first().copied().unwrap_or(0).to_string();
        }
        render_poly(&e.coeffs, &self.inner.var)
    }
}

impl PartialEq for FiniteField {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
            || (self.inner.p == other.inner.p
                && self.inner.k == other.inner.k
                && self.inner.modulus == other.inner.modulus)
    }
}

impl Eq for FiniteField {}

impl fmt::Display for FiniteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.inner.k == 1 {
            write!(f, "GF({})", self.inner.p)
        } else {
            write!(f, "GF({}^{})", self.inner.p, self.inner.k)
        }
    }
}

impl fmt::Debug for FiniteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.inner.k == 1 {
            write!(f, "GF({})", self.inner.p)
        } else {
            write!(
                f,
                "GF({}^{}) [{}]",
                self.inner.p,
                self.inner.k,
                render_poly(&self.inner.modulus, "x")
            )
        }
    }
}

// ---------------------------------------------------------------------------
// FieldElement
// ---------------------------------------------------------------------------

/// An element of GF(p^k), as its coordinates over the prime subfield in the
/// polynomial basis `1, a, a², …`.
///
/// Stored in canonical form: reduced mod `p`, trailing zeros trimmed, so `==`
/// is mathematical equality *within a single field*. It deliberately does not
/// carry a [`FiniteField`] — a matrix of them would then pay a refcount per
/// entry — which is also why comparing elements of two different fields is
/// meaningless and why [`FiniteField::render`] takes the field explicitly.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldElement {
    coeffs: Vec<u64>,
}

impl FieldElement {
    /// Coordinates over the prime subfield, ascending, trailing zeros trimmed.
    pub fn coefficients(&self) -> &[u64] {
        &self.coeffs
    }

    /// Coordinates padded to exactly `k` entries.
    pub fn coefficients_padded(&self, k: usize) -> Vec<u64> {
        let mut v = self.coeffs.clone();
        v.resize(k, 0);
        v
    }

    /// `true` for the additive identity.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The element as a single integer when it lies in the prime subfield.
    pub fn as_u64(&self) -> Option<u64> {
        match self.coeffs.len() {
            0 => Some(0),
            1 => Some(self.coeffs[0]),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Small owned nmod_poly, used for defining polynomials and charpoly readback
// ---------------------------------------------------------------------------

/// Owned `nmod_poly_t`.
///
/// `crate::flint::nmod::FlintNmodPoly` exists but is `pub(crate)` to the `flint`
/// module and exposes neither a `*const` view nor `nmod_poly_is_irreducible`;
/// this is the same three lines with the two accessors this module needs.
pub(crate) struct NmodPoly {
    inner: ffi::NmodPolyStruct,
}

// SAFETY: owns its FLINT allocation, no thread-local state.
unsafe impl Send for NmodPoly {}
unsafe impl Sync for NmodPoly {}

impl NmodPoly {
    pub(crate) fn new(p: u64) -> Self {
        // SAFETY: `zeroed` is a valid starting point; `nmod_poly_init`
        // overwrites every field before any read.
        let mut inner: ffi::NmodPolyStruct = unsafe { std::mem::zeroed() };
        unsafe { ffi::nmod_poly_init(&mut inner, p) };
        Self { inner }
    }

    pub(crate) fn set_coeff(&mut self, i: usize, c: u64) {
        // SAFETY: `inner` is initialised; FLINT grows the coefficient array.
        unsafe { ffi::nmod_poly_set_coeff_ui(&mut self.inner, i as ffi::slong, c) };
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::NmodPolyStruct {
        &self.inner
    }

    pub(crate) fn as_ptr_ref(&self) -> &ffi::NmodPolyStruct {
        &self.inner
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::NmodPolyStruct {
        &mut self.inner
    }

    /// Degree, or `-1` for the zero polynomial.
    pub(crate) fn degree(&self) -> i64 {
        // SAFETY: `inner` is initialised.
        unsafe { ffi::nmod_poly_degree(&self.inner) }
    }

    pub(crate) fn coeff(&self, i: i64) -> u64 {
        // SAFETY: FLINT returns 0 past the end rather than reading out of bounds.
        unsafe { ffi::nmod_poly_get_coeff_ui(&self.inner, i) }
    }

    /// Coefficients ascending, length `degree + 1` (empty for the zero poly).
    pub(crate) fn to_coefficients(&self) -> Vec<u64> {
        let d = self.degree();
        if d < 0 {
            return Vec::new();
        }
        (0..=d).map(|i| self.coeff(i)).collect()
    }
}

impl Drop for NmodPoly {
    fn drop(&mut self) {
        // SAFETY: initialised in `new`.
        unsafe { ffi::nmod_poly_clear(&mut self.inner) };
    }
}

fn read_ctx_modulus(ctx: &FqCtx) -> Vec<u64> {
    // SAFETY: `fq_nmod_ctx_modulus` returns a borrowed pointer into the live
    // context; we only read coefficients from it.
    unsafe {
        let m = ffi::fq_nmod_ctx_modulus(ctx.as_ptr());
        let d = ffi::nmod_poly_degree(m);
        if d < 0 {
            return Vec::new();
        }
        (0..=d).map(|i| ffi::nmod_poly_get_coeff_ui(m, i)).collect()
    }
}

/// Render `c0 + c1·v + c2·v² + …` with zero terms dropped.
fn render_poly(coeffs: &[u64], v: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (i, &c) in coeffs.iter().enumerate().rev() {
        if c == 0 {
            continue;
        }
        parts.push(match (i, c) {
            (0, _) => c.to_string(),
            (1, 1) => v.to_string(),
            (1, _) => format!("{c}*{v}"),
            (_, 1) => format!("{v}^{i}"),
            _ => format!("{c}*{v}^{i}"),
        });
    }
    if parts.is_empty() {
        "0".to_string()
    } else {
        parts.join(" + ")
    }
}
