//! Drop-safe wrappers for FLINT's `nmod_poly_t` and `nmod_poly_factor_t`.
//!
//! Both types are owned: `Drop` calls the matching FLINT clear function so
//! callers never need to pair `init` / `clear` manually.

use super::ffi;

// ---------------------------------------------------------------------------
// FlintNmodPoly — univariate polynomial over ℤ/pℤ
// ---------------------------------------------------------------------------

/// Owned `nmod_poly_t` — univariate polynomial over a word-sized prime field.
///
/// `Drop` calls `nmod_poly_clear`; no manual cleanup is required.
pub struct FlintNmodPoly {
    inner: ffi::NmodPolyStruct,
}

// SAFETY: `nmod_poly_struct` owns its coefficient buffer via FLINT's allocator;
// no thread-local state is involved.
unsafe impl Send for FlintNmodPoly {}
unsafe impl Sync for FlintNmodPoly {}

#[allow(dead_code)]
impl FlintNmodPoly {
    /// Construct a zero polynomial over `ℤ/modulus·ℤ`.
    pub fn new(modulus: u64) -> Self {
        // SAFETY: zeroed() gives a valid starting point; nmod_poly_init
        // overwrites every field before any read.
        let mut inner: ffi::NmodPolyStruct = unsafe { std::mem::zeroed() };
        unsafe { ffi::nmod_poly_init(&mut inner, modulus) };
        Self { inner }
    }

    /// Set the coefficient of `x^i` to `c` (reduced mod `p` internally by FLINT).
    pub fn set_coeff(&mut self, i: usize, c: u64) {
        unsafe { ffi::nmod_poly_set_coeff_ui(&mut self.inner, i as ffi::slong, c) };
    }

    /// Degree of the polynomial; returns `-1` for the zero polynomial.
    pub fn degree(&self) -> i64 {
        unsafe { ffi::nmod_poly_degree(&self.inner) }
    }

    /// Coefficient of `x^j` as a `u64`.
    pub fn get_coeff(&self, j: i64) -> u64 {
        unsafe { ffi::nmod_poly_get_coeff_ui(&self.inner, j) }
    }

    /// The modulus this polynomial lives over.
    pub fn modulus(&self) -> u64 {
        self.inner.mod_.n
    }

    pub(super) fn as_ptr(&self) -> *const ffi::NmodPolyStruct {
        &self.inner
    }

    pub(super) fn as_mut_ptr(&mut self) -> *mut ffi::NmodPolyStruct {
        &mut self.inner
    }
}

impl Drop for FlintNmodPoly {
    fn drop(&mut self) {
        // SAFETY: `self.inner` was initialised by `nmod_poly_init` in `new`.
        unsafe { ffi::nmod_poly_clear(&mut self.inner) };
    }
}

// ---------------------------------------------------------------------------
// FlintNmodPolyFactor — factorisation container for nmod_poly
// ---------------------------------------------------------------------------

/// Owned `nmod_poly_factor_t` — factorisation result for a univariate polynomial
/// over a word-sized prime field.
///
/// `Drop` calls `nmod_poly_factor_clear`; no manual cleanup required.
pub struct FlintNmodPolyFactor {
    inner: ffi::NmodPolyFactorStruct,
}

// SAFETY: FLINT factor structs own their memory; no thread-local state.
unsafe impl Send for FlintNmodPolyFactor {}
unsafe impl Sync for FlintNmodPolyFactor {}

impl FlintNmodPolyFactor {
    /// Initialise an empty factor container.
    pub fn new() -> Self {
        let mut inner = std::mem::MaybeUninit::<ffi::NmodPolyFactorStruct>::uninit();
        unsafe { ffi::nmod_poly_factor_init(inner.as_mut_ptr()) };
        // SAFETY: `nmod_poly_factor_init` fully initialises the struct.
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    /// Factor `poly` into this container (Berlekamp / Cantor–Zassenhaus via FLINT).
    /// Returns the leading coefficient as a `u64`.
    pub fn factor(&mut self, poly: &FlintNmodPoly) -> u64 {
        unsafe { ffi::nmod_poly_factor(&mut self.inner, poly.as_ptr()) }
    }

    /// Number of distinct irreducible factors.
    pub fn len(&self) -> usize {
        self.inner.num.max(0) as usize
    }

    /// Exponent (multiplicity) of the `i`-th factor.
    pub fn exp_at(&self, i: usize) -> u32 {
        debug_assert!(i < self.len());
        // SAFETY: `i < num` so the pointer arithmetic is in bounds.
        unsafe { *self.inner.exp.add(i) as u32 }
    }

    /// Copy the `i`-th irreducible factor into a new [`FlintNmodPoly`].
    pub fn poly_at(&self, modulus: u64, i: usize) -> FlintNmodPoly {
        debug_assert!(i < self.len());
        let mut z = FlintNmodPoly::new(modulus);
        unsafe {
            // FLINT 2.x: nmod_poly_factor_get_nmod_poly takes *mut fac
            // FLINT 3.x: nmod_poly_factor_get_poly takes *const fac
            #[cfg(not(flint3))]
            ffi::nmod_poly_factor_get_nmod_poly(
                z.as_mut_ptr(),
                // The FLINT 2.x signature takes *mut, but the function only reads it.
                &self.inner as *const ffi::NmodPolyFactorStruct as *mut _,
                i as ffi::slong,
            );
            #[cfg(flint3)]
            ffi::nmod_poly_factor_get_poly(z.as_mut_ptr(), &self.inner, i as ffi::slong);
        }
        z
    }
}

impl Default for FlintNmodPolyFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FlintNmodPolyFactor {
    fn drop(&mut self) {
        // SAFETY: `self.inner` was initialised by `nmod_poly_factor_init` in `new`.
        unsafe { ffi::nmod_poly_factor_clear(&mut self.inner) };
    }
}
