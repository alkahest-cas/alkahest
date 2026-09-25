//! Safe wrapper over FLINT's `fmpq_t` — an exact rational number.
//!
//! Only two things happen to an `fmpq` in this crate: FLINT writes one (a
//! Bernoulli number, a harmonic number, a field norm) and we read it out as a
//! [`rug::Rational`]. Both directions go through decimal strings and FLINT's
//! own `fmpq_get_str`, so neither depends on this crate's view of the struct.

use std::ffi::{CStr, CString};

use rug::Rational;

use super::ffi;

/// Owned `fmpq_t`. `Drop` calls `fmpq_clear`.
pub(crate) struct FlintRational {
    inner: ffi::Fmpq,
}

// SAFETY: an `fmpq` owns its two `fmpz`, which own their GMP memory. No
// shared or thread-local state.
unsafe impl Send for FlintRational {}
unsafe impl Sync for FlintRational {}

impl FlintRational {
    pub(crate) fn new() -> Self {
        let mut inner = ffi::Fmpq { num: 0, den: 0 };
        // SAFETY: `fmpq_init` writes both fields of a live `fmpq`.
        unsafe { ffi::fmpq_init(&mut inner) };
        Self { inner }
    }

    /// Build from a [`rug::Rational`] by a decimal round-trip, then let FLINT
    /// restore its own canonical form.
    pub(crate) fn from_rug(r: &Rational) -> Self {
        let mut q = Self::new();
        let num = CString::new(r.numer().to_string()).expect("a decimal has no NUL");
        let den = CString::new(r.denom().to_string()).expect("a decimal has no NUL");
        // SAFETY: `q.inner` is initialised and both strings are NUL-terminated
        // and outlive the calls. `fmpq_canonicalise` re-establishes the
        // lowest-terms invariant every `fmpq_*` operation assumes.
        unsafe {
            ffi::fmpz_set_str(&mut q.inner.num, num.as_ptr(), 10);
            ffi::fmpz_set_str(&mut q.inner.den, den.as_ptr(), 10);
            ffi::fmpq_canonicalise(&mut q.inner);
        }
        q
    }

    /// Read back through FLINT's own printer, so the conversion never depends
    /// on this crate's view of the struct layout.
    pub(crate) fn to_rug(&self) -> Rational {
        // SAFETY: `fmpq_get_str(NULL, ...)` allocates a NUL-terminated string
        // the caller owns and frees with `flint_free`. FLINT prints `"p"` or
        // `"p/q"` in base 10, both of which `Rational` parses.
        let s = unsafe {
            let raw = ffi::fmpq_get_str(std::ptr::null_mut(), 10, &self.inner);
            let owned = CStr::from_ptr(raw).to_string_lossy().into_owned();
            ffi::flint_free(raw.cast());
            owned
        };
        Rational::from_str_radix(&s, 10).expect("FLINT printed a rational")
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::Fmpq {
        &self.inner
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::Fmpq {
        &mut self.inner
    }
}

impl Drop for FlintRational {
    fn drop(&mut self) {
        // SAFETY: initialised in `new`.
        unsafe { ffi::fmpq_clear(&mut self.inner) };
    }
}
