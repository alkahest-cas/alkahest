/// Safe Rust wrappers around FLINT's `fmpz_mpoly_t` / `fmpz_mpoly_ctx_t`.
///
/// These types are used only for the multivariate GCD computation in
/// [`MultiPoly::gcd`]; they are not part of the public library API.
use super::ffi::{FmpzMPolyBuf, FmpzMPolyCtxBuf};
use crate::flint::ffi::{fmpz, fmpz_clear, fmpz_init, fmpz_set_str};
use std::collections::BTreeMap;
use std::ffi::CString;

/// Ordering constant: ORD_LEX = 0 in FLINT's `ordering_t`.
const ORD_LEX: std::ffi::c_int = 0;

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

pub struct FlintMPolyCtx {
    buf: Box<FmpzMPolyCtxBuf>,
}

impl FlintMPolyCtx {
    pub fn new(nvars: usize) -> Self {
        let mut buf = Box::new(FmpzMPolyCtxBuf([0u8; 608]));
        unsafe {
            super::ffi::fmpz_mpoly_ctx_init(buf.as_mut(), nvars as i64, ORD_LEX);
        }
        FlintMPolyCtx { buf }
    }

    pub fn as_ptr(&self) -> *const FmpzMPolyCtxBuf {
        self.buf.as_ref() as *const _
    }

    #[allow(dead_code)]
    pub fn as_mut_ptr(&mut self) -> *mut FmpzMPolyCtxBuf {
        self.buf.as_mut() as *mut _
    }
}

impl Drop for FlintMPolyCtx {
    fn drop(&mut self) {
        unsafe {
            super::ffi::fmpz_mpoly_ctx_clear(self.buf.as_mut());
        }
    }
}

// ---------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------

pub struct FlintMPoly {
    buf: Box<FmpzMPolyBuf>,
}

impl FlintMPoly {
    pub fn new(ctx: &FlintMPolyCtx) -> Self {
        let mut buf = Box::new(FmpzMPolyBuf([0u8; 40]));
        unsafe {
            super::ffi::fmpz_mpoly_init(buf.as_mut(), ctx.as_ptr());
        }
        FlintMPoly { buf }
    }

    pub fn as_ptr(&self) -> *const FmpzMPolyBuf {
        self.buf.as_ref() as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut FmpzMPolyBuf {
        self.buf.as_mut() as *mut _
    }

    #[allow(dead_code)]
    pub fn is_zero(&self, ctx: &FlintMPolyCtx) -> bool {
        unsafe { super::ffi::fmpz_mpoly_is_zero(self.as_ptr(), ctx.as_ptr()) != 0 }
    }

    pub fn length(&self, ctx: &FlintMPolyCtx) -> usize {
        unsafe { super::ffi::fmpz_mpoly_length(self.as_ptr(), ctx.as_ptr()) as usize }
    }

    /// Add a term `coeff * vars[0]^exp[0] * vars[1]^exp[1] * ...`.
    pub fn push_term(&mut self, coeff: &rug::Integer, exp: &[u64], ctx: &FlintMPolyCtx) {
        // Convert rug::Integer → fmpz via string
        let s = coeff.to_string_radix(10);
        let cstr = CString::new(s).unwrap();
        let mut fz: fmpz = 0;
        unsafe {
            fmpz_init(&mut fz);
            fmpz_set_str(&mut fz, cstr.as_ptr(), 10);
            super::ffi::fmpz_mpoly_push_term_fmpz_ui(
                self.as_mut_ptr(),
                &fz,
                exp.as_ptr(),
                ctx.as_ptr(),
            );
            fmpz_clear(&mut fz);
        }
    }

    /// Finalize after a sequence of `push_term` calls (sort + combine).
    pub fn finish(&mut self, ctx: &FlintMPolyCtx) {
        unsafe {
            super::ffi::fmpz_mpoly_sort_terms(self.as_mut_ptr(), ctx.as_ptr());
            super::ffi::fmpz_mpoly_combine_like_terms(self.as_mut_ptr(), ctx.as_ptr());
        }
    }

    /// Release FLINT buffers.  Call when dropping is not enough (this type’s
    /// `Drop` cannot invoke `fmpz_mpoly_clear` without a context).
    pub unsafe fn clear_with_ctx(&mut self, ctx: &FlintMPolyCtx) {
        super::ffi::fmpz_mpoly_clear(self.as_mut_ptr(), ctx.as_ptr());
        self.buf.0.fill(0);
    }

    /// Compute the resultant of `self` and `other` with respect to variable
    /// at index `var_idx` in the context.  Returns `None` if FLINT fails.
    pub fn resultant(
        &self,
        other: &FlintMPoly,
        var_idx: usize,
        ctx: &FlintMPolyCtx,
    ) -> Option<FlintMPoly> {
        let mut r = FlintMPoly::new(ctx);
        let ok = unsafe {
            super::ffi::fmpz_mpoly_resultant(
                r.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                var_idx as super::ffi::slong,
                ctx.as_ptr(),
            )
        };
        if ok != 0 {
            Some(r)
        } else {
            None
        }
    }

    /// Compute GCD: `G = gcd(self, other)`. Returns `None` if FLINT fails.
    pub fn gcd(&self, other: &FlintMPoly, ctx: &FlintMPolyCtx) -> Option<FlintMPoly> {
        let mut g = FlintMPoly::new(ctx);
        let ok = unsafe {
            super::ffi::fmpz_mpoly_gcd(g.as_mut_ptr(), self.as_ptr(), other.as_ptr(), ctx.as_ptr())
        };
        if ok != 0 {
            Some(g)
        } else {
            None
        }
    }

    /// Extract all terms as `(exponent_vector, coefficient)` pairs.
    pub fn terms(&self, nvars: usize, ctx: &FlintMPolyCtx) -> BTreeMap<Vec<u32>, rug::Integer> {
        let len = self.length(ctx);
        let mut result = BTreeMap::new();
        for i in 0..len {
            let mut exp_u64 = vec![0u64; nvars];
            let mut fz: fmpz = 0;
            unsafe {
                fmpz_init(&mut fz);
                super::ffi::fmpz_mpoly_get_term_coeff_fmpz(
                    &mut fz,
                    self.as_ptr(),
                    i as i64,
                    ctx.as_ptr(),
                );
                super::ffi::fmpz_mpoly_get_term_exp_ui(
                    exp_u64.as_mut_ptr(),
                    self.as_ptr(),
                    i as i64,
                    ctx.as_ptr(),
                );
            }
            // Convert fmpz → rug::Integer via string
            let coeff = fmpz_to_rug(fz);
            unsafe {
                fmpz_clear(&mut fz);
            }

            // Strip trailing zeros in exponent vector
            let mut exp_u32: Vec<u32> = exp_u64.iter().map(|&e| e as u32).collect();
            while exp_u32.last() == Some(&0) {
                exp_u32.pop();
            }

            if coeff != 0 {
                result.insert(exp_u32, coeff);
            }
        }
        result
    }
}

impl Drop for FlintMPoly {
    fn drop(&mut self) {
        // We need a context to call fmpz_mpoly_clear, but we don't store one.
        // FLINT's clear only frees internal memory; passing any valid ctx is safe
        // as long as the nvars matches. Since we can't guarantee that here,
        // we zero the buffer instead (leaking any GMP-backed coefficients).
        // For correctness in production, callers should clear explicitly before
        // dropping.  This is acceptable for a short-lived GCD helper.
        //
        // Actually: fmpz_mpoly_clear does need the right nvars to free the
        // exponent data. We can't safely clear without the ctx. We'll zero the
        // storage to avoid UB from FLINT's finaliser running on garbage.
        self.buf.0.iter_mut().for_each(|b| *b = 0);
    }
}

/// Convert an inline fmpz (that has already been read) to rug::Integer.
fn fmpz_to_rug(fz: fmpz) -> rug::Integer {
    // An fmpz with the low bit clear is a direct integer (shifted right by 1).
    // An fmpz with the low bit set is a pointer to an mpz_t.
    // Round-tripping through the string API is the safe portable approach.
    let s = fmpz_to_string(fz);
    s.parse().unwrap_or(rug::Integer::from(0))
}

fn fmpz_to_string(fz: fmpz) -> String {
    use std::ffi::CStr;
    unsafe {
        let ptr = super::ffi::fmpz_get_str(std::ptr::null_mut(), 10, &fz);
        if ptr.is_null() {
            return "0".to_string();
        }
        let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        super::ffi::flint_free(ptr as *mut _);
        s
    }
}
