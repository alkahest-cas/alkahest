//! Drop-safe wrappers for FLINT's `fmpz_mpoly_t`, `fmpz_mpoly_ctx_t`, and
//! `fmpz_mpoly_factor_t`.
//!
//! # Memory safety design
//!
//! `fmpz_mpoly_clear` requires the matching context (same `nvars`).  Storing
//! it separately would force every caller to pass it in at drop time, which
//! is error-prone.  Instead:
//!
//! * [`FlintMPolyCtx`] is always wrapped in an [`Arc`].
//! * Both [`FlintMPoly`] and [`FlintMPolyFactor`] clone that `Arc` on
//!   construction, guaranteeing the context outlives the polynomial.
//! * `Drop` impls call the correct FLINT clear function using `self.ctx`.
//!
//! Callers never need to pair `init` / `clear` manually; the RAII types
//! handle everything.

use super::ffi::{FmpzMPolyBuf, FmpzMPolyCtxBuf, FmpzMPolyFactorStruct};
use crate::flint::integer::FlintInteger;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Ordering constant: `ORD_LEX = 0` in FLINT's `ordering_t`.
const ORD_LEX: std::ffi::c_int = 0;

// ---------------------------------------------------------------------------
// FlintMPolyCtx — multivariate polynomial context
// ---------------------------------------------------------------------------

/// Owned `fmpz_mpoly_ctx_t` for lexicographic ordering over `nvars` variables.
///
/// Always construct via [`FlintMPolyCtx::new`], which returns an [`Arc`].
/// `Drop` calls `fmpz_mpoly_ctx_clear`.
pub struct FlintMPolyCtx {
    buf: Box<FmpzMPolyCtxBuf>,
    /// Number of polynomial variables; stored here so callers don't need to
    /// track it separately and so [`FlintMPoly::terms`] can use it directly.
    nvars: usize,
}

impl FlintMPolyCtx {
    /// Create a new lexicographic-order context for `nvars` variables and
    /// return it wrapped in an [`Arc`] so it can be shared with [`FlintMPoly`]
    /// and [`FlintMPolyFactor`] instances.
    pub fn new(nvars: usize) -> Arc<Self> {
        let mut buf = Box::new(FmpzMPolyCtxBuf([0u8; 608]));
        unsafe {
            super::ffi::fmpz_mpoly_ctx_init(buf.as_mut(), nvars as i64, ORD_LEX);
        }
        Arc::new(FlintMPolyCtx { buf, nvars })
    }

    /// Number of variables in this context.
    pub fn nvars(&self) -> usize {
        self.nvars
    }

    pub(super) fn as_ptr(&self) -> *const FmpzMPolyCtxBuf {
        self.buf.as_ref() as *const _
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
// FlintMPoly — sparse multivariate polynomial over ℤ
// ---------------------------------------------------------------------------

/// Owned `fmpz_mpoly_t` — sparse multivariate polynomial over ℤ.
///
/// The polynomial stores an [`Arc`] reference to its context so `Drop` can
/// always call `fmpz_mpoly_clear` correctly, even when the polynomial
/// outlives the original `ctx` binding at the call site.
pub struct FlintMPoly {
    buf: Box<FmpzMPolyBuf>,
    ctx: Arc<FlintMPolyCtx>,
}

// SAFETY: FLINT mpoly owns its memory; the Arc ensures the context outlives us.
unsafe impl Send for FlintMPoly {}
unsafe impl Sync for FlintMPoly {}

#[allow(dead_code)]
impl FlintMPoly {
    /// Create a new zero polynomial in the given context.
    pub fn new(ctx: Arc<FlintMPolyCtx>) -> Self {
        let mut buf = Box::new(FmpzMPolyBuf([0u8; 40]));
        unsafe {
            super::ffi::fmpz_mpoly_init(buf.as_mut(), ctx.as_ptr());
        }
        FlintMPoly { buf, ctx }
    }

    pub(crate) fn as_ptr(&self) -> *const FmpzMPolyBuf {
        self.buf.as_ref() as *const _
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut FmpzMPolyBuf {
        self.buf.as_mut() as *mut _
    }

    /// Whether this polynomial is zero.
    pub fn is_zero(&self) -> bool {
        unsafe { super::ffi::fmpz_mpoly_is_zero(self.as_ptr(), self.ctx.as_ptr()) != 0 }
    }

    /// Number of terms.
    pub fn length(&self) -> usize {
        unsafe { super::ffi::fmpz_mpoly_length(self.as_ptr(), self.ctx.as_ptr()) as usize }
    }

    /// Add the term `coeff · x₀^exp[0] · x₁^exp[1] · …`.
    ///
    /// `exp` must have length equal to `ctx.nvars()`.
    pub fn push_term(&mut self, coeff: &rug::Integer, exp: &[u64]) {
        // FlintInteger::from_rug is drop-safe — no manual fmpz_init/fmpz_clear needed.
        let fz = FlintInteger::from_rug(coeff);
        unsafe {
            super::ffi::fmpz_mpoly_push_term_fmpz_ui(
                self.as_mut_ptr(),
                fz.inner_ptr(),
                exp.as_ptr(),
                self.ctx.as_ptr(),
            );
        }
    }

    /// Sort terms and combine like monomials after a sequence of [`push_term`] calls.
    pub fn finish(&mut self) {
        unsafe {
            super::ffi::fmpz_mpoly_sort_terms(self.as_mut_ptr(), self.ctx.as_ptr());
            super::ffi::fmpz_mpoly_combine_like_terms(self.as_mut_ptr(), self.ctx.as_ptr());
        }
    }

    /// Compute `G = gcd(self, other)`. Returns `None` if FLINT fails.
    ///
    /// The two polynomials must share the same context (same `Arc` or same
    /// `nvars` and ordering; FLINT only checks compatibility, not identity).
    pub fn gcd(&self, other: &FlintMPoly) -> Option<FlintMPoly> {
        let mut g = FlintMPoly::new(Arc::clone(&self.ctx));
        let ok = unsafe {
            super::ffi::fmpz_mpoly_gcd(
                g.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                self.ctx.as_ptr(),
            )
        };
        if ok != 0 {
            Some(g)
        } else {
            None
        }
    }

    /// Compute the resultant of `self` and `other` w.r.t. variable `var_idx`.
    /// Returns `None` if FLINT fails.
    pub fn resultant(&self, other: &FlintMPoly, var_idx: usize) -> Option<FlintMPoly> {
        let mut r = FlintMPoly::new(Arc::clone(&self.ctx));
        let ok = unsafe {
            super::ffi::fmpz_mpoly_resultant(
                r.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                var_idx as super::ffi::slong,
                self.ctx.as_ptr(),
            )
        };
        if ok != 0 {
            Some(r)
        } else {
            None
        }
    }

    /// Extract all terms as `(exponent_vector, coefficient)` pairs.
    ///
    /// Exponent vectors have length equal to `ctx.nvars()` with trailing zeros
    /// stripped.  Terms with zero coefficient are omitted.
    pub fn terms(&self) -> BTreeMap<Vec<u32>, rug::Integer> {
        let nvars = self.ctx.nvars();
        let len = self.length();
        let mut result = BTreeMap::new();
        for i in 0..len {
            // FlintInteger is drop-safe — no raw fmpz_init/fmpz_clear needed.
            let mut coeff_fz = FlintInteger::new();
            let mut exp_u64 = vec![0u64; nvars];
            unsafe {
                super::ffi::fmpz_mpoly_get_term_coeff_fmpz(
                    coeff_fz.inner_mut_ptr(),
                    self.as_ptr(),
                    i as i64,
                    self.ctx.as_ptr(),
                );
                super::ffi::fmpz_mpoly_get_term_exp_ui(
                    exp_u64.as_mut_ptr(),
                    self.as_ptr(),
                    i as i64,
                    self.ctx.as_ptr(),
                );
            }
            let coeff = coeff_fz.to_rug();
            if coeff == 0 {
                continue;
            }
            // Truncate trailing zeros from the exponent vector.
            let mut exp_u32: Vec<u32> = exp_u64.iter().map(|&e| e as u32).collect();
            while exp_u32.last() == Some(&0) {
                exp_u32.pop();
            }
            result.insert(exp_u32, coeff);
        }
        result
    }

    /// Exact division: `Some(Q)` where `Q = self / divisor` if `divisor | self`,
    /// `None` otherwise (i.e. when the division is not exact or FLINT fails).
    ///
    /// Uses `fmpz_mpoly_divides` which fills `Q` and returns 1 on success.
    pub fn divides(&self, divisor: &FlintMPoly) -> Option<FlintMPoly> {
        let mut q = FlintMPoly::new(Arc::clone(&self.ctx));
        let ok = unsafe {
            super::ffi::fmpz_mpoly_divides(
                q.as_mut_ptr(),
                self.as_ptr(),
                divisor.as_ptr(),
                self.ctx.as_ptr(),
            )
        };
        if ok != 0 {
            Some(q)
        } else {
            None
        }
    }

    /// A shared reference to this polynomial's context.
    pub fn ctx(&self) -> &Arc<FlintMPolyCtx> {
        &self.ctx
    }
}

impl Drop for FlintMPoly {
    fn drop(&mut self) {
        // SAFETY: `self.buf` was initialised by `fmpz_mpoly_init` in `new`.
        // `self.ctx` is kept alive by the Arc until after this call.
        unsafe {
            super::ffi::fmpz_mpoly_clear(self.buf.as_mut(), self.ctx.as_ptr());
        }
        // `self.ctx` Arc drops here, potentially freeing the context.
    }
}

// ---------------------------------------------------------------------------
// FlintMPolyFactor — factorisation container for fmpz_mpoly
// ---------------------------------------------------------------------------

/// Owned `fmpz_mpoly_factor_t`.
///
/// Stores an [`Arc`] reference to the context so `Drop` can call
/// `fmpz_mpoly_factor_clear` correctly without any manual cleanup by callers.
pub struct FlintMPolyFactor {
    inner: FmpzMPolyFactorStruct,
    ctx: Arc<FlintMPolyCtx>,
}

// SAFETY: FLINT factor struct owns its memory; Arc context is thread-safe.
unsafe impl Send for FlintMPolyFactor {}
unsafe impl Sync for FlintMPolyFactor {}

impl FlintMPolyFactor {
    /// Initialise an empty factor container for the given context.
    pub fn new(ctx: Arc<FlintMPolyCtx>) -> Self {
        let mut inner = std::mem::MaybeUninit::<FmpzMPolyFactorStruct>::uninit();
        unsafe { super::ffi::fmpz_mpoly_factor_init(inner.as_mut_ptr(), ctx.as_ptr()) };
        // SAFETY: `fmpz_mpoly_factor_init` fully initialises the struct.
        Self {
            inner: unsafe { inner.assume_init() },
            ctx,
        }
    }

    /// Factor `poly` into this container (Bernardin–Monagan EEZ via FLINT).
    ///
    /// Returns `true` on success.  On failure the container is still in a
    /// consistent (empty) state and will be cleared correctly by `Drop`.
    pub fn factor(&mut self, poly: &FlintMPoly) -> bool {
        unsafe {
            super::ffi::fmpz_mpoly_factor(&mut self.inner, poly.as_ptr(), self.ctx.as_ptr()) != 0
        }
    }

    /// Check that `constant_den == 1` (FLINT always sets this for integer
    /// polynomials; a value other than 1 signals an internal failure).
    pub fn constant_den_is_one(&self) -> bool {
        unsafe { super::ffi::fmpz_cmp_ui(std::ptr::addr_of!(self.inner.constant_den), 1) == 0 }
    }

    /// The constant (unit) factor as a [`FlintInteger`].
    pub fn unit(&self) -> FlintInteger {
        let mut u = FlintInteger::new();
        unsafe {
            super::ffi::fmpz_mpoly_factor_get_constant_fmpz(
                u.inner_mut_ptr(),
                &self.inner,
                self.ctx.as_ptr(),
            );
        }
        u
    }

    /// Number of distinct irreducible factors.
    pub fn len(&self) -> usize {
        unsafe {
            super::ffi::fmpz_mpoly_factor_length(&self.inner, self.ctx.as_ptr()).max(0) as usize
        }
    }

    /// Copy the `i`-th irreducible factor into a new [`FlintMPoly`].
    pub fn base_at(&self, i: usize) -> FlintMPoly {
        debug_assert!(i < self.len());
        let mut base = FlintMPoly::new(Arc::clone(&self.ctx));
        unsafe {
            super::ffi::fmpz_mpoly_factor_get_base(
                base.as_mut_ptr(),
                &self.inner,
                i as super::ffi::slong,
                self.ctx.as_ptr(),
            );
        }
        base
    }

    /// Exponent (multiplicity) of the `i`-th factor.
    ///
    /// Note: FLINT's `fmpz_mpoly_factor_get_exp_si` takes `*mut` (despite
    /// only reading the data), so we need `&mut self` here.
    pub fn exp_at(&mut self, i: usize) -> u32 {
        debug_assert!(i < self.len());
        unsafe {
            super::ffi::fmpz_mpoly_factor_get_exp_si(
                &mut self.inner,
                i as super::ffi::slong,
                self.ctx.as_ptr(),
            ) as u32
        }
    }
}

impl Drop for FlintMPolyFactor {
    fn drop(&mut self) {
        // SAFETY: `self.inner` was initialised by `fmpz_mpoly_factor_init` in `new`.
        // `self.ctx` Arc keeps the context alive until after this call.
        unsafe {
            super::ffi::fmpz_mpoly_factor_clear(&mut self.inner, self.ctx.as_ptr());
        }
    }
}
