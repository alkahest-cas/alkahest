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
        crate::flint::note_thread_uses_flint();
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
        crate::flint::note_thread_uses_flint();
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
        // FLINT reads `nvars` words from `exp`.
        assert!(exp.len() >= self.ctx.nvars(), "exponent vector too short");
        if let Some(c) = coeff.to_i64() {
            // Word-sized: no temporary fmpz to initialise and clear.
            // SAFETY: `exp` holds `nvars` words; the polynomial is initialised.
            unsafe {
                super::ffi::fmpz_mpoly_push_term_si_ui(
                    self.as_mut_ptr(),
                    c,
                    exp.as_ptr(),
                    self.ctx.as_ptr(),
                );
            }
            return;
        }
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

    /// Build a polynomial from terms whose exponent vectors are read
    /// zero-padded or truncated to `ctx.nvars()` words.
    ///
    /// When the terms arrive in strictly descending lexicographic order with
    /// nonzero coefficients — a `TermMap` iterated in reverse, whose keys
    /// order exactly as their padded vectors do — the pushed polynomial is
    /// already canonical and FLINT's sort and combine passes are skipped.
    /// Any other input (a zero coefficient, a repeated monomial, keys out of
    /// order) is detected on the fly and finished with [`Self::finish`], so
    /// the result is canonical either way.
    pub(crate) fn from_terms<'a, I>(ctx: Arc<FlintMPolyCtx>, terms: I) -> FlintMPoly
    where
        I: IntoIterator<Item = (&'a [u32], &'a rug::Integer)>,
    {
        let nvars = ctx.nvars();
        let mut fp = FlintMPoly::new(ctx);
        let mut e = vec![0u64; nvars];
        let mut prev: Option<Vec<u64>> = None;
        let mut canonical = true;
        for (exp, c) in terms {
            e.iter_mut().for_each(|v| *v = 0);
            for (slot, &x) in e.iter_mut().zip(exp) {
                *slot = u64::from(x);
            }
            if canonical {
                if *c == 0 {
                    canonical = false;
                } else {
                    match prev.as_mut() {
                        Some(p) if e.as_slice() >= p.as_slice() => canonical = false,
                        Some(p) => p.copy_from_slice(&e),
                        None => prev = Some(e.clone()),
                    }
                }
            }
            fp.push_term(c, &e);
        }
        if !canonical {
            fp.finish();
        }
        debug_assert!(fp.is_canonical());
        fp
    }

    /// `(per-variable degrees, total degree)`, all `0` for the zero
    /// polynomial, or `None` if some degree does not fit in a signed word.
    pub(crate) fn degrees(&self) -> Option<(Vec<u64>, u64)> {
        let nvars = self.ctx.nvars();
        // SAFETY: both pointers are initialised; `degs` holds `nvars` words,
        // and `fmpz_mpoly_degrees_si` is only called once the degrees are
        // known to fit (it throws otherwise).
        unsafe {
            if super::ffi::fmpz_mpoly_degrees_fit_si(self.as_ptr(), self.ctx.as_ptr()) == 0 {
                return None;
            }
            let mut degs = vec![0 as super::ffi::slong; nvars];
            super::ffi::fmpz_mpoly_degrees_si(degs.as_mut_ptr(), self.as_ptr(), self.ctx.as_ptr());
            let per: Vec<u64> = degs.iter().map(|&d| d.max(0) as u64).collect();
            // Each degree is below 2^63 and `nvars` is small, but sum in u128
            // so the total cannot wrap even so.
            let total: u128 = per.iter().map(|&d| u128::from(d)).sum();
            if total > u128::from(i64::MAX as u64) {
                return None;
            }
            let t = super::ffi::fmpz_mpoly_total_degree_si(self.as_ptr(), self.ctx.as_ptr());
            Some((per, t.max(0) as u64))
        }
    }

    /// `-self`.
    pub(crate) fn neg(&self) -> FlintMPoly {
        let zero = FlintMPoly::new(Arc::clone(&self.ctx));
        zero.sub(self)
    }

    /// FLINT's internal consistency check (sorted, no zero or repeated terms).
    pub(crate) fn is_canonical(&self) -> bool {
        // SAFETY: the polynomial is initialised.
        unsafe { super::ffi::fmpz_mpoly_is_canonical(self.as_ptr(), self.ctx.as_ptr()) != 0 }
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

    /// `(G, self/G, other/G)` with `G = gcd(self, other)` (positive leading
    /// coefficient, as [`Self::gcd`]) in one FLINT call, or `None` if FLINT
    /// fails.
    pub(crate) fn gcd_cofactors(
        &self,
        other: &FlintMPoly,
    ) -> Option<(FlintMPoly, FlintMPoly, FlintMPoly)> {
        let mut g = FlintMPoly::new(Arc::clone(&self.ctx));
        let mut abar = FlintMPoly::new(Arc::clone(&self.ctx));
        let mut bbar = FlintMPoly::new(Arc::clone(&self.ctx));
        // SAFETY: all five polynomials are initialised in compatible contexts;
        // the three outputs are fresh, so none aliases another or an input.
        let ok = unsafe {
            super::ffi::fmpz_mpoly_gcd_cofactors(
                g.as_mut_ptr(),
                abar.as_mut_ptr(),
                bbar.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                self.ctx.as_ptr(),
            )
        };
        if ok != 0 {
            Some((g, abar, bbar))
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
    ///
    /// # Panics
    ///
    /// If an exponent exceeds `u32::MAX` (FLINT exponents are unbounded, so a
    /// product or resultant can produce one). It used to be truncated to its
    /// low 32 bits — `x^(2^32)` read back as `1`. Use [`Self::try_terms`] or
    /// [`Self::terms_u64`] to handle that case.
    pub fn terms(&self) -> BTreeMap<Vec<u32>, rug::Integer> {
        self.try_terms()
            .unwrap_or_else(|| crate::poly::exponent::overflow_panic())
    }

    /// [`Self::terms`], or `None` if some exponent exceeds `u32::MAX`.
    pub(crate) fn try_terms(&self) -> Option<BTreeMap<Vec<u32>, rug::Integer>> {
        self.collect_terms(|e| u32::try_from(e).ok())
    }

    /// All terms with `u64` exponents (trailing zeros stripped, zero
    /// coefficients omitted), or `None` if some exponent does not fit in a
    /// machine word.
    pub(crate) fn terms_u64(&self) -> Option<BTreeMap<Vec<u64>, rug::Integer>> {
        self.collect_terms(Some)
    }

    /// The terms as a map with exponents narrowed by `narrow` (trailing zeros
    /// stripped, zero coefficients omitted), or `None` if some exponent does
    /// not fit in a machine word or `narrow` refuses one.
    ///
    /// Reads the coefficients in place and walks the terms in ascending
    /// order, so the map is bulk-built from sorted input instead of one
    /// insertion per term.
    fn collect_terms<E, F>(&self, narrow: F) -> Option<BTreeMap<Vec<E>, rug::Integer>>
    where
        E: Ord,
        F: Fn(u64) -> Option<E>,
    {
        let nvars = self.ctx.nvars();
        let len = self.length();
        // One check for the whole polynomial instead of one per term.
        // SAFETY: both pointers are initialised.
        let all_fit =
            unsafe { super::ffi::fmpz_mpoly_degrees_fit_si(self.as_ptr(), self.ctx.as_ptr()) != 0 };
        let mut exp_u64 = vec![0u64; nvars];
        let mut out: Vec<(Vec<E>, rug::Integer)> = Vec::with_capacity(len);
        // FLINT stores terms in descending order; read them ascending.
        for i in (0..len).rev() {
            // SAFETY: `i < length`, and both pointers are initialised.
            if !all_fit
                && unsafe {
                    super::ffi::fmpz_mpoly_term_exp_fits_ui(
                        self.as_ptr(),
                        i as i64,
                        self.ctx.as_ptr(),
                    )
                } == 0
            {
                // `fmpz_mpoly_get_term_exp_ui` would throw (abort) here.
                return None;
            }
            // SAFETY: `i < length`; `exp_u64` holds `nvars` words; the
            // coefficient pointer is valid while `self` is borrowed and is
            // only read.
            let coeff = unsafe {
                super::ffi::fmpz_mpoly_get_term_exp_ui(
                    exp_u64.as_mut_ptr(),
                    self.as_ptr(),
                    i as i64,
                    self.ctx.as_ptr(),
                );
                let c = super::ffi::fmpz_mpoly_term_coeff_ref(
                    self.as_ptr(),
                    i as i64,
                    self.ctx.as_ptr(),
                );
                crate::flint::integer::fmpz_get_rug(c)
            };
            if coeff == 0 {
                continue;
            }
            // Truncate trailing zeros from the exponent vector.
            let mut k = nvars;
            while k > 0 && exp_u64[k - 1] == 0 {
                k -= 1;
            }
            let key: Option<Vec<E>> = exp_u64[..k].iter().map(|&e| narrow(e)).collect();
            out.push((key?, coeff));
        }
        // Sorted ascending already: `collect` detects the run and bulk-builds.
        Some(out.into_iter().collect())
    }

    /// Product `self * other`. Both must share a context (same `nvars`).
    pub(crate) fn mul(&self, other: &FlintMPoly) -> FlintMPoly {
        let mut r = FlintMPoly::new(Arc::clone(&self.ctx));
        // SAFETY: all three polynomials are initialised in compatible contexts;
        // `r` is freshly allocated so it aliases neither input.
        unsafe {
            super::ffi::fmpz_mpoly_mul(
                r.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                self.ctx.as_ptr(),
            );
        }
        r
    }

    /// Sum `self + other`. Both must share a context (same `nvars`).
    pub(crate) fn add(&self, other: &FlintMPoly) -> FlintMPoly {
        let mut r = FlintMPoly::new(Arc::clone(&self.ctx));
        // SAFETY: as in `mul`.
        unsafe {
            super::ffi::fmpz_mpoly_add(
                r.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                self.ctx.as_ptr(),
            );
        }
        r
    }

    /// Difference `self - other`. Both must share a context (same `nvars`).
    pub(crate) fn sub(&self, other: &FlintMPoly) -> FlintMPoly {
        let mut r = FlintMPoly::new(Arc::clone(&self.ctx));
        // SAFETY: as in `mul`.
        unsafe {
            super::ffi::fmpz_mpoly_sub(
                r.as_mut_ptr(),
                self.as_ptr(),
                other.as_ptr(),
                self.ctx.as_ptr(),
            );
        }
        r
    }

    /// Power `self^k`, or `None` if FLINT reports failure.
    pub(crate) fn pow_ui(&self, k: u64) -> Option<FlintMPoly> {
        let mut r = FlintMPoly::new(Arc::clone(&self.ctx));
        // SAFETY: as in `mul`.
        let ok = unsafe {
            super::ffi::fmpz_mpoly_pow_ui(r.as_mut_ptr(), self.as_ptr(), k, self.ctx.as_ptr())
        };
        if ok != 0 {
            Some(r)
        } else {
            None
        }
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

    /// Partial derivative `∂self/∂x_var`.
    pub(crate) fn derivative(&self, var: usize) -> FlintMPoly {
        let mut r = FlintMPoly::new(Arc::clone(&self.ctx));
        // SAFETY: as in `mul`; `var < nvars` is the caller's invariant.
        debug_assert!(var < self.ctx.nvars());
        unsafe {
            super::ffi::fmpz_mpoly_derivative(
                r.as_mut_ptr(),
                self.as_ptr(),
                var as i64,
                self.ctx.as_ptr(),
            );
        }
        r
    }

    /// Degree in variable `var` (`-1` for the zero polynomial).
    pub(crate) fn degree_in(&self, var: usize) -> i64 {
        // SAFETY: `var < nvars` is the caller's invariant; the polynomial is initialised.
        debug_assert!(var < self.ctx.nvars());
        unsafe { super::ffi::fmpz_mpoly_degree_si(self.as_ptr(), var as i64, self.ctx.as_ptr()) }
    }

    /// Structural equality (same context assumed).
    pub(crate) fn equal(&self, other: &FlintMPoly) -> bool {
        // SAFETY: both polynomials are initialised in compatible contexts.
        unsafe {
            super::ffi::fmpz_mpoly_equal(self.as_ptr(), other.as_ptr(), self.ctx.as_ptr()) != 0
        }
    }

    /// The coefficient of the leading term (term 0 in FLINT's descending
    /// order), or `0` for the zero polynomial.
    pub(crate) fn leading_coeff(&self) -> rug::Integer {
        if self.is_zero() {
            return rug::Integer::new();
        }
        let mut c = FlintInteger::new();
        // SAFETY: the polynomial is non-zero, so term 0 exists.
        unsafe {
            super::ffi::fmpz_mpoly_get_term_coeff_fmpz(
                c.inner_mut_ptr(),
                self.as_ptr(),
                0,
                self.ctx.as_ptr(),
            );
        }
        c.to_rug()
    }

    /// A shared reference to this polynomial's context.
    pub fn ctx(&self) -> &Arc<FlintMPolyCtx> {
        &self.ctx
    }
}

impl Clone for FlintMPoly {
    fn clone(&self) -> Self {
        let mut r = FlintMPoly::new(Arc::clone(&self.ctx));
        // SAFETY: both initialised in the same context; `r` is fresh.
        unsafe {
            super::ffi::fmpz_mpoly_set(r.as_mut_ptr(), self.as_ptr(), self.ctx.as_ptr());
        }
        r
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
        crate::flint::note_thread_uses_flint();
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
        // A real check, not `debug_assert!`: this is a safe fn, and an index
        // past `len()` reads out of bounds of the FLINT array in release.
        assert!(
            i < self.len(),
            "factor index {i} out of range (len {})",
            self.len()
        );
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

    /// The indices `0..len()` in the canonical factor order of
    /// [`crate::poly::factor::cmp_multivariate_terms`], ties broken by
    /// ascending multiplicity. FLINT's own order depends on its version and
    /// internal choices; iterate over this instead of `0..len()`.
    ///
    /// Falls back to FLINT's order only if some exponent does not fit in a
    /// machine word (no such factor can be built from this crate's types).
    pub fn canonical_order(&mut self) -> Vec<usize> {
        let n = self.len();
        let mut keyed = Vec::with_capacity(n);
        for i in 0..n {
            let Some(terms) = self.base_at(i).terms_u64() else {
                return (0..n).collect();
            };
            keyed.push((i, terms, self.exp_at(i)));
        }
        keyed.sort_by(|a, b| {
            crate::poly::factor::cmp_multivariate_terms(&a.1, &b.1).then(a.2.cmp(&b.2))
        });
        keyed.into_iter().map(|(i, _, _)| i).collect()
    }

    /// Exponent (multiplicity) of the `i`-th factor.
    ///
    /// Note: FLINT's `fmpz_mpoly_factor_get_exp_si` takes `*mut` (despite
    /// only reading the data), so we need `&mut self` here.
    pub fn exp_at(&mut self, i: usize) -> u32 {
        // A real check, not `debug_assert!`: this is a safe fn, and an index
        // past `len()` reads out of bounds of the FLINT array in release.
        assert!(
            i < self.len(),
            "factor index {i} out of range (len {})",
            self.len()
        );
        let e = unsafe {
            super::ffi::fmpz_mpoly_factor_get_exp_si(
                &mut self.inner,
                i as super::ffi::slong,
                self.ctx.as_ptr(),
            )
        };
        // A multiplicity is at most the degree of the factored polynomial, and
        // those enter through `u32`-exponent terms, so this cannot fail for a
        // polynomial built by this crate; refuse to truncate if it ever does.
        u32::try_from(e).unwrap_or_else(|_| crate::poly::exponent::overflow_panic())
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
