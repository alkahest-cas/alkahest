use super::ffi;
use std::ffi::CString;
use std::fmt;
use std::ops::{Add, Mul, Sub};

// ---------------------------------------------------------------------------
// FlintPolyFactor — drop-safe factorisation container for fmpz_poly
// ---------------------------------------------------------------------------

/// Owned `fmpz_poly_factor_t`.  `Drop` calls `fmpz_poly_factor_clear`.
pub(crate) struct FlintPolyFactor {
    inner: ffi::FmpzPolyFactorStruct,
}

impl FlintPolyFactor {
    pub fn new() -> Self {
        let mut inner = std::mem::MaybeUninit::<ffi::FmpzPolyFactorStruct>::uninit();
        crate::flint::note_thread_uses_flint();
        unsafe { ffi::fmpz_poly_factor_init(inner.as_mut_ptr()) };
        // SAFETY: `fmpz_poly_factor_init` fully initialises the struct.
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    pub fn factor(&mut self, poly: &FlintPoly) {
        unsafe { ffi::fmpz_poly_factor(&mut self.inner, &poly.inner) };
    }

    /// Number of distinct irreducible factors.
    pub fn len(&self) -> usize {
        self.inner.num.max(0) as usize
    }

    /// The unit (leading-coefficient sign) as a [`super::integer::FlintInteger`].
    pub fn unit(&self) -> super::integer::FlintInteger {
        let mut u = super::integer::FlintInteger::new();
        unsafe { ffi::fmpz_poly_factor_get_fmpz(u.inner_mut_ptr(), &self.inner) };
        u
    }

    /// Copy the `i`-th irreducible factor into a new [`FlintPoly`].
    pub fn poly_at(&self, i: usize) -> FlintPoly {
        // A real check, not `debug_assert!`: this is a safe fn, and an index
        // past `len()` reads out of bounds of the FLINT array in release.
        assert!(
            i < self.len(),
            "factor index {i} out of range (len {})",
            self.len()
        );
        let mut p = FlintPoly::new();
        unsafe { ffi::fmpz_poly_factor_get_fmpz_poly(&mut p.inner, &self.inner, i as ffi::slong) };
        p
    }

    /// Exponent of the `i`-th factor.
    pub fn exp_at(&self, i: usize) -> u32 {
        // A real check, not `debug_assert!`: this is a safe fn, and an index
        // past `len()` reads out of bounds of the FLINT array in release.
        assert!(
            i < self.len(),
            "factor index {i} out of range (len {})",
            self.len()
        );
        // SAFETY: `i < num` so the pointer arithmetic is in bounds.
        unsafe { *self.inner.exp.add(i) as u32 }
    }
}

impl Drop for FlintPolyFactor {
    fn drop(&mut self) {
        // SAFETY: `self.inner` was initialised by `fmpz_poly_factor_init` in `new`.
        unsafe { ffi::fmpz_poly_factor_clear(&mut self.inner) };
    }
}

/// The largest coefficient index a FLINT polynomial wrapper will accept.
///
/// A dense polynomial of length `n + 1` needs `(n + 1)` machine words before a
/// single coefficient is big; past this no allocation can succeed, and an
/// index above `i64::MAX` would reach FLINT as a negative `slong`.
pub const MAX_COEFF_INDEX: usize = (isize::MAX as usize) / std::mem::size_of::<ffi::fmpz>() - 1;

/// Panic (rather than let FLINT corrupt the heap or abort) on an index no
/// polynomial can have.
#[track_caller]
pub(crate) fn check_coeff_index(n: usize) {
    assert!(
        n <= MAX_COEFF_INDEX,
        "coefficient index {n} is past the largest addressable polynomial length"
    );
}

/// Safe wrapper over FLINT's `fmpz_poly_t` — dense univariate polynomial
/// over the integers (`ℤ[x]`).
///
/// Coefficients are stored in ascending degree order:
/// `[c₀, c₁, …, cₙ]` represents `c₀ + c₁·x + … + cₙ·xⁿ`.
///
/// Memory is managed by FLINT. `Drop` calls `fmpz_poly_clear`.
pub struct FlintPoly {
    inner: ffi::FmpzPolyStruct,
}

// SAFETY: FlintPoly owns its coefficient buffer via FLINT's allocator.
unsafe impl Send for FlintPoly {}
unsafe impl Sync for FlintPoly {}

impl FlintPoly {
    pub fn new() -> Self {
        let mut inner = ffi::FmpzPolyStruct {
            coeffs: std::ptr::null_mut(),
            alloc: 0,
            length: 0,
        };
        crate::flint::note_thread_uses_flint();
        unsafe { ffi::fmpz_poly_init(&mut inner) };
        FlintPoly { inner }
    }

    /// Construct from coefficient slice in ascending degree order.
    ///
    /// # Panics
    ///
    /// As [`set_coeff_flint`](Self::set_coeff_flint), for a slice longer
    /// than [`MAX_COEFF_INDEX`].
    /// `from_coefficients(&[1, 2, 3])` → `1 + 2x + 3x²`.
    pub fn from_coefficients(coeffs: &[i64]) -> Self {
        let mut p = Self::new();
        if let Some(last) = coeffs.len().checked_sub(1) {
            check_coeff_index(last);
        }
        for (i, &c) in coeffs.iter().enumerate() {
            unsafe { ffi::fmpz_poly_set_coeff_si(&mut p.inner, i as ffi::slong, c) };
        }
        p
    }

    /// Number of coefficients stored (degree + 1 for non-zero poly, 0 for zero poly).
    pub fn length(&self) -> usize {
        unsafe { ffi::fmpz_poly_length(&self.inner) as usize }
    }

    /// Degree of the polynomial (-1 for zero polynomial).
    pub fn degree(&self) -> i64 {
        unsafe { ffi::fmpz_poly_degree(&self.inner) }
    }

    /// Coefficient of `x^n` as `i64`. Returns 0 for out-of-range indices.
    pub fn get_coeff(&self, n: usize) -> i64 {
        // An index past the length is zero by definition. Checking here also
        // keeps `n > i64::MAX` from reaching FLINT as a *negative* `slong`,
        // which it would use to read before the coefficient array.
        if n >= self.length() {
            return 0;
        }
        unsafe { ffi::fmpz_poly_get_coeff_si(&self.inner, n as ffi::slong) }
    }

    /// Coefficient vector in ascending degree order.
    pub fn coefficients(&self) -> Vec<i64> {
        (0..self.length()).map(|i| self.get_coeff(i)).collect()
    }

    pub fn is_zero(&self) -> bool {
        self.length() == 0
    }

    /// `self^exp`.
    ///
    /// # Panics
    ///
    /// When the result could not fit in memory (see [`Self::checked_pow`]).
    /// FLINT used to be asked for it anyway and abort the process;
    /// `(x + 1)^(2^21)` wanted half a terabyte.
    pub fn pow(&self, exp: u32) -> Self {
        self.checked_pow(exp).unwrap_or_else(|| {
            panic!(
                "FlintPoly::pow: the result of raising a degree-{} polynomial to the \
                 power {exp} would not fit in memory (E-POLY-004)",
                self.degree()
            )
        })
    }

    /// `self^exp`, or `None` when the result's estimated size — degree
    /// `deg·exp`, coefficients of up to `exp·log₂‖self‖₁` bits — would not fit
    /// the machine's memory, the active `Budget(max_bytes=…)` or `RLIMIT_AS`.
    /// Checked before FLINT allocates, since FLINT aborts on failure.
    pub fn checked_pow(&self, exp: u32) -> Option<Self> {
        let deg = self.degree();
        if deg >= 0 && exp > 1 {
            let coeffs: Vec<rug::Integer> = (0..self.length())
                .map(|i| self.get_coeff_flint(i).to_rug())
                .collect();
            let nonzero = coeffs.iter().filter(|c| **c != 0).count();
            let dense = (deg as f64) * f64::from(exp) + 1.0;
            let terms = crate::poly::size::power_term_bound(nonzero, exp, dense);
            let bits = f64::from(exp) * crate::poly::size::log2_l1_norm(&coeffs);
            // The dense coefficient array costs one 8-byte slot per degree
            // whether or not the coefficient is zero.
            crate::poly::size::check_power_size(terms, bits, 8).ok()?;
            crate::poly::size::check_power_size(dense, 0.0, 8).ok()?;
        }
        Some(self.pow_unchecked(exp))
    }

    fn pow_unchecked(&self, exp: u32) -> Self {
        // A monomial `c·x^k` is raised directly. FLINT treats a length-2 poly
        // (`x` itself is `[0, 1]`) as a binomial and builds every binomial
        // coefficient C(exp, i) before multiplying by the zero constant term,
        // so `x^(2^21)` took gigabytes and minutes; `c^exp·x^(k·exp)` needs
        // one coefficient.
        let deg = self.degree();
        if deg >= 1 && exp > 1 {
            if let Some(top) = (deg as usize).checked_mul(exp as usize) {
                if (0..deg as usize).all(|i| self.get_coeff_flint(i).to_rug() == 0) {
                    let mut res = Self::new();
                    let c = self.leading_coeff_fmpz().pow(u64::from(exp));
                    res.set_coeff_flint(top, &c);
                    return res;
                }
            }
        }
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_pow(&mut res.inner, &self.inner, exp as ffi::ulong) };
        res
    }

    /// [`div_exact`](Self::div_exact), or `None` when `divisor` is the zero
    /// polynomial.
    pub fn checked_div_exact(&self, divisor: &Self) -> Option<Self> {
        (!divisor.is_zero()).then(|| self.div_exact(divisor))
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_gcd(&mut res.inner, &self.inner, &other.inner) };
        res
    }

    /// Exact polynomial division: returns `self / divisor`, assuming `divisor` divides `self`.
    ///
    /// # Panics
    ///
    /// When `divisor` is the zero polynomial (FLINT would `abort()` the
    /// process). [`checked_div_exact`](Self::checked_div_exact) returns
    /// `None` instead.
    pub fn div_exact(&self, divisor: &Self) -> Self {
        assert!(
            !divisor.is_zero(),
            "attempt to divide a FlintPoly by the zero polynomial"
        );
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_div(&mut res.inner, &self.inner, &divisor.inner) };
        res
    }

    /// Negate all coefficients: `-self`.
    pub fn neg(&self) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_neg(&mut res.inner, &self.inner) };
        res
    }

    /// Multiply every coefficient by the integer `c`.
    pub fn scalar_mul_fmpz(&self, c: &super::integer::FlintInteger) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_scalar_mul_fmpz(&mut res.inner, &self.inner, c.inner_ptr()) };
        res
    }

    /// Divide every coefficient by `c` (exact — caller ensures divisibility).
    ///
    /// # Panics
    ///
    /// When `c` is zero (FLINT would `abort()` the process).
    pub fn scalar_divexact_fmpz(&self, c: &super::integer::FlintInteger) -> Self {
        assert!(!c.is_zero(), "attempt to divide a FlintPoly by zero");
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_scalar_divexact_fmpz(&mut res.inner, &self.inner, c.inner_ptr()) };
        res
    }

    /// Leading coefficient as a `FlintInteger` (0 for the zero polynomial).
    pub fn leading_coeff_fmpz(&self) -> super::integer::FlintInteger {
        let deg = self.degree();
        if deg < 0 {
            return super::integer::FlintInteger::from_i64(0);
        }
        self.get_coeff_flint(deg as usize)
    }

    /// Compute the resultant of `self` and `other` as a `FlintInteger`.
    ///
    /// Returns the integer `res(self, other)`.  For the zero polynomial the
    /// resultant is defined to be 0.
    pub fn resultant(&self, other: &Self) -> super::integer::FlintInteger {
        let mut res = super::integer::FlintInteger::new();
        unsafe { ffi::fmpz_poly_resultant(res.inner_mut_ptr(), &self.inner, &other.inner) };
        res
    }

    /// Pseudo-division: returns `(Q, R, d)` such that `lc(other)^d * self = Q * other + R`.
    ///
    /// # Panics
    ///
    /// When `other` is the zero polynomial (FLINT would `abort()` the process).
    pub fn pseudo_divrem(&self, other: &Self) -> (Self, Self, u64) {
        assert!(
            !other.is_zero(),
            "attempt to pseudo-divide a FlintPoly by the zero polynomial"
        );
        let mut q = Self::new();
        let mut r = Self::new();
        let mut d: ffi::ulong = 0;
        unsafe {
            ffi::fmpz_poly_pseudo_divrem(
                &mut q.inner,
                &mut r.inner,
                &mut d,
                &self.inner,
                &other.inner,
            )
        };
        (q, r, d)
    }

    /// Set coefficient of x^n from a `FlintInteger` (supports values beyond i64 range).
    ///
    /// # Panics
    ///
    /// When `n` is past the longest coefficient array that can be addressed
    /// ([`MAX_COEFF_INDEX`]). FLINT would otherwise receive `n` as a negative
    /// `slong` and write outside the array, or fail its allocation and
    /// `abort()`.
    pub fn set_coeff_flint(&mut self, n: usize, c: &super::integer::FlintInteger) {
        check_coeff_index(n);
        unsafe { ffi::fmpz_poly_set_coeff_fmpz(&mut self.inner, n as ffi::slong, c.inner_ptr()) };
    }

    /// Get coefficient of x^n as a `FlintInteger`. Zero past the length.
    pub fn get_coeff_flint(&self, n: usize) -> super::integer::FlintInteger {
        let mut c = super::integer::FlintInteger::new();
        if n >= self.length() {
            return c;
        }
        unsafe { ffi::fmpz_poly_get_coeff_fmpz(c.inner_mut_ptr(), &self.inner, n as ffi::slong) };
        c
    }

    /// Discriminant of the polynomial (`fmpz_poly_discriminant`).
    ///
    /// This is the discriminant of *this polynomial*, `(-1)^(d(d-1)/2) ·
    /// res(f, f') / lc(f)`. For the defining polynomial of a number field that
    /// is **not** the same thing as the discriminant of the field: they differ
    /// by the square of the index `[O_K : Z[a]]`. Zero for degree < 1.
    pub fn discriminant(&self) -> super::integer::FlintInteger {
        let mut res = super::integer::FlintInteger::new();
        unsafe { ffi::fmpz_poly_discriminant(res.inner_mut_ptr(), &self.inner) };
        res
    }

    /// Raw pointer to the underlying `fmpz_poly_struct`, for the few callers
    /// outside this module that hand a polynomial straight to FLINT.
    pub(crate) fn inner_ptr(&self) -> *const ffi::FmpzPolyStruct {
        &self.inner
    }

    /// Formal derivative: `d/dx [c₀ + c₁x + … + cₙxⁿ] = c₁ + 2c₂x + … + ncₙxⁿ⁻¹`.
    ///
    /// One `fmpz_poly_derivative` call; the coefficients never leave FLINT.
    pub fn derivative(&self) -> Self {
        let mut result = Self::new();
        // SAFETY: both structs are initialised and owned; `result` is distinct
        // from `self`.
        unsafe { ffi::fmpz_poly_derivative(&mut result.inner, &self.inner) };
        result
    }

    /// Construct a polynomial from a slice of [`rug::Integer`] coefficients
    /// in ascending degree order.
    pub fn from_rug_coefficients(coeffs: &[rug::Integer]) -> Self {
        let mut p = Self::new();
        for (i, c) in coeffs.iter().enumerate() {
            let fi = super::integer::FlintInteger::from_rug(c);
            p.set_coeff_flint(i, &fi);
        }
        p
    }

    /// Complete factorization over ℤ via FLINT (`fmpz_poly_factor` — modular
    /// Berlekamp, Zassenhaus combination, van Hoeij LLL, etc.).
    ///
    /// Returns `(unit, factors)` with `self = unit · ∏ fᵢ^eᵢ`.  The zero
    /// polynomial yields `Err(())`. `unit` carries the content and the sign
    /// of the leading coefficient; every factor is primitive with a positive
    /// leading coefficient.
    ///
    /// # Factor order
    ///
    /// The factors are listed in the canonical order of
    /// [`crate::poly::factor::cmp_univariate_factors`]: ascending degree, then
    /// the coefficients compared as signed integers from the leading term
    /// down, then multiplicity. The order is a function of the factors alone,
    /// so it does not depend on the FLINT version (FLINT's own order comes
    /// out of its LLL-based recombination and changed between 3.5 and 3.6).
    ///
    /// Binomials `a·x^(k+n) + b·x^k` whose primitive part is `u^n·x^n ± v^n`
    /// (in particular `x^n ± 1`) are factored directly through cyclotomic
    /// polynomials instead of by FLINT, with identical factors and unit.
    #[allow(clippy::result_unit_err)]
    pub fn factor_over_z(
        &self,
    ) -> Result<(super::integer::FlintInteger, Vec<(FlintPoly, u32)>), ()> {
        let (unit, mut factors) = match crate::poly::factor::binomial::factor_binomial_z(self) {
            Some(fast) => fast,
            None => self.factor_over_z_flint_order()?,
        };
        factors.sort_by(crate::poly::factor::cmp_univariate_factors);
        Ok((unit, factors))
    }

    /// [`Self::factor_over_z`] straight from `fmpz_poly_factor`, in FLINT's
    /// order and without the binomial fast path (the reference the fast path
    /// is tested against).
    #[allow(clippy::result_unit_err)]
    pub(crate) fn factor_over_z_flint_order(
        &self,
    ) -> Result<(super::integer::FlintInteger, Vec<(FlintPoly, u32)>), ()> {
        if self.is_zero() {
            return Err(());
        }
        // FlintPolyFactor is drop-safe: no manual fmpz_poly_factor_clear needed.
        let mut fac = FlintPolyFactor::new();
        fac.factor(self);
        let unit = fac.unit();
        let factors = (0..fac.len())
            .map(|i| (fac.poly_at(i), fac.exp_at(i)))
            .collect();
        Ok((unit, factors))
    }

    /// Compare in the canonical factor order: degree first, then the
    /// coefficients as signed integers from the leading term down.
    pub(crate) fn cmp_canonical(&self, other: &Self) -> std::cmp::Ordering {
        let (la, lb) = (self.length(), other.length());
        if la != lb {
            return la.cmp(&lb);
        }
        for i in (0..la).rev() {
            // SAFETY: `i < length` for both polynomials, so both pointers
            // address initialised coefficients.
            let c = unsafe { ffi::fmpz_cmp(self.inner.coeffs.add(i), other.inner.coeffs.add(i)) };
            if c != 0 {
                return c.cmp(&0);
            }
        }
        std::cmp::Ordering::Equal
    }

    /// `Some((k, n))` when `self = a·x^(k+n) + b·x^k` with `a, b ≠ 0` and
    /// `n ≥ 1` — exactly two non-zero coefficients.
    pub(crate) fn binomial_shape(&self) -> Option<(usize, usize)> {
        let len = self.length();
        if len < 2 {
            return None;
        }
        // SAFETY: the coefficient array holds `len` initialised `fmpz`; an
        // `fmpz` is zero exactly when its word is 0.
        let coeffs = unsafe { std::slice::from_raw_parts(self.inner.coeffs, len) };
        let k = coeffs.iter().position(|&c| c != 0)?;
        let top = len - 1;
        if k == top || coeffs[k + 1..top].iter().any(|&c| c != 0) {
            return None;
        }
        Some((k, top - k))
    }

    /// Swinnerton–Dyer polynomial `S_n` (irreducible over ℚ for prime `n`).
    pub fn swinnerton_dyer(n: u64) -> Self {
        let mut p = Self::new();
        unsafe {
            ffi::fmpz_poly_swinnerton_dyer(&mut p.inner, n as ffi::ulong);
        }
        p
    }

    /// Cyclotomic polynomial `Φ_n`.
    pub fn cyclotomic(n: u64) -> Self {
        let mut p = Self::new();
        unsafe {
            ffi::fmpz_poly_cyclotomic(&mut p.inner, n as ffi::ulong);
        }
        p
    }
}

impl Default for FlintPoly {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FlintPoly {
    fn drop(&mut self) {
        unsafe { ffi::fmpz_poly_clear(&mut self.inner) };
    }
}

impl Clone for FlintPoly {
    fn clone(&self) -> Self {
        let mut new = Self::new();
        unsafe { ffi::fmpz_poly_set(&mut new.inner, &self.inner) };
        new
    }
}

impl PartialEq for FlintPoly {
    fn eq(&self, other: &Self) -> bool {
        unsafe { ffi::fmpz_poly_equal(&self.inner, &other.inner) != 0 }
    }
}
impl Eq for FlintPoly {}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

impl Add for FlintPoly {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        &self + &rhs
    }
}
impl<'b> Add<&'b FlintPoly> for &FlintPoly {
    type Output = FlintPoly;
    fn add(self, rhs: &'b FlintPoly) -> FlintPoly {
        let mut res = FlintPoly::new();
        unsafe { ffi::fmpz_poly_add(&mut res.inner, &self.inner, &rhs.inner) };
        res
    }
}

impl Sub for FlintPoly {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        &self - &rhs
    }
}
impl<'b> Sub<&'b FlintPoly> for &FlintPoly {
    type Output = FlintPoly;
    fn sub(self, rhs: &'b FlintPoly) -> FlintPoly {
        let mut res = FlintPoly::new();
        unsafe { ffi::fmpz_poly_sub(&mut res.inner, &self.inner, &rhs.inner) };
        res
    }
}

impl Mul for FlintPoly {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}
impl<'b> Mul<&'b FlintPoly> for &FlintPoly {
    type Output = FlintPoly;
    fn mul(self, rhs: &'b FlintPoly) -> FlintPoly {
        let mut res = FlintPoly::new();
        unsafe { ffi::fmpz_poly_mul(&mut res.inner, &self.inner, &rhs.inner) };
        res
    }
}

// ---------------------------------------------------------------------------
// Display / Debug
// ---------------------------------------------------------------------------

impl fmt::Display for FlintPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        unsafe {
            let var = CString::new("x").unwrap();
            let ptr = ffi::fmpz_poly_get_str_pretty(&self.inner, var.as_ptr());
            if ptr.is_null() {
                return write!(f, "0");
            }
            let s = std::ffi::CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or("<utf8-err>")
                .to_owned();
            ffi::flint_free(ptr as *mut std::ffi::c_void);
            write!(f, "{}", s)
        }
    }
}

impl fmt::Debug for FlintPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlintPoly({})", self)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- out-of-range indices and zero divisors ---------------------------
    //
    // Each of these used to reach FLINT: a `usize` index past `i64::MAX`
    // became a negative `slong` (an out-of-bounds read, or a heap-corrupting
    // write), and a zero divisor made FLINT `abort()` the process.

    #[test]
    fn get_coeff_past_the_end_is_zero() {
        let p = FlintPoly::from_coefficients(&[1, 2, 3]);
        for n in [3, 4, 1 << 40, i64::MAX as usize, usize::MAX] {
            assert_eq!(p.get_coeff(n), 0, "{n}");
            assert!(p.get_coeff_flint(n).is_zero(), "{n}");
        }
        assert_eq!(p.get_coeff(2), 3);
        assert_eq!(p.get_coeff_flint(1).to_i64(), 2);
    }

    #[test]
    #[should_panic(expected = "past the largest addressable polynomial length")]
    fn set_coeff_at_usize_max_panics_instead_of_corrupting_the_heap() {
        let mut p = FlintPoly::from_coefficients(&[1, 2, 3]);
        p.set_coeff_flint(
            usize::MAX,
            &super::super::integer::FlintInteger::from_i64(5),
        );
    }

    #[test]
    #[should_panic(expected = "past the largest addressable polynomial length")]
    fn set_coeff_past_i64_max_panics() {
        let mut p = FlintPoly::new();
        p.set_coeff_flint(
            i64::MAX as usize + 1,
            &super::super::integer::FlintInteger::from_i64(5),
        );
    }

    #[test]
    #[should_panic(expected = "zero polynomial")]
    fn div_exact_by_zero_panics_instead_of_aborting() {
        let p = FlintPoly::from_coefficients(&[1, 2, 3]);
        let _ = p.div_exact(&FlintPoly::new());
    }

    #[test]
    #[should_panic(expected = "zero polynomial")]
    fn pseudo_divrem_by_zero_panics_instead_of_aborting() {
        let p = FlintPoly::from_coefficients(&[1, 2, 3]);
        let _ = p.pseudo_divrem(&FlintPoly::new());
    }

    #[test]
    #[should_panic(expected = "divide a FlintPoly by zero")]
    fn scalar_divexact_by_zero_panics_instead_of_aborting() {
        let p = FlintPoly::from_coefficients(&[2, 4]);
        let _ = p.scalar_divexact_fmpz(&super::super::integer::FlintInteger::from_i64(0));
    }

    #[test]
    fn checked_div_exact() {
        let p = FlintPoly::from_coefficients(&[-1, 0, 1]);
        let d = FlintPoly::from_coefficients(&[1, 1]);
        assert!(p.checked_div_exact(&FlintPoly::new()).is_none());
        assert_eq!(
            p.checked_div_exact(&d).unwrap(),
            FlintPoly::from_coefficients(&[-1, 1])
        );
    }

    #[test]
    fn nmod_get_coeff_negative_index_is_zero() {
        let mut p = super::super::nmod::FlintNmodPoly::new(7);
        p.set_coeff(0, 3);
        p.set_coeff(1, 5);
        assert_eq!(p.get_coeff(-1), 0);
        assert_eq!(p.get_coeff(i64::MIN), 0);
        assert_eq!(p.get_coeff(1), 5);
        assert_eq!(p.get_coeff(9), 0);
    }

    #[test]
    #[should_panic(expected = "past the largest addressable polynomial length")]
    fn nmod_set_coeff_at_usize_max_panics() {
        let mut p = super::super::nmod::FlintNmodPoly::new(7);
        p.set_coeff(usize::MAX, 1);
    }

    // --- construction ---

    #[test]
    fn zero_poly() {
        let p = FlintPoly::new();
        assert!(p.is_zero());
        assert_eq!(p.length(), 0);
        assert_eq!(p.degree(), -1);
        assert_eq!(p.coefficients(), Vec::<i64>::new());
    }

    #[test]
    fn from_coefficients_roundtrip() {
        let coeffs = vec![1i64, 2, 3];
        let p = FlintPoly::from_coefficients(&coeffs);
        assert_eq!(p.coefficients(), coeffs);
        assert_eq!(p.degree(), 2);
        assert_eq!(p.length(), 3);
    }

    #[test]
    fn from_coefficients_zero_trailing() {
        // FLINT normalises trailing zeros: [1,0] has degree 0, length 1
        let p = FlintPoly::from_coefficients(&[1, 0]);
        assert_eq!(p.degree(), 0);
        assert_eq!(p.length(), 1);
    }

    #[test]
    fn constant_poly() {
        let p = FlintPoly::from_coefficients(&[42]);
        assert_eq!(p.coefficients(), vec![42]);
        assert_eq!(p.degree(), 0);
    }

    // --- equality ---

    #[test]
    fn equality() {
        let a = FlintPoly::from_coefficients(&[1, 2, 3]);
        let b = FlintPoly::from_coefficients(&[1, 2, 3]);
        let c = FlintPoly::from_coefficients(&[1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn clone_is_independent() {
        let a = FlintPoly::from_coefficients(&[1, 2, 3]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    // --- arithmetic ---

    #[test]
    fn add() {
        // (1 + 2x) + (3 + 4x) = 4 + 6x
        let a = FlintPoly::from_coefficients(&[1, 2]);
        let b = FlintPoly::from_coefficients(&[3, 4]);
        let s = &a + &b;
        assert_eq!(s.coefficients(), vec![4, 6]);
    }

    #[test]
    fn sub() {
        let a = FlintPoly::from_coefficients(&[5, 3]);
        let b = FlintPoly::from_coefficients(&[2, 1]);
        let d = &a - &b;
        assert_eq!(d.coefficients(), vec![3, 2]);
    }

    #[test]
    fn mul() {
        // (1 + x) * (1 + x) = 1 + 2x + x²
        let p = FlintPoly::from_coefficients(&[1, 1]);
        let q = &p * &p;
        assert_eq!(q.coefficients(), vec![1, 2, 1]);
    }

    #[test]
    fn mul_non_trivial() {
        // (1 + 2x + 3x²) * (4 + 5x) = 4 + 13x + 22x² + 15x³
        let a = FlintPoly::from_coefficients(&[1, 2, 3]);
        let b = FlintPoly::from_coefficients(&[4, 5]);
        let c = &a * &b;
        assert_eq!(c.coefficients(), vec![4, 13, 22, 15]);
    }

    #[test]
    fn mul_by_zero() {
        let a = FlintPoly::from_coefficients(&[1, 2, 3]);
        let z = FlintPoly::new();
        assert!((&a * &z).is_zero());
    }

    #[test]
    fn pow_squared() {
        // (x + 1)^2 = x^2 + 2x + 1
        let p = FlintPoly::from_coefficients(&[1, 1]);
        let q = p.pow(2);
        assert_eq!(q.coefficients(), vec![1, 2, 1]);
    }

    #[test]
    fn pow_of_a_monomial_skips_the_binomial_expansion() {
        // (-3x^2)^5 = -243 x^10; (x + 1)^3 still goes through FLINT.
        let q = FlintPoly::from_coefficients(&[0, 0, -3]).pow(5);
        let mut want = vec![0; 11];
        want[10] = -243;
        assert_eq!(q.coefficients(), want);
        assert_eq!(
            FlintPoly::from_coefficients(&[1, 1]).pow(3).coefficients(),
            vec![1, 3, 3, 1]
        );
        // x^(2^21): one coefficient, not 2^21 binomial coefficients (which
        // FLINT's length-2 path built, taking gigabytes).
        let big = FlintPoly::from_coefficients(&[0, 1]).pow(1 << 21);
        assert_eq!(big.degree(), 1 << 21);
        assert_eq!(big.get_coeff(1 << 21), 1);
        assert_eq!(big.get_coeff(0), 0);
    }

    #[test]
    fn pow_zero() {
        let p = FlintPoly::from_coefficients(&[1, 2, 3]);
        // p^0 = 1 (the constant polynomial 1)
        let q = p.pow(0);
        assert_eq!(q.coefficients(), vec![1]);
    }

    #[test]
    fn pow_cubed() {
        // (1 + x)^3 = 1 + 3x + 3x^2 + x^3
        let p = FlintPoly::from_coefficients(&[1, 1]);
        let q = p.pow(3);
        assert_eq!(q.coefficients(), vec![1, 3, 3, 1]);
    }

    #[test]
    fn gcd_trivial() {
        // gcd(x^2 - 1, x - 1) = x - 1 (up to leading coeff sign)
        let a = FlintPoly::from_coefficients(&[-1, 0, 1]); // x^2 - 1
        let b = FlintPoly::from_coefficients(&[-1, 1]); // x - 1
        let g = a.gcd(&b);
        // FLINT normalises to positive leading coefficient
        assert_eq!(g.degree(), 1);
        let coeffs = g.coefficients();
        // Either [1, -1] or [-1, 1] scaled; assert ratio
        assert_eq!(coeffs[1].abs(), coeffs[0].abs());
        assert_ne!(coeffs[0], 0);
    }

    #[test]
    fn gcd_coprime() {
        // gcd(x, x+1) = 1
        let a = FlintPoly::from_coefficients(&[0, 1]);
        let b = FlintPoly::from_coefficients(&[1, 1]);
        let g = a.gcd(&b);
        assert_eq!(g.degree(), 0); // constant
    }

    #[test]
    fn gcd_with_zero() {
        let a = FlintPoly::from_coefficients(&[3, 2, 1]);
        let z = FlintPoly::new();
        // gcd(p, 0) = p (up to units)
        let g = a.gcd(&z);
        assert_eq!(g.degree(), a.degree());
    }

    // --- display ---

    #[test]
    fn display_zero() {
        assert_eq!(FlintPoly::new().to_string(), "0");
    }

    #[test]
    fn display_constant() {
        let p = FlintPoly::from_coefficients(&[5]);
        assert_eq!(p.to_string(), "5");
    }

    #[test]
    fn display_linear() {
        let p = FlintPoly::from_coefficients(&[0, 1]);
        assert_eq!(p.to_string(), "x");
    }

    #[test]
    fn display_quadratic() {
        // 1 + 2x + x^2 → FLINT pretty-prints as "x^2+2*x+1"
        let p = FlintPoly::from_coefficients(&[1, 2, 1]);
        let s = p.to_string();
        // Don't assert exact spacing (FLINT may vary); just check it's non-empty
        // and contains "x^2".
        assert!(s.contains("x^2"), "unexpected display: {s}");
    }

    // --- coefficient round-trip ---

    #[test]
    fn coefficient_round_trip() {
        let orig = vec![10i64, -5, 0, 3, 1];
        let p = FlintPoly::from_coefficients(&orig);
        // FLINT drops trailing zeros; the last 1 is leading, so full round-trip
        assert_eq!(p.coefficients(), orig);
    }
}
