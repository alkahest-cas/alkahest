use super::ffi;
use rug::Complete;
use std::ffi::CString;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Safe wrapper over FLINT's `fmpz_t` — arbitrary-precision integer.
///
/// Memory is managed by FLINT's allocator. `Drop` calls `fmpz_clear`.
/// All raw pointers are confined to this file; callers see only safe Rust.
pub struct FlintInteger {
    /// The `fmpz` storage. Either an inline i64 or a tagged pointer to GMP
    /// memory managed by FLINT. Must never be aliased across two `FlintInteger`
    /// values.
    inner: ffi::fmpz,
}

// SAFETY: fmpz is either an i64 or owns its GMP memory. No shared state.
unsafe impl Send for FlintInteger {}
unsafe impl Sync for FlintInteger {}

impl FlintInteger {
    pub fn new() -> Self {
        let mut inner: ffi::fmpz = 0;
        unsafe { ffi::fmpz_init(&mut inner) };
        FlintInteger { inner }
    }

    pub fn from_i64(val: i64) -> Self {
        let mut f = Self::new();
        unsafe { ffi::fmpz_set_si(&mut f.inner, val) };
        f
    }

    /// Return as `i64`. For values that overflow i64 this wraps/truncates —
    /// use `to_string()` for a lossless decimal representation.
    pub fn to_i64(&self) -> i64 {
        unsafe { ffi::fmpz_get_si(&self.inner) }
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_gcd(&mut res.inner, &self.inner, &other.inner) };
        res
    }

    pub fn pow(&self, exp: u64) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_pow_ui(&mut res.inner, &self.inner, exp) };
        res
    }

    /// Construct from a `rug::Integer` via decimal string round-trip.
    pub fn from_rug(n: &rug::Integer) -> Self {
        let s = n.to_string();
        let cstr = CString::new(s.as_str()).unwrap();
        let mut f = Self::new();
        unsafe { ffi::fmpz_set_str(&mut f.inner, cstr.as_ptr(), 10) };
        f
    }

    /// Expose the raw inner `fmpz` for use by `FlintPoly` coefficient accessors.
    pub(crate) fn inner_ptr(&self) -> *const ffi::fmpz {
        &self.inner
    }

    pub(crate) fn inner_mut_ptr(&mut self) -> *mut ffi::fmpz {
        &mut self.inner
    }

    /// Convert to a `rug::Integer` for cross-validation in tests.
    pub fn to_rug(&self) -> rug::Integer {
        rug::Integer::parse_radix(self.to_string().as_bytes(), 10)
            .unwrap()
            .complete()
    }
}

impl Default for FlintInteger {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FlintInteger {
    fn drop(&mut self) {
        unsafe { ffi::fmpz_clear(&mut self.inner) };
    }
}

impl Clone for FlintInteger {
    fn clone(&self) -> Self {
        let mut new = Self::new();
        unsafe { ffi::fmpz_set(&mut new.inner, &self.inner) };
        new
    }
}

impl PartialEq for FlintInteger {
    fn eq(&self, other: &Self) -> bool {
        unsafe { ffi::fmpz_equal(&self.inner, &other.inner) != 0 }
    }
}
impl Eq for FlintInteger {}

// ---------------------------------------------------------------------------
// Arithmetic — owned and reference variants
// ---------------------------------------------------------------------------

impl Add for FlintInteger {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        &self + &rhs
    }
}
impl<'b> Add<&'b FlintInteger> for &FlintInteger {
    type Output = FlintInteger;
    fn add(self, rhs: &'b FlintInteger) -> FlintInteger {
        let mut res = FlintInteger::new();
        unsafe { ffi::fmpz_add(&mut res.inner, &self.inner, &rhs.inner) };
        res
    }
}

impl Sub for FlintInteger {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        &self - &rhs
    }
}
impl<'b> Sub<&'b FlintInteger> for &FlintInteger {
    type Output = FlintInteger;
    fn sub(self, rhs: &'b FlintInteger) -> FlintInteger {
        let mut res = FlintInteger::new();
        unsafe { ffi::fmpz_sub(&mut res.inner, &self.inner, &rhs.inner) };
        res
    }
}

impl Mul for FlintInteger {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        &self * &rhs
    }
}
impl<'b> Mul<&'b FlintInteger> for &FlintInteger {
    type Output = FlintInteger;
    fn mul(self, rhs: &'b FlintInteger) -> FlintInteger {
        let mut res = FlintInteger::new();
        unsafe { ffi::fmpz_mul(&mut res.inner, &self.inner, &rhs.inner) };
        res
    }
}

/// Truncated (toward-zero) division, matching Rust's built-in integer `/`.
impl Div for FlintInteger {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        &self / &rhs
    }
}
impl<'b> Div<&'b FlintInteger> for &FlintInteger {
    type Output = FlintInteger;
    fn div(self, rhs: &'b FlintInteger) -> FlintInteger {
        let mut res = FlintInteger::new();
        unsafe { ffi::fmpz_tdiv_q(&mut res.inner, &self.inner, &rhs.inner) };
        res
    }
}

/// Remainder after truncated division, matching Rust's built-in `%`.
impl Rem for FlintInteger {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        &self % &rhs
    }
}
impl<'b> Rem<&'b FlintInteger> for &FlintInteger {
    type Output = FlintInteger;
    fn rem(self, rhs: &'b FlintInteger) -> FlintInteger {
        let mut q = FlintInteger::new();
        let mut r = FlintInteger::new();
        unsafe { ffi::fmpz_tdiv_qr(&mut q.inner, &mut r.inner, &self.inner, &rhs.inner) };
        r
    }
}

impl Neg for FlintInteger {
    type Output = Self;
    fn neg(self) -> Self {
        -&self
    }
}
impl Neg for &FlintInteger {
    type Output = FlintInteger;
    fn neg(self) -> FlintInteger {
        let mut res = FlintInteger::new();
        unsafe { ffi::fmpz_neg(&mut res.inner, &self.inner) };
        res
    }
}

// ---------------------------------------------------------------------------
// Display / Debug
// ---------------------------------------------------------------------------

impl fmt::Display for FlintInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // fmpz_get_str(NULL, base, f) allocates a new C string; caller frees
        // with flint_free.
        unsafe {
            let ptr = ffi::fmpz_get_str(std::ptr::null_mut(), 10, &self.inner);
            if ptr.is_null() {
                return write!(f, "<err>");
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

impl fmt::Debug for FlintInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlintInteger({})", self)
    }
}

// ---------------------------------------------------------------------------
// FlintIntFactor — drop-safe factorisation container for fmpz integers
// ---------------------------------------------------------------------------

/// Owned `fmpz_factor_t`.  `Drop` calls `fmpz_factor_clear`.
pub(crate) struct FlintIntFactor {
    inner: ffi::FmpzFactorStruct,
}

impl FlintIntFactor {
    pub fn new() -> Self {
        let mut inner = std::mem::MaybeUninit::<ffi::FmpzFactorStruct>::uninit();
        unsafe { ffi::fmpz_factor_init(inner.as_mut_ptr()) };
        // SAFETY: `fmpz_factor_init` fully initialises the struct.
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    /// Factor `n` into this container.
    pub fn factor(&mut self, n: &FlintInteger) {
        unsafe { ffi::fmpz_factor(&mut self.inner, n.inner_ptr()) };
    }

    /// Sign of the factored integer (`1` or `-1`).
    pub fn sign(&self) -> i32 {
        self.inner.sign
    }

    /// Number of distinct prime factors.
    pub fn len(&self) -> usize {
        self.inner.num.max(0) as usize
    }

    /// The `i`-th prime base as a [`FlintInteger`].
    pub fn base_at(&self, i: usize) -> FlintInteger {
        debug_assert!(i < self.len());
        let mut f = FlintInteger::new();
        // SAFETY: `i < num` so the pointer is in bounds.
        unsafe { ffi::fmpz_set(f.inner_mut_ptr(), self.inner.p.add(i)) };
        f
    }

    /// Exponent of the `i`-th prime factor.
    pub fn exp_at(&self, i: usize) -> u64 {
        debug_assert!(i < self.len());
        unsafe { *self.inner.exp.add(i) }
    }
}

impl Drop for FlintIntFactor {
    fn drop(&mut self) {
        // SAFETY: `self.inner` was initialised by `fmpz_factor_init` in `new`.
        unsafe { ffi::fmpz_factor_clear(&mut self.inner) };
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- construction and equality ---

    #[test]
    fn zero() {
        let z = FlintInteger::new();
        assert_eq!(z, FlintInteger::from_i64(0));
    }

    #[test]
    fn from_i64_roundtrip() {
        for v in [-1000i64, -1, 0, 1, 1000, i64::MAX / 2] {
            let f = FlintInteger::from_i64(v);
            assert_eq!(f.to_i64(), v);
        }
    }

    #[test]
    fn clone_is_independent() {
        let a = FlintInteger::from_i64(42);
        let b = a.clone();
        assert_eq!(a, b);
        // modifying b via arithmetic should not affect a
        let c = &b + &FlintInteger::from_i64(1);
        assert_eq!(a, FlintInteger::from_i64(42));
        assert_eq!(c, FlintInteger::from_i64(43));
    }

    // --- arithmetic ---

    #[test]
    fn add() {
        let a = FlintInteger::from_i64(7);
        let b = FlintInteger::from_i64(5);
        assert_eq!((&a + &b).to_i64(), 12);
    }

    #[test]
    fn sub() {
        let a = FlintInteger::from_i64(7);
        let b = FlintInteger::from_i64(5);
        assert_eq!((&a - &b).to_i64(), 2);
    }

    #[test]
    fn mul() {
        let a = FlintInteger::from_i64(7);
        let b = FlintInteger::from_i64(5);
        assert_eq!((&a * &b).to_i64(), 35);
    }

    #[test]
    fn div_truncated() {
        let a = FlintInteger::from_i64(7);
        let b = FlintInteger::from_i64(3);
        assert_eq!((&a / &b).to_i64(), 2); // truncated toward zero
        let c = FlintInteger::from_i64(-7);
        assert_eq!((&c / &b).to_i64(), -2); // negative: truncates toward zero
    }

    #[test]
    fn rem() {
        let a = FlintInteger::from_i64(7);
        let b = FlintInteger::from_i64(3);
        assert_eq!((&a % &b).to_i64(), 1);
    }

    #[test]
    fn neg() {
        let a = FlintInteger::from_i64(5);
        assert_eq!((-&a).to_i64(), -5);
        assert_eq!((-FlintInteger::from_i64(-3)).to_i64(), 3);
    }

    #[test]
    fn gcd() {
        let a = FlintInteger::from_i64(12);
        let b = FlintInteger::from_i64(8);
        assert_eq!(a.gcd(&b).to_i64(), 4);
        let p = FlintInteger::from_i64(17);
        let q = FlintInteger::from_i64(5);
        assert_eq!(p.gcd(&q).to_i64(), 1); // coprime
    }

    #[test]
    fn pow() {
        let a = FlintInteger::from_i64(2);
        assert_eq!(a.pow(10).to_i64(), 1024);
        assert_eq!(a.pow(0).to_i64(), 1);
    }

    // --- display ---

    #[test]
    fn display() {
        assert_eq!(FlintInteger::from_i64(0).to_string(), "0");
        assert_eq!(FlintInteger::from_i64(-42).to_string(), "-42");
        assert_eq!(FlintInteger::from_i64(1_000_000).to_string(), "1000000");
    }

    // --- cross-validation against rug ---

    #[test]
    fn roundtrip_vs_rug_small() {
        for v in [-999i64, -1, 0, 1, 999] {
            let flint = FlintInteger::from_i64(v);
            let rug_val = rug::Integer::from(v);
            assert_eq!(flint.to_string(), rug_val.to_string(), "mismatch for v={v}");
        }
    }

    #[test]
    fn arithmetic_vs_rug() {
        use rug::ops::DivRounding;
        let pairs: &[(i64, i64)] = &[(0, 0), (7, 5), (-12, 4), (100, 7), (1000, 999)];
        for &(a, b) in pairs {
            let fa = FlintInteger::from_i64(a);
            let fb = FlintInteger::from_i64(b);
            let ra = rug::Integer::from(a);
            let rb = rug::Integer::from(b);
            assert_eq!(
                (&fa + &fb).to_string(),
                rug::Integer::from(&ra + &rb).to_string(),
                "add {a}+{b}"
            );
            assert_eq!(
                (&fa - &fb).to_string(),
                rug::Integer::from(&ra - &rb).to_string(),
                "sub {a}-{b}"
            );
            assert_eq!(
                (&fa * &fb).to_string(),
                rug::Integer::from(&ra * &rb).to_string(),
                "mul {a}*{b}"
            );
            if b != 0 {
                let rug_div = ra.clone().div_trunc(rb.clone());
                assert_eq!((&fa / &fb).to_string(), rug_div.to_string(), "div {a}/{b}");
            }
        }
    }

    #[test]
    fn large_integer_vs_rug() {
        use rug::ops::Pow;
        // 2^100 — larger than i64, exercises GMP allocation path in fmpz
        let two = FlintInteger::from_i64(2);
        let big = two.pow(100);
        let rug_big = rug::Integer::from(2i64).pow(100u32);
        assert_eq!(big.to_string(), rug_big.to_string());
    }
}
