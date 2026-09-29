use super::ffi;
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

    /// `true` when this integer is zero.
    pub fn is_zero(&self) -> bool {
        // `fmpz_is_zero` is a header inline: an `fmpz` is zero exactly when
        // its word is zero (a heap-backed value is a tagged non-zero pointer).
        self.inner == 0
    }

    /// Truncated division `self / rhs`, or `None` when `rhs` is zero.
    ///
    /// The `/` operator panics on a zero divisor (FLINT would abort the
    /// process); this is the form for a divisor that came from user input.
    pub fn checked_div(&self, rhs: &Self) -> Option<Self> {
        (!rhs.is_zero()).then(|| self / rhs)
    }

    /// Remainder of truncated division, or `None` when `rhs` is zero.
    pub fn checked_rem(&self, rhs: &Self) -> Option<Self> {
        (!rhs.is_zero()).then(|| self % rhs)
    }

    pub fn pow(&self, exp: u64) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_pow_ui(&mut res.inner, &self.inner, exp) };
        res
    }

    /// Construct from a `rug::Integer` by copying its limbs.
    ///
    /// Word-sized values take `fmpz_set_si`; larger ones hand rug's limbs to
    /// `fmpz_set_ui_array` and fix the sign with `fmpz_neg`. Linear in the size
    /// of `n` — there is no decimal round-trip.
    pub fn from_rug(n: &rug::Integer) -> Self {
        let mut f = Self::new();
        // SAFETY: `f.inner` is an initialised `fmpz` owned by `f`.
        unsafe { fmpz_set_rug(&mut f.inner, n) };
        f
    }

    /// Expose the raw inner `fmpz` for use by `FlintPoly` coefficient accessors.
    pub(crate) fn inner_ptr(&self) -> *const ffi::fmpz {
        &self.inner
    }

    pub(crate) fn inner_mut_ptr(&mut self) -> *mut ffi::fmpz {
        &mut self.inner
    }

    /// Convert to a `rug::Integer` by copying limbs.
    ///
    /// Values that fit an `i64` take `fmpz_get_si`; larger ones are read out
    /// with `fmpz_get_ui_array` into a Rust-owned buffer that rug then copies
    /// from. Linear in the size of `self` — there is no decimal round-trip.
    pub fn to_rug(&self) -> rug::Integer {
        // SAFETY: `self.inner` is an initialised `fmpz`.
        unsafe { fmpz_get_rug(&self.inner) }
    }
}

// ---------------------------------------------------------------------------
// rug <-> fmpz limb conversion
// ---------------------------------------------------------------------------
//
// The process may hold two GMP copies: the one rug links (via gmp-mpfr-sys)
// and the one FLINT links. So no `mpz_t` ever crosses between the two sides:
// FLINT is never asked to allocate, grow or free memory rug owns, nor the
// reverse. Only plain limb arrays cross. rug's limbs are *read* (never written)
// and copied into FLINT by `fmpz_set_ui_array`; FLINT's limbs are written into
// a Rust-owned `Vec<u64>` by `fmpz_get_ui_array`, and rug copies from that.

/// Size in bytes of one element of `s` — lets the conversion check that rug's
/// limbs are 64-bit words like FLINT's without naming gmp-mpfr-sys' `limb_t`.
fn elem_size<T>(_: &[T]) -> usize {
    std::mem::size_of::<T>()
}

/// Set the initialised `fmpz` at `f` to `n`.
///
/// # Safety
/// `f` must point to an initialised `fmpz` that nothing else aliases.
pub(crate) unsafe fn fmpz_set_rug(f: *mut ffi::fmpz, n: &rug::Integer) {
    if let Some(v) = n.to_i64() {
        ffi::fmpz_set_si(f, v);
        return;
    }
    let limbs = n.as_limbs();
    if elem_size(limbs) == std::mem::size_of::<ffi::ulong>() {
        // SAFETY: a GMP limb is an unsigned integer type; at 8 bytes its layout
        // and alignment are those of `u64`. The slice is only read, and
        // `fmpz_set_ui_array` copies it into FLINT-owned memory.
        ffi::fmpz_set_ui_array(
            f,
            limbs.as_ptr().cast::<ffi::ulong>(),
            limbs.len() as ffi::slong,
        );
    } else {
        let words = n.to_digits::<u64>(rug::integer::Order::Lsf);
        ffi::fmpz_set_ui_array(f, words.as_ptr(), words.len() as ffi::slong);
    }
    if n.cmp0() == std::cmp::Ordering::Less {
        ffi::fmpz_neg(f, f);
    }
}

/// Read the initialised `fmpz` at `f` as a `rug::Integer`.
///
/// # Safety
/// `f` must point to an initialised `fmpz`.
pub(crate) unsafe fn fmpz_get_rug(f: *const ffi::fmpz) -> rug::Integer {
    if ffi::fmpz_fits_si(f) != 0 {
        return rug::Integer::from(ffi::fmpz_get_si(f));
    }
    // `fmpz_get_ui_array` reads a nonnegative input, so take |f| first.
    let negative = ffi::fmpz_cmp_si(f, 0) < 0;
    let mut abs = FlintInteger::new();
    ffi::fmpz_abs(&mut abs.inner, f);
    let len = ffi::fmpz_bits(&abs.inner).div_ceil(64) as usize;
    let mut words = vec![0u64; len];
    ffi::fmpz_get_ui_array(words.as_mut_ptr(), len as ffi::slong, &abs.inner);
    let r = rug::Integer::from_digits(&words, rug::integer::Order::Lsf);
    if negative {
        -r
    } else {
        r
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
    /// # Panics
    ///
    /// On division by zero, as Rust's built-in `/` does. FLINT itself would
    /// `abort()` the process; use [`FlintInteger::checked_div`] for a
    /// non-panicking form.
    fn div(self, rhs: &'b FlintInteger) -> FlintInteger {
        assert!(!rhs.is_zero(), "attempt to divide a FlintInteger by zero");
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
    /// # Panics
    ///
    /// On a zero divisor, as Rust's built-in `%` does. FLINT itself would
    /// `abort()` the process; use [`FlintInteger::checked_rem`] for a
    /// non-panicking form.
    fn rem(self, rhs: &'b FlintInteger) -> FlintInteger {
        assert!(
            !rhs.is_zero(),
            "attempt to calculate the remainder of a FlintInteger with a divisor of zero"
        );
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
        // A real check, not `debug_assert!`: this is a safe fn, and an index
        // past `len()` reads out of bounds of the FLINT array in release.
        assert!(i < self.len(), "factor index {i} out of range (len {})", self.len());
        let mut f = FlintInteger::new();
        // SAFETY: `i < num` so the pointer is in bounds.
        unsafe { ffi::fmpz_set(f.inner_mut_ptr(), self.inner.p.add(i)) };
        f
    }

    /// Exponent of the `i`-th prime factor.
    pub fn exp_at(&self, i: usize) -> u64 {
        // A real check, not `debug_assert!`: this is a safe fn, and an index
        // past `len()` reads out of bounds of the FLINT array in release.
        assert!(i < self.len(), "factor index {i} out of range (len {})", self.len());
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

    // --- zero divisors (FLINT aborts the process; these must not reach it) ---

    #[test]
    #[should_panic(expected = "divide a FlintInteger by zero")]
    fn division_by_zero_panics_instead_of_aborting() {
        let _ = FlintInteger::from_i64(7) / FlintInteger::from_i64(0);
    }

    #[test]
    #[should_panic(expected = "divisor of zero")]
    fn remainder_by_zero_panics_instead_of_aborting() {
        let _ = FlintInteger::from_i64(7) % FlintInteger::from_i64(0);
    }

    #[test]
    fn checked_division() {
        let a = FlintInteger::from_i64(-7);
        let zero = FlintInteger::from_i64(0);
        assert!(a.checked_div(&zero).is_none());
        assert!(a.checked_rem(&zero).is_none());
        assert_eq!(
            a.checked_div(&FlintInteger::from_i64(2)).unwrap().to_i64(),
            -3
        );
        assert_eq!(
            a.checked_rem(&FlintInteger::from_i64(2)).unwrap().to_i64(),
            -1
        );
        assert!(zero.is_zero());
        assert!(!a.is_zero());
        let big = FlintInteger::from_rug(&(rug::Integer::from(1) << 200u32));
        assert!(!big.is_zero());
        assert!(big.checked_div(&zero).is_none());
    }

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

    // --- limb conversion vs the former decimal-string conversion ---

    /// The conversion `from_rug` used before limb copying: decimal string
    /// through `fmpz_set_str`. Kept as the oracle.
    fn from_rug_via_string(n: &rug::Integer) -> FlintInteger {
        let cstr = std::ffi::CString::new(n.to_string()).unwrap();
        let mut f = FlintInteger::new();
        assert_eq!(
            unsafe { ffi::fmpz_set_str(&mut f.inner, cstr.as_ptr(), 10) },
            0
        );
        f
    }

    /// The former `to_rug`: FLINT's decimal printer, parsed by rug.
    fn to_rug_via_string(f: &FlintInteger) -> rug::Integer {
        f.to_string().parse().unwrap()
    }

    fn check_roundtrip(n: &rug::Integer) {
        let f = FlintInteger::from_rug(n);
        assert_eq!(f, from_rug_via_string(n), "from_rug disagrees for {n}");
        assert_eq!(&f.to_rug(), n, "round trip lost {n}");
        assert_eq!(
            f.to_rug(),
            to_rug_via_string(&f),
            "to_rug disagrees for {n}"
        );
    }

    #[test]
    fn limb_conversion_edge_values() {
        use rug::Integer;
        let two64 = Integer::from(1) << 64u32;
        let mut cases = vec![
            Integer::new(),
            Integer::from(1),
            Integer::from(-1),
            Integer::from(i64::MIN),
            Integer::from(i64::MAX),
            Integer::from(i64::MIN) - 1u32,
            Integer::from(i64::MAX) + 1u32,
            Integer::from(u64::MAX),
            -Integer::from(u64::MAX),
            two64.clone(),
            -two64.clone(),
            two64.clone() - 1u32,
            -(two64.clone() + 1u32),
            // Whole limbs of ones, and a top limb with only its high bit set.
            (Integer::from(1) << 4096u32) - 1u32,
            Integer::from(1) << 4095u32,
            -(Integer::from(1) << 12_345u32),
        ];
        // Values FLINT stores inline (|x| < 2^62) and just past that bound.
        for k in [61u32, 62, 63] {
            cases.push(Integer::from(1) << k);
            cases.push(-(Integer::from(1) << k));
            cases.push((Integer::from(1) << k) - 1u32);
        }
        for n in &cases {
            check_roundtrip(n);
        }
    }

    /// A signed integer from its magnitude's 64-bit limbs.
    fn from_limbs(negative: bool, limbs: &[u64]) -> rug::Integer {
        let m = rug::Integer::from_digits(limbs, rug::integer::Order::Lsf);
        if negative {
            -m
        } else {
            m
        }
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_roundtrip_i64(v in any::<i64>()) {
            check_roundtrip(&rug::Integer::from(v));
        }

        #[test]
        fn prop_roundtrip_i128(v in any::<i128>()) {
            check_roundtrip(&rug::Integer::from(v));
        }

        /// Magnitudes from 0 to ~12 800 bits; leading zero limbs included.
        #[test]
        fn prop_roundtrip_multi_limb(
            negative in any::<bool>(),
            limbs in prop::collection::vec(any::<u64>(), 0..200),
        ) {
            check_roundtrip(&from_limbs(negative, &limbs));
        }

        /// Arithmetic done on the FLINT side reads back as rug's result.
        #[test]
        fn prop_product_matches_rug(
            a_neg in any::<bool>(),
            a in prop::collection::vec(any::<u64>(), 0..40),
            b_neg in any::<bool>(),
            b in prop::collection::vec(any::<u64>(), 0..40),
        ) {
            let (ra, rb) = (from_limbs(a_neg, &a), from_limbs(b_neg, &b));
            let prod = &FlintInteger::from_rug(&ra) * &FlintInteger::from_rug(&rb);
            prop_assert_eq!(prod.to_rug(), rug::Integer::from(&ra * &rb));
        }

        /// The same through polynomial coefficients.
        #[test]
        fn prop_poly_coefficients_roundtrip(
            coeffs in prop::collection::vec(
                (any::<bool>(), prop::collection::vec(any::<u64>(), 0..20)),
                0..8,
            ),
        ) {
            let rug_coeffs: Vec<rug::Integer> =
                coeffs.iter().map(|(neg, l)| from_limbs(*neg, l)).collect();
            let p = crate::flint::FlintPoly::from_rug_coefficients(&rug_coeffs);
            for (i, c) in rug_coeffs.iter().enumerate() {
                let got = p.get_coeff_flint(i);
                prop_assert_eq!(&got.to_rug(), c);
                prop_assert_eq!(got, from_rug_via_string(c));
            }
        }
    }

    /// Timing for the limb conversion against the former string conversion.
    /// Not run by default: `cargo test --release -p alkahest-cas --lib
    /// flint::integer::tests::conversion_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn conversion_timing() {
        use std::time::Instant;
        for bits in [1_000u32, 10_000, 100_000, 1_000_000] {
            let n = (rug::Integer::from(3) << bits) / 7u32 - 1u32;
            let reps = (2_000_000 / bits).max(3);
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(FlintInteger::from_rug(&n));
            }
            let limb_from = t.elapsed() / reps;
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(from_rug_via_string(&n));
            }
            let str_from = t.elapsed() / reps;
            let f = FlintInteger::from_rug(&n);
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(f.to_rug());
            }
            let limb_to = t.elapsed() / reps;
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(to_rug_via_string(&f));
            }
            let str_to = t.elapsed() / reps;
            println!(
                "{bits:>8} bits  from_rug: limb {limb_from:?} vs string {str_from:?}   \
                 to_rug: limb {limb_to:?} vs string {str_to:?}"
            );
        }
    }
}
