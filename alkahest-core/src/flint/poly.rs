use super::ffi;
use std::ffi::CString;
use std::fmt;
use std::ops::{Add, Mul, Sub};

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
        unsafe { ffi::fmpz_poly_init(&mut inner) };
        FlintPoly { inner }
    }

    /// Construct from coefficient slice in ascending degree order.
    /// `from_coefficients(&[1, 2, 3])` → `1 + 2x + 3x²`.
    pub fn from_coefficients(coeffs: &[i64]) -> Self {
        let mut p = Self::new();
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
        unsafe { ffi::fmpz_poly_get_coeff_si(&self.inner, n as ffi::slong) }
    }

    /// Coefficient vector in ascending degree order.
    pub fn coefficients(&self) -> Vec<i64> {
        (0..self.length()).map(|i| self.get_coeff(i)).collect()
    }

    pub fn is_zero(&self) -> bool {
        self.length() == 0
    }

    pub fn pow(&self, exp: u32) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_pow(&mut res.inner, &self.inner, exp as ffi::ulong) };
        res
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let mut res = Self::new();
        unsafe { ffi::fmpz_poly_gcd(&mut res.inner, &self.inner, &other.inner) };
        res
    }

    /// Exact polynomial division: returns `self / divisor`, assuming `divisor` divides `self`.
    pub fn div_exact(&self, divisor: &Self) -> Self {
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
    pub fn scalar_divexact_fmpz(&self, c: &super::integer::FlintInteger) -> Self {
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
    pub fn pseudo_divrem(&self, other: &Self) -> (Self, Self, u64) {
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
    pub fn set_coeff_flint(&mut self, n: usize, c: &super::integer::FlintInteger) {
        unsafe { ffi::fmpz_poly_set_coeff_fmpz(&mut self.inner, n as ffi::slong, c.inner_ptr()) };
    }

    /// Get coefficient of x^n as a `FlintInteger`.
    pub fn get_coeff_flint(&self, n: usize) -> super::integer::FlintInteger {
        let mut c = super::integer::FlintInteger::new();
        unsafe { ffi::fmpz_poly_get_coeff_fmpz(c.inner_mut_ptr(), &self.inner, n as ffi::slong) };
        c
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
