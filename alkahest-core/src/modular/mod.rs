//! V2-1 — Modular / CRT framework as a first-class primitive.
//!
//! Provides three core operations over sparse multivariate polynomials:
//!
//! - [`reduce_mod`] — reduce `f ∈ ℤ[x₁,…,xₙ]` to `F_p = ℤ/pℤ`
//! - [`lift_crt`] — reconstruct `f` from modular images via Chinese Remainder Theorem
//! - [`rational_reconstruction`] — recover `a/b` from `n ≡ b⁻¹·a (mod M)`
//!
//! Plus utilities used by higher-level algorithms (GCDs, factorization, Gröbner):
//!
//! - [`mignotte_bound`] — Cauchy–Mignotte coefficient bound
//! - [`select_lucky_prime`] — choose a prime that doesn't collapse the leading coefficient

use crate::errors::AlkahestError;
use crate::kernel::ExprId;
use crate::poly::MultiPoly;
use rug::Integer;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// MultiPolyFp — sparse multivariate polynomial over F_p = ℤ/pℤ
// ---------------------------------------------------------------------------

/// Sparse multivariate polynomial over the prime field `F_p = ℤ/pℤ`.
///
/// Coefficients are stored as `u64` in `[0, p)`.  The prime modulus is stored
/// alongside the polynomial so that callers can check consistency before
/// combining images with [`lift_crt`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MultiPolyFp {
    /// Variable identifiers — same ordering as the originating [`MultiPoly`].
    pub vars: Vec<ExprId>,
    /// The prime modulus `p`.
    pub modulus: u64,
    /// Exponent vector → coefficient in `[0, p)`.  Zero terms are never stored.
    pub terms: BTreeMap<Vec<u32>, u64>,
}

impl MultiPolyFp {
    pub fn zero(vars: Vec<ExprId>, modulus: u64) -> Self {
        MultiPolyFp {
            vars,
            modulus,
            terms: BTreeMap::new(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn total_degree(&self) -> u32 {
        self.terms
            .keys()
            .map(|e| e.iter().sum::<u32>())
            .max()
            .unwrap_or(0)
    }

    pub fn compatible_with(&self, other: &Self) -> bool {
        self.vars == other.vars && self.modulus == other.modulus
    }
}

impl std::fmt::Display for MultiPolyFp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "0 (mod {})", self.modulus);
        }
        let mut first = true;
        for (exp, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            write!(f, "{coeff}")?;
            for (i, &e) in exp.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                if e == 1 {
                    write!(f, "*x{i}")?;
                } else {
                    write!(f, "*x{i}^{e}")?;
                }
            }
        }
        write!(f, " (mod {})", self.modulus)
    }
}

// ---------------------------------------------------------------------------
// ModularValue — a tagged element of ℤ/pℤ for derivation traces
// ---------------------------------------------------------------------------

/// A single element of `ℤ/pℤ`, tagged with its modulus.
///
/// Used as a tracer value to tag which modular image produced a given
/// coefficient during GCD or resultant computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModularValue {
    /// The residue, in `[0, modulus)`.
    pub value: u64,
    /// The prime modulus.
    pub modulus: u64,
}

impl ModularValue {
    pub fn new(value: u64, modulus: u64) -> Self {
        debug_assert!(
            value < modulus,
            "ModularValue: value must be in [0, modulus)"
        );
        ModularValue { value, modulus }
    }

    pub fn zero(modulus: u64) -> Self {
        ModularValue { value: 0, modulus }
    }

    pub fn one(modulus: u64) -> Self {
        ModularValue {
            value: if modulus > 1 { 1 } else { 0 },
            modulus,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.modulus, other.modulus,
            "ModularValue: mismatched moduli"
        );
        let v = ((self.value as u128 + other.value as u128) % self.modulus as u128) as u64;
        ModularValue::new(v, self.modulus)
    }

    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.modulus, other.modulus,
            "ModularValue: mismatched moduli"
        );
        let v = (self.value + self.modulus - other.value % self.modulus) % self.modulus;
        ModularValue::new(v, self.modulus)
    }

    pub fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.modulus, other.modulus,
            "ModularValue: mismatched moduli"
        );
        let v = ((self.value as u128 * other.value as u128) % self.modulus as u128) as u64;
        ModularValue::new(v, self.modulus)
    }

    pub fn neg(&self) -> Self {
        if self.value == 0 {
            self.clone()
        } else {
            ModularValue::new(self.modulus - self.value, self.modulus)
        }
    }

    /// Multiplicative inverse. Returns `None` if `self.value == 0`.
    pub fn inverse(&self) -> Option<Self> {
        if self.value == 0 {
            return None;
        }
        Some(ModularValue::new(
            mod_inverse_u64(self.value, self.modulus),
            self.modulus,
        ))
    }
}

// ---------------------------------------------------------------------------
// ModularError
// ---------------------------------------------------------------------------

/// Error type for modular arithmetic operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ModularError {
    /// The given modulus is not a prime ≥ 2.
    InvalidModulus(u64),
    /// The input polynomials have incompatible variable lists or moduli.
    IncompatiblePolynomials,
    /// CRT lifting requires at least one modular image.
    EmptyImageList,
    /// Rational reconstruction failed: no `a/b` with small norm exists.
    ReconstructionFailed,
}

impl std::fmt::Display for ModularError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModularError::InvalidModulus(p) => {
                write!(f, "invalid modulus {p}: must be prime ≥ 2")
            }
            ModularError::IncompatiblePolynomials => {
                write!(f, "polynomials have incompatible variable lists or moduli")
            }
            ModularError::EmptyImageList => {
                write!(f, "CRT lifting requires at least one modular image")
            }
            ModularError::ReconstructionFailed => write!(
                f,
                "rational reconstruction failed: no a/b ≤ ⌊√(M/2)⌋ with a/b ≡ n (mod M)"
            ),
        }
    }
}

impl std::error::Error for ModularError {}

impl AlkahestError for ModularError {
    fn code(&self) -> &'static str {
        match self {
            ModularError::InvalidModulus(_) => "E-MOD-001",
            ModularError::IncompatiblePolynomials => "E-MOD-002",
            ModularError::EmptyImageList => "E-MOD-003",
            ModularError::ReconstructionFailed => "E-MOD-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            ModularError::InvalidModulus(_) => {
                Some("use a prime modulus p ≥ 2, e.g. 101, 1009, 32749")
            }
            ModularError::IncompatiblePolynomials => {
                Some("ensure all images share the same variable ordering and modulus")
            }
            ModularError::EmptyImageList => Some("provide at least one (MultiPolyFp, prime) pair"),
            ModularError::ReconstructionFailed => {
                Some("provide more modular images so the prime product M exceeds 2 * max_coeff²")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Reduce a polynomial over ℤ to a polynomial over `F_p = ℤ/pℤ`.
///
/// Each coefficient `c` is mapped to the representative in `[0, p)`.
/// Terms whose reduced coefficient is zero are dropped.
///
/// # Errors
///
/// Returns [`ModularError::InvalidModulus`] if `p` is not a prime ≥ 2.
pub fn reduce_mod(poly: &MultiPoly, p: u64) -> Result<MultiPolyFp, ModularError> {
    if !is_prime(p) {
        return Err(ModularError::InvalidModulus(p));
    }

    let mut terms = BTreeMap::new();
    for (exp, coeff) in &poly.terms {
        let c_mod = rug_mod_u64(coeff, p);
        if c_mod != 0 {
            terms.insert(exp.clone(), c_mod);
        }
    }

    Ok(MultiPolyFp {
        vars: poly.vars.clone(),
        modulus: p,
        terms,
    })
}

/// Reconstruct a polynomial over ℤ from modular images via the Chinese Remainder Theorem.
///
/// Given images `[(f mod p₁, p₁), …, (f mod pₖ, pₖ)]` with distinct primes `pᵢ`,
/// returns the unique polynomial `f` with coefficients centered in `(-M/2, M/2]`
/// where `M = p₁ · … · pₖ`.
///
/// All images must share the same variable list.  Terms absent from an image are
/// treated as zero.
///
/// # Errors
///
/// - [`ModularError::EmptyImageList`] — no images provided.
/// - [`ModularError::IncompatiblePolynomials`] — images have different variable lists.
pub fn lift_crt(images: &[(MultiPolyFp, u64)]) -> Result<MultiPoly, ModularError> {
    if images.is_empty() {
        return Err(ModularError::EmptyImageList);
    }

    let vars = images[0].0.vars.clone();
    for (img, _) in images {
        if img.vars != vars {
            return Err(ModularError::IncompatiblePolynomials);
        }
    }

    // Collect every exponent vector that appears in any image.
    let mut all_exps: std::collections::BTreeSet<Vec<u32>> = std::collections::BTreeSet::new();
    for (img, _) in images {
        for exp in img.terms.keys() {
            all_exps.insert(exp.clone());
        }
    }

    let mut terms: BTreeMap<Vec<u32>, Integer> = BTreeMap::new();

    for exp in &all_exps {
        let residues: Vec<(u64, u64)> = images
            .iter()
            .map(|(img, p)| (img.terms.get(exp).copied().unwrap_or(0), *p))
            .collect();

        let (combined, m) = crt_combine(&residues);
        let centered = center_mod(&combined, &m);

        if centered != 0 {
            terms.insert(exp.clone(), centered);
        }
    }

    Ok(MultiPoly { vars, terms })
}

/// Rational number reconstruction from a modular representative.
///
/// Given `n ∈ [0, M)` and modulus `M > 1`, finds the unique rational `a/b`
/// (with `b > 0`, `gcd(|a|, b) = 1`) such that:
///
/// - `b · n ≡ a (mod M)`
/// - `|a| ≤ T` and `b ≤ T`, where `T = ⌊√(M/2)⌋`
///
/// Returns `None` if no such rational exists (the prime product `M` is too
/// small to uniquely determine the value).
pub fn rational_reconstruction(n: &Integer, m: &Integer) -> Option<(Integer, Integer)> {
    if *m <= 1 {
        return None;
    }

    // Map n to [0, m)
    let n_mod = {
        let r = n.clone() % m.clone();
        if r < 0 {
            r + m
        } else {
            r
        }
    };

    if n_mod == 0 {
        return Some((Integer::from(0), Integer::from(1)));
    }

    // T = ⌊√(M/2)⌋
    let half_m = m.clone() >> 1u32;
    let t = half_m.sqrt();

    // Extended Euclidean: r₋₁ = m, r₀ = n; s₋₁ = 0, s₀ = 1
    let mut r_prev = m.clone();
    let mut r_curr = n_mod;
    let mut s_prev = Integer::from(0);
    let mut s_curr = Integer::from(1);

    while r_curr > t {
        if r_curr == 0 {
            return None;
        }
        let q = r_prev.clone() / r_curr.clone();
        let r_next = r_prev.clone() - q.clone() * r_curr.clone();
        let s_next = s_prev.clone() - q * s_curr.clone();
        r_prev = r_curr;
        r_curr = r_next;
        s_prev = s_curr;
        s_curr = s_next;
    }

    if r_curr == 0 {
        return None;
    }

    let b_abs = s_curr.clone().abs();
    if b_abs == 0 || b_abs > t {
        return None;
    }
    if r_curr.clone().abs() > t {
        return None;
    }

    // Normalise so the denominator is positive.
    let (a, b) = if s_curr < 0 {
        (-r_curr, -s_curr)
    } else {
        (r_curr, s_curr)
    };

    Some((a, b))
}

/// Compute a Cauchy–Mignotte coefficient bound for `poly`.
///
/// Returns `B = ‖f‖₁ · 2^d` where `‖f‖₁ = Σ|aᵢ|` is the L¹ norm and
/// `d = total_degree(f)`.  For CRT reconstruction to succeed, the product of
/// primes must exceed `2B`.
pub fn mignotte_bound(poly: &MultiPoly) -> Integer {
    if poly.is_zero() {
        return Integer::from(1);
    }

    let l1: Integer = poly
        .terms
        .values()
        .map(|c| Integer::from(c.abs_ref()))
        .fold(Integer::from(0), |acc, x| acc + x);

    let d = poly.total_degree();
    let scale = Integer::from(1) << d;
    l1 * scale
}

/// Select the smallest prime not in `used` that does not divide `avoid_divisor`.
///
/// Pass the integer content of the polynomial as `avoid_divisor` to skip primes
/// that would cause leading-coefficient collapse (unlucky primes).  Pass
/// `&Integer::from(0)` to apply no divisibility constraint.
///
/// # Panics
///
/// Panics if no suitable prime can be found below 1 000 000 (should never
/// happen in practice).
pub fn select_lucky_prime(avoid_divisor: &Integer, used: &[u64]) -> u64 {
    let mut candidate = 2u64;
    loop {
        if is_prime(candidate) && !used.contains(&candidate) {
            let lucky = if *avoid_divisor == 0 {
                true
            } else {
                let p_int = Integer::from(candidate);
                let rem = avoid_divisor.clone() % p_int.clone();
                let rem = if rem < 0 { rem + p_int } else { rem };
                rem != 0
            };
            if lucky {
                return candidate;
            }
        }
        candidate += 1;
        if candidate > 1_000_000 {
            panic!("select_lucky_prime: no suitable prime found below 1_000_000");
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Iterative CRT combination.
///
/// Returns `(a, M)` where `a ∈ [0, M)` is the CRT representative and
/// `M = p₁ · … · pₖ`.
fn crt_combine(pairs: &[(u64, u64)]) -> (Integer, Integer) {
    if pairs.is_empty() {
        return (Integer::from(0), Integer::from(1));
    }

    let (a0, p0) = pairs[0];
    let mut a = Integer::from(a0); // invariant: a ∈ [0, M) throughout
    let mut m = Integer::from(p0);

    for &(ai, pi) in &pairs[1..] {
        // a_new ≡ a (mod m) and a_new ≡ ai (mod pi)
        // a_new = a + m * t, where t ≡ (ai − a) · m⁻¹ (mod pi)
        let a_mod_pi = rug_mod_u64(&a, pi);
        let diff = ((ai as u128 + pi as u128 - a_mod_pi as u128) % pi as u128) as u64;
        let m_mod_pi = rug_mod_u64(&m, pi);
        let m_inv = mod_inverse_u64(m_mod_pi, pi);
        let t = ((diff as u128 * m_inv as u128) % pi as u128) as u64;
        // a_new = a + m*t; since t < pi, a_new < m*pi = new_m  ✓
        a += m.clone() * t;
        m *= Integer::from(pi);
    }

    (a, m)
}

/// Center `a ∈ [0, M)` in the symmetric range `(-M/2, M/2]`.
fn center_mod(a: &Integer, m: &Integer) -> Integer {
    let half = m.clone() >> 1u32; // ⌊M/2⌋
    if *a > half {
        a.clone() - m
    } else {
        a.clone()
    }
}

/// Reduce a `rug::Integer` to a `u64` representative in `[0, p)`.
fn rug_mod_u64(a: &Integer, p: u64) -> u64 {
    let p_big = Integer::from(p);
    let r = a.clone() % p_big.clone();
    let r = if r < 0 { r + p_big } else { r };
    r.to_u64().expect("modular result fits in u64")
}

/// Extended-GCD modular inverse for `u64`.
///
/// Precondition: `gcd(a, m) = 1`.
fn mod_inverse_u64(a: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut old_r = a as i128;
    let mut r = m as i128;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;

    while r != 0 {
        let q = old_r / r;
        let tmp_r = r;
        r = old_r - q * r;
        old_r = tmp_r;
        let tmp_s = s;
        s = old_s - q * s;
        old_s = tmp_s;
    }

    ((old_s % m as i128 + m as i128) % m as i128) as u64
}

/// Deterministic Miller–Rabin primality test.
///
/// Uses witnesses `{2, 3, 5, 7}` for `n < 3_215_031_751` and
/// `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` for larger values.
/// Both sets are sufficient to decide primality for all 64-bit integers.
pub fn is_prime(n: u64) -> bool {
    match n {
        0 | 1 => return false,
        2 | 3 | 5 | 7 => return true,
        _ if n % 2 == 0 || n % 3 == 0 || n % 5 == 0 => return false,
        _ => {}
    }

    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d >>= 1;
        r += 1;
    }

    let witnesses: &[u64] = if n < 3_215_031_751 {
        &[2, 3, 5, 7]
    } else {
        &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    };

    'outer: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exp >>= 1;
    }
    result
}

#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool_xy() -> (ExprPool, ExprId, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        (p, x, y)
    }

    // --- is_prime ---

    #[test]
    fn prime_small() {
        for &(n, exp) in &[
            (0u64, false),
            (1, false),
            (2, true),
            (3, true),
            (4, false),
            (5, true),
            (9, false),
            (97, true),
            (100, false),
            (101, true),
        ] {
            assert_eq!(is_prime(n), exp, "is_prime({n})");
        }
    }

    #[test]
    fn prime_large() {
        assert!(is_prime(999_983));
        assert!(!is_prime(1_000_000));
        assert!(is_prime(1_000_003));
        // Large Mersenne prime M31
        assert!(is_prime(2_147_483_647));
    }

    // --- mod_inverse_u64 ---

    #[test]
    fn mod_inverse_basic() {
        assert_eq!(mod_inverse_u64(3, 7), 5); // 3·5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inverse_u64(2, 101), 51); // 2·51 = 102 ≡ 1 (mod 101)
        assert_eq!(mod_inverse_u64(1, 7), 1);
    }

    // --- reduce_mod ---

    #[test]
    fn reduce_mod_basic() {
        let (pool, x, y) = pool_xy();
        // 6x + 4 → mod 5 → x + 4
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(6_i32), x]),
            pool.integer(4_i32),
        ]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();
        let fp = reduce_mod(&poly, 5).unwrap();
        assert_eq!(fp.modulus, 5);
        assert_eq!(*fp.terms.get(&vec![1]).unwrap(), 1u64); // 6 mod 5 = 1
        assert_eq!(*fp.terms.get(&vec![]).unwrap(), 4u64); // 4 mod 5 = 4
    }

    #[test]
    fn reduce_mod_negative_coeff() {
        let (pool, x, y) = pool_xy();
        // -3x → mod 7 → 4x
        let expr = pool.mul(vec![pool.integer(-3_i32), x]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();
        let fp = reduce_mod(&poly, 7).unwrap();
        assert_eq!(*fp.terms.get(&vec![1]).unwrap(), 4u64); // -3 mod 7 = 4
    }

    #[test]
    fn reduce_mod_vanishing_term() {
        let (pool, x, y) = pool_xy();
        // 5x + 7 → mod 5 → 2 (x term vanishes)
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(5_i32), x]),
            pool.integer(7_i32),
        ]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();
        let fp = reduce_mod(&poly, 5).unwrap();
        assert!(!fp.terms.contains_key(&vec![1]));
        assert_eq!(*fp.terms.get(&vec![]).unwrap(), 2u64);
    }

    #[test]
    fn reduce_mod_invalid() {
        let (pool, x, y) = pool_xy();
        let poly = MultiPoly::from_symbolic(x, vec![x, y], &pool).unwrap();
        for bad in [0, 1, 4, 6, 9] {
            assert!(
                matches!(reduce_mod(&poly, bad), Err(ModularError::InvalidModulus(_))),
                "expected InvalidModulus for {bad}"
            );
        }
    }

    // --- crt_combine ---

    #[test]
    fn crt_combine_single() {
        let (a, m) = crt_combine(&[(3, 5)]);
        assert_eq!(a, Integer::from(3));
        assert_eq!(m, Integer::from(5));
    }

    #[test]
    fn crt_combine_two() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x ≡ 8 (mod 15)
        let (a, m) = crt_combine(&[(2, 3), (3, 5)]);
        assert_eq!(m, Integer::from(15));
        assert_eq!(a, Integer::from(8));
        assert_eq!(8u64 % 3, 2);
        assert_eq!(8u64 % 5, 3);
    }

    #[test]
    fn crt_combine_three() {
        // x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x ≡ 23 (mod 30)
        let (a, m) = crt_combine(&[(1, 2), (2, 3), (3, 5)]);
        assert_eq!(m, Integer::from(30));
        assert_eq!(a, Integer::from(23));
        assert_eq!(23u64 % 2, 1);
        assert_eq!(23u64 % 3, 2);
        assert_eq!(23u64 % 5, 3);
    }

    // --- lift_crt ---

    #[test]
    fn lift_crt_roundtrip_positive() {
        let (pool, x, y) = pool_xy();
        // f = 3x² + 2x + 1
        let x2 = pool.pow(x, pool.integer(2_i32));
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(3_i32), x2]),
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();

        let p1 = 101u64;
        let p2 = 103u64;
        let fp1 = reduce_mod(&poly, p1).unwrap();
        let fp2 = reduce_mod(&poly, p2).unwrap();
        let lifted = lift_crt(&[(fp1, p1), (fp2, p2)]).unwrap();
        assert_eq!(lifted, poly);
    }

    #[test]
    fn lift_crt_negative_coeff() {
        let (pool, x, y) = pool_xy();
        // f = x - 50; coefficients in (-50, 50] → need M > 100
        let expr = pool.add(vec![x, pool.integer(-50_i32)]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();

        let p1 = 101u64;
        let p2 = 103u64; // M = 101 * 103 = 10403 > 100
        let lifted = lift_crt(&[
            (reduce_mod(&poly, p1).unwrap(), p1),
            (reduce_mod(&poly, p2).unwrap(), p2),
        ])
        .unwrap();
        assert_eq!(lifted, poly);
    }

    #[test]
    fn lift_crt_bivariate() {
        let (pool, x, y) = pool_xy();
        // f = x*y + 3
        let expr = pool.add(vec![pool.mul(vec![x, y]), pool.integer(3_i32)]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();

        let p = 7u64;
        let q = 11u64;
        let lifted = lift_crt(&[
            (reduce_mod(&poly, p).unwrap(), p),
            (reduce_mod(&poly, q).unwrap(), q),
        ])
        .unwrap();
        assert_eq!(lifted, poly);
    }

    #[test]
    fn lift_crt_empty_error() {
        assert!(matches!(lift_crt(&[]), Err(ModularError::EmptyImageList)));
    }

    // --- rational_reconstruction ---

    #[test]
    fn rat_recon_one_half() {
        // 1/2 mod 101: 2⁻¹ ≡ 51 (mod 101), so n=51
        let result = rational_reconstruction(&Integer::from(51), &Integer::from(101));
        assert!(result.is_some());
        let (a, b) = result.unwrap();
        assert_eq!(a, Integer::from(1));
        assert_eq!(b, Integer::from(2));
    }

    #[test]
    fn rat_recon_negative_numerator() {
        // -1/2 mod 101: -1 * 51 = -51 ≡ 50 (mod 101)
        let result = rational_reconstruction(&Integer::from(50), &Integer::from(101));
        assert!(result.is_some());
        let (a, b) = result.unwrap();
        assert_eq!(a, Integer::from(-1));
        assert_eq!(b, Integer::from(2));
    }

    #[test]
    fn rat_recon_zero() {
        let result = rational_reconstruction(&Integer::from(0), &Integer::from(101));
        assert!(result.is_some());
        let (a, b) = result.unwrap();
        assert_eq!(a, Integer::from(0));
        assert_eq!(b, Integer::from(1));
    }

    #[test]
    fn rat_recon_integer() {
        // n = 5, m = 101: T = 7, 5 ≤ 7, so this is just the integer 5
        let result = rational_reconstruction(&Integer::from(5), &Integer::from(101));
        assert!(result.is_some());
        let (a, b) = result.unwrap();
        assert_eq!(b, Integer::from(1));
        assert_eq!(a, Integer::from(5));
    }

    #[test]
    fn rat_recon_m_too_small() {
        // n=2, M=7: T=⌊√3⌋=1; integer 2 can't be reconstructed since |2| > T=1
        // and no other a/b with |a|≤1 and b≤1 satisfies a/b ≡ 2 (mod 7).
        let result = rational_reconstruction(&Integer::from(2), &Integer::from(7));
        assert!(result.is_none());
    }

    // --- mignotte_bound ---

    #[test]
    fn mignotte_constant() {
        let (pool, x, y) = pool_xy();
        let poly = MultiPoly::from_symbolic(pool.integer(5_i32), vec![x, y], &pool).unwrap();
        // L1=5, d=0 → B=5
        assert_eq!(mignotte_bound(&poly), Integer::from(5));
    }

    #[test]
    fn mignotte_linear() {
        let (pool, x, y) = pool_xy();
        // 3x + 2: L1=5, d=1 → B=10
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(3_i32), x]),
            pool.integer(2_i32),
        ]);
        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();
        assert_eq!(mignotte_bound(&poly), Integer::from(10));
    }

    #[test]
    fn mignotte_zero_poly() {
        let (_, x, y) = pool_xy();
        let z = MultiPoly::zero(vec![x, y]);
        assert_eq!(mignotte_bound(&z), Integer::from(1));
    }

    // --- select_lucky_prime ---

    #[test]
    fn lucky_prime_no_constraint() {
        let p = select_lucky_prime(&Integer::from(0), &[]);
        assert!(is_prime(p));
        assert_eq!(p, 2);
    }

    #[test]
    fn lucky_prime_avoids_divisors() {
        // avoid_divisor=6=2×3; lucky prime must not divide 6
        let p = select_lucky_prime(&Integer::from(6), &[]);
        assert!(is_prime(p));
        assert_ne!(6 % p, 0);
        assert_eq!(p, 5); // first prime not dividing 6
    }

    #[test]
    fn lucky_prime_skips_used() {
        let p = select_lucky_prime(&Integer::from(0), &[2, 3, 5]);
        assert_eq!(p, 7);
    }

    #[test]
    fn lucky_prime_combined() {
        // avoid_divisor=30=2×3×5; skip 2, 3, 5, 7 as used
        let p = select_lucky_prime(&Integer::from(30), &[7]);
        assert!(is_prime(p));
        assert_ne!(30 % p, 0);
        assert_ne!(p, 7);
    }

    // --- ModularValue ---

    #[test]
    fn modular_value_add() {
        let a = ModularValue::new(3, 7);
        let b = ModularValue::new(5, 7);
        assert_eq!(a.add(&b), ModularValue::new(1, 7)); // (3+5) mod 7 = 1
    }

    #[test]
    fn modular_value_sub() {
        let a = ModularValue::new(3, 7);
        let b = ModularValue::new(5, 7);
        assert_eq!(a.sub(&b), ModularValue::new(5, 7)); // (3-5) mod 7 = -2 ≡ 5
    }

    #[test]
    fn modular_value_mul() {
        let a = ModularValue::new(3, 7);
        let b = ModularValue::new(5, 7);
        assert_eq!(a.mul(&b), ModularValue::new(1, 7)); // 15 mod 7 = 1
    }

    #[test]
    fn modular_value_neg() {
        assert_eq!(ModularValue::new(3, 7).neg(), ModularValue::new(4, 7));
        assert_eq!(ModularValue::new(0, 7).neg(), ModularValue::new(0, 7));
    }

    #[test]
    fn modular_value_inverse() {
        // 3⁻¹ ≡ 5 (mod 7): 3·5 = 15 ≡ 1 (mod 7)
        assert_eq!(
            ModularValue::new(3, 7).inverse().unwrap(),
            ModularValue::new(5, 7)
        );
        assert!(ModularValue::new(0, 7).inverse().is_none());
    }

    // --- error codes ---

    #[test]
    fn error_codes() {
        assert_eq!(ModularError::InvalidModulus(4).code(), "E-MOD-001");
        assert_eq!(ModularError::IncompatiblePolynomials.code(), "E-MOD-002");
        assert_eq!(ModularError::EmptyImageList.code(), "E-MOD-003");
        assert_eq!(ModularError::ReconstructionFailed.code(), "E-MOD-004");
    }
}
