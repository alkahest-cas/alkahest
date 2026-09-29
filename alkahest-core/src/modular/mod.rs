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
            .map(|e| crate::poly::exponent::total_degree_or_panic(e))
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
        // u128 like `add`: `value + modulus` overflows u64 once the modulus
        // exceeds 2^63 (found by the Kani harness `modular_value_sub_full_width`).
        let v = ((self.value as u128 + self.modulus as u128 - (other.value % self.modulus) as u128)
            % self.modulus as u128) as u64;
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
        let t = crt_step_u64(rug_mod_u64(&a, pi), rug_mod_u64(&m, pi), ai, pi);
        // a_new = a + m*t; since t < pi, a_new < m*pi = new_m  ✓
        a += m.clone() * t;
        m *= Integer::from(pi);
    }

    (a, m)
}

/// The machine-word half of one CRT step: the `t ∈ [0, pi)` with
/// `a_mod_pi + m_mod_pi · t ≡ ai (mod pi)`.
///
/// Preconditions: `a_mod_pi, m_mod_pi < pi` and `gcd(m_mod_pi, pi) = 1`. Split
/// out of [`crt_combine`] so the `u64` arithmetic can be model-checked without
/// the `rug` (GMP) calls around it — see `verification` below.
#[inline]
fn crt_step_u64(a_mod_pi: u64, m_mod_pi: u64, ai: u64, pi: u64) -> u64 {
    let diff = ((ai as u128 + pi as u128 - a_mod_pi as u128) % pi as u128) as u64;
    let m_inv = mod_inverse_u64(m_mod_pi, pi);
    ((diff as u128 * m_inv as u128) % pi as u128) as u64
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

    // Under `kani_loop_contracts` (TESTING.md § 7) the loop is checked once
    // through this invariant, for every `a` and `m`, instead of unrolled.
    // With `s_i·a ≡ r_i (mod m)` the Bézout coefficients alternate in sign
    // and satisfy `|s|·old_r + |old_s|·r = m`, so each stays `<= m` and
    // `q·s` cannot overflow `i128`.
    #[cfg_attr(kani_loop_contracts, kani::loop_invariant(
        old_r >= 0
            && r >= 0
            && old_r <= u64::MAX as i128
            && r <= u64::MAX as i128
            && old_s.unsigned_abs() <= m as u128
            && s.unsigned_abs() <= m as u128
            && ((old_s >= 0 && s <= 0) || (old_s <= 0 && s >= 0))
            && (s.unsigned_abs() * old_r as u128)
                .checked_add(old_s.unsigned_abs() * r as u128)
                == Some(m as u128)
    ))]
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

    for &a in witnesses {
        if a < n && !miller_rabin_round(n, d, r, a) {
            return false;
        }
    }
    true
}

/// One Miller–Rabin round: `true` if `n` is a strong probable prime to base
/// `a`, where `n − 1 = 2^r · d` with `d` odd.
///
/// Preconditions (established by [`is_prime`]): `n >= 2`, `1 <= r <= 63`.
/// Split out so the model checker can treat the witness loop and the squaring
/// loop separately — nested, they unroll to 66 × 66 copies.
fn miller_rabin_round(n: u64, d: u64, r: u32, a: u64) -> bool {
    let mut x = pow_mod(a, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    for _ in 0..r - 1 {
        x = mul_mod(x, x, n);
        if x == n - 1 {
            return true;
        }
    }
    false
}

fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    // `1 % modulus`, not `1`: for `modulus = 1` and `exp = 0` the answer is
    // the residue 0, and a bare `1` escaped the `[0, modulus)` range (found by
    // Kani; now pinned by `pow_mod_exp_zero_full_width`; `is_prime` never passes 1).
    let mut result = 1 % modulus;
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
// Machine-word gcd
// ---------------------------------------------------------------------------

/// `gcd(a, b)` by Euclid; `gcd(0, 0) = 0`.
pub(crate) fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    // Under Kani (`-Z loop-contracts`) the loop is checked through this
    // invariant instead of being unrolled: either nothing has run yet, or
    // `a` is a nonzero earlier remainder bounded by each nonzero argument
    // (`b == a₀` covers the one step where `b₀ > a₀` puts `b₀` in `a`).
    #[cfg_attr(kani_loop_contracts, kani::loop_invariant(
        (a == on_entry(a) && b == on_entry(b))
            || (a != 0
                && a <= on_entry(b)
                && b <= on_entry(b)
                && (on_entry(a) == 0 || b <= on_entry(a))
                && (on_entry(a) == 0 || a <= on_entry(a) || b == on_entry(a)))
    ))]
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `gcd(|a|, |b|)` for signed machine words; `gcd(0, 0) = 0`.
///
/// The one crate-wide copy: four modules (by-parts integration, the algebraic
/// RDE, `find_order` and the q-Zeilberger term code) each had their own, all
/// starting from `a.abs()`, which panics on `i64::MIN` in a debug build and
/// stays negative in a release one. This takes `unsigned_abs` instead. The
/// result is non-negative except in the single case the gcd is `2^63`
/// (`a, b ∈ {0, i64::MIN}`, not both 0), which is returned as `i64::MIN` —
/// same divisors, so `x % g` and "`g == 1`" tests stay correct.
pub(crate) fn gcd_i64(a: i64, b: i64) -> i64 {
    gcd_u64(a.unsigned_abs(), b.unsigned_abs()) as i64
}

// ---------------------------------------------------------------------------
// Kani bounded model checking
// ---------------------------------------------------------------------------
//
// Run with (see TESTING.md § Kani):
//   cargo kani -p alkahest-cas -Z stubbing --harness modular::verification::
//
// Each harness states the exact input region it covers. "Full width" means
// every u64 value the preconditions allow; anything narrower is written down
// next to the harness, together with why it was narrowed. What is *not*
// claimed anywhere below: that the Miller–Rabin witness sets in `is_prime`
// decide primality. That is a number-theoretic fact (Jaeschke / Sorenson–Webster
// bounds), not something a bit-level model checker can establish; the
// harnesses only show `is_prime` cannot panic or overflow.

#[cfg(kani)]
mod verification {
    use super::*;

    /// Stand-ins for the compositional harnesses. Each returns an arbitrary
    /// value in `[0, m)`. For `mul_mod` and `pow_mod` that contract is proven
    /// at full width below; for `mod_inverse_u64` only on bounded ranges (see
    /// `crt_step_u64_no_overflow_full_width`).
    /// Referenced only from `#[kani::stub]`, which rustc's dead-code pass
    /// does not see.
    #[allow(dead_code)]
    mod stubs {
        fn any_below(m: u64) -> u64 {
            kani::any_where(|r: &u64| *r < m)
        }
        pub fn mul_mod(_a: u64, _b: u64, m: u64) -> u64 {
            any_below(m)
        }
        pub fn pow_mod(_base: u64, _exp: u64, m: u64) -> u64 {
            any_below(m)
        }
        pub fn mod_inverse_u64(_a: u64, m: u64) -> u64 {
            any_below(m)
        }
        /// Checks the caller meets `miller_rabin_round`'s preconditions,
        /// which `miller_rabin_round_no_panic_full_width` assumes.
        pub fn miller_rabin_round(n: u64, _d: u64, r: u32, _a: u64) -> bool {
            assert!(n >= 2 && r >= 1 && r <= 63);
            kani::any()
        }
        /// `gcd_u64`'s range contract: zero iff `a = b = 0`, else at most
        /// each nonzero argument.
        pub fn gcd_u64(a: u64, b: u64) -> u64 {
            if a == 0 && b == 0 {
                return 0;
            }
            kani::any_where(|g: &u64| *g >= 1 && (a == 0 || *g <= a) && (b == 0 || *g <= b))
        }
    }

    // --- mul_mod -----------------------------------------------------------

    /// Full width: all `a, b` and all `m > 0`. No panic, and the narrowing
    /// `u128 → u64` cast is lossless because the result is `< m`.
    #[kani::proof]
    fn mul_mod_in_range_full_width() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        let m: u64 = kani::any_where(|m: &u64| *m > 0);
        assert!(mul_mod(a, b, m) < m);
    }

    /// Value check against plain u64 arithmetic. Bounds: `a, b < 2^8`,
    /// `1 <= m < 2^8` — a full-width `a·b mod m` equality is a 128-bit
    /// multiply-and-divide circuit; CaDiCaL had not closed it after 35 min
    /// and Kissat after 24 min, when both were stopped.
    #[kani::proof]
    fn mul_mod_matches_u64_small() {
        let a: u64 = kani::any_where(|a: &u64| *a < (1 << 8));
        let b: u64 = kani::any_where(|b: &u64| *b < (1 << 8));
        let m: u64 = kani::any_where(|m: &u64| *m >= 1 && *m < (1 << 8));
        assert_eq!(mul_mod(a, b, m), a * b % m);
    }

    // --- pow_mod -----------------------------------------------------------

    /// Full width in `base`, `exp` and `m >= 1`, with `mul_mod` replaced by
    /// its proven contract (any value `< m`, `mul_mod_in_range_full_width`).
    /// Shows the square-and-multiply loop itself cannot panic or overflow,
    /// terminates within 64 rounds, and returns a value `< m` — including
    /// `m = 1`, which returned 1 before the `1 % modulus` fix.
    #[kani::proof]
    #[kani::stub(mul_mod, stubs::mul_mod)]
    #[kani::unwind(66)]
    fn pow_mod_in_range_full_width() {
        let base: u64 = kani::any();
        let exp: u64 = kani::any();
        let m: u64 = kani::any_where(|m: &u64| *m >= 1);
        assert!(pow_mod(base, exp, m) < m);
    }

    /// `x^0 ≡ 1 (mod m)` for every `x` and every `m >= 1`, as a canonical
    /// residue — so `pow_mod(x, 0, 1) = 0`, the case Kani caught returning 1.
    ///
    /// No value check for larger exponents. A chain of modular products is
    /// an equivalence-of-multipliers problem that SAT does not close: against
    /// repeated multiplication, `exp <= 3` with `m < 2^8` did not finish in an
    /// hour, a `u32` re-statement with `m < 2^16` not in 50 minutes, and even
    /// `exp <= 1` not in 10. The general exponent is left to the unit tests;
    /// `pow_mod_in_range_full_width` covers panic freedom and range.
    #[kani::proof]
    fn pow_mod_exp_zero_full_width() {
        let base: u64 = kani::any();
        let m: u64 = kani::any_where(|m: &u64| *m >= 1);
        assert_eq!(pow_mod(base, 0, m), 1 % m);
    }

    // --- mod_inverse_u64 ---------------------------------------------------
    //
    // "gcd(a, m) = 1" is phrased as "some x < m has a·x ≡ 1": equivalent, and
    // one multiply instead of a second Euclid loop in the harness.

    /// Small modulus, `a` up to 2^8 (so `a >= m` is covered).
    /// Bounds: `2 <= m < 2^4`, `a < 2^8`. Result `< m`, and whenever `a` is
    /// invertible mod `m`, `a · inv ≡ 1 (mod m)`. Tiny because every Euclid
    /// round is an `i128` division by a symbolic divisor; 8-bit moduli did
    /// not finish in an hour.
    #[kani::proof]
    #[kani::unwind(8)]
    fn mod_inverse_small_modulus() {
        let a: u64 = kani::any_where(|a: &u64| *a < (1 << 8));
        let m: u64 = kani::any_where(|m: &u64| *m >= 2 && *m < (1 << 4));
        let inv = mod_inverse_u64(a, m);
        assert!(inv < m);
        let x: u64 = kani::any_where(|x: &u64| *x < m);
        if mul_mod(a, x, m) == 1 {
            assert_eq!(mul_mod(a, inv, m), 1);
        }
    }

    /// Modulus anywhere in the u64 range, up to `u64::MAX`, small `a`: no
    /// panic or `i128` overflow and a result `< m`. Bounds: `m >= 2` full
    /// width, `1 <= a < 2^4`. After the first swap the remainders are `< a`,
    /// so this exercises the casts, a quotient near `2^64`, and the final
    /// normalisation at the top of the range. (Adding the `a·inv ≡ 1` check
    /// here — a 128-bit remainder by a full-width symbolic `m` — did not
    /// finish in 25 minutes even for `a < 4`; the special values below check
    /// the answer at full width instead.)
    #[kani::proof]
    #[kani::unwind(9)]
    fn mod_inverse_in_range_large_modulus() {
        let a: u64 = kani::any_where(|a: &u64| *a >= 1 && *a < (1 << 4));
        let m: u64 = kani::any_where(|m: &u64| *m >= 2);
        assert!(mod_inverse_u64(a, m) < m);
    }

    /// Exact inverses at full width, where they have a closed form:
    /// `1⁻¹ = 1` and, for odd `m`, `2⁻¹ = (m+1)/2`. Bounds: every `m >= 3`,
    /// so the i128 path — a quotient of `m` itself, the sign flip and the
    /// final normalisation — is checked up to `u64::MAX`. (`(m−1)⁻¹ = m−1`
    /// was dropped: its `m / (m−1)` is a division by a symbolic divisor and
    /// did not finish in 25 minutes.)
    #[kani::proof]
    #[kani::unwind(5)]
    fn mod_inverse_special_values_full_width() {
        let m: u64 = kani::any_where(|m: &u64| *m >= 3);
        assert_eq!(mod_inverse_u64(1, m), 1);
        if m % 2 == 1 {
            assert_eq!(mod_inverse_u64(2, m), m / 2 + 1);
        }
    }

    /// Every `a` and every modulus `m >= 1`, both full width: no panic, no
    /// `i128` overflow, and a result `< m`. The extended-Euclid loop is
    /// checked once through its invariant (the Bézout identity
    /// `|s|·old_r + |old_s|·r = m` with alternating signs), not unrolled.
    /// This is the range contract `stubs::mod_inverse_u64` assumes. The value
    /// `a·inv ≡ 1` is still only checked on the small ranges above.
    #[cfg(kani_loop_contracts)]
    #[kani::proof]
    fn mod_inverse_in_range_inductive() {
        let a: u64 = kani::any();
        let m: u64 = kani::any_where(|m: &u64| *m >= 1);
        assert!(mod_inverse_u64(a, m) < m);
    }

    // --- crt_combine (u64 step) --------------------------------------------

    /// One CRT step: for `ai, a_mod_pi, m_mod_pi < pi` with `m_mod_pi`
    /// invertible mod `pi`, the returned `t` is `< pi` and
    /// `a_mod_pi + m_mod_pi · t ≡ ai (mod pi)` — i.e. `a + M·t` satisfies both
    /// congruences. Bounds: `2 <= pi < 2^4` (the inverse loop, as above).
    #[kani::proof]
    #[kani::unwind(8)]
    fn crt_step_u64_solves_congruence() {
        let pi: u64 = kani::any_where(|p: &u64| *p >= 2 && *p < (1 << 4));
        let ai: u64 = kani::any_where(|x: &u64| *x < pi);
        let a_mod_pi: u64 = kani::any_where(|x: &u64| *x < pi);
        let m_mod_pi: u64 = kani::any_where(|x: &u64| *x < pi);
        let x: u64 = kani::any_where(|x: &u64| *x < pi);
        kani::assume(m_mod_pi * x % pi == 1);
        let t = crt_step_u64(a_mod_pi, m_mod_pi, ai, pi);
        assert!(t < pi);
        assert_eq!((a_mod_pi + m_mod_pi * t) % pi, ai);
    }

    /// Panic freedom of the step's own arithmetic at full width, with the
    /// inverse replaced by an arbitrary value `< pi`. That contract holds
    /// whenever `mod_inverse_u64` returns — its last step is a `rem_euclid`-
    /// style `% m` — and is checked above on the ranges where it finishes;
    /// full-width panic freedom of the inverse itself is *not* proven.
    /// Bounds: `pi >= 2` full width, `ai, a_mod_pi, m_mod_pi < pi`.
    #[kani::proof]
    #[kani::stub(mod_inverse_u64, stubs::mod_inverse_u64)]
    fn crt_step_u64_no_overflow_full_width() {
        let pi: u64 = kani::any_where(|p: &u64| *p >= 2);
        let ai: u64 = kani::any_where(|x: &u64| *x < pi);
        let a_mod_pi: u64 = kani::any_where(|x: &u64| *x < pi);
        let m_mod_pi: u64 = kani::any_where(|x: &u64| *x < pi);
        assert!(crt_step_u64(a_mod_pi, m_mod_pi, ai, pi) < pi);
    }

    // --- ModularValue ------------------------------------------------------

    fn any_value() -> ModularValue {
        let modulus: u64 = kani::any_where(|m: &u64| *m > 0);
        any_value_mod(modulus)
    }

    fn any_value_mod(modulus: u64) -> ModularValue {
        let value: u64 = kani::any_where(|v: &u64| *v < modulus);
        ModularValue { value, modulus }
    }

    /// Full width: every modulus `> 0` and residues in `[0, m)`.
    #[kani::proof]
    fn modular_value_add_full_width() {
        let a = any_value();
        let b = any_value_mod(a.modulus);
        let r = a.add(&b);
        let s = a.value as u128 + b.value as u128;
        let m = a.modulus as u128;
        assert_eq!(r.value as u128, if s >= m { s - m } else { s });
        assert_eq!(r.modulus, a.modulus);
    }

    /// Full width: every modulus `> 0` and residues in `[0, m)`.
    #[kani::proof]
    fn modular_value_sub_full_width() {
        let a = any_value();
        let b = any_value_mod(a.modulus);
        let r = a.sub(&b);
        let expect = if a.value >= b.value {
            a.value - b.value
        } else {
            a.modulus - (b.value - a.value)
        };
        assert_eq!(r.value, expect);
    }

    /// Full width: every modulus `> 0` and residues in `[0, m)`; the result
    /// is a canonical residue (no panic, lossless narrowing). The value itself
    /// is `mul_mod`'s formula, checked by `mul_mod_matches_u64_small`.
    #[kani::proof]
    fn modular_value_mul_in_range_full_width() {
        let a = any_value();
        let b = any_value_mod(a.modulus);
        let r = a.mul(&b);
        assert!(r.value < a.modulus);
        assert_eq!(r.modulus, a.modulus);
    }

    /// Full width: `a + (−a) ≡ 0` and the result is a canonical residue.
    #[kani::proof]
    fn modular_value_neg_full_width() {
        let a = any_value();
        let n = a.neg();
        assert!(n.value < a.modulus);
        assert_eq!((a.value as u128 + n.value as u128) % a.modulus as u128, 0);
    }

    // --- is_prime ------------------------------------------------------------

    /// Panic/overflow freedom of `is_prime` for **every** `u64`, with each
    /// Miller–Rabin round replaced by a stub that *asserts* the round's
    /// preconditions (`n >= 2`, `1 <= r <= 63`) and returns an arbitrary
    /// verdict. Covers the trial-division prefix, the `n − 1 = 2^r · d`
    /// decomposition (≤ 63 halvings, `r` never 0 because `n` is odd there),
    /// and both witness tables. Says nothing about whether the answer is
    /// right — see the note at the top of this section.
    #[kani::proof]
    #[kani::stub(miller_rabin_round, stubs::miller_rabin_round)]
    #[kani::unwind(66)]
    fn is_prime_no_panic_full_width() {
        let n: u64 = kani::any();
        let _ = is_prime(n);
    }

    /// Panic/overflow freedom of one Miller–Rabin round over its whole
    /// precondition (`n >= 2`, `1 <= r <= 63`; `d`, `a` full width), with
    /// `pow_mod` / `mul_mod` replaced by their proven contract (any value in
    /// `[0, n)`). Together with the harness above, `is_prime` cannot panic.
    #[kani::proof]
    #[kani::stub(pow_mod, stubs::pow_mod)]
    #[kani::stub(mul_mod, stubs::mul_mod)]
    #[kani::unwind(64)]
    fn miller_rabin_round_no_panic_full_width() {
        let n: u64 = kani::any_where(|n: &u64| *n >= 2);
        let r: u32 = kani::any_where(|r: &u32| *r >= 1 && *r <= 63);
        let _ = miller_rabin_round(n, kani::any(), r, kani::any());
    }

    /// End-to-end agreement with trial division, unstubbed, for `n < 2^6`.
    /// Exhaustive testing in model-checker form: it pins the small cases
    /// (where witnesses `a >= n` are skipped) and proves nothing about large
    /// `n`.
    #[kani::proof]
    #[kani::unwind(9)]
    fn is_prime_small_agrees_with_trial_division() {
        let n: u64 = kani::any_where(|n: &u64| *n < (1 << 6));
        let mut expect = n >= 2;
        let mut q = 2;
        while q * q <= n {
            if n % q == 0 {
                expect = false;
            }
            q += 1;
        }
        assert_eq!(is_prime(n), expect);
    }

    // --- gcd_u64 / gcd_i64 -------------------------------------------------

    /// Every pair of `u64`s: no panic, and the range contract `gcd_i64`'s
    /// harness stubs `gcd_u64` by — zero iff `a = b = 0`, otherwise at most
    /// each nonzero argument. Euclid's loop is checked once through its
    /// invariant (`-Z loop-contracts`) rather than unrolled ~93 times.
    /// Divisibility is a value identity through the symbolic divider; it is
    /// tested exhaustively on small arguments instead (`gcd_tests`).
    #[cfg(kani_loop_contracts)]
    #[kani::proof]
    fn gcd_u64_in_range_inductive() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        let g = gcd_u64(a, b);
        assert_eq!(g == 0, a == 0 && b == 0);
        assert!(a == 0 || g <= a);
        assert!(b == 0 || g <= b);
    }

    /// Every pair of `i64`s, including `i64::MIN` (where the four copies this
    /// replaced called `.abs()`): no panic, and the result is `>= 0` except for
    /// gcd `2^63`, returned as `i64::MIN`. `gcd_u64` is stubbed by its range
    /// contract (at most each nonzero argument, zero only for `(0, 0)`),
    /// which `gcd_u64_in_range_inductive` proves.
    #[kani::proof]
    #[kani::stub(gcd_u64, stubs::gcd_u64)]
    fn gcd_i64_sign_full_width() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let g = gcd_i64(a, b);
        if g < 0 {
            assert_eq!(g, i64::MIN);
            assert!(a == 0 || a == i64::MIN);
            assert!(b == 0 || b == i64::MIN);
        }
        assert_eq!(g == 0, a == 0 && b == 0);
    }
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

    // --- pow_mod ---

    #[test]
    fn pow_mod_modulus_one_is_zero() {
        // Regression (found by Kani): x^0 mod 1 returned 1, outside [0, 1).
        assert_eq!(pow_mod(5, 0, 1), 0);
        assert_eq!(pow_mod(5, 3, 1), 0);
        assert_eq!(pow_mod(5, 0, 7), 1);
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
    fn modular_value_sub_modulus_above_2_63() {
        // Regression (found by Kani): `value + modulus` overflowed u64 for a
        // modulus > 2^63 — a debug-build panic, a wrapped wrong residue in
        // release. 18446744073709551557 = 2^64 − 59 is the largest u64 prime.
        let m = 18_446_744_073_709_551_557u64;
        let a = ModularValue::new(m - 1, m);
        let b = ModularValue::new(1, m);
        assert_eq!(a.sub(&b), ModularValue::new(m - 2, m));
        assert_eq!(b.sub(&a), ModularValue::new(2, m));
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

#[cfg(test)]
mod gcd_tests {
    use super::{gcd_i64, gcd_u64};

    /// Exhaustive on `a, b < 2^7`: the greatest common divisor, by
    /// definition. (Kani proves only the range contract at full width; its
    /// loop contract abstracts the value.)
    #[test]
    fn gcd_u64_is_gcd_exhaustive_small() {
        for a in 0..128u64 {
            for b in 0..128u64 {
                let g = gcd_u64(a, b);
                if a == 0 && b == 0 {
                    assert_eq!(g, 0);
                    continue;
                }
                assert!(g >= 1 && a % g == 0 && b % g == 0, "gcd({a}, {b}) = {g}");
                assert!((g + 1..=a.max(b)).all(|c| a % c != 0 || b % c != 0));
            }
        }
    }

    #[test]
    fn gcd_i64_signs_and_extremes() {
        for a in -64..64i64 {
            for b in -64..64i64 {
                let g = gcd_i64(a, b);
                assert_eq!(g, gcd_u64(a.unsigned_abs(), b.unsigned_abs()) as i64);
                assert_eq!(g, gcd_i64(-a, b));
                assert!(g >= 0);
            }
        }
        // `.abs()` panicked (debug) / stayed negative (release) on these.
        assert_eq!(gcd_i64(i64::MIN, 6), 2);
        assert_eq!(gcd_i64(6, i64::MIN), 2);
        assert_eq!(gcd_i64(i64::MIN, 3), 1);
        assert_eq!(gcd_i64(i64::MIN, 0), i64::MIN); // 2^63, the one negative result
        assert_eq!(gcd_i64(i64::MIN, i64::MIN), i64::MIN);
        assert_eq!(gcd_i64(i64::MAX, i64::MIN), 1);
        assert_eq!(gcd_u64(u64::MAX, u64::MAX - 1), 1);
        assert_eq!(gcd_u64(0, 0), 0);
    }
}
