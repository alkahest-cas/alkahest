//! V2-4 — Real root isolation via Vincent–Akritas–Strzeboński (VAS).
//!
//! # Algorithm
//!
//! The public entry point [`real_roots`] implements the **VAS continued-fraction
//! method** for isolating all real roots of a univariate polynomial with integer
//! coefficients.  The core loop is the **Möbius-based Descartes bisection** (VCA):
//!
//! 1. Extract the squarefree part `p / gcd(p, p')` to eliminate repeated roots.
//! 2. Separate positive and negative roots (negative = negated positive roots of
//!    `p(−x)`).
//! 3. Maintain a stack of `(poly, Möbius (a,b,c,d))` frames where
//!    `x = (a·t + b)/(c·t + d)`.  The positive real roots of `poly(t)` biject with
//!    the real roots of `p(x)` in the tracking interval.
//! 4. At each frame:
//!    - **Descartes' rule**: count sign variations `V` in the non-zero coefficients.
//!      `V = 0` → no roots; `V = 1` → exactly one root, record the interval.
//!    - **VAS CF step**: compute a Cauchy-based integer lower bound `k` on the
//!      smallest positive root; if `k ≥ 1`, shift `p(x+k)` (Taylor translate)
//!      before splitting — the key VAS speedup over plain bisection.
//!    - **Bisect at t = 1**: push the right child `q(t+1)` and the left child
//!      `(t+1)ⁿ q(1/(t+1))` = `taylor_shift_1(reverse(q))`.
//! 5. Roots exactly at the split point `t = 1` (or at `t = 0` after a CF shift)
//!    are detected by checking `p(1) = 0` before bisecting, recorded as exact-point
//!    intervals, and deflated.  After any deflation a forced bisect avoids producing
//!    overlapping intervals.
//!
//! # Public API
//!
//! - [`real_roots`] — isolate all real roots of a [`UniPoly`].
//! - [`real_roots_symbolic`] — same, starting from a symbolic [`ExprId`].
//! - [`refine_root`] — narrow a [`RootInterval`] to a given bit-precision.
//! - [`RootInterval`] — rational isolating interval `[lo, hi]`.
//! - [`RealRootError`] — error type.

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use crate::poly::error::ConversionError;
use crate::poly::unipoly::UniPoly;
use rug::Integer;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by [`real_roots`] and [`real_roots_symbolic`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealRootError {
    /// The expression could not be converted to a univariate polynomial with
    /// integer coefficients.
    NotAPolynomial(ConversionError),
    /// The polynomial is identically zero.
    ZeroPolynomial,
}

impl From<ConversionError> for RealRootError {
    fn from(e: ConversionError) -> Self {
        RealRootError::NotAPolynomial(e)
    }
}

impl fmt::Display for RealRootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RealRootError::NotAPolynomial(e) => write!(f, "not a polynomial: {e}"),
            RealRootError::ZeroPolynomial => {
                write!(f, "zero polynomial has infinitely many roots (E-ROOT-002)")
            }
        }
    }
}

impl std::error::Error for RealRootError {}

impl crate::errors::AlkahestError for RealRootError {
    fn code(&self) -> &'static str {
        match self {
            RealRootError::NotAPolynomial(_) => "E-ROOT-001",
            RealRootError::ZeroPolynomial => "E-ROOT-002",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            RealRootError::NotAPolynomial(_) => Some(
                "ensure the input is a polynomial expression with integer coefficients \
                 in a single variable",
            ),
            RealRootError::ZeroPolynomial => {
                Some("real_roots is only defined for non-zero polynomials")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RootInterval — rational isolating interval
// ---------------------------------------------------------------------------

/// A closed rational interval `[lo, hi]` containing exactly one real root of a
/// squarefree polynomial.  For an exact rational root `r`, `lo == hi == r`.
#[derive(Debug, Clone)]
pub struct RootInterval {
    pub lo: rug::Rational,
    pub hi: rug::Rational,
}

impl RootInterval {
    /// Construct from two rational endpoints with `lo ≤ hi`.
    pub fn new(lo: rug::Rational, hi: rug::Rational) -> Self {
        debug_assert!(lo <= hi, "RootInterval requires lo ≤ hi");
        RootInterval { lo, hi }
    }

    /// Approximate lower bound as `f64`.
    pub fn lo_f64(&self) -> f64 {
        self.lo.to_f64()
    }

    /// Approximate upper bound as `f64`.
    pub fn hi_f64(&self) -> f64 {
        self.hi.to_f64()
    }

    /// Width `hi − lo` as a [`rug::Rational`].
    pub fn width(&self) -> rug::Rational {
        self.hi.clone() - self.lo.clone()
    }

    /// Lower bound as `(numerator_string, denominator_string)` in decimal.
    pub fn lo_exact(&self) -> (String, String) {
        (self.lo.numer().to_string(), self.lo.denom().to_string())
    }

    /// Upper bound as `(numerator_string, denominator_string)` in decimal.
    pub fn hi_exact(&self) -> (String, String) {
        (self.hi.numer().to_string(), self.hi.denom().to_string())
    }
}

impl fmt::Display for RootInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.lo, self.hi)
    }
}

// ---------------------------------------------------------------------------
// Primitive polynomial operations on Vec<Integer>
// ---------------------------------------------------------------------------
// Polynomials are stored as coefficient vectors in **ascending degree order**:
// index 0 is the constant term.

/// Count sign variations in the non-zero coefficients (Descartes' rule of signs).
fn sign_variations(coeffs: &[Integer]) -> usize {
    let mut count = 0;
    let mut prev: Option<bool> = None;
    for c in coeffs {
        if *c == 0 {
            continue;
        }
        let pos = *c > 0;
        if prev.is_some_and(|p| p != pos) {
            count += 1;
        }
        prev = Some(pos);
    }
    count
}

/// Compute `p(x + 1)` using the O(n²) de Casteljau / Taylor-shift algorithm.
///
/// For each `i = 0..n−1`, for each `j = (i..n−1)` in reverse:
/// `c[j] += c[j+1]`.
fn taylor_shift_by_1(coeffs: &[Integer]) -> Vec<Integer> {
    let mut c: Vec<Integer> = coeffs.to_vec();
    taylor_shift_by_1_in_place(&mut c);
    c
}

/// In-place `p(x + 1)`.
///
/// The accumulation is done through a `split_at_mut` pair rather than
/// `c[j] += c[j + 1].clone()`: the clone allocated and freed a fresh `mpz` on
/// every one of the O(n²) inner steps, which dominated the cost of this
/// function for the small coefficients typical of a VAS frame.
fn taylor_shift_by_1_in_place(c: &mut [Integer]) {
    let n = c.len();
    for i in 0..n.saturating_sub(1) {
        for j in (i..n - 1).rev() {
            let (left, right) = c.split_at_mut(j + 1);
            left[j] += &right[0];
        }
    }
}

/// Compute `p(x + k)` for a non-negative integer `k`.
fn taylor_shift_by(coeffs: &[Integer], k: u64) -> Vec<Integer> {
    if k == 0 {
        return coeffs.to_vec();
    }
    let mut c = coeffs.to_vec();
    let n = c.len();
    let ki = Integer::from(k);
    for i in 0..n.saturating_sub(1) {
        for j in (i..n - 1).rev() {
            let (left, right) = c.split_at_mut(j + 1);
            // Fused multiply-add (`mpz_addmul`); the previous form built and
            // dropped a temporary `Integer` on every inner step.
            left[j] += &right[0] * &ki;
        }
    }
    c
}

/// Reverse the coefficient vector: `[c₀,…,cₙ] → [cₙ,…,c₀]`.
fn reverse_coeffs(coeffs: &[Integer]) -> Vec<Integer> {
    coeffs.iter().cloned().rev().collect()
}

/// Remove trailing zeros (eliminates zero leading coefficients).
fn trim_trailing_zeros(c: &mut Vec<Integer>) {
    while c.last().is_some_and(|v| *v == 0) {
        c.pop();
    }
}

/// Sum all coefficients: evaluates `p(1)`.
fn evaluate_at_1(coeffs: &[Integer]) -> Integer {
    coeffs.iter().fold(Integer::from(0), |acc, c| acc + c)
}

/// Divide by `t` (caller guarantees `c[0] == 0`).
fn divide_by_t(coeffs: &[Integer]) -> Vec<Integer> {
    debug_assert_eq!(coeffs[0], 0, "divide_by_t: constant term must be zero");
    coeffs[1..].to_vec()
}

/// Divide `p` by `(t − 1)` via synthetic division (caller guarantees `p(1) = 0`).
///
/// Recurrence: `q[n−1] = c[n]`, `q[k−1] = c[k] + q[k]` for `k = n−1 … 1`.
fn divide_by_t_minus_1(coeffs: &[Integer]) -> Vec<Integer> {
    let n = coeffs.len() - 1;
    if n == 0 {
        return vec![];
    }
    let mut q = vec![Integer::from(0); n];
    q[n - 1] = coeffs[n].clone();
    for k in (1..n).rev() {
        let qk = q[k].clone();
        q[k - 1] = coeffs[k].clone() + qk;
    }
    q
}

/// Remove the content (integer GCD of all coefficients).
fn remove_content(coeffs: &[Integer]) -> Vec<Integer> {
    if coeffs.is_empty() {
        return vec![];
    }
    let g = coeffs.iter().fold(Integer::from(0), |acc, c| {
        let ca = c.clone().abs();
        acc.gcd(&ca)
    });
    if g == 0 || g == 1 {
        return coeffs.to_vec();
    }
    coeffs.iter().map(|c| c.clone() / g.clone()).collect()
}

/// Formal derivative: `[c₀,c₁,…,cₙ] → [c₁, 2c₂, …, ncₙ]`.
fn derivative_coeffs(coeffs: &[Integer]) -> Vec<Integer> {
    if coeffs.len() <= 1 {
        return vec![];
    }
    coeffs[1..]
        .iter()
        .enumerate()
        .map(|(i, c)| c.clone() * (i as u64 + 1))
        .collect()
}

// ---------------------------------------------------------------------------
// Polynomial GCD via subresultant-style pseudo-remainder
// ---------------------------------------------------------------------------

/// Pseudo-remainder of `a ÷ b` using coefficient-exact arithmetic.
///
/// Computes `R` satisfying `lc(b)^d · a = Q · b + R`.
/// All arithmetic stays in ℤ; no rational numbers required.
fn poly_pseudo_rem(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let db = b.len().saturating_sub(1);
    if db == 0 {
        // `b` is a non-zero constant → remainder is 0.
        if b.iter().any(|c| *c != 0) {
            return vec![];
        }
        return a.to_vec();
    }
    let lc_b = b.last().unwrap().clone();
    let mut r = a.to_vec();

    while r.len().saturating_sub(1) >= db {
        let dr = r.len() - 1;
        let shift = dr - db;
        let lc_r = r.last().unwrap().clone();

        // r ← lc(b) · r − lc(r) · xˢʰⁱᶠᵗ · b
        // Coefficients at positions 0..shift: multiply by lc(b).
        for coeff in r[..shift].iter_mut() {
            *coeff = lc_b.clone() * coeff.clone();
        }
        // Coefficients at positions shift..shift+b.len():
        // scale by lc(b) and subtract lc(r)·b[i].
        for i in 0..b.len() {
            let old = r[i + shift].clone();
            r[i + shift] = lc_b.clone() * old - lc_r.clone() * b[i].clone();
        }

        r.pop();
        trim_trailing_zeros(&mut r);
    }
    r
}

/// GCD of two integer polynomials (normalised to positive leading coefficient).
fn poly_gcd(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let b_zero = b.iter().all(|c| *c == 0);
    if b_zero {
        let mut g = remove_content(a);
        trim_trailing_zeros(&mut g);
        if g.last().is_some_and(|v| *v < 0) {
            for c in g.iter_mut() {
                *c = Integer::from(0) - c.clone();
            }
        }
        return g;
    }

    let prem = poly_pseudo_rem(a, b);
    let prem_zero = prem.iter().all(|c| *c == 0);
    if prem_zero {
        return poly_gcd(b, &[]);
    }
    let mut r = remove_content(&prem);
    trim_trailing_zeros(&mut r);
    poly_gcd(b, &r)
}

/// Exact polynomial division `a / b` (requires `b | a`).
fn poly_exact_div(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let da = a.len() as i64 - 1;
    let db = b.len() as i64 - 1;
    if da < db || b.iter().all(|c| *c == 0) {
        return vec![Integer::from(0)];
    }
    let deg_q = (da - db) as usize;
    let mut q = vec![Integer::from(0); deg_q + 1];
    let mut r = a.to_vec();
    let lc_b = b.last().unwrap().clone();

    for i in (0..=deg_q).rev() {
        let lc_r = r[i + b.len() - 1].clone();
        let qi = lc_r / lc_b.clone();
        q[i] = qi.clone();
        for (j, bj) in b.iter().enumerate() {
            let old = r[i + j].clone();
            r[i + j] = old - qi.clone() * bj.clone();
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Squarefree decomposition
// ---------------------------------------------------------------------------

/// Extract the squarefree part `p / gcd(p, p')`.
fn squarefree_part(coeffs: &[Integer]) -> Vec<Integer> {
    if coeffs.len() <= 1 {
        return coeffs.to_vec();
    }
    let dp = derivative_coeffs(coeffs);
    if dp.iter().all(|c| *c == 0) {
        return coeffs.to_vec();
    }
    let g = poly_gcd(coeffs, &dp);
    if g.len() <= 1 {
        // GCD is a non-zero constant: polynomial is squarefree.
        return coeffs.to_vec();
    }
    let result = poly_exact_div(coeffs, &g);
    let mut r = remove_content(&result);
    trim_trailing_zeros(&mut r);
    // Normalise to positive leading coefficient.
    if r.last().is_some_and(|v| *v < 0) {
        for c in r.iter_mut() {
            *c = Integer::from(0) - c.clone();
        }
    }
    r
}

// ---------------------------------------------------------------------------
// VAS CF lower bound
// ---------------------------------------------------------------------------

/// Descartes' rule of signs applied to the open interval `(0, k)`.
///
/// Substituting `x = k·u` maps `(0, k)` onto `(0, 1)`, and the usual test for
/// `(0, 1)` is `(1+t)ⁿ·p(1/(1+t))`, i.e. reverse the coefficients and Taylor
/// shift by 1. The number of roots in `(0, k)` is at most the number of sign
/// variations of the result, so a count of **zero is a proof** that there is
/// no root there. A non-zero count proves nothing either way, which is why
/// the caller only ever uses `true`.
fn no_root_in_open_interval(coeffs: &[Integer], k: u64) -> bool {
    if k == 0 {
        return true;
    }
    let n = coeffs.len();
    if n < 2 {
        return true;
    }
    // Build `reverse(p(k·u))` directly: index `i` of the result is
    // `coeffs[n−1−i] · k^(n−1−i)`. Scaling and reversing were two separate
    // passes, each allocating a full vector of `Integer` clones.
    let mut c: Vec<Integer> = vec![Integer::new(); n];
    let ki = Integer::from(k);
    let mut power = Integer::from(1);
    for (j, coef) in coeffs.iter().enumerate() {
        c[n - 1 - j] = Integer::from(coef * &power);
        if j + 1 < n {
            power *= &ki;
        }
    }
    // If the reversed, scaled coefficients already have no sign variation then
    // `p(k·u)` has no positive root at all, so `(0, k)` is empty and the O(n²)
    // shift can be skipped outright.
    if sign_variations(&c) == 0 {
        return true;
    }
    taylor_shift_by_1_in_place(&mut c);
    sign_variations(&c) == 0
}

/// Integer lower bound on the smallest positive root of `p`.
///
/// Uses a doubling-then-binary-search over integer evaluation points, then
/// **certifies** the candidate with Descartes' rule before returning it.
/// Precondition: `p(0) ≠ 0` (no root at the origin).
/// Returns an integer `k ≥ 1` for which `p` is proved to have no root in
/// `(0, k)`, or `0` when no such bound could be certified.
///
/// The sign search alone is not sound, and returning its answer directly is
/// how [`real_roots`] used to lose roots. Its stated rule — "`p(k)` has the
/// same sign as `p(0)`, implying all positive roots are `> k`" — is false:
/// equal signs at `0` and `k` imply an *even* number of roots in `(0, k)`,
/// which may be two rather than none. For `25x³ − 325x² + 804x − 540 =
/// 25(x − 6/5)(x − 9/5)(x − 10)` the polynomial is negative at every integer
/// from 0 to 9, so the search returned `k = 9`, [`isolate_positive_roots`]
/// shifted the frame past both `6/5` and `9/5`, and `real_roots` reported a
/// single root where there are three — with no error and no flag. Chebyshev
/// `T₆` lost four of its six roots the same way.
///
/// The sign search is kept as a *proposal* (it is cheap and usually right)
/// and halved until it is certified, so the returned bound is sound by
/// construction.
///
/// `sign_var` must be `sign_variations(coeffs)`, which the caller has already
/// computed. It lets most proposals be certified by a counting argument that
/// costs nothing, leaving the explicit Descartes test — the expensive part —
/// only for `sign_var ≥ 3`. See the comment at the certification step.
fn cf_lower_bound_floor(coeffs: &[Integer], sign_var: usize) -> u64 {
    if coeffs.is_empty() {
        return 0;
    }
    let n = coeffs.len() - 1;
    if n == 0 {
        return 0;
    }

    let p0 = &coeffs[0];
    if *p0 == 0 {
        return 0;
    }
    let sign = *p0 > 0;

    // Horner evaluation at a non-negative integer point.
    let eval_at = |k: u64| -> Integer {
        let k_int = Integer::from(k);
        coeffs
            .iter()
            .rev()
            .fold(Integer::from(0), |acc, c| acc * k_int.clone() + c.clone())
    };

    // If p(1) has a different sign (or is zero), the root is in (0, 1].
    let p1 = evaluate_at_1(coeffs);
    if p1 == 0 || (p1 > 0) != sign {
        return 0;
    }

    // Doubling search: find hi where sign changes.
    let mut lo: u64 = 1;
    let mut hi: u64 = 2;
    let mut found_sign_change = false;
    loop {
        if hi > 1_000_000 {
            break;
        }
        let pval = eval_at(hi);
        if pval == 0 || (pval > 0) != sign {
            found_sign_change = true;
            break;
        }
        lo = hi;
        hi = hi.saturating_mul(2);
    }

    // No sign change found → polynomial is positive for all integers in [1, limit],
    // meaning all positive roots are in (0, 1).  No shift is useful.
    if !found_sign_change {
        return 0;
    }

    // Binary search for the transition.
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let pval = eval_at(mid);
        if pval == 0 || (pval > 0) != sign {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // Certify the proposal, halving until it is proved sound. `k = 0` is
    // trivially certified, so this always terminates with a sound answer.
    //
    // At this point the search has established, for the proposed `lo ≥ 1`:
    //   * `p(0)` and `p(lo)` are both non-zero and share a sign, so the number
    //     of roots in `(0, lo)` counted with multiplicity is **even** — the
    //     very fact the old code mistook for "zero";
    //   * `p(lo+1)` is zero or has the opposite sign, so `(lo, lo+1]` contains
    //     at least **one** root counted with multiplicity.
    //
    // Descartes bounds the total number of positive roots, with multiplicity,
    // by `sign_var`. So if `(0, lo)` were non-empty it would hold at least two
    // roots, and with the one in `(lo, lo+1]` the total would be at least
    // three. For `sign_var ≤ 2` that is a contradiction, and the proposal is
    // certified with no further work.
    //
    // Only `sign_var ≥ 3` needs the explicit test, which is exactly the regime
    // of the polynomials that used to lose roots: `25x³ − 325x² + 804x − 540`
    // and Chebyshev `T₆` both have three sign variations.
    if sign_var > 2 {
        while lo >= 1 && !no_root_in_open_interval(coeffs, lo) {
            lo /= 2;
        }
    }

    lo
}

// ---------------------------------------------------------------------------
// Main VAS bisection algorithm
// ---------------------------------------------------------------------------

/// Stack frame: polynomial together with the Möbius transform tracking which
/// sub-interval of the original positive half-line this frame covers.
///
/// Invariant: the positive real roots of `poly(t)` biject with the roots of
/// the original squarefree polynomial in `(b/d, a/c)` (or `(b/d, +∞)` when
/// `c = 0`) via `x = (a·t + b)/(c·t + d)`.
struct Frame {
    poly: Vec<Integer>,
    a: Integer,
    b: Integer,
    c: Integer,
    d: Integer,
    /// True immediately after a root-at-0 or root-at-1 deflation.
    /// When set, skip the `sign_var == 1` shortcut and always bisect.
    just_deflated: bool,
}

/// Compute both endpoints of the Möbius interval.
///
/// - `at_zero  = b/d`  (value at `t = 0`)
/// - `at_inf   = a/c`  (value at `t → ∞`, or `None` when `c = 0`)
///
/// Returns `(lo, hi)` with `lo ≤ hi`.
fn mobius_interval(
    a: &Integer,
    b: &Integer,
    c: &Integer,
    d: &Integer,
) -> (rug::Rational, Option<rug::Rational>) {
    let at_zero = rug::Rational::from((b.clone(), d.clone()));
    let at_inf = if *c == 0 {
        None
    } else {
        Some(rug::Rational::from((a.clone(), c.clone())))
    };
    match at_inf {
        None => (at_zero, None),
        Some(ai) => {
            if at_zero <= ai {
                (at_zero, Some(ai))
            } else {
                (ai, Some(at_zero))
            }
        }
    }
}

/// Isolate all strictly-positive real roots of `coeffs` via VAS bisection.
///
/// The input polynomial must have a **non-zero constant term** (root at `x = 0`
/// should be removed before calling this function).
fn isolate_positive_roots(coeffs: Vec<Integer>) -> Vec<RootInterval> {
    if coeffs.is_empty() || coeffs.iter().all(|c| *c == 0) {
        return vec![];
    }

    let mut result = Vec::new();
    let mut stack: Vec<Frame> = vec![Frame {
        poly: coeffs,
        a: Integer::from(1),
        b: Integer::from(0),
        c: Integer::from(0),
        d: Integer::from(1),
        just_deflated: false,
    }];

    let max_iters: usize = 500_000;
    let mut iters = 0usize;

    while let Some(mut frame) = stack.pop() {
        iters += 1;
        if iters > max_iters {
            break;
        }

        trim_trailing_zeros(&mut frame.poly);
        if frame.poly.is_empty() || frame.poly.iter().all(|c| *c == 0) {
            continue;
        }

        // ---- Root at t = 0 (constant term = 0) --------------------------------
        // t = 0 corresponds to x = b/d.
        if frame.poly[0] == 0 {
            let root_x = rug::Rational::from((frame.b.clone(), frame.d.clone()));
            result.push(RootInterval::new(root_x.clone(), root_x));
            frame.poly = divide_by_t(&frame.poly);
            trim_trailing_zeros(&mut frame.poly);
            if frame.poly.is_empty() {
                continue;
            }
            // Push back with just_deflated=true so the sign_var=1 shortcut is
            // suppressed (the remaining roots are strictly in (b/d, …), but the
            // Möbius still starts at b/d, risking a half-open overlap).
            frame.just_deflated = true;
            stack.push(frame);
            continue;
        }

        // ---- Root at t = 1 (p(1) = sum of coefficients = 0) ------------------
        // t = 1 corresponds to x = (a+b)/(c+d).
        let val_at_1 = evaluate_at_1(&frame.poly);
        if val_at_1 == 0 {
            let a_plus_b = frame.a.clone() + frame.b.clone();
            let c_plus_d = frame.c.clone() + frame.d.clone();
            if c_plus_d != 0 {
                let root_x = rug::Rational::from((a_plus_b, c_plus_d));
                result.push(RootInterval::new(root_x.clone(), root_x));
            }
            frame.poly = divide_by_t_minus_1(&frame.poly);
            trim_trailing_zeros(&mut frame.poly);
            if frame.poly.is_empty() {
                continue;
            }
            // After deflation by (t−1) the remaining roots are NOT all in
            // (1,∞); they could be anywhere in (0,∞).  Force a bisect pass
            // so that the children's intervals are strictly disjoint from the
            // just-recorded exact root at the split point.
            frame.just_deflated = true;
            stack.push(frame);
            continue;
        }

        let v = sign_variations(&frame.poly);

        match v {
            0 => continue,
            1 if !frame.just_deflated => {
                // Exactly one root; if the tracking interval is bounded record it.
                let (lo, hi_opt) = mobius_interval(&frame.a, &frame.b, &frame.c, &frame.d);
                if let Some(hi) = hi_opt {
                    result.push(RootInterval::new(lo, hi));
                    continue;
                }
                // Unbounded interval (c = 0): fall through to CF + bisect to
                // narrow down a finite upper bound.
            }
            _ => {
                // v == 0 handled above; v ≥ 2 or v == 1 with just_deflated falls here.
            }
        }

        // ---- VAS CF step: shift by integer lower bound k ----------------------
        frame.just_deflated = false; // reset flag before bisection

        let k = cf_lower_bound_floor(&frame.poly, v);
        if k >= 1 {
            let new_p = taylor_shift_by(&frame.poly, k);
            let ki = Integer::from(k);
            let new_b = frame.a.clone() * ki.clone() + frame.b.clone();
            let new_d = frame.c.clone() * ki + frame.d.clone();
            frame.b = new_b;
            frame.d = new_d;
            frame.poly = remove_content(&new_p);
            trim_trailing_zeros(&mut frame.poly);
            if frame.poly.is_empty() {
                continue;
            }
            // Push back so the root-at-0 / root-at-1 checks fire before bisection.
            stack.push(frame);
            continue;
        }

        // ---- Bisect at t = 1 --------------------------------------------------

        let a = frame.a.clone();
        let b = frame.b.clone();
        let c = frame.c.clone();
        let d = frame.d.clone();

        // Right child: roots of q in (1, ∞)  →  poly = q(t+1), Möbius (a, a+b, c, c+d).
        {
            let q_right_raw = taylor_shift_by_1(&frame.poly);
            let mut q_right = remove_content(&q_right_raw);
            trim_trailing_zeros(&mut q_right);
            if !q_right.is_empty() && q_right.iter().any(|c| *c != 0) {
                stack.push(Frame {
                    poly: q_right,
                    a: a.clone(),
                    b: a.clone() + b.clone(),
                    c: c.clone(),
                    d: c.clone() + d.clone(),
                    just_deflated: false,
                });
            }
        }

        // Left child: roots of q in (0, 1)  →  poly = (t+1)ⁿ·q(1/(t+1))
        //            = taylor_shift_1(reverse(q)), Möbius (b, a+b, d, c+d).
        {
            let rev = reverse_coeffs(&frame.poly);
            let q_left_raw = taylor_shift_by_1(&rev);
            let mut q_left = remove_content(&q_left_raw);
            trim_trailing_zeros(&mut q_left);
            if !q_left.is_empty() && q_left.iter().any(|c| *c != 0) {
                stack.push(Frame {
                    poly: q_left,
                    a: b.clone(),
                    b: a + b,
                    c: d.clone(),
                    d: c + d,
                    just_deflated: false,
                });
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Exact rational-root recovery
// ---------------------------------------------------------------------------

/// Evaluate `p` at a rational point in **integer** arithmetic, preserving sign
/// and vanishing.
///
/// For `x = n/d` in canonical form this returns the homogeneous form
/// `H(n, d) = Σ cᵢ·nⁱ·d^(deg−i)`, which is exactly `p(x)·d^deg`. A canonical
/// [`rug::Rational`] has `d > 0`, so `d^deg > 0` and therefore `H` vanishes
/// precisely when `p(x)` does and otherwise carries the same sign.
///
/// Those two facts — vanishing and sign — are all that
/// [`exact_rational_root`] and [`refine_root`] ever ask of an evaluation, and
/// getting them this way avoids rational arithmetic entirely. The previous
/// `rug::Rational` Horner spent three `mpz` GCD canonicalisations and roughly
/// four allocations *per coefficient*, measured at ~17 800 instructions for a
/// single degree-8 evaluation — enough to make rational-root recovery cost
/// half as much again as the whole of `real_roots`. This form is a plain
/// Horner loop with an `mpz_addmul` and no GCD at all.
fn eval_coeffs_homogeneous(coeffs: &[Integer], x: &rug::Rational) -> Integer {
    let n = x.numer();
    let d = x.denom();
    let mut acc = Integer::new();
    // `dp` is `d^k` where `k` counts completed steps, so that the coefficient
    // `c_{deg−k}` is scaled by `d^k` exactly as the homogeneous form requires.
    let mut dp = Integer::from(1);
    let unit_denom = *d == 1;
    for c in coeffs.iter().rev() {
        acc *= n;
        if unit_denom {
            acc += c;
        } else {
            acc += c * &dp;
            dp *= d;
        }
    }
    acc
}

/// Bisection budget for exact rational-root recovery.
///
/// Each halving is one polynomial evaluation, and the loop stops as soon as
/// the bracket is narrower than `1/lc`, so this ceiling is only reached for a
/// bracket that started astronomically wide relative to the leading
/// coefficient — in which case the interval is left alone and behaviour is
/// exactly what it was before.
const RATIONAL_RECOVERY_BISECTIONS: u32 = 512;

/// Leading-coefficient size above which recovery is not attempted.
///
/// The search is over multiples of `1/lc`, so its cost is driven by the size of
/// `lc` rather than by the degree. Beyond this the bracket is returned
/// unchanged: a loose bracket is a weaker answer, never a wrong one.
const RATIONAL_RECOVERY_MAX_LC_BITS: u32 = 128;

/// Recover the **exact** rational root inside `iv`, when the root is rational.
///
/// [`RootInterval`] documents that an exact rational root `r` is reported as
/// `lo == hi == r`, and every caller that has to decide something *at* a root
/// — CAD cell sampling, and therefore [`crate::real::cad::decide`] — depends on
/// it: a sample set built from bracket endpoints and midpoints contains only
/// dyadic rationals, so a root like `2/3` is never tested, and a sentence whose
/// truth turns on that point (`∀x. (3x+2)² > 0`) is decided wrong with no
/// indication that anything was skipped.
///
/// The VAS isolator only delivers `lo == hi` when a root is found exactly at a
/// Möbius endpoint (`t = 0` or `t = 1`), which happens for dyadic roots and not
/// in general. This pass closes the gap.
///
/// The search is exact, not a heuristic: by the rational-root theorem every
/// rational root of an integer polynomial has denominator dividing the leading
/// coefficient `lc`, so once the bracket is narrower than `1/lc` it contains at
/// most one such rational, and that single candidate is checked by exact
/// evaluation. `None` means "no rational root here", never "probably not".
///
/// `coeffs` must be **squarefree** — bisection needs the sign change that a
/// simple root guarantees.
fn exact_rational_root(coeffs: &[Integer], iv: &RootInterval) -> Option<rug::Rational> {
    if iv.lo == iv.hi {
        return Some(iv.lo.clone());
    }
    let lc = coeffs.last()?.clone().abs();
    if lc.is_zero() || lc.significant_bits() > RATIONAL_RECOVERY_MAX_LC_BITS {
        return None;
    }

    let mut lo = iv.lo.clone();
    let mut hi = iv.hi.clone();
    let v_lo = eval_coeffs_homogeneous(coeffs, &lo);
    let v_hi = eval_coeffs_homogeneous(coeffs, &hi);
    if v_lo == 0 || v_hi == 0 || (v_lo > 0) == (v_hi > 0) {
        // Bisection needs a strict sign change across the bracket. A vanishing
        // endpoint is *not* good enough: neighbouring brackets share endpoints,
        // so an endpoint root generally belongs to the neighbour, and
        // collapsing onto it would silently delete the root this bracket was
        // isolating. Leave the bracket alone — a loose bracket is a weaker
        // answer, a lost root is a wrong one.
        return None;
    }
    let lo_positive = v_lo > 0;

    // Narrow until at most one multiple of 1/lc can remain inside.
    let target = rug::Rational::from((Integer::from(1), lc.clone()));
    for _ in 0..RATIONAL_RECOVERY_BISECTIONS {
        if hi.clone() - lo.clone() < target {
            break;
        }
        let mid = (lo.clone() + hi.clone()) / rug::Rational::from(2);
        let v = eval_coeffs_homogeneous(coeffs, &mid);
        if v == 0 {
            return Some(mid);
        }
        if (v > 0) == lo_positive {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Any rational root has denominator dividing `lc`, i.e. is `n/lc` for an
    // integer `n`. At most two such points survive a bracket this narrow.
    let scaled_lo = lo * rug::Rational::from((lc.clone(), Integer::from(1)));
    let scaled_hi = hi * rug::Rational::from((lc.clone(), Integer::from(1)));
    let (mut n, _) = scaled_lo
        .numer()
        .clone()
        .div_rem_ceil(scaled_lo.denom().clone());
    let (n_max, _) = scaled_hi
        .numer()
        .clone()
        .div_rem_floor(scaled_hi.denom().clone());
    while n <= n_max {
        let candidate = rug::Rational::from((n.clone(), lc.clone()));
        if eval_coeffs_homogeneous(coeffs, &candidate) == 0 {
            return Some(candidate);
        }
        n += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Isolate all real roots of `poly`.
///
/// Returns a vector of [`RootInterval`]s sorted by lower endpoint.  Each
/// interval contains exactly one real root of the squarefree part of `poly`.
/// Repeated roots appear once each.
///
/// # Errors
///
/// - [`RealRootError::ZeroPolynomial`] — `poly` is the zero polynomial.
pub fn real_roots(poly: &UniPoly) -> Result<Vec<RootInterval>, RealRootError> {
    let mut coeffs: Vec<Integer> = poly.coefficients();
    trim_trailing_zeros(&mut coeffs);

    if coeffs.is_empty() {
        return Err(RealRootError::ZeroPolynomial);
    }
    if coeffs.len() == 1 {
        return Ok(vec![]); // Non-zero constant: no roots.
    }

    // Normalise to positive leading coefficient.
    if coeffs.last().is_some_and(|v| *v < 0) {
        for c in coeffs.iter_mut() {
            *c = Integer::from(0) - c.clone();
        }
    }

    // Squarefree part.
    let sq = squarefree_part(&coeffs);

    // Check for root at x = 0 (constant term = 0).
    let mut result = Vec::new();
    let working = if sq[0] == 0 {
        result.push(RootInterval::new(
            rug::Rational::from(0),
            rug::Rational::from(0),
        ));
        sq[1..].to_vec()
    } else {
        sq.clone()
    };

    if working.len() <= 1 {
        result.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap_or(std::cmp::Ordering::Equal));
        return Ok(result);
    }

    // Positive roots.
    result.extend(isolate_positive_roots(working.clone()));

    // Negative roots: positive roots of p(−x), then negate.
    let neg_coeffs: Vec<Integer> = working
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i % 2 == 1 {
                Integer::from(0) - c.clone()
            } else {
                c.clone()
            }
        })
        .collect();
    let neg_pos = isolate_positive_roots(neg_coeffs);
    for iv in neg_pos {
        let neg_hi = rug::Rational::from((
            Integer::from(0) - iv.lo.numer().clone(),
            iv.lo.denom().clone(),
        ));
        let neg_lo = rug::Rational::from((
            Integer::from(0) - iv.hi.numer().clone(),
            iv.hi.denom().clone(),
        ));
        result.push(RootInterval::new(neg_lo, neg_hi));
    }

    // Honour the documented contract: an exact rational root is reported as
    // `lo == hi == r`. VAS only produces that for roots it happens to land on
    // (dyadic ones); `2/3` came back as the bracket `[0, 1]`, and CAD's sample
    // set — bracket endpoints and midpoints, all dyadic — then never tests the
    // root itself.
    //
    // Against `working`, not `sq`: `working` is `sq` with the root at the origin
    // divided out, and it is the polynomial whose roots these brackets isolate.
    // Using `sq` made every bracket with `0` as an endpoint collapse onto the
    // origin, which loses a root outright.
    for iv in result.iter_mut() {
        if iv.lo == iv.hi {
            continue;
        }
        if let Some(r) = exact_rational_root(&working, iv) {
            *iv = RootInterval::new(r.clone(), r);
        }
    }

    result.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap_or(std::cmp::Ordering::Equal));
    Ok(result)
}

/// Isolate all real roots of a symbolic expression in `var`.
///
/// Coefficients may be exact rationals (e.g. `x^2 - 1/2`): they are cleared
/// to an integer-coefficient polynomial with the same real roots (by scaling
/// by the LCM of the coefficient denominators) before isolation, since
/// multiplying a polynomial by a nonzero constant does not change its real
/// roots.
///
/// # Errors
///
/// - [`RealRootError::NotAPolynomial`] if the expression cannot be converted.
/// - [`RealRootError::ZeroPolynomial`] if the polynomial is identically zero.
pub fn real_roots_symbolic(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<RootInterval>, RealRootError> {
    let poly = UniPoly::from_symbolic_clear_denoms(expr, var, pool)
        .map_err(RealRootError::NotAPolynomial)?;
    real_roots(&poly)
}

/// The smallest `f64` strictly greater than `v`, for finite `v ≥ 0`.
fn next_up_nonneg(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    if v == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(v.to_bits() + 1)
}

/// Smallest `f64` that is `≥ r`, for a non-negative rational `r`.
///
/// `Rational::to_f64` rounds to nearest, which can land *below* `r` — and a
/// radius rounded down is a ball that does not contain what it claims to.
fn round_up_f64(r: &rug::Rational) -> f64 {
    let v = r.to_f64();
    if !v.is_finite() {
        return v;
    }
    match rug::Rational::from_f64(v) {
        Some(back) if back >= *r => v,
        _ => next_up_nonneg(v),
    }
}

/// Build a ball that provably contains every point of the exact rational
/// interval `[lo, hi]`.
///
/// Both the midpoint and the radius are `f64`, so both are rounded; the
/// midpoint may round either way, and the radius is therefore measured
/// *against the rounded midpoint* and then rounded **up**.
fn ball_covering(lo: &rug::Rational, hi: &rug::Rational, prec: u32) -> ArbBall {
    let mid_rat = rug::Rational::from(lo + hi) / 2u32;
    let center = mid_rat.to_f64();
    let Some(center_rat) = rug::Rational::from_f64(center) else {
        return ArbBall::infinity(prec.max(53));
    };
    let left = rug::Rational::from(&center_rat - lo);
    let right = rug::Rational::from(hi - &center_rat);
    let rad_rat = if left > right { left } else { right };
    let radius = if rad_rat <= 0 {
        0.0
    } else {
        round_up_f64(&rad_rat)
    };
    ArbBall::from_midpoint_radius(center, radius, prec.max(53))
}

/// Narrow a [`RootInterval`] to at least `prec` bits of precision.
///
/// Bisection is performed in **exact rational arithmetic**, and the returned
/// ball is rounded outwards, so it genuinely contains the root — which is what
/// every caller of a "rigorous enclosure" is entitled to assume.
///
/// The previous implementation did neither, and both shortcuts were
/// observable. It bisected with an `f64` Horner evaluation, so for
/// `(10⁹x − 1414213562)(x² − 2)` the sign test `f_lo * f_mid <= 0` was wrong at
/// every step, `hi` collapsed onto `lo`, and the result was an *exact*
/// (zero-radius) ball at `1.41421356205…`, which is not a root of anything —
/// the root in that bracket is `√2`. And even when the bracket was right, the
/// ball was built as `mid = (lo+hi)/2`, `rad = (hi-lo)/2` with round-to-nearest
/// on both, so for `x² − 2` it returned `mid = 1.414213562373095`,
/// `rad = 1.11e-16` whose upper end `mid + rad` is still strictly below `√2`:
/// `(mid + rad)² − 2 = −4.06e-17 < 0` in exact arithmetic. `contains(√2)` was
/// `false` for the ball that was supposed to enclose `√2`.
///
/// For an exact rational root (`lo == hi`) the radius covers the rounding of
/// that rational to `f64`, which is zero only when the rational is itself
/// representable — `6/5` is not.
pub fn refine_root(poly: &UniPoly, interval: &RootInterval, prec: u32) -> ArbBall {
    let prec = prec.max(53);
    if interval.lo == interval.hi {
        return ball_covering(&interval.lo, &interval.hi, prec);
    }

    let coeffs: Vec<Integer> = poly.coefficients();
    let mut lo = interval.lo.clone();
    let mut hi = interval.hi.clone();
    let mut f_lo = eval_coeffs_homogeneous(&coeffs, &lo);
    if f_lo == 0 {
        return ball_covering(&lo, &lo, prec);
    }
    let f_hi = eval_coeffs_homogeneous(&coeffs, &hi);
    if f_hi == 0 {
        return ball_covering(&hi, &hi, prec);
    }
    // Without a strict sign change there is nothing to bisect against; return
    // the bracket as given rather than narrowing onto an arbitrary endpoint.
    if (f_lo > 0) == (f_hi > 0) {
        return ball_covering(&lo, &hi, prec);
    }

    let target_width = rug::Rational::from((Integer::from(1), Integer::from(1) << prec));
    let steps = (prec as usize + 2).saturating_mul(2).min(4096);
    for _ in 0..steps {
        if rug::Rational::from(&hi - &lo) <= target_width {
            break;
        }
        let mid = rug::Rational::from(&lo + &hi) / 2u32;
        let f_mid = eval_coeffs_homogeneous(&coeffs, &mid);
        if f_mid == 0 {
            return ball_covering(&mid, &mid, prec);
        }
        if (f_lo > 0) != (f_mid > 0) {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }

    ball_covering(&lo, &hi, prec)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flint::{FlintInteger, FlintPoly};
    use crate::kernel::{Domain, ExprPool};

    /// Build a `UniPoly` from a slice of `i64` coefficients (ascending degree).
    fn make_poly(coeffs: &[i64]) -> UniPoly {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let mut flint = FlintPoly::new();
        for (i, &c) in coeffs.iter().enumerate() {
            let fi = FlintInteger::from_i64(c);
            flint.set_coeff_flint(i, &fi);
        }
        UniPoly {
            var: x,
            coeffs: flint,
        }
    }

    // ---- sign_variations ----

    #[test]
    fn sv_all_positive() {
        let c: Vec<Integer> = vec![1, 2, 3].into_iter().map(Integer::from).collect();
        assert_eq!(sign_variations(&c), 0);
    }

    #[test]
    fn sv_alternating() {
        let c: Vec<Integer> = vec![1, -1, 1, -1i64]
            .into_iter()
            .map(Integer::from)
            .collect();
        assert_eq!(sign_variations(&c), 3);
    }

    #[test]
    fn sv_with_zeros() {
        // Zeros are ignored: [1, 0, -1] → one sign change.
        let c: Vec<Integer> = vec![1i64, 0, -1].into_iter().map(Integer::from).collect();
        assert_eq!(sign_variations(&c), 1);
    }

    // ---- taylor_shift_by_1 ----

    #[test]
    fn taylor_shift_quadratic() {
        // p(x) = x² + 2x + 1 = [1,2,1]; p(x+1) = x² + 4x + 4 = [4,4,1].
        let c: Vec<Integer> = vec![1, 2, 1i64].into_iter().map(Integer::from).collect();
        let shifted = taylor_shift_by_1(&c);
        let expected: Vec<Integer> = vec![4, 4, 1i64].into_iter().map(Integer::from).collect();
        assert_eq!(shifted, expected);
    }

    #[test]
    fn taylor_shift_linear() {
        // p(x) = 3x + 2; p(x+1) = 3x + 5; [2,3] → [5,3].
        let c: Vec<Integer> = vec![2, 3i64].into_iter().map(Integer::from).collect();
        let shifted = taylor_shift_by_1(&c);
        assert_eq!(shifted[0], Integer::from(5i64));
        assert_eq!(shifted[1], Integer::from(3i64));
    }

    // ---- squarefree_part ----

    #[test]
    fn sqfree_linear_already_squarefree() {
        let c: Vec<Integer> = vec![-1i64, 1].into_iter().map(Integer::from).collect();
        let sq = squarefree_part(&c);
        assert_eq!(sq.len(), 2);
    }

    #[test]
    fn sqfree_removes_double_root() {
        // (x-1)² = x² - 2x + 1 = [1,-2,1]; squarefree part = x - 1 (degree 1).
        let c: Vec<Integer> = vec![1i64, -2, 1].into_iter().map(Integer::from).collect();
        let sq = squarefree_part(&c);
        assert_eq!(sq.len(), 2, "squarefree part of (x-1)² must be degree 1");
    }

    #[test]
    fn sqfree_triple_root() {
        // (x-2)³ = x³ - 6x² + 12x - 8 = [-8,12,-6,1]; squarefree part = x-2.
        let c: Vec<Integer> = vec![-8i64, 12, -6, 1]
            .into_iter()
            .map(Integer::from)
            .collect();
        let sq = squarefree_part(&c);
        assert_eq!(sq.len(), 2, "squarefree part of (x-2)³ must be degree 1");
    }

    // ---- divide_by_t_minus_1 ----

    #[test]
    fn div_t_minus_1_basic() {
        // x² - 1 = (x-1)(x+1); dividing by (t-1) gives (x+1) = [1,1].
        let c: Vec<Integer> = vec![-1i64, 0, 1].into_iter().map(Integer::from).collect();
        assert_eq!(evaluate_at_1(&c), Integer::from(0i64));
        let q = divide_by_t_minus_1(&c);
        assert_eq!(q.len(), 2);
        // x² - 1 = [-1, 0, 1]; divide by (t-1):
        //   q[1] = coeffs[2] = 1
        //   q[0] = coeffs[1] + q[1] = 0 + 1 = 1
        // → q = [1, 1] = x + 1, ascending order.
        assert_eq!(
            q[0],
            Integer::from(1i64),
            "constant term of x+1 should be 1"
        );
        assert_eq!(
            q[1],
            Integer::from(1i64),
            "x-coefficient of x+1 should be 1"
        );
    }

    // ---- poly_pseudo_rem ----

    #[test]
    fn pseudo_rem_double_root() {
        // prem(x² - 2x + 1, 2x - 2) should give 0 (since gcd = x-1).
        let a: Vec<Integer> = vec![1i64, -2, 1].into_iter().map(Integer::from).collect();
        let b: Vec<Integer> = vec![-2i64, 2].into_iter().map(Integer::from).collect();
        let r = poly_pseudo_rem(&a, &b);
        assert!(
            r.iter().all(|c| *c == 0),
            "prem of (x-1)² by 2(x-1) should be 0, got {:?}",
            r
        );
    }

    // ---- isolate_positive_roots ----

    #[test]
    fn isolate_x_minus_1() {
        let c: Vec<Integer> = vec![-1i64, 1].into_iter().map(Integer::from).collect();
        let roots = isolate_positive_roots(c);
        assert_eq!(roots.len(), 1);
        assert!(roots[0].lo <= 1);
        assert!(roots[0].hi >= 1);
    }

    #[test]
    fn isolate_x_squared_minus_1_positive() {
        // x² - 1 = (x-1)(x+1); one positive root at x=1.
        let c: Vec<Integer> = vec![-1i64, 0, 1].into_iter().map(Integer::from).collect();
        let roots = isolate_positive_roots(c);
        assert_eq!(roots.len(), 1);
        assert!(roots[0].lo <= 1);
        assert!(roots[0].hi >= 1);
    }

    #[test]
    fn isolate_two_positive_roots() {
        // (x-1)(x-2) = x² - 3x + 2; roots at 1 and 2.
        let c: Vec<Integer> = vec![2i64, -3, 1].into_iter().map(Integer::from).collect();
        let roots = isolate_positive_roots(c);
        assert_eq!(roots.len(), 2, "expected 2 positive roots");
        let mut sorted = roots;
        sorted.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap());
        // Intervals must be disjoint: sorted[0].hi ≤ sorted[1].lo.
        assert!(
            sorted[0].hi <= sorted[1].lo,
            "intervals must be disjoint: [{},{}] and [{},{}]",
            sorted[0].lo,
            sorted[0].hi,
            sorted[1].lo,
            sorted[1].hi
        );
    }

    // ---- real_roots ----

    #[test]
    fn real_roots_x_squared_minus_1() {
        let poly = make_poly(&[-1, 0, 1]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 2, "x² - 1 has 2 real roots");
        assert!(roots[0].lo < 0);
        assert!(roots[1].lo >= 0);
    }

    #[test]
    fn real_roots_no_real_roots() {
        // x² + 1 has no real roots.
        let poly = make_poly(&[1, 0, 1]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn real_roots_cluster_squarefree() {
        // (x-1)⁵·(x+1)³ has roots at ±1; squarefree part = x²-1.
        let poly = make_poly(&[-1, 0, 1]); // Use squarefree representative.
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn real_roots_disjoint() {
        // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6; roots at 1, 2, 3.
        let poly = make_poly(&[-6, 11, -6, 1]);
        let mut roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 3);
        roots.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap());
        for w in roots.windows(2) {
            assert!(
                w[0].hi <= w[1].lo,
                "intervals must be disjoint: {} and {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn homogeneous_eval_agrees_with_rational_eval_on_sign_and_zero() {
        // The whole point of `eval_coeffs_homogeneous` is that it is a drop-in
        // replacement wherever only the sign and the vanishing of `p(x)` are
        // consulted.  Check that against exact rational evaluation.
        let rational_eval = |coeffs: &[Integer], x: &rug::Rational| -> rug::Rational {
            let mut acc = rug::Rational::from(0);
            for c in coeffs.iter().rev() {
                acc *= x;
                acc += rug::Rational::from((c.clone(), Integer::from(1)));
            }
            acc
        };
        let polys: [&[i64]; 4] = [
            &[-540, 804, -325, 25],         // roots 6/5, 9/5, 10
            &[-1, -1, 0, 0, 0, 0, 0, 0, 1], // x⁸ − x − 1
            &[640, -248, 24],               // roots 5, 16/3
            &[1, 0, 1],                     // no real root
        ];
        for p in polys {
            let coeffs: Vec<Integer> = p.iter().map(|v| Integer::from(*v)).collect();
            for num in -25i64..=25 {
                for den in 1i64..=12 {
                    let x = rug::Rational::from((num, den));
                    let h = eval_coeffs_homogeneous(&coeffs, &x);
                    let r = rational_eval(&coeffs, &x);
                    assert_eq!(h == 0, r == 0, "vanishing disagrees at {x} for {p:?}");
                    if h != 0 {
                        assert_eq!(h > 0, r > 0, "sign disagrees at {x} for {p:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn real_roots_three_rational_roots_kept() {
        // 25x³ − 325x² + 804x − 540 = 25(x − 6/5)(x − 9/5)(x − 10).
        //
        // The polynomial is negative at every integer from 0 to 9, so the
        // `cf_lower_bound_floor` sign search proposes k = 9 and shifting by it
        // would step past both 6/5 and 9/5.  Three sign variations, so the
        // proposal is not covered by the counting argument and the explicit
        // Descartes certification must reject it down to k = 1.
        let poly = make_poly(&[-540, 804, -325, 25]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 3, "25(x−6/5)(x−9/5)(x−10) has 3 real roots");
    }

    #[test]
    fn real_roots_chebyshev_t6_all_six() {
        // T₆(x) = 32x⁶ − 48x⁴ + 18x² − 1; 6 roots in (−1, 1).  Also three sign
        // variations, and used to report only 2.
        let poly = make_poly(&[-1, 0, 18, 0, -48, 0, 32]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 6, "T₆ has 6 real roots");
        for r in &roots {
            assert!(r.lo >= -1);
            assert!(r.hi <= 1);
        }
    }

    #[test]
    fn cf_lower_bound_is_certified_for_high_sign_variation() {
        // Guard the counting argument itself: for the 25x³ polynomial the
        // sign search alone proposes 9, and the certified bound must be 1.
        let coeffs: Vec<Integer> = [-540, 804, -325, 25]
            .iter()
            .map(|v| Integer::from(*v))
            .collect();
        let v = sign_variations(&coeffs);
        assert_eq!(v, 3, "this polynomial has three sign variations");
        assert_eq!(
            cf_lower_bound_floor(&coeffs, v),
            1,
            "certification must reject the uncertified proposal k = 9"
        );
        // And the test it relies on agrees: there *is* a root below 9.
        assert!(!no_root_in_open_interval(&coeffs, 9));
        assert!(no_root_in_open_interval(&coeffs, 1));
    }

    #[test]
    fn real_roots_chebyshev_t4() {
        // T₄(x) = 8x⁴ - 8x² + 1; 4 roots in (-1, 1).
        let poly = make_poly(&[1, 0, -8, 0, 8]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 4, "T₄ has 4 real roots");
        for r in &roots {
            assert!(r.lo >= -1);
            assert!(r.hi <= 1);
        }
    }

    #[test]
    fn real_roots_constant_zero() {
        let poly = make_poly(&[0]);
        assert!(matches!(
            real_roots(&poly),
            Err(RealRootError::ZeroPolynomial)
        ));
    }

    #[test]
    fn real_roots_constant_nonzero() {
        let poly = make_poly(&[5]);
        assert_eq!(real_roots(&poly).unwrap().len(), 0);
    }

    #[test]
    fn real_roots_symbolic_x_squared_minus_4() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let xsq = p.pow(x, p.integer(2_i32));
        let expr = p.add(vec![xsq, p.integer(-4_i32)]);
        let roots = real_roots_symbolic(expr, x, &p).unwrap();
        assert_eq!(roots.len(), 2);
        assert!(roots[0].lo <= -2);
        assert!(roots[0].hi >= -2);
        assert!(roots[1].lo <= 2);
        assert!(roots[1].hi >= 2);
    }

    #[test]
    fn real_roots_five_distinct() {
        // (x-1)(x-2)(x-3)(x-4)(x-5) = x⁵ - 15x⁴ + 85x³ - 225x² + 274x - 120.
        let poly = make_poly(&[-120, 274, -225, 85, -15, 1]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 5, "product (x-1)…(x-5) must have 5 real roots");
        // Each known root must be enclosed.
        for k in 1..=5i64 {
            let rk = rug::Rational::from(k);
            assert!(
                roots.iter().any(|iv| iv.lo <= rk && iv.hi >= rk),
                "root at x={k} not enclosed"
            );
        }
    }

    #[test]
    fn real_roots_disjoint_five() {
        let poly = make_poly(&[-120, 274, -225, 85, -15, 1]);
        let mut roots = real_roots(&poly).unwrap();
        roots.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap());
        for w in roots.windows(2) {
            assert!(
                w[0].hi <= w[1].lo,
                "intervals overlap: {} and {}",
                w[0],
                w[1]
            );
        }
    }

    // ---- real_roots_symbolic with rational coefficients ----

    /// Check that `iv` brackets a true root of `poly`, i.e. `poly` changes
    /// sign (or is exactly zero) at the endpoints of `iv`.
    fn assert_brackets_root(poly: &UniPoly, iv: &RootInterval) {
        if iv.lo == iv.hi {
            let v = poly.eval_rational(&iv.lo);
            assert_eq!(v, rug::Rational::from(0), "exact root {iv} is not a root");
            return;
        }
        let f_lo = poly.eval_rational(&iv.lo);
        let f_hi = poly.eval_rational(&iv.hi);
        assert!(
            f_lo == 0 || f_hi == 0 || (f_lo < 0) != (f_hi < 0),
            "interval {iv} does not bracket a root: f(lo)={f_lo}, f(hi)={f_hi}"
        );
    }

    #[test]
    fn real_roots_symbolic_rational_coefficient_x_squared_minus_half() {
        // x^2 - 1/2 has roots at ±1/√2 ≈ ±0.7071.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let xsq = p.pow(x, p.integer(2_i32));
        let half = p.rational(1, 2);
        let expr = p.add(vec![xsq, p.mul(vec![p.integer(-1_i32), half])]);

        let roots = real_roots_symbolic(expr, x, &p).unwrap();
        assert_eq!(roots.len(), 2, "x^2 - 1/2 has 2 real roots");

        let approx = 1.0_f64 / 2.0_f64.sqrt(); // ≈ 0.7071
                                               // Negative root brackets -0.7071.
        assert!(roots[0].lo_f64() <= -approx && roots[0].hi_f64() >= -approx);
        // Positive root brackets +0.7071.
        assert!(roots[1].lo_f64() <= approx && roots[1].hi_f64() >= approx);

        // Verify isolating intervals actually bracket the true roots via the
        // *cleared* integer polynomial 2x^2 - 1 (same roots as x^2 - 1/2).
        let cleared = UniPoly::from_symbolic_clear_denoms(expr, x, &p).unwrap();
        assert_eq!(cleared.coefficients_i64(), vec![-1, 0, 2]);
        for iv in &roots {
            assert_brackets_root(&cleared, iv);
        }
    }

    #[test]
    fn real_roots_symbolic_2x_squared_minus_1_matches_rational_form() {
        // 2x^2 - 1 (already integer) has the same roots as x^2 - 1/2.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let xsq = p.pow(x, p.integer(2_i32));
        let two_xsq = p.mul(vec![p.integer(2_i32), xsq]);
        let expr = p.add(vec![two_xsq, p.integer(-1_i32)]);

        let roots = real_roots_symbolic(expr, x, &p).unwrap();
        assert_eq!(roots.len(), 2, "2x^2 - 1 has 2 real roots");

        let approx = 1.0_f64 / 2.0_f64.sqrt();
        assert!(roots[0].lo_f64() <= -approx && roots[0].hi_f64() >= -approx);
        assert!(roots[1].lo_f64() <= approx && roots[1].hi_f64() >= approx);

        let poly = UniPoly::from_symbolic(expr, x, &p).unwrap();
        for iv in &roots {
            assert_brackets_root(&poly, iv);
        }
    }

    #[test]
    fn real_roots_symbolic_integer_case_still_works() {
        // x^2 - 2 has roots at ±√2.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let xsq = p.pow(x, p.integer(2_i32));
        let expr = p.add(vec![xsq, p.integer(-2_i32)]);

        let roots = real_roots_symbolic(expr, x, &p).unwrap();
        assert_eq!(roots.len(), 2, "x^2 - 2 has 2 real roots");

        let approx = std::f64::consts::SQRT_2;
        assert!(roots[0].lo_f64() <= -approx && roots[0].hi_f64() >= -approx);
        assert!(roots[1].lo_f64() <= approx && roots[1].hi_f64() >= approx);

        let poly = UniPoly::from_symbolic(expr, x, &p).unwrap();
        for iv in &roots {
            assert_brackets_root(&poly, iv);
        }
    }

    #[test]
    fn refine_root_x_minus_3() {
        let poly = make_poly(&[-3, 1]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 1);
        let ball = refine_root(&poly, &roots[0], 53);
        assert!(ball.contains(3.0), "refined ball must contain x=3");
    }
}
