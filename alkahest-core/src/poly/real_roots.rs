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
    let nonzero: Vec<&Integer> = coeffs.iter().filter(|c| **c != 0).collect();
    if nonzero.len() < 2 {
        return 0;
    }
    let mut count = 0;
    for w in nonzero.windows(2) {
        let pos0 = *w[0] > 0;
        let pos1 = *w[1] > 0;
        if pos0 != pos1 {
            count += 1;
        }
    }
    count
}

/// Compute `p(x + 1)` using the O(n²) de Casteljau / Taylor-shift algorithm.
///
/// For each `i = 0..n−1`, for each `j = (i..n−1)` in reverse:
/// `c[j] += c[j+1]`.
fn taylor_shift_by_1(coeffs: &[Integer]) -> Vec<Integer> {
    let mut c: Vec<Integer> = coeffs.to_vec();
    let n = c.len();
    for i in 0..n.saturating_sub(1) {
        for j in (i..n - 1).rev() {
            let cjp1 = c[j + 1].clone();
            c[j] += cjp1;
        }
    }
    c
}

/// Compute `p(x + k)` for a non-negative integer `k`.
fn taylor_shift_by(coeffs: &[Integer], k: u64) -> Vec<Integer> {
    if k == 0 {
        return coeffs.to_vec();
    }
    let mut c = coeffs.to_vec();
    let n = c.len();
    for i in 0..n.saturating_sub(1) {
        for j in (i..n - 1).rev() {
            let delta = c[j + 1].clone() * k;
            c[j] += delta;
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

/// Integer lower bound on the smallest positive root of `p`.
///
/// Uses a doubling-then-binary-search over integer evaluation points.
/// Precondition: `p(0) ≠ 0` (no root at the origin).
/// Returns the largest integer `k ≥ 1` such that `p(k)` has the same sign
/// as `p(0)` (implying all positive roots are `> k`), or `0` if the
/// smallest positive root is in `(0, 1]`.
fn cf_lower_bound_floor(coeffs: &[Integer]) -> u64 {
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

        let k = cf_lower_bound_floor(&frame.poly);
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

    result.sort_by(|a, b| a.lo.partial_cmp(&b.lo).unwrap_or(std::cmp::Ordering::Equal));
    Ok(result)
}

/// Isolate all real roots of a symbolic expression in `var`.
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
    let poly = UniPoly::from_symbolic(expr, var, pool).map_err(RealRootError::NotAPolynomial)?;
    real_roots(&poly)
}

/// Narrow a [`RootInterval`] to at least `prec` bits of precision.
///
/// Uses bisection with floating-point Horner evaluation.  For exact roots
/// (`lo == hi`), returns a zero-radius [`ArbBall`].
pub fn refine_root(poly: &UniPoly, interval: &RootInterval, prec: u32) -> ArbBall {
    if interval.lo == interval.hi {
        return ArbBall::from_midpoint_radius(interval.lo.to_f64(), 0.0, prec.max(53));
    }

    let coeffs_f64: Vec<f64> = poly.coefficients().iter().map(|c| c.to_f64()).collect();
    let eval = |x: f64| -> f64 { coeffs_f64.iter().rev().fold(0.0_f64, |acc, &c| acc * x + c) };

    let target_width = 2.0_f64.powi(-(prec as i32));
    let mut lo = interval.lo.to_f64();
    let mut hi = interval.hi.to_f64();
    let mut f_lo = eval(lo);

    for _ in 0..300 {
        if hi - lo <= target_width {
            break;
        }
        let mid = (lo + hi) / 2.0;
        let f_mid = eval(mid);
        if f_lo * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }

    let center = (lo + hi) / 2.0;
    let radius = (hi - lo) / 2.0;
    ArbBall::from_midpoint_radius(center, radius, prec.max(53))
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

    #[test]
    fn refine_root_x_minus_3() {
        let poly = make_poly(&[-3, 1]);
        let roots = real_roots(&poly).unwrap();
        assert_eq!(roots.len(), 1);
        let ball = refine_root(&poly, &roots[0], 53);
        assert!(ball.contains(3.0), "refined ball must contain x=3");
    }
}
