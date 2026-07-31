//! Closed forms for recognized infinite "special" sums.
//!
//! Gosper's algorithm (see [`super::gosper`]) cannot sum `1/k^p` in closed
//! form for any `p ≥ 2`: the indefinite antidifference of a power of `k` with
//! negative integer exponent is a polygamma function, not a rational multiple
//! of a hypergeometric term, so [`super::gosper_certificate`] returns `None`
//! and an infinite-bound sum would otherwise fall straight through to
//! [`super::SumError::NotGosperSummable`] (`E-SUM-002`) even for textbook
//! cases like the Basel problem.
//!
//! This module recognizes the Basel-family p-series with an *even* exponent —
//! `Σ_{n=1}^{∞} c/n^{2m} = c·ζ(2m)` — via the classic Bernoulli-number formula
//!
//! ```text
//! ζ(2m) = (-1)^(m+1) · B_{2m} · (2π)^{2m} / (2·(2m)!)
//! ```
//!
//! and returns the closed form as a rational multiple of `π^{2m}` (the
//! interned `pi` symbol, matching the rest of the CAS — see
//! [`crate::transform::fourier`]'s convention).
//!
//! Odd zeta values (`ζ(3)`, `ζ(5)`, …) have no known closed form in
//! elementary constants and are deliberately **not** attempted: matching only
//! recognizes even exponents, so odd-power sums correctly fall through to the
//! honest `NotGosperSummable` error instead of a fabricated value.

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use rug::{Integer, Rational};

/// Upper bound on the recognized exponent `p` in `1/k^p`, purely to keep the
/// Bernoulli-number/factorial computation cheap; textbook sums never come
/// close to this, and beyond it we honestly decline rather than churn on a
/// pathological exponent.
const MAX_RECOGNIZED_POWER: i64 = 60;

/// Pascal's-triangle rows `0..=n`; `rows[i][j] == C(i, j)`.
fn pascal_rows(n: usize) -> Vec<Vec<Integer>> {
    let mut rows: Vec<Vec<Integer>> = Vec::with_capacity(n + 1);
    rows.push(vec![Integer::from(1)]);
    for i in 1..=n {
        let prev = &rows[i - 1];
        let mut row = Vec::with_capacity(i + 1);
        row.push(Integer::from(1));
        for j in 1..i {
            row.push(prev[j - 1].clone() + prev[j].clone());
        }
        row.push(Integer::from(1));
        rows.push(row);
    }
    rows
}

/// Bernoulli numbers `B_0..=B_up_to` via the standard recurrence
/// `B_m = -1/(m+1) · Σ_{j=0}^{m-1} C(m+1,j)·B_j`, with `B_0 = 1`.
fn bernoulli_numbers(up_to: usize) -> Vec<Rational> {
    let rows = pascal_rows(up_to + 1);
    let mut b: Vec<Rational> = Vec::with_capacity(up_to + 1);
    b.push(Rational::from(1));
    for m in 1..=up_to {
        let row = &rows[m + 1];
        let mut sum = Rational::from(0);
        for (j, bj) in b.iter().enumerate().take(m) {
            sum += Rational::from(row[j].clone()) * bj.clone();
        }
        b.push(-sum / Rational::from(m as i64 + 1));
    }
    b
}

fn factorial(n: u64) -> Integer {
    let mut r = Integer::from(1);
    for i in 2..=n {
        r *= i;
    }
    r
}

/// Rational coefficient `c` with `ζ(2m) = c·π^{2m}`, for `m ≥ 1`.
fn zeta_even_coefficient(m: u32) -> Rational {
    let b = bernoulli_numbers(2 * m as usize);
    let b_2m = b[2 * m as usize].clone();
    let sign: i64 = if m % 2 == 1 { 1 } else { -1 }; // (-1)^(m+1)
    let two_pow_2m = Integer::from(1) << (2 * m); // (2π)^{2m} = 2^{2m}·π^{2m}
    let denom = Integer::from(2) * factorial(2 * m as u64);
    Rational::from(sign) * b_2m * Rational::from((two_pow_2m, denom))
}

/// True when `expr` is (or contains) `k`.
fn depends_on(expr: ExprId, k: ExprId, pool: &ExprPool) -> bool {
    if expr == k {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&a| depends_on(a, k, pool)),
        ExprData::Pow { base, exp } => depends_on(base, k, pool) || depends_on(exp, k, pool),
        ExprData::Func { args, .. } => args.iter().any(|&a| depends_on(a, k, pool)),
        _ => false,
    }
}

/// `p` when `exp` is the literal negative integer `-p` (`p > 0`).
fn negative_integer_exponent(exp: ExprId, pool: &ExprPool) -> Option<i64> {
    if let ExprData::Integer(n) = pool.get(exp) {
        let v = n.0.to_i64()?;
        if v < 0 {
            return Some(-v);
        }
    }
    None
}

/// Matches `term` against `c·k^{-p}` for `c` free of `k` and `p > 0`,
/// returning `(c, p)`. Parity of `p` is checked by the caller.
fn match_p_series_term(term: ExprId, k: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
    match pool.get(term) {
        ExprData::Pow { base, exp } if base == k => {
            let p = negative_integer_exponent(exp, pool)?;
            Some((pool.integer(1_i32), p))
        }
        ExprData::Mul(args) => {
            let mut coeff_factors = Vec::new();
            let mut p_found: Option<i64> = None;
            for a in args {
                if p_found.is_none() {
                    if let ExprData::Pow { base, exp } = pool.get(a) {
                        if base == k {
                            if let Some(p) = negative_integer_exponent(exp, pool) {
                                p_found = Some(p);
                                continue;
                            }
                        }
                    }
                }
                if depends_on(a, k, pool) {
                    return None;
                }
                coeff_factors.push(a);
            }
            let p = p_found?;
            let coeff = match coeff_factors.len() {
                0 => pool.integer(1_i32),
                1 => coeff_factors[0],
                _ => pool.mul(coeff_factors),
            };
            Some((coeff, p))
        }
        _ => None,
    }
}

/// Recognizes `Σ_{k=lo}^{hi} term` as a Basel-family even p-series and
/// returns its closed form in terms of `π`, or `None` when the pattern
/// doesn't apply — wrong bounds, an odd or non-positive power, or an
/// exponent past [`MAX_RECOGNIZED_POWER`]. Callers must fall back to the
/// ordinary `NotGosperSummable` error on `None`; this function never
/// fabricates a value for a case it doesn't recognize.
///
/// Only `lo == 1` is recognized (the standard statement of the Basel
/// problem and its even-zeta relatives); a symbolic or non-unit lower bound
/// falls through to the honest error rather than attempting a finite
/// correction sum (which `1/k^p` has no closed form for either).
pub(super) fn basel_family_closed_form(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if hi != pool.pos_infinity() {
        return None;
    }
    if !matches!(pool.get(lo), ExprData::Integer(n) if n.0 == 1) {
        return None;
    }
    let (term_coeff, p) = match_p_series_term(term, k, pool)?;
    if p <= 0 || p % 2 != 0 || p > MAX_RECOGNIZED_POWER {
        return None;
    }
    let m = (p / 2) as u32;
    let zeta_coeff = zeta_even_coefficient(m);
    // `Domain::Real`, matching the default `pool.symbol("pi")` from Python
    // (see `parse_domain_arg`) and the convention used by
    // `crate::transform::fourier`'s `pi` helper — so a caller's own
    // `pool.symbol("pi")` structurally equals (and can be bound as) the `pi`
    // in this result.
    let pi = pool.symbol("pi", Domain::Real);
    let pi_pow = pool.pow(pi, pool.integer(p));
    let (num, den) = (zeta_coeff.numer().clone(), zeta_coeff.denom().clone());
    let zeta_expr = if den == 1 {
        pool.mul(vec![pool.integer(num), pi_pow])
    } else {
        pool.mul(vec![pool.rational(num, den), pi_pow])
    };
    Some(pool.mul(vec![term_coeff, zeta_expr]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::simplify::engine::simplify;
    use std::collections::HashMap;

    fn simp(pool: &ExprPool, e: ExprId) -> ExprId {
        simplify(e, pool).value
    }

    #[test]
    fn zeta_two_is_pi_squared_over_six() {
        let c = zeta_even_coefficient(1);
        assert_eq!(c, Rational::from((1, 6)));
    }

    #[test]
    fn zeta_four_is_pi_fourth_over_ninety() {
        let c = zeta_even_coefficient(2);
        assert_eq!(c, Rational::from((1, 90)));
    }

    #[test]
    fn zeta_six_is_pi_sixth_over_945() {
        let c = zeta_even_coefficient(3);
        assert_eq!(c, Rational::from((1, 945)));
    }

    #[test]
    fn basel_sum_matches_pi_squared_over_six_numerically() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Positive);
        let term = simp(&pool, pool.pow(k, pool.integer(-2_i32)));
        let lo = pool.integer(1_i32);
        let hi = pool.pos_infinity();
        let value = basel_family_closed_form(term, k, lo, hi, &pool).expect("Basel sum");

        let pi = pool.symbol("pi", Domain::Real);
        let mut env = HashMap::new();
        env.insert(pi, std::f64::consts::PI);
        let got = eval_interp(value, &env, &pool).expect("eval");
        let want = std::f64::consts::PI.powi(2) / 6.0;
        assert!((got - want).abs() < 1e-9, "got {got} want {want}");
    }

    #[test]
    fn sum_one_over_n_fourth_matches_pi_fourth_over_ninety() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Positive);
        let term = simp(&pool, pool.pow(k, pool.integer(-4_i32)));
        let lo = pool.integer(1_i32);
        let hi = pool.pos_infinity();
        let value = basel_family_closed_form(term, k, lo, hi, &pool).expect("zeta(4)");

        let pi = pool.symbol("pi", Domain::Real);
        let mut env = HashMap::new();
        env.insert(pi, std::f64::consts::PI);
        let got = eval_interp(value, &env, &pool).expect("eval");
        let want = std::f64::consts::PI.powi(4) / 90.0;
        assert!((got - want).abs() < 1e-9, "got {got} want {want}");
    }

    #[test]
    fn scaled_basel_sum_carries_the_coefficient() {
        // Σ 3/k^2 = 3·ζ(2) = π²/2.
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Positive);
        let term = simp(
            &pool,
            pool.mul(vec![pool.integer(3_i32), pool.pow(k, pool.integer(-2_i32))]),
        );
        let lo = pool.integer(1_i32);
        let hi = pool.pos_infinity();
        let value = basel_family_closed_form(term, k, lo, hi, &pool).expect("3·zeta(2)");

        let pi = pool.symbol("pi", Domain::Real);
        let mut env = HashMap::new();
        env.insert(pi, std::f64::consts::PI);
        let got = eval_interp(value, &env, &pool).expect("eval");
        let want = 3.0 * std::f64::consts::PI.powi(2) / 6.0;
        assert!((got - want).abs() < 1e-9, "got {got} want {want}");
    }

    #[test]
    fn odd_power_is_not_recognized() {
        // ζ(3) (Apéry's constant) has no known closed form in π — must decline.
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Positive);
        let term = simp(&pool, pool.pow(k, pool.integer(-3_i32)));
        let lo = pool.integer(1_i32);
        let hi = pool.pos_infinity();
        assert!(basel_family_closed_form(term, k, lo, hi, &pool).is_none());
    }

    #[test]
    fn finite_upper_bound_is_not_recognized() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Positive);
        let term = simp(&pool, pool.pow(k, pool.integer(-2_i32)));
        let lo = pool.integer(1_i32);
        let hi = pool.integer(100_i32);
        assert!(basel_family_closed_form(term, k, lo, hi, &pool).is_none());
    }

    #[test]
    fn non_unit_lower_bound_is_not_recognized() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Positive);
        let term = simp(&pool, pool.pow(k, pool.integer(-2_i32)));
        let lo = pool.integer(2_i32);
        let hi = pool.pos_infinity();
        assert!(basel_family_closed_form(term, k, lo, hi, &pool).is_none());
    }

    #[test]
    fn divergent_positive_power_is_not_recognized() {
        // Σ k^2 to ∞ diverges — must not be mistaken for a p-series.
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Positive);
        let term = simp(&pool, pool.pow(k, pool.integer(2_i32)));
        let lo = pool.integer(1_i32);
        let hi = pool.pos_infinity();
        assert!(basel_family_closed_form(term, k, lo, hi, &pool).is_none());
    }
}
