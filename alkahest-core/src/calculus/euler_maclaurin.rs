//! Euler–Maclaurin asymptotics of sums (P1 item 10).
//!
//! For a smooth summand `f`, the Euler–Maclaurin formula relates a sum to an
//! integral plus boundary corrections:
//!
//! ```text
//! Σ_{k=a}^{n} f(k) = ∫_a^n f(t) dt + (f(a) + f(n))/2
//!                    + Σ_{j=1}^{m} B_{2j}/(2j)! · (f^{(2j-1)}(n) − f^{(2j-1)}(a))
//!                    + R_m
//! ```
//!
//! Reading it as `n → ∞` and dropping the `a`-endpoint pieces into a single
//! additive constant gives an asymptotic expansion in `n`. That is what
//! [`euler_maclaurin`] returns:
//!
//! ```text
//! Σ_{k=a}^{n} f(k) ~ ∫^n f + C + f(n)/2 + Σ_j B_{2j}/(2j)! · f^{(2j-1)}(n)
//! ```
//!
//! For `f(k) = 1/k` this is the classical `H_n ~ log n + γ + 1/(2n) − 1/(12n²) + …`.
//!
//! # The constant is not free
//!
//! Euler–Maclaurin does **not** determine `C` from the `n`-side terms alone —
//! for the harmonic numbers `C` is Euler's γ, which no amount of boundary
//! algebra at `a` produces. This module therefore determines `C` *numerically*
//! from the exact sum at several large `n`, and says so: the returned
//! [`AsymptoticReport`] carries [`Rigor::NumericallyConsistent`] and an
//! explicitly **assumed** hypothesis naming the constant as fitted rather than
//! proved. The shape of the expansion is derived symbolically; only that one
//! scalar is empirical, and the report never pretends otherwise.
//!
//! # Verification gate
//!
//! Every term is put through the same `o()`-gate as
//! [`mod@crate::calculus::asymptotic`]: the truncated expansion is compared
//! against the exactly-computed sum at increasing `n`, and terms that do not
//! genuinely refine their predecessor are dropped. If nothing survives, the
//! call refuses rather than emitting an unverified expansion.

use super::asymptotic::AsymptoticError;
use super::asymptotic_common::{
    bernoulli_numbers, eval_over, gate_accept, rational_to_expr, verification_points,
    AsymptoticReport, Hypothesis, Rigor, DEFAULT_SLACK,
};
use crate::diff::diff;
use crate::integrate::integrate;
use crate::jit::eval_interp;
use crate::kernel::{subs, ExprId, ExprPool};
use crate::simplify::simplify;
use rug::{Integer, Rational};
use std::collections::HashMap;

/// Largest number of Bernoulli correction terms accepted.
pub const MAX_CORRECTIONS: usize = 8;

/// Check points used by the numeric gate and the constant fit.
const CHECK_POINTS: [f64; 4] = [64.0, 128.0, 256.0, 512.0];

/// Asymptotic expansion of `Σ_{k=a}^{n} f(k)` as `n → ∞`.
///
/// `corrections` is the number of Bernoulli terms to attempt (`m` above);
/// the returned expansion may be shorter if the numeric gate rejects the tail.
///
/// Refuses ([`AsymptoticError`]) rather than guessing when the summand cannot
/// be integrated symbolically, when it is not numerically evaluable at the
/// check points, or when no term survives the gate.
pub fn euler_maclaurin(
    f: ExprId,
    k: ExprId,
    a: i64,
    n: ExprId,
    corrections: usize,
    pool: &ExprPool,
) -> Result<AsymptoticReport, AsymptoticError> {
    if corrections > MAX_CORRECTIONS {
        return Err(AsymptoticError::InvalidTermCount);
    }
    if k == n {
        return Err(AsymptoticError::InvalidTermCount);
    }

    let mut derivation = Vec::new();
    let mut terms: Vec<ExprId> = Vec::new();

    // ∫ f dk, evaluated at n. The lower endpoint is a constant and folds into C.
    let antiderivative = integrate(f, k, pool)
        .map_err(|_| AsymptoticError::UnsupportedScale)?
        .value;
    let integral_at_n = simplify(subs_one(antiderivative, k, n, pool), pool).value;
    derivation.push(format!(
        "∫f dk = {}, evaluated at n",
        pool.display(antiderivative)
    ));
    terms.push(integral_at_n);

    // f(n)/2
    let half = pool.rational(Integer::from(1), Integer::from(2));
    let f_at_n = subs_one(f, k, n, pool);
    terms.push(simplify(pool.mul(vec![half, f_at_n]), pool).value);
    derivation.push("boundary term f(n)/2".to_string());

    // Σ_j B_{2j}/(2j)! · f^{(2j-1)}(n)
    let bern = bernoulli_numbers(2 * corrections + 1);
    let mut deriv = f;
    let mut deriv_order = 0usize;
    let mut factorial = Integer::from(1);
    for j in 1..=corrections {
        // advance `deriv` to f^{(2j-1)}
        let target = 2 * j - 1;
        while deriv_order < target {
            deriv = diff(deriv, k, pool).map_err(AsymptoticError::Diff)?.value;
            deriv = simplify(deriv, pool).value;
            deriv_order += 1;
        }
        // (2j)!
        for t in (2 * j - 1)..=(2 * j) {
            factorial *= Integer::from(t as u32);
        }
        let b = bern[2 * j].clone();
        if b == 0 {
            continue;
        }
        let coeff = Rational::from((b.numer().clone(), b.denom().clone() * factorial.clone()));
        let coeff_expr = rational_to_expr(&coeff, pool);
        let d_at_n = subs_one(deriv, k, n, pool);
        terms.push(simplify(pool.mul(vec![coeff_expr, d_at_n]), pool).value);
        derivation.push(format!("Bernoulli correction j = {j} (B_{} term)", 2 * j));
    }

    // Oracle: the exact sum at each check point.
    let points: Vec<f64> = CHECK_POINTS.to_vec();
    let oracle = exact_sums(f, k, a, &points, pool).ok_or(AsymptoticError::GateFailed)?;

    // The additive constant: fit it from the largest check point, where the
    // dropped tail is smallest.
    let mut term_vals: Vec<Vec<f64>> = Vec::with_capacity(terms.len());
    for &t in &terms {
        term_vals.push(eval_over(t, n, &points, pool).ok_or(AsymptoticError::GateFailed)?);
    }
    let last = points.len() - 1;
    let symbolic_at_last: f64 = term_vals.iter().map(|row| row[last]).sum();
    let constant = oracle[last] - symbolic_at_last;
    derivation.push(format!(
        "additive constant fitted numerically at n = {}: {constant}",
        points[last]
    ));

    // Add the constant, then order the whole sequence by magnitude at the
    // largest check point. Position matters: an asymptotic sequence has to be
    // decreasing, and the constant sits *below* every growing term but *above*
    // every decaying one. Inserting it at a fixed index instead would break
    // the ordering for a growing summand — `Σ k` would put the constant ahead
    // of `n/2` and the gate would (correctly) reject the tail.
    let constant_expr = float_to_expr(constant, pool);
    terms.push(constant_expr);
    term_vals.push(vec![constant; points.len()]);

    let mut order: Vec<usize> = (0..terms.len()).collect();
    order.sort_by(|&i, &j| {
        term_vals[j][last]
            .abs()
            .partial_cmp(&term_vals[i][last].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    terms = order.iter().map(|&i| terms[i]).collect();
    term_vals = order.iter().map(|&i| term_vals[i].clone()).collect();

    let accepted = gate_accept(&oracle, &term_vals, DEFAULT_SLACK);
    if accepted == 0 {
        return Err(AsymptoticError::GateFailed);
    }
    terms.truncate(accepted);
    term_vals.truncate(accepted);
    let verification = verification_points(&points, &oracle, &term_vals, accepted);

    Ok(AsymptoticReport {
        method: "euler-maclaurin",
        var: n,
        terms,
        rigor: Rigor::NumericallyConsistent,
        hypotheses: vec![
            Hypothesis::checked(
                "the summand has a symbolic antiderivative and is finite at every check point",
            ),
            Hypothesis::assumed(
                "the summand is smooth on [a, ∞) and its high derivatives decay, so the \
                 Euler–Maclaurin remainder is asymptotically negligible",
            ),
            Hypothesis::assumed(
                "the additive constant was fitted numerically from the exact sum, not derived",
            ),
        ],
        verification,
        derivation,
    })
}

/// `expr` with the single substitution `from -> to`.
fn subs_one(expr: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(from, to);
    subs(expr, &m, pool)
}

/// Numerically sum `f(k)` for `k = a ..= n` at each check point.
fn exact_sums(f: ExprId, k: ExprId, a: i64, points: &[f64], pool: &ExprPool) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(points.len());
    for &p in points {
        let upper = p as i64;
        let mut acc = 0.0f64;
        for i in a..=upper {
            let mut env = HashMap::new();
            env.insert(k, i as f64);
            let v = eval_interp(f, &env, pool)?;
            if !v.is_finite() {
                return None;
            }
            acc += v;
        }
        if !acc.is_finite() {
            return None;
        }
        out.push(acc);
    }
    Some(out)
}

/// A float as an exact-looking rational expression (the constant is fitted, so
/// it is only meaningful to ~15 digits; this keeps it inside the expression
/// algebra rather than introducing a float node).
fn float_to_expr(v: f64, pool: &ExprPool) -> ExprId {
    match Rational::from_f64(v) {
        Some(q) => {
            // Round to a manageable denominator: the constant is empirical, and
            // an exact binary fraction with a 2^52 denominator is noise.
            let scale = Integer::from(10_000_000_000_000_i64);
            let scaled = (q * Rational::from(scale.clone())).round();
            let num = scaled.numer().clone();
            rational_to_expr(&Rational::from((num, scale)), pool)
        }
        None => pool.integer(0_i32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Real);
        let n = pool.symbol("n", Domain::Real);
        (pool, k, n)
    }

    /// `H_n = Σ_{k=1}^{n} 1/k ~ log n + γ + 1/(2n) − 1/(12n²) + …`
    ///
    /// The leading term must be `log n` and the fitted constant must come out
    /// as Euler's γ ≈ 0.5772, which is the whole point of fitting it: no amount
    /// of boundary algebra at `k = 1` produces γ.
    #[test]
    fn harmonic_numbers_recover_log_plus_gamma() {
        let (pool, k, n) = setup();
        let f = pool.pow(k, pool.integer(-1_i32));
        let r = euler_maclaurin(f, k, 1, n, 2, &pool).expect("expansion");

        assert_eq!(r.method, "euler-maclaurin");
        assert!(
            r.terms.len() >= 2,
            "expected at least log n and the constant"
        );

        // Leading term is log n.
        let leading = pool.display(r.leading().unwrap()).to_string();
        assert!(
            leading.contains("log") || leading.contains("ln"),
            "leading term should be logarithmic, got {leading}"
        );

        // Evaluate the truncated expansion against the true H_n at a large n.
        let partial = r.partial_sum(&pool);
        let mut env = std::collections::HashMap::new();
        env.insert(n, 1000.0);
        let approx = crate::jit::eval_interp(partial, &env, &pool).expect("evaluates");
        let truth: f64 = (1..=1000).map(|i| 1.0 / i as f64).sum();
        assert!(
            (approx - truth).abs() < 1e-6,
            "H_1000: expansion {approx} vs truth {truth}"
        );
    }

    /// The report must say which hypotheses were merely assumed — the fitted
    /// constant is not a proved quantity and must not be presented as one.
    #[test]
    fn constant_is_labelled_as_fitted_not_proved() {
        let (pool, k, n) = setup();
        let f = pool.pow(k, pool.integer(-1_i32));
        let r = euler_maclaurin(f, k, 1, n, 2, &pool).expect("expansion");

        assert_eq!(r.rigor, Rigor::NumericallyConsistent);
        assert!(!r.all_hypotheses_checked());
        assert!(
            r.hypotheses
                .iter()
                .any(|h| h.statement.contains("fitted numerically")),
            "the fitted constant must be declared"
        );
        assert!(!r.verification.is_empty());
        assert!(r.max_relative_error().unwrap() < 1e-6);
    }

    /// `Σ_{k=1}^{n} k = n(n+1)/2` — a polynomial summand, exactly reproduced.
    #[test]
    fn polynomial_summand_is_exact() {
        let (pool, k, n) = setup();
        let r = euler_maclaurin(k, k, 1, n, 1, &pool).expect("expansion");

        let partial = r.partial_sum(&pool);
        for ni in [10.0_f64, 50.0, 200.0] {
            let mut env = std::collections::HashMap::new();
            env.insert(n, ni);
            let approx = crate::jit::eval_interp(partial, &env, &pool).expect("evaluates");
            let truth = ni * (ni + 1.0) / 2.0;
            assert!(
                (approx - truth).abs() / truth < 1e-9,
                "n = {ni}: {approx} vs {truth}"
            );
        }
    }

    /// A summand with no symbolic antiderivative is refused, not guessed at.
    #[test]
    fn refuses_when_the_summand_cannot_be_integrated() {
        let (pool, k, n) = setup();
        // exp(-k^2) has no elementary antiderivative.
        let neg_k2 = pool.mul(vec![pool.integer(-1_i32), k, k]);
        let f = pool.func("exp", vec![neg_k2]);
        let err = euler_maclaurin(f, k, 1, n, 1, &pool).expect_err("must refuse");
        assert!(matches!(err, AsymptoticError::UnsupportedScale));
    }

    #[test]
    fn refuses_absurd_correction_count() {
        let (pool, k, n) = setup();
        let f = pool.pow(k, pool.integer(-1_i32));
        let err = euler_maclaurin(f, k, 1, n, MAX_CORRECTIONS + 1, &pool).expect_err("must refuse");
        assert!(matches!(err, AsymptoticError::InvalidTermCount));
    }

    #[test]
    fn refuses_coincident_index_and_variable() {
        let (pool, k, _n) = setup();
        let f = pool.pow(k, pool.integer(-1_i32));
        assert!(euler_maclaurin(f, k, 1, k, 1, &pool).is_err());
    }
}
