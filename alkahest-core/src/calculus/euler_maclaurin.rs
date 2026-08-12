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
//!
//! The constant is fitted at a point *outside* the gate's check points, and is
//! emitted only if refitting it at a second point reproduces it. Both halves
//! matter, and for the same reason: a
//! constant fitted at a point the gate then scores makes the residual there
//! zero by construction, so the gate cannot reject it — whatever the expansion
//! was actually missing gets emitted as a "constant". `Σ_{k=1}^{n} k⁹` at the
//! default `corrections = 2` used to acquire a term `34359738368`, which is
//! `512⁴/2`: the missing `n⁴/2` of Faulhaber's formula, frozen at the fitting
//! point, presented as a constant of a polynomial identity whose constant term
//! is zero.

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

/// Check points used by the numeric gate.
const CHECK_POINTS: [f64; 4] = [64.0, 128.0, 256.0, 512.0];

/// Where the additive constant is fitted — deliberately **outside**
/// [`CHECK_POINTS`].
///
/// Fitting at a point the gate then scores makes the gate vacuous: the residual
/// there is identically zero by construction, so `gate_accept`'s decay test is
/// satisfied no matter what the fitted number is, and any leftover power of `n`
/// is emitted as a "constant". `Σ k⁹` came back with a spurious `512⁴/2` that
/// way — the dropped `n⁴/2` term frozen at the fitting point.
const CONSTANT_FIT_POINT: f64 = 1024.0;

/// How much the constant fitted at [`CONSTANT_FIT_POINT`] may differ from the
/// one fitted at the previous point, relative to their size.
///
/// A genuine additive constant is the *same* number wherever it is fitted, up to
/// the truncation error and `f64` noise: across the whole clean battery
/// (`γ`, `ζ(2)`, `ζ(3)`, `ζ(½)`, `½log 2π`, `γ₁`, the Faulhaber `−1/12` and
/// `1/120`, …) the observed drift never exceeded `3.2e-3` of the constant.
/// A dropped power of `n` masquerading as a constant grows with the fitting
/// point, and every one of those observed drifted by `≥ 0.93`. The gap is three
/// orders of magnitude wide; `1e-2` sits in it.
const CONSTANT_DRIFT_TOL: f64 = 1e-2;

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

    // Oracle: the exact sum at each gate point, plus one further point reserved
    // for fitting the additive constant.
    let points: Vec<f64> = CHECK_POINTS.to_vec();
    let mut fit_points = points.clone();
    fit_points.push(CONSTANT_FIT_POINT);
    let oracle_all = exact_sums(f, k, a, &fit_points, pool).ok_or(AsymptoticError::GateFailed)?;

    let mut term_vals_all: Vec<Vec<f64>> = Vec::with_capacity(terms.len());
    for &t in &terms {
        term_vals_all.push(eval_over(t, n, &fit_points, pool).ok_or(AsymptoticError::GateFailed)?);
    }

    // `C(m) = Σ_{k≤m} f(k) − Σ_j term_j(m)` — the constant the expansion would
    // need at each point. Its *convergence* is the evidence that it is a
    // constant at all: an additive constant is the same number at every fitting
    // point, whereas a term of the expansion that was dropped (because
    // `corrections` was too small for the summand) grows with the point and only
    // looks constant because it was frozen at one.
    let m = fit_points.len() - 1;
    let fit_at =
        |j: usize| -> f64 { oracle_all[j] - term_vals_all.iter().map(|row| row[j]).sum::<f64>() };
    let constant = fit_at(m);
    let previous = fit_at(m - 1);
    let drift = (constant - previous).abs();
    let scale = constant.abs().max(previous.abs());
    let constant_converged = drift <= CONSTANT_DRIFT_TOL * scale;

    let oracle: Vec<f64> = oracle_all[..points.len()].to_vec();
    let mut term_vals: Vec<Vec<f64>> = term_vals_all
        .into_iter()
        .map(|row| row[..points.len()].to_vec())
        .collect();
    let last = points.len() - 1;

    // Add the constant, then order the whole sequence by magnitude at the
    // largest check point. Position matters: an asymptotic sequence has to be
    // decreasing, and the constant sits *below* every growing term but *above*
    // every decaying one. Inserting it at a fixed index instead would break
    // the ordering for a growing summand — `Σ k` would put the constant ahead
    // of `n/2` and the gate would (correctly) reject the tail.
    let mut constant_slot: Option<usize> = None;
    if constant_converged {
        derivation.push(format!(
            "additive constant fitted numerically at n = {}: {constant} \
             (it moved by {drift:.3e} from the fit at n = {}, so it is a constant)",
            fit_points[m],
            fit_points[m - 1]
        ));
        let constant_expr = float_to_expr(constant, pool);
        constant_slot = Some(terms.len());
        terms.push(constant_expr);
        term_vals.push(vec![constant; points.len()]);
    } else {
        derivation.push(format!(
            "no additive constant is claimed: the fit moved from {previous} at n = {} \
             to {constant} at n = {}, so it is not a constant — most likely a term of \
             the expansion that `corrections` was too small to produce",
            fit_points[m - 1],
            fit_points[m],
        ));
    }

    let mut order: Vec<usize> = (0..terms.len()).collect();
    order.sort_by(|&i, &j| {
        term_vals[j][last]
            .abs()
            .partial_cmp(&term_vals[i][last].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    terms = order.iter().map(|&i| terms[i]).collect();
    term_vals = order.iter().map(|&i| term_vals[i].clone()).collect();
    // Where the constant ended up after the reordering, if it survives the gate.
    let constant_position = constant_slot.and_then(|slot| order.iter().position(|&i| i == slot));

    let accepted = gate_accept(&oracle, &term_vals, DEFAULT_SLACK);
    if accepted == 0 {
        return Err(AsymptoticError::GateFailed);
    }
    terms.truncate(accepted);
    term_vals.truncate(accepted);
    let verification = verification_points(&points, &oracle, &term_vals, accepted);

    let mut hypotheses = vec![
        Hypothesis::checked(
            "the summand has a symbolic antiderivative and is finite at every check point",
        ),
        Hypothesis::assumed(
            "the summand is smooth on [a, ∞) and its high derivatives decay, so the \
             Euler–Maclaurin remainder is asymptotically negligible",
        ),
    ];
    // Only claim the fitted-constant hypothesis when a fitted constant is
    // actually part of the answer.
    if constant_position.is_some_and(|p| p < accepted) {
        hypotheses.push(Hypothesis::assumed(
            "the additive constant was fitted numerically from the exact sum, not derived; \
             it was refit at a second, larger point and agreed",
        ));
    } else {
        hypotheses.push(Hypothesis::checked(
            "no numerically fitted additive constant is part of this expansion",
        ));
    }

    Ok(AsymptoticReport {
        method: "euler-maclaurin",
        var: n,
        terms,
        rigor: Rigor::NumericallyConsistent,
        hypotheses,
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

    /// `Σ_{k=1}^{n} k⁹` is a Faulhaber polynomial, and Faulhaber polynomials
    /// have **zero constant term**. At the default `corrections = 2` the
    /// expansion is genuinely incomplete (`n⁴/2 − 3n²/20` is missing), and the
    /// honest report of that is a shorter expansion — not the missing tail
    /// frozen at the fitting point and relabelled a constant.
    #[test]
    fn faulhaber_gets_no_spurious_constant() {
        let (pool, k, n) = setup();
        let f = pool.pow(k, pool.integer(9_i32));
        let r = euler_maclaurin(f, k, 1, n, 2, &pool).expect("expansion");

        // Every emitted term must actually depend on n: a constant term here
        // would be a claim the polynomial identity contradicts.
        for &t in &r.terms {
            let mut env = std::collections::HashMap::new();
            env.insert(n, 10.0);
            let at_10 = crate::jit::eval_interp(t, &env, &pool).expect("evaluates");
            env.insert(n, 20.0);
            let at_20 = crate::jit::eval_interp(t, &env, &pool).expect("evaluates");
            assert!(
                (at_10 - at_20).abs() > 1e-9 * at_10.abs().max(1.0),
                "constant term {} in Σ k⁹ (value {at_10} at both n = 10 and n = 20)",
                pool.display(t)
            );
        }

        // What is emitted must be a genuine prefix of Faulhaber's formula.
        let partial = r.partial_sum(&pool);
        for ni in [1000.0_f64, 10_000.0] {
            let mut env = std::collections::HashMap::new();
            env.insert(n, ni);
            let approx = crate::jit::eval_interp(partial, &env, &pool).expect("evaluates");
            // Σ k⁹ = n¹⁰/10 + n⁹/2 + 3n⁸/4 − 7n⁶/10 + n⁴/2 − 3n²/20.
            let truth = ni.powi(10) / 10.0 + ni.powi(9) / 2.0 + 0.75 * ni.powi(8)
                - 0.7 * ni.powi(6)
                + 0.5 * ni.powi(4)
                - 0.15 * ni * ni;
            assert!(
                (approx - truth).abs() / truth < 1e-9,
                "n = {ni}: expansion {approx} vs Faulhaber {truth}"
            );
        }
    }

    /// The additive constant is refit at a second point and must agree; the
    /// report says which way that went.
    #[test]
    fn the_constant_is_refit_and_the_report_says_so() {
        let (pool, k, n) = setup();

        let harmonic = pool.pow(k, pool.integer(-1_i32));
        let r = euler_maclaurin(harmonic, k, 1, n, 2, &pool).expect("expansion");
        assert!(
            r.derivation
                .iter()
                .any(|d| d.contains("so it is a constant")),
            "γ must be accepted as a constant: {:?}",
            r.derivation
        );

        let ninth = pool.pow(k, pool.integer(9_i32));
        let r9 = euler_maclaurin(ninth, k, 1, n, 2, &pool).expect("expansion");
        assert!(
            r9.derivation
                .iter()
                .any(|d| d.contains("no additive constant is claimed")),
            "the Σ k⁹ fit is not a constant and must be reported as such: {:?}",
            r9.derivation
        );
        assert!(
            r9.hypotheses
                .iter()
                .all(|h| !h.statement.contains("fitted numerically")),
            "no fitted constant was emitted, so none may be claimed"
        );
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
