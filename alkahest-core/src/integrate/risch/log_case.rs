//! Transcendental Risch integration: the hyperlogarithmic (log) case.
//!
//! Integrates expressions of the form:
//! ```text
//!   ∫ p(x) · log(h(x))^n  dx
//! ```
//! where `p ∈ ℚ(x)` and `n ≥ 0`.
//!
//! **Algorithm** (integration by parts, repeated reduction):
//!
//! For `n ≥ 1`:
//! ```text
//!   ∫ p(x) · log(h)^n dx = P(x) · log(h)^n
//!                          − n · ∫ P(x) · h'(x)/h(x) · log(h)^{n−1} dx
//! ```
//! where `P' = p` (an antiderivative of p in the base field).
//!
//! The recursion terminates at `n = 0`:
//! ```text
//!   ∫ p(x) dx   (base field integral, handled by the rule-based engine)
//! ```
//!
//! For the special case `h = x` (i.e., `log(x)`), the term `h'/h = 1/x` and
//! `P(x)/x` can be split into its polynomial and `1/x` components.  This gives
//! nested log terms like `log(x)^{n+1}/(n+1)`.
//!
//! **Coverage**:
//! - `∫ log(x) dx = x·log(x) − x` (already in the basic engine; handled here too)
//! - `∫ log(x)² dx = x·log(x)² − 2x·log(x) + 2x`
//! - `∫ x·log(x) dx = (x²/2)·log(x) − x²/4`
//! - `∫ log(a·x + b) dx = ((a·x+b)/a)·log(a·x+b) − x`
//!
//! **Non-elementary detection**:
//! - `∫ log(x)/x^2 dx` and similar forms where the coefficient is not integrable
//!   return `NonElementary` or `NotImplemented`.
//!
//! References: Bronstein (2005), §6; Geddes, Czapor, Labahn §12.3.

use crate::deriv::log::{DerivationLog, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::poly_rde::is_free_of_var;
use super::tower::{decompose_as_log_poly, ExtensionKind, TowerLevel};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Integrate `expr` with respect to `var`, given that `expr` involves the
/// hyperlogarithmic generator `level` (with `level.generator = log(h)`).
///
/// Handles expressions that are polynomials in `log(h)` with polynomial (or
/// rational function) coefficients in `x`.
pub fn integrate_log_tower(
    expr: ExprId,
    level: &TowerLevel,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let log_gen = level.generator; // ExprId of log(h)
    let h = match level.kind {
        ExtensionKind::Log { h } => h,
        _ => {
            return Err(IntegrationError::NotImplemented(
                "integrate_log_tower called with non-Log level".to_string(),
            ))
        }
    };

    // Decompose expr as a polynomial in log_gen.
    let coeffs = decompose_as_log_poly(expr, log_gen, pool).ok_or_else(|| {
        IntegrationError::NotImplemented(format!(
            "could not decompose {} as a polynomial in log({})",
            pool.display(expr),
            pool.display(h)
        ))
    })?;

    // Trim trailing zero coefficients from the log polynomial.
    let coeffs = trim_zero_coeffs(coeffs, pool);

    // Integrate the polynomial in log_gen using the IBP recursion.
    integrate_log_poly(&coeffs, log_gen, h, var, pool, log)
}

// ---------------------------------------------------------------------------
// IBP recursion
// ---------------------------------------------------------------------------

/// Integrate `sum_{k=0}^n c_k(x) · log(h)^k` by repeated integration by parts.
///
/// `coeffs[k]` is the coefficient of `log(h)^k` (as a symbolic ExprId).
fn integrate_log_poly(
    coeffs: &[ExprId],
    log_gen: ExprId, // log(h)
    h: ExprId,       // argument of log
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let zero = pool.integer(0_i32);

    if coeffs.is_empty() {
        return Ok(zero);
    }

    // n = highest degree present.
    let n = coeffs.len() - 1;

    // Base case: n = 0, just integrate c_0(x).
    if n == 0 {
        let c0 = coeffs[0];
        return integrate_base(c0, var, pool, log);
    }

    // General step: work from degree n down to 0.
    // ∫ c_n(x)·log(h)^n dx = P_n(x)·log(h)^n − n·∫ P_n(x)·(h'/h)·log(h)^{n-1} dx
    // where P_n is an antiderivative of c_n.

    // Start with the full polynomial and collect terms.
    integrate_log_poly_recursive(coeffs, log_gen, h, var, pool, log)
}

/// Recursive helper: integrate `sum_k c_k · log(h)^k` by reducing from the top.
fn integrate_log_poly_recursive(
    coeffs: &[ExprId],
    log_gen: ExprId,
    h: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let zero = pool.integer(0_i32);

    if coeffs.is_empty() {
        return Ok(zero);
    }

    // Find the highest nonzero degree.
    let n = find_top_degree(coeffs, pool);
    if n == 0 {
        // Only a constant (in log) term.
        return integrate_base(coeffs[0], var, pool, log);
    }

    // Simplify c_n before using it (avoids passing unsimplified expressions to integrate_base).
    let c_n = simplify(coeffs[n], pool).value;

    // If c_n is zero, recurse with degree n-1.
    if is_zero(c_n, pool) {
        return integrate_log_poly_recursive(&coeffs[..n], log_gen, h, var, pool, log);
    }

    // Step 1: compute P_n = ∫ c_n(x) dx (in the base field).
    let p_n_raw = integrate_base(c_n, var, pool, log)?;
    // Simplify P_n to avoid propagating complex unsimplified forms.
    let p_n = simplify(p_n_raw, pool).value;

    // Step 2: build the term P_n · log(h)^n.
    let log_n = log_power(log_gen, n as i64, pool);
    let term_top = if is_one(p_n, pool) {
        log_n
    } else {
        pool.mul(vec![p_n, log_n])
    };

    // Step 3: compute h'(x) / h(x) as a symbolic expression, then simplify.
    let h_prime = differentiate_sym(h, var, pool)?;
    let h_prime_expr = simplify(h_prime, pool).value;
    let h_prime_over_h = if is_one(h, pool) {
        pool.integer(0_i32) // h = 1: h'/h = 0
    } else {
        // Build h'/h and simplify immediately to prevent unsimplified 1/x forms
        // from propagating into the integral.
        let raw = pool.mul(vec![h_prime_expr, pool.pow(h, pool.integer(-1_i32))]);
        simplify(raw, pool).value
    };

    // Step 4: build the correction term for the recursive step:
    // new_c_{n-1} = old_c_{n-1} + (−n) · P_n · (h'/h)
    // Simplify the correction to keep expressions canonical.
    let neg_n = pool.integer(-(n as i64));
    let correction_raw = pool.mul(vec![neg_n, p_n, h_prime_over_h]);
    let correction = simplify(correction_raw, pool).value;

    // Build the new coefficient vector with degree n-1.
    let mut new_coeffs: Vec<ExprId> = if n > 0 { coeffs[..n].to_vec() } else { vec![] };
    if new_coeffs.is_empty() {
        new_coeffs.push(zero);
    }
    // Add and simplify the correction at degree n-1.
    let old_cn1 = new_coeffs[n - 1];
    let combined = pool.add(vec![old_cn1, correction]);
    new_coeffs[n - 1] = simplify(combined, pool).value;

    // Step 5: recursively integrate the remaining polynomial.
    let rest = integrate_log_poly_recursive(&new_coeffs, log_gen, h, var, pool, log)?;

    // Step 6: combine.
    let result = if is_zero(rest, pool) {
        term_top
    } else {
        pool.add(vec![term_top, rest])
    };

    let simplified = simplify(result, pool);
    *log = log.clone().merge(simplified.log);
    log.push(RewriteStep::simple("risch_log_ibp", c_n, simplified.value));

    Ok(simplified.value)
}

// ---------------------------------------------------------------------------
// Base-field integration (no log)
// ---------------------------------------------------------------------------

/// Integrate `expr` with respect to `var` in the base differential field ℚ(x).
///
/// First simplifies `expr`, then tries the rule-based engine, then falls back
/// to the rational-function integrator (Hermite + Rothstein–Trager) for
/// coefficients that arise from the IBP reduction and are not simple enough for
/// the rule engine alone (e.g. `x²/(x+1)`, `1/(x+1)²`).
///
/// **Correctness note:** when called at IBP level k ≥ 1 the result may contain
/// log terms (e.g. `∫ 1/(x+1) dx = log(x+1)`).  If those log terms involve the
/// current log-tower generator `h`, the IBP correction `P·h'/h` becomes
/// transcendental and the recursion will return `NotImplemented` at the next
/// level — this is correct behaviour (the integral is likely non-elementary).
fn integrate_base(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // Simplify first to canonicalize expressions like `(-1) * x * (x^-1) = -1`.
    let expr = crate::simplify::engine::simplify(expr, pool).value;

    if is_zero(expr, pool) {
        return Ok(pool.integer(0_i32));
    }

    let mut inner_log = DerivationLog::new();
    let result = match crate::integrate::engine::integrate_raw(expr, var, pool, &mut inner_log) {
        Ok(r) => r,
        Err(crate::integrate::engine::IntegrationError::NotImplemented(_)) => {
            // Fallback: rational-function integration via Hermite reduction +
            // Rothstein–Trager.  This handles coefficients like x²/(x+1) or
            // 1/(x+1)² that arise in the IBP loop but are beyond the power rule.
            match crate::integrate::risch::rational_integrate::try_integrate_rational(
                expr, var, pool,
            ) {
                Some(r) => r,
                None => {
                    return Err(crate::integrate::engine::IntegrationError::NotImplemented(
                        format!(
                            "integrate_base: {} is not integrable in ℚ(x)",
                            pool.display(expr)
                        ),
                    ))
                }
            }
        }
        Err(other) => return Err(other),
    };
    let result = crate::simplify::engine::simplify(result, pool).value;
    *log = log.clone().merge(inner_log);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Symbolic differentiation helper
// ---------------------------------------------------------------------------

fn differentiate_sym(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, IntegrationError> {
    use crate::diff::diff;
    match diff(expr, var, pool) {
        Ok(d) => Ok(d.value),
        Err(e) => Err(IntegrationError::NotImplemented(format!(
            "could not differentiate {}: {e}",
            pool.display(expr)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` if `expr` is symbolically zero.
fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Integer(n) if n.0 == 0)
}

/// Returns `true` if `expr` is symbolically 1.
fn is_one(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Integer(n) if n.0 == 1)
}

/// Build `log_gen^n` as a symbolic ExprId.
fn log_power(log_gen: ExprId, n: i64, pool: &ExprPool) -> ExprId {
    match n {
        0 => pool.integer(1_i32),
        1 => log_gen,
        _ => pool.pow(log_gen, pool.integer(n)),
    }
}

/// Find the index of the highest nonzero element in `coeffs`.
fn find_top_degree(coeffs: &[ExprId], pool: &ExprPool) -> usize {
    for k in (0..coeffs.len()).rev() {
        if !is_zero(coeffs[k], pool) {
            return k;
        }
    }
    0
}

/// Trim trailing zero coefficients from the log polynomial.
fn trim_zero_coeffs(mut coeffs: Vec<ExprId>, pool: &ExprPool) -> Vec<ExprId> {
    while coeffs.last().is_some_and(|&c| is_zero(c, pool)) {
        coeffs.pop();
    }
    if coeffs.is_empty() {
        coeffs.push(pool.integer(0_i32));
    }
    coeffs
}

// ---------------------------------------------------------------------------
// Detection: does an integrand require the log-tower path?
// ---------------------------------------------------------------------------

/// Returns `true` if `expr` contains a log term that requires the Risch log-tower
/// path (i.e., `log(h)^n` for `n ≥ 2`, or `c(x)·log(h)` for non-constant `c`).
///
/// Excludes `log(x)` alone (handled by the basic engine's `int_log` rule).
pub fn needs_log_risch(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    needs_log_risch_inner(expr, var, pool)
}

fn needs_log_risch_inner(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            // log(h)^n for n ≥ 2: needs Risch.
            if let ExprData::Func { ref name, ref args } = pool.get(base) {
                if name == "log" && args.len() == 1 {
                    if let ExprData::Integer(n) = pool.get(exp) {
                        if n.0 >= 2 {
                            return true;
                        }
                    }
                }
            }
            needs_log_risch_inner(base, var, pool) || needs_log_risch_inner(exp, var, pool)
        }
        ExprData::Mul(args) => {
            // Check for c(x) · log(h) where c is non-constant.
            let has_log = args.iter().any(|&a| is_log_expr(a, pool));
            let has_nonconstant = args
                .iter()
                .any(|&a| !is_free_of_var(a, var, pool) && !is_log_expr(a, pool));
            if has_log && has_nonconstant {
                return true;
            }
            args.iter().any(|&a| needs_log_risch_inner(a, var, pool))
        }
        ExprData::Add(args) => args.iter().any(|&a| needs_log_risch_inner(a, var, pool)),
        _ => false,
    }
}

/// Returns true if `expr` is a `log(h)` call.
fn is_log_expr(expr: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Func { ref name, ref args } if name == "log" && args.len() == 1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn log_x_squared() {
        // ∫ log(x)² dx = x·log(x)² − 2x·log(x) + 2x
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.pow(log_x, pool.integer(2_i32));

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1, "should find exactly one log generator");
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ log(x)² dx should be elementary: {:?}",
            result
        );
        let antideriv = result.unwrap();
        let s = pool.display(antideriv).to_string();
        assert!(s.contains("log"), "result should contain log: {}", s);
    }

    #[test]
    fn x_times_log_x() {
        // ∫ x·log(x) dx = (x²/2)·log(x) − x²/4
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![x, log_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ x·log(x) dx should be elementary: {:?}",
            result
        );
    }

    #[test]
    fn log_x_alone() {
        // ∫ log(x) dx = x·log(x) − x
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);

        use super::super::tower::find_generators;
        let gens = find_generators(log_x, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(log_x, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ log(x) dx should be elementary: {:?}",
            result
        );
        let s = pool.display(result.unwrap()).to_string();
        assert!(s.contains("log"), "result should contain log: {}", s);
    }

    #[test]
    fn needs_log_risch_detection() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);

        // log(x) alone: basic engine handles it
        assert!(!needs_log_risch(log_x, x, &pool));

        // log(x)^2: needs Risch
        let log2 = pool.pow(log_x, pool.integer(2_i32));
        assert!(needs_log_risch(log2, x, &pool));

        // x·log(x): needs Risch (non-constant coefficient)
        let x_log_x = pool.mul(vec![x, log_x]);
        assert!(needs_log_risch(x_log_x, x, &pool));
    }

    // -----------------------------------------------------------------------
    // Rational-coefficient log tower (Gap A: RT fallback in integrate_base)
    // -----------------------------------------------------------------------

    /// Numeric evaluator for IBP verification: supports Integer, Rational,
    /// Add, Mul, Pow, log, atan.
    fn eval_f64(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        use crate::kernel::ExprData;
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_f64(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval_f64(a, x, xv, pool)).product(),
            ExprData::Pow { base, exp } => {
                eval_f64(base, x, xv, pool).powf(eval_f64(exp, x, xv, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_f64(args[0], x, xv, pool);
                match name.as_str() {
                    "log" => a.ln(),
                    "atan" => a.atan(),
                    "sqrt" => a.sqrt(),
                    other => panic!("eval_f64: unsupported func {other}"),
                }
            }
            other => panic!("eval_f64: unsupported node {other:?}"),
        }
    }

    /// Verify d/dx antideriv ≈ integrand numerically at a few points.
    fn verify_numeric(integrand: ExprId, antideriv: ExprId, x: ExprId, pool: &ExprPool) {
        let d = crate::diff::diff(antideriv, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        for &xv in &[0.3_f64, 1.7, 3.1] {
            let lhs = eval_f64(ds, x, xv, pool);
            let rhs = eval_f64(integrand, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-8,
                "d/dx F ≠ f at x={xv}: got {lhs}, expected {rhs}\n  F = {}",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn x_times_log_x_plus_1() {
        // ∫ x·log(x+1) dx = (x²/2 − 1/2)·log(x+1) − x²/4 + x/2
        // Polynomial c_1=x → P_1=x²/2 → c_0 = −x²/(2(x+1)) (rational).
        // The rational base-case integral uses the RT fallback.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_xp1 = pool.func("log", vec![pool.add(vec![x, pool.integer(1_i32)])]);
        let integrand = pool.mul(vec![x, log_xp1]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1, "should find exactly one log generator");
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ x·log(x+1) dx should be elementary: {:?}",
            result
        );
        verify_numeric(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn log_xp1_over_xp1_squared() {
        // ∫ log(x+1)/(x+1)² dx = −log(x+1)/(x+1) − 1/(x+1)
        // Rational c_1 = 1/(x+1)² → Hermite gives P_1 = −1/(x+1) (purely rational)
        // → c_0 = 1/(x+1)² → Hermite again for the base case.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let log_xp1 = pool.func("log", vec![xp1]);
        let integrand = pool.mul(vec![log_xp1, pool.pow(xp1, pool.integer(-2_i32))]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ log(x+1)/(x+1)² dx should be elementary: {:?}",
            result
        );
        verify_numeric(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn log_x_over_xp1_squared() {
        // ∫ log(x)/(x+1)² dx: rational c_1 = 1/(x+1)² with h=x.
        // Hermite → P_1 = −1/(x+1) (rational), correction = 1/(x(x+1)) (rational).
        // Base case: ∫ 1/(x(x+1)) dx = log(x) − log(x+1) via partial fractions.
        // Full result: −log(x)/(x+1) + log(x) − log(x+1).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![log_x, pool.pow(xp1, pool.integer(-2_i32))]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ log(x)/(x+1)² dx should be elementary: {:?}",
            result
        );
        // The test points must avoid the singularity at x=0 and x=-1.
        let d = crate::diff::diff(result.unwrap(), x, &pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[0.5_f64, 1.5, 2.5] {
            let lhs = eval_f64(ds, x, xv, &pool);
            let rhs = eval_f64(integrand, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn x2_times_log_x_plus_1() {
        // ∫ x²·log(x+1) dx: polynomial c_1=x² → P_1=x³/3 → c_0=−x³/(3(x+1))
        // (rational, needs RT + polynomial division).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_xp1 = pool.func("log", vec![pool.add(vec![x, pool.integer(1_i32)])]);
        let integrand = pool.mul(vec![pool.pow(x, pool.integer(2_i32)), log_xp1]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ x²·log(x+1) dx should be elementary: {:?}",
            result
        );
        verify_numeric(integrand, result.unwrap(), x, &pool);
    }
}
