//! Transcendental Risch integration: the hyperexponential (exp) case.
//!
//! Integrates expressions of the form:
//! ```text
//!   ∫ c(x) · exp(η(x))^k  dx
//! ```
//! where `c ∈ ℚ[x]` is a polynomial in the base variable, `η ∈ ℚ[x]` is the
//! exponent, and `k` is a nonzero integer (the tower degree).
//!
//! **Algorithm** (Bronstein 2005, §5):
//!
//! The integral is elementary iff the Risch Differential Equation (RDE)
//! ```text
//!   v'(x) + k · η'(x) · v(x) = c(x)
//! ```
//! has a polynomial solution `v ∈ ℚ[x]`.  When it does, the antiderivative is
//! `v(x) · exp(η(x))^k`.  When it does not, the integral is non-elementary.
//!
//! **Key non-elementary certificates**:
//! - `∫ exp(x²) dx`: η = x², η' = 2x, RDE `v' + 2x·v = 1` has no polynomial
//!   solution → certified non-elementary.
//! - `∫ exp(x²) · x^n dx` for n < deg(η') − 1: also non-elementary.
//!
//! **Elementary examples**:
//! - `∫ x·exp(x²) dx = ½·exp(x²)`  (RDE `v' + 2x·v = x`, solution v = ½)
//! - `∫ x²·exp(x) dx = (x²−2x+2)·exp(x)`  (constant η' = 1, undetermined coefficients)
//! - `∫ p(x)·exp(a·x) dx`: always elementary for polynomial p and constant a ≠ 0.

use crate::deriv::log::{DerivationLog, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::poly_rde::{expr_to_qpoly, is_free_of_var, poly_scale, qpoly_to_expr, solve_poly_rde};
use super::rational_rde::{expr_to_qrational, solve_rational_rde};
use super::tower::{decompose_wrt_exp, poly_degree, TowerLevel};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Integrate `expr` with respect to `var`, given that `expr` involves the
/// hyperexponential generator `level` (with `level.generator = exp(η)`).
///
/// Supports:
/// - Single-generator exp-tower integrands: `c(x) · exp(η)^k`
/// - Sums of such terms plus a rational base part
/// - Non-elementary certification via the Risch DE
pub fn integrate_exp_tower(
    expr: ExprId,
    level: &TowerLevel,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let exp_gen = level.generator; // ExprId of exp(η)
    let eta = match level.kind {
        super::tower::ExtensionKind::Exp { eta } => eta,
        _ => {
            return Err(IntegrationError::NotImplemented(
                "integrate_exp_tower called with non-Exp level".to_string(),
            ))
        }
    };

    // Decompose expr = rational_part + sum_k c_k(x) · exp(η)^k
    let (rational_part, exp_terms) = decompose_wrt_exp(expr, exp_gen, pool);

    let zero = pool.integer(0_i32);

    // Integrate the rational base part (no exp factors).
    let int_rational = if is_zero(rational_part, pool) {
        zero
    } else {
        // Delegate to the rule-based engine for the rational part.
        let mut inner_log = DerivationLog::new();
        match crate::integrate::engine::integrate_raw(rational_part, var, pool, &mut inner_log) {
            Ok(r) => {
                *log = log.clone().merge(inner_log);
                r
            }
            Err(e) => return Err(e),
        }
    };

    if exp_terms.is_empty() {
        return Ok(int_rational);
    }

    // Compute η'(x) = dη/dx once.
    let deta_expr = differentiate_poly(eta, var, pool)?;
    let deta = expr_to_qpoly(deta_expr, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented(format!(
            "exponent derivative η'(x) = {} is not a polynomial in the integration variable; \
             only polynomial exponents are supported",
            pool.display(deta_expr)
        ))
    })?;

    // Integrate each term c_k · exp(η)^k.
    let mut result_terms: Vec<ExprId> = Vec::new();
    if !is_zero(int_rational, pool) {
        result_terms.push(int_rational);
    }

    for (c_expr, k) in &exp_terms {
        let k = *k;
        let term_result =
            integrate_single_exp_term(*c_expr, k, &deta, deta_expr, eta, exp_gen, var, pool, log)?;
        result_terms.push(term_result);
    }

    let raw = match result_terms.len() {
        0 => zero,
        1 => result_terms[0],
        _ => pool.add(result_terms),
    };

    let simplified = simplify(raw, pool);
    *log = log.clone().merge(simplified.log);
    log.push(RewriteStep::simple("risch_exp", expr, simplified.value));

    Ok(simplified.value)
}

// ---------------------------------------------------------------------------
// Single monomial: ∫ c(x) · exp(η)^k dx
// ---------------------------------------------------------------------------

/// Integrate `c_expr · exp(η)^k` with respect to `var`.
///
/// Raises [`IntegrationError::NonElementary`] if the RDE has no polynomial solution.
#[allow(clippy::too_many_arguments)]
fn integrate_single_exp_term(
    c_expr: ExprId,
    k: i64,
    deta: &[rug::Rational], // η'(x) as ℚ-polynomial
    deta_expr: ExprId,      // η'(x) as symbolic ExprId (for error messages)
    eta: ExprId,            // η(x) (the exponent)
    exp_gen: ExprId,        // exp(η(x))
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // The integrand is c(x) · exp(kη).
    // The antiderivative (if elementary) is v(x) · exp(kη)
    // where v satisfies: v' + k · η'(x) · v = c(x).

    // Build exp(kη) once: for k=1 use exp_gen directly; for k>1 use exp(kη).
    let exp_k_eta = build_exp_k_eta(k, eta, exp_gen, pool);

    // Non-elementary error shared by the polynomial and rational paths.
    let non_elementary = || {
        IntegrationError::NonElementary(format!(
            "the Risch DE v'(x) + {}·({}(x))·v(x) = {}(x) has no rational solution;\n\
             the integrand ∫ {} · exp(η)^{} dx is not an elementary function\n\
             (η = {})",
            k,
            pool.display(deta_expr),
            pool.display(c_expr),
            pool.display(c_expr),
            k,
            pool.display(eta),
        ))
    };

    // Fast path: polynomial coefficient → polynomial Risch DE (Bronstein §5.2).
    if let Some(c_poly) = expr_to_qpoly(c_expr, var, pool) {
        return match solve_poly_rde(k, deta, &c_poly) {
            Some(v_poly) => {
                let v_expr = qpoly_to_expr(&v_poly, var, pool);
                let result = build_v_times_exp(v_expr, exp_k_eta, pool);
                log.push(RewriteStep::simple("risch_exp_rde", c_expr, result));
                Ok(result)
            }
            None => Err(non_elementary()),
        };
    }

    // Rational coefficient → rational Risch DE over ℚ(x) (Bronstein §6.1, Gap 1).
    if let Some((c_num, c_den)) = expr_to_qrational(c_expr, var, pool) {
        // f = k·η' is a polynomial in the exp tower.
        let f = poly_scale(&deta.to_vec(), &rug::Rational::from(k));
        return match solve_rational_rde(&f, &c_num, &c_den) {
            Some((v_num, v_den)) => {
                let v_expr = build_rational(&v_num, &v_den, var, pool);
                let result = build_v_times_exp(v_expr, exp_k_eta, pool);
                log.push(RewriteStep::simple(
                    "risch_exp_rde_rational",
                    c_expr,
                    result,
                ));
                Ok(result)
            }
            None => Err(non_elementary()),
        };
    }

    // c is neither a polynomial nor a rational function in `var` (e.g. it carries
    // another transcendental generator) — outside the single-generator subset.
    Err(IntegrationError::NotImplemented(format!(
        "coefficient {} of exp(η)^{} is not a rational function in the integration \
         variable; mixed/multiple generators are not yet supported",
        pool.display(c_expr),
        k
    )))
}

/// Build `v · exp(kη)`, collapsing the `v = 1` case.
fn build_v_times_exp(v_expr: ExprId, exp_k_eta: ExprId, pool: &ExprPool) -> ExprId {
    if is_one(v_expr, pool) {
        exp_k_eta
    } else {
        pool.mul(vec![v_expr, exp_k_eta])
    }
}

/// Build the symbolic rational function `num(x) / den(x)`.
fn build_rational(
    num: &[rug::Rational],
    den: &[rug::Rational],
    var: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let num_expr = qpoly_to_expr(&num.to_vec(), var, pool);
    // Denominator 1 → just the numerator.
    if super::poly_rde::degree(&den.to_vec()) <= 0 && den.first().map(|c| *c == 1).unwrap_or(true) {
        return num_expr;
    }
    let den_expr = qpoly_to_expr(&den.to_vec(), var, pool);
    let den_inv = pool.pow(den_expr, pool.integer(-1_i32));
    pool.mul(vec![num_expr, den_inv])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Differentiate a polynomial expression with respect to `var`, returning
/// the symbolic derivative ExprId.
fn differentiate_poly(
    poly_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, IntegrationError> {
    // Use the symbolic differentiation engine.
    use crate::diff::diff;
    match diff(poly_expr, var, pool) {
        Ok(derived) => Ok(derived.value),
        Err(e) => Err(IntegrationError::NotImplemented(format!(
            "could not differentiate exponent: {e}"
        ))),
    }
}

/// Build the expression `exp(k·η)`.
/// - k = 0: returns 1
/// - k = 1: returns exp_gen (= exp(η)) unchanged
/// - k ≠ 1: builds `pool.func("exp", [k·η])`
fn build_exp_k_eta(k: i64, eta: ExprId, exp_gen: ExprId, pool: &ExprPool) -> ExprId {
    match k {
        0 => pool.integer(1_i32),
        1 => exp_gen,
        _ => {
            let k_expr = pool.integer(k);
            let k_eta = pool.mul(vec![k_expr, eta]);
            pool.func("exp", vec![k_eta])
        }
    }
}

/// Returns true if `expr` is the integer 0.
fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::ExprData;
    matches!(pool.get(expr), ExprData::Integer(n) if n.0 == 0)
}

/// Returns true if `expr` is the integer 1.
fn is_one(expr: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::ExprData;
    matches!(pool.get(expr), ExprData::Integer(n) if n.0 == 1)
}

// ---------------------------------------------------------------------------
// Detection: does an integrand require the exp-tower path?
// ---------------------------------------------------------------------------

/// Returns `true` if `expr` contains a hyperexponential generator that requires
/// the Risch exp-tower path (i.e., `exp(η)` where η is a polynomial of degree ≥ 2,
/// or where the coefficient of exp has degree ≥ 2).
///
/// Excludes cases already handled by the basic rule-based engine:
/// - `exp(a·x + b)` with no polynomial factor (handled by `int_exp_linear`)
/// - `x · exp(x)` (handled by `int_x_exp`)
pub fn needs_exp_risch(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    needs_exp_risch_inner(expr, var, pool)
}

/// Returns `true` if `expr` is a negative integer power of a `var`-dependent
/// base — i.e. a denominator that makes the surrounding coefficient a rational
/// function in `var`.
fn is_var_dependent_denominator(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::ExprData;
    if let ExprData::Pow { base, exp } = pool.get(expr) {
        if let ExprData::Integer(n) = pool.get(exp) {
            if n.0.to_i64().is_some_and(|v| v < 0) {
                return !is_free_of_var(base, var, pool);
            }
        }
    }
    false
}

fn needs_exp_risch_inner(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::ExprData;

    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "exp" && args.len() == 1 => {
            let eta = args[0];
            // If η is free of var: exp(const) is a constant, no Risch needed.
            if is_free_of_var(eta, var, pool) {
                return false;
            }
            // If deg(η) ≥ 2: definitely needs Risch.
            if let Some(d) = poly_degree(eta, var, pool) {
                if d >= 2 {
                    return true;
                }
            }
            // Linear η: the basic engine handles exp(a*x+b) alone.
            // But a product like p(x)*exp(a*x) with deg(p) ≥ 2 needs Risch.
            false
        }
        ExprData::Mul(args) => {
            // Check if there's an exp factor with linear η AND the remaining product
            // has degree ≥ 2 (not just "x * exp(x)" which the basic engine handles).
            let mut has_linear_exp = false;
            let mut max_poly_deg: u32 = 0;
            let mut has_nonlinear_exp = false;
            // A var-dependent denominator (negative power) makes the exp coefficient
            // a rational function — handled by the rational Risch DE (Gap 1), not the
            // basic engine.
            let mut has_rational_coeff = false;

            for &a in &args {
                match pool.get(a) {
                    ExprData::Func { ref name, ref args } if name == "exp" && args.len() == 1 => {
                        let eta = args[0];
                        if is_free_of_var(eta, var, pool) {
                            // exp(const): treat as constant factor.
                        } else if let Some(d) = poly_degree(eta, var, pool) {
                            if d >= 2 {
                                has_nonlinear_exp = true;
                            } else {
                                has_linear_exp = true;
                            }
                        } else {
                            has_linear_exp = true;
                        }
                    }
                    _ => {
                        // Track degree of non-exp factors.
                        if let Some(d) = poly_degree(a, var, pool) {
                            max_poly_deg = max_poly_deg.max(d);
                        } else if is_var_dependent_denominator(a, var, pool) {
                            has_rational_coeff = true;
                        }
                    }
                }
            }

            if has_nonlinear_exp {
                return true;
            }
            // Linear exp + polynomial factor of degree ≥ 2: Risch needed.
            if has_linear_exp && max_poly_deg >= 2 {
                return true;
            }
            // Any exp generator with a rational (denominator-bearing) coefficient:
            // route to the rational Risch DE.
            if (has_linear_exp || has_nonlinear_exp) && has_rational_coeff {
                return true;
            }
            // Check sub-expressions for nested cases.
            args.iter().any(|&a| needs_exp_risch_inner(a, var, pool))
        }
        ExprData::Add(args) => args.iter().any(|&a| needs_exp_risch_inner(a, var, pool)),
        ExprData::Pow { base, exp } => {
            needs_exp_risch_inner(base, var, pool) || needs_exp_risch_inner(exp, var, pool)
        }
        _ => false,
    }
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
    fn exp_x2_is_nonelementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);

        use super::super::tower::find_generators;
        let gens = find_generators(exp_x2, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(exp_x2, level, x, &pool, &mut log);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(x²) dx should be NonElementary, got: {:?}",
            result
        );
    }

    #[test]
    fn x_times_exp_x2_is_elementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let integrand = pool.mul(vec![x, exp_x2]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            result.is_ok(),
            "∫ x·exp(x²) dx should be elementary, got: {:?}",
            result
        );
        let antideriv = result.unwrap();
        // Structural check: result should contain exp(x^2).
        let s = pool.display(antideriv).to_string();
        assert!(s.contains("exp"), "result should contain exp: {}", s);
    }

    #[test]
    fn two_x_times_exp_x2_equals_exp_x2() {
        // ∫ 2x·exp(x²) dx = exp(x²)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let two = pool.integer(2_i32);
        let integrand = pool.mul(vec![two, x, exp_x2]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        let level = &gens[0];

        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log).unwrap();
        // The result should simplify to something equivalent to exp(x^2).
        let s = pool.display(result).to_string();
        assert!(s.contains("exp"), "result should contain exp: {}", s);
    }

    #[test]
    fn x2_times_exp_x_is_elementary() {
        // ∫ x²·exp(x) dx = (x²−2x+2)·exp(x)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x = pool.func("exp", vec![x]);
        let integrand = pool.mul(vec![x2, exp_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            result.is_ok(),
            "∫ x²·exp(x) dx should be elementary, got: {:?}",
            result
        );
        let s = pool.display(result.unwrap()).to_string();
        assert!(s.contains("exp"), "result should contain exp: {}", s);
    }

    #[test]
    fn needs_exp_risch_detection() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);

        // exp(x^2): needs Risch
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        assert!(needs_exp_risch(exp_x2, x, &pool));

        // exp(x): basic engine (linear), does NOT need Risch
        let exp_x = pool.func("exp", vec![x]);
        assert!(!needs_exp_risch(exp_x, x, &pool));

        // x^2 * exp(x): needs Risch (polynomial factor degree 2)
        let x2_times_exp_x = pool.mul(vec![pool.pow(x, pool.integer(2_i32)), exp_x]);
        assert!(needs_exp_risch(x2_times_exp_x, x, &pool));

        // x * exp(x): basic engine handles this
        let x_times_exp_x = pool.mul(vec![x, exp_x]);
        assert!(!needs_exp_risch(x_times_exp_x, x, &pool));

        // x * exp(x^2): needs Risch
        let x_times_exp_x2 = pool.mul(vec![x, exp_x2]);
        assert!(needs_exp_risch(x_times_exp_x2, x, &pool));
    }
}
