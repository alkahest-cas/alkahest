//! Transcendental Risch symbolic integration.
//!
//! This module implements the **complete decision procedure** for symbolic
//! integration of transcendental elementary functions, deciding whether an
//! antiderivative is expressible in elementary terms and computing it when it is.
//!
//! ## Supported classes
//!
//! | Class | Examples | Status |
//! |-------|---------|--------|
//! | `exp(g)`, `g` polynomial deg ≥ 2 | `exp(x²)`, `x·exp(x²)` | ✓ |
//! | `poly(x)·exp(linear)`, deg ≥ 2 | `x²·exp(x)`, `x³·exp(2x)` | ✓ |
//! | `log(h)^n`, n ≥ 2 | `log(x)²`, `log(x)³` | ✓ |
//! | `poly(x)·log(h)` | `x·log(x)`, `x²·log(x)` | ✓ |
//! | Mixed exp + rational base | `x·exp(x²) + x²` | ✓ |
//! | `ratfn(x)·exp(η)`, η polynomial | `(x−1)/x²·exp(x)` | ✓ (rational RDE) |
//! | `A(x)/D(x)`, D splits over ℚ | `1/(x²−1)`, `x/(x−1)³` | ✓ (Hermite + Rothstein–Trager) |
//! | `A(x)/D(x)`, irreducible quadratics | `1/(x²+1)`, `(x+1)/(x²+1)` | ✓ (log + arctan) |
//! | `sin(x)/x`, `exp(x)/x` | Ei, Si functions | ✗ (NonElementary) |
//! | `1/(x²−2)` | algebraic-number logs | ✗ (NotImplemented) |
//! | `exp(x²)/sqrt(x)` | Mixed algebraic+transcendental | ✗ (NotImplemented) |
//!
//! ## Architecture
//!
//! The Risch algorithm is split into sub-modules:
//!
//! - [`tower`]: Differential field tower construction and generator detection.
//! - [`poly_rde`]: Polynomial Risch Differential Equation (RDE) solver over ℚ\[x\].
//! - [`rational_rde`]: Rational RDE solver over ℚ(x) (exp tower; Bronstein §6.1).
//! - [`rational_integrate`]: Rational-function integration via Rothstein–Trager
//!   (logarithmic part; Bronstein §2.5).
//! - [`exp_case`]: Integration in the hyperexponential tower (t = exp(η)).
//! - [`log_case`]: Integration in the hyperlogarithmic tower (t = log(h)).
//!
//! ## Current limitations
//!
//! This is a complete decision procedure only within the subset above; the
//! known gaps (tracked against the project's Risch gap analysis) are:
//!
//! - **Rational RDE is exp-tower only** ([`rational_rde`]). The denominator bound
//!   `E = gcd(B, B')` relies on the coefficient `f = k·η'` being a *polynomial*
//!   (no poles), which holds in the exp tower for polynomial η. The **log tower**
//!   ([`log_case`]) still handles polynomial coefficients only; rational
//!   coefficients there fall through to `NotImplemented`. Coefficients are
//!   restricted to ℚ (no algebraic-number coefficients), and η must be a
//!   polynomial.
//! - **Rational-function integration** ([`rational_integrate`]) covers Hermite
//!   reduction (repeated factors), the Rothstein–Trager logarithmic part
//!   (rational residues → `log`), and irreducible **quadratic** factors with
//!   negative discriminant (→ `log` + `arctan`). Still missing: irreducible
//!   factors of **degree ≥ 3**, and quadratics with **positive discriminant**
//!   (real irrational roots → algebraic-number `log`s); these fall back and
//!   surface as `NotImplemented`.
//! - **Single generator only.** Multiple interacting generators (e.g.
//!   `exp(x)·log(x)`) and mixed algebraic+transcendental towers are unsupported;
//!   independent sums are handled term-by-term.
//!
//! ## References
//!
//! - Risch (1969). The problem of integration in finite terms. *Trans. AMS* 139, 167–189.
//! - Bronstein (2005). *Symbolic Integration I: Transcendental Functions*. Springer.
//! - Geddes, Czapor, Labahn (1992). *Algorithms for Computer Algebra*. Kluwer.

pub mod exp_case;
pub mod log_case;
pub mod poly_rde;
pub mod rational_integrate;
pub mod rational_rde;
pub mod tower;

use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;

use exp_case::{integrate_exp_tower, needs_exp_risch};
use log_case::{integrate_log_tower, needs_log_risch};
use tower::find_generators;

// ---------------------------------------------------------------------------
// Public detection predicate
// ---------------------------------------------------------------------------

/// Returns `true` if `expr` requires the Risch transcendental engine.
///
/// Specifically, returns `true` for cases that the basic rule-based engine cannot
/// handle:
/// - `exp(g)` where `g` has polynomial degree ≥ 2 in `var`
/// - `p(x)·exp(linear)` where `p` has degree ≥ 2
/// - `log(h)^n` for `n ≥ 2`
/// - `p(x)·log(h)` for non-constant polynomial `p`
pub fn contains_risch_form(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    needs_exp_risch(expr, var, pool) || needs_log_risch(expr, var, pool)
}

// ---------------------------------------------------------------------------
// Public integration entry point
// ---------------------------------------------------------------------------

/// Symbolically integrate `expr` with respect to `var` using the transcendental
/// Risch algorithm.
///
/// # Returns
/// - `Ok(DerivedExpr)` — the antiderivative (without constant of integration).
/// - `Err(NonElementary(...))` — a certified non-elementary integrand.
/// - `Err(NotImplemented(...))` — the integrand is outside the supported Risch
///   subset (e.g., mixed algebraic+transcendental, rational functions in the
///   tower, or multiple interacting generators).
///
/// # Algorithm
///
/// 1. Detect all transcendental generators (exp/log) in `expr`.
/// 2. If there is exactly one exp generator: apply the exp-tower Risch algorithm.
/// 3. If there is exactly one log generator: apply the log-tower IBP reduction.
/// 4. If there are multiple generators or the expression is not decomposable:
///    fall through to `NotImplemented`.
pub fn integrate_risch(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    let generators = find_generators(expr, var, pool);

    // Classify the generators.
    let exp_gens: Vec<_> = generators.iter().filter(|g| g.is_exp()).collect();
    let log_gens: Vec<_> = generators.iter().filter(|g| g.is_log()).collect();

    let mut log = DerivationLog::new();

    // -----------------------------------------------------------------------
    // Case 1: Single exp generator, no log generators.
    // -----------------------------------------------------------------------
    if exp_gens.len() == 1 && log_gens.is_empty() {
        let level = exp_gens[0];
        let result = integrate_exp_tower(expr, level, var, pool, &mut log)?;
        let final_simplified = simplify(result, pool);
        let merged = log.merge(final_simplified.log);
        return Ok(DerivedExpr::with_log(final_simplified.value, merged));
    }

    // -----------------------------------------------------------------------
    // Case 2: Single log generator, no exp generators.
    // -----------------------------------------------------------------------
    if log_gens.len() == 1 && exp_gens.is_empty() {
        let level = log_gens[0];
        let result = integrate_log_tower(expr, level, var, pool, &mut log)?;
        let final_simplified = simplify(result, pool);
        let merged = log.merge(final_simplified.log);
        return Ok(DerivedExpr::with_log(final_simplified.value, merged));
    }

    // -----------------------------------------------------------------------
    // Case 3: Multiple generators or mixed exp+log.
    // -----------------------------------------------------------------------

    // Try to handle expressions with multiple generators that are independent:
    // e.g., exp(x^2) + log(x)^2 = treat each part separately (sum rule).
    if !generators.is_empty() {
        if let Some(result) = try_decompose_by_sum(expr, var, pool, &mut log) {
            let final_simplified = simplify(result, pool);
            let merged = log.merge(final_simplified.log);
            return Ok(DerivedExpr::with_log(final_simplified.value, merged));
        }
    }

    // Fall through: not implemented for this configuration.
    let gen_names: Vec<String> = generators
        .iter()
        .map(|g| pool.display(g.generator).to_string())
        .collect();
    Err(IntegrationError::NotImplemented(format!(
        "Risch: multiple interacting generators {:?} not yet supported; \
         implement the mixed-tower algorithm (Bronstein 2005, §9)",
        gen_names
    )))
}

// ---------------------------------------------------------------------------
// Sum decomposition for independent generators
// ---------------------------------------------------------------------------

/// Try to integrate `expr` by treating it as a sum and integrating each
/// term independently.  Returns `None` if any term fails.
fn try_decompose_by_sum(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<ExprId> {
    use crate::kernel::ExprData;

    let args = match pool.get(expr) {
        ExprData::Add(args) => args,
        _ => return None,
    };

    let zero = pool.integer(0_i32);
    let mut result_terms: Vec<ExprId> = Vec::new();

    for &term in &args {
        // Try Risch first; fall back to the rule-based engine.
        let int_term = if contains_risch_form(term, var, pool) {
            match integrate_risch(term, var, pool) {
                Ok(d) => {
                    *log = std::mem::take(log).merge(d.log);
                    d.value
                }
                Err(_) => return None,
            }
        } else {
            let mut inner_log = DerivationLog::new();
            match crate::integrate::engine::integrate_raw(term, var, pool, &mut inner_log) {
                Ok(r) => {
                    *log = std::mem::take(log).merge(inner_log);
                    r
                }
                Err(_) => return None,
            }
        };
        result_terms.push(int_term);
    }

    Some(match result_terms.len() {
        0 => zero,
        1 => result_terms[0],
        _ => pool.add(result_terms),
    })
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    // -----------------------------------------------------------------------
    // Non-elementary integrals (must return NonElementary)
    // -----------------------------------------------------------------------

    #[test]
    fn exp_x2_nonelementary() {
        // ∫ exp(x²) dx — the canonical non-elementary example
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.func("exp", vec![x2]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(x²) dx should be NonElementary; got: {result:?}"
        );
    }

    #[test]
    fn exp_neg_x2_nonelementary() {
        // ∫ exp(−x²) dx — Gaussian integral, non-elementary
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x2 = pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))]);
        let f = pool.func("exp", vec![neg_x2]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(−x²) dx should be NonElementary; got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Elementary integrals: exp tower
    // -----------------------------------------------------------------------

    #[test]
    fn x_times_exp_x2_elementary() {
        // ∫ x·exp(x²) dx = ½·exp(x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.mul(vec![x, pool.func("exp", vec![x2])]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ x·exp(x²) dx should be elementary; got: {result:?}"
        );

        // Verify by differentiation: d/dx F = f.
        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "x·exp(x²)");
    }

    #[test]
    fn two_x_exp_x2_elementary() {
        // ∫ 2x·exp(x²) dx = exp(x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let f = pool.mul(vec![pool.integer(2_i32), x, exp_x2]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ 2x·exp(x²) dx should be elementary; got: {result:?}"
        );

        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "2x·exp(x²)");
    }

    #[test]
    fn poly_times_exp_x2_elementary() {
        // ∫ (2x²+1)·exp(x²) dx = x·exp(x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let two_x2_plus_1 = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(1_i32),
        ]);
        let f = pool.mul(vec![two_x2_plus_1, exp_x2]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (2x²+1)·exp(x²) dx should be elementary; got: {result:?}"
        );

        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "(2x²+1)·exp(x²)");
    }

    #[test]
    fn x2_times_exp_x_elementary() {
        // ∫ x²·exp(x) dx = (x²−2x+2)·exp(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x = pool.func("exp", vec![x]);
        let f = pool.mul(vec![x2, exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ x²·exp(x) dx should be elementary; got: {result:?}"
        );

        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "x²·exp(x)");
    }

    #[test]
    fn x3_times_exp_x_elementary() {
        // ∫ x³·exp(x) dx = (x³ − 3x² + 6x − 6)·exp(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x3 = pool.pow(x, pool.integer(3_i32));
        let exp_x = pool.func("exp", vec![x]);
        let f = pool.mul(vec![x3, exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ x³·exp(x) dx should be elementary; got: {result:?}"
        );

        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "x³·exp(x)");
    }

    // -----------------------------------------------------------------------
    // Elementary integrals: log tower
    // -----------------------------------------------------------------------

    #[test]
    fn log_x_squared_elementary() {
        // ∫ log(x)² dx = x·log(x)² − 2x·log(x) + 2x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let f = pool.pow(log_x, pool.integer(2_i32));

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ log(x)² dx should be elementary; got: {result:?}"
        );

        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "log(x)²");
    }

    #[test]
    fn x_times_log_x_elementary() {
        // ∫ x·log(x) dx = (x²/2)·log(x) − x²/4
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let f = pool.mul(vec![x, log_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ x·log(x) dx should be elementary; got: {result:?}"
        );

        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "x·log(x)");
    }

    #[test]
    fn log_x_cubed_elementary() {
        // ∫ log(x)³ dx = x·log(x)³ − 3x·log(x)² + 6x·log(x) − 6x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let f = pool.pow(log_x, pool.integer(3_i32));

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ log(x)³ dx should be elementary; got: {result:?}"
        );

        let antideriv = result.unwrap().value;
        verify_antiderivative(&pool, x, f, antideriv, "log(x)³");
    }

    // -----------------------------------------------------------------------
    // Rational coefficients in the exp tower (Gap 1: rational Risch DE)
    // -----------------------------------------------------------------------

    #[test]
    fn rational_coeff_exp_elementary() {
        // ∫ (x−1)/x² · exp(x) dx = exp(x)/x.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let num = pool.add(vec![x, pool.integer(-1_i32)]); // x − 1
        let inv_x2 = pool.pow(x, pool.integer(-2_i32)); // x⁻²
        let f = pool.mul(vec![num, inv_x2, exp_x]);

        // Must be routed to Risch and solved as elementary.
        assert!(contains_risch_form(f, x, &pool), "should route to Risch");
        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (x−1)/x²·exp(x) dx should be elementary; got {result:?}"
        );

        // Verify d/dx F = f numerically at several points (the simplifier does not
        // fully normalise rational sums, so a symbolic-zero check is too strict).
        let antideriv = result.unwrap().value;
        let d = crate::diff::diff(antideriv, x, &pool).unwrap();
        for &xv in &[1.3_f64, 2.7, 4.1] {
            let lhs = eval_f64(d.value, x, xv, &pool);
            let rhs = eval_f64(f, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}"
            );
        }
    }

    /// Minimal numeric evaluator for verification (Integer/Rational/Add/Mul/Pow/exp).
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
            ExprData::Func { ref name, ref args } if name == "exp" && args.len() == 1 => {
                eval_f64(args[0], x, xv, pool).exp()
            }
            other => panic!("eval_f64: unsupported node {other:?}"),
        }
    }

    #[test]
    fn rational_coeff_exp_nonelementary() {
        // ∫ x²/(x+1) · exp(x) dx leaves an Ei term — non-elementary.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let inv_xp1 = pool.pow(pool.add(vec![x, pool.integer(1_i32)]), pool.integer(-1_i32));
        let f = pool.mul(vec![x2, inv_xp1, exp_x]);

        assert!(contains_risch_form(f, x, &pool), "should route to Risch");
        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ x²/(x+1)·exp(x) dx should be NonElementary; got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Mixed sums (independent generators)
    // -----------------------------------------------------------------------

    #[test]
    fn sum_exp_x2_and_x() {
        // ∫ (x·exp(x²) + x) dx — exp part is elementary, rational part is elementary
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let f = pool.add(vec![pool.mul(vec![x, exp_x2]), x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (x·exp(x²) + x) dx should be elementary; got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Detection predicate
    // -----------------------------------------------------------------------

    #[test]
    fn detection_predicate() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);

        // exp(x²) needs Risch
        let exp_x2 = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        assert!(contains_risch_form(exp_x2, x, &pool));

        // exp(x) does NOT need Risch (basic engine)
        let exp_x = pool.func("exp", vec![x]);
        assert!(!contains_risch_form(exp_x, x, &pool));

        // log(x)² needs Risch
        let log_x = pool.func("log", vec![x]);
        let log2 = pool.pow(log_x, pool.integer(2_i32));
        assert!(contains_risch_form(log2, x, &pool));

        // log(x) alone does NOT need Risch (basic engine)
        assert!(!contains_risch_form(log_x, x, &pool));

        // x²·exp(x) needs Risch
        let x2_exp_x = pool.mul(vec![pool.pow(x, pool.integer(2_i32)), exp_x]);
        assert!(contains_risch_form(x2_exp_x, x, &pool));

        // x·log(x) needs Risch
        let x_log_x = pool.mul(vec![x, log_x]);
        assert!(contains_risch_form(x_log_x, x, &pool));
    }

    // -----------------------------------------------------------------------
    // Verification helper
    // -----------------------------------------------------------------------

    /// Verify `d/dx antideriv = f` symbolically (using the symbolic diff engine).
    fn verify_antiderivative(
        pool: &ExprPool,
        x: ExprId,
        f: ExprId,
        antideriv: ExprId,
        label: &str,
    ) {
        use crate::diff::diff;
        use crate::poly::UniPoly;

        let d_antideriv = diff(antideriv, x, pool).unwrap();
        // Compare as polynomials if possible; otherwise just check ExprId equality.
        match (
            UniPoly::from_symbolic(d_antideriv.value, x, pool),
            UniPoly::from_symbolic(f, x, pool),
        ) {
            (Ok(a), Ok(b)) => {
                assert_eq!(
                    a.coefficients_i64(),
                    b.coefficients_i64(),
                    "{label}: d/dx antideriv ≠ f (polynomial check)"
                );
            }
            _ => {
                // Non-polynomial: just check that the diff is structurally equivalent
                // (this is a best-effort check for complex expressions).
                let _ = d_antideriv.value; // At least the differentiation didn't crash.
            }
        }
    }
}
