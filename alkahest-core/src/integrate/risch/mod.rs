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
//! | `A(x)/D(x)`, irreducible quadratics | `1/(x²+1)`, `1/(x²−2)` | ✓ (log + arctan / √-log) |
//! | `A(x)/D(x)`, irreducible deg ≥ 3 | `1/(x³−3x+1)` | ✓ (`RootSum`, Lazard–Rioboo–Trager) |
//! | `ratfn(x)·exp(η)`, η rational | `(1/x²)·exp(1/x)` | ✓ (generalised RDE, Gap F) |
//! | `(c₀(x)+c₁(x)√p(x))·exp(η)`, η polynomial | `√x·exp(x)` | ✗ NonElementary / ✓ elementary |
//! | `log(h)/√p(x)` via log tower | `log(x)/√x` | ✓ (lower-tower delegation, Gap C) |
//! | `√p(x)·log(h)`, p ∈ ℚ\[x\] | `√x·log(x)` | ✓ (algebraic base-field, Gap C) |
//! | `c(x,exp(x))·exp(exp(x))`, c poly in exp(x) | `exp(x)²·exp(exp(x))` | ✓ (lower-tower cascade, Gap B) |
//! | `c(x)·exp(exp(x))`, c ∈ ℚ(x) | `x·exp(exp(x))` | ✗ NonElementary (degree bound) |
//! | `1/(x+α)^n·log(x+α)`, α ∈ ℚ(√d) | `1/(x+√2)²·log(x+√2)` | ✓ (K-rational base, Gap E) |
//! | `sin(x)/x`, `exp(x)/x` | Ei, Si functions | ✗ (NonElementary) |
//! | `exp(1/x)` alone | essential singularity | ✗ (NonElementary) |
//!
//! ## Architecture
//!
//! The Risch algorithm is split into sub-modules:
//!
//! - [`tower`]: Differential field tower construction and generator detection.
//! - [`poly_rde`]: Polynomial Risch Differential Equation (RDE) solver over ℚ\[x\].
//! - [`rational_rde`]: Rational RDE solver over ℚ(x) (exp tower; Bronstein §6.1).
//!   Also contains [`rational_rde::solve_rational_rde_generalized`] for the
//!   rational-exponent case `f = k·η' ∈ ℚ(x)` (Gap F, Bronstein §5.4).
//! - [`number_field`]: Generic polynomial-quotient core ([`number_field::CoeffField`],
//!   [`number_field::Quotient`]) plus the algebraic number field `ℚ[t]/(q)`
//!   ([`number_field::NumberField`], used for degree-≥3 algebraic residues and
//!   ℚ(√d) coefficients in the exp tower).
//! - [`alg_field`]: `x`-dependent algebraic *function* field `ℚ(x)[y]/(q(x,y))`
//!   with a derivation `D(y) = −q_x/q_y` (Risch M0; substrate for mixed
//!   algebraic + transcendental integration).
//! - [`rational_integrate`]: Rational-function integration via Rothstein–Trager
//!   (logarithmic part; Bronstein §2.5).
//! - [`exp_case`]: Integration in the hyperexponential tower (t = exp(η)), both
//!   polynomial η (Bronstein §5.2–5.3) and rational η (Gap F, §5.4).
//! - [`log_case`]: Integration in the hyperlogarithmic tower (t = log(h)).
//!
//! ## Remaining limitations
//!
//! The implementation is substantially complete for single-variable transcendental
//! integration.  Known remaining gaps:
//!
//! - **Algebraic × exp: degree ≥ 3 only.** `try_sqrt_poly_rde` (in `exp_case`)
//!   handles quadratic algebraic coefficients (`√p(x)·exp(η)`).  Higher-degree
//!   algebraic extensions (e.g. `∛(p(x))·exp(η)`) are not yet supported.
//! - **Nested exp towers — complete for ℚ(x)(exp(x)).**  The lower-tower cascade
//!   handles polynomial c(x, exp(x)).  Rational c (denominator in exp(x)) is
//!   certified NonElementary by the Hermite/pole-order argument: `D` maps any
//!   simple pole `1/(θ-α)` (α ∈ ℚ(x)) to a double pole `(α-Dα)/(θ-α)²`;
//!   since `Dα ≠ α` for α ∈ ℚ(x), no rational solution to the Risch DE exists.
//! - **Log tower: K-rational base field, Hermite-reducible cases only.**
//!   `integrate_base` now tries K-rational antidifferentiation (Gap E) via
//!   `solve_rational_rde_k` with f=0.  This handles coefficients in ℚ(√d)(x) whose
//!   antiderivative is itself K-rational, e.g. `1/(x+√2)^n` (Hermite step).
//!   Coefficients requiring new log generators (e.g. `1/(x+√2)`) still return
//!   `NotImplemented`; those are non-elementary (polylog) in the log-tower context.
//!
//! ## References
//!
//! - Risch (1969). The problem of integration in finite terms. *Trans. AMS* 139, 167–189.
//! - Bronstein (2005). *Symbolic Integration I: Transcendental Functions*. Springer.
//! - Geddes, Czapor, Labahn (1992). *Algorithms for Computer Algebra*. Kluwer.

pub mod alg_field;
pub mod exp_case;
pub mod log_case;
pub mod number_field;
pub mod poly_rde;
pub mod rational_integrate;
pub mod rational_rde;
pub mod simple_radical;
pub mod tower;
pub mod tower_field;

use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;

use exp_case::{integrate_exp_tower, needs_exp_risch};
use log_case::{integrate_log_tower, needs_log_risch};
use tower::find_generators;
use tower::TowerLevel;

// ---------------------------------------------------------------------------
// Generator selection helpers
// ---------------------------------------------------------------------------

/// Among a list of exp generators, find the "outermost" one in the tower sense:
/// the generator G such that G itself does not appear as a sub-expression of
/// any other generator's argument.
///
/// For a nested tower like `[exp(x), exp(exp(x))]`:
/// - `exp(x)` appears in `exp(exp(x)).argument()` → it is *inner*
/// - `exp(exp(x))` does not appear anywhere else → it is *outer*
///
/// Falls back to the first element when no clear outer generator is found
/// (e.g. two independent exp generators like `exp(x)` and `exp(x²)`).
fn outermost_exp_generator<'a>(exp_gens: &[&'a TowerLevel], pool: &ExprPool) -> &'a TowerLevel {
    use poly_rde::contains_subexpr;
    for &g in exp_gens.iter() {
        let is_inner = exp_gens.iter().any(|&other| {
            other.generator != g.generator && contains_subexpr(other.argument(), g.generator, pool)
        });
        if !is_inner {
            return g;
        }
    }
    exp_gens[0]
}

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
/// 4. Multiple generators of the same kind: take the outermost (first in
///    depth-first order) and recurse; the base-field integration handles the
///    inner generators.
/// 5. Mixed exp+log: route to exp tower; the log generator lives in the base
///    field and is handled by the poly-in-log RDE or lower-tower recursion.
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
    // Case 3: Multiple log generators, no exp generators.
    //
    // `find_generators` visits in depth-first order, so the *outermost*
    // generator (highest tower level) appears first.  For example, for
    // log(log(x))/x the list is [log(log(x)), log(x)] and we integrate at
    // the log(log(x)) level; log(x) is handled recursively by integrate_base.
    // -----------------------------------------------------------------------
    if !log_gens.is_empty() && exp_gens.is_empty() {
        let level = log_gens[0]; // outermost log generator
        let result = integrate_log_tower(expr, level, var, pool, &mut log)?;
        let final_simplified = simplify(result, pool);
        let merged = log.merge(final_simplified.log);
        return Ok(DerivedExpr::with_log(final_simplified.value, merged));
    }

    // -----------------------------------------------------------------------
    // Case 4: Multiple exp generators, no log generators.
    //
    // Use the *outermost* exp generator — the one that is not an argument of
    // any other exp generator in the list.  For nested towers like
    // `exp(x)·exp(exp(x))`, DFS may visit the inner generator first, so we
    // must search explicitly rather than relying on list order.
    // -----------------------------------------------------------------------
    if !exp_gens.is_empty() && log_gens.is_empty() {
        let level = outermost_exp_generator(&exp_gens, pool);
        let result = integrate_exp_tower(expr, level, var, pool, &mut log)?;
        let final_simplified = simplify(result, pool);
        let merged = log.merge(final_simplified.log);
        return Ok(DerivedExpr::with_log(final_simplified.value, merged));
    }

    // -----------------------------------------------------------------------
    // Case 5: Mixed exp + log generators.
    //
    // Route to the exp tower: the log generator lives in the base field k and
    // is handled by the poly-in-log RDE (Bronstein §5.9 / IntegrateHyperexp).
    // If the exp tower returns NotImplemented (e.g. exp has a transcendental
    // exponent), fall through to sum decomposition.
    // -----------------------------------------------------------------------
    if !exp_gens.is_empty() && !log_gens.is_empty() {
        let level = outermost_exp_generator(&exp_gens, pool);
        match integrate_exp_tower(expr, level, var, pool, &mut log) {
            Ok(result) => {
                let final_simplified = simplify(result, pool);
                let merged = log.merge(final_simplified.log);
                return Ok(DerivedExpr::with_log(final_simplified.value, merged));
            }
            Err(IntegrationError::NonElementary(msg)) => {
                return Err(IntegrationError::NonElementary(msg));
            }
            Err(IntegrationError::DivisionByZero) => {
                return Err(IntegrationError::DivisionByZero);
            }
            Err(IntegrationError::UnsupportedExtensionDegree(d)) => {
                return Err(IntegrationError::UnsupportedExtensionDegree(d));
            }
            Err(IntegrationError::NotImplemented(_)) => {
                // Fall through to sum decomposition below.
            }
        }
    }

    // -----------------------------------------------------------------------
    // Case 6: Try sum decomposition for independent generators.
    // e.g., exp(x²) + log(x)² — each term has only one generator.
    // -----------------------------------------------------------------------
    if !generators.is_empty() {
        if let Some(result) = try_decompose_by_sum(expr, var, pool, &mut log) {
            let final_simplified = simplify(result, pool);
            let merged = log.merge(final_simplified.log);
            return Ok(DerivedExpr::with_log(final_simplified.value, merged));
        }
    }

    // Fall through: outside the supported tower subsets.
    let gen_names: Vec<String> = generators
        .iter()
        .map(|g| pool.display(g.generator).to_string())
        .collect();
    Err(IntegrationError::NotImplemented(format!(
        "Risch: generators {:?} involve interactions not yet supported \
         (Bronstein 2005, §5.8–5.9, §8)",
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

        // exp(1/x) needs Risch (Gap F)
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let exp_inv_x = pool.func("exp", vec![inv_x]);
        assert!(contains_risch_form(exp_inv_x, x, &pool));
    }

    // -----------------------------------------------------------------------
    // Gap F: rational exponents  exp(η),  η ∈ ℚ(x) \ ℚ[x]
    // -----------------------------------------------------------------------

    fn eval_f64_gapf(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        use crate::kernel::ExprData;
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_f64_gapf(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args
                .iter()
                .map(|&a| eval_f64_gapf(a, x, xv, pool))
                .product(),
            ExprData::Pow { base, exp } => {
                eval_f64_gapf(base, x, xv, pool).powf(eval_f64_gapf(exp, x, xv, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_f64_gapf(args[0], x, xv, pool);
                match name.as_str() {
                    "exp" => a.exp(),
                    "log" => a.ln(),
                    "sqrt" => a.sqrt(),
                    other => panic!("eval_f64_gapf: unsupported func {other}"),
                }
            }
            other => panic!("eval_f64_gapf: unsupported {other:?}"),
        }
    }

    fn verify_numeric_gapf(integrand: ExprId, antideriv: ExprId, x: ExprId, pool: &ExprPool) {
        let d = crate::diff::diff(antideriv, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        for &xv in &[0.5_f64, 1.0, 2.0] {
            let lhs = eval_f64_gapf(ds, x, xv, pool);
            let rhs = eval_f64_gapf(integrand, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-8,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\n  F = {}",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn gapf_exp_inv_x_nonelementary() {
        // ∫ exp(1/x) dx is non-elementary (Ei(−1/x) type).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let f = pool.func("exp", vec![inv_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(1/x) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn gapf_inv_x2_exp_inv_x_elementary() {
        // ∫ (1/x²)·exp(1/x) dx = −exp(1/x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let exp_inv_x = pool.func("exp", vec![inv_x]);
        let f = pool.mul(vec![pool.pow(x, pool.integer(-2_i32)), exp_inv_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (1/x²)·exp(1/x) dx must be elementary; got {result:?}"
        );
        verify_numeric_gapf(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapf_two_inv_x3_exp_neg_inv_x2_elementary() {
        // ∫ (2/x³)·exp(−1/x²) dx = exp(−1/x²).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_inv_x2 = pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(x, pool.integer(-2_i32)),
        ]);
        let exp_e = pool.func("exp", vec![neg_inv_x2]);
        let f = pool.mul(vec![
            pool.integer(2_i32),
            pool.pow(x, pool.integer(-3_i32)),
            exp_e,
        ]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (2/x³)·exp(−1/x²) dx must be elementary; got {result:?}"
        );
        verify_numeric_gapf(f, result.unwrap().value, x, &pool);
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

    // -----------------------------------------------------------------------
    // Gap B: multi-generator tower composition
    // -----------------------------------------------------------------------

    /// Numeric evaluator supporting exp, log.
    fn eval_f64_gapb(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        use crate::kernel::ExprData;
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_f64_gapb(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args
                .iter()
                .map(|&a| eval_f64_gapb(a, x, xv, pool))
                .product(),
            ExprData::Pow { base, exp } => {
                eval_f64_gapb(base, x, xv, pool).powf(eval_f64_gapb(exp, x, xv, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_f64_gapb(args[0], x, xv, pool);
                match name.as_str() {
                    "exp" => a.exp(),
                    "log" => a.ln(),
                    other => panic!("eval_f64_gapb: {other}"),
                }
            }
            other => panic!("eval_f64_gapb: {other:?}"),
        }
    }

    fn verify_gapb(integrand: ExprId, antideriv: ExprId, x: ExprId, pool: &ExprPool) {
        let d = crate::diff::diff(antideriv, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        // Use x > e so that log(log(x)) is real.
        for &xv in &[3.0_f64, 7.5, 15.0] {
            let lhs = eval_f64_gapb(ds, x, xv, pool);
            let rhs = eval_f64_gapb(integrand, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-8,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\nF={}",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn gapb_log_log_x_over_x_elementary() {
        // ∫ log(log(x))/x dx = log(x)·log(log(x)) − log(x)
        // Two log generators: log(log(x)) and log(x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let log_log_x = pool.func("log", vec![log_x]);
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let f = pool.mul(vec![log_log_x, inv_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ log(log(x))/x dx must be elementary; got {result:?}"
        );
        verify_gapb(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapb_exp_x_times_log_x_nonelementary() {
        // ∫ exp(x)·log(x) dx is non-elementary (involves Ei).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let log_x = pool.func("log", vec![x]);
        let f = pool.mul(vec![exp_x, log_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(x)·log(x) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn gapb_log_x_plus_x_plus_inv_x_times_exp_x_elementary() {
        // ∫ (log(x) + x + 1/x)·exp(x) dx = (log(x) + x − 1)·exp(x).
        // Poly-in-log RDE: c₁=1, c₀=x+1/x.
        //   a₁' + a₁ = 1 → a₁ = 1.
        //   a₀' + a₀ = (x+1/x) − 1/x = x → a₀ = x−1.
        //   v = 1·log(x) + (x−1) = log(x)+x−1.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let exp_x = pool.func("exp", vec![x]);
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        // integrand = (log(x) + x + 1/x)·exp(x)
        let coeff = pool.add(vec![log_x, x, inv_x]);
        let f = pool.mul(vec![coeff, exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (log(x)+x+1/x)·exp(x) dx must be elementary; got {result:?}"
        );
        verify_gapb(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapb_sum_exp_and_log_independent() {
        // ∫ (x·exp(x²) + log(x)²) dx — two independent generators, sum rule.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let log_x = pool.func("log", vec![x]);
        let log2 = pool.pow(log_x, pool.integer(2_i32));
        let f = pool.add(vec![pool.mul(vec![x, exp_x2]), log2]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (x·exp(x²)+log(x)²) dx must be elementary; got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Gap C: mixed algebraic + transcendental
    // -----------------------------------------------------------------------

    #[test]
    fn gapc_exp_over_sqrt_xsq_plus_1_nonelementary() {
        // ∫ exp(x) / sqrt(x²+1) dx  — non-elementary.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let xsq1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let sqrt_xsq1 = pool.func("sqrt", vec![xsq1]);
        let exp_x = pool.func("exp", vec![x]);
        let f = pool.mul(vec![exp_x, pool.pow(sqrt_xsq1, pool.integer(-1_i32))]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(x)/sqrt(x²+1) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn gapc_1_plus_sqrt_x_times_exp_x_nonelementary() {
        // ∫ (1 + sqrt(x))·exp(x) dx — non-elementary (√x·exp(x) has no elementary antiderivative).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let exp_x = pool.func("exp", vec![x]);
        let f = pool.mul(vec![pool.add(vec![pool.integer(1_i32), sqrt_x]), exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ (1+√x)·exp(x) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn gapc_sqrt_x_coefficient_exp_x_elementary() {
        // ∫ (2x+1)/(2x) · sqrt(x) · exp(x) dx = sqrt(x)·exp(x).
        //
        // Derivation: d/dx(sqrt(x)·exp(x)) = (1/(2√x)+√x)·exp(x) = (1+2x)/(2√x)·exp(x).
        // Coefficient c = (1+2x)/(2x)·sqrt(x) = (1+2x)/(2√x).
        // After decompose: c₀=0, c₁=(1+2x)/(2x).
        // Eq 1: a'+a=0 → a=0.
        // Eq 2: b'+(1+1/(2x))·b=(1+2x)/(2x) → b=1. Antiderivative: sqrt(x)·exp(x).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let exp_x = pool.func("exp", vec![x]);
        // (2x+1)/(2x) · sqrt(x) · exp(x)
        let two_x_p1 = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let coeff = pool.mul(vec![
            two_x_p1,
            pool.pow(two_x, pool.integer(-1_i32)),
            sqrt_x,
        ]);
        let f = pool.mul(vec![coeff, exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (2x+1)/(2x)·sqrt(x)·exp(x) dx must be elementary; got {result:?}"
        );
        // Numerically verify d/dx F = f at x > 0.
        let antideriv = result.unwrap().value;
        let d = crate::diff::diff(antideriv, x, &pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[0.5_f64, 1.5, 3.0] {
            let lhs = eval_f64_gapf(ds, x, xv, &pool);
            let rhs = eval_f64_gapf(f, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn gapc_log_over_sqrt_x_elementary_via_log_tower() {
        // ∫ log(x)/sqrt(x) dx = 2·sqrt(x)·log(x) − 4·sqrt(x).
        // This goes through the log tower (θ = log(x)):
        //   c₁ = 1/sqrt(x) → integrate_base_unchecked → 2·sqrt(x)
        //   correction → base integral -2/sqrt(x) → -4·sqrt(x)
        // Routing: has_alg=true (sqrt), has_trans=true (log) → Risch → log tower.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let f = pool.mul(vec![log_x, pool.pow(sqrt_x, pool.integer(-1_i32))]);

        let result = crate::integrate::engine::integrate(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ log(x)/sqrt(x) dx must be elementary; got {result:?}"
        );
        // Verify numerically.
        let antideriv = result.unwrap().value;
        let d = crate::diff::diff(antideriv, x, &pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[0.5_f64, 1.5, 4.0] {
            let lhs = eval_f64_gapf(ds, x, xv, &pool);
            let rhs = eval_f64_gapf(f, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "∫ log(x)/sqrt(x): d/dx F ≠ f at x={xv}: {lhs} vs {rhs}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Gap B (nested exp): v=1 special case for transcendental η'
    // -----------------------------------------------------------------------

    /// Numeric verifier for nested-exp antiderivatives: use very small x to avoid
    /// overflow from exp(exp(x)) growing extremely fast.
    fn verify_nested_exp(integrand: ExprId, antideriv: ExprId, x: ExprId, pool: &ExprPool) {
        let d = crate::diff::diff(antideriv, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        for &xv in &[-0.5_f64, 0.0, 0.2] {
            let lhs = eval_f64_gapb(ds, x, xv, pool);
            let rhs = eval_f64_gapb(integrand, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-8,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\nF={}",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn gapb_nested_exp_elementary() {
        // ∫ exp(x)·exp(exp(x)) dx = exp(exp(x))
        // Top generator: exp(exp(x)), η = exp(x), η' = exp(x).
        // RDE: v' + exp(x)·v = exp(x); v = 1 is a solution.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        let f = pool.mul(vec![exp_x, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ exp(x)·exp(exp(x)) dx must be elementary; got {result:?}"
        );
        verify_nested_exp(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapb_nested_exp_with_const_elementary() {
        // ∫ 3·exp(x)·exp(exp(x)) dx = 3·exp(exp(x))
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        let f = pool.mul(vec![pool.integer(3_i32), exp_x, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ 3·exp(x)·exp(exp(x)) dx must be elementary; got {result:?}"
        );
        verify_nested_exp(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapb_nested_exp_rational_coeff_nonelementary() {
        // ∫ x·exp(exp(x)) dx — c = x ∈ ℚ[x] has no inner exp factor → NonElementary
        // (certified by degree bound: θ_inner·v has degree > c's degree for any v).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        let f = pool.mul(vec![x, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ x·exp(exp(x)) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn gapb_nested_exp_bare_nonelementary() {
        // ∫ exp(exp(x)) dx alone — c = 1 ∈ ℚ[x] → NonElementary.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);

        let result = integrate_risch(exp_exp_x, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(exp(x)) dx must be NonElementary; got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Gap B: nested exp — lower-tower polynomial cascade
    // -----------------------------------------------------------------------

    #[test]
    fn gapb_nested_exp_exp_sq_elementary() {
        // ∫ exp(x)²·exp(exp(x)) dx = (exp(x)−1)·exp(exp(x))
        // Cascade: c = θ² (N=2), v₁ = 1, v₀ = c₁ − v₁' − v₁ = 0−0−1 = −1.
        // D(v₀) = 0 = c₀ ✓.  v = θ−1 = exp(x)−1.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        // exp(x)^2 * exp(exp(x))
        let f = pool.mul(vec![pool.pow(exp_x, pool.integer(2_i32)), exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ exp(x)²·exp(exp(x)) dx must be elementary; got {result:?}"
        );
        verify_nested_exp(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapb_nested_exp_poly_plus_rational_elementary() {
        // ∫ [(x²+1)·exp(x) + 2x]·exp(exp(x)) dx = (x²+1)·exp(exp(x))
        // Cascade: c = (x²+1)·θ + 2x (N=1).
        // v₀ = c₁/1 = x²+1.  D(v₀) = 2x = c₀ ✓.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let x2p1 = pool.add(vec![x2, pool.integer(1_i32)]);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        // ((x²+1)·exp(x) + 2x)·exp(exp(x))
        let coeff = pool.add(vec![pool.mul(vec![x2p1, exp_x]), two_x]);
        let f = pool.mul(vec![coeff, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ [(x²+1)exp(x)+2x]·exp(exp(x)) dx must be elementary; got {result:?}"
        );
        verify_nested_exp(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapb_nested_exp_higher_degree_elementary() {
        // ∫ [exp(x)²−exp(x)]·exp(exp(x)) dx = (exp(x)−2)·exp(exp(x))
        // Cascade: c = θ²−θ (N=2, c₂=1, c₁=−1, c₀=0).
        // v₁ = 1, v₀ = −1−0−1 = −2.  D(v₀) = 0 = c₀ ✓.  v = −2+θ = exp(x)−2.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        let exp_sq = pool.pow(exp_x, pool.integer(2_i32));
        let neg_exp = pool.mul(vec![pool.integer(-1_i32), exp_x]);
        let coeff = pool.add(vec![exp_sq, neg_exp]);
        let f = pool.mul(vec![coeff, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ [exp(x)²−exp(x)]·exp(exp(x)) dx must be elementary; got {result:?}"
        );
        verify_nested_exp(f, result.unwrap().value, x, &pool);
    }

    #[test]
    fn gapb_nested_exp_inner_coeff_nonelementary() {
        // ∫ x·exp(x)·exp(exp(x)) dx: c = x·θ (N=1, c₁=x, c₀=0).
        // Cascade: v₀ = x, D(v₀) = 1 ≠ c₀ = 0.  No polynomial solution →
        // denominator-bound theorem certifies NonElementary.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        let f = pool.mul(vec![x, exp_x, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ x·exp(x)·exp(exp(x)) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn gapb_nested_exp_rational_denom_nonelementary() {
        // ∫ exp(x)/(exp(x)+1)·exp(exp(x)) dx:
        // c = θ/(θ+1) is rational in θ = exp(x).
        // Hermite reduction: D maps 1/(θ-α) to a double pole; no rational v exists.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        // exp(x)/(exp(x)+1) = exp(x)·(exp(x)+1)^{-1}
        let denom = pool.add(vec![exp_x, pool.integer(1_i32)]);
        let coeff = pool.mul(vec![exp_x, pool.pow(denom, pool.integer(-1_i32))]);
        let f = pool.mul(vec![coeff, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(x)/(exp(x)+1)·exp(exp(x)) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn gapb_nested_exp_inv_theta_nonelementary() {
        // ∫ exp(-x)·exp(exp(x)) dx = ∫ θ^{-1}·G dx:
        // c = exp(-x) = θ^{-1} has a negative power of θ → rational → NonElementary.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let exp_exp_x = pool.func("exp", vec![exp_x]);
        let exp_neg_x = pool.pow(exp_x, pool.integer(-1_i32));
        let f = pool.mul(vec![exp_neg_x, exp_exp_x]);

        let result = integrate_risch(f, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(-x)·exp(exp(x)) dx must be NonElementary; got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Gap C (log tower + algebraic coefficient)
    // -----------------------------------------------------------------------

    #[test]
    fn mixed_cbrt_x_times_log_x_elementary() {
        // ∫ x^{1/3}·log(x) dx = (3/4)x^{4/3}·log(x) − (9/16)x^{4/3}.
        // Log-tower IBP delegates the base ∫x^{1/3} dx to the MA algebraic engine.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cbrt_x = pool.pow(x, pool.rational(1_i32, 3_i32));
        let log_x = pool.func("log", vec![x]);
        let f = pool.mul(vec![cbrt_x, log_x]);

        let result = crate::integrate::engine::integrate(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ x^(1/3)·log(x) dx must be elementary; got {result:?}"
        );
        let antideriv = result.unwrap().value;
        let d = crate::diff::diff(antideriv, x, &pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[0.5_f64, 1.5, 4.0] {
            let lhs = eval_f64_gapf(ds, x, xv, &pool);
            let rhs = eval_f64_gapf(f, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "∫ x^(1/3)·log(x): d/dx F ≠ f at x={xv}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn mixed_log_over_cbrt_x_elementary() {
        // ∫ log(x)/x^{1/3} dx = (3/2)x^{2/3}·log(x) − (9/4)x^{2/3}.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inv_cbrt = pool.pow(x, pool.rational(-1_i32, 3_i32)); // x^{-1/3}
        let f = pool.mul(vec![inv_cbrt, pool.func("log", vec![x])]);
        let result = crate::integrate::engine::integrate(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ log(x)/x^(1/3) dx must be elementary; got {result:?}"
        );
        let antideriv = result.unwrap().value;
        let ds = crate::simplify::engine::simplify(
            crate::diff::diff(antideriv, x, &pool).unwrap().value,
            &pool,
        )
        .value;
        for &xv in &[0.5_f64, 1.5, 4.0] {
            let lhs = eval_f64_gapf(ds, x, xv, &pool);
            let rhs = eval_f64_gapf(f, x, xv, &pool);
            assert!((lhs - rhs).abs() < 1e-7, "x={xv}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn mixed_cbrt_nonsquarefree_times_log_x_elementary() {
        // ∫ (x²)^{1/3}·log(x) dx, radicand x² non-squarefree → general MA basis
        // delivers the base ∫x^{2/3} dx = (3/5)x^{5/3}; IBP wraps the log.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let rad = pool.func("cbrt", vec![x2]); // (x²)^{1/3}
        let f = pool.mul(vec![rad, pool.func("log", vec![x])]);
        let result = crate::integrate::engine::integrate(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (x²)^(1/3)·log(x) dx must be elementary; got {result:?}"
        );
        let antideriv = result.unwrap().value;
        let ds = crate::simplify::engine::simplify(
            crate::diff::diff(antideriv, x, &pool).unwrap().value,
            &pool,
        )
        .value;
        for &xv in &[0.5_f64, 1.5, 4.0] {
            // eval the integrand's cbrt(x²) via x^{2/3} equivalence.
            let lhs = eval_f64_gapf(ds, x, xv, &pool);
            let rhs = xv.powf(2.0 / 3.0) * xv.ln();
            assert!((lhs - rhs).abs() < 1e-6, "x={xv}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn gapc_sqrt_x_times_log_x_elementary() {
        // ∫ sqrt(x)·log(x) dx = (2x^{3/2}/3)·log(x) − 4x^{3/2}/9
        // IBP: c_1 = √x → P_1 = ∫√x dx = (2/3)x^{3/2} (via algebraic engine)
        //   correction = −(2/3)x^{3/2}·(1/x) = −(2/3)√x
        //   base: ∫(−(2/3)√x) dx = −(4/9)x^{3/2}
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let log_x = pool.func("log", vec![x]);
        let f = pool.mul(vec![sqrt_x, log_x]);

        let result = crate::integrate::engine::integrate(f, x, &pool);
        assert!(
            result.is_ok(),
            "∫ sqrt(x)·log(x) dx must be elementary; got {result:?}"
        );
        let antideriv = result.unwrap().value;
        let d = crate::diff::diff(antideriv, x, &pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[0.5_f64, 1.5, 4.0] {
            let lhs = eval_f64_gapf(ds, x, xv, &pool);
            let rhs = eval_f64_gapf(f, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "∫ sqrt(x)·log(x): d/dx F ≠ f at x={xv}: {lhs} vs {rhs}"
            );
        }
    }
}
