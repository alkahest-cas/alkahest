//! Algebraic-function Risch integration (V1-2).
//!
//! Extends the Risch integration engine to handle integrands containing
//! algebraic subterms: `sqrt(P(x))`, `P(x)^(p/q)`, and rational combinations
//! thereof over Q(x).
//!
//! Supports degree-2 algebraic extensions K = Q(x)\[y\]/(y² - P(x)):
//! - P of degree 0 (constant): trivial
//! - P of degree 1 (linear): complete elementary integration via substitution
//! - P of degree 2 (quadratic): genus-0 curve, always elementary
//! - P of degree ≥ 3: returns `NonElementary` (elliptic/hyperelliptic integrals)
//!
//! Higher-degree extensions (y^q = P, q > 2) return `UnsupportedExtensionDegree`.
//!
//! References:
//! - Trager (1984). Integration of algebraic functions. MIT PhD thesis.
//! - Bronstein (2005). Symbolic Integration I. Springer, chs. 10–11.

pub(super) mod decompose;
pub(super) mod genus_zero;
pub(super) mod poly_utils;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use decompose::decompose_sqrt;
use genus_zero::integrate_with_sqrt;
use poly_utils::is_zero_expr;

// ---------------------------------------------------------------------------
// Algebraic subterm detection
// ---------------------------------------------------------------------------

/// Returns `true` if `expr` contains any algebraic subterm (sqrt or fractional
/// power) that requires the algebraic integration path.
pub fn contains_algebraic_subterm(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } => {
            if name == "sqrt" {
                return true;
            }
            args.iter().any(|&a| contains_algebraic_subterm(a, pool))
        }
        ExprData::Pow { base, exp } => {
            // Any Rational exponent triggers algebraic detection
            if matches!(pool.get(exp), ExprData::Rational(_)) {
                return true;
            }
            // For integer exponent, recurse into the base only
            contains_algebraic_subterm(base, pool)
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_algebraic_subterm(a, pool))
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Generator discovery
// ---------------------------------------------------------------------------

/// Walk `expr` and collect all algebraic generator IDs (sqrt or P^(1/2)).
fn collect_generators(expr: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } => {
            if name == "sqrt" && args.len() == 1 {
                out.push(expr);
                // Do not recurse into the argument — nested sqrt not supported
            } else {
                for &a in args.iter() {
                    collect_generators(a, pool, out);
                }
            }
        }
        ExprData::Pow { base, exp } => {
            if matches!(pool.get(exp), ExprData::Rational(_)) {
                out.push(expr);
            } else {
                collect_generators(base, pool, out);
                // exp is an integer, no need to recurse into it
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in args.iter() {
                collect_generators(a, pool, out);
            }
        }
        _ => {}
    }
}

/// Extract the radicand from a `sqrt(P)` or `P^(1/2)` expression.
fn get_radicand(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => Some(args[0]),
        ExprData::Pow { base, exp } => {
            // Check for Rational(1/2)
            match pool.get(exp) {
                ExprData::Rational(r) if r.0 == rug::Rational::from((1u32, 2u32)) => Some(base),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Find a unique algebraic generator in `expr`.
/// Returns `Some((sqrt_id, radicand_id))` when there is exactly one generator.
fn find_generator(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let mut generators = Vec::new();
    collect_generators(expr, pool, &mut generators);
    generators.sort_unstable();
    generators.dedup();
    if generators.len() != 1 {
        return None;
    }
    let sqrt_id = generators[0];
    let radicand = get_radicand(sqrt_id, pool)?;
    Some((sqrt_id, radicand))
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Symbolically integrate `expr` with respect to `var`, where `expr` contains
/// algebraic subterms (sqrt or fractional powers).
///
/// Precondition: `contains_algebraic_subterm(expr, pool)` is `true`.
pub fn integrate_algebraic(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    let mut log = DerivationLog::new();

    // Step 1: Find the unique sqrt generator y = sqrt(P(x))
    let (sqrt_id, p_expr) = find_generator(expr, pool).ok_or_else(|| {
        IntegrationError::NotImplemented(
            "algebraic integrator requires exactly one sqrt(P(x)) generator; \
             multiple or nested generators are not supported in v1.1"
                .to_string(),
        )
    })?;

    // Step 2: Validate extension degree (only degree-2 extensions in v1.1)
    // Higher-degree generators like P^(1/3) are rejected here.
    if let ExprData::Pow { exp, .. } = pool.get(sqrt_id) {
        if let ExprData::Rational(r) = pool.get(exp) {
            let q = r.0.denom().to_u32().unwrap_or(0);
            if q != 2 {
                return Err(IntegrationError::UnsupportedExtensionDegree(q));
            }
        }
    }

    // Step 3: Decompose integrand as A(x) + B(x)·sqrt(P)
    let (a_raw, b_raw) = decompose_sqrt(expr, sqrt_id, p_expr, pool).ok_or_else(|| {
        IntegrationError::NotImplemented(
            "could not decompose integrand into A(x) + B(x)·sqrt(P(x)); \
             expression structure is not supported"
                .to_string(),
        )
    })?;

    // Simplify both parts so that subsequent pattern matching works on canonical forms.
    // (e.g. field inversion produces Mul(-1, 1, Pow(Mul(-1, P), -1)) which simplifies to P^-1)
    let a_part = simplify(a_raw, pool).value;
    let b_part = simplify(b_raw, pool).value;

    let zero = pool.integer(0_i32);

    // Step 4: Integrate the rational part ∫ A(x) dx
    let int_a = if is_zero_expr(a_part, pool) {
        zero
    } else {
        crate::integrate::engine::integrate_raw(a_part, var, pool, &mut log)?
    };

    // Step 5: Integrate the algebraic part ∫ B(x)·sqrt(P) dx
    let int_b = if is_zero_expr(b_part, pool) {
        zero
    } else {
        integrate_with_sqrt(b_part, p_expr, sqrt_id, var, pool, &mut log)?
    };

    // Step 6: Combine
    let raw = match (is_zero_expr(int_a, pool), is_zero_expr(int_b, pool)) {
        (true, true) => zero,
        (false, true) => int_a,
        (true, false) => int_b,
        (false, false) => pool.add(vec![int_a, int_b]),
    };

    // Step 7: Simplify and record derivation
    let simplified = simplify(raw, pool);
    log = log.merge(simplified.log);
    log.push(RewriteStep::simple(
        "algebraic_risch",
        expr,
        simplified.value,
    ));

    Ok(DerivedExpr::with_log(simplified.value, log))
}
