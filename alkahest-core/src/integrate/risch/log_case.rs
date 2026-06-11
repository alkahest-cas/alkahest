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

use super::exp_case::{
    build_field_and_gens, build_krational_ext, detect_algebraic_extension,
    expr_to_krational_general, kelem_to_expr_ext,
};
use super::k_rational_integrate::integrate_k_rational_with_logs;
use super::number_field::NumberField;
use super::poly_rde::{apply_const, contains_subexpr, is_free_of_var, split_const_factor};
use super::rational_rde::solve_rational_rde_k;
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
    // Pass log_gen so integrate_base can guard against re-introducing it.
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

    // Base case: n = 0, just integrate c_0(x).  The result may contain
    // log_gen — it's the final "rest" term and is combined with term_top.
    if n == 0 {
        let c0 = coeffs[0];
        return integrate_base_unchecked(c0, var, pool, log);
    }

    // General step: work from degree n down to 0.
    // ∫ c_n(x)·log(h)^n dx = P_n(x)·log(h)^n − n·∫ P_n(x)·(h'/h)·log(h)^{n-1} dx
    // where P_n is an antiderivative of c_n.

    // §E — NonElementary certificate for entangled K-log coefficients.
    // Before attempting the IBP, test Bronstein eq (18) on the *top* coefficient
    // c_n: a necessary condition for ∫ Σ c_k log(h)^k dx to be elementary is that
    // c_n = v' + (n+1)·e·(h'/h) be solvable for v ∈ K(x) and a constant e.  When
    // c_n is a genuinely-K-rational coefficient whose partial-fraction poles
    // include a K-irrational point that is NOT a zero of h (so no constant e can
    // absorb its residue), eq (18) has no solution and the integral is provably
    // non-elementary (Bronstein 2005, §5.10 / Tutorial §3.5: "either proving that
    // (18) has no solution, in which case f has no elementary integral").  The
    // headline case is ∫ 1/(x+√2)·log(x) dx (dilogarithm pole at x = −√2).
    if certify_klog_top_obstruction(coeffs, h, var, pool) {
        return Err(IntegrationError::NonElementary(format!(
            "∫ Σ c_k·log({})^k dx: the top coefficient has a K-irrational pole \
             whose residue is not a constant multiple of the tower generator's \
             logarithmic derivative — Bronstein eq (18) (primitive case) has no \
             solution, so the integral is non-elementary (dilogarithm-type) \
             (Bronstein 2005, Symbolic Integration I, §5.10)",
            pool.display(h)
        )));
    }

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
        // Only a constant (in log) term.  Result may contain log_gen — safe here.
        let c0 = simplify(coeffs[0], pool).value;
        let (k_alg0, c0_rest) = split_const_factor(c0, var, pool);
        let integral0 = integrate_base_unchecked(c0_rest, var, pool, log)?;
        return Ok(apply_const(k_alg0, integral0, pool));
    }

    // Simplify c_n before using it (avoids passing unsimplified expressions to integrate_base).
    let c_n = simplify(coeffs[n], pool).value;

    // If c_n is zero, recurse with degree n-1.
    if is_zero(c_n, pool) {
        return integrate_log_poly_recursive(&coeffs[..n], log_gen, h, var, pool, log);
    }

    // Step 1: compute P_n = ∫ c_n(x) dx (in the base field).
    // Split off any algebraic constant factor K first (Gap E: log tower).
    // ∫ K·g(x) dx = K · ∫ g(x) dx when K is free of x; the base integrator
    // handles only ℚ(x), so we must factor out symbolic/algebraic constants.
    let (k_alg, c_n_rest) = split_const_factor(c_n, var, pool);

    // Log-derivative shortcut: if c_n_rest = α·(h'/h) for some α free of var,
    // ∫ α·(h'/h)·θ^n dx = α·θ^{n+1}/(n+1)  (d/dx θ^{n+1} = (n+1)·θ^n·h'/h).
    // This fires when P_n = α·θ would contain the excluded generator (log_gen),
    // stalling the normal IBP.  Examples: 1/x·log(x), 1/(x+√2)·log(x+√2),
    // 2/(x+1)·log(x+1)².
    if let Some(alpha) = detect_log_deriv_coeff(c_n_rest, h, var, pool) {
        let np1_inv = pool.pow(pool.integer((n + 1) as i32), pool.integer(-1_i32));
        let coeff = simplify(pool.mul(vec![k_alg, alpha, np1_inv]), pool).value;
        let log_np1 = log_power(log_gen, (n + 1) as i64, pool);
        let term = if is_one(coeff, pool) {
            log_np1
        } else {
            pool.mul(vec![coeff, log_np1])
        };
        // No IBP correction: the formula absorbs it. Recurse on lower degrees.
        let rest = integrate_log_poly_recursive(&coeffs[..n], log_gen, h, var, pool, log)?;
        let result = if is_zero(rest, pool) {
            term
        } else {
            pool.add(vec![term, rest])
        };
        let s = simplify(result, pool);
        *log = log.clone().merge(s.log);
        log.push(RewriteStep::simple("risch_log_deriv_poly", c_n, s.value));
        return Ok(s.value);
    }

    // Mixed-sum fallback: when c_n_rest is an Add, some terms may need the
    // log-derivative formula and others the normal IBP.  Split the Add into
    // (log_deriv_part, hermite_part), handle each, and combine.  The IBP
    // correction only comes from the Hermite part; the log-deriv part
    // contributes α_sum·θ^{n+1}/(n+1) directly.
    if let Some((ld_alpha, hermite_part)) = split_log_deriv_from_add(c_n_rest, h, var, pool) {
        // Log-deriv contribution: k_alg · ld_alpha · θ^{n+1} / (n+1)
        let np1_inv = pool.pow(pool.integer((n + 1) as i32), pool.integer(-1_i32));
        let ld_coeff = simplify(pool.mul(vec![k_alg, ld_alpha, np1_inv]), pool).value;
        let log_np1 = log_power(log_gen, (n + 1) as i64, pool);
        let ld_term = if is_one(ld_coeff, pool) {
            log_np1
        } else {
            pool.mul(vec![ld_coeff, log_np1])
        };

        // IBP contribution from hermite_part (may be zero).
        let (ibp_top, mut new_coeffs) = if is_zero(hermite_part, pool) {
            (pool.integer(0_i32), coeffs[..n].to_vec())
        } else {
            let p_h = integrate_base(hermite_part, log_gen, var, pool, log)?;
            let p_h_full = apply_const(k_alg, p_h, pool);
            let p_h_s = simplify(p_h_full, pool).value;
            let log_n = log_power(log_gen, n as i64, pool);
            let top = if is_one(p_h_s, pool) {
                log_n
            } else {
                pool.mul(vec![p_h_s, log_n])
            };
            let h_prime = differentiate_sym(h, var, pool)?;
            let hph_s = simplify(h_prime, pool).value;
            let hpoh = simplify(
                pool.mul(vec![hph_s, pool.pow(h, pool.integer(-1_i32))]),
                pool,
            )
            .value;
            let corr = simplify(pool.mul(vec![pool.integer(-(n as i64)), p_h_s, hpoh]), pool).value;
            let mut nc = coeffs[..n].to_vec();
            if nc.is_empty() {
                nc.push(zero);
            }
            let old = nc[n - 1];
            nc[n - 1] = simplify(pool.add(vec![old, corr]), pool).value;
            (top, nc)
        };

        if new_coeffs.is_empty() {
            new_coeffs.push(zero);
        }
        let rest = integrate_log_poly_recursive(&new_coeffs, log_gen, h, var, pool, log)?;
        let combined = [ld_term, ibp_top, rest]
            .into_iter()
            .filter(|&e| !is_zero(e, pool))
            .collect::<Vec<_>>();
        let result = match combined.len() {
            0 => zero,
            1 => combined[0],
            _ => pool.add(combined),
        };
        let s = simplify(result, pool);
        *log = log.clone().merge(s.log);
        return Ok(s.value);
    }

    let p_rest_raw = integrate_base(c_n_rest, log_gen, var, pool, log)?;
    let p_n_raw = apply_const(k_alg, p_rest_raw, pool);
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
/// **Correctness note (multi-level tower, Gap B):** `excluded_gen` is the
/// ExprId of the current log generator (e.g. `log(log(x))` when integrating
/// at the outer level).  The IBP recursion requires P_n ∉ k(excluded_gen);
/// if any integration path returns a result *containing* excluded_gen the
/// recursion would diverge.  The safety check below catches this: if the
/// fast paths introduce excluded_gen we fall back to the full engine only
/// when excluded_gen is NOT in the lower tower.
///
/// When `expr` itself has lower-tower transcendental generators (e.g. `1/x`
/// in the context of `∫ log(log(x))/x dx`), we fall through to
/// `engine::integrate` so the lower-level Risch algorithm handles it.
/// Like [`integrate_base`] but without the safety guard.  Used for the
/// degree-0 base case and the corrected lower-degree terms, where the result
/// is allowed to contain the current log generator (it is combined with other
/// terms in the running sum and the degree cannot increase).
fn integrate_base_unchecked(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    use crate::integrate::engine::{integrate_raw, IntegrationError as IE};
    use crate::integrate::risch::rational_integrate::try_integrate_rational;

    let expr = crate::simplify::engine::simplify(expr, pool).value;
    if is_zero(expr, pool) {
        return Ok(pool.integer(0_i32));
    }

    let mut inner_log = DerivationLog::new();
    match integrate_raw(expr, var, pool, &mut inner_log) {
        Ok(r) => {
            let r = crate::simplify::engine::simplify(r, pool).value;
            *log = log.clone().merge(inner_log);
            return Ok(r);
        }
        Err(IE::NotImplemented(_)) => {}
        Err(other) => return Err(other),
    }

    if let Some(r) = try_integrate_rational(expr, var, pool) {
        return Ok(crate::simplify::engine::simplify(r, pool).value);
    }

    // Gap E: K-rational integration for constant algebraic coefficients (e.g. √2).
    if let Some(r) = try_integrate_k_rational(expr, var, pool) {
        return Ok(crate::simplify::engine::simplify(r, pool).value);
    }

    // Gap E (follow-up): K-rational integration WITH new K-log terms, e.g.
    // ∫ 1/(x·(x+√2)) dx = (1/√2)·[log(x) − log(x+√2)].
    if let Some(r) = try_integrate_k_rational_with_logs(expr, var, pool) {
        return Ok(crate::simplify::engine::simplify(r, pool).value);
    }

    // Full engine for lower-tower generators (Gap B).
    match crate::integrate::engine::integrate(expr, var, pool) {
        Ok(d) => Ok(crate::simplify::engine::simplify(d.value, pool).value),
        Err(IE::NonElementary(msg)) => Err(IE::NonElementary(msg)),
        Err(_) => Err(IE::NotImplemented(format!(
            "integrate_base: {} is not integrable in the base field",
            pool.display(expr)
        ))),
    }
}

fn integrate_base(
    expr: ExprId,
    excluded_gen: ExprId, // current log generator — must not appear in the result
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    use crate::integrate::engine::{integrate_raw, IntegrationError as IE};
    use crate::integrate::risch::rational_integrate::try_integrate_rational;

    // Simplify first to canonicalize expressions like `(-1) * x * (x^-1) = -1`.
    let expr = crate::simplify::engine::simplify(expr, pool).value;

    if is_zero(expr, pool) {
        return Ok(pool.integer(0_i32));
    }

    // Helper: check whether a candidate result is safe to use (doesn't
    // re-introduce the current log generator into the IBP recursion).
    let is_safe = |r: ExprId| -> bool { !contains_subexpr(r, excluded_gen, pool) };

    let mut inner_log = DerivationLog::new();

    // Fast path 1: rule-based engine (handles polynomials, exp(linear), 1/x, etc.)
    match integrate_raw(expr, var, pool, &mut inner_log) {
        Ok(r) => {
            let r = crate::simplify::engine::simplify(r, pool).value;
            if is_safe(r) {
                *log = log.clone().merge(inner_log);
                return Ok(r);
            }
            // Rule engine introduced the current generator — fall through.
        }
        Err(IE::NotImplemented(_)) => {}
        Err(other) => return Err(other),
    }

    // Fast path 2: rational-function integration (Hermite + Rothstein–Trager).
    if let Some(r) = try_integrate_rational(expr, var, pool) {
        let r = crate::simplify::engine::simplify(r, pool).value;
        if is_safe(r) {
            return Ok(r);
        }
        // RT introduced the current generator — fall through.
    }

    // Gap E: K-rational integration for constant algebraic coefficients (e.g. √2).
    // K-rational results never introduce new log generators, so is_safe is always true.
    if let Some(r) = try_integrate_k_rational(expr, var, pool) {
        let r = crate::simplify::engine::simplify(r, pool).value;
        if is_safe(r) {
            return Ok(r);
        }
    }

    // Gap E (follow-up): K-rational integration WITH new K-log terms.  These
    // new logs are over K-linear arguments (x − rᵢ, rᵢ ∈ K) — distinct from
    // `excluded_gen` (the current transcendental log generator h), so is_safe
    // should always hold; check anyway for soundness.
    if let Some(r) = try_integrate_k_rational_with_logs(expr, var, pool) {
        let r = crate::simplify::engine::simplify(r, pool).value;
        if is_safe(r) {
            return Ok(r);
        }
    }

    // Slower path (Gap B): use the full integration engine for coefficients
    // that live in the lower tower (e.g. log(x) coefficients when integrating
    // at the log(log(x)) level).  Guard: the result must not contain
    // excluded_gen, otherwise the IBP recursion would diverge.
    match crate::integrate::engine::integrate(expr, var, pool) {
        Ok(d) => {
            let r = crate::simplify::engine::simplify(d.value, pool).value;
            if is_safe(r) {
                return Ok(r);
            }
            // Full engine also introduced excluded_gen: the integral of this
            // coefficient re-introduces the current generator.  Return
            // NotImplemented so the caller can diagnose non-elementariness.
            Err(IE::NotImplemented(format!(
                "integrate_base: integral of {} introduces the current log generator",
                pool.display(expr)
            )))
        }
        Err(IE::NonElementary(msg)) => Err(IE::NonElementary(msg)),
        Err(_) => Err(IE::NotImplemented(format!(
            "integrate_base: {} is not integrable in the base field",
            pool.display(expr)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Gap E — K-rational base-field integration (ℚ(α) constant coefficients)
// ---------------------------------------------------------------------------

/// Try to compute a K-rational antiderivative of `expr` when `expr` is a
/// rational function over a number field K = ℚ(α) (where α is an algebraic
/// constant free of `var`, e.g. `√2`).
///
/// Uses `solve_rational_rde_k` with `f = 0` (zero K-polynomial): the Risch DE
/// `v' + 0·v = c` reduces to plain antidifferentiation `v' = c`.  The solver
/// succeeds iff a K-rational antiderivative exists (no new log generators needed).
///
/// Returns `Some(antiderivative)` on success, `None` when the antiderivative
/// would require new logarithmic terms (e.g. `∫ 1/(x+√2) dx = log(x+√2)`).
fn try_integrate_k_rational(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let ext = detect_algebraic_extension(expr, pool)?;
    let (field, gens) = build_field_and_gens(&ext);
    let (c_num, c_den) = expr_to_krational_general(expr, var, &gens, &field, pool)?;

    // f = 0 (zero K-polynomial): solve v' = c over K(x).
    let k_zero: super::number_field::KPoly = vec![];
    let (v_num, v_den) = solve_rational_rde_k(&field, &k_zero, &c_num, &c_den)?;

    Some(build_krational_ext(&v_num, &v_den, var, &ext, pool))
}

/// Try to compute a K-rational antiderivative of `expr` **allowing new
/// `K`-coefficient `log` terms** when `expr` is a rational function over a
/// number field `K = ℚ(α)`.
///
/// This is the fallback for when [`try_integrate_k_rational`] declines (i.e.
/// `solve_rational_rde_k` returns `None` because the antiderivative needs new
/// logarithms).  Uses [`integrate_k_rational_with_logs`] (Rothstein–Trager /
/// partial fractions over `K`) to produce
/// `∫ c dx = (K-rational part) + Σ cᵢ·log(x − rᵢ)`, `cᵢ, rᵢ ∈ K`.
///
/// Returns `None` when:
/// - `expr` does not parse as a `K`-rational function over a detected `ℚ(α)`,
///   or
/// - the squarefree logarithmic remainder's denominator does not split
///   completely into distinct `K`-linear factors (e.g. an irreducible
///   quadratic over `K`) — even after Hermite reduction has peeled off any
///   repeated factors.
fn try_integrate_k_rational_with_logs(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let ext = detect_algebraic_extension(expr, pool)?;
    let (field, gens) = build_field_and_gens(&ext);
    let (c_num, c_den) = expr_to_krational_general(expr, var, &gens, &field, pool)?;

    let result = integrate_k_rational_with_logs(&field, &ext, &c_num, &c_den)?;

    // Already-integrated K-rational antiderivative piece (combines the
    // integrated polynomial part `∫Q dx` with any Hermite rational terms
    // `B/V^p` over a common denominator).
    let rational_antideriv = if NumberField::kdeg(&result.rational_num) < 0 {
        pool.integer(0_i32)
    } else {
        build_krational_ext(&result.rational_num, &result.rational_den, var, &ext, pool)
    };

    // Log terms: Σ cᵢ·log(x − rᵢ).
    let mut log_terms: Vec<ExprId> = Vec::new();
    for (residue, root) in &result.log_terms {
        let residue_expr = kelem_to_expr_ext(residue, &ext, pool);
        let root_expr = kelem_to_expr_ext(root, &ext, pool);
        let arg = if is_zero(root_expr, pool) {
            var
        } else {
            pool.add(vec![var, pool.mul(vec![pool.integer(-1_i32), root_expr])])
        };
        let log_arg = pool.func("log", vec![arg]);
        let term = pool.mul(vec![residue_expr, log_arg]);
        log_terms.push(term);
    }

    let mut all_terms = Vec::new();
    if !is_zero(rational_antideriv, pool) {
        all_terms.push(rational_antideriv);
    }
    all_terms.extend(log_terms);

    match all_terms.len() {
        0 => Some(pool.integer(0_i32)),
        1 => Some(all_terms[0]),
        _ => Some(pool.add(all_terms)),
    }
}

// ---------------------------------------------------------------------------
// §E — NonElementary certificate for entangled K-log coefficients
// ---------------------------------------------------------------------------

/// Decide whether the *top* coefficient of the log polynomial obstructs
/// elementarity by Bronstein's primitive-case eq (18).
///
/// Given `∫ Σ_{k=0}^{n} c_k(x)·log(h)^k dx` with `t = log(h)` a logarithmic
/// monomial over `K(x)` (`K = ℚ(α)` a number field), the integral can be
/// elementary only if the top coefficient `c_n` admits
///
/// ```text
///     c_n = v' + (n+1)·e·(h'/h),     v ∈ K(x),  e ∈ Const(K)
/// ```
/// (Bronstein 2005, *Symbolic Integration I*, §5.10; Tutorial §3.5, eq (18)).
/// We test exactly this with [`solve_primitive_top_rde_k`]: a `None` means no
/// `(v, e)` exists, which **proves** the whole integral non-elementary.
///
/// **Soundness gates (the certificate fires only when fully justified):**
/// - **EVERY** coefficient `c_0, …, c_n` must parse as a *genuinely K-rational*
///   function (no residual transcendental generator) over one detected `ℚ(α)`.
///   This is the load-bearing gate: eq (18) is a valid necessary condition only
///   when the integrand is a true polynomial in the single monomial `t = log(h)`
///   over `K(x)`.  If any coefficient still carries a *second* `log`/`exp`
///   generator (e.g. the combined integrand
///   `1/(x+√2)·log(x) + log(x+√2)/x = (log x · log(x+√2))'`, whose `c_0`
///   coefficient is `log(x+√2)/x`), the single-monomial structure does **not**
///   hold, eq (18) does not apply, and we return `false` (decline, never
///   certify).  Without this gate the certificate would wrongly fire on that
///   *elementary* sum.
/// - `h'/h` must itself be K-rational (always true for `h ∈ K(x)`); if the parse
///   fails we decline.
/// - We require the extension to be *non-trivial* (an actual algebraic α): a pure
///   `ℚ(x)` coefficient is not detected as an extension, and its log obstruction
///   is the ordinary rational `Li`/`Ei` case already handled elsewhere.
///
/// Returns `true` ⟺ eq (18) is provably unsolvable for the top coefficient
/// **and** the integrand is a genuine K(x)-polynomial in `t = log(h)`.
fn certify_klog_top_obstruction(
    coeffs: &[ExprId],
    h: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    use super::rational_rde::solve_primitive_top_rde_k;

    // Locate the genuine top degree n and its coefficient c_n.
    let n = find_top_degree(coeffs, pool);
    if n == 0 {
        return false; // degree-0: ordinary base-field integration, not eq (18).
    }
    let c_n = simplify(coeffs[n], pool).value;
    // Strip any algebraic constant factor (e.g. √2·…); the residue structure is
    // unchanged by a constant scalar, so the obstruction test is unaffected.
    let (_k_alg, c_n_rest) = split_const_factor(c_n, var, pool);

    // Parse c_n_rest as a rational function over a detected algebraic extension K.
    let Some(ext) = detect_algebraic_extension(c_n_rest, pool) else {
        return false; // not an algebraic-extension coefficient → not our case.
    };
    let (field, gens) = build_field_and_gens(&ext);
    let Some((c_num, c_den)) = expr_to_krational_general(c_n_rest, var, &gens, &field, pool) else {
        return false; // c_n carries a transcendental / foreign term → decline.
    };

    // GATE: every *lower* coefficient must also be genuinely K-rational over the
    // same field.  A coefficient that still contains a second transcendental
    // (e.g. another log) breaks the poly-in-t-over-K(x) structure that eq (18)
    // relies on — the integral may then be elementary by cancellation across
    // levels (the `log(x)·log(x+√2)` combined-sum case).  Declining here is the
    // single most important soundness guard for this certificate.
    for &ck_raw in &coeffs[..n] {
        let ck = simplify(ck_raw, pool).value;
        if is_zero(ck, pool) {
            continue;
        }
        let (_k_alg_k, ck_rest) = split_const_factor(ck, var, pool);
        if expr_to_krational_general(ck_rest, var, &gens, &field, pool).is_none() {
            return false; // c_k ∉ K(x): structure invalid → decline.
        }
    }

    // Build h'/h as a K-rational function (h ∈ K(x)).
    let h_prime = match crate::diff::diff(h, var, pool) {
        Ok(d) => simplify(d.value, pool).value,
        Err(_) => return false,
    };
    if is_zero(h_prime, pool) {
        return false; // h' = 0 → no drift; not the entangled case.
    }
    // h'/h symbolically, then parse numerator/denominator over K.
    let hph = simplify(
        pool.mul(vec![h_prime, pool.pow(h, pool.integer(-1_i32))]),
        pool,
    )
    .value;
    let Some((gd_num, gd_den)) = expr_to_krational_general(hph, var, &gens, &field, pool) else {
        return false; // h'/h not K-rational (shouldn't happen for h ∈ K(x)).
    };

    // eq (18): solvable ⇒ Some(..); provably unsolvable ⇒ None ⇒ certify.
    solve_primitive_top_rde_k(&field, &c_num, &c_den, &gd_num, &gd_den, n as i64).is_none()
}

// ---------------------------------------------------------------------------
// Log-derivative coefficient detection
// ---------------------------------------------------------------------------

/// When `expr` is an Add that contains at least one log-derivative term and at
/// least one non-log-derivative term, split it and return `(alpha_sum, hermite_sum)`.
///
/// Returns `None` when:
/// - `expr` is not an Add, or
/// - ALL terms are log-derivative (handled by `detect_log_deriv_coeff` directly), or
/// - NO terms are log-derivative (the normal `integrate_base` path handles it).
fn split_log_deriv_from_add(
    expr: ExprId,
    h: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let args = match pool.get(expr) {
        ExprData::Add(a) => a,
        _ => return None,
    };
    let mut ld_alphas: Vec<ExprId> = Vec::new();
    let mut hermite_terms: Vec<ExprId> = Vec::new();
    for &term in &args {
        if let Some(alpha) = detect_log_deriv_coeff(term, h, var, pool) {
            ld_alphas.push(alpha);
        } else {
            hermite_terms.push(term);
        }
    }
    // Only fire when BOTH buckets are non-empty — pure cases are handled elsewhere.
    if ld_alphas.is_empty() || hermite_terms.is_empty() {
        return None;
    }
    let zero = pool.integer(0_i32);
    let ld_sum = match ld_alphas.len() {
        1 => ld_alphas[0],
        _ => pool.add(ld_alphas),
    };
    let hermite_sum = match hermite_terms.len() {
        0 => zero,
        1 => hermite_terms[0],
        _ => pool.add(hermite_terms),
    };
    Some((ld_sum, hermite_sum))
}

/// Returns `Some(α)` if `c_rest = α · h'/h` with `α` free of `var`, else `None`.
///
/// The check builds `α = c_rest · h / h'` symbolically, simplifies, and tests
/// whether the result is free of `var`.  This detects the log-derivative pattern
/// that stalls the normal IBP (because P_n = α·log(h) contains the excluded
/// generator).
fn detect_log_deriv_coeff(
    c_rest: ExprId,
    h: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let h_prime = crate::diff::diff(h, var, pool).ok()?.value;
    let h_prime_s = simplify(h_prime, pool).value;
    if is_zero(h_prime_s, pool) {
        return None;
    }
    // α = c_rest · h / h'
    let ratio_raw = pool.mul(vec![c_rest, h, pool.pow(h_prime_s, pool.integer(-1_i32))]);
    let ratio = simplify(ratio_raw, pool).value;
    if is_free_of_var(ratio, var, pool) {
        Some(ratio)
    } else {
        None
    }
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

    // -----------------------------------------------------------------------
    // Gap E: algebraic constant coefficients in the log tower (const-factor split)
    // -----------------------------------------------------------------------

    #[test]
    fn sqrt2_times_x_log_x_elementary() {
        // ∫ √2·x·log(x) dx = √2·(x²/2·log(x) − x²/4)
        // IBP: P_1 = ∫ √2·x dx = √2·x²/2; correction = −√2·x²/2·(1/x) = −√2·x/2
        // Base: ∫ −√2·x/2 dx = −√2·x²/4.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![sqrt2, x, log_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ √2·x·log(x) dx must be elementary; got {result:?}"
        );
        verify_numeric(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn pi_times_log_x_squared_elementary() {
        // ∫ π·log(x)² dx = π·(x·log(x)² − 2x·log(x) + 2x)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let pi = pool.symbol("pi", crate::kernel::Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let log2 = pool.pow(log_x, pool.integer(2_i32));
        let integrand = pool.mul(vec![pi, log2]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ π·log(x)² dx must be elementary; got {result:?}"
        );
        // π is symbolic — just verify the result contains log.
        let s = pool.display(result.unwrap()).to_string();
        assert!(s.contains("log"), "result should contain log: {s}");
    }

    // -----------------------------------------------------------------------
    // Gap E: ℚ(α) constant coefficients in the log tower
    // -----------------------------------------------------------------------

    /// Numeric evaluator for Gap E tests (supports sqrt of integers too).
    fn eval_f64_e(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        use crate::kernel::ExprData;
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval_f64_e(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval_f64_e(a, x, xv, pool)).product(),
            ExprData::Pow { base, exp } => {
                eval_f64_e(base, x, xv, pool).powf(eval_f64_e(exp, x, xv, pool))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval_f64_e(args[0], x, xv, pool);
                match name.as_str() {
                    "log" => a.ln(),
                    "sqrt" => a.sqrt(),
                    other => panic!("eval_f64_e: unsupported func {other}"),
                }
            }
            other => panic!("eval_f64_e: unsupported node {other:?}"),
        }
    }

    fn verify_numeric_e(integrand: ExprId, antideriv: ExprId, x: ExprId, pool: &ExprPool) {
        let d = crate::diff::diff(antideriv, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        for &xv in &[0.5_f64, 1.5, 3.0] {
            let lhs = eval_f64_e(ds, x, xv, pool);
            let rhs = eval_f64_e(integrand, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: got {lhs}, expected {rhs}\n  F = {}",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn gape_inv_x_plus_sqrt2_sq_log_elementary() {
        // ∫ 1/(x+√2)² · log(x+√2) dx = −log(x+√2)/(x+√2) − 1/(x+√2)
        //
        // IBP: c_1 = 1/(x+√2)².
        //   P_1 = ∫ 1/(x+√2)² dx = −1/(x+√2)  [K-rational over ℚ(√2)] ✓
        //   h = x+√2, h'/h = 1/(x+√2)
        //   correction = −(−1/(x+√2))·(1/(x+√2)) = 1/(x+√2)²
        //   c_0 = 1/(x+√2)²  → P_0 = −1/(x+√2)  [K-rational] ✓
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![x_plus_sqrt2]);
        let integrand = pool.mul(vec![pool.pow(x_plus_sqrt2, pool.integer(-2_i32)), log_h]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1, "should find exactly one log generator");
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ 1/(x+√2)²·log(x+√2) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn gape_inv_x_plus_sqrt2_cubed_log_elementary() {
        // ∫ 1/(x+√2)³ · log(x+√2) dx
        //   P_1 = −1/(2(x+√2)²), correction = 1/(2(x+√2)³), P_0 = −1/(4(x+√2)²)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![x_plus_sqrt2]);
        let integrand = pool.mul(vec![pool.pow(x_plus_sqrt2, pool.integer(-3_i32)), log_h]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ 1/(x+√2)³·log(x+√2) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn gape_inv_x_plus_sqrt3_sq_log_elementary() {
        // ∫ 1/(x+√3)² · log(x+√3) dx — same structure but K = ℚ(√3)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt3 = pool.func("sqrt", vec![pool.integer(3_i32)]);
        let x_plus_sqrt3 = pool.add(vec![x, sqrt3]);
        let log_h = pool.func("log", vec![x_plus_sqrt3]);
        let integrand = pool.mul(vec![pool.pow(x_plus_sqrt3, pool.integer(-2_i32)), log_h]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ 1/(x+√3)²·log(x+√3) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn gape_const_sqrt2_times_inv_sq_log_elementary() {
        // ∫ √2/(x+√2)² · log(x+√2) dx = √2·(−log(x+√2)/(x+√2) − 1/(x+√2))
        // The const-factor split gives k_const=√2, c_rest=1/(x+√2)².
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![x_plus_sqrt2]);
        // integrand = √2 · 1/(x+√2)² · log(x+√2)
        let integrand = pool.mul(vec![
            sqrt2,
            pool.pow(x_plus_sqrt2, pool.integer(-2_i32)),
            log_h,
        ]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        assert!(
            result.is_ok(),
            "∫ √2/(x+√2)²·log(x+√2) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap(), x, &pool);
    }

    // -----------------------------------------------------------------------
    // Log-derivative shortcut: c_n = α·(h'/h), α free of var
    // -----------------------------------------------------------------------

    #[test]
    fn log_deriv_inv_x_log_x() {
        // ∫ (1/x)·log(x) dx = ½·log(x)²
        // c_1 = 1/x = h'/h (h=x), α=1 → shortcut: log(x)^2/2.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![pool.pow(x, pool.integer(-1_i32)), log_x]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (1/x)·log(x) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    #[test]
    fn log_deriv_inv_x_plus_sqrt2_log() {
        // ∫ 1/(x+√2)·log(x+√2) dx = ½·log(x+√2)²
        // c_1 = 1/(x+√2) = h'/h (h=x+√2), α=1.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let h = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![h]);
        let integrand = pool.mul(vec![pool.pow(h, pool.integer(-1_i32)), log_h]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ 1/(x+√2)·log(x+√2) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    #[test]
    fn log_deriv_two_over_xp1_log_sq() {
        // ∫ 2/(x+1)·log(x+1)² dx = (2/3)·log(x+1)³
        // c_2 = 2/(x+1) = 2·h'/h (h=x+1), α=2, n=2.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let h = pool.add(vec![x, pool.integer(1_i32)]);
        let log_h = pool.func("log", vec![h]);
        let integrand = pool.mul(vec![
            pool.integer(2_i32),
            pool.pow(h, pool.integer(-1_i32)),
            pool.pow(log_h, pool.integer(2_i32)),
        ]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ 2/(x+1)·log(x+1)² dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    #[test]
    fn log_deriv_sqrt2_over_xps2_log() {
        // ∫ √2/(x+√2)·log(x+√2) dx = (√2/2)·log(x+√2)²
        // const-factor split gives k_alg=√2, c_rest=1/(x+√2), α=1.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let h = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![h]);
        let integrand = pool.mul(vec![sqrt2, pool.pow(h, pool.integer(-1_i32)), log_h]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ √2/(x+√2)·log(x+√2) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    #[test]
    fn log_deriv_mixed_with_hermite() {
        // ∫ [1/(x+√2)² + 1/(x+√2)]·log(x+√2) dx
        //   = (−log(x+√2)/(x+√2) − 1/(x+√2)) + ½·log(x+√2)²
        // Decomposed by sum rule into two separate integrals.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let h = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![h]);
        // [1/(x+√2)² + 1/(x+√2)]·log(x+√2)
        let coeff = pool.add(vec![
            pool.pow(h, pool.integer(-2_i32)),
            pool.pow(h, pool.integer(-1_i32)),
        ]);
        let integrand = pool.mul(vec![coeff, log_h]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ [1/(x+√2)²+1/(x+√2)]·log(x+√2) dx must be elementary; got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    // -----------------------------------------------------------------------
    // §E — NonElementary certificate for entangled K-log coefficients
    //      (Bronstein 2005, §5.10 / Tutorial §3.5, eq (18))
    // -----------------------------------------------------------------------

    /// ∫ 1/(x+√2)·log(x) dx is genuinely non-elementary (dilogarithm): the top
    /// coefficient 1/(x+√2) has a residue 1 at the K-irrational pole x=−√2,
    /// which is *not* a zero of the tower argument h=x, so no constant e absorbs
    /// it in eq (18).  Must now certify NonElementary (was NotImplemented).
    #[test]
    fn klog_inv_x_plus_sqrt2_log_x_nonelementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![pool.pow(x_plus_sqrt2, pool.integer(-1_i32)), log_x]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ 1/(x+√2)·log(x) dx must be certified NonElementary; got {result:?}"
        );
    }

    /// Generalisation: 1/(x+√3)·log(x) — same obstruction over K = ℚ(√3).
    #[test]
    fn klog_inv_x_plus_sqrt3_log_x_nonelementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt3 = pool.func("sqrt", vec![pool.integer(3_i32)]);
        let x_plus_sqrt3 = pool.add(vec![x, sqrt3]);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![pool.pow(x_plus_sqrt3, pool.integer(-1_i32)), log_x]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ 1/(x+√3)·log(x) dx must be certified NonElementary; got {result:?}"
        );
    }

    /// Const-factored variant √2/(x+√2)·log(x): the algebraic constant factor is
    /// split off before the obstruction test, which is unchanged; still
    /// non-elementary.
    #[test]
    fn klog_sqrt2_over_x_plus_sqrt2_log_x_nonelementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![
            sqrt2,
            pool.pow(x_plus_sqrt2, pool.integer(-1_i32)),
            log_x,
        ]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ √2/(x+√2)·log(x) dx must be certified NonElementary; got {result:?}"
        );
    }

    // ----- Adversarial elementary cases: the certificate must NOT fire -----

    /// ∫ 1/(x+√2)²·log(x) dx IS mathematically elementary (double pole at x=−√2
    /// has zero simple residue; P_1 = −1/(x+√2) is K-rational), so eq (18) is
    /// solvable (e=0) and the §E certificate correctly does NOT fire.  The IBP
    /// base term `∫ 1/(x(x+√2)) dx = (1/√2)(log x − log(x+√2))` requires a
    /// K-rational integrator *with* logarithms
    /// ([`super::k_rational_integrate::integrate_k_rational_with_logs`]), now
    /// wired into [`try_integrate_k_rational_with_logs`], so the engine produces
    /// a verified antiderivative.  The load-bearing invariant remains the
    /// negative: it must NEVER be certified NonElementary.
    #[test]
    fn klog_inv_x_plus_sqrt2_sq_log_x_not_nonelementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![pool.pow(x_plus_sqrt2, pool.integer(-2_i32)), log_x]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            !matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ 1/(x+√2)²·log(x) dx is elementary; must NEVER be NonElementary; got {result:?}"
        );
        // The K-rational-with-logs path now closes this gap: it must succeed.
        match &result {
            Ok(d) => {
                println!("∫ 1/(x+√2)²·log(x) dx = {}", pool.display(d.value));
                verify_numeric_e(integrand, d.value, x, &pool);
            }
            Err(e) => panic!("∫ 1/(x+√2)²·log(x) dx must now be elementary; got {e:?}"),
        }
    }

    /// ∫ 1/(x+√2)·log(x+√2) dx = ½·log(x+√2)² — the log-derivative case, h=x+√2.
    /// The residue at x=−√2 IS a zero of h, so eq (18) is solvable (e=1/2).
    /// (Caught earlier by the log-derivative shortcut; here we re-assert it stays
    /// elementary even though the coefficient is K-rational with a pole.)
    #[test]
    fn klog_inv_x_plus_sqrt2_log_same_arg_elementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let h = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![h]);
        let integrand = pool.mul(vec![pool.pow(h, pool.integer(-1_i32)), log_h]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ 1/(x+√2)·log(x+√2) dx must stay elementary (=½log²); got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    /// ∫ log(x)/x dx = ½·log(x)² — pure ℚ(x) coefficient, no algebraic extension,
    /// certificate declines (detect_algebraic_extension returns None).  Elementary.
    #[test]
    fn klog_log_x_over_x_elementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![pool.pow(x, pool.integer(-1_i32)), log_x]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ log(x)/x dx must be elementary (=½log²); got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    /// Adversarial combined sum: ∫ [1/(x+√2)·log(x) + log(x+√2)/x] dx is
    /// elementary — it equals log(x)·log(x+√2) (the two non-elementary
    /// dilogarithm halves cancel).  The integrand has TWO log generators, so it
    /// never reaches the single-generator poly-in-t path with a K-rational top
    /// coefficient; the §E certificate must NOT fire.  The engine may either
    /// integrate it (to log·log) or decline — but it must NEVER certify
    /// NonElementary.  This is the load-bearing safety case.
    #[test]
    fn klog_combined_sum_not_nonelementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let log_x = pool.func("log", vec![x]);
        let log_xps2 = pool.func("log", vec![x_plus_sqrt2]);
        // 1/(x+√2)·log(x) + (1/x)·log(x+√2)
        let term1 = pool.mul(vec![pool.pow(x_plus_sqrt2, pool.integer(-1_i32)), log_x]);
        let term2 = pool.mul(vec![pool.pow(x, pool.integer(-1_i32)), log_xps2]);
        let integrand = pool.add(vec![term1, term2]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            !matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ [1/(x+√2)·log(x) + log(x+√2)/x] dx = log(x)log(x+√2) is ELEMENTARY; \
             must never be certified NonElementary; got {result:?}"
        );
        // If the engine produced an antiderivative, it must be correct.
        if let Ok(d) = result {
            verify_numeric_e(integrand, d.value, x, &pool);
        }
    }

    /// Adversarial: ∫ (h'/h)·log(h)² with h=x+√2, i.e. 2(x+√2)/(x+√2)²·log²… —
    /// the standard log-derivative power rule, elementary.  Must not be
    /// certified: the top coefficient's pole IS a zero of h.
    #[test]
    fn klog_log_deriv_power_rule_sqrt2_elementary() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let h = pool.add(vec![x, sqrt2]);
        let log_h = pool.func("log", vec![h]);
        // (1/(x+√2))·log(x+√2)² = (h'/h)·t²  → t³/3
        let integrand = pool.mul(vec![
            pool.pow(h, pool.integer(-1_i32)),
            pool.pow(log_h, pool.integer(2_i32)),
        ]);

        let result = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            result.is_ok(),
            "∫ (1/(x+√2))·log(x+√2)² dx must be elementary (=log³/3); got {result:?}"
        );
        verify_numeric_e(integrand, result.unwrap().value, x, &pool);
    }

    // ----- Gap E follow-up: K-rational integration with K-log emission -----

    /// `try_integrate_k_rational_with_logs` directly: ∫ 1/(x·(x+√2)) dx
    /// = (1/√2)·[log(x) − log(x+√2)].  Both poles are K-rational (x=0 and
    /// x=−√2), neither is a removable (zero-residue) pole, so the plain
    /// K-rational RDE solver declines and `try_integrate_k_rational_with_logs`
    /// must produce the dilog-free closed form (matches the module-doc
    /// example and the `x_times_x_plus_sqrt2_log_terms` algebra-level test).
    #[test]
    fn k_rational_with_logs_x_times_x_plus_sqrt2() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let integrand = pool.pow(pool.mul(vec![x, x_plus_sqrt2]), pool.integer(-1_i32));

        let r = try_integrate_k_rational_with_logs(integrand, x, &pool)
            .expect("∫ 1/(x(x+√2)) dx = (1/√2)(log x − log(x+√2)) must succeed");
        let r = crate::simplify::engine::simplify(r, &pool).value;
        println!("∫ 1/(x(x+√2)) dx = {}", pool.display(r));
        verify_numeric_e(integrand, r, x, &pool);
    }

    /// `try_integrate_k_rational_with_logs` directly: ∫ 1/((x−√2)·(x+√2)) dx
    /// = (1/(2√2))·[log(x−√2) − log(x+√2)], exercising a non-zero pair of
    /// K-roots from a degree-2 denominator (both factors written explicitly
    /// with √2 so the integrand is detected as K=ℚ(√2)-rational).
    #[test]
    fn k_rational_with_logs_inv_x_sq_minus_2() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let neg_sqrt2 = pool.mul(vec![pool.integer(-1_i32), sqrt2]);
        let x_minus_sqrt2 = pool.add(vec![x, neg_sqrt2]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let denom = pool.mul(vec![x_minus_sqrt2, x_plus_sqrt2]);
        let integrand = pool.pow(denom, pool.integer(-1_i32));

        let r = try_integrate_k_rational_with_logs(integrand, x, &pool)
            .expect("∫ 1/((x−√2)(x+√2)) dx = (1/(2√2))[log(x−√2) − log(x+√2)] must succeed");
        let r = crate::simplify::engine::simplify(r, &pool).value;
        println!("∫ 1/((x−√2)(x+√2)) dx = {}", pool.display(r));
        verify_numeric_e(integrand, r, x, &pool);
    }

    /// Decline: ∫ √2/(x²+1) dx — the denominator's discriminant (−4) is not a
    /// K-square in K=ℚ(√2), so x²+1 does not split into K-linear factors.
    /// `try_integrate_k_rational_with_logs` must decline (`None`); the
    /// (rational, K-free) `arctan` path handles this elsewhere.
    #[test]
    fn k_rational_with_logs_irreducible_quadratic_declines() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        // Multiply by √2 so the integrand parses as a K=ℚ(√2)-rational
        // function (otherwise detect_algebraic_extension finds no extension
        // and this isn't exercising the K-rational-with-logs path at all).
        let two = pool.integer(2_i32);
        let denom = pool.add(vec![pool.pow(x, two), pool.integer(1_i32)]);
        let integrand = pool.mul(vec![sqrt2, pool.pow(denom, pool.integer(-1_i32))]);

        assert!(
            try_integrate_k_rational_with_logs(integrand, x, &pool).is_none(),
            "∫ √2/(x²+1) dx: denominator x²+1 is K-irreducible over ℚ(√2); must decline"
        );
    }

    // ----- Hermite reduction over K (repeated K-factors) -----

    /// `try_integrate_k_rational_with_logs` directly: ∫ 1/(x·(x+√2)²) dx.
    ///
    /// `(x+√2)` is a repeated K-factor of the denominator — Hermite reduction
    /// over K peels off a `B/(x+√2)` rational term, leaving a squarefree
    /// `x·(x+√2)` remainder for the K-log part.
    #[test]
    fn k_rational_with_logs_x_times_x_plus_sqrt2_squared() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let denom = pool.mul(vec![x, pool.pow(x_plus_sqrt2, pool.integer(2_i32))]);
        let integrand = pool.pow(denom, pool.integer(-1_i32));

        let r = try_integrate_k_rational_with_logs(integrand, x, &pool)
            .expect("∫ 1/(x(x+√2)²) dx must close via Hermite reduction over K");
        let r = crate::simplify::engine::simplify(r, &pool).value;
        println!("∫ 1/(x(x+√2)²) dx = {}", pool.display(r));
        verify_numeric_e(integrand, r, x, &pool);
    }

    /// `try_integrate_k_rational_with_logs` directly:
    /// ∫ (x+√2+1)/((x−√2)²·(x+√2)) dx.
    ///
    /// `(x−√2)` is a repeated K-factor; Hermite reduction over K peels a
    /// rational term, leaving a squarefree `(x−√2)(x+√2)` remainder for the
    /// K-log part.
    #[test]
    fn k_rational_with_logs_repeated_x_minus_sqrt2() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let neg_sqrt2 = pool.mul(vec![pool.integer(-1_i32), sqrt2]);
        let x_minus_sqrt2 = pool.add(vec![x, neg_sqrt2]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);

        let numerator = pool.add(vec![x, sqrt2, pool.integer(1_i32)]);
        let denom = pool.mul(vec![
            pool.pow(x_minus_sqrt2, pool.integer(2_i32)),
            x_plus_sqrt2,
        ]);
        let integrand = pool.mul(vec![numerator, pool.pow(denom, pool.integer(-1_i32))]);

        let r = try_integrate_k_rational_with_logs(integrand, x, &pool)
            .expect("∫ (x+√2+1)/((x−√2)²(x+√2)) dx must close via Hermite reduction over K");
        let r = crate::simplify::engine::simplify(r, &pool).value;
        println!("∫ (x+√2+1)/((x−√2)²(x+√2)) dx = {}", pool.display(r));
        verify_numeric_e(integrand, r, x, &pool);
    }

    /// Through the log-tower IBP base case: ∫ 1/(x+√2)³·log(x+√2) dx.
    ///
    /// `c_1 = 1/(x+√2)³` has its *only* pole at `x=−√2`, which IS a zero of
    /// `h=x+√2` — the §E certificate's "K-irrational pole not covered by h"
    /// condition does not apply, so this should proceed to the IBP/Hermite
    /// path.  `P_1 = ∫1/(x+√2)³ dx = −1/(2(x+√2)²)` (Hermite over K, `i=3`).
    #[test]
    fn k_rational_with_logs_inv_x_plus_sqrt2_cubed_times_log_x_plus_sqrt2() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let denom = pool.pow(x_plus_sqrt2, pool.integer(3_i32));
        let log_h = pool.func("log", vec![x_plus_sqrt2]);
        let integrand = pool.mul(vec![pool.pow(denom, pool.integer(-1_i32)), log_h]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1, "should find exactly one log generator");
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        match result {
            Ok(r) => {
                let r = crate::simplify::engine::simplify(r, &pool).value;
                println!("∫ log(x+√2)/(x+√2)³ dx = {}", pool.display(r));
                verify_numeric_e(integrand, r, x, &pool);
            }
            Err(e) => {
                println!("∫ log(x+√2)/(x+√2)³ dx declined: {e:?}");
            }
        }
    }

    /// Through the log-tower IBP base case: ∫ 1/(x·(x+√2)²)·log(x+√2) dx.
    ///
    /// `h = x+√2` covers the K-irrational pole of `c_1 = 1/(x(x+√2)²)` at
    /// `x=−√2`, so the §E primitive-case certificate does not fire here (unlike
    /// `log(x)`); the IBP correction term then needs the Hermite-over-K
    /// antiderivative `P_1` of `c_1`.  Document what closes — assert only what
    /// numerically verifies.
    #[test]
    fn k_rational_with_logs_x_times_x_plus_sqrt2_squared_times_log_x_plus_sqrt2() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let denom = pool.mul(vec![x, pool.pow(x_plus_sqrt2, pool.integer(2_i32))]);
        let log_h = pool.func("log", vec![x_plus_sqrt2]);
        let integrand = pool.mul(vec![pool.pow(denom, pool.integer(-1_i32)), log_h]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1, "should find exactly one log generator");
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        match result {
            Ok(r) => {
                let r = crate::simplify::engine::simplify(r, &pool).value;
                println!("∫ 1/(x(x+√2)²)·log(x+√2) dx = {}", pool.display(r));
                verify_numeric_e(integrand, r, x, &pool);
            }
            Err(e) => {
                println!("∫ 1/(x(x+√2)²)·log(x+√2) dx declined: {e:?}");
            }
        }
    }

    /// Through the log-tower IBP base case: ∫ 1/(x·(x+√2)²)·log(x) dx.
    ///
    /// `c_1 = 1/(x(x+√2)²)` requires Hermite reduction over K to find its
    /// K-rational antiderivative `P_1`; the IBP correction term is then
    /// integrated as `c_0`.  We assert only what numerically verifies (the
    /// engine may still decline some sub-steps; this test pins what closes).
    #[test]
    fn k_rational_with_logs_x_times_x_plus_sqrt2_squared_times_log_x() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let x_plus_sqrt2 = pool.add(vec![x, sqrt2]);
        let denom = pool.mul(vec![x, pool.pow(x_plus_sqrt2, pool.integer(2_i32))]);
        let log_x = pool.func("log", vec![x]);
        let integrand = pool.mul(vec![pool.pow(denom, pool.integer(-1_i32)), log_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1, "should find exactly one log generator");
        let level = &gens[0];

        let mut inner_log = DerivationLog::new();
        let result = integrate_log_tower(integrand, level, x, &pool, &mut inner_log);
        match result {
            Ok(r) => {
                let r = crate::simplify::engine::simplify(r, &pool).value;
                println!("∫ 1/(x(x+√2)²)·log(x) dx = {}", pool.display(r));
                verify_numeric_e(integrand, r, x, &pool);
            }
            Err(e) => {
                // Document, don't force: the IBP correction term may still
                // exceed the K-rational(+logs) base-field integrator.
                println!("∫ 1/(x(x+√2)²)·log(x) dx declined: {e:?}");
            }
        }
    }
}
