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

use super::alg_field::{AlgElem, AlgExtension, RatFn};
use super::number_field::{KElem, KPoly, NumberField};
use super::poly_rde::{
    apply_const, contains_subexpr, degree, expr_to_qpoly, is_free_of_var, poly_add, poly_deriv,
    poly_mul, poly_one, poly_scale, poly_zero, qpoly_to_expr, rational_to_expr, solve_poly_rde,
    solve_poly_rde_k, split_const_factor, trim, QPoly,
};
use super::rational_rde::{
    expr_to_qrational, poly_gcd, poly_sub, solve_rational_rde, solve_rational_rde_generalized,
    solve_rational_rde_k,
};
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

    // Fast path: polynomial η' → existing polynomial/rational RDE solvers.
    let deta = match expr_to_qpoly(deta_expr, var, pool) {
        Some(p) => p,
        None => {
            // Gap F: η' is not a polynomial.  Try rational η' → use the
            // generalised RDE solver, or fall back to NotImplemented for
            // genuinely transcendental exponents.
            if let Some((deta_num, deta_den)) = expr_to_qrational(deta_expr, var, pool) {
                return integrate_exp_tower_rational_eta(
                    int_rational,
                    &exp_terms,
                    eta,
                    exp_gen,
                    deta_num,
                    deta_den,
                    var,
                    pool,
                    log,
                );
            }
            // Gap B (nested exp): η' is transcendental (e.g. η = exp(x), η' = exp(x)).
            // Try the v=1 special case: D(1) + k·η'·1 = k·η', so v=1 solves the
            // RDE iff c_rest == k·η'.  Handles ∫ exp(x)·exp(exp(x)) dx = exp(exp(x)).
            return try_transcendental_eta_v1(
                int_rational,
                &exp_terms,
                eta,
                exp_gen,
                deta_expr,
                var,
                pool,
                log,
            );
        }
    };

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
// Rational-exponent path: η ∈ ℚ(x) \ ℚ[x]  (Risch Gap F)
// ---------------------------------------------------------------------------

/// Integrate each `c_k(x) · exp(η)^k` term using the generalised rational
/// RDE `v' + k·η'·v = c` when `η'(x)` is a rational function (not a
/// polynomial).  The pre-integrated rational part `int_rational` is passed
/// in directly (it was already computed in [`integrate_exp_tower`]).
///
/// Returns `Err(NonElementary)` when the generalised RDE certifies that no
/// rational solution exists.
#[allow(clippy::too_many_arguments)]
fn integrate_exp_tower_rational_eta(
    int_rational: ExprId,
    exp_terms: &[(ExprId, i64)],
    eta: ExprId,
    exp_gen: ExprId,
    deta_num: QPoly, // numerator of η'(x)
    deta_den: QPoly, // denominator of η'(x)
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let zero = pool.integer(0_i32);
    let mut result_terms: Vec<ExprId> = Vec::new();
    if !is_zero(int_rational, pool) {
        result_terms.push(int_rational);
    }

    for (c_expr, k) in exp_terms {
        let k = *k;
        let exp_k_eta = build_exp_k_eta(k, eta, exp_gen, pool);

        // Split off any var-free constant factor K so the RDE works on the
        // purely var-dependent remainder (same trick as in integrate_single_exp_term).
        let (k_const, c_rest) = split_const_factor(*c_expr, var, pool);

        // f = k · η' = k · (deta_num / deta_den).
        let k_rat = rug::Rational::from(k);
        let f_num = poly_scale(&deta_num, &k_rat);
        let f_den = deta_den.clone();

        let non_elementary = || {
            IntegrationError::NonElementary(format!(
                "the Risch DE v'(x) + {k}·η'(x)·v(x) = {}(x) has no rational solution; \
                 the integrand ∫ {} · exp(η)^{k} dx is not an elementary function \
                 (η = {})",
                pool.display(*c_expr),
                pool.display(*c_expr),
                pool.display(eta),
            ))
        };

        // Polynomial coefficient → polynomial-ansatz generalised RDE.
        if let Some(c_poly) = expr_to_qpoly(c_rest, var, pool) {
            match solve_rational_rde_generalized(&f_num, &f_den, &c_poly, &poly_one()) {
                Some((v_num, v_den)) => {
                    let v_expr = build_rational(&v_num, &v_den, var, pool);
                    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                    let result = apply_const(k_const, core, pool);
                    log.push(RewriteStep::simple(
                        "risch_exp_rde_rational_eta",
                        *c_expr,
                        result,
                    ));
                    result_terms.push(result);
                    continue;
                }
                None => return Err(non_elementary()),
            }
        }

        // Rational coefficient → rational-ansatz generalised RDE.
        if let Some((c_num, c_den)) = expr_to_qrational(c_rest, var, pool) {
            match solve_rational_rde_generalized(&f_num, &f_den, &c_num, &c_den) {
                Some((v_num, v_den)) => {
                    let v_expr = build_rational(&v_num, &v_den, var, pool);
                    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                    let result = apply_const(k_const, core, pool);
                    log.push(RewriteStep::simple(
                        "risch_exp_rde_rational_eta_rational_coeff",
                        *c_expr,
                        result,
                    ));
                    result_terms.push(result);
                    continue;
                }
                None => return Err(non_elementary()),
            }
        }

        return Err(IntegrationError::NotImplemented(format!(
            "coefficient {} of exp(η)^{k} is not a rational function in {}; \
             algebraic/mixed coefficients with rational exponents are not yet supported",
            pool.display(*c_expr),
            pool.display(var),
        )));
    }

    let raw = match result_terms.len() {
        0 => zero,
        1 => result_terms[0],
        _ => pool.add(result_terms),
    };
    let simplified = simplify(raw, pool);
    *log = log.clone().merge(simplified.log);
    log.push(RewriteStep::simple(
        "risch_exp_rational_eta",
        pool.integer(0_i32),
        simplified.value,
    ));
    Ok(simplified.value)
}

// ---------------------------------------------------------------------------
// Gap B (nested exp) — v=1 special case for transcendental η'
// ---------------------------------------------------------------------------

/// Handle `∫ c(x)·exp(kη) dx` when η'(x) is transcendental (e.g. η = exp(x))
/// by checking whether each coefficient satisfies `c_rest = k·η'` (after
/// splitting off any var-free constant factor and simplifying both sides).
///
/// The Risch DE is `v' + k·η'·v = c`.  When `c = k·η'`, the constant `v = 1`
/// is a solution: `D(1) + k·η'·1 = k·η' = c`.  The antiderivative is then
/// `1 · exp(kη) = exp(kη)`.
///
/// Returns `Err(NotImplemented)` for any term where the v=1 condition fails.
#[allow(clippy::too_many_arguments)]
fn try_transcendental_eta_v1(
    int_rational: ExprId,
    exp_terms: &[(ExprId, i64)],
    eta: ExprId,
    exp_gen: ExprId,
    deta_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let zero = pool.integer(0_i32);
    // Simplify η' once; all coefficient comparisons are done against this.
    let deta_simplified = simplify(deta_expr, pool).value;

    let mut result_terms: Vec<ExprId> = Vec::new();
    if !is_zero(int_rational, pool) {
        result_terms.push(int_rational);
    }

    for (c_expr, k) in exp_terms {
        let k = *k;
        let (k_const, c_rest) = split_const_factor(*c_expr, var, pool);
        let c_rest_simplified = simplify(c_rest, pool).value;

        // k·η' simplified.
        let k_deta_simplified = if k == 1 {
            deta_simplified
        } else {
            simplify(pool.mul(vec![pool.integer(k as i32), deta_expr]), pool).value
        };

        if c_rest_simplified != k_deta_simplified {
            // v=1 check failed.
            //
            // Hermite-reduction certificate: if c_rest is rational (not
            // polynomial) in θ_inner, the derivation D maps every simple pole
            // 1/(θ-α) to a double pole (α-Dα)/(θ-α)², which cannot be
            // cancelled by any rational v ∈ ℚ(x)(θ_inner) → NonElementary.
            if c_is_rational_in_theta(c_rest, deta_simplified, pool) {
                return Err(IntegrationError::NonElementary(format!(
                    "∫ {} · exp(kη) dx: coefficient is rational (not polynomial) \
                     in the inner exp generator; non-elementary by Hermite \
                     reduction / pole-order argument (Bronstein §6.2)",
                    pool.display(*c_expr),
                )));
            }

            // Try the lower-tower polynomial cascade: write c_rest as a
            // polynomial in θ_inner and solve the RDE level by level over ℚ(x).
            let exp_k_eta = build_exp_k_eta(k, eta, exp_gen, pool);
            match lower_tower_poly_cascade(
                c_rest,
                k,
                deta_simplified,
                exp_k_eta,
                k_const,
                *c_expr,
                var,
                pool,
                log,
            ) {
                Some(Ok(r)) => {
                    result_terms.push(r);
                    continue;
                }
                Some(Err(e)) => return Err(e),
                None => {
                    return Err(IntegrationError::NotImplemented(format!(
                        "exponent derivative η'(x) = {} is transcendental and \
                         the lower-tower polynomial cascade did not apply for {}",
                        pool.display(deta_expr),
                        pool.display(*c_expr),
                    )))
                }
            }
        }

        // v = 1 is a solution: ∫ k·η'·exp(kη) dx = exp(kη).
        let exp_k_eta = build_exp_k_eta(k, eta, exp_gen, pool);
        let result = apply_const(k_const, exp_k_eta, pool);
        log.push(RewriteStep::simple("risch_exp_nested_v1", *c_expr, result));
        result_terms.push(result);
    }

    let raw = match result_terms.len() {
        0 => zero,
        1 => result_terms[0],
        _ => pool.add(result_terms),
    };
    let simplified = simplify(raw, pool);
    *log = log.clone().merge(simplified.log);
    log.push(RewriteStep::simple(
        "risch_exp_transcendental_eta",
        pool.integer(0_i32),
        simplified.value,
    ));
    Ok(simplified.value)
}

// ---------------------------------------------------------------------------
// Gap B — rational-in-θ NonElementary certification (Hermite reduction)
// ---------------------------------------------------------------------------

/// Returns `true` when `c_rest` is *rational* (not polynomial) in `theta_inner`.
///
/// For the Risch DE `D(v) + k·θ·v = c` with `θ = exp(x)` (so `D(θ) = θ`),
/// the derivation `D` maps any simple pole `1/(θ-α)` (α ∈ ℚ(x)) to a double
/// pole `(α-Dα)/(θ-α)²`.  Since `Dα ≠ α` for α ∈ ℚ(x) \ {0} (α would need
/// to satisfy `Dα = α`, i.e. α = C·exp(x), which is transcendental), the
/// double pole cannot be cancelled by any rational v.  Hence no rational
/// solution exists → the integral is non-elementary.
///
/// The check detects the three patterns that make c rational in θ_inner:
/// 1. `rational_part` from `decompose_wrt_exp` still contains θ_inner.
/// 2. Any exp-term has a negative power of θ_inner (e.g. `c · θ_inner^{-1}`).
/// 3. Any exp-term coefficient itself contains θ_inner (e.g. `(θ_inner+1)^{-1}`
///    as the coefficient of θ_inner^1).
fn c_is_rational_in_theta(c_rest: ExprId, theta_inner: ExprId, pool: &ExprPool) -> bool {
    use super::tower::decompose_wrt_exp;
    let (c0, exp_terms) = decompose_wrt_exp(c_rest, theta_inner, pool);
    if contains_subexpr(c0, theta_inner, pool) {
        return true;
    }
    for (coeff, j) in &exp_terms {
        if *j < 0 {
            return true;
        }
        if contains_subexpr(*coeff, theta_inner, pool) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Gap B (nested exp) — lower-tower polynomial cascade
// ---------------------------------------------------------------------------

/// Solve `D(v) + k·θ_inner·v = c` for `v ∈ ℚ(x)[θ_inner]` by a top-down
/// cascade when `c` is a polynomial in `θ_inner = deta_simplified` with
/// ℚ(x) coefficients.
///
/// **Algorithm** (Bronstein §5, specialised to θ_inner self-derivative):
///
/// Since `D(θ_inner^j) = j·θ_inner^j` (valid when θ_inner = exp(g) with g' = 1,
/// i.e. θ_inner = exp(x)), writing `v = Σ vⱼ·θ_inner^j` and `c = Σ cⱼ·θ_inner^j`
/// gives:
///
/// ```text
///   θ_inner^N  :  k·v_{N-1} = c_N              → v_{N-1} = c_N/k
///   θ_inner^n  :  (vₙ′+n·vₙ) + k·vₙ₋₁ = cₙ   → vₙ₋₁ = (cₙ−vₙ′−n·vₙ)/k
///   θ_inner^0  :  v₀′ = c₀                      (consistency)
/// ```
///
/// If `c ∈ ℚ(x)` (no θ_inner component) → `NonElementary`.
/// If the consistency check fails → `None` (caller falls through to `NotImplemented`).
/// On success → `Some(Ok(antiderivative))`.
#[allow(clippy::too_many_arguments)]
fn lower_tower_poly_cascade(
    c_rest: ExprId,
    k: i64,
    theta_inner: ExprId, // simplified η' (must be an exp-type expression)
    exp_k_eta: ExprId,
    k_const: ExprId,
    c_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<Result<ExprId, IntegrationError>> {
    use super::tower::decompose_wrt_exp;

    // θ_inner must be an exp expression so that D(θ_inner^j) = j·θ_inner^j.
    // (More precisely: θ_inner = exp(x) so D(exp(x)) = exp(x) = θ_inner.)
    // Check that θ_inner is exp(x) specifically (exponent derivative = self).
    let inner_eta = match pool.get(theta_inner) {
        crate::kernel::ExprData::Func { ref name, ref args }
            if name == "exp" && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None, // θ_inner is not an exp — cascade doesn't apply.
    };
    // Require the inner exponent's derivative to be θ_inner itself.
    // i.e., D(inner_eta) = θ_inner.  For inner_eta = x this is 1, but
    // we actually need D(exp(inner_eta)) = exp(inner_eta); just check it.
    let d_inner = match crate::diff::diff(theta_inner, var, pool) {
        Ok(d) => simplify(d.value, pool).value,
        Err(_) => return None,
    };
    if d_inner != theta_inner {
        // D(exp(inner_eta)) ≠ exp(inner_eta) → the cascade formula is wrong.
        return None;
    }
    let _ = inner_eta;

    // Decompose c_rest = c₀ + Σⱼ cⱼ·θ_inner^j.
    let (c0, exp_terms) = decompose_wrt_exp(c_rest, theta_inner, pool);

    if exp_terms.is_empty() {
        // c ∈ ℚ(x): no θ_inner factor.  The degree bound argument shows no
        // polynomial solution in θ_inner exists → non-elementary.
        return Some(Err(IntegrationError::NonElementary(format!(
            "∫ {} · exp(kη) dx: coefficient has no inner-tower exp factor; \
             non-elementary by degree bound (Bronstein §5)",
            pool.display(c_expr),
        ))));
    }

    // Find the maximum degree in θ_inner.
    let cap_n = exp_terms.iter().map(|(_, j)| *j).max().unwrap_or(0);
    if cap_n <= 0 {
        return None; // Safety: shouldn't happen, but don't crash.
    }
    let cap_n = cap_n as usize;

    // Build coefficient array c[0..=cap_n].
    let zero = pool.integer(0_i32);
    let mut c_coeffs: Vec<ExprId> = vec![zero; cap_n + 1];
    c_coeffs[0] = c0;
    for (coeff, j) in &exp_terms {
        let j = *j;
        if j >= 1 && (j as usize) <= cap_n {
            let old = c_coeffs[j as usize];
            let combined = if is_zero(old, pool) {
                *coeff
            } else {
                pool.add(vec![old, *coeff])
            };
            c_coeffs[j as usize] = simplify(combined, pool).value;
        }
    }

    // v has degree cap_n − 1 in θ_inner.
    let mut v_coeffs: Vec<ExprId> = vec![zero; cap_n]; // v[0..cap_n-1]

    // Top: v[cap_n-1] = c[cap_n] / k.
    let c_top = simplify(c_coeffs[cap_n], pool).value;
    v_coeffs[cap_n - 1] = if k == 1 {
        c_top
    } else {
        let k_inv = pool.pow(pool.integer(k as i32), pool.integer(-1_i32));
        simplify(pool.mul(vec![c_top, k_inv]), pool).value
    };

    // Cascade downwards: v[j-1] = (c[j] - D(v[j]) - j·v[j]) / k.
    for j in (1..cap_n).rev() {
        let vj = v_coeffs[j];
        let dvj = match crate::diff::diff(vj, var, pool) {
            Ok(d) => simplify(d.value, pool).value,
            Err(_) => return None,
        };
        let j_vj = simplify(pool.mul(vec![pool.integer(j as i32), vj]), pool).value;
        let cj = simplify(c_coeffs[j], pool).value;
        // num = c[j] - D(v[j]) - j·v[j]
        let neg1 = pool.integer(-1_i32);
        let num = simplify(
            pool.add(vec![
                cj,
                pool.mul(vec![neg1, dvj]),
                pool.mul(vec![neg1, j_vj]),
            ]),
            pool,
        )
        .value;
        v_coeffs[j - 1] = if k == 1 {
            num
        } else {
            let k_inv = pool.pow(pool.integer(k as i32), pool.integer(-1_i32));
            simplify(pool.mul(vec![num, k_inv]), pool).value
        };
    }

    // Consistency check: D(v[0]) = c[0].
    let dv0 = match crate::diff::diff(v_coeffs[0], var, pool) {
        Ok(d) => simplify(d.value, pool).value,
        Err(_) => return None,
    };
    let residual = simplify(
        pool.add(vec![dv0, pool.mul(vec![pool.integer(-1_i32), c_coeffs[0]])]),
        pool,
    )
    .value;
    if !is_zero(residual, pool) {
        // Polynomial cascade fails: by the denominator-bound theorem for the
        // hyperexponential Risch DE (Bronstein §6.2), when c is polynomial in
        // θ_inner the denominator of v must also be polynomial (i.e., v ∈ ℚ(x)[θ_inner]).
        // Since no polynomial solution exists, there is no rational solution either
        // → the integral is certified non-elementary.
        return Some(Err(IntegrationError::NonElementary(format!(
            "∫ {} · exp(kη) dx: lower-tower cascade consistency check failed; \
             non-elementary by denominator bound (Bronstein §6.2)",
            pool.display(c_expr),
        ))));
    }

    // Build v = Σ v[j] · θ_inner^j.
    let mut v_terms: Vec<ExprId> = Vec::new();
    for (j, &vj) in v_coeffs.iter().enumerate() {
        let vj_s = simplify(vj, pool).value;
        if is_zero(vj_s, pool) {
            continue;
        }
        let theta_j = match j {
            0 => vj_s,
            1 => {
                if is_one(vj_s, pool) {
                    theta_inner
                } else {
                    pool.mul(vec![vj_s, theta_inner])
                }
            }
            _ => {
                let theta_pow = pool.pow(theta_inner, pool.integer(j as i32));
                if is_one(vj_s, pool) {
                    theta_pow
                } else {
                    pool.mul(vec![vj_s, theta_pow])
                }
            }
        };
        v_terms.push(theta_j);
    }
    let v_expr = match v_terms.len() {
        0 => zero,
        1 => v_terms[0],
        _ => pool.add(v_terms),
    };

    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
    let result = apply_const(k_const, core, pool);
    log.push(RewriteStep::simple("risch_exp_lower_tower", c_expr, result));
    Some(Ok(result))
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

    // Split off any var-free (constant) factor of the coefficient:
    //   ∫ K·g(x)·exp(kη) dx = K · ∫ g(x)·exp(kη) dx,   K free of `var`.
    // This lets the RDE work on the purely var-dependent remainder `g` over ℚ,
    // while `K` may be an arbitrary symbolic/algebraic constant (e.g. √2, π) that
    // the ℚ-coefficient RDE solver cannot represent (Gap E).
    let (k_const, c_rest) = split_const_factor(c_expr, var, pool);

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
    if let Some(c_poly) = expr_to_qpoly(c_rest, var, pool) {
        return match solve_poly_rde(k, deta, &c_poly) {
            Some(v_poly) => {
                let v_expr = qpoly_to_expr(&v_poly, var, pool);
                let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                let result = apply_const(k_const, core, pool);
                log.push(RewriteStep::simple("risch_exp_rde", c_expr, result));
                Ok(result)
            }
            None => Err(non_elementary()),
        };
    }

    // Rational coefficient → rational Risch DE over ℚ(x) (Bronstein §6.1, Gap 1).
    if let Some((c_num, c_den)) = expr_to_qrational(c_rest, var, pool) {
        // f = k·η' is a polynomial in the exp tower.
        let f = poly_scale(&deta.to_vec(), &rug::Rational::from(k));
        return match solve_rational_rde(&f, &c_num, &c_den) {
            Some((v_num, v_den)) => {
                let v_expr = build_rational(&v_num, &v_den, var, pool);
                let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                let result = apply_const(k_const, core, pool);
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

    // Algebraic-number coefficient → Risch DE over ℚ(α) (Risch Gap E).  Handles a
    // coefficient whose constants lie in a single quadratic field ℚ(√d) and that
    // cannot be split off as a constant factor — both the *polynomial* case
    // (e.g. `x + √2`) and the *rational* case (e.g. `(x−√2−1)/(x−√2)²`).
    if let Some((d, sqrt_expr)) = detect_sqrt_field(c_rest, pool) {
        let field = NumberField::new(vec![
            rug::Rational::from(-d),
            rug::Rational::from(0),
            rug::Rational::from(1),
        ]);
        // Embed the (ℚ-polynomial) η' into K once.
        let deta_k: KPoly = deta.iter().map(|r| field.from_rational(r)).collect();

        // Polynomial coefficient over ℚ(√d).
        if let Some(c_kpoly) = expr_to_kpoly(c_rest, var, sqrt_expr, &field, pool) {
            return match solve_poly_rde_k(&field, k, &deta_k, &c_kpoly) {
                Some(v) => {
                    let v_expr = kpoly_to_expr_alg(&v, var, sqrt_expr, pool);
                    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                    let result = apply_const(k_const, core, pool);
                    log.push(RewriteStep::simple(
                        "risch_exp_rde_algebraic",
                        c_expr,
                        result,
                    ));
                    Ok(result)
                }
                None => Err(non_elementary()),
            };
        }

        // Rational coefficient over ℚ(√d): rational RDE over the number field.
        if let Some((c_num, c_den)) = expr_to_krational(c_rest, var, sqrt_expr, &field, pool) {
            // f = k·η' embedded in K.
            let f_k = field.kpoly_scale(&deta_k, &field.from_int(k));
            return match solve_rational_rde_k(&field, &f_k, &c_num, &c_den) {
                Some((v_num, v_den)) => {
                    let v_expr = build_krational(&v_num, &v_den, var, sqrt_expr, pool);
                    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                    let result = apply_const(k_const, core, pool);
                    log.push(RewriteStep::simple(
                        "risch_exp_rde_algebraic_rational",
                        c_expr,
                        result,
                    ));
                    Ok(result)
                }
                None => Err(non_elementary()),
            };
        }
    }

    // Gap E extended: compositum ℚ(√a,√b) or n-th root ℚ(n^(1/m)).
    // Tries SingleSqrt too, which duplicates the path above but uses the
    // general parser/reconstructor — harmless, and unifies the code.
    if let Some(ext) = detect_algebraic_extension(c_rest, pool) {
        let (field, gens) = build_field_and_gens(&ext);
        let deta_k: KPoly = deta.iter().map(|r| field.from_rational(r)).collect();

        // Polynomial K-coefficient.
        if let Some(c_kpoly) = expr_to_kpoly_general(c_rest, var, &gens, &field, pool) {
            return match solve_poly_rde_k(&field, k, &deta_k, &c_kpoly) {
                Some(v) => {
                    let v_expr = kpoly_to_expr_ext(&v, var, &ext, pool);
                    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                    let result = apply_const(k_const, core, pool);
                    log.push(RewriteStep::simple(
                        "risch_exp_rde_algebraic_ext",
                        c_expr,
                        result,
                    ));
                    Ok(result)
                }
                None => Err(non_elementary()),
            };
        }

        // Rational K-coefficient.
        if let Some((c_num, c_den)) = expr_to_krational_general(c_rest, var, &gens, &field, pool) {
            let f_k = field.kpoly_scale(&deta_k, &field.from_int(k));
            return match solve_rational_rde_k(&field, &f_k, &c_num, &c_den) {
                Some((v_num, v_den)) => {
                    let v_expr = build_krational_ext(&v_num, &v_den, var, &ext, pool);
                    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
                    let result = apply_const(k_const, core, pool);
                    log.push(RewriteStep::simple(
                        "risch_exp_rde_algebraic_rational_ext",
                        c_expr,
                        result,
                    ));
                    Ok(result)
                }
                None => Err(non_elementary()),
            };
        }
    }

    // Gap B: coefficient lives in the lower tower k = ℚ(x, log(h), …).
    // Handle the case where c_rest is a polynomial in a single log generator
    // with rational-in-x coefficients.  Write v = Σ aⱼ·θʲ and match each
    // degree separately:
    //   [θʲ]: aⱼ' + k·η'·aⱼ = cⱼ − (j+1)·(h'/h)·aⱼ₊₁
    // Each sub-problem is a rational-coefficient RDE in ℚ(x).
    if let Some(result) =
        try_poly_in_log_rde(c_rest, k, deta, exp_k_eta, k_const, c_expr, var, pool, log)
    {
        return result;
    }

    // Gap C: x-dependent algebraic coefficient of the form c₀(x) + c₁(x)·√p(x).
    if let Some(result) =
        try_sqrt_poly_rde(c_rest, k, deta, exp_k_eta, k_const, c_expr, var, pool, log)
    {
        return result;
    }

    // Mixed alg+trans, degree ≥ 3: c_rest = Σᵢ cᵢ(x)·a^{i/n} times exp(kη).
    if let Some(result) =
        try_radical_poly_rde(c_rest, k, deta, exp_k_eta, k_const, c_expr, var, pool, log)
    {
        return result;
    }

    // Outside all supported subsets.
    Err(IntegrationError::NotImplemented(format!(
        "coefficient {} of exp(η)^{} is not a polynomial or rational function over \
         a supported algebraic extension; mixed/nested generators are not yet supported",
        pool.display(c_expr),
        k
    )))
}

// ---------------------------------------------------------------------------
// Gap B — poly-in-log RDE for the hyperexponential case (Bronstein §5.9)
// ---------------------------------------------------------------------------

/// Try to integrate `c_rest · exp(kη)^1` when `c_rest` is a polynomial in
/// a single log generator `θ = log(h)` with rational-in-x coefficients.
///
/// **Algorithm** (level-by-level Risch DE):
///
/// Write `c_rest = Σ cⱼ·θʲ` and ansatz `v = Σ aⱼ·θʲ`.  Differentiating and
/// collecting coefficients of `θʲ` in `v' + k·η'·v = c` gives:
/// ```text
///   aⱼ' + k·η'·aⱼ = cⱼ − (j+1)·(h'/h)·aⱼ₊₁     (j = n, n−1, …, 0)
/// ```
/// Each sub-equation is a rational-coefficient Risch DE in ℚ(x) solved by
/// the existing `solve_rational_rde`.  Returns `None` when `c_rest` is not
/// of the poly-in-log form; returns `Some(Err(NonElementary))` when any
/// level has no rational solution; returns `Some(Ok(result))` on success.
///
/// `(c_expr, k_const, exp_k_eta)` are needed to reconstruct the final
/// antiderivative and for log messages.
#[allow(clippy::too_many_arguments)]
fn try_poly_in_log_rde(
    c_rest: ExprId,
    k: i64,
    deta: &[rug::Rational], // η'(x) as ℚ-polynomial
    exp_k_eta: ExprId,
    k_const: ExprId,
    c_expr: ExprId, // for error messages
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<Result<ExprId, IntegrationError>> {
    use super::rational_rde::{expr_to_qrational, solve_rational_rde};
    use super::tower::{decompose_as_log_poly, find_generators};

    // Check that c_rest has exactly one log generator.
    let gens = find_generators(c_rest, var, pool);
    let log_gens: Vec<_> = gens.iter().filter(|g| g.is_log()).collect();
    if log_gens.len() != 1 || gens.iter().any(|g| g.is_exp()) {
        return None; // Not the poly-in-log form or has exp generators too.
    }
    let log_level = log_gens[0];
    let theta = log_level.generator; // ExprId of log(h)
    let h = log_level.argument(); // h

    // Decompose c_rest = Σ cⱼ·θʲ.
    let c_coeffs = decompose_as_log_poly(c_rest, theta, pool)?;
    let n = c_coeffs.len().saturating_sub(1);

    // Compute h'/h as a rational function for the correction term.
    let h_prime = match crate::diff::diff(h, var, pool) {
        Ok(d) => crate::simplify::engine::simplify(d.value, pool).value,
        Err(_) => return None,
    };
    let h_prime_over_h = {
        let raw = pool.mul(vec![h_prime, pool.pow(h, pool.integer(-1_i32))]);
        crate::simplify::engine::simplify(raw, pool).value
    };
    let (hp_num, hp_den) = expr_to_qrational(h_prime_over_h, var, pool)?; // h'/h not rational — outside scope.

    // f = k·η' as a polynomial (already verified polynomial before reaching here).
    let f: Vec<rug::Rational> = deta
        .iter()
        .map(|r| r.clone() * rug::Rational::from(k))
        .collect();

    // Solve from degree n down to 0, accumulating the aⱼ terms.
    let mut a: Vec<Option<(Vec<rug::Rational>, Vec<rug::Rational>)>> = vec![None; n + 1];
    let mut rhs_coeffs: Vec<ExprId> = c_coeffs; // will be updated with corrections

    for j in (0..=n).rev() {
        let cj_expr = crate::simplify::engine::simplify(rhs_coeffs[j], pool).value;
        let (cj_num, cj_den) = match expr_to_qrational(cj_expr, var, pool) {
            Some(p) => p,
            None => {
                return Some(Err(IntegrationError::NotImplemented(format!(
                    "poly-in-log RDE: coefficient of θ^{j} is not rational in {}",
                    pool.display(var)
                ))))
            }
        };

        match solve_rational_rde(&f, &cj_num, &cj_den) {
            Some(sol) => {
                a[j] = Some(sol);
                // Compute the correction for the next lower degree:
                // rhs[j−1] -= j·(h'/h)·aⱼ
                if j > 0 {
                    if let Some((aj_num, aj_den)) = &a[j] {
                        // correction = j · (h'/h) · aⱼ = j · (hp_num/hp_den) · (aj_num/aj_den)
                        // = (j · hp_num · aj_num) / (hp_den · aj_den)
                        use super::poly_rde::{poly_mul, poly_scale};
                        let j_rat = rug::Rational::from(j as i64);
                        let corr_num = poly_scale(&poly_mul(&hp_num, aj_num), &j_rat);
                        let corr_den = poly_mul(&hp_den, aj_den);
                        // rhs[j-1] = rhs[j-1] - correction (as symbolic expr)
                        let corr_expr = {
                            let cn_expr = qpoly_to_expr(&corr_num, var, pool);
                            let cd_expr = qpoly_to_expr(&corr_den, var, pool);
                            pool.mul(vec![cn_expr, pool.pow(cd_expr, pool.integer(-1_i32))])
                        };
                        let old = rhs_coeffs[j - 1];
                        let neg_corr = pool.mul(vec![pool.integer(-1_i32), corr_expr]);
                        rhs_coeffs[j - 1] = pool.add(vec![old, neg_corr]);
                    }
                }
            }
            None => {
                // No rational solution at this level → NonElementary.
                return Some(Err(IntegrationError::NonElementary(format!(
                    "poly-in-log RDE: no rational solution at degree {j} for \
                     ∫ {} · exp(η)^{k} dx",
                    pool.display(c_expr)
                ))));
            }
        }
    }

    // Reconstruct v = Σ aⱼ·θʲ.
    let mut v_terms: Vec<ExprId> = Vec::new();
    for (j, sol) in a.iter().enumerate() {
        if let Some((vn, vd)) = sol {
            let vn_t = trim(vn.clone());
            let vd_t = trim(vd.clone());
            if vn_t.is_empty() {
                continue; // aⱼ = 0
            }
            let vn_expr = qpoly_to_expr(&vn_t, var, pool);
            let coeff_expr = if vd_t == poly_one() {
                vn_expr
            } else {
                let vd_expr = qpoly_to_expr(&vd_t, var, pool);
                pool.mul(vec![vn_expr, pool.pow(vd_expr, pool.integer(-1_i32))])
            };
            let theta_j = match j {
                0 => coeff_expr,
                1 => pool.mul(vec![coeff_expr, theta]),
                _ => pool.mul(vec![coeff_expr, pool.pow(theta, pool.integer(j as i32))]),
            };
            v_terms.push(theta_j);
        }
    }

    let v_expr = match v_terms.len() {
        0 => pool.integer(0_i32),
        1 => v_terms[0],
        _ => pool.add(v_terms),
    };
    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
    let result = apply_const(k_const, core, pool);
    log.push(RewriteStep::simple("risch_exp_poly_in_log", c_expr, result));
    Some(Ok(result))
}

/// Build `v · exp(kη)`, collapsing the `v = 1` case.
fn build_v_times_exp(v_expr: ExprId, exp_k_eta: ExprId, pool: &ExprPool) -> ExprId {
    if is_one(v_expr, pool) {
        exp_k_eta
    } else {
        pool.mul(vec![v_expr, exp_k_eta])
    }
}

// ---------------------------------------------------------------------------
// Gap E extended: compositum ℚ(√a,√b) and n-th root ℚ(n^(1/m)) detection
// ---------------------------------------------------------------------------

/// Describes an algebraic extension ℚ(α) detected in a coefficient expression.
#[derive(Debug, Clone)]
pub(super) enum AlgebraicExtension {
    /// ℚ(√d) — handled by the existing single-sqrt path; included for completeness.
    SingleSqrt { d: i64, sqrt_expr: ExprId },
    /// ℚ(√a, √b) with primitive element α = √a+√b,
    /// minimal polynomial t⁴ − 2(a+b)t² + (a−b)².
    CompositumTwoSqrts {
        a: i64,
        b: i64,
        sqrt_a: ExprId,
        sqrt_b: ExprId,
    },
    /// ℚ(n^(1/m)) with minimal polynomial t^m − n.
    NthRoot { n: i64, m: u32, root_expr: ExprId },
}

/// Scan `expr` for algebraic generators (sqrts and n-th roots of integer
/// constants) and classify them into an `AlgebraicExtension` variant.
///
/// Returns `None` when no radical is found or the combination is too complex
/// (three or more distinct generators, or mixed sqrt+nth-root types).
pub(super) fn detect_algebraic_extension(
    expr: ExprId,
    pool: &ExprPool,
) -> Option<AlgebraicExtension> {
    let mut sqrts: Vec<(i64, ExprId)> = Vec::new();
    let mut nth_roots: Vec<(i64, u32, ExprId)> = Vec::new();
    scan_algebraic_gens(expr, pool, &mut sqrts, &mut nth_roots);

    // Deduplicate by radicand.
    let mut dsqrts: Vec<(i64, ExprId)> = Vec::new();
    for (d, e) in sqrts {
        if !dsqrts.iter().any(|(dd, _)| *dd == d) {
            dsqrts.push((d, e));
        }
    }
    let mut droots: Vec<(i64, u32, ExprId)> = Vec::new();
    for (n, m, e) in nth_roots {
        if !droots.iter().any(|(nn, mm, _)| *nn == n && *mm == m) {
            droots.push((n, m, e));
        }
    }

    match (dsqrts.len(), droots.len()) {
        (0, 0) => None,
        (1, 0) => {
            let (d, sqrt_expr) = dsqrts[0];
            Some(AlgebraicExtension::SingleSqrt { d, sqrt_expr })
        }
        (2, 0) => {
            let (mut a, mut sqrt_a) = dsqrts[0];
            let (mut b, mut sqrt_b) = dsqrts[1];
            // Canonical: a < b.
            if a > b {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut sqrt_a, &mut sqrt_b);
            }
            Some(AlgebraicExtension::CompositumTwoSqrts {
                a,
                b,
                sqrt_a,
                sqrt_b,
            })
        }
        (0, 1) => {
            let (n, m, root_expr) = droots[0];
            Some(AlgebraicExtension::NthRoot { n, m, root_expr })
        }
        _ => None, // Three+ generators, or mixed sqrt+nth-root: out of scope.
    }
}

/// Scan `expr` collecting all `sqrt(integer)` and `n^(1/m)` generators.
fn scan_algebraic_gens(
    expr: ExprId,
    pool: &ExprPool,
    sqrts: &mut Vec<(i64, ExprId)>,
    nth_roots: &mut Vec<(i64, u32, ExprId)>,
) {
    use crate::kernel::ExprData;
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let arg = args[0];
            match name.as_str() {
                "sqrt" => {
                    if let ExprData::Integer(n) = pool.get(arg) {
                        if let Some(d) = n.0.to_i64() {
                            if d > 1 && !is_perfect_square(d) {
                                sqrts.push((d, expr));
                            }
                        }
                    }
                    // Don't recurse into the argument — the argument is just a number.
                }
                "cbrt" => {
                    if let ExprData::Integer(n) = pool.get(arg) {
                        if let Some(d) = n.0.to_i64() {
                            if d > 1 && !is_perfect_mth_power(d, 3) {
                                nth_roots.push((d, 3, expr));
                            }
                        }
                    }
                }
                _ => {
                    scan_algebraic_gens(arg, pool, sqrts, nth_roots);
                }
            }
        }
        ExprData::Pow { base, exp } => {
            // Detect Integer^Rational(1/m) — e.g. 2^(1/3).
            if let (ExprData::Integer(n_int), ExprData::Rational(r)) =
                (pool.get(base), pool.get(exp))
            {
                if let Some(d) = n_int.0.to_i64() {
                    // r = 1/m with m ≥ 2.
                    if *r.0.numer() == 1 {
                        if let Some(m) = r.0.denom().to_u32() {
                            if m >= 2 && d > 1 && !is_perfect_mth_power(d, m) {
                                nth_roots.push((d, m, expr));
                                return; // Don't recurse.
                            }
                        }
                    }
                }
            }
            scan_algebraic_gens(base, pool, sqrts, nth_roots);
            scan_algebraic_gens(exp, pool, sqrts, nth_roots);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan_algebraic_gens(a, pool, sqrts, nth_roots);
            }
        }
        _ => {}
    }
}

/// Returns true if `d` is a perfect m-th power (i.e. ∃ integer k with k^m = d).
fn is_perfect_mth_power(d: i64, m: u32) -> bool {
    if d <= 0 || m == 0 {
        return false;
    }
    if m == 1 {
        return true;
    }
    let root = (d as f64).powf(1.0 / m as f64).round() as i64;
    (root - 1..=root + 1).any(|k| k > 0 && k.pow(m) == d)
}

/// Build the `NumberField` for an `AlgebraicExtension` and return the list of
/// (generator_symbol, K-element) pairs used by the general K-poly parser.
pub(super) fn build_field_and_gens(
    ext: &AlgebraicExtension,
) -> (NumberField, Vec<(ExprId, KElem)>) {
    match ext {
        AlgebraicExtension::SingleSqrt { d, sqrt_expr } => {
            let field = NumberField::new(vec![
                rug::Rational::from(-d),
                rug::Rational::from(0),
                rug::Rational::from(1),
            ]);
            // √d ↦ K-element [0, 1] in ℚ[t]/(t²−d).
            let kelem = vec![rug::Rational::from(0), rug::Rational::from(1)];
            (field, vec![(*sqrt_expr, kelem)])
        }
        AlgebraicExtension::CompositumTwoSqrts {
            a,
            b,
            sqrt_a,
            sqrt_b,
        } => {
            let a = *a;
            let b = *b;
            // Primitive element α = √a+√b, min poly t⁴ − 2(a+b)t² + (a−b)².
            let field = NumberField::new(vec![
                rug::Rational::from((a - b) * (a - b)),
                rug::Rational::from(0),
                rug::Rational::from(-2 * (a + b)),
                rug::Rational::from(0),
                rug::Rational::from(1),
            ]);
            // √a = ((3a+b)α − α³) / (2(a−b))   →  K-elem [0, (3a+b)/(2(a−b)), 0, −1/(2(a−b))]
            // √b = (α³ − (a+3b)α) / (2(a−b))   →  K-elem [0, −(a+3b)/(2(a−b)), 0, 1/(2(a−b))]
            let two_ab = rug::Rational::from(2 * (a - b));
            let kelem_a = vec![
                rug::Rational::from(0),
                rug::Rational::from(3 * a + b) / two_ab.clone(),
                rug::Rational::from(0),
                rug::Rational::from(-1) / two_ab.clone(),
            ];
            let kelem_b = vec![
                rug::Rational::from(0),
                rug::Rational::from(-(a + 3 * b)) / two_ab.clone(),
                rug::Rational::from(0),
                rug::Rational::from(1) / two_ab,
            ];
            (field, vec![(*sqrt_a, kelem_a), (*sqrt_b, kelem_b)])
        }
        AlgebraicExtension::NthRoot { n, m, root_expr } => {
            let n = *n;
            let m = *m;
            // ℚ[t]/(t^m − n).
            let mut min_poly = vec![rug::Rational::from(0); m as usize + 1];
            min_poly[0] = rug::Rational::from(-n);
            min_poly[m as usize] = rug::Rational::from(1);
            let field = NumberField::new(min_poly);
            // n^(1/m) ↦ K-element [0, 1, 0, …, 0] in ℚ[t]/(t^m−n).
            let mut kelem = vec![rug::Rational::from(0); m as usize];
            kelem[1] = rug::Rational::from(1);
            (field, vec![(*root_expr, kelem)])
        }
    }
}

// ---------------------------------------------------------------------------
// General K-polynomial parser (multiple algebraic generators)
// ---------------------------------------------------------------------------

/// Parse `expr` as a polynomial in `var` whose coefficients lie in `K = ℚ(α)`,
/// given a list of `(generator_symbol, K-element)` pairs.
///
/// Returns `None` if the expression is not of this form (e.g. contains a
/// transcendental generator or an algebraic element not in the list).
fn expr_to_kpoly_general(
    expr: ExprId,
    var: ExprId,
    gens: &[(ExprId, KElem)],
    field: &NumberField,
    pool: &ExprPool,
) -> Option<KPoly> {
    use crate::kernel::ExprData;

    if expr == var {
        return Some(vec![NumberField::k_zero(), field.from_int(1)]);
    }
    for (gen_expr, kelem) in gens {
        if expr == *gen_expr {
            return Some(vec![kelem.clone()]);
        }
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(vec![
            field.from_rational(&rug::Rational::from(n.0.to_i64()?))
        ]),
        ExprData::Rational(r) => Some(vec![field.from_rational(&r.0)]),
        ExprData::Add(args) => {
            let mut acc: KPoly = Vec::new();
            for a in &args {
                let p = expr_to_kpoly_general(*a, var, gens, field, pool)?;
                acc = field.kpoly_add(&acc, &p);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc: KPoly = vec![field.from_int(1)];
            for a in &args {
                let p = expr_to_kpoly_general(*a, var, gens, field, pool)?;
                acc = field.kpoly_mul(&acc, &p);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let n = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_i64()?,
                _ => return None,
            };
            let b = expr_to_kpoly_general(base, var, gens, field, pool)?;
            if n >= 0 {
                let mut acc: KPoly = vec![field.from_int(1)];
                for _ in 0..n {
                    acc = field.kpoly_mul(&acc, &b);
                }
                Some(acc)
            } else {
                // Negative power: only for var-free constant base.
                if NumberField::kdeg(&b) != 0 {
                    return None;
                }
                let inv = field.inv(&b[0])?;
                let mut acc = field.from_int(1);
                for _ in 0..(-n) {
                    acc = field.mul(&acc, &inv);
                }
                Some(vec![acc])
            }
        }
        _ => None,
    }
}

/// Parse `expr` as a rational function in `var` over `K = ℚ(α)`.
/// Returns `(numerator, denominator)` or `None`.
pub(super) fn expr_to_krational_general(
    expr: ExprId,
    var: ExprId,
    gens: &[(ExprId, KElem)],
    field: &NumberField,
    pool: &ExprPool,
) -> Option<(KPoly, KPoly)> {
    use crate::kernel::ExprData;
    let one: KPoly = vec![field.from_int(1)];

    if expr == var {
        return Some((vec![NumberField::k_zero(), field.from_int(1)], one));
    }
    for (gen_expr, kelem) in gens {
        if expr == *gen_expr {
            return Some((vec![kelem.clone()], one.clone()));
        }
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some((
            vec![field.from_rational(&rug::Rational::from(n.0.to_i64()?))],
            one,
        )),
        ExprData::Rational(r) => Some((vec![field.from_rational(&r.0)], one)),
        ExprData::Add(args) => {
            let mut acc: (KPoly, KPoly) = (Vec::new(), one.clone());
            for a in &args {
                let term = expr_to_krational_general(*a, var, gens, field, pool)?;
                acc = krat_add(field, &acc, &term);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc: (KPoly, KPoly) = (one.clone(), one.clone());
            for a in &args {
                let factor = expr_to_krational_general(*a, var, gens, field, pool)?;
                acc = krat_mul(field, &acc, &factor);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let n = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_i64()?,
                _ => return None,
            };
            let (bn, bd) = expr_to_krational_general(base, var, gens, field, pool)?;
            if n >= 0 {
                Some((
                    field.kpoly_pow(&bn, n as u32),
                    field.kpoly_pow(&bd, n as u32),
                ))
            } else {
                let m = (-n) as u32;
                Some((field.kpoly_pow(&bd, m), field.kpoly_pow(&bn, m)))
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// K-element / K-polynomial reconstruction (general algebraic extensions)
// ---------------------------------------------------------------------------

/// Convert a K-element back to a symbolic `ExprId` for the given extension.
fn kelem_to_expr_ext(e: &KElem, ext: &AlgebraicExtension, pool: &ExprPool) -> ExprId {
    match ext {
        AlgebraicExtension::SingleSqrt { sqrt_expr, .. } => kelem_to_expr(e, *sqrt_expr, pool),
        AlgebraicExtension::NthRoot { root_expr, .. } => {
            // c₀ + c₁·ζ + c₂·ζ² + …
            let mut terms: Vec<ExprId> = Vec::new();
            for (i, c) in e.iter().enumerate() {
                if *c == 0 {
                    continue;
                }
                let c_expr = rational_to_expr(c, pool);
                let term = match i {
                    0 => c_expr,
                    1 => {
                        if *c == 1 {
                            *root_expr
                        } else {
                            pool.mul(vec![c_expr, *root_expr])
                        }
                    }
                    _ => {
                        let xp = pool.pow(*root_expr, pool.integer(i as i32));
                        if *c == 1 {
                            xp
                        } else {
                            pool.mul(vec![c_expr, xp])
                        }
                    }
                };
                terms.push(term);
            }
            match terms.len() {
                0 => pool.integer(0_i32),
                1 => terms[0],
                _ => pool.add(terms),
            }
        }
        AlgebraicExtension::CompositumTwoSqrts {
            a,
            b,
            sqrt_a,
            sqrt_b,
        } => {
            // K-element [n₀, n₁, n₂, n₃] in ℚ(α), α = √a+√b.
            // Back-substitute: α=√a+√b, α²=(a+b)+2√(ab), α³=(a+3b)√a+(3a+b)√b.
            //   coeff_1     = n₀ + n₂·(a+b)
            //   coeff_√a    = n₁ + n₃·(a+3b)
            //   coeff_√b    = n₁ + n₃·(3a+b)
            //   coeff_√(ab) = 2·n₂
            let a = *a;
            let b = *b;
            let c = |i: usize| -> rug::Rational {
                e.get(i).cloned().unwrap_or_else(|| rug::Rational::from(0))
            };
            let coeff_1 = c(0) + c(2).clone() * rug::Rational::from(a + b);
            let coeff_sa = c(1) + c(3).clone() * rug::Rational::from(a + 3 * b);
            let coeff_sb = c(1) + c(3).clone() * rug::Rational::from(3 * a + b);
            let coeff_sab = c(2) * rug::Rational::from(2);

            let sqrt_ab = pool.mul(vec![*sqrt_a, *sqrt_b]);
            let mut terms: Vec<ExprId> = Vec::new();
            if coeff_1 != 0 {
                terms.push(rational_to_expr(&coeff_1, pool));
            }
            if coeff_sa != 0 {
                let t = if coeff_sa == 1 {
                    *sqrt_a
                } else {
                    pool.mul(vec![rational_to_expr(&coeff_sa, pool), *sqrt_a])
                };
                terms.push(t);
            }
            if coeff_sb != 0 {
                let t = if coeff_sb == 1 {
                    *sqrt_b
                } else {
                    pool.mul(vec![rational_to_expr(&coeff_sb, pool), *sqrt_b])
                };
                terms.push(t);
            }
            if coeff_sab != 0 {
                let t = if coeff_sab == 1 {
                    sqrt_ab
                } else {
                    pool.mul(vec![rational_to_expr(&coeff_sab, pool), sqrt_ab])
                };
                terms.push(t);
            }
            match terms.len() {
                0 => pool.integer(0_i32),
                1 => terms[0],
                _ => pool.add(terms),
            }
        }
    }
}

/// Convert a K-polynomial `p` in `var` back to a symbolic expression for the
/// given algebraic extension.
fn kpoly_to_expr_ext(p: &KPoly, var: ExprId, ext: &AlgebraicExtension, pool: &ExprPool) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for (i, c) in p.iter().enumerate() {
        if NumberField::is_zero(c) {
            continue;
        }
        let c_expr = kelem_to_expr_ext(c, ext, pool);
        let term = match i {
            0 => c_expr,
            1 => {
                if is_one(c_expr, pool) {
                    var
                } else {
                    pool.mul(vec![c_expr, var])
                }
            }
            _ => {
                let xp = pool.pow(var, pool.integer(i as i32));
                if is_one(c_expr, pool) {
                    xp
                } else {
                    pool.mul(vec![c_expr, xp])
                }
            }
        };
        terms.push(term);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// Build the symbolic rational expression `num(x)/den(x)` for the given extension.
pub(super) fn build_krational_ext(
    num: &KPoly,
    den: &KPoly,
    var: ExprId,
    ext: &AlgebraicExtension,
    pool: &ExprPool,
) -> ExprId {
    let num_expr = kpoly_to_expr_ext(num, var, ext, pool);
    let den_is_one = NumberField::kdeg(den) <= 0
        && den
            .first()
            .map(|c| trim(c.clone()) == vec![rug::Rational::from(1)])
            .unwrap_or(true);
    if den_is_one {
        return num_expr;
    }
    let den_expr = kpoly_to_expr_ext(den, var, ext, pool);
    pool.mul(vec![num_expr, pool.pow(den_expr, pool.integer(-1_i32))])
}

// ---------------------------------------------------------------------------
// Algebraic-number coefficients ℚ(√d)  (Risch Gap E, polynomial RDE)
// ---------------------------------------------------------------------------

/// Detect a single quadratic algebraic constant `√d` (`d` a non-square integer
/// `> 1`) in `expr`.  Returns `(d, sqrt_expr)` when exactly one distinct such
/// radical is present, else `None` (no radical, or a compositum of several
/// distinct radicals — out of scope for the single-quadratic-field path).
fn detect_sqrt_field(expr: ExprId, pool: &ExprPool) -> Option<(i64, ExprId)> {
    let mut found: Vec<(i64, ExprId)> = Vec::new();
    scan_sqrt(expr, pool, &mut found);
    let mut distinct: Vec<(i64, ExprId)> = Vec::new();
    for (d, e) in found {
        if !distinct.iter().any(|(dd, _)| *dd == d) {
            distinct.push((d, e));
        }
    }
    match distinct.len() {
        1 => Some(distinct[0]),
        _ => None,
    }
}

/// Collect every `sqrt(integer)` radical (non-square, `> 1`) occurring in `expr`.
fn scan_sqrt(expr: ExprId, pool: &ExprPool, out: &mut Vec<(i64, ExprId)>) {
    use crate::kernel::ExprData;
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            if let ExprData::Integer(n) = pool.get(args[0]) {
                if let Some(d) = n.0.to_i64() {
                    if d > 1 && !is_perfect_square(d) {
                        out.push((d, expr));
                    }
                }
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan_sqrt(a, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            scan_sqrt(base, pool, out);
            scan_sqrt(exp, pool, out);
        }
        ExprData::Func { ref args, .. } => {
            for &a in args {
                scan_sqrt(a, pool, out);
            }
        }
        _ => {}
    }
}

/// Is `d` a perfect square?
fn is_perfect_square(d: i64) -> bool {
    if d < 0 {
        return false;
    }
    let r = (d as f64).sqrt() as i64;
    (r - 1..=r + 1).any(|c| c >= 0 && c * c == d)
}

/// Parse `expr` as a polynomial in `var` whose coefficients lie in `K = ℚ(√d)`
/// (where `√d` is the symbol `sqrt_expr`), returning a [`KPoly`] ascending in
/// `var`, or `None` if `expr` is not such a polynomial (e.g. a `var` appears in a
/// denominator, or a foreign generator is present).
fn expr_to_kpoly(
    expr: ExprId,
    var: ExprId,
    sqrt_expr: ExprId,
    field: &NumberField,
    pool: &ExprPool,
) -> Option<KPoly> {
    use crate::kernel::ExprData;
    if expr == var {
        return Some(vec![NumberField::k_zero(), field.from_int(1)]);
    }
    if expr == sqrt_expr {
        // The field generator √d as a K-constant: 0 + 1·t.
        return Some(vec![vec![rug::Rational::from(0), rug::Rational::from(1)]]);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(vec![
            field.from_rational(&rug::Rational::from(n.0.to_i64()?))
        ]),
        ExprData::Rational(r) => Some(vec![field.from_rational(&r.0)]),
        ExprData::Add(args) => {
            let mut acc: KPoly = Vec::new();
            for a in &args {
                let p = expr_to_kpoly(*a, var, sqrt_expr, field, pool)?;
                acc = field.kpoly_add(&acc, &p);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc: KPoly = vec![field.from_int(1)];
            for a in &args {
                let p = expr_to_kpoly(*a, var, sqrt_expr, field, pool)?;
                acc = field.kpoly_mul(&acc, &p);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let n = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_i64()?,
                _ => return None,
            };
            let b = expr_to_kpoly(base, var, sqrt_expr, field, pool)?;
            if n >= 0 {
                let mut acc: KPoly = vec![field.from_int(1)];
                for _ in 0..n {
                    acc = field.kpoly_mul(&acc, &b);
                }
                Some(acc)
            } else {
                // Negative power: only a (var-free) constant base is admissible for
                // a *polynomial* coefficient.
                if NumberField::kdeg(&b) != 0 {
                    return None;
                }
                let inv = field.inv(&b[0])?;
                let mut acc = field.from_int(1);
                for _ in 0..(-n) {
                    acc = field.mul(&acc, &inv);
                }
                Some(vec![acc])
            }
        }
        _ => None,
    }
}

/// Reconstruct a `K = ℚ(√d)` element `a + b·√d` as a symbolic expression.
fn kelem_to_expr(e: &KElem, sqrt_expr: ExprId, pool: &ExprPool) -> ExprId {
    let a = e.first().cloned().unwrap_or_else(|| rug::Rational::from(0));
    let b = e.get(1).cloned().unwrap_or_else(|| rug::Rational::from(0));
    let mut terms: Vec<ExprId> = Vec::new();
    if a != 0 {
        terms.push(rational_to_expr(&a, pool));
    }
    if b != 0 {
        let bt = if b == 1 {
            sqrt_expr
        } else {
            pool.mul(vec![rational_to_expr(&b, pool), sqrt_expr])
        };
        terms.push(bt);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// Reconstruct a [`KPoly`] in `var` over `ℚ(√d)` as a symbolic expression.
fn kpoly_to_expr_alg(p: &KPoly, var: ExprId, sqrt_expr: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::new();
    for (i, c) in p.iter().enumerate() {
        if NumberField::is_zero(c) {
            continue;
        }
        let ce = kelem_to_expr(c, sqrt_expr, pool);
        let term = match i {
            0 => ce,
            1 => {
                if is_one(ce, pool) {
                    var
                } else {
                    pool.mul(vec![ce, var])
                }
            }
            _ => {
                let xp = pool.pow(var, pool.integer(i as i32));
                if is_one(ce, pool) {
                    xp
                } else {
                    pool.mul(vec![ce, xp])
                }
            }
        };
        terms.push(term);
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// Parse `expr` as a rational function in `var` over `K = ℚ(√d)` (where `√d` is
/// the symbol `sqrt_expr`), returning `(numerator, denominator)` as `KPoly`s, or
/// `None` if it is not such a rational function.
fn expr_to_krational(
    expr: ExprId,
    var: ExprId,
    sqrt_expr: ExprId,
    field: &NumberField,
    pool: &ExprPool,
) -> Option<(KPoly, KPoly)> {
    use crate::kernel::ExprData;
    let one: KPoly = vec![field.from_int(1)];
    if expr == var {
        return Some((vec![NumberField::k_zero(), field.from_int(1)], one));
    }
    if expr == sqrt_expr {
        return Some((
            vec![vec![rug::Rational::from(0), rug::Rational::from(1)]],
            one,
        ));
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some((
            vec![field.from_rational(&rug::Rational::from(n.0.to_i64()?))],
            one,
        )),
        ExprData::Rational(r) => Some((vec![field.from_rational(&r.0)], one)),
        ExprData::Add(args) => {
            let mut acc: (KPoly, KPoly) = (Vec::new(), one);
            for a in &args {
                let term = expr_to_krational(*a, var, sqrt_expr, field, pool)?;
                acc = krat_add(field, &acc, &term);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc: (KPoly, KPoly) = (one.clone(), one);
            for a in &args {
                let factor = expr_to_krational(*a, var, sqrt_expr, field, pool)?;
                acc = krat_mul(field, &acc, &factor);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let n = match pool.get(exp) {
                ExprData::Integer(n) => n.0.to_i64()?,
                _ => return None,
            };
            let (bn, bd) = expr_to_krational(base, var, sqrt_expr, field, pool)?;
            if n >= 0 {
                Some((
                    field.kpoly_pow(&bn, n as u32),
                    field.kpoly_pow(&bd, n as u32),
                ))
            } else {
                let m = (-n) as u32;
                if NumberField::kdeg(&bn) < 0 {
                    return None; // 1 / 0
                }
                Some((field.kpoly_pow(&bd, m), field.kpoly_pow(&bn, m)))
            }
        }
        _ => None,
    }
}

/// `a/b + c/d = (a·d + c·b)/(b·d)` over `K`.
fn krat_add(field: &NumberField, a: &(KPoly, KPoly), b: &(KPoly, KPoly)) -> (KPoly, KPoly) {
    let num = field.kpoly_add(&field.kpoly_mul(&a.0, &b.1), &field.kpoly_mul(&b.0, &a.1));
    let den = field.kpoly_mul(&a.1, &b.1);
    (num, den)
}

/// `(a/b)·(c/d) = (a·c)/(b·d)` over `K`.
fn krat_mul(field: &NumberField, a: &(KPoly, KPoly), b: &(KPoly, KPoly)) -> (KPoly, KPoly) {
    (field.kpoly_mul(&a.0, &b.0), field.kpoly_mul(&a.1, &b.1))
}

/// Reconstruct a `K = ℚ(√d)` rational function `num(x)/den(x)` as a symbolic
/// expression, collapsing a denominator of 1.
fn build_krational(
    num: &KPoly,
    den: &KPoly,
    var: ExprId,
    sqrt_expr: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let num_expr = kpoly_to_expr_alg(num, var, sqrt_expr, pool);
    // den == 1 (a single K-element equal to 1)?
    let den_is_one = NumberField::kdeg(den) <= 0
        && den
            .first()
            .map(|c| trim(c.clone()) == vec![rug::Rational::from(1)])
            .unwrap_or(true);
    if den_is_one {
        return num_expr;
    }
    let den_expr = kpoly_to_expr_alg(den, var, sqrt_expr, pool);
    let den_inv = pool.pow(den_expr, pool.integer(-1_i32));
    pool.mul(vec![num_expr, den_inv])
}

/// Build the symbolic rational function `num(x) / den(x)`.
pub(super) fn build_rational(
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
// Gap C — mixed algebraic × exp tower:  sqrt(p(x)) coefficients (Bronstein §5.9)
// ---------------------------------------------------------------------------
//
// For an integrand c(x, √p(x))·exp(kη) with c = c₀ + c₁·α (α = √p, p ∈ ℚ[x]),
// the antiderivative has the form (a + b·α)·exp(kη) iff both rational Risch DEs
//
//   a' + f·a = c₀                          (over ℚ(x))
//   b' + (f + p'/(2p))·b = c₁              (over ℚ(x), rational f_eff)
//
// have rational solutions, where f = k·η'.  Each is solved by the existing
// `solve_rational_rde` / `solve_rational_rde_generalized`.  This identity follows
// from the twisted derivation D(a + bα) = a' + (b' + b·p'/(2p))·α.

/// Returns `true` if `expr` has an x-dependent algebraic sub-expression
/// (e.g. `sqrt(x²+1)`, `x^{1/2}`) — as opposed to *constant* algebraic factors
/// like `sqrt(2)` (Gap E) which are free of the integration variable.
fn contains_var_algebraic(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::ExprData;
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            !is_free_of_var(args[0], var, pool)
        }
        ExprData::Pow { base, exp } => {
            if matches!(pool.get(exp), ExprData::Rational(_)) {
                !is_free_of_var(base, var, pool)
            } else {
                contains_var_algebraic(base, var, pool)
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_var_algebraic(a, var, pool))
        }
        _ => false,
    }
}

/// Returns `Some((p_poly, sqrt_expr))` if `expr` contains exactly one
/// x-dependent `sqrt(p(x))` generator (degree(p) ≥ 1).  Returns `None` if
/// there is none or more than one distinct such generator.
fn detect_sqrt_of_poly(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(QPoly, ExprId)> {
    let mut found: Vec<(QPoly, ExprId)> = Vec::new();
    scan_sqrt_of_poly(expr, var, pool, &mut found);
    // Deduplicate by polynomial content.
    let mut distinct: Vec<(QPoly, ExprId)> = Vec::new();
    for (p, e) in found {
        if !distinct
            .iter()
            .any(|(q, _)| trim(q.clone()) == trim(p.clone()))
        {
            distinct.push((p, e));
        }
    }
    if distinct.len() == 1 {
        Some(distinct.remove(0))
    } else {
        None
    }
}

fn scan_sqrt_of_poly(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(QPoly, ExprId)>) {
    use crate::kernel::ExprData;
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            if let Some(p) = expr_to_qpoly(args[0], var, pool) {
                if degree(&p) >= 1 {
                    out.push((p, expr));
                }
            }
            // Don't recurse into the sqrt argument.
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                // base^{m/2}: the radicand is base.
                if *r.0.denom() == 2 {
                    if let Some(p) = expr_to_qpoly(base, var, pool) {
                        if degree(&p) >= 1 {
                            out.push((p, expr));
                            return; // Don't recurse into base or exp.
                        }
                    }
                }
            }
            scan_sqrt_of_poly(base, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan_sqrt_of_poly(a, var, pool, out);
            }
        }
        _ => {}
    }
}

/// Return the polynomial radicand of an alpha generator:
///   `sqrt(p)` → `p`,   `p^{1/2}` → `p`.
fn get_radicand_expr(alpha: ExprId, pool: &ExprPool) -> Option<ExprId> {
    use crate::kernel::ExprData;
    match pool.get(alpha) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => Some(args[0]),
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.numer() == 1 && *r.0.denom() == 2 {
                    return Some(base);
                }
            }
            None
        }
        _ => None,
    }
}

// --- Minimal rational-function (QRat) arithmetic -------------------------

type QRat = (QPoly, QPoly); // (numerator, denominator) of a ℚ(x) element

fn qr_zero() -> QRat {
    (poly_zero(), poly_one())
}
fn qr_one() -> QRat {
    (poly_one(), poly_one())
}

fn qr_add(a: &QRat, b: &QRat) -> QRat {
    (
        poly_add(&poly_mul(&a.0, &b.1), &poly_mul(&b.0, &a.1)),
        poly_mul(&a.1, &b.1),
    )
}

fn qr_mul(a: &QRat, b: &QRat) -> QRat {
    (poly_mul(&a.0, &b.0), poly_mul(&a.1, &b.1))
}

/// Multiply a rational function by a polynomial: (n/d) · p = (n·p)/d.
fn qr_scale_poly(a: &QRat, p: &QPoly) -> QRat {
    (poly_mul(&a.0, p), a.1.clone())
}

// --- KPair: elements c₀ + c₁·α of ℚ(x)(α) with α² = p(x) --------------

type KPair = (QRat, QRat); // (c0, c1) representing c0 + c1·α

fn kp_zero() -> KPair {
    (qr_zero(), qr_zero())
}
fn kp_one() -> KPair {
    (qr_one(), qr_zero())
}
fn kp_alpha() -> KPair {
    (qr_zero(), qr_one())
}
fn kp_from_qr(r: QRat) -> KPair {
    (r, qr_zero())
}

fn kp_add(a: &KPair, b: &KPair) -> KPair {
    (qr_add(&a.0, &b.0), qr_add(&a.1, &b.1))
}

/// Multiply in ℚ(x)(α) with α² = p(x).
/// (a₀ + a₁α)(b₀ + b₁α) = (a₀b₀ + a₁b₁p) + (a₀b₁ + a₁b₀)α
fn kp_mul(a: &KPair, b: &KPair, p: &QPoly) -> KPair {
    let c0 = qr_add(&qr_mul(&a.0, &b.0), &qr_scale_poly(&qr_mul(&a.1, &b.1), p));
    let c1 = qr_add(&qr_mul(&a.0, &b.1), &qr_mul(&a.1, &b.0));
    (c0, c1)
}

/// Invert in ℚ(x)(α): 1/(a₀+a₁α) = (a₀−a₁α)/(a₀²−a₁²p).
fn kp_inv(a: &KPair, p: &QPoly) -> Option<KPair> {
    // norm = a₀² − a₁²·p  (as a rational function)
    let a0sq = qr_mul(&a.0, &a.0);
    let a1sq_p = qr_scale_poly(&qr_mul(&a.1, &a.1), p);
    // norm_num/norm_den = a0sq − a1sq_p
    let norm_num = poly_sub(&poly_mul(&a0sq.0, &a1sq_p.1), &poly_mul(&a1sq_p.0, &a0sq.1));
    let norm_den = poly_mul(&a0sq.1, &a1sq_p.1);
    if trim(norm_num.clone()).is_empty() {
        return None; // zero norm → not invertible
    }
    // 1/norm = norm_den / norm_num
    // c0 = a0 / norm = (a0.0 · norm_den) / (a0.1 · norm_num)
    let inv_c0 = (poly_mul(&a.0 .0, &norm_den), poly_mul(&a.0 .1, &norm_num));
    // c1 = −a1 / norm = (−a1.0 · norm_den) / (a1.1 · norm_num)
    let neg_a1_num = poly_scale(&a.1 .0, &rug::Rational::from(-1));
    let inv_c1 = (
        poly_mul(&neg_a1_num, &norm_den),
        poly_mul(&a.1 .1, &norm_num),
    );
    Some((inv_c0, inv_c1))
}

fn kp_pow(a: &KPair, n: i64, p: &QPoly) -> Option<KPair> {
    if n == 0 {
        return Some(kp_one());
    }
    if n < 0 {
        let inv = kp_inv(a, p)?;
        return kp_pow(&inv, -n, p);
    }
    let mut acc = kp_one();
    for _ in 0..n {
        acc = kp_mul(&acc, a, p);
    }
    Some(acc)
}

/// Decompose `expr` as c₀(x) + c₁(x)·α where α = `alpha` (a sqrt(p(x)) node)
/// and p = `p_poly`.  Returns `None` if the expression cannot be so written.
fn decompose_over_alpha(
    expr: ExprId,
    alpha: ExprId,
    p_poly: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<KPair> {
    use crate::kernel::ExprData;

    // Free of alpha → pure rational c₀.
    if !contains_subexpr(expr, alpha, pool) {
        let r = expr_to_qrational(expr, var, pool)?;
        return Some(kp_from_qr(r));
    }
    // expr IS alpha.
    if expr == alpha {
        return Some(kp_alpha());
    }

    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut acc = kp_zero();
            for &a in &args {
                let t = decompose_over_alpha(a, alpha, p_poly, var, pool)?;
                acc = kp_add(&acc, &t);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = kp_one();
            for &a in &args {
                let t = decompose_over_alpha(a, alpha, p_poly, var, pool)?;
                acc = kp_mul(&acc, &t, p_poly);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            match pool.get(exp) {
                ExprData::Integer(n) => {
                    let n = n.0.to_i64()?;
                    let b = decompose_over_alpha(base, alpha, p_poly, var, pool)?;
                    kp_pow(&b, n, p_poly)
                }
                ExprData::Rational(r) => {
                    // base^{m/2} where base is the radicand of alpha → alpha^m.
                    if get_radicand_expr(alpha, pool) == Some(base) && *r.0.denom() == 2 {
                        let m = r.0.numer().to_i64()?;
                        return kp_pow(&kp_alpha(), m, p_poly);
                    }
                    None
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// --- Degree-d generalization: decompose over an arbitrary radical generator --
//
// These generalize `decompose_over_alpha` (rank-2, KPair) to a degree-`n`
// simple radical extension `ℚ(x)[y]/(yⁿ − p)` represented by `alg_field`.  They
// are the symbolic→`AlgElem` bridge the mixed-integration integral part (MA /
// M1, see temp-alkahest/planning/risch.md) consumes; not yet wired into the
// integrator, hence `#[allow(dead_code)]`.

/// Detect a single x-dependent simple-radical generator `p(x)^{1/n}` (`n ≥ 2`)
/// in `expr`.  Returns `(n, p)` iff there is **exactly one** distinct such
/// generator (keyed by radicand `p` and index `n`): `sqrt`→`n=2`, `cbrt`→`n=3`,
/// `base^{a/n}`→`n =` the reduced exponent denominator.  Returns `None` for no
/// generator or for two distinct ones (a compositum — out of MA scope).
pub(super) fn detect_radical_generator(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(usize, QPoly)> {
    let mut found: Vec<(usize, QPoly)> = Vec::new();
    scan_radical_generator(expr, var, pool, &mut found);
    let mut distinct: Vec<(usize, QPoly)> = Vec::new();
    for (n, p) in found {
        if !distinct
            .iter()
            .any(|(m, q)| *m == n && trim(q.clone()) == trim(p.clone()))
        {
            distinct.push((n, p));
        }
    }
    if distinct.len() == 1 {
        Some(distinct.remove(0))
    } else {
        None
    }
}

fn scan_radical_generator(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    out: &mut Vec<(usize, QPoly)>,
) {
    use crate::kernel::ExprData;
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            if let Some(p) = expr_to_qpoly(args[0], var, pool) {
                if degree(&p) >= 1 {
                    out.push((2, p));
                }
            }
        }
        ExprData::Func { ref name, ref args } if name == "cbrt" && args.len() == 1 => {
            if let Some(p) = expr_to_qpoly(args[0], var, pool) {
                if degree(&p) >= 1 {
                    out.push((3, p));
                }
            }
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if let Some(den) = r.0.denom().to_i64() {
                    if den >= 2 {
                        if let Some(p) = expr_to_qpoly(base, var, pool) {
                            if degree(&p) >= 1 {
                                out.push((den as usize, p));
                                return; // don't recurse into the radicand
                            }
                        }
                    }
                }
            }
            scan_radical_generator(base, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan_radical_generator(a, var, pool, out);
            }
        }
        _ => {}
    }
}

/// Decompose `expr` as `Σⱼ cⱼ(x) yʲ ∈ ℚ(x)[y]/(yⁿ − p)` where `y = p^{1/n}`,
/// returning the coefficient vector as an [`AlgElem`].  The degree-`n`
/// generalization of [`decompose_over_alpha`].  Returns `None` if `expr` is not
/// a polynomial-in-`y` over `ℚ(x)` with this single generator.
pub(super) fn decompose_over_alg_generator(
    expr: ExprId,
    n: usize,
    p_radicand: &QPoly,
    e: &AlgExtension,
    var: ExprId,
    pool: &ExprPool,
) -> Option<AlgElem> {
    use crate::kernel::ExprData;

    // Generator-free → a pure rational function of x → a constant element.
    if let Some((num, den)) = expr_to_qrational(expr, var, pool) {
        return Some(e.constant(RatFn::new(num, den)));
    }

    // `base^{a/d}` with `base` the radicand and `d | n` → `y^{a·(n/d)}`.
    let as_generator_power = |base: ExprId, numr: i64, den: i64| -> Option<AlgElem> {
        if den >= 2 && (n as i64) % den == 0 {
            let bp = expr_to_qpoly(base, var, pool)?;
            if trim(bp) == trim(p_radicand.clone()) {
                return e.pow(&e.generator(), numr * (n as i64 / den));
            }
        }
        None
    };

    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut acc = e.from_int(0);
            for &a in &args {
                let t = decompose_over_alg_generator(a, n, p_radicand, e, var, pool)?;
                acc = e.add(&acc, &t);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = e.from_int(1);
            for &a in &args {
                let t = decompose_over_alg_generator(a, n, p_radicand, e, var, pool)?;
                acc = e.mul(&acc, &t);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let m = m.0.to_i64()?;
                let b = decompose_over_alg_generator(base, n, p_radicand, e, var, pool)?;
                e.pow(&b, m)
            }
            ExprData::Rational(r) => {
                as_generator_power(base, r.0.numer().to_i64()?, r.0.denom().to_i64()?)
            }
            _ => None,
        },
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            as_generator_power(args[0], 1, 2)
        }
        ExprData::Func { ref name, ref args } if name == "cbrt" && args.len() == 1 => {
            as_generator_power(args[0], 1, 3)
        }
        _ => None,
    }
}

/// Detect the radical generator in `expr` and decompose it over the
/// corresponding [`AlgExtension`].  The MA entry point: returns the extension
/// `ℚ(x)[y]/(yⁿ − p)` together with `expr` written as an [`AlgElem`].
#[allow(dead_code)] // consumed by MA (mixed alg+trans integral part, M1)
fn decompose_radical(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(AlgExtension, AlgElem)> {
    let (n, p) = detect_radical_generator(expr, var, pool)?;
    let e = AlgExtension::radical(n, &p);
    let elem = decompose_over_alg_generator(expr, n, &p, &e, var, pool)?;
    Some((e, elem))
}

/// Try to integrate `c_rest · exp(kη)` when `c_rest` contains a single
/// x-dependent `sqrt(p(x))` algebraic generator (Gap C).
///
/// Uses the twisted-derivation decomposition:  for v = a + b·α (α = √p),
///   D(v) + f·v = c  ⟺  a' + f·a = c₀   ∧   b' + (f + p'/(2p))·b = c₁
/// where c = c₀ + c₁·α and f = k·η' is a polynomial.  Returns `None` when
/// `c_rest` is not of this form; `Some(Err(NonElementary))` if either RDE
/// has no rational solution; `Some(Ok(result))` on success.
#[allow(clippy::too_many_arguments)]
fn try_sqrt_poly_rde(
    c_rest: ExprId,
    k: i64,
    deta: &[rug::Rational],
    exp_k_eta: ExprId,
    k_const: ExprId,
    c_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<Result<ExprId, IntegrationError>> {
    // Must have exactly one x-dependent sqrt generator.
    let (p_poly, alpha) = detect_sqrt_of_poly(c_rest, var, pool)?;

    // Decompose c_rest = c0 + c1·alpha.
    let (c0, c1) = decompose_over_alpha(c_rest, alpha, &p_poly, var, pool)?;

    // f = k·η' (polynomial).
    let f: QPoly = deta
        .iter()
        .map(|r| r.clone() * rug::Rational::from(k))
        .collect();

    let ne = || {
        IntegrationError::NonElementary(format!(
            "the Risch DE over ℚ(x)(√({})) for ∫ {} · exp(η)^{k} dx \
             has no rational solution",
            pool.display(pool.func("placeholder", vec![])), // not shown
            pool.display(c_expr),
        ))
    };

    // Equation 1: a' + f·a = c₀.
    let a_sol = match solve_rational_rde(&f, &c0.0, &c0.1) {
        Some(s) => s,
        None => return Some(Err(ne())),
    };

    // Equation 2: b' + (f + p'/(2p))·b = c₁.
    // f_eff = f + p'/(2p) = (2f·p + p') / (2p).
    let p_prime = poly_deriv(&p_poly);
    let f_eff_num = poly_add(
        &poly_scale(&poly_mul(&f, &p_poly), &rug::Rational::from(2)),
        &p_prime,
    );
    let f_eff_den = poly_scale(&p_poly, &rug::Rational::from(2));
    let b_sol = match solve_rational_rde_generalized(&f_eff_num, &f_eff_den, &c1.0, &c1.1) {
        Some(s) => s,
        None => return Some(Err(ne())),
    };

    // Reconstruct v = a + b·alpha.
    let a_expr = build_rational(&a_sol.0, &a_sol.1, var, pool);
    let b_expr = build_rational(&b_sol.0, &b_sol.1, var, pool);

    let a_zero = trim(a_sol.0.clone()).is_empty();
    let b_zero = trim(b_sol.0.clone()).is_empty();

    let v_expr = match (a_zero, b_zero) {
        (true, true) => pool.integer(0_i32),
        (true, false) => pool.mul(vec![b_expr, alpha]),
        (false, true) => a_expr,
        (false, false) => pool.add(vec![a_expr, pool.mul(vec![b_expr, alpha])]),
    };

    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
    let result = apply_const(k_const, core, pool);
    log.push(RewriteStep::simple("risch_exp_sqrt_poly", c_expr, result));
    Some(Ok(result))
}

/// Try to integrate `c_rest · exp(kη)` when `c_rest` contains a single
/// x-dependent **degree-`n` radical** generator `y = a(x)^{1/n}` (`n ≥ 3`,
/// `a ∈ ℚ(x)` squarefree).  The degree-`n` generalization of
/// [`try_sqrt_poly_rde`], built on the M0 `decompose_radical` substrate.
///
/// Seeking the antiderivative `v·exp(kη)` with `v = Σᵢ vᵢ yⁱ`, the equation
/// `D(v) + f·v = c_rest` (`f = kη'`) decouples per power `i` into
/// `vᵢ' + (f + (i/n)·a'/a)·vᵢ = cᵢ` — a (generalized) rational Risch DE over
/// `ℚ(x)`.  No solution for some component ⇒ `NonElementary`.
///
/// Scope: the radicand `a` lives in `ℚ(x)` (constant in the exponential), so
/// each component RDE has `ℚ(x)` coefficients.  A radicand that itself involves
/// the transcendental (`∛(x+eˣ)`) needs the full tower recursion (MD) and is not
/// handled here.
#[allow(clippy::too_many_arguments)]
fn try_radical_poly_rde(
    c_rest: ExprId,
    k: i64,
    deta: &[rug::Rational],
    exp_k_eta: ExprId,
    k_const: ExprId,
    c_expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Option<Result<ExprId, IntegrationError>> {
    let (n, a) = detect_radical_generator(c_rest, var, pool)?;
    if n < 3 {
        return None; // degree 2 handled by try_sqrt_poly_rde
    }
    let a = trim(a);
    if degree(&a) < 1 {
        return None;
    }
    let a_prime = poly_deriv(&a);
    // Squarefree radicand → power basis with ωᵢ = (i/n)·a'/a.  Non-squarefree
    // radicands in the mixed case are not handled yet.
    if degree(&poly_gcd(&a, &a_prime)) >= 1 {
        return None;
    }

    let e = AlgExtension::radical(n, &a);
    let elem = decompose_over_alg_generator(c_rest, n, &a, &e, var, pool)?;

    // f = k·η' (polynomial).
    let f: QPoly = deta
        .iter()
        .map(|r| r.clone() * rug::Rational::from(k))
        .collect();

    let a_expr = qpoly_to_expr(&a, var, pool);
    let ne = || {
        IntegrationError::NonElementary(format!(
            "the Risch DE over ℚ(x)({}^(1/{n})) for ∫ {} · exp(η)^{k} dx \
             has no rational solution",
            pool.display(a_expr),
            pool.display(c_expr),
        ))
    };

    let mut terms: Vec<ExprId> = Vec::new();
    for i in 0..n {
        // cᵢ = numᵢ/denᵢ : the yⁱ-coefficient of c_rest.
        let (c_num, c_den) = match elem.get(i) {
            Some(r) => (r.numer().clone(), r.denom().clone()),
            None => (QPoly::new(), poly_one()),
        };
        if trim(c_num.clone()).is_empty() {
            continue;
        }
        let (vn, vd) = if i == 0 {
            // v₀' + f·v₀ = c₀.
            match solve_rational_rde(&f, &c_num, &c_den) {
                Some(s) => s,
                None => return Some(Err(ne())),
            }
        } else {
            // f_eff = f + (i/n)·a'/a = (n·a·f + i·a') / (n·a).
            let f_eff_num = poly_add(
                &poly_scale(&poly_mul(&f, &a), &rug::Rational::from(n as i64)),
                &poly_scale(&a_prime, &rug::Rational::from(i as i64)),
            );
            let f_eff_den = poly_scale(&a, &rug::Rational::from(n as i64));
            match solve_rational_rde_generalized(&f_eff_num, &f_eff_den, &c_num, &c_den) {
                Some(s) => s,
                None => return Some(Err(ne())),
            }
        };
        if trim(vn.clone()).is_empty() {
            continue;
        }
        let v_expr = build_rational(&vn, &vd, var, pool);
        if i == 0 {
            terms.push(v_expr);
        } else {
            let yi = pool.pow(a_expr, pool.rational(i as i32, n as i32));
            terms.push(pool.mul(vec![v_expr, yi]));
        }
    }

    let v_expr = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };
    let core = build_v_times_exp(v_expr, exp_k_eta, pool);
    let result = apply_const(k_const, core, pool);
    log.push(RewriteStep::simple(
        "risch_exp_radical_poly",
        c_expr,
        result,
    ));
    Some(Ok(result))
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
                // Linear η (degree 1): basic engine handles exp(a·x+b) alone.
                return false;
            }
            // Non-polynomial η that depends on var (e.g. 1/x, 1/(x²+1)):
            // route to the Risch exp-tower path (Gap F).  For rational η the
            // generalised RDE solver handles it or certifies NonElementary; for
            // genuinely transcendental η it falls through to NotImplemented.
            true
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
                            // Non-polynomial η (e.g. 1/x): treat as nonlinear
                            // so the Risch path is taken regardless of the
                            // polynomial degree of the surrounding coefficient.
                            has_nonlinear_exp = true;
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
            // Gap C: any exp generator with an x-dependent algebraic factor
            // (e.g. sqrt(x²+1) or x^{1/2}) — route to the Risch exp-tower path.
            let has_var_alg = args.iter().any(|&a| contains_var_algebraic(a, var, pool));
            if (has_linear_exp || has_nonlinear_exp) && has_var_alg {
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

    // -- degree-d radical decomposition (decompose_over_alpha generalization) --

    fn qp(coeffs: &[i64]) -> QPoly {
        coeffs.iter().map(|&c| rug::Rational::from(c)).collect()
    }

    #[test]
    fn detect_radical_generator_forms() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        // cbrt(x) → (3, x).
        let cbrt_x = pool.func("cbrt", vec![x]);
        assert_eq!(
            detect_radical_generator(cbrt_x, x, &pool),
            Some((3, qp(&[0, 1])))
        );
        // x^(1/3) → (3, x).
        let x_pow = pool.pow(x, pool.rational(1_i32, 3_i32));
        assert_eq!(
            detect_radical_generator(x_pow, x, &pool),
            Some((3, qp(&[0, 1])))
        );
        // sqrt(x+1) → (2, x+1).
        let sqrt = pool.func("sqrt", vec![pool.add(vec![x, pool.integer(1_i32)])]);
        assert_eq!(
            detect_radical_generator(sqrt, x, &pool),
            Some((2, qp(&[1, 1])))
        );
        // No radical → None.
        assert_eq!(
            detect_radical_generator(pool.add(vec![x, pool.integer(1_i32)]), x, &pool),
            None
        );
        // Two distinct generators (compositum) → None.
        let mixed = pool.add(vec![pool.func("sqrt", vec![x]), pool.func("cbrt", vec![x])]);
        assert_eq!(detect_radical_generator(mixed, x, &pool), None);
    }

    #[test]
    fn decompose_x_plus_cbrt_x() {
        // x + ∛x  over  ℚ(x)[y]/(y³ − x)  →  [x, 1, 0].
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.func("cbrt", vec![x])]);
        let (e, elem) = decompose_radical(expr, x, &pool).expect("decomposes");
        assert_eq!(e.degree(), 3);
        let expected = vec![RatFn::from_poly(&qp(&[0, 1])), RatFn::int(1)];
        assert!(e.elem_eq(&elem, &expected), "got {elem:?}");
    }

    #[test]
    fn decompose_cbrt_x_squared() {
        // (∛x)²  →  y²  →  [0, 0, 1];  and x^(2/3) gives the same.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sq = pool.pow(pool.func("cbrt", vec![x]), pool.integer(2_i32));
        let (e, elem) = decompose_radical(sq, x, &pool).expect("decomposes");
        let expected = vec![RatFn::int(0), RatFn::int(0), RatFn::int(1)];
        assert!(e.elem_eq(&elem, &expected), "got {elem:?}");

        let frac = pool.pow(x, pool.rational(2_i32, 3_i32));
        let (e2, elem2) = decompose_radical(frac, x, &pool).expect("decomposes");
        assert!(e2.elem_eq(&elem2, &expected), "x^(2/3): got {elem2:?}");
    }

    #[test]
    fn decompose_inverse_of_one_plus_cbrt_x() {
        // 1/(1 + ∛x) over ℚ(x)(∛x): the decomposition inverts (1+y); check by
        // multiplying back to 1.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let base = pool.add(vec![pool.integer(1_i32), pool.func("cbrt", vec![x])]);
        let inv_expr = pool.pow(base, pool.integer(-1_i32));
        let (e, inv_elem) = decompose_radical(inv_expr, x, &pool).expect("decomposes");
        let base_elem = vec![RatFn::int(1), RatFn::int(1)]; // 1 + y
        let product = e.mul(&inv_elem, &base_elem);
        assert!(
            e.elem_eq(&product, &e.from_int(1)),
            "inv·(1+y) = {product:?}"
        );
    }

    #[test]
    fn decompose_degree2_matches_kpair_semantics() {
        // x + √x over ℚ(x)(√x) → [x, 1] — the rank-2 KPair (c₀=x, c₁=1) as an
        // AlgElem of length 2.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.func("sqrt", vec![x])]);
        let (e, elem) = decompose_radical(expr, x, &pool).expect("decomposes");
        assert_eq!(e.degree(), 2);
        let expected = vec![RatFn::from_poly(&qp(&[0, 1])), RatFn::int(1)];
        assert!(e.elem_eq(&elem, &expected), "got {elem:?}");
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
    fn mixed_cbrt_times_exp_elementary() {
        // ∫ (1 + 1/(3x))·x^{1/3}·exp(x) dx = x^{1/3}·exp(x).
        // Degree-3 radical × exp: D(x^{1/3}·eˣ) = (1/(3x)·x^{1/3} + x^{1/3})eˣ.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x13 = pool.pow(x, pool.rational(1_i32, 3_i32));
        let inv_3x = pool.mul(vec![
            pool.rational(1_i32, 3_i32),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let coeff = pool.add(vec![pool.integer(1_i32), inv_3x]);
        let integrand = pool.mul(vec![coeff, x13, pool.func("exp", vec![x])]);
        verify_exp_tower(integrand, x, &pool);
    }

    #[test]
    fn cbrt_times_exp_is_nonelementary() {
        // ∫ x^{1/3}·exp(x) dx is non-elementary (incomplete-gamma family).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x13 = pool.pow(x, pool.rational(1_i32, 3_i32));
        let integrand = pool.mul(vec![x13, pool.func("exp", vec![x])]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        let level = gens.iter().find(|g| g.is_exp()).expect("an exp generator");
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ x^(1/3)·exp(x) dx should be NonElementary, got: {result:?}"
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

    // -----------------------------------------------------------------------
    // Constant (symbolic / algebraic) coefficient factor (Gap E, exp tower)
    // -----------------------------------------------------------------------

    /// Numeric evaluator supporting the nodes that appear in these antiderivatives
    /// (Integer/Rational/Add/Mul/Pow/exp/sqrt).
    fn eval_f64(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        use crate::kernel::ExprData;
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            // Opaque symbolic constant used in the symbolic-factor test: assign it
            // a fixed value so `d/dx F = f` can be checked numerically.
            ExprData::Symbol { ref name, .. } if name == "pi" => std::f64::consts::PI,
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
                    "exp" => a.exp(),
                    "log" => a.ln(),
                    "sqrt" => a.sqrt(),
                    other => panic!("eval_f64: unsupported func {other}"),
                }
            }
            other => panic!("eval_f64: unsupported node {other:?}"),
        }
    }

    /// Integrate via the exp tower and assert `d/dx F = integrand` numerically.
    fn verify_exp_tower(integrand: ExprId, x: ExprId, pool: &ExprPool) {
        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, pool);
        let level = gens.iter().find(|g| g.is_exp()).expect("an exp generator");
        let mut log = DerivationLog::new();
        let f =
            integrate_exp_tower(integrand, level, x, pool, &mut log).expect("should be elementary");
        let d = crate::diff::diff(f, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        for &xv in &[0.7_f64, 1.3, 2.1] {
            let lhs = eval_f64(ds, x, xv, pool);
            let rhs = eval_f64(integrand, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\n  F = {}",
                pool.display(f)
            );
        }
    }

    #[test]
    fn rational_const_factor_exp_x2() {
        // ∫ (1/2)·x·exp(x²) dx = (1/4)·exp(x²).  Before the constant-factor split
        // the ℚ-constant rode along in the coefficient and the conversion failed.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let exp_x2 = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        let half = pool.rational(1_i32, 2_i32);
        let integrand = pool.mul(vec![half, x, exp_x2]);
        verify_exp_tower(integrand, x, &pool);
    }

    #[test]
    fn algebraic_const_factor_exp_x2() {
        // ∫ √2·x·exp(x²) dx = (√2/2)·exp(x²).  √2 is an algebraic constant the
        // ℚ-coefficient RDE cannot represent; the split pulls it out (Gap E).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let exp_x2 = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let integrand = pool.mul(vec![sqrt2, x, exp_x2]);
        verify_exp_tower(integrand, x, &pool);
    }

    #[test]
    fn symbolic_const_factor_poly_exp_x() {
        // ∫ π·x²·exp(x) dx = π·(x²−2x+2)·exp(x).  π is an opaque symbolic constant.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let pi = pool.symbol("pi", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let integrand = pool.mul(vec![pi, pool.pow(x, pool.integer(2_i32)), exp_x]);
        verify_exp_tower(integrand, x, &pool);
    }

    #[test]
    fn algebraic_const_factor_rational_coeff() {
        // ∫ √2·(x−1)/x²·exp(x) dx = √2·exp(x)/x.  Constant factor on top of a
        // *rational* coefficient (exercises the rational-RDE branch after the split).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let num = pool.add(vec![x, pool.integer(-1_i32)]);
        let inv_x2 = pool.pow(x, pool.integer(-2_i32));
        let integrand = pool.mul(vec![sqrt2, num, inv_x2, exp_x]);
        verify_exp_tower(integrand, x, &pool);
    }

    #[test]
    fn algebraic_const_factor_nonelementary_preserved() {
        // ∫ √2·exp(x²) dx is still non-elementary: pulling out √2 leaves ∫exp(x²).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let exp_x2 = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let integrand = pool.mul(vec![sqrt2, exp_x2]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        let level = gens.iter().find(|g| g.is_exp()).unwrap();
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ √2·exp(x²) dx must remain NonElementary; got {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Algebraic-number coefficients ℚ(√d) entangled with x (Gap E, poly RDE)
    // -----------------------------------------------------------------------

    #[test]
    fn algebraic_coeff_linear_exp_x() {
        // ∫ (x + √2)·exp(x) dx = (x + √2 − 1)·exp(x).  The coefficient (x + √2)
        // lives in ℚ(√2)[x] and cannot be split as a constant factor.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let exp_x = pool.func("exp", vec![x]);
        let coeff = pool.add(vec![x, sqrt2]);
        let integrand = pool.mul(vec![coeff, exp_x]);
        verify_exp_tower(integrand, x, &pool);
    }

    #[test]
    fn algebraic_coeff_quadratic_exp_x() {
        // ∫ (√3·x² + x)·exp(x) dx — √3 entangled with x².
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt3 = pool.func("sqrt", vec![pool.integer(3_i32)]);
        let exp_x = pool.func("exp", vec![x]);
        let coeff = pool.add(vec![
            pool.mul(vec![sqrt3, pool.pow(x, pool.integer(2_i32))]),
            x,
        ]);
        let integrand = pool.mul(vec![coeff, exp_x]);
        verify_exp_tower(integrand, x, &pool);
    }

    #[test]
    fn algebraic_coeff_nonelementary_preserved() {
        // ∫ (x + √2)·exp(x²) dx is non-elementary: the √2·exp(x²) part has no
        // elementary antiderivative, so the RDE over ℚ(√2) has no solution.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let exp_x2 = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        let coeff = pool.add(vec![x, sqrt2]);
        let integrand = pool.mul(vec![coeff, exp_x2]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        let level = gens.iter().find(|g| g.is_exp()).unwrap();
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ (x+√2)·exp(x²) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn algebraic_rational_coeff_exp_x() {
        // ∫ (x − √2 − 1)/(x − √2)² · exp(x) dx = exp(x)/(x − √2).
        // The coefficient is a rational function over ℚ(√2) (RDE v'+v = c → v =
        // 1/(x−√2)); exercises the rational RDE over the number field.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let neg_sqrt2 = pool.mul(vec![pool.integer(-1_i32), sqrt2]);
        let base = pool.add(vec![x, neg_sqrt2]); // x − √2
        let num = pool.add(vec![x, neg_sqrt2, pool.integer(-1_i32)]); // x − √2 − 1
        let exp_x = pool.func("exp", vec![x]);
        let integrand = pool.mul(vec![num, pool.pow(base, pool.integer(-2_i32)), exp_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        let level = gens.iter().find(|g| g.is_exp()).expect("exp generator");
        let mut log = DerivationLog::new();
        let f = integrate_exp_tower(integrand, level, x, &pool, &mut log)
            .expect("∫ (x−√2−1)/(x−√2)²·exp(x) dx should be elementary");
        // Verify d/dx F = integrand at points away from the singularity x = √2.
        let d = crate::diff::diff(f, x, &pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, &pool).value;
        for &xv in &[2.5_f64, 3.3, 4.1] {
            let lhs = eval_f64(ds, x, xv, &pool);
            let rhs = eval_f64(integrand, x, xv, &pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\n  F = {}",
                pool.display(f)
            );
        }
    }

    #[test]
    fn algebraic_rational_coeff_nonelementary() {
        // ∫ x²/(x − √2) · exp(x) dx leaves an Ei term (simple pole at √2 with
        // nonzero residue) — non-elementary over ℚ(√2).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let neg_sqrt2 = pool.mul(vec![pool.integer(-1_i32), sqrt2]);
        let base = pool.add(vec![x, neg_sqrt2]); // x − √2
        let exp_x = pool.func("exp", vec![x]);
        let integrand = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(base, pool.integer(-1_i32)),
            exp_x,
        ]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        let level = gens.iter().find(|g| g.is_exp()).unwrap();
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ x²/(x−√2)·exp(x) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn detect_sqrt_field_cases() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);

        // x + √2 → field ℚ(√2).
        let e = pool.add(vec![x, sqrt2]);
        assert_eq!(detect_sqrt_field(e, &pool).map(|(d, _)| d), Some(2));

        // √4 = 2 is a perfect square → not a field extension.
        let sqrt4 = pool.func("sqrt", vec![pool.integer(4_i32)]);
        let e = pool.add(vec![x, sqrt4]);
        assert_eq!(detect_sqrt_field(e, &pool), None);

        // Two distinct radicals (compositum) → out of scope.
        let sqrt3 = pool.func("sqrt", vec![pool.integer(3_i32)]);
        let e = pool.add(vec![sqrt2, sqrt3]);
        assert_eq!(detect_sqrt_field(e, &pool), None);

        // No radical at all.
        let e = pool.add(vec![x, pool.integer(1_i32)]);
        assert_eq!(detect_sqrt_field(e, &pool), None);
    }

    #[test]
    fn split_const_factor_cases() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);

        // √2·x → (√2, x)
        let c = pool.mul(vec![sqrt2, x]);
        let (k, rest) = split_const_factor(c, x, &pool);
        assert_eq!(k, sqrt2);
        assert_eq!(rest, x);

        // pure constant √2 → (√2, 1)
        let (k, rest) = split_const_factor(sqrt2, x, &pool);
        assert_eq!(k, sqrt2);
        assert_eq!(rest, pool.integer(1_i32));

        // pure variable x → (1, x)
        let (k, rest) = split_const_factor(x, x, &pool);
        assert_eq!(k, pool.integer(1_i32));
        assert_eq!(rest, x);
    }

    // -----------------------------------------------------------------------
    // Gap F: rational exponents  exp(η),  η ∈ ℚ(x) \ ℚ[x]
    // -----------------------------------------------------------------------

    /// Numeric evaluator for Gap-F verification (supports exp).
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
                    "cbrt" => a.cbrt(),
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
                "d/dx F ≠ f at x={xv}: got {lhs}, expected {rhs}\n  F = {}",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn exp_inv_x_nonelementary() {
        // ∫ exp(1/x) dx is non-elementary.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let inv_x = pool.pow(x, pool.integer(-1_i32)); // 1/x
        let f = pool.func("exp", vec![inv_x]);

        assert!(
            needs_exp_risch(f, x, &pool),
            "exp(1/x) should be routed to Risch"
        );

        use super::super::tower::find_generators;
        let gens = find_generators(f, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(f, level, x, &pool, &mut log);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ exp(1/x) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn inv_x2_times_exp_inv_x_elementary() {
        // ∫ (1/x²)·exp(1/x) dx = −exp(1/x).
        // d/dx(−exp(1/x)) = exp(1/x)·(1/x²)  ✓
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let exp_inv_x = pool.func("exp", vec![inv_x]);
        let inv_x2 = pool.pow(x, pool.integer(-2_i32));
        let integrand = pool.mul(vec![inv_x2, exp_inv_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            result.is_ok(),
            "∫ (1/x²)·exp(1/x) dx must be elementary; got {result:?}"
        );
        verify_numeric_gapf(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn two_inv_x3_times_exp_neg_inv_x2_elementary() {
        // ∫ (2/x³)·exp(−1/x²) dx = exp(−1/x²).
        // d/dx(exp(−1/x²)) = exp(−1/x²)·(2/x³)  ✓
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let neg_inv_x2 = pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(x, pool.integer(-2_i32)),
        ]);
        let exp_neg_inv_x2 = pool.func("exp", vec![neg_inv_x2]);
        let two_inv_x3 = pool.mul(vec![pool.integer(2_i32), pool.pow(x, pool.integer(-3_i32))]);
        let integrand = pool.mul(vec![two_inv_x3, exp_neg_inv_x2]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];

        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            result.is_ok(),
            "∫ (2/x³)·exp(−1/x²) dx must be elementary; got {result:?}"
        );
        verify_numeric_gapf(integrand, result.unwrap(), x, &pool);
    }

    #[test]
    fn detection_rational_eta() {
        // needs_exp_risch should detect exp(1/x), exp(1/(x²+1)), etc.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);

        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let exp_inv_x = pool.func("exp", vec![inv_x]);
        assert!(
            needs_exp_risch(exp_inv_x, x, &pool),
            "exp(1/x) should need Risch"
        );

        // exp(x) alone: still handled by basic engine (not Risch).
        let exp_x = pool.func("exp", vec![x]);
        assert!(
            !needs_exp_risch(exp_x, x, &pool),
            "exp(x) alone should NOT route to Risch"
        );

        // x·exp(1/x): coefficient times rational-exp needs Risch.
        let x_exp_inv_x = pool.mul(vec![x, exp_inv_x]);
        assert!(
            needs_exp_risch(x_exp_inv_x, x, &pool),
            "x·exp(1/x) should need Risch"
        );
    }

    // -----------------------------------------------------------------------
    // Gap E extended: compositum ℚ(√a,√b) and n-th root ℚ(n^(1/m))
    // -----------------------------------------------------------------------

    #[test]
    fn compositum_two_sqrts_exp_x2_elementary() {
        // ∫ (x + √2 + √3) · exp(x²) dx
        // RDE: v' + 2x·v = x + √2 + √3.  Solution: v = 1/2 + (√2+√3)·(−1/(2x²)·…
        // Actually, split into two integrals:
        //   ∫ x·exp(x²) dx = ½exp(x²)       (standard)
        //   ∫ (√2+√3)·exp(x²) dx: non-elementary (exp(x²) factor, const coeff)
        // Wait — (√2+√3) is a constant, so split_const_factor handles it already
        // as K_const=(√2+√3), c_rest=1. But ∫ exp(x²) dx is non-elementary.
        // So the whole thing is non-elementary for the constant part.
        // The integrand x·exp(x²) + (√2+√3)·exp(x²) = (x + √2 + √3)·exp(x²).
        // Only the x term is integrable (v=1/2), the (√2+√3) term is not.
        // This should return NonElementary.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let sqrt3 = pool.func("sqrt", vec![pool.integer(3_i32)]);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let exp_x2 = pool.func("exp", vec![x2]);
        let coeff = pool.add(vec![x, sqrt2, sqrt3]);
        let integrand = pool.mul(vec![coeff, exp_x2]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        // (√2+√3) is free of x → split_const_factor → c_rest=1 → NE for exp(x²)
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ (x+√2+√3)·exp(x²) dx: const-offset term is non-elementary; got {result:?}"
        );
    }

    #[test]
    fn compositum_two_sqrts_exp_x_elementary() {
        // ∫ (x + √2 + √3) · exp(x) dx = (x − 1 + √2 + √3)·exp(x).
        // RDE (k=1, η'=1): v' + v = x + √2 + √3.
        // Solve over ℚ(√2,√3) [the compositum]: undetermined coefficients.
        // v = ax + b + c: (a + ax + b + c) + ...
        // Actually v = x - 1 + √2 + √3:
        //   v' + v = 1 + (x-1+√2+√3) = x + √2 + √3 ✓
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let sqrt3 = pool.func("sqrt", vec![pool.integer(3_i32)]);
        let exp_x = pool.func("exp", vec![x]);
        let coeff = pool.add(vec![x, sqrt2, sqrt3]);
        let integrand = pool.mul(vec![coeff, exp_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            result.is_ok(),
            "∫ (x+√2+√3)·exp(x) dx must be elementary; got {result:?}"
        );
        // Verify numerically: d/dx F = integrand.
        let antideriv = result.unwrap();
        verify_numeric_gapf(integrand, antideriv, x, &pool);
    }

    #[test]
    fn nth_root_cbrt3_exp_x_elementary() {
        // ∫ (x + cbrt(3)) · exp(x) dx = (x − 1 + cbrt(3))·exp(x).
        // RDE v' + v = x + cbrt(3).  Solution v = x − 1 + cbrt(3).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        // cbrt(3) as a function node.
        let cbrt3 = pool.func("cbrt", vec![pool.integer(3_i32)]);
        let exp_x = pool.func("exp", vec![x]);
        let coeff = pool.add(vec![x, cbrt3]);
        let integrand = pool.mul(vec![coeff, exp_x]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        assert_eq!(gens.len(), 1);
        let level = &gens[0];
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            result.is_ok(),
            "∫ (x+cbrt(3))·exp(x) dx must be elementary; got {result:?}"
        );
        // Verify d/dx F ≈ f numerically via finite differences.
        // (The symbolic diff engine doesn't support cbrt, so we avoid it.)
        let cbrt3_v: f64 = 3.0f64.powf(1.0 / 3.0);
        let f = result.unwrap();
        let eval = |expr: ExprId, xv: f64| -> f64 { eval_f64_gapf(expr, x, xv, &pool) };
        let h = 1e-6_f64;
        for &xv in &[0.5_f64, 1.2, 2.7] {
            let fd = (eval(f, xv + h) - eval(f, xv - h)) / (2.0 * h);
            let exact = (xv + cbrt3_v) * xv.exp();
            assert!(
                (fd - exact).abs() < 1e-5,
                "finite-diff check at x={xv}: fd={fd}, exact={exact}"
            );
        }
    }

    #[test]
    fn nth_root_pow_1_3_exp_x2_nonelementary() {
        // ∫ 2^(1/3) · exp(x²) dx — non-elementary (like ∫ exp(x²) dx).
        // 2^(1/3) is a constant factor: split_const_factor extracts it, then
        // solve_poly_rde for v'+2xv=1 fails → NonElementary.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let cbrt2 = pool.pow(
            pool.integer(2_i32),
            pool.rational(rug::Integer::from(1), rug::Integer::from(3)),
        );
        let x2 = pool.pow(x, pool.integer(2_i32));
        let integrand = pool.mul(vec![cbrt2, pool.func("exp", vec![x2])]);

        use super::super::tower::find_generators;
        let gens = find_generators(integrand, x, &pool);
        let level = &gens[0];
        let mut log = DerivationLog::new();
        let result = integrate_exp_tower(integrand, level, x, &pool, &mut log);
        assert!(
            matches!(result, Err(IntegrationError::NonElementary(_))),
            "∫ 2^(1/3)·exp(x²) dx must be NonElementary; got {result:?}"
        );
    }

    #[test]
    fn detect_algebraic_extension_cases() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let sqrt2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let sqrt3 = pool.func("sqrt", vec![pool.integer(3_i32)]);
        let cbrt5 = pool.func("cbrt", vec![pool.integer(5_i32)]);

        // Single sqrt → SingleSqrt.
        let e1 = pool.add(vec![x, sqrt2]);
        assert!(
            matches!(
                detect_algebraic_extension(e1, &pool),
                Some(AlgebraicExtension::SingleSqrt { d: 2, .. })
            ),
            "x+√2 should give SingleSqrt(2)"
        );

        // Two sqrts → CompositumTwoSqrts.
        let e2 = pool.add(vec![x, sqrt2, sqrt3]);
        assert!(
            matches!(
                detect_algebraic_extension(e2, &pool),
                Some(AlgebraicExtension::CompositumTwoSqrts { a: 2, b: 3, .. })
            ),
            "x+√2+√3 should give CompositumTwoSqrts(2,3)"
        );

        // cbrt → NthRoot with m=3.
        let e3 = pool.add(vec![x, cbrt5]);
        assert!(
            matches!(
                detect_algebraic_extension(e3, &pool),
                Some(AlgebraicExtension::NthRoot { n: 5, m: 3, .. })
            ),
            "x+cbrt(5) should give NthRoot(5,3)"
        );

        // No radical → None.
        assert!(detect_algebraic_extension(x, &pool).is_none());
    }
}
