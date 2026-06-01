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

use super::number_field::{KElem, KPoly, NumberField};
use super::poly_rde::{
    apply_const, expr_to_qpoly, is_free_of_var, poly_one, poly_scale, qpoly_to_expr,
    rational_to_expr, solve_poly_rde, solve_poly_rde_k, split_const_factor, trim, QPoly,
};
use super::rational_rde::{
    expr_to_qrational, solve_rational_rde, solve_rational_rde_generalized, solve_rational_rde_k,
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
            return Err(IntegrationError::NotImplemented(format!(
                "exponent derivative η'(x) = {} is not a rational function in {}; \
                 only polynomial and rational exponents are supported",
                pool.display(deta_expr),
                pool.display(var),
            )));
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

    // The var-dependent remainder is outside the supported algebraic subsets.
    Err(IntegrationError::NotImplemented(format!(
        "coefficient {} of exp(η)^{} is not a polynomial or rational function over \
         a supported algebraic extension; mixed/nested generators are not yet supported",
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

// ---------------------------------------------------------------------------
// Gap E extended: compositum ℚ(√a,√b) and n-th root ℚ(n^(1/m)) detection
// ---------------------------------------------------------------------------

/// Describes an algebraic extension ℚ(α) detected in a coefficient expression.
#[derive(Debug, Clone)]
enum AlgebraicExtension {
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
fn detect_algebraic_extension(expr: ExprId, pool: &ExprPool) -> Option<AlgebraicExtension> {
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
fn build_field_and_gens(ext: &AlgebraicExtension) -> (NumberField, Vec<(ExprId, KElem)>) {
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
fn expr_to_krational_general(
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
fn build_krational_ext(
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
