//! Integration formulas for genus-0 algebraic extensions.
//!
//! Handles `∫ B(x) · sqrt(P(x)) dx` when P has degree ≤ 2 (genus-0 curve).
//! For degree ≥ 3 with a **polynomial** weight `B`, the *integral part*
//! `∫ B√P = Q·√P` is solved when it exists (Liouville), so elementary cases such
//! as `∫ (P'/2)·√P = ⅓P^{3/2}` are returned rather than wrongly rejected;
//! otherwise (with P squarefree) the integral is genuinely `NonElementary`.
//!
//! Reference: Bronstein (2005) §6.3–6.5; standard CAS table integrals.

use super::poly_utils::{
    as_integer, as_linear, as_quadratic, is_free_of, poly_degree_in, poly_int_coeffs,
};
use crate::deriv::log::{DerivationLog, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::integrate::risch::alg_field::{AlgElem, RatFn};
use crate::integrate::risch::number_field::KElem;
use crate::integrate::risch::poly_rde::{
    degree, expr_to_qpoly, poly_add, poly_deriv, poly_mul, poly_scale, qpoly_to_expr, trim, QPoly,
};
use crate::integrate::risch::rational_rde::{
    expr_to_qrational, poly_divrem, poly_gcd, solve_rational_rde_generalized,
};
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::{Integer, Rational};

use super::find_order::{find_order_placed, FindOrder};
use super::jacobian_torsion::AlgPlace;
use super::residues::{
    finite_residues_algebraic, residue_divisor_placed, residue_sum_complete, AlgResidue,
};
use super::trager_log::trager_log_criterion_alg;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Integrate `B(x) · sqrt(P(x))` with respect to `var`.
///
/// `sqrt_id` is the ExprId of the sqrt expression (used for building results).
/// Returns `Err(NonElementary)` if P has degree ≥ 3 (elliptic/hyperelliptic).
pub fn integrate_with_sqrt(
    b: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // Determine degree of P in var
    let p_deg = poly_degree_in(p, var, pool).unwrap_or(u32::MAX);

    match p_deg {
        0 => integrate_b_sqrt_const(b, p, sqrt_id, var, pool, log),
        1 => integrate_b_sqrt_linear(b, p, sqrt_id, var, pool, log),
        2 => integrate_b_sqrt_quadratic(b, p, sqrt_id, var, pool, log),
        _ => {
            // Check degree using UniPoly for accuracy
            let actual_deg = poly_int_coeffs(p, var, pool)
                .map(|cs| cs.len().saturating_sub(1))
                .unwrap_or(3);
            if actual_deg <= 2 {
                // Re-dispatch with corrected degree
                match actual_deg {
                    0 => integrate_b_sqrt_const(b, p, sqrt_id, var, pool, log),
                    1 => integrate_b_sqrt_linear(b, p, sqrt_id, var, pool, log),
                    2 => integrate_b_sqrt_quadratic(b, p, sqrt_id, var, pool, log),
                    _ => unreachable!(),
                }
            } else {
                integrate_b_sqrt_high_degree(b, p, sqrt_id, var, pool, log)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Case: P is constant
// ∫ B(x) · sqrt(c) dx = sqrt(c) · ∫ B(x) dx
// ---------------------------------------------------------------------------

fn integrate_b_sqrt_const(
    b: ExprId,
    _p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // ∫ B(x)·sqrt(c) dx = sqrt(c) · ∫ B(x) dx
    let int_b = crate::integrate::engine::integrate_raw(b, var, pool, log)?;
    let result = pool.mul(vec![sqrt_id, int_b]);
    log.push(RewriteStep::simple("alg_sqrt_const", b, result));
    Ok(result)
}

// ---------------------------------------------------------------------------
// Case: P = a·x + b  (linear radicand, all integrals elementary)
// ---------------------------------------------------------------------------

fn integrate_b_sqrt_linear(
    b_expr: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // Extract P = a·x + c_const
    let (a, c_const) = as_linear(p, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented("linear radicand extraction failed".to_string())
    })?;

    // Detect if B is of the form polynomial / P^k (Hermite reduction)
    // For the common case, try polynomial B first
    match try_poly_b_linear(b_expr, p, sqrt_id, var, a, c_const, pool, log) {
        Ok(result) => return Ok(result),
        Err(IntegrationError::NotImplemented(_)) => {} // fall through
        Err(e) => return Err(e),
    }

    // Try rational B = R/P^k
    match try_rational_b_linear(b_expr, p, sqrt_id, var, a, c_const, pool, log) {
        Ok(result) => return Ok(result),
        Err(IntegrationError::NotImplemented(_)) => {} // fall through
        Err(e) => return Err(e),
    }

    Err(IntegrationError::NotImplemented(format!(
        "∫ B(x)·sqrt(P(x)) with P linear: B = {} not handled",
        pool.display(b_expr)
    )))
}

/// Integrate when B is a polynomial (possibly with constant coefficients) and P is linear.
#[allow(clippy::too_many_arguments)]
fn try_poly_b_linear(
    b_expr: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    a: ExprId, // coefficient of var in P
    _c_const: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // Check that B is a polynomial in var (degree ≥ 0)
    let _deg = poly_degree_in(b_expr, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented("B is not a polynomial in var".to_string())
    })?;

    // Use the reduction formula via substitution u = P = a*x + c:
    // ∫ B(x) · sqrt(P) dx where P = a*x + c
    //
    // General formula for polynomial B of degree n:
    //   = Q(x) · (P)^(3/2)
    // where Q is a polynomial found by the recurrence:
    //   2/(2k+3) · a^(k+1) · C(n,k) · (-c)^(n-k) contributes to Q
    //
    // Concretely, using the indefinite integral:
    //   ∫ x^k · sqrt(ax+c) dx = (2/(2k+3a)) · x^k · (ax+c)^(3/2)
    //                          - (2k/(2k+3)) · c/a · ∫ x^(k-1) · sqrt(ax+c) dx
    //
    // For k=0: ∫ sqrt(ax+c) dx = (2/(3a)) · (ax+c)^(3/2)
    // For k=1: ∫ x·sqrt(ax+c) dx = (2/(15a^2)) · (3ax-2c) · (ax+c)^(3/2)
    //
    // We implement this via the substitution approach for polynomial B:
    // Write B(x) = sum_{k=0}^{n} b_k * x^k
    // Then ∫ B·sqrt(P) dx = sum_{k=0}^{n} b_k · ∫ x^k · sqrt(ax+c) dx

    // For the substitution-based approach, we integrate each monomial separately.
    // ∫ x^k · sqrt(ax+c) dx via substitution u = ax+c:
    //   = (1/a) · ∫ ((u-c)/a)^k · sqrt(u) du
    //   = (1/a^(k+1)) · ∫ (u-c)^k · u^(1/2) du
    //   = (1/a^(k+1)) · ∫ sum_{j=0}^{k} C(k,j)·(-c)^(k-j)·u^(j+1/2) du
    //   = (1/a^(k+1)) · sum_{j=0}^{k} C(k,j)·(-c)^(k-j) · 2/(2j+3) · u^((2j+3)/2)
    //
    // Converting back: u = ax+c, u^((2j+3)/2) = (ax+c)^((2j+3)/2)
    // These are all multiples of (ax+c)^(3/2), so the result is:
    //   = sum_{j=0}^{k} C(k,j)·(-c)^(k-j) · 2/(a^(k+1)·(2j+3)) · (ax+c)^((2j+3)/2)
    //
    // Factor out sqrt(P) = (ax+c)^(1/2):
    //   (ax+c)^((2j+3)/2) = (ax+c)^(j+1) · (ax+c)^(1/2) = (ax+c)^(j+1) · sqrt(P)
    //
    // So the result is:
    //   sqrt(P) · sum_{j=0}^{k} C(k,j)·(-c)^(k-j) · 2/(a^(k+1)·(2j+3)) · (ax+c)^(j+1)
    //   = sqrt(P) · Q_k(x) where Q_k is a polynomial.
    //
    // For the whole polynomial B = sum b_k · x^k:
    //   ∫ B · sqrt(P) dx = sqrt(P) · sum_k b_k · Q_k(x)
    //   = sqrt(P) · Q(x)  where Q = sum_k b_k · Q_k(x)
    //
    // Since all terms factor out sqrt(P), the result is Q(x)·sqrt(P).

    // Get the coefficient list of B
    let b_coeffs_int = poly_int_coeffs(b_expr, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented("B coefficients not extractable as integers".to_string())
    })?;

    // Get a and c as integers for exact arithmetic
    let a_int = as_integer(a, pool).ok_or_else(|| {
        IntegrationError::NotImplemented("linear coefficient a is not an integer".to_string())
    })?;
    // c_const from the extract
    let p_coeffs_int = poly_int_coeffs(p, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented("P coefficients not extractable".to_string())
    })?;
    let c_int = p_coeffs_int
        .first()
        .cloned()
        .unwrap_or_else(|| rug::Integer::from(0));

    if a_int == 0 {
        return Err(IntegrationError::NotImplemented(
            "degenerate linear P: a=0".to_string(),
        ));
    }

    // Compute Q(x) = sum over k of b_k * Q_k(x)
    // where Q_k(x) = sum_{j=0}^{k} C(k,j) * (-c)^(k-j) * 2/(a^(k+1)*(2j+3)) * (ax+c)^(j+1)
    //
    // We accumulate Q as a polynomial in (P) = (ax+c), or equivalently collect powers of P:
    // Since P^(j+1) = (ax+c)^(j+1) contributes coefficient factors,
    // let's build Q as an expression tree.

    // Build the sum of terms: for each (k, j), add b_k · C(k,j) · (-c)^(k-j) · 2/(a^(k+1)·(2j+3)) · P^(j+1)
    use rug::Rational;
    let mut terms: Vec<ExprId> = Vec::new();

    for (k, b_k) in b_coeffs_int.iter().enumerate() {
        if *b_k == 0 {
            continue;
        }
        let k = k as i64;
        let a_pow = a_int.pow(k as u32 + 1); // a^(k+1)

        for j in 0..=(k as usize) {
            let j = j as i64;
            // C(k, j) = k! / (j! * (k-j)!)
            let binom = binomial_coeff(k as u64, j as u64);
            // (-c)^(k-j)
            let neg_c_pow = neg_c_power(&c_int, k - j);
            // 2 / (a^(k+1) * (2j+3))
            let denom = a_pow * rug::Integer::from(2 * j + 3);
            // coefficient = b_k * C(k,j) * (-c)^(k-j) * 2 / (a^(k+1)*(2j+3))
            let numer = b_k.clone() * binom * neg_c_pow * 2;
            if numer == 0 {
                continue;
            }
            let coeff = Rational::from((numer, denom));
            if coeff == 0 {
                continue;
            }
            // Build P^(j+1)
            let p_pow_expr = if j + 1 == 1 {
                p
            } else {
                pool.pow(p, pool.integer(j + 1))
            };
            // Build coeff as ExprId
            let coeff_expr = pool.rational(coeff.numer().clone(), coeff.denom().clone());
            let term = pool.mul(vec![coeff_expr, p_pow_expr]);
            terms.push(term);
        }
    }

    let q_expr = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };

    let result = pool.mul(vec![q_expr, sqrt_id]);
    log.push(RewriteStep::simple("alg_poly_linear", b_expr, result));
    Ok(result)
}

/// Integrate when B involves `1/P^k` times a polynomial and P is linear.
#[allow(clippy::too_many_arguments)]
fn try_rational_b_linear(
    b_expr: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    a: ExprId,
    _c_const: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // Detect B = c / P  →  ∫ c/P · sqrt(P) dx = ∫ c/sqrt(P) dx = (c * 2/a) * sqrt(P)
    // i.e. ∫ c · P^(-1) · P^(1/2) dx = ∫ c · P^(-1/2) dx = c · 2/a · sqrt(P)

    // B = P^n for integer n: ∫ P^n · sqrt(P) dx = 2/(a·(2n+3)) · P^(n+1) · sqrt(P)
    if let ExprData::Pow { base, exp } = pool.get(b_expr) {
        if base == p {
            if let Some(n) = as_integer(exp, pool) {
                let two_n3 = 2 * n + 3;
                if two_n3 == 0 {
                    return Err(IntegrationError::NotImplemented(
                        "pole in algebraic integration (n = -3/2)".to_string(),
                    ));
                }
                let denom = pool.mul(vec![a, pool.integer(two_n3)]);
                let denom_inv = pool.pow(denom, pool.integer(-1_i32));
                let p_n1 = p_integer_power(p, n + 1, pool);
                let result = pool.mul(vec![pool.integer(2_i32), denom_inv, p_n1, sqrt_id]);
                log.push(RewriteStep::simple("alg_p_power_linear", b_expr, result));
                return Ok(result);
            }
        }
    }

    // B = const_factor * P^n
    if let ExprData::Mul(args) = pool.get(b_expr) {
        let (const_parts, p_parts): (Vec<ExprId>, Vec<ExprId>) =
            args.iter().partition(|&&id| is_free_of(id, var, pool));
        if p_parts.len() == 1 {
            if let ExprData::Pow { base, exp } = pool.get(p_parts[0]) {
                if base == p {
                    if let Some(n) = as_integer(exp, pool) {
                        let two_n3 = 2 * n + 3;
                        if two_n3 == 0 {
                            return Err(IntegrationError::NotImplemented(
                                "pole in algebraic integration (n = -3/2)".to_string(),
                            ));
                        }
                        let const_factor = match const_parts.len() {
                            0 => pool.integer(1_i32),
                            1 => const_parts[0],
                            _ => pool.mul(const_parts),
                        };
                        let denom = pool.mul(vec![a, pool.integer(two_n3)]);
                        let denom_inv = pool.pow(denom, pool.integer(-1_i32));
                        let p_n1 = p_integer_power(p, n + 1, pool);
                        let result = pool.mul(vec![
                            pool.integer(2_i32),
                            const_factor,
                            denom_inv,
                            p_n1,
                            sqrt_id,
                        ]);
                        log.push(RewriteStep::simple("alg_rational_linear", b_expr, result));
                        return Ok(result);
                    }
                }
            }
        }
    }

    Err(IntegrationError::NotImplemented(
        "rational B with linear P: unsupported form".to_string(),
    ))
}

/// Build P^k as an ExprId, correctly handling k = 0 (returns 1) and k = 1 (returns P).
fn p_integer_power(p: ExprId, k: i64, pool: &ExprPool) -> ExprId {
    match k {
        0 => pool.integer(1_i32),
        1 => p,
        _ => pool.pow(p, pool.integer(k)),
    }
}

// ---------------------------------------------------------------------------
// Case: P = a·x² + b·x + c  (quadratic radicand, genus 0)
// ---------------------------------------------------------------------------

fn integrate_b_sqrt_quadratic(
    b_expr: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let (a, b_coeff, c) = as_quadratic(p, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented("quadratic radicand extraction failed".to_string())
    })?;

    // Dispatch based on the form of B
    // Try B = polynomial first
    if let Ok(result) = try_poly_b_quadratic(b_expr, p, sqrt_id, var, a, b_coeff, c, pool, log) {
        return Ok(result);
    }

    // Try B = 1/(something) patterns (rational)
    if let Ok(result) = try_rational_b_quadratic(b_expr, p, sqrt_id, var, a, b_coeff, c, pool, log)
    {
        return Ok(result);
    }

    Err(IntegrationError::NotImplemented(format!(
        "∫ B(x)·sqrt(quadratic): B = {} not handled",
        pool.display(b_expr)
    )))
}

/// Key table integrals for quadratic P = ax²+bx+c:
///
/// J_0 = ∫ 1/sqrt(P) dx:
///   If a > 0: (1/sqrt(a)) · log(2·sqrt(a)·sqrt(P) + 2a·x + b)
///   If a < 0: (1/sqrt(-a)) · arcsin((-2a·x - b)/sqrt(b²-4ac))  [when b²-4ac > 0]
///   Symbolic form (a always as ExprId): (1/sqrt(a)) · log(2·sqrt(a)·sqrt(P) + 2·a·x + b)
///
/// This function returns the symbolic J_0 expression.
fn j0_quadratic(
    _p: ExprId,
    sqrt_id: ExprId, // sqrt(P)
    var: ExprId,
    a: ExprId,
    b_coeff: ExprId,
    pool: &ExprPool,
) -> ExprId {
    // J_0 = (1/sqrt(a)) · log(2·sqrt(a)·sqrt(P) + 2·a·x + b)
    let sqrt_a = pool.func("sqrt", vec![a]);
    let two = pool.integer(2_i32);
    let two_sqrt_a_sqrt_p = pool.mul(vec![two, sqrt_a, sqrt_id]);
    let two_ax = pool.mul(vec![two, a, var]);
    let inner = pool.add(vec![two_sqrt_a_sqrt_p, two_ax, b_coeff]);
    let log_inner = pool.func("log", vec![inner]);
    let sqrt_a_inv = pool.pow(sqrt_a, pool.integer(-1_i32));
    pool.mul(vec![sqrt_a_inv, log_inner])
}

/// Integrate polynomial B(x) times sqrt(quadratic P).
#[allow(clippy::too_many_arguments)]
fn try_poly_b_quadratic(
    b_expr: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    a: ExprId,
    b_coeff: ExprId,
    c: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // B must be a polynomial in var
    let deg = poly_degree_in(b_expr, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented("B is not polynomial for quadratic P".to_string())
    })?;

    // For degree 0: B = const
    // ∫ c·sqrt(P) dx = c · ∫ sqrt(P) dx
    // ∫ sqrt(ax²+bx+c) dx = (2ax+b)/(4a) · sqrt(P) + (4ac-b²)/(8a) · J_0
    if deg == 0 {
        let b_const = b_expr; // free of var
        let result = integrate_sqrt_quadratic_base(p, sqrt_id, var, a, b_coeff, c, pool);
        let scaled = pool.mul(vec![b_const, result]);
        log.push(RewriteStep::simple("alg_const_sqrt_quad", b_expr, scaled));
        return Ok(scaled);
    }

    // For degree 1: B = d·x + e
    // ∫ (d·x + e)·sqrt(P) dx = d · ∫ x·sqrt(P) dx + e · ∫ sqrt(P) dx
    // ∫ x·sqrt(P) dx = P·sqrt(P)/(3a) - b/(6a) · ∫ sqrt(P) dx
    //                 (derived from integration by parts)
    if deg == 1 {
        let b_coeffs = poly_int_coeffs(b_expr, var, pool).ok_or_else(|| {
            IntegrationError::NotImplemented("degree-1 B coefficients not extractable".to_string())
        })?;
        let e_int = b_coeffs
            .first()
            .cloned()
            .unwrap_or_else(|| rug::Integer::from(0));
        let d_int = b_coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(|| rug::Integer::from(0));

        let e_expr = pool.integer(e_int);
        let d_expr = pool.integer(d_int);

        // int_sqrt_p = ∫ sqrt(P) dx
        let int_sqrt_p = integrate_sqrt_quadratic_base(p, sqrt_id, var, a, b_coeff, c, pool);

        // int_x_sqrt_p = ∫ x·sqrt(P) dx = P·sqrt(P)/(3a) - b/(6a) · int_sqrt_p
        let three_a = pool.mul(vec![pool.integer(3_i32), a]);
        let three_a_inv = pool.pow(three_a, pool.integer(-1_i32));
        let p_sqrt_p = pool.mul(vec![p, sqrt_id]);
        let term1 = pool.mul(vec![three_a_inv, p_sqrt_p]);
        let six_a = pool.mul(vec![pool.integer(6_i32), a]);
        let six_a_inv = pool.pow(six_a, pool.integer(-1_i32));
        let term2 = pool.mul(vec![pool.integer(-1_i32), b_coeff, six_a_inv, int_sqrt_p]);
        let int_x_sqrt_p = pool.add(vec![term1, term2]);

        let part_d = pool.mul(vec![d_expr, int_x_sqrt_p]);
        let part_e = pool.mul(vec![e_expr, int_sqrt_p]);
        let result = pool.add(vec![part_d, part_e]);
        log.push(RewriteStep::simple("alg_linear_sqrt_quad", b_expr, result));
        return Ok(result);
    }

    // Higher degrees: use the reduction formula
    // ∫ x^n · sqrt(P) dx with the recursion:
    // ∫ x^n · sqrt(P) dx = x^(n-1) · P · sqrt(P) / (2an+a+1/2) - ...
    // This gets complex; fall through to NotImplemented for now
    Err(IntegrationError::NotImplemented(format!(
        "∫ polynomial(deg {deg}) · sqrt(quadratic): not yet implemented for deg > 1"
    )))
}

/// Base formula: ∫ sqrt(ax²+bx+c) dx
/// = (2ax+b)/(4a) · sqrt(P) + (4ac−b²)/(8a) · J_0(P)
fn integrate_sqrt_quadratic_base(
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    a: ExprId,
    b_coeff: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let two = pool.integer(2_i32);
    let four = pool.integer(4_i32);
    let eight = pool.integer(8_i32);

    // (2ax+b)/(4a) · sqrt(P)
    let two_ax = pool.mul(vec![two, a, var]);
    let two_ax_plus_b = pool.add(vec![two_ax, b_coeff]);
    let four_a = pool.mul(vec![four, a]);
    let four_a_inv = pool.pow(four_a, pool.integer(-1_i32));
    let term1 = pool.mul(vec![four_a_inv, two_ax_plus_b, sqrt_id]);

    // (4ac − b²)/(8a) · J_0
    let four_ac = pool.mul(vec![four, a, c]);
    let b2 = pool.pow(b_coeff, pool.integer(2_i32));
    let neg_b2 = pool.mul(vec![pool.integer(-1_i32), b2]);
    let discriminant = pool.add(vec![four_ac, neg_b2]);
    let eight_a = pool.mul(vec![eight, a]);
    let eight_a_inv = pool.pow(eight_a, pool.integer(-1_i32));
    let j0 = j0_quadratic(p, sqrt_id, var, a, b_coeff, pool);
    let term2 = pool.mul(vec![eight_a_inv, discriminant, j0]);

    pool.add(vec![term1, term2])
}

/// Rational B forms for quadratic P:
/// - B = 1 → handled by poly case (deg 0)
/// - B = P^(-1) → ∫ 1/P · sqrt(P) dx = ∫ 1/sqrt(P) dx = J_0
/// - B = P^(-1/2) (stored as Pow(P, Rational(-1,2))) → should be caught by decomposition
///
/// The key cases here: B contains negative powers of P.
#[allow(clippy::too_many_arguments)]
fn try_rational_b_quadratic(
    b_expr: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    a: ExprId,
    b_coeff: ExprId,
    c: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    // B = P^n for integer n (including negative)
    if let ExprData::Pow { base, exp } = pool.get(b_expr) {
        if base == p {
            if let Some(n) = as_integer(exp, pool) {
                return integrate_p_power_sqrt_quad(n, p, sqrt_id, var, a, b_coeff, c, pool, log);
            }
        }
    }

    // B = constant_factor · P^n
    if let ExprData::Mul(args) = pool.get(b_expr) {
        let (const_parts, p_parts): (Vec<ExprId>, Vec<ExprId>) =
            args.iter().partition(|&&id| is_free_of(id, var, pool));
        if p_parts.len() == 1 {
            if let ExprData::Pow { base, exp } = pool.get(p_parts[0]) {
                if base == p {
                    if let Some(n) = as_integer(exp, pool) {
                        let const_factor = match const_parts.len() {
                            0 => pool.integer(1_i32),
                            1 => const_parts[0],
                            _ => pool.mul(const_parts),
                        };
                        let int_pn_sqrt = integrate_p_power_sqrt_quad(
                            n, p, sqrt_id, var, a, b_coeff, c, pool, log,
                        )?;
                        let result = pool.mul(vec![const_factor, int_pn_sqrt]);
                        return Ok(result);
                    }
                }
            }
        }
    }

    // B = (ax + d) where a, d are constants → degree-1 polynomial handled above
    // (This is a fallback for expressions not caught by try_poly_b_quadratic)
    Err(IntegrationError::NotImplemented(
        "rational B with quadratic P: unsupported form".to_string(),
    ))
}

/// Integrate P^n · sqrt(P) dx = ∫ P^(n + 1/2) dx for quadratic P.
#[allow(clippy::too_many_arguments)]
fn integrate_p_power_sqrt_quad(
    n: i64,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    a: ExprId,
    b_coeff: ExprId,
    c: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    match n {
        // ∫ P^(-1) · sqrt(P) dx = ∫ P^(-1/2) dx = J_0
        -1 => {
            let j0 = j0_quadratic(p, sqrt_id, var, a, b_coeff, pool);
            log.push(RewriteStep::simple("alg_j0_quad", p, j0));
            Ok(j0)
        }
        // ∫ P^0 · sqrt(P) dx = ∫ sqrt(P) dx  (already handled as deg-0 poly * sqrt)
        0 => {
            let result = integrate_sqrt_quadratic_base(p, sqrt_id, var, a, b_coeff, c, pool);
            log.push(RewriteStep::simple("alg_sqrt_quad_base", p, result));
            Ok(result)
        }
        // ∫ P^1 · sqrt(P) dx = ∫ P^(3/2) dx
        // = (2ax+b)/(8a) · P · sqrt(P) + (3(4ac-b²))/(32a) · J_0 ... (reduction formula)
        // Using the general reduction: ∫ P^(m+1/2) dx =
        //   (2ax+b)·P^m·sqrt(P)/(4a(m+1)) + (4ac-b²)(2m+1)/(8a(m+1)) · ∫ P^(m-1/2) dx
        1 => {
            // ∫ P^(3/2) dx:  m=1 in reduction
            // = (2ax+b)·P·sqrt(P)/(8a) + 3·D/(16a) · ∫ P^(1/2) dx
            let two = pool.integer(2_i32);
            let two_ax = pool.mul(vec![two, a, var]);
            let two_ax_b = pool.add(vec![two_ax, b_coeff]);
            let eight_a = pool.mul(vec![pool.integer(8_i32), a]);
            let eight_a_inv = pool.pow(eight_a, pool.integer(-1_i32));
            let term1 = pool.mul(vec![eight_a_inv, two_ax_b, p, sqrt_id]);

            let four_ac = pool.mul(vec![pool.integer(4_i32), a, c]);
            let b2 = pool.pow(b_coeff, pool.integer(2_i32));
            let neg_b2 = pool.mul(vec![pool.integer(-1_i32), b2]);
            let d = pool.add(vec![four_ac, neg_b2]); // D = 4ac-b^2
            let three_d = pool.mul(vec![pool.integer(3_i32), d]);
            let sixteen_a = pool.mul(vec![pool.integer(16_i32), a]);
            let sixteen_a_inv = pool.pow(sixteen_a, pool.integer(-1_i32));
            let int_sqrt_p = integrate_sqrt_quadratic_base(p, sqrt_id, var, a, b_coeff, c, pool);
            let term2 = pool.mul(vec![sixteen_a_inv, three_d, int_sqrt_p]);

            let result = pool.add(vec![term1, term2]);
            log.push(RewriteStep::simple("alg_p_3_2_quad", p, result));
            Ok(result)
        }
        _ => Err(IntegrationError::NotImplemented(format!(
            "∫ P^{n}·sqrt(P) with quadratic P: higher powers not implemented"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

fn binomial_coeff(n: u64, k: u64) -> rug::Integer {
    if k > n {
        return rug::Integer::from(0);
    }
    let k = k.min(n - k);
    let mut result = rug::Integer::from(1u64);
    for i in 0..k {
        result *= rug::Integer::from(n - i);
        result /= rug::Integer::from(i + 1);
    }
    result
}

// ---------------------------------------------------------------------------
// ∫ B(x)·√P dx for deg P ≥ 3 (genus ≥ 1) — sound decision
//
// By Liouville `∫ B√P = b·√P + Σ cⱼ log uⱼ` with `b ∈ ℚ(x)`.  Two parts:
//
//  * **Integral part** `b·√P`: differentiating, `(b√P)' = (b' + (P'/2P)·b)·√P`,
//    so `∫B√P = b√P` iff the rational **Risch DE** `b' + (P'/2P)·b = B` has a
//    rational solution — solved by `solve_rational_rde_generalized`.  This is
//    exact (verified by construction) and covers e.g. `∫(P'/2√P) = √P` and
//    polynomial weights `∫5x⁴√(x⁵+1) = ⅔(x⁵+1)^{3/2}`.
//
//  * **Logarithmic part**: when the RDE has no rational solution there are
//    residues.  Liouville ⟹ no residues ⇒ `∫B√P` is elementary iff it is an
//    exact algebraic derivative — but the RDE just said it is not — so a
//    **complete** empty residue divisor (with P squarefree) certifies
//    `NonElementary`.  With residues, FIND-ORDER decides: a **non-torsion**
//    divisor ⇒ `NonElementary`; otherwise (torsion log part) emitting it on a
//    genus ≥ 2 curve needs Coates' construction (genus 0/1 are handled upstream
//    by the parametrization / genus-1 capstone), so we decline.
//
// Soundness of the residue path requires the residue divisor to be **complete**:
// the residues live at the poles of `B` that are *not* branch points (a pole of
// `B` at a root of `P` is regularized by `√P`'s zero, contributing none).  So we
// require the part of `denom(B)` coprime to `P` to split over ℚ; otherwise we
// decline rather than risk a verdict on an incomplete divisor.
// ---------------------------------------------------------------------------

fn integrate_b_sqrt_high_degree(
    b: ExprId,
    p: ExprId,
    sqrt_id: ExprId,
    var: ExprId,
    pool: &ExprPool,
    log: &mut DerivationLog,
) -> Result<ExprId, IntegrationError> {
    let nonelem = |msg: &str| IntegrationError::NonElementary(msg.to_string());
    let notimpl = |msg: &str| IntegrationError::NotImplemented(msg.to_string());

    let p_poly = expr_to_qpoly(p, var, pool)
        .ok_or_else(|| notimpl("radicand P is not a polynomial in the variable"))?;
    let (b_num, b_den) = expr_to_qrational(b, var, pool)
        .ok_or_else(|| notimpl("weight B is not a rational function"))?;
    if degree(&p_poly) < 3 {
        return Err(notimpl("expected deg P ≥ 3 here"));
    }

    // 1. Integral part: b' + (P'/2P)·b = B  ⇒  ∫B√P = b·√P.
    let p_prime = poly_deriv(&p_poly);
    let two_p = poly_scale(&p_poly, &Rational::from(2));
    if let Some((q_num, q_den)) = solve_rational_rde_generalized(&p_prime, &two_p, &b_num, &b_den) {
        let q_expr = qrational_to_expr(&q_num, &q_den, var, pool);
        let result = pool.mul(vec![q_expr, sqrt_id]);
        log.push(RewriteStep::simple("alg_integral_part_sqrt", b, result));
        return Ok(result);
    }

    // 2. No algebraic primitive — analyze the logarithmic part via residues.
    //    Need P squarefree for the Liouville argument below.
    if degree(&poly_gcd(&p_poly, &p_prime)) > 0 {
        return Err(notimpl("non-squarefree radicand at deg ≥ 3"));
    }

    let h: AlgElem = vec![RatFn::int(0), RatFn::new(b_num.clone(), b_den.clone())];

    // Soundness gate: the residue divisor must be **complete** (residue theorem
    // Σ res = 0, counting rational, algebraic and infinite places).  Otherwise a
    // place is missing and no verdict is safe.
    if residue_sum_complete(2, &p_poly, &h) != 0 {
        return Err(notimpl(
            "residue divisor incomplete (missing places not yet supported)",
        ));
    }

    let alg_res = finite_residues_algebraic(2, &p_poly, &h);
    if alg_res.is_empty() {
        // All residues are rational — the complete divisor is the rational one.
        let divisor = residue_divisor_placed(2, &p_poly, &h);
        if divisor.is_empty() {
            // No residues anywhere ⇒ no log part; RDE ruled out an algebraic
            // primitive ⇒ a nonzero first/second-kind differential ⇒ non-elem.
            return Err(nonelem(
                "∫ B·√P over a genus ≥ 1 curve with no logarithmic part and no \
                 algebraic primitive: non-elementary",
            ));
        }
        return match find_order_placed(2, &p_poly, &divisor) {
            FindOrder::NonElementary => Err(nonelem(
                "residue divisor is non-torsion (FIND-ORDER): no elementary log part",
            )),
            _ => Err(notimpl(
                "genus ≥ 2 logarithmic part: torsion/undecided, log argument not \
                 yet constructible (Coates)",
            )),
        };
    }

    // Algebraic residues present.  Two routes through the Trager ℚ-basis
    // criterion: rational base points with irrational sheets (a compositum of
    // quadratic *sheet* fields) via `try_genus2_alg_log`; a quadratic *base*
    // point (degree-4 tower ℚ(√m,√c)) via `try_alg_base_log`.
    let verdict = if alg_res.iter().all(|r| r.conjugates == 1) {
        try_genus2_alg_log(&p_poly, &h, &alg_res)
    } else {
        try_alg_base_log(&p_poly, &h, &alg_res)
    };
    match verdict {
        Some(FindOrder::NonElementary) => Err(nonelem(
            "Trager ℚ-basis criterion: a residue component is non-torsion ⇒ \
             no elementary logarithmic part",
        )),
        _ => Err(notimpl(
            "genus ≥ 2 logarithmic part with algebraic residues: not decided \
             (torsion log not yet emittable, or out of the handled scope — \
             non-Galois tower / base degree ≥ 3)",
        )),
    }
}

/// Trager ℚ-basis decision for a single **quadratic algebraic base point**: the
/// pole denominator has an irreducible factor `q = x²−m`, the residue
/// `r0 ± r1·√a(α)` (`r0,r1 ∈ ℚ(α)=ℚ(√m)`) living in the degree-4 tower
/// `K = ℚ(√m)[w]/(w²−c)`, `c = a(α)`.  When `K/ℚ` is Galois
/// ([`super::alg_tower::galois_quartic`]) the four conjugate residues and places
/// are expressed in `ℚ[θ]/M`, decomposed over `ℚ`, and each component is tested
/// with [`trager_log_criterion_alg`] (reducing at primes that split `M`).  `None`
/// outside this scope (non-`x²−m` base, non-Galois `K`, more than one algebraic
/// orbit, or base degree ≥ 3).
/// `a mod m` over `ℚ[x]`.
fn qmod_l(a: &QPoly, m: &QPoly) -> QPoly {
    trim(poly_divrem(a, m).1)
}

fn try_alg_base_log(p: &QPoly, h: &AlgElem, alg_res: &[AlgResidue]) -> Option<FindOrder> {
    // Exactly one algebraic orbit, a quadratic base (conjugates == 2).
    if alg_res.len() != 1 || alg_res[0].conjugates != 2 || degree(&alg_res[0].minpoly) != 2 {
        return None;
    }
    let ar = &alg_res[0];
    let q_raw = trim(ar.minpoly.clone());
    if degree(&q_raw) != 2 {
        return None;
    }
    // Complete the square: a general monic base `q = x²+b·x+c₀` becomes the
    // depressed `qn = x²−m` (`m = b²/4−c₀`) under `α = β − b/2` (`β = √m`).  The
    // tower builders (`galois_quartic`/`quartic_closure`) consume `qn`; the field
    // elements (`c = a(α)`, `r0`, `r1`) are rewritten in the β-basis via
    // `α = β + shift`, and each place's x-coordinate is shifted *back* by `shift`
    // so it is the **actual** base-point coordinate on `y²=P`.  (The on-curve
    // check in `reduce_and_build` self-verifies this — a wrong shift only declines,
    // never mis-decides.)  When `b=0` every step below is the identity.
    let lead = q_raw[2].clone();
    let b = q_raw[1].clone() / &lead;
    let c0 = q_raw[0].clone() / &lead;
    let half_b = b / Rational::from(2);
    let shift = -half_b.clone();
    let m_val = half_b.clone() * &half_b - &c0; // m = b²/4 − c₀
    let q = vec![-m_val, Rational::from(0), Rational::from(1)]; // x² − m
                                                                // `e0 + e1·α  ↦  (e0 + e1·shift) + e1·β`   (substitute `α = β + shift`).
    let to_beta = |e: &QPoly| -> QPoly {
        let e0 = e.first().cloned().unwrap_or_else(|| Rational::from(0));
        let e1 = e.get(1).cloned().unwrap_or_else(|| Rational::from(0));
        trim(vec![e0 + e1.clone() * &shift, e1])
    };
    // Add the constant `shift` to a place x-coordinate (in `ℚ[θ]/M`).
    let shift_x = |x: &QPoly| -> QPoly {
        let mut v = x.clone();
        if v.is_empty() {
            v.push(shift.clone());
        } else {
            v[0] = v[0].clone() + &shift;
        }
        trim(v)
    };
    // c = a(α) ∈ ℚ(α): reduce the radicand mod the *raw* minpoly, then β-basis.
    let c = to_beta(&trim(poly_divrem(p, &q_raw).1));
    let r0_b = to_beta(&ar.r0);
    let r1_b = to_beta(&ar.r1);
    let r0 = &r0_b;
    let r1 = &r1_b;

    // Build the conjugate orbit's places and residues in a common field — the
    // degree-4 tower K when Galois, else its degree-8 Galois closure L.
    let (alg_places, alg_residues, dim) =
        if let Some((m, a_in, w_in, autos)) = super::alg_tower::galois_quartic(&q, &c) {
            // Galois: ρ = r0(α)+r1(α)w in ℚ[θ]/M; orbit = {σⱼ(ρ), σⱼ(P₀)}.
            let dim = (degree(&m).max(0) as usize).max(1); // = 4
            let rho = qmod_l(
                &poly_add(
                    &super::alg_tower::compose_mod(r0, &a_in, &m),
                    &qmod_l(
                        &poly_mul(&super::alg_tower::compose_mod(r1, &a_in, &m), &w_in),
                        &m,
                    ),
                ),
                &m,
            );
            let mut places = Vec::new();
            let mut residues = Vec::new();
            for pi in &autos {
                let mut rj = super::alg_tower::compose_mod(&rho, pi, &m);
                rj.resize(dim, Rational::from(0));
                places.push(AlgPlace {
                    minpoly: m.clone(),
                    x_coord: shift_x(&super::alg_tower::compose_mod(&a_in, pi, &m)),
                    y_coord: super::alg_tower::compose_mod(&w_in, pi, &m),
                    coeff: Integer::from(0),
                    orbit: false,
                });
                residues.push(rj);
            }
            (places, residues, dim)
        } else {
            // Non-Galois: work in the degree-8 closure L = K(√(N(c))).  Build the
            // four orbit places (±α, ±√c), (±α, ±√c̄) and residues explicitly.
            let (ml, alpha, w, v) = super::alg_tower::quartic_closure(&q, &c)?;
            let dim = (degree(&ml).max(0) as usize).max(1); // = 8
            let neg_alpha = poly_scale(&alpha, &Rational::from(-1));
            let lin = |coef: &QPoly, a: &QPoly| {
                qmod_l(
                    &poly_add(
                        &vec![coef.first().cloned().unwrap_or_else(|| Rational::from(0))],
                        &poly_scale(
                            a,
                            &coef.get(1).cloned().unwrap_or_else(|| Rational::from(0)),
                        ),
                    ),
                    &ml,
                )
            };
            let r0a = lin(r0, &alpha);
            let r1a = lin(r1, &alpha);
            let r0n = lin(r0, &neg_alpha);
            let r1n = lin(r1, &neg_alpha);
            let mulm = |x: &QPoly, y: &QPoly| qmod_l(&poly_mul(x, y), &ml);
            let sub = |a: &QPoly, b: &QPoly| poly_add(a, &poly_scale(b, &Rational::from(-1)));
            let entries: [(QPoly, QPoly, QPoly); 4] = [
                (alpha.clone(), w.clone(), poly_add(&r0a, &mulm(&r1a, &w))),
                (
                    alpha.clone(),
                    poly_scale(&w, &Rational::from(-1)),
                    sub(&r0a, &mulm(&r1a, &w)),
                ),
                (
                    neg_alpha.clone(),
                    v.clone(),
                    poly_add(&r0n, &mulm(&r1n, &v)),
                ),
                (
                    neg_alpha.clone(),
                    poly_scale(&v, &Rational::from(-1)),
                    sub(&r0n, &mulm(&r1n, &v)),
                ),
            ];
            let mut places = Vec::new();
            let mut residues = Vec::new();
            for (x, y, res) in entries {
                places.push(AlgPlace {
                    minpoly: ml.clone(),
                    x_coord: shift_x(&x),
                    y_coord: y,
                    coeff: Integer::from(0),
                    orbit: false,
                });
                let mut rv = trim(res);
                rv.resize(dim, Rational::from(0));
                residues.push(rv);
            }
            (places, residues, dim)
        };

    // Rational + infinite places carry rational residues (basis index 0).
    let rat_div = residue_divisor_placed(2, p, h);
    let rat_residues: Vec<KElem> = rat_div
        .iter()
        .map(|r| {
            let mut v = vec![Rational::from(0); dim];
            v[0] = r.residue.value.clone();
            v
        })
        .collect();

    Some(trager_log_criterion_alg(
        2,
        p,
        &rat_div,
        &rat_residues,
        &alg_places,
        &alg_residues,
        dim,
    ))
}

/// Trager ℚ-basis decision for `∫ (B·y) dx` on `y²=P` (deg P odd ≥ 5) when the
/// algebraic residues live in a **single quadratic field** `ℚ(√d)` — i.e. every
/// algebraic residue comes from a *rational* base point `α` whose sheet
/// `√a(α)` is irrational.  Distinct sheets `√d₁, …, √d_k` (a **compositum** of
/// quadratic fields) are handled too: a residue is `r0 ± r1·√d_i` (no products
/// of distinct `√d`), so the residues span only `{1, √d₁, …, √d_k}` and the
/// Trager ℚ-basis components **separate** — one rational `1`-component (the
/// conjugate sheet-sums are rational) and one single-quadratic `√d_i`-component
/// per field.  Residues are represented in that basis and fed to
/// [`trager_log_criterion_alg`].  `None` for an algebraic *base* point
/// (`conjugates ≠ 1`, a genuine tower) — still out of scope.
fn try_genus2_alg_log(p: &QPoly, h: &AlgElem, alg_res: &[AlgResidue]) -> Option<FindOrder> {
    // Collect the distinct squarefree sheet discriminants `d_i`; index them so
    // the residue basis is {1 (index 0), √d_0 (1), √d_1 (2), …}.
    let mut d_list: Vec<Integer> = Vec::new();
    for ar in alg_res {
        if ar.conjugates != 1 || degree(&ar.minpoly) != 1 {
            return None; // algebraic base point ⇒ tower field, out of scope
        }
        let alpha = -ar.minpoly[0].clone(); // monic x − α
        let a_at = eval_poly_q(p, &alpha);
        if a_at == 0 {
            return None; // branch point, not a B-pole sheet
        }
        let d = squarefree_part_rat(&a_at);
        if !d_list.contains(&d) {
            d_list.push(d);
        }
    }
    let dim = 1 + d_list.len();
    let d_index = |d: &Integer| d_list.iter().position(|x| x == d).unwrap();

    // Rational + infinite places: residues are rational ⇒ value at basis index 0.
    let rat_div = residue_divisor_placed(2, p, h);
    let rat_residues: Vec<KElem> = rat_div
        .iter()
        .map(|r| {
            let mut v = vec![Rational::from(0); dim];
            v[0] = r.residue.value.clone();
            v
        })
        .collect();

    // Algebraic places: the two sheets (α, ±√a(α)) = (α, ±k√d), a(α) = k²·d.
    let mut alg_places: Vec<AlgPlace> = Vec::new();
    let mut alg_residues: Vec<KElem> = Vec::new();
    for ar in alg_res {
        let alpha = -ar.minpoly[0].clone();
        let a_at = eval_poly_q(p, &alpha);
        let d = squarefree_part_rat(&a_at);
        let d_rat = Rational::from(d.clone());
        let theta_min = vec![-d_rat.clone(), Rational::from(0), Rational::from(1)]; // θ² − d_i
        let k = rat_sqrt(&(a_at / &d_rat))?; // a(α)/d is a perfect square
        let r0 = ar.r0.first().cloned().unwrap_or_else(|| Rational::from(0));
        let r1 = ar.r1.first().cloned().unwrap_or_else(|| Rational::from(0));
        let idx = 1 + d_index(&d);
        for sign in [Rational::from(1), Rational::from(-1)] {
            alg_places.push(AlgPlace {
                minpoly: theta_min.clone(),
                x_coord: vec![alpha.clone()], // x = α
                y_coord: vec![Rational::from(0), sign.clone() * &k], // y = ±k·θ = ±√a(α)
                coeff: Integer::from(0),      // set per-component by the criterion
                orbit: false,                 // a single ℚ(√d_i) sheet (one embedding)
            });
            // residue r0·1 ± r1·k·√d_i: r0 at index 0, ±r1·k at index 1+d_index.
            let mut v = vec![Rational::from(0); dim];
            v[0] = r0.clone();
            v[idx] = sign.clone() * &r1 * &k;
            alg_residues.push(v);
        }
    }

    Some(trager_log_criterion_alg(
        2,
        p,
        &rat_div,
        &rat_residues,
        &alg_places,
        &alg_residues,
        dim,
    ))
}

/// Horner evaluation of `p ∈ ℚ[x]` at a rational point.
fn eval_poly_q(p: &QPoly, x: &Rational) -> Rational {
    p.iter().rev().fold(Rational::from(0), |acc, c| acc * x + c)
}

/// Squarefree part (kernel) of a rational `r ≠ 0`: the sign-carrying product of
/// the primes dividing `r` to an odd power, as an integer (so `r / sqfree` is a
/// perfect square).  Uses `r = (num·den)/den²`.
fn squarefree_part_rat(r: &Rational) -> Integer {
    let prod = r.numer().clone() * r.denom();
    let sign = if prod < 0 {
        Integer::from(-1)
    } else {
        Integer::from(1)
    };
    let mut m = prod.abs();
    let mut sq = Integer::from(1);
    let mut dd = Integer::from(2);
    while Integer::from(&dd * &dd) <= m {
        if m.is_divisible(&dd) {
            let mut e = 0u32;
            while m.is_divisible(&dd) {
                m /= &dd;
                e += 1;
            }
            if e % 2 == 1 {
                sq *= &dd;
            }
        }
        dd += 1;
    }
    sq *= &m; // remaining prime factor (exponent 1)
    sq * sign
}

/// Exact rational square root, or `None` if `r` is not a perfect square in `ℚ`.
fn rat_sqrt(r: &Rational) -> Option<Rational> {
    if *r < 0 {
        return None;
    }
    let n = r.numer().clone();
    let d = r.denom().clone();
    let ns = n.clone().sqrt();
    let ds = d.clone().sqrt();
    if Integer::from(&ns * &ns) == n && Integer::from(&ds * &ds) == d {
        Some(Rational::from((ns, ds)))
    } else {
        None
    }
}

/// Build the expression `num(x)/den(x)`.
fn qrational_to_expr(num: &QPoly, den: &QPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    let n = qpoly_to_expr(num, var, pool);
    if den.len() == 1 && den.first().map(|c| *c == 1).unwrap_or(false) {
        return n;
    }
    let d = qpoly_to_expr(den, var, pool);
    pool.mul(vec![n, pool.pow(d, pool.integer(-1_i32))])
}

fn neg_c_power(c: &rug::Integer, n: i64) -> rug::Integer {
    if n == 0 {
        return rug::Integer::from(1);
    }
    let base = rug::Integer::from(-1) * c;
    if n > 0 {
        let mut result = rug::Integer::from(1);
        for _ in 0..n {
            result *= &base;
        }
        result
    } else {
        // negative power: for integer arithmetic this requires the value to be ±1
        // (for general use, fallback to 0 if not invertible)
        rug::Integer::from(0)
    }
}
