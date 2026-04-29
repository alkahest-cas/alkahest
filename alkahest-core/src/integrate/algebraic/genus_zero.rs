//! Integration formulas for genus-0 algebraic extensions.
//!
//! Handles `∫ B(x) · sqrt(P(x)) dx` when P has degree ≤ 2 (genus-0 curve).
//! For degree ≥ 3, returns `IntegrationError::NonElementary`.
//!
//! Reference: Bronstein (2005) §6.3–6.5; standard CAS table integrals.

use super::poly_utils::{
    as_integer, as_linear, as_quadratic, is_free_of, poly_degree_in, poly_int_coeffs,
};
use crate::deriv::log::{DerivationLog, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprData, ExprId, ExprPool};

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
                    _ => Err(IntegrationError::NonElementary(
                        "integral over genus-1 (elliptic) or higher curve; no elementary antiderivative"
                            .to_string(),
                    )),
                }
            } else {
                Err(IntegrationError::NonElementary(
                    "integral over genus-1 (elliptic) or higher curve; no elementary antiderivative"
                        .to_string(),
                ))
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
