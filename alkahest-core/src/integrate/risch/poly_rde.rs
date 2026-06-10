//! Polynomial Risch Differential Equation (RDE) solver over ℚ\[x\].
//!
//! Solves `y' + k·Dη·y = h` where:
//!   - k is a nonzero integer (the monomial degree in the exp tower)
//!   - Dη ∈ ℚ\[x\] is the derivative of the tower exponent η
//!   - h ∈ ℚ\[x\] is the integrand coefficient
//!
//! Returns `Some(y)` if a polynomial solution y ∈ ℚ\[x\] exists, `None` otherwise.
//!
//! **Key decision criterion** (Bronstein 2005, Thm 5.1):
//! When `Dη` has degree `d ≥ 1`, a polynomial solution exists iff `deg(h) ≥ d`.
//! When `Dη` is a nonzero constant (`d = 0`), a polynomial solution always exists.
//!
//! References:
//!   - Bronstein (2005). *Symbolic Integration I*, §5.2, Algorithm 5.1.
//!   - Risch (1969). The problem of integration in finite terms. *Trans. AMS* 139.

use rug::Rational;

use super::number_field::{KPoly, NumberField};

/// A polynomial over ℚ, stored as coefficient vector in ascending degree order.
/// `poly[i]` is the coefficient of `x^i`.  The zero polynomial is represented
/// as the empty vector.
pub type QPoly = Vec<Rational>;

// ---------------------------------------------------------------------------
// Basic polynomial arithmetic over ℚ
// ---------------------------------------------------------------------------

/// Trim trailing zero coefficients.
pub fn trim(mut p: QPoly) -> QPoly {
    while p.last().is_some_and(|c| *c == 0) {
        p.pop();
    }
    p
}

/// Degree of a polynomial (returns -1 for the zero polynomial).
pub fn degree(p: &QPoly) -> i64 {
    let mut d = p.len() as i64 - 1;
    while d >= 0 && p[d as usize] == 0 {
        d -= 1;
    }
    d
}

/// Return the zero polynomial.
pub fn poly_zero() -> QPoly {
    vec![]
}

/// Return 1 as a constant polynomial.
#[allow(dead_code)]
pub fn poly_one() -> QPoly {
    vec![Rational::from(1)]
}

/// Add two polynomials.
pub fn poly_add(a: &QPoly, b: &QPoly) -> QPoly {
    let n = a.len().max(b.len());
    let mut result = vec![Rational::from(0); n];
    for (i, c) in a.iter().enumerate() {
        result[i] += c;
    }
    for (i, c) in b.iter().enumerate() {
        result[i] += c;
    }
    trim(result)
}

/// Multiply two polynomials.
pub fn poly_mul(a: &QPoly, b: &QPoly) -> QPoly {
    if a.is_empty() || b.is_empty() {
        return poly_zero();
    }
    let mut result = vec![Rational::from(0); a.len() + b.len() - 1];
    for (i, ca) in a.iter().enumerate() {
        for (j, cb) in b.iter().enumerate() {
            result[i + j] += ca.clone() * cb.clone();
        }
    }
    trim(result)
}

/// Scale a polynomial by a rational constant.
pub fn poly_scale(p: &QPoly, s: &Rational) -> QPoly {
    if *s == 0 || p.is_empty() {
        return poly_zero();
    }
    trim(p.iter().map(|c| c.clone() * s.clone()).collect())
}

/// Differentiate a polynomial.
pub fn poly_deriv(p: &QPoly) -> QPoly {
    if p.len() <= 1 {
        return poly_zero();
    }
    trim(
        p[1..]
            .iter()
            .enumerate()
            .map(|(i, c)| c.clone() * Rational::from(i as i64 + 1))
            .collect(),
    )
}

/// Integrate a polynomial (constant of integration = 0).
pub fn poly_integrate(p: &QPoly) -> QPoly {
    let p = trim(p.clone());
    if p.is_empty() {
        return poly_zero();
    }
    let mut result = vec![Rational::from(0)]; // constant term = 0
    for (i, c) in p.iter().enumerate() {
        result.push(c.clone() / Rational::from(i as i64 + 1));
    }
    trim(result)
}

// ---------------------------------------------------------------------------
// Polynomial RDE solver
// ---------------------------------------------------------------------------

/// Solve the polynomial Risch Differential Equation over ℚ\[x\]:
/// ```text
///   y'(x) + k · Dη(x) · y(x) = h(x)
/// ```
///
/// # Arguments
/// - `k`: the integer monomial degree in the exp tower (must be nonzero)
/// - `deta`: the derivative of the tower exponent η, as a ℚ-polynomial
/// - `h`: the right-hand side, as a ℚ-polynomial
///
/// # Returns
/// - `Some(y)` — the unique polynomial solution if one exists
/// - `None` — no polynomial solution (certifies non-elementary integration)
///
/// # Algorithm
///
/// The degree of a polynomial solution (if it exists) is determined by the
/// leading terms.  When `deg(Dη) = d ≥ 1`:
///
/// - If `deg(h) < d`:  no polynomial solution exists — return `None`.
/// - If `deg(h) ≥ d`:  unique solution of degree `m = deg(h) − d`.
///
/// We solve from the highest-degree coefficient downward, then verify the
/// result by substituting back.
///
/// When `Dη` is constant (d = 0), a polynomial solution of degree `deg(h)`
/// always exists (solved by the same downward sweep).
///
/// When `Dη = 0` (k = 0 or Dη vanishes), the equation reduces to `y' = h`,
/// solved by antidifferentiation (always possible for polynomial h).
pub fn solve_poly_rde(k: i64, deta: &[Rational], h: &[Rational]) -> Option<QPoly> {
    let deta = trim(deta.to_vec());
    let h = trim(h.to_vec());

    // h = 0 → y = 0 is always a solution.
    if h.is_empty() {
        return Some(poly_zero());
    }

    let deg_deta = degree(&deta);

    // Dη = 0: equation is y' = h; solution is ∫ h dx (polynomial).
    if deg_deta < 0 {
        return Some(poly_integrate(&h));
    }

    // k must be nonzero (the exp tower monomial degree).
    assert!(k != 0, "solve_poly_rde called with k=0: caller bug");

    let deg_h = degree(&h);

    // Degree of the polynomial solution (if one exists):
    //   from the leading term k·lc(Dη)·y_m = lc(h), we get m + d = deg(h),
    //   so m = deg(h) − d.
    let m_signed = deg_h - deg_deta;

    if m_signed < 0 {
        // No polynomial solution is possible: the degree equation has no solution.
        return None;
    }

    let m = m_signed as usize;
    let lc_deta = deta[deg_deta as usize].clone(); // leading coeff of Dη
    let k_rat = Rational::from(k);

    // Allocate result vector y[0..=m].
    let mut y = vec![Rational::from(0); m + 1];

    // Solve from j = m down to 0.
    // The equation at degree (j + deg_deta) is:
    //
    //   k · lc(Dη) · y[j]
    //   + (y' contribution at degree j+deg_deta)
    //   + (k·Dη·y lower cross-terms)
    //   = h[j + deg_deta]
    //
    // All y[l] for l > j are already known; we solve for y[j].
    for j in (0..=m).rev() {
        let target_deg = j as i64 + deg_deta;

        // Right-hand side: h coefficient at target_deg.
        let mut rhs = if target_deg < h.len() as i64 {
            h[target_deg as usize].clone()
        } else {
            Rational::from(0)
        };

        // Subtract y' contribution at target_deg.
        // y' at degree d equals (d+1) · y[d+1].  We need d = target_deg, so:
        let deriv_idx = target_deg as usize + 1;
        if deriv_idx <= m {
            rhs -= Rational::from(target_deg + 1) * y[deriv_idx].clone();
        }

        // Subtract k · Dη[i] · y[l] for i < deg_deta (the "cross-terms" involving
        // already-known y[l] with l = j + deg_deta − i > j).
        for (i, deta_i) in deta.iter().enumerate().take(deg_deta as usize) {
            let l = (target_deg - i as i64) as usize;
            if l <= m && l != j {
                // y[l] is already known (l > j since i < deg_deta and target_deg = j+deg_deta).
                rhs -= k_rat.clone() * deta_i.clone() * y[l].clone();
            }
        }

        // Solve for y[j]: k · lc(Dη) · y[j] = rhs.
        let divisor = k_rat.clone() * lc_deta.clone();
        y[j] = rhs / divisor;
    }

    let y = trim(y);

    // Verification: compute y' + k·Dη·y and check equality with h.
    // This catches the case where the extra equations (degrees 0 .. deg_deta−1)
    // are inconsistent.
    let y_prime = poly_deriv(&y);
    let k_deta_y = poly_scale(&poly_mul(&deta, &y), &k_rat);
    let lhs = trim(poly_add(&y_prime, &k_deta_y));
    let h_trimmed = trim(h);

    if polys_equal(&lhs, &h_trimmed) {
        Some(y)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Polynomial RDE over a number field K = ℚ(α)  (Risch Gap E)
// ---------------------------------------------------------------------------

/// Solve the polynomial Risch Differential Equation `y' + k·Dη·y = h` with
/// coefficients in a number field `K = ℚ(α)` (represented by `field`).
///
/// This mirrors [`solve_poly_rde`] exactly — same degree analysis and downward
/// coefficient sweep — but every coefficient operation goes through `field`
/// instead of ℚ.  It handles exp-tower integrands whose coefficient is a
/// polynomial in `x` with algebraic-number coefficients that cannot be split off
/// as a constant factor (e.g. `∫ (x + √2)·exp(x) dx`).
///
/// `deta` and `h` are `K`-polynomials in `x` (ascending degree).  Returns the
/// unique `K`-polynomial solution, or `None` when none exists (which certifies a
/// non-elementary integral over `K`, just as in the ℚ case).
pub fn solve_poly_rde_k(field: &NumberField, k: i64, deta: &KPoly, h: &KPoly) -> Option<KPoly> {
    let deta = NumberField::kpoly_trim(deta.clone());
    let h = NumberField::kpoly_trim(h.clone());

    // h = 0 → y = 0.
    if h.is_empty() {
        return Some(Vec::new());
    }

    let deg_deta = NumberField::kdeg(&deta);

    // Dη = 0: equation is y' = h; solution is ∫ h dx.
    if deg_deta < 0 {
        return Some(field.kpoly_integrate(&h));
    }

    assert!(k != 0, "solve_poly_rde_k called with k=0: caller bug");

    let deg_h = NumberField::kdeg(&h);
    let m_signed = deg_h - deg_deta;
    if m_signed < 0 {
        return None; // degree equation has no solution
    }
    let m = m_signed as usize;

    let kk = field.from_int(k);
    let lc = deta[deg_deta as usize].clone(); // leading coeff of Dη
    let divisor_inv = field.inv(&field.mul(&kk, &lc))?;

    let mut y: KPoly = vec![NumberField::k_zero(); m + 1];
    for j in (0..=m).rev() {
        let target_deg = j as i64 + deg_deta;

        // RHS: h coefficient at target_deg.
        let mut rhs = if (target_deg as usize) < h.len() {
            h[target_deg as usize].clone()
        } else {
            NumberField::k_zero()
        };

        // Subtract y' contribution: (target_deg+1)·y[target_deg+1].
        let deriv_idx = target_deg as usize + 1;
        if deriv_idx <= m {
            let coef = field.from_int(target_deg + 1);
            rhs = field.sub(&rhs, &field.mul(&coef, &y[deriv_idx]));
        }

        // Subtract k·Dη[i]·y[l] for i < deg_deta (already-known y[l], l > j).
        for i in 0..deg_deta as usize {
            let deta_i = deta.get(i).cloned().unwrap_or_else(NumberField::k_zero);
            let l = (target_deg - i as i64) as usize;
            if l <= m && l != j {
                let term = field.mul(&field.mul(&kk, &deta_i), &y[l]);
                rhs = field.sub(&rhs, &term);
            }
        }

        // Solve k·lc(Dη)·y[j] = rhs.
        y[j] = field.mul(&rhs, &divisor_inv);
    }

    let y = NumberField::kpoly_trim(y);

    // Verify y' + k·Dη·y == h (guards against an under-sized degree bound and the
    // over-determined lower-degree equations).
    let yp = field.kpoly_deriv(&y);
    let kdeta_y = field.kpoly_scale(&field.kpoly_mul(&deta, &y), &kk);
    let lhs = field.kpoly_add(&yp, &kdeta_y);
    if kpoly_eq(&lhs, &h) {
        Some(y)
    } else {
        None
    }
}

/// Equality of two `K`-polynomials in `x` (coefficient-wise after trimming).
fn kpoly_eq(a: &KPoly, b: &KPoly) -> bool {
    let a = NumberField::kpoly_trim(a.clone());
    let b = NumberField::kpoly_trim(b.clone());
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| trim(x.clone()) == trim(y.clone()))
}

/// Compare two polynomials (after trimming) for equality.
fn polys_equal(a: &QPoly, b: &QPoly) -> bool {
    let a = trim(a.clone());
    let b = trim(b.clone());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| *x == *y)
}

// ---------------------------------------------------------------------------
// Conversion helpers: ExprId ↔ QPoly
// ---------------------------------------------------------------------------

use crate::kernel::{ExprData, ExprId, ExprPool};

/// Convert a symbolic expression to a ℚ\[x\] polynomial.
///
/// Returns `None` if the expression is not a polynomial in `var` with
/// rational (or integer) coefficients.
pub fn expr_to_qpoly(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<QPoly> {
    let mut coeffs: Vec<Rational> = Vec::new();
    if collect_qpoly(expr, var, pool, &mut coeffs, 1) {
        Some(trim(coeffs))
    } else {
        None
    }
}

/// Recursive helper: accumulate `factor * expr` into `coeffs`.
/// Returns `false` on failure (non-polynomial subexpressions).
fn collect_qpoly(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    coeffs: &mut Vec<Rational>,
    factor: i64,
) -> bool {
    // Helper: add `value * factor` at degree `deg`.
    let ensure_len = |coeffs: &mut Vec<Rational>, n: usize| {
        while coeffs.len() < n {
            coeffs.push(Rational::from(0));
        }
    };

    if expr == var {
        ensure_len(coeffs, 2);
        coeffs[1] += Rational::from(factor);
        return true;
    }

    match pool.get(expr) {
        ExprData::Integer(n) => {
            let Some(val) = n.0.to_i64() else {
                return false; // integer too large to fit in i64 — not representable
            };
            ensure_len(coeffs, 1);
            coeffs[0] += Rational::from(factor * val);
            true
        }
        ExprData::Rational(r) => {
            let rat = r.0.clone();
            ensure_len(coeffs, 1);
            coeffs[0] += rat * Rational::from(factor);
            true
        }
        ExprData::Add(args) => {
            for a in &args {
                if !collect_qpoly(*a, var, pool, coeffs, factor) {
                    return false;
                }
            }
            true
        }
        ExprData::Mul(args) => {
            // Partition into constant factor and variable parts.
            let mut rat_factor = Rational::from(factor);
            let mut var_parts: Vec<ExprId> = Vec::new();
            for &a in &args {
                if is_free_of_var(a, var, pool) {
                    // Extract rational value.
                    match to_rational_const(a, pool) {
                        Some(r) => rat_factor *= r,
                        None => return false, // symbolic constant
                    }
                } else {
                    var_parts.push(a);
                }
            }
            if var_parts.is_empty() {
                // All constants.
                ensure_len(coeffs, 1);
                coeffs[0] += rat_factor;
                return true;
            }
            // Exactly one var-part: extract it.
            if var_parts.len() == 1 {
                // Recurse on the single var part with the rational factor.
                let mut sub = Vec::new();
                if !collect_qpoly(var_parts[0], var, pool, &mut sub, 1) {
                    return false;
                }
                let scale = rat_factor;
                ensure_len(coeffs, sub.len());
                for (i, c) in sub.iter().enumerate() {
                    if i >= coeffs.len() {
                        coeffs.push(Rational::from(0));
                    }
                    coeffs[i] += c.clone() * scale.clone();
                }
                return true;
            }
            // Multiple var parts — treat as a product (e.g., x * x = x^2).
            // We only handle the simple cases here.
            false
        }
        ExprData::Pow { base, exp } => {
            // base^n where base == var and n is a positive integer.
            if base == var {
                if let ExprData::Integer(n) = pool.get(exp) {
                    if let Some(n_u) = n.0.to_u32() {
                        ensure_len(coeffs, n_u as usize + 1);
                        coeffs[n_u as usize] += Rational::from(factor);
                        return true;
                    }
                }
            }
            // base is free of var and exp is an integer — treat as constant.
            if is_free_of_var(expr, var, pool) {
                if let Some(r) = to_rational_const(expr, pool) {
                    ensure_len(coeffs, 1);
                    coeffs[0] += r * Rational::from(factor);
                    return true;
                }
            }
            false
        }
        _ => {
            // Free of var: treat as constant coefficient.
            if is_free_of_var(expr, var, pool) {
                if let Some(r) = to_rational_const(expr, pool) {
                    ensure_len(coeffs, 1);
                    coeffs[0] += r * Rational::from(factor);
                    return true;
                }
            }
            false
        }
    }
}

/// Return `true` if `expr` syntactically does not involve `var`.
pub fn is_free_of_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    // Per-call `ExprId`-keyed memo (`var` is fixed for the duration of one call,
    // so `ExprId` alone is a valid key).  Without this, a DAG-shared input — the
    // same subexpression appearing in many positions, as the M4 tower-recursive
    // integrator routinely produces — is re-traversed once per occurrence,
    // degrading to exponential time on highly-shared expressions.  Mirrors the
    // memoised `engine::is_free_of` (commit `cd76984`).
    let mut cache: std::collections::HashMap<ExprId, bool> = std::collections::HashMap::new();
    is_free_of_var_memo(expr, var, pool, &mut cache)
}

fn is_free_of_var_memo(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    cache: &mut std::collections::HashMap<ExprId, bool>,
) -> bool {
    if expr == var {
        return false;
    }
    if let Some(&hit) = cache.get(&expr) {
        return hit;
    }
    let result = match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => args
            .iter()
            .all(|&a| is_free_of_var_memo(a, var, pool, cache)),
        ExprData::Pow { base, exp } => {
            is_free_of_var_memo(base, var, pool, cache)
                && is_free_of_var_memo(exp, var, pool, cache)
        }
        ExprData::Func { ref args, .. } => args
            .iter()
            .all(|&a| is_free_of_var_memo(a, var, pool, cache)),
        _ => true,
    };
    cache.insert(expr, result);
    result
}

/// Try to extract the rational value of a constant expression.
/// Returns `None` for symbolic constants.
fn to_rational_const(expr: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0.to_i64().map(Rational::from),
        ExprData::Rational(r) => Some(r.0.clone()),
        ExprData::Pow { base, exp } => {
            // Handles c^n for integer c, integer n.
            if let ExprData::Integer(n) = pool.get(exp) {
                if let Some(n_i) = n.0.to_i64() {
                    if let Some(b_r) = to_rational_const(base, pool) {
                        if n_i >= 0 {
                            let mut result = Rational::from(1);
                            for _ in 0..n_i {
                                result *= b_r.clone();
                            }
                            return Some(result);
                        } else {
                            // Negative power: 1 / b^|n|
                            if b_r != 0 {
                                let mut result = Rational::from(1);
                                for _ in 0..(-n_i) {
                                    result *= b_r.clone();
                                }
                                return Some(Rational::from(1) / result);
                            }
                        }
                    }
                }
            }
            None
        }
        ExprData::Mul(args) => {
            let mut result = Rational::from(1);
            for &a in &args {
                result *= to_rational_const(a, pool)?;
            }
            Some(result)
        }
        _ => None,
    }
}

/// Convert a ℚ-polynomial back to a symbolic ExprId.
pub fn qpoly_to_expr(poly: &QPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    let poly = trim(poly.clone());
    if poly.is_empty() {
        return pool.integer(0_i32);
    }

    let mut terms: Vec<ExprId> = Vec::new();
    for (deg, coeff) in poly.iter().enumerate() {
        if *coeff == 0 {
            continue;
        }
        let coeff_expr = rational_to_expr(coeff, pool);
        let term = if deg == 0 {
            coeff_expr
        } else if deg == 1 {
            if *coeff == 1 {
                var
            } else {
                pool.mul(vec![coeff_expr, var])
            }
        } else {
            let power = pool.pow(var, pool.integer(deg as i32));
            if *coeff == 1 {
                power
            } else {
                pool.mul(vec![coeff_expr, power])
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

/// Returns `true` if `target` appears as a sub-expression anywhere inside `expr`.
///
/// Used as a safety guard in multi-level integration: after computing P_n =
/// ∫ c_n dx, we verify that P_n does not contain the current log generator,
/// because the IBP recursion would diverge if it did.
pub fn contains_subexpr(expr: ExprId, target: ExprId, pool: &ExprPool) -> bool {
    if expr == target {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| contains_subexpr(a, target, pool))
        }
        ExprData::Pow { base, exp } => {
            contains_subexpr(base, target, pool) || contains_subexpr(exp, target, pool)
        }
        ExprData::Func { ref args, .. } => args.iter().any(|&a| contains_subexpr(a, target, pool)),
        _ => false,
    }
}

/// Build a symbolic ExprId from a rug::Rational.
pub fn rational_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

// ---------------------------------------------------------------------------
// Constant-factor splitting (shared by exp and log tower paths)
// ---------------------------------------------------------------------------

/// Split `c` into `(K, g)` with `c = K · g`, where `K` collects all factors
/// free of `var` and `g` carries the var-dependent part.  Returns `(1, c)` when
/// there is no constant factor and `(c, 1)` when `c` itself is constant.
pub fn split_const_factor(c: ExprId, var: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let one = pool.integer(1_i32);
    match pool.get(c) {
        ExprData::Mul(args) => {
            let mut consts: Vec<ExprId> = Vec::new();
            let mut vars: Vec<ExprId> = Vec::new();
            for &a in &args {
                if is_free_of_var(a, var, pool) {
                    consts.push(a);
                } else {
                    vars.push(a);
                }
            }
            if consts.is_empty() {
                return (one, c);
            }
            let k = match consts.len() {
                1 => consts[0],
                _ => pool.mul(consts),
            };
            let rest = match vars.len() {
                0 => one,
                1 => vars[0],
                _ => pool.mul(vars),
            };
            (k, rest)
        }
        _ => {
            if is_free_of_var(c, var, pool) {
                (c, one)
            } else {
                (one, c)
            }
        }
    }
}

/// Multiply `core` by the constant `k_const`, collapsing `k_const = 1`.
pub fn apply_const(k_const: ExprId, core: ExprId, pool: &ExprPool) -> ExprId {
    if matches!(pool.get(k_const), ExprData::Integer(n) if n.0 == 1) {
        core
    } else {
        pool.mul(vec![k_const, core])
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }
    fn rat_frac(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    // ∫ exp(x^2) dx: y' + 2x·y = 1 → no solution
    #[test]
    fn rde_exp_x2_nonelementary() {
        let deta = vec![rat(0), rat(2)]; // 2x
        let h = vec![rat(1)]; // 1
        assert!(solve_poly_rde(1, &deta, &h).is_none());
    }

    // ∫ x·exp(x^2) dx: y' + 2x·y = x → solution y = 1/2
    #[test]
    fn rde_x_exp_x2_elementary() {
        let deta = vec![rat(0), rat(2)]; // 2x
        let h = vec![rat(0), rat(1)]; // x
        let sol = solve_poly_rde(1, &deta, &h);
        assert!(sol.is_some(), "expected a polynomial solution");
        let y = sol.unwrap();
        assert_eq!(degree(&y), 0, "solution should be a constant");
        assert_eq!(y[0], rat_frac(1, 2), "y = 1/2");
    }

    // ∫ (2x^2+1)·exp(x^2) dx: y' + 2x·y = 2x²+1 → solution y = x
    #[test]
    fn rde_2x2plus1_exp_x2_elementary() {
        let deta = vec![rat(0), rat(2)]; // 2x
        let h = vec![rat(1), rat(0), rat(2)]; // 1 + 2x^2
        let sol = solve_poly_rde(1, &deta, &h);
        assert!(sol.is_some(), "expected a polynomial solution");
        let y = sol.unwrap();
        assert_eq!(degree(&y), 1, "solution should be linear");
        assert_eq!(y[0], rat(0));
        assert_eq!(y[1], rat(1)); // y = x
    }

    // ∫ x^2·exp(x) dx: y' + y = x^2 → solution y = x^2 - 2x + 2
    #[test]
    fn rde_x2_exp_x_elementary() {
        let deta = vec![rat(1)]; // 1 (constant)
        let h = vec![rat(0), rat(0), rat(1)]; // x^2
        let sol = solve_poly_rde(1, &deta, &h);
        assert!(sol.is_some(), "expected a polynomial solution");
        let y = sol.unwrap();
        assert_eq!(degree(&y), 2);
        assert_eq!(y[0], rat(2)); // constant = 2
        assert_eq!(y[1], rat(-2)); // x coefficient = -2
        assert_eq!(y[2], rat(1)); // x^2 coefficient = 1
    }

    // ∫ x·exp(x) dx: y' + y = x → solution y = x - 1
    #[test]
    fn rde_x_exp_x_elementary() {
        let deta = vec![rat(1)]; // 1
        let h = vec![rat(0), rat(1)]; // x
        let sol = solve_poly_rde(1, &deta, &h);
        assert!(sol.is_some());
        let y = sol.unwrap();
        // y = x - 1
        assert_eq!(y[0], rat(-1));
        assert_eq!(y[1], rat(1));
    }

    // Verify poly_deriv + poly_mul + poly_add consistency
    #[test]
    fn rde_verify_consistency() {
        // If solve_poly_rde returns Some(y), then y' + k·Dη·y == h
        let cases = vec![
            // (k, deta, h)
            (1i64, vec![rat(1)], vec![rat(0), rat(0), rat(1)]), // x^2
            (1, vec![rat(0), rat(2)], vec![rat(0), rat(1)]),    // x*exp(x^2)
            (2, vec![rat(1)], vec![rat(0), rat(1), rat(0), rat(1)]), // x+x^3 (k=2)
        ];
        for (k, deta, h) in cases {
            if let Some(y) = solve_poly_rde(k, &deta, &h) {
                let k_rat = Rational::from(k);
                let lhs = trim(poly_add(
                    &poly_deriv(&y),
                    &poly_scale(&poly_mul(&deta, &y), &k_rat),
                ));
                assert!(
                    polys_equal(&lhs, &h),
                    "verification failed for k={k}: lhs={lhs:?}, h={h:?}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Polynomial RDE over a number field K = ℚ(α)  (Gap E)
    // -----------------------------------------------------------------------

    /// ℚ(√2) = ℚ[t]/(t²−2).
    fn field_sqrt2() -> NumberField {
        NumberField::new(vec![rat(-2), rat(0), rat(1)])
    }

    // ∫ (x + √2)·exp(x) dx = (x + √2 − 1)·exp(x):  y' + y = x + √2  →  y = x + √2 − 1.
    #[test]
    fn rde_k_linear_sqrt2() {
        let field = field_sqrt2();
        let deta: KPoly = vec![vec![rat(1)]]; // Dη = 1 (constant)
        let h: KPoly = vec![vec![rat(0), rat(1)], vec![rat(1)]]; // √2 + x
        let y = solve_poly_rde_k(&field, 1, &deta, &h).expect("elementary");
        // y = x + (√2 − 1): y[0] = −1 + √2, y[1] = 1.
        assert_eq!(trim(y[0].clone()), vec![rat(-1), rat(1)]);
        assert_eq!(trim(y[1].clone()), vec![rat(1)]);
    }

    // ∫ (√3·x² + x)·exp(x) dx: y' + y = √3·x² + x, solvable over ℚ(√3).
    #[test]
    fn rde_k_quadratic_sqrt3() {
        let field = NumberField::new(vec![rat(-3), rat(0), rat(1)]);
        let deta: KPoly = vec![vec![rat(1)]]; // Dη = 1
                                              // h = √3·x² + x: x^0=0, x^1=1, x^2=√3.
        let h: KPoly = vec![NumberField::k_zero(), vec![rat(1)], vec![rat(0), rat(1)]];
        let y = solve_poly_rde_k(&field, 1, &deta, &h).expect("elementary");
        // Check y' + y == h by re-deriving.
        let yp = field.kpoly_deriv(&y);
        let lhs = field.kpoly_add(&yp, &y);
        let lhs = NumberField::kpoly_trim(lhs);
        let h = NumberField::kpoly_trim(h);
        assert_eq!(lhs.len(), h.len());
        for (a, b) in lhs.iter().zip(h.iter()) {
            assert_eq!(trim(a.clone()), trim(b.clone()));
        }
    }

    // ∫ (x + √2)·exp(x²) dx: y' + 2x·y = x + √2 has no polynomial solution
    // (the √2·exp(x²) part is non-elementary) → None.
    #[test]
    fn rde_k_nonelementary_sqrt2_gaussian() {
        let field = field_sqrt2();
        let deta: KPoly = vec![NumberField::k_zero(), vec![rat(2)]]; // 2x
        let h: KPoly = vec![vec![rat(0), rat(1)], vec![rat(1)]]; // √2 + x
        assert!(solve_poly_rde_k(&field, 1, &deta, &h).is_none());
    }

    // Sanity: a ℚ-only RDE solved through the K solver matches the ℚ solver.
    // ∫ x²·exp(x) dx: y' + y = x² → y = x² − 2x + 2.
    #[test]
    fn rde_k_reduces_to_rational_case() {
        let field = field_sqrt2();
        let deta: KPoly = vec![vec![rat(1)]];
        let h: KPoly = vec![NumberField::k_zero(), NumberField::k_zero(), vec![rat(1)]]; // x²
        let y = solve_poly_rde_k(&field, 1, &deta, &h).expect("elementary");
        assert_eq!(trim(y[0].clone()), vec![rat(2)]);
        assert_eq!(trim(y[1].clone()), vec![rat(-2)]);
        assert_eq!(trim(y[2].clone()), vec![rat(1)]);
    }
}
