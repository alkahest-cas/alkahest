//! Polynomial Risch Differential Equation (RDE) solver over ℚ[x].
//!
//! Solves `y' + k·Dη·y = h` where:
//!   - k is a nonzero integer (the monomial degree in the exp tower)
//!   - Dη ∈ ℚ[x] is the derivative of the tower exponent η
//!   - h ∈ ℚ[x] is the integrand coefficient
//!
//! Returns `Some(y)` if a polynomial solution y ∈ ℚ[x] exists, `None` otherwise.
//!
//! **Key decision criterion** (Bronstein 2005, Thm 5.1):
//! When `Dη` has degree `d ≥ 1`, a polynomial solution exists iff `deg(h) ≥ d`.
//! When `Dη` is a nonzero constant (`d = 0`), a polynomial solution always exists.
//!
//! References:
//!   - Bronstein (2005). *Symbolic Integration I*, §5.2, Algorithm 5.1.
//!   - Risch (1969). The problem of integration in finite terms. *Trans. AMS* 139.

use rug::Rational;

/// A polynomial over ℚ, stored as coefficient vector in ascending degree order.
/// `poly[i]` is the coefficient of `x^i`.  The zero polynomial is represented
/// as the empty vector.
pub type QPoly = Vec<Rational>;

// ---------------------------------------------------------------------------
// Basic polynomial arithmetic over ℚ
// ---------------------------------------------------------------------------

/// Trim trailing zero coefficients.
pub fn trim(mut p: QPoly) -> QPoly {
    while p.last().map_or(false, |c| *c == 0) {
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

/// Solve the polynomial Risch Differential Equation over ℚ[x]:
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
    debug_assert!(k != 0, "solve_poly_rde called with k=0");

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
        for i in 0..deg_deta as usize {
            let l = (target_deg - i as i64) as usize;
            if l <= m && l != j {
                // y[l] is already known (l > j since i < deg_deta and target_deg = j+deg_deta).
                rhs -= k_rat.clone() * deta[i].clone() * y[l].clone();
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

/// Convert a symbolic expression to a ℚ[x] polynomial.
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
            let val = n.0.to_i64().unwrap_or(0);
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
                let numer = rat_factor.numer().to_i64().unwrap_or(0);
                let denom = rat_factor.denom().to_i64().unwrap_or(1);
                // We need to scale by rat_factor = numer/denom.
                // Use a temporary vector and scale.
                let mut sub = Vec::new();
                if !collect_qpoly(var_parts[0], var, pool, &mut sub, 1) {
                    return false;
                }
                let scale = Rational::from((numer, denom));
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
    if expr == var {
        return false;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().all(|&a| is_free_of_var(a, var, pool))
        }
        ExprData::Pow { base, exp } => {
            is_free_of_var(base, var, pool) && is_free_of_var(exp, var, pool)
        }
        ExprData::Func { ref args, .. } => args.iter().all(|&a| is_free_of_var(a, var, pool)),
        _ => true,
    }
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

/// Build a symbolic ExprId from a rug::Rational.
pub fn rational_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
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
}
