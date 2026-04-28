//! Internal polynomial utilities for the algebraic Risch engine.
//!
//! All functions work on ExprId values in an ExprPool, performing structural
//! analysis without holding pool locks across recursive calls.

use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Integer;

/// Returns `true` if `expr` is structurally zero.
pub fn is_zero_expr(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(r) => r.0 == 0,
        _ => false,
    }
}

/// Returns `true` if `expr` is structurally one.
#[allow(dead_code)]
pub fn is_one_expr(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0 == 1,
        _ => false,
    }
}

/// Returns `true` if `expr` does not syntactically involve `sub`.
pub fn is_free_of_subexpr(expr: ExprId, sub: ExprId, pool: &ExprPool) -> bool {
    if expr == sub {
        return false;
    }
    let children: Vec<ExprId> = pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::Func { args, .. } => args.clone(),
        _ => vec![],
    });
    children.iter().all(|&c| is_free_of_subexpr(c, sub, pool))
}

/// Returns `true` if `expr` does not involve `var`.
pub fn is_free_of(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    is_free_of_subexpr(expr, var, pool)
}

/// Estimate the polynomial degree of `expr` in `var`.
/// Returns `None` if `expr` is not a polynomial in `var`
/// (e.g., contains transcendental functions, non-integer exponents, or negative powers of var).
pub fn poly_degree_in(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<u32> {
    if expr == var {
        return Some(1);
    }
    if is_free_of(expr, var, pool) {
        return Some(0);
    }
    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut max_d = 0u32;
            for a in &args {
                let d = poly_degree_in(*a, var, pool)?;
                max_d = max_d.max(d);
            }
            Some(max_d)
        }
        ExprData::Mul(args) => {
            let mut total = 0u32;
            for a in &args {
                let d = poly_degree_in(*a, var, pool)?;
                total = total.checked_add(d)?;
            }
            Some(total)
        }
        ExprData::Pow { base, exp } if base == var => {
            match pool.get(exp) {
                ExprData::Integer(n) => {
                    let k: Option<u32> = n.0.to_u32();
                    k
                }
                _ => None,
            }
        }
        ExprData::Pow { base, exp } if is_free_of(base, var, pool) => {
            if is_free_of(exp, var, pool) {
                Some(0)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to interpret `expr` as an integer and return its value.
pub fn as_integer(expr: ExprId, pool: &ExprPool) -> Option<i64> {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    }
}

/// Try to interpret `expr` as a rational p/q; returns (p, q) as i64.
#[allow(dead_code)]
pub fn as_rational(expr: ExprId, pool: &ExprPool) -> Option<(i64, i64)> {
    match pool.get(expr) {
        ExprData::Rational(r) => {
            let p = r.0.numer().to_i64()?;
            let q = r.0.denom().to_i64()?;
            Some((p, q))
        }
        ExprData::Integer(n) => {
            let k = n.0.to_i64()?;
            Some((k, 1))
        }
        _ => None,
    }
}

/// Extract polynomial coefficients of `expr` in `var` as `Vec<Option<ExprId>>`.
/// Index `i` is the coefficient of `var^i`.  Returns `None` if `expr` is not
/// a polynomial or has degree > `max_degree`.
///
/// Coefficients are expressed as ExprIds (may be 0 if the degree is absent).
#[allow(dead_code)]
pub fn extract_poly_coeffs(
    expr: ExprId,
    var: ExprId,
    max_degree: u32,
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    let deg = poly_degree_in(expr, var, pool)?;
    if deg > max_degree {
        return None;
    }
    let zero = pool.integer(0_i32);
    let mut coeffs: Vec<ExprId> = vec![zero; (deg + 1) as usize];
    fill_coeffs(expr, var, &mut coeffs, pool);
    Some(coeffs)
}

/// Recursive helper: accumulate polynomial coefficient contributions into `coeffs`.
/// `coeffs[i]` accumulates the coefficient of `var^i`.
fn fill_coeffs(expr: ExprId, var: ExprId, coeffs: &mut Vec<ExprId>, pool: &ExprPool) {
    let n = coeffs.len() as u32;
    // Make sure vector is large enough for the degree of this term
    let deg = poly_degree_in(expr, var, pool).unwrap_or(0);
    if deg as usize >= coeffs.len() {
        let extra = deg as usize + 1 - coeffs.len();
        let zero = pool.integer(0_i32);
        coeffs.extend(std::iter::repeat(zero).take(extra));
    }
    let _ = n;

    // Helper: accumulate `term` into `slot`, avoiding `Add([0, term])` when slot is already 0.
    fn add_to_slot(slot: &mut ExprId, term: ExprId, pool: &ExprPool) {
        if is_zero_expr(*slot, pool) {
            *slot = term;
        } else {
            *slot = pool.add(vec![*slot, term]);
        }
    }

    match pool.get(expr) {
        _ if is_free_of(expr, var, pool) => {
            add_to_slot(&mut coeffs[0], expr, pool);
        }
        ExprData::Symbol { .. } if expr == var => {
            while coeffs.len() < 2 {
                coeffs.push(pool.integer(0_i32));
            }
            let one = pool.integer(1_i32);
            add_to_slot(&mut coeffs[1], one, pool);
        }
        ExprData::Add(args) => {
            for a in &args {
                fill_coeffs(*a, var, coeffs, pool);
            }
        }
        ExprData::Mul(args) => {
            let (const_parts, var_parts): (Vec<ExprId>, Vec<ExprId>) =
                args.iter().partition(|&&a| is_free_of(a, var, pool));
            let c_factor = match const_parts.len() {
                0 => pool.integer(1_i32),
                1 => const_parts[0],
                _ => pool.mul(const_parts),
            };
            if var_parts.len() == 1 {
                let var_part = var_parts[0];
                let mut sub_coeffs = vec![pool.integer(0_i32); coeffs.len()];
                fill_coeffs(var_part, var, &mut sub_coeffs, pool);
                while sub_coeffs.len() < coeffs.len() {
                    sub_coeffs.push(pool.integer(0_i32));
                }
                for (i, sc) in sub_coeffs.iter().enumerate() {
                    if i >= coeffs.len() {
                        coeffs.push(pool.integer(0_i32));
                    }
                    if !is_zero_expr(*sc, pool) {
                        let term = if is_one_expr(c_factor, pool) { *sc } else { pool.mul(vec![c_factor, *sc]) };
                        add_to_slot(&mut coeffs[i], term, pool);
                    }
                }
            }
        }
        ExprData::Pow { base, exp } if base == var => {
            if let Some(k) = as_integer(exp, pool).and_then(|k| u32::try_from(k).ok()) {
                while coeffs.len() <= k as usize {
                    coeffs.push(pool.integer(0_i32));
                }
                let one = pool.integer(1_i32);
                add_to_slot(&mut coeffs[k as usize], one, pool);
            }
        }
        _ => {
            if is_free_of(expr, var, pool) {
                add_to_slot(&mut coeffs[0], expr, pool);
            }
        }
    }
}

/// Extract (a, b) from a linear polynomial `a*var + b`.
/// Returns ExprIds for integer coefficients via UniPoly, so they're always
/// canonical Integer nodes (no unsimplified `Add([0,n])` artefacts).
pub fn as_linear(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    use crate::poly::UniPoly;
    let up = UniPoly::from_symbolic(expr, var, pool).ok()?;
    let cs = up.coefficients();
    // cs[1] must exist and be non-zero for a truly linear polynomial
    let a_int = cs.get(1)?;
    if *a_int == 0 {
        return None;
    }
    let b_int = cs.first().cloned().unwrap_or_else(|| Integer::from(0));
    let a = pool.integer(a_int.clone());
    let b = pool.integer(b_int);
    Some((a, b))
}

/// Extract (a, b, c) from a quadratic `a*var^2 + b*var + c`.
/// Returns ExprIds for integer coefficients via UniPoly.
pub fn as_quadratic(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId, ExprId)> {
    use crate::poly::UniPoly;
    let up = UniPoly::from_symbolic(expr, var, pool).ok()?;
    let cs = up.coefficients();
    // cs[2] must exist and be non-zero for a truly quadratic polynomial
    let a_int = cs.get(2)?;
    if *a_int == 0 {
        return None;
    }
    let b_int = cs.get(1).cloned().unwrap_or_else(|| Integer::from(0));
    let c_int = cs.first().cloned().unwrap_or_else(|| Integer::from(0));
    let a = pool.integer(a_int.clone());
    let b = pool.integer(b_int);
    let c = pool.integer(c_int);
    Some((a, b, c))
}

/// Get (numerator, denominator) integer coefficients from a UniPoly.
/// Used to extract rational coefficients from the radicand.
pub fn poly_int_coeffs(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Vec<Integer>> {
    use crate::poly::UniPoly;
    let up = UniPoly::from_symbolic(expr, var, pool).ok()?;
    Some(up.coefficients())
}
