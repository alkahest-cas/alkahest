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
        ExprData::Pow { base, exp } if base == var => match pool.get(exp) {
            ExprData::Integer(n) => {
                let k: Option<u32> = n.0.to_u32();
                k
            }
            _ => None,
        },
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
///
/// Every caller uses this on an exponent or an integer coefficient, so it folds
/// the arithmetic the parser leaves behind — `x^(-1)` parses to `x^(1 · -1)`,
/// not to a bare `Integer(-1)` node.  Reading only `Integer` made the algebraic
/// engine decompose `1/(√x·(1+√x))` but decline the identical
/// `(√x·(1+√x))^(-1)`.
pub fn as_integer(expr: ExprId, pool: &ExprPool) -> Option<i64> {
    crate::integrate::risch::tower::literal_integer(expr, pool)
}

/// Extract (a, b) from a linear polynomial `a*var + b`.
/// Returns ExprIds for integer coefficients via UniPoly, so they're always
/// canonical Integer nodes (no unsimplified `Add([0,n])` artefacts).
pub fn as_linear(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
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
pub fn poly_int_coeffs(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<Vec<Integer>> {
    use crate::poly::UniPoly;
    let up = UniPoly::from_symbolic(expr, var, pool).ok()?;
    Some(up.coefficients())
}
