//! Truncated Taylor / Laurent series with symbolic [`crate::kernel::ExprData::BigO`] remainder (V2-15).

use crate::diff::{diff, DiffError};
use crate::flint::FlintPoly;
use crate::kernel::{subs, Domain, ExprData, ExprId, ExprPool};
use crate::poly::{RationalFunction, UniPoly};
use crate::simplify::simplify;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of [`series`] — truncated expansion plus big-O bound as one [`ExprId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Series(pub ExprId);

impl Series {
    pub fn expr(self) -> ExprId {
        self.0
    }
}

#[derive(Debug)]
pub enum SeriesError {
    /// Differentiation failed while forming Taylor coefficients.
    Diff(DiffError),
    /// `order` must be positive.
    InvalidOrder,
}

impl fmt::Display for SeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeriesError::Diff(e) => write!(f, "{e}"),
            SeriesError::InvalidOrder => write!(f, "series order must be >= 1"),
        }
    }
}

impl std::error::Error for SeriesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SeriesError::Diff(e) => Some(e),
            SeriesError::InvalidOrder => None,
        }
    }
}

impl crate::errors::AlkahestError for SeriesError {
    fn code(&self) -> &'static str {
        match self {
            SeriesError::Diff(_) => "E-SERIES-001",
            SeriesError::InvalidOrder => "E-SERIES-002",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            SeriesError::Diff(_) => Some(
                "ensure all functions are registered primitives with differentiation rules",
            ),
            SeriesError::InvalidOrder => Some("pass order >= 1 (exclusive truncation degree in x)"),
        }
    }
}

impl From<DiffError> for SeriesError {
    fn from(e: DiffError) -> Self {
        SeriesError::Diff(e)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Truncated Taylor or Laurent expansion of `expr` in `var` about `point`.
///
/// Let `h = var - point`. The returned expression has the shape
/// `⋯ + O(h^k)` where `k = order` for analytic series (`valuation ≥ 0`), and
/// `k = 1` when a polar term (`valuation < 0`) is present — matching the
/// Laurent examples in the roadmap (`1/x` about `0` gives `x⁻¹ + O(x)`).
///
/// The `order` parameter matches the Taylor convention used in the roadmap:
/// include powers `h^e` with `valuation ≤ e < order` when `valuation ≥ 0`, and
/// when `valuation < 0` include the polar tail using `order` Taylor coefficients
/// of the analytic factor `h^{-valuation} · f`.
pub fn series(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Series, SeriesError> {
    if order == 0 {
        return Err(SeriesError::InvalidOrder);
    }

    let xi = pool.symbol("__sxp", Domain::Real);
    let mut map = HashMap::new();
    map.insert(var, pool.add(vec![point, xi]));
    let shifted = subs(expr, &map, pool);

    let h_expr = expansion_increment(pool, var, point);

    series_matched_laurent(shifted, xi, h_expr, order, pool)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn factorial_u32(n: u32) -> rug::Integer {
    let mut r = rug::Integer::from(1);
    for i in 2..=n {
        r *= i;
    }
    r
}

fn expansion_increment(pool: &ExprPool, var: ExprId, point: ExprId) -> ExprId {
    match pool.get(point) {
        ExprData::Integer(n) if n.0 == 0 => var,
        _ => pool.add(vec![var, pool.mul(vec![pool.integer(-1_i32), point])]),
    }
}

fn laurent_big_o_pow(valuation: i32, order: u32) -> i64 {
    if valuation < 0 {
        1
    } else {
        order as i64
    }
}

fn is_structural_zero(id: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(id), ExprData::Integer(n) if n.0 == 0)
}

fn collect_atom_factors(expr: ExprId, pool: &ExprPool) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let n = pool.with(exp, |d| match d {
                ExprData::Integer(i) => Some(i.0.clone()),
                _ => None,
            })?;
            if n > 0 {
                Some((vec![expr], vec![]))
            } else if n < 0 {
                let mag = (-n).to_u32()?;
                let pos_exp = pool.integer(mag as i64);
                Some((vec![], vec![pool.pow(base, pos_exp)]))
            } else {
                Some((vec![pool.integer(1_i32)], vec![]))
            }
        }
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) | ExprData::Symbol { .. }
        | ExprData::Func { .. } => Some((vec![expr], vec![])),
        ExprData::Add(_)
        | ExprData::Mul(_)
        | ExprData::Piecewise { .. }
        | ExprData::Predicate { .. }
        | ExprData::Forall { .. }
        | ExprData::Exists { .. }
        | ExprData::BigO(_) => None,
    }
}

fn collect_term_factors(expr: ExprId, pool: &ExprPool) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    match pool.get(expr) {
        ExprData::Mul(args) => {
            let mut nums = Vec::new();
            let mut dens = Vec::new();
            for &a in &args {
                let (n, d) = collect_atom_factors(a, pool)?;
                nums.extend(n);
                dens.extend(d);
            }
            Some((nums, dens))
        }
        _ => collect_atom_factors(expr, pool),
    }
}

fn product_sorted(pool: &ExprPool, factors: Vec<ExprId>) -> ExprId {
    match factors.len() {
        0 => pool.integer(1_i32),
        1 => factors[0],
        _ => pool.mul(factors),
    }
}

fn unipoly_valuation(p: &UniPoly) -> Option<u32> {
    for (i, c) in p.coefficients().into_iter().enumerate() {
        if c != 0 {
            return Some(i as u32);
        }
    }
    None
}

fn unipoly_strip_low(p: &UniPoly, k: u32) -> UniPoly {
    let coeffs: Vec<rug::Integer> = p
        .coefficients()
        .into_iter()
        .skip(k as usize)
        .collect();
    UniPoly {
        var: p.var,
        coeffs: FlintPoly::from_rug_coefficients(&coeffs),
    }
}

fn unipoly_to_expr(poly: &UniPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    let coeffs = poly.coefficients();
    if coeffs.is_empty() {
        return pool.integer(0_i32);
    }
    let summands: Vec<ExprId> = coeffs
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(deg, coeff)| {
            let c_id = pool.integer(coeff.clone());
            if deg == 0 {
                c_id
            } else {
                let exp_id = pool.integer(deg as i64);
                let x_pow = if deg == 1 {
                    var
                } else {
                    pool.pow(var, exp_id)
                };
                if *coeff == 1 {
                    x_pow
                } else {
                    pool.mul(vec![c_id, x_pow])
                }
            }
        })
        .collect();

    match summands.len() {
        0 => pool.integer(0_i32),
        1 => summands[0],
        _ => pool.add(summands),
    }
}

fn taylor_coefficients(
    mut cur: ExprId,
    xi: ExprId,
    num: u32,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, SeriesError> {
    let mut mapping = HashMap::new();
    mapping.insert(xi, pool.integer(0_i32));
    let mut out = Vec::with_capacity(num as usize);
    for k in 0..num {
        let ev = subs(cur, &mapping, pool);
        let simp = simplify(ev, pool).value;
        let fc = factorial_u32(k);
        let inv_fact = pool.rational(rug::Integer::from(1), fc);
        let coeff = simplify(pool.mul(vec![simp, inv_fact]), pool).value;
        out.push(coeff);
        if k + 1 < num {
            cur = diff(cur, xi, pool)?.value;
        }
    }
    Ok(out)
}

fn assemble_series(
    coeffs: &[ExprId],
    valuation: i32,
    h_expr: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Series {
    let mut terms = Vec::new();
    for (k, coeff) in coeffs.iter().enumerate() {
        if is_structural_zero(*coeff, pool) {
            continue;
        }
        let exp = valuation + k as i32;
        let pow_term = if exp == 0 {
            pool.integer(1_i32)
        } else if exp == 1 {
            h_expr
        } else {
            pool.pow(h_expr, pool.integer(exp as i64))
        };
        terms.push(pool.mul(vec![*coeff, pow_term]));
    }
    let big_o_pow = laurent_big_o_pow(valuation, order);
    let o_term = pool.big_o(pool.pow(h_expr, pool.integer(big_o_pow)));
    terms.push(o_term);
    Series(pool.add(terms))
}

fn series_matched_laurent(
    shifted: ExprId,
    xi: ExprId,
    h_expr: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Series, SeriesError> {
    let (nums, dens) = match collect_term_factors(shifted, pool) {
        Some(p) => p,
        None => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(assemble_series(&coeffs, 0, h_expr, order, pool));
        }
    };

    let n_expr = product_sorted(pool, nums);
    let d_expr = product_sorted(pool, dens);

    let rf = match RationalFunction::from_symbolic(n_expr, d_expr, vec![xi], pool) {
        Ok(r) => r,
        Err(_) => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(assemble_series(&coeffs, 0, h_expr, order, pool));
        }
    };

    if rf.numer.is_zero() {
        let o = pool.big_o(pool.pow(h_expr, pool.integer(laurent_big_o_pow(0, order))));
        return Ok(Series(pool.add(vec![pool.integer(0_i32), o])));
    }

    let n_uni = match UniPoly::from_symbolic(rf.numer.to_expr(pool), xi, pool) {
        Ok(u) => u,
        Err(_) => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(assemble_series(&coeffs, 0, h_expr, order, pool));
        }
    };
    let d_uni = match UniPoly::from_symbolic(rf.denom.to_expr(pool), xi, pool) {
        Ok(u) => u,
        Err(_) => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(assemble_series(&coeffs, 0, h_expr, order, pool));
        }
    };

    let vn = match unipoly_valuation(&n_uni) {
        Some(v) => v,
        None => {
            let o = pool.big_o(pool.pow(h_expr, pool.integer(laurent_big_o_pow(0, order))));
            return Ok(Series(pool.add(vec![pool.integer(0_i32), o])));
        }
    };
    let vd = match unipoly_valuation(&d_uni) {
        Some(v) => v,
        None => {
            let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
            return Ok(assemble_series(&coeffs, 0, h_expr, order, pool));
        }
    };

    let valuation = vn as i32 - vd as i32;
    let n0 = unipoly_strip_low(&n_uni, vn);
    let d0 = unipoly_strip_low(&d_uni, vd);

    let d0c = d0.coefficients();
    if d0c.is_empty() || d0c[0] == 0 {
        let coeffs = taylor_coefficients(shifted, xi, order, pool)?;
        return Ok(assemble_series(&coeffs, 0, h_expr, order, pool));
    }

    let n0_e = unipoly_to_expr(&n0, xi, pool);
    let d0_e = unipoly_to_expr(&d0, xi, pool);
    let inv_d = pool.pow(d0_e, pool.integer(-1_i32));
    let g = simplify(pool.mul(vec![n0_e, inv_d]), pool).value;

    let num_taylor: u32 = if valuation < 0 {
        order
    } else {
        (order as i32 - valuation).max(0) as u32
    };

    if num_taylor == 0 {
        let o = pool.big_o(pool.pow(h_expr, pool.integer(laurent_big_o_pow(valuation, order))));
        return Ok(Series(o));
    }

    let coeffs = taylor_coefficients(g, xi, num_taylor, pool)?;
    Ok(assemble_series(&coeffs, valuation, h_expr, order, pool))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprData};

    fn contains_big_o(id: ExprId, pool: &ExprPool) -> bool {
        match pool.get(id) {
            ExprData::BigO(_) => true,
            ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|e| contains_big_o(*e, pool)),
            ExprData::Pow { base, exp } => contains_big_o(base, pool) || contains_big_o(exp, pool),
            ExprData::Func { args, .. } => args.iter().any(|e| contains_big_o(*e, pool)),
            _ => false,
        }
    }

    #[test]
    fn series_cos_about_zero_has_big_o() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0);
        let cx = p.func("cos", vec![x]);
        let s = series(cx, x, z, 6, &p).unwrap();
        assert!(contains_big_o(s.expr(), &p));
    }

    #[test]
    fn series_inv_x_laurent_has_big_o() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0);
        let ix = p.pow(x, p.integer(-1));
        let s = series(ix, x, z, 4, &p).unwrap();
        assert!(contains_big_o(s.expr(), &p));
    }
}
