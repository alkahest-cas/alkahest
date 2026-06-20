//! Symbolic / parametric Routh–Hurwitz stability analysis.
//!
//! Given a characteristic polynomial
//!
//! ```text
//! p(s) = a_n s^n + a_{n-1} s^{n-1} + … + a_1 s + a_0
//! ```
//!
//! whose coefficients may be *symbolic expressions in free parameters*, this
//! module builds the Routh array and returns the stability condition: the
//! conjunction of strict-positivity constraints on the entries of the first
//! column. The continuous-time system is asymptotically stable (all roots have
//! strictly negative real part) **iff** every first-column entry shares the
//! sign of the leading coefficient (here taken positive) and none vanishes.
//!
//! Unlike [`crate::real::decide`], the body here is allowed to contain free
//! parameters: the output is a *parametric* semialgebraic condition on those
//! parameters rather than a single boolean. For the classic small cases this
//! reproduces the textbook conditions, e.g.
//!
//! - `s^2 + a·s + b`         →  `a > 0 ∧ b > 0`
//! - `s^3 + a·s^2 + b·s + c`  →  `a > 0 ∧ (a·b − c)/a > 0 ∧ c > 0`
//!
//! The construction is purely symbolic (no CAD lift), so it stays cheap even
//! with many parameters; the only guard is a degree cap to bound the array size.

use super::cad::CadError;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprId, ExprPool};
use crate::logic::Formula;
use crate::poly::{poly_normal, resultant, together_parts};

/// Maximum polynomial degree accepted by [`routh_hurwitz`]. The Routh array is
/// `O(n^2)` symbolic entries, each a `2x2` determinant of the row above, so the
/// cost grows quickly with parametric coefficients; this cap bounds runtime.
pub const ROUTH_MAX_DEGREE: usize = 24;

/// Result of a symbolic Routh–Hurwitz analysis.
#[derive(Debug, Clone)]
pub struct RouthHurwitz {
    /// Degree `n` of the characteristic polynomial in the analysis variable.
    pub degree: usize,
    /// Entries of the first column of the Routh array, top (`s^n`) to bottom
    /// (`s^0`), each a (normalised) symbolic expression in the parameters.
    pub first_column: Vec<ExprId>,
    /// The full Routh array, row by row (row 0 corresponds to `s^n`). Rows are
    /// padded with zeros to a common width; useful for inspection / display.
    pub array: Vec<Vec<ExprId>>,
    /// Stability condition: conjunction of `c > 0` over the non-trivial
    /// first-column entries (a literal-`1` entry is omitted as vacuously
    /// positive). [`Formula::True`] when the degree is `0`.
    pub condition: Formula,
}

impl RouthHurwitz {
    /// Intern [`RouthHurwitz::condition`] into the pool as a single predicate
    /// [`ExprId`] (a conjunction of `> 0` atoms).
    pub fn condition_expr(&self, pool: &ExprPool) -> ExprId {
        self.condition.to_expr(pool)
    }
}

/// Extract the dense coefficient vector of `expr` in `var`; index `i` holds the
/// (canonicalised) coefficient of `var^i`. Parameters are treated as symbolic
/// constants. Returns `Err` if `expr` is not a polynomial in `var` or exceeds
/// [`ROUTH_MAX_DEGREE`].
fn coeffs_in_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<Vec<ExprId>, CadError> {
    let deg = poly_degree_in(expr, var, pool).ok_or(CadError::Unsupported(
        "Routh: not a polynomial in the analysis variable",
    ))?;
    if deg as usize > ROUTH_MAX_DEGREE {
        return Err(CadError::Unsupported(
            "Routh: characteristic polynomial degree exceeds the supported cap",
        ));
    }
    let zero = pool.integer(0_i32);
    let mut coeffs = vec![zero; deg as usize + 1];
    fill_coeffs(expr, var, &mut coeffs, pool)?;

    // Canonicalise each coefficient as a multivariate polynomial in the free
    // parameters (everything but `var`). This collapses e.g. `a*b + (-1)*c`
    // into a single normalised term so downstream comparison / display is sane.
    let params: Vec<ExprId> = collect_param_vars(expr, var, pool);
    if !params.is_empty() {
        for c in coeffs.iter_mut() {
            *c = poly_normal(*c, params.clone(), pool)?;
        }
    }
    Ok(coeffs)
}

/// Polynomial degree of `expr` in `var`, or `None` if not polynomial in `var`.
fn poly_degree_in(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<u32> {
    use crate::kernel::expr::ExprData;
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
                max_d = max_d.max(poly_degree_in(*a, var, pool)?);
            }
            Some(max_d)
        }
        ExprData::Mul(args) => {
            let mut total = 0u32;
            for a in &args {
                total = total.checked_add(poly_degree_in(*a, var, pool)?)?;
            }
            Some(total)
        }
        ExprData::Pow { base, exp } if base == var => match pool.get(exp) {
            ExprData::Integer(n) => n.0.to_u32(),
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

fn is_free_of(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::expr::ExprData;
    if expr == var {
        return false;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => args.iter().all(|&a| is_free_of(a, var, pool)),
        ExprData::Pow { base, exp } => is_free_of(base, var, pool) && is_free_of(exp, var, pool),
        ExprData::Integer(_) | ExprData::Rational(_) => true,
        ExprData::Symbol { .. } => expr != var,
        _ => !resultant::collect_free_vars(expr, pool).contains(&var),
    }
}

/// All free symbols of `expr` except `var`, sorted (for deterministic output).
fn collect_param_vars(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut v: Vec<ExprId> = resultant::collect_free_vars(expr, pool)
        .into_iter()
        .filter(|&s| s != var)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Recursive coefficient accumulator: `coeffs[i]` gathers the coefficient of
/// `var^i`.
fn fill_coeffs(
    expr: ExprId,
    var: ExprId,
    coeffs: &mut [ExprId],
    pool: &ExprPool,
) -> Result<(), CadError> {
    use crate::kernel::expr::ExprData;

    fn add_to_slot(slot: &mut ExprId, term: ExprId, pool: &ExprPool) {
        if is_zero_int(*slot, pool) {
            *slot = term;
        } else {
            *slot = pool.add(vec![*slot, term]);
        }
    }

    match pool.get(expr) {
        ExprData::Add(args) => {
            for a in &args {
                fill_coeffs(*a, var, coeffs, pool)?;
            }
            Ok(())
        }
        _ => {
            // Single monomial: factor out the power of `var`.
            let (power, rest) = split_var_power(expr, var, pool)?;
            if power as usize >= coeffs.len() {
                return Err(CadError::Unsupported(
                    "Routh: internal degree miscount while collecting coefficients",
                ));
            }
            add_to_slot(&mut coeffs[power as usize], rest, pool);
            Ok(())
        }
    }
}

/// Given a monomial-like `expr`, return `(k, rest)` such that `expr = var^k *
/// rest` and `rest` is free of `var`. Errors if `var` appears non-polynomially.
fn split_var_power(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<(u32, ExprId), CadError> {
    use crate::kernel::expr::ExprData;
    if expr == var {
        return Ok((1, pool.integer(1_i32)));
    }
    if is_free_of(expr, var, pool) {
        return Ok((0, expr));
    }
    match pool.get(expr) {
        ExprData::Pow { base, exp } if base == var => match pool.get(exp) {
            ExprData::Integer(n) => {
                let k = n.0.to_u32().ok_or(CadError::Unsupported(
                    "Routh: exponent of analysis variable is not a small non-negative integer",
                ))?;
                Ok((k, pool.integer(1_i32)))
            }
            _ => Err(CadError::Unsupported(
                "Routh: analysis variable raised to a non-integer power",
            )),
        },
        ExprData::Mul(args) => {
            let mut power = 0u32;
            let mut rest: Vec<ExprId> = Vec::new();
            for a in &args {
                if is_free_of(*a, var, pool) {
                    rest.push(*a);
                } else {
                    let (k, r) = split_var_power(*a, var, pool)?;
                    power = power
                        .checked_add(k)
                        .ok_or(CadError::Unsupported("Routh: degree overflow in monomial"))?;
                    if !is_one_int(r, pool) {
                        rest.push(r);
                    }
                }
            }
            let rest_expr = if rest.is_empty() {
                pool.integer(1_i32)
            } else if rest.len() == 1 {
                rest[0]
            } else {
                pool.mul(rest)
            };
            Ok((power, rest_expr))
        }
        _ => Err(CadError::Unsupported(
            "Routh: characteristic polynomial is not polynomial in the analysis variable",
        )),
    }
}

fn is_zero_int(e: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::expr::ExprData;
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0)
}

fn is_one_int(e: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::expr::ExprData;
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 1)
}

/// Build the symbolic Routh array and stability condition for the parametric
/// characteristic polynomial `poly` in variable `var`.
///
/// The stability condition is the conjunction `c_0 > 0 ∧ c_1 > 0 ∧ …` over the
/// non-trivial entries of the array's first column, expressed symbolically in
/// the free parameters. (A leading entry equal to the integer `1` is omitted as
/// vacuously positive.)
///
/// # Errors
/// Returns [`CadError::Unsupported`] if `poly` is not polynomial in `var` or
/// exceeds [`ROUTH_MAX_DEGREE`].
pub fn routh_hurwitz(poly: ExprId, var: ExprId, pool: &ExprPool) -> Result<RouthHurwitz, CadError> {
    let coeffs = coeffs_in_var(poly, var, pool)?;
    let n = coeffs.len().saturating_sub(1);
    if n == 0 {
        return Ok(RouthHurwitz {
            degree: 0,
            first_column: coeffs.clone(),
            array: vec![coeffs],
            condition: Formula::True,
        });
    }

    // coeffs[i] = coefficient of s^i; descending order is a_n .. a_0.
    let desc: Vec<ExprId> = coeffs.iter().rev().copied().collect();

    // Width of each row.
    let width = n / 2 + 1;
    let params = collect_param_vars(poly, var, pool);
    let zero = pool.integer(0_i32);

    // First two rows from the coefficients (even / odd index split).
    let mut rows: Vec<Vec<ExprId>> = Vec::new();
    let mut row0 = Vec::with_capacity(width);
    let mut row1 = Vec::with_capacity(width);
    let mut idx = 0usize;
    while idx <= n {
        row0.push(if idx < desc.len() { desc[idx] } else { zero });
        let j = idx + 1;
        row1.push(if j <= n && j < desc.len() {
            desc[j]
        } else {
            zero
        });
        idx += 2;
    }
    while row0.len() < width {
        row0.push(zero);
    }
    while row1.len() < width {
        row1.push(zero);
    }
    rows.push(row0);
    rows.push(row1);

    // Remaining n-1 rows via the Routh recurrence.
    for r in 2..=n {
        let above = rows[r - 1].clone();
        let above2 = rows[r - 2].clone();
        let pivot = above[0];
        let mut new_row = Vec::with_capacity(width);
        for j in 0..width {
            let a = above2[0];
            let b = if j + 1 < above2.len() {
                above2[j + 1]
            } else {
                zero
            };
            let c = pivot;
            let d = if j + 1 < above.len() {
                above[j + 1]
            } else {
                zero
            };
            // Routh entry = (c*b - a*d) / c.
            let cb = pool.mul(vec![c, b]);
            let ad = pool.mul(vec![a, d]);
            let neg_ad = pool.mul(vec![pool.integer(-1_i32), ad]);
            let num = pool.add(vec![cb, neg_ad]);
            let entry = if is_one_int(pivot, pool) {
                num
            } else {
                let inv = pool.pow(pivot, pool.integer(-1_i32));
                pool.mul(vec![num, inv])
            };
            let entry = if params.is_empty() {
                entry
            } else {
                // Best-effort canonicalisation; division can produce a rational
                // function poly_normal cannot handle, so fall back to raw entry.
                poly_normal(entry, params.clone(), pool).unwrap_or(entry)
            };
            new_row.push(entry);
        }
        rows.push(new_row);
    }

    // First column, top to bottom. Each divided Routh entry is a rational
    // function whose denominator is a product of earlier (positive) pivots.
    // Cancelling to a single fraction and taking the *numerator* yields the
    // clean textbook stability polynomials (e.g. `a·b − c` for the cubic):
    // since every pivot in the chain is required positive, `num/denom > 0`
    // is equivalent to `num > 0` under the conjunction of prior conditions.
    let mut first_column: Vec<ExprId> = Vec::with_capacity(rows.len());
    let mut condition = Formula::True;
    for row in &rows {
        let entry = row[0];
        if is_one_int(entry, pool) {
            first_column.push(entry);
            continue;
        }
        // Numerator of the cancelled fraction over the parameters. For numeric
        // instances (`params` empty) this folds the divided entry into a single
        // rational constant; the sign of the numerator is the sign of the entry
        // (the denominator is a product of prior positive pivots).
        let numer = match together_parts(entry, params.clone(), pool) {
            Ok((numer, _denom)) => numer,
            Err(_) => entry,
        };
        // Fold cosmetic artefacts such as `1 * a` into `a`.
        let cond_lhs = crate::simplify::simplify(numer, pool).value;
        first_column.push(cond_lhs);
        if is_one_int(cond_lhs, pool) {
            continue;
        }
        let atom = Formula::Atom {
            kind: PredicateKind::Gt,
            args: vec![cond_lhs, zero],
        };
        condition = match condition {
            Formula::True => atom,
            other => Formula::and(other, atom),
        };
    }

    Ok(RouthHurwitz {
        degree: n,
        first_column,
        array: rows,
        condition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::poly::{real_roots, UniPoly};

    // Helper: collect lhs of each `Gt(lhs, 0)` atom in a conjunction tree.
    fn collect_gt_lhs(f: &Formula, out: &mut Vec<ExprId>) {
        match f {
            Formula::Atom {
                kind: PredicateKind::Gt,
                args,
            } => out.push(args[0]),
            Formula::And(a, b) => {
                collect_gt_lhs(a, out);
                collect_gt_lhs(b, out);
            }
            _ => {}
        }
    }

    /// Count real roots strictly in the positive half-line (isolating interval
    /// lower bound > 0). A positive real root is a definite instability witness.
    fn count_positive_real_roots(poly: ExprId, var: ExprId, pool: &ExprPool) -> usize {
        let up = UniPoly::from_symbolic(poly, var, pool).unwrap();
        let sf = up.squarefree_part();
        if sf.is_zero() {
            return 0;
        }
        real_roots(&sf)
            .unwrap()
            .into_iter()
            .filter(|iv| iv.lo > 0)
            .count()
    }

    #[test]
    fn quadratic_parametric_condition() {
        let p = ExprPool::new();
        let s = p.symbol("s", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        // s^2 + a*s + b
        let poly = p.add(vec![p.pow(s, p.integer(2_i32)), p.mul(vec![a, s]), b]);
        let rh = routh_hurwitz(poly, s, &p).unwrap();
        assert_eq!(rh.degree, 2);
        assert_eq!(rh.first_column.len(), 3);
        // Condition: a > 0 ∧ b > 0
        let mut atoms = Vec::new();
        collect_gt_lhs(&rh.condition, &mut atoms);
        assert_eq!(atoms.len(), 2);
        assert!(atoms.contains(&a));
        assert!(atoms.contains(&b));
    }

    #[test]
    fn cubic_parametric_condition() {
        let p = ExprPool::new();
        let s = p.symbol("s", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let c = p.symbol("c", Domain::Real);
        // s^3 + a*s^2 + b*s + c
        let poly = p.add(vec![
            p.pow(s, p.integer(3_i32)),
            p.mul(vec![a, p.pow(s, p.integer(2_i32))]),
            p.mul(vec![b, s]),
            c,
        ]);
        let rh = routh_hurwitz(poly, s, &p).unwrap();
        assert_eq!(rh.degree, 3);
        // first column: [1, a, (a*b - c)/a, c]; conditions a>0, (a*b-c)/a>0, c>0.
        let mut atoms = Vec::new();
        collect_gt_lhs(&rh.condition, &mut atoms);
        assert_eq!(atoms.len(), 3);
        assert!(atoms.contains(&a));
        assert!(atoms.contains(&c));
    }

    #[test]
    fn numeric_stable_instance_passes() {
        // s^3 + 2 s^2 + 3 s + 1: a=2,b=3,c=1 -> a>0,c>0,ab-c=5>0 => STABLE.
        let p = ExprPool::new();
        let s = p.symbol("s", Domain::Real);
        let poly = p.add(vec![
            p.pow(s, p.integer(3_i32)),
            p.mul(vec![p.integer(2_i32), p.pow(s, p.integer(2_i32))]),
            p.mul(vec![p.integer(3_i32), s]),
            p.integer(1_i32),
        ]);
        // Known-stable instance: no real root with a non-negative isolating
        // interval lower bound (its single real root is strictly negative).
        assert_eq!(count_positive_real_roots(poly, s, &p), 0);
        let rh = routh_hurwitz(poly, s, &p).unwrap();
        for &e in &rh.first_column {
            let up = UniPoly::from_symbolic(e, s, &p).unwrap();
            let val = up.eval_rational(&rug::Rational::from(0));
            assert!(val > 0, "entry not positive: {val}");
        }
    }

    #[test]
    fn numeric_unstable_instance_excluded() {
        // s^3 + s^2 + s + 6: a=1,b=1,c=6 -> ab-c = -5 < 0 => UNSTABLE.
        let p = ExprPool::new();
        let s = p.symbol("s", Domain::Real);
        let poly = p.add(vec![
            p.pow(s, p.integer(3_i32)),
            p.pow(s, p.integer(2_i32)),
            s,
            p.integer(6_i32),
        ]);
        let rh = routh_hurwitz(poly, s, &p).unwrap();
        let mut any_nonpos = false;
        for &e in &rh.first_column {
            let up = UniPoly::from_symbolic(e, s, &p).unwrap();
            let val = up.eval_rational(&rug::Rational::from(0));
            if val <= 0 {
                any_nonpos = true;
            }
        }
        assert!(
            any_nonpos,
            "unstable instance should violate a Routh condition"
        );
    }

    #[test]
    fn numeric_positive_root_instance_cross_check() {
        // s^2 - 3 s + 2 = (s - 1)(s - 2): two positive real roots => UNSTABLE,
        // and a = -3 < 0 so the Routh condition a > 0 is violated. This ties the
        // symbolic condition to actual root locations.
        let p = ExprPool::new();
        let s = p.symbol("s", Domain::Real);
        let poly = p.add(vec![
            p.pow(s, p.integer(2_i32)),
            p.mul(vec![p.integer(-3_i32), s]),
            p.integer(2_i32),
        ]);
        // Two positive real roots => definitely unstable.
        assert_eq!(count_positive_real_roots(poly, s, &p), 2);
        let rh = routh_hurwitz(poly, s, &p).unwrap();
        // First column is [1, -3, 2]; the entry -3 violates positivity.
        let mut any_nonpos = false;
        for &e in &rh.first_column {
            let up = UniPoly::from_symbolic(e, s, &p).unwrap();
            if up.eval_rational(&rug::Rational::from(0)) <= 0 {
                any_nonpos = true;
            }
        }
        assert!(any_nonpos, "positive-root instance must violate Routh");
    }

    #[test]
    fn unsupported_non_polynomial_errors() {
        let p = ExprPool::new();
        let s = p.symbol("s", Domain::Real);
        // 1/s is not polynomial in s.
        let poly = p.add(vec![p.pow(s, p.integer(-1_i32)), p.integer(1_i32)]);
        assert!(routh_hurwitz(poly, s, &p).is_err());
    }
}
