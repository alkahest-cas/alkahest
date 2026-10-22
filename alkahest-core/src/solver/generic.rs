//! Decisions about expressions in the **free parameters**, under the
//! generic-parameter reading of a parametric solve.
//!
//! `solve([a·x − b], [x]) → b/a` is the answer for every `a, b` off a proper
//! algebraic subset (here `a = 0`).  Every decision the back-substitution makes
//! about a parametric quantity is therefore a question about a *function* of
//! the parameters — "is this leading coefficient the zero function?", "does this
//! candidate make the residual the zero function?" — not about a number.
//!
//! The ball checker in [`super::verify`] cannot answer those (a parameter is
//! not a number), and before this module existed the solver answered all of
//! them with "assume non-zero".  That is how `solve([a·x − b·x·y, −c·y + d·x·y])`
//! came to divide by `b·y − a` at `y = a/b` and return `x = 0·0⁻¹`, and how a
//! candidate that is not a solution for *any* parameter value was returned
//! unexamined.
//!
//! Two deciders, strongest first:
//!
//! * **Exact.** A rational function of the parameters is put over a common
//!   denominator ([`super::rational::clear_denominators`]); it is the zero
//!   function exactly when the numerator is the zero polynomial.  A proof both
//!   ways.
//! * **Probing.** An expression with radicals is evaluated in rigorous ball
//!   arithmetic at a few fixed rational parameter points.  A ball separated from
//!   zero at one point proves the function is not identically zero; a ball that
//!   straddles zero proves nothing and is reported as [`Generic::Unknown`].

use super::{gbpoly_to_expr, rational, rational_to_expr, rational_value, substitute_solution};
use super::{verify, ExprId, ExprPool, GbPoly};
use rug::Rational;

/// What could be established about an expression as a function of the free
/// parameters.
#[derive(Clone, Debug)]
pub(super) enum Generic {
    /// The zero function (proved exactly).
    Zero,
    /// Not the zero function.  Carries the cleared numerator when the proof
    /// was exact, so the caller can state *which* factors must not vanish.
    Nonzero(Option<GbPoly>),
    /// Undefined at every probe point (`0·0⁻¹`): not a value at all.
    Undefined,
    /// Nothing proved either way.
    Unknown,
}

/// Number of parameter points [`probe`] evaluates at.
const PROBE_POINTS: usize = 3;

const PROBE_NUMERATORS: [i64; 32] = [
    37, 53, 71, 29, 83, 47, 61, 43, 59, 67, 79, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
    139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
];
const PROBE_DENOMINATORS: [i64; PROBE_POINTS] = [11, 13, 17];

/// The value given to parameter `i` at probe point `point`: a positive
/// non-integer rational, distinct across parameters, so a probe is consistent
/// with parameters declared `Positive` and does not land on the coincidences
/// (`a = b`, `a = 1`) where a generic statement is most likely to fail.
pub(super) fn probe_value(point: usize, i: usize) -> Rational {
    let k = 3 * i + point;
    let num =
        PROBE_NUMERATORS[k % PROBE_NUMERATORS.len()] + 197 * (k / PROBE_NUMERATORS.len()) as i64;
    Rational::from((num, PROBE_DENOMINATORS[point]))
}

/// `expr` with every parameter (`all_vars[n_solve..]`) replaced by its value at
/// probe point `point`.
fn at_probe(
    expr: ExprId,
    all_vars: &[ExprId],
    n_solve: usize,
    point: usize,
    pool: &ExprPool,
) -> ExprId {
    let params = &all_vars[n_solve..];
    let vals: Vec<ExprId> = (0..params.len())
        .map(|i| rational_to_expr(&probe_value(point, i), pool))
        .collect();
    let mut memo = std::collections::HashMap::new();
    substitute_solution(expr, params, &vals, pool, &mut memo)
}

/// Is `expr` — a quantity in the free parameters only — the zero function?
pub(super) fn status(
    expr: ExprId,
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> Generic {
    if let Some(v) = rational_value(expr, pool) {
        return if v == 0 {
            Generic::Zero
        } else {
            Generic::Nonzero(None)
        };
    }
    if let Ok(c) = rational::clear_denominators(expr, all_vars, pool) {
        return if c.numer.is_zero() {
            Generic::Zero
        } else {
            Generic::Nonzero(Some(c.numer))
        };
    }
    probe(expr, all_vars, n_solve, pool)
}

/// Evaluate at the probe points.  See the module documentation.
fn probe(expr: ExprId, all_vars: &[ExprId], n_solve: usize, pool: &ExprPool) -> Generic {
    let mut undefined = 0;
    for point in 0..PROBE_POINTS {
        let at = at_probe(expr, all_vars, n_solve, point, pool);
        match verify::CBallEval::default().eval(at, pool) {
            Ok(b) if b.excludes_zero() => return Generic::Nonzero(None),
            Ok(_) => {}
            Err(verify::VerifyGap::Undefined) => undefined += 1,
            Err(verify::VerifyGap::Unsupported) => return Generic::Unknown,
        }
    }
    if undefined == PROBE_POINTS {
        Generic::Undefined
    } else {
        Generic::Unknown
    }
}

/// Does a solution coordinate denote a value for generic parameters?
///
/// `false` only when it is undefined at every probe point *and* cannot be
/// cleared to a fraction with a non-zero denominator — the `0·0⁻¹` a division
/// by an identically-zero coefficient leaves behind.
pub(super) fn is_undefined(
    expr: ExprId,
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> bool {
    if rational_value(expr, pool).is_some() {
        return false;
    }
    if rational::clear_denominators(expr, all_vars, pool).is_ok() {
        return false;
    }
    matches!(probe(expr, all_vars, n_solve, pool), Generic::Undefined)
}

/// A [`MultiPoly`](crate::poly::MultiPoly) over the solver's indeterminates as
/// a [`GbPoly`] in `n_vars` variables.
pub(super) fn multipoly_to_gbpoly(mp: &crate::poly::MultiPoly, n_vars: usize) -> GbPoly {
    let mut out = GbPoly::zero(n_vars);
    for (exp, c) in &mp.terms {
        let mut e = exp.clone();
        e.resize(n_vars, 0);
        out.terms.insert(e, Rational::from(c.clone()));
    }
    out
}

/// The irreducible factors of `numer` that a statement "`numer ≠ 0`" actually
/// constrains: non-constant ones, minus those the ambient assumptions already
/// prove non-zero (a parameter declared `Positive`).  Empty means the
/// non-vanishing is proved, not assumed.
pub(super) fn nonzero_hypotheses(
    numer: &GbPoly,
    all_vars: &[ExprId],
    pool: &ExprPool,
) -> Vec<ExprId> {
    let mp = super::gbpoly_to_multipoly(numer, all_vars);
    let factors: Vec<GbPoly> = match mp.factor_z() {
        Ok(f) => f
            .factors
            .iter()
            .map(|(p, _)| multipoly_to_gbpoly(p, all_vars.len()))
            .collect(),
        Err(_) => vec![numer.clone()],
    };
    let mut out = Vec::new();
    for f in factors {
        let constant = f.terms.keys().all(|e| e.iter().all(|&k| k == 0));
        if constant {
            continue;
        }
        let Some(e) = gbpoly_to_expr(&f, all_vars, pool) else {
            continue;
        };
        use crate::simplify::assumptions::{assumed_sign, Sign};
        if matches!(assumed_sign(e, pool), Some(Sign::Positive | Sign::Negative)) {
            continue;
        }
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

/// Split one equation `p = 0` of a parametric system into its distinct
/// irreducible factors over ℚ[unknowns, parameters].
///
/// Returns the factors that mention an unknown — `p = 0` is the union of their
/// zero sets — and, separately, the factors in the parameters only, which
/// vanish for no generic parameter value and are therefore *assumed* non-zero.
/// `None` when the factoriser declines.
pub(super) fn split_factors(
    p: &GbPoly,
    all_vars: &[ExprId],
    n_solve: usize,
) -> Option<(Vec<GbPoly>, Vec<GbPoly>)> {
    let mp = super::gbpoly_to_multipoly(p, all_vars);
    let fac = mp.factor_z().ok()?;
    let mut with_unknown = Vec::new();
    let mut params_only = Vec::new();
    for (f, _) in &fac.factors {
        let g = multipoly_to_gbpoly(f, all_vars.len());
        let mentions_unknown = g
            .terms
            .keys()
            .any(|e| e.iter().take(n_solve).any(|&k| k > 0));
        if mentions_unknown {
            with_unknown.push(g);
        } else if !g.terms.keys().all(|e| e.iter().all(|&k| k == 0)) {
            params_only.push(g);
        }
    }
    Some((with_unknown, params_only))
}
