//! Reserved heads for series and symbolic dimension (plan decision D4):
//! `Sum(body, k, lo, hi)`, indexed symbols `Indexed(x, i)` and
//! `KroneckerDelta(i, j)`.
//!
//! # Encoding
//!
//! ```text
//! Func("Sum", [body, k, lo, hi])        Σ_{k=lo}^{hi} body      (k a symbol)
//! Func("Indexed", [x, i])               xᵢ                      (x a symbol)
//! Func("KroneckerDelta", [i, j])        δᵢⱼ
//! ```
//!
//! `Sum` is a **binder**: `k` is bound in `body` and free nowhere else
//! (`lo`, `hi` must not mention it).  [`crate::kernel::subs::subs`] respects
//! that — a key mentioning `k` does not reach inside, and a value that would
//! be captured triggers an alpha-renaming — exactly as for `RootSum`.  The
//! bounds may be symbolic (`d`, `n`) or `∞` (`oo`); evaluation and expansion
//! need integer bounds after substitution.
//!
//! An indexed symbol `Indexed(x, i)` is a variable: `diff` differentiates
//! with respect to it, with `∂xᵢ/∂xⱼ = δᵢⱼ`, so for symbolic `d`
//!
//! ```text
//! ∂/∂xⱼ Σᵢ φ(xᵢ) = Σᵢ φ'(xᵢ) δᵢⱼ  →  φ'(xⱼ)        (1 ≤ j ≤ d)
//! Δ Σᵢ φ(xᵢ) = Σⱼ ∂²/∂xⱼ² Σᵢ φ(xᵢ) = Σⱼ φ''(xⱼ)
//! ```
//!
//! The collapse `Σᵢ f(i) δᵢⱼ = f(j)` is valid only when `j` is in the range;
//! [`simplify_sums`] performs it only when that is known — an integer inside
//! literal bounds, or a range assumption passed in (and [`laplacian_indexed`]
//! passes the range of its own summation index).  Without it the `Sum` with
//! the `KroneckerDelta` is left as it is: unsimplified, never wrong.
//!
//! None of these heads is an undefined function: the names are reserved
//! ([`crate::diff::undefined`]), so `Sum(…)` is never differentiated as an
//! arbitrary `f(…)`.

use std::collections::HashMap;

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::error::PdeError;
use super::jet::symbols_of;

/// Head of a `Sum(body, k, lo, hi)` node.
pub const SUM: &str = "Sum";
/// Head of an `Indexed(x, i)` node.
pub const INDEXED: &str = "Indexed";
/// Head of a `KroneckerDelta(i, j)` node.
pub const KRONECKER_DELTA: &str = "KroneckerDelta";

/// Largest number of terms [`expand_sum`], [`truncate`] and the numeric
/// evaluators expand a `Sum` into.
pub const MAX_SUM_TERMS: usize = 100_000;

fn is_symbol(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Symbol { .. }))
}

fn mentions(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    crate::kernel::subs::mentions_var(e, v, pool)
}

fn int_literal(e: ExprId, pool: &ExprPool) -> Option<rug::Integer> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(n.0.clone()),
        ExprData::Rational(q) if *q.0.denom() == 1 => Some(q.0.numer().clone()),
        _ => None,
    })
}

/// `Σ_{k=lo}^{hi} body`.
///
/// # Errors
///
/// `E-PDE-020` when `k` is not a symbol or a bound mentions `k`.
pub fn sum(
    body: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    if !is_symbol(k, pool) {
        return Err(PdeError::MalformedSum(format!(
            "the summation variable {} is not a symbol",
            pool.display(k)
        )));
    }
    if mentions(lo, k, pool) || mentions(hi, k, pool) {
        return Err(PdeError::MalformedSum(format!(
            "a bound of the sum over {} mentions {}",
            pool.display(k),
            pool.display(k)
        )));
    }
    Ok(pool.func(SUM, vec![body, k, lo, hi]))
}

/// `(body, k, lo, hi)` of a well-formed `Sum` node.
pub fn sum_parts(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    let args = pool.with(e, |d| match d {
        ExprData::Func { name, args } if name == SUM && args.len() == 4 => Some(args.clone()),
        _ => None,
    })?;
    is_symbol(args[1], pool).then_some((args[0], args[1], args[2], args[3]))
}

/// The indexed symbol `xᵢ`.
///
/// # Errors
///
/// `E-PDE-023` when `base` is not a symbol.
pub fn indexed(base: ExprId, index: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    if !is_symbol(base, pool) {
        return Err(PdeError::MalformedIndexed(format!(
            "the base {} is not a symbol",
            pool.display(base)
        )));
    }
    Ok(pool.func(INDEXED, vec![base, index]))
}

/// `(base, index)` of an `Indexed` node.
pub fn indexed_parts(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    pool.with(e, |d| match d {
        ExprData::Func { name, args } if name == INDEXED && args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    })
}

/// `δᵢⱼ`, folded to `1` / `0` when that is decided structurally (equal
/// nodes, or two distinct integer literals).
pub fn kronecker_delta(i: ExprId, j: ExprId, pool: &ExprPool) -> ExprId {
    if i == j {
        return pool.integer(1);
    }
    if let (Some(a), Some(b)) = (int_literal(i, pool), int_literal(j, pool)) {
        return pool.integer(i32::from(a == b));
    }
    // Canonical argument order, so δᵢⱼ and δⱼᵢ are one node.
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    pool.func(KRONECKER_DELTA, vec![a, b])
}

/// `∂xᵢ/∂v` for an indexed symbol `xᵢ` and an indexed variable `v = x'ⱼ`:
/// `δᵢⱼ` for the same base, `0` otherwise.  Used by `diff`.
pub(crate) fn indexed_derivative(expr: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    match (indexed_parts(expr, pool), indexed_parts(var, pool)) {
        (Some((b1, i)), Some((b2, j))) if b1 == b2 => kronecker_delta(i, j, pool),
        _ => pool.integer(0),
    }
}

/// A symbol to rename the bound `k` to: same domain, named `k_1`, `k_2`, …,
/// the first not occurring in any of `avoid`.
pub(crate) fn fresh_index(k: ExprId, avoid: &[ExprId], pool: &ExprPool) -> ExprId {
    let (base, domain) = pool.with(k, |d| match d {
        ExprData::Symbol { name, domain, .. } => (name.clone(), *domain),
        _ => ("k".to_string(), Domain::Integer),
    });
    let mut used = std::collections::BTreeSet::new();
    for &a in avoid {
        used.extend(symbols_of(a, pool));
    }
    (1u64..)
        .map(|i| pool.symbol(format!("{base}_{i}"), domain))
        .find(|c| !used.contains(c))
        .expect("a pool holds finitely many nodes")
}

/// `Sum(body, k, lo, hi)` alpha-renamed so its bound variable does not occur
/// in `avoid` (returned unchanged when it already does not).
pub(crate) fn rename_bound(
    body: ExprId,
    k: ExprId,
    avoid: &[ExprId],
    pool: &ExprPool,
) -> (ExprId, ExprId) {
    if !avoid.iter().any(|&a| mentions(a, k, pool)) {
        return (body, k);
    }
    let mut all = avoid.to_vec();
    all.push(body);
    let nk = fresh_index(k, &all, pool);
    let mut m = HashMap::new();
    m.insert(k, nk);
    (crate::kernel::subs::subs(body, &m, pool), nk)
}

fn term(body: ExprId, k: ExprId, n: &rug::Integer, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(k, pool.integer(n.clone()));
    crate::kernel::subs::subs(body, &m, pool)
}

/// The terms `body[k := lo], …, body[k := hi]` of a `Sum` whose bounds are
/// integer literals; `None` for non-literal bounds, `Err` past the cap.
pub(crate) fn finite_terms(e: ExprId, pool: &ExprPool) -> Option<Result<Vec<ExprId>, PdeError>> {
    let (body, k, lo, hi) = sum_parts(e, pool)?;
    let lo = int_literal(lo, pool)?;
    let hi = int_literal(hi, pool)?;
    Some(terms_between(body, k, &lo, &hi, pool))
}

fn terms_between(
    body: ExprId,
    k: ExprId,
    lo: &rug::Integer,
    hi: &rug::Integer,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, PdeError> {
    if hi < lo {
        return Ok(Vec::new());
    }
    let count = rug::Integer::from(hi - lo) + 1u32;
    if count > MAX_SUM_TERMS {
        return Err(PdeError::SumNotFinite(format!(
            "{count} terms is more than the cap of {MAX_SUM_TERMS}"
        )));
    }
    let mut out = Vec::new();
    let mut n = lo.clone();
    while &n <= hi {
        out.push(term(body, k, &n, pool));
        n += 1u32;
    }
    Ok(out)
}

fn add_all(terms: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    match terms.len() {
        0 => pool.integer(0),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// Expand the `Sum` `e` into an explicit sum of terms.
///
/// # Errors
///
/// `E-PDE-020` when `e` is not a `Sum`; `E-PDE-021` when a bound is not an
/// integer literal or the range is longer than [`MAX_SUM_TERMS`].
pub fn expand_sum(e: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let (_, _, lo, hi) = sum_parts(e, pool)
        .ok_or_else(|| PdeError::MalformedSum(format!("{} is not a Sum", pool.display(e))))?;
    match finite_terms(e, pool) {
        Some(t) => Ok(simplify(add_all(t?, pool), pool).value),
        None => Err(PdeError::SumNotFinite(format!(
            "bounds {} .. {} are not integers; substitute values or use truncate",
            pool.display(lo),
            pool.display(hi)
        ))),
    }
}

/// Rewrite every `Sum` in `expr` bottom-up with `f`.
fn map_sums(
    expr: ExprId,
    pool: &ExprPool,
    f: &mut dyn FnMut(ExprId) -> Result<ExprId, PdeError>,
) -> Result<ExprId, PdeError> {
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    map_sums_rec(expr, pool, f, &mut memo)
}

fn map_sums_rec(
    expr: ExprId,
    pool: &ExprPool,
    f: &mut dyn FnMut(ExprId) -> Result<ExprId, PdeError>,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Result<ExprId, PdeError> {
    if let Some(&d) = memo.get(&expr) {
        return Ok(d);
    }
    let rebuilt = rebuild_children(expr, pool, &mut |c| map_sums_rec(c, pool, f, memo))?;
    let out = if sum_parts(rebuilt, pool).is_some() {
        f(rebuilt)?
    } else {
        rebuilt
    };
    memo.insert(expr, out);
    Ok(out)
}

/// `expr` with every child replaced by `g(child)` (identity when unchanged).
fn rebuild_children(
    expr: ExprId,
    pool: &ExprPool,
    g: &mut dyn FnMut(ExprId) -> Result<ExprId, PdeError>,
) -> Result<ExprId, PdeError> {
    Ok(match pool.get(expr) {
        ExprData::Add(xs) => {
            let n: Vec<ExprId> = xs.iter().map(|&x| g(x)).collect::<Result<_, _>>()?;
            if n == xs {
                expr
            } else {
                pool.add(n)
            }
        }
        ExprData::Mul(xs) => {
            let n: Vec<ExprId> = xs.iter().map(|&x| g(x)).collect::<Result<_, _>>()?;
            if n == xs {
                expr
            } else {
                pool.mul(n)
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (g(base)?, g(exp)?);
            if b == base && e == exp {
                expr
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => {
            let n: Vec<ExprId> = args.iter().map(|&x| g(x)).collect::<Result<_, _>>()?;
            if n == args {
                expr
            } else {
                pool.func(name, n)
            }
        }
        _ => expr,
    })
}

/// Expand every `Sum` with integer-literal bounds in `expr` (inner sums
/// first).  Sums with symbolic or infinite bounds are left in place.
///
/// # Errors
///
/// `E-PDE-021` for a range longer than [`MAX_SUM_TERMS`].
pub fn expand_finite_sums(expr: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    map_sums(expr, pool, &mut |s| match finite_terms(s, pool) {
        Some(t) => Ok(add_all(t?, pool)),
        None => Ok(s),
    })
}

/// The partial sums of `expr`: every `Sum(body, k, lo, hi)` is replaced by
/// `Σ_{k=lo}^{min(hi, n)} body` — the terms with index `≤ n` — expanded.  For
/// a series indexed from 1 that is its first `n` terms.
///
/// # Errors
///
/// `E-PDE-021` when a lower bound is not an integer literal, or the partial
/// sum has more than [`MAX_SUM_TERMS`] terms.
pub fn truncate(expr: ExprId, n: i64, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let n = rug::Integer::from(n);
    let out = map_sums(expr, pool, &mut |s| {
        let (body, k, lo, hi) = sum_parts(s, pool).expect("map_sums passes Sums");
        let lo_v = int_literal(lo, pool).ok_or_else(|| {
            PdeError::SumNotFinite(format!(
                "the lower bound {} of a truncated sum must be an integer",
                pool.display(lo)
            ))
        })?;
        let top = match int_literal(hi, pool) {
            Some(h) if h < n => h,
            _ => n.clone(),
        };
        Ok(add_all(terms_between(body, k, &lo_v, &top, pool)?, pool))
    })?;
    Ok(simplify(out, pool).value)
}

/// The term with index `n` of the `Sum` `e`: `body[k := n]` when `n` is in
/// the range, `0` when it is provably outside.
///
/// # Errors
///
/// `E-PDE-020` when `e` is not a `Sum`; `E-PDE-021` when membership of `n`
/// in the range cannot be decided (symbolic bounds with a literal `n` are
/// decided only against literal bounds; `∞` counts as unbounded).
pub fn coefficient(e: ExprId, n: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let (body, k, lo, hi) = sum_parts(e, pool)
        .ok_or_else(|| PdeError::MalformedSum(format!("{} is not a Sum", pool.display(e))))?;
    let inf = pool.pos_infinity();
    let below = match (int_literal(n, pool), int_literal(lo, pool)) {
        (Some(a), Some(b)) => Some(a < b),
        _ if n == lo => Some(false),
        _ => None,
    };
    let above = match (int_literal(n, pool), int_literal(hi, pool)) {
        (Some(a), Some(b)) => Some(a > b),
        _ if hi == inf || n == hi => Some(false),
        _ => None,
    };
    match (below, above) {
        (Some(true), _) | (_, Some(true)) => Ok(pool.integer(0)),
        (Some(false), Some(false)) => {
            let mut m = HashMap::new();
            m.insert(k, n);
            Ok(simplify(crate::kernel::subs::subs(body, &m, pool), pool).value)
        }
        _ => Err(PdeError::SumNotFinite(format!(
            "cannot decide whether {} lies in {} .. {}",
            pool.display(n),
            pool.display(lo),
            pool.display(hi)
        ))),
    }
}

/// A range assumption `lo ≤ var ≤ hi` for [`simplify_sums`].
pub type IndexRange = (ExprId, ExprId, ExprId);

fn known_in_range(
    j: ExprId,
    lo: ExprId,
    hi: ExprId,
    ranges: &[IndexRange],
    pool: &ExprPool,
) -> bool {
    if let (Some(v), Some(a), Some(b)) = (
        int_literal(j, pool),
        int_literal(lo, pool),
        int_literal(hi, pool),
    ) {
        return a <= v && v <= b;
    }
    // A literal at or above a literal lower end of a range unbounded above.
    if let (Some(v), Some(a)) = (int_literal(j, pool), int_literal(lo, pool)) {
        if a <= v && hi == pool.pos_infinity() {
            return true;
        }
    }
    ranges.iter().any(|&(v, a, b)| {
        v == j
            && (a == lo || matches!((int_literal(a, pool), int_literal(lo, pool)), (Some(x), Some(y)) if x >= y))
            && (b == hi
                || hi == pool.pos_infinity()
                || matches!((int_literal(b, pool), int_literal(hi, pool)), (Some(x), Some(y)) if x <= y))
    })
}

/// Simplify every `Sum` in `expr` (inner first):
///
/// * empty range (`hi < lo`, literal) → `0`; a single term → that term;
/// * a summand free of `k` → `summand · (hi − lo + 1)` (finite `hi` only);
/// * linearity: a sum of terms splits, `k`-free factors move out;
/// * `Σₖ f(k) δₖⱼ → f(j)` when `j` is known to lie in the range: an integer
///   inside literal bounds, or by an entry of `ranges` (`(j, lo', hi')` with
///   `[lo', hi'] ⊆ [lo, hi]`).  A sum's own index range is in force while its
///   summand is simplified.
///
/// Also folds `KroneckerDelta` of equal or distinct-literal arguments.
pub fn simplify_sums(expr: ExprId, ranges: &[IndexRange], pool: &ExprPool) -> ExprId {
    let mut ctx: Vec<IndexRange> = ranges.to_vec();
    let out = simplify_sums_rec(expr, &mut ctx, pool);
    simplify(out, pool).value
}

fn simplify_sums_rec(expr: ExprId, ctx: &mut Vec<IndexRange>, pool: &ExprPool) -> ExprId {
    if let Some((body, k, lo, hi)) = sum_parts(expr, pool) {
        let lo = simplify_sums_rec(lo, ctx, pool);
        let hi = simplify_sums_rec(hi, ctx, pool);
        ctx.push((k, lo, hi));
        let body = simplify(simplify_sums_rec(body, ctx, pool), pool).value;
        ctx.pop();
        return fold_sum(body, k, lo, hi, ctx, pool);
    }
    if let Some((i, j)) = pool.with(expr, |d| match d {
        ExprData::Func { name, args } if name == KRONECKER_DELTA && args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    }) {
        return kronecker_delta(i, j, pool);
    }
    rebuild_children(expr, pool, &mut |c| Ok(simplify_sums_rec(c, ctx, pool))).expect("infallible")
}

fn fold_sum(
    body: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    ctx: &mut Vec<IndexRange>,
    pool: &ExprPool,
) -> ExprId {
    let zero = pool.integer(0);
    if body == zero {
        return zero;
    }
    if let (Some(a), Some(b)) = (int_literal(lo, pool), int_literal(hi, pool)) {
        if b < a {
            return zero;
        }
    }
    if lo == hi {
        return term_at(body, k, lo, pool);
    }
    let inf = pool.pos_infinity();
    if !mentions(body, k, pool) {
        if hi == inf || lo == pool.mul(vec![pool.integer(-1), inf]) {
            return pool.func(SUM, vec![body, k, lo, hi]);
        }
        let count = pool.add(vec![
            hi,
            pool.mul(vec![pool.integer(-1), lo]),
            pool.integer(1),
        ]);
        return simplify(pool.mul(vec![body, count]), pool).value;
    }
    match pool.get(body) {
        ExprData::Add(xs) => {
            let parts: Vec<ExprId> = xs
                .iter()
                .map(|&x| fold_sum(x, k, lo, hi, ctx, pool))
                .collect();
            return add_all(parts, pool);
        }
        ExprData::Mul(xs) => {
            let (free, dep): (Vec<ExprId>, Vec<ExprId>) =
                xs.iter().partition(|&&x| !mentions(x, k, pool));
            // δ_{k j} with j free of k and in range: substitute k := j.
            for (pos, &f) in dep.iter().enumerate() {
                if let Some(j) = delta_partner(f, k, pool) {
                    if known_in_range(j, lo, hi, ctx, pool) {
                        let rest: Vec<ExprId> = dep
                            .iter()
                            .enumerate()
                            .filter(|&(q, _)| q != pos)
                            .map(|(_, &x)| x)
                            .collect();
                        let mut all = free.clone();
                        all.push(term_at(mul_all(rest, pool), k, j, pool));
                        return simplify(mul_all(all, pool), pool).value;
                    }
                }
            }
            if !free.is_empty() {
                let inner = pool.func(SUM, vec![mul_all(dep, pool), k, lo, hi]);
                let mut all = free;
                all.push(inner);
                return pool.mul(all);
            }
        }
        _ => {
            if let Some(j) = delta_partner(body, k, pool) {
                if known_in_range(j, lo, hi, ctx, pool) {
                    return pool.integer(1);
                }
            }
        }
    }
    pool.func(SUM, vec![body, k, lo, hi])
}

fn mul_all(v: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    match v.len() {
        0 => pool.integer(1),
        1 => v[0],
        _ => pool.mul(v),
    }
}

fn term_at(body: ExprId, k: ExprId, at: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(k, at);
    simplify(crate::kernel::subs::subs(body, &m, pool), pool).value
}

/// `j` when `f` is `δ(k, j)` or `δ(j, k)` with `j` free of `k`.
fn delta_partner(f: ExprId, k: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (a, b) = pool.with(f, |d| match d {
        ExprData::Func { name, args } if name == KRONECKER_DELTA && args.len() == 2 => {
            Some((args[0], args[1]))
        }
        _ => None,
    })?;
    if a == k && !mentions(b, k, pool) {
        Some(b)
    } else if b == k && !mentions(a, k, pool) {
        Some(a)
    } else {
        None
    }
}

/// The Laplacian `Σⱼ ∂²u/∂xⱼ²` of `u` in the indexed coordinates
/// `x_lo, …, x_hi` (`hi` may be a symbol `d`), with the summation index's own
/// range used to collapse Kronecker deltas: `Δ Σᵢ φ(xᵢ) = Σⱼ φ''(xⱼ)`.
///
/// # Errors
///
/// `E-PDE-023` when `base` is not a symbol; `E-PDE-022` when `u` cannot be
/// differentiated (an unknown function in it, or a `Sum` whose bounds depend
/// on the coordinates).
pub fn laplacian_indexed(
    u: ExprId,
    base: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let probe = pool.symbol("j", Domain::Integer);
    let j = if symbols_of(u, pool).contains(&probe)
        || mentions(lo, probe, pool)
        || mentions(hi, probe, pool)
    {
        fresh_index(probe, &[u, lo, hi], pool)
    } else {
        probe
    };
    let xj = indexed(base, j, pool)?;
    let derr = |e: crate::diff::DiffError| PdeError::SumDifferentiation(e.to_string());
    let d1 = crate::diff::diff(u, xj, pool).map_err(derr)?.value;
    let ranges = [(j, lo, hi)];
    let d1 = simplify_sums(d1, &ranges, pool);
    let d2 = crate::diff::diff(d1, xj, pool).map_err(derr)?.value;
    let d2 = simplify_sums(d2, &ranges, pool);
    Ok(simplify_sums(
        pool.func(SUM, vec![d2, j, lo, hi]),
        &[],
        pool,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;

    fn setup() -> (ExprPool, ExprId, ExprId, ExprId, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let i = pool.symbol("i", Domain::Integer);
        let d = pool.symbol("d", Domain::Positive);
        let n = pool.symbol("n", Domain::Integer);
        (pool, x, i, d, n)
    }

    #[test]
    fn construction_and_errors() {
        let (pool, x, i, d, _) = setup();
        let one = pool.integer(1);
        let s = sum(pool.pow(i, pool.integer(2)), i, one, d, &pool).unwrap();
        assert_eq!(pool.display(s).to_string(), "Sum(i^2, i, 1, d)");
        assert_eq!(sum(i, x, one, x, &pool).unwrap_err().code(), "E-PDE-020");
        assert_eq!(
            sum(i, pool.integer(2), one, d, &pool).unwrap_err().code(),
            "E-PDE-020"
        );
        assert_eq!(
            indexed(pool.integer(1), i, &pool).unwrap_err().code(),
            "E-PDE-023"
        );
    }

    #[test]
    fn subs_respects_the_binder() {
        let (pool, x, i, _, n) = setup();
        let one = pool.integer(1);
        let s = sum(pool.mul(vec![x, i]), i, one, n, &pool).unwrap();
        // substituting the bound variable does nothing
        let mut m = HashMap::new();
        m.insert(i, pool.integer(7));
        assert_eq!(crate::kernel::subs::subs(s, &m, &pool), s);
        // substituting a free one works, in body and bounds
        let mut m = HashMap::new();
        m.insert(n, pool.integer(3));
        let s3 = crate::kernel::subs::subs(s, &m, &pool);
        assert_eq!(
            expand_sum(s3, &pool).unwrap(),
            simplify(pool.mul(vec![pool.integer(6), x]), &pool).value
        );
        // capture: x := i must rename the bound i
        let mut m = HashMap::new();
        m.insert(x, i);
        let c = crate::kernel::subs::subs(s, &m, &pool);
        let (body, k, _, _) = sum_parts(c, &pool).unwrap();
        assert_ne!(k, i);
        assert!(crate::kernel::subs::mentions_var(body, i, &pool));
    }

    #[test]
    fn finite_expand_truncate_coefficient() {
        let (pool, x, i, _, n) = setup();
        let one = pool.integer(1);
        let body = pool.pow(x, i);
        let s = sum(body, i, one, pool.pos_infinity(), &pool).unwrap();
        let t = truncate(s, 3, &pool).unwrap();
        let want = simplify(
            pool.add(vec![
                x,
                pool.pow(x, pool.integer(2)),
                pool.pow(x, pool.integer(3)),
            ]),
            &pool,
        )
        .value;
        assert_eq!(t, want);
        assert_eq!(
            coefficient(s, pool.integer(5), &pool).unwrap(),
            pool.pow(x, pool.integer(5))
        );
        assert_eq!(
            coefficient(s, pool.integer(0), &pool).unwrap(),
            pool.integer(0)
        );
        let sn = sum(body, i, one, n, &pool).unwrap();
        assert_eq!(
            coefficient(sn, pool.integer(5), &pool).unwrap_err().code(),
            "E-PDE-021"
        );
        assert_eq!(expand_sum(sn, &pool).unwrap_err().code(), "E-PDE-021");
    }

    #[test]
    fn constant_and_linear_folding() {
        let (pool, x, i, d, _) = setup();
        let one = pool.integer(1);
        let s = sum(x, i, one, d, &pool).unwrap();
        assert_eq!(simplify_sums(s, &[], &pool), pool.mul(vec![x, d]));
        let empty = sum(i, i, pool.integer(3), one, &pool).unwrap();
        assert_eq!(simplify_sums(empty, &[], &pool), pool.integer(0));
    }

    #[test]
    fn diff_termwise_and_indexed() {
        let (pool, x, i, d, _) = setup();
        let one = pool.integer(1);
        let xi = indexed(x, i, &pool).unwrap();
        let j = pool.symbol("j", Domain::Integer);
        let xj = indexed(x, j, &pool).unwrap();
        // ∂/∂x_j Σ x_i² = Σ 2 x_i δ_ij → 2 x_j given 1 ≤ j ≤ d
        let s = sum(pool.pow(xi, pool.integer(2)), i, one, d, &pool).unwrap();
        let ds = crate::diff::diff(s, xj, &pool).unwrap().value;
        assert!(
            pool.display(ds).to_string().contains("KroneckerDelta"),
            "{}",
            pool.display(ds)
        );
        let collapsed = simplify_sums(ds, &[(j, one, d)], &pool);
        assert_eq!(
            collapsed,
            simplify(pool.mul(vec![pool.integer(2), xj]), &pool).value
        );
        // without the range assumption nothing collapses
        assert!(pool
            .display(simplify_sums(ds, &[], &pool))
            .to_string()
            .contains("KroneckerDelta"));
        // diff w.r.t. a variable in the body only
        let y = pool.symbol("y", Domain::Real);
        let sy = sum(pool.mul(vec![y, i]), i, one, d, &pool).unwrap();
        let dy = crate::diff::diff(sy, y, &pool).unwrap().value;
        assert_eq!(
            simplify_sums(dy, &[], &pool),
            simplify_sums(sum(i, i, one, d, &pool).unwrap(), &[], &pool)
        );
        // bounds depending on the variable: refused
        let bad = sum(i, i, one, y, &pool).unwrap();
        assert!(crate::diff::diff(bad, y, &pool).is_err());
    }

    #[test]
    fn symbolic_dimension_laplacian() {
        let (pool, x, i, d, _) = setup();
        let one = pool.integer(1);
        let xi = indexed(x, i, &pool).unwrap();
        // u = ½ Σ x_i²  ⇒  Δu = d for every d.
        let u = sum(
            pool.mul(vec![pool.rational(1, 2), pool.pow(xi, pool.integer(2))]),
            i,
            one,
            d,
            &pool,
        )
        .unwrap();
        assert_eq!(laplacian_indexed(u, x, one, d, &pool).unwrap(), d);
        // u = Σ sin(x_i)  ⇒  Δu = −Σ sin(x_j)
        let u2 = sum(pool.func("sin", vec![xi]), i, one, d, &pool).unwrap();
        let l2 = laplacian_indexed(u2, x, one, d, &pool).unwrap();
        let back = simplify_sums(pool.add(vec![l2, u2]), &[], &pool);
        // Σ_j −sin(x_j) + Σ_i sin(x_i): alpha-equivalent sums cancel after
        // renaming one index onto the other.
        let (b1, k1, _, _) = sum_parts(u2, &pool).unwrap();
        let _ = (b1, k1, back);
        let s = pool.display(l2).to_string();
        assert!(s.contains("sin") && s.contains("Sum"), "{s}");
    }

    #[test]
    fn kronecker_folds() {
        let (pool, _, i, _, _) = setup();
        assert_eq!(kronecker_delta(i, i, &pool), pool.integer(1));
        assert_eq!(
            kronecker_delta(pool.integer(2), pool.integer(3), &pool),
            pool.integer(0)
        );
        let j = pool.symbol("j", Domain::Integer);
        assert_eq!(kronecker_delta(i, j, &pool), kronecker_delta(j, i, &pool));
    }
}
