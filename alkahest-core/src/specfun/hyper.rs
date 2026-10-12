//! The generalized hypergeometric function `ₚFq(a₁…aₚ; b₁…b_q; z)` (G11c).
//!
//! # Node
//!
//! `hyper(p, q, a₁, …, aₚ, b₁, …, b_q, z)` — the two leading integer literals
//! fix the split of the parameter list, so the node is an ordinary `Func` that
//! prints, parses and substitutes like any other. A node whose argument count
//! is not `p + q + 3` is malformed: every operation here declines it rather
//! than indexing into it.
//!
//! `ₚFq(a; b; z) = Σ_{k≥0} (a₁)ₖ⋯(aₚ)ₖ / ((b₁)ₖ⋯(b_q)ₖ) · zᵏ/k!` (DLMF 16.2.1),
//! the principal branch for `p = q + 1` (cut `[1, ∞)`).
//!
//! # What it knows
//!
//! * **Canonical form** ([`hyper`], and `simplify` on the node): a parameter
//!   appearing in both lists cancels, an upper parameter `0` or `z = 0` gives
//!   `1`.
//! * **`d/dz`**: `(Πaᵢ / Πbⱼ) · ₚFq(a + 1; b + 1; z)` (DLMF 16.3.1); a
//!   derivative in a parameter has no closed form and declines (`E-DIFF-001`).
//! * **Numbers**: `f64` (certified to 53 bits) and balls from Arb
//!   (`arb_hypgeom_0f1` / `_1f1` / `_2f1` / `_pfq`), real values only.
//! * **[`expand_hyper`]** (SymPy's `hyperexpand`): terminating series (an
//!   upper parameter `−N`) become polynomials, and the closed forms
//!
//!   | node | value |
//!   |---|---|
//!   | `₀F₀(;;z)` | `eᶻ` |
//!   | `₁F₀(a;;z)` | `(1 − z)^{−a}` |
//!   | `₂F₁(1, 1; 2; z)` | `−log(1 − z)/z` |
//!   | `₂F₁(½, 1; 3/2; z)` | `atanh(√z)/√z` |
//!   | `₂F₁(½, ½; 3/2; z)` | `asin(√z)/√z` |
//!   | `₂F₁(a, b; c; 1)`, `c − a − b > 0` a literal | `Γ(c)Γ(c−a−b)/(Γ(c−a)Γ(c−b))` (Gauss, DLMF 15.4.20) |
//!   | `₁F₁(1; 2; z)` | `(eᶻ − 1)/z` |
//!   | `₁F₁(½; 3/2; −w)` | `√π·erf(√w)/(2√w)` |
//!   | `₀F₁(; ½; z)`, `₀F₁(; 3/2; z)` | `cosh(2√z)`, `sinh(2√z)/(2√z)` |
//!   | `₀F₁(; b; −w)` / `₀F₁(; b; z)` | `Γ(b) w^{(1−b)/2} J_{b−1}(2√w)` / `Γ(b) z^{(1−b)/2} I_{b−1}(2√z)` |
//!
//!   Anything else is kept as the node.

use rug::{Integer, Rational};

use crate::ball::ArbBall;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::orthopoly::{int_literal, mk_add, mk_mul, rat_node, rational_literal};
use crate::primitive::Primitive;
use crate::simplify::simplify;

use super::SpecialFunctionError;

/// Largest degree `N` to which a terminating series (`aᵢ = −N`) is expanded
/// into a polynomial by [`expand_hyper`]; past it the call refuses with
/// `E-SPECFN-004`.
pub const MAX_TERMINATING_DEGREE: i64 = 200;

/// The node's name.
pub(crate) const HYPER: &str = "hyper";

/// `(a, b, z)` of a well-formed `hyper(p, q, a…, b…, z)` argument list.
pub(crate) fn parts(
    args: &[ExprId],
    pool: &ExprPool,
) -> Option<(Vec<ExprId>, Vec<ExprId>, ExprId)> {
    if args.len() < 3 {
        return None;
    }
    let p = int_literal(pool, args[0])?.to_usize()?;
    let q = int_literal(pool, args[1])?.to_usize()?;
    if args.len() != p.checked_add(q)?.checked_add(3)? {
        return None;
    }
    Some((
        args[2..2 + p].to_vec(),
        args[2 + p..2 + p + q].to_vec(),
        args[args.len() - 1],
    ))
}

/// The bare node `hyper(p, q, a…, b…, z)` (no canonicalisation).
pub(crate) fn raw_node(a: &[ExprId], b: &[ExprId], z: ExprId, pool: &ExprPool) -> ExprId {
    let mut args = Vec::with_capacity(a.len() + b.len() + 3);
    args.push(pool.integer(a.len() as i64));
    args.push(pool.integer(b.len() as i64));
    args.extend_from_slice(a);
    args.extend_from_slice(b);
    args.push(z);
    pool.func(HYPER, args)
}

fn is_lit(pool: &ExprPool, e: ExprId, v: i64) -> bool {
    rational_literal(pool, e).is_some_and(|q| q == v)
}

/// The `N ≥ 0` with `e = −N`, if `e` is a non-positive integer literal.
fn nonpositive_int(pool: &ExprPool, e: ExprId) -> Option<Integer> {
    let q = rational_literal(pool, e)?;
    (*q.denom() == 1 && q <= 0).then(|| Integer::from(-q.numer()))
}

/// The smallest `N` among the upper parameters equal to `−N`.
fn termination(a: &[ExprId], pool: &ExprPool) -> Option<Integer> {
    a.iter().filter_map(|&x| nonpositive_int(pool, x)).min()
}

/// Cancel parameters common to both lists; `None` when nothing changes.
fn cancelled(a: &[ExprId], b: &[ExprId], pool: &ExprPool) -> Option<(Vec<ExprId>, Vec<ExprId>)> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    let mut changed = false;
    let mut i = 0;
    while i < a.len() {
        if let Some(j) = b.iter().position(|&y| y == a[i]) {
            // Cancelling `a = b = −m` would drop the very term that makes the
            // series terminate before its own pole; keep such pairs.
            if nonpositive_int(pool, a[i]).is_none() {
                a.remove(i);
                b.remove(j);
                changed = true;
                continue;
            }
        }
        i += 1;
    }
    changed.then_some((a, b))
}

/// Whether a lower parameter `−m` makes the series undefined: it is unless an
/// upper parameter `−n` with `n < m` terminates the sum first.
fn lower_pole(a: &[ExprId], b: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let stop = termination(a, pool);
    b.iter().copied().find(|&y| match nonpositive_int(pool, y) {
        Some(m) => !stop.as_ref().is_some_and(|n| *n < m),
        None => false,
    })
}

/// Canonical form of a well-formed node, or `None` when already canonical.
pub(crate) fn canonical(args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let (a, b, z) = parts(args, pool)?;
    if is_lit(pool, z, 0) || a.iter().any(|&x| is_lit(pool, x, 0)) {
        if lower_pole(&a, &b, pool).is_none() {
            return Some(pool.integer(1));
        }
        return None;
    }
    let (a2, b2) = cancelled(&a, &b, pool)?;
    Some(raw_node(&a2, &b2, z, pool))
}

/// The generalized hypergeometric function `ₚFq(a; b; z)`, canonical
/// (common parameters cancel; `₀F₀`… stay nodes — see [`expand_hyper`] for the
/// closed forms).
///
/// # Errors
///
/// `E-SPECFN-006` for a lower parameter that is a non-positive integer `−m`
/// (unless an upper parameter `−n`, `n < m`, terminates the series first);
/// `E-BUDGET-*`.
pub fn hyper(
    a: &[ExprId],
    b: &[ExprId],
    z: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    if let Some(bad) = lower_pole(a, b, pool) {
        return Err(SpecialFunctionError::Pole {
            function: "hyper",
            at: format!(
                "the lower parameter {} (a non-positive integer divides by zero)",
                pool.display(bad)
            ),
        });
    }
    crate::budget::check()?;
    let node = raw_node(a, b, z, pool);
    let args = pool.with(node, |d| match d {
        ExprData::Func { args, .. } => args.clone(),
        _ => Vec::new(),
    });
    Ok(canonical(&args, pool).unwrap_or(node))
}

// ---------------------------------------------------------------------------
// expand_hyper
// ---------------------------------------------------------------------------

fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
    mk_mul(pool, vec![pool.integer(-1), e])
}

fn inv(pool: &ExprPool, e: ExprId) -> ExprId {
    pool.pow(e, pool.integer(-1))
}

fn sqrt(pool: &ExprPool, e: ExprId) -> ExprId {
    pool.func("sqrt", vec![e])
}

/// `w` when `z` is syntactically `−w`.
fn negated(pool: &ExprPool, z: ExprId) -> Option<ExprId> {
    if let Some(q) = rational_literal(pool, z) {
        return (q < 0).then(|| rat_node(pool, &Rational::from(-q)));
    }
    let f = pool.with(z, |d| match d {
        ExprData::Mul(f) => Some(f.clone()),
        _ => None,
    })?;
    let pos = f.iter().position(|&x| is_lit(pool, x, -1))?;
    let rest: Vec<ExprId> = f
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pos)
        .map(|(_, &x)| x)
        .collect();
    Some(mk_mul(pool, rest))
}

/// The rational literals of `v`, sorted, when every entry is one.
fn rational_params(v: &[ExprId], pool: &ExprPool) -> Option<Vec<Rational>> {
    let mut out: Vec<Rational> = v
        .iter()
        .map(|&x| rational_literal(pool, x))
        .collect::<Option<_>>()?;
    out.sort();
    Some(out)
}

fn r(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

/// `Σ_{k=0}^{N} Π(aᵢ)ₖ / Π(bⱼ)ₖ · zᵏ/k!` — exact rational coefficients when
/// every parameter is a literal, symbolic Pochhammer products otherwise.
fn terminating_sum(
    a: &[ExprId],
    b: &[ExprId],
    z: ExprId,
    n: i64,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    let ra = rational_params(a, pool);
    let rb = rational_params(b, pool);
    let mut terms = Vec::with_capacity(n as usize + 1);
    let zk = |k: i64| match k {
        0 => pool.integer(1),
        1 => z,
        _ => pool.pow(z, pool.integer(k)),
    };
    if let (Some(ra), Some(rb)) = (ra, rb) {
        let mut c = Rational::from(1);
        for k in 0..=n {
            if k > 0 {
                crate::budget::check()?;
                for x in &ra {
                    c *= x.clone() + Rational::from(k - 1);
                }
                for y in &rb {
                    let d = y.clone() + Rational::from(k - 1);
                    if d == 0 {
                        // Excluded by `lower_pole`; never divide by it.
                        return Err(SpecialFunctionError::Pole {
                            function: "hyper",
                            at: format!("the lower parameter {y}"),
                        });
                    }
                    c /= d;
                }
                c /= Rational::from(k);
            }
            if c != 0 {
                terms.push(mk_mul(pool, vec![rat_node(pool, &c), zk(k)]));
            }
        }
        return Ok(mk_add(pool, terms));
    }
    // Symbolic parameters: the k-th term as an explicit product.
    let mut factors: Vec<ExprId> = Vec::new();
    for k in 0..=n {
        if k > 0 {
            crate::budget::check()?;
            for &x in a {
                factors.push(pool.add(vec![x, pool.integer(k - 1)]));
            }
            for &y in b {
                factors.push(inv(pool, pool.add(vec![y, pool.integer(k - 1)])));
            }
            factors.push(pool.rational(1, k));
        }
        let mut f = factors.clone();
        f.push(zk(k));
        terms.push(mk_mul(pool, f));
    }
    Ok(simplify(mk_add(pool, terms), pool).value)
}

/// Gauss's sum `₂F₁(a, b; c; 1)` when `c − a − b` is a positive literal.
fn gauss(a: ExprId, b: ExprId, c: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let d = simplify(mk_add(pool, vec![c, neg(pool, a), neg(pool, b)]), pool).value;
    if rational_literal(pool, d).is_none_or(|q| q <= 0) {
        return None;
    }
    if nonpositive_int(pool, c).is_some() {
        return None;
    }
    let cma = simplify(mk_add(pool, vec![c, neg(pool, a)]), pool).value;
    let cmb = simplify(mk_add(pool, vec![c, neg(pool, b)]), pool).value;
    if nonpositive_int(pool, cma).is_some() || nonpositive_int(pool, cmb).is_some() {
        // 1/Γ at a pole is 0: the sum vanishes (Chu–Vandermonde).
        return Some(pool.integer(0));
    }
    let g = |e: ExprId| pool.func("gamma", vec![e]);
    Some(
        simplify(
            mk_mul(pool, vec![g(c), g(d), inv(pool, g(cma)), inv(pool, g(cmb))]),
            pool,
        )
        .value,
    )
}

/// One closed form from the table in the module docs, or `None`.
fn closed_form(
    a: &[ExprId],
    b: &[ExprId],
    z: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    let one = pool.integer(1);
    let one_minus_z = || mk_add(pool, vec![one, neg(pool, z)]);
    let ra = rational_params(a, pool);
    let rb = rational_params(b, pool);
    let half = r(1, 2);
    let three_half = r(3, 2);
    let pi = crate::eval::symbols::pi_symbol(pool);
    Ok(match (a.len(), b.len()) {
        (0, 0) => Some(pool.func("exp", vec![z])),
        (1, 0) => Some(pool.pow(one_minus_z(), neg(pool, a[0]))),
        (2, 1) => {
            if is_lit(pool, z, 1) {
                return Ok(gauss(a[0], a[1], b[0], pool));
            }
            let (Some(ra), Some(rb)) = (ra, rb) else {
                return Ok(None);
            };
            let sz = sqrt(pool, z);
            if ra == [r(1, 1), r(1, 1)] && rb == [r(2, 1)] {
                Some(mk_mul(
                    pool,
                    vec![
                        pool.integer(-1),
                        pool.func("log", vec![one_minus_z()]),
                        inv(pool, z),
                    ],
                ))
            } else if ra == [half.clone(), r(1, 1)] && rb == [three_half.clone()] {
                Some(mk_mul(
                    pool,
                    vec![pool.func("atanh", vec![sz]), inv(pool, sz)],
                ))
            } else if ra == [half.clone(), half.clone()] && rb == [three_half.clone()] {
                Some(mk_mul(
                    pool,
                    vec![pool.func("asin", vec![sz]), inv(pool, sz)],
                ))
            } else {
                None
            }
        }
        (1, 1) => {
            let (Some(ra), Some(rb)) = (ra, rb) else {
                return Ok(None);
            };
            if ra == [r(1, 1)] && rb == [r(2, 1)] {
                Some(mk_mul(
                    pool,
                    vec![
                        mk_add(pool, vec![pool.func("exp", vec![z]), pool.integer(-1)]),
                        inv(pool, z),
                    ],
                ))
            } else if ra == [half.clone()] && rb == [three_half.clone()] {
                let Some(w) = negated(pool, z) else {
                    return Ok(None);
                };
                let sw = sqrt(pool, w);
                Some(mk_mul(
                    pool,
                    vec![
                        pool.rational(1, 2),
                        sqrt(pool, pi),
                        pool.func("erf", vec![sw]),
                        inv(pool, sw),
                    ],
                ))
            } else {
                None
            }
        }
        (0, 1) => {
            let bq = rational_literal(pool, b[0]);
            let two_sqrt = |e: ExprId| mk_mul(pool, vec![pool.integer(2), sqrt(pool, e)]);
            if bq.as_ref() == Some(&half) {
                Some(pool.func("cosh", vec![two_sqrt(z)]))
            } else if bq.as_ref() == Some(&three_half) {
                let t = two_sqrt(z);
                Some(mk_mul(pool, vec![pool.func("sinh", vec![t]), inv(pool, t)]))
            } else {
                // Γ(b)·w^{(1−b)/2}·J_{b−1}(2√w) for z = −w, the I form else.
                let (kind, w) = match negated(pool, z) {
                    Some(w) => (super::BesselKind::J, w),
                    None => (super::BesselKind::I, z),
                };
                let bm1 = simplify(mk_add(pool, vec![b[0], pool.integer(-1)]), pool).value;
                let expo = simplify(
                    mk_mul(
                        pool,
                        vec![
                            pool.rational(1, 2),
                            mk_add(pool, vec![one, neg(pool, b[0])]),
                        ],
                    ),
                    pool,
                )
                .value;
                let bes = super::bessel(kind, bm1, two_sqrt(w), true, pool)?;
                Some(mk_mul(
                    pool,
                    vec![pool.func("gamma", vec![b[0]]), pool.pow(w, expo), bes],
                ))
            }
        }
        _ => None,
    })
}

/// `ₚFq(a; b; z)` in closed form, or `None` to keep the node.
fn reduce(
    a: &[ExprId],
    b: &[ExprId],
    z: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, SpecialFunctionError> {
    let (a, b) = cancelled(a, b, pool).unwrap_or_else(|| (a.to_vec(), b.to_vec()));
    if lower_pole(&a, &b, pool).is_some() {
        return Ok(None);
    }
    if is_lit(pool, z, 0) {
        return Ok(Some(pool.integer(1)));
    }
    if let Some(n) = termination(&a, pool) {
        let Some(n) = n.to_i64().filter(|&n| n <= MAX_TERMINATING_DEGREE) else {
            return Err(SpecialFunctionError::TooLarge {
                what: "terminating hypergeometric degree".into(),
                value: n.to_i64().unwrap_or(i64::MAX),
                max: MAX_TERMINATING_DEGREE,
            });
        };
        return terminating_sum(&a, &b, z, n, pool).map(Some);
    }
    closed_form(&a, &b, z, pool)
}

/// Rewrite every `hyper` node in `expr` into a polynomial or a closed form in
/// elementary and special functions where one is known (the table in the
/// module docs; SymPy's `hyperexpand`). Nodes without one are kept,
/// canonicalised.
///
/// # Errors
///
/// `E-SPECFN-004` for a terminating series of degree past
/// [`MAX_TERMINATING_DEGREE`]; `E-BUDGET-*`.
pub fn expand_hyper(expr: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    let mut memo = std::collections::HashMap::new();
    expand_rec(expr, pool, &mut memo)
}

fn expand_rec(
    expr: ExprId,
    pool: &ExprPool,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> Result<ExprId, SpecialFunctionError> {
    if let Some(&r) = memo.get(&expr) {
        return Ok(r);
    }
    crate::budget::check()?;
    let rec = |v: &[ExprId], memo: &mut std::collections::HashMap<ExprId, ExprId>| {
        v.iter()
            .map(|&a| expand_rec(a, pool, memo))
            .collect::<Result<Vec<_>, _>>()
    };
    let out = match pool.get(expr) {
        ExprData::Add(args) => {
            let a = rec(&args, memo)?;
            if a == args {
                expr
            } else {
                pool.add(a)
            }
        }
        ExprData::Mul(args) => {
            let a = rec(&args, memo)?;
            if a == args {
                expr
            } else {
                pool.mul(a)
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (expand_rec(base, pool, memo)?, expand_rec(exp, pool, memo)?);
            if (b, e) == (base, exp) {
                expr
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => {
            let a = rec(&args, memo)?;
            let node = if a == args {
                expr
            } else {
                pool.func(name.clone(), a.clone())
            };
            if name == HYPER {
                match parts(&a, pool) {
                    Some((pa, pb, z)) => match reduce(&pa, &pb, z, pool)? {
                        Some(e) => e,
                        None => canonical(&a, pool).unwrap_or(node),
                    },
                    None => node,
                }
            } else {
                node
            }
        }
        _ => expr,
    };
    memo.insert(expr, out);
    Ok(out)
}

// ---------------------------------------------------------------------------
// The primitive
// ---------------------------------------------------------------------------

/// Registry entry for `hyper(p, q, a…, b…, z)` (module docs).
pub struct HyperPrimitive;

impl Primitive for HyperPrimitive {
    fn name(&self) -> &'static str {
        HYPER
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        let show = |v: &[ExprId]| {
            v.iter()
                .map(|&x| pool.display(x).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match parts(args, pool) {
            Some((a, b, z)) => format!(
                "{}F{}({}; {}; {})",
                a.len(),
                b.len(),
                show(&a),
                show(&b),
                pool.display(z)
            ),
            None => format!("hyper({})", show(args)),
        }
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        canonical(args, pool)
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let (a, b, z) = parts(args, pool)?;
        if a.iter()
            .chain(b.iter())
            .any(|&c| crate::kernel::subs::mentions_var(c, wrt, pool))
        {
            // ∂/∂aᵢ, ∂/∂bⱼ have no closed form: decline rather than drop.
            return None;
        }
        let dz = crate::diff::diff(z, wrt, pool).ok()?.value;
        if is_lit(pool, dz, 0) {
            return Some(pool.integer(0));
        }
        let shift = |v: &[ExprId]| -> Vec<ExprId> {
            v.iter()
                .map(|&x| match rational_literal(pool, x) {
                    Some(q) => rat_node(pool, &(q + Rational::from(1))),
                    None => pool.add(vec![x, pool.integer(1)]),
                })
                .collect()
        };
        let mut f: Vec<ExprId> = a.clone();
        f.extend(b.iter().map(|&y| inv(pool, y)));
        f.push(raw_node(&shift(&a), &shift(&b), z, pool));
        f.push(dz);
        Some(mk_mul(pool, f))
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        let (a, b, z) = numeric_parts(args)?;
        #[cfg(flint_arb)]
        {
            arb_hyp::f64_value(a, b, z)
        }
        #[cfg(not(flint_arb))]
        {
            let _ = (a, b, z);
            None
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        if args.len() < 3 || !args[0].rad.is_zero() || !args[1].rad.is_zero() {
            return None;
        }
        let p = exact_count(args[0].mid.to_f64())?;
        let q = exact_count(args[1].mid.to_f64())?;
        if args.len() != p + q + 3 {
            return None;
        }
        #[cfg(flint_arb)]
        {
            arb_hyp::ball_value(
                &args[2..2 + p],
                &args[2 + p..2 + p + q],
                &args[args.len() - 1],
            )
        }
        #[cfg(not(flint_arb))]
        {
            None
        }
    }
}

fn exact_count(v: f64) -> Option<usize> {
    (v.is_finite() && v >= 0.0 && v == v.trunc() && v < 1e6).then_some(v as usize)
}

/// `(a, b, z)` of an `f64` argument list.
fn numeric_parts(args: &[f64]) -> Option<(&[f64], &[f64], f64)> {
    if args.len() < 3 || args.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let p = exact_count(args[0])?;
    let q = exact_count(args[1])?;
    if args.len() != p + q + 3 {
        return None;
    }
    Some((
        &args[2..2 + p],
        &args[2 + p..2 + p + q],
        args[args.len() - 1],
    ))
}

#[cfg(flint_arb)]
mod arb_hyp {
    use super::super::numeric::arb_impl::{available, certified_f64, from_arb, to_arb};
    use crate::ball::ArbBall;
    use crate::flint::acb::Arb;
    use crate::flint::arb as ffi;

    /// `ₚFq(a; b; z)` over Arb balls at `prec` bits; `None` when Arb cannot
    /// give a finite real enclosure (a pole, past the cut, divergence).
    fn call(a: &[Arb], b: &[Arb], z: &Arb, prec: i64) -> Option<Arb> {
        let mut out = Arb::new();
        unsafe {
            match (a.len(), b.len()) {
                (0, 1) => {
                    ffi::arb_hypgeom_0f1(out.as_mut_ptr(), b[0].as_ptr(), z.as_ptr(), 0, prec)
                }
                (1, 1) => ffi::arb_hypgeom_1f1(
                    out.as_mut_ptr(),
                    a[0].as_ptr(),
                    b[0].as_ptr(),
                    z.as_ptr(),
                    0,
                    prec,
                ),
                (2, 1) => ffi::arb_hypgeom_2f1(
                    out.as_mut_ptr(),
                    a[0].as_ptr(),
                    a[1].as_ptr(),
                    b[0].as_ptr(),
                    z.as_ptr(),
                    0,
                    prec,
                ),
                _ => {
                    // `arb_srcptr` arrays: shallow (`Copy`) copies of the
                    // structs, read only by the callee; the owning `Arb`s
                    // free the limbs, the vectors only their own buffers.
                    let av: Vec<ffi::ArbStruct> = a.iter().map(|x| *x.as_ptr()).collect();
                    let bv: Vec<ffi::ArbStruct> = b.iter().map(|x| *x.as_ptr()).collect();
                    ffi::arb_hypgeom_pfq(
                        out.as_mut_ptr(),
                        if av.is_empty() {
                            std::ptr::null()
                        } else {
                            av.as_ptr()
                        },
                        a.len() as i64,
                        if bv.is_empty() {
                            std::ptr::null()
                        } else {
                            bv.as_ptr()
                        },
                        b.len() as i64,
                        z.as_ptr(),
                        0,
                        prec,
                    );
                }
            }
        }
        out.is_finite().then_some(out)
    }

    pub(super) fn f64_value(a: &[f64], b: &[f64], z: f64) -> Option<f64> {
        if !available() {
            return None;
        }
        let ab: Vec<Arb> = a.iter().map(|&v| Arb::from_f64(v)).collect();
        let bb: Vec<Arb> = b.iter().map(|&v| Arb::from_f64(v)).collect();
        let zb = Arb::from_f64(z);
        for prec in [128_i64, 256, 512] {
            if let Some(v) = call(&ab, &bb, &zb, prec).as_ref().and_then(certified_f64) {
                return Some(v);
            }
        }
        None
    }

    pub(super) fn ball_value(a: &[ArbBall], b: &[ArbBall], z: &ArbBall) -> Option<ArbBall> {
        if !available() {
            return None;
        }
        let prec = a
            .iter()
            .chain(b.iter())
            .chain(std::iter::once(z))
            .map(|x| x.prec)
            .max()
            .unwrap_or(128);
        let ab: Vec<Arb> = a.iter().map(to_arb).collect::<Option<_>>()?;
        let bb: Vec<Arb> = b.iter().map(to_arb).collect::<Option<_>>()?;
        let zb = to_arb(z)?;
        let v = call(&ab, &bb, &zb, i64::from(prec) + 16)?;
        from_arb(&v, prec)
    }
}
