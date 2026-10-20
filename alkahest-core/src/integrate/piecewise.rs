//! Integration of `Piecewise` integrands (and of `abs`, `sign`, `heaviside`,
//! `min`, `max`, which lower to `Piecewise` first).
//!
//! # Method
//!
//! 1. [`rewrite_piecewise`] lowers `abs`/`sign`/`heaviside`/`min`/`max`.
//! 2. Every comparison in a `Piecewise` condition must be a polynomial
//!    inequality in the integration variable with rational coefficients (no
//!    free parameters).  Each is factored over ℚ; every irreducible factor of
//!    degree 1 or 2 contributes its real roots as *breakpoints* (exact
//!    rationals or quadratic surds), isolated in exact rational arithmetic.
//!    A factor of degree ≥ 3 is a coded refusal.
//! 3. Between two consecutive breakpoints every condition has a constant
//!    truth value, decided exactly at a rational sample point, so the
//!    integrand collapses to one smooth expression per interval, which is
//!    handed to the ordinary [`integrate`] / [`integrate_definite`].
//! 4. **Indefinite**: the pieces are glued into a `Piecewise` antiderivative
//!    `Piecewise((x < b₁, F₀), (x < b₂, F₁ + c₁), …; Fₖ + cₖ)` whose constants
//!    `cᵢ` make it continuous at every breakpoint (the first piece carries
//!    none).  A piece whose antiderivative is not finite at a breakpoint is a
//!    coded refusal.
//! 5. **Definite**: the interval `[a, b]` (rational, numeric constant or
//!    `±∞`) is cut at the breakpoints and each piece is integrated with
//!    [`integrate_definite`], so improper pieces get its pole/limit analysis.

use super::engine::{integrate, integrate_definite, IntegrationError};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::eval::{eval_exact_rational, eval_f64};
use crate::kernel::expr::PredicateKind;
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{ExprData, ExprId, ExprPool, IdMap, IdSet};
use crate::poly::real_roots::real_roots;
use crate::poly::UniPoly;
use crate::simplify::engine::simplify;
use crate::simplify::rewrite_piecewise;
use rug::{Integer, Rational};
use std::cmp::Ordering;
use std::collections::HashMap;

type IntResult = Result<DerivedExpr<ExprId>, IntegrationError>;

fn refuse<T>(msg: impl Into<String>) -> Result<T, IntegrationError> {
    Err(IntegrationError::NotImplemented(msg.into()))
}

fn has_piecewise(expr: ExprId, pool: &ExprPool) -> bool {
    let mut seen = IdSet::default();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        match pool.get(e) {
            ExprData::Piecewise { .. } => return true,
            ExprData::Add(a) | ExprData::Mul(a) => stack.extend(a),
            ExprData::Pow { base, exp } => {
                stack.push(base);
                stack.push(exp);
            }
            ExprData::Func { args, .. } => stack.extend(args),
            _ => {}
        }
    }
    false
}

fn is_nonsmooth_head(name: &str) -> bool {
    matches!(name, "abs" | "sign" | "heaviside" | "min" | "max")
}

/// Does `expr` contain a `Piecewise`, `abs`, `sign`, `heaviside`, `min` or
/// `max` node that depends on `var`?
fn needs_piecewise(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let mut seen = IdSet::default();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        match pool.get(e) {
            ExprData::Piecewise { branches, default } => {
                if mentions_var(e, var, pool) {
                    return true;
                }
                for (c, v) in branches {
                    stack.push(c);
                    stack.push(v);
                }
                stack.push(default);
            }
            ExprData::Func { name, args } => {
                if is_nonsmooth_head(&name) && args.iter().any(|&a| mentions_var(a, var, pool)) {
                    return true;
                }
                stack.extend(args);
            }
            ExprData::Add(a) | ExprData::Mul(a) => stack.extend(a),
            ExprData::Pow { base, exp } => {
                stack.push(base);
                stack.push(exp);
            }
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Breakpoints
// ---------------------------------------------------------------------------

/// A real root of an irreducible integer polynomial, isolated by the closed
/// rational interval `[lo, hi]` (`lo == hi` when exactly rational).
struct Breakpoint {
    expr: ExprId,
    lo: Rational,
    hi: Rational,
    coeffs: Vec<Integer>,
}

fn horner(coeffs: &[Integer], x: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in coeffs.iter().rev() {
        acc = acc * x.clone() + Rational::from(c.clone());
    }
    acc
}

impl Breakpoint {
    fn is_exact(&self) -> bool {
        self.lo == self.hi
    }

    /// One exact bisection step.
    fn refine(&mut self) {
        if self.is_exact() {
            return;
        }
        let plo = horner(&self.coeffs, &self.lo);
        if plo == 0 {
            self.hi = self.lo.clone();
            return;
        }
        let phi = horner(&self.coeffs, &self.hi);
        if phi == 0 {
            self.lo = self.hi.clone();
            return;
        }
        let mid = Rational::from(&self.lo + &self.hi) / 2u32;
        let pm = horner(&self.coeffs, &mid);
        if pm == 0 {
            self.lo = mid.clone();
            self.hi = mid;
        } else if (plo.cmp0() == Ordering::Less) == (pm.cmp0() == Ordering::Less) {
            self.lo = mid;
        } else {
            self.hi = mid;
        }
    }
}

fn rat_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])])
}

/// The comparison atoms `lhs ⋚ rhs` of one `Piecewise` condition.
fn collect_atoms(
    cond: ExprId,
    out: &mut Vec<(ExprId, ExprId)>,
    pool: &ExprPool,
) -> Result<(), IntegrationError> {
    match pool.get(cond) {
        ExprData::Predicate { kind, args } => match kind {
            PredicateKind::Lt
            | PredicateKind::Le
            | PredicateKind::Gt
            | PredicateKind::Ge
            | PredicateKind::Eq
            | PredicateKind::Ne
                if args.len() == 2 =>
            {
                out.push((args[0], args[1]));
                Ok(())
            }
            PredicateKind::And | PredicateKind::Or | PredicateKind::Not => {
                for a in args {
                    collect_atoms(a, out, pool)?;
                }
                Ok(())
            }
            PredicateKind::True | PredicateKind::False => Ok(()),
            _ => refuse("integrate: a Piecewise condition of this kind is not supported"),
        },
        _ => refuse("integrate: a Piecewise condition is not a comparison"),
    }
}

/// Every comparison in every `Piecewise` of `expr` that depends on `var`.
fn all_atoms(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, ExprId)>, IntegrationError> {
    let mut atoms = Vec::new();
    let mut seen = IdSet::default();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        match pool.get(e) {
            ExprData::Piecewise { branches, default } => {
                for (c, v) in branches {
                    let before = atoms.len();
                    collect_atoms(c, &mut atoms, pool)?;
                    for &(l, r) in &atoms[before..] {
                        stack.push(l);
                        stack.push(r);
                    }
                    stack.push(v);
                }
                stack.push(default);
            }
            ExprData::Func { args, .. } => stack.extend(args),
            ExprData::Add(a) | ExprData::Mul(a) => stack.extend(a),
            ExprData::Pow { base, exp } => {
                stack.push(base);
                stack.push(exp);
            }
            _ => {}
        }
    }
    let _ = var;
    Ok(atoms)
}

fn free_symbols_other_than(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let mut seen = IdSet::default();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        match pool.get(n) {
            ExprData::Symbol { .. } if n != var => return true,
            ExprData::Add(a) | ExprData::Mul(a) => stack.extend(a),
            ExprData::Pow { base, exp } => {
                stack.push(base);
                stack.push(exp);
            }
            ExprData::Func { args, .. } => stack.extend(args),
            _ => {}
        }
    }
    false
}

/// Breakpoints of all `atoms`, sorted and pairwise separated.
fn breakpoints(
    atoms: &[(ExprId, ExprId)],
    var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<Breakpoint>, IntegrationError> {
    let mut seen_factors: Vec<Vec<Integer>> = Vec::new();
    let mut bps: Vec<Breakpoint> = Vec::new();
    for &(lhs, rhs) in atoms {
        let diff = sub(lhs, rhs, pool);
        if has_piecewise(diff, pool) {
            // A condition on a nested `Piecewise` (`abs(abs(x) - 2)`): its roots
            // are only polynomial once the inner pieces are resolved, which the
            // recursion on each interval does.
            continue;
        }
        if free_symbols_other_than(diff, var, pool) {
            return refuse(
                "integrate: a Piecewise condition involves a free parameter; the interval \
                 cannot be split at a parametric breakpoint",
            );
        }
        if !mentions_var(diff, var, pool) {
            continue; // constant condition: decided when resolving.
        }
        let poly = UniPoly::from_symbolic_clear_denoms(diff, var, pool).map_err(|_| {
            IntegrationError::NotImplemented(format!(
                "integrate: the Piecewise condition on {} is not a polynomial inequality in {} \
                 with rational coefficients, so its breakpoints cannot be solved exactly",
                pool.display(diff),
                pool.display(var)
            ))
        })?;
        if poly.is_zero() {
            continue;
        }
        let fac = poly
            .factor_z()
            .map_err(|_| IntegrationError::NotImplemented("integrate: factoring failed".into()))?;
        for (f, _) in fac.factors {
            let coeffs = f.coefficients();
            let deg = coeffs.len() - 1;
            if deg == 0 || seen_factors.contains(&coeffs) {
                continue;
            }
            seen_factors.push(coeffs.clone());
            match deg {
                1 => {
                    let r = Rational::from((-coeffs[0].clone(), coeffs[1].clone()));
                    bps.push(Breakpoint {
                        expr: rat_expr(&r, pool),
                        lo: r.clone(),
                        hi: r,
                        coeffs,
                    });
                }
                2 => {
                    let roots = real_roots(&f).map_err(|_| {
                        IntegrationError::NotImplemented("integrate: root isolation failed".into())
                    })?;
                    let (a, b, c) = (&coeffs[2], &coeffs[1], &coeffs[0]);
                    let disc = Integer::from(b * b) - Integer::from(4) * Integer::from(a * c);
                    // `a > 0`: the smaller root first.
                    for (k, iv) in roots.iter().enumerate() {
                        let sign = if k == 0 { -1 } else { 1 };
                        let surd = pool.func("sqrt", vec![pool.integer(disc.clone())]);
                        let num = pool.add(vec![
                            pool.integer(-b.clone()),
                            pool.mul(vec![pool.integer(sign), surd]),
                        ]);
                        let expr =
                            pool.mul(vec![pool.rational(1, Integer::from(2) * a.clone()), num]);
                        bps.push(Breakpoint {
                            expr,
                            lo: iv.lo.clone(),
                            hi: iv.hi.clone(),
                            coeffs: coeffs.clone(),
                        });
                    }
                }
                d => {
                    return refuse(format!(
                        "integrate: a Piecewise condition has an irreducible factor of degree {d} \
                         over ℚ; breakpoints of degree ≥ 3 are not supported"
                    ));
                }
            }
        }
    }
    // Sort and separate.
    for _ in 0..4096 {
        bps.sort_by(|a, b| a.lo.cmp(&b.lo).then(a.hi.cmp(&b.hi)));
        let mut clean = true;
        for i in 0..bps.len().saturating_sub(1) {
            if bps[i].hi >= bps[i + 1].lo {
                clean = false;
                bps[i].refine();
                bps[i + 1].refine();
            }
        }
        if clean {
            return Ok(bps);
        }
    }
    refuse("integrate: could not separate the breakpoints of the Piecewise conditions")
}

// ---------------------------------------------------------------------------
// Resolving a Piecewise at a sample point
// ---------------------------------------------------------------------------

/// Truth of `cond` at the sample, or `None` when it compares a nested
/// `Piecewise` (decided later, on the sub-interval).
fn decide(
    cond: ExprId,
    bind: &HashMap<ExprId, Rational>,
    pool: &ExprPool,
) -> Result<Option<bool>, IntegrationError> {
    let fail = || IntegrationError::NotImplemented("integrate: cannot decide a condition".into());
    match pool.get(cond) {
        ExprData::Predicate { kind, args } => match kind {
            PredicateKind::True => Ok(Some(true)),
            PredicateKind::False => Ok(Some(false)),
            PredicateKind::Not => Ok(decide(args[0], bind, pool)?.map(|b| !b)),
            PredicateKind::And => {
                let mut all = Some(true);
                for a in args {
                    match decide(a, bind, pool)? {
                        Some(false) => return Ok(Some(false)),
                        None => all = None,
                        Some(true) => {}
                    }
                }
                Ok(all)
            }
            PredicateKind::Or => {
                let mut any = Some(false);
                for a in args {
                    match decide(a, bind, pool)? {
                        Some(true) => return Ok(Some(true)),
                        None => any = None,
                        Some(false) => {}
                    }
                }
                Ok(any)
            }
            k => {
                if args.iter().any(|&a| has_piecewise(a, pool)) {
                    return Ok(None);
                }
                let l = eval_exact_rational(args[0], pool, bind).map_err(|_| fail())?;
                let r = eval_exact_rational(args[1], pool, bind).map_err(|_| fail())?;
                let o = l.cmp(&r);
                Ok(Some(match k {
                    PredicateKind::Lt => o == Ordering::Less,
                    PredicateKind::Le => o != Ordering::Greater,
                    PredicateKind::Gt => o == Ordering::Greater,
                    PredicateKind::Ge => o != Ordering::Less,
                    PredicateKind::Eq => o == Ordering::Equal,
                    PredicateKind::Ne => o != Ordering::Equal,
                    _ => return Err(fail()),
                }))
            }
        },
        _ => Err(fail()),
    }
}

/// `expr` with every `Piecewise` replaced by the branch selected at `sample`.
fn resolve(
    e: ExprId,
    bind: &HashMap<ExprId, Rational>,
    pool: &ExprPool,
    memo: &mut IdMap<ExprId>,
) -> Result<ExprId, IntegrationError> {
    if let Some(&r) = memo.get(&e) {
        return Ok(r);
    }
    let r = match pool.get(e) {
        ExprData::Piecewise { branches, default } => {
            let mut chosen = Some(default);
            for &(c, v) in &branches {
                match decide(c, bind, pool)? {
                    Some(true) => {
                        chosen = Some(v);
                        break;
                    }
                    Some(false) => {}
                    None => {
                        chosen = None;
                        break;
                    }
                }
            }
            match chosen {
                Some(b) => resolve(b, bind, pool, memo)?,
                // Undecidable here (nested condition): keep the node, resolve
                // inside its values, and let the recursion on the interval
                // finish the job.
                None => {
                    let nb = branches
                        .iter()
                        .map(|&(c, v)| {
                            Ok((resolve(c, bind, pool, memo)?, resolve(v, bind, pool, memo)?))
                        })
                        .collect::<Result<Vec<_>, IntegrationError>>()?;
                    let nd = resolve(default, bind, pool, memo)?;
                    pool.piecewise(nb, nd)
                }
            }
        }
        ExprData::Predicate { kind, args } => {
            let v = args
                .iter()
                .map(|&x| resolve(x, bind, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            pool.predicate(kind, v)
        }
        ExprData::Add(a) => {
            let v = a
                .iter()
                .map(|&x| resolve(x, bind, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            pool.add(v)
        }
        ExprData::Mul(a) => {
            let v = a
                .iter()
                .map(|&x| resolve(x, bind, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            pool.mul(v)
        }
        ExprData::Pow { base, exp } => pool.pow(
            resolve(base, bind, pool, memo)?,
            resolve(exp, bind, pool, memo)?,
        ),
        ExprData::Func { name, args } => {
            let v = args
                .iter()
                .map(|&x| resolve(x, bind, pool, memo))
                .collect::<Result<Vec<_>, _>>()?;
            pool.func(name, v)
        }
        _ => e,
    };
    memo.insert(e, r);
    Ok(r)
}

/// The integrand's smooth form on each interval between breakpoints.
struct Split {
    bps: Vec<Breakpoint>,
    /// `pieces[i]` is the integrand on `(bps[i-1], bps[i])`.
    pieces: Vec<ExprId>,
}

fn split(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<Split, IntegrationError> {
    let lowered = rewrite_piecewise(expr, pool);
    let atoms = all_atoms(lowered, var, pool)?;
    let bps = breakpoints(&atoms, var, pool)?;
    let k = bps.len();
    let mut samples: Vec<Rational> = Vec::with_capacity(k + 1);
    if k == 0 {
        samples.push(Rational::from(0));
    } else {
        let first = Rational::from(bps[0].lo.clone().floor_ref()) - 1;
        samples.push(first);
        for i in 0..k - 1 {
            samples.push(Rational::from(&bps[i].hi + &bps[i + 1].lo) / 2u32);
        }
        let last = Rational::from(bps[k - 1].hi.clone().ceil_ref()) + 1;
        samples.push(last);
    }
    let mut pieces = Vec::with_capacity(samples.len());
    for s in &samples {
        let mut bind = HashMap::new();
        bind.insert(var, s.clone());
        let mut memo = IdMap::default();
        pieces.push(resolve(lowered, &bind, pool, &mut memo)?);
    }
    // No progress (a nested condition that never becomes polynomial): refuse
    // rather than recurse forever.
    if pieces.contains(&lowered) {
        return refuse(
            "integrate: a Piecewise condition compares a nested Piecewise that does not reduce              to a polynomial inequality, so its breakpoints cannot be solved exactly",
        );
    }
    Ok(Split { bps, pieces })
}

// ---------------------------------------------------------------------------
// Indefinite
// ---------------------------------------------------------------------------

/// `Some(result)` when `expr` has a `Piecewise`/`abs`/`min`/`max`/… in `var`.
pub(crate) fn try_integrate(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<IntResult> {
    if !needs_piecewise(expr, var, pool) {
        return None;
    }
    Some(integrate_piecewise(expr, var, pool))
}

fn value_at(
    f: ExprId,
    var: ExprId,
    at: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, IntegrationError> {
    let mut m = HashMap::new();
    m.insert(var, at);
    let v = simplify(subs(f, &m, pool), pool).value;
    // Finite there?  Evaluate with every other symbol at 1.0.
    let mut env: HashMap<ExprId, f64> = HashMap::new();
    let mut seen = IdSet::default();
    let mut stack = vec![v];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        match pool.get(n) {
            ExprData::Symbol { .. } => {
                env.insert(n, 1.0);
            }
            ExprData::Add(a) | ExprData::Mul(a) => stack.extend(a),
            ExprData::Pow { base, exp } => {
                stack.push(base);
                stack.push(exp);
            }
            ExprData::Func { args, .. } => stack.extend(args),
            _ => {}
        }
    }
    if eval_f64(v, pool, &env).is_err() {
        return refuse(format!(
            "integrate: a piece of the Piecewise integrand has an antiderivative that is not \
             finite at the breakpoint {} (the integrand is singular there), so a continuous \
             antiderivative cannot be assembled",
            pool.display(at)
        ));
    }
    Ok(v)
}

fn integrate_piecewise(expr: ExprId, var: ExprId, pool: &ExprPool) -> IntResult {
    let sp = split(expr, var, pool)?;
    let k = sp.bps.len();
    let mut log = DerivationLog::new();
    let mut fs: Vec<ExprId> = Vec::with_capacity(k + 1);
    for &p in &sp.pieces {
        let r = integrate(p, var, pool)?;
        log = log.merge(r.log);
        fs.push(r.value);
    }
    // Glue: G_0 = F_0; G_i = F_i + c_i with G_i(b_i) = G_{i-1}(b_i).
    let mut gs: Vec<ExprId> = Vec::with_capacity(k + 1);
    gs.push(fs[0]);
    for i in 1..=k {
        let b = sp.bps[i - 1].expr;
        let left = value_at(gs[i - 1], var, b, pool)?;
        let right = value_at(fs[i], var, b, pool)?;
        let c = simplify(sub(left, right, pool), pool).value;
        let g = if c == pool.integer(0_i32) {
            fs[i]
        } else {
            simplify(pool.add(vec![fs[i], c]), pool).value
        };
        gs.push(g);
    }
    let result = if k == 0 {
        gs[0]
    } else {
        let branches: Vec<(ExprId, ExprId)> = (0..k)
            .map(|i| (pool.pred_lt(var, sp.bps[i].expr), gs[i]))
            .collect();
        pool.piecewise(branches, gs[k])
    };
    log.push(RewriteStep::simple("integrate_piecewise", expr, result));
    Ok(DerivedExpr::with_log(result, log))
}

// ---------------------------------------------------------------------------
// Definite
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Bound {
    NegInf,
    PosInf,
    /// A finite constant: its value as a rational (exact when `exact`).
    Fin {
        value: Rational,
        exact: bool,
    },
}

fn classify_bound(b: ExprId, pool: &ExprPool) -> Result<Bound, IntegrationError> {
    let pos = pool.pos_infinity();
    if b == pos {
        return Ok(Bound::PosInf);
    }
    let neg = pool.mul(vec![pool.integer(-1_i32), pos]);
    if b == neg || simplify(neg, pool).value == b {
        return Ok(Bound::NegInf);
    }
    if let Ok(r) = eval_exact_rational(b, pool, &HashMap::new()) {
        return Ok(Bound::Fin {
            value: r,
            exact: true,
        });
    }
    if let Ok(v) = eval_f64(b, pool, &HashMap::new()) {
        if let Some(r) = Rational::from_f64(v) {
            return Ok(Bound::Fin {
                value: r,
                exact: false,
            });
        }
    }
    refuse(
        "integrate: a definite integral of a Piecewise integrand needs numeric bounds \
         (a rational, a numeric constant or ±∞) to decide which pieces it covers",
    )
}

/// Position of breakpoint `bp` relative to a finite bound.
fn compare_bp(bp: &mut Breakpoint, bound: &Bound) -> Result<Ordering, IntegrationError> {
    match bound {
        Bound::NegInf => Ok(Ordering::Greater),
        Bound::PosInf => Ok(Ordering::Less),
        Bound::Fin { value, exact } => {
            for _ in 0..2048 {
                if bp.hi < *value {
                    return Ok(Ordering::Less);
                }
                if bp.lo > *value {
                    return Ok(Ordering::Greater);
                }
                if bp.is_exact() {
                    return Ok(bp.lo.cmp(value));
                }
                let width = Rational::from(&bp.hi - &bp.lo);
                if !*exact && width < Rational::from((1, 1_000_000_000_000_i64)) {
                    return refuse(
                        "integrate: a bound is numerically indistinguishable from a \
                         breakpoint of the Piecewise integrand",
                    );
                }
                bp.refine();
            }
            refuse("integrate: could not locate a bound relative to a breakpoint")
        }
    }
}

fn bound_cmp(a: &Bound, b: &Bound) -> Ordering {
    match (a, b) {
        (Bound::NegInf, Bound::NegInf) | (Bound::PosInf, Bound::PosInf) => Ordering::Equal,
        (Bound::NegInf, _) | (_, Bound::PosInf) => Ordering::Less,
        (Bound::PosInf, _) | (_, Bound::NegInf) => Ordering::Greater,
        (Bound::Fin { value: x, .. }, Bound::Fin { value: y, .. }) => x.cmp(y),
    }
}

/// Definite integral of a `Piecewise`-carrying integrand, or `None` when the
/// integrand has none.
pub(crate) fn try_definite(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<IntResult> {
    if !needs_piecewise(expr, var, pool) {
        return None;
    }
    Some(definite_piecewise(expr, var, lower, upper, pool))
}

fn definite_piecewise(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> IntResult {
    let (ba, bb) = (classify_bound(lower, pool)?, classify_bound(upper, pool)?);
    let mut log = DerivationLog::new();
    let order = bound_cmp(&ba, &bb);
    if order == Ordering::Equal {
        let zero = pool.integer(0_i32);
        return Ok(DerivedExpr::with_log(zero, log));
    }
    let (a_expr, b_expr, a, b, sign) = if order == Ordering::Less {
        (lower, upper, ba, bb, 1)
    } else {
        (upper, lower, bb, ba, -1)
    };
    let mut sp = split(expr, var, pool)?;
    let k = sp.bps.len();
    // Position of each breakpoint relative to a and b.
    let mut rel_a = Vec::with_capacity(k);
    let mut rel_b = Vec::with_capacity(k);
    for bp in sp.bps.iter_mut() {
        rel_a.push(compare_bp(bp, &a)?);
        rel_b.push(compare_bp(bp, &b)?);
    }
    let mut terms: Vec<ExprId> = Vec::new();
    for i in 0..=k {
        // Piece i is (bps[i-1], bps[i]).
        let left_ok = i == 0 || rel_b[i - 1] == Ordering::Less; // bps[i-1] < b
        let right_ok = i == k || rel_a[i] == Ordering::Greater; // bps[i] > a
        if !(left_ok && right_ok) {
            continue;
        }
        let lo = if i == 0 || rel_a[i - 1] != Ordering::Greater {
            a_expr
        } else {
            sp.bps[i - 1].expr
        };
        let hi = if i == k || rel_b[i] != Ordering::Less {
            b_expr
        } else {
            sp.bps[i].expr
        };
        let r = integrate_definite(sp.pieces[i], var, lo, hi, pool)?;
        log = log.merge(r.log);
        terms.push(r.value);
    }
    let mut total = if terms.len() == 1 {
        terms[0]
    } else {
        pool.add(terms)
    };
    if sign < 0 {
        total = pool.mul(vec![pool.integer(-1_i32), total]);
    }
    let value = simplify(total, pool).value;
    log.push(RewriteStep::simple(
        "integrate_piecewise_definite",
        expr,
        value,
    ));
    Ok(DerivedExpr::with_log(value, log))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn num(pool: &ExprPool, e: ExprId) -> f64 {
        eval_f64(e, pool, &HashMap::new()).unwrap()
    }

    #[test]
    fn definite_abs_is_five_halves() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let r = integrate_definite(
            pool.func("abs", vec![x]),
            x,
            pool.integer(-1_i32),
            pool.integer(2_i32),
            &pool,
        )
        .unwrap();
        assert_eq!(r.value, pool.rational(5, 2), "{}", pool.display(r.value));
    }

    #[test]
    fn definite_max_x_x2() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("max", vec![x, pool.pow(x, pool.integer(2_i32))]);
        // ∫_{-1}^{2} max(x, x²) dx = 1/3 (−1..0) + 1/2 (0..1) + 7/3 (1..2) = 19/6
        let r = integrate_definite(e, x, pool.integer(-1_i32), pool.integer(2_i32), &pool).unwrap();
        assert_eq!(r.value, pool.rational(19, 6), "{}", pool.display(r.value));
        // Reversed bounds negate.
        let r2 =
            integrate_definite(e, x, pool.integer(2_i32), pool.integer(-1_i32), &pool).unwrap();
        assert_eq!(r2.value, pool.rational(-19, 6));
    }

    #[test]
    fn indefinite_max_is_continuous_piecewise() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("max", vec![x, pool.pow(x, pool.integer(2_i32))]);
        let f = integrate(e, x, &pool).unwrap().value;
        assert!(matches!(pool.get(f), ExprData::Piecewise { .. }));
        let at = |v: f64| {
            let mut m = HashMap::new();
            m.insert(x, v);
            eval_f64(f, &pool, &m).unwrap()
        };
        // Continuity at the breakpoints 0 and 1.
        for b in [0.0, 1.0] {
            assert!((at(b - 1e-9) - at(b + 1e-9)).abs() < 1e-8);
        }
        // F(2) - F(-1) = 19/6.
        assert!((at(2.0) - at(-1.0) - 19.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn irrational_breakpoint_and_refusal() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // |x² − 2|: breakpoints ±√2.
        let g = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-2_i32)]);
        let r = integrate_definite(
            pool.func("abs", vec![g]),
            x,
            pool.integer(0_i32),
            pool.integer(2_i32),
            &pool,
        )
        .unwrap();
        let v = num(&pool, r.value);
        // ∫_0^√2 (2 − x²) + ∫_√2^2 (x² − 2) = 8√2/3 − 4/3
        let want = 8.0 * 2f64.sqrt() / 3.0 - 4.0 / 3.0;
        assert!((v - want).abs() < 1e-12, "{v} vs {want}");
        // A cubic irreducible condition is refused, not guessed.
        let cubic = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            x,
            pool.integer(-1_i32),
        ]);
        assert!(integrate(pool.func("abs", vec![cubic]), x, &pool).is_err());
    }
}
