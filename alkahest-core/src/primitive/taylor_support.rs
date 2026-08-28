//! Which constructs the validated-bounds subsystem can actually bound.
//!
//! [`crate::validated`] (`bound_on_box`, `verified_integral`,
//! `verified_no_roots`, `verified_sign`) is *not* driven by the
//! [`Capabilities::NUMERIC_BALL`](super::Capabilities::NUMERIC_BALL) bundle
//! slot.  Pointwise ball arithmetic gives an enclosure of `f` at a ball; a
//! Taylor model additionally needs a polynomial expansion with a rigorous
//! Lagrange remainder, which is written per function in
//! [`crate::validated::taylor`].  The two sets differ: `floor` and `ceil` have
//! real ball arithmetic and no Taylor-model rule (they are not
//! differentiable), so `bound_on_box` refuses them with `E-VALIDATED-001`.
//! The gap used to be much wider — `bessel_j0`, `bessel_j1`, `digamma`,
//! `lambert_w` were all on the wrong side of it until 3.9.0 — which is
//! exactly why the flag is derived rather than listed.
//!
//! Before this module the boundary was only discoverable by hitting it. The
//! flag exposed here closes that gap **without introducing a second list to
//! maintain**: every answer is produced by *running* the real evaluator on a
//! probe expression and looking at whether it refuses with
//! [`ValidatedError::Unsupported`].  There is nothing here to keep in sync —
//! adding a rule to `validated::taylor` flips the flag on the next call, and
//! removing one flips it off.
//!
//! # Example
//!
//! ```
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::primitive::{taylor_model_refusal, taylor_model_supports};
//!
//! assert!(taylor_model_supports("sin"));
//! // Real ball arithmetic, no Taylor-model rule — `floor` is not
//! // differentiable, so there is nothing to expand.
//! assert!(!taylor_model_supports("floor"));
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! assert!(taylor_model_refusal(pool.func("sin", vec![x]), &pool).is_none());
//! assert!(taylor_model_refusal(pool.func("floor", vec![x]), &pool).is_some());
//! ```

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::validated::taylor::taylor_range;
use crate::validated::ValidatedError;
use rug::Float;
use std::collections::{HashMap, HashSet};
use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Probe box. Any non-degenerate box works: whether the evaluator has a
/// *rule* for a construct is a structural question, and the box only decides
/// whether that rule then hits a domain violation — a different error, and one
/// this module deliberately reports as "supported" (the rule exists; the
/// caller picked a bad box).
const PROBE_LO: f64 = 0.25;
const PROBE_HI: f64 = 0.5;
/// Order 1 / 64 bits: dispatch does not depend on either, so probe cheaply.
const PROBE_ORDER: usize = 1;
const PROBE_PREC: u32 = 64;
/// Arities probed by [`taylor_model_supports`]. Today the evaluator only has
/// unary function rules, but probing a range means a future binary rule
/// (`atan2`, `hypot`) is picked up with no edit here.
const MAX_PROBE_ARITY: usize = 3;

/// The evaluator's own description of the first construct in `expr` that it
/// has no rigorous Taylor-model rule for, or `None` if it has a rule for
/// every one of them.
///
/// `None` is exactly the condition "`bound_on_box` will not fail with
/// `E-VALIDATED-001`". It is **not** a promise that `bound_on_box` succeeds:
/// a supported function can still hit a pole, a branch cut or an overflow on
/// a particular box (`E-VALIDATED-003` / `E-VALIDATED-004`), which depends on
/// the box and not on the expression.
///
/// Free symbols are enclosed in an arbitrary probe box, so the answer depends
/// only on the shape of `expr`.
pub fn taylor_model_refusal(expr: ExprId, pool: &ExprPool) -> Option<String> {
    let (symbols, _) = walk_expr(expr, pool);
    let mut probe_box: Vec<(ExprId, Float, Float)> = symbols
        .into_iter()
        .map(|s| {
            (
                s,
                Float::with_val(PROBE_PREC, PROBE_LO),
                Float::with_val(PROBE_PREC, PROBE_HI),
            )
        })
        .collect();
    if probe_box.is_empty() {
        // A constant expression still has to be *evaluated* to learn whether
        // its functions are supported, and the evaluator rejects an empty
        // box. Add `expr` itself as the box variable: only `Symbol` nodes are
        // ever matched against the box, so a non-symbol entry is an unused
        // extra dimension — and unlike interning a fresh symbol, it does not
        // mutate the caller's pool.
        probe_box.push((
            expr,
            Float::with_val(PROBE_PREC, PROBE_LO),
            Float::with_val(PROBE_PREC, PROBE_HI),
        ));
    }
    match taylor_range(expr, pool, &probe_box, PROBE_ORDER, PROBE_PREC) {
        Err(ValidatedError::Unsupported { what }) => Some(what),
        _ => None,
    }
}

/// Every function *call* inside `expr` that the Taylor-model evaluator has no
/// rule for, by name, deduplicated and sorted.
///
/// This is the actionable half of [`taylor_model_refusal`]: the refusal names
/// the first blocking construct, this names all the blocking functions at
/// once, so a caller can decide what to substitute. An expression can be
/// refused with an empty list here — the blocker may be a node kind rather
/// than a function (a symbolic exponent over a non-positive base, a
/// `Piecewise`, …); [`taylor_model_refusal`] stays authoritative.
pub fn taylor_model_blockers(expr: ExprId, pool: &ExprPool) -> Vec<String> {
    let (_, calls) = walk_expr(expr, pool);
    let mut out: Vec<String> = calls
        .into_iter()
        .filter(|(name, arity)| !taylor_model_supports_call(name, *arity))
        .map(|(name, _)| name)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Does the Taylor-model evaluator have a rule for `name` at *any* arity it
/// accepts?  This is the per-primitive flag reported as `taylor_model` in
/// `capabilities()["primitives"]`.
pub fn taylor_model_supports(name: &str) -> bool {
    (1..=MAX_PROBE_ARITY).any(|arity| taylor_model_supports_call(name, arity))
}

/// Does the Taylor-model evaluator have a rule for `name` applied to exactly
/// `arity` arguments?
///
/// Arity matters: the evaluator's rules are unary today, so `atan2(x, y)` is
/// refused for a reason that has nothing to do with whether `atan2` could be
/// bounded in principle.
pub fn taylor_model_supports_call(name: &str, arity: usize) -> bool {
    if arity == 0 || arity > MAX_PROBE_ARITY {
        // The evaluator's `Func` arms all destructure at least one argument,
        // and every arm above `MAX_PROBE_ARITY` is the catch-all refusal.
        return false;
    }
    let cache = &cache()[arity - 1];
    if let Some(&hit) = read_lock(cache).get(name) {
        return hit;
    }
    // Computed *outside* the lock: probing runs the validated evaluator, which
    // must never be able to re-enter this cache while it is held.
    let answer = probe_call(name, arity);
    write_lock(cache).insert(name.to_string(), answer);
    answer
}

// ---------------------------------------------------------------------------
// Probing
// ---------------------------------------------------------------------------

/// Build `name(x₁, …, x_arity)` in a scratch pool and ask the real evaluator.
fn probe_call(name: &str, arity: usize) -> bool {
    let pool = ExprPool::new();
    let args: Vec<ExprId> = (0..arity)
        .map(|i| pool.symbol(format!("__taylor_probe_{i}"), Domain::Real))
        .collect();
    let call = pool.func(name, args);
    taylor_model_refusal(call, &pool).is_none()
}

/// Collect the free symbols and the `(function name, arity)` calls in `expr`.
///
/// Enumerating the nodes is not the same as knowing which of them the
/// evaluator supports — that question is only ever answered by running it.
fn walk_expr(expr: ExprId, pool: &ExprPool) -> (Vec<ExprId>, Vec<(String, usize)>) {
    let mut seen: HashSet<ExprId> = HashSet::new();
    let mut symbols: Vec<ExprId> = Vec::new();
    let mut calls: Vec<(String, usize)> = Vec::new();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let children: Vec<ExprId> = pool.with(id, |data| match data {
            ExprData::Symbol { .. } => {
                symbols.push(id);
                vec![]
            }
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => vec![],
            ExprData::Add(args) | ExprData::Mul(args) => args.clone(),
            ExprData::Pow { base, exp } => vec![*base, *exp],
            ExprData::Func { name, args } => {
                calls.push((name.clone(), args.len()));
                args.clone()
            }
            ExprData::Piecewise { branches, default } => {
                let mut ids: Vec<ExprId> = branches.iter().flat_map(|(c, v)| [*c, *v]).collect();
                ids.push(*default);
                ids
            }
            ExprData::Predicate { args, .. } => args.clone(),
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => vec![*var, *body],
            ExprData::BigO(arg) => vec![*arg],
            ExprData::RootSum { poly, var, body } => vec![*poly, *var, *body],
        });
        stack.extend(children);
    }
    // Deterministic order: the walk visits through a stack, and a box whose
    // dimension order depended on traversal luck would make the probe's
    // (irrelevant, but observable) numerics irreproducible.
    symbols.sort_unstable();
    symbols.dedup();
    (symbols, calls)
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// Probing is pure and deterministic — the same name and arity always get the
/// same answer — but it runs a whole Taylor evaluation, and
/// `PrimitiveRegistry::default_registry()` is rebuilt on hot paths (every
/// `diff` of an unregistered `Func` node). So memoise it process-wide, one
/// map per arity so that a `&str` lookup does not have to allocate a key.
type ProbeCache = [RwLock<HashMap<String, bool>>; MAX_PROBE_ARITY];

fn cache() -> &'static ProbeCache {
    static CACHE: OnceLock<ProbeCache> = OnceLock::new();
    CACHE.get_or_init(|| std::array::from_fn(|_| RwLock::new(HashMap::new())))
}

/// A poisoned probe cache is not a correctness problem: entries are
/// deterministic and inserted one at a time, so recovering the map is always
/// better than propagating a panic from an unrelated thread.
fn read_lock(c: &RwLock<HashMap<String, bool>>) -> RwLockReadGuard<'_, HashMap<String, bool>> {
    c.read().unwrap_or_else(|e| e.into_inner())
}

fn write_lock(c: &RwLock<HashMap<String, bool>>) -> RwLockWriteGuard<'_, HashMap<String, bool>> {
    c.write().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validated::bounds::{bound_on_box, BoundOptions};

    /// The whole point of deriving the flag: it agrees with the subsystem it
    /// describes, for every registered primitive, without a list in between.
    #[test]
    fn flag_matches_bound_on_box_for_every_primitive() {
        let reg = crate::primitive::PrimitiveRegistry::default_registry();
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions {
            order: 2,
            prec: 64,
            tol: 1e-3,
            max_subdivisions: 8,
        };
        for (name, caps) in reg.iter() {
            let flag = caps.contains(crate::primitive::Capabilities::TAYLOR_MODEL);
            let call = pool.func(name, vec![x]);
            let refused_as_unsupported = matches!(
                bound_on_box(call, &pool, &[(x, 0.25, 0.5)], &opts),
                Err(ValidatedError::Unsupported { .. })
            );
            assert_eq!(
                flag, !refused_as_unsupported,
                "`{name}`: taylor_model flag = {flag} but bound_on_box \
                 unsupported = {refused_as_unsupported}"
            );
        }
    }

    /// `NUMERIC_BALL` is not a stale bit that should have been `false`: every
    /// primitive that has it and lacks a Taylor-model rule really does
    /// evaluate as a ball. The two flags answer different questions —
    /// pointwise enclosure vs. a polynomial model with a rigorous remainder —
    /// which is exactly why reading the first as the second is a trap.
    #[test]
    fn numeric_ball_is_accurate_where_it_differs_from_taylor_model() {
        use crate::ball::ArbBall;
        let reg = crate::primitive::PrimitiveRegistry::default_registry();
        let mut differing = 0usize;
        for (name, caps) in reg.iter() {
            if !caps.contains(crate::primitive::Capabilities::NUMERIC_BALL)
                || caps.contains(crate::primitive::Capabilities::TAYLOR_MODEL)
            {
                continue;
            }
            differing += 1;
            // Two probe points, because no single one is inside every
            // domain. Declining an out-of-domain argument is not the failure
            // under test.
            assert!(
                [1.0_f64, 0.5].into_iter().any(|v| reg
                    .numeric_ball(name, &[ArbBall::from_f64(v, 128)])
                    .is_some()),
                "`{name}` advertises numeric_ball but has none"
            );
        }
        assert!(
            differing > 0,
            "the two flags have become the same question — this test is now vacuous"
        );
    }

    #[test]
    fn refusal_names_the_blocking_function() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.mul(vec![x, pool.func("floor", vec![x])]);
        let what = taylor_model_refusal(e, &pool).expect("floor has no Taylor rule");
        assert!(what.contains("floor"), "{what}");
        assert_eq!(taylor_model_blockers(e, &pool), vec!["floor".to_string()]);
    }

    /// The names M7 moved across the boundary in 3.9.0, pinned individually.
    /// Losing one of these silently is a coverage regression a planner's route
    /// depends on, and the derived flag makes that possible in one direction
    /// (delete the rule) even though it makes drift impossible in the other.
    #[test]
    fn the_special_function_rules_are_reachable() {
        for name in [
            "asinh",
            "acosh",
            "atanh",
            "erf",
            "erfc",
            "bessel_j0",
            "bessel_j1",
            "digamma",
            "gamma",
            "lambert_w",
            // Exponential-integral family (see `primitive::expint`). Every
            // one of these is `γ + log|x|` (or nothing) plus a truncated
            // `Σ σ(m)·xᵐ/(m·m!)` evaluated in the model algebra, with a
            // geometric tail bound — so losing the rule would be a silent
            // coverage regression exactly like the 3.9.0 set above.
            "Ei",
            "li",
            "Si",
            "Ci",
            "Shi",
            "Chi",
            // 3.10.0: the rigorous tier has to reach trigamma, or an
            // antiderivative carrying it is unverifiable at the enclosure
            // tier even though `diff` handles it.
            "trigamma",
        ] {
            assert!(taylor_model_supports(name), "`{name}` lost its rule");
        }
        // …and the two that must stay out.
        for name in ["floor", "ceil"] {
            assert!(
                !taylor_model_supports(name),
                "`{name}` is not differentiable; a Taylor rule for it would \
                 advertise coverage it cannot deliver"
            );
        }
    }

    #[test]
    fn supported_expression_has_no_refusal_and_no_blockers() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![pool.func("sin", vec![x]), pool.func("exp", vec![x])]);
        assert!(taylor_model_refusal(e, &pool).is_none());
        assert!(taylor_model_blockers(e, &pool).is_empty());
    }

    /// A domain violation is not a refusal to model: `log` has a rule, the
    /// box is just bad. Reporting it as unsupported would send a planner off
    /// a perfectly good route.
    #[test]
    fn domain_violation_is_not_unsupported() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("log", vec![x]);
        assert!(taylor_model_refusal(e, &pool).is_none());
        let opts = BoundOptions::default();
        assert!(bound_on_box(e, &pool, &[(x, -2.0, -1.0)], &opts).is_err());
    }

    /// Constant expressions have no free symbols; the probe must still reach
    /// the function nodes rather than failing on an empty box.
    #[test]
    fn constant_expression_is_classified() {
        let pool = ExprPool::new();
        let two = pool.integer(2_i32);
        assert!(taylor_model_refusal(pool.func("sin", vec![two]), &pool).is_none());
        assert!(taylor_model_refusal(pool.func("floor", vec![two]), &pool).is_some());
    }

    #[test]
    fn arity_is_part_of_the_question() {
        assert!(taylor_model_supports_call("sin", 1));
        assert!(!taylor_model_supports_call("sin", 2));
        assert!(!taylor_model_supports_call("sin", 0));
        assert!(!taylor_model_supports("atan2"));
    }

    /// The cached answer is the freshly probed answer.
    #[test]
    fn cache_is_transparent() {
        for name in ["sin", "erf", "sqrt", "digamma", "bessel_j0", "floor"] {
            let cached = taylor_model_supports_call(name, 1);
            assert_eq!(cached, probe_call(name, 1), "{name}");
            assert_eq!(cached, taylor_model_supports_call(name, 1), "{name}");
        }
    }
}
