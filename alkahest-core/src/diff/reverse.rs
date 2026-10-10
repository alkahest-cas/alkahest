//! Phase 14 — Reverse-mode (adjoint) automatic differentiation.
//!
//! `grad(expr, vars, pool)` computes the partial derivatives of `expr` w.r.t.
//! each variable in `vars` in a single backward pass over the expression DAG.
//!
//! This is O(size of DAG) regardless of the number of variables, whereas
//! repeated `diff` calls are O(#vars × size of DAG).
//!
//! The algorithm mirrors the classic reverse-mode / backpropagation recipe:
//!   1. Topological-sort all nodes reachable from `expr`.
//!   2. Seed: `adjoint[expr]` = 1.
//!   3. Walk nodes in reverse topo order; for each node propagate its adjoint
//!      to its children according to the local derivative rule.
//!   4. Return `adjoints[v]` for each requested variable v.
//!
//! # Never a silent zero
//!
//! Through 3.12.0 a node this walker had no rule for — any function outside
//! `sin`/`cos`/`tan`/`exp`/`log`/`sqrt`/`atan`, a `Pow` with a non-constant
//! exponent, a `Piecewise` — passed no adjoint on, so `grad(erf(x), [x])` was
//! `[0]` and `grad(x^y, [x, y])` was `[0, 0]`.  Now:
//!
//! * a `Pow` uses the general power rule
//!   `∂(u^v) = v·u^(v−1)·∂u + u^v·log(u)·∂v`;
//! * a function uses the primitive registry's `diff_reverse` rule when it has
//!   one, and otherwise takes its local partials from the forward
//!   differentiator [`crate::diff::diff`] (one partial per argument, on a
//!   fresh placeholder symbol), so every function `diff` can differentiate,
//!   reverse mode can too;
//! * a `Piecewise`, `RootSum` or other node the walker does not descend into
//!   is differentiated by [`crate::diff::diff`] directly, once per requested
//!   variable it mentions;
//! * anything none of those can differentiate is an error from [`try_grad`]
//!   (`E-DIFF-001`), never a zero.
//!
//! A sub-expression that does not depend on any requested variable is never
//! differentiated at all, so `grad(x·floor(y), [x])` is `floor(y)` even though
//! `floor` has no derivative rule.

use crate::kernel::subs::mentions_var;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::kernel::{IdMap, IdSet};
use crate::simplify::engine::simplify;
use std::collections::HashMap;

use super::diff_impl::DiffError;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Compute `[∂expr/∂vars[0], ∂expr/∂vars[1], …]` via reverse accumulation.
///
/// The returned vector is in the same order as `vars`.  Variables that do not
/// appear in `expr` yield `0`.
///
/// This signature cannot report failure; prefer [`try_grad`], which returns
/// `Err` (`E-DIFF-001` for a function with no derivative rule) instead.  When
/// `try_grad` would fail, `grad` returns a floating-point `NaN` for **every**
/// entry — never `0`, so a failure cannot pass for a genuine zero partial.
///
/// # Example
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::diff::grad;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let y = pool.symbol("y", Domain::Real);
/// let expr = pool.add(vec![
///     pool.mul(vec![x, x]),  // x²
///     pool.mul(vec![x, y]),  // x·y
/// ]);
/// // grad returns [∂/∂x, ∂/∂y]
/// let gs = grad(expr, &[x, y], &pool);
/// // ∂/∂x (x² + x·y) = 2x + y,  ∂/∂y = x
/// println!("∂/∂x = {}", pool.display(gs[0]));
/// println!("∂/∂y = {}", pool.display(gs[1]));
/// ```
pub fn grad(expr: ExprId, vars: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    match try_grad(expr, vars, pool) {
        Ok(gs) => gs,
        Err(_) => {
            let nan = pool.float(f64::NAN, 53);
            vec![nan; vars.len()]
        }
    }
}

/// Fallible reverse-mode gradient: `[∂expr/∂vars[0], …]`, or the reason a
/// partial could not be computed.
///
/// # Errors
///
/// [`DiffError::UnknownFunction`] (`E-DIFF-001`) when a function that depends
/// on a requested variable has no derivative rule (e.g. `floor(x)`, or an
/// undefined `f(x)`).  The forward differentiator's errors propagate
/// unchanged from the nodes reverse mode delegates to it.
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::diff::try_grad;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let y = pool.symbol("y", Domain::Real);
/// let gs = try_grad(pool.pow(x, y), &[x, y], &pool).unwrap();
/// // ∂/∂x x^y = y·x^(y−1),  ∂/∂y x^y = x^y·log(x)
/// assert_ne!(gs[0], pool.integer(0_i32));
/// assert_ne!(gs[1], pool.integer(0_i32));
///
/// let floor_x = pool.func("floor", vec![x]);
/// assert!(try_grad(floor_x, &[x], &pool).is_err());
/// ```
pub fn try_grad(expr: ExprId, vars: &[ExprId], pool: &ExprPool) -> Result<Vec<ExprId>, DiffError> {
    if vars.is_empty() {
        return Ok(vec![]);
    }

    let topo = topo_sort(expr, pool);
    let depends = dependency_set(&topo, vars, pool);
    let mut adjoints: IdMap<ExprId> = IdMap::default();
    adjoints.insert(expr, pool.integer(1_i32));

    for &node in topo.iter().rev() {
        if !depends.contains(&node) {
            continue;
        }
        let adj = match adjoints.get(&node).copied() {
            Some(a) => a,
            None => continue,
        };
        propagate(node, adj, vars, &depends, &mut adjoints, pool)?;
    }

    let zero = pool.integer(0_i32);
    Ok(vars
        .iter()
        .map(|&v| {
            let g = adjoints.get(&v).copied().unwrap_or(zero);
            simplify(g, pool).value
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Dependency analysis
// ---------------------------------------------------------------------------

/// The nodes of `topo` whose value depends on at least one of `vars`.
///
/// `topo` is children-before-parents, so one forward sweep suffices for the
/// node kinds the walker descends into; any other node kind (a `Piecewise`,
/// `RootSum`, predicate, …) is checked with a free-variable scan.
fn dependency_set(topo: &[ExprId], vars: &[ExprId], pool: &ExprPool) -> IdSet {
    let mut dep: IdSet = IdSet::default();
    for &node in topo {
        if vars.contains(&node) {
            dep.insert(node);
            continue;
        }
        let d = pool.with(node, |data| match data {
            ExprData::Symbol { .. }
            | ExprData::Integer(_)
            | ExprData::Rational(_)
            | ExprData::Float(_) => Some(false),
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
                Some(args.iter().any(|a| dep.contains(a)))
            }
            ExprData::Pow { base, exp } => Some(dep.contains(base) || dep.contains(exp)),
            _ => None,
        });
        let d = d.unwrap_or_else(|| vars.iter().any(|&v| mentions_var(node, v, pool)));
        if d {
            dep.insert(node);
        }
    }
    dep
}

// ---------------------------------------------------------------------------
// Adjoint propagation
// ---------------------------------------------------------------------------

/// Propagate `adj` (the adjoint of `node`, which depends on some of `vars`)
/// to the children of `node`.
fn propagate(
    node: ExprId,
    adj: ExprId,
    vars: &[ExprId],
    depends: &IdSet,
    adjoints: &mut IdMap<ExprId>,
    pool: &ExprPool,
) -> Result<(), DiffError> {
    enum Op {
        Leaf,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow { base: ExprId, exp: ExprId },
        Func { name: String, args: Vec<ExprId> },
        Opaque,
    }

    let op = pool.with(node, |data| match data {
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Op::Leaf,
        ExprData::Add(args) => Op::Add(args.clone()),
        ExprData::Mul(args) => Op::Mul(args.clone()),
        ExprData::Pow { base, exp } => Op::Pow {
            base: *base,
            exp: *exp,
        },
        ExprData::Func { name, args } => Op::Func {
            name: name.clone(),
            args: args.clone(),
        },
        // Piecewise, RootSum, predicates, quantifiers, BigO: the walker does
        // not descend into these; the forward differentiator handles them.
        _ => Op::Opaque,
    });

    match op {
        Op::Leaf => {}

        // d(u + v + …) / d(u) = 1 — adjoint passes through unchanged
        Op::Add(args) => {
            for child in args {
                if depends.contains(&child) {
                    add_adj(child, adj, adjoints, pool);
                }
            }
        }

        // d(u * v * …) / d(u_i) = product of all other factors
        Op::Mul(args) => {
            for (i, &child) in args.iter().enumerate() {
                if !depends.contains(&child) {
                    continue;
                }
                let other_factors: Vec<ExprId> = args
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &a)| a)
                    .collect();
                let factor = match other_factors.len() {
                    0 => pool.integer(1_i32),
                    1 => other_factors[0],
                    _ => pool.mul(other_factors),
                };
                let contrib = pool.mul(vec![adj, factor]);
                add_adj(child, contrib, adjoints, pool);
            }
        }

        // General power rule:
        //   ∂(u^v)/∂u = v·u^(v−1),   ∂(u^v)/∂v = u^v·log(u).
        Op::Pow { base, exp } => {
            if depends.contains(&base) {
                let const_r = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
                    ExprData::Rational(q) => Some(q.0.clone()),
                    _ => None,
                });
                let exp_minus_1 = match const_r {
                    Some(r) => super::diff_impl::const_node(pool, r - 1),
                    None => pool.add(vec![exp, pool.integer(-1_i32)]),
                };
                let base_pow = pool.pow(base, exp_minus_1);
                let contrib = pool.mul(vec![adj, exp, base_pow]);
                add_adj(base, contrib, adjoints, pool);
            }
            if depends.contains(&exp) {
                let log_base = pool.func("log", vec![base]);
                let contrib = pool.mul(vec![adj, node, log_base]);
                add_adj(exp, contrib, adjoints, pool);
            }
        }

        // Chain rule: ∂f(a₁,…,aₙ)/∂aᵢ for each argument that depends on vars.
        Op::Func { name, args } => {
            for (arg, c) in func_cotangents(&name, &args, adj, depends, pool)? {
                add_adj(arg, c, adjoints, pool);
            }
        }

        // No reverse rule for this node kind: differentiate it forward, once
        // per requested variable it mentions, and credit that variable.
        Op::Opaque => {
            for &v in vars {
                if !mentions_var(node, v, pool) {
                    continue;
                }
                let d = super::diff_impl::diff(node, v, pool)?.value;
                let contrib = pool.mul(vec![adj, d]);
                add_adj(v, contrib, adjoints, pool);
            }
        }
    }
    Ok(())
}

/// Accumulate `contribution` into the adjoint of `node` (add to existing).
fn add_adj(node: ExprId, contribution: ExprId, adjoints: &mut IdMap<ExprId>, pool: &ExprPool) {
    match adjoints.get_mut(&node) {
        Some(current) => {
            let new_val = pool.add(vec![*current, contribution]);
            *current = new_val;
        }
        None => {
            adjoints.insert(node, contribution);
        }
    }
}

/// The primitive registry used for reverse-mode dispatch (unprobed: only the
/// `diff_reverse` slot is called, and `None` means "no reverse rule").
fn reverse_registry() -> &'static crate::primitive::PrimitiveRegistry {
    static REGISTRY: std::sync::OnceLock<crate::primitive::PrimitiveRegistry> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(crate::primitive::PrimitiveRegistry::dispatch_registry)
}

/// The cotangent `adj · ∂f/∂aᵢ` for each argument `aᵢ` of `name(args)` that
/// depends on a requested variable.
///
/// Uses the registry's `diff_reverse` rule when the primitive has one.
/// Otherwise each partial comes from the forward differentiator: `aᵢ` is
/// replaced by a fresh placeholder symbol `t`, `∂f/∂t` is taken by
/// [`crate::diff::diff`] (which knows every registry `diff_forward` rule and
/// refuses a function it cannot differentiate), and `t` is substituted back.
fn func_cotangents(
    name: &str,
    args: &[ExprId],
    adj: ExprId,
    depends: &IdSet,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, ExprId)>, DiffError> {
    if let Some(cots) = reverse_registry().diff_reverse(name, args, adj, pool) {
        if cots.len() == args.len() {
            return Ok(args
                .iter()
                .zip(cots)
                .filter(|(a, _)| depends.contains(a))
                .map(|(&a, c)| (a, c))
                .collect());
        }
    }

    let mut out = Vec::new();
    for (i, &arg) in args.iter().enumerate() {
        if !depends.contains(&arg) {
            continue;
        }
        let local = local_partial_via_diff(name, args, i, pool)?;
        out.push((arg, pool.mul(vec![adj, local])));
    }
    Ok(out)
}

/// `∂ name(args) / ∂ args[i]`, evaluated at `args`, via the forward
/// differentiator on a placeholder symbol.
fn local_partial_via_diff(
    name: &str,
    args: &[ExprId],
    i: usize,
    pool: &ExprPool,
) -> Result<ExprId, DiffError> {
    // Complex domain: the placeholder stands for an arbitrary sub-expression,
    // so no real/positivity assumption may leak into the derivative.
    let t = pool.symbol(format!("__alkahest_grad_arg{i}"), Domain::Complex);
    let mut targs = args.to_vec();
    targs[i] = t;
    let f_t = pool.func(name, targs);
    let d = super::diff_impl::diff(f_t, t, pool)?.value;
    let mut map = HashMap::new();
    map.insert(t, args[i]);
    Ok(crate::kernel::subs(d, &map, pool))
}

// ---------------------------------------------------------------------------
// Topological sort (DFS post-order: children before parents)
// ---------------------------------------------------------------------------

fn topo_sort(root: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut visited: IdSet = IdSet::default();
    let mut order: Vec<ExprId> = Vec::new();
    dfs_post(root, pool, &mut visited, &mut order);
    order
}

fn dfs_post(node: ExprId, pool: &ExprPool, visited: &mut IdSet, order: &mut Vec<ExprId>) {
    if !visited.insert(node) {
        return;
    }
    let children = pool.with(node, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        _ => vec![],
    });
    for child in children {
        dfs_post(child, pool, visited, order);
    }
    order.push(node);
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn grad_constant_is_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let five = pool.integer(5_i32);
        let gs = grad(five, &[x], &pool);
        assert_eq!(gs[0], pool.integer(0_i32));
    }

    #[test]
    fn grad_identity() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let gs = grad(x, &[x], &pool);
        assert_eq!(gs[0], pool.integer(1_i32));
    }

    #[test]
    fn grad_x_squared() {
        // ∂(x²)/∂x = 2x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let gs = grad(x2, &[x], &pool);
        // Expect 2*x  (may be in different form; check string repr)
        let result = pool.display(gs[0]).to_string();
        assert!(
            result.contains("x") && result.contains("2"),
            "got: {result}"
        );
    }

    #[test]
    fn grad_multivariate() {
        // f = x*y,  ∂f/∂x = y, ∂f/∂y = x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let f = pool.mul(vec![x, y]);
        let gs = grad(f, &[x, y], &pool);
        assert_eq!(gs[0], y, "∂(xy)/∂x should be y");
        assert_eq!(gs[1], x, "∂(xy)/∂y should be x");
    }

    #[test]
    fn grad_x_squared_plus_xy() {
        // f = x² + x·y
        // ∂f/∂x = 2x + y,  ∂f/∂y = x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let xy = pool.mul(vec![x, y]);
        let f = pool.add(vec![x2, xy]);
        let gs = grad(f, &[x, y], &pool);
        // ∂/∂y should just be x
        assert_eq!(gs[1], x, "∂f/∂y should be x");
        // ∂/∂x should contain both 2*x and y
        let dx_str = pool.display(gs[0]).to_string();
        assert!(
            dx_str.contains("x") && dx_str.contains("y"),
            "got: {dx_str}"
        );
    }

    #[test]
    fn grad_fractional_power_agrees_with_diff() {
        // ∂/∂x (x²+1)^{1/2} via grad vs symbolic diff.
        use crate::diff::diff;
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let base = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.pow(base, pool.rational(1_i32, 2_i32));
        let sym = diff(f, x, &pool).unwrap().value;
        let rev = grad(f, &[x], &pool)[0];
        assert_eq!(
            crate::simplify::engine::simplify(sym, &pool).value,
            crate::simplify::engine::simplify(rev, &pool).value
        );
    }

    #[test]
    fn grad_agrees_with_diff_for_polynomial() {
        // f = x³ + 2x²   ∂f/∂x computed both ways
        use crate::diff::diff;
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let x3 = pool.pow(x, pool.integer(3_i32));
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.add(vec![x3, pool.mul(vec![two, x2])]);

        let sym = diff(f, x, &pool).unwrap().value;
        let rev = grad(f, &[x], &pool)[0];

        // Both should simplify to the same expression
        let sym_s = pool.display(sym).to_string();
        let rev_s = pool.display(rev).to_string();
        assert_eq!(sym_s, rev_s, "diff={sym_s}, grad={rev_s}");
    }

    #[test]
    fn grad_sin() {
        // ∂sin(x)/∂x = cos(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let gs = grad(sin_x, &[x], &pool);
        let expected = pool.func("cos", vec![x]);
        assert_eq!(gs[0], expected);
    }

    #[test]
    fn grad_exp() {
        // ∂exp(x)/∂x = exp(x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let gs = grad(exp_x, &[x], &pool);
        assert_eq!(gs[0], exp_x);
    }

    #[test]
    fn grad_atan() {
        // ∂atan(x)/∂x = 1/(1+x²)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let atan_x = pool.func("atan", vec![x]);
        let gs = grad(atan_x, &[x], &pool);
        let one_plus_x2 = pool.add(vec![pool.integer(1_i32), pool.pow(x, pool.integer(2_i32))]);
        let expected = pool.pow(one_plus_x2, pool.integer(-1_i32));
        assert_eq!(gs[0], expected, "d/dx atan(x) should be 1/(1+x²)");
    }

    #[test]
    fn grad_unrelated_var_is_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.mul(vec![x, x]);
        let gs = grad(expr, &[y], &pool);
        assert_eq!(gs[0], pool.integer(0_i32));
    }

    // -----------------------------------------------------------------------
    // W1 regression: no silent zeros
    // -----------------------------------------------------------------------

    fn eval_at(e: ExprId, pool: &ExprPool, binds: &[(ExprId, f64)]) -> f64 {
        let m: std::collections::HashMap<ExprId, f64> = binds.iter().copied().collect();
        crate::jit::eval_interp(e, &m, pool)
            .unwrap_or_else(|| panic!("eval of {} failed", pool.display(e)))
    }

    /// Central finite difference of `e` in `binds[k]`.
    fn fd(e: ExprId, pool: &ExprPool, binds: &[(ExprId, f64)], k: usize) -> f64 {
        let h = 1e-6;
        let mut up = binds.to_vec();
        let mut dn = binds.to_vec();
        up[k].1 += h;
        dn[k].1 -= h;
        (eval_at(e, pool, &up) - eval_at(e, pool, &dn)) / (2.0 * h)
    }

    fn assert_close(got: f64, want: f64, what: &str) {
        let tol = 1e-6 * (1.0 + want.abs());
        assert!(
            (got - want).abs() < tol,
            "{what}: grad gave {got}, expected {want}"
        );
    }

    #[test]
    fn grad_every_special_function_matches_finite_differences() {
        // Each of these used to come back as a silent 0.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cases: &[(&str, f64)] = &[
            ("erf", 0.37),
            ("erfc", 0.37),
            ("gamma", 1.37),
            ("digamma", 1.37),
            ("lambert_w", 0.37),
            ("asin", 0.37),
            ("acos", 0.37),
            ("sinh", 0.37),
            ("cosh", 0.37),
            ("tanh", 0.37),
            ("asinh", 0.37),
            ("acosh", 1.37),
            ("atanh", 0.37),
            ("bessel_j0", 0.37),
            ("bessel_j1", 0.37),
            ("Si", 0.37),
            ("Ci", 0.37),
            ("Ei", 0.37),
            ("Shi", 0.37),
            ("Chi", 0.37),
            ("dilog", 0.37),
            ("fresnels", 0.37),
            ("fresnelc", 0.37),
            ("EllipticK", 0.37),
            ("EllipticE", 0.37),
        ];
        for &(name, xv) in cases {
            // Compose with x² so the chain rule is exercised too.
            let arg = pool.mul(vec![x, pool.rational(3, 2)]);
            let f = pool.func(name, vec![arg]);
            let g = try_grad(f, &[x], &pool).unwrap_or_else(|e| panic!("{name}: {e}"))[0];
            assert_ne!(g, pool.integer(0_i32), "{name}: silent zero");
            let binds = [(x, xv / 1.5)];
            assert_close(eval_at(g, &pool, &binds), fd(f, &pool, &binds, 0), name);
        }
    }

    #[test]
    fn grad_atan2_both_arguments() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let f = pool.func("atan2", vec![pool.mul(vec![x, y]), pool.add(vec![x, y])]);
        let gs = try_grad(f, &[x, y], &pool).unwrap();
        let binds = [(x, 0.7), (y, -1.3)];
        for k in 0..2 {
            assert_close(
                eval_at(gs[k], &pool, &binds),
                fd(f, &pool, &binds, k),
                "atan2",
            );
        }
    }

    #[test]
    fn grad_general_power_rule() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let binds = [(x, 1.7), (y, 0.6), (a, 2.3)];
        let exprs = [
            pool.pow(x, y),                                       // x^y
            pool.pow(x, a),                                       // x^a, a not a grad var
            pool.pow(pool.integer(2_i32), x),                     // 2^x
            pool.pow(x, x),                                       // x^x
            pool.pow(pool.add(vec![x, y]), pool.mul(vec![x, y])), // (x+y)^(xy)
        ];
        for f in exprs {
            let gs = try_grad(f, &[x, y], &pool).unwrap();
            for k in 0..2 {
                assert_close(
                    eval_at(gs[k], &pool, &binds),
                    fd(f, &pool, &binds, k),
                    &pool.display(f).to_string(),
                );
            }
        }
        // ∂(x^y)/∂x and ∂(x^y)/∂y are both non-zero.
        let gs = try_grad(pool.pow(x, y), &[x, y], &pool).unwrap();
        assert_ne!(gs[0], pool.integer(0_i32));
        assert_ne!(gs[1], pool.integer(0_i32));
    }

    #[test]
    fn grad_black_scholes_vega_and_theta() {
        // C = S·Φ(d1) − K·e^{−rT}·Φ(d2), Φ(z) = (1 + erf(z/√2))/2.
        let pool = p();
        let s = pool.symbol("S", Domain::Real);
        let k = pool.symbol("K", Domain::Real);
        let r = pool.symbol("r", Domain::Real);
        let sig = pool.symbol("sigma", Domain::Real);
        let t = pool.symbol("T", Domain::Real);
        let half = pool.rational(1, 2);
        let neg = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
        let inv = |e: ExprId| pool.pow(e, pool.integer(-1_i32));
        let sqrt_t = pool.pow(t, half);
        let sig_sqrt_t = pool.mul(vec![sig, sqrt_t]);
        let log_sk = pool.func("log", vec![pool.mul(vec![s, inv(k)])]);
        let drift = pool.mul(vec![
            pool.add(vec![
                r,
                pool.mul(vec![half, pool.pow(sig, pool.integer(2_i32))]),
            ]),
            t,
        ]);
        let d1 = pool.mul(vec![pool.add(vec![log_sk, drift]), inv(sig_sqrt_t)]);
        let d2 = pool.add(vec![d1, neg(sig_sqrt_t)]);
        let phi = |z: ExprId| {
            let erf = pool.func(
                "erf",
                vec![pool.mul(vec![z, pool.pow(pool.integer(2_i32), pool.rational(-1, 2))])],
            );
            pool.mul(vec![half, pool.add(vec![pool.integer(1_i32), erf])])
        };
        let disc = pool.func("exp", vec![neg(pool.mul(vec![r, t]))]);
        let c = pool.add(vec![
            pool.mul(vec![s, phi(d1)]),
            neg(pool.mul(vec![k, disc, phi(d2)])),
        ]);
        let gs = try_grad(c, &[sig, t], &pool).unwrap();
        let binds = [(s, 100.0), (k, 95.0), (r, 0.05), (sig, 0.2), (t, 0.5)];
        // Closed forms: vega = S·φ(d1)·√T, ∂C/∂T = S·φ(d1)·σ/(2√T) + r·K·e^{−rT}·Φ(d2).
        let (sv, kv, rv, sg, tv) = (100.0_f64, 95.0_f64, 0.05_f64, 0.2_f64, 0.5_f64);
        let d1v = ((sv / kv).ln() + (rv + 0.5 * sg * sg) * tv) / (sg * tv.sqrt());
        let d2v = d1v - sg * tv.sqrt();
        let pdf = (-0.5 * d1v * d1v).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let cdf2 = 0.5 * (1.0 + libm_erf(d2v / 2f64.sqrt()));
        let vega = sv * pdf * tv.sqrt();
        let dc_dt = sv * pdf * sg / (2.0 * tv.sqrt()) + rv * kv * (-rv * tv).exp() * cdf2;
        assert_close(eval_at(gs[0], &pool, &binds), vega, "vega");
        assert_close(eval_at(gs[1], &pool, &binds), dc_dt, "dC/dT");
    }

    fn libm_erf(z: f64) -> f64 {
        crate::primitive::PrimitiveRegistry::dispatch_registry()
            .numeric_f64("erf", &[z])
            .unwrap()
    }

    #[test]
    fn grad_unknown_function_is_an_error_not_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        for f in [
            pool.func("floor", vec![x]),
            pool.func("my_undefined_f", vec![x]),
            pool.mul(vec![y, pool.func("trigamma", vec![pool.mul(vec![x, x])])]),
        ] {
            let err = try_grad(f, &[x, y], &pool).unwrap_err();
            assert!(matches!(err, DiffError::UnknownFunction(_)), "{err:?}");
            // The infallible wrapper must not report a zero either.
            for g in grad(f, &[x, y], &pool) {
                assert!(pool.with(g, |d| matches!(d, ExprData::Float(f) if f.inner.is_nan())));
            }
        }
    }

    #[test]
    fn grad_ignores_undifferentiable_nodes_free_of_the_variables() {
        // floor(y) has no derivative, but ∂/∂x only needs it as a coefficient.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let fy = pool.func("floor", vec![y]);
        let gs = try_grad(pool.mul(vec![x, fy]), &[x], &pool).unwrap();
        assert_eq!(gs[0], fy);
    }

    #[test]
    fn grad_piecewise_agrees_with_diff() {
        // Used to be treated as an atom: ∂/∂x Piecewise((x², x>0), −x) was 0.
        use crate::diff::diff;
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let cond = pool.pred_gt(x, pool.integer(0_i32));
        let pw = pool.piecewise(
            vec![(cond, pool.pow(x, pool.integer(2_i32)))],
            pool.mul(vec![pool.integer(-1_i32), x]),
        );
        let gs = try_grad(pw, &[x], &pool).unwrap();
        assert_ne!(gs[0], pool.integer(0_i32));
        assert_eq!(gs[0], diff(pw, x, &pool).unwrap().value);
        // Inside a product, the Piecewise still passes its adjoint on.
        let f = pool.mul(vec![y, pw]);
        let gs = try_grad(f, &[x, y], &pool).unwrap();
        assert_ne!(gs[0], pool.integer(0_i32));
        assert!(mentions_var(gs[0], y, &pool), "{}", pool.display(gs[0]));
    }
}
