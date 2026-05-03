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

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute `[∂expr/∂vars[0], ∂expr/∂vars[1], …]` via reverse accumulation.
///
/// The returned vector is in the same order as `vars`.  Variables that do not
/// appear in `expr` yield `0`.
///
/// # Example
///
/// ```
/// use alkahest_core::kernel::{Domain, ExprPool};
/// use alkahest_core::diff::grad;
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
    if vars.is_empty() {
        return vec![];
    }

    let topo = topo_sort(expr, pool);
    let mut adjoints: HashMap<ExprId, ExprId> = HashMap::new();
    adjoints.insert(expr, pool.integer(1_i32));

    for &node in topo.iter().rev() {
        let adj = match adjoints.get(&node).copied() {
            Some(a) => a,
            None => continue,
        };
        propagate(node, adj, &mut adjoints, pool);
    }

    let zero = pool.integer(0_i32);
    vars.iter()
        .map(|&v| {
            let g = adjoints.get(&v).copied().unwrap_or(zero);
            simplify(g, pool).value
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Adjoint propagation
// ---------------------------------------------------------------------------

/// Propagate `adj` (the adjoint of `node`) to the children of `node`.
fn propagate(node: ExprId, adj: ExprId, adjoints: &mut HashMap<ExprId, ExprId>, pool: &ExprPool) {
    enum Op {
        Atom,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow { base: ExprId, exp: ExprId },
        Func { name: String, arg: ExprId },
        UnknownFunc,
    }

    let op = pool.with(node, |data| match data {
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Op::Atom,
        ExprData::Add(args) => Op::Add(args.clone()),
        ExprData::Mul(args) => Op::Mul(args.clone()),
        ExprData::Pow { base, exp } => Op::Pow {
            base: *base,
            exp: *exp,
        },
        ExprData::Func { name, args } if args.len() == 1 => Op::Func {
            name: name.clone(),
            arg: args[0],
        },
        ExprData::Func { .. } => Op::UnknownFunc,
        // PA-9: Piecewise and Predicate are treated as atomic in reverse-mode AD.
        ExprData::Piecewise { .. } | ExprData::Predicate { .. } => Op::Atom,
        ExprData::Forall { .. } | ExprData::Exists { .. } => Op::Atom,
    });

    match op {
        Op::Atom | Op::UnknownFunc => {}

        // d(u + v + …) / d(u) = 1 — adjoint passes through unchanged
        Op::Add(args) => {
            for child in args {
                add_adj(child, adj, adjoints, pool);
            }
        }

        // d(u * v * …) / d(u_i) = product of all other factors
        Op::Mul(args) => {
            let n = args.len();
            for i in 0..n {
                let child = args[i];
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

        // d(base^n) / d(base) = n * base^(n-1)   (integer n only)
        Op::Pow { base, exp } => {
            let maybe_n = pool.with(exp, |d| {
                if let ExprData::Integer(n) = d {
                    Some(n.0.clone())
                } else {
                    None
                }
            });
            if let Some(n) = maybe_n {
                let n_minus_1 = pool.integer(n - 1i32);
                let base_pow = pool.pow(base, n_minus_1);
                let contrib = pool.mul(vec![adj, exp, base_pow]);
                add_adj(base, contrib, adjoints, pool);
            }
            // Non-integer exponent: no propagation (same conservative choice as diff_impl)
        }

        // Chain rule: d(f(u)) / d(u) = f'(u)
        Op::Func { name, arg } => {
            if let Some(local) = func_local_deriv(&name, arg, pool) {
                let contrib = pool.mul(vec![adj, local]);
                add_adj(arg, contrib, adjoints, pool);
            }
        }
    }
}

/// Accumulate `contribution` into the adjoint of `node` (add to existing).
fn add_adj(
    node: ExprId,
    contribution: ExprId,
    adjoints: &mut HashMap<ExprId, ExprId>,
    pool: &ExprPool,
) {
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

/// Local derivative of `f(arg)` with respect to `arg`.
fn func_local_deriv(name: &str, arg: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match name {
        "sin" => Some(pool.func("cos", vec![arg])),
        "cos" => {
            let neg_sin = pool.mul(vec![pool.integer(-1_i32), pool.func("sin", vec![arg])]);
            Some(neg_sin)
        }
        "exp" => Some(pool.func("exp", vec![arg])),
        "log" => Some(pool.pow(arg, pool.integer(-1_i32))),
        "sqrt" => {
            // d/dx sqrt(x) = 1/(2*sqrt(x))
            let sqrt_x = pool.func("sqrt", vec![arg]);
            let two_sqrt = pool.mul(vec![pool.integer(2_i32), sqrt_x]);
            Some(pool.pow(two_sqrt, pool.integer(-1_i32)))
        }
        "tan" => {
            // d/dx tan(x) = cos(x)^(-2)
            let cos_x = pool.func("cos", vec![arg]);
            Some(pool.pow(cos_x, pool.integer(-2_i32)))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Topological sort (DFS post-order: children before parents)
// ---------------------------------------------------------------------------

fn topo_sort(root: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut visited: HashSet<ExprId> = HashSet::new();
    let mut order: Vec<ExprId> = Vec::new();
    dfs_post(root, pool, &mut visited, &mut order);
    order
}

fn dfs_post(node: ExprId, pool: &ExprPool, visited: &mut HashSet<ExprId>, order: &mut Vec<ExprId>) {
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
    fn grad_unrelated_var_is_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.mul(vec![x, x]);
        let gs = grad(expr, &[y], &pool);
        assert_eq!(gs[0], pool.integer(0_i32));
    }
}
