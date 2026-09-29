/// Substitution primitive: replace sub-expressions according to a mapping.
///
/// `subs(expr, mapping, pool)` walks the expression DAG and replaces any node
/// that appears as a key in `mapping` with the corresponding value.
/// The traversal is top-down: if a node matches, its children are not further
/// traversed (the replacement is returned as-is).
///
/// # Keys are matched as whole nodes, never as part of a sum or product
///
/// A key only ever replaces a node that is *structurally identical* to it.  It
/// is not matched against a sub-*multiset* of a wider `Add` or `Mul`: with
/// `x + y → z`, `subs` rewrites `(x + y)` and `sin(x + y)`, but leaves
/// `x + y + 1` alone, because that is one flat three-term `Add` and `x + y` is
/// not a node inside it.
///
/// This used to depend on how the sum happened to be spelled — `Add` and `Mul`
/// were built as left-associative binary chains, so `x + y + 1` contained an
/// `x + y` node and was rewritten, while `1 + x + y` contained a `1 + x` node
/// and was not.  Both now build the same flat node and neither is rewritten,
/// which is the consistent behaviour rather than a coin flip on spelling.
///
/// For associative-commutative matching against part of a sum or product, use
/// the pattern API ([`crate::pattern`]), which is AC-aware; `subs` is
/// deliberately the cheap exact-node primitive.
///
/// # Example
///
/// ```
/// # use alkahest_cas::kernel::{Domain, ExprPool};
/// # use alkahest_cas::kernel::subs::subs;
/// # use std::collections::HashMap;
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let y = pool.symbol("y", Domain::Real);
/// let expr = pool.add(vec![x, pool.integer(1_i32)]);
/// let mut mapping = HashMap::new();
/// mapping.insert(x, y);
/// let result = subs(expr, &mapping, &pool);
/// // (x + 1) with x→y  becomes (y + 1)
/// assert_eq!(result, pool.add(vec![y, pool.integer(1_i32)]));
/// ```
///
/// # Cost on shared sub-expressions
///
/// The walk is memoised per call, so a sub-expression shared by many parents
/// (a DAG, e.g. the Chebyshev recurrence `T_{n+1} = 2x·T_n − T_{n−1}`) is
/// rewritten once, not once per path: the cost is linear in the number of
/// *distinct* nodes.  A node none of whose children changed is returned as
/// the original [`ExprId`] rather than re-interned.
use crate::kernel::eval_const::try_predicate_bool;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::{HashMap, HashSet};

/// Replace sub-expressions according to `mapping`.
///
/// Keys and values are [`ExprId`]s in the same pool.  If `expr` itself appears
/// as a key, the corresponding value is returned immediately.  Otherwise the
/// substitution recurses into children.
pub fn subs(expr: ExprId, mapping: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    subs_memo(expr, mapping, pool, &mut memo)
}

/// Rewrite each of `args`; `None` when every child came back unchanged.
fn subs_args(
    args: &[ExprId],
    mapping: &HashMap<ExprId, ExprId>,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Option<Vec<ExprId>> {
    let new_args: Vec<ExprId> = args
        .iter()
        .map(|&a| subs_memo(a, mapping, pool, memo))
        .collect();
    (new_args.as_slice() != args).then_some(new_args)
}

/// Worker for [`subs`].  `memo` maps a node to its image **under this exact
/// `mapping`**; a binder that shadows a key gets a fresh memo for its body
/// (see the `Forall`/`Exists` arms), because the same node inside and outside
/// that scope can have different images.
fn subs_memo(
    expr: ExprId,
    mapping: &HashMap<ExprId, ExprId>,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> ExprId {
    if let Some(&replacement) = mapping.get(&expr) {
        return replacement;
    }
    if let Some(&done) = memo.get(&expr) {
        return done;
    }
    let out = match pool.get(expr) {
        ExprData::Add(args) => match subs_args(&args, mapping, pool, memo) {
            Some(new_args) => pool.add(new_args),
            None => expr,
        },
        ExprData::Mul(args) => match subs_args(&args, mapping, pool, memo) {
            Some(new_args) => pool.mul(new_args),
            None => expr,
        },
        ExprData::Pow { base, exp } => {
            let b = subs_memo(base, mapping, pool, memo);
            let e = subs_memo(exp, mapping, pool, memo);
            if b == base && e == exp {
                expr
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => match subs_args(&args, mapping, pool, memo) {
            Some(new_args) => pool.func(name, new_args),
            None => expr,
        },
        ExprData::Piecewise { branches, default } => {
            let new_branches: Vec<(ExprId, ExprId)> = branches
                .iter()
                .map(|&(c, v)| {
                    (
                        subs_memo(c, mapping, pool, memo),
                        subs_memo(v, mapping, pool, memo),
                    )
                })
                .collect();
            let nd = subs_memo(default, mapping, pool, memo);
            if nd == default && new_branches == branches {
                expr
            } else {
                pool.piecewise(new_branches, nd)
            }
        }
        ExprData::Predicate { kind, args } => match subs_args(&args, mapping, pool, memo) {
            Some(new_args) => pool.predicate(kind, new_args),
            None => expr,
        },
        ExprData::Forall { var, body } => {
            let nb = subs_binder_body(var, body, mapping, pool, memo);
            if nb == body {
                expr
            } else {
                pool.forall(var, nb)
            }
        }
        ExprData::Exists { var, body } => {
            let nb = subs_binder_body(var, body, mapping, pool, memo);
            if nb == body {
                expr
            } else {
                pool.exists(var, nb)
            }
        }
        ExprData::BigO(arg) => {
            let a = subs_memo(arg, mapping, pool, memo);
            if a == arg {
                expr
            } else {
                pool.big_o(a)
            }
        }
        // Atoms have no children — if not in mapping, return as-is.  (A
        // `RootSum` is deliberately not descended into, as before.)
        _ => expr,
    };
    memo.insert(expr, out);
    out
}

/// Substitute into the body of a `Forall`/`Exists` binding `var`.
///
/// The bound variable shadows any key equal to it.  When it does shadow one,
/// the body is rewritten under the reduced mapping with its **own** memo:
/// entries in the outer memo were computed with `var` still mapped and must
/// not leak into (or out of) the binder's scope.
fn subs_binder_body(
    var: ExprId,
    body: ExprId,
    mapping: &HashMap<ExprId, ExprId>,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> ExprId {
    if mapping.contains_key(&var) {
        let mut inner = mapping.clone();
        inner.remove(&var);
        let mut scope_memo: HashMap<ExprId, ExprId> = HashMap::new();
        subs_memo(body, &inner, pool, &mut scope_memo)
    } else {
        subs_memo(body, mapping, pool, memo)
    }
}

/// Fold predicates with numeric arguments (e.g. `(2 > 0)` → `True`) and simplify
/// piecewise when a branch condition becomes provably true/false.
///
/// Memoised per call, like [`subs`]: linear in the number of distinct nodes.
pub fn fold_predicates(expr: ExprId, pool: &ExprPool) -> ExprId {
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    fold_predicates_memo(expr, pool, &mut memo)
}

fn fold_predicates_memo(
    expr: ExprId,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> ExprId {
    if let Some(&done) = memo.get(&expr) {
        return done;
    }
    let out = fold_predicates_node(expr, pool, memo);
    memo.insert(expr, out);
    out
}

fn fold_predicates_node(
    expr: ExprId,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> ExprId {
    let fold_all = |args: &[ExprId], memo: &mut HashMap<ExprId, ExprId>| -> Vec<ExprId> {
        args.iter()
            .map(|&a| fold_predicates_memo(a, pool, memo))
            .collect()
    };
    match pool.get(expr) {
        ExprData::Predicate { kind, args } => {
            let folded_args = fold_all(&args, memo);
            if let Some(b) = try_predicate_bool(&kind, &folded_args, pool) {
                return pool.predicate(
                    if b {
                        PredicateKind::True
                    } else {
                        PredicateKind::False
                    },
                    vec![],
                );
            }
            pool.predicate(kind.clone(), folded_args)
        }
        ExprData::Piecewise { branches, default } => {
            let mut folded_branches = Vec::with_capacity(branches.len());
            for (c, v) in branches {
                let fc = fold_predicates_memo(c, pool, memo);
                let fv = fold_predicates_memo(v, pool, memo);
                folded_branches.push((fc, fv));
            }
            let fd = fold_predicates_memo(default, pool, memo);
            for (c, v) in &folded_branches {
                if matches!(
                    pool.get(*c),
                    ExprData::Predicate {
                        kind: PredicateKind::True,
                        ..
                    }
                ) {
                    return fold_predicates_memo(*v, pool, memo);
                }
            }
            let remaining: Vec<(ExprId, ExprId)> = folded_branches
                .into_iter()
                .filter(|(c, _)| {
                    !matches!(
                        pool.get(*c),
                        ExprData::Predicate {
                            kind: PredicateKind::False,
                            ..
                        }
                    )
                })
                .collect();
            if remaining.is_empty() {
                return fd;
            }
            pool.piecewise(remaining, fd)
        }
        ExprData::Add(args) => {
            let new_args = fold_all(&args, memo);
            pool.add(new_args)
        }
        ExprData::Mul(args) => {
            let new_args = fold_all(&args, memo);
            pool.mul(new_args)
        }
        ExprData::Pow { base, exp } => {
            let b = fold_predicates_memo(base, pool, memo);
            let e = fold_predicates_memo(exp, pool, memo);
            pool.pow(b, e)
        }
        ExprData::Func { name, args } => {
            let new_args = fold_all(&args, memo);
            pool.func(name, new_args)
        }
        _ => expr,
    }
}

// ---------------------------------------------------------------------------
// Variable occurrence
// ---------------------------------------------------------------------------

/// `true` when `var` occurs **free** anywhere in `expr`.
///
/// Descends into every node kind, including the branching and binding ones a
/// quick `Add`/`Mul`/`Pow`/`Func` walk skips (`Piecewise`, `Predicate`,
/// `RootSum`, `Forall`/`Exists`, `BigO`), so it never under-reports a
/// dependence.  A binder whose bound variable *is* `var` shadows it: the body
/// of `Forall(var, …)`, `Exists(var, …)` or `RootSum(p, var, …)` does not count
/// (a `RootSum`'s defining polynomial is outside the binder and does).
///
/// Iterative, with a visited set, so the cost is linear in the number of
/// distinct nodes of a DAG (a tree walk is exponential on e.g. the Chebyshev
/// recurrence) and deep chains cannot overflow the stack.  A node reached
/// twice is known var-free the second time — the answer does not depend on
/// the path, since shadowing only ever *skips* a body.
pub(crate) fn mentions_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let mut seen: HashSet<ExprId> = HashSet::new();
    let mut stack: Vec<ExprId> = vec![expr];
    while let Some(node) = stack.pop() {
        if node == var {
            return true;
        }
        if !seen.insert(node) {
            continue;
        }
        pool.with(node, |data| match data {
            ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend_from_slice(xs),
            ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => {
                stack.extend_from_slice(args)
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            ExprData::Piecewise { branches, default } => {
                for &(c, v) in branches {
                    stack.push(c);
                    stack.push(v);
                }
                stack.push(*default);
            }
            ExprData::RootSum {
                poly,
                var: bound,
                body,
            } => {
                stack.push(*poly);
                if *bound != var {
                    stack.push(*body);
                }
            }
            ExprData::Forall { var: bound, body } | ExprData::Exists { var: bound, body } => {
                if *bound != var {
                    stack.push(*body);
                }
            }
            ExprData::BigO(a) => stack.push(*a),
            ExprData::Symbol { .. }
            | ExprData::Integer(_)
            | ExprData::Rational(_)
            | ExprData::Float(_) => {}
        });
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::expr::PredicateKind;
    use crate::kernel::{Domain, ExprData, ExprPool};

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn subs_variable() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let mut m = HashMap::new();
        m.insert(x, y);
        assert_eq!(subs(x, &m, &p), y);
    }

    #[test]
    fn subs_in_add() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let one = p.integer(1_i32);
        let expr = p.add(vec![x, one]);
        let mut m = HashMap::new();
        m.insert(x, y);
        let result = subs(expr, &m, &p);
        assert_eq!(result, p.add(vec![y, one]));
    }

    #[test]
    fn subs_identity_when_no_match() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let m: HashMap<ExprId, ExprId> = HashMap::new();
        // No mapping → returns unchanged
        assert_eq!(subs(x, &m, &p), x);
        assert_eq!(subs(p.add(vec![x, y]), &m, &p), p.add(vec![x, y]));
    }

    #[test]
    fn subs_top_level_match_skips_children() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);
        let xpy = p.add(vec![x, y]); // x+y as key
        let mut m = HashMap::new();
        m.insert(xpy, z); // replace x+y → z

        // A whole-node match still fires, and does not descend into `x`/`y`.
        assert_eq!(subs(xpy, &m, &p), z);
        let wrapped = p.func("sin", vec![xpy]);
        assert_eq!(subs(wrapped, &m, &p), p.func("sin", vec![z]));
    }

    /// A key is matched as a whole node, never as a sub-multiset of a wider
    /// sum.  `x + y + 1` is one flat three-term `Add`, so the `x + y` key does
    /// not appear in it and nothing is rewritten.
    ///
    /// This is a deliberate consequence of `Add`/`Mul` being flat at
    /// construction.  It used to depend on spelling — `x + y + 1` parsed to
    /// `Add([Add([x, y]), 1])` and *was* rewritten, while `1 + x + y` parsed to
    /// `Add([Add([1, x]), y])` and was *not*.  Both are now the same node and
    /// neither is rewritten.  AC matching against part of a sum is the pattern
    /// API's job, not `subs`'.
    #[test]
    fn subs_does_not_match_a_sub_multiset_of_a_flat_sum() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);
        let one = p.integer(1_i32);
        let mut m = HashMap::new();
        m.insert(p.add(vec![x, y]), z);

        let flat = p.add(vec![x, y, one]);
        assert_eq!(subs(flat, &m, &p), flat, "no whole-node match, no rewrite");
        // Both spellings of the sum are the same node, so both agree.
        assert_eq!(p.add(vec![p.add(vec![x, y]), one]), flat);
        assert_eq!(p.add(vec![p.add(vec![one, x]), y]), flat);
    }

    #[test]
    fn subs_multiple_vars() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let expr = p.add(vec![x, y]);
        let mut m = HashMap::new();
        m.insert(x, a);
        m.insert(y, b);
        let result = subs(expr, &m, &p);
        assert_eq!(result, p.add(vec![a, b]));
    }

    #[test]
    fn fold_predicates_numeric_gt() {
        let p = pool();
        let pred = p.pred_gt(p.integer(2_i32), p.integer(0_i32));
        let folded = fold_predicates(pred, &p);
        assert!(matches!(
            p.get(folded),
            ExprData::Predicate {
                kind: PredicateKind::True,
                ..
            }
        ));
    }

    #[test]
    fn fold_predicates_piecewise_selects_branch() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let pw = p.piecewise(
            vec![(p.pred_gt(p.integer(1_i32), p.integer(0_i32)), x)],
            p.integer(0_i32),
        );
        let folded = fold_predicates(pw, &p);
        assert_eq!(folded, x);
    }

    // -----------------------------------------------------------------------
    // DAG sharing, binder scopes, and a differential check against the old
    // tree walk.
    // -----------------------------------------------------------------------

    /// The Chebyshev recurrence `T_{k+1} = 2x·T_k − T_{k−1}`: ~3n distinct
    /// nodes, ~fib(n) root-to-leaf paths.
    fn cheb(p: &ExprPool, x: ExprId, n: usize) -> ExprId {
        let (two, m1) = (p.integer(2_i32), p.integer(-1_i32));
        let (mut a, mut b) = (p.integer(1_i32), x);
        for _ in 1..n {
            let c = p.add(vec![p.mul(vec![two, x, b]), p.mul(vec![m1, a])]);
            a = b;
            b = c;
        }
        b
    }

    /// The pre-memo `subs`: a plain tree walk, kept as the differential
    /// reference.
    fn subs_tree(expr: ExprId, mapping: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
        if let Some(&r) = mapping.get(&expr) {
            return r;
        }
        let all = |args: &[ExprId]| -> Vec<ExprId> {
            args.iter().map(|&a| subs_tree(a, mapping, pool)).collect()
        };
        match pool.get(expr) {
            ExprData::Add(args) => pool.add(all(&args)),
            ExprData::Mul(args) => pool.mul(all(&args)),
            ExprData::Pow { base, exp } => pool.pow(
                subs_tree(base, mapping, pool),
                subs_tree(exp, mapping, pool),
            ),
            ExprData::Func { name, args } => pool.func(name, all(&args)),
            ExprData::Piecewise { branches, default } => {
                let nb = branches
                    .iter()
                    .map(|&(c, v)| (subs_tree(c, mapping, pool), subs_tree(v, mapping, pool)))
                    .collect();
                pool.piecewise(nb, subs_tree(default, mapping, pool))
            }
            ExprData::Predicate { kind, args } => pool.predicate(kind, all(&args)),
            ExprData::Forall { var, body } => {
                let mut m2 = mapping.clone();
                m2.remove(&var);
                pool.forall(var, subs_tree(body, &m2, pool))
            }
            ExprData::Exists { var, body } => {
                let mut m2 = mapping.clone();
                m2.remove(&var);
                pool.exists(var, subs_tree(body, &m2, pool))
            }
            ExprData::BigO(a) => pool.big_o(subs_tree(a, mapping, pool)),
            _ => expr,
        }
    }

    /// The pre-memo occurrence check (tree walk), as the reference for
    /// [`mentions_var`].
    fn mentions_tree(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
        if expr == var {
            return true;
        }
        match pool.get(expr) {
            ExprData::Add(xs) | ExprData::Mul(xs) => {
                xs.iter().any(|&a| mentions_tree(a, var, pool))
            }
            ExprData::Pow { base, exp } => {
                mentions_tree(base, var, pool) || mentions_tree(exp, var, pool)
            }
            ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => {
                args.iter().any(|&a| mentions_tree(a, var, pool))
            }
            ExprData::RootSum {
                poly,
                var: bound,
                body,
            } => mentions_tree(poly, var, pool) || (bound != var && mentions_tree(body, var, pool)),
            ExprData::Piecewise { branches, default } => {
                branches
                    .iter()
                    .any(|&(c, v)| mentions_tree(c, var, pool) || mentions_tree(v, var, pool))
                    || mentions_tree(default, var, pool)
            }
            ExprData::Forall { var: bound, body } | ExprData::Exists { var: bound, body } => {
                bound != var && mentions_tree(body, var, pool)
            }
            ExprData::BigO(a) => mentions_tree(a, var, pool),
            _ => false,
        }
    }

    /// Build a random shared DAG from `(op, i, j)` instructions over a growing
    /// node list (so later nodes reuse earlier ones), including binders over
    /// `x` and `y`.
    fn random_dag(p: &ExprPool, ops: &[(u8, usize, usize)]) -> (Vec<ExprId>, [ExprId; 3]) {
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);
        let c = p.symbol("c", Domain::Real);
        let mut nodes = vec![x, y, z, p.integer(2_i32), p.integer(-1_i32)];
        for &(op, i, j) in ops {
            let a = nodes[i % nodes.len()];
            let b = nodes[j % nodes.len()];
            let n = match op % 10 {
                0 => p.add(vec![a, b]),
                1 => p.mul(vec![a, b]),
                2 => p.pow(a, p.integer((j % 3) as i64 + 2)),
                3 => p.func("sin", vec![a]),
                4 => p.forall(x, p.pred_gt(a, b)),
                5 => p.exists(y, p.pred_lt(a, b)),
                6 => p.piecewise(vec![(p.pred_gt(a, p.integer(0_i32)), b)], a),
                7 => p.big_o(a),
                8 => p.root_sum(
                    p.add(vec![p.pow(c, p.integer(2_i32)), a]),
                    c,
                    p.mul(vec![c, b]),
                ),
                _ => p.add(vec![a, b, p.integer(1_i32)]),
            };
            nodes.push(n);
        }
        (nodes, [x, y, z])
    }

    fn ops_strategy() -> impl proptest::strategy::Strategy<Value = Vec<(u8, usize, usize)>> {
        proptest::collection::vec((0u8..10, 0usize..64, 0usize..64), 1..14)
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(300))]

        /// Memoised `subs` agrees with the old tree walk on random shared DAGs
        /// with binders — including mappings that the binders shadow.
        #[test]
        fn subs_matches_tree_walk(ops in ops_strategy(), pick in 0usize..4) {
            let p = pool();
            let (nodes, [x, y, z]) = random_dag(&p, &ops);
            let root = *nodes.last().unwrap();
            let mut m = HashMap::new();
            match pick {
                0 => {
                    m.insert(x, y);
                }
                1 => {
                    m.insert(x, p.add(vec![y, p.integer(1_i32)]));
                    m.insert(y, x);
                }
                2 => {
                    m.insert(z, p.integer(3_i32));
                    m.insert(y, z);
                }
                _ => {
                    m.insert(nodes[nodes.len() / 2], x);
                }
            }
            proptest::prop_assert_eq!(subs(root, &m, &p), subs_tree(root, &m, &p));
        }

        /// The shared visited-set walker agrees with the old tree walk.
        #[test]
        fn mentions_var_matches_tree_walk(ops in ops_strategy()) {
            let p = pool();
            let (nodes, vars) = random_dag(&p, &ops);
            for &root in nodes.iter().rev().take(4) {
                for v in vars {
                    proptest::prop_assert_eq!(
                        mentions_var(root, v, &p),
                        mentions_tree(root, v, &p)
                    );
                }
            }
        }
    }

    /// A node shared between the inside and the outside of a binder has
    /// different images in the two scopes; the memo must not carry one across.
    #[test]
    fn subs_memo_respects_binder_scope() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let s = p.add(vec![x, p.integer(1_i32)]);
        let bound = p.forall(x, p.pred_gt(s, p.integer(0_i32)));
        let mut m = HashMap::new();
        m.insert(x, y);
        // Whichever scope is visited first, the other must not reuse its image.
        for e in [p.add(vec![s, bound]), p.mul(vec![bound, s])] {
            assert_eq!(subs(e, &m, &p), subs_tree(e, &m, &p));
        }
        let out = subs(p.add(vec![s, bound]), &m, &p);
        let expected = p.add(vec![p.add(vec![y, p.integer(1_i32)]), bound]);
        assert_eq!(out, expected, "x inside ∀x stays bound");
    }

    /// `subs`, `fold_predicates` and `mentions_var` on a DAG with ~120
    /// distinct nodes and ~fib(40) ≈ 10⁸ paths.  The tree walk took 0.54 s
    /// (`subs`) at n = 28, growing ×φ per level.
    #[test]
    fn subs_is_linear_on_a_shared_dag() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let e = cheb(&p, x, 40);
        let t0 = std::time::Instant::now();
        let mut m = HashMap::new();
        m.insert(x, y);
        assert_eq!(subs(e, &m, &p), cheb(&p, y, 40));
        // A mapping that touches nothing returns the very same node.
        let mut none = HashMap::new();
        none.insert(p.symbol("w", Domain::Real), x);
        assert_eq!(subs(e, &none, &p), e);
        assert_eq!(fold_predicates(e, &p), e);
        assert!(mentions_var(e, x, &p));
        assert!(!mentions_var(e, y, &p));
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(2),
            "took {:?}",
            t0.elapsed()
        );
    }
}
