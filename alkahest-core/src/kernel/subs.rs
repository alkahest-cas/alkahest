/// Substitution primitive: replace sub-expressions according to a mapping.
///
/// `subs(expr, mapping, pool)` walks the expression DAG and replaces any node
/// that appears as a key in `mapping` with the corresponding value.
/// The traversal is top-down: if a node matches, its children are not further
/// traversed (the replacement is returned as-is).
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
use crate::kernel::eval_const::try_predicate_bool;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// Replace sub-expressions according to `mapping`.
///
/// Keys and values are [`ExprId`]s in the same pool.  If `expr` itself appears
/// as a key, the corresponding value is returned immediately.  Otherwise the
/// substitution recurses into children.
pub fn subs(expr: ExprId, mapping: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
    if let Some(&replacement) = mapping.get(&expr) {
        return replacement;
    }
    let data = pool.get(expr);
    match data {
        ExprData::Add(args) => {
            let new_args: Vec<ExprId> = args.iter().map(|&a| subs(a, mapping, pool)).collect();
            pool.add(new_args)
        }
        ExprData::Mul(args) => {
            let new_args: Vec<ExprId> = args.iter().map(|&a| subs(a, mapping, pool)).collect();
            pool.mul(new_args)
        }
        ExprData::Pow { base, exp } => {
            let b = subs(base, mapping, pool);
            let e = subs(exp, mapping, pool);
            pool.pow(b, e)
        }
        ExprData::Func { name, args } => {
            let new_args: Vec<ExprId> = args.iter().map(|&a| subs(a, mapping, pool)).collect();
            pool.func(name, new_args)
        }
        ExprData::Piecewise { branches, default } => {
            let new_branches: Vec<(ExprId, ExprId)> = branches
                .iter()
                .map(|(c, v)| (subs(*c, mapping, pool), subs(*v, mapping, pool)))
                .collect();
            let nd = subs(default, mapping, pool);
            pool.piecewise(new_branches, nd)
        }
        ExprData::Predicate { kind, args } => {
            let new_args: Vec<ExprId> = args.iter().map(|&a| subs(a, mapping, pool)).collect();
            pool.predicate(kind.clone(), new_args)
        }
        ExprData::Forall { var, body } => {
            let mut m2 = mapping.clone();
            m2.remove(&var);
            let nb = subs(body, &m2, pool);
            pool.forall(var, nb)
        }
        ExprData::Exists { var, body } => {
            let mut m2 = mapping.clone();
            m2.remove(&var);
            let nb = subs(body, &m2, pool);
            pool.exists(var, nb)
        }
        ExprData::BigO(arg) => {
            let a = subs(arg, mapping, pool);
            pool.big_o(a)
        }
        // Atoms have no children — if not in mapping, return as-is
        _ => expr,
    }
}

/// Fold predicates with numeric arguments (e.g. `(2 > 0)` → `True`) and simplify
/// piecewise when a branch condition becomes provably true/false.
pub fn fold_predicates(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Predicate { kind, args } => {
            let folded_args: Vec<ExprId> = args.iter().map(|&a| fold_predicates(a, pool)).collect();
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
                let fc = fold_predicates(c, pool);
                let fv = fold_predicates(v, pool);
                folded_branches.push((fc, fv));
            }
            let fd = fold_predicates(default, pool);
            for (c, v) in &folded_branches {
                if matches!(
                    pool.get(*c),
                    ExprData::Predicate {
                        kind: PredicateKind::True,
                        ..
                    }
                ) {
                    return fold_predicates(*v, pool);
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
            let new_args: Vec<ExprId> = args.iter().map(|&a| fold_predicates(a, pool)).collect();
            pool.add(new_args)
        }
        ExprData::Mul(args) => {
            let new_args: Vec<ExprId> = args.iter().map(|&a| fold_predicates(a, pool)).collect();
            pool.mul(new_args)
        }
        ExprData::Pow { base, exp } => {
            let b = fold_predicates(base, pool);
            let e = fold_predicates(exp, pool);
            pool.pow(b, e)
        }
        ExprData::Func { name, args } => {
            let new_args: Vec<ExprId> = args.iter().map(|&a| fold_predicates(a, pool)).collect();
            pool.func(name, new_args)
        }
        _ => expr,
    }
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
        let expr = p.add(vec![xpy, p.integer(1_i32)]);
        let result = subs(expr, &m, &p);
        // (x+y) + 1 → z + 1
        assert_eq!(result, p.add(vec![z, p.integer(1_i32)]));
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
}
