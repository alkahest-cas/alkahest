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
/// # use alkahest_core::kernel::{Domain, ExprPool};
/// # use alkahest_core::kernel::subs::subs;
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
        // Atoms have no children — if not in mapping, return as-is
        _ => expr,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

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
}
