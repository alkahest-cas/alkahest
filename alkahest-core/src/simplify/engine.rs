use super::rules::{
    AddZero, CanonicalOrder, ConstFold, DivSelf, ExpandMul, FlattenAdd, FlattenMul, MulOne,
    MulZero, PowOne, PowZero, RewriteRule, SubSelf,
};
use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::kernel::{ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls how many full bottom-up passes the simplifier may perform.
#[derive(Debug, Clone)]
pub struct SimplifyConfig {
    /// Maximum number of full bottom-up passes (default 100).
    pub max_iterations: usize,
    /// Whether to distribute multiplication over addition (default false).
    ///
    /// When `true`, the `ExpandMul` rule is included: `(a + b) * c → a*c + b*c`.
    /// Keep disabled unless explicitly expanding, because expansion can loop
    /// against a future `factor` rule.
    pub expand: bool,
    /// Allow branch-cut-sensitive rewrites such as `log(a*b) → log(a) + log(b)`.
    ///
    /// This identity only holds when `a` and `b` are positive reals.  Set this
    /// flag to `true` when you know all variables are positive and want the
    /// full log/exp rule set; leave it `false` (the default) for safe behaviour
    /// over complex numbers or when sign information is unavailable.
    pub allow_branch_cut_rewrites: bool,
}

impl Default for SimplifyConfig {
    fn default() -> Self {
        SimplifyConfig {
            max_iterations: 100,
            expand: false,
            allow_branch_cut_rewrites: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Default rule set
// ---------------------------------------------------------------------------

/// Build the rule set for a given config.
pub fn rules_for_config(config: &SimplifyConfig) -> Vec<Box<dyn RewriteRule>> {
    let mut rules: Vec<Box<dyn RewriteRule>> = vec![
        Box::new(FlattenMul),
        Box::new(FlattenAdd),
        Box::new(MulZero),
        Box::new(AddZero),
        Box::new(MulOne),
        Box::new(PowZero),
        Box::new(PowOne),
        Box::new(ConstFold),
        Box::new(SubSelf),
        Box::new(DivSelf),
        Box::new(CanonicalOrder),
    ];
    if config.expand {
        rules.push(Box::new(ExpandMul));
    }
    rules
}

pub fn default_rules() -> Vec<Box<dyn RewriteRule>> {
    rules_for_config(&SimplifyConfig::default())
}

// ---------------------------------------------------------------------------
// Internal: bottom-up traversal — simplify children, then current node
// ---------------------------------------------------------------------------

fn simplify_node(
    expr: ExprId,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
) -> DerivedExpr<ExprId> {
    // 1. Rebuild with simplified children
    let data = pool.get(expr);
    let (rebuilt, child_log) = simplify_children(data, pool, rules);

    // 2. Apply rules to rebuilt node until no rule fires
    let mut current = rebuilt;
    let mut rule_log = DerivationLog::new();
    loop {
        let mut fired = false;
        for rule in rules {
            if let Some((new_expr, step_log)) = rule.apply(current, pool) {
                rule_log = rule_log.merge(step_log);
                current = new_expr;
                fired = true;
                break; // restart from first rule after any change
            }
        }
        if !fired {
            break;
        }
    }

    DerivedExpr::with_log(current, child_log.merge(rule_log))
}

/// Simplify children of a node and return (rebuilt_expr, child_log).
fn simplify_children(
    data: ExprData,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
) -> (ExprId, DerivationLog) {
    let mut log = DerivationLog::new();
    match data {
        ExprData::Add(args) => {
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.add(new_args), log)
        }
        ExprData::Mul(args) => {
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.mul(new_args), log)
        }
        ExprData::Pow { base, exp } => {
            let rb = simplify_node(base, pool, rules);
            log = log.merge(rb.log);
            let re = simplify_node(exp, pool, rules);
            log = log.merge(re.log);
            (pool.pow(rb.value, re.value), log)
        }
        ExprData::Func { name, args } => {
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.func(name, new_args), log)
        }
        // PA-9: Simplify values in each branch and the default.
        // The condition expressions (predicates) are passed through unchanged
        // since there are no simplification rules for predicates yet.
        ExprData::Piecewise { branches, default } => {
            let new_branches: Vec<(ExprId, ExprId)> = branches
                .into_iter()
                .map(|(cond, val)| {
                    let rv = simplify_node(val, pool, rules);
                    log = std::mem::take(&mut log).merge(rv.log);
                    (cond, rv.value)
                })
                .collect();
            let rd = simplify_node(default, pool, rules);
            log = log.merge(rd.log);
            (pool.piecewise(new_branches, rd.value), log)
        }
        // Predicate args may be simplified as expressions.
        ExprData::Predicate { kind, args } => {
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.predicate(kind, new_args), log)
        }
        ExprData::Forall { var, body } => {
            let rb = simplify_node(body, pool, rules);
            log = log.merge(rb.log);
            (pool.forall(var, rb.value), log)
        }
        ExprData::Exists { var, body } => {
            let rb = simplify_node(body, pool, rules);
            log = log.merge(rb.log);
            (pool.exists(var, rb.value), log)
        }
        ExprData::BigO(arg) => {
            let r = simplify_node(arg, pool, rules);
            log = log.merge(r.log);
            (pool.big_o(r.value), log)
        }
        // Atoms have no children
        atom => (pool.intern(atom), log),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Simplify `expr` with a custom rule set and config.
pub fn simplify_with(
    expr: ExprId,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
    config: SimplifyConfig,
) -> DerivedExpr<ExprId> {
    let mut current = DerivedExpr::new(expr);
    for _ in 0..config.max_iterations {
        let result = simplify_node(current.value, pool, rules);
        let merged_log = current.log.merge(result.log);
        if result.value == current.value {
            return DerivedExpr::with_log(current.value, merged_log);
        }
        current = DerivedExpr::with_log(result.value, merged_log);
    }
    current
}

/// Simplify `expr` with the default rule set.
pub fn simplify(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    let config = SimplifyConfig::default();
    simplify_with(expr, pool, &rules_for_config(&config), config)
}

/// Simplify `expr` with expansion enabled (`(a+b)*c → a*c + b*c`).
pub fn simplify_expanded(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    let config = SimplifyConfig {
        expand: true,
        ..SimplifyConfig::default()
    };
    simplify_with(expr, pool, &rules_for_config(&config), config)
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
    fn simplify_x_plus_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(0_i32)]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, x);
        assert!(!r.log.is_empty(), "should have logged a step");
        assert!(
            r.log.steps().iter().any(|s| s.rule_name == "add_zero"),
            "log should mention add_zero"
        );
    }

    #[test]
    fn simplify_x_times_one() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.integer(1_i32)]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, x);
    }

    #[test]
    fn simplify_x_times_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.integer(0_i32)]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn simplify_x_pow_one() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(1_i32));
        let r = simplify(expr, &pool);
        assert_eq!(r.value, x);
    }

    #[test]
    fn simplify_x_pow_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(0_i32));
        let r = simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(
            r.log.steps().iter().any(|s| !s.side_conditions.is_empty()),
            "pow_zero should record side condition"
        );
    }

    #[test]
    fn simplify_const_fold_add() {
        let pool = p();
        let expr = pool.add(vec![pool.integer(2_i32), pool.integer(3_i32)]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(5_i32));
    }

    #[test]
    fn simplify_const_fold_mul() {
        let pool = p();
        let expr = pool.mul(vec![pool.integer(4_i32), pool.integer(5_i32)]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(20_i32));
    }

    #[test]
    fn simplify_const_fold_pow() {
        let pool = p();
        let expr = pool.pow(pool.integer(2_i32), pool.integer(10_i32));
        let r = simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(1024_i32));
    }

    #[test]
    fn simplify_sub_self() {
        // x + (-1)*x → 0
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.add(vec![x, neg_x]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn simplify_div_self() {
        // x * x^(-1) → 1
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x_inv = pool.pow(x, pool.integer(-1_i32));
        let expr = pool.mul(vec![x, x_inv]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(1_i32));
    }

    #[test]
    fn simplify_nested() {
        // (x + 0) * 1 → x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inner = pool.add(vec![x, pool.integer(0_i32)]);
        let expr = pool.mul(vec![inner, pool.integer(1_i32)]);
        let r = simplify(expr, &pool);
        assert_eq!(r.value, x);
    }

    #[test]
    fn simplify_idempotent_on_already_simple() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = simplify(x, &pool);
        assert_eq!(r.value, x);
        assert!(r.log.is_empty());
    }

    #[test]
    fn simplify_with_custom_config() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(0_i32)]);
        let config = SimplifyConfig {
            max_iterations: 1,
            ..SimplifyConfig::default()
        };
        let r = simplify_with(expr, &pool, &default_rules(), config);
        assert_eq!(r.value, x);
    }
}
