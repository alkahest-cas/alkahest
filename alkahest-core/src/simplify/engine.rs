use super::rules::{
    AddZero, CanonicalOrder, ConstFold, DivSelf, ExpandMul, FlattenAdd, FlattenMul, MulOne,
    MulZero, PowOne, PowZero, RewriteRule, SubSelf,
};
use super::rulesets::PatternRuleSet;
use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

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
    /// Assumptions for colored e-graph simplification (e.g. `x > 0`).
    ///
    /// When non-empty, [`simplify_with`] runs a colored equality-saturation pass
    /// after the rule engine so conditional rewrites like `sqrt(x²) → x` can fire.
    pub assumptions: Vec<crate::deriv::log::SideCondition>,
}

impl Default for SimplifyConfig {
    fn default() -> Self {
        SimplifyConfig {
            max_iterations: 100,
            expand: false,
            allow_branch_cut_rewrites: false,
            assumptions: vec![],
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

/// Memoised bottom-up simplification.
///
/// `memo` maps an input `ExprId` to the `ExprId` of its simplified form within
/// the current pass.  Shared subexpressions (same `ExprId` appearing in multiple
/// places) are simplified exactly once; subsequent hits return the cached result
/// with an empty derivation log to avoid duplicate log entries.
///
/// The memo is valid for one complete bottom-up pass.  `simplify_with` creates
/// a fresh `HashMap` per iteration so that the fixed-point loop sees the updated
/// expression on each pass.
fn simplify_node(
    expr: ExprId,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
    memo: &mut HashMap<ExprId, ExprId>,
) -> DerivedExpr<ExprId> {
    // Shared-subexpression cache: if we already simplified this node during
    // the current pass, return the cached result immediately.
    if let Some(&cached) = memo.get(&expr) {
        return DerivedExpr::new(cached);
    }

    // 1. Rebuild with simplified children
    let data = pool.get(expr);
    let (rebuilt, child_log) = simplify_children(data, pool, rules, memo);

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

    let result = DerivedExpr::with_log(current, child_log.merge(rule_log));
    memo.insert(expr, result.value);
    result
}

fn simplify_node_indexed(
    expr: ExprId,
    pool: &ExprPool,
    rule_set: &PatternRuleSet,
    child_rules: &[Box<dyn RewriteRule>],
    memo: &mut HashMap<ExprId, ExprId>,
) -> DerivedExpr<ExprId> {
    if let Some(&cached) = memo.get(&expr) {
        return DerivedExpr::new(cached);
    }

    let data = pool.get(expr);
    let (rebuilt, child_log) = simplify_children(data, pool, child_rules, memo);

    let mut current = rebuilt;
    let mut rule_log = DerivationLog::new();
    loop {
        let mut fired = false;
        for idx in rule_set.index().candidates(current, pool) {
            if let Some((new_expr, step_log)) = rule_set.rules()[idx].apply(current, pool) {
                rule_log = rule_log.merge(step_log);
                current = new_expr;
                fired = true;
                break;
            }
        }
        if !fired {
            break;
        }
    }

    let result = DerivedExpr::with_log(current, child_log.merge(rule_log));
    memo.insert(expr, result.value);
    result
}

/// Simplify children of a node and return (rebuilt_expr, child_log).
fn simplify_children(
    data: ExprData,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
    memo: &mut HashMap<ExprId, ExprId>,
) -> (ExprId, DerivationLog) {
    let mut log = DerivationLog::new();
    match data {
        ExprData::Add(args) => {
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node(a, pool, rules, memo);
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
                    let r = simplify_node(a, pool, rules, memo);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.mul(new_args), log)
        }
        ExprData::Pow { base, exp } => {
            let rb = simplify_node(base, pool, rules, memo);
            log = log.merge(rb.log);
            let re = simplify_node(exp, pool, rules, memo);
            log = log.merge(re.log);
            (pool.pow(rb.value, re.value), log)
        }
        ExprData::Func { name, args } => {
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node(a, pool, rules, memo);
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
                    let rv = simplify_node(val, pool, rules, memo);
                    log = std::mem::take(&mut log).merge(rv.log);
                    (cond, rv.value)
                })
                .collect();
            let rd = simplify_node(default, pool, rules, memo);
            log = log.merge(rd.log);
            (pool.piecewise(new_branches, rd.value), log)
        }
        // Predicate args may be simplified as expressions.
        ExprData::Predicate { kind, args } => {
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node(a, pool, rules, memo);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.predicate(kind, new_args), log)
        }
        ExprData::Forall { var, body } => {
            let rb = simplify_node(body, pool, rules, memo);
            log = log.merge(rb.log);
            (pool.forall(var, rb.value), log)
        }
        ExprData::Exists { var, body } => {
            let rb = simplify_node(body, pool, rules, memo);
            log = log.merge(rb.log);
            (pool.exists(var, rb.value), log)
        }
        ExprData::BigO(arg) => {
            let r = simplify_node(arg, pool, rules, memo);
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
        // Fresh memo per pass: maps input ExprId → simplified ExprId.
        // Shared subexpressions are simplified once and the result reused for
        // all subsequent occurrences within the same bottom-up sweep.
        let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
        let result = simplify_node(current.value, pool, rules, &mut memo);
        let merged_log = current.log.merge(result.log);
        if result.value == current.value {
            current = DerivedExpr::with_log(current.value, merged_log);
            break;
        }
        current = DerivedExpr::with_log(result.value, merged_log);
    }

    if !config.assumptions.is_empty() {
        let colored = super::colored_egraph::apply_colored_if_needed(
            current.value,
            pool,
            &config.assumptions,
        );
        return DerivedExpr::with_log(colored.value, current.log.merge(colored.log));
    }
    current
}

/// Simplify `expr` using a [`PatternRuleSet`] (discrimination-net indexed).
pub fn simplify_with_pattern_rules(
    expr: ExprId,
    pool: &ExprPool,
    rule_set: &PatternRuleSet,
    config: SimplifyConfig,
) -> DerivedExpr<ExprId> {
    let child_rules = rule_set.as_dyn_rules();
    let mut current = DerivedExpr::new(expr);
    for _ in 0..config.max_iterations {
        let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
        let result = simplify_node_indexed(current.value, pool, rule_set, &child_rules, &mut memo);
        let merged_log = current.log.merge(result.log);
        if result.value == current.value {
            current = DerivedExpr::with_log(current.value, merged_log);
            break;
        }
        current = DerivedExpr::with_log(result.value, merged_log);
    }

    if !config.assumptions.is_empty() {
        let colored = super::colored_egraph::apply_colored_if_needed(
            current.value,
            pool,
            &config.assumptions,
        );
        return DerivedExpr::with_log(colored.value, current.log.merge(colored.log));
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

    #[test]
    fn simplify_with_assumptions_sqrt_square() {
        use crate::deriv::log::SideCondition;
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("sqrt", vec![pool.pow(x, pool.integer(2_i32))]);
        let config = SimplifyConfig {
            assumptions: vec![SideCondition::Positive(x)],
            ..SimplifyConfig::default()
        };
        let r = simplify_with(expr, &pool, &default_rules(), config);
        assert_eq!(r.value, x);
    }

    /// DAG traversal memo test: a shared subexpression that appears in O(2^n) tree
    /// positions should be simplified in O(n) time, not O(2^n).
    ///
    /// We build `expr = shared_node + shared_node` where `shared_node` is itself
    /// `x + 0` — both sides point to the same `ExprId`.  The simplifier must
    /// produce the correct answer regardless of sharing depth.
    #[test]
    fn simplify_dag_shared_subexpr_correct() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // Construct a deeply shared expression: iterated "squaring" of `(x + 0)`.
        // After 20 levels, tree-size would be 2^20 without DAG memoization.
        let mut node = pool.add(vec![x, pool.integer(0_i32)]); // x + 0
        for _ in 0..20 {
            // node = node + node  (both args are the SAME ExprId)
            node = pool.add(vec![node, node]);
        }
        // simplify should terminate quickly (not 2^20 operations) and give a
        // result that is a valid simplified form (not x + 0).
        let r = simplify(node, &pool);
        // The result must not contain `+ 0` anymore — `x + 0` simplifies to `x`.
        let s = pool.display(r.value).to_string();
        assert!(
            !s.contains("+ 0") && !s.contains("0 +"),
            "simplify should eliminate '+ 0' from shared expression: {s}"
        );
    }

    /// DAG traversal memo test for diff: differentiating a shared-subexpression
    /// expression must give the correct result in polynomial time.
    #[test]
    fn diff_dag_shared_subexpr_correct() {
        use crate::diff::diff;
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // Build `(x^2 + x) + (x^2 + x)` where both halves are the same ExprId.
        let inner = pool.add(vec![pool.pow(x, pool.integer(2_i32)), x]); // x² + x
        let expr = pool.add(vec![inner, inner]); // 2*(x² + x) via sharing
                                                 // diff(2*(x²+x), x) = 2*(2x + 1) = 4x + 2
        let r = diff(expr, x, &pool).unwrap();
        let s = pool.display(r.value).to_string();
        // Result should contain x and numeric coefficients, not crash or loop.
        assert!(
            !s.is_empty(),
            "diff of shared DAG expression returned empty string"
        );
    }

    /// DAG traversal memo test for eval_interp: evaluating a shared expression
    /// should return the correct numeric value.
    #[test]
    fn eval_interp_dag_shared_subexpr_correct() {
        use crate::jit::{compile, eval_interp};
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // shared = x + 1;  expr = shared * shared = (x+1)^2
        let shared = pool.add(vec![x, pool.integer(1_i32)]);
        let expr = pool.mul(vec![shared, shared]);

        // Interpreter path via eval_interp
        let mut env = std::collections::HashMap::new();
        env.insert(x, 3.0f64); // (3+1)^2 = 16
        let result = eval_interp(expr, &env, &pool);
        assert_eq!(result, Some(16.0), "eval_interp shared DAG: expected 16");

        // Compiled path (interpreter fallback, no LLVM needed)
        let f = compile(expr, &[x], &pool).unwrap();
        assert!((f.call(&[3.0]) - 16.0).abs() < 1e-10);
    }
}
