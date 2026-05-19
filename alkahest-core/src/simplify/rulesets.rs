/// Opt-in rule bundles for algebraic and transcendental identities.
///
/// These rules are **not** included in the default simplifier; include them via
/// [`simplify_with`](super::engine::simplify_with) when the target domain is known.
///
/// # Example
///
/// ```
/// # use alkahest_cas::kernel::{Domain, ExprPool};
/// # use alkahest_cas::simplify::{simplify_with, SimplifyConfig, rulesets};
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let rules = rulesets::trig_rules();
/// let tan_x = pool.func("tan", vec![x]);
/// let r = simplify_with(tan_x, &pool, &rules, SimplifyConfig::default());
/// // tan(x) → sin(x) * cos(x)^(-1)
/// ```
use crate::deriv::log::{DerivationLog, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::pattern::{Pattern, Substitution};
use crate::simplify::rules::RewriteRule;

fn one_step(name: &'static str, before: ExprId, after: ExprId) -> DerivationLog {
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(name, before, after));
    log
}

// ---------------------------------------------------------------------------
// Trigonometric identity rules
// ---------------------------------------------------------------------------

/// `sin(-x) → -sin(x)` where `-x = (-1)*x`.
pub struct SinNeg;

impl RewriteRule for SinNeg {
    fn name(&self) -> &'static str {
        "sin_neg"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("sin", expr, pool)?;
        let inner = neg_inner(arg, pool)?;
        let after_inner = pool.func("sin", vec![inner]);
        let neg_one = pool.integer(-1_i32);
        let after = pool.mul(vec![neg_one, after_inner]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// `cos(-x) → cos(x)`.
pub struct CosNeg;

impl RewriteRule for CosNeg {
    fn name(&self) -> &'static str {
        "cos_neg"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("cos", expr, pool)?;
        let inner = neg_inner(arg, pool)?;
        let after = pool.func("cos", vec![inner]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// `tan(x) → sin(x) * cos(x)^(-1)`.
pub struct TanExpand;

impl RewriteRule for TanExpand {
    fn name(&self) -> &'static str {
        "tan_expand"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("tan", expr, pool)?;
        let sin_x = pool.func("sin", vec![arg]);
        let cos_x = pool.func("cos", vec![arg]);
        let cos_inv = pool.pow(cos_x, pool.integer(-1_i32));
        let after = pool.mul(vec![sin_x, cos_inv]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// `sin²(x) + cos²(x) → 1`.
///
/// Matches `Add([…, Pow(sin(a), 2), …, Pow(cos(a), 2), …])` where `a` is any
/// sub-expression that appears identically in both.
pub struct SinCosIdentity;

impl RewriteRule for SinCosIdentity {
    fn name(&self) -> &'static str {
        "sin_sq_plus_cos_sq"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // Find a Pow(sin(a), 2) and a matching Pow(cos(a), 2)
        let sin_sq_pos = args.iter().position(|&a| is_sin_sq(a, pool))?;
        let sin_arg = sin_inner(args[sin_sq_pos], pool).unwrap();
        let cos_sq_pos = args.iter().position(|&a| is_cos_sq_of(a, sin_arg, pool))?;

        if sin_sq_pos == cos_sq_pos {
            return None;
        }

        // Replace sin²(a) + cos²(a) with 1 in the arg list
        let one = pool.integer(1_i32);
        let mut new_args: Vec<ExprId> = args
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != sin_sq_pos && i != cos_sq_pos)
            .map(|(_, a)| a)
            .collect();
        new_args.push(one);

        let after = match new_args.len() {
            1 => new_args[0],
            _ => pool.add(new_args),
        };

        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Return all trigonometric identity rules.
pub fn trig_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![
        Box::new(SinNeg),
        Box::new(CosNeg),
        Box::new(TanExpand),
        Box::new(SinCosIdentity),
    ]
}

// ---------------------------------------------------------------------------
// log / exp identity rules
// ---------------------------------------------------------------------------

/// `log(exp(x)) → x`.
pub struct LogOfExp;

impl RewriteRule for LogOfExp {
    fn name(&self) -> &'static str {
        "log_of_exp"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("log", expr, pool)?;
        let inner = func_arg("exp", arg, pool)?;
        Some((inner, one_step(self.name(), expr, inner)))
    }
}

/// `exp(log(x)) → x` (domain: x > 0 assumed).
pub struct ExpOfLog;

impl RewriteRule for ExpOfLog {
    fn name(&self) -> &'static str {
        "exp_of_log"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("exp", expr, pool)?;
        let inner = func_arg("log", arg, pool)?;
        Some((inner, one_step(self.name(), expr, inner)))
    }
}

/// `log(a * b) → log(a) + log(b)`.
///
/// **Branch-cut caveat**: this identity is only valid when all factors are
/// positive reals.  The rule still fires, but each factor is recorded as a
/// [`SideCondition::Positive`] in the derivation log so callers can audit the
/// assumptions made.  Use [`log_exp_rules_safe`] to obtain a rule set that
/// excludes this rule entirely.
pub struct LogOfProduct;

impl RewriteRule for LogOfProduct {
    fn name(&self) -> &'static str {
        "log_of_product"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("log", expr, pool)?;
        let factors = match pool.get(arg) {
            ExprData::Mul(v) if v.len() >= 2 => v,
            _ => return None,
        };
        let logs: Vec<ExprId> = factors.iter().map(|&f| pool.func("log", vec![f])).collect();
        let after = pool.add(logs);
        let conds: Vec<SideCondition> = factors
            .iter()
            .map(|&f| SideCondition::Positive(f))
            .collect();
        let mut log = DerivationLog::new();
        log.push(RewriteStep::with_conditions(
            "log_of_product",
            expr,
            after,
            conds,
        ));
        Some((after, log))
    }
}

/// `log(a^n) → n * log(a)`.
pub struct LogOfPow;

impl RewriteRule for LogOfPow {
    fn name(&self) -> &'static str {
        "log_of_pow"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = func_arg("log", expr, pool)?;
        let (base, exp) = match pool.get(arg) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        let log_base = pool.func("log", vec![base]);
        let after = pool.mul(vec![exp, log_base]);
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Return all log/exp identity rules.
///
/// Includes [`LogOfProduct`] which records [`SideCondition::Positive`] side
/// conditions when it fires.  If you need a fully branch-cut-safe set, use
/// [`log_exp_rules_safe`] instead.
pub fn log_exp_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![
        Box::new(LogOfExp),
        Box::new(ExpOfLog),
        Box::new(LogOfProduct),
        Box::new(LogOfPow),
    ]
}

/// Log/exp rules that are safe for complex numbers (no branch-cut rewrites).
///
/// Excludes [`LogOfProduct`] because `log(a*b) → log(a) + log(b)` is only
/// valid when `a` and `b` are positive reals.
pub fn log_exp_rules_safe() -> Vec<Box<dyn RewriteRule>> {
    vec![Box::new(LogOfExp), Box::new(ExpOfLog), Box::new(LogOfPow)]
}

// ---------------------------------------------------------------------------
// R-5: Pattern-driven user rewrite rules
// ---------------------------------------------------------------------------

/// A rewrite rule specified as a (lhs pattern, rhs template) pair.
///
/// When the rule fires, all wildcards bound by matching `lhs` against the
/// current expression are substituted into `rhs`.
///
/// # Wildcard convention
///
/// Any `Symbol` whose name starts with a lower-case letter is a wildcard.
///
/// # Example
///
/// ```
/// # use alkahest_cas::kernel::{Domain, ExprPool};
/// # use alkahest_cas::simplify::{simplify_with, SimplifyConfig};
/// # use alkahest_cas::simplify::rulesets::PatternRule;
/// # use alkahest_cas::pattern::Pattern;
/// # use alkahest_cas::simplify::rules::RewriteRule;
/// let pool = ExprPool::new();
/// let a = pool.symbol("a", Domain::Real);  // wildcard
/// let b = pool.symbol("b", Domain::Real);  // wildcard
/// // Rule: a*b + a*c → a*(b+c)  (factoring)
/// // lhs pattern: a*b  (simplified, as a*b is the structure)
/// // Here we demonstrate a simpler identity: a + a → 2*a
/// let lhs = pool.add(vec![a, a]);
/// let two_a = pool.mul(vec![pool.integer(2_i32), a]);
/// let rule = PatternRule::new(Pattern::from_expr(lhs), two_a);
/// let x = pool.symbol("x", Domain::Real);
/// let expr = pool.add(vec![x, x]);
/// let r = simplify_with(expr, &pool, &[Box::new(rule)], SimplifyConfig::default());
/// // x + x → 2*x
/// ```
pub struct PatternRule {
    pub lhs: Pattern,
    pub rhs: ExprId,
    name: &'static str,
}

impl PatternRule {
    pub fn new(lhs: Pattern, rhs: ExprId) -> Self {
        PatternRule {
            lhs,
            rhs,
            name: "pattern_rule",
        }
    }

    pub fn named(lhs: Pattern, rhs: ExprId, name: &'static str) -> Self {
        PatternRule { lhs, rhs, name }
    }
}

impl RewriteRule for PatternRule {
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        // Try to match the pattern at the root only (engine does bottom-up traversal)
        let subst = match_at_root(&self.lhs, expr, pool)?;
        let after = subst.apply(self.rhs, pool);
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name, expr, after)))
    }
}

/// Match `pattern` at the root of `expr` (no recursion into children).
fn match_at_root(pattern: &Pattern, expr: ExprId, pool: &ExprPool) -> Option<Substitution> {
    let empty = Substitution {
        bindings: std::collections::HashMap::new(),
    };
    match_root_node(pattern.root, expr, empty, pool)
}

fn match_root_node(
    pat: ExprId,
    expr: ExprId,
    subst: Substitution,
    pool: &ExprPool,
) -> Option<Substitution> {
    use crate::kernel::expr::ExprData as ED;

    enum PN {
        Wildcard(String),
        Integer(i64),
        Symbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Literal,
    }
    enum EN {
        Integer(i64),
        Symbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Other,
    }

    let pn = pool.with(pat, |d| match d {
        ED::Symbol { name, .. } if name.starts_with(|c: char| c.is_lowercase()) => {
            PN::Wildcard(name.clone())
        }
        ED::Symbol { name, .. } => PN::Symbol(name.clone()),
        ED::Integer(n) => PN::Integer(n.0.to_i64().unwrap_or(i64::MIN)),
        ED::Add(v) => PN::Add(v.clone()),
        ED::Mul(v) => PN::Mul(v.clone()),
        ED::Pow { base, exp } => PN::Pow(*base, *exp),
        ED::Func { name, args } => PN::Func(name.clone(), args.clone()),
        _ => PN::Literal,
    });

    let en = pool.with(expr, |d| match d {
        ED::Symbol { name, .. } => EN::Symbol(name.clone()),
        ED::Integer(n) => EN::Integer(n.0.to_i64().unwrap_or(i64::MIN)),
        ED::Add(v) => EN::Add(v.clone()),
        ED::Mul(v) => EN::Mul(v.clone()),
        ED::Pow { base, exp } => EN::Pow(*base, *exp),
        ED::Func { name, args } => EN::Func(name.clone(), args.clone()),
        _ => EN::Other,
    });

    match pn {
        PN::Wildcard(name) => {
            let mut s = subst;
            match s.bindings.get(&name) {
                Some(&existing) if existing != expr => return None,
                _ => {
                    s.bindings.insert(name, expr);
                }
            }
            Some(s)
        }
        PN::Integer(pv) => {
            if matches!(en, EN::Integer(ev) if ev == pv) {
                Some(subst)
            } else {
                None
            }
        }
        PN::Symbol(pname) => {
            if matches!(en, EN::Symbol(ref ename) if *ename == pname) {
                Some(subst)
            } else {
                None
            }
        }
        PN::Add(pargs) => {
            let EN::Add(eargs) = en else { return None };
            match_args_exact(&pargs, &eargs, subst, pool)
        }
        PN::Mul(pargs) => {
            let EN::Mul(eargs) = en else { return None };
            match_args_exact(&pargs, &eargs, subst, pool)
        }
        PN::Pow(pb, pe) => {
            let EN::Pow(eb, ee) = en else { return None };
            let s = match_root_node(pb, eb, subst, pool)?;
            match_root_node(pe, ee, s, pool)
        }
        PN::Func(pname, pargs) => {
            let EN::Func(ename, eargs) = en else {
                return None;
            };
            if pname != ename {
                return None;
            }
            match_args_exact(&pargs, &eargs, subst, pool)
        }
        PN::Literal => {
            if pat == expr {
                Some(subst)
            } else {
                None
            }
        }
    }
}

fn match_args_exact(
    pat_args: &[ExprId],
    expr_args: &[ExprId],
    subst: Substitution,
    pool: &ExprPool,
) -> Option<Substitution> {
    if pat_args.len() != expr_args.len() {
        return None;
    }
    let mut s = subst;
    for (&p, &e) in pat_args.iter().zip(expr_args.iter()) {
        s = match_root_node(p, e, s, pool)?;
    }
    Some(s)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn func_arg(name: &str, expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    pool.with(expr, |data| match data {
        ExprData::Func { name: n, args } if n == name && args.len() == 1 => Some(args[0]),
        _ => None,
    })
}

/// If `expr` is `(-1) * inner` or `inner * (-1)`, return `inner`.
fn neg_inner(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let args = match pool.get(expr) {
        ExprData::Mul(v) => v,
        _ => return None,
    };
    let neg1_pos = args
        .iter()
        .position(|&a| pool.with(a, |d| matches!(d, ExprData::Integer(n) if n.0 == -1)))?;
    let others: Vec<ExprId> = args
        .into_iter()
        .enumerate()
        .filter(|&(i, _)| i != neg1_pos)
        .map(|(_, a)| a)
        .collect();
    Some(match others.len() {
        0 => pool.integer(1_i32),
        1 => others[0],
        _ => pool.mul(others),
    })
}

fn is_sin_sq(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let is_two = pool.with(exp, |d| matches!(d, ExprData::Integer(n) if n.0 == 2));
            let is_sin = pool.with(
                base,
                |d| matches!(d, ExprData::Func { name, .. } if name == "sin"),
            );
            is_two && is_sin
        }
        _ => false,
    }
}

fn sin_inner(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(expr) {
        ExprData::Pow { base, .. } => func_arg("sin", base, pool),
        _ => None,
    }
}

fn is_cos_sq_of(expr: ExprId, arg: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let is_two = pool.with(exp, |d| matches!(d, ExprData::Integer(n) if n.0 == 2));
            let is_cos_of_arg = func_arg("cos", base, pool).is_some_and(|a| a == arg);
            is_two && is_cos_of_arg
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::pattern::Pattern;
    use crate::simplify::engine::{simplify_with, SimplifyConfig};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn sin_neg_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.func("sin", vec![neg_x]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        // sin(-x) → -sin(x)
        let expected = pool.mul(vec![pool.integer(-1_i32), pool.func("sin", vec![x])]);
        assert_eq!(r.value, expected);
    }

    #[test]
    fn cos_neg_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.func("cos", vec![neg_x]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, pool.func("cos", vec![x]));
    }

    #[test]
    fn tan_expand_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("tan", vec![x]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let cos_inv = pool.pow(cos_x, pool.integer(-1_i32));
        let expected = pool.mul(vec![sin_x, cos_inv]);
        assert_eq!(r.value, expected);
    }

    #[test]
    fn sin_cos_identity_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let two = pool.integer(2_i32);
        let sin_sq = pool.pow(sin_x, two);
        let cos_sq = pool.pow(cos_x, two);
        let expr = pool.add(vec![sin_sq, cos_sq]);
        let rules = trig_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, pool.integer(1_i32));
    }

    #[test]
    fn log_of_exp_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("log", vec![pool.func("exp", vec![x])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, x);
    }

    #[test]
    fn exp_of_log_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(r.value, x);
    }

    #[test]
    fn log_of_product_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.func("log", vec![pool.mul(vec![x, y])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        let log_x = pool.func("log", vec![x]);
        let log_y = pool.func("log", vec![y]);
        let expected = pool.add(vec![log_x, log_y]);
        assert_eq!(r.value, expected);
    }

    #[test]
    fn log_of_product_records_positive_side_conditions() {
        // LogOfProduct should record Positive(x) and Positive(y) as side conditions.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.func("log", vec![pool.mul(vec![x, y])]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        let has_positive_conds = r.log.steps().iter().any(|s| {
            s.rule_name == "log_of_product"
                && s.side_conditions
                    .iter()
                    .any(|c| matches!(c, SideCondition::Positive(_)))
        });
        assert!(
            has_positive_conds,
            "log_of_product should record Positive side conditions"
        );
    }

    #[test]
    fn log_of_product_safe_does_not_fire() {
        // log_exp_rules_safe() excludes LogOfProduct — log(x*y) should not expand.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.func("log", vec![pool.mul(vec![x, y])]);
        let rules = log_exp_rules_safe();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        assert_eq!(
            r.value, expr,
            "log(x*y) should NOT be split with log_exp_rules_safe"
        );
    }

    #[test]
    fn log_of_pow_fires() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let n = pool.integer(3_i32);
        let expr = pool.func("log", vec![pool.pow(x, n)]);
        let rules = log_exp_rules();
        let r = simplify_with(expr, &pool, &rules, SimplifyConfig::default());
        let log_x = pool.func("log", vec![x]);
        let expected = pool.mul(vec![n, log_x]);
        assert_eq!(r.value, expected);
    }

    #[test]
    fn pattern_rule_simple() {
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let lhs = pool.add(vec![a, a]);
        let rhs = pool.mul(vec![pool.integer(2_i32), a]);
        let rule = PatternRule::new(Pattern::from_expr(lhs), rhs);
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, x]);
        let r = simplify_with(expr, &pool, &[Box::new(rule)], SimplifyConfig::default());
        let expected = pool.mul(vec![pool.integer(2_i32), x]);
        assert_eq!(r.value, expected);
    }
}
