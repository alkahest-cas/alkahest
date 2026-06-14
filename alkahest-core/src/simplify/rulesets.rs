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
use crate::simplify::discrimination_net::{pattern_head, DiscriminationIndex};
use crate::simplify::rules::{FlattenAdd, FlattenMul, RewriteRule};

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

/// Coefficient-aware Pythagorean identity: `a·sin²(u) + a·cos²(u) → a`.
///
/// Matches `Add([…, c₁·sin²(u), …, c₂·cos²(u), …])` where `u` is any
/// sub-expression appearing identically in both terms and the *coefficients*
/// `c₁`, `c₂` (the remaining multiplicative factors of each term) are
/// structurally equal.  The matched pair is replaced by that common
/// coefficient `a`, so:
///
/// - `sin²(u) + cos²(u) → 1`            (the original bare case, `a = 1`),
/// - `2·sin²(u) + 2·cos²(u) → 2`,
/// - `a·sin²(u) + a·cos²(u) → a`        (symbolic `a`),
/// - `3 + 2·sin²(u) + 2·cos²(u) → 5`    (embedded in a larger sum; the
///   leftover numeric terms are folded so the result is fully reduced even
///   though `trig_rules` carries no general constant-folder).
///
/// The shared coefficient is matched on the canonically sorted factor lists,
/// so factor order is irrelevant.  Only a *single* `sin²`/`cos²` factor per
/// term is considered (terms like `sin²(u)·cos²(u)` are left untouched).
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

        // Find a term `c₁·sin²(u)` …
        let mut sin_pos = None;
        for (i, &a) in args.iter().enumerate() {
            if let Some((u, coeff)) = split_trig_sq("sin", a, pool) {
                sin_pos = Some((i, u, coeff));
                break;
            }
        }
        let (sin_idx, u, sin_coeff) = sin_pos?;

        // … and a matching `c₂·cos²(u)` with the same `u` and the same
        // coefficient factor multiset.
        let mut cos_idx = None;
        for (i, &a) in args.iter().enumerate() {
            if i == sin_idx {
                continue;
            }
            if let Some((cu, cos_coeff)) = split_trig_sq("cos", a, pool) {
                if cu == u && cos_coeff == sin_coeff {
                    cos_idx = Some(i);
                    break;
                }
            }
        }
        let cos_idx = cos_idx?;

        // The shared coefficient `a` (product of the leftover factors; empty → 1).
        let coeff_expr = match sin_coeff.len() {
            0 => pool.integer(1_i32),
            1 => sin_coeff[0],
            _ => pool.mul(sin_coeff.clone()),
        };

        // Replace the matched pair with `a` in the term list, then fold any
        // resulting numeric literals together (e.g. `3 + 2 → 5`).
        let mut new_args: Vec<ExprId> = args
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != sin_idx && i != cos_idx)
            .map(|(_, a)| a)
            .collect();
        new_args.push(coeff_expr);
        new_args = fold_numeric_terms(new_args, pool);

        let after = match new_args.len() {
            0 => pool.integer(0_i32),
            1 => new_args[0],
            _ => pool.add(new_args),
        };

        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Double-angle for sine: `2·sin(u)·cos(u) → sin(2u)`.
///
/// Fires on a `Mul` containing the literal factor `2`, a `sin(u)` and a
/// `cos(u)` with the *same* argument `u`.  Any further factors are preserved,
/// so `k·2·sin(u)·cos(u) → k·sin(2u)`.
pub struct SinDoubleAngle;

impl RewriteRule for SinDoubleAngle {
    fn name(&self) -> &'static str {
        "sin_double_angle"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let factors = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };

        let two_pos = factors
            .iter()
            .position(|&f| pool.with(f, |d| matches!(d, ExprData::Integer(n) if n.0 == 2)))?;
        let sin_pos = factors
            .iter()
            .position(|&f| func_arg("sin", f, pool).is_some())?;
        let u = func_arg("sin", factors[sin_pos], pool).unwrap();
        let cos_pos = factors
            .iter()
            .enumerate()
            .position(|(i, &f)| i != sin_pos && func_arg("cos", f, pool) == Some(u))?;

        if two_pos == sin_pos || two_pos == cos_pos {
            return None;
        }

        let two = pool.integer(2_i32);
        let double_u = pool.mul(vec![two, u]);
        let sin_2u = pool.func("sin", vec![double_u]);

        let mut rest: Vec<ExprId> = factors
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != two_pos && i != sin_pos && i != cos_pos)
            .map(|(_, f)| f)
            .collect();
        rest.push(sin_2u);
        let after = match rest.len() {
            1 => rest[0],
            _ => pool.mul(rest),
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Double-angle for cosine: `cos²(u) − sin²(u) → cos(2u)`.
///
/// Fires on an `Add` containing `cos²(u)` and `(-1)·sin²(u)` with the same
/// argument `u`.  Remaining terms are preserved.
pub struct CosDoubleAngle;

impl RewriteRule for CosDoubleAngle {
    fn name(&self) -> &'static str {
        "cos_double_angle"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // `cos²(u)` term (coefficient must be +1, i.e. no leftover factors).
        let mut cos_hit = None;
        for (i, &a) in args.iter().enumerate() {
            if let Some((u, coeff)) = split_trig_sq("cos", a, pool) {
                if coeff.is_empty() {
                    cos_hit = Some((i, u));
                    break;
                }
            }
        }
        let (cos_idx, u) = cos_hit?;

        // `(-1)·sin²(u)` term.
        let mut sin_idx = None;
        for (i, &a) in args.iter().enumerate() {
            if i == cos_idx {
                continue;
            }
            if let Some((su, coeff)) = split_trig_sq("sin", a, pool) {
                if su == u
                    && coeff.len() == 1
                    && pool.with(coeff[0], |d| matches!(d, ExprData::Integer(n) if n.0 == -1))
                {
                    sin_idx = Some(i);
                    break;
                }
            }
        }
        let sin_idx = sin_idx?;

        let two = pool.integer(2_i32);
        let double_u = pool.mul(vec![two, u]);
        let cos_2u = pool.func("cos", vec![double_u]);

        let mut rest: Vec<ExprId> = args
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != cos_idx && i != sin_idx)
            .map(|(_, a)| a)
            .collect();
        rest.push(cos_2u);
        let after = match rest.len() {
            1 => rest[0],
            _ => pool.add(rest),
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Angle-subtraction for sine: `sin(a)·cos(b) − cos(a)·sin(b) → sin(a−b)`.
///
/// Fires on an `Add` of exactly the two products `sin(a)·cos(b)` and
/// `(-1)·cos(a)·sin(b)` (plus any unrelated terms, which are preserved).
pub struct SinAngleSub;

impl RewriteRule for SinAngleSub {
    fn name(&self) -> &'static str {
        "sin_angle_sub"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // Positive term `sin(a)·cos(b)`.
        for (pi, &pos) in args.iter().enumerate() {
            let Some((a, b)) = match_func_pair("sin", "cos", pos, pool) else {
                continue;
            };
            // Negative term `(-1)·cos(a)·sin(b)`.
            for (ni, &neg) in args.iter().enumerate() {
                if ni == pi {
                    continue;
                }
                let Some(inner) = neg_inner(neg, pool) else {
                    continue;
                };
                if match_func_pair("cos", "sin", inner, pool) != Some((a, b)) {
                    continue;
                }
                let diff = sub(a, b, pool);
                let sin_diff = pool.func("sin", vec![diff]);
                let after = rebuild_add_replacing(&args, pi, ni, sin_diff, pool);
                return Some((after, one_step(self.name(), expr, after)));
            }
        }
        None
    }
}

/// Angle-subtraction for cosine: `cos(a)·cos(b) + sin(a)·sin(b) → cos(a−b)`.
pub struct CosAngleSub;

impl RewriteRule for CosAngleSub {
    fn name(&self) -> &'static str {
        "cos_angle_sub"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        for (ci, &cc) in args.iter().enumerate() {
            let Some((a, b)) = match_func_pair("cos", "cos", cc, pool) else {
                continue;
            };
            for (si, &ss) in args.iter().enumerate() {
                if si == ci {
                    continue;
                }
                // `sin(a)·sin(b)` with the same (a, b) — order-insensitive.
                let Some((sa, sb)) = match_func_pair("sin", "sin", ss, pool) else {
                    continue;
                };
                if !((sa == a && sb == b) || (sa == b && sb == a)) {
                    continue;
                }
                let diff = sub(a, b, pool);
                let cos_diff = pool.func("cos", vec![diff]);
                let after = rebuild_add_replacing(&args, ci, si, cos_diff, pool);
                return Some((after, one_step(self.name(), expr, after)));
            }
        }
        None
    }
}

/// Return all trigonometric identity rules.
///
/// The set leads with the structural normalizers [`FlattenMul`]/[`FlattenAdd`]
/// so the AC-sensitive identity rules below (Pythagorean, double-angle,
/// angle-subtraction) see fully flattened `Add`/`Mul` nodes even when the input
/// arrives as nested binary trees (as it does from the Python surface, which
/// builds `a*b*c` as `a*(b*c)`).  Both normalizers only restructure nested
/// `Add`/`Mul`; they perform no arithmetic and so cannot introduce regressions.
pub fn trig_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![
        Box::new(FlattenMul),
        Box::new(FlattenAdd),
        Box::new(SinNeg),
        Box::new(CosNeg),
        Box::new(TanExpand),
        Box::new(SinCosIdentity),
        Box::new(SinDoubleAngle),
        Box::new(CosDoubleAngle),
        Box::new(SinAngleSub),
        Box::new(CosAngleSub),
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
#[derive(Clone)]
pub struct PatternRule {
    pub lhs: Pattern,
    pub rhs: ExprId,
    name: &'static str,
}

/// Pattern rules plus a discrimination-net index for O(1) head lookup.
pub struct PatternRuleSet {
    rules: Vec<PatternRule>,
    index: DiscriminationIndex,
}

impl PatternRuleSet {
    pub fn new(rules: Vec<PatternRule>, pool: &ExprPool) -> Self {
        let heads = rules.iter().map(|r| pattern_head(r.lhs.root, pool));
        let index = DiscriminationIndex::build(heads);
        PatternRuleSet { rules, index }
    }

    pub fn rules(&self) -> &[PatternRule] {
        &self.rules
    }

    pub fn index(&self) -> &DiscriminationIndex {
        &self.index
    }

    pub fn as_dyn_rules(&self) -> Vec<Box<dyn RewriteRule>> {
        self.rules
            .iter()
            .map(|r| Box::new(r.clone()) as Box<dyn RewriteRule>)
            .collect()
    }
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

/// If `expr` is `Pow(Func(`name`, [arg]), 2)`, return `arg`.
fn trig_sq_inner(name: &str, expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let is_two = pool.with(exp, |d| matches!(d, ExprData::Integer(n) if n.0 == 2));
            if !is_two {
                return None;
            }
            func_arg(name, base, pool)
        }
        _ => None,
    }
}

/// View a single Add-term as a multiset of multiplicative factors.
///
/// A bare (non-`Mul`) term is treated as a one-element factor list; a `Mul`
/// returns its (already canonically sorted) factor vector.  Used to peel a
/// shared coefficient off a `c · sin²(u)` term.
fn factor_list(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(expr) {
        ExprData::Mul(v) => v,
        _ => vec![expr],
    }
}

/// If exactly one factor of `term` is `Pow(`name`(u), 2)`, return
/// `(u, remaining_factors)` where `remaining_factors` is the coefficient.
///
/// `remaining_factors` may be empty (meaning coefficient `1`).
fn split_trig_sq(name: &str, term: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<ExprId>)> {
    let factors = factor_list(term, pool);
    let mut inner = None;
    let mut rest = Vec::with_capacity(factors.len());
    let mut matched = 0usize;
    for &f in &factors {
        if let Some(u) = trig_sq_inner(name, f, pool) {
            if matched == 0 {
                inner = Some(u);
            }
            matched += 1;
            if matched > 1 {
                // Two trig-squared factors in one term — ambiguous; bail.
                return None;
            }
        } else {
            rest.push(f);
        }
    }
    inner.map(|u| (u, rest))
}

/// If `term` is a product `f(a)·g(b)` of exactly two single-argument function
/// applications named `f_name` and `g_name`, return `(a, b)`.
///
/// The product must have exactly those two factors (no extra coefficient).
/// `f` and `g` are matched positionally: the `f_name` factor supplies `a`, the
/// `g_name` factor supplies `b`.  When `f_name == g_name` the two arguments are
/// returned in the canonical factor order of the `Mul`.
fn match_func_pair(
    f_name: &str,
    g_name: &str,
    term: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let factors = match pool.get(term) {
        ExprData::Mul(v) if v.len() == 2 => v,
        _ => return None,
    };
    if f_name == g_name {
        let a = func_arg(f_name, factors[0], pool)?;
        let b = func_arg(g_name, factors[1], pool)?;
        return Some((a, b));
    }
    // Try both orderings since `Mul` is canonically sorted by ExprId.
    if let (Some(a), Some(b)) = (
        func_arg(f_name, factors[0], pool),
        func_arg(g_name, factors[1], pool),
    ) {
        return Some((a, b));
    }
    if let (Some(a), Some(b)) = (
        func_arg(f_name, factors[1], pool),
        func_arg(g_name, factors[0], pool),
    ) {
        return Some((a, b));
    }
    None
}

/// Build `a − b` as `Add([a, (-1)·b])`.
fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let neg_one = pool.integer(-1_i32);
    let neg_b = pool.mul(vec![neg_one, b]);
    pool.add(vec![a, neg_b])
}

/// Rebuild an `Add` from `args`, dropping positions `i` and `j` and appending
/// `replacement`.
fn rebuild_add_replacing(
    args: &[ExprId],
    i: usize,
    j: usize,
    replacement: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let mut rest: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != i && k != j)
        .map(|(_, &a)| a)
        .collect();
    rest.push(replacement);
    match rest.len() {
        1 => rest[0],
        _ => pool.add(rest),
    }
}

/// Numeric value of `expr` as an exact `rug::Rational`, if it is an integer
/// or rational literal.  `Float` is intentionally excluded (folding floats
/// would change the result's exactness).
fn as_exact_rational(expr: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0)),
        ExprData::Rational(r) => Some(r.0),
        _ => None,
    }
}

/// Intern an exact rational, collapsing `n/1` to an `Integer`.
fn intern_rational(value: rug::Rational, pool: &ExprPool) -> ExprId {
    if value.denom() == &rug::Integer::from(1) {
        pool.integer(value.numer().clone())
    } else {
        let (num, den) = value.into_numer_denom();
        pool.rational(num, den)
    }
}

/// Combine any exact numeric literals appearing as top-level terms of an `Add`
/// argument list into a single literal, leaving non-numeric terms untouched.
///
/// Returns the rebuilt term list.  This lets the coefficient-aware Pythagorean
/// rule reduce `3 + 2` (produced after collapsing `2·sin²+2·cos² → 2`) to `5`
/// even though the bare `trig_rules` set has no general constant-folding rule.
fn fold_numeric_terms(terms: Vec<ExprId>, pool: &ExprPool) -> Vec<ExprId> {
    let mut acc: Option<rug::Rational> = None;
    let mut others = Vec::with_capacity(terms.len());
    for t in terms {
        if let Some(r) = as_exact_rational(t, pool) {
            acc = Some(match acc {
                Some(a) => a + r,
                None => r,
            });
        } else {
            others.push(t);
        }
    }
    if let Some(sum) = acc {
        // Drop an exact zero so it does not survive as `… + 0`.
        if sum != rug::Rational::from(0) || others.is_empty() {
            others.push(intern_rational(sum, pool));
        }
    }
    others
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::pattern::Pattern;
    use crate::simplify::engine::{simplify_with, simplify_with_pattern_rules, SimplifyConfig};

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

    /// Helper: `c · sin²(u)` (or any single-arg trig) built as a Mul.
    fn coeff_trig_sq(pool: &ExprPool, coeff: ExprId, fname: &str, u: ExprId) -> ExprId {
        let f = pool.func(fname, vec![u]);
        let sq = pool.pow(f, pool.integer(2_i32));
        pool.mul(vec![coeff, sq])
    }

    #[test]
    fn coeff_pythagorean_two() {
        // 2·sin²(x) + 2·cos²(x) → 2
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let s = coeff_trig_sq(&pool, two, "sin", x);
        let c = coeff_trig_sq(&pool, two, "cos", x);
        let expr = pool.add(vec![s, c]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(
            r.value,
            pool.integer(2_i32),
            "got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn coeff_pythagorean_symbolic_compound_arg() {
        // a·sin²(θ1+θ2) + a·cos²(θ1+θ2) → a   (symbolic a, compound u)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let t1 = pool.symbol("theta1", Domain::Real);
        let t2 = pool.symbol("theta2", Domain::Real);
        let u = pool.add(vec![t1, t2]);
        let s = coeff_trig_sq(&pool, a, "sin", u);
        let c = coeff_trig_sq(&pool, a, "cos", u);
        let expr = pool.add(vec![s, c]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(r.value, a, "got {}", pool.display(r.value));
    }

    #[test]
    fn coeff_pythagorean_embedded_constant_fold() {
        // 3 + 2·sin²(x) + 2·cos²(x) → 5
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let three = pool.integer(3_i32);
        let two = pool.integer(2_i32);
        let s = coeff_trig_sq(&pool, two, "sin", x);
        let c = coeff_trig_sq(&pool, two, "cos", x);
        let expr = pool.add(vec![three, s, c]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        assert_eq!(
            r.value,
            pool.integer(5_i32),
            "got {}",
            pool.display(r.value)
        );
    }

    #[test]
    fn sin_double_angle_fires() {
        // 2·sin(x)·cos(x) → sin(2x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let expr = pool.mul(vec![two, sin_x, cos_x]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let expected = pool.func("sin", vec![two_x]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn cos_double_angle_fires() {
        // cos²(x) − sin²(x) → cos(2x)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cos_sq = pool.pow(pool.func("cos", vec![x]), pool.integer(2_i32));
        let sin_sq = pool.pow(pool.func("sin", vec![x]), pool.integer(2_i32));
        let neg_sin_sq = pool.mul(vec![pool.integer(-1_i32), sin_sq]);
        let expr = pool.add(vec![cos_sq, neg_sin_sq]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let expected = pool.func("cos", vec![two_x]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn sin_angle_sub_fires() {
        // sin(a)·cos(b) − cos(a)·sin(b) → sin(a−b)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let pos = pool.mul(vec![pool.func("sin", vec![a]), pool.func("cos", vec![b])]);
        let neg = pool.mul(vec![
            pool.integer(-1_i32),
            pool.func("cos", vec![a]),
            pool.func("sin", vec![b]),
        ]);
        let expr = pool.add(vec![pos, neg]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let expected = pool.func("sin", vec![diff]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn cos_angle_sub_fires() {
        // cos(a)·cos(b) + sin(a)·sin(b) → cos(a−b)
        let pool = p();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let cc = pool.mul(vec![pool.func("cos", vec![a]), pool.func("cos", vec![b])]);
        let ss = pool.mul(vec![pool.func("sin", vec![a]), pool.func("sin", vec![b])]);
        let expr = pool.add(vec![cc, ss]);
        let r = simplify_with(expr, &pool, &trig_rules(), SimplifyConfig::default());
        let diff = pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])]);
        let expected = pool.func("cos", vec![diff]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
    }

    #[test]
    fn two_link_jacobian_determinant_collapses() {
        // 2-link planar arm Jacobian determinant:
        //   det = l1·l2·[ cos(θ1)·sin(θ1+θ2) − sin(θ1)·cos(θ1+θ2) ]
        //       = l1·l2·sin((θ1+θ2) − θ1) = l1·l2·sin(θ2)
        // We feed the bracket (the angle-difference part) and check it collapses
        // to sin(θ1+θ2 − θ1); the l1·l2 factor is carried verbatim.
        let pool = p();
        let t1 = pool.symbol("theta1", Domain::Real);
        let t2 = pool.symbol("theta2", Domain::Real);
        let sum = pool.add(vec![t1, t2]); // θ1+θ2
                                          // cos(θ1)·sin(θ1+θ2)
        let pos = pool.mul(vec![
            pool.func("cos", vec![t1]),
            pool.func("sin", vec![sum]),
        ]);
        // −sin(θ1)·cos(θ1+θ2)
        let neg = pool.mul(vec![
            pool.integer(-1_i32),
            pool.func("sin", vec![t1]),
            pool.func("cos", vec![sum]),
        ]);
        let bracket = pool.add(vec![pos, neg]);
        let r = simplify_with(bracket, &pool, &trig_rules(), SimplifyConfig::default());
        // sin((θ1+θ2) − θ1) = sin(θ2) up to the flattened, un-cancelled Add the
        // trig ruleset produces: θ1 + θ2 + (-1)·θ1 (no like-term collection in
        // the bare trig set — that is the default simplifier's job).
        let arg = pool.add(vec![t1, t2, pool.mul(vec![pool.integer(-1_i32), t1])]);
        let expected = pool.func("sin", vec![arg]);
        assert_eq!(r.value, expected, "got {}", pool.display(r.value));
        // And under the *default* simplifier the inner sum cancels to sin(θ2).
        let collapsed = crate::simplify::simplify(r.value, &pool);
        let want = pool.func("sin", vec![t2]);
        assert_eq!(
            collapsed.value,
            want,
            "got {}",
            pool.display(collapsed.value)
        );
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
        let rule_set = PatternRuleSet::new(vec![rule], &pool);
        let r = simplify_with_pattern_rules(expr, &pool, &rule_set, SimplifyConfig::default());
        let expected = pool.mul(vec![pool.integer(2_i32), x]);
        assert_eq!(r.value, expected);
    }
}
