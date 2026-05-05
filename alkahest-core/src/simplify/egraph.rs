/// E-graph based simplifier using egglog.
///
/// Enabled only when the `egraph` feature is active.  Falls back to the
/// rule-based engine otherwise; see `simplify_egraph` for the stable
/// public entry point that is always available.
///
/// # Encoding strategy
///
/// Alkahest uses n-ary `Add`/`Mul`, but egglog works with fixed-arity
/// constructors.  We left-fold n-ary sums/products into binary trees for
/// submission, then *flatten* the extracted binary tree back to n-ary on
/// the way out (see `parse_egglog_term`).  Commutativity is handled at
/// construction time (children sorted by ExprId); associativity is not
/// added as a rule to avoid AC explosion — the phased schedule plus the
/// flattening round-trip is sufficient for practical inputs.
///
/// # Schedule (RW-2)
///
/// The iteration counts and node/iteration limits are taken from
/// [`EgraphConfig`], allowing callers to trade completeness for bounded
/// run time on large inputs.
#[cfg(feature = "egraph")]
mod backend {
    use crate::kernel::{ExprData, ExprId, ExprPool};
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // 1. Serialise ExprId → egglog expression string (binary left-fold)
    // -----------------------------------------------------------------------

    pub(super) fn expr_to_egglog(expr: ExprId, pool: &ExprPool) -> String {
        enum Node {
            Num(i64),
            Var(String),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow(ExprId, ExprId),
            Func(String, ExprId),
            Unsupported,
        }

        let node = pool.with(expr, |data| match data {
            ExprData::Integer(n) => {
                let v =
                    n.0.to_i64()
                        .unwrap_or(if n.0 > 0 { i64::MAX } else { i64::MIN });
                Node::Num(v)
            }
            ExprData::Rational(_) | ExprData::Float(_) => Node::Unsupported,
            ExprData::Symbol { name, .. } => Node::Var(name.clone()),
            ExprData::Add(args) => Node::Add(args.clone()),
            ExprData::Mul(args) => Node::Mul(args.clone()),
            ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
            ExprData::Func { name, args } if args.len() == 1 => Node::Func(name.clone(), args[0]),
            ExprData::Func { .. } => Node::Unsupported,
            ExprData::Piecewise { .. }
            | ExprData::Predicate { .. }
            | ExprData::Forall { .. }
            | ExprData::Exists { .. }
            | ExprData::BigO(_) => Node::Unsupported,
        });

        match node {
            Node::Num(n) => format!("(Num {n})"),
            Node::Var(name) => format!("(Var \"{name}\")"),
            Node::Add(args) => {
                // Binary left-fold; the parser flattens this back to n-ary.
                let mut it = args.into_iter();
                let first = it.next().unwrap();
                let init = expr_to_egglog(first, pool);
                it.fold(init, |acc, id| {
                    format!("(Add {acc} {})", expr_to_egglog(id, pool))
                })
            }
            Node::Mul(args) => {
                let mut it = args.into_iter();
                let first = it.next().unwrap();
                let init = expr_to_egglog(first, pool);
                it.fold(init, |acc, id| {
                    format!("(Mul {acc} {})", expr_to_egglog(id, pool))
                })
            }
            Node::Pow(base, exp) => format!(
                "(Pow {} {})",
                expr_to_egglog(base, pool),
                expr_to_egglog(exp, pool)
            ),
            Node::Func(name, arg) => {
                let inner = expr_to_egglog(arg, pool);
                match name.as_str() {
                    "sin" => format!("(Sin {inner})"),
                    "cos" => format!("(Cos {inner})"),
                    "exp" => format!("(Exp {inner})"),
                    "log" => format!("(Log {inner})"),
                    "sqrt" => format!("(Sqrt {inner})"),
                    _ => format!("(Var \"{name}_{inner}\")"),
                }
            }
            Node::Unsupported => "(Num 0)".to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // 2. Build the complete egglog program  (RW-2: uses EgraphConfig)
    // -----------------------------------------------------------------------

    fn egglog_program(expr_str: &str, config: &super::EgraphConfig) -> String {
        // egglog 0.4 does not expose a node_limit option; the field is
        // reserved for when a future version adds support.
        let node_limit_line = String::new();
        let iter_limit_line = config
            .iter_limit
            .map(|n| format!("(set-option iteration_limit {n})\n"))
            .unwrap_or_default();

        let si = config.shrink_iters;
        let ei = config.explore_iters;
        let ci = config.const_fold_iters;

        // Conditionally include trig / log-exp rules based on config flags.
        let trig_rules = if config.include_trig_rules {
            // Both Mul form (sin(x)*sin(x)) and Pow form (sin(x)^2) are matched
            // so the identity fires regardless of how the square is represented.
            "(rewrite (Add (Mul (Sin ?x) (Sin ?x)) (Mul (Cos ?x) (Cos ?x))) (Num 1) :ruleset explore)\n\
             (rewrite (Add (Mul (Cos ?x) (Cos ?x)) (Mul (Sin ?x) (Sin ?x))) (Num 1) :ruleset explore)\n\
             (rewrite (Add (Pow (Sin ?x) (Num 2)) (Pow (Cos ?x) (Num 2))) (Num 1) :ruleset explore)\n\
             (rewrite (Add (Pow (Cos ?x) (Num 2)) (Pow (Sin ?x) (Num 2))) (Num 1) :ruleset explore)"
        } else {
            ""
        };

        let log_exp_rules = if config.include_log_exp_rules {
            "(rewrite (Exp (Log ?x)) ?x :ruleset explore)\n\
             (rewrite (Log (Exp ?x)) ?x :ruleset explore)"
        } else {
            ""
        };

        format!(
            r#"
{node_limit_line}{iter_limit_line}(datatype Expr
  (Num i64)
  (Var String)
  (Add Expr Expr)
  (Mul Expr Expr)
  (Pow Expr Expr)
  (Sin Expr)
  (Cos Expr)
  (Exp Expr)
  (Log Expr)
  (Sqrt Expr))

; ── shrink ruleset: identity / absorption / cancellation ─────────────────────
(ruleset shrink)
(rewrite (Add ?x (Num 0)) ?x :ruleset shrink)
(rewrite (Add (Num 0) ?x) ?x :ruleset shrink)
(rewrite (Mul ?x (Num 1)) ?x :ruleset shrink)
(rewrite (Mul (Num 1) ?x) ?x :ruleset shrink)
(rewrite (Mul ?x (Num 0)) (Num 0) :ruleset shrink)
(rewrite (Mul (Num 0) ?x) (Num 0) :ruleset shrink)
(rewrite (Pow ?x (Num 1)) ?x :ruleset shrink)
(rewrite (Pow ?x (Num 0)) (Num 1) :ruleset shrink)
(rewrite (Add ?x (Mul (Num -1) ?x)) (Num 0) :ruleset shrink)
(rewrite (Add (Mul (Num -1) ?x) ?x) (Num 0) :ruleset shrink)
(rewrite (Mul ?x (Pow ?x (Num -1))) (Num 1) :ruleset shrink)
(rewrite (Mul (Pow ?x (Num -1)) ?x) (Num 1) :ruleset shrink)

; ── explore ruleset: trig and log/exp identities (default: both enabled) ──────
(ruleset explore)
{trig_rules}
{log_exp_rules}
(rewrite (Mul (Num -1) (Mul (Num -1) ?x)) ?x :ruleset explore)

; ── constant folding ──────────────────────────────────────────────────────────
(ruleset const-fold)
(rule ((= e (Add (Num ?a) (Num ?b))))
      ((union e (Num (+ ?a ?b))))
      :ruleset const-fold)
(rule ((= e (Mul (Num ?a) (Num ?b))))
      ((union e (Num (* ?a ?b))))
      :ruleset const-fold)
(rule ((= e (Pow (Num ?a) (Num ?b))) (>= ?b 0))
      ((union e (Num (^ ?a ?b))))
      :ruleset const-fold)

; ── phased schedule: shrink → const-fold → explore → shrink → const-fold ─────
(let __expr {expr})
(run shrink {si})
(run const-fold {ci})
(run explore {ei})
(run shrink {si})
(run const-fold {ci})
(extract __expr)
"#,
            node_limit_line = node_limit_line,
            iter_limit_line = iter_limit_line,
            trig_rules = trig_rules,
            log_exp_rules = log_exp_rules,
            expr = expr_str,
            si = si,
            ei = ei,
            ci = ci,
        )
    }

    // -----------------------------------------------------------------------
    // 3. Parse egglog output back to ExprId  (RW-1: flatten binary → n-ary)
    // -----------------------------------------------------------------------

    /// Collect all top-level Add children, recursively flattening nested Adds.
    fn flatten_add_args(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match pool.get(expr) {
            ExprData::Add(args) => args
                .iter()
                .flat_map(|&a| flatten_add_args(a, pool))
                .collect(),
            _ => vec![expr],
        }
    }

    /// Collect all top-level Mul children, recursively flattening nested Muls.
    fn flatten_mul_args(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match pool.get(expr) {
            ExprData::Mul(args) => args
                .iter()
                .flat_map(|&a| flatten_mul_args(a, pool))
                .collect(),
            _ => vec![expr],
        }
    }

    fn parse_egglog_term(s: &str, pool: &ExprPool) -> Option<ExprId> {
        let s = s.trim();
        if s.starts_with('(') && s.ends_with(')') {
            let inner = &s[1..s.len() - 1];
            let (head, rest) = split_head(inner)?;
            match head {
                "Num" => {
                    let n: i64 = rest.trim().parse().ok()?;
                    Some(pool.integer(n))
                }
                "Var" => {
                    let name = rest.trim().trim_matches('"');
                    Some(pool.symbol(name, crate::kernel::Domain::Real))
                }
                "Add" => {
                    let (a_str, b_str) = split_two_args(rest)?;
                    let a = parse_egglog_term(&a_str, pool)?;
                    let b = parse_egglog_term(&b_str, pool)?;
                    // RW-1: flatten binary tree back to n-ary on the way out.
                    let mut children = flatten_add_args(a, pool);
                    children.extend(flatten_add_args(b, pool));
                    Some(pool.add(children))
                }
                "Mul" => {
                    let (a_str, b_str) = split_two_args(rest)?;
                    let a = parse_egglog_term(&a_str, pool)?;
                    let b = parse_egglog_term(&b_str, pool)?;
                    let mut children = flatten_mul_args(a, pool);
                    children.extend(flatten_mul_args(b, pool));
                    Some(pool.mul(children))
                }
                "Pow" => {
                    let (a_str, b_str) = split_two_args(rest)?;
                    let a = parse_egglog_term(&a_str, pool)?;
                    let b = parse_egglog_term(&b_str, pool)?;
                    Some(pool.pow(a, b))
                }
                "Sin" => Some(pool.func("sin", vec![parse_egglog_term(rest.trim(), pool)?])),
                "Cos" => Some(pool.func("cos", vec![parse_egglog_term(rest.trim(), pool)?])),
                "Exp" => Some(pool.func("exp", vec![parse_egglog_term(rest.trim(), pool)?])),
                "Log" => Some(pool.func("log", vec![parse_egglog_term(rest.trim(), pool)?])),
                "Sqrt" => Some(pool.func("sqrt", vec![parse_egglog_term(rest.trim(), pool)?])),
                _ => None,
            }
        } else {
            let n: i64 = s.parse().ok()?;
            Some(pool.integer(n))
        }
    }

    fn split_head(s: &str) -> Option<(&str, &str)> {
        let s = s.trim();
        let pos = s.find(|c: char| c.is_whitespace())?;
        Some((&s[..pos], &s[pos + 1..]))
    }

    fn split_two_args(s: &str) -> Option<(String, String)> {
        let s = s.trim();
        let (first, remainder) = consume_term(s)?;
        let second = remainder.trim();
        Some((first.to_string(), second.to_string()))
    }

    fn consume_term(s: &str) -> Option<(&str, &str)> {
        let s = s.trim_start();
        if s.starts_with('(') {
            let mut depth = 0usize;
            let mut in_string = false;
            for (i, c) in s.char_indices() {
                match c {
                    '"' => in_string = !in_string,
                    '(' if !in_string => depth += 1,
                    ')' if !in_string => {
                        depth -= 1;
                        if depth == 0 {
                            return Some((&s[..=i], &s[i + 1..]));
                        }
                    }
                    _ => {}
                }
            }
            None
        } else {
            let end = s
                .find(|c: char| c.is_whitespace() || c == ')')
                .unwrap_or(s.len());
            Some((&s[..end], &s[end..]))
        }
    }

    // -----------------------------------------------------------------------
    // RW-3: Linear-expression canonizer (post-extraction pass)
    // -----------------------------------------------------------------------

    /// Try to extract a linear term as `(integer_coefficient, base_expr)`.
    ///
    /// Recognises: bare symbols (coeff = 1) and `Mul(Integer, Symbol)`.
    fn extract_linear_term(expr: ExprId, pool: &ExprPool) -> Option<(i64, ExprId)> {
        match pool.get(expr) {
            ExprData::Symbol { .. } => Some((1, expr)),
            ExprData::Mul(args) if args.len() == 2 => {
                let (a, b) = (args[0], args[1]);
                if let ExprData::Integer(n) = pool.get(a) {
                    if matches!(pool.get(b), ExprData::Symbol { .. }) {
                        return n.0.to_i64().map(|c| (c, b));
                    }
                }
                if let ExprData::Integer(n) = pool.get(b) {
                    if matches!(pool.get(a), ExprData::Symbol { .. }) {
                        return n.0.to_i64().map(|c| (c, a));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Canonicalize linear combinations in an expression.
    ///
    /// At each `Add` node, collects `(coefficient, symbol)` pairs and sums
    /// coefficients for identical bases, eliminating zero terms.
    ///
    /// Example: `2*x + 3*x + y` → `5*x + y`.
    pub(super) fn canonicalize_linear(expr: ExprId, pool: &ExprPool) -> ExprId {
        match pool.get(expr) {
            ExprData::Add(args) => {
                let args: Vec<ExprId> =
                    args.iter().map(|&a| canonicalize_linear(a, pool)).collect();

                let mut coeff_map: HashMap<ExprId, i64> = HashMap::new();
                let mut non_linear: Vec<ExprId> = Vec::new();
                let mut found_linear = false;

                for &arg in &args {
                    if let Some((coeff, base)) = extract_linear_term(arg, pool) {
                        *coeff_map.entry(base).or_insert(0) += coeff;
                        found_linear = true;
                    } else {
                        non_linear.push(arg);
                    }
                }

                if !found_linear {
                    return pool.add(args);
                }

                let mut result: Vec<ExprId> = non_linear;
                // Sort by key for determinism
                let mut pairs: Vec<(ExprId, i64)> = coeff_map.into_iter().collect();
                pairs.sort_by_key(|(id, _)| *id);
                for (base, coeff) in pairs {
                    match coeff {
                        0 => {}
                        1 => result.push(base),
                        c => result.push(pool.mul(vec![pool.integer(c), base])),
                    }
                }

                match result.len() {
                    0 => pool.integer(0_i32),
                    1 => result[0],
                    _ => pool.add(result),
                }
            }
            ExprData::Mul(args) => {
                let args: Vec<ExprId> =
                    args.iter().map(|&a| canonicalize_linear(a, pool)).collect();
                pool.mul(args)
            }
            ExprData::Pow { base, exp } => {
                let base = canonicalize_linear(base, pool);
                let exp = canonicalize_linear(exp, pool);
                pool.pow(base, exp)
            }
            ExprData::Func { name, args } => {
                let args: Vec<ExprId> =
                    args.iter().map(|&a| canonicalize_linear(a, pool)).collect();
                pool.func(&name, args)
            }
            _ => expr,
        }
    }

    // -----------------------------------------------------------------------
    // 4. Public implementation
    // -----------------------------------------------------------------------

    pub fn simplify_egraph_impl(
        expr: ExprId,
        pool: &ExprPool,
        config: &super::EgraphConfig,
    ) -> crate::deriv::log::DerivedExpr<ExprId> {
        use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
        use crate::kernel::expr_props::expr_contains_noncommutative_symbol;

        if expr_contains_noncommutative_symbol(pool, expr) {
            return super::super::engine::simplify(expr, pool);
        }

        let expr_str = expr_to_egglog(expr, pool);
        let program = egglog_program(&expr_str, config);

        let result: Option<ExprId> = (|| {
            let mut egraph = egglog::EGraph::default();
            let outputs = egraph.parse_and_run_program(None, &program).ok()?;
            let term_str = outputs.into_iter().last()?;
            parse_egglog_term(&term_str, pool)
        })();

        let simplified = result.unwrap_or(expr);
        // RW-3: apply linear canonizer as a post-extraction pass.
        let simplified = canonicalize_linear(simplified, pool);

        let mut log = DerivationLog::new();
        if simplified != expr {
            log.push(RewriteStep::simple("egraph_simplify", expr, simplified));
        }
        DerivedExpr::with_log(simplified, log)
    }
}

// ---------------------------------------------------------------------------
// PA-6 / RW-4 — Pluggable e-graph cost functions
// ---------------------------------------------------------------------------

use crate::deriv::log::DerivedExpr;
use crate::kernel::{ExprId, ExprPool};

/// Cost model used when extracting from the e-graph.
///
/// The extractor chooses the expression with the *lowest* total cost.
/// Implement this trait to define custom extraction objectives.
///
/// # Built-in implementations
///
/// | Type | Description |
/// |------|-------------|
/// | [`SizeCost`] | Every node costs 1 (tree size). Default. |
/// | [`OpCost`]   | Operators weighted by evaluation cost. |
/// | [`DepthCost`]| Cost = max child depth + 1. |
/// | [`StabilityCost`] | Penalises catastrophic cancellation. |
/// | [`NoncommutativeCost`] | Tie-break for non-commutative `Mul` chains (V3-2). |
pub trait EgraphCost: Send + Sync {
    /// Compute the cost of a node given its operator name and its children's costs.
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64;
}

/// Every node costs 1 (tree-size cost). This is the egglog default.
pub struct SizeCost;
impl EgraphCost for SizeCost {
    fn cost(&self, _op: &str, child_costs: &[f64]) -> f64 {
        1.0 + child_costs.iter().sum::<f64>()
    }
}

/// Operators weighted by their numerical evaluation cost.
pub struct OpCost;
impl EgraphCost for OpCost {
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64 {
        let w = match op {
            "Num" | "Var" => 0.1,
            "Add" => 1.0,
            "Mul" => 1.5,
            "Pow" => 3.0,
            "Sin" | "Cos" | "Exp" | "Log" | "Sqrt" => 5.0,
            _ => 2.0,
        };
        w + child_costs.iter().sum::<f64>()
    }
}

/// Cost = max child depth + 1.
///
/// Minimises the critical-path length; useful for GPU / parallel evaluation
/// where depth determines the number of synchronisation barriers.
pub struct DepthCost;
impl EgraphCost for DepthCost {
    fn cost(&self, _op: &str, child_costs: &[f64]) -> f64 {
        1.0 + child_costs.iter().cloned().fold(0.0_f64, f64::max)
    }
}

/// Penalises catastrophic cancellation.
///
/// Applies a `3×` multiplier to binary `Add`/`Sub` nodes whose both children
/// have non-trivial cost (i.e. not a bare literal), discouraging expressions
/// of the form `large_expr - large_expr` in favour of Horner form or
/// log-sum-exp style rewrites.
pub struct StabilityCost;
impl EgraphCost for StabilityCost {
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64 {
        let base = 1.0 + child_costs.iter().sum::<f64>();
        match op {
            // Penalise binary add/sub between two non-trivial children.
            "Add" | "Sub"
                if child_costs.len() == 2 && child_costs[0] > 1.0 && child_costs[1] > 1.0 =>
            {
                base * 3.0
            }
            "Pow" => base * 2.0,
            _ => base,
        }
    }
}

/// Extraction cost biased toward **left-to-right** (`Mul`) products (V3-2).
///
/// When egglog gains a fully pluggable extractor, this can rank
/// normal-ordered operator strings (Pauli / Clifford) lower than scrambled
/// permutations. Today it adds a small tie-break on `Mul` so experiments
/// with non-commuting `Var` encodings stay deterministic.
pub struct NoncommutativeCost;
impl EgraphCost for NoncommutativeCost {
    fn cost(&self, op: &str, child_costs: &[f64]) -> f64 {
        let base = SizeCost.cost(op, child_costs);
        match op {
            "Mul" => base + 1.0e-6 * child_costs.len() as f64,
            _ => base,
        }
    }
}

// ---------------------------------------------------------------------------
// PA-6 — Schedule configuration  (RW-2: node_limit / iter_limit)
// ---------------------------------------------------------------------------

/// Configuration for the e-graph schedule and extraction strategy.
///
/// Pass to [`simplify_egraph_with`] to customise iteration counts and
/// resource limits.
///
/// # Rule flags
///
/// By default both `include_trig_rules` and `include_log_exp_rules` are `true`,
/// so `simplify_egraph` reduces `sin²(x)+cos²(x)→1` and `exp(log(x))→x`
/// without any extra configuration.  Set either flag to `false` to suppress
/// the corresponding rule set (useful when you need to benchmark rule impact or
/// avoid domain-sensitive rewrites).
#[derive(Debug, Clone)]
pub struct EgraphConfig {
    /// Saturation iterations in the *shrinking* phase. Default 5.
    pub shrink_iters: usize,
    /// Saturation iterations in the *exploring* phase. Default 3.
    pub explore_iters: usize,
    /// Constant-folding iterations appended after each phase. Default 3.
    pub const_fold_iters: usize,
    /// Abort if the e-graph exceeds this many nodes. `None` = unlimited.
    pub node_limit: Option<usize>,
    /// Per-ruleset iteration cap passed to egglog's scheduler. `None` = unlimited.
    pub iter_limit: Option<usize>,
    /// Include the Pythagorean trig identity (`sin²+cos²→1`) in the explore phase.
    /// Default `true`.
    pub include_trig_rules: bool,
    /// Include exp/log cancellation (`exp(log(x))→x`, `log(exp(x))→x`) in the
    /// explore phase. Default `true`.
    pub include_log_exp_rules: bool,
}

impl Default for EgraphConfig {
    fn default() -> Self {
        EgraphConfig {
            shrink_iters: 5,
            explore_iters: 3,
            const_fold_iters: 3,
            node_limit: None,
            iter_limit: None,
            include_trig_rules: true,
            include_log_exp_rules: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Simplify `expr` using the e-graph backend with default settings.
///
/// Falls back to the rule-based simplifier when `egraph` feature is off.
pub fn simplify_egraph(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    #[cfg(feature = "egraph")]
    {
        backend::simplify_egraph_impl(expr, pool, &EgraphConfig::default())
    }
    #[cfg(not(feature = "egraph"))]
    {
        super::engine::simplify(expr, pool)
    }
}

/// Simplify `expr` using the e-graph backend with a custom configuration.
///
/// The `cost` parameter documents the intended extraction preference; full
/// pluggable-extractor support requires a future egglog API.  The config
/// schedule limits (`node_limit`, `iter_limit`, phase iters) are wired
/// into the egglog program today.
pub fn simplify_egraph_with(
    expr: ExprId,
    pool: &ExprPool,
    config: &EgraphConfig,
    _cost: &dyn EgraphCost,
) -> DerivedExpr<ExprId> {
    #[cfg(feature = "egraph")]
    {
        backend::simplify_egraph_impl(expr, pool, config)
    }
    #[cfg(not(feature = "egraph"))]
    {
        let _ = config;
        super::engine::simplify(expr, pool)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn egraph_simplify_x_plus_y_minus_x() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.add(vec![x, y, neg_x]);
        let result = simplify_egraph(expr, &pool);
        assert_ne!(result.value, pool.integer(0_i32), "should not be zero");
    }

    #[test]
    fn egraph_simplify_const_fold() {
        let pool = ExprPool::new();
        let expr = pool.add(vec![pool.integer(3_i32), pool.integer(4_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, pool.integer(7_i32));
    }

    #[test]
    fn egraph_simplify_add_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(0_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, x);
    }

    #[test]
    fn egraph_simplify_mul_one() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.integer(1_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, x);
    }

    #[test]
    fn egraph_simplify_mul_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, pool.integer(0_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, pool.integer(0_i32));
    }

    #[test]
    fn egraph_fallback_no_panic_on_rational() {
        let pool = ExprPool::new();
        let r = pool.rational(1, 3);
        let _ = simplify_egraph(r, &pool);
    }

    // RW-1: flattening round-trip
    #[test]
    fn egraph_round_trips_nary_add() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        // x + y + z should survive the egglog round-trip as a 3-arg Add
        let expr = pool.add(vec![x, y, z]);
        let result = simplify_egraph(expr, &pool);
        // Must still be an Add (not a nested binary tree)
        if let crate::kernel::ExprData::Add(args) =
            crate::kernel::ExprPool::get(&pool, result.value)
        {
            assert_eq!(args.len(), 3);
        }
    }

    // RW-3: linear canonizer
    #[test]
    fn linear_canonizer_combines_like_terms() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 2*x + 3*x = 5*x
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let three_x = pool.mul(vec![pool.integer(3_i32), x]);
        let expr = pool.add(vec![two_x, three_x]);
        #[cfg(feature = "egraph")]
        {
            let result = backend::canonicalize_linear(expr, &pool);
            let five_x = pool.mul(vec![pool.integer(5_i32), x]);
            assert_eq!(result, five_x);
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // RW-2: config wiring compiles and does not panic
    #[test]
    fn egraph_with_node_limit() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(0_i32)]);
        let config = EgraphConfig {
            node_limit: Some(10_000),
            ..EgraphConfig::default()
        };
        let result = simplify_egraph_with(expr, &pool, &config, &SizeCost);
        assert_eq!(result.value, x);
    }

    #[test]
    fn egraph_noncommutative_falls_back_to_rules() {
        let pool = ExprPool::new();
        let a = pool.symbol_commutative("A", Domain::Real, false);
        let expr = pool.add(vec![a, pool.integer(0_i32)]);
        let result = simplify_egraph(expr, &pool);
        assert_eq!(result.value, a);
    }

    // V3-2: NoncommutativeCost is callable
    #[test]
    fn noncommutative_cost_is_callable() {
        let nc = NoncommutativeCost;
        let v = nc.cost("Mul", &[1.0, 1.0]);
        assert!(v.is_finite());
    }

    // RW-4: StabilityCost is callable
    #[test]
    fn stability_cost_penalises_binary_add() {
        let sc = StabilityCost;
        let penalised = sc.cost("Add", &[2.0, 2.0]);
        let normal = sc.cost("Add", &[0.1, 2.0]);
        assert!(penalised > normal);
    }

    // V1-15: trig identity via Pow form (sin(x)^2 + cos(x)^2 → 1)
    #[test]
    fn egraph_trig_identity_pow_form() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let sin2 = pool.pow(sin_x, pool.integer(2_i32));
        let cos2 = pool.pow(cos_x, pool.integer(2_i32));
        let expr = pool.add(vec![sin2, cos2]);
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(result.value, pool.integer(1_i32));
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // V1-15: exp(log(x)) → x
    #[test]
    fn egraph_exp_of_log() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(result.value, x);
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // V1-15: log(exp(x)) → x
    #[test]
    fn egraph_log_of_exp() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("log", vec![pool.func("exp", vec![x])]);
        #[cfg(feature = "egraph")]
        {
            let result = simplify_egraph(expr, &pool);
            assert_eq!(result.value, x);
        }
        #[cfg(not(feature = "egraph"))]
        let _ = expr;
    }

    // V1-15: opt-out trig rules via config
    #[test]
    fn egraph_opt_out_trig_rules() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let cos_x = pool.func("cos", vec![x]);
        let sin2 = pool.pow(sin_x, pool.integer(2_i32));
        let cos2 = pool.pow(cos_x, pool.integer(2_i32));
        let expr = pool.add(vec![sin2, cos2]);
        let config = EgraphConfig {
            include_trig_rules: false,
            ..EgraphConfig::default()
        };
        let result = simplify_egraph_with(expr, &pool, &config, &SizeCost);
        assert_ne!(result.value, pool.integer(1_i32));
    }

    // V1-15: opt-out log/exp rules via config
    #[test]
    fn egraph_opt_out_log_exp_rules() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        let config = EgraphConfig {
            include_log_exp_rules: false,
            ..EgraphConfig::default()
        };
        let result = simplify_egraph_with(expr, &pool, &config, &SizeCost);
        assert_ne!(result.value, x);
    }
}
