//! Colored e-graphs for conditional simplification (arXiv:2305.19203).
//!
//! A colored e-graph maintains multiple layered congruence relations over a shared
//! term arena. The root color (id 0) holds equalities valid unconditionally; each
//! assumption color coarsens the root relation with additional merges that are only
//! valid under declared side conditions (e.g. `x > 0 ⊢ sqrt(x²) → x`).
//!
//! This module implements a native colored e-graph used by [`simplify_colored`] and
//! [`super::engine::simplify_with`] when [`super::engine::SimplifyConfig`]`::assumptions` is non-empty.
//! It does not require the optional `egraph` / egglog feature.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Colors and union-find
// ---------------------------------------------------------------------------

/// A layer id in the colored e-graph. [`ROOT_COLOR`] is unconditional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorId(pub u32);

/// Root congruence: equalities that hold in all cases.
pub const ROOT_COLOR: ColorId = ColorId(0);

/// Context color used when at least one assumption is supplied.
pub const CONTEXT_COLOR: ColorId = ColorId(1);

/// Per-color union-find over [`ExprId`] handles (shared term arena).
#[derive(Debug, Default)]
struct ColorUnionFind {
    parent: HashMap<ExprId, ExprId>,
}

impl ColorUnionFind {
    fn make_set(&mut self, id: ExprId) {
        self.parent.entry(id).or_insert(id);
    }

    fn find(&mut self, mut x: ExprId) -> ExprId {
        let mut path = Vec::new();
        loop {
            match self.parent.get(&x) {
                Some(&p) if p != x => {
                    path.push(x);
                    x = p;
                }
                _ => break,
            }
        }
        for node in path {
            self.parent.insert(node, x);
        }
        x
    }

    /// Merge `other` into `rep` (directed union: `rep` is the e-class representative).
    fn union_into(&mut self, rep: ExprId, other: ExprId) {
        let rr = self.find(rep);
        let ro = self.find(other);
        if rr != ro {
            self.parent.insert(ro, rr);
        }
    }

    fn same(&mut self, a: ExprId, b: ExprId) -> bool {
        self.find(a) == self.find(b)
    }
}

// ---------------------------------------------------------------------------
// Colored e-graph
// ---------------------------------------------------------------------------

/// Colored e-graph: shared terms with layered union-find structures.
///
/// Root merges propagate to every assumption layer (coarsening). Conditional merges
/// apply only on [`CONTEXT_COLOR`] when the caller has declared matching assumptions.
#[derive(Debug)]
pub struct ColoredEgraph {
    /// Layer 0: unconditional congruence.
    root: ColorUnionFind,
    /// Layer 1: merges valid only under declared assumptions (if any).
    context: ColorUnionFind,
    /// All expression nodes reachable from the input.
    nodes: HashSet<ExprId>,
    /// Assumptions attached to the context layer.
    #[allow(dead_code)]
    assumptions: Vec<SideCondition>,
}

impl ColoredEgraph {
    /// Build a colored e-graph from `expr`, collecting all DAG nodes.
    pub fn from_expr(expr: ExprId, pool: &ExprPool, assumptions: &[SideCondition]) -> Self {
        let mut nodes = HashSet::new();
        collect_nodes(expr, pool, &mut nodes);
        let mut root = ColorUnionFind::default();
        let mut context = ColorUnionFind::default();
        for &id in &nodes {
            root.make_set(id);
            context.make_set(id);
        }
        ColoredEgraph {
            root,
            context,
            nodes,
            assumptions: assumptions.to_vec(),
        }
    }

    /// Merge `other` into `rep` in the root layer and propagate to the context layer.
    pub fn union_root(&mut self, rep: ExprId, other: ExprId) {
        self.root.union_into(rep, other);
        self.context.union_into(rep, other);
    }

    /// Merge `other` into `rep` only under assumptions (context layer).
    pub fn union_context(&mut self, rep: ExprId, other: ExprId) {
        self.context.union_into(rep, other);
    }

    /// Representative of `expr` in the given color layer.
    pub fn find(&mut self, expr: ExprId, color: ColorId) -> ExprId {
        match color {
            ROOT_COLOR => self.root.find(expr),
            CONTEXT_COLOR => self.context.find(expr),
            _ => expr,
        }
    }

    /// Whether `a` and `b` are equal in `color`.
    pub fn same(&mut self, color: ColorId, a: ExprId, b: ExprId) -> bool {
        match color {
            ROOT_COLOR => self.root.same(a, b),
            CONTEXT_COLOR => self.context.same(a, b),
            _ => a == b,
        }
    }

    /// Rebuild `expr` with children canonicalized under `color`.
    pub fn rebuild(&mut self, expr: ExprId, pool: &ExprPool, color: ColorId) -> ExprId {
        self.rebuild_rec(expr, pool, color, &mut HashSet::new())
    }

    fn rebuild_rec(
        &mut self,
        expr: ExprId,
        pool: &ExprPool,
        color: ColorId,
        visiting: &mut HashSet<ExprId>,
    ) -> ExprId {
        let canon = self.find(expr, color);
        if !visiting.insert(canon) {
            return canon;
        }
        let out = match pool.get(canon) {
            ExprData::Integer(_)
            | ExprData::Rational(_)
            | ExprData::Float(_)
            | ExprData::Symbol { .. } => canon,
            ExprData::Add(args) => {
                let mut children: Vec<ExprId> = args
                    .iter()
                    .map(|&a| self.rebuild_rec(a, pool, color, visiting))
                    .collect();
                children.sort_by_key(|id| self.find(*id, color).0);
                if children.is_empty() {
                    pool.integer(0_i32)
                } else if children.len() == 1 {
                    children[0]
                } else {
                    pool.add(children)
                }
            }
            ExprData::Mul(args) => {
                let mut children: Vec<ExprId> = args
                    .iter()
                    .map(|&a| self.rebuild_rec(a, pool, color, visiting))
                    .collect();
                children.sort_by_key(|id| self.find(*id, color).0);
                if children.is_empty() {
                    pool.integer(1_i32)
                } else if children.len() == 1 {
                    children[0]
                } else {
                    pool.mul(children)
                }
            }
            ExprData::Pow { base, exp } => {
                let b = self.rebuild_rec(base, pool, color, visiting);
                let e = self.rebuild_rec(exp, pool, color, visiting);
                pool.pow(b, e)
            }
            ExprData::Func { name, args } => {
                let rebuilt: Vec<ExprId> = args
                    .iter()
                    .map(|&a| self.rebuild_rec(a, pool, color, visiting))
                    .collect();
                pool.func(&name, rebuilt)
            }
            ExprData::Piecewise { branches, default } => {
                let branches: Vec<(ExprId, ExprId)> = branches
                    .iter()
                    .map(|&(c, v)| {
                        (
                            self.rebuild_rec(c, pool, color, visiting),
                            self.rebuild_rec(v, pool, color, visiting),
                        )
                    })
                    .collect();
                let def = self.rebuild_rec(default, pool, color, visiting);
                pool.piecewise(branches, def)
            }
            ExprData::Predicate { kind, args } => {
                let rebuilt: Vec<ExprId> = args
                    .iter()
                    .map(|&a| self.rebuild_rec(a, pool, color, visiting))
                    .collect();
                pool.predicate(kind.clone(), rebuilt)
            }
            ExprData::Forall { var, body } => pool.forall(
                self.rebuild_rec(var, pool, color, visiting),
                self.rebuild_rec(body, pool, color, visiting),
            ),
            ExprData::Exists { var, body } => pool.exists(
                self.rebuild_rec(var, pool, color, visiting),
                self.rebuild_rec(body, pool, color, visiting),
            ),
            ExprData::BigO(arg) => pool.big_o(self.rebuild_rec(arg, pool, color, visiting)),
            ExprData::RootSum { poly, var, body } => pool.root_sum(
                self.rebuild_rec(poly, pool, color, visiting),
                self.rebuild_rec(var, pool, color, visiting),
                self.rebuild_rec(body, pool, color, visiting),
            ),
        };
        visiting.remove(&canon);
        out
    }

    /// Propagate congruence in `color`: if children match, merge parent applications.
    fn rebuild_congruence(&mut self, pool: &ExprPool, color: ColorId) {
        const MAX_ROUNDS: usize = 16;
        let nodes: Vec<ExprId> = self.nodes.iter().copied().collect();
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for &id in &nodes {
                let rep = self.find(id, color);
                if let Some(other) = congruent_partner(rep, &nodes, pool, self, color) {
                    if !self.same(color, rep, other) {
                        let canon = if rep.0 <= other.0 { rep } else { other };
                        let other_id = if rep.0 <= other.0 { other } else { rep };
                        match color {
                            ROOT_COLOR => self.union_root(canon, other_id),
                            CONTEXT_COLOR => self.union_context(canon, other_id),
                            _ => {}
                        }
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
}

fn collect_nodes(expr: ExprId, pool: &ExprPool, out: &mut HashSet<ExprId>) {
    if !out.insert(expr) {
        return;
    }
    match pool.get(expr) {
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                collect_nodes(a, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_nodes(base, pool, out);
            collect_nodes(exp, pool, out);
        }
        ExprData::Func { args, .. } => {
            for &a in &args {
                collect_nodes(a, pool, out);
            }
        }
        ExprData::Piecewise { branches, default } => {
            for (c, v) in &branches {
                collect_nodes(*c, pool, out);
                collect_nodes(*v, pool, out);
            }
            collect_nodes(default, pool, out);
        }
        ExprData::Predicate { args, .. } => {
            for a in args {
                collect_nodes(a, pool, out);
            }
        }
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
            collect_nodes(var, pool, out);
            collect_nodes(body, pool, out);
        }
        ExprData::BigO(arg) => collect_nodes(arg, pool, out),
        _ => {}
    }
}

/// If another node in `nodes` is congruent to `rep` under `color`, return it.
fn congruent_partner(
    rep: ExprId,
    nodes: &[ExprId],
    pool: &ExprPool,
    eg: &mut ColoredEgraph,
    color: ColorId,
) -> Option<ExprId> {
    let data = pool.get(rep);
    for &other in nodes {
        if other == rep {
            continue;
        }
        if eg.find(other, color) == rep {
            continue;
        }
        if congruent_same_op(&data, &pool.get(other), eg, color) {
            return Some(other);
        }
    }
    None
}

fn congruent_same_op(a: &ExprData, b: &ExprData, eg: &mut ColoredEgraph, color: ColorId) -> bool {
    match (a, b) {
        (ExprData::Add(aa), ExprData::Add(bb)) if aa.len() == bb.len() => {
            aa.iter().zip(bb).all(|(x, y)| eg.same(color, *x, *y))
        }
        (ExprData::Mul(aa), ExprData::Mul(bb)) if aa.len() == bb.len() => {
            aa.iter().zip(bb).all(|(x, y)| eg.same(color, *x, *y))
        }
        (ExprData::Pow { base: ab, exp: ae }, ExprData::Pow { base: bb, exp: be }) => {
            eg.same(color, *ab, *bb) && eg.same(color, *ae, *be)
        }
        (ExprData::Func { name: na, args: aa }, ExprData::Func { name: nb, args: bb })
            if na == nb && aa.len() == bb.len() =>
        {
            aa.iter().zip(bb).all(|(x, y)| eg.same(color, *x, *y))
        }
        (ExprData::Integer(na), ExprData::Integer(nb)) => na.0 == nb.0,
        (
            ExprData::Symbol {
                name: na,
                domain: da,
                commutative: ca,
            },
            ExprData::Symbol {
                name: nb,
                domain: db,
                commutative: cb,
            },
        ) => na == nb && da == db && ca == cb,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Assumption checking
// ---------------------------------------------------------------------------

/// True when every `required` condition is implied by `declared`.
pub fn assumptions_satisfy(required: &[SideCondition], declared: &[SideCondition]) -> bool {
    required
        .iter()
        .all(|req| declared.iter().any(|d| condition_implies(d, req)))
}

fn condition_implies(declared: &SideCondition, required: &SideCondition) -> bool {
    match (declared, required) {
        (a, b) if a == b => true,
        // Positive implies NonZero and NonNegative-style use.
        (SideCondition::Positive(id), SideCondition::NonZero(rid)) if id == rid => true,
        (SideCondition::InDomain(id, domain), SideCondition::InDomain(rid, rdom))
            if id == rid && domain_implies(*domain, *rdom) =>
        {
            true
        }
        _ => false,
    }
}

fn domain_implies(have: crate::kernel::Domain, need: crate::kernel::Domain) -> bool {
    use crate::kernel::Domain;
    match (have, need) {
        (a, b) if a == b => true,
        (Domain::Positive, Domain::NonNegative | Domain::NonZero | Domain::Real) => true,
        (Domain::NonNegative, Domain::Real) => true,
        (Domain::NonZero, Domain::Real) => true,
        (Domain::Integer, Domain::Real) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Conditional rewrite rules
// ---------------------------------------------------------------------------

type ConditionalApplyFn = fn(ExprId, &ExprPool) -> Option<(ExprId, Vec<SideCondition>)>;

struct ConditionalRule {
    name: &'static str,
    apply: ConditionalApplyFn,
}

fn is_int_n(expr: ExprId, n: i64, pool: &ExprPool) -> bool {
    matches!(pool.get(expr), ExprData::Integer(v) if v.0.to_i64() == Some(n))
}

fn func_arg(name: &str, expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    match pool.get(expr) {
        ExprData::Func { name: n, args } if n == name && args.len() == 1 => Some(args[0]),
        _ => None,
    }
}

/// `sqrt(x^2) → x` when `x > 0`.
fn rule_sqrt_of_square(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
    let inner = func_arg("sqrt", expr, pool)?;
    let (base, exp) = match pool.get(inner) {
        ExprData::Pow { base, exp } => (base, exp),
        _ => return None,
    };
    if !is_int_n(exp, 2, pool) {
        return None;
    }
    let cond = SideCondition::Positive(base);
    Some((base, vec![cond]))
}

/// `log(a * b) → log(a) + log(b)` when all factors are positive.
fn rule_log_of_product(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
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
    Some((after, conds))
}

/// `exp(log(x)) → x` when `x > 0`.
fn rule_exp_of_log(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
    let arg = func_arg("exp", expr, pool)?;
    let inner = func_arg("log", arg, pool)?;
    Some((inner, vec![SideCondition::Positive(inner)]))
}

/// `log(x) + log(y) + ⋯ → log(x·y·⋯)` when every argument is positive.
fn rule_sum_of_logs(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
    let terms = match pool.get(expr) {
        ExprData::Add(v) if v.len() >= 2 => v,
        _ => return None,
    };
    let mut args = Vec::with_capacity(terms.len());
    for t in terms {
        let a = func_arg("log", t, pool)?;
        args.push(a);
    }
    let after = pool.func("log", vec![pool.mul(args.clone())]);
    let conds: Vec<SideCondition> = args.iter().map(|&a| SideCondition::Positive(a)).collect();
    Some((after, conds))
}

/// `log(a^n) → n * log(a)` when `a > 0`.
fn rule_log_of_pow(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
    let arg = func_arg("log", expr, pool)?;
    let (base, exp) = match pool.get(arg) {
        ExprData::Pow { base, exp } => (base, exp),
        _ => return None,
    };
    let log_base = pool.func("log", vec![base]);
    let after = pool.mul(vec![exp, log_base]);
    Some((after, vec![SideCondition::Positive(base)]))
}

/// `log(a · b⁻¹ · ⋯) → log(a) − log(b) − ⋯` when factors are positive.
fn rule_log_of_quotient(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
    let arg = func_arg("log", expr, pool)?;
    let factors = match pool.get(arg) {
        ExprData::Mul(v) if v.len() >= 2 => v,
        _ => return None,
    };
    let has_reciprocal = factors.iter().any(|&f| match pool.get(f) {
        ExprData::Pow { exp, .. } => match pool.get(exp) {
            ExprData::Integer(n) => n.0 < 0,
            _ => false,
        },
        _ => false,
    });
    if !has_reciprocal {
        return None;
    }
    let mut terms: Vec<ExprId> = Vec::with_capacity(factors.len());
    let mut conds: Vec<SideCondition> = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Pow { base, exp } => {
                conds.push(SideCondition::Positive(base));
                let log_base = pool.func("log", vec![base]);
                terms.push(pool.mul(vec![exp, log_base]));
            }
            _ => {
                conds.push(SideCondition::Positive(f));
                terms.push(pool.func("log", vec![f]));
            }
        }
    }
    Some((pool.add(terms), conds))
}

/// `x^0 → 1` when `x ≠ 0`.
fn rule_pow_zero_nonzero(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
    let (base, exp) = match pool.get(expr) {
        ExprData::Pow { base, exp } => (base, exp),
        _ => return None,
    };
    if !is_int_n(exp, 0, pool) {
        return None;
    }
    Some((pool.integer(1_i32), vec![SideCondition::NonZero(base)]))
}

/// `x * x^-1 → 1` when `x ≠ 0`.
fn rule_mul_inverse_nonzero(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<SideCondition>)> {
    let factors = match pool.get(expr) {
        ExprData::Mul(factors) if factors.len() == 2 => factors,
        _ => return None,
    };
    for &(base, other) in [(factors[0], factors[1]), (factors[1], factors[0])].iter() {
        let ExprData::Pow {
            base: inverse_base,
            exp,
        } = pool.get(other)
        else {
            continue;
        };
        if inverse_base == base && is_int_n(exp, -1, pool) {
            return Some((pool.integer(1_i32), vec![SideCondition::NonZero(base)]));
        }
    }
    None
}

fn default_conditional_rules() -> &'static [ConditionalRule] {
    &[
        ConditionalRule {
            name: "sqrt_of_square_positive",
            apply: rule_sqrt_of_square,
        },
        ConditionalRule {
            name: "log_of_product_positive",
            apply: rule_log_of_product,
        },
        ConditionalRule {
            name: "exp_of_log",
            apply: rule_exp_of_log,
        },
        ConditionalRule {
            name: "sum_of_logs",
            apply: rule_sum_of_logs,
        },
        ConditionalRule {
            name: "log_of_pow",
            apply: rule_log_of_pow,
        },
        ConditionalRule {
            name: "log_of_quotient",
            apply: rule_log_of_quotient,
        },
        ConditionalRule {
            name: "pow_zero_nonzero",
            apply: rule_pow_zero_nonzero,
        },
        ConditionalRule {
            name: "mul_inverse_nonzero",
            apply: rule_mul_inverse_nonzero,
        },
    ]
}

// ---------------------------------------------------------------------------
// Saturation
// ---------------------------------------------------------------------------

/// Run colored equality saturation on `expr` under `assumptions`.
///
/// Returns the expression canonicalized in the context layer when assumptions are
/// non-empty, otherwise in the root layer. Unconditional rules still update the root.
pub fn simplify_colored(
    expr: ExprId,
    pool: &ExprPool,
    assumptions: &[SideCondition],
) -> DerivedExpr<ExprId> {
    let mut eg = ColoredEgraph::from_expr(expr, pool, assumptions);
    let rules = default_conditional_rules();
    let mut log = DerivationLog::new();
    let has_context = !assumptions.is_empty();
    let extract_color = if has_context {
        CONTEXT_COLOR
    } else {
        ROOT_COLOR
    };

    let max_iters = 32;
    for _ in 0..max_iters {
        let mut fired = false;
        let nodes: Vec<ExprId> = eg.nodes.iter().copied().collect();
        for &node in &nodes {
            for rule in rules {
                let Some((after, conds)) = (rule.apply)(node, pool) else {
                    continue;
                };
                let unconditional = conds.is_empty();
                if unconditional {
                    if !eg.same(ROOT_COLOR, node, after) {
                        eg.union_root(after, node);
                        log.push(RewriteStep::simple(rule.name, node, after));
                        fired = true;
                    }
                } else if has_context
                    && assumptions_satisfy(&conds, assumptions)
                    && !eg.same(CONTEXT_COLOR, node, after)
                {
                    eg.union_context(after, node);
                    log.push(RewriteStep::with_conditions(rule.name, node, after, conds));
                    fired = true;
                }
            }
        }
        eg.rebuild_congruence(pool, ROOT_COLOR);
        if has_context {
            eg.rebuild_congruence(pool, CONTEXT_COLOR);
        }
        if !fired {
            break;
        }
        // New canonical forms may enable further matches.
        let rebuilt_root = eg.rebuild(expr, pool, extract_color);
        collect_nodes(rebuilt_root, pool, &mut eg.nodes);
    }

    let simplified = eg.rebuild(expr, pool, extract_color);
    if simplified != expr && log.is_empty() {
        log.push(RewriteStep::simple("colored_rebuild", expr, simplified));
    }
    DerivedExpr::with_log(simplified, log)
}

/// Apply colored simplification when assumptions are present; otherwise no-op.
pub(crate) fn apply_colored_if_needed(
    expr: ExprId,
    pool: &ExprPool,
    assumptions: &[SideCondition],
) -> DerivedExpr<ExprId> {
    if assumptions.is_empty() {
        return DerivedExpr::new(expr);
    }
    simplify_colored(expr, pool, assumptions)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn sqrt_square_positive_simplifies() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Positive);
        let expr = pool.func("sqrt", vec![pool.pow(x, pool.integer(2_i32))]);
        let assumptions = vec![SideCondition::Positive(x)];
        let r = simplify_colored(expr, &pool, &assumptions);
        assert_eq!(r.value, x);
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "sqrt_of_square_positive"));
    }

    #[test]
    fn sqrt_square_without_assumption_unchanged() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("sqrt", vec![pool.pow(x, pool.integer(2_i32))]);
        let r = simplify_colored(expr, &pool, &[]);
        assert_eq!(r.value, expr);
    }

    #[test]
    fn log_product_under_positive_assumptions() {
        let pool = pool();
        let a = pool.symbol("a", Domain::Positive);
        let b = pool.symbol("b", Domain::Positive);
        let expr = pool.func("log", vec![pool.mul(vec![a, b])]);
        let assumptions = vec![SideCondition::Positive(a), SideCondition::Positive(b)];
        let r = simplify_colored(expr, &pool, &assumptions);
        let expected = pool.add(vec![pool.func("log", vec![a]), pool.func("log", vec![b])]);
        assert_eq!(r.value, expected);
    }

    #[test]
    fn exp_of_log_requires_positive_assumption() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![pool.func("log", vec![x])]);
        let r = simplify_colored(expr, &pool, &[]);
        assert_eq!(r.value, expr);

        let r = simplify_colored(expr, &pool, &[SideCondition::Positive(x)]);
        assert_eq!(r.value, x);
    }

    #[test]
    fn colored_rebuild_preserves_repeated_children() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let assumptions = [SideCondition::Positive(x)];

        let doubled = pool.add(vec![x, x]);
        let squared = pool.mul(vec![x, x]);
        assert_eq!(
            simplify_colored(doubled, &pool, &assumptions).value,
            doubled
        );
        assert_eq!(
            simplify_colored(squared, &pool, &assumptions).value,
            squared
        );
    }

    #[test]
    fn assumptions_satisfy_positive_implies_nonzero() {
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let pos = SideCondition::Positive(x);
        let nz = SideCondition::NonZero(x);
        assert!(assumptions_satisfy(&[nz], &[pos]));
    }
}
