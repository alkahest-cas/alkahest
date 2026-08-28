//! Explicit, conservative assumptions for condition-gated simplification.

use crate::deriv::SideCondition;
use crate::errors::AlkahestError;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::logic::{satisfiable, Satisfiability};
use crate::simplify::{rules_for_config, simplify_with, SimplifyConfig};
use crate::DerivedExpr;

/// A contradiction in an explicit assumption context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssumptionError {
    /// The asserted predicate contradicts the context's supported arithmetic facts.
    Contradiction,
}

impl std::fmt::Display for AssumptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contradiction => write!(f, "assumption contradicts the current context"),
        }
    }
}

impl std::error::Error for AssumptionError {}

impl AlkahestError for AssumptionError {
    fn code(&self) -> &'static str {
        "E-SIMPLIFY-001"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some("remove the conflicting refinement or create a separate AssumptionContext")
    }
}

/// User-provided predicates plus the small set of facts safe to use in rewrites.
///
/// Only positivity and non-zero facts are normalized. Other predicates are
/// retained for contradiction checks but do not authorize a simplification.
#[derive(Debug, Clone, Default)]
pub struct AssumptionContext {
    predicates: Vec<ExprId>,
    facts: Vec<SideCondition>,
}

impl AssumptionContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// All predicates explicitly supplied to this context.
    pub fn predicates(&self) -> &[ExprId] {
        &self.predicates
    }

    /// Facts that the conditional simplifier may use.
    pub fn facts(&self) -> &[SideCondition] {
        &self.facts
    }

    /// Add a predicate atomically.
    ///
    /// A definitive `Unsat` result rejects the predicate and leaves this context
    /// unchanged. `Unknown` is retained as provenance but grants no rewrite fact.
    pub fn refine(&mut self, predicate: ExprId, pool: &ExprPool) -> Result<(), AssumptionError> {
        let mut predicates = self.predicates.clone();
        predicates.push(predicate);
        let conjunction = if predicates.len() == 1 {
            predicate
        } else {
            pool.predicate(PredicateKind::And, predicates.clone())
        };
        if matches!(satisfiable(conjunction, pool), Satisfiability::Unsat) {
            return Err(AssumptionError::Contradiction);
        }

        self.predicates = predicates;
        for fact in normalize_predicate(predicate, pool) {
            push_unique(&mut self.facts, fact);
        }
        Ok(())
    }

    /// Simplify under this explicit context.
    ///
    /// Static symbol domains in the expression are added as facts for this call.
    pub fn simplify(&self, expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
        let mut facts = self.facts.clone();
        collect_static_domain_facts(expr, pool, &mut facts);
        let config = SimplifyConfig {
            assumptions: facts,
            ..SimplifyConfig::default()
        };
        let rules = rules_for_config(&config);
        simplify_with(expr, pool, &rules, config)
    }
}

/// Simplify using explicit assumptions without exposing the context internals.
pub fn simplify_with_assumptions(
    expr: ExprId,
    pool: &ExprPool,
    assumptions: &AssumptionContext,
) -> DerivedExpr<ExprId> {
    assumptions.simplify(expr, pool)
}

fn normalize_predicate(predicate: ExprId, pool: &ExprPool) -> Vec<SideCondition> {
    let ExprData::Predicate { kind, args } = pool.get(predicate) else {
        return vec![];
    };
    match kind {
        PredicateKind::And => args
            .iter()
            .flat_map(|&part| normalize_predicate(part, pool))
            .collect(),
        PredicateKind::Gt if args.len() == 2 && is_zero(args[1], pool) => {
            vec![SideCondition::Positive(args[0])]
        }
        PredicateKind::Lt if args.len() == 2 && is_zero(args[0], pool) => {
            vec![SideCondition::Positive(args[1])]
        }
        PredicateKind::Ne if args.len() == 2 && is_zero(args[1], pool) => {
            vec![SideCondition::NonZero(args[0])]
        }
        PredicateKind::Ne if args.len() == 2 && is_zero(args[0], pool) => {
            vec![SideCondition::NonZero(args[1])]
        }
        _ => vec![],
    }
}

fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Integer(value) => value.0 == 0,
        ExprData::Rational(value) => value.0 == 0,
        _ => false,
    }
}

fn push_unique(facts: &mut Vec<SideCondition>, fact: SideCondition) {
    if !facts.contains(&fact) {
        facts.push(fact);
    }
}

/// Walk `expr` and record rewrite facts implied by static symbol domains.
///
/// Only [`Domain::Positive`] (→ Positive + NonZero) and [`Domain::NonZero`] are
/// collected. Other domains do not authorize conditional rewrites.
///
/// # Why this is a worklist and not a recursion
///
/// [`super::engine::simplify_with`] calls this on the *result* of every
/// simplification, so it sees whatever the rules left behind — including an
/// expression the rules could not shrink, at whatever depth the caller built
/// it. As a recursion it descended one stack frame per level with nothing
/// bounding the descent, and running out of stack aborts the process rather
/// than raising anything a caller could catch. That put an abort back on the
/// path `crate::simplify::stack` exists to keep off it, one function later.
///
/// A walk that only *collects* has nothing to compose on the way back up, so
/// it needs no stack at all: the pending nodes live on the heap and depth
/// costs memory instead of stack. `pending` stays empty for an atom and never
/// allocates for one.
///
/// Children are pushed in reverse so popping yields them left to right: the
/// order facts land in `facts` is exactly what the recursion produced, and
/// this is a behaviour-preserving change rather than merely an equivalent one.
pub(crate) fn collect_static_domain_facts(
    expr: ExprId,
    pool: &ExprPool,
    facts: &mut Vec<SideCondition>,
) {
    let mut pending: Vec<ExprId> = Vec::new();
    let mut current = expr;
    loop {
        pool.with(current, |data| match data {
            ExprData::Symbol { domain, .. } => match domain {
                Domain::Positive => {
                    push_unique(facts, SideCondition::Positive(current));
                    push_unique(facts, SideCondition::NonZero(current));
                }
                Domain::NonZero => push_unique(facts, SideCondition::NonZero(current)),
                _ => {}
            },
            ExprData::Add(args)
            | ExprData::Mul(args)
            | ExprData::Func { args, .. }
            | ExprData::Predicate { args, .. } => pending.extend(args.iter().rev()),
            ExprData::Pow { base, exp } => pending.extend([*exp, *base]),
            ExprData::Piecewise { branches, default } => {
                pending.push(*default);
                for (condition, value) in branches.iter().rev() {
                    pending.extend([*value, *condition]);
                }
            }
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                pending.extend([*body, *var]);
            }
            ExprData::BigO(arg) => pending.push(*arg),
            ExprData::RootSum { poly, var, body } => {
                pending.extend([*body, *var, *poly]);
            }
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => {}
        });
        match pending.pop() {
            Some(next) => current = next,
            None => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_refinement_enables_conditional_rewrites() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let mut assumptions = AssumptionContext::new();
        assumptions
            .refine(pool.predicate(PredicateKind::Gt, vec![x, zero]), &pool)
            .unwrap();

        let squared = pool.pow(x, pool.integer(2_i32));
        assert_eq!(
            assumptions
                .simplify(pool.func("sqrt", vec![squared]), &pool)
                .value,
            x
        );
        assert_eq!(
            assumptions
                .simplify(pool.func("exp", vec![pool.func("log", vec![x])]), &pool)
                .value,
            x
        );
    }

    #[test]
    fn nonzero_refinement_enables_cancellation() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let mut assumptions = AssumptionContext::new();
        assumptions
            .refine(pool.predicate(PredicateKind::Ne, vec![x, zero]), &pool)
            .unwrap();

        assert_eq!(
            assumptions.simplify(pool.pow(x, zero), &pool).value,
            pool.integer(1_i32)
        );
        let inverse = pool.pow(x, pool.integer(-1_i32));
        assert_eq!(
            assumptions
                .simplify(pool.mul(vec![x, inverse]), &pool)
                .value,
            pool.integer(1_i32)
        );
    }

    #[test]
    fn contradiction_is_rejected_without_mutating_context() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let positive = pool.predicate(PredicateKind::Gt, vec![x, zero]);
        let nonpositive = pool.predicate(PredicateKind::Le, vec![x, zero]);
        let mut assumptions = AssumptionContext::new();
        assumptions.refine(positive, &pool).unwrap();

        assert_eq!(
            assumptions.refine(nonpositive, &pool),
            Err(AssumptionError::Contradiction)
        );
        assert_eq!(assumptions.predicates(), &[positive]);
    }

    #[test]
    fn positive_refinement_enables_abs_of_positive() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let mut assumptions = AssumptionContext::new();
        assumptions
            .refine(pool.predicate(PredicateKind::Gt, vec![x, zero]), &pool)
            .unwrap();

        assert_eq!(
            assumptions.simplify(pool.func("abs", vec![x]), &pool).value,
            x
        );
    }

    #[test]
    fn abs_without_positive_fact_unchanged() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("abs", vec![x]);
        assert_eq!(AssumptionContext::new().simplify(expr, &pool).value, expr);
    }

    #[test]
    fn static_positive_domain_is_available_without_refinement() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Positive);
        let squared = pool.pow(x, pool.integer(2_i32));

        assert_eq!(
            AssumptionContext::new()
                .simplify(pool.func("sqrt", vec![squared]), &pool)
                .value,
            x
        );
    }
}
