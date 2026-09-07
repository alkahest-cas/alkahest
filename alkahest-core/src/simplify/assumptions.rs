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

// ---------------------------------------------------------------------------
// Ambient assumption scope
// ---------------------------------------------------------------------------
//
// An [`AssumptionContext`] is a value the caller threads into the one call that
// wants it, which works for [`AssumptionContext::simplify`] and for `solve`,
// because the caller talks to those directly. It does not work for a fact that
// only a *sub*-engine needs: `integrate(exp(-k·t), t, 0, ∞)` bottoms out in
// `lim_{t→∞}` sixteen frames down, and threading an `&AssumptionContext`
// through every intermediate signature would be a breaking change to most of
// this crate's public surface for the benefit of one leaf.
//
// So the scope is ambient and thread-local, exactly like [`crate::budget`]:
// `alkahest.context(assumptions=…)` pushes a frame for the duration of its
// `with` block and every engine on that thread can consult it. The asymmetry
// noted in ARCHITECTURE.md applies here too — the stack is per-thread, so a
// `rayon` worker does not inherit its parent's assumptions and will simply
// refuse for want of a fact rather than answer with one it cannot see.
//
// # Why the frame records a pool address
//
// An `ExprId` is an index into one `ExprPool` and means something different in
// another. A frame therefore records the address of the pool its predicates
// were interned in, and a query against any other pool ignores it. The address
// is never dereferenced — it is compared, and nothing else — and
// [`enter_assumptions`] borrows the pool for the guard's lifetime, so the
// address cannot be recycled while a frame that mentions it is live.

use std::cell::RefCell;
use std::marker::PhantomData;

/// A sign established by the ambient assumptions (or by structure).
///
/// Deliberately three-valued and wrapped in an `Option` at every use site:
/// `None` means *unknown*, which is a different claim from
/// [`Sign::Zero`], and conflating the two is how a limit engine ends up
/// guessing `+∞` for `lim_{x→∞} k·x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    /// Strictly less than zero.
    Negative,
    /// Exactly zero.
    Zero,
    /// Strictly greater than zero.
    Positive,
}

impl Sign {
    /// `-1`, `0`, `1` — the encoding the older `structural_sign` helpers use.
    pub fn to_i8(self) -> i8 {
        match self {
            Sign::Negative => -1,
            Sign::Zero => 0,
            Sign::Positive => 1,
        }
    }

    fn of_ordering(ord: std::cmp::Ordering) -> Sign {
        match ord {
            std::cmp::Ordering::Less => Sign::Negative,
            std::cmp::Ordering::Equal => Sign::Zero,
            std::cmp::Ordering::Greater => Sign::Positive,
        }
    }

    fn negate(self) -> Sign {
        match self {
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
            Sign::Positive => Sign::Negative,
        }
    }
}

struct AmbientFrame {
    /// Address of the pool `predicates` were interned in. Compared, never read.
    pool: usize,
    predicates: Vec<ExprId>,
}

thread_local! {
    static AMBIENT: RefCell<Vec<AmbientFrame>> = const { RefCell::new(Vec::new()) };
}

/// RAII handle for one ambient assumption frame.
///
/// Not `Send`: the frame lives on a thread-local stack and must be dropped on
/// the thread that pushed it, the same contract [`crate::budget::BudgetGuard`]
/// carries.
pub struct AssumptionScope {
    _not_send: PhantomData<*const ()>,
}

impl Drop for AssumptionScope {
    fn drop(&mut self) {
        AMBIENT.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Make `assumptions` visible to every engine on this thread until the
/// returned guard is dropped.
///
/// `pool` must be the pool `assumptions`' predicates were interned in; queries
/// made against a different pool ignore this frame.
pub fn enter_assumptions(assumptions: &AssumptionContext, pool: &ExprPool) -> AssumptionScope {
    AMBIENT.with(|s| {
        s.borrow_mut().push(AmbientFrame {
            pool: pool as *const ExprPool as usize,
            predicates: assumptions.predicates().to_vec(),
        })
    });
    AssumptionScope {
        _not_send: PhantomData,
    }
}

/// `true` when some ambient frame on this thread was entered with `pool`.
///
/// The hot-path escape hatch: every query below is a no-op without one, and
/// this is a single thread-local read.
pub fn ambient_assumptions_active(pool: &ExprPool) -> bool {
    let addr = pool as *const ExprPool as usize;
    AMBIENT.with(|s| s.borrow().iter().any(|f| f.pool == addr))
}

/// Symbols the ambient scope asserts are exactly equal to a numeric constant.
///
/// Returned as `(symbol, value)` pairs so a caller can substitute them before
/// doing any analysis. Only equalities against a literal are reported: they are
/// the only ones for which substitution is unconditionally sound and
/// terminating.
pub fn ambient_equalities(pool: &ExprPool) -> Vec<(ExprId, ExprId)> {
    let addr = pool as *const ExprPool as usize;
    let mut out: Vec<(ExprId, ExprId)> = Vec::new();
    AMBIENT.with(|s| {
        for frame in s.borrow().iter().filter(|f| f.pool == addr) {
            for &p in &frame.predicates {
                collect_equalities(p, pool, &mut out);
            }
        }
    });
    out
}

fn collect_equalities(predicate: ExprId, pool: &ExprPool, out: &mut Vec<(ExprId, ExprId)>) {
    let ExprData::Predicate { kind, args } = pool.get(predicate) else {
        return;
    };
    match kind {
        PredicateKind::And => {
            for &part in &args {
                collect_equalities(part, pool, out);
            }
        }
        PredicateKind::Eq if args.len() == 2 => {
            let is_sym = |e: ExprId| matches!(pool.get(e), ExprData::Symbol { .. });
            let is_num = |e: ExprId| {
                matches!(
                    pool.get(e),
                    ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
                )
            };
            if is_sym(args[0]) && is_num(args[1]) {
                out.push((args[0], args[1]));
            } else if is_num(args[0]) && is_sym(args[1]) {
                out.push((args[1], args[0]));
            }
        }
        _ => {}
    }
}

/// The sign of `expr`, as far as structure and the ambient assumptions
/// establish it. `None` means *not established* — never *zero*.
///
/// Composition is the usual sign algebra, and every rule is conservative: a
/// factor of unknown sign makes the product unknown, and a sum is only decided
/// when every summand agrees. Callers that used to fall back to a guess when
/// this returns `None` must refuse instead; that guess is the defect this
/// function exists to remove.
pub fn assumed_sign(expr: ExprId, pool: &ExprPool) -> Option<Sign> {
    assumed_sign_at(expr, pool, 0)
}

/// Bound on how deep the sign algebra recurses. A sign is decided by the
/// leading structure of an expression; anything this far down is a coefficient
/// the caller is better off refusing than exploring.
const MAX_SIGN_DEPTH: u32 = 24;

fn assumed_sign_at(expr: ExprId, pool: &ExprPool, depth: u32) -> Option<Sign> {
    if depth > MAX_SIGN_DEPTH {
        return None;
    }
    // A stated fact about this exact expression wins outright: the user said so.
    if let Some(s) = stated_sign(expr, pool) {
        return Some(s);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(Sign::of_ordering(n.0.cmp(&rug::Integer::ZERO))),
        ExprData::Rational(r) => Some(Sign::of_ordering(r.0.cmp0())),
        ExprData::Float(f) => {
            let v = f.inner.to_f64();
            if !v.is_finite() {
                return None;
            }
            Some(Sign::of_ordering(v.partial_cmp(&0.0)?))
        }
        ExprData::Symbol {
            domain: Domain::Positive,
            ..
        } => Some(Sign::Positive),
        ExprData::Symbol { .. } => None,
        ExprData::Mul(xs) => {
            let mut sign = Sign::Positive;
            for x in xs {
                match assumed_sign_at(x, pool, depth + 1)? {
                    Sign::Zero => return Some(Sign::Zero),
                    Sign::Negative => sign = sign.negate(),
                    Sign::Positive => {}
                }
            }
            Some(sign)
        }
        ExprData::Add(xs) => {
            let mut sign: Option<Sign> = None;
            for x in xs {
                let s = assumed_sign_at(x, pool, depth + 1)?;
                match (sign, s) {
                    (_, Sign::Zero) => {}
                    (None, s) => sign = Some(s),
                    (Some(a), b) if a == b => {}
                    _ => return None,
                }
            }
            sign.or(Some(Sign::Zero))
        }
        ExprData::Pow { base, exp } => {
            let base_sign = assumed_sign_at(base, pool, depth + 1);
            // An even integer power is non-negative whatever the base is, and
            // strictly positive once the base is known non-zero.
            if let ExprData::Integer(n) = pool.get(exp) {
                if n.0.is_even() {
                    return match base_sign {
                        Some(Sign::Zero) => Some(Sign::Zero),
                        Some(_) => Some(Sign::Positive),
                        // A base of unknown sign may still be *zero*, and `0² = 0`
                        // is not positive. `≥ 0` is not a sign, so say nothing. A
                        // caller that separately knows the base is non-zero can
                        // take the stronger step itself.
                        None => None,
                    };
                }
                // An odd power preserves the base's sign; `x^{-1}` too.
                return base_sign;
            }
            // A positive base raised to anything real stays positive.
            match base_sign {
                Some(Sign::Positive) => Some(Sign::Positive),
                _ => None,
            }
        }
        ExprData::Func { name, args } => match name.as_str() {
            "exp" | "cosh" => Some(Sign::Positive),
            "abs" if args.len() == 1 => match assumed_sign_at(args[0], pool, depth + 1) {
                Some(Sign::Zero) => Some(Sign::Zero),
                Some(_) => Some(Sign::Positive),
                None => None,
            },
            "sqrt" | "cbrt" if args.len() == 1 => assumed_sign_at(args[0], pool, depth + 1),
            _ => None,
        },
        _ => None,
    }
}

/// A sign asserted directly about `expr` by some ambient frame, or implied by
/// `expr`'s static [`Domain`].
fn stated_sign(expr: ExprId, pool: &ExprPool) -> Option<Sign> {
    if matches!(
        pool.get(expr),
        ExprData::Symbol {
            domain: Domain::Positive,
            ..
        }
    ) {
        return Some(Sign::Positive);
    }
    let addr = pool as *const ExprPool as usize;
    AMBIENT.with(|s| {
        for frame in s.borrow().iter().rev().filter(|f| f.pool == addr) {
            for &p in &frame.predicates {
                if let Some(sign) = sign_from_predicate(p, expr, pool) {
                    return Some(sign);
                }
            }
        }
        None
    })
}

/// The sign `predicate` states for `subject`, if it states one.
///
/// Only comparisons against literal zero are read. `a > b` with `b` non-zero
/// says nothing about `a`'s sign without also knowing `b`'s, and a partial
/// implementation of that inference is a worse answer than none.
fn sign_from_predicate(predicate: ExprId, subject: ExprId, pool: &ExprPool) -> Option<Sign> {
    let ExprData::Predicate { kind, args } = pool.get(predicate) else {
        return None;
    };
    if matches!(kind, PredicateKind::And) {
        return args
            .iter()
            .find_map(|&part| sign_from_predicate(part, subject, pool));
    }
    if args.len() != 2 {
        return None;
    }
    let (lhs, rhs) = (args[0], args[1]);
    // Orient the comparison so `subject` is on the left, mirroring the operator.
    let kind = if lhs == subject && is_zero(rhs, pool) {
        kind
    } else if rhs == subject && is_zero(lhs, pool) {
        match kind {
            PredicateKind::Lt => PredicateKind::Gt,
            PredicateKind::Le => PredicateKind::Ge,
            PredicateKind::Gt => PredicateKind::Lt,
            PredicateKind::Ge => PredicateKind::Le,
            other => other,
        }
    } else {
        return None;
    };
    match kind {
        PredicateKind::Gt => Some(Sign::Positive),
        PredicateKind::Lt => Some(Sign::Negative),
        PredicateKind::Eq => Some(Sign::Zero),
        // `≥ 0` and `≤ 0` do not establish a *strict* sign, and `≠ 0` does not
        // establish a direction. Reporting one would be a guess.
        _ => None,
    }
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
