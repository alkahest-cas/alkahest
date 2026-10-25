//! The user-facing "simplest form" simplifier: the default rule engine, then
//! the domain passes that the default engine deliberately leaves out.
//!
//! [`simplify`](super::engine::simplify) is the algebraic core every subsystem
//! calls on its own intermediate results (integration, limits, Gröbner
//! back-substitution, …). It must stay cheap and it must not change spellings
//! those subsystems rely on, so it carries no trigonometric, hyperbolic,
//! log/exp, rational-cancellation or radical identities. That made
//! `simplify(sin(x)**2 + cos(x)**2)` return its input while `simplify_trig`,
//! `simplify_egraph` and `is_zero` all knew the answer — the first thing a new
//! user types.
//!
//! [`simplify_full`] runs the core and then tries each domain pass whose
//! trigger is present in the expression:
//!
//! | pass | example |
//! |---|---|
//! | Pythagorean / hyperbolic collection | `y + sin²x + cos²x → y + 1`, `cosh²x − sinh²x → 1` |
//! | trigonometric identities | `simplify_trig` |
//! | log / exp | `exp(log u) → u` (side condition `u ≠ 0`), `exp a · exp b → exp(a+b)` |
//! | radical denesting | `√(3 + 2√2) → 1 + √2` |
//! | rational cancellation | `(x²−1)/(x−1) → x + 1` (side condition `x − 1 ≠ 0`) |
//! | bounded expansion | `(x+y)² − (x−y)² → 4xy`, `(√2+1)(√2−1) → 1` |
//!
//! Each candidate is kept **only if it is strictly smaller** under
//! [`expr_size`], so the pass can never make a result worse than the core
//! gave. Every candidate is an identity of the expression it rewrote; the ones
//! that hold only away from a point (cancellation, `exp ∘ log`) record that
//! point as a [`SideCondition::NonZero`] on their step.
//!
//! Cost: one linear scan of the core result decides which passes are
//! triggered; an expression with no trigonometric, exponential, logarithmic,
//! negative-power, radical or product-of-sum node pays only that scan.

use super::engine::{close_pass, simplify, simplify_expanded, simplify_with, SimplifyConfig};
use super::idmap::IdMap;
use super::rules::{FlattenAdd, FlattenMul, NodeKinds, RewriteRule};
use super::rulesets::{log_exp_rules, CoshSinhIdentity, SinCosIdentity};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::{HashMap, HashSet};

/// Rule name of the rational-cancellation step (`cancel` over the free symbols).
pub(crate) const RATIONAL_CANCEL_RULE: &str = "rational_cancel";
/// Rule name of `exp(log u) → u` under `u ≠ 0`.
pub(crate) const EXP_OF_LOG_NONZERO_RULE: &str = "exp_of_log_nonzero";
/// Rule name of the rational-coefficient Pythagorean collection.
pub(crate) const PYTHAGOREAN_RATIONAL_RULE: &str = "pythagorean_rational";
/// Rule name of `√(a + b√c) → √p ± √q`.
pub(crate) const SQRT_DENEST_RULE: &str = "sqrt_denest";

/// Above this many nodes the rational-cancellation pass is not attempted: it
/// converts the whole expression to a rational function, which is the one pass
/// here whose cost is not linear in the input.
const MAX_CANCEL_SIZE: u64 = 400;
/// Above this many distributed products the expansion candidate is not tried:
/// an expansion that large is never going to be the smaller form.
const MAX_EXPAND_PRODUCTS: u64 = 64;
/// … and above this many nodes, likewise.
const MAX_EXPAND_SIZE: u64 = 200;

/// Tree size of `expr`: every node counted once per occurrence (a shared
/// subexpression costs once per use, which is what a reader sees printed).
/// Saturating, and linear in the DAG thanks to the memo.
pub fn expr_size(expr: ExprId, pool: &ExprPool) -> u64 {
    fn go(e: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, u64>) -> u64 {
        if let Some(&s) = memo.get(&e) {
            return s;
        }
        let children: Vec<ExprId> = pool.with(e, children_of);
        let mut s: u64 = 1;
        for c in children {
            s = s.saturating_add(go(c, pool, memo));
        }
        memo.insert(e, s);
        s
    }
    go(expr, pool, &mut HashMap::new())
}

fn children_of(d: &ExprData) -> Vec<ExprId> {
    match d {
        ExprData::Add(a) | ExprData::Mul(a) => a.clone(),
        ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::Piecewise { branches, default } => {
            let mut v: Vec<ExprId> = branches.iter().flat_map(|(c, b)| [*c, *b]).collect();
            v.push(*default);
            v
        }
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => vec![*var, *body],
        ExprData::BigO(i) => vec![*i],
        ExprData::RootSum { poly, var, body } => vec![*poly, *var, *body],
        _ => vec![],
    }
}

/// Which domain passes an expression could benefit from. Each flag is the
/// precondition of some rule in that pass, so a pass is only run when one of
/// its rules could fire.
#[derive(Default, Debug, Clone, Copy)]
struct Triggers {
    /// An even power of `sin`/`cos`/`sinh`/`cosh` (Pythagorean collection).
    pyth: bool,
    /// `tan`, `sin(u)·cos(u)` in one product, or a sum with a product of two
    /// trig factors in it (double-angle / angle-sum rules).
    trig_other: bool,
    /// `log(exp ·)`, two `exp` factors in a product, or an integer power of
    /// an `exp`.
    exp_log: bool,
    /// `exp(log ·)`.
    exp_of_log: bool,
    /// A negative integer power of something that depends on a symbol.
    rational: bool,
    /// `sqrt(a + b·sqrt(c))` with rational `a, b, c`.
    nested_sqrt: bool,
    /// A product containing a sum, or a small integer power of a sum.
    expandable: bool,
    /// Something these passes must not touch (`∞`, `O(·)`, quantifiers,
    /// piecewise, relations, `RootSum`): skip everything.
    opaque: bool,
}

impl Triggers {
    fn any(&self) -> bool {
        !self.opaque
            && (self.pyth
                || self.trig_other
                || self.exp_log
                || self.exp_of_log
                || self.rational
                || self.nested_sqrt
                || self.expandable)
    }
}

/// `Some((name, arg))` for a unary `sin`/`cos`/`tan`/`sinh`/`cosh`/`tanh`.
fn trig_call(e: ExprId, pool: &ExprPool) -> Option<(u8, ExprId)> {
    pool.with(e, |d| match d {
        ExprData::Func { name, args } if args.len() == 1 => {
            let k = match name.as_str() {
                "sin" => 0,
                "cos" => 1,
                "tan" => 2,
                "sinh" => 3,
                "cosh" => 4,
                "tanh" => 5,
                _ => return None,
            };
            Some((k, args[0]))
        }
        _ => None,
    })
}

fn unary_arg(e: ExprId, fname: &str, pool: &ExprPool) -> Option<ExprId> {
    pool.with(e, |d| match d {
        ExprData::Func { name, args } if name == fname && args.len() == 1 => Some(args[0]),
        _ => None,
    })
}

/// Does `e` mention a symbol (memoised)?
fn symbolic(e: ExprId, pool: &ExprPool, memo: &mut IdMap<bool>) -> bool {
    if let Some(&b) = memo.get(&e) {
        return b;
    }
    let (is_sym, kids) = pool.with(e, |d| {
        (matches!(d, ExprData::Symbol { .. }), children_of(d))
    });
    let b = is_sym || kids.into_iter().any(|c| symbolic(c, pool, memo));
    memo.insert(e, b);
    b
}

fn scan(expr: ExprId, pool: &ExprPool) -> Triggers {
    let mut t = Triggers::default();
    let mut seen: IdMap<()> = IdMap::default();
    let mut has_sym: IdMap<bool> = IdMap::default();
    // `sin a·cos b`-style products feed the angle-sum rules only inside a sum.
    let mut two_trig_product = false;
    let mut has_sum = false;
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if seen.insert(e, ()).is_some() {
            continue;
        }
        // Classify the node and push its children without cloning it.
        let opaque = pool.with(e, |d| {
            match d {
                ExprData::Piecewise { .. }
                | ExprData::Forall { .. }
                | ExprData::Exists { .. }
                | ExprData::BigO(_)
                | ExprData::RootSum { .. }
                | ExprData::Predicate { .. } => return true,
                ExprData::Symbol { name, .. } => {
                    return name == crate::kernel::pool::POS_INFINITY_SYMBOL;
                }
                ExprData::Func { name, args } => {
                    match name.as_str() {
                        "tan" | "tanh" => t.trig_other = true,
                        "exp" if args.len() == 1 => {
                            if unary_arg(args[0], "log", pool).is_some() {
                                t.exp_of_log = true;
                            }
                        }
                        "log" if args.len() == 1 => {
                            if unary_arg(args[0], "exp", pool).is_some() {
                                t.exp_log = true;
                            }
                        }
                        "sqrt" if args.len() == 1 => {
                            if pool.with(args[0], |a| matches!(a, ExprData::Add(_)))
                                && !symbolic(args[0], pool, &mut has_sym)
                            {
                                t.nested_sqrt = true;
                            }
                        }
                        _ => {}
                    }
                    stack.extend_from_slice(args);
                }
                ExprData::Pow { base, exp } => {
                    let int_exp = pool.with(*exp, |x| match x {
                        ExprData::Integer(n) => n.0.to_i64(),
                        _ => None,
                    });
                    if let Some(k) = int_exp {
                        if k < 0 && symbolic(*base, pool, &mut has_sym) {
                            t.rational = true;
                        }
                        if (2..=6).contains(&k)
                            && pool.with(*base, |b| matches!(b, ExprData::Add(_)))
                        {
                            t.expandable = true;
                        }
                        if k >= 2
                            && k % 2 == 0
                            && matches!(trig_call(*base, pool), Some((0 | 1 | 3 | 4, _)))
                        {
                            t.pyth = true;
                        }
                        if unary_arg(*base, "exp", pool).is_some() {
                            t.exp_log = true;
                        }
                    }
                    stack.push(*base);
                    stack.push(*exp);
                }
                ExprData::Mul(args) => {
                    // One probe per factor: is it a sum, `sin u`, `cos u`, `exp ·`?
                    let mut sin_args: Vec<ExprId> = Vec::new();
                    let mut cos_args: Vec<ExprId> = Vec::new();
                    let mut exps = 0;
                    for &a in args {
                        pool.with(a, |x| match x {
                            ExprData::Add(_) => t.expandable = true,
                            ExprData::Func { name, args: fa } if fa.len() == 1 => {
                                match name.as_str() {
                                    "sin" => sin_args.push(fa[0]),
                                    "cos" => cos_args.push(fa[0]),
                                    "exp" => exps += 1,
                                    _ => {}
                                }
                            }
                            _ => {}
                        });
                    }
                    if sin_args.iter().any(|u| cos_args.contains(u)) {
                        t.trig_other = true;
                    }
                    if sin_args.len() + cos_args.len() >= 2 {
                        two_trig_product = true;
                    }
                    if exps >= 2 {
                        t.exp_log = true;
                    }
                    stack.extend_from_slice(args);
                }
                ExprData::Add(args) => {
                    has_sum = true;
                    stack.extend_from_slice(args);
                }
                _ => {}
            }
            false
        });
        if opaque {
            t.opaque = true;
            return t;
        }
    }
    if two_trig_product && has_sum {
        t.trig_other = true;
    }
    t
}

fn is_numeric_literal(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| {
        matches!(d, ExprData::Integer(_) | ExprData::Rational(_))
    })
}

fn exact_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(rug::Rational::from(&n.0)),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    })
}

/// Rough count of the products a full expansion of `expr` would form.
fn expansion_products(expr: ExprId, pool: &ExprPool) -> u64 {
    fn go(e: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, u64>) -> u64 {
        if let Some(&v) = memo.get(&e) {
            return v;
        }
        let v = match pool.get(e) {
            ExprData::Add(args) => args
                .iter()
                .fold(0u64, |acc, &a| acc.saturating_add(go(a, pool, memo))),
            ExprData::Mul(args) => args
                .iter()
                .fold(1u64, |acc, &a| acc.saturating_mul(go(a, pool, memo).max(1))),
            ExprData::Pow { base, exp } => {
                let b = go(base, pool, memo).max(1);
                match pool.get(exp) {
                    ExprData::Integer(n) if n.0 >= 2 => {
                        n.0.to_u32()
                            .and_then(|k| b.checked_pow(k))
                            .unwrap_or(u64::MAX)
                    }
                    _ => 1,
                }
            }
            _ => 1,
        };
        memo.insert(e, v);
        v
    }
    go(expr, pool, &mut HashMap::new())
}

// ---------------------------------------------------------------------------
// Rules used only by this pass
// ---------------------------------------------------------------------------

/// `exp(log u) → u`, recording `u ≠ 0`.
///
/// Holds on the principal branch for every non-zero complex `u`
/// (`exp(log u) = exp(ln|u| + i·Arg u) = |u|·e^{i·Arg u} = u`); `log u` is
/// undefined at `u = 0`, which is the only point excluded. Not the
/// assumption-gated `exp_of_log` (which needs `u > 0` so its Lean certificate
/// can use `Real.exp_log`): this one has no real-analysis certificate — over ℝ
/// Mathlib's `Real.log` of a negative number is `log |u|` — and is withheld.
struct ExpOfLogNonZero;

impl RewriteRule for ExpOfLogNonZero {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::FUNC
    }
    fn name(&self) -> &'static str {
        EXP_OF_LOG_NONZERO_RULE
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = pool.with(expr, |d| match d {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => Some(args[0]),
            _ => None,
        })?;
        let u = pool.with(arg, |d| match d {
            ExprData::Func { name, args } if name == "log" && args.len() == 1 => Some(args[0]),
            _ => None,
        })?;
        if pool.has_non_finite(u) {
            return None;
        }
        // When `u > 0` is known (a `Domain.Positive` symbol, an ambient
        // assumption) this is the library's assumption-gated `exp_of_log`,
        // whose positivity-hypothesis certificate (`Real.exp_log`) applies;
        // record it as that.
        let (name, cond) = match super::assumptions::assumed_sign(u, pool) {
            Some(super::assumptions::Sign::Positive) => ("exp_of_log", SideCondition::Positive(u)),
            _ => (self.name(), SideCondition::NonZero(u)),
        };
        let mut log = DerivationLog::new();
        log.push(RewriteStep::with_conditions(name, expr, u, vec![cond]));
        Some((u, log))
    }
}

/// `√(a + b·√c) → √((a+d)/2) + sgn(b)·√((a−d)/2)` when `d² = a² − b²c` for a
/// rational `d ≥ 0`, with `a, b, c` rational, `c > 0`, `a > 0`.
///
/// Squaring the right-hand side gives `a + 2·sgn(b)·√((a²−d²)/4) = a + b√c`,
/// and both sides are non-negative (`a > |b|√c` follows from `d² ≥ 0`), so the
/// principal square roots agree.
struct SqrtDenest;

impl SqrtDenest {
    /// `term` as `(b, c)` with `term = b·√c`, `b, c` rational, `c > 0`.
    fn as_surd(term: ExprId, pool: &ExprPool) -> Option<(rug::Rational, rug::Rational)> {
        let sqrt_arg = |e: ExprId| -> Option<rug::Rational> {
            let inner = pool.with(e, |d| match d {
                ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => Some(args[0]),
                ExprData::Pow { base, exp }
                    if pool.with(*exp, |x| matches!(x, ExprData::Rational(r) if r.0 == rug::Rational::from((1, 2)))) =>
                {
                    Some(*base)
                }
                _ => None,
            })?;
            exact_rational(inner, pool).filter(|c| *c > 0)
        };
        if let Some(c) = sqrt_arg(term) {
            return Some((rug::Rational::from(1), c));
        }
        let factors = pool.with(term, |d| match d {
            ExprData::Mul(a) if a.len() == 2 => Some(a.clone()),
            _ => None,
        })?;
        let (k, s) = if is_numeric_literal(factors[0], pool) {
            (factors[0], factors[1])
        } else {
            (factors[1], factors[0])
        };
        Some((exact_rational(k, pool)?, sqrt_arg(s)?))
    }

    fn rational_sqrt(q: &rug::Rational) -> Option<rug::Rational> {
        if *q < 0 {
            return None;
        }
        let n = q.numer().clone();
        let d = q.denom().clone();
        if !n.is_perfect_square() || !d.is_perfect_square() {
            return None;
        }
        Some(rug::Rational::from((n.sqrt(), d.sqrt())))
    }

    fn intern(q: rug::Rational, pool: &ExprPool) -> ExprId {
        if *q.denom() == 1 {
            pool.integer(q.numer().clone())
        } else {
            let (n, d) = q.into_numer_denom();
            pool.rational(n, d)
        }
    }
}

impl RewriteRule for SqrtDenest {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::FUNC
    }
    fn name(&self) -> &'static str {
        SQRT_DENEST_RULE
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = pool.with(expr, |d| match d {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => Some(args[0]),
            _ => None,
        })?;
        let terms = pool.with(arg, |d| match d {
            ExprData::Add(a) if a.len() == 2 => Some(a.clone()),
            _ => None,
        })?;
        let (a, (b, c)) = if let Some(a) = exact_rational(terms[0], pool) {
            (a, Self::as_surd(terms[1], pool)?)
        } else {
            (
                exact_rational(terms[1], pool)?,
                Self::as_surd(terms[0], pool)?,
            )
        };
        if a <= 0 || b == 0 {
            return None;
        }
        let disc = rug::Rational::from(&a * &a) - rug::Rational::from(&b * &b) * &c;
        let d = Self::rational_sqrt(&disc)?;
        let half = rug::Rational::from((1, 2));
        let p = rug::Rational::from(&a + &d) * &half;
        let q = rug::Rational::from(&a - &d) * &half;
        if q < 0 {
            return None;
        }
        let sp = pool.func("sqrt", vec![Self::intern(p, pool)]);
        let sq = pool.func("sqrt", vec![Self::intern(q, pool)]);
        let sq = if b < 0 {
            pool.mul(vec![pool.integer(-1_i32), sq])
        } else {
            sq
        };
        let after = pool.add(vec![sp, sq]);
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple(self.name(), expr, after));
        Some((after, log))
    }
}

// ---------------------------------------------------------------------------
// Passes
// ---------------------------------------------------------------------------

fn run_rules(expr: ExprId, pool: &ExprPool, rules: &[Box<dyn RewriteRule>]) -> DerivedExpr<ExprId> {
    close_pass(
        simplify_with(expr, pool, rules, SimplifyConfig::default()),
        pool,
    )
}

/// `c·r·sin²u + c·r·cos²u → c·r` and `c·r·cosh²u − c·r·sinh²u → c·r` for an
/// *exact rational* `c`.
///
/// [`CoshSinhIdentity`] matches coefficients as factor multisets, as the
/// Python operators spell a difference (`2·cosh² + 2·(−1)·sinh²`); once the
/// core has folded the sinh coefficient to `−2` that spelling is gone and the
/// pair no longer matched. This compares the folded rationals instead.
struct PythagoreanRational;

/// `(kind, u, c, rest)` for `term = c · rest · f(u)²`, with `kind` 0..=3 for
/// `sin`, `cos`, `sinh`, `cosh` and `rest` sorted.
type SquareTerm = (u8, ExprId, rug::Rational, Vec<ExprId>);

impl PythagoreanRational {
    fn split(term: ExprId, pool: &ExprPool) -> Option<SquareTerm> {
        let factors = pool.with(term, |d| match d {
            ExprData::Mul(a) => a.clone(),
            _ => vec![term],
        });
        let mut c = rug::Rational::from(1);
        let mut found = None;
        let mut rest = Vec::new();
        for f in factors {
            if let Some(q) = exact_rational(f, pool) {
                c *= q;
                continue;
            }
            let sq = pool.with(f, |d| match d {
                ExprData::Pow { base, exp }
                    if pool.with(*exp, |x| matches!(x, ExprData::Integer(n) if n.0 == 2)) =>
                {
                    match trig_call(*base, pool) {
                        Some((0, u)) => Some((0u8, u)),
                        Some((1, u)) => Some((1, u)),
                        Some((3, u)) => Some((2, u)),
                        Some((4, u)) => Some((3, u)),
                        _ => None,
                    }
                }
                _ => None,
            });
            match sq {
                Some(k) if found.is_none() => found = Some(k),
                _ => rest.push(f),
            }
        }
        let (kind, u) = found?;
        rest.sort();
        Some((kind, u, c, rest))
    }
}

impl RewriteRule for PythagoreanRational {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::ADD
    }
    fn name(&self) -> &'static str {
        PYTHAGOREAN_RATIONAL_RULE
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = pool.with(expr, |d| match d {
            ExprData::Add(a) if a.len() >= 2 => Some(a.clone()),
            _ => None,
        })?;
        let parts: Vec<Option<SquareTerm>> = args.iter().map(|&a| Self::split(a, pool)).collect();
        for (i, pi) in parts.iter().enumerate() {
            let Some((ki, ui, ci, ri)) = pi else { continue };
            // The partner's kind, and the coefficient it must carry.
            let (kj, cj) = match ki {
                0 => (1, ci.clone()),
                2 => (3, rug::Rational::from(-ci)),
                _ => continue,
            };
            let Some(j) = parts.iter().position(
                |pj| matches!(pj, Some((k, u, c, r)) if *k == kj && u == ui && *c == cj && r == ri),
            ) else {
                continue;
            };
            // sin² + cos² → c·rest; −c·sinh² + c·cosh² → c·rest (the cosh one).
            let coeff = cj;
            let mut factors = ri.clone();
            if coeff != 1 {
                let lit = if *coeff.denom() == 1 {
                    pool.integer(coeff.numer().clone())
                } else {
                    let (n, d) = coeff.into_numer_denom();
                    pool.rational(n, d)
                };
                factors.insert(0, lit);
            }
            let replacement = match factors.len() {
                0 => pool.integer(1_i32),
                1 => factors[0],
                _ => pool.mul(factors),
            };
            let mut new_args: Vec<ExprId> = args
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != i && k != j)
                .map(|(_, &a)| a)
                .collect();
            new_args.push(replacement);
            let after = if new_args.len() == 1 {
                new_args[0]
            } else {
                pool.add(new_args)
            };
            let mut log = DerivationLog::new();
            log.push(RewriteStep::simple(self.name(), expr, after));
            return Some((after, log));
        }
        None
    }
}

fn pythagorean_pass(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    static RULES: std::sync::OnceLock<Vec<Box<dyn RewriteRule>>> = std::sync::OnceLock::new();
    let rules = RULES.get_or_init(|| {
        vec![
            Box::new(FlattenMul),
            Box::new(FlattenAdd),
            Box::new(SinCosIdentity),
            Box::new(CoshSinhIdentity),
            Box::new(PythagoreanRational),
        ]
    });
    run_rules(expr, pool, rules)
}

fn exp_log_pass(expr: ExprId, pool: &ExprPool, with_exp_of_log: bool) -> DerivedExpr<ExprId> {
    let mut rules = log_exp_rules();
    if with_exp_of_log {
        rules.insert(0, Box::new(ExpOfLogNonZero));
    }
    run_rules(expr, pool, &rules)
}

fn denest_pass(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    let rules: Vec<Box<dyn RewriteRule>> = vec![Box::new(SqrtDenest)];
    run_rules(expr, pool, &rules)
}

/// Bases of the negative integer powers in `expr` that depend on a symbol —
/// the factors whose vanishing the cancellation step must exclude.
fn denominators(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        let data = pool.get(e);
        if let ExprData::Pow { base, exp } = &data {
            if pool.with(*exp, |d| matches!(d, ExprData::Integer(n) if n.0 < 0))
                && !crate::poly::collect_free_vars(*base, pool).is_empty()
                && !out.contains(base)
            {
                out.push(*base);
            }
        }
        stack.extend(children_of(&data));
    }
    out
}

/// What shape of rational expression `expr` is, for [`cancel_pass`].
enum RationalShape {
    /// `n₁·n₂·…·d₁^(−k₁)·…` at the root: numerator and denominator factors.
    Quotient(Vec<ExprId>, Vec<ExprId>),
    /// Fractions to combine: several distinct denominators, or a fraction
    /// inside a denominator (`1/(1/x + 1/y)`).
    Compound,
    /// Nothing `cancel` could shrink.
    Neither,
}

fn rational_shape(expr: ExprId, pool: &ExprPool) -> RationalShape {
    let neg_pow_base = |e: ExprId| {
        pool.with(e, |d| match d {
            ExprData::Pow { base, exp }
                if pool.with(*exp, |x| matches!(x, ExprData::Integer(n) if n.0 < 0)) =>
            {
                Some(*base)
            }
            _ => None,
        })
    };
    // Fractions to combine: a sum with two or more fractional terms
    // (`1/x + 1/y`), or a fraction inside a denominator (`1/(1/x + 1/y)`). A
    // *product* of reciprocals (`1/((x+1)·log x)`) is not one: combining it
    // only multiplies the denominator out, which hides the factors the
    // integrator and the product engine look for.
    let is_fraction = |t: ExprId| {
        neg_pow_base(t).is_some()
            || pool.with(t, |d| match d {
                ExprData::Mul(fs) => fs.iter().any(|&f| neg_pow_base(f).is_some()),
                _ => false,
            })
    };
    let mut seen = HashSet::new();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        let data = pool.get(e);
        match &data {
            ExprData::Add(terms) if terms.iter().filter(|&&t| is_fraction(t)).count() >= 2 => {
                return RationalShape::Compound;
            }
            ExprData::Pow { base, .. }
                if neg_pow_base(e).is_some() && !denominators(*base, pool).is_empty() =>
            {
                return RationalShape::Compound;
            }
            _ => {}
        }
        stack.extend(children_of(&data));
    }
    let Some(factors) = pool.with(expr, |d| match d {
        ExprData::Mul(a) => Some(a.clone()),
        _ => None,
    }) else {
        return RationalShape::Neither;
    };
    let mut num = Vec::new();
    let mut den = Vec::new();
    for f in factors {
        match neg_pow_base(f) {
            Some(b) => {
                let k = pool.with(f, |d| match d {
                    ExprData::Pow { exp, .. } => *exp,
                    _ => unreachable!(),
                });
                let pos = pool.with(k, |d| match d {
                    ExprData::Integer(n) => pool.integer(-n.0.clone()),
                    _ => unreachable!(),
                });
                den.push(pool.pow(b, pos));
            }
            None => num.push(f),
        }
    }
    // A sum in the denominator is what a common factor can cancel against;
    // a monomial denominator (`x⁻²·(x³ + x)`) is the core's business.
    let den_has_sum = den.iter().any(|&d| {
        pool.with(d, |x| match x {
            ExprData::Pow { base, .. } => pool.with(*base, |b| matches!(b, ExprData::Add(_))),
            ExprData::Add(_) => true,
            _ => false,
        })
    });
    if den.is_empty() || !den_has_sum {
        return RationalShape::Neither;
    }
    RationalShape::Quotient(num, den)
}

/// A lower bound on [`expr_size`] of `p` once built: one node per term of
/// degree ≤ 1 (`x`, `c`), three per higher power (`x^k`), plus the `Add`.
/// Coefficients are not counted: `pseudo_divrem` scales them.
fn poly_size_lower_bound(p: &crate::poly::UniPoly) -> u64 {
    let mut terms = 0u64;
    let mut size = 0u64;
    for (k, c) in p.coefficients().iter().enumerate() {
        if *c == 0 {
            continue;
        }
        terms += 1;
        size += if k <= 1 { 1 } else { 3 };
    }
    size + u64::from(terms > 1)
}

/// `N / D` with `gcd(N, D)` divided out — one polynomial gcd — or `None` when
/// the gcd is a constant (nothing cancels), the result cannot be smaller than
/// `max_size`, or the factors are not polynomials in `vars`.
fn cancel_quotient(
    num: Vec<ExprId>,
    den: Vec<ExprId>,
    vars: Vec<ExprId>,
    max_size: u64,
    pool: &ExprPool,
) -> Option<ExprId> {
    let product = |v: Vec<ExprId>| match v.len() {
        0 => pool.integer(1_i32),
        1 => v[0],
        _ => pool.mul(v),
    };
    let n_expr = product(num);
    let d_expr = product(den);
    // One variable: decide with FLINT's univariate gcd (about a microsecond)
    // whether anything cancels before paying for the multivariate route.
    if let [v] = vars[..] {
        if let (Ok(n), Ok(d)) = (
            crate::poly::UniPoly::from_symbolic(n_expr, v, pool),
            crate::poly::UniPoly::from_symbolic(d_expr, v, pool),
        ) {
            let g = n.gcd(&d).filter(|g| g.degree() > 0)?;
            // Something cancels. Size the quotient before building it: a
            // sparse numerator over a factor of it is usually *dense*
            // (`(x¹⁰ − 1)/(x − 1)` is ten terms), and then not simpler.
            let (qn, _) = n.pseudo_divrem(&g)?;
            let (qd, _) = d.pseudo_divrem(&g)?;
            let mut est = poly_size_lower_bound(&qn);
            if qd.degree() > 0 {
                est += poly_size_lower_bound(&qd) + 3;
            }
            if est >= max_size {
                return None;
            }
        }
    }
    let d0 = crate::poly::MultiPoly::from_symbolic(d_expr, vars.clone(), pool).ok()?;
    let rf = crate::poly::RationalFunction::from_symbolic(n_expr, d_expr, vars, pool).ok()?;
    if rf.denom.total_degree() == d0.total_degree() {
        return None;
    }
    let numer = rf.numer.to_expr(pool);
    let denom = rf.denom.to_expr(pool);
    if pool.with(denom, |d| matches!(d, ExprData::Integer(n) if n.0 == 1)) {
        return Some(numer);
    }
    Some(pool.mul(vec![numer, pool.pow(denom, pool.integer(-1_i32))]))
}

fn cancel_pass(expr: ExprId, max_size: u64, pool: &ExprPool) -> Option<DerivedExpr<ExprId>> {
    let vars = crate::poly::collect_free_vars(expr, pool);
    if vars.is_empty() {
        return None;
    }
    // `cancel` normalises every intermediate sum and product by a gcd, which
    // is ~100 µs even for `(x²−1)/(x−1)`. A plain quotient — the common case —
    // needs exactly one gcd, and none at all when its denominator has no sum
    // in it; only fractions that must be combined take the general route.
    let cancelled = match rational_shape(expr, pool) {
        RationalShape::Neither => return None,
        RationalShape::Quotient(num, den) => cancel_quotient(num, den, vars, max_size, pool)?,
        RationalShape::Compound => crate::poly::cancel(expr, vars, pool).ok()?,
    };
    if cancelled == expr {
        return None;
    }
    let conditions = denominators(expr, pool)
        .into_iter()
        .map(SideCondition::NonZero)
        .collect();
    let mut log = DerivationLog::new();
    log.push(RewriteStep::with_conditions(
        RATIONAL_CANCEL_RULE,
        expr,
        cancelled,
        conditions,
    ));
    let rest = simplify(cancelled, pool);
    Some(DerivedExpr::with_log(rest.value, log.merge(rest.log)))
}

/// How deeply radicals nest in `expr`: `sqrt(3 + 2·sqrt(2))` is 2.
fn radical_nesting(expr: ExprId, pool: &ExprPool) -> u32 {
    fn go(e: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, u32>) -> u32 {
        if let Some(&v) = memo.get(&e) {
            return v;
        }
        let (radical, kids) = pool.with(e, |d| {
            let radical = match d {
                ExprData::Func { name, .. } => name == "sqrt" || name == "cbrt",
                ExprData::Pow { exp, .. } => {
                    pool.with(*exp, |x| matches!(x, ExprData::Rational(_)))
                }
                _ => false,
            };
            (radical, children_of(d))
        });
        let inner = kids
            .into_iter()
            .map(|c| go(c, pool, memo))
            .max()
            .unwrap_or(0);
        let v = inner + u32::from(radical);
        memo.insert(e, v);
        v
    }
    go(expr, pool, &mut HashMap::new())
}

/// The cost a candidate must beat: size first, then radical nesting (so
/// `sqrt(5 − 2·sqrt(6)) → sqrt(3) − sqrt(2)`, the same size, still counts as
/// simpler).
fn cost(expr: ExprId, pool: &ExprPool) -> (u64, u32) {
    (expr_size(expr, pool), radical_nesting(expr, pool))
}

/// Replace `cur` by `cand` when `cand` is strictly cheaper.
fn consider(
    cur: &mut DerivedExpr<ExprId>,
    cur_cost: &mut (u64, u32),
    cand: Option<DerivedExpr<ExprId>>,
    pool: &ExprPool,
) {
    let Some(cand) = cand else { return };
    if cand.value == cur.value {
        return;
    }
    let c = cost(cand.value, pool);
    if c < *cur_cost {
        *cur_cost = c;
        let log = std::mem::take(&mut cur.log).merge(cand.log);
        *cur = DerivedExpr::with_log(cand.value, log);
    }
}

/// Run the domain passes on an already-simplified `result`, keeping each
/// candidate only when it is strictly smaller. See the [module docs](self).
pub fn simplify_domain_passes(result: DerivedExpr<ExprId>, pool: &ExprPool) -> DerivedExpr<ExprId> {
    // A polynomial in plain symbols triggers no pass (no function, no sum
    // under a product or power): answered in O(1) from the pool's cached
    // screening bits, so the common case does not pay for the scan.
    if pool.is_plain_polynomial(result.value) {
        return result;
    }
    let triggers = scan(result.value, pool);
    if !triggers.any() {
        return result;
    }
    let mut cur = result;
    let mut cur_cost = cost(cur.value, pool);
    let budget_ok = || crate::budget::check().is_ok();

    if triggers.pyth && budget_ok() {
        let c = pythagorean_pass(cur.value, pool);
        consider(&mut cur, &mut cur_cost, Some(c), pool);
    }
    if (triggers.pyth || triggers.trig_other) && budget_ok() {
        let c = super::engine::simplify_trig(cur.value, pool);
        consider(&mut cur, &mut cur_cost, Some(c), pool);
    }
    if (triggers.exp_log || triggers.exp_of_log) && budget_ok() {
        let c = exp_log_pass(cur.value, pool, triggers.exp_of_log);
        consider(&mut cur, &mut cur_cost, Some(c), pool);
    }
    if triggers.nested_sqrt && budget_ok() {
        let c = denest_pass(cur.value, pool);
        consider(&mut cur, &mut cur_cost, Some(c), pool);
    }
    // The remaining passes look at the expression as a whole, so re-scan what
    // the passes above left.
    let t = scan(cur.value, pool);
    if t.opaque {
        return cur;
    }
    if t.rational && cur_cost.0 <= MAX_CANCEL_SIZE && budget_ok() {
        let c = cancel_pass(cur.value, cur_cost.0, pool);
        consider(&mut cur, &mut cur_cost, c, pool);
    }
    if t.expandable
        && cur_cost.0 <= MAX_EXPAND_SIZE
        && expansion_products(cur.value, pool) <= MAX_EXPAND_PRODUCTS
        && budget_ok()
    {
        // Only a fraction-free expansion counts: distributing a quotient
        // (`(k²−1)·k⁻² → 1 − k⁻²`) is smaller but turns one fraction into a
        // sum of them, which no consumer of a factored quotient recognises.
        let c = simplify_expanded(cur.value, pool);
        if denominators(c.value, pool).is_empty() {
            consider(&mut cur, &mut cur_cost, Some(c), pool);
        }
    }
    cur
}

/// [`simplify`] followed by [`simplify_domain_passes`]: the simplest form this
/// library can find, never larger than what `simplify` returns.
///
/// This is what Python's `alkahest.simplify` calls. Internal callers that
/// simplify their own intermediate results should keep calling
/// [`simplify`], which is cheaper and keeps the spellings they rely on.
pub fn simplify_full(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    simplify_domain_passes(simplify(expr, pool), pool)
}

/// The first `0^k` with a negative exact exponent in `expr` — a division by an
/// exact zero, which denotes no number — searching through sums, products,
/// powers and function arguments.
pub fn find_division_by_zero(expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let mut seen = HashSet::new();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        let kids = pool.with(e, |d| match d {
            ExprData::Pow { base, exp } => {
                let zero_base = pool.with(*base, |b| match b {
                    ExprData::Integer(n) => n.0 == 0,
                    ExprData::Rational(r) => r.0 == 0,
                    _ => false,
                });
                let neg_exp = pool.with(*exp, |x| match x {
                    ExprData::Integer(n) => n.0 < 0,
                    ExprData::Rational(r) => r.0 < 0,
                    _ => false,
                });
                if zero_base && neg_exp {
                    None
                } else {
                    Some(vec![*base, *exp])
                }
            }
            ExprData::Add(a) | ExprData::Mul(a) => Some(a.clone()),
            ExprData::Func { args, .. } => Some(args.clone()),
            _ => Some(vec![]),
        });
        match kids {
            None => return Some(e),
            Some(k) => stack.extend(k),
        }
    }
    None
}

/// Rule name of `0^r → 0` / `sqrt(0) → 0` for a positive exponent.
pub(crate) const ZERO_RADICAL_RULE: &str = "zero_radical";

/// `0^r → 0` for a positive rational `r`, and `sqrt(0) → 0` / `cbrt(0) → 0`.
///
/// Not in the default rule set on purpose: the series and limit engines rely
/// on `sqrt(0)` (and its reciprocal) surviving as a recognisable "not a value"
/// marker while they work. Producers that hand a *final* closed value to the
/// user — the definite-integral evaluator, whose `F(b) − F(a)` used to print
/// as `2 + (−2·0^(1/2))` — run it once their own gates have passed.
struct ZeroRadical;

impl RewriteRule for ZeroRadical {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::FUNC.or(NodeKinds::POW)
    }
    fn name(&self) -> &'static str {
        ZERO_RADICAL_RULE
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let is_zero = |e: ExprId| pool.with(e, |d| matches!(d, ExprData::Integer(n) if n.0 == 0));
        let fires = pool.with(expr, |d| match d {
            ExprData::Func { name, args } => {
                (name == "sqrt" || name == "cbrt") && args.len() == 1 && is_zero(args[0])
            }
            ExprData::Pow { base, exp } => {
                is_zero(*base) && pool.with(*exp, |x| matches!(x, ExprData::Rational(r) if r.0 > 0))
            }
            _ => false,
        });
        if !fires {
            return None;
        }
        let zero = pool.integer(0_i32);
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple(self.name(), expr, zero));
        Some((zero, log))
    }
}

/// Fold `0^r` (`r > 0`), `sqrt(0)` and `cbrt(0)` to `0` and re-simplify — for a
/// closed *final* value only; see [`ZeroRadical`]. `None` when there is
/// nothing to fold.
pub(crate) fn fold_zero_radicals(expr: ExprId, pool: &ExprPool) -> Option<DerivedExpr<ExprId>> {
    let rules: Vec<Box<dyn RewriteRule>> = vec![Box::new(ZeroRadical)];
    let r = run_rules(expr, pool, &rules);
    (r.value != expr).then_some(r)
}

/// True when `a − a = 0` is an identity for this `a`: no `∞` or non-finite
/// float in it (`∞ − ∞` is indeterminate), no `O(·)` term (`O(x) − O(x)` is
/// `O(x)`), and no division by an exact zero (an undefined value minus itself
/// is still undefined). Used to fold `a - a` when it is written.
pub fn self_difference_is_zero(a: ExprId, pool: &ExprPool) -> bool {
    if pool.has_non_finite(a) || find_division_by_zero(a, pool).is_some() {
        return false;
    }
    let mut seen = HashSet::new();
    let mut stack = vec![a];
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        let data = pool.get(e);
        if matches!(data, ExprData::BigO(_)) {
            return false;
        }
        stack.extend(children_of(&data));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        (pool, x, y)
    }

    fn sq(pool: &ExprPool, e: ExprId) -> ExprId {
        pool.pow(e, pool.integer(2_i32))
    }

    fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
        pool.mul(vec![pool.integer(-1_i32), e])
    }

    #[test]
    fn pythagorean_identity_folds() {
        let (pool, x, y) = setup();
        let s = pool.func("sin", vec![x]);
        let c = pool.func("cos", vec![x]);
        let e = pool.add(vec![sq(&pool, s), sq(&pool, c)]);
        assert_eq!(simplify_full(e, &pool).value, pool.integer(1_i32));
        let e = pool.add(vec![y, sq(&pool, s), sq(&pool, c)]);
        let r = simplify_full(e, &pool).value;
        assert_eq!(
            r,
            simplify(pool.add(vec![y, pool.integer(1_i32)]), &pool).value
        );
    }

    #[test]
    fn hyperbolic_identity_folds() {
        let (pool, x, _) = setup();
        let ch = pool.func("cosh", vec![x]);
        let sh = pool.func("sinh", vec![x]);
        let e = pool.add(vec![sq(&pool, ch), neg(&pool, sq(&pool, sh))]);
        assert_eq!(simplify_full(e, &pool).value, pool.integer(1_i32));
        // With the coefficients already folded (`2·cosh² + (−2)·sinh²`).
        let e = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), sq(&pool, ch)]),
            pool.mul(vec![pool.integer(-2_i32), sq(&pool, sh)]),
        ]);
        assert_eq!(simplify_full(e, &pool).value, pool.integer(2_i32));
    }

    #[test]
    fn exp_of_log_records_nonzero() {
        let (pool, x, _) = setup();
        let e = pool.func("exp", vec![pool.func("log", vec![x])]);
        let r = simplify_full(e, &pool);
        assert_eq!(r.value, x);
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == EXP_OF_LOG_NONZERO_RULE
                && s.side_conditions == vec![SideCondition::NonZero(x)]));
        // A positive argument is the certified `exp_of_log` under `u > 0`.
        let xp = pool.symbol("xp", Domain::Positive);
        let r = simplify_full(pool.func("exp", vec![pool.func("log", vec![xp])]), &pool);
        assert_eq!(r.value, xp);
        assert!(r.log.steps().iter().any(|s| s.rule_name == "exp_of_log"
            && s.side_conditions == vec![SideCondition::Positive(xp)]));
    }

    #[test]
    fn rational_cancel_records_denominator() {
        let (pool, x, _) = setup();
        let m1 = pool.integer(-1_i32);
        let num = pool.add(vec![sq(&pool, x), m1]);
        let den = pool.add(vec![x, m1]);
        let e = pool.mul(vec![num, pool.pow(den, m1)]);
        let r = simplify_full(e, &pool);
        assert_eq!(r.value, pool.add(vec![x, pool.integer(1_i32)]));
        let step = r
            .log
            .steps()
            .iter()
            .find(|s| s.rule_name == RATIONAL_CANCEL_RULE)
            .expect("cancel step");
        assert_eq!(step.side_conditions, vec![SideCondition::NonZero(den)]);
    }

    #[test]
    fn conjugate_surds_multiply_out() {
        let (pool, _, _) = setup();
        let r2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let e = pool.mul(vec![
            pool.add(vec![r2, pool.integer(1_i32)]),
            pool.add(vec![r2, pool.integer(-1_i32)]),
        ]);
        assert_eq!(simplify_full(e, &pool).value, pool.integer(1_i32));
    }

    #[test]
    fn nested_sqrt_denests() {
        let (pool, _, _) = setup();
        let r2 = pool.func("sqrt", vec![pool.integer(2_i32)]);
        let e = pool.func(
            "sqrt",
            vec![pool.add(vec![
                pool.integer(3_i32),
                pool.mul(vec![pool.integer(2_i32), r2]),
            ])],
        );
        let r = simplify_full(e, &pool).value;
        assert_eq!(
            r,
            simplify(pool.add(vec![pool.integer(1_i32), r2]), &pool).value
        );
    }

    #[test]
    fn never_larger_than_simplify() {
        let (pool, x, _) = setup();
        // (x^10 - 1)/(x - 1) cancels to a 10-term sum: larger, so kept as is.
        let m1 = pool.integer(-1_i32);
        let num = pool.add(vec![pool.pow(x, pool.integer(10_i32)), m1]);
        let e = pool.mul(vec![num, pool.pow(pool.add(vec![x, m1]), m1)]);
        let base = simplify(e, &pool).value;
        assert_eq!(simplify_full(e, &pool).value, base);
    }

    #[test]
    fn division_by_zero_is_found() {
        let (pool, x, _) = setup();
        let z = pool.pow(pool.integer(0_i32), pool.integer(-1_i32));
        assert_eq!(find_division_by_zero(pool.mul(vec![x, z]), &pool), Some(z));
        assert_eq!(find_division_by_zero(x, &pool), None);
    }
}
