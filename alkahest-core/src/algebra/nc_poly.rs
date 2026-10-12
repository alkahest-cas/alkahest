//! Noncommutative polynomial normal forms (G10c).
//!
//! An expression over non-commutative symbols (`ExprPool::symbol_commutative(…,
//! false)`) is read as a sum of terms `c · w`, where `c` is a *commutative*
//! coefficient (any expression free of non-commutative symbols) and `w` is a
//! *word*: an ordered list of non-commutative atoms. Products distribute,
//! non-negative integer powers of non-commutative factors unroll
//! (`(σx + σy)² = σx² + σxσy + σyσx + σy²`), and equal words collect.
//!
//! On top of that representation sit word-rewriting systems that run to a
//! fixpoint under a step cap and the active [`crate::budget::Budget`]:
//!
//! * user rules `lhs_word → rhs` ([`nc_rewrite`]) — e.g. `X·P → P·X + iħ`;
//! * the Pauli algebra ([`pauli_reduce`]): `σᵢ² = 1`, `σᵢσⱼ = iεᵢⱼₖσₖ`;
//! * bosonic and fermionic ladder operators ([`normal_order`],
//!   [`vacuum_expectation`]): `[a, a†] = 1`, `{c, c†} = 1`, `c² = 0`.
//!
//! Dirac traces are computed directly ([`gamma_trace`]).
//!
//! Every function either returns an expression equal to its input (as an
//! element of the free algebra modulo the stated relations) or refuses with an
//! `E-NC-*` code; nothing is ever truncated silently.

use crate::budget::{self, BudgetError};
use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::{simplify_with, SimplifyConfig};
use crate::simplify::rules::RewriteRule;
use std::collections::HashMap;
use std::fmt;

/// Default cap on rewrite steps for [`nc_rewrite`], [`pauli_reduce`] and
/// [`normal_order`].
pub const DEFAULT_MAX_STEPS: usize = 200_000;

/// Cap on the number of terms an expansion may hold at once.
const MAX_TERMS: usize = 200_000;

/// Largest non-negative integer power of a non-commutative factor that is
/// unrolled into a product.
const MAX_NC_POWER: i64 = 512;

/// Rounds of "apply the rules, then expand" in [`simplify_with_noncommutative`].
const MAX_FIXPOINT_ROUNDS: usize = 64;

/// Largest number of γ slots (after rewriting a symbolic γ⁵ as `iγ⁰γ¹γ²γ³`)
/// that [`gamma_trace`] expands: the trace of `2k` slots has `(2k−1)!!` terms.
const MAX_GAMMA_SLOTS: usize = 14;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Refusals from the noncommutative-algebra routines (`E-NC-*`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NcError {
    /// Rewriting did not reach a normal form within `limit` steps
    /// (`E-NC-001`). The rule set may not terminate (e.g. `A·B → B·A` with
    /// `B·A → A·B`), or the normal form is very large.
    StepLimit { limit: usize },
    /// An expansion would exceed the term or power cap (`E-NC-002`).
    TooLarge { what: String },
    /// A rewrite rule's left-hand side is not a single word with coefficient
    /// 1 (`E-NC-003`).
    InvalidRule { reason: String },
    /// A ladder-operator specification is malformed (`E-NC-004`).
    InvalidOperator { reason: String },
    /// A γ-matrix slot is not one of `0, 1, 2, 3, 5` or a commutative vector
    /// label (`E-NC-005`).
    InvalidGamma { reason: String },
    /// The γ trace is longer than [`gamma_trace`] expands (`E-NC-006`).
    GammaTraceTooLong { slots: usize, max: usize },
    /// A vacuum expectation was asked of an expression containing
    /// non-commutative atoms that are not ladder operators (`E-NC-007`).
    UnknownOperator { atom: String },
    /// The active budget tripped or cancellation was requested.
    Budget(BudgetError),
}

impl fmt::Display for NcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NcError::StepLimit { limit } => write!(
                f,
                "noncommutative rewriting did not reach a normal form within {limit} steps"
            ),
            NcError::TooLarge { what } => write!(f, "noncommutative expansion too large: {what}"),
            NcError::InvalidRule { reason } => write!(f, "invalid noncommutative rule: {reason}"),
            NcError::InvalidOperator { reason } => {
                write!(f, "invalid ladder-operator specification: {reason}")
            }
            NcError::InvalidGamma { reason } => write!(f, "invalid gamma-matrix slot: {reason}"),
            NcError::GammaTraceTooLong { slots, max } => write!(
                f,
                "gamma trace of {slots} slots exceeds the supported maximum of {max}"
            ),
            NcError::UnknownOperator { atom } => write!(
                f,
                "vacuum expectation: `{atom}` is a noncommutative atom that is not a ladder operator of the given modes"
            ),
            NcError::Budget(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for NcError {}

impl crate::errors::AlkahestError for NcError {
    fn code(&self) -> &'static str {
        match self {
            NcError::StepLimit { .. } => "E-NC-001",
            NcError::TooLarge { .. } => "E-NC-002",
            NcError::InvalidRule { .. } => "E-NC-003",
            NcError::InvalidOperator { .. } => "E-NC-004",
            NcError::InvalidGamma { .. } => "E-NC-005",
            NcError::GammaTraceTooLong { .. } => "E-NC-006",
            NcError::UnknownOperator { .. } => "E-NC-007",
            NcError::Budget(e) => e.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            NcError::StepLimit { .. } => Some(
                "check that the rules terminate (each rule should move words toward one fixed order), or raise max_steps",
            ),
            NcError::TooLarge { .. } => Some(
                "expand a smaller expression, or lower the power of the noncommutative factor",
            ),
            NcError::InvalidRule { .. } => Some(
                "write the left-hand side as a single product of noncommutative symbols, e.g. make_rule(X*P, P*X + I*hbar)",
            ),
            NcError::InvalidOperator { .. } => Some(
                "pass (annihilator, creator) pairs of distinct noncommutative symbols, each used in one mode only",
            ),
            NcError::InvalidGamma { .. } => Some(
                "use 0, 1, 2, 3 for γ^0…γ^3, 5 for γ^5, or a commutative symbol for a slashed vector / Lorentz index",
            ),
            NcError::GammaTraceTooLong { .. } => {
                Some("split the trace (e.g. with Fierz or completeness relations) before calling gamma_trace")
            }
            NcError::UnknownOperator { .. } => Some(
                "list every ladder operator in the modes, or remove the other noncommutative factors first",
            ),
            NcError::Budget(e) => e.remediation(),
        }
    }
}

impl From<BudgetError> for NcError {
    fn from(e: BudgetError) -> Self {
        NcError::Budget(e)
    }
}

// ---------------------------------------------------------------------------
// Terms
// ---------------------------------------------------------------------------

/// One term `coeff · word` of a noncommutative polynomial.
pub type NcTerm = (ExprId, Vec<ExprId>);

fn one(pool: &ExprPool) -> ExprId {
    pool.integer(1_i32)
}

fn times(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let o = one(pool);
    if a == o {
        b
    } else if b == o {
        a
    } else {
        pool.mul(vec![a, b])
    }
}

fn small_int(expr: ExprId, pool: &ExprPool) -> Option<i64> {
    pool.with(expr, |d| match d {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    })
}

fn check_terms(n: usize) -> Result<(), NcError> {
    if n > MAX_TERMS {
        return Err(NcError::TooLarge {
            what: format!("{n} terms (cap {MAX_TERMS})"),
        });
    }
    Ok(())
}

fn product(a: &[NcTerm], b: &[NcTerm], pool: &ExprPool) -> Result<Vec<NcTerm>, NcError> {
    check_terms(a.len().saturating_mul(b.len()))?;
    let mut out = Vec::with_capacity(a.len() * b.len());
    for (ca, wa) in a {
        for (cb, wb) in b {
            let mut w = wa.clone();
            w.extend_from_slice(wb);
            out.push((times(*ca, *cb, pool), w));
        }
    }
    Ok(merge_raw(out, pool))
}

/// Group equal words, adding their coefficients without simplifying them.
fn merge_raw(terms: Vec<NcTerm>, pool: &ExprPool) -> Vec<NcTerm> {
    let mut index: HashMap<Vec<ExprId>, usize> = HashMap::new();
    let mut groups: Vec<(Vec<ExprId>, Vec<ExprId>)> = Vec::new();
    for (c, w) in terms {
        match index.get(&w) {
            Some(&i) => groups[i].1.push(c),
            None => {
                index.insert(w.clone(), groups.len());
                groups.push((w, vec![c]));
            }
        }
    }
    groups
        .into_iter()
        .map(|(w, cs)| {
            let c = if cs.len() == 1 { cs[0] } else { pool.add(cs) };
            (c, w)
        })
        .collect()
}

/// Group equal words, simplify each coefficient, drop the zero ones.
fn merge(terms: Vec<NcTerm>, pool: &ExprPool) -> Vec<NcTerm> {
    let zero = pool.integer(0_i32);
    let mut out: Vec<NcTerm> = merge_raw(terms, pool)
        .into_iter()
        .map(|(c, w)| (simplify_coeff(c, pool), w))
        .filter(|(c, _)| *c != zero)
        .collect();
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

fn simplify_coeff(c: ExprId, pool: &ExprPool) -> ExprId {
    let is_atomic = pool.with(c, |d| {
        matches!(
            d,
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Symbol { .. }
        )
    });
    if is_atomic {
        return c;
    }
    crate::simplify::engine::simplify_expanded(c, pool).value
}

fn expand_raw(expr: ExprId, pool: &ExprPool) -> Result<Vec<NcTerm>, NcError> {
    if pool.is_mult_commutative(expr) {
        return Ok(vec![(expr, Vec::new())]);
    }
    enum Shape {
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Atom,
    }
    let shape = pool.with(expr, |d| match d {
        ExprData::Add(v) => Shape::Add(v.clone()),
        ExprData::Mul(v) => Shape::Mul(v.clone()),
        ExprData::Pow { base, exp } => Shape::Pow(*base, *exp),
        _ => Shape::Atom,
    });
    match shape {
        Shape::Add(args) => {
            let mut out = Vec::new();
            for a in args {
                out.extend(expand_raw(a, pool)?);
                check_terms(out.len())?;
            }
            Ok(merge_raw(out, pool))
        }
        Shape::Mul(args) => {
            let mut acc: Vec<NcTerm> = vec![(one(pool), Vec::new())];
            for a in args {
                if pool.is_mult_commutative(a) {
                    for t in acc.iter_mut() {
                        t.0 = times(t.0, a, pool);
                    }
                } else {
                    let e = expand_raw(a, pool)?;
                    acc = product(&acc, &e, pool)?;
                }
            }
            Ok(acc)
        }
        Shape::Pow(base, exp) => match small_int(exp, pool) {
            Some(0) => Ok(vec![(one(pool), Vec::new())]),
            Some(n) if n > 0 => {
                if n > MAX_NC_POWER {
                    return Err(NcError::TooLarge {
                        what: format!("power {n} of a noncommutative factor (cap {MAX_NC_POWER})"),
                    });
                }
                let b = expand_raw(base, pool)?;
                let mut acc = b.clone();
                for _ in 1..n {
                    budget::check()?;
                    acc = product(&acc, &b, pool)?;
                }
                Ok(acc)
            }
            _ => Ok(vec![(one(pool), vec![expr])]),
        },
        Shape::Atom => Ok(vec![(one(pool), vec![expr])]),
    }
}

/// The terms `(coefficient, word)` of `expr` read as a noncommutative
/// polynomial: products distributed, non-negative integer powers of
/// noncommutative factors unrolled, equal words collected, coefficients
/// simplified and zero terms dropped. Sorted by word.
pub fn nc_terms(expr: ExprId, pool: &ExprPool) -> Result<Vec<NcTerm>, NcError> {
    Ok(merge(expand_raw(expr, pool)?, pool))
}

/// Rebuild an expression from terms; a run of equal atoms becomes a power.
pub fn nc_terms_to_expr(terms: &[NcTerm], pool: &ExprPool) -> ExprId {
    let o = one(pool);
    let parts: Vec<ExprId> = terms
        .iter()
        .map(|(c, w)| {
            let mut factors = Vec::new();
            if *c != o {
                // Numbers first, then the rest of the coefficient: `2·I·ħ·P`.
                match pool.with(*c, |d| match d {
                    ExprData::Mul(v) => Some(v.clone()),
                    _ => None,
                }) {
                    Some(cs) => {
                        let is_num = |x: &ExprId| {
                            pool.with(*x, |d| {
                                matches!(d, ExprData::Integer(_) | ExprData::Rational(_))
                            })
                        };
                        factors.extend(cs.iter().copied().filter(is_num));
                        factors.extend(cs.iter().copied().filter(|x| !is_num(x)));
                    }
                    None => factors.push(*c),
                }
            }
            let mut i = 0;
            while i < w.len() {
                let mut j = i + 1;
                while j < w.len() && w[j] == w[i] {
                    j += 1;
                }
                let run = j - i;
                factors.push(if run == 1 {
                    w[i]
                } else {
                    pool.pow(w[i], pool.integer(run as i64))
                });
                i = j;
            }
            match factors.len() {
                0 => o,
                1 => factors[0],
                _ => pool.mul(factors),
            }
        })
        .collect();
    match parts.len() {
        0 => pool.integer(0_i32),
        1 => parts[0],
        _ => pool.add(parts),
    }
}

/// Expand `expr` as a noncommutative polynomial (see [`nc_terms`]).
///
/// `(σx + σy)²` becomes `σx² + σx·σy + σy·σx + σy²`; `X·(P + 1)·X` becomes
/// `X·P·X + X²`. Commutative subexpressions are left as coefficients.
pub fn nc_expand(expr: ExprId, pool: &ExprPool) -> Result<ExprId, NcError> {
    Ok(nc_terms_to_expr(&nc_terms(expr, pool)?, pool))
}

// ---------------------------------------------------------------------------
// Word rewriting
// ---------------------------------------------------------------------------

/// A rewriting system on words: finds the leftmost reducible position.
trait WordReducer {
    /// `(start, len, replacement)` for the first redex of `word`, if any.
    fn find(&self, word: &[ExprId], pool: &ExprPool) -> Option<(usize, usize, Vec<NcTerm>)>;
}

fn reduce_terms(
    terms: Vec<NcTerm>,
    red: &dyn WordReducer,
    pool: &ExprPool,
    max_steps: usize,
) -> Result<Vec<NcTerm>, NcError> {
    let mut done: Vec<NcTerm> = Vec::new();
    let mut work = terms;
    let mut steps = 0usize;
    while !work.is_empty() {
        let mut next: Vec<NcTerm> = Vec::new();
        for (c, w) in work {
            match red.find(&w, pool) {
                None => done.push((c, w)),
                Some((start, len, repl)) => {
                    steps += 1;
                    if steps > max_steps {
                        return Err(NcError::StepLimit { limit: max_steps });
                    }
                    if steps % 64 == 0 {
                        budget::check()?;
                    }
                    for (rc, rw) in repl {
                        let mut nw = Vec::with_capacity(w.len() - len + rw.len());
                        nw.extend_from_slice(&w[..start]);
                        nw.extend_from_slice(&rw);
                        nw.extend_from_slice(&w[start + len..]);
                        next.push((times(c, rc, pool), nw));
                    }
                }
            }
        }
        budget::check()?;
        work = merge(next, pool);
        check_terms(work.len() + done.len())?;
    }
    Ok(merge(done, pool))
}

/// Literal word rules `lhs → rhs`.
struct WordRules {
    rules: Vec<(Vec<ExprId>, Vec<NcTerm>)>,
}

impl WordReducer for WordRules {
    fn find(&self, word: &[ExprId], _pool: &ExprPool) -> Option<(usize, usize, Vec<NcTerm>)> {
        for start in 0..word.len() {
            for (lhs, rhs) in &self.rules {
                if word[start..].starts_with(lhs) {
                    return Some((start, lhs.len(), rhs.clone()));
                }
            }
        }
        None
    }
}

/// Rewrite `expr` with literal word rules `lhs → rhs` to a normal form.
///
/// Each `lhs` must expand to a single non-empty word with coefficient 1
/// (`X·P`, `a·a†`, `σz²`); every occurrence of that word *inside* a longer
/// word is replaced, products are re-expanded, and the process repeats until
/// no rule applies. Symbols in rules are literal — there are no wildcards.
///
/// # Errors
/// `E-NC-003` for a malformed left-hand side, `E-NC-001` when no normal
/// form is reached within `max_steps` rewrites, `E-NC-002` past the size
/// caps, and the budget's own code when the active budget trips.
pub fn nc_rewrite(
    expr: ExprId,
    rules: &[(ExprId, ExprId)],
    pool: &ExprPool,
    max_steps: usize,
) -> Result<ExprId, NcError> {
    let mut compiled = Vec::with_capacity(rules.len());
    for &(lhs, rhs) in rules {
        let lt = nc_terms(lhs, pool)?;
        let word = match lt.as_slice() {
            [(c, w)] if *c == one(pool) && !w.is_empty() => w.clone(),
            _ => {
                return Err(NcError::InvalidRule {
                    reason: format!(
                        "left-hand side `{}` is not a single product of noncommutative factors",
                        pool.display(lhs)
                    ),
                })
            }
        };
        compiled.push((word, nc_terms(rhs, pool)?));
    }
    let red = WordRules { rules: compiled };
    let terms = reduce_terms(nc_terms(expr, pool)?, &red, pool, max_steps)?;
    Ok(nc_terms_to_expr(&terms, pool))
}

/// `[a, b] = a·b − b·a`, expanded and collected.
pub fn commutator(a: ExprId, b: ExprId, pool: &ExprPool) -> Result<ExprId, NcError> {
    let ab = pool.mul(vec![a, b]);
    let ba = pool.mul(vec![pool.integer(-1_i32), b, a]);
    nc_expand(pool.add(vec![ab, ba]), pool)
}

/// `{a, b} = a·b + b·a`, expanded and collected.
pub fn anticommutator(a: ExprId, b: ExprId, pool: &ExprPool) -> Result<ExprId, NcError> {
    let ab = pool.mul(vec![a, b]);
    let ba = pool.mul(vec![b, a]);
    nc_expand(pool.add(vec![ab, ba]), pool)
}

// ---------------------------------------------------------------------------
// Pauli algebra
// ---------------------------------------------------------------------------

struct Pauli {
    /// Ids of σx, σy, σz.
    ids: [ExprId; 3],
}

impl Pauli {
    fn index(&self, a: ExprId) -> Option<usize> {
        self.ids.iter().position(|&s| s == a)
    }
}

impl WordReducer for Pauli {
    fn find(&self, word: &[ExprId], pool: &ExprPool) -> Option<(usize, usize, Vec<NcTerm>)> {
        for i in 0..word.len().saturating_sub(1) {
            let (Some(p), Some(q)) = (self.index(word[i]), self.index(word[i + 1])) else {
                continue;
            };
            if p == q {
                return Some((i, 2, vec![(one(pool), Vec::new())]));
            }
            // σ_p σ_q = i ε_pqk σ_k
            let k = 3 - p - q;
            let cyclic = (p + 1) % 3 == q;
            let ii = pool.imaginary_unit();
            let c = if cyclic {
                ii
            } else {
                pool.mul(vec![pool.integer(-1_i32), ii])
            };
            return Some((i, 2, vec![(c, vec![self.ids[k]])]));
        }
        None
    }
}

fn collect_atoms(terms: &[NcTerm]) -> Vec<ExprId> {
    let mut seen = Vec::new();
    for (_, w) in terms {
        for &a in w {
            if !seen.contains(&a) {
                seen.push(a);
            }
        }
    }
    seen
}

/// Reduce every word over the Pauli matrices to at most one σ.
///
/// The Pauli matrices are the non-commutative symbols named `sx`, `sy`,
/// `sz`; `σᵢ² = 1` and `σᵢσⱼ = δᵢⱼ + iεᵢⱼₖσₖ` (with `i` the pool's imaginary
/// unit `I`). Products and integer powers are expanded first, so
/// `σz² = 1`, `σxσyσx = −σy` and `(σx + σy)² = 2`. Other non-commutative
/// symbols keep their place in the word and block reduction across them.
pub fn pauli_reduce(expr: ExprId, pool: &ExprPool) -> Result<ExprId, NcError> {
    let terms = nc_terms(expr, pool)?;
    let atoms = collect_atoms(&terms);
    let names = ["sx", "sy", "sz"];
    let mut found: [Option<ExprId>; 3] = [None; 3];
    let mut domain = Domain::Complex;
    for &a in &atoms {
        pool.with(a, |d| {
            if let ExprData::Symbol {
                name,
                domain: dom,
                commutative: false,
            } = d
            {
                if let Some(k) = names.iter().position(|n| n == name) {
                    if found[k].is_none() {
                        found[k] = Some(a);
                        domain = *dom;
                    }
                }
            }
        });
    }
    if found.iter().all(|f| f.is_none()) {
        return Ok(nc_terms_to_expr(&terms, pool));
    }
    let pick =
        |k: usize| found[k].unwrap_or_else(|| pool.symbol_commutative(names[k], domain, false));
    let ids = [pick(0), pick(1), pick(2)];
    let red = Pauli { ids };
    let out = reduce_terms(terms, &red, pool, DEFAULT_MAX_STEPS)?;
    Ok(nc_terms_to_expr(&out, pool))
}

// ---------------------------------------------------------------------------
// Ladder operators
// ---------------------------------------------------------------------------

/// One mode of creation/annihilation operators.
///
/// Bosonic modes satisfy `[a, a†] = 1`; fermionic ones `{c, c†} = 1` and
/// `c² = c†² = 0`. Operators of different modes commute, except two
/// fermionic ones, which anticommute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderMode {
    /// The annihilation operator `a` (a non-commutative symbol).
    pub annihilator: ExprId,
    /// The creation operator `a†` (a non-commutative symbol).
    pub creator: ExprId,
    /// `true` for a fermionic mode.
    pub fermionic: bool,
}

#[derive(Clone, Copy)]
struct LadderOp {
    mode: usize,
    creator: bool,
    fermionic: bool,
}

struct Ladder {
    ops: HashMap<ExprId, LadderOp>,
}

impl Ladder {
    fn new(modes: &[LadderMode], pool: &ExprPool) -> Result<Self, NcError> {
        let mut ops = HashMap::new();
        for (i, m) in modes.iter().enumerate() {
            for (id, creator) in [(m.annihilator, false), (m.creator, true)] {
                let ok = pool.with(id, |d| {
                    matches!(
                        d,
                        ExprData::Symbol {
                            commutative: false,
                            ..
                        }
                    )
                });
                if !ok {
                    return Err(NcError::InvalidOperator {
                        reason: format!(
                            "`{}` is not a noncommutative symbol (create it with commutative=False)",
                            pool.display(id)
                        ),
                    });
                }
                let op = LadderOp {
                    mode: i,
                    creator,
                    fermionic: m.fermionic,
                };
                if ops.insert(id, op).is_some() {
                    return Err(NcError::InvalidOperator {
                        reason: format!("`{}` appears more than once", pool.display(id)),
                    });
                }
            }
        }
        Ok(Ladder { ops })
    }

    /// Replacement for the adjacent pair `u v`, or `None` when it is in
    /// normal order.
    fn swap(&self, u: ExprId, v: ExprId, pool: &ExprPool) -> Option<Vec<NcTerm>> {
        let (p, q) = (self.ops.get(&u)?, self.ops.get(&v)?);
        let sign = if p.fermionic && q.fermionic {
            pool.integer(-1_i32)
        } else {
            one(pool)
        };
        let swapped = (sign, vec![v, u]);
        match (p.creator, q.creator) {
            // a · b†
            (false, true) => {
                if p.mode == q.mode {
                    Some(vec![swapped, (one(pool), Vec::new())])
                } else {
                    Some(vec![swapped])
                }
            }
            // two creators or two annihilators
            (cp, cq) if cp == cq => {
                if p.mode == q.mode {
                    if p.fermionic {
                        Some(Vec::new()) // c c = 0
                    } else {
                        None
                    }
                } else if p.mode > q.mode {
                    Some(vec![swapped])
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl WordReducer for Ladder {
    fn find(&self, word: &[ExprId], pool: &ExprPool) -> Option<(usize, usize, Vec<NcTerm>)> {
        for i in 0..word.len().saturating_sub(1) {
            if let Some(r) = self.swap(word[i], word[i + 1], pool) {
                return Some((i, 2, r));
            }
        }
        None
    }
}

/// Normal-order `expr`: every creation operator to the left of every
/// annihilation operator, using `[a, a†] = 1` (bosons) or `{c, c†} = 1`,
/// `c² = 0` (fermions). Operators of different modes are sorted by the order
/// of `modes` (creators and annihilators alike), with a sign for each swap of
/// two fermionic operators. The result is the commutation-relation identity
/// (Wick's theorem): e.g. `a·a† = a†·a + 1`.
///
/// Non-commutative atoms that are not ladder operators stay where they are
/// and block reordering across them.
///
/// # Errors
/// `E-NC-004` when an operator is not a non-commutative symbol or is listed
/// twice; `E-NC-001`/`E-NC-002` past the caps.
pub fn normal_order(
    expr: ExprId,
    modes: &[LadderMode],
    pool: &ExprPool,
) -> Result<ExprId, NcError> {
    let red = Ladder::new(modes, pool)?;
    let terms = reduce_terms(nc_terms(expr, pool)?, &red, pool, DEFAULT_MAX_STEPS)?;
    Ok(nc_terms_to_expr(&terms, pool))
}

/// Vacuum expectation value `⟨0| expr |0⟩`: the coefficient of the empty
/// word after normal ordering (every normal-ordered word with an operator in
/// it annihilates the vacuum on one side).
///
/// # Errors
/// `E-NC-007` when `expr` contains a non-commutative atom that is not one of
/// the listed ladder operators, plus the refusals of [`normal_order`].
pub fn vacuum_expectation(
    expr: ExprId,
    modes: &[LadderMode],
    pool: &ExprPool,
) -> Result<ExprId, NcError> {
    let red = Ladder::new(modes, pool)?;
    let terms = nc_terms(expr, pool)?;
    for a in collect_atoms(&terms) {
        if !red.ops.contains_key(&a) {
            return Err(NcError::UnknownOperator {
                atom: pool.display(a).to_string(),
            });
        }
    }
    let terms = reduce_terms(terms, &red, pool, DEFAULT_MAX_STEPS)?;
    Ok(terms
        .iter()
        .find(|(_, w)| w.is_empty())
        .map(|(c, _)| *c)
        .unwrap_or_else(|| pool.integer(0_i32)))
}

// ---------------------------------------------------------------------------
// Simplify with user rules to a fixpoint
// ---------------------------------------------------------------------------

/// [`simplify_with`] for expressions containing non-commutative symbols:
/// apply the rules; whenever they fired, re-expand the result as a
/// noncommutative polynomial (so a rule that produced `(P·X + iħ)·P` can see
/// the new `X·P` inside `P·X·P`) and apply them again, up to a fixed number
/// of rounds or until the active budget trips.
///
/// The value is always equal to `expr` under the rules; if the round cap is
/// reached the last expression is returned as it stands, exactly as
/// [`simplify_with`] does when it runs out of passes.
pub fn simplify_with_noncommutative(
    expr: ExprId,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
    config: SimplifyConfig,
) -> DerivedExpr<ExprId> {
    let mut cur = simplify_with(expr, pool, rules, config.clone());
    if cur.value == expr || pool.is_mult_commutative(expr) {
        return cur;
    }
    for _ in 0..MAX_FIXPOINT_ROUNDS {
        // The expansion recurses: stay under the plain depth ceiling.
        if budget::check().is_err()
            || crate::kernel::depth::check_expr_depth(pool, cur.value).is_err()
        {
            break;
        }
        let Ok(expanded) = nc_expand(cur.value, pool) else {
            break;
        };
        if expanded == cur.value {
            break;
        }
        let mut log: DerivationLog = cur.log;
        log = log.merge(crate::simplify::rules::one_step(
            "nc_expand",
            cur.value,
            expanded,
        ));
        let next = simplify_with(expanded, pool, rules, config.clone());
        let fired = next.value != expanded;
        cur = DerivedExpr::with_log(next.value, log.merge(next.log));
        if !fired {
            break;
        }
    }
    cur
}

// ---------------------------------------------------------------------------
// Dirac traces
// ---------------------------------------------------------------------------

/// One factor of a Dirac trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GammaSlot {
    /// `γ^k` for `k ∈ {0, 1, 2, 3}`.
    Index(u8),
    /// `γ⁵ = iγ⁰γ¹γ²γ³`.
    Gamma5,
    /// A slashed vector `p̸ = p_μ γ^μ` (or, read as a label, `γ^μ` for a
    /// symbolic Lorentz index `μ`).
    Vector(ExprId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Idx(u8),
    Vec(ExprId),
}

fn slot_expr(s: Slot, pool: &ExprPool) -> ExprId {
    match s {
        Slot::Idx(k) => pool.integer(k as i64),
        Slot::Vec(v) => v,
    }
}

/// `dot(u, v)` as `Ok(n)` (a number) or `Err(factor)`.
fn dot(u: Slot, v: Slot, pool: &ExprPool) -> Result<i64, ExprId> {
    match (u, v) {
        (Slot::Idx(i), Slot::Idx(j)) => Ok(if i != j {
            0
        } else if i == 0 {
            1
        } else {
            -1
        }),
        (Slot::Idx(_), Slot::Vec(_)) => {
            Err(pool.func("dot", vec![slot_expr(u, pool), slot_expr(v, pool)]))
        }
        (Slot::Vec(_), Slot::Idx(_)) => {
            Err(pool.func("dot", vec![slot_expr(v, pool), slot_expr(u, pool)]))
        }
        (Slot::Vec(a), Slot::Vec(b)) => {
            let (a, b) = if a <= b { (a, b) } else { (b, a) };
            Err(pool.func("dot", vec![a, b]))
        }
    }
}

/// Trace of a product of slashed slots without γ⁵, as `(coeff, factors)`.
fn trace_plain(
    slots: &[Slot],
    pool: &ExprPool,
    out: &mut Vec<(i64, Vec<ExprId>)>,
    coeff: i64,
    factors: &mut Vec<ExprId>,
    counter: &mut usize,
) -> Result<(), NcError> {
    *counter += 1;
    if *counter % 256 == 0 {
        budget::check()?;
    }
    if slots.is_empty() {
        out.push((4 * coeff, factors.clone()));
        return Ok(());
    }
    if slots.len() % 2 == 1 {
        return Ok(());
    }
    let first = slots[0];
    for k in 1..slots.len() {
        let sign = if k % 2 == 1 { 1 } else { -1 };
        let mut rest: Vec<Slot> = Vec::with_capacity(slots.len() - 2);
        rest.extend_from_slice(&slots[1..k]);
        rest.extend_from_slice(&slots[k + 1..]);
        match dot(first, slots[k], pool) {
            Ok(0) => {}
            Ok(n) => trace_plain(&rest, pool, out, coeff * sign * n, factors, counter)?,
            Err(f) => {
                factors.push(f);
                trace_plain(&rest, pool, out, coeff * sign, factors, counter)?;
                factors.pop();
            }
        }
    }
    Ok(())
}

/// `ε(a, b, c, d) = ε^{μνρσ} a_μ b_ν c_ρ d_σ` with `ε^{0123} = +1`.
fn levi_civita(s: &[Slot; 4], pool: &ExprPool) -> Result<i64, ExprId> {
    for i in 0..4 {
        for j in i + 1..4 {
            if s[i] == s[j] {
                return Ok(0);
            }
        }
    }
    if s.iter().all(|x| matches!(x, Slot::Idx(_))) {
        let mut p: Vec<u8> = s
            .iter()
            .map(|x| match x {
                Slot::Idx(k) => *k,
                Slot::Vec(_) => unreachable!(),
            })
            .collect();
        // sign of the permutation by bubble sort
        let mut sign = 1;
        for i in 0..4 {
            for j in 0..3 - i {
                if p[j] > p[j + 1] {
                    p.swap(j, j + 1);
                    sign = -sign;
                }
            }
        }
        return Ok(sign);
    }
    Err(pool.func(
        "levi_civita",
        s.iter().map(|&x| slot_expr(x, pool)).collect(),
    ))
}

/// Trace of a product of Dirac matrices in four dimensions.
///
/// Slots are `γ^0…γ^3` ([`GammaSlot::Index`]), `γ⁵` and slashed vectors
/// `p̸ = p_μγ^μ` ([`GammaSlot::Vector`]); the metric is `diag(+1,−1,−1,−1)`,
/// `{γ^μ, γ^ν} = 2g^{μν}`, `tr 1 = 4`, and `γ⁵ = iγ⁰γ¹γ²γ³`.
///
/// The result is a polynomial in
/// * `dot(p, q) = p·q` (and `dot(k, p) = p^k` for an explicit index `k`;
///   two explicit indices give the metric entry as a number), and
/// * `levi_civita(a, b, c, d) = ε^{μνρσ} a_μ b_ν c_ρ d_σ` with `ε^{0123} = +1`,
///   so `tr(γ⁵ a̸ b̸ c̸ d̸) = −4i·levi_civita(a, b, c, d)`.
///
/// A symbolic `Vector` may equally stand for a free Lorentz index `μ`
/// (`dot(μ, ν) = g^{μν}`, `dot(μ, p) = p^μ`).
///
/// A γ⁵ with more than four other slots is rewritten as `iγ⁰γ¹γ²γ³`, so the
/// answer then involves explicit components `dot(k, p)`.
///
/// # Errors
/// `E-NC-005` for an index outside `{0,1,2,3}`; `E-NC-006` when more than
/// 14 slots remain to be expanded.
pub fn gamma_trace(slots: &[GammaSlot], pool: &ExprPool) -> Result<ExprId, NcError> {
    // Move every γ⁵ to the front: γ⁵ anticommutes with each γ^μ it passes.
    let mut sign: i64 = 1;
    let mut n5 = 0usize;
    let mut rest: Vec<Slot> = Vec::with_capacity(slots.len());
    for s in slots {
        match *s {
            GammaSlot::Gamma5 => {
                if rest.len() % 2 == 1 {
                    sign = -sign;
                }
                n5 += 1;
            }
            GammaSlot::Index(k) => {
                if k > 3 {
                    return Err(NcError::InvalidGamma {
                        reason: format!("index {k} is not 0, 1, 2, 3 or 5"),
                    });
                }
                rest.push(Slot::Idx(k));
            }
            GammaSlot::Vector(v) => {
                if !pool.is_mult_commutative(v) {
                    return Err(NcError::InvalidGamma {
                        reason: format!("vector label `{}` must be commutative", pool.display(v)),
                    });
                }
                rest.push(Slot::Vec(v));
            }
        }
    }
    let with_g5 = n5 % 2 == 1;
    let zero = pool.integer(0_i32);
    if rest.len() % 2 == 1 {
        return Ok(zero);
    }
    let mut terms: Vec<(i64, Vec<ExprId>)> = Vec::new();
    let mut imag = false;
    if with_g5 {
        if rest.len() < 4 {
            return Ok(zero);
        }
        if rest.len() == 4 {
            let s = [rest[0], rest[1], rest[2], rest[3]];
            imag = true;
            match levi_civita(&s, pool) {
                Ok(0) => return Ok(zero),
                Ok(e) => terms.push((-4 * sign * e, Vec::new())),
                Err(f) => terms.push((-4 * sign, vec![f])),
            }
        } else {
            // γ⁵ = i γ⁰γ¹γ²γ³
            let mut full = vec![Slot::Idx(0), Slot::Idx(1), Slot::Idx(2), Slot::Idx(3)];
            full.extend_from_slice(&rest);
            if full.len() > MAX_GAMMA_SLOTS {
                return Err(NcError::GammaTraceTooLong {
                    slots: full.len(),
                    max: MAX_GAMMA_SLOTS,
                });
            }
            imag = true;
            let mut counter = 0;
            trace_plain(&full, pool, &mut terms, sign, &mut Vec::new(), &mut counter)?;
        }
    } else {
        if rest.len() > MAX_GAMMA_SLOTS {
            return Err(NcError::GammaTraceTooLong {
                slots: rest.len(),
                max: MAX_GAMMA_SLOTS,
            });
        }
        let mut counter = 0;
        trace_plain(&rest, pool, &mut terms, sign, &mut Vec::new(), &mut counter)?;
    }
    // Collect equal monomials.
    let mut index: HashMap<Vec<ExprId>, usize> = HashMap::new();
    let mut collected: Vec<(i64, Vec<ExprId>)> = Vec::new();
    for (c, mut f) in terms {
        f.sort_unstable();
        match index.get(&f) {
            Some(&i) => collected[i].0 += c,
            None => {
                index.insert(f.clone(), collected.len());
                collected.push((c, f));
            }
        }
    }
    let parts: Vec<ExprId> = collected
        .into_iter()
        .filter(|(c, _)| *c != 0)
        .map(|(c, mut f)| {
            f.insert(0, pool.integer(c));
            if imag {
                f.push(pool.imaginary_unit());
            }
            if f.len() == 1 {
                f[0]
            } else {
                pool.mul(f)
            }
        })
        .collect();
    let out = match parts.len() {
        0 => zero,
        1 => parts[0],
        _ => pool.add(parts),
    };
    Ok(crate::simplify::engine::simplify(out, pool).value)
}

#[cfg(test)]
#[path = "nc_poly_tests.rs"]
mod tests;
