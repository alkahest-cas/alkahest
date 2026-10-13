//! Propositional logic layer (G10): n-ary connectives, normal forms, exact
//! two-level minimisation, ROBDD construction and equivalence checking.
//!
//! # Formulas
//!
//! A propositional formula is a kernel predicate tree built from `And` / `Or`
//! / `Not` / `True` / `False` over **atoms**.  An atom is a bare symbol (read
//! as a Boolean variable) or — for the transformations that are valid under
//! any interpretation of their atoms ([`to_nnf`], [`to_cnf`], [`to_dnf`],
//! [`simplify_logic`]) — an arithmetic relation such as `x > 0`, treated as an
//! opaque proposition.  Quantifiers and non-Boolean terms are refused
//! ([`BooleanError::NotPropositional`]).
//!
//! The kernel has exactly three connectives, so [`logic_xor`],
//! [`logic_implies`], [`logic_equivalent`] and [`logic_ite`] build their
//! definitions out of them.  Every formula is lifted into a hash-consed
//! **circuit** (a DAG keyed by `ExprId`), so a shared sub-formula — the
//! running parity of an `n`-ary `Xor`, the carry chain of an adder — is
//! processed once rather than once per path.
//!
//! # Minimisation
//!
//! [`simplify_logic`] returns a two-level form (sum of products or product of
//! sums).  Up to [`EXACT_AUTO_MAX_VARS`] variables it is **exact**: prime
//! implicants by Quine–McCluskey, then a minimum cover (fewest terms, then
//! fewest literals) by Petrick's method run as branch-and-bound.  Beyond that
//! it switches to an espresso-style expand/irredundant pass on the ROBDD of the
//! function, which always returns an equivalent cover but not necessarily a
//! minimum one; [`MinimizedLogic::exact`] says which happened.  Every result
//! is checked against the input before it is returned.

use super::bdd::{BddManager, BddRef};
use super::{dpll_sat_bounded, BoolClause, DpllOutcome};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Refusals of the propositional layer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BooleanError {
    /// `E-LOGIC-002`: the expression is not a propositional formula over the
    /// atoms this operation accepts.
    NotPropositional(String),
    /// `E-LOGIC-003`: a size ceiling (BDD nodes, normal-form terms, search
    /// decisions, truth-table width) was reached.  Nothing was truncated.
    TooLarge {
        /// What grew past its ceiling.
        what: &'static str,
        /// The ceiling.
        limit: u64,
    },
    /// `E-LOGIC-004`: an exact minimum was demanded (`exact = true`) and could
    /// not be certified within the limits.
    ExactUnavailable(String),
    /// `E-LOGIC-005`: an argument is malformed (unknown form or method name,
    /// a variable order naming a non-atom, …).
    InvalidArgument(String),
    /// `E-LOGIC-006`: an internal self-check failed; the result was withheld.
    Internal(String),
    /// The active [`crate::budget::Budget`] tripped.
    Budget(crate::budget::BudgetError),
}

impl fmt::Display for BooleanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BooleanError::NotPropositional(s) => write!(f, "not a propositional formula: {s}"),
            BooleanError::TooLarge { what, limit } => {
                write!(f, "{what} exceeded the limit of {limit}")
            }
            BooleanError::ExactUnavailable(s) => write!(f, "exact minimisation unavailable: {s}"),
            BooleanError::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            BooleanError::Internal(s) => write!(f, "internal self-check failed: {s}"),
            BooleanError::Budget(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BooleanError {}

impl crate::errors::AlkahestError for BooleanError {
    fn code(&self) -> &'static str {
        match self {
            BooleanError::NotPropositional(_) => "E-LOGIC-002",
            BooleanError::TooLarge { .. } => "E-LOGIC-003",
            BooleanError::ExactUnavailable(_) => "E-LOGIC-004",
            BooleanError::InvalidArgument(_) => "E-LOGIC-005",
            BooleanError::Internal(_) => "E-LOGIC-006",
            BooleanError::Budget(e) => e.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            BooleanError::NotPropositional(_) => Some(
                "build the formula from symbols with And/Or/Not/Xor/Implies/Equivalent/ITE; quantified or arithmetic formulas go to decide() or alkahest.smt",
            ),
            BooleanError::TooLarge { .. } => Some(
                "raise the limit (max_nodes=…), give a better variable order, or use equivalent(method='sat') for miters whose BDD blows up",
            ),
            BooleanError::ExactUnavailable(_) => Some(
                "exact two-level minimisation is limited to 16 variables and a bounded covering search; pass exact=None to accept the heuristic cover",
            ),
            BooleanError::InvalidArgument(_) => {
                Some("check the documented values of form / method / order")
            }
            BooleanError::Internal(_) => {
                Some("this is a bug: please report the formula that produced it")
            }
            BooleanError::Budget(e) => e.remediation(),
        }
    }
}

fn budget() -> Result<(), BooleanError> {
    crate::budget::check().map_err(BooleanError::Budget)
}

// ---------------------------------------------------------------------------
// Canonical constructors
// ---------------------------------------------------------------------------

fn pred_kind(pool: &ExprPool, id: ExprId) -> Option<PredicateKind> {
    pool.with(id, |d| match d {
        ExprData::Predicate { kind, .. } => Some(kind.clone()),
        _ => None,
    })
}

fn pred_args(pool: &ExprPool, id: ExprId) -> Vec<ExprId> {
    pool.with(id, |d| match d {
        ExprData::Predicate { args, .. } => args.clone(),
        _ => Vec::new(),
    })
}

fn is_relation(k: &PredicateKind) -> bool {
    matches!(
        k,
        PredicateKind::Lt
            | PredicateKind::Le
            | PredicateKind::Gt
            | PredicateKind::Ge
            | PredicateKind::Eq
            | PredicateKind::Ne
    )
}

/// `Some(x)` when `id` is `Not(x)`.
fn as_not(pool: &ExprPool, id: ExprId) -> Option<ExprId> {
    pool.with(id, |d| match d {
        ExprData::Predicate {
            kind: PredicateKind::Not,
            args,
        } if args.len() == 1 => Some(args[0]),
        _ => None,
    })
}

/// Sort key of an argument of `And`/`Or`: atoms (and their negations) first,
/// by printed name, a literal directly after its complement; compound
/// arguments after, in pool order.  Deterministic for a given pool, and the
/// same for every permutation of the arguments, which is what canonicity
/// needs.
fn arg_key(pool: &ExprPool, id: ExprId) -> (u8, String, u8, ExprId) {
    let (base, neg) = match as_not(pool, id) {
        Some(x) => (x, 1u8),
        None => (id, 0u8),
    };
    let atomic = pool.with(base, |d| match d {
        ExprData::Symbol { .. } => true,
        ExprData::Predicate { kind, .. } => {
            is_relation(kind) || matches!(kind, PredicateKind::True | PredicateKind::False)
        }
        _ => false,
    });
    if atomic {
        (0, pool.display(base).to_string(), neg, base)
    } else if tree_size_at_most(pool, base, 48) {
        // Small compound arguments (clauses, cubes) sort by their printed
        // form, so `(a ∨ b) ∧ (a ∨ c)` reads in name order.  Large or
        // heavily shared ones fall back to pool order: printing a DAG as a
        // tree can be exponential.
        (1, pool.display(base).to_string(), neg, base)
    } else {
        (2, String::new(), neg, base)
    }
}

/// `true` when `id`, unfolded as a tree, has at most `limit` nodes.
fn tree_size_at_most(pool: &ExprPool, id: ExprId, limit: usize) -> bool {
    fn go(pool: &ExprPool, id: ExprId, budget: &mut usize) -> bool {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let kids = pool.with(id, |d| match d {
            ExprData::Predicate { args, .. } => args.clone(),
            _ => Vec::new(),
        });
        kids.into_iter().all(|k| go(pool, k, budget))
    }
    let mut b = limit;
    go(pool, id, &mut b)
}

fn nary(pool: &ExprPool, args: &[ExprId], conj: bool) -> ExprId {
    let (me, absorbing, identity) = if conj {
        (
            PredicateKind::And,
            PredicateKind::False,
            PredicateKind::True,
        )
    } else {
        (PredicateKind::Or, PredicateKind::True, PredicateKind::False)
    };
    // Flatten nested same-kind nodes.
    let mut flat: Vec<ExprId> = Vec::with_capacity(args.len());
    let mut stack: Vec<ExprId> = args.iter().rev().copied().collect();
    while let Some(a) = stack.pop() {
        match pred_kind(pool, a) {
            Some(k) if k == me => {
                for x in pred_args(pool, a).into_iter().rev() {
                    stack.push(x);
                }
            }
            Some(k) if k == identity => {}
            Some(k) if k == absorbing => return pool.predicate(absorbing, vec![]),
            _ => flat.push(a),
        }
    }
    let mut seen = HashSet::new();
    flat.retain(|a| seen.insert(*a));
    // x and ¬x together.
    for &a in &flat {
        if let Some(x) = as_not(pool, a) {
            if seen.contains(&x) {
                return pool.predicate(absorbing, vec![]);
            }
        }
    }
    match flat.len() {
        0 => pool.predicate(identity, vec![]),
        1 => flat[0],
        _ => {
            let mut keyed: Vec<_> = flat.into_iter().map(|a| (arg_key(pool, a), a)).collect();
            keyed.sort();
            pool.predicate(me, keyed.into_iter().map(|(_, a)| a).collect())
        }
    }
}

/// n-ary conjunction: flattens nested `And`, drops `True`, absorbs `False`,
/// removes duplicates, collapses `x ∧ ¬x` to `False` and sorts the
/// arguments canonically.  No arguments gives `True`.
pub fn logic_and(pool: &ExprPool, args: &[ExprId]) -> ExprId {
    nary(pool, args, true)
}

/// n-ary disjunction (dual of [`logic_and`]).  No arguments gives `False`.
pub fn logic_or(pool: &ExprPool, args: &[ExprId]) -> ExprId {
    nary(pool, args, false)
}

/// Negation with `¬¬x = x`, `¬⊤ = ⊥`, `¬⊥ = ⊤`.
pub fn logic_not(pool: &ExprPool, a: ExprId) -> ExprId {
    if let Some(x) = as_not(pool, a) {
        return x;
    }
    match pred_kind(pool, a) {
        Some(PredicateKind::True) => pool.pred_false(),
        Some(PredicateKind::False) => pool.pred_true(),
        _ => pool.pred_not(a),
    }
}

/// n-ary exclusive or (odd parity).  The kernel has no `Xor` node, so the
/// result is the parity chain written with `And`/`Or`/`Not`; consecutive
/// stages share their sub-formula in the pool, so its size is linear in the
/// number of arguments.  Constants and repeated arguments (`x ⊕ x = ⊥`) are
/// folded first.  No arguments gives `False`.
pub fn logic_xor(pool: &ExprPool, args: &[ExprId]) -> ExprId {
    let mut parity = false;
    let mut count: HashMap<ExprId, usize> = HashMap::new();
    let mut order = Vec::new();
    for &a in args {
        match pred_kind(pool, a) {
            Some(PredicateKind::True) => parity = !parity,
            Some(PredicateKind::False) => {}
            _ => {
                // ¬x ⊕ … = ¬(x ⊕ …): factor negations out into the parity.
                let (base, neg) = match as_not(pool, a) {
                    Some(x) => (x, true),
                    None => (a, false),
                };
                if neg {
                    parity = !parity;
                }
                let c = count.entry(base).or_insert(0);
                if *c == 0 {
                    order.push(base);
                }
                *c += 1;
            }
        }
    }
    let mut live: Vec<ExprId> = order.into_iter().filter(|a| count[a] % 2 == 1).collect();
    let mut keyed: Vec<_> = live.drain(..).map(|a| (arg_key(pool, a), a)).collect();
    keyed.sort();
    let live: Vec<ExprId> = keyed.into_iter().map(|(_, a)| a).collect();
    let Some(&first) = live.first() else {
        return if parity {
            pool.pred_true()
        } else {
            pool.pred_false()
        };
    };
    // Carry the running parity `p` and its complement `q` together, so no
    // stage negates a compound sub-formula:
    //   p' = (p ∧ ¬x) ∨ (q ∧ x),   q' = (p ∧ x) ∨ (q ∧ ¬x).
    let mut p = first;
    let mut q = logic_not(pool, first);
    for &x in live.iter().skip(1) {
        let nx = logic_not(pool, x);
        let p2 = logic_or(pool, &[logic_and(pool, &[p, nx]), logic_and(pool, &[q, x])]);
        let q2 = logic_or(pool, &[logic_and(pool, &[p, x]), logic_and(pool, &[q, nx])]);
        p = p2;
        q = q2;
    }
    if parity {
        q
    } else {
        p
    }
}

/// `a → b`, i.e. `¬a ∨ b`.
pub fn logic_implies(pool: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
    logic_or(pool, &[logic_not(pool, a), b])
}

/// n-ary equivalence: true when all arguments have the same value,
/// `(a₁ ∧ … ∧ aₙ) ∨ (¬a₁ ∧ … ∧ ¬aₙ)`.  Fewer than two arguments give `True`.
pub fn logic_equivalent(pool: &ExprPool, args: &[ExprId]) -> ExprId {
    let mut uniq: Vec<ExprId> = Vec::new();
    for &a in args {
        if !uniq.contains(&a) {
            uniq.push(a);
        }
    }
    if uniq.len() < 2 {
        return pool.pred_true();
    }
    let all = logic_and(pool, &uniq);
    let negs: Vec<ExprId> = uniq.iter().map(|&a| logic_not(pool, a)).collect();
    let none = logic_and(pool, &negs);
    logic_or(pool, &[all, none])
}

/// If-then-else `(c ∧ t) ∨ (¬c ∧ e)`, folded when `c`, `t` or `e` is a
/// constant or `t = e`.
pub fn logic_ite(pool: &ExprPool, c: ExprId, t: ExprId, e: ExprId) -> ExprId {
    match pred_kind(pool, c) {
        Some(PredicateKind::True) => return t,
        Some(PredicateKind::False) => return e,
        _ => {}
    }
    if t == e {
        return t;
    }
    let l = logic_and(pool, &[c, t]);
    let r = logic_and(pool, &[logic_not(pool, c), e]);
    logic_or(pool, &[l, r])
}

// ---------------------------------------------------------------------------
// Circuits
// ---------------------------------------------------------------------------

/// Most gates a lifted circuit may hold.
const MAX_CIRCUIT_GATES: usize = 1 << 20;

#[derive(Debug, Clone)]
enum Gate {
    Var(u32),
    Const(bool),
    Not(u32),
    And(Vec<u32>),
    Or(Vec<u32>),
}

/// A propositional formula as a DAG of gates, children before parents.
struct Circuit {
    gates: Vec<Gate>,
    /// Atom `ExprId` per variable index, in first-visit (DFS) order.
    atoms: Vec<ExprId>,
    names: Vec<String>,
    has_relations: bool,
    allow_relations: bool,
    memo: HashMap<ExprId, u32>,
    var_of: HashMap<ExprId, u32>,
    name_set: HashSet<String>,
}

impl Circuit {
    fn new(allow_relations: bool) -> Self {
        Circuit {
            gates: Vec::new(),
            atoms: Vec::new(),
            names: Vec::new(),
            has_relations: false,
            allow_relations,
            memo: HashMap::new(),
            var_of: HashMap::new(),
            name_set: HashSet::new(),
        }
    }

    fn n_vars(&self) -> usize {
        self.atoms.len()
    }

    fn push(&mut self, g: Gate) -> Result<u32, BooleanError> {
        if self.gates.len() >= MAX_CIRCUIT_GATES {
            return Err(BooleanError::TooLarge {
                what: "formula gates",
                limit: MAX_CIRCUIT_GATES as u64,
            });
        }
        if self.gates.len() % 4096 == 4095 {
            budget()?;
        }
        self.gates.push(g);
        Ok(self.gates.len() as u32 - 1)
    }

    /// Declare `atom` as a variable (used to fix an order before lifting).
    fn declare_atom(&mut self, pool: &ExprPool, atom: ExprId) -> Result<u32, BooleanError> {
        if let Some(&v) = self.var_of.get(&atom) {
            return Ok(v);
        }
        let name = pool.display(atom).to_string();
        if !self.name_set.insert(name.clone()) {
            return Err(BooleanError::NotPropositional(format!(
                "two different atoms print as '{name}' (same name, different domain?)"
            )));
        }
        let v = self.atoms.len() as u32;
        self.atoms.push(atom);
        self.names.push(name);
        self.var_of.insert(atom, v);
        Ok(v)
    }

    fn lift(&mut self, pool: &ExprPool, e: ExprId) -> Result<u32, BooleanError> {
        if let Some(&g) = self.memo.get(&e) {
            return Ok(g);
        }
        let data = pool.get(e);
        let g = match data {
            ExprData::Symbol { .. } => {
                let v = self.declare_atom(pool, e)?;
                self.push(Gate::Var(v))?
            }
            ExprData::Predicate { kind, args } => match kind {
                PredicateKind::True => self.push(Gate::Const(true))?,
                PredicateKind::False => self.push(Gate::Const(false))?,
                PredicateKind::Not => {
                    if args.len() != 1 {
                        return Err(BooleanError::NotPropositional(
                            "Not must have exactly one argument".into(),
                        ));
                    }
                    let c = self.lift(pool, args[0])?;
                    self.push(Gate::Not(c))?
                }
                PredicateKind::And | PredicateKind::Or => {
                    let mut cs = Vec::with_capacity(args.len());
                    for a in &args {
                        cs.push(self.lift(pool, *a)?);
                    }
                    if kind == PredicateKind::And {
                        self.push(Gate::And(cs))?
                    } else {
                        self.push(Gate::Or(cs))?
                    }
                }
                k if is_relation(&k) => {
                    if !self.allow_relations {
                        return Err(BooleanError::NotPropositional(format!(
                            "'{}' is an arithmetic relation; this operation needs Boolean symbols only",
                            pool.display(e)
                        )));
                    }
                    self.has_relations = true;
                    let v = self.declare_atom(pool, e)?;
                    self.push(Gate::Var(v))?
                }
                _ => unreachable!("every PredicateKind is matched above"),
            },
            ExprData::Forall { .. } | ExprData::Exists { .. } => {
                return Err(BooleanError::NotPropositional(
                    "quantified formulas are not propositional".into(),
                ))
            }
            _ => {
                return Err(BooleanError::NotPropositional(format!(
                    "'{}' is not a Boolean symbol or connective",
                    pool.display(e)
                )))
            }
        };
        self.memo.insert(e, g);
        Ok(g)
    }

    /// Evaluate every gate under one total assignment.
    fn eval_one(&self, assignment: &[bool]) -> Vec<bool> {
        let mut v = vec![false; self.gates.len()];
        for (i, g) in self.gates.iter().enumerate() {
            v[i] = match g {
                Gate::Var(x) => assignment[*x as usize],
                Gate::Const(b) => *b,
                Gate::Not(c) => !v[*c as usize],
                Gate::And(cs) => cs.iter().all(|c| v[*c as usize]),
                Gate::Or(cs) => cs.iter().any(|c| v[*c as usize]),
            };
        }
        v
    }

    /// Truth tables of `roots` over all `2^n` assignments, 64 per word.
    /// Bit `m` (word `m >> 6`, bit `m & 63`) is the value at the assignment
    /// whose variable `i` is bit `i` of `m`.
    fn truth_tables(&self, roots: &[u32]) -> Result<Vec<Vec<u64>>, BooleanError> {
        let n = self.n_vars();
        if n > TRUTH_TABLE_MAX_VARS {
            return Err(BooleanError::TooLarge {
                what: "truth-table variables",
                limit: TRUTH_TABLE_MAX_VARS as u64,
            });
        }
        const PAT: [u64; 6] = [
            0xAAAA_AAAA_AAAA_AAAA,
            0xCCCC_CCCC_CCCC_CCCC,
            0xF0F0_F0F0_F0F0_F0F0,
            0xFF00_FF00_FF00_FF00,
            0xFFFF_0000_FFFF_0000,
            0xFFFF_FFFF_0000_0000,
        ];
        let words = if n <= 6 { 1 } else { 1usize << (n - 6) };
        let valid_mask: u64 = if n >= 6 {
            u64::MAX
        } else {
            (1u64 << (1u32 << n)) - 1
        };
        let mut out = vec![vec![0u64; words]; roots.len()];
        let mut v = vec![0u64; self.gates.len()];
        #[allow(clippy::needless_range_loop)] // `w` also selects the input pattern
        for w in 0..words {
            if w % 256 == 255 {
                budget()?;
            }
            for (i, g) in self.gates.iter().enumerate() {
                v[i] = match g {
                    Gate::Var(x) => {
                        let x = *x as usize;
                        if x < 6 {
                            PAT[x]
                        } else if (w >> (x - 6)) & 1 == 1 {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    Gate::Const(b) => {
                        if *b {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    Gate::Not(c) => !v[*c as usize],
                    Gate::And(cs) => cs.iter().fold(u64::MAX, |acc, c| acc & v[*c as usize]),
                    Gate::Or(cs) => cs.iter().fold(0, |acc, c| acc | v[*c as usize]),
                };
            }
            for (r, &root) in roots.iter().enumerate() {
                out[r][w] = v[root as usize] & valid_mask;
            }
        }
        Ok(out)
    }

    /// BDDs of every gate.  Variables are looked up by name, because the
    /// manager may reorder (see [`BddManager::set_auto_reorder`]) while the
    /// gates are being built; the gates built so far are protected through
    /// the reordering.
    fn to_bdd(&self, mgr: &mut BddManager) -> Result<Vec<BddRef>, BooleanError> {
        let mut b: Vec<BddRef> = Vec::with_capacity(self.gates.len());
        for g in &self.gates {
            let r = match g {
                Gate::Var(x) => mgr.var(&self.names[*x as usize])?,
                Gate::Const(t) => {
                    if *t {
                        BddRef::TRUE
                    } else {
                        BddRef::FALSE
                    }
                }
                Gate::Not(c) => mgr.not(b[*c as usize])?,
                Gate::And(cs) => {
                    let mut acc = BddRef::TRUE;
                    for c in cs {
                        acc = mgr.and(acc, b[*c as usize])?;
                    }
                    acc
                }
                Gate::Or(cs) => {
                    let mut acc = BddRef::FALSE;
                    for c in cs {
                        acc = mgr.or(acc, b[*c as usize])?;
                    }
                    acc
                }
            };
            b.push(r);
            mgr.maybe_reorder(&b)?;
        }
        Ok(b)
    }

    /// Emit the formula rooted at `g` in negation normal form.
    fn to_nnf_expr(
        &self,
        pool: &ExprPool,
        g: u32,
        positive: bool,
        memo: &mut HashMap<(u32, bool), ExprId>,
    ) -> ExprId {
        if let Some(&e) = memo.get(&(g, positive)) {
            return e;
        }
        let e = match &self.gates[g as usize] {
            Gate::Var(x) => {
                let a = self.atoms[*x as usize];
                if positive {
                    a
                } else {
                    pool.pred_not(a)
                }
            }
            Gate::Const(b) => {
                if *b == positive {
                    pool.pred_true()
                } else {
                    pool.pred_false()
                }
            }
            Gate::Not(c) => self.to_nnf_expr(pool, *c, !positive, memo),
            Gate::And(cs) | Gate::Or(cs) => {
                let conj = matches!(self.gates[g as usize], Gate::And(_)) == positive;
                let parts: Vec<ExprId> = cs
                    .iter()
                    .map(|c| self.to_nnf_expr(pool, *c, positive, memo))
                    .collect();
                if conj {
                    logic_and(pool, &parts)
                } else {
                    logic_or(pool, &parts)
                }
            }
        };
        memo.insert((g, positive), e);
        e
    }
}

/// Most variables a truth table is built for.
pub const TRUTH_TABLE_MAX_VARS: usize = 22;

fn lift_one(
    pool: &ExprPool,
    expr: ExprId,
    allow_relations: bool,
) -> Result<(Circuit, u32), BooleanError> {
    let mut c = Circuit::new(allow_relations);
    let r = c.lift(pool, expr)?;
    Ok((c, r))
}

// ---------------------------------------------------------------------------
// Normal forms
// ---------------------------------------------------------------------------

/// Negation normal form: negations only on atoms, connectives `And`/`Or`.
/// Relations are kept as opaque atoms (`¬(x < 0)` stays a negated atom; it is
/// **not** rewritten to `x ≥ 0`, which would assume a total order).
pub fn to_nnf(pool: &ExprPool, expr: ExprId) -> Result<ExprId, BooleanError> {
    let (c, root) = lift_one(pool, expr, true)?;
    let mut memo = HashMap::new();
    Ok(c.to_nnf_expr(pool, root, true, &mut memo))
}

/// Most product terms (or clauses) a distributed normal form may hold.
pub const MAX_NORMAL_FORM_TERMS: usize = 100_000;

type Term = Vec<i32>; // sorted literals, ±(var + 1)
type TermSet = Vec<Term>;

fn normalise_terms(mut set: TermSet) -> TermSet {
    set.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    set.dedup();
    // Absorption: drop any term that is a superset of a kept one.
    let mut kept: Vec<Term> = Vec::with_capacity(set.len());
    'outer: for t in set {
        for k in &kept {
            if k.len() <= t.len() && is_sorted_subset(k, &t) {
                continue 'outer;
            }
        }
        kept.push(t);
    }
    kept
}

fn is_sorted_subset(small: &[i32], big: &[i32]) -> bool {
    let mut j = 0;
    for &x in small {
        while j < big.len() && big[j] < x {
            j += 1;
        }
        if j == big.len() || big[j] != x {
            return false;
        }
        j += 1;
    }
    true
}

fn merge_terms(a: &[i32], b: &[i32]) -> Option<Term> {
    let mut out: Vec<i32> = a.iter().chain(b.iter()).copied().collect();
    out.sort_unstable();
    out.dedup();
    let set: HashSet<i32> = out.iter().copied().collect();
    if out.iter().any(|l| set.contains(&-l)) {
        return None;
    }
    Some(out)
}

fn dnf_terms(
    c: &Circuit,
    g: u32,
    positive: bool,
    memo: &mut HashMap<(u32, bool), TermSet>,
) -> Result<TermSet, BooleanError> {
    if let Some(t) = memo.get(&(g, positive)) {
        return Ok(t.clone());
    }
    let out = match &c.gates[g as usize] {
        Gate::Var(x) => {
            let l = *x as i32 + 1;
            vec![vec![if positive { l } else { -l }]]
        }
        Gate::Const(b) => {
            if *b == positive {
                vec![vec![]]
            } else {
                vec![]
            }
        }
        Gate::Not(x) => dnf_terms(c, *x, !positive, memo)?,
        Gate::And(cs) | Gate::Or(cs) => {
            let conj = matches!(c.gates[g as usize], Gate::And(_)) == positive;
            if conj {
                let mut acc: TermSet = vec![vec![]];
                for ch in cs {
                    let t = dnf_terms(c, *ch, positive, memo)?;
                    let size = acc.len().saturating_mul(t.len());
                    if size > MAX_NORMAL_FORM_TERMS {
                        return Err(BooleanError::TooLarge {
                            what: "normal-form terms",
                            limit: MAX_NORMAL_FORM_TERMS as u64,
                        });
                    }
                    crate::budget::check_growth(size as u64).map_err(BooleanError::Budget)?;
                    let mut next = Vec::with_capacity(size);
                    for a in &acc {
                        for b in &t {
                            if let Some(m) = merge_terms(a, b) {
                                next.push(m);
                            }
                        }
                    }
                    acc = normalise_terms(next);
                    if acc.is_empty() {
                        break;
                    }
                }
                acc
            } else {
                let mut acc: TermSet = Vec::new();
                for ch in cs {
                    acc.extend(dnf_terms(c, *ch, positive, memo)?);
                    if acc.len() > MAX_NORMAL_FORM_TERMS {
                        return Err(BooleanError::TooLarge {
                            what: "normal-form terms",
                            limit: MAX_NORMAL_FORM_TERMS as u64,
                        });
                    }
                }
                normalise_terms(acc)
            }
        }
    };
    budget()?;
    memo.insert((g, positive), out.clone());
    Ok(out)
}

fn literal_expr(pool: &ExprPool, atoms: &[ExprId], lit: i32) -> ExprId {
    let a = atoms[lit.unsigned_abs() as usize - 1];
    if lit > 0 {
        a
    } else {
        pool.pred_not(a)
    }
}

fn dnf_expr(pool: &ExprPool, atoms: &[ExprId], terms: &TermSet) -> ExprId {
    let ts: Vec<ExprId> = terms
        .iter()
        .map(|t| {
            let lits: Vec<ExprId> = t.iter().map(|&l| literal_expr(pool, atoms, l)).collect();
            logic_and(pool, &lits)
        })
        .collect();
    logic_or(pool, &ts)
}

/// `clauses` are the terms of the DNF of `¬f`; negating each literal gives
/// the CNF of `f`.
fn cnf_expr(pool: &ExprPool, atoms: &[ExprId], neg_terms: &TermSet) -> ExprId {
    let cs: Vec<ExprId> = neg_terms
        .iter()
        .map(|t| {
            let lits: Vec<ExprId> = t.iter().map(|&l| literal_expr(pool, atoms, -l)).collect();
            logic_or(pool, &lits)
        })
        .collect();
    logic_and(pool, &cs)
}

/// Disjunctive normal form by distribution (no minimisation beyond removing
/// contradictory, duplicate and absorbed terms).  Refuses past
/// [`MAX_NORMAL_FORM_TERMS`] terms.
pub fn to_dnf(pool: &ExprPool, expr: ExprId) -> Result<ExprId, BooleanError> {
    let (c, root) = lift_one(pool, expr, true)?;
    let mut memo = HashMap::new();
    let terms = dnf_terms(&c, root, true, &mut memo)?;
    Ok(dnf_expr(pool, &c.atoms, &terms))
}

/// Conjunctive normal form by distribution (dual of [`to_dnf`]): tautological
/// clauses, duplicates and subsumed clauses are dropped.
pub fn to_cnf(pool: &ExprPool, expr: ExprId) -> Result<ExprId, BooleanError> {
    let (c, root) = lift_one(pool, expr, true)?;
    let mut memo = HashMap::new();
    let terms = dnf_terms(&c, root, false, &mut memo)?;
    Ok(cnf_expr(pool, &c.atoms, &terms))
}

// ---------------------------------------------------------------------------
// Two-level minimisation
// ---------------------------------------------------------------------------

/// Two-level target form of [`simplify_logic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LogicForm {
    /// Sum of products.
    Dnf,
    /// Product of sums.
    Cnf,
    /// Whichever of the two has fewer literals (then fewer terms); DNF on a tie.
    Auto,
}

impl LogicForm {
    /// Parse `"dnf"` / `"cnf"` / `"auto"`.
    pub fn parse(s: &str) -> Result<Self, BooleanError> {
        match s.to_ascii_lowercase().as_str() {
            "dnf" | "sop" => Ok(LogicForm::Dnf),
            "cnf" | "pos" => Ok(LogicForm::Cnf),
            "auto" => Ok(LogicForm::Auto),
            other => Err(BooleanError::InvalidArgument(format!(
                "form must be 'dnf', 'cnf' or 'auto', not '{other}'"
            ))),
        }
    }
}

/// Result of [`simplify_logic`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MinimizedLogic {
    /// The two-level formula.
    pub expr: ExprId,
    /// `Dnf` or `Cnf` — the form actually returned.
    pub form: LogicForm,
    /// `true` when the cover is proven minimum (fewest terms, then fewest
    /// literals, among all two-level forms of this kind); `false` for the
    /// heuristic cover.
    pub exact: bool,
    /// Number of product terms (DNF) or clauses (CNF).
    pub terms: usize,
    /// Total number of literal occurrences.
    pub literals: usize,
}

/// Variables up to which `exact = None` uses the exact method.
pub const EXACT_AUTO_MAX_VARS: usize = 12;
/// Variables up to which `exact = Some(true)` is attempted at all.
pub const EXACT_MAX_VARS: usize = 16;
/// Most branch-and-bound nodes the exact covering search may visit.
const COVER_STEP_LIMIT: u64 = 2_000_000;
/// Most cubes the heuristic starts from (paths of the BDD).
const HEURISTIC_MAX_CUBES: usize = 50_000;

/// A cube over `n ≤ 32` variables: `mask` bit set = variable absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Cube {
    bits: u32,
    mask: u32,
}

impl Cube {
    fn covers(self, m: u32) -> bool {
        (m & !self.mask) == self.bits
    }
    fn literals(self, n: usize) -> usize {
        n - (self.mask & low_mask(n)).count_ones() as usize
    }
}

fn low_mask(n: usize) -> u32 {
    if n >= 32 {
        u32::MAX
    } else {
        (1u32 << n) - 1
    }
}

fn prime_implicants(minterms: &[u32], n: usize) -> Result<Vec<Cube>, BooleanError> {
    let mut current: HashSet<Cube> = minterms
        .iter()
        .map(|&m| Cube { bits: m, mask: 0 })
        .collect();
    let mut primes = Vec::new();
    let mut total: u64 = current.len() as u64;
    while !current.is_empty() {
        budget()?;
        let mut next: HashSet<Cube> = HashSet::new();
        let mut used: HashSet<Cube> = HashSet::new();
        for &c in &current {
            for i in 0..n {
                let bit = 1u32 << i;
                if c.mask & bit != 0 || c.bits & bit != 0 {
                    continue;
                }
                let partner = Cube {
                    bits: c.bits | bit,
                    mask: c.mask,
                };
                if current.contains(&partner) {
                    next.insert(Cube {
                        bits: c.bits,
                        mask: c.mask | bit,
                    });
                    used.insert(c);
                    used.insert(partner);
                }
            }
        }
        for &c in &current {
            if !used.contains(&c) {
                primes.push(c);
            }
        }
        total += next.len() as u64;
        crate::budget::check_growth(next.len() as u64).map_err(BooleanError::Budget)?;
        if total > 20_000_000 {
            return Err(BooleanError::TooLarge {
                what: "Quine-McCluskey implicants",
                limit: 20_000_000,
            });
        }
        current = next;
    }
    primes.sort();
    Ok(primes)
}

/// Minimum-cost set cover (Petrick's method as branch-and-bound).
struct CoverSearch<'a> {
    /// Rows covered by each column.
    col_rows: &'a [Vec<usize>],
    /// Columns covering each row.
    row_cols: &'a [Vec<usize>],
    cost: &'a [u64],
    best_cost: u64,
    best: Option<Vec<usize>>,
    steps: u64,
    limit: u64,
    aborted: bool,
}

impl CoverSearch<'_> {
    fn lower_bound(&self, covered: &[bool], excluded: &[bool]) -> u64 {
        // Rows that share no candidate column each need a distinct column.
        let mut used_cols: HashSet<usize> = HashSet::new();
        let mut rows: Vec<usize> = (0..covered.len()).filter(|&r| !covered[r]).collect();
        rows.sort_by_key(|&r| self.row_cols[r].iter().filter(|&&c| !excluded[c]).count());
        let mut lb = 0u64;
        for r in rows {
            let cand: Vec<usize> = self.row_cols[r]
                .iter()
                .copied()
                .filter(|&c| !excluded[c])
                .collect();
            if cand.iter().any(|c| used_cols.contains(c)) {
                continue;
            }
            let Some(min) = cand.iter().map(|&c| self.cost[c]).min() else {
                return u64::MAX / 2;
            };
            lb += min;
            used_cols.extend(cand);
        }
        lb
    }

    fn search(
        &mut self,
        covered: &mut Vec<bool>,
        excluded: &mut Vec<bool>,
        chosen: &mut Vec<usize>,
        cost: u64,
    ) {
        if self.aborted {
            return;
        }
        self.steps += 1;
        if self.steps > self.limit || (self.steps % 4096 == 0 && crate::budget::check().is_err()) {
            self.aborted = true;
            return;
        }
        // Pick the uncovered row with the fewest live columns.
        let mut pick: Option<(usize, usize)> = None;
        for (r, &done) in covered.iter().enumerate() {
            if done {
                continue;
            }
            let k = self.row_cols[r].iter().filter(|&&c| !excluded[c]).count();
            if k == 0 {
                return; // infeasible
            }
            if pick.map_or(true, |(_, best)| k < best) {
                pick = Some((r, k));
            }
        }
        let Some((row, _)) = pick else {
            if cost < self.best_cost {
                self.best_cost = cost;
                self.best = Some(chosen.clone());
            }
            return;
        };
        if cost.saturating_add(self.lower_bound(covered, excluded)) >= self.best_cost {
            return;
        }
        let mut cands: Vec<usize> = self.row_cols[row]
            .iter()
            .copied()
            .filter(|&c| !excluded[c])
            .collect();
        cands.sort_by_key(|&c| {
            let gain = self.col_rows[c].iter().filter(|&&r| !covered[r]).count();
            (self.cost[c], std::cmp::Reverse(gain))
        });
        let mut newly_excluded = Vec::new();
        for c in cands {
            let newly: Vec<usize> = self.col_rows[c]
                .iter()
                .copied()
                .filter(|&r| !covered[r])
                .collect();
            for &r in &newly {
                covered[r] = true;
            }
            chosen.push(c);
            self.search(covered, excluded, chosen, cost + self.cost[c]);
            chosen.pop();
            for &r in &newly {
                covered[r] = false;
            }
            if self.aborted {
                break;
            }
            // Every cover containing `c` has now been explored.
            excluded[c] = true;
            newly_excluded.push(c);
        }
        for c in newly_excluded {
            excluded[c] = false;
        }
    }
}

/// Greedy cover (upper bound / fallback).
fn greedy_cover(n_rows: usize, col_rows: &[Vec<usize>], cost: &[u64]) -> Vec<usize> {
    let mut covered = vec![false; n_rows];
    let mut left = n_rows;
    let mut chosen = Vec::new();
    while left > 0 {
        let mut best: Option<(usize, f64)> = None;
        for (c, rows) in col_rows.iter().enumerate() {
            let gain = rows.iter().filter(|&&r| !covered[r]).count();
            if gain == 0 {
                continue;
            }
            let score = cost[c] as f64 / gain as f64;
            if best.map_or(true, |(_, s)| score < s) {
                best = Some((c, score));
            }
        }
        let Some((c, _)) = best else { break };
        for &r in &col_rows[c] {
            if !covered[r] {
                covered[r] = true;
                left -= 1;
            }
        }
        chosen.push(c);
    }
    chosen
}

/// Minimum two-level cover of the ON-set `on` with don't-cares `dc`
/// (minterm lists over `n` variables).  Returns `(cubes, exact)`.
fn exact_sop(
    on: &[u32],
    dc: &[u32],
    n: usize,
    demand_exact: bool,
) -> Result<(Vec<Cube>, bool), BooleanError> {
    if on.is_empty() {
        return Ok((Vec::new(), true));
    }
    let mut all: Vec<u32> = on.iter().chain(dc.iter()).copied().collect();
    all.sort_unstable();
    all.dedup();
    let primes = prime_implicants(&all, n)?;
    // Columns: primes covering at least one ON minterm.
    let mut col_rows: Vec<Vec<usize>> = Vec::new();
    let mut cols: Vec<Cube> = Vec::new();
    for p in primes {
        let rows: Vec<usize> = on
            .iter()
            .enumerate()
            .filter(|(_, &m)| p.covers(m))
            .map(|(i, _)| i)
            .collect();
        if !rows.is_empty() {
            col_rows.push(rows);
            cols.push(p);
        }
    }
    let big = (n as u64) * (on.len() as u64) + 1;
    let cost: Vec<u64> = cols.iter().map(|c| big + c.literals(n) as u64).collect();
    let (chosen, exact) = solve_cover(on.len(), &col_rows, &cost, demand_exact)?;
    let mut out: Vec<Cube> = chosen.into_iter().map(|c| cols[c]).collect();
    if exact {
        out.sort();
    }
    Ok((out, exact))
}

/// Minimum-cost cover of `n_rows` rows by columns (`col_rows[c]` = rows column
/// `c` covers, `cost[c]` its cost): dominated columns are dropped, then
/// Petrick's method runs as branch-and-bound.  Returns the chosen column
/// indices and whether minimality was proven (`false`: the search hit its step
/// limit and the best cover found so far is returned, unless `demand_exact`).
fn solve_cover(
    n_rows: usize,
    col_rows: &[Vec<usize>],
    cost: &[u64],
    demand_exact: bool,
) -> Result<(Vec<usize>, bool), BooleanError> {
    let ncols = col_rows.len();
    // Column dominance: drop a column whose rows are a subset of a column no
    // more expensive (ties keep the lower index).
    let mut alive = vec![true; ncols];
    if ncols <= 4000 {
        let sets: Vec<HashSet<usize>> = col_rows
            .iter()
            .map(|r| r.iter().copied().collect())
            .collect();
        for i in 0..ncols {
            if i % 64 == 0 {
                budget()?;
            }
            for j in 0..ncols {
                if i == j || !alive[j] {
                    continue;
                }
                let dominated = cost[j] <= cost[i]
                    && sets[i].len() <= sets[j].len()
                    && sets[i].iter().all(|r| sets[j].contains(r))
                    && (cost[j] < cost[i] || sets[i].len() < sets[j].len() || j < i);
                if dominated {
                    alive[i] = false;
                    break;
                }
            }
        }
    }
    let keep: Vec<usize> = (0..ncols).filter(|&i| alive[i]).collect();
    let col_rows: Vec<Vec<usize>> = keep.iter().map(|&i| col_rows[i].clone()).collect();
    let cost: Vec<u64> = keep.iter().map(|&i| cost[i]).collect();
    let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
    for (c, rows) in col_rows.iter().enumerate() {
        for &r in rows {
            row_cols[r].push(c);
        }
    }
    let greedy = greedy_cover(n_rows, &col_rows, &cost);
    let greedy_cost: u64 = greedy.iter().map(|&c| cost[c]).sum();
    let mut s = CoverSearch {
        col_rows: &col_rows,
        row_cols: &row_cols,
        cost: &cost,
        best_cost: greedy_cost.saturating_add(1),
        best: None,
        steps: 0,
        limit: COVER_STEP_LIMIT,
        aborted: false,
    };
    let mut covered = vec![false; n_rows];
    let mut excluded = vec![false; col_rows.len()];
    let mut chosen = Vec::new();
    s.search(&mut covered, &mut excluded, &mut chosen, 0);
    if s.aborted {
        budget()?;
        if demand_exact {
            return Err(BooleanError::ExactUnavailable(format!(
                "the covering search exceeded {COVER_STEP_LIMIT} nodes"
            )));
        }
        let best = s.best.unwrap_or(greedy);
        return Ok((best.into_iter().map(|c| keep[c]).collect(), false));
    }
    let best = s.best.unwrap_or(greedy);
    Ok((best.into_iter().map(|c| keep[c]).collect(), true))
}

fn minterms_of(words: &[u64], n: usize) -> Vec<u32> {
    let total = 1usize << n;
    let mut out = Vec::new();
    for (w, &word) in words.iter().enumerate() {
        let mut x = word;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            let m = w * 64 + b;
            if m < total {
                out.push(m as u32);
            }
            x &= x - 1;
        }
    }
    out
}

/// A cover as literal lists over circuit variables: `(var, positive)`.
type Cover = Vec<Vec<(usize, bool)>>;

fn cube_lits(c: Cube, n: usize) -> Vec<(usize, bool)> {
    (0..n)
        .filter(|i| c.mask & (1 << i) == 0)
        .map(|i| (i, c.bits & (1 << i) != 0))
        .collect()
}

fn cover_literals(cover: &Cover) -> usize {
    cover.iter().map(|t| t.len()).sum()
}

/// Espresso-style heuristic: start from the disjoint path cubes of the BDD of
/// `on`, expand each cube as far as `upper` allows (literal by literal), drop
/// contained cubes, then remove redundant cubes.  The result covers `on` and
/// lies inside `upper`, but is not guaranteed minimum.
fn heuristic_sop(
    mgr: &mut BddManager,
    on: BddRef,
    upper: BddRef,
    n: usize,
) -> Result<Cover, BooleanError> {
    let paths = mgr.cubes(on, HEURISTIC_MAX_CUBES)?;
    let mut cubes: Vec<Vec<Option<bool>>> = Vec::with_capacity(paths.len());
    for p in paths {
        let mut c = vec![None; n];
        for (l, v) in p {
            c[l as usize] = Some(v);
        }
        cubes.push(c);
    }
    // EXPAND.
    for (k, c) in cubes.iter_mut().enumerate() {
        if k % 64 == 0 {
            budget()?;
        }
        for i in 0..n {
            if let Some(v) = c[i] {
                c[i] = None;
                if !mgr.holds_on_cube(upper, c) {
                    c[i] = Some(v);
                }
            }
        }
    }
    // Single-cube containment.
    cubes.sort_by_key(|c| c.iter().filter(|x| x.is_some()).count());
    cubes.dedup();
    let mut kept: Vec<Vec<Option<bool>>> = Vec::new();
    for c in cubes {
        let contained = kept
            .iter()
            .any(|k| k.iter().zip(&c).all(|(a, b)| a.is_none() || a == b));
        if !contained {
            kept.push(c);
        }
    }
    // IRREDUNDANT: try to drop the smallest cubes (most literals) first.
    let cube_bdd = |mgr: &mut BddManager, c: &[Option<bool>]| -> Result<BddRef, BooleanError> {
        let mut acc = BddRef::TRUE;
        for (i, v) in c.iter().enumerate() {
            if let Some(v) = v {
                let x = mgr.var_at(i as u32)?;
                let lit = if *v { x } else { mgr.not(x)? };
                acc = mgr.and(acc, lit)?;
            }
        }
        Ok(acc)
    };
    let mut order: Vec<usize> = (0..kept.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(kept[i].iter().filter(|x| x.is_some()).count()));
    let mut removed = vec![false; kept.len()];
    if kept.len() <= 4000 {
        let bdds: Vec<BddRef> = kept
            .iter()
            .map(|c| cube_bdd(mgr, c))
            .collect::<Result<_, _>>()?;
        for &i in &order {
            budget()?;
            let mut rest = BddRef::FALSE;
            for (j, &b) in bdds.iter().enumerate() {
                if j != i && !removed[j] {
                    rest = mgr.or(rest, b)?;
                }
            }
            let nrest = mgr.not(rest)?;
            let uncovered = mgr.and(on, nrest)?;
            if uncovered.is_false() {
                removed[i] = true;
            }
        }
    }
    Ok(kept
        .into_iter()
        .zip(removed)
        .filter(|(_, r)| !r)
        .map(|(c, _)| {
            c.iter()
                .enumerate()
                .filter_map(|(i, v)| v.map(|v| (i, v)))
                .collect()
        })
        .collect())
}

fn cover_bdd(mgr: &mut BddManager, cover: &Cover) -> Result<BddRef, BooleanError> {
    let mut acc = BddRef::FALSE;
    for t in cover {
        let mut c = BddRef::TRUE;
        for &(i, v) in t {
            let x = mgr.var_at(i as u32)?;
            let lit = if v { x } else { mgr.not(x)? };
            c = mgr.and(c, lit)?;
        }
        acc = mgr.or(acc, c)?;
    }
    Ok(acc)
}

fn cover_to_expr(pool: &ExprPool, atoms: &[ExprId], cover: &Cover, cnf: bool) -> ExprId {
    let parts: Vec<ExprId> = cover
        .iter()
        .map(|t| {
            let lits: Vec<ExprId> = t
                .iter()
                .map(|&(i, v)| {
                    // A CNF clause is the negation of a cube of ¬f.
                    let pos = v != cnf;
                    if pos {
                        atoms[i]
                    } else {
                        pool.pred_not(atoms[i])
                    }
                })
                .collect();
            if cnf {
                logic_or(pool, &lits)
            } else {
                logic_and(pool, &lits)
            }
        })
        .collect();
    if cnf {
        logic_and(pool, &parts)
    } else {
        logic_or(pool, &parts)
    }
}

/// Minimise one side: the SOP of `on` with `dc` free.  `cnf = true` means the
/// caller passes `¬f` and will dualise.
#[allow(clippy::too_many_arguments)]
fn minimise_side(
    circ: &Circuit,
    on_root: u32,
    dc_root: Option<u32>,
    negate: bool,
    exact: Option<bool>,
) -> Result<(Cover, bool), BooleanError> {
    let n = circ.n_vars();
    let use_exact = match exact {
        Some(true) => {
            if n > EXACT_MAX_VARS {
                return Err(BooleanError::ExactUnavailable(format!(
                    "{n} variables (exact minimisation is limited to {EXACT_MAX_VARS})"
                )));
            }
            true
        }
        Some(false) => false,
        None => n <= EXACT_AUTO_MAX_VARS,
    };
    if use_exact {
        let mut roots = vec![on_root];
        if let Some(d) = dc_root {
            roots.push(d);
        }
        let tts = circ.truth_tables(&roots)?;
        let words = tts[0].len();
        let valid: u64 = if n >= 6 {
            u64::MAX
        } else {
            (1u64 << (1u32 << n)) - 1
        };
        let dcw: Vec<u64> = if dc_root.is_some() {
            tts[1].clone()
        } else {
            vec![0; words]
        };
        let onw: Vec<u64> = (0..words)
            .map(|w| {
                let f = if negate {
                    !tts[0][w] & valid
                } else {
                    tts[0][w]
                };
                f & !dcw[w]
            })
            .collect();
        let on = minterms_of(&onw, n);
        let dc = minterms_of(&dcw, n);
        let (cubes, ok) = exact_sop(&on, &dc, n, exact == Some(true))?;
        let cover: Cover = cubes.into_iter().map(|c| cube_lits(c, n)).collect();
        // Check: on ⊆ cover ⊆ on ∪ dc, by evaluating the cubes on every minterm.
        let mut cw = vec![0u64; words];
        for t in &cover {
            for m in 0..(1usize << n) {
                if t.iter().all(|&(i, v)| ((m >> i) & 1 == 1) == v) {
                    cw[m >> 6] |= 1 << (m & 63);
                }
            }
            budget()?;
        }
        for w in 0..words {
            if onw[w] & !cw[w] != 0 || cw[w] & !(onw[w] | dcw[w]) != 0 {
                return Err(BooleanError::Internal(
                    "minimised cover disagrees with the truth table".into(),
                ));
            }
        }
        return Ok((cover, ok));
    }
    let mut mgr = BddManager::new();
    for (i, name) in circ.names.iter().enumerate() {
        let l = mgr.declare(name);
        debug_assert_eq!(l as usize, i);
    }
    let b = circ.to_bdd(&mut mgr)?;
    let f = if negate {
        mgr.not(b[on_root as usize])?
    } else {
        b[on_root as usize]
    };
    let d = dc_root.map(|r| b[r as usize]).unwrap_or(BddRef::FALSE);
    let nd = mgr.not(d)?;
    let on = mgr.and(f, nd)?;
    let upper = mgr.or(f, d)?;
    let cover = heuristic_sop(&mut mgr, on, upper, n)?;
    let cb = cover_bdd(&mut mgr, &cover)?;
    let ncb = mgr.not(cb)?;
    let miss = mgr.and(on, ncb)?;
    let nup = mgr.not(upper)?;
    let excess = mgr.and(cb, nup)?;
    if !miss.is_false() || !excess.is_false() {
        return Err(BooleanError::Internal(
            "heuristic cover disagrees with the BDD of the function".into(),
        ));
    }
    Ok((cover, false))
}

/// Two-level minimisation of a propositional formula.
///
/// * `form` — [`LogicForm::Dnf`] (sum of products), [`LogicForm::Cnf`]
///   (product of sums) or [`LogicForm::Auto`] (the one with fewer literals).
/// * `dontcare` — optional formula marking assignments whose value is free;
///   the result agrees with `expr` wherever `dontcare` is false.
/// * `exact` — `None`: exact up to [`EXACT_AUTO_MAX_VARS`] variables,
///   heuristic beyond; `Some(true)`: exact or refuse
///   ([`BooleanError::ExactUnavailable`]); `Some(false)`: heuristic.
///
/// Minimum means fewest terms, then fewest literals, among two-level forms
/// of the requested kind over the formula's atoms.  Relations are opaque
/// atoms, so the minimum is propositional (`x > 0 ∨ x ≤ 0` is not reduced to
/// `True`).
pub fn simplify_logic(
    pool: &ExprPool,
    expr: ExprId,
    form: LogicForm,
    dontcare: Option<ExprId>,
    exact: Option<bool>,
) -> Result<MinimizedLogic, BooleanError> {
    let mut circ = Circuit::new(true);
    let root = circ.lift(pool, expr)?;
    let dc_root = match dontcare {
        Some(d) => Some(circ.lift(pool, d)?),
        None => None,
    };
    let build = |cnf: bool, circ: &Circuit| -> Result<MinimizedLogic, BooleanError> {
        let (cover, ok) = minimise_side(circ, root, dc_root, cnf, exact)?;
        let e = cover_to_expr(pool, &circ.atoms, &cover, cnf);
        // A constant result (an empty cover, or one empty cube) has no terms.
        let constant = cover.is_empty() || cover.iter().any(|t| t.is_empty());
        Ok(MinimizedLogic {
            expr: e,
            form: if cnf { LogicForm::Cnf } else { LogicForm::Dnf },
            exact: ok,
            terms: if constant { 0 } else { cover.len() },
            literals: if constant { 0 } else { cover_literals(&cover) },
        })
    };
    match form {
        LogicForm::Dnf => build(false, &circ),
        LogicForm::Cnf => build(true, &circ),
        LogicForm::Auto => {
            let d = build(false, &circ)?;
            let c = build(true, &circ)?;
            Ok(if (c.literals, c.terms) < (d.literals, d.terms) {
                c
            } else {
                d
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-output two-level minimisation
// ---------------------------------------------------------------------------

/// Most outputs [`simplify_logic_multi`] accepts (product-term tags are 64-bit
/// masks).
pub const MULTI_MAX_OUTPUTS: usize = 64;

/// Result of [`simplify_logic_multi`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MultiMinimizedLogic {
    /// One two-level formula per input formula, in order.
    pub exprs: Vec<ExprId>,
    /// `Dnf` or `Cnf` — the form actually returned.
    pub form: LogicForm,
    /// `true` when the shared cover is proven minimum (fewest distinct terms,
    /// then fewest literals, over all multi-output two-level forms); `false`
    /// for the heuristic cover.
    pub exact: bool,
    /// Distinct product terms (DNF) or clauses (CNF) across all outputs; a
    /// term used by several outputs counts once.  Constant outputs have none.
    pub terms: usize,
    /// Literal occurrences over the distinct terms.
    pub literals: usize,
    /// Terms in each output's own formula.
    pub per_output_terms: Vec<usize>,
    /// Distinct terms that feed more than one output.
    pub shared_terms: usize,
}

/// Prime implicants of the multi-output function whose minterm `m` carries
/// the set (bit mask) of outputs that may be 1 there: `(cube, tag)` where
/// `tag` is the full set of outputs the cube lies inside and no larger cube
/// has the same tag (multi-output Quine-McCluskey).
fn multi_prime_implicants(
    init: &HashMap<u32, u64>,
    n: usize,
) -> Result<Vec<(Cube, u64)>, BooleanError> {
    let mut current: HashMap<Cube, u64> = init
        .iter()
        .map(|(&m, &t)| (Cube { bits: m, mask: 0 }, t))
        .collect();
    let mut primes: Vec<(Cube, u64)> = Vec::new();
    let mut total: u64 = current.len() as u64;
    while !current.is_empty() {
        budget()?;
        let mut next: HashMap<Cube, u64> = HashMap::new();
        let mut used: HashSet<Cube> = HashSet::new();
        for (&c, &t) in &current {
            for i in 0..n {
                let bit = 1u32 << i;
                if c.mask & bit != 0 || c.bits & bit != 0 {
                    continue;
                }
                let partner = Cube {
                    bits: c.bits | bit,
                    mask: c.mask,
                };
                if let Some(&tp) = current.get(&partner) {
                    let both = t & tp;
                    if both == 0 {
                        continue;
                    }
                    *next
                        .entry(Cube {
                            bits: c.bits,
                            mask: c.mask | bit,
                        })
                        .or_insert(0) |= both;
                    if both == t {
                        used.insert(c);
                    }
                    if both == tp {
                        used.insert(partner);
                    }
                }
            }
        }
        for (&c, &t) in &current {
            if !used.contains(&c) {
                primes.push((c, t));
            }
        }
        total += next.len() as u64;
        crate::budget::check_growth(next.len() as u64).map_err(BooleanError::Budget)?;
        if total > 20_000_000 {
            return Err(BooleanError::TooLarge {
                what: "Quine-McCluskey implicants",
                limit: 20_000_000,
            });
        }
        current = next;
    }
    primes.sort();
    Ok(primes)
}

/// A shared cover: distinct terms plus, per output, the indices of the terms
/// that form that output.
struct SharedCover {
    terms: Cover,
    outputs: Vec<Vec<usize>>,
}

impl SharedCover {
    /// Build from per-output lists of cubes (as literal lists): identical
    /// cubes are merged into one shared term; terms no output uses are dropped.
    fn from_lists(lists: Vec<Vec<Vec<(usize, bool)>>>) -> Self {
        let mut index: HashMap<Vec<(usize, bool)>, usize> = HashMap::new();
        let mut terms: Cover = Vec::new();
        let mut outputs = Vec::with_capacity(lists.len());
        for list in lists {
            let mut mine = Vec::new();
            for mut t in list {
                t.sort();
                let k = *index.entry(t.clone()).or_insert_with(|| {
                    terms.push(t);
                    terms.len() - 1
                });
                if !mine.contains(&k) {
                    mine.push(k);
                }
            }
            outputs.push(mine);
        }
        SharedCover { terms, outputs }
    }
}

#[allow(clippy::needless_range_loop)]
/// Minimise all `roots` jointly to one shared two-level cover.  `cnf = true`
/// minimises the complements and dualises, as [`minimise_side`] does.
fn minimise_multi_side(
    circ: &Circuit,
    roots: &[u32],
    dc_root: Option<u32>,
    cnf: bool,
    exact: Option<bool>,
) -> Result<(SharedCover, bool), BooleanError> {
    let n = circ.n_vars();
    let k = roots.len();
    let use_exact = match exact {
        Some(true) => {
            if n > EXACT_MAX_VARS {
                return Err(BooleanError::ExactUnavailable(format!(
                    "{n} variables (exact minimisation is limited to {EXACT_MAX_VARS})"
                )));
            }
            true
        }
        Some(false) => false,
        None => n <= EXACT_AUTO_MAX_VARS,
    };
    if use_exact {
        let mut all = roots.to_vec();
        if let Some(d) = dc_root {
            all.push(d);
        }
        let tts = circ.truth_tables(&all)?;
        let words = tts[0].len();
        let valid: u64 = if n >= 6 {
            u64::MAX
        } else {
            (1u64 << (1u32 << n)) - 1
        };
        let dcw: Vec<u64> = if dc_root.is_some() {
            tts[k].clone()
        } else {
            vec![0; words]
        };
        let onw: Vec<Vec<u64>> = (0..k)
            .map(|i| {
                (0..words)
                    .map(|w| {
                        let f = if cnf { !tts[i][w] & valid } else { tts[i][w] };
                        f & !dcw[w]
                    })
                    .collect()
            })
            .collect();
        let on: Vec<Vec<u32>> = onw.iter().map(|w| minterms_of(w, n)).collect();
        let dc = minterms_of(&dcw, n);
        // Minterm -> outputs that may be 1 there.
        let mut tag: HashMap<u32, u64> = HashMap::new();
        for (i, ms) in on.iter().enumerate() {
            for &m in ms {
                *tag.entry(m).or_insert(0) |= 1 << i;
            }
        }
        if !dc.is_empty() {
            let all_outputs = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
            for &m in &dc {
                *tag.entry(m).or_insert(0) |= all_outputs;
            }
        }
        if on.iter().all(|v| v.is_empty()) {
            return Ok((SharedCover::from_lists(vec![Vec::new(); k]), true));
        }
        let primes = multi_prime_implicants(&tag, n)?;
        // Rows: one per (output, ON minterm).
        let mut row_of: Vec<HashMap<u32, usize>> = Vec::with_capacity(k);
        let mut n_rows = 0usize;
        for ms in &on {
            let mut h = HashMap::new();
            for &m in ms {
                h.insert(m, n_rows);
                n_rows += 1;
            }
            row_of.push(h);
        }
        let mut cols: Vec<Cube> = Vec::new();
        let mut col_tag: Vec<u64> = Vec::new();
        let mut col_rows: Vec<Vec<usize>> = Vec::new();
        for (idx, &(p, t)) in primes.iter().enumerate() {
            if idx % 64 == 0 {
                budget()?;
            }
            let mut rows = Vec::new();
            for (i, ms) in on.iter().enumerate() {
                if t >> i & 1 == 0 {
                    continue;
                }
                for &m in ms {
                    if p.covers(m) {
                        rows.push(row_of[i][&m]);
                    }
                }
            }
            if !rows.is_empty() {
                cols.push(p);
                col_tag.push(t);
                col_rows.push(rows);
            }
        }
        let big = (n as u64) * (n_rows as u64) + 1;
        let cost: Vec<u64> = cols.iter().map(|c| big + c.literals(n) as u64).collect();
        let (chosen, ok) = solve_cover(n_rows, &col_rows, &cost, exact == Some(true))?;
        // Assign each chosen cube to the outputs it helps, then drop a
        // cube from an output when the rest of that output already covers
        // its ON-set.
        let mut lists: Vec<Vec<Vec<(usize, bool)>>> = Vec::with_capacity(k);
        for i in 0..k {
            let mut mine: Vec<Cube> = chosen
                .iter()
                .filter(|&&c| col_tag[c] >> i & 1 == 1)
                .map(|&c| cols[c])
                .filter(|p| on[i].iter().any(|&m| p.covers(m)))
                .collect();
            let mut count: HashMap<u32, usize> = HashMap::new();
            for p in &mine {
                for &m in &on[i] {
                    if p.covers(m) {
                        *count.entry(m).or_insert(0) += 1;
                    }
                }
            }
            mine.sort_by_key(|p| std::cmp::Reverse(p.literals(n)));
            let mut kept = Vec::new();
            for p in mine {
                let redundant = on[i]
                    .iter()
                    .filter(|&&m| p.covers(m))
                    .all(|m| count[m] >= 2);
                if redundant {
                    for &m in &on[i] {
                        if p.covers(m) {
                            *count.get_mut(&m).unwrap() -= 1;
                        }
                    }
                } else {
                    kept.push(p);
                }
            }
            kept.sort();
            lists.push(kept.into_iter().map(|c| cube_lits(c, n)).collect());
        }
        let shared = SharedCover::from_lists(lists);
        // Check every output: on_i ⊆ cover_i ⊆ on_i ∪ dc.
        let nwords = words;
        for i in 0..k {
            let mut cw = vec![0u64; nwords];
            for &t in &shared.outputs[i] {
                for m in 0..(1usize << n) {
                    if shared.terms[t]
                        .iter()
                        .all(|&(v, pos)| ((m >> v) & 1 == 1) == pos)
                    {
                        cw[m >> 6] |= 1 << (m & 63);
                    }
                }
                budget()?;
            }
            for w in 0..nwords {
                if onw[i][w] & !cw[w] != 0 || cw[w] & !(onw[i][w] | dcw[w]) != 0 {
                    return Err(BooleanError::Internal(format!(
                        "multi-output cover disagrees with the truth table of output {i}"
                    )));
                }
            }
        }
        return Ok((shared, ok));
    }
    // Heuristic: minimise each output with the espresso-style pass, pool the
    // cubes, record which outputs each cube is allowed to feed, and keep a
    // smallest irredundant subset.  Never more distinct terms than the sum of
    // the per-output covers.
    let mut mgr = BddManager::new();
    for name in &circ.names {
        mgr.declare(name);
    }
    let b = circ.to_bdd(&mut mgr)?;
    let d = dc_root.map(|r| b[r as usize]).unwrap_or(BddRef::FALSE);
    let nd = mgr.not(d)?;
    let mut ons = Vec::with_capacity(k);
    let mut uppers = Vec::with_capacity(k);
    let mut own: Vec<Cover> = Vec::with_capacity(k);
    for &r in roots {
        let f = if cnf {
            mgr.not(b[r as usize])?
        } else {
            b[r as usize]
        };
        let on = mgr.and(f, nd)?;
        let upper = mgr.or(f, d)?;
        own.push(heuristic_sop(&mut mgr, on, upper, n)?);
        ons.push(on);
        uppers.push(upper);
    }
    let pooled = SharedCover::from_lists(own);
    let nt = pooled.terms.len();
    // Which outputs may each pooled cube feed?
    let mut allowed: Vec<Vec<usize>> = Vec::with_capacity(nt);
    let mut tbdd: Vec<BddRef> = Vec::with_capacity(nt);
    for t in &pooled.terms {
        budget()?;
        let mut cube: Vec<Option<bool>> = vec![None; n];
        for &(v, pos) in t {
            cube[v] = Some(pos);
        }
        allowed.push(
            (0..k)
                .filter(|&i| mgr.holds_on_cube(uppers[i], &cube))
                .collect(),
        );
        let mut acc = BddRef::TRUE;
        for &(v, pos) in t {
            let x = mgr.var_at(v as u32)?;
            let lit = if pos { x } else { mgr.not(x)? };
            acc = mgr.and(acc, lit)?;
        }
        tbdd.push(acc);
    }
    let mut selected = vec![true; nt];
    if nt <= 4000 {
        // Try to drop whole terms: fewest outputs served, most literals first.
        let mut order: Vec<usize> = (0..nt).collect();
        order.sort_by_key(|&t| (allowed[t].len(), std::cmp::Reverse(pooled.terms[t].len())));
        for &t in &order {
            budget()?;
            selected[t] = false;
            let mut ok = true;
            for &i in &allowed[t] {
                let mut rest = BddRef::FALSE;
                for (u, &sel) in selected.iter().enumerate() {
                    if sel && allowed[u].contains(&i) {
                        rest = mgr.or(rest, tbdd[u])?;
                    }
                }
                let nrest = mgr.not(rest)?;
                if !mgr.and(ons[i], nrest)?.is_false() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                selected[t] = true;
            }
        }
    }
    // Per output: the selected terms it may use, minus the redundant ones.
    let mut lists: Vec<Vec<Vec<(usize, bool)>>> = Vec::with_capacity(k);
    for i in 0..k {
        let mut mine: Vec<usize> = (0..nt)
            .filter(|&t| selected[t] && allowed[t].contains(&i))
            .collect();
        mine.sort_by_key(|&t| std::cmp::Reverse(pooled.terms[t].len()));
        let mut keep = vec![true; mine.len()];
        if mine.len() <= 4000 {
            for a in 0..mine.len() {
                budget()?;
                keep[a] = false;
                let mut rest = BddRef::FALSE;
                for (c, &t) in mine.iter().enumerate() {
                    if keep[c] {
                        rest = mgr.or(rest, tbdd[t])?;
                    }
                }
                let nrest = mgr.not(rest)?;
                if !mgr.and(ons[i], nrest)?.is_false() {
                    keep[a] = true;
                }
            }
        }
        lists.push(
            mine.iter()
                .zip(&keep)
                .filter(|(_, &kp)| kp)
                .map(|(&t, _)| pooled.terms[t].clone())
                .collect(),
        );
    }
    let shared = SharedCover::from_lists(lists);
    for i in 0..k {
        let cb = cover_bdd(
            &mut mgr,
            &shared.outputs[i]
                .iter()
                .map(|&t| shared.terms[t].clone())
                .collect(),
        )?;
        let ncb = mgr.not(cb)?;
        let miss = mgr.and(ons[i], ncb)?;
        let nup = mgr.not(uppers[i])?;
        let excess = mgr.and(cb, nup)?;
        if !miss.is_false() || !excess.is_false() {
            return Err(BooleanError::Internal(format!(
                "heuristic multi-output cover disagrees with the BDD of output {i}"
            )));
        }
    }
    Ok((shared, false))
}

/// Jointly minimise several propositional formulas to two-level forms that
/// share product terms (the AND plane of a PLA).
///
/// Arguments are as for [`simplify_logic`]; `dontcare` applies to every
/// output.  Up to [`EXACT_AUTO_MAX_VARS`] variables the shared cover is exact:
/// multi-output Quine-McCluskey primes (a cube is tagged with every output it
/// lies inside) and a minimum cover of the (output, minterm) pairs — fewest
/// distinct terms, then fewest literals counted once per distinct term.
/// Beyond that, or with `exact = Some(false)`, each output is minimised with
/// the espresso-style pass and the pooled cubes are cut down to an
/// irredundant shared subset; [`MultiMinimizedLogic::exact`] says which
/// happened.  The result never has more distinct terms than the per-output
/// covers it starts from, and every output is checked against its input
/// before the result is returned.
///
/// Refuses at most [`MULTI_MAX_OUTPUTS`] outputs
/// ([`BooleanError::TooLarge`]) and an empty list
/// ([`BooleanError::InvalidArgument`]).
pub fn simplify_logic_multi(
    pool: &ExprPool,
    exprs: &[ExprId],
    form: LogicForm,
    dontcare: Option<ExprId>,
    exact: Option<bool>,
) -> Result<MultiMinimizedLogic, BooleanError> {
    if exprs.is_empty() {
        return Err(BooleanError::InvalidArgument(
            "minimize_logic needs at least one formula".into(),
        ));
    }
    if exprs.len() > MULTI_MAX_OUTPUTS {
        return Err(BooleanError::TooLarge {
            what: "multi-output minimisation outputs",
            limit: MULTI_MAX_OUTPUTS as u64,
        });
    }
    let mut circ = Circuit::new(true);
    let roots: Vec<u32> = exprs
        .iter()
        .map(|&e| circ.lift(pool, e))
        .collect::<Result<_, _>>()?;
    let dc_root = match dontcare {
        Some(d) => Some(circ.lift(pool, d)?),
        None => None,
    };
    let build = |cnf: bool, circ: &Circuit| -> Result<MultiMinimizedLogic, BooleanError> {
        let (sc, ok) = minimise_multi_side(circ, &roots, dc_root, cnf, exact)?;
        let constant = |t: &Vec<(usize, bool)>| t.is_empty();
        let mut out_exprs = Vec::with_capacity(roots.len());
        let mut per_output = Vec::with_capacity(roots.len());
        let mut users = vec![0usize; sc.terms.len()];
        for o in &sc.outputs {
            let cover: Cover = o.iter().map(|&t| sc.terms[t].clone()).collect();
            out_exprs.push(cover_to_expr(pool, &circ.atoms, &cover, cnf));
            let is_const = cover.is_empty() || cover.iter().any(constant);
            per_output.push(if is_const { 0 } else { cover.len() });
            if !is_const {
                for &t in o {
                    users[t] += 1;
                }
            }
        }
        let live: Vec<usize> = (0..sc.terms.len()).filter(|&t| users[t] > 0).collect();
        Ok(MultiMinimizedLogic {
            exprs: out_exprs,
            form: if cnf { LogicForm::Cnf } else { LogicForm::Dnf },
            exact: ok,
            terms: live.len(),
            literals: live.iter().map(|&t| sc.terms[t].len()).sum(),
            per_output_terms: per_output,
            shared_terms: live.iter().filter(|&&t| users[t] > 1).count(),
        })
    };
    match form {
        LogicForm::Dnf => build(false, &circ),
        LogicForm::Cnf => build(true, &circ),
        LogicForm::Auto => {
            let d = build(false, &circ)?;
            let c = build(true, &circ)?;
            Ok(if (c.literals, c.terms) < (d.literals, d.terms) {
                c
            } else {
                d
            })
        }
    }
}

// ---------------------------------------------------------------------------
// BDDs of formulas
// ---------------------------------------------------------------------------

/// Build the ROBDD of `expr` in `mgr`.
///
/// Variables already declared in `mgr` keep their levels (so a caller fixes
/// an order by declaring names first); the formula's other atoms are appended
/// in depth-first first-visit order over the canonically sorted formula — a
/// cheap structural heuristic that interleaves the operand bits of
/// bit-sliced arithmetic (`a₀ b₀ a₁ b₁ …`), the order adders and comparators
/// need.  Returns the root and the `(name, atom)` pairs of the formula's
/// variables.  Only Boolean symbols are accepted as atoms.
pub fn bdd_from_expr(
    pool: &ExprPool,
    expr: ExprId,
    mgr: &mut BddManager,
) -> Result<(BddRef, Vec<(String, ExprId)>), BooleanError> {
    let (circ, root) = lift_one(pool, expr, false)?;
    for n in &circ.names {
        mgr.declare(n);
    }
    let b = circ.to_bdd(mgr)?;
    let atoms = circ
        .names
        .iter()
        .cloned()
        .zip(circ.atoms.iter().copied())
        .collect();
    Ok((b[root as usize], atoms))
}

/// Re-express a BDD as a formula by Shannon expansion at every node
/// (`(x ∧ hi) ∨ (¬x ∧ lo)`, folded at the terminals).  `atom_of(name)` maps a
/// manager variable back to its atom.
pub fn bdd_to_expr(
    pool: &ExprPool,
    mgr: &BddManager,
    f: BddRef,
    atom_of: &dyn Fn(&str) -> Option<ExprId>,
) -> Result<ExprId, BooleanError> {
    let mut memo: HashMap<BddRef, ExprId> = HashMap::new();
    fn go(
        pool: &ExprPool,
        mgr: &BddManager,
        f: BddRef,
        atom_of: &dyn Fn(&str) -> Option<ExprId>,
        memo: &mut HashMap<BddRef, ExprId>,
    ) -> Result<ExprId, BooleanError> {
        if f.is_true() {
            return Ok(pool.pred_true());
        }
        if f.is_false() {
            return Ok(pool.pred_false());
        }
        if let Some(&e) = memo.get(&f) {
            return Ok(e);
        }
        let level = mgr.top_level(f).expect("non-terminal");
        let (lo, hi) = mgr.children(f).expect("non-terminal");
        let name = &mgr.var_names()[level as usize];
        let x = atom_of(name).ok_or_else(|| {
            BooleanError::InvalidArgument(format!("no atom is known for BDD variable '{name}'"))
        })?;
        let el = go(pool, mgr, lo, atom_of, memo)?;
        let eh = go(pool, mgr, hi, atom_of, memo)?;
        // Fold the terminal cases ite(x, ⊤, l) = x ∨ l etc.
        let e = if hi.is_true() {
            logic_or(pool, &[x, el])
        } else if hi.is_false() {
            logic_and(pool, &[logic_not(pool, x), el])
        } else if lo.is_true() {
            logic_or(pool, &[logic_not(pool, x), eh])
        } else if lo.is_false() {
            logic_and(pool, &[x, eh])
        } else {
            logic_ite(pool, x, eh, el)
        };
        memo.insert(f, e);
        Ok(e)
    }
    go(pool, mgr, f, atom_of, &mut memo)
}

// ---------------------------------------------------------------------------
// Equivalence, truth tables, SAT over circuits
// ---------------------------------------------------------------------------

/// Method of [`equivalent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EquivMethod {
    /// Truth table up to 16 variables, else BDD (bounded), else SAT.
    Auto,
    /// Exhaustive truth table (≤ [`TRUTH_TABLE_MAX_VARS`] variables).
    TruthTable,
    /// Canonical ROBDDs (bounded by the manager's node ceiling).
    Bdd,
    /// SAT on the miter `f ⊕ g` (DPLL over a Tseitin encoding).
    Sat,
}

impl EquivMethod {
    /// Parse `"auto"` / `"truth_table"` / `"bdd"` / `"sat"`.
    pub fn parse(s: &str) -> Result<Self, BooleanError> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(EquivMethod::Auto),
            "truth_table" | "tt" | "truthtable" => Ok(EquivMethod::TruthTable),
            "bdd" => Ok(EquivMethod::Bdd),
            "sat" => Ok(EquivMethod::Sat),
            other => Err(BooleanError::InvalidArgument(format!(
                "method must be 'auto', 'truth_table', 'bdd' or 'sat', not '{other}'"
            ))),
        }
    }
}

/// Verdict of [`equivalent`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Equivalence {
    /// The formulas agree on every assignment.
    Equivalent,
    /// An assignment (atom name → value) on which they differ.
    Counterexample(Vec<(String, bool)>),
}

/// Node ceiling of the BDD attempt inside [`EquivMethod::Auto`].
const AUTO_BDD_NODES: usize = 200_000;
/// Decision budget of the SAT path.
const SAT_DECISIONS: u64 = 2_000_000;

/// Tseitin-encode the circuit cone of `root` and run the budgeted DPLL.
/// `Ok(Some(model))` is a model over the circuit's variables, `Ok(None)` is
/// a proof of unsatisfiability.
fn circuit_sat(c: &Circuit, root: u32, decisions: u64) -> Result<Option<Vec<bool>>, BooleanError> {
    let n = c.n_vars() as i32;
    let mut lit: Vec<i32> = vec![0; c.gates.len()];
    let mut next = n;
    let mut clauses: Vec<BoolClause> = Vec::new();
    // Gates are topologically ordered, so a single forward pass suffices
    // (gates outside the cone of `root` are encoded too; harmless).
    for (i, g) in c.gates.iter().enumerate() {
        lit[i] = match g {
            Gate::Var(x) => *x as i32 + 1,
            Gate::Not(x) => -lit[*x as usize],
            Gate::Const(b) => {
                next += 1;
                clauses.push(vec![if *b { next } else { -next }]);
                next
            }
            Gate::And(cs) => {
                next += 1;
                let gv = next;
                let mut long = vec![gv];
                for ch in cs {
                    let l = lit[*ch as usize];
                    clauses.push(vec![-gv, l]);
                    long.push(-l);
                }
                clauses.push(long);
                gv
            }
            Gate::Or(cs) => {
                next += 1;
                let gv = next;
                let mut long = vec![-gv];
                for ch in cs {
                    let l = lit[*ch as usize];
                    clauses.push(vec![gv, -l]);
                    long.push(l);
                }
                clauses.push(long);
                gv
            }
        };
    }
    clauses.push(vec![lit[root as usize]]);
    match dpll_sat_bounded(clauses, next as u32, decisions) {
        DpllOutcome::Unsat => Ok(None),
        DpllOutcome::Exhausted => Err(BooleanError::TooLarge {
            what: "SAT decisions",
            limit: decisions,
        }),
        DpllOutcome::Sat(model) => {
            let m: Vec<bool> = model[..c.n_vars()].to_vec();
            if !c.eval_one(&m)[root as usize] {
                return Err(BooleanError::Internal(
                    "SAT model does not satisfy the formula".into(),
                ));
            }
            Ok(Some(m))
        }
    }
}

/// Decide whether two propositional formulas are equivalent.
///
/// Atoms may be Boolean symbols or relations.  With relations present, an
/// `Equivalent` verdict is sound (propositional equivalence implies
/// equivalence under every interpretation), but a propositional
/// counterexample may be unrealisable (`x > 0` and `¬(x ≤ 0)` are
/// propositionally independent), so that case is refused with
/// [`BooleanError::NotPropositional`] instead of being reported as
/// "not equivalent".
pub fn equivalent(
    pool: &ExprPool,
    f: ExprId,
    g: ExprId,
    method: EquivMethod,
) -> Result<Equivalence, BooleanError> {
    let mut circ = Circuit::new(true);
    let rf = circ.lift(pool, f)?;
    let rg = circ.lift(pool, g)?;
    let n = circ.n_vars();
    let method = match method {
        EquivMethod::Auto if n <= 16 => EquivMethod::TruthTable,
        m => m,
    };
    let cex: Option<Vec<bool>> = match method {
        EquivMethod::TruthTable => {
            let t = circ.truth_tables(&[rf, rg])?;
            let mut found = None;
            for (w, (a, b)) in t[0].iter().zip(&t[1]).enumerate() {
                let d = a ^ b;
                if d != 0 {
                    let m = w * 64 + d.trailing_zeros() as usize;
                    found = Some((0..n).map(|i| (m >> i) & 1 == 1).collect());
                    break;
                }
            }
            found
        }
        EquivMethod::Bdd | EquivMethod::Auto => {
            let cap = if method == EquivMethod::Auto {
                AUTO_BDD_NODES
            } else {
                super::bdd::DEFAULT_MAX_BDD_NODES
            };
            let mut mgr = BddManager::with_max_nodes(cap);
            for nm in &circ.names {
                mgr.declare(nm);
            }
            let built = circ.to_bdd(&mut mgr).and_then(|b| {
                let d = mgr.xor(b[rf as usize], b[rg as usize])?;
                Ok(d)
            });
            match built {
                Ok(d) => mgr.any_sat(d).map(|partial| {
                    let mut m = vec![false; n];
                    for (l, v) in partial {
                        m[l as usize] = v;
                    }
                    m
                }),
                Err(BooleanError::TooLarge { .. }) if method == EquivMethod::Auto => {
                    miter_sat(&mut circ, rf, rg)?
                }
                Err(e) => return Err(e),
            }
        }
        EquivMethod::Sat => miter_sat(&mut circ, rf, rg)?,
    };
    match cex {
        None => Ok(Equivalence::Equivalent),
        Some(m) => {
            let vals = circ.eval_one(&m);
            if vals[rf as usize] == vals[rg as usize] {
                return Err(BooleanError::Internal(
                    "counterexample does not separate the formulas".into(),
                ));
            }
            if circ.has_relations {
                return Err(BooleanError::NotPropositional(
                    "the formulas differ propositionally, but they contain arithmetic relations whose truth values may be dependent; decide them with decide() or alkahest.smt".into(),
                ));
            }
            Ok(Equivalence::Counterexample(
                circ.names.iter().cloned().zip(m).collect(),
            ))
        }
    }
}

fn miter_sat(circ: &mut Circuit, rf: u32, rg: u32) -> Result<Option<Vec<bool>>, BooleanError> {
    let nf = circ.push(Gate::Not(rf))?;
    let ng = circ.push(Gate::Not(rg))?;
    let a = circ.push(Gate::And(vec![rf, ng]))?;
    let b = circ.push(Gate::And(vec![nf, rg]))?;
    let x = circ.push(Gate::Or(vec![a, b]))?;
    circuit_sat(circ, x, SAT_DECISIONS)
}

/// Truth table of `expr` over `variables` (all of the formula's symbols, in
/// the given order; `None` sorts them by name).  Row `m` assigns variable `i`
/// the value of bit `n − 1 − i` of `m`, so rows run from all-false to
/// all-true in the usual textbook order.  Returns `(variable names, values)`.
pub fn truth_table(
    pool: &ExprPool,
    expr: ExprId,
    variables: Option<&[ExprId]>,
) -> Result<(Vec<String>, Vec<bool>), BooleanError> {
    let mut circ = Circuit::new(false);
    match variables {
        Some(vs) => {
            for &v in vs {
                if !matches!(pool.get(v), ExprData::Symbol { .. }) {
                    return Err(BooleanError::InvalidArgument(format!(
                        "'{}' is not a symbol",
                        pool.display(v)
                    )));
                }
                circ.declare_atom(pool, v)?;
            }
        }
        None => {
            // Lift once to discover the atoms, then re-declare them sorted.
            let (probe, _) = lift_one(pool, expr, false)?;
            let mut named: Vec<(String, ExprId)> = probe
                .names
                .iter()
                .cloned()
                .zip(probe.atoms.iter().copied())
                .collect();
            named.sort();
            for (_, a) in named {
                circ.declare_atom(pool, a)?;
            }
        }
    }
    let declared = circ.n_vars();
    let root = circ.lift(pool, expr)?;
    if circ.n_vars() != declared {
        return Err(BooleanError::InvalidArgument(format!(
            "the formula has variables not listed: {}",
            circ.names[declared..].join(", ")
        )));
    }
    let n = declared;
    let t = circ.truth_tables(&[root])?;
    let mut rows = Vec::with_capacity(1 << n);
    for r in 0..(1usize << n) {
        // Row r: variable i = bit (n-1-i) of r; circuit minterm m: var i = bit i.
        let mut m = 0usize;
        for i in 0..n {
            if (r >> (n - 1 - i)) & 1 == 1 {
                m |= 1 << i;
            }
        }
        rows.push((t[0][m >> 6] >> (m & 63)) & 1 == 1);
    }
    Ok((circ.names.clone(), rows))
}

/// Satisfiability of a purely propositional formula through the circuit
/// (DAG) encoding.  `None` when `expr` is not purely propositional.
pub(crate) fn satisfiable_circuit(
    expr: ExprId,
    pool: &ExprPool,
    decisions: u64,
) -> Option<super::Satisfiability> {
    let (c, root) = lift_one(pool, expr, false).ok()?;
    Some(match circuit_sat(&c, root, decisions) {
        Ok(None) => super::Satisfiability::Unsat,
        Ok(Some(m)) => super::Satisfiability::Sat(
            c.names
                .iter()
                .cloned()
                .zip(m)
                .map(|(k, v)| {
                    (
                        k,
                        if v {
                            super::BOOL_TRUE
                        } else {
                            super::BOOL_FALSE
                        }
                        .to_string(),
                    )
                })
                .collect(),
        ),
        Err(_) => super::Satisfiability::Unknown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn syms(p: &ExprPool, n: usize) -> Vec<ExprId> {
        (0..n)
            .map(|i| p.symbol(format!("x{i}"), Domain::Real))
            .collect()
    }

    /// Tiny deterministic PRNG for random formulas.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn random_formula(p: &ExprPool, vs: &[ExprId], rng: &mut Lcg, depth: u32) -> ExprId {
        if depth == 0 || rng.next() % 4 == 0 {
            let v = vs[(rng.next() as usize) % vs.len()];
            return if rng.next() % 2 == 0 {
                v
            } else {
                logic_not(p, v)
            };
        }
        let k = 2 + (rng.next() % 2) as usize;
        let args: Vec<ExprId> = (0..k)
            .map(|_| random_formula(p, vs, rng, depth - 1))
            .collect();
        match rng.next() % 5 {
            0 => logic_and(p, &args),
            1 => logic_or(p, &args),
            2 => logic_xor(p, &args),
            3 => logic_implies(p, args[0], args[1]),
            _ => logic_not(p, logic_and(p, &args)),
        }
    }

    fn brute(p: &ExprPool, f: ExprId, vs: &[ExprId]) -> Vec<bool> {
        let mut c = Circuit::new(false);
        for &v in vs {
            c.declare_atom(p, v).unwrap();
        }
        let r = c.lift(p, f).unwrap();
        (0..(1usize << vs.len()))
            .map(|m| {
                let a: Vec<bool> = (0..vs.len()).map(|i| (m >> i) & 1 == 1).collect();
                c.eval_one(&a)[r as usize]
            })
            .collect()
    }

    #[test]
    fn nary_flattens_sorts_and_folds() {
        let p = ExprPool::new();
        let v = syms(&p, 3);
        let a = logic_and(&p, &[v[2], logic_and(&p, &[v[1], v[0]])]);
        let b = logic_and(&p, &[v[0], v[1], v[2]]);
        assert_eq!(a, b);
        assert_eq!(pred_args(&p, a), vec![v[0], v[1], v[2]]);
        assert_eq!(logic_and(&p, &[v[0], logic_not(&p, v[0])]), p.pred_false());
        assert_eq!(logic_or(&p, &[v[0], p.pred_true()]), p.pred_true());
        assert_eq!(logic_and(&p, &[v[0], v[0]]), v[0]);
        assert_eq!(logic_not(&p, logic_not(&p, v[1])), v[1]);
        assert_eq!(logic_xor(&p, &[v[0], v[0]]), p.pred_false());
    }

    #[test]
    fn simplify_textbook_case() {
        let p = ExprPool::new();
        let v = syms(&p, 2);
        let (a, b) = (v[0], v[1]);
        let f = logic_or(
            &p,
            &[
                logic_and(&p, &[a, b]),
                logic_and(&p, &[a, logic_not(&p, b)]),
            ],
        );
        let r = simplify_logic(&p, f, LogicForm::Auto, None, None).unwrap();
        assert_eq!(r.expr, a);
        assert!(r.exact);
    }

    #[test]
    fn random_formulas_minimise_correctly_both_paths() {
        let mut rng = Lcg(7);
        for trial in 0..60 {
            let p = ExprPool::new();
            let n = 2 + trial % 7;
            let v = syms(&p, n);
            let f = random_formula(&p, &v, &mut rng, 4);
            let want = brute(&p, f, &v);
            for form in [LogicForm::Dnf, LogicForm::Cnf] {
                for exact in [None, Some(false)] {
                    let r = simplify_logic(&p, f, form, None, exact).unwrap();
                    assert_eq!(
                        brute(&p, r.expr, &v),
                        want,
                        "trial {trial} {form:?} {exact:?}"
                    );
                }
            }
            for nf in [
                to_cnf(&p, f).unwrap(),
                to_dnf(&p, f).unwrap(),
                to_nnf(&p, f).unwrap(),
            ] {
                assert_eq!(brute(&p, nf, &v), want);
            }
            let x = random_formula(&p, &v, &mut rng, 3);
            let same = brute(&p, x, &v) == want;
            for m in [EquivMethod::TruthTable, EquivMethod::Bdd, EquivMethod::Sat] {
                let e = equivalent(&p, f, x, m).unwrap();
                assert_eq!(matches!(e, Equivalence::Equivalent), same, "{m:?}");
            }
        }
    }

    #[test]
    fn exact_beats_or_matches_heuristic() {
        let mut rng = Lcg(99);
        for _ in 0..30 {
            let p = ExprPool::new();
            let v = syms(&p, 6);
            let f = random_formula(&p, &v, &mut rng, 5);
            let e = simplify_logic(&p, f, LogicForm::Dnf, None, Some(true)).unwrap();
            let h = simplify_logic(&p, f, LogicForm::Dnf, None, Some(false)).unwrap();
            assert!((e.terms, e.literals) <= (h.terms, h.literals));
        }
    }

    #[test]
    fn dont_cares_are_used() {
        // BCD "≥ 5" on 4 bits with codes 10..15 as don't cares: minimum is
        // x3 ∨ x2∧x0 ∨ x2∧x1 (3 terms, 5 literals).
        let p = ExprPool::new();
        let v = syms(&p, 4);
        let minterm = |m: usize| {
            let lits: Vec<ExprId> = (0..4)
                .map(|i| {
                    if (m >> i) & 1 == 1 {
                        v[i]
                    } else {
                        logic_not(&p, v[i])
                    }
                })
                .collect();
            logic_and(&p, &lits)
        };
        let f = logic_or(&p, &(5..10).map(minterm).collect::<Vec<_>>());
        let dc = logic_or(&p, &(10..16).map(minterm).collect::<Vec<_>>());
        let r = simplify_logic(&p, f, LogicForm::Dnf, Some(dc), None).unwrap();
        assert!(r.exact);
        assert_eq!((r.terms, r.literals), (3, 5));
    }

    #[test]
    fn multi_output_matches_each_output_and_never_costs_more() {
        let mut rng = Lcg(2024);
        for trial in 0..60 {
            let p = ExprPool::new();
            let n = 3 + trial % 5;
            let v = syms(&p, n);
            let k = 2 + trial % 3;
            let fs: Vec<ExprId> = (0..k)
                .map(|_| random_formula(&p, &v, &mut rng, 4))
                .collect();
            for form in [LogicForm::Dnf, LogicForm::Cnf] {
                for exact in [None, Some(false)] {
                    let m = simplify_logic_multi(&p, &fs, form, None, exact).unwrap();
                    assert_eq!(m.exprs.len(), k);
                    assert_eq!(m.exact, exact.is_none(), "trial {trial}");
                    let mut sum = 0;
                    for (i, &f) in fs.iter().enumerate() {
                        assert_eq!(
                            brute(&p, m.exprs[i], &v),
                            brute(&p, f, &v),
                            "trial {trial} output {i} {form:?} {exact:?}"
                        );
                        sum += simplify_logic(&p, f, form, None, exact).unwrap().terms;
                    }
                    assert!(
                        m.terms <= sum,
                        "trial {trial} {form:?} {exact:?}: {} shared vs {sum} separate",
                        m.terms
                    );
                    assert!(m.per_output_terms.iter().sum::<usize>() >= m.terms);
                }
            }
        }
    }

    #[test]
    fn multi_output_shares_terms_on_a_half_adder_pair() {
        // sum = a⊕b, carry = a∧b, and a∧b ∨ a∧b∧c (= a∧b): the term a·b is
        // reused, so the shared cover has strictly fewer terms.
        let p = ExprPool::new();
        let v = syms(&p, 3);
        let (a, b, c) = (v[0], v[1], v[2]);
        let sum = logic_xor(&p, &[a, b]);
        let carry = logic_and(&p, &[a, b]);
        let both = logic_and(&p, &[a, b, c]);
        let carry_or = logic_or(&p, &[carry, both]);
        let fs = [sum, carry, carry_or];
        let m = simplify_logic_multi(&p, &fs, LogicForm::Dnf, None, Some(true)).unwrap();
        assert!(m.exact);
        let separate: usize = fs
            .iter()
            .map(|&f| {
                simplify_logic(&p, f, LogicForm::Dnf, None, Some(true))
                    .unwrap()
                    .terms
            })
            .sum();
        assert_eq!(separate, 4);
        assert_eq!(m.terms, 3);
        assert_eq!(m.shared_terms, 1);
        // Two copies of one function share its single term.
        let again = simplify_logic_multi(&p, &[carry, carry], LogicForm::Dnf, None, None).unwrap();
        assert_eq!((again.terms, again.shared_terms), (1, 1));
    }

    #[test]
    fn multi_output_dont_cares_and_refusals() {
        let p = ExprPool::new();
        let v = syms(&p, 4);
        let minterm = |m: usize| {
            let lits: Vec<ExprId> = (0..4)
                .map(|i| {
                    if (m >> i) & 1 == 1 {
                        v[i]
                    } else {
                        logic_not(&p, v[i])
                    }
                })
                .collect();
            logic_and(&p, &lits)
        };
        let f1 = logic_or(&p, &(5..10).map(minterm).collect::<Vec<_>>());
        let f2 = logic_or(&p, &(7..10).map(minterm).collect::<Vec<_>>());
        let dc = logic_or(&p, &(10..16).map(minterm).collect::<Vec<_>>());
        let m = simplify_logic_multi(&p, &[f1, f2], LogicForm::Dnf, Some(dc), None).unwrap();
        assert!(m.exact);
        for (i, f) in [f1, f2].into_iter().enumerate() {
            let got = brute(&p, m.exprs[i], &v);
            let want = brute(&p, f, &v);
            let dcv = brute(&p, dc, &v);
            for k in 0..16 {
                assert!(dcv[k] || got[k] == want[k], "output {i} minterm {k}");
            }
        }
        assert!(matches!(
            simplify_logic_multi(&p, &[], LogicForm::Dnf, None, None),
            Err(BooleanError::InvalidArgument(_))
        ));
        let many = vec![f1; MULTI_MAX_OUTPUTS + 1];
        assert!(matches!(
            simplify_logic_multi(&p, &many, LogicForm::Dnf, None, None),
            Err(BooleanError::TooLarge { .. })
        ));
        let q = ExprPool::new();
        let big = syms(&q, 17);
        let x = logic_xor(&q, &big[..3]);
        let y = logic_and(&q, &big);
        assert!(matches!(
            simplify_logic_multi(&q, &[x, y], LogicForm::Dnf, None, Some(true)),
            Err(BooleanError::ExactUnavailable(_))
        ));
    }

    #[test]
    fn xor_chain_is_linear_and_decided() {
        let p = ExprPool::new();
        let v = syms(&p, 24);
        let x = logic_xor(&p, &v);
        let mut mgr = BddManager::new();
        let (b, _) = bdd_from_expr(&p, x, &mut mgr).unwrap();
        assert_eq!(mgr.node_count(b), 2 * 24 - 1);
        assert_eq!(mgr.sat_count(b, 24), rug::Integer::from(1u32 << 23));
        let s = satisfiable_circuit(x, &p, 1_000_000).unwrap();
        assert!(matches!(s, super::super::Satisfiability::Sat(_)));
    }

    #[test]
    fn truth_table_order() {
        let p = ExprPool::new();
        let v = syms(&p, 2);
        let f = logic_implies(&p, v[0], v[1]);
        let (names, rows) = truth_table(&p, f, None).unwrap();
        assert_eq!(names, vec!["x0", "x1"]);
        assert_eq!(rows, vec![true, true, false, true]);
    }
}
