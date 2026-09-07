//! First-order formulas over symbolic reals (V3-3 / FOFormula).
//!
//! - [`Formula`] is the algebraic view over kernel predicates and quantifiers.
//! - [`satisfiable`] decides two disjoint quantifier-free fragments and answers
//!   [`Satisfiability::Unknown`] for everything else:
//!   1. **Purely propositional** — every leaf is a bare [`ExprData::Symbol`] or
//!      a `True`/`False` constant, combined with `And` / `Or` / `Not`.  This is
//!      decided *completely* (Tseitin + a budgeted DPLL): every such
//!      formula gets `Sat`/`Unsat`, never `Unknown`, up to the search budget.
//!   2. **Single-variable interval arithmetic** — conjunctions of comparisons
//!      between **one** real symbol and a rational constant, plus `Or` / `Not`
//!      (via NNF on relations).  Incomplete; `Unknown` is common here.
//!
//!   A formula that *mixes* the two (a Boolean symbol and an arithmetic
//!   relation in the same formula) is deliberately `Unknown`: deciding it needs
//!   a Boolean-plus-theory combination this module does not implement, and the
//!   only cheap alternative — abstracting each relation to a fresh proposition —
//!   is sound for `Unsat` only and would report `x > 0 ∧ x < 0` as *sat*.  Route
//!   those to [`crate::real::decide`] (CAD) or to `alkahest.smt`.
//! - [`smtlib`] exports a [`Formula`] as SMT-LIB 2 text for an external solver
//!   (P2-3).  `alkahest.smt` drives the solver process from Python.

pub mod smtlib;

use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure to interpret an expression as a boolean formula.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicError {
    /// The expression node is not supported in [`formula_from_expr`].
    UnsupportedExpr(&'static str),
}

impl fmt::Display for LogicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicError::UnsupportedExpr(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for LogicError {}

impl crate::errors::AlkahestError for LogicError {
    fn code(&self) -> &'static str {
        "E-LOGIC-001"
    }
}

// ---------------------------------------------------------------------------
// Formula
// ---------------------------------------------------------------------------

/// First-order formula; [`Formula::Atom`] wraps kernel [`PredicateKind`] + operands.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Formula {
    Atom {
        kind: PredicateKind,
        args: Vec<ExprId>,
    },
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Not(Box<Formula>),
    True,
    False,
    Forall {
        var: ExprId,
        body: Box<Formula>,
    },
    Exists {
        var: ExprId,
        body: Box<Formula>,
    },
}

impl Formula {
    pub fn and(a: Formula, b: Formula) -> Self {
        Formula::And(Box::new(a), Box::new(b))
    }

    pub fn or(a: Formula, b: Formula) -> Self {
        Formula::Or(Box::new(a), Box::new(b))
    }

    #[allow(clippy::should_implement_trait)] // `Not::not` would force unary `!` semantics at call sites we avoid for clarity.
    pub fn not(a: Formula) -> Self {
        Formula::Not(Box::new(a))
    }

    /// Intern this formula as [`ExprData`] nodes.
    pub fn to_expr(&self, pool: &ExprPool) -> ExprId {
        match self {
            Formula::True => pool.pred_true(),
            Formula::False => pool.pred_false(),
            Formula::Atom { kind, args } => pool.predicate(kind.clone(), args.clone()),
            Formula::And(l, r) => pool.pred_and(vec![l.to_expr(pool), r.to_expr(pool)]),
            Formula::Or(l, r) => pool.pred_or(vec![l.to_expr(pool), r.to_expr(pool)]),
            Formula::Not(x) => pool.pred_not(x.to_expr(pool)),
            Formula::Forall { var, body } => pool.forall(*var, body.to_expr(pool)),
            Formula::Exists { var, body } => pool.exists(*var, body.to_expr(pool)),
        }
    }
}

/// Lift a predicate (or quantified) `ExprId` into a structured [`Formula`].
pub fn formula_from_expr(expr: ExprId, pool: &ExprPool) -> Result<Formula, LogicError> {
    match pool.get(expr) {
        ExprData::Predicate { kind, args } => match kind {
            PredicateKind::True => Ok(Formula::True),
            PredicateKind::False => Ok(Formula::False),
            PredicateKind::And => {
                if args.is_empty() {
                    Ok(Formula::True)
                } else {
                    let mut it = args.into_iter();
                    let first = formula_from_expr(it.next().unwrap(), pool)?;
                    it.try_fold(first, |acc, e| {
                        Ok(Formula::and(acc, formula_from_expr(e, pool)?))
                    })
                }
            }
            PredicateKind::Or => {
                if args.is_empty() {
                    Ok(Formula::False)
                } else {
                    let mut it = args.into_iter();
                    let first = formula_from_expr(it.next().unwrap(), pool)?;
                    it.try_fold(first, |acc, e| {
                        Ok(Formula::or(acc, formula_from_expr(e, pool)?))
                    })
                }
            }
            PredicateKind::Not => {
                if args.len() != 1 {
                    return Err(LogicError::UnsupportedExpr("Not predicate arity must be 1"));
                }
                Ok(Formula::not(formula_from_expr(args[0], pool)?))
            }
            PredicateKind::Lt
            | PredicateKind::Le
            | PredicateKind::Gt
            | PredicateKind::Ge
            | PredicateKind::Eq
            | PredicateKind::Ne => {
                if args.len() != 2 {
                    return Err(LogicError::UnsupportedExpr("relation arity must be 2"));
                }
                Ok(Formula::Atom { kind, args })
            }
        },
        ExprData::Forall { var, body } => Ok(Formula::Forall {
            var,
            body: Box::new(formula_from_expr(body, pool)?),
        }),
        ExprData::Exists { var, body } => Ok(Formula::Exists {
            var,
            body: Box::new(formula_from_expr(body, pool)?),
        }),
        _ => Err(LogicError::UnsupportedExpr(
            "expression is not a predicate or quantified formula",
        )),
    }
}

// ---------------------------------------------------------------------------
// Intervals (one variable)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Bound {
    Lower { val: rug::Rational, strict: bool },
    Upper { val: rug::Rational, strict: bool },
}

#[derive(Clone, Debug, Default)]
struct VarInterval {
    lower: Option<Bound>,
    upper: Option<Bound>,
}

impl VarInterval {
    fn is_empty(&self) -> bool {
        match (&self.lower, &self.upper) {
            (
                Some(Bound::Lower {
                    val: lo,
                    strict: ls,
                }),
                Some(Bound::Upper {
                    val: hi,
                    strict: us,
                }),
            ) => {
                if lo > hi {
                    return true;
                }
                if lo < hi {
                    return false;
                }
                *ls || *us
            }
            _ => false,
        }
    }

    fn intersect(&self, other: &VarInterval) -> Option<VarInterval> {
        let lower = match (&self.lower, &other.lower) {
            (None, b) => b.clone(),
            (a, None) => a.clone(),
            (
                Some(Bound::Lower { val: a, strict: sa }),
                Some(Bound::Lower { val: b, strict: sb }),
            ) => {
                if a > b {
                    Some(Bound::Lower {
                        val: a.clone(),
                        strict: *sa,
                    })
                } else if b > a {
                    Some(Bound::Lower {
                        val: b.clone(),
                        strict: *sb,
                    })
                } else {
                    Some(Bound::Lower {
                        val: a.clone(),
                        strict: *sa || *sb,
                    })
                }
            }
            _ => return None,
        };
        let upper = match (&self.upper, &other.upper) {
            (None, b) => b.clone(),
            (a, None) => a.clone(),
            (
                Some(Bound::Upper { val: a, strict: sa }),
                Some(Bound::Upper { val: b, strict: sb }),
            ) => {
                if a < b {
                    Some(Bound::Upper {
                        val: a.clone(),
                        strict: *sa,
                    })
                } else if b < a {
                    Some(Bound::Upper {
                        val: b.clone(),
                        strict: *sb,
                    })
                } else {
                    Some(Bound::Upper {
                        val: a.clone(),
                        strict: *sa || *sb,
                    })
                }
            }
            _ => return None,
        };
        let r = VarInterval { lower, upper };
        if r.is_empty() {
            None
        } else {
            Some(r)
        }
    }
}

fn rat_atom(pool: &ExprPool, id: ExprId) -> Option<rug::Rational> {
    match pool.get(id) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0)),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

fn symbol_key(pool: &ExprPool, id: ExprId) -> Option<String> {
    pool.with(id, |d| match d {
        ExprData::Symbol { name, .. } => Some(name.clone()),
        _ => None,
    })
}

fn atom_to_interval(
    pool: &ExprPool,
    kind: PredicateKind,
    args: &[ExprId],
) -> Option<(ExprId, VarInterval)> {
    if args.len() != 2 {
        return None;
    }
    let (a, b) = (args[0], args[1]);
    let (var, c_id, swapped) = if symbol_key(pool, a).is_some() && rat_atom(pool, b).is_some() {
        (a, b, false)
    } else if rat_atom(pool, a).is_some() && symbol_key(pool, b).is_some() {
        (b, a, true)
    } else {
        return None;
    };
    let c = rat_atom(pool, c_id)?;
    let iv = match (kind, swapped) {
        (PredicateKind::Lt, false) => VarInterval {
            lower: None,
            upper: Some(Bound::Upper {
                val: c,
                strict: true,
            }),
        },
        (PredicateKind::Le, false) => VarInterval {
            lower: None,
            upper: Some(Bound::Upper {
                val: c,
                strict: false,
            }),
        },
        (PredicateKind::Gt, false) => VarInterval {
            lower: Some(Bound::Lower {
                val: c,
                strict: true,
            }),
            upper: None,
        },
        (PredicateKind::Ge, false) => VarInterval {
            lower: Some(Bound::Lower {
                val: c,
                strict: false,
            }),
            upper: None,
        },
        (PredicateKind::Eq, false) => VarInterval {
            lower: Some(Bound::Lower {
                val: c.clone(),
                strict: false,
            }),
            upper: Some(Bound::Upper {
                val: c,
                strict: false,
            }),
        },
        (PredicateKind::Lt, true) => VarInterval {
            lower: Some(Bound::Lower {
                val: c,
                strict: true,
            }),
            upper: None,
        },
        (PredicateKind::Le, true) => VarInterval {
            lower: Some(Bound::Lower {
                val: c,
                strict: false,
            }),
            upper: None,
        },
        (PredicateKind::Gt, true) => VarInterval {
            lower: None,
            upper: Some(Bound::Upper {
                val: c,
                strict: true,
            }),
        },
        (PredicateKind::Ge, true) => VarInterval {
            lower: None,
            upper: Some(Bound::Upper {
                val: c,
                strict: false,
            }),
        },
        _ => return None,
    };
    Some((var, iv))
}

fn is_rel(k: &PredicateKind) -> bool {
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

fn dual_kind(kind: PredicateKind) -> PredicateKind {
    use PredicateKind::*;
    match kind {
        Lt => Ge,
        Le => Gt,
        Gt => Le,
        Ge => Lt,
        Eq => Ne,
        Ne => Eq,
        other => other,
    }
}

fn nnf(f: Formula) -> Formula {
    match f {
        Formula::Not(inner) => match *inner {
            Formula::True => Formula::False,
            Formula::False => Formula::True,
            Formula::Not(g) => nnf(*g),
            Formula::And(a, b) => nnf(Formula::or(Formula::not(*a), Formula::not(*b))),
            Formula::Or(a, b) => nnf(Formula::and(Formula::not(*a), Formula::not(*b))),
            Formula::Forall { var, body } => nnf(Formula::Exists {
                var,
                body: Box::new(Formula::not(*body)),
            }),
            Formula::Exists { var, body } => nnf(Formula::Forall {
                var,
                body: Box::new(Formula::not(*body)),
            }),
            Formula::Atom {
                kind: PredicateKind::True,
                ..
            } => Formula::False,
            Formula::Atom {
                kind: PredicateKind::False,
                ..
            } => Formula::True,
            Formula::Atom { kind, args } if is_rel(&kind) => Formula::Atom {
                kind: dual_kind(kind),
                args,
            },
            inner => Formula::Not(Box::new(inner)),
        },
        Formula::And(a, b) => Formula::and(nnf(*a), nnf(*b)),
        Formula::Or(a, b) => Formula::or(nnf(*a), nnf(*b)),
        Formula::Forall { var, body } => Formula::Forall {
            var,
            body: Box::new(nnf(*body)),
        },
        Formula::Exists { var, body } => Formula::Exists {
            var,
            body: Box::new(nnf(*body)),
        },
        other => other,
    }
}

fn witness_rational(iv: &VarInterval) -> Option<rug::Rational> {
    let eps = || rug::Rational::from((1, 10_000));
    match (&iv.lower, &iv.upper) {
        (None, None) => Some(rug::Rational::from(0)),
        (Some(Bound::Lower { val: lo, strict: s }), None) => {
            let e = eps();
            Some(if *s { lo.clone() + &e } else { lo.clone() })
        }
        (None, Some(Bound::Upper { val: hi, strict: s })) => {
            let e = eps();
            Some(if *s { hi.clone() - &e } else { hi.clone() })
        }
        (
            Some(Bound::Lower {
                val: lo,
                strict: sl,
            }),
            Some(Bound::Upper {
                val: hi,
                strict: su,
            }),
        ) => {
            if lo > hi {
                return None;
            }
            if lo < hi {
                return Some((lo.clone() + hi.clone()) / rug::Rational::from(2));
            }
            // lo == hi
            if *sl || *su {
                None
            } else {
                Some(lo.clone())
            }
        }
        _ => None,
    }
}

fn map_to_witness(
    m: &HashMap<ExprId, VarInterval>,
    pool: &ExprPool,
) -> Result<HashMap<String, String>, SatFail> {
    let mut out = HashMap::new();
    for (&id, iv) in m {
        let name = symbol_key(pool, id).ok_or(SatFail::Unknown)?;
        let w = witness_rational(iv).ok_or(SatFail::Unknown)?;
        out.insert(name, w.to_string());
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Satisfiability {
    Sat(HashMap<String, String>),
    Unsat,
    Unknown,
}

enum SatFail {
    Unsat,
    Unknown,
}

fn merge_maps(
    mut a: HashMap<ExprId, VarInterval>,
    b: HashMap<ExprId, VarInterval>,
) -> Result<HashMap<ExprId, VarInterval>, SatFail> {
    for (k, vb) in b {
        match a.remove(&k) {
            None => {
                a.insert(k, vb);
            }
            Some(va) => {
                let m = va.intersect(&vb).ok_or(SatFail::Unsat)?;
                a.insert(k, m);
            }
        }
    }
    Ok(a)
}

fn sat_intervals(f: &Formula, pool: &ExprPool) -> Result<HashMap<ExprId, VarInterval>, SatFail> {
    match f {
        Formula::True => Ok(HashMap::new()),
        Formula::False => Err(SatFail::Unsat),
        Formula::Forall { .. } => Err(SatFail::Unknown),
        Formula::Exists { body, .. } => sat_intervals(body, pool),
        Formula::And(a, b) => {
            let ma = sat_intervals(a, pool)?;
            let mb = sat_intervals(b, pool)?;
            merge_maps(ma, mb)
        }
        Formula::Or(a, b) => match sat_intervals(a, pool) {
            Ok(m) => Ok(m),
            Err(SatFail::Unsat) => sat_intervals(b, pool),
            Err(SatFail::Unknown) => match sat_intervals(b, pool) {
                Ok(m) => Ok(m),
                Err(SatFail::Unsat) => Err(SatFail::Unknown),
                Err(SatFail::Unknown) => Err(SatFail::Unknown),
            },
        },
        Formula::Not(inner) => {
            if let Formula::Atom { kind, args } = inner.as_ref() {
                if is_rel(kind) {
                    let dual = Formula::Atom {
                        kind: dual_kind(kind.clone()),
                        args: args.clone(),
                    };
                    return sat_intervals(&dual, pool);
                }
            }
            Err(SatFail::Unknown)
        }
        Formula::Atom { kind, args } => {
            if matches!(
                kind,
                PredicateKind::And | PredicateKind::Or | PredicateKind::Not
            ) {
                return Err(SatFail::Unknown);
            }
            if matches!(kind, PredicateKind::True) {
                return Ok(HashMap::new());
            }
            if matches!(kind, PredicateKind::False) {
                return Err(SatFail::Unsat);
            }
            let (v, iv) = atom_to_interval(pool, kind.clone(), args).ok_or(SatFail::Unknown)?;
            if iv.is_empty() {
                return Err(SatFail::Unsat);
            }
            let mut m = HashMap::new();
            m.insert(v, iv);
            Ok(m)
        }
    }
}

fn simplify_formula_constants(f: Formula) -> Formula {
    match f {
        Formula::And(a, b) => {
            let la = simplify_formula_constants(*a);
            let lb = simplify_formula_constants(*b);
            match (&la, &lb) {
                (Formula::False, _) | (_, Formula::False) => Formula::False,
                (Formula::True, x) => x.clone(),
                (x, Formula::True) => x.clone(),
                _ => Formula::and(la, lb),
            }
        }
        Formula::Or(a, b) => {
            let la = simplify_formula_constants(*a);
            let lb = simplify_formula_constants(*b);
            match (&la, &lb) {
                (Formula::True, _) | (_, Formula::True) => Formula::True,
                (Formula::False, x) => x.clone(),
                (x, Formula::False) => x.clone(),
                _ => Formula::or(la, lb),
            }
        }
        Formula::Not(x) => Formula::not(simplify_formula_constants(*x)),
        Formula::Forall { var, body } => Formula::Forall {
            var,
            body: Box::new(simplify_formula_constants(*body)),
        },
        Formula::Exists { var, body } => Formula::Exists {
            var,
            body: Box::new(simplify_formula_constants(*body)),
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Purely propositional fragment
// ---------------------------------------------------------------------------

/// Most distinct Boolean symbols the propositional path will accept.
const MAX_PROP_VARS: usize = 64;
/// Most formula nodes the propositional path will visit while lifting.
const MAX_PROP_NODES: usize = 4096;
/// Decision budget handed to [`dpll_sat_bounded`] by [`satisfiable`].
const PROP_DECISION_BUDGET: u64 = 2_000_000;

/// The string a [`Satisfiability::Sat`] witness uses for a true Boolean.
pub(crate) const BOOL_TRUE: &str = "true";
/// The string a [`Satisfiability::Sat`] witness uses for a false Boolean.
pub(crate) const BOOL_FALSE: &str = "false";

/// A formula in the purely propositional fragment.
///
/// `Var(i)` is a **1-based** proposition index, so it doubles as a [`BoolLit`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum Prop {
    Var(u32),
    Const(bool),
    And(Vec<Prop>),
    Or(Vec<Prop>),
    Not(Box<Prop>),
}

impl Prop {
    /// Evaluate under a total assignment (`model[i - 1]` is the value of `Var(i)`).
    fn eval(&self, model: &[bool]) -> bool {
        match self {
            Prop::Const(b) => *b,
            Prop::Var(i) => model[*i as usize - 1],
            Prop::Not(x) => !x.eval(model),
            Prop::And(xs) => xs.iter().all(|x| x.eval(model)),
            Prop::Or(xs) => xs.iter().any(|x| x.eval(model)),
        }
    }
}

/// Variable table built while lifting an [`ExprId`] into a [`Prop`].
struct PropVars {
    /// Symbol `ExprId` → 1-based proposition index.
    index: HashMap<ExprId, u32>,
    /// Proposition index − 1 → symbol name, for the witness map.
    names: Vec<String>,
    /// Formula nodes visited so far (a DAG node is counted once per visit).
    nodes: usize,
}

/// Lift `expr` into the purely propositional fragment.
///
/// Returns `None` when `expr` is **not** purely propositional — it contains an
/// arithmetic relation, a quantifier, a non-symbol term, two same-named symbols
/// that a witness map could not tell apart, or more material than the budget
/// allows.  `None` therefore means "some other decision procedure's problem",
/// never "unsatisfiable".
fn prop_from_expr(expr: ExprId, pool: &ExprPool, vars: &mut PropVars) -> Option<Prop> {
    vars.nodes += 1;
    if vars.nodes > MAX_PROP_NODES {
        return None;
    }
    match pool.get(expr) {
        ExprData::Symbol { name, .. } => {
            if let Some(&i) = vars.index.get(&expr) {
                return Some(Prop::Var(i));
            }
            // Distinct `ExprId`s can share a name (same name, different
            // `Domain`).  The witness map is keyed by name, so it could not
            // report both; refuse rather than conflate them.
            if vars.names.contains(&name) {
                return None;
            }
            if vars.names.len() >= MAX_PROP_VARS {
                return None;
            }
            let i = vars.names.len() as u32 + 1;
            vars.names.push(name);
            vars.index.insert(expr, i);
            Some(Prop::Var(i))
        }
        ExprData::Predicate { kind, args } => match kind {
            PredicateKind::True => Some(Prop::Const(true)),
            PredicateKind::False => Some(Prop::Const(false)),
            PredicateKind::And => args
                .into_iter()
                .map(|a| prop_from_expr(a, pool, vars))
                .collect::<Option<Vec<_>>>()
                .map(Prop::And),
            PredicateKind::Or => args
                .into_iter()
                .map(|a| prop_from_expr(a, pool, vars))
                .collect::<Option<Vec<_>>>()
                .map(Prop::Or),
            PredicateKind::Not => {
                if args.len() != 1 {
                    return None;
                }
                prop_from_expr(args[0], pool, vars).map(|p| Prop::Not(Box::new(p)))
            }
            // A relation is an arithmetic atom, not a proposition.
            _ => None,
        },
        _ => None,
    }
}

/// Fold `True`/`False` away.  Afterwards a [`Prop::Const`] can only be the root.
fn prop_fold(p: Prop) -> Prop {
    match p {
        Prop::And(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                match prop_fold(x) {
                    Prop::Const(true) => {}
                    Prop::Const(false) => return Prop::Const(false),
                    other => out.push(other),
                }
            }
            match out.len() {
                0 => Prop::Const(true),
                1 => out.pop().expect("len == 1"),
                _ => Prop::And(out),
            }
        }
        Prop::Or(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                match prop_fold(x) {
                    Prop::Const(false) => {}
                    Prop::Const(true) => return Prop::Const(true),
                    other => out.push(other),
                }
            }
            match out.len() {
                0 => Prop::Const(false),
                1 => out.pop().expect("len == 1"),
                _ => Prop::Or(out),
            }
        }
        Prop::Not(x) => match prop_fold(*x) {
            Prop::Const(b) => Prop::Const(!b),
            other => Prop::Not(Box::new(other)),
        },
        other => other,
    }
}

/// Tseitin encoder: a [`Prop`] becomes an equisatisfiable CNF over the original
/// propositions plus one fresh definition variable per gate.
struct Tseitin {
    clauses: Vec<BoolClause>,
    /// Highest variable index allocated so far.
    next: u32,
}

impl Tseitin {
    fn fresh(&mut self) -> Option<BoolLit> {
        self.next = self.next.checked_add(1)?;
        // Definition variables sit above the originals; keep them inside `i32`.
        if self.next > (MAX_PROP_VARS + MAX_PROP_NODES) as u32 {
            return None;
        }
        Some(self.next as BoolLit)
    }

    /// Emit the defining clauses for `p` and return the literal it denotes.
    ///
    /// Both directions of each definition are emitted (`g ↔ gate`), so the CNF
    /// is satisfiable exactly when `p` is, and any CNF model restricted to the
    /// original propositions is a model of `p`.
    fn encode(&mut self, p: &Prop) -> Option<BoolLit> {
        match p {
            // Folded away by `prop_fold` before encoding.
            Prop::Const(_) => None,
            Prop::Var(i) => Some(*i as BoolLit),
            Prop::Not(x) => Some(-self.encode(x)?),
            Prop::And(xs) => {
                let lits = xs
                    .iter()
                    .map(|x| self.encode(x))
                    .collect::<Option<Vec<_>>>()?;
                let g = self.fresh()?;
                // g → xᵢ for every i
                let mut long = Vec::with_capacity(lits.len() + 1);
                long.push(g);
                for l in lits {
                    self.clauses.push(vec![-g, l]);
                    long.push(-l);
                }
                // (⋀ xᵢ) → g
                self.clauses.push(long);
                Some(g)
            }
            Prop::Or(xs) => {
                let lits = xs
                    .iter()
                    .map(|x| self.encode(x))
                    .collect::<Option<Vec<_>>>()?;
                let g = self.fresh()?;
                // g → ⋁ xᵢ
                let mut long = Vec::with_capacity(lits.len() + 1);
                long.push(-g);
                for l in lits {
                    self.clauses.push(vec![g, -l]);
                    long.push(l);
                }
                // xᵢ → g for every i
                self.clauses.push(long);
                Some(g)
            }
        }
    }
}

/// Brute-force reference decision, used only to guard [`Prop`] verdicts in
/// debug builds on instances small enough for it to be free.
#[cfg(debug_assertions)]
fn prop_brute_force_sat(p: &Prop, n_vars: usize) -> bool {
    let mut model = vec![false; n_vars];
    for bits in 0u32..(1u32 << n_vars) {
        for (i, m) in model.iter_mut().enumerate() {
            *m = bits & (1 << i) != 0;
        }
        if p.eval(&model) {
            return true;
        }
    }
    false
}

/// Decide `expr` if — and only if — it is purely propositional.
///
/// `None` means the formula is outside this fragment and the caller should try
/// another procedure.  `Some(Satisfiability::Unknown)` means it *is* in the
/// fragment but exceeded the search budget.
fn satisfiable_propositional(expr: ExprId, pool: &ExprPool) -> Option<Satisfiability> {
    let mut vars = PropVars {
        index: HashMap::new(),
        names: Vec::new(),
        nodes: 0,
    };
    let raw = prop_from_expr(expr, pool, &mut vars)?;
    let folded = prop_fold(raw);
    let n_vars = vars.names.len();

    // A formula that folds to a constant needs no search.
    if let Prop::Const(b) = folded {
        return Some(if b {
            Satisfiability::Sat(HashMap::new())
        } else {
            Satisfiability::Unsat
        });
    }

    let mut enc = Tseitin {
        clauses: Vec::new(),
        next: n_vars as u32,
    };
    let root = enc.encode(&folded)?;
    enc.clauses.push(vec![root]);
    let n_total = enc.next;

    let verdict = dpll_sat_bounded(enc.clauses, n_total, PROP_DECISION_BUDGET);

    #[cfg(debug_assertions)]
    if n_vars <= 10 && !matches!(verdict, DpllOutcome::Exhausted) {
        assert_eq!(
            matches!(verdict, DpllOutcome::Sat(_)),
            prop_brute_force_sat(&folded, n_vars),
            "DPLL disagreed with exhaustive enumeration on {folded:?}"
        );
    }

    match verdict {
        DpllOutcome::Unsat => Some(Satisfiability::Unsat),
        DpllOutcome::Exhausted => Some(Satisfiability::Unknown),
        DpllOutcome::Sat(assign) => {
            // Never hand back an unchecked model: evaluate the *original*
            // formula under it.  A Tseitin or search defect then degrades to an
            // honest `Unknown` instead of a wrong `Sat`.
            let model = &assign[..n_vars];
            if !folded.eval(model) {
                return Some(Satisfiability::Unknown);
            }
            let witness = vars
                .names
                .iter()
                .zip(model)
                .map(|(name, &v)| {
                    (
                        name.clone(),
                        if v { BOOL_TRUE } else { BOOL_FALSE }.to_string(),
                    )
                })
                .collect();
            Some(Satisfiability::Sat(witness))
        }
    }
}

/// Quantifier-free (and single-∃) satisfiability over the supported fragment.
///
/// See the [module docs](self) for exactly which fragments are decided.  The
/// two paths are disjoint and are tried in order:
///
/// 1. purely propositional (Boolean symbols and `∧ ∨ ¬`, hence also `→` and
///    `↔` written out in terms of them) — complete, via DPLL;
/// 2. one real symbol against rational constants — incomplete, via intervals.
///
/// A [`Satisfiability::Sat`] witness maps symbol name → value string, and is
/// **homogeneous**: from the propositional path every value is `"true"` or
/// `"false"`; from the interval path every value is a rational literal.  The
/// two never mix, because a formula containing both a Boolean symbol and an
/// arithmetic relation is [`Satisfiability::Unknown`].
pub fn satisfiable(expr: ExprId, pool: &ExprPool) -> Satisfiability {
    if let Some(verdict) = satisfiable_propositional(expr, pool) {
        return verdict;
    }
    let f = match formula_from_expr(expr, pool) {
        Ok(f) => f,
        Err(_) => return Satisfiability::Unknown,
    };
    let f = nnf(simplify_formula_constants(f));
    match sat_intervals(&f, pool).and_then(|m| map_to_witness(&m, pool)) {
        Ok(w) => Satisfiability::Sat(w),
        Err(SatFail::Unsat) => Satisfiability::Unsat,
        Err(SatFail::Unknown) => Satisfiability::Unknown,
    }
}

// ---------------------------------------------------------------------------
// Boolean DPLL (purely structural; exported for benchmarks / future theory plugins)
// ---------------------------------------------------------------------------

/// Literal: signed index into a fixed proposition table.
pub type BoolLit = i32;

/// Clause disjunction; empty clause = false.
pub type BoolClause = Vec<BoolLit>;

/// Verdict of a budgeted DPLL search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DpllOutcome {
    /// A total assignment; every clause is satisfied by it.
    Sat(Vec<bool>),
    /// Proved: no assignment satisfies every clause.
    Unsat,
    /// The decision budget ran out.  **Not a verdict** — the instance may be
    /// satisfiable or unsatisfiable, and a caller must not read this as either.
    Exhausted,
}

/// Why a branch of the search stopped.
enum DpllStop {
    /// This branch is refuted.
    Conflict,
    /// The decision budget ran out somewhere below here.
    Budget,
}

fn dpll_is_conflict(c: &BoolClause, a: &[Option<bool>]) -> bool {
    c.iter().all(|&lit| {
        let v = lit.unsigned_abs() as usize - 1;
        let sign = lit > 0;
        match a[v] {
            Some(t) => t != sign,
            None => false,
        }
    })
}

/// Unit propagation to fixpoint.  `Err(())` means a clause went false.
///
/// On `Err` the trail is left dirty; every caller that can continue searching
/// restores a snapshot first (see [`dpll_dfs`]).
fn dpll_unit_prop(clauses: &[BoolClause], a: &mut [Option<bool>]) -> Result<(), ()> {
    loop {
        let mut progressed = false;
        for cl in clauses {
            let mut unassigned: Vec<(usize, bool)> = vec![];
            let mut satisfied = false;
            for &lit in cl {
                let v = lit.unsigned_abs() as usize - 1;
                let sign = lit > 0;
                match a[v] {
                    None => unassigned.push((v, sign)),
                    Some(t) if t == sign => satisfied = true,
                    _ => {}
                }
            }
            if satisfied {
                continue;
            }
            if unassigned.is_empty() {
                return Err(());
            }
            if unassigned.len() == 1 {
                let (v, s) = unassigned[0];
                if a[v].is_none() {
                    a[v] = Some(s);
                    progressed = true;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    Ok(())
}

fn dpll_dfs(
    clauses: &[BoolClause],
    a: &mut [Option<bool>],
    budget: &mut u64,
) -> Result<(), DpllStop> {
    dpll_unit_prop(clauses, a).map_err(|()| DpllStop::Conflict)?;
    for cl in clauses {
        if dpll_is_conflict(cl, a) {
            return Err(DpllStop::Conflict);
        }
    }
    let Some((i, _)) = a.iter().enumerate().find(|(_, x)| x.is_none()) else {
        return Ok(());
    };
    if *budget == 0 {
        return Err(DpllStop::Budget);
    }
    *budget -= 1;

    // Both `dpll_unit_prop` above and the recursive calls below write into `a`.
    // Undoing only `a[i]` between the two branches — which is what this used to
    // do — leaves the second branch starting from the first branch's implied
    // literals, which can refute a branch that is in fact satisfiable and so
    // report a satisfiable instance as `Unsat`.  Restore the whole trail.
    let snapshot = a.to_vec();
    for value in [false, true] {
        a[i] = Some(value);
        match dpll_dfs(clauses, a, budget) {
            Ok(()) => return Ok(()),
            Err(DpllStop::Budget) => return Err(DpllStop::Budget),
            Err(DpllStop::Conflict) => a.copy_from_slice(&snapshot),
        }
    }
    Err(DpllStop::Conflict)
}

/// DPLL with a cap on the number of decisions, so a caller can bound the search.
///
/// `max_decisions` counts *branching* decisions only; unit propagation is free.
/// Returns [`DpllOutcome::Exhausted`] rather than guessing when the cap is hit.
pub(crate) fn dpll_sat_bounded(
    clauses: Vec<BoolClause>,
    n_vars: u32,
    max_decisions: u64,
) -> DpllOutcome {
    let mut assign = vec![None; n_vars as usize];
    let mut budget = max_decisions;
    match dpll_dfs(&clauses, &mut assign, &mut budget) {
        Ok(()) => {
            let model: Vec<bool> = assign.into_iter().map(|x| x.unwrap_or(false)).collect();
            // A model is cheap to check and expensive to get wrong; a search
            // defect must not be able to leave here as a `Sat`.
            debug_assert!(
                clauses.iter().all(|cl| cl
                    .iter()
                    .any(|&lit| { model[lit.unsigned_abs() as usize - 1] == (lit > 0) })),
                "dpll_sat_bounded returned a model that falsifies a clause"
            );
            DpllOutcome::Sat(model)
        }
        Err(DpllStop::Conflict) => DpllOutcome::Unsat,
        Err(DpllStop::Budget) => DpllOutcome::Exhausted,
    }
}

/// Very small DPLL without clause learning. Returns `Some(assign)` or `None` if UNSAT.
///
/// Sound and complete for the CNF it is handed, and unbudgeted: on a hard
/// instance it runs until it finishes.  The crate-internal `dpll_sat_bounded`
/// is the variant to use where a caller
/// needs to bound the search and to tell "no model" apart from "gave up".
///
/// # Not an SMT engine (P2-3 design decision D6)
///
/// This solves the *propositional* problem it is handed — a CNF clause list over
/// opaque propositions.  [`satisfiable`] routes the **purely propositional**
/// fragment here, where the propositions are the formula's own Boolean symbols
/// and nothing is abstracted away, so the answer transfers back exactly.
///
/// It is still deliberately **not** wired into [`smtlib`] or `alkahest.smt` as a
/// fallback engine, because the only way to reach it from an *arithmetic*
/// [`Formula`] is to abstract each relation to a fresh proposition, and that
/// abstraction is sound in one direction only: it can confirm `unsat`, but it
/// reports `x > 0 ∧ x < 0` as *sat* with a meaningless model.  Handing that back
/// as a witness — under a bridge whose entire premise is that every `sat` model
/// is checked exactly — would be precisely the silent-error shape the bridge
/// exists to prevent.  `alkahest.smt.solve` therefore refuses with `E-SMT-001`
/// when no external solver is installed rather than degrading to anything
/// in-tree.
pub fn dpll_sat(clauses: Vec<BoolClause>, n_vars: u32) -> Option<Vec<bool>> {
    match dpll_sat_bounded(clauses, n_vars, u64::MAX) {
        DpllOutcome::Sat(model) => Some(model),
        DpllOutcome::Unsat => None,
        // A `u64::MAX` decision budget cannot be spent in finite time, so this
        // is unreachable.  Returning `None` here would mean reporting "gave up"
        // as "unsatisfiable", which is the one answer this must never give.
        DpllOutcome::Exhausted => {
            unreachable!("dpll_sat: the u64::MAX decision budget was exhausted")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn and_contradiction_unsat() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0_i32);
        let f = p.pred_and(vec![p.pred_gt(x, z), p.pred_lt(x, z)]);
        assert_eq!(satisfiable(f, &p), Satisfiability::Unsat);
    }

    #[test]
    fn or_cover_sat() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0_i32);
        let f = p.pred_or(vec![p.pred_gt(x, z), p.pred_le(x, z)]);
        match satisfiable(f, &p) {
            Satisfiability::Sat(m) => assert!(m.contains_key("x")),
            other => panic!("expected Sat, got {other:?}"),
        }
    }

    #[test]
    fn forall_unknown() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let f = p.forall(x, p.pred_gt(x, p.integer(0_i32)));
        assert_eq!(satisfiable(f, &p), Satisfiability::Unknown);
    }

    #[test]
    fn formula_quant_round_trip() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let body = p.pred_gt(x, p.integer(0_i32));
        let q = Formula::Exists {
            var: x,
            body: Box::new(formula_from_expr(body, &p).unwrap()),
        };
        let e = q.to_expr(&p);
        let back = formula_from_expr(e, &p).unwrap();
        assert_eq!(back, q);
    }

    #[test]
    fn dpll_tiny_sat() {
        // (p1 ∨ p2) ∧ (¬p1 ∨ p2)  →  p2 true
        let r = dpll_sat(vec![vec![1, 2], vec![-1, 2]], 2);
        assert!(r.is_some());
    }

    // -----------------------------------------------------------------------
    // DPLL backtracking
    // -----------------------------------------------------------------------

    /// Regression: the search used to undo only its own decision literal on
    /// backtracking, leaving the failed branch's unit-propagated literals in
    /// place.  The second branch then started from a polluted trail and could
    /// be refuted even though it was satisfiable — reporting a *satisfiable*
    /// instance as `Unsat`, which is the worst answer this code can give.
    #[test]
    fn dpll_restores_the_trail_on_backtrack() {
        // (p1 ∨ p2) ∧ (p1 ∨ ¬p2) ∧ (¬p1 ∨ ¬p2): satisfied by p1 = ⊤, p2 = ⊥.
        let clauses = vec![vec![1, 2], vec![1, -2], vec![-1, -2]];
        let model = dpll_sat(clauses.clone(), 2).expect("instance is satisfiable");
        for cl in &clauses {
            assert!(
                cl.iter()
                    .any(|&lit| model[lit.unsigned_abs() as usize - 1] == (lit > 0)),
                "clause {cl:?} falsified by {model:?}"
            );
        }
    }

    #[test]
    fn dpll_genuine_unsat_is_still_unsat() {
        // (p1) ∧ (¬p1)
        assert!(dpll_sat(vec![vec![1], vec![-1]], 1).is_none());
        // All four assignments of two variables excluded.
        let all_excluded = vec![vec![1, 2], vec![1, -2], vec![-1, 2], vec![-1, -2]];
        assert!(dpll_sat(all_excluded, 2).is_none());
    }

    #[test]
    fn dpll_bounded_reports_exhausted_rather_than_guessing() {
        // Zero decisions allowed, and the instance needs one.
        let out = dpll_sat_bounded(vec![vec![1, 2], vec![-1, -2]], 2, 0);
        assert_eq!(out, DpllOutcome::Exhausted);
        // The same instance with a budget is decidable.
        assert!(matches!(
            dpll_sat_bounded(vec![vec![1, 2], vec![-1, -2]], 2, 100),
            DpllOutcome::Sat(_)
        ));
    }

    // -----------------------------------------------------------------------
    // Propositional satisfiability through `satisfiable`
    // -----------------------------------------------------------------------

    fn bools(p: &ExprPool, names: &[&str]) -> Vec<ExprId> {
        names.iter().map(|n| p.symbol(*n, Domain::Real)).collect()
    }

    /// `a → b`, written out: the kernel has no `Implies` predicate.
    fn implies(p: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        p.pred_or(vec![p.pred_not(a), b])
    }

    /// `a ↔ b`, written out as `(a → b) ∧ (b → a)`.
    fn iff(p: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        p.pred_and(vec![implies(p, a, b), implies(p, b, a)])
    }

    fn assert_witness_is(m: &HashMap<String, String>, expect: &[(&str, bool)]) {
        for (name, want) in expect {
            let got = m
                .get(*name)
                .unwrap_or_else(|| panic!("witness has no entry for {name}: {m:?}"));
            assert_eq!(
                got,
                if *want { BOOL_TRUE } else { BOOL_FALSE },
                "witness for {name}"
            );
        }
    }

    #[test]
    fn propositional_unsat() {
        // (A ∨ B) ∧ ¬A ∧ ¬B
        let p = ExprPool::new();
        let v = bools(&p, &["A", "B"]);
        let (a, b) = (v[0], v[1]);
        let f = p.pred_and(vec![p.pred_or(vec![a, b]), p.pred_not(a), p.pred_not(b)]);
        assert_eq!(satisfiable(f, &p), Satisfiability::Unsat);
    }

    #[test]
    fn propositional_sat_gives_a_checked_witness() {
        // (A ∨ B) ∧ ¬A  →  A = ⊥, B = ⊤ is the only model.
        let p = ExprPool::new();
        let v = bools(&p, &["A", "B"]);
        let (a, b) = (v[0], v[1]);
        let f = p.pred_and(vec![p.pred_or(vec![a, b]), p.pred_not(a)]);
        match satisfiable(f, &p) {
            Satisfiability::Sat(m) => {
                assert_eq!(m.len(), 2);
                assert_witness_is(&m, &[("A", false), ("B", true)]);
            }
            other => panic!("expected Sat, got {other:?}"),
        }
    }

    #[test]
    fn propositional_contradiction_and_tautology() {
        let p = ExprPool::new();
        let a = p.symbol("A", Domain::Real);
        // A ∧ ¬A
        let contradiction = p.pred_and(vec![a, p.pred_not(a)]);
        assert_eq!(satisfiable(contradiction, &p), Satisfiability::Unsat);
        // A ∨ ¬A — satisfiable, which is all `satisfiable` claims about it.
        let tautology = p.pred_or(vec![a, p.pred_not(a)]);
        assert!(matches!(satisfiable(tautology, &p), Satisfiability::Sat(_)));
        // ¬(A ∨ ¬A) — the negation of a tautology is unsatisfiable.
        assert_eq!(
            satisfiable(p.pred_not(tautology), &p),
            Satisfiability::Unsat
        );
    }

    #[test]
    fn propositional_bare_symbol_is_a_boolean_variable() {
        let p = ExprPool::new();
        let a = p.symbol("A", Domain::Real);
        match satisfiable(a, &p) {
            Satisfiability::Sat(m) => assert_eq!(m.len(), 1),
            other => panic!("expected Sat, got {other:?}"),
        }
        assert_eq!(
            satisfiable(p.pred_not(a), &p),
            Satisfiability::Sat(HashMap::from([("A".to_string(), BOOL_FALSE.to_string())]))
        );
    }

    #[test]
    fn propositional_implication_and_biconditional() {
        let p = ExprPool::new();
        let v = bools(&p, &["A", "B"]);
        let (a, b) = (v[0], v[1]);

        // (A → B) ∧ A ∧ ¬B denies modus ponens: unsatisfiable.
        let mp = p.pred_and(vec![implies(&p, a, b), a, p.pred_not(b)]);
        assert_eq!(satisfiable(mp, &p), Satisfiability::Unsat);

        // (A → B) ∧ A forces B.
        match satisfiable(p.pred_and(vec![implies(&p, a, b), a]), &p) {
            Satisfiability::Sat(m) => assert_witness_is(&m, &[("A", true), ("B", true)]),
            other => panic!("expected Sat, got {other:?}"),
        }

        // (A ↔ B) ∧ A ∧ ¬B is unsatisfiable; (A ↔ B) ∧ ¬A forces ¬B.
        let biconditional = iff(&p, a, b);
        assert_eq!(
            satisfiable(p.pred_and(vec![biconditional, a, p.pred_not(b)]), &p),
            Satisfiability::Unsat
        );
        match satisfiable(p.pred_and(vec![biconditional, p.pred_not(a)]), &p) {
            Satisfiability::Sat(m) => assert_witness_is(&m, &[("A", false), ("B", false)]),
            other => panic!("expected Sat, got {other:?}"),
        }
    }

    #[test]
    fn propositional_constants_fold() {
        let p = ExprPool::new();
        let a = p.symbol("A", Domain::Real);
        assert_eq!(
            satisfiable(p.pred_true(), &p),
            Satisfiability::Sat(HashMap::new())
        );
        assert_eq!(satisfiable(p.pred_false(), &p), Satisfiability::Unsat);
        assert_eq!(
            satisfiable(p.pred_and(vec![a, p.pred_false()]), &p),
            Satisfiability::Unsat
        );
        assert_eq!(
            satisfiable(p.pred_or(vec![a, p.pred_true()]), &p),
            Satisfiability::Sat(HashMap::new())
        );
    }

    #[test]
    fn propositional_pigeonhole_three_into_two_is_unsat() {
        // Three pigeons, two holes: p{i}{j} = pigeon i sits in hole j.
        let p = ExprPool::new();
        let cell: Vec<Vec<ExprId>> = (0..3)
            .map(|i| {
                (0..2)
                    .map(|j| p.symbol(format!("p{i}{j}"), Domain::Real))
                    .collect()
            })
            .collect();
        let mut clauses = Vec::new();
        // Every pigeon sits somewhere.
        for row in &cell {
            clauses.push(p.pred_or(row.clone()));
        }
        // No hole holds two pigeons.
        for (i, row_i) in cell.iter().enumerate() {
            for row_k in &cell[i + 1..] {
                for (a, b) in row_i.iter().zip(row_k) {
                    clauses.push(p.pred_or(vec![p.pred_not(*a), p.pred_not(*b)]));
                }
            }
        }
        assert_eq!(satisfiable(p.pred_and(clauses), &p), Satisfiability::Unsat);
    }

    // -----------------------------------------------------------------------
    // The boundary: what stays `Unknown`
    // -----------------------------------------------------------------------

    #[test]
    fn mixed_boolean_and_arithmetic_stays_unknown() {
        // A ∧ (x > 0): a Boolean symbol and a real relation in one formula.
        // Deciding this needs a Boolean-plus-theory combination this module
        // does not have; `Unknown` is honest and must not become `Unsat`.
        let p = ExprPool::new();
        let a = p.symbol("A", Domain::Real);
        let x = p.symbol("x", Domain::Real);
        let f = p.pred_and(vec![a, p.pred_gt(x, p.integer(0_i32))]);
        assert_eq!(satisfiable(f, &p), Satisfiability::Unknown);

        // …and the mixed *contradiction* stays `Unknown` too.
        let g = p.pred_and(vec![
            a,
            p.pred_gt(x, p.integer(0_i32)),
            p.pred_lt(x, p.integer(0_i32)),
        ]);
        assert_eq!(satisfiable(g, &p), Satisfiability::Unknown);
    }

    #[test]
    fn non_predicate_term_stays_unknown() {
        // `x + 1` is a term, not a formula.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let t = p.add(vec![x, p.integer(1_i32)]);
        assert_eq!(satisfiable(t, &p), Satisfiability::Unknown);
    }

    #[test]
    fn quantified_propositional_stays_unknown() {
        // ∀A. A — quantification over a Boolean is outside the fragment.
        let p = ExprPool::new();
        let a = p.symbol("A", Domain::Real);
        assert_eq!(satisfiable(p.forall(a, a), &p), Satisfiability::Unknown);
    }

    #[test]
    fn same_name_different_domain_refuses_rather_than_conflating() {
        // Two distinct symbols share the name "A"; a name-keyed witness could
        // not tell them apart, so the propositional path declines.
        let p = ExprPool::new();
        let a_real = p.symbol("A", Domain::Real);
        let a_int = p.symbol("A", Domain::Integer);
        assert_ne!(a_real, a_int);
        let f = p.pred_and(vec![a_real, p.pred_not(a_int)]);
        assert_eq!(satisfiable(f, &p), Satisfiability::Unknown);
    }

    #[test]
    fn arithmetic_path_is_untouched() {
        // The single-variable interval fragment behaves as before, and its
        // witnesses are still rationals rather than "true"/"false".
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let z = p.integer(0_i32);
        assert_eq!(
            satisfiable(p.pred_and(vec![p.pred_gt(x, z), p.pred_lt(x, z)]), &p),
            Satisfiability::Unsat
        );
        match satisfiable(p.pred_gt(x, z), &p) {
            Satisfiability::Sat(m) => {
                let v = m.get("x").expect("witness for x");
                assert_ne!(v, BOOL_TRUE);
                assert_ne!(v, BOOL_FALSE);
                assert!(
                    v.parse::<rug::Rational>().is_ok(),
                    "witness is not a rational: {v}"
                );
            }
            other => panic!("expected Sat, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Cross-check against exhaustive enumeration
    // -----------------------------------------------------------------------

    /// xorshift64, so the corpus is reproducible without a dependency.
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Build a random propositional `ExprId` over `vars`.
    fn random_prop(p: &ExprPool, vars: &[ExprId], depth: u32, rng: &mut u64) -> ExprId {
        if depth == 0 {
            return vars[(xorshift(rng) as usize) % vars.len()];
        }
        match xorshift(rng) % 5 {
            0 => vars[(xorshift(rng) as usize) % vars.len()],
            1 => p.pred_not(random_prop(p, vars, depth - 1, rng)),
            2 => {
                let k = 2 + (xorshift(rng) % 2) as usize;
                p.pred_and(
                    (0..k)
                        .map(|_| random_prop(p, vars, depth - 1, rng))
                        .collect(),
                )
            }
            3 => {
                let k = 2 + (xorshift(rng) % 2) as usize;
                p.pred_or(
                    (0..k)
                        .map(|_| random_prop(p, vars, depth - 1, rng))
                        .collect(),
                )
            }
            // An implication, written out.
            _ => {
                let a = random_prop(p, vars, depth - 1, rng);
                let b = random_prop(p, vars, depth - 1, rng);
                p.pred_or(vec![p.pred_not(a), b])
            }
        }
    }

    /// Evaluate a propositional `ExprId` directly, independently of `Prop`.
    fn eval_expr_prop(p: &ExprPool, e: ExprId, env: &HashMap<ExprId, bool>) -> bool {
        match p.get(e) {
            ExprData::Symbol { .. } => env[&e],
            ExprData::Predicate { kind, args } => match kind {
                PredicateKind::True => true,
                PredicateKind::False => false,
                PredicateKind::Not => !eval_expr_prop(p, args[0], env),
                PredicateKind::And => args.iter().all(|&a| eval_expr_prop(p, a, env)),
                PredicateKind::Or => args.iter().any(|&a| eval_expr_prop(p, a, env)),
                other => panic!("not propositional: {other:?}"),
            },
            other => panic!("not propositional: {other:?}"),
        }
    }

    /// `satisfiable` must agree with brute-force enumeration on every random
    /// propositional formula, in **both** directions.  This is the gate that
    /// catches the DPLL backtracking bug.
    #[test]
    fn propositional_agrees_with_exhaustive_enumeration() {
        let mut rng = 0x2545_F491_4F6C_DD1D_u64;
        for case in 0..400_u32 {
            let p = ExprPool::new();
            let n = 2 + (case % 4) as usize; // 2..=5 variables
            let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
            let vars: Vec<ExprId> = names.iter().map(|s| p.symbol(s, Domain::Real)).collect();
            let f = random_prop(&p, &vars, 3, &mut rng);

            // Ground truth by enumeration.
            let mut truth = false;
            for bits in 0_u32..(1_u32 << n) {
                let env: HashMap<ExprId, bool> = vars
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (v, bits & (1 << i) != 0))
                    .collect();
                if eval_expr_prop(&p, f, &env) {
                    truth = true;
                    break;
                }
            }

            match satisfiable(f, &p) {
                Satisfiability::Unsat => {
                    assert!(!truth, "case {case}: reported Unsat but a model exists");
                }
                Satisfiability::Sat(m) => {
                    assert!(
                        truth,
                        "case {case}: reported Sat but the formula is unsatisfiable"
                    );
                    let env: HashMap<ExprId, bool> = vars
                        .iter()
                        .zip(&names)
                        .map(|(&v, name)| (v, m.get(name).map(|s| s == BOOL_TRUE) == Some(true)))
                        .collect();
                    assert!(
                        eval_expr_prop(&p, f, &env),
                        "case {case}: witness {m:?} does not satisfy the formula"
                    );
                }
                Satisfiability::Unknown => {
                    panic!("case {case}: purely propositional formula answered Unknown")
                }
            }
        }
    }
}
