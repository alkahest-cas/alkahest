//! First-order formulas over symbolic reals (V3-3 / FOFormula).
//!
//! - [`Formula`] is the algebraic view over kernel predicates and quantifiers.
//! - [`satisfiable`] decides a quantifier-free fragment: conjunctions of
//!   comparisons between **one** real symbol and a rational constant, plus `Or`
//!   / `Not` (via NNF on relations).  Other shapes return [`Satisfiability::Unknown`].

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

/// Quantifier-free (and single-∃) satisfiability over the supported fragment.
pub fn satisfiable(expr: ExprId, pool: &ExprPool) -> Satisfiability {
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
// Boolean DPLL skeleton (purely structural; exported for benchmarks / future theory plugins)
// ---------------------------------------------------------------------------

/// Literal: signed index into a fixed proposition table.
pub type BoolLit = i32;

/// Clause disjunction; empty clause = false.
pub type BoolClause = Vec<BoolLit>;

/// Very small DPLL without clause learning. Returns `Some(assign)` or `None` if UNSAT.
pub fn dpll_sat(clauses: Vec<BoolClause>, n_vars: u32) -> Option<Vec<bool>> {
    fn is_conflict(c: &BoolClause, a: &[Option<bool>]) -> bool {
        c.iter().all(|&lit| {
            let v = lit.unsigned_abs() as usize - 1;
            let sign = lit > 0;
            match a[v] {
                Some(t) => t != sign,
                None => false,
            }
        })
    }

    fn unit_prop(clauses: &[BoolClause], a: &mut [Option<bool>]) -> Result<(), ()> {
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

    fn dfs(clauses: &[BoolClause], a: &mut [Option<bool>]) -> Result<(), ()> {
        unit_prop(clauses, a)?;
        for cl in clauses {
            if is_conflict(cl, a) {
                return Err(());
            }
        }
        if let Some((i, _)) = a.iter().enumerate().find(|(_, x)| x.is_none()) {
            a[i] = Some(false);
            if dfs(clauses, a).is_ok() {
                return Ok(());
            }
            a[i] = Some(true);
            if dfs(clauses, a).is_ok() {
                return Ok(());
            }
            a[i] = None;
            Err(())
        } else {
            Ok(())
        }
    }

    let n = n_vars as usize;
    let mut assign = vec![None; n];
    if dfs(&clauses, &mut assign).is_ok() {
        Some(
            assign
                .into_iter()
                .map(|x| x.unwrap_or(false))
                .collect::<Vec<_>>(),
        )
    } else {
        None
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
}
