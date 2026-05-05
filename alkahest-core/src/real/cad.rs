//! Cylindrical Algebraic Decomposition scaffolding and univariate quantifier
//! elimination (V2-9).
//!
//! This module provides:
//!
//! - [`cad_project`] — Brown-style projection eliminating one variable via
//!   discriminants (\(`\mathrm{res}(f, \partial_x f)`\)) and pairwise resultants.
//! - [`cad_lift`] — produce isolating intervals for a squarefree algebraic
//!   core built from polynomials in `main_var` (CAD lift stage along one axis).
//! - [`decide`] — decides closed prenex formulas with **one quantifier block**
//!   (`\exists`, `\forall`), where the quantifier-free body is purely polynomial in
//!   the sole quantified symbol with rational/integer literals (parameters are only
//!   integers).
//!
//! Multivariate QE and nested alternating quantifiers are left for future passes.

use crate::diff::{diff, DiffError};
use crate::errors::AlkahestError;
use crate::kernel::expr::PredicateKind;
use crate::kernel::Domain;
use crate::kernel::{ExprId, ExprPool};
use crate::logic::{formula_from_expr, Formula, LogicError};
use crate::poly::resultant::{self, resultant};
use crate::poly::{
    poly_normal, real_roots, ConversionError, RealRootError, ResultantError, RootInterval, UniPoly,
};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

// ---------------------------------------------------------------------------
// Errors and result wrapper
// ---------------------------------------------------------------------------

/// Errors from CAD helpers and [`decide`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CadError {
    NotPolynomial(ConversionError),
    Diff(DiffError),
    Resultant(ResultantError),
    RealRoots(RealRootError),
    Logic(LogicError),
    /// Feature gap (nested quantifiers, parametric polynomials, transcendental atoms, …).
    Unsupported(&'static str),
}

impl fmt::Display for CadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CadError::NotPolynomial(e) => write!(f, "{e}"),
            CadError::Diff(e) => write!(f, "{e}"),
            CadError::Resultant(e) => write!(f, "{e}"),
            CadError::RealRoots(e) => write!(f, "{e}"),
            CadError::Logic(e) => write!(f, "{e}"),
            CadError::Unsupported(s) => write!(f, "CAD: {s}"),
        }
    }
}

impl std::error::Error for CadError {}

impl AlkahestError for CadError {
    fn code(&self) -> &'static str {
        match self {
            CadError::NotPolynomial(e) => e.code(),
            CadError::Diff(e) => e.code(),
            CadError::Resultant(e) => e.code(),
            CadError::RealRoots(e) => e.code(),
            CadError::Logic(e) => e.code(),
            CadError::Unsupported(_) => "E-CAD-001",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            CadError::NotPolynomial(e) => e.remediation(),
            CadError::Diff(e) => e.remediation(),
            CadError::Resultant(e) => e.remediation(),
            CadError::RealRoots(e) => e.remediation(),
            CadError::Logic(e) => e.remediation(),
            CadError::Unsupported(_) => Some(
                "use a purely polynomial constraint in one real variable without nested quantifiers; multivariate QE is incremental",
            ),
        }
    }
}

impl From<ConversionError> for CadError {
    fn from(value: ConversionError) -> Self {
        CadError::NotPolynomial(value)
    }
}

impl From<DiffError> for CadError {
    fn from(value: DiffError) -> Self {
        CadError::Diff(value)
    }
}

impl From<ResultantError> for CadError {
    fn from(value: ResultantError) -> Self {
        CadError::Resultant(value)
    }
}

impl From<RealRootError> for CadError {
    fn from(value: RealRootError) -> Self {
        CadError::RealRoots(value)
    }
}

impl From<LogicError> for CadError {
    fn from(value: LogicError) -> Self {
        CadError::Logic(value)
    }
}

/// Outcome of real QE [`decide`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QeResult {
    pub truth: bool,
    pub witness: Option<HashMap<ExprId, rug::Rational>>,
}

// ---------------------------------------------------------------------------
// NNF helpers (localized copy of [`crate::logic`] helpers)
// ---------------------------------------------------------------------------

fn dual_kind(kind: PredicateKind) -> PredicateKind {
    use PredicateKind::{Eq, Ge, Gt, Le, Lt, Ne};
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

fn is_rel(kind: &PredicateKind) -> bool {
    use PredicateKind::*;
    matches!(kind, Lt | Le | Gt | Ge | Eq | Ne)
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

fn nnf_formula(f: Formula) -> Formula {
    match f {
        Formula::Not(inner) => match *inner {
            Formula::True => Formula::False,
            Formula::False => Formula::True,
            Formula::Not(g) => nnf_formula(*g),
            Formula::And(a, b) => nnf_formula(Formula::or(Formula::not(*a), Formula::not(*b))),
            Formula::Or(a, b) => nnf_formula(Formula::and(Formula::not(*a), Formula::not(*b))),
            Formula::Forall { var, body } => nnf_formula(Formula::Exists {
                var,
                body: Box::new(Formula::not(*body)),
            }),
            Formula::Exists { var, body } => nnf_formula(Formula::Forall {
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
        Formula::And(a, b) => Formula::and(nnf_formula(*a), nnf_formula(*b)),
        Formula::Or(a, b) => Formula::or(nnf_formula(*a), nnf_formula(*b)),
        Formula::Forall { var, body } => Formula::Forall {
            var,
            body: Box::new(nnf_formula(*body)),
        },
        Formula::Exists { var, body } => Formula::Exists {
            var,
            body: Box::new(nnf_formula(*body)),
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Variable sets
// ---------------------------------------------------------------------------

fn insert_formula_vars(pool: &ExprPool, expr: ExprId, out: &mut BTreeSet<ExprId>) {
    for v in resultant::collect_free_vars(expr, pool) {
        out.insert(v);
    }
}

fn free_vars_formula(f: &Formula, pool: &ExprPool) -> BTreeSet<ExprId> {
    match f {
        Formula::Atom { args, .. } => {
            let mut s = BTreeSet::new();
            for &a in args {
                insert_formula_vars(pool, a, &mut s);
            }
            s
        }
        Formula::And(a, b) | Formula::Or(a, b) => {
            let mut s = free_vars_formula(a, pool);
            s.extend(free_vars_formula(b, pool));
            s
        }
        Formula::Not(x) => free_vars_formula(x, pool),
        Formula::Exists { var, body } => {
            let mut s = free_vars_formula(body, pool);
            s.insert(*var);
            s
        }
        Formula::Forall { var, body } => {
            let mut s = free_vars_formula(body, pool);
            s.insert(*var);
            s
        }
        Formula::True | Formula::False => BTreeSet::new(),
    }
}

fn contains_quantifier(f: &Formula) -> bool {
    match f {
        Formula::Exists { .. } | Formula::Forall { .. } => true,
        Formula::And(a, b) | Formula::Or(a, b) => contains_quantifier(a) || contains_quantifier(b),
        Formula::Not(x) => contains_quantifier(x),
        Formula::True | Formula::False | Formula::Atom { .. } => false,
    }
}

fn is_quantifier_free(f: &Formula) -> bool {
    !contains_quantifier(f)
}

fn free_vars_subset_of_binding(pool: &ExprPool, f: &Formula, allowed: &BTreeSet<ExprId>) -> bool {
    free_vars_formula(f, pool).is_subset(allowed)
}

fn poly_exprs_from_atom(
    pool: &ExprPool,
    kind: &PredicateKind,
    args: &[ExprId],
    _quant_var: ExprId,
) -> Result<Vec<ExprId>, CadError> {
    use PredicateKind::{False, True};
    if matches!(kind, True | False) {
        return Ok(vec![]);
    }
    if !is_rel(kind) {
        return Err(CadError::Unsupported(
            "only relation atoms are supported in CAD QE",
        ));
    }
    if args.len() != 2 {
        return Err(CadError::Logic(LogicError::UnsupportedExpr(
            "relational predicate arity must be 2",
        )));
    }
    let lhs = args[0];
    let rhs = args[1];
    let lhs_mrhs = poly_diff(pool, lhs, rhs)?;
    Ok(vec![lhs_mrhs])
}

fn poly_exprs_from_formula(
    pool: &ExprPool,
    f: &Formula,
    var: ExprId,
) -> Result<Vec<ExprId>, CadError> {
    match f {
        Formula::True | Formula::False => Ok(vec![]),
        Formula::Atom { kind, args } => poly_exprs_from_atom(pool, kind, args, var),
        Formula::Not(inner) => {
            if let Formula::Atom { kind, args } = inner.as_ref() {
                if is_rel(kind) {
                    poly_exprs_from_atom(pool, kind, args, var)
                } else {
                    Err(CadError::Unsupported(
                        "NOT is only supported on relation atoms",
                    ))
                }
            } else {
                Err(CadError::Unsupported(
                    "`Not` expects a relational atom underneath in this CAD fragment",
                ))
            }
        }
        Formula::And(a, b) | Formula::Or(a, b) => {
            let mut v = poly_exprs_from_formula(pool, a, var)?;
            v.extend(poly_exprs_from_formula(pool, b, var)?);
            Ok(v)
        }
        _ => Err(CadError::Unsupported(
            "expected quantifier-free Boolean combination of polynomials",
        )),
    }
}

fn eq_polynomials_for_sampling(
    pool: &ExprPool,
    f: &Formula,
    var: ExprId,
) -> Result<Vec<UniPoly>, CadError> {
    fn rec(
        pool: &ExprPool,
        f: &Formula,
        var: ExprId,
        out: &mut Vec<UniPoly>,
    ) -> Result<(), CadError> {
        match f {
            Formula::Atom {
                kind: PredicateKind::Eq,
                args,
            } => {
                if args.len() != 2 {
                    return Err(CadError::Logic(LogicError::UnsupportedExpr(
                        "Eq arity must be 2",
                    )));
                }
                let d = UniPoly::from_symbolic(poly_diff(pool, args[0], args[1])?, var, pool)?;
                if !d.is_zero() {
                    out.push(d);
                }
                Ok(())
            }
            Formula::And(a, b) | Formula::Or(a, b) => {
                rec(pool, a, var, out)?;
                rec(pool, b, var, out)
            }
            Formula::Not(x) => {
                if let Formula::Atom {
                    kind: PredicateKind::Eq,
                    args,
                } = x.as_ref()
                {
                    // Sampling roots of Eq is irrelevant inside Ne — skip specialized roots.
                    let _ = args;
                    Ok(())
                } else {
                    Err(CadError::Unsupported(
                        "NOT over non-Eq unsupported for sampling roots",
                    ))
                }
            }
            _ => Ok(()),
        }
    }

    let mut out = Vec::new();
    rec(pool, f, var, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Polynomial utilities
// ---------------------------------------------------------------------------

fn poly_diff(pool: &ExprPool, lhs: ExprId, rhs: ExprId) -> Result<ExprId, CadError> {
    let minus_one = pool.integer(-1_i32);
    let neg_rhs = pool.mul(vec![minus_one, rhs]);
    Ok(pool.add(vec![lhs, neg_rhs]))
}

fn combine_algebraic_master(main_var: ExprId, polys: &[UniPoly]) -> UniPoly {
    let mut nz: Vec<UniPoly> = polys
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| p.squarefree_part())
        .collect();
    if nz.is_empty() {
        UniPoly::constant(main_var, 1)
    } else {
        let mut m = nz.swap_remove(0);
        for q in nz {
            m = UniPoly::lcm_poly(&m, &q);
        }
        m.squarefree_part()
    }
}

fn cauchy_bound(p: &UniPoly) -> rug::Rational {
    let coeffs = p.coefficients();
    if coeffs.is_empty() || p.degree() <= 0 {
        return rug::Rational::from((1_u32, 1_u32));
    }
    let n = coeffs.len() - 1;
    let lead = coeffs[n].clone().abs();
    if lead.is_zero() {
        return rug::Rational::from((1_u32, 1_u32));
    }
    let mut num = rug::Integer::from(0);
    for c in coeffs.iter().take(n) {
        num += c.clone().abs();
    }
    let frac = rug::Rational::from((num, lead)) + rug::Rational::from(1);
    frac + rug::Rational::from(1)
}

fn iv_midpoint(lo: &rug::Rational, hi: &rug::Rational) -> rug::Rational {
    (lo.clone() + hi.clone()) / rug::Rational::from((2_u32, 1_u32))
}

// ---------------------------------------------------------------------------
// Sign evaluation at a rational sample
// ---------------------------------------------------------------------------

fn cmp_atom(
    pool: &ExprPool,
    kind: &PredicateKind,
    args: &[ExprId],
    var: ExprId,
    pt: &rug::Rational,
) -> Result<bool, CadError> {
    use PredicateKind::{Eq, False, Ge, Gt, Le, Lt, Ne, True};
    if matches!(kind, True) {
        return Ok(true);
    }
    if matches!(kind, False) {
        return Ok(false);
    }
    let diff = UniPoly::from_symbolic(poly_diff(pool, args[0], args[1])?, var, pool)?;
    let v = diff.eval_rational(pt);
    let z = rug::Rational::from(0);
    Ok(match kind {
        Eq => v == z,
        Ne => v != z,
        Lt => v < z,
        Le => v <= z,
        Gt => v > z,
        Ge => v >= z,
        _ => {
            return Err(CadError::Unsupported("non-relational predicate in atom"));
        }
    })
}

fn eval_qf_formula(
    pool: &ExprPool,
    var: ExprId,
    f: &Formula,
    pt: &rug::Rational,
) -> Result<bool, CadError> {
    match f {
        Formula::True => Ok(true),
        Formula::False => Ok(false),
        Formula::Atom { kind, args } => cmp_atom(pool, kind, args.as_slice(), var, pt),
        Formula::And(a, b) => {
            Ok(eval_qf_formula(pool, var, a, pt)? && eval_qf_formula(pool, var, b, pt)?)
        }
        Formula::Or(a, b) => {
            Ok(eval_qf_formula(pool, var, a, pt)? || eval_qf_formula(pool, var, b, pt)?)
        }
        Formula::Not(x) => Ok(!eval_qf_formula(pool, var, x, pt)?),
        _ => Err(CadError::Unsupported(
            "quantifiers not allowed inside QF eval",
        )),
    }
}

fn intervals_overlap(a: &RootInterval, b: &RootInterval) -> bool {
    !(a.hi < b.lo || b.hi < a.lo)
}

/// Some root of `g` lies in the closure of `iv` (overlap with an isolating interval of `squarefree(g)`).
fn gcd_interval_shares_root_iv(g: &UniPoly, iv: &RootInterval) -> Result<bool, CadError> {
    if g.is_zero() || g.degree() <= 0 {
        return Ok(false);
    }
    let sg = g.squarefree_part();
    let roots_g = real_roots(&sg)?;
    Ok(roots_g.into_iter().any(|rj| intervals_overlap(iv, &rj)))
}

fn eval_qf_formula_on_iv(
    pool: &ExprPool,
    var: ExprId,
    phi: &Formula,
    iv: &RootInterval,
    focus_sf: &UniPoly,
) -> Result<bool, CadError> {
    match phi {
        Formula::True => Ok(true),
        Formula::False => Ok(false),
        Formula::Atom { kind, args } => {
            use PredicateKind::{Eq, False, Ne, True};
            if matches!(kind, True) {
                return Ok(true);
            }
            if matches!(kind, False) {
                return Ok(false);
            }
            let d_poly = UniPoly::from_symbolic(poly_diff(pool, args[0], args[1])?, var, pool)?;
            if matches!(kind, Eq) {
                let gx = focus_sf.gcd(&d_poly).unwrap_or_else(|| UniPoly::zero(var));
                return gcd_interval_shares_root_iv(&gx, iv);
            }
            if matches!(kind, Ne) {
                let gx = focus_sf.gcd(&d_poly).unwrap_or_else(|| UniPoly::zero(var));
                return Ok(!gcd_interval_shares_root_iv(&gx, iv)?);
            }
            let mid = iv_midpoint(&iv.lo, &iv.hi);
            eval_qf_formula(pool, var, phi, &mid)
        }
        Formula::And(a, b) => Ok(eval_qf_formula_on_iv(pool, var, a, iv, focus_sf)?
            && eval_qf_formula_on_iv(pool, var, b, iv, focus_sf)?),
        Formula::Or(a, b) => Ok(eval_qf_formula_on_iv(pool, var, a, iv, focus_sf)?
            || eval_qf_formula_on_iv(pool, var, b, iv, focus_sf)?),
        Formula::Not(x) => Ok(!eval_qf_formula_on_iv(pool, var, x, iv, focus_sf)?),
        _ => Err(CadError::Unsupported(
            "unexpected quantifier during CAD sample refinement",
        )),
    }
}

// ---------------------------------------------------------------------------
// One-quantifier elimination (univariate)
// ---------------------------------------------------------------------------

fn decide_exists_univariate(
    pool: &ExprPool,
    var: ExprId,
    phi: Formula,
) -> Result<QeResult, CadError> {
    let allowed: BTreeSet<ExprId> = [var].into_iter().collect();
    if !free_vars_subset_of_binding(pool, &phi, &allowed) {
        return Err(CadError::Unsupported(
            "quantifier-free body may only reference the bound variable (constants allowed)",
        ));
    }

    let poly_exprs = poly_exprs_from_formula(pool, &phi, var)?;
    let mut polys_uni = Vec::<UniPoly>::new();
    for e in poly_exprs.iter().copied() {
        match UniPoly::from_symbolic(e, var, pool) {
            Ok(p) => {
                if !p.is_zero() {
                    polys_uni.push(p.clone());
                }
            }
            Err(err) => return Err(CadError::NotPolynomial(err)),
        }
    }

    let mut candidates: BTreeSet<rug::Rational> = BTreeSet::new();
    let master = combine_algebraic_master(var, &polys_uni);
    let br = cauchy_bound(&master);
    let roots_iv = real_roots(&master)?;

    let mut breakpoints: Vec<rug::Rational> = Vec::new();
    breakpoints.push(-br.clone());
    for iv in roots_iv.iter() {
        breakpoints.push(iv.lo.clone());
        breakpoints.push(iv.hi.clone());
    }
    breakpoints.push(br.clone());

    breakpoints.sort();
    breakpoints.dedup_by(|a, b| *a == *b);
    // midpoints between consecutive breakpoints strictly inside intervals
    for w in breakpoints.windows(2) {
        let lo = &w[0];
        let hi = &w[1];
        if lo < hi {
            candidates.insert(iv_midpoint(lo, hi));
        }
    }

    for p in eq_polynomials_for_sampling(pool, &phi, var)? {
        let sf = p.squarefree_part();
        let riv = real_roots(&sf)?;
        for iv in riv {
            candidates.insert(iv_midpoint(&iv.lo, &iv.hi));
        }
    }

    for pt in candidates {
        if eval_qf_formula(pool, var, &phi, &pt)? {
            let mut wm = HashMap::new();
            wm.insert(var, pt.clone());
            return Ok(QeResult {
                truth: true,
                witness: Some(wm),
            });
        }
    }

    // Algebraic equality literals are rarely satisfied exactly at purely rational samples;
    // use isolating intervals of squarefree Eq-polynomial factors with gcd-based Eq checks.
    for p_focus in eq_polynomials_for_sampling(pool, &phi, var)? {
        let sf = p_focus.squarefree_part();
        if sf.is_zero() {
            continue;
        }
        for iv in real_roots(&sf)? {
            if eval_qf_formula_on_iv(pool, var, &phi, &iv, &sf)? {
                let mid = iv_midpoint(&iv.lo, &iv.hi);
                let mut wm = HashMap::new();
                wm.insert(var, mid);
                return Ok(QeResult {
                    truth: true,
                    witness: Some(wm),
                });
            }
        }
    }

    Ok(QeResult {
        truth: false,
        witness: None,
    })
}

fn decide_closed_qf(pool: &ExprPool, phi: Formula) -> Result<QeResult, CadError> {
    if !free_vars_formula(&phi, pool).is_empty() {
        return Err(CadError::Unsupported(
            "closed formula unexpectedly contains free symbols",
        ));
    }
    let zero = rug::Rational::from(0);
    let dummy = pool.symbol("__cad_iv_local", Domain::Real);
    Ok(QeResult {
        truth: eval_qf_formula(pool, dummy, &phi, &zero)?,
        witness: None,
    })
}

fn decide_formula_inner(pool: &ExprPool, phi: Formula) -> Result<QeResult, CadError> {
    let phi = simplify_formula_constants(nnf_formula(phi));
    if is_quantifier_free(&phi) {
        return decide_closed_qf(pool, phi);
    }
    match phi {
        Formula::Exists { var, body } => {
            if contains_quantifier(&body) {
                return Err(CadError::Unsupported(
                    "nested quantifiers are not implemented",
                ));
            }
            decide_exists_univariate(pool, var, *body)
        }
        Formula::Forall { var, body } => {
            if contains_quantifier(&body) {
                return Err(CadError::Unsupported(
                    "nested quantifiers are not implemented",
                ));
            }
            let neg_body = nnf_formula(Formula::Not(body));
            let inner = decide_exists_univariate(pool, var, neg_body)?;
            Ok(QeResult {
                truth: !inner.truth,
                witness: None,
            })
        }
        Formula::True => Ok(QeResult {
            truth: true,
            witness: None,
        }),
        Formula::False => Ok(QeResult {
            truth: false,
            witness: None,
        }),
        _ => Err(CadError::Unsupported(
            "sentence must begin with forall/exists after quantifiers are outermost",
        )),
    }
}

/// Decide a closed first-order polynomial sentence (`forall` / `exists` prefix,
/// optionally empty), built from Boolean combinations of polynomial relations.
pub fn decide(formula: &Formula, pool: &ExprPool) -> Result<QeResult, CadError> {
    decide_formula_inner(pool, formula.clone())
}

/// Decide from a predicate / quantified [`ExprId`], via [`formula_from_expr`].
pub fn decide_expr(expr: ExprId, pool: &ExprPool) -> Result<QeResult, CadError> {
    let fm = formula_from_expr(expr, pool)?;
    decide(&fm, pool)
}

/// Brown-style projection polynomials for elimination of `elim_var`.
///
/// The returned polynomials are canonically rewritten with [`poly_normal`] in the
/// union of remaining variables (+ constants only).
///
/// Projection set:
/// resultant(`f`,`∂ f/ ∂ elim`,`elim`), all distinct pairwise resultants (`f`,`g`).
pub fn cad_project(
    polynomials: &[ExprId],
    elim_var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, CadError> {
    if polynomials.is_empty() {
        return Ok(Vec::new());
    }
    let mut all_vars = BTreeSet::new();
    all_vars.insert(elim_var);
    for &p in polynomials {
        all_vars.extend(resultant::collect_free_vars(p, pool));
    }
    let vars_no_elim: Vec<ExprId> = all_vars
        .iter()
        .copied()
        .filter(|&v| v != elim_var)
        .collect();

    let mut uniq: Vec<ExprId> = Vec::new();
    let mut seen: BTreeSet<ExprId> = BTreeSet::new();

    for i in 0..polynomials.len() {
        let f_expr = polynomials[i];
        let df = diff(f_expr, elim_var, pool)?.value;

        let is_zero_f = UniPoly::from_symbolic(f_expr, elim_var, pool)
            .map(|u| u.is_zero())
            .unwrap_or(false);
        let is_zero_df = UniPoly::from_symbolic(df, elim_var, pool)
            .map(|u| u.is_zero())
            .unwrap_or(true);

        // Discriminant / projection coefficient via resultant with ∂f.
        if !is_zero_f && !is_zero_df {
            let rp = resultant(f_expr, df, elim_var, pool)?.value;
            if seen.insert(rp) {
                uniq.push(rp);
            }
        }

        // Pairwise resultants don't require ∂f to be non-zero (Brown projection).
        for &g_expr in polynomials.iter().skip(i + 1) {
            let is_zero_g = UniPoly::from_symbolic(g_expr, elim_var, pool)
                .map(|u| u.is_zero())
                .unwrap_or(false);
            if is_zero_f || is_zero_g {
                continue;
            }
            let r = resultant(f_expr, g_expr, elim_var, pool)?.value;
            if seen.insert(r) {
                uniq.push(r);
            }
        }
    }

    let mut normed = Vec::<ExprId>::new();
    for e in uniq {
        let simplified = if vars_no_elim.is_empty() {
            e
        } else {
            poly_normal(e, vars_no_elim.clone(), pool)?
        };
        normed.push(simplified);
    }

    normed.sort_unstable();
    normed.dedup();
    Ok(normed)
}

/// CAD lifting along `main_var`: isolate real roots of a squarefree amalgam built
/// from projections of the listed polynomial expressions when viewed in `main_var`.
pub fn cad_lift(
    polynomials: &[ExprId],
    main_var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<RootInterval>, CadError> {
    let mut polys_uni = Vec::new();
    for &e in polynomials {
        match UniPoly::from_symbolic(e, main_var, pool) {
            Ok(u) => {
                if !u.is_zero() {
                    polys_uni.push(u);
                }
            }
            Err(e) => return Err(CadError::NotPolynomial(e)),
        }
    }
    let m = combine_algebraic_master(main_var, &polys_uni);
    Ok(real_roots(&m)?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn forall_x_squared_plus_one_positive() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);
        let x_sq = p.pow(x, p.integer(2_i32));
        let body = p.pred_gt(p.add(vec![x_sq, one]), p.integer(0_i32));

        let f = Formula::Forall {
            var: x,
            body: Box::new(formula_from_expr(body, &p).unwrap()),
        };
        let r = decide(&f, &p).unwrap();
        assert!(r.truth);
        assert!(r.witness.is_none());
    }

    #[test]
    fn exists_roots_x_squared_minus_two() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let two = p.integer(2_i32);
        let xs = p.pow(x, p.integer(2_i32));
        let body = p.pred_eq(xs, two);
        let f = Formula::Exists {
            var: x,
            body: Box::new(formula_from_expr(body, &p).unwrap()),
        };
        let r = decide(&f, &p).unwrap();
        assert!(r.truth);
        assert!(r.witness.is_some());
    }

    #[test]
    fn cad_lift_univariate_quadratic() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let xs = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-2_i32)]);
        let ivs = cad_lift(&[xs], x, &p).unwrap();
        assert_eq!(ivs.len(), 2);
        assert!(ivs.iter().all(|iv| iv.lo <= iv.hi));
    }

    #[test]
    fn cad_project_circle_eliminates_y() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let circle = p.add(vec![
            p.pow(x, p.integer(2_i32)),
            p.pow(y, p.integer(2_i32)),
            p.integer(-1_i32),
        ]);
        let line = p.add(vec![y, pool_neg_x(&p, x)]); // y - x
        let pr = cad_project(&[circle, line], y, &p).unwrap();
        assert!(!pr.is_empty());
    }

    fn pool_neg_x(pool: &ExprPool, x: ExprId) -> ExprId {
        pool.mul(vec![pool.integer(-1_i32), x])
    }

    #[test]
    fn unipoly_eval_rational_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let qp = UniPoly::from_symbolic(p.add(vec![x, p.integer(2_i32)]), x, &p).unwrap();
        let z = qp.eval_rational(&rug::Rational::from(-2));
        assert_eq!(z, 0);
    }
}
