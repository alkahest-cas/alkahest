//! Cylindrical Algebraic Decomposition scaffolding and univariate quantifier
//! elimination (V2-9).
//!
//! This module provides:
//!
//! - [`cad_project`] — Brown-style projection eliminating one variable via
//!   discriminants (\(`\mathrm{res}(f, \partial_x f)`\)) and pairwise resultants.
//! - [`cad_lift`] — produce isolating intervals for a squarefree algebraic
//!   core built from polynomials in `main_var` (CAD lift stage along one axis).
//! - [`decide`] — decides closed prenex formulas over a purely polynomial body
//!   with rational/integer literals, for:
//!   - **one quantifier**, one variable (`\exists x`, `\forall x`) — the
//!     original V2-9 fragment;
//!   - **two quantifiers over two distinct variables**, same flavor
//!     (`\exists x \exists y`, `\forall x \forall y`) — decided by eliminating
//!     `y` via [`cad_project`] and re-deciding the resulting univariate-in-`x`
//!     sentence at each CAD cell with the one-variable engine;
//!   - **mixed alternation** (`\exists x \forall y`, `\forall x \exists y`) —
//!     reuses the same projection (it only depends on the atoms' polynomials,
//!     not on which quantifier binds which variable), so it is decided by the
//!     same cell-sampling scheme.
//!
//!   The two-variable path samples only *rational* points of each CAD cell of
//!   `x` (open-cell midpoints, plus any projection root that happens to be
//!   exactly rational). If some projection root is irrational **and** the body
//!   contains an equality/inequation atom, that cell cannot be tested exactly
//!   with rational arithmetic alone (it would need algebraic-number CAD
//!   lifting); `decide` reports `Unsupported` in that case rather than risk an
//!   unsound `true`/`false`. Pure-inequality bodies (no `==`/`!=`) never hit
//!   this restriction, since a change of sign of `\exists y.\,\phi(x,y)`
//!   across a projection root is always visible in the neighboring open cells.
//!
//! Three or more variables, or quantifier prefixes longer than two, are left
//! for future passes (full CAD / general multivariate QE).

use crate::diff::{diff, DiffError};
use crate::errors::AlkahestError;
use crate::kernel::expr::PredicateKind;
use crate::kernel::subs;
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
                "use a purely polynomial constraint in one or two real variables with at most \
                 a 2-quantifier prefix; deeper nesting and full multivariate QE are incremental",
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
            // `Eq`, and also the *non-strict* inequalities.
            //
            // A `≤`/`≥` atom can be satisfied at nothing but its boundary:
            // `x² ≤ 0` holds only at `x = 0`. The open-cell sampling above
            // never lands on such a point, so omitting `Le`/`Ge` here made
            // `∃x. x² ≤ 0` return false — and, by negation, made
            // `∀x. x² > 0` return *true*, a proof of a false statement.
            //
            // Strict atoms need no boundary sample: their solution sets are
            // open, so if non-empty they contain a whole interval and the
            // open-cell pass is already complete for them.
            Formula::Atom {
                kind: PredicateKind::Eq | PredicateKind::Le | PredicateKind::Ge,
                args,
            } => {
                if args.len() != 2 {
                    return Err(CadError::Logic(LogicError::UnsupportedExpr(
                        "comparison arity must be 2",
                    )));
                }
                let d = UniPoly::from_symbolic_clear_denoms(
                    poly_diff(pool, args[0], args[1])?,
                    var,
                    pool,
                )?;
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
                    kind: PredicateKind::Eq | PredicateKind::Le | PredicateKind::Ge,
                    args,
                } = x.as_ref()
                {
                    // Negating any of these yields a *strict* atom (`≠`, `>`,
                    // `<`), whose solution set is open — the open-cell pass
                    // already covers it, so its boundary roots are irrelevant.
                    let _ = args;
                    Ok(())
                } else {
                    Err(CadError::Unsupported(
                        "NOT over strict comparison unsupported for sampling roots",
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

/// Bisection budget for tightening an isolating interval.
///
/// 60 halvings shrink a bracket to ~1e-18 of its width, far below any gap
/// these degree-bounded polynomials produce between distinct roots.
const ISOLATION_REFINEMENTS: u32 = 60;

/// Tighten an isolating interval around its single root by bisection.
///
/// Root isolation only promises "exactly one root in here", and the bracket it
/// returns can be wide enough to swallow the neighbouring cell: the roots of
/// `2x⁴ + x³ - 4x² + 3` isolate to `(-2,-1)` and the exact point `-1`, and the
/// entire region where that polynomial is negative lies *inside* the first
/// bracket. No midpoint of the raw breakpoints lands in it.
///
/// Sound because `master` is squarefree (see [`combine_algebraic_master`]), so
/// every root is simple and a sign change is guaranteed across it. That is the
/// precondition bisection needs — on a polynomial with an even-multiplicity
/// root both endpoints share a sign and this would walk away from the root.
fn refine_isolating(master: &UniPoly, iv: &RootInterval) -> (rug::Rational, rug::Rational) {
    let mut lo = iv.lo.clone();
    let mut hi = iv.hi.clone();

    // Which endpoint's sign steers the bisection. An endpoint can itself be a
    // root of `master` — of a *different* root than the one this bracket
    // isolates, since neighbouring brackets share endpoints: `2x⁴+x³-4x²+3`
    // isolates one root inside `(-2,-1)` while `-1` is another root outright.
    // Collapsing the bracket onto such an endpoint throws away the root it was
    // supposed to isolate, so steer by whichever endpoint has a usable sign.
    let v_lo = master.eval_rational(&lo);
    let v_hi = master.eval_rational(&hi);
    let (anchor_positive, anchor_is_lo) = if v_lo != 0 {
        (v_lo > 0, true)
    } else if v_hi != 0 {
        (v_hi > 0, false)
    } else {
        // Both endpoints are roots: nothing to bisect against. Leave the
        // bracket alone rather than guess.
        return (lo, hi);
    };

    for _ in 0..ISOLATION_REFINEMENTS {
        let mid = iv_midpoint(&lo, &hi);
        let v = master.eval_rational(&mid);
        if v == 0 {
            return (mid.clone(), mid);
        }
        // Move the endpoint on the anchor's side when the midpoint agrees with
        // it; otherwise move the other one.
        if ((v > 0) == anchor_positive) == anchor_is_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, hi)
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
    let diff = UniPoly::from_symbolic_clear_denoms(poly_diff(pool, args[0], args[1])?, var, pool)?;
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
            let d_poly =
                UniPoly::from_symbolic_clear_denoms(poly_diff(pool, args[0], args[1])?, var, pool)?;
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
        match UniPoly::from_symbolic_clear_denoms(e, var, pool) {
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
        // Tight brackets *in addition to* the raw ones. Adding them creates
        // consecutive pairs that straddle the narrow cells between nearby
        // roots, which the raw endpoints are often too coarse to expose.
        //
        // Additive on purpose: replacing the raw endpoints deletes the
        // midpoints that were covering other cells, which is a net loss. Every
        // candidate is verified exactly before it is accepted, so a larger
        // breakpoint set can only find witnesses it was previously missing.
        let (rlo, rhi) = refine_isolating(&master, iv);
        breakpoints.push(rlo);
        breakpoints.push(rhi);
    }
    breakpoints.push(br.clone());

    breakpoints.sort();
    breakpoints.dedup_by(|a, b| *a == *b);
    // The breakpoints themselves, and the midpoints between them.
    //
    // Sampling only the midpoints misses cells that happen to sit *around* a
    // breakpoint. `1 - 3x² + 2x³ + 3x⁴` isolates its roots to `(-2,-1)` and
    // `(-1,0)` and is negative only between them — a band containing `x = -1`.
    // The midpoints `-1.5` and `-0.5` both fall outside it, so `∃x. p < 0`
    // came back false and `∀x. p > 0` came back *true*, a proof of a false
    // statement.
    //
    // Adding candidates is monotonically safe: every candidate is checked by
    // `eval_qf_formula` at that exact rational before it is accepted, so a
    // larger sample set can only turn a missed witness into a found one — it
    // can never manufacture a wrong answer.
    for b in &breakpoints {
        candidates.insert(b.clone());
    }
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
    let mut untested_algebraic_boundary = false;
    for p_focus in eq_polynomials_for_sampling(pool, &phi, var)? {
        let sf = p_focus.squarefree_part();
        if sf.is_zero() {
            continue;
        }
        for iv in real_roots(&sf)? {
            if iv.lo != iv.hi {
                untested_algebraic_boundary = true;
            }
            if eval_qf_formula_on_iv(pool, var, &phi, &iv, &sf)? {
                // The witness is the *root* in `iv`, which is irrational
                // whenever the bracket has not collapsed — and the bracket
                // midpoint is then not a witness at all. `∃x. 3x − 2 = 0` used
                // to come back with `x = 1/2`, which fails the very equation it
                // is offered as a solution to. Report a witness only when it
                // survives the same check any caller would apply.
                let mid = iv_midpoint(&iv.lo, &iv.hi);
                let witness = if eval_qf_formula(pool, var, &phi, &mid)? {
                    let mut wm = HashMap::new();
                    wm.insert(var, mid);
                    Some(wm)
                } else {
                    None
                };
                return Ok(QeResult {
                    truth: true,
                    witness,
                });
            }
        }
    }

    // Nothing satisfied the formula at any sampled point. That is only a *proof*
    // of unsatisfiability if the sample set met every cell — including the
    // zero-dimensional cells at the roots themselves, which are reachable only
    // when the root is an exact rational. With an irrational root and a
    // non-strict atom, the one point that could have satisfied the formula was
    // never tested, and answering `false` here is how `∀x. (x² − 2)² > 0` came
    // back `true`: a machine-checked-looking proof of a false theorem. Refuse.
    if untested_algebraic_boundary && body_has_boundary_atom(&phi) {
        return Err(CadError::Unsupported(ALGEBRAIC_BOUNDARY_MSG));
    }

    Ok(QeResult {
        truth: false,
        witness: None,
    })
}

const ALGEBRAIC_BOUNDARY_MSG: &str = "the formula has a non-strict atom (=, <=, >=) whose only \
    possible solutions are roots of an irrational algebraic number; deciding it needs \
    algebraic-number CAD lifting (full CAD). Refusing rather than reporting an unsatisfiability \
    that was never checked at that point";

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

/// Quantifier flavor for the (at most two) leading blocks of a prenex sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quant {
    Exists,
    Forall,
}

fn decide_formula_inner(pool: &ExprPool, phi: Formula) -> Result<QeResult, CadError> {
    let phi = simplify_formula_constants(nnf_formula(phi));
    if is_quantifier_free(&phi) {
        return decide_closed_qf(pool, phi);
    }
    match phi {
        Formula::Exists { var, body } => decide_quantified(pool, Quant::Exists, var, *body),
        Formula::Forall { var, body } => decide_quantified(pool, Quant::Forall, var, *body),
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

/// Dispatch on whether `body` is quantifier-free (one-variable fragment, V2-9) or
/// itself begins with exactly one more quantifier (two-variable fragment).
///
/// Anything deeper (three or more nested quantifier blocks) is `Unsupported`.
fn decide_quantified(
    pool: &ExprPool,
    outer_q: Quant,
    outer_var: ExprId,
    body: Formula,
) -> Result<QeResult, CadError> {
    if !contains_quantifier(&body) {
        return decide_one_var(pool, outer_q, outer_var, body);
    }
    match body {
        Formula::Exists {
            var: inner_var,
            body: inner_body,
        } if !contains_quantifier(&inner_body) => decide_two_var(
            pool,
            outer_q,
            outer_var,
            Quant::Exists,
            inner_var,
            *inner_body,
        ),
        Formula::Forall {
            var: inner_var,
            body: inner_body,
        } if !contains_quantifier(&inner_body) => decide_two_var(
            pool,
            outer_q,
            outer_var,
            Quant::Forall,
            inner_var,
            *inner_body,
        ),
        _ => Err(CadError::Unsupported(
            "quantifier prefixes of length > 2 are not implemented",
        )),
    }
}

fn decide_one_var(
    pool: &ExprPool,
    q: Quant,
    var: ExprId,
    body: Formula,
) -> Result<QeResult, CadError> {
    match q {
        Quant::Exists => decide_exists_univariate(pool, var, body),
        Quant::Forall => {
            let neg_body = nnf_formula(Formula::Not(Box::new(body)));
            let inner = decide_exists_univariate(pool, var, neg_body)?;
            Ok(QeResult {
                truth: !inner.truth,
                witness: None,
            })
        }
    }
}

/// Two-variable, one-quantifier-alternation-block dispatcher.
///
/// `∃x∃y` and `∀x∀y` (same-flavor blocks) are decided directly by projecting `y`
/// away via [`cad_project`] and re-deciding a univariate-in-`y` sentence at every
/// rational sample of the resulting CAD cells of `x` (see [`decide_exists_exists`]
/// for the soundness argument). Mixed alternation (`∃x∀y`, `∀x∃y`) reuses the same
/// projection — the projection set only depends on the polynomials in the atoms,
/// not on which quantifier binds which variable — so it is handled by the same
/// machinery via De Morgan rewrites.
fn decide_two_var(
    pool: &ExprPool,
    outer_q: Quant,
    outer_var: ExprId,
    inner_q: Quant,
    inner_var: ExprId,
    body: Formula,
) -> Result<QeResult, CadError> {
    match (outer_q, inner_q) {
        (Quant::Exists, Quant::Exists) => decide_exists_exists(pool, outer_var, inner_var, body),
        (Quant::Exists, Quant::Forall) => decide_exists_forall(pool, outer_var, inner_var, body),
        (Quant::Forall, Quant::Forall) => {
            // ∀x∀y φ  ≡  ¬∃x∃y ¬φ
            let neg = nnf_formula(Formula::Not(Box::new(body)));
            let inner = decide_exists_exists(pool, outer_var, inner_var, neg)?;
            Ok(QeResult {
                truth: !inner.truth,
                witness: None,
            })
        }
        (Quant::Forall, Quant::Exists) => {
            // ∀x∃y φ  ≡  ¬∃x∀y ¬φ
            let neg = nnf_formula(Formula::Not(Box::new(body)));
            let inner = decide_exists_forall(pool, outer_var, inner_var, neg)?;
            Ok(QeResult {
                truth: !inner.truth,
                witness: None,
            })
        }
    }
}

/// Does `f` (in NNF) contain an atom whose truth can turn on a single boundary
/// point — `=`, `≤` or `≥`?
///
/// Strict atoms (`<`, `>`, `≠`) have *open* solution sets: if one is satisfiable
/// at all it is satisfiable on a whole interval, so the open-cell sampling in
/// [`decide_exists_univariate`] is complete for them and a bracket that never
/// lands on a root costs nothing. A non-strict atom can be satisfied at nothing
/// but a root (`x² ≤ 0` holds only at `x = 0`), and then the root itself has to
/// be in the sample set or the search is incomplete.
fn body_has_boundary_atom(f: &Formula) -> bool {
    match f {
        Formula::Atom { kind, .. } => matches!(
            kind,
            PredicateKind::Eq | PredicateKind::Le | PredicateKind::Ge
        ),
        Formula::And(a, b) | Formula::Or(a, b) => {
            body_has_boundary_atom(a) || body_has_boundary_atom(b)
        }
        Formula::Not(x) => body_has_boundary_atom(x),
        _ => false,
    }
}

/// Does `f` (in NNF) contain an atom that is not strict — `=`, `≠`, `≤` or `≥`?
///
/// The two-variable analogue of [`body_has_boundary_atom`], and the guard for
/// [`project_and_sample_x`]'s `ambiguous_irrational_root`. `≤`/`≥` are the
/// reason it exists: the original guard tested `=`/`≠` only, so
/// `∃x∃y. (x² − 2)² + y² ≤ 0` — true at `(±√2, 0)` and nowhere else — was
/// answered `false`, and its dual `∀x∀y. (x² − 2)² + y² > 0` came back `true`.
/// That is the *same* completeness gap that
/// [`decide_exists_univariate`] closes one dimension down: an atom whose
/// solution set can be a single boundary point needs that point in the sample
/// set, and no rational sample ever lands on an irrational projection root.
///
/// `≠` is kept in the set even though its solution set is open (so open-cell
/// sampling is already complete for it): the pre-existing guard refused on it,
/// and loosening a refusal is the one direction in which a change here could
/// introduce an unsound answer.
///
/// Strict atoms (`<`, `>`) have open solution sets, so the open-cell midpoints
/// are complete for them and no refusal is warranted.
fn body_has_nonstrict_atom(f: &Formula) -> bool {
    match f {
        Formula::Atom { kind, .. } => matches!(
            kind,
            PredicateKind::Eq | PredicateKind::Ne | PredicateKind::Le | PredicateKind::Ge
        ),
        Formula::And(a, b) | Formula::Or(a, b) => {
            body_has_nonstrict_atom(a) || body_has_nonstrict_atom(b)
        }
        Formula::Not(x) => body_has_nonstrict_atom(x),
        _ => false,
    }
}

/// Rebuild a symbolic rational-literal `ExprId` for `r`.
fn rational_to_expr(pool: &ExprPool, r: &rug::Rational) -> ExprId {
    if *r.denom() == 1_u32 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

/// Substitute `var -> value` (a rational-literal `ExprId`) throughout `body`,
/// round-tripping through the kernel expression DAG so ordinary polynomial
/// arithmetic (e.g. `0^2`) collapses without a separate simplification pass.
fn subst_body_var(
    pool: &ExprPool,
    body: &Formula,
    var: ExprId,
    value: ExprId,
) -> Result<Formula, CadError> {
    let expr = body.to_expr(pool);
    let mut map = HashMap::new();
    map.insert(var, value);
    let substituted = subs(expr, &map, pool);
    Ok(formula_from_expr(substituted, pool)?)
}

/// CAD cells of `x` obtained by projecting `y` out of the atoms of a bivariate
/// formula: rational sample points (open-cell midpoints, plus any *exactly
/// rational* projection root) together with a flag noting whether some
/// projection-polynomial root is irrational and therefore untested.
struct XCells {
    candidates: Vec<rug::Rational>,
    ambiguous_irrational_root: bool,
}

/// Compute [`XCells`] for `∃y`/`∀y`-eliminated `x` from the atoms of `body(x, y)`.
///
/// This is the shared "base case" of 2-variable CAD: project `y` away with
/// [`cad_project`] (Brown projection: discriminants + pairwise resultants),
/// combine the projected (and any `y`-free) polynomials into one squarefree
/// master polynomial in `x`, and isolate its real roots. Between and at those
/// roots, every `y`-dependent atom of `body` is sign-invariant in `x` — the
/// same guarantee [`decide_exists_univariate`] relies on for the univariate
/// fragment, one dimension up.
fn project_and_sample_x(
    pool: &ExprPool,
    x: ExprId,
    y: ExprId,
    body: &Formula,
) -> Result<XCells, CadError> {
    let allowed: BTreeSet<ExprId> = [x, y].into_iter().collect();
    if !free_vars_subset_of_binding(pool, body, &allowed) {
        return Err(CadError::Unsupported(
            "quantifier-free body may only reference the two bound variables (constants allowed)",
        ));
    }

    let bivariate_polys = poly_exprs_from_formula(pool, body, y)?;
    let mut y_dep: Vec<ExprId> = Vec::new();
    let mut y_free: Vec<ExprId> = Vec::new();
    for e in bivariate_polys {
        if resultant::collect_free_vars(e, pool).contains(&y) {
            y_dep.push(e);
        } else {
            // Doesn't depend on `y` at all — it's already a constraint purely on
            // `x` (or a constant); carry it through unprojected.
            y_free.push(e);
        }
    }

    let projected = cad_project(&y_dep, y, pool)?;

    let mut polys_x_uni: Vec<UniPoly> = Vec::new();
    for e in projected.into_iter().chain(y_free) {
        match UniPoly::from_symbolic_clear_denoms(e, x, pool) {
            Ok(p) => {
                if !p.is_zero() {
                    polys_x_uni.push(p);
                }
            }
            Err(err) => return Err(CadError::NotPolynomial(err)),
        }
    }

    let master = combine_algebraic_master(x, &polys_x_uni);
    let br = cauchy_bound(&master);
    let roots_iv = real_roots(&master)?;

    let mut breakpoints: Vec<rug::Rational> = vec![-br.clone()];
    for iv in &roots_iv {
        breakpoints.push(iv.lo.clone());
        breakpoints.push(iv.hi.clone());
    }
    breakpoints.push(br);
    breakpoints.sort();
    breakpoints.dedup_by(|a, b| *a == *b);

    let mut candidates: BTreeSet<rug::Rational> = BTreeSet::new();
    for w in breakpoints.windows(2) {
        if w[0] < w[1] {
            candidates.insert(iv_midpoint(&w[0], &w[1]));
        }
    }

    let mut ambiguous_irrational_root = false;
    for iv in &roots_iv {
        if iv.lo == iv.hi {
            candidates.insert(iv.lo.clone());
        } else {
            ambiguous_irrational_root = true;
        }
    }

    Ok(XCells {
        candidates: candidates.into_iter().collect(),
        ambiguous_irrational_root,
    })
}

const IRRATIONAL_ROOT_MSG: &str = "a non-strict atom (=, /=, <=, >=) combined with an irrational \
    projection root of the eliminated variable would require algebraic-number CAD lifting \
    (full CAD); refusing to guess rather than risk an unsound answer";

/// Decide `∃x∃y. body(x, y)` for a polynomial `body` (2-variable, same-flavor
/// quantifier block).
///
/// Samples the CAD cells of `x` induced by projecting `y` out of `body`'s atoms
/// (see [`project_and_sample_x`]); at each rational sample `x0` the remaining
/// sentence `∃y. body(x0, y)` is genuinely univariate in `y` and decided exactly
/// by [`decide_exists_univariate`] (including its own algebraic-root handling for
/// `y`). If no witness is found and every projection root sampled was rational,
/// the cell decomposition is complete and `false` is sound. If some projection
/// root is irrational *and* `body` contains a non-strict atom (see
/// [`body_has_nonstrict_atom`]), the cell at that exact root cannot be tested
/// rationally, so we report `Unsupported` rather than risk a false negative.
fn decide_exists_exists(
    pool: &ExprPool,
    x: ExprId,
    y: ExprId,
    body: Formula,
) -> Result<QeResult, CadError> {
    let cells = project_and_sample_x(pool, x, y, &body)?;
    for x0 in &cells.candidates {
        let x_expr = rational_to_expr(pool, x0);
        let subst = subst_body_var(pool, &body, x, x_expr)?;
        let inner = decide_exists_univariate(pool, y, subst)?;
        if inner.truth {
            let mut wm = HashMap::new();
            wm.insert(x, x0.clone());
            if let Some(inner_w) = inner.witness {
                if let Some(yv) = inner_w.get(&y) {
                    wm.insert(y, yv.clone());
                }
            }
            return Ok(QeResult {
                truth: true,
                witness: Some(wm),
            });
        }
    }
    if cells.ambiguous_irrational_root && body_has_nonstrict_atom(&body) {
        return Err(CadError::Unsupported(IRRATIONAL_ROOT_MSG));
    }
    Ok(QeResult {
        truth: false,
        witness: None,
    })
}

/// Decide `∃x∀y. body(x, y)` — same projection/sampling scheme as
/// [`decide_exists_exists`], but the inner univariate sentence is `∀y` (decided
/// via the usual `¬∃y¬` rewrite over [`decide_exists_univariate`]).
fn decide_exists_forall(
    pool: &ExprPool,
    x: ExprId,
    y: ExprId,
    body: Formula,
) -> Result<QeResult, CadError> {
    let cells = project_and_sample_x(pool, x, y, &body)?;
    for x0 in &cells.candidates {
        let x_expr = rational_to_expr(pool, x0);
        let subst = subst_body_var(pool, &body, x, x_expr)?;
        let neg = nnf_formula(Formula::Not(Box::new(subst)));
        let inner = decide_exists_univariate(pool, y, neg)?;
        if !inner.truth {
            let mut wm = HashMap::new();
            wm.insert(x, x0.clone());
            return Ok(QeResult {
                truth: true,
                witness: Some(wm),
            });
        }
    }
    if cells.ambiguous_irrational_root && body_has_nonstrict_atom(&body) {
        return Err(CadError::Unsupported(IRRATIONAL_ROOT_MSG));
    }
    Ok(QeResult {
        truth: false,
        witness: None,
    })
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
        // `√2` is not rational, so there is no rational witness to report. This
        // used to assert `witness.is_some()` and passed on the isolating
        // interval's midpoint — a "solution" of `x² = 2` that is not one. A
        // witness is a certificate; a wrong one is worse than none.
        assert!(r.witness.is_none());
    }

    /// Every witness `decide` reports must satisfy the sentence it witnesses.
    ///
    /// `∃x. 3x − 2 = 0` has the rational solution `2/3`. Before exact
    /// rational-root recovery in `real_roots`, the isolating bracket stayed at
    /// `[0, 1]` and the reported witness was its midpoint `1/2`, which fails the
    /// equation outright.
    #[test]
    fn exists_witness_satisfies_the_equation() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let lhs = p.add(vec![p.mul(vec![p.integer(3_i32), x]), p.integer(-2_i32)]);
        let body = p.pred_eq(lhs, p.integer(0_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(formula_from_expr(body, &p).unwrap()),
        };
        let r = decide(&f, &p).unwrap();
        assert!(r.truth);
        let w = r
            .witness
            .expect("2/3 is rational, so a witness is reportable");
        assert_eq!(w[&x], rug::Rational::from((2, 3)));
    }

    /// `∀x. (3x + 2)² > 0` is **false**: the square vanishes at `x = −2/3`.
    ///
    /// The CAD sample set is built from bracket endpoints and midpoints, all
    /// dyadic, so `−2/3` was never tested and the sentence came back `true` —
    /// a proof of a false theorem. Exact rational-root recovery puts the root
    /// itself in the sample set.
    #[test]
    fn forall_square_positive_is_false_at_a_non_dyadic_root() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inner = p.add(vec![p.mul(vec![p.integer(3_i32), x]), p.integer(2_i32)]);
        let body = p.pred_gt(p.pow(inner, p.integer(2_i32)), p.integer(0_i32));
        let f = Formula::Forall {
            var: x,
            body: Box::new(formula_from_expr(body, &p).unwrap()),
        };
        assert!(!decide(&f, &p).unwrap().truth);
    }

    /// The same sentence with an *irrational* touching root is refused, not
    /// answered. `∀x. (x² − 2)² > 0` is false (at `±√2`), and no rational
    /// sample can show it; the honest answer is `Unsupported`.
    #[test]
    fn forall_square_positive_refuses_at_an_irrational_root() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let inner = p.add(vec![p.pow(x, p.integer(2_i32)), p.integer(-2_i32)]);
        let body = p.pred_gt(p.pow(inner, p.integer(2_i32)), p.integer(0_i32));
        let f = Formula::Forall {
            var: x,
            body: Box::new(formula_from_expr(body, &p).unwrap()),
        };
        assert!(matches!(
            decide(&f, &p),
            Err(CadError::Unsupported(ALGEBRAIC_BOUNDARY_MSG))
        ));
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

    // -----------------------------------------------------------------------
    // Two-variable, same-flavor and mixed-alternation quantifier blocks.
    // -----------------------------------------------------------------------

    fn xy_pool() -> (ExprPool, ExprId, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        (p, x, y)
    }

    #[test]
    fn exists_exists_circle_through_origin_true() {
        // ∃x∃y. x^2 + y^2 == 0  →  true, witness (0, 0)
        let (p, x, y) = xy_pool();
        let sum_sq = p.add(vec![p.pow(x, p.integer(2_i32)), p.pow(y, p.integer(2_i32))]);
        let body = p.pred_eq(sum_sq, p.integer(0_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(Formula::Exists {
                var: y,
                body: Box::new(formula_from_expr(body, &p).unwrap()),
            }),
        };
        let r = decide(&f, &p).unwrap();
        assert!(r.truth);
        let wit = r.witness.expect("witness expected for true ∃∃");
        assert_eq!(wit.get(&x), Some(&rug::Rational::from(0)));
        assert_eq!(wit.get(&y), Some(&rug::Rational::from(0)));
    }

    #[test]
    fn exists_exists_circle_plus_one_false() {
        // ∃x∃y. x^2 + y^2 + 1 == 0  →  false (sum of squares can't be negative)
        let (p, x, y) = xy_pool();
        let sum_sq = p.add(vec![
            p.pow(x, p.integer(2_i32)),
            p.pow(y, p.integer(2_i32)),
            p.integer(1_i32),
        ]);
        let body = p.pred_eq(sum_sq, p.integer(0_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(Formula::Exists {
                var: y,
                body: Box::new(formula_from_expr(body, &p).unwrap()),
            }),
        };
        let r = decide(&f, &p).unwrap();
        assert!(!r.truth);
        assert!(r.witness.is_none());
    }

    #[test]
    fn forall_forall_sum_of_squares_nonneg_true() {
        // ∀x∀y. x^2 + y^2 >= 0  →  true
        let (p, x, y) = xy_pool();
        let sum_sq = p.add(vec![p.pow(x, p.integer(2_i32)), p.pow(y, p.integer(2_i32))]);
        let body = p.pred_ge(sum_sq, p.integer(0_i32));
        let f = Formula::Forall {
            var: x,
            body: Box::new(Formula::Forall {
                var: y,
                body: Box::new(formula_from_expr(body, &p).unwrap()),
            }),
        };
        let r = decide(&f, &p).unwrap();
        assert!(r.truth);
        assert!(r.witness.is_none());
    }

    #[test]
    fn forall_forall_product_positive_false() {
        // ∀x∀y. x*y > 0  →  false (e.g. x = -1, y = 1)
        let (p, x, y) = xy_pool();
        let xy = p.mul(vec![x, y]);
        let body = p.pred_gt(xy, p.integer(0_i32));
        let f = Formula::Forall {
            var: x,
            body: Box::new(Formula::Forall {
                var: y,
                body: Box::new(formula_from_expr(body, &p).unwrap()),
            }),
        };
        let r = decide(&f, &p).unwrap();
        assert!(!r.truth);
    }

    #[test]
    fn forall_exists_every_x_has_bigger_y_true() {
        // ∀x∃y. y > x  →  true (no upper bound on ℝ)
        let (p, x, y) = xy_pool();
        let body = p.pred_gt(y, x);
        let f = Formula::Forall {
            var: x,
            body: Box::new(Formula::Exists {
                var: y,
                body: Box::new(formula_from_expr(body, &p).unwrap()),
            }),
        };
        let r = decide(&f, &p).unwrap();
        assert!(r.truth);
    }

    #[test]
    fn exists_forall_no_x_is_upper_bound_false() {
        // ∃x∀y. x >= y  →  false (no real x bounds every real y from above)
        let (p, x, y) = xy_pool();
        let body = p.pred_ge(x, y);
        let f = Formula::Exists {
            var: x,
            body: Box::new(Formula::Forall {
                var: y,
                body: Box::new(formula_from_expr(body, &p).unwrap()),
            }),
        };
        let r = decide(&f, &p).unwrap();
        assert!(!r.truth);
    }

    #[test]
    fn three_variable_prefix_is_unsupported() {
        // ∃x∃y∃z. ... — quantifier prefixes of length > 2 are out of scope.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);
        let body = p.pred_eq(p.add(vec![x, y, z]), p.integer(0_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(Formula::Exists {
                var: y,
                body: Box::new(Formula::Exists {
                    var: z,
                    body: Box::new(formula_from_expr(body, &p).unwrap()),
                }),
            }),
        };
        let err = decide(&f, &p).unwrap_err();
        assert_eq!(err.code(), "E-CAD-001");
        assert!(matches!(err, CadError::Unsupported(_)));
    }

    #[test]
    fn univariate_regression_still_works_after_two_var_addition() {
        // Guards against accidental regressions in decide_exists_univariate /
        // decide_one_var routing while adding the two-variable path.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let body = p.pred_gt(p.pow(x, p.integer(2_i32)), p.integer(4_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(formula_from_expr(body, &p).unwrap()),
        };
        let r = decide(&f, &p).unwrap();
        assert!(r.truth);
    }
}

#[cfg(test)]
mod sample_point_completeness_tests {
    use super::*;
    use crate::kernel::Domain;

    /// Build `Σ coeffs[i]·xⁱ`.
    fn poly(pool: &ExprPool, x: ExprId, coeffs: &[i64]) -> ExprId {
        let terms: Vec<ExprId> = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let ci = pool.integer(c);
                if i == 0 {
                    ci
                } else {
                    pool.mul(vec![ci, pool.pow(x, pool.integer(i as i64))])
                }
            })
            .collect();
        pool.add(terms)
    }

    fn forall(pool: &ExprPool, x: ExprId, body: Formula) -> Result<QeResult, CadError> {
        decide(
            &Formula::Forall {
                var: x,
                body: Box::new(body),
            },
            pool,
        )
    }

    fn atom(kind: PredicateKind, lhs: ExprId, rhs: ExprId) -> Formula {
        Formula::Atom {
            kind,
            args: vec![lhs, rhs],
        }
    }

    /// The three shapes that used to yield a proof of a false statement.
    ///
    /// - `x² > 0` / `x⁴ > 0`: satisfied-only-on-the-boundary negation, missed
    ///   because `Le`/`Ge` boundaries were not sampled.
    /// - `2x⁴+x³-4x²+3 ≥ 0`: false at `x = -6/5`, in a narrow cell between two
    ///   roots that no raw breakpoint or midpoint reaches.
    #[test]
    fn false_universals_are_not_proved() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);

        for (label, coeffs, kind) in [
            ("x^2 > 0", vec![0, 0, 1], PredicateKind::Gt),
            ("x^4 > 0", vec![0, 0, 0, 0, 1], PredicateKind::Gt),
            (
                "2x^4+x^3-4x^2+3 >= 0",
                vec![3, 0, -4, 1, 2],
                PredicateKind::Ge,
            ),
        ] {
            let p = poly(&pool, x, &coeffs);
            let got = forall(&pool, x, atom(kind, p, zero)).expect("decidable");
            assert!(
                !got.truth,
                "`forall x. {label}` is false but decide returned true"
            );
        }
    }

    /// The fix must not flip true statements to false.
    #[test]
    fn true_universals_still_hold() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);

        for (label, coeffs, kind) in [
            ("x^2 >= 0", vec![0, 0, 1], PredicateKind::Ge),
            ("x^2 + 1 > 0", vec![1, 0, 1], PredicateKind::Gt),
            ("(x-1)^2 >= 0", vec![1, -2, 1], PredicateKind::Ge),
        ] {
            let p = poly(&pool, x, &coeffs);
            let got = forall(&pool, x, atom(kind, p, zero)).expect("decidable");
            assert!(
                got.truth,
                "`forall x. {label}` is true but decide returned false"
            );
        }
    }

    /// Bisection must survive a bracket whose endpoint is a *different* root.
    ///
    /// `2x⁴+x³-4x²+3` isolates one root inside `(-2,-1)` while `-1` is itself a
    /// root, so neighbouring brackets share that endpoint. Steering the
    /// bisection by an endpoint that evaluates to zero collapsed the bracket
    /// onto it and discarded the root being isolated — which is precisely how
    /// the narrow negative cell stayed unreachable.
    #[test]
    fn refinement_survives_a_root_valued_endpoint() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = poly(&pool, x, &[3, 0, -4, 1, 2]);
        let up = UniPoly::from_symbolic_clear_denoms(p, x, &pool).expect("polynomial");
        let master = combine_algebraic_master(x, &[up]);

        let iv = RootInterval::new(rug::Rational::from(-2), rug::Rational::from(-1));
        let (lo, hi) = refine_isolating(&master, &iv);

        assert!(lo <= hi, "refined bracket is inverted");
        assert!(
            hi < -1,
            "bracket collapsed onto the endpoint root -1 instead of isolating its own root"
        );
        assert!(lo > -2, "bracket did not tighten at all");
    }

    // -----------------------------------------------------------------------
    // The same completeness gap, two variables up.
    //
    // `project_and_sample_x` flags an irrational projection root as untested,
    // but the flag only escalated to a refusal for `=` / `≠` atoms, so `≤`/`≥`
    // still reported an unsatisfiability that was never checked at the one
    // point that could have satisfied it.
    // -----------------------------------------------------------------------

    /// `(x² − 2)² + y²`, non-negative and zero exactly at `(±√2, 0)`.
    fn touching_at_sqrt_two(pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
        let inner = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-2_i32)]);
        pool.add(vec![
            pool.pow(inner, pool.integer(2_i32)),
            pool.pow(y, pool.integer(2_i32)),
        ])
    }

    /// `∃x∃y. (x²−2)² + y² ≤ 0` is **true** at `(√2, 0)`; no rational sample
    /// can exhibit it, so the only sound answers are `true` and a refusal.
    #[test]
    fn two_var_nonstrict_boundary_at_an_irrational_root_is_not_denied() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let lhs = touching_at_sqrt_two(&pool, x, y);
        let body = atom(PredicateKind::Le, lhs, pool.integer(0_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(Formula::Exists {
                var: y,
                body: Box::new(body),
            }),
        };
        match decide(&f, &pool) {
            Ok(r) => assert!(
                r.truth,
                "`exists x exists y. (x^2-2)^2 + y^2 <= 0` is true at (sqrt 2, 0)"
            ),
            Err(e) => assert_eq!(e.code(), "E-CAD-001"),
        }
    }

    /// The dual: `∀x∀y. (x²−2)² + y² > 0` is **false**, and a `true` here is a
    /// machine-checked-looking proof of a false theorem.
    #[test]
    fn two_var_universal_over_an_irrational_root_is_not_proved() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let lhs = touching_at_sqrt_two(&pool, x, y);
        let body = atom(PredicateKind::Gt, lhs, pool.integer(0_i32));
        let f = Formula::Forall {
            var: x,
            body: Box::new(Formula::Forall {
                var: y,
                body: Box::new(body),
            }),
        };
        match decide(&f, &pool) {
            Ok(r) => assert!(
                !r.truth,
                "`forall x forall y. (x^2-2)^2 + y^2 > 0` is false at (sqrt 2, 0)"
            ),
            Err(e) => assert_eq!(e.code(), "E-CAD-001"),
        }
    }

    /// The control that the guard is not a blanket refusal of `≤`: the same
    /// polynomial shifted up by 1 is never `≤ 0`, and that `false` is sound
    /// because no boundary cell exists at all.
    #[test]
    fn two_var_nonstrict_unsatisfiable_still_decides_false() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let lhs = pool.add(vec![touching_at_sqrt_two(&pool, x, y), pool.integer(1_i32)]);
        let body = atom(PredicateKind::Le, lhs, pool.integer(0_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(Formula::Exists {
                var: y,
                body: Box::new(body),
            }),
        };
        let r = decide(&f, &pool).expect("two squares plus one is decidable");
        assert!(!r.truth, "two squares plus 1 is never <= 0");
    }

    /// The control that a *rational* boundary point is still found: the same
    /// shape with the double root at `x = 2/3` must come back `true`.
    #[test]
    fn two_var_nonstrict_boundary_at_a_rational_root_is_found() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let inner = pool.add(vec![
            pool.mul(vec![pool.integer(3_i32), x]),
            pool.integer(-2_i32),
        ]);
        let lhs = pool.add(vec![
            pool.pow(inner, pool.integer(2_i32)),
            pool.pow(y, pool.integer(2_i32)),
        ]);
        let body = atom(PredicateKind::Le, lhs, pool.integer(0_i32));
        let f = Formula::Exists {
            var: x,
            body: Box::new(Formula::Exists {
                var: y,
                body: Box::new(body),
            }),
        };
        let r = decide(&f, &pool).expect("a rational boundary point is reachable");
        assert!(r.truth, "(3x-2)^2 + y^2 <= 0 holds at (2/3, 0)");
    }
}
