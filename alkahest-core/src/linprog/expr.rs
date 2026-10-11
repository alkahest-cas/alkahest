//! Building a [`LinearProgram`] from alkahest expressions.
//!
//! The objective must be affine in the decision variables and every
//! constraint a `≤`, `≥` or `=` predicate between affine expressions (or an
//! `And` of such), all with exact rational coefficients. The variables are
//! **free** unless a constraint bounds them — the symbolic form states its
//! constraints explicitly, so nothing is assumed. Anything else is refused:
//! a product of two variables, a symbol that is not a decision variable
//! (parameters are not supported), a float or irrational coefficient, or a
//! strict inequality (which in general has no optimum).

use rug::Rational;

use super::{LinearProgram, LinprogError, Sense, VarBound};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};

/// An affine form `Σ coeffs[i]·vars[i] + constant`.
struct Affine {
    coeffs: Vec<Rational>,
    constant: Rational,
}

impl Affine {
    fn constant(n: usize, c: Rational) -> Self {
        Affine {
            coeffs: vec![Rational::new(); n],
            constant: c,
        }
    }

    fn is_constant(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }

    fn scale(&mut self, k: &Rational) {
        for c in self.coeffs.iter_mut() {
            *c *= k;
        }
        self.constant *= k;
    }

    fn add(&mut self, other: &Affine) {
        for (a, b) in self.coeffs.iter_mut().zip(&other.coeffs) {
            *a += b;
        }
        self.constant += &other.constant;
    }
}

fn affine(pool: &ExprPool, e: ExprId, vars: &[ExprId]) -> Result<Affine, LinprogError> {
    let n = vars.len();
    let nonlinear = |why: &str| LinprogError::NonLinear {
        detail: format!("{} — {why}", pool.display(e)),
    };
    if let Some(i) = vars.iter().position(|&v| v == e) {
        let mut a = Affine::constant(n, Rational::new());
        a.coeffs[i] = Rational::from(1);
        return Ok(a);
    }
    match pool.get(e) {
        ExprData::Integer(z) => Ok(Affine::constant(n, Rational::from(z.0))),
        ExprData::Rational(q) => Ok(Affine::constant(n, q.0)),
        ExprData::Float(_) => Err(LinprogError::Inexact {
            detail: format!("{} is a float; pass an exact rational", pool.display(e)),
        }),
        ExprData::Symbol { name, .. } => Err(nonlinear(&format!(
            "symbol {name} is not one of the decision variables (parameters are not supported)"
        ))),
        ExprData::Add(args) => {
            let mut acc = Affine::constant(n, Rational::new());
            for a in args {
                acc.add(&affine(pool, a, vars)?);
            }
            Ok(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = Affine::constant(n, Rational::from(1));
            for a in args {
                let f = affine(pool, a, vars)?;
                if f.is_constant() {
                    acc.scale(&f.constant);
                } else if acc.is_constant() {
                    let k = acc.constant.clone();
                    acc = f;
                    acc.scale(&k);
                } else {
                    return Err(nonlinear("a product of two non-constant factors"));
                }
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => {
            let ExprData::Integer(k) = pool.get(exp) else {
                return Err(not_rational(
                    pool,
                    e,
                    vars,
                    "a power with a non-integer exponent",
                ));
            };
            let b = affine(pool, base, vars)?;
            if k.0 == 1 {
                return Ok(b);
            }
            if k.0 == 0 {
                return Ok(Affine::constant(n, Rational::from(1)));
            }
            if !b.is_constant() {
                return Err(nonlinear("a power of a decision variable"));
            }
            if b.constant == 0 && k.0 < 0 {
                return Err(nonlinear("division by zero"));
            }
            let Some(k) = k.0.to_i32() else {
                return Err(nonlinear("an exponent too large to evaluate"));
            };
            let base = b.constant;
            let mag = rug::ops::Pow::pow(base, k.unsigned_abs());
            let val = if k < 0 { mag.recip() } else { mag };
            Ok(Affine::constant(n, val))
        }
        _ => Err(not_rational(
            pool,
            e,
            vars,
            "only sums and rational multiples of the variables are affine",
        )),
    }
}

/// A subexpression that is neither affine nor a rational number: if it is
/// free of the decision variables it is an irrational or unevaluated constant
/// (`E-LP-002`), otherwise a non-affine use of a variable (`E-LP-003`).
fn not_rational(pool: &ExprPool, e: ExprId, vars: &[ExprId], why: &str) -> LinprogError {
    if mentions_any(pool, e, vars) {
        LinprogError::NonLinear {
            detail: format!("{} — {why}", pool.display(e)),
        }
    } else {
        LinprogError::Inexact {
            detail: format!(
                "{} is not a rational number; coefficients must be exact rationals",
                pool.display(e)
            ),
        }
    }
}

fn mentions_any(pool: &ExprPool, e: ExprId, vars: &[ExprId]) -> bool {
    if vars.contains(&e) {
        return true;
    }
    match pool.get(e) {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().any(|&a| mentions_any(pool, a, vars))
        }
        ExprData::Pow { base, exp } => {
            mentions_any(pool, base, vars) || mentions_any(pool, exp, vars)
        }
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => false,
        // Anything structural (piecewise, predicates, binders) is treated as
        // possibly mentioning a variable, so it is reported as non-affine.
        _ => true,
    }
}

/// The affine form of `expr` in `vars`: `(coefficients, constant)` with
/// `expr = Σ coefficients[i]·vars[i] + constant`.
///
/// # Errors
/// `E-LP-003` when `expr` is not affine in `vars` or mentions another symbol;
/// `E-LP-002` for a float coefficient.
pub fn linear_form(
    pool: &ExprPool,
    expr: ExprId,
    vars: &[ExprId],
) -> Result<(Vec<Rational>, Rational), LinprogError> {
    let a = affine(pool, expr, vars)?;
    Ok((a.coeffs, a.constant))
}

fn collect_relations(
    pool: &ExprPool,
    c: ExprId,
    out: &mut Vec<(PredicateKind, ExprId, ExprId)>,
) -> Result<(), LinprogError> {
    let refuse = |why: &str| LinprogError::UnsupportedConstraint {
        detail: format!("{} — {why}", pool.display(c)),
    };
    let ExprData::Predicate { kind, args } = pool.get(c) else {
        return Err(refuse(
            "not a relation; build one with pool.le / pool.ge / pool.pred_eq",
        ));
    };
    match kind {
        PredicateKind::Le | PredicateKind::Ge | PredicateKind::Eq => {
            if args.len() != 2 {
                return Err(refuse("a relation needs two sides"));
            }
            out.push((kind, args[0], args[1]));
            Ok(())
        }
        PredicateKind::And => {
            for a in args {
                collect_relations(pool, a, out)?;
            }
            Ok(())
        }
        PredicateKind::True => Ok(()),
        PredicateKind::Lt | PredicateKind::Gt => Err(refuse(
            "a strict inequality has no optimum in general; use <= or >=",
        )),
        _ => Err(refuse(
            "only <=, >=, == (and And of them) are linear constraints",
        )),
    }
}

/// Build a [`LinearProgram`] from an affine objective and relational
/// constraints in the decision variables `vars` (which are free unless
/// constrained). Constraints appear in the programme in the order given:
/// each `≤` / `≥` becomes one `A_ub` row (a `≥` is negated into `≤`), each
/// `=` one `A_eq` row, so certificate multipliers line up with them.
///
/// # Errors
/// `E-LP-003` (non-affine), `E-LP-002` (float), `E-LP-004` (strict inequality
/// or a non-relation).
pub fn program_from_exprs(
    pool: &ExprPool,
    objective: ExprId,
    constraints: &[ExprId],
    vars: &[ExprId],
    sense: Sense,
) -> Result<LinearProgram, LinprogError> {
    for (i, &v) in vars.iter().enumerate() {
        if !matches!(pool.get(v), ExprData::Symbol { .. }) {
            return Err(LinprogError::Shape {
                detail: format!("decision variable {} is not a symbol", pool.display(v)),
            });
        }
        if vars[..i].contains(&v) {
            return Err(LinprogError::Shape {
                detail: format!("decision variable {} is listed twice", pool.display(v)),
            });
        }
    }
    let (c, _) = linear_form(pool, objective, vars)?;
    let mut lp = LinearProgram::new(c);
    lp.bounds = vec![VarBound::free(); vars.len()];
    lp.sense = sense;
    let mut rels = Vec::new();
    for &con in constraints {
        collect_relations(pool, con, &mut rels)?;
    }
    for (kind, lhs, rhs) in rels {
        let l = affine(pool, lhs, vars)?;
        let mut r = affine(pool, rhs, vars)?;
        // lhs − rhs  (rel)  0
        r.scale(&Rational::from(-1));
        let mut d = l;
        d.add(&r);
        let rhs_val = Rational::from(-&d.constant);
        match kind {
            PredicateKind::Le => {
                lp.a_ub.push(d.coeffs);
                lp.b_ub.push(rhs_val);
            }
            PredicateKind::Ge => {
                lp.a_ub.push(d.coeffs.into_iter().map(|v| -v).collect());
                lp.b_ub.push(-rhs_val);
            }
            _ => {
                lp.a_eq.push(d.coeffs);
                lp.b_eq.push(rhs_val);
            }
        }
    }
    Ok(lp)
}
