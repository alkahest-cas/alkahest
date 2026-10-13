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

use super::{LinearProgram, LinprogError, ParametricDelta, Sense, VarBound};
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

/// `Σ (a_i + θ·b_i)·vars[i] + (k₀ + θ·k₁)`: affine in the variables with
/// coefficients affine in the parameter `θ`.
struct Bilinear {
    coeffs: Vec<[Rational; 2]>,
    constant: [Rational; 2],
}

impl Bilinear {
    fn constant(n: usize, k0: Rational, k1: Rational) -> Self {
        Bilinear {
            coeffs: vec![[Rational::new(), Rational::new()]; n],
            constant: [k0, k1],
        }
    }

    fn has_vars(&self) -> bool {
        self.coeffs.iter().any(|[a, b]| *a != 0 || *b != 0)
    }

    fn add(&mut self, o: &Bilinear) {
        for (a, b) in self.coeffs.iter_mut().zip(&o.coeffs) {
            a[0] += &b[0];
            a[1] += &b[1];
        }
        self.constant[0] += &o.constant[0];
        self.constant[1] += &o.constant[1];
    }
}

/// `(a₀ + θa₁)(b₀ + θb₁)` when the θ² term vanishes.
fn mul_affine(a: &[Rational; 2], b: &[Rational; 2]) -> Option<[Rational; 2]> {
    if a[1] != 0 && b[1] != 0 {
        return None;
    }
    Some([
        Rational::from(&a[0] * &b[0]),
        Rational::from(&a[0] * &b[1]) + Rational::from(&a[1] * &b[0]),
    ])
}

fn bilinear(
    pool: &ExprPool,
    e: ExprId,
    vars: &[ExprId],
    theta: ExprId,
) -> Result<Bilinear, LinprogError> {
    let n = vars.len();
    let param = |why: &str| LinprogError::UnsupportedParametric {
        detail: format!("{} — {why}", pool.display(e)),
    };
    let nonlinear = |why: &str| LinprogError::NonLinear {
        detail: format!("{} — {why}", pool.display(e)),
    };
    if let Some(i) = vars.iter().position(|&v| v == e) {
        let mut a = Bilinear::constant(n, Rational::new(), Rational::new());
        a.coeffs[i][0] = Rational::from(1);
        return Ok(a);
    }
    if e == theta {
        return Ok(Bilinear::constant(n, Rational::new(), Rational::from(1)));
    }
    let mut with_theta: Vec<ExprId> = vars.to_vec();
    with_theta.push(theta);
    match pool.get(e) {
        ExprData::Integer(z) => Ok(Bilinear::constant(n, Rational::from(z.0), Rational::new())),
        ExprData::Rational(q) => Ok(Bilinear::constant(n, q.0, Rational::new())),
        ExprData::Float(_) => Err(LinprogError::Inexact {
            detail: format!("{} is a float; pass an exact rational", pool.display(e)),
        }),
        ExprData::Symbol { name, .. } => Err(nonlinear(&format!(
            "symbol {name} is neither a decision variable nor the parameter"
        ))),
        ExprData::Add(args) => {
            let mut acc = Bilinear::constant(n, Rational::new(), Rational::new());
            for a in args {
                acc.add(&bilinear(pool, a, vars, theta)?);
            }
            Ok(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = Bilinear::constant(n, Rational::from(1), Rational::new());
            for a in args {
                let f = bilinear(pool, a, vars, theta)?;
                let (scalar, mut other) = if !f.has_vars() {
                    (f.constant.clone(), acc)
                } else if !acc.has_vars() {
                    (acc.constant.clone(), f)
                } else {
                    return Err(nonlinear("a product of two decision variables"));
                };
                let theta_sq = || param("the parameter appears squared");
                for c in other.coeffs.iter_mut() {
                    *c = mul_affine(c, &scalar).ok_or_else(theta_sq)?;
                }
                other.constant = mul_affine(&other.constant, &scalar).ok_or_else(theta_sq)?;
                acc = other;
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => {
            let b = bilinear(pool, base, vars, theta)?;
            if let ExprData::Integer(k) = pool.get(exp) {
                if k.0 == 1 {
                    return Ok(b);
                }
                if k.0 == 0 {
                    return Ok(Bilinear::constant(n, Rational::from(1), Rational::new()));
                }
                if b.has_vars() {
                    return Err(nonlinear("a power of a decision variable"));
                }
                if b.constant[1] != 0 {
                    return Err(param("a power of the parameter"));
                }
            } else if mentions_any(pool, e, &with_theta) && !mentions_any(pool, e, vars) {
                return Err(param("the parameter enters non-linearly"));
            }
            let a = affine(pool, e, vars)?;
            Ok(Bilinear::constant(n, a.constant, Rational::new()))
        }
        _ => {
            if mentions_any(pool, e, &[theta]) && !mentions_any(pool, e, vars) {
                return Err(param("the parameter enters non-linearly"));
            }
            Err(not_rational(
                pool,
                e,
                vars,
                "only sums and rational multiples of the variables are affine",
            ))
        }
    }
}

/// Build a parametric programme from expressions: like
/// [`program_from_exprs`], with one more symbol `parameter` that may appear
/// affinely **either** in the objective's coefficients **or** in the
/// constraints' constant terms (the right-hand side) — not both, and never
/// multiplying a decision variable inside a constraint.
///
/// Returns the programme at `θ = 0` and the direction it moves in.
///
/// # Errors
/// `E-LP-007` for an unsupported parametric form (θ in the constraint matrix,
/// θ², θ in the objective and the right-hand side at once); otherwise as
/// [`program_from_exprs`].
pub fn parametric_from_exprs(
    pool: &ExprPool,
    objective: ExprId,
    constraints: &[ExprId],
    vars: &[ExprId],
    parameter: ExprId,
    sense: Sense,
) -> Result<(LinearProgram, ParametricDelta), LinprogError> {
    if !matches!(pool.get(parameter), ExprData::Symbol { .. }) {
        return Err(LinprogError::Shape {
            detail: format!("the parameter {} is not a symbol", pool.display(parameter)),
        });
    }
    if vars.contains(&parameter) {
        return Err(LinprogError::Shape {
            detail: format!(
                "the parameter {} is also listed as a decision variable",
                pool.display(parameter)
            ),
        });
    }
    // Structure checks and the θ-free programme come from the plain builder
    // applied to the θ = 0 slice below; first validate the variable list.
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
    let obj = bilinear(pool, objective, vars, parameter)?;
    let c: Vec<Rational> = obj.coeffs.iter().map(|k| k[0].clone()).collect();
    let dc: Vec<Rational> = obj.coeffs.iter().map(|k| k[1].clone()).collect();
    let mut lp = LinearProgram::new(c);
    lp.bounds = vec![VarBound::free(); vars.len()];
    lp.sense = sense;
    let mut db_ub = Vec::new();
    let mut db_eq = Vec::new();
    let mut rels = Vec::new();
    for &con in constraints {
        collect_relations(pool, con, &mut rels)?;
    }
    for (kind, lhs, rhs) in rels {
        let l = bilinear(pool, lhs, vars, parameter)?;
        let r = bilinear(pool, rhs, vars, parameter)?;
        let mut d = l;
        let neg = Bilinear {
            coeffs: r
                .coeffs
                .iter()
                .map(|[a, b]| [-a.clone(), -b.clone()])
                .collect(),
            constant: [-r.constant[0].clone(), -r.constant[1].clone()],
        };
        d.add(&neg);
        if d.coeffs.iter().any(|k| k[1] != 0) {
            return Err(LinprogError::UnsupportedParametric {
                detail: format!(
                    "{} — the parameter multiplies a decision variable in a constraint \
                     (only the right-hand side may move)",
                    pool.display(con_of(pool, lhs, rhs))
                ),
            });
        }
        let row: Vec<Rational> = d.coeffs.iter().map(|k| k[0].clone()).collect();
        let (b0, b1) = (-d.constant[0].clone(), -d.constant[1].clone());
        match kind {
            PredicateKind::Le => {
                lp.a_ub.push(row);
                lp.b_ub.push(b0);
                db_ub.push(b1);
            }
            PredicateKind::Ge => {
                lp.a_ub.push(row.into_iter().map(|v| -v).collect());
                lp.b_ub.push(-b0);
                db_ub.push(-b1);
            }
            _ => {
                lp.a_eq.push(row);
                lp.b_eq.push(b0);
                db_eq.push(b1);
            }
        }
    }
    let objective_moves = dc.iter().any(|v| *v != 0);
    let rhs_moves = db_ub.iter().chain(&db_eq).any(|v| *v != 0);
    if objective_moves && rhs_moves {
        return Err(LinprogError::UnsupportedParametric {
            detail: "the parameter moves both the objective and the right-hand side".into(),
        });
    }
    let delta = if objective_moves {
        ParametricDelta::Objective { c: dc }
    } else {
        ParametricDelta::Rhs {
            b_ub: db_ub,
            b_eq: db_eq,
        }
    };
    Ok((lp, delta))
}

/// The relation `lhs ⋈ rhs`, for messages (the left side alone suffices).
fn con_of(_pool: &ExprPool, lhs: ExprId, _rhs: ExprId) -> ExprId {
    lhs
}
