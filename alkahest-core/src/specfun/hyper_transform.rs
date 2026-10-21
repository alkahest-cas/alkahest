//! Transformations of `₂F₁` and `₁F₁` (F26).
//!
//! [`hyper_transform`] rewrites one `hyper` node by a classical identity and
//! returns the exact right-hand side (Gamma prefactors as `gamma` nodes, the
//! transformed series as new `hyper` nodes). Every formula is the
//! non-regularized form of Abramowitz–Stegun 15.3 / DLMF 15.8, 13.2.39, with
//! principal branches throughout; each holds as an identity of analytic
//! functions on the region listed — off it the two sides are different
//! branches, so the region is part of the result's contract.
//!
//! | transformation | identity | valid for |
//! |---|---|---|
//! | `Euler` | `(1−z)^{c−a−b} F(c−a, c−b; c; z)` | `z ∉ [1, ∞)` |
//! | `PfaffA` | `(1−z)^{−a} F(a, c−b; c; z/(z−1))` | `z ∉ [1, ∞)` |
//! | `PfaffB` | `(1−z)^{−b} F(c−a, b; c; z/(z−1))` | `z ∉ [1, ∞)` |
//! | `OneMinusZ` | `Γ(c)Γ(c−a−b)/(Γ(c−a)Γ(c−b)) F(a, b; a+b−c+1; 1−z) + (1−z)^{c−a−b} Γ(c)Γ(a+b−c)/(Γ(a)Γ(b)) F(c−a, c−b; c−a−b+1; 1−z)` | `z ∉ (−∞, 0] ∪ [1, ∞)`, `c−a−b ∉ ℤ` |
//! | `InverseZ` | `Γ(c)Γ(b−a)/(Γ(b)Γ(c−a)) (−z)^{−a} F(a, a−c+1; a−b+1; 1/z) + (a ↔ b)` | `z ∉ [0, ∞)`, `a−b ∉ ℤ` |
//! | `InverseOneMinusZ` | `(1−z)^{−a} Γ(c)Γ(b−a)/(Γ(b)Γ(c−a)) F(a, c−b; a−b+1; 1/(1−z)) + (a ↔ b)` | `z ∉ [0, ∞)`, `a−b ∉ ℤ` |
//! | `OneMinusInverseZ` | `Γ(c)Γ(c−a−b)/(Γ(c−a)Γ(c−b)) z^{−a} F(a, a−c+1; a+b−c+1; 1−1/z) + Γ(c)Γ(a+b−c)/(Γ(a)Γ(b)) (1−z)^{c−a−b} z^{a−c} F(c−a, 1−a; c−a−b+1; 1−1/z)` | `z ∉ (−∞, 0] ∪ [1, ∞)`, `c−a−b ∉ ℤ` |
//! | `Quadratic`, `c = 2b` | `(1−z/2)^{−a} F(a/2, a/2+½; b+½; (z/(2−z))²)` | `z ∉ [1, ∞)` (DLMF 15.8.13) |
//! | `Quadratic`, `c = a−b+1` | `(1+z)^{−a} F(a/2, a/2+½; a−b+1; 4z/(1+z)²)` | `|z| < 1` (DLMF 15.8.15) |
//! | `Quadratic`, `2c = a+b+1` | `F(a/2, b/2; c; 4z(1−z))` | `Re z < ½` (DLMF 15.8.18) |
//! | `Kummer` (`₁F₁`) | `eᶻ ₁F₁(b−a; b; −z)` | all `z` |
//!
//! The six linear fractional maps of `z` are covered by `Euler`/`Pfaff`
//! (`z`, `z/(z−1)`) and the four connection formulas (`1−z`, `1/z`,
//! `1/(1−z)`, `1−1/z`).
//!
//! # Refusals
//!
//! `E-SPECFN-009` ([`SpecialFunctionError::TransformationNotApplicable`]) for
//! a node that is not the right `ₚFq`, for the logarithmic case of a two-term
//! formula (the governing difference an integer *literal* — DLMF 15.8.8,
//! 15.8.10, not built), for `c` a non-positive integer, and for parameters
//! outside every quadratic family. A symbolic difference is taken to be
//! generic (not an integer), as stated in the table.

use rug::Rational;

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::orthopoly::{mk_add, mk_mul, rational_literal};
use crate::simplify::simplify;

use super::hyper::{hyper, parts, HYPER};
use super::SpecialFunctionError;

/// A classical transformation of `₂F₁(a, b; c; z)` (or `₁F₁` for
/// [`Kummer`](Self::Kummer)); the identities and their regions of validity
/// are in the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HyperTransformation {
    /// Euler: `(1−z)^{c−a−b} F(c−a, c−b; c; z)`.
    Euler,
    /// Pfaff in `a`: `(1−z)^{−a} F(a, c−b; c; z/(z−1))`.
    PfaffA,
    /// Pfaff in `b`: `(1−z)^{−b} F(c−a, b; c; z/(z−1))`.
    PfaffB,
    /// The connection formula to `1 − z` (A&S 15.3.6).
    OneMinusZ,
    /// The connection formula to `1/z` (A&S 15.3.7).
    InverseZ,
    /// The connection formula to `1/(1 − z)` (A&S 15.3.8).
    InverseOneMinusZ,
    /// The connection formula to `1 − 1/z` (A&S 15.3.9).
    OneMinusInverseZ,
    /// The quadratic transformation of the family the parameters belong to
    /// (`c = 2b`, `c = a − b + 1` or `2c = a + b + 1`, either order of `a, b`).
    Quadratic,
    /// Kummer's transformation of `₁F₁`: `eᶻ ₁F₁(b−a; b; −z)`.
    Kummer,
}

impl HyperTransformation {
    /// Every transformation, in the order of the module table.
    pub const ALL: [HyperTransformation; 9] = [
        HyperTransformation::Euler,
        HyperTransformation::PfaffA,
        HyperTransformation::PfaffB,
        HyperTransformation::OneMinusZ,
        HyperTransformation::InverseZ,
        HyperTransformation::InverseOneMinusZ,
        HyperTransformation::OneMinusInverseZ,
        HyperTransformation::Quadratic,
        HyperTransformation::Kummer,
    ];

    /// The name used by the Python binding (`"euler"`, `"pfaff"`,
    /// `"pfaff_b"`, `"1-z"`, `"1/z"`, `"1/(1-z)"`, `"1-1/z"`, `"quadratic"`,
    /// `"kummer"`).
    pub fn name(self) -> &'static str {
        match self {
            HyperTransformation::Euler => "euler",
            HyperTransformation::PfaffA => "pfaff",
            HyperTransformation::PfaffB => "pfaff_b",
            HyperTransformation::OneMinusZ => "1-z",
            HyperTransformation::InverseZ => "1/z",
            HyperTransformation::InverseOneMinusZ => "1/(1-z)",
            HyperTransformation::OneMinusInverseZ => "1-1/z",
            HyperTransformation::Quadratic => "quadratic",
            HyperTransformation::Kummer => "kummer",
        }
    }

    /// The transformation called `name` ([`name`](Self::name); `"z/(z-1)"`
    /// is accepted for [`PfaffA`](Self::PfaffA)).
    pub fn from_name(name: &str) -> Option<HyperTransformation> {
        if name == "z/(z-1)" {
            return Some(HyperTransformation::PfaffA);
        }
        HyperTransformation::ALL
            .into_iter()
            .find(|t| t.name() == name)
    }
}

// ---------------------------------------------------------------------------
// Expression helpers
// ---------------------------------------------------------------------------

fn s(pool: &ExprPool, e: ExprId) -> ExprId {
    simplify(e, pool).value
}

fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
    mk_mul(pool, vec![pool.integer(-1), e])
}

fn inv(pool: &ExprPool, e: ExprId) -> ExprId {
    pool.pow(e, pool.integer(-1))
}

/// `Σ terms`, simplified (parameter arithmetic: `c − a − b`, `a/2 + ½`).
fn sum(pool: &ExprPool, terms: Vec<ExprId>) -> ExprId {
    s(pool, mk_add(pool, terms))
}

fn sub(pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
    sum(pool, vec![x, neg(pool, y)])
}

fn half_of(pool: &ExprPool, x: ExprId) -> ExprId {
    s(pool, mk_mul(pool, vec![pool.rational(1, 2), x]))
}

fn one_minus(pool: &ExprPool, z: ExprId) -> ExprId {
    mk_add(pool, vec![pool.integer(1), neg(pool, z)])
}

fn is_integer_lit(pool: &ExprPool, e: ExprId) -> bool {
    rational_literal(pool, e).is_some_and(|q| *q.denom() == 1)
}

fn is_nonpositive_int(pool: &ExprPool, e: ExprId) -> bool {
    rational_literal(pool, e).is_some_and(|q| *q.denom() == 1 && q <= 0)
}

fn is_zero(pool: &ExprPool, e: ExprId) -> bool {
    rational_literal(pool, e).is_some_and(|q| q == 0)
}

/// `Π Γ(num) / Π Γ(den)`, or `None` when a denominator Gamma sits at a pole
/// (`1/Γ(−n) = 0`: the whole term vanishes).
fn gamma_ratio(pool: &ExprPool, num: &[ExprId], den: &[ExprId]) -> Option<ExprId> {
    if den.iter().any(|&d| is_nonpositive_int(pool, d)) {
        return None;
    }
    let g = |e: ExprId| pool.func("gamma", vec![e]);
    let mut f: Vec<ExprId> = num.iter().map(|&x| g(x)).collect();
    f.extend(den.iter().map(|&x| inv(pool, g(x))));
    Some(s(pool, mk_mul(pool, f)))
}

fn refuse(t: HyperTransformation, reason: impl Into<String>) -> SpecialFunctionError {
    SpecialFunctionError::TransformationNotApplicable {
        transformation: t.name(),
        reason: reason.into(),
    }
}

// ---------------------------------------------------------------------------
// The one-term transformations (also used by `expand_hyper`)
// ---------------------------------------------------------------------------

/// One `prefactor · ₚFq(a′; b′; z′)` term.
pub(crate) struct Term {
    pub(crate) prefactor: ExprId,
    pub(crate) a: Vec<ExprId>,
    pub(crate) b: Vec<ExprId>,
    pub(crate) z: ExprId,
}

/// `z/(z − 1)` written as `−z/(1 − z)`, so a closed form that reads a
/// negated argument (`atan`, `cos`, `J`) sees `w = z/(1 − z)`.
fn pfaff_arg(pool: &ExprPool, z: ExprId) -> ExprId {
    mk_mul(
        pool,
        vec![pool.integer(-1), z, inv(pool, one_minus(pool, z))],
    )
}

/// The single-term right-hand side of `t` for `ₚFq(a; b; z)`, or `None` when
/// `t` is not a one-term transformation of this `p, q`.
pub(crate) fn one_term(
    t: HyperTransformation,
    a: &[ExprId],
    b: &[ExprId],
    z: ExprId,
    pool: &ExprPool,
) -> Option<Term> {
    use HyperTransformation as T;
    match (t, a, b) {
        (T::Euler, [a, bb], [c]) => {
            let e = sub(pool, *c, sum(pool, vec![*a, *bb]));
            Some(Term {
                prefactor: pool.pow(one_minus(pool, z), e),
                a: vec![sub(pool, *c, *a), sub(pool, *c, *bb)],
                b: vec![*c],
                z,
            })
        }
        (T::PfaffA, [a, bb], [c]) => Some(Term {
            prefactor: pool.pow(one_minus(pool, z), neg(pool, *a)),
            a: vec![*a, sub(pool, *c, *bb)],
            b: vec![*c],
            z: pfaff_arg(pool, z),
        }),
        (T::PfaffB, [a, bb], [c]) => Some(Term {
            prefactor: pool.pow(one_minus(pool, z), neg(pool, *bb)),
            a: vec![sub(pool, *c, *a), *bb],
            b: vec![*c],
            z: pfaff_arg(pool, z),
        }),
        (T::Kummer, [a], [bb]) => Some(Term {
            prefactor: pool.func("exp", vec![z]),
            a: vec![sub(pool, *bb, *a)],
            b: vec![*bb],
            z: neg(pool, z),
        }),
        _ => None,
    }
}

fn term_expr(t: Term, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    let f = hyper(&t.a, &t.b, t.z, pool)?;
    Ok(mk_mul(pool, vec![t.prefactor, f]))
}

// ---------------------------------------------------------------------------
// The two-term connection formulas
// ---------------------------------------------------------------------------

/// `coef · prefactor · ₂F₁(a′; c′; z′)`, `coef = None` for a vanishing term.
type Piece = (Option<ExprId>, ExprId, [ExprId; 2], ExprId, ExprId);

fn assemble(pieces: Vec<Piece>, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    let mut terms = Vec::with_capacity(2);
    for (coef, pref, [a1, a2], c, z) in pieces {
        let Some(coef) = coef else { continue };
        let f = hyper(&[a1, a2], &[c], z, pool)?;
        terms.push(mk_mul(pool, vec![coef, pref, f]));
    }
    Ok(match terms.len() {
        0 => pool.integer(0),
        _ => mk_add(pool, terms),
    })
}

fn connection(
    t: HyperTransformation,
    a: ExprId,
    b: ExprId,
    c: ExprId,
    z: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    use HyperTransformation as T;
    if is_nonpositive_int(pool, c) {
        return Err(refuse(
            t,
            format!(
                "c = {} is a non-positive integer, so Γ(c) has a pole",
                pool.display(c)
            ),
        ));
    }
    let one = pool.integer(1);
    let s_ = sub(pool, c, sum(pool, vec![a, b])); // c − a − b
    let ms = neg(pool, s_);
    let ms = s(pool, ms); // a + b − c
    let d = sub(pool, a, b); // a − b
    let md = s(pool, neg(pool, d)); // b − a
    let cma = sub(pool, c, a);
    let cmb = sub(pool, c, b);
    let omz = one_minus(pool, z);
    let need_noninteger = |x: ExprId, what: &str| -> Result<(), SpecialFunctionError> {
        if is_integer_lit(pool, x) {
            Err(refuse(
                t,
                format!(
                    "{what} = {} is an integer (the logarithmic case, DLMF 15.8.8/15.8.10)",
                    pool.display(x)
                ),
            ))
        } else {
            Ok(())
        }
    };
    match t {
        T::OneMinusZ => {
            need_noninteger(s_, "c − a − b")?;
            let w = omz;
            assemble(
                vec![
                    (
                        gamma_ratio(pool, &[c, s_], &[cma, cmb]),
                        one,
                        [a, b],
                        sum(pool, vec![ms, one]),
                        w,
                    ),
                    (
                        gamma_ratio(pool, &[c, ms], &[a, b]),
                        pool.pow(omz, s_),
                        [cma, cmb],
                        sum(pool, vec![s_, one]),
                        w,
                    ),
                ],
                pool,
            )
        }
        T::InverseZ => {
            need_noninteger(d, "a − b")?;
            let w = inv(pool, z);
            let mz = neg(pool, z);
            assemble(
                vec![
                    (
                        gamma_ratio(pool, &[c, md], &[b, cma]),
                        pool.pow(mz, neg(pool, a)),
                        [a, sum(pool, vec![a, neg(pool, c), one])],
                        sum(pool, vec![d, one]),
                        w,
                    ),
                    (
                        gamma_ratio(pool, &[c, d], &[a, cmb]),
                        pool.pow(mz, neg(pool, b)),
                        [b, sum(pool, vec![b, neg(pool, c), one])],
                        sum(pool, vec![md, one]),
                        w,
                    ),
                ],
                pool,
            )
        }
        T::InverseOneMinusZ => {
            need_noninteger(d, "a − b")?;
            let w = inv(pool, omz);
            assemble(
                vec![
                    (
                        gamma_ratio(pool, &[c, md], &[b, cma]),
                        pool.pow(omz, neg(pool, a)),
                        [a, cmb],
                        sum(pool, vec![d, one]),
                        w,
                    ),
                    (
                        gamma_ratio(pool, &[c, d], &[a, cmb]),
                        pool.pow(omz, neg(pool, b)),
                        [b, cma],
                        sum(pool, vec![md, one]),
                        w,
                    ),
                ],
                pool,
            )
        }
        T::OneMinusInverseZ => {
            need_noninteger(s_, "c − a − b")?;
            let w = one_minus(pool, inv(pool, z));
            assemble(
                vec![
                    (
                        gamma_ratio(pool, &[c, s_], &[cma, cmb]),
                        pool.pow(z, neg(pool, a)),
                        [a, sum(pool, vec![a, neg(pool, c), one])],
                        sum(pool, vec![ms, one]),
                        w,
                    ),
                    (
                        gamma_ratio(pool, &[c, ms], &[a, b]),
                        mk_mul(pool, vec![pool.pow(omz, s_), pool.pow(z, sub(pool, a, c))]),
                        [cma, sub(pool, one, a)],
                        sum(pool, vec![s_, one]),
                        w,
                    ),
                ],
                pool,
            )
        }
        _ => unreachable!("connection() is only called for the two-term formulas"),
    }
}

// ---------------------------------------------------------------------------
// Quadratic transformations
// ---------------------------------------------------------------------------

fn quadratic(
    a: ExprId,
    b: ExprId,
    c: ExprId,
    z: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    let one = pool.integer(1);
    let half = pool.rational(1, 2);
    for (x, y) in [(a, b), (b, a)] {
        // c = 2y: (1 − z/2)^{−x} F(x/2, x/2 + ½; y + ½; (z/(2 − z))²)
        if is_zero(
            pool,
            sub(pool, c, s(pool, mk_mul(pool, vec![pool.integer(2), y]))),
        ) {
            let x2 = half_of(pool, x);
            let ratio = mk_mul(
                pool,
                vec![
                    z,
                    inv(pool, mk_add(pool, vec![pool.integer(2), neg(pool, z)])),
                ],
            );
            let f = hyper(
                &[x2, sum(pool, vec![x2, half])],
                &[sum(pool, vec![y, half])],
                pool.pow(ratio, pool.integer(2)),
                pool,
            )?;
            let base = mk_add(pool, vec![one, mk_mul(pool, vec![pool.rational(-1, 2), z])]);
            return Ok(mk_mul(pool, vec![pool.pow(base, neg(pool, x)), f]));
        }
    }
    for (x, y) in [(a, b), (b, a)] {
        // c = x − y + 1: (1 + z)^{−x} F(x/2, x/2 + ½; c; 4z/(1 + z)²)
        if is_zero(pool, sub(pool, c, sum(pool, vec![x, neg(pool, y), one]))) {
            let x2 = half_of(pool, x);
            let opz = mk_add(pool, vec![one, z]);
            let arg = mk_mul(
                pool,
                vec![pool.integer(4), z, pool.pow(opz, pool.integer(-2))],
            );
            let f = hyper(&[x2, sum(pool, vec![x2, half])], &[c], arg, pool)?;
            return Ok(mk_mul(pool, vec![pool.pow(opz, neg(pool, x)), f]));
        }
    }
    // 2c = a + b + 1: F(a/2, b/2; c; 4z(1 − z))
    if is_zero(
        pool,
        sub(
            pool,
            s(pool, mk_mul(pool, vec![pool.integer(2), c])),
            sum(pool, vec![a, b, one]),
        ),
    ) {
        let arg = mk_mul(pool, vec![pool.integer(4), z, one_minus(pool, z)]);
        return hyper(&[half_of(pool, a), half_of(pool, b)], &[c], arg, pool);
    }
    Err(refuse(
        HyperTransformation::Quadratic,
        "the parameters are in no quadratic family (c = 2b, c = a − b + 1 or \
         2c = a + b + 1, in either order of a, b)",
    ))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Rewrite the hypergeometric node `expr` by the transformation `t` and
/// return the exact right-hand side (module docs: the identities and the
/// region of `z` on which each holds).
///
/// # Errors
///
/// `E-SPECFN-009` when `expr` is not a `₂F₁` node (`₁F₁` for
/// [`HyperTransformation::Kummer`]), for the logarithmic case of a two-term
/// formula (`a − b` or `c − a − b` an integer literal), for `c` a
/// non-positive integer, for parameters outside every quadratic family, and
/// for a literal `z` outside the identity's region (for `Quadratic`, a
/// literal must lie in `(−1, ½)`, inside every family's region);
/// `E-SPECFN-006` if a transformed node has a pole parameter; `E-BUDGET-*`.
pub fn hyper_transform(
    expr: ExprId,
    t: HyperTransformation,
    pool: &ExprPool,
) -> Result<ExprId, SpecialFunctionError> {
    crate::budget::check()?;
    let args = pool.with(expr, |d| match d {
        ExprData::Func { name, args } if name == HYPER => Some(args.clone()),
        _ => None,
    });
    let Some((a, b, z)) = args.as_deref().and_then(|v| parts(v, pool)) else {
        return Err(refuse(
            t,
            format!("{} is not a hyper(...) node", pool.display(expr)),
        ));
    };
    let want = if t == HyperTransformation::Kummer {
        (1, 1)
    } else {
        (2, 1)
    };
    if (a.len(), b.len()) != want {
        return Err(refuse(
            t,
            format!(
                "it applies to {}F{}, not to {}F{}",
                want.0,
                want.1,
                a.len(),
                b.len()
            ),
        ));
    }
    use HyperTransformation as T;
    // A literal z outside the identity's region: the right-hand side would be
    // a different branch (or not finite), so refuse rather than return it.
    if let Some(q) = rational_literal(pool, z) {
        let outside = match t {
            T::Euler | T::PfaffA | T::PfaffB => q >= 1,
            T::OneMinusZ | T::OneMinusInverseZ => q <= 0 || q >= 1,
            T::InverseZ | T::InverseOneMinusZ => q >= 0,
            // (−1, ½) lies inside every family's region.
            T::Quadratic => q.clone().abs() >= 1 || q >= Rational::from((1, 2)),
            _ => false,
        };
        if outside {
            return Err(refuse(
                t,
                format!(
                    "z = {} is outside the region where the identity holds",
                    pool.display(z)
                ),
            ));
        }
    }
    match t {
        T::Euler | T::PfaffA | T::PfaffB | T::Kummer => {
            let term = one_term(t, &a, &b, z, pool).expect("arity checked above");
            term_expr(term, pool)
        }
        T::Quadratic => quadratic(a[0], a[1], b[0], z, pool),
        _ => connection(t, a[0], a[1], b[0], z, pool),
    }
}
