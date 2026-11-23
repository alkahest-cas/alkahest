//! The shared Sturm–Liouville eigenproblem solver (milestone M3; plan §5 D5,
//! §8 M3/M4).
//!
//! ```text
//! (p X')' + q X + λ w X = 0   on (a, b)
//! ```
//!
//! with separated conditions `αX + βX' = 0` at each end (Dirichlet `β = 0`,
//! Neumann `α = 0`, Robin otherwise), periodic conditions, or — at a singular
//! or infinite end — boundedness / decay.  Separation of variables (M4) calls
//! [`fn@sturm_liouville`] for every separated coordinate, so the result,
//! [`Eigensystem`], carries everything an eigenfunction expansion needs: the
//! eigenvalues as a family in an index `n`, the eigenfunctions, the weight,
//! the squared norms `∫ w X_n²`, whether `λ = 0` or negative eigenvalues occur
//! (with the parameter-sign case analysis as side conditions), and certified
//! enclosures of the first members when the data is numeric.
//!
//! # Method
//!
//! * **Constant coefficients** (`p`, `q`, `w` free of `x`): with
//!   `ν² = (q + λw)/p`, `X = A cos ν(x−a) + B sin ν(x−a)` and the two
//!   conditions give the characteristic determinant
//!
//!   ```text
//!   D(ν) = (α_a α_b + β_a β_b ν²) sin νℓ + ν (α_a β_b − β_a α_b) cos νℓ,   ℓ = b − a,
//!   ```
//!
//!   written without poles.  `D = c·sin νℓ` gives `ν_n = nπ/ℓ`,
//!   `D = c·cos νℓ` gives `ν_n = (n − ½)π/ℓ`; anything else is the
//!   [`RootOfFamily`](super::root_family) `RootOfFamily(D, ν, n, 0)` — for
//!   `X(0) = 0, X'(L) + hX(L) = 0` that is `h sin νL + ν cos νL = 0`.
//!   `ν = 0` (`λ = −q/w`) and imaginary `ν = iκ` (below it) are treated
//!   separately: the first is an eigenvalue exactly when
//!   `α_a(α_b ℓ + β_b) − β_a α_b = 0`, the second cannot occur when
//!   `α_a β_a ≤ 0 ≤ α_b β_b` (the Rayleigh quotient), which is recorded as a
//!   side condition when the signs are symbolic.
//! * **Variable coefficients, regular ends**: `dsolve` solves
//!   `p X'' + p' X' + (q + μ² w) X = 0` (`λ = μ²`), and the same 2×2
//!   determinant of the boundary functionals on its basis is the eigenvalue
//!   condition (closed form when it is `c·sin(κμ)` or `c·cos(κμ)`).
//! * **Singular problems** are recognised from `p`, `q`, `w` and the ends:
//!   Bessel on a disk (`(rR')' − ν²R/r + λrR = 0` on `[0, R₀]`, roots of
//!   `J_ν` as a `RootOfFamily`), Legendre on `[−1, 1]`, Chebyshev on
//!   `[−1, 1]`, Hermite on `ℝ` and Laguerre on `(0, ∞)` (polynomial
//!   eigenfunctions, eigenvalues `n(n+1)`, `n²`, `2n`, `n`).
//!
//! # Verification
//!
//! Every family is checked exactly: the eigenfunction satisfies the ODE
//! identically in the frequency, the left condition identically, and the
//! right condition either at the closed-form eigenvalue (with `n` an integer)
//! or — for a `RootOfFamily` — because its residual is a nonzero multiple of
//! the defining equation, which the root satisfies by definition.  The
//! polynomial families are verified on their first members with the explicit
//! polynomials.  A refuted residual is an error (`E-PDE-153`), never an
//! answer.

use std::collections::HashMap;

use rug::Float;

use crate::ball::{ArbBall, IntervalEval};
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::error::PdeError;
use super::jet::symbols_of;
use super::root_family::{member, root_family_enclosure, root_of_family};
use super::verify::VerificationStatus;

// ---------------------------------------------------------------------------
// Problem description
// ---------------------------------------------------------------------------

/// A condition at one end of the interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EndCondition {
    /// `X = 0`.
    Dirichlet,
    /// `X' = 0`.
    Neumann,
    /// `α X + β X' = 0` (not both zero).  The sign convention is the literal
    /// one: `X'(L) + h X(L) = 0` is `Robin { alpha: h, beta: 1 }` at the
    /// right end, `X'(0) − h X(0) = 0` is `Robin { alpha: −h, beta: 1 }` at
    /// the left.
    Robin {
        /// Coefficient of `X`.
        alpha: ExprId,
        /// Coefficient of `X'`.
        beta: ExprId,
    },
    /// `X` bounded at a singular end (Bessel at `r = 0`, Legendre at `±1`).
    Bounded,
    /// `X` of at most polynomial growth at an infinite end, so that
    /// `∫ w X²` is finite (Hermite, Laguerre).
    Decay,
}

impl EndCondition {
    /// `(α, β)` of a separated regular condition.
    fn coefficients(&self, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
        match *self {
            EndCondition::Dirichlet => Some((pool.integer(1), pool.integer(0))),
            EndCondition::Neumann => Some((pool.integer(0), pool.integer(1))),
            EndCondition::Robin { alpha, beta } => Some((alpha, beta)),
            _ => None,
        }
    }

    /// `"dirichlet"`, `"neumann"`, `"robin"`, `"bounded"` or `"decay"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            EndCondition::Dirichlet => "dirichlet",
            EndCondition::Neumann => "neumann",
            EndCondition::Robin { .. } => "robin",
            EndCondition::Bounded => "bounded",
            EndCondition::Decay => "decay",
        }
    }
}

/// The boundary conditions of an eigenproblem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SlBoundary {
    /// One condition at each end.
    Separated {
        /// At `a`.
        left: EndCondition,
        /// At `b`.
        right: EndCondition,
    },
    /// `X(a) = X(b)`, `X'(a) = X'(b)` (constant coefficients only).
    Periodic,
}

/// `(p X')' + q X + λ w X = 0` on `(a, b)` with boundary conditions.
///
/// `a` / `b` may be `−∞` / `+∞` ([`ExprPool::pos_infinity`]) for the
/// Hermite and Laguerre problems.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SturmLiouvilleProblem {
    /// The independent variable.
    pub var: ExprId,
    /// `p(x)`.
    pub p: ExprId,
    /// `q(x)`.
    pub q: ExprId,
    /// The weight `w(x)`.
    pub w: ExprId,
    /// Left end.
    pub a: ExprId,
    /// Right end.
    pub b: ExprId,
    /// The conditions.
    pub boundary: SlBoundary,
}

fn is_symbol(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Symbol { .. }))
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, pool.pow(b, pool.integer(-1))])
}

pub(crate) fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::engine::simplify(e, pool).value
}

pub(crate) fn subs1(e: ExprId, v: ExprId, by: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(v, by);
    subs(e, &m, pool)
}

pub(crate) fn d(e: ExprId, v: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    crate::diff::diff(e, v, pool)
        .map(|r| r.value)
        .map_err(|err| PdeError::SlUnsupported(format!("cannot differentiate: {err}")))
}

fn is_infinite(e: ExprId, pool: &ExprPool) -> bool {
    let oo = pool.pos_infinity();
    e == oo || e == neg(oo, pool) || simp(e, pool) == simp(neg(oo, pool), pool)
}

fn is_zero_literal(e: ExprId, pool: &ExprPool) -> bool {
    e == pool.integer(0)
}

/// Exact zero test that never certifies by sampling.
fn proved_zero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simp(e, pool);
    if is_zero_literal(s, pool) {
        return true;
    }
    zero_test(s, pool, None).status == ZeroStatus::ProvedZero
}

/// Sign of an expression that does not depend on the variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sgn {
    Pos,
    Neg,
    Zero,
    Unknown,
}

/// Structural / ball-arithmetic sign: numeric expressions through Arb;
/// products and powers of positive-domain symbols and positive constants are
/// positive; sums of positives are positive.
pub(crate) fn sign(e: ExprId, pool: &ExprPool) -> Sgn {
    let e = simp(e, pool);
    if is_zero_literal(e, pool) {
        return Sgn::Zero;
    }
    let free = symbols_of(e, pool)
        .into_iter()
        .any(|s| !is_named_constant(s, pool));
    if !free {
        if let Some(b) = IntervalEval::new(128).eval(e, pool) {
            if b.is_exact() && b.mid == 0 {
                return Sgn::Zero;
            }
            if b.lo().is_finite() && b.lo() > 0 {
                return Sgn::Pos;
            }
            if b.hi().is_finite() && b.hi() < 0 {
                return Sgn::Neg;
            }
        }
        return Sgn::Unknown;
    }
    structural_sign(e, pool)
}

fn structural_sign(e: ExprId, pool: &ExprPool) -> Sgn {
    match pool.get(e) {
        ExprData::Integer(n) => match n.0.cmp0() {
            std::cmp::Ordering::Greater => Sgn::Pos,
            std::cmp::Ordering::Less => Sgn::Neg,
            std::cmp::Ordering::Equal => Sgn::Zero,
        },
        ExprData::Rational(r) => match r.0.cmp0() {
            std::cmp::Ordering::Greater => Sgn::Pos,
            std::cmp::Ordering::Less => Sgn::Neg,
            std::cmp::Ordering::Equal => Sgn::Zero,
        },
        ExprData::Symbol { domain, .. } => {
            if is_named_constant(e, pool) {
                return sign(e, pool);
            }
            match domain {
                Domain::Positive => Sgn::Pos,
                _ => Sgn::Unknown,
            }
        }
        ExprData::Mul(xs) => {
            let mut neg_count = 0;
            for x in xs {
                match structural_sign(x, pool) {
                    Sgn::Pos => {}
                    Sgn::Neg => neg_count += 1,
                    Sgn::Zero => return Sgn::Zero,
                    Sgn::Unknown => return Sgn::Unknown,
                }
            }
            if neg_count % 2 == 0 {
                Sgn::Pos
            } else {
                Sgn::Neg
            }
        }
        ExprData::Add(xs) => {
            let signs: Vec<Sgn> = xs.iter().map(|&x| structural_sign(x, pool)).collect();
            if signs.iter().all(|s| matches!(s, Sgn::Pos)) {
                Sgn::Pos
            } else if signs.iter().all(|s| matches!(s, Sgn::Neg)) {
                Sgn::Neg
            } else {
                Sgn::Unknown
            }
        }
        ExprData::Pow { base, exp } => {
            let bs = structural_sign(base, pool);
            match bs {
                Sgn::Pos => Sgn::Pos,
                Sgn::Neg => {
                    let even = pool.with(exp, |d| match d {
                        ExprData::Integer(n) => Some(n.0.is_even()),
                        _ => None,
                    });
                    match even {
                        Some(true) => Sgn::Pos,
                        Some(false) => Sgn::Neg,
                        None => Sgn::Unknown,
                    }
                }
                _ => Sgn::Unknown,
            }
        }
        ExprData::Func { name, args } if args.len() == 1 => match name.as_str() {
            "exp" | "cosh" => Sgn::Pos,
            "sqrt" if structural_sign(args[0], pool) == Sgn::Pos => Sgn::Pos,
            _ => Sgn::Unknown,
        },
        _ => Sgn::Unknown,
    }
}

fn is_named_constant(s: ExprId, pool: &ExprPool) -> bool {
    pool.with(s, |d| match d {
        ExprData::Symbol { name, .. } => crate::eval::symbols::constant_value_f64(name).is_some(),
        _ => false,
    })
}

/// A fresh symbol named `base` (or `base1`, `base2`, …) that does not occur
/// in `avoid`.
pub(crate) fn fresh_symbol(
    base: &str,
    domain: Domain,
    avoid: &[ExprId],
    pool: &ExprPool,
) -> ExprId {
    let mut taken: Vec<String> = Vec::new();
    for &e in avoid {
        for s in symbols_of(e, pool) {
            if let ExprData::Symbol { name, .. } = pool.get(s) {
                taken.push(name);
            }
        }
    }
    let mut name = base.to_string();
    let mut k = 0;
    while taken.contains(&name) {
        k += 1;
        name = format!("{base}{k}");
    }
    pool.symbol(name, domain)
}

impl SturmLiouvilleProblem {
    /// `(p X')' + q X + λ w X = 0` on `(a, b)`.
    ///
    /// # Errors
    ///
    /// `E-PDE-151` for a Robin condition with `α = β = 0`, or a regular
    /// condition at an infinite end; `E-PDE-010` when `var` is not a symbol.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        var: ExprId,
        p: ExprId,
        q: ExprId,
        w: ExprId,
        a: ExprId,
        b: ExprId,
        boundary: SlBoundary,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        if !is_symbol(var, pool) {
            return Err(PdeError::InvalidDomain(format!(
                "the eigenproblem variable {} is not a symbol",
                pool.display(var)
            )));
        }
        if a == b {
            return Err(PdeError::InvalidDomain("empty interval".into()));
        }
        if let SlBoundary::Separated { left, right } = boundary {
            for (end, c) in [(a, left), (b, right)] {
                if let EndCondition::Robin { alpha, beta } = c {
                    if proved_zero(alpha, pool) && proved_zero(beta, pool) {
                        return Err(PdeError::SlInvalidBoundary(
                            "a Robin condition αX + βX' = 0 needs α and β not both zero".into(),
                        ));
                    }
                }
                if is_infinite(end, pool) && c.coefficients(pool).is_some() {
                    return Err(PdeError::SlInvalidBoundary(format!(
                        "a {} condition at an infinite end; use Decay",
                        c.as_str()
                    )));
                }
            }
        }
        Ok(SturmLiouvilleProblem {
            var,
            p,
            q,
            w,
            a,
            b,
            boundary,
        })
    }

    /// `X'' + λ X = 0` on `(a, b)` — the problem separation of variables
    /// produces for the heat, wave and Laplace equations in Cartesian
    /// coordinates.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn interval(
        var: ExprId,
        a: ExprId,
        b: ExprId,
        boundary: SlBoundary,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        let one = pool.integer(1);
        Self::new(var, one, pool.integer(0), one, a, b, boundary, pool)
    }

    /// Bessel's problem on a disk of radius `radius`:
    /// `(r R')' − (ν²/r) R + λ r R = 0`, `R` bounded at `0`, `outer` at
    /// `r = radius`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn bessel(
        r: ExprId,
        order: ExprId,
        radius: ExprId,
        outer: EndCondition,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        let q = neg(div(pool.pow(order, pool.integer(2)), r, pool), pool);
        Self::new(
            r,
            r,
            q,
            r,
            pool.integer(0),
            radius,
            SlBoundary::Separated {
                left: EndCondition::Bounded,
                right: outer,
            },
            pool,
        )
    }

    /// Legendre's problem `((1 − x²) X')' + λ X = 0` on `[−1, 1]`, bounded at
    /// both ends.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn legendre(x: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        let p = sub(pool.integer(1), pool.pow(x, pool.integer(2)), pool);
        Self::new(
            x,
            p,
            pool.integer(0),
            pool.integer(1),
            pool.integer(-1),
            pool.integer(1),
            SlBoundary::Separated {
                left: EndCondition::Bounded,
                right: EndCondition::Bounded,
            },
            pool,
        )
    }

    /// Hermite's problem `(e^{−x²} X')' + λ e^{−x²} X = 0` on `ℝ`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn hermite(x: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        let g = pool.func("exp", vec![neg(pool.pow(x, pool.integer(2)), pool)]);
        let oo = pool.pos_infinity();
        Self::new(
            x,
            g,
            pool.integer(0),
            g,
            neg(oo, pool),
            oo,
            SlBoundary::Separated {
                left: EndCondition::Decay,
                right: EndCondition::Decay,
            },
            pool,
        )
    }

    /// Laguerre's problem `(x e^{−x} X')' + λ e^{−x} X = 0` on `(0, ∞)`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn laguerre(x: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        let e = pool.func("exp", vec![neg(x, pool)]);
        Self::new(
            x,
            pool.mul(vec![x, e]),
            pool.integer(0),
            e,
            pool.integer(0),
            pool.pos_infinity(),
            SlBoundary::Separated {
                left: EndCondition::Bounded,
                right: EndCondition::Decay,
            },
            pool,
        )
    }

    /// Chebyshev's problem `(√(1−x²) X')' + λ X/√(1−x²) = 0` on `[−1, 1]`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn chebyshev(x: ExprId, pool: &ExprPool) -> Result<Self, PdeError> {
        let s = pool.func(
            "sqrt",
            vec![sub(pool.integer(1), pool.pow(x, pool.integer(2)), pool)],
        );
        Self::new(
            x,
            s,
            pool.integer(0),
            pool.pow(s, pool.integer(-1)),
            pool.integer(-1),
            pool.integer(1),
            SlBoundary::Separated {
                left: EndCondition::Bounded,
                right: EndCondition::Bounded,
            },
            pool,
        )
    }

    /// Read `A X'' + B X' + (C + λ D) X = 0` off an equation in the applied
    /// unknown `X(x)` (and its `Derivative`s) and the eigenvalue symbol
    /// `lam`, and bring it to Sturm–Liouville form with the integrating
    /// factor `m = exp(∫ B/A)`: `p = m`, `q = mC/A`, `w = mD/A` (or
    /// `p = A`, `q = C`, `w = D` when `B = A'` already).
    ///
    /// # Errors
    ///
    /// `E-PDE-150` when the equation is not linear of second order in `X`
    /// and affine in `λ`, or the integrating factor does not close;
    /// `E-PDE-001` for a malformed unknown.
    #[allow(clippy::too_many_arguments)]
    pub fn from_ode(
        equation: ExprId,
        unknown: ExprId,
        lam: ExprId,
        a: ExprId,
        b: ExprId,
        boundary: SlBoundary,
        pool: &ExprPool,
    ) -> Result<Self, PdeError> {
        let v = super::jet::decode_unknown(unknown, pool)?;
        if v.args.len() != 1 {
            return Err(PdeError::SlUnsupported(format!(
                "{} is not a function of one variable",
                pool.display(unknown)
            )));
        }
        let x = v.args[0];
        let ys: Vec<ExprId> = (0..3)
            .map(|k| fresh_symbol(&format!("__sl_y{k}"), Domain::Real, &[equation], pool))
            .collect();
        let mut e = equation;
        for k in (0..3u32).rev() {
            let node = crate::diff::undefined::build(unknown, &[k], pool);
            e = replace_node(e, node, ys[k as usize], pool);
        }
        if crate::diff::undefined::has_undefined_function(e, pool)
            && mentions_name(e, &v.name, pool)
        {
            return Err(PdeError::SlUnsupported(format!(
                "{} occurs other than as X, X', X'' (order above two, or at another argument)",
                pool.display(unknown)
            )));
        }
        // Linear in (y, y', y''): every second derivative vanishes and the
        // remainder at y = 0 is zero.
        let mut coeffs = Vec::new();
        for &y in &ys {
            let c = simp(d(e, y, pool)?, pool);
            for &z in &ys {
                if !proved_zero(d(c, z, pool)?, pool) {
                    return Err(PdeError::SlUnsupported(
                        "the eigenvalue equation is not linear in X".into(),
                    ));
                }
            }
            coeffs.push(c);
        }
        let mut z0 = HashMap::new();
        for &y in &ys {
            z0.insert(y, pool.integer(0));
        }
        if !proved_zero(subs(e, &z0, pool), pool) {
            return Err(PdeError::SlUnsupported(
                "the eigenvalue equation is not homogeneous in X".into(),
            ));
        }
        let (c0, c1, c2) = (coeffs[0], coeffs[1], coeffs[2]);
        if proved_zero(c2, pool) {
            return Err(PdeError::SlUnsupported(
                "the eigenvalue equation is not of second order".into(),
            ));
        }
        for c in [c1, c2] {
            if mentions_var(c, lam, pool) {
                return Err(PdeError::SlUnsupported(
                    "λ multiplies a derivative of X; the problem is not of Sturm–Liouville type"
                        .into(),
                ));
            }
        }
        let dd = simp(d(c0, lam, pool)?, pool);
        if mentions_var(dd, lam, pool) || proved_zero(dd, pool) {
            return Err(PdeError::SlUnsupported(
                "the equation must be affine in λ with a nonzero λ-coefficient".into(),
            ));
        }
        let cc = simp(subs1(c0, lam, pool.integer(0), pool), pool);
        let (p, q, w) = if proved_zero(sub(c1, d(c2, x, pool)?, pool), pool) {
            (c2, cc, dd)
        } else {
            let ratio = simp(div(c1, c2, pool), pool);
            let integral = crate::integrate::integrate(ratio, x, pool).map_err(|err| {
                PdeError::SlUnsupported(format!(
                    "the integrating factor exp(∫ B/A) does not close: {err}"
                ))
            })?;
            let m = simp(pool.func("exp", vec![integral.value]), pool);
            (
                m,
                simp(pool.mul(vec![m, cc, pool.pow(c2, pool.integer(-1))]), pool),
                simp(pool.mul(vec![m, dd, pool.pow(c2, pool.integer(-1))]), pool),
            )
        };
        Self::new(x, p, q, w, a, b, boundary, pool)
    }
}

fn mentions_name(e: ExprId, name: &str, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = crate::diff::undefined::view(n, pool) {
            if v.name == name {
                return true;
            }
            stack.extend(v.args);
            continue;
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    false
}

/// Replace every occurrence of the node `old` by `new` (structural).
pub(crate) fn replace_node(e: ExprId, old: ExprId, new: ExprId, pool: &ExprPool) -> ExprId {
    fn go(
        e: ExprId,
        old: ExprId,
        new: ExprId,
        pool: &ExprPool,
        memo: &mut HashMap<ExprId, ExprId>,
    ) -> ExprId {
        if e == old {
            return new;
        }
        if let Some(&r) = memo.get(&e) {
            return r;
        }
        let r = match pool.get(e) {
            ExprData::Add(xs) => {
                pool.add(xs.iter().map(|&x| go(x, old, new, pool, memo)).collect())
            }
            ExprData::Mul(xs) => {
                pool.mul(xs.iter().map(|&x| go(x, old, new, pool, memo)).collect())
            }
            ExprData::Pow { base, exp } => pool.pow(
                go(base, old, new, pool, memo),
                go(exp, old, new, pool, memo),
            ),
            ExprData::Func { name, args } => pool.func(
                name,
                args.iter().map(|&x| go(x, old, new, pool, memo)).collect(),
            ),
            _ => e,
        };
        memo.insert(e, r);
        r
    }
    go(e, old, new, pool, &mut HashMap::new())
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// A certified enclosure `lo ≤ λ_n ≤ hi` of one eigenvalue.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EigenEnclosure {
    /// The index `n`.
    pub index: i64,
    /// Lower end (rounded down).
    pub lo: Float,
    /// Upper end (rounded up).
    pub hi: Float,
}

/// How an eigen-family was verified.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EigenVerification {
    /// Overall: `ExactlyVerified` only when every check is a proof.
    pub status: VerificationStatus,
    /// The ODE residual, identically in the frequency.
    pub ode: ZeroStatus,
    /// The left condition.
    pub left: ZeroStatus,
    /// The right condition.
    pub right: ZeroStatus,
    /// `"closed_form"` (residuals vanish at the closed-form eigenvalue),
    /// `"defining_equation"` (the right residual is a nonzero multiple of the
    /// `RootOfFamily` equation), `"members"` (checked exactly on the members
    /// listed in [`Self::members`]), `"structural"` (a singular end whose
    /// condition holds by the regularity of the eigenfunction).
    pub method: &'static str,
    /// Indices checked one by one, when [`Self::method`] is `"members"`.
    pub members: Vec<i64>,
}

/// One family of eigenpairs `(λ_n, X_n)`, `n = first, first + 1, …`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EigenFamily {
    /// The index symbol `n` (an integer-domain symbol).
    pub index: ExprId,
    /// The first value of the index.
    pub first: i64,
    /// The frequency symbol `ν` of the defining equation (`λ` as a function
    /// of it is [`Self::eigenvalue`] with `ν → ν_n`).
    pub frequency_var: ExprId,
    /// `ν_n`: closed form in `n`, or a `RootOfFamily(F, ν, n, 0)` node.
    pub frequency: ExprId,
    /// `λ_n` in terms of `n` (through [`Self::frequency`]).
    pub eigenvalue: ExprId,
    /// `λ` as a function of [`Self::frequency_var`].
    pub eigenvalue_of_frequency: ExprId,
    /// The eigenfunctions of `λ_n` (one; two for periodic problems).
    pub eigenfunctions: Vec<ExprId>,
    /// The same as functions of [`Self::frequency_var`] (they satisfy the
    /// ODE identically in it).
    pub eigenfunctions_of_frequency: Vec<ExprId>,
    /// `∫ w X_n²` per eigenfunction, when it closes.
    pub norms_squared: Vec<Option<ExprId>>,
    /// Whether `ν_n` is in closed form.
    pub closed_form: bool,
    /// The pole-free defining equation `F(ν) = 0` whose positive roots are
    /// the `ν_n` (for a closed form, the equation it solves); `None` for the
    /// polynomial families, whose index is the degree.
    pub defining_equation: Option<ExprId>,
    /// Certified enclosures of the first eigenvalues (empty when a
    /// parameter is symbolic; see [`Self::enclosures`]).
    pub enclosures: Vec<EigenEnclosure>,
    /// The exact checks.
    pub verification: EigenVerification,
}

/// An isolated eigenpair (the `ν = 0` mode, or one below it).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EigenMode {
    /// `λ`.
    pub eigenvalue: ExprId,
    /// `X`.
    pub eigenfunction: ExprId,
    /// `∫ w X²`.
    pub norm_squared: Option<ExprId>,
    /// `None` when unconditional; otherwise the predicate under which this
    /// is an eigenpair (`h = 0`, …).
    pub condition: Option<ExprId>,
    /// Exact check status of the ODE and both conditions.
    pub verification: VerificationStatus,
}

/// What is known about eigenvalues below the threshold `λ₀` (the `ν = 0`
/// value; `0` when `q = 0`).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LowerSpectrum {
    /// Proved: no eigenvalue below `λ₀`.
    None,
    /// None below `λ₀` provided `condition` (a predicate) holds; otherwise
    /// the eigenvalues there are `λ₀ − (p/w) κ_n²` for the positive roots of
    /// `family`'s defining equation (finitely many — at most two).
    Conditional {
        /// The sign condition (the Rayleigh-quotient hypothesis).
        condition: ExprId,
        /// The family of eigenvalues below `λ₀` when it fails.
        family: Option<Box<EigenFamily>>,
    },
    /// The hypothesis is refuted: eigenvalues below `λ₀` are the members of
    /// `family` (finitely many; `count` when certified).
    Present {
        /// The family `λ₀ − (p/w) κ_n²`.
        family: Box<EigenFamily>,
        /// The number of such eigenvalues, when certified.
        count: Option<usize>,
    },
}

/// The solution of a Sturm–Liouville problem: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Eigensystem {
    /// `"regular_constant"`, `"regular"`, `"periodic"`, `"bessel"`,
    /// `"legendre"`, `"chebyshev"`, `"hermite"` or `"laguerre"`.
    pub kind: &'static str,
    /// The variable.
    pub var: ExprId,
    /// The weight `w`.
    pub weight: ExprId,
    /// The interval.
    pub interval: (ExprId, ExprId),
    /// `λ₀`: the eigenvalue at which the frequency vanishes (`−q/w` for
    /// constant coefficients, `0` otherwise).
    pub threshold: ExprId,
    /// The eigen-families above `λ₀`.
    pub families: Vec<EigenFamily>,
    /// Isolated eigenpairs: `λ₀` itself when it is an eigenvalue (possibly
    /// conditionally).
    pub modes: Vec<EigenMode>,
    /// Eigenvalues below `λ₀`.
    pub lower: LowerSpectrum,
    /// Conditions (predicates) under which the result is complete and
    /// correct: positivity of `p` and `w`, nonvanishing coefficients, …
    pub side_conditions: Vec<ExprId>,
    /// Prose for the cases the side conditions exclude.
    pub notes: Vec<String>,
}

impl EigenFamily {
    /// The eigenvalue `λ_k` (a concrete member).
    pub fn eigenvalue_at(&self, k: i64, pool: &ExprPool) -> Result<ExprId, PdeError> {
        let nu = self.frequency_at(k, pool)?;
        Ok(simp(
            subs1(self.eigenvalue_of_frequency, self.frequency_var, nu, pool),
            pool,
        ))
    }

    /// `ν_k`.
    pub fn frequency_at(&self, k: i64, pool: &ExprPool) -> Result<ExprId, PdeError> {
        if self.closed_form {
            Ok(simp(
                subs1(self.frequency, self.index, pool.integer(k), pool),
                pool,
            ))
        } else {
            member(self.frequency, k, pool)
        }
    }

    /// The `k`-th eigenfunction(s).
    pub fn eigenfunctions_at(&self, k: i64, pool: &ExprPool) -> Result<Vec<ExprId>, PdeError> {
        let nu = self.frequency_at(k, pool)?;
        Ok(self
            .eigenfunctions_of_frequency
            .iter()
            .map(|&f| {
                let e = subs1(f, self.frequency_var, nu, pool);
                let e = crate::orthopoly::expand_orthopoly(e, pool).unwrap_or(e);
                simp(e, pool)
            })
            .collect())
    }

    /// Certified enclosures of `λ_first … λ_{first+count−1}`, after
    /// substituting `params` (parameter → number); see
    /// [`family_enclosures`].
    ///
    /// # Errors
    ///
    /// `E-PDE-031` when a root cannot be certified (or a parameter is still
    /// symbolic).
    pub fn enclosures(
        &self,
        count: usize,
        params: &HashMap<ExprId, ExprId>,
        bits: u32,
        pool: &ExprPool,
    ) -> Result<Vec<EigenEnclosure>, PdeError> {
        family_enclosures(
            FamilyRef {
                index: self.index,
                first: self.first,
                frequency_var: self.frequency_var,
                frequency: self.frequency,
                eigenvalue_of_frequency: self.eigenvalue_of_frequency,
                closed_form: self.closed_form,
            },
            count,
            params,
            bits,
            pool,
        )
    }
}

/// The data of an [`EigenFamily`] that its enclosures depend on — what a
/// caller holding the family's expressions (e.g. from Python) passes to
/// [`family_enclosures`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilyRef {
    /// [`EigenFamily::index`].
    pub index: ExprId,
    /// [`EigenFamily::first`].
    pub first: i64,
    /// [`EigenFamily::frequency_var`].
    pub frequency_var: ExprId,
    /// [`EigenFamily::frequency`].
    pub frequency: ExprId,
    /// [`EigenFamily::eigenvalue_of_frequency`].
    pub eigenvalue_of_frequency: ExprId,
    /// [`EigenFamily::closed_form`].
    pub closed_form: bool,
}

/// Certified enclosures of the first `count` eigenvalues of a family.
///
/// # Errors
///
/// `E-PDE-031` when a root cannot be certified (or a parameter is still
/// symbolic).
pub fn family_enclosures(
    fam: FamilyRef,
    count: usize,
    params: &HashMap<ExprId, ExprId>,
    bits: u32,
    pool: &ExprPool,
) -> Result<Vec<EigenEnclosure>, PdeError> {
    let mut out = Vec::new();
    let prec = bits.max(53);
    let lam_expr = simp(subs(fam.eigenvalue_of_frequency, params, pool), pool);
    for j in 0..count as i64 {
        let k = fam.first + j;
        let nu_ball = if fam.closed_form {
            let e = simp(
                subs1(
                    subs(fam.frequency, params, pool),
                    fam.index,
                    pool.integer(k),
                    pool,
                ),
                pool,
            );
            IntervalEval::new(prec + 32).eval(e, pool)
        } else {
            let m = member(subs(fam.frequency, params, pool), k.max(1), pool)?;
            let (lo, hi) = root_family_enclosure(m, prec, pool)?;
            let mid = Float::with_val(prec + 32, &lo + &hi) / 2u32;
            let rad = Float::with_val_round(prec + 32, &hi - &mid, rug::float::Round::Up).0;
            Some(ArbBall {
                mid,
                rad,
                prec: prec + 32,
            })
        };
        let ball = nu_ball.and_then(|b| {
            let mut ie = IntervalEval::new(prec + 32);
            ie.bind(fam.frequency_var, b);
            ie.bind(fam.index, ArbBall::from_f64(k as f64, prec + 32));
            ie.eval(lam_expr, pool)
        });
        let b = ball.ok_or_else(|| {
            PdeError::RootFamilyNotCertified(format!(
                "λ_{k} = {} has no ball enclosure (a parameter is symbolic?)",
                pool.display(fam.eigenvalue_of_frequency)
            ))
        })?;
        if !(b.lo().is_finite() && b.hi().is_finite()) {
            return Err(PdeError::RootFamilyNotCertified(format!(
                "λ_{k}: non-finite enclosure"
            )));
        }
        out.push(EigenEnclosure {
            index: k,
            lo: b.lo(),
            hi: b.hi(),
        });
    }
    Ok(out)
}

/// Options for [`sturm_liouville_with`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SlOptions {
    /// How many eigenvalues to enclose when the problem is numeric.
    pub enclosures: usize,
    /// Precision of the enclosures, in bits.
    pub bits: u32,
    /// How many members of a polynomial family to verify one by one.
    pub check_members: usize,
    /// The name of the index symbol.
    pub index_name: String,
}

impl Default for SlOptions {
    fn default() -> Self {
        SlOptions {
            enclosures: 5,
            bits: 64,
            check_members: 6,
            index_name: "n".into(),
        }
    }
}

impl SlOptions {
    /// Default options with `count` enclosures.
    pub fn with_enclosures(mut self, count: usize) -> Self {
        self.enclosures = count;
        self
    }

    /// Default options with enclosures `bits` wide.
    pub fn with_bits(mut self, bits: u32) -> Self {
        self.bits = bits;
        self
    }

    /// Default options with the index symbol named `name`.
    pub fn with_index_name(mut self, name: &str) -> Self {
        self.index_name = name.to_string();
        self
    }
}

/// Solve a Sturm–Liouville eigenproblem with default options.
///
/// # Errors
///
/// See [`sturm_liouville_with`].
pub fn sturm_liouville(
    problem: &SturmLiouvilleProblem,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    sturm_liouville_with(problem, &SlOptions::default(), pool)
}

/// Solve a Sturm–Liouville eigenproblem.
///
/// # Errors
///
/// * `E-PDE-150` — outside the implemented classes;
/// * `E-PDE-151` — a malformed end condition (a singular-end condition at a
///   regular end of an unrecognised problem);
/// * `E-PDE-152` — the eigenvalue condition vanishes identically;
/// * `E-PDE-153` — an eigenpair was refuted by its exact check.
pub fn sturm_liouville_with(
    problem: &SturmLiouvilleProblem,
    opts: &SlOptions,
    pool: &ExprPool,
) -> Result<Eigensystem, PdeError> {
    let x = problem.var;
    let all = [problem.p, problem.q, problem.w, problem.a, problem.b, x];
    let n = fresh_symbol(&opts.index_name, Domain::Integer, &all, pool);
    if let Some(sys) = singular::recognise(problem, n, opts, pool)? {
        return Ok(sys);
    }
    let regular_ends = match problem.boundary {
        SlBoundary::Separated { left, right } => {
            left.coefficients(pool).is_some() && right.coefficients(pool).is_some()
        }
        SlBoundary::Periodic => true,
    };
    if !regular_ends {
        return Err(PdeError::SlInvalidBoundary(
            "a bounded / decay condition is only understood for the recognised singular problems \
             (Bessel, Legendre, Chebyshev, Hermite, Laguerre)"
                .into(),
        ));
    }
    if is_infinite(problem.a, pool) || is_infinite(problem.b, pool) {
        return Err(PdeError::SlUnsupported(
            "a regular problem needs a finite interval".into(),
        ));
    }
    let p = simp(problem.p, pool);
    let q = simp(problem.q, pool);
    let w = simp(problem.w, pool);
    let constant =
        !mentions_var(p, x, pool) && !mentions_var(q, x, pool) && !mentions_var(w, x, pool);
    let mut sys = if constant {
        regular::constant_coefficients(problem, p, q, w, n, pool)?
    } else {
        regular::variable_coefficients(problem, p, q, w, n, pool)?
    };
    fill_enclosures(&mut sys, opts, pool);
    Ok(sys)
}

/// Enclose the first eigenvalues of every family whose data is numeric.
fn fill_enclosures(sys: &mut Eigensystem, opts: &SlOptions, pool: &ExprPool) {
    if opts.enclosures == 0 {
        return;
    }
    for fam in &mut sys.families {
        let free = symbols_of(fam.eigenvalue, pool)
            .into_iter()
            .chain(symbols_of(fam.frequency, pool))
            .any(|s| {
                s != fam.index
                    && s != fam.frequency_var
                    && !is_named_constant(s, pool)
                    && !bound_in_family(s, fam.frequency, pool)
            });
        if free {
            continue;
        }
        if let Ok(v) = fam.enclosures(opts.enclosures, &HashMap::new(), opts.bits, pool) {
            fam.enclosures = v;
        }
    }
}

fn bound_in_family(s: ExprId, fam: ExprId, pool: &ExprPool) -> bool {
    super::root_family::root_family_parts(fam, pool).is_some_and(|(_, lam, _, _)| lam == s)
}

/// Exact check of an eigenfunction: the ODE residual `(pX')' + qX + λwX`.
fn ode_residual(
    x: ExprId,
    p: ExprId,
    q: ExprId,
    w: ExprId,
    lam: ExprId,
    xf: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let dx = d(xf, x, pool)?;
    let flux = d(pool.mul(vec![p, dx]), x, pool)?;
    Ok(pool.add(vec![
        flux,
        pool.mul(vec![q, xf]),
        pool.mul(vec![lam, w, xf]),
    ]))
}

fn status_of(z: ZeroStatus) -> VerificationStatus {
    match z {
        ZeroStatus::ProvedZero => VerificationStatus::ExactlyVerified,
        ZeroStatus::ProvedNonzero => VerificationStatus::Refuted,
        ZeroStatus::ProbablyZero => VerificationStatus::NumericallyChecked,
        _ => VerificationStatus::Unverified,
    }
}

fn combine(statuses: &[ZeroStatus]) -> VerificationStatus {
    if statuses.iter().all(|s| *s == ZeroStatus::ProvedZero) {
        VerificationStatus::ExactlyVerified
    } else if statuses.contains(&ZeroStatus::ProvedNonzero) {
        VerificationStatus::Refuted
    } else if statuses
        .iter()
        .all(|s| matches!(s, ZeroStatus::ProvedZero | ZeroStatus::ProbablyZero))
    {
        VerificationStatus::NumericallyChecked
    } else {
        VerificationStatus::Unverified
    }
}

/// Zero status of an exact residual: `simplify` first, then `zero_test`.
fn zstatus(e: ExprId, pool: &ExprPool) -> ZeroStatus {
    let s = simp(e, pool);
    if is_zero_literal(s, pool) {
        return ZeroStatus::ProvedZero;
    }
    let st = zero_test(s, pool, None).status;
    if st == ZeroStatus::ProvedZero {
        return st;
    }
    // Bessel residuals: the derivative rules leave J_{ν±k}; the three-term
    // recurrence reduces them to two consecutive orders, after which the
    // residual is rational in independent atoms.
    if let Some(r) = bessel_recurrence_normal(s, pool) {
        let r = simp(r, pool);
        if is_zero_literal(r, pool) || zero_test(r, pool, None).status == ZeroStatus::ProvedZero {
            return ZeroStatus::ProvedZero;
        }
    }
    st
}

/// Rewrite every `bessel_j` / `bessel_i` (and `bessel_j0`, `bessel_j1`) of one
/// argument `z` whose orders differ by integers in terms of the two lowest
/// orders `J_b`, `J_{b+1}`, by `J_{m+1} = (2m/z) J_m − J_{m−1}` and
/// `I_{m+1} = I_{m−1} − (2m/z) I_m`.  `None` when there is nothing to do.
fn bessel_recurrence_normal(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // (kind, order, arg, node)
    let mut nodes: Vec<(char, ExprId, ExprId, ExprId)> = Vec::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Func { name, args } = pool.get(n) {
            let hit = match (name.as_str(), args.len()) {
                ("bessel_j", 2) => Some(('J', args[0], args[1])),
                ("bessel_i", 2) => Some(('I', args[0], args[1])),
                ("bessel_j0", 1) => Some(('J', pool.integer(0), args[0])),
                ("bessel_j1", 1) => Some(('J', pool.integer(1), args[0])),
                _ => None,
            };
            if let Some((k, o, z)) = hit {
                nodes.push((k, o, z, n));
                continue;
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    if nodes.len() < 3 {
        return None;
    }
    let mut out = e;
    let mut changed = false;
    let groups: Vec<(char, ExprId)> = {
        let mut g: Vec<(char, ExprId)> = nodes.iter().map(|&(k, _, z, _)| (k, z)).collect();
        g.sort();
        g.dedup();
        g
    };
    for (kind, z) in groups {
        let members: Vec<(ExprId, ExprId)> = nodes
            .iter()
            .filter(|&&(k, _, zz, _)| k == kind && zz == z)
            .map(|&(_, o, _, n)| (o, n))
            .collect();
        let base = members[0].0;
        // integer offsets from the first order
        let mut offs = Vec::new();
        for &(o, n) in &members {
            let dlt = simp(sub(o, base, pool), pool);
            let k = pool.with(dlt, |d| match d {
                ExprData::Integer(i) => i.0.to_i64(),
                _ => None,
            })?;
            offs.push((k, n));
        }
        let lo = offs.iter().map(|&(k, _)| k).min()?;
        let hi = offs.iter().map(|&(k, _)| k).max()?;
        if hi - lo < 2 || hi - lo > 40 {
            continue;
        }
        let ord = |k: i64| simp(pool.add(vec![base, pool.integer(k)]), pool);
        let name = if kind == 'J' { "bessel_j" } else { "bessel_i" };
        let mut val: HashMap<i64, ExprId> = HashMap::new();
        val.insert(lo, pool.func(name, vec![ord(lo), z]));
        val.insert(lo + 1, pool.func(name, vec![ord(lo + 1), z]));
        for m in lo + 1..hi {
            let two_m_over_z =
                pool.mul(vec![pool.integer(2), ord(m), pool.pow(z, pool.integer(-1))]);
            let next = if kind == 'J' {
                sub(pool.mul(vec![two_m_over_z, val[&m]]), val[&(m - 1)], pool)
            } else {
                sub(val[&(m - 1)], pool.mul(vec![two_m_over_z, val[&m]]), pool)
            };
            val.insert(m + 1, next);
        }
        for &(k, n) in &offs {
            out = replace_node(out, n, val[&k], pool);
            changed = true;
        }
    }
    changed.then_some(out)
}

mod regular;
mod singular;

#[cfg(test)]
mod tests;
