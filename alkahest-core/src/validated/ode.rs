//! Validated (rigorous) integration of ODE initial value problems.
//!
//! [`verified_ode`] encloses the solution of
//!
//! ```text
//! y'(t) = f(t, y(t)),     y(t₀) ∈ Y₀ ⊂ ℝⁿ,     t ∈ [t₀, t₁]
//! ```
//!
//! in a sequence of boxes that are **guaranteed** to contain the true
//! solution — every solution, when `Y₀` is a box rather than a point.  It is
//! the interval Taylor series method of Moore, with Lohner's QR
//! coordinate-change against the wrapping effect, as described in
//! Nedialkov, Jackson & Corliss, *Validated solutions of initial value
//! problems for ordinary differential equations* (Appl. Math. Comput. 105,
//! 1999).  One step from `t_j` to `t_{j+1} = t_j + h` is:
//!
//! 1. **Existence and an a-priori enclosure (Picard–Lindelöf).**  Find a box
//!    `B` with `Y_j + [0,h]·f([t_j,t_{j+1}], B) ⊂ int B`.  The Picard operator
//!    then maps continuous paths in `B` to continuous paths in `B`, and `f` is
//!    Lipschitz on `B` (its Jacobian is enclosed by a finite ball), so the
//!    solution through any `y* ∈ Y_j` exists on the whole step and stays in
//!    `B`.  If no such `B` is found the step is halved — and eventually the
//!    integration is **refused** (`E-VALIDATED-006`), which is what happens on
//!    approach to a finite-time blow-up such as `y' = y²`.
//! 2. **Tightening the enclosure.**  With existence known, Taylor's theorem
//!    with Lagrange remainder gives, for every `s ∈ [0,h]`,
//!    `y(t_j+s) ∈ Σ_{k≤p} [0,h]^k F_k(Y_j) + [0,h]^{p+1} F_{p+1}([t_j,t_{j+1}], B)`
//!    where `F_k` is the `k`-th Taylor coefficient of the solution (computed
//!    by automatic differentiation of `f` in ball arithmetic).  `B` is
//!    intersected with this.
//! 3. **Mean-value (Lohner) form.**  Write `Y_j ⊆ ŷ_j + A_j r_j` with `ŷ_j` a
//!    point, `A_j` an exact floating-point matrix and `r_j` an interval
//!    vector.  Then
//!    `y(t_{j+1}) ∈ Σ_{k≤p} h^k F_k(ŷ_j) + h^{p+1} F_{p+1}(B) + S_j A_j r_j`,
//!    `S_j = I + Σ_{k≥1} h^k ∂F_k/∂y(hull Y_j)`.  The new frame `A_{j+1}` is
//!    the orthogonal factor of a QR factorisation of `mid(S_j A_j)` (columns
//!    pivoted by their contribution); its inverse is enclosed rigorously by
//!    a Neumann-series bound, so a rounding error in the QR costs width,
//!    never soundness.
//!
//! Everything is evaluated in [`ArbBall`] arithmetic at the requested
//! precision, the time grid consists of exact binary64 numbers, and every
//! step length `h = t_{j+1} - t_j` is itself an enclosure.  Nothing in the
//! method is a floating-point estimate except the *choice* of step size and
//! of the frame `A_j`, and neither can make an enclosure wrong.
//!
//! # Refusals
//!
//! A call either returns enclosures that provably contain the solution, or
//! fails with a coded [`VerifiedOdeError`]:
//!
//! * `E-VALIDATED-001..005` — the right-hand side uses an unsupported
//!   construct, names an unbound symbol, leaves its domain (`log(y)` with
//!   `y ≤ 0` at the initial point), overflows, or the request is malformed;
//! * `E-VALIDATED-006` — the step size fell below `h_min`: no a-priori
//!   enclosure could be validated (finite-time blow-up, a singularity of `f`
//!   ahead, or a non-Lipschitz point such as `sqrt(y)` at `y = 0`);
//! * `E-VALIDATED-007` — the enclosure grew wider than `max_rel_width` times
//!   the magnitude of the solution, so it no longer says anything useful
//!   (instability, very long chaotic horizons);
//! * `E-VALIDATED-008` — more than `max_steps` steps would be needed (stiff
//!   problems, which an explicit Taylor method resolves only with step sizes
//!   of order `1/‖∂f/∂y‖`);
//! * `E-BUDGET-*` — the active [`crate::budget::Budget`] ran out.
//!
//! Supported primitives in `f`: `+`, `·`, integer, rational and real-constant
//! powers, symbolic powers `a^b = exp(b·log a)`, `exp`, `log`, `sqrt`, `sin`,
//! `cos`, `tan`, `sinh`, `cosh`, `tanh`, `atan`, and the named constants `pi`,
//! `EulerGamma`, `Catalan`.
//!
//! # Example
//!
//! ```
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::validated::ode::{verified_ode, VerifiedOdeOptions};
//!
//! let pool = ExprPool::new();
//! let y = pool.symbol("y", Domain::Real);
//! // y' = -y, y(0) = 1  ⇒  y(1) = e^{-1}
//! let rhs = pool.mul(vec![pool.integer(-1_i32), y]);
//! let sol = verified_ode(&[rhs], &[y], None, &[(1.0, 1.0)], (0.0, 1.0), &pool,
//!                        &VerifiedOdeOptions::default()).unwrap();
//! let (lo, hi) = sol.bounds_at(1.0).unwrap()[0];
//! let exact = (-1.0_f64).exp();
//! assert!(lo <= exact && exact <= hi && hi - lo < 1e-12);
//! ```

use super::{from_bounds, hull, is_finite, mag, ub, width, ValidatedError};
use crate::ball::ArbBall;
use crate::budget::{self, BudgetTrip};
use crate::errors::AlkahestError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::float::Round;
use rug::Float;
use std::collections::HashMap;
use std::fmt;

type VResult<T> = std::result::Result<T, ValidatedError>;

/// Highest Taylor order accepted by [`verified_ode`].
pub const MAX_ODE_ORDER: usize = 60;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why [`verified_ode`] (or a query on its result) refused.
///
/// Every variant is a refusal: no enclosure is returned that is not proven.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VerifiedOdeError {
    /// A validated-numerics refusal from evaluating the right-hand side:
    /// unsupported primitive, unbound symbol, domain violation, overflow, or a
    /// malformed request (`E-VALIDATED-001` … `E-VALIDATED-005`).
    Validated(ValidatedError),
    /// The step size fell below the minimum before an a-priori enclosure
    /// could be validated (`E-VALIDATED-006`).
    StepSizeUnderflow {
        /// Time reached when the integration stopped.
        t: f64,
        /// The last step size tried.
        h: f64,
        /// Why the last attempt failed.
        reason: String,
    },
    /// The enclosure became wider than `max_rel_width · max(1, |y|)` in some
    /// component (`E-VALIDATED-007`).
    EnclosureTooWide {
        /// Time at which the width limit was exceeded.
        t: f64,
        /// Index of the offending component.
        component: usize,
        /// Its width (rounded up).
        width: f64,
    },
    /// More than `max_steps` steps would be needed (`E-VALIDATED-008`).
    StepLimit {
        /// Time reached when the limit was hit.
        t: f64,
        /// The step limit.
        steps: usize,
    },
    /// The active budget ran out or the computation was cancelled.
    Budget(BudgetTrip),
}

impl fmt::Display for VerifiedOdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifiedOdeError::Validated(e) => write!(f, "verified_ode: {e}"),
            VerifiedOdeError::StepSizeUnderflow { t, h, reason } => write!(
                f,
                "verified_ode: step size underflow at t = {t} (h = {h:e}): {reason}; \
                 the solution may blow up, leave the domain of f, or reach a \
                 non-Lipschitz point near here"
            ),
            VerifiedOdeError::EnclosureTooWide {
                t,
                component,
                width,
            } => write!(
                f,
                "verified_ode: enclosure of component {component} grew to width {width:e} \
                 at t = {t}; it no longer carries information"
            ),
            VerifiedOdeError::StepLimit { t, steps } => write!(
                f,
                "verified_ode: step limit of {steps} reached at t = {t} \
                 (the problem may be stiff)"
            ),
            VerifiedOdeError::Budget(b) => write!(f, "verified_ode: {b}"),
        }
    }
}

impl std::error::Error for VerifiedOdeError {}

impl AlkahestError for VerifiedOdeError {
    fn code(&self) -> &'static str {
        match self {
            VerifiedOdeError::Validated(e) => e.code(),
            VerifiedOdeError::StepSizeUnderflow { .. } => "E-VALIDATED-006",
            VerifiedOdeError::EnclosureTooWide { .. } => "E-VALIDATED-007",
            VerifiedOdeError::StepLimit { .. } => "E-VALIDATED-008",
            VerifiedOdeError::Budget(b) => b.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            VerifiedOdeError::Validated(e) => e.remediation(),
            VerifiedOdeError::StepSizeUnderflow { .. } => Some(
                "integrate over a shorter span that stops before the blow-up or singularity, or start from a point where the right-hand side is smooth (e.g. away from sqrt(0))",
            ),
            VerifiedOdeError::EnclosureTooWide { .. } => Some(
                "shorten the time span, shrink the initial box, raise the Taylor order or the precision, or lower tol",
            ),
            VerifiedOdeError::StepLimit { .. } => Some(
                "raise max_steps or shorten the span; stiff problems need step sizes of order 1/|df/dy| with an explicit validated method",
            ),
            VerifiedOdeError::Budget(b) => b.remediation(),
        }
    }
}

impl From<ValidatedError> for VerifiedOdeError {
    fn from(e: ValidatedError) -> Self {
        VerifiedOdeError::Validated(e)
    }
}

fn invalid(what: impl Into<String>) -> VerifiedOdeError {
    VerifiedOdeError::Validated(ValidatedError::InvalidInput { what: what.into() })
}

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

/// Tuning knobs for [`verified_ode`].  None of them affects soundness.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VerifiedOdeOptions {
    /// Taylor order `p` of each step (`1..=MAX_ODE_ORDER`).  Default 20.
    pub order: usize,
    /// Working precision in bits.  Default 128.
    pub prec: u32,
    /// Target local error per step, relative to `max(1, |y|)`; drives the
    /// step-size choice only.  Default `1e-16`.
    pub tol: f64,
    /// Maximum number of accepted steps.  Default 20 000.
    pub max_steps: usize,
    /// Smallest step size tried before refusing with `E-VALIDATED-006`.
    /// `None` means `1e-12 · (t₁ - t₀)`.
    pub h_min: Option<f64>,
    /// Refuse (`E-VALIDATED-007`) once a component's width exceeds
    /// `max_rel_width · max(1, |midpoint|)`.  Default 1.
    pub max_rel_width: f64,
}

impl Default for VerifiedOdeOptions {
    fn default() -> Self {
        VerifiedOdeOptions {
            order: 20,
            prec: 128,
            tol: 1e-16,
            max_steps: 20_000,
            h_min: None,
            max_rel_width: 1.0,
        }
    }
}

/// Everything needed to enclose the solution anywhere inside one step.
#[derive(Debug, Clone)]
struct StepRecord {
    t0: f64,
    t1: f64,
    /// Point Taylor coefficients `F_k(ŷ_j)`, `k = 0..=p`, indexed `[k][i]`.
    coeffs: Vec<Vec<ArbBall>>,
    /// `(∂F_k/∂y)(hull Y_j) · A_j`, indexed `[k][i][l]` (`k = 0` is `A_j`).
    ja: Vec<Vec<Vec<ArbBall>>>,
    /// Lohner interval coordinates `r_j` (in the pivoted frame of `ja`).
    r: Vec<ArbBall>,
    /// `F_{p+1}` over the step's a-priori enclosure.
    rem: Vec<ArbBall>,
    /// The validated a-priori enclosure of the whole step.
    apriori: Vec<ArbBall>,
}

/// Rigorous enclosures of an ODE solution, returned by [`verified_ode`].
///
/// `times()[i]` is an exact binary64 grid point and `enclosures()[i]` a box
/// (one ball per component) that contains `y(times()[i])`.
/// [`Self::enclosure_at`] encloses the solution at any `t` in the span.
#[derive(Debug, Clone)]
pub struct VerifiedOdeSolution {
    times: Vec<f64>,
    boxes: Vec<Vec<ArbBall>>,
    steps: Vec<StepRecord>,
    order: usize,
    prec: u32,
}

impl VerifiedOdeSolution {
    /// The grid `t₀ < t₁ < … < t_N` (exact binary64 values).
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Enclosures of `y(t_i)` on the grid.
    pub fn enclosures(&self) -> &[Vec<ArbBall>] {
        &self.boxes
    }

    /// Number of grid points (`steps() + 1`).
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Always `false` — a solution has at least the initial point.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Number of accepted integration steps.
    pub fn steps(&self) -> usize {
        self.steps.len()
    }

    /// State dimension.
    pub fn dim(&self) -> usize {
        self.boxes.first().map_or(0, |b| b.len())
    }

    /// Grid enclosures as `(lo, hi)` pairs rounded **outward** to `f64`.
    pub fn bounds(&self) -> Vec<Vec<(f64, f64)>> {
        self.boxes.iter().map(|b| to_bounds(b)).collect()
    }

    /// Widest component width over the whole grid, rounded up.
    pub fn max_width(&self) -> f64 {
        self.boxes
            .iter()
            .flat_map(|b| b.iter())
            .map(|c| width(c).to_f64_round(Round::Up))
            .fold(0.0, f64::max)
    }

    /// A rigorous enclosure of `y(t)` for any `t` in the integration span.
    ///
    /// Grid points return the grid box; interior points evaluate the step's
    /// Taylor polynomial in mean-value form plus its remainder and intersect
    /// the result with the step's a-priori enclosure.
    pub fn enclosure_at(&self, t: f64) -> Result<Vec<ArbBall>, VerifiedOdeError> {
        let (first, last) = (self.times[0], *self.times.last().unwrap());
        if !(t >= first && t <= last) {
            return Err(invalid(format!(
                "t = {t} is outside the integrated span [{first}, {last}]"
            )));
        }
        if let Ok(i) = self
            .times
            .binary_search_by(|x| x.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Less))
        {
            return Ok(self.boxes[i].clone());
        }
        // Step whose interval contains `t`.
        let j = self.times.partition_point(|&x| x < t) - 1;
        let st = &self.steps[j];
        debug_assert!(st.t0 < t && t < st.t1);
        let prec = self.prec;
        let p = self.order;
        let s = ArbBall::from_f64(t, prec) - ArbBall::from_f64(st.t0, prec);
        let spow = powers(&s, p + 1, prec);
        let n = st.r.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut acc = zero(prec);
            for (sk, ck) in spow.iter().zip(&st.coeffs) {
                acc = acc.add_ref(&sk.mul_ref(&ck[i]));
            }
            for l in 0..n {
                let mut m = zero(prec);
                for (sk, jk) in spow.iter().zip(&st.ja) {
                    m = m.add_ref(&sk.mul_ref(&jk[i][l]));
                }
                acc = acc.add_ref(&m.mul_ref(&st.r[l]));
            }
            acc = acc.add_ref(&spow[p + 1].mul_ref(&st.rem[i]));
            out.push(intersect(&acc, &st.apriori[i]).unwrap_or_else(|| st.apriori[i].clone()));
        }
        Ok(out)
    }

    /// [`Self::enclosure_at`] as `(lo, hi)` pairs rounded outward to `f64`.
    pub fn bounds_at(&self, t: f64) -> Result<Vec<(f64, f64)>, VerifiedOdeError> {
        Ok(to_bounds(&self.enclosure_at(t)?))
    }
}

fn to_bounds(b: &[ArbBall]) -> Vec<(f64, f64)> {
    b.iter()
        .map(|c| {
            (
                c.lo().to_f64_round(Round::Down),
                c.hi().to_f64_round(Round::Up),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Ball helpers
// ---------------------------------------------------------------------------

fn zero(prec: u32) -> ArbBall {
    ArbBall::from_f64(0.0, prec)
}

fn one(prec: u32) -> ArbBall {
    ArbBall::from_f64(1.0, prec)
}

fn int_ball(k: usize, prec: u32) -> ArbBall {
    ArbBall::from_integer(&rug::Integer::from(k), prec)
}

fn bdiv(a: &ArbBall, b: &ArbBall) -> VResult<ArbBall> {
    match a.clone() / b.clone() {
        Some(q) if is_finite(&q) => Ok(q),
        _ => Err(ValidatedError::DomainViolation {
            what: "division by an enclosure that contains zero in an ODE right-hand side".into(),
        }),
    }
}

/// `[x, y] ∩ [u, v]`, or `None` when they are disjoint.
fn intersect(a: &ArbBall, b: &ArbBall) -> Option<ArbBall> {
    if !is_finite(a) {
        return Some(b.clone());
    }
    if !is_finite(b) {
        return Some(a.clone());
    }
    let (alo, ahi, blo, bhi) = (a.lo(), a.hi(), b.lo(), b.hi());
    let lo = if alo > blo { alo } else { blo };
    let hi = if ahi < bhi { ahi } else { bhi };
    if lo > hi {
        return None;
    }
    let prec = a.prec.max(b.prec);
    Some(from_bounds(&lo, &hi, prec))
}

/// `b^0, b^1, …, b^m`.
fn powers(b: &ArbBall, m: usize, prec: u32) -> Vec<ArbBall> {
    let mut out = Vec::with_capacity(m + 1);
    out.push(one(prec));
    for k in 1..=m {
        let next = out[k - 1].mul_ref(b);
        out.push(next);
    }
    out
}

/// `[0, h]^0, …, [0, h]^m` as the intervals `[0, ub(h^k)]` (and `{1}` for
/// `k = 0`).  `h` must be a positive enclosure.
fn unit_powers(h: &ArbBall, m: usize, prec: u32) -> Vec<ArbBall> {
    let hp = powers(h, m, prec);
    let z = Float::new(prec);
    let mut out = Vec::with_capacity(m + 1);
    out.push(one(prec));
    for item in hp.iter().skip(1) {
        out.push(from_bounds(&z, &ub(item), prec));
    }
    out
}

/// `true` when `inner` lies in the topological interior of `outer`.
fn strictly_inside(inner: &ArbBall, outer: &ArbBall) -> bool {
    if !is_finite(inner) || !is_finite(outer) {
        return false;
    }
    let prec = outer.prec;
    // The largest interval certainly inside the real ball `outer`.
    let olo = Float::with_val_round(prec, &outer.mid - &outer.rad, Round::Up).0;
    let ohi = Float::with_val_round(prec, &outer.mid + &outer.rad, Round::Down).0;
    inner.lo() > olo && inner.hi() < ohi
}

/// Widen `b` about its midpoint: radius `·(1 + frac)` plus a small absolute
/// slack proportional to `1 + |mid|`.
fn widen(b: &ArbBall, frac: f64) -> ArbBall {
    let prec = b.prec;
    let mut slack = Float::with_val(prec, b.mid.abs_ref()) + 1u32;
    slack >>= prec.saturating_sub(12).min(200);
    let mut out = b.clone();
    out.rad = Float::with_val(prec, &b.rad * (1.0 + frac)) + slack;
    out
}

fn mag_f64(b: &ArbBall) -> f64 {
    mag(b).to_f64_round(Round::Up)
}

// ---------------------------------------------------------------------------
// Value + gradient scalars (forward-mode AD over balls)
// ---------------------------------------------------------------------------

/// A ball value with an (optional) gradient with respect to the initial
/// state of the Taylor series.  An empty gradient means "identically zero".
#[derive(Clone, Debug)]
struct D {
    v: ArbBall,
    g: Vec<ArbBall>,
}

impl D {
    fn c(v: ArbBall) -> D {
        D { v, g: Vec::new() }
    }

    fn add(&self, o: &D) -> D {
        let g = if self.g.is_empty() {
            o.g.clone()
        } else if o.g.is_empty() {
            self.g.clone()
        } else {
            self.g.iter().zip(&o.g).map(|(a, b)| a.add_ref(b)).collect()
        };
        D {
            v: self.v.add_ref(&o.v),
            g,
        }
    }

    fn neg(&self) -> D {
        D {
            v: -self.v.clone(),
            g: self.g.iter().map(|a| -a.clone()).collect(),
        }
    }

    fn sub(&self, o: &D) -> D {
        self.add(&o.neg())
    }

    fn mul(&self, o: &D) -> D {
        let g = match (self.g.is_empty(), o.g.is_empty()) {
            (true, true) => Vec::new(),
            (false, true) => self.g.iter().map(|a| a.mul_ref(&o.v)).collect(),
            (true, false) => o.g.iter().map(|b| b.mul_ref(&self.v)).collect(),
            (false, false) => self
                .g
                .iter()
                .zip(&o.g)
                .map(|(a, b)| a.mul_ref(&o.v).add_ref(&b.mul_ref(&self.v)))
                .collect(),
        };
        D {
            v: self.v.mul_ref(&o.v),
            g,
        }
    }

    fn scale(&self, k: &ArbBall) -> D {
        D {
            v: self.v.mul_ref(k),
            g: self.g.iter().map(|a| a.mul_ref(k)).collect(),
        }
    }

    fn div(&self, o: &D) -> VResult<D> {
        let q = bdiv(&self.v, &o.v)?;
        let g = if o.g.is_empty() {
            self.g
                .iter()
                .map(|a| bdiv(a, &o.v))
                .collect::<VResult<_>>()?
        } else {
            let n = o.g.len();
            let mut g = Vec::with_capacity(n);
            for l in 0..n {
                let num = match self.g.get(l) {
                    Some(a) => a.clone() - q.mul_ref(&o.g[l]),
                    None => -q.mul_ref(&o.g[l]),
                };
                g.push(bdiv(&num, &o.v)?);
            }
            g
        };
        Ok(D { v: q, g })
    }

    fn div_int(&self, k: usize, prec: u32) -> VResult<D> {
        let kb = int_ball(k, prec);
        Ok(D {
            v: bdiv(&self.v, &kb)?,
            g: self
                .g
                .iter()
                .map(|a| bdiv(a, &kb))
                .collect::<VResult<_>>()?,
        })
    }

    /// `f(x)` given `f(x.v)` and `f'(x.v)` (the latter only used when `x`
    /// carries a gradient).
    fn chain(fv: ArbBall, dfv: impl FnOnce() -> VResult<ArbBall>, x: &D) -> VResult<D> {
        if x.g.is_empty() {
            return Ok(D::c(fv));
        }
        let d = dfv()?;
        Ok(D {
            v: fv,
            g: x.g.iter().map(|a| d.mul_ref(a)).collect(),
        })
    }
}

// ---------------------------------------------------------------------------
// Expression → straight-line Taylor-series program
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Node {
    Const(ArbBall),
    State(usize),
    Time,
    Add(Vec<usize>),
    Mul(usize, usize),
    Div(usize, usize),
    Exp(usize),
    Log(usize),
    Sqrt(usize),
    PowReal(usize, ArbBall),
    /// `sin(a)`, paired with the `cos(a)` node.
    Sin(usize, usize),
    /// `cos(a)`, paired with the `sin(a)` node.
    Cos(usize, usize),
    Sinh(usize, usize),
    Cosh(usize, usize),
    /// `atan(a)` with the node of `1 + a²`.
    Atan(usize, usize),
}

struct Tape {
    nodes: Vec<Node>,
    outputs: Vec<usize>,
}

struct Compiler<'a> {
    pool: &'a ExprPool,
    vars: &'a [ExprId],
    time: Option<ExprId>,
    prec: u32,
    nodes: Vec<Node>,
    memo: HashMap<ExprId, usize>,
    pairs: HashMap<(usize, bool), (usize, usize)>,
}

impl<'a> Compiler<'a> {
    fn push(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn konst(&mut self, b: ArbBall) -> usize {
        self.push(Node::Const(b))
    }

    fn pow_int(&mut self, a: usize, n: u64) -> usize {
        // Binary exponentiation; n >= 1.
        let mut result: Option<usize> = None;
        let mut base = a;
        let mut e = n;
        loop {
            if e & 1 == 1 {
                result = Some(match result {
                    None => base,
                    Some(r) => self.push(Node::Mul(r, base)),
                });
            }
            e >>= 1;
            if e == 0 {
                break;
            }
            base = self.push(Node::Mul(base, base));
        }
        result.unwrap_or(a)
    }

    /// The `(sin, cos)` (or `(sinh, cosh)`) node pair of `a`.
    fn pair(&mut self, a: usize, hyperbolic: bool) -> (usize, usize) {
        if let Some(&p) = self.pairs.get(&(a, hyperbolic)) {
            return p;
        }
        let s = self.nodes.len();
        let c = s + 1;
        if hyperbolic {
            self.nodes.push(Node::Sinh(a, c));
            self.nodes.push(Node::Cosh(a, s));
        } else {
            self.nodes.push(Node::Sin(a, c));
            self.nodes.push(Node::Cos(a, s));
        }
        self.pairs.insert((a, hyperbolic), (s, c));
        (s, c)
    }

    fn compile(&mut self, expr: ExprId) -> VResult<usize> {
        if let Some(&i) = self.memo.get(&expr) {
            return Ok(i);
        }
        let i = self.compile_uncached(expr)?;
        self.memo.insert(expr, i);
        Ok(i)
    }

    fn compile_uncached(&mut self, expr: ExprId) -> VResult<usize> {
        let prec = self.prec;
        match self.pool.get(expr) {
            ExprData::Integer(n) => Ok(self.konst(ArbBall::from_integer(&n.0, prec))),
            ExprData::Rational(r) => Ok(self.konst(ArbBall::from_rational(&r.0, prec))),
            ExprData::Float(f) => Ok(self.konst(ArbBall::from_float(&f.inner, prec))),
            ExprData::Symbol { name, .. } => {
                if let Some(i) = self.vars.iter().position(|&v| v == expr) {
                    Ok(self.push(Node::State(i)))
                } else if self.time == Some(expr) {
                    Ok(self.push(Node::Time))
                } else if let Some(c) = crate::ball::named_constant_ball(&name, prec) {
                    Ok(self.konst(c))
                } else {
                    Err(ValidatedError::UnboundSymbol { name })
                }
            }
            ExprData::Add(args) => {
                let ids = args
                    .iter()
                    .map(|&a| self.compile(a))
                    .collect::<VResult<Vec<_>>>()?;
                Ok(self.push(Node::Add(ids)))
            }
            ExprData::Mul(args) => {
                let mut acc: Option<usize> = None;
                for a in args {
                    let t = self.compile(a)?;
                    acc = Some(match acc {
                        None => t,
                        Some(r) => self.push(Node::Mul(r, t)),
                    });
                }
                Ok(acc.unwrap_or_else(|| self.konst(one(prec))))
            }
            ExprData::Pow { base, exp } => {
                let b = self.compile(base)?;
                match self.pool.get(exp) {
                    ExprData::Integer(n) => {
                        let nv = n.0.to_i64().ok_or_else(|| ValidatedError::Unsupported {
                            what: "integer exponent does not fit in i64".into(),
                        })?;
                        if nv == 0 {
                            Ok(self.konst(one(prec)))
                        } else if nv > 0 {
                            Ok(self.pow_int(b, nv as u64))
                        } else {
                            let pw = self.pow_int(b, nv.unsigned_abs());
                            let o = self.konst(one(prec));
                            Ok(self.push(Node::Div(o, pw)))
                        }
                    }
                    ExprData::Rational(r) => {
                        let half = rug::Rational::from((1, 2));
                        if r.0 == half {
                            Ok(self.push(Node::Sqrt(b)))
                        } else if r.0 == -half {
                            let s = self.push(Node::Sqrt(b));
                            let o = self.konst(one(prec));
                            Ok(self.push(Node::Div(o, s)))
                        } else {
                            Ok(self.push(Node::PowReal(b, ArbBall::from_rational(&r.0, prec))))
                        }
                    }
                    ExprData::Float(f) => {
                        Ok(self.push(Node::PowReal(b, ArbBall::from_float(&f.inner, prec))))
                    }
                    _ => {
                        // a^e = exp(e · log a), defined for a > 0.
                        let e = self.compile(exp)?;
                        let l = self.push(Node::Log(b));
                        let m = self.push(Node::Mul(e, l));
                        Ok(self.push(Node::Exp(m)))
                    }
                }
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let a = self.compile(args[0])?;
                match name.as_str() {
                    "exp" => Ok(self.push(Node::Exp(a))),
                    "log" | "ln" => Ok(self.push(Node::Log(a))),
                    "sqrt" => Ok(self.push(Node::Sqrt(a))),
                    "sin" => Ok(self.pair(a, false).0),
                    "cos" => Ok(self.pair(a, false).1),
                    "tan" => {
                        let (s, c) = self.pair(a, false);
                        Ok(self.push(Node::Div(s, c)))
                    }
                    "sinh" => Ok(self.pair(a, true).0),
                    "cosh" => Ok(self.pair(a, true).1),
                    "tanh" => {
                        let (s, c) = self.pair(a, true);
                        Ok(self.push(Node::Div(s, c)))
                    }
                    "atan" => {
                        let sq = self.push(Node::Mul(a, a));
                        let o = self.konst(one(prec));
                        let q = self.push(Node::Add(vec![o, sq]));
                        Ok(self.push(Node::Atan(a, q)))
                    }
                    other => Err(ValidatedError::Unsupported {
                        what: format!(
                            "function `{other}` in an ODE right-hand side (verified_ode supports \
                             exp, log, sqrt, sin, cos, tan, sinh, cosh, tanh, atan and powers)"
                        ),
                    }),
                }
            }
            ExprData::Func { name, args } => Err(ValidatedError::Unsupported {
                what: format!("function `{name}` with {} arguments", args.len()),
            }),
            other => Err(ValidatedError::Unsupported {
                what: format!("expression node {other:?} in an ODE right-hand side"),
            }),
        }
    }
}

fn compile_tape(
    rhs: &[ExprId],
    vars: &[ExprId],
    time: Option<ExprId>,
    pool: &ExprPool,
    prec: u32,
) -> VResult<Tape> {
    let mut c = Compiler {
        pool,
        vars,
        time,
        prec,
        nodes: Vec::new(),
        memo: HashMap::new(),
        pairs: HashMap::new(),
    };
    let outputs = rhs
        .iter()
        .map(|&e| c.compile(e))
        .collect::<VResult<Vec<_>>>()?;
    Ok(Tape {
        nodes: c.nodes,
        outputs,
    })
}

/// Elementary function of the order-0 coefficient, with its derivative for
/// the gradient part.
fn elem0(node: &Node, x: &D, prec: u32) -> VResult<D> {
    let dom = |what: &str| ValidatedError::DomainViolation {
        what: format!("{what} in an ODE right-hand side"),
    };
    let v = &x.v;
    match node {
        Node::Exp(_) => {
            let e = v.exp();
            let e2 = e.clone();
            D::chain(e, || Ok(e2), x)
        }
        Node::Log(_) => {
            let l = v
                .log()
                .ok_or_else(|| dom("log of an enclosure that is not > 0"))?;
            D::chain(l, || bdiv(&one(prec), v), x)
        }
        Node::Sqrt(_) => {
            let s = v
                .sqrt()
                .ok_or_else(|| dom("sqrt of an enclosure that is not >= 0"))?;
            let s2 = s.clone();
            D::chain(s, || bdiv(&one(prec), &s2.mul_ref(&int_ball(2, prec))), x)
        }
        Node::PowReal(_, e) => {
            let l = v
                .log()
                .ok_or_else(|| dom("real power of an enclosure that is not > 0"))?;
            let p = l.mul_ref(e).exp();
            let p2 = p.clone();
            D::chain(p, || bdiv(&p2.mul_ref(e), v), x)
        }
        Node::Sin(..) => D::chain(v.sin(), || Ok(v.cos()), x),
        Node::Cos(..) => D::chain(v.cos(), || Ok(-v.sin()), x),
        Node::Sinh(..) => D::chain(v.sinh(), || Ok(v.cosh()), x),
        Node::Cosh(..) => D::chain(v.cosh(), || Ok(v.sinh()), x),
        Node::Atan(..) => D::chain(
            v.atan(),
            || bdiv(&one(prec), &one(prec).add_ref(&v.mul_ref(v))),
            x,
        ),
        _ => unreachable!("elem0 on a non-elementary node"),
    }
}

/// Taylor coefficients `y_0 … y_kmax` of the solution of `y' = f(t, y)`
/// through `(t0, y0)`, where `t0` and `y0` may be enclosures (then each
/// coefficient encloses the coefficient of every solution through a point
/// of the box) and `y0` may carry gradients (then each coefficient carries
/// its derivative with respect to the initial state).  Indexed `[k][i]`.
fn series(tape: &Tape, t0: &ArbBall, y0: &[D], kmax: usize, prec: u32) -> VResult<Vec<Vec<D>>> {
    let nn = tape.nodes.len();
    let zero_d = D::c(zero(prec));
    let mut s: Vec<Vec<D>> = vec![Vec::with_capacity(kmax); nn];
    let mut y: Vec<Vec<D>> = Vec::with_capacity(kmax + 1);
    y.push(y0.to_vec());
    let kb = |j: usize| int_ball(j, prec);
    for k in 0..kmax {
        for idx in 0..nn {
            let val: D = match &tape.nodes[idx] {
                Node::Const(c) => {
                    if k == 0 {
                        D::c(c.clone())
                    } else {
                        zero_d.clone()
                    }
                }
                Node::State(i) => y[k][*i].clone(),
                Node::Time => match k {
                    0 => D::c(t0.clone()),
                    1 => D::c(one(prec)),
                    _ => zero_d.clone(),
                },
                Node::Add(args) => {
                    let mut acc = s[args[0]][k].clone();
                    for a in &args[1..] {
                        acc = acc.add(&s[*a][k]);
                    }
                    acc
                }
                Node::Mul(a, b) => {
                    let (sa, sb) = (&s[*a], &s[*b]);
                    let mut acc = sa[0].mul(&sb[k]);
                    for j in 1..=k {
                        acc = acc.add(&sa[j].mul(&sb[k - j]));
                    }
                    acc
                }
                Node::Div(a, b) => {
                    let (sa, sb, sw) = (&s[*a], &s[*b], &s[idx]);
                    let mut num = sa[k].clone();
                    for j in 1..=k {
                        num = num.sub(&sb[j].mul(&sw[k - j]));
                    }
                    num.div(&sb[0])?
                }
                node @ (Node::Exp(a)
                | Node::Sin(a, _)
                | Node::Cos(a, _)
                | Node::Sinh(a, _)
                | Node::Cosh(a, _))
                    if k == 0 =>
                {
                    elem0(node, &s[*a][0], prec)?
                }
                node @ (Node::Log(a) | Node::Sqrt(a) | Node::PowReal(a, _) | Node::Atan(a, _))
                    if k == 0 =>
                {
                    elem0(node, &s[*a][0], prec)?
                }
                Node::Exp(a) => {
                    // k e_k = Σ_{j=1}^k j a_j e_{k-j}
                    let (sa, se) = (&s[*a], &s[idx]);
                    let mut acc = zero_d.clone();
                    for j in 1..=k {
                        acc = acc.add(&sa[j].mul(&se[k - j]).scale(&kb(j)));
                    }
                    acc.div_int(k, prec)?
                }
                Node::Sin(a, partner) | Node::Sinh(a, partner) => {
                    // k s_k = Σ j a_j c_{k-j}
                    let (sa, sc) = (&s[*a], &s[*partner]);
                    let mut acc = zero_d.clone();
                    for j in 1..=k {
                        acc = acc.add(&sa[j].mul(&sc[k - j]).scale(&kb(j)));
                    }
                    acc.div_int(k, prec)?
                }
                Node::Cos(a, partner) => {
                    // k c_k = -Σ j a_j s_{k-j}
                    let (sa, ss) = (&s[*a], &s[*partner]);
                    let mut acc = zero_d.clone();
                    for j in 1..=k {
                        acc = acc.add(&sa[j].mul(&ss[k - j]).scale(&kb(j)));
                    }
                    acc.neg().div_int(k, prec)?
                }
                Node::Cosh(a, partner) => {
                    // k ch_k = Σ j a_j sh_{k-j}
                    let (sa, ss) = (&s[*a], &s[*partner]);
                    let mut acc = zero_d.clone();
                    for j in 1..=k {
                        acc = acc.add(&sa[j].mul(&ss[k - j]).scale(&kb(j)));
                    }
                    acc.div_int(k, prec)?
                }
                Node::Log(a) => {
                    // v_k = (k a_k - Σ_{j=1}^{k-1} j v_j a_{k-j}) / (k a_0)
                    let (sa, sv) = (&s[*a], &s[idx]);
                    let mut num = sa[k].scale(&kb(k));
                    for j in 1..k {
                        num = num.sub(&sv[j].mul(&sa[k - j]).scale(&kb(j)));
                    }
                    num.div(&sa[0].scale(&kb(k)))?
                }
                Node::Sqrt(a) => {
                    // q_k = (a_k - Σ_{j=1}^{k-1} q_j q_{k-j}) / (2 q_0)
                    let (sa, sq) = (&s[*a], &s[idx]);
                    let mut num = sa[k].clone();
                    for j in 1..k {
                        num = num.sub(&sq[j].mul(&sq[k - j]));
                    }
                    num.div(&sq[0].scale(&kb(2)))?
                }
                Node::PowReal(a, e) => {
                    // v_k = (e Σ_{j=1}^k j a_j v_{k-j} - Σ_{j=1}^{k-1} j v_j a_{k-j}) / (k a_0)
                    let (sa, sv) = (&s[*a], &s[idx]);
                    let mut t1 = zero_d.clone();
                    for j in 1..=k {
                        t1 = t1.add(&sa[j].mul(&sv[k - j]).scale(&kb(j)));
                    }
                    let mut num = t1.scale(e);
                    for j in 1..k {
                        num = num.sub(&sv[j].mul(&sa[k - j]).scale(&kb(j)));
                    }
                    num.div(&sa[0].scale(&kb(k)))?
                }
                Node::Atan(a, q) => {
                    // v_k = (k a_k - Σ_{j=1}^{k-1} j v_j q_{k-j}) / (k q_0)
                    let (sa, sq, sv) = (&s[*a], &s[*q], &s[idx]);
                    let mut num = sa[k].scale(&kb(k));
                    for j in 1..k {
                        num = num.sub(&sv[j].mul(&sq[k - j]).scale(&kb(j)));
                    }
                    num.div(&sq[0].scale(&kb(k)))?
                }
            };
            s[idx].push(val);
        }
        let mut next = Vec::with_capacity(tape.outputs.len());
        for &o in &tape.outputs {
            next.push(s[o][k].div_int(k + 1, prec)?);
        }
        y.push(next);
    }
    for row in &y {
        for d in row {
            if !is_finite(&d.v) || d.g.iter().any(|g| !is_finite(g)) {
                return Err(ValidatedError::NotFinite {
                    what: "a Taylor coefficient of the ODE solution".into(),
                });
            }
        }
    }
    Ok(y)
}

// ---------------------------------------------------------------------------
// Small dense linear algebra
// ---------------------------------------------------------------------------

type BMat = Vec<Vec<ArbBall>>;

fn identity(n: usize, prec: u32) -> BMat {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|l| if i == l { one(prec) } else { zero(prec) })
                .collect()
        })
        .collect()
}

fn matmul(a: &BMat, b: &BMat, prec: u32) -> BMat {
    let n = a.len();
    let m = b.first().map_or(0, |r| r.len());
    let inner = b.len();
    (0..n)
        .map(|i| {
            (0..m)
                .map(|l| {
                    let mut acc = zero(prec);
                    for k in 0..inner {
                        acc = acc.add_ref(&a[i][k].mul_ref(&b[k][l]));
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

fn matvec(a: &BMat, r: &[ArbBall], prec: u32) -> Vec<ArbBall> {
    a.iter()
        .map(|row| {
            let mut acc = zero(prec);
            for (x, y) in row.iter().zip(r) {
                acc = acc.add_ref(&x.mul_ref(y));
            }
            acc
        })
        .collect()
}

/// Orthogonal factor `Q` of a Householder QR of `m` (binary64; only used to
/// pick a coordinate frame — its inverse is enclosed rigorously elsewhere).
#[allow(clippy::needless_range_loop)] // index-heavy dense linear algebra
fn householder_q(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut r: Vec<Vec<f64>> = m.to_vec();
    let mut q: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|l| if i == l { 1.0 } else { 0.0 }).collect())
        .collect();
    for k in 0..n.saturating_sub(1) {
        let norm: f64 = (k..n).map(|i| r[i][k] * r[i][k]).sum::<f64>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            continue;
        }
        let alpha = if r[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k..n).map(|i| r[i][k]).collect();
        v[0] -= alpha;
        let vn: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if vn == 0.0 || !vn.is_finite() {
            continue;
        }
        for x in &mut v {
            *x /= vn;
        }
        for col in 0..n {
            let dot: f64 = (k..n).map(|i| v[i - k] * r[i][col]).sum();
            for i in k..n {
                r[i][col] -= 2.0 * v[i - k] * dot;
            }
        }
        for row in q.iter_mut() {
            let dot: f64 = (k..n).map(|i| row[i] * v[i - k]).sum();
            for i in k..n {
                row[i] -= 2.0 * dot * v[i - k];
            }
        }
    }
    if q.iter().flatten().all(|x| x.is_finite()) {
        Some(q)
    } else {
        None
    }
}

/// A new frame `Q` (exact entries) and a rigorous enclosure of `Q⁻¹`.
fn frame_and_inverse(sa: &BMat, prec: u32) -> (BMat, BMat) {
    let n = sa.len();
    let mid: Vec<Vec<f64>> = sa
        .iter()
        .map(|row| row.iter().map(|b| b.mid_f64()).collect())
        .collect();
    let fallback = || (identity(n, prec), identity(n, prec));
    let Some(q) = householder_q(&mid) else {
        return fallback();
    };
    let qb: BMat = q
        .iter()
        .map(|row| row.iter().map(|&x| ArbBall::from_f64(x, prec)).collect())
        .collect();
    // C = Qᵀ, E = I - C·Q; if ‖E‖∞ ≤ e < 1 then Q⁻¹ = Σ_k Eᵏ C and every
    // entry of Q⁻¹ - C is bounded by e·‖C‖∞ / (1 - e).
    let cb: BMat = (0..n)
        .map(|i| (0..n).map(|l| qb[l][i].clone()).collect())
        .collect();
    let cq = matmul(&cb, &qb, prec);
    let mut e_norm = Float::new(prec);
    let mut c_norm = Float::new(prec);
    for i in 0..n {
        let mut es = Float::new(prec);
        let mut cs = Float::new(prec);
        for l in 0..n {
            let id = if i == l { one(prec) } else { zero(prec) };
            let e = id - cq[i][l].clone();
            es = Float::with_val_round(prec, &es + mag(&e), Round::Up).0;
            cs = Float::with_val_round(prec, &cs + mag(&cb[i][l]), Round::Up).0;
        }
        if es > e_norm {
            e_norm = es;
        }
        if cs > c_norm {
            c_norm = cs;
        }
    }
    if e_norm.is_nan() || e_norm >= 0.5 {
        return fallback();
    }
    let eb = ArbBall {
        mid: e_norm.clone(),
        rad: Float::new(prec),
        prec,
    };
    let cnb = ArbBall {
        mid: c_norm,
        rad: Float::new(prec),
        prec,
    };
    let Ok(delta) = bdiv(&eb.mul_ref(&cnb), &(one(prec) - eb.clone())) else {
        return fallback();
    };
    let delta = ub(&delta);
    let inv: BMat = cb
        .iter()
        .map(|row| {
            row.iter()
                .map(|c| {
                    super::inflate(&ArbBall {
                        mid: c.mid.clone(),
                        rad: Float::with_val(prec, &c.rad + &delta),
                        prec,
                    })
                })
                .collect()
        })
        .collect();
    (qb, inv)
}

// ---------------------------------------------------------------------------
// The integrator
// ---------------------------------------------------------------------------

struct Lohner {
    t: f64,
    yhat: Vec<ArbBall>,
    a: BMat,
    r: Vec<ArbBall>,
    y: Vec<ArbBall>,
}

/// Outcome of one attempt at a step of a given size.
enum Attempt {
    Done {
        encl: Vec<ArbBall>,
        rem: Vec<ArbBall>,
    },
    Retry(String),
}

struct Integrator<'a> {
    tape: &'a Tape,
    n: usize,
    p: usize,
    prec: u32,
    tol: f64,
}

impl<'a> Integrator<'a> {
    fn ball(&self, x: f64) -> ArbBall {
        ArbBall::from_f64(x, self.prec)
    }

    /// Step-size suggestion from the decay of the point Taylor coefficients.
    fn suggest_h(&self, c: &[Vec<D>], scale: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        for k in [self.p.saturating_sub(1).max(1), self.p] {
            let m = c[k].iter().map(|d| mag_f64(&d.v)).fold(0.0, f64::max);
            if m > 0.0 && m.is_finite() {
                let hk = (self.tol * scale / m).powf(1.0 / k as f64);
                best = Some(best.map_or(hk, |b: f64| b.min(hk)));
            }
        }
        best.map(|h| 0.9 * h)
    }

    /// Try to validate an a-priori enclosure on `[t, t+h]` from the box `hbox`
    /// and to bound the Taylor remainder over it.
    fn attempt(&self, hbox: &[ArbBall], ch: &[Vec<D>], tb: &ArbBall, hb: &ArbBall) -> Attempt {
        let prec = self.prec;
        let n = self.n;
        let p = self.p;
        let h01 = from_bounds(&Float::new(prec), &ub(hb), prec);
        let f_over = |b: &[ArbBall]| -> VResult<Vec<ArbBall>> {
            // Values *and* the Jacobian of f over the box: a finite Jacobian
            // enclosure is what makes f Lipschitz on it.
            let yd: Vec<D> = b
                .iter()
                .enumerate()
                .map(|(i, v)| D {
                    v: v.clone(),
                    g: (0..n)
                        .map(|l| if l == i { one(prec) } else { zero(prec) })
                        .collect(),
                })
                .collect();
            let s = series(self.tape, tb, &yd, 1, prec)?;
            Ok(s[1].iter().map(|d| d.v.clone()).collect())
        };
        // 1. Picard–Lindelöf: find B with hbox + [0,h]·f(T, B) ⊂ int B.
        let f0 = match f_over(hbox) {
            Ok(f) => f,
            Err(e) => return Attempt::Retry(e.to_string()),
        };
        let mut b: Vec<ArbBall> = (0..n)
            .map(|i| widen(&hbox[i].add_ref(&h01.mul_ref(&f0[i])), 0.2))
            .collect();
        let mut validated: Option<Vec<ArbBall>> = None;
        for _ in 0..8 {
            let fb = match f_over(&b) {
                Ok(f) => f,
                Err(e) => return Attempt::Retry(e.to_string()),
            };
            let e: Vec<ArbBall> = (0..n)
                .map(|i| hbox[i].add_ref(&h01.mul_ref(&fb[i])))
                .collect();
            if (0..n).all(|i| strictly_inside(&e[i], &b[i])) {
                validated = Some(e);
                break;
            }
            b = (0..n).map(|i| widen(&hull(&b[i], &e[i]), 0.5)).collect();
        }
        let Some(mut encl) = validated else {
            return Attempt::Retry("no a-priori enclosure validated (Picard–Lindelöf)".into());
        };
        // 2. Tighten with the Taylor form over [0, h].
        let up = unit_powers(hb, p + 1, prec);
        for _ in 0..2 {
            let ed: Vec<D> = encl.iter().map(|v| D::c(v.clone())).collect();
            let cb = match series(self.tape, tb, &ed, p + 1, prec) {
                Ok(c) => c,
                Err(e) => return Attempt::Retry(e.to_string()),
            };
            let mut next = Vec::with_capacity(n);
            for i in 0..n {
                let mut acc = zero(prec);
                for (k, uk) in up.iter().enumerate().take(p + 1) {
                    acc = acc.add_ref(&uk.mul_ref(&ch[k][i].v));
                }
                acc = acc.add_ref(&up[p + 1].mul_ref(&cb[p + 1][i].v));
                next.push(intersect(&acc, &encl[i]).unwrap_or_else(|| encl[i].clone()));
            }
            encl = next;
        }
        // 3. Remainder coefficient over the final enclosure.
        let ed: Vec<D> = encl.iter().map(|v| D::c(v.clone())).collect();
        let cb = match series(self.tape, tb, &ed, p + 1, prec) {
            Ok(c) => c,
            Err(e) => return Attempt::Retry(e.to_string()),
        };
        let rem = cb[p + 1].iter().map(|d| d.v.clone()).collect();
        Attempt::Done { encl, rem }
    }

    fn step(
        &self,
        st: &Lohner,
        t_end: f64,
        h_min: f64,
    ) -> Result<(StepRecord, Lohner), VerifiedOdeError> {
        let prec = self.prec;
        let (n, p) = (self.n, self.p);
        let tj = self.ball(st.t);
        // Point series at ŷ and value+Jacobian series over the hull of Y_j.
        let yh: Vec<D> = st.yhat.iter().map(|v| D::c(v.clone())).collect();
        let c = series(self.tape, &tj, &yh, p, prec)?;
        let hbox: Vec<ArbBall> = (0..n).map(|i| hull(&st.y[i], &st.yhat[i])).collect();
        let hd: Vec<D> = hbox
            .iter()
            .enumerate()
            .map(|(i, v)| D {
                v: v.clone(),
                g: (0..n)
                    .map(|l| if l == i { one(prec) } else { zero(prec) })
                    .collect(),
            })
            .collect();
        let ch = series(self.tape, &tj, &hd, p, prec)?;
        let scale = st.yhat.iter().map(mag_f64).fold(1.0, f64::max);
        let remaining = t_end - st.t;
        let mut h = self
            .suggest_h(&c, scale)
            .unwrap_or(remaining)
            .min(remaining);
        if h.is_nan() || h <= 0.0 || !h.is_finite() {
            h = remaining;
        }
        let rem_target = self.tol * scale * 1e3;
        let mut rem_retries = 0;
        let (t_next, hb, encl, rem) = loop {
            let t_next = if st.t + h >= t_end { t_end } else { st.t + h };
            if t_next <= st.t || (h < h_min && t_next < t_end) {
                return Err(VerifiedOdeError::StepSizeUnderflow {
                    t: st.t,
                    h,
                    reason: "the step size fell below h_min".into(),
                });
            }
            let hb = self.ball(t_next) - tj.clone();
            let tb = from_bounds(
                &Float::with_val(prec, st.t),
                &Float::with_val(prec, t_next),
                prec,
            );
            match self.attempt(&hbox, &ch, &tb, &hb) {
                Attempt::Retry(reason) => {
                    let next_h = 0.5 * (t_next - st.t);
                    if next_h < h_min {
                        return Err(VerifiedOdeError::StepSizeUnderflow {
                            t: st.t,
                            h: next_h,
                            reason,
                        });
                    }
                    h = next_h;
                }
                Attempt::Done { encl, rem } => {
                    let hpow = ub(&powers(&hb, p + 1, prec)[p + 1]).to_f64_round(Round::Up);
                    let rw = rem.iter().map(|r| mag_f64(r) * hpow).fold(0.0, f64::max);
                    let next_h = 0.5 * (t_next - st.t);
                    if rw > rem_target && rem_retries < 4 && next_h >= h_min {
                        rem_retries += 1;
                        h = next_h;
                        continue;
                    }
                    break (t_next, hb, encl, rem);
                }
            }
        };
        // Point part u = Σ h^k F_k(ŷ) + h^{p+1} F_{p+1}(B).
        let hpow = powers(&hb, p + 1, prec);
        let u: Vec<ArbBall> = (0..n)
            .map(|i| {
                let mut acc = zero(prec);
                for k in 0..=p {
                    acc = acc.add_ref(&hpow[k].mul_ref(&c[k][i].v));
                }
                acc.add_ref(&hpow[p + 1].mul_ref(&rem[i]))
            })
            .collect();
        // Jacobians J_k over the hull, and S = Σ h^k J_k (J_0 = I).
        let jac = |k: usize| -> BMat {
            if k == 0 {
                return identity(n, prec);
            }
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|l| ch[k][i].g.get(l).cloned().unwrap_or_else(|| zero(prec)))
                        .collect()
                })
                .collect()
        };
        // Pivot the old frame: largest contribution first.
        let mut order: Vec<usize> = (0..n).collect();
        let contrib: Vec<f64> = (0..n)
            .map(|l| {
                let col: f64 = (0..n)
                    .map(|i| st.a[i][l].mid_f64().powi(2))
                    .sum::<f64>()
                    .sqrt();
                col * width(&st.r[l]).to_f64_round(Round::Up)
            })
            .collect();
        order.sort_by(|&x, &y| {
            contrib[y]
                .partial_cmp(&contrib[x])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let a_piv: BMat = (0..n)
            .map(|i| order.iter().map(|&l| st.a[i][l].clone()).collect())
            .collect();
        let r_piv: Vec<ArbBall> = order.iter().map(|&l| st.r[l].clone()).collect();
        let mut ja: Vec<BMat> = Vec::with_capacity(p + 1);
        let mut sa: BMat = a_piv.clone();
        ja.push(a_piv.clone());
        for (k, hk) in hpow.iter().enumerate().take(p + 1).skip(1) {
            let jka = matmul(&jac(k), &a_piv, prec);
            for i in 0..n {
                for l in 0..n {
                    sa[i][l] = sa[i][l].add_ref(&hk.mul_ref(&jka[i][l]));
                }
            }
            ja.push(jka);
        }
        // New centre and frame.
        let yhat_new: Vec<ArbBall> = u
            .iter()
            .map(|b| ArbBall {
                mid: b.mid.clone(),
                rad: Float::new(prec),
                prec,
            })
            .collect();
        let du: Vec<ArbBall> = (0..n).map(|i| u[i].clone() - yhat_new[i].clone()).collect();
        let sar = matvec(&sa, &r_piv, prec);
        let (q, qinv) = frame_and_inverse(&sa, prec);
        let qinv_sa = matmul(&qinv, &sa, prec);
        let mut r_new = matvec(&qinv_sa, &r_piv, prec);
        let qinv_du = matvec(&qinv, &du, prec);
        for i in 0..n {
            r_new[i] = r_new[i].add_ref(&qinv_du[i]);
        }
        let qr = matvec(&q, &r_new, prec);
        let mut y_new = Vec::with_capacity(n);
        for i in 0..n {
            let direct = yhat_new[i].add_ref(&sar[i]).add_ref(&du[i]);
            let framed = yhat_new[i].add_ref(&qr[i]);
            let b = intersect(&direct, &framed).unwrap_or(direct);
            if !is_finite(&b) {
                return Err(VerifiedOdeError::Validated(ValidatedError::NotFinite {
                    what: format!("enclosure of component {i} at t = {t_next}"),
                }));
            }
            y_new.push(b);
        }
        let record = StepRecord {
            t0: st.t,
            t1: t_next,
            coeffs: c
                .iter()
                .take(p + 1)
                .map(|row| row.iter().map(|d| d.v.clone()).collect())
                .collect(),
            ja,
            r: r_piv,
            rem,
            apriori: encl,
        };
        let next = Lohner {
            t: t_next,
            yhat: yhat_new,
            a: q,
            r: r_new,
            y: y_new,
        };
        Ok((record, next))
    }
}

/// Rigorously enclose the solution of `y' = f(t, y)`, `y(t₀) ∈ y0`, on
/// `[t₀, t₁] = t_span`.
///
/// * `rhs[i]` is `f_i`, an expression in the state symbols `vars` and
///   (optionally) the time symbol `time`.
/// * `y0[i] = (lo, hi)` is the initial interval of component `i`; pass
///   `(v, v)` for a point.
/// * `t_span = (t₀, t₁)` with `t₀ < t₁` (forward integration).
///
/// Returns enclosures on an adaptive grid plus dense enclosures anywhere in
/// the span ([`VerifiedOdeSolution::enclosure_at`]); refuses with a coded
/// [`VerifiedOdeError`] rather than return anything unproven.  See the
/// [module documentation](self) for the method and the refusal codes.
pub fn verified_ode(
    rhs: &[ExprId],
    vars: &[ExprId],
    time: Option<ExprId>,
    y0: &[(f64, f64)],
    t_span: (f64, f64),
    pool: &ExprPool,
    opts: &VerifiedOdeOptions,
) -> Result<VerifiedOdeSolution, VerifiedOdeError> {
    let n = vars.len();
    if n == 0 {
        return Err(invalid("verified_ode needs at least one state variable"));
    }
    if rhs.len() != n {
        return Err(invalid(format!(
            "{} right-hand sides for {n} state variables",
            rhs.len()
        )));
    }
    if y0.len() != n {
        return Err(invalid(format!(
            "{} initial values for {n} state variables",
            y0.len()
        )));
    }
    for (i, v) in vars.iter().enumerate() {
        if !matches!(pool.get(*v), ExprData::Symbol { .. }) {
            return Err(invalid(format!("state variable {i} is not a symbol")));
        }
        if vars[..i].contains(v) {
            return Err(invalid(format!("state variable {i} is repeated")));
        }
        if Some(*v) == time {
            return Err(invalid("the time symbol is also a state variable"));
        }
    }
    if let Some(t) = time {
        if !matches!(pool.get(t), ExprData::Symbol { .. }) {
            return Err(invalid("the time variable is not a symbol"));
        }
    }
    for (i, &(lo, hi)) in y0.iter().enumerate() {
        if !(lo.is_finite() && hi.is_finite()) || lo > hi {
            return Err(invalid(format!(
                "initial interval {i} must be finite with lo <= hi (got [{lo}, {hi}])"
            )));
        }
    }
    let (t0, t1) = t_span;
    if !(t0.is_finite() && t1.is_finite()) || t1 <= t0 {
        return Err(invalid(format!(
            "t_span must be finite with t0 < t1 (got ({t0}, {t1})); integrate backwards by substituting t -> -t"
        )));
    }
    if opts.order == 0 || opts.order > MAX_ODE_ORDER {
        return Err(invalid(format!(
            "Taylor order must be in 1..={MAX_ODE_ORDER}"
        )));
    }
    if opts.prec < 53 {
        return Err(invalid("precision must be at least 53 bits"));
    }
    if !(opts.tol > 0.0 && opts.tol.is_finite()) {
        return Err(invalid("tol must be a positive finite number"));
    }
    if opts.max_rel_width.is_nan() || opts.max_rel_width <= 0.0 {
        return Err(invalid("max_rel_width must be positive"));
    }
    let h_min = match opts.h_min {
        Some(h) if h > 0.0 && h.is_finite() => h,
        Some(_) => return Err(invalid("h_min must be a positive finite number")),
        None => 1e-12 * (t1 - t0),
    };
    let prec = opts.prec;
    let tape = compile_tape(rhs, vars, time, pool, prec)?;
    let integ = Integrator {
        tape: &tape,
        n,
        p: opts.order,
        prec,
        tol: opts.tol,
    };

    // Initial Lohner state: ŷ = midpoint, A = I, r = Y₀ - ŷ.
    let y_init: Vec<ArbBall> = y0
        .iter()
        .map(|&(lo, hi)| from_bounds(&Float::with_val(prec, lo), &Float::with_val(prec, hi), prec))
        .collect();
    let yhat: Vec<ArbBall> = y_init
        .iter()
        .map(|b| ArbBall {
            mid: b.mid.clone(),
            rad: Float::new(prec),
            prec,
        })
        .collect();
    let r: Vec<ArbBall> = (0..n)
        .map(|i| y_init[i].clone() - yhat[i].clone())
        .collect();
    let mut st = Lohner {
        t: t0,
        yhat,
        a: identity(n, prec),
        r,
        y: y_init.clone(),
    };
    let mut sol = VerifiedOdeSolution {
        times: vec![t0],
        boxes: vec![y_init],
        steps: Vec::new(),
        order: opts.order,
        prec,
    };
    while st.t < t1 {
        budget::check_all().map_err(VerifiedOdeError::Budget)?;
        if sol.steps.len() >= opts.max_steps {
            return Err(VerifiedOdeError::StepLimit {
                t: st.t,
                steps: opts.max_steps,
            });
        }
        let (rec, next) = integ.step(&st, t1, h_min)?;
        for (i, b) in next.y.iter().enumerate() {
            let w = width(b).to_f64_round(Round::Up);
            let lim = opts.max_rel_width * mag_f64(b).max(1.0);
            if w.is_nan() || w > lim {
                return Err(VerifiedOdeError::EnclosureTooWide {
                    t: next.t,
                    component: i,
                    width: w,
                });
            }
        }
        sol.times.push(next.t);
        sol.boxes.push(next.y.clone());
        sol.steps.push(rec);
        st = next;
    }
    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn neg(pool: &ExprPool, a: ExprId) -> ExprId {
        pool.mul(vec![pool.integer(-1_i32), a])
    }

    fn contains(b: &ArbBall, v: f64) -> bool {
        b.lo() <= v && v <= b.hi()
    }

    #[test]
    fn exponential_decay_encloses_exp() {
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let rhs = neg(&pool, y);
        let sol = verified_ode(
            &[rhs],
            &[y],
            None,
            &[(1.0, 1.0)],
            (0.0, 5.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        for (t, b) in sol.times().iter().zip(sol.enclosures()) {
            let truth = Float::with_val(200, -t).exp();
            assert!(b[0].lo() <= truth && truth <= b[0].hi(), "t={t} {:?}", b[0]);
            assert!(width(&b[0]) < 1e-14, "too wide at t={t}: {}", b[0]);
        }
        for k in 0..50 {
            let t = 0.1 * k as f64 + 0.037;
            let e = sol.enclosure_at(t).unwrap();
            // Compare against exp at high precision.
            let truth = Float::with_val(200, -t).exp();
            assert!(e[0].lo() <= truth && truth <= e[0].hi(), "t={t}");
            assert!(width(&e[0]) < 1e-13);
        }
    }

    #[test]
    fn rotation_stays_tight() {
        // x' = -y, y' = x : rotation; the wrapping effect would blow up a naive
        // interval method, the QR frame keeps it tight.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let sol = verified_ode(
            &[neg(&pool, y), x],
            &[x, y],
            None,
            &[(1.0, 1.0), (0.0, 0.0)],
            (0.0, 20.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        for (t, b) in sol.times().iter().zip(sol.enclosures()) {
            let tt = Float::with_val(200, *t);
            let (c, s) = (
                Float::with_val(200, tt.cos_ref()),
                Float::with_val(200, tt.sin_ref()),
            );
            assert!(b[0].lo() <= c && c <= b[0].hi(), "t={t}");
            assert!(b[1].lo() <= s && s <= b[1].hi(), "t={t}");
        }
        assert!(sol.max_width() < 1e-12, "max width {}", sol.max_width());
    }

    #[test]
    fn interval_initial_condition_encloses_every_solution() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let sol = verified_ode(
            &[neg(&pool, y), x],
            &[x, y],
            None,
            &[(0.9, 1.1), (-0.1, 0.1)],
            (0.0, 6.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        // Corners of the initial box, rotated.
        let end = sol.enclosures().last().unwrap();
        let t = 6.0_f64;
        for &(x0, y0) in &[(0.9, -0.1), (0.9, 0.1), (1.1, -0.1), (1.1, 0.1), (1.0, 0.0)] {
            let xt = x0 * t.cos() - y0 * t.sin();
            let yt = x0 * t.sin() + y0 * t.cos();
            assert!(contains(&end[0], xt) && contains(&end[1], yt));
        }
        // The rotated box of half-width 0.1 fits in a box of half-width
        // 0.1·√2: the frame avoids the 2^k wrapping growth.
        assert!(width(&end[0]) < 0.3 && width(&end[1]) < 0.3);
    }

    #[test]
    fn logistic_matches_closed_form() {
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        // y' = y (1 - y), y(0) = 1/2  ⇒  y = 1 / (1 + e^{-t})
        let rhs = pool.mul(vec![y, pool.add(vec![pool.integer(1_i32), neg(&pool, y)])]);
        let sol = verified_ode(
            &[rhs],
            &[y],
            None,
            &[(0.5, 0.5)],
            (0.0, 10.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        for k in 0..=100 {
            let t = 0.1 * k as f64;
            let e = sol.enclosure_at(t).unwrap();
            let tt = Float::with_val(200, -t);
            let truth = Float::with_val(200, 1) / (Float::with_val(200, 1) + tt.exp());
            assert!(e[0].lo() <= truth && truth <= e[0].hi(), "t={t}");
            assert!(width(&e[0]) < 1e-13);
        }
    }

    #[test]
    fn time_dependent_rhs() {
        // y' = cos(t) y, y(0) = 1 ⇒ y = exp(sin t)
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let rhs = pool.mul(vec![pool.func("cos", vec![t]), y]);
        let sol = verified_ode(
            &[rhs],
            &[y],
            Some(t),
            &[(1.0, 1.0)],
            (0.0, 4.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        for (tv, b) in sol.times().iter().zip(sol.enclosures()) {
            let truth = Float::with_val(200, Float::with_val(200, *tv).sin()).exp();
            assert!(b[0].lo() <= truth && truth <= b[0].hi(), "t={tv}");
        }
    }

    #[test]
    fn lorenz_moderate_horizon() {
        // σ = 10, ρ = 28, β = 8/3 on [0, 2]; reference: RK4 with 200 000
        // steps (error ~1e-11 at this horizon), compared with 1e-8 slack.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let fx = pool.mul(vec![pool.integer(10_i32), pool.add(vec![y, neg(&pool, x)])]);
        let fy = pool.add(vec![
            pool.mul(vec![x, pool.add(vec![pool.integer(28_i32), neg(&pool, z)])]),
            neg(&pool, y),
        ]);
        let fz = pool.add(vec![
            pool.mul(vec![x, y]),
            neg(&pool, pool.mul(vec![pool.rational(8, 3), z])),
        ]);
        let started = std::time::Instant::now();
        let sol = verified_ode(
            &[fx, fy, fz],
            &[x, y, z],
            None,
            &[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            (0.0, 2.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        eprintln!(
            "lorenz: {} steps, max width {:e}, {:?}",
            sol.steps(),
            sol.max_width(),
            started.elapsed()
        );
        let f = |v: [f64; 3]| {
            [
                10.0 * (v[1] - v[0]),
                v[0] * (28.0 - v[2]) - v[1],
                v[0] * v[1] - 8.0 / 3.0 * v[2],
            ]
        };
        let mut v = [1.0, 1.0, 1.0];
        let nsteps = 200_000;
        let hh = 2.0 / nsteps as f64;
        for _ in 0..nsteps {
            let k1 = f(v);
            let k2 = f([0, 1, 2].map(|i| v[i] + 0.5 * hh * k1[i]));
            let k3 = f([0, 1, 2].map(|i| v[i] + 0.5 * hh * k2[i]));
            let k4 = f([0, 1, 2].map(|i| v[i] + hh * k3[i]));
            for i in 0..3 {
                v[i] += hh / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
        }
        let end = sol.enclosures().last().unwrap();
        for i in 0..3 {
            let lo = end[i].lo().to_f64() - 1e-8;
            let hi = end[i].hi().to_f64() + 1e-8;
            assert!(
                lo <= v[i] && v[i] <= hi,
                "component {i}: {} vs {}",
                end[i],
                v[i]
            );
        }
        assert!(sol.max_width() < 1e-9, "max width {:e}", sol.max_width());
    }

    #[test]
    fn blow_up_is_refused() {
        // y' = y², y(0) = 1 blows up at t = 1.
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let rhs = pool.pow(y, pool.integer(2_i32));
        let err = verified_ode(
            &[rhs],
            &[y],
            None,
            &[(1.0, 1.0)],
            (0.0, 2.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err.code(),
                "E-VALIDATED-006" | "E-VALIDATED-007" | "E-VALIDATED-004"
            ),
            "{err}"
        );
        if let VerifiedOdeError::StepSizeUnderflow { t, .. } = err {
            assert!(t < 1.0 && t > 0.99, "stopped at {t}");
        }
        // Before the blow-up the enclosures are fine and contain 1/(1-t).
        let sol = verified_ode(
            &[rhs],
            &[y],
            None,
            &[(1.0, 1.0)],
            (0.0, 0.9),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        for (t, b) in sol.times().iter().zip(sol.enclosures()) {
            let truth = Float::with_val(200, 1) / (Float::with_val(200, 1) - *t);
            assert!(b[0].lo() <= truth && truth <= b[0].hi(), "t={t}");
        }
    }

    #[test]
    fn stiff_problem_hits_step_limit() {
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let rhs = pool.mul(vec![pool.integer(-100_000_000_i64), y]);
        let opts = VerifiedOdeOptions {
            max_steps: 50,
            ..Default::default()
        };
        let err =
            verified_ode(&[rhs], &[y], None, &[(1.0, 1.0)], (0.0, 1.0), &pool, &opts).unwrap_err();
        assert_eq!(err.code(), "E-VALIDATED-008", "{err}");
    }

    #[test]
    fn domain_and_unsupported_refusals() {
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let opts = VerifiedOdeOptions::default();
        // log(y) at y = -1.
        let e = verified_ode(
            &[pool.func("log", vec![y])],
            &[y],
            None,
            &[(-1.0, -1.0)],
            (0.0, 1.0),
            &pool,
            &opts,
        )
        .unwrap_err();
        assert_eq!(e.code(), "E-VALIDATED-003");
        // Unbound symbol.
        let e =
            verified_ode(&[z], &[y], None, &[(1.0, 1.0)], (0.0, 1.0), &pool, &opts).unwrap_err();
        assert_eq!(e.code(), "E-VALIDATED-002");
        // Unsupported function.
        let e = verified_ode(
            &[pool.func("floor", vec![y])],
            &[y],
            None,
            &[(1.0, 1.0)],
            (0.0, 1.0),
            &pool,
            &opts,
        )
        .unwrap_err();
        assert_eq!(e.code(), "E-VALIDATED-001");
        // sqrt(y) at y = 0 is not Lipschitz: refuse, never pick one solution.
        let e = verified_ode(
            &[pool.func("sqrt", vec![y])],
            &[y],
            None,
            &[(0.0, 0.0)],
            (0.0, 1.0),
            &pool,
            &opts,
        )
        .unwrap_err();
        assert!(
            matches!(e.code(), "E-VALIDATED-003" | "E-VALIDATED-006"),
            "{e}"
        );
        // Malformed.
        let e =
            verified_ode(&[y], &[y], None, &[(2.0, 1.0)], (0.0, 1.0), &pool, &opts).unwrap_err();
        assert_eq!(e.code(), "E-VALIDATED-005");
        let e =
            verified_ode(&[y], &[y], None, &[(1.0, 1.0)], (1.0, 0.0), &pool, &opts).unwrap_err();
        assert_eq!(e.code(), "E-VALIDATED-005");
    }

    #[test]
    fn budget_is_honoured() {
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let _g = crate::budget::enter(crate::budget::Budget::new().with_max_steps(3));
        let err = verified_ode(
            &[neg(&pool, y)],
            &[y],
            None,
            &[(1.0, 1.0)],
            (0.0, 100.0),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap_err();
        assert!(err.code().starts_with("E-BUDGET"), "{err}");
    }

    #[test]
    fn elementary_rhs_series_are_consistent() {
        // y' = atan(y) + sinh(y)/cosh(y) - tan(y)·0 + sqrt(y) + y^(3/2) etc:
        // just check the enclosure contains a high-accuracy RK solution of
        // y' = exp(-y) + sin(y), y(0) = 0.5 computed by a tiny-step reference.
        let pool = ExprPool::new();
        let y = pool.symbol("y", Domain::Real);
        let rhs = pool.add(vec![
            pool.func("exp", vec![neg(&pool, y)]),
            pool.func("sin", vec![y]),
            pool.func("atan", vec![y]),
            pool.func("tanh", vec![y]),
            pool.pow(y, pool.rational(3, 2)),
        ]);
        let sol = verified_ode(
            &[rhs],
            &[y],
            None,
            &[(0.5, 0.5)],
            (0.0, 0.5),
            &pool,
            &VerifiedOdeOptions::default(),
        )
        .unwrap();
        // Reference: classical RK4 with 20000 steps in f64.
        let f = |v: f64| (-v).exp() + v.sin() + v.atan() + v.tanh() + v.powf(1.5);
        let mut v = 0.5;
        let hh = 0.5 / 20000.0;
        for _ in 0..20000 {
            let k1 = f(v);
            let k2 = f(v + 0.5 * hh * k1);
            let k3 = f(v + 0.5 * hh * k2);
            let k4 = f(v + hh * k3);
            v += hh / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        let end = &sol.enclosures().last().unwrap()[0];
        let lo = end.lo().to_f64() - 1e-9;
        let hi = end.hi().to_f64() + 1e-9;
        assert!(lo <= v && v <= hi, "{end} vs {v}");
        assert!(width(end) < 1e-12);
    }
}
