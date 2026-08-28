//! Deciding the boundary hypothesis an Almkvist–Zeilberger certificate rests on.
//!
//! [`super::almkvist_zeilberger`] proves an identity about the *integrand*:
//!
//! ```text
//! Σ_i a_i(n)·F(n+i, x) = D_x( G(n,x) ),    G(n,x) = R(n,x)·F(n,x)
//! ```
//!
//! Turning that into a statement about `f(n) = ∫_a^b F(n,x) dx` is a second,
//! separate step, and it is where a valid certificate can produce a false
//! theorem. This module takes it, three-valued, in the same discipline as
//! [`mod@super::super::boundary`]:
//!
//! | [`IntegralBoundaryStatus`] | claim |
//! |---|---|
//! | [`Vanishes`](IntegralBoundaryStatus::Vanishes) | proved `[G]_a^b = 0`; the homogeneous recurrence `Σ_i a_i(n)·f(n+i) = 0` holds |
//! | [`Nonzero`](IntegralBoundaryStatus::Nonzero) | `[G]_a^b` computed exactly and proved `≢ 0`; the *inhomogeneous* recurrence holds, with the right-hand side explicit |
//! | [`Unknown`](IntegralBoundaryStatus::Unknown) | neither was established; **nothing** may be claimed about the integral |
//!
//! Only the first two are results. `Unknown` carries a reason, so a caller can
//! tell "the boundary term diverges" from "both endpoints are finite and their
//! difference could not be decided" and act differently. `Unknown` is never
//! collapsed into `Vanishes`.
//!
//! # Why this is simpler than the discrete case, and where it is not
//!
//! Simpler: the limits do not move with `n`, so `∫_a^b F(n+i,x) dx` *is*
//! `f(n+i)` and the whole inhomogeneity is the endpoint difference. The
//! discrete module's `D_i(n)` correction terms — the reason
//! `Σ_{k=0}^{n} C(n,k)` comes out right — have no analogue here, because a
//! continuous integration range that moved with `n` would not be a
//! creative-telescoping problem in the first place (it would need a Leibniz
//! rule term this module does not model, which is why `n`-dependent limits are
//! refused rather than mishandled).
//!
//! Harder: an endpoint is a *limit*, not an evaluation, and it can diverge. The
//! order of `G` at an endpoint is generally an element of `Q(n)` — `n + 1` for
//! `G = x^(n+1)·e^(−x)` at `0` — so "does it vanish" is a question about `n`,
//! not a yes/no. That is reported, never assumed: see
//! [`IntegralBoundaryStatus::conditions`].
//!
//! # How a verdict is reached
//!
//! Nothing here is numeric. Write `G = r(n,x)·wⁿ·exp(η(x))·∏_j B_j(x)^(e_j)`
//! with `e_j = α_j·n + β_j`, which is what [`super::hyperexp`] already parses,
//! with `r` the certificate times the integrand's own rational part.
//!
//! **At a finite rational endpoint `c`.** If `η` has a pole at `c`, the verdict
//! is `Unknown` — `exp` of a pole is `0` on one side and `∞` on the other, and
//! deciding which needs an interval this module is not given. Otherwise `exp(η)`
//! contributes a finite nonzero factor, and the order of `G` at `c` is
//!
//! ```text
//! ν = ord_c(r) + Σ_j e_j·ord_c(B_j)   ∈ Q(n), affine in n,
//! ```
//!
//! computed by exact deflation of the root `c` over `Q(n)`. `G → 0` exactly
//! when `ν > 0`; when `ν` is a concrete positive rational that is
//! unconditional, and when it carries `n` it becomes a stated condition. `ν = 0`
//! with nothing vanishing or blowing up gives the endpoint value by direct
//! substitution — a product of a nonzero `Q(n)` element, `wⁿ`, `exp` of a
//! finite value and nonzero powers, hence *provably* nonzero. Anything else
//! (`ν < 0`, or `ν = 0` reached by cancellation between a zero of `r` and a pole
//! of some `B_j`) is `Unknown`.
//!
//! **At `±∞`.** If `deg η ≥ 1` with a leading coefficient that is a concrete
//! rational, `exp(η)` decides everything: it beats every power of `x`, so the
//! endpoint contributes `0` unconditionally when `η → −∞` and `Unknown`
//! (divergence) when `η → +∞`. Otherwise `exp(η)` tends to a finite nonzero
//! constant and the endpoint is decided by the algebraic degree
//! `δ = deg(r) + Σ_j e_j·deg(B_j) ∈ Q(n)`, with `G → 0` exactly when `δ < 0`.
//!
//! # The residual hypothesis
//!
//! Orders and degrees are computed over `Q(n)`, so a coefficient that is a
//! nonzero element of `Q(n)` is nonzero for all but finitely many `n`. A verdict
//! is therefore a statement about the `n` at which the integrand, the
//! certificate and the integral are all defined and the integral converges —
//! stated rather than hidden, in [`IntegralBoundaryStatus::side_conditions`].
//! Convergence in particular is *not* established here: this module decides the
//! boundary term of an identity, not the existence of `f(n)`.

use super::hyperexp::HyperExpTerm;
use super::search::AzResult;
use super::DiffTelescopingError;
use crate::holonomic::hyperterm::{as_ratk, rn_to_expr};
use crate::holonomic::qfield::{
    rn_add, rn_int, rn_is_zero, rn_mul, rn_rat, rn_var, rn_zero, PolyK, RatK, Rn,
};
use crate::kernel::{ExprId, ExprPool};
use rug::Rational;

/// One end of the integration range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrationLimit {
    /// A finite rational endpoint.
    Finite(Rational),
    /// `−∞`.
    NegInfinity,
    /// `+∞`.
    PosInfinity,
}

impl IntegrationLimit {
    /// A finite endpoint from an integer, for the common cases `0` and `1`.
    pub fn at(v: i64) -> IntegrationLimit {
        IntegrationLimit::Finite(Rational::from(v))
    }

    fn describe(&self) -> String {
        match self {
            IntegrationLimit::Finite(q) => q.to_string(),
            IntegrationLimit::NegInfinity => "-oo".into(),
            IntegrationLimit::PosInfinity => "+oo".into(),
        }
    }
}

/// The verdict on `[G]_a^b`, the inhomogeneity of the recurrence for the
/// *integral*.
///
/// See the [module documentation](self) for what each variant is allowed to
/// mean and why.
#[derive(Debug, Clone)]
pub enum IntegralBoundaryStatus {
    /// Proved `[G]_a^b = 0`: the homogeneous recurrence
    /// `Σ_i a_i(n)·f(n+i) = 0` holds over the range that was supplied, for
    /// every `n` satisfying `conditions`.
    Vanishes {
        /// Each entry is an expression in `n` that must be **strictly
        /// positive** for the proof to apply. Empty means unconditional.
        ///
        /// These are not caveats bolted on afterwards: they are the exact
        /// inequalities the order computation produced, e.g. `n + 1 > 0` for
        /// `x^(n+1)·e^(−x)` at `0`.
        conditions: Vec<ExprId>,
    },
    /// `[G]_a^b` was computed exactly and proved not identically zero. The
    /// **inhomogeneous** recurrence `Σ_i a_i(n)·f(n+i) = rhs(n)` holds — still a
    /// theorem about the integral, just not the homogeneous one.
    Nonzero {
        /// `[G]_a^b`, as an expression in `n` alone.
        rhs: ExprId,
        /// Conditions on `n`, as in [`Vanishes`](IntegralBoundaryStatus::Vanishes).
        conditions: Vec<ExprId>,
    },
    /// Neither could be established. **No** recurrence for the integral
    /// follows; the certificate remains a true statement about the integrand
    /// and nothing more.
    Unknown {
        /// Why the verdict could not be reached.
        reason: String,
    },
}

impl IntegralBoundaryStatus {
    /// `"vanishes"`, `"nonzero"` or `"unknown"` — the stable tag to record.
    pub fn tag(&self) -> &'static str {
        match self {
            IntegralBoundaryStatus::Vanishes { .. } => "vanishes",
            IntegralBoundaryStatus::Nonzero { .. } => "nonzero",
            IntegralBoundaryStatus::Unknown { .. } => "unknown",
        }
    }

    /// Whether a recurrence for the *integral* may be read off at all — true
    /// for [`Vanishes`](IntegralBoundaryStatus::Vanishes) (homogeneous) and
    /// [`Nonzero`](IntegralBoundaryStatus::Nonzero) (inhomogeneous), false for
    /// [`Unknown`](IntegralBoundaryStatus::Unknown).
    pub fn implies_integral_recurrence(&self) -> bool {
        !matches!(self, IntegralBoundaryStatus::Unknown { .. })
    }

    /// The positivity conditions on `n` this verdict is conditional on, or an
    /// empty slice.
    pub fn conditions(&self) -> &[ExprId] {
        match self {
            IntegralBoundaryStatus::Vanishes { conditions }
            | IntegralBoundaryStatus::Nonzero { conditions, .. } => conditions,
            IntegralBoundaryStatus::Unknown { .. } => &[],
        }
    }

    /// What is still assumed after this verdict, as plain strings — the same
    /// shape as the discrete module's
    /// [`side_conditions`](super::super::boundary::BoundaryStatus::side_conditions).
    ///
    /// This is *not* a fixed string: a discharged hypothesis and an open one
    /// read differently, which is the whole point of computing a verdict rather
    /// than restating the caveat.
    pub fn side_conditions(&self, range: &str, pool: &ExprPool) -> Vec<String> {
        let render = |conds: &[ExprId]| -> Vec<String> {
            conds
                .iter()
                .map(|c| format!("{} > 0", pool.display(*c)))
                .collect()
        };
        match self {
            IntegralBoundaryStatus::Vanishes { conditions } => {
                let mut out = vec![format!(
                    "the boundary term [R*F] over {range} was proved to vanish in exact \
                     arithmetic, so the homogeneous recurrence sum_i a_i(n)*f(n+i) = 0 holds \
                     for the integral"
                )];
                out.extend(render(conditions));
                out.push(
                    "the verdict is a statement about the n at which the integrand, the \
                     certificate and the integral are all defined; convergence of the \
                     integral itself is assumed, not proved here"
                        .into(),
                );
                out
            }
            IntegralBoundaryStatus::Nonzero { conditions, .. } => {
                let mut out = vec![format!(
                    "the boundary term [R*F] over {range} was computed exactly and is not \
                     identically zero, so the recurrence for the integral is inhomogeneous: \
                     sum_i a_i(n)*f(n+i) = rhs(n)"
                )];
                out.extend(render(conditions));
                out.push(
                    "the verdict is a statement about the n at which the integrand, the \
                     certificate and the integral are all defined; convergence of the \
                     integral itself is assumed, not proved here"
                        .into(),
                );
                out
            }
            IntegralBoundaryStatus::Unknown { reason } => vec![format!(
                "no recurrence for the integral over {range} may be claimed: {reason}. The \
                 certificate remains a true identity about the integrand"
            )],
        }
    }
}

/// Decide the boundary hypothesis for `f(n) = ∫_lower^upper F(n,x) dx`.
///
/// `term` must be the same `F(n,x)` that produced `result`, and `n`, `x` the
/// same symbols. The limits must not depend on `n` — an `n`-dependent limit
/// would add a Leibniz term this module does not model, so the type does not
/// admit one.
///
/// Never panics on user input; an input it cannot analyse becomes
/// [`IntegralBoundaryStatus::Unknown`], and a `term` that is not the parsed
/// integrand is a [`DiffTelescopingError`].
pub fn integral_boundary_status(
    result: &AzResult,
    term: ExprId,
    n: ExprId,
    x: ExprId,
    pool: &ExprPool,
    lower: &IntegrationLimit,
    upper: &IntegrationLimit,
) -> Result<IntegralBoundaryStatus, DiffTelescopingError> {
    let f = HyperExpTerm::parse(term, n, x, pool)?;
    let r = as_ratk(result.certificate, n, x, pool, 0).ok_or_else(|| {
        DiffTelescopingError::InvalidInput(
            "the certificate is not a rational function of the two symbols supplied; it must \
             come from the same almkvist_zeilberger call as `term`"
                .into(),
        )
    })?;
    // G = R·F: the certificate folds into the rational prefactor and nothing
    // else changes, which is exactly why the endpoint analysis below can be
    // written once for the whole class.
    let g = HyperExpTerm {
        rat: f.rat.mul(&r),
        w: f.w.clone(),
        eta: f.eta.clone(),
        powers: f.powers.clone(),
    };
    if g.rat.is_zero() {
        // R·F ≡ 0 is a perfectly good (if degenerate) antiderivative.
        return Ok(IntegralBoundaryStatus::Vanishes { conditions: vec![] });
    }

    let hi = endpoint_limit(&g, upper, n, pool);
    let lo = endpoint_limit(&g, lower, n, pool);
    Ok(combine(hi, lo, lower, upper, pool))
}

// ---------------------------------------------------------------------------
// Endpoint analysis
// ---------------------------------------------------------------------------

/// What `G` does at one endpoint. `Finite` is *provably nonzero*: it is only
/// produced when every factor's value is a nonzero element of its own field.
enum EndpointLimit {
    Zero { conditions: Vec<ExprId> },
    Finite(ExprId),
    Unknown(String),
}

fn endpoint_limit(
    g: &HyperExpTerm,
    limit: &IntegrationLimit,
    n: ExprId,
    pool: &ExprPool,
) -> EndpointLimit {
    match limit {
        IntegrationLimit::Finite(c) => finite_endpoint(g, c, n, pool),
        IntegrationLimit::PosInfinity => infinite_endpoint(g, true, n, pool),
        IntegrationLimit::NegInfinity => infinite_endpoint(g, false, n, pool),
    }
}

fn finite_endpoint(g: &HyperExpTerm, c: &Rational, n: ExprId, pool: &ExprPool) -> EndpointLimit {
    // exp(η): a pole at the endpoint makes the limit direction-dependent (0 on
    // one side, ∞ on the other), which this module refuses to guess at.
    let Some(eta_val) = ratk_eval(&g.eta, c) else {
        return EndpointLimit::Unknown(format!(
            "the exponential argument has a pole at x = {c}, so exp(eta) has no two-sided limit \
             there"
        ));
    };

    let Some((rat_ord, rat_val)) = ratk_order_and_value(&g.rat, c) else {
        return EndpointLimit::Unknown(format!("the rational part of R*F degenerates at x = {c}"));
    };

    // ν = ord(r) + Σ_j e_j·ord(B_j), an element of Q(n) affine in n.
    let mut nu = rn_int(rat_ord);
    let mut any_power_singular = false;
    let mut power_vals: Vec<(Rn, Rn)> = Vec::with_capacity(g.powers.len());
    for p in &g.powers {
        let Some((ord, val)) = ratk_order_and_value(&p.base, c) else {
            return EndpointLimit::Unknown(format!("a power factor's base degenerates at x = {c}"));
        };
        let e = rn_add(
            &rn_mul(&rn_int(p.alpha), &rn_var()),
            &rn_rat(p.beta.clone()),
        );
        if ord != 0 {
            any_power_singular = true;
            nu = rn_add(&nu, &rn_mul(&e, &rn_int(ord)));
        }
        power_vals.push((e, val));
    }

    match positivity(&nu, n, pool) {
        Positivity::Positive => EndpointLimit::Zero { conditions: vec![] },
        Positivity::Conditional(cond) => EndpointLimit::Zero {
            conditions: vec![cond],
        },
        Positivity::Zero => {
            if any_power_singular || rat_ord != 0 {
                // ν = 0 reached by cancellation: the leading coefficient is a
                // product of *limits* of vanishing factors raised to symbolic
                // powers, which is not something to guess at.
                return EndpointLimit::Unknown(format!(
                    "the order of R*F at x = {c} is zero only through cancellation between a \
                     vanishing rational part and a vanishing power factor; the limiting value \
                     is not determined by this analysis"
                ));
            }
            EndpointLimit::Finite(endpoint_value(g, &rat_val, &eta_val, &power_vals, n, pool))
        }
        Positivity::Negative => EndpointLimit::Unknown(format!(
            "R*F is unbounded as x -> {c} (order {}), so the boundary term does not exist",
            pool.display(rn_to_expr(pool, n, &nu))
        )),
        Positivity::Undecided => EndpointLimit::Unknown(format!(
            "the order of R*F at x = {c} is {}, whose sign is not an affine condition on n",
            pool.display(rn_to_expr(pool, n, &nu))
        )),
    }
}

fn infinite_endpoint(g: &HyperExpTerm, at_plus: bool, n: ExprId, pool: &ExprPool) -> EndpointLimit {
    let where_ = if at_plus { "+oo" } else { "-oo" };
    // exp(η) dominates every power of x, so when η has a genuine polynomial
    // part its sign decides the endpoint outright.
    let eta_deg = ratk_degree(&g.eta);
    if eta_deg >= 1 {
        let Some(lead) = ratk_leading_rational(&g.eta) else {
            return EndpointLimit::Unknown(format!(
                "the exponential argument grows at {where_} with a leading coefficient that \
                 depends on n, so the direction of exp(eta) is not decided"
            ));
        };
        let sign_at_end = if at_plus || eta_deg % 2 == 0 {
            lead.clone()
        } else {
            -lead.clone()
        };
        return if sign_at_end < 0 {
            // exp of something tending to −∞ beats any power of x, for every n.
            EndpointLimit::Zero { conditions: vec![] }
        } else {
            EndpointLimit::Unknown(format!(
                "R*F is unbounded as x -> {where_}: the exponential argument tends to +oo"
            ))
        };
    }

    // η tends to a finite value, so exp(η) is a nonzero constant and the
    // algebraic degree decides.
    let mut delta = rn_int(ratk_degree(&g.rat));
    for p in &g.powers {
        let e = rn_add(
            &rn_mul(&rn_int(p.alpha), &rn_var()),
            &rn_rat(p.beta.clone()),
        );
        delta = rn_add(&delta, &rn_mul(&e, &rn_int(ratk_degree(&p.base))));
    }
    // G → 0 at infinity exactly when δ < 0, i.e. when −δ > 0.
    let neg_delta = rn_mul(&delta, &rn_int(-1));
    match positivity(&neg_delta, n, pool) {
        Positivity::Positive => EndpointLimit::Zero { conditions: vec![] },
        Positivity::Conditional(cond) => EndpointLimit::Zero {
            conditions: vec![cond],
        },
        Positivity::Zero | Positivity::Negative => EndpointLimit::Unknown(format!(
            "R*F does not tend to 0 as x -> {where_}: its algebraic degree there is {}",
            pool.display(rn_to_expr(pool, n, &delta))
        )),
        Positivity::Undecided => EndpointLimit::Unknown(format!(
            "the degree of R*F at {where_} is {}, whose sign is not an affine condition on n",
            pool.display(rn_to_expr(pool, n, &delta))
        )),
    }
}

/// `G(c)` when nothing vanishes or blows up there: a product of a nonzero
/// `Q(n)` element, `wⁿ`, `exp` of a finite value and nonzero powers.
fn endpoint_value(
    g: &HyperExpTerm,
    rat_val: &Rn,
    eta_val: &Rn,
    power_vals: &[(Rn, Rn)],
    n: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let mut factors = vec![rn_to_expr(pool, n, rat_val)];
    if g.w != 1 {
        factors.push(pool.pow(rational_expr(pool, &g.w), n));
    }
    if !rn_is_zero(eta_val) {
        factors.push(pool.func("exp", vec![rn_to_expr(pool, n, eta_val)]));
    }
    for (e, val) in power_vals {
        // A zero exponent or a base of 1 contributes nothing; dropping them
        // keeps `rhs` readable rather than littered with `1**n`.
        if rn_is_zero(e) || val.eq(&crate::holonomic::qfield::rn_one()) {
            continue;
        }
        factors.push(pool.pow(rn_to_expr(pool, n, val), rn_to_expr(pool, n, e)));
    }
    crate::simplify::simplify(pool.mul(factors), pool).value
}

fn combine(
    hi: EndpointLimit,
    lo: EndpointLimit,
    lower: &IntegrationLimit,
    upper: &IntegrationLimit,
    pool: &ExprPool,
) -> IntegralBoundaryStatus {
    let range = format!("x = {}..{}", lower.describe(), upper.describe());
    match (hi, lo) {
        (EndpointLimit::Unknown(r), _) => IntegralBoundaryStatus::Unknown {
            reason: format!("at the upper limit of {range}: {r}"),
        },
        (_, EndpointLimit::Unknown(r)) => IntegralBoundaryStatus::Unknown {
            reason: format!("at the lower limit of {range}: {r}"),
        },
        (EndpointLimit::Zero { conditions: mut a }, EndpointLimit::Zero { conditions: b }) => {
            for c in b {
                if !a.contains(&c) {
                    a.push(c);
                }
            }
            IntegralBoundaryStatus::Vanishes { conditions: a }
        }
        (EndpointLimit::Finite(v), EndpointLimit::Zero { conditions }) => {
            IntegralBoundaryStatus::Nonzero { rhs: v, conditions }
        }
        (EndpointLimit::Zero { conditions }, EndpointLimit::Finite(v)) => {
            IntegralBoundaryStatus::Nonzero {
                rhs: crate::simplify::simplify(pool.mul(vec![pool.integer(-1_i32), v]), pool).value,
                conditions,
            }
        }
        (EndpointLimit::Finite(u), EndpointLimit::Finite(l)) => {
            if u == l {
                // Structurally the same value at both ends: the difference is
                // exactly zero, no numerics involved.
                IntegralBoundaryStatus::Vanishes { conditions: vec![] }
            } else {
                IntegralBoundaryStatus::Unknown {
                    reason: format!(
                        "both endpoints of {range} are finite and nonzero ({} and {}), and their \
                         difference could not be shown to be nonzero without leaving exact \
                         arithmetic",
                        pool.display(u),
                        pool.display(l)
                    ),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exact `Q(n)` helpers
// ---------------------------------------------------------------------------

/// The sign of an element of `Q(n)` that is affine in `n`, as far as it can be
/// decided without knowing `n`.
enum Positivity {
    Positive,
    Zero,
    Negative,
    /// Positive exactly when the carried expression is `> 0`.
    Conditional(ExprId),
    /// Not an affine condition in `n` — nothing is claimed.
    Undecided,
}

fn positivity(v: &Rn, n: ExprId, pool: &ExprPool) -> Positivity {
    if v.den.degree() != 0 {
        return Positivity::Undecided;
    }
    match v.num.degree() {
        d if d < 0 => Positivity::Zero,
        0 => {
            let Some(q) = rn_as_rational(v) else {
                return Positivity::Undecided;
            };
            if q > 0 {
                Positivity::Positive
            } else if q == 0 {
                Positivity::Zero
            } else {
                Positivity::Negative
            }
        }
        1 => Positivity::Conditional(rn_to_expr(pool, n, v)),
        _ => Positivity::Undecided,
    }
}

fn rn_as_rational(v: &Rn) -> Option<Rational> {
    if v.num.degree() > 0 || v.den.degree() > 0 {
        return None;
    }
    let a = v
        .num
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(|| Rational::from(0));
    let b = v.den.coeffs.first().cloned()?;
    if b == 0 {
        return None;
    }
    Some(a / b)
}

fn rational_expr(pool: &ExprPool, q: &Rational) -> ExprId {
    if *q.clone().denom() == 1 {
        pool.integer(q.numer().clone())
    } else {
        pool.mul(vec![
            pool.integer(q.numer().clone()),
            pool.pow(pool.integer(q.denom().clone()), pool.integer(-1_i32)),
        ])
    }
}

/// `p(c)` for `p ∈ Q(n)[x]` at a rational `c`, by Horner over `Q(n)`.
fn polyk_eval(p: &PolyK, c: &Rational) -> Rn {
    let cc = rn_rat(c.clone());
    let mut acc = rn_zero();
    for coeff in p.coeffs.iter().rev() {
        acc = rn_add(&rn_mul(&acc, &cc), coeff);
    }
    acc
}

/// `p(x)/(x − c)` when `p(c) = 0`, by synthetic division over `Q(n)`.
fn polyk_deflate(p: &PolyK, c: &Rational) -> PolyK {
    let d = p.degree();
    if d <= 0 {
        return PolyK::zero();
    }
    let d = d as usize;
    let cc = rn_rat(c.clone());
    let mut q = vec![rn_zero(); d];
    q[d - 1] = p.coeff(d);
    for i in (1..d).rev() {
        q[i - 1] = rn_add(&p.coeff(i), &rn_mul(&cc, &q[i]));
    }
    PolyK::from_coeffs(q)
}

/// `(multiplicity of the root c, value of p/(x−c)^mult at c)`.
///
/// `None` for the zero polynomial, which has no well-defined order.
fn polyk_order_and_value(p: &PolyK, c: &Rational) -> Option<(i64, Rn)> {
    if p.is_zero() {
        return None;
    }
    let mut cur = p.clone();
    let mut m = 0i64;
    loop {
        let val = polyk_eval(&cur, c);
        if !rn_is_zero(&val) {
            return Some((m, val));
        }
        if cur.degree() <= 0 {
            return None;
        }
        cur = polyk_deflate(&cur, c);
        m += 1;
    }
}

/// `(ord_c(r), lim_{x→c} r(x)/(x−c)^ord)` for `r ∈ Q(n)(x)`.
fn ratk_order_and_value(r: &RatK, c: &Rational) -> Option<(i64, Rn)> {
    let (mn, vn) = polyk_order_and_value(&r.num, c)?;
    let (md, vd) = polyk_order_and_value(&r.den, c)?;
    let v = crate::holonomic::qfield::rn_div(&vn, &vd)?;
    Some((mn - md, v))
}

/// `r(c)`, or `None` when `c` is a pole.
fn ratk_eval(r: &RatK, c: &Rational) -> Option<Rn> {
    let den = polyk_eval(&r.den, c);
    if rn_is_zero(&den) {
        return None;
    }
    crate::holonomic::qfield::rn_div(&polyk_eval(&r.num, c), &den)
}

/// `deg num − deg den`, the growth exponent at infinity.
fn ratk_degree(r: &RatK) -> i64 {
    (r.num.degree() as i64) - (r.den.degree() as i64)
}

/// The leading coefficient ratio at infinity, when it is a concrete rational.
fn ratk_leading_rational(r: &RatK) -> Option<Rational> {
    let lead = crate::holonomic::qfield::rn_div(&r.num.leading_coeff(), &r.den.leading_coeff())?;
    rn_as_rational(&lead)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomic::azeil::{almkvist_zeilberger, AzOpts};
    use crate::kernel::Domain;

    fn nx(pool: &ExprPool) -> (ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("x", Domain::Real),
        )
    }

    fn verdict(
        f: ExprId,
        n: ExprId,
        x: ExprId,
        pool: &ExprPool,
        lo: IntegrationLimit,
        hi: IntegrationLimit,
    ) -> (AzResult, IntegralBoundaryStatus) {
        let out = almkvist_zeilberger(f, n, x, pool, &AzOpts::default()).expect("certificate");
        let status = integral_boundary_status(&out.value, f, n, x, pool, &lo, &hi)
            .expect("boundary analysis");
        (out.value, status)
    }

    /// `∫_0^∞ xⁿ·e^(−x) dx = n!`: the boundary term vanishes at both ends, the
    /// lower one conditionally on `n + 1 > 0`.
    #[test]
    fn gamma_boundary_vanishes() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), x])]),
        ]);
        let (r, status) = verdict(
            f,
            n,
            x,
            &pool,
            IntegrationLimit::at(0),
            IntegrationLimit::PosInfinity,
        );
        assert_eq!(r.order, 1);
        assert_eq!(status.tag(), "vanishes");
        assert!(status.implies_integral_recurrence());
        assert_eq!(
            status.conditions().len(),
            1,
            "x^(n+1) -> 0 at the origin only for n + 1 > 0"
        );
        let conds = status.side_conditions("x = 0..+oo", &pool);
        assert!(
            conds.iter().any(|c| c.contains("> 0")),
            "the positivity condition must be reported: {conds:?}"
        );
    }

    /// `∫_{−∞}^{∞} x^(2n)·e^(−x²) dx`: `e^(−x²)` kills both ends
    /// unconditionally.
    #[test]
    fn gaussian_boundary_vanishes_unconditionally() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, pool.mul(vec![pool.integer(2_i32), n])),
            pool.func(
                "exp",
                vec![pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))])],
            ),
        ]);
        let (_, status) = verdict(
            f,
            n,
            x,
            &pool,
            IntegrationLimit::NegInfinity,
            IntegrationLimit::PosInfinity,
        );
        assert_eq!(status.tag(), "vanishes");
        assert!(
            status.conditions().is_empty(),
            "exp(-x^2) needs no condition on n at either end"
        );
    }

    /// `∫_0^1 xⁿ dx = 1/(n+1)`: the boundary term genuinely does **not**
    /// vanish, and the verdict must say so rather than quietly claim the
    /// homogeneous relation `f(n) = 0`.
    #[test]
    fn nonvanishing_boundary_is_reported_as_nonzero() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.pow(x, n);
        let (r, status) = verdict(
            f,
            n,
            x,
            &pool,
            IntegrationLimit::at(0),
            IntegrationLimit::at(1),
        );
        assert_eq!(r.order, 0, "F = D_x(x/(n+1) * F)");
        match &status {
            IntegralBoundaryStatus::Nonzero { rhs, conditions } => {
                // rhs must be 1/(n+1): check at a couple of integer n.
                for ni in [2.0_f64, 7.0] {
                    let env = std::collections::HashMap::from([(n, ni)]);
                    let got = crate::eval_f64(*rhs, &pool, &env).expect("rhs evaluates");
                    assert!(
                        (got - 1.0 / (ni + 1.0)).abs() < 1e-12,
                        "boundary term at n={ni}: got {got}, want {}",
                        1.0 / (ni + 1.0)
                    );
                }
                assert_eq!(conditions.len(), 1, "x^(n+1) -> 0 at 0 needs n + 1 > 0");
            }
            other => panic!("expected Nonzero, got {other:?}"),
        }
        assert!(status.implies_integral_recurrence());
    }

    /// A divergent integral must be `Unknown`, not `Vanishes`.
    #[test]
    fn divergent_upper_limit_is_unknown() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        // ∫_0^∞ xⁿ·eˣ dx diverges; the boundary term at +∞ is unbounded.
        let f = pool.mul(vec![pool.pow(x, n), pool.func("exp", vec![x])]);
        let (_, status) = verdict(
            f,
            n,
            x,
            &pool,
            IntegrationLimit::at(0),
            IntegrationLimit::PosInfinity,
        );
        assert_eq!(status.tag(), "unknown");
        assert!(
            !status.implies_integral_recurrence(),
            "nothing may be claimed about a divergent integral"
        );
    }

    /// `∫_0^1 xⁿ(1−x)ⁿ dx`: vanishes at both ends, conditionally.
    #[test]
    fn central_beta_boundary_vanishes() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let one_minus = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let f = pool.mul(vec![pool.pow(x, n), pool.pow(one_minus, n)]);
        let (r, status) = verdict(
            f,
            n,
            x,
            &pool,
            IntegrationLimit::at(0),
            IntegrationLimit::at(1),
        );
        assert_eq!(r.order, 1);
        assert_eq!(status.tag(), "vanishes", "got {status:?}");
        assert!(
            !status.conditions().is_empty(),
            "the endpoints vanish only for n large enough"
        );
    }

    /// An `Unknown` verdict must never be readable as a vanishing boundary, and
    /// `side_conditions` must say so in words.
    #[test]
    fn unknown_reads_as_unknown() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![pool.pow(x, n), pool.func("exp", vec![x])]);
        let (_, status) = verdict(
            f,
            n,
            x,
            &pool,
            IntegrationLimit::at(0),
            IntegrationLimit::PosInfinity,
        );
        let sc = status.side_conditions("x = 0..+oo", &pool);
        assert!(
            sc.iter().any(|s| s.contains("may be claimed")),
            "an Unknown verdict must state that nothing follows: {sc:?}"
        );
    }

    #[test]
    fn certificate_from_a_different_call_is_refused() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.pow(x, n);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let mut bogus = out.value.clone();
        bogus.certificate = pool.func("exp", vec![x]);
        let err = integral_boundary_status(
            &bogus,
            f,
            n,
            x,
            &pool,
            &IntegrationLimit::at(0),
            &IntegrationLimit::at(1),
        )
        .expect_err("a non-rational certificate is not from this engine");
        assert!(matches!(err, DiffTelescopingError::InvalidInput(_)));
    }
}
