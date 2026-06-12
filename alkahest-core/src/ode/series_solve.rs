//! Series solutions of second-order linear ODEs (power series + Frobenius).
//!
//! Solves
//!
//! ```text
//! p(x)·y'' + q(x)·y' + r(x)·y = 0
//! ```
//!
//! around a chosen expansion point `x₀`, with `p`, `q`, `r` expressions in `x`
//! that are analytic at `x₀` (polynomials, or rationals/elementary functions
//! whose Taylor expansion at `x₀` has rational coefficients). All coefficient
//! recurrences run in **exact ℚ arithmetic**, so the returned coefficients are
//! exact.
//!
//! # Two regimes
//!
//! * **Ordinary point** (`p(x₀) ≠ 0`): the equation is normalised to
//!   `y'' + P·y' + Q·y = 0` with `P = q/p`, `Q = r/p` analytic, and the two
//!   independent power-series solutions are produced from the initial data
//!   `(a₀,a₁) = (1,0)` and `(0,1)`:
//!
//!   ```text
//!   (n+2)(n+1)·a_{n+2} = −∑_{j=0}^{n} [ P_j·(n+1−j)·a_{n+1−j} + Q_j·a_{n−j} ].
//!   ```
//!
//! * **Regular singular point** (`p(x₀) = 0` but `(x−x₀)q/p` and
//!   `(x−x₀)²r/p` analytic): the **Frobenius** method. Writing
//!   `P(t)=t·q/p`, `Q(t)=t²·r/p` (`t = x−x₀`), the indicial equation is
//!
//!   ```text
//!   I(ρ) = ρ(ρ−1) + P₀·ρ + Q₀ = 0,
//!   ```
//!
//!   with roots `r₁ ≥ r₂`. The Frobenius recurrence for `y = t^ρ ∑ bₙ tⁿ` is
//!
//!   ```text
//!   I(ρ+n)·bₙ = −∑_{k=1}^{n} [ P_k·(ρ+n−k) + Q_k ]·b_{n−k},  b₀ = 1.
//!   ```
//!
//!   - **non-integer `r₁−r₂`**: two independent Frobenius series
//!     `t^{r₁}∑…`, `t^{r₂}∑…`.
//!   - **equal roots `r₁=r₂`**: first solution `y₁ = t^{r₁}∑ bₙ tⁿ` and a
//!     **logarithmic** second solution
//!     `y₂ = y₁·ln t + t^{r₁} ∑ b'ₙ tⁿ`, where `b'ₙ = d bₙ/dρ |_{ρ=r₁}` is
//!     obtained by differentiating the recurrence in `ρ`.
//!   - **positive-integer `r₁−r₂`**: the larger-root solution `y₁` always; the
//!     second solution carries a log term `c·y₁·ln t + t^{r₂}∑…` and is returned
//!     when the construction is feasible. When the indicial obstruction makes
//!     the exact closed second series intractable here, only `y₁` is returned
//!     (documented decline) — SymPy struggles with the same case.
//!
//! Only **rational** indicial roots are handled; an irrational (quadratic-surd)
//! indicial root declines (the recurrence would require surd arithmetic).
//!
//! # Exact residual gate
//!
//! Every returned solution is substituted back into the ODE and the residual is
//! checked to vanish **exactly** (in ℚ) up to the guaranteed order: all power-
//! series coefficients of `p·y'' + q·y' + r·y` through the requested order are
//! verified to be `0`. A solution whose residual does not vanish exactly is
//! withheld and the call declines.

use crate::calculus::fps::{Fps, FpsError};
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::{Integer, Rational};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Description of a second-order linear ODE `p·y'' + q·y' + r·y = 0`.
#[derive(Clone, Debug)]
pub struct SeriesOde {
    /// Independent variable `x`.
    pub x: ExprId,
    /// Coefficient of `y''`.
    pub p: ExprId,
    /// Coefficient of `y'`.
    pub q: ExprId,
    /// Coefficient of `y`.
    pub r: ExprId,
}

impl SeriesOde {
    /// Build an ODE `p·y'' + q·y' + r·y = 0` in the variable `x`.
    pub fn new(x: ExprId, p: ExprId, q: ExprId, r: ExprId) -> Self {
        SeriesOde { x, p, q, r }
    }
}

/// Classification of the expansion point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointKind {
    /// `p(x₀) ≠ 0` — ordinary point (plain power series).
    Ordinary,
    /// `p(x₀) = 0`, `t·q/p` and `t²·r/p` analytic — regular singular point.
    RegularSingular,
}

/// One independent series solution.
#[derive(Clone, Debug)]
pub struct SeriesSolution {
    /// Frobenius exponent `ρ` (`0` at an ordinary point). Rational.
    pub exponent: Rational,
    /// Coefficients `[a₀, a₁, …, a_{N-1}]` of the bracketed series
    /// `∑ aₙ (x−x₀)ⁿ` (the part multiplying `(x−x₀)^exponent` and, if
    /// [`Self::log_coeff`] is set, the non-log part).
    pub coeffs: Vec<Rational>,
    /// If `Some(c)`, the solution includes a logarithmic term
    /// `c · y₁ · ln(x−x₀)` added to `(x−x₀)^exponent · ∑ coeffs`.
    /// `y₁` is the first (larger-root) Frobenius solution; [`Self::log_base`]
    /// holds its bracketed coefficients.
    pub log_coeff: Option<Rational>,
    /// The bracketed coefficients of the `y₁` factor of the log term (only set
    /// when [`Self::log_coeff`] is `Some`).
    pub log_base: Option<(Rational, Vec<Rational>)>,
}

/// Result of [`series_solve`]: the two independent solutions, the point kind,
/// and the truncation order `N`.
#[derive(Clone, Debug)]
pub struct SeriesResult {
    /// Classification of `x₀`.
    pub kind: PointKind,
    /// Expansion point `x₀` (as an expression).
    pub x0: ExprId,
    /// Number of bracketed coefficients computed (`a₀ … a_{N-1}`).
    pub order: usize,
    /// The independent solution branches (one or two).
    pub solutions: Vec<SeriesSolution>,
}

/// Errors / documented declines from [`series_solve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeriesError {
    /// A coefficient (`p`, `q`, `r`, or a normalised quotient) is not analytic
    /// at `x₀` with rational Taylor coefficients.
    NotAnalytic(String),
    /// The point is neither ordinary nor regular-singular (irregular singular
    /// point), so no Frobenius series exists.
    IrregularSingular,
    /// The indicial roots are irrational (quadratic surd); not handled.
    IrrationalIndicialRoot,
    /// `p ≡ 0` (not a second-order ODE).
    DegenerateLeadingCoefficient,
    /// A candidate series did not pass the exact-residual gate.
    VerificationFailed(String),
    /// The (log-case) second solution is intractable here; only the first
    /// solution was produced.
    SecondSolutionDeclined(String),
}

impl fmt::Display for SeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeriesError::NotAnalytic(m) => write!(f, "series_solve: not analytic at x₀: {m}"),
            SeriesError::IrregularSingular => {
                write!(
                    f,
                    "series_solve: irregular singular point (no Frobenius series)"
                )
            }
            SeriesError::IrrationalIndicialRoot => {
                write!(f, "series_solve: irrational indicial roots are not handled")
            }
            SeriesError::DegenerateLeadingCoefficient => {
                write!(
                    f,
                    "series_solve: leading coefficient p(x) is identically zero"
                )
            }
            SeriesError::VerificationFailed(m) => {
                write!(
                    f,
                    "series_solve: candidate failed exact-residual check: {m}"
                )
            }
            SeriesError::SecondSolutionDeclined(m) => {
                write!(f, "series_solve: second solution declined: {m}")
            }
        }
    }
}

impl std::error::Error for SeriesError {}

impl crate::errors::AlkahestError for SeriesError {
    fn code(&self) -> &'static str {
        match self {
            SeriesError::NotAnalytic(_) => "E-ODE-020",
            SeriesError::IrregularSingular => "E-ODE-021",
            SeriesError::IrrationalIndicialRoot => "E-ODE-022",
            SeriesError::DegenerateLeadingCoefficient => "E-ODE-023",
            SeriesError::VerificationFailed(_) => "E-ODE-024",
            SeriesError::SecondSolutionDeclined(_) => "E-ODE-025",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            SeriesError::NotAnalytic(_) => {
                Some("ensure p, q, r are analytic at x₀ with rational Taylor coefficients")
            }
            SeriesError::IrregularSingular => {
                Some("series_solve handles ordinary and regular-singular points only")
            }
            SeriesError::IrrationalIndicialRoot => {
                Some("the indicial equation has irrational roots, outside the ℚ-recurrence path")
            }
            SeriesError::DegenerateLeadingCoefficient => {
                Some("supply a non-zero coefficient for y''")
            }
            SeriesError::VerificationFailed(_) => {
                Some("the candidate series did not satisfy the ODE exactly; it is withheld")
            }
            SeriesError::SecondSolutionDeclined(_) => Some(
                "the logarithmic second solution is intractable for this equation; only the \
                 first solution is returned (SymPy also struggles with this case)",
            ),
        }
    }
}

/// Solve `p·y'' + q·y' + r·y = 0` as a series about `x₀` to `order` bracketed
/// terms.
///
/// Returns the point classification and the independent solution branches (see
/// [`SeriesResult`]). Every returned branch passes the exact-residual gate.
///
/// # Errors
///
/// Declines (see [`SeriesError`]) at irregular singular points, irrational
/// indicial roots, non-analytic coefficients, or when the exact-residual check
/// fails. The logarithmic second solution at a positive-integer root gap may be
/// declined while the first solution is still returned.
pub fn series_solve(
    ode: &SeriesOde,
    x0: ExprId,
    order: usize,
    pool: &ExprPool,
) -> Result<SeriesResult, SeriesError> {
    let order = order.max(2);
    // Shift to t = x − x₀: substitute x → x₀ + t with a fresh variable t.
    let t = pool.symbol("·t_series·", crate::kernel::Domain::Real);
    let shifted_x = pool.add(vec![x0, t]);
    let p = subs_x(ode.p, ode.x, shifted_x, pool);
    let q = subs_x(ode.q, ode.x, shifted_x, pool);
    let r = subs_x(ode.r, ode.x, shifted_x, pool);

    // Taylor series of p, q, r about t = 0.
    let need = order + 8; // headroom for normalisation / verification
    let pa = expr_coeffs(p, t, need, pool)?;
    let qa = expr_coeffs(q, t, need, pool)?;
    let ra = expr_coeffs(r, t, need, pool)?;

    if pa.iter().all(|c| *c == 0) {
        return Err(SeriesError::DegenerateLeadingCoefficient);
    }

    let p0 = pa[0].clone();
    if p0 != 0 {
        ordinary_point(&pa, &qa, &ra, x0, order, pool)
    } else {
        frobenius(&pa, &qa, &ra, x0, order, pool)
    }
}

// ---------------------------------------------------------------------------
// Ordinary point
// ---------------------------------------------------------------------------

fn ordinary_point(
    pa: &[Rational],
    qa: &[Rational],
    ra: &[Rational],
    x0: ExprId,
    order: usize,
    pool: &ExprPool,
) -> Result<SeriesResult, SeriesError> {
    // Normalise: P = q/p, Q = r/p as power series (p(0) ≠ 0).
    let pfps = Fps::from_poly(pa);
    let qfps = Fps::from_poly(qa);
    let rfps = Fps::from_poly(ra);
    let big_p = qfps.div(&pfps).map_err(map_fps)?;
    let big_q = rfps.div(&pfps).map_err(map_fps)?;
    let pc = big_p.coeffs(order + 2);
    let qc = big_q.coeffs(order + 2);

    let sol1 = power_series_solution(&pc, &qc, [Rational::from(1), Rational::from(0)], order);
    let sol2 = power_series_solution(&pc, &qc, [Rational::from(0), Rational::from(1)], order);

    let mut solutions = Vec::new();
    for coeffs in [sol1, sol2] {
        let s = SeriesSolution {
            exponent: Rational::from(0),
            coeffs,
            log_coeff: None,
            log_base: None,
        };
        verify_solution(pa, qa, ra, &s, order, pool)?;
        solutions.push(s);
    }

    Ok(SeriesResult {
        kind: PointKind::Ordinary,
        x0,
        order,
        solutions,
    })
}

/// Power-series recurrence
/// `(n+2)(n+1)a_{n+2} = −∑_{j=0}^{n}[P_j (n+1−j) a_{n+1−j} + Q_j a_{n−j}]`.
fn power_series_solution(
    pc: &[Rational],
    qc: &[Rational],
    init: [Rational; 2],
    order: usize,
) -> Vec<Rational> {
    let mut a: Vec<Rational> = Vec::with_capacity(order);
    a.push(init[0].clone());
    if order >= 2 {
        a.push(init[1].clone());
    }
    for n in 0..order.saturating_sub(2) {
        let mut acc = Rational::from(0);
        for j in 0..=n {
            let pj = pc.get(j).cloned().unwrap_or_else(|| Rational::from(0));
            if pj != 0 {
                acc += pj * Rational::from(n + 1 - j) * a[n + 1 - j].clone();
            }
            let qj = qc.get(j).cloned().unwrap_or_else(|| Rational::from(0));
            if qj != 0 {
                acc += qj * a[n - j].clone();
            }
        }
        let denom = Rational::from((n + 2) * (n + 1));
        a.push(-acc / denom);
    }
    a.truncate(order);
    a
}

// ---------------------------------------------------------------------------
// Frobenius (regular singular point)
// ---------------------------------------------------------------------------

fn frobenius(
    pa: &[Rational],
    qa: &[Rational],
    ra: &[Rational],
    x0: ExprId,
    order: usize,
    pool: &ExprPool,
) -> Result<SeriesResult, SeriesError> {
    // P(t) = t·q/p, Q(t) = t²·r/p must be analytic at t = 0. At a regular
    // singular point p(0) = 0, so `Fps::div` (which needs a nonzero denominator
    // constant term) cannot be used directly. We instead form the numerators
    // t·q and t²·r as coefficient vectors and divide by p after cancelling the
    // common `t^val(p)` factor; an irregular point shows up as a numerator whose
    // valuation is strictly smaller than `val(p)`.
    let nq = shift_coeffs(qa, 1); // t·q
    let nr = shift_coeffs(ra, 2); // t²·r
    let pc = series_quotient_shifted(&nq, pa, order + 2)?; // t·q / p
    let qc = series_quotient_shifted(&nr, pa, order + 2)?; // t²·r / p
    let p0 = pc[0].clone();
    let q0 = qc[0].clone();

    // Indicial equation: ρ² + (P₀ − 1)ρ + Q₀ = 0.
    let b = p0.clone() - Rational::from(1);
    let c = q0.clone();
    let (r1, r2) = match rational_quadratic_roots(&b, &c) {
        Some(roots) => roots,
        None => return Err(SeriesError::IrrationalIndicialRoot),
    };

    let mut solutions = Vec::new();

    // First (larger-root) solution always exists.
    let coeffs1 = frobenius_coeffs(&pc, &qc, &r1, order);
    let sol1 = SeriesSolution {
        exponent: r1.clone(),
        coeffs: coeffs1.clone(),
        log_coeff: None,
        log_base: None,
    };
    verify_solution(pa, qa, ra, &sol1, order, pool)?;
    solutions.push(sol1);

    let diff = r1.clone() - r2.clone();
    let diff_is_int = diff.denom() == &Integer::from(1);

    if !diff_is_int {
        // Case (a): two independent Frobenius series.
        let coeffs2 = frobenius_coeffs(&pc, &qc, &r2, order);
        let sol2 = SeriesSolution {
            exponent: r2.clone(),
            coeffs: coeffs2,
            log_coeff: None,
            log_base: None,
        };
        verify_solution(pa, qa, ra, &sol2, order, pool)?;
        solutions.push(sol2);
    } else if r1 == r2 {
        // Case (b1): equal roots — log second solution via d/dρ.
        match frobenius_log_equal(&pc, &qc, &r1, &coeffs1, order) {
            Ok(sol2) => {
                verify_solution(pa, qa, ra, &sol2, order, pool)?;
                solutions.push(sol2);
            }
            Err(e) => return finish_with_decline(solutions, x0, order, e),
        }
    } else {
        // Case (b2): roots differ by a positive integer m = r1 − r2.
        match frobenius_log_integer(pa, qa, ra, &r1, &r2, &coeffs1, order) {
            Ok(sol2) => {
                verify_solution(pa, qa, ra, &sol2, order, pool)?;
                solutions.push(sol2);
            }
            Err(e) => return finish_with_decline(solutions, x0, order, e),
        }
    }

    Ok(SeriesResult {
        kind: PointKind::RegularSingular,
        x0,
        order,
        solutions,
    })
}

/// Indicial polynomial `I(ρ+n) = (ρ+n)(ρ+n−1) + P₀(ρ+n) + Q₀`.
fn indicial(rho: &Rational, n: usize, p0: &Rational, q0: &Rational) -> Rational {
    let s = rho.clone() + Rational::from(n);
    s.clone() * (s.clone() - Rational::from(1)) + p0.clone() * s + q0.clone()
}

/// Frobenius coefficients `bₙ` for exponent `rho` with `b₀ = 1`:
/// `I(ρ+n)·bₙ = −∑_{k=1}^{n}[P_k(ρ+n−k) + Q_k] b_{n−k}`.
fn frobenius_coeffs(
    pc: &[Rational],
    qc: &[Rational],
    rho: &Rational,
    order: usize,
) -> Vec<Rational> {
    let p0 = pc[0].clone();
    let q0 = qc[0].clone();
    let mut b: Vec<Rational> = Vec::with_capacity(order);
    b.push(Rational::from(1));
    for n in 1..order {
        let mut acc = Rational::from(0);
        for k in 1..=n {
            let pk = pc.get(k).cloned().unwrap_or_else(|| Rational::from(0));
            let qk = qc.get(k).cloned().unwrap_or_else(|| Rational::from(0));
            if pk == 0 && qk == 0 {
                continue;
            }
            let factor = pk * (rho.clone() + Rational::from(n - k)) + qk;
            acc += factor * b[n - k].clone();
        }
        let ind = indicial(rho, n, &p0, &q0);
        if ind == 0 {
            // Should not happen for the larger root / non-integer gap; guard.
            b.push(Rational::from(0));
        } else {
            b.push(-acc / ind);
        }
    }
    b
}

/// Equal-root logarithmic second solution.
///
/// With `b(ρ)` the recurrence solution (`b₀(ρ)=1`), the second solution is
/// `y₂ = y₁·ln t + t^{r} ∑ b'ₙ tⁿ` where `b'ₙ = d bₙ/dρ` evaluated at `ρ=r`.
/// We carry `(bₙ, b'ₙ)` through the recurrence by symbolic differentiation in
/// `ρ` (dual-number style with exact ℚ arithmetic).
fn frobenius_log_equal(
    pc: &[Rational],
    qc: &[Rational],
    rho: &Rational,
    coeffs1: &[Rational],
    order: usize,
) -> Result<SeriesSolution, SeriesError> {
    let dcoeffs = frobenius_dcoeffs(pc, qc, rho, order)?;
    Ok(SeriesSolution {
        exponent: rho.clone(),
        coeffs: dcoeffs,
        log_coeff: Some(Rational::from(1)),
        log_base: Some((rho.clone(), coeffs1.to_vec())),
    })
}

/// Derivative coefficients `b'ₙ = d bₙ/dρ |_ρ` (with `b'₀ = 0`).
///
/// Differentiating `I(ρ+n)·bₙ = −∑_{k≥1}[P_k(ρ+n−k)+Q_k] b_{n−k}` in `ρ`:
/// `I'(ρ+n)·bₙ + I(ρ+n)·b'ₙ = −∑_{k≥1}[ P_k·b_{n−k} + (P_k(ρ+n−k)+Q_k)·b'_{n−k} ]`,
/// so `b'ₙ = ( −I'(ρ+n)·bₙ − ∑_{k≥1}[P_k b_{n−k} + (…)b'_{n−k}] ) / I(ρ+n)`.
fn frobenius_dcoeffs(
    pc: &[Rational],
    qc: &[Rational],
    rho: &Rational,
    order: usize,
) -> Result<Vec<Rational>, SeriesError> {
    let p0 = pc[0].clone();
    let q0 = qc[0].clone();
    let b = frobenius_coeffs(pc, qc, rho, order);
    let mut db: Vec<Rational> = Vec::with_capacity(order);
    db.push(Rational::from(0)); // b'₀ = 0
    for n in 1..order {
        let ind = indicial(rho, n, &p0, &q0);
        if ind == 0 {
            return Err(SeriesError::SecondSolutionDeclined(format!(
                "indicial obstruction at n={n} (equal-root log path)"
            )));
        }
        // I'(ρ+n) = 2(ρ+n) − 1 + P₀.
        let s = rho.clone() + Rational::from(n);
        let ind_prime = Rational::from(2) * s.clone() - Rational::from(1) + p0.clone();
        let mut acc = ind_prime * b[n].clone();
        for k in 1..=n {
            let pk = pc.get(k).cloned().unwrap_or_else(|| Rational::from(0));
            let qk = qc.get(k).cloned().unwrap_or_else(|| Rational::from(0));
            if pk != 0 {
                acc += pk.clone() * b[n - k].clone();
            }
            let factor = pk * (rho.clone() + Rational::from(n - k)) + qk;
            if factor != 0 {
                acc += factor * db[n - k].clone();
            }
        }
        db.push(-acc / ind);
    }
    Ok(db)
}

/// Positive-integer-gap (`m = r₁ − r₂`) logarithmic second solution.
///
/// Standard form `y₂ = C·y₁·ln t + t^{r₂} ∑ cₙ tⁿ` with `c₀ = 1`. The
/// coefficients are determined by the recurrence using the *smaller* root `r₂`;
/// the obstruction occurs at `n = m`, where `I(r₂+m) = I(r₁) = 0`. The log
/// coefficient `C` is fixed so that the `n = m` equation is consistent. When the
/// resulting `C = 0`, no log is needed (the smaller root already yields an
/// independent series); otherwise the log term is included.
fn frobenius_log_integer(
    pa: &[Rational],
    qa: &[Rational],
    ra: &[Rational],
    r1: &Rational,
    r2: &Rational,
    coeffs1: &[Rational],
    order: usize,
) -> Result<SeriesSolution, SeriesError> {
    // We construct the smaller-root bracket cₙ and the log coefficient C directly
    // from the *exact* raw-coefficient residual `t²·L[y₂]/t^{r₂}` (the same series
    // the verification gate checks), so construction and verification can never
    // disagree.
    //
    // Let v = val(p) (≥ 1 at a regular singular point). The residual coefficient
    // at index `idx` first involves the bracket unknown `c_{idx−v}` with a
    // "diagonal" multiplier `diag(idx)`. Away from the obstruction `diag ≠ 0`, so
    // the unknown is solved directly. At the single obstruction index
    // (`diag = 0`, where `r₂ + (idx−v)` hits the larger root) we instead use the
    // log coefficient `C` to satisfy the equation; the corresponding bracket
    // coefficient is free and set to 0.
    let p = Fps::from_poly(pa);
    let q = Fps::from_poly(qa);
    let r = Fps::from_poly(ra);
    let v = valuation(pa).ok_or(SeriesError::DegenerateLeadingCoefficient)?;

    // Residual coefficient at `idx` for a given bracket and log coefficient.
    let resid = |bracket: &[Rational], big_c: &Rational, idx: usize| -> Rational {
        let log = if *big_c == 0 {
            None
        } else {
            Some((big_c, r1, coeffs1))
        };
        residual_series(&p, &q, &r, r2, bracket, log).coeff(idx)
    };

    let zero = Rational::from(0);
    let one = Rational::from(1);
    let mut c: Vec<Rational> = vec![Rational::from(0); order];
    c[0] = one.clone();
    let mut big_c = zero.clone();

    // Sweep residual indices. The bracket unknown introduced at residual index
    // `idx` is `j = idx − v`; we want residual coefficients to vanish through the
    // guaranteed window so that all `order` bracket terms are pinned.
    for idx in v..(order + v) {
        let j = idx - v;
        if j >= order {
            break;
        }
        // Base residual with c[j] = 0 and the current C.
        c[j] = zero.clone();
        let base = resid(&c, &big_c, idx);
        // Diagonal: marginal effect of c[j] on residual index `idx`.
        c[j] = one.clone();
        let with_one = resid(&c, &big_c, idx);
        c[j] = zero.clone();
        let diag = with_one - base.clone();

        if diag != 0 {
            c[j] = -base / diag;
            continue;
        }
        // Obstruction (diag = 0): the equation `base + C·(∂R/∂C)|idx = 0` fixes C.
        // c[j] stays free (0). Determine ∂R/∂C at this index.
        let base_c0 = resid(&c, &zero, idx);
        let with_c1 = resid(&c, &one, idx);
        let dc = with_c1 - base_c0.clone();
        if dc == 0 {
            // No way to cancel via the log term either → genuinely intractable.
            return Err(SeriesError::SecondSolutionDeclined(format!(
                "indicial obstruction at bracket index {j} cannot be resolved by a log term"
            )));
        }
        big_c = -base_c0 / dc;
        // c[j] remains 0 (free parameter; any value shifts y₂ by a multiple of y₁).
    }

    let (log_coeff, log_base) = if big_c == 0 {
        (None, None)
    } else {
        (Some(big_c), Some((r1.clone(), coeffs1.to_vec())))
    };
    Ok(SeriesSolution {
        exponent: r2.clone(),
        coeffs: c,
        log_coeff,
        log_base,
    })
}

fn finish_with_decline(
    solutions: Vec<SeriesSolution>,
    x0: ExprId,
    order: usize,
    _e: SeriesError,
) -> Result<SeriesResult, SeriesError> {
    // Return the first solution but record that the second was declined by
    // leaving only one branch. Callers can detect the single-branch result.
    Ok(SeriesResult {
        kind: PointKind::RegularSingular,
        x0,
        order,
        solutions,
    })
}

// ---------------------------------------------------------------------------
// Symbolic rendering
// ---------------------------------------------------------------------------

impl SeriesSolution {
    /// Render this solution as a truncated symbolic expression in `x`, expanded
    /// about `x₀`, with an `O((x−x₀)^order)` tail on the bracketed part.
    ///
    /// The exponent `(x−x₀)^ρ`, any logarithm `c·y₁·ln(x−x₀)`, and the
    /// power-series body are assembled into a single expression.
    pub fn to_expr(&self, x: ExprId, x0: ExprId, order: usize, pool: &ExprPool) -> ExprId {
        let t = shift_expr(x, x0, pool);
        let body = series_body_expr(&self.coeffs, t, order, /*with_o=*/ true, pool);
        let prefactor = pow_expr(t, &self.exponent, pool);
        let main = pool.mul(vec![prefactor, body]);
        if let (Some(c), Some((rho1, base))) = (&self.log_coeff, &self.log_base) {
            let y1_body = series_body_expr(base, t, order, /*with_o=*/ false, pool);
            let y1 = pool.mul(vec![pow_expr(t, rho1, pool), y1_body]);
            let ln = pool.func("ln", vec![t]);
            let log_term = pool.mul(vec![rat_to_expr(c, pool), y1, ln]);
            pool.add(vec![log_term, main])
        } else {
            main
        }
    }
}

impl SeriesResult {
    /// Whether the logarithmic second solution was declined (single branch at a
    /// regular singular point with integer/equal root gap).
    pub fn second_solution_declined(&self) -> bool {
        self.kind == PointKind::RegularSingular && self.solutions.len() == 1
    }
}

fn series_body_expr(
    coeffs: &[Rational],
    t: ExprId,
    order: usize,
    with_o: bool,
    pool: &ExprPool,
) -> ExprId {
    let mut terms = Vec::new();
    for (k, c) in coeffs.iter().enumerate().take(order) {
        if *c == 0 {
            continue;
        }
        let ce = rat_to_expr(c, pool);
        let term = if k == 0 {
            ce
        } else if k == 1 {
            pool.mul(vec![ce, t])
        } else {
            pool.mul(vec![ce, pool.pow(t, pool.integer(k as i64))])
        };
        terms.push(term);
    }
    if terms.is_empty() {
        terms.push(pool.integer(0));
    }
    if with_o {
        terms.push(pool.big_o(pool.pow(t, pool.integer(order as i64))));
    }
    pool.add(terms)
}

/// `(x − x₀)^ρ` as an expression (`x` itself when `x₀ = 0`).
fn pow_expr(t: ExprId, rho: &Rational, pool: &ExprPool) -> ExprId {
    if *rho == 0 {
        return pool.integer(1);
    }
    if *rho == 1 {
        return t;
    }
    pool.pow(t, rat_to_expr(rho, pool))
}

/// `x − x₀` simplified to `x` when `x₀` is literally `0`.
fn shift_expr(x: ExprId, x0: ExprId, pool: &ExprPool) -> ExprId {
    if matches!(pool.get(x0), ExprData::Integer(ref n) if n.0 == 0) {
        return x;
    }
    let neg = pool.mul(vec![pool.integer(-1), x0]);
    pool.add(vec![x, neg])
}

// ---------------------------------------------------------------------------
// Exact residual verification
// ---------------------------------------------------------------------------

/// Substitute the truncated series back into `p·y'' + q·y' + r·y` and require
/// every power-series coefficient through the guaranteed order to vanish in ℚ.
///
/// For a Frobenius solution `y = tᵖ ∑ aₙ tⁿ`, the residual has the form
/// `tᵖ ∑ Rₘ tᵐ` (plus, for the log case, a separately-handled `ln t` part); we
/// check `Rₘ = 0` for `m` up to the guaranteed order.
fn verify_solution(
    pa: &[Rational],
    qa: &[Rational],
    ra: &[Rational],
    sol: &SeriesSolution,
    order: usize,
    _pool: &ExprPool,
) -> Result<(), SeriesError> {
    // The verification is purely on the bracketed (non-log, non-tᵖ) series. For
    // log solutions the log part cancels by construction of y₁; we additionally
    // verify the residual of the full y₂ via the coefficient recurrence identity
    // below (the non-log coefficients), which is the strong exact gate.
    let rho = &sol.exponent;
    let p = Fps::from_poly(pa);
    let q = Fps::from_poly(qa);
    let r = Fps::from_poly(ra);

    // The recurrence chooses `order` bracketed coefficients, so the residual is
    // guaranteed to vanish through residual index `order − 1` (the higher-index
    // residual coefficients depend on bracket terms we have not yet computed and
    // are *not* guaranteed). Verify exactly that guaranteed window.
    let check_to = order.saturating_sub(1);

    let log = sol
        .log_base
        .as_ref()
        .zip(sol.log_coeff.as_ref())
        .map(|((rho1, base), c)| (c, rho1, base.as_slice()));
    let residual = residual_series(&p, &q, &r, rho, &sol.coeffs, log);

    for m in 0..=check_to {
        let cm = residual.coeff(m);
        if cm != 0 {
            return Err(SeriesError::VerificationFailed(format!(
                "residual coefficient at order {m} is {cm} (exponent {rho})"
            )));
        }
    }
    Ok(())
}

/// The exact residual series `t²·L[y]/t^{ρ}` for
/// `y = C·y₁·ln t + t^{ρ} ∑ bracket[n] tⁿ`, where `L = p D² + q D + r` with the
/// raw coefficient slices `pa, qa, ra`, and `y₁ = t^{ρ₁} ∑ base[n] tⁿ`.
///
/// Multiplying the ODE by `t²` keeps every piece a genuine power series even at
/// a regular singular point (`p` vanishing at `0`). The coefficient of `tᵐ` in
/// the result is the exact ℚ residual of the candidate at `t^{ρ+m}`; a solution
/// is valid iff these vanish up to the guaranteed order.
///
/// Derivation (see [`verify_solution`]):
///   t²·L[t^{ρ}∑ b tⁿ]/t^{ρ} = p ⊛ D2 + (t q) ⊛ D1 + (t² r) ⊛ B,
/// with `B[n]=bₙ`, `D1[n]=(ρ+n)bₙ`, `D2[n]=(ρ+n)(ρ+n−1)bₙ`; the log term adds
///   C · t^{ρ₁−ρ} · [ p ⊛ S2 + (t q) ⊛ S1 ],
/// with `S1[n]=base[n]`, `S2[n]=(2(ρ₁+n)−1)·base[n]`.
fn residual_series<'p>(
    p: &Fps<'p>,
    q: &Fps<'p>,
    r: &Fps<'p>,
    rho: &Rational,
    bracket: &[Rational],
    log: Option<(&Rational, &Rational, &[Rational])>, // (C, ρ₁, base)
) -> Fps<'p> {
    let bfps = Fps::from_poly(bracket);
    let d1 = {
        let bracket = bracket.to_vec();
        let rho = rho.clone();
        Fps::from_fn(move |n| {
            (rho.clone() + Rational::from(n))
                * bracket.get(n).cloned().unwrap_or_else(|| Rational::from(0))
        })
    };
    let d2 = {
        let bracket = bracket.to_vec();
        let rho = rho.clone();
        Fps::from_fn(move |n| {
            let s = rho.clone() + Rational::from(n);
            s.clone()
                * (s - Rational::from(1))
                * bracket.get(n).cloned().unwrap_or_else(|| Rational::from(0))
        })
    };
    let tq = mul_t(q);
    let t2r = mul_t(&mul_t(r));
    let mut residual = p.mul(&d2).add(&tq.mul(&d1)).add(&t2r.mul(&bfps));

    if let Some((c, rho1, base)) = log {
        let m_shift = (rho1.clone() - rho.clone())
            .numer()
            .to_i64()
            .map(|v| v.max(0) as usize)
            .unwrap_or(0);
        let s1 = Fps::from_poly(base);
        let s2 = {
            let base = base.to_vec();
            let rho1 = rho1.clone();
            Fps::from_fn(move |n| {
                (Rational::from(2) * (rho1.clone() + Rational::from(n)) - Rational::from(1))
                    * base.get(n).cloned().unwrap_or_else(|| Rational::from(0))
            })
        };
        let coupling = p
            .mul(&s2)
            .add(&tq.mul(&s1))
            .scale(c.clone())
            .shift_up(m_shift);
        residual = residual.add(&coupling);
    }
    residual
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Multiply an Fps by `t` (shift coefficients up by one).
fn mul_t<'p>(f: &Fps<'p>) -> Fps<'p> {
    f.shift_up(1)
}

/// Shift a coefficient vector up by `k` (multiply the underlying series by `tᵏ`).
fn shift_coeffs(c: &[Rational], k: usize) -> Vec<Rational> {
    let mut out = vec![Rational::from(0); k];
    out.extend(c.iter().cloned());
    out
}

/// Valuation (index of the first nonzero coefficient) of a coefficient slice,
/// or `None` if all the given coefficients are zero.
fn valuation(c: &[Rational]) -> Option<usize> {
    c.iter().position(|x| *x != 0)
}

/// Divide the series `num / den` (given as ascending coefficient slices),
/// returning `n` coefficients. Both series may vanish at `t = 0`; the quotient
/// is analytic iff `val(num) ≥ val(den)`. Declines as
/// [`SeriesError::IrregularSingular`] when the quotient would have a pole.
fn series_quotient_shifted(
    num: &[Rational],
    den: &[Rational],
    n: usize,
) -> Result<Vec<Rational>, SeriesError> {
    let vden = match valuation(den) {
        Some(v) => v,
        None => return Err(SeriesError::DegenerateLeadingCoefficient),
    };
    let vnum = valuation(num);
    // num ≡ 0 → quotient ≡ 0 (analytic).
    let vnum = match vnum {
        Some(v) => v,
        None => return Ok(vec![Rational::from(0); n]),
    };
    if vnum < vden {
        // Quotient has a pole of order vden − vnum: not analytic.
        return Err(SeriesError::IrregularSingular);
    }
    // Strip the common t^vden: num/den = (num >> vden) / (den >> vden), and the
    // de-valuated denominator has a nonzero constant term, so Fps::div applies.
    let num_shifted: Vec<Rational> = num.iter().skip(vden).cloned().collect();
    let den_shifted: Vec<Rational> = den.iter().skip(vden).cloned().collect();
    let nf = Fps::from_poly(&num_shifted);
    let df = Fps::from_poly(&den_shifted);
    let quot = nf.div(&df).map_err(map_fps)?;
    Ok(quot.coeffs(n))
}

/// Rational quadratic `ρ² + bρ + c = 0`: return `(larger, smaller)` rational
/// roots, or `None` if the discriminant is not a perfect rational square.
fn rational_quadratic_roots(b: &Rational, c: &Rational) -> Option<(Rational, Rational)> {
    // disc = b² − 4c.
    let disc = b.clone() * b.clone() - Rational::from(4) * c.clone();
    if disc < 0 {
        return None; // complex roots — not handled
    }
    let sqrt = rational_sqrt(&disc)?;
    let half = Rational::from((1, 2));
    let r1 = half.clone() * (-b.clone() + sqrt.clone());
    let r2 = half * (-b.clone() - sqrt);
    if r1 >= r2 {
        Some((r1, r2))
    } else {
        Some((r2, r1))
    }
}

/// Exact rational square root, or `None` if not a perfect square.
fn rational_sqrt(x: &Rational) -> Option<Rational> {
    if *x < 0 {
        return None;
    }
    if *x == 0 {
        return Some(Rational::from(0));
    }
    let num = x.numer().clone();
    let den = x.denom().clone();
    let ns = num.sqrt_ref().into();
    let ds = den.sqrt_ref().into();
    let ns: Integer = ns;
    let ds: Integer = ds;
    if ns.clone() * ns.clone() == num && ds.clone() * ds.clone() == den {
        Some(Rational::from((ns, ds)))
    } else {
        None
    }
}

/// Substitute `x → replacement` in `expr`.
fn subs_x(expr: ExprId, x: ExprId, replacement: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(x, replacement);
    crate::kernel::subs::subs(expr, &m, pool)
}

/// Taylor coefficients of `expr` in `var` about `0`, requiring analyticity with
/// rational coefficients.
fn expr_coeffs(
    expr: ExprId,
    var: ExprId,
    n: usize,
    pool: &ExprPool,
) -> Result<Vec<Rational>, SeriesError> {
    let f = Fps::from_expr(expr, var, pool).map_err(map_fps)?;
    Ok(f.coeffs(n))
}

fn map_fps(e: FpsError) -> SeriesError {
    match e {
        FpsError::NotAnalyticAtZero | FpsError::DenominatorVanishesAtZero => {
            SeriesError::NotAnalytic(e.to_string())
        }
        FpsError::NonRationalCoefficient => SeriesError::NotAnalytic(e.to_string()),
        other => SeriesError::NotAnalytic(other.to_string()),
    }
}

fn rat_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    let num = r.numer().clone();
    let den = r.denom().clone();
    if den == 1 {
        pool.integer(num)
    } else {
        pool.rational(num, den)
    }
}

#[cfg(test)]
mod tests;
