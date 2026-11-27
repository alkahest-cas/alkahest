//! Green's functions of Sturm–Liouville problems: the eigenfunction series
//! (static, heat and wave) and the closed form by variation of parameters.
//!
//! Both are for the positive operator `𝓛u = −[(p u')' + q u]` of M3's
//! [`SturmLiouvilleProblem`] (`𝓛φₙ = λₙ w φₙ`) with its separated boundary
//! conditions, and solve `𝓛 G = δ(x − ξ)`.

use crate::kernel::{Domain, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::sturm_liouville::{
    simp, sturm_liouville, subs1, EndCondition, LowerSpectrum, SlBoundary, SturmLiouvilleProblem,
};
use super::super::sum::sum;
use super::super::verify::VerificationStatus;
use super::{
    at, check, combine, dd, div, fresh, is_symbol, neg, refuse_refuted, sqrt, sub, GreenCheck,
    GreenFunction,
};

/// Which Green's function of a Sturm–Liouville operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeriesKind {
    /// `𝓛 G = δ(x − ξ)`: `G = Σ φₙ(x) φₙ(ξ) / (λₙ ‖φₙ‖²)`.
    Static,
    /// `w Gₜ + κ 𝓛 G = 0`, `G(x, ξ, 0) = δ(x − ξ)/w(ξ)`:
    /// `G = Σ e^{−κλₙt} φₙ(x) φₙ(ξ) / ‖φₙ‖²`.
    Heat {
        /// `κ`.
        diffusivity: ExprId,
        /// `t`.
        time: ExprId,
    },
    /// `w Gₜₜ + c² 𝓛 G = 0`, `G(·, 0) = 0`, `Gₜ(x, ξ, 0) = δ(x − ξ)/w(ξ)`:
    /// `G = Σ sin(c√λₙ t) φₙ(x) φₙ(ξ) / (c√λₙ ‖φₙ‖²)`.
    Wave {
        /// `c`.
        speed: ExprId,
        /// `t`.
        time: ExprId,
    },
}

impl SeriesKind {
    /// `"static"`, `"heat"` or `"wave"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            SeriesKind::Static => "static",
            SeriesKind::Heat { .. } => "heat",
            SeriesKind::Wave { .. } => "wave",
        }
    }

    fn time(&self) -> Option<ExprId> {
        match *self {
            SeriesKind::Static => None,
            SeriesKind::Heat { time, .. } | SeriesKind::Wave { time, .. } => Some(time),
        }
    }
}

/// `𝓛 f = −[(p f')' + q f]`.
fn sl_op(f: ExprId, x: ExprId, p: ExprId, q: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    let pf = pool.mul(vec![p, dd(f, x, pool)?]);
    Ok(neg(
        pool.add(vec![dd(pf, x, pool)?, pool.mul(vec![q, f])]),
        pool,
    ))
}

/// The time factor of one mode with eigenvalue `lam` (`1/λ` static).
fn time_factor(kind: SeriesKind, lam: ExprId, root: Option<ExprId>, pool: &ExprPool) -> ExprId {
    match kind {
        SeriesKind::Static => pool.pow(lam, pool.integer(-1)),
        SeriesKind::Heat { diffusivity, time } => pool.func(
            "exp",
            vec![neg(pool.mul(vec![diffusivity, lam, time]), pool)],
        ),
        SeriesKind::Wave { speed, time } => {
            if simp(lam, pool) == pool.integer(0) {
                time
            } else {
                let om = pool.mul(vec![speed, root.unwrap_or_else(|| sqrt(lam, pool))]);
                div(pool.func("sin", vec![pool.mul(vec![om, time])]), om, pool)
            }
        }
    }
}

/// Checks of one mode `T = factor(λ) φ(x) φ(ξ)` (norm left out: constant).
#[allow(clippy::too_many_arguments)]
fn mode_checks(
    kind: SeriesKind,
    phi: ExprId,
    lam: ExprId,
    x: ExprId,
    xi: ExprId,
    problem: &SturmLiouvilleProblem,
    label: &str,
    pool: &ExprPool,
) -> Result<Vec<GreenCheck>, PdeError> {
    let (p, q, w) = (problem.p, problem.q, problem.w);
    let mut out = Vec::new();
    // 𝓛 φ = λ w φ (identically in the frequency).
    out.push(check(
        "term_ode",
        sub(
            sl_op(phi, x, p, q, pool)?,
            pool.mul(vec![lam, w, phi]),
            pool,
        ),
        "substitution",
        format!("{label}: 𝓛φ = λ w φ"),
        pool,
    ));
    let phi_xi = subs1(phi, x, xi, pool);
    let t_mode = pool.mul(vec![time_factor(kind, lam, None, pool), phi, phi_xi]);
    let swap = pool.symbol("__green_swap", Domain::Real);
    let swapped = subs1(
        subs1(subs1(t_mode, x, swap, pool), xi, x, pool),
        swap,
        xi,
        pool,
    );
    out.push(check(
        "symmetry",
        sub(t_mode, swapped, pool),
        "substitution",
        format!("{label}: G(x, ξ) = G(ξ, x) term by term"),
        pool,
    ));
    match kind {
        SeriesKind::Static => {}
        SeriesKind::Heat { diffusivity, time } => {
            let r = pool.add(vec![
                pool.mul(vec![w, dd(t_mode, time, pool)?]),
                pool.mul(vec![diffusivity, sl_op(t_mode, x, p, q, pool)?]),
            ]);
            out.push(check(
                "term_pde",
                r,
                "substitution",
                format!("{label}: w Tₜ + κ 𝓛T = 0"),
                pool,
            ));
            out.push(check(
                "term_ic",
                sub(
                    at(t_mode, time, pool.integer(0), pool),
                    pool.mul(vec![phi, phi_xi]),
                    pool,
                ),
                "substitution",
                format!("{label}: T(0) = φ(x)φ(ξ)"),
                pool,
            ));
        }
        SeriesKind::Wave { speed, time } => {
            let r = pool.add(vec![
                pool.mul(vec![w, dd(dd(t_mode, time, pool)?, time, pool)?]),
                pool.mul(vec![
                    pool.pow(speed, pool.integer(2)),
                    sl_op(t_mode, x, p, q, pool)?,
                ]),
            ]);
            let mut ck = check(
                "term_pde",
                r,
                "substitution",
                format!("{label}: w Tₜₜ + c² 𝓛T = 0"),
                pool,
            );
            if ck.status != ZeroStatus::ProvedZero {
                ck.status = super::zstatus(super::expand_powers_pos(simp(r, pool), pool), pool);
            }
            out.push(ck);
            out.push(check(
                "term_ic",
                at(t_mode, time, pool.integer(0), pool),
                "substitution",
                format!("{label}: T(0) = 0"),
                pool,
            ));
            out.push(check(
                "term_ic",
                sub(
                    at(dd(t_mode, time, pool)?, time, pool.integer(0), pool),
                    pool.mul(vec![phi, phi_xi]),
                    pool,
                ),
                "substitution",
                format!("{label}: Tₜ(0) = φ(x)φ(ξ)"),
                pool,
            ));
        }
    }
    Ok(out)
}

/// The eigenfunction-series Green's function of a Sturm–Liouville problem
/// (M3's [`Eigensystem`](super::super::Eigensystem)): see [`SeriesKind`].
///
/// Every family enters as `Sum(term, n, first, ∞)` (the index is the
/// family's own integer symbol, frequencies may be `RootOfFamily` nodes);
/// isolated modes as explicit terms.  A zero eigenvalue of the static
/// problem makes `G` not exist; the zero mode is then left out (the
/// **modified** Green's function, for data orthogonal to the mode, recorded as
/// a side condition), with `method = "modified_eigenfunction_series"`.
///
/// Checks, exact and term by term: `𝓛φ = λwφ` identically in the frequency;
/// the boundary conditions (M3's verification of the family); symmetry;
/// for heat and wave the PDE and initial values of every term; and for a
/// static problem with a closed-form family the generalised Fourier
/// coefficients of the closed-form [`ode_green_function`],
/// `∫ G(x, ξ) φₙ(x) w(x) dx = φₙ(ξ)/λₙ`, when `integrate` closes them.
///
/// # Errors
///
/// * the errors of [`sturm_liouville`](super::super::sturm_liouville());
/// * `E-PDE-251` — `ξ` (or the time) is not a symbol;
/// * `E-PDE-253` — a squared norm did not close, eigenvalues below the
///   threshold are present or conditional, or a family is not exactly
///   verified;
/// * `E-PDE-255` — a check refuted a term (never returned).
pub fn eigen_green_function(
    problem: &SturmLiouvilleProblem,
    xi: ExprId,
    kind: SeriesKind,
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    let x = problem.var;
    if !is_symbol(xi, pool) || xi == x {
        return Err(PdeError::GreenInvalidInput(
            "the source point ξ must be a symbol other than the variable".into(),
        ));
    }
    if let Some(t) = kind.time() {
        if !is_symbol(t, pool) {
            return Err(PdeError::GreenInvalidInput(format!(
                "the time {} is not a symbol",
                pool.display(t)
            )));
        }
    }
    let sys = sturm_liouville(problem, pool)?;
    match &sys.lower {
        LowerSpectrum::None => {}
        LowerSpectrum::Conditional { condition, .. } => {
            return Err(PdeError::GreenSeriesUnsupported(format!(
                "eigenvalues below the threshold are excluded only under {}; declare the \
                 parameters positive (or give numbers) so the sign is known",
                pool.display(*condition)
            )))
        }
        LowerSpectrum::Present { .. } => {
            return Err(PdeError::GreenSeriesUnsupported(
                "the problem has eigenvalues below the threshold, which the series does not \
                 include"
                    .into(),
            ))
        }
    }
    let mut terms: Vec<ExprId> = Vec::new();
    let mut checks: Vec<GreenCheck> = Vec::new();
    let mut side: Vec<ExprId> = sys.side_conditions.clone();
    let mut notes: Vec<String> = sys.notes.clone();
    let mut method = "eigenfunction_series".to_string();
    let (a, b) = sys.interval;
    side.push(pool.pred_gt(xi, a));
    side.push(pool.pred_lt(xi, b));
    for (mi, mode) in sys.modes.iter().enumerate() {
        if let Some(cond) = mode.condition {
            side.push(pool.pred_not(cond));
            notes.push(format!(
                "the mode λ = {} occurs only when {}; the series assumes it does not",
                pool.display(mode.eigenvalue),
                pool.display(cond)
            ));
            continue;
        }
        if mode.verification != VerificationStatus::ExactlyVerified {
            return Err(PdeError::GreenSeriesUnsupported(format!(
                "the mode λ = {} is not exactly verified",
                pool.display(mode.eigenvalue)
            )));
        }
        let norm = mode.norm_squared.ok_or_else(|| {
            PdeError::GreenSeriesUnsupported(format!(
                "the squared norm of the mode λ = {} did not close",
                pool.display(mode.eigenvalue)
            ))
        })?;
        let lam = simp(mode.eigenvalue, pool);
        if kind == SeriesKind::Static && lam == pool.integer(0) {
            method = "modified_eigenfunction_series".into();
            let phi0 = mode.eigenfunction;
            notes.push(format!(
                "λ = 0 is an eigenvalue (φ₀ = {}): no Green's function exists; this is the \
                 modified Green's function, valid for data f with ∫ f φ₀ dx = 0 (the solution \
                 is then unique up to adding a multiple of φ₀)",
                pool.display(phi0)
            ));
            continue;
        }
        let phi = mode.eigenfunction;
        terms.push(div(
            pool.mul(vec![
                time_factor(kind, lam, None, pool),
                phi,
                subs1(phi, x, xi, pool),
            ]),
            norm,
            pool,
        ));
        checks.extend(mode_checks(
            kind,
            phi,
            lam,
            x,
            xi,
            problem,
            &format!("mode {mi}"),
            pool,
        )?);
    }
    for (fi, fam) in sys.families.iter().enumerate() {
        if fam.verification.status != VerificationStatus::ExactlyVerified {
            return Err(PdeError::GreenSeriesUnsupported(format!(
                "eigen-family {fi} is not exactly verified ({})",
                fam.verification.status.as_str()
            )));
        }
        for (ei, (&phi_n, norm)) in fam
            .eigenfunctions
            .iter()
            .zip(&fam.norms_squared)
            .enumerate()
        {
            let norm = norm.ok_or_else(|| {
                PdeError::GreenSeriesUnsupported(format!(
                    "the squared norm of eigen-family {fi} did not close"
                ))
            })?;
            // √λₙ = νₙ when λ(ν) = ν² (the frequencies are the positive roots).
            let root = (simp(
                sub(
                    fam.eigenvalue_of_frequency,
                    pool.pow(fam.frequency_var, pool.integer(2)),
                    pool,
                ),
                pool,
            ) == pool.integer(0))
            .then_some(fam.frequency);
            let body = div(
                pool.mul(vec![
                    time_factor(kind, fam.eigenvalue, root, pool),
                    phi_n,
                    subs1(phi_n, x, xi, pool),
                ]),
                norm,
                pool,
            );
            terms.push(sum(
                simp(body, pool),
                fam.index,
                pool.integer(fam.first),
                pool.pos_infinity(),
                pool,
            )?);
            // Term checks identically in the frequency ν.
            let phi_nu = fam.eigenfunctions_of_frequency[ei];
            checks.extend(mode_checks(
                kind,
                phi_nu,
                fam.eigenvalue_of_frequency,
                x,
                xi,
                problem,
                &format!("family {fi}"),
                pool,
            )?);
        }
        let bc_ok = fam.verification.left == ZeroStatus::ProvedZero
            && fam.verification.right == ZeroStatus::ProvedZero;
        checks.push(GreenCheck {
            what: "term_bc",
            status: if bc_ok {
                ZeroStatus::ProvedZero
            } else {
                ZeroStatus::Unknown
            },
            method: format!("M3 eigenpair verification ({})", fam.verification.method),
            detail: format!("family {fi}: every term meets both boundary conditions"),
        });
    }
    if terms.is_empty() {
        return Err(PdeError::GreenSeriesUnsupported(
            "the eigensystem has no families".into(),
        ));
    }
    // Generalised Fourier coefficients of the closed form (static problems).
    if kind == SeriesKind::Static && method == "eigenfunction_series" {
        if let Ok(closed) = ode_green_function(problem, xi, pool) {
            if let Some((gl, gr)) = closed.branches {
                for (fi, fam) in sys.families.iter().enumerate() {
                    if !fam.closed_form || fam.eigenfunctions.len() != 1 {
                        continue;
                    }
                    let phi = fam.eigenfunctions[0];
                    let w = problem.w;
                    let s = fresh("s", Domain::Real, &[gl, gr, phi, w, x, xi], pool);
                    let left = simp(
                        pool.mul(vec![
                            subs1(gr, x, s, pool),
                            subs1(phi, x, s, pool),
                            subs1(w, x, s, pool),
                        ]),
                        pool,
                    );
                    let right = simp(
                        pool.mul(vec![
                            subs1(gl, x, s, pool),
                            subs1(phi, x, s, pool),
                            subs1(w, x, s, pool),
                        ]),
                        pool,
                    );
                    // ∫_a^ξ (x < ξ: x is "s", branch G(s < ξ)) …
                    let i1 = crate::integrate::integrate_definite(right, s, a, xi, pool);
                    let i2 = crate::integrate::integrate_definite(left, s, xi, b, pool);
                    if let (Ok(i1), Ok(i2)) = (i1, i2) {
                        let lhs = pool.add(vec![i1.value, i2.value]);
                        let want = div(subs1(phi, x, xi, pool), fam.eigenvalue, pool);
                        let mut ck = check(
                            "coefficients",
                            sub(lhs, want, pool),
                            "integrate (n a symbolic integer)",
                            format!(
                                "family {fi}: ∫ G_closed(x, ξ) φₙ(x) w(x) dx = φₙ(ξ)/λₙ, so the \
                                 series is the expansion of the closed form"
                            ),
                            pool,
                        );
                        if ck.status != ZeroStatus::ProvedZero {
                            let r = super::reduce_pi(
                                simp(super::expand_func_args(sub(lhs, want, pool), pool), pool),
                                pool,
                            );
                            let st = super::zstatus(r, pool);
                            if st == ZeroStatus::ProvedZero {
                                ck.status = st;
                            } else {
                                notes.push(format!(
                                    "the coefficient identity of family {fi} was not decided \
                                     symbolically"
                                ));
                                continue;
                            }
                        }
                        checks.push(ck);
                    }
                }
            }
        }
    }
    refuse_refuted("the eigenfunction-series Green's function", &checks)?;
    let status = combine(&checks);
    notes.push(match kind {
        SeriesKind::Static => "the eigenfunctions are complete in L²_w(a, b) (Sturm–Liouville \
             theory), so the series converges to G(·, ξ) in L²_w; it converges uniformly when \
             G is continuous"
            .into(),
        SeriesKind::Heat { .. } => "for t > 0 the series converges absolutely and uniformly \
             with all derivatives (e^{−κλₙt} decay); as t → 0⁺ it tends to δ(x − ξ)/w(ξ) by \
             completeness"
            .into(),
        SeriesKind::Wave { .. } => "the series converges in the sense of distributions; \
             Gₜ(·, 0) = δ(x − ξ)/w(ξ) by completeness"
            .into(),
    });
    let value = simp(pool.add(terms), pool);
    Ok(GreenFunction {
        operator: "sturm_liouville".into(),
        method,
        space: vec![x],
        source: vec![xi],
        time: kind.time(),
        value,
        series: None,
        branches: None,
        poisson_kernel: None,
        boundary_vars: Vec::new(),
        checks,
        status,
        side_conditions: side,
        notes,
        references: vec![
            "Haberman, Applied PDE §9.3.3–9.3.5 (eigenfunction expansion of Green's \
             functions), §9.4 (modified Green's functions)"
                .into(),
            "Strauss, PDE §7.1, §11.1".into(),
            "Duffy, Green's Functions with Applications §2.4–2.5 (eigenfunction expansions)".into(),
        ],
    })
}

/// `(α, β)` of a regular end condition.
fn end_coefficients(c: EndCondition, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    match c {
        EndCondition::Dirichlet => Some((pool.integer(1), pool.integer(0))),
        EndCondition::Neumann => Some((pool.integer(0), pool.integer(1))),
        EndCondition::Robin { alpha, beta } => Some((alpha, beta)),
        _ => None,
    }
}

/// A basis of `(p y')' + q y = 0`.
fn homogeneous_basis(
    problem: &SturmLiouvilleProblem,
    pool: &ExprPool,
) -> Result<[ExprId; 2], PdeError> {
    let x = problem.var;
    let p = simp(problem.p, pool);
    let q = simp(problem.q, pool);
    if q == pool.integer(0) && !super::mentions(p, x, pool) {
        return Ok([pool.integer(1), x]);
    }
    let avoid = [p, q, problem.a, problem.b, x];
    let y = fresh("y", Domain::Real, &avoid, pool);
    let (inp, yp, ypp) = crate::ode::dsolve::OdeInput::second_order(x, y, pool);
    let eq = pool.add(vec![
        pool.mul(vec![p, ypp]),
        pool.mul(vec![dd(p, x, pool)?, yp]),
        pool.mul(vec![q, y]),
    ]);
    let res = crate::ode::dsolve::dsolve(&inp.with_equation(eq), pool).map_err(|e| {
        PdeError::GreenUnsupported(format!("dsolve does not solve (p y')' + q y = 0: {e}"))
    })?;
    let sol = res
        .solutions
        .first()
        .ok_or_else(|| PdeError::GreenUnsupported("dsolve returned no solution".into()))?;
    if sol.constants.len() != 2 {
        return Err(PdeError::GreenUnsupported(
            "dsolve returned a solution without two constants".into(),
        ));
    }
    let (c1, c2) = (sol.constants[0], sol.constants[1]);
    let y1 = simp(dd(sol.y_of_x, c1, pool)?, pool);
    let y2 = simp(dd(sol.y_of_x, c2, pool)?, pool);
    if super::mentions(y1, c1, pool)
        || super::mentions(y1, c2, pool)
        || super::mentions(y2, c1, pool)
        || super::mentions(y2, c2, pool)
    {
        return Err(PdeError::GreenUnsupported(
            "dsolve's general solution is not linear in its constants".into(),
        ));
    }
    Ok([y1, y2])
}

/// The Green's function of `𝓛 = −[(p u')' + q u]` with the problem's
/// separated regular boundary conditions, by variation of parameters:
/// `G = −u₁(x_<) u₂(x_>) / (p W)` with `u₁` meeting the left condition, `u₂`
/// the right one and `p W = p (u₁u₂' − u₁'u₂)` constant (Abel).
///
/// [`GreenFunction::branches`] holds `(G for x < ξ, G for x > ξ)`;
/// [`GreenFunction::value`] joins them with `heaviside`.  Checks: `𝓛 G = 0`
/// on each side, each boundary condition, continuity at `x = ξ`, the jump
/// `[∂ₓG] = −1/p(ξ)` (so `𝓛 G = δ(x − ξ)` by the jump formula), symmetry,
/// and `(pW)' = 0`.
///
/// # Errors
///
/// * `E-PDE-250` — a non-separated, singular or infinite end, or `dsolve`
///   does not solve the homogeneous equation;
/// * `E-PDE-251` — `ξ` is not a symbol;
/// * `E-PDE-254` — `p W ≡ 0`: the homogeneous problem has a nontrivial
///   solution (`λ = 0` is an eigenvalue) and no Green's function exists;
/// * `E-PDE-255` — a check refuted the construction (never returned).
pub fn ode_green_function(
    problem: &SturmLiouvilleProblem,
    xi: ExprId,
    pool: &ExprPool,
) -> Result<GreenFunction, PdeError> {
    ode_green_parts(problem, xi, pool).map(|r| r.0)
}

/// [`ode_green_function`] with `(u₁, u₂, A)`, `G = A u₁(x_<) u₂(x_>)`.
pub(crate) fn ode_green_parts(
    problem: &SturmLiouvilleProblem,
    xi: ExprId,
    pool: &ExprPool,
) -> Result<(GreenFunction, ExprId, ExprId, ExprId), PdeError> {
    let x = problem.var;
    if !is_symbol(xi, pool) || xi == x {
        return Err(PdeError::GreenInvalidInput(
            "the source point ξ must be a symbol other than the variable".into(),
        ));
    }
    let (left, right) = match problem.boundary {
        SlBoundary::Separated { left, right } => (left, right),
        _ => {
            return Err(PdeError::GreenUnsupported(
                "variation of parameters needs separated boundary conditions (use \
                 eigen_green_function for periodic ones)"
                    .into(),
            ))
        }
    };
    let (ca, cb) = match (end_coefficients(left, pool), end_coefficients(right, pool)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return Err(PdeError::GreenUnsupported(
                "variation of parameters needs Dirichlet, Neumann or Robin conditions at both \
                 (finite) ends"
                    .into(),
            ))
        }
    };
    let (a, b) = (problem.a, problem.b);
    let (p, q) = (problem.p, problem.q);
    let [y1, y2] = homogeneous_basis(problem, pool)?;
    let functional =
        |f: ExprId, end: ExprId, (al, be): (ExprId, ExprId)| -> Result<ExprId, PdeError> {
            Ok(simp(
                pool.add(vec![
                    pool.mul(vec![al, at(f, x, end, pool)]),
                    pool.mul(vec![be, at(dd(f, x, pool)?, x, end, pool)]),
                ]),
                pool,
            ))
        };
    let u1 = simp(
        sub(
            pool.mul(vec![functional(y2, a, ca)?, y1]),
            pool.mul(vec![functional(y1, a, ca)?, y2]),
            pool,
        ),
        pool,
    );
    let u2 = simp(
        sub(
            pool.mul(vec![functional(y2, b, cb)?, y1]),
            pool.mul(vec![functional(y1, b, cb)?, y2]),
            pool,
        ),
        pool,
    );
    if u1 == pool.integer(0) || u2 == pool.integer(0) {
        return Err(PdeError::GreenUnsupported(
            "a boundary condition annihilates the whole basis".into(),
        ));
    }
    let wr = sub(
        pool.mul(vec![u1, dd(u2, x, pool)?]),
        pool.mul(vec![dd(u1, x, pool)?, u2]),
        pool,
    );
    let pw = simp(pool.mul(vec![p, wr]), pool);
    let mut checks = Vec::new();
    checks.push(check(
        "abel",
        dd(pw, x, pool)?,
        "substitution",
        "(p W)' = 0 (Abel's identity)",
        pool,
    ));
    let pw_c = simp(pw, pool);
    let pw_status = super::zstatus(pw_c, pool);
    if pw_status == ZeroStatus::ProvedZero {
        return Err(PdeError::GreenDoesNotExist(format!(
            "u₁ = {} satisfies both boundary conditions (p W = 0): λ = 0 is an eigenvalue",
            pool.display(u1)
        )));
    }
    let mut side = Vec::new();
    if pw_status != ZeroStatus::ProvedNonzero {
        side.push(pool.pred_ne(pw_c, pool.integer(0)));
    }
    side.push(pool.pred_gt(xi, a));
    side.push(pool.pred_lt(xi, b));
    let amp = neg(pool.pow(pw_c, pool.integer(-1)), pool);
    let g_left = simp(pool.mul(vec![amp, u1, subs1(u2, x, xi, pool)]), pool);
    let g_right = simp(pool.mul(vec![amp, subs1(u1, x, xi, pool), u2]), pool);
    for (g, label) in [(g_left, "x < ξ"), (g_right, "x > ξ")] {
        checks.push(check(
            "homogeneous",
            sl_op(g, x, p, q, pool)?,
            "substitution",
            format!("𝓛G = 0 for {label}"),
            pool,
        ));
    }
    checks.push(check(
        "bc",
        functional(g_left, a, ca)?,
        "substitution",
        "the left condition (on the branch x < ξ)",
        pool,
    ));
    checks.push(check(
        "bc",
        functional(g_right, b, cb)?,
        "substitution",
        "the right condition (on the branch x > ξ)",
        pool,
    ));
    checks.push(check(
        "continuity",
        sub(at(g_left, x, xi, pool), at(g_right, x, xi, pool), pool),
        "substitution",
        "G is continuous at x = ξ",
        pool,
    ));
    let jump = sub(
        at(dd(g_right, x, pool)?, x, xi, pool),
        at(dd(g_left, x, pool)?, x, xi, pool),
        pool,
    );
    checks.push(check(
        "jump",
        pool.add(vec![
            jump,
            pool.pow(subs1(p, x, xi, pool), pool.integer(-1)),
        ]),
        "jump_formula",
        "[∂ₓG](ξ) = −1/p(ξ), so 𝓛G = δ(x − ξ)",
        pool,
    ));
    let swap = pool.symbol("__green_swap", Domain::Real);
    let swapped = subs1(
        subs1(subs1(g_right, x, swap, pool), xi, x, pool),
        swap,
        xi,
        pool,
    );
    checks.push(check(
        "symmetry",
        sub(g_left, swapped, pool),
        "substitution",
        "G(x, ξ) = G(ξ, x)",
        pool,
    ));
    refuse_refuted("the variation-of-parameters Green's function", &checks)?;
    let status = combine(&checks);
    let value = pool.add(vec![
        pool.mul(vec![g_left, pool.func("heaviside", vec![sub(xi, x, pool)])]),
        pool.mul(vec![
            g_right,
            pool.func("heaviside", vec![sub(x, xi, pool)]),
        ]),
    ]);
    let gf = GreenFunction {
        operator: "sturm_liouville".into(),
        method: "variation_of_parameters".into(),
        space: vec![x],
        source: vec![xi],
        time: None,
        value,
        series: None,
        branches: Some((g_left, g_right)),
        poisson_kernel: None,
        boundary_vars: Vec::new(),
        checks,
        status,
        side_conditions: side,
        notes: vec![format!(
            "u₁ = {} (left condition), u₂ = {} (right condition), p W = {}",
            pool.display(u1),
            pool.display(u2),
            pool.display(pw_c)
        )],
        references: vec![
            "Haberman, Applied PDE §9.3.2 (variation of parameters), §9.3.6 (jump conditions)"
                .into(),
            "Duffy, Green's Functions with Applications §2.2".into(),
        ],
    };
    Ok((gf, u1, u2, amp))
}
