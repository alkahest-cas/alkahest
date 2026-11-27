//! Green's functions and fundamental solutions (milestone M5 of the PDE plan).
//!
//! # Convention
//!
//! A Green's function of a linear operator `L` solves `L G = δ` — the
//! operator exactly as written, with no sign flip — which is the convention of
//! Mathematica's `GreenFunction`, so `Δ G = δ` gives `G = log r / (2π)` in
//! the plane.  Time-dependent kernels are causal (`G = 0` for `t < 0`) and
//! solve `L G = δ(x) δ(t)`; the Helmholtz kernels are outgoing for the time
//! factor `e^{−iωt}`.  The one exception is the eigenfunction series of a
//! Sturm–Liouville problem ([`eigen_green_function`]), which is for the
//! positive operator `𝓛u = −[(p u')' + q u]` whose eigenvalues
//! `𝓛φₙ = λₙ w φₙ` M3 computes: `𝓛 G = δ(x − ξ)` and
//! `G = Σ φₙ(x) φₙ(ξ) / (λₙ ‖φₙ‖²)`.
//!
//! # What lives here
//!
//! * [`fundamental_solution`] / [`fundamental_solution_nd`] — the table:
//!   Laplace (1–3 D and `n` D with symbolic `n`), heat (1–3 D and symbolic
//!   `n`), wave (d'Alembert, Poisson and Kirchhoff kernels in 1, 2, 3 D),
//!   Helmholtz and modified Helmholtz (1–3 D; Hankel `H₀⁽¹⁾` and `K₀` in the
//!   plane), biharmonic (1–3 D), Klein–Gordon and telegraph in 1–3 D (the
//!   light-cone kernels: `J₀`, `I₀` in 1 D; `cos`, `cosh` in 2 D by descent
//!   from the 3-D wave kernel; the light-cone `δ` plus a `J₁` / `I₁` tail in
//!   3 D by the dimension walk).  Each entry carries its
//!   derivation, the distributional identity it satisfies, references, and
//!   exact checks (below).
//! * [`image_green_function`] — the method of images: half- and
//!   quarter-spaces with Dirichlet or Neumann walls for every operator of the
//!   table that is invariant under the reflections; strips and slabs as an
//!   infinite image series (M0's `Sum` head, with the closed conformal-map
//!   form in the plane); the disk and the ball by Kelvin inversion, with the
//!   Poisson kernels.
//! * [`eigen_green_function`] — eigenfunction series on a bounded interval
//!   from M3's [`Eigensystem`](super::Eigensystem): static, heat and wave
//!   versions; [`ode_green_function`] — the closed form
//!   `G = −u₁(x_<) u₂(x_>) / (p W)` by variation of parameters.
//! * [`green_solve`] — the representation `u = ∫ G f + boundary terms` of a
//!   two-point boundary-value problem, of the heat and wave equations on the
//!   line with sources, and of Poisson / Dirichlet problems on the regions of
//!   the image method, closed wherever `integrate` succeeds.
//!
//! # Verification (D9)
//!
//! Every returned kernel is checked exactly:
//!
//! * **`homogeneous`** — `L G = 0` away from the singular set, by
//!   substitution (Cartesian coordinates, and the radial form `G(r)` for a
//!   symbolic dimension; Bessel residuals reduced by the three-term
//!   recurrences).
//! * **`distributional`** (1-D stationary operators) — `L G = δ` in the sense
//!   of distributions, by the jump formula: `L G± = 0` on both sides and
//!   `[G^{(j)}](0) = 0` for `j < m − 1`, `a_m [G^{(m−1)}](0) = 1`.
//! * **`flux`** — `lim_{r→0} |S^{n−1}| r^{n−1} ∂_r G = 1` (the strength of
//!   the point source); for Bessel kernels the limit is taken on the DLMF
//!   leading terms, cited in the check's method.
//! * **`mass`** / **`transfer`** — `∫ G dx = 1` for the heat kernel,
//!   `= t` for the wave kernels, and the spatial Fourier transform equal to
//!   the transfer function of the transformed ODE where `integrate` closes it.
//! * **`characteristic`** — the Riemann function of Klein–Gordon equals 1 on
//!   the characteristics `x = ±ct`.
//! * Image kernels: **`bc`** on every wall, **`pde`** in the region; Kelvin
//!   kernels: the inversion identity on the sphere; Poisson kernels: equal to
//!   the normal derivative of `G`, harmonic, unit mass where it closes.
//!
//! The theorem that turns these checks into `L G = δ` (approximate identities
//! for the heat kernel, Riemann's method for the light-cone kernels, the
//! divergence theorem for the flux) is recorded in
//! [`FundamentalSolution::identity`] with a reference; nothing numeric ever
//! certifies (numeric evidence could only refute).

use crate::kernel::subs::mentions_var;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::error::PdeError;
use super::sturm_liouville::{d, fresh_symbol, replace_node, simp, subs1};
use super::verify::VerificationStatus;

mod fundamental;
mod images;
mod light_cone_nd;
mod neumann_disk;
mod represent;
mod series;
mod waveguide;

pub(crate) use fundamental::light_cone;
pub use fundamental::{fundamental_solution, fundamental_solution_nd};
pub use images::{image_green_function, ImageRegion, WallCondition};
pub use neumann_disk::{neumann_disk_solve, NeumannDiskSolution};
pub use represent::{green_solve, GreenSolution, GreenSolutionKind, GreenTerm};
pub(crate) use series::ode_green_parts;
pub use series::{eigen_green_function, ode_green_function, SeriesKind};

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// A linear constant-coefficient operator of the table.  Parameters are
/// expressions (normally positive symbols).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GreenOperator {
    /// `Δ`.
    Laplace,
    /// `Δ²`.
    Biharmonic,
    /// `Δ + k²` (outgoing kernels).
    Helmholtz {
        /// The wavenumber `k`.
        k: ExprId,
    },
    /// `Δ − k²`.
    ModifiedHelmholtz {
        /// `k`.
        k: ExprId,
    },
    /// `∂ₜ − κ Δ`.
    Heat {
        /// The diffusivity `κ`.
        diffusivity: ExprId,
    },
    /// `∂ₜₜ − c² Δ`.
    Wave {
        /// The speed `c`.
        speed: ExprId,
    },
    /// `∂ₜₜ − c² Δ + m²`.
    KleinGordon {
        /// `c`.
        speed: ExprId,
        /// `m`.
        mass: ExprId,
    },
    /// `∂ₜₜ + 2a ∂ₜ − c² Δ`.
    Telegraph {
        /// `c`.
        speed: ExprId,
        /// The damping `a`.
        damping: ExprId,
    },
}

impl GreenOperator {
    /// `"laplace"`, `"biharmonic"`, `"helmholtz"`, `"modified_helmholtz"`,
    /// `"heat"`, `"wave"`, `"klein_gordon"` or `"telegraph"`.
    pub fn name(&self) -> &'static str {
        match self {
            GreenOperator::Laplace => "laplace",
            GreenOperator::Biharmonic => "biharmonic",
            GreenOperator::Helmholtz { .. } => "helmholtz",
            GreenOperator::ModifiedHelmholtz { .. } => "modified_helmholtz",
            GreenOperator::Heat { .. } => "heat",
            GreenOperator::Wave { .. } => "wave",
            GreenOperator::KleinGordon { .. } => "klein_gordon",
            GreenOperator::Telegraph { .. } => "telegraph",
        }
    }

    /// Whether the operator has a time variable.
    pub fn is_time_dependent(&self) -> bool {
        matches!(
            self,
            GreenOperator::Heat { .. }
                | GreenOperator::Wave { .. }
                | GreenOperator::KleinGordon { .. }
                | GreenOperator::Telegraph { .. }
        )
    }

    /// The parameters, by name.
    pub fn parameters(&self) -> Vec<(&'static str, ExprId)> {
        match *self {
            GreenOperator::Laplace | GreenOperator::Biharmonic => vec![],
            GreenOperator::Helmholtz { k } | GreenOperator::ModifiedHelmholtz { k } => {
                vec![("k", k)]
            }
            GreenOperator::Heat { diffusivity } => vec![("diffusivity", diffusivity)],
            GreenOperator::Wave { speed } => vec![("speed", speed)],
            GreenOperator::KleinGordon { speed, mass } => vec![("speed", speed), ("mass", mass)],
            GreenOperator::Telegraph { speed, damping } => {
                vec![("speed", speed), ("damping", damping)]
            }
        }
    }

    /// `L e` in the Cartesian variables `space` (and `time`).
    ///
    /// # Errors
    ///
    /// `E-PDE-251` when a time-dependent operator has no time variable, or
    /// `e` cannot be differentiated.
    pub fn apply(
        &self,
        e: ExprId,
        space: &[ExprId],
        time: Option<ExprId>,
        pool: &ExprPool,
    ) -> Result<ExprId, PdeError> {
        let lap = |f: ExprId| -> Result<ExprId, PdeError> {
            let mut parts = Vec::new();
            for &v in space {
                parts.push(dd(dd(f, v, pool)?, v, pool)?);
            }
            Ok(if parts.is_empty() {
                pool.integer(0)
            } else {
                pool.add(parts)
            })
        };
        let need_t = || {
            time.ok_or_else(|| {
                PdeError::GreenInvalidInput(format!(
                    "the {} operator needs a time variable",
                    self.name()
                ))
            })
        };
        let r = match *self {
            GreenOperator::Laplace => lap(e)?,
            GreenOperator::Biharmonic => lap(lap(e)?)?,
            GreenOperator::Helmholtz { k } => {
                pool.add(vec![lap(e)?, pool.mul(vec![sq(k, pool), e])])
            }
            GreenOperator::ModifiedHelmholtz { k } => {
                sub(lap(e)?, pool.mul(vec![sq(k, pool), e]), pool)
            }
            GreenOperator::Heat { diffusivity } => {
                let t = need_t()?;
                sub(dd(e, t, pool)?, pool.mul(vec![diffusivity, lap(e)?]), pool)
            }
            GreenOperator::Wave { speed } => {
                let t = need_t()?;
                sub(
                    dd(dd(e, t, pool)?, t, pool)?,
                    pool.mul(vec![sq(speed, pool), lap(e)?]),
                    pool,
                )
            }
            GreenOperator::KleinGordon { speed, mass } => {
                let t = need_t()?;
                pool.add(vec![
                    dd(dd(e, t, pool)?, t, pool)?,
                    neg(pool.mul(vec![sq(speed, pool), lap(e)?]), pool),
                    pool.mul(vec![sq(mass, pool), e]),
                ])
            }
            GreenOperator::Telegraph { speed, damping } => {
                let t = need_t()?;
                pool.add(vec![
                    dd(dd(e, t, pool)?, t, pool)?,
                    pool.mul(vec![pool.integer(2), damping, dd(e, t, pool)?]),
                    neg(pool.mul(vec![sq(speed, pool), lap(e)?]), pool),
                ])
            }
        };
        Ok(r)
    }

    /// `L e` for a radial `e(r[, t])` in dimension `n` (`Δ = ∂ᵣᵣ + (n−1)/r ∂ᵣ`).
    ///
    /// # Errors
    ///
    /// As [`Self::apply`].
    pub fn apply_radial(
        &self,
        e: ExprId,
        r: ExprId,
        n: ExprId,
        time: Option<ExprId>,
        pool: &ExprPool,
    ) -> Result<ExprId, PdeError> {
        let lap = |f: ExprId| -> Result<ExprId, PdeError> {
            let f1 = dd(f, r, pool)?;
            Ok(pool.add(vec![
                dd(f1, r, pool)?,
                pool.mul(vec![
                    sub(n, pool.integer(1), pool),
                    pool.pow(r, pool.integer(-1)),
                    f1,
                ]),
            ]))
        };
        let need_t = || {
            time.ok_or_else(|| {
                PdeError::GreenInvalidInput(format!(
                    "the {} operator needs a time variable",
                    self.name()
                ))
            })
        };
        Ok(match *self {
            GreenOperator::Laplace => lap(e)?,
            GreenOperator::Biharmonic => lap(lap(e)?)?,
            GreenOperator::Helmholtz { k } => {
                pool.add(vec![lap(e)?, pool.mul(vec![sq(k, pool), e])])
            }
            GreenOperator::ModifiedHelmholtz { k } => {
                sub(lap(e)?, pool.mul(vec![sq(k, pool), e]), pool)
            }
            GreenOperator::Heat { diffusivity } => {
                let t = need_t()?;
                sub(dd(e, t, pool)?, pool.mul(vec![diffusivity, lap(e)?]), pool)
            }
            GreenOperator::Wave { speed } => {
                let t = need_t()?;
                sub(
                    dd(dd(e, t, pool)?, t, pool)?,
                    pool.mul(vec![sq(speed, pool), lap(e)?]),
                    pool,
                )
            }
            _ => {
                return Err(PdeError::GreenUnsupported(format!(
                    "no radial form of the {} operator",
                    self.name()
                )))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// One exact check of a Green's function.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct GreenCheck {
    /// `"homogeneous"`, `"homogeneous_radial"`, `"distributional"`,
    /// `"flux"`, `"mass"`, `"transfer"`, `"concentration"`,
    /// `"characteristic"`, `"laplacian"`, `"bc"`, `"pde"`, `"kelvin"`,
    /// `"poisson_kernel"`, `"symmetry"`, `"term_ode"`, `"term_bc"`,
    /// `"term_pde"`, `"jump"`, `"continuity"`, `"coefficients"`,
    /// `"pairing"`, `"radiation"`, `"reduction"`, `"solution"`,
    /// `"descent"`, `"dimension_walk"`, `"commutation"`, `"longitudinal"`,
    /// `"source"`, `"boundary_kernel"`, `"compatibility"`, `"data"`.
    pub what: &'static str,
    /// The verdict (`proved_zero` for a passed check).
    pub status: ZeroStatus,
    /// How it was decided (`"substitution"`, `"jump_formula"`,
    /// `"limit"`, `"leading_term (DLMF 10.30.2)"`, `"integrate"`,
    /// `"sifting"`, …).
    pub method: String,
    /// What was checked, in words.
    pub detail: String,
}

/// An entry of the fundamental-solution table: see the [module docs](self).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FundamentalSolution {
    /// The operator.
    pub operator: GreenOperator,
    /// The dimension: an integer, or the symbol `n`.
    pub dimension: ExprId,
    /// The Cartesian space variables (`x − ξ`; empty for a symbolic
    /// dimension).
    pub space: Vec<ExprId>,
    /// The time variable (`t − τ`), for an evolution operator.
    pub time: Option<ExprId>,
    /// The radial variable `r = |x|` of [`Self::radial`].
    pub radius: ExprId,
    /// `G` as a distribution: the Cartesian kernel with its `heaviside`
    /// (causality / light-cone) and `diracdelta` factors; for a symbolic
    /// dimension, the radial form.
    pub kernel: ExprId,
    /// `G` on its support away from the singular set, as a function of `r`
    /// (and `t`).
    pub radial: ExprId,
    /// The smooth Cartesian part of the kernel on its support (no
    /// `heaviside` factor).
    pub smooth: ExprId,
    /// Where [`Self::smooth`] is the kernel: `t > 0`, `c t − |x| > 0`; `None`
    /// for all `x ≠ 0`.
    pub support: Option<ExprId>,
    /// The distributional identity, in words: `"Δ G = δ(x)"`,
    /// `"(∂ₜ − κΔ) G = δ(x) δ(t)"`, …
    pub identity: String,
    /// How the identity is established: `"jump_formula"` (proved by the
    /// checks) or the theorem connecting the checks to it, with a reference.
    pub identity_proof: String,
    /// The derivation, step by step.
    pub derivation: Vec<String>,
    /// The exact checks.
    pub checks: Vec<GreenCheck>,
    /// Exactly verified only when every check is a proof.
    pub status: VerificationStatus,
    /// Conditions on the parameters (predicates): `κ > 0`, `n ≥ 3`, …
    pub side_conditions: Vec<ExprId>,
    /// References (textbook sections, DLMF equations).
    pub references: Vec<String>,
}

/// A Green's function of a boundary-value problem: see [`image_green_function`],
/// [`eigen_green_function`], [`ode_green_function`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct GreenFunction {
    /// The operator name (`"laplace"`, …, or `"sturm_liouville"`).
    pub operator: String,
    /// `"images"`, `"image_series"`, `"conformal_map"`, `"kelvin"`,
    /// `"eigenfunction_series"`, `"modified_eigenfunction_series"`,
    /// `"variation_of_parameters"`.
    pub method: String,
    /// The field variables `x`.
    pub space: Vec<ExprId>,
    /// The source point `ξ`.
    pub source: Vec<ExprId>,
    /// The time variable (the source time is `0`).
    pub time: Option<ExprId>,
    /// `G(x, ξ[, t])`: closed form, or an infinite `Sum`.
    pub value: ExprId,
    /// The image or eigenfunction series, when [`Self::value`] is a closed
    /// form that also has one (strip in the plane).
    pub series: Option<ExprId>,
    /// For [`ode_green_function`]: the branches `(x < ξ, x > ξ)`.
    pub branches: Option<(ExprId, ExprId)>,
    /// The Poisson kernel `∂G/∂n_ξ` on the boundary (Dirichlet Laplace
    /// problems), in [`Self::space`] and [`Self::boundary_vars`].
    pub poisson_kernel: Option<ExprId>,
    /// The coordinates of a boundary point in [`Self::poisson_kernel`].
    pub boundary_vars: Vec<ExprId>,
    /// The exact checks.
    pub checks: Vec<GreenCheck>,
    /// Exactly verified only when every check is a proof.
    pub status: VerificationStatus,
    /// Conditions (predicates): the source in the region, parameter signs, …
    pub side_conditions: Vec<ExprId>,
    /// Hypotheses in prose (convergence, completeness, …).
    pub notes: Vec<String>,
    /// References.
    pub references: Vec<String>,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

pub(crate) fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

pub(crate) fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

pub(crate) fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, pool.pow(b, pool.integer(-1))])
}

pub(crate) fn sq(a: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(a, pool.integer(2))
}

pub(crate) fn sqrt(a: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("sqrt", vec![a])
}

pub(crate) fn pi(pool: &ExprPool) -> ExprId {
    pool.symbol("pi", Domain::Real)
}

pub(crate) fn dd(e: ExprId, v: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    d(e, v, pool).map_err(|err| PdeError::GreenInvalidInput(err.to_string()))
}

pub(crate) fn is_symbol(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Symbol { .. }))
}

/// A fresh symbol avoiding the symbols of `avoid`.
pub(crate) fn fresh(base: &str, domain: Domain, avoid: &[ExprId], pool: &ExprPool) -> ExprId {
    fresh_symbol(base, domain, avoid, pool)
}

/// Exact zero status: `simplify`, `zero_test`, then the Bessel recurrences.
pub(crate) fn zstatus(e: ExprId, pool: &ExprPool) -> ZeroStatus {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return ZeroStatus::ProvedZero;
    }
    let st = zero_test(s, pool, None).status;
    if st == ZeroStatus::ProvedZero {
        return st;
    }
    if let Some(r) = bessel_normal(s, pool) {
        let r = simp(r, pool);
        if r == pool.integer(0) {
            return ZeroStatus::ProvedZero;
        }
        let st2 = zero_test(r, pool, None).status;
        if st2 == ZeroStatus::ProvedZero || st2 == ZeroStatus::ProvedNonzero {
            return st2;
        }
    }
    // Second attempts at a proof (never at a refutation): Bessel functions
    // at 0, powers of positive factors distributed (symbolic exponents
    // included), `π` as a generic positive symbol, `|v|` split by sign.
    let folded = simp(fold_bessel_at_zero(s, pool), pool);
    if folded != s {
        let st2 = zstatus(folded, pool);
        if st2 == ZeroStatus::ProvedZero {
            return st2;
        }
    }
    if super::transform_methods::prove_harder(s, pool) {
        return ZeroStatus::ProvedZero;
    }
    let ex = simp(expand_func_args(s, pool), pool);
    if ex != s
        && (ex == pool.integer(0) || zero_test(ex, pool, None).status == ZeroStatus::ProvedZero)
    {
        return ZeroStatus::ProvedZero;
    }
    let has_trig = {
        let mut stack = vec![ex];
        let mut hit = false;
        while let Some(n) = stack.pop() {
            if let ExprData::Func { name, .. } = pool.get(n) {
                if matches!(name.as_str(), "sin" | "cos" | "sinh" | "cosh") {
                    hit = true;
                    break;
                }
            }
            super::jet::push_children(n, pool, &mut stack);
        }
        hit
    };
    if has_trig {
        let te = simp(
            expand_func_args(trig_to_exp(reduce_pi(ex, pool), pool), pool),
            pool,
        );
        if te == pool.integer(0) || zero_test(te, pool, None).status == ZeroStatus::ProvedZero {
            return ZeroStatus::ProvedZero;
        }
    }
    let pp = fresh("__green_pi", Domain::Positive, &[s], pool);
    let e1 = simp(
        expand_powers_pos(simp(subs1(s, pi(pool), pp, pool), pool), pool),
        pool,
    );
    if e1 == pool.integer(0) || zero_test(e1, pool, None).status == ZeroStatus::ProvedZero {
        return ZeroStatus::ProvedZero;
    }
    // A numeric "nonzero" through an opaque Sum / Bessel atom is not a
    // refutation; neither is an inconclusive sample.
    if st == ZeroStatus::ProvedNonzero && has_opaque(s, pool) {
        return ZeroStatus::Unknown;
    }
    st
}

/// `J₀(0) = I₀(0) = 1`, `J_ν(0) = I_ν(0) = 0` for an integer `ν > 0`.
pub(crate) fn fold_bessel_at_zero(e: ExprId, pool: &ExprPool) -> ExprId {
    let zero = pool.integer(0);
    let e = simp(replace_node(e, sqrt(zero, pool), zero, pool), pool);
    let mut out = e;
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if let Some((k, o, z)) = bessel_view(n, pool) {
            if matches!(k, 'J' | 'I') && simp(z, pool) == zero {
                let v = pool.with(o, |d| match d {
                    ExprData::Integer(i) => i.0.to_i64(),
                    _ => None,
                });
                match v {
                    Some(0) => out = replace_node(out, n, pool.integer(1), pool),
                    Some(m) if m > 0 => out = replace_node(out, n, zero, pool),
                    _ => {}
                }
                continue;
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    out
}

/// `(∏ fᵢ^{eᵢ})^e → ∏ fᵢ^{eᵢ e}` and `√z → z^{1/2}` when every factor `fᵢ`
/// is positive and the exponents are real — valid exactly there.  Unlike
/// the transform module's version the exponent may be symbolic (`−n/2`).
pub(crate) fn expand_powers_pos(e: ExprId, pool: &ExprPool) -> ExprId {
    use super::sturm_liouville::{sign, Sgn};
    fn real_exp(e: ExprId, pool: &ExprPool) -> bool {
        let mut stack = vec![e];
        while let Some(n) = stack.pop() {
            match pool.get(n) {
                ExprData::Symbol { domain, name, .. } => {
                    if name == "I"
                        || !matches!(
                            domain,
                            Domain::Real
                                | Domain::Positive
                                | Domain::Integer
                                | Domain::NonNegative
                                | Domain::NonZero
                        )
                    {
                        return false;
                    }
                }
                ExprData::Integer(_) | ExprData::Rational(_) => {}
                ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend(xs),
                _ => return false,
            }
        }
        true
    }
    fn split(b: ExprId, pool: &ExprPool) -> Option<Vec<(ExprId, ExprId)>> {
        let factors = match pool.get(b) {
            ExprData::Mul(xs) => xs,
            _ => vec![b],
        };
        let mut out = Vec::new();
        for f in factors {
            let (base, ex) = match pool.get(f) {
                ExprData::Pow { base, exp } => (base, exp),
                ExprData::Rational(r) => {
                    // p/q → p · q^{−1}
                    let (num, den) = (r.0.numer().clone(), r.0.denom().clone());
                    if num < 0 {
                        return None;
                    }
                    if num != 1 {
                        out.push((pool.integer(num), pool.integer(1)));
                    }
                    out.push((pool.integer(den), pool.integer(-1)));
                    continue;
                }
                _ => (f, pool.integer(1)),
            };
            if !(sign(base, pool) == Sgn::Pos || nonneg(base, pool)) || !real_exp(ex, pool) {
                return None;
            }
            out.push((base, ex));
        }
        Some(out)
    }
    // Sums and products of squares of real expressions and of positive
    // factors (√(ab) = √a √b needs a, b ≥ 0 only).
    fn nonneg(e: ExprId, pool: &ExprPool) -> bool {
        match pool.get(e) {
            ExprData::Integer(i) => i.0 >= 0,
            ExprData::Rational(r) => r.0 >= 0,
            ExprData::Symbol { domain, .. } => {
                matches!(domain, Domain::Positive | Domain::NonNegative)
                    || sign(e, pool) == Sgn::Pos
            }
            ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&x| nonneg(x, pool)),
            ExprData::Pow { base, exp } => {
                let even = pool.with(exp, |d| match d {
                    ExprData::Integer(i) => i.0.is_even(),
                    _ => false,
                });
                (even && real_exp(base, pool) || real_base(base, pool) && even)
                    || (nonneg(base, pool) && real_exp(exp, pool))
            }
            ExprData::Func { name, args } => {
                matches!(name.as_str(), "exp" | "cosh" | "abs") && real_base(args[0], pool)
                    || name == "sqrt" && nonneg(args[0], pool)
            }
            _ => false,
        }
    }
    // A real-valued polynomial-like expression (real symbols, no I).
    fn real_base(e: ExprId, pool: &ExprPool) -> bool {
        let mut stack = vec![e];
        while let Some(n) = stack.pop() {
            match pool.get(n) {
                ExprData::Symbol { domain, name, .. } => {
                    if name == "I" || domain == Domain::Complex {
                        return false;
                    }
                }
                ExprData::Integer(_) | ExprData::Rational(_) => {}
                ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend(xs),
                ExprData::Pow { base, exp } => {
                    let int = pool.with(exp, |d| matches!(d, ExprData::Integer(_)));
                    if !int {
                        return false;
                    }
                    stack.push(base);
                }
                _ => return false,
            }
        }
        true
    }
    fn go(e: ExprId, pool: &ExprPool, depth: usize) -> ExprId {
        if depth > 64 {
            return e;
        }
        match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| go(x, pool, depth + 1)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| go(x, pool, depth + 1)).collect()),
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                go(pool.pow(args[0], pool.rational(1, 2)), pool, depth + 1)
            }
            ExprData::Func { name, args } => {
                pool.func(name, args.iter().map(|&x| go(x, pool, depth + 1)).collect())
            }
            ExprData::Pow { base, exp } => {
                let base = go(base, pool, depth + 1);
                let atomic = !pool.with(base, |d| {
                    matches!(
                        d,
                        ExprData::Mul(_) | ExprData::Pow { .. } | ExprData::Rational(_)
                    )
                });
                if !atomic && real_exp(exp, pool) {
                    if let Some(parts) = split(base, pool) {
                        return pool.mul(
                            parts
                                .into_iter()
                                .map(|(b, ex)| {
                                    go(
                                        pool.pow(b, simp(pool.mul(vec![ex, exp]), pool)),
                                        pool,
                                        depth + 1,
                                    )
                                })
                                .collect(),
                        );
                    }
                }
                pool.pow(base, exp)
            }
            _ => e,
        }
    }
    go(e, pool, 0)
}

/// Distribute products over sums and expand small integer powers of sums.
pub(crate) fn expand_poly(e: ExprId, pool: &ExprPool) -> ExprId {
    fn terms(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match pool.get(e) {
            ExprData::Add(xs) => xs.iter().flat_map(|&x| terms(x, pool)).collect(),
            _ => vec![e],
        }
    }
    fn mul_out(a: &[ExprId], b: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
        let mut out = Vec::with_capacity(a.len() * b.len());
        for &x in a {
            for &y in b {
                out.push(pool.mul(vec![x, y]));
            }
        }
        out
    }
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| expand_poly(x, pool)).collect()),
        ExprData::Mul(xs) => {
            let mut acc = vec![pool.integer(1)];
            for x in xs {
                let ts = terms(expand_poly(x, pool), pool);
                if acc.len() * ts.len() > 4096 {
                    return e;
                }
                acc = mul_out(&acc, &ts, pool);
            }
            pool.add(acc)
        }
        ExprData::Pow { base, exp } => {
            let k = pool.with(exp, |d| match d {
                ExprData::Integer(i) => i.0.to_i64(),
                _ => None,
            });
            let b = expand_poly(base, pool);
            match k {
                Some(k) if (2..=6).contains(&k) && matches!(pool.get(b), ExprData::Add(_)) => {
                    let ts = terms(b, pool);
                    let mut acc = vec![pool.integer(1)];
                    for _ in 0..k {
                        if acc.len() * ts.len() > 4096 {
                            return e;
                        }
                        acc = mul_out(&acc, &ts, pool);
                    }
                    pool.add(acc)
                }
                _ => pool.pow(b, exp),
            }
        }
        _ => e,
    }
}

/// Expand the polynomial arguments of every function (so `e^{(a+b)²}` and
/// `e^{(−a−b)²}`, `log √((a+b)²)` and `log √((−a−b)²)` meet).
pub(crate) fn expand_func_args(e: ExprId, pool: &ExprPool) -> ExprId {
    fn go(e: ExprId, pool: &ExprPool, inside: bool) -> ExprId {
        match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| go(x, pool, inside)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| go(x, pool, inside)).collect()),
            ExprData::Pow { base, exp } => {
                let b = go(base, pool, inside);
                let p = pool.pow(b, go(exp, pool, inside));
                if inside {
                    expand_poly(p, pool)
                } else {
                    p
                }
            }
            ExprData::Func { name, args } => pool.func(
                name,
                args.iter()
                    .map(|&x| simp(expand_poly(go(x, pool, true), pool), pool))
                    .collect(),
            ),
            _ => e,
        }
    }
    go(e, pool, false)
}

/// `cos(z + mπ) = (−1)^m cos z`, `sin(z + mπ) = (−1)^m sin z` for an
/// integer `m` (after expanding the argument).
pub(crate) fn reduce_pi(e: ExprId, pool: &ExprPool) -> ExprId {
    let p = pi(pool);
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| reduce_pi(x, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| reduce_pi(x, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(reduce_pi(base, pool), reduce_pi(exp, pool)),
        ExprData::Func { name, args }
            if args.len() == 1 && matches!(name.as_str(), "sin" | "cos") =>
        {
            let a = simp(expand_poly(reduce_pi(args[0], pool), pool), pool);
            let ts: Vec<ExprId> = match pool.get(a) {
                ExprData::Add(xs) => xs,
                _ => vec![a],
            };
            let mut m: i64 = 0;
            let mut rest = Vec::new();
            for t in ts {
                let c = simp(div(t, p, pool), pool);
                match pool.with(c, |d| match d {
                    ExprData::Integer(i) => i.0.to_i64(),
                    _ => None,
                }) {
                    Some(k) => m += k,
                    None => rest.push(t),
                }
            }
            if m == 0 {
                return pool.func(name, vec![a]);
            }
            let z = if rest.is_empty() {
                pool.integer(0)
            } else {
                pool.add(rest)
            };
            let f = pool.func(name, vec![z]);
            if m % 2 == 0 {
                f
            } else {
                neg(f, pool)
            }
        }
        ExprData::Func { name, args } => {
            pool.func(name, args.iter().map(|&x| reduce_pi(x, pool)).collect())
        }
        _ => e,
    }
}

/// A check that every residual of a list vanishes (linear operators on a
/// sum of image terms: each term separately).
pub(crate) fn check_all(
    what: &'static str,
    residuals: &[ExprId],
    method: &str,
    detail: impl Into<String>,
    pool: &ExprPool,
) -> GreenCheck {
    let detail = detail.into();
    let mut worst = ZeroStatus::ProvedZero;
    for &r in residuals {
        let c = check(what, r, method, detail.clone(), pool);
        match c.status {
            ZeroStatus::ProvedZero => {}
            ZeroStatus::ProvedNonzero => {
                worst = ZeroStatus::ProvedNonzero;
                break;
            }
            s => {
                if worst == ZeroStatus::ProvedZero {
                    worst = s;
                }
            }
        }
    }
    GreenCheck {
        what,
        status: worst,
        method: method.to_string(),
        detail,
    }
}

/// In every product, merge powers of one base whose sign is positive:
/// `b^p · b^q = b^{p+q}` (exact for `b > 0`).
pub(crate) fn merge_powers(e: ExprId, pool: &ExprPool) -> ExprId {
    use super::sturm_liouville::{sign, Sgn};
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| merge_powers(x, pool)).collect()),
        ExprData::Mul(xs) => {
            let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
            let mut rest = Vec::new();
            for x in xs {
                let x = merge_powers(x, pool);
                let (b, ex) = match pool.get(x) {
                    ExprData::Pow { base, exp } => (base, exp),
                    _ => (x, pool.integer(1)),
                };
                if sign(b, pool) == Sgn::Pos {
                    match groups.iter_mut().find(|(bb, _)| *bb == b) {
                        Some(g) => g.1.push(ex),
                        None => groups.push((b, vec![ex])),
                    }
                } else {
                    rest.push(x);
                }
            }
            for (b, exps) in groups {
                rest.push(pool.pow(b, simp(pool.add(exps), pool)));
            }
            pool.mul(rest)
        }
        ExprData::Pow { base, exp } => pool.pow(merge_powers(base, pool), exp),
        ExprData::Func { name, args } => {
            pool.func(name, args.iter().map(|&x| merge_powers(x, pool)).collect())
        }
        _ => e,
    }
}

fn has_opaque(e: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if let ExprData::Func { name, .. } = pool.get(n) {
            if name == "Sum" || name == "RootOfFamily" || name == "Integral" {
                return true;
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    false
}

/// A check record from a residual.
pub(crate) fn check(
    what: &'static str,
    residual: ExprId,
    method: &str,
    detail: impl Into<String>,
    pool: &ExprPool,
) -> GreenCheck {
    let status = zstatus(residual, pool);
    GreenCheck {
        what,
        status,
        method: method.to_string(),
        detail: detail.into(),
    }
}

/// The overall status of a list of checks: refuted if any is proved
/// nonzero, exactly verified if all are proved zero, else unverified.
pub(crate) fn combine(checks: &[GreenCheck]) -> VerificationStatus {
    if checks.iter().any(|c| c.status == ZeroStatus::ProvedNonzero) {
        VerificationStatus::Refuted
    } else if !checks.is_empty() && checks.iter().all(|c| c.status == ZeroStatus::ProvedZero) {
        VerificationStatus::ExactlyVerified
    } else {
        VerificationStatus::Unverified
    }
}

/// Refuse a refuted construction (`E-PDE-255`).
pub(crate) fn refuse_refuted(what: &str, checks: &[GreenCheck]) -> Result<(), PdeError> {
    let bad: Vec<String> = checks
        .iter()
        .filter(|c| c.status == ZeroStatus::ProvedNonzero)
        .map(|c| format!("{} ({})", c.what, c.detail))
        .collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(PdeError::GreenVerificationFailed(format!(
            "{what} failed {}",
            bad.join(", ")
        )))
    }
}

/// Canonical Bessel kind letter of a node: `(kind, order, argument)`.
fn bessel_view(e: ExprId, pool: &ExprPool) -> Option<(char, ExprId, ExprId)> {
    pool.with(e, |d| match d {
        ExprData::Func { name, args } => match (name.as_str(), args.len()) {
            ("bessel_j", 2) => Some(('J', args[0], args[1])),
            ("bessel_y", 2) => Some(('Y', args[0], args[1])),
            ("bessel_i", 2) => Some(('I', args[0], args[1])),
            ("bessel_k", 2) => Some(('K', args[0], args[1])),
            ("hankel1", 2) => Some(('H', args[0], args[1])),
            ("hankel2", 2) => Some(('G', args[0], args[1])),
            ("bessel_j0", 1) => Some(('J', pool.integer(0), args[0])),
            ("bessel_j1", 1) => Some(('J', pool.integer(1), args[0])),
            _ => None,
        },
        _ => None,
    })
}

fn bessel_name(kind: char) -> &'static str {
    match kind {
        'J' => "bessel_j",
        'Y' => "bessel_y",
        'I' => "bessel_i",
        'K' => "bessel_k",
        'H' => "hankel1",
        _ => "hankel2",
    }
}

/// Rewrite the Bessel-type functions of one argument whose orders differ by
/// integers in terms of the two lowest orders, by the three-term
/// recurrences (DLMF 10.6.1, 10.29.1):
/// `C_{m+1} = (2m/z) C_m − C_{m−1}` for `C = J, Y, H⁽¹⁾, H⁽²⁾`,
/// `I_{m+1} = I_{m−1} − (2m/z) I_m`, `K_{m+1} = K_{m−1} + (2m/z) K_m`.
/// `bessel_j0` / `bessel_j1` are folded into `bessel_j`.  `None` when
/// nothing changes.
pub(crate) fn bessel_normal(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let mut nodes: Vec<(char, ExprId, ExprId, ExprId)> = Vec::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some((k, o, z)) = bessel_view(n, pool) {
            nodes.push((k, o, z, n));
            continue;
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    if nodes.is_empty() {
        return None;
    }
    let mut out = e;
    let mut changed = false;
    let mut groups: Vec<(char, ExprId)> = nodes.iter().map(|&(k, _, z, _)| (k, z)).collect();
    groups.sort();
    groups.dedup();
    for (kind, z) in groups {
        let members: Vec<(ExprId, ExprId)> = nodes
            .iter()
            .filter(|&&(k, _, zz, _)| k == kind && zz == z)
            .map(|&(_, o, _, n)| (o, n))
            .collect();
        let base = members[0].0;
        let mut offs = Vec::new();
        for &(o, n) in &members {
            let dlt = simp(sub(o, base, pool), pool);
            let Some(k) = pool.with(dlt, |d| match d {
                ExprData::Integer(i) => i.0.to_i64(),
                _ => None,
            }) else {
                return None;
            };
            offs.push((k, n));
        }
        let lo = offs.iter().map(|&(k, _)| k).min()?;
        let hi = offs.iter().map(|&(k, _)| k).max()?;
        if hi - lo > 40 {
            continue;
        }
        let ord = |k: i64| simp(pool.add(vec![base, pool.integer(k)]), pool);
        let name = bessel_name(kind);
        let mut val: std::collections::HashMap<i64, ExprId> = std::collections::HashMap::new();
        val.insert(lo, pool.func(name, vec![ord(lo), z]));
        val.insert(lo + 1, pool.func(name, vec![ord(lo + 1), z]));
        for m in lo + 1..hi {
            let tmz = pool.mul(vec![pool.integer(2), ord(m), pool.pow(z, pool.integer(-1))]);
            let next = match kind {
                'I' => sub(val[&(m - 1)], pool.mul(vec![tmz, val[&m]]), pool),
                'K' => pool.add(vec![val[&(m - 1)], pool.mul(vec![tmz, val[&m]])]),
                _ => sub(pool.mul(vec![tmz, val[&m]]), val[&(m - 1)], pool),
            };
            val.insert(m + 1, next);
        }
        for &(k, n) in &offs {
            if val[&k] != n {
                out = replace_node(out, n, val[&k], pool);
                changed = true;
            }
        }
    }
    changed.then_some(out)
}

/// Replace `abs(v)` by `sign·v` (`sign = ±1`).
pub(crate) fn branch_abs(e: ExprId, v: ExprId, sign: i64, pool: &ExprPool) -> ExprId {
    let a = pool.func("abs", vec![v]);
    let by = if sign >= 0 { v } else { neg(v, pool) };
    simp(replace_node(e, a, by, pool), pool)
}

/// Rewrite `sin`, `cos`, `sinh`, `cosh` as exponentials.
pub(crate) fn trig_to_exp(e: ExprId, pool: &ExprPool) -> ExprId {
    let mut nodes = Vec::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if let ExprData::Func { name, args } = pool.get(n) {
            if args.len() == 1 && matches!(name.as_str(), "sin" | "cos" | "sinh" | "cosh") {
                nodes.push((name, args[0], n));
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    let i = pool.imaginary_unit();
    let mut out = e;
    for (name, a, n) in nodes {
        let a = trig_to_exp(a, pool);
        let (p, m) = match name.as_str() {
            "sin" | "cos" => (
                pool.func("exp", vec![pool.mul(vec![i, a])]),
                pool.func("exp", vec![neg(pool.mul(vec![i, a]), pool)]),
            ),
            _ => (
                pool.func("exp", vec![a]),
                pool.func("exp", vec![neg(a, pool)]),
            ),
        };
        let rep = match name.as_str() {
            "sin" => div(sub(p, m, pool), pool.mul(vec![pool.integer(2), i]), pool),
            "sinh" => pool.mul(vec![pool.rational(1, 2), sub(p, m, pool)]),
            _ => pool.mul(vec![pool.rational(1, 2), pool.add(vec![p, m])]),
        };
        out = replace_node(out, n, rep, pool);
    }
    out
}

/// Whether `e` mentions `v`.
pub(crate) fn mentions(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    mentions_var(e, v, pool)
}

/// `subs1` then `simplify`.
pub(crate) fn at(e: ExprId, v: ExprId, by: ExprId, pool: &ExprPool) -> ExprId {
    simp(subs1(e, v, by, pool), pool)
}

#[cfg(test)]
mod tests;
