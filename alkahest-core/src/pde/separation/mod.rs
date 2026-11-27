//! Separation of variables and eigenfunction-series solutions on bounded
//! domains (milestone M4 of the PDE plan, §8).
//!
//! # Separability ([`separate`](fn@separate))
//!
//! The product ansatz `u = X(x)T(t)…` (or the additive one for
//! Hamilton–Jacobi equations) is substituted, and `L[XT]/(XT)` is tested for
//! splitting in the factors' jet coordinates, with the classical multipliers
//! (`r²` in polar, `r² sin²θ` in spherical coordinates).  The result names
//! the separated ODEs and the separation constants.
//!
//! # Series solutions ([`separation_solve`])
//!
//! A linear equation on a product of intervals, a curvilinear region
//! ([`PdeDomain::Curvilinear`]: disk, annulus, sector, cylinder, ball) or a
//! Cartesian [`PdeDomain::Disk`] / [`PdeDomain::Ball`], possibly times a
//! half-line in time:
//!
//! 1. **Homogenise.**  Boundary data of an evolution problem are lifted: the
//!    steady state when the data and source do not depend on time (and it
//!    closes), otherwise a polynomial lifting `A + Bx` / `Ax + Bx²`; the
//!    remainder has homogeneous conditions and a source (Duhamel).  An
//!    elliptic problem with data on several faces is split by superposition
//!    (one sub-problem per data-carrying variable, the source with
//!    homogeneous data).
//! 2. **Separate** with the data-carrying direction `s` (time, or the
//!    direction of the data) last; each other direction is a Sturm–Liouville
//!    problem solved by [`sturm_liouville`](mod@super::sturm_liouville) (closed-form or
//!    `RootOfFamily` eigenvalues), with Legendre (associated Legendre: experimental) in the
//!    polar angle of a ball and spherical Bessel functions in its radius.  A
//!    constant-coefficient direction with a first-derivative term is first
//!    made self-adjoint by `X = e^{−σx}Y`, `σ = a₁/(2a₂)`.  A
//!    problem with no `s` (Poisson with homogeneous data) is expanded in
//!    every direction with a spectral parameter.
//! 3. **Project** the data on the eigenfunctions, `c = ∫ W f Φ / ‖Φ‖²`, with
//!    an exactly checked antiderivative (FTC), or keep the coefficient as an
//!    unevaluated `Integral(f, s, lo, hi)`, the reserved head of
//!    [`mod@super::integral`] (same name and argument order).  Indices where
//!    a closed-form coefficient's denominator vanishes are expanded
//!    separately, and data polynomial in `cos θ` give finite Legendre
//!    expansions.
//! 4. **Solve the amplitude** equation in `s` (constant-coefficient, Euler or
//!    Bessel type), with the initial values or two-point data, and Duhamel's
//!    integral for a source.
//! 5. **Assemble** a nested `Sum` over the indices.
//!
//! # Verification (plan §7)
//!
//! * every mode satisfies the PDE, identically in its indices (for a
//!   `RootOfFamily` member identically in its frequency): `L[TΦ] = Φ·M[T]`
//!   with `T` an undefined function, and the amplitude solves `M[T] = q`
//!   with its data;
//! * the eigenfunctions satisfy the homogeneous conditions (the
//!   Sturm–Liouville solver's exact checks);
//! * coefficients are exact by the fundamental theorem of calculus;
//! * the projection of the data is checked numerically on the first members
//!   ([`ProjectionCheck`]) — which may only refute; the truncated series is
//!   never claimed equal to the data pointwise.  [`SeriesSolution::convergence`]
//!   says in which sense the data are attained.
//!
//! # Errors
//!
//! `E-PDE-200 … E-PDE-209`.
//!
//! [`PdeDomain::Curvilinear`]: super::PdeDomain::Curvilinear
//! [`PdeDomain::Disk`]: super::PdeDomain::Disk
//! [`PdeDomain::Ball`]: super::PdeDomain::Ball

pub(crate) mod conditions;
pub mod coords;
pub(crate) mod modes;
pub(crate) mod numeric;
pub mod project;
pub mod separate;
pub mod solve;

pub use coords::{
    from_cartesian, isotropic_to_curvilinear, laplacian, to_cartesian, CoordinateSystem,
};
pub use numeric::numeric_truncation;
pub use project::{has_integral, integral_parts, INTEGRAL};
pub use separate::{
    is_separable, separate, separate_with, SeparateOptions, SeparatedOde, Separation,
    SeparationKind,
};
pub use solve::{
    separation_solve, separation_solve_with, ModeGroup, ProjectionCheck, SeparationOptions,
    SeriesCheck, SeriesSolution,
};

#[cfg(test)]
mod tests;
