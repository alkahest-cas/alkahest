//! The PDE foundation layer (milestone M0 of the PDE plan).
//!
//! **Experimental surface.**  Everything here is new and additive; the Python
//! API lives under `alkahest.experimental` until it has been exercised by the
//! solver milestones built on it.
//!
//! # Representation (decisions D1/D2)
//!
//! An unknown is an applied undefined function `u(x, t)` and its partial
//! derivatives are the reserved `Derivative(u(x, t), kₓ, kₜ)` node of
//! [`crate::diff::undefined`] — canonical per argument slot, hash-consed, with
//! an exact chain rule.  No kernel node is added.
//!
//! # What lives here
//!
//! * [`jet`] — [`JetSpace`]: the multi-index bijection `Derivative ↔ u_α`,
//!   total derivatives, prolongation, polynomial-system conversion, and
//!   adapters showing the ODE spellings (`y'`, `q_dot`, `dy/dt`) are
//!   instances of it; [`functions_to_jets`] generalises
//!   [`crate::diff::functions_to_symbols`] to several variables.
//! * [`problem`] — [`PdeProblem`] (equations, unknowns, [`PdeDomain`],
//!   boundary / initial [`PdeCondition`]s) and [`ArbitraryFunctions`], the
//!   `F1, F2, …` generator for general solutions (D6).
//! * [`classify`] — [`classify_pde`] → [`PdeInfo`]: order, linearity class,
//!   homogeneity, constant coefficients, elliptic / parabolic / hyperbolic
//!   with regions for variable coefficients, named-equation matching.
//! * [`verify`] — [`verify_pde_solution`]: residuals of the PDE, BCs and ICs
//!   with the `zero_test` status vocabulary; numeric evidence may only refute
//!   (D9).
//! * [`mod@sum`] — the reserved heads `Sum(body, k, lo, hi)` (a binder; symbolic
//!   bounds such as a dimension `d`), `Indexed(x, i)` and
//!   `KroneckerDelta(i, j)`, with term-wise `diff`, expansion, truncation,
//!   coefficients and folding — enough for `Δ Σᵢ φ(xᵢ) = Σᵢ φ''(xᵢ)` with
//!   symbolic `d` ([`laplacian_indexed`]).
//! * [`root_family`] — `RootOfFamily(F, λ, n, branch)`, the `n`-th root of a
//!   transcendental equation, with certified ball-arithmetic enclosures.
//! * [`transform`] — `laplace_transform` / `fourier_transform` of unknowns and
//!   their derivatives (`s^k U − …` with initial values, `(2πiξ)^k û`).
//! * [`solve`] — [`pdsolve`]: the solver, a dispatcher over method tiers
//!   (M1: first-order linear / semilinear / quasilinear equations by
//!   characteristics, Cauchy problems, complete integrals); every answer is
//!   verified, codes `E-PDE-050 … E-PDE-099`.
//! * [`second_order`] — milestone M2: canonical forms, factored
//!   constant-coefficient operators, d'Alembert, the Laplace cascade
//!   (codes `E-PDE-100 … E-PDE-149`).
//! * [`mod@sturm_liouville`] — (milestone M3) the shared eigenproblem solver:
//!   [`SturmLiouvilleProblem`] → [`Eigensystem`] (closed-form or
//!   `RootOfFamily` eigenvalues with certified enclosures, eigenfunctions,
//!   weights, norms, `λ = 0` / negative-eigenvalue case analysis; Bessel,
//!   Legendre, Chebyshev, Hermite and Laguerre problems).
//! * [`transform_methods`] — (M3) [`fourier_method`], [`laplace_method`],
//!   [`image_method`] (sine / cosine transforms) and [`transform_solve`]:
//!   closed forms or verified kernel representations.
//! * [`green`] — (milestone M5) Green's functions: the fundamental-solution
//!   table ([`fundamental_solution`]), the method of images
//!   ([`image_green_function`], Kelvin inversion for the disk and ball),
//!   eigenfunction-series and variation-of-parameters Green's functions
//!   ([`eigen_green_function`], [`ode_green_function`]) and the
//!   representation `u = ∫ G f + boundary terms` ([`green_solve`]).
//! * [`error`] — [`PdeError`], codes `E-PDE-001 … E-PDE-049` (M0),
//!   `E-PDE-150 … E-PDE-199` (M3) and `E-PDE-250 … E-PDE-299` (M5).

pub mod classify;
pub mod error;
pub mod green;
pub mod integral;
pub mod jet;
pub mod mol;
pub(crate) mod numeric;
pub mod problem;
pub mod root_family;
pub mod second_order;
pub mod solve;
pub mod sturm_liouville;
pub mod sum;
pub mod transform;
pub mod transform_methods;
pub mod verify;

pub use classify::{classify_pde, Linearity, NamedEquation, PdeInfo, PdeType, TypeRegion};
pub use error::PdeError;
pub use green::{
    eigen_green_function, fundamental_solution, fundamental_solution_nd, green_solve,
    image_green_function, ode_green_function, FundamentalSolution, GreenCheck, GreenFunction,
    GreenOperator, GreenSolution, GreenSolutionKind, GreenTerm, ImageRegion, SeriesKind,
    WallCondition,
};
pub use integral::{
    combine_integrals, has_integral, indefinite_integral, indefinite_parts, integral,
    integral_parts, INTEGRAL,
};
pub use jet::{
    functions_to_jets, multi_indices, JetCoordinate, JetNaming, JetSpace, JetSymbol, JetSystem,
    MultiIndex,
};
pub use problem::{ArbitraryFunctions, ConditionKind, PdeCondition, PdeDomain, PdeProblem};
pub use root_family::{
    member, root_family_enclosure, root_family_parts, root_of_family, ROOT_OF_FAMILY,
};
pub use second_order::{
    canonical_form, canonical_forms, laplace_cascade, laplace_invariants, solve_second_order,
    solve_second_order_with, CanonicalForm, SecondOrderOptions, SecondOrderSolution,
    DEFAULT_CASCADE_STEPS,
};
pub use solve::{
    pdsolve, pdsolve_all, pdsolve_hints, PdeSolution, PdeSolutionKind, PdsolveOptions, Tier, TIERS,
};
pub use sturm_liouville::{
    family_enclosures, sturm_liouville, sturm_liouville_with, EigenEnclosure, EigenFamily,
    EigenMode, EigenVerification, Eigensystem, EndCondition, FamilyRef, LowerSpectrum, SlBoundary,
    SlOptions, SturmLiouvilleProblem,
};
pub use sum::{
    alpha_normalize, coefficient, expand_finite_sums, expand_sum, indexed, indexed_parts,
    kronecker_delta, laplacian_indexed, simplify_sums, sum, sum_parts, truncate, IndexRange,
    INDEXED, KRONECKER_DELTA, MAX_SUM_TERMS, SUM,
};
pub use transform::{transformed_name, TransformKind};
pub use transform_methods::{
    fourier_method, image_method, laplace_method, transform_solve, KernelTerm, SolutionKind,
    TransformCheck, TransformSolution,
};
pub use verify::{
    verify_pde_solution, verify_solution, PdeVerification, ResidualCheck, VerificationStatus,
};
