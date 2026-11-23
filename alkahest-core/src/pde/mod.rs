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
//! * [`error`] — [`PdeError`], codes `E-PDE-001 … E-PDE-049`.

pub mod classify;
pub mod error;
pub mod jet;
pub(crate) mod numeric;
pub mod problem;
pub mod root_family;
pub mod sum;
pub mod transform;
pub mod verify;

pub use classify::{classify_pde, Linearity, NamedEquation, PdeInfo, PdeType, TypeRegion};
pub use error::PdeError;
pub use jet::{
    functions_to_jets, multi_indices, JetCoordinate, JetNaming, JetSpace, JetSymbol, JetSystem,
    MultiIndex,
};
pub use problem::{ArbitraryFunctions, ConditionKind, PdeCondition, PdeDomain, PdeProblem};
pub use root_family::{
    member, root_family_enclosure, root_family_parts, root_of_family, ROOT_OF_FAMILY,
};
pub use sum::{
    alpha_normalize, coefficient, expand_finite_sums, expand_sum, indexed, indexed_parts,
    kronecker_delta, laplacian_indexed, simplify_sums, sum, sum_parts, truncate, IndexRange,
    INDEXED, KRONECKER_DELTA, MAX_SUM_TERMS, SUM,
};
pub use transform::{transformed_name, TransformKind};
pub use verify::{
    verify_pde_solution, verify_solution, PdeVerification, ResidualCheck, VerificationStatus,
};
