//! Finite Expression Method (FEX) evaluation engine — **experimental**.
//!
//! FEX [Liang & Yang, *Finite Expression Method for Solving High-Dimensional
//! PDEs*, arXiv 2206.10121 / JMLR 26 (2025)] searches over closed-form
//! candidates `u(x) = T(e, θ; x)` built from a fixed binary tree of operator
//! slots, scoring each skeleton `e` by how well its best parameters `θ`
//! satisfy a PDE. Candidate scoring dominates the cost. This module makes it
//! fast and exact:
//!
//! * [`Template`] — the template IR (R1): tree shape, operator tables
//!   ([`UnaryOp`], [`BinaryOp`]), leaves `Σᵢ αᵢ φ(xᵢ) + β` contracted over
//!   the `d` coordinates, and [`Template::instantiate`] to an ordinary
//!   Alkahest expression;
//! * [`canonical_key`], [`DedupCache`], [`redundant_params`],
//!   [`same_function`] — canonicalisation and deduplication (R5);
//! * [`Evaluator`] — the template interpreter (R2–R4): fused exact
//!   `u, ∇u, Δu` (Forward-Laplacian structure, O(d) per node), the PDE and
//!   boundary residuals and the full parameter Jacobian in one pass, batched
//!   over candidates and points, with overflow guards;
//! * [`Problem`] / [`PdeOperator`] — the PDE, as a constant-coefficient
//!   linear-plus-cubic form (Poisson, transport, cubic Schrödinger) or an
//!   arbitrary expression in [`JetSymbols`], with least-squares,
//!   normalised and Rayleigh-quotient losses;
//! * [`fit`] / [`fit_batch`] — Levenberg–Marquardt, Adam and BFGS.
//!
//! Dimension `d` is concrete here. Exact verification for symbolic `d`
//! needs the `Sum` head and is out of scope for this module.
//!
//! Error codes are `E-FEX-001` … `E-FEX-008` ([`FexError`]).

mod canon;
mod error;
mod eval;
mod fit;
mod ops;
mod problem;
mod template;

#[cfg(test)]
mod tests;

pub use canon::{
    canonical_key, canonical_skeletons, redundant_params, same_function, DedupCache, DedupEntry,
};
pub use error::FexError;
pub use eval::{Evaluator, Jets};
pub use fit::{fit, fit_batch, FitMethod, FitOptions, FitResult};
pub use ops::{BinaryOp, UnaryOp, ALL_UNARY, PAPER_BINARY, PAPER_UNARY};
pub use problem::{
    CompiledOperatorHandle, CoordFn, JetSymbols, LinearOperator, LossKind, PdeOperator, Problem,
    ScalarField,
};
pub use template::{shape_preset, SlotInfo, SlotKind, Template, MAX_TEMPLATE_NODES};
