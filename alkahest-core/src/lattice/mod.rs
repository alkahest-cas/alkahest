//! Lattices over ℤ: LLL reduction, and an exact toolkit for the invariants a
//! sphere-packing or lattice-coding calculation asks for.
//!
//! Row convention: basis vectors are the **rows** of an `m × n` matrix, so
//! `basis[i] ∈ ℚⁿ` and the Gram matrix is `B Bᵀ`.
//!
//! # What lives here
//!
//! * [`lattice_reduce_rows`] and friends — LLL on an integer row basis, backed
//!   by FLINT's `fmpz_lll` and then *verified* against the exact rational LLL
//!   inequalities. See [`lll`] for why the verification is not redundant.
//! * [`Lattice`] — a rank-`m` lattice given by a basis **or** by a Gram matrix,
//!   with determinant, dual, shortest and closest vectors, theta series,
//!   kissing number and packing densities.
//! * [`zn`], [`a_n`], [`d_n`], [`e8`], [`leech`] — the standard lattices, each
//!   *constructed* rather than tabulated, and each checked against the counts
//!   that only the right lattice produces.
//!
//! # Scope limits
//!
//! SVP, CVP, minimal vectors and theta series are **exact enumeration**
//! (Fincke–Pohst, integer arithmetic throughout, no floating-point pruning).
//! Their cost is exponential in the rank. The rank is capped at
//! [`MAX_ENUM_RANK`] = 24 and every enumerating method takes a node budget;
//! above either, the answer is a typed refusal
//! ([`LatticeError::RankTooLarge`], [`LatticeError::EnumerationBudget`]) and
//! never a heuristic. If a *short* vector is what you need rather than *the
//! shortest*, reduce the basis and take the first row.
//!
//! Theta series additionally need an integral Gram matrix
//! ([`LatticeError::NonIntegralGram`]).

mod constructors;
mod enumerate;
mod flint_backend;
#[allow(clippy::module_inception)]
mod lattice;
pub mod lll;
mod quadform;

pub use constructors::{a_n, d_n, e8, leech, zn};
pub use lattice::{
    Lattice, LatticeVector, DEFAULT_ENUM_NODE_BUDGET, MAX_ENUM_RANK, MAX_THETA_NORM,
};
pub use lll::{
    lattice_reduce_rows, lattice_reduce_rows_exact, lattice_reduce_rows_with_delta,
    validate_lll_rows, LatticeError,
};

#[cfg(test)]
mod tests;
