//! V2-6 — LLL lattice reduction (exact integer arithmetic over ℚ projections).
//!
//! Row convention: basis vectors are the rows of an `m × n` matrix stacked as
//! `basis[i]` ∈ ℤⁿ⁺⋯ (each row shares the ambient dimension).

pub mod lll;

pub use lll::{
    lattice_reduce_rows, lattice_reduce_rows_with_delta, validate_lll_rows, LatticeError,
};
