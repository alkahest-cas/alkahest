//! Polynomial ideals — primary decomposition and radicals (V2-12).
//!
//! Gianni–Trager–Zacharias-style splitting is implemented via saturations and
//! univariate factorization (zero-dimensional factors in the lowest Lex
//! variable). This covers the roadmap examples; pathological ideals may need a
//! broader implementation in future work.

#[cfg(feature = "groebner")]
pub mod primary;

#[cfg(feature = "groebner")]
pub use primary::{primary_decomposition, radical, PrimaryComponent, PrimaryDecompositionError};
