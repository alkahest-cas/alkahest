//! Polynomial ideals — primary decomposition and radicals (V2-12).
//!
//! Gianni–Trager–Zacharias-style splitting is implemented via saturations and
//! univariate factorization. Only the ideal classes whose components can be
//! *certified* primary — and whose radicals can be certified radical — are
//! answered; everything else refuses through [`IdealRefusal`] rather than
//! reporting the input ideal as though it were its own radical. See
//! [`primary`] for the full list of what is certified and why.

#[cfg(feature = "groebner")]
pub mod primary;

#[cfg(feature = "groebner")]
pub use primary::{
    primary_decomposition, radical, take_ideal_refusal, IdealRefusal, PrimaryComponent,
    PrimaryDecompositionError,
};
