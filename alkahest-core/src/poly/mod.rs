pub mod error;
// V2-3 — Sparse interpolation
pub mod interp;
pub mod multipoly;
pub mod rational;
// V2-2 — Resultants and subresultant PRS
pub mod resultant;
pub mod unipoly;

#[cfg(feature = "groebner")]
pub mod groebner;

#[cfg(test)]
mod proptests;

pub use error::ConversionError;
// V2-3 — Sparse interpolation
pub use interp::{sparse_interpolate, sparse_interpolate_univariate, SparseInterpError};
pub use multipoly::MultiPoly;
pub use rational::RationalFunction;
// V2-2 — Resultants and subresultant PRS
pub use resultant::{resultant, subresultant_prs, ResultantError};
pub use unipoly::UniPoly;

use crate::kernel::{ExprId, ExprPool};

/// Normalize a polynomial expression to canonical sum-of-products form.
///
/// Converts `expr` to a [`MultiPoly`] (expanding and collecting all terms)
/// then converts back to a symbolic expression.  The result is sorted by the
/// [`std::collections::BTreeMap`] ordering on exponent vectors, which is lexicographic ascending.
///
/// Returns `Err` if `expr` is not a polynomial in `vars` (non-polynomial
/// function, non-constant exponent, non-integer coefficient, etc.).
///
/// # Example
/// ```text
/// poly_normal((x+1)*(x-1), [x])  →  x² + (-1)
/// poly_normal(2*x + 3*x, [x])    →  5*x
/// ```
pub fn poly_normal(
    expr: ExprId,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<ExprId, ConversionError> {
    let mp = MultiPoly::from_symbolic(expr, vars, pool)?;
    Ok(mp.to_expr(pool))
}
