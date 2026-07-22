// Rational-function cancel/together normalization
pub mod cancel;
pub mod error;
// V2-7 — Polynomial factorization
pub mod factor;
pub mod gauss;
// V2-3 — Sparse interpolation
pub mod interp;
pub mod multipoly;
// Partial-fraction decomposition (apart) over ℚ
pub mod partial_fractions;
// M3 prereq — Newton–Puiseux fractional-power local expansions of plane curves
pub mod puiseux;
pub mod rational;
pub mod residue;
// V2-2 — Resultants and subresultant PRS
pub mod resultant;
// V2-4 — Real root isolation (VAS)
pub mod real_roots;
pub mod unipoly;

#[cfg(feature = "groebner")]
pub mod groebner;

#[cfg(test)]
mod proptests;

pub use cancel::{cancel, together, together_parts};
pub use error::{ConversionError, FactorError};
pub use factor::{
    factor_multivariate_z, factor_univariate_mod_p, factor_univariate_z, MultiPolyFactorization,
    UniPolyFactorModP, UniPolyFactorization,
};
// V2-3 — Sparse interpolation and sparse modular GCD
pub use gauss::GaussRat;
pub use interp::{
    gcd_sparse_modular, sparse_interpolate, sparse_interpolate_univariate, SparseGcdError,
    SparseInterpError,
};
pub use multipoly::MultiPoly;
pub use partial_fractions::{apart, ApartError};
pub use rational::RationalFunction;
pub use residue::{residue, ResidueError};
// V2-2 — Resultants and subresultant PRS
pub use resultant::{collect_free_vars, resultant, subresultant_prs, ResultantError};
// V2-4 — Real root isolation (VAS)
pub use real_roots::{real_roots, real_roots_symbolic, refine_root, RealRootError, RootInterval};
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
