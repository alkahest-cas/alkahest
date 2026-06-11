//! Symbolic integral transforms.
//!
//! Currently the [`laplace`] submodule provides the **Laplace transform**
//! `L{f(t)}(s)` and its **inverse** `L⁻¹{F(s)}(t)`.
//!
//! These are *formal* transforms: no region-of-convergence is tracked and no
//! convergence side-conditions are emitted (matching SymPy's `noconds=True`
//! default).  See [`laplace`] for the rule coverage, fallbacks, and declines.

pub mod laplace;

pub use laplace::{inverse_laplace_transform, laplace_transform, LaplaceError};
