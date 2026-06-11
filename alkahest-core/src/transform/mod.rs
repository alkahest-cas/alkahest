//! Symbolic integral transforms.
//!
//! - The [`laplace`] submodule provides the **Laplace transform** `L{f(t)}(s)`
//!   and its **inverse** `L⁻¹{F(s)}(t)`.
//! - The [`fourier`] submodule provides the **Fourier transform** `F{f(x)}(ξ)`
//!   and its **inverse** `F⁻¹{g(ξ)}(x)` (unitary, ordinary-frequency convention).
//!
//! These are *formal* transforms: no region-of-convergence is tracked and no
//! convergence side-conditions are emitted (matching SymPy's `noconds=True`
//! default).  See each submodule for its rule coverage, fallbacks, and declines.

pub mod fourier;
pub mod laplace;

pub use fourier::{fourier_transform, inverse_fourier_transform, FourierError};
pub use laplace::{inverse_laplace_transform, laplace_transform, LaplaceError};
