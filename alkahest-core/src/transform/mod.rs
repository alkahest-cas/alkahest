//! Symbolic integral and discrete transforms.
//!
//! - The [`laplace`] submodule provides the **Laplace transform** `L{f(t)}(s)`
//!   and its **inverse** `L⁻¹{F(s)}(t)`.
//! - The [`fourier`] submodule provides the **Fourier transform** `F{f(x)}(ξ)`
//!   and its **inverse** `F⁻¹{g(ξ)}(x)` (unitary, ordinary-frequency convention).
//! - The [`ztransform`] submodule provides the **(unilateral) Z-transform**
//!   `Z{a[n]}(z)` and its **inverse** `Z⁻¹{A(z)}(n)`.
//!
//! These are *formal* transforms: no region-of-convergence is tracked and no
//! convergence side-conditions are emitted (matching SymPy's `noconds=True`
//! default).  See each submodule for its rule coverage, fallbacks, and declines.

pub mod fourier;
pub mod laplace;
pub mod ztransform;

pub use fourier::{fourier_transform, inverse_fourier_transform, FourierError};
pub use laplace::{inverse_laplace_transform, laplace_transform, LaplaceError};
pub use ztransform::{
    inverse_z_transform, z_shift_advance, z_shift_delay, z_transform, ZTransformError,
};
