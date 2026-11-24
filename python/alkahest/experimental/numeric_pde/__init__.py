"""alkahest.experimental.numeric_pde — symbolic discretisation and scheme analysis.

PDE milestone M7a. Everything here is exact: finite-difference weights are
rationals (or exact expressions for symbolic grids), truncation errors are
exact symbolic terms, and every stability verdict is either *proved* by exact
real-algebraic computation or reported as ``"unknown"`` — never inferred
from sampled floating-point values.

* :func:`fd_weights` / :func:`stencil` / :func:`laplacian_stencil` /
  :func:`truncation_error` — Fornberg weights on arbitrary offsets (uniform,
  staggered, one-sided, symbolic non-uniform), tensor-product and compact
  (Mehrstellen) multi-dimensional stencils, leading truncation errors.
* :func:`cheb_diff_matrix` / :func:`fourier_diff_matrix` — spectral
  collocation matrices, exact or numeric.
* :class:`Scheme`, :func:`von_neumann`, :func:`modified_equation`,
  :func:`consistency` — amplification factors and proved stability
  conditions, modified equations (numerical dissipation and dispersion),
  consistency order.
* :meth:`Stencil.kernel` / :meth:`Stencil.emit_c` — compiled kernels via the
  JIT, or C99 source via :func:`alkahest.emit_c_expr`.

Errors are :class:`alkahest.exceptions.NumericPdeError` with codes
``E-PDE-350 … E-PDE-358``.
"""

from ...exceptions import NumericPdeError
from ._fd import (
    FDStencil,
    Stencil,
    StencilKernel,
    fd_weights,
    laplacian_stencil,
    stencil,
    truncation_error,
)
from ._scheme import (
    ConsistencyResult,
    ModifiedEquation,
    Scheme,
    VonNeumannResult,
    consistency,
    modified_equation,
    von_neumann,
)
from ._spectral import cheb_diff_matrix, cheb_nodes, fourier_diff_matrix, fourier_nodes

__all__ = [
    "ConsistencyResult",
    "FDStencil",
    "ModifiedEquation",
    "NumericPdeError",
    "Scheme",
    "Stencil",
    "StencilKernel",
    "VonNeumannResult",
    "cheb_diff_matrix",
    "cheb_nodes",
    "consistency",
    "fd_weights",
    "fourier_diff_matrix",
    "fourier_nodes",
    "laplacian_stencil",
    "modified_equation",
    "stencil",
    "truncation_error",
    "von_neumann",
]
