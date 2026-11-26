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
* :func:`method_of_lines` → :class:`MolSystem` (milestone M7b) — a symbolic
  ``PDE`` problem discretised in space (stencils of any order, upwind /
  WENO-5, Chebyshev / Fourier collocation; Dirichlet, Neumann, Robin and
  periodic boundaries, eliminated or as DAE constraints) and compiled to
  stencil kernels applied over index ranges, with an exact sparse Jacobian,
  native RK45 / BDF integrators, JAX / StableHLO export and a von Neumann
  stability hook.

Errors are :class:`alkahest.exceptions.NumericPdeError` with codes
``E-PDE-350 … E-PDE-358`` (analysis) and ``E-PDE-359 … E-PDE-372``
(method of lines).
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
from ._mol import MolResult, MolStabilityWarning, MolSystem, method_of_lines
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
    "MolResult",
    "MolStabilityWarning",
    "MolSystem",
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
    "method_of_lines",
    "modified_equation",
    "stencil",
    "truncation_error",
    "von_neumann",
]
