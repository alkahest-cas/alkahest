"""
Alkahest exception hierarchy.

All native exceptions are registered in the compiled extension module
(``alkahest.alkahest``) as proper PyO3 exception classes carrying ``.code``,
``.remediation``, and ``.span`` attributes.  The Python classes below are
thin wrappers used for ``isinstance`` checks and for raising exceptions
from pure-Python code.

Canonical code ranges — authoritative source is ``alkahest_core::errors::codes::REGISTRY``:

    E-POLY-001 … E-POLY-010    ConversionError (001–007), FactorError (008–010)
    E-DIFF-001 … E-DIFF-004    DiffError  (003-004 = forward-mode variants)
    E-INT-001  … E-INT-002     IntegrationError
    E-MAT-001  … E-MAT-003     MatrixError
    E-ODE-001  … E-ODE-003     OdeError
    E-DAE-001  … E-DAE-003     DaeError
    E-SOLVE-001 … E-SOLVE-003  SolverError  (polynomial system)
    E-SOLVE-010 … E-SOLVE-011  SolverError  (GPU Gröbner)
    E-JIT-001   … E-JIT-003    JitError
    E-LAT-001 … E-LAT-004      LatticeError
    E-PSLQ-001 … E-PSLQ-003    PslqError
    E-CAD-001                  CadError
    E-CUDA-001  … E-CUDA-006   CudaError
    E-IO-001    … E-IO-009     IoError  (formerly PoolPersistError / E-POOL-*)
    E-PARSE-*                  ParseError  (reserved; parser not yet integrated)
    E-DOMAIN-*                 DomainError  (reserved; Python-only pending Rust impl)
"""

from __future__ import annotations


class AlkahestError(Exception):
    """Base class for all alkahest errors.

    Attributes
    ----------
    code : str
        Stable diagnostic code, e.g. ``"E-POLY-001"``.
    remediation : str | None
        Human-readable fix suggestion.
    span : tuple[int, int] | None
        Source byte span ``(start, end)``, or ``None``.
    """

    def __init__(
        self,
        message: str,
        code: str = "E-UNKNOWN",
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.remediation = remediation
        self.span = span


class ConversionError(AlkahestError):
    """Expression could not be converted to the requested type (e.g. not polynomial)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-POLY-001", remediation=remediation, span=span)


class FactorError(AlkahestError):
    """Polynomial factorization failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-POLY-008", remediation=remediation, span=span)


class DomainError(AlkahestError):
    """A side condition was violated (e.g. division by a known-zero expression)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-DOMAIN-001", remediation=remediation, span=span)


class DiffError(AlkahestError):
    """Differentiation failed (e.g. unknown function)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-DIFF-001", remediation=remediation, span=span)


class PoolError(AlkahestError):
    """Pool was closed, mismatched, or otherwise invalid."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-POOL-001", remediation=remediation, span=span)


class IntegrationError(AlkahestError):
    """Symbolic integration failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-INT-001", remediation=remediation, span=span)


class MatrixError(AlkahestError):
    """Matrix operation failed (dimension mismatch, singular, etc.)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-MAT-001", remediation=remediation, span=span)


class LatticeError(AlkahestError):
    """LLL lattice reduction failed (structure, Lovász parameter, or iteration limit)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-LAT-001", remediation=remediation, span=span)


class PslqError(AlkahestError):
    """Integer-relation heuristic failed (input, coefficient bound, or lattice step)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-PSLQ-001", remediation=remediation, span=span)


class OdeError(AlkahestError):
    """ODE construction or lowering failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-ODE-001", remediation=remediation, span=span)


class DaeError(AlkahestError):
    """DAE structural analysis failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-DAE-001", remediation=remediation, span=span)


class JitError(AlkahestError):
    """JIT compilation failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-JIT-001", remediation=remediation, span=span)


class SolverError(AlkahestError):
    """Polynomial system solver failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-SOLVE-001", remediation=remediation, span=span)


class CudaError(AlkahestError):
    """NVPTX / CUDA compilation or launch failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-CUDA-001", remediation=remediation, span=span)


class IoError(AlkahestError):
    """Checkpoint / restore I/O error (formerly PoolPersistError, E-POOL-* codes)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-IO-001", remediation=remediation, span=span)


class ParseError(AlkahestError):
    """Parse error with source span (reserved; parser not yet integrated)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-PARSE-001", remediation=remediation, span=span)
