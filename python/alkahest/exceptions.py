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
    E-HOMOTOPY-002 … E-HOMOTOPY-004 HomotopyError (numerical continuation — V2-14)
    E-SOLVE-010 … E-SOLVE-011  SolverError  (GPU Gröbner)
    E-JIT-001   … E-JIT-003    JitError
    E-LAT-001 … E-LAT-004      LatticeError
    E-PSLQ-001 … E-PSLQ-003    PslqError
    E-CAD-001                  CadError
    E-SUM-001 … E-SUM-003      SumError
    E-PROD-001 … E-PROD-004    ProductError (V2-22)
    E-REC-001 … E-REC-002      LinearRecurrenceError
    E-RSOLVE-001 … E-RSOLVE-005 RsolveError (V2-18 difference equations)
    E-DIOPH-001 … E-DIOPH-004 DiophantineError (V2-19)
    E-NT-001 … E-NT-005    NumberTheoryError (V3-1 integer number theory)
    E-SERIES-001 … E-SERIES-002 SeriesError
    E-LIMIT-001 … E-LIMIT-005 LimitError
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


class SeriesError(AlkahestError):
    """Symbolic series expansion failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-SERIES-001", remediation=remediation, span=span)


class LimitError(AlkahestError):
    """Symbolic limit computation failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-LIMIT-005", remediation=remediation, span=span)


class SumError(AlkahestError):
    """Symbolic summation failed (not hypergeometric or not Gosper-summable)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-SUM-001", remediation=remediation, span=span)


class ProductError(AlkahestError):
    """Symbolic discrete product failed (unsupported term or factorisation)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-PROD-001", remediation=remediation, span=span)


class LinearRecurrenceError(AlkahestError):
    """Linear recurrence solving failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-REC-001", remediation=remediation, span=span)


class RsolveError(AlkahestError):
    """Difference equation / rsolve failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-RSOLVE-001", remediation=remediation, span=span)


class DiophantineError(AlkahestError):
    """Integer Diophantine solving failed (linear / quadratic patterns)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-DIOPH-001", remediation=remediation, span=span)


class MatrixError(AlkahestError):
    """Matrix operation failed (dimension mismatch, singular, etc.)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-MAT-001", remediation=remediation, span=span)


class EigenError(AlkahestError):
    """Eigen-decomposition failed (unsplit characteristic polynomial, defective matrix, etc.)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-EIGEN-001", remediation=remediation, span=span)


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


class NumberTheoryError(AlkahestError):
    """Integer number-theory primitive failed (parity, modulus, or unsolvable congruence)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-NT-001", remediation=remediation, span=span)


class ParseError(AlkahestError):
    """Parse error with source span (reserved; parser not yet integrated)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-PARSE-001", remediation=remediation, span=span)
