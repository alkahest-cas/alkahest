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
    E-PSLQ-001 … E-PSLQ-004    PslqError  (004 = input precision below requested)
    E-CAD-001                  CadError
    E-ROOT-001 … E-ROOT-002    RealRootError (V2-4 VAS real root isolation)
    E-RES-001 … E-RES-003      ResultantError (V2-2)
    E-INTERP-001 … E-INTERP-004 SparseInterpError (V2-3)
    E-INTERP-010 … E-INTERP-012 SparseGcdError (V2-3 sparse modular GCD)
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
    E-CERT-001                 CertificateUnavailableError  (Python-only; certificate ledger)
    E-BUDGET-001 … E-BUDGET-003 BudgetExceededError (P1 search plumbing item 4)
    E-VALIDATED-001 … E-VALIDATED-005  ValidatedError (P1 item 9 — validated numerics)
    E-BATCH-001                 (Python-only; alkahest._batch fallback for a
                                 batch_map/batch_map_iter item whose exception carried no
                                 .code of its own — see docs/mdbook/src/batch.md)
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


class AssumptionError(AlkahestError):
    """An explicit simplification assumption contradicted the current context."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-SIMPLIFY-001", remediation=remediation, span=span)


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


class ValidatedError(AlkahestError):
    """A rigorous bound could not be established, so none is returned.

    Every variant is a refusal, never a guess: an unsupported primitive
    (``E-VALIDATED-001``), a free symbol with no interval (``E-VALIDATED-002``),
    a singularity or branch cut inside the box (``E-VALIDATED-003``), an
    enclosure that overflowed to infinity (``E-VALIDATED-004``), or a malformed
    request (``E-VALIDATED-005``).

    Note that running out of *budget* is not an error — that returns a wide but
    still rigorous enclosure with ``budget_exhausted=True``, because a loose
    true bound is useful and a tight false one is not.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-VALIDATED-001", remediation=remediation, span=span)


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


class LinearAlgebraError(AlkahestError):
    """Symbolic linear algebra operation failed (nullspace, decompositions, normal forms)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-LINALG-001", remediation=remediation, span=span)


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
    """Integer-relation heuristic failed (input, coefficient bound, or lattice step).

    ``E-PSLQ-004`` is raised when the supplied constants carry less precision
    than the search requests — see :func:`alkahest.guess_relation`.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
        code: str = "E-PSLQ-001",
    ):
        super().__init__(message, code=code, remediation=remediation, span=span)


class CadError(AlkahestError):
    """Cylindrical algebraic decomposition / real QE failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-CAD-001", remediation=remediation, span=span)


class RealRootError(AlkahestError):
    """Real root isolation failed (conversion to univariate polynomial, etc.)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-ROOT-001", remediation=remediation, span=span)


class ResultantError(AlkahestError):
    """Polynomial resultant or subresultant PRS computation failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-RES-001", remediation=remediation, span=span)


class SparseInterpError(AlkahestError):
    """Sparse modular polynomial interpolation failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-INTERP-001", remediation=remediation, span=span)


class SparseGcdError(AlkahestError):
    """Sparse modular GCD computation failed."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-INTERP-010", remediation=remediation, span=span)


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


class CertificateUnavailableError(AlkahestError):
    """A Lean certificate was required but the emitter withheld one.

    Raised by :func:`alkahest.require_certificate` and by any derivation-producing
    call made under ``with alkahest.context(require_certificate=True)``. The
    computation itself succeeded — what is missing is the machine-checkable
    evidence, so this is a *policy* failure, not a mathematical one.

    ``.remediation`` names the blocking rewrite rules where they can be
    identified. :func:`alkahest.certificate_coverage` tabulates which shapes
    certify.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-CERT-001", remediation=remediation, span=span)


class BudgetExceededError(AlkahestError):
    """A :class:`~alkahest.Budget` was exceeded, or cancellation was requested.

    Raised by heavy engines (:func:`alkahest.integrate` today) at a
    cooperative checkpoint inside the Rust kernel — see
    ``alkahest_core::budget`` and :mod:`alkahest._budget`. A fine, expected
    answer for a fan-out search loop, not a crash: ``.code`` distinguishes
    the three causes:

    - ``E-BUDGET-001`` — the active budget's wall-clock limit elapsed.
    - ``E-BUDGET-002`` — the active budget's step counter exceeded ``max_steps``.
    - ``E-BUDGET-003`` — :func:`alkahest.request_cancel` was called and not
      yet cleared.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-BUDGET-001", remediation=remediation, span=span)


class ParseError(AlkahestError):
    """Parse error with source span (reserved; parser not yet integrated)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-PARSE-001", remediation=remediation, span=span)
