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
    E-SERIES-001 … E-SERIES-003 SeriesError  (003 = expansion ran past its work
                                 ceiling / budget before reaching the requested
                                 order; refused rather than returned short)
    E-LIMIT-001 … E-LIMIT-005 LimitError
    E-CUDA-001  … E-CUDA-006   CudaError
    E-IO-001    … E-IO-009     IoError  (formerly PoolPersistError / E-POOL-*)
    E-PARSE-*                  ParseError  (reserved; parser not yet integrated)
    E-DOMAIN-*                 DomainError  (reserved; Python-only pending Rust impl)
    E-CERT-001                 CertificateUnavailableError  (Python-only; certificate ledger)
    E-BUDGET-001 … E-BUDGET-003 BudgetExceededError (P1 search plumbing item 4)
    E-VALIDATED-001 … E-VALIDATED-005  ValidatedError (P1 item 9 — validated numerics)
    E-SOS-001 … E-SOS-005      SosError (P1 item 8 — positivity certificates)
    E-HOLO-001 … E-HOLO-008    HolonomicError (P1 item 7 — creative telescoping,
                                 plus M6 — modular / p-adic evaluation;
                                 005 = a guessed recurrence the terms cannot
                                 confirm, raised from Python, so it is absent
                                 from the Rust REGISTRY as E-PSLQ-004 is)
    E-BATCH-001                 (Python-only; alkahest._batch fallback for a
                                 batch_map/batch_map_iter item whose exception carried no
                                 .code of its own — see docs/mdbook/src/batch.md)
    E-ANSATZ-001 … E-ANSATZ-004 AnsatzError (Python-only; P2 item 1 — conjecture generation)
    E-XCHECK-001 … E-XCHECK-004 CrossCheckError (Python-only; P2 item 2 — differential testing)
    E-SMT-001 … E-SMT-004       SmtError (P2 item 3 — SMT/SAT bridge)
    E-DEPTH-001                 DepthLimitError (expression nesting ceiling — see
                                 alkahest_core::kernel::depth; refuses rather than
                                 letting a recursive walk overflow the native stack)
"""

from __future__ import annotations

try:  # The compiled extension imports no Python modules, so this cannot cycle.
    from .alkahest import AlkahestError as _NativeAlkahestError
except ImportError:  # pragma: no cover - pure-Python fallback (no extension)
    _NativeAlkahestError = Exception


class AlkahestError(_NativeAlkahestError):
    """Base class for all alkahest errors.

    Inherits the **native** base class registered by the extension. That is
    load-bearing rather than cosmetic: the Rust engines raise the native
    classes, the pure-Python subsystems (``ansatz``, ``crosscheck``, ``smt``,
    the batch helpers) raise the wrappers below, and before this the two
    hierarchies were disjoint. ``except alkahest.AlkahestError`` — the
    documented way to catch anything this library raises — therefore caught
    the Rust half and silently missed the Python half.

    Subclasses keep their keyword constructors: only the message is forwarded
    to the native base, whose ``__init__`` is ``BaseException``'s and takes
    positional arguments only.

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


class DepthLimitError(AlkahestError):
    """An expression was too deeply nested to walk by recursion.

    Alkahest processes expressions by structural recursion, and a native stack
    overflow is a ``SIGSEGV`` rather than an exception — it would kill the
    interpreter outright, with no traceback for a caller to log.  Past a
    measured ceiling the operation therefore declines instead, which is
    something ``except Exception`` can actually catch.

    Depth is *nesting*, not size.  ``pool.add([t1, ..., t100000])`` has depth 2
    and is fine; ``t1 + t2 + ... + t100000`` written with repeated ``+`` builds
    100 000 nested binary ``Add`` nodes and is not.  Building the wide form, or
    splitting the work into subexpressions, is the fix.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-DEPTH-001", remediation=remediation, span=span)


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


class SosError(AlkahestError):
    """A positivity question was not answered with a certificate.

    The three outcomes are deliberately distinct, because conflating them
    loses information a search loop needs:

    * ``E-SOS-003`` — the target is *definitely* negative somewhere; the
      message carries a witness point. The claim is false.
    * ``E-SOS-002`` — no certificate of the searched shape at this degree.
      This is **not** a proof that none exists; raise the degree, or fall back
      to the complete (and far more expensive) :func:`alkahest.decide`.
    * ``E-SOS-001`` / ``E-SOS-004`` — the input was not a polynomial in the
      given variables, or the call was malformed.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-SOS-001", remediation=remediation, span=span)


class HolonomicError(AlkahestError):
    """Creative telescoping refused: input outside the proper hypergeometric
    class (``E-HOLO-001``), bounded search exhausted (``E-HOLO-002``), a
    candidate certificate failed exact verification (``E-HOLO-003``), the call
    was malformed (``E-HOLO-004``), or a guessed recurrence is not supported by
    the terms supplied (``E-HOLO-005``).

    M6 added three more, all from modular / ``p``-adic evaluation
    (:class:`alkahest.ModularRecurrence`, :func:`alkahest.binomial_mod`): the
    modulus is not a prime power the backend supports (``E-HOLO-006``), a step
    of the recurrence does not determine its next term as a ``p``-adic integer
    (``E-HOLO-007``), or the working precision the singular steps demand is
    past a machine-word modulus (``E-HOLO-008``).

    A refusal here is informative, not a failure: it says the term is not one
    Zeilberger's algorithm decides at the requested bounds, so a loop can close
    that branch instead of re-attempting it.

    ``E-HOLO-005`` is the one exception to "close that branch". It comes from
    :func:`alkahest.guess_holonomic` and means *the data could not answer* —
    too few terms to test every candidate in bounds, or a fit with no surplus
    equations to confirm it. Recording it as "this sequence has no recurrence"
    closes a branch that was never explored. ``E-HOLO-008`` is the same shape:
    a resource ceiling, not a mathematical verdict.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-HOLO-001", remediation=remediation, span=span)


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


class HomotopyError(AlkahestError):
    """Numerical polynomial continuation failed (V2-14 homotopy solver).

    The native class is registered by the extension as ``PyHomotopyError``;
    this is the Python-side wrapper used for ``isinstance`` checks, and its
    absence is what ``scripts/check_error_codes.py`` was reporting.

    - ``E-HOMOTOPY-002`` — the start system has too few paths for the target.
    - ``E-HOMOTOPY-003`` — path tracking failed for a random gamma.
    - ``E-HOMOTOPY-004`` — the tracker exhausted its step budget.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-HOMOTOPY-002", remediation=remediation, span=span)


class ModularError(AlkahestError):
    """A modular / CRT reconstruction step failed (``E-MOD-001`` … ``E-MOD-004``).

    As with :class:`HomotopyError`, the native class exists as ``PyModularError``
    and this wrapper is what makes it catchable by name from Python.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-MOD-001", remediation=remediation, span=span)


class ParamGroebnerError(AlkahestError):
    """A Gröbner basis over ``Q(params)`` refused or was called wrongly (M9).

    Raised by :class:`~alkahest.experimental.ParametricGroebnerBasis` and by
    ``GroebnerBasis.compute(..., params=[...])``.  The native class exists as
    ``PyParamGroebnerError``; this wrapper is what makes it catchable by name.

    - ``E-PARAMGB-001`` — no generators were supplied.
    - ``E-PARAMGB-002`` — the generators disagree on the variable/parameter
      lists.
    - ``E-PARAMGB-003`` — a specialisation was given the wrong number of values.
    - ``E-PARAMGB-004`` — **a result, not a malfunction.** The requested
      parameter point is on the degeneracy locus: one of the conditions the
      basis assumed vanishes there, so the generic basis says nothing about it.
      Read ``ParametricGroebnerBasis.conditions()`` to see which hypersurfaces
      those are, or compute over ℚ directly at that point.
    """

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-PARAMGB-004", remediation=remediation, span=span)


class AnsatzError(AlkahestError):
    """An ansatz family could not be built, or could not be fitted.

    Raised by :mod:`alkahest.ansatz` (P2 item 1 — conjecture generation).
    ``.code`` distinguishes the causes, and note that only the first two are
    "you called it wrong"; the others are *results*:

    - ``E-ANSATZ-001`` — a coefficient symbol name collides with a symbol
      already interned in the pool, so the fit would silently solve for the
      wrong thing.
    - ``E-ANSATZ-002`` — the requested family exceeds ``max_terms``.
      ``C(n+d, d)`` grows fast; refusing beats materialising it.
    - ``E-ANSATZ-003`` — the constraints are inconsistent: **no member of this
      family satisfies them**. For a search loop this is a positive result — a
      closed branch — not a malfunction.
    - ``E-ANSATZ-004`` — the residual is nonlinear in the unknowns and the
      escalation path (:func:`alkahest.solve`) needs a ``groebner`` build,
      which this one is not.
    """

    def __init__(
        self,
        message: str,
        code: str = "E-ANSATZ-001",
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code=code, remediation=remediation, span=span)


class CrossCheckError(AlkahestError):
    """A cross-CAS differential check could not be *posed*, so it was refused.

    Raised by :mod:`alkahest.crosscheck` (P2 item 2). A refusal here always
    beats a guess: a divergence is only informative if both systems were asked
    the same question, and a best-effort translation manufactures false
    divergences, which are worse than no signal at all.

    - ``E-XCHECK-001`` — the expression contains a node (or an active
      assumption) with no faithful mapping into the oracle's language.
    - ``E-XCHECK-002`` — no oracle is installed. Never reported as agreement;
      see :func:`alkahest.crosscheck.oracles`.
    - ``E-XCHECK-003`` — the operation has no defined comparison rung. A caller
      error, raised before any oracle is consulted.
    - ``E-XCHECK-004`` — the oracle itself declined: it has no implementation
      for the operation, raised, or returned an unevaluated form. An
      environmental outcome rather than a caller error, and **not** a
      divergence — comparing against a refusal would fabricate one.
    """

    def __init__(
        self,
        message: str,
        code: str = "E-XCHECK-001",
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code=code, remediation=remediation, span=span)


class SmtError(AlkahestError):
    """An SMT-LIB export, solver invocation, or model lift failed.

    Raised by :mod:`alkahest.smt` (P2 item 3).

    - ``E-SMT-001`` — no solver binary was found. Reported as a refusal rather
      than a silent fallback to the weak interval :func:`alkahest.satisfiable`,
      which would answer ``Unknown`` and look like the solver had run.
    - ``E-SMT-002`` — the formula is outside the supported SMT-LIB fragment;
      check :func:`alkahest.smt.supported` first.
    - ``E-SMT-003`` — the model contains a value (an ``root-obj`` algebraic
      number) that cannot yet be lifted **exactly**. Refused rather than
      truncated to a float: a float witness recorded as an exact one is the
      silent-error shape this whole subsystem exists to avoid.
    - ``E-SMT-004`` — a returned model failed back-substitution. The bridge or
      the solver is broken; this must never be downgraded to a warning.
    """

    def __init__(
        self,
        message: str,
        code: str = "E-SMT-001",
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code=code, remediation=remediation, span=span)


class ParseError(AlkahestError):
    """Parse error with source span (reserved; parser not yet integrated)."""

    def __init__(
        self,
        message: str,
        remediation: str | None = None,
        span: tuple[int, int] | None = None,
    ):
        super().__init__(message, code="E-PARSE-001", remediation=remediation, span=span)
