"""Zero recognition: :func:`is_zero` and :func:`equals` (G4).

Both rest on the native ``_zero_test`` decision procedure, which reports how
it reached its verdict:

``proved_zero``
    The expression simplifies to ``0``, or its rational normal form over atoms
    (trig and hyperbolic functions rewritten through ``exp``, radicals and
    ``i`` reduced by their defining relations, ``log`` split where every
    factor is provably positive) has a zero numerator.  This is exact symbolic
    reasoning — never sampling.
``proved_nonzero``
    An exact non-zero normal form of independent variables, or a rigorous Arb
    ball that excludes zero at an exact rational point inside every symbol's
    domain and every assumption.
``probably_zero``
    Every rigorous ball contained zero with a radius below ``1e-30`` at
    several admissible points (or, for complex-valued input, double-precision
    samples vanished).  This is evidence, not proof: the verdict is ``None``.
``unknown``
    Nothing conclusive.

``True`` therefore always means *proved*; numeric evidence alone can only
return ``False`` (a point where the expression is provably non-zero) or
``None``.
"""

from __future__ import annotations

from typing import Any

from .alkahest import _zero_test

__all__ = ["ZeroTest", "equals", "is_zero"]


class ZeroTest:
    """Verdict of :func:`equals` (or of a zero test) with its provenance.

    Attributes
    ----------
    value : bool or None
        ``True`` (proved equal / zero), ``False`` (proved different / non-zero)
        or ``None`` (undetermined).
    status : str
        ``"proved_zero"``, ``"proved_nonzero"``, ``"probably_zero"`` or
        ``"unknown"``.
    method : str
        ``"simplify"``, ``"rational_normal_form"``, ``"arb_ball"``,
        ``"arb_sampling"``, ``"complex_f64_sampling"`` or ``"none"``.
    evidence : list[str]
        Human-readable sample points and enclosures, when numeric.

    ``bool(test)`` is the proved verdict; it raises ``ValueError`` when the
    verdict is ``None`` so that an undecided comparison cannot silently read as
    "different" in an ``if``.
    """

    __slots__ = ("evidence", "method", "status", "value")

    def __init__(self, value: bool | None, status: str, method: str, evidence: list[str]):
        self.value = value
        self.status = status
        self.method = method
        self.evidence = list(evidence)

    @property
    def proved(self) -> bool:
        """Whether :attr:`value` is a proof (``True`` or ``False``)."""
        return self.value is not None

    def __bool__(self) -> bool:
        if self.value is None:
            raise ValueError(
                f"equality undetermined (status={self.status!r}); "
                "inspect .value / .status instead of using the result as a bool"
            )
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ZeroTest):
            return (self.value, self.status, self.method) == (
                other.value,
                other.status,
                other.method,
            )
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"ZeroTest(value={self.value!r}, status={self.status!r}, method={self.method!r})"


def _resolve_assumptions(assumptions: Any) -> Any:
    if assumptions is None:
        from ._context import active_assumptions

        assumptions = active_assumptions()
    return assumptions


def _test(expr: Any, assumptions: Any) -> ZeroTest:
    from . import _coerce_expr

    value, status, method, evidence = _zero_test(
        _coerce_expr(expr), _resolve_assumptions(assumptions)
    )
    return ZeroTest(value, status, method, evidence)


def is_zero(expr: Any, *, assumptions: Any = None) -> bool | None:
    """Decide whether *expr* is identically zero on its symbols' domains.

    Parameters
    ----------
    expr : Expr or DerivedResult
    assumptions : Assumptions, optional
        Extra facts (``x > 0``, ``x != 0``, …).  Defaults to the active
        ``alkahest.context(assumptions=...)``, if any.  Symbol domains
        (``Domain.Positive``, ``Domain.Integer``, …) always apply.

    Returns
    -------
    bool or None
        ``True`` only when proved by exact symbolic reasoning; ``False`` when
        proved non-zero (exactly, or by a rigorous Arb ball at an admissible
        point); ``None`` when undetermined.  Use :func:`equals` to see the
        provenance (``status``, ``method``, ``evidence``).

    Example
    -------
    >>> p = ExprPool(); x = p.symbol("x")
    >>> is_zero(sin(x)**2 + cos(x)**2 - 1)
    True
    >>> is_zero(sin(x) - x)
    False
    """
    return _test(expr, assumptions).value


def equals(a: Any, b: Any, *, assumptions: Any = None) -> ZeroTest:
    """Decide whether ``a == b`` identically, with provenance.

    Runs the :func:`is_zero` procedure on ``a - b``.

    Parameters
    ----------
    a, b : Expr, DerivedResult or number
    assumptions : Assumptions, optional
        As for :func:`is_zero`.

    Returns
    -------
    ZeroTest
        ``.value`` is ``True`` / ``False`` / ``None``; ``.status`` and
        ``.method`` say how it was reached.  ``status == "probably_zero"``
        means numerically indistinguishable but *not proved* (``value`` is
        ``None``).

    Example
    -------
    >>> p = ExprPool(); x = p.symbol("x")
    >>> equals(exp(2*x), exp(x)**2).value
    True
    """
    from . import _coerce_expr

    a = _coerce_expr(a)
    b = _coerce_expr(b)
    return _test(a - b, assumptions)
