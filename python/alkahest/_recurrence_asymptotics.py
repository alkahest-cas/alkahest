"""Growth of a P-recursive sequence, from the recurrence a loop just certified.

``zeilberger`` and ``guess_holonomic`` both hand back a recurrence, and the
next question is always the same: how fast does the sequence grow? That
question has an answer that follows from the recurrence — Poincaré–Perron —
and until now the asymptotics side of Alkahest (``asymptotic_expand``,
``euler_maclaurin``, ``coefficient_asymptotics``) and the holonomic side did
not compose at all. This module is the join.

Why this is Python and not Rust
-------------------------------

The mathematics is in the kernel: ``alkahest_cas::holonomic::asymptotics``
computes the characteristic polynomial, its roots and their exact
multiplicities, the polynomial exponent, and the fitted connection constant.
What is here is the third row of ``CONTRIBUTING.md`` § *Rust vs Python* —
docstring-driven overload dispatch. A caller has a
:class:`~alkahest.ZeilbergerCertificate`, or a
:class:`~alkahest.GuessedRecurrence`, or a plain list of coefficient
polynomials, and should not have to know which shape the kernel wants.

What is proved and what is fitted
---------------------------------

The returned object keeps them apart, because conflating them is the failure
mode this codebase's issue log is full of:

* ``growth_rate``, ``polynomial_exponent``, ``roots()``, ``verdict`` are
  **derived** — functions of the coefficient polynomials and nothing else.
* ``connection_constant`` is **fitted** — it depends on the initial conditions,
  is extrapolated from the exact terms, and is reported with the drift between
  two independent extrapolations so a caller can see what it is worth.

``report()`` carries the hypotheses, each marked ``checked`` or ``assumed``.
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Any

from .alkahest import RecurrenceAsymptotics
from .alkahest import asymptotics_from_recurrence as _native

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .alkahest import Expr

__all__ = ["RecurrenceAsymptotics", "asymptotics_from_recurrence"]


def _coefficients(rec: Any) -> tuple[Any, int | None]:
    """The coefficient polynomials of *rec*, and the index its terms start at.

    Accepts the two objects that produce recurrences in this library plus the
    raw form. Duck-typed rather than ``isinstance``-checked so that a wrapper
    around either still works — the attributes named here are documented API on
    both classes.
    """
    # GuessedRecurrence: integer coefficient tuples, lowest degree first, plus
    # the index its first term belongs to. Handed straight through as integers
    # so arbitrary-size coefficients stay exact.
    coeffs = getattr(rec, "coeffs", None)
    if coeffs is None:
        # A plain sequence of coefficient polynomials.
        return list(rec), None
    start = getattr(rec, "start", None)
    return list(coeffs), start


def _exact(value: Any, where: str) -> int | tuple[int, int]:
    """One sequence term as an exact integer or ``(numerator, denominator)``.

    A ``float`` is refused rather than converted, the same way
    :func:`alkahest.guess_holonomic` refuses one: ``0.1`` is not one tenth, and
    everything downstream of this point is exact arithmetic that would fit a
    perfectly convergent growth law to a sequence nobody asked about.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, Rational):
        frac = Fraction(value)
        return (frac.numerator, frac.denominator)
    raise TypeError(
        f"{where} must be an exact rational (int or fractions.Fraction), got "
        f"{type(value).__name__}; a float cannot be one, and a growth law "
        "fitted to rounded terms describes a different sequence"
    )


def asymptotics_from_recurrence(
    rec: Any,
    n: Expr,
    *,
    terms: Sequence[Any] | None = None,
    start: int | None = None,
) -> RecurrenceAsymptotics:
    """Asymptotic growth of the sequence *rec* is a recurrence for.

    *rec* is a :class:`~alkahest.ZeilbergerCertificate`, a
    :class:`~alkahest.GuessedRecurrence`, or a sequence of coefficient
    polynomials ``[p_0, …, p_J]`` (each an :class:`~alkahest.Expr` in *n*, or a
    sequence of ascending integer coefficients) for

    ``Σ_{i=0}^{J} p_i(n) · u(n+i) = 0``.

    *n* is the index variable; it also says which
    :class:`~alkahest.ExprPool` the result is built in.

    :param terms: exact leading terms of the sequence, ``terms[0] = u(start)``.
        ``int`` or :class:`fractions.Fraction`; a ``float`` is refused. Without
        them the growth rate and the polynomial exponent are still returned —
        they follow from the recurrence — but there is no connection constant
        and no way to check that the sequence follows the dominant root.
    :param start: index of ``terms[0]``. Defaults to
        :attr:`alkahest.GuessedRecurrence.start` when *rec* is one, else ``0``.

    :returns: a :class:`~alkahest.experimental.RecurrenceAsymptotics`, whose
        ``growth_rate`` and ``polynomial_exponent`` are **derived** and whose
        ``connection_constant`` is **fitted**.

    :raises alkahest.AsymptoticError: for malformed input only — fewer than two
        coefficients, a coefficient that is not a polynomial in *n* over ``ℚ``,
        or a characteristic polynomial all of whose roots are zero. A recurrence
        whose hypotheses fail is *reported* through ``verdict``, not refused.
    :raises TypeError: when a term is not an exact rational.

    Central binomial coefficients, ``C(2n,n) ~ 4ⁿ/√(πn)``:

    >>> import alkahest as ak
    >>> from alkahest.experimental import asymptotics_from_recurrence
    >>> pool = ak.ExprPool()
    >>> n = pool.symbol("n")
    >>> # (n+1)·u(n+1) − (4n+2)·u(n) = 0
    >>> r = asymptotics_from_recurrence([(-2, -4), (1, 1)], n, terms=[1])
    >>> r.verdict
    'single_dominant_root'
    >>> r.growth_rate, r.polynomial_exponent
    (4.0, -0.5)
    >>> round(r.connection_constant, 6)          # 1/sqrt(pi), *fitted*
    0.56419
    >>> r.connection_constant_converged
    True

    The exponential rate and the exponent are exact when the root is rational,
    and they are derived rather than fitted:

    >>> str(r.growth_rate_exact), str(r.polynomial_exponent_exact)
    ('4', '-1/2')

    Equal-modulus roots are reported, not guessed at — ``u(n+2) = 4·u(n)`` has
    characteristic roots ``±2`` and its solutions oscillate:

    >>> osc = asymptotics_from_recurrence([(-4,), (0,), (1,)], n, terms=[1, 2])
    >>> osc.verdict, osc.growth_rate
    ('equal_modulus_roots', None)
    """
    coeffs, rec_start = _coefficients(rec)
    if start is None:
        start = rec_start if rec_start is not None else 0
    exact = [_exact(t, "every term of the sequence") for t in (terms or ())]
    return _native(coeffs, n, terms=exact, start=start)
