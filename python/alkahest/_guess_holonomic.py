"""Fit a P-recursive (holonomic) recurrence to the first terms of a sequence.

The research loop this exists for is *guess then prove*: fit a recurrence to
the terms you can compute, then certify it with :func:`alkahest.zeilberger`.
Alkahest shipped the proving half only, so every loop hand-rolled the guessing
half out of :class:`alkahest.Matrix` and exact rationals — thirty lines that are
easy to write and easy to write *wrong*, because the interesting part is not
the linear algebra, it is knowing when the fit is worth anything.

Why this is Python and not Rust
-------------------------------

Per ``CONTRIBUTING.md`` § *Rust vs Python*, the kernel gets mathematical
operations, hot paths, and anything whose correctness depends on exhaustive
``match``. None of that applies here. The one mathematical step — an exact
nullspace over ``Q`` — is :meth:`alkahest.Matrix.nullspace`, which is already
in the kernel and already exact; everything this module adds on top is
composition of existing kernel calls, caller-side validation, and the evidence
bookkeeping that decides whether a fit gets endorsed. That is the Python column
of the table, point by point. The cost of being wrong here is a false lemma
rather than a slow one, so the code that decides *whether to believe a fit*
should be the code that is easiest to read and audit.

The guard, and what it refuses
------------------------------

An unguarded fitter is worse than nothing. A recurrence of order ``J`` with
polynomial coefficients of degree ``D`` has ``(J+1)(D+1)`` unknown
coefficients; a homogeneous linear system in ``U`` unknowns has a nontrivial
solution as soon as it has fewer than ``U`` independent equations, whatever the
data. So *some* recurrence always fits, and a fit that consumed all its data is
not evidence of anything — it is interpolation with extra steps.

Two layers, therefore:

1. **Over-determined by construction.** An ``(order, degree)`` candidate is
   probed only when the terms supply at least ``U + min_surplus`` equations,
   ``min_surplus`` defaulting to ``U`` itself. Candidates that cannot clear that
   are never fitted, so a returned recurrence was never in a position to
   interpolate. Nothing is quietly skipped: if the bounds could not be swept in
   full, the call *refuses* (``E-HOLO-005``) instead of returning ``None``,
   because "no relation fits" and "you did not give me enough terms" are
   different answers and a search loop that conflates them closes a branch it
   never explored.
2. **Reported evidence.** The result carries
   :attr:`GuessedRecurrence.surplus_terms` — how many equations the fit did not
   need and satisfies anyway — and :attr:`GuessedRecurrence.dimension`, the
   dimension of the solution space. Confirmation means surplus above the
   threshold *and* dimension exactly one; a fit the caller can choose from
   several is not a fit.

This is :func:`alkahest.relation_confidence`'s discipline for sequences:
credibility is judged against what the data can actually support, and a fit
that fails is refused rather than returned with a caveat nobody reads. Pass
``check_evidence=False`` to get the raw fit with :attr:`~GuessedRecurrence.
confirmed` set honestly, the way ``check_precision=False`` works on
:func:`alkahest.guess_relation`.
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Any

from .alkahest import ExprPool, Matrix
from .alkahest import HolonomicError as _HolonomicError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .alkahest import Expr

__all__ = ["GuessedRecurrence", "guess_holonomic"]

# The *native* PyO3 class, not the pure-Python one in `exceptions.py`:
# `alkahest/__init__.py` overlays the native classes over the module namespace,
# so `ak.HolonomicError` is this one and a refusal raised from here has to be
# catchable by it. Subclassing the Python shim instead compiles, passes a
# hand-check, and then slips through every `except ak.HolonomicError` a caller
# writes — the exact silent failure this module is written to avoid.


class HolonomicEvidenceError(_HolonomicError):
    """``E-HOLO-005`` — the terms cannot support the fit that was asked for.

    A subclass of :class:`alkahest.HolonomicError`, so ``except
    ak.HolonomicError`` catches it alongside every other holonomic refusal
    while it still carries its own stable code.

    Raised in two situations, both of which are *undecided*, not *negative*:
    the ``(order, degree)`` grid could not be swept in full because the terms
    supplied too few equations, or a fit was found that the surplus/uniqueness
    guard will not endorse.
    """

    def __init__(self, message: str, remediation: str):
        super().__init__(message)
        self.code = "E-HOLO-005"
        self.remediation = remediation


def _exact(value: Any, where: str) -> Fraction:
    """Coerce one sequence term to an exact rational, refusing inexact input.

    ``float`` is refused rather than converted. ``0.1`` is not one tenth, and a
    recurrence fitted to binary approximations of the terms you meant is a
    recurrence for a different sequence — silently, since the arithmetic
    downstream of this point is exact and will happily certify the wrong answer.
    """
    if isinstance(value, Rational):
        # int, Fraction, and anything else registered as numbers.Rational.
        return Fraction(value)
    raise TypeError(
        f"{where} must be an exact rational (int or fractions.Fraction), got "
        f"{type(value).__name__}; a float cannot be one, and fitting a "
        "recurrence to rounded terms fits a different sequence — convert with "
        "Fraction(value).limit_denominator(...) if the rounding is intended"
    )


def _fraction_from_expr(entry: Expr) -> Fraction:
    """Read an exact rational back out of a nullspace entry.

    Refuses anything that is not a rational literal. The nullspace of an
    all-rational matrix is spanned by rational vectors, so a non-literal entry
    means an assumption of this module no longer holds, and approximating it
    would be exactly the silent-wrong-answer this file exists to avoid.
    """
    node = entry.node()
    if entry.node_tag() == "integer":
        return Fraction(int(node[1]))
    if node and node[0] == "rational":
        return Fraction(int(node[1]), int(node[2]))
    # Unreachable by construction — the matrix is all-integer — so this is an
    # invariant check, not a user-facing path. It still refuses rather than
    # approximating, because an approximation here would be certified later.
    error = _HolonomicError(
        f"nullspace entry {entry} is not an exact rational literal; the "
        "recurrence fit cannot be read off exactly"
    )
    error.code = "E-HOLO-003"
    error.remediation = "report this as a bug, with the input sequence"
    raise error


def _primitive(vector: Sequence[Fraction]) -> tuple[int, ...]:
    """Scale a rational solution vector to content-free integers, sign fixed.

    A nullspace vector is only defined up to a scalar, so two runs on the same
    data must not disagree merely about normalisation. Clearing denominators
    and dividing by the gcd is the same normal form the kernel's
    ``clear_denominators`` puts Zeilberger's coefficients into, which is what
    makes a guessed recurrence comparable to a certified one by eye.
    """
    denominator = 1
    for value in vector:
        denominator = denominator * value.denominator // _gcd(denominator, value.denominator)
    integers = [int(value * denominator) for value in vector]
    content = 0
    for value in integers:
        content = _gcd(content, abs(value))
    if content > 1:
        integers = [value // content for value in integers]
    for value in reversed(integers):
        if value != 0:
            if value < 0:
                integers = [-v for v in integers]
            break
    return tuple(integers)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


class GuessedRecurrence:
    """A P-recursive recurrence fitted to a finite sequence, with its evidence.

    The relation is

    ``Σ_{i=0}^{order} p_i(n) · u(n+i) = 0``

    where ``u(start) = terms[0]`` and ``p_i`` is the polynomial whose
    coefficients are ``coeffs[i]``, lowest degree first. The coefficients are
    integers, content-free, with the leading one positive, so the same data
    always produces the same object.

    **A fitted recurrence is a conjecture.** It is exact on the terms it was
    shown — that is checked, not assumed — and says nothing about the terms it
    was not. :attr:`surplus_terms` is how much of the data agreed without being
    asked to; hand the result to :func:`alkahest.zeilberger` when the sequence
    has a hypergeometric summand, and to :meth:`holds_for` when more terms turn
    up.
    """

    __slots__ = (
        "_coeffs",
        "_degree",
        "_dimension",
        "_min_surplus",
        "_n_equations",
        "_n_terms",
        "_order",
        "_rank",
        "_start",
        "_untested",
    )

    def __init__(
        self,
        *,
        order: int,
        degree: int,
        start: int,
        coeffs: tuple[tuple[int, ...], ...],
        n_terms: int,
        n_equations: int,
        rank: int,
        dimension: int,
        min_surplus: int,
        untested: int,
    ):
        self._order = order
        self._degree = degree
        self._start = start
        self._coeffs = coeffs
        self._n_terms = n_terms
        self._n_equations = n_equations
        self._rank = rank
        self._dimension = dimension
        self._min_surplus = min_surplus
        self._untested = untested

    @property
    def order(self) -> int:
        """Recurrence order ``J``; ``len(coeffs) == order + 1``.

        Minimal within the searched bounds: the search ascends order-major, so
        no lower order fits at any degree up to ``max_degree`` *that the terms
        were able to test* — see :attr:`untested_candidates`, which is ``0``
        when there was no such qualification. It is not a proof of minimality
        for the sequence; certify with
        :func:`alkahest.zeilberger(..., minimal=True)` for that.
        """
        return self._order

    @property
    def degree(self) -> int:
        """Largest degree in ``n`` of any coefficient polynomial that was fitted.

        The bound the ansatz was solved at, not the degree the answer turned out
        to need: a trailing zero coefficient is possible and means the fit found
        a sparser relation inside the space it was given.
        """
        return self._degree

    @property
    def start(self) -> int:
        """Index ``n`` that ``terms[0]`` was taken to be, i.e. ``u(start)``."""
        return self._start

    @property
    def coeffs(self) -> tuple[tuple[int, ...], ...]:
        """``(p_0, …, p_J)``, each a tuple of integers lowest-degree-first."""
        return self._coeffs

    @property
    def n_terms(self) -> int:
        """How many terms were supplied."""
        return self._n_terms

    @property
    def n_equations(self) -> int:
        """How many equations those terms produced, ``n_terms - order``."""
        return self._n_equations

    @property
    def equations_used(self) -> int:
        """Independent equations the fit consumed (the coefficient-matrix rank)."""
        return self._rank

    @property
    def surplus_terms(self) -> int:
        """Equations the fit did **not** need and satisfies anyway.

        This is the number to judge the guess by: it is how much of the data
        confirmed a recurrence that was already pinned down without it. Zero
        means the fit interpolated its own input and is evidence of nothing.
        """
        return self._n_equations - self._rank

    @property
    def dimension(self) -> int:
        """Dimension of the solution space at this ``(order, degree)``.

        ``1`` for a genuine fit. Larger means the data admits several
        independent relations of this shape and does not single one out; the
        vector reported is then an arbitrary choice among them, which is why
        :attr:`confirmed` is ``False``.
        """
        return self._dimension

    @property
    def min_surplus(self) -> int:
        """Surplus equations demanded of a confirmed fit at this candidate."""
        return self._min_surplus

    @property
    def untested_candidates(self) -> int:
        """``(order, degree)`` candidates below this one the terms could not test.

        The search ascends order-major, so every candidate before this one was
        either fitted and rejected or **skipped for want of surplus equations**.
        ``0`` means :attr:`order` is the smallest order that fits anywhere in
        the bounds; a positive count means it is the smallest among those the
        terms were able to decide, and a shorter relation may be hiding in the
        ones they were not.

        The same discipline as
        :attr:`alkahest.ZeilbergerCertificate.order_is_minimal`, for the same
        reason: minimality is usually the interesting half of the answer, and a
        search that skipped candidates has not established it.
        """
        return self._untested

    @property
    def confirmed(self) -> bool:
        """Whether the data supports the fit: enough surplus, and unique.

        ``True`` requires ``surplus_terms >= min_surplus`` **and**
        ``dimension == 1``. It is never a claim that the recurrence holds for
        the whole sequence — only that these terms are entitled to suggest it.
        """
        return self.surplus_terms >= self._min_surplus and self._dimension == 1

    def evidence(self) -> dict:
        """The confirmation numbers as a dict, for logging next to the result.

        Sibling of :func:`alkahest.relation_confidence`'s return value: the
        judgement plus everything that went into it, so a research loop can
        record *why* a fit was believed rather than only that it was.
        """
        return {
            "n_terms": self._n_terms,
            "n_equations": self._n_equations,
            "equations_used": self._rank,
            "surplus_terms": self.surplus_terms,
            "min_surplus": self._min_surplus,
            "dimension": self._dimension,
            "untested_candidates": self._untested,
            "confirmed": self.confirmed,
        }

    def holds_for(self, terms: Sequence[Any]) -> bool:
        """Exactly re-check the recurrence against a (longer) list of terms.

        *terms* must begin at the same index the fit did, i.e. ``terms[0]`` is
        ``u(start)``. The intended use is fresh data: fit on what you had, then
        compute more terms and call this. Arithmetic is exact, so a ``True``
        here is a fact about those terms, not a tolerance — and a fit confirmed
        on terms it was never shown is the strongest thing short of a proof.
        """
        values = [_exact(t, "every term") for t in terms]
        for row in range(len(values) - self._order):
            index = self._start + row
            total = Fraction(0)
            for i, poly in enumerate(self._coeffs):
                total += _horner(poly, index) * values[row + i]
            if total != 0:
                return False
        return True

    def to_exprs(self, pool: ExprPool, var: Expr) -> list[Expr]:
        """``[p_0(var), …, p_J(var)]`` as expressions in *pool*.

        Hands the guess to the rest of the library — most usefully to compare
        against :attr:`alkahest.ZeilbergerCertificate.coeffs` once the same
        recurrence has been certified.
        """
        out = []
        for poly in self._coeffs:
            monomials = []
            for j, c in enumerate(poly):
                if c == 0:
                    continue
                coefficient = pool.integer(c)
                if j == 0:
                    monomials.append(coefficient)
                elif j == 1:
                    monomials.append(coefficient * var)
                else:
                    monomials.append(coefficient * var**j)
            if not monomials:
                out.append(pool.integer(0))
            elif len(monomials) == 1:
                out.append(monomials[0])
            else:
                out.append(pool.add(monomials))
        return out

    def __repr__(self) -> str:
        return (
            f"GuessedRecurrence(order={self._order}, degree={self._degree}, "
            f"coeffs={self._coeffs}, surplus_terms={self.surplus_terms}, "
            f"dimension={self._dimension}, confirmed={self.confirmed})"
        )


def _horner(poly: Sequence[int], x: int) -> Fraction:
    total = Fraction(0)
    for c in reversed(poly):
        total = total * x + c
    return total


def guess_holonomic(
    terms: Sequence[Any],
    max_order: int = 4,
    max_degree: int = 4,
    *,
    start: int = 0,
    min_surplus: int | None = None,
    check_evidence: bool = True,
) -> GuessedRecurrence | None:
    """Fit a P-recursive recurrence to *terms*, or refuse.

    Searches for the smallest ``order`` (then smallest ``degree``) such that

    ``Σ_{i=0}^{order} p_i(n) · u(n+i) = 0``

    holds on every equation the terms provide, with ``deg p_i <= degree`` and
    ``u(start) = terms[0]``. Everything is exact rational arithmetic:
    ``terms`` must be Python ``int`` or :class:`fractions.Fraction` (arbitrary
    size), and a ``float`` is refused rather than rounded.

    **A candidate is fitted only when it is over-determined**: a recurrence with
    ``U = (order+1)(degree+1)`` unknowns is tried only where the terms give at
    least ``U + min_surplus`` equations, ``min_surplus`` defaulting to ``U``.
    Below that a nullspace vector exists no matter what the data is, so a fit
    would be interpolation wearing a recurrence's clothes.

    Returns ``None`` only for a *genuine* negative — every ``(order, degree)``
    in bounds was fitted with adequate surplus and none of them worked. If the
    terms were too few to sweep the bounds, the call raises
    ``HolonomicError`` (``E-HOLO-005``) saying how many terms the remaining
    candidates need, because a loop that reads "not holonomic" off a grid it
    never swept has closed a branch it never explored.

    :param terms: the first terms of the sequence, exact ``int`` / ``Fraction``.
    :param max_order: largest recurrence order to try.
    :param max_degree: largest coefficient-polynomial degree to try.
    :param start: index ``n`` of ``terms[0]``; the coefficient polynomials are
        polynomials in that ``n``.
    :param min_surplus: surplus equations demanded of a fit. Defaults to the
        candidate's own unknown count, i.e. the data must be twice what the
        ansatz needs. Setting it to ``0`` disables the requirement without
        disabling the reporting.
    :param check_evidence: when ``False``, every candidate is fitted regardless
        of surplus and the first fit is returned with
        :attr:`GuessedRecurrence.confirmed` set honestly. This is the escape
        hatch, not the default — the same role ``check_precision=False`` plays
        on :func:`alkahest.guess_relation`.

    :raises HolonomicError: ``E-HOLO-005`` when the terms cannot support the
        search, or when a fit was found that the evidence does not justify.
    :raises TypeError: when a term is not an exact rational.
    :raises ValueError: when the bounds are not positive.

    >>> import alkahest as ak
    >>> motzkin = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188,
    ...            5798, 15511, 41835, 113634, 310572, 853467,
    ...            2356779, 6536382, 18199284, 50852019]
    >>> guess = ak.guess_holonomic(motzkin)
    >>> guess.order, guess.degree
    (2, 1)
    >>> guess.coeffs
    ((-3, -3), (-5, -2), (4, 1))
    >>> guess.confirmed, guess.surplus_terms
    (True, 14)

    which is ``(n+4)·M(n+2) = (2n+5)·M(n+1) + (3n+3)·M(n)``, Motzkin's
    recurrence. Seven terms suffice to *determine* it and confirm nothing, so
    they are refused rather than fitted:

    >>> try:
    ...     ak.guess_holonomic(motzkin[:7])
    ... except ak.HolonomicError as exc:
    ...     print(exc.code, str(exc).startswith("7 terms are not enough"))
    E-HOLO-005 True

    A sequence with no such relation comes back ``None`` — but only once every
    candidate in bounds has actually been tested:

    >>> ak.guess_holonomic([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
    ...                     43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    ...                     101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    ...                     151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    ...                     199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
    ...                     263, 269, 271, 277, 281]) is None
    True
    """
    if max_order < 1:
        raise ValueError("max_order must be at least 1")
    if max_degree < 0:
        raise ValueError("max_degree must be at least 0")
    if min_surplus is not None and min_surplus < 0:
        raise ValueError("min_surplus must not be negative")

    rationals = [_exact(t, "every term of the sequence") for t in terms]
    n_terms = len(rationals)
    # The relation is homogeneous in the terms, so scaling all of them by one
    # constant scales every matrix entry by it and leaves the nullspace exactly
    # where it was. Doing it once here means the linear algebra runs over `Z`
    # rather than `Q` — cheaper, and it keeps every entry an arbitrary-precision
    # Python `int` (see `_fit`).
    denominator = 1
    for value in rationals:
        denominator = denominator * value.denominator // _gcd(denominator, value.denominator)
    values = [int(value * denominator) for value in rationals]

    skipped = []
    for order in range(1, max_order + 1):
        n_equations = n_terms - order
        for degree in range(max_degree + 1):
            unknowns = (order + 1) * (degree + 1)
            threshold = unknowns if min_surplus is None else min_surplus
            if check_evidence and n_equations < unknowns + threshold:
                skipped.append((order, degree, unknowns + threshold + order))
                continue
            if n_equations < 1:
                skipped.append((order, degree, unknowns + threshold + order))
                continue
            fitted = _fit(values, order, degree, start, threshold, len(skipped))
            if fitted is None:
                continue
            if check_evidence and not fitted.confirmed:
                raise _unjustified(fitted)
            return fitted

    if skipped:
        raise _too_few_terms(n_terms, skipped)
    return None


def _fit(
    values: Sequence[int],
    order: int,
    degree: int,
    start: int,
    min_surplus: int,
    untested: int,
) -> GuessedRecurrence | None:
    """Solve one ``(order, degree)`` candidate exactly, or return ``None``.

    *values* are integers — :func:`guess_holonomic` clears the denominators
    once, up front — so every matrix entry goes in through ``pool.integer``,
    which takes a Python ``int`` of any size. ``pool.rational`` does not: it
    marshals through a C ``long`` and raises ``OverflowError`` on a term as
    ordinary as ``(2n)!``.

    The pool is local to the call: the entries are the sequence terms times
    powers of the index and grow fast, and there is no reason for a caller's
    pool to end up holding them.
    """
    n_equations = len(values) - order
    pool = ExprPool()
    rows = []
    for row in range(n_equations):
        index = start + row
        entry = []
        for i in range(order + 1):
            term = values[row + i]
            power = 1
            for _ in range(degree + 1):
                entry.append(pool.integer(term * power))
                power *= index
        rows.append(entry)

    matrix = Matrix.from_rows(rows)
    basis = matrix.nullspace()
    if not basis:
        return None

    flat = [_fraction_from_expr(basis[0].get(i, 0)) for i in range((order + 1) * (degree + 1))]
    integers = _primitive(flat)
    coeffs = tuple(
        tuple(integers[i * (degree + 1) : (i + 1) * (degree + 1)]) for i in range(order + 1)
    )
    if not any(coeffs[order]):
        # The leading polynomial vanished identically, so this is a relation of
        # lower order dressed up as one of order `order` — and the ascending
        # search already refused that order at this degree. Refuse rather than
        # report an order the relation does not have.
        return None

    return GuessedRecurrence(
        order=order,
        degree=degree,
        start=start,
        coeffs=coeffs,
        n_terms=len(values),
        n_equations=n_equations,
        # `nullspace` is exact, so rank follows from the rank-nullity theorem
        # and does not need a second elimination over the same matrix.
        rank=(order + 1) * (degree + 1) - len(basis),
        dimension=len(basis),
        min_surplus=min_surplus,
        untested=untested,
    )


def _unjustified(fitted: GuessedRecurrence) -> HolonomicEvidenceError:
    if fitted.dimension > 1:
        reason = (
            f"the terms admit {fitted.dimension} independent relations of order "
            f"{fitted.order} and degree {fitted.degree}, so they do not single one "
            "out and the vector reported would be an arbitrary choice among them"
        )
    else:
        reason = (
            f"it consumed {fitted.equations_used} of the {fitted.n_equations} "
            f"equations the terms provide and only {fitted.surplus_terms} were left "
            f"to confirm it, short of the {fitted.min_surplus} required"
        )
    return HolonomicEvidenceError(
        f"a recurrence of order {fitted.order} and degree {fitted.degree} fits the "
        f"{fitted.n_terms} terms supplied, but {reason}; a fit the data cannot "
        "confirm is interpolation, not evidence",
        remediation=(
            "supply more terms, lower max_order/max_degree, or pass "
            "check_evidence=False to accept the fit unconfirmed (the returned "
            "object's .confirmed and .surplus_terms say what it is worth)"
        ),
    )


def _too_few_terms(n_terms: int, skipped: Sequence[tuple[int, int, int]]) -> HolonomicEvidenceError:
    needed = min(entry[2] for entry in skipped)
    order, degree, _ = min(skipped, key=lambda entry: entry[2])
    return HolonomicEvidenceError(
        f"{n_terms} terms are not enough to test every recurrence in bounds: "
        f"{len(skipped)} of the (order, degree) candidates would have had no "
        "surplus equations left to confirm a fit, so they were not fitted at all "
        f"— the cheapest of them, order {order} degree {degree}, needs {needed} "
        "terms. No relation was found among the candidates that could be tested, "
        "but that is not evidence the sequence has none",
        remediation=(
            f"supply at least {needed} terms, lower max_order/max_degree so the "
            "untestable candidates fall outside the search, or pass "
            "check_evidence=False to fit them anyway and read .confirmed"
        ),
    )
