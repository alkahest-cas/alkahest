"""Novelty filtering — normalise a claim, hash it, look it up before reporting it.

A search loop over this library rediscovers Vandermonde's identity within the
hour. The difference between "produced 400 certified recurrences" and "produced
three that nobody had" is not the mathematics; it is a filter that puts every
claim into a canonical form, hashes it, and asks whether it is already written
down somewhere before anything calls it a finding.

Three pieces, in order:

1. :class:`RecurrenceClaim` — a P-recursive recurrence in **normal form**.
   Rescaling, sign flips, index shifts, a different clearing of denominators
   and a stray polynomial factor are all presentation, not content, so they are
   quotiented out (see :attr:`RecurrenceClaim.normal_form`).
2. :attr:`RecurrenceClaim.claim_hash` — a stable content address of that normal
   form, so a loop can dedupe its own output with a ``set``.
3. :func:`check_novelty` — the claim against one or more sources
   (:class:`OeisCache` offline, :class:`OeisWeb` when explicitly opted into),
   returning a :class:`NoveltyVerdict`.

What a negative verdict is allowed to claim
-------------------------------------------

**Nothing about novelty.** ``status == "not_found"`` means *this claim was not
found in the sources that were actually searched* — one encyclopaedia of
integer sequences, through the formula lines of the entries that matched the
terms it was given, through a parser that understands a fraction of what those
lines can say. The literature is not OEIS. :attr:`NoveltyVerdict.found` is
therefore three-valued in the manner of :func:`alkahest.relation_confidence`'s
``credible`` and :attr:`alkahest.GuessedRecurrence.confirmed`:

* ``True``  — a source states this claim (:attr:`~NoveltyVerdict.hedged` says
  whether it states it as a theorem or as a conjecture).
* ``False`` — the sources searched do not state it. Not "novel".
* ``None``  — no source could answer. Never a pass.

There is deliberately no ``novel`` attribute anywhere in this module, and
``bool(verdict)`` raises rather than silently reading ``True``, because
``if check_novelty(...):`` is the exact sentence this file exists to prevent.
:meth:`NoveltyVerdict.report` carries the scope of the search — entries
examined, statements compared, statements that could not be used — so the size
of a negative is visible next to it.

Why this is Python and not Rust
-------------------------------

``CONTRIBUTING.md`` § *Rust vs Python*: this is HTTP, JSON, text parsing of a
third party's prose formula lines, and an experimental API that will change
shape as more sources are added — rows 2, 3 and 5 of the Python column. The
arithmetic it does (exact polynomial normalisation in ``ℚ[n]``) is
:mod:`fractions` over degree-≤10 polynomials, not a hot path.

The kernel's own content-primitive scaling (``clear_denominators`` in
``alkahest-core/src/holonomic/qfield.rs``) is internal to the Rust holonomic
module and not on the Python surface, and :func:`alkahest.poly_normal` refuses
rational coefficients outright (``E-POLY-002``) — which is precisely the input
a differently-scaled presentation arrives as. So the scaling here is local, and
small enough to read in one sitting.

Being a good citizen against oeis.org
-------------------------------------

:class:`OeisWeb` is opt-in, never constructed by default, serves from its cache
before it touches the network, sleeps between requests, and returns
``unavailable`` rather than raising when the network is not there. **No test in
this repository requires the network**; the offline path is
:class:`OeisCache`, whose fixtures are recorded once and committed. OEIS data
is © The OEIS Foundation Inc., licensed CC BY-NC-SA 4.0 — a cache written by
this module records that in the file.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from alkahest.research import claim_id as _claim_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

__all__ = [
    "NOVELTY_STATUSES",
    "STATUS_MEANINGS",
    "NoveltyMatch",
    "NoveltyVerdict",
    "OeisCache",
    "OeisEntry",
    "OeisWeb",
    "RecordedRecurrence",
    "RecurrenceClaim",
    "SourceAnswer",
    "check_novelty",
]

#: Every verdict a novelty check can reach.
NOVELTY_STATUSES = ("recorded", "recorded_conjecturally", "not_found", "unavailable")

#: Deliberately unflattering glosses, in the spirit of
#: :data:`alkahest.research.STATUS_BADGES`: a reader must not be able to mistake
#: "not in the one place I looked" for "new".
STATUS_MEANINGS = {
    "recorded": "a source states this claim; it is not new",
    "recorded_conjecturally": (
        "a source states this claim but marks it conjectural or empirical; "
        "proving it is a result, restating it is not"
    ),
    "not_found": (
        "not found in the sources searched — this is not evidence of novelty, "
        "only the absence of evidence from the places actually looked at"
    ),
    "unavailable": ("no source could answer; nothing was established either way"),
}

#: Trailing windows of the entry's own data a parsed recurrence must satisfy
#: before it is believed to be what the formula line meant.
_MIN_CONFIRMATIONS = 3

# ---------------------------------------------------------------------------
# Univariate polynomials over ℚ.
#
# A polynomial is a tuple of `Fraction` coefficients, lowest degree first, with
# trailing zeros trimmed; the zero polynomial is `()`. Everything is exact.
# ---------------------------------------------------------------------------


def _trim(coeffs: Iterable[Fraction]) -> tuple:
    out = list(coeffs)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


def _p_add(a: tuple, b: tuple) -> tuple:
    n = max(len(a), len(b))
    return _trim(
        (a[i] if i < len(a) else Fraction(0)) + (b[i] if i < len(b) else Fraction(0))
        for i in range(n)
    )


def _p_neg(a: tuple) -> tuple:
    return tuple(-c for c in a)


def _p_sub(a: tuple, b: tuple) -> tuple:
    return _p_add(a, _p_neg(b))


def _p_mul(a: tuple, b: tuple) -> tuple:
    if not a or not b:
        return ()
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return _trim(out)


def _p_pow(a: tuple, e: int) -> tuple:
    out = (Fraction(1),)
    for _ in range(e):
        out = _p_mul(out, a)
    return out


def _p_eval(a: tuple, x: Fraction) -> Fraction:
    total = Fraction(0)
    for c in reversed(a):
        total = total * x + c
    return total


def _p_shift(a: tuple, s: int) -> tuple:
    """``p(n + s)``, exactly."""
    if s == 0:
        return a
    linear = (Fraction(s), Fraction(1))
    out: tuple = ()
    for c in reversed(a):
        out = _p_add(_p_mul(out, linear), (c,))
    return _trim(out)


def _p_divmod(a: tuple, b: tuple) -> tuple:
    """Exact quotient and remainder of *a* by *b* over ``ℚ``."""
    if not b:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    rem = list(a)
    while len(rem) >= len(b) and _trim(rem):
        shift = len(rem) - len(b)
        factor = rem[-1] / b[-1]
        quotient[shift] = factor
        for i, c in enumerate(b):
            rem[shift + i] -= factor * c
        rem = list(_trim(rem))
    return _trim(quotient), _trim(rem)


def _p_gcd(a: tuple, b: tuple) -> tuple:
    """Monic gcd over ``ℚ``; ``()`` only when both are zero."""
    while b:
        a, b = b, _p_divmod(a, b)[1]
    if not a:
        return ()
    lead = a[-1]
    return tuple(c / lead for c in a)


def _p_text(a: tuple) -> str:
    """Canonical text for a polynomial: descending powers, explicit signs."""
    if not a:
        return "0"
    parts = []
    for power in range(len(a) - 1, -1, -1):
        c = a[power]
        if c == 0:
            continue
        monomial = "n" if power == 1 else f"n^{power}"
        if power == 0:
            parts.append(str(c))
        elif c == 1:
            parts.append(monomial)
        elif c == -1:
            parts.append(f"-{monomial}")
        else:
            parts.append(f"{c}*{monomial}")
    body = parts[0]
    for part in parts[1:]:
        body += f" - {part[1:]}" if part.startswith("-") else f" + {part}"
    return body


# ---------------------------------------------------------------------------
# Linear forms in the shifts of one unknown sequence, over ℚ(n).
# ---------------------------------------------------------------------------


class _Unsupported(Exception):
    """The text says something this module deliberately does not model."""


class _Form:
    """``(poly + Σ_j shifts[j]·u(n+j)) / den``, all polynomials over ``ℚ``."""

    __slots__ = ("den", "poly", "shifts")

    def __init__(self, poly: tuple, shifts: dict | None = None, den: tuple | None = None):
        self.poly = poly
        self.shifts = shifts or {}
        self.den = (Fraction(1),) if den is None else den

    @staticmethod
    def constant(value: Fraction) -> _Form:
        return _Form(_trim((value,)))

    @staticmethod
    def variable() -> _Form:
        return _Form((Fraction(0), Fraction(1)))

    @staticmethod
    def sequence_term(shift: int) -> _Form:
        return _Form((), {shift: (Fraction(1),)})

    @property
    def is_linear_free(self) -> bool:
        return not any(self.shifts.values())

    def __add__(self, other: _Form) -> _Form:
        den = _p_mul(self.den, other.den)
        poly = _p_add(_p_mul(self.poly, other.den), _p_mul(other.poly, self.den))
        shifts: dict = {}
        for j, p in self.shifts.items():
            shifts[j] = _p_mul(p, other.den)
        for j, p in other.shifts.items():
            shifts[j] = _p_add(shifts.get(j, ()), _p_mul(p, self.den))
        return _Form(poly, shifts, den)

    def __neg__(self) -> _Form:
        return _Form(_p_neg(self.poly), {j: _p_neg(p) for j, p in self.shifts.items()}, self.den)

    def __sub__(self, other: _Form) -> _Form:
        return self + (-other)

    def __mul__(self, other: _Form) -> _Form:
        if not self.is_linear_free and not other.is_linear_free:
            raise _Unsupported("product of two sequence terms — not a linear recurrence")
        if self.is_linear_free:
            self, other = other, self
        # `other` is now free of sequence terms.
        den = _p_mul(self.den, other.den)
        poly = _p_mul(self.poly, other.poly)
        shifts = {j: _p_mul(p, other.poly) for j, p in self.shifts.items()}
        return _Form(poly, shifts, den)

    def __truediv__(self, other: _Form) -> _Form:
        if not other.is_linear_free:
            raise _Unsupported("division by a sequence term — not a linear recurrence")
        if not other.poly:
            raise _Unsupported("division by zero")
        den = _p_mul(self.den, other.poly)
        poly = _p_mul(self.poly, other.den)
        shifts = {j: _p_mul(p, other.den) for j, p in self.shifts.items()}
        return _Form(poly, shifts, den)

    def power(self, exponent: int) -> _Form:
        if exponent == 1:
            return self
        if not self.is_linear_free:
            raise _Unsupported("a sequence term raised to a power — not linear")
        if exponent < 0:
            return _Form.constant(Fraction(1)) / self.power(-exponent)
        return _Form(_p_pow(self.poly, exponent), {}, _p_pow(self.den, exponent))

    def as_integer(self) -> int:
        """The exponent this form denotes, or refuse."""
        if self.shifts or len(self.den) != 1 or len(self.poly) > 1:
            raise _Unsupported("exponent is not an integer constant")
        value = self.poly[0] / self.den[0] if self.poly else Fraction(0)
        if value.denominator != 1:
            raise _Unsupported("exponent is not an integer constant")
        return int(value)


# ---------------------------------------------------------------------------
# Normalisation.
# ---------------------------------------------------------------------------


def _normalise(shifts: dict) -> tuple:
    """Put ``Σ_j c_j(n)·u(n+j) = 0`` into normal form.

    Returns ``(p_0, …, p_J)``, each a tuple of ``int`` coefficients lowest
    degree first, for the equivalent claim ``Σ_i p_i(n)·u(n+i) = 0``.
    """
    live = {j: p for j, p in shifts.items() if p}
    if len(live) < 2:
        raise ValueError(
            "a recurrence claim needs at least two sequence terms with nonzero "
            f"coefficients, got {len(live)}"
        )
    low, high = min(live), max(live)
    # (1) Move the window to start at 0: the claim at index n+low is the same
    #     claim, so substitute n → n - low.
    polys = [_p_shift(live.get(low + i, ()), -low) for i in range(high - low + 1)]
    # (2) Divide out a common polynomial factor: `(n+1)·L` and `L` are the same
    #     recurrence written twice, up to the finitely many n where the factor
    #     vanishes.
    common: tuple = ()
    for p in polys:
        common = _p_gcd(common, p)
    if len(common) > 1:
        polys = [_p_divmod(p, common)[0] for p in polys]
    # (3) Clear denominators, then divide by the integer content: this is where
    #     "×(−2)" and "the same thing over a common denominator" become equal.
    multiplier = 1
    for p in polys:
        for c in p:
            multiplier = multiplier * c.denominator // gcd(multiplier, c.denominator)
    integral = [[int(c * multiplier) for c in p] for p in polys]
    content = 0
    for p in integral:
        for c in p:
            content = gcd(content, abs(c))
    if content > 1:
        integral = [[c // content for c in p] for p in integral]
    # (4) Sign: the first nonzero coefficient, scanning shifts upwards and
    #     degrees downwards, is positive.
    for p in integral:
        lead = next((c for c in reversed(p) if c != 0), 0)
        if lead != 0:
            if lead < 0:
                integral = [[-c for c in q] for q in integral]
            break
    return tuple(tuple(p) for p in integral)


def _poly_from_expr(expr: Any, var: str) -> tuple:
    """Read a polynomial in *var* out of an :class:`alkahest.Expr`, exactly."""
    node = expr.node()
    head = node[0]
    if head == "integer":
        return _trim((Fraction(int(node[1])),))
    if head == "rational":
        return _trim((Fraction(int(node[1]), int(node[2])),))
    if head == "symbol":
        if node[1] != var:
            raise ValueError(
                f"coefficient mentions the symbol {node[1]!r}, but a recurrence "
                f"coefficient must be a polynomial in {var!r} alone"
            )
        return (Fraction(0), Fraction(1))
    if head == "add":
        out: tuple = ()
        for child in node[1]:
            out = _p_add(out, _poly_from_expr(child, var))
        return out
    if head == "mul":
        out = (Fraction(1),)
        for child in node[1]:
            out = _p_mul(out, _poly_from_expr(child, var))
        return out
    if head == "pow":
        exponent = _poly_from_expr(node[2], var)
        if len(exponent) > 1 or (exponent and exponent[0].denominator != 1) or not exponent:
            raise ValueError("a recurrence coefficient may only use non-negative integer powers")
        power = int(exponent[0])
        if power < 0:
            raise ValueError("a recurrence coefficient may only use non-negative integer powers")
        return _p_pow(_poly_from_expr(node[1], var), power)
    raise ValueError(f"{expr} is not a polynomial in {var!r}; a recurrence coefficient must be one")


def _coerce_poly(value: Any, var: str | None) -> tuple:
    """One coefficient polynomial, from an ``Expr`` or a sequence of rationals."""
    if hasattr(value, "node"):
        if var is None:
            raise TypeError(
                "coefficients given as Expr need the index variable: pass "
                "var=n (the symbol the coefficients are polynomials in)"
            )
        return _poly_from_expr(value, var)
    if isinstance(value, (int, Fraction)):
        return _trim((Fraction(value),))
    return _trim(Fraction(c) for c in value)


# ---------------------------------------------------------------------------
# The claim.
# ---------------------------------------------------------------------------


class RecurrenceClaim:
    """A P-recursive recurrence, in a normal form two presentations share.

    The claim is ``Σ_{i} p_i(n)·u(n+i) = 0`` — a *homogeneous* linear relation
    with polynomial coefficients, the thing :func:`alkahest.zeilberger` and
    :func:`alkahest.guess_holonomic` both produce.

    What the normal form quotients out, and therefore what
    :attr:`claim_hash` proves equal:

    1. **Scale.** Multiplying every coefficient by a nonzero rational — so
       ``×(−2)`` and "cleared by a different denominator" agree. Denominators
       are cleared and the integer content divided out; the sign is fixed by
       making the first nonzero coefficient positive, scanning shifts upwards
       and degrees downwards.
    2. **Index shift.** ``Σ p_i(n)·u(n+i) = 0`` and the same relation written
       about ``u(n−1)`` or ``u(n+1)`` are one statement re-indexed, so the
       window is moved to start at ``u(n)`` and the coefficients substituted
       accordingly.
    3. **A common polynomial factor.** ``(n+1)·L`` and ``L`` are the same
       recurrence, up to the finitely many ``n`` where the factor vanishes.
    4. **Zero coefficients at either end** of the window, which only pad the
       stated order.

    What it does **not** quotient out: a genuinely different relation, an
    operator of different order that happens to be a left multiple, or the
    range of ``n`` a source claims the relation on. The normal form is a
    statement about the relation, not about its domain of validity — two
    sources stating the same recurrence from different starting indices agree
    here, which is what a novelty filter wants and is *not* a proof that the
    two statements are interchangeable at small ``n``.

    >>> from alkahest.experimental.novelty import RecurrenceClaim
    >>> # (n+1)·u(n+1) − (4n+2)·u(n) = 0   — central binomial coefficients
    >>> a = RecurrenceClaim([(-2, -4), (1, 1)])
    >>> # the same relation, scaled by −2, stated about u(n+7) and u(n+8)
    >>> b = RecurrenceClaim([(-60, -8), (16, 2)], offset=7)
    >>> a.claim_hash == b.claim_hash
    True
    >>> a.order, a.degree
    (1, 1)
    """

    __slots__ = ("_coefficients", "_hash", "_normal_form")

    def __init__(self, coefficients: Sequence[Any], *, offset: int = 0, var: Any = None):
        """
        :param coefficients: ``[c_0, …, c_J]``, the coefficient of ``u(n+offset+i)``.
            Each is an :class:`alkahest.Expr` polynomial in *var*, or a sequence
            of exact rationals lowest degree first.
        :param offset: index of the first coefficient's shift; ``coefficients[0]``
            multiplies ``u(n + offset)``.
        :param var: the index symbol, required when the coefficients are
            :class:`alkahest.Expr`. An ``Expr`` or a name.
        :raises ValueError: when fewer than two coefficients are nonzero — a
            claim with one term is not a recurrence.
        """
        name = None
        if var is not None:
            name = var if isinstance(var, str) else var.node()[1]
        shifts = {offset + i: _coerce_poly(c, name) for i, c in enumerate(coefficients)}
        self._coefficients = _normalise(shifts)
        self._normal_form = "recurrence/1 " + " + ".join(
            f"({_p_text(tuple(Fraction(c) for c in p))})*u(n+{i})"
            for i, p in enumerate(self._coefficients)
        )
        self._hash = _claim_id(self._normal_form, method="recurrence")

    @classmethod
    def from_recurrence(cls, rec: Any, var: Any = None) -> RecurrenceClaim:
        """From a :class:`~alkahest.ZeilbergerCertificate`, a
        :class:`~alkahest.GuessedRecurrence`, or a raw coefficient list.

        Duck-typed on ``.coeffs`` exactly as
        :func:`alkahest.experimental.asymptotics_from_recurrence` is, so a
        wrapper around either still works. *var* is required when the
        coefficients are expressions.
        """
        coeffs = getattr(rec, "coeffs", None)
        return cls(list(rec) if coeffs is None else list(coeffs), var=var)

    @classmethod
    def from_text(cls, text: str) -> RecurrenceClaim | None:
        """Parse one prose formula line, e.g. an OEIS ``a(n) = …`` statement.

        Returns ``None`` — never a guess — when the line is not a homogeneous
        linear recurrence with polynomial coefficients in the single sequence
        ``a``: a sum, a generating function, a nonlinear relation, a reference
        to another sequence, an inhomogeneous relation, or a statement the
        parser simply does not cover. Callers that need to know how often that
        happened should count the ``None``s; :meth:`NoveltyVerdict.report`
        does.
        """
        try:
            relation = _parse_relation(text)
        except _Unsupported:
            return None
        if relation is None:
            return None
        try:
            return cls([relation[j] for j in sorted(relation)], offset=min(relation))
        except ValueError:
            return None

    @property
    def order(self) -> int:
        """``J`` — the span of the window in normal form."""
        return len(self._coefficients) - 1

    @property
    def degree(self) -> int:
        """Largest degree in ``n`` of any coefficient in normal form."""
        return max((len(p) - 1 for p in self._coefficients if p), default=0)

    @property
    def normal_form(self) -> str:
        """The canonical text the hash is taken of.

        Versioned (``recurrence/1``): a change to the normal form changes every
        hash, so the tag is part of what is hashed.
        """
        return self._normal_form

    @property
    def claim_hash(self) -> str:
        """Content address of :attr:`normal_form`, e.g. ``'clm_9f1c0b2a7d4e5f60'``.

        Computed by :func:`alkahest.research.claim_id`, so this library has one
        content-addressing scheme rather than two. Equal hashes mean equal
        normal forms; a loop can dedupe with ``seen.add(claim.claim_hash)``.
        """
        return self._hash

    def coefficients(self) -> tuple:
        """``(p_0, …, p_J)`` in normal form, each a tuple of ``int``, ascending."""
        return self._coefficients

    def holds_for(self, terms: Sequence[Any], *, start: int = 0) -> bool:
        """Exactly re-check the normal form against terms, ``terms[0] = u(start)``.

        Arithmetic is exact, so ``True`` is a fact about those terms — not a
        tolerance, and not a proof about the sequence. Requires every window
        the terms provide to hold; see :meth:`confirmations` for the lenient
        count a recurrence stated only for large ``n`` needs.

        **What *start* means, precisely — read this before passing anything
        other than the default:** *start* is the true index of ``terms[0]``,
        full stop — not of some other element, not of "the real data" if
        *terms* happens to carry a padding element you consider bogus. The
        window at array position ``row`` is checked at ``n = start + row``;
        every coefficient here can be a genuine polynomial in ``n`` (this is
        P-recursive, not just P-recursive-in-name), so a wrong *start* is not
        a small numeric error that shifts a few results — it evaluates every
        coefficient at the wrong point and generically makes the *entire*
        array fail, trailing windows included, even ones that never touch the
        element you meant to exclude. There is no way for this method to
        recover the right offset on its own: if you slice, drop, or prepend
        elements relative to some original indexing, you must adjust *start*
        by the same amount, e.g. ``terms = [junk, *real]`` needs
        ``start=-1`` (``junk`` sits where ``u(-1)`` would), not ``start=0``.
        """
        windows = len(terms) - self.order
        return windows > 0 and self.confirmations(terms, start=start) == windows

    def confirmations(self, terms: Sequence[Any], *, start: int = 0) -> int:
        """How many **trailing** consecutive windows of *terms* the claim holds on.

        Trailing, because a recurrence is routinely stated only for ``n`` past
        some initial segment, and an exception at ``n = 0`` says nothing about
        whether the relation is the one that was meant. "Lenient" describes
        *which* windows are required to hold (only the trailing run, not the
        front), never the meaning of *start*: see :meth:`holds_for` for what
        *start* must denote and why getting it wrong corrupts every window,
        not only the ones near the misindexed element — this method cannot
        detect or route around a caller's off-by-one, it can only be lenient
        about a relation that is *correctly* indexed but genuinely does not
        hold for small ``n``.
        """
        values = [Fraction(t) for t in terms]
        count = 0
        for row in range(len(values) - self.order - 1, -1, -1):
            index = Fraction(start + row)
            total = Fraction(0)
            for i, poly in enumerate(self._coefficients):
                total += _p_eval(tuple(Fraction(c) for c in poly), index) * values[row + i]
            if total != 0:
                break
            count += 1
        return count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecurrenceClaim):
            return NotImplemented
        return self._normal_form == other._normal_form

    def __hash__(self) -> int:
        return hash(self._normal_form)

    def __repr__(self) -> str:
        return (
            f"RecurrenceClaim(order={self.order}, degree={self.degree}, claim_hash={self._hash!r})"
        )


# ---------------------------------------------------------------------------
# Parsing prose formula lines.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*|\d+|\*\*|\S)")
_OPERATORS = frozenset("+-*/^")
_OPENERS = frozenset("([{")
_CLOSERS = frozenset(")]}")
#: A statement worth handing to the parser at all: it mentions a shifted term.
_LOOKS_LIKE_RECURRENCE = re.compile(r"a\(\s*n\s*[-+]\s*\d+\s*\)")
#: OEIS's own hedges. An entry that marks a formula this way is telling you the
#: recurrence was fitted and never proved — which is the whole reason a novelty
#: filter over OEIS is worth anything.
_HEDGE_RE = re.compile(
    r"\b(conjectur\w*|empirical\w*|apparently|seems? to|guessed|unproved|unproven)\b",
    re.IGNORECASE,
)


def _tokenise(text: str) -> list:
    return _TOKEN_RE.findall(text)


def _is_word(token: str) -> bool:
    return token[0].isalpha() or token[0] == "_"


class _Parser:
    """Recursive descent over ``+ - * / ^ ( )``, integers, ``n`` and ``a(n±k)``.

    Everything else — another sequence's ``A123456(n)``, ``Sum_{…}``, a symbol
    that is not the index — raises :class:`_Unsupported`. Refusing is the point:
    a parser that guesses at prose invents claims that were never made.
    """

    def __init__(self, tokens: Sequence[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def expression(self) -> _Form:
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.tokens[self.pos]
            self.pos += 1
            rhs = self.term()
            node = node + rhs if op == "+" else node - rhs
        return node

    def term(self) -> _Form:
        node = self.factor()
        while self.peek() in ("*", "/"):
            op = self.tokens[self.pos]
            self.pos += 1
            rhs = self.factor()
            node = node * rhs if op == "*" else node / rhs
        return node

    def factor(self) -> _Form:
        if self.peek() in ("+", "-"):
            op = self.tokens[self.pos]
            self.pos += 1
            node = self.factor()
            return -node if op == "-" else node
        node = self.atom()
        if self.peek() in ("^", "**"):
            self.pos += 1
            return node.power(self.factor().as_integer())
        return node

    def atom(self) -> _Form:
        token = self.peek()
        if token is None:
            raise _Unsupported("expression ended early")
        self.pos += 1
        if token == "(":
            node = self.expression()
            if self.peek() != ")":
                raise _Unsupported("unbalanced parenthesis")
            self.pos += 1
            return node
        if token.isdigit():
            return _Form.constant(Fraction(int(token)))
        if token == "n":
            return _Form.variable()
        if token == "a":
            if self.peek() != "(":
                raise _Unsupported("sequence name not applied to an index")
            self.pos += 1
            index = self.expression()
            if self.peek() != ")":
                raise _Unsupported("unbalanced parenthesis in a sequence index")
            self.pos += 1
            return _Form.sequence_term(_shift_of(index))
        raise _Unsupported(f"unsupported token {token!r}")


def _shift_of(index: _Form) -> int:
    """``k`` from an index that reads ``n + k``; refuse anything else.

    ``a(2*n)`` and ``a(0)`` are not shifts of the running index: the first is a
    different sequence, the second an initial condition. Both must be refused
    rather than approximated into a shift.
    """
    if index.shifts or len(index.den) != 1:
        raise _Unsupported("sequence index is not of the form n+k")
    poly = index.poly
    if len(poly) != 2 or poly[1] != 1 or poly[0].denominator != 1:
        raise _Unsupported("sequence index is not of the form n+k")
    return int(poly[0])


def _clean(text: str) -> str:
    """Strip the wrapping OEIS puts around a formula line."""
    text = text.split(" - _", 1)[0]
    text = text.replace("(Start)", " ").replace("(End)", " ")
    return text.strip().rstrip(".")


def _top_level_equals(tokens: Sequence[str]) -> int | None:
    depth = 0
    for i, token in enumerate(tokens):
        if token in _OPENERS:
            depth += 1
        elif token in _CLOSERS:
            depth -= 1
        elif token == "=" and depth == 0:
            return i
    return None


def _parse_all(tokens: Sequence[str]) -> _Form | None:
    parser = _Parser(tokens)
    try:
        form = parser.expression()
    except _Unsupported:
        return None
    return form if parser.pos == len(tokens) else None


def _boundary_ok(token: str | None) -> bool:
    """Whether prose may legitimately abut the formula at this token.

    A word (``with``, ``for``, ``unless``), punctuation, or the end of the line
    ends the formula. An arithmetic operator or a bracket does **not** — it
    means the expression continues and the parser stopped inside it, and
    truncating there would silently invent a different, shorter claim.
    ``a(n) = a(n-1) + A002026(n-1)`` must be refused, not read as
    ``a(n) = a(n-1)``.
    """
    if token is None:
        return True
    if token in _OPERATORS or token in _OPENERS or token in _CLOSERS:
        return False
    return _is_word(token) or token in {".", ",", ";", ":", "=", "!"}


def _parse_relation(text: str) -> dict | None:
    """``{shift: coefficient polynomial}`` for a prose linear recurrence, or ``None``."""
    tokens = _tokenise(_clean(text))
    split = _top_level_equals(tokens)
    if split is None:
        return None
    lhs = None
    for start in range(split):
        if not _boundary_ok(tokens[start - 1] if start else None):
            continue
        lhs = _parse_all(tokens[start:split])
        if lhs is not None:
            break
    if lhs is None:
        return None
    rhs = None
    for stop in range(len(tokens), split, -1):
        if not _boundary_ok(tokens[stop] if stop < len(tokens) else None):
            continue
        rhs = _parse_all(tokens[split + 1 : stop])
        if rhs is not None:
            break
    if rhs is None:
        return None
    form = lhs - rhs
    if form.poly:
        # Inhomogeneous: `a(n) = a(n-1) + 1` is a different kind of claim and
        # is not silently truncated into a homogeneous one.
        return None
    live = {j: p for j, p in form.shifts.items() if p}
    return live or None


# ---------------------------------------------------------------------------
# Sources.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedRecurrence:
    """A recurrence a source states, and how it states it."""

    claim: RecurrenceClaim
    statement: str
    hedged: bool
    confirmations: int


class OeisEntry:
    """One OEIS entry, reduced to what a recurrence claim can be checked against."""

    __slots__ = ("_scan", "id", "name", "offset", "statements", "terms")

    def __init__(
        self,
        id: str,
        name: str = "",
        terms: Sequence[int] = (),
        statements: Sequence[str] = (),
        offset: int = 0,
    ):
        self.id = id
        self.name = name
        self.terms = tuple(int(t) for t in terms)
        self.statements = tuple(statements)
        self.offset = offset
        self._scan: tuple | None = None

    @classmethod
    def from_oeis_json(cls, payload: dict) -> OeisEntry:
        """From one element of ``https://oeis.org/search?…&fmt=json``.

        Only the formula and comment lines that mention a shifted ``a(n±k)``
        are kept: the rest cannot state a recurrence, and a cache that keeps
        them is a cache nobody commits.
        """
        offset = 0
        raw_offset = str(payload.get("offset", "0")).split(",")[0].strip()
        if raw_offset.lstrip("-").isdigit():
            offset = int(raw_offset)
        lines = list(payload.get("formula") or ()) + list(payload.get("comment") or ())
        return cls(
            id=f"A{int(payload['number']):06d}",
            name=payload.get("name", ""),
            terms=[int(t) for t in str(payload.get("data", "")).split(",") if t.strip()],
            statements=[ln for ln in lines if _LOOKS_LIKE_RECURRENCE.search(ln)],
            offset=offset,
        )

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "offset": self.offset,
            "terms": list(self.terms),
            "statements": list(self.statements),
        }

    def _scanned(self) -> tuple:
        if self._scan is None:
            usable, unusable = [], []
            for statement in self.statements:
                claim = RecurrenceClaim.from_text(statement)
                if claim is None:
                    unusable.append(statement)
                    continue
                # The line is only believed once it reproduces the entry's own
                # terms. This is what stops a mis-read of somebody's prose from
                # entering the index as a claim they never made.
                confirmations = claim.confirmations(self.terms, start=self.offset)
                if confirmations < min(_MIN_CONFIRMATIONS, len(self.terms) - claim.order):
                    unusable.append(statement)
                    continue
                usable.append(
                    RecordedRecurrence(
                        claim=claim,
                        statement=statement.strip(),
                        hedged=bool(_HEDGE_RE.search(statement)),
                        confirmations=confirmations,
                    )
                )
            self._scan = (tuple(usable), tuple(unusable))
        return self._scan

    def recurrences(self) -> tuple:
        """Every recurrence this entry states that also reproduces its own terms."""
        return self._scanned()[0]

    def unusable_statements(self) -> tuple:
        """Lines that mention ``a(n±k)`` but could not be turned into a claim.

        Either the parser does not cover them or they failed the check against
        the entry's own data. They are counted into
        :meth:`NoveltyVerdict.report` so that the width of a ``not_found`` is
        visible: a claim can only be found in a statement that was understood.
        """
        return self._scanned()[1]

    def __repr__(self) -> str:
        return f"OeisEntry({self.id!r}, {self.name[:40]!r}, terms={len(self.terms)})"


@dataclass(frozen=True)
class SourceAnswer:
    """What a source was able to say.

    ``exhaustive`` is the honest half: ``True`` means these are *all* the
    entries the source has for the query, so a claim missing from them is
    genuinely missing from the source. ``False`` means the source found what it
    had locally but cannot promise there is nothing else, and a non-match must
    therefore be reported as ``unavailable`` rather than ``not_found``.
    """

    entries: tuple
    exhaustive: bool


def _query_key(terms: Sequence[int] | None, ids: Sequence[str] | None) -> str:
    if ids:
        return "id:" + ",".join(sorted(ids))
    return "seq:" + ",".join(str(int(t)) for t in terms or ())


class OeisCache:
    """A file-backed OEIS cache — the offline path, and the test fixture format.

    Holds two things: **entries**, keyed by A-number, and **queries** already
    put to OEIS, keyed by what was asked. The second is what makes an honest
    negative possible: a cache that only stores hits can never distinguish "I
    asked and OEIS had nothing" from "I never asked", and reporting the second
    as the first is exactly the overclaim this module is for.

    >>> from alkahest.experimental.novelty import OeisCache, OeisEntry
    >>> cache = OeisCache()
    >>> cache.add(OeisEntry("A000045", "Fibonacci", terms=[1, 1, 2, 3, 5, 8]))
    >>> cache.record_query(terms=[1, 1, 2, 3, 5, 8], ids=None, found=["A000045"])
    >>> cache.n_entries
    1
    """

    #: Written into every file this class saves.
    LICENSE = (
        "Sequence data and formula lines are from the On-Line Encyclopedia of "
        "Integer Sequences (https://oeis.org), (c) The OEIS Foundation Inc., "
        "licensed CC BY-NC-SA 4.0."
    )

    def __init__(self, path: Any = None):
        """:param path: a JSON file to load, if it exists."""
        self.path = Path(path) if path is not None else None
        self._entries: dict = {}
        self._queries: dict = {}
        if self.path is not None and self.path.exists():
            self.load(self.path)

    @property
    def name(self) -> str:
        """Identifies this source in a verdict."""
        return "oeis-cache"

    @property
    def n_entries(self) -> int:
        """Entries held."""
        return len(self._entries)

    @property
    def n_queries(self) -> int:
        """Queries whose full result is known, including the empty ones."""
        return len(self._queries)

    def add(self, entry: OeisEntry) -> None:
        """Store *entry*, replacing any entry with the same A-number."""
        self._entries[entry.id] = entry

    def record_query(
        self,
        *,
        terms: Sequence[int] | None = None,
        ids: Sequence[str] | None = None,
        found: Sequence[str] = (),
    ) -> None:
        """Record that this exact query was put to OEIS and returned *found*.

        An empty *found* is meaningful — it is the only way a cache can support
        a ``not_found`` verdict.
        """
        self._queries[_query_key(terms, ids)] = list(found)

    def load(self, path: Any) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for raw in payload.get("entries", {}).values():
            self.add(
                OeisEntry(
                    id=raw["id"],
                    name=raw.get("name", ""),
                    terms=raw.get("terms", ()),
                    statements=raw.get("statements", ()),
                    offset=raw.get("offset", 0),
                )
            )
        self._queries.update(payload.get("queries", {}))

    def save(self, path: Any = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to: construct with one or pass one here")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "kind": "alkahest.oeis_cache",
                    "version": 1,
                    "license": self.LICENSE,
                    "entries": {k: v.to_json() for k, v in sorted(self._entries.items())},
                    "queries": {k: sorted(v) for k, v in sorted(self._queries.items())},
                },
                indent=1,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def lookup(
        self,
        *,
        terms: Sequence[int] | None = None,
        ids: Sequence[str] | None = None,
    ) -> SourceAnswer | None:
        """Entries for a query, or ``None`` when this cache cannot answer it.

        A **hit** is real evidence wherever it comes from, so a stored entry
        whose data contains the queried run is returned even if that exact
        query was never recorded. A **miss** is only reported as exhaustive
        when the query itself was recorded, because otherwise all it means is
        that the cache is small.
        """
        key = _query_key(terms, ids)
        recorded = self._queries.get(key)
        if ids:
            entries = tuple(self._entries[i] for i in ids if i in self._entries)
            if entries or recorded is not None:
                missing = [i for i in ids if i not in self._entries]
                return SourceAnswer(entries, exhaustive=recorded is not None or not missing)
            return None
        wanted = tuple(int(t) for t in terms or ())
        if not wanted:
            return None
        hits = {e.id: e for e in self._entries.values() if _contains_run(e.terms, wanted)}
        if recorded is not None:
            # Everything the source returned for this exact query, whether or
            # not the stored data happens to contain the run verbatim — OEIS
            # matches on more than a literal prefix.
            for entry_id in recorded:
                if entry_id in self._entries:
                    hits.setdefault(entry_id, self._entries[entry_id])
        elif not hits:
            return None
        return SourceAnswer(tuple(hits[k] for k in sorted(hits)), exhaustive=recorded is not None)

    def __repr__(self) -> str:
        return f"OeisCache(entries={self.n_entries}, queries={self.n_queries})"


def _contains_run(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start] == first and tuple(haystack[start : start + len(needle)]) == tuple(
            needle
        ):
            return True
    return False


class OeisWeb:
    """Live OEIS lookup. **Opt-in**: nothing constructs one for you.

    Serves from its :class:`OeisCache` before it touches the network, sleeps
    ``min_interval`` seconds between requests, sends a User-Agent that says who
    is calling, and **returns ``None`` instead of raising** when the network is
    not there — so an offline run degrades to ``unavailable``, which is the
    honest verdict, rather than to an exception or, far worse, to a negative.

    No test in this repository constructs one. Record a fixture once::

        web = OeisWeb(cache=OeisCache())
        web.lookup(ids=["A005259"])
        web.cache.save("tests/data/oeis_novelty_fixture.json")
    """

    #: Shared across instances so several sources cannot bypass the interval.
    _last_request: ClassVar[list] = [0.0]

    def __init__(
        self,
        cache: OeisCache | None = None,
        *,
        min_interval: float = 1.0,
        timeout: float = 30.0,
        user_agent: str = "alkahest-novelty/1.0 (+https://github.com/alkahest-cas/alkahest)",
        max_results: int = 10,
    ):
        self.cache = cache if cache is not None else OeisCache()
        self.min_interval = float(min_interval)
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self.max_results = int(max_results)
        #: Why the last lookup came back ``None``, for a report.
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        """Identifies this source in a verdict."""
        return "oeis"

    def lookup(
        self,
        *,
        terms: Sequence[int] | None = None,
        ids: Sequence[str] | None = None,
    ) -> SourceAnswer | None:
        cached = self.cache.lookup(terms=terms, ids=ids)
        if cached is not None and cached.exhaustive:
            return cached
        query = (
            " ".join(f"id:{i}" for i in ids) if ids else ",".join(str(int(t)) for t in terms or ())
        )
        if not query:
            return None
        payload = self._fetch(query)
        if payload is None:
            return cached
        entries = []
        for raw in payload[: self.max_results]:
            try:
                entry = OeisEntry.from_oeis_json(raw)
            except (KeyError, TypeError, ValueError):
                continue
            self.cache.add(entry)
            entries.append(entry)
        self.cache.record_query(terms=terms, ids=ids, found=[e.id for e in entries])
        return SourceAnswer(tuple(entries), exhaustive=True)

    def _fetch(self, query: str) -> list | None:
        self.last_error = None
        wait = self.min_interval - (time.monotonic() - self._last_request[0])
        if wait > 0:
            time.sleep(wait)
        url = "https://oeis.org/search?" + urllib.parse.urlencode({"q": query, "fmt": "json"})
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                body = handle.read().decode("utf-8")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            self._last_request[0] = time.monotonic()
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            self.last_error = f"JSONDecodeError: {exc}"
            return None
        if isinstance(decoded, dict):
            decoded = decoded.get("results") or []
        return decoded if isinstance(decoded, list) else []

    def __repr__(self) -> str:
        return f"OeisWeb(cache={self.cache!r}, min_interval={self.min_interval})"


# ---------------------------------------------------------------------------
# The verdict.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoveltyMatch:
    """One place a source states the claim."""

    source: str
    entry: str
    statement: str
    hedged: bool

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "entry": self.entry,
            "statement": self.statement,
            "hedged": self.hedged,
        }


class NoveltyVerdict:
    """The result of a novelty check. Read :attr:`status`, never ``bool()``.

    See the module docstring for what a negative is allowed to claim. In short:
    :attr:`found` is ``True``/``False``/``None`` and ``False`` means "not in
    the sources searched", which is not novelty.
    """

    __slots__ = (
        "_claim_hash",
        "_consulted",
        "_entries",
        "_matches",
        "_statements",
        "_unavailable",
        "_unusable",
    )

    def __init__(
        self,
        *,
        claim_hash: str,
        matches: Sequence[NoveltyMatch],
        consulted: Sequence[str],
        unavailable: Sequence[str],
        entries: int,
        statements: int,
        unusable: int,
    ):
        self._claim_hash = claim_hash
        self._matches = tuple(matches)
        self._consulted = tuple(consulted)
        self._unavailable = tuple(unavailable)
        self._entries = entries
        self._statements = statements
        self._unusable = unusable

    @property
    def status(self) -> str:
        """One of :data:`NOVELTY_STATUSES`."""
        if self._matches:
            return "recorded_conjecturally" if self.hedged else "recorded"
        if self._consulted:
            return "not_found"
        return "unavailable"

    @property
    def found(self) -> bool | None:
        """``True`` / ``False`` / ``None`` — see the module docstring.

        ``None`` is not a pass and ``False`` is not novelty; both mean the
        question is open, for different reasons.
        """
        if self._matches:
            return True
        return False if self._consulted else None

    @property
    def hedged(self) -> bool | None:
        """Whether every source that states the claim marks it conjectural.

        ``None`` when nothing states it. OEIS labels its guessed formulas
        (``Conjecture``, ``Empirical``), and that label is the difference
        between "this is known" and "this is believed" — proving a hedged
        recurrence is a result; restating an unhedged one is not.
        """
        if not self._matches:
            return None
        return all(m.hedged for m in self._matches)

    @property
    def claim_hash(self) -> str:
        """Content address of the claim that was checked."""
        return self._claim_hash

    @property
    def entries_examined(self) -> int:
        """Source entries whose statements were compared against the claim."""
        return self._entries

    @property
    def statements_compared(self) -> int:
        """Statements that were parsed into a claim and compared."""
        return self._statements

    @property
    def statements_unusable(self) -> int:
        """Statements that mention a recurrence but could not be compared.

        Every one of these is a way this verdict could be wrong in the
        ``not_found`` direction, which is why it is reported rather than
        swallowed.
        """
        return self._unusable

    @property
    def means(self) -> str:
        """The one-line gloss of :attr:`status` from :data:`STATUS_MEANINGS`."""
        return STATUS_MEANINGS[self.status]

    def matches(self) -> tuple:
        """Every :class:`NoveltyMatch` found."""
        return self._matches

    def sources_consulted(self) -> tuple:
        """Names of the sources that were able to answer."""
        return self._consulted

    def sources_unavailable(self) -> tuple:
        """Names of the sources that were not."""
        return self._unavailable

    def report(self) -> dict:
        """The verdict and everything that went into it, for a log.

        Sibling of :meth:`alkahest.GuessedRecurrence.evidence` and
        :func:`alkahest.relation_confidence`'s return value: a research loop
        should be able to record *why* a claim was reported, not only that it
        was.
        """
        return {
            "status": self.status,
            "found": self.found,
            "hedged": self.hedged,
            "means": self.means,
            "claim_hash": self._claim_hash,
            "matches": [m.to_json() for m in self._matches],
            "sources_consulted": list(self._consulted),
            "sources_unavailable": list(self._unavailable),
            "entries_examined": self._entries,
            "statements_compared": self._statements,
            "statements_unusable": self._unusable,
        }

    def __bool__(self) -> bool:
        raise TypeError(
            "a NoveltyVerdict has no truth value: `if verdict:` would read as "
            "'is this novel?' and there is no such answer here. Test "
            "verdict.status ('recorded', 'recorded_conjecturally', "
            "'not_found', 'unavailable') or verdict.found, which is "
            "True/False/None and whose False means 'not in the sources "
            "searched', not 'new'"
        )

    def __repr__(self) -> str:
        return (
            f"NoveltyVerdict(status={self.status!r}, matches={len(self._matches)}, "
            f"entries_examined={self._entries}, sources_consulted="
            f"{list(self._consulted)})"
        )


def check_novelty(
    claim: RecurrenceClaim,
    sources: Sequence[Any],
    *,
    terms: Sequence[int] | None = None,
    ids: Sequence[str] | None = None,
) -> NoveltyVerdict:
    """Look *claim* up in *sources* and report what was found.

    :param claim: the normalised claim — build it with
        :meth:`RecurrenceClaim.from_recurrence` from a
        :class:`~alkahest.ZeilbergerCertificate` or a
        :class:`~alkahest.GuessedRecurrence`.
    :param sources: objects with a ``name`` and a
        ``lookup(*, terms=None, ids=None)`` returning a :class:`SourceAnswer`
        or ``None``. :class:`OeisCache` offline, :class:`OeisWeb` live.
        **There is no default**: a check with no source returns
        ``unavailable``, and this module will not quietly reach for the network
        on your behalf.
    :param terms: exact leading terms of the sequence, to identify it. Give
        enough that the identification is not accidental — ten is plenty for a
        sequence that grows.
    :param ids: source-specific identifiers to check instead, e.g.
        ``["A005259"]``.

    :returns: a :class:`NoveltyVerdict`. Never raises for a missing source or a
        dead network; those are ``unavailable``.
    :raises TypeError: when *claim* is not a :class:`RecurrenceClaim`.
    :raises ValueError: when neither *terms* nor *ids* is given.

    >>> from alkahest.experimental.novelty import (
    ...     OeisCache, OeisEntry, RecurrenceClaim, check_novelty)
    >>> entry = OeisEntry(
    ...     "A000984", "Central binomial coefficients",
    ...     terms=[1, 2, 6, 20, 70, 252, 924, 3432],
    ...     statements=["D-finite with recurrence: n*a(n) + 2*(1-2*n)*a(n-1)=0."])
    >>> cache = OeisCache()
    >>> cache.add(entry)
    >>> claim = RecurrenceClaim([(-2, -4), (1, 1)])       # (n+1)u(n+1) = (4n+2)u(n)
    >>> verdict = check_novelty(claim, [cache], terms=[1, 2, 6, 20, 70, 252])
    >>> verdict.status, verdict.found, verdict.hedged
    ('recorded', True, False)

    The same lookup with nothing to look in is ``unavailable``, not novel:

    >>> check_novelty(claim, [], terms=[1, 2, 6, 20]).found is None
    True
    """
    if not isinstance(claim, RecurrenceClaim):
        raise TypeError(
            "claim must be a RecurrenceClaim; build one with "
            "RecurrenceClaim.from_recurrence(certificate, var=n) so that what "
            "is looked up is the normal form, not one presentation of it"
        )
    if not terms and not ids:
        raise ValueError(
            "give terms= (the sequence's leading terms) or ids= (source "
            "identifiers); there is nothing to look up otherwise"
        )
    matches, consulted, unavailable = [], [], []
    seen_entries: dict = {}
    statements = unusable = 0
    for source in sources:
        name = getattr(source, "name", type(source).__name__)
        answer = source.lookup(terms=terms, ids=ids)
        if answer is None or not answer.exhaustive:
            unavailable.append(name)
            if answer is None:
                continue
        else:
            consulted.append(name)
        for entry in answer.entries:
            if entry.id in seen_entries:
                continue
            seen_entries[entry.id] = True
            unusable += len(entry.unusable_statements())
            for record in entry.recurrences():
                statements += 1
                if record.claim.claim_hash == claim.claim_hash:
                    matches.append(
                        NoveltyMatch(
                            source=name,
                            entry=entry.id,
                            statement=record.statement,
                            hedged=record.hedged,
                        )
                    )
    return NoveltyVerdict(
        claim_hash=claim.claim_hash,
        matches=matches,
        consulted=consulted,
        unavailable=unavailable,
        entries=len(seen_entries),
        statements=statements,
        unusable=unusable,
    )
