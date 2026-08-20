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

:class:`QRecurrenceClaim` is (1) and (2) for a ``q``-recurrence, whose
coefficients are Laurent polynomials in ``q`` and ``q^n`` and so are not
polynomials in ``n`` at all. No source here can state one, which
:func:`check_novelty` reports as *unavailable* rather than as a negative.

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

Two things feed that honesty and are easy to get wrong, so they are stated
here as well as at their definitions. A ``terms=`` search of OEIS is **paged**:
``fmt=json`` returns at most ten results with no total count, so a full page is
not an exhaustive answer and :class:`OeisWeb` keeps asking until it sees a short
one or gives up and says ``exhaustive=False``. And the *terms* a caller looks a
claim up by are **checked against the claim** — a claim that does not reproduce
them was never about the sequence that was searched for, and
:attr:`NoveltyVerdict.terms_check` says so.

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
this repository requires the network**: the offline path is :class:`OeisCache`,
whose fixtures are recorded once and committed, and the two tests that do
construct an :class:`OeisWeb` — for the paging in :meth:`OeisWeb.lookup` —
replace its transport with recorded pages. OEIS data
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
    "TERMS_CHECKS",
    "NoveltyMatch",
    "NoveltyVerdict",
    "OeisCache",
    "OeisEntry",
    "OeisWeb",
    "QRecurrenceClaim",
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

#: Every answer :attr:`NoveltyVerdict.terms_check` can give.
TERMS_CHECKS = ("holds", "fails", "not_checked")

#: Trailing windows of the entry's own data a parsed recurrence must satisfy
#: before it is believed to be what the formula line meant.  The same rule
#: re-checks the caller's own claim against the terms it looked it up by.
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
# Laurent polynomials over ℚ in two variables, `q` and `Q = q^n`.
#
# This is what a `q`-recurrence coefficient is: `q_zeilberger` returns
# `1 + q*q^n - q*q^(2*n) - q^2*q^(3*n)`, which is neither a polynomial in `n`
# nor one in `q` alone.  A term is a dict entry `(i, j) -> c` for `c*q^i*Q^j`,
# with `i` and `j` allowed to be negative; the zero polynomial is `{}`.
# ---------------------------------------------------------------------------


def _q_trim(terms: dict) -> dict:
    return {m: c for m, c in terms.items() if c}


def _q_add(a: dict, b: dict) -> dict:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, Fraction(0)) + c
    return _q_trim(out)


def _q_mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for (i, j), x in a.items():
        for (k, e), y in b.items():
            m = (i + k, j + e)
            out[m] = out.get(m, Fraction(0)) + x * y
    return _q_trim(out)


def _q_pow(a: dict, e: int) -> dict:
    out = {(0, 0): Fraction(1)}
    for _ in range(e):
        out = _q_mul(out, a)
    return out


def _q_substitute_shift(a: dict, s: int) -> dict:
    """``a`` with ``n → n + s``, i.e. ``Q → q^s·Q``."""
    if s == 0:
        return dict(a)
    return {(i + s * j, j): c for (i, j), c in a.items()}


def _q_monomial_content(polys: Sequence[dict]) -> tuple:
    """The largest ``(q^i, Q^j)`` dividing every term of every polynomial."""
    live = [m for p in polys for m in p]
    if not live:
        return (0, 0)
    return (min(i for i, _ in live), min(j for _, j in live))


def _q_columns(a: dict) -> list:
    """*a* as a polynomial in ``Q`` whose coefficients are polynomials in ``q``.

    Requires non-negative exponents — divide the monomial content out first.
    A list indexed by the power of ``Q``, each entry a ``_p_*`` tuple.
    """
    if not a:
        return []
    columns: list = [()] * (max(j for _, j in a) + 1)
    for (i, j), c in a.items():
        column = list(columns[j]) + [Fraction(0)] * (i + 1 - len(columns[j]))
        column[i] += c
        columns[j] = _trim(column)
    return columns


def _q_from_columns(columns: Sequence[tuple]) -> dict:
    return _q_trim(
        {(i, j): c for j, column in enumerate(columns) for i, c in enumerate(column) if c}
    )


def _q_col_trim(columns: Sequence[tuple]) -> list:
    out = list(columns)
    while out and not out[-1]:
        out.pop()
    return out


def _q_col_content(columns: Sequence[tuple]) -> tuple:
    common: tuple = ()
    for column in columns:
        common = _p_gcd(common, column)
    return common


def _q_col_primitive(columns: Sequence[tuple]) -> list:
    common = _q_col_content(columns)
    if not common:
        return list(columns)
    return [_p_divmod(column, common)[0] for column in columns]


def _q_col_prem(a: Sequence[tuple], b: Sequence[tuple]) -> list:
    """Pseudo-remainder of *a* by *b* in ``ℚ[q][Q]``."""
    rem = _q_col_trim(a)
    b = _q_col_trim(b)
    while rem and len(rem) >= len(b):
        shift = len(rem) - len(b)
        lead_a, lead_b = rem[-1], b[-1]
        scaled = [_p_mul(c, lead_b) for c in rem]
        for i, c in enumerate(b):
            scaled[shift + i] = _p_sub(scaled[shift + i], _p_mul(lead_a, c))
        rem = _q_col_trim(scaled)
    return rem


def _q_col_gcd(a: Sequence[tuple], b: Sequence[tuple]) -> list:
    """gcd in ``ℚ[q][Q]`` by the primitive Euclidean algorithm."""
    a, b = _q_col_trim(a), _q_col_trim(b)
    if not a:
        return list(b)
    if not b:
        return list(a)
    content = _p_gcd(_q_col_content(a), _q_col_content(b))
    a, b = _q_col_primitive(a), _q_col_primitive(b)
    while b:
        a, b = b, _q_col_primitive(_q_col_prem(a, b))
    return [_p_mul(column, content) for column in a]


def _q_col_divexact(a: Sequence[tuple], b: Sequence[tuple]) -> list | None:
    """``a / b`` in ``ℚ[q][Q]`` when it is exact, else ``None``."""
    rem = _q_col_trim(a)
    b = _q_col_trim(b)
    quotient: list = [()] * max(1, len(rem) - len(b) + 1)
    while rem and len(rem) >= len(b):
        shift = len(rem) - len(b)
        factor, residue = _p_divmod(rem[-1], b[-1])
        if residue or not factor:
            return None
        quotient[shift] = factor
        scaled = list(rem)
        for i, c in enumerate(b):
            scaled[shift + i] = _p_sub(scaled[shift + i], _p_mul(factor, c))
        rem = _q_col_trim(scaled)
    return None if rem else _q_col_trim(quotient)


def _q_text(a: dict) -> str:
    """Canonical text for a ``q``-coefficient: descending in ``q^n``, then ``q``."""
    if not a:
        return "0"
    parts = []
    for i, j in sorted(a, key=lambda m: (m[1], m[0]), reverse=True):
        c = a[(i, j)]
        factors = []
        if i:
            factors.append("q" if i == 1 else f"q^{i}")
        if j:
            factors.append("q^n" if j == 1 else f"q^({j}*n)")
        if not factors:
            parts.append(str(c))
        elif c == 1:
            parts.append("*".join(factors))
        elif c == -1:
            parts.append("-" + "*".join(factors))
        else:
            parts.append("*".join([str(c), *factors]))
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
            hint = " — a q-recurrence is a QRecurrenceClaim, not this one" if node[1] == "q" else ""
            raise ValueError(
                f"coefficient mentions the symbol {node[1]!r}, but a recurrence "
                f"coefficient must be a polynomial in {var!r} alone{hint}"
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
    def from_text(cls, text: str, *, names: Sequence[str] = ()) -> RecurrenceClaim | None:
        """Parse one prose formula line, e.g. an OEIS ``a(n) = …`` statement.

        The sequence may be written ``a(n)``, as any single letter (``F(n) =
        F(n-1) + F(n-2)`` is how A000045 states the Fibonacci recurrence, in its
        *name*), or under any identifier in *names* — pass the entry's own
        A-number there so a line that spells it out is read as being about
        itself. Whichever is used, **one** line may only name one sequence: a
        relation between two of them is not a recurrence for either.

        Returns ``None`` — never a guess — when the line is not a homogeneous
        linear recurrence with polynomial coefficients in a single sequence: a
        sum, a generating function, a nonlinear relation, a reference to another
        sequence, an inhomogeneous relation, or a statement the parser simply
        does not cover. Callers that need to know how often that happened should
        count the ``None``s; :meth:`NoveltyVerdict.report` does.

        :param names: extra identifiers that denote the sequence the line is
            about, e.g. ``names=("A000045",)``.
        """
        own = frozenset({"a", *names})
        try:
            relation = _parse_relation(text, own)
        except _Unsupported:
            return None
        if relation is None:
            return None
        try:
            return cls([relation[j] for j in sorted(relation)], offset=min(relation))
        except ValueError:
            return None

    @property
    def claim_kind(self) -> str:
        """``"recurrence"`` — what a source must be able to state to match this.

        See :class:`QRecurrenceClaim`, whose kind is ``"q-recurrence"``; a
        source that cannot state a kind is ``unavailable`` for it, never
        ``not_found``.
        """
        return "recurrence"

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
# The `q`-analogue of the claim.
# ---------------------------------------------------------------------------


def _q_exponent(expr: Any, var: str) -> tuple | None:
    """``(d, c)`` for an exponent ``d + c·n`` with integer ``d`` and ``c``."""
    try:
        poly = _poly_from_expr(expr, var)
    except (ValueError, _Unsupported):
        return None
    if len(poly) > 2:
        return None
    padded = [poly[i] if i < len(poly) else Fraction(0) for i in range(2)]
    if any(c.denominator != 1 for c in padded):
        return None
    return int(padded[0]), int(padded[1])


def _qpoly_from_expr(expr: Any, qname: str, var: str) -> tuple:
    """``(numerator, denominator)`` in ``ℚ[q^±1, Q^±1]``, ``Q = q^n``, exactly."""
    node = expr.node()
    head = node[0]
    one = {(0, 0): Fraction(1)}
    if head == "integer":
        return _q_trim({(0, 0): Fraction(int(node[1]))}), one
    if head == "rational":
        return _q_trim({(0, 0): Fraction(int(node[1]), int(node[2]))}), one
    if head == "symbol":
        if node[1] == qname:
            return {(1, 0): Fraction(1)}, one
        raise ValueError(
            f"coefficient mentions the symbol {node[1]!r}, but a q-recurrence "
            f"coefficient must be a rational function of {qname!r} and "
            f"{qname}^{var}"
        )
    if head == "add":
        num: dict = {}
        den = one
        for child in node[1]:
            other_num, other_den = _qpoly_from_expr(child, qname, var)
            num = _q_add(_q_mul(num, other_den), _q_mul(other_num, den))
            den = _q_mul(den, other_den)
        return num, den
    if head == "mul":
        num, den = one, one
        for child in node[1]:
            other_num, other_den = _qpoly_from_expr(child, qname, var)
            num, den = _q_mul(num, other_num), _q_mul(den, other_den)
        return num, den
    if head == "pow":
        exponent = _q_exponent(node[2], var)
        if exponent is None:
            raise ValueError(
                f"{expr} is not a rational function of {qname!r} and {qname}^{var}: "
                f"an exponent must be an integer or an integer multiple of {var!r}"
            )
        offset, slope = exponent
        base = node[1].node()
        if slope:
            if base[0] != "symbol" or base[1] != qname:
                raise ValueError(
                    f"{expr} raises something other than {qname!r} to a power in {var}"
                )
            return {(offset, slope): Fraction(1)}, one
        num, den = _qpoly_from_expr(node[1], qname, var)
        if offset < 0:
            num, den, offset = den, num, -offset
        if not num:
            raise ValueError(f"{expr} divides by zero")
        return _q_pow(num, offset), _q_pow(den, offset)
    raise ValueError(f"{expr} is not a rational function of {qname!r} and {qname}^{var}")


def _q_coerce(value: Any, qname: str | None, var: str | None) -> tuple:
    """One ``q``-coefficient, from an ``Expr`` or a ``{(i, j): rational}`` map."""
    if hasattr(value, "node"):
        if qname is None or var is None:
            raise TypeError(
                "coefficients given as Expr need both index variables: pass "
                "var=n (the index) and q=q (the base), the two symbols a "
                "q-recurrence coefficient is built from"
            )
        return _qpoly_from_expr(value, qname, var)
    one = {(0, 0): Fraction(1)}
    if isinstance(value, (int, Fraction)):
        return _q_trim({(0, 0): Fraction(value)}), one
    return _q_trim({(int(i), int(j)): Fraction(c) for (i, j), c in dict(value).items()}), one


def _q_normalise(shifts: dict) -> tuple:
    """Put ``Σ_j c_j(q, q^n)·u(n+j) = 0`` into normal form.

    The steps are those of :func:`_normalise`, read in ``ℚ[q^±1, Q^±1]``
    instead of ``ℚ[n]``: clear the coefficients' denominators, move the window
    to ``u(n)`` — which acts on the coefficients, since ``n → n − low`` sends
    ``Q`` to ``q^{−low}·Q`` — divide out the common monomial and the common
    polynomial factor, clear rational denominators and the integer content, and
    fix the sign.
    """
    keys = sorted(shifts)
    numerators = {}
    for j in keys:
        product = shifts[j][0]
        for k in keys:
            if k != j:
                product = _q_mul(product, shifts[k][1])
        numerators[j] = product
    live = {j: p for j, p in numerators.items() if p}
    if len(live) < 2:
        raise ValueError(
            "a q-recurrence claim needs at least two sequence terms with "
            f"nonzero coefficients, got {len(live)}"
        )
    low, high = min(live), max(live)
    polys = [_q_substitute_shift(live.get(low + i, {}), -low) for i in range(high - low + 1)]
    shift_i, shift_j = _q_monomial_content(polys)
    polys = [{(i - shift_i, j - shift_j): c for (i, j), c in p.items()} for p in polys]
    columns = [_q_columns(p) for p in polys]
    common: list = []
    for column in columns:
        common = _q_col_gcd(common, column)
    if len(common) > 1 or (common and len(common[0]) > 1):
        divided = [_q_col_divexact(column, common) for column in columns]
        if all(d is not None for d in divided):
            columns = divided
    polys = [_q_from_columns(column) for column in columns]
    multiplier = 1
    for p in polys:
        for c in p.values():
            multiplier = multiplier * c.denominator // gcd(multiplier, c.denominator)
    integral = [{m: int(c * multiplier) for m, c in p.items()} for p in polys]
    content = 0
    for p in integral:
        for c in p.values():
            content = gcd(content, abs(c))
    if content > 1:
        integral = [{m: c // content for m, c in p.items()} for p in integral]
    for p in integral:
        if not p:
            continue
        if p[max(p, key=lambda m: (m[1], m[0]))] < 0:
            integral = [{m: -c for m, c in other.items()} for other in integral]
        break
    return tuple(
        tuple(sorted(p.items(), key=lambda item: (item[0][1], item[0][0]))) for p in integral
    )


class QRecurrenceClaim:
    """A ``q``-recurrence, in a normal form two presentations share.

    The claim is ``Σ_i c_i(q, q^n)·u(n+i) = 0`` — what
    :func:`alkahest.experimental.q_zeilberger` produces, whose coefficients are
    Laurent polynomials in ``q`` and ``q^n`` over ``ℚ`` and are therefore not
    polynomials in ``n`` at all. :class:`RecurrenceClaim` refuses them
    (*"coefficient mentions the symbol 'q'"*), which left every ``q``-result
    with no route into :func:`check_novelty`; this is that route.

    Coefficients may be given as :class:`alkahest.Expr` (pass both ``var=n`` and
    ``q=q``) or as ``{(i, j): rational}`` maps, where ``(i, j)`` is the monomial
    ``q^i·(q^n)^j`` and either exponent may be negative. Rational functions are
    accepted and cleared; the normal form is always Laurent-polynomial.

    What the normal form quotients out is exactly what :attr:`RecurrenceClaim`
    quotients out, read over ``ℚ[q^±1, (q^n)^±1]``: scale, index shift — which
    here acts on the coefficients, because ``n → n+1`` sends ``q^n`` to
    ``q·q^n`` — a common monomial or polynomial factor, and zero padding.

    :attr:`claim_hash` is tagged ``q-recurrence/1`` where
    :class:`RecurrenceClaim`'s is tagged ``recurrence/1``, so the two can share
    a ``set`` without colliding even when the coefficients look alike.

    **No source in this module can state a ``q``-recurrence.** OEIS indexes
    integer sequences, and the formula parser reads ``ℚ[n]`` coefficients only,
    so :func:`check_novelty` reports every OEIS source as *unavailable* for a
    claim of this kind rather than manufacturing a ``not_found`` out of a
    search that could not have matched. What the class is good for today is the
    other half of the job: a stable content address a loop can dedupe its own
    ``q``-output with.

    >>> from alkahest.experimental.novelty import QRecurrenceClaim
    >>> # (1 - q^n)·u(n) - u(n+1) = 0
    >>> a = QRecurrenceClaim([{(0, 0): 1, (0, 1): -1}, {(0, 0): -1}])
    >>> # the same relation about u(n+3), scaled by -2q
    >>> b = QRecurrenceClaim(
    ...     [{(1, 0): -2, (4, 1): 2}, {(1, 0): 2}], offset=3)
    >>> a.claim_hash == b.claim_hash
    True
    >>> a.order, a.degree, a.q_degree
    (1, 1, 0)
    """

    __slots__ = ("_coefficients", "_hash", "_normal_form")

    def __init__(
        self,
        coefficients: Sequence[Any],
        *,
        offset: int = 0,
        var: Any = None,
        q: Any = None,
    ):
        """
        :param coefficients: ``[c_0, …, c_J]``, the coefficient of ``u(n+offset+i)``.
            Each an :class:`alkahest.Expr` in *var* and *q*, or a
            ``{(i, j): rational}`` map for ``Σ c_ij·q^i·(q^n)^j``.
        :param offset: index of the first coefficient's shift.
        :param var: the index symbol, required for ``Expr`` coefficients.
        :param q: the base symbol, required for ``Expr`` coefficients.
        :raises ValueError: when fewer than two coefficients are nonzero.
        """
        name = None if var is None else (var if isinstance(var, str) else var.node()[1])
        base = None if q is None else (q if isinstance(q, str) else q.node()[1])
        shifts = {offset + i: _q_coerce(c, base, name) for i, c in enumerate(coefficients)}
        self._coefficients = _q_normalise(shifts)
        self._normal_form = "q-recurrence/1 " + " + ".join(
            f"({_q_text({m: Fraction(c) for m, c in p})})*u(n+{i})"
            for i, p in enumerate(self._coefficients)
        )
        self._hash = _claim_id(self._normal_form, method="q-recurrence")

    @classmethod
    def from_recurrence(cls, rec: Any, var: Any = None, q: Any = None) -> QRecurrenceClaim:
        """From a :class:`~alkahest.QZeilbergerCertificate` or a coefficient list.

        Duck-typed on ``.coeffs``, exactly as
        :meth:`RecurrenceClaim.from_recurrence` is. Both *var* and *q* are
        required when the coefficients are expressions.
        """
        coeffs = getattr(rec, "coeffs", None)
        return cls(list(rec) if coeffs is None else list(coeffs), var=var, q=q)

    @property
    def claim_kind(self) -> str:
        """``"q-recurrence"`` — what a source must be able to state to match this."""
        return "q-recurrence"

    @property
    def order(self) -> int:
        """``J`` — the span of the window in normal form."""
        return len(self._coefficients) - 1

    @property
    def degree(self) -> int:
        """Largest power of ``q^n`` in any coefficient in normal form."""
        return max((max(j for (_, j), _ in p) for p in self._coefficients if p), default=0)

    @property
    def q_degree(self) -> int:
        """Largest power of ``q`` alone in any coefficient in normal form."""
        return max((max(i for (i, _), _ in p) for p in self._coefficients if p), default=0)

    @property
    def normal_form(self) -> str:
        """The canonical text the hash is taken of, tagged ``q-recurrence/1``."""
        return self._normal_form

    @property
    def claim_hash(self) -> str:
        """Content address of :attr:`normal_form`, e.g. ``'clm_9f1c0b2a7d4e5f60'``."""
        return self._hash

    def coefficients(self) -> tuple:
        """``(c_0, …, c_J)`` in normal form.

        Each is a tuple of ``((i, j), coefficient)`` pairs for
        ``coefficient·q^i·(q^n)^j``, ascending in ``j`` then ``i``.
        """
        return self._coefficients

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QRecurrenceClaim):
            return NotImplemented
        return self._normal_form == other._normal_form

    def __hash__(self) -> int:
        return hash(self._normal_form)

    def __repr__(self) -> str:
        return (
            f"QRecurrenceClaim(order={self.order}, degree={self.degree}, "
            f"q_degree={self.q_degree}, claim_hash={self._hash!r})"
        )


# ---------------------------------------------------------------------------
# Parsing prose formula lines.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*|\d+|\*\*|\S)")
_OPERATORS = frozenset("+-*/^")
_OPENERS = frozenset("([{")
_CLOSERS = frozenset(")]}")
#: A statement worth handing to the parser at all: it mentions a shifted term.
#: OEIS does not only write ``a(n-1)``. An entry's own definition line names the
#: sequence after the objects it counts — ``F(n) = F(n-1) + F(n-2)`` is the whole
#: content of A000045's name — so any single letter counts as a candidate here.
#: Which of them the parser will accept as a term of *this* sequence is decided
#: in :class:`_Parser`, not here. An ``A123456(n-1)`` cross-reference is
#: deliberately *not* a candidate on its own: over the 377-entry sample this
#: module was measured against it added 256 lines and not one parse.
_LOOKS_LIKE_RECURRENCE = re.compile(r"\b[A-Za-z]\(\s*n\s*[-+]\s*\d+\s*\)")
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


#: Identifiers that may denote *the* sequence a line is about, beyond ``a`` and
#: whatever the caller adds: any single letter, because OEIS names a sequence
#: after what it counts (``F`` for Fibonacci, ``L`` for Lucas, ``T``, ``b``).
#: A multi-letter identifier is never one of these, which is what keeps
#: ``floor``, ``sqrt``, ``binomial``, ``Sum`` and ``A123456`` out.
def _is_sequence_name(token: str, own: frozenset) -> bool:
    return token in own or (len(token) == 1 and token.isalpha())


class _Parser:
    """Recursive descent over ``+ - * / ^ ( )``, integers, ``n`` and ``F(n±k)``.

    Everything else — a function that is not a sequence, ``Sum_{…}``, a symbol
    that is not the index — raises :class:`_Unsupported`. Refusing is the point:
    a parser that guesses at prose invents claims that were never made.

    The sequence identifiers a parse used are collected in :attr:`names`, and
    :func:`_parse_relation` refuses the line unless they all denote the same
    sequence — so ``a(n) = a(n-1) + A002026(n-1)`` is still refused, and so is
    ``F(n) = L(n-1) + L(n-2)``, while ``F(n) = F(n-1) + F(n-2)`` is read.
    """

    def __init__(self, tokens: Sequence[str], own: frozenset = frozenset({"a"})):
        self.tokens = tokens
        self.pos = 0
        self.own = own
        #: Sequence identifiers this parse applied to an index.
        self.names: set = set()

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
        while True:
            if self.peek() in ("*", "/"):
                op = self.tokens[self.pos]
                self.pos += 1
                node = node * self.factor() if op == "*" else node / self.factor()
            elif self._starts_factor():
                # Juxtaposition is multiplication: `2a(n-2)`, `(n+1)a(n-1)`.
                # OEIS's machine-written "D-finite with recurrence" lines always
                # spell the `*` out, but the hand-written ones do not.
                node = node * self.factor()
            else:
                return node

    def _starts_factor(self) -> bool:
        """Whether an implicit ``*`` may be read before the next token.

        Only a bracket, a number or a sequence application counts. A bare word
        never does — ``a(n) = a(n-1) + a(n-2) for n > 2`` must end at ``for``
        rather than read ``for`` as a factor, and a bare ``n`` is excluded for
        the same reason: ``, n > 2`` is prose, not a coefficient.
        """
        token = self.peek()
        if token is None:
            return False
        if token == "(" or token.isdigit():
            return True
        return (
            _is_sequence_name(token, self.own)
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1] == "("
        )

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
        if _is_sequence_name(token, self.own):
            if self.peek() != "(":
                raise _Unsupported("sequence name not applied to an index")
            self.pos += 1
            index = self.expression()
            if self.peek() != ")":
                raise _Unsupported("unbalanced parenthesis in a sequence index")
            self.pos += 1
            self.names.add(token)
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


def _parse_all(tokens: Sequence[str], own: frozenset) -> tuple | None:
    """``(form, sequence identifiers used)`` for a whole token run, or ``None``."""
    parser = _Parser(tokens, own)
    try:
        form = parser.expression()
    except _Unsupported:
        return None
    return (form, parser.names) if parser.pos == len(tokens) else None


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


def _parse_relation(text: str, own: frozenset = frozenset({"a"})) -> dict | None:
    """``{shift: coefficient polynomial}`` for a prose linear recurrence, or ``None``.

    *own* is the set of identifiers known to name the sequence the line is about
    — ``a`` always, plus the entry's own A-number when there is one.
    """
    tokens = _tokenise(_clean(text))
    split = _top_level_equals(tokens)
    if split is None:
        return None
    lhs = None
    for start in range(split):
        if not _boundary_ok(tokens[start - 1] if start else None):
            continue
        lhs = _parse_all(tokens[start:split], own)
        if lhs is not None:
            break
    if lhs is None:
        return None
    rhs = None
    for stop in range(len(tokens), split, -1):
        if not _boundary_ok(tokens[stop] if stop < len(tokens) else None):
            continue
        rhs = _parse_all(tokens[split + 1 : stop], own)
        if rhs is not None:
            break
    if rhs is None:
        return None
    used = lhs[1] | rhs[1]
    if len(used) > 1 and not used <= own:
        # Two different sequences in one relation: `a(n) = a(n-1) + A002026(n-1)`
        # is a statement about two sequences, not a recurrence for either.
        return None
    form = lhs[0] - rhs[0]
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

        Only the formula and comment lines that mention a shifted sequence term
        are kept: the rest cannot state a recurrence, and a cache that keeps
        them is a cache nobody commits. The entry's ``name`` is stored whole and
        scanned as a candidate line in its own right — see
        :meth:`candidate_lines`.
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

    def candidate_lines(self) -> tuple:
        """Every line that may state a recurrence: the name, then the statements.

        The **name** is here because that is where OEIS puts the recurrence for
        the entries that are defined by one: A000045's whole name is *"Fibonacci
        numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1"*, and a
        filter that reads only the formula lines cannot find the Fibonacci
        recurrence in the Fibonacci entry.
        """
        name = self.name.strip()
        if name and _LOOKS_LIKE_RECURRENCE.search(name):
            return (name, *self.statements)
        return self.statements

    def _scanned(self) -> tuple:
        if self._scan is None:
            usable, unusable = [], []
            for statement in self.candidate_lines():
                claim = RecurrenceClaim.from_text(statement, names=(self.id,))
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
        """Candidate lines that could not be turned into a claim.

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

    #: The kinds of claim this source is able to state at all. OEIS indexes
    #: integer sequences and the formula parser reads ``ℚ[n]`` coefficients, so
    #: a :class:`QRecurrenceClaim` is not something a search here could match —
    #: :func:`check_novelty` reads this and reports *unavailable* rather than
    #: turning a search that could not match into a ``not_found``.
    CLAIM_KINDS: ClassVar[tuple] = ("recurrence",)

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

    **A ``terms=`` search is paged, an ``ids=`` lookup is not.** ``fmt=json``
    answers a search with a bare list of at most :data:`PAGE_SIZE` results and
    no total count, so a single full page is not evidence that there is nothing
    else: the search continues at ``&start=`` until a short page comes back
    (there is no more) or *max_results* is reached (there may be, and the
    answer says :attr:`SourceAnswer.exhaustive` is ``False``, which
    :func:`check_novelty` turns into ``unavailable`` rather than ``not_found``).
    An ``id:A…`` query asks for named entries and gets exactly them, so it is
    exhaustive after one request.

    No test in this repository points one at the network. Record a fixture once::

        web = OeisWeb(cache=OeisCache())
        web.lookup(ids=["A005259"])
        web.cache.save("tests/data/oeis_novelty_fixture.json")
    """

    #: Results one ``fmt=json`` request returns. OEIS's own page size; the JSON
    #: form carries no total count, so this is the only signal a search is over.
    PAGE_SIZE: ClassVar[int] = 10

    #: As :attr:`OeisCache.CLAIM_KINDS` — the same encyclopaedia, live.
    CLAIM_KINDS: ClassVar[tuple] = ("recurrence",)

    #: Shared across instances so several sources cannot bypass the interval.
    _last_request: ClassVar[list] = [0.0]

    def __init__(
        self,
        cache: OeisCache | None = None,
        *,
        min_interval: float = 1.0,
        timeout: float = 30.0,
        user_agent: str = "alkahest-novelty/1.0 (+https://github.com/alkahest-cas/alkahest)",
        max_results: int = 50,
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
        payload, complete = self._fetch_all(query, paged=not ids)
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
        exhaustive = complete and len(payload) <= self.max_results
        if exhaustive:
            # Only a complete answer may be recorded as one: the cache reads a
            # recorded query as "OEIS returned exactly this", and a truncated
            # page list stored under that key would turn into a false negative
            # on every later offline run.
            self.cache.record_query(terms=terms, ids=ids, found=[e.id for e in entries])
        return SourceAnswer(tuple(entries), exhaustive=exhaustive)

    def _fetch_all(self, query: str, *, paged: bool) -> tuple:
        """``(rows, complete)`` for *query*; ``(None, False)`` if nothing arrived.

        ``complete`` is ``True`` only when OEIS has been seen to run out of
        results — a page shorter than :data:`PAGE_SIZE`, or a page that repeats
        entries already collected (OEIS clamps ``start`` past the end rather
        than returning nothing). Stopping at *max_results* instead gives
        ``False``, and so does a request that fails partway through.
        """
        rows: list = []
        seen: set = set()
        start = 0
        while True:
            page = self._fetch(query, start=start)
            if page is None:
                return (rows, False) if rows else (None, False)
            numbered = [(raw, raw.get("number") if isinstance(raw, dict) else None) for raw in page]
            fresh = [raw for raw, number in numbered if number is None or number not in seen]
            seen.update(number for _, number in numbered if number is not None)
            rows.extend(fresh)
            if not paged or len(page) < self.PAGE_SIZE or not fresh:
                return rows, True
            start += len(page)
            if len(rows) >= self.max_results:
                return rows, False

    def _fetch(self, query: str, *, start: int = 0) -> list | None:
        self.last_error = None
        wait = self.min_interval - (time.monotonic() - self._last_request[0])
        if wait > 0:
            time.sleep(wait)
        parameters: dict = {"q": query, "fmt": "json"}
        if start:
            parameters["start"] = start
        url = "https://oeis.org/search?" + urllib.parse.urlencode(parameters)
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
        "_terms_check",
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
        terms_check: str = "not_checked",
    ):
        self._claim_hash = claim_hash
        self._matches = tuple(matches)
        self._consulted = tuple(consulted)
        self._unavailable = tuple(unavailable)
        self._entries = entries
        self._statements = statements
        self._unusable = unusable
        self._terms_check = terms_check

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
    def terms_check(self) -> str:
        """Whether the claim survived the terms it was looked up by.

        One of :data:`TERMS_CHECKS`. ``check_novelty(claim, …, terms=…)`` uses
        *terms* twice: to identify the sequence to a source, and — since the
        two are supposed to be about the same sequence — to re-check the claim
        itself, on the same lenient trailing-window rule a source's own formula
        line has to pass (:meth:`RecurrenceClaim.confirmations`).

        * ``"holds"`` — the claim reproduces those terms.
        * ``"fails"`` — it does not. **The lookup was then about a different
          sequence from the claim**, so nothing it returned bears on the claim;
          either the claim is wrong, the terms are, or *start* is (see
          :meth:`RecurrenceClaim.holds_for` for what *start* must denote).
        * ``"not_checked"`` — no *terms* were given, there were too few of them
          to fill one window, or the claim is of a kind integer terms cannot
          check (:class:`QRecurrenceClaim`).
        """
        return self._terms_check

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
            "terms_check": self._terms_check,
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
        disagreement = ", terms_check='fails'" if self._terms_check == "fails" else ""
        return (
            f"NoveltyVerdict(status={self.status!r}, matches={len(self._matches)}, "
            f"entries_examined={self._entries}, sources_consulted="
            f"{list(self._consulted)}{disagreement})"
        )


def check_novelty(
    claim: RecurrenceClaim | QRecurrenceClaim,
    sources: Sequence[Any],
    *,
    terms: Sequence[int] | None = None,
    ids: Sequence[str] | None = None,
    start: int = 0,
) -> NoveltyVerdict:
    """Look *claim* up in *sources* and report what was found.

    :param claim: the normalised claim — build it with
        :meth:`RecurrenceClaim.from_recurrence` from a
        :class:`~alkahest.ZeilbergerCertificate` or a
        :class:`~alkahest.GuessedRecurrence`, or with
        :meth:`QRecurrenceClaim.from_recurrence` from a
        :class:`~alkahest.QZeilbergerCertificate`.
    :param sources: objects with a ``name`` and a
        ``lookup(*, terms=None, ids=None)`` returning a :class:`SourceAnswer`
        or ``None``. :class:`OeisCache` offline, :class:`OeisWeb` live.
        **There is no default**: a check with no source returns
        ``unavailable``, and this module will not quietly reach for the network
        on your behalf. A source may declare a ``CLAIM_KINDS`` tuple; one that
        cannot state ``claim.claim_kind`` is reported *unavailable* for it,
        because a search that could not have matched is not a negative.
    :param terms: exact leading terms of the sequence, to identify it. Give
        enough that the identification is not accidental — ten is plenty for a
        sequence that grows. They are **also checked against the claim**: see
        :attr:`NoveltyVerdict.terms_check`, and *start* below.
    :param ids: source-specific identifiers to check instead, e.g.
        ``["A005259"]``.
    :param start: the true index of ``terms[0]``, for that cross-check only —
        it is never sent to a source. Exactly the parameter of
        :meth:`RecurrenceClaim.holds_for`, and exactly as load-bearing: a
        recurrence with polynomial coefficients evaluated at the wrong ``n``
        confirms nothing, so a wrong *start* shows up as
        ``terms_check == "fails"``.

    :returns: a :class:`NoveltyVerdict`. Never raises for a missing source or a
        dead network; those are ``unavailable``.
    :raises TypeError: when *claim* is not a claim type of this module.
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
    if not isinstance(claim, (RecurrenceClaim, QRecurrenceClaim)):
        raise TypeError(
            "claim must be a RecurrenceClaim or a QRecurrenceClaim; build one "
            "with RecurrenceClaim.from_recurrence(certificate, var=n) so that "
            "what is looked up is the normal form, not one presentation of it"
        )
    if not terms and not ids:
        raise ValueError(
            "give terms= (the sequence's leading terms) or ids= (source "
            "identifiers); there is nothing to look up otherwise"
        )
    terms_check = _cross_check_terms(claim, terms, start)
    matches, consulted, unavailable = [], [], []
    seen_entries: dict = {}
    statements = unusable = 0
    for source in sources:
        name = getattr(source, "name", type(source).__name__)
        if claim.claim_kind not in getattr(source, "CLAIM_KINDS", ("recurrence",)):
            # The source cannot state a claim of this kind at all, so its
            # silence is not evidence of anything.
            unavailable.append(name)
            continue
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
        terms_check=terms_check,
    )


def _cross_check_terms(claim: Any, terms: Sequence[int] | None, start: int) -> str:
    """Re-check *claim* against the terms it is being looked up by.

    ``terms`` drives the search; it is also, by construction, a statement about
    the same sequence the claim is about, so the two can be held against each
    other for free. The rule is the lenient one a source's own formula line has
    to pass in :meth:`OeisEntry.recurrences`: the trailing windows must confirm,
    because a recurrence is routinely stated only for ``n`` past some initial
    segment.
    """
    if not terms or not isinstance(claim, RecurrenceClaim):
        return "not_checked"
    windows = len(terms) - claim.order
    if windows <= 0:
        return "not_checked"
    try:
        confirmations = claim.confirmations(terms, start=start)
    except (TypeError, ValueError, ZeroDivisionError):
        return "not_checked"
    return "holds" if confirmations >= min(_MIN_CONFIRMATIONS, windows) else "fails"
