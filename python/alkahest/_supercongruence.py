"""Supercongruence sweeps over a P-recursive sequence.

A supercongruence is a claim like ``A(p−1) ≡ 1 (mod p⁴)`` about a holonomic
sequence, and the only way anyone has ever produced evidence for one is to
check it at every prime in a range. That loop has three parts: pick the index
and the expected value as functions of ``p``, get the residue, and decide what
the run means. :class:`alkahest.ModularRecurrence` does the middle part in
Rust; this module is the other two.

Why this is Python and not Rust
-------------------------------

Per ``CONTRIBUTING.md`` § *Rust vs Python*: the arithmetic is already in the
kernel, and what is left is composition of kernel calls, keyword defaults,
callables supplied by the caller, and the bookkeeping that turns a pile of
residues into a verdict. That is the Python column, points 1, 3 and 4. It is
also the part a researcher edits — ``index``, ``expect`` and ``modulus_scale``
change per conjecture — and edits to it should not need a recompile.

What a clean run means
----------------------

Nothing, and the result object says so. :attr:`CongruenceSweep.holds` is
``True`` when no counterexample turned up in the range tested, which is
*falsification failed*, not *theorem proved*. The one thing a sweep can settle
is the sharpness of the modulus, and it does:
:attr:`CongruenceSweep.valuations` is the histogram of ``v_p(LHS − RHS)`` and
:attr:`CongruenceSweep.sharp` is ``True`` when some prime achieved exactly the
claimed exponent, i.e. when ``p^(k+1)`` would be false.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .alkahest import HolonomicError as _HolonomicError
from .alkahest import ModularRecurrence
from .number_theory import isprime as _isprime

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

__all__ = ["CongruenceSweep", "supercongruence_sweep"]

#: Refusals that are a fact about one prime, not about the call. These are
#: recorded in :meth:`CongruenceSweep.skipped` and the sweep carries on;
#: everything else (a malformed recurrence, a bad index) propagates.
#:
#: ``E-HOLO-006`` is one of them because the two ways to earn it are not alike.
#: A composite base is a fact about the *call*, and the sweep rules that out
#: itself before evaluating anything (see :func:`supercongruence_sweep`), so
#: the only ``E-HOLO-006`` that can reach the loop is ``p**k`` past the
#: machine-word ceiling — the same "this prime is out of reach of this backend"
#: that ``E-HOLO-008`` is, and it gets the same treatment. Letting it propagate
#: destroyed every residue already computed and left the caller to work out the
#: per-``k`` cap ``int((2**62) ** (1 / (k + 1)))`` by hand, which is exactly the
#: accounting :meth:`CongruenceSweep.skipped` exists to do.
_PER_PRIME_REFUSALS = frozenset({"E-HOLO-006", "E-HOLO-007", "E-HOLO-008"})


class CongruenceSweep:
    """The outcome of :func:`alkahest.supercongruence_sweep`.

    Everything the sweep learned, including the parts that argue against the
    conjecture. The residues are kept so a caller can re-derive any of it.
    """

    __slots__ = (
        "_claimed",
        "_counterexamples",
        "_extra_precision",
        "_residues",
        "_skipped",
        "_valuations",
    )

    def __init__(
        self,
        *,
        residues: dict[int, int],
        counterexamples: list[tuple[int, int, int]],
        valuations: dict[int, int],
        skipped: list[tuple[int, str]],
        claimed: int,
        extra_precision: int,
    ):
        self._residues = residues
        self._counterexamples = counterexamples
        self._valuations = valuations
        self._skipped = skipped
        self._claimed = claimed
        self._extra_precision = extra_precision

    @property
    def holds(self) -> bool:
        """Whether every prime tested satisfied the congruence.

        **This is falsification failing, not a proof.** A ``True`` here says
        the claim survived :attr:`n_tested` primes and nothing more; the
        recurrence it was evaluated from is itself only as good as its
        certificate.
        """
        return not self._counterexamples

    @property
    def n_tested(self) -> int:
        """How many primes produced a residue."""
        return len(self._residues)

    @property
    def largest(self) -> int:
        """The largest prime tested, or ``0`` if none were."""
        return max(self._residues, default=0)

    @property
    def claimed_exponent(self) -> int:
        """The exponent ``k`` the congruence was checked at."""
        return self._claimed

    @property
    def sharp(self) -> bool:
        """Whether some prime achieved exactly ``v_p = k``.

        ``True`` means ``p^(k+1)`` is *false* at that prime, so the modulus in
        the conjecture is best possible and not an artefact of a cautious
        statement. ``False`` with a clean sweep is the interesting case: every
        prime did better than claimed, and the conjecture as stated is probably
        not the sharp one.

        Undecidable when ``extra_precision`` is ``0``, since then no residue
        can distinguish ``v_p = k`` from ``v_p > k``; it is ``False`` there.
        """
        return self._extra_precision > 0 and self._claimed in self._valuations

    @property
    def n_skipped(self) -> int:
        """Primes the evaluation refused, and therefore says nothing about."""
        return len(self._skipped)

    def residues(self) -> dict[int, int]:
        """``{p: (LHS − RHS) mod p**(k + extra_precision)}``.

        The residue of the *difference*, so ``0`` is the congruence holding.
        """
        return dict(self._residues)

    def counterexamples(self) -> list[tuple[int, int, int]]:
        """``[(p, residue, v_p)]`` for every prime where the claim failed.

        Empty for a clean run. A non-empty list is the only kind of result a
        sweep can produce that is a mathematical fact rather than evidence.
        """
        return list(self._counterexamples)

    def valuations(self) -> dict[int, int]:
        """Histogram of ``v_p(LHS − RHS)``, capped at ``k + extra_precision``.

        The key that :attr:`sharp` reads. A histogram concentrated well above
        ``k`` means the conjecture is understated.
        """
        return dict(self._valuations)

    def skipped(self) -> list[tuple[int, str]]:
        """``[(p, reason)]`` for primes the evaluation refused.

        A refusal is *undecided*, not *satisfied*: a sweep that silently
        dropped these would be reporting a range it never covered. The three
        causes are all "this backend cannot reach this prime": ``p**(k+extra)``
        past the machine-word ceiling of ``2**62`` (``E-HOLO-006``), a run of
        singular indices demanding more working precision than a machine-word
        modulus can hold (``E-HOLO-008``), and a sequence that is not
        ``p``-integral there (``E-HOLO-007``). A refusal about the *call* — a
        composite in *primes*, a malformed recurrence — is not skipped, it is
        raised.
        """
        return list(self._skipped)

    def __repr__(self) -> str:
        verdict = "holds" if self.holds else f"FAILS at {[c[0] for c in self._counterexamples]}"
        return (
            f"CongruenceSweep({verdict} for {self.n_tested} primes, "
            f"largest={self.largest}, claimed=p^{self._claimed}, "
            f"sharp={self.sharp}, skipped={self.n_skipped})"
        )


def _as_callable(value: Any, name: str) -> Callable[[int], int]:
    if callable(value):
        return value
    if isinstance(value, int):
        return lambda _p, _v=value: _v
    raise TypeError(f"{name} must be an int or a callable of p, got {type(value).__name__}")


def _require_prime(p: Any) -> None:
    """Refuse a composite in *primes* before any residue is computed.

    The kernel refuses it too, with the same ``E-HOLO-006``, but it does so
    from inside the loop where this module can no longer tell that refusal
    apart from "this prime is past the machine-word ceiling" — which is a fact
    about one prime and belongs in :meth:`CongruenceSweep.skipped`. Deciding
    the *call*-level half here is what lets the other half be skipped rather
    than fatal. The message is the kernel's, so the two cannot drift.
    """
    if isinstance(p, int) and p >= 2 and _isprime(p):
        return
    error = _HolonomicError(
        f"holonomic: unsupported modulus: {p} is not prime; the lifting "
        "argument this module rests on needs a prime power modulus, and v_p "
        "is not defined otherwise"
    )
    error.code = "E-HOLO-006"
    error.remediation = (
        "the modulus must be p**k with p prime, k >= 1 and p**k < 2**62; for a "
        "composite modulus, evaluate at each prime power and recombine by CRT"
    )
    raise error


def supercongruence_sweep(
    recurrence: ModularRecurrence,
    primes: Iterable[int],
    k: int,
    *,
    index: Callable[[int], int] | None = None,
    expect: Callable[[int], int] | int = 0,
    extra_precision: int = 1,
    max_counterexamples: int = 10,
) -> CongruenceSweep:
    """Check ``S(index(p)) ≡ expect(p) (mod p**k)`` at every prime in *primes*.

    This is the loop a supercongruence investigation runs, with the residue
    coming from :meth:`alkahest.ModularRecurrence.value_mod` rather than from
    big-integer arithmetic — so the cost per prime is ``O(index(p))``
    machine-word multiplications instead of ``O(index(p))`` operations on
    integers with ``Θ(index(p))`` digits.

    :param recurrence: the sequence, as a
        :class:`alkahest.ModularRecurrence`.
    :param primes: the primes to test. Checked for primality here, and a
        composite raises ``HolonomicError`` (``E-HOLO-006``) rather than being
        skipped, because a sweep that silently drops its inputs reports a range
        it did not cover. The check is up front so that the *other*
        ``E-HOLO-006`` — ``p**(k + extra_precision)`` past the machine-word
        ceiling, a fact about one prime rather than about the call — can be
        recorded in :meth:`CongruenceSweep.skipped` and the sweep continue.
    :param k: the claimed exponent. ``a(p-1) ≡ 1 (mod p**4)`` is ``k=4``.
    :param index: ``p -> n``, the index to evaluate at. Defaults to ``p - 1``,
        which is the shape of nearly every Apéry-like supercongruence.
    :param expect: ``p -> value``, or a constant. Defaults to ``0``.
    :param extra_precision: how many digits beyond ``k`` to compute, so that
        ``v_p(LHS − RHS)`` can be *measured* rather than merely bounded below
        by ``k``. This is what makes :attr:`CongruenceSweep.sharp` and the
        valuation histogram possible; ``0`` disables both. One is enough —
        a residue that is ``0`` mod ``p**k`` and non-zero mod ``p**(k+1)`` has
        ``v_p`` exactly ``k`` — and it is the default because every extra digit
        costs modulus headroom: the evaluation runs at ``p**(k + extra + loss)``
        and refuses past ``2**62``, so a generous ``extra_precision`` buys a
        finer histogram at the price of the larger primes in the range.
    :param max_counterexamples: stop after this many failures. A conjecture
        that fails at the first ten primes does not need the eleventh.

    >>> import alkahest as ak
    >>> apery = ak.ModularRecurrence(
    ...     [[1, 3, 3, 1], [-117, -231, -153, -34], [8, 12, 6, 1]], [1, 5]
    ... )
    >>> sweep = ak.supercongruence_sweep(
    ...     apery, [5, 7, 11, 13, 17, 19, 23, 29, 31], k=3, expect=1
    ... )
    >>> sweep.holds, sweep.n_tested, sweep.sharp
    (True, 9, True)

    ``sharp`` being ``True`` is the sweep earning its keep: some prime has
    ``v_p(A(p−1) − 1)`` exactly ``3``, so the mod-``p⁴`` version of the same
    statement is false and the ``p³`` in Beukers' theorem is best possible.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if extra_precision < 0:
        raise ValueError("extra_precision must not be negative")
    if max_counterexamples < 1:
        raise ValueError("max_counterexamples must be at least 1")
    index_of = index if index is not None else (lambda p: p - 1)
    expect_at = _as_callable(expect, "expect")

    precision = k + extra_precision
    residues: dict[int, int] = {}
    counterexamples: list[tuple[int, int, int]] = []
    valuations: dict[int, int] = {}
    skipped: list[tuple[int, str]] = []

    for p in primes:
        # The one `E-HOLO-006` that is a fact about the *call* rather than
        # about one prime, decided here rather than left to the evaluation, so
        # that the code can be skipped below without a list of composites
        # coming back `holds=True` over zero primes — the sweep lying about a
        # range it never covered.
        _require_prime(p)
        modulus = p**precision
        try:
            value = recurrence.value_mod(index_of(p), p, precision)
        except _HolonomicError as exc:
            # A refusal about *this prime* is recorded and reported; a refusal
            # about the call itself is re-raised.
            if getattr(exc, "code", None) not in _PER_PRIME_REFUSALS:
                raise
            skipped.append((p, str(exc)))
            continue
        residue = (value - expect_at(p)) % modulus
        residues[p] = residue
        v = _valuation(residue, p, precision)
        valuations[v] = valuations.get(v, 0) + 1
        if v < k:
            counterexamples.append((p, residue, v))
            if len(counterexamples) >= max_counterexamples:
                break

    return CongruenceSweep(
        residues=residues,
        counterexamples=counterexamples,
        valuations=valuations,
        skipped=skipped,
        claimed=k,
        extra_precision=extra_precision,
    )


def _valuation(residue: int, p: int, cap: int) -> int:
    """``v_p(residue)`` for a residue known mod ``p**cap``, saturating at *cap*.

    A return of *cap* means "at least *cap*", because a residue of ``0`` mod
    ``p**cap`` carries no more information than that. Callers treat it as a
    lower bound and nothing more — which is precisely why
    :attr:`CongruenceSweep.sharp` needs ``extra_precision > 0`` to say
    anything.
    """
    if residue == 0:
        return cap
    v = 0
    while v < cap and residue % p == 0:
        residue //= p
        v += 1
    return v
