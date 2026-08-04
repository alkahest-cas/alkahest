"""Contract vocabulary for the deterministic silent-error gate.

A **silent error** is a confident, plausible, mathematically wrong answer
returned with no exception, no ``NaN``, and no verification flag that would let
a caller tell it apart from a correct result.  For an autonomous research loop
this is the worst possible failure: a refusal costs one dead branch, a silent
error poisons every downstream claim derived from it, and the loop's own
consistency checks will happily confirm the poisoned branch.

``agent-benchmark/`` already measures this rate, but it needs an LLM and a
network round-trip per task, so it cannot gate CI.  This package measures the
same quantity at the **library** level: no agent, no model, no network — just
alkahest calls with declared contracts, so it can run on every pull request.

Vocabulary
----------
Every case declares one contract:

``Raises(code)``
    The call must raise an alkahest error carrying exactly the stable
    ``E-SUBSYSTEM-NNN`` *code*.  Refusing with a *different* code is a contract
    failure but **not** a silent error — the caller still knows it was refused.

``Returns(value)``
    The call must produce *value* (exactly for ints/bools/strings, within
    ``tol`` for floats).  A different value is a silent error.  A refusal here
    is a coverage regression: it fails the gate, but is classified
    ``honest_refusal``, not ``silent_error``.

``RefusesOr(value)``
    Either refusing *or* returning *value* is acceptable — these are the traps
    where "this does not exist" and "here is the principal value / one-sided
    limit" are both defensible answers.  Any *other* confident finite answer is
    a silent error.  ``RefusesOr()`` with no value means no finite answer is
    acceptable at all (divergent integral, limit that does not exist).

Outcomes mirror :class:`agent_benchmark.tasks.base.Outcome` one-for-one so the
two vocabularies stay comparable; :data:`BENCHMARK_OUTCOME` is the explicit map
and ``tests/silent_errors/test_catalogue_sync.py`` asserts it stays total.

What counts as a refusal
------------------------
Matching ``refusal_or_value`` in ``agent-benchmark/tasks/base.py``, a refusal is
any of:

* an ``AlkahestError`` subclass (the honest, coded path);
* a returned expression that cannot be reduced to a number — alkahest raises
  ``ValueError`` from ``eval_expr`` for unbound/undefined nodes such as ``∞``,
  ``0^-1`` or a bare ``O(x^n)`` remainder;
* ``NaN`` or ``±inf``, which are an implicit admission of failure rather than a
  stated answer.

These are *weak* refusals — a caller has to look at the value to notice — so
cases that only pass because of them say so in their ``note``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import alkahest as ak

# ---------------------------------------------------------------------------
# Outcome taxonomy — kept in lockstep with agent-benchmark
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    """Result of checking one case against its contract."""

    CORRECT = "correct"
    #: A confident answer that is mathematically wrong.  The metric that matters.
    #: Named ``wrong_answer`` on the agent-benchmark side.
    SILENT_ERROR = "silent_error"
    #: alkahest declined — raised a coded error, or returned something that does
    #: not reduce to a finite number.  Always a *safe* outcome, even when the
    #: contract wanted a value.
    HONEST_REFUSAL = "honest_refusal"
    #: The call blew up in a way that is neither an answer nor an alkahest
    #: refusal (harness bug, unexpected Python exception, wrong return type).
    NO_ANSWER = "no_answer"


#: Explicit mapping onto ``agent_benchmark.tasks.base.Outcome`` values.  The two
#: enums differ in exactly one name (``silent_error`` vs ``wrong_answer``); this
#: map is asserted total by the catalogue-sync test.
BENCHMARK_OUTCOME: dict[Outcome, str] = {
    Outcome.CORRECT: "correct",
    Outcome.SILENT_ERROR: "wrong_answer",
    Outcome.HONEST_REFUSAL: "honest_refusal",
    Outcome.NO_ANSWER: "no_answer",
}

#: Verification statuses in increasing strength.  ``Case.verification_floor``
#: names the weakest status a case is allowed to report.
VERIFICATION_ORDER: tuple[str, ...] = (
    "unverified",
    "numerically_checked",
    "certificate_available",
    "exactly_verified",
    "externally_verified",
)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class _Nothing:
    """Sentinel: ``RefusesOr()`` accepts no finite value at all."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "NOTHING"


NOTHING = _Nothing()


@dataclass(frozen=True)
class Raises:
    """Must raise an alkahest error with this exact stable code."""

    code: str

    def describe(self) -> str:
        return f"RAISES({self.code})"


@dataclass(frozen=True)
class Returns:
    """Must return this value (``tol`` applies to floats)."""

    value: Any
    tol: float = 1e-9

    def describe(self) -> str:
        return f"RETURNS({self.value!r})"


@dataclass(frozen=True)
class RefusesOr:
    """Refusing is fine; so is this specific value.  Anything else is a lie."""

    value: Any = NOTHING
    tol: float = 1e-9

    def describe(self) -> str:
        if self.value is NOTHING:
            return "REFUSES_OR(<no finite value>)"
        return f"REFUSES_OR({self.value!r})"


Contract = Raises | Returns | RefusesOr


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measured:
    """An answer plus the ``DerivedResult.verification`` dict that produced it.

    Cases whose op naturally holds a :class:`~alkahest.DerivedResult` return one
    of these so the runner can check ``verification_floor`` without re-running
    the computation.
    """

    answer: Any
    verification: dict[str, Any] | None = None


@dataclass(frozen=True)
class Case:
    """One declarative trap."""

    #: Stable identifier; also the pytest parameter id.  Never reuse one.
    id: str
    #: Subsystem bucket for the per-subsystem breakdown in the summary.
    subsystem: str
    #: Human-readable statement of the mathematics being tested.
    statement: str
    #: Zero-argument callable returning a plain answer (float / int / bool /
    #: str / tuple / list) or a :class:`Measured`.  It may raise.
    op: Callable[[], Any]
    contract: Contract
    #: Where the expected value came from.  Every case must name a source that
    #: is not alkahest itself — a hand derivation, a textbook fact, or an
    #: independent library.  Cases whose "expected" value was read off
    #: alkahest's own output prove nothing.
    verified_by: str
    #: Names of ``Kind.TRAP`` tasks in ``agent-benchmark/tasks/catalogue.py``
    #: that this case is the library-level counterpart of.
    benchmark_tasks: tuple[str, ...] = ()
    #: Weakest acceptable ``DerivedResult.verification["status"]``.  Requires the
    #: op to return a :class:`Measured`.
    verification_floor: str | None = None
    #: Set to a bug description to mark the case ``xfail(strict=True)``: alkahest
    #: genuinely fails this today.  Never delete a failing case — an absent case
    #: cannot catch the fix or a later re-regression.
    xfail: str | None = None
    #: Free-text caveat, e.g. "passes only via a weak refusal".
    note: str = ""
    #: Extra tags for filtering / reporting.
    tags: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """What actually came back."""

    #: ``"value"``, ``"refusal"`` or ``"no_answer"``.
    kind: str
    value: Any = None
    code: str | None = None
    detail: str = ""
    verification: dict[str, Any] | None = None

    def describe(self) -> str:
        if self.kind == "refusal":
            return f"refused[{self.code or 'uncoded'}]: {self.detail}"
        if self.kind == "no_answer":
            return f"no answer: {self.detail}"
        return f"value {self.value!r}"


#: Exceptions alkahest raises when it declines to produce a number.  ``ValueError``
#: is what ``eval_expr`` raises for an expression containing ``∞``, ``0^-1`` or a
#: bare ``O(x^n)`` remainder; ``OverflowError``/``ZeroDivisionError`` are the
#: float-layer equivalents.  Anything outside this tuple is a harness bug and is
#: reported as ``no_answer`` rather than being silently forgiven.
_REFUSAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ak.AlkahestError,
    ValueError,
    OverflowError,
    ZeroDivisionError,
)


def probe(op: Callable[[], Any]) -> Answer:
    """Run *op* and normalise whatever happens into an :class:`Answer`."""
    try:
        raw = op()
    except _REFUSAL_EXCEPTIONS as exc:
        return Answer(
            kind="refusal",
            code=getattr(exc, "code", None),
            detail=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}",
        )
    except Exception as exc:
        return Answer(kind="no_answer", detail=f"{type(exc).__name__}: {exc}")

    verification = None
    if isinstance(raw, Measured):
        verification = raw.verification
        raw = raw.answer

    if isinstance(raw, float) and not math.isfinite(raw):
        # NaN / ±inf are an implicit admission of failure, not a stated answer —
        # same call agent-benchmark's refusal_or_value() makes.
        return Answer(kind="refusal", detail=f"non-finite {raw!r}", verification=verification)

    return Answer(kind="value", value=raw, verification=verification)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _values_match(got: Any, want: Any, tol: float) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(float(got), float(want), rel_tol=tol, abs_tol=tol)
    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple)):
        return len(got) == len(want) and all(_values_match(g, w, tol) for g, w in zip(got, want))
    return got == want


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    passed: bool
    reason: str


def classify(contract: Contract, answer: Answer) -> Verdict:
    """Score *answer* against *contract*."""
    if answer.kind == "no_answer":
        return Verdict(Outcome.NO_ANSWER, False, f"call did not answer — {answer.detail}")

    if isinstance(contract, Raises):
        if answer.kind == "refusal":
            if answer.code == contract.code:
                return Verdict(Outcome.CORRECT, True, f"raised {contract.code} as required")
            return Verdict(
                Outcome.HONEST_REFUSAL,
                False,
                f"refused, but with {answer.code or 'no code'} instead of {contract.code}",
            )
        return Verdict(
            Outcome.SILENT_ERROR,
            False,
            f"SILENT ERROR: expected {contract.code}, got a confident {answer.value!r}",
        )

    if isinstance(contract, Returns):
        if answer.kind == "refusal":
            return Verdict(
                Outcome.HONEST_REFUSAL,
                False,
                f"refused ({answer.describe()}) where {contract.value!r} is computable",
            )
        if _values_match(answer.value, contract.value, contract.tol):
            return Verdict(Outcome.CORRECT, True, f"returned {contract.value!r}")
        return Verdict(
            Outcome.SILENT_ERROR,
            False,
            f"SILENT ERROR: returned {answer.value!r}, correct answer is {contract.value!r}",
        )

    if isinstance(contract, RefusesOr):
        if answer.kind == "refusal":
            return Verdict(Outcome.HONEST_REFUSAL, True, f"refused — {answer.detail}")
        if contract.value is not NOTHING and _values_match(
            answer.value, contract.value, contract.tol
        ):
            return Verdict(Outcome.CORRECT, True, f"returned the acceptable {contract.value!r}")
        expected = (
            "no finite value" if contract.value is NOTHING else f"refusal or {contract.value!r}"
        )
        return Verdict(
            Outcome.SILENT_ERROR,
            False,
            f"SILENT ERROR: returned a confident {answer.value!r}; expected {expected}",
        )

    raise TypeError(f"unknown contract type: {contract!r}")  # pragma: no cover


def check_verification_floor(case: Case, answer: Answer) -> str | None:
    """Return a failure message if the case's verification floor is not met."""
    if case.verification_floor is None:
        return None
    if answer.kind != "value":
        return None  # a refusal carries no verification metadata to check
    if answer.verification is None:
        return (
            f"verification_floor={case.verification_floor} declared but the op "
            "did not return a Measured(...) carrying verification metadata"
        )
    status = answer.verification.get("status")
    if status not in VERIFICATION_ORDER:
        return f"unknown verification status {status!r} (known: {VERIFICATION_ORDER})"
    if VERIFICATION_ORDER.index(status) < VERIFICATION_ORDER.index(case.verification_floor):
        return (
            f"verification status {status!r} is weaker than the declared floor "
            f"{case.verification_floor!r}"
        )
    return None


@dataclass(frozen=True)
class Result:
    """One evaluated case."""

    case: Case
    answer: Answer
    verdict: Verdict
    verification_error: str | None

    @property
    def passed(self) -> bool:
        return self.verdict.passed and self.verification_error is None

    @property
    def message(self) -> str:
        parts = [f"[{self.case.id}] {self.case.statement}"]
        parts.append(f"  contract : {self.case.contract.describe()}")
        parts.append(f"  observed : {self.answer.describe()}")
        parts.append(f"  outcome  : {self.verdict.outcome.value} — {self.verdict.reason}")
        if self.verification_error:
            parts.append(f"  verify   : {self.verification_error}")
        parts.append(f"  source   : {self.case.verified_by}")
        return "\n".join(parts)


def evaluate(case: Case) -> Result:
    """Run one case end to end."""
    answer = probe(case.op)
    verdict = classify(case.contract, answer)
    verification_error = check_verification_floor(case, answer)
    return Result(case=case, answer=answer, verdict=verdict, verification_error=verification_error)
