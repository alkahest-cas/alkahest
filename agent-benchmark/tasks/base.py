"""Task definitions and answer verification.

The old harness scored a run with two booleans (``ok``, ``answer_correct``),
which collapsed three very different failures into one bucket: the CAS was
wrong, the agent wrote broken code, and the agent misread a correct result all
looked identical.  That distinction is the whole point of the benchmark, so
verification returns an :class:`Outcome` instead.

The headline metric is deliberately **not** raw accuracy.  On easy problems every
mainstream CAS is correct, so accuracy saturates and measures nothing.  What
differentiates a CAS *for agent use* is how often it lets an agent state a
confident wrong answer — :data:`Outcome.WRONG_ANSWER` — versus surfacing a
refusal the agent can act on.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Outcome(str, Enum):
    """Result of verifying one agent answer."""

    CORRECT = "correct"
    #: A confident answer that is mathematically wrong.  The metric that matters.
    WRONG_ANSWER = "wrong_answer"
    #: The agent recognised it could not answer (non-elementary, divergent, …).
    #: Credited as success on trap tasks, where refusal *is* the right answer.
    HONEST_REFUSAL = "honest_refusal"
    #: Script ran but emitted no parseable ANSWER line.
    NO_ANSWER = "no_answer"


class Kind(str, Enum):
    """What a task is designed to measure."""

    #: Floor: any working CAS should pass.  Establishes the arms are functional.
    CONTROL = "control"
    #: A plausible-but-wrong result is available; measures silent-error rate.
    TRAP = "trap"
    #: Large enough that a slow CAS times out; measures headroom.
    SCALE = "scale"
    #: Requires a sound numeric enclosure, not just a close-looking float.
    RIGOR = "rigor"
    #: Requires emitting a machine-checkable proof artifact.
    CERTIFICATE = "certificate"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.*?)\s*$", re.MULTILINE)

#: Words an agent uses when it correctly declines.  Matched on the ANSWER line
#: only, so ordinary prose elsewhere in stdout cannot trigger a false refusal.
REFUSAL_TOKENS = (
    "nonelementary",
    "non-elementary",
    "no elementary",
    "divergent",
    "diverges",
    "undefined",
    "no real solution",
    "no_real_solution",
    "does not exist",
    "dne",
    "unsupported",
    "cannot",
    "refuse",
)


def answer_line(output: str) -> str | None:
    """Return the value after the last ``ANSWER:`` marker, or None."""
    matches = _ANSWER_RE.findall(output or "")
    return matches[-1].strip() if matches else None


def _is_refusal(ans: str) -> bool:
    low = ans.lower()
    return any(tok in low for tok in REFUSAL_TOKENS)


def _as_float(ans: str) -> float | None:
    try:
        return float(ans)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Verifier combinators
# ---------------------------------------------------------------------------


def numeric(expected: float, tol: float = 1e-4) -> Callable[[str], Outcome]:
    """Answer must be a float within *tol* of *expected*."""

    def check(output: str) -> Outcome:
        ans = answer_line(output)
        if ans is None:
            return Outcome.NO_ANSWER
        if _is_refusal(ans):
            return Outcome.HONEST_REFUSAL
        val = _as_float(ans)
        if val is None:
            return Outcome.NO_ANSWER
        if math.isnan(val) or math.isinf(val):
            return Outcome.WRONG_ANSWER
        return Outcome.CORRECT if abs(val - expected) <= tol else Outcome.WRONG_ANSWER

    return check


def exact_int(expected: int) -> Callable[[str], Outcome]:
    """Answer must be exactly the integer *expected*."""

    def check(output: str) -> Outcome:
        ans = answer_line(output)
        if ans is None:
            return Outcome.NO_ANSWER
        if _is_refusal(ans):
            return Outcome.HONEST_REFUSAL
        try:
            return Outcome.CORRECT if int(ans) == expected else Outcome.WRONG_ANSWER
        except (TypeError, ValueError):
            # A float that is integral in value still counts.
            val = _as_float(ans)
            if val is not None and abs(val - expected) < 1e-9:
                return Outcome.CORRECT
            return Outcome.NO_ANSWER

    return check


def keyword(*accepted: str) -> Callable[[str], Outcome]:
    """Answer must contain one of *accepted* (case-insensitive)."""

    def check(output: str) -> Outcome:
        ans = answer_line(output)
        if ans is None:
            return Outcome.NO_ANSWER
        low = ans.lower()
        if any(a.lower() in low for a in accepted):
            return Outcome.CORRECT
        if _is_refusal(ans):
            return Outcome.HONEST_REFUSAL
        return Outcome.WRONG_ANSWER

    return check


def refusal_or_value(expected: float | None = None, tol: float = 1e-4) -> Callable[[str], Outcome]:
    """Trap verifier: an honest refusal is success.

    Used where the mathematically correct response is "this does not exist" —
    divergent integrals, non-elementary antiderivatives, empty real solution
    sets.  If *expected* is given (e.g. a principal value or a one-sided limit),
    that numeric answer is also accepted as correct.  Anything else confident is
    a silent error.
    """

    def check(output: str) -> Outcome:
        ans = answer_line(output)
        if ans is None:
            return Outcome.NO_ANSWER
        if _is_refusal(ans):
            return Outcome.HONEST_REFUSAL
        val = _as_float(ans)
        if val is None:
            return Outcome.NO_ANSWER
        if math.isnan(val):
            # NaN is an implicit admission of failure, not a stated answer.
            return Outcome.HONEST_REFUSAL
        if math.isinf(val):
            return Outcome.HONEST_REFUSAL
        if expected is not None and abs(val - expected) <= tol:
            return Outcome.CORRECT
        return Outcome.WRONG_ANSWER

    return check


def enclosure(true_value: float, max_radius: float) -> Callable[[str], Outcome]:
    """Rigor verifier: the reported interval must *contain* the true value.

    The answer line must be ``lo,hi``.  Checking only a midpoint (as the old
    ``ball_sin_cos`` task did) is passed by any ordinary float computation and
    therefore measures nothing about rigour.  Here an interval that excludes the
    true value is a wrong answer even if its midpoint is close, and an interval
    so wide it is useless is also rejected.

    Containment is checked with a few-ULP slack.  A correct high-precision
    enclosure can collapse to a single float64 on *either* side of the true value
    when its endpoints are printed — mpmath's 30-digit enclosure of
    ``sin(cos(1))`` does exactly this — so a strict comparison would reject
    genuinely rigorous work for a printing artifact rather than a maths error.
    """
    # Slack scaled to the value's own magnitude, floored for values near zero.
    slack = max(abs(true_value), 1.0) * 8 * sys.float_info.epsilon

    def check(output: str) -> Outcome:
        ans = answer_line(output)
        if ans is None:
            return Outcome.NO_ANSWER
        if _is_refusal(ans):
            return Outcome.HONEST_REFUSAL
        parts = [p.strip() for p in ans.replace("[", "").replace("]", "").split(",")]
        if len(parts) != 2:
            return Outcome.NO_ANSWER
        lo, hi = _as_float(parts[0]), _as_float(parts[1])
        if lo is None or hi is None:
            return Outcome.NO_ANSWER
        if lo > hi:
            lo, hi = hi, lo
        if not (lo - slack <= true_value <= hi + slack):
            return Outcome.WRONG_ANSWER  # unsound enclosure
        if (hi - lo) > max_radius * 2:
            return Outcome.WRONG_ANSWER  # sound but vacuous
        return Outcome.CORRECT

    return check


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "Write a complete, self-contained Python script that solves the problem below.\n"
    "Your script must print exactly one line in the format:\n\n"
    "    ANSWER: <value>\n\n"
    "as its final output line.\n\n"
    "If the problem has no answer of the requested form — the quantity is "
    "undefined, the integral diverges, the antiderivative is not elementary, or "
    "there is no real solution — print the single word describing that instead, "
    "e.g. `ANSWER: divergent`, `ANSWER: nonelementary`, `ANSWER: no_real_solution`, "
    "`ANSWER: undefined`. A wrong numeric answer is worse than a correct refusal.\n\n"
    "Problem:\n"
)


@dataclass(frozen=True)
class AgentTask:
    """One benchmark problem."""

    name: str
    kind: Kind
    category: str
    difficulty: int  # 1 = easy, 2 = medium, 3 = hard
    problem: str
    verify: Callable[[str], Outcome]
    #: Expected answer, shown only with --debug.
    hint: str = ""
    #: Per-task execution budget.  Scale tasks need a longer one, and hitting the
    #: limit is itself a result rather than an error.
    timeout_s: int = 45
    #: Free-text note explaining what the task discriminates, used in the report.
    rationale: str = ""
    #: Extra guidance appended after the problem statement.
    instructions: str = ""

    @property
    def credit_refusal(self) -> bool:
        """Whether an honest refusal counts as success for this task."""
        return self.kind is Kind.TRAP

    @property
    def prompt(self) -> str:
        body = _PREAMBLE + self.problem
        if self.instructions:
            body += "\n\n" + self.instructions
        return body

    def score(self, outcome: Outcome) -> bool:
        """Whether *outcome* counts as a success for this task."""
        if outcome is Outcome.CORRECT:
            return True
        return self.credit_refusal and outcome is Outcome.HONEST_REFUSAL


@dataclass
class TaskSuite:
    """A named collection of tasks."""

    tasks: list[AgentTask] = field(default_factory=list)

    def by_name(self) -> dict[str, AgentTask]:
        return {t.name: t for t in self.tasks}
