"""The deterministic silent-error gate.

Runs every case in :mod:`corpus` exactly once, scores it against its declared
contract, writes a machine-readable summary, and fails if the silent-error count
is anything other than zero.

Why this exists as well as ``agent-benchmark/``: the benchmark measures the same
quantity through an LLM, which makes it a research instrument, not a gate — it
needs API keys, it is non-deterministic, and it costs money per run. This runs
in seconds with no network and no model, so it can block a merge. The two are
kept in sync by ``test_catalogue_sync.py``.

Reading the output
------------------
The summary is printed to stdout and written to
``target/silent-errors/summary.json`` (override with
``ALKAHEST_SILENT_ERROR_REPORT``). The line to look at in CI is::

    silent-error rate: 0.0% (0 / N)

Anything above zero means alkahest returned a confident wrong answer somewhere,
and the offending case ids are listed immediately below it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from contracts import BENCHMARK_OUTCOME, Outcome, Result, evaluate
from corpus import CASES

# ---------------------------------------------------------------------------
# Evaluate the corpus once, lazily, and share the results between the
# per-case tests and the summary test.  Re-running would double the cost and
# could report a different rate than the one the per-case tests saw.
# ---------------------------------------------------------------------------

_RESULTS: dict[str, Result] = {}


def _results() -> dict[str, Result]:
    if not _RESULTS:
        for case in CASES:
            _RESULTS[case.id] = evaluate(case)
    return _RESULTS


def _report_path() -> Path:
    override = os.environ.get("ALKAHEST_SILENT_ERROR_REPORT")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "target" / "silent-errors" / "summary.json"


def _case_params() -> list[pytest.ParameterSet]:
    params = []
    for case in CASES:
        marks = []
        if case.xfail is not None:
            # strict=True is load-bearing: when the bug is fixed the case flips
            # from xfail to an unexpected pass, which pytest reports as a
            # failure.  That failure is the signal to delete the marker and let
            # the case become an ordinary regression test.
            marks.append(pytest.mark.xfail(strict=True, reason=case.xfail))
        params.append(pytest.param(case, id=case.id, marks=marks))
    return params


@pytest.mark.parametrize("case", _case_params())
def test_case_honours_its_contract(case) -> None:
    """Each trap must satisfy its declared contract."""
    result = _results()[case.id]
    assert result.passed, "\n" + result.message


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _build_summary() -> dict:
    results = _results()

    by_outcome: dict[str, int] = {o.value: 0 for o in Outcome}
    by_subsystem: dict[str, dict[str, int]] = {}
    silent_errors: list[dict[str, str]] = []
    known_silent_errors: list[dict[str, str]] = []
    unexpected_failures: list[dict[str, str]] = []

    for case in CASES:
        result = results[case.id]
        outcome = result.verdict.outcome
        by_outcome[outcome.value] += 1

        bucket = by_subsystem.setdefault(
            case.subsystem, {o.value: 0 for o in Outcome} | {"cases": 0, "known_broken": 0}
        )
        bucket[outcome.value] += 1
        bucket["cases"] += 1
        if case.xfail is not None:
            bucket["known_broken"] += 1

        entry = {
            "id": case.id,
            "subsystem": case.subsystem,
            "statement": case.statement,
            "contract": case.contract.describe(),
            "observed": result.answer.describe(),
            "reason": result.verdict.reason
            if result.verification_error is None
            else f"{result.verdict.reason}; {result.verification_error}",
        }
        if outcome is Outcome.SILENT_ERROR:
            if case.xfail is not None:
                known_silent_errors.append(entry | {"bug": case.xfail})
            else:
                silent_errors.append(entry)
        elif not result.passed and case.xfail is None:
            unexpected_failures.append(entry)

    # The headline rate excludes cases already marked as known bugs, so it
    # tracks *regressions*.  The known set is reported separately and loudly:
    # it is the fix queue, not an excuse.
    scored = len(CASES) - sum(1 for c in CASES if c.xfail is not None)
    rate = (len(silent_errors) / scored) if scored else 0.0

    return {
        "schema": "alkahest.silent-error-gate/1",
        "total_cases": len(CASES),
        "scored_cases": scored,
        "known_broken_cases": len(CASES) - scored,
        "silent_error_count": len(silent_errors),
        "silent_error_rate": rate,
        "known_silent_error_count": len(known_silent_errors),
        "by_outcome": by_outcome,
        "benchmark_outcome_names": {k.value: v for k, v in BENCHMARK_OUTCOME.items()},
        "by_subsystem": dict(sorted(by_subsystem.items())),
        "silent_errors": silent_errors,
        "known_silent_errors": known_silent_errors,
        "unexpected_failures": unexpected_failures,
    }


def _format_summary(summary: dict) -> str:
    lines = [
        "",
        "=" * 72,
        "SILENT-ERROR GATE",
        "=" * 72,
        f"cases            : {summary['total_cases']} "
        f"({summary['scored_cases']} scored, {summary['known_broken_cases']} known-broken)",
        "silent-error rate: "
        f"{summary['silent_error_rate'] * 100:.1f}% "
        f"({summary['silent_error_count']} / {summary['scored_cases']})",
        "",
        "outcomes:",
    ]
    for name, count in summary["by_outcome"].items():
        lines.append(f"  {name:<16} {count:>4}")

    lines += ["", f"{'subsystem':<28}{'cases':>7}{'ok':>7}{'refusal':>9}{'silent':>8}{'known':>7}"]
    for subsystem, counts in summary["by_subsystem"].items():
        lines.append(
            f"  {subsystem:<26}{counts['cases']:>7}{counts['correct']:>7}"
            f"{counts['honest_refusal']:>9}{counts['silent_error']:>8}{counts['known_broken']:>7}"
        )

    if summary["known_silent_errors"]:
        lines += ["", "KNOWN silent errors (strict xfail — fix queue):"]
        for entry in summary["known_silent_errors"]:
            lines.append(f"  - {entry['id']}: {entry['reason']}")

    if summary["silent_errors"]:
        lines += ["", "!! NEW silent errors (gate failure):"]
        for entry in summary["silent_errors"]:
            lines.append(f"  - {entry['id']}: {entry['reason']}")

    if summary["unexpected_failures"]:
        lines += ["", "Contract failures that are not silent errors:"]
        for entry in summary["unexpected_failures"]:
            lines.append(f"  - {entry['id']}: {entry['reason']}")

    lines.append("=" * 72)
    return "\n".join(lines)


#: Rendered summary, picked up by ``pytest_terminal_summary`` in this package's
#: conftest so the numbers land in the CI log even under pytest's default output
#: capture (no ``-s`` needed, and it prints whether the gate passed or failed).
SUMMARY_TEXT: str = ""


@pytest.fixture(scope="module")
def summary() -> dict:
    global SUMMARY_TEXT
    built = _build_summary()
    path = _report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUMMARY_TEXT = _format_summary(built) + f"\nmachine-readable summary: {path}"
    return built


def test_silent_error_count_is_zero(summary: dict) -> None:
    """The gate. Zero silent errors outside the known-broken set, always.

    A *new* silent error means alkahest started returning a confident wrong
    answer where it previously refused or was right. In an autoresearch loop
    that is not one wrong answer — it is a false lemma that every downstream
    derivation inherits, and one the loop's own consistency checks will confirm.
    """
    offenders = "\n".join(f"  - {e['id']}: {e['reason']}" for e in summary["silent_errors"])
    assert summary["silent_error_count"] == 0, (
        f"\n{summary['silent_error_count']} NEW silent error(s):\n{offenders}\n"
        "A silent error is a confident, plausible, wrong answer returned with no "
        "exception and no verification flag. Fix the underlying bug, or — if it "
        "cannot be fixed now — mark the case xfail(strict=True) with the bug named, "
        "which keeps it counted in known_silent_errors instead of hiding it."
    )


def test_no_case_answers_nothing(summary: dict) -> None:
    """``no_answer`` means the harness broke, not that alkahest refused."""
    broken = [
        e for e in summary["unexpected_failures"] if e["reason"].startswith("call did not answer")
    ]
    assert not broken, (
        "cases failed with an exception outside the refusal set — this is a corpus "
        f"bug, not an alkahest result:\n{json.dumps(broken, indent=2)}"
    )


def test_summary_is_written(summary: dict) -> None:
    """The machine-readable artifact exists and round-trips."""
    path = _report_path()
    assert path.is_file(), f"summary was not written to {path}"
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["schema"] == "alkahest.silent-error-gate/1"
    assert reloaded["total_cases"] == len(CASES)


def test_every_case_declares_a_source() -> None:
    """No case may assert a value that was read off alkahest's own output.

    ``verified_by`` has to name an independent derivation. A corpus whose
    expectations were harvested from the library it is testing measures nothing
    but self-consistency.
    """
    missing = [c.id for c in CASES if len(c.verified_by.strip()) < 12]
    assert not missing, f"cases with no meaningful `verified_by`: {missing}"


def test_known_broken_cases_name_a_bug() -> None:
    """An ``xfail`` must say what is broken, not just that something is."""
    vague = [c.id for c in CASES if c.xfail is not None and len(c.xfail.strip()) < 40]
    assert not vague, f"xfail cases with no bug description: {vague}"
