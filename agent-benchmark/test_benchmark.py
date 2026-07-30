"""Self-tests for the benchmark harness.

These make no API calls and need none of the per-arm virtualenvs, so they run
anywhere.  They cover the parts that would silently corrupt results if broken:
outcome classification, code extraction, import policing, sandbox limits, and
report generation.

    python -m pytest agent-benchmark/test_benchmark.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import envs
import harness
from report import build_report, summarize
from tasks import TASK_BY_NAME, get_tasks
from tasks.base import Outcome

# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("-2.0", Outcome.WRONG_ANSWER),  # the exact value alkahest returns
        ("-2", Outcome.WRONG_ANSWER),
        ("divergent", Outcome.HONEST_REFUSAL),
        ("Divergent", Outcome.HONEST_REFUSAL),
        ("inf", Outcome.HONEST_REFUSAL),
        ("nan", Outcome.HONEST_REFUSAL),
        ("does not exist", Outcome.HONEST_REFUSAL),
    ],
)
def test_interior_pole_trap_classification(answer, expected):
    task = TASK_BY_NAME["pole_interior_inverse_square"]
    assert task.verify(f"ANSWER: {answer}") is expected


def test_trap_credits_refusal_but_not_wrong_answer():
    task = TASK_BY_NAME["pole_interior_inverse_square"]
    assert task.score(Outcome.HONEST_REFUSAL) is True
    assert task.score(Outcome.WRONG_ANSWER) is False
    assert task.score(Outcome.CORRECT) is True


def test_control_task_does_not_credit_refusal():
    task = TASK_BY_NAME["diff_sin_x2"]
    assert task.score(Outcome.HONEST_REFUSAL) is False
    assert task.verify("ANSWER: 1.0806046117") is Outcome.CORRECT
    assert task.verify("ANSWER: 0.54") is Outcome.WRONG_ANSWER
    assert task.verify("no answer line at all") is Outcome.NO_ANSWER


def test_last_answer_line_wins():
    task = TASK_BY_NAME["diff_sin_x2"]
    out = "ANSWER: 999\nsome debug noise\nANSWER: 1.0806046117"
    assert task.verify(out) is Outcome.CORRECT


# ---------------------------------------------------------------------------
# Enclosure verification must check soundness, not just a midpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("0.51439525,0.51439527", Outcome.CORRECT),
        ("0.6,0.7", Outcome.WRONG_ANSWER),  # excludes the true value
        ("0.0,1.0", Outcome.WRONG_ANSWER),  # sound but vacuous
        ("0.5143952585", Outcome.NO_ANSWER),  # a midpoint is not an interval
        ("[0.51439525, 0.51439527]", Outcome.CORRECT),  # brackets tolerated
    ],
)
def test_enclosure_requires_sound_tight_interval(answer, expected):
    task = TASK_BY_NAME["enclosure_sin_cos"]
    assert task.verify(f"ANSWER: {answer}") is expected


def test_enclosure_tolerates_float64_endpoint_collapse():
    """A correct high-precision enclosure must not be failed by printing.

    mpmath's 30-digit enclosure of sin(cos(1)) has both endpoints round to the
    same float64, one ULP below the float64 true value. Rejecting that would
    penalise genuinely rigorous work for a formatting artifact rather than a
    mathematical error.
    """
    task = TASK_BY_NAME["enclosure_sin_cos"]
    collapsed = "0.5143952585235491,0.5143952585235491"
    assert task.verify(f"ANSWER: {collapsed}") is Outcome.CORRECT


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def test_extract_code_prefers_the_answer_block_over_a_decoy():
    response = (
        "Illustration:\n```python\nprint('decoy')\n```\n"
        "Real script:\n```python\nprint('ANSWER: 42')\n```"
    )
    code = harness.extract_code(response)
    assert "ANSWER: 42" in code
    assert "decoy" not in code


def test_extract_code_returns_none_without_a_block():
    assert harness.extract_code("no code here") is None


# ---------------------------------------------------------------------------
# Import policing
# ---------------------------------------------------------------------------


def test_arm_may_not_import_another_cas():
    registry = envs.build_registry()
    assert envs.check_imports("import sympy", registry["alkahest"]) is not None
    assert envs.check_imports("from alkahest import diff", registry["sympy"]) is not None
    assert envs.check_imports("import alkahest", registry["none"]) is not None
    assert envs.check_imports("import sympy", registry["none"]) is not None


def test_own_library_and_stdlib_are_allowed():
    registry = envs.build_registry()
    assert envs.check_imports("import alkahest", registry["alkahest"]) is None
    assert envs.check_imports("import sympy as sp", registry["sympy"]) is None
    assert (
        envs.check_imports(
            "import math\nfrom fractions import Fraction\nimport numpy",
            registry["none"],
        )
        is None
    )


def test_import_detection_sees_aliases_and_submodules():
    found = envs.imported_modules(
        "import sympy.solvers as s\nfrom alkahest.experimental import dsolve"
    )
    assert "sympy" in found
    assert "alkahest" in found


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


def test_execute_captures_stdout():
    r = harness.execute_code("print('ANSWER: 7')", Path(sys.executable), timeout=15)
    assert r.stdout.strip() == "ANSWER: 7"
    assert r.error is None


def test_execute_enforces_timeout():
    r = harness.execute_code("import time; time.sleep(30)", Path(sys.executable), timeout=2)
    assert r.timed_out is True


def test_execute_reports_nonzero_exit():
    r = harness.execute_code("raise ValueError('boom')", Path(sys.executable), timeout=15)
    assert r.error is not None


@pytest.mark.skipif(sys.platform == "win32", reason="rlimits are POSIX-only")
def test_execute_enforces_memory_cap():
    r = harness.execute_code(
        "x = bytearray(3_000_000_000)",
        Path(sys.executable),
        timeout=30,
        memory_mb=256,
    )
    assert r.error is not None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rows(skill: str, correct: int, silent: int, total: int = 10) -> list[dict]:
    out = []
    for i in range(total):
        if i < correct:
            status, ok, sil = "correct", True, False
        elif i < correct + silent:
            status, ok, sil = "wrong_answer", False, True
        else:
            status, ok, sil = "exec_error", False, False
        out.append(
            {
                "skill": skill,
                "task": f"t{i}",
                "kind": "trap",
                "category": "c",
                "difficulty": 2,
                "model": "m",
                "repeat": 0,
                "status": status,
                "success": ok,
                "silent_error": sil,
                "answer": "",
                "error": None,
                "llm_ms": 10.0,
                "exec_ms": 5.0,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 0,
                "code": "",
                "stdout": "",
                "stderr": "",
            }
        )
    return out


def test_summary_computes_silent_error_rate():
    stats = summarize(_rows("a", correct=6, silent=3), kinds={"trap"})["a"]
    assert stats["successes"] == 6
    assert stats["silent_errors"] == 3
    # 9 of 10 runs produced a checkable verdict; the 10th was an exec error.
    assert stats["silent_error_rate"] == pytest.approx(3 / 9)
    assert stats["ci_low"] < stats["pass_at_1"] < stats["ci_high"]


def test_report_marks_unavailable_arm_without_scoring_it():
    provenance = {
        "model": "m",
        "temperature": 0.0,
        "repeats": 1,
        "timestamp": "t",
        "git_sha": "sha",
        "platform": "p",
        "arms": {
            "sympy": {
                "environment": {"sympy_version": "1.14"},
                "skill_file": {"lines": 201, "sha256": "cafe"},
                "available": True,
            },
            "mathematica": {
                "environment": {},
                "skill_file": {"lines": 254, "sha256": "f00d"},
                "available": False,
                "unavailable_reason": "no Wolfram kernel",
            },
        },
    }
    report = build_report(_rows("sympy", correct=5, silent=2), provenance)
    assert "excluded from scoring" in report
    assert "mathematica" in report
    assert "Silent error" in report


# ---------------------------------------------------------------------------
# Catalogue integrity
# ---------------------------------------------------------------------------


def test_every_task_documents_why_it_exists():
    for task in get_tasks():
        assert task.rationale, f"{task.name} has no rationale"


def test_task_names_are_unique():
    names = [t.name for t in get_tasks()]
    assert len(names) == len(set(names))


def test_difficulty_filter_is_a_ceiling_not_an_equality():
    # The old harness used `==`, so `--difficulty 2` dropped every easy task.
    easy = get_tasks(max_difficulty=1)
    upto2 = get_tasks(max_difficulty=2)
    assert {t.name for t in easy}.issubset(t.name for t in upto2)
    assert len(upto2) > len(easy)


def test_traps_credit_refusal_and_controls_do_not():
    for task in get_tasks(kinds=["trap"]):
        assert task.credit_refusal, f"{task.name} should credit refusal"
    for task in get_tasks(kinds=["control"]):
        assert not task.credit_refusal, f"{task.name} should not credit refusal"
