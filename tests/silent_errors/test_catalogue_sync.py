"""Keep this corpus in sync with the agent-benchmark trap catalogue.

``agent-benchmark/tasks/catalogue.py`` is where new silent-error traps get
written first, because that is where someone notices a CAS handing an agent a
confident wrong answer. But the benchmark needs an LLM, so a trap that lives
only there is untested on every pull request — it drifts, gets fixed, gets
re-broken, and nobody finds out until the next manual benchmark run.

This module is the ratchet. Add a ``Kind.TRAP`` task upstream without a
library-level counterpart here and the build goes red with the task name and
what to do about it.

The mapping is deliberately many-to-many: one benchmark task may be covered by
several library cases (the interior-pole family), and one library case may cover
several tasks. All that is required is that every TRAP task is claimed by at
least one case via :attr:`Case.benchmark_tasks`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from contracts import BENCHMARK_OUTCOME, Outcome
from corpus import CASES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_DIR = _REPO_ROOT / "agent-benchmark"
_CATALOGUE = _BENCHMARK_DIR / "tasks" / "catalogue.py"


def _trap_names_by_import() -> set[str] | None:
    """Import the catalogue and read the TRAP task names off it.

    ``agent-benchmark/tasks`` imports nothing outside the standard library, so
    this normally works and gives us the authoritative list. Returns ``None`` if
    the import is not clean (a new dependency, a syntax error on another Python
    version) so the caller can fall back to source parsing.
    """
    added = False
    if str(_BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(_BENCHMARK_DIR))
        added = True
    try:
        from tasks.base import Kind  # type: ignore[import-not-found]
        from tasks.catalogue import ALL_TASKS  # type: ignore[import-not-found]

        return {t.name for t in ALL_TASKS if t.kind is Kind.TRAP}
    except Exception:
        return None
    finally:
        if added:
            sys.path.remove(str(_BENCHMARK_DIR))


def _trap_names_by_parsing() -> set[str]:
    """Fallback: read ``name=`` from every ``AgentTask(kind=Kind.TRAP, ...)``.

    Uses ``ast`` rather than a regex so it cannot be fooled by a task name that
    happens to appear inside a rationale string.
    """
    tree = ast.parse(_CATALOGUE.read_text(encoding="utf-8"), filename=str(_CATALOGUE))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "AgentTask"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        kind = kwargs.get("kind")
        is_trap = (
            isinstance(kind, ast.Attribute)
            and kind.attr == "TRAP"
            and isinstance(kind.value, ast.Name)
            and kind.value.id == "Kind"
        )
        name = kwargs.get("name")
        if is_trap and isinstance(name, ast.Constant) and isinstance(name.value, str):
            names.add(name.value)
    return names


def benchmark_trap_names() -> set[str]:
    if not _CATALOGUE.is_file():
        pytest.fail(
            f"{_CATALOGUE} is missing. The silent-error gate is defined against the "
            "agent-benchmark trap catalogue; if the benchmark moved, update this test "
            "rather than deleting it."
        )
    return _trap_names_by_import() or _trap_names_by_parsing()


def corpus_claimed_names() -> set[str]:
    return {name for case in CASES for name in case.benchmark_tasks}


def test_every_benchmark_trap_has_a_library_case() -> None:
    """A new ``Kind.TRAP`` task must come with a library-level counterpart.

    This is the anti-drift check. If it fails, add a case to
    ``tests/silent_errors/corpus.py`` that exercises the same mathematics
    through the alkahest API directly and list the task name in its
    ``benchmark_tasks`` tuple.
    """
    upstream = benchmark_trap_names()
    claimed = corpus_claimed_names()
    uncovered = sorted(upstream - claimed)
    assert not uncovered, (
        "agent-benchmark TRAP tasks with no library-level case in "
        f"tests/silent_errors/corpus.py: {uncovered}\n"
        "Each of these is a known silent-error shape that only gets exercised when "
        "someone runs the LLM benchmark by hand. Add a Case whose op calls the same "
        "operation directly and set benchmark_tasks=(<task name>,)."
    )


def test_no_case_claims_a_nonexistent_task() -> None:
    """Catch typos and tasks renamed or deleted upstream."""
    upstream = benchmark_trap_names()
    claimed = corpus_claimed_names()
    dangling = sorted(claimed - upstream)
    assert not dangling, (
        f"cases reference agent-benchmark tasks that no longer exist: {dangling}\n"
        "Either the task was renamed upstream (update benchmark_tasks) or it stopped "
        "being Kind.TRAP (drop the reference; keep the case)."
    )


def test_both_name_discovery_routes_agree() -> None:
    """The import route and the source-parsing fallback must see the same tasks.

    If they diverge, the fallback has silently gone stale and would stop
    detecting new traps the day the import breaks.
    """
    imported = _trap_names_by_import()
    if imported is None:
        pytest.skip("agent-benchmark catalogue is not importable in this environment")
    assert imported == _trap_names_by_parsing()


def test_outcome_taxonomy_matches_the_benchmark() -> None:
    """This gate's outcomes must stay a relabelling of the benchmark's, no more.

    The two vocabularies differ in exactly one name — ``silent_error`` here is
    ``wrong_answer`` there — and comparing a library-level rate with a benchmark
    rate is only meaningful while that stays true.
    """
    assert set(BENCHMARK_OUTCOME) == set(Outcome), "BENCHMARK_OUTCOME must be total over Outcome"

    added = False
    if str(_BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(_BENCHMARK_DIR))
        added = True
    try:
        from tasks.base import Outcome as BenchmarkOutcome  # type: ignore[import-not-found]
    except Exception:
        pytest.skip("agent-benchmark tasks package is not importable in this environment")
    finally:
        if added:
            sys.path.remove(str(_BENCHMARK_DIR))

    upstream_values = {o.value for o in BenchmarkOutcome}
    mapped = set(BENCHMARK_OUTCOME.values())
    assert mapped == upstream_values, (
        f"outcome vocabularies drifted: this gate maps to {sorted(mapped)}, "
        f"agent-benchmark defines {sorted(upstream_values)}"
    )


def test_trap_coverage_is_reported() -> None:
    """Emit the coverage mapping so a reviewer can see which case covers what."""
    upstream = sorted(benchmark_trap_names())
    lines = ["agent-benchmark TRAP task -> library cases"]
    for task in upstream:
        covering = sorted(c.id for c in CASES if task in c.benchmark_tasks)
        lines.append(f"  {task:<32} {', '.join(covering)}")
    print("\n".join(lines))
    assert upstream, "the benchmark catalogue reported no TRAP tasks at all"
