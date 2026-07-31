"""Report generation.

Three things this reports that the old one did not, and why they matter:

* **Confidence intervals.**  With one sample per cell there are no error bars, so
  15/17 vs 14/17 reads as a difference when it is noise.  Runs are repeated and
  pass@1 is given with a 95% interval.
* **Silent-error rate.**  The share of attempts that produced a confident wrong
  answer.  This is the metric that actually distinguishes CAS libraries for agent
  use; raw accuracy saturates on easy problems.
* **Separated token accounting.**  Prompt and completion tokens are reported
  apart, because skill guides differ several-fold in length and a combined total
  would mostly rank documentation size.

Capability tasks (Lean certificates) are reported in their own table and excluded
from the headline accuracy, since only one arm can attempt them at all.  Folding
them into the average would overstate the difference.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from tasks.base import Kind

if TYPE_CHECKING:
    from collections.abc import Iterable

# Statuses that mean the agent produced a checkable verdict.
_VERDICT_STATUSES = {"correct", "wrong_answer", "honest_refusal", "no_answer"}


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sensibly at 0% and 100%, unlike normal."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def summarize(results: list[dict], *, kinds: Iterable[str] | None = None) -> dict:
    """Aggregate per-skill statistics, optionally restricted to certain kinds."""
    if kinds is not None:
        kinds = set(kinds)
        results = [r for r in results if r["kind"] in kinds]

    by_skill: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_skill[r["skill"]].append(r)

    summary: dict[str, Any] = {}
    for skill, runs in by_skill.items():
        attempts = len(runs)
        successes = sum(1 for r in runs if r["success"])
        verdicts = [r for r in runs if r["status"] in _VERDICT_STATUSES]
        silent = sum(1 for r in runs if r.get("silent_error"))
        lo, hi = _wilson(successes, attempts)

        # pass@k: a task counts if any repeat succeeded.
        per_task: dict[str, list[bool]] = defaultdict(list)
        for r in runs:
            per_task[r["task"]].append(bool(r["success"]))
        pass_at_k = sum(1 for v in per_task.values() if any(v)) / len(per_task) if per_task else 0.0

        summary[skill] = {
            "attempts": attempts,
            "tasks": len(per_task),
            "successes": successes,
            "pass_at_1": successes / attempts if attempts else 0.0,
            "ci_low": lo,
            "ci_high": hi,
            "pass_at_k": pass_at_k,
            "silent_errors": silent,
            "silent_error_rate": silent / len(verdicts) if verdicts else 0.0,
            "verdicts": len(verdicts),
            "timeouts": sum(1 for r in runs if r["status"] == "timeout"),
            "exec_errors": sum(1 for r in runs if r["status"] == "exec_error"),
            "no_code": sum(1 for r in runs if r["status"] == "no_code"),
            "wrong_library": sum(1 for r in runs if r["status"] == "wrong_library"),
            "llm_errors": sum(1 for r in runs if r["status"] == "llm_error"),
            "refusals": sum(1 for r in runs if r["status"] == "honest_refusal"),
            "prompt_tokens": sum(r["prompt_tokens"] for r in runs),
            "completion_tokens": sum(r["completion_tokens"] for r in runs),
            "cached_tokens": sum(r.get("cached_tokens", 0) for r in runs),
            "median_exec_ms": _median([r["exec_ms"] for r in runs if r["exec_ms"]]),
            "median_llm_ms": _median([r["llm_ms"] for r in runs if r["llm_ms"]]),
        }
    return summary


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def build_report(results: list[dict], provenance: dict) -> str:
    if not results:
        return "No results.\n"

    lines: list[str] = ["# Agent Benchmark Report", ""]

    # ── Provenance ──────────────────────────────────────────────────────────
    lines += [
        "## Run provenance",
        "",
        f"- **Model:** `{provenance.get('model')}`",
        f"- **Temperature:** {provenance.get('temperature')}",
        f"- **Repeats per task:** {provenance.get('repeats')}",
        f"- **Timestamp:** {provenance.get('timestamp')}",
        f"- **Repo commit:** `{provenance.get('git_sha')}`",
        f"- **Platform:** {provenance.get('platform')}",
        "",
        "| Arm | Library build | Skill guide | Available |",
        "|---|---|---|---|",
    ]
    for name, arm in provenance.get("arms", {}).items():
        env = arm.get("environment", {})
        version = (
            env.get("alkahest_version")
            or env.get("sympy_version")
            or env.get("wolframclient_version")
            or env.get("numpy_version")
            or "—"
        )
        digest = arm.get("skill_file", {})
        guide = f"{digest.get('lines', '?')} lines / `{digest.get('sha256', '?')}`"
        avail = (
            "yes" if arm.get("available") else f"**no** — {arm.get('unavailable_reason', '')[:60]}"
        )
        lines.append(f"| {name} | {version} | {guide} | {avail} |")
    lines.append("")

    unavailable = [n for n, a in provenance.get("arms", {}).items() if not a.get("available")]
    if unavailable:
        lines += [
            f"> Arms excluded from scoring (environment unavailable): "
            f"**{', '.join(unavailable)}**.  They are not counted as failures.",
            "",
        ]

    # ── Headline ────────────────────────────────────────────────────────────
    scored_kinds = {Kind.CONTROL.value, Kind.TRAP.value, Kind.SCALE.value, Kind.RIGOR.value}
    head = summarize(results, kinds=scored_kinds)

    lines += [
        "## Headline: accuracy and silent-error rate",
        "",
        "`pass@1` is the per-attempt success rate with a 95% Wilson interval. "
        "**Silent error** is the share of checkable attempts that produced a "
        "confident but mathematically wrong answer — the number that matters for "
        "agent use, since a refusal an agent can see is recoverable and a wrong "
        "number is not.",
        "",
        "| Arm | Tasks | Attempts | pass@1 (95% CI) | pass@k | Silent error | Refusals |",
        "|---|---|---|---|---|---|---|",
    ]
    for skill, s in sorted(head.items(), key=lambda kv: -kv[1]["pass_at_1"]):
        ci = f"{_pct(s['pass_at_1'])} ({_pct(s['ci_low'])}–{_pct(s['ci_high'])})"
        lines.append(
            f"| {skill} | {s['tasks']} | {s['attempts']} | {ci} | "
            f"{_pct(s['pass_at_k'])} | {_pct(s['silent_error_rate'])} "
            f"({s['silent_errors']}) | {s['refusals']} |"
        )
    lines.append("")

    # ── Per-kind breakdown ──────────────────────────────────────────────────
    lines += ["## By task kind", ""]
    kinds_present = [k for k in Kind if any(r["kind"] == k.value for r in results)]
    header = "| Arm |" + "".join(f" {k.value} |" for k in kinds_present)
    lines += [header, "|---|" + "---|" * len(kinds_present)]
    skills = sorted({r["skill"] for r in results})
    for skill in skills:
        row = [skill]
        for k in kinds_present:
            sub = summarize([r for r in results if r["skill"] == skill], kinds={k.value})
            s = sub.get(skill)
            row.append(_pct(s["pass_at_1"]) if s else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Failure taxonomy ────────────────────────────────────────────────────
    lines += [
        "## Failure taxonomy",
        "",
        "| Arm | Wrong answer | No answer | Exec error | Timeout | No code "
        "| Wrong library | LLM error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for skill, s in sorted(head.items()):
        no_answer = sum(
            1
            for r in results
            if r["skill"] == skill and r["kind"] in scored_kinds and r["status"] == "no_answer"
        )
        lines.append(
            f"| {skill} | {s['silent_errors']} | {no_answer} | {s['exec_errors']} | "
            f"{s['timeouts']} | {s['no_code']} | {s['wrong_library']} | {s['llm_errors']} |"
        )
    lines.append("")
    lines += [
        "> `wrong library` counts attempts that tried to import a CAS the arm was "
        "not given.  Execution happens in a per-arm virtualenv, so these cannot "
        "silently succeed.",
        "",
    ]

    # ── Capability tasks ────────────────────────────────────────────────────
    cert_results = [r for r in results if r["kind"] == Kind.CERTIFICATE.value]
    if cert_results:
        lines += [
            "## Capability tasks (not in headline accuracy)",
            "",
            "These require emitting a machine-checkable proof artifact.  Arms "
            "without that capability are expected to fail or refuse; the table is "
            "a capability matrix, not a fair accuracy comparison, and is excluded "
            "from the headline numbers above.",
            "",
            "| Arm | Attempted | Succeeded |",
            "|---|---|---|",
        ]
        cert = summarize(results, kinds={Kind.CERTIFICATE.value})
        for skill, s in sorted(cert.items()):
            lines.append(f"| {skill} | {s['attempts']} | {s['successes']} |")
        lines.append("")

    # ── Cost and latency ────────────────────────────────────────────────────
    lines += [
        "## Cost and latency",
        "",
        "Prompt tokens are dominated by the skill guide, so they track "
        "documentation length rather than task difficulty — compare completion "
        "tokens for reasoning cost. `exec ms` excludes model latency.",
        "",
        "| Arm | Prompt tok | Cached tok | Completion tok | Median exec ms | Median LLM ms |",
        "|---|---|---|---|---|---|",
    ]
    for skill, s in sorted(head.items()):
        lines.append(
            f"| {skill} | {s['prompt_tokens']:,} | {s['cached_tokens']:,} | "
            f"{s['completion_tokens']:,} | {s['median_exec_ms']:.0f} | "
            f"{s['median_llm_ms']:.0f} |"
        )
    lines.append("")

    # ── Full grid ───────────────────────────────────────────────────────────
    lines += ["## Per-task results", ""]
    task_names = sorted({r["task"] for r in results})
    lines.append("| Task | Kind | " + " | ".join(skills) + " |")
    lines.append("|---|---|" + "---|" * len(skills))
    for task_name in task_names:
        first = next(r for r in results if r["task"] == task_name)
        row = [f"`{task_name}`", first["kind"]]
        for skill in skills:
            cell = [r for r in results if r["task"] == task_name and r["skill"] == skill]
            if not cell:
                row.append("—")
                continue
            wins = sum(1 for r in cell if r["success"])
            silent = sum(1 for r in cell if r.get("silent_error"))
            mark = f"{wins}/{len(cell)}"
            if silent:
                mark += f" ⚠{silent}"
            row.append(mark)
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Cells show successes / attempts; ⚠n marks n confidently wrong answers.",
        "",
    ]

    return "\n".join(lines) + "\n"
