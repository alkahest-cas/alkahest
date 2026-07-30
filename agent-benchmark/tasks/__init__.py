"""Benchmark task registry."""

from __future__ import annotations

from .base import AgentTask, Kind, Outcome, answer_line
from .catalogue import ALL_TASKS

TASK_BY_NAME: dict[str, AgentTask] = {t.name: t for t in ALL_TASKS}

__all__ = [
    "ALL_TASKS",
    "TASK_BY_NAME",
    "AgentTask",
    "Kind",
    "Outcome",
    "answer_line",
    "get_tasks",
]


def get_tasks(
    names: list[str] | None = None,
    *,
    kinds: list[str] | None = None,
    max_difficulty: int | None = None,
) -> list[AgentTask]:
    """Select tasks by name, kind, and/or difficulty ceiling."""
    tasks = list(ALL_TASKS)
    if names:
        tasks = [TASK_BY_NAME[n] for n in names if n in TASK_BY_NAME]
    if kinds:
        wanted = set(kinds)
        tasks = [t for t in tasks if t.kind.value in wanted]
    if max_difficulty is not None:
        # A ceiling, not an exact match: the old harness filtered with `==`,
        # so `--difficulty 2` silently dropped every easy task.
        tasks = [t for t in tasks if t.difficulty <= max_difficulty]
    return tasks
