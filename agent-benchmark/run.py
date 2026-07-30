"""Agent benchmark CLI.

Typical use
-----------
    # One-time: build the isolated per-arm environments (needs network)
    python agent-benchmark/run.py --setup-envs

    # Run the benchmark
    ANTHROPIC_API_KEY=sk-... python agent-benchmark/run.py --repeats 5

    # Only the traps, against a different model
    OPENAI_API_KEY=sk-... python agent-benchmark/run.py \\
        --model gpt-4o --kinds trap --repeats 5

    # Preview prompts without spending anything
    python agent-benchmark/run.py --dry-run

Results stream to JSONL as they complete, so an interrupted run keeps whatever
it finished.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import envs
from harness import build_provenance, run_one
from report import build_report
from tasks import ALL_TASKS, TASK_BY_NAME, Kind, get_tasks

_HERE = Path(__file__).parent
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_OUT = _HERE / "results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark LLM agents across CAS skill guides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--skills",
        default="alkahest,sympy,none",
        help=(
            "Comma-separated arms (default: alkahest,sympy,none). "
            "'none' is the no-CAS control arm. Add 'mathematica' if a Wolfram "
            "kernel is available."
        ),
    )
    p.add_argument("--tasks", default=None, help="Comma-separated task names")
    p.add_argument(
        "--kinds",
        default=None,
        help=f"Comma-separated task kinds: {','.join(k.value for k in Kind)}",
    )
    p.add_argument(
        "--max-difficulty",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Include tasks at or below this difficulty (a ceiling, not an exact match)",
    )
    p.add_argument("--model", default=_DEFAULT_MODEL, help="LiteLLM model string")
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Samples per (skill, task). Use >=5 for meaningful confidence "
            "intervals; requires --temperature > 0 to produce any variation."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Parallel in-flight model calls (default 4)",
    )
    p.add_argument(
        "--memory-mb",
        type=int,
        default=4096,
        help="Address-space cap for generated code (default 4096)",
    )
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--alkahest-spec",
        default="alkahest",
        help=(
            "pip requirement for the alkahest arm (default: the published PyPI "
            "wheel, i.e. what a prospective user would get). Point at a local "
            "wheel or a pinned version for reproducibility."
        ),
    )
    p.add_argument(
        "--setup-envs",
        action="store_true",
        help="Create/refresh the per-arm virtualenvs, then exit",
    )
    p.add_argument(
        "--recreate-envs",
        action="store_true",
        help="With --setup-envs, delete and rebuild from scratch",
    )
    p.add_argument(
        "--no-isolation",
        action="store_true",
        help=(
            "UNSAFE: run generated code in the current interpreter instead of "
            "per-arm venvs. Results are not comparable across arms; for debugging "
            "the harness only."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print prompts and exit")
    p.add_argument("--debug", action="store_true", help="Print generated code")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--list-skills", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = envs.build_registry(args.alkahest_spec)

    if args.list_tasks:
        print(f"{'name':32s} {'kind':12s} diff  category")
        for t in ALL_TASKS:
            print(f"{t.name:32s} {t.kind.value:12s} {t.difficulty}     {t.category}")
        return 0

    if args.list_skills:
        for name, spec in registry.items():
            built = "built" if envs.env_exists(name) else "NOT BUILT"
            exists = "ok" if spec.skill_file.exists() else "MISSING SKILL FILE"
            print(f"{name:14s} {built:10s} {exists:20s} {spec.skill_file}")
        return 0

    skill_names = [s.strip() for s in args.skills.split(",") if s.strip()]
    unknown = [s for s in skill_names if s not in registry]
    if unknown:
        print(f"ERROR: unknown skills {unknown}; available {list(registry)}", file=sys.stderr)
        return 1
    specs = {name: registry[name] for name in skill_names}

    if args.setup_envs:
        print("Building isolated environments (one venv per arm):")
        for name, spec in specs.items():
            print(f"[{name}]")
            try:
                envs.create_env(spec, recreate=args.recreate_envs)
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr)
                continue
            info = envs.probe_env(spec)
            available, reason = envs.arm_available(spec, info)
            print(f"  {'ready' if available else 'UNAVAILABLE: ' + reason}")
            print(f"  {json.dumps(info)[:300]}")
        return 0

    # Resolve tasks
    task_names = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown_tasks = [n for n in task_names if n not in TASK_BY_NAME]
        if unknown_tasks:
            print(f"ERROR: unknown tasks {unknown_tasks}", file=sys.stderr)
            return 1
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    tasks = get_tasks(task_names, kinds=kinds, max_difficulty=args.max_difficulty)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 1

    if args.dry_run:
        for spec in specs.values():
            for task in tasks:
                print("=" * 72)
                print(f"SKILL {spec.name}   TASK {task.name} [{task.kind.value}]")
                print("-" * 72)
                print(task.prompt)
                print()
        return 0

    if args.repeats > 1 and args.temperature == 0.0:
        print(
            "WARNING: --repeats > 1 with --temperature 0.0 will produce nearly "
            "identical samples and confidence intervals that understate the true "
            "variance. Consider --temperature 0.7.",
            file=sys.stderr,
        )

    # Resolve interpreters per arm.
    pythons: dict[str, Path] = {}
    for name in specs:
        if args.no_isolation:
            pythons[name] = Path(sys.executable)
        else:
            if not envs.env_exists(name):
                print(
                    f"ERROR: no environment for '{name}'. Run:\n"
                    f"  python {Path(__file__).name} --setup-envs --skills {args.skills}",
                    file=sys.stderr,
                )
                return 1
            pythons[name] = envs.env_python(name)

    if args.no_isolation:
        print(
            "WARNING: --no-isolation is set. Every arm can import every CAS, so "
            "cross-arm results are NOT comparable.",
            file=sys.stderr,
        )

    provenance = build_provenance(specs, args.model, args.temperature, args.repeats, envs.ENV_ROOT)

    # Drop arms whose environment cannot run, rather than scoring them as
    # a wall of failures that would flatter the remaining arms.
    active: dict[str, object] = {}
    for name, spec in specs.items():
        arm = provenance["arms"][name]
        if arm["available"] or args.no_isolation:
            active[name] = spec
        else:
            print(
                f"SKIPPING arm '{name}': {arm['unavailable_reason']}",
                file=sys.stderr,
            )
    if not active:
        print("No usable arms.", file=sys.stderr)
        return 1

    skill_texts = {
        name: spec.skill_file.read_text(encoding="utf-8") if spec.skill_file.exists() else ""
        for name, spec in active.items()
    }

    jobs = [(name, task, rep) for name in active for task in tasks for rep in range(args.repeats)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    report_path = args.output_dir / "report.md"
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(
        f"Arms {list(active)} | {len(tasks)} tasks | {args.repeats} repeats "
        f"| {len(jobs)} runs | model {args.model}"
    )

    results: list[dict] = []
    done = 0
    with (
        results_path.open("w", encoding="utf-8") as fh,
        concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool,
    ):
        futures = {
            pool.submit(
                run_one,
                active[name],
                skill_texts[name],
                task,
                args.model,
                pythons[name],
                repeat=rep,
                temperature=args.temperature,
                memory_mb=args.memory_mb,
                keep_code=args.debug,
            ): (name, task, rep)
            for name, task, rep in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            name, task, _rep = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  harness error on {name}/{task.name}: {exc}", file=sys.stderr)
                continue
            row = dataclasses.asdict(result)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            results.append(row)
            done += 1
            flag = " SILENT-WRONG" if row["silent_error"] else ""
            print(
                f"  [{done}/{len(jobs)}] {name:11s} {task.name:30s} "
                f"{row['status']:14s} {row['exec_ms']:7.0f}ms{flag}"
            )
            if args.debug and row.get("code"):
                print(f"--- code ---\n{row['code']}\n---")

    report = build_report(results, provenance)
    report_path.write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"Results    -> {results_path}")
    print(f"Report     -> {report_path}")
    print(f"Provenance -> {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
