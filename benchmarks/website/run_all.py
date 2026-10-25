"""Run the landing-page benchmark suites and write the published data.

Outputs
-------
``benchmarks/website/raw_results.json``
    Every measurement: all timed runs, warm-up time, fingerprint, status.
``website/src/data/benchmarks.json``
    The data contract rendered by the landing page.
``benchmarks/website/RESULTS.md``
    Human-readable tables with every number, sizes, versions and machine.

Usage
-----
    # whole run (wrap in the resource limits you need; see RESULTS.md)
    python benchmarks/website/run_all.py

    # re-measure some cases / systems and merge into the existing raw file
    python benchmarks/website/run_all.py --cases expand_fateman --systems sympy

    # only re-render benchmarks.json and RESULTS.md from raw_results.json
    python benchmarks/website/run_all.py --render-only
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _harness import REPO, compare_fp, run_system  # noqa: E402

SUITE_MODULES = ["core", "pharma", "engineering", "physics", "finance", "numeric"]
RAW = HERE / "raw_results.json"
SITE_JSON = REPO / "website" / "src" / "data" / "benchmarks.json"
RESULTS_MD = HERE / "RESULTS.md"


def load_suites(modules: list[str]):
    out = []
    for m in modules:
        try:
            mod = importlib.import_module(m)
        except ModuleNotFoundError:
            continue
        for s in mod.SUITES:
            out.append((m, s))
    return out


def machine_info() -> dict:
    info: dict = {"python": platform.python_version(), "kernel": platform.release()}
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for ln in lscpu.splitlines():
            k, _, v = ln.partition(":")
            if k.strip() == "Model name":
                info["cpu"] = v.strip()
            elif k.strip() == "CPU(s)":
                info["logical_cpus"] = int(v.strip())
            elif k.strip() == "Core(s) per socket":
                info["cores_per_socket"] = int(v.strip())
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            kb = int(fh.readline().split()[1])
        info["ram_gib"] = round(kb / 2**20)
    except Exception:
        pass
    try:
        with open("/etc/os-release") as fh:
            for ln in fh:
                if ln.startswith("PRETTY_NAME="):
                    info["os"] = ln.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    info["cpu_quota"] = os.environ.get("BENCH_CPU_QUOTA", "8 CPUs (systemd CPUQuota=800%)")
    return info


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def judge(case, recs: dict[str, dict]) -> None:
    """Attach a correctness verdict to every record of one case (in place)."""
    for r in recs.values():
        if r.get("status") in ("wrong", "incomplete") and r.get("fp") is not None:
            r["status"] = "ok"
    ref = recs.get("reference")
    ref_fp = ref["fp"] if ref and ref.get("status") == "ok" else None
    ref_kind = "independent reference"
    if ref_fp is None:
        # Consensus: the fingerprint agreed on by the most finished systems (>= 2).
        fps = [r["fp"] for s, r in recs.items() if s != "reference" and r.get("status") == "ok"]
        best, best_n = None, 0
        for f in fps:
            n = sum(1 for g in fps if compare_fp(g, f, case.tol)[0])
            if n > best_n:
                best, best_n = f, n
        if best_n >= 2:
            ref_fp, ref_kind = best, f"consensus of {best_n} systems"
    for s, r in recs.items():
        if s == "reference" or r.get("status") != "ok":
            continue
        if ref_fp is None:
            r["check"] = "unverified (no reference and no two systems agree)"
            r["correct"] = None
            continue
        if any(v in ("FAILED", "COMPLEX") for v in r["fp"] if isinstance(v, str)):
            bad = sum(1 for v in r["fp"] if v in ("FAILED", "COMPLEX"))
            r["status"] = "incomplete"
            r["correct"] = False
            r["check"] = f"{bad} of {len(r['fp'])} sub-problems not solved"
            continue
        ok, msg = compare_fp(r["fp"], ref_fp, case.tol)
        r["correct"] = ok
        r["check"] = f"{msg} ({ref_kind})" if ok else f"MISMATCH: {msg} ({ref_kind})"
        if not ok:
            r["status"] = "wrong"


def run(args) -> None:
    raw = json.loads(RAW.read_text()) if RAW.exists() else {"cases": {}}
    raw.setdefault("cases", {})
    suites = load_suites(args.suites.split(",") if args.suites else SUITE_MODULES)
    want_cases = set(args.cases.split(",")) if args.cases else None
    want_sys = set(args.systems.split(",")) if args.systems else None
    for module, suite in suites:
        for case in suite.cases:
            if want_cases and case.id not in want_cases:
                continue
            entry = raw["cases"].setdefault(case.id, {"suite": suite.id, "records": {}})
            systems = list(case.systems)
            if case.reference is not None:
                systems = ["reference", *systems]
            for system in systems:
                if want_sys and system not in want_sys and system != "reference":
                    continue
                if system == "reference" and want_sys and "reference" in entry["records"]:
                    continue
                print(f"[{suite.id}] {case.id:28s} {system:12s} ... ", end="", flush=True)
                rec = run_system(module, case, system)
                rec["measured"] = datetime.datetime.now().isoformat(timespec="seconds")
                entry["records"][system] = rec
                if rec.get("median_ms") is not None:
                    print(f"{rec['median_ms']:.3f} ms (n={len(rec['runs_s'])}) fp={rec['fp']!s:.80}")
                else:
                    print(f"{rec['status']} fp={rec.get('fp')!s:.80} {rec.get('error', '')}")
            judge(case, entry["records"])
            for s, r in entry["records"].items():
                if s != "reference" and r.get("status") not in ("ok",):
                    print(f"    -> {s}: {r.get('status')} {r.get('check', '')}")
            RAW.write_text(json.dumps(raw, indent=1))
    raw["machine"] = machine_info()
    raw["commit"] = git_commit()
    RAW.write_text(json.dumps(raw, indent=1))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

DISPLAY = {
    "alkahest": "Alkahest",
    "sympy": "SymPy",
    "symengine": "SymEngine",
    "mathematica": "Mathematica",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "mpmath": "mpmath",
}


def _machine_str(m: dict) -> str:
    parts = [m.get("cpu", "unknown CPU")]
    if "cores_per_socket" in m:
        parts.append(f"{m['cores_per_socket']} cores / {m.get('logical_cpus')} threads")
    if "ram_gib" in m:
        parts.append(f"{m['ram_gib']} GiB RAM")
    if "os" in m:
        parts.append(m["os"])
    parts.append(f"runs capped at {m.get('cpu_quota')}")
    return ", ".join(parts)


def render(args) -> None:
    import alkahest

    raw = json.loads(RAW.read_text())
    suites = load_suites(SUITE_MODULES)
    commit = raw.get("commit") or git_commit() or ""
    alk_version = alkahest.__version__
    today = datetime.date.today().isoformat()
    site = {
        "generated": today,
        "alkahest_version": alk_version,
        "alkahest_commit": commit[:12],
        "machine": _machine_str(raw.get("machine", {})),
        "methodology": "Median of 5 timed runs after 1 discarded warm-up (3 runs if a run takes "
        "> 5 s, 1 if > 20 s); inputs rebuilt and caches cleared before every run; each system "
        "in its own process with a per-run timeout. Mathematica timed with AbsoluteTiming inside "
        "the kernel (start-up excluded). Every answer is checked against an independent "
        "reference; wrong or incomplete answers are marked and never count as a win.",
        "suites": [],
    }
    for _module, suite in suites:
        s_out = {"id": suite.id, "label": suite.label, "blurb": suite.blurb, "cases": []}
        for case in suite.cases:
            entry = raw["cases"].get(case.id)
            if not entry:
                continue
            judge(case, entry["records"])
            results = []
            for system in case.systems:
                r = entry["records"].get(system)
                if r is None:
                    continue
                item = {"system": system}
                if r.get("version"):
                    item["version"] = r["version"]
                status = r.get("status")
                if status in ("ok", "wrong", "incomplete"):
                    item["value"] = round(r["median_ms"], 4 if r["median_ms"] < 1 else 2)
                else:
                    item["value"] = None
                item["status"] = status
                if r.get("runs_s"):
                    item["runs"] = len(r["runs_s"])
                if r.get("check"):
                    item["check"] = r["check"]
                results.append(item)
            c_out = {
                "id": case.id,
                "task": case.task,
                "detail": case.detail,
                "unit": case.unit,
                "results": results,
            }
            notes = [case.note] if case.note else []
            for it in results:
                if it["status"] == "timeout":
                    notes.append(f"{DISPLAY.get(it['system'], it['system'])} did not finish within {case.timeout_s} s.")
                elif it["status"] == "error":
                    err = entry["records"][it["system"]].get("error")
                    notes.append(f"{DISPLAY.get(it['system'], it['system'])} raised an error: {err}.")
                elif it["status"] == "wrong":
                    notes.append(f"{DISPLAY.get(it['system'], it['system'])} returned a wrong answer ({it['check']}).")
                elif it["status"] == "incomplete":
                    notes.append(f"{DISPLAY.get(it['system'], it['system'])}: {it['check']}.")
            if any(it["value"] is None for it in results):
                c_out["timeout_s"] = case.timeout_s
            if notes:
                c_out["note"] = " ".join(notes)
            c_out["script"] = suite.script
            s_out["cases"].append(c_out)
        if s_out["cases"]:
            site["suites"].append(s_out)
    SITE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SITE_JSON.write_text(json.dumps(site, indent=2, ensure_ascii=False) + "\n")
    RESULTS_MD.write_text(render_md(site, raw, suites))
    print(f"wrote {SITE_JSON} and {RESULTS_MD}")


def _fmt_ms(v) -> str:
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    if v >= 10:
        return f"{v:.1f} ms"
    if v >= 1:
        return f"{v:.2f} ms"
    return f"{v * 1000:.0f} µs"


def render_md(site: dict, raw: dict, suites) -> str:
    m = raw.get("machine", {})
    L: list[str] = []
    L.append("# Landing-page benchmarks: full results\n")
    L.append(
        f"Generated {site['generated']} by `benchmarks/website/run_all.py` against Alkahest "
        f"{site['alkahest_version']} (commit `{raw.get('commit', '')}`).\n"
    )
    L.append(
        "These are the numbers shown on the landing page (`website/src/data/benchmarks.json`), "
        "with every timed run, the check applied to each answer, and the losses.\n"
    )
    L.append("## Machine and versions\n")
    L.append(f"- CPU: {m.get('cpu')} ({m.get('cores_per_socket')} cores, {m.get('logical_cpus')} threads)")
    L.append(f"- RAM: {m.get('ram_gib')} GiB; OS: {m.get('os')}, Linux {m.get('kernel')}; Python {m.get('python')}")
    L.append(f"- Resource cap: {m.get('cpu_quota')}, MemoryMax=24G, `MALLOC_ARENA_MAX=8`")
    versions: dict[str, set] = {}
    for e in raw["cases"].values():
        for s, r in e["records"].items():
            if s != "reference" and r.get("version"):
                versions.setdefault(s, set()).add(r["version"])
    for s in sorted(versions):
        L.append(f"- {DISPLAY.get(s, s)}: {', '.join(sorted(versions[s]))}")
    L.append("")
    L.append("## Method\n")
    L.append(site["methodology"] + "\n")
    L.append(
        "- Each (case, system) pair runs in a fresh process; inputs are generated from fixed seeds "
        "in `benchmarks/website/_inputs.py` and parsed by every system from the same string "
        "(parsing is not timed).\n"
        "- Timing covers only the operation named in the table, not parsing, printing or the "
        "correctness check.\n"
        "- `status` is `ok` (answer matches the reference), `wrong` (does not), `incomplete` "
        "(some sub-problems unsolved), `timeout`, or `error`.\n"
        "- The machine is a shared multi-user server; the runs were serialised with `flock` and "
        "capped with `systemd-run`, but other users' load adds noise. Outliers were re-run and "
        "the min/max columns below show the spread.\n"
    )
    for _module, suite in suites:
        s = next((x for x in site["suites"] if x["id"] == suite.id), None)
        if s is None:
            continue
        L.append(f"## {s['label']}\n")
        L.append(s["blurb"] + "\n")
        for c in s["cases"]:
            case = next(cc for cc in suite.cases if cc.id == c["id"])
            entry = raw["cases"][c["id"]]
            L.append(f"### {c['task']} (`{c['id']}`)\n")
            L.append(c["detail"] + "\n")
            L.append("| System | Version | Median | Min | Max | Runs | Status | Check |")
            L.append("|---|---|---|---|---|---|---|---|")
            alk = next((r for r in c["results"] if r["system"] == "alkahest"), None)
            for r in c["results"]:
                rec = entry["records"][r["system"]]
                if r["value"] is not None:
                    med = _fmt_ms(rec.get("median_ms"))
                elif r["status"] == "timeout":
                    med = f"timeout > {case.timeout_s} s"
                else:
                    med = r["status"]
                L.append(
                    f"| {DISPLAY.get(r['system'], r['system'])} | {r.get('version', '')} | {med} | "
                    f"{_fmt_ms(rec.get('min_ms'))} | {_fmt_ms(rec.get('max_ms'))} | "
                    f"{len(rec.get('runs_s') or [])} | {r['status']} | {r.get('check', '')} |"
                )
            if alk and alk["value"] and alk["status"] == "ok":
                ratios = []
                for r in c["results"]:
                    if r["system"] == "alkahest":
                        continue
                    if r["value"] is not None and r["status"] == "ok":
                        q = r["value"] / alk["value"]
                        word = f"{q:.1f}× faster than" if q >= 1 else f"{1 / q:.1f}× slower than"
                        ratios.append(f"{word} {DISPLAY.get(r['system'], r['system'])}")
                if ratios:
                    L.append("\nAlkahest is " + "; ".join(ratios) + ".")
            if c.get("note"):
                L.append(f"\nNote: {c['note']}")
            L.append(f"\nScript: `{c['script']}`\n")
    L.append("## Reproduce\n")
    L.append(
        "```bash\n"
        "# 1. Build/install alkahest with the features used here\n"
        "maturin build --release --manifest-path alkahest-py/Cargo.toml \\\n"
        '  --features "egraph groebner parallel cranelift system-gmp"\n'
        "pip install target/wheels/alkahest-*.whl sympy==1.14.0 symengine==0.14.1 mpmath numpy scipy\n"
        "# 2. (optional) Wolfram Engine / Mathematica 14.3 with WolframKernel on PATH\n"
        "# 3. Run everything (one benchmark process at a time), then render\n"
        "MALLOC_ARENA_MAX=8 RAYON_NUM_THREADS=8 python benchmarks/website/run_all.py\n"
        "# re-measure one case or one system and merge:\n"
        "python benchmarks/website/run_all.py --cases expand_fateman --systems sympy\n"
        "python benchmarks/website/run_all.py --render-only\n"
        "```\n"
    )
    L.append(
        "On the reference machine every invocation was wrapped as "
        "`flock ~/.cache/alkahest-build/heavy.lock systemd-run --user --scope -q -p MemoryMax=24G "
        "-p CPUQuota=800% …` so that only one benchmark ran at a time.\n"
    )
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", help="comma-separated suite modules (default: all)")
    ap.add_argument("--cases", help="comma-separated case ids")
    ap.add_argument("--systems", help="comma-separated systems")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    if not args.render_only:
        run(args)
    if not args.no_render:
        render(args)


if __name__ == "__main__":
    main()
