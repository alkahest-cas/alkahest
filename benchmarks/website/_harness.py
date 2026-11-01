"""Shared harness for the landing-page benchmarks (``benchmarks/website/``).

Every (case, system) pair runs in its **own fresh subprocess** with a per-run
timeout, so one hang or crash cannot take the whole run down and no system
benefits from another's warm state.

Protocol of one measurement
---------------------------
1. ``prepare()`` builds the inputs (parsing, symbol creation, array allocation).
   This is **not** timed and is repeated before every run, so each run starts
   from fresh objects.  SymPy's global cache is cleared and ``gc.collect()``
   runs before each run; Mathematica runs ``ClearSystemCache[]``.
2. One **warm-up** run is executed and discarded (imports, JIT, lazy tables).
3. ``repeats`` timed runs follow (fewer when one run is slow: 3 runs if the
   warm-up took > 5 s, 1 run if it took > 20 s).  The reported value is the
   **median** of the timed runs.
4. The result of the last run is reduced to a *fingerprint* (a short list of
   numbers: exact integers or floats) which the parent compares with an
   independent reference.  A system whose fingerprint disagrees is reported
   with ``status: "wrong"`` and never counts as a win.

Python systems time with ``time.perf_counter`` around the work only.
Mathematica is driven through ``WolframKernel -script``; timing is
``AbsoluteTiming`` *inside the kernel*, so kernel start-up is excluded.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import json
import math
import os
import queue
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WOLFRAM_KERNEL = os.environ.get("WOLFRAM_KERNEL", "WolframKernel")
TAG = "@@"
SETUP_SLACK_S = 300

# ---------------------------------------------------------------------------
# Case description
# ---------------------------------------------------------------------------


@dataclass
class WL:
    """A Mathematica implementation.

    ``setup`` is WL code run (untimed) before every run; ``run`` is the timed
    WL expression, whose value is stored in ``res``; ``fp`` is a WL
    expression (it may use ``res``) evaluating to a list of numbers.
    """

    setup: str
    run: str
    fp: str


# A Python implementation is a zero-argument callable returning (go, fp):
# ``go()`` is the timed work, ``fp(result)`` maps its result to a list of
# numbers.  It is called afresh before every run.
PyImpl = Callable[[], "tuple[Callable[[], Any], Callable[[Any], list]]"]


@dataclass
class Case:
    id: str
    task: str
    detail: str
    systems: dict[str, Any]  # system name -> PyImpl | WL
    reference: PyImpl | None = None  # independent reference fingerprint
    tol: float = 1e-9  # relative tolerance for float fingerprints
    repeats: int = 5
    timeout_s: int = 120
    unit: str = "ms"
    note: str | None = None


@dataclass
class Suite:
    id: str
    label: str
    blurb: str
    script: str
    cases: list[Case] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def system_version(system: str) -> str | None:
    try:
        if system == "alkahest":
            import alkahest

            return alkahest.__version__
        if system == "sympy":
            import sympy

            return sympy.__version__
        if system == "symengine":
            import symengine

            return symengine.__version__
        if system in ("numpy", "numpy (hand-written)"):
            import numpy

            return numpy.__version__
        if system == "scipy":
            import scipy

            return scipy.__version__
        if system == "mpmath":
            import mpmath

            return mpmath.__version__
    except Exception:
        return None
    return None


_WL_VERSION: str | None = None


def wolfram_version() -> str | None:
    global _WL_VERSION
    if _WL_VERSION is None:
        try:
            out = subprocess.run(
                [WOLFRAM_KERNEL, "-noprompt", "-run", "Print[$VersionNumber];Exit[]"],
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
            line = [ln for ln in out.splitlines() if ln.strip()][-1]
            _WL_VERSION = line.strip().rstrip(".")
            # $VersionNumber is "14.3"; add the release number.
            out2 = subprocess.run(
                [WOLFRAM_KERNEL, "-noprompt", "-run", "Print[$ReleaseNumber];Exit[]"],
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
            rel = [ln for ln in out2.splitlines() if ln.strip()][-1].strip()
            _WL_VERSION = f"{_WL_VERSION}.{rel}"
        except Exception:
            _WL_VERSION = None
    return _WL_VERSION


# ---------------------------------------------------------------------------
# Worker (runs inside the child process)
# ---------------------------------------------------------------------------


def _emit(**kw: Any) -> None:
    sys.stdout.write(TAG + json.dumps(kw, default=_json_default) + "\n")
    sys.stdout.flush()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
    except ImportError:
        pass
    return str(o)


def _normalise_fp(vals: list) -> list:
    out = []
    for v in vals:
        if isinstance(v, bool):
            out.append(int(v))
        elif isinstance(v, int):
            # Big exact integers travel as strings.
            out.append(str(v) if abs(v) > 2**53 else v)
        elif isinstance(v, str):
            out.append(v)
        else:
            out.append(float(v))
    return out


def _clear_caches() -> None:
    if "sympy" in sys.modules:
        try:
            from sympy.core.cache import clear_cache

            clear_cache()
        except Exception:
            pass
    gc.collect()


def adaptive_runs(warm_s: float, repeats: int) -> int:
    if warm_s > 20:
        return 1
    if warm_s > 5:
        return min(repeats, 3)
    return repeats


def worker_main(module: str, case_id: str, system: str) -> None:
    sys.path.insert(0, str(HERE))
    mod = importlib.import_module(module)
    case = next(c for s in mod.SUITES for c in s.cases if c.id == case_id)
    impl = case.reference if system == "reference" else case.systems[system]
    _emit(event="version", version=system_version(system))
    if system == "reference":
        go, fp = impl()
        res = go()
        _emit(event="fp", fp=_normalise_fp(fp(res)))
        return
    res = None
    fp = None
    _clear_caches()
    go, fp = impl()
    _emit(event="start")
    t0 = time.perf_counter()
    res = go()
    warm = time.perf_counter() - t0
    _emit(event="warmup", s=warm)
    n = adaptive_runs(warm, case.repeats)
    for i in range(n):
        del res
        _clear_caches()
        go, fp = impl()
        _emit(event="start")
        t0 = time.perf_counter()
        res = go()
        dt = time.perf_counter() - t0
        _emit(event="run", i=i, s=dt)
    _emit(event="fp", fp=_normalise_fp(fp(res)))


# ---------------------------------------------------------------------------
# Mathematica script generation
# ---------------------------------------------------------------------------

_WL_PRELUDE = r"""
$HistoryLength = 0;
emit[a_Association] := (WriteString[$Output, "@@" <> ExportString[a, "RawJSON", "Compact" -> True] <> "\n"]);
fpv[x_String] := x;
fpv[x_Integer] := If[Abs[x] > 2^53, ToString[x], x];
fpv[x_Rational] := N[x];
fpv[x_?NumericQ] := N[x];
fpv[x_] := ToString[InputForm[x]];
"""


def wl_script(impl: WL, repeats: int, timeout_s: int) -> str:
    return (
        _WL_PRELUDE
        + f"""
emit[<|"event" -> "version", "version" -> ToString[$VersionNumber] <> "." <> ToString[$ReleaseNumber]|>];
doSetup[] := ({impl.setup});
ClearSystemCache[]; doSetup[];
emit[<|"event" -> "start"|>];
{{zzBenchTw, res}} = AbsoluteTiming[TimeConstrained[{impl.run}, {timeout_s}, $timedOut]];
If[res === $timedOut, emit[<|"event" -> "timeout"|>]; Exit[]];
emit[<|"event" -> "warmup", "s" -> zzBenchTw|>];
zzBenchNr = Which[zzBenchTw > 20, 1, zzBenchTw > 5, Min[{repeats}, 3], True, {repeats}];
Do[
  Clear[res]; ClearSystemCache[]; doSetup[];
  emit[<|"event" -> "start"|>];
  {{zzBenchT, res}} = AbsoluteTiming[{impl.run}];
  emit[<|"event" -> "run", "i" -> zzBenchI, "s" -> zzBenchT|>],
  {{zzBenchI, 0, zzBenchNr - 1}}];
emit[<|"event" -> "fp", "fp" -> Map[fpv, Flatten[{{{impl.fp}}}]]|>];
Exit[];
"""
    )


# ---------------------------------------------------------------------------
# Parent side: launching workers with timeouts
# ---------------------------------------------------------------------------


def _reader(stream, q: queue.Queue) -> None:
    for line in iter(stream.readline, ""):
        q.put(line)
    q.put(None)


def _launch(cmd: list[str], run_timeout: float, env: dict | None = None) -> dict:
    """Run ``cmd``; collect tagged JSON events.

    After a ``start`` event the next event must arrive within ``run_timeout``
    (the per-run limit); any other gap (imports, input set-up, fingerprint)
    may take up to ``SETUP_SLACK_S``.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
        cwd=str(HERE),
    )
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
    err_lines: list[str] = []

    def _drain_err():
        for ln in iter(proc.stderr.readline, ""):
            err_lines.append(ln)

    threading.Thread(target=_drain_err, daemon=True).start()
    events: list[dict] = []
    timed_out = False
    waiting_for_run = False
    while True:
        try:
            line = q.get(timeout=run_timeout if waiting_for_run else SETUP_SLACK_S)
        except queue.Empty:
            timed_out = True
            break
        if line is None:
            break
        if line.startswith(TAG):
            try:
                ev = json.loads(line[len(TAG) :])
            except json.JSONDecodeError:
                continue
            events.append(ev)
            waiting_for_run = ev.get("event") == "start"
            if ev.get("event") == "timeout":
                timed_out = True
                break
    if timed_out and events and events[-1].get("event") != "start":
        events.append({"event": "setup_timeout"})
    if timed_out:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    return {
        "events": events,
        "timed_out": timed_out,
        "returncode": proc.returncode,
        "stderr": "".join(err_lines)[-2000:],
    }


def run_system(module: str, case: Case, system: str) -> dict:
    """Measure one (case, system) pair; returns a raw result record."""
    impl = case.reference if system == "reference" else case.systems[system]
    per_line = case.timeout_s
    if isinstance(impl, WL):
        with tempfile.NamedTemporaryFile("w", suffix=".wl", delete=False) as fh:
            fh.write(wl_script(impl, case.repeats, case.timeout_s))
            path = fh.name
        try:
            out = _launch([WOLFRAM_KERNEL, "-noprompt", "-script", path], per_line)
        finally:
            os.unlink(path)
    else:
        env = dict(os.environ)
        env.setdefault("MALLOC_ARENA_MAX", "8")
        env.setdefault("RAYON_NUM_THREADS", "8")
        out = _launch(
            [sys.executable, str(HERE / "_harness.py"), "--worker", module, case.id, system],
            per_line,
            env=env,
        )
    evs = out["events"]
    version = next((e.get("version") for e in evs if e.get("event") == "version"), None)
    warm = next((e["s"] for e in evs if e.get("event") == "warmup"), None)
    runs = [e["s"] for e in evs if e.get("event") == "run"]
    fp = next((e["fp"] for e in evs if e.get("event") == "fp"), None)
    rec: dict[str, Any] = {
        "system": system,
        "version": version,
        "warmup_s": warm,
        "runs_s": runs,
        "fp": fp,
    }
    if out["timed_out"]:
        rec["status"] = "timeout"
    elif fp is None:
        rec["status"] = "error"
        rec["error"] = out["stderr"].strip().splitlines()[-1:] if out["stderr"] else ["no output"]
        rec["stderr"] = out["stderr"]
    else:
        rec["status"] = "ok"
    if runs and rec["status"] == "ok":
        rec["median_ms"] = statistics.median(runs) * 1000.0
        rec["min_ms"] = min(runs) * 1000.0
        rec["max_ms"] = max(runs) * 1000.0
    return rec


# ---------------------------------------------------------------------------
# Fingerprint comparison
# ---------------------------------------------------------------------------


def _num(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    return v


def compare_fp(got: list, ref: list, tol: float) -> tuple[bool, str]:
    if got is None or ref is None:
        return False, "missing fingerprint"
    if len(got) != len(ref):
        return False, f"fingerprint length {len(got)} != reference {len(ref)}"
    for i, (g, r) in enumerate(zip(got, ref)):
        g, r = _num(g), _num(r)
        if isinstance(g, int) and isinstance(r, int):
            if g != r:
                return False, f"entry {i}: {g} != {r}"
            continue
        if isinstance(g, str) or isinstance(r, str):
            if str(g) != str(r):
                return False, f"entry {i}: {g!r} != {r!r}"
            continue
        g, r = float(g), float(r)
        if math.isnan(g) or math.isnan(r):
            return False, f"entry {i}: NaN"
        if abs(g - r) > tol * max(1.0, abs(r)):
            return False, f"entry {i}: {g!r} vs reference {r!r}"
    return True, "matches reference"


# ---------------------------------------------------------------------------
# Small helpers shared by suites
# ---------------------------------------------------------------------------


def wl_str(s: str) -> str:
    """Python-syntax expression string -> Wolfram Language syntax."""
    return s.replace("**", "^")


DATA_DIR = HERE / ".data"


def shared_array(name: str, n: int, seed: int, lo: float, hi: float):
    """Uniform random float64 array, cached on disk so every system (including
    Mathematica, via ``BinaryReadList[path, "Real64"]``) reads identical data."""
    import numpy as np

    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}_{n}_{seed}.f64"
    if not path.exists():
        arr = np.random.default_rng(seed).uniform(lo, hi, n).astype("<f8")
        tmp = path.with_suffix(".tmp")
        arr.tofile(tmp)
        tmp.replace(path)
    return np.fromfile(path, dtype="<f8")


def shared_array_path(name: str, n: int, seed: int, lo: float, hi: float) -> str:
    shared_array(name, n, seed, lo, hi)
    return str(DATA_DIR / f"{name}_{n}_{seed}.f64")


def wl_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("usage: _harness.py --worker MODULE CASE SYSTEM", file=sys.stderr)
        sys.exit(2)
