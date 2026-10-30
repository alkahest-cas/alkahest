"""Run Math-playground requests safely: parse in-process, compute in a sandboxed child.

:func:`interpret_payload` serves the live preview (pure parsing, no Alkahest
call) and :func:`run_math` computes.  ``run_math`` first parses in-process, so
malformed or oversized input is refused without spawning anything, then hands
the request to :mod:`alkahest.playground._worker` in a fresh ``python -I``
child that, before it reads the request, applies to itself (where the OS has
``resource``; Linux and macOS):

* ``RLIMIT_AS`` — address-space cap (``ALKAHEST_MATH_MEMORY_MB``, default 2048);
  Alkahest itself refuses with ``E-BUDGET-005`` as it nears this, and the
  ``Budget(max_bytes=…)`` it runs under caps GMP growth below it;
* ``RLIMIT_CPU`` — CPU-seconds cap, a backstop for the wall clock;
* ``RLIMIT_FSIZE = 0`` / ``RLIMIT_NOFILE`` / no core dumps.

Independently of the OS, the parent enforces:

* a hard wall-clock timeout (``ALKAHEST_MATH_TIMEOUT_S``, default 15 s) after
  which the child (and, on POSIX, its whole process group) is killed — the
  cooperative ``Budget(wall_ms=…)`` cannot interrupt a single FLINT call, this
  can;
* a concurrency cap (``ALKAHEST_MATH_CONCURRENCY``, default 2), with
  ``E-PLAYGROUND-010`` when the queue does not clear within a few seconds.

Everything here is synchronous and thread-safe (the stdlib server in
:mod:`alkahest.playground.server` runs one thread per request); async callers
wrap :func:`run_math` in ``asyncio.to_thread``.  Only the standard library is
used, so the module is importable on any platform; limits the OS cannot apply
are skipped rather than failing the request.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from . import mathlang as ml

WORKER = Path(__file__).with_name("_worker.py")
_POSIX = os.name == "posix"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


HARD_TIMEOUT_S = _env_int("ALKAHEST_MATH_TIMEOUT_S", 15)
WALL_MS = _env_int("ALKAHEST_MATH_WALL_MS", 8_000)
MEMORY_MB = _env_int("ALKAHEST_MATH_MEMORY_MB", 2_048)
CONCURRENCY = _env_int("ALKAHEST_MATH_CONCURRENCY", 2)
QUEUE_WAIT_S = 5.0
MAX_BODY_BYTES = 256 * 1024

_semaphore: threading.BoundedSemaphore | None = None
_semaphore_lock = threading.Lock()


def _sem() -> threading.BoundedSemaphore:
    global _semaphore
    with _semaphore_lock:
        if _semaphore is None:
            _semaphore = threading.BoundedSemaphore(max(1, CONCURRENCY))
        return _semaphore


def _error(code: str, message: str, hint: str | None = None, source: str = "playground", **kw):
    out: dict[str, Any] = {
        "ok": False,
        "error_code": code,
        "error_message": message,
        "error_hint": hint,
        "error_source": source,
    }
    out.update(kw)
    return out


def _validate(payload: Any) -> tuple[str, list]:
    if not isinstance(payload, dict):
        raise ml.MathInputError(ml.E_SYNTAX, "The request must be a JSON object.")
    text = payload.get("input")
    defs = payload.get("definitions") or []
    if not isinstance(text, str):
        raise ml.MathInputError(ml.E_SYNTAX, "'input' must be a string.")
    if not isinstance(defs, list):
        raise ml.MathInputError(ml.E_SYNTAX, "'definitions' must be a list.")
    return text, defs


def interpret_payload(payload: Any) -> dict:
    """Parse only; returns the interpretation LaTeX or a coded input error."""
    try:
        text, defs = _validate(payload)
        session = ml.build_session(defs)
        cmd, latex = ml.interpret(text, session)
        return {"ok": True, "interpreted_latex": latex, "kind": cmd.kind}
    except ml.MathInputError as e:
        return _error(e.code, e.message, e.hint, source="input")
    except RecursionError:
        return _error(ml.E_TOO_LARGE_INPUT, "The expression is nested too deeply.", source="input")


def _child_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C.UTF-8",
        "MALLOC_ARENA_MAX": "2",
        "RAYON_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    # Needed for the interpreter / native libraries to load on some platforms.
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SYSTEMROOT", "TEMP", "TMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _alkahest_root() -> str:
    """The ``sys.path`` entry the parent's ``alkahest`` was imported from."""
    import alkahest

    return str(Path(alkahest.__file__).resolve().parent.parent)


def _spawn() -> subprocess.Popen:
    limits = {
        "memory_mb": int(MEMORY_MB),
        "cpu_s": int(HARD_TIMEOUT_S) + 2,
        "alkahest_root": _alkahest_root(),
    }
    kwargs: dict[str, Any] = {}
    if _POSIX:
        # setsid() in the child, done by the C fork/exec path (thread-safe, unlike
        # preexec_fn); lets us kill anything the calculation might start.
        kwargs["start_new_session"] = True
    elif sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    return subprocess.Popen(
        [sys.executable, "-I", str(WORKER), json.dumps(limits)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(),
        cwd=str(WORKER.parent),
        **kwargs,
    )


def _kill(proc: subprocess.Popen) -> None:
    if _POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    with contextlib.suppress(OSError):
        proc.kill()


def _signal_of(rc: int | None, name: str) -> bool:
    sig = getattr(signal, name, None)
    return sig is not None and rc == -int(sig)


def run_math(payload: Any) -> dict:
    """Compute one Math-playground request; never raises."""
    pre = interpret_payload(payload)
    if not pre.get("ok"):
        return pre
    interp = pre.get("interpreted_latex")
    text, defs = _validate(payload)
    request = {
        "input": text,
        "definitions": defs,
        "wall_ms": WALL_MS,
        "max_bytes": max(64, MEMORY_MB // 2) * 1024 * 1024,
    }
    data = json.dumps(request).encode("utf-8")
    sem = _sem()
    if not sem.acquire(timeout=QUEUE_WAIT_S):
        return _error(
            ml.E_BUSY,
            "The server is busy with other calculations; try again in a moment.",
            interpreted_latex=interp,
        )
    try:
        try:
            proc = _spawn()
        except (OSError, ValueError, subprocess.SubprocessError, ImportError) as exc:
            return _error(
                ml.E_INTERNAL,
                f"Could not start the calculation process ({type(exc).__name__}).",
                interpreted_latex=interp,
            )
        try:
            out, err = proc.communicate(data, timeout=HARD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill(proc)
            with contextlib.suppress(Exception):
                proc.communicate(timeout=5)
            return _error(
                ml.E_TIMEOUT,
                f"Stopped after {HARD_TIMEOUT_S} s without an answer.",
                "The playground runs every calculation under a hard time limit. "
                "This is a resource limit, not a statement that no answer exists.",
                interpreted_latex=interp,
            )
        finally:
            if proc.poll() is None:
                _kill(proc)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=5)
        rc = proc.returncode
        if rc != 0 or not out:
            stderr = err.decode("utf-8", "replace")
            if _signal_of(rc, "SIGXCPU") or _signal_of(rc, "SIGKILL"):
                return _error(
                    ml.E_TIMEOUT,
                    "Stopped: the calculation used up its CPU-time allowance.",
                    interpreted_latex=interp,
                )
            if (
                "MemoryError" in stderr
                or "Cannot allocate" in stderr
                or "memory allocation" in stderr
                or _signal_of(rc, "SIGABRT")
            ):
                return _error(
                    ml.E_MEMORY,
                    f"Stopped: the calculation needed more than {MEMORY_MB} MB of memory.",
                    "This is a resource limit, not a statement that no answer exists.",
                    interpreted_latex=interp,
                )
            return _error(
                ml.E_INTERNAL,
                f"The calculation process exited unexpectedly (code {rc}).",
                stderr.strip().splitlines()[-1][:300] if stderr.strip() else None,
                interpreted_latex=interp,
            )
        try:
            res = json.loads(out.decode("utf-8"))
        except ValueError:
            return _error(ml.E_INTERNAL, "The calculation returned malformed output.")
        if isinstance(res, dict) and not res.get("interpreted_latex"):
            res["interpreted_latex"] = interp
        return res
    finally:
        sem.release()
