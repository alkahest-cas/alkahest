"""Run Math-tab requests safely: parse in-process, compute in a sandboxed child.

The FastAPI app calls :func:`interpret_payload` for the live preview (pure
parsing, no Alkahest) and :func:`run_math` to compute.  ``run_math`` first
parses in-process, so malformed or oversized input is refused without
spawning anything, then hands the request to :mod:`math_worker` in a fresh
``python -I`` child with:

* ``RLIMIT_AS`` — address-space cap (``ALKAHEST_MATH_MEMORY_MB``, default 2048);
  Alkahest itself refuses with ``E-BUDGET-005`` as it nears this, and the
  ``Budget(max_bytes=…)`` it runs under caps GMP growth below it;
* ``RLIMIT_CPU`` — CPU-seconds cap, a backstop for the wall clock;
* ``RLIMIT_FSIZE = 0`` / ``RLIMIT_NOFILE`` / no core dumps;
* a hard wall-clock timeout (``ALKAHEST_MATH_TIMEOUT_S``, default 15 s) after
  which the child's whole process group is SIGKILLed — the cooperative
  ``Budget(wall_ms=…)`` cannot interrupt a single FLINT call, this can;
* a concurrency cap (``ALKAHEST_MATH_CONCURRENCY``, default 2), with
  ``E-PLAYGROUND-010`` when the queue does not clear within a few seconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import mathlang as ml

WORKER = Path(__file__).with_name("math_worker.py")


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

_semaphore: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, CONCURRENCY))
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


def _limits() -> None:  # runs in the child between fork and exec
    import resource

    mem = int(MEMORY_MB) * 1024 * 1024
    cpu = int(HARD_TIMEOUT_S) + 2
    for which, value in (
        (resource.RLIMIT_AS, mem),
        (resource.RLIMIT_CPU, cpu),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_NOFILE, 256),
    ):
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(which, (value, value))
    with contextlib.suppress(OSError):
        os.nice(5)


def _child_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "MALLOC_ARENA_MAX": "2",
        "RAYON_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "1",
    }
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return env


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def _spawn() -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        str(WORKER),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=_limits,
        start_new_session=True,
        env=_child_env(),
        cwd=str(WORKER.parent),
    )


async def run_math(payload: Any) -> dict:
    """Compute one Math-tab request; never raises."""
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
    try:
        await asyncio.wait_for(sem.acquire(), timeout=QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        return _error(
            ml.E_BUSY,
            "The server is busy with other calculations; try again in a moment.",
            interpreted_latex=interp,
        )
    try:
        try:
            proc = await _spawn()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return _error(
                ml.E_INTERNAL,
                f"Could not start the calculation process ({type(exc).__name__}).",
                interpreted_latex=interp,
            )
        try:
            out, err = await asyncio.wait_for(proc.communicate(data), timeout=HARD_TIMEOUT_S)
        except asyncio.TimeoutError:
            _kill_group(proc)
            await proc.wait()
            return _error(
                ml.E_TIMEOUT,
                f"Stopped after {HARD_TIMEOUT_S} s without an answer.",
                "The playground runs every calculation under a hard time limit. "
                "This is a resource limit, not a statement that no answer exists.",
                interpreted_latex=interp,
            )
        finally:
            if proc.returncode is None:
                _kill_group(proc)
                await proc.wait()
        if proc.returncode != 0 or not out:
            stderr = err.decode("utf-8", "replace")
            rc = proc.returncode
            if rc in (-signal.SIGXCPU, -signal.SIGKILL):
                return _error(
                    ml.E_TIMEOUT,
                    "Stopped: the calculation used up its CPU-time allowance.",
                    interpreted_latex=interp,
                )
            if "MemoryError" in stderr or "Cannot allocate" in stderr or rc == -signal.SIGABRT:
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
