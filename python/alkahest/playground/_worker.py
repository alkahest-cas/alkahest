"""Subprocess entry point for one Math-playground request.

Run as ``python -I _worker.py '<limits json>'`` by
:mod:`alkahest.playground.service`: ``-I`` means no user site, no ``PYTHON*``
environment and no current directory on ``sys.path``.  Before reading the
request the process caps its own address space, CPU time, open files and file
size (where the OS has :mod:`resource`), then reads one JSON request on stdin,
writes one JSON response on stdout and exits.  The parent kills it at a hard
wall-clock deadline.  A fresh process per request means nothing one request
does can be seen by the next.

Only the standard library is imported until the limits are in place.
"""

import contextlib
import json
import os
import sys

MAX_REQUEST_BYTES = 512 * 1024


def _apply_limits(memory_mb: int, cpu_s: int) -> None:
    try:
        import resource
    except ImportError:  # Windows: the parent's wall-clock kill still applies
        return
    caps = [
        ("RLIMIT_AS", memory_mb * 1024 * 1024),
        ("RLIMIT_CPU", cpu_s),
        ("RLIMIT_FSIZE", 0),
        ("RLIMIT_CORE", 0),
        ("RLIMIT_NOFILE", 256),
    ]
    for name, value in caps:
        which = getattr(resource, name, None)
        if which is None:
            continue
        # macOS refuses some limits (notably RLIMIT_AS); skip rather than fail.
        with contextlib.suppress(ValueError, OSError):
            _soft, hard = resource.getrlimit(which)
            if hard != resource.RLIM_INFINITY and value > hard:
                value = hard
            resource.setrlimit(which, (value, value))
    with contextlib.suppress(OSError, AttributeError):
        os.nice(5)


def _import_executor(alkahest_root: str):
    """Import the executor from the same Alkahest installation as the parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    # `python -I script.py` puts the script's directory first; drop it so the
    # sibling modules are only importable as `alkahest.playground.*`.
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    if alkahest_root and alkahest_root not in sys.path:
        sys.path.insert(0, alkahest_root)
    import alkahest

    # When the playground is used from a source checkout next to an installed
    # wheel that predates it, make this directory's package resolvable too.
    pkg_dir = os.path.dirname(here)
    if pkg_dir not in [os.path.abspath(p) for p in alkahest.__path__]:
        alkahest.__path__.append(pkg_dir)
    from alkahest.playground import math_exec

    return math_exec


def main() -> int:
    try:
        cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    except ValueError:
        cfg = {}
    _apply_limits(int(cfg.get("memory_mb") or 2048), int(cfg.get("cpu_s") or 17))
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        print(json.dumps({"ok": False, "error_code": "E-PLAYGROUND-002"}))
        return 0
    request = json.loads(raw.decode("utf-8"))
    math_exec = _import_executor(str(cfg.get("alkahest_root") or ""))
    response = math_exec.execute(request)
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
