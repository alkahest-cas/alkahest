"""Subprocess entry point for one Math-tab request.

Reads one JSON request on stdin, writes one JSON response on stdout, exits.
:mod:`math_service` starts it with ``python -I`` (no user site, no PYTHON*
environment, no current directory on ``sys.path``) under ``setrlimit`` caps on
address space, CPU time, open files and file size, and kills its whole process
group at a hard wall-clock deadline.  A fresh process per request means
nothing one request does can be seen by the next.
"""

from __future__ import annotations

import json
import os
import sys

MAX_REQUEST_BYTES = 512 * 1024


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        print(json.dumps({"ok": False, "error_code": "E-PLAYGROUND-002"}))
        return 0
    request = json.loads(raw.decode("utf-8"))
    import math_exec

    response = math_exec.execute(request)
    sys.stdout.write(json.dumps(response))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
