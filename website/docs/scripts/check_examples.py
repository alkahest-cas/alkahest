#!/usr/bin/env python3
"""Run the code samples shown on the docs home page against an installed
``alkahest`` and check the outputs promised in their trailing comments.

    python check_examples.py [path/to/python]

Each ``print(...)  # expected`` line must produce output containing
``expected`` (text before the first ``": "`` of the comment), in order.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CONTENT = Path(__file__).resolve().parent.parent / "content"


def expectations(src: str) -> list[str]:
    out = []
    for line in src.splitlines():
        m = re.match(r"\s*print\(.*\)\s+#\s*(.+)$", line)
        if m:
            out.append(m.group(1).split(": ")[0].strip())
    return out


def main() -> int:
    python = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    failed = 0
    for path in sorted(CONTENT.glob("*.py")):
        proc = subprocess.run(
            [python, "-I", str(path)], capture_output=True, text=True, timeout=300
        )
        ok = proc.returncode == 0
        pos = 0
        missing = []
        for exp in expectations(path.read_text(encoding="utf-8")):
            i = proc.stdout.find(exp, pos)
            if i < 0:
                missing.append(exp)
            else:
                pos = i + len(exp)
        status = "ok" if ok and not missing else "FAIL"
        print(f"{status:4} {path.name}")
        if not ok:
            print(proc.stderr[-2000:])
        for m in missing:
            print(f"     expected output not found: {m!r}")
        if status != "ok":
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
