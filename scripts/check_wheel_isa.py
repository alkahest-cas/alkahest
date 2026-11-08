"""Fail if a Linux wheel ships native code that needs AVX, AVX2 or AVX-512.

The default PyPI wheel must run on any x86-64 machine. FLINT's ``configure``
picks ``-march`` from the CPU it is built on, so a wheel built on a GitHub
runner that happened to have AVX-512 crashed with SIGILL in ``factor`` and
polynomial GCD on every CPU without it (3.13.0). This scans the bundled FLINT
and MPFR with ``objdump`` for instructions that use the ymm/zmm registers or
AVX-512 opmasks.

Not scanned, because their vector code is chosen at run time: GMP (built with
``--enable-fat``, one kernel per CPU family) and the extension module itself,
whose AVX/FMA paths sit behind ``is_x86_feature_detected!`` (kernel/eval_const.rs).

Usage: python scripts/check_wheel_isa.py dist/*.whl
"""

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Bundled libraries that are compiled with fixed flags and so must be portable.
CHECKED = re.compile(r"(^|/)(libflint[^/]*\.so[^/]*|libmpfr[^/]*\.so[^/]*)$")
# ymm = AVX/AVX2, zmm and {%k1}..{%k7} opmasks = AVX-512 (AT&T syntax).
VECTOR = re.compile(r"%[yz]mm\d|\{%k[1-7]\}")


def scan(path: Path) -> list[str]:
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if VECTOR.search(line)]


def main(wheels: list[str]) -> int:
    if not wheels:
        print("usage: check_wheel_isa.py WHEEL...", file=sys.stderr)
        return 2
    bad = 0
    for wheel in wheels:
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(wheel) as zf:
            names = [n for n in zf.namelist() if CHECKED.search(n)]
            if not names:
                print(f"{wheel}: no bundled FLINT/MPFR found")
                bad += 1
                continue
            for name in names:
                path = Path(zf.extract(name, tmp))
                hits = scan(path)
                status = "ok" if not hits else f"{len(hits)} AVX/AVX2/AVX-512 instructions"
                print(f"{Path(wheel).name}: {name}: {status}")
                if hits:
                    bad += 1
                    for line in hits[:5]:
                        print(f"    {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
