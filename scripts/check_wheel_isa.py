"""Fail if a Linux wheel ships native code tied to newer CPUs than its baseline.

The default PyPI wheels must run on any machine of their architecture. FLINT's
``configure`` picks ``-march`` from the CPU it is built on, so a wheel built on a
GitHub runner that happened to have AVX-512 crashed with SIGILL in ``factor`` and
polynomial GCD on every CPU without it (3.13.0). This scans the bundled libraries
with ``objdump``, on the architecture the wheel was built for:

x86_64
    FLINT and MPFR, for instructions that use the ymm/zmm registers or AVX-512
    opmasks. Not scanned, because their vector code is chosen at run time: GMP
    (built with ``--enable-fat``, one kernel per CPU family) and the extension
    module itself, whose AVX/FMA paths sit behind ``is_x86_feature_detected!``
    (kernel/eval_const.rs).

aarch64
    FLINT, MPFR and GMP, for SVE/SVE2 (the z and p registers). GMP is included
    because ``--enable-fat`` does not exist on aarch64. The instruction stream is
    the reliable signal: ``.note.gnu.property`` records only BTI/PAC there, not
    an architecture level, and AArch64 objects carry no arch attribute that a
    toolchain has to set. NEON is baseline on ARMv8-A and is what FLINT's
    ``fft_small`` uses. LSE atomics are not flagged: GCC's outline-atomics helpers
    contain them and select them at run time.

Usage: python scripts/check_wheel_isa.py dist/*.whl
"""

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

X86_CHECKED = re.compile(r"(^|/)(libflint[^/]*\.so[^/]*|libmpfr[^/]*\.so[^/]*)$")
AARCH64_CHECKED = re.compile(r"(^|/)lib(flint|mpfr|gmp)[^/]*\.so[^/]*$")
# ymm = AVX/AVX2, zmm and {%k1}..{%k7} opmasks = AVX-512 (AT&T syntax).
X86_VECTOR = re.compile(r"%[yz]mm\d|\{%k[1-7]\}")
# SVE/SVE2: z0.d-style vector registers, p0/m, p0/z and p0.b-style predicates.
AARCH64_VECTOR = re.compile(
    r"\bz(?:[0-9]|[12][0-9]|3[01])\.[bhsdq]\b|\bp(?:[0-9]|1[0-5])(?:/[mz]|\.[bhsdq])\b"
)
# A disassembled instruction: "  1234:\t<mnemonic>...".
INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\s", re.M)
# <symbol+0x10> annotations: a name there is not an operand.
SYMBOL = re.compile(r"<[^>]*>")


def scan(path: Path, vector: re.Pattern[str]) -> tuple[int, list[str]]:
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    count = len(INSTRUCTION.findall(out))
    hits = [line.strip() for line in out.splitlines() if vector.search(SYMBOL.sub("", line))]
    return count, hits


def main(wheels: list[str]) -> int:
    if not wheels:
        print("usage: check_wheel_isa.py WHEEL...", file=sys.stderr)
        return 2
    bad = 0
    for wheel in wheels:
        if "aarch64" in Path(wheel).name:
            checked, vector, label = AARCH64_CHECKED, AARCH64_VECTOR, "SVE/SVE2"
        else:
            checked, vector, label = X86_CHECKED, X86_VECTOR, "AVX/AVX2/AVX-512"
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(wheel) as zf:
            names = [n for n in zf.namelist() if checked.search(n)]
            if not names:
                print(f"{wheel}: no bundled FLINT/MPFR found")
                bad += 1
                continue
            for name in names:
                count, hits = scan(Path(zf.extract(name, tmp)), vector)
                # An empty disassembly would otherwise pass as "no hits".
                if count == 0:
                    print(f"{Path(wheel).name}: {name}: objdump produced no instructions")
                    bad += 1
                    continue
                status = "ok" if not hits else f"{len(hits)} {label} instructions"
                print(f"{Path(wheel).name}: {name}: {count} instructions, {status}")
                if hits:
                    bad += 1
                    for line in hits[:5]:
                        print(f"    {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
