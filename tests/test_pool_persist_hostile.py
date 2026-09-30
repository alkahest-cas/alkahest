"""Loading a corrupt or hostile ``.akp`` pool file must never kill the process.

Audit B8/B9: a four-byte length of ``0xFFFFFFF0`` used to be allocated
before its bytes were read (an allocation failure aborts the interpreter —
nothing to catch), a zero denominator or a zero float precision panicked
inside ``rug``, and a duplicated node shifted every later child reference.

Every load runs in a subprocess, so an abort shows up as a non-zero exit
code instead of taking the test runner down with it.
"""

from __future__ import annotations

import os
import random
import struct
import subprocess
import sys
import textwrap

import pytest
from alkahest import ExprPool


def _hdr(n: int, ver: int = 5) -> bytes:
    return b"ALKP" + struct.pack("<IIQ", ver, 0, n)


def _s(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def _sym(name: str) -> bytes:
    return b"\x00\x00\x01" + _s(name.encode())


def _int(n: int) -> bytes:
    return b"\x01" + _s(str(n).encode())


def _rat(n: int, d: int) -> bytes:
    return b"\x02" + _s(str(n).encode()) + _s(str(d).encode())


def _flt(prec: int, m: str) -> bytes:
    return b"\x03" + struct.pack("<I", prec) + _s(m.encode())


def _add(*ids: int) -> bytes:
    return b"\x04" + struct.pack("<I", len(ids)) + b"".join(struct.pack("<I", i) for i in ids)


def _pow(b: int, e: int) -> bytes:
    return b"\x06" + struct.pack("<II", b, e)


# name -> (bytes, expected outcome: "err" or "ok")
CRAFTED = {
    "huge_strlen": (_hdr(1) + b"\x00\x00\x01" + struct.pack("<I", 0xFFFFFFF0), "err"),
    "huge_arity": (_hdr(1) + b"\x04" + struct.pack("<I", 0xFFFFFFF0), "err"),
    "huge_branches": (_hdr(1) + b"\x08" + struct.pack("<I", 0xFFFFFFF0), "err"),
    "huge_count": (_hdr(2**62) + _sym("x"), "err"),
    "rational_den0": (_hdr(1) + _rat(1, 0), "err"),
    "float_prec0": (_hdr(1) + _flt(0, "1.0"), "err"),
    "float_prec_huge": (_hdr(1) + _flt(0xFFFFFFFF, "1.0"), "err"),
    "forward_child": (_hdr(2) + _sym("x") + _add(0, 7), "err"),
    "nonreduced_den1": (_hdr(2) + _rat(4, 2) + _int(2), "ok"),
    "non_canonical_add": (_hdr(3) + _sym("x") + _sym("y") + _add(1, 0), "ok"),
    "duplicate_node_drift": (_hdr(4) + _sym("x") + _sym("x") + _sym("y") + _pow(1, 2), "ok"),
}

_LOADER = textwrap.dedent(
    """
    import sys
    from alkahest import ExprPool
    for path in sys.argv[1:]:
        try:
            q = ExprPool.load_from(path)
        except Exception as e:  # a PanicException is BaseException: not caught
            print("err", type(e).__name__, str(e).splitlines()[0][:200])
            continue
        x = q.symbol("x"); y = q.symbol("y")
        print("ok", str(x ** y), str(q.add([x, y])))
    """
)


def _run_loader(paths: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _LOADER, *paths],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("name", sorted(CRAFTED))
def test_crafted_pool_file_is_refused_or_loaded_never_aborts(tmp_path, name):
    data, expected = CRAFTED[name]
    path = tmp_path / f"{name}.akp"
    path.write_bytes(data)
    r = _run_loader([str(path)])
    assert r.returncode == 0, f"{name}: process died rc={r.returncode}\n{r.stderr[-2000:]}"
    line = r.stdout.strip()
    assert line.startswith(expected), f"{name}: {line}\n{r.stderr[-2000:]}"
    if expected == "err":
        assert "IoError" in line.split()[1], line


def _valid_pool_bytes(tmp_path) -> bytes:
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    e = (x + p.integer(10**30)) ** y * p.rational(-3, 7) + p.float(2.5)
    e = p.func("sin", [e]) + p.func("f", [x, y, e])
    _ = e
    path = tmp_path / "valid.akp"
    p.save_to(str(path))
    return path.read_bytes()


def test_truncated_and_bit_flipped_pool_files_never_abort(tmp_path):
    good = _valid_pool_bytes(tmp_path)
    rng = random.Random(20260930)
    variants: list[bytes] = [good[:n] for n in range(0, len(good), max(1, len(good) // 60))]
    for _ in range(400):
        b = bytearray(good)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(b))
            b[i] ^= 1 << rng.randrange(8)
        variants.append(bytes(b))
    # Overwrite a random aligned word with a huge length / id.
    for _ in range(100):
        b = bytearray(good)
        i = rng.randrange(20, len(b) - 4)
        b[i : i + 4] = struct.pack("<I", rng.choice([0xFFFFFFFF, 0xFFFFFFF0, 0x7FFFFFFF, 1 << 24]))
        variants.append(bytes(b))
    paths = []
    for k, v in enumerate(variants):
        path = tmp_path / f"v{k}.akp"
        path.write_bytes(v)
        paths.append(str(path))
    r = _run_loader(paths)
    assert r.returncode == 0, f"loader died rc={r.returncode}\n{r.stderr[-3000:]}"
    lines = r.stdout.strip().splitlines()
    assert len(lines) == len(paths), r.stderr[-3000:]
    for path, line in zip(paths, lines):
        ok = line.startswith("ok") or (line.startswith("err") and "IoError" in line.split()[1])
        assert ok, f"{os.path.basename(path)}: {line}"
