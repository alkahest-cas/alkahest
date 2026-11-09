"""The macOS wheel checker, against hand-built Mach-O load commands.

`scripts/check_wheel_macos.py` is the guard that keeps Homebrew paths out of the
macOS wheels, and CI only ever shows it a good wheel. These make sure it also
says no. They need no Mac: the checker parses the load commands itself.
"""

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_wheel_macos.py"
spec = importlib.util.spec_from_file_location("check_wheel_macos", SCRIPT)
checker = importlib.util.module_from_spec(spec)
# Registered first: @dataclass looks the module up while the class is created.
sys.modules["check_wheel_macos"] = checker
spec.loader.exec_module(checker)

ARM64 = checker.CPU_TYPE_ARM64
LC_LOAD_DYLIB = 0x0C


def _string_command(cmd, text):
    body = text.encode() + b"\0"
    # dylib_command carries timestamp/current/compat after the name offset;
    # rpath_command has only the offset.
    offset = 12 if cmd == checker.LC_RPATH else 24
    padded = len(body) + (-(offset + len(body)) % 8)
    fields = struct.pack("<3I", cmd, offset + padded, offset).ljust(offset, b"\0")
    return fields + body.ljust(padded, b"\0")


def macho(deps=(), rpaths=(), minos=(11, 0), cputype=ARM64):
    commands = [_string_command(LC_LOAD_DYLIB, d) for d in deps]
    commands += [_string_command(checker.LC_RPATH, r) for r in rpaths]
    if minos is not None:
        version = minos[0] << 16 | minos[1] << 8
        commands.append(struct.pack("<6I", checker.LC_BUILD_VERSION, 24, 1, version, version, 0))
    blob = b"".join(commands)
    header = struct.pack("<8I", checker.MH_MAGIC_64, cputype, 0, 6, len(commands), len(blob), 0, 0)
    return header + blob


DYLIBS = "alkahest/.dylibs"
EXT = "alkahest/alkahest.cpython-312-darwin.so"
SYSTEM = "/usr/lib/libSystem.B.dylib"
GOOD_DEPS = (
    SYSTEM,
    "@loader_path/.dylibs/libflint.22.dylib",
    "@loader_path/.dylibs/libgmp.10.dylib",
    "@loader_path/.dylibs/libmpfr.6.dylib",
)


def good_members():
    return {
        EXT: macho(GOOD_DEPS),
        f"{DYLIBS}/libflint.22.dylib": macho(
            (SYSTEM, "@loader_path/libgmp.10.dylib", "@loader_path/libmpfr.6.dylib")
        ),
        f"{DYLIBS}/libgmp.10.dylib": macho((SYSTEM,)),
        f"{DYLIBS}/libmpfr.6.dylib": macho((SYSTEM, "@loader_path/libgmp.10.dylib")),
        "alkahest/__init__.py": b"# not a Mach-O\n",
    }


def wheel(tmp_path, members, tag="macosx_11_0_arm64"):
    path = tmp_path / f"alkahest-3.13.1-cp312-cp312-{tag}.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return str(path)


def test_a_repaired_wheel_passes(tmp_path):
    assert checker.check_wheel(wheel(tmp_path, good_members()), (11, 0)) == []


def test_the_published_homebrew_linkage_fails(tmp_path):
    """The bug itself: 3.13.1's load commands, no bundled copies."""
    members = {
        EXT: macho(
            (
                "/opt/homebrew/opt/flint/lib/libflint.24.0.dylib",
                "/opt/homebrew/opt/gmp/lib/libgmp.10.dylib",
                "/opt/homebrew/opt/mpfr/lib/libmpfr.6.dylib",
                SYSTEM,
            )
        )
    }
    problems = checker.check_wheel(wheel(tmp_path, members), (11, 0))
    assert sum("outside /usr/lib" in p for p in problems) == 3
    assert sum("is not bundled" in p for p in problems) == 3


@pytest.mark.parametrize(
    "dep",
    [
        "@loader_path/.dylibs/libmissing.dylib",
        "@rpath/libflint.22.dylib",
        "@executable_path/libflint.22.dylib",
        "/usr/local/lib/libgmp.10.dylib",
    ],
)
def test_an_unresolvable_dependency_fails(tmp_path, dep):
    members = good_members()
    members[EXT] = macho((*GOOD_DEPS, dep))
    assert any(dep in p for p in checker.check_wheel(wheel(tmp_path, members), None))


def test_rpath_resolving_inside_the_wheel_passes(tmp_path):
    members = good_members()
    members[EXT] = macho((*GOOD_DEPS, "@rpath/libgmp.10.dylib"), rpaths=("@loader_path/.dylibs",))
    assert checker.check_wheel(wheel(tmp_path, members), None) == []


def test_a_leftover_absolute_rpath_fails(tmp_path):
    members = good_members()
    members[EXT] = macho(GOOD_DEPS, rpaths=("/Users/runner/macos-deps/lib",))
    assert any("LC_RPATH" in p for p in checker.check_wheel(wheel(tmp_path, members), None))


def test_a_library_newer_than_the_tag_fails(tmp_path):
    members = good_members()
    members[f"{DYLIBS}/libgmp.10.dylib"] = macho((SYSTEM,), minos=(15, 0))
    problems = checker.check_wheel(wheel(tmp_path, members), (11, 0))
    assert any("needs macOS 15.0" in p for p in problems)


def test_a_tag_above_the_target_fails(tmp_path):
    path = wheel(tmp_path, good_members(), tag="macosx_15_0_arm64")
    assert any("above the target" in p for p in checker.check_wheel(path, (11, 0)))
    assert checker.check_wheel(path, None) == []


def test_missing_minimum_os_fails(tmp_path):
    members = good_members()
    members[EXT] = macho(GOOD_DEPS, minos=None)
    assert any("minimum-macOS" in p for p in checker.check_wheel(wheel(tmp_path, members), None))


def test_x86_slice_fails(tmp_path):
    members = good_members()
    members[EXT] = macho(GOOD_DEPS, cputype=0x01000007)
    assert any("not arm64" in p for p in checker.check_wheel(wheel(tmp_path, members), None))


def test_universal_binaries_are_read():
    arm = macho(("/opt/homebrew/lib/libflint.dylib",))
    header = struct.pack(">2I", checker.FAT_MAGIC, 1) + struct.pack(
        ">5I", ARM64, 0, 28, len(arm), 3
    )
    images = checker.parse_macho(header + arm)
    assert [i.dependencies for i in images] == [["/opt/homebrew/lib/libflint.dylib"]]
