"""Fail if a macOS wheel depends on anything a clean Mac does not have.

Every macOS arm64 wheel up to 3.13.1 linked ``/opt/homebrew/opt/flint/lib/...``
(and gmp, mpfr) with no bundled copies: ``import alkahest`` was a dyld error on
any Mac without exactly those Homebrew kegs, and the release smoke test passed
only because the runner had them. For every Mach-O in the wheel this checks that

* each dependency is in ``/usr/lib/`` or ``/System/``, or is an ``@loader_path``
  or ``@rpath`` reference that resolves to a file inside the wheel;
* no ``LC_RPATH`` points outside the wheel or the system;
* its minimum macOS is no newer than the wheel's platform tag promises (a dylib
  built for macOS 15 inside a ``macosx_11_0`` wheel fails to load on macOS 14);
* it is arm64.

and that FLINT, GMP and MPFR are actually among the bundled libraries, so a wheel
with no native code in it cannot pass. ``--target`` also fails the wheel if its
tag has risen above that macOS version.

It reads the load commands itself rather than running ``otool -L``, so that it
runs anywhere, in particular on a wheel downloaded to a Linux machine.

Usage: python scripts/check_wheel_macos.py --target 11.0 dist/*.whl
"""

from __future__ import annotations

import argparse
import posixpath
import re
import struct
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
CPU_TYPE_ARM64 = 0x0100000C

LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
LC_RPATH = 0x8000001C
# Every load command that makes dyld load another image.
DYLIB_LOADS = {
    0x0C,  # LC_LOAD_DYLIB
    0x20,  # LC_LAZY_LOAD_DYLIB
    0x80000018,  # LC_LOAD_WEAK_DYLIB
    0x8000001F,  # LC_REEXPORT_DYLIB
    0x80000023,  # LC_LOAD_UPWARD_DYLIB
}

SYSTEM_PREFIXES = ("/usr/lib/", "/System/")
REQUIRED_LIBRARIES = ("libflint", "libgmp", "libmpfr")
PLATFORM_TAG = re.compile(r"-macosx_(\d+)_(\d+)_(\w+)\.whl$")


@dataclass
class Image:
    """What dyld reads from one architecture slice of a Mach-O."""

    cputype: int
    dependencies: list[str] = field(default_factory=list)
    rpaths: list[str] = field(default_factory=list)
    minos: tuple[int, int] | None = None


def _string_at(command: bytes, offset: int) -> str:
    return command[offset:].split(b"\0", 1)[0].decode()


def _parse_thin(data: bytes) -> Image:
    _, cputype, _, _, ncmds, _, _, _ = struct.unpack_from("<8I", data, 0)
    image = Image(cputype)
    pos = 32  # sizeof(mach_header_64)
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<2I", data, pos)
        command = data[pos : pos + size]
        if cmd in DYLIB_LOADS:
            name_offset = struct.unpack_from("<I", command, 8)[0]
            image.dependencies.append(_string_at(command, name_offset))
        elif cmd == LC_RPATH:
            path_offset = struct.unpack_from("<I", command, 8)[0]
            image.rpaths.append(_string_at(command, path_offset))
        elif cmd in (LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX):
            # build_version_command has `platform` before minos; the older
            # version_min_command does not.
            offset = 12 if cmd == LC_BUILD_VERSION else 8
            version = struct.unpack_from("<I", command, offset)[0]
            image.minos = (version >> 16, (version >> 8) & 0xFF)
        pos += size
    return image


def parse_macho(data: bytes) -> list[Image] | None:
    """The slices of a Mach-O (one unless universal), or None if `data` is not one."""
    if len(data) < 8:
        return None
    if struct.unpack_from("<I", data, 0)[0] == MH_MAGIC_64:
        return [_parse_thin(data)]
    if struct.unpack_from(">I", data, 0)[0] == FAT_MAGIC:
        count = struct.unpack_from(">I", data, 4)[0]
        images = []
        for index in range(count):
            offset, size = struct.unpack_from(">2I", data, 8 + index * 20 + 8)
            images.append(_parse_thin(data[offset : offset + size]))
        return images
    return None


def _inside_wheel(loader: str, relative: str, files: set[str]) -> bool:
    candidate = posixpath.normpath(posixpath.join(posixpath.dirname(loader), relative))
    return candidate in files


def dependency_problem(dep: str, loader: str, rpaths: list[str], files: set[str]) -> str | None:
    """Why `dep`, loaded by the wheel member `loader`, would not resolve on a clean Mac."""
    if dep.startswith(SYSTEM_PREFIXES):
        return None
    if dep.startswith("@loader_path/"):
        if _inside_wheel(loader, dep[len("@loader_path/") :], files):
            return None
        return f"{dep}: no such file in the wheel"
    if dep.startswith("@rpath/"):
        name = dep[len("@rpath/") :]
        for rpath in rpaths:
            if rpath.startswith(SYSTEM_PREFIXES):
                return None
            if rpath == "@loader_path" or rpath.startswith("@loader_path/"):
                relative = rpath[len("@loader_path") :].lstrip("/")
                if _inside_wheel(loader, posixpath.join(relative, name), files):
                    return None
        return f"{dep}: resolves to nothing in the wheel (rpaths: {rpaths})"
    return f"{dep}: outside /usr/lib, /System and the wheel"


def rpath_problem(rpath: str) -> str | None:
    if rpath.startswith(SYSTEM_PREFIXES) or rpath.split("/", 1)[0] == "@loader_path":
        return None
    return f"LC_RPATH {rpath}: outside the wheel and the system"


def _version(pair: tuple[int, int]) -> str:
    return f"{pair[0]}.{pair[1]}"


def check_wheel(path: str, target: tuple[int, int] | None) -> list[str]:
    """Every reason `path` would not work on a clean Mac; empty if it is fine."""
    name = Path(path).name
    tag = PLATFORM_TAG.search(name)
    if tag is None:
        return [f"{name}: no macosx_X_Y_arch platform tag"]
    tag_version = (int(tag[1]), int(tag[2]))
    if tag[3] != "arm64":
        return [f"{name}: only arm64 wheels are built, this is {tag[3]}"]

    problems = []
    if target is not None and tag_version > target:
        problems.append(
            f"{name}: tagged macOS {_version(tag_version)}, above the target {_version(target)}"
        )

    with zipfile.ZipFile(path) as zf:
        files = {n for n in zf.namelist() if not n.endswith("/")}
        images = {}
        for member in sorted(files):
            parsed = parse_macho(zf.read(member))
            if parsed is not None:
                images[member] = parsed

    for member, slices in images.items():
        for image in slices:
            if image.cputype != CPU_TYPE_ARM64:
                problems.append(f"{member}: not arm64 (cputype {image.cputype:#x})")
            if image.minos is None:
                problems.append(f"{member}: no minimum-macOS load command")
            elif image.minos > tag_version:
                problems.append(
                    f"{member}: needs macOS {_version(image.minos)}, "
                    f"the wheel promises {_version(tag_version)}"
                )
            for rpath in image.rpaths:
                problem = rpath_problem(rpath)
                if problem:
                    problems.append(f"{member}: {problem}")
            for dep in image.dependencies:
                problem = dependency_problem(dep, member, image.rpaths, files)
                if problem:
                    problems.append(f"{member}: {problem}")
            minos = "unknown" if image.minos is None else _version(image.minos)
            print(f"{name}: {member}: minos {minos}, {len(image.dependencies)} dependencies")

    basenames = [posixpath.basename(member) for member in images]
    for library in REQUIRED_LIBRARIES:
        if not any(base.startswith(library) for base in basenames):
            problems.append(f"{name}: {library} is not bundled")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", help="highest macOS version the wheel tag may claim, e.g. 11.0")
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args(argv)
    target = None
    if args.target:
        major, minor = args.target.split(".")
        target = (int(major), int(minor))

    failed = False
    for wheel in args.wheels:
        problems = check_wheel(wheel, target)
        for problem in problems:
            print(f"::error::{problem}")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
