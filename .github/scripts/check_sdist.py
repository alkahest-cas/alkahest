"""Fail if the source distribution is incomplete or carries junk.

``maturin sdist`` packs what the Cargo manifests reach. A path dependency that
only appears under ``[patch.crates-io]`` (``vendor/egglog``) is not one of those,
so it has to be listed under ``[tool.maturin] include``; without it every source
build fails at ``cargo metadata``. The forbidden list keeps a release from
uploading whatever happened to be lying around the checkout.

Usage: python .github/scripts/check_sdist.py dist/*.tar.gz
"""

import re
import sys
import tarfile
from pathlib import Path

REQUIRED = (
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "alkahest-core/Cargo.toml",
    "alkahest-core/build.rs",
    "alkahest-core/src/lib.rs",
    "alkahest-py/Cargo.toml",
    "alkahest-py/src/lib.rs",
    "python/alkahest/__init__.py",
    "vendor/egglog/Cargo.toml",
)
JUNK = re.compile(
    r"(^|/)(dist|wheels[^/]*|temp-alkahest|node_modules|website|demo-playground|target"
    r"|__pycache__|\.git)/"
    r"|\.(whl|so|pyd|dylib|pyc)$"
)
# Far above the ~5 MB it should be, far below what a stray artifact adds.
MAX_BYTES = 25 * 1024 * 1024


def check(path: Path) -> list[str]:
    errors = []
    with tarfile.open(path) as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
    # Every member sits under the top-level alkahest-X.Y.Z/ directory.
    relative = [name.split("/", 1)[1] for name in names if "/" in name]
    present = set(relative)
    errors += [f"missing: {name}" for name in REQUIRED if name not in present]
    errors += [f"junk: {name}" for name in relative if JUNK.search(name)]
    size = path.stat().st_size
    if size > MAX_BYTES:
        errors.append(f"{size} bytes exceeds the {MAX_BYTES} limit")
    top = sorted({name.split("/", 1)[0] for name in relative})
    print(f"{path.name}: {size / 1024 / 1024:.1f} MiB, {len(relative)} files, top level: {top}")
    return errors


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: check_sdist.py SDIST...", file=sys.stderr)
        return 2
    bad = 0
    for path in paths:
        for error in check(Path(path)):
            print(f"  {error}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
