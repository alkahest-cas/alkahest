"""The package must be readable by the oldest Python it claims to support.

`pyproject.toml` declares a ``requires-python`` floor and the release matrix
builds a wheel for it, but nothing imports the result on that interpreter
until a tag is pushed (3.8.0 shipped a module that died on ``import
alkahest`` under the then-floor 3.9). So check the source on every PR: each
module must parse with ``feature_version`` set to the floor, which rejects
syntax newer than the floor (``except*``, PEP 695 ``type`` aliases, ...).

The floor is read from ``pyproject.toml`` rather than copied here.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "python" / "alkahest"


def _declared_floor() -> tuple[int, int]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^requires-python\s*=\s*">=\s*(\d+)\.(\d+)"', text, re.MULTILINE)
    assert m, 'pyproject.toml has no `requires-python = ">=X.Y"`'
    return int(m.group(1)), int(m.group(2))


PY_MIN = _declared_floor()


def _python_files() -> list[pathlib.Path]:
    return sorted(p for p in PACKAGE.rglob("*.py"))


def test_floor_is_310() -> None:
    """Python 3.9 was dropped after 3.13.2; the floor is 3.10."""
    assert PY_MIN == (3, 10)


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_parses_at_the_declared_floor(path: pathlib.Path) -> None:
    try:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=PY_MIN)
    except SyntaxError as exc:  # pragma: no cover - only on a regression
        pytest.fail(
            f"{path.name}:{exc.lineno} is not valid Python {PY_MIN[0]}.{PY_MIN[1]} "
            f"syntax ({exc.msg}), but pyproject.toml declares that floor"
        )


def test_the_check_rejects_syntax_above_the_floor() -> None:
    """The gate must fail on code the floor cannot parse, or it is decoration."""
    with pytest.raises(SyntaxError):
        ast.parse("try:\n    pass\nexcept* ValueError:\n    pass\n", feature_version=PY_MIN)
