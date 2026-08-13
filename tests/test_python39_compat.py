"""The package must import on the oldest Python it claims to support.

`pyproject.toml` says ``requires-python = ">=3.9"`` and the release matrix
builds a 3.9 wheel, but nothing imports the result on 3.9 until a tag is
pushed. 3.8.0 shipped a `str | None` return annotation into
``_probe_oracles``; the wheel built and installed cleanly and then died on
``import alkahest`` with

    TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'

because PEP 604 unions in a *runtime-evaluated* annotation need 3.10. Every
`def` annotation is evaluated at import time unless the module opts into
``from __future__ import annotations`` or quotes the annotation, so this is
not a typing nicety — it makes the package unimportable.

Ruff will not catch it: its pyupgrade rules propose `X | Y` for newer targets
and never reject it for older ones. So check it directly, on source, in a test
that runs on every PR rather than only at release.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "python" / "alkahest"

# The floor declared in pyproject.toml. PEP 604 unions became valid in
# annotations at runtime in 3.10, and PEP 585 builtin generics (`list[int]`)
# in 3.9 — so `list[int]` is fine here and `int | None` is not.
PEP604_MIN = (3, 10)

# The floor from pyproject.toml, as ast.parse wants it.
PY_MIN = (3, 9)


def _python_files() -> list[pathlib.Path]:
    return sorted(p for p in PACKAGE.rglob("*.py"))


def _has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
        ):
            return True
    return False


def _union_offenders(tree: ast.Module) -> list[tuple[int, str]]:
    """Runtime-evaluated annotations using `X | Y`.

    A quoted annotation is an `ast.Constant` string and is never evaluated at
    import time, so it is safe regardless of the union inside it.
    """
    found: list[tuple[int, str]] = []

    def check(annotation: ast.expr | None) -> None:
        if annotation is None or isinstance(annotation, ast.Constant):
            return
        for sub in ast.walk(annotation):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                found.append((sub.lineno, ast.unparse(annotation)))
                return

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check(node.returns)
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                check(arg.annotation)
            for arg in (args.vararg, args.kwarg):
                if arg is not None:
                    check(arg.annotation)
        elif isinstance(node, ast.AnnAssign):
            check(node.annotation)
    return found


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_runtime_pep604_union_without_future_import(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if _has_future_annotations(tree):
        return
    offenders = _union_offenders(tree)
    assert not offenders, (
        f"{path.relative_to(PACKAGE.parent.parent)} uses PEP 604 `X | Y` in a "
        f"runtime-evaluated annotation but does not have "
        f"`from __future__ import annotations`. This raises TypeError on "
        f"import under Python 3.9, which pyproject.toml still supports. "
        f"Quote the annotation or add the future import. Offenders: {offenders}"
    )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_parses_as_python_39(path: pathlib.Path) -> None:
    """Catch syntax that 3.9 cannot even parse — `match`, PEP 604 in a stub, etc.

    Complements the check above, which is about annotations *evaluated* at
    import; this one is about the file being readable at all.
    """
    try:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=PY_MIN)
    except SyntaxError as exc:  # pragma: no cover - only on a regression
        pytest.fail(
            f"{path.name}:{exc.lineno} is not valid Python 3.9 syntax ({exc.msg}), "
            f"but pyproject.toml declares requires-python >= 3.9"
        )


def test_the_check_would_catch_the_38_regression() -> None:
    """The gate must fail on the exact code that shipped, or it is decoration."""
    regressed = ast.parse("def _probe_oracles() -> tuple[tuple[str, str | None], ...]: ...")
    assert _union_offenders(regressed), "gate failed to flag the 3.8.0 regression"

    quoted = ast.parse('def _probe_oracles() -> "tuple[tuple[str, str | None], ...]": ...')
    assert not _union_offenders(quoted), "quoted annotations are not evaluated at import"
