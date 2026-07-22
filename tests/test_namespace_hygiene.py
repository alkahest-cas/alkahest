"""Top-level ``alkahest`` must not leak import-machinery names into ``dir()``."""

from __future__ import annotations

import alkahest


def test_no_stdlib_or_submodule_leaks_in_dir():
    names = set(dir(alkahest))
    assert "contextlib" not in names
    assert "suppress" not in names
    assert "exceptions" not in names
    # Native extension submodule is loaded via ``from .alkahest import …``
    # but must not pollute the public package namespace.
    assert "alkahest" not in names


def test_exceptions_submodule_still_importable():
    from alkahest.exceptions import ParseError

    assert issubclass(ParseError, Exception)


def test_intentional_exports_still_present():
    assert "version" in alkahest.__all__
    assert callable(alkahest.version)
    assert "parse" in alkahest.__all__
