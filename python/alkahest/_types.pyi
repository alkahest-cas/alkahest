"""Stub types for alkahest — filled out in later phases."""

from __future__ import annotations

class Expr:
    """A symbolic expression node (Phase 1)."""

    ...

class Pool:
    """Expression intern pool and context manager (Phase 7)."""
    def __enter__(self) -> Pool: ...
    def __exit__(self, *args: object) -> None: ...

class DerivedResult:
    """Result of a symbolic transformation with a derivation log (Phase 4+)."""

    value: Expr
    derivation: str
    steps: list[dict[str, object]]

def version() -> str: ...
