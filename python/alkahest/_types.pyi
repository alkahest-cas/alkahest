"""Stub types for alkahest — filled out in later phases."""

from __future__ import annotations

class Expr:
    """A symbolic expression node (Phase 1)."""

    ...

class Pool:
    """Expression intern pool and context manager (Phase 7)."""
    def __enter__(self) -> Pool: ...
    def __exit__(self, *args: object) -> None: ...

class Domain:
    """Symbol domain (structural identity of symbols in a pool)."""

    Real: Domain
    Complex: Domain
    Integer: Domain
    Positive: Domain
    NonNegative: Domain
    NonZero: Domain

class DerivedResult:
    """Result of a symbolic transformation with a derivation log (Phase 4+)."""

    value: Expr
    derivation: str
    steps: list[dict[str, object]]
    certificate: str | None
    certificate_status: dict[str, object]

class Certifiability:
    """Verdict from :func:`certifiable` — truthy iff a certificate is produced."""

    certifiable: bool
    operation: str
    shape: str
    verdict: str
    reason: str
    detail: str
    checked: bool
    result: DerivedResult | None
    error: Exception | None
    evidence: dict[str, object] | None

    def __bool__(self) -> bool: ...
    def as_dict(self) -> dict[str, object]: ...

class BoundsSupport:
    """Verdict from :func:`bounds_supported` — truthy iff every construct in
    the expression has a rigorous Taylor-model rule."""

    supported: bool
    blocker: str | None
    functions: list[str]
    detail: str

    def __bool__(self) -> bool: ...
    def as_dict(self) -> dict[str, object]: ...

def bounds_supported(expr: Expr) -> BoundsSupport: ...
def certifiable(
    op: str | object,
    *args: object,
    mode: str = ...,
    **kwargs: object,
) -> Certifiability: ...
def certificate_coverage(operation: str | None = ...) -> list[dict[str, object]]: ...
def require_certificate(result: DerivedResult) -> DerivedResult: ...
def version() -> str: ...
