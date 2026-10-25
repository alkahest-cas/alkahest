"""Stub types for alkahest — filled out in later phases."""

from __future__ import annotations

from fractions import Fraction

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

# ---------------------------------------------------------------------------
# First-reach conveniences (Python numbers are coerced exactly: ``int`` and
# ``Fraction`` to integer / rational nodes, ``float`` to a float node).
# ---------------------------------------------------------------------------

Number = int | Fraction | float

pi: Expr
"""The constant π, in the active ``context(pool=...)`` pool or :func:`default_pool`."""
E: Expr
"""Euler's number ``exp(1)``."""
I: Expr
"""The imaginary unit."""
oo: Expr
"""Positive infinity (SymPy's spelling)."""

def default_pool() -> Pool:
    """The process-wide pool used when no ``Expr`` argument names one."""
    ...

def factor(expr: Expr | DerivedResult | Number, vars: list[Expr] | None = ...) -> Expr:
    """Factor over ℤ (numerator and denominator), as an ``Expr`` product."""
    ...

def expand(expr: Expr | DerivedResult | Number) -> Expr:
    """Distribute products and positive integer powers over sums."""
    ...

def factorial(n: Expr | Number) -> Expr:
    """``n!``: an exact integer for integer ``n >= 0``, else ``gamma(n + 1)``."""
    ...

def binomial(n: Expr | Number, k: Expr | Number) -> Expr:
    """``C(n, k)``: exact for integer arguments, else a ratio of gammas."""
    ...

def sin(x: Expr | DerivedResult | Number) -> Expr:
    """``sin(x)``; a Python number argument is coerced exactly (``sin(2)``)."""
    ...

def cos(x: Expr | DerivedResult | Number) -> Expr: ...
def tan(x: Expr | DerivedResult | Number) -> Expr: ...
def exp(x: Expr | DerivedResult | Number) -> Expr: ...
def log(x: Expr | DerivedResult | Number) -> Expr: ...
def sqrt(x: Expr | DerivedResult | Number) -> Expr: ...
def gamma(x: Expr | DerivedResult | Number) -> Expr: ...
