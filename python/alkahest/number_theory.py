"""alkahest.number_theory — integer number theory (V3-1, FLINT-backed).

Public API mirrors common SymPy ``ntheory`` entry points: arbitrary-precision
arguments are accepted as Python ``int`` objects.
"""

from __future__ import annotations

from .alkahest import (  # noqa: F401
    DirichletChi,
    NumberTheoryError,
    nt_discrete_log,
    nt_factorint,
    nt_isprime,
    nt_jacobi,
    nt_nextprime,
    nt_nthroot_mod,
    nt_totient,
)

__all__ = [
    "DirichletChi",
    "NumberTheoryError",
    "discrete_log",
    "factorint",
    "isprime",
    "jacobi_symbol",
    "nextprime",
    "nthroot_mod",
    "totient",
]


def _decimal(n: int) -> str:
    return str(int(n))


def isprime(n: int) -> bool:
    """Return ``True`` if ``n`` is a (proved) prime (``fmpz_is_prime``)."""
    return nt_isprime(str(int(n)))


def factorint(n: int) -> dict[int, int]:
    """Prime factorisation of ``n`` with SymPy-compatible sign handling."""
    zi = int(n)
    if zi == 0:
        return {0: 1}
    sign, pairs = nt_factorint(_decimal(zi))
    out: dict[int, int] = {int(p): int(e) for p, e in pairs}
    if sign < 0:
        out[-1] = 1 + out.get(-1, 0)
    return out


def nextprime(n: int, proved: bool = True) -> int:
    """Smallest prime strictly greater than ``n``."""
    return int(nt_nextprime(_decimal(int(n)), proved))


def totient(n: int) -> int:
    """Euler totient φ(n) for integers n ≥ 1."""
    return int(nt_totient(_decimal(int(n))))


def jacobi_symbol(a: int, n: int) -> int:
    """Jacobi symbol (a | n) for odd integers n > 1."""
    return nt_jacobi(str(int(a)), _decimal(int(n)))


def nthroot_mod(a: int, k: int, m: int) -> int:
    """Some integer ``x`` with ``pow(x, k, m) == a % m`` for prime modulus ``m``."""
    return int(nt_nthroot_mod(_decimal(int(a)), int(k), _decimal(int(m))))


def discrete_log(residue: int, base: int, modulus: int) -> int:
    """Exponent ``e`` with ``pow(base, e, modulus) == residue % modulus`` (prime ``modulus``)."""
    return int(
        nt_discrete_log(
            _decimal(int(residue)),
            _decimal(int(base)),
            _decimal(int(modulus)),
        )
    )
