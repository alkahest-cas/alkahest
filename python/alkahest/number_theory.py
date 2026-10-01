"""alkahest.number_theory — integer number theory (V3-1, FLINT-backed).

Public API mirrors common SymPy ``ntheory`` entry points: arbitrary-precision
arguments are accepted as Python ``int`` objects.
"""

from __future__ import annotations

from .alkahest import (
    DirichletChi,
    NumberTheoryError,
    _decimal_to_int,
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


# The natives take a Python ``int`` directly (converted by bytes on the Rust
# side) and return decimal strings; ``_decimal_to_int`` parses those with GMP.
# Neither direction goes through ``str(int)`` / ``int(str)``, which are
# quadratic and raise past ``sys.get_int_max_str_digits()`` (4300 digits).


def isprime(n: int) -> bool:
    """Return ``True`` if ``n`` is a (proved) prime (``fmpz_is_prime``)."""
    return nt_isprime(int(n))


def factorint(n: int) -> dict[int, int]:
    """Prime factorisation of ``n`` with SymPy-compatible sign handling.

    Honours an active :class:`~alkahest.Budget`: under one, the factorisation
    climbs a ladder of bounded-effort passes (trial division, then ECM tuned
    for ever larger factors) and checks the budget between them, handing only
    cofactors of at most 160 bits to FLINT's full (uninterruptible) factoriser.
    A composite still unsplit when the budget runs out raises
    :exc:`BudgetExceededError` (``E-BUDGET-*``); the overshoot is at most one
    pass. With no budget the call is unbounded — a product of large primes
    with no small factor can take arbitrarily long.
    """
    sign, pairs = nt_factorint(int(n))
    out: dict[int, int] = {_decimal_to_int(p): int(e) for p, e in pairs}
    if sign < 0:
        out[-1] = 1 + out.get(-1, 0)
    return out


def nextprime(n: int, proved: bool = True) -> int:
    """Smallest prime strictly greater than ``n``."""
    return _decimal_to_int(nt_nextprime(int(n), proved))


def totient(n: int) -> int:
    """Euler totient φ(n) for integers n ≥ 1."""
    return _decimal_to_int(nt_totient(int(n)))


def jacobi_symbol(a: int, n: int) -> int:
    """Jacobi symbol (a | n) for odd integers n > 1."""
    return nt_jacobi(int(a), int(n))


def nthroot_mod(a: int, k: int, m: int) -> int:
    """Some integer ``x`` with ``pow(x, k, m) == a % m`` for prime modulus ``m``."""
    return _decimal_to_int(nt_nthroot_mod(int(a), int(k), int(m)))


def discrete_log(residue: int, base: int, modulus: int) -> int:
    """Exponent ``e`` with ``pow(base, e, modulus) == residue % modulus`` (prime ``modulus``)."""
    return _decimal_to_int(nt_discrete_log(int(residue), int(base), int(modulus)))
