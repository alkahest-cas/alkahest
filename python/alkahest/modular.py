"""alkahest.modular — Modular / CRT framework (V2-1).

Public API
----------
reduce_mod(poly, p)
    Reduce a :class:`~alkahest.MultiPoly` over ℤ to F_p = ℤ/pℤ.
    Returns a :class:`MultiPolyFp`.

lift_crt(images)
    Reconstruct a :class:`~alkahest.MultiPoly` over ℤ from a list of
    ``(MultiPolyFp, prime)`` pairs via the Chinese Remainder Theorem.

rational_reconstruction(n, m)
    Find a/b with b·n ≡ a (mod m) and small norm.
    Returns ``(a, b)`` as Python ints, or ``None``.

mignotte_bound(poly)
    Return the Cauchy–Mignotte coefficient bound as a Python ``int``.

select_lucky_prime(avoid_divisor, used)
    Return the smallest prime not in *used* that does not divide
    *avoid_divisor*.
"""

from __future__ import annotations

from .alkahest import (  # noqa: F401 – re-export the Rust class
    ModularError,
    MultiPolyFp,
    modular_lift_crt,
    modular_mignotte_bound,
    modular_rational_reconstruction,
    modular_reduce,
    modular_select_lucky_prime,
)

__all__ = [
    "MultiPolyFp",
    "ModularError",
    "reduce_mod",
    "lift_crt",
    "rational_reconstruction",
    "mignotte_bound",
    "select_lucky_prime",
]


def reduce_mod(poly, p: int) -> MultiPolyFp:
    """Reduce *poly* (a :class:`~alkahest.MultiPoly` over ℤ) to F_p = ℤ/pℤ.

    Parameters
    ----------
    poly:
        A :class:`~alkahest.MultiPoly` with integer coefficients.
    p:
        A prime modulus.  Raises :class:`ModularError` if ``p`` is not prime.

    Returns
    -------
    MultiPolyFp
        Polynomial with coefficients in ``[0, p)``.
    """
    return modular_reduce(poly, p)


def lift_crt(images: list[tuple[MultiPolyFp, int]]) -> object:
    """Reconstruct a polynomial over ℤ from modular images via CRT.

    Parameters
    ----------
    images:
        A list of ``(MultiPolyFp, prime)`` pairs.  All images must share the
        same variable ordering.

    Returns
    -------
    MultiPoly
        The reconstructed polynomial with coefficients centered in
        ``(-M/2, M/2]`` where ``M = p_1 · … · p_k``.
    """
    if not images:
        raise ValueError("images must be non-empty")
    polys = [img for img, _ in images]
    primes = [int(prime) for _, prime in images]
    return modular_lift_crt(polys, primes)


def rational_reconstruction(n: int, m: int) -> tuple[int, int] | None:
    """Find a/b with b·n ≡ a (mod m) and |a|, b ≤ ⌊√(m/2)⌋.

    Parameters
    ----------
    n:
        The modular representative (Python arbitrary-precision int).
    m:
        The modulus (Python arbitrary-precision int).

    Returns
    -------
    tuple[int, int] or None
        ``(a, b)`` in lowest terms with ``b > 0``, or ``None`` if the prime
        product is too small to uniquely determine the rational.
    """
    result = modular_rational_reconstruction(str(n), str(m))
    if result is None:
        return None
    a_str, b_str = result
    return int(a_str), int(b_str)


def mignotte_bound(poly) -> int:
    """Return the Cauchy–Mignotte coefficient bound for *poly*.

    The CRT product of primes must exceed ``2 * mignotte_bound(poly)`` to
    guarantee correct reconstruction.
    """
    return int(modular_mignotte_bound(poly))


def select_lucky_prime(avoid_divisor: int = 0, used: list[int] | None = None) -> int:
    """Return the smallest prime not in *used* that does not divide *avoid_divisor*.

    Parameters
    ----------
    avoid_divisor:
        Integer content or leading coefficient to avoid (arbitrary-precision).
        Pass ``0`` to apply no divisibility constraint.
    used:
        Primes already consumed.  Default is empty.
    """
    if used is None:
        used = []
    return modular_select_lucky_prime(str(avoid_divisor), [int(p) for p in used])
