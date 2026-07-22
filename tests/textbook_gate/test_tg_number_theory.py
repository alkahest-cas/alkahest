"""Textbook gate — elementary number theory.

First-course discrete-math number theory: prime factorization, primality
testing, `nextprime`, Euler's totient, the Jacobi symbol, discrete logarithms,
and modular n-th roots. Everything in `alkahest.number_theory` takes and
returns plain Python `int`/`bool`/`dict` — no `ExprPool`/`Expr` involved, so
unlike the rest of the textbook gate this file needs no fixtures and no
numeric-tolerance helpers from `tests/_tg_helpers.py`. See
`tests/textbook_gate/README.md` for the general verification philosophy.

**Notable finding while writing this file:** the public `discrete_log`
wrapper's parameter order is `discrete_log(residue, base, modulus)` — *not*
`discrete_log(base, residue, modulus)`. Calling it with `(base, residue,
modulus)` either raises `NumberTheoryError` ("no discrete logarithm or
modular root exists") or silently returns an exponent that does **not**
satisfy `pow(base, e, modulus) == residue`, depending on the inputs. The
correct order is confirmed by `alkahest/number_theory.py`'s own docstring:
``discrete_log(residue, base, modulus)`` returns the exponent ``e`` with
``pow(base, e, modulus) == residue % modulus``. Every case below calls it as
`nt.discrete_log(residue, base, modulus)` and self-verifies with
`pow(base, e, modulus) == residue % modulus`.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from alkahest import number_theory as nt

# --- factorization -----------------------------------------------------------


def _reconstruct(factors: dict[int, int]) -> int:
    n = 1
    for p, e in factors.items():
        assert nt.isprime(p), f"factorint returned non-prime factor {p}"
        n *= p**e
    return n


def test_factorint_single_prime():
    factors = nt.factorint(97)
    assert factors == {97: 1}
    assert _reconstruct(factors) == 97


def test_factorint_prime_power():
    factors = nt.factorint(128)  # 2^7
    assert factors == {2: 7}
    assert _reconstruct(factors) == 128


def test_factorint_product_of_distinct_primes():
    n = 2 * 3 * 5 * 7
    factors = nt.factorint(n)
    assert factors == {2: 1, 3: 1, 5: 1, 7: 1}
    assert _reconstruct(factors) == n


def test_factorint_highly_composite():
    n = 720
    factors = nt.factorint(n)
    assert _reconstruct(factors) == n
    assert factors == {2: 4, 3: 2, 5: 1}


# --- primality ---------------------------------------------------------------


def test_isprime_small_primes():
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        assert nt.isprime(p), f"{p} should be prime"


def test_isprime_small_composites():
    for c in [1, 4, 6, 8, 9, 10, 12, 15, 21, 25]:
        assert not nt.isprime(c), f"{c} should not be prime"


def test_isprime_known_larger_primes():
    assert nt.isprime(97)
    assert nt.isprime(101)
    assert not nt.isprime(100)


def test_isprime_carmichael_number_561():
    """561 = 3*11*17 is the smallest Carmichael number — composite, but a
    Fermat-test stress case since it passes `a^560 == 1 (mod 561)` for every
    `a` coprime to it. A proper primality test must still reject it."""
    assert nt.isprime(561) is False
    assert nt.factorint(561) == {3: 1, 11: 1, 17: 1}


# --- nextprime -----------------------------------------------------------------


def test_nextprime_from_14():
    assert nt.nextprime(14) == 17


def test_nextprime_from_100():
    assert nt.nextprime(100) == 101


def test_nextprime_from_2():
    assert nt.nextprime(2) == 3


# --- Euler's totient -----------------------------------------------------------


def test_totient_prime():
    p = 13
    assert nt.totient(p) == p - 1


def test_totient_prime_squared():
    # phi(p^2) = p^2 - p
    assert nt.totient(9) == 6  # 3^2


def test_totient_product_of_two_distinct_primes():
    p, q = 3, 5
    assert nt.totient(p * q) == (p - 1) * (q - 1)


def test_totient_composite_repeated_factor():
    assert nt.totient(8) == 4  # 2^3: phi = 2^3 - 2^2 = 4


# --- Jacobi symbol ---------------------------------------------------------------


def test_jacobi_symbol_quadratic_residue():
    # 2 is a QR mod 7 (3^2 = 9 = 2 mod 7)
    assert nt.jacobi_symbol(2, 7) == 1


def test_jacobi_symbol_quadratic_nonresidue():
    # 3 is not a QR mod 7 (squares mod 7 are {0,1,2,4})
    assert nt.jacobi_symbol(3, 7) == -1


def test_jacobi_symbol_of_one_is_always_one():
    for n in [3, 7, 9, 11, 15]:
        assert nt.jacobi_symbol(1, n) == 1


def test_jacobi_symbol_zero_when_not_coprime():
    # gcd(3, 9) = 3 != 1, so the Jacobi symbol is 0 by convention
    assert nt.jacobi_symbol(3, 9) == 0
    assert math.gcd(3, 9) != 1


# --- discrete logarithm -------------------------------------------------------
#
# nt.discrete_log(residue, base, modulus) returns e with
# pow(base, e, modulus) == residue % modulus (see module docstring above for
# the parameter-order finding). Every case below is verified fully generally
# via that identity, rather than hardcoding an expected exponent.


def test_discrete_log_mod_11():
    base, modulus = 2, 11
    residue = pow(base, 3, modulus)  # = 8
    e = nt.discrete_log(residue, base, modulus)
    assert pow(base, e, modulus) == residue % modulus


def test_discrete_log_mod_23():
    base, modulus = 7, 23
    residue = pow(base, 5, modulus)
    e = nt.discrete_log(residue, base, modulus)
    assert pow(base, e, modulus) == residue % modulus


def test_discrete_log_identity_residue_is_exponent_zero():
    base, modulus = 5, 13
    e = nt.discrete_log(1, base, modulus)
    assert pow(base, e, modulus) == 1 % modulus


# --- nth root mod p ------------------------------------------------------------
#
# nthroot_mod(a, k, m) only supports k=2 or gcd(k, m-1) == 1 (documented
# restriction, confirmed empirically below, not a bug). Verified via
# pow(result, k, m) == a % m, a fully general self-consistent check.


def test_nthroot_mod_square_root():
    a, m = 2, 7
    x = nt.nthroot_mod(a, 2, m)
    assert pow(x, 2, m) == a % m


def test_nthroot_mod_higher_root_with_coprime_exponent():
    # m=11, m-1=10; k=3 has gcd(3,10)=1, so this satisfies the restriction.
    a, k, m = 4, 3, 11
    assert math.gcd(k, m - 1) == 1
    x = nt.nthroot_mod(a, k, m)
    assert pow(x, k, m) == a % m


def test_nthroot_mod_restriction_rejects_unsupported_exponent():
    """k=3 against m=7 (m-1=6) has gcd(3,6)=3 != 1 and k != 2 — outside the
    documented support, so alkahest correctly refuses rather than silently
    returning a wrong/nonexistent root."""
    assert math.gcd(3, 7 - 1) != 1
    with pytest.raises(ak.NumberTheoryError):
        nt.nthroot_mod(2, 3, 7)
