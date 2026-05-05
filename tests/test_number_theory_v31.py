"""V3-1 — Python surface for integer number theory."""

from __future__ import annotations

import pytest
from alkahest import NumberTheoryError
from alkahest.number_theory import (
    DirichletChi,
    discrete_log,
    factorint,
    isprime,
    jacobi_symbol,
    nextprime,
    nthroot_mod,
    totient,
)

M127 = 2**127 - 1
F5 = 2**32 - 1


def test_isprime_mersenne_and_small() -> None:
    assert isprime(2)
    assert not isprime(1)
    assert isprime(M127)


def test_factorint_f5_matches_sympy_shape() -> None:
    fac = factorint(F5)
    assert fac[3] == 1 and fac[5] == 1 and fac[17] == 1 and fac[257] == 1 and fac[65537] == 1
    prod = 1
    for p, e in fac.items():
        prod *= pow(p, e)
    assert prod == F5


def test_factorint_zero_negative() -> None:
    assert factorint(0) == {0: 1}
    assert factorint(-12) == {-1: 1, 2: 2, 3: 1}


def test_nextprime_and_totient() -> None:
    assert nextprime(13) == 17
    assert totient(12) == 4


def test_jacobi() -> None:
    assert jacobi_symbol(2, 15) == 1


def test_nthroot_sqrt_mod_prime() -> None:
    x = nthroot_mod(144, 2, 401)
    assert (x * x) % 401 == 144 % 401


def test_discrete_log() -> None:
    assert discrete_log(13, 3, 17) == 4
    assert pow(3, 4, 17) == 13


def test_dirichlet_phi() -> None:
    chi = DirichletChi(15)
    assert chi.conductor == "15"
    assert chi.eval(14) == -1
    assert chi.eval(3) == 0


def test_nthroot_unsupported_raises() -> None:
    # 3011 ≡ 1 (mod 5) ⇒ gcd(5, 3010) ≠ 1
    with pytest.raises(NumberTheoryError) as ei:
        nthroot_mod(42, 5, 3011)
    assert "E-NT-" in ei.value.code
