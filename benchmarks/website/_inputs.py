"""Deterministic input generators shared by every system in a case.

Every system receives the *same* input, generated here as a Python-syntax
string from a fixed seed and parsed by each system itself (untimed).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fractions import Fraction


def rand_dense_univariate(deg: int, bits: int, seed: int, var: str = "x") -> str:
    rng = random.Random(seed)
    terms = []
    for k in range(deg + 1):
        c = rng.randint(-(2**bits), 2**bits)
        if k == deg and c == 0:
            c = 1
        if c:
            terms.append(f"({c})*{var}**{k}")
    return " + ".join(terms)


def rand_sparse_multivariate(nvars: int, nterms: int, maxdeg: int, coeff: int, seed: int, names: list[str]) -> str:
    """Random sparse polynomial: ``nterms`` monomials of total degree <= maxdeg."""
    rng = random.Random(seed)
    seen = set()
    terms = []
    while len(terms) < nterms:
        exps = [0] * nvars
        d = rng.randint(1, maxdeg)
        for _ in range(d):
            exps[rng.randrange(nvars)] += 1
        key = tuple(exps)
        if key in seen:
            continue
        seen.add(key)
        c = rng.randint(1, coeff) * rng.choice((-1, 1))
        mono = "*".join(f"{n}**{e}" for n, e in zip(names, exps) if e)
        terms.append(f"({c})*{mono}")
    terms.append(f"({rng.randint(1, coeff)})")
    return " + ".join(terms)


def legendre_integer(n: int, var: str = "x") -> str:
    """``2**n * P_n(x)`` as an integer-coefficient polynomial string."""
    # Rodrigues / explicit sum: P_n = 2^-n sum_k (-1)^k C(n,k) C(2n-2k,n) x^(n-2k)
    from math import comb

    terms = []
    for k in range(n // 2 + 1):
        c = (-1) ** k * comb(n, k) * comb(2 * n - 2 * k, n)
        terms.append(f"({c})*{var}**{n - 2 * k}")
    return " + ".join(terms)


def cyclic(n: int) -> tuple[list[str], list[str]]:
    xs = [f"x{i}" for i in range(n)]
    eqs = []
    for d in range(1, n):
        eqs.append(" + ".join("*".join(xs[(i + j) % n] for j in range(d)) for i in range(n)))
    eqs.append("*".join(xs) + " - 1")
    return eqs, xs


def katsura(n: int) -> tuple[list[str], list[str]]:
    """Katsura-n in n+1 unknowns u0..un (the usual benchmark convention)."""
    us = [f"u{i}" for i in range(n + 1)]

    def u(i: int) -> str | None:
        i = abs(i)
        return us[i] if i <= n else None

    eqs = [" + ".join(u(i) for i in range(-n, n + 1)) + " - 1"]
    for k in range(n):
        terms = []
        for i in range(-n, n + 1):
            a, b = u(i), u(k - i)
            if a is not None and b is not None:
                terms.append(f"{a}*{b}")
        eqs.append(" + ".join(terms) + f" - {us[k]}")
    return eqs, us


def frac_str(q: Fraction) -> str:
    return f"({q.numerator})/({q.denominator})" if q.denominator != 1 else f"({q.numerator})"
