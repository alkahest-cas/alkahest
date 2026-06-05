from __future__ import annotations

from typing import TYPE_CHECKING

import alkahest as ak
from alkahest import Expr, ExprPool

if TYPE_CHECKING:
    from random import Random

TIERS = {
    0: "rational",
    1: "exp_log",
    2: "algebraic_coeff",
    3: "rational_exp",
    4: "nested_tower",
}


def random_elementary(pool: ExprPool, tier: int, rng: Random) -> Expr:
    """Build a random antiderivative *F* at the given Risch tier.

    The caller computes ``f = diff(F, x)`` to obtain the integrand.
    """
    x = pool.symbol("x")
    if tier == 0:
        return _rational(pool, x, rng)
    if tier == 1:
        return _exp_log(pool, x, rng)
    if tier == 2:
        return _algebraic_coeff(pool, x, rng)
    raise NotImplementedError(f"Tier {tier} grammar not yet implemented")


def _rational(pool: ExprPool, x: Expr, rng: Random) -> Expr:
    """Random polynomial in *x* with small integer coefficients."""
    degree = rng.randint(1, 4)
    terms: list[Expr] = []
    zero = pool.integer(0)
    for i in range(degree + 1):
        c = pool.integer(rng.randint(-3, 3))
        if c == zero:
            continue
        terms.append(c * x ** pool.integer(i))
    if not terms:
        return x
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def _exp_log(pool: ExprPool, x: Expr, rng: Random) -> Expr:
    inner = _rational(pool, x, rng)
    choice = rng.choice(["exp", "log", "product"])
    if choice == "exp":
        return ak.exp(inner)
    if choice == "log":
        # Keep the logarithm argument positive on a typical evaluation domain.
        arg = ak.simplify(inner * inner + pool.integer(1)).value
        return ak.log(arg)
    return _rational(pool, x, rng) * ak.exp(inner)


def _algebraic_coeff(pool: ExprPool, x: Expr, rng: Random) -> Expr:
    d = rng.choice([2, 3, 5, 7])
    sqrt_d = pool.integer(d) ** (pool.integer(1) / pool.integer(2))
    return sqrt_d * _rational(pool, x, rng) + _exp_log(pool, x, rng)
