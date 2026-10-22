#!/usr/bin/env python3
"""V2-22 — Discrete symbolic products (∏).

Demonstrates factorial via ∏ k from 1 to n and the Wallis-style partial identity
∏_{k=2}^n (1 - 1/k²) simplifying to (n+1)/(2n).

Requires an installed alkahest (e.g. ``maturin develop --release`` in a venv)::

    python examples/products.py
"""

from __future__ import annotations

import alkahest as ah


def main() -> None:
    pool = ah.ExprPool()
    k = pool.symbol("k")
    n = pool.symbol("n")
    factorial_expr = ah.Product(k, (k, pool.integer(1), n)).doit().value
    print("factorial-style product simplifies to Gamma(n+1) shape:")
    print(ah.simplify(factorial_expr).value)

    two = pool.integer(2)
    kp2 = k**2
    term = ah.simplify(((k + pool.integer(-1)) * (k + pool.integer(1))) / kp2).value
    wallis_partial = ah.simplify(ah.product_definite(term, k, two, n).value).value
    print("\nWallis-style ∏_{k=2}^n (1 - 1/k²) →")
    print(wallis_partial)


if __name__ == "__main__":
    main()
