#!/usr/bin/env python3
"""V2-22 — Discrete symbolic products (∏).

Demonstrates factorial via ∏ k from 1 to n and the Wallis-style partial identity
∏_{k=2}^n (1 - 1/k²) simplifying to (n+1)/(2n).

Run from the repository root (see :file:`pytest.ini` pythonpath)::

    PYTHONPATH=python python examples/products.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Dev layout: full `alkahest` package lives under `python/` (see CONTRIBUTING.md).
_root = Path(__file__).resolve().parents[1]
_pp = str(_root / "python")
sys.path.insert(0, _pp)

import alkahest as ah  # noqa: E402


def main() -> None:
    pool = ah.ExprPool()
    k = pool.symbol("k")
    n = pool.symbol("n")
    factorial_expr = ah.Product(k, (k, pool.integer(1), n)).doit().value
    print("factorial-style product simplifies to Gamma(n+1) shape:")
    print(ah.simplify(factorial_expr).value)

    two = pool.integer(2)
    kp2 = k ** 2
    term = ah.simplify(((k + pool.integer(-1)) * (k + pool.integer(1))) / kp2).value
    wallis_partial = ah.simplify(ah.product_definite(term, k, two, n).value).value
    print("\nWallis-style ∏_{k=2}^n (1 - 1/k²) →")
    print(wallis_partial)


if __name__ == "__main__":
    main()
