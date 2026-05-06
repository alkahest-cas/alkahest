"""V3-2 — Non-commutative Pauli and Clifford-style generators.

Uses ``pool.symbol(..., commutative=False)`` so ``A*B`` and ``B*A`` stay
distinct, and ``ak.simplify_pauli`` / ``simplify_clifford_orthogonal``
for registered product rules.

Run from the repository root (same layout as pytest ``pythonpath = python``)::

    PYTHONPATH=python python examples/noncommutative.py
"""

from __future__ import annotations

import alkahest as ak


def main() -> None:
    pool = ak.ExprPool()
    sx = pool.symbol("sx", "complex", commutative=False)
    sy = pool.symbol("sy", "complex", commutative=False)
    sz = pool.symbol("sz", "complex", commutative=False)
    ab = sx * sy
    ba = sy * sx
    print("sx*sy != sy*sx:", ab != ba)
    r = ak.simplify_pauli(ab)
    print("simplify_pauli(sx*sy):", r.value)

    e1 = pool.symbol("cliff_e1", commutative=False)
    e2 = pool.symbol("cliff_e2", commutative=False)
    r2 = ak.simplify_clifford_orthogonal(e1 * e2)
    print("cliff_e1*cliff_e2 ->", r2.value)


if __name__ == "__main__":
    main()
