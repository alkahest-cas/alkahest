#!/usr/bin/env python3
"""
Lean Gosper corpus generator.

Generates sorry-free Finset.sum certificates for Gosper telescopes (and the
one Finset.prod factorial identity). Used by the Lean CI job. A separate
generator from ``tests/lean_corpus.py`` because Gosper certificates are a
different statement from the rewrite log (emitting ``F = G`` would be false).

Usage::

    python tests/lean_gosper_corpus.py --output /tmp/lean_gosper/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest

FORBIDDEN_TOKENS = ("sorry", "admit", "axiom")


def _sum_k_indefinite(pool):
    k = pool.symbol("k")
    return alkahest.sum_indefinite(k, k)


def _sum_constant(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.sum_definite(pool.integer(5), k, pool.integer(1), n)


def _sum_k(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.sum_definite(k, k, pool.integer(1), n)


def _sum_k_squared(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.sum_definite(k**2, k, pool.integer(1), n)


def _sum_k_cubed(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.sum_definite(k**3, k, pool.integer(1), n)


def _sum_odd(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.sum_definite(pool.integer(2) * k + pool.integer(1), k, pool.integer(1), n)


def _sum_k_one_to_ten(pool):
    k = pool.symbol("k")
    return alkahest.sum_definite(k, k, pool.integer(1), pool.integer(10))


def _sum_telescope_reciprocal(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.sum_definite(1 / (k * (k + pool.integer(1))), k, pool.integer(1), n)


def _sum_geometric_two(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.sum_definite(pool.integer(2) ** k, k, pool.integer(0), n)


def _prod_factorial(pool):
    k = pool.symbol("k")
    n = pool.symbol("n")
    return alkahest.product_definite(k, k, pool.integer(1), n)


# (name, expected_tactic_fragment, DerivedResult builder)
STRICT_CASES = [
    ("gosper_indefinite_k", "sum_range_sub", _sum_k_indefinite),
    ("gosper_sum_constant", "sum_range_sub", _sum_constant),
    ("gosper_sum_k", "sum_Ico_eq_sum_range", _sum_k),
    ("gosper_sum_k_squared", "sum_Ico_eq_sum_range", _sum_k_squared),
    ("gosper_sum_k_cubed", "sum_Ico_eq_sum_range", _sum_k_cubed),
    ("gosper_sum_odd", "sum_Ico_eq_sum_range", _sum_odd),
    ("gosper_sum_k_1_10", "sum_range_sub", _sum_k_one_to_ten),
    (
        "gosper_sum_reciprocal_telescope",
        "sum_Ico_eq_sum_range",
        _sum_telescope_reciprocal,
    ),
    ("gosper_sum_geometric_two", "sum_range_sub", _sum_geometric_two),
    ("product_factorial", "prod_Ico_id_eq_factorial", _prod_factorial),
]


def generate_proof(name: str, expected_tactic: str, result_builder, pool) -> str:
    """Generate one Gosper/product Lean proof from a recognised shape."""
    result = result_builder(pool)
    lean_src = result.certificate or alkahest.to_lean(result)
    if not lean_src:
        raise ValueError(f"{name}: Gosper/product certificate was withheld")
    if expected_tactic not in lean_src:
        raise ValueError(f"{name}: expected tactic {expected_tactic!r} missing from certificate")
    for token in FORBIDDEN_TOKENS:
        if token in lean_src:
            raise ValueError(f"{name}: generated Lean source contains {token!r}")
    to_lean_src = alkahest.to_lean(result)
    if to_lean_src != lean_src:
        raise ValueError(f"{name}: to_lean(result) does not match .certificate")
    return lean_src


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Finset.sum / Finset.prod Lean proofs for Gosper telescopes"
    )
    parser.add_argument("--output", default=".", help="Output directory for .lean files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pool = alkahest.ExprPool()
    success = 0
    for name, expected_tactic, builder in STRICT_CASES:
        try:
            lean_src = generate_proof(name, expected_tactic, builder, pool)
            out_path = os.path.join(args.output, f"{name}.lean")
            with open(out_path, "w") as f:
                f.write(lean_src)
            print(f"Generated: {out_path}")
            success += 1
        except Exception as e:
            print(f"ERROR generating {name}: {e}", file=sys.stderr)

    print(f"\n{success}/{len(STRICT_CASES)} Gosper/product proofs generated in {args.output}")
    return 0 if success == len(STRICT_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
