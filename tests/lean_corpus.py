#!/usr/bin/env python3
"""
Lean corpus generator — V5-8.

Generates a strict, no-admission Lean proof corpus for a curated set of
deterministic derivations. Used by the Lean CI job.

Usage::

    python tests/lean_corpus.py --output /tmp/lean_proofs/
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest

STRICT_CASES = [
    # (name, expected_rule, DerivedResult builder)
    (
        "add_zero",
        "add_zero",
        lambda pool: alkahest.simplify(pool.symbol("x") + pool.integer(0)),
    ),
    (
        "mul_one",
        "mul_one",
        lambda pool: alkahest.simplify(pool.symbol("x") * pool.integer(1)),
    ),
    (
        "mul_zero",
        "mul_zero",
        lambda pool: alkahest.simplify(pool.symbol("x") * pool.integer(0)),
    ),
    (
        "const_fold_2_plus_3",
        "const_fold",
        lambda pool: alkahest.simplify(pool.integer(2) + pool.integer(3)),
    ),
    (
        "const_fold_3_times_4",
        "const_fold",
        lambda pool: alkahest.simplify(pool.integer(3) * pool.integer(4)),
    ),
    (
        "pow_one",
        "pow_one",
        lambda pool: alkahest.simplify(pool.symbol("x") ** 1),
    ),
    (
        "diff_x_cubed",
        "diff_univariate_poly",
        lambda pool: alkahest.diff(pool.symbol("x") ** 3, pool.symbol("x")),
    ),
    (
        "diff_sin",
        "diff_sin",
        lambda pool: alkahest.diff(alkahest.sin(pool.symbol("x")), pool.symbol("x")),
    ),
]
FORBIDDEN_TOKENS = ("sorry", "admit", "axiom")


def generate_proof(name: str, expected_rule: str, result_builder, pool) -> str:
    """Generate one strict Lean proof from a non-empty expected derivation."""
    result = result_builder(pool)
    rules = [step["rule"] for step in result.steps]
    if not rules:
        raise ValueError(f"{name}: derivation log is empty")
    if expected_rule not in rules:
        raise ValueError(f"{name}: expected rule {expected_rule!r}, got {rules!r}")

    lean_src = alkahest.to_lean(result)
    for token in FORBIDDEN_TOKENS:
        if token in lean_src:
            raise ValueError(f"{name}: generated Lean source contains {token!r}")
    return lean_src


def main():
    parser = argparse.ArgumentParser(description="Generate Lean proofs for Alkahest identities")
    parser.add_argument("--output", default=".", help="Output directory for .lean files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pool = alkahest.ExprPool()
    success = 0
    for name, expected_rule, builder in STRICT_CASES:
        try:
            lean_src = generate_proof(name, expected_rule, builder, pool)
            out_path = os.path.join(args.output, f"{name}.lean")
            with open(out_path, "w") as f:
                f.write(lean_src)
            print(f"Generated: {out_path}")
            success += 1
        except Exception as e:
            print(f"ERROR generating {name}: {e}", file=sys.stderr)

    print(f"\n{success}/{len(STRICT_CASES)} strict proofs generated in {args.output}")
    return 0 if success == len(STRICT_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
