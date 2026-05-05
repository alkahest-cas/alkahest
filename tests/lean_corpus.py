#!/usr/bin/env python3
"""
Lean corpus generator — V5-8.

Generates .lean proof files for a curated set of algebraic identities
and writes them to the output directory. Used by the Lean CI job.

Usage::

    python tests/lean_corpus.py --output /tmp/lean_proofs/
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest

IDENTITIES = [
    # (description, expr_builder)
    ("add_zero", lambda pool: pool.symbol("x") + pool.integer(0)),
    ("mul_one", lambda pool: pool.symbol("x") * pool.integer(1)),
    ("mul_zero", lambda pool: pool.symbol("x") * pool.integer(0)),
    ("const_fold_2_plus_3", lambda pool: pool.integer(2) + pool.integer(3)),
    ("const_fold_3_times_4", lambda pool: pool.integer(3) * pool.integer(4)),
    ("pow_one", lambda pool: pool.symbol("x") ** 1),
    (
        "sin_neg_x",
        lambda pool: pool.func("sin", [pool.integer(-1) * pool.symbol("x")]),
    ),
    (
        "cos_neg_x",
        lambda pool: pool.func("cos", [pool.integer(-1) * pool.symbol("x")]),
    ),
]


def generate_proof(name: str, expr_builder, pool) -> str:
    """Generate a Lean proof for one identity."""
    expr = expr_builder(pool)
    return alkahest.to_lean(expr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lean proofs for Alkahest identities"
    )
    parser.add_argument(
        "--output", default=".", help="Output directory for .lean files"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pool = alkahest.ExprPool()
    success = 0
    for name, builder in IDENTITIES:
        try:
            lean_src = generate_proof(name, builder, pool)
            out_path = os.path.join(args.output, f"{name}.lean")
            with open(out_path, "w") as f:
                f.write(lean_src)
            print(f"Generated: {out_path}")
            success += 1
        except Exception as e:
            print(f"ERROR generating {name}: {e}", file=sys.stderr)

    print(f"\n{success}/{len(IDENTITIES)} proofs generated in {args.output}")
    return 0 if success == len(IDENTITIES) else 1


if __name__ == "__main__":
    sys.exit(main())
