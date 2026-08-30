#!/usr/bin/env python3
"""
Lean Tendsto corpus generator.

Generates Filter.Tendsto certificates for the recognised `x → +∞` fragment
wired through ``alkahest.limit``. Used by the Lean CI job. A separate
generator from ``tests/lean_corpus.py`` because Tendsto certificates have
no rewrite log (the strict corpus requires a recorded rule).

Usage::

    python tests/lean_tendsto_corpus.py --output /tmp/lean_tendsto/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest

FORBIDDEN_TOKENS = ("sorry", "admit", "axiom")


def _exp_neg_at_top(pool):
    x = pool.symbol("x")
    return alkahest.limit(alkahest.exp(-x), x, pool.pos_infinity())


def _exp_at_top(pool):
    x = pool.symbol("x")
    return alkahest.limit(alkahest.exp(x), x, pool.pos_infinity())


# (name, expected_tactic_fragment, DerivedResult builder)
# Builders must emit a non-empty certificate; withheld shapes belong in
# Python unit tests, not this typecheck corpus.
STRICT_CASES = [
    (
        "tendsto_exp_neg_atTop_nhds_zero",
        "tendsto_exp_neg_atTop_nhds_zero",
        _exp_neg_at_top,
    ),
    (
        "tendsto_exp_atTop",
        "tendsto_exp_atTop",
        _exp_at_top,
    ),
]


def generate_proof(name: str, expected_tactic: str, result_builder, pool) -> str:
    """Generate one Tendsto Lean proof from a recognised limit pattern."""
    result = result_builder(pool)
    lean_src = result.certificate or alkahest.to_lean(result)
    if not lean_src:
        raise ValueError(f"{name}: Tendsto certificate was withheld")
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
        description="Generate Filter.Tendsto Lean proofs for recognised limit() patterns"
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

    print(f"\n{success}/{len(STRICT_CASES)} Tendsto proofs generated in {args.output}")
    return 0 if success == len(STRICT_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
