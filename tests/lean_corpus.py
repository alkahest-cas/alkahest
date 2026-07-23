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


def _log_of_product_case(pool):
    """log(x*y) -> log(x) + log(y), certified under explicit x > 0, y > 0.

    The default `log_exp_rules()` set omits the expand-style `log_of_product`
    rewrite (it would oscillate against `sum_of_logs`), so this case goes
    through the colored e-graph's conditional `log_of_product_positive` rule
    instead, reached via `alkahest.Assumptions`.
    """
    x = pool.symbol("x")
    y = pool.symbol("y")
    assumptions = alkahest.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    assumptions.refine(pool.gt(y, pool.integer(0)))
    return assumptions.simplify(alkahest.log(x * y))


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
        "diff_x_squared",
        "diff_univariate_poly",
        lambda pool: alkahest.diff(pool.symbol("x") ** 2, pool.symbol("x")),
    ),
    (
        "diff_sin",
        "diff_sin",
        lambda pool: alkahest.diff(alkahest.sin(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "diff_sum_sin_cos",
        "sum_rule",
        lambda pool: alkahest.diff(
            alkahest.sin(pool.symbol("x")) + alkahest.cos(pool.symbol("x")),
            pool.symbol("x"),
        ),
    ),
    (
        "diff_product_sin_exp",
        "product_rule",
        lambda pool: alkahest.diff(
            alkahest.sin(pool.symbol("x")) * alkahest.exp(pool.symbol("x")),
            pool.symbol("x"),
        ),
    ),
    (
        "log_of_exp",
        "log_of_exp",
        lambda pool: alkahest.simplify_log_exp(alkahest.log(alkahest.exp(pool.symbol("x")))),
    ),
    (
        "tan_expand",
        "tan_expand",
        lambda pool: alkahest.simplify_trig(alkahest.tan(pool.symbol("x"))),
    ),
    (
        "log_of_pow",
        "log_of_pow",
        lambda pool: alkahest.simplify_log_exp(alkahest.log(pool.symbol("x") ** 3)),
    ),
    (
        "exp_of_log",
        "exp_of_log",
        lambda pool: alkahest.simplify_log_exp(alkahest.exp(alkahest.log(pool.symbol("x")))),
    ),
    (
        "log_of_product",
        "log_of_product_positive",
        _log_of_product_case,
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
