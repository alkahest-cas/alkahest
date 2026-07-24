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


def _positive_log_case(pool, builder):
    """Run ``builder(x[, y])`` under explicit positivity assumptions."""
    x = pool.symbol("x")
    y = pool.symbol("y")
    assumptions = alkahest.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    assumptions.refine(pool.gt(y, pool.integer(0)))
    return assumptions.simplify(builder(x, y))


def _log_of_product_case(pool):
    """log(x*y) -> log(x) + log(y), certified under explicit x > 0, y > 0.

    The default `log_exp_rules()` set omits the expand-style `log_of_product`
    rewrite (it would oscillate against `sum_of_logs`), so this case goes
    through the colored e-graph's conditional `log_of_product_positive` rule
    instead, reached via `alkahest.Assumptions`.
    """
    return _positive_log_case(pool, lambda x, y: alkahest.log(x * y))


def _exp_of_log_case(pool):
    """exp(log(x)) -> x under x > 0."""
    x = pool.symbol("x")
    assumptions = alkahest.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    return assumptions.simplify(alkahest.exp(alkahest.log(x)))


def _log_of_pow_case(pool):
    """log(x^3) -> 3*log(x) under x > 0."""
    x = pool.symbol("x")
    assumptions = alkahest.Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    return assumptions.simplify(alkahest.log(x**3))


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
        _log_of_pow_case,
    ),
    (
        "exp_of_log",
        "exp_of_log",
        _exp_of_log_case,
    ),
    (
        "log_of_product",
        "log_of_product_positive",
        _log_of_product_case,
    ),
    # Chain rule for unary composites f(x^n), f ∈ {sin, cos, exp}, n ≥ 2.
    (
        "diff_chain_sin_x_squared",
        "diff_sin",
        lambda pool: alkahest.diff(alkahest.sin(pool.symbol("x") ** 2), pool.symbol("x")),
    ),
    (
        "diff_chain_exp_x_squared",
        "diff_exp",
        lambda pool: alkahest.diff(alkahest.exp(pool.symbol("x") ** 2), pool.symbol("x")),
    ),
    (
        "diff_chain_cos_x_squared",
        "diff_cos",
        lambda pool: alkahest.diff(alkahest.cos(pool.symbol("x") ** 2), pool.symbol("x")),
    ),
    (
        "diff_chain_sin_x_cubed",
        "diff_sin",
        lambda pool: alkahest.diff(alkahest.sin(pool.symbol("x") ** 3), pool.symbol("x")),
    ),
    # Indefinite integrals, certified via the FTC derivative relation
    # `deriv (fun x => F) x = f` (Part A). The recorded step is the integration
    # rule; `to_lean` differentiates the antiderivative and certifies that.
    (
        "int_cos",
        "int_cos",
        lambda pool: alkahest.integrate(alkahest.cos(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "int_sin",
        "int_sin",
        lambda pool: alkahest.integrate(alkahest.sin(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "int_exp",
        "int_exp",
        lambda pool: alkahest.integrate(alkahest.exp(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "int_power_x_squared",
        "int_power_rule",
        lambda pool: alkahest.integrate(pool.symbol("x") ** 2, pool.symbol("x")),
    ),
    # Definite integrals, certified via the second fundamental theorem of
    # calculus for interval integrals:
    #   ∫ x in a..b, f x = F b - F a
    # discharged by `intervalIntegral.integral_eq_sub_of_hasDerivAt` with a
    # `HasDerivAt` witness on `Set.uIcc a b` and an `IntervalIntegrable` side
    # condition. The recorded step is `fundamental_theorem_of_calculus`; the
    # emitter builds the antiderivative + FTC proof for the certifiable fragment
    # (pointwise sin/cos/exp of the variable, integer powers xⁿ).
    (
        "int_def_cos_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            alkahest.cos(pool.symbol("x")), pool.symbol("x"), pool.integer(0), pool.integer(1)
        ),
    ),
    (
        "int_def_sin_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            alkahest.sin(pool.symbol("x")), pool.symbol("x"), pool.integer(0), pool.integer(1)
        ),
    ),
    (
        "int_def_exp_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            alkahest.exp(pool.symbol("x")), pool.symbol("x"), pool.integer(0), pool.integer(1)
        ),
    ),
    (
        "int_def_x_squared_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            pool.symbol("x") ** 2, pool.symbol("x"), pool.integer(0), pool.integer(1)
        ),
    ),
    # `Real.deriv_log` holds unconditionally (no positivity hypothesis needed).
    (
        "diff_log",
        "diff_log",
        lambda pool: alkahest.diff(alkahest.log(pool.symbol("x")), pool.symbol("x")),
    ),
    # `Real.hasDerivAt_sqrt` needs `x ≠ 0`; upgraded to an explicit
    # `(x : ℝ) (hx : 0 < x)` binder, mirroring #236's positivity mechanism.
    (
        "diff_sqrt",
        "diff_sqrt",
        lambda pool: alkahest.diff(alkahest.sqrt(pool.symbol("x")), pool.symbol("x")),
    ),
    # `tan` is dispatched through the generic `diff_primitive_registry` rule;
    # mapped to `Real.hasDerivAt_tan` + `Real.inv_one_add_tan_sq` (needs
    # `cos x ≠ 0`) to reconcile Alkahest's `1 + tan²x` form.
    (
        "diff_tan",
        "diff_primitive_registry",
        lambda pool: alkahest.diff(alkahest.tan(pool.symbol("x")), pool.symbol("x")),
    ),
    # Generalized power rule with chain: `d/dx sin(x)² = 2 sin x cos x`, via
    # `HasDerivAt.pow` — unconditional.
    (
        "diff_power_of_primitive_sin_squared",
        "power_rule",
        lambda pool: alkahest.diff(alkahest.sin(pool.symbol("x")) ** 2, pool.symbol("x")),
    ),
    # `d/dx (1 / sin x)`, via `HasDerivAt.inv`; needs `sin x ≠ 0`.
    (
        "diff_inv_of_primitive_one_over_sin",
        "power_rule",
        lambda pool: alkahest.diff(alkahest.sin(pool.symbol("x")) ** -1, pool.symbol("x")),
    ),
    # `d/dx (sin x / cos x)`, via `HasDerivAt.mul` + `HasDerivAt.inv`; needs
    # `cos x ≠ 0`. Also exercises the `collect_mul_factors:
    # cos x * (cos x)⁻¹ = 1` cleanup step, closed via the nonzero-hypothesis
    # `field_simp` upgrade rather than the (unsound here) bare `ring`.
    (
        "diff_quotient_sin_over_cos",
        "product_rule",
        lambda pool: alkahest.diff(
            alkahest.sin(pool.symbol("x")) / alkahest.cos(pool.symbol("x")), pool.symbol("x")
        ),
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
