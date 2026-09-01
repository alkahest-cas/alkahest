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


def _require_goal(result, fragment, label):
    """Pin the *printed* goal, not just the rule that produced it.

    Alkahest canonicalises commutative children by raw ``ExprId``, so which
    addend or factor a goal prints first depends on what else the pool interned
    — and a certificate whose goal does not match its cited Mathlib lemma is
    exactly the failure this corpus exists to catch. Cases below that
    deliberately construct one intern order assert it here, so that a change in
    canonicalisation fails the generator loudly instead of silently retesting
    the order that already worked.
    """
    src = result.certificate or ""
    if fragment not in src:
        raise ValueError(f"{label}: expected the goal to print {fragment!r}; got:\n{src}")
    return result


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


def _pythagorean_sin_first(_pool):
    """`sin² + cos² = 1` printed with `sin²` first (a fresh pool's order)."""
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    result = alkahest.simplify_trig(alkahest.sin(x) ** 2 + alkahest.cos(x) ** 2)
    return _require_goal(result, "(Real.sin ((x : ℝ))) ^ (2 : ℕ) + ", "pythagorean_sin_first")


def _pythagorean_cos_first(_pool):
    """The same identity printed with `cos²` first.

    Children of a commutative node are sorted by raw ``ExprId``, so interning
    ``cos(x)^2`` before the sum is built flips the printed order.
    """
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    alkahest.cos(x) ** 2
    result = alkahest.simplify_trig(alkahest.sin(x) ** 2 + alkahest.cos(x) ** 2)
    return _require_goal(result, "(Real.cos ((x : ℝ))) ^ (2 : ℕ) + ", "pythagorean_cos_first")


def _diff_exp_over_cos(_pool):
    """`d/dx (exp x / cos x)` with the numerator printed first."""
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    result = alkahest.diff(alkahest.exp(x) / alkahest.cos(x), x)
    return _require_goal(
        result, "(Real.exp ((x : ℝ)) * (Real.cos ((x : ℝ)))⁻¹)", "diff_exp_over_cos"
    )


def _diff_inv_first_cos_exp(_pool):
    """The same derivative with the *inverse* factor printed first.

    Interning `cos(x)⁻¹` first makes commutative canonicalisation put it on the
    left, which is the order `HasDerivAt.mul` has to be assembled in.
    """
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    alkahest.cos(x) ** -1
    result = alkahest.diff(alkahest.exp(x) / alkahest.cos(x), x)
    return _require_goal(
        result, "((Real.cos ((x : ℝ)))⁻¹ * Real.exp ((x : ℝ)))", "diff_inv_first_cos_exp"
    )


def _int_def_arctan_sq_first(_pool):
    """`∫₀¹ (x² + 1)⁻¹` — the `x²`-first intern order of `1 + x²`."""
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    result = alkahest.integrate(1 / (1 + x**2), x, pool.integer(0), pool.integer(1))
    return _require_goal(result, "((((x : ℝ)) ^ (2 : ℕ) + (1 : ℝ)))⁻¹", "int_def_arctan_sq_first")


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
    (
        "diff_sin_cos_x",
        "diff_sin",
        lambda pool: alkahest.diff(alkahest.sin(alkahest.cos(pool.symbol("x"))), pool.symbol("x")),
    ),
    (
        "diff_exp_neg_x",
        "diff_exp",
        lambda pool: alkahest.diff(alkahest.exp(-pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "diff_cos_two_x",
        "diff_cos",
        lambda pool: alkahest.diff(alkahest.cos(2 * pool.symbol("x")), pool.symbol("x")),
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
    # ∫ log x dx = x·log x − x, certified via the FTC derivative of F now that
    # the log/sqrt combine fragment closes `d/dx (x·log x − x)`.
    (
        "int_log",
        "int_log",
        lambda pool: alkahest.integrate(alkahest.log(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "int_cos_two_x",
        "u_substitution",
        lambda pool: alkahest.integrate(alkahest.cos(2 * pool.symbol("x")), pool.symbol("x")),
    ),
    # Definite integrals, certified via the second fundamental theorem of
    # calculus for interval integrals:
    #   ∫ x in a..b, f x = F b - F a
    # discharged by `intervalIntegral.integral_eq_sub_of_hasDerivAt` with a
    # `HasDerivAt` witness on `Set.uIcc a b` and an `IntervalIntegrable` side
    # condition. The recorded step is `fundamental_theorem_of_calculus`; the
    # emitter builds the antiderivative + FTC proof for the certifiable fragment
    # (pointwise sin/cos/exp of the variable or of a linear/affine argument,
    # integer powers xⁿ, (1+x²)⁻¹, and pointwise log when both endpoints are
    # strictly positive).
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
    (
        "int_def_cos_two_x_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            alkahest.cos(2 * pool.symbol("x")),
            pool.symbol("x"),
            pool.integer(0),
            pool.integer(1),
        ),
    ),
    # Expanded definite-integral fragment: finite sums and numeric-literal
    # constant multiples of the base pointwise/`xⁿ` family, certified via
    # `HasDerivAt.add`/`.const_mul`/`.mul_const` and the `IntervalIntegrable`
    # analogues composed on top of the same interval-FTC lemma.
    (
        "int_def_sum_sin_cos_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            alkahest.sin(pool.symbol("x")) + alkahest.cos(pool.symbol("x")),
            pool.symbol("x"),
            pool.integer(0),
            pool.integer(1),
        ),
    ),
    (
        "int_def_const_mul_cos_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            3 * alkahest.cos(pool.symbol("x")),
            pool.symbol("x"),
            pool.integer(0),
            pool.integer(1),
        ),
    ),
    (
        "int_def_neg_exp_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            -alkahest.exp(pool.symbol("x")),
            pool.symbol("x"),
            pool.integer(0),
            pool.integer(1),
        ),
    ),
    (
        "int_def_rational_coeff_x_squared_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            pool.rational(1, 2) * pool.symbol("x") ** 2,
            pool.symbol("x"),
            pool.integer(0),
            pool.integer(1),
        ),
    ),
    (
        "int_def_three_term_combo_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            pool.symbol("x") ** 2
            + alkahest.sin(pool.symbol("x"))
            + 3 * alkahest.cos(pool.symbol("x")),
            pool.symbol("x"),
            pool.integer(0),
            pool.integer(1),
        ),
    ),
    # Definite ∫_1^2 log x. IntervalIntegrable log needs 0 ∉ uIcc a b, so the
    # endpoints must be strictly positive (1 and 2, discharged by norm_num).
    # ∫_0^1 log stays withheld (singular at 0).
    (
        "int_def_log_1_2",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            alkahest.log(pool.symbol("x")),
            pool.symbol("x"),
            pool.integer(1),
            pool.integer(2),
        ),
    ),
    # `Real.deriv_log` holds unconditionally (no positivity hypothesis needed).
    (
        "diff_log",
        "diff_log",
        lambda pool: alkahest.diff(alkahest.log(pool.symbol("x")), pool.symbol("x")),
    ),
    # Combine fragment: log/sqrt inside product/sum, with `(hx : 0 < x)`.
    (
        "diff_product_x_log",
        "product_rule",
        lambda pool: alkahest.diff(
            pool.symbol("x") * alkahest.log(pool.symbol("x")), pool.symbol("x")
        ),
    ),
    (
        "diff_product_exp_log",
        "product_rule",
        lambda pool: alkahest.diff(
            alkahest.exp(pool.symbol("x")) * alkahest.log(pool.symbol("x")),
            pool.symbol("x"),
        ),
    ),
    (
        "diff_sum_log_x",
        "sum_rule",
        lambda pool: alkahest.diff(
            alkahest.log(pool.symbol("x")) + pool.symbol("x"), pool.symbol("x")
        ),
    ),
    (
        "diff_product_x_sqrt",
        "product_rule",
        lambda pool: alkahest.diff(
            pool.symbol("x") * alkahest.sqrt(pool.symbol("x")), pool.symbol("x")
        ),
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
    # Hyperbolic: unconditional on ℝ, same registry dispatch. Sums/products
    # join the everywhere-differentiable fragment (`Real.deriv_sinh` /
    # `Real.deriv_cosh` in the combine tactic).
    (
        "diff_sinh",
        "diff_primitive_registry",
        lambda pool: alkahest.diff(alkahest.sinh(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "diff_cosh",
        "diff_primitive_registry",
        lambda pool: alkahest.diff(alkahest.cosh(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "diff_sum_sinh_cosh",
        "sum_rule",
        lambda pool: alkahest.diff(
            alkahest.sinh(pool.symbol("x")) + alkahest.cosh(pool.symbol("x")),
            pool.symbol("x"),
        ),
    ),
    (
        "diff_product_exp_sinh",
        "product_rule",
        lambda pool: alkahest.diff(
            alkahest.exp(pool.symbol("x")) * alkahest.sinh(pool.symbol("x")),
            pool.symbol("x"),
        ),
    ),
    # Inverse trig: atan is unconditional; asin needs |x| < 1.
    (
        "diff_atan",
        "diff_primitive_registry",
        lambda pool: alkahest.diff(alkahest.atan(pool.symbol("x")), pool.symbol("x")),
    ),
    (
        "diff_asin",
        "diff_primitive_registry",
        lambda pool: alkahest.diff(alkahest.asin(pool.symbol("x")), pool.symbol("x")),
    ),
    # `tanh` has no Mathlib v4.9.0 `hasDerivAt_tanh`; constructed from
    # `hasDerivAt_sinh` / `hasDerivAt_cosh` via `HasDerivAt.div`, with
    # `cosh_sq_sub_sinh_sq` reconciling `1 - tanh²` and `1/cosh²`.
    (
        "diff_tanh",
        "diff_primitive_registry",
        lambda pool: alkahest.diff(alkahest.tanh(pool.symbol("x")), pool.symbol("x")),
    ),
    # ∫ (1+x²)⁻¹ dx = atan x, certified via the FTC derivative relation
    # once d/dx atan(x) intern-equals the integrand.
    (
        "int_inv_one_plus_x_squared",
        "rothstein_trager",
        lambda pool: alkahest.integrate(1 / (1 + pool.symbol("x") ** 2), pool.symbol("x")),
    ),
    # Definite interval-FTC: ∫₀¹ (1+x²)⁻¹ = arctan 1 − arctan 0 (not π/4).
    (
        "int_def_inv_one_plus_x_squared_0_1",
        "fundamental_theorem_of_calculus",
        lambda pool: alkahest.integrate(
            1 / (1 + pool.symbol("x") ** 2),
            pool.symbol("x"),
            pool.integer(0),
            pool.integer(1),
        ),
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
    # `d/dx x⁻¹` via `hasDerivAt_inv`; needs `x ≠ 0`. Pretty-printed `(x)⁻¹`.
    (
        "diff_x_inv",
        "power_rule",
        lambda pool: alkahest.diff(pool.symbol("x") ** -1, pool.symbol("x")),
    ),
    # `d/dx x⁻²` via `hasDerivAt_inv` then `HasDerivAt.pow 2`; needs `x ≠ 0`.
    # Pretty-printed `(x)⁻¹ ^ (2 : ℕ)`.
    (
        "diff_x_neg_two",
        "power_rule",
        lambda pool: alkahest.diff(pool.symbol("x") ** -2, pool.symbol("x")),
    ),
    # `∫ x⁻¹ dx = log x`, certified via FTC reuse of `Real.deriv_log`.
    (
        "int_x_inv",
        "log_rule",
        lambda pool: alkahest.integrate(pool.symbol("x") ** -1, pool.symbol("x")),
    ),
    # `∫ x⁻² dx = -x⁻¹`, certified via FTC reuse of `d/dx (-x⁻¹)`. The
    # antiderivative is a product; `product_rule` closes on the negative-power
    # combine fragment (`x ≠ 0`, `deriv_inv`), not the unconditional simp set.
    (
        "int_x_neg_two",
        "int_power_rule",
        lambda pool: alkahest.integrate(pool.symbol("x") ** -2, pool.symbol("x")),
    ),
    # `d/dx (-x⁻¹)` — the product_rule step the integral above reuses.
    (
        "diff_neg_x_inv",
        "product_rule",
        lambda pool: alkahest.diff(-(pool.symbol("x") ** -1), pool.symbol("x")),
    ),
    # --- Audit regressions (certificates that were emitted but did not
    # --- typecheck; see the module docstring of `alkahest_cas::lean`).
    #
    # `d/dx (xⁿ log x)` for `n ≥ 2`: `field_simp` discharges the equation but
    # leaves its own `True ∨ x = 1 ∨ x = -1` side goal, which `ring` cannot
    # close. `n = 1` (already covered by the textbook-gate pool) never hit it.
    (
        "diff_x_squared_log_x",
        "product_rule",
        lambda pool: alkahest.diff(
            pool.symbol("x") ** 2 * alkahest.log(pool.symbol("x")), pool.symbol("x")
        ),
    ),
    (
        "diff_x_cubed_log_x",
        "product_rule",
        lambda pool: alkahest.diff(
            pool.symbol("x") ** 3 * alkahest.log(pool.symbol("x")), pool.symbol("x")
        ),
    ),
    (
        "diff_x_squared_sqrt_x",
        "product_rule",
        lambda pool: alkahest.diff(
            pool.symbol("x") ** 2 * alkahest.sqrt(pool.symbol("x")), pool.symbol("x")
        ),
    ),
    # `d/dx (exp x / cos x)`: `HasDerivAt.mul` proves a fact about
    # `left * right` in that literal order, so the witness has to follow the
    # order the goal prints, not always numerator-first.
    (
        "diff_quotient_exp_over_cos",
        "product_rule",
        _diff_exp_over_cos,
    ),
    (
        "diff_quotient_inv_first_cos_exp",
        "product_rule",
        _diff_inv_first_cos_exp,
    ),
    # `x² · x⁻¹ = x`: net exponent 1, which the kernel spells as the bare base
    # rather than `x^1`. That spelling escaped the nonzero-hypothesis override
    # and picked up `collect_mul_factors`' unconditional `by ring`, which
    # cannot discharge a goal containing `⁻¹`.
    (
        "simplify_x_squared_over_x",
        "collect_mul_factors",
        lambda pool: alkahest.simplify(pool.symbol("x") ** 2 * pool.symbol("x") ** -1),
    ),
    # Both printed orders of the Pythagorean identity. `rw
    # [Real.sin_sq_add_cos_sq]` found no instance of its pattern in the second.
    ("simplify_trig_pythagorean_sin_first", "sin_sq_plus_cos_sq", _pythagorean_sin_first),
    ("simplify_trig_pythagorean_cos_first", "sin_sq_plus_cos_sq", _pythagorean_cos_first),
    # `∫₀¹ (x² + 1)⁻¹` — the flipped intern order of the case above.
    # `Real.hasDerivAt_arctan'` has value `(1 + x²)⁻¹`, and the certificate's
    # top-level `simpa` cannot reorder inside `HasDerivAt`'s value argument.
    (
        "int_def_inv_x_squared_plus_one_0_1",
        "fundamental_theorem_of_calculus",
        _int_def_arctan_sq_first,
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
