"""Silent-error cases for evaluation.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Callable

import alkahest as ak
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import POOL, X, _int, _rat

HAND = "hand derivation from the definition"


def _enclosure_contains(expr: ak.Expr, lo: float, hi: float, truth: float) -> Callable[[], bool]:
    """Answer = does the *validated* enclosure of ``expr`` over the box contain
    the value it claims to enclose?"""

    def op() -> bool:
        enc = ak.bound_on_box(expr, [(X, lo, hi)])
        return bool(enc.lower <= truth <= enc.upper)

    return op


CASES: list[Case] = [
    # Evaluation at points where the expression as written is undefined.
    # -----------------------------------------------------------------------
    Case(
        id="eval_removable_singularity",
        subsystem="evaluation",
        statement="(x²-1)/(x-1) has no VALUE at x=1 — 2 is the limit, not the value",
        op=lambda: float(ak.eval_expr((X**2 - 1) / (X - 1), {X: 1})),
        contract=Raises("E-EVAL-009"),
        verified_by="0/0 is undefined. x+1 is a *different function*: it is defined at 1.",
        benchmark_tasks=("removable_singularity_value",),
    ),
    Case(
        id="eval_after_explicit_cancel",
        subsystem="evaluation",
        statement="cancel((x²-1)/(x-1)) = x+1 evaluates to 2 at x=1 — an explicit rewrite is fine",
        op=lambda: float(ak.eval_expr(ak.cancel((X**2 - 1) / (X - 1)), {X: 1})),
        contract=Returns(2.0),
        verified_by="1+1 = 2. Pairs with eval_removable_singularity: the sin is doing the "
        "cancellation silently, not offering it.",
    ),
    Case(
        id="eval_simple_pole",
        subsystem="evaluation",
        statement="1/x is undefined at x=0",
        op=lambda: float(ak.eval_expr(1 / X, {X: 0})),
        contract=Raises("E-EVAL-009"),
        verified_by=HAND,
    ),
    Case(
        id="eval_log_at_zero",
        subsystem="evaluation",
        statement="log x is undefined at x=0",
        op=lambda: float(ak.eval_expr(ak.log(X), {X: 0})),
        contract=Raises("E-EVAL-009"),
        verified_by=HAND,
    ),
    Case(
        id="eval_log_of_negative",
        subsystem="evaluation",
        statement="log(-1) has no real value — returning one is a branch-cut violation",
        op=lambda: float(ak.eval_expr(ak.log(X), {X: -1})),
        contract=Raises("E-EVAL-009"),
        verified_by="The real logarithm is defined on (0,∞). The principal complex value is iπ.",
    ),
    Case(
        id="eval_sqrt_of_negative",
        subsystem="evaluation",
        statement="√(-1) has no real value",
        op=lambda: float(ak.eval_expr(ak.sqrt(X), {X: -1})),
        contract=Raises("E-EVAL-009"),
        verified_by="The real square root is defined on [0,∞).",
    ),
    Case(
        id="eval_arcsin_out_of_range",
        subsystem="evaluation",
        statement="arcsin(2) has no real value",
        op=lambda: float(ak.eval_expr(ak.asin(X), {X: 2})),
        contract=Raises("E-EVAL-009"),
        verified_by="Real arcsin has domain [-1,1].",
    ),
    Case(
        id="eval_artanh_out_of_range",
        subsystem="evaluation",
        statement="artanh(2) has no real value",
        op=lambda: float(ak.eval_expr(ak.atanh(X), {X: 2})),
        contract=Raises("E-EVAL-009"),
        verified_by="Real artanh has domain (-1,1).",
    ),
    Case(
        id="eval_odd_root_of_negative",
        subsystem="evaluation",
        statement="(-8)^(1/3): the principal branch is complex; only -2 is a defensible real value",
        op=lambda: float(ak.eval_expr(_int(-8) ** _rat(1, 3), {})),
        contract=RefusesOr(-2.0),
        verified_by="Principal cube root of -8 is 1+i√3 (modulus 2, argument π/3). The real "
        "cube root is -2. Any other real number — notably +2 — is a branch-cut lie.",
    ),
    # ── validated bounds ────────────────────────────────────────────────────
    #
    # An enclosure that does not contain the value it encloses is the one thing
    # a "validated" subsystem may never do: downstream it is not a wrong number
    # but a false theorem.
    Case(
        id="validated_cos_enclosure_contains_cos_one",
        subsystem="evaluation",
        statement="the validated enclosure of cos x at x = 1 must contain cos 1 = 0.5403…",
        op=_enclosure_contains(ak.cos(X), 1.0, 1.0, math.cos(1.0)),
        contract=Returns(True),
        verified_by=(
            "cos 1 = 0.5403023058681398 (math.cos, and alkahest's own interval_eval agrees). "
            "bound_on_box returned [-0.5403023058681398, -0.5403023058681397]: the Taylor-model "
            "evaluator negated every cosine coefficient while leaving the symmetric remainder "
            "bound alone, so the enclosure came back tight, confident and sign-flipped."
        ),
    ),
    Case(
        id="validated_no_roots_respects_a_real_root",
        subsystem="evaluation",
        statement="cos x - 0.9 has a root at arccos(0.9) = 0.4510 ∈ [0,1], so 'no roots' is false",
        op=lambda: ak.verified_no_roots(ak.cos(X) - POOL.float(0.9, 53), [(X, 0.0, 1.0)]),
        contract=RefusesOr("false"),
        verified_by=(
            "arccos(0.9) = 0.45102681179626236 lies in [0,1] and cos is continuous, so a root "
            "certainly exists there. alkahest answered 'true' — a machine-checked-looking proof "
            "of a false theorem, not merely a wrong number."
        ),
    ),
    Case(
        id="validated_control_sin_enclosure",
        subsystem="evaluation",
        statement="the validated enclosure of sin x at x = 1 contains sin 1 = 0.8415…",
        op=_enclosure_contains(ak.sin(X), 1.0, 1.0, math.sin(1.0)),
        contract=Returns(True),
        verified_by=(
            "sin 1 = 0.8414709848078965 (math.sin). The control for the cos cases: sin was always "
            "correct, so a gate that simply stopped trusting the Taylor-model path would not pass."
        ),
    ),
    # Γ at its poles.
    # -----------------------------------------------------------------------
    Case(
        id="gamma_at_a_negative_integer_pole",
        subsystem="evaluation",
        statement="Γ(-2) does not exist — Γ has a simple pole at every non-positive integer",
        op=lambda: float(ak.eval_expr(ak.gamma(_int(-2)), {})),
        contract=Raises("E-EVAL-009"),
        verified_by=(
            "1/Γ is entire with a simple zero at 0, -1, -2, …, so Γ has a pole there and no "
            "finite value. Alkahest already raised E-EVAL-009 for Γ(0); the reflection formula "
            "π/(sin(πx)·Γ(1-x)) produced 6.4e15 at x = -2 only because sin(π·(-2.0)) rounds to "
            "2.45e-16 rather than 0 in binary floating point."
        ),
    ),
    Case(
        id="gamma_control_negative_half_integer",
        subsystem="evaluation",
        statement="Γ(-1/2) = -2√π — a negative argument that is not a pole",
        op=lambda: float(ak.eval_expr(ak.gamma(_rat(-1, 2)), {})),
        contract=Returns(-3.5449077018110318, tol=1e-9),
        verified_by=(
            "Γ(1/2) = √π and Γ(x+1) = x·Γ(x), so Γ(-1/2) = Γ(1/2)/(-1/2) = -2√π = "
            "-3.5449077018110318. The control for the pole guard: refusing the whole negative "
            "half-line would pass the trap above and fail this."
        ),
    ),
    Case(
        id="eval_large_integer_literal_is_the_nearest_double",
        subsystem="evaluation",
        statement="10^30 evaluates to the nearest double, 1e30, not the one below it",
        op=lambda: float(ak.eval_expr(_int(10**30) + _int(0) * X, {X: 0.0})),
        contract=Returns(1e30, tol=0.0),
        verified_by=(
            "10^30 = 2^30·5^30 needs 70 significant bits, so it is not exact in binary64. "
            "Python's int->float is correctly rounded (round-half-even) and gives 1e30; "
            "truncation towards zero gives 9.999999999999999e29, one ulp low."
        ),
        note=(
            "One ulp, but biased: truncation never rounds up, so a sum of large exact "
            "integers drifts in one direction. It is also a place two evaluators disagreed "
            "- emit_c_expr wrote the exact decimal and let the C compiler round it."
        ),
    ),
    Case(
        id="eval_rational_literal_is_the_nearest_double",
        subsystem="evaluation",
        statement="2/5 evaluates to 0.4, the nearest double, under every mode",
        op=lambda: float(ak.evaluate(_rat(2, 5), {}, mode="f64").value),
        contract=Returns(0.4, tol=0.0),
        verified_by=(
            "float(Fraction(2, 5)) = 0.4 exactly (Python rounds correctly). The double "
            "below it is 0.39999999999999997, which is what rounding towards zero returns."
        ),
    ),
    Case(
        id="eval_rational_with_both_parts_past_the_float_range",
        subsystem="evaluation",
        statement="(3·10^400+1)/(2·10^400) is 1.5, not an inf/inf NaN",
        op=lambda: float(
            ak.eval_expr(POOL.rational(3 * 10**400 + 1, 2 * 10**400) + _int(0) * X, {X: 0.0})
        ),
        contract=Returns(1.5, tol=1e-15),
        verified_by=(
            "float(Fraction(3*10**400+1, 2*10**400)) = 1.5. The parts are coprime, so the "
            "fraction is in lowest terms and both of them overflow binary64 on their own."
        ),
        note=(
            "The evaluators split three ways here: `evaluate(mode='f64')` was right, "
            "`eval_expr` refused with E-EVAL-009 (a weak refusal produced by NaN), and "
            "compile_expr/numpy_eval returned a bare NaN."
        ),
    ),
    Case(
        id="eval_control_small_rational_literal",
        subsystem="evaluation",
        statement="1/2 evaluates to 0.5 — the literals that were always exact stay exact",
        op=lambda: float(ak.eval_expr(_rat(1, 2), {})),
        contract=Returns(0.5, tol=0.0),
        verified_by="1/2 is exact in binary64. Control for the two rounding cases above.",
    ),
]
