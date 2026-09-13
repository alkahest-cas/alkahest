"""Silent-error cases for simplification.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import alkahest as ak
from contracts import Case, RefusesOr, Returns

from ._shared import _Z, POOL, X, Y, _int, _num, _rat


def simplified_value(
    simplifier: Callable[[ak.Expr], Any], expr: ak.Expr, at: float | None = None
) -> Callable[[], float]:
    """Answer = the simplified expression's numeric value at *at*.

    Simplification is only ever allowed to change an expression's *form*.  Any
    rewrite that changes its value at a point — the signature of a branch-cut
    violation — shows up here as a wrong number.
    """

    def op() -> float:
        out = simplifier(expr)
        value = out.value if isinstance(out, ak.DerivedResult) else out
        env = {} if at is None else {X: at}
        return float(ak.eval_expr(value, env))

    return op


def _undisclosed_expansion_limit(base: ak.Expr, exponent: int) -> Callable[[], float]:
    """Answer = 1.0 if a bounded expansion no-oped without saying so, else 0.0.

    ``simplify_expanded`` is asked to expand.  When the internal size bound stops
    it, the *value* it returns is the input — mathematically equal, and so
    impossible to tell apart from "this is already expanded" or "this cannot be
    expanded further".  Either the expansion happens, or the derivation log
    records that a bound was reached; silently doing neither is the defect.
    """

    def op() -> float:
        power = base ** _int(exponent)
        r = ak.simplify_expanded(power)
        # Compared against plain `simplify`, not against the input: both flatten
        # `((x+y)+z)` to `(x+y+z)`, so only expansion can separate them.
        expanded = str(r.value) != str(ak.simplify(power).value)
        disclosed = any("limit" in step["rule"] for step in r.steps)
        return 0.0 if (expanded or disclosed) else 1.0

    return op


def _expansion_within_the_budget(base: ak.Expr, exponent: int, at: float) -> Callable[[], float]:
    """Answer = the expanded polynomial's value at *at*.

    The control: a power inside the bound must actually be expanded (not the
    original ``Pow``), must record **no** limit step — so the disclosure cannot
    be emitted unconditionally — and must still agree with the input at a sample
    point, which is what makes "expanded" a claim about form only.
    """

    def op() -> float:
        power = base ** _int(exponent)
        r = ak.simplify_expanded(power)
        if str(r.value) == str(ak.simplify(power).value):
            raise AssertionError("a power inside the expansion budget was left unexpanded")
        if any("limit" in step["rule"] for step in r.steps):
            raise AssertionError("a limit step was recorded for an expansion that happened")
        return float(ak.eval_expr(r.value, {X: at}))

    return op


CASES: list[Case] = [
    # Branch-cut discipline in simplification.  Every case here is a rewrite
    # that a naive rule system performs and that changes the function's value.
    # The check is value preservation at a point where the naive rule breaks.
    # -----------------------------------------------------------------------
    Case(
        id="simplify_sqrt_of_square",
        subsystem="simplification",
        statement="√(x²) = |x|, not x: at x=-2 the value is 2",
        op=simplified_value(ak.simplify, ak.sqrt(X**2), at=-2.0),
        contract=Returns(2.0),
        verified_by="√((-2)²) = √4 = 2. The rewrite √(x²) → x gives -2.",
    ),
    Case(
        id="simplify_egraph_sqrt_of_square",
        subsystem="simplification",
        statement="the e-graph simplifier must not rewrite √(x²) to x either",
        op=simplified_value(ak.simplify_egraph, ak.sqrt(X**2), at=-2.0),
        contract=Returns(2.0),
        verified_by="Same identity; checked separately because the e-graph engine has its own "
        "rule set and its own extraction.",
    ),
    Case(
        id="simplify_rational_power_of_square",
        subsystem="simplification",
        statement="(x²)^(1/2) = |x|: at x=-2 the value is 2, not -2 and not 1",
        op=simplified_value(ak.simplify, (X**2) ** _rat(1, 2), at=-2.0),
        contract=Returns(2.0),
        verified_by="(x^a)^b = x^{ab} is invalid for non-integer b on negative bases: "
        "((-2)²)^(1/2) = 4^(1/2) = 2, while (-2)^1 = -2.",
    ),
    Case(
        id="simplify_log_of_square",
        subsystem="simplification",
        statement="log(x²) ≠ 2·log x on the negatives: at x=-2 the value is log 4",
        op=simplified_value(ak.simplify, ak.log(X**2), at=-2.0),
        contract=Returns(math.log(4.0)),
        verified_by="log((-2)²) = log 4 ≈ 1.3862944. 2·log(-2) is undefined over ℝ.",
    ),
    Case(
        id="simplify_arcsin_of_sin",
        subsystem="simplification",
        statement="arcsin(sin x) = x only on [-π/2, π/2]: at x=3 the value is π-3",
        op=simplified_value(ak.simplify, ak.asin(ak.sin(X)), at=3.0),
        contract=Returns(math.pi - 3.0),
        verified_by="sin 3 = sin(π-3) and π-3 ≈ 0.1416 ∈ [-π/2, π/2], so arcsin(sin 3) = π-3.",
    ),
    Case(
        id="simplify_arccos_of_cos",
        subsystem="simplification",
        statement="arccos(cos x) = x only on [0, π]: at x=4 the value is 2π-4",
        op=simplified_value(ak.simplify, ak.acos(ak.cos(X)), at=4.0),
        contract=Returns(2 * math.pi - 4.0),
        verified_by="cos 4 = cos(2π-4) and 2π-4 ≈ 2.2832 ∈ [0, π].",
    ),
    Case(
        id="simplify_arctan_of_tan",
        subsystem="simplification",
        statement="arctan(tan x) = x only on (-π/2, π/2): at x=2 the value is 2-π",
        op=simplified_value(ak.simplify, ak.atan(ak.tan(X)), at=2.0),
        contract=Returns(2.0 - math.pi),
        verified_by="tan has period π, so arctan(tan 2) = 2-π ≈ -1.1416.",
    ),
    Case(
        id="simplify_egraph_rational_power",
        subsystem="simplification",
        statement="the e-graph simplifier must preserve (x²)^(1/2): at x=-2 the value is 2",
        op=simplified_value(ak.simplify_egraph, (X**2) ** _rat(1, 2), at=-2.0),
        contract=Returns(2.0),
        verified_by="((-2)²)^(1/2) = 2, by hand.",
    ),
    Case(
        id="simplify_egraph_square_root_power",
        subsystem="simplification",
        statement="simplify_egraph(x^(1/2)) at x=4 is 2",
        op=simplified_value(ak.simplify_egraph, X ** _rat(1, 2), at=4.0),
        contract=Returns(2.0),
        verified_by="4^(1/2) = 2.",
    ),
    Case(
        id="simplify_egraph_rational_literal",
        subsystem="simplification",
        statement="simplify_egraph(1/2) is 1/2",
        op=simplified_value(ak.simplify_egraph, _rat(1, 2)),
        contract=Returns(0.5),
        verified_by="A rational literal simplifies to itself.",
    ),
    Case(
        id="simplify_egraph_rational_coefficient",
        subsystem="simplification",
        statement="simplify_egraph(x/2) at x=3 is 1.5",
        op=simplified_value(ak.simplify_egraph, X * _rat(1, 2), at=3.0),
        contract=Returns(1.5),
        verified_by="3 · (1/2) = 1.5, by hand.",
    ),
    Case(
        id="simplify_egraph_rational_summand",
        subsystem="simplification",
        statement="simplify_egraph(x + 1/2) at x=1 is 1.5",
        op=simplified_value(ak.simplify_egraph, X + _rat(1, 2), at=1.0),
        contract=Returns(1.5),
        verified_by="1 + 1/2 = 1.5.",
    ),
    Case(
        id="simplify_egraph_float_summand",
        subsystem="simplification",
        statement="simplify_egraph(x + 0.5) at x=1 is 1.5",
        op=simplified_value(ak.simplify_egraph, X + POOL.float(0.5), at=1.0),
        contract=Returns(1.5),
        verified_by="1 + 0.5 = 1.5.",
    ),
    Case(
        id="simplify_egraph_control_pythagorean",
        subsystem="simplification",
        statement="simplify_egraph(sin²x + cos²x) = 1",
        op=simplified_value(ak.simplify_egraph, ak.sin(X) ** 2 + ak.cos(X) ** 2, at=0.7),
        contract=Returns(1.0),
        verified_by="Pythagorean identity. The control proving the e-graph engine is live and "
        "rewriting, not just echoing its input.",
    ),
    Case(
        id="simplify_egraph_control_add_zero",
        subsystem="simplification",
        statement="simplify_egraph(x + 0) at x=3 is 3",
        op=simplified_value(ak.simplify_egraph, X + _int(0), at=3.0),
        contract=Returns(3.0),
        verified_by="Additive identity; a genuine (Num 0) summand, unlike the xfail cases where "
        "the 0 is fabricated by the serialiser.",
    ),
    Case(
        id="simplify_control_cancel_x_over_x",
        subsystem="simplification",
        statement="cancel(x/x) = 1",
        op=lambda: _num(ak.cancel(X / X)),
        contract=Returns(1.0),
        verified_by="Valid for x≠0, which is where the expression is defined.",
    ),
    # Division by a literal zero.  `x · x^-1 → 1` and `x · 0 → 0` are both
    # deliberate conventions (see simplify_control_cancel_x_over_x), and both
    # are false when the base really is zero: `0 · 0^-1` is `0 · ∞`, which has
    # no value under any convention.  `simplify(0^-1)` already leaves the power
    # alone and `eval_expr(0^-1)` raises E-EVAL-009, so a product that quietly
    # collapses to a number is contradicting the rest of the library.
    # -----------------------------------------------------------------------
    Case(
        id="simplify_zero_times_zero_reciprocal",
        subsystem="simplification",
        statement="0 · 0^-1 is undefined — not 1, not 0",
        op=simplified_value(ak.simplify, _int(0) * _int(0) ** _int(-1)),
        contract=RefusesOr(),
        verified_by=(
            "0^-1 is division by zero, so the product has no value: it is the indeterminate "
            "form 0·∞. Summing the exponents to 0^0 = 1 is invalid precisely because the "
            "base is zero — b^k·b^m = b^(k+m) needs b ≠ 0 once one exponent is negative."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_zero_reciprocal_in_longer_product",
        subsystem="simplification",
        statement="5 · 0^-1 · 0 is undefined — the arrangement must not change the answer",
        op=simplified_value(ak.simplify, _int(5) * _int(0) ** _int(-1) * _int(0)),
        contract=RefusesOr(),
        verified_by=(
            "Same undefined product with a spectator factor: 5·(0·∞) is still indeterminate. "
            "This arrangement is folded by the numeric constant folder rather than by the "
            "exponent collector, so it is a second, independent route to the same lie — and "
            "it used to give 0 where the two-factor form gave 1, which is its own proof that "
            "at least one of them is wrong."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_symbolic_zero_times_its_reciprocal",
        subsystem="simplification",
        statement="(x-x) · (x-x)^-1 is undefined: the base is identically zero",
        op=simplified_value(ak.simplify, (X - X) * (X - X) ** _int(-1), at=2.0),
        contract=RefusesOr(),
        verified_by=(
            "x - x is the zero function, so (x-x)^-1 is nowhere defined and the product has "
            "no value at any x. Cancelling b·b^-1 → 1 asserts b ≠ 0, which is false here. "
            "This is the shape `diff(2/(x-x), x)` reaches, so it is not a hand-written "
            "curiosity."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_egraph_zero_times_zero_reciprocal",
        subsystem="simplification",
        statement="the e-graph simplifier must not give 0 · 0^-1 a value either",
        op=simplified_value(ak.simplify_egraph, _int(0) * _int(0) ** _int(-1)),
        contract=RefusesOr(),
        verified_by=(
            "Same undefined product; checked separately because the e-graph engine has its "
            "own rule set. It is the worse of the two failures: its shrink rules contain "
            "both (Mul ?x (Num 0)) → (Num 0) and (Mul ?x (Pow ?x (Num -1))) → (Num 1), so "
            "on this input it unions 0 and 1 into a single e-class — every other e-class in "
            "the run is then equally suspect."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="diff_reciprocal_of_identically_zero_denominator",
        subsystem="simplification",
        statement="d/dx [2/(x-x)] has no value: the function is nowhere defined",
        op=lambda: _num(ak.diff(_int(2) / (X - X), X)),
        contract=RefusesOr(),
        verified_by=(
            "2/(x-x) = 2/0 has empty domain, so it has no derivative anywhere; 1 is a value "
            "it can never take. Reached through an ordinary `diff` call, without writing "
            "0^-1 by hand: the quotient rule produces 0·0^-1 terms and the simplifier used "
            "to collapse them."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_control_symbol_over_symbol",
        subsystem="simplification",
        statement="simplify(x · x^-1) = 1 for a symbolic x",
        op=simplified_value(ak.simplify, X * X ** _int(-1), at=2.0),
        contract=Returns(1.0),
        verified_by=(
            "2 · (1/2) = 1. The documented convention for a base that is not provably zero, "
            "and the control that the zero-base guard did not simply switch factor "
            "collection off."
        ),
    ),
    Case(
        id="simplify_control_zero_times_symbol",
        subsystem="simplification",
        statement="simplify(0 · x) = 0",
        op=simplified_value(ak.simplify, _int(0) * X, at=3.0),
        contract=Returns(0.0),
        verified_by=(
            "0 · 3 = 0. The control for the absorption rule: it must keep firing on products "
            "that really are zero, and only decline when a co-factor is undefined."
        ),
    ),
    Case(
        id="simplify_control_like_terms_cancel_to_zero",
        subsystem="simplification",
        statement="simplify(2x - 2x) = 0",
        op=simplified_value(ak.simplify, _int(2) * X - _int(2) * X, at=5.0),
        contract=Returns(0.0),
        verified_by=(
            "10 - 10 = 0. The control for like-term collection, which must still drop terms "
            "whose coefficients cancel — the guard only applies when the surviving factor is "
            "a division by zero."
        ),
    ),
    Case(
        id="simplify_egraph_control_symbol_over_symbol",
        subsystem="simplification",
        statement="simplify_egraph(x · x^-1) = 1 for a symbolic x",
        op=simplified_value(ak.simplify_egraph, X * X ** _int(-1), at=2.0),
        contract=Returns(1.0),
        verified_by=(
            "2 · (1/2) = 1. The e-graph control: it must still cancel a symbolic base, so "
            "the zero-base bail-out cannot be passed by disabling the engine."
        ),
    ),
    Case(
        id="expand_power_bound_is_not_a_silent_no_op",
        subsystem="simplification",
        statement="simplify_expanded((x+y+z)^9) must expand it or record the bound it hit",
        op=_undisclosed_expansion_limit(X + Y + _Z, 9),
        contract=Returns(0.0),
        verified_by=(
            "By the multinomial theorem (x+y+z)^9 expands to C(11,2) = 55 distinct monomials, "
            "so 'already expanded' is false and the returned Pow is not the answer to the "
            "question asked. Returning the input unchanged is a correct *value* and a "
            "misleading *result*: .steps is documented as a faithful record of what happened, "
            "and it recorded nothing at all."
        ),
        note="Passes by disclosure (a derivation step) or by doing the expansion.",
    ),
    Case(
        id="expand_control_power_inside_the_budget",
        subsystem="simplification",
        statement="simplify_expanded((x+1)^6) = 729 at x = 2, expanded and unremarked",
        op=_expansion_within_the_budget(X + _int(1), 6, 2.0),
        contract=Returns(729.0),
        verified_by=(
            "(2+1)^6 = 3^6 = 729 by hand; the binomial expansion 1 + 6x + 15x² + 20x³ + 15x⁴ "
            "+ 6x⁵ + x⁶ at x = 2 gives 1+12+60+160+240+192+64 = 729, so the expanded form must "
            "agree. The control for the case above: it fails if expansion stops firing, and "
            "also if a limit step is recorded for an expansion that in fact happened."
        ),
    ),
]
