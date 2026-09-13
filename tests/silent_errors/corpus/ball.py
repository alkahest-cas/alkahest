"""Silent-error cases for ball.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

from typing import Any, Callable

import alkahest as ak
from contracts import Case, RefusesOr, Returns

from ._shared import X, _int


def _ball_encloses(expr: ak.Expr, binding: dict[Any, Any], truth: float) -> Callable[[], bool]:
    """Answer = does ``interval_eval``'s ball claim to contain *truth*?

    ``ArbBall.contains`` is the documented way to read the guarantee, so a
    ``False`` here is the library contradicting its own contract — and it is a
    *directed* lie: a caller cross-checking a symbolic identity against it reads
    ``False`` as a refutation rather than as "no information".
    """

    def op() -> bool:
        return bool(ak.interval_eval(expr, binding).contains(truth))

    return op


def _ball_upper(expr: ak.Expr, binding: dict[Any, Any]) -> Callable[[], float]:
    """Answer = the ball's upper endpoint, for the cases where no bound exists."""

    def op() -> float:
        return float(ak.interval_eval(expr, binding).hi)

    return op


CASES: list[Case] = [
    # ── validated bounds and ball arithmetic ────────────────────────────────
    #
    # The archetype for this layer is not a wrong number, it is a wrong
    # *interval*.  An enclosure that does not enclose is a false lemma every
    # downstream derivation inherits, and it is indistinguishable from a sound
    # one at the call site.
    Case(
        id="ball_indeterminate_product_still_encloses",
        subsystem="ball",
        statement="interval_eval((x^-3)^2) over x in [-3.325, -1.325] must contain 2.325^-6",
        op=_ball_encloses(
            (X ** _int(-3)) ** _int(2),
            {X: ak.ArbBall(-2.325, 1.0)},
            0.006331864446927654,
        ),
        contract=Returns(True),
        verified_by=(
            "x^-6 is even and decreasing in |x|, so on |x| in [1.325, 3.325] its range is "
            "[3.325^-6, 1.325^-6] = [7.400313e-4, 0.1848012] (mpmath, 50 digits), and the "
            "midpoint value 2.325^-6 = 6.331864e-3 lies inside it. Every sound enclosure of "
            "this expression over this box therefore contains that number: `contains` may "
            "answer False only for values outside [7.4e-4, 0.185], which this is not."
        ),
        note=(
            "Two failures meet here. x^3 by repeated ball squaring lost the sign of the box "
            "and came out straddling zero, so 1/x^3 was the indeterminate ball [0 +- inf]; "
            "squaring *that* computed a radius of 0*inf = NaN, and every comparison against a "
            "NaN endpoint is false, so `contains` answered False for every real number, the "
            "true value included. Powering from the endpoints fixes the first, and NaN no "
            "longer escapes any operation, which fixes the second. "
            "ball_removable_quotient_across_zero_still_encloses is the case that still reaches "
            "the NaN path, by a route no precision improvement can close."
        ),
    ),
    Case(
        id="ball_control_reciprocal_power_encloses",
        subsystem="ball",
        statement="interval_eval(x^-6) over x in [2, 3] must bracket [1/729, 1/64] tightly",
        op=lambda: bool(
            ak.interval_eval(X ** _int(-6), {X: ak.ArbBall(2.5, 0.5)}).lo
            <= (1.0 / 729.0) * (1 + 1e-12)
            and ak.interval_eval(X ** _int(-6), {X: ak.ArbBall(2.5, 0.5)}).hi >= 1.0 / 64.0
        ),
        contract=Returns(True),
        verified_by=(
            "x^-6 is decreasing on [2, 3], so its range there is exactly [3^-6, 2^-6] = "
            "[1/729, 1/64], attained at the two endpoints. 1/64 is exact in binary, so the "
            "upper bound is compared with no slack; 1/729 is not, and the returned enclosure "
            "is tighter than an f64 can express, so that comparison carries a 1e-12 relative "
            "slack -- far below the 1e-3 width of the interval and far above the 1e-16 the "
            "f64 literal is off by."
        ),
        note=(
            "Also the tightness control. The enclosure used to be [-inf, inf] here, because "
            "x^6 computed by repeated ball squaring straddled zero and its reciprocal was "
            "therefore unbounded: sound, and completely useless."
        ),
    ),
    Case(
        id="ball_reciprocal_across_zero_is_not_a_bound",
        subsystem="ball",
        statement="1/x over x in [-1, 1] has no finite enclosure -- 0 is in the box",
        op=_ball_upper(_int(1) / X, {X: ak.ArbBall(0.0, 1.0)}),
        contract=RefusesOr(),
        verified_by=(
            "1/x is unbounded on every neighbourhood of 0 and undefined at it, so no finite "
            "interval contains its range on [-1, 1]. Any finite number returned here would be "
            "a bound that is not one."
        ),
        note=(
            "Passes via a weak refusal: the ball comes back as [-inf, inf] rather than as an "
            "exception. That is the honest 'no information' answer for ball arithmetic, but a "
            "caller has to look at the value to notice."
        ),
    ),
    Case(
        id="ball_removable_quotient_across_zero_still_encloses",
        subsystem="ball",
        statement="sin(x)/x over x in [-0.5, 0.5] must not claim to exclude 0.9588510772",
        op=_ball_encloses(ak.sin(X) / X, {X: ak.ArbBall(0.0, 0.5)}, 0.958851077208406),
        contract=Returns(True),
        verified_by=(
            "sin(0.5)/0.5 = 0.95885107720840600... (mpmath, 50 digits) is the value the "
            "expression takes at a point of the ball, so no sound enclosure over that ball can "
            "exclude it. Pointwise ball arithmetic cannot resolve the 0/0 at the centre, so "
            "the only correct answers are 'the whole line' or a refusal -- never 'that value "
            "is outside'."
        ),
        note=(
            "Same NaN mechanism as ball_indeterminate_product_still_encloses reached from the "
            "other side: 1/x is indeterminate and sin(x) has midpoint 0, so the product's "
            "radius was |0| * inf = NaN."
        ),
    ),
    Case(
        id="ball_sqrt_of_a_ball_straddling_zero_refuses",
        subsystem="ball",
        statement="sqrt of the ball [-1, 1] is not real -- refuse rather than take the real part",
        op=lambda: float(ak.ArbBall(0.0, 1.0).sqrt().lo),
        contract=RefusesOr(),
        verified_by=(
            "sqrt is undefined on the negative half of [-1, 1], so no real enclosure of it "
            "exists there. The plausible wrong answer is [0, 1]: the image of the part of the "
            "ball where sqrt happens to be defined, which silently shrinks the domain."
        ),
    ),
    Case(
        id="ball_floor_straddling_an_integer_covers_both_values",
        subsystem="ball",
        statement="floor over x in [0.75, 1.25] takes both 0 and 1, so the ball is >= 1 wide",
        op=lambda: float(
            ak.interval_eval(ak.floor(X), {X: ak.ArbBall(1.0, 0.25)}).hi
            - ak.interval_eval(ak.floor(X), {X: ak.ArbBall(1.0, 0.25)}).lo
        ),
        contract=Returns(1.0, tol=1e-12),
        verified_by=(
            "floor(0.75) = 0 and floor(1.25) = 1, so the range is exactly {0, 1} and the "
            "narrowest sound interval is [0, 1], of width 1. Evaluating the midpoint and "
            "adding the input radius -- the Lipschitz shortcut that works for sin and exp -- "
            "would give width 0.5 around floor(1) = 1, an interval that misses 0 entirely."
        ),
    ),
    Case(
        id="ball_unsupported_primitive_refuses",
        subsystem="ball",
        statement="sign(x) has no ball rule; interval_eval must refuse rather than use f64",
        op=lambda: float(ak.interval_eval(ak.sign(X), {X: ak.ArbBall(0.5, 0.25)}).hi),
        contract=RefusesOr(),
        verified_by=(
            "capabilities() reports numeric_ball = False for `sign`. The dangerous answer is "
            "the f64 one, sign(0.5) = 1 as an exact ball: right on this box, wrong on any box "
            "straddling 0, with nothing in the result to say which case the caller got."
        ),
    ),
]
