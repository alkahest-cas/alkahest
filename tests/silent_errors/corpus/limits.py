"""Silent-error cases for limits.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

from typing import Callable

import alkahest as ak
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import CALCULUS, X, _int, _num


def limit_value(expr: ak.Expr, point: ak.Expr, direction: str | None = None) -> Callable[[], float]:
    """Answer = the numeric value of lim expr, optionally one-sided."""

    def op() -> float:
        got = (
            ak.limit(expr, X, point)
            if direction is None
            else ak.limit(expr, X, point, dir=direction)
        )
        return _num(got)

    return op


CASES: list[Case] = [
    # Limits that do not exist, versus one-sided limits that do.
    # -----------------------------------------------------------------------
    Case(
        id="limit_two_sided_simple_pole",
        subsystem="limits",
        statement="lim_{x→0} 1/x does not exist (-∞ from the left, +∞ from the right)",
        op=limit_value(1 / X, _int(0)),
        contract=Raises("E-LIMIT-003"),
        verified_by=CALCULUS,
    ),
    Case(
        id="limit_abs_over_x",
        subsystem="limits",
        statement="lim_{x→0} |x|/x does not exist (-1 from the left, +1 from the right)",
        op=limit_value(ak.abs(X) / X, _int(0)),
        contract=RefusesOr(),
        verified_by="|x|/x = sign(x); the one-sided limits are -1 and +1 and disagree.",
        note="alkahest refuses with E-LIMIT-005 ('could not be computed'), which is safe but "
        "less informative than E-LIMIT-003 ('two-sided limit undefined').",
    ),
    Case(
        id="limit_x_over_abs_two_sided",
        subsystem="limits",
        statement="lim_{x→0} x/|x| does not exist — it is sign(x), same function reordered",
        op=limit_value(X / ak.abs(X), _int(0)),
        contract=RefusesOr(),
        verified_by="x/|x| = sign(x) for x≠0; one-sided limits -1 and +1 disagree. Numerically: "
        "f(-0.001) = -1.0, f(+0.001) = +1.0.",
    ),
    # Deliberately RefusesOr rather than Returns, and the distinction is worth
    # spelling out because it looks like a weakened test.
    #
    # These limits are genuinely computable — a first-course student gets ±1 —
    # and alkahest refuses them.  That is an under-answer, not a silent error:
    # computing them symbolically needs sign-aware handling of `abs` under a
    # one-sided approach, which the engine does not have.  It could only produce
    # ±1 by trusting its own numeric samples, i.e. by guessing, which is exactly
    # the behaviour this whole gate exists to prevent.
    #
    # RefusesOr keeps the property that matters: a *different* confident value
    # (the `0` this used to return) is still scored as a silent error.  What is
    # relaxed is "must compute", not "must not lie".
    #
    # The real fix is abs-aware one-sided limits; until then, refusal is honest.
    Case(
        id="limit_x_over_abs_right",
        subsystem="limits",
        statement="lim_{x→0+} x/|x| = 1",
        op=limit_value(X / ak.abs(X), _int(0), direction="+"),
        contract=RefusesOr(1.0),
        verified_by="For x>0, x/|x| = x/x = 1 identically.",
    ),
    Case(
        id="limit_x_over_abs_left",
        subsystem="limits",
        statement="lim_{x→0-} x/|x| = -1",
        op=limit_value(X / ak.abs(X), _int(0), direction="-"),
        contract=RefusesOr(-1.0),
        verified_by="For x<0, x/|x| = x/(-x) = -1 identically.",
    ),
    Case(
        id="limit_tanh_of_reciprocal",
        subsystem="limits",
        statement="lim_{x→0} tanh(1/x) does not exist (-1 from the left, +1 from the right)",
        op=limit_value(ak.tanh(1 / X), _int(0)),
        contract=RefusesOr(),
        verified_by="tanh(t) → ±1 as t → ±∞, and 1/x → ±∞ as x → 0±.",
    ),
    Case(
        id="limit_arctan_of_reciprocal",
        subsystem="limits",
        statement="lim_{x→0} arctan(1/x) does not exist (-π/2 from the left, +π/2 from the right)",
        op=limit_value(ak.atan(1 / X), _int(0)),
        contract=RefusesOr(),
        verified_by="arctan(t) → ±π/2 as t → ±∞.",
    ),
    Case(
        id="limit_exp_of_negative_reciprocal_two_sided",
        subsystem="limits",
        statement="lim_{x→0} e^{-1/x} does not exist (0 from the right, +∞ from the left)",
        op=limit_value(ak.exp(-1 / X), _int(0)),
        contract=RefusesOr(),
        verified_by="As x→0+, -1/x → -∞ so e^{-1/x} → 0; as x→0-, -1/x → +∞ so e^{-1/x} → +∞.",
    ),
    Case(
        id="limit_exp_of_negative_reciprocal_left",
        subsystem="limits",
        statement="lim_{x→0-} e^{-1/x} = +∞",
        op=limit_value(ak.exp(-1 / X), _int(0), direction="-"),
        contract=RefusesOr(),
        verified_by="x→0- ⇒ -1/x → +∞ ⇒ e^{-1/x} → +∞. A finite answer is wrong; +inf reads as "
        "a refusal under this gate's taxonomy, matching agent-benchmark.",
    ),
    Case(
        id="limit_control_sinc",
        subsystem="limits",
        statement="lim_{x→0} sin(x)/x = 1",
        op=limit_value(ak.sin(X) / X, _int(0)),
        contract=Returns(1.0),
        verified_by=CALCULUS,
    ),
    Case(
        id="limit_control_half_angle",
        subsystem="limits",
        statement="lim_{x→0} (1-cos x)/x² = 1/2",
        op=limit_value((1 - ak.cos(X)) / X**2, _int(0)),
        contract=Returns(0.5),
        verified_by="1-cos x = x²/2 - x⁴/24 + …",
    ),
    Case(
        id="limit_control_squeeze",
        subsystem="limits",
        statement="lim_{x→0} x·sin(1/x) = 0 — exists even though sin(1/x) does not",
        op=limit_value(X * ak.sin(1 / X), _int(0)),
        contract=Returns(0.0),
        verified_by="|x sin(1/x)| ≤ |x| → 0 (squeeze). The control for the DNE cases: a limit "
        "engine that refused everything oscillatory would fail here.",
    ),
    Case(
        id="limit_control_one_sided_pole",
        subsystem="limits",
        statement="lim_{x→0+} 1/x = +∞ — the one-sided limit exists as an extended real",
        op=limit_value(1 / X, _int(0), direction="+"),
        contract=RefusesOr(),
        verified_by="Diverges to +∞. alkahest returns the symbol ∞, which does not reduce to a "
        "float and therefore reads as a refusal here — the safe classification.",
    ),
    # One-sided limits taken from outside the domain.
    # -----------------------------------------------------------------------
    Case(
        id="limit_sqrt_from_the_left_of_zero",
        subsystem="limits",
        statement="lim_{x→0⁻} √x does not exist over ℝ — √ is real only for x ≥ 0",
        op=limit_value(ak.sqrt(X), _int(0), direction="-"),
        contract=RefusesOr(),
        verified_by=(
            "√x is real for x ≥ 0 only, so no sequence xₙ ↑ 0 has √xₙ defined and there is "
            "nothing for the one-sided limit to be. Alkahest returned 0, which is exactly the "
            "correct answer to the *other* one-sided question — the two are indistinguishable to "
            "a caller reasoning about domains of definition."
        ),
    ),
    Case(
        id="limit_control_sqrt_from_the_right_of_zero",
        subsystem="limits",
        statement="lim_{x→0⁺} √x = 0",
        op=limit_value(ak.sqrt(X), _int(0), direction="+"),
        contract=Returns(0.0),
        verified_by="0 ≤ √x ≤ √δ for 0 < x < δ, so the right-hand limit is 0 by squeeze.",
    ),
    Case(
        id="limit_control_sqrt_of_square_from_the_left",
        subsystem="limits",
        statement="lim_{x→0⁻} √(x²) = 0 — same head and point, but the left side is in the domain",
        op=limit_value(ak.sqrt(X**2), _int(0), direction="-"),
        contract=Returns(0.0),
        verified_by=(
            "√(x²) = |x| for every real x, and |x| → 0 from either side. The direct control for "
            "the domain guard: a guard that fired on `sqrt` approached from the left, rather than "
            "on the domain, would refuse this."
        ),
    ),
    Case(
        id="limit_arccos_from_the_right_of_one",
        subsystem="limits",
        statement="lim_{x→1⁺} arccos x does not exist over ℝ — arccos is defined only on [-1,1]",
        op=limit_value(ak.acos(X), _int(1), direction="+"),
        contract=RefusesOr(),
        verified_by=(
            "cos maps ℝ onto [-1,1], so arccos has no real value at any x > 1 and no right "
            "neighbourhood of 1 lies in its domain. Alkahest returned arccos(1) = 0."
        ),
    ),
    Case(
        id="limit_control_arccos_from_the_left_of_one",
        subsystem="limits",
        statement="lim_{x→1⁻} arccos x = 0",
        op=limit_value(ak.acos(X), _int(1), direction="-"),
        contract=Returns(0.0),
        verified_by="arccos is continuous on [-1,1] and arccos 1 = 0.",
    ),
]
