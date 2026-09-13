"""Silent-error cases for series.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Callable

import alkahest as ak
import alkahest.experimental as ex

# ``tests/`` is on sys.path via the root conftest, so the textbook gate's series
# helper (which strips the trailing O(...) term that eval_expr cannot evaluate)
# is importable and worth reusing rather than duplicating.
from _tg_helpers import eval_series_truncated
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import X, _int, _rat


def series_at(expr: ak.Expr, about: ak.Expr, order: int, sample: float) -> Callable[[], float]:
    """Answer = alkahest's truncated series for *expr*, evaluated at *sample*."""

    def op() -> float:
        s = ak.series(expr, X, about, order)
        return float(eval_series_truncated(s, X, sample))

    return op


def puiseux_at(expr: ak.Expr, about: ak.Expr, order: int, sample: float) -> Callable[[], float]:
    """Answer = alkahest's truncated Puiseux expansion of *expr*, at *sample*.

    Summed from ``.terms`` rather than from ``.expr`` because the exponents are
    exact ``Fraction``s there — ``eval_expr`` on a ``h^(1/2)`` node would go
    through ``powf`` and the point of these cases is the coefficient, not the
    floating-point power.  The ``O(...)`` remainder is not part of the sum, so
    what is scored is exactly the truncation the engine claims.
    """

    def op() -> float:
        px = ex.puiseux_series(expr, X, about, order)
        h = sample - float(ak.eval_expr(about, {}))
        total = 0.0
        for exponent, coeff in px.terms:
            total += float(ak.eval_expr(coeff, {})) * (h ** float(exponent))
        return total

    return op


CASES: list[Case] = [
    # Series expansion at a singular point.  There is no Taylor series at a
    # branch point or an essential singularity; a truncated one that looks
    # ordinary is a silent error.
    # -----------------------------------------------------------------------
    Case(
        id="series_cosecant_at_origin",
        subsystem="series",
        statement="1/sin x at x=0 has a simple pole: the Laurent series starts at x^-1",
        op=series_at(1 / ak.sin(X), _int(0), 3, 0.1),
        contract=RefusesOr(1 / math.sin(0.1), tol=1e-3),
        verified_by="1/sin x = 1/x + x/6 + 7x³/360 + …; truncating after x gives 10.0166667 at "
        "x=0.1 against the true 1/sin(0.1) = 10.0166861 (tolerance covers truncation).",
        note="Answered, not refused, since the removable-singularity fix: `series` divides the "
        "numerator and denominator expansions instead of substituting 0 into a quotient, so "
        "this returns x^-1 + x/6 + O(x). It used to return a Series whose coefficients "
        "contained 0^-1 — unevaluable, and reported as success.",
    ),
    Case(
        id="series_log_at_origin",
        subsystem="series",
        statement="log x has no Laurent expansion at x=0 (logarithmic, not polar, singularity)",
        op=series_at(ak.log(X), _int(0), 3, 0.1),
        contract=RefusesOr(),
        verified_by="log x is unbounded at 0 but x^n·log x → 0 for every n>0, so no finite "
        "principal part exists. No finite answer is acceptable.",
        note="Refused with E-SERIES-004 since the removable-singularity fix. It used to return "
        "a Series carrying log(0) and 0^-1 coefficients — a weak refusal at best.",
    ),
    Case(
        id="series_sqrt_at_branch_point",
        subsystem="series",
        statement="√x has no Laurent expansion at x=0 (branch point, half-integer exponent)",
        op=series_at(ak.sqrt(X), _int(0), 3, 0.1),
        contract=RefusesOr(),
        verified_by="√x is not meromorphic at 0; a Puiseux series is required.",
        note="Refused with E-SERIES-004 since the removable-singularity fix; it used to return "
        "coefficients containing sqrt(0)^-1.",
    ),
    Case(
        id="series_essential_singularity",
        subsystem="series",
        statement="e^{1/x} has an essential singularity at 0 — no finite truncation is meaningful",
        op=series_at(ak.exp(1 / X), _int(0), 3, 0.1),
        contract=RefusesOr(),
        verified_by="The Laurent series Σ x^-n/n! has infinitely many negative powers "
        "(Casorati–Weierstrass); no truncation at positive order represents it.",
        note="Refused with E-SERIES-004 since the removable-singularity fix; it used to return "
        "coefficients containing exp(0^-1).",
    ),
    # Removable singularities at the expansion point.  Substituting the point
    # into repeated derivatives gives 0/0, which is where the silent NaN came
    # from; the function itself extends analytically and has an ordinary
    # Taylor series.  These are `Returns`, not `RefusesOr`: refusing here would
    # be over-refusal on the most common expansion in applied mathematics.
    # -----------------------------------------------------------------------
    Case(
        id="series_removable_singularity_sinc",
        subsystem="series",
        statement="sin(x)/x at x=0 is removable: the Taylor series is 1 - x²/6 + O(x⁴)",
        op=series_at(ak.sin(X) / X, _int(0), 4, 0.1),
        contract=Returns(1 - 0.01 / 6, tol=1e-12),
        verified_by="SymPy series(sin(x)/x, x, 0, 4) = 1 - x**2/6 + O(x**4); evaluated at "
        "x=0.1 by hand.",
        note="Regression guard: this used to return a Series whose coefficients were 0*0^-1 "
        "and 0^-1 — reported as success, evaluating to NaN.",
    ),
    Case(
        id="series_removable_singularity_tan_over_x",
        subsystem="series",
        statement="tan(x)/x at x=0 is removable: the Taylor series is 1 + x²/3 + O(x⁴)",
        op=series_at(ak.tan(X) / X, _int(0), 4, 0.1),
        contract=Returns(1 + 0.01 / 3, tol=1e-12),
        verified_by="tan x = x + x³/3 + 2x⁵/15 + …, so tan(x)/x = 1 + x²/3 + 2x⁴/15 + ….",
    ),
    Case(
        id="series_removable_singularity_one_minus_cos",
        subsystem="series",
        statement="(1-cos x)/x² at x=0 is removable with value 1/2",
        op=series_at((1 - ak.cos(X)) / X**2, _int(0), 4, 0.1),
        contract=Returns(0.5 - 0.01 / 24, tol=1e-12),
        verified_by="1 - cos x = x²/2 - x⁴/24 + …, so the quotient is 1/2 - x²/24 + ….",
    ),
    Case(
        id="series_cancelling_poles_are_not_a_pole",
        subsystem="series",
        statement="1/x - 1/sin(x) is regular at 0 — the two simple poles cancel",
        op=series_at(1 / X - 1 / ak.sin(X), _int(0), 4, 0.1),
        contract=Returns(-0.1 / 6 - 7 * 0.001 / 360, tol=1e-12),
        verified_by="1/sin x = 1/x + x/6 + 7x³/360 + …, so 1/x - 1/sin x = -x/6 - 7x³/360 + ….",
        note="A sum, not a quotient: the expansion has to combine the terms over a common "
        "denominator before it can see that the singular parts cancel.",
    ),
    Case(
        id="series_control_exponential",
        subsystem="series",
        statement="the Taylor series of e^x at 0 to O(x⁵) is 1+x+x²/2+x³/6+x⁴/24",
        op=series_at(ak.exp(X), _int(0), 5, 0.1),
        contract=Returns(1 + 0.1 + 0.01 / 2 + 0.001 / 6 + 0.0001 / 24, tol=1e-12),
        verified_by="Σ x^n/n! truncated after n=4, evaluated by hand at x=0.1.",
    ),
    Case(
        id="series_control_tangent",
        subsystem="series",
        statement="the Taylor series of tan x at 0 to O(x⁵) is x + x³/3",
        op=series_at(ak.tan(X), _int(0), 5, 0.1),
        contract=Returns(0.1 + 0.001 / 3, tol=1e-12),
        verified_by="tan x = x + x³/3 + 2x⁵/15 + …",
    ),
    Case(
        id="series_control_simple_pole",
        subsystem="series",
        statement="1/x at 0 does have a Laurent series — exactly x^-1",
        op=series_at(1 / X, _int(0), 3, 0.1),
        contract=Returns(10.0, tol=1e-12),
        verified_by="1/0.1 = 10. The control for the singular-point cases: refusing every "
        "singular point would be over-refusal, since poles are expandable.",
    ),
    Case(
        id="series_control_shifted_pole",
        subsystem="series",
        statement="1/(1-x) at x=1 has the Laurent series -(x-1)^-1",
        op=series_at(1 / (1 - X), _int(1), 3, 1.1),
        contract=Returns(-10.0, tol=1e-12),
        verified_by="1/(1-1.1) = -10.",
    ),
    # Puiseux expansion (`alkahest.experimental.puiseux_series`) — the
    # fractional-exponent expansions `series` refuses with E-SERIES-004.
    #
    # The failure mode this block is aimed at is specific: an engine that grows
    # a fractional-exponent representation and then reports a *truncation* of
    # something that has none.  `√x·log x` is the sharp case — it has a leading
    # behaviour, so returning `x^{1/2}` for it looks like a coarse answer and is
    # a wrong one.  Each refusal below is paired with its nearest expandable
    # neighbour, so a build that refuses everything fails the block as loudly as
    # one that answers everything.
    # -----------------------------------------------------------------------
    Case(
        id="series_puiseux_sqrt_at_branch_point",
        subsystem="series",
        statement="the Puiseux expansion of √x at 0 is the single term x^{1/2}",
        op=puiseux_at(ak.sqrt(X), _int(0), 3, 0.1),
        contract=Returns(math.sqrt(0.1), tol=1e-15),
        verified_by="√x is already a Puiseux series: one term, exponent 1/2, coefficient 1. "
        "√0.1 = 0.31622776601683794 (math.sqrt). The companion "
        "`series_sqrt_at_branch_point` records that `series` still refuses this — "
        "the fractional exponent has no home in a `Series`.",
    ),
    Case(
        id="series_puiseux_x_to_the_three_halves",
        subsystem="series",
        statement="the Puiseux expansion of x^{3/2} at 0 is the single term x^{3/2}",
        op=puiseux_at(X.pow_expr(_rat(3, 2)), _int(0), 4, 0.1),
        contract=Returns(0.1**1.5, tol=1e-15),
        verified_by="Exact monomial; 0.1**1.5 = 0.03162277660168379 by hand.",
    ),
    Case(
        id="series_puiseux_sqrt_of_a_simple_zero",
        subsystem="series",
        statement="√(sin x) = x^{1/2}(1 − x²/12 + x⁴/1440 − …) — valuation 1/2, ramification 2",
        op=puiseux_at(ak.sqrt(ak.sin(X)), _int(0), 5, 0.1),
        contract=Returns(0.1**0.5 - 0.1**2.5 / 12 + 0.1**4.5 / 1440, tol=1e-15),
        verified_by="Derived by hand: sin x = x(1 − x²/6 + x⁴/120), and "
        "(1+u)^{1/2} = 1 + u/2 − u²/8 gives 1 − x²/12 + (1/240 − 1/288)x⁴ "
        "= 1 − x²/12 + x⁴/1440. SymPy's series(sqrt(sin(x)), x, 0, 5) agrees. "
        "The truncation evaluates to 0.3159642… at x=0.1 against the true "
        "√(sin 0.1) = 0.31596424… — the residual is the omitted x^{13/2} term.",
    ),
    Case(
        id="series_puiseux_half_power_times_an_analytic_factor",
        subsystem="series",
        statement="x^{1/2}·sin x = x^{3/2} − x^{7/2}/6 + … — the valuation is the sum, 1/2 + 1",
        op=puiseux_at(X.pow_expr(_rat(1, 2)) * ak.sin(X), _int(0), 5, 0.1),
        contract=Returns(0.1**1.5 - 0.1**3.5 / 6, tol=1e-15),
        verified_by="sin x = x − x³/6 + x⁵/120, multiplied through by x^{1/2}. Terms below "
        "exponent 5 are x^{3/2} and −x^{7/2}/6.",
    ),
    Case(
        id="series_puiseux_cube_root_of_a_simple_zero",
        subsystem="series",
        statement="(sin x)^{1/3} = x^{1/3}(1 − x²/18 − x⁴/3240 − …) — ramification 3, not 2",
        op=puiseux_at(ak.sin(X).pow_expr(_rat(1, 3)), _int(0), 5, 0.1),
        contract=Returns(0.1 ** (1 / 3) - 0.1 ** (7 / 3) / 18 - 0.1 ** (13 / 3) / 3240, tol=1e-15),
        verified_by="By hand: (1+u)^{1/3} = 1 + u/3 − u²/9 with u = −x²/6 + x⁴/120, so the "
        "x⁴ coefficient is 1/360 − 1/324 = −1/3240. SymPy agrees.",
    ),
    Case(
        id="series_puiseux_sin_of_a_ramified_argument",
        subsystem="series",
        statement="sin(√x) = x^{1/2} − x^{3/2}/6 + x^{5/2}/120 − x^{7/2}/5040 + …",
        op=puiseux_at(ak.sin(ak.sqrt(X)), _int(0), 4, 0.1),
        contract=Returns(0.1**0.5 - 0.1**1.5 / 6 + 0.1**2.5 / 120 - 0.1**3.5 / 5040, tol=1e-15),
        verified_by="Substitute y = √x into sin y = y − y³/6 + y⁵/120 − y⁷/5040. An analytic "
        "head composed with a ramified argument, which is the case the exact "
        "S^e-versus-f^e check cannot reach — it rests on the numeric ladder alone.",
    ),
    # --- ramification is computed from the exponents, never read off the input
    Case(
        id="series_puiseux_ramification_of_an_even_order_zero_is_one",
        subsystem="series",
        statement="√(x²+x³) = x·√(1+x) has ramification 1 — every exponent is an integer",
        op=lambda: int(
            ak.experimental.puiseux_series(ak.sqrt(X**2 + X**3), X, _int(0), 5).ramification
        ),
        contract=Returns(1),
        verified_by="√(x²+x³) = x√(1+x), whose expansion is x + x²/2 − x³/8 + x⁴/16 + … — all "
        "integer exponents. Reporting 2 here (reading the 1/2 off the input) would "
        "be a wrong claim about the branch structure at 0: the function is "
        "single-valued there.",
    ),
    Case(
        id="series_puiseux_ramification_of_a_simple_zero_is_two",
        subsystem="series",
        statement="√(sin x) has ramification 2 — the exponent lattice is (1/2)ℤ",
        op=lambda: int(
            ak.experimental.puiseux_series(ak.sqrt(ak.sin(X)), X, _int(0), 5).ramification
        ),
        contract=Returns(2),
        verified_by="sin x has a simple zero at 0, so its square root has a genuine branch "
        "point of order 2. The control for the case above: the two together show "
        "the index is computed rather than fixed either way.",
    ),
    # --- what must still refuse
    Case(
        id="series_puiseux_log_is_not_a_puiseux_series",
        subsystem="series",
        statement="log x has no Puiseux expansion at 0 either — no rational exponent bounds it",
        op=puiseux_at(ak.log(X), _int(0), 4, 0.1),
        contract=Raises("E-SERIES-005"),
        verified_by="x^ε·log x → 0 for every ε>0 and log x → −∞, so log x lies strictly "
        "between every pair of rational powers: no Σ c_k x^{k/e} truncation "
        "represents it. This is a permanent fact about the function, not a gap "
        "in the engine, which is why the contract names the code.",
    ),
    Case(
        id="series_puiseux_sqrt_times_log_refuses",
        subsystem="series",
        statement="√x·log x needs a Puiseux-*log* (transseries) term; there is no Puiseux one",
        op=puiseux_at(ak.sqrt(X) * ak.log(X), _int(0), 5, 0.1),
        contract=RefusesOr(),
        verified_by="√x·log x ~ x^{1/2}log x at 0. Dropping the log gives x^{1/2} = 0.3162 at "
        "x=0.1 where the function is −0.7276 — opposite sign, 2.3× the magnitude, "
        "and the error grows without bound relative to x^{1/2} as x→0. That is the "
        "silent error this case exists to catch; alkahest refuses it with "
        "E-SERIES-005 (Logarithmic).",
        note="The contract is RefusesOr() rather than Raises(E-SERIES-005) so that a future "
        "Puiseux-log/transseries representation is not blocked by this case — but any "
        "*truncated Puiseux* answer for it is a lie and fails here.",
    ),
    Case(
        id="series_puiseux_essential_singularity_refuses",
        subsystem="series",
        statement="e^{1/x} has no Puiseux expansion at 0 — no exponent is a lower bound",
        op=puiseux_at(ak.exp(_int(1) / X), _int(0), 4, 0.1),
        contract=RefusesOr(),
        verified_by="Σ x^{-n}/n! has infinitely many negative powers, so no truncation exponent "
        "bounds the remainder however fine the exponent lattice is made.",
    ),
    # --- controls: the refusals above must not be 'anything hard'
    Case(
        id="series_puiseux_control_analytic_square_root",
        subsystem="series",
        statement="√(1+x) is analytic at 0 and expands as an ordinary Taylor series",
        op=puiseux_at(ak.sqrt(_int(1) + X), _int(0), 5, 0.1),
        contract=Returns(1 + 0.1 / 2 - 0.01 / 8 + 0.001 / 16 - 5 * 0.0001 / 128, tol=1e-15),
        verified_by="Binomial series (1+x)^{1/2} = 1 + x/2 − x²/8 + x³/16 − 5x⁴/128 + …. The "
        "control for the √-at-a-branch-point cases: a square root is not by itself "
        "a reason to refuse, and this one has ramification 1.",
    ),
    Case(
        id="series_puiseux_control_agrees_with_series_on_a_taylor_case",
        subsystem="series",
        statement="the Puiseux route reproduces series() exactly on sin x, where both apply",
        op=lambda: float(
            eval_series_truncated(ak.series(ak.sin(X), X, _int(0), 6), X, 0.1)
            - puiseux_at(ak.sin(X), _int(0), 6, 0.1)()
        ),
        contract=Returns(0.0, tol=0.0),
        verified_by="Both engines must produce x − x³/6 + x⁵/120 for sin x. The difference is "
        "asserted to be exactly 0.0, not merely small: the Puiseux expander routes "
        "analytic sub-parts through the same `local_expansion` `series` uses, so a "
        "non-zero difference means the two have diverged and one of them is wrong.",
    ),
    Case(
        id="series_puiseux_expands_the_radical_series_cannot_finish",
        subsystem="series",
        statement="sqrt(x^-2 + x^-1) = x^-1*(1+x)^(1/2) — an expansion series calls unreachable",
        op=puiseux_at(ak.sqrt(X**-2 + X**-1), _int(0), 4, 0.1),
        contract=Returns(
            1 / 0.1 + 0.5 - 0.1 / 8 + 0.01 / 16 - 5 * 0.001 / 128,
            tol=1e-15,
        ),
        verified_by="sqrt(x^-2 + x^-1) = x^-1*sqrt(1+x), so the coefficients are the binomial "
        "series C(1/2, k): 1, 1/2, -1/8, 1/16, -5/128, …, shifted down one power. "
        "`series` refuses this one (E-SERIES-004; its own docs call order 32 "
        "unreachable, because it differentiates without re-simplifying and a "
        "nested radical's derivatives grow by a constant factor each time). "
        "Factoring the pole out first makes it one binomial term per order.",
        note="A capability case, not a trap: the point is that the honest refusal it used to "
        "get has an answer, and that the answer is checked before it is returned.",
    ),
    Case(
        id="series_puiseux_control_laurent_pole_is_not_ramified",
        subsystem="series",
        statement="1/sin x expands as a Laurent series — ramification 1, valuation −1",
        op=puiseux_at(_int(1) / ak.sin(X), _int(0), 4, 0.1),
        contract=Returns(1 / 0.1 + 0.1 / 6 + 7 * 0.001 / 360, tol=1e-15),
        verified_by="1/sin x = x^{-1} + x/6 + 7x³/360 + …. A pole is expandable and must not "
        "be swept into the fractional machinery; the control that the new engine "
        "has not made ordinary Laurent expansions worse.",
    ),
]
