"""Silent-error cases for validated.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import alkahest as ak
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import X, _int, _rat


def _encloses(build: Callable[[], Any], truth: float) -> Callable[[], bool]:
    """Answer = does the returned ``Enclosure`` contain *truth*?"""

    def op() -> bool:
        r = build()
        return bool(r.lower <= truth <= r.upper)

    return op


CASES: list[Case] = [
    Case(
        id="validated_integral_removable_log_quotient",
        subsystem="validated",
        statement="int_0^1 log(1+x)/x dx = pi^2/12; the enclosure must contain it",
        op=_encloses(
            lambda: ak.verified_integral(ak.log(_int(1) + X) / X, X, 0.0, 1.0),
            math.pi**2 / 12,
        ),
        contract=Returns(True),
        verified_by=(
            "Expanding log(1+x)/x = sum_{n>=1} (-1)^(n+1) x^(n-1)/n and integrating term by "
            "term gives sum (-1)^(n+1)/n^2 = eta(2) = pi^2/12 = 0.8224670334241132.... The "
            "integrand is singular only as an *expression*; it extends continuously by 1 at "
            "x = 0, so refusing here would be a coverage regression rather than a lie."
        ),
    ),
    Case(
        id="validated_integral_removable_at_a_grid_point",
        subsystem="validated",
        statement="int_0^2 (x^2-1)/(x-1) dx = 4, integrating the continuous extension x+1",
        op=_encloses(
            lambda: ak.verified_integral((X * X - _int(1)) / (X - _int(1)), X, 0.0, 2.0),
            4.0,
        ),
        contract=Returns(True),
        verified_by=(
            "(x^2-1)/(x-1) = x+1 for every x != 1, and int_0^2 (x+1) dx = [x^2/2 + x]_0^2 = 4 "
            "exactly. The singular point x = 1 is the midpoint of the interval, so it lies on "
            "the bisection grid -- the easy half of the removable-singularity path, and the "
            "control for the pole cases below."
        ),
    ),
    Case(
        id="validated_integral_pole_inverse_square",
        subsystem="validated",
        statement="int_0^2 (x-1)^-2 dx diverges (double pole at x = 1)",
        op=lambda: float(
            ak.verified_integral(_int(1) / (X - _int(1)) ** _int(2), X, 0.0, 2.0).lower
        ),
        contract=Raises("E-VALIDATED-003"),
        verified_by=(
            "int (x-1)^-2 dx = -1/(x-1), so the naive FTC gives -1 - 1 = -2 -- a clean, "
            "plausible, negative number for the integral of a strictly positive function. The "
            "true value is +infinity: the integrand is ~ t^-2 on both sides of 1. This is the "
            "archetype the whole gate is named for, asked of the rigorous integrator."
        ),
    ),
    Case(
        id="validated_integral_simple_pole_off_the_bisection_grid",
        subsystem="validated",
        statement="int_0^1 dx/(x - 1/3) diverges; 1/3 is never a dyadic bisection point",
        op=lambda: float(ak.verified_integral(_int(1) / (X - _rat(1, 3)), X, 0.0, 1.0).lower),
        contract=Raises("E-VALIDATED-003"),
        verified_by=(
            "The one-sided integrals are -inf and +inf, so the integral does not converge; its "
            "Cauchy principal value is log(2) = 0.6931471805..., which is the plausible wrong "
            "answer. 1/3 has no finite binary expansion, so no bisection of [0, 1] ever lands "
            "on it: the singular point has to be found by the Newton search on the denominator "
            "and then rejected, because the numerator 1 does not vanish there."
        ),
    ),
    Case(
        id="validated_integral_double_zero_denominator_refused",
        subsystem="validated",
        statement="int_-1^1 sin(x)/x^2 dx diverges: the denominator has a double zero",
        op=lambda: float(ak.verified_integral(ak.sin(X) / (X * X), X, -1.0, 1.0).lower),
        contract=Raises("E-VALIDATED-003"),
        verified_by=(
            "sin(x)/x^2 = 1/x - x/6 + ... near 0, so neither one-sided integral converges. The "
            "odd symmetry makes 0 the plausible wrong answer, and it is exactly what a routine "
            "that cancelled one power of x without checking the order would report. The "
            "removable-singularity path must decline because D' = 2x vanishes at the singular "
            "point, which is what separates this from sin(x)/x."
        ),
    ),
    Case(
        id="validated_integral_integrable_endpoint_singularity_refused",
        subsystem="validated",
        statement="int_0^1 -log(x) dx = 1 converges, but is not a removable N/D quotient",
        op=lambda: float(ak.verified_integral(_int(-1) * ak.log(X), X, 0.0, 1.0).lower),
        contract=RefusesOr(1.0),
        verified_by=(
            "int_0^1 -log x dx = [x - x log x]_0^1 = 1 exactly (the x log x term tends to 0). "
            "The integral exists, so 1 is a correct answer if it can be certified; today it "
            "cannot -- log's enclosure reaches 0 at the endpoint and the integrand is not a "
            "0/0 quotient -- and the refusal is honest. Any *other* finite number is a lie."
        ),
        note=(
            "Documented limitation (docs/mdbook/src/validated-bounds.md, 'What is still "
            "refused'). Paired with validated_integral_removable_log_quotient, the nearest "
            "convergent neighbour, which must keep working."
        ),
    ),
    Case(
        id="validated_bound_interior_pole_refuses",
        subsystem="validated",
        statement="the range of 1/x over [-1, 1] is not a bounded interval",
        op=lambda: float(ak.bound_on_box(_int(1) / X, [(X, -1.0, 1.0)]).upper),
        contract=Raises("E-VALIDATED-003"),
        verified_by=(
            "1/x is unbounded above and below on [-1, 1] and undefined at 0, so no finite "
            "[lo, hi] encloses its range. A branch-and-bound that quietly dropped the "
            "sub-boxes it could not model would report the range over the rest -- a "
            "comfortable finite interval, with nothing saying part of the domain was skipped."
        ),
    ),
    Case(
        id="validated_bound_unsupported_primitive_refuses",
        subsystem="validated",
        statement="floor has a ball rule but no Taylor model; bound_on_box must refuse",
        op=lambda: float(ak.bound_on_box(ak.floor(X), [(X, 0.0, 2.5)]).upper),
        contract=Raises("E-VALIDATED-001"),
        verified_by=(
            "capabilities() reports numeric_ball = True and taylor_model = False for `floor`, "
            "and floor is not differentiable, so no Lagrange remainder exists for it at any "
            "order. Falling back to the pointwise ball rule would produce a rigorous-looking "
            "range that is really just an evaluation over the box hull, with none of the "
            "subdivision guarantees the caller of bound_on_box is relying on."
        ),
    ),
    Case(
        id="validated_bound_starved_budget_still_encloses",
        subsystem="validated",
        statement="bound_on_box(exp, [-5,5], max_subdivisions=0) is wide but must contain e^5",
        op=_encloses(
            lambda: ak.bound_on_box(ak.exp(X), [(X, -5.0, 5.0)], max_subdivisions=0),
            148.4131591025766,
        ),
        contract=Returns(True),
        verified_by=(
            "exp(5) = 148.41315910257660342... (mpmath, 50 digits) is attained at the right "
            "endpoint, so it is in the range and must be in any enclosure of it. Exhausting "
            "the work budget is documented as *not* an error -- the result comes back with "
            "budget_exhausted = True and is still sound -- and this case is what makes that "
            "promise testable rather than aspirational."
        ),
    ),
    Case(
        id="validated_integral_starved_budget_still_encloses",
        subsystem="validated",
        statement="int_0^5 e^x dx = e^5 - 1 must be enclosed even with zero subdivisions",
        op=_encloses(
            lambda: ak.verified_integral(ak.exp(X), X, 0.0, 5.0, max_subdivisions=0),
            147.4131591025766,
        ),
        contract=Returns(True),
        verified_by=(
            "int_0^5 e^x dx = e^5 - 1 = 147.41315910257660342... exactly. With no subdivisions "
            "the single Taylor model over the whole interval gives a very wide interval; wide "
            "is fine, not-containing is not."
        ),
    ),
    Case(
        id="validated_bound_tan_up_to_the_pole",
        subsystem="validated",
        statement="tan on [1, 1.5707963267948966] reaches 1.6331239353195370e16",
        op=_encloses(
            lambda: ak.bound_on_box(ak.tan(X), [(X, 1.0, 1.5707963267948966)]),
            1.633123935319537e16,
        ),
        contract=Returns(True),
        verified_by=(
            "The f64 literal 1.5707963267948966 is exactly "
            "1.5707963267948965579989817342720925808, which is 6.123234e-17 *below* pi/2, so "
            "tan is finite on the closed box and tan of that endpoint is "
            "1.6331239353195369756e16 (mpmath, 60 digits). A bound therefore exists, and it "
            "is enormous -- which is the point: a routine that clipped, overflowed or bisected "
            "away from the endpoint would report a comfortable finite maximum for a function "
            "that is 10^16 at the edge of the box."
        ),
    ),
    Case(
        id="validated_no_roots_control_root_free_box",
        subsystem="validated",
        statement="x^2 + 1 has no real root, so verified_no_roots on [-5, 5] is 'true'",
        op=lambda: str(ak.verified_no_roots(X * X + _int(1), [(X, -5.0, 5.0)])),
        contract=Returns("true"),
        verified_by=(
            "x^2 + 1 >= 1 > 0 for every real x. The control for the three-valued predicate: a "
            "verdict function that answered 'undecided' to everything would be perfectly sound "
            "and perfectly useless, and only a positive case catches that."
        ),
    ),
    Case(
        id="validated_no_roots_even_count_still_false",
        subsystem="validated",
        statement="x^2 - 2 has two roots in [-2, 2] and both endpoints are positive",
        op=lambda: str(ak.verified_no_roots(X * X - _int(2), [(X, -2.0, 2.0)])),
        contract=Returns("false"),
        verified_by=(
            "+-sqrt(2) = +-1.41421356... both lie in [-2, 2], while f(-2) = f(2) = 2 > 0. An "
            "endpoint-only sign test sees no change and would answer 'true' -- a *certified* "
            "claim that a box containing two roots is root-free, which is the worst shape a "
            "verdict can have."
        ),
    ),
    Case(
        id="validated_no_roots_double_root_at_the_centre",
        subsystem="validated",
        statement="(x-1)^2 has a root at the centre of [0, 2] and never changes sign",
        op=lambda: str(ak.verified_no_roots((X - _int(1)) ** _int(2), [(X, 0.0, 2.0)])),
        contract=Returns("false"),
        verified_by=(
            "(1-1)^2 = 0, so x = 1 is a root of even multiplicity and it is the exact midpoint "
            "of [0, 2]. There is no sign change anywhere, so the intermediate value theorem "
            "cannot see it; the verdict has to come from exact substitution at a distinguished "
            "point of the box."
        ),
    ),
    Case(
        id="validated_sign_tangent_at_the_endpoint_true",
        subsystem="validated",
        statement="Cusa-Huygens: x(2 + cos x) - 3 sin x >= 0 on [0, 1.5], tight at x = 0",
        op=lambda: str(
            ak.verified_sign(
                X * (_int(2) + ak.cos(X)) - _int(3) * ak.sin(X), [(X, 0.0, 1.5)], "nonnegative"
            )
        ),
        contract=Returns("true"),
        verified_by=(
            "Taylor at 0: x(2 + cos x) - 3 sin x = x^5/60 - x^7/1260 + ..., leading coefficient "
            "1/60 > 0, and a 2001-point mpmath sweep of [0, 1.5] at 60 digits finds no "
            "negative value. The margin vanishes at x = 0, so every range enclosure straddles "
            "zero however fine the subdivision; only the endpoint expansion with a proven "
            "Lagrange remainder can decide it."
        ),
    ),
    Case(
        id="validated_sign_tangent_at_the_endpoint_false",
        subsystem="validated",
        statement="x - sin x - x^3/6 >= 0 is FALSE on [0, 0.5] (it is -x^5/120 + ...)",
        op=lambda: str(
            ak.verified_sign(X - ak.sin(X) - X ** _int(3) / _int(6), [(X, 0.0, 0.5)], "nonnegative")
        ),
        contract=Returns("false"),
        verified_by=(
            "x - sin x = x^3/6 - x^5/120 + x^7/5040 - ..., so the expression is "
            "-x^5/120 + x^7/5040 - ... which is < 0 throughout (0, 0.5]; at x = 0.5 mpmath "
            "gives -2.58872e-4. The mirror image of the Cusa-Huygens case -- same shape, same "
            "tangency at the endpoint, opposite leading sign -- so an endpoint expansion that "
            "misread the leading coefficient's sign would certify a false inequality here."
        ),
    ),
    Case(
        id="validated_sign_interior_tangency_stays_undecided",
        subsystem="validated",
        statement="(x - 7/10)^2 (x+1) >= 0 on [0, 3/2] touches zero in the interior",
        op=lambda: str(
            ak.verified_sign(
                (X - _rat(7, 10)) ** _int(2) * (X + _int(1)), [(X, 0.0, 1.5)], "nonnegative"
            )
        ),
        contract=Returns("undecided"),
        verified_by=(
            "The expression is a square times (x+1) > 0 on [0, 3/2], so it is non-negative "
            "there, and it vanishes at the interior point x = 7/10. The statement is TRUE and "
            "'undecided' is the honest answer, because the endpoint expansion does not reach "
            "an interior tangency. The case pins that down: upgrading it to 'true' on the "
            "strength of an enclosure that merely touches zero would let every "
            "grazing-but-negative expression certify too."
        ),
        note=(
            "Contract is Returns('undecided') deliberately. A later improvement that genuinely "
            "proves it 'true' should trip this case and be reviewed rather than pass silently."
        ),
    ),
    Case(
        id="validated_sign_strict_fails_where_the_margin_vanishes",
        subsystem="validated",
        statement="x - sin x > 0 is FALSE on [0, 1]: the expression is 0 at x = 0",
        op=lambda: str(ak.verified_sign(X - ak.sin(X), [(X, 0.0, 1.0)], "positive")),
        contract=Returns("false"),
        verified_by=(
            "x - sin x >= 0 on [0, 1] with equality exactly at x = 0, so the non-strict "
            "inequality holds and the strict one does not. Reporting 'true' for the strict "
            "form is the classic boundary error, and it is the kind that survives every "
            "numerical spot-check that happens to avoid the endpoint."
        ),
    ),
]
