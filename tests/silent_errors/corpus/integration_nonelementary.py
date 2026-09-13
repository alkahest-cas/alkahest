"""Silent-error cases for integration nonelementary.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Callable

import alkahest as ak
from contracts import Case, Measured, Raises, Returns

from ._shared import X, _num


def antiderivative_slope(integrand: ak.Expr, at: float) -> Callable[[], Measured]:
    """Answer = d/dx of alkahest's antiderivative, evaluated at *at*.

    This is the fundamental theorem of calculus used as a checker: it is immune
    to ``+C`` and to every legitimate difference in antiderivative form, and it
    catches the one thing that matters — an antiderivative whose derivative is
    not the integrand.  A refusal (``E-INT-004``) surfaces as a refusal, so this
    also detects a *false* non-elementarity verdict, which is exactly as
    damaging as a wrong formula (report7-20.md B2).
    """

    def op() -> Measured:
        r = ak.integrate(integrand, X)
        slope = ak.diff(r.value, X).value
        return Measured(float(ak.eval_expr(slope, {X: at})), r.verification)

    return op


#: The registered non-elementary output basis.  An antiderivative naming one of
#: these is, by construction, not an elementary function — which is how the
#: "no elementary antiderivative" claim survives being *answered* instead of
#: refused.  Mirrors ``SPECIAL_BASIS`` in ``integrate/special.rs``.
_NONELEMENTARY_BASIS = (
    "Ei",
    "li",
    "Si",
    "Ci",
    "Shi",
    "Chi",
    "erf",
    "erfc",
    "fresnels",
    "fresnelc",
    "dilog",
    "EllipticF",
    "EllipticE",
    "EllipticK",
)


def nonelementary_closed_form_slope(integrand: ak.Expr, at: float) -> Callable[[], Measured]:
    """Answer = d/dx of alkahest's antiderivative at *at*, for an integrand
    that has **no elementary** antiderivative but does have a closed form over
    the registered special-function basis.

    Two traps in one, and the second is the reason this helper exists rather
    than :func:`antiderivative_slope`:

    * the derivative must be the integrand — a wrong closed form is a wrong
      theorem, exactly as a wrong elementary one would be; and
    * the antiderivative must still **name** a non-elementary function.
      Returning something elementary here would be the assertion *"this
      integral is elementary"*, which is false — the same silent error the
      ``Raises("E-INT-004")`` contract used to catch, in the shape it takes
      once refusal is replaced by emission.
    """

    def op() -> Measured:
        r = ak.integrate(integrand, X)
        shown = str(r.value)
        if not any(name in shown for name in _NONELEMENTARY_BASIS):
            raise AssertionError(f"antiderivative {shown} is elementary — the integral is not")
        slope = ak.diff(r.value, X).value
        return Measured(float(ak.eval_expr(slope, {X: at})), r.verification)

    return op


def _exp_log_sum(x: float) -> float:
    """d/dx [e^x·log x] = e^x·log x + e^x/x."""
    return math.exp(x) * math.log(x) + math.exp(x) / x


def _risch_gaussian_pair(x: float) -> float:
    """d/dx [x·e^{x²}] = e^{x²} + 2x²·e^{x²}."""
    return math.exp(x * x) + 2 * x * x * math.exp(x * x)


def _sin_log_pair(x: float) -> float:
    """d/dx [sin x·log x] = cos x·log x + sin x / x."""
    return math.cos(x) * math.log(x) + math.sin(x) / x


CASES: list[Case] = [
    # Non-elementarity — in BOTH directions.  A wrong antiderivative and a
    # false "provably non-elementary" verdict are equally poisonous: the second
    # tells a search loop that a branch is permanently closed when it is not.
    # -----------------------------------------------------------------------
    Case(
        id="nonelementary_exp_x_squared",
        subsystem="integration_nonelementary",
        statement="∫ e^{x²} dx has no elementary antiderivative",
        op=lambda: _num(ak.integrate(ak.exp(X**2), X).value),
        contract=Raises("E-INT-004"),
        verified_by="Liouville/Risch: the antiderivative is (√π/2)·erfi(x); erfi is not "
        "elementary. Standard textbook example.",
        benchmark_tasks=("nonelementary_expx2",),
    ),
    # The four below are *answered* rather than refused since 3.10.0: each has a
    # closed form over the registered special-function basis.  The trap is
    # unchanged in content — the claim "no elementary antiderivative exists" is
    # still pinned, now by requiring the answer to name a non-elementary
    # function rather than by requiring a refusal.  See
    # `nonelementary_closed_form_slope`.
    Case(
        id="nonelementary_gaussian",
        subsystem="integration_nonelementary",
        statement="∫ e^{-x²} dx has no elementary antiderivative; it is (√π/2)·erf(x)",
        op=nonelementary_closed_form_slope(ak.exp(-(X**2)), 0.5),
        contract=Returns(0.7788007830714049),
        verified_by="(√π/2)·erf(x); erf is not elementary (Liouville). "
        "d/dx at 0.5 is e^{-0.25} = 0.7788007830714049.",
    ),
    Case(
        id="nonelementary_sinc",
        subsystem="integration_nonelementary",
        statement="∫ sin(x)/x dx has no elementary antiderivative; it is Si(x)",
        op=nonelementary_closed_form_slope(ak.sin(X) / X, 1.0),
        contract=Returns(0.8414709848078965),
        verified_by="Si(x), the sine integral — a special function, not elementary. "
        "d/dx at 1 is sin(1)/1 = 0.8414709848078965.",
    ),
    Case(
        id="nonelementary_logarithmic_integral",
        subsystem="integration_nonelementary",
        statement="∫ dx/log x has no elementary antiderivative; it is li(x)",
        op=nonelementary_closed_form_slope(1 / ak.log(X), 2.0),
        contract=Returns(1.4426950408889634),
        verified_by="li(x), the logarithmic integral. d/dx at 2 is 1/log 2 = 1.4426950408889634.",
    ),
    Case(
        id="nonelementary_exponential_integral",
        subsystem="integration_nonelementary",
        statement="∫ e^x/x dx has no elementary antiderivative; it is Ei(x)",
        op=nonelementary_closed_form_slope(ak.exp(X) / X, 2.0),
        contract=Returns(3.6945280494653252),
        verified_by="Ei(x), the exponential integral. d/dx at 2 is e²/2 = 3.6945280494653252.",
    ),
    Case(
        id="nonelementary_double_exponential",
        subsystem="integration_nonelementary",
        statement="∫ e^{e^x} dx has no elementary antiderivative",
        op=lambda: _num(ak.integrate(ak.exp(ak.exp(X)), X).value),
        contract=Raises("E-INT-004"),
        verified_by="Reduces to Ei(e^x) under u = e^x.",
    ),
    Case(
        id="elementary_sum_of_two_nonelementary_parts",
        subsystem="integration_nonelementary",
        statement="∫ (e^{x²} + 2x²e^{x²}) dx = x·e^{x²} — elementary, though each summand is not",
        op=antiderivative_slope(ak.exp(X**2) + 2 * X**2 * ak.exp(X**2), 0.5),
        contract=Returns(_risch_gaussian_pair(0.5)),
        verified_by="Product rule: d/dx[x·e^{x²}] = e^{x²} + 2x²e^{x²}. The textbook "
        "counterexample to term-by-term non-elementarity reasoning.",
    ),
    Case(
        id="elementary_exp_times_log_sum",
        subsystem="integration_nonelementary",
        statement="∫ (e^x·log x + e^x/x) dx = e^x·log x — the report7-20 B2 regression",
        op=antiderivative_slope(ak.exp(X) * ak.log(X) + ak.exp(X) / X, 2.0),
        contract=Returns(_exp_log_sum(2.0)),
        verified_by="Product rule: d/dx[e^x log x] = e^x log x + e^x/x. alkahest 3.6.0 "
        "returned a *false* E-INT-004 'no elementary antiderivative exists' here "
        "(report7-20.md, bug B2); this pins the fix.",
    ),
    Case(
        id="elementary_sin_times_log_sum",
        subsystem="integration_nonelementary",
        statement="∫ (cos x·log x + sin x/x) dx = sin x·log x — elementary, parts are not",
        op=antiderivative_slope(ak.cos(X) * ak.log(X) + ak.sin(X) / X, 2.0),
        contract=Returns(_sin_log_pair(2.0)),
        verified_by="Product rule: d/dx[sin x·log x] = cos x·log x + sin x/x. ∫sin x/x alone "
        "is Si(x) and non-elementary.",
    ),
    Case(
        id="elementary_x_log_x",
        subsystem="integration_nonelementary",
        statement="∫ x·log x dx = x²(2log x - 1)/4",
        op=antiderivative_slope(X * ak.log(X), 2.0),
        contract=Returns(2.0 * math.log(2.0)),
        verified_by="Integration by parts; checked by differentiating back.",
        verification_floor="numerically_checked",
    ),
    Case(
        id="elementary_cubic_partial_fractions",
        subsystem="integration_nonelementary",
        statement="∫ dx/(1+x³) is elementary (log + arctan)",
        op=antiderivative_slope(1 / (1 + X**3), 0.5),
        contract=Returns(1.0 / (1.0 + 0.125)),
        verified_by="1+x³ factors over ℚ as (x+1)(x²-x+1); partial fractions give logs and an "
        "arctan. Checked by differentiating back.",
    ),
    Case(
        id="elementary_circular_arc",
        subsystem="integration_nonelementary",
        statement="∫ √(1-x²) dx = [x√(1-x²) + arcsin x]/2",
        op=antiderivative_slope(ak.sqrt(1 - X**2), 0.5),
        contract=Returns(math.sqrt(1 - 0.25)),
        verified_by="Trigonometric substitution x = sin θ; checked by differentiating back.",
    ),
    Case(
        id="elementary_tangent",
        subsystem="integration_nonelementary",
        statement="∫ tan x dx = -log|cos x|",
        op=antiderivative_slope(ak.tan(X), 1.0),
        contract=Returns(math.tan(1.0)),
        verified_by="u = cos x. Sample point 1.0 rad keeps cos x > 0.",
    ),
    Case(
        id="elementary_x_exp_x",
        subsystem="integration_nonelementary",
        statement="∫ x·e^x dx = (x-1)e^x",
        op=antiderivative_slope(X * ak.exp(X), 1.5),
        contract=Returns(1.5 * math.exp(1.5)),
        verified_by="Integration by parts; d/dx[(x-1)e^x] = x·e^x.",
        verification_floor="numerically_checked",
    ),
]
