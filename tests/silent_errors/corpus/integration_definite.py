"""Silent-error cases for integration definite.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Callable

import alkahest as ak
from contracts import Case, Measured, Raises, RefusesOr, Returns

from ._shared import CALCULUS, POOL, X, _int, _num, _rat, _survives_a_panic


def definite(integrand: ak.Expr, lo: ak.Expr, hi: ak.Expr) -> Callable[[], Measured]:
    """Answer = the numeric value of ∫_lo^hi integrand dx."""

    def op() -> Measured:
        r = ak.integrate(integrand, X, lo, hi)
        return Measured(_num(r.value), r.verification)

    return op


_LN2 = math.log(2.0)
#: A free *parameter*, distinct from the integration variable ``X``.
_A_PARAM = POOL.symbol("aparam")


def parametric_definite(
    integrand: ak.Expr, lo: ak.Expr, hi: ak.Expr, at: float
) -> Callable[[], float]:
    """Answer = ∫_lo^hi integrand dx, with the parameter ``aparam`` set to *at*.

    A parametric answer must be scored at a concrete parameter value, not left
    symbolic: an expression with an unbound symbol fails ``eval_expr`` and would
    score as a *refusal*, hiding the very thing under test.  The library's
    contract here is that the closed form is returned unconditionally, so
    substituting afterwards is exactly what a caller does with it.
    """

    def op() -> float:
        r = ak.integrate(integrand, X, lo, hi)
        return float(ak.eval_expr(r.value, {_A_PARAM: at}))

    return op


CASES: list[Case] = [
    # Definite integration through an interior pole.  Naive FTC produces a
    # clean finite number for every one of these; every one of them diverges.
    # -----------------------------------------------------------------------
    Case(
        id="int_pole_inverse_square_symmetric",
        subsystem="integration_definite",
        statement="∫_{-1}^{1} x^-2 dx diverges (double pole at x=0, strictly interior)",
        op=definite(1 / X**2, _int(-1), _int(1)),
        contract=Raises("E-INT-001"),
        verified_by="∫x^-2 = -1/x; both one-sided pieces diverge to +∞. Naive FTC gives -2.",
        benchmark_tasks=("pole_interior_inverse_square",),
    ),
    Case(
        id="int_pole_inverse_symmetric",
        subsystem="integration_definite",
        statement="∫_{-1}^{1} 1/x dx diverges (simple pole at x=0); only the PV is 0",
        op=definite(1 / X, _int(-1), _int(1)),
        contract=Raises("E-INT-001"),
        verified_by="∫1/x = log|x|; -∞ + ∞ is not a value. Cauchy PV is 0, the integral is not.",
        benchmark_tasks=("pole_interior_inverse",),
    ),
    Case(
        id="int_pole_rational_at_one",
        subsystem="integration_definite",
        statement="∫_0^2 dx/(x²-1) diverges (pole at x=1, interior, not at the origin)",
        op=definite(1 / (X**2 - 1), _int(0), _int(2)),
        contract=Raises("E-INT-001"),
        verified_by="1/(x²-1) = ½[1/(x-1) - 1/(x+1)]; the 1/(x-1) piece diverges at x=1.",
        benchmark_tasks=("pole_interior_rational",),
    ),
    Case(
        id="int_pole_double_at_one",
        subsystem="integration_definite",
        statement="∫_0^2 (x-1)^-2 dx diverges (double pole at x=1)",
        op=definite(1 / (X - 1) ** 2, _int(0), _int(2)),
        contract=Raises("E-INT-001"),
        verified_by="∫(x-1)^-2 = -1/(x-1); naive FTC gives -1-1 = -2, a plausible wrong number.",
    ),
    Case(
        id="int_pole_shifted_simple",
        subsystem="integration_definite",
        statement="∫_1^3 dx/(x-2) diverges (pole at x=2, away from 0 and from both endpoints)",
        op=definite(1 / (X - 2), _int(1), _int(3)),
        contract=Raises("E-INT-001"),
        verified_by="Substituting u=x-2 gives ∫_{-1}^{1} du/u, the divergent case above.",
    ),
    Case(
        id="int_pole_on_negative_axis",
        subsystem="integration_definite",
        statement="∫_{-2}^{0} dx/(x+1) diverges (pole at x=-1)",
        op=definite(1 / (X + 1), _int(-2), _int(0)),
        contract=Raises("E-INT-001"),
        verified_by="u=x+1 gives ∫_{-1}^{1} du/u again; naive FTC gives log1-log(-1) = 0.",
    ),
    Case(
        id="int_pole_odd_cubic",
        subsystem="integration_definite",
        statement="∫_{-1}^{1} x^-3 dx diverges (triple pole at 0)",
        op=definite(1 / X**3, _int(-1), _int(1)),
        contract=Raises("E-INT-001"),
        verified_by="∫x^-3 = -1/(2x²); both sides diverge to -∞. Odd symmetry makes 0 tempting.",
    ),
    Case(
        id="int_pole_odd_rational",
        subsystem="integration_definite",
        statement="∫_{-2}^{2} x/(x²-1) dx diverges (poles at ±1); odd symmetry suggests 0",
        op=definite(X / (X**2 - 1), _int(-2), _int(2)),
        contract=Raises("E-INT-001"),
        verified_by="Antiderivative ½log|x²-1| diverges at x=±1. The integrand is odd, so a "
        "symmetry argument gives the plausible-but-wrong answer 0.",
    ),
    Case(
        id="int_pole_two_interior_poles",
        subsystem="integration_definite",
        statement="∫_{-3}^{3} dx/(x²-4) diverges (poles at x=±2, both interior)",
        op=definite(1 / (X**2 - 4), _int(-3), _int(3)),
        contract=Raises("E-INT-001"),
        verified_by="Partial fractions ¼[1/(x-2) - 1/(x+2)]; both pieces diverge.",
    ),
    Case(
        id="int_pole_product_form",
        subsystem="integration_definite",
        statement="∫_0^2 dx/(x(x-1)) diverges (poles at x=0 endpoint and x=1 interior)",
        op=definite(1 / (X * (X - 1)), _int(0), _int(2)),
        contract=Raises("E-INT-001"),
        verified_by="1/(x(x-1)) = 1/(x-1) - 1/x; both terms diverge inside [0,2].",
    ),
    Case(
        id="int_pole_reflected",
        subsystem="integration_definite",
        statement="∫_0^2 dx/(1-x) diverges (pole at x=1, sign-flipped denominator)",
        op=definite(1 / (1 - X), _int(0), _int(2)),
        contract=Raises("E-INT-001"),
        verified_by="Antiderivative -log|1-x| diverges at x=1; naive FTC gives 0.",
    ),
    Case(
        id="int_endpoint_pole_inverse",
        subsystem="integration_definite",
        statement="∫_0^1 dx/x diverges (non-integrable singularity at the lower endpoint)",
        op=definite(1 / X, _int(0), _int(1)),
        contract=Raises("E-INT-001"),
        verified_by="lim_{ε→0+} -log ε = +∞.",
    ),
    Case(
        id="int_endpoint_pole_inverse_square",
        subsystem="integration_definite",
        statement="∫_0^1 x^-2 dx diverges (double pole at the lower endpoint)",
        op=definite(1 / X**2, _int(0), _int(1)),
        contract=Raises("E-INT-001"),
        verified_by="lim_{ε→0+} (1/ε - 1) = +∞.",
    ),
    Case(
        id="int_endpoint_log_over_x",
        subsystem="integration_definite",
        statement="∫_0^1 log(x)/x dx diverges to -∞ (endpoint singularity)",
        op=definite(ak.log(X) / X, _int(0), _int(1)),
        contract=Raises("E-INT-001"),
        verified_by="∫log(x)/x = ½log²x; lim_{ε→0+} -½log²ε = -∞.",
    ),
    Case(
        id="int_pole_tangent_over_period",
        subsystem="integration_definite",
        statement="∫_0^π tan x dx diverges (pole at x=π/2, interior)",
        op=definite(ak.tan(X), _int(0), POOL.float(math.pi)),
        contract=RefusesOr(),
        verified_by="∫tan = -log|cos x|; diverges to +∞ from both sides of π/2. Symmetry about "
        "π/2 makes 0 the tempting wrong answer.",
        note="Weak refusal: alkahest returns -log(cos π) with no error; it only fails to reduce "
        "to a number because log of a negative is a domain error. A transcendental-pole "
        "check comparable to the rational-pole one would upgrade this to E-INT-001.",
    ),
    Case(
        id="int_pole_secant_over_period",
        subsystem="integration_definite",
        statement="∫_0^π dx/cos x diverges (pole at x=π/2)",
        op=definite(1 / ak.cos(X), _int(0), POOL.float(math.pi)),
        contract=RefusesOr(),
        verified_by="∫sec = log|sec x + tan x|; unbounded as x→π/2.",
        note="Weak refusal, same shape as int_pole_tangent_over_period.",
    ),
    # Improper integrals over an infinite range.
    # -----------------------------------------------------------------------
    Case(
        id="int_infinite_harmonic_tail",
        subsystem="integration_definite",
        statement="∫_1^∞ dx/x diverges",
        op=definite(1 / X, _int(1), POOL.pos_infinity()),
        contract=RefusesOr(),
        verified_by="log R → ∞. The borderline exponent: ∫x^-p converges iff p>1.",
    ),
    Case(
        id="int_infinite_linear",
        subsystem="integration_definite",
        statement="∫_0^∞ x dx diverges",
        op=definite(X, _int(0), POOL.pos_infinity()),
        contract=RefusesOr(),
        verified_by="∫_0^R x dx = R²/2, which grows without bound as R → ∞.",
    ),
    Case(
        id="int_infinite_oscillatory",
        subsystem="integration_definite",
        statement="∫_0^∞ sin x dx does not converge (1-cos R has no limit)",
        op=definite(ak.sin(X), _int(0), POOL.pos_infinity()),
        contract=RefusesOr(),
        verified_by="∫_0^R sin = 1-cos R oscillates in [0,2]. Abel/Cesàro summation gives 1, "
        "which is exactly the plausible wrong answer.",
    ),
    Case(
        id="int_infinite_exponential_growth",
        subsystem="integration_definite",
        statement="∫_0^∞ e^x dx diverges",
        op=definite(ak.exp(X), _int(0), POOL.pos_infinity()),
        contract=RefusesOr(),
        verified_by="e^R - 1 → ∞.",
    ),
    # Integration controls: convergent integrals that must NOT be over-refused.
    # A gate made only of refusals is passed by a library that refuses
    # everything, so each refusal class needs its nearest convergent neighbour.
    # -----------------------------------------------------------------------
    Case(
        id="int_control_polynomial",
        subsystem="integration_definite",
        statement="∫_0^1 x² dx = 1/3",
        op=definite(X**2, _int(0), _int(1)),
        contract=Returns(1 / 3),
        verified_by=CALCULUS,
    ),
    Case(
        id="int_control_arctangent_symmetric",
        subsystem="integration_definite",
        statement="∫_{-1}^{1} dx/(1+x²) = π/2",
        op=definite(1 / (X**2 + 1), _int(-1), _int(1)),
        contract=Returns(math.pi / 2),
        verified_by="2·arctan 1 = π/2.",
    ),
    Case(
        id="int_control_integrable_endpoint_singularity",
        subsystem="integration_definite",
        statement="∫_0^1 x^{-1/2} dx = 2 — singular at the endpoint but convergent",
        op=definite(1 / ak.sqrt(X), _int(0), _int(1)),
        contract=Returns(2.0),
        verified_by="2√x |_0^1 = 2. Guards the interior-pole check against over-refusal: "
        "a singularity is not the same thing as a divergence.",
    ),
    Case(
        id="int_control_log_endpoint",
        subsystem="integration_definite",
        statement="∫_0^1 log x dx = -1 — log diverges at 0 but the integral converges",
        op=definite(ak.log(X), _int(0), _int(1)),
        contract=Returns(-1.0),
        verified_by="x log x - x |_0^1 = -1, using x log x → 0.",
    ),
    Case(
        id="int_control_pole_outside_interval",
        subsystem="integration_definite",
        statement="∫_2^3 dx/(x-1) = log 2 — the pole at x=1 is outside [2,3]",
        op=definite(1 / (X - 1), _int(2), _int(3)),
        contract=Returns(_LN2),
        verified_by="log 2 - log 1 = log 2.",
    ),
    Case(
        id="int_control_partial_fractions",
        subsystem="integration_definite",
        statement="∫_1^2 dx/(x²+x) = log(4/3)",
        op=definite(1 / (X**2 + X), _int(1), _int(2)),
        contract=Returns(math.log(4 / 3)),
        verified_by="1/(x²+x) = 1/x - 1/(x+1); [log(x/(x+1))]_1^2 = log(2/3) - log(1/2).",
    ),
    Case(
        id="int_control_convergent_tail",
        subsystem="integration_definite",
        statement="∫_1^∞ x^-2 dx = 1 — the convergent side of the p-test",
        op=definite(1 / X**2, _int(1), POOL.pos_infinity()),
        contract=Returns(1.0),
        verified_by="1 - 1/R → 1.",
    ),
    Case(
        id="int_control_exponential_tail",
        subsystem="integration_definite",
        statement="∫_0^∞ e^{-x} dx = 1",
        op=definite(ak.exp(-X), _int(0), POOL.pos_infinity()),
        contract=Returns(1.0),
        verified_by="1 - e^{-R} → 1.",
    ),
    # ── 3.8 round two ───────────────────────────────────────────────────────
    #
    # Every guard in `integrate_definite` binds only the integration variable,
    # so one free *parameter* in the integrand switched all of them off and the
    # FTC difference was returned as if it held for every parameter value.
    Case(
        id="int_pole_interior_with_symbolic_parameter",
        subsystem="integration_definite",
        statement=(
            "∫_{-1}^{1} (x-a)^-2 dx diverges for every a in (-1,1); at a=0 it is the archetype"
        ),
        op=parametric_definite((X - _A_PARAM) ** _int(-2), _int(-1), _int(1), 0.0),
        contract=Raises("E-INT-001"),
        verified_by=(
            "(x-a)^-2 >= 0 wherever it is defined, and for |a| < 1 the double pole at x=a is "
            "strictly inside, so the integral is +inf. The FTC difference -1/(1-a) - 1/(1+a) is "
            "negative there; at a=0 it is exactly the -2 that README.md names as the archetype. "
            "A negative value for a non-negative integrand needs no oracle."
        ),
    ),
    Case(
        id="int_control_parametric_no_pole",
        subsystem="integration_definite",
        statement="∫_0^1 a·x² dx = a/3, a parametric integral with no pole anywhere",
        op=parametric_definite(_A_PARAM * X ** _int(2), _int(0), _int(1), 3.0),
        contract=Returns(1.0),
        verified_by=(
            "∫_0^1 x² dx = 1/3 by the power rule, so the answer is a/3 = 1 at a = 3. The control "
            "for int_pole_interior_with_symbolic_parameter: the parametric guard must refuse "
            "poles, not parameters."
        ),
    ),
    Case(
        id="int_tan_squared_across_pole",
        subsystem="integration_definite",
        statement="∫_0^2 tan²x dx diverges (double pole at π/2 ≈ 1.5708, strictly interior)",
        op=definite(ak.tan(X) ** _int(2), _int(0), _int(2)),
        contract=Raises("E-INT-001"),
        verified_by=(
            "tan²x >= 0 everywhere it is defined and π/2 < 2, so the integral is +inf. The FTC "
            "difference tan(2) - 2 = -4.185 is negative. Internally decisive too: tan² = sec² - 1, "
            "and ∫_0^2 sec²x dx was already refused, so the two answers cannot both stand."
        ),
    ),
    Case(
        id="int_tan_squared_grid_lands_on_pole",
        subsystem="integration_definite",
        statement="∫_0^π tan²x dx diverges — and here the sampling grid falls on the pole itself",
        op=definite(ak.tan(X) ** _int(2), POOL.float(0.0, 53), POOL.float(math.pi, 53)),
        contract=Raises("E-INT-001"),
        verified_by=(
            "tan²x >= 0 and π/2 is interior, so the integral is +inf; alkahest returned -π. A "
            "separate cause from int_tan_squared_across_pole: on [0, π] coarse sample 128 of 257 "
            "falls within 1e-5 of π/2, so the blow-up had already happened before refinement and "
            "a growth test measured against the coarse *maximum* could not fire."
        ),
    ),
    Case(
        id="int_control_bounded_trig_over_period",
        subsystem="integration_definite",
        statement="∫_0^π cos²x dx = π/2 — a bounded trig integrand over the same interval",
        op=definite(ak.cos(X) ** _int(2), POOL.float(0.0, 53), POOL.float(math.pi, 53)),
        contract=Returns(math.pi / 2, tol=1e-12),
        verified_by=(
            "cos²x = (1 + cos 2x)/2, and ∫_0^π cos 2x dx = 0, so the value is π/2. The control for "
            "the two tan cases: the pole scan must not start refusing every trig integrand on "
            "[0, π] just because one of them has a pole there."
        ),
    ),
    Case(
        id="int_weierstrass_jump_across_pi",
        subsystem="integration_definite",
        statement=(
            "∫_0^{3.2} dx/(cos x - 3)² = 0.4202: bounded integrand, but the half-angle "
            "antiderivative jumps at π"
        ),
        op=definite((ak.cos(X) - _int(3)) ** _int(-2), POOL.float(0.0, 53), POOL.float(3.2, 53)),
        contract=RefusesOr(0.42017177259447200),
        verified_by=(
            "1/(cos x - 3)² is continuous with values in [1/16, 1/4] on [0, 3.2], so the integral "
            "lies in [0.2, 0.8] — a negative answer is impossible. Value from mpmath.quad at "
            "dps=30, anchored by the closed form ∫_0^π dx/(3-cos x)² = 3π/8^{3/2} = "
            "0.4165202754523468, "
            "which the same quadrature reproduces to 20 digits. alkahest returned -0.41287, the "
            "Weierstrass-substitution error: tan(x/2) blows up at x = π, inside the interval."
        ),
    ),
    Case(
        id="int_control_weierstrass_below_pi",
        subsystem="integration_definite",
        statement="∫_0^3 dx/(cos x - 3)² = 0.40766 — same integrand, interval stops short of π",
        op=definite((ak.cos(X) - _int(3)) ** _int(-2), POOL.float(0.0, 53), POOL.float(3.0, 53)),
        contract=Returns(0.40765593108334156, tol=1e-9),
        verified_by=(
            "mpmath.quad at dps=30, anchored by ∫_0^π dx/(3-cos x)² = 3π/8^{3/2}: the [0,3] value "
            "must be slightly below it and the [0,3.2] value slightly above, since the integrand "
            "is positive. The control for int_weierstrass_jump_across_pi — the jump guard must "
            "refuse intervals that cross π, not the whole (a + b·cos x) family."
        ),
    ),
    # ── the antiderivative's *domain* over the interval ─────────────────────
    #
    # The sibling of the two Weierstrass cases above.  There the antiderivative
    # is defined on the interval and *jumps*; here it is not defined on the
    # interval at all, because the branch alkahest emitted is real only
    # elsewhere.  Both make `F(b) − F(a)` not the integral, but only the first
    # is visible to a scan that needs `F` at both ends of a cell to form a
    # ratio — a hole makes every cell undecidable and the scan silent.  Each of
    # these was answered `Solved`, with a value containing a `log` of a
    # negative number or an `asin` outside [−1, 1].
    Case(
        id="int_domain_hole_atanh_symmetric",
        subsystem="integration_definite",
        statement="∫_{-1/2}^{1/2} atanh(x) dx = 0, and its antiderivative is not real there",
        op=definite(ak.atanh(X), _rat(-1, 2), _rat(1, 2)),
        contract=RefusesOr(0.0),
        verified_by=(
            "atanh is odd and continuous on (-1, 1), so the integral over a symmetric interval "
            "is 0 by antisymmetry. alkahest's antiderivative is x·atanh(x) + ½·log(x² - 1), "
            "whose logarithm is log of a negative number for every |x| < 1 — i.e. on the whole "
            "interval, and exactly where the integrand is defined. The real branch is "
            "x·atanh(x) + ½·log(1 - x²); recovering the value needs that, not a wider search."
        ),
    ),
    Case(
        id="int_domain_hole_quartic_below_its_poles",
        subsystem="integration_definite",
        statement="∫_{-3}^{-2} dx/(x⁴-1) = 0.030418: bounded integrand, non-real antiderivative",
        op=definite(1 / (X**4 - _int(1)), _int(-3), _int(-2)),
        contract=RefusesOr(0.030417749724959134),
        verified_by=(
            "1/(x⁴-1) is continuous on [-3, -2] (its poles are at ±1), so the integral is an "
            "ordinary number; value from ¼·log|(x-1)/(x+1)| - ½·atan(x) evaluated at the two "
            "endpoints. alkahest emits -¼·log(x+1) + ¼·log(x-1) - ½·atan(x), and both "
            "logarithms are of negative numbers below -1. The endpoint gate cannot catch it: "
            "it asks eval_f64, which does not implement atan and so reports 'cannot decide'."
        ),
    ),
    Case(
        id="int_domain_hole_cubic_below_its_pole",
        subsystem="integration_definite",
        statement="∫_{-4}^{-2} dx/(1+x³) = -0.100340: bounded integrand, non-real antiderivative",
        op=definite(1 / (_int(1) + X**3), _int(-4), _int(-2)),
        contract=RefusesOr(-0.10034029061616050),
        verified_by=(
            "1/(1+x³) has its only real pole at x = -1, outside [-4, -2], so the integrand is "
            "continuous and bounded there; value from mpmath.quad at dps=30, anchored by the "
            "real closed form ⅓·log|x+1| - ⅙·log(x²-x+1) + atan((2x-1)/√3)/√3. alkahest emits "
            "the same formula with log(x+1) rather than log|x+1|, which is not real for x < -1."
        ),
    ),
    Case(
        id="int_control_improper_but_convergent_endpoint",
        subsystem="integration_definite",
        statement="∫_0^1 x^{-1/2} dx = 2 — the integrand blows up at an endpoint and it converges",
        op=definite(_int(1) / ak.sqrt(X), _int(0), _int(1)),
        contract=Returns(2.0, tol=1e-12),
        verified_by=(
            "∫x^{-1/2} = 2√x, continuous on [0, 1]; the improper integral converges to 2. The "
            "control for the three domain-hole cases: a *genuine* improper integral has a "
            "non-finite integrand and a perfectly good antiderivative, and must not be swept "
            "into the same refusal."
        ),
    ),
    Case(
        id="int_control_same_antiderivative_where_it_is_real",
        subsystem="integration_definite",
        statement="∫_2^3 dx/(x²-1) = ½·log(3/2) — the refused formula, on an interval it holds on",
        op=definite(1 / (X**2 - _int(1)), _int(2), _int(3)),
        contract=Returns(0.5 * math.log(1.5), tol=1e-12),
        verified_by=(
            "½·log((x-1)/(x+1)) is an antiderivative; at 3 it is ½·log(1/2), at 2 it is "
            "½·log(1/3), so the value is ½·log(3/2), and the poles at ±1 are outside [2, 3]. "
            "This is the same ½·log(x-1) - ½·log(x+1) that int_domain_hole_* refuses below -1, "
            "evaluated where both logarithms are real: the rule has to be about the interval, "
            "not about the shape of the formula."
        ),
    ),
    # Rust panics crossing the FFI boundary.
    #
    # Not silent errors — but `pyo3_runtime.PanicException` inherits
    # `BaseException`, so an unattended loop's `except Exception` does not catch
    # it and the run dies on an input it was supposed to survive.  Scored
    # `no_answer`: neither an answer nor a refusal.
    # -----------------------------------------------------------------------
    Case(
        id="integrate_radical_of_log_of_zero",
        subsystem="integration_definite",
        statement="∫_{-1}^{1} √(log(x-x)) dx has no value — log 0 is undefined",
        op=_survives_a_panic(definite(ak.sqrt(ak.log(X - X)), POOL.float(-1.0), POOL.float(1.0))),
        contract=RefusesOr(),
        verified_by=(
            "x - x = 0 and log 0 is undefined, so the integrand has no value at any point and "
            "the integral does not exist. Any finite answer is a lie about a function that does "
            "not exist."
        ),
        note=(
            "Pre-fix this was a Rust panic (RatFn: zero denominator) arriving as "
            "pyo3_runtime.PanicException, a BaseException that `except Exception` does not "
            "catch. The op wraps it so the gate reports the failure instead of dying."
        ),
    ),
]
