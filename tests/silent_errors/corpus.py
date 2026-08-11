"""The declarative trap corpus.

Every case here is a *classic* silent-error shape — a place where a CAS has a
clean, plausible, wrong answer available and has to choose not to give it.  The
expected value of every case was derived by hand from the definition (and, for
the classical constants, cross-checked against ``math``), never read off
alkahest's own output; :attr:`Case.verified_by` records which.

Adding a case: see ``tests/silent_errors/README.md``.

Cost discipline: this corpus runs on every pull request, so each op must finish
in well under a second.  Known-slow inputs are documented in the README rather
than being added here — a gate that times out is a gate that gets disabled.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import alkahest as ak
import alkahest.number_theory as nt

# ``tests/`` is on sys.path via the root conftest, so the textbook gate's series
# helper (which strips the trailing O(...) term that eval_expr cannot evaluate)
# is importable and worth reusing rather than duplicating.
from _tg_helpers import eval_series_truncated
from contracts import Case, Measured, Raises, RefusesOr, Returns

# ---------------------------------------------------------------------------
# Shared pool.  Expressions are immutable and pool-scoped; one pool for the
# whole corpus keeps case construction cheap and interning consistent.
# ---------------------------------------------------------------------------

POOL = ak.ExprPool()
X = POOL.symbol("x")
N = POOL.symbol("n")
K = POOL.symbol("k")


def _int(v: int) -> ak.Expr:
    return POOL.integer(v)


def _rat(a: int, b: int) -> ak.Expr:
    return POOL.rational(a, b)


def _num(value: Any) -> float:
    """Reduce an Expr / DerivedResult / number to a float."""
    if isinstance(value, ak.DerivedResult):
        value = value.value
    if isinstance(value, (int, float)):
        return float(value)
    return float(ak.eval_expr(value, {}))


# ---------------------------------------------------------------------------
# Answer helpers — each returns the plain "answer" a case is scored on.
# ---------------------------------------------------------------------------


def definite(integrand: ak.Expr, lo: ak.Expr, hi: ak.Expr) -> Callable[[], Measured]:
    """Answer = the numeric value of ∫_lo^hi integrand dx."""

    def op() -> Measured:
        r = ak.integrate(integrand, X, lo, hi)
        return Measured(_num(r.value), r.verification)

    return op


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


def series_at(expr: ak.Expr, about: ak.Expr, order: int, sample: float) -> Callable[[], float]:
    """Answer = alkahest's truncated series for *expr*, evaluated at *sample*."""

    def op() -> float:
        s = ak.series(expr, X, about, order)
        return float(eval_series_truncated(s, X, sample))

    return op


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


def real_solution_count(equations: list[ak.Expr], unknowns: list[ak.Expr]) -> Callable[[], int]:
    """Answer = how many real solutions ``solve(..., domain="real")`` reports."""

    def op() -> int:
        sols = ak.solve(equations, unknowns, domain="real")
        return len(sols)

    return op


def universal_holds(poly: ak.Expr, kind: str) -> Callable[[], bool]:
    """Answer = ``decide``'s verdict on ``forall x. poly <kind> 0``."""

    def op() -> bool:
        rel = {"ge": POOL.ge, "le": POOL.le, "gt": POOL.gt, "lt": POOL.lt}[kind]
        truth, _witness = ak.decide(ak.Forall(X, rel(poly, _int(0))))
        return truth

    return op


def _matrix(rows: list[list[int]]) -> ak.Matrix:
    return ak.Matrix([[_int(v) for v in row] for row in rows])


SINGULAR_2X2 = [[1, 2], [2, 4]]
SINGULAR_3X3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
ZERO_2X2 = [[0, 0], [0, 0]]
NON_SQUARE = [[1, 2, 3], [4, 5, 6]]
#: det = (2^30+1)(2^30-1) - 2^60 = -1 exactly; float64 evaluation cancels to 0.0.
CANCELLING_2X2 = [[2**30 + 1, 2**30], [2**30, 2**30 - 1]]

# --- Transcendental rank traps -------------------------------------------
# ``exp(a)²`` and ``exp(2a)`` are the same function written two ways.  Any
# elimination that cannot see that will "clear" a column it has not cleared.
_A = POOL.symbol("a")
_EXP_A = ak.exp(_A)

#: Row 2 is exactly ``exp(a)`` × row 1, so the rank is 1 — but only if
#: ``exp(a)·exp(a) − exp(a+a)`` is recognised as zero.
EXP_DEPENDENT_ROWS = ak.Matrix(
    [
        [_int(1), _EXP_A, _EXP_A],
        [_EXP_A, _EXP_A * _EXP_A, ak.exp(_A + _A)],
    ]
)

#: The control: identical except for the last entry, which breaks the
#: proportionality, so the rank really is 2.
EXP_INDEPENDENT_ROWS = ak.Matrix(
    [
        [_int(1), _EXP_A, _EXP_A],
        [_EXP_A, _EXP_A * _EXP_A, _EXP_A],
    ]
)

#: ``mystery`` has no differentiation rule, no numeric kernel and no interval
#: kernel, so ``mystery(a)`` can be neither normalised to zero nor rigorously
#: enclosed away from it.  Whether it is the zero function is not knowable here,
#: and column 1 has no other candidate.
UNDECIDABLE_PIVOT = ak.Matrix(
    [
        [POOL.func("mystery", [_A]), _int(0)],
        [_int(0), _int(0)],
    ]
)


def _rref_zero_rows(m: ak.Matrix, at: float = 0.7) -> Callable[[], int]:
    """Answer = how many rows of ``m.rref()`` vanish, sampled at ``a = at``.

    Scored numerically rather than structurally so the case is immune to the
    form the entries come back in; what it pins down is the only thing that
    matters — a row of an rref is either identically zero or it is not.  For a
    rank-deficient matrix the missing zero row reappears as a spurious pivot,
    which for an augmented system reads as ``0 = 1``: the textbook signature of
    an inconsistent system, and a false "no solution" verdict for a search loop.
    """

    def op() -> int:
        count = 0
        for row in m.rref().to_list():
            if all(abs(float(ak.eval_expr(entry, {_A: at}))) < 1e-12 for entry in row):
                count += 1
        return count

    return op


# ---------------------------------------------------------------------------
# Reference values (hand-derived; `math` used only to evaluate the closed form)
# ---------------------------------------------------------------------------

_E = math.e
_LN2 = math.log(2.0)


def _exp_log_sum(x: float) -> float:
    """d/dx [e^x·log x] = e^x·log x + e^x/x."""
    return math.exp(x) * math.log(x) + math.exp(x) / x


def _risch_gaussian_pair(x: float) -> float:
    """d/dx [x·e^{x²}] = e^{x²} + 2x²·e^{x²}."""
    return math.exp(x * x) + 2 * x * x * math.exp(x * x)


def _sin_log_pair(x: float) -> float:
    """d/dx [sin x·log x] = cos x·log x + sin x / x."""
    return math.cos(x) * math.log(x) + math.sin(x) / x


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

HAND = "hand derivation from the definition"
CALCULUS = "first-course calculus fact, re-derived by hand"


CASES: list[Case] = [
    # ── real quantifier elimination ──────────────────────────────────────────
    #
    # `decide` is the engine behind every stability proof and bound check, so a
    # false `True` here is a machine-checked-looking proof of a false theorem —
    # the most damaging silent error in the library.
    Case(
        id="decide_forall_touching_zero_strict",
        subsystem="real_qe",
        statement="forall x. x^2 > 0 is FALSE (x = 0)",
        op=universal_holds(X ** _int(2), "gt"),
        contract=Returns(False),
        verified_by="x=0 gives 0 > 0, which is false. Fixed: Le/Ge boundary sampling.",
    ),
    Case(
        id="decide_forall_quartic_touching_zero",
        subsystem="real_qe",
        statement="forall x. x^4 > 0 is FALSE (x = 0)",
        op=universal_holds(X ** _int(4), "gt"),
        contract=Returns(False),
        verified_by="x=0 gives 0 > 0, false.",
    ),
    Case(
        id="decide_forall_shifted_square_strict",
        subsystem="real_qe",
        statement="forall x. (x-1)^2 > 0 is FALSE (x = 1)",
        op=universal_holds((X - _int(1)) ** _int(2), "gt"),
        contract=Returns(False),
        verified_by="x=1 gives 0 > 0, false.",
    ),
    Case(
        id="decide_forall_nonneg_square",
        subsystem="real_qe",
        statement="forall x. x^2 >= 0 is TRUE",
        op=universal_holds(X ** _int(2), "ge"),
        contract=Returns(True),
        verified_by="Squares are non-negative. Guards against over-refusing the fix.",
    ),
    Case(
        id="decide_forall_narrow_negative_cell",
        subsystem="real_qe",
        statement="forall x. 2x^4 + x^3 - 4x^2 + 3 >= 0 is FALSE (x = -6/5 gives -213/625)",
        op=universal_holds(
            _int(2) * X ** _int(4) + X ** _int(3) - _int(4) * X ** _int(2) + _int(3), "ge"
        ),
        contract=Returns(False),
        verified_by=(
            "Exact rational evaluation at x=-6/5: 2(1296/625) + (-216/125) - 4(36/25) + 3 "
            "= -213/625 < 0."
        ),
    ),
    Case(
        id="decide_forall_narrow_positive_cell",
        subsystem="real_qe",
        statement="forall x. -4x^4 - 4x^3 + 3x^2 - 3 <= 0 is FALSE (x = 4 gives -1235... )",
        op=universal_holds(
            -_int(4) * X ** _int(4) - _int(4) * X ** _int(3) + _int(3) * X ** _int(2) - _int(3),
            "le",
        ),
        contract=Returns(False),
        verified_by="Exact evaluation finds a point where the polynomial is positive.",
    ),
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
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
    Case(
        id="nonelementary_gaussian",
        subsystem="integration_nonelementary",
        statement="∫ e^{-x²} dx has no elementary antiderivative",
        op=lambda: _num(ak.integrate(ak.exp(-(X**2)), X).value),
        contract=Raises("E-INT-004"),
        verified_by="(√π/2)·erf(x); erf is not elementary (Liouville).",
    ),
    Case(
        id="nonelementary_sinc",
        subsystem="integration_nonelementary",
        statement="∫ sin(x)/x dx has no elementary antiderivative",
        op=lambda: _num(ak.integrate(ak.sin(X) / X, X).value),
        contract=Raises("E-INT-004"),
        verified_by="Si(x), the sine integral — a special function, not elementary.",
    ),
    Case(
        id="nonelementary_logarithmic_integral",
        subsystem="integration_nonelementary",
        statement="∫ dx/log x has no elementary antiderivative",
        op=lambda: _num(ak.integrate(1 / ak.log(X), X).value),
        contract=Raises("E-INT-004"),
        verified_by="li(x), the logarithmic integral.",
    ),
    Case(
        id="nonelementary_exponential_integral",
        subsystem="integration_nonelementary",
        statement="∫ e^x/x dx has no elementary antiderivative",
        op=lambda: _num(ak.integrate(ak.exp(X) / X, X).value),
        contract=Raises("E-INT-004"),
        verified_by="Ei(x), the exponential integral.",
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
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # Solving: complex roots handed back where a real solution was requested.
    # -----------------------------------------------------------------------
    Case(
        id="solve_x_squared_plus_one_real",
        subsystem="solving",
        statement="x² = -1 has no real solutions",
        op=real_solution_count([X**2 + 1], [X]),
        contract=Returns(0),
        verified_by="x² ≥ 0 for all real x. ±i are not real solutions.",
        benchmark_tasks=("solve_x2_plus_1_real",),
    ),
    Case(
        id="solve_irreducible_quadratic_real",
        subsystem="solving",
        statement="x²+x+1 = 0 has no real solutions (discriminant -3)",
        op=real_solution_count([X**2 + X + 1], [X]),
        contract=Returns(0),
        verified_by="b²-4ac = 1-4 = -3 < 0.",
    ),
    Case(
        id="solve_quartic_plus_one_real",
        subsystem="solving",
        statement="x⁴+1 = 0 has no real solutions",
        op=real_solution_count([X**4 + 1], [X]),
        contract=RefusesOr(0),
        verified_by="x⁴ ≥ 0, so x⁴+1 ≥ 1 > 0.",
        note="alkahest refuses with E-SOLVE-002 (degree > 2 back-substitution). Refusing is "
        "safe; handing back the four complex 8th roots of unity would not be.",
    ),
    Case(
        id="solve_real_roots_of_x_squared_plus_one",
        subsystem="solving",
        statement="real_roots(x²+1) is empty",
        op=lambda: len(ak.real_roots(X**2 + 1, X)),
        contract=Returns(0),
        verified_by="No real root; Sturm's theorem gives a count of 0.",
    ),
    Case(
        id="solve_real_roots_of_quartic_plus_one",
        subsystem="solving",
        statement="real_roots(x⁴+1) is empty",
        op=lambda: len(ak.real_roots(X**4 + 1, X)),
        contract=Returns(0),
        verified_by="x⁴+1 ≥ 1 > 0 on ℝ.",
    ),
    Case(
        id="solve_real_roots_of_cubic_unity",
        subsystem="solving",
        statement="x³-1 has exactly one real root (the other two are complex)",
        op=lambda: len(ak.real_roots(X**3 - 1, X)),
        contract=Returns(1),
        verified_by="x³-1 = (x-1)(x²+x+1); the quadratic factor has discriminant -3.",
    ),
    Case(
        id="solve_real_roots_of_double_root",
        subsystem="solving",
        statement="(x-1)² has one distinct real root",
        op=lambda: len(ak.real_roots((X - 1) ** 2, X)),
        contract=Returns(1),
        verified_by="Only x=1, with multiplicity 2. real_roots reports isolating intervals, "
        "i.e. distinct roots.",
    ),
    Case(
        id="solve_sqrt_equals_negative",
        subsystem="solving",
        statement="√x = -1 has no real solution; squaring introduces the extraneous root x=1",
        op=real_solution_count([ak.sqrt(X) + 1], [X]),
        contract=RefusesOr(0),
        verified_by="The principal square root is non-negative. Squaring both sides is not an "
        "equivalence, and yields the extraneous x=1.",
        benchmark_tasks=("sqrt_eq_negative",),
        note="alkahest refuses with E-SOLVE-001 (not a polynomial) — safe, and it never reports "
        "the extraneous root.",
    ),
    Case(
        id="solve_real_domain_does_not_overfilter",
        subsystem="solving",
        statement="x² = 1 genuinely has two real solutions; domain='real' must not drop them",
        op=real_solution_count([X**2 - 1], [X]),
        contract=Returns(2),
        verified_by="x = ±1. The control for solve_x_squared_plus_one_real: a solver that "
        "returns [] for everything would otherwise pass that case.",
    ),
    Case(
        id="solve_real_roots_residual",
        subsystem="solving",
        statement="each solution of x²=2 satisfies the equation (residual ≈ 0)",
        op=lambda: max(
            abs(float(ak.eval_expr(X**2 - 2, {X: _num(sol[X])})))
            for sol in ak.solve([X**2 - 2], [X], domain="real")
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by="Substituting a returned root back into the equation must give 0; this is "
        "form-independent and catches a solver that returns confident non-roots.",
    ),
    Case(
        id="solve_zero_polynomial",
        subsystem="solving",
        statement="the zero polynomial has infinitely many roots — no finite root list is honest",
        op=lambda: len(ak.real_roots(_int(0), X)),
        contract=Raises("E-ROOT-002"),
        verified_by="Every real number is a root of 0.",
    ),
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
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
        note="Weak refusal: alkahest returns a Series whose coefficients contain 0^-1, so it "
        "cannot be evaluated. It does not raise, so a caller who never evaluates the "
        "coefficients gets no signal.",
    ),
    Case(
        id="series_log_at_origin",
        subsystem="series",
        statement="log x has no Laurent expansion at x=0 (logarithmic, not polar, singularity)",
        op=series_at(ak.log(X), _int(0), 3, 0.1),
        contract=RefusesOr(),
        verified_by="log x is unbounded at 0 but x^n·log x → 0 for every n>0, so no finite "
        "principal part exists. No finite answer is acceptable.",
        note="Weak refusal: the returned Series contains log(0) and 0^-1 coefficients.",
    ),
    Case(
        id="series_sqrt_at_branch_point",
        subsystem="series",
        statement="√x has no Laurent expansion at x=0 (branch point, half-integer exponent)",
        op=series_at(ak.sqrt(X), _int(0), 3, 0.1),
        contract=RefusesOr(),
        verified_by="√x is not meromorphic at 0; a Puiseux series is required.",
        note="Weak refusal: coefficients contain sqrt(0)^-1.",
    ),
    Case(
        id="series_essential_singularity",
        subsystem="series",
        statement="e^{1/x} has an essential singularity at 0 — no finite truncation is meaningful",
        op=series_at(ak.exp(1 / X), _int(0), 3, 0.1),
        contract=RefusesOr(),
        verified_by="The Laurent series Σ x^-n/n! has infinitely many negative powers "
        "(Casorati–Weierstrass); no truncation at positive order represents it.",
        note="Weak refusal: coefficients contain exp(0^-1).",
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
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # Linear algebra on singular and ill-conditioned inputs.
    # -----------------------------------------------------------------------
    Case(
        id="matrix_inverse_singular_2x2",
        subsystem="linear_algebra",
        statement="[[1,2],[2,4]] is singular and has no inverse",
        op=lambda: _matrix(SINGULAR_2X2).inverse().to_list(),
        contract=Raises("E-MAT-003"),
        verified_by="det = 1·4 - 2·2 = 0; row 2 is 2× row 1.",
    ),
    Case(
        id="matrix_inverse_zero_2x2",
        subsystem="linear_algebra",
        statement="the zero matrix has no inverse",
        op=lambda: _matrix(ZERO_2X2).inverse().to_list(),
        contract=Raises("E-MAT-003"),
        verified_by="det = 0·0 - 0·0 = 0; the zero matrix has rank 0.",
    ),
    Case(
        id="matrix_inverse_singular_3x3",
        subsystem="linear_algebra",
        statement="[[1,2,3],[4,5,6],[7,8,9]] is singular and has no inverse",
        op=lambda: _matrix(SINGULAR_3X3).inverse().to_list(),
        contract=Raises("E-MAT-003"),
        verified_by="row3 - row2 = row2 - row1 = (3,3,3), so the rows are linearly dependent "
        "and det = 0. Rank 2, not 3.",
    ),
    Case(
        id="matrix_determinant_non_square",
        subsystem="linear_algebra",
        statement="the determinant of a 2×3 matrix is undefined",
        op=lambda: _num(_matrix(NON_SQUARE).det()),
        contract=Raises("E-MAT-002"),
        verified_by="Determinants are defined only for square matrices.",
    ),
    Case(
        id="matrix_inverse_non_square",
        subsystem="linear_algebra",
        statement="a 2×3 matrix has no inverse",
        op=lambda: _matrix(NON_SQUARE).inverse().to_list(),
        contract=Raises("E-MAT-002"),
        verified_by="Inverses are defined only for square matrices.",
    ),
    Case(
        id="matrix_determinant_singular_is_zero",
        subsystem="linear_algebra",
        statement="det[[1,2],[2,4]] = 0 exactly",
        op=lambda: _num(_matrix(SINGULAR_2X2).det()),
        contract=Returns(0.0),
        verified_by="1·4 - 2·2 = 0.",
    ),
    Case(
        id="matrix_determinant_singular_3x3_is_zero",
        subsystem="linear_algebra",
        statement="det[[1,2,3],[4,5,6],[7,8,9]] = 0 exactly",
        op=lambda: _num(_matrix(SINGULAR_3X3).det()),
        contract=Returns(0.0),
        verified_by="Cofactor expansion: 1(45-48) - 2(36-42) + 3(32-35) = -3 + 12 - 9 = 0.",
    ),
    Case(
        id="matrix_rank_singular",
        subsystem="linear_algebra",
        statement="rank[[1,2],[2,4]] = 1",
        op=lambda: _matrix(SINGULAR_2X2).rank(),
        contract=Returns(1),
        verified_by="One independent row.",
    ),
    Case(
        id="matrix_rank_exp_proportional_rows",
        subsystem="linear_algebra",
        statement="rank[[1, e^a, e^a], [e^a, e^a·e^a, e^(a+a)]] = 1",
        op=lambda: EXP_DEPENDENT_ROWS.rank(),
        contract=Returns(1),
        verified_by="Row 2 = e^a · row 1 entry by entry: e^a·1 = e^a, e^a·e^a is the (2,2) "
        "entry verbatim, and e^a·e^a = e^(a+a) by the exponential functional equation "
        "e^u·e^v = e^(u+v), which is the (2,3) entry. Two proportional rows span a "
        "1-dimensional row space, so the rank is 1 for every value of a.",
    ),
    Case(
        id="matrix_rref_exp_proportional_rows_has_zero_row",
        subsystem="linear_algebra",
        statement="the rref of [[1, e^a, e^a], [e^a, e^a·e^a, e^(a+a)]] has exactly one zero row",
        op=_rref_zero_rows(EXP_DEPENDENT_ROWS),
        contract=Returns(1),
        verified_by="A 2×3 matrix of rank 1 has 2 − 1 = 1 zero row in reduced row echelon "
        "form. The wrong answer here is 0 zero rows, i.e. a second pivot in the last "
        "column — read as an augmented system that is the row 0 = 1 of an inconsistent "
        "one, for a system that is consistent.",
    ),
    Case(
        id="matrix_rank_undecidable_pivot_refuses",
        subsystem="linear_algebra",
        statement="rank[[mystery(a), 0], [0, 0]] cannot be stated — mystery(a) may be the "
        "zero function",
        op=lambda: UNDECIDABLE_PIVOT.rank(),
        contract=RefusesOr(),
        verified_by="The rank is 1 if mystery is not identically zero and 0 if it is. "
        "mystery is an uninterpreted function symbol, so both readings are consistent with "
        "everything alkahest knows and neither number is derivable. Deciding whether an "
        "expression over a transcendental extension vanishes is undecidable in general "
        "(Richardson 1968), so this class cannot be normalised away — the only honest "
        "answer is a refusal. Contract is code-agnostic on purpose: what is being pinned "
        "is that no rank is asserted, not which E-LINALG code says so.",
        note="This is the pair to matrix_rank_exp_proportional_rows: that case is 'prove "
        "zero when it is zero', this one is 'do not claim non-zero when you cannot'. A "
        "library that only did the first would pass that case by pivoting on anything it "
        "failed to reduce, which is the bug that motivated both.",
    ),
    Case(
        id="matrix_rank_exp_independent_rows",
        subsystem="linear_algebra",
        statement="rank[[1, e^a, e^a], [e^a, e^a·e^a, e^a]] = 2",
        op=lambda: EXP_INDEPENDENT_ROWS.rank(),
        contract=Returns(2),
        verified_by="The control for matrix_rank_exp_proportional_rows: only the last entry "
        "differs. Row 2 − e^a · row 1 = (0, 0, e^a − e^a·e^a) = (0, 0, e^a(1 − e^a)), which "
        "is not the zero function (it is e·(1−e) ≠ 0 at a = 1), so the rows are independent "
        "and the rank is 2. A gate made only of 'prove this is zero' cases is passed by a "
        "library that calls everything zero.",
    ),
    Case(
        id="matrix_determinant_catastrophic_cancellation",
        subsystem="linear_algebra",
        statement="det[[2³⁰+1, 2³⁰],[2³⁰, 2³⁰-1]] = -1, not 0",
        op=lambda: _num(_matrix(CANCELLING_2X2).det()),
        contract=Returns(-1.0),
        verified_by="(2³⁰+1)(2³⁰-1) - 2³⁰·2³⁰ = (2⁶⁰ - 1) - 2⁶⁰ = -1 exactly. In float64 both "
        "products round to 2⁶⁰ and the difference cancels to 0.0 — a plausible, wrong, "
        "and *sign-flipping* answer (singular vs invertible).",
    ),
    Case(
        id="matrix_inverse_roundtrip",
        subsystem="linear_algebra",
        statement="inverse[[1,2],[3,4]] = [[-2,1],[3/2,-1/2]]",
        op=lambda: [
            [_num(entry) for entry in row] for row in _matrix([[1, 2], [3, 4]]).inverse().to_list()
        ],
        contract=Returns([[-2.0, 1.0], [1.5, -0.5]]),
        verified_by="1/det · adj = (1/-2)·[[4,-2],[-3,1]] = [[-2,1],[1.5,-0.5]], by hand. The "
        "control for the singular-inverse refusals.",
    ),
    # -----------------------------------------------------------------------
    # Divergent sums and products.  Every one of these has a famous "value"
    # attached to it by some summation method; none of them converges.
    # -----------------------------------------------------------------------
    Case(
        id="sum_harmonic_divergent",
        subsystem="sums_products",
        statement="Σ_{k≥1} 1/k diverges",
        op=lambda: _num(ak.sum_definite(1 / K, K, _int(1), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="Partial sums exceed log n; the classic Oresme grouping argument.",
    ),
    Case(
        id="sum_geometric_divergent",
        subsystem="sums_products",
        statement="Σ_{k≥1} 2^k diverges",
        op=lambda: _num(ak.sum_definite(_int(2) ** K, K, _int(1), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="|r| = 2 > 1. Blindly applying a/(1-r) gives 2/(1-2) = -2, a clean wrong "
        "number for a sum of positive terms.",
    ),
    Case(
        id="sum_grandi_divergent",
        subsystem="sums_products",
        statement="Σ_{k≥0} (-1)^k diverges (Grandi's series)",
        op=lambda: _num(ak.sum_definite(_int(-1) ** K, K, _int(0), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="Partial sums alternate 1,0,1,0,… and have no limit. The Abel and Cesàro "
        "sums are 1/2 — the canonical plausible wrong answer.",
    ),
    Case(
        id="product_divergent_constant",
        subsystem="sums_products",
        statement="Π_{k≥1} 2 diverges",
        op=lambda: _num(ak.product_definite(_int(2), K, _int(1), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by="The partial products are 2^n, which grow without bound.",
        note="Weak refusal: alkahest returns the symbol 2^∞, which does not reduce to a float.",
    ),
    Case(
        id="sum_control_first_ten",
        subsystem="sums_products",
        statement="Σ_{k=1}^{10} k = 55",
        op=lambda: _num(ak.sum_definite(K, K, _int(1), _int(10)).value),
        contract=Returns(55.0),
        verified_by="10·11/2 = 55.",
    ),
    Case(
        id="sum_control_faulhaber_symbolic",
        subsystem="sums_products",
        statement="Σ_{k=1}^{n} k = n(n+1)/2; at n=7 that is 28",
        op=lambda: float(ak.eval_expr(ak.sum_definite(K, K, _int(1), N).value, {N: 7})),
        contract=Returns(28.0),
        verified_by="7·8/2 = 28. Checks the closed form rather than its printed shape.",
    ),
    Case(
        id="product_control_factorial",
        subsystem="sums_products",
        statement="Π_{k=1}^{5} k = 120",
        op=lambda: _num(ak.product_definite(K, K, _int(1), _int(5)).value),
        contract=Returns(120.0),
        verified_by="1·2·3·4·5 = 120, i.e. 5! computed by hand.",
    ),
    Case(
        id="product_control_contains_zero",
        subsystem="sums_products",
        statement="Π_{k=0}^{5} k = 0 — the k=0 factor annihilates the product",
        op=lambda: _num(ak.product_definite(K, K, _int(0), _int(5)).value),
        contract=Returns(0.0),
        verified_by="0·1·2·3·4·5 = 0. A gamma-quotient closed form that forgets the pole at "
        "k=0 would report 120.",
    ),
    # -----------------------------------------------------------------------
    # Number theory at 0, 1, negatives, and the pseudoprime traps.
    # -----------------------------------------------------------------------
    Case(
        id="nt_isprime_one",
        subsystem="number_theory",
        statement="1 is not prime",
        op=lambda: nt.isprime(1),
        contract=Returns(False),
        verified_by="A prime has exactly two distinct positive divisors; 1 has one. Excluding 1 "
        "is what makes factorisation unique.",
    ),
    Case(
        id="nt_isprime_two",
        subsystem="number_theory",
        statement="2 is prime",
        op=lambda: nt.isprime(2),
        contract=Returns(True),
        verified_by="Divisors 1 and 2. The control for the edge cases: 'always False' must fail.",
    ),
    Case(
        id="nt_isprime_zero",
        subsystem="number_theory",
        statement="0 is not prime",
        op=lambda: nt.isprime(0),
        contract=RefusesOr(False),
        verified_by="0 has infinitely many divisors.",
        note="alkahest refuses with E-NT-002 (expects a positive integer) rather than "
        "answering False; both are safe.",
    ),
    Case(
        id="nt_isprime_negative",
        subsystem="number_theory",
        statement="-7 is not prime under the standard positive-integer definition",
        op=lambda: nt.isprime(-7),
        contract=RefusesOr(False),
        verified_by="Primality is defined on integers > 1. (-7 is a prime *element* of ℤ, which "
        "is a different statement and must not be conflated.)",
        note="alkahest refuses with E-NT-002.",
    ),
    Case(
        id="nt_isprime_carmichael_561",
        subsystem="number_theory",
        statement="561 = 3·11·17 is composite despite passing the Fermat test for every "
        "coprime base",
        op=lambda: nt.isprime(561),
        contract=Returns(False),
        verified_by="561 = 3·11·17; it is the smallest Carmichael number, so a^560 ≡ 1 (mod "
        "561) for every a coprime to 561. A Fermat-test primality check answers True.",
    ),
    Case(
        id="nt_isprime_strong_pseudoprime_2047",
        subsystem="number_theory",
        statement="2047 = 23·89 is composite despite being a strong pseudoprime to base 2",
        op=lambda: nt.isprime(2047),
        contract=Returns(False),
        verified_by="2047 = 2¹¹-1 = 23·89. It is the smallest strong pseudoprime to base 2, so "
        "a single-base Miller–Rabin with a=2 answers True.",
    ),
    Case(
        id="nt_factorint_zero",
        subsystem="number_theory",
        statement="0 has no prime factorisation",
        op=lambda: nt.factorint(0),
        contract=Raises("E-NT-002"),
        verified_by="0 is divisible by every prime to every power; no finite factorisation exists.",
    ),
    Case(
        id="nt_factorint_one",
        subsystem="number_theory",
        statement="1 factors as the empty product",
        op=lambda: nt.factorint(1),
        contract=Returns({}),
        verified_by="1 is the empty product. Reporting {1: 1} would break unique factorisation.",
    ),
    Case(
        id="nt_totient_one",
        subsystem="number_theory",
        statement="φ(1) = 1",
        op=lambda: nt.totient(1),
        contract=Returns(1),
        verified_by="The only k in [1,1] with gcd(k,1)=1 is k=1. Πp|n(1-1/p) over an empty "
        "prime set gives 1, so the formula agrees.",
    ),
    Case(
        id="nt_totient_zero",
        subsystem="number_theory",
        statement="φ(0) is undefined",
        op=lambda: nt.totient(0),
        contract=RefusesOr(),
        verified_by="Euler's totient is defined on positive integers.",
        note="alkahest refuses with E-NT-002.",
    ),
    Case(
        id="nt_totient_negative",
        subsystem="number_theory",
        statement="φ(-5) is undefined",
        op=lambda: nt.totient(-5),
        contract=RefusesOr(),
        verified_by="Euler's totient is defined on positive integers.",
        note="alkahest refuses with E-NT-002.",
    ),
    Case(
        id="nt_jacobi_even_denominator",
        subsystem="number_theory",
        statement="the Jacobi symbol (2/4) is undefined — the denominator must be odd",
        op=lambda: nt.jacobi_symbol(2, 4),
        contract=Raises("E-NT-002"),
        verified_by="(a/n) is defined as a product of Legendre symbols over the odd prime "
        "factorisation of n; even n has no such factorisation.",
    ),
    Case(
        id="nt_nthroot_mod_non_residue",
        subsystem="number_theory",
        statement="3 is a quadratic non-residue mod 7, so √3 mod 7 does not exist",
        op=lambda: nt.nthroot_mod(3, 2, 7),
        contract=Raises("E-NT-003"),
        verified_by="Squares mod 7 are {1,2,4} (1²=1, 2²=4, 3²=2). 3 is not among them.",
    ),
    Case(
        id="nt_discrete_log_no_solution",
        subsystem="number_theory",
        statement="3 is not a power of 2 mod 7, so log_2 3 mod 7 does not exist",
        op=lambda: nt.discrete_log(3, 2, 7),
        contract=Raises("E-NT-003"),
        verified_by="⟨2⟩ = {2,4,1} mod 7 has order 3 and does not contain 3.",
    ),
    Case(
        id="nt_discrete_log_control",
        subsystem="number_theory",
        statement="3² ≡ 2 (mod 7), so log_3 2 mod 7 = 2",
        op=lambda: nt.discrete_log(2, 3, 7),
        contract=Returns(2),
        verified_by="3² = 9 ≡ 2 (mod 7). The control for nt_discrete_log_no_solution.",
    ),
    Case(
        id="nt_nextprime_one",
        subsystem="number_theory",
        statement="the smallest prime greater than 1 is 2",
        op=lambda: nt.nextprime(1),
        contract=Returns(2),
        verified_by="2 is the smallest prime.",
    ),
]


#: Fast lookup by id.
CASES_BY_ID: dict[str, Case] = {c.id: c for c in CASES}

if len(CASES_BY_ID) != len(CASES):  # pragma: no cover - corpus authoring guard
    seen: set[str] = set()
    dupes = sorted({c.id for c in CASES if c.id in seen or seen.add(c.id)})  # type: ignore[func-returns-value]
    raise RuntimeError(f"duplicate case ids in the silent-error corpus: {dupes}")
