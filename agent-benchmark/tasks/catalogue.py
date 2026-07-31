"""The benchmark task catalogue.

Every expected value here was verified empirically against both SymPy 1.14 and
alkahest before being written down, and every task in the TRAP and SCALE
sections was checked to actually discriminate — that is, at least one mainstream
CAS gets it wrong or fails on it.  Tasks that both libraries handle correctly are
kept only in the CONTROL section, whose job is to prove the arms are wired up,
not to separate them.

Nothing here is chosen to flatter alkahest.  Several tasks are ones alkahest
currently fails (`basel_sum`, the interior-pole traps); a benchmark that only
contains problems the home library wins is not evidence, and anyone evaluating
the library will notice immediately.
"""

from __future__ import annotations

import math

from .base import (
    AgentTask,
    Kind,
    enclosure,
    exact_int,
    keyword,
    numeric,
    refusal_or_value,
)

# ---------------------------------------------------------------------------
# Control — every working CAS should pass these.
# ---------------------------------------------------------------------------

CONTROL: list[AgentTask] = [
    AgentTask(
        name="diff_sin_x2",
        kind=Kind.CONTROL,
        category="differentiation",
        difficulty=1,
        problem=(
            "Differentiate sin(x**2) with respect to x, then evaluate the "
            "derivative at x = 1.0.\nPrint the result as a float."
        ),
        verify=numeric(2 * math.cos(1.0), tol=1e-4),
        hint=f"{2 * math.cos(1.0):.10f}",
        rationale="Floor task: basic chain rule.",
    ),
    AgentTask(
        name="integrate_x2_definite",
        kind=Kind.CONTROL,
        category="integration",
        difficulty=1,
        problem=(
            "Compute the definite integral of x**2 from x = 0 to x = 1.\n"
            "Print the result as a float."
        ),
        verify=numeric(1 / 3, tol=1e-4),
        hint="0.3333333333",
        rationale="Floor task: proper definite integral, no singularities.",
    ),
    AgentTask(
        name="trig_identity",
        kind=Kind.CONTROL,
        category="simplification",
        difficulty=1,
        problem=("Simplify sin(x)**2 + cos(x)**2 to a constant.\nPrint the integer result."),
        verify=exact_int(1),
        hint="1",
        rationale="Floor task: the canonical trig identity.",
    ),
    AgentTask(
        name="matrix_det_3x3",
        kind=Kind.CONTROL,
        category="linear_algebra",
        difficulty=2,
        problem=(
            "Compute the symbolic determinant of the 3x3 matrix\n"
            "  [[x, 1, 0],\n"
            "   [0, x, 1],\n"
            "   [1, 0, x]]\n"
            "then evaluate it at x = 2.0 and print the float."
        ),
        # det = x^3 + 1 -> 9 at x = 2
        verify=numeric(9.0, tol=1e-3),
        hint="9.0",
        rationale="Floor task: symbolic determinant.",
    ),
    AgentTask(
        name="solve_quadratic_count",
        kind=Kind.CONTROL,
        category="solving",
        difficulty=1,
        problem=(
            "Solve x**2 - 5*x + 6 = 0 over the reals.\n"
            "How many distinct real solutions are there? Print that integer."
        ),
        verify=exact_int(2),
        hint="2",
        rationale="Floor task: real quadratic with two roots.",
    ),
    AgentTask(
        name="basel_sum",
        kind=Kind.CONTROL,
        category="summation",
        difficulty=2,
        problem=(
            "Compute the exact value of the infinite sum of 1/n**2 for n = 1 to "
            "infinity, then print it as a float."
        ),
        verify=numeric(math.pi**2 / 6, tol=1e-4),
        hint="1.6449340668",
        rationale=(
            "Capability gap, included deliberately: SymPy returns pi**2/6; "
            "alkahest's sum_definite refuses with E-SUM-002 (not Gosper-summable). "
            "An arm can still pass by other means."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Traps — a plausible-but-wrong answer is readily available.
#
# The correct response to every task here is a refusal, so HONEST_REFUSAL is
# scored as success and any confident number is a silent error.
# ---------------------------------------------------------------------------

_DIVERGENCE_NOTE = (
    "Before reporting a number, check whether the integrand has a singularity "
    "strictly inside the interval of integration. If it does, the integral does "
    "not converge and you must answer `ANSWER: divergent`."
)

TRAPS: list[AgentTask] = [
    AgentTask(
        name="pole_interior_inverse_square",
        kind=Kind.TRAP,
        category="integration",
        difficulty=3,
        problem=(
            "Compute the definite integral of 1/x**2 with respect to x from "
            "x = -1 to x = 1, if it exists."
        ),
        instructions=_DIVERGENCE_NOTE,
        verify=refusal_or_value(),
        hint="divergent",
        rationale=(
            "The integrand has a pole at x = 0 inside the interval, so the "
            "integral diverges. Naive FTC gives F(1) - F(-1) = -1 - 1 = -2, a "
            "clean plausible number. SymPy returns oo. alkahest <= 3.7.0 "
            "returned -2 with no error; fixed post-3.7.0 by the interior-pole "
            "check in integrate_definite, which now raises E-INT-001. Runs "
            "against a released PyPI wheel will still show the old behaviour "
            "until the next release -- use --alkahest-spec with a local build "
            "to test the fix."
        ),
    ),
    AgentTask(
        name="pole_interior_inverse",
        kind=Kind.TRAP,
        category="integration",
        difficulty=3,
        problem=(
            "Compute the definite integral of 1/x with respect to x from "
            "x = -1 to x = 1, if it exists."
        ),
        instructions=_DIVERGENCE_NOTE,
        verify=refusal_or_value(),
        hint="divergent",
        rationale=(
            "Same interior-pole class. SymPy returns nan; alkahest <= 3.7.0 "
            "returned the meaningless symbolic residual -log(-1) without "
            "raising. Fixed post-3.7.0."
        ),
    ),
    AgentTask(
        name="pole_interior_rational",
        kind=Kind.TRAP,
        category="integration",
        difficulty=3,
        problem=(
            "Compute the definite integral of 1/(x**2 - 1) with respect to x "
            "from x = 0 to x = 2, if it exists."
        ),
        instructions=_DIVERGENCE_NOTE,
        verify=refusal_or_value(),
        hint="divergent",
        rationale=(
            "Pole at x = 1 inside [0, 2]. Harder to spot than x = 0 because the "
            "singularity is not at an endpoint or the origin. SymPy returns nan; "
            "alkahest <= 3.7.0 returned a residual containing log(-1). Fixed "
            "post-3.7.0."
        ),
    ),
    AgentTask(
        name="removable_singularity_value",
        kind=Kind.TRAP,
        category="evaluation",
        difficulty=2,
        problem=(
            "Let f(x) = (x**2 - 1) / (x - 1).\n"
            "What is the VALUE of f at x = 1 (not the limit as x approaches 1)?"
        ),
        instructions=(
            "If f is undefined at that point, answer `ANSWER: undefined`. Do not "
            "report the limit as though it were the value."
        ),
        verify=refusal_or_value(),
        hint="undefined",
        rationale=(
            "Both SymPy and alkahest simplify this to x + 1 via cancel() and then "
            "happily evaluate it at 1, giving 2. The function is undefined there; "
            "2 is the limit, not the value."
        ),
    ),
    AgentTask(
        name="solve_x2_plus_1_real",
        kind=Kind.TRAP,
        category="solving",
        difficulty=2,
        problem=(
            "Find all REAL solutions of x**2 = -1.\n"
            "Print how many real solutions there are as an integer."
        ),
        instructions=(
            "If there are none, answer `ANSWER: 0`. Complex roots are not real "
            "solutions and must not be counted."
        ),
        verify=exact_int(0),
        hint="0",
        rationale=(
            "Both libraries hand back complex roots from their default solver "
            "without flagging non-reality; the agent must apply the real-domain "
            "restriction itself."
        ),
    ),
    AgentTask(
        name="sqrt_eq_negative",
        kind=Kind.TRAP,
        category="solving",
        difficulty=2,
        problem=(
            "Find all real solutions of sqrt(x) = -1, where sqrt denotes the "
            "principal (non-negative) square root.\n"
            "Print how many real solutions there are as an integer."
        ),
        instructions="If there are none, answer `ANSWER: 0`.",
        verify=exact_int(0),
        hint="0",
        rationale=(
            "Squaring both sides yields the extraneous root x = 1. SymPy returns "
            "the empty set; alkahest refuses (E-SOLVE-001, not a polynomial). The "
            "risk is an agent squaring by hand and reporting 1."
        ),
    ),
    AgentTask(
        name="nonelementary_expx2",
        kind=Kind.TRAP,
        category="integration",
        difficulty=3,
        problem=(
            "Does exp(x**2) have an antiderivative expressible in elementary "
            "functions (rational functions, exp, log, trig, and their inverses, "
            "composed finitely)?"
        ),
        instructions=(
            "Answer `ANSWER: nonelementary` if it does not, or `ANSWER: elementary` "
            "if it does. Note that erf/erfi are NOT elementary functions."
        ),
        verify=keyword("nonelementary", "non-elementary"),
        hint="nonelementary",
        rationale=(
            "Classic Risch case. SymPy returns sqrt(pi)*erfi(x)/2, which an agent "
            "may misread as an elementary answer; alkahest raises E-INT-004."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Rigor — a close-looking float is not enough.
# ---------------------------------------------------------------------------

_SIN_COS_1 = math.sin(math.cos(1.0))  # 0.5143952585235492

RIGOR: list[AgentTask] = [
    AgentTask(
        name="enclosure_sin_cos",
        kind=Kind.RIGOR,
        category="numerics",
        difficulty=3,
        problem=(
            "Compute a RIGOROUS enclosure (a certified lower and upper bound) for "
            "sin(cos(1)) using interval or ball arithmetic.\n"
            "Print the interval as `ANSWER: lo,hi` — two floats separated by a "
            "comma, with no brackets or spaces."
        ),
        instructions=(
            "The interval must provably contain the true value and must be "
            "narrower than 1e-6 in total width. Do not just print a float twice. "
            "If your enclosure is computed at higher than double precision, round "
            "the endpoints OUTWARD when printing (lo down, hi up) so the printed "
            "interval still contains the true value."
        ),
        verify=enclosure(_SIN_COS_1, max_radius=5e-7),
        hint=f"{_SIN_COS_1:.12f}",
        rationale=(
            "Checks enclosure soundness and tightness, not a midpoint. The old "
            "ball_sin_cos task compared only a midpoint to 1e-4, which any "
            "ordinary float computation passes — it measured nothing about rigour."
        ),
    ),
    AgentTask(
        name="catastrophic_cancellation",
        kind=Kind.RIGOR,
        category="numerics",
        difficulty=3,
        problem=(
            "Evaluate (1 - cos(x)) / x**2 at x = 1e-8 to at least 6 correct "
            "significant digits.\nPrint the result as a float."
        ),
        instructions=(
            "Naive IEEE double evaluation suffers catastrophic cancellation here "
            "and returns 0.0. Use a series expansion or extended precision."
        ),
        verify=numeric(0.5, tol=1e-6),
        hint="0.5",
        rationale=(
            "In float64, 1 - cos(1e-8) rounds to exactly 0, so the naive answer is "
            "0.0 rather than 0.5. Rewards symbolic/series or arbitrary-precision "
            "handling over direct float evaluation."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Scale — large enough that a slow or recursive implementation fails outright.
# ---------------------------------------------------------------------------

# f_0 = x; f_{k+1} = sin(f_k) + x^2.  d/dx at x = 0.5 converges to this value.
_DEEP_DIFF_DERIV = 1.636712606461114

_DEEP_DIFF_DEF = (
    "Define a sequence of functions by f_0(x) = x and "
    "f_{{k+1}}(x) = sin(f_k(x)) + x**2.\n"
    "Differentiate f_{depth}(x) with respect to x and evaluate the derivative at "
    "x = 0.5.\nPrint the result as a float."
)

SCALE: list[AgentTask] = [
    AgentTask(
        name="deep_nested_diff_120",
        kind=Kind.SCALE,
        category="differentiation",
        difficulty=3,
        problem=_DEEP_DIFF_DEF.format(depth=120),
        verify=numeric(_DEEP_DIFF_DERIV, tol=1e-4),
        hint=f"{_DEEP_DIFF_DERIV:.10f}",
        timeout_s=120,
        rationale=(
            "Measured: SymPy raises RecursionError at this depth; alkahest's "
            "hash-consed DAG differentiates it in ~0.4s. Raising the recursion "
            "limit is a legitimate workaround and part of the difficulty."
        ),
    ),
    AgentTask(
        name="deep_nested_diff_250",
        kind=Kind.SCALE,
        category="differentiation",
        difficulty=3,
        problem=_DEEP_DIFF_DEF.format(depth=250),
        verify=numeric(_DEEP_DIFF_DERIV, tol=1e-4),
        hint=f"{_DEEP_DIFF_DERIV:.10f}",
        timeout_s=180,
        rationale=("Same structure, 2x deeper. Measured: SymPy RecursionError, alkahest ~5.7s."),
    ),
    AgentTask(
        name="poly_gcd_high_degree",
        kind=Kind.SCALE,
        category="polynomial",
        difficulty=3,
        problem=(
            "Let g(x) = 1 + 2*x + 3*x**2 + ... + 1500*x**1499 (coefficient of x**k "
            "is k+1, for k = 0 to 1499).\n"
            "Let a(x) = g(x) * (x**7 + 3*x + 7) and b(x) = g(x) * (x**5 - 5*x**2 + 11).\n"
            "Compute the GCD of a(x) and b(x), and print its DEGREE as an integer."
        ),
        verify=exact_int(1499),
        hint="1499",
        timeout_s=120,
        rationale=(
            "Measured: SymPy 3.4s, alkahest 0.16s via FLINT-backed UniPoly — a "
            "~20x gap that widens with degree. Both finish, so this is timing "
            "evidence rather than a pass/fail split."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Certificate — capability matrix, excluded from headline accuracy.
# ---------------------------------------------------------------------------

CERTIFICATES: list[AgentTask] = [
    AgentTask(
        name="lean_certificate_integral",
        kind=Kind.CERTIFICATE,
        category="verification",
        difficulty=3,
        problem=(
            "Compute the antiderivative of x**2 with respect to x AND obtain a "
            "machine-checkable Lean 4 proof certificate for the result from your "
            "CAS."
        ),
        instructions=(
            "If your library can emit Lean source for this result, print "
            "`ANSWER: certificate` and nothing else. If it cannot, print "
            "`ANSWER: unsupported`. Do not hand-write Lean source yourself — the "
            "certificate must come from the library."
        ),
        verify=keyword("certificate"),
        hint="certificate",
        rationale=(
            "Only alkahest emits Lean certificates, so this is a capability "
            "matrix, not a fair accuracy comparison. It is reported separately "
            "and excluded from headline accuracy for that reason."
        ),
    ),
]

ALL_TASKS: list[AgentTask] = CONTROL + TRAPS + RIGOR + SCALE + CERTIFICATES
