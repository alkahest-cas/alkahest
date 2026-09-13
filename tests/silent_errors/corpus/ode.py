"""Silent-error cases for ode.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import alkahest as ak
import alkahest.experimental as ex
from contracts import Case, RefusesOr, Returns

from ._shared import POOL, X, _int, _rat

_ODE_T = POOL.symbol("t")
_ODE_Y = POOL.symbol("y_ode")
_ODE_DERIVS = tuple(POOL.symbol("y_ode" + "'" * k) for k in range(1, 5))
_YP, _YPP, _YPPP = _ODE_DERIVS[0], _ODE_DERIVS[1], _ODE_DERIVS[2]
_PI = POOL.symbol("pi")
_ODE_CONSTANTS = (_rat(13, 10), _rat(7, 10), _rat(19, 10), _rat(1, 2))
_ODE_SAMPLES = (0.31, 0.57, 1.13)
_ODE_Y_SAMPLES = (0.43, 1.7)
_ODE_K = POOL.symbol("k")
_ODE_KM = POOL.symbol("Km")
_ODE_VM = POOL.symbol("Vm")
_ODE_KA = POOL.symbol("ka")
_ODE_KE = POOL.symbol("ke")
_S1 = POOL.symbol("s1")
_S2 = POOL.symbol("s2")
_S3 = POOL.symbol("s3")


def _ode_branch(equation: ak.Expr, order: int) -> dict[str, Any]:
    branches = ex.dsolve(equation, X, _ODE_Y, list(_ODE_DERIVS[:order]))
    if not branches:
        raise ValueError("dsolve reported success with no branch")
    return branches[0]


def _ode_env(extra: dict[ak.Expr, float]) -> dict[ak.Expr, float]:
    env: dict[ak.Expr, float] = {_PI: math.pi}
    env.update(extra)
    return env


def ode_residual(
    equation: ak.Expr, order: int, params: dict[ak.Expr, ak.Expr] | None = None
) -> Callable[[], float]:
    """Answer = max |equation| after substituting alkahest's *own* answer back.

    Never asserts the shape of a solution — `C1·eˣ` and `e^{x+C1}` are the same
    family and a corpus that preferred one would measure spelling.  What it
    asserts is the only thing that means anything: that differentiating the
    returned `y(x)` and putting it, and its derivatives, back into the equation
    leaves zero.  A wrong solution cannot survive that; a differently-written
    right one is unaffected.

    Implicit answers (`G(x, y) = 0`, which separable and exact classes often
    return) are handled in their own right rather than being scored as a
    refusal: the implicit function theorem gives the slope `y' = −Gₓ/G_y`, and
    the equation must vanish on a free `(x, y)` grid — a stronger claim than the
    explicit one, since it is an identity in two variables.

    *params* binds the equation's own symbolic coefficients.  Sampling one of
    them at a **negative** value is the point of some of these cases.
    """
    params = dict(params or {})

    def op() -> float:
        sol = _ode_branch(equation, order)
        binding: dict[ak.Expr, ak.Expr] = {
            c: _ODE_CONSTANTS[i % len(_ODE_CONSTANTS)] for i, c in enumerate(sol["constants"])
        }
        binding.update(params)
        worst = 0.0
        if sol["form"] == "explicit":
            y = ak.subs(sol["y_of_x"], binding)
            mapping: dict[ak.Expr, ak.Expr] = {_ODE_Y: y}
            cur = y
            for dsym in _ODE_DERIVS[:order]:
                cur = ak.diff(cur, X).value
                mapping[dsym] = cur
            resid = ak.subs(ak.subs(equation, mapping), params)
            for xv in _ODE_SAMPLES:
                worst = max(worst, abs(float(ak.eval_expr(resid, _ode_env({X: xv})))))
            return worst
        g = sol["implicit_relation"]
        gx = ak.subs(ak.diff(g, X).value, params)
        gy = ak.subs(ak.diff(g, _ODE_Y).value, params)
        resid = ak.subs(ak.subs(equation, {_ODE_DERIVS[0]: -gx / gy}), params)
        for xv in _ODE_SAMPLES:
            for yv in _ODE_Y_SAMPLES:
                worst = max(worst, abs(float(ak.eval_expr(resid, _ode_env({X: xv, _ODE_Y: yv})))))
        return worst

    return op


def _numeric_rank(rows: list[list[float]], tol: float = 1e-7) -> int:
    """Rank of a small float matrix by Gauss–Jordan with a pivot threshold."""
    rows = [list(r) for r in rows]
    rank = 0
    for col in range(len(rows[0])):
        piv = next((r for r in range(rank, len(rows)) if abs(rows[r][col]) > tol), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        pivot = rows[rank][col]
        for r in range(len(rows)):
            if r != rank and rows[r][col] != 0.0:
                factor = rows[r][col] / pivot
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    return rank


def ode_family_rank(
    equation: ak.Expr, order: int, samples: tuple[float, ...] = (0.31, 0.57, 0.83, 1.19, 1.61)
) -> Callable[[], int]:
    """Answer = how many independent directions the returned family spans.

    For an order-`n` linear ODE the general solution is an `n`-dimensional
    family, so `rank[∂y/∂C_i(x_j)]` must be `n`.  A family that is a *solution*
    but not the general one — the repeated-root answer written with the
    distinct-root formula — has full residual agreement and a deficient rank,
    and this is the only check in the corpus that can tell the difference.

    A caller who integrates such an answer against initial conditions gets an
    unsolvable linear system, or, worse, a least-squares fit that quietly
    ignores half of them.
    """

    def op() -> int:
        sol = _ode_branch(equation, order)
        if sol["form"] != "explicit":
            raise ValueError("a family's dimension is only defined for an explicit answer")
        binding = {
            c: _ODE_CONSTANTS[i % len(_ODE_CONSTANTS)] for i, c in enumerate(sol["constants"])
        }
        rows = []
        for c in sol["constants"]:
            partial = ak.subs(ak.diff(sol["y_of_x"], c).value, binding)
            rows.append([float(ak.eval_expr(partial, _ode_env({X: s}))) for s in samples])
        return _numeric_rank(rows)

    return op


def ode_states_its_branch_condition(equation: ak.Expr, order: int) -> Callable[[], bool]:
    """Answer = did the answer come with the condition it is only valid under?

    A symbolic-coefficient answer whose discriminant can vanish is *undefined*
    at the confluence, not merely non-general.  Returning it bare is a silent
    error of the narrowing kind: the formula is right where it is defined and
    the caller has no way to learn where that is.
    """

    def op() -> bool:
        sol = _ode_branch(equation, order)
        return bool(sol["side_conditions"]) and bool(sol["notes"])

    return op


def _system_solution(states: list[ak.Expr], rhs: list[ak.Expr]) -> dict[str, Any]:
    return ex.dsolve_system(ak.ODE(states, rhs, _ODE_T))


def system_residual(
    states: list[ak.Expr], rhs: list[ak.Expr], params: dict[ak.Expr, ak.Expr] | None = None
) -> Callable[[], float]:
    """Answer = max |y_i'(t) − rhs_i(y(t))| over every component and sample.

    The system analogue of :func:`ode_residual`: every component of the returned
    `y(t)` is differentiated and checked against *its own* equation with the
    whole state substituted, so a solution that is right in one coordinate and
    wrong in another cannot average out.
    """
    params = dict(params or {})

    def op() -> float:
        sol = _system_solution(states, rhs)
        binding: dict[ak.Expr, ak.Expr] = {
            c: _ODE_CONSTANTS[i % len(_ODE_CONSTANTS)] for i, c in enumerate(sol["constants"])
        }
        binding.update(params)
        ys = [ak.subs(e, binding) for e in sol["y_of_t"]]
        state_map = dict(zip(states, ys))
        worst = 0.0
        for i in range(len(states)):
            resid = ak.subs(ak.diff(ys[i], _ODE_T).value - ak.subs(rhs[i], state_map), params)
            for tv in _ODE_SAMPLES:
                worst = max(worst, abs(float(ak.eval_expr(resid, _ode_env({_ODE_T: tv})))))
        return worst

    return op


def system_states_its_branch_condition(
    states: list[ak.Expr], rhs: list[ak.Expr]
) -> Callable[[], bool]:
    """Answer = did the system's answer disclose the confluence it divides by?"""

    def op() -> bool:
        sol = _system_solution(states, rhs)
        return bool(sol["side_conditions"]) and bool(sol["notes"])

    return op


CASES: list[Case] = [
    Case(
        id="ode_repeated_root_family_spans_two_dimensions",
        subsystem="ode",
        statement="y'' − 2y' + y = 0 has a 2-dimensional solution space: (C1 + C2·x)·eˣ",
        op=ode_family_rank(_YPP - _int(2) * _YP + _ODE_Y, 2),
        contract=Returns(2),
        verified_by=(
            "The characteristic polynomial is (r−1)², a double root, so a fundamental system "
            "is {eˣ, x·eˣ} (reduction of order: y = v·eˣ gives v'' = 0). Existence and "
            "uniqueness make the solution space of a second-order linear ODE exactly "
            "2-dimensional, so the rank of [∂y/∂C_i] must be 2."
        ),
        note=(
            "The one ODE silent error that substituting the answer back cannot find. The "
            "distinct-root formula applied without checking the discriminant yields "
            "C1·eˣ + C2·eˣ, whose residual is identically zero and whose family is a line, "
            "not a plane. alkahest's own gate is substitution-based and is blind to it by "
            "construction, which is exactly why the check belongs here."
        ),
    ),
    Case(
        id="ode_triple_root_family_spans_three_dimensions",
        subsystem="ode",
        statement="y''' − 3y'' + 3y' − y = 0 has the 3-dimensional space (C1+C2x+C3x²)eˣ",
        op=ode_family_rank(_YPPP - _int(3) * _YPP + _int(3) * _YP - _ODE_Y, 3),
        contract=Returns(3),
        verified_by=(
            "The characteristic polynomial is (r−1)³. A root of multiplicity m contributes "
            "{eʳˣ, x·eʳˣ, …, x^{m−1}·eʳˣ}, so {eˣ, x·eˣ, x²·eˣ} is a fundamental system and "
            "the rank must be 3."
        ),
        note=(
            "The second-order case above can be passed by special-casing a double root; a "
            "triple root cannot, so the two together test the rule rather than one instance."
        ),
    ),
    Case(
        id="ode_euler_cauchy_repeated_root_needs_its_logarithm",
        subsystem="ode",
        statement="x²y'' − xy' + y = 0 has the 2-dimensional space C1·x + C2·x·log x",
        op=ode_family_rank(X * X * _YPP - X * _YP + _ODE_Y, 2),
        contract=Returns(2),
        verified_by=(
            "x = e^t turns it into the constant-coefficient equation with indicial polynomial "
            "r² − 2r + 1 = (r−1)², a double root at r = 1, so the second basis element is "
            "x·log x, not a second copy of x."
        ),
        note=(
            "The Euler–Cauchy spelling of the trap above: a solver that reads two equal "
            "indicial roots as two basis elements x¹ and x¹ returns a rank-1 family whose "
            "residual is zero."
        ),
    ),
    Case(
        id="ode_control_distinct_roots_still_span",
        subsystem="ode",
        statement="y'' + y = 0 has the 2-dimensional space C1·cos x + C2·sin x",
        op=ode_family_rank(_YPP + _ODE_Y, 2),
        contract=Returns(2),
        verified_by=(
            "Characteristic roots ±i, distinct, so {cos x, sin x} is a fundamental system "
            "(Wronskian cos²+sin² = 1 ≠ 0 everywhere). The control for the three repeated-root "
            "cases: a solver that answered every equation with a rank-deficient family, or a "
            "rank routine that always said 2, would be caught by one of the four."
        ),
    ),
    Case(
        id="ode_resonant_forcing_is_not_the_naive_ansatz",
        subsystem="ode",
        statement="y'' + y = cos x needs the secular ½·x·sin x, not A·cos x + B·sin x",
        op=ode_residual(_YPP + _ODE_Y - POOL.func("cos", [X]), 2),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "cos x solves the homogeneous equation, so every A·cos x + B·sin x is annihilated "
            "by y'' + y and no such ansatz can produce the forcing term. Variation of "
            "parameters gives y_p = ½·x·sin x, and (½x sin x)'' + ½x sin x = cos x by hand."
        ),
        note=(
            "The classic undetermined-coefficients trap: the linear system for A and B is "
            "singular, and an implementation that solves it by elimination without checking "
            "produces a clean wrong particular solution instead of detecting the resonance."
        ),
    ),
    Case(
        id="ode_sqrt_of_a_square_coefficient_keeps_its_modulus",
        subsystem="ode",
        statement="y' = √(k²)·y at k = −13/10 solves as e^{+1.3x}, not e^{−1.3x}",
        op=ode_residual(
            _YP - POOL.func("sqrt", [_ODE_K ** _int(2)]) * _ODE_Y, 1, {_ODE_K: _rat(-13, 10)}
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "√(k²) = |k| for real k, so at k = −13/10 the equation is y' = 1.3·y and its "
            "solutions are C·e^{1.3x}. The tempting rewrite √(k²) → k is valid only for "
            "k ≥ 0 and gives C·e^{−1.3x}, which is a solution of a different equation."
        ),
        note=(
            "The sign-of-a-parameter trap, sampled where it bites. A verifier that only ever "
            "binds symbolic parameters to positive values cannot distinguish the two answers "
            "at all — both certify — so this case is as much a test of the gate as of the "
            "solver."
        ),
    ),
    Case(
        id="ode_michaelis_menten_negative_km",
        subsystem="ode",
        statement="(Km + y)·y' + Vm·y = 0 at Km = −11/10: solve it, or say you cannot",
        op=ode_residual(
            (_ODE_KM + _ODE_Y) * _YP + _ODE_VM * _ODE_Y,
            1,
            {_ODE_KM: _rat(-11, 10), _ODE_VM: _rat(3, 5)},
        ),
        contract=RefusesOr(0.0, tol=1e-9),
        verified_by=(
            "Separating variables gives Km·log y + y + Vm·x = C for every non-zero Km; the "
            "sign of Km changes nothing about the derivation. Inverting it through Lambert W "
            "does depend on the sign — the argument of W₀ is negative for Km < 0 and leaves "
            "[−1/e, ∞) — so 'no answer here' is defensible and 'here is a formula' is only "
            "acceptable if the formula solves the equation."
        ),
        note=(
            "Passes today by a *weak* refusal: the Lambert-W inversion is returned and "
            "eval_expr declines it at these samples. The Km > 0 control below is the half "
            "that must keep working."
        ),
    ),
    Case(
        id="ode_control_michaelis_menten_positive_km",
        subsystem="ode",
        statement="(Km + y)·y' + Vm·y = 0 at Km = 11/10, Vm = 3/5 solves",
        op=ode_residual(
            (_ODE_KM + _ODE_Y) * _YP + _ODE_VM * _ODE_Y,
            1,
            {_ODE_KM: _rat(11, 10), _ODE_VM: _rat(3, 5)},
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Km·log y + y + Vm·x = C, differentiated: (Km/y + 1)·y' + Vm = 0, i.e. "
            "(Km + y)·y' + Vm·y = 0. The control for the negative-Km case: without it, a "
            "solver that refused this equation outright would score a clean pass there."
        ),
    ),
    Case(
        id="ode_riccati_without_a_particular_solution",
        subsystem="ode",
        statement="y' = y² + x has no elementary solution — refuse, or produce one that works",
        op=ode_residual(_YP - _ODE_Y * _ODE_Y - X, 1),
        contract=RefusesOr(0.0, tol=1e-9),
        verified_by=(
            "y = −u'/u linearises it to u'' + x·u = 0, the Airy equation, whose solutions are "
            "not elementary (Liouville's theorem; Kovacic's algorithm returns no case). So no "
            "elementary closed form exists and the honest answer is a refusal."
        ),
        note=(
            "The Riccati trap: guess a polynomial particular solution, substitute it into the "
            "reduction, and out comes a clean formula for an equation that has none. The "
            "contract admits an answer only if it solves the equation, so a future "
            "Airy-function answer passes and a guessed one does not."
        ),
    ),
    Case(
        id="ode_abel_first_kind_has_no_closed_form",
        subsystem="ode",
        statement="y' = y³ + x — refuse, or produce a y(x) that solves it",
        op=ode_residual(_YP - _ODE_Y ** _int(3) - X, 1),
        contract=RefusesOr(0.0, tol=1e-9),
        verified_by=(
            "An Abel equation of the first kind with no known closed-form solution; it is not "
            "separable, linear, exact, homogeneous, Bernoulli, Clairaut or Riccati, and no "
            "integrating factor of the standard forms applies. Kamke lists no solution."
        ),
        note=(
            "The nearest neighbour of the Riccati case one power up: the classes that *do* "
            "match a y² right-hand side must not match this one by pattern alone."
        ),
    ),
    Case(
        id="ode_clairaut_family_is_the_general_solution",
        subsystem="ode",
        statement="y = x·y' + (y')² solves as the line family y = C·x + C²",
        op=ode_residual(_ODE_Y - X * _YP - _YP ** _int(2), 1),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Substituting y = Cx + C² gives y' = C and Cx + C² = x·C + C², an identity. "
            "(The parabola y = −x²/4 is the *singular* solution, an envelope of the family "
            "and not a member of it; a solver that returned it as 'the general solution' "
            "would still have residual zero, which is what the rank case below is for.)"
        ),
    ),
    Case(
        id="ode_clairaut_family_is_one_dimensional",
        subsystem="ode",
        statement="the Clairaut general solution is a one-parameter family, not a single curve",
        op=ode_family_rank(_ODE_Y - X * _YP - _YP ** _int(2), 1),
        contract=Returns(1),
        verified_by=(
            "y = Cx + C² carries one arbitrary constant, so ∂y/∂C = x + 2C is not identically "
            "zero and the rank is 1. The envelope y = −x²/4 carries none, so returning it "
            "instead gives rank 0 — the same equation's other, non-general, solution."
        ),
    ),
    Case(
        id="ode_symbolic_discriminant_confluence_is_disclosed",
        subsystem="ode",
        statement="y'' − k²y = 0's two-exponential answer is not general at k = 0, and says so",
        op=ode_states_its_branch_condition(_YPP - _ODE_K ** _int(2) * _ODE_Y, 2),
        contract=Returns(True),
        verified_by=(
            "The characteristic roots are ±k. At k = 0 they coincide and the general solution "
            "is C1 + C2·x, which no C1·e^{kx} + C2·e^{−kx} spans — at k = 0 that family "
            "degenerates to the constants. So the returned family is the general solution on "
            "k ≠ 0 only, and the condition is part of the answer rather than an aside."
        ),
        note=(
            "A narrowing silent error: the formula is right everywhere it is defined, and a "
            "caller who substitutes k = 0 into it gets a one-dimensional family presented as "
            "a two-dimensional one with nothing to warn them."
        ),
    ),
    Case(
        id="ode_system_confluent_rates_are_disclosed",
        subsystem="ode",
        statement="the two-compartment model x'=−ka·x, y'=ka·x−ke·y divides by (ka−ke)",
        op=system_states_its_branch_condition(
            [_S1, _S2], [-_ODE_KA * _S1, _ODE_KA * _S1 - _ODE_KE * _S2]
        ),
        contract=Returns(True),
        verified_by=(
            "The coefficient matrix has eigenvalues −ka and −ke; the second component of the "
            "solution carries a factor 1/(ke − ka) (integrate ka·C·e^{−ka t} against the "
            "e^{−ke t} kernel). At ka = ke that expression is 0/0 — undefined as written, "
            "though its limit is the confluent t·e^{−ka t} form — so the caller must be told."
        ),
        note=(
            "The commonest real parameterisation of this model in pharmacokinetics is "
            "ka ≈ ke, so the excluded point is not a corner case: it is where a fitter lands."
        ),
    ),
    Case(
        id="ode_control_numeric_system_states_no_condition",
        subsystem="ode",
        statement="x'=x+2y, y'=3x+2y has integer eigenvalues 4 and −1 and needs no caveat",
        op=system_states_its_branch_condition(
            [_S1, _S2], [_S1 + _int(2) * _S2, _int(3) * _S1 + _int(2) * _S2]
        ),
        contract=Returns(False),
        verified_by=(
            "det(A − λI) = (1−λ)(2−λ) − 6 = λ² − 3λ − 4 = (λ−4)(λ+1): two distinct integer "
            "eigenvalues, nothing undecided. The control for the two disclosure cases — a "
            "solver that emitted a side condition unconditionally would pass both of those "
            "and fail here."
        ),
    ),
    Case(
        id="ode_system_casus_irreducibilis_eigenvalues",
        subsystem="ode",
        statement="x'=−z, y'=x+3z, z'=y — three real eigenvalues written with acos and π",
        op=system_residual([_S1, _S2, _S3], [-_S3, _S1 + _int(3) * _S3, _S2]),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "The coefficient matrix is the companion matrix of λ³ − 3λ + 1, whose discriminant "
            "is positive, so it has three distinct real roots and Cardano's cube roots are "
            "complex — the casus irreducibilis. The roots are 2·cos((acos(−1/2) + 2πk)/3), "
            "k = 0,1,2, correct only at the true π. Each returned component is differentiated "
            "and matched against its own equation."
        ),
        note=(
            "Refused with E-ODE-034 before 3.10: the verifier collected `pi` as a free symbol "
            "and bound it to a sample value like 1.7, so the correct eigenvalues disagreed at "
            "every sample. π is a plain symbol in this library, which is why the sample "
            "environment here binds it explicitly too."
        ),
    ),
    Case(
        id="ode_control_two_compartment_system_solves",
        subsystem="ode",
        statement="x'=−0.8x, y'=0.8x−0.3y — both components must solve their own equation",
        op=system_residual(
            [_S1, _S2],
            [-_ODE_KA * _S1, _ODE_KA * _S1 - _ODE_KE * _S2],
            {_ODE_KA: _rat(4, 5), _ODE_KE: _rat(3, 10)},
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "x = C·e^{−0.8t} by inspection; y' + 0.3y = 0.8·C·e^{−0.8t} is first-order linear "
            "with solution y = (0.8C/(0.3−0.8))·e^{−0.8t} + D·e^{−0.3t}. Checked by "
            "substitution rather than by comparing to that form, so any equivalent spelling "
            "passes."
        ),
    ),
    Case(
        id="ode_control_linear_first_order",
        subsystem="ode",
        statement="y' − 3y = x solves (integrating factor e^{−3x})",
        op=ode_residual(_YP - _int(3) * _ODE_Y - X, 1),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "y = C·e^{3x} − x/3 − 1/9: y' = 3C·e^{3x} − 1/3 and 3y + x = 3C·e^{3x} − x − 1/3 "
            "+ x, equal. Derived by hand from the integrating factor."
        ),
    ),
    Case(
        id="ode_control_separable_movable_pole",
        subsystem="ode",
        statement="y' = 1 + y² solves as tan(x + C), poles and all",
        op=ode_residual(_YP - _int(1) - _ODE_Y * _ODE_Y, 1),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "d/dx tan(x+C) = sec²(x+C) = 1 + tan²(x+C). The solution has a movable pole at "
            "x + C = π/2, which is a property of the equation, not an error: a gate that "
            "declined every candidate with a singularity would refuse this correct answer."
        ),
    ),
    Case(
        id="ode_control_euler_cauchy_distinct_roots",
        subsystem="ode",
        statement="x²y'' + xy' − y = 0 solves as C1·x + C2/x",
        op=ode_residual(X * X * _YPP + X * _YP - _ODE_Y, 2),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Indicial equation r(r−1) + r − 1 = r² − 1, roots ±1, so x and x⁻¹ are solutions; "
            "x²·(2x⁻³) + x·(−x⁻²) − x⁻¹ = 2x⁻¹ − x⁻¹ − x⁻¹ = 0 by hand."
        ),
    ),
    Case(
        id="ode_control_implicit_relation_defines_the_slope_field",
        subsystem="ode",
        statement="(2x + y) + (x + 2y)·y' = 0 is exact; its implicit answer must define the ODE",
        op=ode_residual((_int(2) * X + _ODE_Y) + (X + _int(2) * _ODE_Y) * _YP, 1),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "∂/∂y(2x+y) = 1 = ∂/∂x(x+2y), so the equation is exact with potential "
            "F = x² + xy + y². The implicit function theorem gives y' = −F_x/F_y = "
            "−(2x+y)/(x+2y) on the whole grid, which is the equation itself. Scored on the "
            "two-variable identity, not on the shape of the relation."
        ),
        note=(
            "The control for the implicit half of ode_residual: without it every implicit "
            "answer in the corpus could be scored as a refusal and nobody would notice."
        ),
    ),
]
