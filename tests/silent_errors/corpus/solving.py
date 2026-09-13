"""Silent-error cases for solving.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable

import alkahest as ak
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import _A, _B, _Z, POOL, X, Y, _int, _num, _survives_a_panic


def real_solution_count(equations: list[ak.Expr], unknowns: list[ak.Expr]) -> Callable[[], int]:
    """Answer = how many real solutions ``solve(..., domain="real")`` reports."""

    def op() -> int:
        sols = ak.solve(equations, unknowns, domain="real")
        return len(sols)

    return op


def solution_count(
    equations: list[ak.Expr], unknowns: list[ak.Expr], **kwargs: Any
) -> Callable[[], int]:
    """Answer = how many solutions ``solve`` reports over ℂ.

    A count is the sharpest single number for a solver: it moves if a spurious
    tuple is added, if a true one is dropped, and if one root is reported twice.
    A parametric (``GroebnerBasis``) answer is not a count and is surfaced as a
    refusal rather than silently scored.
    """

    def op() -> int:
        sols = ak.solve(equations, unknowns, **kwargs)
        if not isinstance(sols, list):
            raise ak.SolverError("solve returned a parametric ideal, not a solution list")
        return len(sols)

    return op


def numeric_solution_count(equations: list[ak.Expr], unknowns: list[ak.Expr]) -> Callable[[], int]:
    """Answer = how many returned tuples actually name a point of ℂⁿ.

    An entry whose coordinate is ``0·0⁻¹`` is not a solution and not a
    refusal either — it is a list entry that looks like an answer.  Counting
    only the tuples that evaluate keeps the score a number rather than an
    exception, so the case is scored as the wrong *count* it is.
    """

    def op() -> int:
        sols = ak.solve(equations, unknowns)
        if not isinstance(sols, list):
            raise ak.SolverError("solve returned a parametric ideal, not a solution list")
        n = 0
        for sol in sols:
            if all(ak.evaluate(sol[v], {}, mode="complex").status == "ok" for v in unknowns):
                n += 1
        return n

    return op


def max_solution_residual(equations: list[ak.Expr], unknowns: list[ak.Expr]) -> Callable[[], float]:
    """Answer = max |eq(sol)| over every returned solution and every equation.

    Substitution back into the original system is self-certifying: no oracle is
    consulted, and any tuple that is not a solution shows up as a residual the
    solver itself cannot explain away.  A coordinate that is not a number
    (``0·0⁻¹``) makes ``eval_expr`` raise, which the runner scores as a refusal.
    """

    def op() -> float:
        sols = ak.solve(equations, unknowns)
        if not isinstance(sols, list) or not sols:
            raise ak.SolverError("solve produced no solution list to substitute back")
        worst = 0.0
        for sol in sols:
            point = {}
            for v in unknowns:
                got = ak.evaluate(sol[v], {}, mode="complex")
                if got.status != "ok":
                    raise ak.SolverError(f"solution coordinate is not a number: {got.status}")
                point[v] = complex(got.value)
            for eq in equations:
                residual = ak.evaluate(eq, point, mode="complex")
                if residual.status != "ok":
                    raise ak.SolverError(f"residual did not evaluate: {residual.status}")
                worst = max(worst, abs(complex(residual.value)))
        return worst

    return op


def _real_root_count(coeffs: list[int]) -> Callable[[], int]:
    """Answer = how many real-root intervals ``real_roots`` reports.

    *coeffs* is in ascending degree order.
    """

    def op() -> int:
        expr = _int(0)
        for i, c in enumerate(coeffs):
            expr = expr + _int(c) * X ** _int(i)
        return len(ak.real_roots(expr, X))

    return op


def _refined_ball_brackets_root(coeffs: list[int], index: int) -> Callable[[], bool]:
    """Answer = does ``refine_root``'s ball actually contain a root?

    Checked in exact ``Fraction`` arithmetic on the ball's own endpoints: the
    polynomial must vanish at one of them or change sign across them.  This is
    the only thing the word "rigorous" can mean for an enclosure, and it needs
    no reference value — the root itself may be irrational.
    """

    def op() -> bool:
        expr = _int(0)
        for i, c in enumerate(coeffs):
            expr = expr + _int(c) * X ** _int(i)
        ball = ak.refine_root(expr, ak.real_roots(expr, X)[index], X)
        mid, rad = Fraction(ball.mid), Fraction(ball.rad)

        def value_at(t: Fraction) -> Fraction:
            return sum((Fraction(c) * t**i for i, c in enumerate(coeffs)), Fraction(0))

        lo_v, hi_v = value_at(mid - rad), value_at(mid + rad)
        return lo_v == 0 or hi_v == 0 or (lo_v > 0) != (hi_v > 0)

    return op


def _poly(coeffs: list[int]) -> ak.Expr:
    """``Σ coeffs[i]·xⁱ`` from ascending-degree integer coefficients."""
    out = _int(0)
    for i, c in enumerate(coeffs):
        out = out + _int(c) * X ** _int(i)
    return out


def _subresultant_chain(
    f_coeffs: list[int], g_coeffs: list[int], samples: tuple[float, ...] = (2.0, 3.0)
) -> Callable[[], tuple[float, ...]]:
    """Answer = every subresultant after ``[p, q]``, sampled at fixed points.

    Two sample points rather than one so the *polynomial* is pinned, not just a
    value: a chain element off by a scalar or by a term shows up at both.
    """

    def op() -> tuple[float, ...]:
        chain = ak.subresultant_prs(_poly(f_coeffs), _poly(g_coeffs), X)[2:]
        return tuple(float(ak.eval_expr(e, {X: s})) for e in chain for s in samples)

    return _survives_a_panic(op)


def _radical_membership(
    polys: list[ak.Expr], unknowns: list[ak.Expr], probes: list[ak.Expr]
) -> Callable[[], tuple[bool, ...]]:
    """Answer = which *probes* the reported √I contains.

    Membership is the only thing a caller can ask a ``GroebnerBasis``, so it is
    what the contract has to be written against: a radical that does not contain
    a polynomial whose square it does contain is refuted by its own answers, no
    oracle needed.
    """

    def op() -> tuple[bool, ...]:
        r = ak.radical(polys, unknowns)
        return tuple(bool(r.contains(p)) for p in probes)

    return op


def _component_count(polys: list[ak.Expr], unknowns: list[ak.Expr]) -> Callable[[], int]:
    """Answer = how many components ``primary_decomposition`` reports."""

    def op() -> int:
        return len(ak.primary_decomposition(polys, unknowns))

    return op


def _associated_primes_survive_a_witness(
    polys: list[ak.Expr],
    unknowns: list[ak.Expr],
    witnesses: list[tuple[ak.Expr, ak.Expr]],
) -> Callable[[], bool]:
    """Answer = does every reported ``associated_prime`` pass the definition?

    A prime ``P`` containing ``a·b`` must contain ``a`` or ``b``.  Each witness
    is such a pair, so a component that holds the product and neither factor is
    *not* prime — and the field is named ``associated_prime``, so a caller is
    entitled to treat it as one.  The check is the definition itself, run
    against the library's own membership test.
    """

    def op() -> bool:
        dec = ak.primary_decomposition(polys, unknowns)
        if not dec:
            raise ak.SolverError("primary_decomposition returned no components to check")
        for component in dec:
            prime = component.associated_prime()
            for a, b in witnesses:
                if prime.contains(a * b) and not (prime.contains(a) or prime.contains(b)):
                    return False
        return True

    return op


def _shortest_chain_length(equations: list[ak.Expr], unknowns: list[ak.Expr]) -> Callable[[], int]:
    """Answer = the fewest polynomials in any chain ``triangularize`` returns.

    A triangular set cutting out a *finite* set in ``n`` variables needs one
    polynomial per variable: with fewer, some variable is unconstrained and the
    chain describes a positive-dimensional set.  The minimum over chains is the
    number that moves the moment one generator is dropped.
    """

    def op() -> int:
        chains = ak.triangularize(equations, unknowns)
        if not chains:
            raise ak.SolverError("triangularize reported the unit ideal")
        return min(len(c.polys()) for c in chains)

    return op


def _undisclosed_solve_hypotheses(
    equations: list[ak.Expr],
    unknowns: list[ak.Expr],
    witness: dict[ak.Expr, float],
    hypothesis_about: ak.Expr,
) -> Callable[[], float]:
    """Answer = how many returned coordinates fail at *witness* without being excluded.

    ``solve([a·x − b], [x])`` returns ``b/a``.  That is the solution **for
    ``a ≠ 0``**: at ``a = 0`` the equation reads ``−b = 0``, so the system has no
    solution when ``b ≠ 0`` and *every* ``x`` when ``b = 0`` — and ``b/a`` is
    neither, it is not even a number there.  A parametric tuple is returned
    unverified by design (there is nothing to substitute back), so stating the
    hypothesis is the only honest signal available.

    The case is satisfied either way an honest library can behave: state the
    condition on *hypothesis_about* (:func:`alkahest.solve_side_conditions`), or
    do not return a tuple that fails at the witness.  Counting unexcluded
    refuting witnesses keeps the answer a finite number — the coordinate itself
    does not evaluate there, which is exactly the complaint.
    """

    def op() -> float:
        sols = ak.solve(equations, unknowns)
        if not isinstance(sols, list) or not sols:
            raise ak.SolverError("solve produced no parametric solution to audit")
        stated = any(str(hypothesis_about) in str(c) for c in ak.solve_side_conditions())
        if stated:
            return 0.0
        refuting = 0.0
        for sol in sols:
            for value in sol.values():
                if ak.evaluate(value, witness, mode="complex").status != "ok":
                    refuting += 1.0
        return refuting

    return op


def _solve_states_no_unnecessary_hypothesis(
    equations: list[ak.Expr],
    unknowns: list[ak.Expr],
    env: dict[ak.Expr, float],
    expected: float,
) -> Callable[[], float]:
    """Answer = how many hypotheses ``solve`` reported for an answer that needs none.

    The control for :func:`_undisclosed_solve_hypotheses`: a gate that a stated
    condition passes must also fail a library that states one unconditionally.
    ``2x − b = 0`` divides by the literal ``2``, provably non-zero, so the
    correct number of hypotheses is ``0`` — and the solution itself is still
    checked at *env*, so "state nothing and solve nothing" does not pass either.
    """

    def op() -> float:
        sols = ak.solve(equations, unknowns)
        if not isinstance(sols, list) or len(sols) != 1:
            raise ak.SolverError("expected exactly one parametric solution")
        (sol,) = sols
        conditions = ak.solve_side_conditions()
        got = float(ak.eval_expr(next(iter(sol.values())), env))
        if abs(got - expected) > 1e-9:
            raise AssertionError(f"solution evaluates to {got}, expected {expected}")
        return float(len(conditions))

    return op


def _roots_of(equation: ak.Expr, unknown: ak.Expr) -> list[float]:
    """Every value ``solve`` returns for *unknown*, as floats."""
    sols = ak.solve([equation], [unknown])
    if not isinstance(sols, list):
        raise ak.SolverError("solve returned an ideal, not a solution list")
    out: list[float] = []
    for sol in sols:
        value = sol[unknown]
        out.append(
            float(value) if isinstance(value, (int, float)) else float(ak.eval_expr(value, {}))
        )
    return out


def _solve_returns_a_pole_as_a_root(
    equation: ak.Expr, unknown: ak.Expr, pole: float
) -> Callable[[], float]:
    """Answer = how many returned roots sit at *pole*, where the equation is undefined.

    ``x/(x−1) = 1/(x−1)`` has no solution: at ``x = 1`` both sides read ``1/0``,
    and nowhere else are they unequal.  Multiplying up turns it into
    ``(x−1)² = 0``, whose only root is exactly that point — so a solver that
    clears denominators and stops has a clean, plausible, wrong answer waiting
    for it.  Counting the returned roots that land on the pole keeps the answer
    a finite number.
    """

    def op() -> float:
        return float(sum(1 for r in _roots_of(equation, unknown) if abs(r - pole) <= 1e-9))

    return op


def _solve_root_count(equation: ak.Expr, unknown: ak.Expr) -> Callable[[], float]:
    """Answer = how many roots ``solve`` returns for a rational equation."""

    def op() -> float:
        return float(len(_roots_of(equation, unknown)))

    return op


def _solve_finds_all_of(
    equation: ak.Expr, unknown: ak.Expr, expected: tuple[float, ...]
) -> Callable[[], float]:
    """Answer = how many of *expected* ``solve`` actually returned.

    The control for the pole cases: excluding a root because a denominator
    vanishes there must not degenerate into excluding roots, and refusing the
    whole equation must not pass either.
    """

    def op() -> float:
        got = _roots_of(equation, unknown)
        return float(sum(1 for e in expected if any(abs(r - e) <= 1e-9 for r in got)))

    return op


CASES: list[Case] = [
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
    # Solving: the solution *set* — no spurious tuples, no dropped branches,
    # no root counted twice.  A count is the sharpest single number here: it
    # moves in all three directions at once.
    # -----------------------------------------------------------------------
    Case(
        id="solve_branch_where_leading_coefficient_vanishes",
        subsystem="solving",
        statement="-3x-2xy = 0 ∧ -3y-x² = 0 has three solutions, two of them on the branch "
        "y = -3/2 where the first equation degenerates",
        op=solution_count([_int(-3) * X + _int(-2) * X * Y, _int(-3) * Y - X ** _int(2)], [X, Y]),
        contract=Returns(3),
        verified_by=(
            "-3x - 2xy = -x(3 + 2y), so either x = 0 or y = -3/2. x = 0 forces -3y = 0, giving "
            "(0,0). y = -3/2 satisfies the first equation for every x, and the second then reads "
            "9/2 - x² = 0, giving x = ±3/√2. Three points: (0,0) and (±3/√2, -3/2). Substituting "
            "each back gives 0 in both equations — no oracle involved."
        ),
    ),
    Case(
        id="solve_branch_residual_after_degenerate_split",
        subsystem="solving",
        statement="every tuple solve returns for -3x-2xy = 0 ∧ -3y-x² = 0 satisfies both equations",
        op=max_solution_residual(
            [_int(-3) * X + _int(-2) * X * Y, _int(-3) * Y - X ** _int(2)], [X, Y]
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Substitution back into the stated system is self-certifying. The reported answer "
            "(0, -3/2) has residual -3y - x² = 9/2 ≠ 0, which needs no oracle to reject."
        ),
    ),
    Case(
        id="solve_control_circle_meets_line_twice",
        subsystem="solving",
        statement="x²+y² = 1 ∧ y = x has exactly two solutions",
        op=solution_count([X ** _int(2) + Y ** _int(2) - _int(1), Y - X], [X, Y]),
        contract=Returns(2),
        verified_by=(
            "Substituting y = x gives 2x² = 1, so x = ±1/√2 and the points are ±(1/√2, 1/√2). "
            "The control for solve_branch_where_leading_coefficient_vanishes: a solver that "
            "refused every two-variable system, or that dropped one root of every quadratic, "
            "would otherwise pass that case."
        ),
    ),
    Case(
        id="solve_repeated_root_is_one_solution",
        subsystem="solving",
        statement="the solution set of x² = 0 is {0} — one element, not ±√0",
        op=solution_count([X ** _int(2)], [X]),
        contract=Returns(1),
        verified_by=(
            "x² = 0 ⟺ x = 0. The root has multiplicity two, but solve returns a set and has no "
            "multiplicity channel, so two entries is a wrong count, not an annotation."
        ),
    ),
    Case(
        id="solve_control_distinct_roots_are_two_solutions",
        subsystem="solving",
        statement="x² = 1 has two distinct solutions",
        op=solution_count([X ** _int(2) - _int(1)], [X]),
        contract=Returns(2),
        verified_by=(
            "x = ±1, and 1 ≠ -1. The control for solve_repeated_root_is_one_solution: "
            "de-duplicating on a tolerance that is too loose collapses these two as well."
        ),
    ),
    Case(
        id="solve_repeated_roots_do_not_multiply_across_variables",
        subsystem="solving",
        statement="x² = y² = z² = 0 has the single solution (0,0,0)",
        op=solution_count(
            [X ** _int(2), Y ** _int(2), POOL.symbol("z") ** _int(2)],
            [X, Y, POOL.symbol("z")],
        ),
        contract=Returns(1),
        verified_by=(
            "Each equation forces its variable to 0, so the variety is the single point "
            "(0,0,0). A per-variable duplicate multiplies out: 2³ = 8 copies of the origin, "
            "and 'this system has eight solutions' is a false lemma of exactly the shape a "
            "combinatorial search makes."
        ),
    ),
    Case(
        id="solve_control_distinct_roots_do_multiply",
        subsystem="solving",
        statement="x² = 1 ∧ y² = 1 has four solutions",
        op=solution_count([X ** _int(2) - _int(1), Y ** _int(2) - _int(1)], [X, Y]),
        contract=Returns(4),
        verified_by=(
            "The variety is {±1} × {±1}, four points. The control for "
            "solve_repeated_roots_do_not_multiply_across_variables: a solver that collapsed "
            "every product of branches to one point would otherwise pass it."
        ),
    ),
    Case(
        id="solve_undefined_coordinate_is_not_a_solution",
        subsystem="solving",
        statement="xy - y = 0 ∧ y - 2x² = 0 has two solutions, and neither coordinate is 0·0⁻¹",
        op=numeric_solution_count([X * Y - Y, Y - _int(2) * X ** _int(2)], [X, Y]),
        contract=Returns(2),
        verified_by=(
            "y(x-1) = 0 forces y = 0 or x = 1. y = 0 gives 2x² = 0, so (0,0); x = 1 gives "
            "y = 2, so (1,2). Two points. alkahest listed (0·0⁻¹, 0) — which denotes no number "
            "at all — in place of (0,0), so only one of its two entries named a point. "
            "solve_control_circle_meets_line_twice is the control: it fails the moment a "
            "solver answers with fewer points than a two-variable system has."
        ),
    ),
    Case(
        id="solve_homotopy_sparse_system_is_not_empty",
        subsystem="solving",
        statement="x³ = x ∧ y = x has three real solutions; homotopy must not report none",
        op=solution_count([X ** _int(3) - X, Y - X], [X, Y], method="homotopy"),
        contract=Returns(3),
        verified_by=(
            "x³ - x = x(x-1)(x+1), so x ∈ {-1, 0, 1} and y = x: the points (-1,-1), (0,0), "
            "(1,1). All three are non-singular (det J = 3x² - 1 ∈ {-1, 2}), so a continuation "
            "method has no excuse. An empty list is a claim that the system has no solutions."
        ),
    ),
    Case(
        id="solve_homotopy_bkk_deficient_system",
        subsystem="solving",
        statement="x²y = 1 ∧ xy² = 2 has one real solution (2^{-1/3}, 2^{2/3})",
        op=solution_count(
            [X ** _int(2) * Y - _int(1), X * Y ** _int(2) - _int(2)], [X, Y], method="homotopy"
        ),
        contract=Returns(1),
        verified_by=(
            "Multiplying the two equations gives (xy)³ = 2, so xy = 2^{1/3} over ℝ; dividing "
            "the second by the first gives y = 2x. Hence 2x² = 2^{1/3}, x = 2^{-1/3}, "
            "y = 2^{2/3}. Mixed volume 3 against a Bézout bound of 9 puts this system on the "
            "polyhedral branch, which supplied no continuation paths at all."
        ),
    ),
    Case(
        id="solve_control_homotopy_no_real_solutions",
        subsystem="solving",
        statement="x² = -1 ∧ y = x has no real solutions; homotopy must still say so",
        op=solution_count([X ** _int(2) + _int(1), Y - X], [X, Y], method="homotopy"),
        contract=Returns(0),
        verified_by=(
            "x² ≥ 0 on ℝ. The control for the two homotopy cases above: the fix for an empty "
            "list must not be to invent endpoints, and 'no real solutions' has to stay "
            "expressible."
        ),
    ),
    Case(
        id="solve_zero_polynomial",
        subsystem="solving",
        statement="the zero polynomial has infinitely many roots — no finite root list is honest",
        op=lambda: len(ak.real_roots(_int(0), X)),
        contract=Raises("E-ROOT-002"),
        verified_by="Every real number is a root of 0.",
    ),
    # ── root isolation ──────────────────────────────────────────────────────
    #
    # `real_roots` is load-bearing under `decide`, `solve` and the integrator's
    # own interior-pole detector, so a dropped root is inherited everywhere.
    Case(
        id="real_roots_three_rational_roots_kept",
        subsystem="solving",
        statement="25x³ - 325x² + 804x - 540 = 25(x - 6/5)(x - 9/5)(x - 10) has three real roots",
        op=_real_root_count([-540, 804, -325, 25]),
        contract=Returns(3),
        verified_by=(
            "Expanding 25(x - 6/5)(x - 9/5)(x - 10) gives the stated coefficients, and exact "
            "rational evaluation confirms p(6/5) = p(9/5) = p(10) = 0. alkahest reported only "
            "x = 10: the continued-fraction lower bound assumed 'p(k) has the sign of p(0) ⇒ no "
            "root below k', which is false when the count below k is even."
        ),
    ),
    Case(
        id="real_roots_chebyshev_t6_all_six",
        subsystem="solving",
        statement="the Chebyshev polynomial T₆ = 32x⁶ - 48x⁴ + 18x² - 1 has six real roots",
        op=_real_root_count([-1, 0, 18, 0, -48, 0, 32]),
        contract=Returns(6),
        verified_by=(
            "T₆(cos θ) = cos 6θ, so the roots are cos((2k+1)π/12) for k = 0..5 — six distinct "
            "values in (-1, 1). alkahest reported two."
        ),
    ),
    Case(
        id="refine_root_ball_brackets_sqrt_two",
        subsystem="solving",
        statement="refine_root's ball for x² - 2 must actually contain √2",
        op=_refined_ball_brackets_root([-2, 0, 1], 1),
        contract=Returns(True),
        verified_by=(
            "Checked in exact Fraction arithmetic on the ball's own endpoints: x² - 2 must vanish "
            "at one of them or change sign across them. alkahest returned mid = 1.414213562373095, "
            "rad = 1.11e-16, for which (mid + rad)² - 2 = -4.06e-17 < 0 — the entire ball lies "
            "strictly below √2, so it does not contain the root it claims to enclose."
        ),
    ),
    Case(
        id="refine_root_ball_brackets_large_coefficients",
        subsystem="solving",
        statement=(
            "refine_root must not report a zero-radius ball at a non-root of "
            "10⁹x³ - 1414213562x² - 2·10⁹x + 2828427124"
        ),
        op=_refined_ball_brackets_root([2828427124, -2000000000, -1414213562, 1000000000], 2),
        contract=Returns(True),
        verified_by=(
            "The polynomial is (10⁹x - 1414213562)(x² - 2), so the third bracket isolates √2. "
            "alkahest returned an *exact* (radius-0) ball at 1.4142135620573204, where the "
            "polynomial is -5.12e-11 ≠ 0 in exact arithmetic: the f64 Horner sign test is "
            "unreliable at these coefficient sizes and the bracket collapsed onto its endpoint."
        ),
    ),
    Case(
        id="solve_spurious_solution_two_by_two",
        subsystem="solving",
        statement="solve([x²-xy, xy-y]) must not report (-1, 1), which satisfies neither equation",
        op=lambda: max(
            abs(float(ak.eval_expr(eq, {X: _num(sol[X]), Y: _num(sol[Y])})))
            for sol in ak.solve([X ** _int(2) - X * Y, X * Y - Y], [X, Y])
            for eq in (X ** _int(2) - X * Y, X * Y - Y)
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "xy - y = y(x-1) = 0 forces y = 0 or x = 1; y = 0 gives x² = 0 so (0,0), and x = 1 "
            "gives 1 - y = 0 so (1,1). The solution set is {(0,0), (1,1)}. Substituting alkahest's "
            "third answer (-1, 1) gives x² - xy = 1 + 1 = 2 ≠ 0 — self-certifying, no oracle."
        ),
    ),
    # Elimination: the subresultant chain must *be* the subresultants.
    #
    # `subresultant_prs` and `resultant` disagreeing on the same input is its
    # own proof that one of them is wrong, and no oracle settles it — SymPy's
    # `resultant` is itself wrong for odd×odd degrees (3.8-silent-error-hunt-2,
    # finding 12), so every expectation below comes from the Sylvester
    # determinants directly.
    # -----------------------------------------------------------------------
    Case(
        id="subresultant_chain_ends_at_the_resultant",
        subsystem="solving",
        statement="the last element of the subresultant PRS of x²-3x+2 and 2x is Res = 8",
        op=_subresultant_chain([2, -3, 1], [0, 2]),
        contract=Returns((8.0, 8.0)),
        verified_by=(
            "The Sylvester matrix of x²-3x+2 and 2x is [[1,-3,2],[2,0,0],[0,2,0]]; expanding "
            "along the second row gives -(2)·det[[-3,2],[2,0]] = -(2)·(-4) = 8. Equivalently "
            "Res(f, 2x) = 2²·f(0) = 4·2 = 8 by the product formula. alkahest's own resultant() "
            "says 8 while subresultant_prs said 4 — two answers in one library that cannot both "
            "be right."
        ),
    ),
    Case(
        id="subresultant_chain_defective_case_is_the_subresultants",
        subsystem="solving",
        statement="the chain of 3x³-x and -3x²+2x-3 is S₁ = -24x-18, S₀ = -396",
        op=_subresultant_chain([0, -1, 0, 3], [-3, 2, -3]),
        contract=Returns((-66.0, -90.0, -396.0, -396.0)),
        verified_by=(
            "By hand from the recurrence with the canonical pseudo-division exponent δ+1 = 2: "
            "9·(3x³-x) mod (-3x²+2x-3) = -24x-18 and β₁ = (-1)^{δ+1} = 1, so S₁ = -24x-18, "
            "giving S₁(2) = -66 and S₁(3) = -90. One more step: 576·(-3x²+2x-3) mod (-24x-18) "
            "= -3564 and β₂ = 9, so S₀ = -396 — which is also the 5×5 Sylvester determinant and "
            "what resultant() reports. alkahest returned 8x+6 and -44, i.e. S₁/(-3) and S₀/9, "
            "because FLINT's pseudo-division uses the *minimal* exponent d and the recurrence "
            "assumed δ+1."
        ),
    ),
    Case(
        id="subresultant_chain_equal_degrees_terminates",
        subsystem="solving",
        statement="the chain of 2x²+2x+1 and 2x²+x+1 is S₁ = -2x, S₀ = Res = 2",
        op=_subresultant_chain([1, 2, 2], [1, 1, 2]),
        contract=Returns((-4.0, -6.0, 2.0, 2.0)),
        verified_by=(
            "g - f = -x exactly (the leading coefficients match), so g mod f = -x with quotient "
            "1, and Res(f,g) = lc(f)^{deg g - deg(g mod f)}·Res(f, -x) = 2·((-1)²·f(0)) = 2·1 = 2. "
            "The first pseudo-remainder is 2f mod g = 2x and β₁ = (-1)^{δ+1} = -1 with δ = 0, so "
            "S₁ = -2x, giving S₁(2) = -4 and S₁(3) = -6."
        ),
        note=(
            "Pre-fix this was not a wrong answer: the missing scale factor made the β division "
            "inexact, and FLINT's scalar_divexact calls flint_abort — SIGABRT, uncatchable by "
            "any Python handler, the whole process gone. A regression therefore takes the gate "
            "down rather than reporting; the Rust unit test "
            "poly::resultant::tests::sprs_survives_an_inexact_scaling_input is the primary guard."
        ),
    ),
    Case(
        id="subresultant_control_monic_divisor",
        subsystem="solving",
        statement="the chain of x³+x+1 and x²+1 is the single constant S₀ = Res = 1",
        op=_subresultant_chain([1, 1, 0, 1], [1, 0, 1]),
        contract=Returns((1.0, 1.0)),
        verified_by=(
            "x²+1 has roots ±i, and Res(f,g) = lc(g)^{deg f}·Π_{g(β)=0} f(β) = 1·f(i)·f(-i) = "
            "(i³+i+1)(-i³-i+1) = (1)(1) = 1 since i³ = -i. lc(g) = 1, so the pseudo-division "
            "scaling this fix corrects is trivial here and the answer was already right — a fix "
            "that merely refused, or that rescaled everything, would break this case."
        ),
    ),
    Case(
        id="subresultant_control_two_step_chain",
        subsystem="solving",
        statement="the chain of x⁴-1 and x²+x+1 is S₁ = -x+1, S₀ = Res = 3",
        op=_subresultant_chain([-1, 0, 0, 0, 1], [1, 1, 1]),
        contract=Returns((-1.0, -2.0, 3.0, 3.0)),
        verified_by=(
            "x²+x+1 has the primitive cube roots of unity ω, ω̄ as roots, and "
            "Res(f,g) = lc(g)^{deg f}·f(ω)f(ω̄) = (ω⁴-1)(ω̄⁴-1) = (ω-1)(ω̄-1) = "
            "1 - (ω+ω̄) + 1 = 1+1+1 = 3. A two-element chain with a monic divisor: correct "
            "before the fix as well, so it holds the fix to changing only what was broken."
        ),
    ),
    # Ideal theory: radicals, associated primes, triangular decomposition.
    #
    # The shape of the failure these guard against is a routine that cannot
    # compute the answer returning its *input* instead — √I = I asserted with
    # nothing behind it, or the ideal itself reported as a primary component.
    # That is worse than an ordinary wrong number, because the caller reads a
    # field named `associated_prime` and reasonably takes the name as a
    # guarantee.
    # -----------------------------------------------------------------------
    Case(
        id="ideal_radical_of_a_square_contains_its_base",
        subsystem="solving",
        statement="√⟨(x−y)²⟩ = ⟨x−y⟩, so the radical contains x−y as well as (x−y)²",
        op=_radical_membership([(X - Y) ** 2], [X, Y], [(X - Y) ** 2, X - Y, Y]),
        contract=Returns((True, True, False)),
        verified_by=(
            "ℚ[x,y]/(x−y) ≅ ℚ[y] is an integral domain, so ⟨x−y⟩ is prime; it contains "
            "(x−y)², hence √⟨(x−y)²⟩ ⊆ ⟨x−y⟩, and (x−y)² ∈ ⟨(x−y)²⟩ gives the reverse "
            "containment — the radical is exactly ⟨x−y⟩. y ∉ ⟨x−y⟩ because every element "
            "of ⟨x−y⟩ vanishes on the line x = y and y does not. No oracle: the answer "
            "`contains((x−y)²)=True, contains(x−y)=False` is refuted by the definition of "
            "a radical on its own."
        ),
    ),
    Case(
        id="ideal_associated_prime_of_a_difference_of_squares_is_prime",
        subsystem="solving",
        statement="every associated prime of ⟨x²−y²⟩ must be prime: it holds (x−y)(x+y)",
        op=_associated_primes_survive_a_witness([X**2 - Y**2], [X, Y], [(X - Y, X + Y)]),
        contract=Returns(True),
        verified_by=(
            "Definition of a prime ideal: ab ∈ P ⇒ a ∈ P or b ∈ P. Here ab = x²−y² lies in "
            "every component of a decomposition of ⟨x²−y²⟩, so a component holding neither "
            "x−y nor x+y is not prime. ⟨x²−y²⟩ itself is the failing case: x−y ∉ ⟨x²−y²⟩ by "
            "degree, and x+y ∉ ⟨x²−y²⟩ likewise. The witness is checked with the library's "
            "own membership test, so nothing outside alkahest is consulted."
        ),
    ),
    Case(
        id="ideal_primary_decomposition_of_a_difference_of_squares",
        subsystem="solving",
        statement="⟨x²−y²⟩ = ⟨x−y⟩ ∩ ⟨x+y⟩ — two components, and ⟨x²−y²⟩ is not primary",
        op=_component_count([X**2 - Y**2], [X, Y]),
        contract=Returns(2),
        verified_by=(
            "x−y and x+y are non-associate irreducibles of the UFD ℚ[x,y], so their "
            "generated ideals are prime and coprime, and ⟨x−y⟩ ∩ ⟨x+y⟩ = ⟨(x−y)(x+y)⟩ = "
            "⟨x²−y²⟩. Two prime components, neither redundant since neither contains the "
            "other. ⟨x²−y²⟩ on its own is not primary: (x−y)(x+y) ∈ I, x−y ∉ I, and no "
            "power of x+y is divisible by x²−y² because x−y is irreducible and not "
            "associate to x+y."
        ),
    ),
    Case(
        id="ideal_squarefree_monomial_decomposition_is_irredundant",
        subsystem="solving",
        statement="⟨xz, yz⟩ = ⟨z⟩ ∩ ⟨x,y⟩ — a radical ideal has exactly its minimal primes",
        op=_component_count([X * _Z, Y * _Z], [X, Y, _Z]),
        contract=Returns(2),
        verified_by=(
            "A monomial ideal generated by square-free monomials is radical, so its "
            "associated primes are exactly its minimal primes. V(xz, yz) = V(z) ∪ V(x,y), "
            "and ⟨z⟩ ∩ ⟨x,y⟩ = ⟨xz, yz⟩ by the coprime split ⟨J, uv⟩ = ⟨J,u⟩ ∩ ⟨J,v⟩ applied "
            "twice. A third component ⟨x,z⟩ is provably redundant because it contains ⟨z⟩, "
            "so intersecting with it changes nothing."
        ),
    ),
    Case(
        id="solve_triangularize_keeps_both_generators_of_a_two_point_ideal",
        subsystem="solving",
        statement="triangularize([x−y−1, y²−2]) must return chains of two polynomials",
        op=_shortest_chain_length([X - Y - 1, Y**2 - 2], [X, Y]),
        contract=Returns(2),
        verified_by=(
            "{x−y−1, y²−2} is already a reduced lex Gröbner basis, and its variety is the "
            "two points (1±√2, ±√2) — a zero-dimensional set. A triangular set cutting out "
            "a finite set in two variables needs one polynomial per variable: a single "
            "non-constant polynomial in x and y cuts out a curve, so a one-polynomial chain "
            "cannot describe two points whichever generator was kept."
        ),
    ),
    # Controls: the ideal routines must still *answer* where the mathematics is
    # within reach, so a library that refused every ideal question could not
    # pass the four traps above by attrition.
    Case(
        id="ideal_control_radical_of_a_monomial_ideal",
        subsystem="solving",
        statement="√⟨x², xy⟩ = ⟨x⟩",
        op=_radical_membership([X**2, X * Y], [X, Y], [X, X * Y, Y]),
        contract=Returns((True, True, False)),
        verified_by=(
            "x² and xy both lie in ⟨x⟩, and ⟨x⟩ is prime (ℚ[x,y]/(x) ≅ ℚ[y] is a domain), "
            "so √⟨x², xy⟩ ⊆ ⟨x⟩; x² ∈ ⟨x², xy⟩ gives x ∈ √I, so the two are equal. "
            "y ∉ ⟨x⟩ because y does not vanish on the line x = 0."
        ),
    ),
    Case(
        id="ideal_control_radical_of_a_zero_dimensional_ideal",
        subsystem="solving",
        statement="√⟨x²+y², xy⟩ = ⟨x,y⟩",
        op=_radical_membership([X**2 + Y**2, X * Y], [X, Y], [X, Y]),
        contract=Returns((True, True)),
        verified_by=(
            "y(x²+y²) − x(xy) = y³ and x(x²+y²) − y(xy) = x³ are both in I, so x and y lie "
            "in √I; and I ⊆ ⟨x,y⟩ since every generator has zero constant term, so "
            "√I ⊆ √⟨x,y⟩ = ⟨x,y⟩ (⟨x,y⟩ is maximal, hence prime). The control for the "
            "radical traps: this ideal is neither monomial nor principal, so a fix that "
            "simply stopped answering outside those two classes would fail here."
        ),
    ),
    Case(
        id="ideal_control_primary_decomposition_of_two_points",
        subsystem="solving",
        statement="⟨x²−1, y⟩ = ⟨x−1, y⟩ ∩ ⟨x+1, y⟩ — two maximal components",
        op=_component_count([X**2 - 1, Y], [X, Y]),
        contract=Returns(2),
        verified_by=(
            "V(x²−1, y) = {(1,0), (−1,0)}, two distinct rational points, and the ideal is "
            "radical because x²−1 is square-free — so it is the intersection of the two "
            "maximal ideals of those points. The control for the decomposition traps: a "
            "library that refused every primary decomposition would fail here."
        ),
    ),
    Case(
        id="solve_control_triangularize_a_linear_system",
        subsystem="solving",
        statement="triangularize([x+y−1, x−y]) returns a chain of two polynomials",
        op=_shortest_chain_length([X + Y - 1, X - Y], [X, Y]),
        contract=Returns(2),
        verified_by=(
            "The system has the single solution (½, ½); its reduced lex basis is "
            "{x − ½, y − ½}, already triangular with one polynomial per variable. The "
            "control for the triangularize trap: refusing every system would fail here."
        ),
    ),
    # Hypotheses and bounds that were reached silently (3.8 pre-release sweep).
    #
    # Neither of these returns a *false* number: the parametric solution is
    # right for almost every parameter value, and the unexpanded power is equal
    # to its input.  Both are still answers that claim more than was done — the
    # shape this corpus exists to catch, one step earlier than a wrong number.
    # -----------------------------------------------------------------------
    Case(
        id="solve_parametric_division_states_its_hypothesis",
        subsystem="solving",
        statement="solve([a·x − b], [x]) = b/a holds only for a ≠ 0, and must say so",
        op=_undisclosed_solve_hypotheses(
            [_A * X - _B], [X], {_A: 0.0, _B: 1.0}, hypothesis_about=_A
        ),
        contract=Returns(0.0),
        verified_by=(
            "By hand from the definition: a·x = b has the unique solution b/a when a ≠ 0. "
            "At a = 0 the equation reads 0·x − b = 0, i.e. −b = 0, so for b ≠ 0 there is no x "
            "at all and for b = 0 every x is a solution — neither is b/a, which is not defined "
            "there. The returned tuple is parametric, so it is never substituted back and "
            "carries no verification of its own; the hypothesis is the only auditable signal."
        ),
        note=(
            "Scored on disclosure, like the zeilberger boundary case: stating a ≠ 0 in "
            "solve_side_conditions() scores 0, and so would refusing to return b/a at all."
        ),
    ),
    Case(
        id="solve_control_provable_divisor_states_nothing",
        subsystem="solving",
        statement="solve([2x − b], [x]) = b/2 needs no hypothesis — and must state none",
        op=_solve_states_no_unnecessary_hypothesis([_int(2) * X - _B], [X], {_B: 6.0}, 3.0),
        contract=Returns(0.0),
        verified_by=(
            "2x = b has the solution b/2 for every b: the divisor is the literal 2, which is "
            "non-zero by inspection, so no side condition is needed. At b = 6 the solution is 3. "
            "The control for the case above — a library that emits a hypothesis unconditionally "
            "would pass that one and fail this."
        ),
    ),
    # Spurious roots from clearing a denominator.
    #
    # `N/D = 0` means `N = 0 and D != 0`.  Multiplying up drops the second
    # conjunct, and the root it leaves behind is not a near-miss — it is the one
    # point where the equation has no value at all.  Every one of these has a
    # confident wrong answer available to a solver that stops after clearing.
    # -----------------------------------------------------------------------
    Case(
        id="solve_rational_does_not_return_a_root_at_a_pole",
        subsystem="solving",
        statement="x/(x−1) = 1/(x−1) has no solution; x = 1 is a pole of both sides",
        op=_solve_returns_a_pole_as_a_root(X / (X - _int(1)) - _int(1) / (X - _int(1)), X, 1.0),
        contract=RefusesOr(0.0),
        verified_by=(
            "By hand: for x ≠ 1 the equation is x/(x−1) = 1/(x−1), i.e. x = 1 after "
            "multiplying by the non-zero (x−1) — which contradicts x ≠ 1. At x = 1 neither "
            "side is defined. So the solution set is empty. SymPy's solve returns [] for "
            "the same input. Clearing denominators gives (x−1)² = 0 whose root is 1, so a "
            "solver that stops there reports the one point that is excluded."
        ),
        note="Refusing to accept the rational form at all also scores 0 — that was the "
        "behaviour before rational equations were supported.",
    ),
    Case(
        id="solve_reciprocal_equals_zero_has_no_solution",
        subsystem="solving",
        statement="1/x = 0 has no solution — not x = 0, and not x = ∞",
        op=_solve_root_count(_int(1) / X, X),
        contract=RefusesOr(0.0),
        verified_by=(
            "By hand: 1/x = 0 would need 1 = 0 after multiplying by x, which is false for "
            "every x; and x = 0 is not in the domain. A reciprocal is never zero. The trap "
            "is a solver that cancels x against the numerator and reports x = 0."
        ),
    ),
    Case(
        id="solve_removable_singularity_is_not_a_solution",
        subsystem="solving",
        statement="x²/x = 0 has no solution: at x = 0 the expression is 0/0",
        op=_solve_root_count(X * X / X, X),
        contract=RefusesOr(0.0),
        verified_by=(
            "By hand: for x ≠ 0 the expression equals x, which is non-zero there; at x = 0 "
            "it is 0/0 and has no value. So no x satisfies it. This is the case that "
            "reducing to lowest terms gets wrong — cancelling gives x = 0, a point the "
            "original expression is not defined at, so the cancelled form must not be the "
            "one the exclusion test is run against."
        ),
    ),
    Case(
        id="solve_nested_reciprocal_keeps_the_inner_domain_condition",
        subsystem="solving",
        statement="1/(1/x − 1) = 0 has no solution: 1/x is undefined at x = 0",
        op=_solve_root_count(_int(1) / (_int(1) / X - _int(1)), X),
        contract=RefusesOr(0.0),
        verified_by=(
            "By hand: the expression is defined only for x ∉ {0, 1}, and there it equals "
            "x/(1−x), which is zero only at x = 0 — a point outside its domain. So the "
            "solution set is empty. SymPy's solve returns [] for the same input. The trap is "
            "specific: a reciprocal swaps numerator and denominator, so the inner denominator "
            "x becomes the outer numerator and the condition x ≠ 0 vanishes from any "
            "product-of-denominators bookkeeping. What is left, 1 − x, is non-zero at x = 0, "
            "so the cleared numerator's only root passes every check that is not the domain."
        ),
    ),
    Case(
        id="solve_control_rational_equation_keeps_its_real_roots",
        subsystem="solving",
        statement="(x²−4)/(x−1) = 0 has the roots ±2; excluding poles must not exclude them",
        op=_solve_finds_all_of((X * X - _int(4)) / (X - _int(1)), X, (-2.0, 2.0)),
        contract=Returns(2.0),
        verified_by=(
            "By hand: the numerator vanishes at x = ±2 and the denominator is 1 and −3 "
            "there, so both are genuine solutions; the pole at x = 1 is not a root of the "
            "numerator and never was a candidate. The control for the three cases above — "
            "a library that refuses every rational equation, or drops every root it cannot "
            "prove non-singular, passes those and fails this."
        ),
    ),
]
