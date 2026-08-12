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
from fractions import Fraction
from typing import Any, Callable

import alkahest as ak
import alkahest.experimental as ex
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
#: Second variable, for the two-variable `decide` cases.
Y = POOL.symbol("y")
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


def universal_holds(poly: ak.Expr, kind: str) -> Callable[[], bool]:
    """Answer = ``decide``'s verdict on ``forall x. poly <kind> 0``."""

    def op() -> bool:
        rel = {"ge": POOL.ge, "le": POOL.le, "gt": POOL.gt, "lt": POOL.lt}[kind]
        truth, _witness = ak.decide(ak.Forall(X, rel(poly, _int(0))))
        return truth

    return op


def _witness_residual(sentence: ak.Expr, body: ak.Expr) -> float:
    """Answer = |body(witness)| for the witness ``decide`` returns.

    A witness is a certificate, and the only thing a certificate means is that
    substituting it back works.  Scoring the *residual* rather than the witness's
    value keeps the case independent of which of several solutions is reported.
    A missing witness is scored as a refusal, not as zero.
    """
    _truth, witness = ak.decide(sentence)
    if not witness:
        # No `code=` kwarg: `CadError.__init__` does not take one, and passing it
        # raised `TypeError`, which the runner scores `no_answer` (a corpus bug)
        # instead of the intended honest refusal.
        raise ak.CadError("decide reported no witness (E-CAD-001)")
    value = Fraction(witness[str(X)])
    return abs(float(ak.eval_expr(body, {X: float(value)})))


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

#: ``det = mystery(a)``, so whether this matrix is invertible is exactly as
#: undecidable as whether ``mystery`` is the zero function.  ``rank()`` refuses
#: it; ``nullspace()`` used to return the 1-dimensional basis ``(-1, mystery(a))``
#: — the answer that is right only when ``det = 0``.
UNDECIDABLE_DETERMINANT = ak.Matrix(
    [
        [POOL.func("mystery", [_A]), _int(1)],
        [_int(0), _int(1)],
    ]
)

#: ``det = x``: generically non-zero, so the kernel is trivial.  This needs no
#: uninterpreted function at all — it is an ordinary symbolic matrix, and the
#: cheapest possible trigger for the same defect.
GENERICALLY_INVERTIBLE = ak.Matrix([[X, _int(0)], [_int(0), _int(1)]])

#: ``det = x·x − x·x = 0`` identically: genuinely rank 1, so the kernel really is
#: 1-dimensional.  The control that stops the gate being passed by refusing every
#: symbolic matrix.
GENUINELY_RANK_ONE = ak.Matrix([[X, X], [X, X]])


def _nullspace_dim(m: ak.Matrix) -> Callable[[], int]:
    """Answer = the dimension of ``m.nullspace()``."""
    return lambda: len(m.nullspace())


def _kernel_residual(m: ak.Matrix, at: float = 0.7) -> Callable[[], float]:
    """Answer = max |M·v| over the returned basis, sampled at ``x = at``.

    A basis vector that is not annihilated is the whole failure: the dimension
    can be right while the vector is wrong, so scoring the dimension alone would
    miss it.  Sampled numerically rather than compared structurally so the case
    does not depend on the form the entries come back in.
    """

    def op() -> float:
        worst = 0.0
        for v in m.nullspace():
            for row in (m @ v).to_list():
                for entry in row:
                    worst = max(worst, abs(float(ak.eval_expr(entry, {X: at, _A: at}))))
        return worst

    return op


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


# ---------------------------------------------------------------------------
# Round-two helpers (3.8 silent-error hunt #2)
# ---------------------------------------------------------------------------

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


def _enclosure_contains(expr: ak.Expr, lo: float, hi: float, truth: float) -> Callable[[], bool]:
    """Answer = does the *validated* enclosure of ``expr`` over the box contain
    the value it claims to enclose?"""

    def op() -> bool:
        enc = ak.bound_on_box(expr, [(X, lo, hi)])
        return bool(enc.lower <= truth <= enc.upper)

    return op


def _relation_residual(values: list[int]) -> Callable[[], int]:
    """Answer = the *exact* integer residual ``Σ aᵢ·valuesᵢ`` of the relation
    ``guess_relation`` reports.  Zero, or nothing at all, are the only honest
    answers; any other integer means the reported "relation" is not one."""

    def op() -> int:
        coeffs = ak.guess_relation(values)
        if coeffs is None:
            raise ak.PslqError("guess_relation reported no relation")
        return sum(a * v for a, v in zip(coeffs, values))

    return op


def _rsolve_residual(equation: ak.Expr, initials: dict[int, ak.Expr]) -> Callable[[], float]:
    """Answer = the worst residual of ``rsolve``'s answer *in the given equation*.

    Substituting the closed form back into the very equation that was passed in
    is self-certifying: it needs no oracle, and it is the one property a
    recurrence solver may never get wrong.  A solver that quietly re-indexes the
    equation returns the solution of a *different* problem, which is a clean,
    plausible, wrong sequence.
    """

    def op() -> float:
        closed = ak.rsolve(equation, N, "f", initials)
        residual = ak.simplify(_substitute_sequence(equation, closed)).value
        return max(abs(float(ak.eval_expr(residual, {N: float(j)}))) for j in range(6))

    return op


#: Shifts the recurrence cases are written with.
_SEQ_SHIFTS = (2, 1, 0, -1, -2)


def _seq(shift: int) -> ak.Expr:
    """``f(n + shift)`` — the term shape ``rsolve`` reads."""
    return POOL.func("f", [N if shift == 0 else N + _int(shift)])


def _substitute_sequence(equation: ak.Expr, closed: ak.Expr) -> ak.Expr:
    """``equation`` with every ``f(n + c)`` replaced by ``closed`` shifted by c.

    Written against the fixed shift set the recurrence cases are built from
    (:data:`_SEQ_SHIFTS`, via :func:`_seq`) rather than by walking the expression
    tree, so the substitution itself stays obviously correct.
    """
    out = equation
    for c in _SEQ_SHIFTS:
        arg = N if c == 0 else N + _int(c)
        shifted = closed if c == 0 else ak.subs(closed, {N: arg})
        out = ak.subs(out, {_seq(c): shifted})
    return out


def _basis_independence(equation: ak.Expr) -> Callable[[], bool]:
    """Answer = whether ``rsolve``'s *general* solution spans two dimensions.

    The general solution of a second-order linear recurrence is a two-parameter
    family.  Returning ``C₀·rⁿ + C₁·rⁿ`` for a repeated root looks like one but
    is not: both basis elements are the same function, so the family is
    one-dimensional and cannot meet two independent initial conditions.
    """

    def op() -> bool:
        general = ak.rsolve(equation, N, "f", None)
        c0 = POOL.symbol("C0")
        c1 = POOL.symbol("C1")
        rows = []
        for at in (0.0, 1.0):
            first = ak.eval_expr(general, {c0: 1.0, c1: 0.0, N: at})
            second = ak.eval_expr(general, {c0: 0.0, c1: 1.0, N: at})
            rows.append((float(first), float(second)))
        det = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        return abs(det) > 1e-9

    return op


def _constant_terms(report: Any) -> list[float]:
    """The values of every term of an asymptotic expansion that does not move.

    A term with the same value at ``n = 10`` and ``n = 20`` is a constant, and a
    constant claimed for a sum whose closed form is a polynomial with zero
    constant term is a fabricated one.
    """
    out = []
    for term in report.terms:
        lo = float(ak.eval_expr(term, {N: 10.0}))
        hi = float(ak.eval_expr(term, {N: 20.0}))
        if abs(lo - hi) <= 1e-9 * max(1.0, abs(lo)):
            out.append(lo)
    return out


def _binom(top: ak.Expr, bot: ak.Expr) -> ak.Expr:
    """``C(top, bot)`` as a Γ-quotient, the shape ``zeilberger`` parses."""
    return ak.gamma(top + _int(1)) / (ak.gamma(bot + _int(1)) * ak.gamma(top - bot + _int(1)))


def _zeilberger_sum_recurrence_defect(
    term: ak.Expr, exact_sum: Callable[[int], Fraction], disclosure_counts: bool
) -> Callable[[], float]:
    """Answer = how badly the *sum* recurrence read off the certificate fails.

    Zeilberger verifies ``Σ_i a_i(n)·F(n+i,k) = G(n,k+1) − G(n,k)``, an identity
    in ``k``.  Summing it gives ``Σ_i a_i(n)·S(n+i) = G(n,k_hi+1) − G(n,k_lo)``,
    so the familiar homogeneous recurrence needs that boundary difference to
    vanish — a hypothesis the algorithm does not establish.

    With *disclosure_counts* the case is satisfied either way an honest library
    can behave: prove the hypothesis (residual genuinely zero) or state it as a
    side condition on the certificate.  Silently omitting it scores the residual,
    which is what a caller who trusted the recurrence would inherit.
    """

    def op() -> float:
        cert = ak.zeilberger(term, N, K)
        if disclosure_counts:
            conditions = getattr(cert, "side_conditions", ())
            if any("boundary" in str(c).lower() for c in conditions):
                return 0.0
        worst = 0.0
        for ni in range(1, 6):
            total = Fraction(0)
            for i, a in enumerate(cert.coeffs):
                coeff = Fraction(float(ak.eval_expr(a, {N: float(ni)}))).limit_denominator(10**9)
                total += coeff * exact_sum(ni + i)
            worst = max(worst, abs(float(total)))
        return worst

    return op


def _sum_binomial_over_k_plus_one(m: int) -> Fraction:
    """``Σ_{k=0}^{m} C(m,k)/(k+1) = (2^{m+1} − 1)/(m+1)``, by hand."""
    return sum((Fraction(math.comb(m, j), j + 1) for j in range(m + 1)), Fraction(0))


def _sum_binomial_row(m: int) -> Fraction:
    """``Σ_{k=0}^{m} C(m,k) = 2^m``."""
    return Fraction(2**m)


def _survives_a_panic(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap *fn* so a Rust panic fails this case instead of killing the run.

    PyO3 turns an escaping Rust panic into ``pyo3_runtime.PanicException``,
    which inherits ``BaseException``.  That is the whole reason the class
    matters — a loop's ``except Exception`` does not catch it — but it also
    means an unwrapped op would take the gate process down with it and no case
    would be reported at all.  Re-raising as ``RuntimeError`` keeps the failure
    (scored ``no_answer``: neither an answer nor a refusal) while leaving the
    rest of the corpus scoreable.
    """

    def op() -> Any:
        try:
            return fn()
        except Exception:
            raise
        except BaseException as exc:  # PanicException is a BaseException — the point
            raise RuntimeError(
                f"escaping Rust panic: {type(exc).__module__}.{type(exc).__name__}: {exc}"
            ) from exc

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


def _lll_rows_stay_in_the_lattice(
    rows: list[list[int]], generator: list[int]
) -> Callable[[], bool]:
    """Answer = does LLL return a basis of the *same* lattice ``ℤ·generator``?

    Every returned row must be an integer multiple of *generator* (nothing left
    the lattice), the generator itself must still be reachable (nothing was
    lost), and the row count must be preserved.  Exact integer arithmetic, no
    reference implementation.
    """

    def op() -> bool:
        reduced = ak.lattice.lll_reduce_rows(rows)
        if len(reduced) != len(rows):
            return False
        multiples = []
        for row in reduced:
            ratios = {Fraction(v, g) for v, g in zip(row, generator) if g != 0}
            leftover = any(v != 0 for v, g in zip(row, generator) if g == 0)
            if leftover or len(ratios) != 1:
                return False
            (r,) = ratios
            if r.denominator != 1:
                return False
            multiples.append(abs(r.numerator))
        return 1 in multiples

    return _survives_a_panic(op)


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
    # The CAD sample set is built from isolating-bracket endpoints and their
    # midpoints, which are all *dyadic* rationals.  A statement whose truth turns
    # on the value at a root with any other denominator was therefore decided
    # without that point ever being tested.  x^2 > 0 above passes because 0 is
    # dyadic; these three are the same trap one denominator to the right.
    Case(
        id="decide_forall_square_touching_at_two_thirds",
        subsystem="real_qe",
        statement="forall x. (3x+2)^2 > 0 is FALSE (x = -2/3)",
        op=universal_holds((_int(3) * X + _int(2)) ** _int(2), "gt"),
        contract=Returns(False),
        verified_by=(
            "9x^2+12x+4 at x=-2/3 is 9(4/9) + 12(-2/3) + 4 = 4 - 8 + 4 = 0 exactly, and 0 > 0 "
            "is false. -2/3 has denominator 3, so no bisection of a rational bracket ever "
            "lands on it."
        ),
    ),
    Case(
        id="decide_forall_square_touching_at_one_fifth",
        subsystem="real_qe",
        statement="forall x. (5x-1)^2 > 0 is FALSE (x = 1/5)",
        op=universal_holds((_int(5) * X - _int(1)) ** _int(2), "gt"),
        contract=Returns(False),
        verified_by="25x^2-10x+1 at x=1/5 is 25/25 - 10/5 + 1 = 1 - 2 + 1 = 0; 0 > 0 is false.",
    ),
    Case(
        id="decide_exists_nonstrict_boundary_at_two_thirds",
        subsystem="real_qe",
        statement="exists x. (3x+2)^2 <= 0 is TRUE (x = -2/3)",
        op=lambda: ak.decide(ak.Exists(X, POOL.le((_int(3) * X + _int(2)) ** _int(2), _int(0))))[0],
        contract=Returns(True),
        verified_by=(
            "The square vanishes at x=-2/3 (see decide_forall_square_touching_at_two_thirds), "
            "and 0 <= 0 holds. The dual of the forall case: a missed existential witness is "
            "what makes the universal come back true."
        ),
    ),
    Case(
        id="decide_witness_satisfies_linear_equation",
        subsystem="real_qe",
        statement="the witness decide returns for exists x. 3x - 2 = 0 must satisfy it",
        op=lambda: _witness_residual(
            ak.Exists(X, POOL.pred_eq(_int(3) * X - _int(2), _int(0))),
            _int(3) * X - _int(2),
        ),
        contract=Returns(0.0, tol=1e-12),
        verified_by=(
            "3x = 2 has the single solution x = 2/3, and 3(2/3) - 2 = 0. A witness is a "
            "certificate: substituting it back is the whole of its meaning, so a witness "
            "with a non-zero residual is a wrong answer no matter what the truth value says."
        ),
    ),
    Case(
        id="decide_forall_square_touching_at_irrational_root",
        subsystem="real_qe",
        statement="forall x. (x^2-2)^2 > 0 is FALSE (x = ±sqrt(2)); no rational sample shows it",
        op=universal_holds((X ** _int(2) - _int(2)) ** _int(2), "gt"),
        contract=RefusesOr(False),
        verified_by=(
            "(x^2-2)^2 vanishes at x=±sqrt(2), where 0 > 0 is false. sqrt(2) is irrational, "
            "so a decision procedure that only evaluates at rational points cannot exhibit "
            "the counterexample — refusing is honest, returning True is a proof of a false "
            "theorem."
        ),
        note="Passes by refusal (E-CAD-001); deciding it needs algebraic-number CAD lifting.",
    ),
    # The same completeness gap, one variable up.  `project_and_sample_x` flags
    # an irrational projection root as untested, but the flag only escalated to
    # a refusal when the body contained an `=` / `≠` atom — so `≤` and `≥` in
    # two variables kept reporting an unsatisfiability that was never checked at
    # the one point that could have satisfied them.
    Case(
        id="decide_exists_exists_nonstrict_boundary_at_irrational_x",
        subsystem="real_qe",
        statement="exists x. exists y. (x^2-2)^2 + y^2 <= 0 is TRUE (at x = ±√2, y = 0)",
        op=lambda: ak.decide(
            ak.Exists(
                X,
                ak.Exists(Y, POOL.le((X ** _int(2) - _int(2)) ** _int(2) + Y ** _int(2), _int(0))),
            )
        )[0],
        contract=RefusesOr(True),
        verified_by=(
            "Both summands are squares, so the sum is >= 0 and equals 0 exactly when "
            "x^2 = 2 and y = 0, i.e. at (±√2, 0) — two real points. So the sentence is TRUE. "
            "√2 is irrational, so no rational sample point ever lands on it: a procedure "
            "that only evaluates at rationals must refuse, and a `False` is a claim that "
            "these two points do not exist."
        ),
        note="Passes by refusal (E-CAD-001); deciding it needs algebraic-number CAD lifting.",
    ),
    Case(
        id="decide_forall_forall_strict_positive_at_irrational_root",
        subsystem="real_qe",
        statement="forall x. forall y. (x^2-2)^2 + y^2 > 0 is FALSE (0 at x = ±√2, y = 0)",
        op=lambda: ak.decide(
            ak.Forall(
                X,
                ak.Forall(Y, POOL.gt((X ** _int(2) - _int(2)) ** _int(2) + Y ** _int(2), _int(0))),
            )
        )[0],
        contract=RefusesOr(False),
        verified_by=(
            "The negation of the case above: the sum vanishes at (√2, 0), where 0 > 0 is "
            "false, so the universal is FALSE. `∀x∀y φ` is decided as `¬∃x∃y ¬φ`, so a "
            "missed existential witness surfaces here as a proof of a false theorem — the "
            "shape of error a stability proof or a bound check would inherit whole."
        ),
        note="Passes by refusal (E-CAD-001); the dual of the exists/exists case.",
    ),
    Case(
        id="decide_exists_exists_nonstrict_boundary_at_two_thirds",
        subsystem="real_qe",
        statement="exists x. exists y. (3x-2)^2 + y^2 <= 0 is TRUE (at x = 2/3, y = 0)",
        op=lambda: ak.decide(
            ak.Exists(
                X, ak.Exists(Y, POOL.le((_int(3) * X - _int(2)) ** _int(2) + Y ** _int(2), _int(0)))
            )
        )[0],
        contract=Returns(True),
        verified_by=(
            "(3x-2)^2 + y^2 = 0 exactly at x = 2/3, y = 0: 3(2/3) - 2 = 0. The boundary point "
            "is rational here, so the CAD sample set can reach it and there is nothing to "
            "refuse. The control for the two irrational-root cases above: without it the "
            "gate would be passed by a `decide` that refuses every non-strict two-variable "
            "sentence."
        ),
    ),
    Case(
        id="decide_exists_exists_nonstrict_genuinely_unsatisfiable",
        subsystem="real_qe",
        statement="exists x. exists y. (x^2-2)^2 + y^2 + 1 <= 0 is FALSE (the sum is >= 1)",
        op=lambda: ak.decide(
            ak.Exists(
                X,
                ak.Exists(
                    Y,
                    POOL.le((X ** _int(2) - _int(2)) ** _int(2) + Y ** _int(2) + _int(1), _int(0)),
                ),
            )
        )[0],
        contract=Returns(False),
        verified_by=(
            "Two squares plus 1 is >= 1 > 0 everywhere, so nothing satisfies `<= 0` and the "
            "sentence is FALSE. Same polynomial shape and the same `<=` atom as the "
            "irrational-root case, so this is the control that the completeness guard "
            "refuses only where a boundary point is genuinely untested, rather than "
            "refusing every `<=` it sees."
        ),
    ),
    Case(
        id="decide_forall_forall_control_two_squares_plus_one",
        subsystem="real_qe",
        statement="forall x. forall y. x^2 + y^2 + 1 > 0 is TRUE",
        op=lambda: ak.decide(
            ak.Forall(X, ak.Forall(Y, POOL.gt(X ** _int(2) + Y ** _int(2) + _int(1), _int(0))))
        )[0],
        contract=Returns(True),
        verified_by=(
            "Squares are non-negative, so x^2 + y^2 + 1 >= 1 > 0 for every real (x, y). "
            "The positive control for the two-variable universal path."
        ),
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
    # -----------------------------------------------------------------------
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
    # Division by a literal zero.  `x · x^-1 → 1` and `x · 0 → 0` are both
    # deliberate conventions (see simplify_control_cancel_x_over_x), and both
    # are false when the base really is zero: `0 · 0^-1` is `0 · ∞`, which has
    # no value under any convention.  `simplify(0^-1)` already leaves the power
    # alone and `eval_expr(0^-1)` raises E-EVAL-009, so a product that quietly
    # collapses to a number is contradicting the rest of the library.
    # -----------------------------------------------------------------------
    Case(
        id="simplify_zero_times_zero_reciprocal",
        subsystem="simplification",
        statement="0 · 0^-1 is undefined — not 1, not 0",
        op=simplified_value(ak.simplify, _int(0) * _int(0) ** _int(-1)),
        contract=RefusesOr(),
        verified_by=(
            "0^-1 is division by zero, so the product has no value: it is the indeterminate "
            "form 0·∞. Summing the exponents to 0^0 = 1 is invalid precisely because the "
            "base is zero — b^k·b^m = b^(k+m) needs b ≠ 0 once one exponent is negative."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_zero_reciprocal_in_longer_product",
        subsystem="simplification",
        statement="5 · 0^-1 · 0 is undefined — the arrangement must not change the answer",
        op=simplified_value(ak.simplify, _int(5) * _int(0) ** _int(-1) * _int(0)),
        contract=RefusesOr(),
        verified_by=(
            "Same undefined product with a spectator factor: 5·(0·∞) is still indeterminate. "
            "This arrangement is folded by the numeric constant folder rather than by the "
            "exponent collector, so it is a second, independent route to the same lie — and "
            "it used to give 0 where the two-factor form gave 1, which is its own proof that "
            "at least one of them is wrong."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_symbolic_zero_times_its_reciprocal",
        subsystem="simplification",
        statement="(x-x) · (x-x)^-1 is undefined: the base is identically zero",
        op=simplified_value(ak.simplify, (X - X) * (X - X) ** _int(-1), at=2.0),
        contract=RefusesOr(),
        verified_by=(
            "x - x is the zero function, so (x-x)^-1 is nowhere defined and the product has "
            "no value at any x. Cancelling b·b^-1 → 1 asserts b ≠ 0, which is false here. "
            "This is the shape `diff(2/(x-x), x)` reaches, so it is not a hand-written "
            "curiosity."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_egraph_zero_times_zero_reciprocal",
        subsystem="simplification",
        statement="the e-graph simplifier must not give 0 · 0^-1 a value either",
        op=simplified_value(ak.simplify_egraph, _int(0) * _int(0) ** _int(-1)),
        contract=RefusesOr(),
        verified_by=(
            "Same undefined product; checked separately because the e-graph engine has its "
            "own rule set. It is the worse of the two failures: its shrink rules contain "
            "both (Mul ?x (Num 0)) → (Num 0) and (Mul ?x (Pow ?x (Num -1))) → (Num 1), so "
            "on this input it unions 0 and 1 into a single e-class — every other e-class in "
            "the run is then equally suspect."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="diff_reciprocal_of_identically_zero_denominator",
        subsystem="simplification",
        statement="d/dx [2/(x-x)] has no value: the function is nowhere defined",
        op=lambda: _num(ak.diff(_int(2) / (X - X), X)),
        contract=RefusesOr(),
        verified_by=(
            "2/(x-x) = 2/0 has empty domain, so it has no derivative anywhere; 1 is a value "
            "it can never take. Reached through an ordinary `diff` call, without writing "
            "0^-1 by hand: the quotient rule produces 0·0^-1 terms and the simplifier used "
            "to collapse them."
        ),
        note="Passes by a weak refusal: eval_expr raises E-EVAL-009 on the preserved 0^-1.",
    ),
    Case(
        id="simplify_control_symbol_over_symbol",
        subsystem="simplification",
        statement="simplify(x · x^-1) = 1 for a symbolic x",
        op=simplified_value(ak.simplify, X * X ** _int(-1), at=2.0),
        contract=Returns(1.0),
        verified_by=(
            "2 · (1/2) = 1. The documented convention for a base that is not provably zero, "
            "and the control that the zero-base guard did not simply switch factor "
            "collection off."
        ),
    ),
    Case(
        id="simplify_control_zero_times_symbol",
        subsystem="simplification",
        statement="simplify(0 · x) = 0",
        op=simplified_value(ak.simplify, _int(0) * X, at=3.0),
        contract=Returns(0.0),
        verified_by=(
            "0 · 3 = 0. The control for the absorption rule: it must keep firing on products "
            "that really are zero, and only decline when a co-factor is undefined."
        ),
    ),
    Case(
        id="simplify_control_like_terms_cancel_to_zero",
        subsystem="simplification",
        statement="simplify(2x - 2x) = 0",
        op=simplified_value(ak.simplify, _int(2) * X - _int(2) * X, at=5.0),
        contract=Returns(0.0),
        verified_by=(
            "10 - 10 = 0. The control for like-term collection, which must still drop terms "
            "whose coefficients cancel — the guard only applies when the surviving factor is "
            "a division by zero."
        ),
    ),
    Case(
        id="simplify_egraph_control_symbol_over_symbol",
        subsystem="simplification",
        statement="simplify_egraph(x · x^-1) = 1 for a symbolic x",
        op=simplified_value(ak.simplify_egraph, X * X ** _int(-1), at=2.0),
        contract=Returns(1.0),
        verified_by=(
            "2 · (1/2) = 1. The e-graph control: it must still cancel a symbolic base, so "
            "the zero-base bail-out cannot be passed by disabling the engine."
        ),
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
        id="matrix_nullspace_undecidable_determinant_refuses",
        subsystem="linear_algebra",
        statement="the nullspace of [[mystery(a), 1], [0, 1]] cannot be stated — its "
        "dimension is 0 or 1 depending on whether mystery(a) vanishes",
        op=_nullspace_dim(UNDECIDABLE_DETERMINANT),
        contract=RefusesOr(),
        verified_by="det = mystery(a)·1 − 1·0 = mystery(a). If mystery is not identically "
        "zero the matrix is invertible and the kernel is {0}; if it is, the kernel is "
        "1-dimensional. Both are consistent with everything alkahest knows about an "
        "uninterpreted function symbol, so neither dimension is derivable. The wrong "
        "answer alkahest gave was the basis v = (-1, mystery(a)): multiplying back, "
        "M·v = (mystery(a)·(-1) + 1·mystery(a), 0·(-1) + 1·mystery(a)) = (0, mystery(a)), "
        "which is the zero vector only when mystery(a) = 0 — precisely the thing that was "
        "never established. rank() already refuses this matrix, so the two calls also "
        "contradicted each other.",
        note="Shipped in 3.7: the 2x2 fast path's full-rank gate only recognised a "
        "*literal* non-zero determinant, so any symbolic determinant fell through into "
        "the rank-1 branch. That reads 'cannot prove det != 0' as 'det = 0' — the mirror "
        "of the rref defect that motivated the three-valued zero test, which read "
        "'cannot prove zero' as 'non-zero'.",
    ),
    Case(
        id="matrix_nullspace_generic_determinant_is_trivial",
        subsystem="linear_algebra",
        statement="the nullspace of [[x, 0], [0, 1]] is {0} — dimension 0",
        op=_nullspace_dim(GENERICALLY_INVERTIBLE),
        contract=Returns(0),
        verified_by="det = x·1 − 0·0 = x, which is not the zero function, so the matrix is "
        "invertible for all x != 0 and its kernel is trivial — the same generic-rank "
        "reading rank() uses when it reports 2. The wrong answer was the 1-dimensional "
        "basis v = (0, x), for which M·v = (x·0 + 0·x, 0·0 + 1·x) = (0, x) != 0. Needs no "
        "uninterpreted function: an ordinary symbolic matrix was enough, and rank 2 with "
        "nullity 1 makes 3 for a 2-column matrix, violating rank–nullity across two "
        "public calls.",
    ),
    Case(
        id="matrix_nullspace_singular_symbolic_still_answers",
        subsystem="linear_algebra",
        statement="the nullspace of [[x, x], [x, x]] is 1-dimensional",
        op=_nullspace_dim(GENUINELY_RANK_ONE),
        contract=Returns(1),
        verified_by="det = x·x − x·x = 0 identically, and the matrix is not the zero "
        "matrix, so it has rank 1 and by rank–nullity a 1-dimensional kernel, spanned by "
        "(1, -1). The control for the two cases above: a library that fixed them by "
        "refusing every symbolic matrix would pass both and fail this one.",
    ),
    Case(
        id="matrix_nullspace_basis_is_actually_annihilated",
        subsystem="linear_algebra",
        statement="every returned nullspace basis vector v of [[x, x], [x, x]] satisfies M·v = 0",
        op=_kernel_residual(GENUINELY_RANK_ONE),
        contract=Returns(0.0),
        verified_by="M·(1, -1) = (x − x, x − x) = (0, 0) for every x, so the residual is "
        "exactly zero; sampled at x = 0.7. Scoring the dimension alone would miss the "
        "actual failure mode, which was a basis of the right *size* whose vector was not "
        "in the kernel.",
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
    # The dropped rational scale in `product_definite`, and poles strictly
    # inside a summation range.  Both were reported in
    # `temp-alkahest/testing/3.8-silent-error-hunt-2.md` and fixed for 3.8.0.
    # -----------------------------------------------------------------------
    Case(
        id="product_definite_keeps_rational_scale",
        subsystem="sums_products",
        statement="Π_{k=1}^{5} 1/2 = 1/32",
        op=lambda: _num(ak.product_definite(_rat(1, 2), K, _int(1), _int(5))),
        contract=Returns(1.0 / 32.0, tol=1e-12),
        verified_by="Five factors of 1/2 multiply to 2^-5 = 1/32, by the definition of a product.",
    ),
    Case(
        id="product_definite_wallis_partial_product",
        subsystem="sums_products",
        statement="Π_{k=1}^{6} (2k-1)/(2k) = C(12,6)/4^6 = 924/4096",
        op=lambda: _num(
            ak.product_definite(
                (_int(2) * K - _int(1)) * (_int(2) * K) ** _int(-1), K, _int(1), _int(6)
            )
        ),
        contract=Returns(924.0 / 4096.0, tol=1e-9),
        verified_by=(
            "1·3·5·7·9·11 / (2·4·6·8·10·12) = 10395/46080 = 924/4096 = 0.2255859375, multiplied "
            "out by hand; it is also the standard Π(2k-1)/(2k) = C(2n,n)/4ⁿ at n = 6. alkahest "
            "returned 14.4375, which is 2⁶ times too large — one factor of the denominator's "
            "leading coefficient per index, from the scale ratuni_poly_to_univ discarded."
        ),
    ),
    Case(
        id="product_definite_empty_range_of_a_zero_term",
        subsystem="sums_products",
        statement="Π_{k=1}^{0} 0 = 1 — an empty product takes no factors at all",
        op=lambda: _num(ak.product_definite(_int(0), K, _int(1), _int(0))),
        contract=Returns(1.0),
        verified_by=(
            "The empty product is 1 by universal convention, whatever the term is: no factor is "
            "ever taken. alkahest returned 0 here while returning 1 for Π_{k=1}^{0} k, so its own "
            "two answers for the same empty range disagreed — the zero-numerator shortcut ran "
            "before the empty-range check."
        ),
    ),
    Case(
        id="product_control_integer_coefficient_ratio",
        subsystem="sums_products",
        statement="Π_{k=1}^{4} (k+1)/k = 5 — telescoping, no denominators to clear",
        op=lambda: _num(ak.product_definite((K + _int(1)) * K ** _int(-1), K, _int(1), _int(4))),
        contract=Returns(5.0, tol=1e-9),
        verified_by=(
            "(2/1)(3/2)(4/3)(5/4) telescopes to 5/1 = 5. The control for the rational-scale "
            "cases: this one has monic numerator and denominator, so it was already correct "
            "before the fix and must stay correct after it — a product_definite that started "
            "refusing every rational term would not pass here."
        ),
    ),
    Case(
        id="sum_definite_interior_pole_refused",
        subsystem="sums_products",
        statement="Σ_{k=1}^{10} 1/((k-3)(k-2)) is undefined — the k=2 and k=3 terms divide by zero",
        op=lambda: _num(
            ak.sum_definite(((K - _int(3)) * (K - _int(2))) ** _int(-1), K, _int(1), _int(10))
        ),
        contract=RefusesOr(),
        verified_by=(
            "The k=2 term is 1/((-1)·0) and the k=3 term is 1/(0·1); neither is a number, so the "
            "sum has no value. alkahest returned -5/8. Its own docstring promises E-SUM-003 for "
            "exactly this."
        ),
    ),
    Case(
        id="sum_definite_interior_pole_negative_lower_bound",
        subsystem="sums_products",
        statement="Σ_{k=-2}^{5} 1/(k(k+1)) is undefined — the k=-1 and k=0 terms divide by zero",
        op=lambda: _num(ak.sum_definite((K * (K + _int(1))) ** _int(-1), K, _int(-2), _int(5))),
        contract=RefusesOr(),
        verified_by=(
            "1/(k(k+1)) at k = -1 is 1/((-1)·0) and at k = 0 is 1/(0·1); both terms of the sum "
            "are undefined, so the sum is. alkahest returned -2/3 — the telescoped difference "
            "G(6) - G(-2), which is a perfectly finite number and not the sum of anything."
        ),
    ),
    Case(
        id="sum_control_pole_below_the_range",
        subsystem="sums_products",
        statement="Σ_{k=4}^{10} 1/((k-3)(k-2)) = 1 - 1/8 = 7/8",
        op=lambda: _num(
            ak.sum_definite(((K - _int(3)) * (K - _int(2))) ** _int(-1), K, _int(4), _int(10))
        ),
        contract=Returns(0.875, tol=1e-12),
        verified_by=(
            "1/((k-3)(k-2)) = 1/(k-3) - 1/(k-2), so Σ_{k=4}^{10} telescopes to 1/1 - 1/8 = 7/8; "
            "adding the seven terms 1/2, 1/6, 1/12, 1/20, 1/30, 1/42, 1/56 by hand gives the "
            "same. The control for the interior-pole cases: the same integrand with both poles "
            "just below the range must still be summed, so refusing every 1/((k-a)(k-b)) does "
            "not pass the gate."
        ),
    ),
    # -----------------------------------------------------------------------
    # Recurrences.  A recurrence solver's one inviolable property is that its
    # answer satisfies the equation it was handed; checking that needs no
    # oracle at all.
    # -----------------------------------------------------------------------
    Case(
        id="rsolve_forward_shift_solves_its_own_equation",
        subsystem="sums_products",
        statement="rsolve(f(n+1) - f(n) - n², f(0)=0) must satisfy f(n+1) - f(n) = n²",
        op=_rsolve_residual(_seq(1) - _seq(0) - N ** _int(2), {0: _int(0)}),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Iterating the given equation from f(0) = 0 gives 0, 0, 1, 5, 14, 30, i.e. "
            "f(n) = Σ_{j=0}^{n-1} j² = n³/3 - n²/2 + n/6. alkahest returned n³/3 + n²/2 + n/6, "
            "whose values are 0, 1, 5, 14, 30 — the solution of f(n+1) - f(n) = (n+1)², a "
            "different equation. Substituting back into the equation supplied is self-certifying."
        ),
    ),
    Case(
        id="rsolve_control_lag_shift_spelling",
        subsystem="sums_products",
        statement="rsolve(f(n) - f(n-1) - n², f(0)=0) must satisfy f(n) - f(n-1) = n²",
        op=_rsolve_residual(_seq(0) - _seq(-1) - N ** _int(2), {0: _int(0)}),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Iterating from f(0) = 0 gives 0, 1, 5, 14, 30, 55 = Σ_{j=1}^{n} j². The control for "
            "rsolve_forward_shift_solves_its_own_equation: the lag spelling was always handled "
            "correctly, so a fix that simply started refusing shifted equations would fail here."
        ),
    ),
    Case(
        id="rsolve_order_two_repeated_root_spans_two_dimensions",
        subsystem="sums_products",
        statement="the general solution of f(n+2) - 4f(n+1) + 4f(n) = 0 is a two-parameter family",
        op=_basis_independence(_seq(2) - _int(4) * _seq(1) + _int(4) * _seq(0)),
        contract=Returns(True),
        verified_by=(
            "r² - 4r + 4 = (r-2)² has the double root 2, so the general solution is (A + Bn)·2ⁿ; "
            "(n+2)2ⁿ⁺² - 4(n+1)2ⁿ⁺¹ + 4n·2ⁿ = 2ⁿ(4n+8-8n-8+4n) = 0 verifies the second branch by "
            "hand. alkahest returned C₀·(½(4+√0))ⁿ + C₁·(½(4-√0))ⁿ — the same function twice, a "
            "one-parameter family presented as the general solution of a second-order equation, "
            "whose 2×2 initial-condition matrix is singular."
        ),
    ),
    Case(
        id="rsolve_control_order_two_distinct_roots",
        subsystem="sums_products",
        statement="the general solution of f(n+2) - 3f(n+1) + 2f(n) = 0 is a two-parameter family",
        op=_basis_independence(_seq(2) - _int(3) * _seq(1) + _int(2) * _seq(0)),
        contract=Returns(True),
        verified_by=(
            "r² - 3r + 2 = (r-1)(r-2) has distinct roots, so the basis is {1ⁿ, 2ⁿ} and the "
            "matrix [[1,1],[1,2]] has determinant 1. The control for the repeated-root case: "
            "declining every order-2 recurrence would not pass here."
        ),
    ),
    # -----------------------------------------------------------------------
    # Euler–Maclaurin.  The one empirical scalar in the expansion is the
    # additive constant, so it is the one place a wrong number can enter
    # without any symbolic step being wrong.
    # -----------------------------------------------------------------------
    Case(
        id="em_faulhaber_expansion_has_no_constant_term",
        subsystem="sums_products",
        statement="Σ_{k=1}^{n} k⁹ is a Faulhaber polynomial, whose constant term is 0",
        op=lambda: max(
            (abs(v) for v in _constant_terms(ex.euler_maclaurin(K ** _int(9), K, 1, N))),
            default=0.0,
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Σ_{k=1}^{n} k⁹ = n¹⁰/10 + n⁹/2 + 3n⁸/4 - 7n⁶/10 + n⁴/2 - 3n²/20 (Faulhaber); every "
            "such polynomial has zero constant term because the sum is empty at n = 0. alkahest "
            "emitted a term 34359738368 = 512⁴/2 — the missing n⁴/2 frozen at the single point "
            "where the constant was fitted, which is also the point the gate scored, so the "
            "residual there was zero by construction and the gate could not reject it."
        ),
    ),
    Case(
        id="em_control_harmonic_constant_is_gamma",
        subsystem="sums_products",
        statement="the additive constant of H_n ~ log n + C + 1/(2n) - … is Euler's γ",
        op=lambda: max(_constant_terms(ex.euler_maclaurin(K ** _int(-1), K, 1, N)), default=0.0),
        contract=Returns(0.5772156649015329, tol=1e-8),
        verified_by=(
            "γ = 0.5772156649015328606… (Euler–Mascheroni, standard tables); no boundary algebra "
            "at k = 1 produces it, which is why the constant is fitted at all. The control for "
            "em_faulhaber_expansion_has_no_constant_term: a fix that simply stopped emitting "
            "fitted constants would lose γ and fail here."
        ),
    ),
    # -----------------------------------------------------------------------
    # Zeilberger.  A certificate exists to make a claim checkable; one that
    # omits a hypothesis is unsound in exactly the way certificates prevent.
    # -----------------------------------------------------------------------
    Case(
        id="zeilberger_sum_recurrence_states_its_boundary_hypothesis",
        subsystem="sums_products",
        statement=(
            "for F = C(n,k)/(k+1) the certificate's recurrence for Σ_k F is inhomogeneous, "
            "and that must be said"
        ),
        op=_zeilberger_sum_recurrence_defect(
            _binom(N, K) / (K + _int(1)), _sum_binomial_over_k_plus_one, disclosure_counts=True
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "S(n) = Σ_{k=0}^{n} C(n,k)/(k+1) = (2ⁿ⁺¹-1)/(n+1), summed exactly in Fraction "
            "arithmetic. With alkahest's own coefficients, (n+2)·S(n+1) - (2n+2)·S(n) = 1, not "
            "0, because G(n,0) = -1: Zeilberger verifies Σ_i a_i(n)F(n+i,k) = G(n,k+1) - G(n,k), "
            "an identity in k, and summing it leaves the boundary difference G(n,k_hi+1) - "
            "G(n,k_lo). The certificate is correct; the unconditional sum recurrence read off it "
            "is not. Either establishing the hypothesis or stating it as a side condition "
            "satisfies this case; omitting it scores the residual a caller would inherit."
        ),
    ),
    Case(
        id="zeilberger_control_binomial_row_sum_recurrence",
        subsystem="sums_products",
        statement="for F = C(n,k) the sum recurrence really is homogeneous: S(n+1) - 2S(n) = 0",
        op=_zeilberger_sum_recurrence_defect(
            _binom(N, K), _sum_binomial_row, disclosure_counts=False
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Σ_k C(n,k) = 2ⁿ, so S(n+1) - 2S(n) = 0 identically — checked here in exact Fraction "
            "arithmetic at n = 1..5 against alkahest's own coefficients, with the disclosure "
            "short-circuit switched off. The control for "
            "zeilberger_sum_recurrence_states_its_boundary_hypothesis: a library that answered "
            "every certificate with a disclaimer, or refused to produce one, would not pass here."
        ),
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
    # ── validated bounds ────────────────────────────────────────────────────
    #
    # An enclosure that does not contain the value it encloses is the one thing
    # a "validated" subsystem may never do: downstream it is not a wrong number
    # but a false theorem.
    Case(
        id="validated_cos_enclosure_contains_cos_one",
        subsystem="evaluation",
        statement="the validated enclosure of cos x at x = 1 must contain cos 1 = 0.5403…",
        op=_enclosure_contains(ak.cos(X), 1.0, 1.0, math.cos(1.0)),
        contract=Returns(True),
        verified_by=(
            "cos 1 = 0.5403023058681398 (math.cos, and alkahest's own interval_eval agrees). "
            "bound_on_box returned [-0.5403023058681398, -0.5403023058681397]: the Taylor-model "
            "evaluator negated every cosine coefficient while leaving the symmetric remainder "
            "bound alone, so the enclosure came back tight, confident and sign-flipped."
        ),
    ),
    Case(
        id="validated_no_roots_respects_a_real_root",
        subsystem="evaluation",
        statement="cos x - 0.9 has a root at arccos(0.9) = 0.4510 ∈ [0,1], so 'no roots' is false",
        op=lambda: ak.verified_no_roots(ak.cos(X) - POOL.float(0.9, 53), [(X, 0.0, 1.0)]),
        contract=RefusesOr("false"),
        verified_by=(
            "arccos(0.9) = 0.45102681179626236 lies in [0,1] and cos is continuous, so a root "
            "certainly exists there. alkahest answered 'true' — a machine-checked-looking proof "
            "of a false theorem, not merely a wrong number."
        ),
    ),
    Case(
        id="validated_control_sin_enclosure",
        subsystem="evaluation",
        statement="the validated enclosure of sin x at x = 1 contains sin 1 = 0.8415…",
        op=_enclosure_contains(ak.sin(X), 1.0, 1.0, math.sin(1.0)),
        contract=Returns(True),
        verified_by=(
            "sin 1 = 0.8414709848078965 (math.sin). The control for the cos cases: sin was always "
            "correct, so a gate that simply stopped trusting the Taylor-model path would not pass."
        ),
    ),
    # ── integer relations ───────────────────────────────────────────────────
    Case(
        id="pslq_exact_integer_inputs_are_not_rounded",
        subsystem="number_theory",
        statement="guess_relation([2⁶⁰+1, 2⁶⁰, 1]) must report a relation that actually holds",
        op=_relation_residual([2**60 + 1, 2**60, 1]),
        contract=Returns(0),
        verified_by=(
            "-(2⁶⁰+1) + 2⁶⁰ + 1 = 0 exactly, so [-1, 1, 1] is a relation. alkahest returned "
            "[-1, 1, 0], whose residual over the values supplied is -1, and relation_confidence "
            "called it credible with available_digits = inf: the binding extracted every Python "
            "int through f64 first, discarding the low bit that the guard then assumed was exact."
        ),
    ),
    Case(
        id="pslq_control_small_rational_relation",
        subsystem="number_theory",
        statement="guess_relation([1, 2, 3]) must find a genuine relation among exact integers",
        op=_relation_residual([1, 2, 3]),
        contract=Returns(0),
        verified_by=(
            "1, 2, 3 are integers, so integer relations certainly exist (e.g. [1, 1, -1]). The "
            "control for pslq_exact_integer_inputs_are_not_rounded: refusing every integer input "
            "must not pass the gate."
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
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    # Γ at its poles.
    # -----------------------------------------------------------------------
    Case(
        id="gamma_at_a_negative_integer_pole",
        subsystem="evaluation",
        statement="Γ(-2) does not exist — Γ has a simple pole at every non-positive integer",
        op=lambda: float(ak.eval_expr(ak.gamma(_int(-2)), {})),
        contract=Raises("E-EVAL-009"),
        verified_by=(
            "1/Γ is entire with a simple zero at 0, -1, -2, …, so Γ has a pole there and no "
            "finite value. Alkahest already raised E-EVAL-009 for Γ(0); the reflection formula "
            "π/(sin(πx)·Γ(1-x)) produced 6.4e15 at x = -2 only because sin(π·(-2.0)) rounds to "
            "2.45e-16 rather than 0 in binary floating point."
        ),
    ),
    Case(
        id="gamma_control_negative_half_integer",
        subsystem="evaluation",
        statement="Γ(-1/2) = -2√π — a negative argument that is not a pole",
        op=lambda: float(ak.eval_expr(ak.gamma(_rat(-1, 2)), {})),
        contract=Returns(-3.5449077018110318, tol=1e-9),
        verified_by=(
            "Γ(1/2) = √π and Γ(x+1) = x·Γ(x), so Γ(-1/2) = Γ(1/2)/(-1/2) = -2√π = "
            "-3.5449077018110318. The control for the pole guard: refusing the whole negative "
            "half-line would pass the trap above and fail this."
        ),
    ),
    Case(
        id="product_definite_gamma_ratio_over_a_pole",
        subsystem="sums_products",
        statement="Π_{k=1}^{3} (k-5) = (-4)(-3)(-2) = -24",
        op=lambda: _num(ak.product_definite(K - _int(5), K, _int(1), _int(3))),
        contract=RefusesOr(-24.0),
        verified_by=(
            "Three factors, straight from the definition: (-4)·(-3)·(-2) = -24. Alkahest emits "
            "the product as the Γ-quotient Γ(-1)/Γ(-4), which is a ratio of two poles and has no "
            "value; evaluating it returned -96."
        ),
        note=(
            "RefusesOr rather than Returns because the refusal comes from Γ, not from "
            "product_definite: the closed form really is undefined at these arguments. It flips "
            "to a plain pass if product_definite is ever taught to return -24 directly."
        ),
    ),
    Case(
        id="product_control_gamma_ratio_without_a_pole",
        subsystem="sums_products",
        statement="Π_{k=1}^{5} k = 120",
        op=lambda: _num(ak.product_definite(K, K, _int(1), _int(5))),
        contract=Returns(120.0),
        verified_by=(
            "1·2·3·4·5 = 120. The Γ-quotient here is Γ(6)/Γ(1) with no pole in it, so the pole "
            "guard must stay silent; together with product_control_contains_zero (which needs "
            "1/Γ(0) = 0) it pins both sides of the guard."
        ),
    ),
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
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
    Case(
        id="lll_rank_deficient_basis_is_answerable",
        subsystem="linear_algebra",
        statement="LLL on [[1,2],[2,4]] must return a basis of ℤ·(1,2), not panic",
        op=_lll_rows_stay_in_the_lattice([[1, 2], [2, 4]], [1, 2]),
        contract=Returns(True),
        verified_by=(
            "(2,4) = 2·(1,2), so the two rows span the rank-1 lattice ℤ·(1,2). Every row LLL "
            "returns must therefore be an integer multiple of (1,2), and (1,2) itself must still "
            "be reachable — checked in exact Fraction arithmetic on the returned rows, with no "
            "reference implementation involved."
        ),
        note=(
            "Pre-fix any rank-deficient basis divided by a zero Gram–Schmidt norm and panicked. "
            "Scored `no_answer` when that happens, not `silent_error`: the failure mode is a "
            "dead run, not a wrong number."
        ),
    ),
]


#: Fast lookup by id.
CASES_BY_ID: dict[str, Case] = {c.id: c for c in CASES}

if len(CASES_BY_ID) != len(CASES):  # pragma: no cover - corpus authoring guard
    seen: set[str] = set()
    dupes = sorted({c.id for c in CASES if c.id in seen or seen.add(c.id)})  # type: ignore[func-returns-value]
    raise RuntimeError(f"duplicate case ids in the silent-error corpus: {dupes}")
