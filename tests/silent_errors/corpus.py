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
import re
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
#: Symbolic geometric ratio, for the `Σ rᵏ` cases.
R = POOL.symbol("r")


def _int(v: int) -> ak.Expr:
    return POOL.integer(v)


def _rat(a: int, b: int) -> ak.Expr:
    return POOL.rational(a, b)


#: Apéry A005259, index-shifted to `Σ_i a_i(n)·A(n+i) = 0`:
#: `(n+2)³A(n+2) − (34n³+153n²+231n+117)A(n+1) + (n+1)³A(n) = 0`. Built once —
#: construction is cheap, but the corpus is a hot path on every pull request.
_APERY_MOD = ak.ModularRecurrence(
    [[1, 3, 3, 1], [-117, -231, -153, -34], [8, 12, 6, 1]],
    [1, 5],
)


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


# ---------------------------------------------------------------------------
# Integral-transform helpers
# ---------------------------------------------------------------------------

#: Symbols the transform cases share.  `T`/`S` are the Laplace pair, `XX`/`XI`
#: the Fourier pair, `NN`/`ZZ` the Z pair; `PI` is the interned π the Fourier
#: table emits and has to be bound before anything can be evaluated.
T = POOL.symbol("t")
S = POOL.symbol("s")
XX = POOL.symbol("xspace")
XI = POOL.symbol("xi")
NN = POOL.symbol("nidx")
ZZ = POOL.symbol("z")
PI = POOL.symbol("pi")


def _unconditional(build: Callable[[], ak.Expr], env: dict) -> Callable[[], float]:
    """Answer = a transform's value at *env*, but only if it was unconditional.

    ``experimental.transform_side_conditions()`` reports the hypotheses the call
    that just ran had to assume about a symbolic parameter.  A number returned
    *under an undischarged hypothesis* is not an unconditional answer — the
    caller was told, which is exactly the distinction this gate measures — so it
    is surfaced as a refusal rather than scored as a stated value.

    ``PI`` is bound for the Fourier table, which emits the interned π symbol.
    """

    def op() -> float:
        out = build()
        conds = ex.transform_side_conditions()
        if conds:
            raise ValueError(f"answer holds only under {conds}")
        return float(ak.eval_expr(out, {**env, PI: math.pi}))

    return op


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


#: ``exp(A)`` traps.  Every one of these is **defective** — the characteristic
#: polynomial has a repeated root whose eigenspace is too small — which is the
#: only case ``e^A`` needs anything beyond a diagonalisation, and the case that
#: was wrong in 3.10.0.
NILPOTENT_2X2 = [[0, 1], [0, 0]]
DEFECTIVE_2X2 = [[2, 1], [0, 2]]
NILPOTENT_3X3 = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
JORDAN_3X3 = [[2, 1, 0], [0, 2, 1], [0, 0, 2]]
TWO_JORDAN_BLOCKS = [[3, 1, 0, 0], [0, 3, 0, 0], [0, 0, 3, 1], [0, 0, 0, 3]]
#: Defective with nothing on the surface to say so: no zero off-diagonal, and
#: the repeated eigenvalue 2 only appears after the characteristic polynomial
#: is factored as (λ − 2)².
DEFECTIVE_DENSE_2X2 = [[1, 1], [-1, 3]]
ROTATION_2X2 = [[0, 1], [-1, 0]]

#: ``[[a, 1], [0, b]]``: diagonalisable for ``a != b`` and defective at ``a = b``,
#: where ``e^A`` is the confluent form and the generic formula divides by
#: ``a − b``.  Which branch holds is not decidable from the matrix alone.
_B = POOL.symbol("b")
SYMBOLIC_GAP_2X2 = ak.Matrix([[_A, _int(1)], [_int(0), _B]])


def _exp_scalar(entry: Any, bindings: dict[Any, float]) -> float:
    """Reduce one ``exp(M)`` entry to ``Re + |Im|``.

    Evaluated in **complex** mode because a matrix with a complex spectrum comes
    back written over ``sqrt(-1)`` — ``exp([[0,1],[-1,0]])[0][1]`` is
    ``(e^{i} − e^{−i})/(2i)``, which is exactly ``sin 1`` and which the real
    evaluator declines rather than mis-evaluating.

    ``Re + |Im|`` rather than ``Re`` so that an answer which is right on the
    real axis and wrong off it is still scored wrong: every matrix in these
    cases is real, so every entry of ``e^M`` is real and a correct answer has
    ``Im = 0`` exactly.
    """
    result = ak.evaluate(entry, bindings, mode="complex")
    if result.value is None:
        # ``evaluate`` reports a decline in the result rather than raising;
        # re-raise it as the ``ValueError`` the gate reads as a weak refusal, so
        # "could not be evaluated" is not scored as a corpus bug.
        raise ValueError(f"{result.status}: {result.reason}")
    value = complex(result.value)
    return value.real + abs(value.imag)


def _exp_entry(rows: list[list[int]], i: int, j: int) -> Callable[[], float]:
    """Answer = entry ``(i, j)`` of ``exp(M)``, as a float.

    One entry rather than the whole matrix because that is where the failure
    lives: every *diagonal* entry of a defective ``e^A`` was already right in
    3.10.0, so a case scored on the diagonal would have passed throughout.
    """
    return lambda: _exp_scalar(_matrix(rows).matrix_exp().to_list()[i][j], {})


def _exp_entry_at(m: ak.Matrix, i: int, j: int, **at: float) -> Callable[[], float]:
    """Answer = entry ``(i, j)`` of ``exp(m)`` evaluated at the given symbols."""
    return lambda: _exp_scalar(
        m.matrix_exp().to_list()[i][j], {POOL.symbol(k): v for k, v in at.items()}
    )


def _jordan_p_rank(rows: list[list[int]]) -> Callable[[], int]:
    """Answer = the rank of the ``P`` returned by ``jordan_form``.

    ``M = P·J·P⁻¹`` is a claim about ``P`` being a *basis*.  A rank-deficient
    ``P`` makes the identity false and ``P⁻¹`` non-existent, and neither matrix
    looks wrong on inspection — so the rank is the thing to score.
    """

    def op() -> int:
        p, _j = _matrix(rows).jordan_form()
        return p.rank()

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


#: Eight 20-digit decimal strings and the relation ``guess_relation`` returned
#: over eight such constants during the 2026-08-13 autoresearch run
#: (``temp-alkahest/testing/autoresearch-issues-2026-08-13.md`` §2).
#: Re-evaluating that relation at 60 digits gives 3.59e-14 — it is noise bought
#: with 20 digits, and ``relation_confidence`` called it credible.  Only the
#: *count* of constants and the declared precision enter the verdict, so these
#: stand in for the run's values at the same length.
_PURCHASED_20_DIGIT_CONSTANTS = [f"1.{str(k + 1) * 20}"[:22] for k in range(8)]
_PURCHASED_20_DIGIT_COEFFS = [-19, -13, 28, 1, 26, -11, 20, -65]

#: ``relation_confidence``'s three-valued verdict as a word, so that *unknown*
#: is scored as its own answer rather than collapsing into a truthy pass.
_CONFIDENCE_WORD = {True: "credible", False: "purchasable", None: "unknown"}


def _relation_verdict(constants: list, coeffs: list, **kwargs: Any) -> Callable[[], str]:
    """Answer = ``relation_confidence``'s verdict on a *found* relation:
    ``"credible"``, ``"purchasable"``, or ``"unknown"`` when the inputs' own
    precision is not knowable."""

    def op() -> str:
        return _CONFIDENCE_WORD[ak.relation_confidence(constants, coeffs, **kwargs)["credible"]]

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


def _zeilberger_boundary_tag(term: ak.Expr) -> Callable[[], str]:
    """Answer = ``cert.boundary``, the three-valued verdict on the *sum*.

    The certificate is an identity about the summand and always holds; the
    verdict is the separate claim that a recurrence for ``S(n)`` follows from
    it.  Scoring the verdict rather than the coefficients is what makes a
    ``"vanishes"`` about a sum that does not exist a silent error rather than a
    detail buried in a side-condition string.
    """

    def op() -> str:
        return str(ak.zeilberger(term, N, K).boundary)

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


# ---------------------------------------------------------------------------
# Ideal-theory helpers (3.8 silent-error hunt #2, findings 16)
# ---------------------------------------------------------------------------

#: Third variable, for the three-variable monomial-ideal cases.
_Z = POOL.symbol("z")


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


# ---------------------------------------------------------------------------
# Hypotheses and bounds that were reached silently (3.8 pre-release sweep)
# ---------------------------------------------------------------------------

#: Parameters for the parametric-solve cases.  Free symbols that a `solve` call
#: does not list as unknowns become parameters, and the answer is then only
#: claimed "generically" — the whole point of these two cases.
_A = POOL.symbol("a")
_B = POOL.symbol("b")


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


def _undisclosed_expansion_limit(base: ak.Expr, exponent: int) -> Callable[[], float]:
    """Answer = 1.0 if a bounded expansion no-oped without saying so, else 0.0.

    ``simplify_expanded`` is asked to expand.  When the internal size bound stops
    it, the *value* it returns is the input — mathematically equal, and so
    impossible to tell apart from "this is already expanded" or "this cannot be
    expanded further".  Either the expansion happens, or the derivation log
    records that a bound was reached; silently doing neither is the defect.
    """

    def op() -> float:
        power = base ** _int(exponent)
        r = ak.simplify_expanded(power)
        # Compared against plain `simplify`, not against the input: both flatten
        # `((x+y)+z)` to `(x+y+z)`, so only expansion can separate them.
        expanded = str(r.value) != str(ak.simplify(power).value)
        disclosed = any("limit" in step["rule"] for step in r.steps)
        return 0.0 if (expanded or disclosed) else 1.0

    return op


def _expansion_within_the_budget(base: ak.Expr, exponent: int, at: float) -> Callable[[], float]:
    """Answer = the expanded polynomial's value at *at*.

    The control: a power inside the bound must actually be expanded (not the
    original ``Pow``), must record **no** limit step — so the disclosure cannot
    be emitted unconditionally — and must still agree with the input at a sample
    point, which is what makes "expanded" a claim about form only.
    """

    def op() -> float:
        power = base ** _int(exponent)
        r = ak.simplify_expanded(power)
        if str(r.value) == str(ak.simplify(power).value):
            raise AssertionError("a power inside the expansion budget was left unexpanded")
        if any("limit" in step["rule"] for step in r.steps):
            raise AssertionError("a limit step was recorded for an expansion that happened")
        return float(ak.eval_expr(r.value, {X: at}))

    return op


def _perron_growth_rate(polys: list[tuple[int, ...]], terms: list[int]) -> float:
    """The growth rate ``asymptotics_from_recurrence`` is prepared to claim.

    NaN when it claims none — which the probe reads as a refusal, and which is
    the right answer whenever Poincaré–Perron's hypotheses fail. A confident
    root reported where the hypotheses do not hold is the silent error.
    """
    r = ex.asymptotics_from_recurrence(polys, N, terms=terms)
    if r.growth_rate is None or r.follows_dominant_root is False:
        return float("nan")
    return float(r.growth_rate)


def _perron_connection_constant(polys: list[tuple[int, ...]], terms: list[int]) -> float:
    r = ex.asymptotics_from_recurrence(polys, N, terms=terms)
    if not r.connection_constant_converged:
        return float("nan")
    return float(r.connection_constant)


# ---------------------------------------------------------------------------
# Validated numerics / ball arithmetic
#
# This layer's whole value proposition is that a returned interval *contains*
# the true value, so the answer scored here is the containment claim itself:
# the op asks the enclosure whether it holds an independently computed truth,
# and ``False`` — an enclosure that does not enclose — is exactly the silent
# error, because nothing in the returned object distinguishes it from a sound
# one.  Truths come from a closed form or from mpmath at 50 digits; each
# literal is derived in the case's ``verified_by``.
# ---------------------------------------------------------------------------


def _encloses(build: Callable[[], Any], truth: float) -> Callable[[], bool]:
    """Answer = does the returned ``Enclosure`` contain *truth*?"""

    def op() -> bool:
        r = build()
        return bool(r.lower <= truth <= r.upper)

    return op


def _ball_encloses(expr: ak.Expr, binding: dict[Any, Any], truth: float) -> Callable[[], bool]:
    """Answer = does ``interval_eval``'s ball claim to contain *truth*?

    ``ArbBall.contains`` is the documented way to read the guarantee, so a
    ``False`` here is the library contradicting its own contract — and it is a
    *directed* lie: a caller cross-checking a symbolic identity against it reads
    ``False`` as a refutation rather than as "no information".
    """

    def op() -> bool:
        return bool(ak.interval_eval(expr, binding).contains(truth))

    return op


def _ball_upper(expr: ak.Expr, binding: dict[Any, Any]) -> Callable[[], float]:
    """Answer = the ball's upper endpoint, for the cases where no bound exists."""

    def op() -> float:
        return float(ak.interval_eval(expr, binding).hi)

    return op


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


_C_LITERAL = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+\.\d*(?:[eE][-+]?\d+)?")
_MLIR_CONSTANT = re.compile(r"dense<([^>]+)>")
_AGREEMENT_EXPRS: list[tuple[str, Callable[[], ak.Expr]]] = [
    ("rational_2_5", lambda: _rat(2, 5) * X),
    ("rational_neg_7_3", lambda: _rat(-7, 3) + X),
    ("rational_355_113", lambda: _rat(355, 113) * X),
    ("large_integer", lambda: _int(10**30) + X),
    ("large_rational", lambda: POOL.rational(3 * 10**400 + 1, 2 * 10**400) * X),
    ("mixed_poly", lambda: _rat(2, 5) * X**3 + _rat(-7, 3) * X**2 + _rat(1, 7)),
    ("transcendental", lambda: ak.sin(X) * ak.cos(X) + _rat(1, 3) * X),
]
_AGREEMENT_POINTS = (-8.0, -1.5, -0.25, 0.5, 1.0, 2.0, 7.0)


def _c_body(code: str) -> str:
    """The single statement inside an emitted C function, whitespace-normalised."""
    return " ".join(code.splitlines()[1].split())


def _largest_c_literal(code: str) -> float:
    """The largest-magnitude `double` literal in an emitted C function."""
    return max((float(m) for m in _C_LITERAL.findall(code)), key=abs)


def _largest_mlir_constant(mlir: str) -> float:
    """The largest-magnitude `stablehlo.constant` in an emitted MLIR module."""
    return max((float(m) for m in _MLIR_CONSTANT.findall(mlir)), key=abs)


def _mlir_literals_missing_a_decimal_point(mlir: str) -> int:
    """How many emitted constants MLIR's grammar would reject.

    ``float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?`` — the point is
    required, and ``mlir-opt`` answers ``error: expected '>'`` without one.
    """
    return sum(1 for lit in _MLIR_CONSTANT.findall(mlir) if "." not in lit)


def _evaluators_disagreeing_on() -> int:
    """Count the (expression, point) pairs where the four f64 paths differ.

    ``eval_expr`` (registry interpreter), ``evaluate(mode="f64")`` (the
    ``eval::eval_f64`` facade the verification gates call), ``compile_expr``
    and ``numpy_eval`` all run the same arithmetic over the same DAG, so any
    difference means at least one of them is wrong — and the gates and the user
    are then not looking at the same number.  Comparison is exact, not
    tolerance-based: a tolerance is exactly what hides an ulp-level literal bug.
    """
    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - numpy is absent on some CI shards
        np = None

    disagreements = 0
    for _name, build in _AGREEMENT_EXPRS:
        expr = build()
        compiled = ak.compile_expr(expr, [X])
        for point in _AGREEMENT_POINTS:
            values = {
                float(ak.eval_expr(expr, {X: point})),
                float(ak.evaluate(expr, {X: point}, mode="f64").value),
                float(compiled([point])),
            }
            if np is not None:
                values.add(float(ak.numpy_eval(compiled, np.array([point]))[0]))
            if len(values) > 1:
                disagreements += 1
    return disagreements


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

#: Frequency variable for the characteristic-function cases.  Its own symbol so
#: differentiating φ cannot collide with a case that also uses ``x``.
PROB_T = POOL.symbol("prob_t")
#: Support variable for the CDF cases.
PROB_X = POOL.symbol("prob_x")


def _prob_value(build: Callable[[], Any], env: dict | None = None) -> Callable[[], float]:
    """Answer = a probability closed form reduced to a number.

    ``PI`` is bound because the Gaussian normalising constant is written with the
    interned π symbol.
    """

    def op() -> float:
        return float(ak.eval_expr(build(), {**(env or {}), PI: math.pi}))

    return op


def _prob_phi(build: Callable[[], Any], env: dict | None = None) -> Callable[[], tuple]:
    """Answer = ``(Re φ(t), Im φ(t))``.

    ``φ`` is complex-valued, so the real evaluator refuses it with ``E-EVAL-009``
    and ``evaluate(..., mode="complex")`` is the route.  Reading that refusal as
    a failure would make every characteristic-function case vacuously "refused".
    """

    def op() -> tuple:
        v = ak.evaluate(build(), {**(env or {}), PI: math.pi}, mode="complex").value
        return (float(v.real), float(v.imag))

    return op


def _prob_unconditional(build: Callable[[], Any], env: dict) -> Callable[[], float]:
    """Answer = a probability closed form's value, but only if it was *unconditional*.

    ``experimental.prob_side_conditions()`` reports the hypotheses the call that
    just ran had to assume.  A number returned under an undischarged hypothesis
    is not an unconditional answer — the caller was told — so it is surfaced as
    a refusal rather than scored as a stated value.  That is the whole point of
    the channel: without it, ``F(x)`` for a symbolic ``x`` and ``F(x)`` for an
    ``x`` known to be in the support are indistinguishable at the call site.
    """

    def op() -> float:
        out = build()
        conds = ex.prob_side_conditions()
        if conds:
            raise ValueError(f"answer holds only under {conds}")
        return float(ak.eval_expr(out, {**env, PI: math.pi}))

    return op


def _phi_real_mode(build: Callable[[], Any], env: dict) -> Callable[[], float]:
    """Answer = ``phi(t)`` forced through a **real** evaluator.

    ``phi`` is complex.  A real evaluator that quietly hands back ``Re phi`` is
    the silent error this case exists to catch: it is a clean number of the
    right magnitude, and nothing in the return value says the imaginary part —
    which for a non-symmetric law carries the whole of the mean — was dropped.
    Refusing is the only acceptable outcome.
    """

    def op() -> float:
        r = ak.evaluate(build(), {**env, PI: math.pi}, mode="f64")
        if r.value is None:
            raise ValueError(f"real evaluation declined: {r.status} {r.reason}")
        return float(r.value)

    return op


def _moment_via_charfun(build: Callable[[], Any], n: int) -> Callable[[], float]:
    """Answer = ``φ⁽ⁿ⁾(0) / iⁿ`` — i.e. ``E[Xⁿ]`` reached by differentiating the
    characteristic function at the origin instead of by the moment table.

    ``φ⁽ⁿ⁾(0) = iⁿ·E[Xⁿ]``, so the n-th derivative is real for even ``n`` and
    purely imaginary for odd ``n``; this reads off whichever part carries it.
    The two routes share no closed form, which is what makes the agreement worth
    asserting.
    """

    def op() -> float:
        e = build().characteristic_function(PROB_T)
        for _ in range(n):
            e = ak.simplify(ak.diff(e, PROB_T))
            if isinstance(e, ak.DerivedResult):
                e = e.value
        v = ak.evaluate(e, {PROB_T: 0.0, PI: math.pi}, mode="complex").value
        return [v.real, v.imag, -v.real, -v.imag][n % 4]

    return op


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
    # -----------------------------------------------------------------------
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
    # The matrix exponential of a DEFECTIVE matrix.
    #
    # `e^A` for a diagonalisable A is `P·e^D·P⁻¹` and was always right.  For a
    # defective A the Jordan block contributes `e^λ·Σ N^k/k!`, and in 3.10.0 the
    # nilpotent power `N^k` was written `λ^k` — so every nilpotent block lost its
    # off-diagonal entirely (`exp([[0,1],[0,0]])` came back as the identity) and
    # every other defective block was off by a factor of `λ^k`.  Clean, plausible
    # matrices, no exception, no flag.
    #
    # Every expectation below is `sympy.Matrix(M).exp()`, checked independently.
    # The diagonal entries are deliberately *not* scored: they were right
    # throughout, so a case reading one would have passed the whole time.
    # -----------------------------------------------------------------------
    Case(
        id="matrix_exp_nilpotent_2x2_off_diagonal",
        subsystem="linear_algebra",
        statement="exp([[0,1],[0,0]])[0][1] = 1",
        op=_exp_entry(NILPOTENT_2X2, 0, 1),
        contract=Returns(1.0),
        verified_by="N² = 0, so the series terminates: e^N = I + N = [[1,1],[0,1]]. "
        "sympy.Matrix([[0,1],[0,0]]).exp() agrees. alkahest 3.10.0 returned the identity, "
        "i.e. 0 here — the off-diagonal of e^N for a nilpotent N is the one entry that "
        "cannot be zero, since e^N = I would force N = log I = 0.",
    ),
    Case(
        id="matrix_exp_defective_off_diagonal_is_not_doubled",
        subsystem="linear_algebra",
        statement="exp([[2,1],[0,2]])[0][1] = e², not 2e²",
        op=_exp_entry(DEFECTIVE_2X2, 0, 1),
        contract=Returns(7.38905609893065),
        verified_by="A = 2I + N with N² = 0, and 2I commutes with N, so "
        "e^A = e²·(I + N) = [[e², e²], [0, e²]]; e² = 7.38905609893065. "
        "sympy.Matrix([[2,1],[0,2]]).exp() agrees. alkahest 3.10.0 returned 2e² = "
        "14.7781121978613 — exactly the factor λ¹ that the k = 1 term should not carry.",
    ),
    Case(
        id="matrix_exp_nilpotent_3x3_second_superdiagonal",
        subsystem="linear_algebra",
        statement="exp([[0,1,0],[0,0,1],[0,0,0]])[0][2] = 1/2",
        op=_exp_entry(NILPOTENT_3X3, 0, 2),
        contract=Returns(0.5),
        verified_by="N³ = 0, so e^N = I + N + N²/2 and the (0,2) entry is (N²)₀₂/2! = 1/2. "
        "sympy agrees. Two distinct defects met here in 3.10.0: the λ^k factor zeroed it, "
        "and the block-size detector read J[i][i+sz] instead of J[i+sz−1][i+sz], splitting "
        "the 3×3 block into a 2×2 and a 1×1 so the 1/2 had nowhere to come from.",
    ),
    Case(
        id="matrix_exp_full_jordan_block_corner",
        subsystem="linear_algebra",
        statement="exp([[2,1,0],[0,2,1],[0,0,2]])[0][2] = e²/2",
        op=_exp_entry(JORDAN_3X3, 0, 2),
        contract=Returns(3.694528049465325),
        verified_by="e^{2I+N} = e²(I + N + N²/2); the corner is e²/2 = 3.694528049465325. "
        "sympy.Matrix([[2,1,0],[0,2,1],[0,0,2]]).exp() agrees. The 3×3 block is the "
        "smallest matrix on which the block-size misdetection is visible on its own.",
    ),
    Case(
        id="matrix_exp_two_jordan_blocks_one_eigenvalue",
        subsystem="linear_algebra",
        statement="exp(J₂(3) ⊕ J₂(3))[2][3] = e³",
        op=_exp_entry(TWO_JORDAN_BLOCKS, 2, 3),
        contract=Returns(20.085536923187668),
        verified_by="e^A is block diagonal with each block e³(I + N) = [[e³, e³],[0, e³]]; "
        "e³ = 20.085536923187668. sympy agrees. This is the shape that breaks a Jordan-basis "
        "route rather than the block formula: both chains come out of the same kernel, so a "
        "P built by taking the same generator twice is singular.",
    ),
    Case(
        id="matrix_exp_defective_without_a_zero_off_diagonal",
        subsystem="linear_algebra",
        statement="exp([[1,1],[-1,3]])[0][0] = 0",
        op=_exp_entry(DEFECTIVE_DENSE_2X2, 0, 0),
        contract=Returns(0.0),
        verified_by="det(λI − A) = λ² − 4λ + 4 = (λ − 2)², and A − 2I = [[-1,1],[-1,1]] has "
        "rank 1, so A is defective with one 2×2 block. e^A = e²(I + (A − 2I)) = "
        "[[0, e²], [-e², 2e²]], whose (0,0) entry is exactly 0. sympy agrees. Nothing on the "
        "surface of this matrix announces defectiveness — no zero off-diagonal, no repeated "
        "diagonal entry — so it is the case a triangular-only fast path would miss.",
    ),
    Case(
        id="matrix_exp_rotation_off_diagonal_is_sin_one",
        subsystem="linear_algebra",
        statement="exp([[0,1],[-1,0]])[0][1] = sin 1",
        op=_exp_entry(ROTATION_2X2, 0, 1),
        contract=Returns(0.8414709848078965),
        verified_by="A generates rotation: e^{θA} = [[cos θ, sin θ], [-sin θ, cos θ]], so at "
        "θ = 1 the (0,1) entry is sin 1 = 0.8414709848078965 (Euler / the 2×2 rotation "
        "group). sympy.Matrix([[0,1],[-1,0]]).exp() agrees. The complex-spectrum control: "
        "λ = ±i are distinct, so the answer must be real and must not acquire an imaginary "
        "part from the route that produced it.",
    ),
    Case(
        id="matrix_exp_symbolic_defective_gap",
        subsystem="linear_algebra",
        statement="exp([[a,1],[0,b]])[0][1] at a = b = 3/2 is e^{3/2}, not a 0/0 form",
        op=_exp_entry_at(SYMBOLIC_GAP_2X2, 0, 1, a=1.5, b=1.5),
        contract=RefusesOr(4.4816890703380645),
        verified_by="For a != b the entry is (e^a − e^b)/(a − b); its limit as b → a is e^a, "
        "which is also what the defective case gives directly: at a = b the matrix is "
        "a·I + N with N² = 0, so e^A = e^a(I + N) and the entry is e^{3/2} = "
        "4.4816890703380645. sympy: `sp.Matrix([[a,1],[0,b]]).exp().subs(b,a)` and "
        "`sp.limit((sp.exp(a)-sp.exp(b))/(a-b), b, a)` both give exp(a). Refusing is "
        "acceptable — which branch holds is not decidable from the matrix — but a *different* "
        "finite number is not, and neither is the generic form evaluated at the confluence, "
        "which is 0/0.",
        note="Passes today by a refusal, not by producing the confluent value: alkahest "
        "returns the generic (e^a − e^b)/(a − b) — correct everywhere except the confluence "
        "— and evaluating it at a = b raises E-EVAL-004. That is a refusal a caller has to "
        "look at the value to notice, so what carries the real signal is "
        "`alkahest.matrix_exp_side_conditions()`, which lists `a − b ≠ 0`; "
        "test_matrix_exp_reports_the_eigenvalue_gap_it_divided_by in "
        "tests/test_linear_algebra.py pins that. The stronger outcome is the "
        "Returns(e^{3/2}) branch, i.e. taking the confluent limit when the parameters are "
        "bound.",
    ),
    Case(
        id="matrix_exp_control_diagonalizable_2x2",
        subsystem="linear_algebra",
        statement="exp([[1,2],[3,4]])[0][0] = 51.968956198705",
        op=_exp_entry([[1, 2], [3, 4]], 0, 0),
        contract=Returns(51.968956198705),
        verified_by="Eigenvalues (5 ± √33)/2 are distinct, so A = PDP⁻¹ and e^A = Pe^DP⁻¹; "
        "sympy.Matrix([[1,2],[3,4]]).exp().evalf(20) gives 51.968956198705 in the (0,0) "
        "entry. The control for the defective cases above: a library that 'fixed' them by "
        "refusing every matrix with a repeated eigenvalue would still have to answer this "
        "one, and one that broke the diagonalisable route while repairing the Jordan route "
        "would fail here.",
    ),
    Case(
        id="matrix_exp_control_diagonal_2x2",
        subsystem="linear_algebra",
        statement="exp(diag(1,2))[1][1] = e²",
        op=_exp_entry([[1, 0], [0, 2]], 1, 1),
        contract=Returns(7.38905609893065),
        verified_by="exp(diag(d₁,…,dₙ)) = diag(e^{d₁},…,e^{dₙ}); e² = 7.38905609893065. "
        "sympy agrees. The second control, one step simpler than the diagonalisable one: it "
        "pins the entrywise fast path, which is the only route that never consults a "
        "spectrum at all.",
    ),
    Case(
        id="matrix_exp_control_zero_matrix_is_identity",
        subsystem="linear_algebra",
        statement="exp(0) = I, so exp([[0,0],[0,0]])[0][1] = 0",
        op=_exp_entry(ZERO_2X2, 0, 1),
        contract=Returns(0.0),
        verified_by="e^0 = I by the series, whose every term past the first vanishes. The "
        "companion to matrix_exp_nilpotent_2x2_off_diagonal: the identity is the *right* "
        "answer here and the wrong one there, and the two differ in a single entry of the "
        "input — so a gate holding only the nilpotent case could be passed by never "
        "returning the identity at all.",
    ),
    Case(
        id="jordan_form_transform_is_a_basis",
        subsystem="linear_algebra",
        statement="the P of jordan_form(J₂(3) ⊕ J₂(3)) has rank 4",
        op=_jordan_p_rank(TWO_JORDAN_BLOCKS),
        contract=RefusesOr(4),
        verified_by="M = P·J·P⁻¹ is a similarity, so P must be invertible and a 4×4 "
        "invertible matrix has rank 4; sympy.Matrix(M).jordan_form() returns P = I here, "
        "det 1. Both chains of this matrix are drawn from ker(M − 3I)² = ℝ⁴, and alkahest "
        "3.10.0 took the same generator for both, returning a P with two identical columns "
        "— rank 3, det 0, so the identity it claims is false and P⁻¹ does not exist. "
        "Refusing is acceptable; a rank-deficient P silently labelled a similarity transform "
        "is not.",
    ),
    Case(
        id="jordan_form_control_diagonalizable_transform",
        subsystem="linear_algebra",
        statement="the P of jordan_form([[1,2],[3,4]]) has rank 2",
        op=_jordan_p_rank([[1, 2], [3, 4]]),
        contract=Returns(2),
        verified_by="Distinct eigenvalues (5 ± √33)/2 give two independent eigenvectors, so "
        "P is invertible and has rank 2; sympy.Matrix([[1,2],[3,4]]).jordan_form() returns a "
        "P with det ≠ 0. The control for the case above: a library that answered it by "
        "refusing every jordan_form would pass that one and fail this.",
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
    # Poincaré–Perron.  A recurrence always *has* a characteristic polynomial,
    # so a growth rate is always available to report; the question is whether
    # the theorem's hypotheses license reporting it.
    # -----------------------------------------------------------------------
    Case(
        id="perron_equal_modulus_roots_get_no_growth_rate",
        subsystem="sums_products",
        statement="u(n+2) = 4·u(n) has characteristic roots ±2 and no single growth rate",
        op=lambda: _perron_growth_rate([(-4,), (0,), (1,)], [1, 2]),
        contract=RefusesOr(),
        verified_by=(
            "the general solution is A·2ⁿ + B·(−2)ⁿ, so u(n+1)/u(n) does not converge: for "
            "u(0)=1, u(1)=2 the ratio is 2 at every step, but for u(0)=1, u(1)=0 the sequence "
            "is 1, 0, 4, 0, 16, … and the ratio alternates between 0 and ∞. Poincaré's theorem "
            "requires the roots to have distinct moduli and these do not, so 'ρ = 2' is a "
            "statement about one solution presented as one about the recurrence."
        ),
    ),
    Case(
        id="perron_subdominant_solution_does_not_get_the_dominant_rate",
        subsystem="sums_products",
        statement="u(n+2) = 3u(n+1) − 2u(n) with u(0) = u(1) = 1 is the constant sequence",
        op=lambda: _perron_growth_rate([(2,), (-3,), (1,)], [1, 1]),
        contract=RefusesOr(1.0),
        verified_by=(
            "χ(t) = t² − 3t + 2 = (t−1)(t−2), so the general solution is A + B·2ⁿ; "
            "u(0) = u(1) = 1 forces B = 0 and u ≡ 1. Poincaré's conclusion is that the ratio "
            "tends to *some* characteristic root, not the largest, so reporting 2 here would "
            "be exponential growth claimed for a constant sequence."
        ),
    ),
    Case(
        id="perron_control_fibonacci_connection_constant_is_one_over_root_five",
        subsystem="sums_products",
        statement="F(n) ~ φⁿ/√5, so the fitted connection constant must be 1/√5",
        op=lambda: _perron_connection_constant([(-1,), (-1,), (1,)], [0, 1]),
        contract=Returns(0.4472135954999579, tol=1e-9),
        verified_by=(
            "Binet: F(n) = (φⁿ − ψⁿ)/√5 with |ψ| < 1, so F(n)·φ⁻ⁿ → 1/√5 = 0.4472135954999579… "
            "(math.sqrt(5)). The control for the two refusal cases above: an implementation "
            "that declined to claim a growth law whenever the hypotheses were awkward would "
            "pass those and fail this one."
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
        id="pslq_confidence_declared_precision_refutes_a_purchased_relation",
        subsystem="number_theory",
        statement=(
            "relation_confidence must not call 8 coefficients of size ≤ 65 credible when the "
            "inputs carry 20 digits"
        ),
        op=_relation_verdict(_PURCHASED_20_DIGIT_CONSTANTS, _PURCHASED_20_DIGIT_COEFFS, digits=20),
        contract=Returns("purchasable"),
        verified_by=(
            "A counting argument, independent of alkahest: there are ~H^n = 65^8 ≈ 10^14.5 integer "
            "vectors with |aᵢ| ≤ 65, and the smallest |Σ aᵢcᵢ| among them is ~10^-14.5 for *any* 8 "
            "constants. At 20 digits such a vector is therefore always findable and is evidence of "
            "nothing. These are the coefficients guess_relation returned on 2026-08-13 over 8 "
            "constants at 20 digits; re-evaluating that relation at 60 digits gives 3.59e-14, the "
            "noise floor the counting argument predicts, not the ~1e-20 a true relation would give."
        ),
    ),
    Case(
        id="pslq_confidence_unknown_precision_is_not_a_pass",
        subsystem="number_theory",
        statement=(
            "relation_confidence must answer 'unknown', not 'credible', when the inputs are "
            "decimal strings of undeclared accuracy"
        ),
        op=_relation_verdict(_PURCHASED_20_DIGIT_CONSTANTS, _PURCHASED_20_DIGIT_COEFFS),
        contract=Returns("unknown"),
        verified_by=(
            "A decimal string is both an exact rational and the way a truncated constant is "
            "spelled, and the string does not say which — so no verdict is derivable from it. The "
            "old contract assumed exact and answered 'credible' for every relation among strings, "
            "which is the input a PSLQ loop actually produces."
        ),
        note="The gate's own shape: 'unknown' is the honest answer, and must not read as a pass.",
    ),
    Case(
        id="pslq_control_confidence_credible_at_full_precision",
        subsystem="number_theory",
        statement=(
            "relation_confidence must still call the same relation credible when the inputs "
            "carry 200 digits"
        ),
        op=_relation_verdict(_PURCHASED_20_DIGIT_CONSTANTS, _PURCHASED_20_DIGIT_COEFFS, digits=200),
        contract=Returns("credible"),
        verified_by=(
            "The control for the two cases above: 8 coefficients of size ≤ 65 cost ~14.5 digits, "
            "so 200 digits of agreement is ~185 digits more than the relation could have been "
            "bought with. A gate that answered 'purchasable' here would be passing by refusing "
            "everything."
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
    # A pole of the *summand* inside the summation range, seen from the
    # holonomic side.  `sum_definite` has had an interior-pole guard since
    # 3.8.0; `zeilberger`'s boundary verdict did not, and a verdict is a much
    # more dangerous thing to get wrong than a number, because it is labelled
    # "proved".
    # -----------------------------------------------------------------------
    Case(
        id="zeilberger_boundary_pole_inside_range",
        subsystem="sums_products",
        statement=(
            "Σ_{k=0}^{n} C(n,k)/(k-3) has no value for n ≥ 3, so its certificate implies no "
            "recurrence for the sum"
        ),
        op=_zeilberger_boundary_tag(_binom(N, K) / (K - _int(3))),
        contract=Returns("unknown"),
        verified_by=(
            "The k=3 term of the sum is C(n,3)/0. alkahest returned boundary='vanishes' — "
            "'proved: Σ_i a_i(n)·S(n+i) = 0' — with coefficients (2n+2), (2-3n), (n-1). At n=1 "
            "the last one is 0, so the claim reads 4·S(1) - S(2) = 0 with every quantity in it "
            "defined: S(1) = 1/(-3) + 1/(-2) = -5/6 and S(2) = 1/(-3) + 2/(-2) + 1/(-1) = -7/3, "
            "computed term by term from the definition. That is -1, not 0, and solving the "
            "claimed recurrence for S(2) gives -10/3 against the true -7/3."
        ),
    ),
    Case(
        id="zeilberger_control_pole_below_the_range",
        subsystem="sums_products",
        statement="Σ_{k=0}^{n} C(n,k)/(k+1): the pole is at k = -1, outside the range",
        op=_zeilberger_boundary_tag(_binom(N, K) / (K + _int(1))),
        contract=Returns("nonzero"),
        verified_by=(
            "Every term C(n,k)/(k+1) with 0 ≤ k ≤ n is finite, so the sum exists and the "
            "boundary analysis must still answer. It is the A279013-shaped case whose true "
            "recurrence is inhomogeneous: (n+2)·S(n+1) - (2n+2)·S(n) = 1, checked against "
            "S(0) = 1, S(1) = 3/2, S(2) = 7/3 from Σ_{k=0}^{m} C(m,k)/(k+1) = (2^{m+1}-1)/(m+1). "
            "The control for the interior-pole guard: refusing this would trade a false verdict "
            "for a dead engine."
        ),
    ),
    Case(
        id="zeilberger_control_natural_boundary_still_vanishes",
        subsystem="sums_products",
        statement="Σ_{k=0}^{n} C(n,k) = 2ⁿ — the textbook natural boundary",
        op=_zeilberger_boundary_tag(_binom(N, K)),
        contract=Returns("vanishes"),
        verified_by=(
            "C(n,k) is finite at every integer k, and vanishes outside 0 ≤ k ≤ n, so the "
            "homogeneous S(n+1) = 2·S(n) holds — as 1, 2, 4, 8 confirms. The second control: a "
            "guard that fired on the shape rather than on a pole would break this."
        ),
    ),
    # -----------------------------------------------------------------------
    # `verify_wz_pair` — a verifier's false *negative* is not a lie, but it is
    # a verifier that cannot verify.
    # -----------------------------------------------------------------------
    Case(
        id="wz_pair_polynomial_is_verified",
        subsystem="sums_products",
        statement="(F, G) = (n·k, k(k-1)/2) is a WZ pair: both differences are k",
        op=lambda: bool(ak.verify_wz_pair(N * K, K * (K - _int(1)) * _rat(1, 2), N, K)),
        contract=Returns(True),
        verified_by=(
            "F(n+1,k) - F(n,k) = (n+1)k - nk = k, and G(n,k+1) - G(n,k) = (k+1)k/2 - k(k-1)/2 = "
            "k. Expanded by hand; both sides are the polynomial k. alkahest returned False — "
            "simplify does not expand a product, so k·(n+1) - n·k and k(k+1)/2 - k(k-1)/2 were "
            "compared structurally and found different."
        ),
    ),
    Case(
        id="wz_pair_control_non_pair_is_refuted",
        subsystem="sums_products",
        statement="(F, G) = (n·k, 0) is not a WZ pair: k ≠ 0",
        op=lambda: bool(ak.verify_wz_pair(N * K, _int(0), N, K)),
        contract=Returns(False),
        verified_by=(
            "F(n+1,k) - F(n,k) = k while G(n,k+1) - G(n,k) = 0, and k is not identically zero. "
            "The control for the case above: a verifier that answered True by giving up would "
            "pass that one and fail this."
        ),
    ),
    # -----------------------------------------------------------------------
    # A geometric series whose ratio is a symbol: elementary, and the r = 1
    # branch is a second case rather than a detail.
    # -----------------------------------------------------------------------
    Case(
        id="sum_geometric_symbolic_ratio",
        subsystem="sums_products",
        statement="Σ_{k=0}^{n} rᵏ = (r^{n+1} - 1)/(r - 1); at r = 3, n = 4 that is 121",
        op=lambda: float(
            ak.eval_expr(ak.sum_definite(R**K, K, _int(0), N).value, {R: 3.0, N: 4.0})
        ),
        contract=Returns(121.0, tol=1e-9),
        verified_by=(
            "1 + 3 + 9 + 27 + 81 = 121, summed term by term. alkahest refused with E-SUM-001 "
            "('geometric base must be a rational constant') — Gosper's certificate lives in "
            "Q(k) and a symbolic ratio is not in Q, so the whole layer underneath could not "
            "see an elementary series."
        ),
    ),
    Case(
        id="sum_geometric_symbolic_ratio_at_one",
        subsystem="sums_products",
        statement="the closed form for Σ_{k=0}^{n} rᵏ is 0/0 at r = 1 and must not answer there",
        op=lambda: float(
            ak.eval_expr(ak.sum_definite(R**K, K, _int(0), N).value, {R: 1.0, N: 4.0})
        ),
        contract=RefusesOr(),
        verified_by=(
            "Σ_{k=0}^{4} 1ᵏ = 5, but (1^5 - 1)/(1 - 1) is 0/0 — the r = 1 branch is a separate "
            "case, which is why SymPy answers this with a Piecewise. Any finite value out of "
            "the r ≠ 1 formula at r = 1 would be an arithmetic accident. alkahest records "
            "r - 1 ≠ 0 as a side condition on the derivation step and the expression itself "
            "declines to evaluate."
        ),
        note=(
            "Weak refusal: the guard is that 0/0 has no float, not a coded error. The stronger "
            "signal is the recorded side condition, which the contract vocabulary here cannot "
            "express."
        ),
    ),
    Case(
        id="sum_geometric_symbolic_ratio_to_infinity",
        subsystem="sums_products",
        statement="Σ_{k=0}^{∞} rᵏ converges only for |r| < 1, which nothing states here",
        op=lambda: _num(ak.sum_definite(R**K, K, _int(0), POOL.pos_infinity()).value),
        contract=RefusesOr(),
        verified_by=(
            "The series diverges for every |r| ≥ 1, and r is an unconstrained symbol. Returning "
            "1/(1-r) would be the geometric-series answer stated outside its disc of "
            "convergence — the same error as summing Σ2ᵏ to -1."
        ),
    ),
    Case(
        id="zeilberger_even_row_sum_holds_where_the_certificate_is_defined",
        subsystem="sums_products",
        statement=(
            "for F = C(2n,2k) the verdict is 'vanishes', and S(n+1) = 4·S(n) does hold at "
            "every n where the certificate is defined"
        ),
        op=_zeilberger_sum_recurrence_defect(
            _binom(_int(2) * N, _int(2) * K),
            lambda m: Fraction(sum(math.comb(2 * m, 2 * j) for j in range(m + 1))),
            disclosure_counts=False,
        ),
        contract=Returns(0.0, tol=1e-9),
        verified_by=(
            "Σ_{k=0}^{n} C(2n,2k) is the even half of row 2n, i.e. 2^{2n-1} for n ≥ 1: "
            "1, 2, 8, 32, 128, 512 summed term by term. S(n+1) - 4·S(n) = 0 for every n ≥ 1, "
            "checked in exact Fraction arithmetic against alkahest's own coefficients (-4, 1). "
            "It fails by -2 at n = 0 only, and n = 0 is exactly where the certificate — which "
            "carries a 1/n — is undefined, the residual hypothesis the side condition names. "
            "Pinned here so that a future verdict that ignored that hypothesis, or a guard that "
            "over-refused this shape, would show up as a change."
        ),
    ),
    Case(
        id="sum_control_empty_range",
        subsystem="sums_products",
        statement="Σ_{k=5}^{4} k = 0 — an empty range takes no terms",
        op=lambda: _num(ak.sum_definite(K, K, _int(5), _int(4)).value),
        contract=Returns(0.0),
        verified_by=(
            "hi = lo - 1 is the empty range under every convention: no term is ever taken, so "
            "the sum is 0. The telescoped G(hi+1) - G(lo) = G(5) - G(5) gives it for free, "
            "which is exactly what makes it a good check that the bounds are wired the right "
            "way round."
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
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
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
    Case(
        id="expand_power_bound_is_not_a_silent_no_op",
        subsystem="simplification",
        statement="simplify_expanded((x+y+z)^9) must expand it or record the bound it hit",
        op=_undisclosed_expansion_limit(X + Y + _Z, 9),
        contract=Returns(0.0),
        verified_by=(
            "By the multinomial theorem (x+y+z)^9 expands to C(11,2) = 55 distinct monomials, "
            "so 'already expanded' is false and the returned Pow is not the answer to the "
            "question asked. Returning the input unchanged is a correct *value* and a "
            "misleading *result*: .steps is documented as a faithful record of what happened, "
            "and it recorded nothing at all."
        ),
        note="Passes by disclosure (a derivation step) or by doing the expansion.",
    ),
    Case(
        id="expand_control_power_inside_the_budget",
        subsystem="simplification",
        statement="simplify_expanded((x+1)^6) = 729 at x = 2, expanded and unremarked",
        op=_expansion_within_the_budget(X + _int(1), 6, 2.0),
        contract=Returns(729.0),
        verified_by=(
            "(2+1)^6 = 3^6 = 729 by hand; the binomial expansion 1 + 6x + 15x² + 20x³ + 15x⁴ "
            "+ 6x⁵ + x⁶ at x = 2 gives 1+12+60+160+240+192+64 = 729, so the expanded form must "
            "agree. The control for the case above: it fails if expansion stops firing, and "
            "also if a limit step is recorded for an expansion that in fact happened."
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
    # -----------------------------------------------------------------------
    # M6 — a holonomic sequence mod p^k, at the indices where the recurrence
    # cannot simply be divided through.
    # -----------------------------------------------------------------------
    Case(
        id="modular_recurrence_through_a_singular_index",
        subsystem="number_theory",
        statement="A(13) mod 13³ = 5, reached only by dividing by (n+2)³ = 13³ at n = 11",
        op=lambda: _APERY_MOD.value_mod(13, 13, 3),
        contract=Returns(5),
        verified_by=(
            "A(13) = Σ_k C(13,k)²·C(13+k,k)² was summed as an exact Python integer from the "
            "definition and reduced mod 13³, independently of any recurrence. The recurrence "
            "route must cross n = 11, where the leading coefficient (n+2)³ is exactly 13³ and "
            "has no inverse mod 13³ at all."
        ),
        note=(
            "The classic shape: dividing by a non-unit. A modular inverse routine that returns "
            "something for a non-unit — or a Fermat-style pow(a, m-2, m), which is wrong for a "
            "prime *power* — produces a plausible residue here with no error of any kind. "
            "alkahest measures v_p of the leading coefficient first and runs the forward pass "
            "at 13⁶ so the three lost digits are ones it had already bought."
        ),
    ),
    Case(
        id="modular_recurrence_refuses_a_vanishing_leading_coefficient",
        subsystem="number_theory",
        statement="(n−4)·S(n+1) = S(n) determines no S(5), at any modulus",
        op=lambda: ak.ModularRecurrence([[-1], [-4, 1]], [1]).value_mod(5, 7, 3),
        contract=Raises("E-HOLO-007"),
        verified_by=(
            "At n = 4 the relation reads 0·S(5) = S(4) with S(4) = 1/24 ≠ 0, so no S(5) "
            "satisfies it — over ℤ, over ℚ, or over ℤ/7³. There is no right answer to return, "
            "and a larger modulus does not create one."
        ),
        note=(
            "The step before it, S(4) = 1/((−4)(−3)(−2)(−1)) = 1/24, is ordinary and is "
            "answered; only the undetermined one refuses. A gate that refused the whole "
            "recurrence would pass this case for the wrong reason."
        ),
    ),
    Case(
        id="binomial_mod_prime_power_far_above_the_prime",
        subsystem="number_theory",
        statement="binomial(2p−1, p−1) ≡ 1 (mod p³) at p = 101 — Wolstenholme",
        op=lambda: ak.binomial_mod(201, 100, 101, 3),
        contract=Returns(1),
        verified_by=(
            "Wolstenholme's theorem, and separately math.comb(201, 100) % 101**3 = 1. The "
            "argument is far larger than p, so Lucas' theorem alone (a mod-p statement) cannot "
            "reach it; the prime-power machinery has to."
        ),
        note=(
            "The trap is answering the mod-p question and labelling it mod-p³: Lucas gives "
            "1 here too, and a k>1 implementation that silently degrades to k=1 agrees with "
            "the truth on exactly this input while being wrong on most others."
        ),
    ),
    # -- transforms: the unilateral shift is a hypothesis, not a fact ---------
    Case(
        id="transform_laplace_step_advanced_past_the_origin",
        subsystem="transform",
        statement="L{θ(t+a)}(s) at a = 1.5 is 1/s, not e^{a s}/s",
        op=_unconditional(
            lambda: ex.laplace_transform(ex.heaviside(T + POOL.symbol("shift_a")), T, S),
            {POOL.symbol("shift_a"): 1.5, S: 2.5},
        ),
        contract=RefusesOr(0.4),
        verified_by=(
            "By hand from the definition: for a > 0, θ(t+a) ≡ 1 on the whole range "
            "[0,∞) the unilateral integral sees, so L{θ(t+a)} = ∫₀^∞ e^{−st} dt = 1/s "
            "= 0.4 at s = 2.5.  The second-shift rule e^{−a s}G(s) is the transform "
            "only for a shift a ≥ 0, and θ(t+a) is θ(t−(−a))."
        ),
        note=(
            "Was a silent error: the rule fired on the negative shift and returned "
            "e^{1.5·2.5}/2.5 = 17.0084 — a factor of 42 wrong, and growing with a — "
            "with an empty side-condition list.  The literal form θ(t+1) was already "
            "refused; only the symbolic shift slipped through.  It now passes by "
            "reporting `a ≤ 0`, which the caller can check and reject."
        ),
    ),
    Case(
        id="transform_laplace_impulse_before_the_origin",
        subsystem="transform",
        statement="L{δ(t+a)}(s) at a = 1.5 is 0 — the impulse is outside [0,∞)",
        op=_unconditional(
            lambda: ex.laplace_transform(ex.dirac_delta(T + POOL.symbol("shift_b")), T, S),
            {POOL.symbol("shift_b"): 1.5, S: 2.5},
        ),
        contract=RefusesOr(0.0),
        verified_by=(
            "∫₀^∞ δ(t+1.5)·e^{−st} dt = 0: the sifting point t = −1.5 is not in the "
            "domain of integration, so the unilateral transform of an impulse placed "
            "before the origin is identically zero for every s."
        ),
        note=(
            "Same defect as the step case, one table row over: the answer was "
            "e^{1.5·2.5} = 42.52 for a quantity that is 0.  Reported as `a ≤ 0` now."
        ),
    ),
    Case(
        id="transform_inverse_laplace_of_an_advance",
        subsystem="transform",
        statement="no causal f has L{f} = e^{+2s}/(s+1)",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(ak.exp(2 * S) / (S + 1), S, T),
            {T: 1.0},
        ),
        contract=RefusesOr(),
        verified_by=(
            "A unilateral transform is bounded as Re s → +∞ (dominated convergence "
            "on ∫₀^∞ f e^{−st}), and e^{2s}/(s+1) grows without bound there, so it is "
            "the transform of no causal function at all.  Numerically: F(1.7) = "
            "11.0978, while the transform of the shift rule's answer θ(t+2)e^{−2−t} "
            "is e^{−2}/(s+1) = 0.0501 — the two are not the same function."
        ),
        note=(
            "Was worse than a wrong answer: it came back with the side condition "
            "`−2 ∈ NonNegative`, i.e. a hypothesis that can never be discharged, "
            "attached to a value that is wrong under every reading.  A refutation is "
            "not a hypothesis and now refuses."
        ),
    ),
    # -- transforms: Fourier positivity selects the branch --------------------
    Case(
        id="transform_fourier_lorentzian_negative_amplitude",
        subsystem="transform",
        statement="F{−2/(1+4π²x²)}(ξ) = −e^{−|ξ|}, not e^{+|ξ|}",
        op=_unconditional(
            lambda: ex.fourier_transform(_int(-2) / (1 + 4 * PI**2 * XX**2), XX, XI),
            {XI: 0.3},
        ),
        contract=RefusesOr(-0.7408182206817179),
        verified_by=(
            "∫ c/(c₀+4π²x²)·e^{−2πiξx} dx = c/(2√c₀)·e^{−√c₀|ξ|} (residues at "
            "x = ±i√c₀/2π); at c = −2, c₀ = 1, ξ = 0.3 that is −e^{−0.3} = "
            "−0.7408182206817179.  mpmath quadosc on the same integrand agrees to "
            "10 significant figures."
        ),
        note=(
            "The trap is that c₀ = a² is satisfied by a and by −a alike, so reading "
            "the rate off the numerator picks a branch rather than deriving one.  "
            "alkahest returned e^{+|ξ|} = +1.3499: wrong sign, and *growing* where "
            "the true transform decays.  Input is all-literal — no parameter, "
            "nothing unknown, nothing to report — so the only honest options were "
            "the right number or a refusal."
        ),
    ),
    Case(
        id="transform_fourier_growing_two_sided_exponential",
        subsystem="transform",
        statement="e^{+3|x|} has no Fourier transform — the integral diverges",
        op=_unconditional(
            lambda: ex.fourier_transform(ak.exp(3 * ak.abs(XX)), XX, XI),
            {XI: 0.3},
        ),
        contract=RefusesOr(),
        verified_by=(
            "|e^{3|x|}·e^{−2πiξx}| = e^{3|x|} ≥ 1 for every x, so ∫_{−∞}^{∞} of it "
            "diverges to +∞ at every ξ.  There is no value to return."
        ),
        note=(
            "alkahest applied the a > 0 table row with a = −3 and returned the "
            "finite −6/(9+4π²ξ²) = −0.478 at ξ = 0.3.  The Gaussian row already "
            "rejected a literal non-positive curvature; this row did not, so the "
            "refusal was inconsistent as well as absent."
        ),
    ),
    # -- transforms: controls ------------------------------------------------
    Case(
        id="transform_control_laplace_step_delayed",
        subsystem="transform",
        statement="L{θ(t−2)}(s) = e^{−2s}/s — the nearest convergent neighbour",
        op=_unconditional(lambda: ex.laplace_transform(ex.heaviside(T - 2), T, S), {S: 2.5}),
        contract=Returns(0.002695178799634187),
        verified_by=(
            "∫₂^∞ e^{−2.5t} dt = e^{−5}/2.5 = 0.002695178799634187 (math.exp(-5)/2.5). "
            "The delay is a genuine delay, so the second-shift rule applies with no "
            "hypothesis at all."
        ),
        note=(
            "The control for the two advanced-shift traps: a gate made only of "
            "refusals is passed by a library that refuses every shifted step."
        ),
    ),
    Case(
        id="transform_control_fourier_lorentzian_positive_amplitude",
        subsystem="transform",
        statement="F{2/(1+4π²x²)}(ξ) = e^{−|ξ|}",
        op=_unconditional(
            lambda: ex.fourier_transform(_int(2) / (1 + 4 * PI**2 * XX**2), XX, XI),
            {XI: 0.3},
        ),
        contract=Returns(0.7408182206817179),
        verified_by=(
            "Same residue computation as the negative-amplitude case with c = +2: "
            "e^{−0.3} = 0.7408182206817179 (math.exp(-0.3)), cross-checked with "
            "mpmath quadosc."
        ),
        note="Control: the sign fix must not turn the whole Lorentzian row into a refusal.",
    ),
    Case(
        id="transform_control_fourier_decaying_two_sided_exponential",
        subsystem="transform",
        statement="F{e^{−3|x|}}(ξ) = 6/(9+4π²ξ²)",
        op=_unconditional(lambda: ex.fourier_transform(ak.exp(-3 * ak.abs(XX)), XX, XI), {XI: 0.3}),
        contract=Returns(0.4779712002165985),
        verified_by=(
            "∫ e^{−3|x|}e^{−2πiξx} dx = 2a/(a²+4π²ξ²) with a = 3; at ξ = 0.3 that "
            "is 6/(9 + 4π²·0.09) = 0.4779712002165985 in double precision."
        ),
        note="Control for the divergent-rate refusal.",
    ),
    Case(
        id="transform_control_fourier_gaussian_is_self_dual",
        subsystem="transform",
        statement="F{e^{−πx²}}(ξ) = e^{−πξ²} in the unitary ordinary-frequency convention",
        op=_unconditional(lambda: ex.fourier_transform(ak.exp(-PI * XX**2), XX, XI), {XI: 0.5}),
        contract=Returns(0.45593812776599624),
        verified_by=(
            "The self-dual Gaussian of the unitary ordinary-frequency convention "
            "∫f(x)e^{−2πiξx}dx; at ξ = 0.5 the value is e^{−π/4} = "
            "0.45593812776599624 (math.exp(-math.pi/4))."
        ),
        note=(
            "The convention control: a 2π slip in the kernel moves this number and "
            "nothing else in the corpus would notice."
        ),
    ),
    Case(
        id="transform_control_fourier_roundtrip_recovers_the_input",
        subsystem="transform",
        statement="F⁻¹{F{e^{−x²}}}(x) = e^{−x²}",
        op=_unconditional(
            lambda: ex.inverse_fourier_transform(
                ex.fourier_transform(ak.exp(-(XX**2)), XX, XI), XI, XX
            ),
            {XX: 0.7},
        ),
        contract=Returns(0.6126263941844161),
        verified_by=(
            "e^{−0.49} = 0.6126263941844161 (math.exp(-0.49)).  The forward and "
            "inverse kernels differ only in the sign of the exponent, so a "
            "convention mismatch between the two directions leaves a stray 2π (or "
            "1/2π) here and nowhere else."
        ),
        note="Control against a forward/inverse convention mismatch.",
    ),
    Case(
        id="transform_control_inverse_laplace_delayed_sine",
        subsystem="transform",
        statement="L⁻¹{e^{−2s}/(s²+1)}(t) = θ(t−2)·sin(t−2)",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(ak.exp(-2 * S) / (S**2 + 1), S, T),
            {T: 3.0},
        ),
        contract=Returns(0.8414709848078965),
        verified_by=(
            "The second-shift rule with a genuine delay: at t = 3 the value is "
            "sin(1) = 0.8414709848078965 (math.sin(1))."
        ),
        note="Control for `transform_inverse_laplace_of_an_advance`.",
    ),
    # -- transforms: the ℚ(params) decomposition reports its confluences ------
    Case(
        id="transform_inverse_laplace_bateman_at_the_confluent_point",
        subsystem="transform",
        statement="L⁻¹{1/((s+ka)(s+ke))} at ka = ke is t·e^{−ka t}, not 0/0",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(
                1 / ((S + POOL.symbol("ka")) * (S + POOL.symbol("ke"))), S, T
            ),
            {POOL.symbol("ka"): 1.3, POOL.symbol("ke"): 1.3, T: 1.0},
        ),
        contract=RefusesOr(0.2725317930340126),
        verified_by=(
            "At ka = ke the input is the ordinary 1/(s+ka)², whose inverse is "
            "t·e^{−ka t}; at ka = ke = 1.3, t = 1 that is e^{−1.3} = "
            "0.2725317930340126 (math.exp(-1.3)).  The generic decomposition "
            "(e^{−ka t} − e^{−ke t})/(ke − ka) is 0/0 there — a true identity of "
            "rational functions that is not an identity of values."
        ),
        note=(
            "The case the ℚ(params) partial-fraction path exists for.  It passes "
            "because `ka − ke ≠ 0` is reported, not because the arithmetic happens "
            "to blow up: a caller that reads the hypothesis knows to take the "
            "confluent branch.  The E-EVAL-009 that eval_expr would raise is a "
            "second, weaker net under it."
        ),
    ),
    Case(
        id="transform_inverse_laplace_second_order_overdamped",
        subsystem="transform",
        statement="L⁻¹{1/(s²+2ζωs+ω²)} at ζ = 2 is hyperbolic, not the printed sine",
        op=_unconditional(
            lambda: ex.inverse_laplace_transform(
                1
                / (
                    S**2
                    + 2 * POOL.symbol("zeta") * POOL.symbol("omega") * S
                    + POOL.symbol("omega") ** 2
                ),
                S,
                T,
            ),
            {POOL.symbol("zeta"): 2.0, POOL.symbol("omega"): 1.0, T: 1.0},
        ),
        contract=RefusesOr(0.2139091302602793),
        verified_by=(
            "At ζ = 2, ω = 1 the characteristic roots are real: −2 ± √3.  The "
            "inverse is (e^{r₁t} − e^{r₂t})/(r₁ − r₂), which at t = 1 is "
            "(exp(-2+3**0.5) - exp(-2-3**0.5)) / (2*3**0.5) = 0.2139091302602793.  "
            "The under-damped closed form K·e^{−ζωt}sin(ω√(1−ζ²)t)/(ω√(1−ζ²)) needs "
            "ω²(1−ζ²) > 0, which is false here."
        ),
        note=(
            "Passes because ω²(1−ζ²) > 0 is reported alongside ω ≠ 0 and ζ ≠ ±1.  "
            "Without that report the printed sine is an imaginary-argument "
            "expression handed back as if it were the real answer."
        ),
    ),
    Case(
        id="transform_inverse_z_repeated_pole_confluence",
        subsystem="transform",
        statement="Z⁻¹{z/((z−a)(z−b))} at a = b is n·a^{n−1}, not (aⁿ−bⁿ)/(a−b)",
        op=_unconditional(
            lambda: ex.inverse_z_transform(
                ZZ / ((ZZ - POOL.symbol("za")) * (ZZ - POOL.symbol("zb"))), ZZ, NN
            ),
            {POOL.symbol("za"): 0.5, POOL.symbol("zb"): 0.5, NN: 3.0},
        ),
        contract=RefusesOr(0.75),
        verified_by=(
            "At a = b the transform is z/(z−a)², whose inverse is n·a^{n−1}; at "
            "a = 0.5, n = 3 that is 3·0.25 = 0.75.  Read off the geometric-series "
            "expansion of z/(z−a)² = Σ n a^{n−1} z^{−n}, independently of any "
            "partial-fraction identity."
        ),
        note=(
            "The Z-side twin of the Bateman confluence — same ℚ(params) machinery, "
            "same reported `a − b ≠ 0`."
        ),
    ),
    Case(
        id="transform_inverse_laplace_improper_rational_refuses",
        subsystem="transform",
        statement="L⁻¹{s²/(s²+1)} is δ(t) − sin t, not an ordinary function",
        op=_unconditional(lambda: ex.inverse_laplace_transform(S**2 / (S**2 + 1), S, T), {T: 1.0}),
        contract=RefusesOr(),
        verified_by=(
            "s²/(s²+1) = 1 − 1/(s²+1), and L⁻¹{1} is the Dirac δ, a distribution.  "
            "No locally integrable function has this transform, so every finite "
            "value at t = 1 is wrong — including the −sin(1) = −0.841 that dropping "
            "the polynomial part would give."
        ),
        note=(
            "Control for the refusal the changelog claims: the polynomial part is "
            "declined rather than quietly discarded, which would have produced a "
            "clean, plausible, wrong function."
        ),
    ),
    # ── validated bounds and ball arithmetic ────────────────────────────────
    #
    # The archetype for this layer is not a wrong number, it is a wrong
    # *interval*.  An enclosure that does not enclose is a false lemma every
    # downstream derivation inherits, and it is indistinguishable from a sound
    # one at the call site.
    Case(
        id="ball_indeterminate_product_still_encloses",
        subsystem="ball",
        statement="interval_eval((x^-3)^2) over x in [-3.325, -1.325] must contain 2.325^-6",
        op=_ball_encloses(
            (X ** _int(-3)) ** _int(2),
            {X: ak.ArbBall(-2.325, 1.0)},
            0.006331864446927654,
        ),
        contract=Returns(True),
        verified_by=(
            "x^-6 is even and decreasing in |x|, so on |x| in [1.325, 3.325] its range is "
            "[3.325^-6, 1.325^-6] = [7.400313e-4, 0.1848012] (mpmath, 50 digits), and the "
            "midpoint value 2.325^-6 = 6.331864e-3 lies inside it. Every sound enclosure of "
            "this expression over this box therefore contains that number: `contains` may "
            "answer False only for values outside [7.4e-4, 0.185], which this is not."
        ),
        note=(
            "Two failures meet here. x^3 by repeated ball squaring lost the sign of the box "
            "and came out straddling zero, so 1/x^3 was the indeterminate ball [0 +- inf]; "
            "squaring *that* computed a radius of 0*inf = NaN, and every comparison against a "
            "NaN endpoint is false, so `contains` answered False for every real number, the "
            "true value included. Powering from the endpoints fixes the first, and NaN no "
            "longer escapes any operation, which fixes the second. "
            "ball_removable_quotient_across_zero_still_encloses is the case that still reaches "
            "the NaN path, by a route no precision improvement can close."
        ),
    ),
    Case(
        id="ball_control_reciprocal_power_encloses",
        subsystem="ball",
        statement="interval_eval(x^-6) over x in [2, 3] must bracket [1/729, 1/64] tightly",
        op=lambda: bool(
            ak.interval_eval(X ** _int(-6), {X: ak.ArbBall(2.5, 0.5)}).lo
            <= (1.0 / 729.0) * (1 + 1e-12)
            and ak.interval_eval(X ** _int(-6), {X: ak.ArbBall(2.5, 0.5)}).hi >= 1.0 / 64.0
        ),
        contract=Returns(True),
        verified_by=(
            "x^-6 is decreasing on [2, 3], so its range there is exactly [3^-6, 2^-6] = "
            "[1/729, 1/64], attained at the two endpoints. 1/64 is exact in binary, so the "
            "upper bound is compared with no slack; 1/729 is not, and the returned enclosure "
            "is tighter than an f64 can express, so that comparison carries a 1e-12 relative "
            "slack -- far below the 1e-3 width of the interval and far above the 1e-16 the "
            "f64 literal is off by."
        ),
        note=(
            "Also the tightness control. The enclosure used to be [-inf, inf] here, because "
            "x^6 computed by repeated ball squaring straddled zero and its reciprocal was "
            "therefore unbounded: sound, and completely useless."
        ),
    ),
    Case(
        id="ball_reciprocal_across_zero_is_not_a_bound",
        subsystem="ball",
        statement="1/x over x in [-1, 1] has no finite enclosure -- 0 is in the box",
        op=_ball_upper(_int(1) / X, {X: ak.ArbBall(0.0, 1.0)}),
        contract=RefusesOr(),
        verified_by=(
            "1/x is unbounded on every neighbourhood of 0 and undefined at it, so no finite "
            "interval contains its range on [-1, 1]. Any finite number returned here would be "
            "a bound that is not one."
        ),
        note=(
            "Passes via a weak refusal: the ball comes back as [-inf, inf] rather than as an "
            "exception. That is the honest 'no information' answer for ball arithmetic, but a "
            "caller has to look at the value to notice."
        ),
    ),
    Case(
        id="ball_removable_quotient_across_zero_still_encloses",
        subsystem="ball",
        statement="sin(x)/x over x in [-0.5, 0.5] must not claim to exclude 0.9588510772",
        op=_ball_encloses(ak.sin(X) / X, {X: ak.ArbBall(0.0, 0.5)}, 0.958851077208406),
        contract=Returns(True),
        verified_by=(
            "sin(0.5)/0.5 = 0.95885107720840600... (mpmath, 50 digits) is the value the "
            "expression takes at a point of the ball, so no sound enclosure over that ball can "
            "exclude it. Pointwise ball arithmetic cannot resolve the 0/0 at the centre, so "
            "the only correct answers are 'the whole line' or a refusal -- never 'that value "
            "is outside'."
        ),
        note=(
            "Same NaN mechanism as ball_indeterminate_product_still_encloses reached from the "
            "other side: 1/x is indeterminate and sin(x) has midpoint 0, so the product's "
            "radius was |0| * inf = NaN."
        ),
    ),
    Case(
        id="ball_sqrt_of_a_ball_straddling_zero_refuses",
        subsystem="ball",
        statement="sqrt of the ball [-1, 1] is not real -- refuse rather than take the real part",
        op=lambda: float(ak.ArbBall(0.0, 1.0).sqrt().lo),
        contract=RefusesOr(),
        verified_by=(
            "sqrt is undefined on the negative half of [-1, 1], so no real enclosure of it "
            "exists there. The plausible wrong answer is [0, 1]: the image of the part of the "
            "ball where sqrt happens to be defined, which silently shrinks the domain."
        ),
    ),
    Case(
        id="ball_floor_straddling_an_integer_covers_both_values",
        subsystem="ball",
        statement="floor over x in [0.75, 1.25] takes both 0 and 1, so the ball is >= 1 wide",
        op=lambda: float(
            ak.interval_eval(ak.floor(X), {X: ak.ArbBall(1.0, 0.25)}).hi
            - ak.interval_eval(ak.floor(X), {X: ak.ArbBall(1.0, 0.25)}).lo
        ),
        contract=Returns(1.0, tol=1e-12),
        verified_by=(
            "floor(0.75) = 0 and floor(1.25) = 1, so the range is exactly {0, 1} and the "
            "narrowest sound interval is [0, 1], of width 1. Evaluating the midpoint and "
            "adding the input radius -- the Lipschitz shortcut that works for sin and exp -- "
            "would give width 0.5 around floor(1) = 1, an interval that misses 0 entirely."
        ),
    ),
    Case(
        id="ball_unsupported_primitive_refuses",
        subsystem="ball",
        statement="sign(x) has no ball rule; interval_eval must refuse rather than use f64",
        op=lambda: float(ak.interval_eval(ak.sign(X), {X: ak.ArbBall(0.5, 0.25)}).hi),
        contract=RefusesOr(),
        verified_by=(
            "capabilities() reports numeric_ball = False for `sign`. The dangerous answer is "
            "the f64 one, sign(0.5) = 1 as an exact ball: right on this box, wrong on any box "
            "straddling 0, with nothing in the result to say which case the caller got."
        ),
    ),
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
    Case(
        id="eval_large_integer_literal_is_the_nearest_double",
        subsystem="evaluation",
        statement="10^30 evaluates to the nearest double, 1e30, not the one below it",
        op=lambda: float(ak.eval_expr(_int(10**30) + _int(0) * X, {X: 0.0})),
        contract=Returns(1e30, tol=0.0),
        verified_by=(
            "10^30 = 2^30·5^30 needs 70 significant bits, so it is not exact in binary64. "
            "Python's int->float is correctly rounded (round-half-even) and gives 1e30; "
            "truncation towards zero gives 9.999999999999999e29, one ulp low."
        ),
        note=(
            "One ulp, but biased: truncation never rounds up, so a sum of large exact "
            "integers drifts in one direction. It is also a place two evaluators disagreed "
            "- emit_c_expr wrote the exact decimal and let the C compiler round it."
        ),
    ),
    Case(
        id="eval_rational_literal_is_the_nearest_double",
        subsystem="evaluation",
        statement="2/5 evaluates to 0.4, the nearest double, under every mode",
        op=lambda: float(ak.evaluate(_rat(2, 5), {}, mode="f64").value),
        contract=Returns(0.4, tol=0.0),
        verified_by=(
            "float(Fraction(2, 5)) = 0.4 exactly (Python rounds correctly). The double "
            "below it is 0.39999999999999997, which is what rounding towards zero returns."
        ),
    ),
    Case(
        id="eval_rational_with_both_parts_past_the_float_range",
        subsystem="evaluation",
        statement="(3·10^400+1)/(2·10^400) is 1.5, not an inf/inf NaN",
        op=lambda: float(
            ak.eval_expr(POOL.rational(3 * 10**400 + 1, 2 * 10**400) + _int(0) * X, {X: 0.0})
        ),
        contract=Returns(1.5, tol=1e-15),
        verified_by=(
            "float(Fraction(3*10**400+1, 2*10**400)) = 1.5. The parts are coprime, so the "
            "fraction is in lowest terms and both of them overflow binary64 on their own."
        ),
        note=(
            "The evaluators split three ways here: `evaluate(mode='f64')` was right, "
            "`eval_expr` refused with E-EVAL-009 (a weak refusal produced by NaN), and "
            "compile_expr/numpy_eval returned a bare NaN."
        ),
    ),
    Case(
        id="eval_control_small_rational_literal",
        subsystem="evaluation",
        statement="1/2 evaluates to 0.5 — the literals that were always exact stay exact",
        op=lambda: float(ak.eval_expr(_rat(1, 2), {})),
        contract=Returns(0.5, tol=0.0),
        verified_by="1/2 is exact in binary64. Control for the two rounding cases above.",
    ),
    Case(
        id="codegen_horner_preserves_coefficients_past_i64",
        subsystem="codegen",
        statement="horner((x+1)^80) is the same polynomial: 2^80 at x = 1",
        op=lambda: float(ak.eval_expr(ak.horner((X + _int(1)) ** _int(80), X), {X: 1.0})),
        contract=Returns(float(2**80), tol=0.0),
        verified_by=(
            "(1+1)^80 = 2^80 = 1208925819614629174706176, exactly representable in binary64. "
            "C(80,40) = 1.075e23 exceeds i64::MAX = 9.22e18, so a coefficient vector taken "
            "as i64 wraps modulo 2^64 in the middle of the row."
        ),
        note=(
            "horner is a *rewrite*; it promises the same polynomial. Going through "
            "coefficients_i64 made that promise false with no error and no flag, and "
            "returned 2.12e20 here."
        ),
    ),
    Case(
        id="codegen_control_horner_small_coefficients",
        subsystem="codegen",
        statement="horner(x²+2x+1) = 16 at x = 3 — the ordinary case is unchanged",
        op=lambda: float(ak.eval_expr(ak.horner(X**2 + _int(2) * X + _int(1), X), {X: 3.0})),
        contract=Returns(16.0, tol=0.0),
        verified_by="(3+1)² = 16. Control for codegen_horner_preserves_coefficients_past_i64.",
    ),
    Case(
        id="codegen_emit_c_preserves_coefficients_past_i64",
        subsystem="codegen",
        statement="emit_c of (2^70+3)·x + 1 keeps the coefficient rather than wrapping it to 3",
        op=lambda: _largest_c_literal(ak.emit_c(_int(2**70 + 3) * X + _int(1), X)),
        contract=Returns(1.1805916207174113e21, tol=0.0),
        verified_by=(
            "float(2**70 + 3) = 1.1805916207174113e21 (Python rounds correctly). "
            "(2**70 + 3) % 2**64 = 3, which is what the emitted C used to contain."
        ),
    ),
    Case(
        id="codegen_emit_c_expr_matches_the_interpreter_on_a_rational",
        subsystem="codegen",
        statement="the C emitted for (2/5)·x carries the same double the interpreter uses",
        op=lambda: _largest_c_literal(ak.emit_c_expr(_rat(2, 5) * X, [X], var_names=["x"])),
        contract=Returns(0.4, tol=0.0),
        verified_by=(
            "float(Fraction(2, 5)) = 0.4. Emitting 0.39999999999999997 instead would make "
            "the compiled artefact disagree with eval_expr by an ulp on the same expression."
        ),
    ),
    Case(
        id="codegen_emit_c_expr_emits_compilable_c_for_a_non_finite_literal",
        subsystem="codegen",
        statement="a non-finite constant emits the <math.h> macro, not the bare token `NaN`",
        op=lambda: _c_body(ak.emit_c_expr(POOL.float(float("inf")) * X, [X], var_names=["x"])),
        contract=Returns("return (x * INFINITY);"),
        verified_by=(
            "C has no identifiers `inf` or `NaN`; INFINITY and NAN are the <math.h> macros, "
            "and the emitted code already documents that <math.h> is required. Checked by "
            "compiling both forms with cc: the bare token is an undeclared identifier."
        ),
        note="emit_c_expr reported success while returning a function that does not compile.",
    ),
    Case(
        id="codegen_stablehlo_preserves_coefficients_past_i64",
        subsystem="codegen",
        statement="to_stablehlo of 2^70·x + 1 does not lower the coefficient to the constant 0",
        op=lambda: _largest_mlir_constant(ak.to_stablehlo(_int(2**70) * X + _int(1), [X])),
        contract=Returns(1.1805916207174113e21, tol=0.0),
        verified_by=(
            "float(2**70) = 1.1805916207174113e21. The emitter read the coefficient with "
            "to_i64().unwrap_or(0), so the module it produced computed x·0 + 1 = 1 — valid "
            "MLIR, no diagnostic, a different function."
        ),
    ),
    Case(
        id="codegen_stablehlo_emits_parseable_float_literals",
        subsystem="codegen",
        statement="every emitted MLIR constant carries the decimal point the grammar requires",
        op=lambda: _mlir_literals_missing_a_decimal_point(
            ak.to_stablehlo(_int(10**30) * X + _rat(1, 10**30), [X])
        ),
        contract=Returns(0),
        verified_by=(
            "MLIR's float-literal grammar requires the point, and mlir-opt 15.0.4 was run "
            "on `dense<1e30>`, `dense<1e16>` and `dense<5e-324>`: each is rejected with "
            "`error: expected '>'`, while `dense<1.0e30>` and `dense<0.4>` parse."
        ),
        note=(
            "to_stablehlo documents its output as valid input to mlir-opt / XLA. Rust's "
            "shortest-round-trip float format drops the point whenever the mantissa is a "
            "single digit, so the module did not parse and the call still reported success."
        ),
    ),
    Case(
        id="codegen_control_stablehlo_small_coefficient",
        subsystem="codegen",
        statement="to_stablehlo of 7·x + 1 carries the coefficient 7",
        op=lambda: _largest_mlir_constant(ak.to_stablehlo(_int(7) * X + _int(1), [X])),
        contract=Returns(7.0, tol=0.0),
        verified_by="7 is exact in binary64. Control for the i64-overflow case above.",
    ),
    Case(
        id="codegen_compiled_fn_agrees_with_the_interpreter_bit_for_bit",
        subsystem="codegen",
        statement="eval_expr, evaluate(f64), compile_expr and numpy_eval give one value, not four",
        op=_evaluators_disagreeing_on,
        contract=Returns(0),
        verified_by=(
            "Not a mathematical claim but a consistency one: the four paths run the same "
            "arithmetic on the same DAG, so any difference means at least one is wrong. "
            "The count is of (expression, point) pairs where the four do not agree exactly."
        ),
        note=(
            "This is the case that catches a literal- or lowering-level divergence "
            "regardless of which side is wrong; the per-value cases above pin down which."
        ),
    ),
    # -----------------------------------------------------------------------
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
    Case(
        id="codegen_compiled_fn_does_not_invent_a_value_at_a_pole",
        subsystem="codegen",
        statement="compile_expr(1/x)([0.0]) must not return a finite number",
        op=lambda: float(ak.compile_expr(_int(1) / X, [X])([0.0])),
        contract=RefusesOr(),
        verified_by="1/0 is undefined; no real number is the value of 1/x at 0.",
        note=(
            "Passes via a *weak* refusal: the compiled path returns ±inf/NaN where eval_expr "
            "raises E-EVAL-009. That asymmetry is documented (CompiledFn has no error "
            "channel) and is the reason this case exists rather than a Raises() one."
        ),
    ),
    # -----------------------------------------------------------------------
    # Probability distributions and expectations
    #
    # Every expected value below is mpmath at 40–50 dps of the *defining*
    # integral or sum, cross-checked against sympy.stats where sympy has the
    # law.  None was read off alkahest.
    #
    # Not here, deliberately: the Black–Scholes derivation
    # `E[max(S-K,0)]` under a LogNormal.  It is correct — it agrees with
    # `quad((s-K)·p(s), [K, ∞))` to 1e-10 and satisfies put–call parity — but it
    # takes ~2.4 s, because the payoff split, the log-normal reduction and two
    # `erf` antiderivatives all run inside it.  That is a timeout hazard on a
    # gate that runs on every pull request, so it lives in
    # `alkahest-core/src/prob/tests.rs` instead.  See README § Cost discipline.
    # -----------------------------------------------------------------------
    Case(
        id="prob_lognormal_mean_is_not_exp_mu",
        subsystem="probability",
        statement="E[X] for X ~ LogNormal(1/5, 1/2) is e^{mu+sigma^2/2}, not e^{mu}",
        op=_prob_value(lambda: ex.LogNormal(_rat(1, 5), _rat(1, 2)).mean()),
        contract=Returns(1.3840306459807514, tol=1e-12),
        verified_by="mpmath 50 dps: quad(x·lognormal_pdf(x, 0.2, 0.5), [0, 1, 10, inf]) = "
        "1.3840306459807514212, equal to exp(0.2 + 0.125). The trap is that "
        "mu is *not* the mean of X — reading it as one gives e^0.2 = 1.2214, "
        "which is 12% low and entirely plausible.",
    ),
    Case(
        id="prob_lognormal_variance_is_not_sigma_squared",
        subsystem="probability",
        statement="Var[X] for X ~ LogNormal(1/5, 1/2) is (e^{s^2}-1)e^{2m+s^2}, not s^2",
        op=_prob_value(lambda: ex.LogNormal(_rat(1, 5), _rat(1, 2)).variance()),
        contract=Returns(0.5440622821430536, tol=1e-12),
        verified_by="mpmath 50 dps: quad((x-m)^2·lognormal_pdf(x, 0.2, 0.5), [0, 1, 10, inf]) "
        "with m = e^{0.325} gives 0.54406228214305359365. Reading sigma as the "
        "standard deviation of X gives 0.25 — a factor of 2.2 out.",
    ),
    Case(
        id="prob_gamma_is_shape_scale_not_shape_rate",
        subsystem="probability",
        statement="E[X] for X ~ Gamma(7/2, 4/5) under the shape-scale convention is k*theta",
        op=_prob_value(lambda: ex.Gamma(_rat(7, 2), _rat(4, 5)).mean()),
        contract=Returns(2.8, tol=1e-12),
        verified_by="mpmath: quad(x·gamma_pdf(x, k=3.5, theta=0.8), [0, 1, 10, inf]) = 2.8 = "
        "k·theta. The shape-*rate* convention — which SciPy's `gamma(a, scale=)` "
        "and R's `rgamma(rate=)` both also expose — would give k/theta = 4.375. "
        "Both are clean numbers and neither looks wrong on its own.",
    ),
    Case(
        id="prob_normal_fourth_moment_is_three_sigma_fourth",
        subsystem="probability",
        statement="E[X^4] for X ~ Normal(0, 3/2) is 3*sigma^4, not sigma^4",
        op=_prob_value(lambda: ex.Normal(_int(0), _rat(3, 2)).moment(4)),
        contract=Returns(15.1875, tol=1e-12),
        verified_by="Isserlis/Wick: E[X^4] = 3 sigma^4 for a centred Gaussian, = 3·(1.5)^4 = "
        "15.1875; confirmed by mpmath quad(x^4·normal_pdf(x, 0, 1.5), "
        "[-inf, 0, inf]). Dropping the double-factorial gives 5.0625.",
    ),
    Case(
        id="prob_poisson_third_moment_is_the_touchard_polynomial",
        subsystem="probability",
        statement="E[X^3] for X ~ Poisson(12/5) is lam^3 + 3 lam^2 + lam, not lam^3",
        op=_prob_value(lambda: ex.Poisson(_rat(12, 5)).moment(3)),
        contract=Returns(33.504, tol=1e-12),
        verified_by="Touchard: E[X^n] = sum_r S(n,r) lam^r, so E[X^3] = lam^3+3lam^2+lam = "
        "33.504 at lam = 2.4; confirmed by mpmath nsum(k^3·poisson_pmf(k), "
        "[0, inf]) = 33.504. A Poisson has all its cumulants equal to lam, which "
        "makes 'the third moment is lam^3' a tempting and wrong shortcut (13.824).",
    ),
    Case(
        id="prob_binomial_second_moment_is_not_the_mean_squared",
        subsystem="probability",
        statement="E[X^2] for X ~ Binomial(5, 7/20) is np(1-p) + (np)^2, not (np)^2",
        op=_prob_value(lambda: ex.Binomial(_int(5), _rat(7, 20)).moment(2)),
        contract=Returns(4.2, tol=1e-12),
        verified_by="Exact sum over the support: sum_k k^2 C(5,k) p^k (1-p)^{5-k} at p = 0.35 "
        "is 4.2, and Var + mean^2 = 5(0.35)(0.65) + (1.75)^2 = 1.1375 + 3.0625 = "
        "4.2. Confusing E[X^2] with E[X]^2 gives 3.0625.",
    ),
    Case(
        id="prob_uniform_third_moment_is_not_the_cube_of_the_mean",
        subsystem="probability",
        statement="E[X^3] for X ~ Uniform(-2, 3) is (b^4-a^4)/(4(b-a)), not ((a+b)/2)^3",
        op=_prob_value(lambda: ex.Uniform(_int(-2), _int(3)).moment(3)),
        contract=Returns(3.25, tol=1e-12),
        verified_by="(b^4 - a^4)/(4(b-a)) = (81-16)/20 = 3.25, matching mpmath "
        "quad(x^3/(b-a), [-2, 3]). E[X]^3 = 0.5^3 = 0.125 — an asymmetric "
        "interval makes the two differ by a factor of 26.",
    ),
    Case(
        id="prob_control_erlang_cdf_closes",
        subsystem="probability",
        statement="P(X <= 12/5) for X ~ Gamma(3, 4/5) — integer shape, so the Erlang sum closes",
        op=_prob_value(lambda: ex.Gamma(_int(3), _rat(4, 5)).cdf(PROB_X), {PROB_X: 2.4}),
        contract=Returns(0.5768099188731565, tol=1e-12),
        verified_by="mpmath 50 dps: quad(gamma_pdf(t, k=3, theta=0.8), [0, 2.4]) = "
        "0.57680991887315648468, equal to the Erlang form "
        "1 - e^{-u} sum_{j<3} u^j/j! at u = 3. The control for the "
        "non-integer-shape refusal below: the family that *does* close must not "
        "be swept up in it.",
    ),
    Case(
        id="prob_gamma_cdf_at_a_non_integer_shape_refuses",
        subsystem="probability",
        statement="P(X <= x) for Gamma with a symbolic shape needs the incomplete gamma",
        op=lambda: ex.Gamma(POOL.symbol("kshape"), POOL.symbol("kscale")).cdf(PROB_X),
        contract=Raises("E-PROB-004"),
        verified_by="F(x) = P(k, x/theta), the regularised lower incomplete gamma "
        "(DLMF 8.2.4). It is not elementary for general k, and alkahest has no "
        "primitive for it, so there is nothing to return. Emitting the Erlang "
        "finite sum anyway would be wrong for every non-integer k.",
    ),
    Case(
        id="prob_cdf_below_the_support_is_zero_not_the_formula",
        subsystem="probability",
        statement="P(X <= -5) for X ~ Gamma(3, 4/5) is 0; the Erlang formula there gives -7396.87",
        op=_prob_value(lambda: ex.Gamma(_int(3), _rat(4, 5)).cdf(_int(-5))),
        contract=Returns(0.0, tol=1e-15),
        verified_by="A Gamma puts no mass below 0, so P(X <= -5) = 0 by definition. The "
        "in-support closed form 1 - e^{-u} sum_{j<3} u^j/j! at u = -6.25 "
        "evaluates to -7396.8706522947593 — a *negative probability*, returned "
        "with no complaint. Value computed by hand from the Erlang sum and "
        "confirmed with mpmath.",
    ),
    Case(
        id="prob_cdf_above_the_support_is_one_not_the_formula",
        subsystem="probability",
        statement="P(X <= 7) for X ~ Uniform(-2, 3) is 1; the linear formula there gives 1.8",
        op=_prob_value(lambda: ex.Uniform(_int(-2), _int(3)).cdf(_int(7))),
        contract=Returns(1.0, tol=1e-15),
        verified_by="All of a Uniform(-2, 3)'s mass is below 7, so the probability is 1. "
        "(x-a)/(b-a) at x = 7 is 9/5 = 1.8 — a probability greater than one. "
        "Both values are one line of arithmetic from the definition.",
    ),
    Case(
        id="prob_control_cdf_inside_the_support_is_the_formula",
        subsystem="probability",
        statement="P(X <= 1/2) for X ~ Uniform(-2, 3) is 1/2 — the control for the two clamps",
        op=_prob_value(lambda: ex.Uniform(_int(-2), _int(3)).cdf(_rat(1, 2))),
        contract=Returns(0.5, tol=1e-14),
        verified_by="(0.5 - (-2))/(3 - (-2)) = 2.5/5 = 0.5. Without this the two clamp cases "
        "would be passed by an implementation that returned 0 or 1 for every "
        "argument.",
    ),
    Case(
        id="prob_quantile_outside_zero_one_is_refused",
        subsystem="probability",
        statement="Uniform(-2, 3).quantile(2) has no meaning; the formula returns 8",
        op=lambda: ex.Uniform(_int(-2), _int(3)).quantile(_int(2)),
        contract=Raises("E-PROB-001"),
        verified_by="F^-1 is defined on [0, 1]. a + p(b-a) at p = 2 is -2 + 2·5 = 8, a point "
        "outside the support [−2, 3] that the quantile is supposed to name. "
        "There is no probability 2, so no value is correct here.",
    ),
    Case(
        id="prob_control_exponential_quantile_closes",
        subsystem="probability",
        statement="Exponential(7/4).quantile(9/10) = -log(1-p)/lambda",
        op=_prob_value(lambda: ex.Exponential(_rat(7, 4)).quantile(_rat(9, 10))),
        contract=Returns(1.3157629102823118, tol=1e-12),
        verified_by="-log(1 - 0.9)/1.75 = log(10)/1.75 = 1.3157629102823118194 (mpmath, "
        "50 dps), and substituting it back gives 1 - e^{-1.75 q} = 0.9 exactly. "
        "The control for the out-of-range refusal above.",
    ),
    Case(
        id="prob_normal_quantile_refuses_for_want_of_erfinv",
        subsystem="probability",
        statement="The normal quantile needs erf^-1, which is not a registered primitive",
        op=lambda: ex.Normal(_int(0), _int(1)).quantile(_rat(9, 10)),
        contract=Raises("E-PROB-004"),
        verified_by="Phi^-1(p) = sqrt(2)·erf^-1(2p-1) (DLMF 7.17). erf^-1 has no elementary "
        "closed form and alkahest does not implement it, so the honest answer "
        "is that there is none to give — not a series truncation dressed as one.",
    ),
    Case(
        id="prob_characteristic_function_convention_is_e_itx",
        subsystem="probability",
        statement="phi_Normal(t) = e^{i mu t - sigma^2 t^2/2}; a 2pi or a sign slip is invisible",
        op=_prob_phi(
            lambda: ex.Normal(_rat(7, 10), _rat(11, 10)).characteristic_function(_rat(13, 10))
        ),
        contract=Returns((0.2207720571448093, 0.2839944144298271), tol=1e-12),
        verified_by="mpmath 50 dps, from the definition phi(t) = int p(x)e^{itx}dx: "
        "quad(normal_pdf(x, 0.7, 1.1)·cos(1.3x), [-inf, 0, inf]) = "
        "0.22077205714480930348 and the sin integral = 0.28399441442982708525, "
        "equal to e^{i(0.7)(1.3) - (1.21)(1.69)/2}. alkahest's `fourier_transform` "
        "is unitary ordinary-frequency (e^{-2 pi i x xi}), so phi(t) = F{p}(-t/2pi): "
        "a 2 pi *and* a sign apart. Both near-misses were computed and are "
        "detectably different — dropping the 2 pi gives Re = 2.49e-18 rather than "
        "0.2208, and flipping the sign negates the imaginary part — so this "
        "case can actually fail.",
    ),
    Case(
        id="prob_moment_from_the_characteristic_function_agrees_with_the_table",
        subsystem="probability",
        statement="phi''(0) = -E[X^2] for Binomial(5, 7/20): two routes, no shared closed form",
        op=_moment_via_charfun(lambda: ex.Binomial(_int(5), _rat(7, 20)), 2),
        contract=Returns(4.2, tol=1e-10),
        verified_by="Exact sum over the support: sum_k k^2 C(5,k) p^k (1-p)^{5-k} at p = 0.35 "
        "is 4.2 = np(1-p) + (np)^2. Reached here the other way — differentiate "
        "phi(t) = (1-p+p e^{it})^5 twice and evaluate at t = 0, using "
        "phi^(n)(0) = i^n E[X^n] (Feller II, XV.4) — so the moment table and the "
        "phi table have to agree without sharing a line of code. It is "
        "deliberately the same number as "
        "`prob_binomial_second_moment_is_not_the_mean_squared`: that case pins "
        "the table, this one pins the two routes to each other, and a "
        "convention slip in phi shows up here as a wrong moment.",
    ),
    Case(
        id="prob_lognormal_characteristic_function_refuses",
        subsystem="probability",
        statement="phi for a log-normal has no closed form at all",
        op=lambda: ex.LogNormal(_int(0), _int(1)).characteristic_function(PROB_T),
        contract=Raises("E-PROB-004"),
        verified_by="The log-normal moment series sum (it)^n e^{n mu + n^2 sigma^2/2}/n! "
        "diverges for every t != 0 (the moments grow like e^{n^2 sigma^2/2}, "
        "faster than n!), and the log-normal is the standard example of a "
        "distribution not determined by its moments — Heyde 1963, J. London "
        "Math. Soc. 38. There is no elementary or standard-special-function "
        "expression to return.",
    ),
    Case(
        id="prob_control_normal_characteristic_function_closes",
        subsystem="probability",
        statement="phi for a normal is e^{i mu t - sigma^2 t^2/2} — the control for the refusal",
        op=_prob_phi(lambda: ex.Normal(_int(0), _int(1)).characteristic_function(_int(1))),
        contract=Returns((0.6065306597126334, 0.0), tol=1e-12),
        verified_by="phi_{N(0,1)}(1) = e^{-1/2} = 0.60653065971263342360, real because a "
        "centred normal is symmetric; mpmath quad(normal_pdf(x)·cos x, "
        "[-inf, 0, inf]) agrees and the sin integral is 0. A gate made only of "
        "refusals is passed by a module that refuses every characteristic "
        "function.",
    ),
    Case(
        id="prob_divergent_expectation_is_refused_not_evaluated",
        subsystem="probability",
        statement="E[e^{X^2}] under Normal(0,1) diverges: the integrand is e^{x^2/2}",
        op=lambda: ex.expectation(
            POOL.func("exp", [PROB_X**2]), PROB_X, ex.Normal(_int(0), _int(1))
        ),
        contract=Raises("E-PROB-006"),
        verified_by="The integrand is e^{x^2/2}/sqrt(2 pi), which grows without bound, so the "
        "integral diverges and E[e^{X^2}] does not exist. Formally completing "
        "the square gives -1/sqrt(-1) style nonsense that simplifies to a clean "
        "finite number, which is exactly the trap: the MGF of a chi-squared "
        "exists only for t < 1/2 and this is t = 1.",
    ),
    Case(
        id="prob_control_normal_mgf_at_one_converges",
        subsystem="probability",
        statement="E[e^X] under Normal(0,1) is e^{1/2} — the convergent neighbour of E[e^{X^2}]",
        op=_prob_value(
            lambda: ex.expectation(POOL.func("exp", [PROB_X]), PROB_X, ex.Normal(_int(0), _int(1)))
        ),
        contract=Returns(1.6487212707001282, tol=1e-11),
        verified_by="The Gaussian MGF is E[e^{tX}] = e^{mu t + sigma^2 t^2/2}, so at t = 1 it "
        "is sqrt(e) = 1.6487212707001281468; mpmath quad(e^x·normal_pdf(x), "
        "[-inf, 0, inf]) agrees to 20 digits. Without this the divergence case "
        "would be passed by a module that refused every exponential "
        "expectation.",
    ),
    Case(
        id="prob_product_of_two_variates_needs_a_joint_law",
        subsystem="probability",
        statement="E[A*B] is not determined by the marginals of A and B",
        op=lambda: ex.expectation_affine(
            POOL.symbol("pv_a") * POOL.symbol("pv_b"),
            [
                (POOL.symbol("pv_a"), ex.Normal(_int(0), _int(1))),
                (POOL.symbol("pv_b"), ex.Normal(_int(0), _int(1))),
            ],
        ),
        contract=Raises("E-PROB-002"),
        verified_by="E[AB] = E[A]E[B] + Cov(A,B), and the covariance is not a function of the "
        "two marginals: A ~ N(0,1) with B = A gives E[AB] = 1, while A ~ N(0,1) "
        "with B = -A gives -1, and both have the same marginals. Answering 0 "
        "(the independent case) would be right for one joint law and wrong for "
        "uncountably many others, with nothing in the call to say which was "
        "assumed.",
    ),
    Case(
        id="prob_control_linearity_of_expectation_needs_no_independence",
        subsystem="probability",
        statement="E[3X + 2Y + 5] = 3E[X] + 2E[Y] + 5 for any joint law",
        op=_prob_value(
            lambda: ex.expectation_affine(
                3 * POOL.symbol("pv_a") + 2 * POOL.symbol("pv_b") + _int(5),
                [
                    (POOL.symbol("pv_a"), ex.Normal(_int(1), _int(2))),
                    (POOL.symbol("pv_b"), ex.Exponential(_int(4))),
                ],
            )
        ),
        contract=Returns(8.5, tol=1e-12),
        verified_by="Linearity of expectation holds for every joint law, dependent or not: "
        "3(1) + 2(1/4) + 5 = 8.5, using E[Normal(1,2)] = 1 and "
        "E[Exponential(4)] = 1/4 (rate parametrisation). The control for the "
        "product refusal — the distinction being that linearity needs no "
        "independence and a covariance does.",
    ),
    Case(
        id="prob_a_negative_scale_is_not_a_distribution",
        subsystem="probability",
        statement="Normal(mu, -1) does not exist; sigma > 0 is part of the definition",
        op=lambda: ex.Normal(POOL.symbol("pmu"), _int(-1)),
        contract=Raises("E-PROB-001"),
        verified_by="The density e^{-(x-mu)^2/(2 sigma^2)}/(sigma sqrt(2 pi)) integrates to "
        "-1 at sigma = -1: sigma^2 is blind to the sign but the normalising "
        "constant is not. Every moment then comes back with the wrong sign and "
        "nothing in the arithmetic complains.",
    ),
    Case(
        id="prob_control_a_symbolic_scale_is_carried_not_decided",
        subsystem="probability",
        statement="Normal(mu, sigma) with symbolic sigma is accepted, carrying sigma > 0",
        op=lambda: len(ex.Normal(POOL.symbol("pmu"), POOL.symbol("psigma")).constraints()),
        contract=Returns(1),
        verified_by="A symbolic sigma cannot be decided either way, so refusing it would be a "
        "false refusal — the distribution is perfectly well defined for every "
        "sigma > 0. The constraint is returned as a predicate for the caller to "
        "discharge; there is exactly one of them for a Normal. The control that "
        "the constructor check above is a *decision* and not a blanket refusal.",
    ),
    Case(
        id="prob_symbolic_cdf_discloses_that_it_is_the_in_support_branch",
        subsystem="probability",
        statement="F(x) for symbolic x is the in-support branch; at x = -5 it gives -7396.87",
        op=_prob_unconditional(lambda: ex.Gamma(_int(3), _rat(4, 5)).cdf(PROB_X), {PROB_X: -5.0}),
        contract=RefusesOr(0.0),
        verified_by="A Gamma puts no mass below 0, so P(X <= -5) = 0 by definition. The "
        "Erlang closed form 1 - e^{-u} sum_{j<3} u^j/j! at u = -6.25 is "
        "-7396.8706522947593 — a negative probability. With a symbolic "
        "argument alkahest cannot place x, so it publishes the hypothesis "
        "'x > 0' on `prob_side_conditions()`; this case scores a disclosed "
        "answer as a refusal, which is what makes it able to fail if the "
        "disclosure is ever dropped.",
        note="The companion `prob_cdf_below_the_support_is_zero_not_the_formula` covers "
        "the decidable argument, where alkahest returns the exact 0 instead of "
        "disclosing a hypothesis. Both routes have to hold.",
    ),
    Case(
        id="prob_characteristic_function_is_not_silently_realified",
        subsystem="probability",
        statement="phi_Normal(1.3) is complex; a real evaluator must refuse, not return Re",
        op=_phi_real_mode(
            lambda: ex.Normal(_rat(7, 10), _rat(11, 10)).characteristic_function(_rat(13, 10)),
            {},
        ),
        contract=RefusesOr(),
        verified_by="phi(t) = e^{i mu t - sigma^2 t^2/2} = 0.22077205714480930348 + "
        "0.28399441442982708525i (mpmath 50 dps, from int p(x)e^{itx}dx). No "
        "real number is its value. Handing back the real part alone would be a "
        "clean number of the right magnitude that has silently discarded the "
        "imaginary part — which for a non-symmetric law carries the mean, since "
        "phi'(0) = i·E[X].",
        note="The control is `prob_characteristic_function_convention_is_e_itx`, which "
        "asks the same question through `evaluate(..., mode='complex')` and gets "
        "both components. A gate made only of this case would be passed by a "
        "module whose characteristic functions could not be evaluated at all.",
    ),
]


#: Fast lookup by id.
CASES_BY_ID: dict[str, Case] = {c.id: c for c in CASES}

if len(CASES_BY_ID) != len(CASES):  # pragma: no cover - corpus authoring guard
    seen: set[str] = set()
    dupes = sorted({c.id for c in CASES if c.id in seen or seen.add(c.id)})  # type: ignore[func-returns-value]
    raise RuntimeError(f"duplicate case ids in the silent-error corpus: {dupes}")
