"""Silent-error cases for real qe.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

import alkahest as ak
from contracts import Case, RefusesOr, Returns

from ._shared import POOL, X, Y, _int


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
]
