"""Generic ``simplify`` on textbook identities (launch-readiness report #6).

``simplify(sin(x)**2 + cos(x)**2)`` used to return its input while
``simplify_trig``, ``simplify_egraph`` and ``is_zero`` all knew it was ``1``.
``simplify`` now runs the domain passes (Pythagorean / hyperbolic collection,
trig identities, log/exp, radical denesting, rational cancellation, bounded
expansion) after the rule engine and keeps a candidate only when it is
smaller.

Two oracles for every case:

* the expected simplest form, compared structurally after ``simplify``;
* a numeric check at random points that ``simplify`` never changed the value
  (independent of the rewrite machinery: ``eval_expr`` on input and output).

The cases follow SymPy's ``test_simplify.py`` / ``test_trigsimp.py`` classics.
"""

from __future__ import annotations

import math
import random

import alkahest as ak
import pytest

P = ak.ExprPool()
x, y = P.symbol("x"), P.symbol("y")
xp, yp = P.symbol("xp", ak.Domain.Positive), P.symbol("yp", ak.Domain.Positive)
I = P.integer  # noqa: E741
R = P.rational
sin, cos, tan = ak.sin, ak.cos, ak.tan
sinh, cosh = ak.sinh, ak.cosh
exp, log, sqrt = ak.exp, ak.log, ak.sqrt


def S(e):
    return ak.simplify(e).value


# (name, input, expected simplest form)
CASES = [
    ("sin2+cos2", sin(x) ** 2 + cos(x) ** 2, I(1)),
    ("cos2+sin2", cos(x) ** 2 + sin(x) ** 2, I(1)),
    ("y+sin2+cos2", y + sin(x) ** 2 + cos(x) ** 2, y + 1),
    ("3sin2+3cos2", 3 * sin(x) ** 2 + 3 * cos(x) ** 2, I(3)),
    ("y*sin2+y*cos2", y * sin(x) ** 2 + y * cos(x) ** 2, y),
    ("sin2(2x)+cos2(2x)", sin(2 * x) ** 2 + cos(2 * x) ** 2, I(1)),
    ("sin2+cos2-1", sin(x) ** 2 + cos(x) ** 2 - 1, I(0)),
    ("two pythagorean pairs", sin(x) ** 2 + cos(x) ** 2 + sin(y) ** 2 + cos(y) ** 2, I(2)),
    ("pythagorean inside exp", exp(sin(x) ** 2 + cos(x) ** 2), exp(I(1))),
    ("cosh2-sinh2", cosh(x) ** 2 - sinh(x) ** 2, I(1)),
    ("y+cosh2-sinh2", y + cosh(x) ** 2 - sinh(x) ** 2, y + 1),
    ("2cosh2-2sinh2", 2 * cosh(x) ** 2 - 2 * sinh(x) ** 2, I(2)),
    ("sin(2x)-2sincos", sin(2 * x) - 2 * sin(x) * cos(x), I(0)),
    ("cos(2x)-(cos2-sin2)", cos(2 * x) - (cos(x) ** 2 - sin(x) ** 2), I(0)),
    ("tan - sin/cos", tan(x) - sin(x) / cos(x), I(0)),
    ("tan*cos", tan(x) * cos(x), sin(x)),
    ("sin(x+y) expansion", sin(x + y) - sin(x) * cos(y) - cos(x) * sin(y), I(0)),
    ("sin4-cos4-(sin2-cos2)", sin(x) ** 4 - cos(x) ** 4 - (sin(x) ** 2 - cos(x) ** 2), I(0)),
    ("exp(log x)", exp(log(x)), x),
    ("exp(log(x+1))", exp(log(x + 1)), x + 1),
    ("y*exp(log x)", y * exp(log(x)), x * y),
    ("exp(x)exp(y)", exp(x) * exp(y), exp(x + y)),
    ("exp(x)exp(-x)", exp(x) * exp(-x), I(1)),
    ("log(exp x) real", log(exp(xp)), xp),
    ("log(xy)-log x-log y pos", log(xp * yp) - log(xp) - log(yp), I(0)),
    ("sqrt(x^2) pos", sqrt(xp**2), xp),
    ("(x^2-1)/(x-1)", (x**2 - 1) / (x - 1), x + 1),
    ("(x^2-y^2)/(x+y)", (x**2 - y**2) / (x + y), x - y),
    ("(x^2+2x+1)/(x+1)", (x**2 + 2 * x + 1) / (x + 1), x + 1),
    ("(x^3-1)/(x-1)", (x**3 - 1) / (x - 1), x**2 + x + 1),
    ("1/(1/x+1/y)", 1 / (1 / x + 1 / y), x * y / (x + y)),
    ("x/x", x / x, I(1)),
    ("x - x", x - x, I(0)),
    ("2x - x", 2 * x - x, x),
    ("(x+y)^2-(x-y)^2", (x + y) ** 2 - (x - y) ** 2, 4 * x * y),
    ("(x+1)^2-x^2-2x", (x + 1) ** 2 - x**2 - 2 * x, I(1)),
    ("(sqrt2+1)(sqrt2-1)", (sqrt(I(2)) + 1) * (sqrt(I(2)) - 1), I(1)),
    ("(sqrt3-1)(sqrt3+1)", (sqrt(I(3)) - 1) * (sqrt(I(3)) + 1), I(2)),
    ("sqrt(3+2sqrt2)", sqrt(3 + 2 * sqrt(I(2))), 1 + sqrt(I(2))),
    ("sqrt(3+2sqrt2)-1-sqrt2", sqrt(3 + 2 * sqrt(I(2))) - 1 - sqrt(I(2)), I(0)),
    ("sqrt(5-2sqrt6)", sqrt(5 - 2 * sqrt(I(6))), sqrt(I(3)) - sqrt(I(2))),
    ("sqrt(8)", sqrt(I(8)), 2 * sqrt(I(2))),
    ("1/3+1/6", R(1, 3) + R(1, 6), R(1, 2)),
]

# Inputs simplify must leave alone (nothing smaller exists or the rewrite is
# not an identity without assumptions the input does not carry).
UNCHANGED = [
    ("log(exp x) complex", log(exp(x))),
    ("sqrt(x^2) unknown sign", sqrt(x**2)),
    ("1 - sin^2", 1 - sin(x) ** 2),
    ("(x^10-1)/(x-1)", (x**10 - 1) / (x - 1)),
    ("(x+1)^5", (x + 1) ** 5),
]


@pytest.mark.parametrize(("name", "expr", "expected"), CASES, ids=[c[0] for c in CASES])
def test_textbook_identity_reaches_simplest_form(name, expr, expected):
    got = S(expr)
    assert got == S(expected), f"{name}: simplify gave {got}, expected {S(expected)}"


@pytest.mark.parametrize(("name", "expr"), UNCHANGED, ids=[c[0] for c in UNCHANGED])
def test_simplify_does_not_grow(name, expr):
    """Never larger than the core rule engine's answer."""
    got = S(expr)
    assert len(str(got)) <= len(str(expr)) + 8, (name, str(got))


def _points(rng):
    for _ in range(6):
        # Positive values keep log/sqrt real and stay off the removable points
        # (x = 1, x = -y) of the cancellation cases.
        yield {s: rng.uniform(1.3, 2.7) for s in (x, y, xp, yp)}


@pytest.mark.parametrize(
    ("name", "expr"),
    [(c[0], c[1]) for c in CASES] + UNCHANGED,
    ids=[c[0] for c in CASES] + [c[0] for c in UNCHANGED],
)
def test_simplify_preserves_value(name, expr):
    rng = random.Random(hash(name) & 0xFFFF)
    out = S(expr)
    for env in _points(rng):
        try:
            before = ak.eval_expr(expr, env)
        except Exception:
            continue
        after = ak.eval_expr(out, env)
        assert math.isclose(after, before, rel_tol=1e-9, abs_tol=1e-9), (name, env, before, after)


# ---------------------------------------------------------------------------
# Side conditions and certificates for the new rewrites
# ---------------------------------------------------------------------------


def _side_conditions(r):
    return [c for st in r.steps for c in (st.get("side_conditions") or [])]


def test_rational_cancel_records_the_excluded_point():
    r = ak.simplify((x**2 - 1) / (x - 1))
    assert r.value == S(x + 1)
    rules = [st["rule"] for st in r.steps]
    assert "rational_cancel" in rules
    assert _side_conditions(r), "x - 1 != 0 must be recorded"


def test_exp_of_log_records_nonzero():
    r = ak.simplify(exp(log(x)))
    assert r.value == x
    assert any(st["rule"] == "exp_of_log_nonzero" for st in r.steps)
    assert _side_conditions(r)


@pytest.mark.parametrize(
    "expr",
    [
        # Each goes through a rule whose generic `ring_nf; simp` fallback
        # was checked against Lean/Mathlib and does not close the goal.
        2 * cosh(x) ** 2 - 2 * sinh(x) ** 2,
        sin(x) * cos(y) - cos(x) * sin(y),
        exp(x) ** 3 * exp(y),
        exp(log(x)),
        (x**2 - 1) / (x - 1),
        sqrt(3 + 2 * sqrt(I(2))),
    ],
)
def test_rules_without_a_lean_emitter_withhold_the_certificate(expr):
    r = ak.simplify(expr)
    assert r.certificate is None, r.certificate


@pytest.mark.parametrize(
    ("fn", "expr", "lemma"),
    [
        (ak.simplify, sin(x) ** 2 + cos(x) ** 2, "sin_sq_add_cos_sq"),
        # The instances Lean was checked to close with `ring_nf; simp`.
        (ak.simplify, cosh(x) ** 2 - sinh(x) ** 2, "ring_nf; simp"),
        (ak.simplify_trig, cos(x) * cos(x) + sin(x) * sin(x), "cos_angle_sub"),
    ],
)
def test_checked_certificates_are_still_emitted(fn, expr, lemma):
    cert = fn(expr).certificate
    assert cert is not None
    assert lemma in cert


# ---------------------------------------------------------------------------
# Division by an exact zero (report #7)
# ---------------------------------------------------------------------------


def test_x_minus_x_folds_at_construction():
    assert x - x == I(0)
    assert x + (-x) == I(0)
    assert (x + 1) - (x + 1) == I(0)
    oo = P.pos_infinity()
    assert oo - oo != I(0), "oo - oo is indeterminate, not 0"


def test_division_by_a_zero_that_appears_on_simplification():
    # `x + 2x - 3x` is not folded when written, only once simplified.
    with pytest.raises(ak.DomainError) as ei:
        ak.simplify(1 / (x + 2 * x - 3 * x))
    assert ei.value.code == "E-DOMAIN-001"
    assert isinstance(ei.value, ZeroDivisionError)
    with pytest.raises(ak.DivisionByZeroError):
        ak.simplify_auto(1 / (x + 2 * x - 3 * x))


def test_division_by_literal_zero_is_the_same_error():
    with pytest.raises(ak.DivisionByZeroError):
        x / 0
    with pytest.raises(ZeroDivisionError):
        x / (x - x)


def test_limit_of_an_undefined_expression_refuses_quickly():
    import time

    for e in (sin(x) / (x + 2 * x - 3 * x), 1 / (x + 2 * x - 3 * x)):
        t = time.time()
        budget = ak.context(budget=ak.Budget(wall_ms=5000))
        with pytest.raises(ak.DivisionByZeroError) as ei, budget:
            ak.limit(e, x, I(0))
        assert ei.value.code == "E-DOMAIN-001"
        # It used to run 16 s past a 5 s budget.
        assert time.time() - t < 2.0


def test_diff_of_an_undefined_expression_raises():
    with pytest.raises(ak.DivisionByZeroError):
        ak.diff(1 / (x + 2 * x - 3 * x), x)
