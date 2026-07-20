"""Textbook gate — trigonometric simplification.

First-course trig identities run through ``ak.simplify_trig``: the
Pythagorean identity, double-angle formulas, odd/even symmetry, and
reciprocal-style cancellations. See ``tests/textbook_gate/README.md`` for the
verification philosophy (numeric checks, never string-matching alkahest's
normal form).

Most of these identities are only *value-preserving* under ``simplify_trig``
rather than fully collapsing to a canonical closed form — e.g.
``sin(x)*sin(x) + cos(x)*cos(x)`` simplifies to ``cos(x + (x * -1))`` (a
constant-valued but structurally non-constant expression), not the bare
integer ``1``. That's still a correct simplification (nothing was lost), so
it's checked the same way the README prescribes: does the simplified form
numerically agree with the expected reference, not what its printed shape is.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from _tg_helpers import DEFAULT_POINTS, assert_definite_value, assert_matches_reference, eval_at


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


def assert_simplify_trig_preserves_value(
    original: ak.Expr, var: ak.Expr, points=DEFAULT_POINTS
) -> ak.Expr:
    """Run ``simplify_trig`` on ``original`` and check the result numerically
    agrees with ``original`` at each sample point. Returns the simplified Expr
    for callers that want an additional, stronger check against a reference.
    """
    simplified = ak.simplify_trig(original).value
    assert_matches_reference(simplified, var, lambda v: eval_at(original, var, v), points)
    return simplified


# --- Pythagorean identity ----------------------------------------------------


def test_pythagorean_identity_folds_to_constant(x):
    """sin(x)^2 + cos(x)^2 -> 1, fully collapsed (no free symbols left)."""
    r = ak.simplify_trig(ak.sin(x) ** 2 + ak.cos(x) ** 2).value
    assert_definite_value(r, 1.0)


def test_pythagorean_identity_unfactored_form(x):
    """sin(x)*sin(x) + cos(x)*cos(x) is also constant-valued (=1) but does not
    structurally collapse to a bare integer — verify via numeric evaluation
    rather than assert_definite_value, since the result still contains `x` in
    its AST (e.g. `cos(x + (x * -1))`) even though it's constant everywhere.
    """
    r = ak.simplify_trig(ak.sin(x) * ak.sin(x) + ak.cos(x) * ak.cos(x)).value
    assert_matches_reference(r, x, lambda v: 1.0)


# --- double angle -------------------------------------------------------------


def test_double_angle_sine(x):
    """2 sin(x) cos(x) -> sin(2x): check the simplified result specifically
    matches sin(2x)'s values (stronger than mere value-preservation of the
    original, since 2 sin(x) cos(x) and sin(2x) are the same value already).
    """
    r = ak.simplify_trig(2 * ak.sin(x) * ak.cos(x)).value
    assert_matches_reference(r, x, lambda v: math.sin(2 * v))


def test_double_angle_cosine_via_sin_squared(x):
    """1 - 2 sin(x)^2 -> cos(2x)."""
    r = ak.simplify_trig(1 - 2 * ak.sin(x) ** 2).value
    assert_matches_reference(r, x, lambda v: math.cos(2 * v))


def test_double_angle_cosine_via_cos_squared(x):
    """2 cos(x)^2 - 1 -> cos(2x)."""
    r = ak.simplify_trig(2 * ak.cos(x) ** 2 - 1).value
    assert_matches_reference(r, x, lambda v: math.cos(2 * v))


def test_half_angle_sine_squared(x):
    """(1 - cos(2x))/2 -> sin(x)^2."""
    r = ak.simplify_trig((1 - ak.cos(2 * x)) / 2).value
    assert_matches_reference(r, x, lambda v: math.sin(v) ** 2)


def test_double_angle_cosine_difference_is_zero(x):
    """cos(2x) - (1 - 2 sin(x)^2) simplifies to (numerically) zero."""
    r = ak.simplify_trig(ak.cos(2 * x) - (1 - 2 * ak.sin(x) ** 2)).value
    assert_matches_reference(r, x, lambda v: 0.0)


# --- odd / even symmetry ------------------------------------------------------


def test_sine_is_odd(x):
    """sin(-x) -> -sin(x)."""
    r = ak.simplify_trig(ak.sin(-x)).value
    assert_matches_reference(r, x, lambda v: -math.sin(v))


def test_cosine_is_even(x):
    """cos(-x) -> cos(x)."""
    r = ak.simplify_trig(ak.cos(-x)).value
    assert_matches_reference(r, x, math.cos)


# --- reciprocal / ratio combos -------------------------------------------------


def test_tan_times_cos_is_sin(x):
    """tan(x) * cos(x) -> sin(x)."""
    r = ak.simplify_trig(ak.tan(x) * ak.cos(x)).value
    assert_matches_reference(r, x, math.sin)


def test_sin_over_tan_is_cos(x):
    """sin(x) / tan(x) -> cos(x)."""
    r = ak.simplify_trig(ak.sin(x) / ak.tan(x)).value
    assert_matches_reference(r, x, math.cos)


def test_sin_over_cos_minus_tan_is_zero(x):
    """sin(x)/cos(x) - tan(x) simplifies to (numerically) zero."""
    r = ak.simplify_trig(ak.sin(x) / ak.cos(x) - ak.tan(x)).value
    assert_matches_reference(r, x, lambda v: 0.0)


def test_tan_minus_sin_over_cos_is_zero(x):
    """tan(x) - sin(x)/cos(x) simplifies to (numerically) zero (same identity,
    subtraction order flipped — checks the simplifier isn't order-sensitive).
    """
    r = ak.simplify_trig(ak.tan(x) - ak.sin(x) / ak.cos(x)).value
    assert_matches_reference(r, x, lambda v: 0.0)


def test_sin_times_reciprocal_sin_is_one(x):
    """sin(x) * (1/sin(x)) -> 1."""
    r = ak.simplify_trig(ak.sin(x) * (1 / ak.sin(x))).value
    assert_matches_reference(r, x, lambda v: 1.0)


# --- Pythagorean-like identity that is not a pure power sum -------------------


def test_one_plus_tan_squared_matches_sec_squared(x):
    """1 + tan(x)^2 = sec(x)^2 = 1/cos(x)^2. alkahest has no `sec` primitive and
    `simplify_trig` does not rewrite this into a single power of cos(x) — this
    is checked purely as value-preservation against the known closed form,
    not as a claim that alkahest folds it to a canonical shape.
    """
    r = ak.simplify_trig(1 + ak.tan(x) ** 2).value
    assert_matches_reference(r, x, lambda v: 1 / math.cos(v) ** 2)


# --- hyperbolic Pythagorean identity: not handled by simplify_trig ------------


@pytest.mark.xfail(
    strict=True,
    reason="simplify_trig does not touch hyperbolic functions at all: "
    "cosh(x)^2 - sinh(x)^2 is left as `(cosh(x)^2 + (sinh(x)^2 * -1))` with `x` "
    "still structurally present, so it cannot even be evaluated with no bindings "
    "(eval_expr raises 'unbound variable') despite being identically 1 for all "
    "real x. Not documented in report7-20.md as of this writing.",
)
def test_hyperbolic_pythagorean_identity_folds_to_constant(x):
    """cosh(x)^2 - sinh(x)^2 -> 1 (unconditionally valid for real x)."""
    r = ak.simplify_trig(ak.cosh(x) ** 2 - ak.sinh(x) ** 2).value
    assert_definite_value(r, 1.0)
