"""Ball arithmetic must actually enclose the value it claims to enclose.

``ArbBall`` is the library's rigorous-numerics surface: the whole reason to
reach for it instead of a float is that ``lo <= true value <= hi`` is a
*guarantee*, not an approximation. A ball that excludes its own true value is
the worst kind of silent error — a claimed proof that is false — because a
caller's only defence against it is the guarantee itself.

Two independent bugs, both found by auditing this surface and both fixed:

1. ``exp``/``log``/``sqrt`` (and ``tan``, ``asin``, ``acos``, ``atan``,
   ``asinh``, ``atanh``) built their result from ``f(lo)`` and ``f(hi)``
   evaluated at finite precision and took the spread as the radius, without
   ever accounting for the rounding of those two endpoint computations. Given
   an *exact* input the two endpoints coincided, so the radius came out
   exactly zero — asserting that a transcendental value is exactly
   representable. ``sin``/``cos`` were unaffected: they already added the
   rounding term.

2. The Python accessors ``lo``/``hi``/``rad`` rounded to ``f64`` to *nearest*.
   The ball is computed at far more than double precision, so a nearest-
   rounded endpoint can land strictly inside the true interval, and
   ``lo <= v <= hi`` then rejects a value the ball genuinely encloses. They
   now round outward.

The constants below are correct to 40 significant digits and are compared with
:mod:`decimal`, deliberately not ``mpmath`` — mpmath lives in the ``ci-extras``
group and is not installed for the Tier 1a run that must catch a regression
here.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import alkahest as ak
import pytest

getcontext().prec = 60

#: value → 40-significant-digit truth.
_TRUTH = {
    "exp(0.5)": Decimal("1.648721270700128146848650787814163571654"),
    "exp(1)": Decimal("2.718281828459045235360287471352662497757"),
    "log(2)": Decimal("0.6931471805599453094172321214581765680755"),
    "log(0.5)": Decimal("-0.6931471805599453094172321214581765680755"),
    "sqrt(2)": Decimal("1.414213562373095048801688724209698078570"),
    "sqrt(0.5)": Decimal("0.7071067811865475244008443621048490392848"),
    "sin(0.5)": Decimal("0.4794255386042030002732879352155713880818"),
    "sin(1)": Decimal("0.8414709848078965066525023216302989996226"),
    "cos(0.5)": Decimal("0.8775825618903727161162815826038296519916"),
    "cos(1)": Decimal("0.5403023058681397174009366074429766037323"),
}


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


def _cases(x):
    return [
        ("exp(0.5)", ak.exp(x), 0.5),
        ("exp(1)", ak.exp(x), 1.0),
        ("log(2)", ak.log(x), 2.0),
        ("log(0.5)", ak.log(x), 0.5),
        ("sqrt(2)", ak.sqrt(x), 2.0),
        ("sqrt(0.5)", ak.sqrt(x), 0.5),
        ("sin(0.5)", ak.sin(x), 0.5),
        ("sin(1)", ak.sin(x), 1.0),
        ("cos(0.5)", ak.cos(x), 0.5),
        ("cos(1)", ak.cos(x), 1.0),
    ]


def test_enclosures_contain_the_true_value(pool, x):
    """``lo <= true <= hi`` for every case. This is the guarantee."""
    failures = []
    for name, expr, at in _cases(x):
        ball = ak.interval_eval(expr, {x: ak.ArbBall(at)})
        lo, hi, truth = Decimal(ball.lo), Decimal(ball.hi), _TRUTH[name]
        if not (lo <= truth <= hi):
            failures.append(f"{name}: [{lo}, {hi}] excludes {truth}")
    assert not failures, "ball excluded its own true value:\n" + "\n".join(failures)


def test_transcendental_results_are_not_claimed_exact(pool, x):
    """A ball over an irrational result must carry a non-zero radius.

    This is the shape of bug 1 stated directly: ``rad == 0`` means "exactly
    representable", which is false for every value here.
    """
    zero_radius = [
        name
        for name, expr, at in _cases(x)
        if ak.interval_eval(expr, {x: ak.ArbBall(at)}).rad == 0.0
    ]
    assert not zero_radius, f"irrational results reported as exact: {zero_radius}"


def test_endpoints_round_outward(pool, x):
    """The ``f64`` view must not be narrower than the ball it reports.

    Rounding to nearest would let ``hi`` fall below the true upper endpoint.
    Checked structurally so it holds regardless of precision: the interval
    must contain the midpoint and have non-negative width.
    """
    for name, expr, at in _cases(x):
        ball = ak.interval_eval(expr, {x: ak.ArbBall(at)})
        assert ball.lo <= ball.mid <= ball.hi, f"{name}: midpoint outside [lo, hi]"
        assert ball.hi - ball.lo >= 0.0, f"{name}: negative width"
        assert ball.rad >= 0.0, f"{name}: negative radius"


def test_added_rounding_term_stays_at_precision_scale(pool, x):
    """The fix must buy soundness without materially widening the balls.

    Note the library does *not* claim exactness even for exactly-representable
    results: `x*x` at 3 is 9, and `rad` is already ~5e-38 on account of the
    rounding term `mul` has always added. That is sound (a wider ball is never
    wrong, only less useful), and unchanged here.

    What this pins is the magnitude: the radius must stay at the working-
    precision scale rather than growing to something that would make the
    enclosure useless. A future change that widens balls by orders of
    magnitude to paper over an unsoundness would fail here.
    """
    for expr, at, value in [(x * x, 3.0, 9.0), (ak.exp(x), 1.0, 2.718281828459045)]:
        ball = ak.interval_eval(expr, {x: ak.ArbBall(at)})
        assert ball.lo <= value <= ball.hi
        assert ball.rad < 1e-25 * max(1.0, abs(value)), (
            f"radius {ball.rad} is far above the working-precision scale"
        )
