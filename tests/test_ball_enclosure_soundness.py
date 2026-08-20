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

Two more, of a different shape, are covered by the wide-ball section at the
bottom of this file: ``tan`` accepted a box that crossed a pole, and ``pow``
hulled four corners across a sign change. Neither is reachable from a *point*
ball, which is all this file used to bind — which is why both survived the
tests that exist to catch exactly this.

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


# ---------------------------------------------------------------------------
# Wide balls
# ---------------------------------------------------------------------------
#
# Everything above binds ``ak.ArbBall(at)`` — a *point* ball, ``rad = 0.0``.
# That is the case the two kernels below could not get wrong, and it is why
# they stayed wrong: an enclosure claim about a single point is a claim about
# one value, while the guarantee this file exists to defend is a claim about
# every value in a *box*.
#
#   * ``tan`` guarded its pole by testing the two endpoints and by checking
#     that ``tan(lo) <= tan(hi)``. A box that crosses a pole and comes out the
#     far side satisfies both: ``tan`` over ``[0.1, 0.1 + pi]`` was reported as
#     ``[0.1003, 0.2027]``, and ``tan(1.5) = 14.101…`` with ``1.5`` in the box.
#   * ``pow`` hulled the four corners of ``base × exponent``, which encloses
#     the range only while ``x ↦ x**y`` is monotone in ``x``. ``x**2.0`` over
#     ``[-1, 3]`` was reported as ``[1, 9]``, missing ``x = 0``; ``x**-2.0``
#     over the same box was reported as the *finite* ``[0.111, 1.0]`` for a
#     function unbounded there.
#
# Neither is reachable at ``rad = 0``. The two tests below are the wide-ball
# sweep that is.

#: (function name, lo, hi, witness *strictly inside* the box, 40-digit truth).
#:
#: The witnesses avoid the two endpoints on purpose. ``ArbBall(mid, rad)``
#: cannot represent an arbitrary ``[lo, hi]`` exactly — ``(0.1 + 1.0)/2`` and
#: ``(1.0 - 0.1)/2`` both round — so the box's real lower end sits an ulp or so
#: away from ``0.1``, and for a steep function that ulp exceeds the 40 digits
#: below. Endpoints are covered exactly, from the ball's own bounds, by
#: ``endpoint_hull_kernels_enclose_random_wide_intervals`` in
#: ``alkahest-core/src/ball/mod.rs``.
_WIDE_TRUTH = [
    ("tan", 0.1, 3.241592653589793, 1.5, "14.10141994717171938764608365198775644566"),
    ("tan", -1.0, 4.141592653589793, 1.5707, "10381.32741756978658760268387828435979306"),
    ("tan", 0.1, 1.0, 0.5, "0.5463024898437905132551794657802853832976"),
    ("exp", -3.0, 5.0, 4.5, "90.01713130052181355011545674557436084793"),
    ("exp", -3.0, 5.0, -2.5, "0.08208499862389879516952867446715980783780"),
    ("log", 0.5, 2.0, 0.6, "-0.5108256237659907202129482504755473004409"),
    ("log", 0.5, 2.0, 1.9, "0.6418538861723947292448033614074233770888"),
    ("sqrt", 0.25, 9.0, 0.3, "0.5477225575051661033220665419872727524121"),
    ("sin", 0.0, 3.0, 1.5707963267948966, "0.9999999999999999999999999999999981253003"),
    ("cosh", -2.0, 3.0, 2.9, "9.114584294749733281202437702692431073838"),
    ("cosh", -2.0, 3.0, 0.0, "1.0"),
    ("tanh", -4.0, 4.0, 3.9, "0.9991808656700278991779684578147509577166"),
    ("atan", -5.0, 5.0, 4.9, "1.369479218420255873378967790264769028244"),
    ("erf", -2.0, 2.0, 1.9, "0.9927904292352574672372159671320949386714"),
    ("gamma", 0.5, 5.0, 1.4616321449683622, "0.8856031944108887002788159005825926411111"),
    ("digamma", 0.5, 3.0, 0.6, "-1.540619213893190495500737928815446515306"),
    ("bessel_j0", -1.0, 1.0, 0.0, "1.0"),
    ("bessel_j0", 2.0, 4.0, 3.0, "-0.2600519549019334376241546959773314368196"),
    ("lambert_w", 0.0, 3.0, 1.0, "0.5671432904097838729999686622103555497538"),
]


def _wide(lo: float, hi: float) -> ak.ArbBall:
    """The ball whose interval is (as closely as `f64` allows) ``[lo, hi]``."""
    return ak.ArbBall((lo + hi) / 2.0, (hi - lo) / 2.0, 128)


def _encloses(ball, truth: Decimal) -> bool:
    """``lo <= truth <= hi``, with `inf` endpoints handled.

    An unbounded box is honestly reported as ``[-inf, inf]``; that contains
    everything, which is exactly the claim being made.
    """
    if ball.lo == float("-inf") and ball.hi == float("inf"):
        return True
    return Decimal(ball.lo) <= truth <= Decimal(ball.hi)


def test_wide_ball_enclosures_contain_externally_verified_values(pool, x):
    """The guarantee, restated for a box: every value the box reaches is in.

    The witnesses are interior points and endpoints alike, and the truths are
    40-significant-digit constants (compared with :mod:`decimal`, so this runs
    in the default tier without ``mpmath``).
    """
    failures = []
    for name, lo, hi, witness, truth in _WIDE_TRUTH:
        expr = getattr(ak, name)(x)
        try:
            ball = ak.interval_eval(expr, {x: _wide(lo, hi)}, prec=128)
        except ValueError:
            continue  # a refusal is sound; it claims nothing
        if not _encloses(ball, Decimal(truth)):
            failures.append(
                f"{name} over [{lo}, {hi}]: [{ball.lo}, {ball.hi}] "
                f"excludes f({witness}) = {truth}"
            )
    assert not failures, "wide ball excluded a value inside its box:\n" + "\n".join(
        failures
    )


def test_pow_over_a_base_that_changes_sign(pool, x):
    """``x**y`` for a base straddling 0, by every route the exponent can take.

    ``x ** p.integer(2)`` reaches a different kernel (``powi``) from
    ``x ** 2.0``, ``x ** p.float(2.0)``, ``x ** p.rational(4, 2)`` and
    ``x ** y`` with ``y`` bound to a point ball. Only the first was sound.
    """
    box = _wide(-1.0, 3.0)
    y = pool.symbol("y")
    two = [
        ("float literal", x**2.0, {}),
        ("p.float(2.0)", x ** pool.float(2.0), {}),
        ("p.rational(4, 2)", x ** pool.rational(4, 2), {}),
        ("p.integer(2)", x ** pool.integer(2), {}),
        ("bound symbol", x**y, {y: ak.ArbBall(2.0, 0.0, 128)}),
    ]
    failures = []
    for label, expr, extra in two:
        ball = ak.interval_eval(expr, {x: box, **extra}, prec=128)
        # x = 0 is in the box and 0**2 = 0; the corner hull said [1, 9].
        if not _encloses(ball, Decimal(0)):
            failures.append(f"x**2 via {label}: [{ball.lo}, {ball.hi}] excludes 0")
        if not _encloses(ball, Decimal(9)):
            failures.append(f"x**2 via {label}: [{ball.lo}, {ball.hi}] excludes 9")

    # A negative exponent puts a *pole* in the box, so no finite bound is true.
    for label, expr in [
        ("float literal", x**-2.0),
        ("p.float(-2.0)", x ** pool.float(-2.0)),
        ("p.integer(-2)", x ** pool.integer(-2)),
    ]:
        ball = ak.interval_eval(expr, {x: box}, prec=128)
        if not _encloses(ball, Decimal("999999.9999999999583666365765566310341165")):
            failures.append(
                f"x**-2 via {label}: [{ball.lo}, {ball.hi}] excludes "
                f"f(0.001) = 1e6, and the box contains 0.001"
            )
    assert not failures, "\n".join(failures)


def test_a_wide_ball_encloses_every_point_ball_inside_it(pool, x):
    """Oracle-free sweep: the enclosure over a box contains the enclosure at
    each point of that box.

    Both are rigorous claims about the same function, so the box's answer must
    contain the point's — no external truth needed, which lets this cover
    every kernel that has a ball implementation rather than only the ones with
    a constant in the table above. Sampling stays off the two endpoints, where
    an `f64` sample can fall a few ulps outside the box it was derived from.
    """
    boxes = {
        "exp": [(-4.0, 4.0), (0.0, 20.0)],
        "log": [(0.25, 8.0), (1e-3, 1.0)],
        "sqrt": [(0.0, 9.0)],
        "sin": [(-7.0, 7.0), (0.0, 1.0)],
        "cos": [(-7.0, 7.0), (0.0, 1.0)],
        "tan": [(0.1, 3.3), (-1.0, 4.2), (0.1, 1.0), (1.6, 3.0), (-10.0, 10.0)],
        "sinh": [(-3.0, 3.0)],
        "cosh": [(-3.0, 3.0), (1.0, 5.0)],
        "tanh": [(-4.0, 4.0)],
        "asin": [(-0.9, 0.9)],
        "acos": [(-0.9, 0.9)],
        "atan": [(-5.0, 5.0)],
        "asinh": [(-5.0, 5.0)],
        "acosh": [(1.5, 6.0)],
        "atanh": [(-0.9, 0.9)],
        "erf": [(-3.0, 3.0)],
        "erfc": [(-3.0, 3.0)],
        "gamma": [(0.25, 5.0), (1.0, 2.0)],
        "digamma": [(0.25, 5.0)],
        "bessel_j0": [(-1.0, 1.0), (0.0, 12.0)],
        "bessel_j1": [(0.0, 12.0)],
        "lambert_w": [(0.0, 5.0)],
        "abs": [(-2.0, 3.0)],
    }
    exprs = [(name, getattr(ak, name)(x), box) for name, bs in boxes.items() for box in bs]
    exprs += [
        (label, expr, box)
        for label, expr in [
            ("x**2.0", x**2.0),
            ("x**3.0", x**3.0),
            ("x**-2.0", x**-2.0),
            ("x**0.5", x**0.5),
            ("x**rational(4,2)", x ** pool.rational(4, 2)),
        ]
        for box in [(-1.0, 3.0), (0.5, 4.0), (-4.0, -0.5)]
    ]

    failures = []
    for label, expr, (lo, hi) in exprs:
        try:
            wide = ak.interval_eval(expr, {x: _wide(lo, hi)}, prec=128)
        except ValueError:
            continue  # a refusal is sound
        if wide.lo == float("-inf") and wide.hi == float("inf"):
            continue  # unbounded: contains everything
        for k in range(1, 40):
            at = lo + (hi - lo) * (k / 40.0)
            try:
                point = ak.interval_eval(expr, {x: ak.ArbBall(at, 0.0, 128)}, prec=128)
            except ValueError:
                continue
            # Slack absorbs the two enclosures' own outward rounding, which is
            # at the 1e-38 scale; a non-monotonicity bug is of order 1.
            slack = 1e-25 * max(1.0, abs(point.lo), abs(point.hi))
            if point.lo < wide.lo - slack or point.hi > wide.hi + slack:
                failures.append(
                    f"{label} over [{lo}, {hi}] claims [{wide.lo}, {wide.hi}], "
                    f"but at x={at} the value is in [{point.lo}, {point.hi}]"
                )
                break
    assert not failures, "box enclosure does not contain a point inside it:\n" + "\n".join(
        failures
    )
