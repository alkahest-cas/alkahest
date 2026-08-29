"""The dilogarithm ``Li₂``.

**Branch cut.**  ``dilog`` is ``Li₂`` on its **principal branch**, cut along
``[1, ∞)`` — DLMF §25.12(i), Lewin *Polylogarithms and Associated Functions*
§1.1, Mathematica ``PolyLog[2, z]``.  It is real on ``(−∞, 1]``, endpoint
included (``Li₂(1) = π²/6``), and declines past the cut, where the principal
value is complex: ``Li₂(x ± i0) = Re Li₂(x) ∓ iπ·log x``.

**``dilog``, not ``polylog(s, x)``.**  ``∂Li_s/∂s`` has no closed form, so a
binary ``polylog`` would ship with a permanently declined partial; and the
validated Taylor tier's rules are all unary, so it would also be invisible to
``bound_on_box``.  ``Li₁(x)`` needs no primitive — it is ``-log(1 - x)``.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest

PI2 = math.pi * math.pi


@pytest.fixture
def pool():
    p = ak.ExprPool()
    with ak.context(pool=p):
        yield p


@pytest.fixture
def x(pool):
    return ak.symbol("x", pool=pool)


def enclose(expr, var, at, **kw):
    """Rigorous enclosure of `expr` at the single point `at`."""
    r = ak.bound_on_box(expr, [(var, at, at)], **kw)
    return r.lower, r.upper


def contains(bounds, want, slack=1e-12):
    lo, hi = bounds
    return lo - slack <= want <= hi + slack


def test_exported_and_displayed(x):
    assert "dilog" in ak.__all__
    assert ak.unicode_str(ak.dilog(x)) == "Li₂(x)"
    assert ak.latex(ak.dilog(x)) == r"\operatorname{Li}_2\!\left(x\right)"


def test_dilog_published_anchors(x):
    """``Li₂(1) = π²/6`` and ``Li₂(−1) = −π²/12`` are DLMF 25.12.2;
    ``Li₂(½) = π²/12 − log²2/2`` is Lewin eq. 1.16 (Landen)."""
    assert contains(enclose(ak.dilog(x), x, -1.0), -PI2 / 12.0)
    assert contains(enclose(ak.dilog(x), x, 0.5), PI2 / 12.0 - math.log(2) ** 2 / 2)
    assert contains(enclose(ak.dilog(x), x, 0.0), 0.0)
    # x = 1 is in the *function's* domain but not the Taylor model's — Li₂′ is
    # unbounded there — so the anchor is approached rather than evaluated.
    near_one = enclose(ak.dilog(x), x, 1.0 - 1e-9)
    assert abs(0.5 * (near_one[0] + near_one[1]) - PI2 / 6.0) < 1e-7


def test_dilog_is_strictly_increasing_on_its_real_domain(x):
    """``Li₂′ = −log(1−x)/x > 0`` on all of ``(−∞, 1)`` — for ``0 < x < 1``
    both factors are positive and for ``x < 0`` both are negative — so the
    range over ``[−1, 0]`` is exactly ``[Li₂(−1), 0]``."""
    r = ak.bound_on_box(ak.dilog(x), [(x, -1.0, 0.0)])
    assert contains((r.lower, r.upper), -PI2 / 12.0)
    assert contains((r.lower, r.upper), 0.0)


def test_dilog_refuses_past_its_branch_cut(x):
    """Past the cut the principal value is complex, so the real tier declines
    rather than picking a side or quietly returning the real part."""
    for lo, hi in [(1.0, 2.0), (0.5, 1.5), (2.0, 3.0)]:
        with pytest.raises(Exception):
            ak.bound_on_box(ak.dilog(x), [(x, lo, hi)])


def test_dilog_derivative_round_trips(x):
    d = ak.diff(ak.dilog(x), x).value
    for at in (-3.0, -0.4, 0.25, 0.8, 0.99):
        want = -math.log(1.0 - at) / at
        assert contains(enclose(d, x, at), want, slack=1e-10), f"Li2'({at})"


def test_dilog_far_negative_branch(x):
    """The inversion identity ``Li₂(x) = −Li₂(1/x) − π²/6 − ½log²(−x)`` takes
    over below ``x = −1``; ``−Li₂(1/x)`` is the only term left out here."""
    for at in (-2.0, -10.0, -1000.0):
        want = -PI2 / 6.0 - 0.5 * math.log(-at) ** 2
        lo, hi = enclose(ak.dilog(x), x, at)
        mid = 0.5 * (lo + hi)
        assert mid < want + 1.0
        assert mid > want - 1.0


def test_capabilities_and_bounds_support(x):
    rows = {r["name"]: r for r in ak.capabilities()["primitives"]}
    for flag in ("numeric_f64", "numeric_ball", "taylor_model", "diff_forward", "diff_reverse"):
        assert rows["dilog"][flag], flag
    assert ak.bounds_supported(ak.dilog(x)).supported
