"""Validated bounds for `asinh`, `acosh`, `atanh`, `erf`, `erfc` (3.9.0).

These five gained Taylor-model rules in 3.9.0, so `bound_on_box` — and
`verified_sign`, `verified_no_roots`, `verified_integral` on top of it — now
answer for them instead of refusing with `E-VALIDATED-001`.

Every one of these results is a **certificate**, so the only thing worth
testing at length is whether the enclosure is true. A returned bound that does
not contain the value it claims to enclose is not a loose answer, it is a false
theorem: 3.8 shipped a `cos` rule whose polynomial was sign-flipped, and it
passed `sin² + cos² = 1` the whole time. So the tests below are containment
tests against an independent high-precision reference, plus refusals on the
boxes where the rules must decline rather than answer.

Two reference sources, deliberately:

* 40-significant-digit constants as :class:`decimal.Decimal`, which run in
  every CI tier. mpmath lives in the ``ci-extras`` group and is not installed
  for Tier 1a, so a regression must be catchable without it.
* mpmath, for the dense sweeps — 200 randomised boxes per function — where
  hard-coded constants are not an option.

Comparing an `f64` truth against `BoundResult.lower` / `.upper` is sound even
when the enclosure is far tighter than an `f64` ulp: those two accessors round
*outward* (down and up respectively) from the arbitrary-precision enclosure,
so ``round_down(L) <= round_nearest(v) <= round_up(U)`` whenever
``L <= v <= U``. The same comparison against a *non*-outward-rounded endpoint
would be wrong, which is exactly the trap the Rust-side tests avoid by
comparing at ``prec + 64``.
"""

from __future__ import annotations

import random
from decimal import Decimal

import alkahest as ak
import pytest

_UNSUPPORTED = "E-VALIDATED-001"
_DOMAIN = "E-VALIDATED-003"

_FUNCS = {
    "asinh": ak.asinh,
    "acosh": ak.acosh,
    "atanh": ak.atanh,
    "erf": ak.erf,
    "erfc": ak.erfc,
}

#: 40-significant-digit truth, independent of anything in the library.
_TRUTH = {
    "asinh": {
        -1.0: Decimal("-0.8813735870195430252326093249797923090282"),
        0.0: Decimal("0.0"),
        0.5: Decimal("0.4812118250596034474977589134243684231352"),
        1.0: Decimal("0.8813735870195430252326093249797923090282"),
        3.0: Decimal("1.818446459232066823483698963560708993786"),
        -7.25: Decimal("-2.678871313462425994814667457251795274043"),
        100.0: Decimal("5.298342365610588757368825689112906302142"),
    },
    "acosh": {
        1.5: Decimal("0.9624236501192068949955178268487368462704"),
        2.0: Decimal("1.316957896924816708625046347307968444027"),
        2.5: Decimal("1.566799236972411078664056862580483493862"),
        1.00390625: Decimal("0.08835960065848382344165699567728061952018"),
        40.0: Decimal("4.38187034804006698696313269586603717077"),
    },
    "atanh": {
        -0.5: Decimal("-0.5493061443340548456976226184612628523237"),
        0.0: Decimal("0.0"),
        0.5: Decimal("0.5493061443340548456976226184612628523237"),
        0.875: Decimal("1.354025100551105032998002285074356672087"),
        0.9999990463256836: Decimal("7.278045157460789803881506424625541533523"),
    },
    "erf": {
        0.0: Decimal("0.0"),
        0.5: Decimal("0.5204998778130465376827466538919645287365"),
        1.0: Decimal("0.8427007929497148693412206350826092592961"),
        2.0: Decimal("0.9953222650189527341620692563672529286109"),
        -3.0: Decimal("-0.9999779095030014145586272238704176796202"),
    },
    "erfc": {
        0.0: Decimal("1.0"),
        0.5: Decimal("0.4795001221869534623172533461080354712635"),
        1.0: Decimal("0.1572992070502851306587793649173907407039"),
        2.0: Decimal("0.004677734981047265837930743632747071389108"),
        -3.0: Decimal("1.99997790950300141455862722387041767962"),
    },
}


def _bound(name, lo, hi, **opts):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    return ak.bound_on_box(_FUNCS[name](x), [(x, lo, hi)], **opts)


# ---------------------------------------------------------------------------
# The enclosure must contain the value — checked without mpmath
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_TRUTH))
def test_point_values_are_bracketed(name):
    """A degenerate box is a point evaluation, pinned to 40 digits.

    This is the check a sign flip or a wrong expansion point cannot survive.
    A containment sweep over a wide box sometimes can: a flipped `cos` still
    lands inside a wide symmetric enclosure.
    """
    for point, truth in _TRUTH[name].items():
        r = _bound(name, point, point)
        assert r.lower <= float(truth) <= r.upper, (
            f"{name}({point}) = {truth} escaped [{r.lower}, {r.upper}]"
        )


@pytest.mark.parametrize("name", sorted(_TRUTH))
def test_enclosure_brackets_both_endpoints_of_a_real_box(name):
    """Each of these is monotone, so the true range *is* the endpoint pair.

    Asserting containment of both endpoints therefore asserts containment of
    the whole range, and — because it is checked at both ends — a rule whose
    polynomial is shifted or scaled cannot pass it.
    """
    boxes = {
        "asinh": [(-1.0, 1.0), (0.5, 3.0), (-7.25, -1.0)],
        "acosh": [(1.5, 2.5), (1.00390625, 1.5), (2.0, 40.0)],
        "atanh": [(-0.5, 0.5), (0.5, 0.875), (0.875, 0.9999990463256836)],
        "erf": [(0.0, 1.0), (0.5, 2.0), (-3.0, 0.0)],
        "erfc": [(0.0, 1.0), (0.5, 2.0), (-3.0, 0.0)],
    }[name]
    for lo, hi in boxes:
        r = _bound(name, lo, hi)
        for point in (lo, hi):
            truth = float(_TRUTH[name][point])
            assert r.lower <= truth <= r.upper, (
                f"{name} on [{lo},{hi}]: {name}({point}) = {truth} escaped [{r.lower}, {r.upper}]"
            )


@pytest.mark.parametrize("name", sorted(_TRUTH))
def test_the_enclosure_is_tight_not_merely_true(name):
    """`[-inf, inf]` would pass every containment test above.

    These are monotone, so the true range width is the endpoint difference,
    and a converged branch-and-bound should land within `tol` of it.
    """
    lo, hi = {
        "asinh": (0.5, 3.0),
        "acosh": (1.5, 2.5),
        "atanh": (-0.5, 0.5),
        "erf": (0.0, 1.0),
        "erfc": (0.0, 1.0),
    }[name]
    r = _bound(name, lo, hi)
    span = abs(float(_TRUTH[name][hi] - _TRUTH[name][lo]))
    assert r.width <= span + 1e-6, f"{name} on [{lo},{hi}]: width {r.width} vs range {span}"


# ---------------------------------------------------------------------------
# Domain guards: an off-domain box refuses, it does not answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "lo", "hi", "why"),
    [
        ("acosh", 0.0, 0.5, "entirely below the branch point"),
        ("acosh", -2.0, -1.0, "negative"),
        ("acosh", 0.5, 2.0, "straddles the branch point"),
        ("acosh", 1.0, 2.0, "touches the branch point, where acosh' is infinite"),
        ("acosh", 0.999999, 1.5, "reaches just below 1"),
        ("atanh", 1.5, 2.0, "entirely above +1"),
        ("atanh", -2.0, -1.5, "entirely below -1"),
        ("atanh", 0.5, 2.0, "leaves through +1 — a one-sided guard would miss this"),
        ("atanh", -2.0, 0.5, "leaves through -1"),
        ("atanh", -1.0, 0.5, "touches -1"),
        ("atanh", -0.5, 1.0, "touches +1"),
        ("atanh", -2.0, 2.0, "contains both poles"),
    ],
)
def test_off_domain_boxes_refuse_rather_than_bound(name, lo, hi, why):
    """A bound off the domain is a wrong answer, not a loose one.

    The rule itself always says `E-VALIDATED-003`; which code reaches the
    caller depends on how the branch-and-bound above it gives up on a
    violation it cannot bisect away, and with a small budget the same box
    comes back as `E-VALIDATED-004` instead — pre-existing behaviour, shared
    with `log(-1 - x²)` and `asin(2 + x²)`. Both are refusals. What must
    never appear is `E-VALIDATED-001`, which would claim there is no rule.
    """
    with pytest.raises(ak.ValidatedError) as excinfo:
        _bound(name, lo, hi)
    assert excinfo.value.code in {_DOMAIN, "E-VALIDATED-004"}, (
        f"{name} on [{lo},{hi}] ({why}) refused with "
        f"{excinfo.value.code}, expected a refusal about the domain"
    )
    assert excinfo.value.code != _UNSUPPORTED


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [
        ("asinh", -1.0, 1.0),
        ("asinh", -1e6, -999999.0),
        ("asinh", -30.0, 30.0),
        ("erf", -50.0, 50.0),
        ("erf", 20.0, 21.0),
        ("erfc", -50.0, 50.0),
    ],
)
def test_entire_functions_never_refuse_on_domain_grounds(name, lo, hi):
    """`asinh`, `erf` and `erfc` are entire — no box is off-domain.

    The far-negative `asinh` boxes are the interesting ones: an
    implementation routed through ``log(x + sqrt(1 + x**2))`` loses the
    argument to cancellation there and refuses.
    """
    r = _bound(name, lo, hi)
    assert r.lower <= r.upper


def test_a_domain_refusal_is_not_reported_as_unsupported():
    """`acosh` *has* a rule; `[0, 0.5]` is just a box outside its domain.

    Reporting that as `E-VALIDATED-001` would tell a planner the function is
    uncovered and send it off a route that works one box over — which is the
    exact confusion `bounds_supported` exists to prevent.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    assert ak.bounds_supported(ak.acosh(x))
    with pytest.raises(ak.ValidatedError) as excinfo:
        ak.bound_on_box(ak.acosh(x), [(x, 0.0, 0.5)])
    assert excinfo.value.code != _UNSUPPORTED


# ---------------------------------------------------------------------------
# The rest of the stack lights up too
# ---------------------------------------------------------------------------


def test_verified_sign_and_no_roots_reach_the_new_functions():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    # atanh(x) > 0 on [0.1, 0.9]
    assert ak.verified_sign(ak.atanh(x), [(x, 0.1, 0.9)], "positive") == "true"
    # erf(x) - 1 < 0 everywhere: erf < 1 strictly.
    assert ak.verified_sign(ak.erf(x) - pool.integer(1), [(x, -2.0, 2.0)], "negative") == "true"
    # asinh has its only root at 0, so it has none on [1, 2] …
    assert ak.verified_no_roots(ak.asinh(x), [(x, 1.0, 2.0)]) == "true"
    # … and does have one on [-1, 1].
    assert ak.verified_no_roots(ak.asinh(x), [(x, -1.0, 1.0)]) == "false"


def test_verified_integral_of_erf_matches_its_closed_form():
    """∫₀¹ erf = erf(1) - (1 - e^{-1})/√π."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.verified_integral(ak.erf(x), x, 0.0, 1.0)
    exact = 0.4860649581122559  # erf(1) + (exp(-1) - 1)/sqrt(pi), to f64
    assert r.lower <= exact <= r.upper
    assert r.width < 1e-6


def test_the_new_rules_compose_with_the_rest_of_the_algebra():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    f = ak.asinh(x * x + y) * ak.erf(y) + ak.atanh(x / 4)
    r = ak.bound_on_box(f, [(x, 0.0, 1.0), (y, 0.5, 1.5)], tol=1e-6, max_subdivisions=4096)
    assert r.lower <= r.upper
    assert r.width < 5.0


# ---------------------------------------------------------------------------
# mpmath sweeps — dense samples and 200 randomised boxes per function
# ---------------------------------------------------------------------------

mpmath = pytest.importorskip("mpmath")


def _mp_ref(name):
    return {
        "asinh": mpmath.asinh,
        "acosh": mpmath.acosh,
        "atanh": mpmath.atanh,
        "erf": mpmath.erf,
        "erfc": mpmath.erfc,
    }[name]


def _assert_covers(name, lo, hi, result, n=64):
    """Every sampled true value must be inside the enclosure."""
    fun = _mp_ref(name)
    with mpmath.workdps(50):
        for k in range(n + 1):
            t = mpmath.mpf(lo) + (mpmath.mpf(hi) - mpmath.mpf(lo)) * k / n
            t = min(max(t, mpmath.mpf(lo)), mpmath.mpf(hi))
            v = float(fun(t))
            assert result.lower <= v <= result.upper, (
                f"{name}({t}) = {v} escaped [{result.lower}, {result.upper}] on [{lo}, {hi}]"
            )


@pytest.mark.parametrize(
    ("name", "boxes"),
    [
        (
            "asinh",
            [(-1.0, 1.0), (0.0, 3.0), (-5.0, -4.0), (7.0, 9.0), (-100.0, -99.0)],
        ),
        (
            "acosh",
            [(1.5, 2.0), (1.05, 1.1), (1.001, 1.0011), (3.0, 8.0), (10.0, 10.5)],
        ),
        (
            "atanh",
            [(-0.5, 0.5), (0.9, 0.95), (-0.999, -0.998), (0.999999, 0.9999995)],
        ),
        ("erf", [(-1.0, 1.0), (1.5, 2.0), (2.0, 6.0), (-8.0, -7.5)]),
        ("erfc", [(-1.0, 1.0), (1.5, 2.0), (2.0, 6.0), (-8.0, -7.5)]),
    ],
)
def test_dense_samples_stay_inside_the_enclosure(name, boxes):
    """Including boxes that run right up to a domain boundary."""
    for lo, hi in boxes:
        _assert_covers(name, lo, hi, _bound(name, lo, hi), n=128)


@pytest.mark.parametrize("name", ["asinh", "acosh", "atanh", "erf", "erfc"])
def test_randomised_box_sweep(name):
    """200 boxes per function, centres and widths both varying.

    A refusal is skipped rather than failed: refusing is always sound, and
    ``test_off_domain_boxes_refuse_rather_than_bound`` is what pins that
    refusals happen for the right reason. Only a *returned bound* can be
    wrong, which is what this sweep looks for.
    """
    rng = random.Random(20260815 + len(name))
    opts = {"order": 6, "prec": 128, "tol": 1e-6, "max_subdivisions": 256}
    checked = 0
    for _ in range(200):
        if name == "acosh":
            lo = 1.0 + rng.random() ** 4 * 30.0 + 1e-3
            hi = lo + rng.random() ** 3 * 2.0
        elif name == "atanh":
            c = (rng.random() - 0.5) * 1.9
            w = rng.random() ** 3 * (1.0 - abs(c)) * 0.9
            lo, hi = c - w, c + w
        else:
            c = (rng.random() - 0.5) * (20.0 if name == "asinh" else 16.0)
            w = rng.random() ** 3 * 3.0
            lo, hi = c - w, c + w
        try:
            r = _bound(name, lo, hi, **opts)
        except ak.ValidatedError:
            continue
        checked += 1
        _assert_covers(name, lo, hi, r, n=24)
    assert checked > 150, f"{name}: only {checked}/200 boxes produced a bound"
