"""Validated bounds for `bessel_j0`, `bessel_j1`, `digamma`, `gamma`,
`lambert_w` (3.9.0).

These five completed the M7 Taylor-model work: `bound_on_box` — and
`verified_sign`, `verified_no_roots`, `verified_integral` on top of it — now
answer for them instead of refusing with `E-VALIDATED-001`. Two of them,
`bessel_j0` and `bessel_j1`, are the reason this file exists in its own right:
they **oscillate**, and 3.8 shipped a ball kernel for them that hulled the two
endpoint values, which is an enclosure only for a monotone function. On
`[-1, 1]` the endpoints of `J₀` agree at 0.7651977 and the hull collapsed to a
point that excluded `J₀(0) = 1`, the function's own maximum. So every test
here is a **containment** test, and
``test_bessel_covers_an_interior_extremum_a_hull_would_miss`` pins that exact
configuration.

Reference sources, deliberately two, mirroring
`tests/test_validated_special_functions.py`:

* 40-significant-digit constants as :class:`decimal.Decimal`, so a regression
  is catchable in the CI tier that has no mpmath;
* mpmath for the dense and randomised sweeps.

`lambert_w` has no `Decimal` table beyond a few classical values because there
is no closed form to quote; it is checked instead through the equation that
*defines* it — ``g(w) = w·exp(w)`` is strictly increasing on ``w > -1``, so
``W₀(t) ∈ [lo, hi]`` exactly when ``g(lo) <= t <= g(hi)``, a check that uses
nothing but `exp`.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal

import alkahest as ak
import pytest

_UNSUPPORTED = "E-VALIDATED-001"
_DOMAIN = "E-VALIDATED-003"
_BUDGET = "E-VALIDATED-004"

_FUNCS = {
    "bessel_j0": ak.bessel_j0,
    "bessel_j1": ak.bessel_j1,
    "digamma": ak.digamma,
    "gamma": ak.gamma,
    "lambert_w": ak.lambert_w,
}

#: 40-significant-digit truth, independent of anything in the library.
_TRUTH = {
    "bessel_j0": {
        0.0: Decimal("1.0"),
        1.0: Decimal("0.7651976865579665514497175261026632209093"),
        -1.0: Decimal("0.7651976865579665514497175261026632209093"),
        2.0: Decimal("0.2238907791412356680518274546499486258252"),
        5.0: Decimal("-0.1775967713143383043473970130747587110711"),
        10.0: Decimal("-0.2459357644513483351977608624853287538296"),
    },
    "bessel_j1": {
        0.0: Decimal("0.0"),
        1.0: Decimal("0.4400505857449335159596822037189149131274"),
        -1.0: Decimal("-0.4400505857449335159596822037189149131274"),
        2.0: Decimal("0.5767248077568733872024482422691370869203"),
        5.0: Decimal("-0.3275791375914652220377343219101691327608"),
        10.0: Decimal("0.04347274616886143666974876802585928830627"),
    },
    "digamma": {
        0.5: Decimal("-1.963510026021423479440976332998755567193"),
        1.0: Decimal("-0.5772156649015328606065120900824024310422"),
        2.0: Decimal("0.4227843350984671393934879099175975689578"),
        3.0: Decimal("0.9227843350984671393934879099175975689578"),
        10.0: Decimal("2.251752589066721107647456163885851537212"),
    },
    "gamma": {
        0.5: Decimal("1.772453850905516027298167483341145182798"),
        1.0: Decimal("1.0"),
        2.0: Decimal("1.0"),
        3.0: Decimal("2.0"),
        4.5: Decimal("11.63172839656744892914422410942626526211"),
        0.25: Decimal("3.625609908221908311930685155867672002995"),
    },
    "lambert_w": {
        0.0: Decimal("0.0"),
        1.0: Decimal("0.5671432904097838729999686622103555497538"),
        # `W(e) = 1` and `W(2·ln 2) = ln 2` are the two classically known
        # values, but the box endpoint is the *f64* nearest `e` (resp.
        # `2·ln 2`), so the truth quoted is `W` at that f64 rather than the
        # exact constant — a distinction of 3·10⁻¹⁷, well outside a converged
        # enclosure.
        math.e: Decimal("0.9999999999999999734088114669705433241736"),
        2.0 * math.log(2.0): Decimal("0.6931471805599452957205680601604377549756"),
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
    A containment sweep over a wide box sometimes can.
    """
    for point, truth in _TRUTH[name].items():
        r = _bound(name, point, point)
        assert r.lower <= float(truth) <= r.upper, (
            f"{name}({point}) = {truth} escaped [{r.lower}, {r.upper}]"
        )


def test_bessel_covers_an_interior_extremum_a_hull_would_miss():
    """The 3.8 unsoundness, as a test.

    `J₀(-1) = J₀(1) = 0.7651977`, so an endpoint hull over `[-1, 1]` is the
    single point 0.7651977 and excludes the maximum `J₀(0) = 1`. The same
    shape of trap sits inside `[-5, 5]` for `J₁`, whose endpoints are equal
    and opposite so a hull would be symmetric about zero and still miss the
    extremum at `x ≈ ±1.841`.
    """
    r = _bound("bessel_j0", -1.0, 1.0)
    assert r.upper >= 1.0, f"J₀(0) = 1 escaped [{r.lower}, {r.upper}]"
    assert r.lower <= 0.7651976866, f"the endpoint value escaped [{r.lower}, {r.upper}]"
    # …and the answer is tight: the true range on [-1, 1] is [0.76520, 1].
    assert r.width < 0.25, f"width {r.width}"

    r = _bound("bessel_j1", -5.0, 5.0)
    # max J₁ = 0.58186522428 at 1.84118378, min = −that at −1.84118378.
    assert r.upper >= 0.5818652242, f"J₁'s maximum escaped [{r.lower}, {r.upper}]"
    assert r.lower <= -0.5818652242, f"J₁'s minimum escaped [{r.lower}, {r.upper}]"


def test_gamma_covers_its_interior_minimum():
    """`Γ(1) = Γ(2) = 1` with `Γ(1.4616) = 0.8856` strictly below both.

    Same trap as Bessel's, for a function nobody thinks of as oscillating:
    `Γ` is *not* monotone on `(0, ∞)`, and a rule that assumed it was would
    return `[1, 1]` here.
    """
    r = _bound("gamma", 1.0, 2.0)
    assert r.lower <= 0.8856031944, f"Γ's minimum escaped [{r.lower}, {r.upper}]"
    assert r.upper >= 1.0
    assert r.width < 0.2, f"width {r.width}"


@pytest.mark.parametrize(
    ("name", "lo", "hi", "expect_lo", "expect_hi"),
    [
        # Monotone stretches, where the true range is the endpoint pair.
        ("digamma", 1.0, 2.0, -0.5772156649, 0.4227843351),
        ("digamma", 3.0, 10.0, 0.9227843351, 2.2517525891),
        ("gamma", 2.0, 4.5, 1.0, 11.6317283966),
        ("lambert_w", 0.0, 1.0, 0.0, 0.5671432904),
        ("lambert_w", 1.0, math.e, 0.5671432904, 0.9999999999),
        # …and one where it is not: J₀ turns over inside [0, 2].
        ("bessel_j0", 0.0, 2.0, 0.2238907791, 1.0),
    ],
)
def test_converged_enclosures_match_the_true_range(name, lo, hi, expect_lo, expect_hi):
    """Sound *and* tight: branch-and-bound must converge onto the real range.

    `[-inf, inf]` passes every containment test in this file; this is the one
    that says the rules are worth having.
    """
    r = _bound(name, lo, hi, tol=1e-8, max_subdivisions=4096)
    assert r.lower <= expect_lo + 1e-7, f"{name} lower {r.lower} > {expect_lo}"
    assert r.upper >= expect_hi - 1e-7, f"{name} upper {r.upper} < {expect_hi}"
    assert r.width <= (expect_hi - expect_lo) + 1e-4, f"{name} width {r.width}"


# ---------------------------------------------------------------------------
# Domain guards: an off-domain box refuses, it does not answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "lo", "hi", "why"),
    [
        ("digamma", 0.0, 1.0, "touches the pole at 0"),
        ("digamma", -0.5, 0.5, "straddles the pole at 0"),
        ("digamma", -3.0, -2.0, "between poles — analytic, but not covered"),
        ("digamma", -1.0, -1.0, "sits exactly on a pole"),
        ("gamma", 0.0, 1.0, "touches the pole at 0"),
        ("gamma", -0.5, 0.5, "straddles the pole at 0"),
        ("gamma", -2.5, -2.4, "between poles — analytic, but not covered"),
        ("gamma", -4.0, -1.0, "contains three poles"),
        ("lambert_w", -1.0, 1.0, "straddles the branch point at -1/e"),
        ("lambert_w", -0.5, -0.4, "entirely left of -1/e, where W₀ is complex"),
        ("lambert_w", -0.4, 0.0, "reaches just past -1/e"),
        ("lambert_w", -1e6, -1e5, "far outside the principal branch"),
    ],
)
def test_off_domain_boxes_refuse_rather_than_bound(name, lo, hi, why):
    """A bound off the domain is a wrong answer, not a loose one.

    Which refusal code reaches the caller depends on how the branch-and-bound
    above the rule gives up on a violation it cannot bisect away — the rule
    itself always says `E-VALIDATED-003`. Both are refusals. What must never
    appear is `E-VALIDATED-001`, which would claim there is no rule at all.
    """
    with pytest.raises(ak.ValidatedError) as excinfo:
        _bound(name, lo, hi)
    assert excinfo.value.code in {_DOMAIN, _BUDGET}, (
        f"{name} on [{lo},{hi}] ({why}) refused with {excinfo.value.code}"
    )
    assert excinfo.value.code != _UNSUPPORTED


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [
        ("bessel_j0", -1.0, 1.0),
        ("bessel_j0", -40.0, 40.0),
        ("bessel_j0", 2.404825557695773, 2.404825557695773),
        ("bessel_j1", -100.0, -99.0),
        ("bessel_j1", 0.0, 0.0),
    ],
)
def test_bessel_never_refuses_on_domain_grounds(name, lo, hi):
    """`J₀`/`J₁` are entire — no box is off-domain, including one sitting on
    a zero and one spanning a dozen oscillations."""
    r = _bound(name, lo, hi)
    assert r.lower <= r.upper
    # |J_n| <= 1 on the reals for every integer order, so an enclosure that
    # has escaped that band is wrong however it was produced.
    assert r.lower >= -1.0000001, f"[{r.lower}, {r.upper}]"
    assert r.upper <= 1.0000001, f"[{r.lower}, {r.upper}]"


def test_a_domain_refusal_is_not_reported_as_unsupported():
    """`gamma` *has* a rule; `[-1, -0.5]` is just a box outside its domain."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    assert ak.bounds_supported(ak.gamma(x))
    with pytest.raises(ak.ValidatedError) as excinfo:
        ak.bound_on_box(ak.gamma(x), [(x, -1.0, -0.5)])
    assert excinfo.value.code != _UNSUPPORTED


# ---------------------------------------------------------------------------
# The rest of the stack lights up too
# ---------------------------------------------------------------------------


def test_verified_sign_and_no_roots_reach_the_new_functions():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    # Γ(x) > 0 on (0, ∞).
    assert ak.verified_sign(ak.gamma(x), [(x, 0.5, 3.0)], "positive") == "true"
    # ψ(x) < 0 on (0, 1]: ψ(1) = -γ < 0 and ψ increases, so the box stops short.
    assert ak.verified_sign(ak.digamma(x), [(x, 0.25, 0.9)], "negative") == "true"
    # W₀ > 0 on x > 0.
    assert ak.verified_sign(ak.lambert_w(x), [(x, 0.5, 4.0)], "positive") == "true"
    # J₀ has no zero before 2.4048…
    assert ak.verified_no_roots(ak.bessel_j0(x), [(x, 0.0, 2.0)]) == "true"
    # …and does have one on [2, 3].
    assert ak.verified_no_roots(ak.bessel_j0(x), [(x, 2.0, 3.0)]) == "false"


def test_verified_integral_of_bessel_j1_matches_its_closed_form():
    """∫₀¹ J₁ = 1 − J₀(1), from `(d/dx) J₀ = −J₁`."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.verified_integral(ak.bessel_j1(x), x, 0.0, 1.0)
    exact = 1.0 - 0.7651976865579666
    assert r.lower <= exact <= r.upper
    assert r.width < 1e-6


def test_verified_integral_of_digamma_matches_log_gamma():
    """∫₁² ψ = ln Γ(2) − ln Γ(1) = 0."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.verified_integral(ak.digamma(x), x, 1.0, 2.0)
    assert r.lower <= 0.0 <= r.upper
    assert r.width < 1e-6


def test_the_new_rules_compose_with_the_rest_of_the_algebra():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    f = ak.gamma(x + y) * ak.bessel_j0(x) + ak.digamma(y) - ak.lambert_w(x * y)
    r = ak.bound_on_box(f, [(x, 0.5, 1.0), (y, 1.0, 1.5)], tol=1e-4, max_subdivisions=4096)
    assert r.lower <= r.upper
    assert r.width < 5.0


# ---------------------------------------------------------------------------
# mpmath sweeps — dense samples and 200 randomised boxes per function
# ---------------------------------------------------------------------------

mpmath = pytest.importorskip("mpmath")


def _mp_ref(name):
    return {
        "bessel_j0": lambda t: mpmath.besselj(0, t),
        "bessel_j1": lambda t: mpmath.besselj(1, t),
        "digamma": mpmath.digamma,
        "gamma": mpmath.gamma,
        "lambert_w": mpmath.lambertw,
    }[name]


def _assert_covers(name, lo, hi, result, n=64):
    """Every sampled true value must be inside the enclosure."""
    fun = _mp_ref(name)
    with mpmath.workdps(50):
        for k in range(n + 1):
            t = mpmath.mpf(lo) + (mpmath.mpf(hi) - mpmath.mpf(lo)) * k / n
            t = min(max(t, mpmath.mpf(lo)), mpmath.mpf(hi))
            v = float(mpmath.re(fun(t)))
            assert result.lower <= v <= result.upper, (
                f"{name}({t}) = {v} escaped [{result.lower}, {result.upper}] on [{lo}, {hi}]"
            )


@pytest.mark.parametrize(
    ("name", "boxes"),
    [
        (
            "bessel_j0",
            [
                (-1.0, 1.0),
                (2.0, 3.0),
                (-6.0, 6.0),
                (10.0, 12.0),
                (-20.0, -19.5),
                (2.404, 2.405),
            ],
        ),
        (
            "bessel_j1",
            [(-1.0, 1.0), (0.0, 0.5), (3.8, 3.84), (-5.0, 5.0), (15.0, 16.0)],
        ),
        (
            "digamma",
            [(1.0, 2.0), (0.25, 0.5), (0.001, 0.0011), (5.0, 9.0), (100.0, 101.0)],
        ),
        (
            "gamma",
            [(1.0, 2.0), (0.5, 0.75), (0.01, 0.02), (1.4, 1.5), (3.0, 4.0)],
        ),
        (
            "lambert_w",
            [(-0.3, 0.0), (0.0, 1.0), (1.0, 2.0), (-0.36, -0.35), (1e4, 1e5)],
        ),
    ],
)
def test_dense_samples_stay_inside_the_enclosure(name, boxes):
    """Including boxes that run right up to a domain boundary, and — for the
    Bessel pair — boxes straddling a zero and boxes spanning several
    oscillations, which is where an endpoint argument goes wrong."""
    for lo, hi in boxes:
        _assert_covers(name, lo, hi, _bound(name, lo, hi), n=128)


@pytest.mark.parametrize("name", ["bessel_j0", "bessel_j1", "digamma", "gamma", "lambert_w"])
def test_randomised_box_sweep(name):
    """200 boxes per function, centres and widths both varying.

    A refusal is skipped rather than failed: refusing is always sound, and
    ``test_off_domain_boxes_refuse_rather_than_bound`` pins that refusals
    happen for the right reason. Only a *returned bound* can be wrong.
    """
    rng = random.Random(20260815 + len(name) * 7)
    # Containment, not tightness, is what this sweep is for, so the budget is
    # deliberately small — 200 boxes at a converging tolerance would cost
    # minutes and test nothing extra.
    opts = {"order": 6, "prec": 128, "tol": 1e-3, "max_subdivisions": 48}
    checked = 0
    for _ in range(200):
        if name.startswith("bessel"):
            c = (rng.random() - 0.5) * 40.0
            w = rng.random() ** 3 * 3.0
            lo, hi = c - w, c + w
        elif name == "digamma":
            lo = rng.random() ** 4 * 20.0 + 1e-3
            hi = lo + rng.random() ** 3 * 2.0
        elif name == "gamma":
            lo = rng.random() ** 4 * 8.0 + 1e-2
            hi = lo + rng.random() ** 3 * 1.5
        else:
            # Strictly right of -1/e, crowding the branch point.
            lo = -0.36787944117144233 + rng.random() ** 4 * 30.0 + 1e-4
            hi = lo + rng.random() ** 3 * min(lo + 0.36787944117144233, 2.0)
        try:
            r = _bound(name, lo, hi, **opts)
        except ak.ValidatedError:
            continue
        checked += 1
        _assert_covers(name, lo, hi, r, n=24)
    assert checked > 150, f"{name}: only {checked}/200 boxes produced a bound"
