"""P1 item 9 — rigorous global bounds (Taylor models / validated numerics).

Ball arithmetic gives rigorous *pointwise* enclosures. This module gives
rigorous *global* ones: the range of `f` over a box, a definite integral, the
absence of roots.

The contract is soundness over tightness — a returned enclosure may be wide,
but it must never be wrong. Most of these tests are therefore containment
tests: they check that the enclosure really does contain values the function
actually takes, including in the cases where naive interval arithmetic is loose
(the dependency problem) and where a narrowing bug would be tempting.
"""

import math

import alkahest as ak
import pytest


def _box(var, lo, hi):
    return [(var, lo, hi)]


# ---------------------------------------------------------------------------
# Soundness: the enclosure must contain the true values
# ---------------------------------------------------------------------------


def test_enclosure_contains_densely_sampled_values():
    """The strongest cheap check against unsound narrowing."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.sin(x) * ak.exp(x) + x * x

    r = ak.bound_on_box(f, _box(x, -2.0, 2.0))

    for i in range(401):
        xi = -2.0 + 4.0 * i / 400
        true = math.sin(xi) * math.exp(xi) + xi * xi
        assert r.lower <= true <= r.upper, f"enclosure missed f({xi}) = {true}"


def test_dependency_problem_x_minus_x():
    """`x - x` is identically 0; naive intervals give [-1, 1]."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    r = ak.bound_on_box(x - x, _box(x, -1.0, 1.0))

    assert r.contains(0.0)
    assert r.width < 1e-15


def test_dependency_problem_x_times_one_minus_x():
    """`x(1-x)` on [0,1] has true range [0, 1/4]; naive intervals give [0,1]."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = x * (pool.integer(1) - x)

    r = ak.bound_on_box(f, _box(x, 0.0, 1.0))

    # Sound: contains the true range.
    assert r.lower <= 0.0
    assert r.upper >= 0.25
    # And tight: much better than the naive [0, 1].
    assert r.upper < 0.30


def test_two_dimensional_box():
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    f = x * x + y * y

    r = ak.bound_on_box(f, [(x, -1.0, 1.0), (y, -1.0, 1.0)])

    assert r.lower <= 0.0
    assert r.upper >= 2.0


# ---------------------------------------------------------------------------
# Verified integrals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "a", "b", "exact"),
    [
        (lambda pool, x: ak.sin(x), 0.0, math.pi, 2.0),
        (lambda pool, x: ak.exp(x), 0.0, 1.0, math.e - 1.0),
        (lambda pool, x: x * x, 0.0, 3.0, 9.0),
    ],
)
def test_verified_integral_encloses_the_exact_value(build, a, b, exact):
    pool = ak.ExprPool()
    x = pool.symbol("x")

    r = ak.verified_integral(build(pool, x), x, a, b)

    assert r.lower <= exact <= r.upper, f"{exact} not in [{r.lower}, {r.upper}]"
    assert r.width < 1e-6


def test_verified_integral_is_an_enclosure_not_an_estimate():
    """A wide-but-true answer under a tiny budget beats a tight-but-false one."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    r = ak.verified_integral(ak.sin(x), x, 0.0, math.pi, max_subdivisions=1)

    assert r.lower <= 2.0 <= r.upper


# ---------------------------------------------------------------------------
# Removable singularities: the integrand is singular, the integral is not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "a", "b", "exact", "label"),
    [
        (
            lambda pool, x: ak.log(pool.integer(1) + x) / x,
            0.0,
            1.0,
            math.pi**2 / 12,
            "log(1+x)/x",
        ),
        (lambda pool, x: ak.sin(x) / x, -1.0, 1.0, 1.8921661407343664, "sin(x)/x"),
        (lambda pool, x: ak.sin(x) / x, 0.0, 1.0, 0.9460830703671832, "sin(x)/x on [0,1]"),
        (
            lambda pool, x: (ak.exp(x) - pool.integer(1)) / x,
            0.0,
            1.0,
            1.3179021514544038,
            "(exp(x)-1)/x",
        ),
        (
            lambda pool, x: (pool.integer(1) - ak.cos(x)) / x,
            0.0,
            1.0,
            0.23981174200056,
            "(1-cos x)/x = Cin(1)",
        ),
    ],
)
def test_removable_singularity_encloses_the_exact_value(build, a, b, exact, label):
    """The enclosure has to *bracket* the truth — returning one is worthless
    if it is wrong. These integrands are all undefined at a point of the
    interval and extend continuously across it."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    r = ak.verified_integral(build(pool, x), x, a, b)

    assert r.lower <= exact <= r.upper, f"{label}: {exact} not in [{r.lower}, {r.upper}]"
    assert r.width < 1e-6, f"{label}: enclosure width {r.width}"


def test_removable_branch_agrees_with_ordinary_quadrature_on_the_regular_part():
    """Splitting the interval so that only one half touches the singularity
    must not move the answer: a biased removable branch would show up here."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.log(pool.integer(1) + x) / x

    whole = ak.verified_integral(f, x, 0.0, 1.0)
    left = ak.verified_integral(f, x, 0.0, 0.25)
    right = ak.verified_integral(f, x, 0.25, 1.0)

    assert whole.lower <= left.upper + right.upper
    assert whole.upper >= left.lower + right.lower


def test_a_genuine_pole_is_still_refused():
    """`1/x` on [-1,1]: the numerator does not vanish, so nothing is removable."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ak.ValidatedError) as excinfo:
        ak.verified_integral(pool.integer(1) / x, x, -1.0, 1.0)

    assert excinfo.value.code == "E-VALIDATED-003"


def test_a_double_pole_with_a_simple_numerator_zero_is_still_refused():
    """`sin(x)/x**2 ~ 1/x` does not converge; the numerator's zero is only
    order 1 against the denominator's order 2, and `D' = 2x` vanishes."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ak.ValidatedError):
        ak.verified_integral(ak.sin(x) / (x * x), x, -1.0, 1.0)


def test_a_second_order_removable_singularity_is_refused_not_guessed():
    """`(1-cos x)/x**2` really is removable (it tends to 1/2), but the proof
    needs a *second*-order argument: `D' = 2x` vanishes at 0, so Cauchy's mean
    value theorem does not apply and the enclosure is declined rather than
    stretched to fit."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ak.ValidatedError):
        ak.verified_integral((pool.integer(1) - ak.cos(x)) / (x * x), x, 0.0, 1.0)


@pytest.mark.parametrize(
    ("build", "a", "b", "label"),
    [
        (lambda pool, x: -ak.log(x), 0.0, 1.0, "-log x"),
        (lambda pool, x: ak.log(x) * ak.log(x), 0.0, 1.0, "(log x)^2"),
        (lambda pool, x: ak.exp(x * ak.log(x)), 0.0, 1.0, "x^x"),
    ],
)
def test_integrable_but_not_removable_singularities_refuse_with_an_honest_message(
    build, a, b, label
):
    """These integrals all exist. What does not exist is a rigorous enclosure
    of the *integrand*, and the error text must say which of the two it means
    rather than implying the integral is undefined."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ak.ValidatedError) as excinfo:
        ak.verified_integral(build(pool, x), x, a, b)

    message = str(excinfo.value)
    assert "integrand is singular" in message, f"{label}: {message}"
    assert "integrable singularity" in message, f"{label}: {message}"
    assert excinfo.value.remediation


# ---------------------------------------------------------------------------
# Three-valued predicates — the third value is never collapsed
# ---------------------------------------------------------------------------


def test_no_roots_verified_true():
    """`(x-5)^2 + 1` has no root on [0,1] — provable."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = (x - pool.integer(5)) * (x - pool.integer(5)) + pool.integer(1)

    assert ak.verified_no_roots(f, _box(x, 0.0, 1.0)) == "true"


@pytest.mark.parametrize(
    ("lo", "hi", "roots"),
    [
        (-2.0, 0.0, "1 root, endpoint signs + -> -"),
        (0.0, 2.0, "1 root, endpoint signs - -> +"),
        (1.3, 1.5, "1 root, endpoint signs - -> +"),
        (-2.0, 2.0, "2 roots, endpoint signs + -> +"),
        (-10.0, 10.0, "2 roots, endpoint signs + -> +"),
    ],
)
def test_no_roots_false_whatever_the_root_count(lo, hi, roots):
    """`x**2 - 2` has provable roots on every one of these boxes. An even
    number of them used to defeat the test, because only the box's *own*
    endpoints were checked for a sign change."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    assert ak.verified_no_roots(x * x - pool.integer(2), [(x, lo, hi)]) == "false", roots


def test_no_roots_false_when_the_roots_hide_behind_a_positive_factor():
    """`(x**2-2)(x**2+1)` has the same two roots; the second factor never
    vanishes and never changes the sign pattern at the endpoints."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = (x * x - pool.integer(2)) * (x * x + pool.integer(1))

    assert ak.verified_no_roots(f, [(x, -2.0, 2.0)]) == "false"


def test_no_roots_false_across_a_multivariate_box():
    """A box is convex, so two points of opposite proven sign certify a root
    anywhere in it — `x - y` is +2 at (1,-1) and -2 at (-1,1)."""
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")

    assert ak.verified_no_roots(x - y, [(x, -1.0, 1.0), (y, -1.0, 1.0)]) == "false"


@pytest.mark.parametrize(
    ("build", "lo", "hi"),
    [
        (lambda pool, x: x * x + pool.integer(1), -10.0, 10.0),
        (lambda pool, x: x * x + pool.integer(2), -2.0, 2.0),
        (lambda pool, x: ak.exp(x), -5.0, 5.0),
        (
            lambda pool, x: (x - pool.integer(5)) * (x - pool.integer(5)) + pool.integer(1),
            0.0,
            1.0,
        ),
    ],
)
def test_no_roots_true_cases_stay_true(build, lo, hi):
    """The existence search must never be able to turn a proven `"true"` into
    anything else — it only runs once the enclosure already contains zero."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    assert ak.verified_no_roots(build(pool, x), [(x, lo, hi)]) == "true"


@pytest.mark.parametrize(
    ("build", "lo", "hi", "why"),
    [
        (
            lambda pool, x: (x - pool.integer(1)) * (x - pool.integer(1)),
            0.0,
            2.0,
            "double root at x=1: no sign change anywhere",
        ),
        (
            lambda pool, x: (x * x - pool.integer(1)) * (x * x - pool.integer(1)),
            -2.0,
            2.0,
            "two double roots at x=+-1",
        ),
    ],
)
def test_a_root_that_cannot_be_witnessed_stays_undecided(build, lo, hi, why):
    """These expressions *do* have roots in the box, but they never change
    sign, so no intermediate-value witness exists. `"undecided"` is the honest
    answer; reporting `"false"` here would be a guess dressed as a proof."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    assert ak.verified_no_roots(build(pool, x), [(x, lo, hi)]) == "undecided", why


def test_sign_positive_verified_true():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    assert ak.verified_sign(ak.exp(x), _box(x, -5.0, 5.0), "positive") == "true"


def test_sign_false_is_certified_by_a_witness():
    """`x > 0` on [-1,1] is false; one point disproves a "for all" claim."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    assert ak.verified_sign(x, _box(x, -1.0, 1.0), "positive") == "false"


def test_verdicts_are_the_three_documented_strings():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    for predicate in ("positive", "negative", "nonnegative", "nonpositive"):
        v = ak.verified_sign(x, _box(x, -1.0, 1.0), predicate)
        assert v in ("true", "false", "undecided")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_singularity_in_the_box_is_refused():
    """`1/x` on a box containing 0 has no bounded range — refuse, don't guess."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = pool.integer(1) / x

    with pytest.raises(ak.ValidatedError) as excinfo:
        ak.bound_on_box(f, _box(x, -1.0, 1.0))

    assert excinfo.value.code.startswith("E-VALIDATED-")
    assert excinfo.value.remediation


def test_unbound_symbol_is_refused():
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")

    with pytest.raises(ak.ValidatedError):
        ak.bound_on_box(x + y, _box(x, 0.0, 1.0))


def test_empty_box_is_refused():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ValueError):
        ak.bound_on_box(x, [])


def test_bad_predicate_name_is_refused():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ValueError):
        ak.verified_sign(x, _box(x, 0.0, 1.0), "mostly_positive")


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_exported_surface():
    for name in (
        "bound_on_box",
        "verified_integral",
        "verified_no_roots",
        "verified_sign",
        "Enclosure",
        "ValidatedError",
    ):
        assert name in ak.__all__, name
    assert issubclass(ak.ValidatedError, ak.AlkahestError)


def test_enclosure_repr_and_helpers():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    r = ak.bound_on_box(x, _box(x, 0.0, 1.0))

    assert "Enclosure(" in repr(r)
    assert r.width >= 1.0
    assert r.subdivisions >= 0
    assert isinstance(r.budget_exhausted, bool)


# ---------------------------------------------------------------------------
# Inequalities that are tight where the box ends
#
# The classical trigonometric inequalities are asymptotically tight as x -> 0.
# Subdivision alone provably cannot certify them there: wherever the margin goes
# to zero, every enclosure of the range straddles zero however fine the boxes
# get. `verified_sign` therefore splits -- a truncated Taylor expansion with a
# proven Lagrange remainder on a collar at the endpoint, branch-and-bound on the
# rest -- and the two pieces share their join point, so nothing is left out.
#
# The soundness half of this section matters more than the reach half: a `true`
# is a certificate, so every "must not be true" case below is load-bearing.
# ---------------------------------------------------------------------------


_CLASSICAL_NAMES = ("cusa_huygens", "huygens", "mitrinovic_adamovic", "wilker")


def _classical(pool, x):
    """The four classical inequalities, cleared of denominators.

    Each is stated as `f(x) >= 0` on `(0, pi/2)`, tight as `x -> 0`:

    * Cusa-Huygens        `sin x / x < (2 + cos x) / 3`
    * Mitrinovic-Adamovic `(sin x / x)^3 > cos x`
    * Wilker              `(sin x / x)^2 + tan x / x > 2`
    * Huygens             `2 sin x / x + tan x / x > 3`

    Wilker and Huygens are multiplied through by `x^2 cos x` and `x cos x`,
    which are positive on the open interval, so the sign is unchanged.
    """
    return {
        "cusa_huygens": x * (pool.integer(2) + ak.cos(x)) - pool.integer(3) * ak.sin(x),
        "mitrinovic_adamovic": ak.sin(x) ** 3 - x**3 * ak.cos(x),
        "wilker": ak.sin(x) ** 2 * ak.cos(x) + x * ak.sin(x) - pool.integer(2) * x**2 * ak.cos(x),
        "huygens": pool.integer(2) * ak.sin(x) * ak.cos(x)
        + ak.sin(x)
        - pool.integer(3) * x * ak.cos(x),
    }


@pytest.mark.parametrize("name", _CLASSICAL_NAMES)
@pytest.mark.parametrize("lo", [0.0, 0.01, 0.1])
def test_classical_trig_inequalities_are_certified_up_to_the_tight_endpoint(name, lo):
    """All four, including at `x = 0` itself where the margin vanishes."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    assert ak.verified_sign(_classical(pool, x)[name], _box(x, lo, 1.5), "nonnegative") == "true"


def test_jordan_inequality_with_an_exactly_rational_bound():
    """`sin x >= (2/pi) x`, stated exactly.

    `pi` is a plain symbol here, so the constant is rationalised instead:
    `636619772368/10^12 > 2/pi`, which makes `D sin x - N x >= 0` *stronger*
    than Jordan's inequality on the same interval. It is tight at `x = 0`.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.sin(x) * pool.integer(10**12) - pool.integer(636619772368) * x

    assert ak.verified_sign(f, _box(x, 0.0, 1.5), "nonnegative") == "true"


def test_jordan_with_too_large_a_constant_is_refuted_at_the_far_endpoint():
    """The same constant on `[0, pi/2]`, where `N/D > 2/pi` makes it false."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.sin(x) * pool.integer(10**12) - pool.integer(636619772368) * x

    assert ak.verified_sign(f, _box(x, 0.0, math.pi / 2), "nonnegative") == "false"


@pytest.mark.parametrize("name", _CLASSICAL_NAMES)
def test_reversing_a_tight_inequality_never_certifies_it(name):
    """The reverse of each is false, and tight at the same endpoint.

    This is the control that makes the endpoint machinery falsifiable: a sign
    error in the series argument would show up here as a `true`.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    reversed_f = pool.integer(-1) * _classical(pool, x)[name]

    assert ak.verified_sign(reversed_f, _box(x, 0.0, 1.5), "nonnegative") != "true"


def test_false_only_in_a_tiny_neighbourhood_of_the_endpoint_is_not_certified():
    """`x^3 - x^2/1000` is negative exactly on `(0, 1/1000)`.

    The violation is invisible to endpoint and centre sampling and is far
    narrower than the default tolerance, so nothing but the endpoint expansion
    can see it at all. It must never come back `true`.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = x**3 - pool.rational(1, 1000) * x**2

    assert ak.verified_sign(f, _box(x, 0.0, 1.5), "nonnegative") != "true"


def test_a_strict_inequality_is_false_where_the_function_vanishes_exactly():
    """`x^2 > 0` fails at `x = 0`; `x^2 >= 0` holds."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    assert ak.verified_sign(x**2, _box(x, 0.0, 1.0), "positive") == "false"
    assert ak.verified_sign(x**2, _box(x, 0.0, 1.0), "nonnegative") == "true"


def test_tightness_away_from_an_endpoint_stays_undecided():
    """A margin that vanishes in the *interior* is still out of reach.

    `(x - 7/10)^2 (x + 1)` is non-negative on `[0, 3/2]` and touches zero at
    `x = 7/10`. The endpoint expansion does not apply there, and the honest
    answer remains `undecided` -- it must not be upgraded to a wrong `true`.
    """
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = (x - pool.rational(7, 10)) ** 2 * (x + pool.integer(1))

    assert ak.verified_sign(f, _box(x, 0.0, 1.5), "nonnegative") == "undecided"


def test_a_shallow_interior_dip_is_never_certified_true():
    """`(x - 7/10)^2 - 10^-6` dips below zero only near `x = 7/10`."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = (x - pool.rational(7, 10)) ** 2 - pool.rational(1, 10**6)

    assert ak.verified_sign(f, _box(x, 0.0, 1.5), "nonnegative") != "true"


# ---------------------------------------------------------------------------
# Termination: a tolerance the search cannot reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "digits",
    [3, 6, 9, 12],
)
def test_verified_sign_terminates_for_any_size_of_rational_constant(digits):
    """Rationalising a constant to more digits must not change the cost class.

    Before the fix, `N/D` at 12 digits ran for over 300 s while 9 digits took
    0.08 s: a sub-box bisected down to the width floor was pushed back onto the
    active list and immediately re-selected, so the loop spun without ever
    consuming its subdivision budget. Three extra digits were enough to push the
    tolerance out of reach and trigger it.
    """
    import time

    d = 10**digits
    n = int(0.636619772368 * d)
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.sin(x) * pool.integer(d) - pool.integer(n) * x

    start = time.monotonic()
    verdict = ak.verified_sign(f, _box(x, 0.0, 1.5), "nonnegative")
    elapsed = time.monotonic() - start

    assert verdict in ("true", "false", "undecided")
    assert elapsed < 60.0, f"{digits} digits took {elapsed:.1f}s"


def test_bound_on_box_terminates_when_the_tolerance_is_unreachable():
    """`tol` below what the width floor can deliver must stop, not spin."""
    import time

    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.sin(x) * pool.integer(10**12) - pool.integer(636619772368) * x

    start = time.monotonic()
    r = ak.bound_on_box(f, _box(x, 0.0, 1.5), tol=1e-40)
    elapsed = time.monotonic() - start

    assert elapsed < 60.0, f"took {elapsed:.1f}s"
    assert r.lower <= 0.0 <= r.upper
