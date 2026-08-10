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
# Three-valued predicates — the third value is never collapsed
# ---------------------------------------------------------------------------


def test_no_roots_verified_true():
    """`(x-5)^2 + 1` has no root on [0,1] — provable."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = (x - pool.integer(5)) * (x - pool.integer(5)) + pool.integer(1)

    assert ak.verified_no_roots(f, _box(x, 0.0, 1.0)) == "true"


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
