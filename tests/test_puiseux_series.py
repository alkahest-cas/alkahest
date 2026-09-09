"""``alkahest.experimental.puiseux_series`` — fractional-exponent expansion.

The Rust side owns the mathematics (``alkahest-core/src/calculus/puiseux.rs``
carries the coefficient tests and the corrupted-expansion rejection test, and
``tests/silent_errors/corpus.py`` scores the refusals against their controls).
What this file covers is the **Python surface**: that the accessors report what
the Rust type reports, that the refusals arrive as coded ``SeriesError``s a
caller can branch on, and that the two documented footguns — ``Fraction``
exponents going through ``f64``, and ``series`` still refusing what this
expands — behave as documented rather than as folklore.
"""

from __future__ import annotations

from fractions import Fraction

import alkahest as ak
import pytest
from alkahest.experimental import puiseux_series

# The bindings raise the *native* exception type, which `alkahest.SeriesError`
# is; importing it from `alkahest.exceptions` gets the pure-Python subclass and
# `pytest.raises` on that would never match.
SeriesError = ak.SeriesError


@pytest.fixture
def pool_and_x():
    pool = ak.ExprPool()
    return pool, pool.symbol("x")


def _value_at(px, h: float) -> float:
    """Sum the truncated expansion at ``h``, from the exact ``.terms``."""
    return sum(float(ak.eval_expr(c, {})) * h ** float(e) for e, c in px.terms)


# ---------------------------------------------------------------------------
# The gap `series` cannot represent
# ---------------------------------------------------------------------------


def test_sqrt_has_a_half_integer_valuation(pool_and_x):
    pool, x = pool_and_x
    px = puiseux_series(ak.sqrt(x), x, pool.integer(0), 5)
    assert px.ramification == 2
    assert px.valuation == Fraction(1, 2)
    assert px.remainder_order == 5
    assert [(e, str(c)) for e, c in px.terms] == [(Fraction(1, 2), "1")]


def test_sqrt_of_sin_matches_the_textbook_expansion(pool_and_x):
    """``sqrt(sin x) = x^(1/2)(1 - x^2/12 + x^4/1440 - ...)``.

    Coefficients checked against the hand derivation recorded in the Rust
    tests; SymPy's ``series(sqrt(sin(x)), x, 0, 5)`` is the same.
    """
    pool, x = pool_and_x
    px = puiseux_series(ak.sqrt(ak.sin(x)), x, pool.integer(0), 5)
    assert px.ramification == 2
    assert [(e, str(c)) for e, c in px.terms] == [
        (Fraction(1, 2), "1"),
        (Fraction(5, 2), "-1/12"),
        (Fraction(9, 2), "1/1440"),
    ]
    # …and the truncation really does approximate the function.
    assert _value_at(px, 0.01) == pytest.approx(0.01**0.5 * (1 - 0.0001 / 12), rel=1e-15)


def test_half_power_times_an_analytic_factor(pool_and_x):
    pool, x = pool_and_x
    e = x.pow_expr(pool.rational(1, 2)) * ak.sin(x)
    px = puiseux_series(e, x, pool.integer(0), 5)
    assert px.valuation == Fraction(3, 2)
    assert [(e_, str(c)) for e_, c in px.terms] == [
        (Fraction(3, 2), "1"),
        (Fraction(7, 2), "-1/6"),
    ]


def test_a_cube_root_has_ramification_three(pool_and_x):
    pool, x = pool_and_x
    e = ak.sin(x).pow_expr(pool.rational(1, 3))
    px = puiseux_series(e, x, pool.integer(0), 5)
    assert px.ramification == 3
    assert [(e_, str(c)) for e_, c in px.terms] == [
        (Fraction(1, 3), "1"),
        (Fraction(7, 3), "-1/18"),
        (Fraction(13, 3), "-1/3240"),
    ]


def test_ramification_is_computed_not_read_off_the_input(pool_and_x):
    """``sqrt(x**2 + x**3) = x*sqrt(1+x)`` is single-valued at 0: index 1."""
    pool, x = pool_and_x
    px = puiseux_series(ak.sqrt(x**2 + x**3), x, pool.integer(0), 5)
    assert px.ramification == 1
    assert [e for e, _ in px.terms] == [1, 2, 3, 4]
    assert all(isinstance(e, int) for e, _ in px.terms)


# ---------------------------------------------------------------------------
# Refusals arrive as coded errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["log", "sqrt_times_log", "exp_of_inverse", "sin_of_inverse"],
)
def test_a_non_puiseux_shape_refuses_with_e_series_005(pool_and_x, name):
    pool, x = pool_and_x
    expr = {
        "log": lambda: ak.log(x),
        "sqrt_times_log": lambda: ak.sqrt(x) * ak.log(x),
        "exp_of_inverse": lambda: ak.exp(pool.integer(1) / x),
        "sin_of_inverse": lambda: ak.sin(pool.integer(1) / x),
    }[name]()
    with pytest.raises(SeriesError) as excinfo:
        puiseux_series(expr, x, pool.integer(0), 4)
    assert excinfo.value.code == "E-SERIES-005"
    assert excinfo.value.remediation


def test_order_zero_is_a_user_error(pool_and_x):
    pool, x = pool_and_x
    with pytest.raises(SeriesError) as excinfo:
        puiseux_series(ak.sqrt(x), x, pool.integer(0), 0)
    assert excinfo.value.code == "E-SERIES-002"


def test_a_branch_that_is_not_real_above_the_point_is_withheld(pool_and_x):
    """``sqrt(-x)`` at 0 needs a complex coefficient.

    Nothing can be measured just above the point, so the expansion is withheld
    (``E-SERIES-006``) rather than returned with ``(-1)**(1/2)`` inside it.
    """
    pool, x = pool_and_x
    with pytest.raises(SeriesError) as excinfo:
        puiseux_series(ak.sqrt(pool.integer(-1) * x), x, pool.integer(0), 3)
    assert excinfo.value.code == "E-SERIES-006"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda pool, x: ak.sqrt(x),
        lambda pool, x: ak.sqrt(ak.sin(x)),
        lambda pool, x: ak.sin(ak.sqrt(x)),
        lambda pool, x: ak.sin(x).pow_expr(pool.rational(1, 3)),
        lambda pool, x: x.pow_expr(pool.rational(1, 2)) * ak.sin(x),
    ],
)
def test_every_returned_expansion_carries_a_measured_decay_rate(pool_and_x, build):
    pool, x = pool_and_x
    px = puiseux_series(build(pool, x), x, pool.integer(0), 5)
    ev = px.evidence
    assert set(ev) == {"rungs", "conclusive_rungs", "worst_margin", "power_check_passed"}
    assert ev["rungs"] >= 1
    # The contract that makes the type mean something: an expansion whose rate
    # was never measured is not returned, so this is never 0.
    assert ev["conclusive_rungs"] >= 1
    # Tolerance is 0.55/e, capped at 0.4; the margin never sits below it.
    assert ev["worst_margin"] >= -0.55 / px.ramification - 1e-9


def test_the_exact_power_check_runs_where_it_applies(pool_and_x):
    """``sqrt(sin x)`` squared is ``sin x`` — checkable with no tolerance."""
    pool, x = pool_and_x
    px = puiseux_series(ak.sqrt(ak.sin(x)), x, pool.integer(0), 5)
    assert px.evidence["power_check_passed"] is True


# ---------------------------------------------------------------------------
# The two documented footguns
# ---------------------------------------------------------------------------


def test_a_non_dyadic_fraction_exponent_is_refused_not_rounded(pool_and_x):
    """``x ** Fraction(1, 3)`` is not ``x**(1/3)``.

    ``Expr.__pow__`` coerces through ``f64``, so the exponent that reaches the
    kernel is ``6004799503160661/18014398509481984``. Expanding *that* honestly
    needs a ramification index of ``2**54``; rounding it to ``1/3`` would be
    answering a question nobody asked. The refusal is the correct behaviour and
    the docstring points at ``pow_expr(pool.rational(1, 3))``.
    """
    pool, x = pool_and_x
    with pytest.raises(SeriesError) as excinfo:
        puiseux_series(x ** Fraction(1, 3), x, pool.integer(0), 3)
    assert excinfo.value.code == "E-SERIES-005"

    # The exact route works.
    px = puiseux_series(x.pow_expr(pool.rational(1, 3)), x, pool.integer(0), 3)
    assert px.ramification == 3
    assert px.valuation == Fraction(1, 3)


def test_a_dyadic_fraction_exponent_survives_f64_and_works(pool_and_x):
    """``Fraction(3, 2)`` *is* exact in binary, so ``x ** Fraction(3, 2)`` works."""
    pool, x = pool_and_x
    px = puiseux_series(x ** Fraction(3, 2), x, pool.integer(0), 4)
    assert px.ramification == 2
    assert px.valuation == Fraction(3, 2)


def test_series_still_refuses_what_puiseux_series_expands(pool_and_x):
    """The two entry points are siblings, not a replacement.

    ``series`` keeps returning ``E-SERIES-004`` for a fractional valuation —
    ``Series`` is a bare expression with no ramification to report and every
    consumer of ``local_expansion`` (``limit``, ``gruntz``, ``asymptotic``)
    reads an integer valuation. Widening it there is the change that could move
    a limit; this asserts it was not made.
    """
    pool, x = pool_and_x
    cases = [
        # A branch point: `series` forms `sqrt(0)**-1` as a coefficient and
        # refuses it as an indeterminate form.
        (ak.sqrt(x), "E-SERIES-004"),
        (ak.sqrt(ak.sin(x)), "E-SERIES-004"),
        # A literal non-integer power never even reaches a coefficient — `diff`
        # has no rule for it. A different code, the same refusal.
        (x ** Fraction(3, 2), "E-SERIES-001"),
    ]
    for expr, code in cases:
        with pytest.raises(SeriesError) as excinfo:
            ak.series(expr, x, pool.integer(0), 5)
        assert excinfo.value.code == code, str(expr)
        # …and the Puiseux route handles all three.
        assert puiseux_series(expr, x, pool.integer(0), 5).ramification == 2


def test_an_analytic_expansion_agrees_with_series(pool_and_x):
    """Where both apply the coefficients are identical, not merely close."""
    pool, x = pool_and_x
    for expr, order in [
        (ak.sin(x), 6),
        (ak.exp(x), 5),
        (ak.sqrt(pool.integer(1) + x), 5),
    ]:
        s = ak.series(expr, x, pool.integer(0), order)
        px = puiseux_series(expr, x, pool.integer(0), order)
        assert px.ramification == 1
        assert str(px.expr) == str(s.expr)


def test_repr_names_the_ramification(pool_and_x):
    pool, x = pool_and_x
    px = puiseux_series(ak.sqrt(x), x, pool.integer(0), 3)
    assert repr(px).startswith("PuiseuxExpansion(")
    assert "ramification=2" in repr(px)


def test_expansion_about_a_nonzero_point(pool_and_x):
    pool, x = pool_and_x
    px = puiseux_series(ak.sqrt(x - pool.integer(1)), x, pool.integer(1), 3)
    assert px.ramification == 2
    assert px.valuation == Fraction(1, 2)
    # `h` is `x - point`, not `x`, and the exponent is the half-integer.
    rendered = str(px.expr)
    assert "(1/2)" in rendered
    assert "x" in rendered
    assert "-1" in rendered


def test_a_budget_stops_the_expansion_and_is_attributed(pool_and_x):
    """A budget trip raises ``BudgetExceededError``, not ``E-SERIES-003``."""
    pool, x = pool_and_x
    with (
        ak.context(pool=pool, budget=ak.Budget(max_steps=3)),
        pytest.raises(ak.BudgetExceededError) as excinfo,
    ):
        puiseux_series(ak.sqrt(ak.sin(x)), x, pool.integer(0), 12)
    assert excinfo.value.code.startswith("E-BUDGET-")
