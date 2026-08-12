"""V2-15 — user-facing series() with BigO remainder."""

import alkahest
import pytest


def _has_big_o(expr: alkahest.Expr) -> bool:
    n = expr.node()
    tag = n[0]
    if tag == "big_o":
        return True
    if tag == "add":
        return any(_has_big_o(c) for c in n[1])
    if tag == "mul":
        return any(_has_big_o(c) for c in n[1])
    if tag == "pow":
        return _has_big_o(n[1]) or _has_big_o(n[2])
    if tag == "func":
        return any(_has_big_o(c) for c in n[2])
    return False


def test_series_cos_about_zero():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    cx = alkahest.cos(x)
    s = alkahest.series(cx, x, z, 6)
    assert _has_big_o(s.expr)


def test_series_inv_x_laurent():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    ix = x ** (-1)
    s = alkahest.series(ix, x, z, 4)
    assert _has_big_o(s.expr)
    t = str(s.expr)
    assert "O(" in t


def test_big_o_expr_node():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    o = p.big_o(x**6)
    assert o.node()[0] == "big_o"


def test_series_order_zero_raises():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    with pytest.raises(alkahest.SeriesError):
        alkahest.series(x, x, p.integer(0), 0)


def test_series_accepts_bare_int_point():
    """series()'s `point` accepts a bare Python int, coerced into expr's pool."""
    p = alkahest.ExprPool()
    x = p.symbol("x")
    cx = alkahest.cos(x)
    s_int = alkahest.series(cx, x, 0, 6)
    s_expr = alkahest.series(cx, x, p.integer(0), 6)
    assert str(s_int.expr) == str(s_expr.expr)


# ---------------------------------------------------------------------------
# Termination: a runaway expansion refuses instead of running forever, and
# never returns a shorter series wearing the requested order's O(.) label.
#
# `series` builds coefficients by differentiating without re-simplifying, so an
# expression whose derivatives do not close grows by a constant factor per
# coefficient: `sqrt(t**-2 + t**-1)` costs 0.15 s at order 13 and doubles per
# order, i.e. order 32 is not slow but unfinishable. Before the work ceiling
# this call never returned.
# ---------------------------------------------------------------------------


def _runaway_radical(p: alkahest.ExprPool) -> tuple[alkahest.Expr, alkahest.Expr]:
    t = p.symbol("t")
    return alkahest.sqrt(t ** (-2) + t ** (-1)), t


def test_series_refuses_a_runaway_expansion_rather_than_truncating():
    """The refusal *is* the assertion: this returning at all is the fix.

    No wall-clock bound is asserted — a regression hangs the test rather than
    failing a timing assertion, and timing bounds are flaky under the sanitizer
    jobs. What is asserted is the shape of the answer: a coded refusal, not a
    nine-term series labelled `O(t^32)`, which would be a false statement about
    the remainder that no caller could audit.
    """
    p = alkahest.ExprPool()
    expr, t = _runaway_radical(p)
    with pytest.raises(alkahest.SeriesError) as excinfo:
        alkahest.series(expr, t, p.integer(0), 32)
    assert excinfo.value.code == "E-SERIES-003"


def test_series_order_zero_keeps_its_own_code():
    """The refusal above is carried on the same variant as `order == 0`, so the
    user error must keep reporting `E-SERIES-002` and not be re-attributed."""
    p = alkahest.ExprPool()
    x = p.symbol("x")
    with pytest.raises(alkahest.SeriesError) as excinfo:
        alkahest.series(x, x, p.integer(0), 0)
    assert excinfo.value.code == "E-SERIES-002"


def test_series_honours_an_active_budget():
    """`series` joins `integrate` and `limit` in honouring `Budget`, and a
    budget trip is reported as one — `E-BUDGET-*`, not "this order is
    unreachable"."""
    p = alkahest.ExprPool()
    expr, t = _runaway_radical(p)
    with (
        alkahest.context(pool=p, budget=alkahest.Budget(max_steps=3)),
        pytest.raises(alkahest.BudgetExceededError) as excinfo,
    ):
        alkahest.series(expr, t, p.integer(0), 32)
    assert excinfo.value.code.startswith("E-BUDGET-")


def test_ordinary_high_order_series_still_expands():
    """The control: the ceiling costs no coverage. `sin` at order 24 interns a
    couple of hundred nodes against a ceiling of 50 000."""
    p = alkahest.ExprPool()
    x = p.symbol("x")
    s = alkahest.series(alkahest.sin(x), x, p.integer(0), 24)
    assert _has_big_o(s.expr)
    # sin's expansion runs to x^23: the last odd power below order 24.
    assert "x^23" in str(s.expr)
