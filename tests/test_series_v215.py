"""V2-15 — user-facing series() with BigO remainder."""

import pytest

import alkahest


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
