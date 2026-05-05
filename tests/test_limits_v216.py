"""V2-16 — limits + SymPy cross-checks (curated, expected-to-pass corpus)."""

from __future__ import annotations

import pytest

import alkahest
from alkahest import parse

sympy = pytest.importorskip("sympy")

INF = "\u221e"


def _expr_to_sympy(e):
    n = e.node()
    tag = n[0]
    if tag == "symbol":
        return sympy.oo if n[1] == INF else sympy.Symbol(n[1])
    if tag == "integer":
        return sympy.Integer(int(n[1]))
    if tag == "rational":
        return sympy.Rational(int(n[1]), int(n[2]))
    if tag == "add":
        return sympy.Add(*[_expr_to_sympy(c) for c in n[1]])
    if tag == "mul":
        return sympy.Mul(*[_expr_to_sympy(c) for c in n[1]])
    if tag == "pow":
        return sympy.Pow(_expr_to_sympy(n[1]), _expr_to_sympy(n[2]))
    if tag == "func":
        name, args = n[1], n[2]
        av = [_expr_to_sympy(a) for a in args]
        return getattr(sympy, name)(*av)
    raise AssertionError(tag)


def _equiv(a, b) -> bool:
    if a == b or a.equals(b):
        return True
    diff = sympy.simplify(a - b)
    return diff == 0


def _alk_lim(s: str, *, pt: object = 0, direction: str | None = None):
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    expr = parse(s.replace("^", "**"), pool, {"x": x})
    if pt == "oo":
        pto = pool.pos_infinity()
    elif isinstance(pt, int):
        pto = pool.integer(pt)
    else:
        raise AssertionError(pt)
    d = "+-" if direction is None else direction
    return alkahest.limit(expr, x, pto, d)


def _sp_lim(s: str, *, pt, direction=None):
    x = sympy.Symbol("x")
    ex = sympy.sympify(s.replace("^", "**"))
    if pt == "oo":
        return sympy.limit(ex, x, sympy.oo)
    if direction == "+":
        return sympy.limit(ex, x, pt, dir="+")
    if direction == "-":
        return sympy.limit(ex, x, pt, dir="-")
    return sympy.limit(ex, x, pt)


def test_limit_basic_three():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    zero = p.integer(0)
    oo = p.pos_infinity()
    lx = parse("x * log(x)", p, {"x": x})
    assert alkahest.limit(alkahest.sin(x) / x, x, zero) == p.integer(1)
    assert alkahest.limit(lx, x, zero, "+") == zero
    assert alkahest.limit(alkahest.exp(x), x, oo) == oo


def test_limit_neg_infinity_exp():
    p = alkahest.ExprPool()
    x = p.symbol("x")
    assert alkahest.limit(alkahest.exp(x), x, -p.pos_infinity()) == p.integer(0)


@pytest.mark.parametrize(
    ("formula", "point", "direction"),
    [
        ("sin(x)/x", 0, None),
        ("tan(x)/x", 0, None),
        ("(exp(x)-1)/x", 0, None),
        ("sinh(x)/x", 0, None),
        ("1/x*sin(x)", 0, None),
        ("x*log(x)", 0, "+"),
        ("(x^2-9)/(x-3)", 3, None),
        ("x/(sin(x)+cos(x))", 0, None),
        ("1/x", 0, "+"),
        ("1/x", 0, "-"),
        ("1/x^2", 0, None),
        ("exp(x)", "oo", None),
        ("x^2", "oo", None),
        ("x/(x+1)", "oo", None),
        ("1/x", "oo", None),
        ("(2*x^2+3*x)/(x^2+5)", "oo", None),
    ],
)
def test_limit_sympy_oracle(formula, point, direction):
    if direction == "+":
        alk, sp_ref = _alk_lim(formula, pt=point, direction="+"), _sp_lim(
            formula, pt=point, direction="+"
        )
    elif direction == "-":
        alk, sp_ref = _alk_lim(formula, pt=point, direction="-"), _sp_lim(
            formula, pt=point, direction="-"
        )
    else:
        alk, sp_ref = _alk_lim(formula, pt=point), _sp_lim(formula, pt=point)
    assert _equiv(_expr_to_sympy(alk), sp_ref), f"{formula=} {point=} {direction=}"


def test_one_over_x_bidirectional_raises():
    with pytest.raises(alkahest.LimitError):
        _alk_lim("1/x", pt=0, direction="+-")
