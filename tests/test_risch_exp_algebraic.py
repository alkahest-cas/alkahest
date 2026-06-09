"""Non-diagonal-f Risch (PR2): ∫ R(x, α)·exp(β) dx with β an *algebraic*
function of x (a radical α = p(x)^{1/n}).

Seeking an antiderivative `v·exp(β)`, the integral reduces to the in-field Risch
DE `D(v) + f·v = R` with `f = D(β)` a *non-base* element of `ℚ(x)(α)`, solved by
the generalized coupled solver and accepted only after a numeric `d/dx F = f`
gate.  Each elementary case asserts that `diff(integrate(...))` numerically
matches the integrand at points where the radical and exp are real; the
non-elementary case asserts a clean decline (never a wrong elementary form).
"""

from __future__ import annotations

import math

import pytest
from alkahest import ExprPool, diff, exp, integrate, sqrt

_UNARY = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "cbrt": lambda v: math.copysign(abs(v) ** (1.0 / 3.0), v),
}


def _num_eval(expr, xv):
    """Recursively evaluate *expr* at ``x = xv`` via the node() reflection API."""
    n = expr.node()
    tag = n[0]
    if tag == "symbol":
        return xv
    if tag == "integer":
        return float(int(n[1]))
    if tag == "rational":
        return float(int(n[1])) / float(int(n[2]))
    if tag == "add":
        return sum(_num_eval(a, xv) for a in n[1])
    if tag == "mul":
        prod = 1.0
        for a in n[1]:
            prod *= _num_eval(a, xv)
        return prod
    if tag == "pow":
        return _num_eval(n[1], xv) ** _num_eval(n[2], xv)
    if tag == "func":
        name, args = n[1], n[2]
        if name in _UNARY and len(args) == 1:
            return _UNARY[name](_num_eval(args[0], xv))
    raise AssertionError(f"cannot evaluate node {n}")


def _check_diff(integrand, x, sample_xs):
    """Integrate, then verify d/dx F = integrand numerically."""
    res = integrate(integrand, x)
    f = res.value
    d = diff(f, x).value
    checked = 0
    for xv in sample_xs:
        lhs = _num_eval(d, xv)
        rhs = _num_eval(integrand, xv)
        assert abs(lhs - rhs) < 1e-6 * (1.0 + abs(rhs)), (
            f"x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {f}"
        )
        checked += 1
    assert checked >= 3
    return f


def test_exp_sqrt_x_with_coeff():
    """Headline: ∫ exp(√x)·(1/(2√x) + 1/2) dx = √x·exp(√x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    sx = sqrt(x)
    coeff = pool.rational(1, 2) * sx**-1 + pool.rational(1, 2)
    integrand = exp(sx) * coeff
    _check_diff(integrand, x, [0.6, 1.4, 2.7, 3.5])


def test_exp_sqrt_x_bare():
    """Elementary: ∫ exp(√x) dx = 2(√x − 1)·exp(√x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = exp(sqrt(x))
    _check_diff(integrand, x, [0.6, 1.4, 2.7, 3.5])


def test_exp_sqrt_x_plus_1():
    """√(x+1) variant: ∫ exp(√(x+1))/(2√(x+1)) dx = exp(√(x+1))."""
    pool = ExprPool()
    x = pool.symbol("x")
    rad = sqrt(x + pool.integer(1))
    integrand = exp(rad) * (pool.rational(1, 2) * rad**-1)
    _check_diff(integrand, x, [0.6, 1.4, 2.7, 3.5])


def test_exp_sqrt_x_over_x_nonelementary():
    """∫ exp(√x)/x dx is non-elementary (Ei-type) — must not return a wrong
    elementary form."""
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = exp(sqrt(x)) * x**-1
    with pytest.raises(Exception) as exc_info:
        integrate(integrand, x)
    msg = str(exc_info.value)
    assert "E-INT-004" in msg or "elementary" in msg.lower() or "NonElementary" in msg
