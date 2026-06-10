"""Non-linear integration by substitution (derivative-divides heuristic).

`∫ f(g(x))·g'(x) dx = ∫ f(u) du` with `u = g(x)`.  The engine tries this only
after the direct rules and the rational-function path decline, and emits a result
only when its internal `d/dx F = integrand` soundness gate passes — so the
returned antiderivative is always verifiable.

Each test integrates a composite integrand and checks that the symbolic
derivative of the result numerically matches the integrand at several sample
points.

Run after building the extension:
    maturin develop --release
    pytest tests/test_u_substitution.py -v
"""

from __future__ import annotations

import math

from alkahest import ExprPool, cos, diff, exp, integrate, log, sin, tan

_UNARY = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
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


def _check(integrand, x, sample_xs):
    """Integrate, then verify d/dx F = integrand numerically at the samples."""
    res = integrate(integrand, x)
    f = res.value
    d = diff(f, x).value
    for xv in sample_xs:
        lhs = _num_eval(d, xv)
        rhs = _num_eval(integrand, xv)
        assert abs(lhs - rhs) < 1e-7 * (1.0 + abs(rhs)), (
            f"d/dx F={lhs} != integrand={rhs} at x={xv} (F={f})"
        )
    return f


def test_usub_x_sin_x_squared():
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = x * sin(x**2)
    _check(integrand, x, [0.3, 0.7, 1.1, 1.9])


def test_usub_2x_exp_x_squared():
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = pool.integer(2) * x * exp(x**2)
    _check(integrand, x, [-0.6, 0.2, 0.9, 1.4])


def test_usub_ln_x_over_x():
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = log(x) / x
    _check(integrand, x, [0.5, 1.3, 2.2, 3.0])


def test_usub_tan_x():
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = tan(x)
    _check(integrand, x, [0.2, 0.6, 1.0, 1.3])


def test_usub_exp_x_cos_exp_x():
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = exp(x) * cos(exp(x))
    _check(integrand, x, [-0.5, 0.1, 0.4, 0.8])


def test_usub_x_cos_x_squared_plus_1():
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = x * cos(x**2 + pool.integer(1))
    _check(integrand, x, [0.3, 0.9, 1.5, 2.1])
