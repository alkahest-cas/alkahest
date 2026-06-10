"""
Definite integration via the fundamental theorem of calculus.

``integrate(f, x, a, b)`` returns ``F(b) − F(a)`` where ``F = ∫ f dx`` is the
elementary antiderivative.  Only the "antiderivative exists and is finite at the
bounds" case is handled; non-elementary / unsupported integrands propagate the
underlying integration error rather than guessing a value.

Run after building the extension:
    maturin develop --release
    pytest tests/test_definite_integral.py -v
"""

import math

import alkahest
import pytest
from alkahest import ExprPool, eval_expr, integrate


def _value(result):
    """Numeric value of a (constant) definite-integral DerivedResult.

    The native ``eval_expr`` covers rationals/log; the small recursive fallback
    below additionally handles ``atan``/``sqrt`` constants that appear in
    rational-function antiderivatives.
    """
    expr = result.value
    try:
        return eval_expr(expr, {})
    except Exception:
        return _py_eval(expr)


def _py_eval(expr):
    """Minimal float evaluator for closed-form constants (no free symbols)."""
    s = str(expr)
    # Parse via Python after mapping function names; the printed form uses the
    # standard infix syntax with named unary functions.
    import math

    env = {
        "atan": math.atan,
        "log": math.log,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "exp": math.exp,
        "__builtins__": {},
    }
    return float(eval(s, env))


def test_x_squared_0_1():
    # ∫_0^1 x² dx = 1/3.
    pool = ExprPool()
    x = pool.symbol("x")
    r = integrate(x**2, x, 0, 1)
    assert abs(_value(r) - 1.0 / 3.0) < 1e-12


def test_two_x_0_1():
    # ∫_0^1 2x dx = 1.
    pool = ExprPool()
    x = pool.symbol("x")
    r = integrate(2 * x, x, 0, 1)
    assert abs(_value(r) - 1.0) < 1e-12


def test_one_over_x_1_2():
    # ∫_1^2 1/x dx = log(2).
    pool = ExprPool()
    x = pool.symbol("x")
    r = integrate(1 / x, x, 1, 2)
    assert abs(_value(r) - math.log(2.0)) < 1e-12


def test_arctan_0_1():
    # ∫_0^1 1/(x²+1) dx = atan(1) − atan(0) = π/4.
    pool = ExprPool()
    x = pool.symbol("x")
    r = integrate(1 / (x**2 + 1), x, 0, 1)
    assert abs(_value(r) - math.pi / 4) < 1e-12


def test_polynomial_general_bounds():
    # ∫_1^3 (x² + 2x) dx = [x³/3 + x²]_1^3 = (9 + 9) − (1/3 + 1) = 18 − 4/3.
    pool = ExprPool()
    x = pool.symbol("x")
    r = integrate(x**2 + 2 * x, x, 1, 3)
    assert abs(_value(r) - (18.0 - 4.0 / 3.0)) < 1e-12


def test_definite_matches_quadrature():
    # Cross-check against a midpoint Riemann sum.
    pool = ExprPool()
    x = pool.symbol("x")
    f = 1 / (x**2 + 1)
    r = integrate(f, x, 0, 2)
    a, b, n = 0.0, 2.0, 200_000
    h = (b - a) / n
    approx = sum(eval_expr(f, {x: a + (i + 0.5) * h}) for i in range(n)) * h
    assert abs(_value(r) - approx) < 1e-4


def test_indefinite_still_works_two_args():
    # The 2-arg form is unchanged: returns the antiderivative.
    pool = ExprPool()
    x = pool.symbol("x")
    F = integrate(x**2, x).value
    # d/dx F = x².
    dF = alkahest.diff(F, x).value
    for pt in (0.5, 1.7, 3.2):
        assert abs(eval_expr(dF, {x: pt}) - pt**2) < 1e-9


def test_nonelementary_propagates():
    # ∫_0^1 exp(x²) dx is non-elementary — must error, not return a number.
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(Exception):
        integrate(alkahest.exp(x**2), x, 0, 1)


def test_unsupported_propagates():
    # ∫_1^2 sin(x)/x dx is non-elementary in the definite form too.
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(Exception):
        integrate(alkahest.sin(x) / x, x, 1, 2)


def test_one_bound_raises():
    # Exactly one bound is a usage error.
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(ValueError):
        integrate(x**2, x, 0)
