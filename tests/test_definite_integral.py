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


def test_nonelementary_closed_form_evaluates():
    # ∫_1^2 sin(x)/x dx has no *elementary* antiderivative, but it does have a
    # closed form over the registered basis — `Si(2) − Si(1)` — so the definite
    # form is now a number rather than a refusal.  Checked against the value,
    # not against the shape.
    pool = ExprPool()
    x = pool.symbol("x")
    value = integrate(alkahest.sin(x) / x, x, 1, 2).value
    got = float(alkahest.eval_expr(value, {}))
    assert abs(got - 0.6593299064355118) < 1e-9, got


def test_one_bound_raises():
    # Exactly one bound is a usage error.
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(ValueError):
        integrate(x**2, x, 0)


# ---------------------------------------------------------------------------
# The antiderivative's domain over the interval of integration
#
# The FTC needs F continuous on [a, b].  A candidate that is *undefined* across
# a sub-interval where the integrand is an ordinary finite real is not that
# function, and F(b) − F(a) is not the integral — whatever the two endpoint
# values happen to come out as.  Each of these used to be answered `Solved`
# with a value the evaluator itself rejects (a `log` of a negative number, or
# an `asin` outside [−1, 1]); the jump scan could not see it, because forming
# its ratio needs F at both ends of a cell and a hole makes every cell
# undecidable.
# ---------------------------------------------------------------------------


def _refusal_code(exc):
    return getattr(exc, "code", "") or str(exc)


@pytest.mark.parametrize(
    ("build", "lo", "hi", "what"),
    [
        # F = x·atanh x + ½·log(x²−1): the log is of a negative number for
        # every |x| < 1, i.e. exactly where atanh is defined.  True value 0.
        (lambda x: alkahest.atanh(x), -0.5, 0.5, "atanh over a symmetric interval"),
        # F = −¼log(x+1) + ¼log(x−1) − ½atan(x): both logs are negative below
        # −1, where 1/(x⁴−1) is finite.  True value 0.03042.
        (lambda x: 1 / (x**4 - 1), -3, -2, "1/(x^4-1) below its poles"),
        # F = ⅓log(x+1) − ⅙log(x²−x+1) + …: the first log is negative below
        # −1, where 1/(1+x³) is finite.  True value −0.10034.
        (lambda x: 1 / (1 + x**3), -4, -2, "1/(1+x^3) below its pole"),
    ],
)
def test_definite_refuses_an_antiderivative_undefined_on_the_interval(build, lo, hi, what):
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(Exception) as exc:
        integrate(build(x), x, lo, hi)
    assert "E-INT-001" in _refusal_code(exc.value), f"{what}: {exc.value}"


def test_definite_hole_refusal_names_the_reason():
    # The diagnostic has to say which of the guards fired, or a caller cannot
    # tell "your integrand has a pole" from "our answer has the wrong branch".
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(Exception) as exc:
        integrate(alkahest.atanh(x), x, -0.5, 0.5)
    assert "undefined across" in str(exc.value), str(exc.value)


@pytest.mark.parametrize(
    ("build", "lo", "hi", "expected"),
    [
        # A genuine improper integral: the *integrand* blows up at an endpoint
        # and the antiderivative does not.  Must keep working.
        (lambda x: x ** (-0.5), 0, 1, 2.0),
        (lambda x: 1 / alkahest.sqrt(x), 0, 1, 2.0),
        # F = ½log(x−1) − ½log(x+1) is perfectly defined on [2, 3]: the same
        # antiderivative shape as the refused case above, on an interval where
        # it holds.  The rule must be about the interval, not the formula.
        (lambda x: 1 / (x**2 - 1), 2, 3, 0.5 * (math.log(0.5) - math.log(1 / 3))),
        # A bounded integrand whose antiderivative is defined throughout.
        (lambda x: 1 / (2 + alkahest.cos(x)), 0, 3, 1.6726765368776789),
    ],
)
def test_definite_domain_rule_leaves_correct_answers_alone(build, lo, hi, expected):
    pool = ExprPool()
    x = pool.symbol("x")
    r = integrate(build(x), x, lo, hi)
    assert abs(_value(r) - expected) < 1e-9
