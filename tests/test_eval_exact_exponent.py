"""The exponent an expression holds exactly must survive numeric evaluation.

``ExprPool.integer`` is ``rug``-backed and unbounded, so ``x ** (10**30 + 1)``
interns the exponent exactly (that is what #376 fixed).  Every numeric
evaluator then reduced it to an ``f64`` before computing, and ``1e30`` is
**even**, so ``eval_expr(x ** (10**30 + 1), {x: -1})`` returned ``1.0`` where
the answer is ``-1``: the right magnitude with the wrong sign, and nothing
downstream could tell.

The whole argument is the parity of the exponent — ``10**30 + 1`` ends in 1, so
it is odd, and a product of an odd number of factors of ``-1`` is ``-1``.  No
library is needed to check it, and Python's own integers agree.

Every refusal here is paired with the nearest neighbour that must still be
answered, and every wide case with the small one a fast path must not break.
"""

import math
from fractions import Fraction

import alkahest as ak
import pytest
from alkahest.experimental import evaluate

#: Odd, and its nearest ``f64`` (``1e30``) is even.
ODD = 10**30 + 1
#: The even neighbour, which was never wrong.
EVEN = 10**30


@pytest.fixture
def x():
    pool = ak.ExprPool()
    return pool.symbol("x")


# ---------------------------------------------------------------------------
# eval_expr — the interpreter behind `ak.eval_expr`
# ---------------------------------------------------------------------------


def test_python_agrees_that_the_exponent_is_odd():
    """The oracle, such as it is."""
    assert ODD % 2 == 1
    assert (-1) ** ODD == -1
    # And the mechanism: the exponent's float image is a different, even number.
    assert float(ODD) == 1e30
    assert (-1.0) ** 1e30 == 1.0


def test_eval_expr_keeps_the_parity_of_a_wide_exponent(x):
    assert ak.eval_expr(x**ODD, {x: -1}) == -1.0


@pytest.mark.parametrize(
    ("exponent", "expected"),
    [
        (3, -1.0),  # the ordinary case
        (4, 1.0),
        (EVEN, 1.0),  # the even neighbour of the trap
        (2**53, 1.0),  # exactly representable, even
        (2**53 + 1, -1.0),  # the first odd integer f64 cannot hold
        (2**60, 1.0),  # past 2**53 and still exactly representable
        (2**60 + 1, -1.0),
        (-(10**30) - 1, -1.0),  # negative and odd: 1/(-1)**odd is still -1
    ],
)
def test_eval_expr_signs_every_integer_exponent(x, exponent, expected):
    assert ak.eval_expr(x**exponent, {x: -1}) == expected


def test_eval_expr_wide_exponent_on_a_positive_base_is_unchanged(x):
    assert ak.eval_expr(x**ODD, {x: 1}) == 1.0
    assert ak.eval_expr(x**ODD, {x: 0}) == 0.0


def test_a_magnitude_f64_cannot_hold_is_a_refusal_not_a_signed_guess(x):
    """`(-1)` to that power is fine; `2` is not, and says so."""
    for base in (2, -2, 2.0, -2.0):
        with pytest.raises(ak.DomainError) as excinfo:
            ak.eval_expr(x**ODD, {x: base})
        assert excinfo.value.code == "E-EVAL-009"

    # Control: the same base at an exponent whose power exists.
    assert ak.eval_expr(x**3, {x: -2}) == -8.0


def test_an_underflowing_wide_power_keeps_its_sign(x):
    """`(-0.5) ** odd` is a tiny *negative* number: -0.0, not +0.0."""
    result = ak.eval_expr(x**ODD, {x: -0.5})
    assert result == 0.0
    assert math.copysign(1.0, result) == -1.0


def test_a_float_exponent_is_left_alone(x):
    """Nothing was lost rounding `1e30`, because it was never exact."""
    assert ak.eval_expr(x**1e30, {x: -1}) == (-1.0) ** 1e30


# ---------------------------------------------------------------------------
# evaluate(mode=...) — the f64, complex, exact and interval backends
# ---------------------------------------------------------------------------


def test_f64_mode_keeps_the_parity(x):
    result = evaluate(x**ODD, {x: -1}, mode="f64")
    assert (result.status, result.value) == ("ok", -1.0)
    assert evaluate(x**EVEN, {x: -1}, mode="f64").value == 1.0


def test_complex_mode_keeps_the_parity(x):
    """The complex arm read `n.to_i64().unwrap_or(0)`: a wide exponent became 0."""
    result = evaluate(x**ODD, {x: complex(-1, 0)}, mode="complex")
    assert result.status == "ok"
    assert result.value == complex(-1, 0)

    # Control: the even neighbour, and a small exponent.
    assert evaluate(x**EVEN, {x: complex(-1, 0)}, mode="complex").value == complex(1, 0)
    assert evaluate(x**3, {x: complex(-1, 0)}, mode="complex").value == complex(-1, 0)


def test_complex_mode_refuses_a_wide_exponent_off_the_real_axis():
    """`z**n = |z|**n * exp(i*n*theta)`, and `n*theta mod 2*pi` has no bits left."""
    pool = ak.ExprPool()
    i = pool.imaginary_unit()

    result = evaluate(i**ODD, {}, mode="complex")
    assert (result.status, result.reason) == ("unsupported", "E-EVAL-012")

    # Control: i**3 = -i is still answered.
    assert evaluate(i**3, {}, mode="complex").value == complex(0, -1)


def test_exact_mode_answers_a_wide_exponent_where_it_costs_nothing(x):
    """Was `E-EVAL-003`, "only integer exponents" — for an integer exponent."""
    assert evaluate(x**ODD, {x: -1}, mode="exact").value == Fraction(-1)
    assert evaluate(x**EVEN, {x: -1}, mode="exact").value == Fraction(1)
    assert evaluate(x**ODD, {x: 1}, mode="exact").value == Fraction(1)
    assert evaluate(x**ODD, {x: 0}, mode="exact").value == Fraction(0)


def test_exact_mode_refuses_a_power_too_large_to_build(x):
    """This used to abort the process: GMP calls `abort()` on a failed malloc."""
    result = evaluate(x ** (10**12), {x: 2}, mode="exact")
    assert (result.status, result.reason) == ("unsupported", "E-EVAL-012")

    result = evaluate(x**ODD, {x: Fraction(1, 2)}, mode="exact")
    assert (result.status, result.reason) == ("unsupported", "E-EVAL-012")


def test_exact_mode_still_builds_an_affordable_power(x):
    assert evaluate(x**1000, {x: 2}, mode="exact").value == Fraction(2**1000)
    assert evaluate(x**3, {x: Fraction(1, 3)}, mode="exact").value == Fraction(1, 27)
    assert evaluate(x**-3, {x: Fraction(2, 3)}, mode="exact").value == Fraction(27, 8)


def test_exact_mode_still_refuses_a_genuinely_non_integer_exponent(x):
    result = evaluate(x ** Fraction(1, 2), {x: 4}, mode="exact")
    assert (result.status, result.reason) == ("unsupported", "E-EVAL-003")


def test_exact_mode_names_the_binding_it_cannot_take(x):
    """A rejected binding is `E-EVAL-002`, not `E-EVAL-001` ("unbound symbol").

    The symbol *was* bound.  Exact mode used to run against the partial map it
    had built before it gave up, and report the binding it never inserted as
    missing.
    """
    result = evaluate(x**3, {x: 0.1}, mode="exact")
    assert (result.status, result.reason) == ("unsupported", "E-EVAL-002")

    # Control: an exact binding of the same shape is answered.
    assert evaluate(x**3, {x: Fraction(1, 10)}, mode="exact").value == Fraction(1, 1000)


def test_interval_mode_stays_sound(x):
    """Ball evaluation never widened wrongly — it declines instead."""
    result = evaluate(x**ODD, {x: ak.ArbBall(-1, 0)}, mode="interval")
    assert (result.status, result.reason) == ("unsupported", "E-EVAL-010")

    # Control: an exponent it does handle encloses the right value.
    enclosure = evaluate(x**3, {x: ak.ArbBall(-1, 0)}, mode="interval").value
    assert enclosure.contains(-1.0)


# ---------------------------------------------------------------------------
# compile() — the backends lower the exponent to an f64 constant
# ---------------------------------------------------------------------------


def test_a_compiled_wide_exponent_agrees_with_the_interpreter(x):
    """Whatever tier is compiled in, the answer is the interpreter's."""
    compile_ = getattr(ak, "compile", None)
    if compile_ is None:
        pytest.skip("built without a JIT backend")
    f = compile_(x**ODD + x * x, [x])
    assert f(-1.0) == 0.0
    assert f.tier == "interpreter"
