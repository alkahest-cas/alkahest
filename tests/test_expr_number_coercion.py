"""Python numbers reaching the pool as the value that was written.

``Expr``'s arithmetic dunders coerce a Python operand to a pool node.  They used
to reach for ``extract::<f64>()`` early, and pyo3's ``f64`` extraction is not a
test for "is a float" — it goes through ``__float__``, so it also swallowed a
Python ``int`` wider than ``i64``, a :class:`fractions.Fraction`, a
:class:`decimal.Decimal` and every NumPy scalar.

So ``x ** (10**30 + 1)`` became ``x ** 1e30`` (the ``+ 1`` gone in silence, and
an exact integer power turned into a float one) and ``x ** Fraction(1, 3)``
became ``x ** 0.3333333333333333`` — not a cube root, and not anything
``integrate``/``puiseux_series``/the polynomial converters can recognise as one,
since they all read an exponent structurally.

The kernel never forced the loss: ``ExprPool.integer`` and ``ExprPool.rational``
are ``rug``-backed and unbounded.  These tests pin the coercion ladder, and pin
the neighbours that were always exact so the fix cannot be traded for a
regression in the other direction.

The probe is the *node*, not an evaluated number, because ``eval_expr`` reduces
every exponent to an ``f64`` before computing: ``8 ** (1/3)`` and
``8 ** 0.3333333333333333`` are both ``2.0``.  What the coercion destroyed was
the expression, and only the expression can show it.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import alkahest as ak
import pytest

numpy = pytest.importorskip("numpy", reason="NumPy scalar coercion needs NumPy")


@pytest.fixture
def pool():
    return ak.ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def _exponent(expr: ak.Expr) -> str:
    """The exponent node of a power, as the kernel renders it."""
    node = expr.node()
    assert node[0] == "pow", node
    return str(node[2])


# ---------------------------------------------------------------------------
# Integers of any width
# ---------------------------------------------------------------------------


def test_an_int_wider_than_i64_keeps_every_digit(x):
    """``10**30 + 1`` is not ``1e30``, and the difference is not a rounding one.

    The nearest double to ``10**30 + 1`` is ``1000000000000000019884624838656``
    — a different integer, of different parity, 19884624838655 away.
    """
    assert _exponent(x ** (10**30 + 1)) == "1000000000000000000000000000001"
    assert int(float(10**30 + 1)) == 1000000000000000019884624838656


@pytest.mark.parametrize("n", [0, 1, -1, 7, 2**62 + 1, -(2**62) - 1])
def test_an_int_inside_i64_is_still_exact(x, n):
    """The control: the widths that always worked still work."""
    assert _exponent(x**n) == str(n)


def test_every_arithmetic_dunder_keeps_a_wide_int(x):
    """``__pow__`` was the loudest case, but they all shared the helper."""
    big = 10**30 + 1
    text = str(big)
    assert text in str(x + big)
    assert text in str(big + x)
    assert text in str(x * big)
    assert text in str(big * x)
    assert text in str(x - big)
    assert text in str(big - x)
    assert text in str(x / big)
    assert text in str(big / x)
    # No float node anywhere: a float renders with an exponent or a decimal point.
    assert "e30" not in str(x * big)


def test_a_wide_int_substituted_into_an_expression_stays_exact(pool, x):
    """``subs`` shares the coercion path with the operators."""
    got = ak.subs(x + pool.integer(0), {x: 10**30 + 1})
    assert "1000000000000000000000000000001" in str(got)


# ---------------------------------------------------------------------------
# Exact rationals
# ---------------------------------------------------------------------------


def test_a_fraction_exponent_is_the_exact_rational(pool, x):
    """``x ** Fraction(1, 3)`` is the cube root, not a float power."""
    assert _exponent(x ** Fraction(1, 3)) == "1/3"
    assert x ** Fraction(1, 3) == x.pow_expr(pool.rational(1, 3))
    # The double is a genuinely different number.
    assert Fraction(1 / 3) != Fraction(1, 3)


def test_a_dyadic_fraction_is_still_exact(pool, x):
    """The control: ``3/2`` survived ``f64`` and must survive the new path."""
    assert _exponent(x ** Fraction(3, 2)) == "3/2"
    assert x ** Fraction(3, 2) == x.pow_expr(pool.rational(3, 2))


def test_a_whole_fraction_is_an_integer_node_not_a_rational_one(pool, x):
    """``Fraction(4, 2)`` must intern as ``2``.

    ``pool.rational(2, 1)`` is a node *distinct* from ``pool.integer(2)``, and
    every structural match on an integer exponent would miss it.
    """
    assert x ** Fraction(4, 2) == x**2
    assert _exponent(x ** Fraction(4, 2)) == "2"


def test_a_fraction_in_the_other_operators_too(x):
    assert "1/3" in str(x + Fraction(1, 3))
    assert "1/3" in str(x * Fraction(1, 3))
    assert "3/2" in str(x / Fraction(3, 2))


def test_a_decimal_is_its_exact_decimal_value(x):
    """``Decimal("0.1")`` is one tenth; the double named ``0.1`` is not."""
    assert _exponent(x ** Decimal("0.1")) == "1/10"
    assert Decimal("0.1").as_integer_ratio() == (1, 10)
    assert Fraction(0.1) != Fraction(1, 10)


def test_a_decimal_without_an_exact_ratio_falls_back_to_a_float(x):
    """``Decimal("NaN")`` has no exact ratio — refusing it would be worse."""
    assert "NaN" in str(x ** Decimal("NaN"))


# ---------------------------------------------------------------------------
# Floats stay floats
# ---------------------------------------------------------------------------


def test_a_python_float_is_still_a_float_node(x):
    """A float *is* an IEEE double; keeping it one loses nothing.

    Turning ``0.1`` into 3602879701896397/36028797018963968 would be exact and
    useless — nobody who writes ``0.1`` means that rational.
    """
    exponent = _exponent(x**0.1)
    assert "e-1" in exponent or "." in exponent
    assert "/" not in exponent


def test_numpy_scalars_follow_their_own_kind(x):
    """Integer scalars are exact; float scalars are floats, at any width."""
    assert _exponent(x ** numpy.int64(3)) == "3"
    assert _exponent(x ** numpy.int32(-5)) == "-5"
    for scalar in (numpy.float64(0.1), numpy.float32(0.1)):
        assert "/" not in _exponent(x**scalar)


def test_bool_is_an_int(x):
    assert _exponent(x**True) == "1"
    assert _exponent(x**False) == "0"


# ---------------------------------------------------------------------------
# The non-numeric arms of __pow__
# ---------------------------------------------------------------------------


def test_a_non_number_exponent_is_a_type_error_not_a_wrong_answer(x):
    with pytest.raises(TypeError):
        _ = x ** "two"


def test_an_expr_from_another_pool_is_refused(x):
    """``__pow__`` read the other pool's raw id and returned ``x^x``.

    Every other operator raised the pool-mismatch error; this one silently
    reinterpreted an ``ExprId`` in the wrong arena, which is how
    ``pool_a.symbol("x") ** pool_b.symbol("y")`` came back as ``x^x``.
    """
    other = ak.ExprPool()
    y = other.symbol("y")
    with pytest.raises(ak.PoolError):
        _ = x**y


def test_pow_with_a_modulus_is_refused_not_ignored(x):
    """``pow(x, 2, 5)`` used to return ``x^2`` and drop the modulus."""
    with pytest.raises(TypeError):
        pow(x, 2, 5)
    # The two-argument form is untouched.
    assert str(pow(x, 2)) == str(x**2)


# ---------------------------------------------------------------------------
# Matrix scalars share the helper
# ---------------------------------------------------------------------------


def test_matrix_scalar_multiplication_is_exact_too(pool, x):
    m = ak.Matrix.from_rows([[x, pool.integer(1)], [pool.integer(0), x]])
    scaled = m * Fraction(1, 3)
    assert "1/3" in str(scaled.get(0, 0))
    wide = m * (10**30 + 1)
    assert "1000000000000000000000000000001" in str(wide.get(0, 0))
