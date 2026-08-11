"""V2-9 / V2-9b CAD / real QE bindings (pytest).

V2-9 covers one quantifier, one variable. V2-9b (this file's `*_two_var*` /
`*_exists_exists*` / `*_forall_forall*` / `*_mixed_alternation*` tests) extends
`decide` to two real variables with a quantifier prefix of length <= 2:
same-flavor blocks (``exists x exists y``, ``forall x forall y``) and mixed
alternation (``exists x forall y``, ``forall x exists y``), all decided via
CAD projection + rational-cell sampling. See `alkahest_core::real::cad` module
docs for the soundness argument and its `Unsupported` boundary.
"""

import pytest


def test_decide_forall_x_squared_plus_one_positive():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    z = pool.integer(0)
    one = pool.integer(1)
    body = pool.gt(x**2 + one, z)
    phi = alkahest.Forall(x, body)
    truth, wit = alkahest.decide(phi)
    assert truth is True
    assert wit is None


def test_decide_exists_x_squared_equals_two():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    two = pool.integer(2)
    body = pool.pred_eq(x**2, two)
    phi = alkahest.Exists(x, body)
    truth, wit = alkahest.decide(phi)
    assert truth is True
    # ±√2 is irrational, so there is no rational witness. This used to assert a
    # witness dict and was satisfied by the isolating interval's midpoint — a
    # "solution" of x² = 2 that does not solve it. A witness is a certificate;
    # reporting a wrong one is worse than reporting none.
    assert wit is None


def test_decide_exists_witness_actually_satisfies_the_equation():
    """A reported witness must satisfy the sentence it witnesses.

    ``∃x. 3x − 2 = 0`` has the rational solution ``2/3``.  Root isolation used
    to leave the bracket at ``[0, 1]`` and the witness came back as its midpoint
    ``1/2``, which fails the very equation it was offered as a solution to.
    """
    from fractions import Fraction

    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    body = pool.pred_eq(pool.integer(3) * x - pool.integer(2), pool.integer(0))
    truth, wit = alkahest.decide(alkahest.Exists(x, body))
    assert truth is True
    assert wit is not None
    assert Fraction(wit["x"]) == Fraction(2, 3)


def test_decide_forall_strict_square_false_at_non_dyadic_root():
    """``∀x. (3x + 2)² > 0`` is FALSE — the square vanishes at x = −2/3.

    The CAD sample set is built from isolating-bracket endpoints and midpoints,
    all dyadic, so ``−2/3`` was never tested and the sentence came back ``True``:
    a machine-checked-looking proof of a false theorem.
    """
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    inner = pool.integer(3) * x + pool.integer(2)
    body = pool.gt(inner**2, pool.integer(0))
    truth, _ = alkahest.decide(alkahest.Forall(x, body))
    assert truth is False


def test_decide_forall_strict_square_refuses_at_irrational_root():
    """``∀x. (x² − 2)² > 0`` is FALSE at ±√2, and no rational sample shows it.

    The honest answer is a refusal, not ``True``.
    """
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    inner = x**2 - pool.integer(2)
    body = pool.gt(inner**2, pool.integer(0))
    with pytest.raises(alkahest.CadError):
        alkahest.decide(alkahest.Forall(x, body))


def test_cad_lift_quadratic_roots():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    p = x**2 + pool.integer(-2)
    intervals = alkahest.cad_lift([p], x)
    assert len(intervals) == 2


# ---------------------------------------------------------------------------
# Two-variable, same-flavor quantifier blocks (exists exists / forall forall)
# ---------------------------------------------------------------------------


def test_decide_exists_exists_circle_through_origin_true():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    body = pool.pred_eq(x**2 + y**2, pool.integer(0))
    phi = alkahest.Exists(x, alkahest.Exists(y, body))
    truth, wit = alkahest.decide(phi)
    assert truth is True
    assert wit == {"x": "0", "y": "0"}


def test_decide_exists_exists_circle_plus_one_false():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    body = pool.pred_eq(x**2 + y**2 + pool.integer(1), pool.integer(0))
    phi = alkahest.Exists(x, alkahest.Exists(y, body))
    truth, wit = alkahest.decide(phi)
    assert truth is False
    assert wit is None


def test_decide_forall_forall_sum_of_squares_nonneg_true():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    body = pool.ge(x**2 + y**2, pool.integer(0))
    phi = alkahest.Forall(x, alkahest.Forall(y, body))
    truth, wit = alkahest.decide(phi)
    assert truth is True
    assert wit is None


def test_decide_forall_forall_product_positive_false():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    body = pool.gt(x * y, pool.integer(0))
    phi = alkahest.Forall(x, alkahest.Forall(y, body))
    truth, _wit = alkahest.decide(phi)
    assert truth is False


# ---------------------------------------------------------------------------
# Mixed alternation (exists forall / forall exists) — sound via the same
# projection since it only depends on the atoms' polynomials.
# ---------------------------------------------------------------------------


def test_decide_forall_exists_every_x_has_bigger_y_true():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    body = pool.gt(y, x)
    phi = alkahest.Forall(x, alkahest.Exists(y, body))
    truth, _wit = alkahest.decide(phi)
    assert truth is True


def test_decide_exists_forall_no_x_is_upper_bound_false():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    body = pool.ge(x, y)
    phi = alkahest.Exists(x, alkahest.Forall(y, body))
    truth, _wit = alkahest.decide(phi)
    assert truth is False


# ---------------------------------------------------------------------------
# Unsupported shapes: quantifier prefixes longer than two variables raise a
# clean CadError (E-CAD-001) rather than guessing.
# ---------------------------------------------------------------------------


def test_decide_three_variable_prefix_raises_cad_error():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    z = pool.symbol("z")
    body = pool.pred_eq(x + y + z, pool.integer(0))
    phi = alkahest.Exists(x, alkahest.Exists(y, alkahest.Exists(z, body)))
    with pytest.raises(alkahest.CadError) as excinfo:
        alkahest.decide(phi)
    assert "E-CAD-001" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Univariate regressions (V2-9) — must still work unchanged.
# ---------------------------------------------------------------------------


def test_decide_univariate_regression_exists_quadratic_root():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    body = pool.pred_eq(x**2, pool.integer(2))
    phi = alkahest.Exists(x, body)
    truth, wit = alkahest.decide(phi)
    assert truth is True
    # No *rational* witness exists for ±√2 — see
    # test_decide_exists_x_squared_equals_two for why reporting the isolating
    # interval's midpoint instead is a wrong certificate rather than a weak one.
    assert wit is None


def test_decide_univariate_regression_forall_square_nonneg():
    import alkahest
    from alkahest import ExprPool

    pool = ExprPool()
    x = pool.symbol("x")
    body = pool.ge(x**2, pool.integer(0))
    phi = alkahest.Forall(x, body)
    truth, wit = alkahest.decide(phi)
    assert truth is True
    assert wit is None
