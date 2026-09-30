"""Canonical forms, non-finite values and function arity (core audit A5, A8, C1).

* **A5** — ``oo`` is a positive symbol and IEEE floats are atoms, so the
  simplifier's field identities used to fire on them: ``oo - oo``, ``NaN - NaN``
  and ``inf - inf`` simplified to ``0``, ``oo * 0`` to ``0``, ``oo / oo`` to
  ``1``.  Those are indeterminate forms; they now stay unevaluated.
* **A8** — one value, one node: ``rational(4, 2)`` is ``integer(2)``, a one-term
  ``Add``/``Mul`` is its term, the empty sum/product is ``0``/``1``, ``-0.0`` is
  ``0.0``; and a float prints as something that reads back as a float, at a
  precision that keeps its digits.
* **C1** — a built-in function at the wrong arity (``sqrt()``,
  ``EllipticPi(x)``) is refused at construction instead of panicking later in
  every printer, differentiator and integrator.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from alkahest.exceptions import ParseError


@pytest.fixture
def pool():
    return ak.ExprPool()


def _simp(e):
    return ak.simplify(e).value


def _is_exact_number(e) -> bool:
    return e.node()[0] in ("integer", "rational")


# ---------------------------------------------------------------------------
# A5 — indeterminate forms
# ---------------------------------------------------------------------------


def _indeterminate_cases(p):
    oo = p.pos_infinity()
    x = p.symbol("x")
    nan = p.float(float("nan"))
    inf = p.float(float("inf"))
    zero = p.integer(0)
    return {
        "oo - oo": oo - oo,
        "oo * 0": oo * zero,
        "0 * oo": zero * oo,
        "oo / oo": oo / oo,
        "x*oo - x*oo": x * oo - x * oo,
        "nan - nan": nan - nan,
        "nan * 0": nan * zero,
        "inf * 0": inf * zero,
        "inf - inf": inf - inf,
        "x*inf - x*inf": x * inf - x * inf,
        "nan / nan": nan / nan,
    }


@pytest.mark.parametrize("label", list(_indeterminate_cases(ak.ExprPool())))
def test_indeterminate_forms_do_not_fold_to_a_number(pool, label):
    e = _indeterminate_cases(pool)[label]
    r = _simp(e)
    assert not _is_exact_number(r), f"simplify({label}) = {r}"
    r = ak.simplify_egraph(e).value
    assert not _is_exact_number(r), f"simplify_egraph({label}) = {r}"


def test_two_oo_minus_oo_is_not_oo(pool):
    oo = pool.pos_infinity()
    assert _simp(2 * oo - oo) != oo


def test_finite_identities_still_apply(pool):
    x = pool.symbol("x")
    oo = pool.pos_infinity()
    assert _simp(x - x) == pool.integer(0)
    assert _simp(x * 0) == pool.integer(0)
    assert _simp(x / x) == pool.integer(1)
    # A finite pair beside an infinite term still cancels.
    assert _simp(x - x + oo) == oo


def test_nan_float_is_not_equal_to_itself_for_cancellation(pool):
    nan = pool.float(float("nan"))
    r = _simp(nan - nan)
    assert r != pool.integer(0)
    # Numerically it is undefined, as NaN − NaN is.
    with pytest.raises(ak.DomainError):
        ak.eval_expr(r, {})


# ---------------------------------------------------------------------------
# A8 — canonical forms
# ---------------------------------------------------------------------------


def test_rational_with_unit_denominator_is_the_integer(pool):
    assert pool.rational(4, 2) == pool.integer(2)
    assert pool.rational(4, 2).node()[0] == "integer"
    assert pool.rational(-9, 3) == pool.integer(-3)
    x = pool.symbol("x")
    # The substitution the audit found missed (A7) now matches: one node.
    assert ak.subs(x + pool.rational(2, 1), {pool.integer(2): pool.symbol("y")}) == (
        x + pool.symbol("y")
    )


def test_singleton_and_empty_sums_and_products(pool):
    x = pool.symbol("x")
    assert pool.add([x]) == x
    assert pool.mul([x]) == x
    assert pool.add([]) == pool.integer(0)
    assert pool.mul([]) == pool.integer(1)
    assert _simp(pool.mul([x]) - x) == pool.integer(0)
    for e, want in [(pool.add([]), "0"), (pool.mul([]), "1")]:
        assert str(e) == want
        assert ak.unicode_str(e) == want
        assert ak.latex(e) == want


def test_signed_zero_floats_are_one_node(pool):
    assert pool.float(-0.0) == pool.float(0.0)
    assert str(pool.float(-0.0)) == "0.0"


@pytest.mark.parametrize("value", [0.0, 1.0, 2.0**60, 0.1, 1e-300])
def test_float_prints_as_a_float(pool, value):
    e = pool.float(value)
    s = str(e)
    back = ak.parse(s, pool)
    assert back.node()[0] == "float", f"{value} printed as {s!r}, re-parsed as {back.node()}"
    assert back == e


def _significant_digits(s: str) -> int:
    mantissa = s.lower().split("e")[0]
    return len(mantissa.replace("-", "").replace(".", "").lstrip("0"))


@pytest.mark.parametrize("prec", [64, 113, 200, 1000])
def test_high_precision_float_keeps_its_digits(pool, prec):
    e = pool.float(1.0 / 3.0, prec)
    s = str(e)
    assert _significant_digits(s) > 17
    back = ak.parse(s, pool)
    assert back.node()[0] == "float"
    # Read at 53 bits, the reprint had 17 digits; now none are dropped.
    assert _significant_digits(str(back)) == _significant_digits(s)
    assert abs(ak.eval_expr(back, {}) - 1.0 / 3.0) < 1e-15
    # Once read back, printing and reading again is a fixpoint.
    assert ak.parse(str(back), pool) == back


# ---------------------------------------------------------------------------
# C1 — wrong-arity built-ins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "nargs"),
    [
        ("sqrt", 0),
        ("sin", 0),
        ("sin", 2),
        ("EllipticPi", 1),
        ("EllipticPi", 4),
        ("EllipticE", 3),
        ("atan2", 1),
        ("gamma", 0),
        ("lambert_w", 0),
        ("max", 1),
    ],
)
def test_pool_func_refuses_a_wrong_arity_builtin(pool, name, nargs):
    x = pool.symbol("x")
    with pytest.raises(ak.PoolError) as info:
        pool.func(name, [x] * nargs)
    assert info.value.code == "E-POOL-002"
    assert name in str(info.value)


def test_pool_func_accepts_right_arities_and_user_functions(pool):
    x = pool.symbol("x")
    pool.func("sin", [x])
    pool.func("EllipticE", [x])
    pool.func("EllipticE", [x, x])
    pool.func("EllipticPi", [x, x, x])
    pool.func("max", [x, x, x])
    for n in range(4):
        pool.func("f", [x] * n)


@pytest.mark.parametrize(
    "src",
    [
        "sin()",
        "sin(x, x)",
        "sqrt()",
        "gamma(x, y, z)",
        "lambert_w(x, 1, 2)",
        "EllipticPi(x)",
        "EllipticPi(1, 2, 3, 4)",
        "atan2(x)",
        "Si()",
        "dilog(x, y)",
    ],
)
def test_parse_refuses_a_wrong_arity_builtin(pool, src):
    with pytest.raises(ParseError) as info:
        ak.parse(src, pool)
    assert info.value.code.startswith("E-PARSE-")
    assert "takes" in str(info.value)
