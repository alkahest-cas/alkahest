"""Big Python ints cross the PyO3 boundary by bytes, not by decimal text.

CPython (3.10.7+/3.11+) caps int <-> str conversion at
``sys.get_int_max_str_digits()`` digits (4300 by default), and that conversion
is quadratic.  The binding used to round-trip every integer that did not fit an
``i64`` through ``str(n)`` one way and ``int(text)`` the other, so
``ExprPool().integer(10**5000 + 7)`` raised ``ValueError: Exceeds the limit
(4300) for integer string conversion`` — as did every big result coming back.

Every test here runs twice: under the default limit, and under the tightest
limit Python allows (640 digits), which is what a hardened deployment sets.
"""

from __future__ import annotations

import sys
from fractions import Fraction

import alkahest as ak
import pytest
from alkahest import modular, number_theory

_HAS_LIMIT = hasattr(sys, "set_int_max_str_digits")

BIG = 10**5000 + 7
VALUES = [
    0,
    1,
    -1,
    2**63 - 1,
    -(2**63),
    2**63,
    -(2**63) - 1,
    2**64 - 1,
    2**64,
    -(2**64),
    BIG,
    -BIG,
    -(10**20000),
    2**100_000 + 12345,
]


@pytest.fixture(params=["default", "tight"], autouse=True)
def _str_digit_limit(request):
    """Run each test under the default limit and under the minimum one."""
    if not _HAS_LIMIT:
        if request.param == "tight":
            pytest.skip("this Python has no int/str digit limit")
        yield
        return
    old = sys.get_int_max_str_digits()
    if request.param == "tight":
        sys.set_int_max_str_digits(640)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(old)


def _node_int(expr) -> int:
    node = expr._node_exact()
    assert node[0] == "integer", node
    return node[1]


# ---------------------------------------------------------------------------
# Python -> Rust -> Python round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", VALUES, ids=lambda n: f"bits{n.bit_length()}{'-' if n < 0 else ''}")
def test_integer_round_trip(n):
    pool = ak.ExprPool()
    e = pool.integer(n)
    value = _node_int(e)
    assert type(value) is int
    assert value == n


@pytest.mark.parametrize("n", VALUES, ids=lambda n: f"bits{n.bit_length()}{'-' if n < 0 else ''}")
def test_node_keeps_its_string_shape(n):
    # `node()` is public and has always returned decimal text; it still does,
    # and the text agrees with the exact value (compared without `int(str)`).
    pool = ak.ExprPool()
    node = pool.integer(n).node()
    assert node[0] == "integer"
    assert isinstance(node[1], str)
    assert ak.alkahest._decimal_to_int(node[1]) == n


def test_bool_and_index_protocol():
    class Idx:
        def __index__(self):
            return BIG

    pool = ak.ExprPool()
    assert _node_int(pool.integer(True)) == 1
    assert _node_int(pool.integer(Idx())) == BIG


def test_arithmetic_round_trip():
    pool = ak.ExprPool()
    a = pool.integer(BIG)
    b = pool.integer(-(10**20000))
    total = ak.simplify(a + b).value
    assert _node_int(total) == BIG - 10**20000
    product = ak.simplify(a * pool.integer(3)).value
    assert _node_int(product) == 3 * BIG


def test_operators_with_big_int_operands():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    s = x + 10**5000
    consts = [a for a in s._node_exact()[1] if a._node_exact()[0] == "integer"]
    assert [_node_int(c) for c in consts] == [10**5000]
    p = x ** (10**5000)
    exp = p._node_exact()[2]
    assert _node_int(exp) == 10**5000
    r = 10**5000 - x
    assert r is not None


def test_rational_round_trip():
    pool = ak.ExprPool()
    q = pool.rational(10**5000, 3)
    tag, num, den = q._node_exact()
    assert (tag, num, den) == ("rational", 10**5000, 3)
    q = pool.rational(-(2**70000) - 1, 2**64 + 1)
    tag, num, den = q._node_exact()
    assert Fraction(num, den) == Fraction(-(2**70000) - 1, 2**64 + 1)


def test_fraction_operand():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = Fraction(10**5000, 3)
    s = x + f
    consts = [a._node_exact() for a in s._node_exact()[1] if a._node_exact()[0] == "rational"]
    assert consts == [["rational", 10**5000, 3]]
    huge = Fraction(-(10**6000) - 1, 10**4500 + 3)
    e = x * huge
    consts = [a._node_exact() for a in e._node_exact()[1] if a._node_exact()[0] == "rational"]
    assert Fraction(consts[0][1], consts[0][2]) == huge


def test_decimal_to_int_helper():
    to_int = ak.alkahest._decimal_to_int
    for n in VALUES:
        # Build the text with GMP (via node()) so the test never calls str(int).
        text = ak.ExprPool().integer(n).node()[1]
        assert to_int(text) == n
    with pytest.raises(ValueError):
        to_int("12a")


# ---------------------------------------------------------------------------
# evaluate(mode="exact")
# ---------------------------------------------------------------------------


def test_evaluate_exact_big_result():
    pool = ak.ExprPool()
    e = pool.integer(3) ** pool.integer(12000)
    res = ak.evaluate(e, {}, mode="exact")
    assert res.status == "ok", res.reason
    assert res.value == Fraction(3**12000)


def test_evaluate_exact_big_binding():
    pool = ak.ExprPool()
    y = pool.symbol("y")
    for mode in ("exact", "auto"):
        res = ak.evaluate(y + 1, {y: 10**5000}, mode=mode)
        assert res.status == "ok", (mode, res.reason)
        assert res.value == 10**5000 + 1
        res = ak.evaluate(y * 2, {y: Fraction(10**5000, 7)}, mode=mode)
        assert res.status == "ok", (mode, res.reason)
        assert res.value == Fraction(2 * 10**5000, 7)


# ---------------------------------------------------------------------------
# Polynomial coefficients
# ---------------------------------------------------------------------------


def test_unipoly_big_coefficients():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    p = ak.UniPoly.from_symbolic(x**2 * pool.integer(BIG) - pool.integer(10**20000), x)
    assert p.coefficients() == [-(10**20000), 0, BIG]
    assert p.leading_coeff == BIG


# ---------------------------------------------------------------------------
# Number theory
# ---------------------------------------------------------------------------


def test_number_theory_big_arguments():
    assert number_theory.isprime(10**5000 + 1) is False
    assert number_theory.isprime(2**127 - 1) is True
    j = number_theory.jacobi_symbol(3, 10**4400 + 1)
    assert j in (-1, 1)
    # (3 | n) for n ≡ 1 (mod 4) is (n mod 3 | 3) by reciprocity.
    assert j == {1: 1, 2: -1}[(10**4400 + 1) % 3]
    n = 2**15000
    assert number_theory.totient(n) == 2**14999
    assert number_theory.factorint(-(2**15000)) == {2: 15000, -1: 1}
    # Past u64, so the result comes back through the big-int path; a proved
    # next prime of a multi-thousand-digit number would take far too long.
    assert number_theory.nextprime(2**64) == 2**64 + 13


def test_modular_big_arguments():
    m = 10**5000 + 7
    # 1/3 mod m reconstructs to (1, 3).
    inv3 = pow(3, -1, m)
    assert modular.rational_reconstruction(inv3, m) == (1, 3)
    assert modular.select_lucky_prime(2**20000, []) > 2


def test_arithmetic_functions_big_values():
    ex = pytest.importorskip("alkahest.experimental")
    assert ex.divisor_sigma(0, 2**15000) == 15001
    assert ex.divisor_sigma(1, 2**15000) == 2**15001 - 1
    assert ex.moebius_mu(2**15000) == 0
    # p(n) grows past the digit limit quickly: p(10**7) has ~3500 digits,
    # so use a moderate n whose value is still a multi-limb int.
    assert ex.partition_number(1000) == 24061467864032622473692149727991


def test_lattice_big_entries():
    ex = pytest.importorskip("alkahest.experimental")
    lat = ex.Lattice.from_basis([[BIG, 0], [0, Fraction(1, 10**5000)]])
    # det(B Bᵀ) for a diagonal basis is the product of the squared entries.
    assert lat.determinant() == Fraction(BIG**2, 10**10000)
