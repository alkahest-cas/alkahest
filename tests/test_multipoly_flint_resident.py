"""MultiPoly keeps its FLINT form between operations; results are unchanged.

A large product, a GCD or a factorisation of a ``MultiPoly`` now runs on a
cached FLINT form, and a FLINT result is converted back to terms only when
they are read (``str``, ``total_degree``, ``integer_content``, ...). These
tests pin that every observable result is what the term-map arithmetic gives.
"""

import alkahest as ak
import pytest


def _poly(s, names):
    return ak.MultiPoly.from_symbolic(ak.parse(s), [ak.symbol(v) for v in names])


VARS = ["x", "y", "z"]


def _expanded(s):
    return _poly(str(ak.expand(ak.parse(s))), VARS)


def test_large_product_matches_expanded_string():
    f = _expanded("(1 + x + y + z)^6")
    one = _poly("1", VARS)
    prod = f * (f + one)
    expect = _expanded("(1 + x + y + z)^6 * ((1 + x + y + z)^6 + 1)")
    assert str(prod) == str(expect)
    assert prod.total_degree == 12
    assert prod.integer_content() == "1"
    assert not prod.is_zero


def test_chain_of_flint_resident_results():
    a = _expanded("(x + 2*y - z + 3)^5")
    b = _expanded("(x - y + 2*z - 1)^5")
    c = _expanded("(x*y + z + 1)^2")
    # ((a*b + c) * c - a) * b, every intermediate left in FLINT
    r = (a * b + c) * c - a
    r = r * b
    expect = _expanded(
        "(((x + 2*y - z + 3)^5 * (x - y + 2*z - 1)^5 + (x*y + z + 1)^2)"
        " * (x*y + z + 1)^2 - (x + 2*y - z + 3)^5) * (x - y + 2*z - 1)^5"
    )
    assert str(r) == str(expect)
    assert str(r - r) == "0"
    assert (r - r).is_zero


def test_gcd_and_factor_of_flint_resident_products():
    a = _expanded("(x + y + 1)^3")
    b = _expanded("(x - z)^2 * (y + 2)")
    g = _expanded("(x*y - z + 4)^2")
    ag, bg = a * g, b * g
    assert str(ag.gcd(bg)) == str(g)
    fac = (ag * bg).factor_z()
    assert fac.verification["status"] == "exactly_verified"
    # x + y + 1 (3), x - z (2), y + 2 (1), x*y - z + 4 (4)
    mults = sorted(e for _, e in fac.factor_list())
    assert mults == [1, 2, 3, 4]
    # The same factorisation as of the polynomial rebuilt from its terms.
    rebuilt = _expanded("(x + y + 1)^3 * (x - z)^2 * (y + 2) * (x*y - z + 4)^4")
    assert [(str(p), e) for p, e in rebuilt.factor_z().factor_list()] == [
        (str(p), e) for p, e in fac.factor_list()
    ]


def test_product_exponent_overflow_still_refused():
    big = _poly("x^3000000000 + y", ["x", "y"])
    many = _poly(str(ak.expand(ak.parse("(x + y + 1)^12"))), ["x", "y"])
    with pytest.raises(Exception, match="E-POLY-004"):
        (big * many) * big


def test_univariate_factor_uses_cyclotomic_structure():
    p = _poly("x^2310 - 1", ["x"])
    fac = p.factor_z()
    assert len(fac.factor_list()) == 32
    assert fac.verification["status"] == "exactly_verified"
    # Same in a ring with more variables, the polynomial in only one of them.
    q = _poly("y^60 - 1", ["x", "y"])
    assert len(q.factor_z().factor_list()) == 12


def test_expand_without_denominators_is_the_cancelled_sum():
    x, y = ak.symbol("x"), ak.symbol("y")
    for e in [
        (x + y + 1) ** 4 * (x - 1),
        (x + 1) ** 2 - (x + 1) ** 2,
        ak.parse("(x + 1)^3 / 2"),
        ak.parse("2*(x+y)^2 + 3"),
    ]:
        out = ak.expand(e)
        again = ak.cancel(out)
        assert str(again) == str(ak.cancel(e))
    assert str(ak.expand((x + 1) ** 2 - (x + 1) ** 2)) == "0"
