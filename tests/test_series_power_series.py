"""``series`` through truncated power-series arithmetic over the rationals.

When every coefficient is rational the expansion no longer differentiates the
expression ``order`` times; it combines exact coefficient vectors.  These are
the user-visible consequences, each checked against Mathematica 14 ``Series``
/ ``Limit``.
"""

from fractions import Fraction

import alkahest as ak


def _terms(series_expr, x):
    """``({exponent: coefficient}, big_o_exponent)``; coefficients must be literals."""

    def power(f):
        if f == x:
            return 1
        n = f.node()
        if n[0] == "pow" and n[1] == x and n[2].node()[0] == "integer":
            return int(n[2].node()[1])
        return None

    def number(e):
        n = e.node()
        if n[0] == "integer":
            return Fraction(int(n[1]))
        if n[0] == "rational":
            return Fraction(int(n[1]), int(n[2]))
        raise AssertionError(f"coefficient {e} is not a rational literal")

    n = series_expr.node()
    parts = n[1] if n[0] == "add" else [series_expr]
    terms, big_o = {}, None
    for t in parts:
        tn = t.node()
        if tn[0] == "big_o":
            big_o = power(tn[1])
            continue
        k, c = 0, Fraction(1)
        for f in tn[1] if tn[0] == "mul" else [t]:
            e = power(f)
            if e is None:
                c *= number(f)
            else:
                k += e
        if c != 0:
            terms[k] = c
    return terms, big_o


def test_a_deep_composition_expands_exactly():
    p = ak.ExprPool()
    x = p.symbol("x")
    e = ak.sin(ak.tan(x)) - ak.tan(ak.sin(x))
    got, big_o = _terms(ak.series(e, x, p.integer(0), 16).expr, x)
    assert big_o == 16
    assert got == {
        7: Fraction(-1, 30),
        9: Fraction(-29, 756),
        11: Fraction(-1913, 75600),
        13: Fraction(-95, 7392),
        15: Fraction(-311148869, 54486432000),
    }


def test_sin_tan_limit_is_not_refuted_by_rounding():
    # Was E-LIMIT-005: at x = 1e-3 both terms round to the same double, the
    # samples settled on 0, and the numeric check refused the right answer.
    p = ak.ExprPool()
    x = p.symbol("x")
    e = (ak.sin(ak.tan(x)) - ak.tan(ak.sin(x))) / x**7
    assert ak.limit(e, x, p.integer(0)).value == p.rational(-1, 30)


def test_tanh_coefficients_are_literals():
    # Were tanh(0)-laden: (1 + -1*tanh(0)^2)*x + ...
    p = ak.ExprPool()
    x = p.symbol("x")
    got, _ = _terms(ak.series(ak.tanh(x), x, p.integer(0), 6).expr, x)
    assert got == {1: 1, 3: Fraction(-1, 3), 5: Fraction(2, 15)}


def test_poles_the_derivative_route_got_wrong_or_refused():
    p = ak.ExprPool()
    x = p.symbol("x")
    z = p.integer(0)
    # Was a bare O(x).
    got, big_o = _terms(ak.series(x * ak.tanh(x) ** -3, x, z, 1).expr, x)
    assert (got, big_o) == ({-2: 1, 0: 1}, 1)
    # Were E-SERIES-004.
    got, _ = _terms(ak.series(ak.atanh(x) ** -1, x, z, 6).expr, x)
    assert got == {-1: 1, 1: Fraction(-1, 3), 3: Fraction(-4, 45), 5: Fraction(-44, 945)}
    got, _ = _terms(ak.series(ak.log(1 + ak.atanh(x)) ** -3, x, z, 2).expr, x)
    assert got == {
        -3: 1,
        -2: Fraction(3, 2),
        -1: Fraction(-1, 2),
        0: -1,
        1: Fraction(-23, 240),
    }
