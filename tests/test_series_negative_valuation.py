"""Series at a negative valuation: every term below the requested order.

Audit A9 (core-audit 2026-09-29). ``series`` returned ``order`` coefficients of
the *unit part* ``h**-v * f`` and then labelled any Laurent result ``O(h)``, so
a pole of order ``>= order`` silently lost terms that the remainder claimed
were present: ``series(sin(x)**-4, x, 0, 4)`` was ``x**-4 + 2/3*x**-2 + O(x)``
(missing ``11/45``), ``(x + x**2)**-3`` at order 3 lacked ``-10``.  Separately,
``((x - 2)/x)**-1`` expanded to a bare ``O(x**4)`` because ``simplify`` folded
the singular-in-form coefficients ``(-2*0**-1)**-1`` to ``0``.

``order`` now means what it means for ``puiseux_series``: every term with
exponent ``< order`` is present and the remainder is ``O(h**order)``.

``ORACLE`` holds the coefficients of ``Series[f^-k, {x, 0, 4}]`` from
Mathematica 14 (wolframscript), one entry per ``(f, k)``.
"""

from fractions import Fraction

import alkahest as ak
import pytest

ORACLE = {
    ("sin", 1): {-1: "1", 1: "1/6", 3: "7/360"},
    ("sin", 2): {-2: "1", 0: "1/3", 2: "1/15", 4: "2/189"},
    ("sin", 3): {-3: "1", -1: "1/2", 1: "17/120", 3: "457/15120"},
    ("sin", 4): {-4: "1", -2: "2/3", 0: "11/45", 2: "62/945", 4: "41/2835"},
    ("sin", 5): {-5: "1", -3: "5/6", -1: "3/8", 1: "367/3024", 3: "11513/362880"},
    ("sin", 6): {-6: "1", -4: "1", -2: "8/15", 0: "191/945", 2: "289/4725", 4: "491/31185"},
    ("tan", 1): {-1: "1", 1: "-1/3", 3: "-1/45"},
    ("tan", 2): {-2: "1", 0: "-2/3", 2: "1/15", 4: "2/189"},
    ("tan", 3): {-3: "1", -1: "-1", 1: "4/15", 3: "1/945"},
    ("tan", 4): {-4: "1", -2: "-4/3", 0: "26/45", 2: "-64/945", 4: "-19/2835"},
    ("tan", 5): {-5: "1", -3: "-5/3", -1: "1", 1: "-44/189", 3: "16/2835"},
    ("tan", 6): {-6: "1", -4: "-2", -2: "23/15", 0: "-502/945", 2: "304/4725", 4: "128/31185"},
    ("asin", 1): {-1: "1", 1: "-1/6", 3: "-17/360"},
    ("asin", 2): {-2: "1", 0: "-1/3", 2: "-1/15", 4: "-31/945"},
    ("asin", 3): {-3: "1", -1: "-1/2", 1: "-7/120", 3: "-457/15120"},
    ("asin", 4): {-4: "1", -2: "-2/3", 0: "-1/45", 2: "-4/189", 4: "-41/2835"},
    ("asin", 5): {-5: "1", -3: "-5/6", -1: "1/24", 1: "-31/3024", 3: "-3287/362880"},
    ("asin", 6): {-6: "1", -4: "-1", -2: "2/15", 0: "-2/945", 2: "-1/225", 4: "-31/7425"},
    ("atan", 1): {-1: "1", 1: "1/3", 3: "-4/45"},
    ("atan", 2): {-2: "1", 0: "2/3", 2: "-1/15", 4: "32/945"},
    ("atan", 3): {-3: "1", -1: "1", 1: "1/15", 3: "-1/945"},
    ("atan", 4): {-4: "1", -2: "4/3", 0: "14/45", 2: "-4/189", 4: "19/2835"},
    ("atan", 5): {-5: "1", -3: "5/3", -1: "2/3", 1: "2/189", 3: "11/2835"},
    ("atan", 6): {-6: "1", -4: "2", -2: "17/15", 0: "124/945", 2: "-1/225", 4: "2/7425"},
    ("sinh", 1): {-1: "1", 1: "-1/6", 3: "7/360"},
    ("sinh", 2): {-2: "1", 0: "-1/3", 2: "1/15", 4: "-2/189"},
    ("sinh", 3): {-3: "1", -1: "-1/2", 1: "17/120", 3: "-457/15120"},
    ("sinh", 4): {-4: "1", -2: "-2/3", 0: "11/45", 2: "-62/945", 4: "41/2835"},
    ("sinh", 5): {-5: "1", -3: "-5/6", -1: "3/8", 1: "-367/3024", 3: "11513/362880"},
    ("sinh", 6): {-6: "1", -4: "-1", -2: "8/15", 0: "-191/945", 2: "289/4725", 4: "-491/31185"},
    ("x+x^2", 1): {-1: "1", 0: "-1", 1: "1", 2: "-1", 3: "1", 4: "-1"},
    ("x+x^2", 2): {-2: "1", -1: "-2", 0: "3", 1: "-4", 2: "5", 3: "-6", 4: "7"},
    ("x+x^2", 3): {-3: "1", -2: "-3", -1: "6", 0: "-10", 1: "15", 2: "-21", 3: "28", 4: "-36"},
    ("x+x^2", 4): {
        -4: "1",
        -3: "-4",
        -2: "10",
        -1: "-20",
        0: "35",
        1: "-56",
        2: "84",
        3: "-120",
        4: "165",
    },
    ("x+x^2", 5): {
        -5: "1",
        -4: "-5",
        -3: "15",
        -2: "-35",
        -1: "70",
        0: "-126",
        1: "210",
        2: "-330",
        3: "495",
        4: "-715",
    },
    ("x+x^2", 6): {
        -6: "1",
        -5: "-6",
        -4: "21",
        -3: "-56",
        -2: "126",
        -1: "-252",
        0: "462",
        1: "-792",
        2: "1287",
        3: "-2002",
        4: "3003",
    },
    ("expm1", 1): {-1: "1", 0: "-1/2", 1: "1/12", 3: "-1/720"},
    ("expm1", 2): {-2: "1", -1: "-1", 0: "5/12", 1: "-1/12", 2: "1/240", 3: "1/720", 4: "-1/6048"},
    ("expm1", 3): {
        -3: "1",
        -2: "-3/2",
        -1: "1",
        0: "-3/8",
        1: "19/240",
        2: "-1/160",
        3: "-1/945",
        4: "1/4032",
    },
    ("expm1", 4): {
        -4: "1",
        -3: "-2",
        -2: "11/6",
        -1: "-1",
        0: "251/720",
        1: "-3/40",
        2: "221/30240",
        3: "11/15120",
        4: "-199/725760",
    },
    ("expm1", 5): {
        -5: "1",
        -4: "-5/2",
        -3: "35/12",
        -2: "-25/12",
        -1: "1",
        0: "-95/288",
        1: "863/12096",
        2: "-95/12096",
        3: "-47/103680",
        4: "79/290304",
    },
    ("expm1", 6): {
        -6: "1",
        -5: "-3",
        -4: "17/4",
        -3: "-15/4",
        -2: "137/60",
        -1: "-1",
        0: "19087/60480",
        1: "-275/4032",
        2: "9829/1209600",
        3: "19/80640",
        4: "-8213/31933440",
    },
    ("log1p", 1): {-1: "1", 0: "1/2", 1: "-1/12", 2: "1/24", 3: "-19/720", 4: "3/160"},
    ("log1p", 2): {-2: "1", -1: "1", 0: "1/12", 2: "-1/240", 3: "1/240", 4: "-221/60480"},
    ("log1p", 3): {
        -3: "1",
        -2: "3/2",
        -1: "1/2",
        1: "1/240",
        2: "-1/480",
        3: "1/945",
        4: "-11/20160",
    },
    ("log1p", 4): {
        -4: "1",
        -3: "2",
        -2: "7/6",
        -1: "1/6",
        0: "-1/720",
        2: "1/3024",
        3: "-1/3024",
        4: "199/725760",
    },
    ("log1p", 5): {
        -5: "1",
        -4: "5/2",
        -3: "25/12",
        -2: "5/8",
        -1: "1/24",
        1: "-1/6048",
        2: "1/12096",
        3: "-19/725760",
        4: "-1/483840",
    },
    ("log1p", 6): {
        -6: "1",
        -5: "3",
        -4: "13/4",
        -3: "3/2",
        -2: "31/120",
        -1: "1/120",
        0: "1/30240",
        2: "-1/57600",
        3: "1/57600",
        4: "-101/7603200",
    },
    ("1-cos", 1): {-2: "2", 0: "1/6", 2: "1/120", 4: "1/3024"},
    ("1-cos", 2): {-4: "4", -2: "2/3", 0: "11/180", 2: "31/7560", 4: "41/181440"},
    ("1-cos", 3): {-6: "8", -4: "2", -2: "4/15", 0: "191/7560", 2: "289/151200", 4: "491/3991680"},
    ("1-cos", 4): {
        -8: "16",
        -6: "16/3",
        -4: "14/15",
        -2: "4/35",
        0: "2497/226800",
        2: "317/356400",
        4: "341749/5448643200",
    },
    ("1-cos", 5): {
        -10: "32",
        -8: "40/3",
        -6: "26/9",
        -4: "82/189",
        -2: "16/315",
        0: "14797/2993760",
        2: "6803477/16345929600",
        4: "50971/1634592960",
    },
    ("1-cos", 6): {
        -12: "64",
        -10: "32",
        -8: "124/15",
        -6: "278/189",
        -4: "958/4725",
        -2: "16/693",
        0: "92427157/40864824000",
        2: "3203699/16345929600",
        4: "170403199/11115232128000",
    },
    ("(x-2)/x", 1): {1: "-1/2", 2: "-1/4", 3: "-1/8", 4: "-1/16"},
    ("(x-2)/x", 2): {2: "1/4", 3: "1/4", 4: "3/16"},
    ("(x-2)/x", 3): {3: "-1/8", 4: "-3/16"},
    ("(x-2)/x", 4): {4: "1/16"},
    ("(x-2)/x", 5): {5: "-1/32"},
    ("(x-2)/x", 6): {6: "1/64"},
    ("x*cos", 1): {-1: "1", 1: "1/2", 3: "5/24"},
    ("x*cos", 2): {-2: "1", 0: "1", 2: "2/3", 4: "17/45"},
    ("x*cos", 3): {-3: "1", -1: "3/2", 1: "11/8", 3: "241/240"},
    ("x*cos", 4): {-4: "1", -2: "2", 0: "7/3", 2: "94/45", 4: "502/315"},
    ("x*cos", 5): {-5: "1", -3: "5/2", -1: "85/24", 1: "541/144", 3: "26837/8064"},
    ("x*cos", 6): {-6: "1", -4: "3", -2: "5", 0: "92/15", 2: "130/21", 4: "25757/4725"},
}

FUNCS = {
    "sin": lambda p, x: ak.sin(x),
    "tan": lambda p, x: ak.tan(x),
    "asin": lambda p, x: ak.asin(x),
    "atan": lambda p, x: ak.atan(x),
    "sinh": lambda p, x: ak.sinh(x),
    "x+x^2": lambda p, x: x + x * x,
    "expm1": lambda p, x: ak.exp(x) - 1,
    "log1p": lambda p, x: ak.log(1 + x),
    "1-cos": lambda p, x: 1 - ak.cos(x),
    "(x-2)/x": lambda p, x: (x - 2) * x**-1,
    "x*cos": lambda p, x: x * ak.cos(x),
}


def _number(e):
    n = e.node()
    if n[0] == "integer":
        return Fraction(int(n[1]))
    if n[0] == "rational":
        return Fraction(int(n[1]), int(n[2]))
    raise AssertionError(f"coefficient {e} is not a rational literal")


def _exponent_of_power(e, x):
    """``e`` as ``x**k``, or ``None``."""
    if e == x:
        return 1
    n = e.node()
    if n[0] == "pow" and n[1] == x and n[2].node()[0] == "integer":
        return int(n[2].node()[1])
    return None


def _terms(series_expr, x):
    """``({exponent: coefficient}, big_o_exponent)`` of a ``series`` result."""
    n = series_expr.node()
    parts = n[1] if n[0] == "add" else [series_expr]
    terms, big_o = {}, None
    for t in parts:
        tn = t.node()
        if tn[0] == "big_o":
            big_o = _exponent_of_power(tn[1], x)
            continue
        factors = tn[1] if tn[0] == "mul" else [t]
        k, c = 0, Fraction(1)
        for f in factors:
            e = _exponent_of_power(f, x)
            if e is None:
                c *= _number(f)
            else:
                k += e
        assert k not in terms, f"exponent {k} appears twice in {series_expr}"
        if c != 0:
            terms[k] = c
    return terms, big_o


CASES = [(f, k, order) for (f, k) in ORACLE for order in range(1, 6)]


@pytest.mark.parametrize(("fname", "k", "order"), CASES)
def test_negative_power_series_matches_mathematica(fname, k, order):
    p = ak.ExprPool()
    x = p.symbol("x")
    f = FUNCS[fname](p, x) ** -k
    got, big_o = _terms(ak.series(f, x, p.integer(0), order).expr, x)
    want = {e: Fraction(c) for e, c in ORACLE[(fname, k)].items() if e < order}
    assert big_o == order
    assert got == want


def test_audit_a9_examples():
    """The four reports, spelled out."""
    p = ak.ExprPool()
    x = p.symbol("x")
    z = p.integer(0)

    got, big_o = _terms(ak.series(ak.sin(x) ** -4, x, z, 4).expr, x)
    assert got[0] == Fraction(11, 45)
    assert big_o == 4

    got, _ = _terms(ak.series((x + x * x) ** -3, x, z, 3).expr, x)
    assert got[0] == -10

    got, _ = _terms(ak.series(ak.asin(x) ** -6, x, z, 3).expr, x)
    assert got == {-6: 1, -4: -1, -2: Fraction(2, 15), 0: Fraction(-2, 945), 2: Fraction(-1, 225)}

    got, big_o = _terms(ak.series(((x - 2) * x**-1) ** -1, x, z, 4).expr, x)
    assert got == {1: Fraction(-1, 2), 2: Fraction(-1, 4), 3: Fraction(-1, 8)}
    assert big_o == 4


def test_simple_pole_keeps_the_requested_remainder():
    p = ak.ExprPool()
    x = p.symbol("x")
    got, big_o = _terms(ak.series(x**-1, x, p.integer(0), 4).expr, x)
    assert got == {-1: 1}
    assert big_o == 4


def test_a_singular_form_is_not_laundered_into_a_wrong_coefficient():
    """Found by re-running the audit's differential harness against the fix:
    ``simplify`` folds ``0*(1 + 0**-1)`` and ``(-2*asin(0)**-1)**-1`` to ``0``,
    so expanding by substitution silently dropped terms of functions that are
    perfectly analytic at the point."""
    p = ak.ExprPool()
    x = p.symbol("x")
    z = p.integer(0)

    got, big_o = _terms(ak.series(x * (x**-1 + 1), x, z, 3).expr, x)
    assert got == {0: 1, 1: 1}
    assert big_o == 3

    # A removable singularity inside a function argument.
    got, big_o = _terms(ak.series(ak.exp(x * (-2 * x**-1) ** -1), x, z, 5).expr, x)
    assert got == {0: 1, 2: Fraction(-1, 2), 4: Fraction(1, 8)}
    assert big_o == 5
    got, big_o = _terms(ak.series(ak.log(x * x**-1) * (x + x**3), x, z, 3).expr, x)
    assert got == {}
    assert big_o == 3

    # 2x·e^x·asin(x)/(x − 2): Mathematica gives -x^2 - 3x^3/2 + O(x^4)
    f = 2 * x * ak.exp(x) * ((x - 2) * ak.asin(x) ** -1) ** -1
    got, big_o = _terms(ak.series(f, x, z, 4).expr, x)
    assert got == {2: -1, 3: Fraction(-3, 2)}
    assert big_o == 4


@pytest.mark.parametrize(
    "build",
    [
        # x^2/log(x) has no Laurent expansion at 0; it came back with
        # coefficients like 2*log(0)**-1, which evaluates to -0.0.
        lambda p, x: x * x * ak.log(x) ** -1,
        # 4x^2*sqrt(x^3) = 4x^(7/2), which is not O(x^4): it came back as
        # 4*sqrt(0)*x^2 + O(x^4).
        lambda p, x: 4 * x * x * ak.sqrt(x**3),
    ],
)
def test_a_non_laurent_point_is_refused_not_zeroed(build):
    p = ak.ExprPool()
    x = p.symbol("x")
    with pytest.raises(ak.SeriesError) as excinfo:
        ak.series(build(p, x), x, p.integer(0), 4)
    assert excinfo.value.code == "E-SERIES-004"
