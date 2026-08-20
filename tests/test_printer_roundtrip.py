"""Printed expressions must re-read as the expression that was printed.

Regression guard for the `(-1)^n` → `-1^n` printer bug: a negative (or
otherwise non-atomic) power base was emitted without parentheses, so every
exported form meant `-(1^n)` under the standard precedence used by Python,
sympy, LaTeX — and by alkahest's own parser.
"""

import alkahest as ak
import pytest
from alkahest import ExprPool, latex, parse, simplify, unicode_str

sympy = pytest.importorskip("sympy")


@pytest.fixture
def pool():
    return ExprPool()


@pytest.fixture
def syms(pool):
    return {
        "n": pool.symbol("n"),
        "x": pool.symbol("x"),
        "a": pool.symbol("a"),
        "b": pool.symbol("b"),
    }


def _cases(pool, syms):
    """`label -> (expr, python_source, latex, unicode)` for each printed form.

    `python_source` is an independent, unambiguous spelling of the same
    mathematics; the round-trip test sympifies it and the printed form and
    demands they agree.
    """
    n, x, a, b = syms["n"], syms["x"], syms["a"], syms["b"]
    one = pool.integer(1)
    cases = [
        ("(-1)^n", pool.integer(-1) ** n, "(-1)**n", r"\left(-1\right)^n", "(-1)^(n)"),
        ("(-2)^n", pool.integer(-2) ** n, "(-2)**n", r"\left(-2\right)^n", "(-2)^(n)"),
        (
            "(-1/2)^n",
            pool.rational(-1, 2) ** n,
            "(sympy.Rational(-1, 2))**n",
            r"\left(-\frac{1}{2}\right)^n",
            "(-½)^(n)",
        ),
        (
            "(1/2)^n",
            pool.rational(1, 2) ** n,
            "(sympy.Rational(1, 2))**n",
            r"\left(\frac{1}{2}\right)^n",
            "½^(n)",
        ),
        (
            "(3/7)^n",
            pool.rational(3, 7) ** n,
            "(sympy.Rational(3, 7))**n",
            r"\left(\frac{3}{7}\right)^n",
            "(3/7)^(n)",
        ),
        ("(x + 1)^n", (x + one) ** n, "(x + 1)**n", r"\left(x + 1\right)^n", "(x + 1)^(n)"),
        ("(-x)^n", (pool.integer(-1) * x) ** n, "(-x)**n", r"\left(-x\right)^n", "(-x)^(n)"),
        ("(a*b)^n", (a * b) ** n, "(a*b)**n", r"\left(a b\right)^n", "(a·b)^(n)"),
        # negative exponents
        ("x^-2", x ** pool.integer(-2), "x**-2", "x^{-2}", "x⁻²"),
        (
            "(-2)^-3",
            pool.integer(-2) ** pool.integer(-3),
            "(-2)**-3",
            r"\left(-2\right)^{-3}",
            "(-2)⁻³",
        ),
        (
            "(-x)^-1",
            (pool.integer(-1) * x) ** pool.integer(-1),
            "(-x)**-1",
            r"\frac{1}{\left(-x\right)}",
            "(-x)⁻¹",
        ),
        # nested powers
        (
            "(x^2)^3",
            (x ** pool.integer(2)) ** pool.integer(3),
            "(x**2)**3",
            r"\left(x^2\right)^3",
            "(x²)³",
        ),
        (
            "((-1)^n)^2",
            (pool.integer(-1) ** n) ** pool.integer(2),
            "((-1)**n)**2",
            r"\left(\left(-1\right)^n\right)^2",
            "((-1)^(n))²",
        ),
        # a negative base inside a product — the M1 boundary shape `-16 * (-2)^n`
        (
            "-16 * (-2)^n",
            pool.integer(-16) * (pool.integer(-2) ** n),
            "-16 * (-2)**n",
            r"-16 \left(-2\right)^n",
            "-16·(-2)^(n)",
        ),
    ]
    return {c[0]: c[1:] for c in cases}


LABELS = [
    "(-1)^n",
    "(-2)^n",
    "(-1/2)^n",
    "(1/2)^n",
    "(3/7)^n",
    "(x + 1)^n",
    "(-x)^n",
    "(a*b)^n",
    "x^-2",
    "(-2)^-3",
    "(-x)^-1",
    "(x^2)^3",
    "((-1)^n)^2",
    "-16 * (-2)^n",
]


def _sympify(src):
    return sympy.sympify(src, locals={"sympy": sympy})


@pytest.mark.parametrize("label", LABELS)
def test_str_round_trips_through_sympy(pool, syms, label):
    """`sympify(str(e).replace("^", "**"))` is the expression that was printed."""
    expr, source, _tex, _uni = _cases(pool, syms)[label]
    printed = _sympify(str(expr).replace("^", "**"))
    expected = _sympify(source)
    assert printed == expected, f"{label}: str={str(expr)!r} reads as {printed}, want {expected}"


@pytest.mark.parametrize("label", LABELS)
def test_str_round_trips_through_alkahest_parse(pool, syms, label):
    """alkahest can re-read its own output."""
    expr, _source, _tex, _uni = _cases(pool, syms)[label]
    reparsed = parse(str(expr), pool, syms)
    assert simplify(reparsed).value == simplify(expr).value, (
        f"{label}: str={str(expr)!r} parses to {reparsed}"
    )


@pytest.mark.parametrize("label", LABELS)
def test_latex(pool, syms, label):
    expr, _source, tex, _uni = _cases(pool, syms)[label]
    assert latex(expr) == tex, label


@pytest.mark.parametrize("label", LABELS)
def test_unicode(pool, syms, label):
    expr, _source, _tex, uni = _cases(pool, syms)[label]
    assert unicode_str(expr) == uni, label


def test_case_table_is_complete(pool, syms):
    """`LABELS` and the case table must not drift apart."""
    assert sorted(_cases(pool, syms)) == sorted(LABELS)


def test_negative_base_agrees_with_evaluation(pool, syms):
    """The printed form of `(-1)^n` must not flip sign when re-read at `n = 4`."""
    n = syms["n"]
    expr = pool.integer(-1) ** n
    direct = simplify(ak.subs(expr, {n: pool.integer(4)})).value
    reparsed = parse(str(expr), pool, syms)
    via_text = simplify(ak.subs(reparsed, {n: pool.integer(4)})).value
    assert str(direct) == "1"
    assert str(via_text) == str(direct)
