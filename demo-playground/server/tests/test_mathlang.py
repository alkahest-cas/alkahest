"""Parser tests for the Math tab's input language (no Alkahest needed)."""

from __future__ import annotations

import mathlang as ml
import pytest
from mathlang import MathInputError


def parse(text, session=None):
    cmd, latex = ml.interpret(text, session)
    return cmd, latex


def refuse(text, code=None, session=None):
    with pytest.raises(MathInputError) as info:
        ml.interpret(text, session)
    if code is not None:
        assert info.value.code == code, info.value.message
    assert info.value.message
    return info.value


# ── the documented examples ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "var", "definite"),
    [
        ("integrate sin(x)^2 dx", "x", False),
        ("∫ x e^x dx", "x", False),
        ("integrate 1/(1+x^2) from 0 to oo", None, True),
        ("integrate x^2 dx from 0 to 1", "x", True),
        ("integral of t^2 dt", "t", False),
        ("integrate(x^2, x, 0, 1)", "x", True),
        ("integrate x y with respect to y", "y", False),
    ],
)
def test_integrate(text, var, definite):
    cmd, latex = parse(text)
    assert cmd.kind == "integrate"
    assert cmd.var == var
    assert (cmd.lower is not None) == definite
    assert latex.startswith(r"\int")
    assert r"\,d" in latex


def test_integrate_infinite_bound_latex():
    _, latex = parse("integrate 1/(1+x^2) from 0 to oo")
    assert r"\int_{0}^{\infty}" in latex
    assert r"\frac{1}{1 + x^{2}}" in latex


@pytest.mark.parametrize(
    ("text", "var", "order"),
    [
        ("d/dx x^3 sin(x)", "x", 1),
        ("d/d x x^2", "x", 1),
        ("derivative of tan(x)", None, 1),
        ("differentiate t^3 with respect to t", "t", 1),
        ("diff(x^3 sin(x), x, 2)", "x", 2),
        ("d²/dx² x^3 sin(x)", "x", 2),
        ("d^3/dx^3 x^5", "x", 3),
        ("second derivative of x^4", None, 2),
        ("∂/∂y x y^2", "y", 1),
    ],
)
def test_derivatives(text, var, order):
    cmd, latex = parse(text)
    assert cmd.kind == "diff"
    assert cmd.var == var
    assert cmd.order == order
    assert latex.startswith(r"\frac{d")


def test_second_derivative_latex():
    _, latex = parse("d²/dx² x^3 sin(x)")
    assert latex.startswith(r"\frac{d^{2}}{dx^{2}}")


def test_solve_single():
    cmd, latex = parse("solve x^2 - 5x + 6 = 0")
    assert cmd.kind == "solve"
    assert len(cmd.equations) == 1
    assert "5 x" in latex


def test_solve_system():
    cmd, latex = parse("solve x^2 + y^2 = 1, x - y = 0 for x, y")
    assert cmd.kind == "solve"
    assert len(cmd.equations) == 2
    assert cmd.vars == ["x", "y"]
    assert r"\text{for }x, y" in latex


def test_solve_call_form_and_and():
    cmd, _ = parse("solve(x + y = 2, x - y = 0, x, y)")
    assert cmd.vars == ["x", "y"]
    cmd, _ = parse("solve x + y = 2 and x - y = 0")
    assert len(cmd.equations) == 2


def test_bare_equation_is_solve():
    cmd, _ = parse("x^2 = 4")
    assert cmd.kind == "solve"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("simplify (x^2-1)/(x-1)", "simplify"),
        ("factor x^4 - 1", "factor"),
        ("expand (x+1)^5", "expand"),
        ("factor(x^2 - y^2)", "factor"),
    ],
)
def test_algebra_commands(text, kind):
    cmd, latex = parse(text)
    assert cmd.kind == kind
    assert latex.startswith(r"\operatorname{" + kind + "}")


@pytest.mark.parametrize(
    ("text", "var", "direction"),
    [
        ("limit sin(x)/x as x -> 0", "x", None),
        ("limit sin(x)/x as x → 0", "x", None),
        ("limit 1/x as x -> 0+", "x", "+"),
        ("limit 1/x as x -> 0 from the left", "x", "-"),
        ("lim x->0 sin(x)/x", "x", None),
        ("lim x->0^+ 1/x", "x", "+"),
        ("lim_{x->0} sin(x)/x", "x", None),
        ("limit(sin(x)/x, x, 0)", "x", None),
        ("limit (1 + 1/n)^n as n -> oo", "n", None),
    ],
)
def test_limits(text, var, direction):
    cmd, latex = parse(text)
    assert cmd.kind == "limit"
    assert cmd.var == var
    assert cmd.direction == direction
    assert latex.startswith(r"\lim_{")


def test_series():
    cmd, latex = parse("series exp(x) at x = 0 to order 6")
    assert cmd.kind == "series"
    assert (cmd.var, cmd.order) == ("x", 6)
    assert cmd.point is not None
    assert cmd.point.value == "0"
    assert "O" in latex
    cmd, _ = parse("taylor sin(x) around x = pi up to order 4")
    assert cmd.order == 4
    assert cmd.point is not None
    assert cmd.point.kind == "const"
    cmd, _ = parse("series(cos(t), t, 0, 8)")
    assert (cmd.var, cmd.order) == ("t", 8)


def test_sum_and_product():
    cmd, latex = parse("sum 1/k^2 for k = 1 to oo")
    assert cmd.kind == "sum"
    assert cmd.var == "k"
    assert latex.startswith(r"\sum_{k=1}^{\infty}")
    cmd, _ = parse("sum 1/i^2 for i = 1 to oo")
    # a bound index named i is a variable, not the imaginary unit
    assert cmd.expr is not None
    assert "i" in ml.free_names(cmd.expr)
    cmd, latex = parse("product k for k = 1 to 5")
    assert cmd.kind == "product"
    assert latex.startswith(r"\prod")


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("numeric pi to 50 digits", 50),
        ("N[pi, 50]", 50),
        ("N[sqrt(2)]", 15),
        ("approx e", 15),
        ("N(pi, 30)", 30),
    ],
)
def test_numeric(text, digits):
    cmd, _ = parse(text)
    assert cmd.kind == "numeric"
    assert cmd.digits == digits


def test_bare_expression():
    cmd, latex = parse("sin(pi/6) + sqrt(8)")
    assert cmd.kind == "expr"
    assert latex == r"\sin\left(\frac{\pi}{6}\right) + \sqrt{8}"


# ── expression reading ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "latex"),
    [
        ("2x", "2 x"),
        ("x sin(x)", r"x \sin\left(x\right)"),
        ("(x+1)(x-1)", r"\left(x + 1\right) \left(x - 1\right)"),
        ("x e^x", "x e^{x}"),
        ("sin^2 x + cos^2 x", r"\sin^{2}\left(x\right) + \cos^{2}\left(x\right)"),
        ("sin x cos x", r"\sin\left(x\right) \cos\left(x\right)"),
        ("sin x^2", r"\sin\left(x^{2}\right)"),
        ("x⁻¹", "x^{-1}"),
        ("x²", "x^{2}"),
        ("√8 + π", r"\sqrt{8} + \pi"),
        ("2 × 3 ÷ 4 − 1", r"\frac{2 \cdot 3}{4} - 1"),
        ("|x - 1|", r"\left|x - 1\right|"),
        ("5!", "5!"),
        ("-x^2", "-x^{2}"),
        ("2^3^2", "2^{3^{2}}"),
        ("x**2", "x^{2}"),
        ("α + β", r"\alpha + \beta"),
        ("x_1 + x2", "x_{1} + x_{2}"),
        ("sin^-1(x)", r"\arcsin\left(x\right)"),
        ("ln(x)", r"\ln\left(x\right)"),
        ("e^(i pi)", r"e^{i \pi}"),
    ],
)
def test_expression_latex(text, latex):
    _, got = parse(text)
    assert got == latex


def test_implicit_letters_split():
    cmd, latex = parse("2xy")
    assert ml.free_names(cmd.expr) == {"x", "y"}
    assert latex == "2 x y"


def test_definitions_and_reuse():
    cmd, latex = parse("f = x^2 sin(x)")
    assert cmd.kind == "define"
    assert cmd.name == "f"
    assert cmd.params == []
    assert latex.startswith("f :=")
    cmd, _ = parse("f(x) := x^2 + 1")
    assert cmd.params == ["x"]
    session = ml.build_session([{"name": "f", "params": [], "value": "(x^2 * sin(x))"}])
    cmd, latex = parse("d/dx f", session)
    assert cmd.kind == "diff"
    assert latex == r"\frac{d}{dx}f"
    session = ml.build_session([{"name": "g", "params": ["t"], "value": "(t^2 + 1)"}])
    cmd, latex = parse("g(3) + 1", session)
    assert cmd.expr is not None
    assert cmd.expr.children[0].kind == "ucall"


def test_definition_names_are_validated():
    for bad in (
        {"name": "__import__", "params": [], "value": "1"},
        {"name": "sin", "params": [], "value": "1"},
        {"name": "f", "params": ["__class__"], "value": "1"},
        {"name": "f", "params": [], "value": "__import__('os')"},
        {"name": "f", "params": [], "value": "x" * (ml.MAX_DEFINITION_CHARS + 1)},
        {"name": 3, "params": [], "value": "1"},
        "not a dict",
    ):
        with pytest.raises(MathInputError):
            ml.build_session([bad])
    with pytest.raises(MathInputError):
        ml.build_session([{"name": "a", "value": "1"}] * (ml.MAX_SESSION_DEFINITIONS + 1))


def test_solve_cannot_be_defined():
    refuse("s = solve x^2 = 4", ml.E_UNSUPPORTED)


# ── refusals and malicious input ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        '__import__("os").system("ls")',
        "x.__class__",
        "().__class__.__bases__[0].__subclasses__()",
        "os.system('rm -rf /')",
        "open('/etc/passwd').read()",
        "eval('1')",
        "lambda: 1",
        "x; import os",
        "`ls`",
        "$(ls)",
        "x # comment",
        "f'{x}'",
        "x\\y",
        "@x",
    ],
)
def test_malicious_input_is_refused(text):
    err = refuse(text)
    assert err.code.startswith("E-PLAYGROUND-")


def test_dunder_name_refused():
    err = refuse("__import__", ml.E_UNKNOWN_NAME)
    assert "underscore" in err.message


def test_attribute_access_refused():
    err = refuse("x.real", ml.E_SYNTAX)
    assert "'.'" in err.message


def test_long_unknown_words_refused():
    refuse("open(x)", ml.E_UNKNOWN_NAME)
    refuse("system(1)", ml.E_UNKNOWN_NAME)


@pytest.mark.parametrize(
    "text",
    [
        "2^2^2^2^2^2",
        "2^2^2^2^2^2^2^2",
        "10^10^10",
        "9^9^9",
        "(2^2^2^2^2)^(2^2^2^2^2)",
        "x^100000",
        "1001!",
        "(10^100)!",
        "2^(-(2^2^2^2^2))",
    ],
)
def test_huge_numbers_refused(text):
    refuse(text, ml.E_NUMBER_TOO_LARGE)


def test_huge_power_through_definition_refused():
    session = ml.build_session([{"name": "a", "params": [], "value": "2^2^2^2^2"}])
    refuse("2^a", ml.E_NUMBER_TOO_LARGE, session)


def test_reasonable_powers_allowed():
    parse("2^1000")
    parse("(x+1)^1000")
    parse("1000!")
    parse("x^-5")


def test_long_input_refused():
    refuse("x+" * 400 + "x", ml.E_TOO_LARGE_INPUT)
    refuse("1" * 101, ml.E_NUMBER_TOO_LARGE)


def test_deep_nesting_refused():
    refuse("(" * 100 + "x" + ")" * 100, ml.E_TOO_LARGE_INPUT)
    refuse("-" * 200 + "x", ml.E_TOO_LARGE_INPUT)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("", ml.E_SYNTAX),
        ("   ", ml.E_SYNTAX),
        ("integrate", ml.E_SYNTAX),
        ("d/dx", ml.E_SYNTAX),
        ("sin", ml.E_SYNTAX),
        ("2 3", ml.E_SYNTAX),
        ("(x+1", ml.E_SYNTAX),
        ("x +", ml.E_SYNTAX),
        ("sum 1/k", ml.E_SYNTAX),
        ("limit sin(x)/x", ml.E_SYNTAX),
        ("solve x = 1 = 2", ml.E_SYNTAX),
        ("series exp(x) to order 99", ml.E_NUMBER_TOO_LARGE),
        ("diff(x, x, 50)", ml.E_NUMBER_TOO_LARGE),
        ("numeric pi to 5000 digits", ml.E_NUMBER_TOO_LARGE),
        ("integrate x dy from 0 to 1 dx", ml.E_SYNTAX),
    ],
)
def test_syntax_refusals(text, code):
    refuse(text, code)


def test_every_error_has_stable_code():
    codes = {
        ml.E_SYNTAX,
        ml.E_TOO_LARGE_INPUT,
        ml.E_UNKNOWN_NAME,
        ml.E_NUMBER_TOO_LARGE,
        ml.E_NEED_VARIABLE,
        ml.E_TIMEOUT,
        ml.E_MEMORY,
        ml.E_UNSUPPORTED,
        ml.E_INTERNAL,
        ml.E_BUSY,
    }
    assert len(codes) == 10
    assert all(c.startswith("E-PLAYGROUND-0") for c in codes)
