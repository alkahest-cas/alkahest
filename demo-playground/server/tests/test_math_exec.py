"""Executor tests: the Math tab's examples against a real Alkahest build."""

from __future__ import annotations

import pytest

ak = pytest.importorskip("alkahest")

import math_exec  # noqa: E402
import mathlang as ml  # noqa: E402


def run(text, definitions=None):
    return math_exec.execute({"input": text, "definitions": definitions or [], "wall_ms": 8000})


def ok(text, definitions=None):
    res = run(text, definitions)
    assert res["ok"], res
    assert res["latex"]
    assert res["interpreted_latex"]
    return res


def same(text_a, text_b):
    """Two result texts denote the same expression (checked by Alkahest)."""
    pool = ak.ExprPool()
    a = ak.parse(text_a, pool)
    b = ak.parse(text_b, pool)
    return ak.is_zero(ak.simplify(a - b).value) is True


# ── the documented examples ──────────────────────────────────────────────────


def test_integrate_sin_squared():
    res = ok("integrate sin(x)^2 dx")
    assert same(res["text"], "x/2 - sin(2*x)/4")
    assert res["verified"] is True


def test_integrate_x_exp_x_unicode():
    res = ok("∫ x e^x dx")
    assert same(res["text"], "exp(x)*(x - 1)")


def test_integrate_improper_answers_or_declines_with_code():
    res = run("integrate 1/(1+x^2) from 0 to oo")
    if res["ok"]:
        assert same(res["text"], "pi/2")
    else:
        assert res["error_code"].startswith("E-INT-")
        assert res["error_source"] == "alkahest"
        assert r"\int_{0}^{\infty}" in res["interpreted_latex"]


def test_definite_integral_with_certificate():
    res = ok("integrate x^2 from 0 to 1")
    assert res["text"] == "1/3"
    assert "certificate" in res
    assert res["certificate"].startswith("import Mathlib")


def test_derivative_product():
    res = ok("d/dx x^3 sin(x)")
    assert same(res["text"], "3*x^2*sin(x) + x^3*cos(x)")
    assert res.get("certificate")


def test_derivative_of_tan():
    res = ok("derivative of tan(x)")
    assert same(res["text"], "1 + tan(x)^2")


@pytest.mark.parametrize(
    "text", ["diff(x^3 sin(x), x, 2)", "d²/dx² x^3 sin(x)", "d^2/dx^2 x^3 sin(x)"]
)
def test_second_derivative(text):
    res = ok(text)
    assert same(res["text"], "6*x*sin(x) + 6*x^2*cos(x) - x^3*sin(x)")


def test_solve_quadratic():
    res = ok("solve x^2 - 5x + 6 = 0")
    assert res["text"] == "x = 2; x = 3"
    assert r"\text{or}" in res["latex"]


def test_solve_system():
    res = ok("solve x^2 + y^2 = 1, x - y = 0 for x, y")
    assert res["text"].count("x =") == 2
    assert "sqrt(2)" in res["text"]
    assert r"\begin{aligned}" in res["latex"]


def test_simplify_cancels():
    res = ok("simplify (x^2-1)/(x-1)")
    assert same(res["text"], "x + 1")


def test_simplify_trig_identity():
    res = ok("simplify sin(x)^2 + cos(x)^2")
    assert res["text"] == "1"


def test_factor_polynomial():
    res = ok("factor x^4 - 1")
    assert "(x + 1)" in res["text"]
    assert "(1 + x^2)" in res["text"]
    assert res["verified"] is True


def test_factor_multivariate_and_integer():
    assert ok("factor x^2 - y^2")["latex"].count(r"\left(") == 2
    assert ok("factor 360")["text"] == "2^3 * 3^2 * 5"


def test_expand():
    res = ok("expand (x+1)^5")
    assert same(res["text"], "x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1")


def test_limit():
    res = ok("limit sin(x)/x as x -> 0")
    assert res["text"] == "1"


def test_one_sided_limit():
    res = ok("limit 1/x as x -> 0+")
    assert res["latex"] == r"\infty"


def test_series():
    res = ok("series exp(x) at x = 0 to order 6")
    assert "O(x^6)" in res["text"]
    assert r"\mathcal{O}" in res["latex"]


def test_sum_basel():
    res = ok("sum 1/k^2 for k = 1 to oo")
    assert same(res["text"], "pi^2/6")


def test_bare_expression_with_approx():
    res = ok("sin(pi/6) + sqrt(8)")
    assert same(res["text"], "1/2 + 2*sqrt(2)")
    assert res["approx"].startswith("3.328427124746")


def test_numeric_50_digits():
    res = ok("numeric pi to 50 digits")
    assert res["text"] == "3.1415926535897932384626433832795028841971693993751"
    assert res["verification"] == "rigorous_enclosure"


def test_numeric_bracket_form():
    res = ok("N[sqrt(2)]")
    assert res["text"].startswith("1.41421356237")


# ── definitions carried between cells ────────────────────────────────────────


def test_definition_then_derivative():
    d = ok("f = x^2 sin(x)")
    assert d["definition"]["name"] == "f"
    defs = [d["definition"]]
    res = ok("d/dx f", defs)
    assert same(res["text"], "2*x*sin(x) + x^2*cos(x)")
    res = ok("d²/dx² f", defs)
    assert same(res["text"], "2*sin(x) + 4*x*cos(x) - x^2*sin(x)")


def test_definition_from_a_command():
    d = ok("F = integrate x^2 dx")
    assert same(d["definition"]["value"], "x^3/3")
    res = ok("d/dx F", [d["definition"]])
    assert same(res["text"], "x^2")


def test_function_definition_and_call():
    d = ok("g(t) = t^2 + 1")
    defs = [d["definition"]]
    assert ok("g(3)", defs)["text"] == "10"
    assert same(ok("g(x) + g(2)", defs)["text"], "x^2 + 6")
    assert same(ok("d/dt g", defs)["text"], "2*t")


def test_bound_variable_shadows_definition():
    d = ok("x = 5")
    res = ok("d/dx x^2", [d["definition"]])
    assert same(res["text"], "2*x")


# ── refusals ─────────────────────────────────────────────────────────────────


def test_non_elementary_integral_declined_with_code():
    res = run("integrate e^(x^2) dx")
    assert not res["ok"]
    assert res["error_code"] == "E-INT-004"
    assert res["error_source"] == "alkahest"
    assert res["error_message"]
    assert res["interpreted_latex"].startswith(r"\int")


def test_divergent_sum_declined_with_code():
    res = run("sum 1/k for k = 1 to oo")
    assert res["error_code"] == "E-SUM-003"
    assert "diverges" in res["error_message"]


def test_numeric_with_free_symbol():
    res = run("numeric x + 1")
    assert res["error_code"] == ml.E_UNSUPPORTED


def test_ambiguous_variable():
    res = run("integrate x y + z")
    # x is preferred when present
    assert res["ok"]
    res = run("integrate a b")
    assert res["error_code"] == ml.E_NEED_VARIABLE


def test_underdetermined_solve():
    res = run("solve a x + b = 0")
    assert res["error_code"] == ml.E_NEED_VARIABLE
    assert ok("solve a x + b = 0 for x")["text"]


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('id')",
        "x.__class__",
        "2^2^2^2^2^2",
        "x" * 600,
    ],
)
def test_malicious_input_never_reaches_alkahest(text):
    res = run(text)
    assert not res["ok"]
    assert res["error_code"].startswith("E-PLAYGROUND-")
    assert res["error_source"] == "input"


def test_malicious_definition_refused():
    res = run("f + 1", [{"name": "f", "params": [], "value": "__import__('os')"}])
    assert not res["ok"]
    assert res["error_code"].startswith("E-PLAYGROUND-")


def test_execute_never_raises_on_garbage():
    for req in (
        {"input": 3},
        {"input": "x", "definitions": "nope"},
        {"input": "x", "definitions": [None]},
        {},
    ):
        res = math_exec.execute(req)
        assert res["ok"] is False
        assert res["error_code"].startswith("E-PLAYGROUND-")
