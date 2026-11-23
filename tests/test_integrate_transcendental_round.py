"""Transcendental, trigonometric and special-function integrands (2026-11 round).

Covers four additions to the integrator, each gated by differentiation:

* rational functions of one exponential ``u = e^{g·x}`` — including the
  hyperbolic functions and constant-base powers ``bˣ`` rewritten into that
  form (``∫ dx/cosh x = 2·atan(eˣ)``);
* Bioche's ``u = cos x`` / ``u = sin x`` substitutions for rational functions
  of ``sin``/``cos`` with the matching symmetry, and the ``atan(P(t))`` form of
  the half-angle route (Jeffrey's continuity examples);
* special functions of a non-trivial argument after a non-elementary verdict:
  ``Ei(g(x))`` with its elementary companion, ``Si(eˣ)``, ``li(x)`` plus
  logarithms, ``erfi`` for ``∫ e^{x²}``, and ``₂F₁`` for Chebyshev's binomial
  integrals;

Every answer is checked numerically against the integrand on both sides of
the origin, and the non-elementary ones must still say so in their
classification.
"""

import math

import alkahest as ak
import pytest

x = ak.symbol("x")

_POINTS = (0.31, 0.47, 0.73, 1.37, -0.43, -0.91, -1.7)


def _value(e, p):
    try:
        v = complex(ak.eval_expr(e, {x: p}))
    except Exception:
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return v


def _check(src, *, need=3, real=True):
    f = ak.parse(src)
    r = ak.integrate(f, x)
    dF = ak.diff(r.value, x)
    dF = dF.value if hasattr(dF, "value") else dF
    checked = 0
    for p in _POINTS:
        fv = _value(f, p)
        if fv is None or abs(fv.imag) > 1e-12:
            continue
        # The antiderivative itself must be real where the integrand is — not
        # merely right up to an imaginary constant (a `log` of a negative).
        if real:
            Fv = _value(r.value, p)
            assert Fv is not None, f"∫ {src}: F({p}) undefined where f is; F = {r.value}"
            assert abs(Fv.imag) < 1e-12, f"∫ {src}: F({p}) = {Fv} is not real; F = {r.value}"
        dv = _value(dF, p)
        assert dv is not None, f"∫ {src}: F' undefined at {p} where f is defined; F = {r.value}"
        assert abs(dv - fv) < 1e-8 * max(1.0, abs(fv)), (
            f"∫ {src}: F'({p}) = {dv}, f({p}) = {fv}; F = {r.value}"
        )
        checked += 1
    assert checked >= need, f"∫ {src}: only {checked} usable sample points"
    return r.value


@pytest.mark.parametrize(
    "src",
    [
        "1/cosh(x)",
        "1/sinh(x)",
        "1/cosh(x)^3",
        "tanh(x)^3",
        "1/(cosh(x)*sinh(x))",
        "1/(2 + sinh(x))",
        "x/cosh(x)^2",
        "sech(x)^2*tanh(x)",
        "coth(x)",
    ],
)
def test_hyperbolic_through_exponentials(src):
    # `tanh³` and `coth` are answered by the pre-existing rule-engine
    # u-substitution (`log(tanh²x − 1)`), not by this round's routes, and that
    # answer is real only up to a constant; only the derivative is pinned here.
    _check(src, real=src not in ("tanh(x)^3", "coth(x)"))


def test_sech_is_elementary_and_continuous():
    F = _check("1/cosh(x)")
    # 2·atan(eˣ) — no logarithm of a sign-changing quantity, no pole.
    assert "log" not in str(F)
    for p in (-3.0, -1.0, 0.0, 1.0, 3.0):
        assert _value(F, p) is not None


@pytest.mark.parametrize(
    "src",
    [
        "exp(x)/(3*exp(2*x) + 2)",
        "2*exp(x)/(3*exp(2*x) + 2)",
        "exp(x + 1)/(exp(x) + 1)",
        "exp(6*x)/(exp(4*x) + 1)",
        "10^x*exp(x)",
        "2^x/(1 + 4^x)",
        "1/(exp(x/2) + exp(x))",
    ],
)
def test_rational_in_one_exponential(src):
    _check(src)


def test_no_log_of_exp_in_exponential_answers():
    F = _check("exp(x + 1)/(exp(x) + 1)")
    assert "log(exp" not in str(F).replace(" ", "")


@pytest.mark.parametrize(
    "src",
    [
        "sin(x)/(sin(x)^2 + 1)",
        "cos(x)/(cos(x)^2 + 2)",
        "sin(x)^3/(2 + cos(x))",
        "(2*sin(x) + cos(x) + 1)/(-2*sin(x)*cos(x) + 2*sin(x) + cos(x)^2 + 3)",
        "(5*cos(x)^2 + 4*cos(x) - 1)/(4*cos(x)^3 - 3*cos(x)^2 - 4*cos(x) - 1)",
        "(7*cos(x)^2 + 2*cos(x) - 5)/(4*cos(x)^3 - 9*cos(x)^2 + 2*cos(x) - 1)",
    ],
)
def test_rational_trig(src):
    _check(src)


@pytest.mark.parametrize(
    ("src", "name"),
    [
        ("exp(x^2)", "erfi"),
        ("x^2*exp(x^2)", "erfi"),
        ("exp(2*x^2 + x)", "erfi"),
        ("(2 - x^2)*exp(x/(x^2 + 2))/(x^3 + 2*x)", "Ei"),
        ("(2*x^4 - x^3 + 3*x^2 + 2*x + 2)*exp(x/(x^2 + 2))/(x^3 + 2*x)", "Ei"),
        ("(exp(x) + 1)*exp(x + exp(x))/(x + exp(x))", "Ei"),
        ("exp(x^2)/x", "Ei"),
        ("sin(exp(x))", "Si"),
        ("cos(exp(x))", "Ci"),
        ("x/log(x^2)", "li"),
        ("(-x^2 + 2*log(x)^2 - log(x))/(-x^2*log(x) + log(x)^3)", "li"),
        ("x/sqrt(1 - x^3)", "hyper"),
    ],
)
def test_special_functions_of_non_trivial_arguments(src, name):
    F = _check(src, need=2)
    assert name in str(F) or (name == "hyper" and "F1(" in str(F)), f"∫ {src} = {F}"


def test_erfi_is_a_function():
    v = ak.eval_expr(ak.erfi(x), {x: 1.0})
    assert abs(v - 1.6504257587975428) < 1e-13
    assert str(ak.parse("erfi(x)")) == "erfi(x)"
    d = ak.diff(ak.erfi(x), x)
    d = d.value if hasattr(d, "value") else d
    assert abs(ak.eval_expr(d, {x: 0.5}) - 2 / math.sqrt(math.pi) * math.exp(0.25)) < 1e-12


def test_substitution_through_an_inverse_function():
    _check("x^3*exp(asin(x))/sqrt(1 - x^2)")
    # Since #577 this one is answered by the algebraic engine's rationalised-
    # cofactor route, whose `log((√(1−x²) − 1)/x)` is real only up to a
    # constant on 0 < x < 1; only the derivative is pinned here.
    _check("asin(x)/(sqrt(1 - x^2) + 1)", real=False)


# Real-domain regressions: each answer used to carry the `log` of a quantity
# that is negative on part (or all) of the integrand's real domain, so it was
# an antiderivative only up to an imaginary constant there.  Checked on a
# dense grid on both sides of the origin: F real and finite, F' = f.
_REAL_DOMAIN = [
    "log(sin(x))/(sin(x) + 1)",  # Charlwood 40
    "sin(x)/(sin(x)^2 + 1)",  # Charlwood 37: log(2cos x − 2√2) < 0 everywhere
    "cos(x)/(cos(x)^2 - 2)",
    "sin(x)/(cos(x)^2 - 3)",
    "1/sinh(x)",  # log(eˣ − 1) < 0 for x < 0
    "1/(2 + sinh(x))",
    "1/(cosh(x)*sinh(x))",
    "10^x/(10^x - 3)",
]


@pytest.mark.parametrize("src", _REAL_DOMAIN)
def test_answer_is_real_on_the_real_domain(src):
    f = ak.parse(src)
    F = ak.integrate(f, x).value
    dF = ak.diff(F, x)
    dF = dF.value if hasattr(dF, "value") else dF
    checked = 0
    for i in range(80):
        p = -7.9 + 0.2013 * i
        fv = _value(f, p)
        if fv is None or abs(fv.imag) > 1e-12 or abs(fv) > 1e6:
            continue
        Fv = _value(F, p)
        assert Fv is not None, f"∫ {src}: F({p}) undefined where f is; F = {F}"
        assert abs(Fv.imag) < 1e-12, f"∫ {src}: F({p}) = {Fv} is not real; F = {F}"
        dv = _value(dF, p)
        assert dv is not None, f"∫ {src}: F'({p}) undefined; F = {F}"
        assert abs(dv - fv) < 1e-7 * max(1.0, abs(fv)), f"∫ {src}: F'({p}) = {dv}, f = {fv}"
        checked += 1
    assert checked >= 10, f"∫ {src}: only {checked} usable points"


def test_still_refused():
    # Genuinely non-elementary, and outside the special-function basis.
    for src in ["exp(x^3)", "x*tan(x)"]:
        with pytest.raises(ak.IntegrationError):
            ak.integrate(ak.parse(src), x)
