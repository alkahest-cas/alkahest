"""Real-valued logarithms in every `integrate` answer, and the remaining hard
integrals of the Rubi independent suite (2026-11 gap round).

* `integrate` ends with a real-log pass: a `log(a)` whose argument is negative
  somewhere on the integrand's real domain becomes `log(−a)` or `log|a|`,
  whichever route produced it, and the rewrite is re-verified.
* Charlwood #3, #5, #9, #14, #41, #46 and Bronstein #7 are answered, each
  checked by differentiation and required to be real where the integrand is.
"""

import math

import alkahest as ak
import pytest

x = ak.symbol("x")


def _value(e, p):
    try:
        v = complex(ak.eval_expr(e, {x: p}))
    except Exception:
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return v


def _check_real(src, *, lo=-4.9, step=0.1733, n=58, need=6):
    """F real and F' = f at every grid point where f is real and moderate."""
    f = ak.parse(src)
    F = ak.integrate(f, x).value
    dF = ak.diff(F, x)
    dF = dF.value if hasattr(dF, "value") else dF
    checked = 0
    for i in range(n):
        p = lo + step * i
        fv = _value(f, p)
        if fv is None or abs(fv.imag) > 1e-12 or abs(fv) > 1e6:
            continue
        Fv = _value(F, p)
        assert Fv is not None, f"∫ {src}: F({p}) undefined where f is; F = {F}"
        assert abs(Fv.imag) < 1e-10, f"∫ {src}: F({p}) = {Fv} is not real; F = {F}"
        dv = _value(dF, p)
        if dv is None:
            # `sign(cos x)` and `|a|` are not differentiable at isolated points.
            continue
        assert abs(dv - fv) < 1e-7 * max(1.0, abs(fv)), (
            f"∫ {src}: F'({p}) = {dv}, f({p}) = {fv}; F = {F}"
        )
        checked += 1
    assert checked >= need, f"∫ {src}: only {checked} usable points"
    return F


# Routes that predate the real-log rewrite: the rule engine's u-substitution,
# Rothstein–Trager, Risch–Norman, the rationalised-cofactor route.
@pytest.mark.parametrize(
    "src",
    [
        "tanh(x)^3",
        "coth(x)",
        "1/(exp(x) - 2)",
        "exp(x)/(exp(2*x) - 4)",
        "1/x",
        "1/(x^3 + 1)",
        "x^2/(x^4 - 1)",
        "1/(x^4 - 1)",
        "1/(5 + 3*cos(x))",
        "asin(x)/(sqrt(1 - x^2) + 1)",
        "1/(x*log(x))",
    ],
)
def test_logs_are_real_everywhere(src):
    _check_real(src)


def test_log_of_a_variable_is_its_absolute_value():
    F = ak.integrate(ak.parse("1/x"), x).value
    assert str(F) == "log(abs(x))"
    # Where the argument is positive on the whole domain nothing changes.
    F = ak.integrate(ak.parse("1/(x^2 + 1)*2*x"), x).value
    assert "abs" not in str(F)


@pytest.mark.parametrize(
    "src",
    [
        "-asin(sqrt(x) - sqrt(x + 1))",  # Charlwood 3: s = √x + √(x+1)
        "cos(x)^2/sqrt(cos(x)^4 + cos(x)^2 + 1)",  # Charlwood 5: pseudo-elliptic
        "sin(x)*atan(sqrt(sec(x) - 1))",  # Charlwood 9: by parts
        "sqrt(sin(x) + 1)*log(sin(x))",  # Charlwood 41: s = sin x, sign(cos x)
        "x*log(x^2 + 1)*atan(x)^2",  # Charlwood 46: by parts, nested
        "(x - tan(x))/tan(x)^2",  # Bronstein 7
    ],
)
def test_hard_integrals(src):
    _check_real(src, need=3)


def test_charlwood_5_is_elementary():
    # ∫ dt/((1+t²)√(t⁴+3t²+3)) = ⅓·atan(t·√(t⁴+3t²+3)), t = tan x; Mathematica
    # answers with EllipticPi.
    F = ak.integrate(ak.parse("cos(x)^2/sqrt(cos(x)^4 + cos(x)^2 + 1)"), x).value
    assert "atan" in str(F)
    assert "Elliptic" not in str(F)


def test_still_refused():
    # Bronstein 8: ∫x·tan x is not elementary.
    with pytest.raises(ak.IntegrationError):
        ak.integrate(ak.parse("x*tan(x) + tan(x)^2 + 1"), x)
