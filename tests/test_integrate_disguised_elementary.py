"""No false non-elementarity certificates for elementary integrands in disguise.

`∫ dy/exp(log y)` used to come back as `E-INT-004` ("no elementary
antiderivative exists"): the Risch tower took `exp(log y)` — which *is* `y` —
for a transcendental monomial.  A false certificate is worse than a refusal: it
tells a search loop that a branch is permanently closed.

These cases are built adversarially.  Each integrand is the derivative of a
known elementary function `F`, written with exp/log identities
(`exp(log u) = u`, `exp(k·log u) = uᵏ`, `log(exp u) = u`, `√u = exp(log(u)/2)`)
in the integrand, in `F`, or both.  Every one must come back either with a
verified antiderivative or with a *non-certifying* refusal — never `E-INT-004`.
"""

import itertools
import math

import alkahest as ak
import pytest

x = ak.symbol("x")

# Disguises of `u` (as functions of an expression string).
_DISGUISES = [
    lambda u: f"exp(log({u}))",
    lambda u: f"exp(2*log({u}))/({u})",
    lambda u: f"exp(log({u})/2)^2",
    lambda u: f"exp(-log({u}))^(-1)",
    lambda u: f"log(exp({u}))",
]

# Elementary antiderivatives F, as strings in `u` = an inner function of x.
_INNER = ["x", "x + 1", "x^2 + 1", "exp(x)", "2*x + 3"]

_DIRECT = [
    "1/exp(log(x))",  # the reported case: the answer is log(x)
    "exp(-log(x))",
    "exp(-2*log(x))",
    "exp(log(x)/2)",
    "exp(-log(x)/2)",
    "exp(3*log(x))",
    "exp(x + log(x))",
    "exp(x - 2*log(x))*(x - 1)",
    "exp(x)*exp(-log(x))*(x - 1)/x",
    "1/exp(log(x + 1))",
    "x/exp(log(x^2 + 1))",
    "log(exp(x))",
    "log(exp(x^2))*x",
    "exp(log(x))*log(x)",
    "1/(exp(log(x))*log(x))",
    "exp(log(x)*2)/exp(log(x)*3)",
    "exp(2*log(x) + x)",
]


def _value(e, p):
    try:
        v = complex(ak.eval_expr(e, {x: p}))
    except Exception:
        return None
    return v if math.isfinite(v.real) and math.isfinite(v.imag) else None


def _outcome(f):
    """'ok' (verified), or the error code."""
    try:
        r = ak.integrate(f, x)
    except ak.IntegrationError as exc:
        return exc.code, None
    dF = ak.diff(r.value, x)
    dF = dF.value if hasattr(dF, "value") else dF
    checked = 0
    for p in (0.31, 0.47, 0.73, 1.37, 2.2):
        fv = _value(f, p)
        if fv is None:
            continue
        dv = _value(dF, p)
        assert dv is not None, f"∫ {f}: F' undefined at {p}; F = {r.value}"
        assert abs(dv - fv) < 1e-7 * max(1.0, abs(fv)), f"∫ {f}: F'({p}) = {dv} ≠ {fv}"
        checked += 1
    assert checked >= 2, f"∫ {f}: too few sample points"
    return "ok", r.value


def _generated():
    """Derivatives of disguised elementary functions: d/dx[D(u)], d/dx[log D(u)],
    d/dx[D(u)²], d/dx[1/D(u)] for every disguise D and inner function u."""
    out = []
    for disguise, inner in itertools.product(_DISGUISES, _INNER):
        d = disguise(inner)
        for shape in (f"{d}", f"log({d})", f"({d})^2", f"1/({d})"):
            out.append(shape)
    return out


@pytest.mark.parametrize("src", _DIRECT)
def test_disguised_integrand_is_never_certified(src):
    code, _ = _outcome(ak.parse(src))
    assert code != "E-INT-004", f"∫ {src} dx was certified non-elementary, but it is elementary"


@pytest.mark.parametrize("src", _generated())
def test_derivative_of_disguised_elementary_function_is_never_certified(src):
    F = ak.parse(src)
    f = ak.diff(F, x)
    f = f.value if hasattr(f, "value") else f
    code, _ = _outcome(f)
    assert code != "E-INT-004", f"∫ d/dx[{src}] dx was certified non-elementary"


def test_the_reported_case_is_answered():
    code, F = _outcome(ak.parse("1/exp(log(x))"))
    assert code == "ok"
    assert "log" in str(F)


def test_a_genuine_certificate_survives():
    # `exp(x³)·exp(log x)/x = exp(x³)` stays non-elementary once normalised.
    with pytest.raises(ak.IntegrationError) as info:
        ak.integrate(ak.parse("exp(x^3)*exp(log(x))/x"), x)
    assert info.value.code == "E-INT-004"
