"""``symbolic_grad`` (reverse mode) must be right or refuse — never a silent 0.

Through 3.12.0 the reverse-mode walker only knew sin/cos/tan/exp/log/sqrt/atan
and constant exponents; every other node contributed nothing, so
``symbolic_grad(erf(x), [x])`` was ``[0]``, ``symbolic_grad(x**y, [x, y])`` was
``[0, 0]`` and a Black–Scholes price had vega = 0.  Every partial below is
checked against a central finite difference at random points inside the
function's domain.
"""

from __future__ import annotations

import math
import random

import alkahest as ak
import pytest

H = 1e-6


def _fd(expr, binds, var):
    up = dict(binds)
    dn = dict(binds)
    up[var] = binds[var] + H
    dn[var] = binds[var] - H
    return (ak.eval_expr(expr, up) - ak.eval_expr(expr, dn)) / (2 * H)


def _close(got, want, what):
    tol = 1e-5 * max(1.0, abs(want))
    assert abs(got - want) < tol, f"{what}: symbolic_grad gave {got}, finite difference {want}"


# (constructor, low, high): sample points strictly inside the real domain.
UNARY = [
    ("erf", -1.5, 1.5),
    ("erfc", -1.5, 1.5),
    ("gamma", 0.3, 3.0),
    ("digamma", 0.3, 3.0),
    ("lambert_w", 0.1, 3.0),
    ("asin", -0.8, 0.8),
    ("acos", -0.8, 0.8),
    ("sinh", -2.0, 2.0),
    ("cosh", -2.0, 2.0),
    ("tanh", -2.0, 2.0),
    ("asinh", -2.0, 2.0),
    ("atanh", -0.8, 0.8),
    ("bessel_j0", 0.1, 5.0),
    ("bessel_j1", 0.1, 5.0),
    ("sin_integral", 0.1, 3.0),
    ("exp_integral_ei", 0.1, 3.0),
    ("dilog", -0.9, 0.9),
    ("fresnels", 0.1, 2.0),
    ("fresnelc", 0.1, 2.0),
    ("elliptic_k", 0.05, 0.9),
]


@pytest.mark.parametrize(("name", "lo", "hi"), UNARY, ids=[u[0] for u in UNARY])
def test_unary_function_gradient_matches_finite_differences(name, lo, hi):
    p = ak.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    f = getattr(ak, name)
    # Chain rule through a scaled argument, times a second variable.
    e = y * f(x * p.rational(9, 10))
    rng = random.Random(hash(name) & 0xFFFF)
    gx, gy = ak.symbolic_grad(e, [x, y])
    for _ in range(4):
        binds = {x: rng.uniform(lo, hi), y: rng.uniform(0.5, 2.0)}
        _close(ak.eval_expr(gx, binds), _fd(e, binds, x), f"d/dx y*{name}(x)")
        _close(ak.eval_expr(gy, binds), _fd(e, binds, y), f"d/dy y*{name}(x)")


def test_repro_from_report():
    p = ak.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    zero = p.integer(0)
    gx, gy = ak.symbolic_grad(x**y, [x, y])
    assert gx != zero
    assert gy != zero
    binds = {x: 1.7, y: 0.6}
    _close(ak.eval_expr(gx, binds), 0.6 * 1.7 ** (0.6 - 1), "d/dx x^y")
    _close(ak.eval_expr(gy, binds), 1.7**0.6 * math.log(1.7), "d/dy x^y")
    (g,) = ak.symbolic_grad(ak.lambert_w(x), [x])
    assert g != zero
    (g,) = ak.symbolic_grad(ak.erf(x), [x])
    _close(ak.eval_expr(g, {x: 0.3}), 2 / math.sqrt(math.pi) * math.exp(-0.09), "erf'")


def test_powers_and_atan2():
    p = ak.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    a = p.symbol("a")
    two = p.integer(2)
    rng = random.Random(7)
    exprs = {
        "x**y": x**y,
        "x**a": x**a,
        "2**x": two**x,
        "x**x": x**x,
        "(x+y)**(x*y)": (x + y) ** (x * y),
        "atan2(y, x)": ak.atan2(y, x),
        "atan2(x*y, x+y)": ak.atan2(x * y, x + y),
    }
    for label, e in exprs.items():
        gs = ak.symbolic_grad(e, [x, y])
        for _ in range(4):
            binds = {x: rng.uniform(0.3, 2.0), y: rng.uniform(0.3, 2.0), a: rng.uniform(-2, 2)}
            for g, v in zip(gs, (x, y)):
                _close(ak.eval_expr(g, binds), _fd(e, binds, v), f"d/d{v} {label}")


def _black_scholes(p):
    S, K, r, sig, T = (p.symbol(n) for n in ("S", "K", "r", "sigma", "T"))
    half = p.rational(1, 2)
    sqrt2 = ak.sqrt(p.integer(2))

    def Phi(z):
        return half * (1 + ak.erf(z / sqrt2))

    d1 = (ak.log(S / K) + (r + half * sig**2) * T) / (sig * ak.sqrt(T))
    d2 = d1 - sig * ak.sqrt(T)
    C = S * Phi(d1) - K * ak.exp(-r * T) * Phi(d2)
    return C, (S, K, r, sig, T)


def test_black_scholes_greeks():
    p = ak.ExprPool()
    C, (S, K, r, sig, T) = _black_scholes(p)
    vega, dC_dT, delta = ak.symbolic_grad(C, [sig, T, S])
    sv, kv, rv, sgv, tv = 100.0, 95.0, 0.05, 0.2, 0.5
    binds = {S: sv, K: kv, r: rv, sig: sgv, T: tv}
    d1 = (math.log(sv / kv) + (rv + 0.5 * sgv**2) * tv) / (sgv * math.sqrt(tv))
    d2 = d1 - sgv * math.sqrt(tv)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)

    def cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    _close(ak.eval_expr(vega, binds), sv * pdf * math.sqrt(tv), "vega")
    # theta = -dC/dT
    theta = -(sv * pdf * sgv / (2 * math.sqrt(tv)) + rv * kv * math.exp(-rv * tv) * cdf(d2))
    _close(-ak.eval_expr(dC_dT, binds), theta, "theta")
    _close(ak.eval_expr(delta, binds), cdf(d1), "delta")


def test_piecewise_gradient_matches_diff():
    p = ak.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    pw = ak.piecewise([(p.gt(x, p.integer(0)), x**2)], -x)
    (g,) = ak.symbolic_grad(pw, [x])
    assert g != p.integer(0)
    assert g == ak.diff(pw, x).value
    gx, gy = ak.symbolic_grad(y * pw, [x, y])
    assert gx != p.integer(0)
    assert gy == pw


def test_undifferentiable_function_raises_instead_of_zero():
    p = ak.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    with pytest.raises(ak.DiffError) as info:
        ak.symbolic_grad(ak.floor(x), [x])
    assert info.value.code == "E-DIFF-001"
    with pytest.raises(ak.DiffError):
        ak.symbolic_grad(y * ak.trigamma(x), [x, y])
    # A non-differentiable factor that does not depend on the variable is fine.
    (g,) = ak.symbolic_grad(x * ak.floor(y), [x])
    assert g == ak.floor(y)


def test_diff_symbolic_exponent_no_longer_refuses():
    # G3: ``diff(x**y, x)`` used to raise E-DIFF-002 pointing at diff_forward,
    # which refused too (E-DIFF-004).
    p = ak.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    binds = {x: 1.3, y: 2.2}
    for e in (x**y, x**x, p.integer(3) ** (x * y)):
        for v in (x, y):
            _close(ak.eval_expr(ak.diff(e, v).value, binds), _fd(e, binds, v), f"diff {e}")
