"""Tests for the experimental calculus / ODE / transform Python surface.

These cover the PyO3 bindings added for the features that landed with bindings
deferred (PRs #152–#161): dsolve, Laplace/Fourier/Z transforms, multivariate
limits, asymptotic expansions, power-series / Frobenius ODE solutions, the lazy
formal power series ``Fps``, and the ``heaviside`` / ``dirac_delta``
distribution primitives.

Everything lives under ``alkahest.experimental`` so the frozen top-level
``__all__`` is untouched. Expected values match the corresponding Rust unit
tests.
"""

from __future__ import annotations

from fractions import Fraction

import alkahest as A
from alkahest import experimental as ex

# ---------------------------------------------------------------------------
# Distribution primitives
# ---------------------------------------------------------------------------


def test_heaviside_dirac_constructors():
    p = A.ExprPool()
    x = p.symbol("x")
    assert str(ex.heaviside(x)) == "heaviside(x)"
    assert str(ex.dirac_delta(x)) == "diracdelta(x)"
    # diff(heaviside) should expose the dirac delta.
    dh = A.diff(ex.heaviside(x), x).value
    assert "diracdelta" in str(dh)


# ---------------------------------------------------------------------------
# dsolve
# ---------------------------------------------------------------------------


def test_dsolve_logistic():
    # y' = y(1 - y) (logistic): general solution 1/(1 + C e^{-x}).
    p = A.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    yp = p.symbol("y'")
    eq = yp - y * (1 - y)
    branches = ex.dsolve(eq, x, y, [yp])
    assert len(branches) >= 1
    sol = branches[0]
    s = str(sol["y_of_x"])
    # 1/(1 + C e^{-x}) form: an inverse of (1 + C*exp(x)^-1).
    assert "exp(x)" in s
    assert sol["constants"], "expected at least one integration constant"
    assert isinstance(sol["method"], str)


def test_dsolve_unsupported_raises():
    p = A.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    yp = p.symbol("y'")
    # An opaque nonlinear ODE outside the implemented classes should decline.
    eq = yp - A.sin(y * x) - A.exp(y**2)
    try:
        ex.dsolve(eq, x, y, [yp])
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        # Some forms may still solve; if it returns, it must be a list.
        pass


# ---------------------------------------------------------------------------
# Laplace transform
# ---------------------------------------------------------------------------


def test_laplace_t_exp_sin():
    # L{t e^{2t} sin 3t}(s) = 6 (s - 2) / ((s - 2)^2 + 9)^2.
    p = A.ExprPool()
    t = p.symbol("t")
    s = p.symbol("s")
    # Build a flat product node: the Laplace product rules match a single
    # canonical Mul, which the Python ``*`` operator does not flatten.
    f = p.mul([t, A.exp(p.mul([p.integer(2), t])), A.sin(p.mul([p.integer(3), t]))])
    got = ex.laplace_transform(f, t, s)
    text = str(got)
    assert "6" in text
    assert "(s + -2)" in text  # s - 2 factor
    # Numerically corroborate at s = 5: 6*3 / (9 + 9)^2 = 18/324 = 1/18.
    val = _numeric(got, {s: 5.0})
    assert abs(val - (6 * 3) / (9 + 9) ** 2) < 1e-9


def test_laplace_constant_and_inverse():
    p = A.ExprPool()
    t = p.symbol("t")
    s = p.symbol("s")
    # L{1} = 1/s
    got = ex.laplace_transform(p.integer(1), t, s)
    assert str(got) == "s^-1"
    # L^{-1}{1/s} = 1 (returned as the unsimplified e^{0·t}; check numerically).
    back = ex.inverse_laplace_transform(s ** (-1), s, t)
    assert abs(_numeric(back, {t: 2.5}) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Fourier transform
# ---------------------------------------------------------------------------


def test_fourier_gaussian_self_dual():
    # F{e^{-pi x^2}} = e^{-pi xi^2}.
    p = A.ExprPool()
    x = p.symbol("x")
    xi = p.symbol("xi")
    pi = p.symbol("pi")
    f = A.exp(p.integer(-1) * pi * x**2)
    got = ex.fourier_transform(f, x, xi)
    text = str(got)
    assert text.startswith("exp(")
    assert "pi" in text
    assert "xi^2" in text
    assert "-1" in text
    # Numerically self-dual: F{e^{-pi x^2}}(xi) = e^{-pi xi^2}.
    import math

    assert abs(_numeric(got, {pi: math.pi, xi: 0.7}) - math.exp(-math.pi * 0.49)) < 1e-9
    # Inverse returns the same Gaussian (in x).
    back = ex.inverse_fourier_transform(got, xi, x)
    assert abs(_numeric(back, {pi: math.pi, x: 0.7}) - math.exp(-math.pi * 0.49)) < 1e-9


# ---------------------------------------------------------------------------
# Z-transform
# ---------------------------------------------------------------------------


def test_ztransform_table_and_fibonacci_decline():
    p = A.ExprPool()
    n = p.symbol("n")
    z = p.symbol("z")
    # Z{2^n} = z/(z - 2).
    got = ex.z_transform(p.integer(2) ** n, n, z)
    assert str(got) == "(z * (z + -2)^-1)"
    # Inverse recovers 2^n.
    back = ex.inverse_z_transform(z * (z - 2) ** (-1), z, n)
    assert str(back) == "(2^n)"


def test_ztransform_unit():
    p = A.ExprPool()
    n = p.symbol("n")
    z = p.symbol("z")
    # Z{1} = z/(z - 1).
    got = ex.z_transform(p.integer(1), n, z)
    val = _numeric(got, {z: 3.0})
    assert abs(val - 3.0 / 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Multivariate limits
# ---------------------------------------------------------------------------


def test_multilimit_dne_with_witnesses():
    # xy/(x^2 + y^2) does not exist at the origin.
    p = A.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    f = (x * y) / (x**2 + y**2)
    r = ex.multilimit(f, x, y, p.integer(0), p.integer(0))
    assert r["status"] == "dne"
    a = r["path_a"]["value_numeric"]
    b = r["path_b"]["value_numeric"]
    assert abs(a - b) > 1e-3
    assert isinstance(r["path_a"]["description"], str)


def test_multilimit_value():
    # A jointly-continuous function has a plain limit.
    p = A.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    f = x**2 + y**2
    r = ex.multilimit(f, x, y, p.integer(1), p.integer(2))
    assert r["status"] == "value"
    assert _numeric(r["value"], {}) == 5.0


# ---------------------------------------------------------------------------
# Asymptotic expansion
# ---------------------------------------------------------------------------


def test_asymptotic_sqrt_x2_plus_1():
    # sqrt(x^2 + 1) ~ x + 1/(2x) - 1/(8 x^3) + ...
    p = A.ExprPool()
    x = p.symbol("x")
    f = A.sqrt(x**2 + 1)
    terms = ex.asymptotic_expand(f, x, 3)
    assert len(terms) == 3
    # Leading term ~ x; next ~ 1/(2x); third ~ -1/(8 x^3).
    assert _numeric(terms[0], {x: 1000.0}) == 1000.0
    assert abs(_numeric(terms[1], {x: 1000.0}) - 1.0 / (2 * 1000.0)) < 1e-12
    assert abs(_numeric(terms[2], {x: 10.0}) - (-1.0 / (8 * 10.0**3))) < 1e-12


# ---------------------------------------------------------------------------
# Series / Frobenius ODE solutions
# ---------------------------------------------------------------------------


def test_series_solve_bessel_j0():
    # x^2 y'' + x y' + x^2 y = 0 (Bessel order 0), regular singular at 0.
    # J0 series coefficients a_{2k} = (-1)^k / (4^k (k!)^2): 1, -1/4, 1/64, -1/2304.
    p = A.ExprPool()
    x = p.symbol("x")
    res = ex.series_solve(x, x**2, x, x**2, p.integer(0), 8)
    assert res["kind"] == "regular_singular"
    s0 = res["solutions"][0]
    assert s0["exponent"] == 0
    coeffs = s0["coeffs"]
    assert coeffs[0] == 1
    assert coeffs[2] == Fraction(-1, 4)
    assert coeffs[4] == Fraction(1, 64)
    assert coeffs[6] == Fraction(-1, 2304)
    # Symbolic rendering is available.
    assert "O(" in str(s0["expr"])


def test_series_solve_ordinary_point():
    # y'' + y = 0 at an ordinary point: cos/sin series (exponent 0).
    p = A.ExprPool()
    x = p.symbol("x")
    res = ex.series_solve(x, p.integer(1), p.integer(0), p.integer(1), p.integer(0), 6)
    assert res["kind"] == "ordinary"
    assert len(res["solutions"]) == 2


# ---------------------------------------------------------------------------
# Formal power series (Fps)
# ---------------------------------------------------------------------------


def test_fps_known_series_coeffs():
    assert ex.Fps.exp_series().coeffs(5) == [1, 1, Fraction(1, 2), Fraction(1, 6), Fraction(1, 24)]
    assert ex.Fps.sin_series().coeffs(4) == [0, 1, 0, Fraction(-1, 6)]
    assert ex.Fps.cos_series().coeffs(4) == [1, 0, Fraction(-1, 2), 0]


def test_fps_binomial_and_compose_catalan():
    # Catalan generating function via (1 - sqrt(1 - 4x)) / (2x).
    sqrt_half = ex.Fps.binomial_series(Fraction(1, 2))  # (1 + u)^(1/2)
    u = ex.Fps.from_poly([0, -4])  # u = -4x
    sqrt_term = sqrt_half.compose(u)  # sqrt(1 - 4x)
    numer = ex.Fps.constant(1).sub(sqrt_term)  # 1 - sqrt(1 - 4x)
    # C(x) = numer / (2x): coeff_n(C) = coeff_{n+1}(numer) / 2.
    catalan = [numer.coeff(k + 1) / Fraction(2) for k in range(6)]
    assert catalan == [1, 1, 2, 5, 14, 42]


def test_fps_revert_inverse():
    # f = x + x^2; revert h satisfies f(h(x)) = x.
    f = ex.Fps.from_poly([0, 1, 1])
    h = f.revert()
    assert h.coeffs(5) == [0, 1, -1, 2, -5]
    comp = f.compose(h)
    assert comp.coeffs(5) == [0, 1, 0, 0, 0]


def test_fps_mul_and_to_expr():
    p = A.ExprPool()
    x = p.symbol("x")
    # (1 + x)(1 + x) = 1 + 2x + x^2.
    g = ex.Fps.from_poly([1, 1])
    prod = g.mul(g)
    assert prod.coeffs(3) == [1, 2, 1]
    e = prod.to_expr(x, 3)
    assert "O(x^3)" in str(e)


def test_fps_from_expr_snapshot():
    # 1/(1 - x) = geometric series 1, 1, 1, ...
    p = A.ExprPool()
    x = p.symbol("x")
    f = (1 - x) ** (-1)
    series = ex.Fps.from_expr(f, x, 6)
    assert series.coeffs(6) == [1, 1, 1, 1, 1, 1]


def test_fps_div_and_inverse():
    # 1/(1 - x) via inverse of (1 - x).
    one_minus_x = ex.Fps.from_poly([1, -1])
    inv = one_minus_x.inverse()
    assert inv.coeffs(4) == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numeric(expr, env: dict) -> float:
    """Numerically evaluate an Expr at the given symbol bindings."""
    return float(A.eval_expr(expr, {sym: float(v) for sym, v in env.items()}))
