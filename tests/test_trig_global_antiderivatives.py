"""
Antiderivatives and ODE solutions that are valid on all of ℝ, not just on (−π, π).

The Weierstrass half-angle substitution `t = tan(x/2)` used to leave
`atan(a·tan(x/2) + b)` in its antiderivatives, which jumps by ±π at every odd
multiple of π.  Two consequences, both silent:

* `dsolve(y'' + y = cos³t)` returned a particular solution containing
  `sin t·atan(tan(t/2))` — a sawtooth, with `y'` jumping from −1.178 to +1.178
  at t = π — because variation of parameters handed the integrator
  `cos⁴t/(cos²t + sin²t)` and the half-angle route took it.
* Definite integrals over an interval containing an odd multiple of π had to be
  refused (`∫₀^{2π} dx/(2 + cos x)`).

The integrator now rewrites each such `atan` into its continuous form, integrates
trig polynomials over a Pythagorean denominator by power reduction instead, and
`dsolve` refuses a candidate that jumps where the equation is regular.

Oracles: residuals are evaluated from the returned solution on a grid over
[−20, 20]; continuity is checked by finite differences of y and y′; definite
integrals are compared with composite Gauss–Legendre quadrature (pure Python, so
the oracle runs in CI without mpmath).
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from alkahest import experimental as ex

GRID = [-20.0 + 0.37 * i for i in range(109)]  # ≈ [−20, 20], avoids lattice points


def _gauss_legendre(n):
    """Nodes and weights of the n-point Gauss–Legendre rule on [−1, 1]."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, z
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
            dp = n * (z * p1 - p0) / (z * z - 1)
            step = p1 / dp
            z -= step
            if abs(step) < 1e-16:
                break
        nodes.append(z)
        weights.append(2 / ((1 - z * z) * dp * dp))
    return nodes, weights


_GL = _gauss_legendre(10)


def _quad(f, a, b, pieces=600):
    """Composite 10-point Gauss–Legendre: ~1e-14 for the smooth integrands here."""
    total = 0.0
    h = (b - a) / pieces
    for j in range(pieces):
        mid, half = a + (j + 0.5) * h, 0.5 * h
        total += half * sum(w * f(mid + half * z) for z, w in zip(*_GL))
    return total


def _ev(expr, sym, v):
    try:
        return ak.eval_expr(expr, {sym: v})
    except ValueError:
        return math.nan


def _d(expr, sym):
    return ak.diff(expr, sym).value


def _scalar(order):
    pool = ak.ExprPool()
    t = pool.symbol("t")
    y = pool.symbol("y")
    derivs = [pool.symbol("y" + "p" * k) for k in range(1, order + 1)]
    return pool, t, y, derivs


def _ode_cases():
    """(label, order, builder) — builder(t, y, derivs) -> equation = 0."""
    c, s, e = ak.cos, ak.sin, ak.exp
    return [
        ("y''+y=cos^3", 2, lambda t, y, d: d[1] + y - c(t) ** 3),
        ("y''+y=sin^2 cos", 2, lambda t, y, d: d[1] + y - s(t) ** 2 * c(t)),
        ("y''+y=sin^3", 2, lambda t, y, d: d[1] + y - s(t) ** 3),
        ("y''+y=cos (resonant)", 2, lambda t, y, d: d[1] + y - c(t)),
        ("y''+4y=sin 2t (resonant)", 2, lambda t, y, d: d[1] + 4 * y - s(2 * t)),
        ("y''+9y=cos^3 (resonant)", 2, lambda t, y, d: d[1] + 9 * y - c(t) ** 3),
        ("y''+y=t sin t (resonant)", 2, lambda t, y, d: d[1] + y - t * s(t)),
        ("y''+y=sin cos^2", 2, lambda t, y, d: d[1] + y - s(t) * c(t) ** 2),
        ("y''+3y'+2y=cos^2", 2, lambda t, y, d: d[1] + 3 * d[0] + 2 * y - c(t) ** 2),
        ("y''-y=e^t (resonant)", 2, lambda t, y, d: d[1] - y - e(t)),
        ("y''+2y'+y=e^-t (double)", 2, lambda t, y, d: d[1] + 2 * d[0] + y - e(-t)),
        ("y''+y'=t^2", 2, lambda t, y, d: d[1] + d[0] - t**2),
        ("y'+y=sin t", 1, lambda t, y, d: d[0] + y - s(t)),
        ("y''+4y=t cos t", 2, lambda t, y, d: d[1] + 4 * y - t * c(t)),
        ("y''+y=e^t sin t", 2, lambda t, y, d: d[1] + y - e(t) * s(t)),
        ("y'''+y'=cos (resonant)", 3, lambda t, y, d: d[2] + d[0] - c(t)),
    ]


@pytest.mark.parametrize(
    ("label", "order", "build"), _ode_cases(), ids=[c[0] for c in _ode_cases()]
)
def test_dsolve_solution_is_global(label, order, build):
    pool, t, y, derivs = _scalar(order)
    eq = build(t, y, derivs)
    try:
        sols = ex.dsolve(eq, t, y, derivs)
    except Exception as err:  # a coded refusal is acceptable; a wrong answer is not
        declined = str(err)
    else:
        declined = None
    if declined is not None:
        assert "E-ODE" in declined, f"{label}: uncoded failure {declined}"
        pytest.skip(f"{label}: declined ({declined})")
    sol = sols[0]
    assert sol["form"] == "explicit"
    consts = sol["constants"]
    vals = [pool.rational(3, 7), pool.rational(-5, 11), pool.rational(2, 3)]
    ys = ak.subs(sol["y_of_x"], {k: vals[i] for i, k in enumerate(consts)})
    assert "atan(tan" not in str(ys), f"{label}: sawtooth in {ys}"

    # y, y', …, y^(order) as expressions.
    ds = [ys]
    for _ in range(order):
        ds.append(_d(ds[-1], t))
    residual = eq
    residual = ak.subs(residual, {y: ds[0], **{derivs[k]: ds[k + 1] for k in range(order)}})

    checked = 0
    for tv in GRID:
        r = _ev(residual, t, tv)
        mags = [abs(_ev(d, t, tv)) for d in ds]
        if not math.isfinite(r):
            continue
        scale = 1.0 + sum(m for m in mags if math.isfinite(m))
        assert abs(r) < 1e-8 * scale, f"{label}: residual {r} at t={tv}\n y = {ys}"
        checked += 1
    assert checked > 90, label

    # Continuity of y … y^(order−1): increments over a small step are bounded
    # by the step times the next derivative (mean value theorem), so a jump
    # anywhere on the grid — including across every odd multiple of π — fails.
    h = 1e-3
    pts = [k * math.pi for k in (-5, -3, -1, 1, 3, 5)] + GRID
    for k in range(order):
        f, fp = ds[k], ds[k + 1]
        for tv in pts:
            a, b = _ev(f, t, tv - h), _ev(f, t, tv + h)
            m = max(abs(_ev(fp, t, tv - h)), abs(_ev(fp, t, tv)), abs(_ev(fp, t, tv + h)))
            assert abs(b - a) <= 2 * h * (1.5 * m + 1.0) + 1e-9, (
                f"{label}: derivative {k} jumps at t={tv}: {a} -> {b}\n y = {ys}"
            )


def test_cos_cubed_repro_matches_closed_form():
    """The repro: y_p = (3/8)·t·sin t − (1/32)·cos 3t + homogeneous."""
    pool, t, y, (yp, ypp) = _scalar(2)
    sol = ex.dsolve(ypp + y - ak.cos(t) ** 3, t, y, [yp, ypp])[0]
    c1, c2 = sol["constants"]
    yp0 = ak.subs(sol["y_of_x"], {c1: pool.integer(0), c2: pool.integer(0)})
    # y_p − (3/8 t sin t − 1/32 cos 3t) must lie in span{cos t, sin t}: fit at two
    # points, check at many.
    ref = pool.rational(3, 8) * t * ak.sin(t) - pool.rational(1, 32) * ak.cos(3 * t)
    diff_ = yp0 - ref
    a = _ev(diff_, t, 0.0)
    b = _ev(diff_, t, math.pi / 2)
    for tv in GRID:
        expect = a * math.cos(tv) + b * math.sin(tv)
        assert abs(_ev(diff_, t, tv) - expect) < 1e-9, tv
    # The derivative is continuous at t = π (it jumped by 2.36 before).
    d = _d(yp0, t)
    assert abs(_ev(d, t, math.pi + 1e-7) - _ev(d, t, math.pi - 1e-7)) < 1e-5


def test_simplify_keeps_atan_of_tan():
    """atan(tan(u)) = u only on (−π/2, π/2): simplify must not drop it."""
    pool = ak.ExprPool()
    u = pool.symbol("u")
    e = ak.atan(ak.tan(u))
    for s in (ak.simplify(e).value, ak.simplify(ak.atan(ak.tan(u / 2))).value):
        assert "atan" in str(s), s


def _definite_cases():
    c, s = ak.cos, ak.sin
    return [
        ("1/(2+cos)", lambda x: 1 / (2 + c(x)), 0.0, 2 * math.pi),
        ("1/(cos-3)", lambda x: 1 / (c(x) - 3), 0.0, 3.2),
        ("1/(2+sin)", lambda x: 1 / (2 + s(x)), -4.0, 9.5),
        ("1/(5+4cos)", lambda x: 1 / (5 + 4 * c(x)), -7.0, 7.0),
        ("1/(3+cos+2sin)", lambda x: 1 / (3 + c(x) + 2 * s(x)), 1.0, 12.0),
        ("cos/(3+sin+cos)", lambda x: c(x) / (3 + s(x) + c(x)), -10.0, 4.0),
        ("(1+sin)/(3+cos)", lambda x: (1 + s(x)) / (3 + c(x)), 2.0, 20.0),
        ("sin/(2+cos)", lambda x: s(x) / (2 + c(x)), -3.5, 3.5),
        ("100/(101+100sin)", lambda x: 100 / (101 + 100 * s(x)), 0.5, 7.5),
        ("4/(5+4cos)", lambda x: 4 / (5 + 4 * c(x)), -9.0, 2.0),
        ("1/(7-cos)", lambda x: 1 / (7 - c(x)), 3.0, 16.0),
        ("1/(4+3sin)", lambda x: 1 / (4 + 3 * s(x)), -2.0, 30.0),
        ("1/(2-sin+cos)", lambda x: 1 / (2 - s(x) + c(x)), -6.5, 6.5),
        ("cos/(2+cos)", lambda x: c(x) / (2 + c(x)), 0.0, 4 * math.pi),
        ("1/(3+2cos)", lambda x: 1 / (3 + 2 * c(x)), math.pi - 0.1, 3 * math.pi + 0.1),
    ]


@pytest.mark.parametrize(
    ("label", "build", "a", "b"), _definite_cases(), ids=[c[0] for c in _definite_cases()]
)
def test_definite_rational_trig_crossing_odd_pi(label, build, a, b):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = build(x)
    num = _quad(lambda v: _ev(f, x, v), a, b)
    r = ak.integrate(f, x, pool.float(a), pool.float(b))
    got = ak.eval_expr(r.value, {})
    assert abs(got - float(num)) < 1e-9 * (1 + abs(num)), f"{label}: {got} vs {num}"


def test_definite_over_full_period_exact_bounds():
    """∫₀^{2π} dx/(2 + cos x) = 2π/√3 (it was refused: the antiderivative jumped at π)."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pi = pool.symbol("pi")
    r = ak.integrate(1 / (2 + ak.cos(x)), x, pool.integer(0), 2 * pi)
    assert abs(ak.eval_expr(r.value, {}) - 2 * math.pi / math.sqrt(3)) < 1e-12


@pytest.mark.parametrize(
    ("label", "build", "a", "b"), _definite_cases(), ids=[c[0] for c in _definite_cases()]
)
def test_indefinite_rational_trig_is_continuous(label, build, a, b):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    big_f = ak.integrate(build(x), x).value
    for k in (-5, -3, -1, 1, 3, 5):
        lo = _ev(big_f, x, k * math.pi - 1e-8)
        hi = _ev(big_f, x, k * math.pi + 1e-8)
        assert abs(hi - lo) < 1e-6, f"{label}: F jumps at {k}π: {lo} -> {hi}\n F = {big_f}"
