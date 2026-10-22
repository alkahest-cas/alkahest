"""``series`` at the pole of zeta, and the pole-value guard (X7 / N5).

``series(zeta(s), s, 1, 2)`` used to return ``zeta(1) + (s-1)*zeta_prime(1) +
O((s-1)**2)`` -- a Taylor formula built from the pole value, although
``ak.zeta(1)`` itself raises ``E-SPECFN-006``.  The correct answer is the
Laurent series ``1/(s-1) + gamma - gamma_1 (s-1) + gamma_2/2 (s-1)**2 - ...``
with Stieltjes constants.

Oracles: ``mpmath.stieltjes`` (the coefficients) and ``mpmath.zeta`` (the
truncated series near the pole).
"""

from __future__ import annotations

import re

import alkahest as ak
import pytest

mp = pytest.importorskip("mpmath")
mp.mp.dps = 30


def _pool():
    p = ak.ExprPool()
    return p, p.symbol("s")


def _eval(e, env=None):
    return ak.eval_expr(e, env or {})


def test_zeta_laurent_at_one_matches_mpmath_stieltjes():
    p, s = _pool()
    r = ak.series(ak.zeta(s), s, p.integer(1), 6)
    txt = str(r.expr)
    assert "zeta" not in txt, txt
    assert "EulerGamma" in txt
    assert "StieltjesGamma1" in txt
    # Coefficient of (s-1)^k, read numerically by evaluating at points.
    for h in (0.01, -0.013, 0.02):
        got = _eval(r.truncated(), {s: 1.0 + h})
        want = mp.zeta(1 + h)
        # truncation error is O(h^6); the series carries 1/h.
        assert abs(got - float(want)) < 1e-9, (h, got, want)


def test_zeta_laurent_coefficients_are_stieltjes_constants():
    p, s = _pool()
    r = ak.series(ak.zeta(s), s, p.integer(1), 5)
    h = p.symbol("h")
    poly = ak.subs(r.truncated(), {s: h + 1})
    # Fit the coefficient of h^k and compare with (-1)^k gamma_k / k!.
    xs = [mp.mpf(i) / 50 for i in range(1, 9)]
    vals = [_eval(poly, {h: float(x)}) - 1.0 / float(x) for x in xs]
    A = mp.matrix([[x**k for k in range(5)] for x in xs[:5]])
    sol = mp.lu_solve(A, mp.matrix(vals[:5]))
    assert abs(sol[0] - mp.euler) < 1e-8
    assert abs(sol[1] + mp.stieltjes(1)) < 1e-6
    assert abs(sol[2] - mp.stieltjes(2) / 2) < 1e-4


def test_zeta_one_plus_s_about_zero():
    p, s = _pool()
    r = ak.series(ak.zeta(1 + s), s, p.integer(0), 3)
    assert "zeta" not in str(r.expr)
    for h in (0.01, -0.02, 0.03):
        got = _eval(r.truncated(), {s: h})
        assert abs(got - float(mp.zeta(1 + h))) < 1e-6


def test_zeta_pole_inside_a_larger_expression():
    p, s = _pool()
    # (s-1) * zeta(s) -> 1 + gamma (s-1) + ...  : removable pole
    f = (s - 1) * ak.zeta(s)
    r = ak.series(f, s, p.integer(1), 3)
    for h in (0.01, -0.02):
        got = _eval(r.truncated(), {s: 1.0 + h})
        assert abs(got - float(h * mp.zeta(1 + h))) < 1e-6


def test_stieltjes_constant_symbols():
    p = ak.ExprPool()
    assert str(ak.stieltjes_constant(p, 0)) == "EulerGamma"
    for n in (1, 2, 5, 12, 30):
        g = ak.stieltjes_constant(p, n)
        assert abs(_eval(g) - float(mp.stieltjes(n))) < 1e-12
        # ball at 200 bits
        b = ak.interval_eval(g, {}, 200)
        assert b.rad < 1e-50
        want = mp.stieltjes(n)
        assert b.contains(float(want)) or abs(float(b.mid) - float(want)) < 1e-15
    with pytest.raises(ValueError):
        ak.stieltjes_constant(p, 31)
    assert "\\gamma_{1}" in ak.latex(ak.stieltjes_constant(p, 1))


def test_hurwitz_zeta_integer_shift_at_pole():
    p, s = _pool()
    r = ak.series(ak.hurwitz_zeta(s, p.integer(3)), s, p.integer(1), 3)
    assert "hurwitz_zeta" not in str(r.expr)
    for h in (0.01, -0.02):
        got = _eval(r.truncated(), {s: 1.0 + h})
        assert abs(got - float(mp.zeta(1 + h, 3))) < 1e-6


# -- The guard: no series may contain a pole value ---------------------------

_POLE_VALUE = re.compile(
    r"bessel_[yk]\([0-9], 0\)|bessel_j\(-1/3, 0\)|zeta_prime\(1\)|zeta\(1\)"
    r"|hurwitz_zeta\(1,|loggamma\(0\)|digamma\(0\)|gamma\(0\)"
    r"|polylog\(1, 1\)|trigamma\(0\)"
)


def _no_pole_value_or_refused(make):
    p, s = _pool()
    try:
        r = ak.series(make(p, s), s, make.point(p), 2)
    except Exception as e:  # a coded refusal is acceptable
        coded = re.search(r"E-[A-Z]+-\d+", str(e))
        if not coded:
            pytest.fail(f"uncoded error: {e}")
        return
    txt = str(r.expr)
    assert not _POLE_VALUE.search(txt), txt


class _C:
    def __init__(self, f, point):
        self.f, self.point_v = f, point

    def __call__(self, p, s):
        return self.f(p, s)

    def point(self, p):
        return p.integer(self.point_v)


@pytest.mark.parametrize(
    "case",
    [
        _C(lambda p, s: ak.zeta(s), 1),
        _C(lambda p, s: ak.hurwitz_zeta(s, p.rational(1, 3)), 1),
        _C(lambda p, s: ak.hurwitz_zeta(s, p.integer(2)), 1),
        _C(lambda p, s: ak.polylog(s, p.integer(1)), 1),
        _C(lambda p, s: ak.digamma(s), 0),
        _C(lambda p, s: ak.trigamma(s), 0),
        _C(lambda p, s: ak.polygamma(2, s), 0),
        _C(lambda p, s: ak.gamma(s), 0),
        _C(lambda p, s: ak.gamma(s), -2),
        _C(lambda p, s: ak.loggamma(s), 0),
        _C(lambda p, s: ak.zeta_prime(s), 1),
        _C(lambda p, s: ak.bessel_y(p.integer(0), s), 0),
        _C(lambda p, s: ak.bessel_y(p.integer(1), s), 0),
        _C(lambda p, s: ak.bessel_k(p.integer(0), s), 0),
        _C(lambda p, s: ak.bessel_k(p.integer(1), s), 0),
        _C(lambda p, s: ak.bessel_j(p.rational(-1, 3), s), 0),
    ],
)
def test_no_series_contains_a_pole_value(case):
    _no_pole_value_or_refused(case)


def test_regular_points_are_unaffected():
    p, s = _pool()
    r = ak.series(ak.zeta(s), s, p.integer(2), 2)
    for h in (0.01, -0.02):
        got = _eval(r.truncated(), {s: 2.0 + h})
        assert abs(got - float(mp.zeta(2 + h))) < 1e-3
