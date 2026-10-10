"""Definite trigonometric integrals with elliptic and arctan-type antiderivatives.

Regression for report 10-9 W3: ``∫_0^{π/2} dx/√(1 − sin²x/4)`` returned an
"exact" expression built from f64-derived rationals (``4503599618403511/2^52``)
and ``tan(π/2)``, 3e-9 away from ``K(1/4)``.  Three things are pinned here:

* the complete elliptic integrals come back exactly as ``EllipticK``/``EllipticE``
  (parameter convention ``m = k²``);
* an antiderivative that passes through a pole of ``tan`` at an endpoint, or
  inside the interval, is evaluated by one-sided limits (or refused), never by
  substituting the pole;
* no definite result carries a binary fraction lifted from a float.

The differential test compares every answer with ``mpmath.quad``.  A coded
refusal is acceptable for an individual case; a wrong value never is.
"""

import re

import alkahest as ak
import pytest
from alkahest import cos, eval_expr, integrate, sin, sqrt

mp = pytest.importorskip("mpmath")

_FRACTION = re.compile(r"(\d+)/(\d+)")


def _float_lifted(expr) -> list:
    """Denominators that are powers of two of at least 2**40."""
    out = []
    for _, d in _FRACTION.findall(str(expr)):
        d = int(d)
        if d >= 2**40 and d & (d - 1) == 0:
            out.append(d)
    return out


@pytest.fixture
def env():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    pi = ak.parse("pi", pool)
    return pool, x, pi


def test_report_repro_is_exactly_complete_k(env):
    pool, x, pi = env
    f = 1 / sqrt(1 - pool.rational(1, 4) * sin(x) ** 2)
    r = integrate(f, x, pool.integer(0), pi / 2)
    assert str(r.value) == "EllipticK(1/4)"
    assert abs(eval_expr(r.value, {}) - float(mp.ellipk(mp.mpf(1) / 4))) < 1e-14


def test_complete_second_kind_and_cos_forms(env):
    pool, x, pi = env
    half = pool.rational(1, 2)
    r = integrate(sqrt(1 - pool.rational(1, 4) * sin(x) ** 2), x, pool.integer(0), pi / 2)
    assert str(r.value) == "EllipticE(1/4)"
    r = integrate(1 / sqrt(1 - half * cos(x) ** 2), x, pool.integer(0), pi / 2)
    assert str(r.value) == "EllipticK(1/2)"


def test_indefinite_legendre_form_is_exact(env):
    pool, x, _ = env
    f = 1 / sqrt(1 - pool.rational(1, 4) * sin(x) ** 2)
    r = integrate(f, x)
    assert str(r.value) == "EllipticF(x, 1/4)"
    assert r.verification["status"] == "exactly_verified"


def test_weierstrass_endpoint_pole_is_a_limit_not_tan_pi_half(env):
    # F = (2/3)·atan(3·tan(x/2)) reaches tan(π/2) at x = π.
    pool, x, pi = env
    r = integrate(1 / (5 - 4 * cos(x)), x, pool.integer(0), pi)
    assert "tan" not in str(r.value)
    assert abs(eval_expr(r.value, {}) - float(mp.pi / 3)) < 1e-14


def test_weierstrass_interior_jump_is_split(env):
    # ∫_0^{2π} dx/(2 + cos x) = 2π/√3; the antiderivative jumps at π.
    pool, x, pi = env
    r = integrate(1 / (2 + cos(x)), x, pool.integer(0), 2 * pi)
    assert "tan" not in str(r.value)
    assert abs(eval_expr(r.value, {}) - float(2 * mp.pi / mp.sqrt(3))) < 1e-13


def test_limit_at_a_tan_pole_is_one_sided(env):
    _, x, pi = env
    f = ak.atan(2 * ak.tan(x))
    below = ak.limit(f, x, pi / 2, "-")
    above = ak.limit(f, x, pi / 2, "+")
    assert "tan" not in str(below.value)
    assert "tan" not in str(above.value)
    assert abs(eval_expr(below.value, {}) - float(mp.pi / 2)) < 1e-15
    assert abs(eval_expr(above.value, {}) + float(mp.pi / 2)) < 1e-15
    with pytest.raises(Exception):
        ak.limit(f, x, pi / 2)


def _cases(pool, x, pi):
    Z, R = pool.integer, pool.rational
    msin, mcos, msqrt = mp.sin, mp.cos, mp.sqrt
    return [
        # (label, integrand, a, b, mpmath integrand, mp a, mp b, must_succeed)
        (
            "K(1/4)",
            1 / sqrt(1 - R(1, 4) * sin(x) ** 2),
            Z(0),
            pi / 2,
            lambda t: 1 / msqrt(1 - msin(t) ** 2 / 4),
            0,
            mp.pi / 2,
            True,
        ),
        (
            "E(1/4)",
            sqrt(1 - R(1, 4) * sin(x) ** 2),
            Z(0),
            pi / 2,
            lambda t: msqrt(1 - msin(t) ** 2 / 4),
            0,
            mp.pi / 2,
            True,
        ),
        (
            "K(1/2) cos form",
            1 / sqrt(1 - R(1, 2) * cos(x) ** 2),
            Z(0),
            pi / 2,
            lambda t: 1 / msqrt(1 - mcos(t) ** 2 / 2),
            0,
            mp.pi / 2,
            True,
        ),
        (
            "E cos form",
            sqrt(3 - cos(x) ** 2),
            Z(0),
            pi / 2,
            lambda t: msqrt(3 - mcos(t) ** 2),
            0,
            mp.pi / 2,
            True,
        ),
        (
            "2-sin^2 over [0, pi]",
            1 / sqrt(2 - sin(x) ** 2),
            Z(0),
            pi,
            lambda t: 1 / msqrt(2 - msin(t) ** 2),
            0,
            mp.pi,
            True,
        ),
        (
            "across 3pi/2",
            1 / sqrt(1 - R(1, 3) * sin(x) ** 2),
            Z(0),
            3 * pi / 2,
            lambda t: 1 / msqrt(1 - msin(t) ** 2 / 3),
            0,
            3 * mp.pi / 2,
            True,
        ),
        (
            "negative m",
            1 / sqrt(1 + 2 * sin(x) ** 2),
            Z(0),
            pi / 2,
            lambda t: 1 / msqrt(1 + 2 * msin(t) ** 2),
            0,
            mp.pi / 2,
            True,
        ),
        (
            "incomplete F",
            1 / sqrt(1 - R(1, 4) * sin(x) ** 2),
            Z(0),
            Z(1),
            lambda t: 1 / msqrt(1 - msin(t) ** 2 / 4),
            0,
            1,
            True,
        ),
        (
            "sin^2(2x)",
            1 / sqrt(1 - R(1, 2) * sin(2 * x) ** 2),
            Z(0),
            pi / 4,
            lambda t: 1 / msqrt(1 - msin(2 * t) ** 2 / 2),
            0,
            mp.pi / 4,
            True,
        ),
        (
            "sqrt(2+cos)^-1 half angle",
            1 / sqrt(2 + cos(x)),
            Z(0),
            pi,
            lambda t: 1 / msqrt(2 + mcos(t)),
            0,
            mp.pi,
            True,
        ),
        ("sqrt(3+cos)", sqrt(3 + cos(x)), Z(0), pi, lambda t: msqrt(3 + mcos(t)), 0, mp.pi, True),
        (
            "sqrt(3-sin)^-1",
            1 / sqrt(3 - sin(x)),
            Z(0),
            pi,
            lambda t: 1 / msqrt(3 - msin(t)),
            0,
            mp.pi,
            True,
        ),
        (
            "1/(2+cos) [0, pi/2]",
            1 / (2 + cos(x)),
            Z(0),
            pi / 2,
            lambda t: 1 / (2 + mcos(t)),
            0,
            mp.pi / 2,
            True,
        ),
        (
            "1/(2+cos) [0, 2pi]",
            1 / (2 + cos(x)),
            Z(0),
            2 * pi,
            lambda t: 1 / (2 + mcos(t)),
            0,
            2 * mp.pi,
            True,
        ),
        (
            "1/(5-4cos) [0, pi]",
            1 / (5 - 4 * cos(x)),
            Z(0),
            pi,
            lambda t: 1 / (5 - 4 * mcos(t)),
            0,
            mp.pi,
            True,
        ),
        (
            "1/(5-4cos) [pi/2, 3pi/2]",
            1 / (5 - 4 * cos(x)),
            pi / 2,
            3 * pi / 2,
            lambda t: 1 / (5 - 4 * mcos(t)),
            mp.pi / 2,
            3 * mp.pi / 2,
            True,
        ),
        (
            "1/(3+cos) reversed [pi, 0]",
            1 / (3 + cos(x)),
            pi,
            Z(0),
            lambda t: 1 / (3 + mcos(t)),
            mp.pi,
            0,
            True,
        ),
        ("1/(1+x^2)", 1 / (1 + x**2), Z(0), Z(1), lambda t: 1 / (1 + t**2), 0, 1, True),
        (
            "1/(1+sin^2) [0, pi]",
            1 / (1 + sin(x) ** 2),
            Z(0),
            pi,
            lambda t: 1 / (1 + msin(t) ** 2),
            0,
            mp.pi,
            False,
        ),
        (
            "1/(3+cos^2) [0, 3pi/2]",
            1 / (3 + cos(x) ** 2),
            Z(0),
            3 * pi / 2,
            lambda t: 1 / (3 + mcos(t) ** 2),
            0,
            3 * mp.pi / 2,
            False,
        ),
        (
            "1/sqrt(x^3+1)",
            1 / sqrt(x**3 + 1),
            Z(0),
            Z(1),
            lambda t: 1 / msqrt(t**3 + 1),
            0,
            1,
            False,
        ),
        (
            "1/sqrt(1-x^4)",
            1 / sqrt(1 - x**4),
            Z(0),
            R(1, 2),
            lambda t: 1 / msqrt(1 - t**4),
            0,
            mp.mpf(1) / 2,
            False,
        ),
    ]


def test_differential_against_mpmath(env):
    pool, x, pi = env
    mp.mp.dps = 30
    solved = 0
    refusals = []
    for label, f, a, b, g, lo, hi, must in _cases(pool, x, pi):
        pts = mp.linspace(lo, hi, 7)
        want = mp.quad(g, pts)
        try:
            r = integrate(f, x, a, b)
        except Exception as e:  # a coded refusal is acceptable, a wrong value is not
            refusals.append((label, must, str(e)))
            continue
        got = eval_expr(r.value, {})
        rel = abs(got - want) / abs(want)
        assert rel < 1e-12, f"{label}: {r.value} = {got}, mpmath {want} (rel {float(rel):.2e})"
        if not _float_lifted(f):
            assert not _float_lifted(r.value), f"{label}: float-lifted constant in {r.value}"
        solved += 1
    for label, must, msg in refusals:
        assert not must, f"{label}: refused: {msg}"
        assert "E-" in msg, f"{label}: uncoded refusal {msg!r}"
    assert solved >= 18
