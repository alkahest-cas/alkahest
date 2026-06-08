"""PR1: elliptic special-function primitives.

Covers construction, string rendering, parse round-trip, and that the
symbolic derivative of EllipticF w.r.t. its amplitude reduces to the
elementary integrand 1/sqrt(1 - m*sin^2(phi)).

Parameter convention: m = k^2 (matches Mathematica's EllipticF[phi, m]).
"""

from __future__ import annotations

import math

import pytest
from alkahest import (
    ExprPool,
    diff,
    elliptic_e,
    elliptic_f,
    elliptic_k,
    elliptic_pi,
    eval_expr,
    parse,
    simplify,
)


@pytest.fixture
def pool():
    return ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


# ---------------------------------------------------------------------------
# Construction + rendering
# ---------------------------------------------------------------------------


def test_construct_and_str(pool, x):
    m = pool.rational(1, 2)
    assert "EllipticK" in str(elliptic_k(m))
    assert "EllipticE" in str(elliptic_e(m))  # complete
    assert "EllipticE" in str(elliptic_e(x, m))  # incomplete
    assert "EllipticF" in str(elliptic_f(x, m))
    n = pool.rational(1, 4)
    assert "EllipticPi" in str(elliptic_pi(n, x, m))


def test_elliptic_e_overload(pool, x):
    m = pool.rational(1, 2)
    # Complete form has one argument; incomplete form has two.
    assert str(elliptic_e(m)) != str(elliptic_e(x, m))


# ---------------------------------------------------------------------------
# Parse round-trip
# ---------------------------------------------------------------------------


def test_parse_roundtrip_F(pool, x):
    e = parse("EllipticF(x, 1/2)", pool, {"x": x})
    assert "EllipticF" in str(e)


def test_parse_roundtrip_all(pool, x):
    for src, name in [
        ("EllipticK(1/2)", "EllipticK"),
        ("EllipticE(1/2)", "EllipticE"),
        ("EllipticE(x, 1/2)", "EllipticE"),
        ("EllipticF(x, 1/2)", "EllipticF"),
        ("EllipticPi(1/4, x, 1/2)", "EllipticPi"),
    ]:
        e = parse(src, pool, {"x": x})
        assert name in str(e), f"{src} -> {e}"


def test_parse_simplify_roundtrip(pool, x):
    e = parse("EllipticF(x, 1/2)", pool, {"x": x})
    s = simplify(e)
    assert "EllipticF" in str(s.value)


# ---------------------------------------------------------------------------
# Differentiation
# ---------------------------------------------------------------------------


def test_diff_elliptic_f_wrt_phi(pool, x):
    m = pool.rational(1, 2)
    f = elliptic_f(x, m)
    d = diff(f, x)
    # d/dx F(x, 1/2) = 1 / sqrt(1 - (1/2) sin^2 x).  The derivative is now an
    # elementary expression (no EllipticF/E), evaluable via eval_expr.
    val = eval_expr(d, {x: 0.7})
    expect = 1.0 / math.sqrt(1.0 - 0.5 * math.sin(0.7) ** 2)
    assert abs(val - expect) < 1e-9


def test_diff_elliptic_e_incomplete_wrt_phi(pool, x):
    m = pool.rational(1, 2)
    e = elliptic_e(x, m)
    d = diff(e, x)
    # d/dx E(x, 1/2) = sqrt(1 - (1/2) sin^2 x).
    val = eval_expr(d, {x: 0.7})
    expect = math.sqrt(1.0 - 0.5 * math.sin(0.7) ** 2)
    assert abs(val - expect) < 1e-9


def test_diff_elliptic_pi_wrt_phi(pool, x):
    n = pool.rational(1, 4)
    m = pool.rational(1, 2)
    p = elliptic_pi(n, x, m)
    d = diff(p, x)
    # d/dphi Pi(n, phi, m) = 1 / ((1 - n sin^2 phi) sqrt(1 - m sin^2 phi)).
    val = eval_expr(d, {x: 0.6})
    s2 = math.sin(0.6) ** 2
    expect = 1.0 / ((1.0 - 0.25 * s2) * math.sqrt(1.0 - 0.5 * s2))
    assert abs(val - expect) < 1e-9
