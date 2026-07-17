"""Special-function foundation primitives (Lambert W₀, digamma, Bessel J₀/J₁)."""

from __future__ import annotations

import math

import pytest
from alkahest import ExprPool, diff, eval_expr, parse
from alkahest import experimental as ex


@pytest.fixture
def pool():
    return ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def test_construct_and_str(pool, x):
    assert "lambert_w" in str(ex.lambert_w(x))
    assert "digamma" in str(ex.digamma(x))
    assert "bessel_j0" in str(ex.bessel_j0(x))
    assert "bessel_j1" in str(ex.bessel_j1(x))


def test_lambert_w_special_values(pool):
    em = -math.exp(-1.0)
    w0 = ex.lambert_w(pool.integer(0))
    assert eval_expr(w0, {}) == 0.0
    w_branch = ex.lambert_w(pool.float(em, 53))
    assert abs(eval_expr(w_branch, {}) + 1.0) < 1e-12


def test_digamma_at_one(pool):
    psi1 = ex.digamma(pool.integer(1))
    val = eval_expr(psi1, {})
    assert abs(val + 0.5772156649015329) < 1e-9


def test_bessel_j0_at_zero(pool):
    j0 = ex.bessel_j0(pool.integer(0))
    assert abs(eval_expr(j0, {}) - 1.0) < 1e-12


def test_parse_roundtrip(pool, x):
    for src, name in [
        ("lambert_w(x)", "lambert_w"),
        ("digamma(x)", "digamma"),
        ("bessel_j0(x)", "bessel_j0"),
        ("bessel_j1(x)", "bessel_j1"),
    ]:
        e = parse(src, pool, {"x": x})
        assert name in str(e), f"{src} -> {e}"


def test_diff_bessel_j0(pool, x):
    j0 = ex.bessel_j0(x)
    d = diff(j0, x)
    s = str(d.value)
    assert "bessel_j1" in s and "-1" in s


def test_diff_lambert_w(pool, x):
    w = ex.lambert_w(x)
    d = diff(w, x)
    assert "lambert_w" in str(d.value)


def test_registry_coverage():
    reg = __import__("alkahest").PrimitiveRegistry()
    report = {row["name"]: row for row in reg.coverage_report()}
    for name in ("lambert_w", "digamma", "bessel_j0", "bessel_j1"):
        assert name in report
        assert report[name]["numeric_f64"]
        assert report[name]["numeric_ball"]
