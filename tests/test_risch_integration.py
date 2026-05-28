"""
Transcendental Risch integration tests (issue #4).

Tests the full Risch decision procedure for elementary antiderivatives of
transcendental functions, covering:

  - Non-elementary certification: exp(x²), exp(-x²), sin(x)/x
  - Elementary exp tower: x·exp(x²), (2x²+1)·exp(x²), x²·exp(x), x³·exp(x), …
  - Elementary log tower: log(x)², log(x)³, x·log(x), x²·log(x), …
  - Mixed sums with independent generators

Run after building the extension:
    maturin develop --release
    pytest tests/test_risch_integration.py -v

References:
  - Risch (1969), Trans. AMS 139.
  - Bronstein (2005), Symbolic Integration I.
  - SymPy risch_integrate for oracle comparisons.
"""

import pytest
from alkahest.alkahest import ArbBall, ExprPool, diff, integrate, interval_eval

# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

_TEST_POINTS = (0.5, 1.2, 2.7, 3.14)


def check_antiderivative(pool, x, f, F, label="", points=_TEST_POINTS):
    """
    Numerically verify ∫ f dx = F by checking d/dx F(x) ≈ f(x) at several points.
    """
    dF_expr = diff(F, x).value
    for pt in points:
        bindings = {x: ArbBall(pt)}
        lhs = interval_eval(dF_expr, bindings).mid
        rhs = interval_eval(f, bindings).mid
        assert abs(lhs - rhs) < 1e-8, (
            f"{label}: d/dx F({pt:.4f}) = {lhs}, f({pt:.4f}) = {rhs} — mismatch\n"
            f"  F = {F}\n  f = {f}"
        )


# ---------------------------------------------------------------------------
# Non-elementary integrals (must raise with E-INT-004 / NonElementary)
# ---------------------------------------------------------------------------


def test_exp_x2_nonelementary():
    """∫ exp(x²) dx — the Gaussian integral, certified non-elementary."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = pool.func("exp", [x**2])
    with pytest.raises(Exception) as exc_info:
        integrate(f, x)
    msg = str(exc_info.value).lower()
    assert "elementary" in msg or "e-int-004" in str(exc_info.value), (
        f"Expected NonElementary error; got: {exc_info.value}"
    )


def test_exp_neg_x2_nonelementary():
    """∫ exp(-x²) dx — the Gaussian integral (negative exponent), non-elementary."""
    pool = ExprPool()
    x = pool.symbol("x")
    neg_x2 = pool.integer(-1) * x**2
    f = pool.func("exp", [neg_x2])
    with pytest.raises(Exception) as exc_info:
        integrate(f, x)
    msg = str(exc_info.value).lower()
    assert "elementary" in msg or "e-int-004" in str(exc_info.value), (
        f"Expected NonElementary error; got: {exc_info.value}"
    )


def test_exp_x3_nonelementary():
    """∫ exp(x³) dx — non-elementary (degree 3 exponent)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = pool.func("exp", [x**3])
    with pytest.raises(Exception) as exc_info:
        integrate(f, x)
    assert "elementary" in str(exc_info.value).lower() or "e-int-004" in str(exc_info.value)


def test_exp_x2_plus_x_nonelementary():
    """∫ exp(x² + x) dx — non-elementary (completing the square doesn't help)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = pool.func("exp", [x**2 + x])
    with pytest.raises(Exception) as exc_info:
        integrate(f, x)
    assert "elementary" in str(exc_info.value).lower() or "e-int-004" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Elementary exp tower: exp(g) with nonlinear g
# ---------------------------------------------------------------------------


def test_x_times_exp_x2():
    """∫ x·exp(x²) dx = ½·exp(x²)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * pool.func("exp", [x**2])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x·exp(x²)")


def test_two_x_times_exp_x2():
    """∫ 2x·exp(x²) dx = exp(x²)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = pool.integer(2) * x * pool.func("exp", [x**2])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "2x·exp(x²)")


def test_poly_times_exp_x2():
    """∫ (2x²+1)·exp(x²) dx = x·exp(x²)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = pool.integer(2) * x**2 + pool.integer(1)
    f = p * pool.func("exp", [x**2])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "(2x²+1)·exp(x²)")


def test_4x3_times_exp_x2():
    """∫ 4x³·exp(x²) dx = (2x²-1)·exp(x²) + C  (via RDE)."""
    pool = ExprPool()
    x = pool.symbol("x")
    # (2x² - 1)' = 4x, and (2x² - 1)·(2x) = 4x³-2x, so we actually integrate 4x³:
    # d/dx[(2x²-1)·exp(x²)] = 4x·exp(x²) + (2x²-1)·2x·exp(x²) = (4x + 4x³ - 2x)·exp(x²) = (4x³+2x)·exp(x²)
    # So ∫ (4x³+2x)·exp(x²) dx = (2x²-1)·exp(x²)
    p = pool.integer(4) * x**3 + pool.integer(2) * x
    f = p * pool.func("exp", [x**2])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "(4x³+2x)·exp(x²)")


# ---------------------------------------------------------------------------
# Elementary exp tower: poly(x) · exp(linear)
# ---------------------------------------------------------------------------


def test_x2_times_exp_x():
    """∫ x²·exp(x) dx = (x²-2x+2)·exp(x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 * pool.func("exp", [x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x²·exp(x)")


def test_x3_times_exp_x():
    """∫ x³·exp(x) dx = (x³-3x²+6x-6)·exp(x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**3 * pool.func("exp", [x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x³·exp(x)")


def test_x4_times_exp_x():
    """∫ x⁴·exp(x) dx — degree-4 polynomial times exp(x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**4 * pool.func("exp", [x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x⁴·exp(x)")


def test_x2_times_exp_2x():
    """∫ x²·exp(2x) dx — polynomial × exp(2x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 * pool.func("exp", [pool.integer(2) * x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x²·exp(2x)")


def test_x2_times_exp_neg_x():
    """∫ x²·exp(-x) dx — polynomial × exp(-x)."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 * pool.func("exp", [pool.integer(-1) * x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x²·exp(-x)")


# ---------------------------------------------------------------------------
# Elementary log tower: log(x)^n
# ---------------------------------------------------------------------------


def test_log_x_squared():
    """∫ log(x)² dx = x·log(x)² − 2x·log(x) + 2x."""
    pool = ExprPool()
    x = pool.symbol("x")
    log_x = pool.func("log", [x])
    f = log_x**2
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "log(x)²", points=(0.5, 1.2, 2.7))


def test_log_x_cubed():
    """∫ log(x)³ dx = x·log(x)³ − 3x·log(x)² + 6x·log(x) − 6x."""
    pool = ExprPool()
    x = pool.symbol("x")
    log_x = pool.func("log", [x])
    f = log_x**3
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "log(x)³", points=(0.5, 1.2, 2.7))


def test_log_x_fourth():
    """∫ log(x)⁴ dx — degree-4 power of log."""
    pool = ExprPool()
    x = pool.symbol("x")
    log_x = pool.func("log", [x])
    f = log_x**4
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "log(x)⁴", points=(0.5, 1.2, 2.7))


# ---------------------------------------------------------------------------
# Elementary log tower: poly(x) · log(x)
# ---------------------------------------------------------------------------


def test_x_times_log_x():
    """∫ x·log(x) dx = (x²/2)·log(x) − x²/4."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * pool.func("log", [x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x·log(x)", points=(0.5, 1.2, 2.7))


def test_x2_times_log_x():
    """∫ x²·log(x) dx = (x³/3)·log(x) − x³/9."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**2 * pool.func("log", [x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x²·log(x)", points=(0.5, 1.2, 2.7))


def test_x3_times_log_x():
    """∫ x³·log(x) dx = (x⁴/4)·log(x) − x⁴/16."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x**3 * pool.func("log", [x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x³·log(x)", points=(0.5, 1.2, 2.7))


def test_const_times_log_x():
    """∫ 3·log(x) dx = 3x·log(x) − 3x."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = pool.integer(3) * pool.func("log", [x])
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "3·log(x)", points=(0.5, 1.2, 2.7))


def test_x2_times_log_x_squared():
    """∫ x²·log(x)² dx — polynomial × power of log."""
    pool = ExprPool()
    x = pool.symbol("x")
    log_x = pool.func("log", [x])
    f = x**2 * log_x**2
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x²·log(x)²", points=(0.5, 1.2, 2.7))


# ---------------------------------------------------------------------------
# Regression: existing tests should still pass through Risch routing
# ---------------------------------------------------------------------------


def test_exp_x2_poly_sum():
    """∫ (x·exp(x²) + x²) dx = ½·exp(x²) + x³/3."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * pool.func("exp", [x**2]) + x**2
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "x·exp(x²) + x²")


def test_log_x_plus_x():
    """∫ (log(x)² + x) dx."""
    pool = ExprPool()
    x = pool.symbol("x")
    log_x = pool.func("log", [x])
    f = log_x**2 + x
    result = integrate(f, x)
    check_antiderivative(pool, x, f, result.value, "log(x)² + x", points=(0.5, 1.2, 2.7))


# ---------------------------------------------------------------------------
# Derivation log checks
# ---------------------------------------------------------------------------


def test_exp_x2_derivation_log_nonempty():
    """The derivation log for Risch integration should be non-empty."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = x * pool.func("exp", [x**2])
    result = integrate(f, x)
    steps = result.steps  # .steps is a property (list) not a method
    assert len(steps) > 0, "Risch integration should produce a non-empty derivation log"
    rule_names = [s.get("rule", s.get("rule_name", "")) if isinstance(s, dict) else getattr(s, "rule_name", getattr(s, "rule", "")) for s in steps]
    assert any("risch" in r.lower() for r in rule_names), (
        f"Risch integration should produce a 'risch_*' log step; got: {rule_names}"
    )
