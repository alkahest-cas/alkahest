"""PR2: first-kind elliptic-integral *output* for genus-1 radicands.

`∫ c·dx/√(P(x))` with `P` a cubic or quartic that is non-elementary reduces to
an incomplete elliptic integral of the first kind in Legendre normal form,
`c·g·EllipticF(φ(x), m)` (Mathematica `m = k²` convention).  The engine emits
this only when its internal numeric `d/dx F = integrand` gate passes, so the
returned form is always verifiable.

Each test asserts `EllipticF` appears in the rendered antiderivative and that
its symbolic derivative numerically matches the integrand `1/√P` at points
where `P > 0` (so both sides are real).
"""

from __future__ import annotations

import math

import pytest
from alkahest import ExprPool, diff, integrate, sqrt

# The closed forms use `asin`/`acos`, which the native `eval_expr`/`interval_eval`
# do not support; evaluate the derivative tree directly via the `node()` API.
_UNARY = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "cbrt": lambda v: math.copysign(abs(v) ** (1.0 / 3.0), v),
}


def _num_eval(expr, xv):
    """Recursively evaluate *expr* at ``x = xv`` using the node() reflection API."""
    n = expr.node()
    tag = n[0]
    if tag == "symbol":
        return xv
    if tag == "integer":
        return float(int(n[1]))
    if tag == "rational":
        return float(int(n[1])) / float(int(n[2]))
    if tag == "add":
        return sum(_num_eval(a, xv) for a in n[1])
    if tag == "mul":
        prod = 1.0
        for a in n[1]:
            prod *= _num_eval(a, xv)
        return prod
    if tag == "pow":
        return _num_eval(n[1], xv) ** _num_eval(n[2], xv)
    if tag == "func":
        name, args = n[1], n[2]
        if name in _UNARY and len(args) == 1:
            return _UNARY[name](_num_eval(args[0], xv))
    raise AssertionError(f"cannot evaluate node {n}")


def _check(pool, x, integrand, p_func, sample_xs):
    """Integrate, assert EllipticF in the result, and verify d/dx F = integrand."""
    res = integrate(integrand, x)
    f = res.value
    assert "EllipticF" in str(f), f"expected EllipticF, got {f}"
    d = diff(f, x).value
    checked = 0
    for xv in sample_xs:
        pv = p_func(xv)
        assert pv > 0, f"sample {xv} not in domain P>0"
        lhs = _num_eval(d, xv)
        rhs = 1.0 / math.sqrt(pv)
        assert abs(lhs - rhs) < 1e-6 * (1.0 + abs(rhs)), (
            f"x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {f}"
        )
        checked += 1
    assert checked >= 3
    return f


def test_integral_one_over_sqrt_cubic_x3_plus_1():
    """Headline: ∫ dx/√(x³+1) -> c·EllipticF(φ(x), m) (one real root + pair)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 + pool.integer(1)
    integrand = sqrt(p) ** -1
    _check(pool, x, integrand, lambda v: v**3 + 1.0, [0.5, 1.0, 2.0, 3.0])


def test_integral_one_over_sqrt_cubic_three_real():
    """∫ dx/√(x³−x) -> EllipticF (three real roots), verified on x > 1."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 - x
    integrand = sqrt(p) ** -1
    _check(pool, x, integrand, lambda v: v**3 - v, [1.2, 2.0, 4.0])


def test_integral_one_over_sqrt_quartic_1_minus_x4():
    """∫ dx/√(1−x⁴) -> EllipticF (two real roots + complex pair), |x| < 1."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = pool.integer(1) - x**4
    integrand = sqrt(p) ** -1
    _check(pool, x, integrand, lambda v: 1.0 - v**4, [-0.8, -0.2, 0.3, 0.8])


def test_quintic_first_kind_still_non_elementary():
    """∫ dx/√(x⁵+1) is genus-2: no first-kind reduction, still non-elementary."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**5 + pool.integer(1)
    integrand = sqrt(p) ** -1
    with pytest.raises(Exception) as exc_info:
        integrate(integrand, x)
    msg = str(exc_info.value)
    assert "E-INT-004" in msg or "elementary" in msg.lower() or "NonElementary" in msg


# ---------------------------------------------------------------------------
# PR3: second-kind output  ∫√P dx  →  algebraic part + EllipticF/EllipticE
# ---------------------------------------------------------------------------


def _check_second_kind(pool, x, integrand, p_func, sample_xs, must_contain):
    """Integrate ∫√P, assert the required elliptic functions appear, and verify
    d/dx F = √P numerically where P > 0."""
    res = integrate(integrand, x)
    f = res.value
    s = str(f)
    for needle in must_contain:
        assert needle in s, f"expected {needle}, got {f}"
    d = diff(f, x).value
    checked = 0
    for xv in sample_xs:
        pv = p_func(xv)
        assert pv > 0, f"sample {xv} not in domain P>0"
        lhs = _num_eval(d, xv)
        rhs = math.sqrt(pv)
        assert abs(lhs - rhs) < 1e-6 * (1.0 + abs(rhs)), (
            f"x={xv}: d/dx F = {lhs}, integrand √P = {rhs}\n  F = {f}"
        )
        checked += 1
    assert checked >= 3
    return f


def test_integral_sqrt_cubic_x3_plus_1():
    """Headline: ∫√(x³+1) dx → algebraic part + EllipticF (one real root + pair)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 + pool.integer(1)
    _check_second_kind(
        pool, x, sqrt(p), lambda v: v**3 + 1.0, [0.5, 1.0, 2.0, 3.0], ["EllipticF"]
    )


def test_integral_sqrt_cubic_three_real_needs_e():
    """∫√(x³−x) dx (region x>1) genuinely needs EllipticE (three real roots)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 - x
    _check_second_kind(
        pool, x, sqrt(p), lambda v: v**3 - v, [1.2, 1.6, 2.2, 3.3], ["EllipticE"]
    )


def test_integral_sqrt_quartic_1_minus_x4():
    """∫√(1−x⁴) dx → algebraic part + elliptic functions (region |x|<1)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = pool.integer(1) - x**4
    _check_second_kind(
        pool, x, sqrt(p), lambda v: 1.0 - v**4, [-0.8, -0.3, 0.3, 0.8], ["Elliptic"]
    )
