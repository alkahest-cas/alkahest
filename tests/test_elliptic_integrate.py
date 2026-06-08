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


def test_integral_one_over_sqrt_quartic_x4_plus_1():
    """∫ dx/√(x⁴+1) -> EllipticF (all-complex-root quartic, no real roots)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**4 + pool.integer(1)
    integrand = sqrt(p) ** -1
    _check(pool, x, integrand, lambda v: v**4 + 1.0, [-1.5, -0.5, 0.5, 1.0, 2.0])


def test_integral_one_over_sqrt_quartic_x4_plus_x2_plus_1():
    """∫ dx/√(x⁴+x²+1) -> EllipticF (two complex-conjugate pairs)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**4 + x**2 + pool.integer(1)
    integrand = sqrt(p) ** -1
    _check(pool, x, integrand, lambda v: v**4 + v**2 + 1.0, [-1.5, -0.5, 0.5, 1.0, 2.0])


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
    _check_second_kind(pool, x, sqrt(p), lambda v: v**3 + 1.0, [0.5, 1.0, 2.0, 3.0], ["EllipticF"])


def test_integral_sqrt_cubic_three_real_needs_e():
    """∫√(x³−x) dx (region x>1) genuinely needs EllipticE (three real roots)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 - x
    _check_second_kind(pool, x, sqrt(p), lambda v: v**3 - v, [1.2, 1.6, 2.2, 3.3], ["EllipticE"])


def test_integral_sqrt_quartic_1_minus_x4():
    """∫√(1−x⁴) dx → algebraic part + elliptic functions (region |x|<1)."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = pool.integer(1) - x**4
    _check_second_kind(pool, x, sqrt(p), lambda v: 1.0 - v**4, [-0.8, -0.3, 0.3, 0.8], ["Elliptic"])


# ---------------------------------------------------------------------------
# General second kind: ∫ poly(x)/√P dx → algebraic part + EllipticF + EllipticE
# ---------------------------------------------------------------------------


def _check_poly_over_sqrt(pool, x, integrand, integrand_func, p_func, sample_xs, must_contain):
    """Integrate ∫ R(x)/√P, assert the required elliptic functions appear, and
    verify d/dx F = R(x)/√P numerically where P > 0."""
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
        rhs = integrand_func(xv)
        assert abs(lhs - rhs) < 1e-6 * (1.0 + abs(rhs)), (
            f"x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {f}"
        )
        checked += 1
    assert checked >= 3
    return f


def test_integral_x_over_sqrt_cubic_x3_plus_1():
    """Headline: ∫ x/√(x³+1) dx → algebraic part + EllipticF + EllipticE."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 + pool.integer(1)
    _check_poly_over_sqrt(
        pool,
        x,
        x / sqrt(p),
        lambda v: v / math.sqrt(v**3 + 1.0),
        lambda v: v**3 + 1.0,
        [0.3, 0.6, 1.0, 2.0, 3.0],
        ["Elliptic"],
    )


def test_integral_x_plus_1_over_sqrt_cubic_x3_plus_1():
    """General polynomial numerator: ∫ (x+1)/√(x³+1) dx."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 + pool.integer(1)
    _check_poly_over_sqrt(
        pool,
        x,
        (x + pool.integer(1)) / sqrt(p),
        lambda v: (v + 1.0) / math.sqrt(v**3 + 1.0),
        lambda v: v**3 + 1.0,
        [0.3, 0.6, 1.0, 2.0, 3.0],
        ["Elliptic"],
    )


# ---------------------------------------------------------------------------
# Third kind: ∫ R(x)/((x−p)√P) dx → EllipticPi  (real pole p off the roots of P)
# ---------------------------------------------------------------------------


def test_integral_third_kind_cubic_three_real_emits_pi():
    """∫ dx/((x−3)√(x³−x)) → EllipticPi (+ EllipticF), gate-verified on x>1.

    The radicand ``x³−x`` has three real roots {−1,0,1}; on the region ``x>1``
    the first-kind substitution uses ``asin(√·)`` so ``sin²φ`` is Möbius in x and
    the simple real pole at ``x=3`` reduces to a *single* incomplete elliptic
    integral of the third kind.  The native gate verifies d/dx F = integrand
    before emitting, so the EllipticPi form is always correct.
    """
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 - x
    integrand = ((x - pool.integer(3)) * sqrt(p)) ** -1
    res = integrate(integrand, x)
    f = res.value
    s = str(f)
    assert "EllipticPi" in s, f"expected EllipticPi, got {f}"
    d = diff(f, x).value
    checked = 0
    for xv in [1.2, 1.6, 2.2, 4.0, 5.0]:
        pv = xv**3 - xv
        assert pv > 0, f"sample {xv} not in domain P>0"
        lhs = _num_eval(d, xv)
        rhs = 1.0 / ((xv - 3.0) * math.sqrt(pv))
        assert abs(lhs - rhs) < 1e-6 * (1.0 + abs(rhs)), (
            f"x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {f}"
        )
        checked += 1
    assert checked >= 3


def test_integral_third_kind_cubic_one_real_declines():
    """∫ dx/((x−2)√(x³+1)) declines: the ``cosφ`` substitution for the
    one-real-root cubic makes ``sin²φ`` a *quadratic* rational of x, so a single
    EllipticPi has a spurious twin pole and no finite F/E/Π/algebraic combination
    reproduces the integrand.  The soundness gate therefore never emits an
    (incorrect) closed form — the engine reports the integral as not handled."""
    pool = ExprPool()
    x = pool.symbol("x")
    p = x**3 + pool.integer(1)
    integrand = ((x - pool.integer(2)) * sqrt(p)) ** -1
    with pytest.raises(Exception):
        integrate(integrand, x)
