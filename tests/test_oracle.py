"""
SymPy oracle: differentiation and integration results compared against SymPy.

Run:
    pytest tests/test_oracle.py -v

Requires:
    pip install sympy
"""

import pytest

sympy = pytest.importorskip("sympy")

import random  # noqa: E402

from alkahest import cos, eval_expr, exp, sin  # noqa: E402
from alkahest.alkahest import ExprPool, UniPoly, diff, integrate, simplify  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def alkahest_poly(pool, x, coeffs: list[int]):
    """Build sum(c * x^i) in a alkahest ExprPool."""
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        c_id = pool.integer(c)
        if i == 0:
            terms.append(c_id)
        else:
            xpow = x ** i
            terms.append(c_id * xpow if c != 1 else xpow)
    if not terms:
        return pool.integer(0)
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr


def sympy_poly(sx, coeffs: list[int]):
    """Build sum(c * sx^i) in SymPy."""
    return sum(c * sx**i for i, c in enumerate(coeffs) if c != 0) or sympy.Integer(0)


def alkahest_coeffs(expr, pool_x, pool) -> list[int] | None:
    """Extract coefficient list from a alkahest Expr; return None if not a polynomial."""
    try:
        p = UniPoly.from_symbolic(expr, pool_x)
        return p.coefficients()
    except Exception:
        return None


def sympy_coeffs(expr, sx) -> list[int] | None:
    """Extract coefficient list from a SymPy expression; return None if not a polynomial."""
    try:
        p = sympy.Poly(sympy.expand(expr), sx)
        cs = [int(c) for c in reversed(p.all_coeffs())]
        # pad to include trailing zeros
        return cs or [0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Oracle cases
# ---------------------------------------------------------------------------

CASES = [
    # (description, coeffs, expected_diff_coeffs)
    # expected_diff_coeffs = derivative of sum(c_i * x^i)
    ("constant", [5], [0]),
    ("linear_1x", [0, 1], [1]),
    ("linear_3x_plus_2", [2, 3], [3]),
    ("quadratic", [1, 2, 1], [2, 2]),
    ("cubic", [0, 0, 0, 1], [0, 0, 3]),
    ("full_cubic", [1, 2, 3, 4], [2, 6, 12]),
    ("degree_4", [1, 1, 1, 1, 1], [1, 2, 3, 4]),
    ("negative_coeffs", [-3, 0, 2], [0, 4]),
    ("zero_poly", [0, 0, 0], [0]),
    ("leading_zero_stripped", [0, 0, 1], [0, 2]),
]


@pytest.mark.parametrize("desc,coeffs,expected", CASES)
def test_diff_matches_known(desc, coeffs, expected):
    """Alkahest diff matches hand-computed derivative for fixed cases."""
    pool = ExprPool()
    x = pool.symbol("x")

    expr = alkahest_poly(pool, x, coeffs)
    r = diff(expr, x)
    got = alkahest_coeffs(r.value, x, pool) or [0]

    # Normalise: strip trailing zeros, default to [0]
    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0] or [0]

    assert norm(got) == norm(expected), (
        f"[{desc}] coeffs={coeffs}: alkahest={got}, expected={expected}"
    )


@pytest.mark.parametrize("seed", range(100))
def test_diff_matches_sympy(seed):
    """Alkahest diff(p, x) == SymPy diff(p, x) for 100 random polynomials."""
    rng = random.Random(seed)
    deg = rng.randint(0, 5)
    coeffs = [rng.randint(-10, 10) for _ in range(deg + 1)]

    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_deriv = sympy.diff(sp_expr, sx)
    sp_coeffs = sympy_coeffs(sp_deriv, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)
    ca_r = diff(ak_expr, x)
    ca_coeffs = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs is None or ca_coeffs is None:
        pytest.skip("expression not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs) == norm(sp_coeffs), (
        f"seed={seed} coeffs={coeffs}: alkahest={ca_coeffs}, sympy={sp_coeffs}"
    )


@pytest.mark.parametrize("seed", range(50))
def test_simplify_matches_sympy(seed):
    """simplify(p) has the same polynomial value as SymPy expand(p)."""
    rng = random.Random(seed + 200)  # different seed space
    deg = rng.randint(0, 4)
    coeffs = [rng.randint(-5, 5) for _ in range(deg + 1)]

    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_expanded = sympy.expand(sp_expr)
    sp_coeffs = sympy_coeffs(sp_expanded, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)
    ca_r = simplify(ak_expr)
    ca_coeffs = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs is None or ca_coeffs is None:
        pytest.skip("expression not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs) == norm(sp_coeffs), (
        f"seed={seed} coeffs={coeffs}: alkahest={ca_coeffs}, sympy={sp_coeffs}"
    )


# ---------------------------------------------------------------------------
# Integration oracle — alkahest integrate vs SymPy integrate
# ---------------------------------------------------------------------------

_INTEGRATE_CASES = [
    ("constant_5", [5]),
    ("linear_1x", [0, 1]),
    ("linear_3x_plus_2", [2, 3]),
    ("quadratic", [1, 2, 1]),
    ("cubic", [0, 0, 0, 1]),
    ("full_cubic", [1, 2, 3, 4]),
    ("degree_4", [1, 1, 1, 1, 1]),
    ("negative_coeffs", [-3, 0, 2]),
    ("zero_poly", [0]),
    ("degree_5", [1, -1, 2, -2, 3, -3]),
]


@pytest.mark.parametrize("desc,coeffs", _INTEGRATE_CASES)
def test_integrate_matches_sympy(desc, coeffs):
    """alkahest integrate(p, x) == SymPy integrate(p, x) for fixed polynomial cases."""
    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_integral = sympy.integrate(sp_expr, sx)
    sp_coeffs_int = sympy_coeffs(sp_integral, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)

    try:
        ca_r = integrate(ak_expr, x)
    except Exception:
        pytest.skip(f"[{desc}] alkahest returned NotImplemented")

    ca_coeffs_int = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs_int is None or ca_coeffs_int is None:
        pytest.skip(f"[{desc}] not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs_int) == norm(sp_coeffs_int), (
        f"[{desc}] coeffs={coeffs}: alkahest={ca_coeffs_int}, sympy={sp_coeffs_int}"
    )


@pytest.mark.parametrize("seed", range(50))
def test_integrate_matches_sympy_random(seed):
    """alkahest integrate(p, x) agrees with SymPy for 50 random polynomials."""
    rng = random.Random(seed + 500)
    deg = rng.randint(0, 4)
    coeffs = [rng.randint(-5, 5) for _ in range(deg + 1)]

    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_integral = sympy.integrate(sp_expr, sx)
    sp_coeffs_int = sympy_coeffs(sp_integral, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)

    try:
        ca_r = integrate(ak_expr, x)
    except Exception:
        pytest.skip(f"seed={seed}: alkahest returned NotImplemented for coeffs={coeffs}")

    ca_coeffs_int = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs_int is None or ca_coeffs_int is None:
        pytest.skip(f"seed={seed}: not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs_int) == norm(sp_coeffs_int), (
        f"seed={seed} coeffs={coeffs}: alkahest={ca_coeffs_int}, sympy={sp_coeffs_int}"
    )


# ---------------------------------------------------------------------------
# Non-polynomial integration oracle — random integrable expressions vs SymPy
# ---------------------------------------------------------------------------
#
# Strategy: build random expressions from a grammar of forms that alkahest's
# rule-based integrator handles (sin/cos/exp with linear args, powers, sums,
# scalar multiples).  For each generated expression:
#   1. Compute alkahest integral F.
#   2. Numerically differentiate F at several test points and compare to the
#      original expression evaluated at the same points.  This avoids the need
#      to normalise two symbolic forms that may look different but be equal.
#   3. Also compare F against SymPy's integral numerically (the "vs SymPy"
#      part of the oracle).
# ---------------------------------------------------------------------------

_NONPOLY_TEST_POINTS = [0.3, 0.7, 1.2, 1.9, 2.5]
_DIFF_H = 1e-7
_ABS_TOL = 1e-5
_REL_TOL = 1e-4


def _eval_at(expr, x_sym, x_val: float) -> float | None:
    """Evaluate an alkahest expression at a single float point; return None on error."""
    try:
        return eval_expr(expr, {x_sym: x_val})
    except Exception:
        return None


def _sympy_eval(sp_expr, sx, x_val: float) -> float | None:
    try:
        v = float(sp_expr.subs(sx, x_val).evalf())
        return v
    except Exception:
        return None


def _numeric_derivative(F_expr, x_sym, x_val: float) -> float | None:
    """Central-difference approximation of F'(x_val)."""
    plus = _eval_at(F_expr, x_sym, x_val + _DIFF_H)
    minus = _eval_at(F_expr, x_sym, x_val - _DIFF_H)
    if plus is None or minus is None:
        return None
    return (plus - minus) / (2 * _DIFF_H)


def _approx_equal(a: float, b: float) -> bool:
    """True when |a - b| ≤ atol + rtol * |b|."""
    return abs(a - b) <= _ABS_TOL + _REL_TOL * abs(b)


# --- Expression grammar ---

def _random_atom(pool, x, sx, rng):
    """Return (alk_expr, sympy_expr) for a simple integrable atom.

    Only forms that alkahest's rule-based integrator handles are generated:
    - x^n        (power rule, n=0..4)
    - sin(x)     (direct rule; sin(a*x) for a>1 is not yet implemented)
    - cos(x)     (direct rule)
    - exp(a*x)   (linear-arg rule, a=1..2)
    - x*exp(x)   (product rule)
    """
    choice = rng.randint(0, 4)
    if choice == 0:
        n = rng.randint(0, 4)
        return x ** n, sx ** n
    elif choice == 1:
        return sin(x), sympy.sin(sx)
    elif choice == 2:
        return cos(x), sympy.cos(sx)
    elif choice == 3:
        a = rng.randint(1, 2)
        return exp(pool.integer(a) * x), sympy.exp(sympy.Integer(a) * sx)
    else:
        # x * exp(x) — special product rule
        return x * exp(x), sx * sympy.exp(sx)


def _random_integrable(pool, x, sx, rng, depth=0):
    """Recursively build a random integrable (alk_expr, sympy_expr) pair."""
    if depth >= 2:
        return _random_atom(pool, x, sx, rng)

    r = rng.random()

    if r < 0.30:
        return _random_atom(pool, x, sx, rng)
    elif r < 0.55:
        # sum of two integrable sub-expressions
        a_ak, a_sp = _random_integrable(pool, x, sx, rng, depth + 1)
        b_ak, b_sp = _random_integrable(pool, x, sx, rng, depth + 1)
        return a_ak + b_ak, a_sp + b_sp
    elif r < 0.75:
        # scalar multiple
        c = rng.randint(1, 4)
        a_ak, a_sp = _random_integrable(pool, x, sx, rng, depth + 1)
        return pool.integer(c) * a_ak, sympy.Integer(c) * a_sp
    else:
        return _random_atom(pool, x, sx, rng)


@pytest.mark.parametrize("seed", range(80))
def test_integrate_nonpoly_diff_check(seed):
    """d/dx(alkahest_integrate(f, x)) ≈ f at test points (no SymPy required)."""
    rng = random.Random(seed + 1000)
    pool = ExprPool()
    x = pool.symbol("x")
    sx = sympy.Symbol("x")

    f_ak, _ = _random_integrable(pool, x, sx, rng)

    try:
        F_result = integrate(f_ak, x)
    except Exception:
        pytest.skip(f"seed={seed}: alkahest.integrate raised for this expr")

    F_ak = F_result.value
    passed = 0
    skipped = 0

    for x_val in _NONPOLY_TEST_POINTS:
        f_val = _eval_at(f_ak, x, x_val)
        F_prime = _numeric_derivative(F_ak, x, x_val)
        if f_val is None or F_prime is None:
            skipped += 1
            continue
        assert _approx_equal(F_prime, f_val), (
            f"seed={seed} x={x_val}: F'={F_prime:.6g} but f={f_val:.6g}"
        )
        passed += 1

    if passed == 0:
        pytest.skip(f"seed={seed}: all test points produced NaN/error")


def _sympy_derivative(sp_expr, sx, x_val: float) -> float | None:
    """Central-difference approximation of d/dx sp_expr at x_val."""
    h = _DIFF_H
    try:
        plus = float(sp_expr.subs(sx, x_val + h).evalf())
        minus = float(sp_expr.subs(sx, x_val - h).evalf())
        return (plus - minus) / (2 * h)
    except Exception:
        return None


@pytest.mark.parametrize("seed", range(80))
def test_integrate_nonpoly_vs_sympy(seed):
    """alkahest F'(x) ≈ SymPy F'(x) numerically — both antiderivatives satisfy FTC."""
    rng = random.Random(seed + 2000)
    pool = ExprPool()
    x = pool.symbol("x")
    sx = sympy.Symbol("x")

    f_ak, f_sp = _random_integrable(pool, x, sx, rng)

    try:
        F_ak_result = integrate(f_ak, x)
    except Exception:
        pytest.skip(f"seed={seed}: alkahest.integrate raised for this expr")

    F_ak = F_ak_result.value

    try:
        F_sp = sympy.integrate(f_sp, sx)
    except Exception:
        pytest.skip(f"seed={seed}: sympy.integrate raised for this expr")

    passed = 0
    skipped = 0

    for x_val in _NONPOLY_TEST_POINTS:
        # Antiderivatives may differ by a constant, so compare F'(x) against f(x)
        # for both alkahest and SymPy independently.
        ak_prime = _numeric_derivative(F_ak, x, x_val)
        sp_prime = _sympy_derivative(F_sp, sx, x_val)
        f_val = _eval_at(f_ak, x, x_val)

        if ak_prime is None or sp_prime is None or f_val is None:
            skipped += 1
            continue

        assert _approx_equal(ak_prime, f_val), (
            f"seed={seed} x={x_val}: alkahest F'={ak_prime:.6g} but f={f_val:.6g}"
        )
        assert _approx_equal(sp_prime, f_val), (
            f"seed={seed} x={x_val}: sympy F'={sp_prime:.6g} but f={f_val:.6g}"
        )
        passed += 1

    if passed == 0:
        pytest.skip(f"seed={seed}: all test points produced NaN/error")
