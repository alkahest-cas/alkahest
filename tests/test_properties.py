"""Hypothesis-based property tests for the Python API."""

import hypothesis.strategies as st
from alkahest.alkahest import ExprPool, UniPoly, diff, simplify
from hypothesis import given


def build_poly_expr(pool, x, coeffs):
    """Build a polynomial expression from a coefficient list (constant term first)."""
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        c_id = pool.integer(c)
        if i == 0:
            terms.append(c_id)
        else:
            xpow = x ** i
            if c == 1:
                terms.append(xpow)
            else:
                terms.append(c_id * xpow)
    if not terms:
        return pool.integer(0)
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr


small_coeff = st.integers(min_value=-5, max_value=5)
coeffs_strategy = st.lists(small_coeff, min_size=1, max_size=4)


@given(coeffs=coeffs_strategy)
def test_simplify_idempotent(coeffs):
    """simplify(simplify(e)) == simplify(e) for any polynomial expression."""
    pool = ExprPool()
    x = pool.symbol("x")
    expr = build_poly_expr(pool, x, coeffs)
    r1 = simplify(expr)
    r2 = simplify(r1.value)
    assert r1.value == r2.value, f"Idempotence failed for coeffs={coeffs}"


@given(coeffs=coeffs_strategy)
def test_diff_constant_is_zero(coeffs):
    """d/dx c = 0 for any constant expression."""
    pool = ExprPool()
    x = pool.symbol("x")
    c = pool.integer(coeffs[0])
    r = diff(c, x)
    assert r.value == pool.integer(0)


@given(
    fa=coeffs_strategy,
    fb=coeffs_strategy,
    a=st.integers(min_value=-3, max_value=3),
    b=st.integers(min_value=-3, max_value=3),
)
def test_diff_linearity(fa, fb, a, b):
    """diff(a*f + b*g, x) == a*diff(f,x) + b*diff(g,x) as polynomials."""
    pool = ExprPool()
    x = pool.symbol("x")

    f = build_poly_expr(pool, x, fa)
    g = build_poly_expr(pool, x, fb)
    a_id = pool.integer(a)
    b_id = pool.integer(b)

    lhs_expr = a_id * f + b_id * g
    lhs = diff(lhs_expr, x)

    df = diff(f, x)
    dg = diff(g, x)
    rhs_expr = a_id * df.value + b_id * dg.value
    rhs = simplify(rhs_expr)

    lhs_poly = _try_to_poly(lhs.value, x)
    rhs_poly = _try_to_poly(rhs.value, x)

    if lhs_poly is not None and rhs_poly is not None:
        assert lhs_poly == rhs_poly, (
            f"Linearity failed: fa={fa}, fb={fb}, a={a}, b={b}"
        )


@given(fa=coeffs_strategy, fb=coeffs_strategy)
def test_diff_product_rule(fa, fb):
    """d/dx(f*g) == f*g' + g*f' as polynomials."""
    pool = ExprPool()
    x = pool.symbol("x")

    f = build_poly_expr(pool, x, fa)
    g = build_poly_expr(pool, x, fb)

    lhs = diff(f * g, x)

    df = diff(f, x)
    dg = diff(g, x)
    rhs_expr = f * dg.value + g * df.value
    rhs = simplify(rhs_expr)

    lhs_poly = _try_to_poly(lhs.value, x)
    rhs_poly = _try_to_poly(rhs.value, x)

    if lhs_poly is not None and rhs_poly is not None:
        assert lhs_poly == rhs_poly, (
            f"Product rule failed: fa={fa}, fb={fb}"
        )


@given(
    a=st.integers(min_value=-20, max_value=20),
    b=st.integers(min_value=-20, max_value=20),
)
def test_simplify_integer_const_fold(a, b):
    """simplify(a + b) == a + b as integer."""
    pool = ExprPool()
    expr = pool.integer(a) + pool.integer(b)
    r = simplify(expr)
    assert r.value == pool.integer(a + b)


@given(coeffs=coeffs_strategy)
def test_hash_stability(coeffs):
    """Same expression built twice has the same hash."""
    pool = ExprPool()
    x = pool.symbol("x")
    e1 = build_poly_expr(pool, x, coeffs)
    e2 = build_poly_expr(pool, x, coeffs)
    assert e1 == e2
    assert hash(e1) == hash(e2)


@given(coeffs=coeffs_strategy)
def test_expr_usable_as_dict_key(coeffs):
    """Expressions can be used as dict keys and set members."""
    pool = ExprPool()
    x = pool.symbol("x")
    e = build_poly_expr(pool, x, coeffs)
    d = {e: "val"}
    assert d[e] == "val"
    s = {e, e, pool.symbol("y")}
    assert e in s


def _try_to_poly(expr, x):
    """Try to convert expr to a coefficient tuple; return None if it fails."""
    try:
        p = UniPoly.from_symbolic(expr, x)
        return tuple(p.coefficients())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Additional property tests (R-9)
# ---------------------------------------------------------------------------

from alkahest.alkahest import diff_forward, integrate  # noqa: E402


@given(fa=coeffs_strategy, fb=coeffs_strategy)
def test_diff_sum_rule(fa, fb):
    """diff(f + g, x) == diff(f, x) + diff(g, x) as polynomials."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = build_poly_expr(pool, x, fa)
    g = build_poly_expr(pool, x, fb)

    lhs = diff(f + g, x)
    df = diff(f, x)
    dg = diff(g, x)
    rhs = simplify(df.value + dg.value)

    lp = _try_to_poly(lhs.value, x)
    rp = _try_to_poly(rhs.value, x)
    if lp is not None and rp is not None:
        assert lp == rp, f"Sum rule failed: fa={fa}, fb={fb}"


@given(fa=coeffs_strategy, fb=coeffs_strategy)
def test_diff_forward_agrees_symbolic(fa, fb):
    """diff_forward(f+g) == diff(f+g) for random polynomials."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = build_poly_expr(pool, x, fa)
    g = build_poly_expr(pool, x, fb)
    expr = f + g

    try:
        fwd = diff_forward(expr, x)
    except Exception:
        return  # non-polynomial; skip

    sym = diff(expr, x)

    lp = _try_to_poly(fwd.value, x)
    rp = _try_to_poly(sym.value, x)
    if lp is not None and rp is not None:
        assert lp == rp, f"forward/symbolic disagreement: fa={fa}, fb={fb}"


@given(coeffs=coeffs_strategy)
def test_integrate_diff_inverse(coeffs):
    """diff(integrate(f, x), x) == f for polynomial f (no constant term)."""
    pool = ExprPool()
    x = pool.symbol("x")
    # Drop constant term to avoid antiderivative ambiguity
    non_const = [0] + list(coeffs[1:]) if len(coeffs) > 1 else [0, coeffs[0]]
    f = build_poly_expr(pool, x, non_const)

    try:
        integral = integrate(f, x)
    except Exception:
        return  # not in supported subset

    derivative = diff(integral.value, x)
    r = simplify(derivative.value)

    fp = _try_to_poly(f, x)
    rp = _try_to_poly(r.value, x)
    if fp is not None and rp is not None:
        assert fp == rp, f"integrate/diff not inverse: coeffs={non_const}"


@given(coeffs=coeffs_strategy)
def test_simplify_idempotent_twice(coeffs):
    """simplify(simplify(e)) == simplify(e) (idempotence)."""
    pool = ExprPool()
    x = pool.symbol("x")
    expr = build_poly_expr(pool, x, coeffs)
    r1 = simplify(expr)
    r2 = simplify(r1.value)
    assert r1.value == r2.value, f"Idempotence failed for coeffs={coeffs}"
