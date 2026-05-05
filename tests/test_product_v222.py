"""V2-22 — discrete symbolic ∏ (SymPy oracle for Wallis piece)."""

from __future__ import annotations

import math

import pytest


def sympy_available():
    try:
        import sympy  # noqa: F401

        return True
    except ImportError:
        return False


def test_factorial_piece_matches_gamma_numeric():
    import alkahest as ah

    pool = ah.ExprPool()
    k = pool.symbol("k")
    n = pool.symbol("n")
    one = pool.integer(1)

    prod = ah.simplify(ah.product_definite(k, k, one, n).value).value
    fc = ah.compile_expr(prod, [n])
    for nv in range(1, 13):
        y = fc([float(nv)])
        fac = math.factorial(nv)
        assert math.isfinite(y)
        assert abs(y - fac) < 5e-4 * abs(fac)


def test_wallis_rational_piece():
    import alkahest as ah

    pool = ah.ExprPool()
    k = pool.symbol("k")
    n = pool.symbol("n")
    kp2 = k ** 2
    numer = ah.simplify((k + pool.integer(-1)) * (k + pool.integer(1))).value
    term = ah.simplify(numer / kp2).value
    prod = ah.product_definite(term, k, pool.integer(2), n).value
    fc = ah.compile_expr(prod, [n])

    for nv in range(3, 48):
        y = fc([float(nv)])
        want = (nv + 1) / (2 * nv)
        assert abs(y - want) <= 6e-5 * abs(want)


@pytest.mark.skipif(not sympy_available(), reason="SymPy oracle optional")
def test_sympy_wallis_piece():
    import alkahest as ah
    from sympy import Product, simplify, symbols

    k_sym, n_sym = symbols("k n", integer=True, positive=True)
    sp_prod = simplify(Product((k_sym**2 - 1) / k_sym**2, (k_sym, 2, n_sym)).doit())

    pool = ah.ExprPool()
    ka = pool.symbol("k")
    na = pool.symbol("n")
    kp2 = ka ** 2
    term = ah.simplify(
        (ka + pool.integer(-1)) * (ka + pool.integer(1)) / kp2,
    ).value

    ah_val = ah.simplify(ah.product_definite(term, ka, pool.integer(2), na).value).value

    for nv in range(3, 15):
        sp_n = simplify(sp_prod.subs(n_sym, nv))
        sympy_float = float(sp_n.evalf())
        y = ah.compile_expr(ah_val, [na])([float(nv)])
        assert abs(sympy_float - y) < 2e-4 * max(abs(sympy_float), 1.0)


def test_product_syntax_class():
    import alkahest as ah

    pool = ah.ExprPool()
    k = pool.symbol("k")
    n = pool.symbol("n")
    P = ah.Product(k, (k, pool.integer(1), n))
    assert P.doit().value is not None
