"""V2-7 — polynomial factorization (Python surface)."""

import alkahest


def test_unipoly_factor_quadratic():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    x2 = x * x
    p = alkahest.UniPoly.from_symbolic(x2 - pool.integer(1), x)
    fac = p.factor_z()
    assert int(fac.unit) in (1, -1)
    assert len(fac.factor_list()) == 2
    for base, exp in fac.factor_list():
        assert exp == 1
        assert base.degree() == 1


def test_multipoly_factor_product():
    pool = alkahest.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    x2 = x * x
    y2 = y * y
    f1 = x2 + y2 - pool.integer(1)
    x_minus_y = x - y
    e = f1 * x_minus_y
    mp = alkahest.MultiPoly.from_symbolic(e, [x, y])
    fac = mp.factor_z()
    assert len(fac.factor_list()) >= 2


def test_factor_univariate_mod_p_x_squared_plus_one_char2():
    # x^2 + 1 = (x+1)^2 over F_2
    r = alkahest.factor_univariate_mod_p([1, 0, 1], 2)
    assert r.modulus == 2
    assert len(r.factor_list()) == 1
    cfs, e = r.factor_list()[0]
    assert e == 2
    assert cfs == [1, 1]


def test_factor_zero_raises():
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    z = alkahest.UniPoly.from_symbolic(pool.integer(0), x)
    try:
        z.factor_z()
    except alkahest.FactorError as exc:
        assert exc.code == "E-POLY-008"
    else:
        raise AssertionError("expected FactorError")
