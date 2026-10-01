"""V2-7 — polynomial factorization (Python surface)."""

import alkahest
import pytest


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
        assert base.degree == 1
    assert fac.verification == {
        "status": "exactly_verified",
        "evidence": "factor_product",
        "method": "in_kernel_exact_reconstruction",
        "lean_checked": False,
    }


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
    assert fac.verification["status"] == "exactly_verified"
    assert fac.verification["evidence"] == "factor_product"
    assert fac.verification["lean_checked"] is False


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
    with pytest.raises(alkahest.FactorError) as exc_info:
        z.factor_z()
    assert exc_info.value.code == "E-POLY-008"


def _factor_strs(poly):
    fac = poly.factor_z()
    return fac.unit, [(str(b), e) for b, e in fac.factor_list()]


def test_factor_list_canonical_order_univariate():
    # Ascending degree, then coefficients from the leading term down — the
    # order SymPy's factor_list uses, and the same on every FLINT version.
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    p = alkahest.UniPoly.from_symbolic(x**12 - pool.integer(1), x)
    assert _factor_strs(p) == (
        "1",
        [
            ("x-1", 1),
            ("x+1", 1),
            ("x^2-x+1", 1),
            ("x^2+1", 1),
            ("x^2+x+1", 1),
            ("x^4-x^2+1", 1),
        ],
    )
    e = x**4 * (x**2 - 2) * (x**2 + 2) * (x**3 + 1) * (3 * x**2 + 1)
    assert _factor_strs(alkahest.UniPoly.from_symbolic(e, x))[1] == [
        ("x", 4),
        ("x+1", 1),
        ("x^2-x+1", 1),
        ("x^2-2", 1),
        ("x^2+2", 1),
        ("3*x^2+1", 1),
    ]


def test_cyclotomic_fast_path_unit_and_content():
    # 6x^6 - 6 and -x^5 + 32 = -((x)^5 - 2^5) take the binomial fast path;
    # the content and sign stay in the unit, the factors stay primitive.
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    p = alkahest.UniPoly.from_symbolic(6 * x**6 - 6, x)
    assert _factor_strs(p) == (
        "6",
        [("x-1", 1), ("x+1", 1), ("x^2-x+1", 1), ("x^2+x+1", 1)],
    )
    q = alkahest.UniPoly.from_symbolic(-(x**5) + 32, x)
    assert _factor_strs(q) == ("-1", [("x-2", 1), ("x^4+2*x^3+4*x^2+8*x+16", 1)])
    big = alkahest.UniPoly.from_symbolic(x**720 - 1, x)
    fac = big.factor_z()
    assert len(fac.factor_list()) == 30  # one Φ_d per divisor d of 720
    assert fac.verification["status"] == "exactly_verified"


def test_factor_list_canonical_order_multivariate_and_mod_p():
    pool = alkahest.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    mp = alkahest.MultiPoly.from_symbolic(x**6 - y**6, [x, y])
    # x - y, x + y, x^2 - xy + y^2, x^2 + xy + y^2 (x0 = x, x1 = y).
    assert _factor_strs(mp) == (
        "1",
        [
            ("-x1 + x0", 1),
            ("x1 + x0", 1),
            ("x1^2 - x0x1 + x0^2", 1),
            ("x1^2 + x0x1 + x0^2", 1),
        ],
    )
    r = alkahest.factor_univariate_mod_p([-1, 0, 0, 0, 0, 0, 0, 0, 1], 17)
    assert [f for f, _ in r.factor_list()] == [[c, 1] for c in (1, 2, 4, 8, 9, 13, 15, 16)]
