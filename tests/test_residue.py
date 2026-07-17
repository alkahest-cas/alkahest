import alkahest as ak
import pytest
from alkahest.experimental import residue


def test_residue_simple_pole_at_origin():
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)
    expr = z**-1

    assert str(ak.simplify(residue(expr, z, 0)).value) == "1"


def test_residue_double_pole_is_zero():
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)
    a = pool.integer(2)
    expr = (z - a) ** -2

    assert str(ak.simplify(residue(expr, z, 2)).value) == "0"


def test_residue_at_i_for_reciprocal_quadratic():
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)
    expr = (z**2 + 1) ** -1

    r = ak.simplify(residue(expr, z, 1j)).value
    i = pool.symbol("I", ak.Domain.Complex)
    expected = ak.simplify(pool.rational(-1, 2) * i).value
    assert str(r) == str(expected)


def test_non_pole_returns_zero():
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)

    assert str(ak.simplify(residue(z**-1, z, 1)).value) == "0"


def test_non_rational_declines():
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)

    with pytest.raises(ValueError, match="E-RESIDUE-001"):
        residue(ak.sin(z), z, 0)
