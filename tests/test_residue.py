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


def test_refusals_are_coded_alkahest_errors():
    """`E-RESIDUE-*` must arrive as an `AlkahestError` carrying `.code`.

    They used to be bare `ValueError`s with the code glued into the message,
    so the only way to branch on one was to string-match. `AlkahestError`
    subclasses `ValueError`, so the older idiom keeps working.
    """
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)

    with pytest.raises(ak.AlkahestError) as excinfo:
        residue(ak.sin(z), z, 0)
    assert excinfo.value.code == "E-RESIDUE-001"
    assert excinfo.value.remediation
    assert isinstance(excinfo.value, ValueError)


@pytest.mark.parametrize("bad_point", ["symbol", "expr", "string", "object"])
def test_non_constant_point_raises_coded_error_not_attributeerror(bad_point):
    """A point that is not an exact constant must be refused with a code.

    Found by the crash sweep on a deeply nested polynomial, but the depth was
    incidental: `residue(f, z, point)` parsed `point` through `exact_binding`,
    which reached straight for `point.numerator` and let the resulting bare
    `AttributeError: 'Expr' object has no attribute 'numerator'` escape. That
    named our implementation rather than the caller's mistake, was not an
    `AlkahestError` (so `except ak.AlkahestError` missed it entirely), and
    carried no code to branch on. Passing an `Expr` as the point is the
    natural mistake here — `residue(f, z, a)` reads perfectly well — so it has
    to be a refusal, not a crash.
    """
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)
    point = {
        "symbol": pool.symbol("a"),
        "expr": pool.integer(0),
        "string": "0",
        "object": object(),
    }[bad_point]

    with pytest.raises(ak.AlkahestError) as excinfo:
        residue(z**-1, z, point)
    assert excinfo.value.code == "E-RESIDUE-005"
    assert excinfo.value.remediation
    assert not isinstance(excinfo.value, AttributeError)


def test_deeply_nested_polynomial_point_still_refused_cleanly():
    """The crash-sweep repro verbatim: a deep input plus a non-constant point."""
    pool = ak.ExprPool()
    z = pool.symbol("z", ak.Domain.Complex)
    deep = z
    for _ in range(300):
        deep = deep * z + pool.integer(1)

    with pytest.raises(ak.AlkahestError) as excinfo:
        residue(deep, z, pool.integer(0))
    assert excinfo.value.code == "E-RESIDUE-005"

    # ...and the same expression with a *valid* point is not refused for the
    # point's sake, so the fix did not simply reject more inputs.
    assert ak.residue(deep, z, 0) is not None
