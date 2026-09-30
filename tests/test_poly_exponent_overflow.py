"""Polynomial exponents past ``u32`` are refused or computed exactly, never wrapped.

The sparse polynomial types key their terms by ``u32`` exponents and used to
multiply monomials with a plain ``+``: in a release build ``x^(2^31)·x^(2^31)``
became ``x^0 = 1``. Every case below returned an answer for a *different*
polynomial (or, for ``UniPoly``, aborted the interpreter inside FLINT) before
the fix. ``W`` is kept as an unevaluated product, as the audit built it.
"""

from __future__ import annotations

import alkahest as ak
import pytest

B31 = 2**31


@pytest.fixture
def env():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    big = pool.mul([x ** pool.integer(B31), x ** pool.integer(B31)])  # x^(2^32)
    return pool, x, y, big


def _code(exc) -> str:
    return getattr(exc.value, "code", None) or str(exc.value)


def test_multipoly_product_is_refused(env):
    pool, x, _, big = env
    with pytest.raises(ak.ConversionError) as e:
        ak.MultiPoly.from_symbolic(big, [x])
    assert "E-POLY-004" in _code(e)
    with pytest.raises(ak.ConversionError):
        ak.MultiPoly.from_symbolic((x ** pool.integer(65536)) ** pool.integer(65536), [x])


def test_multipoly_total_degree_is_refused(env):
    pool, x, y, _ = env
    e = pool.mul([x ** pool.integer(B31), y ** pool.integer(B31)])
    with pytest.raises(ak.ConversionError):
        ak.MultiPoly.from_symbolic(e, [x, y])
    ok = pool.mul([x ** pool.integer(B31), y ** pool.integer(B31 - 1)])
    assert ak.MultiPoly.from_symbolic(ok, [x, y]).total_degree == 2**32 - 1


def test_factor_z_is_refused(env):
    pool, x, y, big = env
    with pytest.raises(ak.ConversionError):
        ak.MultiPoly.from_symbolic(big - y ** pool.integer(2), [x, y])


def test_poly_normal_is_refused(env):
    pool, x, y, big = env
    with pytest.raises(ak.ConversionError):
        ak.poly_normal(pool.mul([big, y]) - y, [x, y])


def test_horner_and_real_roots_are_refused(env):
    pool, x, _, big = env
    # horner maps the conversion error to ValueError, as it always has.
    with pytest.raises((ak.ConversionError, ValueError), match="too large"):
        ak.horner(big + x, x)
    with pytest.raises(ak.AlkahestError):
        ak.real_roots(big - pool.integer(4), x)


def test_solve_is_refused(env):
    pool, x, y, big = env
    with pytest.raises(ak.SolverError):
        ak.solve([big - pool.integer(2), y - x], [x, y])
    # expr_to_gbpoly maps its errors to ValueError, as it always has.
    with pytest.raises((ak.SolverError, ValueError), match="E-POLY-004"):
        ak.expr_to_gbpoly(big - pool.integer(2), [x, y])


def test_resultant_past_u32_is_exact(env):
    pool, x, y, _ = env
    n = 3 * 10**9
    yn = y ** pool.integer(n)
    r = ak.resultant(x ** pool.integer(2) - yn, x - yn, x).value
    # y^(2N) - y^N; it came back as -y^3000000000 + y^1705032704.
    assert "y^6000000000" in str(r)
    assert "y^3000000000" in str(r)
    assert "1705032704" not in str(r)


@pytest.mark.parametrize("e", [B31, 2**32, 2**63, 2**64 - 1])
def test_unipoly_dense_ceiling_refuses_instead_of_aborting(env, e):
    pool, x, _, _ = env
    with pytest.raises(ak.ConversionError) as err:
        ak.UniPoly.from_symbolic(x ** pool.integer(e) + pool.integer(1), x)
    assert "E-POLY-004" in _code(err)


def test_unipoly_dense_ceiling_under_a_memory_budget(env):
    pool, x, _, _ = env
    f = x ** pool.integer(2**23) + pool.integer(1)
    with (
        ak.context(budget=ak.Budget(max_bytes=10**6)),
        pytest.raises(ak.ConversionError),
    ):
        ak.UniPoly.from_symbolic(f, x)


def test_cancel_of_a_wrapped_product_is_refused_quickly(env):
    pool, x, _, big = env
    with pytest.raises(ak.ConversionError):
        ak.cancel((big - pool.integer(1)) / (x - pool.integer(1)))


def test_rational_conversions_decline_instead_of_truncating(env):
    """``n as u32`` in the Risch rational conversion: residue answered 1 and
    apart returned ``1/x`` for ``x^-(2^32+1)``."""
    pool, x, _, _ = env
    f = x ** pool.integer(-(2**32 + 1))
    with pytest.raises(ak.AlkahestError):
        ak.residue(f, x, 0)
    with pytest.raises((ak.AlkahestError, ValueError)):
        ak.apart(pool.integer(1) / x ** pool.integer(2**32 + 1), x)
    r = ak.residue(x ** pool.integer(-1), x, 0)
    assert getattr(r, "value", r) == pool.integer(1)


def test_diophantine_does_not_drop_a_wrapped_term(env):
    """x² + y² − 65 + x^(2^31)·y^(2^31) was solved as x² + y² = 65."""
    pool, x, y, _ = env
    eq = (
        x ** pool.integer(2)
        + y ** pool.integer(2)
        - pool.integer(65)
        + pool.mul([x ** pool.integer(B31), y ** pool.integer(B31)])
    )
    with pytest.raises(ak.DiophantineError):
        ak.diophantine(eq, [x, y])


def test_together_is_refused(env):
    pool, _, y, big = env
    with pytest.raises(ak.ConversionError):
        ak.together(big + pool.integer(1) / y)


def test_unipoly_pow_past_the_dense_ceiling_does_not_abort(env):
    """``UniPoly ** 2^31`` asked FLINT for 16 GiB and aborted the process."""
    pool, x, _, _ = env
    p = ak.UniPoly.from_symbolic(x + pool.integer(1), x)
    with pytest.raises(ak.ConversionError) as err:
        p ** (2**31)
    assert "E-POLY-004" in _code(err)
    assert (p**3).degree == 3


def test_unipoly_mul_past_the_dense_ceiling_does_not_abort(env):
    """``*`` is checked like ``**``: a product whose dense coefficient array
    would not fit is refused before FLINT allocates it."""
    pool, x, _, _ = env
    p = ak.UniPoly.from_symbolic(x ** pool.integer(2**16) + pool.integer(1), x)
    assert (p * p).degree == 2**17
    with (
        ak.context(budget=ak.Budget(max_bytes=10**6)),
        pytest.raises(ak.ConversionError) as err,
    ):
        p * p
    assert "E-POLY-004" in _code(err)


def test_multipoly_mul_operator_is_refused(env):
    """``MultiPoly * MultiPoly`` used to wrap x^(2^31)·x^(2^31) to 1."""
    pool, x, y, _ = env
    a = ak.MultiPoly.from_symbolic(x ** pool.integer(B31), [x, y])
    with pytest.raises(ak.ConversionError) as err:
        a * a
    assert "E-POLY-004" in _code(err)
    b = ak.MultiPoly.from_symbolic(x ** pool.integer(B31 - 1), [x, y])
    assert (a * b).total_degree == 2**32 - 1
