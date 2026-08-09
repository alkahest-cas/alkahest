"""P1 item 7 — creative telescoping / holonomic (D-finite) machinery.

Zeilberger's algorithm is a *decision procedure that emits a certificate*: for a
proper hypergeometric term ``F(n, k)`` it either returns a recurrence together
with a rational certificate that has been re-checked as an exact identity, or it
refuses.  These tests cover both halves — the classical identities it must
decide, and the boundaries at which it must refuse rather than guess.
"""

import math

import alkahest as ak
import pytest


def _binomial(pool, top, bot):
    """``C(top, bot)`` as a ratio of gammas — the proper hypergeometric form."""
    one = pool.integer(1)
    return ak.gamma(top + one) / (ak.gamma(bot + one) * ak.gamma(top - bot + one))


def _num(expr, env):
    """Float value of ``expr`` under ``env``."""
    return float(ak.evaluate(expr, env).value)


def _eval_coeffs(cert, n, value):
    """Numeric ``[a_0(n), …, a_J(n)]`` at ``n = value``."""
    return [_num(c, {n: float(value)}) for c in cert.coeffs]


# ---------------------------------------------------------------------------
# Identities the algorithm must decide
# ---------------------------------------------------------------------------


def test_binomial_row_sum_gives_order_one_recurrence():
    """``Σ_k C(n,k) = 2^n`` — the textbook order-1 case.

    The certificate says ``S(n+1) − 2·S(n) = 0``, so ``a_0/a_1 == -2``.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")

    cert = ak.zeilberger(_binomial(pool, n, k), n, k)

    assert cert.order == 1
    assert len(cert.coeffs) == 2
    for ni in (3.0, 7.0, 11.5):
        a0, a1 = _eval_coeffs(cert, n, ni)
        assert a1 != 0.0
        assert a0 / a1 == pytest.approx(-2.0, rel=1e-12)


def test_recurrence_is_satisfied_by_the_actual_sum():
    """The returned recurrence must hold for the sum it claims to describe.

    This is the end-to-end check that matters: it does not inspect the
    certificate at all, it just evaluates ``S(n) = Σ_k C(n,k)`` numerically and
    asserts ``Σ_i a_i(n)·S(n+i) = 0``.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")

    cert = ak.zeilberger(_binomial(pool, n, k), n, k)

    def s(m):
        return sum(math.comb(m, j) for j in range(m + 1))

    for ni in range(2, 9):
        a = _eval_coeffs(cert, n, ni)
        total = sum(a[i] * s(ni + i) for i in range(len(a)))
        assert total == pytest.approx(0.0, abs=1e-6 * max(1.0, abs(s(ni + len(a) - 1))))


def test_certificate_is_returned_and_nontrivial():
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")

    cert = ak.zeilberger(_binomial(pool, n, k), n, k)

    assert cert.certificate is not None
    assert str(cert.certificate) != "0"
    # The derivation log is what makes the result auditable downstream.
    assert cert.derivation
    assert "ZeilbergerCertificate(" in repr(cert)


def test_certificate_telescopes_the_summand_numerically():
    """``Σ_i a_i(n)·F(n+i,k) = G(n,k+1) − G(n,k)`` with ``G = R·F``.

    Verified pointwise at integer ``(n, k)`` away from the poles of ``R``.  The
    identity is checked exactly in ``Q(n)(k)`` inside the engine; this is an
    independent numeric spot-check of the same claim.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    f = _binomial(pool, n, k)

    cert = ak.zeilberger(f, n, k)
    order = cert.order

    def big_f(ni, ki):
        if ki < 0 or ki > ni:
            return 0.0
        return float(math.comb(int(ni), int(ki)))

    def g(ni, ki):
        """``G(n,k) = R(n,k)·F(n,k)``."""
        r = _num(cert.certificate, {n: float(ni), k: float(ki)})
        return r * big_f(ni, ki)

    for ni in (5, 6, 7):
        a = _eval_coeffs(cert, n, ni)
        for ki in (1, 2, 3):
            lhs = sum(a[i] * big_f(ni + i, ki) for i in range(order + 1))
            rhs = g(ni, ki + 1) - g(ni, ki)
            assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Refusals — a refusal is an informative answer, not a failure
# ---------------------------------------------------------------------------


def test_refuses_non_hypergeometric_input():
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")

    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.zeilberger(ak.sin(n * k), n, k)

    assert excinfo.value.code == "E-HOLO-001"
    assert excinfo.value.remediation


def test_refuses_coincident_indices():
    pool = ak.ExprPool()
    n = pool.symbol("n")

    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.zeilberger(n, n, n)

    assert excinfo.value.code == "E-HOLO-004"


def test_refuses_when_search_bounds_are_exhausted():
    """No recurrence within the bounds: refuse, do not return a guess.

    ``Σ_k 1/(k² + n)`` is a proper hypergeometric term — both shift quotients
    are rational — but has no low-order P-recursive relation, so the bounded
    search must come back empty rather than returning something unverified.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    term = pool.integer(1) / (k * k + n)

    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.zeilberger(term, n, k, max_order=1, max_degree=2)

    assert excinfo.value.code == "E-HOLO-002"
    # E-HOLO-002 says "not found within these bounds", not "does not exist" —
    # the remediation has to tell an agent it can retry wider.
    assert "max_order" in excinfo.value.remediation


def test_rejects_nonsense_bounds():
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")

    with pytest.raises(ak.HolonomicError) as excinfo:
        ak.zeilberger(_binomial(pool, n, k), n, k, max_order=0)

    assert excinfo.value.code == "E-HOLO-004"


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_exported_surface():
    assert "zeilberger" in ak.__all__
    assert "ZeilbergerCertificate" in ak.__all__
    assert "HolonomicError" in ak.__all__
    assert issubclass(ak.HolonomicError, ak.AlkahestError)
