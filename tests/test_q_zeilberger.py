"""M4(b) — ``q``-analogue creative telescoping (``q``-Zeilberger).

The engine is the ``q``-shifted twin of :func:`alkahest.zeilberger`: it proves
``Σ_i a_i(q**n)·F(n+i,k) = ΔG`` for a ``q``-hypergeometric summand and
re-checks the certificate as an exact identity in ``Q(q)(q**n)(q**k)`` before
returning it.  These tests cover the three things that make it worth having:

* a classical ``q``-identity decided **and independently re-checked against the
  actual exact ``q``-series terms** — the check a certificate that implies a
  false sum recurrence would not survive;
* the boundary verdict, which must say ``"unknown"`` rather than imply a
  recurrence it cannot prove;
* the refusals, which must be coded errors rather than answers.
"""

import alkahest as ak
import pytest
from alkahest.experimental import q_zeilberger, qbinomial, qpochhammer


def _syms(pool):
    return pool.symbol("q"), pool.symbol("n"), pool.symbol("k")


def _is_zero(pool, expr):
    """Exact: ``expr`` expands to the zero polynomial in ``q``."""
    return ak.simplify_expanded(expr).value == pool.integer(0)


def _q_vandermonde_term(pool, q, n, k):
    """``[n;k]_q² · q^{k²}`` — the summand of the ``q``-Vandermonde square sum."""
    b = qbinomial(pool, n, k)
    return b * b * q ** (k * k)


# ---------------------------------------------------------------------------
# The identity, end to end
# ---------------------------------------------------------------------------


def test_q_vandermonde_square_sum_recurrence_is_order_one():
    """``Σ_k [n;k]_q²·q^{k²} = [2n;n]_q`` — the ``q``-analogue of
    ``Σ_k C(n,k)² = C(2n,n)``.

    The recurrence must be order 1, and the boundary verdict must license
    reading it as a statement about the *sum*.
    """
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    cert = q_zeilberger(_q_vandermonde_term(pool, q, n, k), q, n, k)

    assert cert.order == 1
    assert len(cert.coeffs) == 2
    assert cert.boundary == "vanishes"
    assert cert.implies_sum_recurrence
    # The summand vanishes outside 0 <= k <= n, which is what makes the
    # Z-sum a finite sum — and the verdict says so rather than assuming it.
    assert cert.support == ("0", "n")


def test_recurrence_annihilates_the_exact_q_series_terms():
    """The independent check: the returned coefficients annihilate the actual
    sequence, in exact ``Q(q)`` arithmetic.

    ``sum_term`` is computed from the definition of the ``q``-Pochhammer
    symbol, never through the shift quotients the search used, so this is a
    check *of* the certificate rather than a restatement of it.  A valid
    certificate whose sum recurrence is false — the classical A279013 failure
    mode — fails here and nowhere else.
    """
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    cert = q_zeilberger(_q_vandermonde_term(pool, q, n, k), q, n, k)

    for n0 in range(6):
        total = pool.integer(0)
        for i, a in enumerate(cert.coeffs):
            total = total + ak.subs(a, {n: pool.integer(n0)}) * cert.sum_term(n0 + i)
        assert _is_zero(pool, total), f"the recurrence must annihilate S at n = {n0}"


def test_sum_terms_are_the_central_gaussian_binomial():
    """``S(n) = [2n;n]_q``, checked against ``Π_{i=1..n} (1−q^{n+i})/(1−q^i)``
    built independently out of pool arithmetic."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    cert = q_zeilberger(_q_vandermonde_term(pool, q, n, k), q, n, k)
    one = pool.integer(1)

    for n0 in range(6):
        num, den = one, one
        for i in range(1, n0 + 1):
            num = num * (one - q ** pool.integer(n0 + i))
            den = den * (one - q ** pool.integer(i))
        # S(n)·Π(1−q^i) − Π(1−q^{n+i}) must be exactly the zero polynomial.
        assert _is_zero(pool, cert.sum_term(n0) * den - num), f"S({n0}) != [2n;n]_q"


def test_alternating_q_binomial_sum_has_a_half_integral_exponent():
    """``Σ_k (−1)^k q^{k(k−1)/2} [n;k]_q = 0`` for ``n ≥ 1``.

    ``q^{k(k−1)/2}`` is not a rational function of ``q^k`` — the exponent is
    half-integral — but every shift quotient of it is, which is exactly the
    boundary of the supported class.
    """
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    half = pool.rational(1, 2)
    term = pool.integer(-1) ** k * q ** (half * k * (k - pool.integer(1))) * qbinomial(pool, n, k)
    cert = q_zeilberger(term, q, n, k)
    assert cert.boundary == "vanishes"
    for n0 in range(1, 5):
        assert _is_zero(pool, cert.sum_term(n0)), f"the alternating sum must vanish at n = {n0}"


def test_galois_numbers_need_order_two():
    """``Σ_k [n;k]_q`` — the Galois numbers, an order-2 recurrence."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    cert = q_zeilberger(qbinomial(pool, n, k), q, n, k)
    assert cert.order == 2
    assert cert.boundary == "vanishes"
    for n0 in range(4):
        total = pool.integer(0)
        for i, a in enumerate(cert.coeffs):
            total = total + ak.subs(a, {n: pool.integer(n0)}) * cert.sum_term(n0 + i)
        assert _is_zero(pool, total)


# ---------------------------------------------------------------------------
# The boundary verdict
# ---------------------------------------------------------------------------


def test_unbounded_support_gives_no_claim_about_the_sum():
    """``1/(q;q)_{n−k}`` telescopes perfectly well and has no ``Z``-sum.

    The verdict must be ``"unknown"``: the certificate is true about the
    summand, and nothing follows about any sum.  This is the case where
    assuming the boundary would manufacture a false theorem.
    """
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    term = qpochhammer(pool, 1, 1, n - k) ** pool.integer(-1)
    cert = q_zeilberger(term, q, n, k)

    assert cert.boundary == "unknown"
    assert not cert.implies_sum_recurrence
    assert cert.support is None
    assert "support" in cert.boundary_reason
    assert any("no recurrence for the sum follows" in s for s in cert.side_conditions)


def test_side_conditions_record_that_q_is_generic():
    """Every verdict is an identity in ``Q(q)``; specialising ``q`` to a root of
    unity is a separate step, and the side conditions say so rather than
    leaving a ``q``-supercongruence reader to assume otherwise."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    cert = q_zeilberger(_q_vandermonde_term(pool, q, n, k), q, n, k)
    assert any("root of unity" in s for s in cert.side_conditions)


# ---------------------------------------------------------------------------
# Refusals — coded errors, not answers
# ---------------------------------------------------------------------------


def test_refuses_non_q_hypergeometric_input():
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    with pytest.raises(ak.HolonomicError) as excinfo:
        q_zeilberger(pool.func("sin", [n * k]), q, n, k)
    assert excinfo.value.code == "E-HOLO-020"


def test_refuses_a_classical_hypergeometric_term():
    """A bare ``k`` outside an exponent is not ``q``-hypergeometric: the two
    classes are different and neither engine silently accepts the other's."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    with pytest.raises(ak.HolonomicError) as excinfo:
        q_zeilberger(qbinomial(pool, n, k) / (k + pool.integer(1)), q, n, k)
    assert excinfo.value.code == "E-HOLO-020"


def test_refuses_a_shift_the_base_does_not_divide():
    """``(q^k; q²)_n`` shifted in ``k`` moves its first argument by 1, which
    ``q²`` does not divide, so the shift quotient is an infinite product — in
    the shape of the class, outside it in substance.  ``E-HOLO-024``."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    with pytest.raises(ak.HolonomicError) as excinfo:
        q_zeilberger(qpochhammer(pool, k, 2, n), q, n, k)
    assert excinfo.value.code == "E-HOLO-024"


def test_refuses_coincident_symbols():
    pool = ak.ExprPool()
    q, n, _k = _syms(pool)
    with pytest.raises(ak.HolonomicError) as excinfo:
        q_zeilberger(n, q, n, n)
    assert excinfo.value.code == "E-HOLO-023"


def test_exhausted_search_is_not_a_negative_answer():
    """``E-HOLO-021`` means "not found within these bounds", not "does not
    exist": the Galois-number sum needs order 2 and is refused at ``max_order=1``
    while being decidable one order up."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    with pytest.raises(ak.HolonomicError) as excinfo:
        q_zeilberger(qbinomial(pool, n, k), q, n, k, max_order=1, max_degree=3)
    assert excinfo.value.code == "E-HOLO-021"
    assert q_zeilberger(qbinomial(pool, n, k), q, n, k, max_order=2).order == 2


def test_invalid_bounds_are_refused():
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    with pytest.raises(ak.HolonomicError) as excinfo:
        q_zeilberger(qbinomial(pool, n, k), q, n, k, max_order=0)
    assert excinfo.value.code == "E-HOLO-023"


# ---------------------------------------------------------------------------
# API shape
# ---------------------------------------------------------------------------


def test_accessors_are_properties_and_repr_is_informative():
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    cert = q_zeilberger(_q_vandermonde_term(pool, q, n, k), q, n, k)
    # Scalars are properties, not bound methods — `if cert.order:` must mean
    # what it looks like it means.
    assert isinstance(cert.order, int)
    assert isinstance(cert.order_is_minimal, bool)
    assert isinstance(cert.probes, int)
    assert isinstance(cert.boundary, str)
    assert isinstance(cert.boundary_reason, str)
    assert isinstance(cert.derivation, str)
    assert isinstance(cert.side_conditions, list)
    assert "QZeilbergerCertificate(order=1" in repr(cert)
    assert "boundary=vanishes" in repr(cert)
