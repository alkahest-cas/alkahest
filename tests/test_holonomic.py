"""P1 item 7 — creative telescoping / holonomic (D-finite) machinery.

Zeilberger's algorithm is a *decision procedure that emits a certificate*: for a
proper hypergeometric term ``F(n, k)`` it either returns a recurrence together
with a rational certificate that has been re-checked as an exact identity, or it
refuses.  These tests cover both halves — the classical identities it must
decide, and the boundaries at which it must refuse rather than guess.
"""

import math
import time

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


def test_order_two_identity_is_decided_at_the_default_bounds():
    """``Σ_k (−1)^k C(n,k)³`` (Dixon) at the *shipped defaults*.

    Regression test for the search strategy.  ``max_order`` / ``max_degree`` are
    upper bounds, so an identity that needs order 2 must be found at whatever
    bound the caller passes, at the cost of the order-2 search — not at the cost
    of the whole degree sweep at order 1, where no relation exists.  When that
    was the search order, every order ≥ 2 identity (Dixon, Franel, Apéry) was
    unreachable at the defaults while taking seconds at ``max_degree=4``.

    The wall-clock bound is deliberately loose: the failure mode it guards is
    minutes-to-never, not a few hundred milliseconds either way.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    binom = _binomial(pool, n, k)
    term = pool.integer(-1) ** k * binom * binom * binom

    started = time.monotonic()
    cert = ak.zeilberger(term, n, k)  # default max_order / max_degree
    elapsed = time.monotonic() - started

    assert cert.order == 2
    assert elapsed < 60.0, f"took {elapsed:.1f}s at the default bounds"

    def s(m):
        return sum((-1) ** j * math.comb(m, j) ** 3 for j in range(m + 1))

    for ni in range(2, 8):
        a = _eval_coeffs(cert, n, ni)
        total = sum(a[i] * s(ni + i) for i in range(len(a)))
        scale = max([1.0] + [abs(a[i] * s(ni + i)) for i in range(len(a))])
        assert total == pytest.approx(0.0, abs=1e-6 * scale)


def test_franel_is_decided_quickly_at_the_default_bounds():
    """``Σ_k C(n,k)³`` (Franel) at the shipped defaults, under a wall clock.

    Regression test for the *post-search* cost, which is a different failure
    from the one above: the search reaches (order 2, degree 3) in a fifth of a
    second, and the exact ``Q(n)(k)`` work that follows it — normalising the
    certificate and re-verifying the identity — used to take ~29 s, all of it a
    Euclidean remainder sequence over ``Q(n)`` swelling its own coefficients.
    Doing that gcd in ``Z[n][k]`` instead makes it milliseconds.

    Nothing about verification is relaxed to get there: the certificate is still
    checked as an exact identity before it is returned, which is why this test
    also re-checks the recurrence against the actual sum.  The bound is loose on
    purpose — it guards a two-orders-of-magnitude regression, not jitter.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    binom = _binomial(pool, n, k)
    term = binom * binom * binom

    started = time.monotonic()
    cert = ak.zeilberger(term, n, k)  # default max_order / max_degree
    elapsed = time.monotonic() - started

    assert cert.order == 2
    assert elapsed < 10.0, f"Franel took {elapsed:.1f}s at the default bounds"

    def s(m):
        return sum(math.comb(m, j) ** 3 for j in range(m + 1))

    for ni in range(2, 8):
        a = _eval_coeffs(cert, n, ni)
        total = sum(a[i] * s(ni + i) for i in range(len(a)))
        scale = max([1.0] + [abs(a[i] * s(ni + i)) for i in range(len(a))])
        assert total == pytest.approx(0.0, abs=1e-6 * scale)


def test_apery_returns_aperys_recurrence():
    """``Σ_k C(n,k)²·C(n+k,k)²`` must reproduce Apéry's recurrence exactly.

    ``(n+1)³·S(n) − (34n³+153n²+231n+117)·S(n+1) + (n+2)³·S(n+2) = 0`` — the
    recurrence behind the irrationality of ζ(3).  It is asserted coefficient by
    coefficient (not just "some verified relation"), because this is the value
    that pins the ``Q(n)`` normalisation down to one representative: a faster
    gcd that returned a different multiple would show up right here.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    binom_nk = _binomial(pool, n, k)
    binom_npk = _binomial(pool, n + k, k)
    term = binom_nk * binom_nk * binom_npk * binom_npk

    started = time.monotonic()
    cert = ak.zeilberger(term, n, k)
    elapsed = time.monotonic() - started

    assert cert.order == 2
    assert elapsed < 10.0, f"Apéry took {elapsed:.1f}s at the default bounds"

    def want(ni):
        return [
            (ni + 1) ** 3,
            -(34 * ni**3 + 153 * ni**2 + 231 * ni + 117),
            (ni + 2) ** 3,
        ]

    for ni in (2.0, 5.0, 9.0):
        got = _eval_coeffs(cert, n, ni)
        expected = want(ni)
        # Coefficients are only defined up to one common scale.
        scale = got[2] / expected[2]
        assert scale != 0.0
        for g, e in zip(got, expected):
            assert g == pytest.approx(e * scale, rel=1e-12)


def test_raising_the_bounds_does_not_slow_down_an_easy_input():
    """Bounds are bounds: a wider search must not cost more on an easy term.

    ``Σ_k C(n,k)`` is decided at order 1, degree 0.  Quadrupling both bounds
    used to multiply the work (the old search started its degree sweep at the
    bound); now it must be free, so the two calls stay within a small factor of
    each other and return the same order.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    term = _binomial(pool, n, k)

    started = time.monotonic()
    tight = ak.zeilberger(term, n, k, max_order=1, max_degree=2)
    tight_elapsed = time.monotonic() - started

    started = time.monotonic()
    wide = ak.zeilberger(term, n, k, max_order=4, max_degree=64)
    wide_elapsed = time.monotonic() - started

    assert wide.order == tight.order == 1
    assert wide_elapsed < max(5.0, 20.0 * tight_elapsed), (
        f"wide bounds cost {wide_elapsed:.3f}s vs {tight_elapsed:.3f}s tight"
    )


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


def _a357558_summand(pool, n, k, minus_one):
    """``(−1)^(n+k)·k·C(n,k)·C(n+k,k)²`` — OEIS A357558's summand.

    ``minus_one`` is the caller's spelling of the constant −1.
    """
    binom_nk = _binomial(pool, n, k)
    binom_npk = _binomial(pool, n + k, k)
    return minus_one ** (n + k) * k * binom_nk * binom_npk * binom_npk


def _cert_text(cert):
    return (str(cert.certificate), [str(c) for c in cert.coeffs])


def test_constant_base_is_folded_not_refused():
    """``(-one)**(n+k)`` must work exactly like ``pool.integer(-1)**(n+k)``.

    The pool does no arithmetic at construction, so ``-one`` (with
    ``one = pool.integer(1)``) is the node ``Mul(1, -1)``, not the literal −1.
    The hypergeometric parser used to demand a *literal* rational base and
    refused this as E-HOLO-001 "not a proper hypergeometric term" — a refusal
    that reads as a capability limit even though ``1 * -1`` is the rational −1.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    one = pool.integer(1)

    folded = ak.zeilberger(_a357558_summand(pool, n, k, -one), n, k, max_order=3)
    literal = ak.zeilberger(_a357558_summand(pool, n, k, pool.integer(-1)), n, k, max_order=3)

    assert folded.order == literal.order == 2
    assert _cert_text(folded) == _cert_text(literal)


def test_a357558_recurrence_is_satisfied_by_the_actual_sum():
    """End-to-end on the OEIS target that this refusal made look impossible."""
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    one = pool.integer(1)

    started = time.monotonic()
    cert = ak.zeilberger(_a357558_summand(pool, n, k, -one), n, k, max_order=3)
    elapsed = time.monotonic() - started

    assert cert.order == 2
    assert elapsed < 60.0, f"took {elapsed:.1f}s"

    def s(m):
        return sum(
            (-1) ** (m + j) * j * math.comb(m, j) * math.comb(m + j, j) ** 2 for j in range(m + 1)
        )

    for ni in range(2, 8):
        a = _eval_coeffs(cert, n, ni)
        total = sum(a[i] * s(ni + i) for i in range(len(a)))
        scale = max([1.0] + [abs(a[i] * s(ni + i)) for i in range(len(a))])
        assert total == pytest.approx(0.0, abs=1e-6 * scale)


# ---------------------------------------------------------------------------
# Refusals — a refusal is an informative answer, not a failure
# ---------------------------------------------------------------------------


def test_refuses_symbolic_and_zero_bases_under_a_symbolic_exponent():
    """Folding constants must not widen the accepted class.

    ``x^k`` is genuinely symbolic, and ``(1 - 1)^(n+k)`` folds to ``0`` raised
    to a symbolic power; both are still outside the proper hypergeometric class.
    """
    pool = ak.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    x = pool.symbol("x")
    one = pool.integer(1)

    for term in (x**k, (2 * x) ** k, (one - one) ** (n + k), (one * one - one) ** k):
        with pytest.raises(ak.HolonomicError) as excinfo:
            ak.zeilberger(term * _binomial(pool, n, k), n, k)
        assert excinfo.value.code == "E-HOLO-001"


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
