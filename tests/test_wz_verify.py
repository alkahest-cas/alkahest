"""N7 -- ``verify_wz_pair`` is a tri-state decision, not a "could not simplify" flag.

The report's valid pair ``F = C(n,k)^2 / C(2n,n)`` used to come back ``False``
because the check was a structural comparison after ``simplify``, which cannot
cancel ratios of gamma values.  It now divides by ``F`` and decides the
identity exactly in ``Q(n)(k)``.  ``True`` means proved, ``False`` means
refuted, and an undecidable input raises ``SumError`` (``E-SUM-001``).
"""

import random

import alkahest as ak
import pytest

sp = pytest.importorskip("sympy")


@pytest.fixture
def nk():
    pool = ak.ExprPool()
    return pool, pool.symbol("n"), pool.symbol("k")


def C(pool, a, b):
    one = pool.integer(1)
    return ak.gamma(a + one) / (ak.gamma(b + one) * ak.gamma(a - b + one))


def _residual_zero(F, G, pts):
    """sympy oracle: F(n+1,k)-F(n,k) == G(n,k+1)-G(n,k) at every point."""
    for nv, kv in pts:
        nv, kv = sp.Integer(nv), sp.Integer(kv)
        lhs = F(nv + 1, kv) - F(nv, kv)
        rhs = G(nv, kv + 1) - G(nv, kv)
        if sp.simplify(lhs - rhs) != 0:
            return False
    return True


def _points(seed, count=12):
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        n = rng.randint(3, 12)
        k = rng.randint(1, n - 2)
        out.append((n, k))
    return out


def _squares_pair(pool, n, k):
    one = pool.integer(1)
    two = pool.integer(2)
    F = C(pool, n, k) ** 2 / C(pool, two * n, n)
    R = (
        k**2
        * (two * k - pool.integer(3) * n - pool.integer(3))
        / (two * (n + one - k) ** 2 * (two * n + one))
    )
    return F, R * F


def _squares_sympy():
    def F(n, k):
        return sp.binomial(n, k) ** 2 / sp.binomial(2 * n, n)

    def G(n, k):
        return k**2 * (2 * k - 3 * n - 3) / (2 * (n + 1 - k) ** 2 * (2 * n + 1)) * F(n, k)

    return F, G


def test_report_pair_is_proved(nk):
    pool, n, k = nk
    F, G = _squares_pair(pool, n, k)
    assert ak.verify_wz_pair(F, G, n, k) is True
    # the oracle agrees at random integer points
    assert _residual_zero(*_squares_sympy(), _points(1))


def test_corrupted_pairs_are_refuted(nk):
    pool, n, k = nk
    F, G = _squares_pair(pool, n, k)
    two = pool.integer(2)
    for bad in (two * G, G + k, -G, pool.integer(0) * G):
        assert ak.verify_wz_pair(F, bad, n, k) is False
    # and the oracle confirms the corrupted G genuinely fails
    Fs, Gs = _squares_sympy()
    assert not _residual_zero(Fs, lambda a, b: 2 * Gs(a, b), _points(2))


def test_zeilberger_certificate_gives_wz_pair_for_binomial_sum(nk):
    """sum_k C(n,k) = 2^n: build the WZ pair from alkahest's own certificate."""
    pool, n, k = nk
    cert = ak.zeilberger(C(pool, n, k), n, k)
    assert cert.order == 1
    a0, a1 = cert.coeffs
    assert str(a0) == "-2"
    assert str(a1) == "1"
    two = pool.integer(2)
    Fw = C(pool, n, k) / two**n  # F / h(n) with h(n+1)/h(n) = 2
    Gw = cert.certificate * C(pool, n, k) / two ** (n + pool.integer(1))
    assert ak.verify_wz_pair(Fw, Gw, n, k) is True
    # corrupted: wrong normalisation of G
    assert ak.verify_wz_pair(Fw, Gw * two, n, k) is False
    assert ak.verify_wz_pair(Fw, Gw + k, n, k) is False

    def Fs(a, b):
        return sp.binomial(a, b) / sp.Integer(2) ** a

    def Gs(a, b):
        return b / (b - a - 1) * sp.binomial(a, b) / sp.Integer(2) ** (a + 1)

    # k * (k - n - 1)^-1 is the certificate (sign included by construction)
    assert _residual_zero(Fs, Gs, _points(3))


def test_polynomial_and_rational_pairs_unchanged(nk):
    pool, n, k = nk
    one, two = pool.integer(1), pool.integer(2)
    assert ak.verify_wz_pair(n * k, k * (k - one) / two, n, k) is True
    assert ak.verify_wz_pair(n * k, k * (k - one), n, k) is False


def test_pair_with_shifted_gammas_normalises(nk):
    """F = n! k!-free example whose G carries gamma ratios that must cancel."""
    pool, n, k = nk
    one = pool.integer(1)
    # F = 1/(n+k+1)!... pick F = n!/ (n+k)! style: pair is built so the check
    # needs Gamma(n+k+2)/Gamma(n+k+1) -> n+k+1 cancellation.
    F = ak.gamma(n + one) * ak.gamma(k + one) / ak.gamma(n + k + pool.integer(2))
    # Beta-like term: F(n,k) = B(n+1,k+1).  F(n+1,k)+F(n,k+1) = F(n,k); take
    # G so the pair is NOT a WZ pair (G = F): residual is nonzero => refuted.
    assert ak.verify_wz_pair(F, F, n, k) is False


def test_undecided_raises_coded_error(nk):
    _pool, n, k = nk
    s = ak.sin(k)
    with pytest.raises(ak.SumError) as ei:
        ak.verify_wz_pair(s, s, n, k)
    assert ei.value.code == "E-SUM-001"
    assert "undecided" in str(ei.value)


def test_docstring_present():
    assert ak.verify_wz_pair.__doc__
    assert "WZ" in ak.verify_wz_pair.__doc__


def test_random_scalings_of_g_are_refuted(nk):
    """Any G -> c*G with c != 1 breaks the identity (F(n+1,k)-F(n,k) != 0)."""
    pool, n, k = nk
    F, G = _squares_pair(pool, n, k)
    rng = random.Random(7)
    for _ in range(6):
        c = rng.choice([-3, -2, 2, 3, 5, 7])
        assert ak.verify_wz_pair(F, pool.integer(c) * G, n, k) is False
