"""The boundary verdict: whether a Zeilberger certificate says anything about
the *sum*.

Zeilberger's algorithm proves an identity about the summand.  Reading a
recurrence for ``S(n) = Σ_k F(n,k)`` off it is a second claim, and it can be
false while the certificate is perfectly valid — that is how a verified
certificate turns into a false theorem.  ``cert.boundary`` decides it:
``"vanishes"``, ``"nonzero"`` (with ``cert.boundary_rhs``) or ``"unknown"``.

Every assertion below is checked against the actual sequence, in exact rational
arithmetic, with no reference to the certificate machinery: the point of the
feature is that the verdict and the sequence agree.
"""

import math
from fractions import Fraction

import alkahest as ak
import pytest


def _binomial(pool, top, bot):
    """``C(top, bot)`` as a ratio of gammas — the proper hypergeometric form."""
    one = pool.integer(1)
    return ak.gamma(top + one) / (ak.gamma(bot + one) * ak.gamma(top - bot + one))


def _exact(pool, n, expr, ni):
    """``expr`` at ``n = ni`` as an exact ``Fraction``.

    ``gamma`` at a positive integer folds to a rational under ``simplify``, so
    the whole check stays in exact arithmetic — no float ever sees a value that
    a verdict depends on.
    """
    # `simplify` may hand back a parenthesised atom, e.g. "(20)".
    return Fraction(str(ak.simplify(ak.subs(expr, {n: pool.integer(ni)})).value).strip("()"))


def _residual(pool, n, cert, s, ni):
    """``Σ_i a_i(n)·S(n+i)`` at ``n = ni``, exactly."""
    return sum(_exact(pool, n, c, ni) * s(ni + i) for i, c in enumerate(cert.coeffs))


# ---------------------------------------------------------------------------
# The failure this exists for
# ---------------------------------------------------------------------------


def _a279013_summand(pool, n, k):
    """``C(2k,k)/(k+1) · C(2n−1, n−k)`` — OEIS A279013's summand."""
    one, two = pool.integer(1), pool.integer(2)
    return _binomial(pool, two * k, k) / (k + one) * _binomial(pool, two * n - one, n - k)


def _a279013(m):
    return sum(math.comb(2 * j, j) // (j + 1) * math.comb(2 * m - 1, m - j) for j in range(m + 1))


def test_a279013_is_not_reported_as_a_recurrence_for_the_sum():
    """The regression this feature exists for.

    ``a(n) = Σ_{k=0}^{n} C(2k,k)/(k+1)·C(2n−1,n−k)`` gets a **verified** order-2
    certificate in a tenth of a second.  The homogeneous recurrence read off it
    fails against the real sequence at the very first term — the certificate is
    a correct telescoping identity for the summand and the boundary simply does
    not vanish.  Every signal the API offered before 3.9.0 said "proved".

    So: the verdict must not be ``"vanishes"``, and the sequence is what says so.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")

    cert = ak.zeilberger(_a279013_summand(pool, n, k), n, k)

    assert cert.order == 2
    assert cert.boundary != "vanishes", (
        "the homogeneous recurrence is false for A279013 — reporting a vanishing "
        "boundary here is the false-theorem bug"
    )
    assert cert.boundary in ("nonzero", "unknown")

    # The real sequence: 2, 8, 35, 161, 768, 3773, …
    assert [_a279013(m) for m in range(1, 7)] == [2, 8, 35, 161, 768, 3773]

    # The homogeneous recurrence really does fail, at n = 1 and everywhere after.
    for ni in range(1, 5):
        assert _residual(pool, n, cert, _a279013, ni) != 0


def test_a279013_inhomogeneous_recurrence_holds_against_the_real_sequence():
    """``"nonzero"`` is a result, not a refusal: ``b(n)`` must be *right*.

    Checked in exact rational arithmetic against the OEIS terms — the same
    independent check that caught the original defect, now run against the
    inhomogeneity the engine reports.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")

    cert = ak.zeilberger(_a279013_summand(pool, n, k), n, k)
    if cert.boundary != "nonzero":  # pragma: no cover - guarded by the test above
        pytest.skip(f"boundary came back {cert.boundary!r}, nothing to check")

    for ni in range(1, 7):
        lhs = _residual(pool, n, cert, _a279013, ni)
        rhs = _exact(pool, n, cert.boundary_rhs, ni)
        assert lhs == rhs, f"Σ a_i(n)·S(n+i) = b(n) fails at n = {ni}"
    assert cert.implies_sum_recurrence


# ---------------------------------------------------------------------------
# The natural-boundary cases must keep working, and must be proved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "build", "sequence"),
    [
        (
            "franel",
            lambda pool, n, k, b: b * b * b,
            lambda m: sum(math.comb(m, j) ** 3 for j in range(m + 1)),
        ),
        (
            "dixon",
            lambda pool, n, k, b: pool.integer(-1) ** k * b * b * b,
            lambda m: sum((-1) ** j * math.comb(m, j) ** 3 for j in range(m + 1)),
        ),
        (
            "apery",
            lambda pool, n, k, b: b * b * _binomial(pool, n + k, k) * _binomial(pool, n + k, k),
            lambda m: sum(math.comb(m, j) ** 2 * math.comb(m + j, j) ** 2 for j in range(m + 1)),
        ),
    ],
)
def test_natural_boundary_identities_are_proved_to_vanish(name, build, sequence):
    """Franel, Dixon and Apéry over ``k = 0..n``.

    These are the cases where the homogeneous recurrence *is* true, so the
    verdict has to be ``"vanishes"`` — a feature that returned ``"unknown"``
    here would be sound but useless.  The recurrence is then checked against the
    sequence itself, exactly.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    cert = ak.zeilberger(build(pool, n, k, _binomial(pool, n, k)), n, k)

    assert cert.boundary == "vanishes", f"{name}: {cert.boundary_reason}"
    assert cert.boundary_rhs is None
    assert cert.implies_sum_recurrence
    for ni in range(2, 8):
        assert _residual(pool, n, cert, sequence, ni) == 0, f"{name} at n = {ni}"


def test_row_sum_boundary_accounts_for_the_shifted_range():
    """``Σ_{k=0}^{n} C(n,k) = 2ⁿ`` — the case that pins the *whole* formula.

    The telescoped part `G(n,n+1) − G(n,0)` is `−1` here, not `0`; what makes
    the homogeneous recurrence true is that `Σ_{k=0}^{n} C(n+1,k)` is `S(n+1)−1`,
    not `S(n+1)`.  A verdict computed from the endpoints alone reports
    ``"nonzero"`` on a textbook identity.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    cert = ak.zeilberger(_binomial(pool, n, k), n, k)

    assert cert.boundary == "vanishes", cert.boundary_reason
    for ni in range(2, 9):
        assert _residual(pool, n, cert, lambda m: 2**m, ni) == 0


def test_counterexample_summand_gets_the_inhomogeneous_recurrence():
    """``F = C(n,k)/(k+1)``: ``S(n) = (2ⁿ⁺¹−1)/(n+1)``.

    The textbook case where the boundary does not vanish.  The verdict must be
    ``"nonzero"`` — and the returned ``b(n)`` must make the recurrence true.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    cert = ak.zeilberger(_binomial(pool, n, k) / (k + one), n, k)

    assert cert.boundary == "nonzero", cert.boundary_reason
    s = lambda m: Fraction(2 ** (m + 1) - 1, m + 1)  # noqa: E731
    for ni in range(1, 7):
        assert _residual(pool, n, cert, s, ni) == _exact(pool, n, cert.boundary_rhs, ni)


# ---------------------------------------------------------------------------
# The limits are part of the claim
# ---------------------------------------------------------------------------


def _a361712_summand(pool, n, k):
    one = pool.integer(1)
    return (
        _binomial(pool, n, k)
        * _binomial(pool, n, k)
        * _binomial(pool, n + k, k)
        * _binomial(pool, n + k - one, k)
    )


def test_a361712_verdict_follows_the_summation_range():
    """``C(n,k)²·C(n+k,k)·C(n+k−1,k)`` over ``0..n`` and over ``0..n−1``.

    The certificate is the same object; what changes is the range.  Over
    ``k = 0..n`` the boundary vanishes; truncating to ``k = 0..n−1`` — the range
    OEIS's formula field uses — leaves the ``k = n`` term behind and the
    recurrence becomes inhomogeneous.  Both are checked against the two
    sequences.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    cert = ak.zeilberger(_a361712_summand(pool, n, k), n, k)

    def full(m):
        return sum(
            math.comb(m, j) ** 2 * math.comb(m + j, j) * math.comb(m + j - 1, j)
            for j in range(m + 1)
        )

    def truncated(m):
        return sum(
            math.comb(m, j) ** 2 * math.comb(m + j, j) * math.comb(m + j - 1, j) for j in range(m)
        )

    assert cert.boundary == "vanishes", cert.boundary_reason
    for ni in range(2, 8):
        assert _residual(pool, n, cert, full, ni) == 0

    short = cert.boundary_at(0, n - one)
    assert short["boundary"] == "nonzero", short["reason"]
    for ni in range(2, 8):
        assert _residual(pool, n, cert, truncated, ni) == _exact(pool, n, short["rhs"], ni)


def test_limits_are_reported_and_can_be_supplied():
    """The assumed range is echoed back, and passing one overrides it."""
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    f = _binomial(pool, n, k) * _binomial(pool, n, k) * _binomial(pool, n, k)

    default = ak.zeilberger(f, n, k)
    assert [str(x) for x in default.limits] == ["0", "n"]

    short = ak.zeilberger(f, n, k, limits=(0, n - one))
    assert str(short.limits[0]) == "0"
    assert str(short.limits[1]) == str(n - one)
    assert short.boundary == "nonzero"
    # Same certificate, different claim about the sum.
    assert str(short.certificate) == str(default.certificate)


def test_a_range_that_cannot_be_placed_is_unknown_not_vanishing():
    """An upper limit that is not integer-affine in ``n`` is ``"unknown"``.

    Refusing is the whole discipline: the same summand over ``k = 0..n`` is
    proved to vanish, so this is not a capability limit being reported as a
    verdict — it is the verdict for a range the analysis cannot place.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    m = pool.symbol("m")
    f = _binomial(pool, n, k) * _binomial(pool, n, k) * _binomial(pool, n, k)

    assert ak.zeilberger(f, n, k).boundary == "vanishes"

    cert = ak.zeilberger(f, n, k, limits=(0, m))
    assert cert.boundary == "unknown"
    assert not cert.implies_sum_recurrence
    assert cert.boundary_rhs is None
    assert cert.boundary_reason


def test_limits_reject_nonsense():
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    with pytest.raises(TypeError):
        ak.zeilberger(_binomial(pool, n, k), n, k, limits=(0, "n"))


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_side_conditions_track_the_verdict():
    """The defect was never missing information — it was information that did
    not *vary*.  A fixed caveat reads identically for a correct and an incorrect
    case, so nothing a loop reads can tell them apart.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    m = pool.symbol("m")

    proved = ak.zeilberger(_binomial(pool, n, k), n, k)
    refuted = ak.zeilberger(_binomial(pool, n, k) / (k + pool.integer(1)), n, k)
    undecided = ak.zeilberger(_binomial(pool, n, k), n, k, limits=(0, m))

    texts = [c.side_conditions for c in (proved, refuted, undecided)]
    for t in texts:
        assert isinstance(t, list)
        assert t
        assert all(isinstance(s, str) for s in t)
    assert len({tuple(t) for t in texts}) == 3, "the three verdicts must read differently"

    assert "proved" in proved.side_conditions[0]
    assert "FALSE" in refuted.side_conditions[0]
    assert "NOTHING" in undecided.side_conditions[0]
    # The range the verdict is about is named, so a caller summing over
    # something else can see the mismatch.
    assert "k = 0..n" in proved.side_conditions[0]


def test_repr_carries_the_verdict():
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    assert "boundary=vanishes" in repr(ak.zeilberger(_binomial(pool, n, k), n, k))


def test_boundary_at_does_not_disturb_the_certificate():
    """``boundary_at`` answers about another range without re-running anything."""
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    cert = ak.zeilberger(_binomial(pool, n, k), n, k)

    assert cert.boundary_at(0, n)["boundary"] == "vanishes"
    assert cert.boundary_at(0, n - one)["boundary"] == "nonzero"
    # The object's own verdict is unchanged by asking about other ranges.
    assert cert.boundary == "vanishes"
    assert set(cert.boundary_at(0, n)) == {
        "boundary",
        "rhs",
        "reason",
        "valid_from",
        "certificate_poles",
        "side_conditions",
    }
    # The domain travels with the verdict here too, not only on the attribute.
    assert cert.boundary_at(0, n)["valid_from"] == cert.boundary_valid_from
    assert cert.boundary_at(3, n - pool.integer(3))["valid_from"] == 5


# ---------------------------------------------------------------------------
# The domain the verdict is claimed on
# ---------------------------------------------------------------------------


def test_a_backwards_range_is_not_a_recurrence():
    """``k = 5..3`` is empty, so every ``S(n)`` is ``0``.

    The engine used to answer ``"nonzero"`` here with a degree-9 ``b(n)`` whose
    residual ran ``4, 107, 800, 2725, …`` — a valid certificate implying a false
    recurrence, which is the failure the whole verdict exists to prevent.  There
    is no ``n`` at which the relation holds, so there is nothing to claim.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    b = _binomial(pool, n, k)

    for lo, hi in [(5, 3), (3, 1), (2, 0), (4, 2)]:
        cert = ak.zeilberger(b * b, n, k, limits=(lo, hi))
        assert cert.boundary == "unknown", f"k = {lo}..{hi}: {cert.boundary_reason}"
        assert not cert.implies_sum_recurrence
        assert cert.boundary_rhs is None
        # Nothing is claimed at any n, so the bound has nothing to bound.
        assert cert.boundary_valid_from is None
        assert "backwards" in cert.boundary_reason


def test_an_n_dependent_range_that_starts_empty_carries_its_domain():
    """``k = 3..n−3`` is the realistic form: empty at ``n = 3, 4``, a range after.

    The verdict is a theorem for ``n ≥ 5`` and false below it, so it is returned
    *with* that bound rather than discarded or over-claimed.  Both halves are
    checked against the actual sum.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    three = pool.integer(3)
    cert = ak.zeilberger(_binomial(pool, n, k), n, k, limits=(three, n - three))

    assert cert.boundary_valid_from == 5
    assert cert.boundary == "nonzero", cert.boundary_reason
    assert any("n >= 5" in s for s in cert.side_conditions)

    def s(m):
        return sum(math.comb(m, j) for j in range(3, m - 3 + 1))

    # A theorem from n = 5 on ...
    for ni in range(5, 10):
        assert _residual(pool, n, cert, s, ni) == _exact(pool, n, cert.boundary_rhs, ni)
    # ... and false below it, which is exactly what the bound says.
    for ni in (3, 4):
        assert s(ni) == 0
        assert _residual(pool, n, cert, s, ni) != _exact(pool, n, cert.boundary_rhs, ni)


def test_an_exactly_empty_range_is_still_a_proved_zero():
    """``k = 0..−1`` and ``k = n+1..n`` are empty too — and were always right.

    ``κ₁ = κ₀ − 1`` is the one empty range both readings agree is ``0``, and the
    telescoping handles it, so it keeps its ``"vanishes"``.  The old behaviour
    was inconsistent in this pair's favour; the fix must not swap which half is
    wrong.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    b = _binomial(pool, n, k)
    one = pool.integer(1)

    for limits in [(0, -1), (n + one, n)]:
        cert = ak.zeilberger(b * b, n, k, limits=limits)
        assert cert.boundary == "vanishes", cert.boundary_reason
        assert cert.implies_sum_recurrence
        assert cert.boundary_valid_from is None
        assert cert.certificate_poles == []


# ---------------------------------------------------------------------------
# Poles inside the range
# ---------------------------------------------------------------------------


def test_an_interior_certificate_pole_is_reported():
    """``C(n,k)/(n−2k+1)`` over ``k = 0..n``.

    ``S(n)`` is undefined for every odd ``n`` — the summand has a pole at
    ``k = (n+1)/2`` — *and* the certificate has one at ``k = (n+3)/2``, an
    integer strictly inside the range.  The telescoping breaks in the middle of
    the sum, where no boundary value can see it; the verdict was ``"vanishes"``.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    cert = ak.zeilberger(_binomial(pool, n, k) / (n - pool.integer(2) * k + one), n, k)

    assert cert.boundary == "unknown", cert.boundary_reason
    assert not cert.implies_sum_recurrence
    poles = [str(p) for p in cert.certificate_poles]
    assert poles, "the poles must be reported, not just the refusal"
    assert all("1/2" in p for p in poles), poles
    assert any("interior" in s for s in cert.side_conditions)

    # The sum really is undefined at the odd n, which is what the pole says.
    for m in (3, 5, 7):
        assert (m - 2 * ((m + 1) // 2) + 1) == 0


def test_a_summand_pole_inside_the_range_is_reported():
    """``C(n,k)/(k−3)`` over ``k = 0..n``: ``S(n)`` does not exist for ``n ≥ 3``.

    The old answer was ``"vanishes"`` with ``implies_sum_recurrence`` — the same
    strings a genuine theorem gets.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    cert = ak.zeilberger(_binomial(pool, n, k) / (k - pool.integer(3)), n, k)

    assert cert.boundary == "unknown", cert.boundary_reason
    assert not cert.implies_sum_recurrence
    assert [str(p) for p in cert.certificate_poles] == ["3"]


def test_verdicts_that_were_already_right_are_left_alone():
    """Guards, so that refusing empty ranges and interior poles cannot spread.

    ``k = −n..n`` is honestly ``"unknown"``, ``k = −n..0`` is a real
    ``"nonzero"``, and ``C(n,k)/(n−k+1)`` over ``k = 0..n`` has a certificate
    pole *exactly* at ``k = k_hi+1`` that a zero of the summand cancels — a
    ``0·∞`` endpoint the analysis resolves rather than refuses.
    """
    pool = ak.ExprPool()
    n, k = pool.symbol("n"), pool.symbol("k")
    one = pool.integer(1)
    b = _binomial(pool, n, k)

    assert ak.zeilberger(b * b, n, k, limits=(-n, n)).boundary == "unknown"

    half = ak.zeilberger(b, n, k, limits=(-n, 0))
    assert half.boundary == "nonzero", half.boundary_reason

    cont = ak.zeilberger(b / (n - k + one), n, k)
    assert cont.boundary == "nonzero", cont.boundary_reason
    assert cont.certificate_poles == []
    s = lambda m: sum(Fraction(math.comb(m, j), m - j + 1) for j in range(m + 1))  # noqa: E731
    for ni in range(2, 7):
        assert _residual(pool, n, cont, s, ni) == _exact(pool, n, cont.boundary_rhs, ni)
