"""M4 — root-of-unity specialisation of a proved ``q``-Zeilberger certificate.

:func:`alkahest.experimental.q_zeilberger` proves an identity in ``Q(q)`` with
``q`` **transcendental**.  ``QZeilbergerCertificate.specialize_at_root_of_unity``
takes the further step of setting ``q = ζ_d`` for a primitive ``d``-th root of
unity — the step to the ``q``-supercongruence literature — and it takes that
step as a three-valued *decision*
(``"specializes"`` / ``"obstructed"`` / ``"unknown"``) rather than an
assumption: the pole and vanishing hypotheses are decided exactly by
polynomial divisibility by ``Φ_d(q)`` over ``Q``, never numerically.

These tests cover, at the Python surface:

* an end-to-end classical identity, specialised at several roots of unity and
  checked against a value computed **independently of everything under
  test** — a Gaussian binomial built from the Pascal recurrence in raw Python
  ``complex`` arithmetic, evaluated at a floating-point root of unity by
  walking the returned :class:`~alkahest.Expr`'s node tree by hand. Nothing in
  this check touches the Rust cyclotomic-field arithmetic, the specialisation
  machinery, or ``sympy``;
* the two refusals that matter most: a genuine pole at ``ζ_d`` (the
  ``q``-analogue of the A279013 failure mode — a certificate that is perfectly
  valid in ``Q(q)`` and would silently produce a false statement if
  specialised anyway), and a summation window that changes shape under
  specialisation (``q``-Lucas killing terms);
* the degenerate-but-true verdicts (``q → 1``, a leading coefficient that
  dies) that must be reported rather than presented as ordinary content;
* malformed requests, which are coded errors, not verdicts;
* the accessor convention and repr.
"""

import cmath

import alkahest as ak
import pytest
from alkahest.experimental import (
    QRootOfUnitySpecialization,
    cyclotomic_polynomial,
    q_zeilberger,
    qbinomial,
    qpochhammer,
)


def _syms(pool):
    return pool.symbol("q"), pool.symbol("n"), pool.symbol("k")


def _q_vandermonde_term(pool, q, n, k):
    """``[n;k]_q² · q^{k²}`` — the summand of ``Σ_k [n;k]_q²q^{k²} = [2n;n]_q``."""
    b = qbinomial(pool, n, k)
    return b * b * q ** (k * k)


def _vandermonde_cert(pool):
    q, n, k = _syms(pool)
    return q_zeilberger(_q_vandermonde_term(pool, q, n, k), q, n, k), q, n, k


# ---------------------------------------------------------------------------
# An independent numeric yardstick.
#
# `Expr.node()` exposes the expression tree as plain Python data
# (`["add", [child, ...]]`, `["mul", [...]]`, `["pow", base, exp]`,
# `["integer", "3"]`, `["rational", "1", "3"]`, `["symbol", "q"]`). Walking it
# by hand with `complex` arithmetic never touches the Rust cyclotomic field,
# the specialisation machinery, or sympy — an independent check of the exact
# algebra against floating-point evaluation at an actual root of unity.
# ---------------------------------------------------------------------------


def _eval_at(expr, zeta: complex) -> complex:
    tag, *rest = expr.node()
    if tag == "integer":
        return complex(int(rest[0]))
    if tag == "rational":
        return complex(int(rest[0])) / complex(int(rest[1]))
    if tag == "symbol":
        assert rest[0] == "q", f"unexpected symbol {rest[0]!r}"
        return zeta
    if tag == "add":
        return sum((_eval_at(c, zeta) for c in rest[0]), complex(0))
    if tag == "mul":
        acc = complex(1)
        for c in rest[0]:
            acc *= _eval_at(c, zeta)
        return acc
    if tag == "pow":
        base, exp = rest
        if hasattr(exp, "node"):
            exp = round(_eval_at(exp, zeta).real)
        return _eval_at(base, zeta) ** exp
    raise ValueError(f"unhandled node tag {tag!r} in {expr.node()!r}")


def _gaussian_binomial_pascal(zeta: complex, n: int, k: int) -> complex:
    """``[n;k]_ζ`` via the Pascal recurrence ``[n;k] = [n−1;k−1] + ζ^k·[n−1;k]``.

    Deliberately unrelated to the returned certificate, the exact cyclotomic
    arithmetic under test, or the ``q``-Pochhammer expansion the kernel used
    to build the summand: plain floating-point complex arithmetic from the
    first line.
    """
    if k < 0 or k > n:
        return complex(0)
    row = [complex(1)]
    for _i in range(1, n + 1):
        nxt = [complex(0)] * (len(row) + 1)
        for j, cell in enumerate(row):
            nxt[j] += cell * (zeta**j)
            nxt[j + 1] += cell
        row = nxt
    return row[k]


# ---------------------------------------------------------------------------
# The flagship: an end-to-end identity, independently checked.
# ---------------------------------------------------------------------------


def test_q_vandermonde_square_sum_specializes_and_matches_independent_numerics():
    """``Σ_k [n;k]_q²q^{k²} = [2n;n]_q`` at ``q = ζ_d``, for every ``d`` up to 6
    and every ``n`` up to 5: the returned specialised value must equal the
    Gaussian-binomial sum recomputed from scratch in floating-point ``complex``
    arithmetic at the actual numeric root of unity ``e^{2πi/d}``.
    """
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)

    checked = 0
    for d in range(1, 7):
        zeta = cmath.exp(2j * cmath.pi / d)
        for n0 in range(6):
            spec = cert.specialize_at_root_of_unity(d, n0)
            assert spec.status == "specializes", f"d={d}, n={n0}: {spec.reason}"
            assert spec.is_termwise_regular  # every summand is a polynomial in q

            got = _eval_at(spec.sum_value(0), zeta)
            want = sum(
                _gaussian_binomial_pascal(zeta, n0, kk) ** 2 * zeta ** (kk * kk)
                for kk in range(n0 + 1)
            )
            assert abs(got - want) < 1e-6, (
                f"d={d}, n={n0}: got {got}, independently computed {want}"
            )
            checked += 1
    assert checked == 6 * 6


def test_the_specialised_recurrence_matches_independent_numerics_too():
    """Not just the sum values: the specialised **coefficients**, evaluated the
    same independent way, must annihilate the independently-computed sums —
    whenever the recurrence is not vacuous."""
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)

    nontrivial = 0
    for d in range(2, 7):
        zeta = cmath.exp(2j * cmath.pi / d)
        for n0 in range(6):
            spec = cert.specialize_at_root_of_unity(d, n0)
            if not spec.specializes or spec.is_vacuous:
                continue
            nontrivial += 1
            order = cert.order
            acc = complex(0)
            for i in range(order + 1):
                c = _eval_at(spec.coefficient(i), zeta)
                s = sum(
                    _gaussian_binomial_pascal(zeta, n0 + i, kk) ** 2 * zeta ** (kk * kk)
                    for kk in range(n0 + i + 1)
                )
                acc += c * s
            assert abs(acc) < 1e-6, (
                f"d={d}, n={n0}: specialised recurrence does not annihilate the independent values"
            )
    assert nontrivial >= 10, "the check must not be vacuously empty"


# ---------------------------------------------------------------------------
# Degeneracy — true but reported rather than hidden.
# ---------------------------------------------------------------------------


def test_classical_limit_q_to_one_is_vacuous_but_values_are_correct():
    """At ``d = 1`` (``q → 1``) every coefficient carries a factor of
    ``(1 − q)``, so the specialised recurrence is ``0 = 0``. The verdict is
    still ``"specializes"`` and the *values* are the classical central
    binomial coefficients, but ``is_vacuous`` must say the recurrence itself
    is empty."""
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)

    for n0 in range(6):
        spec = cert.specialize_at_root_of_unity(1, n0)
        assert spec.status == "specializes"
        assert spec.is_vacuous
        assert not spec.leading_coefficient_survives
        assert any("VACUOUS" in s for s in spec.side_conditions)

        # C(2n, n), computed independently.
        c = 1
        for j in range(n0):
            c = c * (2 * n0 - j) // (j + 1)
        got = _eval_at(spec.sum_value(0), complex(1))
        assert abs(got - c) < 1e-9


def test_a_root_of_unity_can_kill_the_leading_coefficient():
    """At ``d = 2, n = 1`` the leading coefficient dies while the recurrence
    is not vacuous — a true statement that is not a usable recurrence, and
    ``leading_coefficient_survives`` must say so."""
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)

    spec = cert.specialize_at_root_of_unity(2, 1)
    assert spec.status == "specializes"
    assert not spec.is_vacuous
    assert not spec.leading_coefficient_survives
    assert any("leading coefficient" in s for s in spec.side_conditions)


def test_the_support_shrinks_at_a_root_of_unity_and_says_so():
    """``q``-Lucas kills terms: ``[2;1]_q = 1 + q`` is non-zero in ``Q(q)`` and
    zero at ``ζ_2``, so at ``d = 2, n = 2`` the effective window is ``{0, 2}``
    and not ``{0, 1, 2}`` — and this must be reported, not silently absorbed."""
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)

    spec = cert.specialize_at_root_of_unity(2, 2)
    assert spec.status == "specializes"
    assert spec.window == (0, 2)
    assert spec.effective_support == [0, 2]
    assert spec.support_shrinks
    assert any("support shrank" in s for s in spec.side_conditions)

    # The value is still correct: [4;2]_zeta_2 = C(2,1)*[0;0] = 2.
    got = _eval_at(spec.sum_value(0), complex(-1))
    assert abs(got - 2) < 1e-9


# ---------------------------------------------------------------------------
# Refusals — the part that must actually be reachable, not merely claimed.
# ---------------------------------------------------------------------------


def test_a_pole_at_the_root_of_unity_is_obstructed_not_specialized():
    """The A279013 hazard, transplanted: a certificate that is perfectly valid
    in ``Q(q)`` (it re-checks cleanly) carries a summand with a genuine pole at
    ``ζ_3``. Specialising it anyway would produce a confidently wrong
    statement, so the verdict must be ``"obstructed"`` and no specialised
    value may be offered — at every ``n`` tested, not just one."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    b = qbinomial(pool, n, k)
    # (q^3; q^3)_1 = 1 - q^3, a constant q-Pochhammer factor with a pole at zeta_3.
    pole = qpochhammer(pool, 3, 3, 1) ** pool.integer(-1)
    term = b * b * q ** (k * k) * pole
    cert = q_zeilberger(term, q, n, k)
    assert cert.boundary == "vanishes"

    obstructed = 0
    for n0 in range(7):
        spec = cert.specialize_at_root_of_unity(3, n0)
        assert spec.status == "obstructed", f"n={n0}: must be obstructed at zeta_3"
        assert not spec.specializes
        assert "valuation" in spec.reason
        assert any("obstructed" in s for s in spec.side_conditions)
        with pytest.raises(ValueError):
            spec.sum_value(0)
        with pytest.raises(ValueError):
            spec.coefficient(0)
        # The valuation is still available for both shifts -- one of them is
        # exactly the exhibited obstruction (S(n0) or S(n0+1), depending on
        # n0 mod 3), and neither shift's pole is ever hidden.
        valuations = [spec.sum_valuation(i) for i in range(cert.order + 1)]
        assert any(v is not None and v < 0 for v in valuations), (n0, valuations)
        obstructed += 1
    assert obstructed == 7

    # The same certificate at a d where the factor is a unit specialises fine:
    # the refusal is about zeta_3, not about the term in general.
    spec = cert.specialize_at_root_of_unity(5, 2)
    assert spec.status == "specializes", spec.reason


def test_an_unknown_generic_verdict_stays_unknown_at_a_root_of_unity():
    """A certificate whose *generic* boundary verdict is ``"unknown"`` has no
    proved ``Q(q)`` statement to specialise, so the root-of-unity verdict must
    stay ``"unknown"`` too — never silently promoted to a claim."""
    pool = ak.ExprPool()
    q, n, k = _syms(pool)
    term = qpochhammer(pool, 1, 1, n - k) ** pool.integer(-1)
    cert = q_zeilberger(term, q, n, k)
    assert cert.boundary == "unknown"

    spec = cert.specialize_at_root_of_unity(3, 2)
    assert spec.status == "unknown"
    assert not spec.specializes
    assert "already" in spec.reason


def test_malformed_requests_are_coded_errors():
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)

    with pytest.raises(ak.HolonomicError) as excinfo:
        cert.specialize_at_root_of_unity(0, 3)
    assert excinfo.value.code == "E-HOLO-023"

    with pytest.raises(ak.HolonomicError) as excinfo:
        cert.specialize_at_root_of_unity(100_000, 3)
    assert excinfo.value.code == "E-HOLO-023"

    with pytest.raises(ak.HolonomicError) as excinfo:
        cert.specialize_at_root_of_unity(3, -1)
    assert excinfo.value.code == "E-HOLO-023"


# ---------------------------------------------------------------------------
# The modulus, cyclotomic_polynomial, and API shape.
# ---------------------------------------------------------------------------


def test_cyclotomic_polynomial_matches_the_classical_table():
    """Checked via the same independent node-walker, at points that pin down
    each small-degree integer polynomial (e.g. a monic quadratic is
    determined by its values at two points)."""
    pool = ak.ExprPool()

    # Phi_1 = q - 1: Phi_1(0) = -1, Phi_1(2) = 1.
    phi1 = cyclotomic_polynomial(pool, 1)
    assert _eval_at(phi1, complex(0)) == complex(-1)
    assert _eval_at(phi1, complex(2)) == complex(1)
    # Phi_2 = q + 1.
    phi2 = cyclotomic_polynomial(pool, 2)
    assert _eval_at(phi2, complex(0)) == complex(1)
    # Phi_3 = q^2 + q + 1.
    phi3 = cyclotomic_polynomial(pool, 3)
    assert _eval_at(phi3, complex(0)) == complex(1)
    assert _eval_at(phi3, complex(1)) == complex(3)
    # Phi_4 = q^2 + 1.
    phi4 = cyclotomic_polynomial(pool, 4)
    assert _eval_at(phi4, complex(0)) == complex(1)
    assert _eval_at(phi4, complex(1)) == complex(2)


def test_cyclotomic_polynomial_rejects_out_of_range_orders():
    pool = ak.ExprPool()
    with pytest.raises(ValueError):
        cyclotomic_polynomial(pool, 0)
    with pytest.raises(ValueError):
        cyclotomic_polynomial(pool, 100_000)


def test_the_modulus_is_exposed_for_independent_checking():
    """``spec.modulus()`` must be exactly ``Φ_d(q)`` — the same polynomial
    :func:`cyclotomic_polynomial` returns — so a caller can redo the whole
    divisibility check by hand."""
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)
    spec = cert.specialize_at_root_of_unity(6, 3)

    modulus = spec.modulus()
    phi6 = cyclotomic_polynomial(pool, 6)
    zeta6 = cmath.exp(2j * cmath.pi / 6)
    # Independent numeric check that the two polynomials agree (both should
    # be q^2 - q + 1): compare at two points, which pins down a quadratic
    # given it is known monic of degree 2.
    for point in (complex(0), complex(2), complex(-1)):
        assert abs(_eval_at(modulus, point) - _eval_at(phi6, point)) < 1e-9
    assert abs(_eval_at(modulus, zeta6)) < 1e-9  # Phi_6(zeta_6) = 0


def test_accessors_are_properties_and_repr_is_informative():
    pool = ak.ExprPool()
    cert, _q, _n, _k = _vandermonde_cert(pool)
    spec = cert.specialize_at_root_of_unity(3, 2)

    assert isinstance(spec, QRootOfUnitySpecialization)
    # Zero-arg O(1) scalars are properties, not bound methods.
    assert isinstance(spec.d, int)
    assert isinstance(spec.n, int)
    assert isinstance(spec.status, str)
    assert isinstance(spec.specializes, bool)
    assert isinstance(spec.reason, str)
    assert isinstance(spec.is_vacuous, bool)
    assert isinstance(spec.leading_coefficient_survives, bool)
    assert isinstance(spec.is_termwise_regular, bool)
    assert isinstance(spec.support_shrinks, bool)
    assert isinstance(spec.side_conditions, list)
    assert spec.window is None or isinstance(spec.window, tuple)
    assert isinstance(spec.effective_support, list)

    r = repr(spec)
    assert "QRootOfUnitySpecialization(d=3, n=2" in r
    assert "specializes" in r
