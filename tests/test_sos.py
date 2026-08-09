"""P1 item 8 — positivity certificates (SOS / Positivstellensatz).

`decide` answers real-algebraic questions completely and pays doubly
exponential cost for it. These tests cover the cheaper, certificate-producing
route — and, just as importantly, the boundaries where it must decline instead
of overclaiming. The three outcomes (certified / definitely negative / no
certificate at this degree) must stay distinguishable.
"""

import alkahest as ak
import pytest


def _square(e):
    return e * e


# ---------------------------------------------------------------------------
# Certificates it must find
# ---------------------------------------------------------------------------


def test_perfect_square_is_certified():
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    p = _square(x) - pool.integer(2) * x * y + _square(y)  # (x - y)^2

    cert = ak.sos_decompose(p, [x, y])

    assert cert.kind == "sos"
    assert cert.verify()
    assert cert.num_squares >= 1
    assert " = " in cert.identity  # renders the full identity, not just the RHS


def test_scaled_square_is_certified():
    """``(x/2 + 1/3)²`` — its only Gram matrix is PSD but not diagonally
    dominant, so a plain DD search would refuse a perfect square."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    p = _square(x) / pool.integer(4) + x / pool.integer(3) + pool.rational(1, 9)

    cert = ak.sos_decompose(p, [x])

    assert cert.verify()


def test_certificate_reexpands_to_the_target():
    """The whole point: the identity can be re-checked without trusting us."""
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    p = _square(_square(x)) + _square(_square(y)) + pool.integer(1)

    cert = ak.sos_decompose(p, [x, y])

    difference = ak.simplify(cert.expression - p).value
    assert ak.evaluate(difference, {x: 1.7, y: -0.4}).value == pytest.approx(0.0, abs=1e-12)
    # and exactly, via the certificate's own re-expansion
    assert cert.verify()


def test_box_constrained_positivity_via_handelman():
    """``x − x² ≥ 0`` on ``0 ≤ x ≤ 1``, certified as ``x·(1 − x)``."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    g1 = x
    g2 = pool.integer(1) - x
    p = x - _square(x)

    cert = ak.prove_nonneg(p, [x], constraints=[g1, g2])

    assert cert.kind == "handelman"
    assert cert.verify()
    assert "0 <=" in cert.claim


def test_certificate_exposes_an_audit_trail():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    cert = ak.sos_decompose(_square(x), [x])

    assert cert.log
    assert cert.degree >= 0
    assert "PositivityCertificate(" in repr(cert)


# ---------------------------------------------------------------------------
# The three outcomes must stay distinct
# ---------------------------------------------------------------------------


def test_negative_polynomial_yields_a_witness_not_a_shrug():
    """``x² − 1`` is negative at 0: that is a definite answer, code E-SOS-003."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    p = _square(x) - pool.integer(1)

    with pytest.raises(ak.SosError) as excinfo:
        ak.sos_decompose(p, [x])

    assert excinfo.value.code == "E-SOS-003"
    assert "< 0" in str(excinfo.value)


def test_motzkin_refuses_without_claiming_negativity():
    """The Motzkin polynomial is non-negative but not a sum of squares.

    The one thing the implementation must not do is call it negative. It has to
    come back with E-SOS-002 — "no certificate of this shape" — which is an
    honest statement about the search, not about the polynomial.
    """
    pool = ak.ExprPool()
    x, y = pool.symbol("x"), pool.symbol("y")
    # x^4·y^2 + x^2·y^4 − 3·x^2·y^2 + 1
    p = (
        _square(x) * _square(x) * _square(y)
        + _square(x) * _square(y) * _square(y)
        - pool.integer(3) * _square(x) * _square(y)
        + pool.integer(1)
    )

    with pytest.raises(ak.SosError) as excinfo:
        ak.sos_decompose(p, [x, y])

    assert excinfo.value.code == "E-SOS-002"
    # The remediation must tell an agent that this is not a proof of non-SOS.
    assert "decide" in excinfo.value.remediation


def test_non_polynomial_is_refused():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ak.SosError) as excinfo:
        ak.sos_decompose(ak.sin(x), [x])

    assert excinfo.value.code == "E-SOS-001"


def test_no_variables_is_refused():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ak.SosError) as excinfo:
        ak.sos_decompose(_square(x), [])

    assert excinfo.value.code == "E-SOS-004"


def test_false_constrained_claim_gets_a_witness():
    """``x − 1/2 ≥ 0`` is false on ``x ≥ 0`` — witness, not a refusal."""
    pool = ak.ExprPool()
    x = pool.symbol("x")

    with pytest.raises(ak.SosError) as excinfo:
        ak.prove_nonneg(x - pool.rational(1, 2), [x], constraints=[x])

    assert excinfo.value.code == "E-SOS-003"


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_exported_surface():
    for name in ("sos_decompose", "prove_nonneg", "PositivityCertificate", "SosError"):
        assert name in ak.__all__, name
    assert issubclass(ak.SosError, ak.AlkahestError)
