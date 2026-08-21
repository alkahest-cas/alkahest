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


def test_motzkin_certifies_via_a_multiplier():
    """The Motzkin polynomial is non-negative but not a sum of squares
    *itself* — the textbook example (Hilbert 1888) of that phenomenon.
    Multiplying by (x^2+y^2) is classically known to fix it (Reznick's
    theorem), and `sos_decompose` finds that certificate by searching for a
    multiplier automatically: the call succeeds, it does not refuse.
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

    cert = ak.sos_decompose(p, [x, y])

    assert cert.kind == "sos"
    # `verify()` re-expands the identity exactly in Q and confirms it holds —
    # the real soundness argument, not a numeric search's own confidence.
    assert cert.verify() is True
    # The identity is `(target) * (multiplier) = rhs` for a multiplier
    # certificate, which `identity` renders as two parenthesised factors on
    # the left of `=` — distinct from a direct certificate's plain `target =
    # rhs`. This is the only Python-level signal that a multiplier was used
    # (there is no separate `.multiplier` accessor at this surface).
    assert cert.identity.count("=") == 1
    lhs, _rhs = cert.identity.split("=", 1)
    assert lhs.strip().startswith("(")
    assert lhs.count(")") >= 2


def test_homogeneous_motzkin_certifies_at_multiplier_power_one():
    """The *homogeneous* ternary Motzkin form ``x⁴y² + x²y⁴ − 3x²y²z² + z⁶``.

    Multiplying by ``σ = x²+y²+z²`` makes it a sum of squares — that identity
    is why Motzkin is the standard example of a PSD form that is not itself
    SOS::

        σ·M = (½x³y+xy³−3⁄2xyz²)² + ¾(x³y−xyz²)² + (xy²z−xz³)²
              + (x²yz−yz³)² + (x²y²−z⁴)²

    Alkahest used to refuse this and, worse, recorded the refusal as a
    mathematical fact ("not classically expected to be SOS at N = 1"). It was
    a missing half-Newton-polytope reduction in the Gram-basis construction.
    Asserted from Python as well as from Rust because the false claim reached
    the user-facing documentation, and the user-facing entry point is here.
    """
    pool = ak.ExprPool()
    x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
    m = (
        _square(x) * _square(x) * _square(y)
        + _square(x) * _square(y) * _square(y)
        - pool.integer(3) * _square(x) * _square(y) * _square(z)
        + _square(z) * _square(z) * _square(z)
    )

    cert = ak.sos_decompose(m, [x, y, z])

    assert cert.kind == "sos"
    # The exact re-expansion in Q is the soundness argument; the numeric
    # search only ever proposed the Gram matrix.
    assert cert.verify() is True
    # A multiplier was needed (Motzkin is not itself SOS), which at this
    # surface shows as a two-factor left-hand side in the identity.
    lhs, _rhs = cert.identity.split("=", 1)
    assert lhs.strip().startswith("(")
    assert lhs.count(")") >= 2


@pytest.mark.slow
def test_a_refusal_says_whether_it_actually_searched():
    """``E-SOS-002`` covers "searched and found nothing" *and* "never looked".

    Marked ``slow`` (~45 s: the Horn form's direct PSD search runs to
    exhaustion before any multiplier power is considered). The same
    assertion is made in the default CI tier by the Rust test
    ``real::sos::tests::a_refusal_reports_which_multiplier_powers_were_actually_searched``;
    this one exists because the false claim it guards against reached the
    *user-facing* surface, and this is that surface.

    The Horn form (copositivity of the Horn matrix) is the case where the
    difference bites: its ``N = 1`` multiplier family has 420 free
    parameters, over the numeric search's ceiling, so no multiplier power is
    searched at all. The refusal is legitimate; presenting it as an exhausted
    search would not be. The message must carry the trace that distinguishes
    them.
    """
    pool = ak.ExprPool()
    v = [pool.symbol(f"h{i}") for i in range(5)]
    h = [
        [1, -1, 1, 1, -1],
        [-1, 1, -1, 1, 1],
        [1, -1, 1, -1, 1],
        [1, 1, -1, 1, -1],
        [-1, 1, 1, -1, 1],
    ]
    p = pool.integer(0)
    for i in range(5):
        for j in range(5):
            p = p + pool.integer(h[i][j]) * _square(v[i]) * _square(v[j])

    with pytest.raises(ak.SosError) as excinfo:
        ak.sos_decompose(p, v)

    assert excinfo.value.code == "E-SOS-002"
    msg = str(excinfo.value)
    assert "what the search actually did:" in msg
    assert "NOT SEARCHED" in msg


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
