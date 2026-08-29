"""The three derivatives that used to be missing: ``gamma``, ``digamma`` and
``EllipticPi`` — plus ``trigamma``, the new primitive ``digamma`` needs.

Why this matters beyond convenience: the integrator's verification gate checks
``d/dx F = f``, and it cannot check an antiderivative containing a function it
cannot differentiate.  ``gamma`` and ``EllipticPi`` are both emitted, so any
result carrying one was unverifiable.

``trigamma`` is where the polygamma ladder deliberately stops.  ``ψ₁′ = ψ₂``,
``ψ₂′ = ψ₃``, … has no closed-form terminator short of a binary
``polygamma(n, x)``, so *some* rung must decline; this one does, loudly
(``E-DIFF-001``), rather than returning a placeholder.  Moving the boundary
from ``ψ₀`` to ``ψ₁`` is what buys ``Γ′ = Γψ`` and ``Γ″ = Γ(ψ² + ψ₁)`` landing
on functions the gate can evaluate and bound.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest

PI2 = math.pi * math.pi


@pytest.fixture
def pool():
    p = ak.ExprPool()
    with ak.context(pool=p):
        yield p


@pytest.fixture
def x(pool):
    return ak.symbol("x", pool=pool)


def enclose(expr, var, at, **kw):
    """Rigorous enclosure of `expr` at the single point `at`."""
    r = ak.bound_on_box(expr, [(var, at, at)], **kw)
    return r.lower, r.upper


def contains(bounds, want, slack=1e-12):
    lo, hi = bounds
    return lo - slack <= want <= hi + slack


# ---------------------------------------------------------------------------
# Γ′ = Γ·ψ  and  ψ′ = ψ₁
# ---------------------------------------------------------------------------


def test_gamma_now_differentiates(x):
    d = ak.diff(ak.gamma(x), x).value
    s = str(d)
    assert "gamma" in s
    assert "digamma" in s
    # Γ′(2) = Γ(2)·ψ(2) = 1·(1 − γ) = 0.4227843351
    assert contains(enclose(d, x, 2.0), 0.42278433509846713, slack=1e-9)


def test_digamma_now_differentiates(x):
    d = ak.diff(ak.digamma(x), x).value
    assert "trigamma" in str(d)
    # ψ′(1) = ψ₁(1) = π²/6
    assert contains(enclose(d, x, 1.0), PI2 / 6.0, slack=1e-12)


def test_trigamma_declines_rather_than_guessing(x):
    """A future ``polygamma(n, x)`` should flip this assertion, not delete
    it."""
    with pytest.raises(Exception) as exc:
        ak.diff(ak.trigamma(x), x)
    assert "E-DIFF-001" in str(exc.value)
    # …but it evaluates, which is what Γ″ = Γ(ψ² + ψ₁) needs of it.
    assert contains(enclose(ak.trigamma(x), x, 1.0), PI2 / 6.0)


# ---------------------------------------------------------------------------
# EllipticPi in all three arguments
# ---------------------------------------------------------------------------


def test_elliptic_pi_differentiates_in_all_three_arguments(pool):
    """Before this change only ``∂/∂φ`` existed, and even that rule bailed out
    entirely as soon as ``n`` or ``m`` depended on the differentiation
    variable — so ``diff(Π(n(x), φ, m), x)`` failed with ``E-DIFF-001``."""
    n = ak.symbol("n", pool=pool)
    phi = ak.symbol("phi", pool=pool)
    m = ak.symbol("m", pool=pool)
    e = ak.elliptic_pi(n, phi, m)
    for var in (n, phi, m):
        d = ak.diff(e, var).value
        assert d is not None


def test_elliptic_pi_phi_partial_value(pool):
    """∂/∂φ Π(n; φ | m) = 1/((1 − n sin²φ)·√(1 − m sin²φ))."""
    phi = ak.symbol("phi", pool=pool)
    e = ak.elliptic_pi(pool.rational(1, 4), phi, pool.rational(1, 2))
    d = ak.diff(e, phi)
    v = 0.6
    s2 = math.sin(v) ** 2
    want = 1.0 / ((1.0 - 0.25 * s2) * math.sqrt(1.0 - 0.5 * s2))
    assert abs(ak.eval_expr(d, {phi: v}) - want) < 1e-9


# ---------------------------------------------------------------------------
# Trigamma itself
# ---------------------------------------------------------------------------


def test_trigamma_is_exported_and_displays():
    assert "trigamma" in ak.__all__


def test_trigamma_display(x):
    assert ak.unicode_str(ak.trigamma(x)) == "ψ₁(x)"
    assert ak.latex(ak.trigamma(x)) == r"\psi_1\!\left(x\right)"


def test_trigamma_closed_forms(x):
    """Abramowitz & Stegun §6.4: ``ψ₁(1) = π²/6``, ``ψ₁(½) = π²/2``, and the
    recurrence ``ψ₁(x+1) = ψ₁(x) − 1/x²``."""
    assert contains(enclose(ak.trigamma(x), x, 1.0), PI2 / 6.0)
    assert contains(enclose(ak.trigamma(x), x, 0.5), PI2 / 2.0)
    assert contains(enclose(ak.trigamma(x), x, 2.0), PI2 / 6.0 - 1.0)


def test_trigamma_refuses_at_and_below_its_poles(x):
    """Double poles at every non-positive integer.  As with ``digamma``, the
    strips *between* the negative poles are analytic but not covered — the
    reflection formula is in the ``f64`` kernel, not in the Taylor rule."""
    for lo, hi in [(-0.5, 0.5), (0.0, 1.0), (-3.0, -2.0)]:
        with pytest.raises(Exception):
            ak.bound_on_box(ak.trigamma(x), [(x, lo, hi)])


def test_trigamma_capabilities(x):
    rows = {r["name"]: r for r in ak.capabilities()["primitives"]}
    assert rows["trigamma"]["numeric_f64"]
    assert rows["trigamma"]["numeric_ball"]
    assert rows["trigamma"]["taylor_model"]
    # Deliberate: the ladder stops here.
    assert not rows["trigamma"]["diff_forward"]
    for name in ("gamma", "digamma"):
        assert rows[name]["diff_forward"], name
        assert rows[name]["diff_reverse"], name
    assert ak.bounds_supported(ak.trigamma(x)).supported
