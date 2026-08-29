"""Fresnel integrals ``S(x)`` and ``C(x)``.

**Convention — normalised (π/2).**  ``S(x) = ∫₀ˣ sin(πt²/2) dt`` and
``C(x) = ∫₀ˣ cos(πt²/2) dt``: DLMF §7.2(iii), Abramowitz & Stegun §7.3, SymPy
``fresnels``/``fresnelc``, SciPy ``scipy.special.fresnel``, Mathematica
``FresnelS``/``FresnelC``.  ``S(∞) = C(∞) = 1/2``.

The competing *unnormalised* ``∫₀ˣ sin(t²) dt`` is a different function — it
tends to ``√(π/8) ≈ 0.6267`` — and silently mixing the two is wrong by a factor
of ``√(π/2)`` *and* evaluated at the wrong point.
``test_fresnel_tends_to_one_half`` is what pins the convention.

Reference values are Abramowitz & Stegun Table 7.7, with the full 16 digits
from ``scipy.special.fresnel`` computed offline; SciPy is not a dependency of
Alkahest and is not imported here.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest


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


# (x, S(x), C(x))
FRESNEL_REFERENCE = [
    (0.5, 0.06473243285999929, 0.4923442258714464),
    (1.0, 0.4382591473903547, 0.779893400376823),
    (2.0, 0.34341567836369824, 0.48825340607534073),
    (3.0, 0.496312998967375, 0.6057207892976857),
    (5.0, 0.49919138191711687, 0.5636311887040122),
    # Past the series/asymptotic switchover at |x| = 6.
    (8.0, 0.46021421439301446, 0.49980218037719715),
    (20.0, 0.4840845359259539, 0.4999873349723444),
]


def test_exported_and_displayed(x):
    for name in ("fresnels", "fresnelc"):
        assert name in ak.__all__
    assert ak.unicode_str(ak.fresnels(x)) == "S(x)"
    assert ak.unicode_str(ak.fresnelc(x)) == "C(x)"
    assert ak.latex(ak.fresnels(x)) == r"S\!\left(x\right)"
    assert ak.latex(ak.fresnelc(x)) == r"C\!\left(x\right)"


@pytest.mark.parametrize(("at", "s", "c"), FRESNEL_REFERENCE)
def test_fresnel_matches_published_values(at, s, c, x):
    assert contains(enclose(ak.fresnels(x), x, at), s), f"S({at})"
    assert contains(enclose(ak.fresnelc(x), x, at), c), f"C({at})"


def test_fresnel_tends_to_one_half(x):
    """The normalisation check: in the *unnormalised* convention both limits
    are ``√(π/8) ≈ 0.6267`` instead."""
    for at in (200.0, 2000.0):
        for f in (ak.fresnels, ak.fresnelc):
            lo, hi = enclose(f(x), x, at)
            assert abs(0.5 * (lo + hi) - 0.5) < 1.0 / at
            assert abs(0.5 * (lo + hi) - math.sqrt(math.pi / 8)) > 0.1


@pytest.mark.parametrize("f", [lambda z: ak.fresnels(z), lambda z: ak.fresnelc(z)])
def test_fresnel_is_odd(f, x):
    for at in (0.7, 3.3, 9.5):
        pos = enclose(f(x), x, at)
        neg = enclose(f(x), x, -at)
        assert contains(neg, -0.5 * (pos[0] + pos[1]), slack=1e-10)


def test_fresnel_is_continuous_across_the_algorithm_switchover(x):
    """``|x| = 6`` is where the Maclaurin series hands over to the asymptotic
    expansion.  A jump there is the classic bug in this kind of code."""
    for f in (ak.fresnels, ak.fresnelc):
        for d in (1e-9, 1e-6, 1e-3):
            lo = enclose(f(x), x, 6.0 - d)
            hi = enclose(f(x), x, 6.0 + d)
            mid_lo = 0.5 * (lo[0] + lo[1])
            mid_hi = 0.5 * (hi[0] + hi[1])
            # |S′| ≤ 1 and |C′| ≤ 1, so the true values differ by ≤ 2d.
            assert abs(mid_hi - mid_lo) <= 2.0 * d + 1e-12


def test_fresnel_derivatives_round_trip(x):
    """The property the integrator's verification gate depends on:
    ``diff(fresnels(x))`` must *numerically* be ``sin(πx²/2)``."""
    ds = ak.diff(ak.fresnels(x), x).value
    dc = ak.diff(ak.fresnelc(x), x).value
    for at in (0.0, 0.3, 1.0, 2.7, 7.5, -1.4):
        t = math.pi * at * at / 2.0
        assert contains(enclose(ds, x, at), math.sin(t), slack=1e-12), f"S'({at})"
        assert contains(enclose(dc, x, at), math.cos(t), slack=1e-12), f"C'({at})"


def test_fresnel_is_entire_so_bounds_never_refuse(x):
    for f in (ak.fresnels, ak.fresnelc):
        for lo, hi in [(-40.0, -39.0), (-0.5, 0.5), (9.0, 9.5)]:
            r = ak.bound_on_box(f(x), [(x, lo, hi)])
            assert r.lower <= r.upper


@pytest.mark.parametrize("name", ["fresnels", "fresnelc"])
def test_capabilities_and_bounds_support(name, x):
    rows = {r["name"]: r for r in ak.capabilities()["primitives"]}
    for flag in ("numeric_f64", "numeric_ball", "taylor_model", "diff_forward", "diff_reverse"):
        assert rows[name][flag], f"{name}: {flag}"
    assert ak.bounds_supported(getattr(ak, name)(x)).supported
