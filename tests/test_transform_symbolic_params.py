"""Inverse Laplace / Z transforms whose coefficients are symbolic parameters.

Two things are checked for every case: the *value* of the answer against the
known closed form at concrete parameter points, and the *hypotheses* the answer
rests on.  The second is not decoration — the two-compartment answer is `0/0`
rather than the Bateman function at ``ka = ke``, and an answer whose validity
silently depends on that is the class of defect these transforms exist to avoid.
"""

from __future__ import annotations

import math

import alkahest as A
import pytest
from alkahest import experimental as ex


def _numeric(expr, env: dict) -> float:
    return float(A.eval_expr(expr, {sym: float(v) for sym, v in env.items()}))


# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------


def test_undamped_oscillator_symbolic_frequency():
    """L⁻¹{1/(s² + ω²)} = sin(ωt)/ω, valid for ω ≠ 0."""
    p = A.ExprPool()
    s, t, w = p.symbol("s"), p.symbol("t"), p.symbol("w")

    f = ex.inverse_laplace_transform(1 / (s**2 + w**2), s, t)
    assert ex.transform_side_conditions() == ["w ≠ 0"]

    for wv, tv in [(2.0, 0.7), (0.4, 3.1), (-1.5, 1.2)]:
        got = _numeric(f, {w: wv, t: tv})
        assert abs(got - math.sin(wv * tv) / wv) < 1e-9


def test_second_order_step_response():
    """The workhorse of classical control: K/(s² + 2ζωs + ω²)."""
    p = A.ExprPool()
    s, t = p.symbol("s"), p.symbol("t")
    k, w, zeta = p.symbol("K"), p.symbol("w"), p.symbol("zeta")

    f = ex.inverse_laplace_transform(k / (s**2 + 2 * zeta * w * s + w**2), s, t)
    conds = ex.transform_side_conditions()

    # ω ≠ 0, ζ ≠ 1, ζ ≠ −1, and the under-damped condition ω²(1−ζ²) > 0
    # (i.e. |ζ| < 1) without which the printed `sin` is a `sinh` in disguise.
    assert sum(c.endswith("≠ 0") for c in conds) == 3, conds
    assert sum(c.endswith("> 0") for c in conds) == 1, conds
    assert "w ≠ 0" in conds, conds

    for kv, wv, zv, tv in [(1.0, 2.0, 0.3, 0.8), (2.5, 5.0, 0.1, 1.7), (0.75, 1.25, 0.7, 3.0)]:
        wd = wv * math.sqrt(1.0 - zv * zv)
        want = kv * math.exp(-zv * wv * tv) * math.sin(wd * tv) / wd
        got = _numeric(f, {k: kv, w: wv, zeta: zv, t: tv})
        assert abs(got - want) < 1e-9 * (1.0 + abs(want)), (got, want)


def test_bateman_function():
    """The workhorse of pharmacokinetics, and the coincidence it rests on."""
    p = A.ExprPool()
    s, t = p.symbol("s"), p.symbol("t")
    d, ka, ke = p.symbol("D"), p.symbol("ka"), p.symbol("ke")

    f = ex.inverse_laplace_transform(d * ka / ((s + ka) * (s + ke)), s, t)
    conds = ex.transform_side_conditions()
    assert len(conds) == 1, conds
    assert "ka" in conds[0], conds[0]
    assert "ke" in conds[0], conds[0]
    assert conds[0].endswith("≠ 0"), conds[0]

    for dv, kav, kev, tv in [(100.0, 1.5, 0.25, 2.0), (50.0, 0.4, 2.2, 0.7), (1.0, 3.0, 0.1, 5.5)]:
        want = dv * kav * (math.exp(-kav * tv) - math.exp(-kev * tv)) / (kev - kav)
        got = _numeric(f, {d: dv, ka: kav, ke: kev, t: tv})
        assert abs(got - want) < 1e-9 * (1.0 + abs(want)), (got, want)


def test_repeated_symbolic_pole_is_the_other_branch():
    """At ka = ke the answer is t·e^{−ka·t}, and it needs no hypothesis."""
    p = A.ExprPool()
    s, t, ka = p.symbol("s"), p.symbol("t"), p.symbol("ka")

    f = ex.inverse_laplace_transform(1 / (s + ka) ** 2, s, t)
    assert ex.transform_side_conditions() == []
    for kav, tv in [(1.5, 2.0), (0.25, 4.0)]:
        assert abs(_numeric(f, {ka: kav, t: tv}) - tv * math.exp(-kav * tv)) < 1e-9


def test_rational_coefficients_assume_nothing():
    p = A.ExprPool()
    s, t = p.symbol("s"), p.symbol("t")
    ex.inverse_laplace_transform(1 / ((s + 1) * (s + 2)), s, t)
    assert ex.transform_side_conditions() == []


def test_side_conditions_are_consuming():
    """One call's hypotheses cannot be read as a later call's."""
    p = A.ExprPool()
    s, t, ka, ke = p.symbol("s"), p.symbol("t"), p.symbol("ka"), p.symbol("ke")
    ex.inverse_laplace_transform(1 / ((s + ka) * (s + ke)), s, t)
    first = ex.transform_side_conditions()
    assert len(first) == 1
    assert ex.transform_side_conditions() == first, "repeated reads of one call agree"

    ex.inverse_laplace_transform(1 / ((s + 1) * (s + 2)), s, t)
    assert ex.transform_side_conditions() == []


# ---------------------------------------------------------------------------
# Z
# ---------------------------------------------------------------------------


def test_geometric_round_trip_with_symbolic_ratio():
    """Z{aⁿ} → z/(z−a) → aⁿ. The round trip that used to break."""
    p = A.ExprPool()
    n, z, a = p.symbol("n"), p.symbol("z"), p.symbol("a")

    forward = ex.z_transform(a**n, n, z)
    back = ex.inverse_z_transform(forward, z, n)
    assert ex.transform_side_conditions() == []

    for av, nv in [(0.5, 3), (2.0, 4), (-1.5, 2)]:
        assert abs(_numeric(back, {a: av, n: nv}) - av**nv) < 1e-9


def test_repeated_symbolic_z_pole_reports_its_division():
    """Z⁻¹{z/(z−a)²} = n·a^{n−1} divides by a; at a = 0 the table declines."""
    p = A.ExprPool()
    n, z, a = p.symbol("n"), p.symbol("z"), p.symbol("a")

    seq = ex.inverse_z_transform(z / (z - a) ** 2, z, n)
    assert ex.transform_side_conditions() == ["a ≠ 0"]
    for av, nv in [(0.5, 3), (2.0, 5)]:
        assert abs(_numeric(seq, {a: av, n: nv}) - nv * av ** (nv - 1)) < 1e-9


# ---------------------------------------------------------------------------
# apart
# ---------------------------------------------------------------------------


def test_apart_over_parameter_field_reports_its_genericity():
    p = A.ExprPool()
    s, ka, ke = p.symbol("s"), p.symbol("ka"), p.symbol("ke")

    out = A.apart(1 / ((s + ka) * (s + ke)), s)
    conds = ex.apart_side_conditions()
    assert len(conds) == 1, conds
    assert conds[0].endswith("≠ 0")

    # It is an identity where the hypothesis holds.
    for sv, kav, kev in [(1.5, 0.5, 2.0), (-3.0, 1.0, -0.25)]:
        env = {s: sv, ka: kav, ke: kev}
        lhs = 1.0 / ((sv + kav) * (sv + kev))
        assert abs(_numeric(out, env) - lhs) < 1e-9 * (1.0 + abs(lhs))


def test_apart_over_q_is_unchanged_and_assumes_nothing():
    p = A.ExprPool()
    s = p.symbol("s")
    A.apart(1 / (s**2 - 1), s)
    assert ex.apart_side_conditions() == []


def test_apart_still_refuses_a_var_dependent_generator():
    """`exp(x)` is not a parameter; the historical refusal stands."""
    p = A.ExprPool()
    x = p.symbol("x")
    with pytest.raises(Exception):
        A.apart(A.exp(x) / (x**2 - 1), x)
