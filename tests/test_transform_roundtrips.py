"""Forward∘inverse identity checks for Laplace and Z transforms.

These are cheap correctness oracles: on the documented table,
``L⁻¹{L{f}} ≈ f`` and ``Z⁻¹{Z{a}} ≈ a`` (numeric sampling).  Failures here
are high-embarrassment incorrectness.
"""

from __future__ import annotations

import math
import random

import alkahest as A
import pytest
from alkahest import experimental as ex


def _numeric(expr, env: dict) -> float:
    return float(A.eval_expr(expr, {sym: float(v) for sym, v in env.items()}))


def _agree(a, b, var, samples, tol=1e-5) -> None:
    for x in samples:
        va = _numeric(a, {var: x})
        vb = _numeric(b, {var: x})
        assert abs(va - vb) <= tol * (1.0 + abs(va) + abs(vb)), (
            f"mismatch at {var}={x}: {a} → {va} vs {b} → {vb}"
        )


# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,builder",
    [
        ("1", lambda p, t: p.integer(1)),
        ("t", lambda p, t: t),
        ("t^2", lambda p, t: t**2),
        ("exp(2t)", lambda p, t: A.exp(2 * t)),
        ("sin(3t)", lambda p, t: A.sin(3 * t)),
        ("cos(2t)", lambda p, t: A.cos(2 * t)),
        ("t*exp(t)", lambda p, t: p.mul([t, A.exp(t)])),
        ("t*sin(2t)", lambda p, t: p.mul([t, A.sin(2 * t)])),
        ("t*cos(t)", lambda p, t: p.mul([t, A.cos(t)])),
        (
            "exp(2t)*sin(3t)",
            lambda p, t: p.mul([A.exp(2 * t), A.sin(3 * t)]),
        ),
        (
            "t*exp(t)*sin(2t)",
            lambda p, t: p.mul([t, A.exp(t), A.sin(2 * t)]),
        ),
    ],
)
def test_laplace_forward_inverse_roundtrip(name, builder):
    p = A.ExprPool()
    t = p.symbol("t")
    s = p.symbol("s")
    f = builder(p, t)
    big_f = ex.laplace_transform(f, t, s)
    back = ex.inverse_laplace_transform(big_f, s, t)
    _agree(back, f, t, [0.25, 0.5, 1.0, 1.5, 2.0])


def test_laplace_inverse_forward_rational():
    p = A.ExprPool()
    t = p.symbol("t")
    s = p.symbol("s")
    # L⁻¹{1/(s-2)} = e^{2t}; L of that recovers 1/(s-2).
    f = ex.inverse_laplace_transform(1 / (s - 2), s, t)
    back = ex.laplace_transform(f, t, s)
    _agree(back, 1 / (s - 2), s, [3.0, 4.0, 5.0, 7.0])


def test_laplace_roundtrip_fuzz_trig_exp():
    """Light fuzz: random ω, a over a small integer grid."""
    rng = random.Random(0x1A0ACE)
    p = A.ExprPool()
    t = p.symbol("t")
    s = p.symbol("s")
    for _ in range(12):
        omega = rng.choice([1, 2, 3, 4, 5])
        a = rng.choice([-3, -2, -1, 0, 1, 2])
        kind = rng.choice(["sin", "cos", "t_sin", "exp_sin"])
        if kind == "sin":
            f = A.sin(omega * t)
        elif kind == "cos":
            f = A.cos(omega * t)
        elif kind == "t_sin":
            f = p.mul([t, A.sin(omega * t)])
        else:
            if a == 0:
                f = A.sin(omega * t)
            else:
                f = p.mul([A.exp(a * t), A.sin(omega * t)])
        big_f = ex.laplace_transform(f, t, s)
        back = ex.inverse_laplace_transform(big_f, s, t)
        _agree(back, f, t, [0.4, 0.9, 1.3, 2.1])


# ---------------------------------------------------------------------------
# Z-transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,builder",
    [
        ("1", lambda p, n: p.integer(1)),
        ("2^n", lambda p, n: p.integer(2) ** n),
        ("(-1)^n", lambda p, n: p.integer(-1) ** n),
        ("n", lambda p, n: n),
        ("n*2^n", lambda p, n: p.mul([n, p.integer(2) ** n])),
        ("sin(n)", lambda p, n: A.sin(n)),
        ("cos(n)", lambda p, n: A.cos(n)),
        ("3*sin(2n)", lambda p, n: 3 * A.sin(2 * n)),
        ("cos(3n)", lambda p, n: A.cos(3 * n)),
    ],
)
def test_z_forward_inverse_roundtrip(name, builder):
    p = A.ExprPool()
    n = p.symbol("n")
    z = p.symbol("z")
    xn = builder(p, n)
    big_x = ex.z_transform(xn, n, z)
    back = ex.inverse_z_transform(big_x, z, n)
    _agree(back, xn, n, list(range(0, 8)))


def test_z_roundtrip_fuzz_sin_cos():
    rng = random.Random(0x27A11)
    p = A.ExprPool()
    n = p.symbol("n")
    z = p.symbol("z")
    for _ in range(10):
        omega = rng.choice([1, 2, 3, 4])
        amp = rng.choice([1, 2, 3, -1])
        if rng.random() < 0.5:
            xn = amp * A.sin(omega * n)
        else:
            xn = amp * A.cos(omega * n)
        big_x = ex.z_transform(xn, n, z)
        back = ex.inverse_z_transform(big_x, z, n)
        _agree(back, xn, n, list(range(0, 7)))


def test_z_sin_matches_unit_circle_samples():
    """Sanity: Z⁻¹{z/(z²+1)} samples equal sin(π n / 2)."""
    p = A.ExprPool()
    n = p.symbol("n")
    z = p.symbol("z")
    big_x = z / (z**2 + 1)
    xn = ex.inverse_z_transform(big_x, z, n)
    for k in range(0, 8):
        got = _numeric(xn, {n: k})
        want = math.sin(math.pi * k / 2.0)
        assert abs(got - want) < 1e-9
