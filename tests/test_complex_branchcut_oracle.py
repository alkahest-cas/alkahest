"""mpmath oracle: complex / branch-cut numeric evaluation.

Fuzzes ``evaluate(..., mode="complex")`` for sqrt, log, Arg, powers, and the
imaginary unit against mpmath's principal branch.

Run:
    pytest tests/test_complex_branchcut_oracle.py -v

Requires:
    pip install mpmath
"""

from __future__ import annotations

import math
import random

import pytest

mpmath = pytest.importorskip("mpmath")

import alkahest as ak  # noqa: E402
from alkahest import (  # noqa: E402
    arg,
    conjugate,
    cos,
    evaluate,
    exp,
    im,
    log,
    re,
    sin,
    sqrt,
)

mp = mpmath
mp.mp.dps = 40


def _mpc(z: complex):
    return mp.mpc(z.real, z.imag)


def _close(got: complex, expect: complex, *, rel: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    err = abs(got - expect)
    scale = max(abs(got), abs(expect), 1.0)
    return err <= max(abs_tol, rel * scale)


def _points(seed: int = 0) -> list[complex]:
    rng = random.Random(seed)
    pts: list[complex] = []
    # On / near the negative-real branch cut
    for x in (-3.0, -2.0, -1.0, -0.5, -1e-8, -100.0):
        pts.extend(
            [
                complex(x, 0.0),
                complex(x, 1e-12),
                complex(x, -1e-12),
                complex(x, 1e-6),
                complex(x, -1e-6),
            ]
        )
    # Axes and quadrants
    pts.extend(
        [
            1 + 0j,
            0 + 1j,
            0 - 1j,
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
            0.5 + 0j,
            -0.5 + 0.1j,
        ]
    )
    for _ in range(80):
        pts.append(complex(rng.uniform(-5, 5), rng.uniform(-5, 5)))
    return pts


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def z(pool: ak.ExprPool):
    return pool.symbol("z", ak.Domain.Complex)


@pytest.fixture
def i(pool: ak.ExprPool):
    return pool.imaginary_unit()


# ---------------------------------------------------------------------------
# Fixed principal-branch cases
# ---------------------------------------------------------------------------


def test_sqrt_log_neg_one_principal(pool):
    got_sqrt = evaluate(sqrt(pool.integer(-1)), {}, mode="complex").value
    assert _close(got_sqrt, 1j)
    got = evaluate(log(pool.integer(-1)), {}, mode="complex").value
    assert _close(got, complex(0, math.pi))


def test_half_power_neg_one_is_principal_i(pool):
    expr = pool.integer(-1) ** pool.rational(1, 2)
    assert _close(evaluate(expr, {}, mode="complex").value, 1j)


def test_imaginary_unit_auto_binds(i):
    r = evaluate(i, {}, mode="complex")
    assert r.status == "ok"
    assert r.value == 1j
    assert evaluate(i * i, {}, mode="complex").value == -1 + 0j
    assert _close(evaluate(arg(i), {}, mode="complex").value, complex(math.pi / 2, 0))


def test_arg_declines_on_negative_real_cut(pool, z):
    """Exact negative reals are discontinuous for Arg — decline, don't guess."""
    r = evaluate(arg(pool.integer(-1)), {}, mode="complex")
    assert r.status == "unsupported"
    assert r.reason == "E-EVAL-011"
    r2 = evaluate(arg(z), {z: -2 + 0j}, mode="complex")
    assert r2.status == "unsupported"
    assert r2.reason == "E-EVAL-011"


def test_arg_matches_mpmath_off_cut(z):
    for w in (1 + 0j, 0 + 1j, 0 - 1j, -1 + 1e-9j, -1 - 1e-9j, 2 - 3j):
        r = evaluate(arg(z), {z: w}, mode="complex")
        assert r.status == "ok", (w, r.reason)
        expect = complex(float(mp.arg(_mpc(w))), 0.0)
        assert _close(r.value, expect), (w, r.value, expect)


def test_real_bindings_accepted_in_complex_mode(pool, z):
    r = evaluate(sqrt(z), {z: -4.0}, mode="complex")
    assert r.status == "ok"
    assert _close(r.value, 2j)


# ---------------------------------------------------------------------------
# Fuzz vs mpmath
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_unary_ops_match_mpmath(z, seed):
    ops = [
        ("sqrt", sqrt, lambda w: complex(mp.sqrt(_mpc(w)))),
        ("log", log, lambda w: complex(mp.log(_mpc(w)))),
        ("exp", exp, lambda w: complex(mp.exp(_mpc(w)))),
        ("sin", sin, lambda w: complex(mp.sin(_mpc(w)))),
        ("cos", cos, lambda w: complex(mp.cos(_mpc(w)))),
        ("conjugate", conjugate, lambda w: complex(w.real, -w.imag)),
        ("re", re, lambda w: complex(w.real, 0.0)),
        ("im", im, lambda w: complex(w.imag, 0.0)),
    ]
    for w in _points(seed):
        if abs(w) < 1e-18:
            continue
        for name, build, ref in ops:
            r = evaluate(build(z), {z: w}, mode="complex")
            assert r.status == "ok", f"{name}@{w}: {r.reason}"
            expect = ref(w)
            assert _close(r.value, expect), f"{name}@{w}: got {r.value}, want {expect}"


@pytest.mark.parametrize("seed", range(3))
def test_integer_powers_match_mpmath(z, seed):
    for w in _points(seed):
        if abs(w) < 1e-8:
            continue
        for n in range(-4, 5):
            r = evaluate(z**n, {z: w}, mode="complex")
            expect = complex(mp.power(_mpc(w), n))
            assert r.status == "ok", (w, n, r.reason)
            assert _close(r.value, expect, rel=1e-8, abs_tol=1e-8), (w, n, r.value, expect)


@pytest.mark.parametrize("seed", range(2))
def test_principal_noninteger_powers_match_mpmath(z, seed):
    exponents = [0.5, -0.5, 1.5, 2.5, 1 / 3, -1 / 3, math.pi]
    for w in _points(seed):
        if abs(w) < 1e-6:
            continue
        for e in exponents:
            r = evaluate(z**e, {z: w}, mode="complex")
            expect = complex(mp.power(_mpc(w), e))
            assert r.status == "ok", (w, e, r.reason)
            assert _close(r.value, expect, rel=1e-7, abs_tol=1e-7), (w, e, r.value, expect)


def test_rational_half_power_matches_sqrt_on_cut(pool, z):
    """(-x)^(1/2) must agree with principal sqrt for negative reals."""
    for x in (-0.25, -1.0, -4.0, -9.0):
        half = evaluate(z ** pool.rational(1, 2), {z: complex(x, 0)}, mode="complex")
        root = evaluate(sqrt(z), {z: complex(x, 0)}, mode="complex")
        assert half.status == root.status == "ok"
        assert _close(half.value, root.value)
        assert _close(half.value, complex(mp.sqrt(x)))


def test_symbolic_arg_folds_remain_branch_safe(pool, i):
    """Symbolic Arg only folds domain-safe cases; negatives stay symbolic."""
    assert str(ak.simplify(arg(pool.integer(3))).value) == "0"
    assert "pi" in str(ak.simplify(arg(i)).value)
    assert str(ak.simplify(arg(pool.integer(-1))).value) == "arg(-1)"
    assert str(ak.simplify(arg(pool.integer(0))).value) == "arg(0)"
