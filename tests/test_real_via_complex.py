"""N11 — real values behind complex intermediates.

A closed form can be real while some of its intermediates are not: the
oscillatory inverse Laplace ``cos(t·√ω²)`` at a point where ``ω² < 0`` (it is
``cosh(t·√−ω²)`` there), Cardano's formula for a cubic with three real roots,
a ZOH matrix built from those eigenvalues. Real ``f64`` arithmetic takes
``√(negative)`` to NaN, and ``compile_expr`` used to hand that NaN back as the
value, silently; ``eval_expr`` raised E-EVAL-009.

Now a point whose real evaluation is NaN is re-evaluated in complex arithmetic
with a running rounding-error bound, and the real part is returned when the
imaginary part is zero within that bound. NaN (or E-EVAL-009) stays where the
value is not a real number — ``log(−1)``, ``√−2``, ``asin 2``.

Oracles: ``scipy.linalg.expm`` (two-compartment model, ZOH), ``numpy``
eigenvalues (Cardano), and complex-mode evaluation of the expression itself
(random expressions).
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import shutil
import subprocess
import tempfile

import alkahest as ak
import pytest
from alkahest import experimental as ex

np = pytest.importorskip("numpy")
sl = pytest.importorskip("scipy.linalg")

RNG = random.Random(11)


def _find_cc():
    for candidate in ("/usr/bin/gcc", "/usr/bin/cc"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("cc") or shutil.which("gcc")


CC = _find_cc()
requires_cc = pytest.mark.skipif(CC is None, reason="no C compiler found")


def _c_function(code: str, name: str, n_args: int):
    """Compile emitted C into a shared object and return it via ctypes."""
    d = tempfile.mkdtemp()
    src, so = os.path.join(d, "f.c"), os.path.join(d, "f.so")
    with open(src, "w") as fh:
        fh.write("#include <math.h>\n" + code)
    r = subprocess.run(
        [CC, "-std=c99", "-O2", "-shared", "-fPIC", "-o", so, src, "-lm"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + code[-2000:]
    f = getattr(ctypes.CDLL(so), name)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double] * n_args
    return f


def _complex_value(e, env) -> complex:
    return complex(ak.evaluate(e, env, mode="complex").value)


# ---------------------------------------------------------------------------
# The report's repro: two-compartment inverse Laplace, symbolic rates
# ---------------------------------------------------------------------------


def _two_compartment(domain=None):
    pool = ak.ExprPool()
    sym = (lambda n: pool.symbol(n, domain)) if domain else pool.symbol
    s, t = pool.symbol("s"), pool.symbol("t")
    k12, k21, k10, d, v1 = (sym(n) for n in ("k12", "k21", "k10", "D", "V1"))
    big_f = (s + k21) * d / v1 / (s * s + (k10 + k12 + k21) * s + k10 * k21)
    r = ex.inverse_laplace_transform(big_f, s, t)
    return r, ex.transform_side_conditions(), (k10, k12, k21, t, d, v1)


def _expm_c1(k10, k12, k21, t, d, v1):
    a = np.array([[-(k10 + k12), k21], [k12, -k21]])
    return (sl.expm(a * t) @ np.array([d / v1, 0.0]))[0]


def test_report_repro_default_domain_evaluates_in_real_mode():
    r, _conds, syms = _two_compartment()
    point = (0.3, 0.5, 0.2, 1.7, 100.0, 10.0)
    ref = _expm_c1(*point)
    assert abs(ref - 3.117450651919716) < 1e-12
    env = dict(zip(syms, point))
    assert abs(ak.eval_expr(r, env) - ref) < 1e-12
    f = ak.compile_expr(r, list(syms))
    assert abs(f(list(point)) - ref) < 1e-12
    for _ in range(20):
        pt = [RNG.uniform(0.01, 3.0) for _ in range(3)] + [RNG.uniform(0, 10), 100.0, 7.0]
        ref = _expm_c1(*pt)
        got = f(pt)
        assert not math.isnan(got), pt
        assert abs(got - ref) < 1e-9 * (1 + abs(ref)), (pt, got, ref)
        assert abs(ak.eval_expr(r, dict(zip(syms, pt))) - ref) < 1e-9 * (1 + abs(ref))


def test_report_repro_batch_paths():
    r, _conds, syms = _two_compartment()
    f = ak.compile_expr(r, list(syms))
    n = 50
    cols = [
        np.array([RNG.uniform(0.01, 3.0) for _ in range(n)]),
        np.array([RNG.uniform(0.01, 3.0) for _ in range(n)]),
        np.array([RNG.uniform(0.01, 3.0) for _ in range(n)]),
        np.linspace(0.0, 12.0, n),
        np.full(n, 100.0),
        np.full(n, 7.0),
    ]
    ref = np.array([_expm_c1(*(c[j] for c in cols)) for j in range(n)])
    for out in (ak.numpy_eval(f, *cols), ak.numpy_eval_par(f, *cols)):
        out = np.asarray(out)
        assert not np.isnan(out).any()
        assert np.allclose(out, ref, rtol=1e-9, atol=1e-12)


def test_report_repro_positive_rates_is_real_form():
    r, conds, syms = _two_compartment(ak.Domain.Positive)
    text = str(r)
    assert "cos(" not in text, text
    assert "sin(" not in text, text
    assert not conds, conds
    point = (0.3, 0.5, 0.2, 1.7, 100.0, 10.0)
    assert abs(ak.eval_expr(r, dict(zip(syms, point))) - _expm_c1(*point)) < 1e-12
    f = ak.compile_expr(r, list(syms))
    assert abs(f(list(point)) - _expm_c1(*point)) < 1e-12


@requires_cc
def test_report_repro_emitted_c():
    r, _conds, syms = _two_compartment()
    names = ["k10", "k12", "k21", "t", "D", "V1"]
    code = ak.emit_c_expr(r, list(syms), names, "conc")
    f = _c_function(code, "conc", 6)
    for _ in range(10):
        pt = [RNG.uniform(0.01, 3.0) for _ in range(3)] + [RNG.uniform(0, 10), 100.0, 7.0]
        ref = _expm_c1(*pt)
        assert abs(f(*pt) - ref) < 1e-9 * (1 + abs(ref)), pt


# ---------------------------------------------------------------------------
# eng: symbolic Cardano eigenvalues, symbolic ZOH
# ---------------------------------------------------------------------------


def _chain_eigenvalues():
    pool = ak.ExprPool()
    i = pool.integer
    k, m = pool.symbol("k"), pool.symbol("m")
    stiff = ak.Matrix([[2 * k, -k, i(0)], [-k, 2 * k, -k], [i(0), -k, k]])
    mass = ak.Matrix([[m, i(0), i(0)], [i(0), m, i(0)], [i(0), i(0), m]])
    ev = list(ak.Matrix((mass.inverse() * stiff).to_list()).eigenvals())
    return ev, k, m


def test_cardano_eigenvalues_compile_to_real_values():
    ev, k, m = _chain_eigenvalues()
    assert len(ev) == 3
    fs = [ak.compile_expr(e, [k, m]) for e in ev]
    vec = ak.compile_expr(ev, [k, m])
    for _ in range(10):
        kv, mv = RNG.uniform(0.1, 5.0), RNG.uniform(0.1, 5.0)
        ref = np.linalg.eigvalsh(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 1.0]]) * kv / mv)
        got = sorted(f([kv, mv]) for f in fs)
        assert not any(math.isnan(g) for g in got), got
        assert np.allclose(got, ref, rtol=1e-9), (got, ref)
        assert np.allclose(sorted(vec([kv, mv])), ref, rtol=1e-9)
        evald = sorted(ak.eval_expr(e, {k: kv, m: mv}) for e in ev)
        assert np.allclose(evald, ref, rtol=1e-9)
        for e, g in zip(ev, (f([kv, mv]) for f in fs)):
            c = _complex_value(e, {k: kv, m: mv})
            assert abs(c.real - g) < 1e-9 * (1 + abs(g))


@requires_cc
def test_cardano_eigenvalues_emitted_c():
    ev, k, m = _chain_eigenvalues()
    code = ak.emit_c_vec(ev, [k, m], ["k", "m"], "eig")
    d = tempfile.mkdtemp()
    src, so = os.path.join(d, "f.c"), os.path.join(d, "f.so")
    with open(src, "w") as fh:
        fh.write("#include <math.h>\n" + code)
    r = subprocess.run(
        [CC, "-std=c99", "-O2", "-shared", "-fPIC", "-o", so, src, "-lm"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    eig = ctypes.CDLL(so).eig
    eig.restype = None
    out = (ctypes.c_double * 3)()
    eig.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    for _ in range(5):
        kv, mv = RNG.uniform(0.1, 5.0), RNG.uniform(0.1, 5.0)
        eig(kv, mv, out)
        ref = np.linalg.eigvalsh(np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 1.0]]) * kv / mv)
        assert np.allclose(sorted(out), ref, rtol=1e-9), (list(out), ref)


def test_zoh_entries_compile_to_expm():
    pool = ak.ExprPool()
    i = pool.integer
    a, b = pool.symbol("a"), pool.symbol("b")
    A = ak.Matrix([[i(0), i(1)], [-a, -b]])
    B = ak.Matrix([[i(0)], [i(1)]])
    C = ak.Matrix([[i(1), i(0)]])
    d = ak.StateSpace(A, B, C).discretize(pool.rational(1, 10), "zoh")
    entries = [x for row in d.A.to_list() for x in row]
    f = ak.compile_expr(entries, [a, b])
    # under-damped (b² < 4a) and over-damped (b² > 4a)
    for av, bv in [(2.0, 0.9), (2.0, 3.0), (5.0, 0.3), (0.5, 2.5)]:
        ref = sl.expm(0.1 * np.array([[0.0, 1.0], [-av, -bv]])).ravel()
        got = np.asarray(f([av, bv]))
        assert not np.isnan(got).any(), (av, bv, got)
        assert np.allclose(got, ref, rtol=1e-8, atol=1e-12), (av, bv, got, ref)
        evald = [ak.eval_expr(e, {a: av, b: bv}) for e in entries]
        assert np.allclose(evald, ref, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# Genuine domain errors stay NaN / E-EVAL-009
# ---------------------------------------------------------------------------


def test_genuinely_non_real_values_stay_nan():
    p = ak.ExprPool()
    x = p.symbol("x")
    cases = [
        (ak.log(x), -1.0),
        (ak.sqrt(x), -2.0),
        (ak.sqrt(x), -1e-300),
        (ak.asin(x), 2.0),
        (ak.acos(x), -1.5),
        (x ** p.rational(1, 3), -8.0),
        (ak.sqrt(x) + 1, -4.0),
        (ak.sqrt(x) * ak.sqrt(1 - x), -0.5),
        (ak.exp(ak.sqrt(x)), -1.0),
        (ak.cos(ak.sqrt(x)) + ak.sqrt(x), -1.0),
    ]
    for e, at in cases:
        assert math.isnan(ak.compile_expr(e, [x])([at])), (e, at)
        with pytest.raises(Exception, match=r"E-EVAL-009|undefined"):
            ak.eval_expr(e, {x: at})
        c = ak.evaluate(e, {x: at}, mode="complex").value
        assert c is None or abs(complex(c).imag) > 1e-20, (e, at, c)


def test_real_values_through_complex_intermediates():
    p = ak.ExprPool()
    x, t = p.symbol("x"), p.symbol("t")
    cases = [
        # cos(t√x) at x < 0 is cosh(t√−x)
        (ak.cos(t * ak.sqrt(x)), lambda xv, tv: math.cosh(tv * math.sqrt(-xv))),
        # sin(t√x)/√x at x < 0 is sinh(t√−x)/√−x
        (
            ak.sin(t * ak.sqrt(x)) / ak.sqrt(x),
            lambda xv, tv: math.sinh(tv * math.sqrt(-xv)) / math.sqrt(-xv),
        ),
        # √x·√x = x on the principal branch
        (ak.sqrt(x) * ak.sqrt(x) * t, lambda xv, tv: xv * tv),
        # |√x|² = −x for x < 0
        (ak.abs(ak.sqrt(x)) ** 2 + t, lambda xv, tv: -xv + tv),
    ]
    for e, ref in cases:
        f = ak.compile_expr(e, [x, t])
        for _ in range(10):
            xv, tv = -RNG.uniform(0.01, 4.0), RNG.uniform(0.0, 3.0)
            want = ref(xv, tv)
            assert abs(f([xv, tv]) - want) < 1e-10 * (1 + abs(want)), (e, xv, tv)
            assert abs(ak.eval_expr(e, {x: xv, t: tv}) - want) < 1e-10 * (1 + abs(want))


# ---------------------------------------------------------------------------
# Randomised oracle: compiled value vs the complex-mode real part
# ---------------------------------------------------------------------------


def _random_expr(rng, p, x, y):
    """A random expression that goes complex for some inputs and may or may
    not come back to the real axis."""

    def poly():
        return rng.choice([x, y, x - y, x * y - 1, x * x - 2 * y, 1 - x * x - y * y])

    r = ak.sqrt(poly())
    shapes = [
        lambda: ak.cos(r * rng.choice([x, y, p.integer(2)])),  # even in r → real
        lambda: r * r + poly(),  # real
        lambda: ak.sin(r) / r,  # even → real
        lambda: r + poly(),  # complex where poly < 0
        lambda: ak.exp(r) + ak.exp(-r),  # 2cosh(r) → real
        lambda: ak.exp(r) - ak.exp(-r),  # 2sinh(r) → imaginary where poly < 0
        lambda: ak.log(poly()) + 1,  # complex where poly < 0
        lambda: (poly() + 3) ** p.rational(1, 3) * x,
    ]
    return rng.choice(shapes)()


def test_random_expressions_never_nan_when_complex_value_is_real():
    rng = random.Random(5)
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    real_hits = nan_hits = 0
    for _ in range(60):
        e = _random_expr(rng, p, x, y)
        f = ak.compile_expr(e, [x, y])
        for _ in range(8):
            xv, yv = rng.uniform(-3, 3), rng.uniform(-3, 3)
            try:
                c = _complex_value(e, {x: xv, y: yv})
            except Exception:
                continue
            got = f([xv, yv])
            scale = 1.0 + abs(c)
            if abs(c.imag) <= 1e-13 * scale:
                assert not math.isnan(got), (e, xv, yv, c)
                assert abs(got - c.real) <= 1e-9 * scale, (e, xv, yv, got, c)
                real_hits += 1
            elif abs(c.imag) > 1e-6 * scale:
                assert math.isnan(got), (e, xv, yv, got, c)
                nan_hits += 1
    assert real_hits > 50, (real_hits, nan_hits)
    assert nan_hits > 20, (real_hits, nan_hits)
