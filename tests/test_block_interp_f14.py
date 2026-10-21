"""Block-vectorised interpreter tier (F14, a G7 follow-up).

The interpreter tier — every build without a JIT, and every expression a JIT
declines, ``Piecewise`` above all — evaluated a batch one point at a time, and
``Piecewise`` through a tree walk that built a hash map per point (25x slower
than the strict program at 10^6 points). It now runs a flat program over
blocks of points, ``Piecewise`` as a masked select with per-point failure
flags. These tests pin the values: batch == single-point call bit for bit
(including NaN / inf / signed zero inputs), a branch a point does not select
cannot fail it, ``And`` / ``Or`` short-circuit per point, the parallel batch
equals the sequential one, and the compile-time refusals are unchanged.
"""

from __future__ import annotations

import math
import random

import alkahest as ak
import pytest
from alkahest import ExprPool, compile_expr, numpy_eval

np = pytest.importorskip("numpy")

SPECIALS = [math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0, -1.0, 5e-324, 1e300]


def _same(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a), np.signbit(b))


def _points(n, lo, hi, seed):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(lo, hi, n)
    xs[::13] = np.resize(SPECIALS, xs[::13].shape)
    return xs


def _per_point(f, *cols):
    return np.array([f(list(pt)) for pt in zip(*cols)])


def _par(f, *cols):
    if hasattr(ak, "numpy_eval_par"):
        return ak.numpy_eval_par(f, *cols)
    return numpy_eval(f, *cols)


def test_piecewise_batch_equals_point_calls_and_numpy():
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    zero, one = p.integer(0), p.integer(1)
    pw = ak.piecewise(
        [(p.lt(x, zero), x**2 * y), (p.lt(x, one), ak.sin(x) + y**3)],
        ak.exp(-x) * y,
    )
    f = compile_expr(pw, [x, y])
    n = 40_000  # several Rayon chunks; many blocks with a partial last one
    xs, ys = _points(n, -2.0, 3.0, 1), _points(n, -1.0, 1.0, 2)
    out = numpy_eval(f, xs, ys)
    assert _same(out[:3000], _per_point(f, xs[:3000], ys[:3000]))
    assert _same(_par(f, xs, ys), out)
    # Against NumPy where the inputs are ordinary numbers.
    ok = np.isfinite(xs) & np.isfinite(ys) & (np.abs(xs) < 1e3)
    with np.errstate(all="ignore"):
        ref = np.where(xs < 0, xs**2 * ys, np.where(xs < 1, np.sin(xs) + ys**3, np.exp(-xs) * ys))
    np.testing.assert_allclose(out[ok], ref[ok], rtol=1e-14, atol=1e-300)


def test_a_branch_no_point_selects_cannot_fail_it():
    p = ExprPool()
    x = p.symbol("x")
    zero, one = p.integer(0), p.integer(1)
    w = ak.lambert_w(x)  # no value below -1/e
    xs = np.array([-5.0, -1.0, -0.2, 0.0, 0.5, 3.0])
    wv = [compile_expr(w, [x])([v]) for v in (0.5, 3.0)]
    cases = {
        "guarded": (ak.piecewise([(p.gt(x, zero), w)], one), [1, 1, 1, 1, *wv]),
        "in_condition": (
            ak.piecewise([(p.gt(w, zero), p.integer(2))], one),
            [math.nan, math.nan, 1, 1, 2, 2],
        ),
        "and_short_circuits": (
            ak.piecewise([(p.pred_and([p.gt(x, zero), p.gt(w, zero)]), p.integer(3))], one),
            [1, 1, 1, 1, 3, 3],
        ),
        "or_short_circuits": (
            ak.piecewise([(p.pred_or([p.lt(x, zero), p.gt(w, zero)]), p.integer(4))], one),
            [4, 4, 4, 1, 4, 4],
        ),
        "or_consults_w": (
            ak.piecewise([(p.pred_or([p.gt(x, zero), p.gt(w, zero)]), p.integer(4))], one),
            [math.nan, math.nan, 1, 1, 4, 4],
        ),
    }
    for name, (expr, want) in cases.items():
        f = compile_expr(expr, [x])
        out = numpy_eval(f, xs)
        assert _same(out, np.array(want, dtype=float)), (name, out)
        assert _same(_per_point(f, xs), out), name


def test_special_values_flow_through_conditions_as_at_a_single_point():
    p = ExprPool()
    x = p.symbol("x")
    zero = p.integer(0)
    pw = ak.piecewise([(p.ge(x, zero), ak.sqrt(x)), (p.lt(x, zero), -x)], p.integer(7))
    f = compile_expr(pw, [x])
    xs = np.array(SPECIALS * 40)
    out = numpy_eval(f, xs)
    assert _same(out, _per_point(f, xs))
    # NaN: neither condition holds, so the default; -0 >= 0 holds: sqrt(-0) = -0.
    assert out[0] == 7.0
    assert out[4] == 0.0
    assert math.copysign(1.0, out[4]) == -1.0
    assert out[2] == math.inf  # -(-inf)


def test_piecewise_compared_inside_a_condition_has_a_value():
    # The tree walk the tier used before had no value for a Piecewise inside a
    # comparison (it returned NaN); eval_expr does, and so does the batch now.
    p = ExprPool()
    x = p.symbol("x")
    zero = p.integer(0)
    inner = ak.piecewise([(p.gt(x, zero), x)], zero)
    outer = ak.piecewise([(p.lt(inner, p.integer(1)), x + 10)], -x)
    f = compile_expr(outer, [x])
    xs = np.array([-3.0, 0.25, 0.75, 2.0, 5.0])
    out = numpy_eval(f, xs)
    want = [ak.eval_expr(outer, {x: float(v)}) for v in xs]
    assert _same(out, np.array(want))
    assert _same(_per_point(f, xs), out)


def test_unbound_symbol_in_a_branch_is_still_refused():
    p = ExprPool()
    x, z = p.symbol("x"), p.symbol("z")
    pw = ak.piecewise([(p.gt(x, p.integer(0)), z)], x)
    with pytest.raises(ak.JitError) as info:
        compile_expr(pw, [x])
    assert info.value.code == "E-JIT-005"
    fx = p.func("f", [x])
    with pytest.raises(ak.JitError) as info:
        compile_expr(ak.piecewise([(p.gt(x, p.integer(0)), fx)], x), [x])
    assert info.value.code == "E-JIT-006"


def _random_expr(p, x, y, rnd, depth):
    zero = p.integer(0)
    if depth == 0 or rnd.random() < 0.2:
        return rnd.choice([x, y, p.integer(rnd.randint(-3, 3)), p.rational(rnd.randint(-5, 5), 3)])
    kind = rnd.randrange(7)
    a = _random_expr(p, x, y, rnd, depth - 1)
    if kind == 0:
        return a + _random_expr(p, x, y, rnd, depth - 1)
    if kind == 1:
        return a * _random_expr(p, x, y, rnd, depth - 1)
    if kind == 2:
        return a ** p.integer(rnd.randint(-3, 5))
    if kind == 3:
        return rnd.choice([ak.sin, ak.exp, ak.log, ak.sqrt, ak.erf, ak.lambert_w])(a)
    if kind == 4:
        return ak.atan2(a, _random_expr(p, x, y, rnd, depth - 1))
    cmp = rnd.choice([p.lt, p.le, p.gt, p.ge])
    cond = cmp(a, _random_expr(p, x, y, rnd, 1))
    if kind == 6:
        cond = p.pred_or([cond, p.gt(y, zero)])
    return ak.piecewise(
        [(cond, _random_expr(p, x, y, rnd, depth - 1))],
        _random_expr(p, x, y, rnd, depth - 1),
    )


def test_random_expressions_batch_equals_point_calls():
    rnd = random.Random(14)
    p = ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    xs, ys = _points(700, -4.0, 4.0, 3), _points(700, -3.0, 3.0, 4)
    for _ in range(60):
        e = _random_expr(p, x, y, rnd, 4)
        f = compile_expr(e, [x, y])
        out = numpy_eval(f, xs, ys)  # a JIT where the build has one and it applies
        assert _same(out, _per_point(f, xs, ys)), str(e)
