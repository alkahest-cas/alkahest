"""CodSpeed benchmarks for batch numeric evaluation (report 10-9, G7).

The workloads the report measured against NumPy / ``lambdify``: a degree-10
polynomial, a Lennard-Jones force (and its full gradient through the
multi-output ``compile_expr([...])``), the one-compartment Bateman function,
and a Black-Scholes price (``erf``, which used to push the whole expression
off the JIT). Each benchmark times ``numpy_eval`` alone — compilation happens
once, outside the timed call — at a size small enough for instruction-count
mode; ``benchmarks/python_bench.py``-style wall-clock comparisons at 10^6
points are in the G7 pull request.

The CodSpeed job builds the extension without a JIT feature, so there every
benchmark here measures the interpreter tier — the block-vectorised program
(F14) — which is also what ``Piecewise`` runs on in every build (the JITs
decline it): ``test_numpy_eval_piecewise`` and ``test_numpy_eval_smoothstep``.

Run locally (wall-clock):
    pytest benchmarks/bench_codspeed_numeric.py -v

Run with CodSpeed instrumentation (CI):
    pytest benchmarks/bench_codspeed_numeric.py --codspeed
"""

from __future__ import annotations

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")

N = 4_096


def _rng():
    return np.random.default_rng(0)


def _poly10():
    p = ak.ExprPool()
    x = p.symbol("x")
    poly = sum((p.integer(i + 1) * x**i for i in range(11)), p.integer(0))
    return ak.compile_expr(poly, [x], expected_evals=N), [np.linspace(0.0, 1.0, N)]


def _lj_grad():
    p = ak.ExprPool()
    r = [p.symbol(f"r{i}") for i in range(3)]
    r2 = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    grad = ak.symbolic_grad(4 * (r2**-6 - r2**-3), r)
    rng = _rng()
    return grad, r, [rng.uniform(0.9, 2.0, N) for _ in range(3)]


def _bateman():
    p = ak.ExprPool()
    ka, ke, v, d, t = (p.symbol(n) for n in ("ka", "ke", "V", "D", "t"))
    expr = d * ka / (v * (ka - ke)) * (ak.exp(-ke * t) - ak.exp(-ka * t))
    rng = _rng()
    cols = [
        rng.uniform(0.5, 2, N),
        rng.uniform(0.05, 0.3, N),
        rng.uniform(10, 50, N),
        np.full(N, 100.0),
        rng.uniform(0, 24, N),
    ]
    return ak.compile_expr(expr, [ka, ke, v, d, t], expected_evals=N), cols


def _black_scholes():
    p = ak.ExprPool()
    s, k, t, r, sg = (p.symbol(n) for n in ("S", "K", "T", "r", "sigma"))
    d1 = (ak.log(s / k) + (r + sg**2 / 2) * t) / (sg * ak.sqrt(t))
    d2 = d1 - sg * ak.sqrt(t)

    def phi(z):
        return (1 + ak.erf(z / ak.sqrt(p.integer(2)))) / 2

    price = s * phi(d1) - k * ak.exp(-r * t) * phi(d2)
    rng = _rng()
    cols = [
        rng.uniform(50, 150, N),
        rng.uniform(50, 150, N),
        rng.uniform(0.1, 2, N),
        rng.uniform(0, 0.1, N),
        rng.uniform(0.1, 0.5, N),
    ]
    return ak.compile_expr(price, [s, k, t, r, sg], expected_evals=N), cols


def test_numpy_eval_poly10(benchmark):
    f, cols = _poly10()
    benchmark(ak.numpy_eval, f, *cols)


def test_numpy_eval_lennard_jones_force(benchmark):
    grad, r, cols = _lj_grad()
    f = ak.compile_expr(grad[0], r, expected_evals=N)
    benchmark(ak.numpy_eval, f, *cols)


def test_numpy_eval_lennard_jones_gradient_multi_output(benchmark):
    grad, r, cols = _lj_grad()
    f = ak.compile_expr(grad, r)
    benchmark(ak.numpy_eval, f, *cols)


def test_numpy_eval_bateman(benchmark):
    f, cols = _bateman()
    benchmark(ak.numpy_eval, f, *cols)


def test_numpy_eval_black_scholes(benchmark):
    f, cols = _black_scholes()
    benchmark(ak.numpy_eval, f, *cols)


def _piecewise():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    pw = ak.piecewise(
        [
            (p.lt(x, p.integer(0)), x**2 * y),
            (p.lt(x, p.integer(1)), ak.sin(x) + y**3),
        ],
        ak.exp(-x) * y,
    )
    rng = _rng()
    cols = [rng.uniform(-2, 3, N), rng.uniform(-1, 1, N)]
    return ak.compile_expr(pw, [x, y], expected_evals=N), cols


def _smoothstep():
    p = ak.ExprPool()
    x = p.symbol("x")
    step = ak.piecewise(
        [(p.lt(x, p.integer(0)), p.integer(0)), (p.gt(x, p.integer(1)), p.integer(1))],
        x * x * (3 - 2 * x),
    )
    return ak.compile_expr(step, [x], expected_evals=N), [_rng().uniform(-1, 2, N)]


def test_numpy_eval_piecewise(benchmark):
    f, cols = _piecewise()
    benchmark(ak.numpy_eval, f, *cols)


def test_numpy_eval_smoothstep(benchmark):
    f, cols = _smoothstep()
    benchmark(ak.numpy_eval, f, *cols)
