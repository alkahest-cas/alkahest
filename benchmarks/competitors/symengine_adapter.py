"""SymEngine adapter for cross-CAS benchmarks (V1-13).

SymEngine is a fast C++ symbolic math library with Python bindings.
Install: pip install symengine

SymEngine lacks trigonometric simplification and a general solver, so
``bench_trig_identity`` and ``bench_solve_circle_line`` raise
``NotImplementedError`` and are omitted from the report for this system.
``bench_ball_sin_cos`` delegates to mpmath interval arithmetic.
``bench_poly_jit_eval`` uses ``symengine.lambdify`` + NumPy (1M points).
"""

from __future__ import annotations

from typing import Any

from .base import CASAdapter


class SymEngineAdapter(CASAdapter):
    """Adapter wrapping SymEngine (symengine Python bindings)."""

    name = "SymEngine"

    def is_available(self) -> bool:
        try:
            import symengine  # noqa: F401
            return True
        except ImportError:
            return False

    def bench_poly_diff(self, size: int) -> Any:
        import symengine as se
        x = se.Symbol("x")
        poly = sum(x ** k for k in range(size + 1))
        return se.diff(poly, x)

    def bench_trig_identity(self, size: int) -> Any:
        # SymEngine has no trigsimp; not_implemented so the run is skipped.
        raise NotImplementedError("symengine has no trigsimp")

    def bench_jacobian_nxn(self, size: int) -> Any:
        import symengine as se
        xs = [se.Symbol(f"x{i}") for i in range(size)]
        fns = se.DenseMatrix(
            size, 1, [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
        )
        xmat = se.DenseMatrix(size, 1, xs)
        return fns.jacobian(xmat)

    def bench_ball_sin_cos(self, size: int) -> Any:
        import mpmath
        mpmath.mp.prec = 128
        iv = mpmath.iv
        radius = 1.0 / size
        x_iv = iv.mpf([1.0 - radius, 1.0 + radius])
        return str(iv.sin(iv.cos(x_iv)))

    def bench_poly_jit_eval(self, size: int) -> Any:
        import symengine as se
        import numpy as np
        x = se.Symbol("x")
        poly = sum(x ** k for k in range(size + 1))
        fn = se.lambdify([x], [poly])
        xs = np.linspace(0.0, 1.0, 1_000_000)
        return fn(xs)

    def bench_solve_circle_line(self, size: int) -> Any:
        raise NotImplementedError("symengine has no general solve")

    # ── New comprehensive task methods ────────────────────────────────────────

    def bench_integrate_poly(self, size: int) -> Any:
        import symengine as se
        x = se.Symbol("x")
        poly = sum(x ** k for k in range(size + 1))
        return se.integrate(poly, x)

    def bench_gradient_nvar(self, size: int) -> Any:
        import symengine as se
        xs = [se.Symbol(f"x{i}") for i in range(size)]
        f = sum(xi ** 2 for xi in xs)
        return [se.diff(f, xi) for xi in xs]

    def bench_series_expansion(self, size: int) -> Any:
        import symengine as se
        x = se.Symbol("x")
        try:
            return se.series(se.sin(x), x, 0, size)
        except AttributeError:
            raise NotImplementedError("symengine.series not available in this version")
