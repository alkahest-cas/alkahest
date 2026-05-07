"""SymPy adapter for cross-CAS benchmarks (V1-13)."""

from __future__ import annotations

from typing import Any

from .base import CASAdapter


class SymPyAdapter(CASAdapter):
    """Adapter wrapping SymPy."""

    name = "SymPy"

    def is_available(self) -> bool:
        try:
            import sympy  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Task-named benchmark methods (match tasks.py task names) ─────────────

    def bench_poly_diff(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        expr = sum(x ** k for k in range(size + 1))
        return str(sp.diff(expr, x))

    def bench_trig_identity(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        expr = sum(sp.sin(x) ** 2 + sp.cos(x) ** 2 for _ in range(size))
        return str(sp.trigsimp(expr))

    def bench_jacobian_nxn(self, size: int) -> Any:
        import sympy as sp
        xs = [sp.Symbol(f"x{i}") for i in range(size)]
        fns = [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
        jac = sp.Matrix(fns).jacobian(xs)
        return [[str(jac[i, j]) for j in range(size)] for i in range(size)]

    def bench_ball_sin_cos(self, size: int) -> Any:
        import mpmath
        mpmath.mp.prec = 128
        iv = mpmath.iv
        radius = 1.0 / size
        x_iv = iv.mpf([1.0 - radius, 1.0 + radius])
        result = iv.sin(iv.cos(x_iv))
        return str(result)

    def bench_poly_jit_eval(self, size: int) -> Any:
        import sympy as sp
        import numpy as np
        x = sp.Symbol("x")
        poly = sum(x ** k for k in range(size + 1))
        fn = sp.lambdify(x, poly, "numpy")
        xs = np.linspace(0.0, 1.0, 1_000_000)
        return fn(xs)

    def bench_solve_circle_line(self, size: int) -> Any:
        import sympy as sp
        x, y = sp.symbols("x y")
        return [str(s) for s in sp.solve(
            [x ** 2 + y ** 2 - size ** 2, y - x], [x, y], dict=True
        )]

    # ── New comprehensive task methods ───────────────────────────────────────

    def bench_integrate_poly(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        poly = sum(x ** k for k in range(size + 1))
        return str(sp.integrate(poly, x))

    def bench_series_expansion(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return str(sp.series(sp.sin(x), x, 0, size))

    def bench_limit_computation(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return str(sp.limit((x ** size - 1) / (x - 1), x, 1))

    def bench_gradient_nvar(self, size: int) -> Any:
        import sympy as sp
        xs = [sp.Symbol(f"x{i}") for i in range(size)]
        f = sum(xi ** 2 for xi in xs)
        return [str(sp.diff(f, xi)) for xi in xs]

    def bench_matrix_det_nxn(self, size: int) -> Any:
        import sympy as sp
        rows = [[sp.Symbol(f"a{i}{j}") for j in range(size)] for i in range(size)]
        return str(sp.Matrix(rows).det())

    def bench_real_roots_poly(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return len(sp.Poly(x ** size - x - 1, x).real_roots())

    def bench_horner_form_poly(self, size: int) -> Any:
        import sympy as sp
        from sympy.polys.polyfuncs import horner
        x = sp.Symbol("x")
        poly = sum(x ** k for k in range(size + 1))
        return str(horner(poly))

    def bench_log_exp_simplify(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x", real=True, positive=True)
        expr = x
        for _ in range(size):
            expr = sp.log(sp.exp(expr))
        return str(sp.simplify(expr))

    def bench_resultant_poly(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return str(sp.resultant(x ** size + x + 1, x ** size - x - 1, x))

    def bench_recurrence_solve(self, size: int) -> Any:
        import sympy as sp
        n_sym = sp.Symbol("n", integer=True)
        a = sp.Function("a")
        if size == 1:
            eq = a(n_sym + 1) - 2 * a(n_sym)
            return str(sp.rsolve(eq, a(n_sym), {a(0): 1}))
        else:
            eq = a(n_sym + 2) - a(n_sym + 1) - a(n_sym)
            return str(sp.rsolve(eq, a(n_sym), {a(0): 1, a(1): 1}))

    # ── Legacy names ─────────────────────────────────────────────────────────

    def bench_integrate(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return str(sp.integrate(x ** size, x))

    def bench_simplify(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        expr = (sp.sin(x) ** 2 + sp.cos(x) ** 2) ** size
        return str(sp.simplify(expr))

    def bench_diff(self, size: int) -> Any:
        return self.bench_poly_diff(size)

    def bench_groebner(self, size: int) -> Any:
        import sympy as sp
        x, y = sp.symbols("x y")
        polys = [x ** 2 + y ** 2 - 1, x - y]
        return [str(p) for p in sp.groebner(polys, [x, y], order="lex")]

    def bench_poly_gcd(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return str(sp.gcd(x ** size - 1, x ** (size // 2) - 1))

    def bench_jacobian(self, size: int) -> Any:
        import sympy as sp
        x, y = sp.symbols("x y")
        exprs = [sp.sin(x * y), sp.cos(x + y), sp.exp(x - y)]
        jac = sp.Matrix(exprs).jacobian([x, y])
        return [[str(jac[i, j]) for j in range(2)] for i in range(3)]

    def bench_polynomial_solve(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return [str(s) for s in sp.solve(x ** 2 - size, x)]
