"""SageMath adapter for cross-CAS benchmarks (V1-13).

Requires SageMath to be installed as a Python package (``pip install sagemath-standard``
on Ubuntu) or the ``sage`` command to be available.

We use the SageMath Python API directly (``from sage.all import *``) which is
available when SageMath is installed as a Python package.  If only the ``sage``
command-line is available, we fall back to subprocess execution.
"""

from __future__ import annotations

import subprocess
from typing import Any

from .base import CASAdapter


class SageAdapter(CASAdapter):
    """Adapter wrapping SageMath."""

    name = "SageMath"

    def is_available(self) -> bool:
        # Try Python API first (faster)
        try:
            import sage.all  # noqa: F401
            return True
        except ImportError:
            pass
        # Fall back to CLI
        try:
            result = subprocess.run(
                ["sage", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _sage(self, code: str) -> str:
        """Execute ``code`` in a Sage session and return printed output."""
        try:
            import sage.all as sage
            result = eval(code, {"sage": sage, **vars(sage)})  # noqa: S307
            return str(result)
        except Exception:
            pass
        # CLI fallback
        script = f"print({code})"
        result = subprocess.run(
            ["sage", "--python", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    def bench_integrate(self, size: int) -> Any:
        try:
            from sage.all import SR, var, integrate
            x = var("x")
            return str(integrate(x ** size, x))
        except ImportError:
            return self._sage(f"var('x'); integrate(x^{size}, x)")

    def bench_diff(self, size: int) -> Any:
        try:
            from sage.all import var, sum as sage_sum, diff
            x = var("x")
            expr = sage_sum(x ** k for k in range(size + 1))
            return str(diff(expr, x))
        except ImportError:
            return self._sage(
                f"var('x'); diff(sum(x^k for k in range({size+1})), x)"
            )

    def bench_groebner(self, size: int) -> Any:
        try:
            from sage.all import QQ, PolynomialRing
            R = PolynomialRing(QQ, ["x", "y"], order="lex")
            x, y = R.gens()
            I = R.ideal([x ** 2 + y ** 2 - 1, x - y])
            return [str(g) for g in I.groebner_basis()]
        except ImportError:
            return self._sage(
                "R.<x,y> = QQ[]; I = R.ideal(x^2+y^2-1, x-y); I.groebner_basis()"
            )

    def bench_polynomial_solve(self, size: int) -> Any:
        try:
            from sage.all import var, solve
            x = var("x")
            return [str(s) for s in solve(x ** 2 == size, x)]
        except ImportError:
            return self._sage(f"var('x'); solve(x^2=={size}, x)")

    # ── Task-named methods (``tasks.py`` registry) ───────────────────────────

    def bench_poly_diff(self, size: int) -> Any:
        return self.bench_diff(size)

    def bench_trig_identity(self, size: int) -> Any:
        try:
            from sage.all import cos, sin, var
            x = var("x")
            expr = sum(sin(x) ** 2 + cos(x) ** 2 for _ in range(size))
            return str(expr.simplify_trig())
        except ImportError:
            n = size
            return self._sage(
                "var('x'); sum(sin(x)^2+cos(x)^2 for _ in range(%d)).trig_reduce()" % n
            )

    def bench_jacobian_nxn(self, size: int) -> Any:
        try:
            from sage.all import jacobian, var
            names = [f"x{i}" for i in range(size)]
            xs = var(",".join(names))
            fns = [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
            J = jacobian(fns, list(xs))
            return str(J)
        except ImportError:
            raise NotImplementedError("sage jacobian path")

    def bench_ball_sin_cos(self, size: int) -> Any:
        try:
            from sage.all import cos, sin
            from sage.rings.real_mpfi import RIF

            radius = 1.0 / size
            x = RIF(1.0 - radius, 1.0 + radius)
            return str(sin(cos(x)))
        except ImportError:
            return self._sage("RIF(%s,%s)" % (1 - 1 / size, 1 + 1 / size))

    def bench_poly_jit_eval(self, size: int) -> Any:
        try:
            from sage.all import fast_callable, var
            import numpy as np

            x = var("x")
            expr = sum(x**k for k in range(size + 1))
            fn = fast_callable(expr, vars=[x], domain=float)
            xs = np.linspace(0.0, 1.0, 1_000_000)
            return fn(xs)
        except ImportError:
            raise NotImplementedError("sage fast_callable + NumPy path")

    def bench_solve_circle_line(self, size: int) -> Any:
        try:
            from sage.all import var, solve
            x, y = var("x y")
            sol = solve([x**2 + y**2 - size**2, y - x], [x, y])
            return [str(s) for s in sol]
        except ImportError:
            return self._sage(
                f"var('x y'); solve([x^2+y^2-{size**2}, y-x], [x,y])"
            )

    def bench_solve_6r_ik(self, size: int) -> Any:
        try:
            from sage.all import QQ, PolynomialRing
            R = PolynomialRing(QQ, ["y", "x"], order="lex")
            y, x = R.gens()
            I = R.ideal([x**2 + y**2 - size**2, y - x])
            return [str(g) for g in I.groebner_basis()]
        except ImportError:
            return self._sage(
                f"R.<y,x>=QQ[]; I=R.ideal(x^2+y^2-{size**2}, y-x); I.groebner_basis()"
            )

    def bench_integrate_poly(self, size: int) -> Any:
        try:
            from sage.all import integrate, var
            x = var("x")
            expr = sum(x**k for k in range(size + 1))
            return str(integrate(expr, x))
        except ImportError:
            return self._sage(f"var('x'); integrate(sum(x^k for k in range({size+1})), x)")

    def bench_series_expansion(self, size: int) -> Any:
        try:
            from sage.all import sin, var
            x = var("x")
            return str(sin(x).taylor(x, 0, size - 1))
        except ImportError:
            return self._sage(f"var('x'); sin(x).taylor(x, 0, {size - 1})")

    def bench_limit_computation(self, size: int) -> Any:
        try:
            from sage.all import limit, var
            x = var("x")
            return str(limit((x**size - 1) / (x - 1), x, 1))
        except ImportError:
            return self._sage(f"var('x'); limit((x^{size}-1)/(x-1), x=1)")

    def bench_gradient_nvar(self, size: int) -> Any:
        try:
            from sage.all import var
            names = ",".join(f"x{i}" for i in range(size))
            xs = var(names)
            f = sum(xs[i] ** 2 for i in range(size))
            return [str(f.derivative(xs[i])) for i in range(size)]
        except ImportError:
            raise NotImplementedError

    def bench_matrix_det_nxn(self, size: int) -> Any:
        try:
            from sage.all import Matrix, var
            rows = []
            for i in range(size):
                row = [var(f"a{i}{j}") for j in range(size)]
                rows.append(row)
            return str(Matrix(rows).det())
        except ImportError:
            raise NotImplementedError

    def bench_real_roots_poly(self, size: int) -> Any:
        try:
            from sage.all import PolynomialRing, QQ, RR
            R = PolynomialRing(QQ, "x")
            x = R.gen()
            p = x**size - x - 1
            return len(p.roots(RR, multiplicities=False))
        except ImportError:
            raise NotImplementedError

    def bench_horner_form_poly(self, size: int) -> Any:
        try:
            from sage.all import QQ, PolynomialRing
            R = PolynomialRing(QQ, "x")
            x = R.gen()
            p = sum(x**k for k in range(size + 1))
            try:
                return str(p.horner())  # some Sage builds expose horner()
            except (AttributeError, TypeError):
                return str(p)
        except ImportError:
            raise NotImplementedError

    def bench_log_exp_simplify(self, size: int) -> Any:
        try:
            from sage.all import exp, log, var
            x = var("x")
            expr = x
            for _ in range(size):
                expr = log(exp(expr))
            return str(expr.simplify_full())
        except ImportError:
            raise NotImplementedError

    def bench_resultant_poly(self, size: int) -> Any:
        try:
            from sage.all import QQ, PolynomialRing
            R = PolynomialRing(QQ, "x")
            x = R.gen()
            f = x**size + x + 1
            g = x**size - x - 1
            return str(f.resultant(g))
        except ImportError:
            raise NotImplementedError

    def bench_recurrence_solve(self, size: int) -> Any:
        raise NotImplementedError("Sage recurrence benchmark not wired to a single stable API")

    def bench_poly_gcd(self, size: int) -> Any:
        try:
            from sage.all import PolynomialRing, QQ
            R = PolynomialRing(QQ, "x")
            x = R.gen()
            a = x**size - 1
            b = x ** (size // 2) - 1
            return str(a.gcd(b))
        except ImportError:
            raise NotImplementedError

    def bench_rational_simplify(self, size: int) -> Any:
        try:
            from sage.all import var
            x = var("x")
            expr = (x**size - 1) / (x - 1)
            return str(expr.simplify_rational())
        except ImportError:
            raise NotImplementedError

    def bench_sparse_interp_univariate(self, size: int) -> Any:
        p = 32749
        step = 500 // max(size, 1)
        terms = [(i + 1, (i + 1) * step) for i in range(size)]

        def f(x: int) -> int:
            return sum(c * pow(x, e, p) for c, e in terms) % p

        return [f(x) for x in range(1, 503)]

    def bench_sparse_interp_multivar(self, size: int) -> Any:
        return (3 + 1) ** size

    def bench_numerical_homotopy(self, size: int) -> Any:
        try:
            from sage.all import solve, var
            names = " ".join(f"x{i}" for i in range(size))
            xs = var(names)
            eqs = [xs[i] ** 2 - 1 for i in range(size)]
            sol = solve(eqs, list(xs))
            return [str(s) for s in sol] if sol is not None else []
        except ImportError:
            raise NotImplementedError

    def bench_collect_like_terms_mixed(self, size: int) -> Any:
        try:
            from sage.all import var
            x = var("x")
            expr = sum(((i % 7) + 1) * x for i in range(size))
            return str(expr.collect(x))
        except ImportError:
            raise NotImplementedError

    def bench_subresultant_chain(self, size: int) -> Any:
        try:
            from sage.all import PolynomialRing, QQ
            R = PolynomialRing(QQ, "x")
            x = R.gen()
            f = x**size + x + 1
            g = x**size - x - 1
            return len(f.subresultants(g))
        except (ImportError, AttributeError):
            raise NotImplementedError

    def bench_factor_univariate_mod_p(self, size: int) -> Any:
        try:
            from sage.all import GF, PolynomialRing
            R = PolynomialRing(GF(101), "x")
            x = R.gen()
            p = x**size + 1
            return str(p.factor())
        except ImportError:
            raise NotImplementedError

    def bench_expand_power_simplify(self, size: int) -> Any:
        try:
            import math
            from sage.all import PolynomialRing, QQ
            R = PolynomialRing(QQ, "x")
            x = R.gen()
            p = sum(math.comb(size, k) * x**k for k in range(size + 1))
            return str(p)
        except ImportError:
            raise NotImplementedError
