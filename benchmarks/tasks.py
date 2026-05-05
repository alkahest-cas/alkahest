"""Benchmark task catalogue for the cross-CAS comparison driver.

PA-4 — Cross-CAS benchmark driver.

Each task class exposes:
  - ``name: str``
  - ``size_params: list[int]``  (the "size" axis — degree, dimension, …)
  - ``run_alkahest(size) -> object``
  - ``run_sympy(size) -> object``   (optional; skipped if SymPy not installed)
  - ``expected_result(size) -> object | None``  (for correctness cross-checks)

Add new tasks by subclassing :class:`BenchTask`.
"""

from __future__ import annotations

import abc
import math
from typing import Any


class BenchTask(abc.ABC):
    """Abstract base class for a benchmark task."""

    name: str = ""
    size_params: list[int] = [5, 10, 20]

    @abc.abstractmethod
    def run_alkahest(self, size: int) -> Any:
        """Run the task using Alkahest and return the result."""

    def run_sympy(self, size: int) -> Any:  # noqa: ARG002
        """Run the task using SymPy (optional)."""
        raise NotImplementedError

    def expected_result(self, size: int) -> Any:  # noqa: ARG002
        """Return the expected result for a correctness cross-check."""
        return None


# ---------------------------------------------------------------------------
# Task 1 — Polynomial differentiation: d/dx of x^n + x^{n-1} + … + 1
# ---------------------------------------------------------------------------


class DegreeNPolyDiff(BenchTask):
    """Differentiate a degree-N polynomial."""

    name = "poly_diff"
    size_params = [10, 50, 100, 200]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        # Build 1 + x + x^2 + … + x^size
        terms = [p.integer(1)]
        for k in range(1, size + 1):
            terms.append(x ** k)
        poly = sum(terms[1:], terms[0])
        result = alkahest.diff(poly, x)
        return result.value

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        poly = sum(x**k for k in range(size + 1))
        return sp.diff(poly, x)

    def expected_result(self, size: int) -> Any:
        # Leading coefficient should be `size`
        return size


# ---------------------------------------------------------------------------
# Task 2 — Polynomial GCD
# ---------------------------------------------------------------------------


class DegreeNPolyGCD(BenchTask):
    """GCD of two degree-N polynomials that share a common factor."""

    name = "poly_gcd"
    size_params = [5, 10, 20, 40]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        # Build (x-1)^size  and  (x-1)^(size//2) * (x+1)^(size//2)
        # GCD should be (x-1)^(size//2)
        one = p.integer(1)
        xm1 = p.add([x, p.mul([p.integer(-1), one])])  # x - 1
        # use poly conversion
        poly_a = alkahest.UniPoly.from_symbolic(
            alkahest.poly_normal(x ** size - one, [x]).value
            if hasattr(alkahest, "poly_normal")
            else x ** size,
            x,
        )
        poly_b = alkahest.UniPoly.from_symbolic(
            alkahest.poly_normal(x ** (size // 2) - one, [x]).value
            if hasattr(alkahest, "poly_normal")
            else x ** (size // 2),
            x,
        )
        return alkahest.UniPoly.gcd(poly_a, poly_b)

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        a = x**size - 1
        b = x ** (size // 2) - 1
        return sp.gcd(a, b)


# ---------------------------------------------------------------------------
# Task 3 — Rational simplification
# ---------------------------------------------------------------------------


class RationalSimplification(BenchTask):
    """Simplify (x^n - 1) / (x - 1) to 1 + x + … + x^{n-1}."""

    name = "rational_simplify"
    size_params = [5, 10, 20]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        numer_uni = alkahest.UniPoly.from_coefficients([p.integer(-1)] + [p.integer(0)] * (size - 1) + [p.integer(1)], x)
        denom_uni = alkahest.UniPoly.from_coefficients([p.integer(-1), p.integer(1)], x)
        rf = alkahest.RationalFunction(numer_uni, denom_uni)
        return rf

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        expr = (x**size - 1) / (x - 1)
        return sp.simplify(expr)


# ---------------------------------------------------------------------------
# Task 4 — Trigonometric identity simplification: sin²(x) + cos²(x) → 1
# ---------------------------------------------------------------------------


class TrigIdentitySimplify(BenchTask):
    """Simplify N copies of sin²(x) + cos²(x) down to N."""

    name = "trig_identity"
    size_params = [1, 5, 10, 20]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        expr = p.integer(0)
        for _ in range(size):
            sin2 = alkahest.sin(x) ** 2
            cos2 = alkahest.cos(x) ** 2
            expr = expr + sin2 + cos2
        return alkahest.simplify_trig(expr)

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        expr = sum(sp.sin(x) ** 2 + sp.cos(x) ** 2 for _ in range(size))
        return sp.trigsimp(expr)

    def expected_result(self, size: int) -> Any:
        return size


# ---------------------------------------------------------------------------
# Task 5 — Jacobian of an N×N system
# ---------------------------------------------------------------------------


class JacobianNxN(BenchTask):
    """Compute the Jacobian of an N-variable polynomial system."""

    name = "jacobian_nxn"
    size_params = [3, 5, 8, 10]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        xs = [p.symbol(f"x{i}") for i in range(size)]
        # f_i = x_i^2 + x_{i-1} * x_i  (wraps around)
        fns = [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
        mat = alkahest.Matrix(fns)
        return alkahest.jacobian(mat, xs)

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        xs = [sp.Symbol(f"x{i}") for i in range(size)]
        fns = [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
        return sp.Matrix(fns).jacobian(xs)


# ---------------------------------------------------------------------------
# Task 6 — Ball arithmetic: rigorous sin(cos(x)) at a narrow interval
# ---------------------------------------------------------------------------


class BallSinCos(BenchTask):
    """Rigorous evaluation of sin(cos(x)) at x ∈ [1 ± ε], varying ε."""

    name = "ball_sin_cos"
    size_params = [1, 10, 100, 1000]  # size = 1/ε factor (larger = tighter ball)

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        expr = alkahest.sin(alkahest.cos(x))
        radius = 1.0 / size
        ball = alkahest.ArbBall.from_midpoint_radius(1.0, radius, 128)
        ev = alkahest.interval_eval(expr, {x: ball})
        return ev

    def expected_result(self, size: int) -> Any:
        import math

        return math.sin(math.cos(1.0))


# ---------------------------------------------------------------------------
# Task 7 — JIT compilation and evaluation of a polynomial
# ---------------------------------------------------------------------------


class ODEJITCompile(BenchTask):
    """Compile a degree-N polynomial and evaluate it at 10^6 points."""

    name = "poly_jit_eval"
    size_params = [5, 10, 20]

    def run_alkahest(self, size: int) -> Any:
        import numpy as np
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        terms = [x**k for k in range(size + 1)]
        poly = terms[0]
        for t in terms[1:]:
            poly = poly + t
        compiled = alkahest.compile_expr(poly, [x])
        xs = np.linspace(0.0, 1.0, 1_000_000)
        return alkahest.numpy_eval(compiled, xs)

    def run_sympy(self, size: int) -> Any:
        import numpy as np
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        poly = sum(x**k for k in range(size + 1))
        fn = sp.lambdify(x, poly, "numpy")
        xs = np.linspace(0.0, 1.0, 1_000_000)
        return fn(xs)


# ---------------------------------------------------------------------------
# Task 8 — V1-4 polynomial system solver (circle ∩ line)
# ---------------------------------------------------------------------------


class SolveCircleLine(BenchTask):
    """Solve the intersection of an N-circle with the line y = x.

    Each size yields a 2-equation, 2-variable system whose Gröbner basis
    triangularises to a univariate quadratic.  The circle radius scales
    with ``size`` so the basis computation has meaningfully different
    coefficients per run; the number of real solutions stays at 2.
    """

    name = "solve_circle_line"
    size_params = [1, 2, 5, 10]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        y = p.symbol("y")
        neg_one = p.integer(-1)
        r2 = p.integer(size * size)
        # x² + y² - r²
        eq1 = x ** 2 + y ** 2 + neg_one * r2
        # y - x
        eq2 = y + neg_one * x
        return alkahest.solve([eq1, eq2], [x, y])

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x, y = sp.symbols("x y")
        return sp.solve([x ** 2 + y ** 2 - size ** 2, y - x], [x, y], dict=True)

    def expected_result(self, size: int) -> Any:
        return 2  # number of real solutions


# ---------------------------------------------------------------------------
# Task — V2-11: triangular decomposition proxy (6R IK row name in ROADMAP)
# ---------------------------------------------------------------------------


class Solve6rIk(BenchTask):
    """Planar ``circle ∩ line`` constraints — proxy for IK-style polynomial solving.

    Full 6R inverse kinematics is not inlined here; the system matches the
    :class:`SolveCircleLine` structure (two equations, two variables).  The
    Alkahest path benchmarks :func:`alkahest.triangularize`; SymPy uses a lex
    Gröbner basis as the comparison decomposition.
    """

    name = "solve_6r_ik"
    size_params = [1, 2, 5]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        y = p.symbol("y")
        neg_one = p.integer(-1)
        r2 = p.integer(size * size)
        eq1 = x**2 + y**2 + neg_one * r2
        eq2 = y + neg_one * x
        return alkahest.triangularize([eq1, eq2], [x, y])

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x, y = sp.symbols("x y")
        return sp.groebner([x**2 + y**2 - size**2, y - x], y, x, order="lex")


# ---------------------------------------------------------------------------
# Task 9 — V2-3: Sparse interpolation vs dense (univariate)
# ---------------------------------------------------------------------------


class SparseInterpVsDense(BenchTask):
    """Recover a T-term univariate polynomial.

    ``size`` = number of terms T.  Sparse path uses 2·T evaluations;
    dense path uses ``max_degree + 1`` evaluations.
    Demonstrates O(T) vs O(D) query complexity.
    """

    name = "sparse_interp_univariate"
    size_params = [2, 5, 10, 20]

    _PRIME = 32749
    _MAX_DEGREE = 500  # degree of the leading monomial; dense needs 501 evals

    def _make_poly(self, size: int) -> list[tuple[int, int]]:
        """Return (coeff, exp) for a size-term polynomial with high max degree."""
        import math

        p = self._PRIME
        # Spread T terms evenly across [1, MAX_DEGREE]
        step = self._MAX_DEGREE // max(size, 1)
        return [(i + 1, (i + 1) * step) for i in range(size)]

    def _oracle(self, terms: list[tuple[int, int]]) -> "callable":
        p = self._PRIME

        def f(x: int) -> int:
            return sum(c * pow(x, e, p) for c, e in terms) % p

        return f

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        terms = self._make_poly(size)
        oracle = self._oracle(terms)
        return alkahest.sparse_interp_univariate(oracle, size + 2, self._PRIME)

    def run_sympy(self, size: int) -> Any:
        """Dense interpolation baseline: evaluate at max_degree+1 points."""
        # Dense interpolation from max_degree+1 evaluations (naive Python)
        p = self._PRIME
        terms = self._make_poly(size)

        def f(x: int) -> int:
            return sum(c * pow(x, e, p) for c, e in terms) % p

        d = self._MAX_DEGREE
        pts = list(range(1, d + 2))
        vals = [f(x) for x in pts]
        # Lagrange interpolation (just count the evals as the benchmark)
        return vals  # return raw evaluations as the "result"

    def expected_result(self, size: int) -> Any:
        return size  # number of recovered terms


# ---------------------------------------------------------------------------
# Task 10 — V2-3: Multivariate sparse interpolation (20 variables)
# ---------------------------------------------------------------------------


class SparseInterpMultivar(BenchTask):
    """Recover a T-term polynomial in N variables.

    ``size`` = number of variables N.  Demonstrates ≥5× fewer oracle
    calls than the dense path O((D+1)^N).
    """

    name = "sparse_interp_multivar"
    size_params = [2, 5, 10, 20]

    _PRIME = 32749
    _TERMS_PER_SIZE = {2: 3, 5: 5, 10: 10, 20: 15}

    def _make_terms(self, n_vars: int, n_terms: int) -> list[tuple[int, list[int]]]:
        """Build a canonical sparse polynomial with n_terms monomials in n_vars."""
        terms = []
        for i in range(n_terms):
            coeff = (i + 1) * 7 % self._PRIME or 1
            exp = [0] * n_vars
            # Use a single variable per term (diagonal structure for reproducibility)
            var_idx = i % n_vars
            exp[var_idx] = (i % 3) + 1  # degree 1, 2, or 3
            terms.append((coeff, exp))
        return terms

    def _oracle(self, terms, n_vars: int) -> "callable":
        p = self._PRIME

        def f(pt: list[int]) -> int:
            acc = 0
            for coeff, exps in terms:
                term = coeff
                for vi, e in enumerate(exps):
                    if vi < len(pt) and e > 0:
                        term = term * pow(pt[vi], e, p) % p
                acc = (acc + term) % p
            return acc

        return f

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        n_vars = size
        n_terms = self._TERMS_PER_SIZE.get(size, min(size, 15))
        terms = self._make_terms(n_vars, n_terms)
        oracle = self._oracle(terms, n_vars)

        pool = alkahest.ExprPool()
        vs = [pool.symbol(f"x{i}") for i in range(n_vars)]
        return alkahest.sparse_interp(
            oracle, vs,
            term_bound=n_terms + 5,
            degree_bound=4,
            prime=self._PRIME,
            seed=0,
        )

    def run_sympy(self, size: int) -> Any:
        """Dense baseline: count oracle calls required by dense interpolation."""
        # Dense interpolation over N variables with max degree D=3 requires
        # (D+1)^N evaluations.  For N=20, D=3: 4^20 ≈ 10^12 — completely infeasible.
        # We return the evaluation count as a stand-in.
        n_vars = size
        degree = 3
        dense_eval_count = (degree + 1) ** n_vars
        return dense_eval_count

    def expected_result(self, size: int) -> Any:
        return self._TERMS_PER_SIZE.get(size, min(size, 15))


# ---------------------------------------------------------------------------
# V2-14 — Homotopy continuation (numerical algebraic geometry)
# ---------------------------------------------------------------------------


class HomotopySeparateQuadratics(BenchTask):
    """Separate quadratics Π (x_i^2 − 1) — classic Bézout-total-degree case.

    ``size`` controls the dimension ``n``.  Path count equals ``2^n``.
    """

    name = "numerical_homotopy"
    size_params = [2, 3]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        neg1 = p.integer(-1)
        xs = [p.symbol(f"x{i}") for i in range(size)]
        eqs = [x_i**2 + neg1 for x_i in xs]
        return alkahest.solve(eqs, xs, method="homotopy")

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        xs = sp.symbols(" ".join(f"x{i}" for i in range(size)))
        eqs = [xs[i] ** 2 - 1 for i in range(size)]
        return sp.solve(eqs, xs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TASKS: list[BenchTask] = [
    DegreeNPolyDiff(),
    TrigIdentitySimplify(),
    JacobianNxN(),
    BallSinCos(),
    ODEJITCompile(),
    SolveCircleLine(),
    Solve6rIk(),
    HomotopySeparateQuadratics(),
    SparseInterpVsDense(),
    SparseInterpMultivar(),
]
