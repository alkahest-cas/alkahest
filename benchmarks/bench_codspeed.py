"""
CodSpeed benchmarks for the Alkahest Python API.

This file wraps the existing BenchTask.run_alkahest() methods from tasks.py
and a selection of callables from python_bench.py.  It does NOT replace either
of those runners — cas_comparison.py and python_bench.py continue to work
unchanged.

Run locally (wall-clock, no instrumentation):
    pytest benchmarks/bench_codspeed.py -v

Run with CodSpeed instrumentation (instruction count, used in CI):
    pytest benchmarks/bench_codspeed.py --codspeed

Each test_* function receives the ``benchmark`` fixture from pytest-codspeed
and calls benchmark(fn) — CodSpeed controls iteration counts automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `from tasks import ...` and `from python_bench import ...` resolve
# regardless of whether pytest is invoked from the repo root or benchmarks/.
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import pytest

import alkahest as _alk
from alkahest.alkahest import ExprPool, MultiPoly, UniPoly, diff, simplify

from python_bench import build_poly
from tasks import (
    ALL_TASKS,
    BallSinCos,
    CollectLikeTermsMixed,
    DegreeNPolyDiff,
    DegreeNPolyGCD,
    ExpandPowerSimplify,
    FactorUniModP,
    GradientNVar,
    HomotopySeparateQuadratics,
    HornerFormPoly,
    IntegratePoly,
    JacobianNxN,
    LimitComputation,
    LogExpSimplify,
    MatrixDetNxN,
    ODEJITCompile,
    RationalSimplification,
    RealRootsPoly,
    RecurrenceSolve,
    ResultantPoly,
    SeriesExpansion,
    Solve6rIk,
    SolveCircleLine,
    SparseInterpMultivar,
    SparseInterpVsDense,
    SubresultantChain,
    TrigIdentitySimplify,
)

# ---------------------------------------------------------------------------
# Groebner availability guard
# ---------------------------------------------------------------------------
# Tasks that require the groebner Cargo feature; skip gracefully when missing.
_HAS_GROEBNER = hasattr(_alk, "solve") and hasattr(_alk, "triangularize")


# ---------------------------------------------------------------------------
# Helper: instantiate each task once at module level
# ---------------------------------------------------------------------------
_POLY_DIFF = DegreeNPolyDiff()
_INTEGRATE = IntegratePoly()
_SERIES = SeriesExpansion()
_LIMIT = LimitComputation()
_POLY_GCD = DegreeNPolyGCD()
_RATIONAL = RationalSimplification()
_RESULTANT = ResultantPoly()
_SUBRESULTANT = SubresultantChain()
_FACTOR_MODP = FactorUniModP()
_REAL_ROOTS = RealRootsPoly()
_HORNER = HornerFormPoly()
_EXPAND = ExpandPowerSimplify()
_JACOBIAN = JacobianNxN()
_GRADIENT = GradientNVar()
_MATRIX_DET = MatrixDetNxN()
_TRIG = TrigIdentitySimplify()
_LOG_EXP = LogExpSimplify()
_COLLECT = CollectLikeTermsMixed()
_SOLVE_CL = SolveCircleLine()
_SOLVE_IK = Solve6rIk()
_HOMOTOPY = HomotopySeparateQuadratics()
_BALL = BallSinCos()
_JIT = ODEJITCompile()
_SPARSE_UNI = SparseInterpVsDense()
_SPARSE_MV = SparseInterpMultivar()
_RSOLVE = RecurrenceSolve()


# ===========================================================================
# Section 1: Differentiation / series / limits / integration
# ===========================================================================


def test_poly_diff_deg50(benchmark):
    """Differentiate 1 + x + … + x^50 (size=50)."""
    benchmark(_POLY_DIFF.run_alkahest, 50)


def test_integrate_poly_deg16(benchmark):
    """Integrate 1 + x + … + x^16 (size=16)."""
    benchmark(_INTEGRATE.run_alkahest, 16)


def test_series_sin_order12(benchmark):
    """Taylor series of sin(x) to 12 terms."""
    benchmark(_SERIES.run_alkahest, 12)


def test_limit_poly_deg8(benchmark):
    """Limit (x^8 − 1)/(x − 1) as x → 1."""
    benchmark(_LIMIT.run_alkahest, 8)


# ===========================================================================
# Section 2: Polynomials & rational functions
# ===========================================================================


def test_poly_gcd_deg20(benchmark):
    """GCD of two degree-20 polynomials sharing a common factor."""
    benchmark(_POLY_GCD.run_alkahest, 20)


def test_rational_simplify_deg10(benchmark):
    """Simplify (x^10 − 1)/(x − 1)."""
    benchmark(_RATIONAL.run_alkahest, 10)


def test_resultant_deg8(benchmark):
    """Resultant of two degree-8 polynomials."""
    benchmark(_RESULTANT.run_alkahest, 8)


def test_subresultant_chain_deg8(benchmark):
    """Subresultant PRS for two degree-8 polynomials."""
    benchmark(_SUBRESULTANT.run_alkahest, 8)


def test_factor_univariate_mod_p_deg16(benchmark):
    """Factor 1 + x^16 over GF(101)."""
    benchmark(_FACTOR_MODP.run_alkahest, 16)


def test_real_roots_deg8(benchmark):
    """Isolate real roots of x^8 − x − 1."""
    benchmark(_REAL_ROOTS.run_alkahest, 8)


def test_horner_form_deg50(benchmark):
    """Convert 1 + x + … + x^50 to Horner form."""
    benchmark(_HORNER.run_alkahest, 50)


def test_expand_power_deg14(benchmark):
    """Collect (x+1)^14 after full expansion."""
    benchmark(_EXPAND.run_alkahest, 14)


# ===========================================================================
# Section 3: Linear algebra & AD-style
# ===========================================================================


def test_jacobian_8x8(benchmark):
    """Jacobian of an 8-variable polynomial system."""
    benchmark(_JACOBIAN.run_alkahest, 8)


def test_gradient_nvar16(benchmark):
    """Gradient of x0² + … + x15²."""
    benchmark(_GRADIENT.run_alkahest, 16)


def test_matrix_det_4x4(benchmark):
    """Determinant of a 4×4 symbolic matrix."""
    benchmark(_MATRIX_DET.run_alkahest, 4)


# ===========================================================================
# Section 4: Simplification families
# ===========================================================================


def test_trig_identity_size10(benchmark):
    """Simplify 10 copies of sin²(x) + cos²(x)."""
    benchmark(_TRIG.run_alkahest, 10)


def test_log_exp_simplify_depth4(benchmark):
    """Simplify depth-4 nested log(exp(…)) chain."""
    benchmark(_LOG_EXP.run_alkahest, 4)


def test_collect_like_terms_64(benchmark):
    """collect_like_terms on 64 linear x terms."""
    benchmark(_COLLECT.run_alkahest, 64)


# ===========================================================================
# Section 5: Solvers & decomposition (require groebner feature)
# ===========================================================================


def test_solve_circle_line_size5(benchmark):
    """Solve x² + y² = 25, y = x (Gröbner)."""
    if not _HAS_GROEBNER:
        pytest.skip("requires --features groebner")
    benchmark(_SOLVE_CL.run_alkahest, 5)


def test_solve_6r_ik_size2(benchmark):
    """Triangularize 2-equation circle/line system (Gröbner)."""
    if not _HAS_GROEBNER:
        pytest.skip("requires --features groebner")
    benchmark(_SOLVE_IK.run_alkahest, 2)


def test_numerical_homotopy_size3(benchmark):
    """Homotopy solve Π(xi² − 1) in 3 variables."""
    if not _HAS_GROEBNER:
        pytest.skip("requires --features groebner")
    benchmark(_HOMOTOPY.run_alkahest, 3)


# ===========================================================================
# Section 6: Rigorous / numeric paths
# ===========================================================================


def test_ball_sin_cos_eps1e2(benchmark):
    """Rigorous sin(cos(x)) at x ∈ [1 ± 1/100] using Arb ball arithmetic."""
    benchmark(_BALL.run_alkahest, 100)


def test_poly_jit_eval_deg10(benchmark):
    """JIT-compile a degree-10 polynomial and evaluate at 1 000 points.

    The original tasks.py variant uses 10^6 points — fine for wall-clock runs
    but far too slow under Valgrind instruction-counting.  This wrapper caps
    the evaluation at 1 000 points so the benchmark stays within CodSpeed's
    time budget while still exercising both the compilation path and the
    vectorised eval kernel.  numpy is an optional dep in CI; skip gracefully
    when absent.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not installed — skipping JIT eval benchmark")

    def _jit_small():
        import alkahest
        p = alkahest.ExprPool()
        x = p.symbol("x")
        terms = [x ** k for k in range(11)]
        poly = terms[0]
        for t in terms[1:]:
            poly = poly + t
        compiled = alkahest.compile_expr(poly, [x])
        xs = np.linspace(0.0, 1.0, 1_000)
        return alkahest.numpy_eval(compiled, xs)

    benchmark(_jit_small)


# ===========================================================================
# Section 7: Sparse interpolation
# ===========================================================================


def test_sparse_interp_univariate_10terms(benchmark):
    """Recover a 10-term univariate polynomial via sparse interpolation."""
    benchmark(_SPARSE_UNI.run_alkahest, 10)


def test_sparse_interp_multivar_2vars(benchmark):
    """Recover a sparse polynomial in 2 variables (CodSpeed-safe size).

    The 10-variable variant runs for hours under Valgrind instruction-counting
    and has never completed within CI's 2-hour timeout.  2 variables (3 terms)
    gives a meaningful correctness + performance signal at a fraction of the
    cost; wall-clock runs via cas_comparison.py can still use larger sizes.
    """
    benchmark(_SPARSE_MV.run_alkahest, 2)


# ===========================================================================
# Section 8: Recurrences
# ===========================================================================


def test_rsolve_fibonacci(benchmark):
    """Solve Fibonacci recurrence in closed form."""
    benchmark(_RSOLVE.run_alkahest, 2)


# ===========================================================================
# Section 9: ExprPool micros (from python_bench.py, not in tasks.py)
# These cover the low-level kernel operations that the cross-CAS tasks
# don't exercise directly.
# ===========================================================================


def test_intern_symbol_cached(benchmark):
    """ExprPool.symbol('x') × 100 — hits the intern cache every time."""
    def fn():
        p = ExprPool()
        for _ in range(100):
            p.symbol("x")
    benchmark(fn)


def test_intern_integer_unique_100(benchmark):
    """ExprPool.integer(i) × 100 unique values — table grows each call."""
    def fn():
        p = ExprPool()
        for i in range(100):
            p.integer(i)
    benchmark(fn)


def test_intern_hash_consing(benchmark):
    """Build 1 + 2x + x² twice; assert structural sharing gives equal IDs."""
    def fn():
        p = ExprPool()
        x = p.symbol("x")
        e1 = build_poly(p, x, [1, 2, 1])
        e2 = build_poly(p, x, [1, 2, 1])
        assert e1 == e2
    benchmark(fn)


def test_simplify_add_zero(benchmark):
    """simplify(x + 0) — trivial rule fires immediately."""
    def fn():
        p = ExprPool()
        x = p.symbol("x")
        return simplify(x + p.integer(0))
    benchmark(fn)


def test_simplify_const_fold(benchmark):
    """simplify(3 + 4 + 5) — constant folding."""
    def fn():
        p = ExprPool()
        return simplify(p.integer(3) + p.integer(4) + p.integer(5))
    benchmark(fn)


def test_diff_sin_x_squared(benchmark):
    """d/dx sin(x²) — chain rule through a transcendental."""
    def fn():
        p = ExprPool()
        x = p.symbol("x")
        return diff(_alk.sin(x ** 2), x)
    benchmark(fn)


def test_unipoly_from_symbolic_deg8(benchmark):
    """UniPoly.from_symbolic on a degree-8 dense polynomial."""
    def fn():
        p = ExprPool()
        x = p.symbol("x")
        expr = build_poly(p, x, list(range(1, 10)))   # deg 8, 9 terms
        return UniPoly.from_symbolic(expr, x)
    benchmark(fn)


def test_unipoly_mul_deg4_x_deg4(benchmark):
    """Multiply two degree-4 UniPolys."""
    def fn():
        p = ExprPool()
        x = p.symbol("x")
        f = UniPoly.from_symbolic(build_poly(p, x, [1, 2, 3, 4, 5]), x)
        g = UniPoly.from_symbolic(build_poly(p, x, [5, 4, 3, 2, 1]), x)
        return f * g
    benchmark(fn)


def test_multipoly_from_symbolic_bivariate(benchmark):
    """MultiPoly.from_symbolic for x²y + xy² + xy + x + y."""
    def fn():
        p = ExprPool()
        x, y = p.symbol("x"), p.symbol("y")
        expr = (x ** 2) * y + x * (y ** 2) + x * y + x + y
        return MultiPoly.from_symbolic(expr, [x, y])
    benchmark(fn)
