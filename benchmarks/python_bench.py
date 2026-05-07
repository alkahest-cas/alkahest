"""
alkahest Python-level benchmark suite.

Measures wall-clock time and peak heap memory for the Python API surface.
Run with:

    python benchmarks/python_bench.py [--quick]

Use --quick for a fast smoke pass (~2 s); omit it for the full suite (~30 s).
"""

import argparse
import gc
import sys
import timeit
import tracemalloc
from dataclasses import dataclass

# Ensure the package is importable from the repo root.
sys.path.insert(0, "python")

import alkahest as _alkahest  # noqa: E402
from alkahest.alkahest import (  # noqa: E402
    ExprPool,
    MultiPoly,
    UniPoly,
    diff,
    simplify,
)


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    iterations: int
    mean_us: float       # mean time per iteration in microseconds
    peak_kb: float       # peak heap usage in kibibytes (tracemalloc)
    extra: str = ""      # optional annotation (e.g. log steps, pool size)


def bench(name: str, setup: str, stmt: str, iterations: int, repeat: int = 5) -> Result:
    """Time `stmt` with `setup` using timeit, return mean µs."""
    ns = timeit.repeat(stmt=stmt, setup=setup, number=iterations, repeat=repeat,
                       globals={"ExprPool": ExprPool, "UniPoly": UniPoly,
                                "MultiPoly": MultiPoly, "diff": diff, "simplify": simplify})
    mean_us = min(ns) / iterations * 1e6

    # Memory: run once under tracemalloc.
    gc.collect()
    tracemalloc.start()
    exec(setup, {"ExprPool": ExprPool, "UniPoly": UniPoly,  # noqa: S102
                 "MultiPoly": MultiPoly, "diff": diff, "simplify": simplify})
    exec(stmt, {"ExprPool": ExprPool, "UniPoly": UniPoly,  # noqa: S102
                "MultiPoly": MultiPoly, "diff": diff, "simplify": simplify})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return Result(name=name, iterations=iterations, mean_us=mean_us, peak_kb=peak / 1024)


def bench_fn(name: str, fn, n: int = 10_000, repeat: int = 5) -> Result:
    """Benchmark a zero-argument callable; return Result."""
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = timeit.default_timer()
        for _ in range(n):
            fn()
        times.append(timeit.default_timer() - t0)
    mean_us = min(times) / n * 1e6

    # Memory
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return Result(name=name, iterations=n, mean_us=mean_us, peak_kb=peak / 1024)


def build_poly(pool, x, coeffs):
    """Build coeffs[0] + coeffs[1]*x + … in `pool`."""
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        c_id = pool.integer(c)
        if i == 0:
            terms.append(c_id)
        else:
            xpow = x ** i
            terms.append(c_id * xpow if c != 1 else xpow)
    if not terms:
        return pool.integer(0)
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr


def print_table(results: list[Result]) -> None:
    w_name = max(len(r.name) for r in results) + 2
    header = f"{'Benchmark':<{w_name}}  {'Mean (µs)':>12}  {'Peak (KiB)':>12}  {'Notes'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<{w_name}}  {r.mean_us:>12.2f}  {r.peak_kb:>12.2f}  {r.extra}")


def write_markdown(results: list[Result], path) -> None:
    import datetime
    lines = [
        "# Alkahest Python microbenchmark results\n",
        f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "| Benchmark | Mean (µs) | Peak (KiB) | Notes |",
        "|-----------|-----------|------------|-------|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.mean_us:.2f} | {r.peak_kb:.2f} | {r.extra} |")
    from pathlib import Path as _Path
    _Path(path).write_text("\n".join(lines) + "\n")
    print(f"Markdown results written to {path}")


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


def run(n: int) -> list[Result]:
    results = []

    # -- ExprPool interning --------------------------------------------------

    def intern_cached():
        p = ExprPool()
        for _ in range(100):
            p.symbol("x")

    results.append(bench_fn("intern/symbol_cached_x100", intern_cached, n=n))

    def intern_unique():
        p = ExprPool()
        for i in range(100):
            p.integer(i)

    results.append(bench_fn("intern/integer_unique_100", intern_unique, n=n))

    # -- Simplify ------------------------------------------------------------

    def simplify_add_zero():
        p = ExprPool()
        x = p.symbol("x")
        expr = x + p.integer(0)
        return simplify(expr)

    r = bench_fn("simplify/add_zero", simplify_add_zero, n=n)
    ex = simplify_add_zero()
    r.extra = f"steps={len(ex.steps)}"
    results.append(r)

    def simplify_const_fold():
        p = ExprPool()
        expr = p.integer(3) + p.integer(4) + p.integer(5)
        return simplify(expr)

    results.append(bench_fn("simplify/const_fold_3terms", simplify_const_fold, n=n))

    for deg in [1, 2, 3, 4]:
        coeffs = list(range(1, deg + 2))

        def simplify_poly(coeffs=coeffs):
            p = ExprPool()
            x = p.symbol("x")
            expr = build_poly(p, x, coeffs)
            return simplify(expr)

        r = bench_fn(f"simplify/poly_deg{deg}", simplify_poly, n=n)
        ex = simplify_poly()
        r.extra = f"steps={len(ex.steps)}"
        results.append(r)

    # -- Diff ----------------------------------------------------------------

    for deg in [1, 2, 3, 4]:
        coeffs = list(range(1, deg + 2))

        def diff_poly(coeffs=coeffs):
            p = ExprPool()
            x = p.symbol("x")
            expr = build_poly(p, x, coeffs)
            return diff(expr, x)

        r = bench_fn(f"diff/poly_deg{deg}", diff_poly, n=n)
        ex = diff_poly()
        r.extra = f"steps={len(ex.steps)}"
        results.append(r)

    def diff_sin_x2():
        p = ExprPool()
        x = p.symbol("x")
        from alkahest.alkahest import sin
        expr = sin(x ** 2)
        return diff(expr, x)

    r = bench_fn("diff/sin_x_squared", diff_sin_x2, n=n)
    ex = diff_sin_x2()
    r.extra = f"steps={len(ex.steps)}"
    results.append(r)

    # -- UniPoly -------------------------------------------------------------

    for deg in [2, 4, 8]:
        coeffs = list(range(1, deg + 2))

        def make_unipoly(coeffs=coeffs):
            p = ExprPool()
            x = p.symbol("x")
            expr = build_poly(p, x, coeffs)
            return UniPoly.from_symbolic(expr, x)

        r = bench_fn(f"unipoly/from_symbolic_deg{deg}", make_unipoly, n=n)
        results.append(r)

    def unipoly_mul():
        p = ExprPool()
        x = p.symbol("x")
        f = UniPoly.from_symbolic(build_poly(p, x, [1, 2, 3, 4, 5]), x)
        g = UniPoly.from_symbolic(build_poly(p, x, [5, 4, 3, 2, 1]), x)
        return f * g

    results.append(bench_fn("unipoly/mul_deg4_x_deg4", unipoly_mul, n=n))

    # -- MultiPoly -----------------------------------------------------------

    def multipoly_bivariate():
        p = ExprPool()
        x = p.symbol("x")
        y = p.symbol("y")
        x2y = (x ** 2) * y
        xy2 = x * (y ** 2)
        expr = x2y + x * y + xy2 + x + y
        return MultiPoly.from_symbolic(expr, [x, y])

    results.append(bench_fn("multipoly/from_symbolic_bivariate", multipoly_bivariate, n=n))

    # -- Hash-consing sharing test -------------------------------------------

    def sharing_test():
        p = ExprPool()
        x = p.symbol("x")
        e1 = build_poly(p, x, [1, 2, 1])
        e2 = build_poly(p, x, [1, 2, 1])
        assert e1 == e2, "hash-consing broken"

    results.append(bench_fn("intern/hash_consing_verify", sharing_test, n=n))

    # -- Integration ---------------------------------------------------------

    def integrate_poly_deg4():
        p = ExprPool()
        x = p.symbol("x")
        expr = build_poly(p, x, [1, 2, 3, 4, 5])
        return _alkahest.integrate(expr, x)

    results.append(bench_fn("integrate/poly_deg4", integrate_poly_deg4, n=n))

    # -- Series --------------------------------------------------------------

    def series_sin_order8():
        p = ExprPool()
        x = p.symbol("x")
        return _alkahest.series(_alkahest.sin(x), x, p.integer(0), 8)

    results.append(bench_fn("series/sin_order8", series_sin_order8, n=n))

    # -- Limit ---------------------------------------------------------------

    def limit_sin_over_x():
        p = ExprPool()
        x = p.symbol("x")
        expr = _alkahest.sin(x) / x
        return _alkahest.limit(expr, x, p.integer(0))

    results.append(bench_fn("limit/sin_over_x", limit_sin_over_x, n=n))

    # -- Gradient ------------------------------------------------------------

    def grad_nvar_5():
        p = ExprPool()
        xs = [p.symbol(f"x{i}") for i in range(5)]
        f = xs[0] ** 2 + xs[1] ** 2 + xs[2] ** 2 + xs[3] ** 2 + xs[4] ** 2
        return _alkahest.symbolic_grad(f, xs)

    results.append(bench_fn("grad/nvar_5", grad_nvar_5, n=n))

    # -- Matrix determinant --------------------------------------------------

    def matrix_det_4x4():
        p = ExprPool()
        rows = [[p.symbol(f"a{i}{j}") for j in range(4)] for i in range(4)]
        return _alkahest.Matrix(rows).det()

    results.append(bench_fn("matrix/det_4x4", matrix_det_4x4, n=n))

    # -- Real roots ----------------------------------------------------------

    def real_roots_deg8():
        p = ExprPool()
        x = p.symbol("x")
        poly = x ** 8 - x - p.integer(1)
        return _alkahest.real_roots(poly, x)

    results.append(bench_fn("real_roots/deg8", real_roots_deg8, n=n))

    # -- Horner form ---------------------------------------------------------

    def horner_deg20():
        p = ExprPool()
        x = p.symbol("x")
        terms = [p.integer(1)] + [x ** k for k in range(1, 21)]
        poly = terms[0]
        for t in terms[1:]:
            poly = poly + t
        return _alkahest.horner(poly, x)

    results.append(bench_fn("horner/deg20", horner_deg20, n=n))

    # -- Log-exp simplify ----------------------------------------------------

    def log_exp_simplify_depth4():
        p = ExprPool()
        x = p.symbol("x")
        expr = x
        for _ in range(4):
            expr = _alkahest.log(_alkahest.exp(expr))
        return _alkahest.simplify_log_exp(expr)

    results.append(bench_fn("log_exp/simplify_depth4", log_exp_simplify_depth4, n=n))

    # -- Collect like terms --------------------------------------------------

    def collect_terms_50():
        p = ExprPool()
        x = p.symbol("x")
        terms = [x] * 50 + [x ** 2] * 50
        expr = terms[0]
        for t in terms[1:]:
            expr = expr + t
        return _alkahest.collect_like_terms(expr)

    results.append(bench_fn("collect/terms_50", collect_terms_50, n=n))

    # -- Resultant -----------------------------------------------------------

    def resultant_deg8():
        p = ExprPool()
        x = p.symbol("x")
        one = p.integer(1)
        f = x ** 8 + x + one
        g = x ** 8 - x - one
        return _alkahest.resultant(f, g, x)

    results.append(bench_fn("resultant/deg8", resultant_deg8, n=n))

    # -- Recurrence solve (Fibonacci) ----------------------------------------

    def rsolve_fibonacci():
        p = ExprPool()
        n_sym = p.symbol("n")
        a_n = p.func("a", [n_sym])
        a_n1 = p.func("a", [n_sym + p.integer(1)])
        a_n2 = p.func("a", [n_sym + p.integer(2)])
        eq = a_n2 - a_n1 - a_n
        return _alkahest.rsolve(eq, n_sym, "a", {0: p.integer(1), 1: p.integer(1)})

    results.append(bench_fn("rsolve/fibonacci", rsolve_fibonacci, n=n))

    # -- Emit C --------------------------------------------------------------

    def emit_c_deg20():
        p = ExprPool()
        x = p.symbol("x")
        poly = build_poly(p, x, list(range(1, 22)))
        return _alkahest.emit_c(poly, x)

    results.append(bench_fn("emit_c/deg20", emit_c_deg20, n=n))

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduce iterations for a fast smoke pass")
    parser.add_argument("--output", default=None,
                        help="Write results to a Markdown file at this path")
    args = parser.parse_args()

    n = 200 if args.quick else 2_000
    print(f"\nalkahest Python benchmark  (iterations={n})\n")
    results = run(n)
    print_table(results)
    print()
    if args.output:
        write_markdown(results, args.output)
