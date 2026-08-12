# Polynomial system solving

Alkahest solves systems of polynomial equations symbolically using Gröbner bases.

## solve

`solve` finds the solutions of a system of polynomial equations in a list of variables. It uses the `groebner` Cargo feature, which is **included in all PyPI wheels** (default feature since 2.3.1) and in all source builds.

```python
from alkahest import ExprPool, solve, sqrt

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

# Linear system
solutions = solve([x + y - pool.integer(1), x - y], [x, y])
# → [{x: 1/2, y: 1/2}]

# Circle intersected with a line: irrational solutions
solutions = solve(
    [x**2 + y**2 - pool.integer(1), y - x],
    [x, y]
)
# → [{x: sqrt(2)/2, y: sqrt(2)/2}, {x: -sqrt(2)/2, y: -sqrt(2)/2}]

# Parametric solve: free symbols omitted from `vars` stay as parameters
solutions = solve([x**2 - y], [x])
# → [{x: sqrt(y)}, {x: -sqrt(y)}]
```

Solutions are symbolic: irrational roots are returned as `Expr` trees (e.g. `sqrt(2)/2`) rather than floats. Quadratic elimination produces exact symbolic answers. Free symbols that appear in the equations but are not listed in `vars` are treated as parameters, so solutions may depend on those symbols.

### Solution types

The return value is a list of dicts mapping `Expr` variable → `Expr` solution:

```python
for sol in solutions:
    for var, val in sol.items():
        print(f"{var} = {val}")
        # Evaluate numerically if needed
        from alkahest import eval_expr
        numeric = eval_expr(val, {})
```

`solve` returns an empty list for inconsistent systems and a `GroebnerBasis` handle for parametric families (infinite solution sets).

Pass `numeric=True` to return float values directly: `solve(eqs, vars, numeric=True)`.

## GroebnerBasis

A `GroebnerBasis` can be constructed directly for ideal-theoretic operations:

```python
from alkahest import GroebnerBasis, GbPoly

# Compute a Gröbner basis under GrLex order
polys = [x**2 + y**2 - pool.integer(1), x - y]
gb = GroebnerBasis.compute(polys, [x, y])

# Check ideal membership
print(gb.contains(x - pool.rational(1, 2)))  # False

# Reduce a polynomial modulo the ideal
reduced = gb.reduce(x**3 + y**3)
```


### Monomial orders

Supported orders: `Lex` (lexicographic), `GrLex` (graded lexicographic), `GRevLex` (graded reverse lexicographic). `GRevLex` is generally fastest for basis computation; `Lex` is required for elimination.

### Parallel F4

With `--features "groebner parallel"`, Gröbner basis computation uses Rayon for parallel S-polynomial reduction via the F4 algorithm.

### GPU-accelerated Macaulay matrix (groebner-cuda) — Rust only, not wired into the solver

`--features "groebner-cuda"` compiles a CUDA kernel for the mod-p row reduction of the Macaulay matrix, with multi-prime CRT lifts reconstructing rational coefficients, and falls back to pure-Rust row reduction when no CUDA device is present.

**It does not accelerate anything on this page.** `GroebnerBasis.compute`, `solve` and `triangularize` run Buchberger/F4 on the CPU regardless; the GPU routine is reachable only as the Rust function `alkahest_cas::poly::groebner::compute_groebner_basis_gpu`, and production dispatch deliberately does not prefer it until the [benchmark harness](https://github.com/alkahest-cas/alkahest/blob/main/docs/symbolic-gpu-benchmarks.md) says it wins. There is correspondingly **no `capabilities()["features"]["groebner_cuda"]` bit** — it used to exist and report that the kernel had been compiled in, which no Python observation could confirm or refute, so 3.8 removed it. See [GPU support](./gpu.md#groebner-cuda-is-not-reachable-from-python).

Because the Rust entry point falls back to CPU row reduction when no device is present, it returns a `GpuBackendReport` alongside the basis: `let (basis, backend) = compute_groebner_basis_gpu(gens, order, Some(0))?;` and `backend.ran_on_gpu()` is the only way to tell a real GPU run from a fallback, since the basis is identical either way.

## Elimination ideals

`GroebnerBasis.eliminate` computes the elimination ideal by dropping generators involving specified variables:

```python
# Eliminate y to get a univariate ideal in x
x_ideal = gb.eliminate([y])
```

This is the algebraic geometry operation underlying implicitization of parametric curves and surfaces.

## Performance

On the `solve_circle_line` benchmark (2-variable quadratic system), Alkahest is approximately **40× faster** than SymPy due to the FLINT-backed polynomial arithmetic and the compiled F4 core.

**Upcoming (v2.0):** F5 / signature-based Gröbner basis, real root isolation, primary decomposition, and other advanced algorithms.
