# Getting started

## Install

### PyPI (recommended)

Alkahest is on the [Python Package Index](https://pypi.org/project/alkahest/):

```bash
pip install alkahest
```

Default wheels omit the LLVM JIT feature so installs stay small and avoid a runtime dependency on LLVM. Numeric APIs still work through the interpreter; for native LLVM CPU JIT, use an opt-in JIT-enabled wheel or [build from source](#from-source) (see also the project `README` on GitHub).

**JIT-enabled wheels (optional):** tagged releases attach Linux manylinux `+jit` wheels on [GitHub Releases](https://github.com/alkahest-cas/alkahest/releases). Install the `.whl` whose tags match your Python and platform, or build from source on macOS and Windows where JIT wheels are not built in CI yet.

### From source

For optional Cargo features (`jit`, `groebner`, `cuda`, …), GPU/NVPTX, or development, build the PyO3 extension with [maturin](https://github.com/PyO3/maturin).

Prerequisites (typical): **Rust** stable (≥ 1.76) and nightly, **LLVM 15**, **FLINT** (≥ 2.9, pulls in GMP/MPFR). See the repository `README` for distro-specific package names.

```bash
pip install maturin
git clone https://github.com/alkahest-cas/alkahest.git
cd alkahest
maturin develop --manifest-path alkahest-py/Cargo.toml --release
```

Optional features (combine as needed):

```bash
# LLVM JIT for native compiled evaluation
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features jit

# E-graph simplification (egglog)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features egraph

# Parallel simplification (sharded ExprPool)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features parallel

# Gröbner basis solver (+ Diophantine / homotopy-related paths)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features groebner

# CUDA / NVPTX codegen (requires CUDA toolkit and LLVM with NVPTX target)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features cuda

# Full native build (all optional features above; add cuda separately if needed)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "parallel egraph jit groebner"
```

## First steps

Every computation starts with an `ExprPool`. It owns all expressions; you create symbols and integers from it.

```python
import alkahest
from alkahest import ExprPool, diff, simplify, integrate, sin, exp, cos

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")
```

### Building expressions

Python operators build expression trees:

```python
expr = x**2 + pool.integer(2) * x + pool.integer(1)
print(expr)  # x^2 + 2*x + 1
```

Math functions accept expressions:

```python
f = sin(x**2) + exp(x * y)
```

### Parsing expressions from strings

Use `parse` when the expression comes from user input or a config file:

```python
from alkahest import parse

e = parse("x^2 + 2*x + 1", pool, {"x": x})
print(e)   # x^2 + 2*x + 1
```

Identifiers not in the `symbols` dict are auto-created as symbols in `pool`.
Both `^` and `**` denote exponentiation. See [Parsing from strings](./parsing.md)
for the full syntax reference.

### Simplification

```python
r = simplify(x + pool.integer(0))
print(r.value)  # x
print(r.steps)  # [RewriteStep(rule='add_zero', ...)]
```

### Differentiation

```python
dr = diff(sin(x**2), x)
print(dr.value)  # 2*x*cos(x^2)
```

### Integration

```python
r = integrate(exp(x), x)
print(r.value)   # exp(x)

r = integrate(sin(x), x)
print(r.value)   # -cos(x)
```

### Polynomial arithmetic

```python
from alkahest import UniPoly, RationalFunction

# Convert to FLINT-backed univariate polynomial
p = UniPoly.from_symbolic(x**3 + pool.integer(-1), x)
q = UniPoly.from_symbolic(x + pool.integer(-1), x)
print(p.gcd(q))          # x - 1
print(p // q)            # x^2 + x + 1
```

### Compiled evaluation

```python
from alkahest import compile_expr, eval_expr

# Scalar evaluation via a dict binding
result = eval_expr(x**2 + y, {x: 3.0, y: 1.0})
print(result)  # 10.0

# JIT-compiled callable
f = compile_expr(x**2 + pool.integer(1), [x])
print(f([3.0]))  # 10.0
```

### Vectorized evaluation over NumPy arrays

```python
import numpy as np
from alkahest import compile_expr, numpy_eval

f = compile_expr(sin(x) * exp(pool.integer(-1) * x), [x])
xs = np.linspace(0, 10, 1_000_000)
ys = numpy_eval(f, xs)  # vectorised; much faster than a Python loop
```

### Context manager

```python
with alkahest.context(pool=pool, simplify=True):
    z = alkahest.symbol("z")  # uses the active pool
    expr = z**2 + alkahest.sin(z)
```

## Running the examples

The `examples/` directory in the Git repository has runnable end-to-end scripts. With `alkahest` installed (`pip install alkahest` or `maturin develop` as above), from the repository root run:

```bash
python examples/calculus.py
python examples/polynomials.py
python examples/jit_eval.py
python examples/ball_arithmetic.py
python examples/ode_modeling.py
```

If you are developing without installing the extension into the active environment, set `PYTHONPATH=python` so the pure-Python package is importable alongside your build.
