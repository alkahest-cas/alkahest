# Getting started

## Install

Alkahest is built with [maturin](https://github.com/PyO3/maturin).

```bash
pip install maturin
git clone https://github.com/alkahest/alkahest
cd alkahest
maturin develop --release
```

To enable optional features:

```bash
# LLVM JIT for native compiled evaluation
maturin develop --release --features jit

# E-graph simplification (egglog)
maturin develop --release --features egraph

# Parallel simplification (sharded ExprPool)
maturin develop --release --features parallel

# Gröbner basis solver
maturin develop --release --features groebner

# CUDA / NVPTX codegen (requires CUDA toolkit and LLVM with NVPTX target)
maturin develop --release --features cuda

# Full build (all optional features except cuda/rocm)
maturin develop --release --features "jit egraph parallel groebner"
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

The `examples/` directory has runnable end-to-end scripts:

```bash
PYTHONPATH=python python examples/calculus.py
PYTHONPATH=python python examples/polynomials.py
PYTHONPATH=python python examples/jit_eval.py
PYTHONPATH=python python examples/ball_arithmetic.py
PYTHONPATH=python python examples/ode_modeling.py
```
