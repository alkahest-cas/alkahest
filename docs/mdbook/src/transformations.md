# Transformations

Transformations are higher-order operations that take a function and return a new function. They compose freely and operate on traced symbolic representations.

## Tracing

`trace` symbolically executes a Python function by replacing its inputs with symbolic variables and recording the computation as an expression DAG.

```python
import alkahest
from alkahest import ExprPool

pool = ExprPool()

@alkahest.trace(pool)
def f(x, y):
    return x**2 + alkahest.sin(y)

print(f.expr)    # x^2 + sin(y)
print(f.symbols) # [x, y]
```

The decorator takes the pool as an argument. Variable names are inferred from the function signature.

### Numeric evaluation

`TracedFn` objects are callable with numeric values:

```python
print(f(3.0, 0.0))   # 9.0

import numpy as np
xs = np.linspace(0, 1, 1000)
ys = np.zeros(1000)
result = f(xs, ys)   # vectorised automatically
```

## Gradient (`grad`, not `symbolic_grad`)

`alkahest.grad` applies to a **TracedFn** from `@trace`. For partial derivatives of a
bare `Expr`, use [`symbolic_grad`](./calculus.md#symbolic-gradient-symbolic_grad) instead.

`grad` differentiates a traced function symbolically with respect to all (or a subset of) its inputs:

```python
df = alkahest.grad(f)
# df(x, y) returns [∂f/∂x, ∂f/∂y] = [2*x, cos(y)]

grads = df(1.0, 0.0)   # [2.0, 1.0]
```

Differentiate with respect to a subset:

```python
df_x = alkahest.grad(f, wrt=[f.symbols[0]])  # ∂f/∂x only
```

## JIT compilation

`jit` wraps a traced function in the LLVM JIT backend. The first call triggers compilation; subsequent calls run the compiled code directly.

```python
fast_f = alkahest.jit(f)
print(fast_f(3.0, 0.0))  # 9.0, via LLVM-compiled code
```

Vectorised evaluation is automatic when array inputs are detected:

```python
xs = np.linspace(0, 10, 1_000_000)
ys = np.zeros_like(xs)
result = fast_f(xs, ys)   # zero-copy batch path
```

## Composing transformations

Transformations stack:

```python
# Compiled gradient
fast_df = alkahest.jit(alkahest.grad(f))
grads = fast_df(xs, ys)   # compiled, vectorised gradient

# Second derivative: grad of grad
d2f = alkahest.grad(alkahest.grad(f))
```

Note that `grad` returns a `GradTracedFn`, not a `TracedFn`. `jit` can be applied to `GradTracedFn` when it wraps a single scalar output. For multi-output cases, compile each gradient expression individually with `compile_expr`.

## trace_fn

Functional (non-decorator) version of `trace`:

```python
from alkahest import trace_fn

fn = trace_fn(lambda x, y: x * alkahest.exp(y), pool)
```

## PyTrees

Transformations work over nested data structures (lists, dicts, tuples, dataclasses). The Python layer flattens and unflattens them automatically:

```python
from alkahest import flatten_exprs, unflatten_exprs, map_exprs

# A system of equations as a list
eqs = [x**2 + y - pool.integer(1), x - y**2 + pool.integer(1)]

# Map simplification over the list
simplified = map_exprs(simplify, eqs)

# Flatten to a list of ExprIds and the structure descriptor
flat, treedef = flatten_exprs(eqs)
restored = unflatten_exprs(flat, treedef)
```

This follows the JAX pytree pattern. The Rust kernel sees only flat sequences; structure reconstruction is a Python-layer concern.

## Context manager

`alkahest.context` sets a default pool and configuration for a block:

```python
with alkahest.context(pool=pool, simplify=True):
    z = alkahest.symbol("z")          # uses the active pool
    expr = z**2 + alkahest.sin(z)     # auto-simplified
```

Inside the context, `alkahest.symbol(name)` creates a symbol in the active pool without passing the pool explicitly. This is a convenience wrapper — the pool is still explicit at the structural level.
