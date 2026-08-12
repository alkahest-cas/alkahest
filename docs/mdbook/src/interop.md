# Interoperability

Alkahest integrates with the Python numerical ecosystem at well-defined boundaries.

## NumPy

### Batch evaluation

`numpy_eval` vectorises a compiled function over NumPy arrays with zero unnecessary copies:

```python
import numpy as np
from alkahest import ExprPool, compile_expr, numpy_eval, sin

pool = ExprPool()
x = pool.symbol("x")

f = compile_expr(sin(x) ** 2 + x, [x])
xs = np.linspace(0, 2 * np.pi, 1_000_000)
ys = numpy_eval(f, xs)   # returns a NumPy array, shape (1_000_000,)
```

Inputs are converted to `f64` arrays via DLPack or `__array__`. The call is vectorised through `CompiledFn.call_batch_buffer` in Rust: each array is read via the buffer protocol in one bulk copy (no per-element Python `float` boxing, no `.tolist()`), the native call runs with the GIL released, and the result is written directly into a preallocated output array — no Python loop, and no intermediate Python list on either side. `numpy_eval_par` (requires `--features parallel`) additionally distributes points across CPU cores via Rayon.

### Array protocol

`CompiledFn` objects implement `__array__` for direct NumPy coercion:

```python
result = np.asarray(f([1.0]))  # scalar result as a 0-d array
```

## PyTorch

PyTorch CPU tensors are accepted wherever NumPy arrays are (via `__dlpack__`):

```python
import torch
xs = torch.linspace(0, 1, 10_000)
ys = numpy_eval(f, xs)   # returns a NumPy array
```

For GPU tensors, use the `compile_cuda` path (requires `--features cuda`; see [GPU support](./gpu.md)). Note that its Python `call_batch` takes and returns **host** sequences: a CUDA tensor is copied to the host and back. The zero-copy device-pointer entry point (`call_device_ptrs`) exists in the Rust crate only and has no PyO3 binding.

## JAX

### numpy_eval with JAX arrays

JAX arrays implement `__dlpack__` and are accepted by `numpy_eval`:

```python
import jax.numpy as jnp
xs = jnp.linspace(0, 1, 10_000)
ys = numpy_eval(f, xs)
```

### JAX primitive source (to_jax)

`to_jax` registers a symbolic expression as a JAX primitive, making it callable inside JAX computations including `jax.jit`, `jax.grad`, and `jax.vmap`:

```python
from alkahest import to_jax, ExprPool, sin

pool = ExprPool()
x = pool.symbol("x")

jax_fn = to_jax(sin(x) ** 2, [x])

import jax
import jax.numpy as jnp

# Use inside jax.jit / jax.grad
jit_fn = jax.jit(jax_fn)
grad_fn = jax.grad(lambda x: jax_fn(x).sum())
```

The primitive registers:
- A concrete `def_impl` that calls the Rust evaluator
- An abstract evaluation rule for shape/dtype propagation
- A JVP (forward-mode) rule derived from the symbolic gradient
- A vmap batching rule

### StableHLO / XLA

`to_stablehlo` emits textual MLIR in the StableHLO dialect, which XLA and JAX's XLA backend can compile:

```python
from alkahest import to_stablehlo

mlir_text = to_stablehlo(expr, [x, y], fn_name="my_kernel")
# Pass to xla_client.compile() or save to .mlir file
```

## SymPy interop

Alkahest's kernel does not import SymPy. Two supported bridges exist on top of it:
`alkahest.crosscheck.to_sympy` translates an `Expr` into a SymPy expression, and
[`alkahest.crosscheck`](./crosscheck.md) drives SymPy as a differential-testing oracle.
The test oracle in `tests/test_oracle.py` uses SymPy as a ground-truth reference. For
ad-hoc mixed workflows, converting through the string representation is fine.

### The interop trap: casus-irreducibilis cube roots

Read this before round-tripping a symbolic result into another CAS. **An expression can
be correct in Alkahest and evaluate to a wrong number somewhere else**, because the two
systems do not agree on which branch a cube root denotes.

`Matrix.eigenvals()` on a 3×3 with an irreducible cubic characteristic polynomial and
three real roots returns the Cardano form, and in the *casus irreducibilis* one of the
cube-root radicands is negative:

```python
import alkahest as ak

pool = ak.ExprPool()
I = pool.integer
M = ak.Matrix.from_rows([[I(2), I(0), I(-2)], [I(2), I(0), I(-1)], [I(1), I(1), I(2)]])

for value in M.eigenvals():
    print(value)
# two conjugate-looking siblings, then:
# (4/3 + (sqrt(298/27) + -89/27)^(1/3) + (-89/27 + (-1 * sqrt(298/27)))^(1/3))
```

That expression denotes the eigenvalue **under the real cube-root convention**. Alkahest
is consistent about this and honest at the boundary: `eval_expr` on it refuses with
`E-EVAL-009`, and `interval_eval` returns `ArbBall(1.629231 ± inf)` — an enclosure that
is true and useless, rather than a number that is neither.

Hand the *same* expression to a principal-branch evaluator — SymPy, NumPy, most
calculators — and `(negative)^(1/3)` takes the principal complex root instead. You get a
confident number back, and it is not an eigenvalue. In one sweep of 720 random integer
matrices, 14 produced eigenvalues of this shape.

So, when a loop exports symbolic results to another tool:

- Prefer transporting a **verified numeric enclosure** (`refine_root`, `interval_eval`,
  `bound_on_box`) rather than a radical expression, whenever the consumer only needs a
  number.
- If you must transport the expression, evaluate it in Alkahest **first**. A refusal
  (`E-EVAL-009`, or an infinite ball) is the signal that the expression is branch-sensitive
  and must not be handed over as-is.
- Never treat "the other tool produced a float" as confirmation. Substitute the value back
  into the characteristic polynomial (or whatever defined it) and check the residual.

This is the general shape of the hazard, not a quirk of `eigenvals`: **an honest refusal
inside Alkahest becomes somebody else's silent error the moment the expression crosses the
boundary.** [`alkahest.crosscheck`](./crosscheck.md) reports exactly this situation as
`incomparable` rather than `diverge`, for the same reason.

## DLPack

All DLPack-compatible arrays (NumPy, PyTorch, JAX, CuPy) are accepted at the `numpy_eval` boundary. The DLPack conversion is zero-copy for CPU arrays with matching dtypes; a device array is copied to the host first. There is no device-pointer boundary exposed to Python — `call_device_ptrs` is a Rust-crate API.

## Exporting C code

`emit_c` generates a standalone C function for embedding in other projects:

```python
from alkahest import emit_c

c_code = emit_c(
    sin(x) * exp(pool.integer(-1) * x),
    [x],
    var_name="x",
    fn_name="damped_sin",
)
print(c_code)
# double damped_sin(double x) { return sin(x) * exp(-x); }
```

The emitted code uses only standard `<math.h>` functions and has no Alkahest dependency.
