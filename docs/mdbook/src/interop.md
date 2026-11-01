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

Two argument conventions meet here, and mixing them up is the most common way
this call fails:

```python
y = pool.symbol("y")
ys = np.linspace(0, 1, 1_000_000)
f = compile_expr(x ** 2 + y, [x, y])   # a CompiledFn — numpy_eval takes this, not the Expr

numpy_eval(f, xs, ys)                  # ✓ one positional argument per input variable
f([1.0, 2.0])                          # ✓ CompiledFn evaluates ONE point, given as one sequence
try:
    numpy_eval(f, [xs, ys])            # ✗ ValueError — that is one argument, not two
except ValueError as e:
    print(type(e).__name__)
try:
    f(1.0, 2.0)                        # ✗ TypeError
except TypeError as e:
    print(type(e).__name__)
```

Both wrong spellings raise an error naming the right one. `numpy_eval(expr, xs)`
— the expression instead of the compiled function — is a `TypeError` telling you
to call `compile_expr` first.

Inputs are converted to `f64` arrays via DLPack or `__array__`. The call is vectorised through `CompiledFn.call_batch_buffer` in Rust: each array is read via the buffer protocol in one bulk copy (no per-element Python `float` boxing, no `.tolist()`), the native call runs with the GIL released, and the result is written directly into a preallocated output array — no Python loop, and no intermediate Python list on either side. `numpy_eval_par` additionally distributes points across CPU cores via Rayon — but **only when the extension was built with `--features parallel`**. Every published wheel is; `parallel` is not a Cargo default, so a source build that omits it gets a silent alias for `numpy_eval`: same result, no speedup, no warning. Check `capabilities()["features"]["parallel"]` before attributing a timing to it; see [Getting started](./getting-started.md#install).

### Array protocol

`CompiledFn` objects implement `__array__` for direct NumPy coercion:

```python
result = np.asarray(f([1.0, 2.0]))  # scalar result as a 0-d array
```

## PyTorch

PyTorch CPU tensors are accepted wherever NumPy arrays are (via `__dlpack__`):

<!-- doc-check: needs torch -->
```python
import torch
xs = torch.linspace(0, 1, 10_000)
ys = numpy_eval(f, xs)   # returns a NumPy array
```

For GPU tensors, use the `compile_cuda` path (requires `--features cuda`; see [GPU support](./gpu.md)). Note that its Python `call_batch` takes and returns **host** sequences: a CUDA tensor is copied to the host and back. The zero-copy device-pointer entry point (`call_device_ptrs`) exists in the Rust crate only and has no PyO3 binding.

## JAX

### numpy_eval with JAX arrays

JAX arrays implement `__dlpack__` and are accepted by `numpy_eval`:

<!-- doc-check: needs jax -->
```python
import jax.numpy as jnp
xs = jnp.linspace(0, 1, 10_000)
ys = numpy_eval(f, xs)
```

### JAX primitive source (to_jax)

`to_jax` registers a symbolic expression as a JAX primitive, making it callable inside JAX computations including `jax.jit`, `jax.grad`, and `jax.vmap`:

<!-- doc-check: needs jax -->
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

expr = x ** 2 + y
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

`Matrix.eigenvals()` itself no longer produces such radicals: in the *casus
irreducibilis* (an irreducible cubic characteristic polynomial with three real roots) it
returns the trigonometric form, and for one real root the Cardano radicands are kept
positive, so every evaluator reads its results the same way:

```python
import alkahest as ak

pool = ak.ExprPool()
I = pool.integer
M = ak.Matrix.from_rows([[I(0), I(0), I(-1)], [I(1), I(0), I(4)], [I(0), I(1), I(0)]])

for value in M.eigenvals():       # roots of λ³ − 4λ + 1, all real
    print(value)
# (2 * sqrt(4/3) * cos((1/3 * (acos((-1/2 * (4/3 * sqrt(4/3))^-1)) + (2 * pi)))))
# ...
```

The hazard is still there for any `(negative)^(1/3)` you build, parse or import — the
shape older Cardano output had:

```python
r = ak.parse("(-89/27 - sqrt(298/27))^(1/3)", pool)
try:
    ak.eval_expr(r, {})
except ak.AlkahestError as e:
    print(e.code)                 # E-EVAL-009
print(ak.interval_eval(r, {}))    # ArbBall(0 ± inf)
```

Alkahest is honest at the boundary: `eval_expr` refuses with `E-EVAL-009`, and
`interval_eval` returns an enclosure that is true and useless, rather than a number
that may be the wrong one. Hand the *same* expression to a principal-branch evaluator
— SymPy, NumPy, most calculators — and `(negative)^(1/3)` takes the principal complex
root, `0.9388 + 1.6260i`, where the real-cube-root reading is `−1.8775`. You get a
confident number back, and it is not the one the author meant.

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

`emit_c_expr` generates a standalone C function for embedding in other projects
(`emit_c` is the univariate-polynomial, Horner-form variant):

```python
from alkahest import emit_c_expr, exp

c_code = emit_c_expr(sin(x) * exp(-x), [x], fn_name="damped_sin")
print(c_code)
# double damped_sin(double x) {
#     return (sin(x) * exp((-x)));
# }
```

The emitted code uses only standard `<math.h>` functions and has no Alkahest dependency.
