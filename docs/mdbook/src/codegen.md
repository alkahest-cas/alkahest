# Code generation

Alkahest can compile symbolic expressions to fast native or GPU code. Compiled code bypasses Python entirely during evaluation.

## The compilation pipeline

Expressions lower through multiple IR levels:

```
ExprPool (hash-consed DAG)
    ↓  e-graph extraction + canonicalization
Canonical expression form
    ↓  alkahest MLIR dialect
High-level MLIR (math-aware ops: horner, poly_eval, interval_eval)
    ↓  lowering passes
Standard MLIR (arith, math, linalg, gpu)
    ↓
LLVM IR / PTX / StableHLO (depending on target)
    ↓
Native machine code / GPU kernel / XLA
```

The custom `alkahest` MLIR dialect is where math-aware optimizations happen: Horner's method for polynomials, fused multiply-add emission, numerically stable rearrangements via `StabilityCost`.

## compile_expr

`compile_expr` produces a callable from a symbolic expression and a list of input variables:

```python
from alkahest import ExprPool, compile_expr, sin, cos

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

f = compile_expr(x**2 + sin(y), [x, y])
print(f([3.0, 0.0]))   # 9.0
```

The callable takes a list of floats (one per variable) and returns a float. For batch evaluation see `numpy_eval` below.

CPU compilation uses a **three-tier dispatch** chosen by `CompileConfig` (expression size and `expected_evals`, not feature-flag order):

1. **Interpreter** — small DAGs and few planned evaluations (zero compile latency)
2. **Cranelift** (`--features cranelift`) — pure Rust, ~10× faster compile than LLVM; no system LLVM required
3. **LLVM** (`--features jit`) — inkwell / LLVM 21 MCJIT; best throughput for large batch sweeps

Default PyPI wheels use the interpreter only. `+jit` / `+full` release wheels enable LLVM. Add `cranelift` to a from-source build for the fast-compile tier without LLVM.

Both native JIT backends also emit a **column** entry point (`fn(cols, outs, start, end)`) that loads the input column pointers once and evaluates a range of points in one native loop; `CompiledFn.call_batch` / `call_bulk` and `numpy_eval` call it once per batch, and `numpy_eval_par` once per Rayon chunk (a sub-range, nothing copied).

The **interpreter** tier (no JIT in the build, or an expression a JIT declines — `Piecewise` in particular) evaluates a batch a **block** of up to 256 points at a time: the expression is flattened once into a list of operations over numbered slots, and each operation runs over the whole block before the next one, so there is one dispatch per operation per block and sums, products and integer powers vectorise. `Piecewise` is a masked select: every branch is computed on the block, and per-point failure flags ensure a branch a point does not select (say `lambert_w(x)` guarded by `x > 0`) cannot fail that point; `And`/`Or` short-circuit per point. Each value is bit for bit what a single-point call (and `eval_expr`) gives.

### Real values through complex intermediates

A real-valued closed form can pass through complex numbers on the way:
`cos(t·√ω²)` at a point where `ω² < 0` is `cosh(t·√−ω²)`; Cardano's formula for
a cubic with three real roots, and a ZOH matrix built from such eigenvalues,
are real while their radicals are not. Real `double` arithmetic takes
`√(negative)` to NaN. `compile_expr`, `numpy_eval`/`numpy_eval_par`,
`eval_expr` and the C from `emit_c_expr`/`emit_c_vec` therefore re-evaluate a
point whose real result is NaN in complex arithmetic (principal branches, as
`evaluate(..., mode="complex")`), carrying a running bound on the rounding
error, and return the **real part when the imaginary part is zero within that
bound**:

```python
from alkahest import sqrt

t, w2 = pool.symbol("t"), pool.symbol("w2")
f = compile_expr(cos(t * sqrt(w2)), [t, w2])
f([0.7, -4.0])           # 2.1508984653931407 == cosh(1.4), not nan
```

NaN (and `E-EVAL-009` from `eval_expr`) now means the value **is not a real
number** at that point — `log(−1)`, `√−2`, `asin(2)`, `(−8)^(1/3)` on the
principal branch — or cannot be decided in double precision (an argument whose
error disk straddles a branch cut). It never means "real, but the real
arithmetic could not reach it". Only points whose real result is NaN pay for
the complex pass; expressions with no square root, logarithm,
`acos`/`asin`, non-integer power or imaginary unit, or with a node the complex
evaluator has no kernel for (`Piecewise`, special functions), are compiled
exactly as before. An expression written with the imaginary unit `I` is
emitted to C as its complex evaluation alone.

## CompileCache

Repeated compilation of the same expression is expensive. `CompileCache` memoizes by `(ExprId, input variables)` — hash-consing makes `ExprId` a stable content key:

```python
from alkahest import CompileCache

cache = CompileCache()
f = cache.compile(x**2, [x])         # JIT compiles on first call — (expr, inputs)
g = cache.compile(x**2, [x])         # cache hit — O(1)
print(cache.stats())                 # {'compiles': 1.0, 'hits': 1.0, 'len': 1.0, 'hit_rate': 0.5}
```

## eval_expr

For one-off evaluation without compiling:

```python
from alkahest import eval_expr

result = eval_expr(x**2 + sin(y), {x: 3.0, y: 0.0})
print(result)   # 9.0
```

`eval_expr` is slower than a compiled function for repeated evaluation but has no compilation overhead.

## numpy_eval

`numpy_eval` vectorises a compiled function over NumPy arrays via the batch path:

```python
import numpy as np
from alkahest import numpy_eval

f = compile_expr(sin(x) * cos(x), [x])
xs = np.linspace(0, 2 * 3.14159, 1_000_000)
ys = numpy_eval(f, xs)   # vectorised, zero-copy
```

Also accepts PyTorch CPU tensors and JAX arrays via DLPack.

### Parallel batch evaluation

With `--features parallel`, `numpy_eval_par` distributes evaluation across CPU cores (Rayon), releasing the GIL during computation:

```python
from alkahest import numpy_eval_par

ys = numpy_eval_par(f, xs)   # same API as numpy_eval; multi-core
```

> **`numpy_eval_par` is only parallel when the extension was built with
> `parallel`.** Every published wheel is — but `parallel` is not a Cargo
> *default*, so a source build that omits it gets a silent no-op alias:
> `numpy_eval_par` falls back to `numpy_eval` with the same correct result, the
> same single-threaded throughput, and no warning. Timing the two against each
> other on such a build measures one code path twice.
>
> ```python
> import alkahest as ak
> ak.capabilities()["features"]["parallel"]   # True on PyPI wheels
> ```
>
> If that prints `False`, rebuild with `--features parallel` — see
> [Getting started](./getting-started.md#install).

### A `CompiledFn` belongs to the thread that compiled it

`numpy_eval_par` fans out internally, so you do not need threads of your own to
use it. If you add them anyway, **compile inside each thread**: the object
returned by `compile_expr` owns JIT code pages and a backend handle, and is
pinned to its creating thread. Touching it from another one is refused:

```python
import threading

f = compile_expr(sin(x) * cos(x), [x])   # thread A

def worker():
    try:
        numpy_eval(f, xs)
    except BaseException as e:            # PanicException is not an Exception
        print(type(e).__name__)           # PanicException: ... alkahest::PyCompiledFn
                                          # is unsendable, but sent to another thread

th = threading.Thread(target=worker)
th.start(); th.join()
```

Two things to know about that refusal. It has nothing to do with `parallel` —
plain `numpy_eval` is refused identically. And `PanicException` derives from
`BaseException`, *not* `Exception`, so a worker wrapped in a bare
`except Exception:` will not catch it and the thread will die silently.

`ExprPool` and `Expr` have no such restriction: share the pool, build
expressions anywhere, and call `compile_expr` per thread.

## Horner-form emission

`horner` rewrites a polynomial expression into Horner's form, which is numerically better conditioned and faster to evaluate:

```python
from alkahest import horner

# x^3 + 2x^2 + 3x + 4 → 4 + x*(3 + x*(2 + x))
h = horner(x**3 + pool.integer(2)*x**2 + pool.integer(3)*x + pool.integer(4), x)
```

`emit_c` emits the Horner form of a univariate polynomial as a C function string for
embedding in other projects:

```python
from alkahest import emit_c

c_code = emit_c(x**3 + 2*x**2 + 3*x + 4, x, fn_name="f")
print(c_code)
# double f(double x) {
#     return 4.0 + x * (3.0 + x * (2.0 + x * (1.0)));
# }
```

`emit_c_expr` (one output) and `emit_c_vec` (many outputs into `double *out`)
emit ISO C99 for arbitrary expressions. Every sub-expression used more than
once — across all outputs jointly — is computed once into a `const double`
temporary, and small integer and half-integer powers become products,
divisions and one shared `sqrt`:

```python
from alkahest import emit_c_vec, sin, cos

print(emit_c_vec([sin(x) * cos(y), sin(x) + cos(y)], [x, y]))
# void eval_vec(double x, double y, double *out) {
#     const double _t0 = sin(x);
#     const double _t1 = cos(y);
#     out[0] = (_t0 * _t1);
#     out[1] = (_t0 + _t1);
# }
```

`pi` and `exp(1)` are emitted as the nearest `double`; `sign`, `heaviside`,
`min`/`max` and `Piecewise` as ternaries / `fmin`/`fmax`; `lambert_w` through
an emitted, `#ifndef`-guarded `static` helper (relative error ≤ 1e-15 away from
the branch point). Special functions — integer-order `bessel_j`/`bessel_y`/
`bessel_i`/`bessel_k`, `digamma`, `trigamma`, `polygamma` of order 0/1,
`dilog`, complete `EllipticK`/`EllipticE`, `fresnels`/`fresnelc`, `Ei`, `li`,
`Si`, `Ci`, `Shi`, `Chi` — lower to `static` ISO C99 helpers emitted only when
called (no POSIX `jn`/`yn`, no complex type), each checked against mpmath to
about 1e-15 relative error on the domain stated in the comment above it.
Operand pairs shared between sums (or products) — `a+b` in both `a+b+c` and
`a+b+d` — are computed once. A function with no C lowering (Airy, `zeta`,
`polylog`, `hyper`, a Bessel function of non-integer order) is refused with
`E-CODEGEN-001`, a node with no straight-line form (`O(x)`) with
`E-CODEGEN-002`, and a free symbol not in `vars` with `E-CODEGEN-003`.

## MLIR dialect

The `alkahest-mlir` crate exposes the custom MLIR dialect. `emit_mlir` / `emit_mlir_with` return `Result<String, LoweringError>`: the `arith + math` and `llvm` stages lower `Piecewise` to `cmpf`/`fcmp` + `select` and refuse an unbound symbol (`E-MLIR-001`), an unsupported function (`E-MLIR-002`), a node with no numeric value (`E-MLIR-003`) or a non-finite constant (`E-MLIR-004`); the StableHLO stage forwards `E-STABLEHLO-NNN`. The dialect ops are:

| Op | Description |
|---|---|
| `alkahest.sym` | Symbolic variable reference |
| `alkahest.const` | Constant value |
| `alkahest.add`, `alkahest.mul` | Arithmetic |
| `alkahest.pow` | Exponentiation |
| `alkahest.horner` | Horner polynomial evaluation |
| `alkahest.poly_eval` | Generic polynomial evaluation |
| `alkahest.series_taylor` | Taylor series evaluation |
| `alkahest.interval_eval` | Ball arithmetic evaluation |
| `alkahest.rational_fn` | Rational function evaluation |

Three lowering targets are available:

- **ArithMath** — lowers to `arith` + `math` MLIR dialects; uses `math.fma` for Horner chains
- **StableHlo** — lowers to StableHLO ops for XLA/JAX integration
- **Llvm** — lowers to `llvm` dialect for LLVM IR / PTX emission

```python
from alkahest import to_stablehlo

# Emit textual MLIR in the StableHLO dialect
expr = x**2 + sin(y)
mlir_text = to_stablehlo(expr, [x, y], fn_name="my_fn")
print(mlir_text)  # valid input to mlir-opt / XLA
```

## GPU codegen (NVPTX)

With `--features cuda` and an LLVM 21 installation with NVPTX support — **not** in any
published wheel, so `pip install alkahest` never has this. Full detail, including
build prerequisites, the supported node set, error codes and the state of testing, is
in [GPU support (CUDA)](./gpu.md).

<!-- doc-check: skip — needs CUDA -->
```python
from alkahest import compile_cuda

f_gpu = compile_cuda(expr, [x, y])
result = f_gpu.call_batch(inputs)   # runs on CUDA device 0
```

Guard on the capability bit before reaching for it, since the name does not exist at
all without the feature:

```python
import alkahest

if alkahest.capabilities()["features"]["cuda"]:
    ...
```

The GPU compiler:
1. Lowers the expression through inkwell to NVPTX LLVM IR for `sm_86` (Ampere)
2. Links `libdevice.10.bc` for transcendental functions (`__nv_sin`, etc.)
3. Emits PTX via LLVM's target machine
4. Loads the PTX via the CUDA driver (`cudarc`)

The benchmark `nvptx/nvptx_polynomial_1M` shows **16.2× speedup** over the CPU JIT on a 1M-point polynomial evaluation on an RTX 3090.

**Upcoming (v1.1):** AMD ROCm / `amdgcn` target (hardware-blocked pending RDNA3 availability).

## Caching

Use `CompileCache` for explicit per-session memoization of compiled functions (see above). The persistent `ExprPool` (V1-14) can serialize expression DAGs across sessions; combine with `CompileCache` to avoid recompilation after reload.

Tier dispatch always falls back to the interpreter when native JIT features are unavailable or compilation fails.
