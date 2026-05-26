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

CPU compilation uses a **three-tier dispatch** (tried in order):

1. **Cranelift** (`--features cranelift`) — pure Rust, ~10× faster compile than LLVM; no system LLVM required
2. **LLVM** (`--features jit`) — inkwell / LLVM 15 MCJIT; best generated code for hot loops
3. **Interpreter** — always available; zero compile latency, tree-walking over the DAG

Default PyPI wheels use the interpreter only. `+jit` / `+full` release wheels enable LLVM. Add `cranelift` to a from-source build for the fast-compile tier without LLVM.

## CompileCache

Repeated compilation of the same expression is expensive. `CompileCache` memoizes by `(ExprId, input variables)` — hash-consing makes `ExprId` a stable content key:

```python
from alkahest import ExprPool, compile_expr, CompileCache

pool = ExprPool()
x = pool.symbol("x")
cache = CompileCache()
f = cache.compile(x**2, [x], pool)   # JIT compiles on first call
g = cache.compile(x**2, [x], pool)   # cache hit — O(1)
print(cache.stats())                 # hits, compiles, hit_rate
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

If the extension was built without `parallel`, `numpy_eval_par` transparently falls back to `numpy_eval`.

## Horner-form emission

`horner` rewrites a polynomial expression into Horner's form, which is numerically better conditioned and faster to evaluate:

```python
from alkahest import horner

# x^3 + 2x^2 + 3x + 4 → x*(x*(x + 2) + 3) + 4
h = horner(x**3 + pool.integer(2)*x**2 + pool.integer(3)*x + pool.integer(4), x)
```

`emit_c` emits a C function string for embedding in other projects:

```python
from alkahest import emit_c

c_code = emit_c(expr, [x, y], fn_name="f")
# → "double f(double x, double y) { return ...; }"
```

## MLIR dialect

The `alkahest-mlir` crate exposes the custom MLIR dialect. The dialect ops are:

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
mlir_text = to_stablehlo(expr, [x, y], fn_name="my_fn")
print(mlir_text)  # valid input to mlir-opt / XLA
```

## GPU codegen (NVPTX)

With `--features cuda` and an LLVM installation with NVPTX support:

```python
from alkahest import compile_cuda

f_gpu = compile_cuda(expr, [x, y])
result = f_gpu.call_batch(inputs)   # runs on the first CUDA device
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
