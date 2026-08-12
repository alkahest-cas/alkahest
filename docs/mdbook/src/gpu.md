# GPU support (CUDA)

Alkahest has two *independent* CUDA features. Neither is in the wheel published to
PyPI, so **`pip install alkahest` gives you no GPU support at all** — a source build
is required, and each feature has different build prerequisites.

| Cargo feature | What it provides | Build prerequisites | Reachable from Python? |
|---|---|---|---|
| `cuda` | NVPTX codegen: [`compile_cuda`](#compile_cuda) turns an expression into a GPU kernel | **LLVM 15 with the NVPTX target** (`cuda` implies `alkahest-core/jit`, i.e. inkwell), plus `libcuda.so.1` at runtime | **Yes** — `compile_cuda`, `CudaCompiledFn` |
| `groebner-cuda` | A Macaulay-matrix mod-p row reduction kernel used by the Rust function `compute_groebner_basis_gpu` | Only `cudarc` — **no LLVM**, because the kernel is a static PTX string rather than LLVM output | **No** — see [below](#groebner-cuda-is-not-reachable-from-python) |

The two do not imply each other. `cuda = ["jit", "dep:cudarc"]` and
`groebner-cuda = ["groebner", "dep:cudarc"]` (`alkahest-core/Cargo.toml`).

## Building

```bash
# NVPTX expression codegen. Needs LLVM 15 built with NVPTX:
#   llc --version | grep nvptx     # must list nvptx64
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features cuda

# GPU Gröbner kernel (Rust-only; nothing changes at the Python surface)
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features groebner-cuda

# Both
maturin develop --manifest-path alkahest-py/Cargo.toml --release \
    --features "cuda groebner-cuda"
```

`cudarc` uses dynamic loading, so **the extension builds on a machine with no CUDA
installed**; the driver is only needed when a kernel actually launches. LLVM 15 with
NVPTX, by contrast, is needed at *build* time for `cuda` — a build without it fails
or produces `E-CUDA-001` at compile time.

`libdevice.10.bc` (from the CUDA toolkit) is linked into every generated module so
that `sin`, `cos`, … resolve to `__nv_*`. If it is not found automatically, point at
it explicitly:

```bash
export ALKAHEST_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
```

## What `capabilities()` reports

```python
import alkahest as ak

features = ak.capabilities()["features"]
features["cuda"]           # `--features cuda` was compiled in
features["groebner_cuda"]  # `--features groebner-cuda` was compiled in
features["llvm_jit"]       # True on any `cuda` build: `cuda` implies the LLVM backend
```

Read these bits precisely — each says **what was linked**, and nothing more:

- `cuda == True` guarantees `ak.compile_cuda` and `ak.CudaCompiledFn` exist and that
  PTX can be emitted on the host. It does **not** promise a GPU: the driver is loaded
  lazily, so a machine with no device compiles happily and fails at `call_batch` with
  `E-CUDA-003`. The only way to find out is to launch something.
- `llvm_jit == True` on a `cuda` build even when *alkahest-py*'s own `jit` feature was
  never named, because `alkahest-core`'s `cuda` feature turns on `jit`. Cranelift and
  LLVM are not mutually exclusive; a CUDA build can link both.
- `groebner_cuda == True` changes nothing observable from Python. See below.

`ak.CudaError` is importable on **every** build, CUDA or not — it is an exception
class, not an entry point, and code that writes `except ak.CudaError` around a
compile step must keep working when moved between wheels. `compile_cuda` and
`CudaCompiledFn` genuinely do not exist without the feature, and are appended to
`__all__` only when they do.

## `compile_cuda`

```python
import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")
expr = ak.sin(x) * ak.cos(y) + (x * x + y * y) * pool.rational(1, 100)

fn = ak.compile_cuda(expr, [x, y])   # -> CudaCompiledFn
fn.n_inputs                          # 2
fn.ptx                               # generated PTX assembly (str), `.target sm_86`

out = fn.call_batch([xs, ys])        # list[float], one output per point
```

`call_batch` takes one column per symbolic input (structure-of-arrays: `xs` is every
`x` value, not the first point), all of equal length, and returns a Python list with
one `float` per point. It copies host → device, launches on **device 0**, and copies
back; a mismatched column count or ragged columns raise `ValueError` before anything
touches the GPU.

Pipeline: expression → LLVM IR via inkwell → link `libdevice.10.bc` → internalize and
DCE → PTX for `sm_86` (Ampere) → loaded through the CUDA driver by `cudarc`.

### Limits worth knowing before you reach for it

- **`sm_86` is hard-coded.** Newer or older architectures rely on the driver's PTX JIT.
- **`f64` only**, one output value per point. There is no vector or complex return.
- **Supported nodes**: integer/rational/float constants, `+`, `*`, `**`, and the
  unary functions `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`. Integer exponents
  in `0..=16` are unrolled to multiplies; anything else goes through `__nv_pow`.
  Any other function — `atan`, `sinh`, `erf`, … — is **refused** with `E-CUDA-002`
  rather than approximated, as is any symbol you forgot to pass in `inputs`.
- **Host lists in, host list out.** The zero-copy device-pointer entry point
  (`call_device_ptrs`) exists in the Rust crate only; it has no PyO3 binding, so a
  CuPy or Torch CUDA tensor is round-tripped through host memory today.

### Errors

All are `CudaError`, a subclass of `AlkahestError`, each carrying `.code` and
`.remediation` (see [Error handling](./errors.md)).

| Code | Meaning |
|---|---|
| `E-CUDA-001` | LLVM has no NVPTX target — rebuild LLVM with `nvptx64` in `LLVM_TARGETS_TO_BUILD` |
| `E-CUDA-002` | PTX generation failed: unbound symbol, unsupported node, or a verifier complaint |
| `E-CUDA-003` | CUDA driver error — no device, context creation, module load, or a memcpy |
| `E-CUDA-004` | Not implemented |
| `E-CUDA-005` | `libdevice` bitcode not found — install the CUDA toolkit or set `ALKAHEST_LIBDEVICE_PATH` |
| `E-CUDA-006` | Kernel launch failed |

## `groebner-cuda` is not reachable from Python

The feature compiles a real, tested CUDA kernel — `MacaulayMatrix::reduce_gpu` plus a
multi-prime CRT lift — and exports `compute_groebner_basis_gpu` from the Rust crate.
But **no shipped code path calls it.** `GroebnerBasis.compute`, `solve`, and
`triangularize` all go through `compute_buchberger_basis` on the CPU, and
`alkahest-py` never references the GPU entry point at all. So on a
`--features groebner-cuda` build:

- `capabilities()["features"]["groebner_cuda"]` is `True`,
- no Python name appears or disappears,
- no Python call gets faster.

This is deliberate rather than accidental — the crossover policy in
[`docs/symbolic-gpu-benchmarks.md`](https://github.com/alkahest-cas/alkahest/blob/main/docs/symbolic-gpu-benchmarks.md)
says production dispatch must not prefer the GPU until the benchmark harness says it
wins, and that wiring does not exist yet. It is recorded here because the bit is
otherwise unfalsifiable from Python: there is no observation a caller can make that
distinguishes `groebner_cuda: True` from `False`. Rust users can call
`alkahest_cas::poly::groebner::compute_groebner_basis_gpu` directly; it falls back to
CPU row reduction when no device is present.

## State of testing — read this before trusting the feature

**Rust, on hardware.** `alkahest-core/tests/nvptx_gpu.rs` and
`alkahest-core/tests/groebner_cuda.rs` run under
`.github/workflows/cuda_nightly.yml` on a self-hosted dual-RTX-3090 runner: 17
CUDA-gated tests, plus `compute-sanitizer` `memcheck` and `racecheck`. The last full
run was green and both sanitizers clean. `--target-processes all` is load-bearing in
that workflow — without it the sanitizer instruments `cargo`, a process that makes no
CUDA calls, and reports success having inspected nothing.

**Python.** `tests/test_cuda.py` covers the binding: the capability/namespace
contract, PTX emission, the `CudaError` refusals, `call_batch` argument validation,
and — the point of the exercise — GPU-versus-CPU numerical agreement on polynomial,
transcendental, `compile_expr` and `numpy_eval` comparisons. Only the contract tier
runs without the feature; everything else skips, which is what happens in CI and on
the wheel. Setting `ALKAHEST_GPU_TESTS=1` (as the nightly does for Rust) turns those
skips into a hard error, so a job that promises hardware cannot quietly report
success without reaching it.

**The honest gap:** no CI job has ever built the *Python extension* with `cuda` or
`groebner-cuda`. The nightly runs `cargo`, never `maturin`, so the Python tier is
only exercised when someone builds with the feature and runs `pytest` by hand on a
GPU box. Until a `maturin develop --features cuda` + `pytest tests/test_cuda.py` step
is added to the nightly, treat the Python GPU surface as verified by hand and not by
CI — which is precisely how the `compile_cuda` export gap survived three releases
while `capabilities()` advertised the feature.

## See also

- [Code generation](./codegen.md) — the CPU JIT tiers, `emit_c`, StableHLO
- [Error handling](./errors.md) — the full code registry
- `examples/gpu_batch_eval.py` — CPU/GPU batch comparison, degrades cleanly to
  CPU-only on a wheel without the feature
