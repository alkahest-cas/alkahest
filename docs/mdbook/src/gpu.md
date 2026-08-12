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
features["llvm_jit"]       # True on any `cuda` build: `cuda` implies the LLVM backend
```

Read these bits precisely — each says **what was linked**, and nothing more:

- `cuda == True` guarantees `ak.compile_cuda` and `ak.CudaCompiledFn` exist and that
  PTX can be emitted on the host. It does **not** promise a GPU: the driver is loaded
  lazily, so a machine with no device compiles happily and fails at `call_batch` with
  `E-CUDA-003`. Ask `ak.cuda_device_count()` — it reports `0` when there is no usable
  device, and never raises.
- `llvm_jit == True` on a `cuda` build even when *alkahest-py*'s own `jit` feature was
  never named, because `alkahest-core`'s `cuda` feature turns on `jit`. Cranelift and
  LLVM are not mutually exclusive; a CUDA build can link both.

**There is no `groebner_cuda` bit** (contract v3 and later — `capabilities()["features"]`
raises `KeyError` for it). It was removed rather than wired up because it was
*unfalsifiable*: no Python observation distinguished `True` from `False`. See
[below](#groebner-cuda-is-not-reachable-from-python).

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
touches the GPU. `fn.call_batch_on(ordinal, inputs)` is the same call on a chosen
device.

Pipeline: expression → LLVM IR via inkwell → link `libdevice.10.bc` → internalize and
DCE → PTX for `sm_86` (Ampere) → loaded through the CUDA driver by `cudarc`.

### Discovering the valid device ordinals

```python
n = ak.cuda_device_count()          # 0 when there is no GPU here
for dev in range(n):
    fn.call_batch_on(dev, [[0.0]] * fn.n_inputs)
```

The valid arguments to `call_batch_on` are exactly `range(ak.cuda_device_count())`.
Both directions are pinned by tests that run on hardware
(`tests/test_cuda.py::test_device_count_agrees_with_the_ordinals_that_launch` and
`nvptx_gpu::cuda_device_count_matches_the_ordinals_that_launch`): every ordinal below
the count launches, and the ordinal *at* the count is refused.

`cuda_device_count()` **never raises**. Every "no GPU here" shape — no driver, no
device, driver too old — reports `0`, because that is the single answer a caller acts
on. This matters more than it looks: `cudarc` *panics* rather than returning `Err`
when `libcuda.so` cannot be `dlopen`'d at all, so a naive binding would abort the
process on precisely the machines a capability probe exists to report on. The same
bug bit `groebner_cuda.rs::gpu_available` and `nvptx_gpu.rs::device_available`.

Earlier releases documented a workaround here — launch on an ordinal and catch
`E-CUDA-003` — because a `cuda_device_count` binding could not be verified by
anything on an ordinary dev box: `cuda` implies LLVM 15 with NVPTX, so it could not
even be *compiled*, and no CI job built the Python extension with the feature. It
shipped once both could be done on real hardware, which is the standard the
capability overclaims this page documents were failing.

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

- no Python name appears or disappears,
- no Python call gets faster,
- and, since 3.8, **no capability bit claims otherwise.**

This is deliberate rather than accidental — the crossover policy in
[`docs/symbolic-gpu-benchmarks.md`](https://github.com/alkahest-cas/alkahest/blob/main/docs/symbolic-gpu-benchmarks.md)
says production dispatch must not prefer the GPU until the benchmark harness says it
wins, and that wiring does not exist yet. Rust users can call
`alkahest_cas::poly::groebner::compute_groebner_basis_gpu` directly.

### Why the bit was removed rather than wired up

`capabilities()["features"]["groebner_cuda"]` used to report `True` here. It was the
only occurrence of the string `groebner_cuda` anywhere in `alkahest-py` — there was no
binding to go with it, no `*gpu*` name in the public or the private module, and
`GroebnerBasis` exposing only CPU methods. That made it strictly worse than the `cuda`
overclaim fixed in `d139a46`, which at least had a private route in.

An unreachable `True` is the same class of defect as a silent wrong answer: it makes a
caller trust something it should not. The two ways out are to add a binding or to drop
the claim, and dropping it was the right one days before a release — a binding would
have been new public API that no CI job can build (no job builds the Python extension
with either CUDA feature; see below) and that nobody without a GPU can run. Adding
unverifiable surface is how the original defect got in. The bit is gone; the kernel is
unchanged and still Rust-reachable. If dispatch ever prefers the GPU, the binding lands
first and a bit follows it.

### `compute_groebner_basis_gpu` now reports where it ran

The Rust entry point falls back to CPU row reduction when no device is present, and it
used to say nothing about having done so — a `device_id: None` run, a run whose driver
calls all failed, and a real GPU run returned identical, indistinguishable values. A
function named `..._gpu` that quietly runs on the CPU is a footgun of exactly the kind
this release has spent its time eliminating, so both it and `reduce_batch` now return a
`GpuBackendReport` alongside the polynomials:

```rust
use alkahest_cas::poly::groebner::{compute_groebner_basis_gpu, MonomialOrder};

let (basis, backend) = compute_groebner_basis_gpu(gens, MonomialOrder::Lex, Some(0))?;
assert!(backend.ran_on_gpu(), "fell back to the CPU: {backend:?}");
```

`ran_on_gpu()` is true only when at least one mod-p row reduction executed on a device
and none fell back; `fell_back_to_cpu()` is its counterpart; `reductions_on_gpu`,
`reductions_on_cpu` and `first_gpu_error` carry the detail. The stderr warning on
fallback is still emitted, but it is no longer the only channel. This is a **breaking
change to the Rust signature** — a compile error on upgrade, which is the correct
failure mode for a caller who was reading a result as a GPU result.

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

**A second gate that was inspecting nothing.** `cargo test --features groebner-cuda`
could not pass on a machine with no NVIDIA driver at all, contradicting the header
comment of `alkahest-core/tests/groebner_cuda.rs`. `cudarc` *panics* rather than
returning `Err` when `libcuda.so` cannot be `dlopen`ed, so `gpu_available()` — whose
whole job is to answer "should the GPU tier run?" — aborted the three GPU tests
instead of skipping them. It now treats a missing library and a missing device alike
(both mean *not available*), while still failing hard when `ALKAHEST_GPU_TESTS=1`
asserted a device that is not usable. The `ALKAHEST_GPU_TESTS=1` tier additionally
asserts `GpuBackendReport::ran_on_gpu()`, so a "GPU test" that silently reduced every
matrix on the CPU now fails rather than passing on identical results.

## See also

- [Code generation](./codegen.md) — the CPU JIT tiers, `emit_c`, StableHLO
- [Error handling](./errors.md) — the full code registry
- `examples/gpu_batch_eval.py` — CPU/GPU batch comparison, degrades cleanly to
  CPU-only on a wheel without the feature
