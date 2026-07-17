# Symbolic GPU benchmark harness

Candidate symbolic kernels (finite-field linear algebra, NTT/FFT polynomial multiply,
multiprecision arithmetic) are **not** enabled by default in Alkahest. GPU paths exist
behind optional Cargo features (`cuda`, `groebner-cuda`) and are evaluated with a
dedicated harness before any production dispatch considers them.

## Harness location

`alkahest-core/benches/symbolic_gpu_bench.rs` — a small standalone binary (not
CodSpeed/criterion) that prints structured metrics and writes JSON Lines to
`target/symbolic_gpu_bench.jsonl`.

Each record contains:

| Field | Meaning |
|---|---|
| `kernel_family` | e.g. `unipoly_mul`, `macaulay_reduce_mod_p` |
| `backend` | `cpu` or `gpu` |
| `input_family` | Problem generator label |
| `size` | Primary scale parameter (polynomial degree or system row count) |
| `coeff_bits` | Max bit width of input coefficients |
| `wall_ns` | Mean wall time per iteration (nanoseconds) |
| `alloc_bytes` | Heap bytes allocated during the timed iteration (cheap counter) |
| `correct` | Correctness check passed |

## Run

```bash
# CPU baselines only — suitable for CI (default `groebner` feature)
cargo bench -p alkahest-cas --bench symbolic_gpu_bench -- --nocapture

# Macaulay-matrix mod-p row reduction (CPU always; GPU optional)
cargo bench -p alkahest-cas --bench symbolic_gpu_bench \
  --features groebner-cuda -- --nocapture

# Time the CUDA elimination kernel when a device is present
ALKAHEST_GPU_BENCH=1 cargo bench -p alkahest-cas --bench symbolic_gpu_bench \
  --features groebner-cuda -- --nocapture
```

Override the report path:

```bash
SYMBOLIC_GPU_BENCH_OUT=/tmp/symbolic_gpu.jsonl cargo bench -p alkahest-cas \
  --bench symbolic_gpu_bench -- --nocapture
```

Existing NVPTX expression-JIT timing lives in `alkahest-core/benches/alkahest_bench.rs`
(`bench_nvptx`, `--features cuda`). That bench is separate from this harness.

## Crossover policy

**Policy: use GPU only when the harness says it wins — never as the default path.**

The harness includes scaffolding (`CrossoverPolicy` in the bench source) that
compares paired CPU/GPU records per `(kernel_family, size)`:

- `min_gpu_speedup` (default **1.10**) — GPU must beat CPU by at least 10%.
- `min_size` (default **1**) — ignore crossover below this size.

The harness prints recommendations to stderr; production code does **not** read these
results yet. Future dispatch should:

1. Load baseline JSON from a pinned CI artifact or local run.
2. Apply the same policy constants (or stricter ones).
3. Fall back to CPU when GPU is unavailable, fails correctness, or loses on wall time.

Until that wiring exists, all user-facing APIs keep CPU-first behavior (`compute_groebner_basis_gpu`
already falls back when no device is present).

## Kernel coverage

| Kernel | CPU baseline | Optional GPU | Feature gate |
|---|---|---|---|
| `unipoly_mul` | FLINT `fmpz_poly_mul` (NTT/FFT internally) | — | default |
| `macaulay_reduce_mod_p` | `MacaulayMatrix::reduce_cpu` | `reduce_gpu` (PTX) | `groebner-cuda` |
| NVPTX batch eval | `alkahest_bench` / `examples/cpu_jit_probe.rs` | `compile_cuda` | `cuda` |
