# Alkahest Roadmap

## Status

| Version | Status | Highlights |
|---------|--------|-----------|
| **0.1 – 0.5** | ✅ Complete | Foundations through Lean certificates and GPU (see [CHANGELOG](CHANGELOG.md)) |
| **1.0.0** | ✅ Complete | Production NVPTX, MLIR dialect, Gröbner solver, semver API, persistent pool (see [CHANGELOG](CHANGELOG.md#100)) |
| **2.0.0** | ✅ Complete | Full mathematical coverage: summation, series, limits, eigenvalues, number theory, noncommutative algebra, regular chains, primary decomposition, differential algebra, homotopy continuation, Diophantine equations, symbolic products, rsolve, algebraic Risch, LaTeX/Unicode output, string parsing (see [CHANGELOG](CHANGELOG.md#200--2026-05-06)) |
| **Next** | 🚧 In progress | Gruntz comparability-graph limits (V2-17), see below |

**Test coverage:** 362 Rust unit/proptest/doctest + 438 Python tests = 800 passing, zero errors.

---

## Planned

Items in rough priority order. None are committed to a specific release date.

### Near-term
- **Full Gruntz limits** ✅ implemented in `feature/gruntz-limits` — comparability-graph MRV algorithm for exp-log sequences (`alkahest-core/src/calculus/gruntz.rs`)
- **Lean `Filter.Tendsto` certificate export** ✅ implemented — `lean::emit_tendsto_cert` generates Lean 4 `Filter.Tendsto` proof obligations; pattern-dispatches to Mathlib theorems (`tendsto_exp_neg_atTop_nhds_zero`, `tendsto_exp_atTop`, etc.) and falls back to `sorry` for complex cases
- **Polyhedral / mixed-volume homotopy** ✅ implemented — BKK mixed-volume start systems for 2-variable sparse polynomial systems (`alkahest-core/src/solver/polyhedral.rs`); `solve_numerical` automatically selects polyhedral homotopy when MV < Bézout bound

### Mathematical coverage
- **F5 / signature-based Gröbner** — eliminate zero reductions; ≥ 2× speedup over F4 on Cyclic-7 and larger
- **Sparse multivariate interpolation** (Ben-Or/Tiwari, Zippel) — black-box recovery; substrate for faster modular algorithms
- **Full cylindrical algebraic decomposition** (real quantifier elimination) — Brown projection + lift; `decide(formula)` for first-order sentences over ℝ
- **Generalized Pell and higher-degree Diophantine** — `x² - D·y² = N` for arbitrary N; quadratics with cross-term
- **Higher-degree algebraic Risch** — multiple generators; cbrt and nth-root extensions; full Trager algorithm

### Infrastructure
- **Complete Lean certificate coverage** — bring Lean 4 export (rewrite-tagged proof traces and algorithmic witnesses) up to parity with every supported proof-producing operation; track gaps and add regression checks so new algorithms and rules do not land without matching certificate paths or Mathlib theorem hooks.
- **First-class Rust crate** — publish `alkahest-core` on [crates.io](https://crates.io) and document direct Rust use of the semver-stable `alkahest_core::stable` API (examples, feature-flag matrix for optional backends) so the kernel is a normal library dependency, not only a Python extension build artifact.
- **LLVM JIT + full-feature wheels — PyTorch-style auxiliary index** — keep default PyPI wheels free of the LLVM/inkwell dependency and heavy optional features; publish **`+jit`** (JIT only) and **`+full`** (`jit groebner parallel egraph`) under PEP 440 local versions on a **separate PEP 503 index** (or GitHub Release assets until that index exists) so `pip install alkahest` stays on the small default while `pip install 'alkahest==…+jit'` / `'…+full'` with `--extra-index-url …` opts in. Rationale: if local-version wheels and the plain release were both uploaded to the same PyPI project, many resolvers would treat the local segment as newer and pull the large binary by default.
- **Native Rust codegen (exploratory)** — optional backend to compile hot numeric eval paths to pure Rust machine code (or a pure-Rust codegen crate such as Cranelift) so prebuilt wheels can offer strong CPU performance **without** shipping libLLVM; LLVM/NVPTX remains for GPU and MLIR interop where needed.
- **AMD ROCm / `amdgcn` codegen** — hardware-blocked until RDNA3 / MI-series runner is available
