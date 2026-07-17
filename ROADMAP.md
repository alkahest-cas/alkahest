# Alkahest Roadmap

## Status

| Version | Status | Highlights |
|---------|--------|-----------|
| **0.1 – 0.5** | ✅ Complete | Foundations through Lean certificates and GPU (see [CHANGELOG](CHANGELOG.md)) |
| **1.0.0** | ✅ Complete | Production NVPTX, MLIR dialect, Gröbner solver, semver API, persistent pool (see [CHANGELOG](CHANGELOG.md#100)) |
| **2.0.0** | ✅ Complete | Full mathematical coverage: summation, series, limits, eigenvalues, number theory, noncommutative algebra, regular chains, primary decomposition, differential algebra, homotopy continuation, Diophantine equations, symbolic products, rsolve, algebraic Risch, LaTeX/Unicode output, string parsing (see [CHANGELOG](CHANGELOG.md#200--2026-05-06)) |
| **2.0.3** | ✅ Complete | Full Gruntz MRV limits, polyhedral BKK homotopy, Lean Filter.Tendsto certificates, F5 Gröbner, demo playground Lean panel (see [CHANGELOG](CHANGELOG.md#203--2026-05-21)) |
| **2.0.4** | ✅ Complete | Sparse multivariate interpolation (Ben-Or/Tiwari, Zippel) + sparse modular GCD substrate (see [CHANGELOG](CHANGELOG.md#204--2026-05-22)) |
| **Next** | 🚧 In progress | CAD, generalized Pell, full algebraic Risch — see below |

**Test coverage:** ~607 Rust unit/proptest/doctest + 1028 Python tests, zero errors.

---

## Planned

Items in rough priority order. None are committed to a specific release date.

### Mathematical coverage
- **Full cylindrical algebraic decomposition** (real quantifier elimination) — Brown projection + lift; `decide(formula)` for first-order sentences over ℝ
- **Generalized Pell and higher-degree Diophantine** — `x² - D·y² = N` for arbitrary N; quadratics with cross-term
- **Higher-degree algebraic Risch** — multiple generators; cbrt and nth-root extensions; full Trager algorithm
- **Transcendental Risch (Phase 2)** — ✅ **Implemented** (issue #4): polynomial RDE solver, hyperexponential and hyperlogarithmic tower cases, NonElementary certification; remaining gaps: multiple generators, mixed exp+log towers, rational coefficients in RDE (exp(x)/x), full Liouville structure theorem

### Infrastructure
- **Complete Lean certificate coverage** — bring Lean 4 export (rewrite-tagged proof traces and algorithmic witnesses) up to parity with every supported proof-producing operation; track gaps and add regression checks so new algorithms and rules do not land without matching certificate paths or Mathlib theorem hooks.
- **LLVM JIT + full-feature wheels — PyTorch-style auxiliary index** — keep default PyPI wheels free of the LLVM/inkwell dependency and heavy optional features; publish **`+jit`** (JIT only) and **`+full`** (`jit groebner parallel egraph`) under PEP 440 local versions on a **separate PEP 503 index** (or GitHub Release assets until that index exists) so `pip install alkahest` stays on the small default while `pip install 'alkahest==…+jit'` / `'…+full'` with `--extra-index-url …` opts in. Rationale: if local-version wheels and the plain release were both uploaded to the same PyPI project, many resolvers would treat the local segment as newer and pull the large binary by default.
- **Native Rust codegen (exploratory)** — optional backend to compile hot numeric eval paths to pure Rust machine code (or a pure-Rust codegen crate such as Cranelift) so prebuilt wheels can offer strong CPU performance **without** shipping libLLVM; LLVM/NVPTX remains for GPU and MLIR interop where needed.
- **AMD ROCm / `amdgcn` codegen** — hardware-blocked until RDNA3 / MI-series runner is available
- **Incremental / Merkle expression storage + slotted e-graphs (P2, design-only)** — deprioritized vs mainline; see [RFC 0001](docs/rfcs/0001-incremental-merkle-pool.md) and the bounded `alkahest_core::experimental::merkle_pool` prototype. No default `ExprPool` change; revisit only if RFC success metrics are measured on notebook/agent traces.
