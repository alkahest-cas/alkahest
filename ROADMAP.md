# Alkahest Roadmap

## Status

| Version | Status | Highlights |
|---------|--------|-----------|
| **0.1 – 0.5** | ✅ Complete | Foundations through Lean certificates and GPU (see [CHANGELOG](CHANGELOG.md)) |
| **1.0.0** | ✅ Complete | Production NVPTX, MLIR dialect, Gröbner solver, semver API, persistent pool (see [CHANGELOG](CHANGELOG.md#100)) |
| **2.0.0** | ✅ Complete | Full mathematical coverage: summation, series, limits, eigenvalues, number theory, noncommutative algebra, regular chains, primary decomposition, differential algebra, homotopy continuation, Diophantine equations, symbolic products, rsolve, algebraic Risch, LaTeX/Unicode output, string parsing (see [CHANGELOG](CHANGELOG.md#200--2026-05-06)) |
| **Next** | 📋 Planned | See below |

**Test coverage:** 362 Rust unit/proptest/doctest + 438 Python tests = 800 passing, zero errors.

---

## Planned

Items in rough priority order. None are committed to a specific release date.

### Near-term
- **Full Gruntz limits** — comparability-graph algorithm for general transcendental sequences; Lean `Filter.Tendsto` certificates (current implementation uses prototype L'Hôpital rules)
- **Polyhedral / mixed-volume homotopy** — needed for deficient systems whose affine root count is below the Bézout bound (e.g. Katsura family)

### Mathematical coverage
- **Sparse multivariate interpolation** (Ben-Or/Tiwari, Zippel) — black-box recovery; substrate for faster modular algorithms
- **Full cylindrical algebraic decomposition** (real quantifier elimination) — Brown projection + lift; `decide(formula)` for first-order sentences over ℝ
- **Generalized Pell and higher-degree Diophantine** — `x² - D·y² = N` for arbitrary N; quadratics with cross-term
- **Higher-degree algebraic Risch** — multiple generators; cbrt and nth-root extensions; full Trager algorithm

### Infrastructure
- **Complete Lean certificate coverage** — bring Lean 4 export (rewrite-tagged proof traces and algorithmic witnesses) up to parity with every supported proof-producing operation; track gaps and add regression checks so new algorithms and rules do not land without matching certificate paths or Mathlib theorem hooks.
- **LLVM JIT + full-feature wheels — PyTorch-style auxiliary index** — keep default PyPI wheels free of the LLVM/inkwell dependency and heavy optional features; publish **`+jit`** (JIT only) and **`+full`** (`jit groebner parallel egraph`) under PEP 440 local versions on a **separate PEP 503 index** (or GitHub Release assets until that index exists) so `pip install alkahest` stays on the small default while `pip install 'alkahest==…+jit'` / `'…+full'` with `--extra-index-url …` opts in. Rationale: if local-version wheels and the plain release were both uploaded to the same PyPI project, many resolvers would treat the local segment as newer and pull the large binary by default.
- **Native Rust codegen (exploratory)** — optional backend to compile hot numeric eval paths to pure Rust machine code (or a pure-Rust codegen crate such as Cranelift) so prebuilt wheels can offer strong CPU performance **without** shipping libLLVM; LLVM/NVPTX remains for GPU and MLIR interop where needed.
- **AMD ROCm / `amdgcn` codegen** — hardware-blocked until RDNA3 / MI-series runner is available
