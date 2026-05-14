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
- **PyPI wheel publishing** — manylinux 2_28 / macOS arm64 / Windows GNU wheel matrix via `release.yml`; default uploads exclude LLVM **+jit** builds (GitHub Release assets + future PEP 503 “extra index” — see README)
- **Full Gruntz limits** — comparability-graph algorithm for general transcendental sequences; Lean `Filter.Tendsto` certificates (current implementation uses prototype L'Hôpital rules)
- **Polyhedral / mixed-volume homotopy** — needed for deficient systems whose affine root count is below the Bézout bound (e.g. Katsura family)

### Mathematical coverage
- **F5 / signature-based Gröbner** — eliminate zero reductions; ≥ 2× speedup over F4 on Cyclic-7 and larger
- **Sparse multivariate interpolation** (Ben-Or/Tiwari, Zippel) — black-box recovery; substrate for faster modular algorithms
- **Full cylindrical algebraic decomposition** (real quantifier elimination) — Brown projection + lift; `decide(formula)` for first-order sentences over ℝ
- **Generalized Pell and higher-degree Diophantine** — `x² - D·y² = N` for arbitrary N; quadratics with cross-term
- **Higher-degree algebraic Risch** — multiple generators; cbrt and nth-root extensions; full Trager algorithm

### Infrastructure
- **LLVM JIT wheels — PyTorch-style auxiliary index** — keep default PyPI wheels free of the LLVM/inkwell dependency; publish LLVM-enabled builds under a PEP 440 local version (for example `2.0.0+jit`) on a **separate PEP 503 index** (or GitHub Release assets until that index exists) so `pip install alkahest` stays on the small default while `pip install 'alkahest==…+jit' --extra-index-url …` opts into native CPU JIT. Rationale: if `+jit` and the plain release were both uploaded to the same PyPI project, many resolvers would treat the local segment as newer and pull LLVM by default.
- **Native Rust codegen (exploratory)** — optional backend to compile hot numeric eval paths to pure Rust machine code (or a pure-Rust codegen crate such as Cranelift) so prebuilt wheels can offer strong CPU performance **without** shipping libLLVM; LLVM/NVPTX remains for GPU and MLIR interop where needed.
- **AMD ROCm / `amdgcn` codegen** — hardware-blocked until RDNA3 / MI-series runner is available
