# Alkahest Roadmap

## Status

| Version | Status | Highlights |
|---------|--------|-----------|
| **0.1 – 0.5** | ✅ Complete | Foundations through Lean certificates and GPU (see [CHANGELOG](CHANGELOG.md)) |
| **1.0.0** | ✅ Complete | Production NVPTX, MLIR dialect, Gröbner solver, semver API, persistent pool |
| **2.0.0** | ✅ Complete | Full mathematical coverage: summation, series, limits, eigenvalues, number theory, noncommutative algebra, regular chains, primary decomposition, differential algebra, homotopy continuation, Diophantine equations, symbolic products, rsolve, algebraic Risch, LaTeX/Unicode output, string parsing |
| **Next** | 📋 Planned | See below |

**Test coverage:** 362 Rust unit/proptest/doctest + 438 Python tests = 800 passing, zero errors.

---

## 2.0.0 — Complete

All items below shipped in the 2.0.0 release. See [CHANGELOG](CHANGELOG.md) for full details.

### Calculus and series
- `series(expr, var, point, order)` — truncated Taylor and Laurent expansions; `BigO` kernel node
- `limit(expr, var, point)` — finite-point L'Hôpital + local expansions; limits at ±∞ via `x ↦ 1/t`; `LimitDirection`
- Algebraic-function Risch integration (Trager genus-0): integrals of `A(x) + B(x)·sqrt(P(x))` over ℚ(x)

### Discrete mathematics
- Symbolic summation: `sum_indefinite`, `sum_definite` (Gosper's algorithm); `verify_wz_pair`
- Linear recurrence solving: `solve_linear_recurrence_homogeneous`
- Difference equations: `rsolve` for constant-coefficient recurrences with polynomial RHS
- Symbolic products: `product_definite`, `product_indefinite`, `Product` (Γ-ratio telescoping)

### Algebra and number theory
- Matrix eigenvalues, eigenvectors, diagonalization (`eigenvals`, `eigenvects`, `diagonalize`)
- Integer number theory module: `isprime`, `factorint`, `nextprime`, `totient`, `jacobi_symbol`, `nthroot_mod`, `discrete_log`, `DirichletChi`
- Diophantine equations: linear families, sum of two squares, unit Pell equation
- Noncommutative algebra: `commutative=False` symbols; Pauli and Clifford algebra rewrite tables; pool format v4

### Advanced polynomial solvers
- Regular chains / triangular decomposition: `triangularize`, `RegularChain`
- Primary decomposition and radical: `primary_decomposition`, `radical`, `PrimaryComponent`
- Differential algebra / Rosenfeld–Gröbner for polynomial DAEs: `rosenfeld_groebner`, `dae_index_reduce`
- Numerical algebraic geometry: total-degree homotopy continuation with Smale certification (`solve_numerical`, `CertifiedSolution`)

### Developer experience
- LaTeX and Unicode pretty-printing: `latex(expr)`, `unicode_str(expr)`
- String expression parsing: `parse(source, pool)`, `ParseError`
- E-graph default rules: trig and log/exp identities on by default; `EgraphConfig` opt-out
- Python API completeness: `ExprPool.save_to/load_from`, `GroebnerBasis.compute`, symbolic `solve` output
- Windows + macOS CI parity (`ci-cross.yml`)

---

## 1.0.0 — Complete

- Production NVPTX codegen for `sm_86` (Ampere): 16.2× speedup over CPU JIT on RTX 3090
- Custom `alkahest` MLIR dialect: 11 ops, three lowering targets (ArithMath, StableHLO, LLVM)
- CUDA Macaulay-matrix row reduction (`--features groebner-cuda`)
- Gröbner-based polynomial system solver: Lex basis → triangular backsolve → exact symbolic solutions
- Polynomial factorization over ℤ, ℤ[x₁…xₙ], and 𝔽ₚ (Zassenhaus, van Hoeij, Berlekamp, Cantor–Zassenhaus)
- Integer Hermite and Smith normal forms; polynomial-matrix HNF/Smith over ℚ[x]
- Exact LLL lattice reduction; approximate integer-relation finding (`guess_relation`)
- Semver-stable API: `alkahest_core::stable` / `alkahest_core::experimental` split; `alkahest.__all__` freeze
- 23-primitive registry with full diff/MLIR/Lean bundles
- Persistent `ExprPool`: versioned binary format, atomic crash-safe writes
- Cross-CAS benchmark driver: SymPy, SymEngine, WolframEngine, Maple, SageMath adapters

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
