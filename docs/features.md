# Feature surface

Current stable feature surface.

## Core expression kernel

- Hash-consed DAG with structural equality as pointer comparison
- N-ary `Add` / `Mul` with AC normalization at construction
- Arbitrary-precision integers and rationals (FLINT/GMP)
- Symbol domains: `real`, `positive`, `nonnegative`, `integer`, `complex`
- Non-commutative symbols: `pool.symbol("A", commutative=False)`
- Persistent pool: serialize to disk and reload across sessions
- Sharded pool for concurrent insertion (`--features parallel`)

## Simplification

- Rule-based fixpoint simplification (`simplify`)
- Domain-specific rule sets: trig (`simplify_trig`), log/exp (`simplify_log_exp`), expanded (`simplify_expanded`)
- Custom rule sets via `make_rule` / `simplify_with`
- Colored e-graphs for conditional simplification (`SimplifyConfig::assumptions`; e.g. `x > 0` enables `sqrt(x²) → x`)
- E-graph equality saturation via vendored egglog (`simplify_egraph`, `--features egraph`; included in default PyPI wheels)
- Match-disjoint egglog rule scheduling (`EgraphConfig::disjoint_schedule`, default on)
- Discrimination-net indexing for user `PatternRule` sets (Rust `PatternRuleSet` / `simplify_with_pattern_rules`)
- Pluggable cost functions: `SizeCost`, `DepthCost`, `OpCost`, `StabilityCost`
- Phased saturation with `node_limit` / `iter_limit` config
- `collect_like_terms`, `poly_normal`
- Branch-cut-aware log/exp rewrites with `SideCondition` tracking
- Parallel simplification (`simplify_par`, `--features parallel`)
- Pauli and Clifford algebra rewrite tables (`simplify_pauli`, `simplify_clifford_orthogonal`)

## Polynomial algebra (FLINT-backed)

- `UniPoly`: dense univariate polynomial arithmetic, GCD, degree, coefficients
- `MultiPoly`: sparse multivariate polynomial arithmetic, GCD, total degree
- `RationalFunction`: quotient with automatic GCD normalization
- Horner-form rewriting (`horner`); SIMD batch f64 Horner eval (`eval_horner_f64_batch`, Rust)
- C code emission (`emit_c`)
- Polynomial factorization over ℤ, ℤ[x₁,...,xₙ], and 𝔽ₚ (Zassenhaus, van Hoeij, Berlekamp, Cantor–Zassenhaus via FLINT); integer factorization outputs include exact in-kernel factor-product reconstruction metadata
- Hermite and Smith normal forms for integer matrices and polynomial matrices over ℚ[x]
- LLL lattice reduction over ℤ (`alkahest.lattice`)
- Approximate integer-relation finding (`guess_relation`)
- Modular/CRT arithmetic (`alkahest.modular`)
- Resultants and subresultant PRS
- Sparse multivariate interpolation: Ben-Or/Tiwari univariate recovery (`sparse_interp_univariate`) and Zippel multivariate recovery (`sparse_interp`); `MultiPolyFp` sparse polynomial over 𝔽ₚ; rational reconstruction, CRT lifting, Mignotte bound
- Sparse modular GCD: Zippel evaluation–interpolation GCD over ℤ[x₁,…,xₙ] using `sparse_interp` as oracle, CRT lifting over lucky primes (`gcd_sparse`, `SparseGcdError`)

## Calculus

- Symbolic differentiation (`diff`, `diff_forward`)
- Forward-mode automatic differentiation
- Reverse-mode partials on `Expr` (`symbolic_grad`) — distinct from JAX-style `grad` on `TracedFn`
- Symbolic integration: power rule, log, exp tower, linear substitution, trig, and full rational-function integration (Hermite reduction, Rothstein–Trager, arctan for irreducible quadratics, √-coefficient logs, `RootSum` for degree-≥3 factors via Lazard–Rioboo–Trager)
- Rational Risch DE for `f(x)·exp(η)` integrands with `f ∈ ℚ(x)` (Bronstein §6.1)
- Non-elementary certification via Liouville's theorem (`E-INT-004`): `sin(x)/x`, `exp(x)/x`, `log(x)^(−n)`, etc. raise `NonElementary` instead of `NotImplemented`
- `RootSum` kernel node: first-class symbolic sum over algebraic roots, with differentiation, display (Debug / LaTeX / unicode), persistence (pool format V5), and PyO3 bridge
- Truncated Taylor and Laurent series (`series`, `Series`)
- Limits (`limit`, `LimitDirection`): L'Hôpital, local expansions, limits at ±∞

## Discrete mathematics

- Symbolic summation: indefinite and definite via Gosper's algorithm (`sum_indefinite`, `sum_definite`)
- Linear recurrence solving (`solve_linear_recurrence_homogeneous`)
- Difference equations / `rsolve`: constant-coefficient recurrences with polynomial RHS
- Symbolic products: definite and indefinite via Γ-ratio telescoping (`product_definite`, `product_indefinite`, `Product`)

## Number theory

- `alkahest.number_theory`: `isprime`, `factorint`, `nextprime`, `totient`, `jacobi_symbol`
- `nthroot_mod` (prime modulus), `discrete_log` (moderate primes)
- Quadratic Dirichlet characters (`DirichletChi`) on odd square-free conductors
- Diophantine equations: linear families, sum of two squares, unit Pell equation (`diophantine`)

## Polynomial system solving (default feature: `groebner`)

- Gröbner basis: Buchberger F4 with product-criterion pruning
- Parallel F4 S-polynomial reduction via Rayon (`--features parallel`)
- CUDA Macaulay-matrix row reduction (`--features groebner-cuda`)
- Monomial orders: Lex, GrLex, GRevLex
- `solve` — symbolic solution of polynomial systems (exact symbolic output)
- Regular chains / triangular decomposition (`triangularize`, `RegularChain`)
- Primary decomposition and radical (`primary_decomposition`, `radical`)
- Differential algebra / Rosenfeld–Gröbner for polynomial DAEs (`rosenfeld_groebner`)
- Numerical algebraic geometry: total-degree homotopy continuation with Smale certification (`solve_numerical`, `CertifiedSolution`)
- Eigenvalues, eigenvectors, diagonalization for symbolic matrices (`eigenvals`, `eigenvects`, `diagonalize`)

## Transformations

- `trace` / `trace_fn` — symbolic function tracing
- `grad` — gradient of a `TracedFn` (`@trace`); pairs with `jit`. Use `symbolic_grad` for `Expr` partials
- `jit` — LLVM JIT compilation of traced functions
- `CompiledTracedFn` for array-vectorised evaluation
- JAX-style pytree flattening (`flatten_exprs`, `unflatten_exprs`, `map_exprs`)
- Context manager (`alkahest.context(pool=..., simplify=...)`)

## Code generation

- Three-tier CPU evaluation: interpreter → Cranelift JIT (`--features cranelift`) → LLVM JIT (`--features jit`), selected by `CompileConfig` (DAG size + `expected_evals`)
- `CompileCache` — memoize compiled functions keyed by `(ExprId, input variables)`; Python `CompileCache` class
- Bulk column-major batch evaluation (`CompiledFn::call_bulk` / `call_batch`; native `alkahest_eval_bulk` when JIT backends are enabled)
- LLVM JIT for native CPU code (`--features jit`; `+jit` / `+full` release wheels)
- NVPTX (CUDA GPU) codegen for `sm_86` (`--features cuda`, 16.2× over CPU on RTX 3090)
- Custom `alkahest` MLIR dialect with three lowering targets: ArithMath, StableHLO, LLVM
- `to_stablehlo` — emit textual StableHLO MLIR for XLA/JAX
- DAG-aware memoization on hot recursive paths (simplify, diff, integrate, interpreter eval)

## Ball arithmetic

- `ArbBall`: real interval `[mid ± rad]` backed by Arb (FLINT)
- `AcbBall`: complex ball arithmetic
- `interval_eval`: rigorously evaluate a symbolic expression with ball inputs
- Guaranteed enclosures for all arithmetic and transcendental operations

## Numerical interop

- `compile_expr` + `eval_expr` for scalar evaluation
- `numpy_eval` for vectorised batch evaluation (NumPy, PyTorch, JAX arrays)
- `numpy_eval_par` for multi-core batch evaluation (`--features parallel`; falls back to `numpy_eval`)
- DLPack support for zero-copy interop
- `to_jax` — register a symbolic expression as a JAX primitive with JVP and vmap rules

## Mathematical operations

- Symbolic matrices (`Matrix`), determinant, inverse, transpose, Jacobian
- ODE representation and first-order lowering (`ODE`, `lower_to_first_order`)
- DAE structural analysis: Pantelides index reduction (`DAE`, `pantelides`)
- Acausal component modeling (`AcausalSystem`, `Port`, `Component`, `resistor`, `capacitor`, `voltage_source`)
- Laplace transform (`alkahest.experimental.laplace_transform` / `inverse_laplace_transform`)
- Sensitivity analysis: forward (`sensitivity_system`) and adjoint (`adjoint_system`)
- Hybrid systems with events (`HybridODE`, `Event`)
- Piecewise expressions and predicates

## Plotting

- **No bundled dependency** — detects and uses whatever the user has installed (Matplotlib or Plotly).
- `plot(expr, var, range_)` — 1-D curve (Matplotlib or Plotly backend).
- `plot3d(expr, var_x, var_y, x_range, y_range)` — 3-D surface.
- `plot_parametric(expr_x, expr_y, param, range_)` — parametric curve.
- `plot_implicit(expr, var_x, var_y, x_range, y_range)` — zero-set of a 2-variable expression (contour at 0).
- `plot_roots(unipoly, var)` — real root markers on the x-axis (rug plot via `real_roots`).
- `plot_series(series_result, original_expr, var, range_)` — Taylor/Laurent truncation vs exact.
- `plot_dag(expr)` — expression DAG via Graphviz Python package (falls back to raw DOT string).
- `plot_svg(expr, var, range_)` — standalone SVG polyline rendered entirely in Rust; no Python plotting dep required; suitable for embedding in HTML or Jupyter.
- `alkahest.experimental._fastplotlib` — GPU-accelerated `fplot` / `fplot3d` via fastplotlib (WGPU); recommended for dense grids with the `+full` JIT wheel.

## Output and parsing

- LaTeX pretty-printing (`latex(expr)`)
- Unicode pretty-printing (`unicode_str(expr)`)
- String expression parsing (`parse(string, pool, bindings)`)

## Lean certificates (proof export)

- Derivation logs always on: ordered `RewriteStep` list with rule names and side conditions
- Lean 4 proof term export for: polynomial differentiation, trig differentiation, basic arithmetic rewrites
- Algorithmic certificates (witness-based): polynomial GCD, factoring (claims verified by Lean `ring_nf`)
- Lean CI: auto-generates proof corpus and verifies via lean compiler
- 20+ rule → Lean / Mathlib theorem mappings

## Primitive registry

- 23 registered primitives with full bundles: sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, exp, log, sqrt, abs, sign, erf, erfc, gamma, floor, ceil, round, min, max
- Each primitive: numerical evaluator, ball evaluator, forward/reverse diff, MLIR lowering, Lean theorem tag

## Error handling

- Structured exception hierarchy with stable codes (`E-POLY-*`, `E-DIFF-*`, etc.)
- Every exception: `.code`, `.message`, `.remediation`, `.span`
- Subsystems: ConversionError, DomainError, DiffError, IntegrationError, MatrixError, OdeError, DaeError, JitError, CudaError, PoolError, SolverError, LimitError, SeriesError, ProductError, DiophantineError, NumberTheoryError, EigenError, HomotopyError, DiffAlgError

## Cross-CAS benchmarks

- Benchmark driver against SymPy, SymEngine, WolframEngine, Maple, SageMath
- HTML + JSONL reports via Criterion dashboard
- CodSpeed continuous benchmarking (Rust + Python) in CI
- Nightly CI runs with `--competitors` flag
- Agent benchmark suite: 17 tasks across 6 categories comparing alkahest, SymPy, and Mathematica skill guides

## Reinforcement learning (`alkahest.rl`)

Optional Python extra: `pip install "alkahest[rl]"` (Python ≥ 3.10; pulls `verifiers` + `datasets`).

- **Core** (`alkahest.rl.core`): `BaseGenerator`, `BaseVerifier`, `Rubric`, `CurriculumScheduler` — framework-agnostic
- **Integration env** (`alkahest.rl.envs.integration`): Risch-tier task grammar, layered `IntegrationVerifier` (simplify → e-graph → interval spot checks), Prime Intellect `load_environment()` entry point
- Hard-negative NonElementary samples train honest refusal; curriculum scheduler advances tiers on pass rate
- veRL recipe: `recipes/verl_integration_reward.py`
- Environments Hub manifest: `python/alkahest/rl/envs/integration/` (`prime env push`)

See the [RL guide](./mdbook/src/rl.md) for install, API, and Hub publishing steps.

## Planned

- AMD ROCm / `amdgcn` GPU codegen (hardware-blocked)
- Generalized Pell equations and quadratic Diophantines with cross-term
- Higher-degree algebraic Risch (multiple generators, cbrt/nth-root extensions)
- Cylindrical algebraic decomposition (full real QE)
- PyPI wheel publishing
