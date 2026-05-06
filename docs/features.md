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
- E-graph equality saturation via egglog (`simplify_egraph`, `--features egraph`)
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
- Horner-form rewriting (`horner`)
- C code emission (`emit_c`)
- Polynomial factorization over ℤ, ℤ[x₁,...,xₙ], and 𝔽ₚ (Zassenhaus, van Hoeij, Berlekamp, Cantor–Zassenhaus via FLINT)
- Hermite and Smith normal forms for integer matrices and polynomial matrices over ℚ[x]
- LLL lattice reduction over ℤ (`alkahest.lattice`)
- Approximate integer-relation finding (`guess_relation`)
- Modular/CRT arithmetic (`alkahest.modular`)
- Resultants and subresultant PRS

## Calculus

- Symbolic differentiation (`diff`, `diff_forward`)
- Forward-mode automatic differentiation
- Reverse-mode automatic differentiation (`symbolic_grad`)
- Symbolic integration: power rule, log, exp tower, linear substitution, trig (Risch subset)
- Algebraic-function Risch integration (Trager's algorithm)
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

## Polynomial system solving (requires `--features groebner`)

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
- `grad` — gradient transformation (symbolically differentiates a traced function)
- `jit` — LLVM JIT compilation of traced functions
- `CompiledTracedFn` for array-vectorised evaluation
- JAX-style pytree flattening (`flatten_exprs`, `unflatten_exprs`, `map_exprs`)
- Context manager (`alkahest.context(pool=..., simplify=...)`)

## Code generation

- LLVM JIT for native CPU code (`--features jit`)
- NVPTX (CUDA GPU) codegen for `sm_86` (`--features cuda`, 16.2× over CPU on RTX 3090)
- Custom `alkahest` MLIR dialect with three lowering targets: ArithMath, StableHLO, LLVM
- `to_stablehlo` — emit textual StableHLO MLIR for XLA/JAX
- Compilation result caching keyed by expression hash

## Ball arithmetic

- `ArbBall`: real interval `[mid ± rad]` backed by Arb (FLINT)
- `AcbBall`: complex ball arithmetic
- `interval_eval`: rigorously evaluate a symbolic expression with ball inputs
- Guaranteed enclosures for all arithmetic and transcendental operations

## Numerical interop

- `compile_expr` + `eval_expr` for scalar evaluation
- `numpy_eval` for vectorised batch evaluation (NumPy, PyTorch, JAX arrays)
- DLPack support for zero-copy interop
- `to_jax` — register a symbolic expression as a JAX primitive with JVP and vmap rules

## Mathematical operations

- Symbolic matrices (`Matrix`), determinant, inverse, transpose, Jacobian
- ODE representation and first-order lowering (`ODE`, `lower_to_first_order`)
- DAE structural analysis: Pantelides index reduction (`DAE`, `pantelides`)
- Acausal component modeling (`AcausalSystem`, `Port`, `resistor`)
- Sensitivity analysis: forward (`sensitivity_system`) and adjoint (`adjoint_system`)
- Hybrid systems with events (`HybridODE`, `Event`)
- Piecewise expressions and predicates

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
- Nightly CI runs with `--competitors` flag
- Agent benchmark suite: 17 tasks across 6 categories comparing alkahest, SymPy, and Mathematica skill guides

## Planned

- AMD ROCm / `amdgcn` GPU codegen (hardware-blocked)
- Full Gruntz algorithm for limits (currently prototype rules)
- Generalized Pell equations and quadratic Diophantines with cross-term
- Sparse multivariate interpolation
- F5 / signature-based Gröbner
- Cylindrical algebraic decomposition (full real QE)
- PyPI wheel publishing
