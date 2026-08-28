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
- Parallel simplification, two schedulers (`--features parallel`): fork-join
  (`simplify_par`) and level-scheduled (`simplify_redex`, deterministic derivation
  logs), with `simplify_auto` dispatching on shape and `simplify_strategy` reporting
  the choice — see [Simplification](mdbook/src/simplification.md) for which wins where
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
- Coefficient asymptotics of rational generating functions (`experimental.coefficient_asymptotics`): singularity analysis with the leading constant by Richardson extrapolation; declines when the dominant singularity is not unique (equal-modulus poles make the coefficients oscillate)
- Asymptotics of sums (`experimental.euler_maclaurin`): Euler–Maclaurin expansion of `Σ_{k=a}^{n} f(k)` with Bernoulli corrections, numerically gated, returning an `AsymptoticReport` that marks each hypothesis checked or assumed (the additive constant — γ for the harmonic numbers — is fitted, not proved, and labelled as such)

- Validated numerics (`bound_on_box`, `verified_integral`, `verified_no_roots`, `verified_sign`): Taylor models over a box with Moore–Skelboe branch-and-bound; rigorous range enclosures, definite-integral enclosures and three-valued (`true`/`false`/`undecided`) predicates. Sound before tight — a wide bound is returned rather than a wrong one, and unbounded cases refuse (`E-VALIDATED-*`). Coverage is the elementary fragment plus `erf`/`erfc`, the Bessel pair `bessel_j0`/`bessel_j1`, `gamma`, `digamma`, `lambert_w` and the exponential-integral family `Ei`/`li`/`Si`/`Ci`/`Shi`/`Chi`, and is queryable before you commit to a route: `bounds_supported(expr)`, and `taylor_model` per primitive in `capabilities()["primitives"]` (not `numeric_ball`, which is pointwise ball arithmetic — the two now differ only on `floor`/`ceil`, which are not differentiable and will not get a rule)

## Discrete mathematics

- Symbolic summation: indefinite and definite via Gosper's algorithm (`sum_indefinite`, `sum_definite`)
- Linear recurrence solving (`solve_linear_recurrence_homogeneous`)
- Difference equations / `rsolve`: constant-coefficient recurrences with polynomial RHS
- Symbolic products: definite and indefinite via Γ-ratio telescoping (`product_definite`, `product_indefinite`, `Product`)
- Creative telescoping / Zeilberger's algorithm (`zeilberger`): P-recursive recurrence for a proper hypergeometric term plus a rational certificate, re-checked as an exact `Q(n)(k)` identity before it is returned; refuses (`E-HOLO-*`) rather than guessing outside the class or beyond the search bounds. `order_is_minimal` reports whether the search established that no lower-order relation exists — the default cost-ordered search usually cannot, and says so; `minimal=True` searches order-ascending and can establish it, at a cost that grows with `max_degree` (free at `max_degree=4`, ~13 s versus 0.08 s at 16 on Apéry), so it is opt-in rather than the default
- Boundary verdict for creative telescoping (`ZeilbergerCertificate.boundary`): whether the certificate implies a recurrence for the **sum** over the range in `limits` (default `k = 0..n`, echoed back rather than inferred) — `"vanishes"` (homogeneous recurrence proved by exact order counting in `Q(n)`), `"nonzero"` (inhomogeneous recurrence proved, with `b(n)` in `boundary_rhs`) or `"unknown"` (nothing may be claimed). `boundary_at(k_lo, k_hi)` re-decides for another range without re-running the search
- `q`-analogue creative telescoping (`experimental.q_zeilberger`): `q`-Zeilberger for `q`-hypergeometric summands (Gaussian binomials `qbinomial(N, K)`, `q`-Pochhammer symbols `qpochhammer(u, d, v)`, powers of `q` with a degree-≤2 exponent in `n, k`), which the classical engine cannot express at all. The certificate is re-checked as an exact `Q(q)(q**n)(q**k)` identity before return; `sum_term(n0)` gives the exact `q`-series value from the definition of the `q`-Pochhammer symbol, so the returned recurrence can be checked independently of the machinery that produced it. The boundary verdict is two-valued — `"vanishes"` (proved for `S(n) = Σ_{k ∈ Z} F(n,k)`, with the proved support window in `support`) or `"unknown"` — and `q` is treated as transcendental throughout, so a verdict does not license specialising `q` to a root of unity. Refuses with `E-HOLO-020` (outside the class), `E-HOLO-021` (bounds exhausted), `E-HOLO-023` (malformed call) or `E-HOLO-024` (in the shape of the class but with a non-rational shift quotient, e.g. `(q; q**2)_k` shifted in `k`)
- Recurrence guessing (`guess_holonomic`): fit a P-recursive recurrence to the first terms of a sequence in exact rational arithmetic, the guessing half of *guess then prove*. Only fits candidates the terms over-determine, reports how many surplus terms confirmed the fit, and refuses (`E-HOLO-005`) rather than returning an interpolation or reporting an untested grid as a negative
- Modular / `p`-adic evaluation of a holonomic sequence (`ModularRecurrence`): `S(N) mod p^k` straight from `Σ_i a_i(n)·S(n+i) = b(n)`, in machine-word modular arithmetic and `O(1)` memory, without ever forming `S(N)` over `ℤ`. Indices where the leading coefficient `a_J(n)` is not a unit mod `p` are handled by a first pass that measures the total `p`-adic precision loss and a forward pass that runs at `p^(k+loss)`; a step that cannot be justified refuses (`E-HOLO-007`) and a working precision past the 64-bit modulus refuses (`E-HOLO-008`), so no path returns a residue that is silently short of the precision it claims. `supercongruence_sweep` drives it over a range of primes and reports counterexamples, the `v_p(LHS − RHS)` histogram and whether the claimed modulus is sharp
- `binomial(a, b) mod p^k` (`binomial_mod`): Lucas at `k = 1`, Andrew Granville / Davis–Webb for prime powers, with the `p`-free factorial taken by a product tree over blocks of `p` so the cost is `O(p·k³ + log_p(a)·p·k)` rather than `O(p^k)`; `a` far larger than `p` is the ordinary case

- Positivity certificates (`sos_decompose`, `prove_nonneg`): exact rational sum-of-squares and Handelman certificates on basic semialgebraic sets, re-expanded and checked identically before return; distinguishes "certified", "definitely negative (with witness)" and "no certificate at this degree" (`E-SOS-*`)

## Number theory

- `alkahest.number_theory`: `isprime`, `factorint`, `nextprime`, `totient`, `jacobi_symbol`
- `nthroot_mod` (prime modulus), `discrete_log` (moderate primes)
- Quadratic Dirichlet characters (`DirichletChi`) on odd square-free conductors
- Diophantine equations: linear families, sum of two squares, unit Pell equation (`diophantine`)

## Polynomial system solving (default feature: `groebner`)

- Gröbner basis: Buchberger F4 with product-criterion pruning
- Parallel F4 S-polynomial reduction via Rayon (`--features parallel`)
- CUDA Macaulay-matrix row reduction (`--features groebner-cuda`) — a Rust-crate entry point (`compute_groebner_basis_gpu`); **not** wired into `solve`/`GroebnerBasis.compute`, so it accelerates nothing from Python
- Monomial orders: Lex, GrLex, GRevLex
- `solve` — symbolic solution of polynomial systems (exact symbolic output)
- Regular chains / triangular decomposition (`triangularize`, `RegularChain`)
- Primary decomposition and radical (`primary_decomposition`, `radical`)
- Differential algebra / Rosenfeld–Gröbner for polynomial DAEs (`rosenfeld_groebner`)
- Gröbner bases over the coefficient field `Q(params)` (`GroebnerBasis.compute(polys, vars, params=[...])`, experimental `ParametricGroebnerBasis` / `ParametricGbPoly`) — parameters live in the coefficients, not the ring, so they never enter the monomial order or generate S-pairs; the basis is generic and reports the parameter hypersurfaces (`conditions()`) it assumed non-zero
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
- NVPTX (CUDA GPU) codegen for `sm_86` via `compile_cuda` (`--features cuda`, 16.2× over CPU on RTX 3090; source build only — no published wheel carries it)
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
- `numpy_eval_par` for multi-core batch evaluation — enabled in every published wheel. Requires `--features parallel` in a source build; without it this is a silent alias for `numpy_eval` (correct results, no speedup). Check `capabilities()["features"]["parallel"]`
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
- Exponential-integral family as first-class primitives: `Ei`, `li`, `Si`, `Ci`, `Shi`, `Chi` (conventions per DLMF §6.2 — `Ci`, `Chi` and `li` refuse on the negative reals, where they are complex, rather than returning a real part). Each carries a derivative rule (`exp(x)/x`, `1/log x`, `sin(x)/x`, `cos(x)/x`, `sinh(x)/x`, `cosh(x)/x`), an `f64` and a ball evaluator, and a rigorous Taylor-model rule. Exposed to Python as `exp_integral_ei`, `log_integral`, `sin_integral`, `cos_integral`, `sinh_integral`, `cosh_integral`
- Each primitive: numerical evaluator, ball evaluator, forward/reverse diff, MLIR lowering, Lean theorem tag

## Error handling

- Structured exception hierarchy with stable codes (`E-POLY-*`, `E-DIFF-*`, etc.)
- Every exception: `.code`, `.message`, `.remediation`, `.span`
- Subsystems: ConversionError, DomainError, DiffError, IntegrationError, MatrixError, LinearAlgebraError, EigenError, CadError, OdeError, DaeError, JitError, CudaError, PoolError, SolverError, SosError, HolonomicError, ValidatedError, LimitError, SeriesError, SumError, ProductError, PslqError, DiophantineError, NumberTheoryError, HomotopyError, DiffAlgError, BudgetExceededError, AnsatzError, CrossCheckError, SmtError, CertificateUnavailableError
- **Refusals are distinguished from verdicts.** `E-CAD-001`, `E-LINALG-010`, `E-MAT-004`, `E-SOS-002`, `E-ANSATZ-003`, `E-SMT-003` and `E-BUDGET-*` mean *undecided*, not *false*

## Autoresearch modules

- `alkahest.ansatz` — parametric families (`polynomial`, `rational`, `exponential_polynomial`, `linear_combination`, `quadratic_form`) with `fit`, `enumerate_family`, `certify_nonneg`
- `alkahest.crosscheck` — differential testing against an external CAS oracle: `check`, `sweep`, `run_frozen_corpus`, `to_sympy`, `register_oracle`; a missing oracle reports `unavailable`, never `agree`
- `alkahest.smt` — SMT-LIB 2 export (`to_smtlib`) and z3/cvc5 bridge (`solve`, `supported`, `solvers`); `sat` models are checked in-process, `unsat` is reported as `externally_asserted`
- `alkahest.research` — session claim graphs and provenance
- `Budget` / `context(budget=…)` / `request_cancel` / `batch_map` / `*_many` — bounded, cancellable, non-aborting fan-out

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
