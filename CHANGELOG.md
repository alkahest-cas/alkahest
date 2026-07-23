# Changelog

## Unreleased

### Fixes

- **Laplace hyperbolic inverse:** irreducible quadratics with `ω² < 0`
  (e.g. `1/(s²−2)`) now invert to sinh/cosh instead of `sin(√(−κ²))` (which
  evaluated to NaN / declined). Forward sinh/cosh folds `(√c)²→c` in the
  denominator and the inverse peels s-free amplitudes so `L⁻¹{L{sinh(√2 t)}}`
  round-trips. Literal negative Heaviside/Dirac shifts `θ(t+a)`, `δ(t+a)`
  with `a > 0` are refused (`E-TRANSFORM-001`) rather than emitting the wrong
  unilateral formula.

- **Transform round-trips:** Inverse Laplace now inverts repeated irreducible
  quadratic poles of order 2 (needed for `L⁻¹{L{t sin}}` / `t cos`). Inverse Z
  matches the forward sin/cos table forms directly so transcendental
  coefficients (`sin(ω)`, `cos(ω)`) do not block `Z⁻¹{Z{sin(ωn)}}` via `apart`.
  Locked in by Rust unit tests and `tests/test_transform_roundtrips.py`.

- **`log(exp(z))` over ℂ:** `simplify_log_exp` only folds `log(exp(x))→x` when
  every free symbol in `x` is real-valued; `Domain.Complex` (and `I`) refuse
  the rewrite. Egglog no longer loads `Log∘Exp` (no domain check). Prevents
  silent wrong answers when `Im(z) ∉ (−π, π]`.

- **Complex branch-cut evaluation:** `evaluate(..., mode="complex")` now
  auto-binds the canonical imaginary unit `I → 1j`, accepts real scalar
  bindings, and evaluates non-integer powers on the principal branch via
  `exp(w·Log z)` (so e.g. `(-1)**(1/2) → i`). Complex `sqrt` uses the same
  Log path to avoid cancellation near the negative-real cut. Locked in by an
  mpmath fuzz oracle (`tests/test_complex_branchcut_oracle.py`). Exact `Arg`
  on the cut still declines (`E-EVAL-011`). `ExprPool.imaginary_unit()` is
  exposed in Python; `Expr ** float` builds a float-exponent node.

- **Assumption-gated log/exp rewrites:** `simplify_log_exp` and egglog no longer
  apply branch-cut identities (`exp(log(x))→x`, `log(x)+log(y)→log(xy)`,
  `log(a^n)→n·log(a)`, `log(a/b)→log(a)−log(b)`) without positivity facts.
  Pass an `Assumptions` context or use `Domain.Positive` symbols; safe rules
  `log(exp(x))→x` and `exp(x)·exp(y)→exp(x+y)` still apply unconditionally.
  Static symbol domains are now collected into the colored e-graph pass for
  all `simplify_with` callers.

- **E-graph constant folding:** `simplify_egraph((x+x)/2)` now returns `x`
  instead of leaving `((x * 2) * 1/2)`. The post-extraction const-fold pass
  flattens nested `Add`/`Mul` so coefficients from linear canonization and
  reciprocal folds meet in one n-ary product.

### Features

- **Parametric `solve`:** free symbols omitted from `vars` are treated as
  parameters, so e.g. `solve([x**2 - y], [x])` returns `±sqrt(y)` instead of
  raising `SolverError`.

### Output hygiene

- Parenthesize nested powers in `str` / LaTeX / Unicode so `x^(1/2)^3` is unambiguous.
- `MultiPoly.to_expr` omits unit coefficients (`cancel((x²−1)/(x−1))` → `x + 1`).
- `simplify(gamma(1))` → `1` via a new `PrimitiveFold` rule.
- Literal division by zero raises `ZeroDivisionError` instead of building `0^-1`.

### API

- Hide import-machinery leaks (`contextlib`, `exceptions`, `alkahest`) from
  ``dir(alkahest)`` / autocomplete; submodules remain explicitly importable.
- `UniPoly.from_coefficients` accepts Python ``int`` coefficients (not only ``Expr``).
- `cancel` / `together` / `MultiPoly.from_symbolic` / `radical` infer free symbols when
  *vars* is omitted.
- Structured error messages now include the stable code prefix, e.g. ``[E-INT-004] …``.

### Docs

- Document `parse` in the README quickstart; clarify that `limit` / `series` are not `DerivedResult`.
- Expand `sum_definite` / `sum_indefinite` / `diophantine` / `solve` docs (Faulhaber gap, binary Diophantine patterns, parametric solve).

## 3.6.0 — 2026-07-17

### Release / packaging

- **Cranelift JIT in default PyPI wheels:** default Linux/macOS/Windows wheels ship `egraph` + `groebner` + Cranelift JIT (`cranelift_jit`); LLVM `+jit` / `+full` remain GitHub Release–only local versions.

### Complex / numeric evaluation

- **Complex numeric evaluation and rational residues:** complex-mode numeric evaluation with rational residue support.
- **Principal Arg and complex symbolics:** branch-safe `Arg` folds and conservative symbolic complex primitives.
- **Unified experimental evaluation API.**

### Special functions / solver

- **Special-function foundation:** Lambert W, digamma, Bessel J₀/J₁ primitives.
- **Lambert W / trig transcendental solve:** `solve` recognises `α·u·e^u = c` (affine `u`) via principal `W₀`, and `sin`/`cos`/`tan` of an affine argument equal to a constant (principal inverse only — no `2πk` family). Thin experimental constructor: `alkahest.experimental.lambert_w`.
- **Transcendental solve** for exp/log equations.

### Simplification

- **Trig normal form (`simplify_trig_normal_form`):** opt-in fixed-point simplifier for sin/cos polynomials (DCM `Rᵀ·R − I` → `0` in one call).
- **Sound assumptions:** conditional rewrites require explicit assumptions.

### Integration / Risch

- Genus-0 √quadratic (including arcsin / negative leading coeff), Weierstrass `t=tan(x/2)`, trig powers & products, inverse-trig / reciprocal-trig / inverse-hyperbolic antiderivatives, Coates genus≥2 hyperelliptic logs, and exact vs numeric verification status.

### Linear algebra / ODE / real

- **Matrix:** symbolic eigenvalues / `matrix_exp`; `Matrix.rref` on the agent surface.
- **ODE:** numeric RK4/RK45 integrator and `dsolve` Python binding.
- **Parametric Routh–Hurwitz (`routh_hurwitz`).**

### API / agents

- **`capabilities()` / feature parity reporting** and agent capability / verification contract metadata.

## 3.5.1 — 2026-06-15

### Integration / Risch

- **Exact elliptic-integral constants:** genus-1 elliptic antiderivatives now print their reduction constants as exact algebraic numbers (`√3`, `3^(-1/4)`, `(2+√3)/4`, `12^(-1/4)`, `2√3-2√2`, …) instead of `2^53`-denominator float reconstructions. `∫dx/√(x³+1)` → `3^(-1/4)·EllipticF(acos((√3-(x+1))/(x+1+√3)), 1/2+√3/4)`.
- **No-real-root quartic normalization:** the `atan` substitution's Möbius coefficients are normalized so they reduce to simple `a+b√n` forms (e.g. `∫dx/√(x⁴+1)`).
- **Region-aware soundness gate:** the elliptic verification gate samples each `P > 0` interval (derived from `P`'s real roots), so correct reductions whose valid region is narrow or shifted no longer spuriously decline (e.g. `∫dx/√(x³-7x-6)`, region `x ≥ 3`).

## 3.5.0 — 2026-06-12

### Kernel

- **Imaginary unit:** canonical `I = √(−1)` as a kernel-blessed `Complex` symbol (`ExprPool::imaginary_unit()`); `i^n` power cycling and `Mul` collapse via `i² = −1`.

### Transforms

- **Fourier / Laplace / Z-transform:** symbolic forward and inverse transforms.
- **Fourier:** shifted Gaussian `F{e^{−a(x−b)²}}` with explicit phase factor via completing the square.
- **Z-transform inverse:** irreducible quadratic denominators (complex-conjugate poles) → real damped sinusoids.

### Calculus

- **Formal power series:** lazy FPS ring over ℚ with analytic operations.
- **Multivariate limits:** path-certificate non-existence.
- **Asymptotic expansions** at infinity.

### ODE

- **Classical `dsolve`:** first-order classes, linear constant-coefficient, and Euler–Cauchy.
- **Series solutions:** power-series and Frobenius methods for linear ODEs.

### Python

- **Experimental surface** (`alkahest.experimental`) for calculus, ODE, and transform APIs.

### Integration / Risch

- **Elementary products:** `x·exp(a·x)` (and related cases).
- **K-rational Hermite** reduction in `k_rational_integrate`.

### Poly

- **Puiseux tower continuation** with additive API (semver-safe re-land).

### Lean certificates

- **Differentiation:** `to_lean` / `DerivedResult.certificate` on `diff` results now emit `deriv (fun x => …) x = …` goals with Mathlib derivative lemmas instead of incorrect rewrite equalities (e.g. `x³ = 3x²`).

### Demo playground

- **Outputs:** render cell results as markdown; copy cell with output.
- **Lean certificate** cell in the default notebook.
- **Server kernel:** isolated `alkahest-playground` kernelspec in the server venv; matplotlib inline + figure flush; route matplotlib/numpy/playground_helpers cells to the server.
- **Lean verify:** legacy diff certificate shim in `playground_helpers` for older wheels; `start.sh` builds local alkahest via `maturin develop` when developing in-repo.

### Fixes

- **JIT:** cover all numeric primitives in `eval_interp` (+ registry sync test).
- **simplify:** fold elementary constants, trivial powers, and rational canonicalization.
- **lean:** emit `deriv` goals for diff certificates.

## 3.4.0 — 2026-06-10

### Calculus / integration (Risch roadmap)

- **M4 algebraic tower:** `AlgExtension` as a `DifferentialField`; algebraic top-generator dispatch via radical-over-exp substitution; coupled `coupled_radical_rde` over exp/log tower bases; K-rational integration with K-log emission; certify `NonElementary` for entangled K-log coefficients.
- **Non-diagonal f Risch DE:** generalize coupled algebraic Risch DE to f ∈ ℚ(x)(α); ∫R·exp(β) with β algebraic; non-diagonal f for `RadicalExt` over ℚ(x); polymorphic RDE degree bounds (Bronstein §6.5).
- **Algebraic singular places:** van Hoeij enlargement; Newton–Puiseux expansion at algebraic base points.
- **Genus-1 elliptic:** diagnose and decline-stability for remaining genus-1 elliptic configs; M3 capstone tests.
- **Integration utilities:** partial fractions (`apart`) and definite integration via FTC; non-linear u-substitution (derivative-divides heuristic).

### Demo playground

- Clear notebook control and calculus starter demo.

### Fixes

- **simplify:** correct e-graph integer `Pow` constant folding.
- **poly:** accept integer-valued `Rational` nodes in `RationalFunction::from_symbolic`.

## 3.3.0 — 2026-06-08

### Calculus / integration (Risch roadmap)

- **M4 tower recursion:** `DifferentialField` trait with ℚ(x)/exp/log implementations; multi-generator recursive integrator (exp × radical-over-tower); radical extension as a generic `DifferentialField` with tower-recursive `rational_rde`.
- **Elliptic integral output:** `EllipticF` / `EllipticE` / `EllipticPi` / `EllipticK` primitives; first-, second-, and third-kind elliptic output for genus-1 ∫dx/√(cubic|quartic) and ∫R/√(cubic|quartic); all-complex-root genus-1 quartics (∫dx/√(x⁴+1)); cosφ-config third-kind output.
- **Genus-1 capstone:** wire quartic y²=quartic and cubic cases into the public engine; genus-1 quartic without a rational root (Nagell); genus-0 Euler substitution for ∫R(x,√quadratic)dx; Miller log-argument construction; Abel–Jacobi in FIND-ORDER.
- **Algebraic extensions:** tower algebraic base, conjugate reduction, non-Galois quartic, general quadratic; algebraic residues and ramified places; lazy Hermite; Trager Q-basis and algebraic places; FIND-ORDER for non-branch and algebraic places; genus-2 compositum and end-to-end path.

### Reinforcement learning

- Hub package import fixes and CI metadata for symbolic integration; Environments Hub install path updated to `alkahest` org.

## 3.2.0 — 2026-06-05

### Reinforcement learning

- **`alkahest.rl`:** framework-agnostic core (`BaseGenerator`, `BaseVerifier`, `Rubric`, `CurriculumScheduler`) and a symbolic integration environment (`alkahest.rl.envs.integration`) with Risch-tier task grammar, layered `IntegrationVerifier`, and Prime Intellect `verifiers` entry point (`load_environment`).
- Optional pip extra: `pip install "alkahest[rl]"` (Python ≥ 3.10; pulls `verifiers` + `datasets`).
- veRL recipe: `recipes/verl_integration_reward.py`.
- Environments Hub manifest: `python/alkahest/rl/envs/integration/`.

### Calculus / integration (Risch roadmap)

- Algebraic Risch extensions: tower field integration, simple radicals, coupled algebraic RDE, genus-0 reduction and parametrization.
- Genus-1 stack (in progress): integral basis (van Hoeij), Hermite on curve, residue divisor, FIND-ORDER, elliptic engine.
- Newton–Puiseux fractional-power expansions; algebraic-coefficient Puiseux.

### Linear algebra

- Expanded matrix coverage (`alkahest-core/src/matrix/linear_algebra.rs`); Python bindings and tests.

## Unreleased (historical notes)

### Breaking / default-feature change

- **`groebner` is now a default Cargo feature in `alkahest-cas`**, matching the Python wheel defaults. `alkahest-cas = "2"` now includes Gröbner-backed APIs (`solve`, `diophantine`, homotopy) without explicitly listing the feature. To opt out: `alkahest-cas = { version = "2", default-features = false }`.

## Unreleased (2.2.x)

### Calculus

- **Transcendental Risch integration (issue #4):** Implements the complete Risch decision procedure for elementary antiderivatives over the transcendental differential field tower K = ℚ(x)(t₁,…,tₙ) with tᵢ = exp(ηᵢ) or log(hᵢ). Modules: `risch/poly_rde.rs` (polynomial Risch DE solver over ℚ[x]), `risch/tower.rs` (generator detection and tower decomposition), `risch/exp_case.rs` (hyperexponential case via RDE), `risch/log_case.rs` (hyperlogarithmic case via IBP recursion), `risch/mod.rs` (router and detection predicate). The engine checks `contains_risch_form` before the rule-based fallback. **Non-elementary certification:** when the polynomial RDE y' + k·Dη·y = h has no polynomial solution, the integrand is certified non-elementary (`IntegrationError::NonElementary`, error code `E-INT-004`). **Elementary cases covered:** p(x)·exp(g(x)) for any polynomial p and any degree, log(x)ⁿ for any n, p(x)·log(x)ⁿ via IBP recursion. Derivation log records `risch_exp_rde` and `risch_exp` / `risch_log` steps. 24 Python tests in `tests/test_risch_integration.py` (4 non-elementary, 13 exp-tower, 7 log-tower). References: Risch (1969), *Trans. AMS* 139; Bronstein (2005), *Symbolic Integration I*, Ch. 5–7.

### Infrastructure (JIT and evaluation)

- **Cranelift Tier-1 JIT** (`--features cranelift`): pure-Rust backend in `jit/cranelift_backend.rs`; usage-based tier selection via `CompileConfig` (interp → Cranelift → LLVM).
- **`CompileCache`**: memoize `ExprId + inputs → Arc<CompiledFn>`; Python `CompileCache` class with hit/miss stats.
- **Bulk JIT evaluation**: native `alkahest_eval_bulk` in Cranelift/LLVM backends; `CompiledFn::call_bulk` / `call_batch` column-major batch path.
- **Parallel batch evaluation**: `CompiledFn::call_batch_par`, `numpy_eval_par` (Rayon, `--features parallel`, GIL released).
- **DAG traversal memo tables**: per-call `HashMap<ExprId, T>` on simplify, diff, forward diff, integrate `is_free_of`, and JIT interpreter paths.
- **SIMD Horner f64 eval**: `eval_horner_f64` / `eval_horner_f64_batch` (4-wide `wide::f64x4`) on the interpreter numeric path.

### Infrastructure (simplification and FFI)

- **Colored e-graphs**: native layered union-find (`simplify/colored_egraph.rs`); `SimplifyConfig::assumptions` wired through `simplify_with`.
- **Match-disjoint egglog schedule**: shrink/explore rules split by LHS root symbol; `EgraphConfig::disjoint_schedule` (default `true`).
- **Discrimination-net pattern indexing**: `DiscriminationIndex` / `PatternRuleSet` for user `PatternRule` sets (`simplify_with_pattern_rules`; Rust API).
- **FLINT drop-safe wrappers**: RAII `Drop` on all FLINT factor types; `FlintMPolyCtx` ref-counted via `Arc`.
- **Vendored egglog v0.4.0** (`vendor/egglog`): default PyPI wheels now ship with `egraph` feature.

### Tooling and CI

- **CodSpeed** continuous benchmarking (Rust + Python).
- **uv / ruff / ty** integrated for Python dev workflow (`pyproject.toml` dependency groups).

## 2.0.4 — 2026-05-22

### Polynomial algorithms

- **V2-3 — Sparse multivariate interpolation (Ben-Or/Tiwari, Zippel):** Rust `alkahest_core::poly::interp` — `sparse_interpolate_univariate(eval, T, p)` recovers a sparse univariate `f ∈ Fₚ[x]` from exactly `2T` evaluations via Berlekamp–Massey + Cantor–Zassenhaus root-finding + BSGS discrete-log + Vandermonde solve; `sparse_interpolate(eval, vars, T, D, p, seed)` recovers a sparse multivariate polynomial via Zippel's variable-by-variable algorithm with batched Vandermonde lifting. Supporting infrastructure: `MultiPolyFp` (sparse polynomial over `Fₚ`), `reduce_mod`, `lift_crt`, `rational_reconstruction`, `mignotte_bound`, `select_lucky_prime`. Python: `sparse_interp_univariate`, `sparse_interp`, `SparseInterpError`, `MultiPolyFp`, `modular` submodule. ROADMAP acceptance criteria: 10-variable 15-term polynomial recovered at ≥ 90% success over 20 random seeds (`test_roadmap_10var_15term`). Tests: Rust `poly::interp`, Python `tests/test_sparse_interp.py` (18 fast + 1 slow).

- **Sparse modular GCD (`gcd_sparse_modular` / `gcd_sparse`) — substrate for faster modular algorithms:** Rust `alkahest_core::poly::interp::gcd_sparse_modular` — Zippel evaluation–interpolation GCD over ℤ[x₁,…,xₙ]; for each lucky prime `p`: probes the GCD degree in `x₁` via one specialization, then for each `x₁^k` degree runs `sparse_interpolate` to recover the coefficient polynomial `c_k(x₂,…,xₙ)`, assembles the modular GCD image, and repeats until the CRT product exceeds the Mignotte bound; CRT lifting via `lift_crt`; result normalised to primitive part with positive leading coefficient. `SparseGcdError` (`E-INTERP-010…012`). Python: `gcd_sparse`, `SparseGcdError`. Rust unit tests: `gcd_sparse_univariate_linear_factor`, `gcd_sparse_univariate_coprime`, `gcd_sparse_bivariate_common_factor`. Python integration tests in `tests/test_sparse_interp.py::TestSparseGcd` (activated after wheel rebuild).

## 2.0.3 — 2026-05-21

### Calculus

- **Full Gruntz limits:** Rust `alkahest_core::calculus::gruntz` — Gruntz (1996) MRV comparability-graph algorithm for limits of exp-log combinations as var → +∞. Steps: collect diverging `exp(h)` subexpressions, build comparability ordering via limit ratios, extract the maximally-ranked (MRV) set, pick ω → 0⁺, rewrite as Laurent series in ω, and read off the limit from the leading power. Thread-local depth counter (max 8) prevents unbounded re-entry. Gruntz is invoked from `limit_inner` before the 1/t substitution so exp structure is visible; existing L'Hôpital and series fallback paths are preserved. 6 new tests in `tests/test_gruntz_v217.py`; Rust unit tests in `gruntz.rs`.

### Advanced polynomial solvers

- **Polyhedral / mixed-volume homotopy:** Rust `alkahest_core::solver::polyhedral` — Newton polytopes, Graham-scan convex hull, Shoelace mixed-volume for n=2; binomial start system per mixed cell via complex log branch enumeration; `polyhedral_cell_iter` yields `(GbPoly start system, start points)` per cell. `solve_numerical` auto-selects polyhedral start when MV < Bézout bound; new Euler–Newton tracker `track_path_sys`. `PolyhedralError` (`E-POLYHEDRAL-*`). Python tests in `tests/test_polyhedral_v217.py`.

- **F5 signature-based Gröbner basis:** Updated `alkahest-core/src/poly/groebner/f5.rs` — corrected signature comparison, S-polynomial formation, and reduction bookkeeping; new Criterion benchmark group `groebner_f5` in `benches/alkahest_bench.rs`.

### Lean 4

- **`Filter.Tendsto` certificate export:** `alkahest_core::lean::emit_tendsto_cert(expr, var, lim, pool)` generates a Lean 4 snippet with the appropriate `Filter.Tendsto` statement; pattern-dispatches to Mathlib theorems (`tendsto_exp_neg_atTop_nhds_zero`, `tendsto_exp_atTop`, etc.) and falls back to `by sorry` for unsupported cases. Codomain filter is `nhds L` for finite limits and `Filter.atTop` for +∞. `emit_limit_header()` emits the required Mathlib imports.

### Demo playground

- **Lean certificate panel:** `LeanCertificate.tsx` renders `Filter.Tendsto` proofs inline in notebook output cells with syntax highlighting and a copy button.
- **F5 verification in notebook:** `demo-playground/server/lean_verify.py` — server-side Lean 4 subprocess verification; `output_parse.py` and `playground_helpers.py` added for structured kernel output; agent chat gains awareness of Lean verification results.

### Packaging

- **Crate renamed to `alkahest-cas`:** The published Rust crate is now `alkahest-cas` on crates.io (was `alkahest-core`). All internal references updated; README badge updated.

## 2.0.2 — 2026-05-17

### Packaging / releases

- Version **2.0.2** (workspace + `pyproject.toml`). Git tag **`v2.0.2`** for release CI (PyPI default wheels + **`+jit` / `+full`** on GitHub Releases). (`v2.01.0` / `2.01.0` is not a valid Cargo semver — leading zeros in numeric components.)

## 2.0.1 — 2026-05-16

### Packaging / releases

- Version **2.0.1** (workspace + `pyproject.toml`).
- **Release CI (`+full` wheels):** Linux `linux_x86_64` wheels with PEP 440 local version **`X.Y.Z+full`**, built with Cargo features `jit groebner parallel egraph`, attached to **GitHub Releases** next to existing **`+jit`** wheels. **`+jit`** and **`+full`** wheels are **not** uploaded to the main PyPI simple API (same policy as before for `+jit`) so `pip install alkahest` stays on the small default wheels.

## 2.0.0 — 2026-05-06

### Calculus and series

- **V2-15 — `series()` / Laurent expansions:** Rust `alkahest_core::calculus::series` — `series(expr, var, point, order)`, `Series`, `SeriesError` (`E-SERIES-*`); truncated Taylor expansions via differentiation and Laurent tails for univariate rationals with poles; kernel `ExprData::BigO` (`ExprPool::big_o`); pool file format **v3** (node tag 12). Python: `series`, `Series`, `SeriesError`, `ExprPool.big_o`; `_pretty` recognizes `big_o` nodes for Unicode/LaTeX-style printing of $\mathcal{O}(\cdots)$. Tests: Rust `calculus::series`, Python `tests/test_series_v215.py`.

- **V2-16 — `limit()` (prototype rules):** Rust `calculus::limits` — `limit`, `LimitDirection`, `LimitError` (`E-LIMIT-*`); finite points via 0/0 L’Hôpital, local Laurent/Taylor expansions (`local_expansion`), specials, and guarded direct substitution (`0/0`, `0·pole` rejection); limits at `±∞` via `x ↦ ±1/t` with nested rational power flattening and polynomial quotient normalization before `t → 0⁺`; `ExprPool::pos_infinity()` (`∞` symbol). Python: `limit`, `LimitError`, `ExprPool.pos_infinity`. Limitations: not full Gruntz; oscillatory or unconstrained transcendental tails may return `Unsupported`. Tests: Rust `calculus::limits::tests`, Python `tests/test_limits_v216.py`.

- **Algebraic-function Risch integration (Trager):** `alkahest-core/src/integrate/algebraic/` — genus-0 integrals involving `sqrt(P(x))` over ℚ(x) for P of degree 0/1/2 (J₀ formula + substitution); `NonElementary` guard for deg P ≥ 3; mixed integrands `A(x) + B(x)·sqrt(P(x))` via field decomposition. 14 tests in `tests/test_algebraic_integration.py`; 10 worked examples in `examples/risch_integration.py`.

### Discrete mathematics

- **V2-10 — Symbolic summation (Gosper / Zeilberger):** Rust `alkahest_core::sum` — `sum_indefinite(term, k)`, `sum_definite(term, k, lo, hi)` for terms with rational shift ratio (polynomials × `gamma` of a linear expression in `k`); `solve_linear_recurrence_homogeneous` for constant-coefficient homogeneous recurrences; `verify_wz_pair(F, G, n, k)` for checking discrete telescoping certificates. `SumError` (`E-SUM-*`). Python: `sum_indefinite`, `sum_definite`, `solve_linear_recurrence_homogeneous`, `verify_wz_pair`, `SumError`. Tests: Rust `sum::tests`, Python `tests/test_sum_v210.py`.

- **V2-18 — Difference equations (`rsolve`):** Rust `alkahest_core::sum::rsolve` — linear recurrences with constant coefficients and polynomial right-hand side in the recurrence index; `rsolve(eq, n, fn_name, initials)` returns a closed-form `DerivedResult`; `RsolveError` (`E-RSOLVE-*`). Python: `rsolve`, `RsolveError`. Limitations: non-homogeneous order > 2 and polynomial-coefficient recurrences not implemented. Tests: `tests/test_rsolve.py`, Rust `sum::rsolve`.

- **V2-22 — Symbolic discrete products (`∏`):** Rust `alkahest_core::sum::product` — `product_definite` / `product_indefinite` for terms that are rational in the index variable with numerator and denominator polynomials that factor into ℤ-linear terms (Γ-ratio telescoping + leading powers); `ProductError` (`E-PROD-*`). Stable re-exports in `alkahest_core::stable`. Python: `product_definite`, `product_indefinite`, `Product` (SymPy-shaped `Product(term, (k, lo, hi))`), `ProductError`; `examples/products.py`; tests Rust `sum::product`, Python `tests/test_product_v222.py`.

### Algebra and number theory

- **V2-17 — Matrix eigenvalues / eigenvectors / diagonalize:** Rust `alkahest_core::matrix::eigen` — `characteristic_polynomial_lambda_minus_m`, `eigenvalues`, `eigenvectors`, `diagonalize`, `EigenError` (`E-EIGEN-*`); splits `det(λI−M)` via FLINT ℤ factorization after clearing rational denominators in the coefficients of χ; linear and quadratic characteristic factors; rotation `[[0,-1],[1,0]]` diagonalizes over ℚ(i). Python: `Matrix.characteristic_polynomial_lambda_minus_m`, `eigenvals`, `eigenvects`, `diagonalize`, `EigenError`. Limitations: defective matrices return `NonDiagonalizable`; irreducible χ factors of degree &gt; 2 are rejected. Tests: Rust `matrix::eigen`, Python `tests/test_eigen_v217.py`.

- **V3-1 — Integer number theory:** Rust `alkahest_core::number_theory` — FLINT-backed `isprime`, `factorint`, `nextprime`, `totient`, `jacobi_symbol`, `nthroot_mod` (prime modulus), `discrete_log` (moderate primes), `QuadraticDirichlet`; `NumberTheoryError` (`E-NT-*`); stable re-exports. Python: module `alkahest.number_theory` plus `DirichletChi` / `NumberTheoryError` from the native extension. Tests: Rust `number_theory::tests`, Python `tests/test_number_theory_v31.py`.

- **V2-19 — Diophantine equations (`diophantine`):** Rust `alkahest-core::solver::diophantine` — two integer unknowns; linear parametric families (extended gcd); `x² + y² = n` (enumeration); unit Pell `x² - D y² = 1` (fundamental `(x₀,y₀)` via continued-fraction convergents); `DiophantineError` (`E-DIOPH-*`). Python (`groebner`): `diophantine`, `DiophantineSolution`, `DiophantineError`. CI builds the wheel with `--features groebner`; `pytest.ini` sets `pythonpath = python`. Tests: Rust `solver::diophantine`, Python `tests/test_diophantine_v219.py`.

- **V3-2 — Non-commutative algebra:** `ExprData::Symbol` carries `commutative: bool` (default `true`). `ExprPool::mul` and `canonical_order` skip sorting when any factor subtree contains `commutative: false`; `collect_mul_factors` merges powers **globally** only for fully commutative products and **adjacent** identical bases otherwise. E-graph simplification falls back to the rule engine when a non-commutative symbol appears. `alkahest_core::algebra::noncommutative` — Pauli table (`sx`/`sy`/`sz`) and orthogonal Clifford snippet (`cliff_e1`/`cliff_e2`); `NoncommutativeCost` (e-graph tie-break). Pool file format **v4** adds `commutative` on symbol nodes. Python: `ExprPool.symbol(..., commutative=False)`, `simplify_pauli`, `simplify_clifford_orthogonal`; `examples/noncommutative.py`; `tests/test_noncommutative_v32.py`.

### Advanced polynomial solvers

- **V2-11 — Regular chains / triangular decomposition:** Rust `triangularize`, `RegularChain`, `extract_regular_chain_from_basis`, `main_variable_recursive` (`alkahest_core::solver::regular_chains`); optional bottom-univariate factor splitting via V2-7; `solve_polynomial_system` fallback backsolve from an extracted chain after a lex-basis stall. Python: `triangularize`, `RegularChain`; benchmark task `solve_6r_ik` (planar IK proxy). Tests: `tests/test_regular_chains_v211.py`, Rust `solver::regular_chains`.

- **V2-12 — Primary decomposition:** Rust `primary_decomposition`, `radical`, `PrimaryComponent`, `PrimaryDecompositionError` (`alkahest_core::ideal::primary`); partial GTZ-style splitting (saturations + Lex univariate factorization). Python: `primary_decomposition`, `radical`, `PrimaryComponent`; tests: `tests/test_primary_decomposition_v212.py`, Rust `ideal::primary`.

- **V2-13 — Differential algebra / Rosenfeld–Gröbner:** Rust `rosenfeld_groebner`, `rosenfeld_groebner_with_options`, `dae_index_reduce`, `DifferentialRing` / `DifferentialIdeal` / `RegularDifferentialChain`, `DiffAlgError` (`alkahest_core::diffalg`); Python (`groebner`): `rosenfeld_groebner`, `dae_index_reduce`, `RosenfeldGroebnerResult`, `DaeIndexReduction`. Tests: `tests/test_diffalg_v213.py`, Rust `diffalg::tests`.

- **V2-14 — Numerical algebraic geometry:** Total-degree homotopy continuation in `ℂⁿ` with predictor–corrector tracking, Newton polish, conservative Smale heuristic, and `ArbBall` enclosures (`alkahest_core::solver::homotopy`); `solve_numerical`, `HomotopyOpts`, `CertifiedPoint`, `HomotopyError` (`E-HOMOTOPY-*`). Python (groebner): `solve(..., method="homotopy")`, `solve_numerical`, `CertifiedSolution`, benchmark task `numerical_homotopy`. Limitation: deficient systems (fewer roots than the Bézout bound) need a polyhedral start — not included. Tests: `tests/test_homotopy_v214.py`, Rust `solver::homotopy`.

### Developer experience

- **LaTeX / Unicode pretty-printing:** Pure-Python tree walk; `latex(expr)` emits `\sin\!\left(x\right)`, `\frac`, `\sqrt`, `\mathcal{O}` etc.; `unicode_str(expr)` emits `sin(x)² + cos(x)²` style. `Expr.node()` kernel hook for tree introspection. Exported from `alkahest.__all__`. 74 tests.

- **String expression parsing (`parse`):** Pratt recursive-descent parser in `python/alkahest/_parse.py`; `parse(source, pool, symbols=None) -> Expr`; supports integer/float literals, all 23 registered primitives, `^` / `**`, unary `-`, parentheses; `ParseError` (`E-PARSE-001`) with byte-level `.span`. 54 tests in `tests/test_parse.py`.

- **E-graph default rule completeness:** `simplify_egraph` now loads trig (`sin²+cos²→1`) and log/exp (`exp(log x)→x`) rules by default; opt-out via `EgraphConfig(include_trig_rules=False, include_log_exp_rules=False)`; `simplify_egraph_with(expr, config)` Python API.

- **Python API completeness:** `ExprPool.save_to(path)` / `load_from(path)` PyO3 bindings; `GroebnerBasis.compute(polys, vars)` static method; `solve()` returns `dict[Expr, Expr]` by default (`numeric=True` for float output); `IoError` exported from `alkahest`.

- **Windows + macOS CI parity:** `ci-cross.yml` matrix — `macos-14` (parallel + egraph + jit, FLINT via Homebrew) and `windows-2022` GNU (parallel + egraph, FLINT via MSYS2). `build.rs` Windows link-search branch added. Known limitation: `jit` excluded on Windows (inkwell pins LLVM 15; MSYS2 ships 17+).

## 1.0.0

### Features

- Integer Hermite / Smith normal forms (`IntegerMatrix`, FLINT HNF + pure-Rust SNF) and polynomial-matrix HNF/Smith over ℚ\[x\] (`RatUniPoly`, `PolyMatrixQ`); stable re-exports in `alkahest_core::stable`
- Exact LLL lattice reduction over ℤ (`alkahest_core::lattice::lattice_reduce_rows`; optional Lovász `δ`), plus an augmented-lattice + LLL heuristic for approximate integer relations (`guess_integer_relation` / Python `guess_relation` — **not** the Ferguson–Bailey PSLQ iteration); exposes `LatticeError` (`E-LAT-*`) and `PslqError` (`E-PSLQ-*`)
- Production NVPTX codegen for `sm_86` (Ampere): full inkwell-driven lowering, `libdevice.10.bc` linking, PTX emission via LLVM target machine, `cudarc 0.19` runtime — 16.2× speedup over CPU JIT on RTX 3090
- Gröbner-based polynomial system solver (`alkahest.solve`): Lex basis → triangular back-substitution → exact symbolic solutions including irrational roots (`sqrt(2)/2`)
- **V2-7 — Polynomial factorization:** FLINT-backed `fmpz_poly_factor` for ℤ\[x\] (Zassenhaus + van Hoeij), `fmpz_mpoly_factor` for multivariate ℤ, and `nmod_poly_factor` for 𝔽_p\[x\]; Rust `factor_univariate_z`, `factor_multivariate_z`, `factor_univariate_mod_p` + Python `UniPoly.factor_z`, `MultiPoly.factor_z`, `factor_univariate_mod_p`; `FactorError` (`E-POLY-008…010`)
- Custom `alkahest` MLIR dialect: `Sym`, `Const`, `Add`, `Mul`, `Pow`, `Call`, `Horner`, `PolyEval`, `SeriesTaylor`, `IntervalEval`, `RationalFn` ops; three lowering targets (ArithMath, StableHLO, LLVM); 1000-case round-trip proptest
- CUDA Macaulay-matrix row reduction (`--features groebner-cuda`): PTX elimination kernel, multi-prime CRT rational reconstruction, CPU fallback when no CUDA device present
- Semver-stable 1.0 API: `alkahest_core::stable` / `alkahest_core::experimental` split; `alkahest.__all__` freeze; `cargo semver-checks` + `scripts/check_api_freeze.py` in CI
- Primitive registry expanded to 23 primitives: added `tan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `erf`, `erfc`, `abs`, `sign`, `floor`, `ceil`, `round`, `atan2`, `gamma`, `min`, `max`
- Cross-CAS benchmark driver: Mathematica WolframEngine 14.3 and SymEngine 0.14 adapters; all six benchmark tasks implemented; nightly `--competitors` CI; per-competitor ratio columns in HTML report
- Persistent `ExprPool`: `save_to`, `load_from`, `open_persistent`, `checkpoint`; versioned binary format (`ALKP` magic); atomic temp-rename + fsync crash safety; all `ExprData` variants including `Piecewise` and `Predicate`

### Internal

- Structured errors across all subsystems: `.code`, `.remediation`, `.span` on every `AlkahestError` variant; `CudaError` (`E-CUDA-001…004`) and `SolverError` (`E-SOLVE-001…003`) added; PyO3 exception classes with typed attributes

## 0.5.0

### Features

- Lean 4 certificate exporter: pure-Rust, no FFI; 20+ rule→tactic mappings (`norm_num`, `simp`, `ring`, `rw`); `emit_lean_expr`, `emit_step`, `emit_goal`
- StableHLO / XLA bridge: pure-text MLIR emitter for `Add`, `Mul`, `Pow`, `sin`, `cos`, `exp`, `log`, `sqrt` → `stablehlo.*` ops via `to_stablehlo`
- Expanded Risch integration: exp/log tower + linear substitution; `∫ log(x) dx`, `∫ exp(a·x+b) dx`, `∫ c·x·exp(x) dx`, `∫ 1/(a·x+b) dx`; `is_linear_in` helper
- Branch-cut-aware log/exp simplification: `LogOfProduct` records `SideCondition::Positive` per factor; `SimplifyConfig::allow_branch_cut_rewrites`; `log_exp_rules_safe()` excludes `LogOfProduct`
- JAX primitive source integration: `to_jax` registers a symbolic expression with `def_impl`, `def_abstract_eval`, JVP rule (via symbolic grad), and vmap batching; graceful no-JAX fallback
- Parallel F4 Gröbner basis: Buchberger + product-criterion pruning; Rayon parallel S-poly reduction; `interreduce`; `Lex`/`GrLex`/`GRevLex` orders (`--features groebner`)

### Internal

- Structured errors MVP: `remediation()` and `span()` on `ConversionError` and `IntegrationError`
- Lean CI: GitHub Actions workflow generates 8 proof files and verifies via `lean` compiler; Mathlib build cached
- CUDA compute-sanitizer nightly: `memcheck` + `racecheck` on self-hosted `gpu-3090` runner; sanitizer logs uploaded as artifacts
- GPU benchmark suite: `GPUPolynomialEval` (1M-pt, 5-var), `GPUJacobian` (65k-pt), `DLPackZeroCopy`; `--gpu` flag added to `cas_comparison.py`

## 0.4.0

### Features

- Horner-form code emission: `horner(expr, var)`, `emit_c(expr, var, var_name, fn_name)`
- NumPy / JAX batch evaluation: `CompiledFn.call_batch_raw`, `numpy_eval` accepting NumPy, PyTorch, and JAX arrays via DLPack
- `collect_like_terms`: `2*x + 3*x → 5*x`
- `poly_normal`: polynomial normal form over given variables
- FLINT 3.x feature gate (`--features flint3`)
- Sharded `ExprPool`: concurrent insertion via `DashMap` (`--features parallel`)

### Internal

- GitHub Actions CI: tier-1 PR checks (< 10 min) + nightly integration (4–8 h) with AFL++ fuzzing, deep proptest, Valgrind, and SymPy oracle

## 0.3.0

### Features

- Reverse-mode automatic differentiation (`symbolic_grad`)
- Symbolic matrices and Jacobian (`Matrix`, `jacobian`)
- ODE representation and first-order lowering (`ODE`, `lower_to_first_order`)
- DAE structural analysis and Pantelides index reduction (`DAE`, `pantelides`)
- Acausal component modeling (`AcausalSystem`, `Port`, `resistor`)
- Sensitivity analysis: forward (`sensitivity_system`) and adjoint (`adjoint_system`)
- Hybrid system event handling (`HybridODE`, `Event`)
- LLVM JIT compiled evaluation (`compile_expr`, `CompiledFn`, `eval_expr`; `--features jit`, LLVM 15)
- Ball arithmetic (`ArbBall`, `AcbBall`, `interval_eval`) backed by Arb/FLINT
- Parallel simplification (`simplify_par`; `--features parallel`)
- Multivariate polynomial GCD via FLINT (`MultiPoly::gcd`, `RationalFunction::new`)

### Internal

- SymPy oracle cross-validation test suite for `integrate`
- E-graph vs rule-based Criterion benchmark (`bench_simplifier_comparison`)
- Rule engine hardening: trig/log rule sets, pattern rules, substitution, CI, AFL++ fuzzing

## 0.2.0

### Features

- E-graph equality saturation via egglog (`simplify_egraph`, `--features egraph`)
- Associative-commutative pattern matcher
- Forward-mode automatic differentiation (`diff_forward`)
- Rule-based integration: Risch subset (power, trig, exp/log table entries)
- `RationalFunction` arithmetic with multivariate GCD normalization

### Internal

- Pluggable e-graph cost functions: `SizeCost`, `DepthCost`, `OpCost`, `StabilityCost`; phased saturation via `node_limit` / `iter_limit`
- `PrimitiveRegistry` with `Capabilities` bitflags and `coverage_report()`; sin/cos/exp/log/sqrt registered
- `TracedFn`, `trace`, `grad`, `jit`, `trace_fn` Python transformation façade
- DLPack + `__array__` protocol on compiled functions
- `Piecewise` / `Predicate` expression nodes; diff/simplify/pattern/poly updated
- JAX-style pytree support (`flatten_exprs`, `unflatten_exprs`, `map_exprs`, `TreeDef`)
- `alkahest.context(pool=..., domain=..., simplify=True)` context manager
- Flat n-ary egglog: binary output flattened back to n-ary `Add`/`Mul` on extraction
- `canonicalize_linear` post-extraction pass
- Cross-CAS benchmark driver: HTML/JSONL report, Criterion dashboard

## 0.1.0

### Features

- Hash-consed expression DAG (`ExprPool`, `ExprId`): structural equality as pointer comparison, automatic subexpression sharing
- N-ary `Add` / `Mul` with AC normalization at construction
- Arbitrary-precision integers and rationals (FLINT/GMP)
- Symbol domains: `real`, `positive`, `nonnegative`, `integer`, `complex`
- Rule-based simplification with fixpoint iteration: identity elements, constant folding, polynomial normalization
- Symbolic differentiation with chain/product/quotient rules (`diff`)
- `UniPoly`: dense univariate polynomial backed by FLINT; GCD, degree, coefficients, arithmetic
- `MultiPoly`: sparse multivariate polynomial over ℤ
- `RationalFunction`: quotient with automatic GCD normalization
- PyO3 bindings for the full core API
- Derivation logs: ordered `RewriteStep` list on every `DerivedResult`
