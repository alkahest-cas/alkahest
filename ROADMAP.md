# Alkahest Road Map

## Status

| Milestone | Status | Highlights |
|---|---|---|
| **v0.1 — Foundations** | ✅ Complete | Hash-consed kernel, rule-based simplifier, differentiation, `UniPoly`/`MultiPoly`/`RationalFunction`, PyO3 bindings, derivation logs |
| **v0.2 — Transformations** | ✅ Complete | E-graph backend (`egglog`), AC pattern matcher, forward-mode AD, rule-based integration (Risch subset), `RationalFunction` arithmetic with multivariate GCD |
| **v0.3 — Modeling + Infrastructure** | ✅ Complete | Reverse-mode AD, symbolic matrices & Jacobian, ODE/DAE analysis, acausal modelling, sensitivity analysis, hybrid systems, LLVM JIT, ball arithmetic, parallel simplification |
| **v0.4 — Numerics & Code Emission** | ✅ Complete | Horner-form emission, NumPy batch eval, `collect_like_terms`, `poly_normal`, GitHub Actions CI, sharded `ExprPool`, FLINT3 feature gate |
| **Pre-v0.5 cleanup** | ✅ Complete | CI green (LLVM 15), primitive registry, `cas.trace`/`grad`/`jit` façade, DLPack, `Piecewise`/`Predicate`, pluggable e-graph cost, PyTree, cross-CAS benchmark driver, flat n-ary egglog, linear canonizer, `StabilityCost`, Criterion dashboard, `alkahest.context(...)` |
| **v0.5 — Verified + GPU** | ✅ Complete | Lean 4 certificate export, StableHLO/XLA bridge, NVPTX JIT stub (sm_86 PTX), expanded Risch (log/exp tower + linear substitution), parallel F4 Gröbner basis, branch-cut-aware simplification, structured errors with remediation hints, JAX primitive integration, Lean CI + CUDA sanitizer CI, GPU benchmark suite |
| **v1.0 — Production** | ✅ Complete | Production NVPTX codegen (16.2× over CPU JIT), structured errors, Gröbner-based solver with symbolic √ solutions, custom MLIR dialect, CUDA Macaulay row reduction, semver API, 23-primitive registry, cross-CAS benchmarks, persistent `ExprPool` |
| **v1.1 — Post-launch** | 📋 Planned | Algebraic-function Risch (Trager), AMD ROCm codegen (hardware-blocked), PyPI wheel publishing, Win+macOS CI parity, docs site, LaTeX/Unicode printing, string parsing |
| **v2.0 — Mathematical Coverage** | 📋 Planned (15 items) | Modular/CRT framework, resultants, sparse interpolation, real root isolation, HNF/SNF, LLL+PSLQ, polynomial factorization, F5 signature-based Gröbner, logic/FOFormula (CAD prerequisite), CAD (real QE), Zeilberger/creative telescoping, regular chains, primary decomposition, differential algebra, homotopy continuation |
| **v2.1 — SymPy Parity** | 📋 Planned | Limits (V2-16), `rsolve` scope (V2-18); **symbolic `∏` (V2-22) ✅**; **integer number theory (V3-1) ✅**; **V2-19 `diophantine` ✅**; eigenpairs ✅ (V2-17); `series` / `parse` / LaTeX ✅ (V2-15, V2-20, V2-21) |
| **v3.0 — Domain Expansions** | ✅ Complete (V3-2) | Noncommutative algebra (kernel extension) |

**Test coverage:** 362 Rust unit/proptest/doctest cases + 438 Python tests (+ 52 skipped feature-gated/oracle tests) = 800 tests passing, zero errors.

---

## v1.1 — Post-launch

Items deferred from v1.0 due to algorithmic complexity or hardware availability, plus two gaps identified in a post-launch API audit.

| Item | Status | Notes |
|---|---|---|
| **V1-2** Algebraic-function Risch (Trager) | ✅ Complete | Genus-0 cases (P deg 0/1/2); NonElementary guard; 14-test suite; 10 worked examples |
| **V1-6** AMD ROCm codegen (`amdgcn`) | ⏸ Hardware-blocked | Requires RDNA3 / MI-series; design-only until hardware available |
| **V1-9** PyPI wheels (manylinux / macOS / Windows) | 🏗 Scaffolded | Publish gate lifts after two weeks of green v1.0 main |
| **V1-10** Windows + macOS CI parity | ✅ Complete | macOS-14 (parallel+egraph+jit) + Windows-2022 GNU (parallel+egraph); FLINT via brew/MSYS2 |
| **V1-11** Documentation site (Sphinx + mdBook) | 🏗 Scaffolded | Content written; CI build (`docs.yml`) pending deployment |
| **V1-15** E-graph default rule completeness | ✅ Complete | Trig (sin²+cos²→1, Pow form) and log/exp (exp(log)→x) load by default; `EgraphConfig.include_trig_rules/include_log_exp_rules` opt-out flags; `simplify_egraph_with` Python API |
| **V1-16** Python API completeness | ✅ Complete | `ExprPool.save_to/load_from`, `GroebnerBasis.compute()`, `solve()` returns `Expr` by default (`numeric=True` for floats) |
| **V2-20** LaTeX / Unicode pretty-printing | ✅ Complete | Pure-Python tree walk; `Expr.node()` kernel hook; `latex()` + `unicode_str()` API; 74 tests |
| **V2-21** String / expression parsing (`parse(str)`) | ✅ Complete | Pratt recursive-descent parser; `parse(str, pool)` public API; 54-test suite |

---

### V1-2. Algebraic-function Risch (complete V5-4) → ✅ Complete

**What:** Extend the Risch engine from the exp/log tower (shipped in V5-4) to
cover algebraic extensions — integrals of expressions involving
`sqrt(P(x))` over ℚ(x) for genus-0 curves.

**Delivered:**
- `alkahest-core/src/integrate/algebraic/` — decompose, genus_zero, poly_utils modules.
- Handles P of degree 0 (const), 1 (linear), 2 (quadratic) via the J₀ formula and substitution.
- `NonElementary` error guard for deg P ≥ 3 (elliptic/hyperelliptic integrals).
- `UnsupportedExtensionDegree` guard for non-square-root radicals (planned V2).
- Mixed integrands A(x) + B(x)·sqrt(P(x)) via field decomposition.
- 14 tests in `tests/test_algebraic_integration.py`; 10 worked examples in `examples/risch_integration.py`.

**Limitations (V2):** Multiple generators; higher-degree radicals (cbrt, nth-root); Trager's full algorithm for arbitrary algebraic extensions; oracle corpus coverage measurement.

---

### V1-6. AMD ROCm codegen (`amdgcn` target) → moved to v1.1 ⏸ Hardware-blocked

*Requires AMD RDNA3 (MI-series or 7900 XTX). Stays design-only until hardware is available.*

**Design:**
- `alkahest-core/src/jit/amdgcn.rs` behind new `rocm` Cargo feature.
- Same expression-walker as host + NVPTX paths; only the inkwell `TargetMachine` differs.
- HSA runtime via `hip-runtime-sys` for kernel launch.

**Acceptance:** `cargo test --features rocm` green; Criterion row `amdgcn_polynomial_1M` present.

---

### V1-9. PyPI release with manylinux / macOS / Windows wheels 🏗 Scaffolded

`.github/workflows/release.yml` lands the full wheel matrix (Linux manylinux 2_28, macOS universal2, Windows MSVC) × (Python 3.9 – 3.13). Publish step is gated (`if-disabled: true`) until V1-8 has two weeks of green main. Manual trigger via `workflow_dispatch` exercises the build + smoke-import path without publishing.

**Design:**
- `maturin build --release --strip --manylinux 2_28` for Linux wheels.
- macOS universal2 binary; cross-compiled aarch64 via GitHub's `macos-14` runner.
- Windows: MSVC toolchain; LLVM via `choco install llvm`.
- Feature flags: wheels ship with `parallel`, `jit`, `egraph` enabled by default; `cuda` shipped as opt-in wheel (`alkahest-cuda` namespace).

**Acceptance:** `pip install alkahest` green on PyPI; GitHub release auto-publishes wheels.

---

### V1-10. Windows + macOS CI → ✅ Complete

**Delivered:**
- `alkahest-core/build.rs` — added Windows branch: emits `cargo:rustc-link-search` for `C:/msys64/mingw64/lib` (overridable via `MSYS2_PREFIX`).
- `.github/workflows/ci-cross.yml` — two-cell matrix:
  - `macos-14` / `stable` / `--features "parallel egraph jit"` + Python tests (`maturin develop` + `pytest`).
  - `windows-2022` / `stable-x86_64-pc-windows-gnu` / `--features "parallel egraph"`; FLINT via MSYS2 `mingw-w64-x86_64-flint`.

**Known limitation:** `jit` excluded on Windows because `inkwell` pins LLVM 15 and MSYS2 ships LLVM 17+. Tracked for resolution when inkwell adds an LLVM 17 feature flag.

---

### V1-11. Documentation site 🏗 Scaffolded

`docs/mdbook/` holds the mdBook user guide (17 chapters, fully written); `docs/sphinx/` holds the Python API reference (11 sections, fully written); `.github/workflows/docs.yml` runs `mdbook build` + `sphinx-build -W` + `cargo doc --workspace`. Deployment to GitHub Pages is the remaining step.

**Acceptance:** Docs site live at `alkahest.github.io/alkahest/`; link from `README.md`.

---

### V1-15. E-graph default rule completeness

**What:** `simplify_egraph` currently misses two common identities: the Pythagorean trig identity (`sin²(x) + cos²(x) → 1`) and exp/log cancellation (`exp(log(x)) → x`). Users must call `simplify_trig` / `simplify_log_exp` separately.

**Design:**
- Add `trig_rules_default()` and `log_exp_rules_default()` (safe subset, excludes `LogOfProduct`) and load them in `simplify_egraph` by default.
- `SimplifyConfig` gains `include_trig_rules: bool` (default `true`) and `include_log_exp_rules: bool` (default `true`).
- `simplify_egraph_with(expr, config)` honours both flags.

**Test plan:**
1. `simplify_egraph(sin(x)**2 + cos(x)**2)` → `1`.
2. `simplify_egraph(exp(log(x)))` → `x`.
3. `simplify_egraph(log(exp(x)))` → `x`.
4. Existing egraph proptests (1 000-case) still pass.
5. Opt-out: `simplify_egraph_with(expr, SimplifyConfig { include_trig_rules: false, .. })` does not apply trig rules.

**Acceptance:** Both identities reduced by `simplify_egraph` with no extra configuration; proptests green.

---

### V1-16. Python API completeness ✅

Three gaps found in the post-launch API audit — all involve exposing already-shipped Rust functionality to Python.

1. **`ExprPool` persistence** — `ExprPool.save_to(path)` and `ExprPool.load_from(path)` added to PyO3 bindings. `IoError` exported from `alkahest`.
2. **`GroebnerBasis.compute(polys, vars)`** — `#[staticmethod]` PyO3 binding added; `GroebnerBasis` and `GbPoly` exported from `alkahest`.
3. **Symbolic `solve()` output** — `solve()` now returns `dict[Expr, Expr]` by default; `solve(..., numeric=True)` retains the legacy float output.

**Implementation:** All three in `alkahest-py/src/lib.rs`; exports added to `python/alkahest/__init__.py`.

**Tests:** `tests/test_v16.py` (15 tests) and updated `tests/test_v10.py` (symbolic + numeric solver tests). Full suite: 468 passed, 51 skipped.

---

### V2-21. String / expression parsing (`parse(str)`) ✅

**What:** A Pratt (top-down operator precedence) recursive-descent parser that
converts a human-readable math string into an `Expr` node in a given
`ExprPool`.

**Delivered:**
- `python/alkahest/_parse.py` — pure-Python tokenizer + Pratt parser.
- Supports: integer and float literals, identifiers (auto-interned symbols),
  `+` `-` `*` `/` `^` `**`, unary `+`/`-`, parenthesised sub-expressions, and
  all 20 registered single- and two-argument primitives (`sin`, `cos`, `exp`,
  `log`, `sqrt`, `atan2`, …).
- `parse(source, pool, symbols=None) -> Expr` public API; optional `symbols`
  map for pre-binding names to existing `Expr` objects.
- `ParseError` (E-PARSE-001) raised on lexical and syntax errors, with `.span`
  set to the byte range of the offending token and `.remediation` hint.
- `Expr.pow_expr(exp: Expr)` PyO3 binding added to `alkahest-py/src/lib.rs`
  to support symbolic exponents (the existing `__pow__` only accepts Python
  `int`).
- `parse` and `ParseError` exported from `alkahest.__all__`.
- 54 tests in `tests/test_parse.py` covering atoms, unary operators, binary
  arithmetic, precedence, right-associative `^`, function calls, whitespace,
  symbol-map reuse, round-trip equivalence, and error paths.

---

## v2.0 — Mathematical Coverage

Fifteen algorithms that round out the mathematical surface. Ordered foundations-first (V2-1 … V2-6 enable later items). V3-3 (Logic / `FOFormula`) is pulled forward from v3.0 because V2-9 CAD depends on it.

| Milestone summary | Items |
|---|---|
| Foundations | V2-1 … V2-6 |
| Factorization & Gröbner extensions | V2-7, V2-8 |
| Logic / FOFormula (CAD prerequisite) | V3-3 |
| Real algebraic geometry | V2-9 |
| Symbolic summation | V2-10 |
| Advanced solvers & decompositions | V2-11 … V2-14 |

See also **v2.1 — SymPy Parity** (V2-15 … V2-22) below.

---

### V2-1. Modular / CRT framework as a first-class primitive

**What:** Promote modular reduction, multi-modular CRT lifting, and rational reconstruction from FLINT-internal plumbing to a user-visible transformation layer. Substrate for sparse interpolation (V2-3), factorization (V2-7), F5 (V2-8), primary decomposition (V2-12), differential algebra (V2-13).

**Design:**
- `alkahest-core/src/modular/mod.rs` — `reduce_mod(expr, p) -> MultiPoly<Fp>`, `lift_crt(images) -> MultiPoly<Z>`, `rational_reconstruction(n, m) -> Option<Rational>`.
- Lucky-prime selection; Mignotte bound for required prime count.
- New tracer `ModularValue`; rewrite existing GCD / resultant paths to use this layer.

**Test plan:** Round-trip proptest `lift(reduce(f, p)) == f` for 1 000 random `MultiPoly<Z>`; rational reconstruction sanity; unlucky-prime skip.

**Acceptance:** Criterion `poly_gcd_100` within 5 % of pre-refactor baseline; `alkahest.modular` documented.

---

### V2-2. Resultants and subresultant PRS

**What:** First-class `resultant(p, q, var)` and `subresultant_prs(p, q, var)` as elimination primitives. Required for implicitization, GCDs over algebraic extensions, algebraic-Risch (V1-2), and CAD projection (V2-9).

**Design:**
- `alkahest-core/src/poly/resultant.rs` — thin wrapper over `fmpz_mpoly_resultant` / `fmpq_mpoly_resultant`; pure-Rust subresultant PRS for cases FLINT doesn't cover.
- Certificate: derivation step `Resultant { lhs, rhs, var, eliminant }` with Lean theorem `Polynomial.resultant_eq_zero_iff_common_root`.

**Test plan:** Sylvester-matrix determinant agreement on 500 random pairs; implicitization of `(t², t³) → y² - x³ = 0`; bivariate sanity `res(x²+y²-1, y-x, y) == 2x²-1`.

**Acceptance:** `alkahest.resultant` public; Lean derivation step exported; Criterion row `resultant_degree_20`.

---

### V2-3. Sparse interpolation (Ben-Or/Tiwari, Zippel)

**What:** Recover a sparse polynomial from black-box evaluations. Workhorse of every modern modular polynomial algorithm — without it V2-7 and V2-8 degrade to dense cost on sparse inputs.

**Design:**
- `alkahest-core/src/poly/interp.rs` — `sparse_interpolate(eval, n_vars, bound) -> MultiPoly<Fp>`.
- Ben-Or/Tiwari (Prony-style) for univariate; Zippel for multivariate; dense fallback.

**Test plan:** Univariate: recover `x^100 + 3·x^17 + 5` from 4 evaluations; multivariate: 10-variable 15-term polynomial with ≥ 95 % success over 1 000 trials.

**Acceptance:** `alkahest.sparse_interp` public; ≥ 5× speedup over dense path on 20-variable inputs.

---

### V2-4. Real root isolation (Vincent–Akritas–Strzeboński)

**What:** Given `p ∈ ℚ[x]`, return disjoint rational intervals each containing exactly one real root. Foundation for CAD (V2-9), certified numerical evaluation, and symbolic/numerical mixing.

**Design:**
- `alkahest-core/src/poly/real_roots.rs` — Vincent–Akritas–Strzeboński continued-fraction method; Descartes-on-bitstream fallback; `refine_interval` via `ArbBall`.

**Test plan:** Chebyshev `T_n` for `n ∈ {10, 50, 100}`; cluster test `(x-1)^5 · (x+1)^3`; SageMath oracle on 500 random polynomials of degree ≤ 30.

**Acceptance:** `alkahest.real_roots(p)` public; proptest verifies disjointness and completeness over 5 000 random polynomials.

---

### V2-5. Hermite and Smith normal forms (Storjohann)

**What:** Canonical forms for integer / polynomial matrices. HNF is the module-theoretic analogue of row echelon; SNF is diagonal-Smith. Required for lattice algorithms (V2-6), module-isomorphism checks, and presentation simplification.

**Design:**
- `alkahest-core/src/matrix/normal_form.rs` — `hermite_form(mat) -> (H, U)`, `smith_form(mat) -> (S, U, V)`.
- Storjohann's modular HNF for `Matrix<Z>`; Kannan–Bachem fallback for `Matrix<UniPoly<Q>>`.

**Test plan:** `U · M == H` on 500 random 10×10 integer matrices; SNF diagonal divisibility; parity with Sage/Pari on 100 curated cases.

**Acceptance:** HNF/SNF for both `Matrix<Z>` and `Matrix<UniPoly<Q>>`; Criterion `hnf_50x50` within 2× of Pari.

---

### V2-6. LLL lattice reduction + PSLQ integer relations

**What:** LLL unlocks the practical van Hoeij factorization (V2-7) and algebraic-number minimal-polynomial recovery; PSLQ enables closed-form recognition of numerical constants.

**Design:**
- `alkahest-core/src/lattice/lll.rs` — bind `fplll` via C API; pure-Rust exact fallback on Arb.
- `alkahest-core/src/numeric/pslq.rs` — pure-Rust PSLQ on MPFR.
- `alkahest.guess_relation([pi, E, exp(1) * pi], precision=200)`.

**Test plan:** LLL: Schnorr–Euchner benchmark lattices reduce to known-minimal basis; PSLQ: `pslq([pi² / 6, zeta(2)])` → `[1, -1]` at 100-digit precision.

**Acceptance:** Both algorithms public under `alkahest.lattice` and `alkahest.guess_relation`.

---

### V2-7. Polynomial factorization (CZ, Berlekamp, Zassenhaus, van Hoeij)

**What:** Complete factorization over 𝔽_p, ℤ, ℚ, and ℚ(α). Blocks rational integration (partial fractions), Risch (squarefree prerequisite), and primary decomposition (V2-12).

**Design:**
- `alkahest-core/src/poly/factor/mod.rs` — `factor(p) -> Factorization`.
- Finite fields: Cantor–Zassenhaus + Berlekamp via `fmpz_mod_poly_factor`.
- ℤ[x]: Berlekamp–Zassenhaus for small degree; van Hoeij knapsack-LLL (uses V2-6) for high degree.
- Multivariate: Bernardin–Monagan EEZ on top of `fmpz_mpoly_factor`.
- Certificate: `AlgorithmicCertificate::Factorization { claimed }` verified by Lean `ring_nf`.

**Test plan:** Swinnerton-Dyer `S_5` returns `irreducible` in < 1 s; `Φ_105` factors correctly over GF(2); `(x²+y²-1)(x-y)` recovers its factors; Lean certificate typechecks.

**Acceptance:** `alkahest.factor(p)` on all supported rings; oracle match rate > 98 % on 2 000-factorization SymPy corpus.

---

### V2-8. F5 / signature-based Gröbner basis

**What:** Faugère's F5 (2002) and its descendants (G2V, GVW). Successor to the F4 shipped in V5-11; signatures eliminate zero reductions up front.

**Design:**
- `alkahest-core/src/poly/groebner/f5.rs` behind the `groebner` feature.
- Shares the matrix-reduction kernel with F4 (reuses V1-7's CUDA path when `groebner-cuda` is enabled).
- Signature order: lex-monomial × generator-index.

**Test plan:** Same-basis agreement with F4 on Katsura-{5,6,7}, Cyclic-{5,6}; zero-reduction count < 10 % of F4's on Cyclic-6; ≥ 2× speedup over F4 on Cyclic-7.

**Acceptance:** `GroebnerBasis::compute_f5` API; benchmark dashboard shows F4 vs F5 Pareto frontier on 12-system corpus.

---

### V3-3. Logic / `FOFormula` *(pulled forward — CAD prerequisite)*

**What:** Promote `Predicate` / `PredicateKind` to a first-class `Formula` type with `And`, `Or`, `Not`, `Forall`, `Exists`, and an embedded DPLL/CDCL SAT check.

**Design:**
- `alkahest-core/src/logic/mod.rs` — `Formula` enum; refactor `ExprData::Predicate` to `Formula::Atom`.
- `alkahest.satisfiable(formula) -> bool | dict`.
- Python: `alkahest.And`, `alkahest.Or`, `alkahest.Not`, `alkahest.Forall`, `alkahest.Exists`.

**Test plan:** `satisfiable(And(x > 0, x < 0))` → `False`; `satisfiable(Or(x > 0, x <= 0))` → `True`; existing `Piecewise` and Lean exporter tests still pass.

**Acceptance:** `Formula` in stable API; `satisfiable` public; zero regressions in `Piecewise` / Lean tests.

---

### V2-9. Cylindrical Algebraic Decomposition (real QE) → 🏗 In progress / partial

**What:** Decide first-order sentences over ℝ via CAD. Requires V2-2 (resultants), V2-4 (real root isolation), V2-7 (squarefree factorization), V3-3 (FOFormula).

**Design:**
- `alkahest-core/src/real/cad.rs` — `cad_project`, `cad_lift`, `decide` / `decide_expr` (`QeResult`), `CadError`.
- Brown's projection (`resultant(f, ∂f)`, pairwise resultants); lift via squarefree lcm + [`real_roots`].
- Univariate QE: one outer `\forall`/`\exists` over a purely polynomial ℤ-body; algebraic `\exists` roots handled via gcd–interval checks after rational-cell sampling.

**Test plan:** `∀x. x² + 1 > 0` → `True`; `∃x. x² - 2 = 0` → `True` with witness (`rust` + `tests/test_cad_decide.py`).

**Acceptance:** `alkahest.decide(formula)`, `cad_project`, `cad_lift` on stable surface; Lean export / full 30-problem corpus still open.

---

### V2-10. Creative telescoping / Zeilberger

**What:** Symbolic summation — indefinite (Gosper), definite (Zeilberger), and linear recurrences (Petkovšek).

**Design:**
- `alkahest-core/src/sum/mod.rs` — `sum_indefinite(term, k)`, `sum_definite(term, k, lo, hi)`, `solve_recurrence(rec)`.
- Holonomic-function representation via Ore algebra; Chyzak's fast Zeilberger variant.

**Test plan:** Gosper: `∑_k k · k! = (n+1)! - 1`; Zeilberger: `∑_{k=0}^n C(n,k)² = C(2n, n)` certified by its WZ pair; Petkovšek: Fibonacci recurrence; Mathematica oracle on 40 identities.

**Acceptance:** `alkahest.sum` public; WZ-pair algorithmic certificates; Petkovšek path via V2-10.

---

### V2-11. Regular chains / triangular decomposition

**What:** Kalkbrener / Lazard triangular decomposition — alternative to Gröbner, often faster on structured systems. Uses V2-2 (subresultant PRS) and V2-7 (factorization).

**Design:**
- `alkahest-core/src/solver/regular_chains.rs` — `triangularize(eqs) -> Vec<RegularChain>`.
- Integration with `alkahest.solve`: fallback to `triangularize` when Gröbner times out.

**Test plan:** Linear → single chain matches V1-4's output; 6R-manipulator IK decomposes to ≤ 16 chains; Maple `RegularChains` parity on 15 curated systems.

**Acceptance:** `alkahest.triangularize` public; benchmark row `solve_6r_ik` shows triangular-decomp faster than Gröbner.

---

### V2-12. Primary decomposition (Gianni–Trager–Zacharias) → ✅ Complete

**What:** `I = ⋂ Qᵢ` with associated primes, built on Gröbner bases and polynomial factorization.

**Delivered:**
- `alkahest-core/src/ideal/primary.rs` — `primary_decomposition`, `radical`, `PrimaryComponent`, `PrimaryDecompositionError`.
- Splitting: saturations `I = (I : x_i^∞) ∩ (I + (x_i))` when verified via ideal intersection; zero-dimensional refinement by factoring the univariate generator in the first Lex variable over ℚ (FLINT).
- PyO3: `primary_decomposition(polys, vars)`, `radical(polys, vars)`, `PrimaryComponent` with `.primary()` / `.associated_prime()` → `GroebnerBasis`.
- Tests: `(xy, xz)`, embedded `(x², xy)`, split `(x²-1, y)`; Python `tests/test_primary_decomposition_v212.py`.

**Limitations:** General high-dimensional ideals without separating variables / reducible univariate slices may fall back to a single primary component or hit recursion limits; full GTZ equidimensional machinery is not implemented.

**Acceptance:** `alkahest.primary_decomposition` and `alkahest.radical` on stable surface; complements `solve` / Gröbner tooling.

### V2-13. Differential algebra / Rosenfeld–Gröbner → ✅ Complete

**What:** Gröbner-style elimination for polynomial DAE jets — prolongation of generators plus ordinary ℚ Gröbner bases; complements Pantelides when the structural index is out of reach.

**Delivered:**
- `alkahest-core/src/diffalg/mod.rs` — `rosenfeld_groebner`, `rosenfeld_groebner_with_options`, `rosenfeld_groebner_algebraic`, `dae_index_reduce`, `DifferentialRing`, `DifferentialRanking`, `DifferentialIdeal`, `RegularDifferentialChain`, `RosenfeldGroebnerResult`, `DiffAlgError`.
- `DAE::` / Pantelides alignment via `extend_dae_for_derivative_symbols` (shared derivative-state extension).
- Python (`groebner`): `rosenfeld_groebner`, `dae_index_reduce`, `RosenfeldGroebnerResult`, `DaeIndexReduction`.
- Tests: Rust `diffalg::tests`; Python `tests/test_diffalg_v213.py`.

**Limitations:** Single-branch coherent component (no full multi-case Rosenfeld–Gröbner over initials); nonlinear jets often **truncate** after a bounded prolongation budget rather than saturating the full differential ideal.

**Acceptance:** `rosenfeld_groebner` / `dae_index_reduce` on stable surface behind `groebner`.

### V2-14. Numerical algebraic geometry (homotopy continuation) → ✅ Partial

**What:** Solve polynomial systems numerically via homotopy continuation, certified by Smale-style `α` heuristics and `ArbBall` enclosures.

**Delivered:**
- `alkahest-core/src/solver/homotopy.rs` — total-degree Bézout start (`H = (1-t)·γ·G + t·F` with `G_i = z_i^{d_i}-1`), predictor–Euler + damped Newton corrector, terminal real Newton polish, heuristic Smale estimate, enclosures.
- `solve_numerical`, `CertifiedPoint`, `HomotopyOpts`, `HomotopyError` (`E-HOMOTOPY-*`); stable re-export with `--features groebner`.
- Python: `solve(..., method="homotopy")` (float dict solutions), `solve_numerical` → `CertifiedSolution`; exception `HomotopyError`.
- Benchmark task `numerical_homotopy` in `benchmarks/tasks.py`.

**Limitations:** **Polyhedral / mixed-volume** starts are **not** implemented — deficient systems whose affine root count in `ℂⁿ` is below the Bézout bound are **unsupported** at full root count (e.g. scaling Katsura-8 requires that machinery). Acceptance items that assumed full Katsura coverage remain future work behind polyhedral continuation.

**Test plan:** Decoupled quadratics Π (`x²-1`): all real bitangents recovered; Rust unit tests circle–line intersection; Python `tests/test_homotopy_v214.py`.

**Acceptance:** `numerical_homotopy` benchmark column in `benchmarks/tasks.py`; Katsura-8 / exhaustive Smale corpus deferred until mixed-volume backend.

---

## v2.1 — SymPy Parity

Gaps from the SymPy gap analysis (and integer number theory promoted as V3-1).

| Milestone summary | Items |
|---|---|
| Core calculus / algebra | V2-16 … V2-18; **V2-19 ✅**; V2-20 … V2-21 ✅ |
| Symbolic products | **V2-22 ✅** |
| Integer number theory | **V3-1 ✅** |

---

### V2-15. User-facing `series()` / Laurent expansion → ✅ Complete

**What:** Promote `SeriesTaylor` from an internal MLIR op to a stable user API. Prerequisite for the Gruntz limit algorithm (V2-16).

**Delivered:**
- `alkahest-core/src/calculus/series.rs` — `series(expr, var, point, order) -> Series` with truncated Taylor coefficients from symbolic differentiation; Laurent-style tails for quotients whose denominator vanishes at the expansion point (via valuation stripping + analytic Taylor factor).
- `ExprData::BigO(ExprId)`; `ExprPool::big_o`; `Series(ExprId)` newtype returning one pooled expression `⋯ + O(…)`.
- Pool persistence **format version 3** (tag `12 BigO`).
- Python (`stable`): `series`, `Series`, `SeriesError`, `ExprPool.big_o`; `Expr.node()` emits `["big_o", child]`; `_pretty` renders `\mathcal{O}(…)`.

**Test plan:** `series(cos(x), x, 0, 6)` yields `BigO(x⁶)`; Laurent `series(1/x, x, 0, 4)` → `x⁻¹ + O(x)` (remainder exponent convention documented on Rust `series()`).

**Acceptance:** `alkahest.series` / `alkahest_core::stable::series`; `BigO` checkpoint round-trip — ✅.

---

### V2-16. Limits (Gruntz algorithm)

**What:** `limit(expr, var, point, dir)` — one- and two-sided limits, limits at infinity.

**Design (shipped in v2):**
- `alkahest-core/src/calculus/limits.rs` — finite points: L’Hôpital (0/0), [`local_expansion`](alkahest-core/src/calculus/series.rs) / Laurent tails, algebraic specials (`exp`/`log`), `x·log x` at `0⁺`; at `±∞`: substitute `x ↦ ±1/t` then reduce nested integer powers, clear `t^{-1}` in rational tails via polynomial quotient normalization (`RationalFunction` + expanded simplify), then `t → 0⁺`.

**Remaining / not in scope yet:** Full Gruntz comparability-graph algorithm; oscillatory endpoints; general transcendentals comparable only via Gruntz; Lean `Filter.Tendsto` certificates.

**Test plan:** `limit(x * log(x), x, 0)` → `0`; `limit(sin(x)/x, x, 0)` → `1`; `limit(exp(x), x, oo)` → `oo`; one-sided; SymPy oracle on a curated corpus (`tests/test_limits_v216.py`).

**Acceptance:** `alkahest.limit` in stable API — ✅ prototype rules + oracle corpus passing; Gruntz + Mathlib certificate — 🔲 future.

---

### V2-17. Eigenvalues and eigenvectors

**What:** `M.eigenvals()`, `M.eigenvects()`, `M.diagonalize()` for symbolic matrices.

**Design:**
- `alkahest-core/src/matrix/eigen.rs` — characteristic polynomial via `det(λI − M)`; real root isolation (V2-4) for real eigenvalues; Jordan normal form for defective matrices.

**Test plan:** `[[2,1],[0,2]]` → eigenvalue `2`, multiplicity 2, defective; `[[0,-1],[1,0]]` → `±i`; SymPy oracle on 50 random 3×3 rational matrices.

**Acceptance:** `Matrix.eigenvals`, `Matrix.eigenvects`, `Matrix.diagonalize` in stable API; oracle ≥ 95 %. — ✅

---

### V2-18. Difference equations (`rsolve`)

**What:** Solver for linear recurrences with polynomial coefficients. V2-10 ships Petkovšek for hypergeometric recurrences; this extends to the full `rsolve` scope.

**Test plan:** `f(n) - f(n-1) - 1 = 0` → `n + C₀`; `f(n) - 2*f(n-1) = 0` → `C₀ * 2**n`; Fibonacci; SymPy oracle ≥ 90 % on 40 curated recurrences.

**Acceptance:** `alkahest.rsolve` public; oracle ≥ 90 %.

---

### V2-19. Diophantine equations → ✅ Complete

**What:** Parametric integer solution families for linear two-variable equations, finitely many solutions for equal-coefficient `x² + y² = n`, and the **unit** Pell equation `x² - D·y² = 1` (fundamental solution via the continued-fraction period of `√D`).

**Delivered:**
- `alkahest-core/src/solver/diophantine.rs` — `diophantine(equation, vars)` → `DiophantineSolution` (`ParametricLinear`, `Finite`, `PellFundamental`, `NoSolution`); `DiophantineError` (`E-DIOPH-*`); linear via extended gcd; Pell unit via convergents; circle cases by bounded enumeration.
- Python (`groebner`): `diophantine`, `DiophantineSolution`, `DiophantineError`.
- Tests: Rust `solver::diophantine`; Python `tests/test_diophantine_v219.py` (optional SymPy oracle on sum-of-squares).

**Limitations:** no `x·y` term; ellipse requires equal coefficients on `x²` and `y²`; hyperbolic branch implements **only** `x² - D·y² = 1` after normalization (not general `N` or `+1` right-hand-side families).

**Acceptance:** `alkahest.diophantine` on stable surface when built with `--features groebner`; CI `maturin develop --features groebner`.

---

### V2-22. Symbolic products (`∏`) ✅

**What:** `Product(k, (k, 1, n)).doit()` — multiplicative analogue of V2-10's `Sum`.

**Delivered:**
- `alkahest-core/src/sum/product.rs` — `product_definite`, `product_indefinite` for `q ∈ ℚ(k)` clearing to ℤ polynomials whose FLINT linear factors admit `Γ`-telescoping; `ProductError` (`E-PROD-*`).
- Python: `product_definite`, `product_indefinite`, `Product`, `ProductError`; `examples/products.py` (factorial shortcut + Wallis-style partial product).

**Test plan:** `Product(k,(k,1,n)).doit()` ≡ `Γ(n+1)` numerically; `∏_{k=2}^n (1-1/k²)` → `(n+1)/(2n)`; optional SymPy oracle in `tests/test_product_v222.py`.

**Acceptance:** `alkahest.Product` / kernel products on stable surface — ✅.

---

### V3-1. Integer number theory *(promoted from v3.0)* → ✅ Complete

**What:** `alkahest.number_theory` module: `isprime`, `factorint`, `nextprime`, `totient`, `nthroot_mod`, `discrete_log`, `jacobi_symbol`, quadratic Dirichlet characters. Thin FLINT bindings.

**Delivered:**
- `alkahest-core/src/number_theory/mod.rs` and extended `flint::ffi`; stable re-exports in `alkahest_core::stable`; `NumberTheoryError` (`E-NT-*`).
- Python `alkahest.number_theory`, native `DirichletChi` / `NumberTheoryError`; tests `tests/test_number_theory_v31.py` plus Rust `number_theory::tests`.

**Limitations:** `discrete_log` scans exponents (< prime field size); `nthroot_mod` for prime modulus with `k=2` or `\gcd(k,p−1)=1`; `DirichletChi` implements quadratic characters on odd square-free conductors only.

**Design:**
- Thin wrappers translating decimal strings through FLINT `fmpz_*` primitives.

**Test plan:** `isprime(2**127 - 1)` → `True`; `factorint(2**32 - 1)` → `{3:1, 5:1, 17:1, 257:1, 65537:1}`; `discrete_log` vs brute-force for small `p`; SymPy `ntheory` oracle ≥ 99 %.

**Acceptance:** `alkahest.number_theory` in stable API; oracle ≥ 99 %.

---

## v3.0 — Domain Expansions

| Milestone summary | Items |
|---|---|
| Noncommutative algebra | V3-2 |

---

### V3-2. Noncommutative algebra → ✅ Complete

**What:** `Symbol('A', commutative=False)`; support for matrix Lie algebras, Pauli algebra, and Clifford algebras. Significant kernel change — the AC sorter and e-graph rules both assume commutativity.

**Delivered:**
- `ExprData::Symbol { commutative: bool }`; `ExprPool::symbol_commutative`; `ExprPool::mul` skips canonical sorting when any multiplicative subtree is non-commutative; `canonical_order` / `collect_mul_factors` aligned; e-graph path delegates to rule-based simplify when NC symbols appear; pool format **v4**.
- `alkahest-core/src/algebra/noncommutative.rs` — `PauliSpinAlgebraRule`, `CliffordOrthogonalRule`, `imag_unit_atom`, `pauli_product_rules`, `clifford_orthogonal_rules`.
- `NoncommutativeCost` in `simplify/egraph.rs`.
- Python: `ExprPool.symbol(..., commutative=False)`, `simplify_pauli`, `simplify_clifford_orthogonal`; `examples/noncommutative.py`; `tests/test_noncommutative_v32.py`.

**Test plan:** `A * B ≠ B * A` when both non-commutative; Pauli `σx * σy = i·σz` via registered rules; all existing commutative tests still pass.

**Acceptance:** — ✅
