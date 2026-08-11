# Changelog

## Unreleased

### Silent errors fixed — do results you already computed need rechecking?

A *silent error* is a confident, plausible, mathematically wrong answer with no
exception, no `NaN` and no verification flag. Six were found and fixed this
release. **Four of them shipped in 3.7 or earlier**, so if you have results
from an affected call, re-run them. The other two were in code added during
this release cycle and never reached a published wheel.

| Affected call | Wrong answer it gave | First shipped in | Recheck? |
|---|---|---|---|
| `decide(Forall(x, φ))` where the counterexample is a rational root whose denominator is **not a power of two** | `True` for a **false** universal theorem, e.g. `∀x. (3x+2)² > 0` (false at `x = −2/3`) | ≤ 3.7 | **Yes** — any `decide` verdict |
| `decide(Exists(x, φ))` with an `=` atom | `(True, witness)` where the witness does **not** satisfy the sentence, e.g. `∃x. 3x−2 = 0 → x = 1/2` | ≤ 3.7 | **Yes** — any cited witness |
| `Matrix.nullspace()` on a 2×2 with a symbolic determinant | A confident wrong kernel basis; `[[x,0],[0,1]]` returned `(0, x)`, for which `M·v ≠ 0` | 3.7 | **Yes** — verify `M·v = 0` numerically |
| `simplify` / `simplify_egraph` on a product containing `0⁻¹` | `1`, or `0`, depending on the engine — for an expression with no value at all. Reachable from `diff(2/(x − x), x)` | ≤ 3.7 | Yes, if any input could reduce to `0⁻¹` |
| `decide` on a two-variable sentence true only at an irrational point | `False` for a satisfiable `∃x∃y`, and `True` for its false `∀x∀y` dual | this cycle (2-var `decide` is new) | No published release affected |
| `batch_map(..., parallel=True)` under `context(budget=…)` | Ran **unbudgeted**, so candidates a sequential sweep reported as `E-BUDGET-001` came back as `E-INT-001` — a *mathematical* verdict | this cycle (batch APIs are new) | No published release affected |

Also fixed, and not a silent error but worse for an unattended loop: a Rust
panic escaped `interval_eval` as `pyo3_runtime.PanicException`, which inherits
from `BaseException` and therefore slips past `except Exception`. Shipped in
3.7 — a loop that survived everything else died on `x^(3/2)` over a negative
ball.

The deterministic silent-error gate (`tests/silent_errors/`, Tier-1 CI) now
scores **0 silent errors out of 166 scored cases** (126 correct, 40 honest
refusals) across evaluation, integration, limits, linear algebra, number
theory, real QE, series, simplification, solving, and sums/products. That is a
statement about the corpus, not a guarantee about the library.

### Behaviour changes to plan for

Fixing a silent error means some calls that used to return now refuse. Every one
of these is a call whose previous answer was not justified:

- **`decide` raises `CadError` (`E-CAD-001`) where it used to answer**, whenever
  the formula has a non-strict atom (`=`, `≠`, `≤`, `≥`) and a boundary root has
  not been shown rational. This includes mixed-alternation sentences that route
  through De Morgan — `∀x∃y. p > 0` becomes `¬∃x∀y. p ≤ 0`, and the negation
  makes a strict body non-strict. `decide` is **not** a complete decision
  procedure in this implementation; treat `E-CAD-001` as *undecided*, never as
  *false*.
- **`rank`, `rref`, `nullspace`, `eigenvects`, `jordan_form` raise
  `E-LINALG-010`, and `inverse` raises the new `E-MAT-004`**, when an entry's or
  the determinant's vanishing can be decided neither way. Previously "could not
  prove non-zero" was silently read as "zero".
- **`simplify` leaves `0 · 0⁻¹` unevaluated** instead of returning `1` (or `0`).
  A result containing `(0 * 0^-1)` is Alkahest declining to give an
  indeterminate form a value, not a simplifier failure.

### Known limits — documented, not fixed

These are properties of the design as it stands. They are called out here
because 3.8 is aimed at long unattended loops, and each of them is a way such a
loop fails.

- **`ExprPool` never reclaims.** The arena is append-only: no `clear`, no
  refcount, no GC, and the storage cannot shrink. The only way to free interned
  nodes is to **drop the whole pool** — and every `Expr`, `Matrix`, `Series` and
  `DerivedResult` holds a *strong* reference to its pool, so retaining one
  interesting result retains everything. Growth on a shared pool is linear and
  unbounded (~200 bytes/node; measured ~2 KB per `integrate` call over 20 000
  calls, 0 B/call with a fresh pool per iteration) while per-call latency stays
  **flat**, so the failure mode is a clean OOM with no slowdown to warn you
  first. `ExprPool` also exposes no `__len__` or `stats()`, so the growth is not
  observable from Python. The supported pattern is **one pool per problem**,
  documented in [`budgets.md`](docs/mdbook/src/budgets.md#exprpool-never-reclaims).
- **`run_with_wall_fallback` does not bound wall time for an uncooperative
  callee.** It joins its worker before the exception propagates, so it returns
  when the callee returns: `run_with_wall_fallback(time.sleep, 3.0,
  budget=Budget(wall_ms=50))` raises `E-BUDGET-001` after 3000 ms, and the
  message reports the real elapsed time so this shows up in a log rather than
  being inferred later. Python cannot kill a thread, and abandoning one would
  leak a live thread that still allocates into the pool and can only be stopped
  through the process-wide cancel flag. Only an **OS-level bound** (subprocess,
  process watchdog) is a hard deadline.
- **`wall_ms` granularity is one primitive operation, and FLINT calls cannot be
  interrupted.** After the checkpoint work above the overshoot is a small
  additive term (1.0–1.2×), but past a certain degree a single operation is a
  FLINT factorisation or resultant — one foreign-function call, ~2 s on a
  degree-62 integrand, which no cooperative mechanism can stop part-way.
- **`Matrix.eigenvals()` grows the pool on identical input** (~1.9 KB/call,
  measured over 20 000 calls on the same 2×2 integer matrix): it interns a fresh
  `__eigen_lambda_N` gensym per call. Every other Python-facing entry point
  measured is flat on repeated input. Cache eigenvalue results.
- **`Matrix.eigenvals()` can emit casus-irreducibilis cube roots** — correct
  under Alkahest's real cube-root convention (and honestly refused by
  `eval_expr` with `E-EVAL-009`, with `interval_eval` returning an unbounded
  ball) but evaluated on the **principal** branch by SymPy, NumPy and most other
  tools, which return a confident number that is not an eigenvalue. 14 of 720
  random integer matrices produced one. An honest refusal here becomes somebody
  else's silent error the moment the expression crosses the boundary, so
  evaluate inside Alkahest before exporting, or export a verified numeric
  enclosure instead. See [`interop.md`](docs/mdbook/src/interop.md).
- **The LLVM JIT leaks an LLVM `Context` per compile** (`Box::leak`, on the
  error paths as well as the success path). Feature-gated behind `jit`, so
  default PyPI wheels (Cranelift) are unaffected; do not compile in a loop under
  a `+jit` / `+full` wheel.
- **No sanitizer covers any Python-facing path.** The PR-gating ASan job runs
  with `detect_leaks=0`, the nightly LSan shard cannot reach a `cdylib` with no
  `#[test]` functions, and `pytest` is never run under a sanitizer. The
  behavioural substitute is the fresh-pool sweep described in
  [`TESTING.md`](TESTING.md#3-memory-safety--sanitizers).

### Added

- **`alkahest.ansatz` — parametric families and coefficient fitting** (P2
  autoresearch item 1). "Guess the shape, let the CAS pin the constants" is the
  most common move in experimental mathematics and everybody re-improvises the
  plumbing for it. `ansatz.polynomial`, `.rational`, `.exponential_polynomial`,
  `.linear_combination` and `.quadratic_form` build an `Ansatz` — an object
  rather than a bare `Expr`, because a bare expression loses the distinction
  between an *unknown coefficient* and an *independent variable*, and every
  downstream step needs it. `ansatz.fit(A, residual)` solves for the
  coefficients and returns an `AnsatzSolution` carrying `expr`, `assignment`,
  `rank`, `free`, `residual`, `points` and a `status` — `fit` reports
  `exactly_verified` only when the residual is symbolically zero, never on the
  strength of the collocation points alone (`certify="residual" | "exact" |
  "none"`). `enumerate_family` walks a coefficient grid for conjecture
  generation; `certify_nonneg` hands a fitted candidate to `sos_decompose`.
  Pure Python over primitives that are already fast in Rust (`Matrix.rref`,
  `simplify`, `subs`), so it works without the `groebner` feature; a residual
  genuinely nonlinear in the unknowns refuses with `E-ANSATZ-004` rather than
  degrading silently, and *no member of this family fits* is `E-ANSATZ-003` —
  a closed branch for that family, deliberately not phrased as a proof that no
  such object exists. See
  [`docs/mdbook/src/ansatz.md`](docs/mdbook/src/ansatz.md).

- **`alkahest.crosscheck` — cross-CAS differential testing** (P2 autoresearch
  item 2). A loop that only checks itself finds the bugs it already knows
  about. `crosscheck.check(op, …)` runs one comparison against an external
  oracle (SymPy today; `register_oracle` takes others) through a ladder of
  four rungs — syntactic, symbolic, rigorous-numeric, invariant — and reports
  `agree` / `diverge` / `incomparable` / `unavailable`. The rungs exist because
  most apparent disagreements are not disagreements: two antiderivatives differ
  by a constant, two simplifiers pick different normal forms. Only the
  invariant rung (differentiate the antiderivative, substitute the solution
  back, telescope the antidifference) settles those, and an operation that has
  no invariant stops at rung 3 rather than pretending. **A missing oracle is
  `unavailable`, never `agree`** (`E-XCHECK-002`) — the one failure mode that
  would quietly turn the whole module into a no-op. `sweep(cases=…, seed=…)`
  generates a seeded corpus and prints its seed in `summary()` always, because
  a sweep is only useful as a bug report if the run that found something can be
  reproduced; the seed defaults to `budget_seed()`, so a nightly job and a
  local reproduction share one knob. `run_frozen_corpus()` replays 9 pinned
  cases whose expected outcome is recorded with the reason. See
  [`docs/mdbook/src/crosscheck.md`](docs/mdbook/src/crosscheck.md).

- **`alkahest.smt` — SMT-LIB 2 export and a z3/cvc5 bridge** (P2 autoresearch
  item 3). Discrete and mixed integer/real/boolean subproblems are not
  Alkahest's problem class, and the fastest way to make it worse would be to
  pretend otherwise. `to_smtlib` emits a complete runnable script (the emitter
  lives in Rust next to `Formula`, with no `_ =>` arm anywhere in it, so a
  kernel node added later fails to compile rather than silently emitting
  plausible-but-wrong SMT-LIB); `smt.solve` runs an installed solver and reads
  the answer back. The trust asymmetry is the design: a **`sat` model is lifted
  to exact rationals and substituted back and checked in-process**
  (`exactly_verified`; a model that fails raises `E-SMT-004`), while **`unsat`
  is reported as `externally_asserted`** and is deliberately excluded from
  `research.MACHINE_CHECKED_STATUSES`, because consuming an unsat proof is a
  different project. Decimal literals are parsed from the *string*, so `0.1`
  becomes `Fraction(1, 10)` and never the nearest binary double; an algebraic
  witness (`root-obj`) is refused with `E-SMT-003` rather than evaluated to a
  float, since a float witness recorded as an exact one is precisely the silent
  error the bridge exists to prevent. `smt.supported(f)` answers "would this
  route work, and should I take it" *before* any solver runs, and recommends
  `prefer_in_tree` for real arithmetic with no integer variables — the in-tree
  routes produce artifacts, `nlsat` produces only an answer. `solve` takes
  quantifier-free formulas; `to_smtlib` exports quantified ones. See
  [`docs/mdbook/src/smt.md`](docs/mdbook/src/smt.md).

- **Asymptotics of sums — Euler–Maclaurin** (P1 mathematics item 10):
  `alkahest.experimental.euler_maclaurin(f, k, a, n, corrections=…)` expands
  `Σ_{k=a}^{n} f(k)` as `n → ∞`, recovering
  `H_n ~ log n + γ + 1/(2n) − 1/(12n²) + …` for `f = 1/k`. `series` and Gruntz
  `limit` expand a *function*; this is the sum side, which is how conjectures
  about growth rates get settled. Returns an `AsymptoticReport` that records
  not just the terms but **how much is proved**: `rigor`, a per-hypothesis
  checked/assumed ledger, and the numeric evidence from the `o()`-gate. The
  additive constant (γ above) is *not* determined by Euler–Maclaurin from the
  `n`-side terms — it is fitted numerically, and the report says so rather than
  presenting it as derived. Terms are ordered by magnitude at the check points,
  so the constant lands below every growing term and above every decaying one.
  Refuses (`AsymptoticError`) when the summand has no symbolic antiderivative,
  is not evaluable at the check points, or no term survives the gate. See
  [`docs/mdbook/src/asymptotics.md`](docs/mdbook/src/asymptotics.md).

- **Rigorous global bounds — Taylor models and validated numerics** (P1
  mathematics item 9): `alkahest.bound_on_box(expr, box)` returns a rigorous
  enclosure of the *range* of an expression over an axis-aligned box;
  `alkahest.verified_integral(expr, var, a, b)` a rigorous enclosure of a
  definite integral; `alkahest.verified_no_roots` and
  `alkahest.verified_sign` three-valued (`"true"` / `"false"` /
  `"undecided"`) predicates. Ball arithmetic already gave rigorous *pointwise*
  evaluation — this closes the gap to rigorous statements quantified over a
  region, which is what turns a numeric observation into a theorem. New
  `alkahest_core::validated` module: Taylor model arithmetic in normalised box
  coordinates (polynomial part plus rigorously enclosing remainder, so `x - x`
  cancels symbolically instead of widening to `[-2, 2]`), and Moore–Skelboe
  branch-and-bound that prunes sub-boxes proven not to contain the extremum.
  Soundness over tightness throughout: outward rounding everywhere, and
  exhausting the work budget returns a wide-but-true enclosure with
  `budget_exhausted=True` rather than an error. Genuine failures refuse with
  `ValidatedError` (`E-VALIDATED-001` unsupported primitive, `-002` unbound
  symbol, `-003` singularity in the box, `-004` overflow, `-005` malformed
  request). See
  [`docs/mdbook/src/validated-bounds.md`](docs/mdbook/src/validated-bounds.md).
- **Positivity certificates — SOS and Positivstellensatz-lite** (P1 mathematics
  item 8): `alkahest.sos_decompose(p, vars)` returns an exact rational
  sum-of-squares decomposition `p = Σ σ_j q_j²`, and
  `alkahest.prove_nonneg(p, vars, constraints=[...], level=...)` returns a
  Handelman certificate `p = Σ_α c_α Π g_i^{α_i}` (`c_α ≥ 0`) on a basic
  semialgebraic set. This is the fast, certificate-producing complement to the
  complete-but-doubly-exponential `decide`: the output is a short algebraic
  identity anyone can re-expand, exportable to Lean via
  `PositivityCertificate.to_lean()`. New `alkahest_core::real::sos` module with
  its own exact rational simplex (Bland's rule — no floating point anywhere
  near a certificate), an ℚ multivariate polynomial layer, and a
  generator-cone Gram search. Every certificate is re-expanded and compared
  against the target identically before it is returned. The three outcomes are
  kept distinct on purpose: certified, `E-SOS-003` definitely negative (with a
  witness point), and `E-SOS-002` no certificate of this shape at this degree
  — which is a statement about the search, not a proof that none exists (the
  Motzkin polynomial refuses here rather than being misreported). See
  [`docs/mdbook/src/positivity.md`](docs/mdbook/src/positivity.md).
- **Creative telescoping — Zeilberger's algorithm** (P1 mathematics item 7):
  `alkahest.zeilberger(term, n, k, max_order=…, max_degree=…)` takes a proper
  hypergeometric term `F(n, k)` and returns a `ZeilbergerCertificate` carrying
  a P-recursive recurrence `Σ_i a_i(n)·F(n+i,k) = G(n,k+1) − G(n,k)` together
  with the rational certificate `R` (`G = R·F`) — so `S(n) = Σ_k F(n,k)`
  satisfies `Σ_i a_i(n)·S(n+i) = 0`. This is the first operation in the CAS
  that is both a *decision procedure* over its class and a *certificate
  emitter*, which is what makes discovery→proof automatic in an agent loop
  rather than heuristic. New `alkahest_core::holonomic` module: exact `Q(n)`
  and `Q(n)(k)` arithmetic towers, proper-hypergeometric recognition
  (`gamma`, `factorial`, `binomial`, `pochhammer` heads), and the Gosper-style
  reduction over `Q(n)`. Every certificate is re-checked as an exact
  `Q(n)(k)` identity before return; a candidate that fails is discarded, never
  returned with a caveat. Refuses via `HolonomicError` with stable codes —
  `E-HOLO-001` (outside the proper hypergeometric class), `E-HOLO-002` (search
  bounds exhausted), `E-HOLO-003` (candidate failed exact verification),
  `E-HOLO-004` (malformed call). See
  [`docs/mdbook/src/telescoping.md`](docs/mdbook/src/telescoping.md).

- **Docs: autoresearch / search-plumbing guide.** New mdBook chapter
  [`search-plumbing.md`](docs/mdbook/src/search-plumbing.md) ties budgets,
  batch APIs, compact `DerivedResult` envelopes, claim graphs, and certificate
  coverage together; Sphinx gains [`api/workload.rst`](docs/sphinx/api/workload.rst)
  plus `DerivedResult.to_dict` / `BudgetExceededError` entries. Cross-links from
  getting-started, intro, README, claim-graphs, batch, and budgets.

- **Budgets, cooperative cancellation, and a determinism seed** (P1 search
  plumbing item 4): `alkahest.Budget(wall_ms=..., max_steps=..., seed=...)`
  and `alkahest.context(budget=...)` push a wall-clock/step budget into a
  new Rust-side cooperative checkpoint (`alkahest_core::budget`), so a
  fan-out loop trying many candidate integrals/rewrites can bound one
  candidate's cost instead of hanging on it. `alkahest.integrate` checks it
  at its top-level entry and its recursion boundary and raises
  `BudgetExceededError` (`E-BUDGET-001` wall clock, `E-BUDGET-002` step
  limit, `E-BUDGET-003` cancelled) rather than running unbounded;
  `alkahest.simplify` checks it once per rewrite pass and, since it has no
  error channel, stops early instead of raising (`run_with_wall_fallback`
  supplements this with a hard deadline via a worker thread when needed).
  `alkahest.request_cancel()` / `clear_cancel()` / `is_cancelled()` expose a
  process-wide cancellation flag so an orchestrator thread can stop a heavy
  call running elsewhere; `alkahest.budget_seed()` exposes the active
  budget's seed to RNG-consuming samplers for reproducible runs. See
  [`docs/mdbook/src/budgets.md`](docs/mdbook/src/budgets.md).

- **Batch and streaming evaluation** (`alkahest._batch`, Python-only): `batch_map` /
  `batch_map_iter` call a function once per item and **never raise** for a single bad
  element — the exception is captured into a `BatchItem(index, ok, value, error,
  elapsed_ms)`, with `error["code"]` set to the failing exception's own `E-*` code
  (`E-BATCH-001` as a fallback for exceptions with none). `batch_map` always returns
  results in input order, whether or not `parallel=True` fans the calls out over a
  `ThreadPoolExecutor`; `batch_map_iter` streams in input order when sequential and in
  completion order when parallel. `integrate_many`, `simplify_many`, and `diff_many` are
  thin `batch_map` wrappers over the three most common derivation entry points. See
  [`docs/mdbook/src/batch.md`](docs/mdbook/src/batch.md).

- **`DerivedResult.to_dict` / `.to_json`: versioned, token-efficient result
  envelopes** (P1 search plumbing item 6). Combines `.value`, `.verification`,
  `.certificate_status`, and `.steps` into one dict/JSON string with a stable
  `"kind": "alkahest.derived_result"` discriminator and independent
  `RESULT_SCHEMA_VERSION` / `STEPS_SCHEMA_VERSION` constants (also exported at
  module level and as `DerivedResult.SCHEMA_VERSION` /
  `.STEPS_SCHEMA_VERSION`). `mode="compact"` drops `before`/`after` step text
  and uses short step keys (`r`/`s`), but never renames, hides, or drops
  `verification["status"]` and never includes Lean certificate source in
  either mode. See `docs/mdbook/src/derivations.md`.

- **Python bindings for the parallel simplifiers**: `simplify_redex`,
  `simplify_auto` and `simplify_strategy` join the existing `simplify_par`.
  All take a single expression and return the same result as `simplify`; only
  the schedule differs. Each falls back to sequential `simplify` when the
  extension is built without `--features parallel` (as the PyPI wheel is), so
  the calls are always available and `simplify_strategy` then reports
  `"sequential"`. The three parallel entry points now release the GIL for the
  duration of the native call, so other Python threads run alongside them.

- **`simplify_redex`: level-scheduled parallel simplification** (Rust,
  `--features parallel`, exported through `experimental`). Buckets the
  expression DAG by height and simplifies each level with one `par_iter`, using
  a flat `Vec<AtomicU32>` indexed by `ExprId` as the memo instead of a hashed
  side table. Borrowed from HVM2's redex-bag scheduling; the interaction-net
  runtime itself does not transfer, since alkahest's hot paths are FLINT/Arb
  arithmetic. Does **not** replace `simplify_par` — best time over 1–32 threads
  on 32 cores: deep chain 23.1 ms → 5.5 ms, but a wide sum of independent terms
  5.1 ms → 10.3 ms. Fork-join keeps a chain on one worker and wins on width;
  level scheduling wins on depth, and on every shape at one thread. The
  traversal is iterative (no stack-overflow risk on deep inputs) and each node
  is visited once, so the derivation log is deterministic across thread counts.
  A barrier-free variant using per-node counters of unreduced children — HVM2's
  actual discipline — was measured and was never reliably faster, so it is not
  included.

### Fixes

- **`decide` proved false universal theorems** (silent error; shipped in 3.7).
  `∀x. (3x+2)² > 0` returned `(True, None)`. It is false at `x = −2/3`, exactly:
  `9·(4/9) + 12·(−2/3) + 4 = 0`, and `0 > 0` is false. No approximation appears
  anywhere in that argument, and `decide` is the engine behind every stability
  proof and bound check, so a false `True` here is a machine-checked-looking
  proof of a false theorem. Sweeping `∀x. (a·x − b)² > 0` over `a ∈ 1..9`,
  `b ∈ −6..6` gave a clean rule: the verdict was wrong **exactly when the double
  root `b/a` in lowest terms has a denominator that is not a power of two** —
  which is why `x² > 0` and `(x−1)² > 0`, the two cases already in the corpus,
  passed. The bug lived one denominator to the right of every existing test.
  Two layers, and the deeper one was a broken documented contract:
  `RootInterval` promises `lo == hi == r` for an exact rational root `r`, but
  the VAS isolator only recorded an exact root when the transformed polynomial
  vanished at a Möbius endpoint, which happens for dyadic roots and not in
  general (`real_roots(3x − 2, x)` returned the open bracket `(0, 1)`). CAD then
  built its sample set from rational bracket endpoints and midpoints and
  concluded `false` when none satisfied the formula — sound for a *strict* atom,
  whose solution set is open, but not for a non-strict one, whose solution set
  can be the single untested root; `∀x. φ` goes through `¬∃x. ¬φ`, so the missed
  witness became a `True` universal. Fixed exactly, not heuristically: by the
  rational-root theorem every rational root of an integer polynomial has
  denominator dividing the leading coefficient, so once a bracket is bisected
  below width `1/lc` it contains at most one such rational and exact rational
  evaluation settles it — `None` means "no rational root here", never "probably
  not". (Bisection requires a strict sign change and refuses to collapse onto a
  vanishing *endpoint*: neighbouring brackets share endpoints, and collapsing
  onto one deletes the root the bracket was isolating.) Where the boundary root
  is genuinely irrational the sample set is incomplete and nothing can fix that
  by sampling, so `decide` now refuses with `E-CAD-001` rather than fabricating
  a `false`. A randomised differential test against a `sympy.real_roots`
  multiplicity analysis found **18 wrong verdicts in the first 150 random
  polynomials** before the fix and **0 in 1 000** after.
- **`decide` returned existential witnesses that do not satisfy the sentence**
  (silent error; shipped in 3.7). `∃x. 3x − 2 = 0` returned
  `(True, {'x': '1/2'})`, and `3·(1/2) − 2 = −1/2 ≠ 0`. The verdict was right;
  the certificate was false — and a witness is the one part of an answer that
  looks like it needs no trust, so it is exactly the artefact a loop cites
  downstream. The `Eq`-interval fallback proved satisfiability on an isolating
  interval and then reported the interval **midpoint**. It now runs the same
  check any caller would (`eval_qf_formula` at the reported point) and reports
  `witness=None` rather than a point that fails. With the exact-rational-root
  recovery above in place the true witness is usually reported outright:
  `∃x. 3x − 2 = 0` → `(True, {'x': '2/3'})`, while `∃x. x² = 2` → `(True, None)`
  because no rational witness exists. Two existing tests that asserted the bogus
  witness are corrected with the reason spelled out.
- **A Rust panic escaped `interval_eval` as a `BaseException`** (shipped in 3.7).
  `interval_eval(x**Rational(3,2), {x: ArbBall(-3.3, 0.0)})` panicked at
  `ball/mod.rs` and surfaced as `pyo3_runtime.PanicException`, which inherits
  from `BaseException` — so a loop's `except Exception` handler did not catch it
  and the run died on an input it was supposed to survive. Not a silent error,
  but for multi-day unattended operation arguably worse than one. `ArbBall::pow_f`
  guarded a negative base with `!exp.is_exact()`, but `x^(3/2)` arrives as an
  *exact* point ball at 1.5, `(−3.3)^1.5` is `NaN`, and the corner-ordering
  `partial_cmp(...).unwrap()` then panicked; the same shape existed in
  `ArbBall::Div` via `∞/∞`, reachable from `(x^(3/2))^-2`. A negative base now
  requires an exact **integer** exponent, and both `pow_f` and `Div` check the
  corner set for `NaN`, returning the existing "no enclosure" answers. 306
  panicking expressions in the first fuzz run; **0** after, across 7 200
  expressions × 14 points.
- **`run_with_wall_fallback` poisoned the whole process on timeout.**
  `request_cancel()` sets a process-wide, sticky flag, and
  `run_with_wall_fallback` never cleared it — so one expired candidate, the exact
  event the API exists to handle, made every subsequent cooperative call in the
  process fail with `E-BUDGET-003` forever. A multi-day loop would have died at
  its first slow integral and then reported a cancellation storm that was really
  one timeout. The executor is now wrapped in `try/finally` and the flag restored
  *after* `ThreadPoolExecutor.__exit__` has joined the worker (so the cancelled
  call has already observed it), and only when this call was the one that raised
  it — an orchestrator with its own outstanding `request_cancel()` keeps its
  request. Survives 20 of 20 timeout+work cycles. Two regression tests in
  `tests/test_budget.py`. *(Introduced during this release cycle; no published
  release is affected.)*
- **`batch_map(parallel=True)` ran completely unbudgeted.** `BudgetGuard` is
  `!Send` and the budget frame stack is thread-local, so `context(budget=…)` had
  no effect on work fanned out over a `ThreadPoolExecutor`: measured, the main
  thread saw the budget and all four workers saw `False`. For unattended
  operation that is the safety mechanism silently not applying — and worse than
  simply "slower", because the candidates a sequential sweep reported as
  `E-BUDGET-001` came back from a parallel one as `E-INT-001`, the integrator's
  verdict that *no elementary antiderivative exists*. A loop records that as a
  permanently closed branch when nothing was decided. `batch_map` now snapshots
  the active budget on the calling thread and re-enters it inside every worker
  task, and `run_with_wall_fallback` likewise enters its `budget` argument on the
  worker thread it spawns. The semantics are documented rather than fudged:
  `wall_ms` stays a single sweep-wide deadline (captured at the `batch_map` call,
  since Python cannot read the frame's start instant), while `max_steps` becomes
  **per item**, because the Rust step counter lives in the frame and is not
  readable from Python. One item tripping its budget never cancels its siblings;
  `request_cancel()` still reaches every worker, because the flag is process-wide.
  *(Introduced during this release cycle; no published release is affected.)*
- **`simplify` gave `0 · 0⁻¹` a value** (silent error; shipped in 3.7). `0⁻¹` is division by
  zero, so `0 · 0⁻¹` is the indeterminate form `0·∞` and has no value under any
  convention — but `simplify` returned `1`, `simplify_egraph` returned `0`, and
  `simplify(5 · 0⁻¹ · 0)` returned `0`, so the three answers were their own
  proof that at least two of them were wrong. The rest of the library was
  already right: `eval_expr(0⁻¹)` raises `E-EVAL-009` and `simplify(0⁻¹)` leaves
  the power unevaluated. Four rules were each collapsing the surrounding
  product on their own: `collect_mul_factors` summed the exponents of a common
  base (`0¹ · 0⁻¹ → 0⁰ → 1`), which is `b^k·b^m = b^(k+m)` — an identity that
  needs `b ≠ 0` the moment one exponent is negative; `const_fold` absorbed the
  product to `0` because one factor was the literal zero; `collect_add_terms`
  dropped a summand whose integer coefficient was `0` without checking that the
  surviving factor was a *number*; and the e-graph's shrink ruleset contains
  both `(Mul ?x (Num 0)) → (Num 0)` and `(Mul ?x (Pow ?x (Num -1))) → (Num 1)`,
  so on this input it unioned `0` and `1` into one e-class. All four now decline.
  Reachable without writing `0⁻¹` by hand: `diff(2/(x - x), x)` returned `1` for
  a function whose domain is empty; it now returns an expression that
  `eval_expr` refuses. Scope, stated plainly: the guards test for a **literal**
  zero base, which — because the rule engine normalises strictly bottom-up —
  also covers every base the simplifier can reduce to zero, `x - x` included. A
  base that is zero but not provably so keeps the documented `b · b⁻¹ → 1`
  convention: a three-valued `zero_status` on the `Mul` rewrite path costs
  several 128-bit ball evaluations per node, which this path cannot afford.
  `simplify_egraph` is the exception — it hands the whole call to the rule
  engine when it finds a provably-zero denominator, and uses the full
  `zero_status` to decide that, because building and saturating an egglog
  program dwarfs the test. No measurable cost on
  `bench_codspeed.py::test_log_exp_simplify_depth4` (paired A/B over 20
  interleaved runs: median −0.8%, inside the ±13% noise of the machine).
  Nine cases added to the silent-error corpus, four of them controls that
  `x · x⁻¹ → 1`, `0 · x → 0`, `2x − 2x → 0` and the e-graph engine itself still
  work.
- **`decide` could deny a two-variable statement that is true only at an
  irrational point** (silent error; two-variable `decide` is new in this cycle,
  so no published release is affected). The univariate completeness guard shipped
  earlier in this release refuses rather than report an unsatisfiability it
  never checked at a boundary root; the two-variable path had the same guard but
  keyed on `=` / `≠` atoms only, so `≤` and `≥` still fell through.
  `∃x∃y. (x²−2)² + y² ≤ 0` — true at `(±√2, 0)`, where both squares vanish, and
  false everywhere else — came back `False`, and its dual
  `∀x∀y. (x²−2)² + y² > 0` came back `True`, a machine-checked-looking proof of
  a false theorem. `project_and_sample_x` already flagged the untested
  irrational projection root; the flag now escalates for every non-strict atom,
  matching `body_has_boundary_atom` one dimension down. Strict atoms are
  unaffected: their solution sets are open, so the open-cell midpoints are
  complete for them. Both sentences now refuse with `E-CAD-001`. The cost is
  more refusals in the mixed-alternation cases, which route through De Morgan
  and so present a negated (hence non-strict) body: `∀x∃y. p > 0` becomes
  `¬∃x∀y. p ≤ 0` and refuses where it used to answer. Five corpus cases and
  four Rust unit tests, including the controls that a *rational* boundary point
  is still found (`∃x∃y. (3x−2)² + y² ≤ 0` → `True` at `(2/3, 0)`) and that a
  genuinely unsatisfiable `≤` still decides `False`.
- **`Matrix.nullspace()` returned a confident wrong basis for any 2×2 with a
  symbolic determinant** (silent error; shipped in 3.7). The 2×2 fast path
  returns the perpendicular of a non-vanishing row, which is the kernel *only*
  when `det = 0`, and its full-rank gate recognised only a **literal** non-zero
  constant. Every non-literal determinant fell through into the rank-1 answer —
  "could not prove `det ≠ 0`" read as "`det = 0`", the exact mirror of the `rref`
  defect that motivated the three-valued zero test. `[[x, 0], [0, 1]]` returned
  the basis `(0, x)`, for which `M·v = (0, x) ≠ 0`, while `rank()` on the same
  matrix said 2 — two public calls making 2 + 1 = 3 for a 2-column matrix. No
  exotic function was needed to trigger it. The gate now uses the three-valued
  `zero_status`: proven non-zero → trivial kernel, proven zero → the
  perpendicular, undecidable → refuse with `E-LINALG-010`, matching what `rank`
  already did. The eigen paths (`eigenvects`, `jordan_form`, `matrix_exp`) are
  untouched: `det(A − λI) = 0` holds there *by construction* — λ is a root of the
  characteristic polynomial — so the caller states it via the new
  `KnownSingular` parameter rather than asking the simplifier to rediscover it
  from nested radicals, which it often cannot. Four cases added to the
  silent-error corpus, including a control that a genuinely rank-1 symbolic
  matrix still returns its kernel and one that checks `M·v = 0` numerically
  rather than just the dimension.
- **`nullspace`, `eigenvects` and `jordan_form` reported an undecidable entry
  with a vague code.** All three share one elimination routine, whose error
  type carried no payload (`Result<_, ()>`), so the specific refusal —
  "one entry's vanishing could be proven neither way, substitute concrete
  parameters and it works" (`E-LINALG-010`) — died at that boundary and came
  back as the generic `E-LINALG-002` / `E-EIGEN-006` "could not compute
  nullspace basis". The routine is `pub(crate)`, so widening its error type
  costs nothing on the public API (`cargo semver-checks` agrees). A *genuine*
  kernel failure still reports `E-LINALG-002` and can never inherit a previous
  refusal's code — `KernelFailed` is deliberately not an out-of-band carrier.
- **`Budget(wall_ms=…)` overshot `integrate` by an unbounded factor.** The
  checkpoints existed; the seconds were being spent between them. A 300 ms
  budget on `∫ cos x·sinⁿx/(sin⁹x + sin x + 1) dx` returned after 2–4 s, and the
  same family at degree 40 never returned at all — it had to be killed from
  outside the process. Measured rather than guessed: 98.7% of one such call was
  a single number-field Euclidean GCD (`alg_log_argument` → `kpoly_gcd`), and
  the residue after fixing that was a single ℚ[x] GCD normalising `A/D` to
  lowest terms (480 ms of a 482 ms call). Both Euclidean loops now check the
  budget per step, `integrate_raw` checks on entry (so a *sum* is bounded
  between summands), and the rational route checks at each stage boundary.
  The same ladder now overshoots by 1.0–1.2×, and the degree-40 case returns in
  317 ms. Because a GCD has no error channel and stopping one early returns a
  *wrong* GCD, the budgeted variants return `None` rather than a truncated
  answer, and the public `poly_gcd` / `NumberField::kpoly_gcd` signatures are
  unchanged. What remains is documented in `docs/mdbook/src/budgets.md`: the
  granularity is one primitive polynomial operation, and past a certain degree
  that is a FLINT call, which no cooperative mechanism can interrupt.
- **`request_cancel()` could not reach a running `integrate` or `limit`.**
  Two independent causes. The bindings held the GIL for the whole call, so a
  watchdog thread could not execute a single bytecode until the operation it
  wanted to cancel had already finished — only a flag set *before* the call was
  ever observed, which is the opposite of what a fan-out search loop needs.
  Both now release the GIL around the core call, using the idiom `simplify_par`
  already established. And `integrate`'s u-substitution search discarded every
  error from its recursive call, budget trips included, so it moved on to the
  next of up to twelve candidates instead of stopping; a budget error now
  propagates, and the search checks the budget once per candidate — the
  granularity where the seconds actually go.
- **Docs: `simplify_par` was documented with a signature it never had.** Both
  the Sphinx API page and the mdbook chapter showed it taking a list of
  expressions and returning a list; it takes one expression and returns one
  `DerivedResult`, and the documented call raises `TypeError`.
- **Docs: the documented local Valgrind command checked nothing.** `TESTING.md`
  globbed `target/.../deps/alkahest_core-*` and `CONTRIBUTING.md` /`TESTING.md`
  said `cargo test -p alkahest-core`, but the package is named **`alkahest-cas`**
  (`alkahest-core/` is only the directory). The glob matched zero binaries, the
  `[ -x "$bin" ] || continue` guard skipped the empty expansion, and the loop
  **exited 0 having run Valgrind on nothing**. Both are corrected, with a note
  on the naming so it does not come back. Also corrected in the same pass:
  `TESTING.md` claimed UndefinedBehaviorSanitizer coverage
  (`-Zsanitizer=undefined` appears nowhere in the repo), and `CONTRIBUTING.md`
  claimed Tier-1 CI runs "ASan on FFI tests" when that job is scoped to the crate
  *below* the FFI boundary and runs with `detect_leaks=0`.
- **Docs: `Matrix.inv()` and `M[i, j]` do not exist.** The Sphinx matrix page
  documented both; the methods are `inverse()` and `get(i, j)`, and `Matrix` is
  not subscriptable. The same page attributed the singular-matrix refusal to
  `E-MAT-001` (shape mismatch) rather than `E-MAT-003`.
- **Docs: the Rust crate path was wrong throughout.** Guide pages wrote
  `alkahest_core::…`, which is the *workspace-local alias* `alkahest-py` gives
  the dependency. A downstream crate writes `alkahest_cas::…`; corrected in the
  mdBook chapters, `ARCHITECTURE.md` and `CONTRIBUTING.md`.
- **Withhold Lean certificates for Basel-family infinite sums.** The
  `basel_zeta_even` derivation step had no Mathlib proof and fell through to
  the default `by ring_nf; simp` tactic, emitting false equalities (e.g.
  `3/k² = π²/2`) into the textbook-gate Lean pool. Treated like Gosper:
  withhold the whole certificate rather than emit a broken one.
- **Rustdoc:** drop a private-item intra-doc link from `sum_definite` that
  broke `cargo doc -D warnings`.
- **Ruff:** quiet unused `wit` bindings and a compound assert in
  `tests/test_cad_decide.py`.

### Performance

- **`numpy_eval` / `numpy_eval_par` no longer round-trip through a Python
  list of floats.** The previous implementation converted every NumPy array
  to a flat Python list via `.tolist()` before crossing into Rust
  (`compiled_fn.call_batch_raw(inputs_flat, ...)`), which boxes/unboxes one
  `PyFloat` object per element on both the input and output side — the root
  cause of `numpy_eval` measuring ~25× slower than `sympy.lambdify(...,
  "numpy")` for large batches. `CompiledFn` gains
  `call_batch_buffer`/`call_batch_buffer_par`, which read NumPy (or any
  buffer-protocol) `float64` arrays via a single bulk copy per array, run
  the native `call_batch`/`call_batch_par` with the GIL released, and write
  results directly into a caller-supplied output array. `numpy_eval` and
  `numpy_eval_par` (and the JAX primitive's concrete-eval path) use this
  fast path automatically, with a transparent fallback to the legacy
  `call_batch_raw`/`call_batch_raw_par` for older extension builds that
  lack it. `call_batch_raw`/`call_batch_raw_par` are unchanged and kept for
  backward compatibility. Non-contiguous or non-float64 inputs are still
  converted once via `np.ascontiguousarray(..., dtype=np.float64)`, never
  via `.tolist()`.

### Additions — earlier in this cycle

- **`decide` now handles two real variables with a quantifier prefix of
  length ≤ 2** (`alkahest_core::real::cad`), not just the single-variable
  fragment from V2-9: `∃x∃y`, `∀x∀y` (same-flavor blocks), and mixed
  alternation `∃x∀y` / `∀x∃y`, all for purely polynomial bodies over ℚ. The
  approach eliminates one variable via the existing [`cad_project`] (Brown
  projection), then re-decides the resulting univariate-in-the-other-variable
  sentence at every rational CAD-cell sample with the existing univariate
  engine — e.g. `∃x∃y. x²+y²=0` → `true` (witness `x=0, y=0`), `∃x∃y.
  x²+y²+1=0` → `false`, `∀x∀y. x²+y²≥0` → `true`, `∀x∀y. x·y>0` → `false`.
  If a projection root is irrational *and* the body contains an
  equality/inequation atom, that CAD cell can't be tested exactly with
  rational sampling alone (it would need full algebraic-number CAD lifting),
  so `decide` raises `Unsupported` (`E-CAD-001`) rather than risk an unsound
  `true`/`false` — same for quantifier prefixes longer than two variables.
  The univariate fragment's behavior is unchanged; `decide_exists_univariate`
  internals were also generalized (via `UniPoly::from_symbolic_clear_denoms`)
  to accept rational (not just integer) coefficient polynomials, which the
  two-variable path needs after substituting a rational sample for the
  eliminated variable.
- **`Assumptions` is now first-class for agent workflows.** Previously
  `Assumptions` had to be threaded through `simplify()` / `simplify_log_exp()`
  by hand on every call. Now:
  - `alkahest.context(pool=p, assumptions=my_assumptions)` sets a thread-local
    default; `simplify()`, `simplify_log_exp()`, and `solve()` pick it up
    automatically whenever the caller omits their own `assumptions=`/explicit
    argument (an explicit argument always overrides the context). See
    `alkahest._context.active_assumptions` and the updated `context()`
    docstring.
  - `solve(equations, vars, assumptions=...)` (or the context default) now
    drops any returned solution that assigns a non-positive value to a
    variable the assumptions declare `> 0` — e.g.
    `solve([x**2 - 4], [x], domain="real", assumptions=positive_x)` returns
    only `x = 2`. This composes with `domain="real"` as a final filter rather
    than replacing its complex/real logic, and is a no-op (returns the
    `GroebnerBasis`/list unfiltered) when there's nothing to check.
    `Assumptions.is_positive(expr)` is the new agent-facing predicate this is
    built on (true for an explicit `refine(x > 0)` fact or a `Domain.Positive`
    symbol).
  - **New sound rewrite: `abs(x) → x` under `x > 0`.** Joins the existing
    `sqrt(x**2) → x` / `exp(log(x)) → x` family gated on
    `Assumptions`/`Domain.Positive`. `abs(x) → -x` under `x < 0` is *not*
    added (a distinct, currently untracked, side condition) — only the sound
    direction ships. Emits a Lean certificate (`abs_of_pos`) for the
    bare-symbol case, withheld (no `sorry`) otherwise, matching the existing
    `exp_of_log`/`sqrt_of_square` certificate discipline.
  - `tests/test_assumptions.py` now imports `Assumptions` from the stable
    `alkahest` top level instead of `alkahest.experimental` (the experimental
    module still re-exports it unchanged, so old imports keep working).
- **Basel-family infinite sums:** `sum_definite(expr, k, lo, hi)` now recognizes
  `hi = pool.pos_infinity()` p-series with an even power, e.g.
  `sum_definite(1/k**2, k, 1, pool.pos_infinity())` → `π²/6` (the Basel
  problem) and `Σ 1/k⁴ = π⁴/90`, via a Bernoulli-number/even-zeta table
  (`alkahest_core::sum::special`) rather than Gosper, which cannot sum `1/k^p`
  in closed form. Odd powers (`ζ(3)`, …) and any other unrecognized infinite
  bound still honestly raise `E-SUM-002` instead of guessing. `sum_definite` /
  `sum_indefinite` docstrings also no longer claim Faulhaber/geometric sums
  are unsupported — they've worked via general Gosper summation all along.

### Lean certificates

- **Definite-integral certificates now cover finite sums and constant
  multiples**, not just a single `sin`/`cos`/`exp`/`xⁿ` term: `∫ (sin x + cos
  x)`, `∫ 3·cos x`, `∫ -exp x`, and mixed multi-term combinations like
  `∫ (x² + sin x + 3·cos x)` now emit a type-checking interval-FTC proof
  (`HasDerivAt.add`/`.const_mul`/`.mul_const` and the `IntervalIntegrable`
  analogues composed over the existing base fragment). A numeric-literal
  coefficient (`Integer` or `Rational`) is required — a symbolic factor (e.g.
  `y · cos x`) and any addend outside the certifiable base fragment still
  withhold the *entire* certificate, never a partial one.

### Fixes — earlier in this cycle

- **`alkahest.SumError` now actually catches native summation errors.**
  `sum_definite` / `sum_indefinite` raise the native `E-SUM-*` exception, but
  `SumError` was missing from the native-exception overlay list, so
  `except alkahest.SumError` (or `pytest.raises(alkahest.SumError)`) silently
  failed to catch it — the only error class with this gap.
- **`eval_expr` no longer returns `nan`/`inf` as if it were a value.**
  Substituting into an expression whose denominator is zero at that point
  (e.g. `(x²-1)/(x-1)` evaluated *as written* at `x = 1`) reduces to `0 ·
  (1/0)` under plain IEEE-754 arithmetic — previously `eval_expr` handed that
  `nan` straight back as a normal-looking float. It now raises `DomainError`
  (`E-EVAL-009`) instead. `cancel()` first is still the correct way to get
  the limit: `eval_expr(cancel((x**2-1)/(x-1)), {x: 1})` legitimately
  returns `2`. The structured `evaluate(..., mode="f64")` API already
  reported this case as `status="unsupported"`; `eval_expr` (the raw
  `float`-returning entry point, and the tree-walking interpreter
  `eval_interp` it's built on internally) is now consistent with it. New
  `alkahest_core::jit::eval_interp_checked` for Rust callers that want the
  same check without a panic-on-`None` `.unwrap()`.
- **`solve(..., domain="real")` filters out complex roots.** `solve([x**2 +
  1], [x])` always returns the complex roots `±i` (`x² = -1` has no real
  solutions) — previously there was no way to ask for real solutions only
  short of manually inspecting each returned expression for an imaginary
  part. `domain="real"` (default `None`, unchanged existing behavior) now
  filters the solver's output: `solve([x**2 + 1], [x], domain="real")` →
  `[]`, `solve([x**2 - 1], [x], domain="real")` → `±1`. Composes with
  `numeric=True` and the `numeric=True` degree-limit fallback to homotopy
  continuation (which already returns real roots only).
- **Definite integrals no longer integrate through poles.** `integrate(f, x, a, b)`
  computed the antiderivative and returned `F(b) − F(a)` without checking for
  singularities of `f` inside `[a, b]`, so divergent integrals came back as clean,
  plausible, wrong values with no error raised — `∫_{-1}^{1} x⁻² dx` returned
  `-2`, `∫_{-1}^{1} x⁻¹ dx` returned `-log(-1)`, and `∫_0^2 dx/(x²-1)` returned a
  residual containing `log(-1)`. (`verification.status` was `"unverified"` for
  these, but it is also `"unverified"` for correct results, so it did not
  distinguish them.) Rational integrands are now checked for real poles on the
  closed interval — with factors shared with the numerator divided out, so
  removable singularities such as `(x²-1)/(x-1)` at `x = 1` are still accepted —
  and an improper integral raises `E-INT-001` instead of returning a value.
  Non-polynomial denominators (`1/sin(x)`) are not analysed and are unaffected,
  as are symbolic bounds, which cannot be compared against root locations.

## 3.7.0 — 2026-07-25

### Matrix / linear algebra

- **SymPy-style matrix multiply:** `A * B` is the matrix product (same as
  `A @ B`); `A * k` / `k * A` scalar-multiply; `A ** n` for non-negative
  integer powers; named `multiply` / `scalar_mul` / `hadamard` methods.
- **Symbolic 3×3 eigenvalues:** closed-form eigenvalues for parametric 3×3
  matrices whose characteristic polynomial is an irreducible cubic over the
  parameter field (Cardano / trigonometric path), not only 2×2.

### Lean certificates

- **Definite-integral certificates:** definite `integrate` emits a
  type-checking Lean proof via Mathlib's FTC / interval-integral lemmas
  (previously only indefinite integrals had certificates).
- Broader Lean coverage for differentiation (chain rule, log/sqrt/tan,
  quotient) and assumption-gated exp/log identities; certificates that do
  not typecheck are withheld rather than emitting broken proofs.

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
  silent wrong answers when `Im(z) ∉ (−π, π]`. The real-valued check now also
  accounts for branch-cut sub-terms: a non-integer power of a negative real
  (`(−20)^(1/2) = √20·i`) and `sqrt`/`log`/inverse-trig of out-of-range real
  arguments are no longer misclassified as real, so `log(exp(√(−20)))` and
  `log(exp(log(−5)))` no longer fold to a wrong principal value.

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

- **Accurate `erf`/`erfc` in f64 eval:** use libm rather than a coarse
  approximation on the numeric evaluation path.

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

### Docs / release

- Document `parse` in the README quickstart; clarify that `limit` / `series` are not `DerivedResult`.
- Expand `sum_definite` / `sum_indefinite` / `diophantine` / `solve` docs (Faulhaber gap, binary Diophantine patterns, parametric solve).
- Release wheel smoke runs the README quickstart + fresh-interpreter `parse` against the built wheel; Windows runners force UTF-8 so Lean certificates containing `∫` do not abort the smoke step.

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
