# Changelog

## Unreleased

- **The rational-function route returned answers nothing had checked, and
  nothing *could* check.** `integrate`'s Rothstein–Trager fallback was the one
  route in the integrator that returned its result directly, with no
  verification step at all. Every other route gates on `d/dx F = f`; this one
  did not, so `∫dx/(x⁵−x−1)`, `∫x¹²/(x⁹+x+1) dx`, `∫dx/(x³−2)` and every other
  rational function with an algebraic residue came back as an unchecked
  assertion. On a 92-case rational corpus, 52 of the 92 answers were in that
  state.

  Simply wrapping the route in the gate would have deleted the capability
  rather than checked it. Those answers are `RootSum` nodes, and no
  verification tier could read one: `simplify` treats a `RootSum` as an opaque
  atom so the symbolic arm can never reduce the residual to zero, and neither
  `jit::eval_interp` nor the route-level `integrate::gate::eval_at` had a
  `RootSum` rule, so the numeric arm reported "unevaluable". Declaring the
  route trusted instead was not an option either — reading a method's soundness
  as a claim about a result nobody looked at is the shape of every false
  certificate that has had to be removed from this codebase.

  So the answers were made checkable. A `RootSum` now **evaluates
  numerically**, in `eval_expr` and everywhere else: the minimal polynomial's
  roots are found by a rescaled Durand–Kerner iteration and the body is summed
  over them in IEEE-754 complex arithmetic. The sum over a conjugate-closed
  root set is real, and a residual imaginary part above rounding noise is
  reported as "cannot evaluate" rather than rounded away. With that in place
  the existing gate works unmodified, and the route is gated like every other.

  Measured before → after: the 92-case rational corpus goes from 92 solved /
  40 verified to 92 / 92, with all 52 moves being UNVERIFIED → VERIFIED at
  42/42 sample points and zero disagreements; Charlwood's Fifty (33 solved / 32
  verified), the 40-case integration probe (35 solved, 4 `E-INT-004`) and the
  110-case Liouville corpus (84 solved / 84 verified) are unchanged case for
  case. No new `E-INT-004` anywhere. Against a 60-digit `mpmath` reference the
  answers that had never been checked are correct to about fifty places, so
  none of them turns out to be wrong.

  **Behaviour to plan for.** `eval_expr` on an expression containing a
  `RootSum` now returns a number where it previously raised "expression could
  not be evaluated"; code that used the exception to detect a `RootSum` should
  inspect the expression instead.

  **What is honestly declined.** `Σ_{m(c)=0} body(x, c)` is ill-conditioned in
  the roots — a one-ulp perturbation of a *correct* `f64` root set moves the
  value by `2e−8` at degree 15 and `1e−2` at degree 21, against the gate's
  `1e−7` tolerance. Above roughly degree 14 the answer is therefore correct and
  uncheckable, and `integrate` declines it with `E-INT-001` rather than ship
  it. `∫dx/(xⁿ−x−1)` is solved through `n = 14` and declined from `n = 15`; it
  was previously "solved" for every `n` the route could reach, unverified. A
  decline there is never upgraded to `E-INT-004` — every rational function is
  elementary, and nothing in this change decides otherwise. Raising the ceiling
  needs high-precision *roots*, not just a high-precision body; `eval`'s
  `root_sum` module carries the measurements.
- **`dsolve`'s verification gate could certify a wrong `y(x)`.** Every solution
  `dsolve` returns is checked by substituting the candidate back into the
  original equation and requiring the residual to vanish — symbolically, or
  numerically at 15 samples (5 values of `x` × 3 assignments of the integration
  constants). A sample where the residual did not evaluate to a finite number
  was **skipped** as "a pole here", and the candidate was then certified on the
  remaining samples alone, subject only to a floor of six of them.

  That floor is exactly six, and skipping is what makes it reachable. For
  `y' = 0` — regular at every `x`, general solution `y = C1` — the candidate

  ```
  y = C1 + √(x − ½)·(x − 0.61)²·(x − 0.79)²
  ```

  is NaN at the samples `0.11, 0.27, 0.43` (negative radicand) and has a double
  root at `0.61` and `0.79`, so the residual is exactly zero at the two samples
  that do evaluate. Nine skips, six agreements, gate cleared, wrong answer
  returned. The mistake is treating "did not evaluate" as "no information": a
  candidate that blows up where the ODE itself is perfectly well-defined is
  evidence that the candidate is wrong.

  The residual alone cannot draw that distinction, because it conflates the
  equation's coefficients and forcing term with the candidate's contribution
  into a single expression. The gate now keeps the two halves separately and
  **classifies** a non-finite sample instead of discarding it:

  1. Probe the equation at that `x` with finite dummy values for `y, y', …`. If
     nothing evaluates, the ODE is genuinely singular there — a `√(a − x)`
     coefficient past its branch point, a pole in the forcing term — and the
     sample really does carry no information. Skip.
  2. Otherwise evaluate the candidate and its derivatives on their own. Not
     finite, where the ODE is regular, is now a disagreement.
  3. If both sides are finite and only the *simplified residual* was not, the
     non-finiteness was an artefact of the residual's algebraic form; the
     verdict comes from the original equation at the candidate's own values,
     recovering a sample that used to be thrown away.

  `E-ODE-011` now states the tally: agreeing, disagreeing, blow-ups at a regular
  point, and each kind of skip.

  The direction of the trade is deliberate: a decline is acceptable, a wrong
  `y(x)` is not. For a *nonlinear* ODE a correct solution may have a movable
  singularity at a regular point (`y' = 1 + y²` has `y = tan(x + C)`), and a
  sample landing on one is now a decline rather than a skip.

  **No capability change.** The ODE corpus is unchanged at 89/101, case for
  case, and the classifier is never entered on it — all 15 samples of all 35
  numerically-certified solutions are finite — so no evaluation work is added
  and latency is unchanged. All 12 remaining declines are "no class produced a
  candidate"; none is the gate refusing a correct answer (`cargo test
  -p alkahest-cas ode::dsolve::corpus::decline_split_report -- --ignored
  --nocapture`).

  Two notes for anyone reasoning about this gate. The numeric fallback is not a
  rarity — it is what certifies 35 of the 89 solved corpus cases, the symbolic
  branch closing the other 54. And it is **not** blocked by a non-elementary
  answer: `verify::eval` has no `f64` kernel for `Ei`/`Si`/`Ci`, but those
  cancel out of the residual when the candidate is substituted, leaving
  elementary leftovers such as `x⁻¹·eˣ·e⁻ˣ − x⁻¹` that the sampler evaluates
  perfectly well. `y'' − y = 1/x`, `y'' − y = eˣ/x`, `y'' − 4y = 1/x` and
  `y''' − y' = 1/x` are all certified numerically, not symbolically, so the
  conflated residual is kept as the primary check — evaluating the split form
  instead would lose them.

- **`ak.simplify` refused expressions it could handle, and crashed on one it
  could not.** The `E-DEPTH-001` ceiling (`MAX_EXPR_DEPTH`, 2 048) is
  calibrated against the shallowest walker that recurses without a net —
  `symbolic_grad`, which segfaults at 4 687 — and was then applied uniformly at
  the PyO3 boundary. That has been costing capability since the simplification
  traversals stopped needing it: `simplify::engine` and `simplify::parallel`
  run under the segmented-stack trampoline, and `simplify::redex` schedules the
  DAG level by level and never recurses at all. Twelve entry points —
  `simplify`, `simplify_par`, `simplify_redex`, `simplify_auto`,
  `simplify_expanded`, `simplify_trig`, `simplify_trig_normal_form`,
  `simplify_with`, `simplify_strategy`, `collect_like_terms`, `simplify_pauli`,
  `simplify_clifford_orthogonal` — now accept input of any depth. Measured on
  the release build with the usual 8 MiB main-thread stack, a 100 000-level
  `sin` chain: `simplify` 0.09 s and 163 MiB, `simplify_par` 0.10 s,
  `simplify_redex` 0.09 s, `simplify_auto` 0.07 s. Previously all four raised
  at 2 049.

  The ceiling was not pure ceremony there, though, and this is not a removed
  check. Two plain recursions are still reachable from those entry points, both
  confirmed to kill the process rather than raise:

  - Every default simplification pass ends in the assumption-gated **colored
    e-graph**, which runs whenever the expression carries a `Domain.Positive`
    or `Domain.NonZero` symbol. `ColoredEgraph::from_expr` and `rebuild`
    descend one native frame per level: `SIGSEGV` between 60 000 and 100 000
    levels. It is also quadratic in node count (5 000 levels: 4.8 s; 20 000
    levels: 100 s), so putting it on the trampoline would have traded a crash
    for a hang.
  - `alkahest-py` renders a returned derivation log with the same recursive
    printer `str()` uses. A deep chain wrapped in a single redex — trampolined
    traversal, one very deep step in the log — segfaulted at 30 000.

  So the guard is now per *route* rather than per operation. A new
  `alkahest_cas::simplify::check_simplify_depth` applies `MAX_EXPR_DEPTH` only
  to inputs that would reach the colored pass, deciding by running the
  simplifier's own `collect_static_domain_facts` so that the guard and the
  thing it guards cannot drift apart. It is `O(1)` at or under the ceiling, so
  the hot path costs exactly what it did before; only an expression already
  past 2 048 levels pays the one extra iterative walk. `make_derived_result`
  records an over-deep step's *depth* in place of its text, which is visible in
  the `derivation` string rather than silent, and cannot affect any result
  shallow enough to print. `E-DEPTH-001` keeps its meaning wherever it still
  fires: same code, same `limit`, same reason.

  Two entry points that could already reach the colored recursion had **no
  depth guard at all** and would segfault on a deep input: `simplify_with`, and
  `AssumptionContext.simplify` (which is what `ak.simplify(expr,
  assumptions=…)` and `with ak.context(assumptions=…)` route through). Both are
  guarded now — the latter unconditionally, because explicit facts send every
  input through the e-graph whatever its symbols' domains are.

  **Behaviour to plan for.** `ak.simplify(deep)` no longer raises
  `DepthLimitError` for a deep expression over ordinary symbols, and
  `simplify_many` no longer reports one on the item; code branching on that
  refusal will now take the success path. Conversely, `ak.simplify(deep,
  assumptions=ctx)` now raises where it previously took the process out.

- **`simplify` could not cancel `¾·sin x − ¾·sin x`, and that cost the
  verification gate its strongest verdict.** `collect_add_terms` split each
  summand into a coefficient and a base using an *integer*-only extractor, so
  the two terms above had different bases (`¾·sin x` and `−¾·sin x`) and never
  met in the coefficient map. Integer coefficients cancelled; every
  non-integer ratio did not. Coefficients are now `rug::Rational` throughout —
  exact arithmetic, no approximation — which also closes like terms over an
  irrational constant (`√3·(−1/32) + √3·(1/32) → 0`), the same root cause seen
  from a different angle. Merging works as well as cancelling: `x/2 + x/3`
  is now `(5/6)·x`. The guard that refuses to drop a zero-coefficient term
  containing `0^(−n)` is unchanged and still applies.

  Separately, `simplify` did not distribute a leading `−1` over an `Add`. The
  gate builds its residual as `d/dx F + (−1)·f`; when `f` is a sum, the
  negation stayed wrapped and the terms it was meant to cancel against sat at
  a different `Add` level where like-term collection could never see them. A
  new `distribute_neg_over_add` rule pushes the negation through. It is *not*
  `ExpandMul` in miniature and is not gated behind `SimplifyConfig::expand`:
  distributing a general factor grows the expression and fights factoring,
  whereas `−1` is absorbed into each term's existing numeric coefficient, so
  the term count is unchanged and the rewrite is a strict normal-form
  direction. Only the two-factor product `(−1)·S` fires; `(−1)·y·(a + b)` is
  real expansion and is still left to `ExpandMul`.

  Measured on a 65-integrand corpus spanning the families the integrator
  advertises (`alkahest-core/tests/gate_verdict_census.rs`), the gate's verdict
  distribution moves from **22 `Proven` / 33 `SampledOnly`** to **30 `Proven` /
  25 `SampledOnly`**, with no `Failed` and no new declines: eight
  antiderivatives that were only ever checked at sample points are now backed
  by a symbolic identity. `poly::cancel` no longer refuses a genuine fraction
  with `NonIntegerCoefficient` either — `MultiPoly` is a polynomial over ℤ, but
  a `RationalFunction` represents `p/q` exactly, and the literal now routes
  there.

  Downstream, one certificate shape class moves: `∫ x²·cos(x) dx` was withheld
  because the by-parts answer came back as `x²sin x + (−1)·(x·cos x·(−2) +
  2 sin x)`, an undistributed negation the Lean emitter could not close. It now
  comes back as `x²sin x + 2x cos x − 2 sin x` and certifies. The checked-in
  ledger is regenerated accordingly.

  **`simplify` is faster, not slower, for this.** The obvious implementation —
  collect every coefficient as a `rug::Rational` — costs about 9% of
  `simplify`'s wall time, because `mpq_add` canonicalizes through a GCD on
  every addition and carries a denominator limb that is almost always `1`. A
  small internal `Coeff` type keeps integer coefficients on `mpz` and escalates
  only for a genuine fraction; the reject paths of the two new rules allocate
  nothing. Measured with an interleaved, rotating-order, min-of-18 A/B (so a
  loaded machine moves both arms equally): a Jacobian-shaped corpus in which no
  rule fires is **−5.7%**, re-simplifying an already-normal expression is
  **−7.9%**. Expressions that the new rules actually rewrite cost more (+23%
  for rational coefficients, +43% for negated sums) because they now do the
  work that produces a cancelled answer.

- **`√(u^(2k))` is `|u|^k`, and is now reduced only where the absolute value is
  provably redundant.** The general identity is not `u^k`: a blanket rewrite
  makes `√((−3)²)` return `−3`. The new `sqrt_of_even_power` rule fires in
  exactly two cases — `u` structurally non-negative (a `Domain::Positive` /
  `Domain::NonNegative` symbol, a non-negative literal, `|·|`, `exp`, `cosh`,
  an even power of a real, or a sum/product of those), or `k` even and `u`
  structurally real, where `|u|^k = u^k` because an even power of a real is
  already non-negative. So `√(x⁴) → x²` for real `x` with no sign hypothesis,
  while `√(x²)` is left alone. Every complex-domain base declines: for complex
  `z`, `√(z²)` is `±z` depending on branch and `|z|` is not even the right kind
  of answer. Both spellings are handled (`sqrt(r)` and `r^(1/2)`).

  This is deliberately weaker than what an explicit assumption buys you: with
  an `AssumptionContext` carrying `x > 0`, the colored e-graph's
  `sqrt_of_square_positive` already fired on symbols this rule must refuse, and
  still does. Either way the hypothesis is *recorded*: the step carries
  `InDomain(u, NonNegative)` or `InDomain(u, Real)` as a `SideCondition`, so a
  reader of the derivation log is told what the rewrite rested on rather than
  having to infer it.

- **The spelling of a by-parts residual decided whether it integrated.**
  `∫x/((1−x²)·√(1−2x²))` — the residual Charlwood #49 reduces to — closes in
  milliseconds when written that way, and declines with `algebraic integrator
  requires exactly one sqrt(P(x)) generator` when written the way `diff`
  spells it. Composite arguments such as `asin(x/√(1−x²))` throw off several
  radical generators, so the residual almost never arrived in the one form the
  algebraic engine accepts. `by_parts` now normalises a residual's radicals
  before handing it on — seeing through `(a·b)^k` and `(b^m)^k`, combining
  radicals, splitting `√(N/D)` into `√N·√D⁻¹` with a numerically-ranked
  orientation, and pulling repeated polynomial factors out of a radicand via
  FLINT (`√(x + 2x² + x³) → (1+x)·√x`). These are offered to the existing
  engine as alternative *spellings*; no new integration rule is introduced and
  nothing new can be certified non-elementary. Charlwood's Fifty goes from 30
  to 33 (`asin(x/√(1−x²))`, `atan(√(1+x)−√x)`, `asin(x)/(1+x²)^(3/2)`), and a
  25-case composite-argument corpus from 12 to 17, every answer checked by
  differentiation. The pass is gated on a residual having two distinct
  variable-dependent radicals or one with a non-polynomial radicand, which
  keeps the cost off the common decline path: median decline latency over a
  110-case corpus moves 124 ms → 128 ms, worst case 2244 ms → 2203 ms.

- **A Lean certificate cited a theorem it had not stated.** The kernel folds
  `-e` into `Mul[e, -1]`, and the Lean printer rendered that literally, so the
  `Filter.Tendsto` certificate for `exp(-x) → 0` emitted the goal
  `Tendsto (fun x => rexp (x * -1)) atTop (𝓝 0)` while citing
  `tendsto_exp_neg_atTop_nhds_zero`, which proves the `rexp (-x)` form. Lean
  rejected it, so nothing unsound was ever handed out — but the emitter's
  contract is that what it emits typechecks, and `Verify Lean 4 proofs` has
  been red on `main` since the Tendsto certificates landed. The Tendsto
  emitter now renders a lone `-1` factor as a negation of the remaining
  product. That rendering is deliberately *not* the default: the `diff` and
  definite-integral emitters pair their printed goal with witness terms built
  as strings in the matching `c * f` shape (`.const_mul ((-1 : ℝ))`), and
  printing `-e` there leaves Lean unifying `-rexp x` against `-1 * rexp x` by
  defeq until it exhausts the `whnf` heartbeat budget. Their output is
  byte-identical to before. The existing test asserted only that the emitter
  *named* the theorem, never that the printed goal was the theorem's
  statement; it now checks both.

  `Tier 1a` was red on `main` for the same stretch and from the same commit,
  for an unrelated reason: `ruff format --check` failed on three of the test
  files added with the Tendsto work. Reformatted; the changes are whitespace
  only.

- **`limit()` now returns `DerivedResult`**, matching `diff` / `integrate`, so
  `.certificate` can carry a Lean `Filter.Tendsto` proof. Recognised `x → +∞`
  patterns (`exp(-x) → 0`, `xⁿ exp(-x) → 0`, `exp(x) → +∞`, a crude exp-ratio)
  emit Mathlib Tendsto source; unrecognised shapes — including finite and
  one-sided limits — withhold rather than emitting `sorry`. Use `.value` for
  the limit expression. Nested calls still work via `_coerce_expr`.
- **SOS positivity certificates are now in the Lean CI corpus.**
  `PositivityCertificate.to_lean()` already emitted sorry-free Lean (`ring` +
  `positivity` / `nlinarith`); those sketches are now generated from real
  `sos_decompose` / `prove_nonneg` results and typechecked against the pinned
  Lean 4.9 / Mathlib v4.9.0 toolchain, with the same withhold-rather-than-sorry
  discipline as the derivation corpus.
- **`dsolve` closes variation of parameters at every order, Euler–Cauchy with
  forcing, and second-order equations with variable coefficients — and the
  thing that was actually blocking it was not the integrator.**

  Measured first, on a 101-ODE corpus (`ode/dsolve/corpus.rs`, test-only) run
  at three revisions. The recent integration work — router fall-through,
  general by-parts, Risch–Norman, the residue-theorem route, ten new special
  functions — moved `dsolve` from **59 to 60**. One ODE.

  The reason: `dsolve` builds its own integrands and was handing them over in a
  spelling no integrator should have to recognise. `μ = exp(∫p dx)` was emitted
  as a literal `exp` node, so the linear class asked for `∫q·e^{−log x} dx`
  when it meant `∫q/x dx`; `μ·q` arrived as `x·e^{−x}·e^{x}` when it meant `x`;
  a Wronskian of `{cos, sin}` arrived as `cos²x + sin²x` instead of `1`. 14 of
  the 41 declines were elementary integrals, misspelled. Fixing that alone —
  folding `exp(c·log u + rest) → u^c·exp(rest)` when building an integrating
  factor, and trying each integrand in several equal-valued spellings — took
  the corpus to **68**.

  On top of that:

  - **Variation of parameters at arbitrary order**, by Cramer's rule on the
    Wronskian (`ode/dsolve/variation.rs`), replacing the second-order-only
    formula. `y''' + y' = sec x` and `y''' + y' = tan x` now solve; no
    undetermined-coefficients ansatz can express either forcing.
  - **Euler–Cauchy with a right-hand side.** It previously declined for every
    non-zero `r(x)`; it now builds the basis of powers of `x` (or the
    repeated/complex forms) and closes the forcing by variation of parameters.
  - **General variable-coefficient second order**, previously unreachable: one
    homogeneous solution from a short ansatz list, the second by reduction of
    order, the forcing by variation of parameters. This is what admits
    Legendre and other equations with a polynomial coefficient on the second
    derivative.

  **82 of 101**, from 59 before the integration work and 60 after it. Every
  solution is still gated on substitution back into the original equation, and
  the particular solution allocates no constants of its own — the general
  solution has exactly `order`-many, pinned by a test.

  **On the merged base that reads 89 of 101**, and the seven split cleanly.
  Four are `main` moving underneath the branch — the special-function emitters
  becoming reachable closes `y'' + y = log x`, `y'' − 4y = 1/x`,
  `y' + y = 1/x` and `y'' − 3y' + 2y = 1/(1 + e^{−x})`. The other three were
  never declines at all: `y'' − y = 1/x`, `y'' − y = eˣ/x` and
  `y''' − y' = 1/x` came back `VERIFY_FAIL`. `dsolve` had found the answer and
  its own gate threw it away, because the residual only cancels once
  `1/(2x) − 1/(2x)` is collected over ℚ — the `collect_add_terms` fix above.
  `y'' − y = 1/x` now returns

  ```text
  C₁·eˣ + C₂·e^{−x} + ½·eˣ·Ei(−x) − ½·e^{−x}·Ei(x)
  ```

  which is the general solution, checked here by re-substitution and
  independently by differentiating it at 30 digits.

  So the **special-function-quadrature class is closed**, and the test that
  asserted it was not has been split rather than weakened.
  `quadrature_over_the_special_function_basis_closes` pins the solved cases
  (`Ei` for `y'' − y = 1/x`, `Si`/`Ci` for `y'' + y = 1/x`); the decline path
  is kept on `y'' + y = x/(1+x²)`, which needs `∫x·sin(x)/(1+x²) dx`, and
  since `x/(1+x²) = ½[1/(x−i) + 1/(x+i)]` that is `Si`/`Ci` at the complex
  arguments `x ± i` — out of reach of the deliberately real-only `expint`
  kernels *and* of an emitter table that wants a linear denominator. The
  comment there names both conditions, so the next person to close one is
  told the test's premise has expired rather than finding out by mystery
  failure.

  Removed `integrate_pexp_trig`, the hand-rolled undetermined-coefficients
  fallback for polynomial × exponential × sinusoid antiderivatives. General
  by-parts covers that family: instrumented, it fired **0 times** across the
  whole corpus, and the ODE suite is green without it.

  **What `dsolve` still needs from `integrate`** (probe:
  `cargo test -p alkahest-cas ode::dsolve::corpus::integrator_gap_probe --
  --ignored --nocapture`). Re-run on the merged base, most of the list this
  entry originally carried has closed: `∫e^{ax}/x dx` is `Ei(ax)`,
  `∫sin(x)·log(x) dx` is `Ci(x) − log(x)·cos(x)`, and `∫e^{2x}/(1 + e^x) dx`
  no longer depends on whether it is spelled `exp(2x)` or `exp(x)^2` — both
  give `eˣ − log(1 + eˣ)`. Two remain:

  - `∫sin(x)/x² dx` is still answered with a `NonElementary` *certificate*.
    The certificate is correct — no *elementary* antiderivative exists — but
    `−sin(x)/x + Ci(x)` is over the registered basis and no matcher finds it.
    `integrate::special`'s module docs name this one as the standing
    counterexample to re-reading that verdict as the stronger claim.
  - `∫sin x·tan x dx` declines, while the same function multiplied by a
    redundant `1` spelled `cos²x + sin²x` closes. `dsolve` currently keeps the
    un-normalised spelling of every integrand alive precisely because of this.

  Two `simplify` gaps also had to be worked around from inside `ode/`: a
  one-element `Mul` is not collapsed (so a singleton `Mul` holding `−1` is not
  the same node as `−1`, and `x·x⁻¹` survives), and `x²·(−1·x²)⁻¹` does not
  cancel because the `Pow` wraps a whole `Mul`.
- **Pointwise derivative certificates for `sinh`, `cosh`, `atan`, and `asin`.**
  Hyperbolic sine/cosine are unconditional on ℝ (`Real.deriv_sinh` /
  `Real.deriv_cosh`) and join the everywhere-differentiable sum/product
  fragment, so `sinh x + cosh x` and `exp x · sinh x` certify too. `atan`
  is unconditional (`Real.hasDerivAt_arctan'`, reconciling Alkahest's
  `(1+x²)⁻¹` with Mathlib's `1/(1+x²)`). `asin` carries an explicit
  `(x : ℝ) (hx : -1 < x ∧ x < 1)` binder, closed by
  `Real.hasDerivAt_arcsin`. Composites (`asin(x²)`, …) stay withheld.

- **Pointwise `d/dx tanh x` now emits a Lean certificate.** Mathlib v4.9.0 has
  no `hasDerivAt_tanh` and no `1 - tanh² = 1/cosh²` analogue of
  `Real.inv_one_add_tan_sq`, so the earlier emitter withheld. The certificate
  constructs the derivative from `Real.hasDerivAt_sinh` / `Real.hasDerivAt_cosh`
  via `HasDerivAt.div`, using `Real.cosh_pos` (`cosh x ≠ 0` is free on ℝ, not a
  binder) and `Real.cosh_sq_sub_sinh_sq` to reconcile Alkahest's `1 - tanh(x)²`
  with the quotient-rule `1/cosh²`. Composites `tanh(x²)` stay withheld.
  This is pointwise only and does **not** join the everywhere-differentiable
  simp set.

- **More `Filter.Tendsto` certificates from `limit()`.** The emitter now
  matches sorry-free Mathlib v4.9.0 lemmas beyond the `x → +∞` exponential
  fragment: `sin x / x → 1` as `x → 0` (`hasDerivAt_iff_tendsto_slope` of
  `Real.hasDerivAt_sin`), `(1 + 1/x)^x → e` (`tendsto_one_plus_div_rpow_exp`),
  `x^n → +∞` (`tendsto_pow_atTop`), `1/x → 0` (`tendsto_inv_atTop_zero`), and
  `x → +∞` (`tendsto_id`). Leading-term polynomial *ratios* stay withheld —
  they share a ledger shape with extra-term rationals the emitter does not
  prove. The ledger's limit classifier now records the approach point
  (`atTop` / `zero` / `finite`) so those Tendsto domain filters do not mix
  certified and withheld observations. The Tendsto printer continues to use
  `expr_to_lean_neg` only; the default `expr_to_lean` is unchanged.

- **FTC for `∫ dx/(1+x²) = atan x`**, indefinite (via the reused
  differentiation certificate) and definite (`HasDerivAt` witness
  `hasDerivAt_arctan'`, `IntervalIntegrable` of the continuous integrand
  `(1+x²)⁻¹`). The definite certificate proves `arctan b − arctan a`, not
  `π/4`.
- **Pointwise Lean certificates for negative integer powers of the
  differentiation variable.** `d/dx x⁻ⁿ` needs `x ≠ 0`, which `deriv_pow`
  cannot discharge, so the emitter now binds `(x : ℝ) (hx : x ≠ 0)` and
  closes the pretty-printed `(x)⁻¹` / `(x)⁻¹ ^ (k : ℕ)` spelling via
  `hasDerivAt_inv` (and `HasDerivAt.pow` for `n ≤ -2`). No `sorry`.
  `∫ x⁻¹ dx = log x` certifies via the existing FTC reuse (`Real.deriv_log`).
  Combine steps (`product_rule` / `sum_rule`) on bodies built from
  `{wrt, constants, wrt⁻ⁿ}` now bind `(x : ℝ) (hx : x ≠ 0)` and close with
  `deriv_inv` / `differentiableAt_inv` (not the unconditional simp set), so
  `∫ x⁻² dx = -x⁻¹` certifies via FTC reuse of `d/dx (-x⁻¹)`.
- **`log` and `sqrt` now certify inside `product_rule` / `sum_rule`.** The
  combine tactic was unconditional-only (`deriv_sin`/`cos`/`exp` with no
  hyps), so `d/dx (x log x)` was withheld even though pointwise `d/dx log x`
  already certified. A second fragment threads `(hx : 0 < x)` and closes
  via `Real.hasDerivAt_log` / `Real.hasDerivAt_sqrt`; `sin·exp` stays on the
  old path with no extra binder. Composites (`log(x²)`, `log(sqrt(x²−1)+x)`)
  stay withheld — general `HasDerivAt.comp` is a different encoding. Do not
  dump `Real.deriv_log` into the unconditional simp set: `deriv_mul` still
  needs `DifferentiableAt log`, which requires `x ≠ 0`.

  Because `d/dx (x log x − x)` intern-equals `log x`, the existing FTC path
  now certifies the indefinite integral `∫ log x dx = x log x − x`.
  Definite `∫_a^b log x` certifies when both endpoints are strictly positive
  (`0 < a`, `0 < b` — numeric literals via `norm_num`, or explicit binders),
  via `intervalIntegral.integral_eq_sub_of_hasDerivAt` with
  `hasDerivAt_mul_log` and `intervalIntegrable_log` /
  `Set.not_mem_uIcc_of_lt`. `∫_0^1 log` (singular at 0) and negative
  endpoints stay withheld. `∫ x log x` and
  `d/dx` of its antiderivative stay withheld: differentiating
  `½ x² log x − x²/4` produces an n-ary inverse cancellation
  (`x² · x⁻¹ · ½ = x · ½`) that `ring` cannot close, and the two-factor
  `field_simp` encoding does not cover a spectator coefficient. A withheld
  certificate beats a `.lean` file that fails under `warningAsError`.
- **Chain-rule Lean certificates now cover linear and primitive inners**, not
  just `f(xⁿ)`. `d/dx sin(x²)` already closed via `hasDerivAt_pow` +
  `HasDerivAt.sin`; the same encoding now names an inner `HasDerivAt` for
  `c·x` / `x+c` / `a+bx` (`.const_mul` / `.mul_const` / `.add_const`) and for
  a pointwise `sin`/`cos`/`exp` of the variable (`Real.hasDerivAt_cos` and
  friends), then lifts through the outer `.sin`/`.cos`/`.exp`. That unlocks
  `cos(2x)`, `exp(-x)` (interned `x * -1`, not `hasDerivAt_neg'`),
  `sin(cos x)`, and `exp(-x²)` (`hasDerivAt_pow` scaled by `-1`). Nested
  two-deep composites (`sin(cos(x²))`) and `log`/`tan`/`sqrt` chain still
  withhold.

  The FTC fragment now reuses a combine tactic for `{const} * f(linear)`
  (`f ∈ {sin,cos,exp}`, binder-free linear/affine inner) via
  `HasDerivAt.const_mul` / `.mul_const` — the same scaling the definite
  constant-multiple arm already used — rather than widening
  `diff_body_unconditional` to all products. That closes `∫ cos(2x) dx`
  (indefinite, via `deriv F = f` with `F = (1/2) sin(2x)`) and
  `∫₀¹ cos(2x)` (interval FTC, `HasDerivAt.sin` of the linear inner then
  `.div_const 2`). `∫ x·exp(x²)` stays withheld: its antiderivative is
  `{const} * exp(x²)`, a power inner outside this arm.

- **Gosper sums now emit sorry-free Lean `Finset.sum` certificates.** The
  kernel already checks `G = R·F` in `ℚ(k)`; emitting the rewrite log as
  `F = G` would be false (e.g. `k = k(k−1)/2`). The new `emit_gosper_cert`
  states the discrete FTC `G(k+1) − G(k) = F(k)` and the telescope
  `∑_{k=a}^{b} F(k) = G(b+1) − G(a)` on `Finset.Icc`, discharged by Mathlib
  4.9 `sum_range_sub` / `sum_Ico_eq_sum_range` / `Nat.Ico_succ_right`.
  Polynomials (`Σ 5`, `Σ k`, `Σ k²`, `Σ (2k+1)`), the classic
  `Σ 1/(k(k+1))` telescope, and `Σ 2^k` certify; Basel / `gamma` / `n!`
  via gamma stay withheld. The identity product `∏_{k=1}^{n} k = n!` is
  certified separately via `Finset.prod_Ico_id_eq_factorial` (v4.9.0 has
  no `prod_Icc_id`).

- **A budget could be outrun by a single allocation, and was.**
  `alkahest.integrate` on `1/(x·log x·(1 + log²(log x)))` — the derivative of
  `atan(log(log x))`, an ordinary two-level log tower — ignored a
  `Budget(wall_ms=3000)` completely and died on a 4 GiB allocation, after
  reaching 26 GB of resident memory in ten minutes on the machine this was
  found on. `handle_alloc_error` aborts without unwinding, so the wall clock,
  `request_cancel` and `catch_unwind` were all equally powerless.
- **General integration by parts (`alkahest_cas::integrate::by_parts`), behind
  its own entry point.** `∫u·dv = u·v − ∫v·du` with a LIATE choice heuristic, a
  cycle detector that *solves the linear equation* rather than recursing, and a
  growth check that backs out of a wrong split. Before this there was no
  `by_parts` rule anywhere in `integrate/`: the three existing by-parts helpers
  in `engine.rs` are each pinned to one shape (inverse-trig with the argument
  *exactly* the variable, polynomial × trig with a linear argument, the
  `exp·sin` closed form), and everything else hit `Node::Mul`'s
  `irreducible product of var-dependent factors` decline.

  The module is **not yet wired into `integrate`** — it is reached through
  `integrate_by_parts(expr, var, pool)` / `try_by_parts(…)`, the same way a new
  route is characterised before it joins the default order.

  - **The outcome type has two variants, `Solved` and `Declined`, and there is
    no path from this module to `NonElementary`.** A by-parts failure means "my
    split did not work", never "no antiderivative exists". `Declined` converts
    to `E-INT-001` and to nothing else, pinned by a test.
  - **Every returned antiderivative has passed `d/dx F = f`**
    (`verify_antiderivative_status`) before it is returned, so a wrong LIATE
    guess costs CPU and nothing else.
  - **Cycles are closed, not just detected.** `I = acc + mult·c·I` is solved for
    `I`. `∫sin(log x) dx = (x·sin(log x) − x·cos(log x))/2` is closed this way
    and by nothing else in the codebase.
  - Measured on Charlwood's Fifty: closes **#21** `x³·asin(x)/√(1−x⁴)`,
    **#22** `x³·asec(x)/√(x⁴−1)` and **#35** `x·asec(x)/√(x²−1)`, all verified
    by differentiation. The other 14 of the 17 C1/C3 problems produce *correct*
    by-parts residuals that the downstream engine then declines; the binding
    constraint on that cluster is the algebraic engine, not the by-parts layer.
  - Cost on integrands it fails on — the price paid on every decline —
    **5.1 ms mean, 13.9 ms worst** in a release build over the 40-case probe's
    six `E-INT-001` cases plus eight controls. The 40-case probe is unchanged
    at 27 solved / 7 `E-INT-004` / 6 `E-INT-001`.
  - Three normalisers the module needs and `simplify` does not provide are
    local to it, each a *proposal* re-checked by the gate: merging radicals
    (`√(1−x⁴)/√(1−x²) → √(1+x²)`), rationalising radical denominators
    (which collapses the algebraic engine's Euler-substitution answers), and
    combining like terms over *rational* coefficients — `simplify` leaves
    `sin(x)·3/4 + sin(x)·(−3/4)` standing and `poly::cancel::cancel` refuses it
    with `NonIntegerCoefficient`.

- **`Add` and `Mul` are flat at construction, so associativity holds
  structurally.** `ExprPool::mul` / `ExprPool::add` now splice nested
  same-operator children before interning, so `(a·b)·c`, `a·(b·c)` and `a·b·c`
  are one expression rather than three. This is the last of the three
  dimensions of the form-robustness bug fixed in this release (after route
  fall-through and exponent spelling); previously `parse("x*y*z")` produced the
  left-associative `(z * (x * y))` while `pool.mul([x, y, z])` produced a flat
  three-child node, and the two compared unequal.

  The growth was not diffuse. Under an instrumented allocator every large
  request came from one place: `integrate::risch::tower::decompose_log_inner`,
  growing a single `Vec<ExprId>` of log-polynomial coefficients. It reads the
  exponent off each factor of a product and adds it to a running degree, and
  when the tower generator appears in a *denominator* (`log(x)^-1`, which is
  most of this integrand) that degree is **negative**; `deg = log_power as
  usize` then wraps it to ≈ 2⁶⁴ and `while coeffs.len() <= deg { push }` runs
  until the allocator gives up. One step, no loop back to any checkpoint —
  which is exactly the shape `budget::check` cannot bound, however short the
  deadline.

  `budget::check_growth(units)` is the size half of the mechanism. A step that
  is about to grow a structure says so *before* growing, and is refused if
  `units` is more work than the budget allows; it also performs every check
  `check` does, so one call covers both halves. It applies
  `budget::DEFAULT_MAX_GROWTH_UNITS` (2²⁶) **even when no `Budget` is active**,
  because the failure it prevents is an abort rather than a slow answer and a
  caller cannot opt into a limit for an allocation they did not know a call
  would make. `Budget { max_steps: Some(n), .. }` replaces that default, which
  is what makes `E-BUDGET-002`'s existing remediation ("raise
  `Budget(max_steps=...)`") true of a growth refusal and not merely plausible
  — and why this needed no new error code and no new variant on the exhaustive
  public `BudgetError` (compare the `[[budget]]` carrier `IntegrationError`
  already uses to avoid the same break).

  The checkpoint sits at the three places `decompose_log_inner` grows its
  coefficient list. Because that function reports failure as a bare `None`,
  which also means "this is not a polynomial in log(h)", the refusal travels
  out of band (`tower::take_decompose_budget_trip`, the same pattern
  `calculus::series::take_series_refusal` uses) so both of its callers can
  re-raise it as `E-BUDGET-*` instead of reporting a resource limit as a
  mathematical verdict.

  The reproducer now returns in **47 µs** with `E-BUDGET-002`, having interned
  19 pool nodes. Making that integral *work* is a separate question — the
  negative-degree wrap above is a real defect in the decomposition and is left
  for the Risch owner — but it now stops when asked. Ordinary declines are
  unaffected: `∫ exp(x)/x dx` still returns `E-INT-*`, asserted alongside.

- **`simplify` could abort the process on a deep enough expression.** The
  sequential bottom-up traversal (`simplify::engine`'s `simplify_node`, and its
  discrimination-net twin `simplify_node_indexed`) recursed once per level of
  the expression with nothing bounding the descent. At a measured 10 832 bytes
  of stack per level in the fattest configuration this crate is built in, an
  8 MiB stack ran out around 750 levels; in release, a few hundred bytes a
  frame put the cliff at order 10⁴–10⁵. A stack overflow is not an error: it
  aborts the process without unwinding, so no `Result`, no `catch_unwind` and
  no `Budget` could have intervened. `simplify::parallel` had carried a
  segmented-stack governor since the previous entry below; the sequential
  path, which every `simplify` call goes through, had nothing — and that
  path's own test comment conceded it.

  That governor is now `simplify::stack`, shared by both traversals and no
  longer behind `--features parallel`. When the current thread's segment has
  spent either of its budgets — a count of recursion levels entered, or a
  measurement of stack bytes consumed — the next level continues on a freshly
  spawned 16 MiB thread. Depth is therefore bounded by how many threads the OS
  will hand out rather than by any one stack.

  **A depth *limit* was considered and rejected.** `simplify` has no `Result`
  return type, so a limit would have had to either change a widely used public
  signature or follow the existing budget-trip precedent and return the
  best-effort value simplified so far. The second is honest for a budget trip
  (the caller asked to stop) and misleading for a depth abort (the caller
  asked for a simplified expression and would silently receive a partly
  unsimplified one, indistinguishable from a fixed point). The trampoline
  truncates nothing, so the question does not arise and no new `E-*` code was
  needed.

  Two details the sequential path needed that the parallel one did not.
  `ExpandPow`'s declined-expansion record is thread-local and drained once per
  pass by the caller, so a traversal deep enough to spill onto a segment
  thread recorded declines where that drain would never see them — the segment
  root now drains its own and carries them home in the returned log.  And
  `simplify::assumptions::collect_static_domain_facts`, which `simplify_with`
  runs on the *result* of every simplification (so it sees whatever the rules
  could not shrink), was a second unbounded recursion one function further
  along the same path and put the abort straight back. It is now an explicit
  worklist: a walk that only collects has nothing to compose on the way back
  up, so depth costs heap instead of stack and no thread is spawned at all.

  Regression tests drive a 3 000-level chain (four times past the old cliff)
  through both traversals on a deliberately small 1 MiB thread, so they do not
  depend on the harness's stack size. Measured cost on the hot path: none
  detectable — interleaved runs of `perf_simplify_hot_path` give a median
  81.4 ms before and 81.8 ms after, and the 608 `integrate::` unit tests run in
  the same time to within run-to-run noise.

- **The nightly `asan` shard's stack overflow: `simplify_par`'s stack governor
  read `0` under AddressSanitizer, however deep it went.** Every scheduled CI
  run since 2026-08-19 died ~16 minutes in with `AddressSanitizer:
  stack-overflow`, in `simplify::parallel`'s
  `par_survives_deep_chain_on_worker_thread`.

  `simplify_par` recurses (`simplify_node_par` → `simplify_children_par` →
  `seq_children` → `simplify_node_par`), and rayon workers get a 2 MiB stack,
  so the traversal carries a governor that continues on a freshly spawned
  16 MiB thread before running out. The governor's trigger was
  `stack_used()`, which takes the address of a local and subtracts it from a
  per-thread baseline. Under ASan that local does not live on the stack:
  stack-use-after-return detection moves locals whose address escapes into a
  per-thread *fake stack* ring, whose addresses **ascend** with recursion
  depth and then wrap. Ascending addresses take `stack_used`'s re-baseline
  branch on every call, so it returned `0` at every depth; after a wrap it
  returned a bounded value that never approached the 12 MiB budget. The
  governor never fired, and the real stack — which had been filling up all
  along at a measured 10 832 bytes per level — ran out.

  The previous mitigation, `RUST_MIN_STACK=33554432` on the shard, could not
  have worked: a probe that reports `0` reports `0` at any stack size.

  `with_stack_segment` now refills on **either** of two conditions: an exact
  count of recursion levels entered since the current segment began
  (`SEGMENT_DEPTH` against a budget derived from the byte budgets by
  `WORST_CASE_LEVEL_BYTES = 16 KiB`, ~50% headroom over the fattest measured
  configuration), or the old byte measurement. The count is what bounds the
  recursion, because it cannot under-read; the byte probe is kept as a
  backstop for frames fatter than the count was calibrated for, and is now
  documented as advisory — it can only make the governor refill *earlier*,
  never later. A tight foreign-thread level budget costs at most one extra
  thread spawn per top-level call, since every deeper refill uses the much
  larger owned-segment budget.

  Two regression tests, neither of which can abort the test process the way
  the bug did: `stack_segments_are_bounded_by_level_count_alone` drives 2 000
  levels of a deliberately thin recursion (frames far too small for the byte
  budget ever to fire) through `with_stack_segment` and asserts no thread ran
  more consecutive levels than its budget, and
  `stack_probe_rebaselines_after_unwinding` now asserts the re-baselining
  decision on synthetic addresses via the extracted `probe_against_base`,
  because real addresses are not a usable oracle under instrumentation that
  relocates locals — the old version of that test asserted the ASan
  under-read was impossible, and would have started failing as soon as the
  overflow stopped masking it.

  Release builds run the same recursion but not the same failure:
  uninstrumented, the byte probe reads honestly, so the governor was already
  firing and deep expressions were already safe. The nightly `tsan` and
  `lsan` shards failed for an unrelated reason — the wall-clock assertion in
  `holonomic::telescoping2d`, replaced with counter-based instrumentation in
  this same release.

- **The PR AddressSanitizer job no longer spends an hour inside one Gaussian
  elimination.** Removing the ASan carve-out from
  `chained_product_at_original_bounds_refuses_fast_via_resource_ceiling` (now
  that it asserts on `SearchStats` counters rather than elapsed seconds) made
  that one test genuinely run under ASan, where it measured 3550 s — ~59 min
  against a 90-minute job ceiling that the rest of the suite already fills to
  ~60 min.

  The cost was never what the test asserts. The ceilings are calibrated in
  *unknowns*, and `rational_nullspace` is `O(rows · cols²)` over
  unbounded-precision rationals, so letting the cumulative budget through to
  the 245-unknown rung of this input's probe ladder buys an hour of
  arithmetic to arrive at counters the skip logic had already decided before
  any polynomial was built. `search::Ceilings` makes the three numbers data
  rather than constants read directly by the search loop; the search always
  runs on `Ceilings::PRODUCTION`, and nothing outside the module's own tests
  can supply anything else. The regression test now runs the identical input,
  identical degree bounds and identical code path with the ceilings scaled by
  ~1/5, so the affordable large probe is the 50-unknown rung instead — same
  four verdicts on the same four rungs, same assertions, **143 s under ASan
  and 20 s uninstrumented** against 3550 s and ≈450 s. (A 25x cut rather than
  to nothing, because every probe assembles a system of the same ~10 000 rows
  whatever its column count; it is the `cols²` in `rows · cols²` that scales
  away.) `production_ceilings_classify_the_chained_product_probe_ladder`
  pins the shipped `(400, 150, 300)` against the real 5/50/245/770 ladder by
  arithmetic, so the calibration claims in `search`'s docs cannot drift
  unnoticed. Both ceilings were verified still to be caught by the test:
  disabling the cumulative one takes `large_attempted` from 1 to 6 (and the
  test from under a second to 131 s) and fails it; disabling the per-probe
  one fails it too.
- **Risch–Norman ("parallel Risch") heuristic integration**, new in
  `alkahest_cas::integrate::norman` and, on the experimental Python surface, as
  `alkahest.experimental.integrate_parallel_risch`. Instead of building a
  differential-field tower and recursing on it, the heuristic posits
  `F = P/Q + Σ dⱼ·log(pⱼ)` over the monomial basis of `ℚ(x, exp …, log …)`,
  differentiates that ansatz, and solves **one linear system over ℚ**. It is
  the mechanism that is the default integrator in Maple, FriCAS and Reduce.

  **This does not change `integrate`'s routing.** It is a separate entry point
  so its coverage can be measured before any dispatch change is proposed.
  Measured on a 103-case suite (the 40-case coverage probe plus 63 textbook
  integrands): 50 solved by both, **8 solved only by Risch–Norman** — the
  `exp(k·x)/(exp(x)+1)` family, `1/(x·log x·log log x)`, `log(x²)`,
  `(exp(x)+x)/(x·exp(x))` — 23 solved only by the existing engine (trigonometric,
  algebraic, `RootSum` and `arctan` answers, none of which this ring can
  express), 22 by neither.

  **A decline is not a verdict.** The result type is
  `Solved { antiderivative, verification }` / `Declined(reason)`; there is no
  variant a caller can read as a proof of non-elementarity, and
  `DeclineReason::into_integration_error` maps *every* decline to
  `NotImplemented`. `exp(x²)` (no elementary antiderivative) and `1/(x²+1)`
  (which is `atan x`, but needs a constant field bigger than `ℚ`) both come back
  declined, and the API deliberately cannot tell them apart.

  Soundness rests on two independent guards. Before the system is built, the
  algebraic-independence precondition of Bronstein's structure theorems
  (*Structure theorems for parallel integration*, JSC 42(7), 2007) is checked —
  exponential arguments are reduced to a `ℤ`-lattice basis, logarithm arguments
  are tested for multiplicative independence modulo constants — and a tower that
  cannot be certified independent is declined rather than attempted. After the
  system is solved, the rebuilt candidate is differentiated with the
  general-purpose `diff` (not the ring's own derivation table), read back into
  `ℚ(x, θ)` and required to equal the integrand exactly; anything that cannot be
  re-read falls through to `verify_antiderivative_status`, and a candidate that
  passes neither gate is discarded.
- **`∫_{-∞}^{∞} P(x)/Q(x) dx` now takes the residue theorem**, in a new
  `integrate::residue_theorem` module reached from `integrate_definite` before
  the fundamental-theorem path. `∫_{-∞}^{∞} dx/(x⁴+1)` is `π/√2`,
  `∫_{-∞}^{∞} dx/(x⁶+1)` is `2π/3` (it used to *hang*), `∫_{-∞}^{∞} dx/(x²+1)²`
  is `π/2`.

  **Convergence is checked, not assumed.** `deg Q ≥ deg P + 2` on the reduced
  fraction, and `Q` with no real root — the latter by complete VAS real-root
  isolation over ℤ. A failure of either is reported as *divergent*, never given
  a value: `∫ x dx/(x²+1)` has Cauchy principal value `0`, and returning `0`
  would be wrong.

  **How the upper half-plane is selected.** Choosing the poles with `Im α > 0`
  is semi-algebraic, so no rational symmetric-function identity gives the sum,
  and this crate has no certified complex root isolation to build one from. The
  half-plane split is instead pushed into a *spectral (Hurwitz) factorisation*,
  where it becomes an ordinary polynomial factorisation: normalise to an even
  denominator (`D = Q(x)Q(-x)` when `Q` is not already even), rotate
  `Ď(s) = D(s/i)` so "upper half `x`-plane" becomes "left half `s`-plane", and
  split `Ď = A(s)·A(-s)` with `A` Hurwitz. `A` is found by factoring `Ď` over ℚ
  and sorting the irreducible factors with an exact Routh array; a small
  rational linear system then gives the answer as `2π·(leading coeff of G)`.
  For `deg D ≤ 4` — where `A` is typically irrational, as `x⁴+1` needs
  `s²+√2s+1` — a closed form in radicals derived from the substitution `u = x²`
  covers the gap and reaches `ℚ(√·)·π`.

  **Covered:** every denominator whose Hurwitz spectral factor is rational
  (`x²+1`, `x²+4`, `x²+2x+2`, `(x²+1)ⁿ`, `x⁶+1`, …) and, in radicals, every
  denominator of degree ≤ 4 after even-normalisation (`x⁴+1`, `x⁴+x²+1`,
  `x²/(x⁴+1)`). **Declined, explicitly:** anything else — `x⁸+1` is the smallest
  example, being ℚ-irreducible with roots in both half-planes and too large for
  the radical case.

  **Nothing is returned unverified.** Every value is bracketed by a *rigorous
  enclosure of the same integral* before it leaves the module. The whole real
  line is covered with no truncation error: `[-1, 1]` directly, and each tail
  through the exact change of variable `x = ±1/t`, which maps it to `[0, 1]` and
  keeps the integrand rational and regular. All three pieces go through
  `validated::bounds::verified_integral` (adaptive Taylor models in
  outward-rounded ball arithmetic). A candidate outside the enclosure is treated
  as a bug and declined, not returned.

- **`integrate_definite` no longer turns an unevaluated endpoint value into a
  number.** `∫_{-∞}^{∞} dx/(x⁴+1)` returned `0` (true value `π/√2 ≈ 2.2214`),
  and so did `∫_{-∞}^{∞} x²dx/(x⁴+1)`. Both are worse than a decline: a wrong
  number arrived with no warning.

  The mechanism was cancellation of two *unevaluated* expressions. Those
  integrands get a `RootSum` antiderivative from Lazard–Rioboo–Trager, and
  neither of the two ways of evaluating one at a bound actually evaluates it:

  * `calculus::limit` has no `RootSum` rule and, rather than erroring, returns
    the input **unchanged** — so `lim_{x→+∞} F` and `lim_{x→−∞} F` were the same
    expression, still containing `x`, and `F(+∞) − F(−∞)` cancelled to `0`.
  * `kernel::subs` does not descend into a `RootSum` node either, so
    substituting a *finite* bound is a silent no-op and the same cancellation
    happens on a bounded interval: `∫_0^1 dx/(x⁴+1)` "=" `0`, true value
    `≈ 0.8669`.

  `eval_bound` now requires the endpoint value to be free of the integration
  variable, whichever route produced it, and declines (`E-INT-001`) otherwise.
  Both underlying gaps — `limit` returning its input for a shape it has no rule
  for, and `subs` not traversing `RootSum` — are still there; this is the guard
  that stops them reaching an answer.

  Two further wrong answers on the same path are fixed:

  * **`∫_0^∞ dx/(x−3)² = −1/3`** — divergent (double pole at `x = 3`, strictly
    inside). The exact pole check read an infinite bound as "no numeric bound"
    and switched itself off for *every* improper integral; it now compares
    isolating intervals against `±∞` directly.
  * **`∫_0^∞ x^{−2} dx` returned the expression `0^{−1}`** — a successful result
    the evaluator itself rejects. The "must denote a finite real" gate only ran
    when both bounds were finite; a value containing `∞` or an unresolved
    `0^{negative}` is now refused whatever the bounds look like.
- **The last three sites that turned a solver's silence into a theorem now have
  to prove it.** `∫ (√x + 1/√(4x))·eˣ dx` is `√x·eˣ` — elementary, checked by
  differentiation — and was certified `E-INT-004`, "no elementary antiderivative
  exists". The integrand mentions `√x` and `√(4x)`, so the compositum route built
  a degree-4 extension `ℚ(x)[y]/(α⁴ − 10x·α² + 9x²)` for what is really the
  quadratic `ℚ(x)(√x)`. That quartic is *reducible*, so the coefficient ring is a
  product of two fields; the true antiderivative lives in one factor and does not
  lift to the product. The ansatz found nothing and the site published that as a
  proof.

  The three algebraic-RDE call sites — `exp_case.rs`'s nested-radical and
  compositum paths and `exp_algebraic.rs` — now consume a three-valued
  `AlgRdeOutcome` (`Solved` / `NoRationalSolution` / `Declined(reason)`) matching
  the `RdeOutcome` contract the rational solvers already use. Only a **proved**
  non-existence may certify; a decline reports `E-INT-001` naming the premise it
  could not discharge. `alg_rde`'s own module documentation had said the return
  was incomplete ("a denominator/degree bound too small to contain the true
  solution yields `None`") while its callers read it as a proof.

  Proving non-existence for a *coupled* system needed three new complete bounds,
  all of them in `alg_rde.rs`:

  - **The extension must be a field.** A shape-specific irreducibility witness
    for each of the three reachable minimal polynomials (pure radical `yⁿ − p`,
    compositum `√p + √q`, nesting `√(a + √b)`). Without one the solver declines —
    which is what removes the false certificate above.
  - **A complete denominator bound at the finite poles**, the matrix form of the
    scalar resonance already in `rational_rde`: a pole of `b` must be a pole of
    `M` or of `c`, and its order is bounded by the largest positive-integer
    eigenvalue of the residue matrix. The eigenvalue search terminates against a
    spectral-radius bound on the rational matrix representing that residue on
    `ℚ[x]/(Dm)`, so no algebraic pole is ever named.
  - **A complete degree bound at infinity**, by two independent arguments: an
    integer *shearing* (gauge) search that restaggers the power-basis weights
    until the leading matrix is invertible, and — for a radical totally ramified
    at infinity, where no integer shearing can work because `α` has fractional
    degree — a valuation bound on the single place above `∞`.

  Together these keep every verdict that was correct. Across a 42-case sweep of
  the three families, 39 verdicts are unchanged (including `∫ exp(√x)/x dx`,
  `∫ (√x+√(x+1))·eˣ dx`, `∫ √(x+√x)·eˣ dx` and `∫ exp(√(x²+1)) dx`, which the
  ramified-place and shearing bounds respectively rescue), nothing that solved
  stopped solving, and the three that moved `E-INT-004 → E-INT-001` are exactly
  the degenerate-compositum family whose premise was never established. The
  40-case integration probe is unchanged at 27 solved / 7 `E-INT-004` /
  6 `E-INT-001`.

  Still open, and reported rather than fixed here: `detect_two_sqrt_compositum`
  treats `√x` and `√(4x)` as independent radicals. Normalising a radicand by its
  rational square content would let that integral *solve* instead of decline.

- **`integrate` could emit a false `E-INT-004` ("no elementary antiderivative
  exists") for an integrand that has one.** `∫ exp(x + log x) dx` — which is
  just `∫ x·eˣ dx = (x−1)·eˣ` — was certified non-elementary. So were
  `exp(x + 2·log x)`, `exp(x + log(x²+1))`, and `exp(x + log x)/x`, whose
  antiderivative is plain `eˣ`.

  The rational Risch DE solver bounds the denominator of a solution `v` of
  `v' + f·v = c` by `E = gcd(D, D')`, `D` the denominator of `c` (Bronstein
  §6.1). That bound is exact **only when `f` is a polynomial**. When `f` has a
  *simple pole with a positive integer residue* `ρ`, the leading terms of `v'`
  and `f·v` cancel and `v` acquires a pole of order `ρ` at a point `c` is
  regular at — a pole the ansatz `v = N/E` cannot represent. `v' + (2/x+1)·v = 1`
  has `c = 1`, hence `E = 1`, yet its solution is `(x²−2x+2)/x²`. The solver
  returned "no rational solution", and the exp-tower caller turned that into a
  certificate. An exponent with a logarithmic part (`η = x + log x` ⇒
  `f = η' = 1 + 1/x`) is exactly how a user reaches it.

  Two independent fixes:

  1. **The denominator bound now accounts for `f`'s poles.**
     `rational_rde::resonant_denominator` computes `∏_k gcd(d₁, A − k·d₁'·W)^k`
     over the positive integers `k` — Bronstein §6.1's `WeakNormalizer`, with
     the Rothstein–Trager resultant replaced by an eigenvalue bound on the
     residue element of `ℚ[x]/(d₁)` plus a direct GCD test, so resonant poles at
     *irrational* points (e.g. residue 1 at `±i` for `f = 2x/(x²+1)`) are found
     too. The degree bound is likewise now derived from the valuation argument
     at infinity rather than a generous estimate.

  2. **The solvers' return type no longer conflates a decline with a proof.**
     `RdeOutcome` is three-valued — `Solved`, `NoRationalSolution` (*proved*, and
     the only outcome that may license `NonElementary`) and `Declined` (nothing
     may be concluded, mapped to `E-INT-001`). The two-valued
     `solve_rational_rde` / `solve_rational_rde_generalized` remain as shims for
     source compatibility and are documented as unsafe to conclude from; new
     code should use `solve_rational_rde_checked` /
     `solve_rational_rde_generalized_checked`. Every call site in
     `risch::exp_case` and `risch::simple_radical` was rewired so only a
     *proved* non-existence can produce `E-INT-004`.

  A residue past the internal resonance-search cap now **declines** rather than
  certifying: `∫ x⁴⁰⁰⁰·eˣ dx` written as `∫ exp(x + 4000·log x) dx` reports
  `E-INT-001`, which is the honest weaker verdict. No verdict on the 40-case
  integration probe moved between `E-INT-004` and `E-INT-001` (27 solved / 7 /
  6, unchanged).
- **`integrate` no longer certifies elementary nested-exponential integrands as
  non-elementary.** `∫ eˣ·e^(eˣ)/(e^(eˣ)+1) dx` — whose antiderivative is
  `log(e^(eˣ)+1)` — came back as a *certified* `E-INT-004`, "no elementary
  antiderivative exists". The trigger was `simplify`: the raw parse fell through
  to u-substitution and solved, while the simplified spelling entered the exp
  tower and got a false proof, so any pipeline that normalised before
  integrating hit it.

  Three premises in `integrate/risch/` were being used as proofs without being
  established.

  1. **The Laurent decomposition was never validated.** `extract_exp_factor`
     peels integer powers of the generator `t` off a product and puts every
     other factor into the coefficient, without checking that what is left is
     free of `t` — so `t/(t+1)` arrived as the monomial `(t+1)⁻¹·t¹`, a
     "coefficient" that is not in the base field at all. Every certificate in
     the exp case rests on the Laurent theorem ("for `k ≠ 0`, `∫cₖtᵏ` is
     elementary iff the Risch DE `v' + kη'v = cₖ` has a solution in `K`",
     Bronstein §5.3), which does not apply to a genuine rational function of
     `t`: that needs Hermite reduction plus a Rothstein–Trager residue reduction
     over `K[t]` (§5.6), and *those* can contribute **new logarithms** the RDE
     never sees. `integrate_exp_tower` now validates the decomposition and
     declines with `E-INT-001` instead of certifying.

  2. **"the coefficient is rational in the inner generator" was not a proof.**
     The old predicate certified on *any* denominator in the inner generator,
     and tested for one with `contains_subexpr` — which answers `true` for a
     coefficient mentioning the **outer** generator, since `exp(exp(x))`
     syntactically contains `exp(x)`. It is replaced by a test that parses the
     coefficient into `ℚ(x)(θ)` and certifies only when the denominator is
     squarefree, i.e. every pole is simple: there a solution would have to be
     regular at the pole while `c` is not (Bronstein §6.1). Higher-order poles
     are now undecided rather than certified.

  3. **The lower-tower cascade only covered non-negative degrees.** It is now a
     Laurent cascade over `j ∈ ℤ` and therefore a *complete decision* for
     Laurent coefficients: the recursion `vⱼ₋₁ = (cⱼ − vⱼ′ − j·vⱼ)/k` is forced
     from the top degree down to `min(L, 0)`, and the integral is elementary iff
     the last value produced vanishes. Residuals are certified non-zero only by
     an exact test — a rational-function numerator, or a **ball enclosure that
     excludes zero** — never by "the simplifier did not reach `0`".

  A fourth false certificate, found while auditing, is fixed in the same pass:
  `known_nonelementary` returned the logarithmic-integral (`li`) diagnostic for
  *any* product containing a `log(linear)^(-n)` factor, so
  `∫ −dx/(x·log²x) = 1/log x` was certified non-elementary. That family is
  elementary exactly when the polynomial denominator is a constant multiple of
  the log's argument, and the certificate now declines there;
  `try_log_derivative` also accepts a constant multiple of `h'/h` rather than
  demanding equality, so those integrals are answered instead of declined.

  Measured over a 138-verdict corpus: **9 false `E-INT-004`s became answers**
  (the four reported reproducers plus `log(e^(eˣ)+2)`, `1/(e^(eˣ)+1)ⁿ`,
  `atan(e^(eˣ))`, `½log(e^(2eˣ)+1)`), **4 `E-INT-001`s became answers** (the
  `c·h'/h·log(h)^(-n)` family), and **3 integrands moved `E-INT-004` →
  `E-INT-001`** — `∫ e^(eˣ)/(e^(eˣ)+1) dx`, `∫ eˣ/(eˣ+1)²·e^(eˣ) dx` and
  `∫ x·eˣ·e^(eˣ)/(e^(eˣ)+1) dx`. Those three are probably non-elementary, but
  the implementation cannot prove it, and a weaker honest verdict beats a
  stronger false one. The 40-case integration probe is unchanged at 27 solved /
  7 `E-INT-004` / 6 `E-INT-001`.

  **Still not done:** Hermite reduction plus Rothstein–Trager over `K[t]` for a
  rational function of the *outer* generator, and `RdeNormalDenominator` for a
  higher-order pole in the inner generator. Both are now declined rather than
  guessed, and the `risch` module docs say so.
- **`integrate` was emitting false non-elementarity certificates.**
  `∫x dx/√(1−x⁴)` is `½asin(x²)`; Alkahest answered `E-INT-004`, *"no elementary
  antiderivative exists"*. So did a whole family — `∫x√(1−x⁴) dx`,
  `∫x² dx/√(1−x⁶)`, `∫x dx/√(9−x⁴)`, … — 23 of the 30 `E-INT-004` verdicts in an
  algebraic-integrand probe. A certificate is a theorem, so a wrong one is
  strictly worse than declining.

  Both premises of the inference in `algebraic::genus_zero` were unsound.

  1. **"The residue divisor is empty."** It came from `residue_divisor_placed`,
     sanity-checked with `residue_sum_complete != 0`. Neither is a completeness
     argument. On `y² = 1−x⁴` the two places over `x = ∞` carry residues `±i`;
     the places-at-infinity routine reads them off a *rational* Puiseux
     expansion, and since those branches have leading coefficient `±i ∉ ℚ` it
     finds no branches at all and reports nothing. The residue theorem then
     holds vacuously — an empty list sums to zero, and so does an omitted
     conjugate pair — so the check could not catch it, and the code read "found
     nothing" as "there is nothing". `residues::residues_at_infinity_exact` now
     computes those residues in closed form over `ℚ(√lc)` instead of searching
     for them over `ℚ`, and `residues::certified_residue_divisor` supplies the
     predicate that was missing: every place enumerated, every nonzero residue
     representable, or else a refusal. It also closes a second hole in the same
     routine — for odd `deg a` with a non-square leading coefficient the residue
     at `∞` is rational, but the Puiseux search still misses it.

  2. **"There is no algebraic primitive."** This came from
     `solve_rational_rde_generalized` returning `None`, which conflates "no
     rational solution exists" with "my denominator bound was too weak" — it
     returns `None` for solvable equations. A *decline* was being used as a
     proof step. `algebraic::sqrt_rde::decide` now answers the same question
     three-valued (`Solved` / `NoRationalSolution` / `Undecided`), deriving the
     denominator bound from that equation's own pole structure rather than
     generically, so `NoRationalSolution` is a proof. Only that verdict licenses
     a certificate; its `Solved` also recovers antiderivatives the generic
     solver's bound missed.

  Where a premise is unavailable the answer is now `E-INT-001`, an honest
  decline. **Genuine certificates are unaffected** — `∫dx/√(x⁵+1)`,
  `∫x dx/√(1−x⁶)` and the rest keep `E-INT-004`, now with both premises
  established.

- **Two more false-certificate mechanisms in the same area, found while
  sweeping for the one above.**

  * **A pole at `x = 0` was invisible.** `residues::finite_residues_algebraic`
    factors the pole denominator with `poly::puiseux::factor_over_q`, which
    deliberately divides out the largest power of `x` first — for its Puiseux
    callers the root `c = 0` is not a branch and must not appear. Here it is an
    ordinary place, and dropping it meant `∫√(x⁴−1)/x dx` had *no* enumerated
    residues: the pole at `0` sits on an irrational sheet (`a(0) = −1`), so the
    rational-Puiseux routine could not see it either. The integral is
    elementary; it was certified non-elementary. `factor_over_q` is unchanged
    (its other callers want the current behaviour); the residue routines now go
    through a wrapper that keeps the factor.

  * **The simple-radical route certifies without a logarithmic part at all.**
    `risch::simple_radical` solves the component Risch DE `vⱼ′ + (j·p′/(n·p))vⱼ
    = bⱼ` of the Liouville decomposition `∫bⱼyʲ dx = vⱼyʲ + Σcₖlog uₖ`, and
    reports `NonElementary` when the solver returns `None` — ignoring the log
    part entirely, and treating a *decline* as a disproof. But every
    `∫R(x, x^{1/n}) dx` with `R` rational is elementary, since `x = uⁿ` makes
    the integrand rational. `∫∛x/(x²+1) dx` equals
    `−½log(u²+1) + ¼log(u⁴−u²+1) + (√3/2)·atan((2√3u²−√3)/3)` at `u = x^{1/3}`,
    and was certified non-elementary; so were `∫∛x/(x³−1) dx` and
    `∫x^{2/5}/(x−1) dx`. Pending a log-part analysis in that route, the
    algebraic engine's call site downgrades its `NonElementary` to
    `NotImplemented` — containment, documented as such, and reversible in one
    place.

- **New: the power pullback `u = x^k`** (`algebraic::pullback`). An integrand of
  the shape `x^{k−1}·g(x^k)` satisfies `∫f dx = (1/k)·∫g(u) du`, which drops the
  curve's genus by a factor of `k` — `∫x dx/√(1−x⁴)` becomes `½∫du/√(1−u²)` on a
  genus-0 curve and closes as `½asin(x²)`. Recognition is exact structural
  matching, not a numeric fit, and every emission is gated on a numeric
  `d/dx F = f` check against the original integrand. It runs **last** and only
  against an `E-INT-001`, so no integral that already solves changes shape and no
  `E-INT-004` verdict can be talked down by it. Beyond the family above this also
  closes the `asinh` half (`∫x dx/√(1+x⁴)`), previously an honest decline.
- **`Add` and `Mul` are flat at construction, so associativity holds
  structurally.** `ExprPool::mul` / `ExprPool::add` now splice nested
  same-operator children before interning, so `(a·b)·c`, `a·(b·c)` and `a·b·c`
  are one expression rather than three. This is the last of the three
  dimensions of the form-robustness bug fixed in this release (after route
  fall-through and exponent spelling); previously `parse("x*y*z")` produced the
  left-associative `(z * (x * y))` while `pool.mul([x, y, z])` produced a flat
  three-child node, and the two compared unequal.

  ```text
  parse("x*y*z") == pool.mul([x, y, z])        # was False, now True
  pool.mul([pool.mul([x, y]), z]) == pool.mul([x, y, z])   # was False, now True
  ```

  It is **associativity and nothing else** — no reordering beyond the canonical
  sort the pool already applied, no constant folding, no identity elimination —
  so no value changes anywhere. `simplify` already performed exactly this
  transformation (`flatten_mul` / `flatten_add`); doing it at construction just
  means everything that inspects structure *before* `simplify` runs now sees the
  shape the user wrote. Every matcher that scans the top-level arguments of a
  product or a sum used to see two children where the user wrote three.

  The fix is in the two kernel constructors rather than in the parsers, which
  is what makes it reach every construction path at once: both parsers (the
  Rust one *and* the separate pure-Python Pratt parser in `alkahest/_parse.py`,
  which has no binding to the Rust one and would otherwise have needed the same
  fix twice), the builder API, the Python operator overloads, and every internal
  `pool.mul(vec![…])` call site. Splicing preserves argument order, so it is
  sound for the V3-2 non-commutative generators as well. `pool_persist` restores
  nodes through `intern`, not through `mul`/`add`, so a `.pool` file written by
  an older build still round-trips byte-for-byte with its `ExprId`s intact; the
  splice is a fixpoint on an explicit worklist precisely so that such a restored
  nested node still flattens the next time it reaches a constructor.

  **Behaviour changes to plan for.**

  - **Depth.** Flattening *reduces* depth, so a left-associated chain of `+` or
    `*` no longer approaches the `E-DEPTH-001` ceiling: 100 000 terms
    accumulated one at a time are now the same depth-2 node as adding them all
    at once. Expressions previously refused as too deep are now accepted.
    `Pow` towers and `Func` chains are unaffected.
  - **Sharing.** Distinct spellings of one product now intern to one node, which
    *improves* sharing. The reverse case is `e = e + e` in a loop: that used to
    build a depth-*n* DAG of *n* nodes and now builds a flat node of 2ⁿ children
    (measured: 20 iterations → 2 097 152 children, 203 MB). Combining an
    expression with itself under the same operator many times is now
    proportionally expensive.
  - **`subs` with a compound key.** A key is matched as a whole node, never as a
    sub-multiset of a wider sum or product, so `x + y → z` no longer rewrites
    `x + y + 1`. This used to depend on spelling — `x + y + 1` parsed to
    `Add([Add([x, y]), 1])` and *was* rewritten, `1 + x + y` parsed to
    `Add([Add([1, x]), y])` and was *not*. Both are now the same node and
    neither is. AC matching against part of a sum is the pattern API's job.
  - **Certificate ledger.** Four `simplify_trig` shape classes move from
    `certified` to `withheld` (66 → 62 of 156). No mathematics changed: their
    only recorded rewrite step was `flatten_mul`/`flatten_add`, which can no
    longer fire because the expression is never in the unflattened form. The
    simplified *values* are identical.
  - **Provenance.** `alkahest.research`'s dependency inference walks
    subexpressions to decide which earlier claim a value came from; a result
    buried in a wider product is no longer a subexpression of it. The walker now
    also enumerates the proper sub-sums and sub-products of flat `Add`/`Mul`
    nodes (bounded to arity ≤ 6), which restores the edges and finds ones the
    old binary chains missed.

  **Measured.** The 40-case integration probe holds at 27 solved / 7
  `E-INT-004` / 6 `E-INT-001`, with all 40 verdicts identical and no regression;
  five results print with one fewer nesting level (the same terms, re-associated).
  Charlwood's Fifty goes from **12/50 to 14/50** solved-and-verified, with zero
  regressions and zero false certificates. The two new solves — both `E-INT-001
  irreducible product of var-dependent factors` before, both verified by
  differentiating the answer back at every usable sample point with no
  disagreement — are

  ```text
  ∫ x·asin(x)/√(1−x²) dx  =  x − asin(x)·√(1−x²)          (18 points, 0 mismatches)
  ∫ x·atan(x)/√(1+x²) dx                                  (42 points, 0 mismatches)
  ```

  Both are exactly the failure this fixes: the integrand parsed as a two-child
  `Mul` with the inverse-trig factor buried in a nested `Mul`, so the top-level
  factor scan never saw it. Seven further Charlwood answers changed term order
  only, and stayed verified.

  **Cost.** `intern/build_add3` — the microbenchmark that builds a three-argument
  `Add`, i.e. the most directly affected hot path — moves 72.2 ns → 73.6 ns
  (+2.0%), which is the "does any child need splicing?" scan. Every other
  `intern`/`simplify` microbenchmark moves within ±10% with no consistent sign,
  and the 618-test `integrate::` suite runs 12.44 s → 13.02 s. The Rust suite as
  a whole is 91.6 s → 103.7 s, most of which is the new 50 000-node splice
  regression test.

- **The parsers fold `-<numeric literal>`, so `x^(-1)` and `1/x` no longer
  disagree about what a `-1` exponent is.** Prefix `-` built `(-1) · operand`
  unconditionally, which for a literal operand left an unevaluated product in
  the pool: `x^(-1)` interned as `x^(1 · -1)` where `1/x` interned as `x^(-1)`.
  The two are the same function, but every structural detector that reads an
  exponent by matching an integer node saw only the second, which is the root
  cause behind the spelling-sensitivity fixed below. Unary minus applied to an
  `Integer` or `Rational` literal now emits the negated literal directly, in
  both `alkahest-core/src/parse.rs` and its Python mirror
  `alkahest/_parse.py` (`alkahest.parse`). **Nothing else folds**: `-x` is still
  `(-1) · x`, `-(2+3)` keeps its tree, `-2^2` is still `-(2^2) = -4`, and float
  literals are deliberately left alone. Mathematical values are unchanged
  throughout — this is a representation fix.

  The `(-1) · literal` shape stays reachable through the public builder API
  (`Expr.__neg__`, `pool.mul([pool.integer(-1), …])`), so the detectors keep
  their own normalising view of an integer exponent
  (`risch::tower::literal_integer`) as a second layer. Both layers are pinned
  by tests.

  Measured over a 56-integrand probe restricted to inputs that contain a unary
  minus on a literal — the only ones the fold can reach: **5 newly solved, 0
  regressions, 0 changed answers**, and one verdict *upgrade*. Newly solved (all
  verified by differentiating the answer back): `∫ dx/cos²x`, `∫ dx/sin x`,
  `∫ dx/(1+sin x)`, `∫ dx/(2+cos x)` and `∫ dx/(1+e⁻ˣ)`, each written with a
  `^(-n)` exponent rather than `/`. The upgrade is `∫ log(x)^(-1) dx`, which now
  certifies `E-INT-004` (li is non-elementary) instead of the weaker
  `E-INT-001` — the same verdict the identical `1/log(x)` spelling already had.
  The `∫ eˣe^(eˣ)/(e^(eˣ)+1) dx` family, including its `-3·`/`-4·` and `^(-n)`
  variants, is solved identically before and after.

- **`integrate` no longer lets the *spelling* of an integrand decide the
  answer.** Three separate defects combined into one user-visible failure mode:
  the same mathematical object, written differently, got different verdicts.

  ```text
  x^(-1)*log(x)^(-1)      ->  log(log(x))
  1/(x*log(x))            ->  E-INT-001        # the same function
  exp(x)*(1+exp(x))^(-1)  ->  log(exp(x)+1)
  exp(x)/(exp(x)+1)       ->  E-INT-001        # the same function
  x^(-2)                  ->  E-INT-001        # while 1/x^2 integrated fine
  ```

  1. **The router's sub-engine dispatch was unconditional.** A structural
     pre-check (`contains_algebraic_subterm` / `contains_risch_form`) sent the
     integrand to the algebraic or Risch engine and `return`ed whatever came
     back — so an `IntegrationError::NotImplemented`, which is a *decline* and
     not a verdict, was handed to the caller and made everything below the
     dispatch unreachable for any integrand carrying an exp/log/radical
     generator: `try_log_derivative`, the rule engine, Rothstein–Trager and the
     derivative-divides u-substitution. `try_log_derivative`'s own doc comment
     advertised `∫ 1/(x·log x) dx`, a case that by construction always carries a
     log generator and so could never reach it from a top-level call. A
     sub-engine decline now falls through, carrying its diagnostic with it (so a
     specific Risch message is never degraded into a generic one if nothing
     downstream succeeds). **A budget trip and a `NonElementary` verdict still
     short-circuit** — the first because it travels *as* a `NotImplemented` and
     reading it as a decline would turn "stop spending" into "keep spending",
     the second because it is a proof and no fallback can overturn it. Nothing
     downstream can produce a wrong antiderivative: `try_u_substitution` verifies
     `d/dx F = f`, `try_log_derivative` fires only on an exact `h'/h` match, and
     Rothstein–Trager is exact.

  2. **The detectors read tree shape, and `/` and `^(-1)` did not give the same
     tree.** Prefix negation is `(-1) · operand`, so `x^(-1)` produced the
     unevaluated exponent `1 · -1` while `1/x` gave the literal `-1`, and
     `(a·b)^n` was never read as `a^n·b^n`. Two helpers in `risch::tower` —
     `literal_integer` (folds a var-free integer exponent without invoking the
     simplifier) and
     `distribute_integer_pow_over_mul` (`(a·b)^n = a^n·b^n`, an identity for
     *integer* `n`, which is why `(a·b)^(1/2)` stays out) — now give every
     detector and matcher a spelling-independent reading: `needs_log_risch`,
     `needs_exp_risch`, `is_var_dependent_denominator`, `extract_exp_factor`,
     `expr_to_qpoly`, `expr_to_qrational`, `extract_log_power`, and the
     algebraic engine's own `poly_utils::as_integer`. The user's expression
     itself is untouched, so error messages and derivation logs still echo what
     was written.

  3. **`exp(η)` was never offered to the u-substitution search.**
     `collect_usub_candidates` offers `Func` *arguments* and `Pow` *bases*, so
     `∫ exp(x)/(exp(x)+1) dx` (where `exp(x)` happens to be a top-level factor,
     hence offered) and the equal `∫ dx/(1+e⁻ˣ)` (where it is not) behaved
     differently. Each hyperexponential generator is now a candidate, appended
     after the structural ones so the existing search order — and every answer it
     already found — is unchanged. Substituting `t = exp(η)` is the change of
     variable of Bronstein §5.2; this closes the sub-case where the reduced
     integrand `R(x,t)/(η'·t)` is free of `x`.

  Measured over a 164-integrand textbook probe: **16 newly solved, 0
  regressions, 0 changed answers**, plus one verdict *upgrade* (`exp(x)*x^(-1)`
  now certifies `E-INT-004`/Ei like the identical `exp(x)/x`, instead of the
  weaker `E-INT-001`). New: `∫ dx/(x·log x)` and `∫ dx/(x·log(x)^n)` in every
  spelling, `∫ eˣ/(eˣ±1) dx`, `∫ eˣ/(eˣ+1)² dx`, `∫ dx/(1+eˣ)`, `∫ dx/(1+e⁻ˣ)`,
  `∫ dx/(eˣ−1)`, `∫ 2x/((x²+1)·log(x²+1)) dx`, and — embarrassingly —
  `∫ x^(-2) dx` and `∫ (x²+1)^(-1) dx`, which had failed outright while `1/x^2`
  and `1/(x^2+1)` worked.

  **Not done, and now written up in the `risch::exp_case` module docs:** Hermite
  reduction in `K[t]` (Bronstein §5.2–5.3) for genuinely rational functions of a
  hyperexponential generator. `decompose_wrt_exp` requires a Laurent polynomial
  `Σ cₖ(x)·tᵏ` with `cₖ ∈ ℚ(x)`, so a denominator in `t` has no representation
  at all. The `K(t)` arithmetic and tower derivation it would build on already
  exist (`tower_field::TExpr`); the squarefree factorisation and gcd chain over
  `K[t]`, and the resultant/factorisation over `K[t][z]` that Rothstein–Trager
  needs there, do not. `∫ e²ˣ/(eˣ+1) dx` and `∫ dx/(eˣ+e⁻ˣ)` additionally need
  the tower normalised onto a single generator (`exp(2x) = t²`, `exp(-x) = t⁻¹`)
  and are still declined.
- **RL integration environment: tiers 3 and 4 implemented via the LIOUVILLE
  generator, and every generated pair is now verified before it is emitted.**
  `alkahest.rl.envs.integration.grammar.random_elementary` raised
  `NotImplementedError` for tiers 3 and 4. Both now use the LIOUVILLE
  construction of Barket, England & Gerhard ([arXiv:2406.11631]), which samples
  `F = v0 + sum ci*log(vi)` — the shape Liouville's theorem guarantees — over a
  deliberately non-square-free denominator and differentiates it, so the
  integrand is elementary-integrable by construction. Tier 3 is `Q(x)(theta)`
  rational in one exp/log monomial (Hermite reduction + Rothstein-Trager
  residue/log part); tier 4 is a two-generator tower `Q(x)(theta1)(theta2)`
  covering nested exp/log, an algebraically independent second monomial, a
  `sqrt` layer over a transcendental, and `Q(sqrt d)` coefficients. A new
  `alkahest.rl.envs.integration.corpus` module gates every pair — symbolically
  (`simplify(diff(F) - f) == 0`) *and*, independently of `diff`, by a
  Richardson-extrapolated finite difference at points where every `log`
  argument is positive, every `sqrt` radicand is positive and every denominator
  is bounded away from zero — and exposes a CLI
  (`python -m alkahest.rl.envs.integration.corpus`) that writes a verified
  `(integrand, integral)` corpus to JSON with length-balance, uniqueness and
  `integrate`-solvability statistics. `env._make_row` routes tiers 3-4 through
  that gate and resamples on failure, so no unverified pair can reach a training
  set; elementary rows also gained an `F_str` field carrying the reference
  antiderivative. Measured over 5,000 draws: the BWD integrand/integral length
  bias is gone (BWD's ratio *grows* with size, 1.67 -> 2.37 -> 3.13 by
  integral-size bin; LIOUVILLE's is flat or falling, tier 3 ending at 1.16),
  uniqueness after replacing integer coefficients with `CONST` is 97.9% (tier 4)
  / 90.2% (tier 3) against 74.9% for a BWD baseline, and 3.6-4.3% of draws are
  discarded — every one of them domain-restricted rather than wrong, with zero
  numeric mismatches. **Fixes** a pre-existing bug in tier 2, which built
  `sqrt(d)` as `d ** (1/2)` — a `pow` with a non-integer exponent, which
  `diff` rejects with `E-DIFF-002`, so every tier-2 row raised.

  [arXiv:2406.11631]: https://arxiv.org/abs/2406.11631
- **Continuous creative telescoping — Almkvist–Zeilberger, the differential
  twin of Zeilberger's algorithm** (`alkahest_cas::experimental::almkvist_zeilberger`,
  `dgosper`, `integral_boundary_status`; Rust-only, no PyO3 binding yet). The
  holonomic subsystem had a complete discrete stack — proper hypergeometric
  recognition, Zeilberger, the q-analogue, multi-sum Apagodu–Zeilberger, a
  three-valued boundary verdict — and nothing on the `∫ F(n,x) dx` side.
  `holonomic::azeil` is that side: given `F(n,x)` hyperexponential in `x` and
  hypergeometric in `n`, it finds an order `J`, polynomial coefficients
  `a_0(n) … a_J(n)` and an exact rational certificate `R ∈ Q(n)(x)` with

  ```
  Σ_i a_i(n)·F(n+i,x) = D_x( R(n,x)·F(n,x) ),
  ```

  which is a recurrence for `f(n) = ∫_a^b F(n,x) dx` **once the boundary term
  `[R·F]_a^b` is discharged** — a separate question the certificate says
  nothing about, decided three-valued by `integral_boundary_status`. This is
  the principled replacement for a Meijer-G engine on definite integrals of
  special functions: more general, and it produces a certificate.

  Worked end to end, with the certificate checked against the classical
  recurrence rather than against the machinery that produced it:
  `∫_0^∞ xⁿe^{−x} dx = n!` (order 1, `R = x`), `∫_0^1 xⁿ(1−x)^{1/2} dx` and
  `∫_0^1 xⁿ(1−x)ⁿ dx` (order 1), `∫_{−∞}^{∞} x^{2n}e^{−x²} dx` (order 1,
  `R = x`), `∫_0^∞ xⁿe^{−x²} dx` (order **2**, middle coefficient zero),
  Bessel `J_n(2)` via the Schläfli contour (order 2, `R = 1`, recovering
  `J_n + J_{n+2} = (n+1)J_{n+1}`) and Legendre `P_n(3)` likewise (order 2,
  recovering `(n+1)P_n − 3(2n+3)P_{n+1} + (n+2)P_{n+2} = 0`).

  **One solver, not two.** Dividing the identity through by `F` turns it into
  the parametric Risch differential equation `R′ + θ·R = Σ_i a_i·r_i` with
  `θ = ∂_xF/F`; indefinite integration (`dgosper`, the differential mirror of
  Gosper's algorithm) is the same equation at `J = 0`, `a_0 = 1`.
  `integrate::risch::rational_rde::solve_rational_rde_generalized` is
  deliberately **not** reused for that, and the reason is correctness rather
  than style: its denominator bound `E = gcd(D, D′)` is taken from the
  right-hand side alone, which is Bronstein-exact only when `f` is a
  polynomial. Probed directly, it returns `None` for `R′ + (1/x+1)R = 1` and
  `R′ + (2/x+1)R = 1` — the equations behind `∫x·eˣ dx` and `∫x²·eˣ dx`, both
  of which do have rational solutions. Both are regression tests here. (That
  gap is in the generalized entry point, not in the exponential tower that is
  its main caller, where `f = k·η′` is a polynomial; it is worth its own
  investigation for the callers in `simple_radical`, `genus_zero` and the
  fractional-exponent `exp_case` paths, which do pass rational `f`.) The `Q(n)`
  Gaussian elimination *is* shared with `zeilberger`.

  **Honest limitations.** The certificate ansatz is `R = P(x)/(D(x)^κ·B(x))`
  with `D` the denominator of `θ` and `B` a common denominator of the shift
  ratios. That denominator's *support* is derived, not guessed — a pole of `R`
  must lie over a pole of `θ` or of the right-hand side — but its
  *multiplicity* `κ` is a bounded search, because at a simple pole of `θ` with
  residue `ρ` the certificate may have a pole of order `ρ` whenever `ρ` is a
  positive integer, and with `ρ ∈ Q(n)` symbolic (the usual case, `θ = n/x + …`)
  that is not decidable at all. The search is order-major ascending, so a
  returned order is the least one reachable within the degree bounds — the
  claim `zeilberger`'s cost-ordered plan trades away. `B(x)^β` for non-integer
  `β` is treated formally, so reading a certificate as an identity of
  *functions* needs a consistent branch on the interval. Only `n`-independent
  integration limits are analysed; an `n`-dependent one is not representable
  rather than mishandled. Out-of-class inputs get typed refusals under a new
  `E-HOLO-06x` sub-block — with `NotHypergeometricInN` (`E-HOLO-061`)
  deliberately distinct from a shape failure, because `exp(n·x)` closes the
  branch for every algorithm in this family and raising search bounds cannot
  help.

- **`telescope2d` generalizes from two bound indices to an arbitrary `m ≥ 1`:
  `experimental.telescope_md`** (M4 extension). `telescope2d(term, n, j, k)`
  only ever reached exactly two bound indices; the underlying ansatz search
  and boundary/face analysis are now implemented for general `m`, with
  `telescope2d` itself unchanged in behavior (it is now a thin `m = 2`
  wrapper over the general engine, not a separate implementation) and a new
  `telescope_md(term, n, [x_1, …, x_m])` for `m ≠ 2` — including `m = 1`,
  which degenerates cleanly to a single-sum-shaped search, and `m ≥ 3`,
  genuinely new. Same scope as before, generalized: proper hypergeometric
  summands only (no broader rational-summand class, no sum of several
  hypergeometric terms), a fixed (non-minimal) certificate denominator built
  from `F`'s own shift-ratio denominators rather than a minimal multivariate
  Gosper normal form, and constant-box-only boundary analysis. **Not
  attempted**: a genuine minimal Gosper-style certificate denominator (the
  roadmap's stated remaining-gap item 3) — real algorithm-design work, not an
  engineering extension of what already exists here.

  **The boundary is `2m` face sums, not `2^m` corner evaluations** — the
  `m = 2` module's "four strip sums, not four corners" result, generalized:
  summing the telescoping identity over an `m`-dimensional box gives `2m`
  sums, each over an `(m − 1)`-dimensional face where one bound index is
  fixed to a boundary value, not `2^m` point evaluations at the box's
  corners. The same sufficient (not necessary) "face vanishes pointwise"
  criterion the `m = 2` module used — a dominant `1/Γ` zero among `F`'s own
  gamma factors, or the certificate's own numerator vanishing there — carries
  over unchanged in kind: fix one axis to a constant and check that a gamma
  factor's argument no longer depends on `n` or any *other* bound index.

  **Resource ceilings, added after this generalization surfaced a real
  scaling cliff.** The linear system a probe builds has one equation per
  distinct monomial and one unknown per certificate-numerator box
  coefficient; both dimensions grow with `m` and the certificate degree
  bound far faster than the degree numbers suggest, and `rational_nullspace`
  is a plain dense `O(rows · cols²)` exact-rational Gaussian elimination.
  Profiling a genuinely coupled `m = 3` example
  (`C(n,x)·C(x,y)·C(y,z)`) at degree bounds that work fine at `m = 2` found a
  ≈10 000-row, 245-unknown system whose elimination step alone took ≈47
  seconds *per probe*, and the next certificate-degree step up (770 unknowns)
  was still running after several minutes — genuine `O(rows·cols²)`
  arithmetic cost on a correctly-posed system, not a bug or an infinite loop,
  but exactly the kind of resource cliff a caller needs protecting from.
  `holonomic::telescoping2d::search::MAX_ANSATZ_UNKNOWNS` now refuses any
  single probe above 400 unknowns outright, and
  `MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS` caps the total work spent on probes
  at or above 150 unknowns to 300 across one whole search call — so a search
  with no certificate in reach cannot be made to pay the same expensive
  elimination over and over across every `(order, a_degree)` combination
  tried. Neither ceiling touches the `m = 2` search, whose default probes
  never exceed ≈140 unknowns. A search that hits a ceiling reports
  `SearchExhausted` naming the ceiling explicitly, never a false certificate
  and never a silent hang.

  Verified on the 4-category multinomial coefficient
  `F(n,x,y,z) = n!/(x!y!z!(n−x−y−z)!)` (via `factorial`, not a
  product-of-binomials encoding, to avoid the redundant cancelling-gamma-
  factor pairs a naive `C(n,x)·C(n−x,y)·C(n−x−y,z)` encoding would carry) — a
  genuinely non-separable `m = 3` sum (all three bound indices interact
  through the shared `n − x − y − z` term) whose closed form,
  `Σ_{x,y,z} F = 4ⁿ` (the multinomial theorem), is checked by direct exact
  summation (`rug::Rational`, never floats) against the returned recurrence,
  plus the `4ⁿ`-decoupled fixed-support variant where the boundary genuinely
  is `n`-independent and `Vanishes` is provided rather than refused. A
  regression test pins the original scaling-cliff example
  (`C(n,x)·C(x,y)·C(y,z)` at `order ≤ 2`, `cert_degree ≤ 3`) to now return a
  bounded, ceiling-cited refusal rather than run unboundedly.

  New: `alkahest.experimental.telescope_md`,
  `alkahest.experimental.TelescopingMdCertificate`. Experimental, same
  refusal codes as `telescope2d` (`E-HOLO-040`/`041`/`042`).

- **The elliptic route's soundness gate becomes a reusable, graded facility:
  `integrate::gate`.** The *propose an ansatz → fit the coefficients
  numerically → snap to rationals → verify symbolically → emit only if the
  gate passes* pattern lived in one module
  (`integrate/algebraic/elliptic_output.rs`) as a private boolean check. It is
  now `alkahest-core/src/integrate/gate.rs`, with a **graded** verdict in
  place of a boolean: `Proven` (the simplified residual `d/dx F − f` is a
  syntactic zero), `EnclosureVerified { boxes, residual_bound }` (a *rigorous*
  bound on `|d/dx F − f|` over stated closed boxes, via Taylor models in
  outward-rounded ball arithmetic — a statement about a whole interval, not
  about finitely many points), `SampledOnly { points, tolerance }` (the
  historical `f64` screen, now named rather than implied), `Failed` (refuted
  at a specific point) and `Declined` (the gate could not run, and says
  nothing). `GateOptions::min_strength` is the caller's floor, so a route can
  demand a rigorous enclosure and decline anything weaker.

  The module documents what each verdict does *and does not* prove. No verdict
  is a proof of the integral; the rigorous tier can never cover branch points,
  poles or the unbounded tails, and it refuses anything it cannot
  Taylor-model rather than passing it silently. Domain-awareness is
  caller-supplied (samples + predicate + boxes) instead of hardcoded to
  "radicand positive", and the private numeric evaluator is replaced by one
  that dispatches through the shared `PrimitiveRegistry`.

  `elliptic_output.rs` is refactored onto it with **no behavioural change** —
  same acceptance rule, same tolerance, same sample grid, all pre-existing
  tests unchanged. Cost, measured on a release build: the default gate is
  ~0.38 ms per candidate, while the rigorous tier costs 1.3 s – 9.9 s, so it
  is tiered (cheap symbolic check → `f64` screen → enclosure only on
  survivors) and off by default on that hot path; `try_elliptic_output_with`
  takes an explicit `GateOptions` for callers who want it. A test certifies
  all four first-kind reductions to a residual bound of 3e-9 – 8e-9 over boxes
  strictly inside the radicand-positive region.

- **`∫√(tan x) dx` and the rationalizing-substitution family.** Radicals with
  a *non-polynomial* radicand previously declined with "radicand P is not a
  polynomial in the variable". For radicands whose derivative is a rational
  function of themselves — `tan`, `cot`, `tanh`, `exp` of a linear argument —
  `uⁿ = g(x)` now rationalizes the integrand, and the `u`-integral is closed
  either by the existing engine or, where Rothstein–Trager would return an
  unevaluable `RootSum`, by a fitted **real partial-fraction ansatz**
  (`log(u − r)`, `log(u² − 2αu + α²+β²)`, `atan((u − α)/β)`, polynomial
  ladder). The back-substituted result is gate-verified against the original
  integrand in `x`; `∫√(tan x) dx` comes out in the classical closed form and
  reaches `EnclosureVerified` with a residual bound of 9.8e-9. A bare `x`
  outside the radical, a polynomial radicand, and `√(sin x)` all decline.
- **`gamma`, `digamma` and `EllipticPi` can be differentiated — and
  `trigamma` is new.** All three parsed and evaluated but could not be
  differentiated in every argument, and an antiderivative carrying one of them
  is *unverifiable*: the integrator's gate checks `d/dx F = f` and cannot
  check what it cannot differentiate. Now `d/dx Γ(x) = Γ(x)·ψ(x)` (DLMF
  5.2.2), `d/dx ψ(x) = ψ₁(x)`, and `Π(n; φ | m)` differentiates in **all
  three** of its arguments. Previously only `∂Π/∂φ` existed, and even that
  rule bailed out entirely as soon as `n` or `m` depended on the
  differentiation variable — so `diff(Π(n(x), φ, m), x)` failed with
  `E-DIFF-001`. The `∂/∂n` and `∂/∂m` reductions (DLMF 19.4.7 rewritten for
  the parameter convention `m = k²`; Byrd & Friedman 710 for `∂/∂n`) were
  checked against central differences of the quadrature evaluator before being
  written down.

  `trigamma(x)` = `ψ₁ = ψ′` is a new primitive with the full bundle
  (`numeric_f64`, `numeric_ball`, a rigorous Taylor-model rule so
  `bound_on_box` reaches it, unicode `ψ₁` + LaTeX `\psi_1`, PyO3 binding).
  It is the one deliberate exception to "every primitive differentiates":
  `ψ₁′ = ψ₂` and the polygamma ladder has no closed-form terminator short of a
  binary `polygamma(n, x)`, so `diff(trigamma(x), x)` **declines** with
  `E-DIFF-001` rather than returning a placeholder. Moving the boundary from
  `ψ₀` to `ψ₁` is what makes `Γ′ = Γψ` and `Γ″ = Γ(ψ² + ψ₁)` both land on
  functions the gate can evaluate and bound.

  New: `alkahest.trigamma` (additive to `__all__`).

- **Fresnel integrals `fresnels`/`fresnelc`**, in the **normalised (π/2)
  convention** — DLMF §7.2(iii), A&S §7.3, SymPy, SciPy, Mathematica — with
  `S(∞) = C(∞) = 1/2`, `S′ = sin(πx²/2)` and `C′ = cos(πx²/2)`. The
  unnormalised `∫₀ˣ sin(t²)dt` is a *different* function (limit `√(π/8)`);
  mixing the two is a silent `√(π/2)` error, so the convention is pinned by a
  test rather than left implicit. Maclaurin series below `|x| = 6`, summed in
  MPFR at a working precision raised by the series' own `≈ 2.27x²` bits of
  cancellation (in plain `f64` it loses ~10 digits at `x = 4`); DLMF
  7.12.1–7.12.3 asymptotics above it, where DLMF §7.12(ii)'s
  first-neglected-term remainder bound makes the truncation rigorous rather
  than merely plausible. Worst relative error `3.3·10⁻¹⁵` against
  `scipy.special.fresnel` over `[0, 40]` plus spot checks to `10⁸`. Full
  bundle including a Taylor-model rule, so `bound_on_box` reaches them.

  New: `alkahest.fresnels`, `alkahest.fresnelc`.

- **Dilogarithm `dilog`** — `Li₂` on the **principal branch, cut along
  `[1, ∞)`** (DLMF §25.12(i), Lewin §1.1, Mathematica `PolyLog[2, z]`): real
  on `(−∞, 1]` with `Li₂(1) = π²/6`, and declining for `x > 1` where the
  principal value is complex, rather than silently returning its real part.
  `Li₂′ = −log(1−x)/x`. Bernoulli series on `[−1, ½]`, reached by the
  inversion (`x < −1`) and reflection (`x > ½`) identities; worst relative
  error `5.0·10⁻¹⁶` over a 34 000-point sweep of `[−10⁶, 1]` against MPFR's
  correctly-rounded `mpfr_li2`, an independent implementation. Full bundle
  including a Taylor-model rule whose coefficient recurrence runs forwards
  above `m₀ = 0.4` and backwards (Miller) below it, because each direction is
  stable exactly where the other is not.

  Shipped as `dilog` rather than a general `polylog(s, x)`: `∂Li_s/∂s` has no
  closed form, so a binary `polylog` would ship with a *permanently* declined
  partial, and every `Func` rule in the validated Taylor tier is unary, so it
  would also be invisible to `bound_on_box`. `Li₁` needs no primitive — it is
  `-log(1 - x)`.

  New: `alkahest.dilog`. None of the four new names is wired into either
  parser — like `digamma` and `bessel_j0`, they are constructor-only for now.
  Nothing in `integrate/` was touched: emitting Fresnel or `Li₂` forms from
  `∫sin(x²)dx`, `∫log(x)/(1+x)dx` and friends is a separate change.

## 3.9.0 — 2026-08-14

Everything in this section landed **after `v3.8.0` was tagged and published**,
so none of it is in the released 3.8.0 wheel. It is the result of acting on a
trial autoresearch run: fifteen logged issues, of which this release fixes
fourteen (issue #6 was expectation-setting, not a defect, and issue #10 is
reduced but not closed — see below).

**Upgrading from 3.8.0 is not transparent.** Two changes need a read before you
upgrade: `relation_confidence` now returns a *tri-state* verdict, and seventeen
zero-argument accessors became properties, which is a hard break with no alias.
Both are detailed under *Behaviour changes to plan for*.

### Behaviour changes to plan for

- **`relation_confidence` returns `credible: None` — *unknown* — for inputs
  whose precision it cannot establish, where it used to return `True`.** It
  judged only `float` inputs, on the premise that "decimal strings and ints are
  exactly the values they spell, so a relation among them holds exactly". That
  premise is false for the one way PSLQ is actually driven: a high-precision
  decimal string standing for a **truncated** numerical value. So on the input
  an experimental-mathematics loop produces, the gate passed everything —
  including three relations found during the 2026-08-13 autoresearch run that
  re-evaluation at 60 digits refutes. A gate that cannot fail is worse than no
  gate, because loop authors wire it into promotion logic. Now: `float` is 53
  bits as before, `mpmath.mpf` reports its working precision, `int` and
  `Fraction` are exact, and everything else — decimal strings included — is
  `None` until the caller declares `digits=` or `precision_bits=` (a cap, not an
  override: declaring 200 digits for a `float` still yields ~16). A relation must
  also clear the available precision by `margin_digits` (default 10) rather than
  merely fitting inside it, because PSLQ returns the *smallest* relation the
  precision can buy, so a purchased one lands just under the old bound rather
  than over it; all three of the run's spurious relations were 5–8 digits under
  it. The verdict dict gains `margin_digits` and `precision_source`, and
  `available_digits` / `spare_digits` are `None` when the verdict is. **If you
  branch on `credible`, treat `None` as "not checked" — `if verdict["credible"]`
  is the correct polarity, `is not False` is not.** `guess_relation` is
  unchanged for decimal strings: an unjudgeable input is returned unjudged, never
  refused, and `E-PSLQ-004` still fires on float inputs. Unknown precision does
  not disable the gate, either: available precision is a `min` over the inputs,
  so one input whose precision *is* known bounds the whole set, and a relation
  already too expensive for that bound comes back `False` even while
  `available_digits` stays `None` — a single `float` among decimal strings caps
  the search at ~16 digits however many digits the strings carry.
- **`pool.symbol("x")` now takes its domain from the ambient
  `alkahest.context(domain=...)`**, falling back to `Domain.Real` only outside a
  context block; an explicit `domain=` argument still wins. It previously
  ignored the context entirely, while the documented `alkahest.symbol("x")` —
  which is a thin wrapper that resolves the context and forwards it — did not.
  The two constructors sit side by side and only one consulted the domain, so
  building an integer problem through the pool silently emitted
  `(set-logic QF_NRA)` and `Real` sorts, and `solve()` answered the *real
  relaxation*: an Erdős–Straus instance came back `sat` with `z = 252/13` while
  `status`, `verification.status` (`exactly_verified`) and `supported().reason`
  (`ok`) all stayed green, because the model does satisfy the formula as
  emitted. Nothing was unsound; the question being answered had changed. Code
  that relied on pool symbols being `Real` inside an integer context should pass
  `domain="real"` explicitly.
- **Seventeen zero-argument scalar accessors became properties — drop the
  `()`.** There was no rule a caller could predict: `Enclosure.width` and
  `RegularChain.n_vars` were properties while `DAE.n_equations()` and
  `MultiPoly.total_degree()` were methods, so the same shape of question was
  asked two different ways depending on the class. The convention now is: **a
  zero-argument, O(1), non-allocating accessor returning a scalar or a flag is a
  property; anything that returns a collection, allocates, or does real work is
  a method.** No compatibility alias is provided, deliberately — an accessor
  that answers to both forms leaves the ambiguity in place. Migration is
  mechanical:

  | Before | After |
  |---|---|
  | `UniPoly.degree()` | `UniPoly.degree` |
  | `UniPoly.is_zero()` | `UniPoly.is_zero` |
  | `MultiPoly.is_zero()` | `MultiPoly.is_zero` |
  | `MultiPoly.total_degree()` | `MultiPoly.total_degree` |
  | `MultiPolyFp.is_zero()` | `MultiPolyFp.is_zero` |
  | `MultiPolyFp.total_degree()` | `MultiPolyFp.total_degree` |
  | `RationalFunction.is_zero()` | `RationalFunction.is_zero` |
  | `GbPoly.is_zero()` | `GbPoly.is_zero` |
  | `GbPoly.n_vars()` | `GbPoly.n_vars` |
  | `ODE.order()` | `ODE.order` |
  | `DAE.n_equations()` | `DAE.n_equations` |
  | `DAE.n_variables()` | `DAE.n_variables` |
  | `HybridODE.n_events()` | `HybridODE.n_events` |
  | `Component.n_equations()` | `Component.n_equations` |
  | `Component.n_ports()` | `Component.n_ports` |
  | `OdeTrajectory.t_final()` | `OdeTrajectory.t_final` |
  | `ArbBall.is_exact()` | `ArbBall.is_exact` |

  **Calling the old form now raises `TypeError: 'int' object is not callable`
  (or `'bool'`, `'float'`), which is loud. The reverse mistake is not.**
  Writing `if dae.n_equations:` against a *pre*-3.8.0 build silently reads a
  bound method, which is always truthy, and `f"{dae.n_equations}"` formats as
  `<built-in method ...>`; so grep for these names rather than waiting for a
  traceback. Accessors that were *already* properties (`Enclosure.lower`,
  `.upper`, `.width`, `.subdivisions`, `Matrix.rows`, `.cols`,
  `RegularChain.n_vars`, `RosenfeldGroebnerResult.consistent`, `.truncated`,
  `ArbBall.mid`, `.rad`, `.lo`, `.hi`, …) are unchanged; they were already
  correct under the rule, as were the collection-returning methods that sit
  beside them (`RegularChain.polys()`, `RosenfeldGroebnerResult.final_basis()`).
  Three zero-argument scalar calls stay methods because they do real work rather
  than read a field: `Matrix.rank()` (Gaussian elimination), `ODE.is_autonomous()`
  (walks every RHS expression) and `PositivityCertificate.verify()` (re-runs the
  exact SOS identity check). `tests/test_accessor_convention.py` pins the
  converted set and scans `alkahest-py/src/lib.rs` for new offenders, so the
  inconsistency cannot creep back.

### Fixed

- **A verified Zeilberger certificate could imply a *false* recurrence for the
  sum, and nothing in the result distinguished that case from a correct one.**
  This is the only defect logged in the 2026-08-13 autoresearch run that could
  make a loop report a theorem that is not one; everything else was a refusal, a
  slowdown or a discoverability gap. Zeilberger's algorithm proves
  `Σ_i a_i(n)·F(n+i,k) = ΔG`, an identity about the **summand**; a recurrence
  for `S(n) = Σ_k F(n,k)` needs a boundary term to vanish, which is a separate
  statement. On OEIS A279013, `a(n) = Σ_{k=0}^{n} C(2k,k)/(k+1)·C(2n−1,n−k)`,
  `zeilberger` returned a verified order-2 certificate in 0.1 s whose
  homogeneous recurrence fails against the real sequence (2, 8, 35, 161, 768,
  3773) at the very first term. It was caught only by a harness that carried an
  independent numeric check against OEIS data — every signal reachable through
  the API said "proved".

  The information was not missing: `side_conditions` already named the exact
  condition and even gave a counterexample. It was **invariant** — the same
  string for a correct and an incorrect case — so nothing a loop reads could
  tell them apart. `zeilberger` now computes a verdict.
  `ZeilbergerCertificate.boundary` is `"vanishes"` (proved; the homogeneous
  recurrence holds for the sum), `"nonzero"` (proved; the **inhomogeneous**
  `Σ_i a_i(n)·S(n+i) = b(n)` holds, with `b(n)` in `boundary_rhs`) or
  `"unknown"` (neither; nothing may be claimed about the sum).
  `boundary_reason`, `implies_sum_recurrence`, `limits` and `boundary_at(k_lo,
  k_hi)` come with it, and `side_conditions` now varies with the verdict.
  A279013 comes back `"nonzero"` with a `b(n)` that reproduces the sequence
  exactly; Franel, Dixon, Apéry and the binomial row sum come back `"vanishes"`.

  **`"vanishes"` is a proof, not a numeric check.** Each endpoint of `G` is
  evaluated by exact order counting in `Q(n)` — the multiplicity of the endpoint
  as a root of the certificate's numerator and denominator, deflated exactly,
  plus `−e` for every `Γ(a·n+b·k+c)^e` factor whose argument lands on a
  non-positive integer there, with the `(−1)^m/(m!·b)` residue carried exactly so
  that a pole cancelling a zero still gives the right finite value. A strictly
  positive total order *is* an exact zero; a negative one means `G` is unbounded
  at the endpoint and is reported `"unknown"`. Terms are then collected into
  hypergeometric similarity classes (`Γ(x+1) = x·Γ(x)` worked off until every
  argument is `a·n + c` with `c ∈ [0,1)`) and `"vanishes"` requires every
  collected coefficient to be the zero element of `Q(n)`. No floating point,
  interval or sampled value can produce it. Symmetrically `"nonzero"` requires a
  *witness* — an integer `n₀` with `b(n₀) ≠ 0` in exact rational arithmetic —
  so sampling that finds only zeros yields `"unknown"`, never `"vanishes"`.

  **The summation range is now part of the call**, because getting it wrong
  silently is the whole bug. `zeilberger(..., limits=(k_lo, k_hi))` takes it,
  each endpoint an `Expr` or an `int`; it defaults to `(0, n)` — the `Σ_{k=0}^n`
  convention the classical identities and the OEIS formula field use — and the
  range actually used is echoed back on `cert.limits`, so the assumption is on
  the record rather than inferred from the summand. The verdict is about that
  range and changes with it: A361712's certificate is `"vanishes"` over
  `k = 0..n` and `"nonzero"` over the `k = 0..n-1` that OEIS truncates to.
  A range the analysis cannot place — endpoints that are not integer-affine in
  `n` — is `"unknown"`, never `"vanishes"`. `boundary_at(k_lo, k_hi)` re-decides
  for another range without re-running the search.

  Getting this right needed one piece the issue report did not mention: when the
  limits move with `n`, `Σ_{k=0}^{n} F(n+i,k)` is **not** `S(n+i)`. For
  `Σ_{k=0}^{n} C(n,k) = 2ⁿ` the telescoped difference alone is `−1`, and it is
  the missing term `C(n+1,n+1) = 1` that cancels it — a verdict computed from the
  endpoints alone reports `"nonzero"` on a textbook identity. The full
  `b(n) = G(n,k_hi+1) − G(n,k_lo) + Σ_i a_i(n)·D_i(n)`, with `D_i` the finitely
  many values of `F` between the range at `n` and the range at `n+i`, is what is
  decided. New core module `holonomic::boundary` (`BoundaryStatus`,
  `boundary_status`, `natural_limits`); `zeilberger()`'s own signature is
  unchanged and the core function returns `Unknown` rather than guessing when no
  limits are supplied.

- **Interval evaluation refused `bessel_j0` / `bessel_j1`, and its Bessel ball
  kernel was not an enclosure.** `evaluate(bessel_j0(x), {x: ArbBall(...)},
  mode="interval")` came back `status="unsupported"` with `E-EVAL-010` even
  though both functions have rigorous ball kernels and `capabilities()` reported
  `numeric_ball: True` for both. The evaluator carried its own hand-written
  match over function names — the third such list in this area — and these two
  had simply never been added to it. It no longer has one: every `Func` node is
  now dispatched through the primitive registry, so the set of functions
  interval evaluation accepts **is** the set the registry advertises a
  `numeric_ball` kernel for, by construction rather than by agreement. Auditing
  the two sets against each other also turned up the gap in the other direction:
  `atanh` had a ball kernel and did *not* advertise it, because the capability
  probe tested ball kernels at `1.0` only and `atanh`'s domain is the open
  interval `(-1, 1)` — it declined the sole probe point and lost a bit it had
  earned, while keeping `numeric_f64` because that probe already tried `0.5`.
  Ball kernels are now probed at the same points as `f64` ones. Wrong-arity
  calls are declined rather than silently bounding the first argument
  (`sin(x, y)` was `sin(x)`), which the generic dispatch made reachable.

  **`ArbBall::bessel_jn` was separately unsound and is rewritten.** It hulled
  `Jₙ(lo)` and `Jₙ(hi)`, which is only an enclosure for a *monotone* function —
  `J₀` on `[-1, 1]` has equal endpoints (`≈ 0.7652`), so the hull collapsed to a
  point that excluded `J₀(0) = 1`, the function's own maximum. It is now a
  midpoint evaluation plus a mean-value bound with `L = 1`, which is rigorous at
  every order: `|Jₙ| ≤ 1` for all real `x`, and `J₀′ = −J₁`,
  `Jₙ′ = (Jₙ₋₁ − Jₙ₊₁)/2`, so `|Jₙ′| ≤ 1`. A randomised sweep samples the true
  function inside 200 random intervals and checks nothing escapes the enclosure.
- **`verified_no_roots` could not prove a root *exists* past an even root
  count.** The `"false"` direction fired only when a sign change was visible at
  the box's own endpoints, so any even number of roots defeated it however
  obvious they were: `x²−2` on `[-2, 0]` was `"false"`, but the same expression
  on `[-2, 2]` — which contains *both* roots — was `"undecided"`, as were
  `[-10, 10]` and `(x²−2)(x²+1)` on `[-2, 2]`. The machinery to settle it was
  already running; one bisection of `[-2, 2]` makes each half answer
  immediately. The `"false"` direction now searches for its two witness points
  by subdividing the box, and the intermediate-value argument is stated over the
  box rather than over an interval: a box is convex, so two points at which the
  expression is *rigorously proven* to have opposite signs certify a root on the
  segment joining them, which stays inside the box. That also lifts the test to
  several variables — `x − y` on `[-1, 1]²` is now `"false"` where it used to be
  `"undecided"`. Continuity, which the argument needs, is exactly what the
  full-box enclosure succeeding already certifies. **Nothing was weakened to buy
  this**: a root that never produces a sign change and never lands on a point the
  search can *prove* is a root — a double root like `(x−1/3)²` on `[0, 2]`, or
  `(x²−1)²` on `[-2, 2]` — still answers `"undecided"`, because no witness pair
  exists and none is invented.
- **`verified_no_roots` could not see a root sitting exactly on a box
  endpoint.** `x` on `[0, 1]` and `x−1` on `[0, 1]` both came back
  `"undecided"`, as did `sin(x)` on `[0, 1]` and `log(x)` on `[1, 2]`. The
  sign-change search above provably cannot settle them — `x` is non-negative
  everywhere on `[0, 1]`, so the negative witness it looks for does not exist to
  be found, however long it runs. But no search is needed: the box is **closed**,
  so a point of it at which the expression is *proven* to be zero is a root in
  the box, full stop. `"false"` is now returned when either of two independent
  proofs is in hand — the existing sign change, or a point whose value is pinned
  to zero. That also settles a root of even multiplicity that lands on a point
  the search visits (`(x−1)²` on `[0, 1]` and on `[0, 2]`, and the multivariate
  `(x−½)² + (y−½)²` on `[0, 1]²`), which a sign change can never reach.
  **The certificate is not weakened**: a value is pinned to zero only by a
  degenerate `[0, 0]` enclosure — an enclosure is a superset of the value, so
  `[0, 0]` forces it — or by substituting the point's exact rational coordinates
  and simplifying to the literal `0`, cross-checked against the enclosure exactly
  as the removable-singularity path already is. An enclosure that merely
  *contains* zero proves nothing and is not used: `exp(x) − 1 + 10⁻⁴⁰` on
  `[0, 1]` has no root at all, but its value at `x = 0` is far below the width of
  any enclosure computable there, and it stays `"undecided"` rather than being
  claimed either way.
- **`verified_sign` could hang on a rational constant with enough digits.**
  `sin(x)·D − N·x ≥ 0` on `[0, 3/2]` took 0.05 s at `N/D = 636/1000`, 0.08 s at
  nine digits, and **over 300 s at twelve** — three extra digits turned
  milliseconds into a hang. The cause was neither exact rational arithmetic nor
  repeated conversion but a **non-terminating loop** in the branch-and-bound:
  a sub-box bisected down to the width floor was pushed back onto the active
  list, immediately re-selected as the smallest key with nothing changed, and
  the loop spun *without ever consuming its subdivision budget* — which is why
  capping `max_subdivisions` at 64 did not help. It only triggered once `tol`
  became unreachable, and `tol` is an **absolute** width: at `D = 10¹²` the
  function is of order `10¹¹`, so the default `1e-9` asks for twenty digits and
  the floor arrives first. Nine digits happened to converge just above the
  floor, twelve just below. Boxes that reach the floor are now retired out of
  the active list, their keys still folded into the final bound, so the search
  always makes progress. Cost is now flat in the size of the constant: 0.33 s at
  three digits through 0.47 s at sixteen.
- **Inequalities that are tight at an endpoint no longer stay `"undecided"`.**
  Cusa–Huygens, Mitrinović–Adamović, Wilker and Huygens were `"true"` on
  `[0.1, 1.5]` but `"undecided"` on `[0.01, 1.5]` and at `x = 0` — precisely the
  point that makes them worth stating. Two independent things were in the way.
  First, `tol` was also the wrong *stopping rule* for a sign question: on
  `[0.01, 1.5]` the true minimum of the Cusa–Huygens form is `1.7·10⁻¹³`, so the
  search met the `1e-9` tolerance and stopped with an enclosure that still
  straddled zero. `verified_sign` now re-runs the search with the sign itself as
  the goal, refining while the running bound straddles zero instead of to an
  absolute width. Second, where the margin genuinely *vanishes* no subdivision
  can ever help, so the box is split: a collar `[a, a+δ]` at the endpoint is
  handled by a truncated Taylor expansion there, and the rest by the usual
  branch-and-bound. The two pieces are closed and share the join point `δ`, so
  their union is the original box with no gap. All four are now `"true"` on
  `[0, 1.5]`, as is Jordan's inequality stated exactly as
  `10¹²·sin x − 636619772368·x ≥ 0`.
  **The remainder is proven, not assumed.** Coefficients `c_k = g⁽ᵏ⁾(a)/k!` are
  accepted as *zero* only when substitution followed by `simplify` lands on the
  literal integer `0` — no numeric enclosure can prove a value is zero, and none
  is asked to — cross-checked against ball arithmetic, and the tail is a
  Lagrange remainder `|R(t)| ≤ t^m·sup|g⁽ᵐ⁾|/m!` whose sup is a rigorous
  enclosure over the whole collar. Analyticity, which Taylor's theorem needs, is
  certified by requiring every derivative `g … g⁽ᵐ⁾` to enclose successfully
  there. With `c_0 … c_{j−1}` proven zero, `g(a+t) ≥ t^j·[c_j − T(δ)]` and
  `t^j ≥ 0` finishes it. **Nothing was traded for the extra reach**: a margin
  that vanishes in the *interior* — `(x − 7/10)²(x + 1)` on `[0, 3/2]` — is
  still `"undecided"`, because the expansion does not apply there. A leading
  coefficient proven *negative* now returns `"false"` rather than `"undecided"`,
  which settles cases no sampling could see: `x³ − x²/1000` is negative only on
  `(0, 1/1000)`, and each of the four inequalities reversed is refuted at the
  same endpoint where the original is certified. A strict `"positive"` query is
  `"false"` where the expression is proven to vanish exactly, so `x² > 0` on
  `[0, 1]` is `"false"` while `x² ≥ 0` is `"true"`.
- **`verified_integral` refused removable singularities.** Taylor-model
  quadrature raised `E-VALIDATED-003` on any sub-interval where the reciprocal's
  enclosure contained zero, which put `∫₀¹ ln(1+x)/x dx = π²/12` out of reach
  even though nothing about that integral is singular — only the expression as
  written is, and the integrand extends continuously to 1 at `x = 0`. An
  integrand that splits as `N(x)/D(x)` with `N(p) = D(p) = 0` at a point `p` of
  the offending sub-interval is now enclosed through **Cauchy's mean value
  theorem** instead: `N(x)/D(x) = N′(ξ)/D′(ξ)` for some `ξ` in the sub-interval,
  so the piece is bounded by an enclosure of `N′/D′`, which is regular. The
  value returned is the integral of the continuous extension.
  `∫₀¹ ln(1+x)/x`, `∫_{-1}^{1} sin(x)/x`, `∫₀¹ (eˣ−1)/x` and
  `∫₀¹ (1−cos x)/x` now come back as enclosures that bracket `π²/12`,
  `2·Si(1)`, `Σ 1/(n·n!)` and `Cin(1)` to better than `10⁻⁹` wide.
  Three guards keep a genuine pole out: the two zeros are established
  **symbolically** (substitute the exact rational `p`, simplify, require a
  literal zero — no numeric enclosure can prove a value is exactly zero, so none
  is asked to), `D′` must be certified non-vanishing on the sub-interval, and
  `N` and `D` must each enclose successfully over it, which is what certifies
  they are analytic and hence that the symbolic derivatives are the real ones.
  `1/x`, `sin(x)/x²` and `(x−p)²/(x−p)³` on boxes containing the pole are all
  still refused.
- **A `verified_integral` refusal now says whether the *integral* fails to exist
  or only the enclosure of the *integrand* does.** An integrable singularity
  that is not removable — `∫₀¹ −ln x dx = 1`, `∫₀¹ (ln x)² dx = 2`,
  `∫₀¹ dx/√(1−x²) = π/2`, `∫₀¹ xˣ dx`, `∫₀¹ ln(x)·ln(1−x) dx` — is still
  refused, because no rigorous bound on the singular tail can be derived from
  the expression alone today. But the `E-VALIDATED-003` message now names the
  location (left endpoint, right endpoint or interior, with the approximate
  coordinate), reports the underlying cause, and states explicitly that an
  integrable singularity still has a finite integral that this routine cannot
  certify. Widening an enclosure to make those cases "pass" would have broken
  the contract that makes the module worth using, so they refuse honestly
  instead.
- **The ODE/DAE guide documented an API that does not exist.**
  [`docs/mdbook/src/ode-dae.md`](docs/mdbook/src/ode-dae.md) showed keyword
  constructors — `ODE(state=…, derivatives=…, independent=…)`,
  `DAE(equations=…, variables=…, independent=…)` — and a one-argument
  `lower_to_first_order(higher_order_ode)`, none of which were ever real: the
  actual calls are `DAE.new(equations, variables, derivatives, time_var)` with
  derivatives as *separate symbols* like `pool.symbol("dx/dt")`, and
  `lower_to_first_order(var, rhs, order, time_var)`. It also printed
  `reduced.differentiated` on the `pantelides` result, which the returned `DAE`
  does not have. Following the page failed on its first line, and it was the
  only documentation there was, because `DAE.new`, `ODE.new`,
  `rosenfeld_groebner` and the `RosenfeldGroebnerResult` accessors had no
  docstrings at all. The page is rewritten around calls that run — every block
  is executed by `tests/test_docs_ode_dae.py` — those docstrings now exist,
  `reduced.index` works as documented (`DAE.index` is exposed), and
  `.differentiated` is gone rather than faked.
- **`zeilberger` no longer refuses a constant base just because it is not
  already a literal.** `(-one)**(n+k)`, with `one = pool.integer(1)`, builds the
  node `Mul(1, -1)` — the pool does no arithmetic at construction — and the
  proper-hypergeometric parser demanded a *literal* rational base, so it raised
  `E-HOLO-001` *"not a proper hypergeometric term: power with symbolic exponent
  needs a rational base, got (1 * -1)"*. `1 * -1` **is** the rational −1, and
  the same summand written `pool.integer(-1)**(n+k)` was decided in 0.4 s. That
  made a spelling look like a capability limit: an autoresearch run recorded the
  OEIS targets A357558 and A357559 (`Σ (−1)^(n+k)·k·C(n,k)·C(n+k,k)²` and its
  `k³` sibling) as outside Alkahest's reach when both in fact yield an order-2
  recurrence. The base is now constant-folded first — `Mul`/`Add`/integer-`Pow`
  towers over integer and rational literals, e.g. `1 * -1`, `2/4`, `(-2)^3`,
  `3 - 4` — under the parser's existing depth bound and a new bit-width budget
  on folded values, so a nest like `((2^32)^32)^32` is refused rather than
  evaluated. What counts as a proper hypergeometric term is otherwise unchanged:
  a genuinely symbolic base is still refused, and a base that folds to `0` still
  refuses as `0` raised to a symbolic power.
- **`SmtResult.verification` now carries the emitted sorts alongside the
  status.** The logic and the sorts decide *which question was solved* — `Int`
  versus `Real` is `QF_NIA` versus `QF_NRA` and hence an integer problem versus
  its real relaxation — but they were reachable only via `SmtResult.logic` and a
  separately-called `supported()`, neither of which a loop recording `status`
  will look at. `verification` gains `sorts` (`{"x": "Int", ...}`) next to
  `logic` and `status` on all three statuses, with a new `SmtResult.sorts`
  property to read it back, so the sorts survive being recorded into a claim
  graph rather than having to be re-derived.
- **`E-SMT-002` now says how to fix a quantified formula.** `solve()` correctly
  refuses explicitly quantified input, but "does there exist x, y, z such that…"
  is the natural way to write a satisfiability question and `Exists` is exported
  at top level, so the refusal was landing on the obvious spelling without
  saying that free variables in a sat query are already implicitly existential.
  When the formula is a prefix of `Exists` over a quantifier-free body, the
  message now leads with dropping the quantifiers and passing the body, and
  names the bound variables; under a `Forall`, where that rewrite is invalid, it
  says so instead. `supported()` gives the same guidance. The quantifiers are
  **not** stripped automatically: `solve()` back-substitutes its model against
  the formula it was given, and rewriting it silently would answer about a
  different expression.
- **A source build without FLINT now fails immediately, with an install hint,
  instead of at link time.** `alkahest-core/build.rs` probes for a linkable
  FLINT (library file in any search directory, `cc -print-file-name`, headers /
  pkg-config, `ldconfig`) and, finding none, stops with the package name for
  every common platform rather than letting the build run to completion and die
  in `rust-lld: error: unable to find library -lflint`. **FLINT remains a hard
  requirement and `cargo:rustc-link-lib=flint` remains unconditional** — that is
  now documented at the emit site, in `flint/mod.rs` and in the `flint3` feature
  comment, because it looks gateable and is not: `src/flint/` is compiled
  unconditionally, `UniPoly` *is* a `FlintPoly`, and factorization, resultants,
  normal forms and `number_theory` call FLINT with no pure-Rust fallback.
  Gating the link behind `cfg(flint3)` was measured: it produces a `cdylib` with
  68 undefined FLINT symbols that links cleanly and then fails at
  `import alkahest` with `undefined symbol: nmod_poly_init` — a worse failure,
  later.
- **New `FLINT_LIB_DIR` / `FLINT_INCLUDE_DIR` build-time overrides.** They add a
  link search path and feed FLINT version detection, so a FLINT built into a
  user-local prefix — no root, no system package — is both linkable and
  correctly identified as FLINT 3. Verified end to end on a host with no system
  FLINT: `FLINT_LIB_DIR=$PREFIX/lib FLINT_INCLUDE_DIR=$PREFIX/include cargo
  build -p alkahest-py --release --features "parallel egraph cranelift groebner"`
  succeeds against a locally built FLINT 3.2.2. `ALKAHEST_SKIP_FLINT_CHECK=1`
  bypasses the probe; `DOCS_RS` skips it automatically, since rustdoc never
  links.
- **`fmpz_mat_struct` layout detection read the wrong header.** The probe looked
  for a `stride` field in `flint/fmpz_mat.h`, but FLINT 3 declares the struct in
  `flint/fmpz_types.h` — so *every* FLINT 3 was reported as the older
  `rows: **fmpz` layout, whatever it actually used. Both fields are
  pointer-sized, so this is not a size mismatch: on a FLINT that uses `stride` it
  would have made `FlintMat` dereference an integer as a pointer. The probe now
  extracts the `fmpz_mat_struct` typedef body from either header and skips a
  header that does not contain the declaration, rather than reading its absence
  as "no stride field". The version fallback (used only when no header is found)
  moved from "3.1+ is stride" to "3.5+ is stride"; FLINT 3.2.2 is confirmed to
  still use `rows`.
- **`E-SOS-002` now says "undecided, not a refutation" in the message itself.**
  The text was accurate and specific but could be read as "not SOS", which for a
  search loop is a wrongly closed branch — and a wrongly closed branch is
  invisible, since nothing downstream ever contradicts it. The message and the
  registered remediation now say to record it as `unknown`, and state that the
  diagonally dominant cone is a *strict subset* of the SOS cone, so refusal is
  compatible with `p` being SOS. All the original specifics (the basis degree,
  `raise basis_degree`, the Motzkin caveat, `decide` as the fallback) are kept.
  The corresponding sections of `positivity.md`, `errors.md` and the agent skill
  spell out the three worlds that produce `E-SOS-002` and note that `E-SOS-003`,
  which carries a witness point, is the only SOS *refutation*.
- **`parallel` now ships in the default PyPI wheel, on every platform.** It was
  the one feature whose absence was *silent*: `numpy_eval_par` and
  `simplify_par` exist in every build and quietly fall back to their sequential
  counterparts, so benchmarking `numpy_eval_par` against `numpy_eval` on a PyPI
  wheel timed one code path twice — and the only build that had threads was a
  Linux-only `+full` wheel from GitHub Releases, so no macOS or Windows user
  could obtain them from any wheel at all. `release-build.yml` now builds the
  default wheel with `egraph groebner cranelift parallel` on Linux
  (manylinux_2_28), macOS arm64 and Windows (MinGW); `+jit` gains `parallel`
  too, so no opt-in wheel is a silent downgrade from the default one. rayon and
  dashmap are pure Rust with no system dependency, and `ci-cross.yml` was
  already building `parallel` on both macos-14 and windows-2022, so this adds
  no toolchain requirement. `ci.yml`'s "PyPI-parity" `maturin develop` and the
  `wheel-smoke` job build it too, so the binary PR CI tests is once again the
  binary users install.

  `capabilities()["features"]["parallel"]` remains the way to check, because
  `parallel` is still not a Cargo *default*: a source build that does not ask
  for it gets the silent single-threaded aliases. `README.md`,
  `getting-started.md`, `codegen.md`, `interop.md`, `features.md` and the agent
  skill say so at each `*_par` entry point.

  **`+full` gains `cranelift`, making it the only wheel with both JIT
  backends.** Moving `parallel` into the default wheel briefly left `+full` with
  a feature set identical to `+jit`'s — `parallel` had been the entire
  distinction, and `groebner`/`egraph` are Cargo defaults, so naming them added
  nothing — which would have shipped two byte-identical wheels under different
  names. `+full` is now a strict superset of the default wheel, which is what
  its name promises, and its smoke test asserts both backends so the variants
  cannot silently converge again.

- **ThreadSanitizer was never given the parallel code to sanitize.** The nightly
  `tsan` shard ran `cargo +nightly test --workspace` with *default* features, so
  rayon and dashmap were not compiled in at all: `ExprPool`'s index was a plain
  `Mutex<HashMap>`, `simplify_par` / `simplify_redex` / `simplify_auto` and
  `CompiledFn::call_batch_par` did not exist, and F4 reduced sequentially. The
  shard was reporting a clean bill for code it had never seen — and per
  `CONTRIBUTING.md` no sanitizer runs `pytest`, so nothing covered the PyO3
  boundary either. It now builds with `--features parallel`.

  **The nightly `asan` shard had the same blind spot and is fixed the same
  way.** AddressSanitizer was checking memory safety on a build with no
  concurrent code compiled into it — precisely the code whose memory safety is
  hardest to get right. It now runs `--features parallel` too. Only the nightly
  shard: Tier 1a's ASan job stays as it is, so the PR critical path does not
  grow.

  New `alkahest-core/tests/parallel_stress.rs` gives that shard something to
  sanitize, from real OS threads rather than Rayon's own pool: concurrent
  interning checked against a single-threaded node-count baseline (a lost
  `DashMap::entry` race would show up as duplicate nodes, i.e. structural
  equality quietly ceasing to imply id equality), lock-free `ExprPool::with`
  reads against a growing `boxcar::Vec`, concurrent `simplify_par` /
  `simplify_redex` on one shared pool, interning *while* a GIL-free
  `simplify_par` walks the same pool, and nested `call_batch_par`. New
  `tests/test_parallel_threadsafety.py` covers the same shapes above the FFI
  boundary, where `ExprPool` is a plain sendable `#[pyclass]` and
  `py_simplify_par` holds a `PyRef` borrow across `Python::allow_threads`.

  Both suites are clean, under TSan and without. Two TSan findings, neither a
  defect in the parallel paths: a reported data race whose two stacks are
  entirely inside `crossbeam_epoch`/`crossbeam_deque` (epoch-based reclamation
  uses an asymmetric `membarrier` barrier that TSan cannot model) — suppressed
  by name in a new `tsan.supp`, deliberately narrow so a genuine race in one of
  our Rayon closures still fails the shard; and a SIGSEGV that is a stack
  overflow, not memory corruption, because `with_stack_segment`'s governor
  refills at 512 KiB against Rayon's default 2 MiB worker stack and TSan's
  instrumented frames do not fit that margin. The shard sets
  `RUST_MIN_STACK=33554432`; it does not reproduce in an uninstrumented build.

- **Documented that a `CompiledFn` is pinned to the thread that compiled it.**
  Surfaced while writing the tests above: `PyCompiledFn` is
  `#[pyclass(unsendable)]`, so using a `compile_expr` result from another
  `threading.Thread` raises
  `pyo3_runtime.PanicException: alkahest::PyCompiledFn is unsendable, but sent
  to another thread`. Behaviour is unchanged and the check is a safety net — it
  fires before anything unsound happens — but it becomes much easier to hit now
  that `parallel` ships by default and the obvious misuse is to compile once and
  fan the handle out over a thread pool. Two details make it sharper than an
  ordinary error, and both are now in `codegen.md` and the agent skill: it has
  nothing to do with `parallel` (plain `numpy_eval` is refused identically), and
  `PanicException` derives from `BaseException`, not `Exception`, so a worker
  wrapped in a bare `except Exception:` will not catch it. Compile per thread;
  `ExprPool` and `Expr` are shareable.

- **`unsafe impl Send for ExprPool` / `unsafe impl Sync for ExprPool` removed.**
  They were unnecessary in both builds — every field already derives the traits
  (`boxcar::Vec<Node>`, `DashMap` under `parallel`, `Mutex<PoolIndex>` without
  it) — and an unconditional `unsafe impl` on a type that qualifies anyway is
  worse than nothing, because it also silences the check *for the future*: add
  an `Rc`, a `Cell` or a raw pointer to `ExprPool`, `Node` or `ExprData` and the
  compiler would have gone on certifying the pool as shareable across Rayon
  workers and across `Python::allow_threads`. Replaced with a `const _` static
  assertion that the three types are `Send + Sync`, which costs nothing at run
  time and now fails the build instead.
- **`numpy_eval` now explains its calling convention instead of describing the
  symptom.** Passing the `Expr` rather than the `CompiledFn` raised
  `AttributeError: 'builtins.Expr' object has no attribute 'n_inputs'`, which
  names an implementation detail; it is now a `TypeError` saying to compile the
  expression first. Passing the arrays packed in one list — `numpy_eval(f, [a,
  b])` — raised `ValueError: expected 2 input array(s), got 1`, true but not
  actionable; the `ValueError` now says that arrays are separate positional
  arguments and to unpack with `numpy_eval(f, *arrays)`, and recognises a 2-D
  array whose first axis matches the arity as the same mistake. `numpy_eval_par`
  validates identically, and against its own name rather than the name of the
  function it falls back to. `CompiledFn.__call__` — which goes the *other* way,
  taking one point as a single sequence — answers `f(1.0, 2.0)` and `f(1.0)`
  with that convention and a pointer to `numpy_eval` for batches. Exception
  types are unchanged.
- **The Cranelift backend was never linted, and had stopped being clean.**
  `ci.yml` ran `cargo clippy -- -D warnings` for default / `egraph` /
  `parallel` / `jit` / `groebner-cuda` but not for `cranelift`, so three
  warnings accumulated in code that ships in the **default PyPI wheel** — and
  `CONTRIBUTING.md` tells contributors to run `--all-features`, which therefore
  failed on a clean checkout. A `cargo clippy (cranelift feature)` step now runs
  alongside the others; it needs no system LLVM, because the backend is pure
  Rust. The three lints are fixed rather than suppressed: `emit_eval_body` took
  nine positional arguments and now takes an `EvalTarget` (root node, inputs,
  pool) and an `InputLayout` enum (`Scalar { ptr }` / `Batch { ptr, point_idx,
  n_points }`) — which also makes the half-specified batch layout a point index
  with no point count, previously two independent `Option`s — unrepresentable;
  and the two `return`s in `compile_jit_only`'s cranelift arm collapse to the
  one-line `return` the other backends' arms already use.
- **`docs/mdbook/src/representations.md` documented a method that does not
  exist, and five outputs that are not what the code prints.** The page showed
  `p.leading_coeff()` on `UniPoly`, which had no leading-coefficient accessor at
  all (it now has one — see *Added*); `sparse_interp_univariate(..., T=3)` and
  `sparse_interp(..., T=2, D=5)`, whose parameters are `term_bound` and
  `degree_bound`; and `p.to_symbolic(pool)`, under the claim that "all
  specialized types can be converted back to a generic `Expr`" — no polynomial
  type exposes a symbolic conversion to Python, so that whole section was
  fiction. Every code block on the page is now executed against the built
  extension, and the `# output` comments match what `print` actually emits
  (`MultiPoly` prints in ascending exponent-vector order, an `ArbBall` prints as
  `midpoint ± radius` rather than as an interval, and
  `sparse_interp_univariate` returns a list of `(coefficient, exponent)` pairs
  rather than a `MultiPolyFp`). The round-trip section now shows the conversion
  that does exist, `GbPoly.to_expr()`, and says plainly that `UniPoly`,
  `MultiPoly` and `RationalFunction` have none.
- **Four `examples/` scripts had rotted against the 3.9.0 API**, two of them
  *silently* — they exited 0 while printing the wrong thing.
  `phase3_polynomials.py` died at line 30 on `ExprPool.pow`, which no longer
  exists on the Python side (use `a ** b`), and then on the trailing `pool`
  argument that `UniPoly` / `MultiPoly` / `RationalFunction.from_symbolic` no
  longer take, and on `UniPoly.pow(3)`. `agent_workflow.py` called `round()` on
  the exact symbolic solutions `solve` now returns, and passes `numeric=True`.
  The two silent ones: `risch_integration.py` claimed a non-elementary refusal
  for `∫√(x³+1)dx`, which is genus 1 and now returns `EllipticF`, so the guard
  moved to a genus-2 integrand that still raises `E-INT-004`; and
  `lean_certificates.py` printed an empty Lean export, because `to_lean` withholds
  a vacuous `e = e := rfl` when the simplifier found no rewrite. All 17 example
  scripts now run end to end. `examples/` is not covered by CI's ruff or pytest,
  which is why this went unnoticed.

### Added

- **Modular / `p`-adic evaluation of holonomic sequences: `ModularRecurrence`,
  `binomial_mod`, `supercongruence_sweep`** (M6). A supercongruence claim such
  as `A(p−1) ≡ 1 (mod p³)` is checked by evaluating a P-recursive sequence at
  one index per prime, over a range of primes. Alkahest had the *proving* half
  of that workload (`zeilberger`, `guess_holonomic`, the boundary verdict) and
  none of the evidence half: there was no way to evaluate a holonomic sequence
  modulo `p^k`, so every sweep ran outside the library in Python big-integer
  arithmetic, where `A(p−1)` is an integer with `Θ(p)` digits that gets touched
  `Θ(p)` times. Now:

  - `ModularRecurrence(coeffs, initial, *, rhs=None, start=0)` runs
    `Σ_i a_i(n)·S(n+i) = b(n)` forward in `Z/p^K` — `O(N)` machine-word
    multiplications and `O(1)` memory, whatever the size of `S(N)` over `Z`.
    `coeffs[i]` is lowest-degree-first, the same convention
    `GuessedRecurrence.coeffs` already uses, so a fitted recurrence is handed
    straight over. `value_mod(n, p, k)`, `values_mod(indices, p, k)` (one
    forward pass for a scattered index set) and `evaluate(...)`, which returns
    a `ModularEvaluation` carrying the precision accounting. Coefficients and
    initial values are arbitrary-precision, and initial values may be
    `fractions.Fraction`, so harmonic-number-style sequences work.
  - `binomial_mod(a, b, p, k)` — the Andrew Granville / Davis–Webb
    factorisation of `n!` into its `p`-free part, which at `k = 1` *is* Lucas'
    theorem. `O(p·k³ + log_p(a)·p·k)`, so `a` far larger than `p` is the
    ordinary case rather than the hard one; the `p`-free factorial is taken by
    a product tree over blocks of `p` consecutive integers rather than term by
    term, which is what keeps `p^k` out of the cost.
  - `supercongruence_sweep(recurrence, primes, k, *, index=…, expect=…)` →
    `CongruenceSweep`, with `holds`, `counterexamples()`, the histogram of
    `v_p(LHS − RHS)` in `valuations()`, and `sharp` — `True` when some prime
    achieves exactly `v_p = k`, i.e. when the modulus in the conjecture is best
    possible and `p^(k+1)` is false.

  **Singular indices are the part that had to be got right.** Stepping forward
  means dividing by the leading coefficient `a_J(n)`, which need not be a unit
  mod `p`: for the Apéry recurrence `a_2(n) = (n+2)³`, the index `n = p−2` is
  exactly the one a sweep crosses when it asks for `A(p)` instead of `A(p−1)`.
  A first pass computes `v_p(a_J(n))` at every step — falling back to exact
  integer arithmetic at the rare index a residue cannot decide — so the total
  precision loss `L` is known *before* the first sequence value exists, and the
  forward pass runs at working precision `k + L`. The division by `p^v` is
  checked, not assumed. Three new codes, and no path that returns a residue it
  cannot justify: `E-HOLO-006` (the modulus is not a prime power the
  machine-word backend supports), `E-HOLO-007` (a step does not determine its
  next term as a `p`-adic integer — the leading coefficient vanishes
  identically there, or the sequence leaves `Z_p`, as `H_p = H_{p−1} + 1/p`
  does), `E-HOLO-008` (`k + L` needs a modulus past `2^62`). The last is a
  real limit rather than a formality: reaching `A(199)` at `p = 5` crosses 40
  singular steps costing three digits each, and 120 digits of `5` is past any
  64-bit modulus, so that call refuses instead of answering.

  Measured on Apéry `A(p−1) mod p⁴` for the 237 primes below 1500: 95 ms,
  against 632 ms for iterating the recurrence exactly and reducing, and 3.47 s
  for the incremental-binomial sum the research harness uses today. The gap
  widens with the range, both exact routes being quadratic in `p` — at the 428
  primes below 3000 it is 270 ms against 4.7 s and 50 s.

- **`q`-analogue creative telescoping: `experimental.q_zeilberger`** (M4b).
  `q`-hypergeometric sums — Gaussian binomials `[n;k]_q`, `q`-Pochhammer
  symbols — are not proper hypergeometric terms in `(n,k)`, so `zeilberger`
  refused every one of them with `E-HOLO-001` and the whole `q`-literature was
  out of reach. `q_zeilberger` is the same algorithm read in the `q`-shift: with
  `x = qⁿ`, `y = q^k`, `k ↦ k+1` is `y ↦ q·y`, which is an automorphism of
  `Q(q)(x)(y)` exactly as `k ↦ k+1` is one of `Q(n)(k)`, so the Gosper normal
  form, the key equation `A(y)·X(q·y) − B(y/q)·X(y) = C(y)·N(y)` and the
  coefficient-comparison linear system all carry over. The bottom two levels of
  the coefficient tower are the *existing* `qfield` code re-read (`Q(v)` and
  `Q(v)(w)` naming their variables `q` and `x`), so only the third level, `y`,
  is new arithmetic.

  The discipline is unchanged: the pair `(a_i, R)` is substituted back and
  checked as an exact identity in `Q(q)(qⁿ)(q^k)` before it is returned, and a
  candidate that fails is discarded rather than returned with a caveat.
  Verified end to end on `Σ_k [n;k]_q²·q^{k²} = [2n;n]_q` (the `q`-Vandermonde
  convolution at `m = r = n`, i.e. the `q`-analogue of `Σ_k C(n,k)² = C(2n,n)`),
  which comes out as the order-1 relation
  `(1 − q^{n+1})·S(n+1) = (1 + q^{n+1})(1 − q^{2n+1})·S(n)` in ~0.2 s; also on
  the Galois numbers `Σ_k [n;k]_q` (order 2) and the alternating
  `Σ_k (−1)^k q^{k(k−1)/2}[n;k]_q = 0`.

  `QZeilbergerCertificate.sum_term(n0)` returns `S(n0)` as an exact polynomial
  in `q`, computed from the *definition* of the `q`-Pochhammer symbol and never
  through the shift quotients the search used. That is what the tests check the
  returned recurrence against, and it is the check that matters: the A279013
  failure this release's boundary work came from was a certificate that
  re-verified perfectly while implying a false recurrence for the sum, and only
  an independent look at the actual terms catches that.

  **The boundary verdict is two-valued here — `"vanishes"` or `"unknown"`, with
  no `"nonzero"` arm — and the sum it is about is `S(n) = Σ_{k ∈ Z} F(n,k)`.**
  Fixing the range at all of `Z` is what makes the proof short: the range does
  not move with `n`, so the `D_i` correction terms the classical
  `boundary_status` needs do not arise, and `"vanishes"` follows from two
  structural facts about the summand alone — that it vanishes outside an affine
  window in `k`, and that it is finite at every integer `k`. Both are decided by
  reading off when a `q`-Pochhammer factor is `1 − q⁰` (exactly zero) or has one
  inside the reciprocal product a negative length denotes (exactly infinite),
  which is a linear condition on `(n,k)` plus a divisibility, settled by
  Fourier–Motzkin over the rationals; rational-empty implies integer-empty, so a
  region proved empty is a proof and not a sample. What it does *not* do is
  evaluate the certificate at an endpoint, and that is the load-bearing part:
  `R` really does have poles at integer `k` — on the `q`-Vandermonde summand a
  double pole exactly where the summand has a double zero, making `G(n,n+1)` a
  finite *non-zero* limit of `0·∞` — so a proof that multiplied the two values
  there would be wrong. Instead it takes one `k` past both the window and the
  finitely many poles, where `G = R·0 = 0` unambiguously, and inducts downwards
  through `G(n,k) = G(n,k+1) − Σ_i a_i(qⁿ)·F(n+i,k)`, whose right-hand side the
  support analysis has already shown finite everywhere. The window is reported on
  `.support` so the caller knows which finite sum the verdict is about
  (`("0", "n")` for the `q`-Vandermonde summand). An inhomogeneous `b(n)` is
  *not* computed: it needs endpoint values of `G` that are not rational in `qⁿ`,
  so a summand whose support cannot be bounded gets `"unknown"` and no claim
  about its sum at all — `1/(q;q)_{n−k}` telescopes perfectly well and has no
  `Z`-sum, and the verdict says exactly that.

  **`q` is treated as transcendental**, and every verdict says so in
  `side_conditions`. These are identities in `Q(q)`; specialising `q` to a root
  of unity — which is what `q`-supercongruence work does — is a separate step
  with its own hypotheses that this engine does not take.

  Refusals are coded and disjoint from the classical engine's, so a caller can
  tell which one declined: `E-HOLO-020` outside the class (a bare `n` or `k`, a
  `gamma`, a `sin`), `E-HOLO-021` bounds exhausted, `E-HOLO-022` a candidate
  that failed verification, `E-HOLO-023` a malformed call, and `E-HOLO-024` for
  an input in the *shape* of the class whose shift quotient is not rational —
  the canonical case being `(q^k; q²)_n` under `k ↦ k+1`, where the first
  argument moves by `1` and the base `q²` does not divide it, making the
  quotient an infinite product. Like `E-HOLO-020` and unlike `E-HOLO-021`, that
  is a permanent answer about the input.

  New: `alkahest.experimental.q_zeilberger`,
  `alkahest.experimental.QZeilbergerCertificate`, and the two term builders
  `qbinomial(pool, N, K)` and `qpochhammer(pool, u, d, v)`. Experimental, so
  the surface may change; the mathematics it refuses to guess at will not.

- **Root-of-unity specialisation for `q_zeilberger`:
  `QZeilbergerCertificate.specialize_at_root_of_unity`** (M4). `q_zeilberger`
  proves identities in `Q(q)` with `q` transcendental and said, in every
  verdict's `side_conditions`, that specialising `q` to a root of unity — the
  step the `q`-supercongruence literature actually needs — was a separate step
  this engine did not take. It now does, and takes it as a three-valued
  **decision** rather than an assumption, because doing it unconditionally is
  the `q`-analogue of the A279013 failure mode transplanted to a new
  subsystem: a certificate that is perfectly valid in `Q(q)` and would
  silently produce a false statement if evaluated at a point where a
  coefficient or a sum value has a pole.

  The arithmetic underneath (`alkahest_core::holonomic::qzeil::cyclotomic`) is
  exact throughout: `Φ_d(q)`, the `d`-th cyclotomic polynomial, is built from
  `q^d − 1 = ∏_{e∣d} Φ_e(q)` by exact polynomial division over `Q`; `Q(ζ_d) =
  Q[q]/(Φ_d(q))` is the residue field, with inversion by the extended
  Euclidean algorithm (total on non-zero elements because `Φ_d` is irreducible
  over `Q`); and "does `p` vanish at `ζ_d`" is decided as "does `Φ_d` divide
  `p`" — a divisibility question over `Q`, never a floating-point evaluation.
  The same machinery gives the exact `Φ_d`-adic valuation of any element of
  `Q(q)`, which is the `q`-supercongruence statement in its precise form:
  `v ≥ r` is exactly `Φ_d(q)^r ∣ S(n)`.

  `spec = cert.specialize_at_root_of_unity(d, n)` returns a
  `QRootOfUnitySpecialization` whose `status` is `"specializes"` (every
  coefficient and every sum value proved to have non-negative `Φ_d`-adic
  valuation, so the specialisation map is defined on all of them, and the
  specialised identity was re-checked as an exact statement in `Q(ζ_d)` before
  being returned), `"obstructed"` (a pole at `ζ_d` was **exhibited** — no
  specialised value is offered, and this is a proof the route is blocked, not
  that the specialised identity is false), or `"unknown"` (the generic
  boundary verdict was already `"unknown"`). Three further conditions are
  reported on a `"specializes"` verdict rather than folded into it:
  `is_vacuous` (every coefficient died — always true at `d = 1`, the `q → 1`
  limit — so the recurrence is `0 = 0`, still a theorem but an empty one),
  `leading_coefficient_survives` (`False` means the specialised recurrence no
  longer determines the last value from the earlier ones), and
  `support_shrinks` / `effective_support` (the `q`-Lucas phenomenon —
  `[2;1]_q = 1 + q` is non-zero in `Q(q)` and zero at `ζ_2`, so the surviving
  window at a root of unity can be a strict subset of the generic one, though
  it can never grow, since a ring homomorphism sends the generic zero
  summands to zero).

  Verified end to end on `Σ_k [n;k]_q²·q^{k²} = [2n;n]_q` specialised at every
  `ζ_d` for `d = 1..6`: the returned value checked against a Gaussian binomial
  built independently by the Pascal recurrence directly in `Q(ζ_d)`, against
  the closed form `[2n;n]_{ζ_d}` predicted by the `q`-Lucas theorem from the
  base-`d` digits of `2n` and `n`, and — at the Python surface — against the
  same sum recomputed in floating-point `complex` arithmetic at the actual
  numeric root of unity `e^{2πi/d}`, evaluated by walking the returned
  expression's node tree by hand rather than through anything under test. Two
  refusals are exercised concretely rather than merely asserted: a genuine
  pole at `ζ_3` (obstructed at every `n` tested, never silently specialised)
  and the support-shrinks case above.

  New: `QZeilbergerCertificate.specialize_at_root_of_unity`,
  `alkahest.experimental.QRootOfUnitySpecialization`,
  `alkahest.experimental.cyclotomic_polynomial`. Experimental.

- **Double-sum creative telescoping: `experimental.telescope2d`** (M4).
  Everything `zeilberger`/`q_zeilberger` reach is a single sum over one
  index; `telescope2d` is the Apagodu–Zeilberger generalization to **two**
  bound indices — `F(n,j,k)` proper hypergeometric in each of `n`, `j`, `k`
  — finding `a_0(n), …, a_J(n)` (not all zero) and *two* rational
  certificates `c_1, c_2 ∈ Q(n,j,k)` with
  `Σ_i a_i(n)·F(n+i,j,k) = Δ_j(c_1·F) + Δ_k(c_2·F)`, re-checked as an exact
  identity in `Q(n,j,k)` — never trusted from the search — before being
  returned. This is a genuinely scoped-down piece of a much larger problem
  (full Wegschaider reduction is arbitrary rational summands over arbitrarily
  many indices), not a claim to have closed it: see the honest-limitations
  list in `alkahest_cas::holonomic::telescoping2d`'s module docs.

  **Method.** There is no standard 2-D analogue of Gosper's normal form for a
  general proper hypergeometric `F(n,j,k)`, so unlike the single-sum engine
  this is undetermined coefficients, not a normal-form reduction: the
  certificate ansatz `c_1 = P_1(n,j,k)/E_1`, `c_2 = P_2(n,j,k)/E_2` uses a
  *fixed*, search-independent denominator built from `F`'s own shift-ratio
  denominators — and, critically, **not just the raw denominator of the
  ratio being telescoped in that direction**. A certificate built from a
  product of two single-sum WZ pairs needs a factor from the *other*
  direction's `n`-shift ratio too (`c_1 ∝ R_A(n,j)·B(n+1,k)/B(n,k)` for
  `F = A(n,j)·B(n,k)`), which is why `E_1 := D_j·∏_i D_{n,i}` and
  symmetrically for `E_2`. Clearing that denominator turns the identity into
  one linear system over `Q`, solved by plain Gaussian elimination (nullspace,
  not a fixed right-hand side, since the leading `a_J(n)` is not normalized to
  a constant up front) — reusing nothing from `qfield`'s `Q(n)[k]` tower,
  because a fixed-denominator linear ansatz never needs that tower's gcd
  machinery; a new, deliberately simpler sparse `Q[n,j,k]` polynomial ring
  covers everything the search needs (`holonomic::telescoping2d::poly`).

  **The boundary is four strip sums, not four corner evaluations** — the
  single most important thing to get right at two dimensions, and the
  subject of its own module (`holonomic::telescoping2d::boundary`) kept
  strictly apart from the search so a returned certificate is checkable
  without reference to how it was found. Telescoping
  `Σ_j Σ_k [Δ_j G_1 + Δ_k G_2]` over a rectangle gives
  `Σ_k[G_1(j_hi+1,k) − G_1(j_lo,k)] + Σ_j[G_2(j,k_hi+1) − G_2(j,k_lo)]` —
  four *one-dimensional sums* along the rectangle's edges. Summing a strip in
  closed form is in general its own creative-telescoping problem, so this
  version proves the **sufficient** (not necessary) condition that each strip
  is identically the zero function of its remaining free variable — via the
  same `1/Γ` non-positive-integer-argument identity the single-sum engine's
  order counting uses, checked either on `F`'s own gamma factors (the natural
  boundary) or on the certificate's own numerator (the classical-WZ-style
  case, e.g. a certificate `∝ k` that is zero at `k = 0` even though `F`
  itself is not). It is honestly conservative: only **constant** (not
  `n`-dependent) rectangles are supported — the `n`-dependent-range case
  needs the same `D_i(n)` correction-term bookkeeping
  `holonomic::boundary`'s `b(n)` formula has, doubled for two independently
  moving bounds, and this version does not implement it — and an unresolved
  strip is reported `"unknown"`, never guessed. `BoundaryStatus2d::Nonzero`
  exists in the type for parity with the single-sum engine's three-valued
  discipline but is not yet produced by any code path: an inhomogeneous
  boundary term needs closed-form strip summation this version does not
  attempt.

  Verified on `S(n) = Σ_j Σ_k C(n,j)·C(j,k)`, a genuinely **non-separable**
  double sum (`C(j,k)` couples to the outer sum's own index `j`, so it is not
  a product of two independent single-index sums) with a known closed form,
  `S(n) = 3ⁿ` by the binomial theorem — checked by direct exact summation
  (`rug::Rational`, not floats) against the returned order-1 recurrence, and
  again on `2ⁿ·C(10,j)·C(j,k)` (the same non-separable coupling with the `n`
  dependence factored out, so the boundary genuinely is `n`-independent and
  `Vanishes` is provided rather than refused) — plus the separable-product
  fallback case (`C(n,j)·C(n,k)`, `S(n) = 4ⁿ`) the task's own scoping note
  flags as a weaker test, kept in the suite explicitly labeled as such since
  it does not exercise the corner-term logic.

  New: `alkahest.experimental.telescope2d`,
  `alkahest.experimental.Telescoping2dCertificate`. Experimental — the ansatz
  degree budgets, the fixed-denominator choice and the constant-rectangle
  boundary restriction are all real, stated limitations of this first cut,
  not polish left for later. Refusals: `E-HOLO-040` outside the proper
  hypergeometric class, `E-HOLO-041` when the bounded search is exhausted,
  `E-HOLO-042` for a malformed call.

- **Validated bounds reach five more functions: `asinh`, `acosh`, `atanh`,
  `erf`, `erfc`.** `bound_on_box` — and everything built on it,
  `verified_sign`, `verified_no_roots`, `verified_integral` — covered exactly
  13 elementary primitives and refused these five with `E-VALIDATED-001` even
  though `numeric_ball` reported `True` for all of them. The gap was not an
  oversight about ball arithmetic; a Taylor model needs a per-function rule
  with a *rigorous* remainder, and there was none. There is now, and each one
  is a proof rather than an estimate:

  - `asinh` uses `aₖ = b_{k-1}/k` from the exact point recurrence for
    `(1+x²)^{-1/2}` and the remainder bound
    `1/[(p+1)(1+ξ²)^{(p+1)/2}]`, which follows from
    `Σⱼ C(n,j)AⱼA_{n-j} = n!` for `Aⱼ = (2j-1)!!/2ʲ` — the same bound `atan`
    already used, arrived at the same way.
  - `acosh` uses the matching recurrence for `(x²-1)^{-1/2}`. Every term of
    its Leibniz expansion carries the sign `(-1)ⁿ`, so no cancellation is
    available to exploit and `|v⁽ⁿ⁾(x)| ≤ n!(x-1)^{-(n+1)}`, decreasing in
    `x`; the supremum therefore sits at the low end of the enclosure.
  - `atanh` expands `(1-x²)^{-1} = ½[(1-x)^{-1} + (1+x)^{-1}]` in closed
    form and bounds the two poles' contributions separately, each at its own
    maximising endpoint.
  - `erf` expands the Gaussian by its Hermite recurrence and bounds
    `|(e^{-z²})⁽ⁿ⁾|/n!` by a Cauchy estimate on a circle of radius `r`,
    using the *exact* minimum of `Re(z²)` on that circle rather than a
    triangle-inequality proxy. The bound holds for **every** `r > 0`, so the
    `r` the rule picks is a tightness choice that cannot affect soundness.
    `erfc` is `1 - erf`.

  All three inverse hyperbolics carry their branch's domain and refuse
  outside it with `E-VALIDATED-003` rather than returning a number: `acosh`
  needs the whole enclosure strictly above `1`, `atanh` needs it strictly
  inside `(-1, 1)` — **both** ends, so `[0.5, 2]` is refused rather than
  bounded off-domain — and a box that merely *touches* a boundary is refused
  too, because the derivative is unbounded there. `asinh`, `erf` and `erfc`
  are entire and never refuse on domain grounds; `asinh` in particular is
  expanded directly rather than through `log(x + √(1+x²))`, which loses the
  argument to cancellation for `x ≪ 0`.

  Each rule also computes a second, independent bound on the same remainder —
  a geometric bound on the series tail inside the disc of convergence — and
  keeps whichever is smaller. The Lagrange form takes the supremum of
  `|f⁽ᵖ⁺¹⁾|` over the *whole* argument enclosure, which near a singularity
  dwarfs the coefficients at the centre: on `atanh` over `[-0.5, 0.5]` at
  order 10 it gives `9.1e-2` where the tail bound gives `8.9e-5`.

  `capabilities()["primitives"][i]["taylor_model"]` and `bounds_supported`
  pick all five up with no edit — both are derived by running the evaluator,
  which is what that design was for.

- **…and five more after them: `bessel_j0`, `bessel_j1`, `digamma`, `gamma`,
  `lambert_w`.** That completes the M7 list and takes validated bounds from 13
  primitives to **23**. The two Bessel functions are the ones worth reading
  about: they *oscillate*, which is the property that made 3.8's ball kernel
  for them unsound (it hulled the two endpoint values, so on `[-1, 1]` it
  excluded `J₀(0) = 1`, the function's own maximum). Nothing in the rules
  below assumes monotonicity of anything.

  - `bessel_j0` / `bessel_j1` get both halves from one identity. Iterating
    `2Jν′ = J_{ν−1} − J_{ν+1}` by Pascal's rule gives
    `Jν⁽ⁿ⁾ = 2⁻ⁿ Σⱼ (−1)ʲ C(n,j) J_{ν−n+2j}`, which is the coefficient
    formula; feeding `|J_m(x)| ≤ 1` (from `J_m(x) = (1/π)∫₀^π cos(mθ − x sin θ)
    dθ`) into the same line collapses the binomials against the `2⁻ⁿ` and
    yields `|Jν⁽ⁿ⁾| ≤ 1` for **every** order, which is exactly the remainder
    `sin` and `cos` already use. Entire, so no box refuses on domain grounds.
    A Cauchy estimate against the entire-function growth
    (`|Jν(z)| ≤ |z/2|^ν e^{|Im z|}/ν!`) is available and was tried; minimised
    over the circle radius it is a factor `√(2πn)` *worse*, so it is not used.
  - `digamma` expands with `aₖ = (−1)^{k+1} ζ(k+1, m₀)`, the Hurwitz zeta
    being what `ψ`'s Taylor coefficients literally are. The remainder is
    `ζ(p+2, ξ)`, which every term of shows is decreasing in `ξ`, so its
    supremum is at the low end of the enclosure, where
    `ζ(s, L) ≤ L^{-s} + L^{1-s}/(s−1)` by comparing the sum to its integral.
  - `gamma` had **no ball arithmetic at all** before this release, let alone a
    Taylor rule. Coefficients come from `Γ′ = ψΓ` as the convolution
    `c_{n+1} = (1/(n+1)) Σⱼ dⱼ c_{n−j}` over the same `ψ` coefficients. The
    remainder is Cauchy's estimate, closed by two facts from the Euler
    integral: `|Γ(u+iv)| ≤ Γ(u)` for `u > 0`, and `Γ″ > 0` on `(0, ∞)` so `Γ`
    is convex and its maximum over an interval is at an endpoint. Every circle
    radius gives a valid bound, so the candidates the rule tries are a
    tightness choice only.
  - `lambert_w` uses the classical closed form
    `W₀⁽ⁿ⁾ = e^{−nw} pₙ(w)/(1+w)^{2n−1}` with `pₙ` carried as exact integers,
    and bounds each of the three factors by its own proved monotonicity in
    `w`, over a panelled split of the `w` range so the three maxima are not
    all taken at the same end.

  `Γ` and `ψ` refuse anything reaching `0` with `E-VALIDATED-003` — the strips
  between the negative poles are analytic but are not covered, since both the
  coefficients and the remainder are written for the positive axis. `W₀`
  refuses at and left of `−1/e`, where it has a square-root branch point and
  every derivative is unbounded; that guard is not a comparison against a
  rounded `−1/e` but the failure of the certified bracket itself.

  The Hurwitz zeta underneath `ψ` and `Γ` is Euler–Maclaurin after 100 exact
  terms, with exact rational Bernoulli numbers from their own recurrence and a
  remainder of twice the first omitted term — the factor 2 covering both
  standard forms of the Euler–Maclaurin remainder so the bound does not depend
  on which convention is quoted. It is cross-checked in tests against MPFR's
  Riemann `ζ(s)` at `a = 1` and against `ζ(s,a) − ζ(s,a+1) = a^{-s}` at
  arbitrary `a`.

  The reported set is now 23 names. `floor` and `ceil` are the only two left
  with ball arithmetic and no Taylor rule, and that is deliberate rather than
  pending: they are not differentiable, so on a box containing an integer no
  Taylor model exists, and on one that does not they are a constant. The four
  elliptic integrals still have neither ball arithmetic nor a Taylor rule.

- **`ArbBall::lambert_w0` was unsound; `ArbBall::gamma` is new.** The Lambert
  kernel hulled two `f64` evaluations — ~10⁻¹⁶ of error — inside a ball whose
  radius it set to `|mid|·2⁻ᵖʳᵉᶜ`, 5·10⁻⁴⁰ at the default 128 bits. On the
  degenerate ball `[1, 1]` that is an enclosure of width 10⁻³⁹ centred
  3·10⁻¹⁷ from `W₀(1)`: an interval that does not contain the value it
  encloses. It is now a Newton *guess* whose bracket is certified afterwards by
  evaluating `g(w) = w·eʷ` in ball arithmetic at both candidate endpoints —
  `g` is strictly increasing on `w ≥ −1`, so `g(v) ≤ x` proves `W₀(x) ≥ v` and
  `g(u) ≥ x` proves `W₀(x) ≤ u`, and how the candidates were produced never
  enters the argument. `ArbBall::gamma` uses convexity for both ends: the
  maximum of a convex function on `[a, b]` is at an endpoint, and each of its
  two tangent lines is a lower bound everywhere on the interval.

- **`guess_holonomic(terms, max_order, max_degree)` — the guessing half of
  *guess then prove*, with the guard that makes a fit mean something.** Alkahest
  shipped the proving half only (`zeilberger`), so the 2026-08-13 autoresearch
  run hand-rolled a fitter out of `Matrix.nullspace` and exact rationals in
  ~30 lines; it recovered Motzkin's recurrence from 21 terms, and every loop
  since has rewritten it. The rewrite is not the problem — the guard is. A
  recurrence of order `J` with degree-`D` coefficients has `U = (J+1)(D+1)`
  unknowns, and a homogeneous system in `U` unknowns has a nonzero solution the
  moment it has fewer than `U` independent equations, *whatever the numbers
  are*. An unguarded fitter therefore returns a recurrence for every input it
  is ever shown, and a hand-rolled one has nothing to say about which of those
  answers is evidence.

  So a candidate is fitted only where the terms **over-determine** it: at least
  `U + min_surplus` equations, `min_surplus` defaulting to `U` itself. The
  result carries `surplus_terms` (equations that were not needed and agreed
  anyway), `equations_used`, `dimension`, `untested_candidates` and
  `confirmed`, plus `evidence()` for logging the lot — `untested_candidates` is
  the minimality caveat, `0` when the returned order really is the smallest that
  fits anywhere in the bounds. Motzkin from 21 terms comes back order 2, degree 1,
  confirmed by 14 surplus equations; the same recurrence from the 7 terms that
  merely *determine* it is refused (`E-HOLO-005`), because that fit exists for
  any seven numbers.

  **`None` means the grid was swept and nothing fit.** If some `(order, degree)`
  candidate had to be skipped for lack of terms, the call refuses with
  `E-HOLO-005` and says how many terms the cheapest skipped candidate needs,
  rather than returning a negative for a question it never asked — 60 primes
  give an honest `None`, 40 primes give a refusal. This is the
  `relation_confidence` discipline for sequences, and it is deliberately the
  same shape: `check_evidence=False` is the escape hatch, in the role
  `check_precision=False` plays on `guess_relation`.

  Exact throughout: Python `int` of any size and `fractions.Fraction`;
  a `float` term is refused, not rounded. `holds_for(more_terms)` re-checks
  exactly against data the fit never saw and `to_exprs(pool, n)` hands the
  coefficient polynomials back to the expression layer, so a guess can be
  compared with a `ZeilbergerCertificate` coefficient for coefficient. Pure
  Python (`alkahest.guess_holonomic`, `alkahest.GuessedRecurrence`), because the
  one mathematical step is `Matrix.nullspace`, which was already in the kernel
  and already exact — everything on top is composition, validation and evidence
  bookkeeping, which is the Python column of `CONTRIBUTING.md` § *Rust vs
  Python* line by line, and the part where being easy to audit matters more than
  being fast.

- **`zeilberger` now says whether its order is minimal, and `minimal=True`
  makes it so.** The certified order is usually the publishable part — the one
  novel result of the 2026-08-13 run (OEIS A359643) was interesting *precisely*
  because the certified order was 4 where OEIS recorded a guessed 5. But nothing
  in 3.9.0 established minimality, and the natural assumption that an ascending
  search gives it for free is **false for this search**: since the iterative
  deepening added earlier in this cycle, the `(order, degree)` grid is visited
  cheapest-estimated-cost-first (weight `3·(J−1) + d`), which is exactly what
  took Dixon, Franel and Apéry from timeout to sub-second. Cheapest-first is not
  order-ascending, so the plan can reach a cheap order-2 probe long before an
  expensive order-1 one, and a returned order 2 does **not** rule out order 1.

  `ZeilbergerCertificate.order_is_minimal` reports it, computed from the probes
  that actually happened rather than from the mode, so it cannot drift away from
  what the search did. `False` means **not established**, never "a lower order
  exists" — a lower-order relation that had been found would have been the one
  returned. It is `True` for free at order 1, and `True` whenever the
  cost-ordered plan happened to spend every lower order first, which it does at
  narrow `max_degree`.

  `zeilberger(..., minimal=True)` walks the grid order-major instead — every
  degree `0..max_degree` at order `J` refused before order `J+1` is tried — so a
  returned order really is the least one reachable within `max_degree`. Same
  bounds, same exact `Q(n)(k)` verification, same certificate; only what was
  ruled out differs. **The default is unchanged and stays fast**, deliberately:
  the price of minimality is the whole hopeless low-order sweep, and it is
  charged against `max_degree`, which is the bound minimality is claimed
  relative to. Measured, `max_order=4`, default → `minimal=True`: Franel
  0.23 s → 0.23 s at `max_degree=6` (the default was already minimal there),
  0.56 s at 8, **9.7 s at 16**; Apéry 0.08 s → 0.11 s at 6, 0.29 s at 8,
  **13.1 s at 16**. The default column is flat in `max_degree` and the
  `minimal=True` column is not, which is the whole trade in one line. The sweep
  grows like `3^d`, so the useful move
  is to claim minimality against the smallest `max_degree` you are willing to
  state rather than against the default. In Rust the same thing is
  `holonomic::zeilberger_search` with `OrderSearch::MinimalOrder`, returning a
  `ZeilbergerSearchReport`; `zeilberger()` keeps its signature and its
  cost-ordered behaviour exactly.

- **A certified recurrence now answers the next question: how fast does the
  sequence grow?** `alkahest.experimental.asymptotics_from_recurrence(rec, n,
  terms=…)` takes what `zeilberger` or `guess_holonomic` just produced — or a
  bare list of coefficient polynomials — and returns
  `RecurrenceAsymptotics`. Until now the asymptotics family
  (`asymptotic_expand`, `euler_maclaurin`, `coefficient_asymptotics`) and the
  holonomic subsystem did not compose at all, so every loop that certified a
  recurrence stopped one step short of the growth law the recurrence already
  determines.

  **The derived half and the fitted half are separate fields, because the
  constant is the part that is usually hard and is exactly the part a loop is
  tempted to overclaim.** Poincaré–Perron gives the growth rate `ρ` (a root of
  the characteristic polynomial `χ(t) = Σᵢ [n^D]pᵢ · tⁱ`) and the polynomial
  exponent `α = −χ₁(ρ)/(ρ·χ'(ρ))` in `u(n) ~ C·ρⁿ·n^α`; both are functions of
  the coefficient polynomials and nothing else, and both come out **exact** as
  `growth_rate_exact` / `polynomial_exponent_exact` when the root is rational.
  The connection constant `C` does *not* follow from the recurrence — it is
  determined by the initial conditions — so it is extrapolated numerically from
  the exact terms, exposed only as `connection_constant`, and carries
  `connection_constant_converged` and `connection_constant_drift` from a second,
  independent extrapolation over a smaller index range. `evidence()` returns the
  two halves under separate `derived` / `fitted` keys, and `report()` is the
  family's usual `AsymptoticReport`, whose `rigor` here is always
  `numerically_consistent` and whose hypotheses name the fitted constant as
  `assumed`. This is the discipline `euler_maclaurin` uses for the `γ` in
  `H_n ~ log n + γ + …`, applied to the same kind of quantity.

  Measured against sequences whose asymptotics are known independently: the
  fitted constant reproduces `1/√5` for Fibonacci to `1.8e-14` (the control —
  it is the one case where `C` is derivable), `1/√π` for the central binomial
  coefficients to `3.6e-11` and for Catalan to `8.4e-9`, `3√3/(2√π)` for
  Motzkin to `7.9e-8`, and `(1+√2)²/(2^{9/4}π^{3/2})` for Apéry to `5.5e-11`.
  For **OEIS A359643** — the one novel result of the 2026-08-13 run — the
  order-4 recurrence gives `ρ = 283/27` and `α = −1/2` exactly and fits
  `C = √(283/3)/(2^{7/2}√π)` to `1.7e-10`, i.e. the whole of the entry's
  `a(n) ~ 283^(n+1/2)/(2^(7/2)·√(πn)·3^(3n+1/2))`.

  **Poincaré–Perron's hypotheses are stated and checked, not assumed.** Each way
  they fail gets its own `verdict` and no growth rate at all, rather than one of
  the roots reported as though it had won: `equal_modulus_roots` (`u(n+2) =
  4u(n)` has roots `±2` and its solutions oscillate), `repeated_dominant_root`
  (`χ'(ρ) = 0`, so the exponent formula does not apply), and
  `degenerate_leading_coefficient` (`deg χ < J`, a root at infinity, outside the
  theorem). Root multiplicity is **exact** — it comes from the squarefree
  decomposition of `χ` over `ℚ`, not from clustering numeric roots, which
  matters because A359643's `χ = (t−1)³·(27t−283)` has a triple root that is not
  the dominant one and a tolerance-based test would refuse the case. A leading
  coefficient vanishing at finitely many `n` is a reported side condition
  (`singular_indices`), not a refusal. And because Poincaré's conclusion is only
  that `u(n+1)/u(n)` tends to *some* root, a sequence whose dominant component
  is zero — the constant solution of `u(n+2) = 3u(n+1) − 2u(n)`, say — is caught
  and reported as `follows_dominant_root == False` instead of being handed the
  generic solution's growth rate. The sequence is run forward in exact rational
  arithmetic for that reason: `f64` iteration is attracted to the dominant
  solution and would manufacture the component the check exists to look for.

  In Rust: `holonomic::asymptotics_from_recurrence`, with
  `CharacteristicAnalysis`, `ConnectionConstant` and `PerronVerdict`.

- **Validated-bounds coverage is queryable: `bounds_supported(expr)` and a
  `taylor_model` bit in `capabilities()["primitives"]`.** The only
  per-function coverage flag the agent contract exposed was `numeric_ball`,
  and it is not the flag that governs `bound_on_box` / `verified_integral` /
  `verified_no_roots` / `verified_sign`. Ball arithmetic is *pointwise*; a
  Taylor model needs a rule with a rigorous Lagrange remainder, and eleven
  primitives have the first without the second — `erf`, `erfc`, `bessel_j0`,
  `bessel_j1`, `digamma`, `lambert_w`, `acosh`, `asinh`, `atanh`, `floor`,
  `ceil`. So
  `numeric_ball` said `True` for `bessel_j0` and every bound over a box died
  on `E-VALIDATED-001`. The boundary was enforced correctly and could not be
  found ahead of time, which is how a planning loop loses a whole designed
  workload (Turán-type inequalities for Bessel functions, in the 2026-08-13
  autoresearch run) to a route it could have ruled out for free.

  `taylor_model` reports it per primitive. When the flag was added that was
  `True` for the elementary fragment (`exp`, `log`, `sqrt`, `sin`, `cos`,
  `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `abs`) and `False`
  for every special function; by the time 3.9.0 shipped the two rounds of
  Taylor-model rules above had moved ten of those names across, leaving only
  `floor` and `ceil` with ball arithmetic and no rule. The flag needed no edit
  for either round, which is the point of deriving it.
  `ak.bounds_supported(expr)` asks for a whole expression, without
  running the bound: it is truthy when nothing in the expression will be
  refused as unsupported, and carries `.blocker` (the evaluator's own
  description of the first construct it has no rule for) and `.functions`
  (every blocking function, so a substitution can be planned in one round
  rather than found one at a time).

  **Neither is a maintained list.** Both are derived by running the real
  Taylor evaluator on a probe expression and asking whether it refuses with
  `E-VALIDATED-001` — a second hand-written table is how `numeric_ball` came
  to be read as coverage in the first place, and would have been a worse
  outcome than no flag at all. `tests/test_taylor_model_coverage.py`
  re-derives the bit the only other way there is, by calling `bound_on_box`
  on every registered primitive, and fails if the two ever disagree.

  `numeric_ball` itself is *accurate* and stays as it is: those eleven
  primitives really did have ball arithmetic. It answers a different
  question, and now says so next to a flag that answers this one. A `True`
  from either means "not `E-VALIDATED-001`" — a covered function can still be
  refused on a particular box for a domain violation (`E-VALIDATED-003`) or a
  non-finite enclosure (`-004`), which no box-free predicate can rule out.

  This is deliberately *not* folded into `certifiable`, which asks whether an
  operation emits a **Lean** certificate and answers from the certificate
  ledger. A rigorous enclosure is not a Lean proof term and the validated
  subsystem has no ledger rows; one predicate returning `True` for two kinds
  of evidence would be a worse contract than two predicates.

  New in Rust: `alkahest_cas::{taylor_model_refusal, taylor_model_blockers,
  taylor_model_supports, taylor_model_supports_call}` and
  `Capabilities::TAYLOR_MODEL` (also a `taylor_model` column in
  `CoverageReport::to_markdown`). `capabilities()["contract_version"]` stays
  `3`: the row gained a key and lost none, which is the same additive rule
  the `__all__` freeze check applies.

- **Gröbner results can be read back — `GbPoly.to_expr`, iteration over a
  `GroebnerBasis`, and `expr_to_gbpoly`.** Everything that returned a basis
  returned a handle nobody could open. `GbPoly` exposed only `is_zero` and
  `n_vars`; `GroebnerBasis` exposed only its own constructors plus `reduce` and
  `contains`, and `reduce` took a `GbPoly` that no exported function could
  build — `expr_to_gbpoly`, named in `compute_raw`'s own docstring, was never
  registered on the module. So `rosenfeld_groebner(...).final_basis()`,
  `triangularize(...)`, `primary_decomposition(...)` and a parametric `solve`
  all handed back objects whose only readable property was how many generators
  they had. Differential elimination was write-only: the input–output equations
  it computes could not be looked at, which is the whole of structural
  identifiability.

  Now: `alkahest.expr_to_gbpoly(expr, vars)` converts in, `GbPoly.to_expr()`
  converts back out, and `GbPoly.terms()` gives `(exponent tuple, exact
  int/Fraction)` pairs. A `GroebnerBasis` is a sequence — `len()`, indexing,
  iteration — with `polynomials()`, `to_exprs()`, `variables()` and an `order`
  property. `reduce()` now accepts an `Expr` as well as a `GbPoly`, so the
  membership and reduction API is reachable from expressions alone.
  `GroebnerBasis.eliminate(vars)` is bound too — the mdbook and Sphinx pages
  had documented it for releases, but it existed only in Rust, so the
  implicitization example on the solving page could not run.

  The part that needed a real fix rather than an accessor is the **variable
  ordering**. A `GbPoly` stores exponent vectors, not names, so a basis without
  its variable list cannot be read at all — and `rosenfeld_groebner` discovered
  its jet variables internally (`t`, `x`, `dx/dt`, `ddx/dt/dt`, …) and threw
  them away. Every object that hands out a `GbPoly` now carries that list:
  `RosenfeldGroebnerResult.variables()`, `GroebnerBasis.variables()`,
  `RegularChain.variables()`, and for a parametric `solve` the solve variables
  *followed by the free parameters*, which is the order the exponent vectors
  were actually built in. Asking for an `Expr` with too few variables named
  raises `ValueError` rather than silently misreading exponent slots.

  New in Rust: `alkahest_cas::gbpoly_to_expr`, `GroebnerBasis::order()`,
  `MonomialOrder::as_str()`, `solver::collect_parameters`, and
  `rosenfeld_groebner_ranked` / `dae_index_reduce_ranked`, which return the
  `DifferentialRanking` alongside the result (the existing entry points are
  unchanged wrappers, so no struct gained a field).

- **`DAE` can be read: `equations()`, `variables()`, `derivatives()`,
  `time_var`, `index`.** It previously exposed `n_equations` and `n_variables`
  and nothing else, so a prolonged system that reported six variables for a
  two-variable input gave no way to find out what any of them were.
  `pantelides` sets `index` on the DAE it returns — the number of
  differentiation rounds — and the equations it appended, plus the higher jets
  (`ddx/dt/dt`, …) they introduced, are now visible in `equations()` and
  `derivatives()`.
- **`UniPoly.leading_coeff`** — the leading (highest-degree) coefficient, `0`
  for the zero polynomial so it pairs with `degree == -1`. A **property**, per
  the accessor convention: it is a single FLINT coefficient read. It is also
  *exact*, returned as a Python `int` of any size, which `coefficients()` is
  not — that one is `i64` and truncates silently, so `3x² + 1` scaled by `2¹⁰⁰`
  reports a leading coefficient of `1` through the list and the true value
  through this accessor. Documented in `representations.md`, which had been
  showing a `leading_coeff()` *method* that never existed.

- **Gröbner bases over the coefficient field `Q(params)`: `GroebnerBasis.compute(polys, vars,
  params=[...])`, `experimental.ParametricGroebnerBasis` / `ParametricGbPoly`**
  (M9). Eliminating states from an ODE model's jet equations needs the rate
  constants to *not* enter the monomial order — computed over `Q[states, Y,
  params]` they are ordinary ring variables and generate S-pairs like any
  other, which is exactly the growth the elimination does not need. With
  `params=[...]` the same Buchberger engine (identical Gebauer–Möller pair
  management from `poly::groebner::pairs`, shared with the ℚ engine so a
  specialisation keeps every leading monomial and the whole pair schedule)
  runs over `Q(params)[vars]` instead: the parameters live in the
  coefficients as elements of `QParam`, a canonical, reduced fraction of
  sparse `ParamPoly`s (FLINT's `fmpz_mpoly_gcd` — a Hensel/Zippel hybrid, not
  a hand-rolled Euclidean gcd, which is exactly the swelling `holonomic::qfield`
  exists to avoid one variable further in) rather than ring monomials.

  Measured on a catenary compartmental model (linear ODE chain, output the
  first compartment, `n+1` derivatives eliminating `n` states down to the
  input–output relation): at `n = 4` states / 7 rate constants, the
  parametric route computes in 0.27s against 4.2s direct (both `Lex`,
  `--release`) — roughly 15× — and returns 5 total basis generators against
  25; at `n = 5` states / 9 rate constants the parametric route finishes in
  6.9s while the direct `Q[states, Y, params]` computation had not finished
  after 240s (>34× and counting). Numbers are wall-clock on one machine, not
  a guaranteed ratio — the point is the qualitative shape (S-pairs among the
  parameters are the cost the direct route pays and the parametric route
  never generates), not the specific multiplier.

  **The result is generic, and says so.** A leading coefficient in
  `Q(params)` can be a non-zero rational function of the parameters and still
  vanish at a particular parameter point, and there the basis the algorithm
  built is not the basis the same algorithm would have built over ℚ at that
  point. Every such assumption is logged as it happens — an inversion
  contributes its numerator and its denominator, an input coefficient
  contributes its denominator — and `conditions()` reports the union,
  factored into irreducible, primitive hypersurfaces so "wrong somewhere on
  this degree-12 surface" reads as a list of conditions rather than one
  opaque polynomial. `specialize(values)` substitutes and refuses with
  `ParamGroebnerError` (`E-PARAMGB-004`) on the locus rather than returning
  something that is not a basis; `is_regular_at` / `vanishing_conditions`
  check first. The set is sufficient, not necessary by construction — on the
  worked linear system `{a·x − y, x + y − 1}`, `a = -1` is a real
  disagreement (the direct basis over ℚ there is the unit ideal, not the
  2-generator triangular basis the generic formula predicts), while `a = 0`
  is flagged (the algorithm inverts `a` to make `a·x − y` monic) but the
  direct computation at `a = 0` agrees with the `a → 0` limit of the generic
  answer exactly — a refusal, not a wrong answer, on the conservative side of
  the report.

  Reads back the same way `GbPoly` does: `ParametricGbPoly.to_expr` /
  `.terms()`, `ParametricGroebnerBasis.to_exprs()`, `.conditions()` as
  `Expr`, and `.specialize(...)` returns an ordinary `GroebnerBasis` whose
  generators are `GbPoly` — the same read path issue #11 was about, so
  nothing here is write-only. `eliminate` has the same `Lex`-with-eliminated-
  variables-first contract as `GroebnerBasis.eliminate`, and refuses to
  eliminate a coefficient-field parameter (there is nothing to eliminate — it
  was never a ring variable) rather than silently ignoring the request.
  Errors: `E-PARAMGB-001` no generators, `E-PARAMGB-002` generators disagree
  on the variable/parameter shape, `E-PARAMGB-003` wrong specialisation
  arity, `E-PARAMGB-004` degenerate point (a result, not a malfunction — see
  above). New: `alkahest.experimental.ParametricGbPoly`,
  `ParametricGroebnerBasis`, `ParamGroebnerError`. Experimental; requires
  `--features groebner`. Tests: Rust `poly::groebner::parametric` and
  `poly::groebner::paramfield` (19 unit tests), Python
  `tests/test_parametric_groebner.py`, including a
  structural-identifiability worked example (two-compartment linear ODE
  model, states eliminated with rate constants in `Q(params)`, the recovered
  input–output relation checked against the model's characteristic
  polynomial `y'' − tr(M)·y' + det(M)·y = 0` at several parameter points).

- **Novelty filtering against OEIS: `experimental.novelty`** (M11). A search
  loop over this library can rediscover a known identity within the hour, and
  nothing before this told the difference between "produced 400 certified
  recurrences" and "produced three that nobody had". `RecurrenceClaim` puts a
  P-recursive relation into normal form — rescaling, sign flips, index shifts,
  a different clearing of denominators and a common polynomial factor are all
  quotiented out, so `(n+1)·u(n+1) − (4n+2)·u(n) = 0` and the same relation
  scaled by −2 and stated about `u(n+7)` hash equal via `claim_hash`, while
  genuinely different recurrences do not collide. `check_novelty(claim,
  sources, terms=…)` checks a claim against one or more sources — `OeisCache`
  offline (file-backed, the committed test fixture format) or `OeisWeb` when
  explicitly opted into (never constructed by default, serves from cache
  first, rate-limited, degrades to `unavailable` rather than raising when the
  network is unreachable) — and returns a `NoveltyVerdict`.

  **`NoveltyVerdict.found` is three-valued**, in the manner of
  `relation_confidence`'s tri-state `credible` and `GuessedRecurrence.confirmed`:
  `True` a source states the claim, `False` the sources searched do not state
  it (not "novel" — "not found in the one place looked"), `None` no source
  could answer. There is no `novel` attribute anywhere in the module, and
  `bool(verdict)` raises rather than reading `True`, because `if
  check_novelty(...):` is the exact sentence this module exists to prevent.
  `NoveltyVerdict.report()` carries the scope of the search — entries
  examined, statements compared, statements a parser could not use — so a
  negative's coverage is visible next to it. `RecurrenceClaim.from_text`
  parses OEIS's own prose formula lines by recursive descent over
  `+ - * / ^ ( )`, `n` and `a(n±k)`, refusing (returning `None`, never
  guessing) anything it does not model — a sum, a generating function, a
  reference to another sequence, an inhomogeneous relation — and every parsed
  line is re-checked against the entry's own data before it can produce a
  match, since a formula line is prose from a third party and the parser is
  the weakest link in the chain. No test in the repository requires the
  network: the offline path runs against `tests/data/oeis_novelty_fixture.json`,
  a cache recorded once from oeis.org (© The OEIS Foundation Inc., CC
  BY-NC-SA 4.0) and committed, exercising the four sequences this project
  already certifies recurrences for (Apéry, Motzkin, Catalan, central
  binomial) plus the session's own novel result (A359643, which OEIS records
  only as an unproved conjecture — the distinction `hedged` exists to keep
  separate from a proof).

- **`sos_decompose` searches the general PSD Gram cone and Reznick
  multipliers, not just diagonal dominance** (M10). `sos_decompose` already
  covered the diagonally dominant subcone (`gram::dsos_search`) — fast, but a
  strict subset of SOS, so it refused things that genuinely are SOS just not
  DSOS. It now falls back, when DSOS fails, to searching the *entire* PSD
  Gram cone (`real::sos::psd::psd_search`) over the same monomial basis, and
  — when even that fails on `p` itself — to a Positivstellensatz-lite
  multiplier search: trying `(x_1²+…+x_n²)^N·p` for `N = 1..4`
  (`MAX_MULTIPLIER_POWER`) and searching the PSD Gram cone of the product, up
  to a monomial-count budget (`MAX_MULTIPLIER_BASIS_LEN`). This is the
  standard route past DSOS's and even plain-SOS's incompleteness: some
  positive-definite forms are not SOS at all (Hilbert 1888), but Reznick's
  theorem guarantees `(Σxᵢ²)^N·p` is SOS for *some* `N` — the search just
  does not know `N` in advance and is honest when it runs out of budget
  before finding it (`multiplier_search_reports_undecided_not_not_sos_when_out_of_budget`):
  `SosError::NoCertificate` (`E-SOS-002`), never a claim that no certificate
  exists or that `p` is not SOS.

  The PSD Gram search underneath (`real::sos::psd`, `real::sos::sdp`,
  `real::sos::linalg` — an exact rational affine-system solver that reports
  free parameters on underdetermined systems, a Jacobi eigendecomposition,
  and a PSD-cone projection) is a floating-point *suggestion* mechanism only:
  it proposes a Gram matrix numerically, rounds it to nearby rationals, and
  the rounded result is re-expanded and compared against the target with
  exact rational arithmetic before anything is returned — a `Some` here is
  always sound regardless of what the numeric search actually converged to,
  the same discipline the rest of this module already applies. The search
  itself anneals a shrinking sequence of eigenvalue floors with several
  random restarts (`psd::FLOOR_SCHEDULE`, `psd::multistart_anneal`) rather
  than a single fixed-floor pass, specifically because the certificates this
  feature exists for are frequently *tight* — the witnessing Gram matrix is
  PSD but singular, sitting exactly on the boundary of the PSD cone rather
  than its interior, which a plain fixed-floor search reliably stalls short
  of.

  **Motzkin and Robinson now both certify — the gap that motivated this
  feature is closed, though not unconditionally.** The first version of this
  search (annealed alternating projection with several random restarts)
  reliably fell short of both classical textbook examples: a diagnostic
  trajectory (`psd::diag::diag_step1_step2_trajectory_and_family_sanity`)
  showed it converging *monotonically* toward Motzkin's boundary certificate
  (minimum eigenvalue running from roughly `−1.6` to roughly `−0.0018` as the
  floor annealed to `0`) without ever closing the last, asymptotically slow
  stretch to exactly `0` — the textbook signature of alternating projection
  stalling at a *tangential* (non-transversal) set intersection, which is
  exactly what a *singular* witnessing Gram matrix (sitting on the PSD cone's
  boundary rather than its interior) produces. The search now also tries
  Douglas–Rachford splitting with over-relaxation
  (`sdp::Family::douglas_rachford_from`) and a facial-reduction step
  (`psd::facial_reduction_search`) — both are standard escapes for exactly
  this stall — and with them, both `(x²+y²)·Motzkin(x,y)` (the affine,
  2-variable case, found via the full `sos_decompose` multiplier search, not
  just a hand-fed pre-multiplied target) and `(x²+y²+z²)·Robinson(x,y,z)`
  (via `psd_search` directly) are found and exactly re-verified —
  `real::sos::tests::motzkin_certifies_via_a_reznick_multiplier` and
  `psd::tests::psd_search_certifies_robinsons_form_with_a_reznick_multiplier`
  check the identities by hand, independent of the search that proposed them.

  **What's still open, now attempted to closure and precisely quantified
  (2026-08-17, round 3):** the homogeneous 3-variable form of Motzkin,
  `(x²+y²+z²)²·(x⁴y²+x²y⁴−3x²y²z²+z⁶)` — multiplier power `N = 2`, not `N = 1`
  (round 2 had already narrowed the target to this specific case and power;
  `N = 1` is not classically expected to work for this *homogeneous ternary*
  form at all, unlike the affine 2-variable case) — still is not found
  (`psd::tests::psd_search_does_not_yet_reach_homogeneous_motzkin_times_sum_of_squares`),
  despite a real attempt rather than a budget skip. Three approaches were
  tried:

  1. **Deep Douglas–Rachford on the raw 165-free-parameter family:** escalating
     from 15,000 to 615,000 cumulative iterations moved the minimum eigenvalue
     from `≈ −1.9·10⁻⁶` to only `≈ −1.7·10⁻⁷` — real progress, but clearly
     *sublinear*, not the finite-step convergence a transversal intersection
     would show.
  2. **Symmetry reduction** (new: `psd::symmetry_reduced_search`,
     `psd::detect_polynomial_symmetry_group`) — added specifically because of
     this case. `q`'s own signed-permutation symmetry (every exponent is even,
     so all eight sign patterns fix it; swapping `x, y` also fixes it — order
     16) restricts the 165-parameter family to a **26**-parameter symmetric
     slice, by construction still reproducing `q` exactly (a `G`-average of the
     original family). Deep DR on the smaller family converges *faster in wall
     time* (roughly 30× per iteration) but at essentially the **same iteration
     count** it does not reach a meaningfully different minimum eigenvalue
     (`≈ −2.4·10⁻⁸` at 2,000,000 iterations) — the bottleneck is the
     intersection's geometry, not raw parameter count.
  3. **Exact algebraic zero-vector restriction**, going further than either
     prior round: Motzkin's zero at `(1,1,1)` forces `Q·z(1,1,1) = 0` for any
     PSD witnessing `Q` (`z(1,1,1)` is literally the all-ones vector on this
     degree-5 basis — no rounding, no numerics). Imposing that exactly on the
     symmetric family collapses it again to **16** parameters. An additional
     tangent-direction candidate (the exact gradient of the basis vector at
     `(1,1,1)`) was also tried and found *inconsistent* with the family —
     confirming the corank contributed by this zero really is 1, not a missed
     higher-corank guess. Deep DR on this 16-parameter family still does not
     close it: 6,000,000 iterations reach only `≈ −1.4·10⁻⁸`, and rational
     rounding fails even with denominators tested up to roughly `10⁹`.

  All three lines of evidence agree: this is now a genuine, quantified
  numerical-hardness finding (an unusually slowly-converging tangential
  intersection), not an under-tuned budget or an unexplored structural avenue.
  `symmetry_reduced_search` ships anyway as a real, general capability — it is
  wired into `psd_search` as a further fallback (only once the direct search
  and facial reduction have both already failed, so it adds no cost to any
  case those already close) and will help *other* targets whose Gram family
  happens to have exploitable symmetry, even though it did not close this one.
  So a boundary-only certificate is still not guaranteed to be found in
  general; `E-SOS-002` still means "not found within this search", never "not
  SOS" or "not non-negative".

  Everything reachable today is exact end to end: `verify()` re-expands
  every returned certificate with exact rational arithmetic, `to_lean()`
  emits a sorry-free Lean sketch, and `PositivityCertificate.multiplier()` is
  populated exactly when the certificate needed one (`None` for a direct SOS
  decomposition — a method rather than a field, since adding a field to this
  already fully-public, exhaustively-constructible struct is a semver break
  regardless of the field's own visibility; see the method's own doc comment).
  No new public API surface — `sos_decompose` and `PositivityCertificate` are
  unchanged in shape; this is entirely a strengthening of what the existing
  search covers before it refuses.

### Performance

- **`zeilberger`'s exact `Q(n)(k)` post-processing no longer swells its own
  coefficients.** With the search fixed (below), what was left was entirely
  after it: on `Σ_k C(n,k)³` the search reached `(order 2, degree 3)` in 0.22 s
  and the run then spent ~29 s normalising the certificate and re-verifying it.
  The cause was `PolyK::gcd` — a textbook Euclidean remainder sequence over the
  *field* `Q(n)`, whose coefficients are rational functions in `n`: every
  division step adds numerator and denominator degrees and no step ever removes
  content, the classic intermediate-expression-swell blowup, and
  `RatK::normalize` ran it on every normalisation. The gcd now leaves the field
  and runs **Brown's subresultant PRS in `Z[n][k]`** (Collins 1967, Brown 1971;
  Knuth TAOCP 2 § 4.6.1), with both cofactors divided out in the same integral
  domain, and `Q[n]` gcds (`rn_mul` / `rn_add` / `rn_inv`, which cancel
  crosswise now rather than reducing the full cross-multiplied product) go
  through the same subresultant sequence over `Z[n]`. At the shipped defaults,
  measured before and after on one machine: `Σ (−1)^k C(n,k)³` **1.6 s →
  0.11 s** (15×), `Σ_k C(n,k)³` **56 s → 0.07 s** (800×),
  `Σ_k C(n,k)²C(n+k,k)²` **16.5 s → 0.05 s** (330×, and still Apéry's
  recurrence coefficient for coefficient). Two OEIS targets that timed out past 300 s at certificate
  degree ≥ 3 are now decided: **A357510** `Σ k·C(n,k)²·C(n+k,k)²` and
  **A357512** `Σ k⁵·C(n,k)²·C(n+k,k)²` both yield a verified order-3 recurrence
  in under a second. This is a change of algorithm, not of contract: a monic
  gcd is unique, so every certificate is the same one as before, and every
  certificate is still checked as an exact `Q(n)(k)` identity before it is
  returned — nothing here is probabilistic and no verification was weakened.
- **`zeilberger`'s `max_order` / `max_degree` are now upper bounds instead of
  starting points.** The search used to sweep certificate degrees
  `d = 0..=max_degree` at order 1 before ever trying order 2, and a single
  degree probe gets ~3× more expensive per degree step (measured on
  `Σ (−1)^k C(n,k)³`: 0.7 ms at `d = 0`, 0.6 s at `d = 7`, 84 s at `d = 12`).
  Every order ≥ 2 identity — Dixon, Franel, Apéry — therefore ran for minutes
  or never at the shipped defaults while being seconds away at `max_degree=4`,
  i.e. **raising the bound made easy inputs slower rather than admitting harder
  ones**. The `(order, degree)` grid is now visited by iterative deepening,
  cheapest estimated probe first (one extra order is priced at three extra
  degrees, which is what the measurements say), and the first *verified*
  relation is returned. `Σ (−1)^k C(n,k)³` at the defaults goes from >400 s
  (killed) to **0.67 s**; `Σ_k C(n,k)³` from >400 s (killed) to ~31 s. Nothing
  is skipped — the plan still visits every pair inside the bounds, so an
  exhausted search costs what it always did — and verification is unchanged: a
  candidate that fails the exact `Q(n)(k)` check is still discarded, never
  returned.

### Testing

- The deterministic silent-error gate grew from **213 to 241 scored cases**
  (`tests/silent_errors/`), still at **0 silent errors**. The new cases cover the
  PSLQ precision verdict as a *word* rather than a truthy value, so an `unknown`
  verdict cannot silently collapse into a pass.
- **`test_run_with_wall_fallback_bounds_a_cooperative_callee` no longer fails
  because the box is busy.** Its bound read `elapsed_ms < 20 * 300` against a
  call that measurably costs 2.7–5.3 s of work when it is behaving — a 13%
  margin — so it went red repeatedly during saturated parallel runs and always
  passed in isolation, which is the pattern that teaches people to ignore red.
  The bound is now on **process CPU time**, which tracks the work the callee
  actually did rather than the time the harness spent waiting for it. Wall time
  here is `wall_ms` — a real-time timer, which does not stretch — plus the join
  of a still-running worker, and only that second term was being measured
  against load. Idle vs. 24 spinners on 12 cores: wall 5.3 s → 24.9 s (4× over
  the old bound, a guaranteed failure), CPU 5.3 s → 8.8 s against a 60 s
  ceiling. The
  property is unchanged and still enforced from both ends — a callee that stops
  seeing the budget burns CPU without limit and trips the assertion, and one
  that stops coming back at all trips the `timeout(120)` marker — plus the test
  now also pins that the call ended on the fallback's own join rather than on
  some other check that raises the same code.

## 3.8.0 — 2026-08-12

### Silent errors fixed — do results you already computed need rechecking?

A *silent error* is a confident, plausible, mathematically wrong answer with no
exception, no `NaN` and no verification flag. Eleven were found and fixed this
release. **Eight of them shipped in 3.7 or earlier**, so if you have results
from an affected call, re-run them. The other three were in code added during
this release cycle and never reached a published wheel.

| Affected call | Wrong answer it gave | First shipped in | Recheck? |
|---|---|---|---|
| `decide(Forall(x, φ))` where the counterexample is a rational root whose denominator is **not a power of two** | `True` for a **false** universal theorem, e.g. `∀x. (3x+2)² > 0` (false at `x = −2/3`) | ≤ 3.7 | **Yes** — any `decide` verdict |
| `decide(Exists(x, φ))` with an `=` atom | `(True, witness)` where the witness does **not** satisfy the sentence, e.g. `∃x. 3x−2 = 0 → x = 1/2` | ≤ 3.7 | **Yes** — any cited witness |
| `Matrix.nullspace()` on a 2×2 with a symbolic determinant | A confident wrong kernel basis; `[[x,0],[0,1]]` returned `(0, x)`, for which `M·v ≠ 0` | 3.7 | **Yes** — verify `M·v = 0` numerically |
| `simplify` / `simplify_egraph` on a product containing `0⁻¹` | `1`, or `0`, depending on the engine — for an expression with no value at all. Reachable from `diff(2/(x − x), x)` | ≤ 3.7 | Yes, if any input could reduce to `0⁻¹` |
| `decide` on a two-variable sentence true only at an irrational point | `False` for a satisfiable `∃x∃y`, and `True` for its false `∀x∀y` dual | this cycle (2-var `decide` is new) | No published release affected |
| `batch_map(..., parallel=True)` under `context(budget=…)` | Ran **unbudgeted**, so candidates a sequential sweep reported as `E-BUDGET-001` came back as `E-INT-001` — a *mathematical* verdict | this cycle (batch APIs are new) | No published release affected |
| `product_definite` on a term with any non-integer coefficient | Off by `c^(hi−lo+1)`: `Π_{k=1}^{5} ½` returned `1` instead of `1/32`, `Π (2k−1)/(2k)` at `n = 6` returned `14.4375` instead of `0.2255859375` | ≤ 3.7 | **Yes** — any `product_definite` / `product_indefinite` result |
| `sum_definite` where the summand has a pole strictly *between* the bounds | A clean finite number for a sum with an undefined term: `Σ_{k=1}^{10} 1/((k−3)(k−2))` returned `−5/8` | ≤ 3.7 | **Yes** — any `sum_definite` over a range containing a denominator root |
| `euler_maclaurin` when `corrections` is too small for the summand | A fabricated additive constant — the missing term frozen at the fitting point. `Σ k⁹` at the default `corrections = 2` acquired `34359738368 = 512⁴/2` in a Faulhaber polynomial whose constant term is `0` | this cycle (Euler–Maclaurin is new) | No published release affected |
| `rsolve` on a **forward-shift** spelling with a non-zero right-hand side | The solution of a *different* equation: `f(n+1) − f(n) = n²` with `f(0) = 0` returned `Σ_{j=1}^{n} j²` instead of `Σ_{j=0}^{n−1} j²` | ≤ 3.7 | **Yes** — any `rsolve` written with `f(n+i)`, `i > 0`, and an inhomogeneous term |
| `rsolve` / `solve_linear_recurrence_homogeneous` on an order-2 recurrence with a **repeated** characteristic root | `C₀·rⁿ + C₁·rⁿ` — a one-parameter family presented as the general solution of a second-order equation, losing the `n·rⁿ` branch | ≤ 3.7 | **Yes** — check the discriminant of `r² + b r + c` |

Also fixed, and not a silent error but worse for an unattended loop: a Rust
panic escaped `interval_eval` as `pyo3_runtime.PanicException`, which inherits
from `BaseException` and therefore slips past `except Exception`. Shipped in
3.7 — a loop that survived everything else died on `x^(3/2)` over a negative
ball.

The deterministic silent-error gate (`tests/silent_errors/`, Tier-1 CI) now
scores **0 silent errors out of 213 scored cases** across evaluation,
integration, limits, linear algebra, number theory, real QE, series,
simplification, solving, and sums/products. Every trap added this cycle was
re-run against a build with the fix reverted and confirmed to score
`silent_error` there, and every trap is paired with a **control** — its nearest
convergent neighbour — so a subsystem cannot pass the gate by refusing
everything. That is a statement about the corpus, not a guarantee about the
library.

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
- **`sum_definite` raises `SumError` (`E-SUM-003`) when the summand is undefined
  at an integer inside `[lo, hi]`**, not only when the pole lands on `lo` or
  `hi+1`. The refusal names the offending index. Sums whose poles lie outside
  the range are unaffected: `Σ_{k=4}^{10} 1/((k−3)(k−2))` still returns `7/8`.
- **`euler_maclaurin` may return a shorter expansion, with no additive
  constant.** The constant is now fitted at a point outside the gate's check
  points and re-fitted at a second one; if the two disagree it is not a constant
  and none is claimed. The report says which way that went in `derivation`, and
  the `"fitted numerically"` hypothesis is only listed when a fitted constant is
  actually part of the answer. Genuine constants (`γ`, `ζ(2)`, `½log 2π`, …) are
  unaffected — they agree across fitting points to 13+ digits.
- **`product_definite(term, k, lo, hi)` with `lo > hi` returns `1` even for a
  zero term.** The empty product takes no factors; it previously returned `0`
  for `Π_{k=1}^{0} 0` while returning `1` for `Π_{k=1}^{0} k`.
- **`capabilities()["contract_version"]` is `3`, and `features` lost two keys:
  `groebner_cuda` and `numpy`.** Indexing either now raises `KeyError`; use
  `features.get(name, False)` if you need to span versions. Both were removed
  rather than wired up because neither was *falsifiable* — no observation a
  Python caller could make distinguished `True` from `False`:
  - `groebner_cuda` reported that the CUDA Macaulay-matrix kernel had been
    compiled in. The string `groebner_cuda` occurred exactly once anywhere in
    `alkahest-py` — the capability line itself. There was no binding, no
    `*gpu*` name in the public or the private module, and `GroebnerBasis`
    exposes only CPU methods. The kernel is unchanged and still reachable from
    Rust as `alkahest_cas::poly::groebner::compute_groebner_basis_gpu`; if
    dispatch ever prefers it, the binding lands first and a bit follows it.
  - `numpy` mapped to a Cargo feature gating the `numpy` crate, which
    `alkahest-py` never used an item from. The feature and the dependency are
    both gone. `ak.numpy_eval` and `ak.numpy_eval_par` are unaffected — they go
    through the buffer protocol and always worked with the bit `False`, which
    is its value on every wheel ever published.

  An unreachable `True` makes a caller trust something it should not, which is
  the same class of defect as a silent wrong answer; a bit that correlates with
  nothing is better removed than left to be misread.
  `tests/test_agent_contract.py::test_every_advertised_feature_has_an_entry_point`
  now walks `features` and fails on any key without a named, reachable entry
  point, so the next one cannot ship.
- **Rust, `--features groebner-cuda`: `compute_groebner_basis_gpu` and
  `reduce_batch` return `(polys, GpuBackendReport)` instead of `polys`.** Both
  fall back to CPU row reduction — when `device_id` is `None`, and when the
  driver fails — and the basis is identical either way, so a caller previously
  had no way to tell a GPU run from a CPU one. `GpuBackendReport::ran_on_gpu()`
  is true only when at least one mod-p reduction ran on a device and none fell
  back; `reductions_on_gpu`, `reductions_on_cpu` and `first_gpu_error` carry
  the detail. A compile error on upgrade is the intended failure mode for code
  that was recording these results as GPU results. Nothing at the Python
  surface changes: the feature has no binding.
- **`residue(f, z, point)` refuses a non-constant `point` with
  `AlkahestError` / `E-RESIDUE-005`** instead of leaking
  `AttributeError: 'Expr' object has no attribute 'numerator'` from the
  argument parser. `AttributeError` is not an `AlkahestError`, so
  `except ak.AlkahestError` missed it entirely. The existing `E-RESIDUE-001..4`
  refusals are now `AlkahestError`s carrying `.code` and `.remediation` too,
  rather than bare `ValueError`s with the code glued into the message;
  `AlkahestError` subclasses `ValueError`, so `except ValueError` still works.
- **`series` refuses instead of running forever, with `SeriesError` /
  `E-SERIES-003`.** `series(sqrt(t**-2 + t**-1), t, 0, 32)` never returned:
  coefficients are formed by repeated differentiation without re-simplifying, so
  a nested radical's derivatives grow by a constant factor per coefficient and
  the cost doubles per order. It now honours an active `Budget` (raising
  `BudgetExceededError`) and, with none, an internal work ceiling. It never
  returns a *shorter* series: `O(h^order)` on fewer coefficients than were asked
  for is a false statement about the remainder, which is worse than the refusal.
  Ordinary expansions are unaffected — the heaviest in the suites intern a few
  thousand nodes against a ceiling of 50 000.
- **`simplify_expanded` records a derivation step when its expansion bound stops
  it** (`expand_pow_limit_reached`, a no-op step naming the power it declined),
  and the bound itself is now a budget on the number of distributed products
  rather than a flat exponent cap. `(x+y)**6` and `(x+y+1)**7` now expand where
  the exponent-only cap refused them while permitting a twenty-term sum to the
  fourth power; anything above the budget comes back unexpanded *and says so*
  instead of looking like an expression that was already expanded.

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
- **There is no `cuda_device_count()`.** `CudaCompiledFn.call_batch_on(ordinal,
  …)` selects a device, but the valid range can only be discovered by trying an
  ordinal and catching `CudaError` (`E-CUDA-003`); the loop that does it is in
  [`gpu.md`](docs/mdbook/src/gpu.md#discovering-the-valid-device-ordinals). Not
  added yet on purpose: `cuda` implies LLVM 15 with NVPTX, so such a binding
  cannot be compiled on an ordinary dev box, no CI job builds the Python
  extension with either CUDA feature, and exercising it needs a device — it
  would ship with no verification of any kind, which is the provenance of the
  capability overclaims fixed above. It belongs in the same change as the
  missing `maturin develop --features cuda` + `pytest tests/test_cuda.py`
  nightly step.

### Fixed

- **`cargo test --features groebner-cuda` could not pass on a machine with no
  NVIDIA driver**, contradicting the header comment of
  `alkahest-core/tests/groebner_cuda.rs`. `cudarc` *panics* rather than
  returning `Err` when `libcuda.so` cannot be `dlopen`ed, so `gpu_available()`
  — whose entire job is to decide whether the GPU tier can run — aborted three
  tests instead of skipping them. A missing library and a missing device now
  both mean *not available*, while `ALKAHEST_GPU_TESTS=1` asserting a device
  that is not usable still fails hard. The GPU tier additionally asserts
  `GpuBackendReport::ran_on_gpu()`, so a "GPU test" whose reductions all landed
  on the CPU fails rather than passing on identical results.
- **`product_definite` dropped the scale it used to clear denominators.**
  `ratuni_poly_to_univ` multiplies a `ℚ[k]` polynomial through by the LCM of its
  coefficient denominators and never returned that factor, so every index
  contributed one spurious copy of it and the answer was off by
  `c^(hi−lo+1)`. It is called separately on numerator and denominator, so the
  two cancelled only when they happened to be equal — which is why integer-
  coefficient products were always right and `Π ½` was not. The scale is now
  returned and re-applied (`product_indefinite` gets `c^k`, the same factor in
  antidifference form). A 1936-case sweep over `(a₁k+b₁)/(a₂k+b₂)` against exact
  `Fraction` arithmetic finds 0 mismatches, down from 26 of 160.
- **`sum_definite` could not see a pole strictly inside the summation range.**
  The only check was `contains_zero_to_negative_power` applied to the telescoped
  difference `G(hi+1) − G(lo)`, which never mentions the interior indices, so
  only poles landing exactly on an endpoint were caught. The summand itself is
  now scanned, the same way the definite integrator's interior-pole guards look
  at the integrand rather than at `F(b) − F(a)`: the integer roots of the
  summand's own denominators are read off its ℤ-factorisation (so the cost does
  not grow with the range), each candidate is substituted, and refusal requires
  seeing an actual `0^{negative}` survive simplification — positive evidence,
  never a guess.
- **`euler_maclaurin` fitted its additive constant at the point its own gate
  scored.** The residual there was then zero by construction, so the `o()`-gate's
  decay test was satisfied whatever the number was, and any term the expansion
  was missing came back as a "constant" — `Σ k⁹` acquired `512⁴/2`. The constant
  is now fitted outside the gate's check points and only emitted if a second fit
  reproduces it; across the clean battery a genuine constant drifts by ≤ 3.2e-3
  of itself, a fabricated one by ≥ 0.93.
- **`rsolve` solved a shifted equation for forward-shift spellings.**
  `extract_recurrence` re-indexes the sequence terms into lag form (`f(n+o) ↦
  f(n−(max_o−o))`), which is the original equation with `n ↦ n − max_o`, but left
  the right-hand side at `n`. The right-hand side is now shifted with them, so
  `f(n+1) − f(n) = n²` and `f(n) − f(n−1) = (n−1)²` mean the same thing again.
  The answer is checked by substituting it back into the equation as supplied.
- **Order-2 recurrences with a repeated characteristic root lost a branch.**
  `rsolve` returned `C₀·rⁿ + C₁·rⁿ` and `solve_linear_recurrence_homogeneous`
  divided by `r₁ − r₂ = 0`, producing a closed form containing `0^{-1}` that
  evaluated nowhere. Both now use the basis `{rⁿ, n·rⁿ}`; order ≥ 3 already
  handled multiplicity correctly.
- **A Zeilberger certificate claimed a recurrence for the sum without its
  boundary hypothesis.** The verified statement is the telescoping identity in
  `k`; summing it over `k = k_lo..k_hi` leaves the boundary difference
  `G(n, k_hi+1) − G(n, k_lo)`, and `Σ_i a_i(n)·S(n+i) = 0` holds only when that
  vanishes. Both the core docs and the Python docstring asserted it
  unconditionally. It is false for `F = C(n,k)/(k+1)`, where `G(n,0) = −1` and
  `(n+2)·S(n+1) − (2n+2)·S(n) = 1`. The certificates themselves were and are
  correct; what was missing was the hypothesis. `ZeilbergerCertificate` now
  carries `side_conditions` (the hypothesis, in the same spirit as
  `DerivedResult.verification["side_conditions"]`) and `boundary_term`
  (`G(n,k) = R(n,k)·F(n,k)`), so a caller can discharge or refute it for their
  own range.
- **Claim graphs: a merge could close a dependency cycle, making the graph
  unreadable.** Claim IDs are content-addressed over the *normalised*
  statement, so two textually different statements (`"a"` and `" a"`) share an
  ID. Re-adding one took `ClaimGraph.add`'s merge path, which unions in its
  dependency edges — and those can point at claims recorded later, including
  ones that already depend on it. The resulting graph served fine in memory and
  serialised fine, then could never be read back: `from_json` topologically
  sorts and raised `CycleError`. `add` now refuses an edge that would close a
  cycle, naming both claims, so "a `ClaimGraph` is acyclic" is a real invariant
  and a JSON round-trip is total. Legitimate acyclic merging is unaffected.
  Found by the `test_json_round_trip_is_lossless` property test.

- **Sparse interpolation: Zippel oracle cost is now polynomial, not
  multiplicative.** `sparse_interp` was formulated recursively — interpolate the
  coefficients of `x₁` as polynomials in the remaining variables, recursively —
  which makes each level's oracle a batched Vandermonde lift calling the level
  below `t` times, so black-box evaluations grew as the **product** `∏ tᵢ` down
  the recursion instead of the sum. Measured on the V2-3 roadmap corpus:
  70 calls at 2 variables, 1,771 at 3, 139,552 at 4, 15,019,900 at 5 (75 s) —
  a factor of 25 → 79 → 108 per added variable, extrapolating to ~1e17 at ten,
  i.e. it never returned. Replaced with Zippel's actual iterative algorithm,
  which introduces one variable at a time and recovers the coefficients of the
  *known* skeleton from a transposed Vandermonde system: `O(n·d·T)` calls.
  The same corpus now takes **34 / 62 / 97 / 139 calls at 2–5 variables and
  601 for the 10-variable, 15-term roadmap case**, which completes in
  milliseconds. That acceptance criterion (≥ 95% success over 20 seeds) passes
  for the first time and its test is un-skipped; a new Rust test asserts the
  oracle *call count* grows linearly in the variable count, since a functional
  test alone cannot tell a correct implementation from one that never finishes.
  `sparse_interp` additionally verifies each candidate against the black box at
  random points and re-draws its anchors on mismatch, so an unlucky anchor
  (Zippel's skeleton hypothesis is probabilistic) now produces a refusal rather
  than a confidently wrong polynomial.
- **`solve` states the hypotheses a parametric answer rests on.**
  `solve([a*x - b], [x])` returns `b/a`, which is the solution *for `a ≠ 0`*: at
  `a = 0` the equation reads `-b = 0`, so there is no solution when `b ≠ 0` and
  every `x` when `b = 0`, and `b/a` is not even a number there. The
  generic-parameter reading is deliberate, but a parametric tuple is not a number
  and is therefore returned **unverified** — nothing substitutes it back — so the
  hypothesis was the only auditable signal and it was not being given. New
  `alkahest.solve_side_conditions() -> list[str]` reports the non-vanishing
  hypotheses the most recent `solve` assumed, in the shape
  `DerivedResult.verification["side_conditions"]` and
  `ZeilbergerCertificate.side_conditions` already use. An empty list means the
  solver *proved* every divisor non-zero: `solve([2*x - b], [x])` reports none.
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
