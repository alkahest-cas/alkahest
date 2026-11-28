# Lean certificates

Alkahest can emit Lean 4 source for derivations. Generated source becomes a
machine-checked proof only after it typechecks with the pinned Lean/Mathlib
toolchain and without admitted placeholders.

## How a certificate is checked: reflection, with a tactic fallback

Certificates come in two kinds, and `certificate_status["path"]` says which one
a result has.

* **Proof by reflection (`"kernel"`).** The certificate is *data*: the claim as
  an expression AST plus a witness (an atom table, the relations used, the
  cofactors). A checker written in Lean, [`Alkahest.Kernel`](./lean-kernel.md),
  decides it with `decide +kernel`, and a soundness theorem proved once in the
  library (`integral_sound`, `ode_sound`, `pde_sound`) turns `check = true`
  into the statement. The checker computes derivatives itself, so nothing
  Alkahest computed is trusted: a wrong witness makes the check fail. This is
  the default for indefinite integrals, for `dsolve` solutions
  (`experimental.dsolve_certificate`) and for the PDE residuals of
  `fex.finalize`.
* **Tactic proofs (`"tactic"`).** A Mathlib proof script built node by node
  (`HasDerivAt` lemmas, `linear_combination`, `simp`, interval FTC, …). Integrals
  whose answer the kernel does not encode (special functions such as `erf` and
  `Si`, `RootSum`, roots other than `√`) fall back to it, and derivatives,
  definite integrals, limits, sums, resultants and rewrite logs use it
  throughout.

On the published 3.15.0 wheel, 218 of the 230 indefinite-integral certificates
emitted over three corpora (272 integrands) take the kernel path. The table
and the method are in
[How often each path is taken](./lean-kernel.md#how-often-each-path-is-taken).

```python
import alkahest as ak

x = ak.symbol("x")
r = ak.integrate(x**2, x)
r.certificate_status["path"]                 # "kernel"
"decide +kernel" in r.certificate            # True

r = ak.integrate(ak.exp(-x**2), x)           # sqrt(pi)/2 * erf(x)
r.certificate_status["path"]                 # "tactic"
r.certificate_status["kernel_detail"]        # "withheld_special_function"
```

`alkahest.set_lean_certificate_mode("tactic")` emits tactic proofs only, and
`"kernel_native"` closes kernel certificates with `native_decide`, which is
faster on very large certificates but adds Lean's compiler to the trusted base.
That mode is opt-in, and CI rejects it in the default corpora. See
[The trusted base](./lean-kernel.md#the-trusted-base).

### Checking a certificate

Use the pinned toolchain in the repository's `lean/` directory. Kernel
certificates also need the `Alkahest` library built once:

```bash
cd lean
lake exe cache get        # Mathlib's prebuilt oleans
lake build Alkahest       # only needed for kernel certificates
lake env lean -DwarningAsError=true -DautoImplicit=false path/to/cert.lean
```

Then read the theorem statement: it is what was proved. A certificate with
`sorry`, `admit` or `axiom` is not a proof, and Alkahest never emits one.

## Three levels of evidence

**Derivation logs** — always on, always cheap. Records every rewrite rule applied, with rule name and arguments. Human-readable; machine-parseable; forms the basis for Lean export.

**Lean certificate export** — for the operations listed under
[Coverage](#coverage). The library emits a `.lean` file containing a theorem
and its proof, checked either by the reflective kernel or by Mathlib tactics
(above). Emission makes a certificate *available*; it does not
by itself mean Lean has checked the source.

In the agent contract, `certificate_available` has this same meaning. Only a
corpus artifact compiled by pinned Lean/Mathlib without admissions can be
described as `lean_checked` — and because that compilation happens out of
process, in CI, no `DerivedResult` ever reports `lean_checked` as its status.
It is therefore *not* listed in
`capabilities()["verification"]["statuses"]`, which enumerates only the
statuses the library can actually emit.

**Algorithmic certificates** — planned evidence for operations where rewrite
sequences do not work. Do not treat a derivation log as an independently
verified witness.

## Theorem mapping

Every primitive in the registry is tagged with a Lean 4 / Mathlib theorem name:

| Primitive rule | Mathlib theorem |
|---|---|
| `diff_sin` | `Real.hasDerivAt_sin` |
| `diff_exp` | `Real.hasDerivAt_exp` |
| `diff_log` | `Real.hasDerivAt_log` |
| `diff_chain` | `HasDerivAt.comp` |
| `diff_add` | `HasDerivAt.add` |
| `diff_mul` | `HasDerivAt.mul` |
| `add_zero` | `add_zero` |
| `mul_one` | `mul_one` |

The full mapping lives in `alkahest-core/src/lean/`.

## Exporting a certificate

```python
from alkahest import ExprPool, diff, sin

pool = ExprPool()
x = pool.symbol("x", "real")

dr = diff(sin(x**2), x)

# The certificate is in dr.certificate when Lean export is enabled
if dr.certificate:
    with open("proof.lean", "w") as f:
        f.write(dr.certificate)
```

The emitted `.lean` file imports Mathlib and contains a proof term that Lean can verify:

```lean
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

-- Alkahest certificate: d/dx sin(x²) = 2*x*cos(x²)
theorem alkahest_diff_sin_sq (x : ℝ) :
    HasDerivAt (fun x => Real.sin (x ^ 2)) (2 * x * Real.cos (x ^ 2)) x := by
  have h1 : HasDerivAt (fun x => x ^ 2) (2 * x) x := ...
  exact (Real.hasDerivAt_sin _).comp x h1
```

## Strict Lean CI

The CI pipeline (`.github/workflows/lean.yml`) builds `Alkahest.Kernel`,
rejects admissions in it, runs `lean/test/KernelSmoke.lean`, and compiles
these corpora:

| corpus | what it emits |
|---|---|
| `tests/lean_corpus.py` | the strict corpus: rewrites, `d/dx x³`, chain rules, FTC integrals, sums |
| `tests/lean_tendsto_corpus.py` | `Filter.Tendsto` limits |
| `tests/lean_gosper_corpus.py` | Gosper `Finset.sum` telescopes and `∏ k = n!` |
| `tests/lean_integrate_corpus.py` | indefinite integrals, twice: the default mode and `--mode tactic` |
| `tests/lean_ode_corpus.py` | `dsolve` solutions through the kernel |
| `tests/lean_fex_corpus.py` | `fex.finalize` PDE residuals through the kernel |
| `tests/lean_corpus_sample.py`, `tests/lean_sos_corpus.py` | the textbook-gate pool and SOS certificates |

Every corpus entry must contain no `sorry`, `admit`, `axiom` or
`native_decide`, and must typecheck with warnings treated as errors. The
Lean compiler has no `--no-sorries` command-line flag, so the source admission
check is explicit in both the generator and CI.

### Toolchain and porting notes

The checker is pinned in `lean/`: `lean-toolchain` (Lean 4 **v4.36.0**), the
Mathlib tag in `lakefile.lean` (**v4.36.0**) and the exact dependency revisions
in the committed `lake-manifest.json`. CI downloads Mathlib's prebuilt oleans
with `lake exe cache get`, so Mathlib is never built from source. To check a
certificate locally:

```bash
cd lean && lake exe cache get
lake env lean -DwarningAsError=true -DautoImplicit=false path/to/cert.lean
```

The pin moved from v4.9.0 (mid-2024) to v4.36.0. What changed in the emitted
Lean:

- **Linters.** Every certificate switches off `linter.unusedSimpArgs`,
  `linter.unusedTactic` and `linter.unreachableTactic`. These are style
  linters; under `-DwarningAsError=true` they would reject a proof whose
  `simp` set names a lemma it did not need. `sorry` is still a warning and
  still fails the check.
- **Lemma renames.** `mul_inv_cancel` → `mul_inv_cancel₀` (fields),
  `Nat.Ico_succ_right` → `Finset.Ico_add_one_right_eq_Icc`,
  `Set.not_mem_uIcc_of_lt` → `Set.notMem_uIcc_of_lt`,
  `tendsto_one_plus_div_rpow_exp` → `Real.tendsto_one_add_div_rpow_exp`,
  `differentiableAt_id'` → `differentiableAt_id`, `differentiableAt_inv.mpr h`
  → `differentiableAt_inv h`, `deriv_pow''` → `deriv_fun_pow`, and
  `if_pos`/`if_neg` → `ite_eq_left`/`ite_eq_right`.
- **Pointwise vs. lambda forms.** `HasDerivAt.add`/`.mul`/`.neg`/`.pow`/…
  and `deriv_add`/`deriv_mul`/… now state the derivative of `f + g` (a
  function-level sum), while their `fun_`-prefixed twins (`HasDerivAt.fun_add`,
  `deriv_fun_mul`, …) state it for `fun x => f x + g x`. Certificates that
  `rw` or `simpa` against a lambda use the `fun_` forms; `simp` sets carry both.
- **Moved modules.** The `sinh`/`cosh` derivative lemmas live in
  `Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp`; the interval-FTC
  lemmas in `Mathlib.MeasureTheory.Integral.IntervalIntegral.{Basic,FundThmCalculus}`;
  `Finset` big operators in `Mathlib.Algebra.BigOperators.Group.Finset.Basic`.
- **New coverage.** Mathlib v4.36 has `HasDerivAt.arsinh` and
  `Real.hasDerivAt_arcosh`, so integration certificates now cover `asinh` and
  `acosh` (on `1 < u`), and `tanh` through `Real.tanh_eq_sinh_div_cosh`. It still
  has no `erf`, `Si`, `Ei` or `li`; answers using them stay withheld with
  `withheld_special_function`.

1. Generates proof files via `tests/lean_corpus.py`
2. Compiles them with the pinned Lean/Mathlib toolchain (with Mathlib cached)
3. Fails the build if a proof contains an admission or does not typecheck

**The corpus is a lower bound on correctness, not a proof of it.** A shape the
corpus never emits is never typechecked, and the emitter is broader than the
corpus: several arms shipped emitting `.lean` files that Lean rejects simply
because nothing generated them. Two lessons are baked into the corpus now.

*Assert the goal, not the theorem's name.* A test that checks only that a
Mathlib lemma is mentioned somewhere in the file passes just as happily when
the goal above it says something else. Corpus entries that depend on a
particular printed form pin that form (`_require_goal`), and the tactic
assertion for an instantiated lemma includes the argument
(`tendsto_pow_mul_exp_neg_atTop_nhds_zero 2`, not the bare name).

*Commutative order is not stable.* Children of `Add`/`Mul` are canonicalised by
raw `ExprId`, so which addend or factor prints first depends on what else the
pool interned — the same call certified in a fresh pool and emitted a
non-compiling file in a warm one. Where a certificate's tactic is sensitive to
that order, the corpus builds **both** orders deliberately (a fresh `ExprPool`
per case, with the flipping term interned first).

## Knowing where the line is, before you compute

The boundary above is not something to discover by running an operation and
finding `.certificate` is `None`. It is tabulated in
[Certificate coverage](./certificate-coverage.md) — a table **generated by
running a corpus and recording what actually emitted**, checked into the repo,
and pinned by a CI drift check. Three ways to use it:

```python
import alkahest as ak

p = ak.ExprPool()
x = p.symbol("x")

# 1. Ask before you commit to a route. Truthy iff a certificate is produced;
#    carries the reason when it is not.
answer = ak.certifiable("diff", ak.log(ak.sin(x)), x)
bool(answer)        # False

# `mode="ledger"` answers from the table alone, running nothing at all —
# for scoring many candidate routes. The default `mode="verify"` never says
# True without the certificate in hand, and hands the computed result back on
# `answer.result` so the check costs nothing extra.

# 2. Read the whole boundary.
[r["shape"] for r in ak.certificate_coverage("diff") if r["verdict"] == "certified"]

# 3. Refuse to degrade silently.
with ak.context(require_certificate=True):
    try:
        ak.diff(ak.log(ak.sin(x)), x)
    except ak.CertificateUnavailableError as e:
        print(e.code)               # E-CERT-001
```

`capabilities()["verification"]["coverage"]` summarises the same ledger, and
each primitive's `lean_theorem` bit is read from it — so the agent contract and
the coverage table cannot disagree.

## Coverage

The strict CI corpus currently covers:

- Basic arithmetic rewrites (`add_zero`, `mul_one`, `mul_zero`, constant
  folding, and `pow_one`)
- The polynomial differentiation fast path for `d/dx x³`
- Chain-rule differentiation of `f(g x)` for `f ∈ {sin, cos, exp}` and inner
  `g` a power `xⁿ` (`n ≥ 2`), a linear form `c·x` / `x+c` / `a+bx` (including
  interned `-x` as `x * -1`), or a pointwise `sin`/`cos`/`exp` of the variable
  (`sin(cos x)`, `exp(-x)`, `cos(2x)`, `exp(-x²)`)
- Most indefinite integrals, as `HasDerivAt (fun x => F) f x` under explicit
  domain hypotheses (`log` arguments `≠ 0`, `sqrt` and rational-power bases
  `> 0`, denominators `≠ 0`, `arcsin`/`arccos` arguments in `(-1, 1)`, `cos u ≠ 0`
  under `tan u`). Polynomial × exp/sin/cos, rational functions with log/atan
  parts, log/exp towers, inverse trig, hyperbolic functions, multiple-angle
  trig and algebraic answers are covered, through the kernel where it encodes
  the answer and by tactics otherwise. `erf`, `erfi` and `Si` answers are
  certified by tactics, defining each function by its integral in the same
  file. `Ei`, `li`, `Shi`, `Ci`, Fresnel and elliptic answers and `RootSum`
  answers are withheld, and `certificate_status["detail"]` names the reason
  (`withheld_special_function`, `withheld_rootsum`,
  `withheld_identity_not_reduced`, …).
- Solutions of `dsolve` (`experimental.dsolve_certificate`) and the PDE
  residuals of `fex.finalize` (Poisson, cubic Schrödinger, the eigenproblem
  and linear transport), through the kernel. See
  [The reflective certificate kernel](./lean-kernel.md).
- Products and sums that mix the everywhere-differentiable fragment with
  pointwise `log(x)` / `sqrt(x)` (`x log x`, `exp x · log x`, `log x + x`,
  `x sqrt x`), with an explicit `(hx : 0 < x)` binder. Composites
  (`log(x²)`, `log(sqrt(x²−1)+x)`) stay withheld. `∫ x log x` stays
  withheld: `d/dx` of its antiderivative has an n-ary inverse cancellation
  (`x² · x⁻¹ · ½`) that the two-factor `field_simp` encoding does not close.
- Definite integrals `∫ x in a..b, f x = F b - F a` of the same base family
  (`sin`, `cos`, `exp`, `xⁿ`), **plus finite sums and numeric-literal constant
  multiples of those terms** (`∫ (sin x + cos x)`, `∫ 3·cos x`, `∫ -exp x`,
  `∫ (x² + sin x + 3·cos x)`, …), certified via Mathlib's interval-FTC lemma
  (`intervalIntegral.integral_eq_sub_of_hasDerivAt`) composed with
  `HasDerivAt.add`/`.const_mul`/`.mul_const` and the matching
  `IntervalIntegrable` combinators. A symbolic (non-literal) coefficient, or
  any addend outside the base family, withholds the whole certificate.
  Pointwise `∫_a^b log x` is a separate arm: it needs `0 < a` and `0 < b`
  (`intervalIntegrable_log` requires `0 ∉ uIcc a b`) and proves
  `F b − F a` for `F = x log x − x`. `∫_0^1 log` and negative endpoints stay
  withheld.
- Gosper finite sums `∑_{k=a}^{b} F(k) = G(b+1) − G(a)` for polynomial,
  rational-telescope, and geometric terms, certified as a `Finset.sum`
  identity via `sum_range_sub` / `sum_Ico_eq_sum_range` — never as the false
  rewrite `F = G`. Infinite Basel sums and `gamma` encodings stay withheld.
- The discrete product `∏_{k=1}^{n} k = n!`, via Mathlib
  `Finset.prod_Ico_id_eq_factorial`.
- Resultants, stated as `Matrix.det` of the Sylvester matrix of `p` and `q`
  equal to the result, for Sylvester matrices up to 7×7. The determinant is
  expanded by a controlled `simp only` lemma set (the full simp set rewrites
  `Fin.succ (Fin.succ 0)` to `2` and then cannot evaluate the vector
  literals); above 6×6 the proof first splits along row 0 into explicit
  minors, `have h_j : Matrix.det (minor j) = v_j`, with the values computed by
  Alkahest. 8×8 and larger refuse with `withheld_resultant_shape`.
- Zeilberger / WZ certificates (`to_lean(zeilberger(F, n, k))`, or
  `ZeilbergerCertificate.to_lean()`): for every `F : ℝ → ℝ → ℝ` satisfying the
  summand's hypergeometric shift relations with denominators cleared
  (`(n - k + 1) ^ 2 * F (n + 1) k = (n + 1) ^ 2 * F n k`, …), and away from the
  zeros of the factors divided by (`hN : n - k + 1 ≠ 0`, …), the telescoping
  identity `Σ_j a_j(n)·F(n+j, k) = R(n,k+1)·F(n,k+1) − R(n,k)·F(n,k)`. The proof
  clears denominators with an explicit `linear_combination` of the shift
  relations (cofactors computed from the factorisations) and divides back out
  with `div_sub_div` / `eq_div_iff`. The corpus covers `C(n,k)²`, the WZ pair
  for `Σ C(n,k) = 2ⁿ`, Dixon, Apéry and Franel. The recurrence for the *sum*
  additionally needs the boundary terms (`ZeilbergerCertificate.boundary`) and
  is not part of the statement.
- Derivatives of a `Piecewise` in `x` whose conditions compare `x` with an
  `x`-free term (`<`, `≤`, `>`, `≥`): one statement per branch, on the open
  region where that branch is selected —
  `example (x : ℝ) (hc1 : (0 : ℝ) < x) : deriv (fun x => if x > 0 then x ^ 2 else x * -1) x = x * 2`
  — and one for the default under the strict negation of every condition,
  proved by `Filter.EventuallyEq.deriv_eq` and the branch's own derivative
  certificate. The single equation `deriv (if …) = if …` is never stated (it
  is false at a breakpoint), and the certificate is withheld when a condition
  has another shape or a branch's derivative is not certifiable.

Other exports are generated source, not CI-qualified Lean proofs. In
particular, nested two-deep composites (`sin(cos(x²))`), `log`/`tan`/`sqrt`
chain, conditional logarithm/power rewrites, limits outside the `Tendsto`
fragment, and unsupported expression forms can require side conditions
or currently use a placeholder tactic. They must remain unverified until their
proof encoding and strict corpus coverage are added.

Planned algorithmic certificates include:

- Polynomial factoring
- Polynomial GCD

## Side conditions in proofs

Integration certificates state their domain as hypotheses of the theorem
(`(h1 : x ≠ 0)`, `(h2 : 0 < x ^ 2 - 1)`, …), discharge the ones that hold
everywhere (`x² + 1 > 0`, `exp u > 0`) instead, and withhold a certificate
whose domain has no sample point, so the theorem cannot be vacuous. For other
operations, side conditions (domain constraints and branch-cut restrictions)
are recorded in the derivation log but not translated into Lean hypotheses, so
conditional rewrites are excluded from the strict corpus.
