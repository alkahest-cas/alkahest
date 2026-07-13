# Lean certificates

Alkahest can emit Lean 4 source for derivations. Generated source becomes a
machine-checked proof only after it typechecks with the pinned Lean/Mathlib
toolchain and without admitted placeholders.

## Three levels of evidence

**Derivation logs** — always on, always cheap. Records every rewrite rule applied, with rule name and arguments. Human-readable; machine-parseable; forms the basis for Lean export.

**Lean certificate export** — for computations expressible as sequences of
rewrites tagged with Lean theorem names. The library emits a `.lean` file
containing a proof term. Emission makes a certificate *available*; it does not
by itself mean Lean has checked the source.

In the agent contract, `certificate_available` has this same meaning. Only a
corpus artifact compiled by pinned Lean/Mathlib without admissions can be
described as `lean_checked`.

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
from alkahest import diff, sin

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

The CI pipeline (`.github/workflows/lean.yml`) generates a deliberately small,
strict corpus of basic arithmetic rewrites and `d/dx x³`. Every corpus entry
must have the expected non-empty derivation log, must contain no `sorry`,
`admit`, or `axiom`, and must typecheck with warnings treated as errors. The
pinned Lean 4.9 compiler does not provide a `--no-sorries` command-line flag,
so the source admission check is explicit in both the generator and CI.

1. Generates proof files via `tests/lean_corpus.py`
2. Compiles them with the pinned Lean/Mathlib toolchain (with Mathlib cached)
3. Fails the build if a proof contains an admission or does not typecheck

## Coverage

The strict CI corpus currently covers:

- Basic arithmetic rewrites (`add_zero`, `mul_one`, `mul_zero`, constant
  folding, and `pow_one`)
- The polynomial differentiation fast path for `d/dx x³`

Other exports are generated source, not CI-qualified Lean proofs. In
particular, non-polynomial differentiation, conditional logarithm/power
rewrites, integration, limits, and unsupported expression forms can require
side conditions or currently use a placeholder tactic. They must remain
unverified until their proof encoding and strict corpus coverage are added.

Planned algorithmic certificates include:

- Polynomial factoring
- Polynomial GCD

## Side conditions in proofs

Side conditions (domain constraints and branch-cut restrictions) are recorded in
the derivation log. They are not yet translated into Lean hypotheses by the
exporter, so conditional rewrites are excluded from the strict corpus.
