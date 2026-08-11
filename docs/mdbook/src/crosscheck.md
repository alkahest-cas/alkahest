# Cross-CAS differential testing

[Lean certificates](./lean-certs.md) cover the fragment where Alkahest can *prove* its
answer. Outside that fragment nothing is checking the answer at all, and the failure that
matters for a search loop is not a crash but a **silent error**: a confident, plausible,
wrong result that the loop then builds a hundred derived claims on top of. An independent
implementation is the cheapest instrument that catches exactly those — the ones
certificates do not cover.

`alkahest.crosscheck` runs one query through Alkahest and through an oracle and reports
whether they agree.

```python
import alkahest as ak
from alkahest.crosscheck import check, oracles

print(oracles())          # {'sympy': '1.14.0'} — or {'sympy': None} if not installed

pool = ak.ExprPool()
x = pool.symbol("x")

with ak.context(pool=pool):
    out = check("integrate", ak.sin(x) * ak.cos(x), x)

print(out.outcome, out.rung_name, out.reason)
```

```text
agree invariant invariant_holds
```

`sin(x)·cos(x)` has three standard antiderivatives that differ by constants; that is why
the check settled on the **invariant** rung and not by comparing forms.

## Four outcomes, and two of them are not "clean"

`CrossCheck.outcome` is four-valued, never a boolean:

| Outcome | Meaning |
| --- | --- |
| `agree` | Both systems answered and a named rung settled it |
| `diverge` | Both answered and the answers are not the same |
| `incomparable` | The question could not be posed identically to both systems |
| `unavailable` | No oracle is installed |

**`incomparable` and `unavailable` are not weaker forms of `agree`.** They say the check
did not happen. Code that treats them as clean has reintroduced the exact failure this
module exists to prevent: *a loop that believes it is cross-checking, and is not, is worse
off than one that knows it isn't.* Two API decisions enforce that:

- `CrossCheck` deliberately defines **no `__bool__`**, so `if check(...):` does not
  compile into a silent "it agreed";
- `CrossCheck.checked` is `True` only for `agree` and `diverge`, so the common mistake has
  to be written out explicitly.

With SymPy absent, `check` returns `outcome="unavailable"` and `reason="no_oracle"`. It
never returns `agree`. Call `oracles()` at session start to find out **before** you plan
around a check that will only ever return `unavailable` — it reports every *known* oracle,
including the absent ones, as `None`.

## The comparison ladder

Structural equality is useless: the two systems normalise differently, and a naive
comparison of antiderivatives, solution sets or factorisations produces nothing but noise.
So comparison is a ladder, it is **per-operation**, and the rung that settled a check is
always recorded on `CrossCheck.rung`.

| Rung | Name | What it does | What it licenses |
| --- | --- | --- | --- |
| 1 | `syntactic` | Compare canonical forms after translation | Proof of agreement |
| 2 | `symbolic` | `simplify(a − b) == 0`, attempted **independently in both systems** — either one proving it counts | Proof of agreement |
| 3 | `rigorous_numeric` | Sample and evaluate with `ArbBall` / `interval_eval` | Rigorous **refutation**; agreement only "not refuted at these points" |
| 4 | `invariant` | The operation's own defining property, checked on both answers | Proof of agreement |

**Rung 4 leads wherever it exists.** It sidesteps equal-up-to-a-constant,
up-to-ordering and up-to-a-unit entirely — the three things that make naive comparison
useless.

Rung 3 is where Alkahest has an unfair advantage over a float-only harness:
[ball arithmetic](./ball-arithmetic.md) distinguishes "differs by 1e-16 of float noise"
from "genuinely differs by 1e-16". Note the asymmetry, which the record carries: a ball is
a *rigorous* enclosure of the Alkahest answer, so a value outside it is a real
disagreement — but agreement at sampled points is only a failure to refute, and rung-3
agreement is therefore reported with `conclusive=False` rather than being promoted or
discarded.

### The invariants, per operation

| Operation | Rungs | Rung-4 invariant |
| --- | --- | --- |
| `integrate` (indefinite) | 4, 1, 2, 3 | `d/dx F − f ≡ 0`, in each system |
| `simplify`, `simplify_expanded` | 4, 1, 2, 3 | `out − in ≡ 0` — a simplifier's whole contract |
| `sum_indefinite` | 4, 1, 2, 3 | `S(k+1) − S(k) − t(k) ≡ 0` — Gosper's defining property |
| `solve` | 4 | Substitute every solution back, **then** compare set sizes |
| `diff` | 1, 2, 3 | — (integrating back is weaker than the derivative it would check) |
| `limit` | 1, 2, 3 | — |
| `series` | 1, 2, 3 | — (the `O()` remainder is stripped first; that is a normalisation, not a rung) |
| `integrate` (definite) | 1, 2, 3 | — |

`solve` is worth spelling out. Solution *sets* compare badly by construction: ordering,
radical form and branch choice all differ. Substituting back checks each system's answers
on their own terms, and only *then* does the set comparison mean something — with both
sides verified, a size difference is a genuinely **missed solution**, not a formatting
artefact.

An operation with no entry in `OPERATIONS` raises `E-XCHECK-003` rather than falling back
to a generic structural comparison. The fallback is deliberately absent: it is precisely
how a harness starts reporting two normal forms as a divergence.

## A divergence names two suspects

```python
out = check("integrate", integrand, x)
if out.outcome == "diverge":
    d = out.divergence
    print(d.statement())
    print(d.point)             # {'x': 1.257...} — the witness
    print(d.alkahest_value, d.oracle_value)
    print(d.support)           # 'unresolved' | 'alkahest_supported' | 'oracle_supported'
```

The record carries the witness point and **both** values, and the wording never implies
Alkahest is right. `Divergence.support` carries whatever the rigorous escalation could
establish, and its default is `unresolved`:

- `alkahest_supported` — the oracle's answer fails the operation invariant while
  Alkahest's satisfies it;
- `oracle_supported` — **Alkahest's answer fails the invariant under rigorous ball
  arithmetic** while the oracle's satisfies it. This is a silent-error finding. The
  residual is built entirely from Alkahest expressions, so there is no oracle float in it
  to blame, and `Divergence.silent_error_candidate` is `True` exactly here;
- `unresolved` — the two disagree and the evidence does not say which is at fault. Most
  findings start here, and that is the honest default.

When [`verified_sign`](./validated-bounds.md) can certify that the failing residual keeps
one sign across the whole sampling box, `Divergence.region` records the box — upgrading a
finding from "wrong at this point" to "wrong on this interval", which is a much shorter
argument to hand a reviewer.

A `silent_error_candidate` should be routed into `tests/silent_errors/corpus.py`. That
routing is the whole point of the feature: it converts a fuzzing signal into a permanent
regression gate.

**An honest refusal is never a divergence.** If Alkahest declines
(`E-INT-004: no elementary antiderivative exists`, an interior-pole definite integral, a
two-sided limit at a pole) the outcome is `incomparable` with `reason="alkahest_refused"`.
The same holds in the other direction: SymPy returning an unevaluated `Integral` is a
refusal, scored `reason="oracle_refused"`, not an answer to compare against.

## One translator, total-or-refuse

A divergence is only informative if both systems were asked the same question, and the
ways to accidentally ask a *different* one are well known: branch cuts, assumption
handling, `∞`, unnormalised forms. So:

- every tag `Expr.node()` can emit appears in an explicit table
  (`Translator._DISPATCH`, total over `NODE_TAGS`), and an unknown tag raises
  **`E-XCHECK-001`**;
- every primitive is either in `FUNCTION_MAP` or in `REFUSED_FUNCTIONS` **with the reason
  spelt out** — `heaviside` (SymPy fixes `Heaviside(0) = 1/2` and Alkahest fixes nothing
  there), the elliptic integrals (modulus-vs-parameter convention), `round` (no documented
  half-way rule to compare against);
- quantifiers refuse: SymPy has no term-level `∀`, and encoding one as something SymPy
  *will* accept produces an object no rung can use;
- an active `Assumptions` context that cannot be mapped faithfully refuses too.

**False divergences are worse than no signal** — they train both the loop and the team to
ignore the alarm, and a best-effort translator manufactures them by construction.

```python
from alkahest.crosscheck import to_sympy

to_sympy(ak.sqrt(x**pool.integer(2)))                       # sqrt(x**2)
to_sympy(ak.sqrt(x**pool.integer(2)), assumptions=positive) # x
to_sympy(pool.func("heaviside", [x]))                       # raises E-XCHECK-001
```

`to_sympy` is the one translator this package ships. The four hand-rolled `_expr_to_sympy`
helpers in `tests/` (`test_eigen_v217.py`, `test_oracle.py`, `test_gruntz_v217.py`,
`test_diophantine_v219.py`) are meant to migrate onto it; that duplication *is* what this
item exists to remove. Building the QA harness and the runtime mode separately would
produce two translators that disagree, which is the worst possible outcome for a tool
whose entire job is detecting disagreement.

### Assumptions

Only sign and non-zero conditions on a **bare symbol** map onto oracle symbol flags:

| Alkahest predicate | SymPy symbol flag |
| --- | --- |
| `x > 0` | `positive=True` |
| `x >= 0` | `nonnegative=True` |
| `x < 0` | `negative=True` |
| `x <= 0` | `nonpositive=True` |
| `x != 0` | `nonzero=True` |

Anything else — a relation between two symbols, a condition on a composite, a disjunction
— has no per-symbol counterpart, so it raises rather than being dropped. Dropping it would
ask the oracle a *weaker* question, and every legitimate refinement would then look like a
divergence.

## Opt-in per call site, not a context flag

There is deliberately **no** `context(crosscheck=True)`. That is the obvious design and
the wrong one: it puts an oracle round-trip on every call, and stage-2 falsification runs
millions of times where SymPy is orders of magnitude slower. Call `check` where you want
it — at [claim-recording](./claim-graphs.md) frequency, hundreds of times, not millions.
That is "falsify fast, certify slow" applied to the QA layer.

## Two tiers in CI

A randomised sweep cannot be a per-PR gate: it is nondeterministic, and a SymPy upgrade
would turn it red for reasons unrelated to the pull request. So the arrangement mirrors
how `tests/silent_errors/` relates to `agent-benchmark/`:

### Tier 1 — the seeded nightly sweep

```python
from alkahest.crosscheck import sweep

report = sweep(cases=200, seed=None)   # seed defaults to budget_seed(), then to DEFAULT_SEED
print(report.summary())                # prints the seed — always
for finding in report.silent_error_candidates:
    print(finding.divergence.statement())
```

The seed is recorded on the report and printed by `summary()`, because a sweep is only
useful as a bug report if the run that found something can be reproduced exactly. Under
`context(budget=Budget(seed=...))` the sweep takes its seed from the
[budget](./budgets.md), so a nightly job and a local reproduction share one knob.
`SweepReport.to_dict()` is JSON-serialisable and suitable for filing as a CI artifact.

**Neither side of a check is bounded, and this module does not pretend otherwise.** The
heavy engines hold the GIL, so a non-terminating call cannot be timed out from Python — a
worker thread cannot be stopped, and abandoning one wedges the interpreter just the same.
At this commit `limit(sqrt(x**2 + x) - x, x, oo)` is one such call. So:

- run the nightly job under an **OS-level timeout**;
- wrap the sweep in `context(budget=…)` for the engines that *are* cooperative
  (`integrate`, and best-effort `simplify` — see [Budgets](./budgets.md)), where a trip
  surfaces as `reason="alkahest_refused"` with an `E-BUDGET-00x` code, which is a fine
  answer;
- `SWEEP_OPERATIONS` deliberately excludes `limit` for this reason.

### Tier 2 — the frozen corpus

`FROZEN_CORPUS` is a tuple of `FrozenCase`s re-run on every pull request:

```python
from alkahest.crosscheck import run_frozen_corpus

for case, outcome in run_frozen_corpus():
    if outcome is None:
        ...   # skipped: does not apply to the installed oracle version
    else:
        assert outcome.outcome == case.expected
```

Every case records the **oracle version range** its expectation was established against
(`oracle_versions=">=1.12,<2"`). Without that the corpus rots silently the first time the
oracle changes an answer — which it will — and a red gate would then be indistinguishable
from a real regression. A case whose range excludes the installed version is *skipped*,
never quietly passed. Cases carry `found_by` (a seed, or a provenance note) and `note`
(what the case protects), and both are asserted non-empty.

**The ratchet**: a divergence the nightly sweep finds must be promoted into a
`FrozenCase` with `found_by` naming the seed, or it only ever gets exercised by a job
nobody reads. Cases are added, never silently deleted — an expectation that changes is a
re-pin with a new `oracle_versions` range, and the old range records what used to be true.

### Visible from the session-start probe

`capabilities()["verification"]` reports the installed oracles and SMT solvers, so an
agent can see them from the probe it already makes at session start rather than
importing `alkahest.crosscheck` to find out:

```python
caps = ak.capabilities()["verification"]
caps["oracles"]      # {"sympy": "1.14.0"}  — absent oracles appear as None
caps["smt_solvers"]  # {"z3": "4.13.0", "cvc5": None}
```

Absent tools are reported **negatively** rather than omitted, so "not installed" stays
distinguishable from "agreed". Both keys probe the environment — detecting an oracle
imports it — so they are cached for the life of the process; see the note on
[`capabilities()`](./python-api.md).

### Not yet wired

- A nightly workflow that runs `sweep`, prints the seed, and files
  `SweepReport.to_dict()` as an artifact. SymPy already ships in the `ci-extras`
  dependency group, so this is a job, not a new dependency.

## Oracles are a plugin interface

`Oracle` is an ABC and `SymPyOracle` is the first implementation. The comparator talks to
oracles *only* through that interface, so a second backend is a class plus one
`register_oracle` call — no change to the ladder, the outcomes, or the corpus machinery.

```python
from alkahest.crosscheck import Oracle, register_oracle

class WolframOracle(Oracle):
    name = "wolfram"
    ...

register_oracle(WolframOracle)
```

Designing the second oracle in later would mean rewriting the comparator. With two
present, two-out-of-three voting turns "someone is wrong" into "Alkahest is probably
wrong", which is a materially more useful signal.

Every method may answer "I don't know" — `is_zero` returns `None`, `run` raises — and the
comparator turns that into `incomparable` rather than guessing.

## Error codes

| Code | Meaning |
| --- | --- |
| `E-XCHECK-001` | A node, a primitive, or an active assumption has no faithful translation. Surfaces as `outcome="incomparable"`, `reason="untranslatable"` |
| `E-XCHECK-002` | No oracle is installed. Surfaces as `outcome="unavailable"` — **never** as agreement |
| `E-XCHECK-003` | The operation has no defined comparison rung — a caller error, raised before any oracle is consulted |
| `E-XCHECK-004` | The oracle itself declined, raised, or returned an unevaluated form — not a divergence |

An unknown *operation* raises out of `check` rather than becoming an outcome: that is a
caller mistake, not a property of the mathematics. Untranslatable input and missing
oracles are outcomes, because a loop has to be able to keep going.

## See also

- [Autoresearch / agent loops](./search-plumbing.md)
- [Ball arithmetic](./ball-arithmetic.md) — the engine behind rung 3
- [Rigorous global bounds](./validated-bounds.md) — `verified_sign`, used to widen a
  pointwise refutation to a region
- [Claim graphs](./claim-graphs.md) — where a check belongs in a research session
- [Certificate coverage](./certificate-coverage.md) — the other half: where Alkahest can
  prove rather than compare
- [Error handling](./errors.md)
