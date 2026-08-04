# The silent-error gate

A **silent error** is a confident, plausible, mathematically wrong answer
returned with no exception, no `NaN`, and no verification flag that would let
the caller tell it apart from a correct result. `∫_{-1}^{1} x^-2 dx = -2` is
the archetype: the number is clean, the derivation is one power-rule step, and
it is wrong — the integral diverges.

This package measures how often alkahest does that, deterministically, with no
LLM, fast enough to run on every pull request.

## Why it is separate from the textbook gate

`tests/textbook_gate/` asks *"does alkahest get first-course problems right?"*.
This asks the harder question: *"when alkahest cannot get one right, does it
say so?"* Those are different failure modes with wildly different costs.

For an interactive user a wrong answer costs five minutes. For an autonomous
research loop it is a **false lemma that every downstream derivation inherits**,
and one the loop's own consistency checks will happily confirm, because they are
checking against the same poisoned result. A refusal closes one branch; a silent
error poisons a whole subtree. See
`temp-alkahest/planning/autoresearch.md`, "P0 — trust", item 1.

`tests/textbook_gate/test_tg_silent_errors.py` held the first four of these
regressions. This package generalises them into a declarative corpus with a
measured rate and an anti-drift link to `agent-benchmark/`.

## Why it is separate from agent-benchmark

`agent-benchmark/` already measures silent-error rate — through an LLM writing
scripts against alkahest and SymPy. That makes it the right instrument for
"how does this CAS behave in an agent's hands", and the wrong instrument for a
merge gate: it needs API keys, it is non-deterministic, and it costs money per
run.

This gate measures the **library-level** rate: the same traps, called directly,
no model in the loop. It runs in well under a second.

`test_catalogue_sync.py` is the ratchet between the two. Every `Kind.TRAP` task
in `agent-benchmark/tasks/catalogue.py` must be claimed by at least one case
here, so a trap discovered while running the benchmark cannot end up covered
only by the benchmark. The outcome vocabularies are asserted to stay one
relabelling apart (`silent_error` here is `wrong_answer` there), so the two
rates remain comparable.

## The contract vocabulary

Every case declares exactly one contract. The vocabulary is small on purpose:
its job is to make the difference between *refusing* and *lying* explicit at the
point where the case is written.

| Contract | Meaning | A different value is… |
|---|---|---|
| `Raises(code)` | must raise an alkahest error with this exact stable `E-SUBSYSTEM-NNN` code | a **silent error** |
| `Returns(value)` | must produce this value (`tol` applies to floats) | a **silent error** |
| `RefusesOr(value)` | refusing *or* returning `value` is acceptable | a **silent error** |
| `RefusesOr()` | no finite value is acceptable at all | a **silent error** |

`RefusesOr` is for the traps where "this does not exist" and "here is the
principal value / the one-sided limit" are both defensible — divergent
integrals, `(-8)^(1/3)`, `√x = -1`. `RefusesOr()` with no argument is the
stronger form: the quantity does not exist under any convention, so *any*
finite answer is a lie.

Refusing where `Returns` was declared is **not** a silent error — it fails the
gate as a coverage regression, but it is classified `honest_refusal`, because
the caller was still told.

### Outcomes

Mirrors `agent-benchmark/tasks/base.py`'s `Outcome` one-for-one:

| This gate | agent-benchmark | Meaning |
|---|---|---|
| `correct` | `correct` | contract satisfied with a value |
| `silent_error` | `wrong_answer` | **the metric**: a confident wrong answer |
| `honest_refusal` | `honest_refusal` | alkahest declined; always safe |
| `no_answer` | `no_answer` | the call broke in a way that is neither — a corpus bug |

### What counts as a refusal

Matching `refusal_or_value` in `agent-benchmark/tasks/base.py`:

* an `AlkahestError` subclass — the honest, coded path;
* a returned expression that cannot be reduced to a number (alkahest raises
  `ValueError` from `eval_expr` for `∞`, `0^-1`, or a bare `O(x^n)` remainder);
* `NaN` or `±inf`, which are an implicit admission of failure rather than a
  stated answer.

The last two are **weak** refusals: a caller has to look at the value to notice
anything is wrong. Cases that only pass because of one say so in their `note`,
and those notes are where the next round of hardening should start
(`int_pole_tangent_over_period` and the `series_*` singular cases in
particular).

### Verification floors

A case may declare `verification_floor="numerically_checked"` (or stronger) and
return a `Measured(answer, verification)` instead of a bare answer. The runner
then checks `DerivedResult.verification["status"]` against

```
unverified < numerically_checked < certificate_available < exactly_verified < externally_verified
```

This catches the *other* way trust degrades: the answer stays right while the
evidence behind it quietly weakens.

## Adding a case

1. **Work the mathematics out first, independently.** Every case records where
   its expected value came from in `verified_by`, and `test_every_case_declares_a_source`
   enforces that the field is non-trivial. A corpus whose expectations were read
   off alkahest's own output measures self-consistency and nothing else. Hand
   derivations, textbook facts, and independent libraries all qualify; "alkahest
   printed this" does not.
2. **Pick the weakest contract that still catches the lie.** If a principal
   value is defensible, use `RefusesOr(pv)` rather than `Raises`; over-tight
   contracts turn into churn the first time a subsystem legitimately improves.
3. **Add the control.** A gate made only of refusals is passed by a library that
   refuses everything. Each refusal class needs its nearest convergent
   neighbour: `int_pole_inverse_square_symmetric` is paired with
   `int_control_integrable_endpoint_singularity`, the DNE limits with
   `limit_control_squeeze`, the singular-point series with
   `series_control_simple_pole`.
4. **Append to `CASES` in `corpus.py`** with a stable, never-reused `id`.

```python
Case(
    id="int_pole_double_at_one",
    subsystem="integration_definite",
    statement="∫_0^2 (x-1)^-2 dx diverges (double pole at x=1)",
    op=definite(1 / (X - 1) ** 2, _int(0), _int(2)),
    contract=Raises("E-INT-001"),
    verified_by="∫(x-1)^-2 = -1/(x-1); naive FTC gives -1-1 = -2, a plausible wrong number.",
),
```

The `op` is a zero-argument callable returning a plain **answer** — a float,
int, bool, string, list, or a `Measured`. Reducing the library's return value to
an answer is the case's own job, which is what lets a definite integral, a
solution count, and a series coefficient all be scored by the same four
contracts. `corpus.py` provides `definite`, `antiderivative_slope`,
`limit_value`, `series_at`, `simplified_value` and `real_solution_count` for the
common shapes.

### Verifying an antiderivative

Never assert the *shape* of an antiderivative. Use `antiderivative_slope`, which
differentiates whatever alkahest returned and evaluates it at a sample point:
immune to `+C` and to every legitimate difference in form, and it catches the
only thing that matters — an antiderivative whose derivative is not the
integrand. It also detects a **false** non-elementarity verdict, which is
exactly as damaging as a wrong formula: it tells a search loop a branch is
permanently closed when it is not (report7-20.md, bug B2).

### Cost discipline

The corpus runs on every pull request, so every op must finish in well under a
second. Known-slow inputs are documented here rather than added:

* `∫ (1/log x - 1/log²x) dx` (= `x/log x`, elementary, each part is not) does
  not terminate within 30 s. It is a good case and belongs in the corpus once
  `integrate` grows a budget; adding it today would make the gate a timeout
  hazard rather than a signal.

## Known-broken cases: `xfail(strict=True)`, never deletion

A case alkahest currently fails is not removed — it is marked:

```python
xfail="SILENT ERROR: alkahest returns 0, a value the function never takes. …",
```

which the runner turns into `pytest.mark.xfail(strict=True, reason=...)`.
`strict=True` is load-bearing: when the bug is fixed the case flips from `xfail`
to an *unexpected pass*, which pytest reports as a failure. That failure is the
signal — delete the `xfail` field and the case becomes an ordinary regression
test. An absent case can catch neither the fix nor the re-regression.

The `xfail` string must name the bug, not just assert that one exists
(`test_known_broken_cases_name_a_bug` enforces a minimum length). Known-broken
cases are excluded from the headline rate — which therefore tracks
*regressions* — and reported separately under `known_silent_errors`, loudly, as
the fix queue.

## The measured rate

`test_silent_error_count_is_zero` is the gate: zero silent errors outside the
known-broken set, always.

The runner also writes a machine-readable summary to
`target/silent-errors/summary.json` (override with
`ALKAHEST_SILENT_ERROR_REPORT`) and prints it through pytest's
terminal-summary hook, so the numbers appear in the CI log whether the gate
passed or failed and without anyone needing `-s`:

```
cases            : 133 (120 scored, 13 known-broken)
silent-error rate: 0.0% (0 / 120)

subsystem                     cases     ok  refusal  silent  known
  integration_definite           28     22        6       0      0
  ...
```

The JSON carries `by_outcome`, `by_subsystem`, the full `silent_errors` and
`known_silent_errors` lists, and `benchmark_outcome_names` for cross-referencing
an `agent-benchmark` report.

## Running it

```bash
pytest tests/silent_errors/ -v
```

It is also collected by the ordinary `pytest tests/` run. CI gives it a separate
Tier 1a step so a red gate is identifiable at a glance, and passes
`--ignore=tests/silent_errors` to the general pytest step so it is not run
twice.
