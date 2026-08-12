# Autoresearch / agent loops

Alkahest is useful inside unsupervised or lightly-supervised math search loops
because it is designed to be called **many times under a budget**, with results
that stay auditable. The pieces below are the *search plumbing* that sits next
to the mathematics:

| Need | API | Guide |
|---|---|---|
| Bound one candidate so a hard instance cannot stall the sweep | `Budget`, `context(budget=…)`, `request_cancel` | [Budgets](./budgets.md) |
| Fan out many candidates without one failure aborting the batch | `batch_map`, `integrate_many`, … | [Batch](./batch.md) |
| Cheap, versioned payloads for logs / LLM context | `DerivedResult.to_dict(mode="compact")` | [Derivation logs](./derivations.md#machine-parseable-output-to_dict--to_json) |
| Accumulate claims across iterations | `alkahest.research` claim graph | [Claim graphs](./claim-graphs.md) |
| Ask “will this call certify?” before spending compute | `certifiable`, `require_certificate` | [Certificate coverage](./certificate-coverage.md) |
| Propose a parametric family and fit it | `alkahest.ansatz` (`polynomial`, `rational`, `fit`, …) | [Ansatz families](./ansatz.md) |
| Differential-test a result against another CAS | `alkahest.crosscheck` (`check`, `sweep`) | [Cross-CAS testing](./crosscheck.md) |
| Hand a discrete / mixed int-real subproblem to a solver | `alkahest.smt` (`to_smtlib`, `solve`, `supported`) | [SMT bridge](./smt.md) |

A minimal loop shape:

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
candidates = [x**2, ak.sin(x), ak.log(ak.log(x))]

with ak.research.session(title="Sweep", pool=pool, capture=True) as s:
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=200, max_steps=50_000, seed=7)):
        for item in ak.integrate_many(candidates, x, parallel=True):
            if not item.ok:
                # E-BUDGET-* → deprioritize; E-INT-* → record and move on
                continue
            # Token-cheap record for the next iteration / a human referee
            _ = item.value.to_dict(mode="compact")

print(s.graph.to_markdown())
```

That snippet uses **one pool for the whole sweep**, which is right for a sweep that
ends. It is wrong for a loop that runs for days: see
[running for days without dying](#running-for-days-without-dying) below.

Honesty rules that matter in a loop:

- A **budget trip is a fine answer**, not a crash — catch `BudgetExceededError`
  (`E-BUDGET-*`) and deprioritize that candidate.
- A **refusal is not a negative result.** `E-CAD-001`, `E-SOS-002`, `E-LINALG-010`,
  `E-MAT-004`, `E-SMT-003` and `E-ANSATZ-003` all mean *undecided by this route*.
  Recording any of them as "proved false" or "no such object exists" is the most
  expensive mistake a search loop can make, because it closes a branch permanently.
  The only codes that are genuine mathematical verdicts are the ones documented as
  such — e.g. `E-INT-004` (proven non-elementary). See
  [Refusals](./errors.md#refusals-when-alkahest-declines-to-answer).
- A **batch never drops a slot** — failures become `BatchItem(ok=False, error=…)`.
- **Compact mode never hides verification status** — `verification["status"]`
  stays readable; Lean source is omitted on purpose.
- **Certificates are withheld rather than lied about** — see
  [certificate coverage](./certificate-coverage.md).

## Running for days without dying

Four limits bound an unattended run. None of them is a bug you can wait out; all four
are properties of the design, and a loop has to be written around them.

**1. Memory is not budgeted, and `ExprPool` never reclaims.** A pool created once at
startup grows linearly and forever — roughly 200 bytes per interned node, on the order
of 2–3.5 KB per `integrate` — at *flat* per-call latency, so the run dies by OOM with no
slowdown to warn you. Use **one pool per problem**, drop it when the problem is done, and
carry `to_dict()` envelopes rather than live `Expr` handles between iterations (holding
any `Expr`, `Matrix` or `DerivedResult` pins its entire pool).
[Full treatment](./budgets.md#exprpool-never-reclaims).

**2. `wall_ms` is cooperative, and its granularity is one primitive operation.** The
call stops at the first checkpoint after the deadline. On a high-degree integrand that
operation is a FLINT call, which nothing short of an OS-level kill interrupts — a 300 ms
budget can return after ~2 s there. [Details](./budgets.md#how-tightly-wall_ms-binds).

**3. `run_with_wall_fallback` does not bound wall time for an uncooperative callee.**
It joins its worker before raising, so it returns when the callee returns.
`run_with_wall_fallback(time.sleep, 3.0, budget=Budget(wall_ms=50))` raises after
3000 ms. The only hard bound is an **outer process timeout**.
[Details](./budgets.md#it-does-not-bound-wall-time-for-an-uncooperative-callee).

**4. Some questions get refused, not answered.** `decide` is not complete; it raises
`E-CAD-001` rather than fabricate a verdict it cannot justify, and the linear-algebra
zero test refuses with `E-LINALG-010` / `E-MAT-004` rather than pick a branch. Budget
for refusals in the loop's control flow, not just for failures.

The skeleton that respects all four:

```python
import alkahest as ak

def run_one(problem):
    pool = ak.ExprPool()                  # (1) fresh pool per problem
    x = pool.symbol("x")
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=500, seed=7)):   # (2)
        try:
            result = ak.integrate(build(pool, problem), x)
        except ak.BudgetExceededError:
            return {"status": "undecided", "why": "budget"}
        except ak.AlkahestError as e:
            kind = "verdict" if e.code == "E-INT-004" else "undecided"   # (4)
            return {"status": kind, "code": e.code}
    return {"status": "ok", "result": result.to_dict(mode="compact")}    # no live Expr escapes
```

Run the driver itself under an OS-level timeout (3), not `run_with_wall_fallback`.

See also the runnable experimental-mathematics demo
[`examples/pslq_research_loop.py`](https://github.com/alkahest-cas/alkahest/blob/main/examples/pslq_research_loop.py).
