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

Honesty rules that matter in a loop:

- A **budget trip is a fine answer**, not a crash — catch `BudgetExceededError`
  (`E-BUDGET-*`) and deprioritize that candidate.
- A **batch never drops a slot** — failures become `BatchItem(ok=False, error=…)`.
- **Compact mode never hides verification status** — `verification["status"]`
  stays readable; Lean source is omitted on purpose.
- **Certificates are withheld rather than lied about** — see
  [certificate coverage](./certificate-coverage.md).

See also the runnable experimental-mathematics demo
[`examples/pslq_research_loop.py`](https://github.com/alkahest-cas/alkahest/blob/main/examples/pslq_research_loop.py).
