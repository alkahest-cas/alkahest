# Budgets, cancellation, and determinism

A fan-out loop trying thousands of candidate rewrites/integrals/Gröbner bases cannot
afford one pathological candidate to hang the whole batch — and an orchestrator that
decides a candidate isn't worth more time needs a way to *stop it now*, not wait for an
OS-level kill (`SIGKILL`, a process timeout). `alkahest.Budget` and
`alkahest.context(budget=...)` give heavy engines a cheap, structured way to bail out
honestly — raising `BudgetExceededError` — instead of running unbounded.

```python
import alkahest as ak

p = ak.ExprPool()
x = p.symbol("x", "real")

with ak.context(pool=p, budget=ak.Budget(wall_ms=50, max_steps=10_000, seed=7)):
    try:
        ak.integrate(hard_expr, x)
    except ak.BudgetExceededError as e:
        assert e.code.startswith("E-BUDGET-")
        # ... deprioritize this candidate and move on to the next one ...
```

## Model

A `Budget` is an immutable `(wall_ms, max_steps, seed)` triple. Every field is optional;
`Budget()` never trips a check on its own — only `alkahest.request_cancel()` can stop a
call entered with a bare `Budget()`.

`context(budget=...)` pushes the budget into a **thread-local** stack on the Rust side
(`alkahest_core::budget`) for the scope of the `with` block, and pops it on exit —
including on an exception, matching every other resource the context manager owns.
Budgets nest like every other `context(...)` key: only the *innermost* frame is
consulted, so a nested `context(budget=...)` **shadows** the outer one rather than
combining limits with it. A nested `context(...)` that omits `budget=` leaves the outer
one active (nothing is pushed, so nothing shadows it):

```python
with ak.context(pool=p, budget=ak.Budget(seed=1, max_steps=1000)):
    ak.budget_seed()   # 1
    with ak.context(pool=p):
        ak.budget_seed()  # 1 — no budget= here, outer frame still active
    with ak.context(pool=p, budget=ak.Budget(max_steps=2)):
        ak.budget_seed()  # None — this frame set no seed; it does not inherit
    ak.budget_seed()   # 1 — back to the outer frame
```

## What checks the budget today

The Rust engines call a cheap cooperative checkpoint (`alkahest_core::budget::check`) at
a handful of strategic points — not blanket-inserted into every loop:

- **`alkahest.integrate`** — at the top-level entry (covers every route: algebraic,
  Risch/transcendental, rational-function, log-derivative) and at the
  `integrate_inner` recursion boundary that u-substitution and the rational-function
  fallback re-enter. A trip here raises `BudgetExceededError` — integration has a
  `Result` return type with a natural place to signal it.
- **`alkahest.limit`** — at every `limit_inner` recursion boundary, in the Gruntz
  comparability sweep, in the pole-clearing loop of the `x ↦ 1/t` substitution, and
  between Taylor coefficients of the local expansion (the loop that can grow without
  bound on nested radicals). `LimitError` is an exhaustive public enum and cannot grow
  a `Budget` variant without a major semver break, so a trip is reported internally as
  `LimitError::DepthExceeded` and the `E-BUDGET-*` cause is recovered out-of-band
  (`alkahest_core::calculus::limits::last_budget_trip`); the Python binding raises
  `BudgetExceededError` exactly as `integrate` does. With **no** budget active the
  same paths are bounded by an internal work ceiling, so an unsolvable limit refuses
  with `LimitError` / `E-LIMIT-004` instead of running unboundedly.
- **`alkahest.simplify`** (and `simplify_with`, `simplify_batch`) — once per full
  bottom-up rewrite pass. `simplify` has **no error channel** (`DerivedExpr` isn't a
  `Result`), so a trip here stops further passes early and returns the best value
  simplified so far — exactly like running out of the existing `max_iterations` cap
  already does, silently. If you need a hard raise on a `simplify` call specifically,
  wrap it in `alkahest.run_with_wall_fallback` (below).

Other heavy primitives (Gröbner bases, homotopy continuation, …) do not yet check the
budget; wiring them is a follow-up, not part of this cut. Calling `check()` is cheap
when no budget is active and cancellation has not been requested (an atomic load, and —
only if a budget is active — an `Instant::now()`), so it is safe to sprinkle at more
call sites over time without a performance concern gating it.

## Cancellation

`alkahest.request_cancel()` sets a single **process-wide** flag — deliberately not
scoped to a thread or a `Budget` frame. It models "the orchestrator wants the current
heavy operation to stop right now", e.g. because a fan-out loop decided a candidate has
used enough wall time, and the operation might be running on a different thread than the
one that decided to give up on it. `alkahest.is_cancelled()` reads it;
`alkahest.clear_cancel()` resets it — call this before starting the next candidate, or
every subsequent call trips `E-BUDGET-003` immediately.

```python
import threading

def watchdog():
    time.sleep(0.05)
    ak.request_cancel()

threading.Thread(target=watchdog, daemon=True).start()
try:
    ak.integrate(hard_expr, x)
except ak.BudgetExceededError as e:
    assert e.code == "E-BUDGET-003"
finally:
    ak.clear_cancel()
```

## Determinism seed

`Budget(seed=...)` doesn't do anything by itself — it makes the seed available via
`alkahest.budget_seed()` (Rust: `alkahest_core::budget::seed()`) to any RNG-consuming
sampler that chooses to consult it, instead of threading an explicit seed parameter
through every call in a pipeline. Two runs entering `Budget(seed=7)` observe the same
`budget_seed()` at every call site that reads it, so a search loop that seeds its own
sampling from the ambient budget is reproducible run-to-run.

## The Python-layer wall-clock fallback

Because `simplify` cannot raise through its own return type, `context(budget=...)`
*alone* only bounds it the same way `max_iterations` already does — silently, by
returning early. If you need a hard deadline specifically on a call like that,
`alkahest.run_with_wall_fallback` is a **supplement**, not a replacement: it runs the
call on a worker thread and raises `BudgetExceededError` (`E-BUDGET-001`) if it doesn't
finish in time.

```python
result = ak.run_with_wall_fallback(ak.simplify, big_expr, budget=ak.Budget(wall_ms=200))
```

Python cannot forcibly kill a thread, so on a timeout the call keeps running in the
background until it either finishes or reaches a Rust cooperative checkpoint —
`run_with_wall_fallback` also calls `request_cancel()` on timeout so any checkpoint the
call reaches asks it to stop. Prefer the Rust cooperative check alone
(`context(budget=...)`) wherever a call already honors it (`integrate` and `limit`
today); reach for this only when you need a hard deadline on a path that doesn't.

## Error codes

| Code | Cause |
|---|---|
| `E-BUDGET-001` | The active budget's wall-clock limit elapsed |
| `E-BUDGET-002` | The active budget's step counter exceeded `max_steps` |
| `E-BUDGET-003` | `request_cancel()` was called and not yet cleared |

All three are `Cause::Resource` in the Rust registry (`alkahest_core::errors::codes`) —
a budget/cancellation trip is an environment/policy limit, not a statement about the
mathematics, so it is never conflated with e.g. `IntegrationError::NonElementary` (a
proof that no elementary antiderivative exists). `alkahest.integrate` and
`alkahest.limit` raise `BudgetExceededError` directly rather than wrapping it in
`IntegrationError` / `LimitError`, so callers can catch it uniformly regardless of which
engine tripped it:

```python
try:
    ak.integrate(hard_expr, x)
except ak.BudgetExceededError as e:
    ...  # deprioritize and move on
except ak.IntegrationError as e:
    ...  # a genuine "no elementary antiderivative" or "not implemented" verdict
```

## API reference

| Name | Kind | Description |
|---|---|---|
| `Budget(wall_ms=None, max_steps=None, seed=None)` | class | Immutable budget triple |
| `context(budget=...)` | context manager | Push/pop the budget for a `with` block |
| `active_budget()` | function | The `Budget` from the innermost active context, or `None` |
| `budget_seed()` | function | The seed of the innermost active budget, or `None` |
| `is_budget_active()` | function | `True` if a budget is active on this thread |
| `request_cancel()` | function | Set the process-wide cancellation flag |
| `clear_cancel()` | function | Clear it |
| `is_cancelled()` | function | Read it |
| `run_with_wall_fallback(fn, *args, budget, **kwargs)` | function | Python-layer wall-clock fallback for calls without a Rust checkpoint |
| `BudgetExceededError` | exception | `E-BUDGET-001..003`, subclass of `AlkahestError` |

On the Rust side (`alkahest_core::budget`): `Budget`, `enter`, `BudgetGuard`, `check`,
`seed`, `is_active`, `request_cancel`, `clear_cancel`, `is_cancelled`, `BudgetError`.

## See also

- [Autoresearch / agent loops](./search-plumbing.md)
- [Batch and streaming evaluation](./batch.md) — budgets compose with `*_many` /
  `batch_map`; a trip becomes one failed `BatchItem`, not a killed process
- [Error handling](./errors.md) — `E-BUDGET-*` in the exception hierarchy
- [Claim graphs](./claim-graphs.md) — session-level provenance around budgeted calls
