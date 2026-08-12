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
(`alkahest_cas::budget`) for the scope of the `with` block, and pops it on exit —
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

The Rust engines call a cheap cooperative checkpoint (`alkahest_cas::budget::check`) at
a handful of strategic points — not blanket-inserted into every loop:

- **`alkahest.integrate`** — at the top-level entry (covers every route: algebraic,
  Risch/transcendental, rational-function, log-derivative); at the `integrate_inner`
  recursion boundary; at every `integrate_raw` entry, which the sum rule and the
  constant-multiple rule recurse through, so a long sum is bounded *between*
  summands; once per candidate of the derivative-divides u-substitution search
  (each surviving candidate runs a full recursive `integrate`, and there are up to
  twelve); at the stage boundaries of the rational-function route (normalisation,
  Hermite reduction, Rothstein–Trager, the partial-fraction pass and each of its
  irreducible factors); and inside the two Euclidean loops that dominate a hard
  rational integrand — the ℚ[x] GCD used to reduce `A/D` to lowest terms and the
  number-field GCD of Lazard–Rioboo–Trager. A trip raises `BudgetExceededError` —
  integration has a `Result` return type with a natural place to signal it.

  Those last few are not decoration. See
  [how tightly `wall_ms` binds](#how-tightly-wall_ms-binds) — before they existed a
  300 ms budget on `∫ cos x·sin¹²x/(sin⁹x + sin x + 1) dx` returned after 3.4 s, and
  the same integrand at degree 40 never returned at all.
- **`alkahest.limit`** — at every `limit_inner` recursion boundary, in the Gruntz
  comparability sweep, in the pole-clearing loop of the `x ↦ 1/t` substitution, and
  between Taylor coefficients of the local expansion (the loop that can grow without
  bound on nested radicals). `LimitError` is an exhaustive public enum and cannot grow
  a `Budget` variant without a major semver break, so a trip is reported internally as
  `LimitError::DepthExceeded` and the `E-BUDGET-*` cause is recovered out-of-band
  (`alkahest_cas::calculus::limits::last_budget_trip`); the Python binding raises
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

## How tightly `wall_ms` binds

`wall_ms` is **cooperative**: the call stops at the first checkpoint *after* the
deadline, so it always overshoots by however long the engine had left in the stretch
it was in. That makes the useful question "how long is the longest stretch", not "is
it exact" — and a budget whose overshoot grows without bound is not a budget at all.

Measured on `∫ cos x·sinⁿx/(sin^d x + sin x + 1) dx`, the family that goes through
the Weierstrass half-angle substitution and then Rothstein–Trager (elapsed until the
trip, `wall_ms=300`):

| integrand | before | now |
|---|---|---|
| `n=12, d=9` | 3384 ms | 344 ms |
| `n=16, d=9` | 2107 ms | 360 ms |
| `n=20, d=9` | 3967 ms | 345 ms |
| `n=24, d=9` | 3148 ms | 313 ms |
| `n=40, d=17` | **never returned** (killed at 90 s) | 305 ms |
| `1/(sin⁹x + sin x + 1)` | 110 s | 333 ms |

and across budget sizes on the worst of them (`n=40, d=17`): 53 ms for `wall_ms=50`,
106 ms for 100, 305 ms for 300, 1071 ms for 1000, 3158 ms for 3000 — the overshoot
is a small additive term, not a multiple of the budget and not a function of the
problem size.

**What is left, honestly.** The residual granularity is *one primitive polynomial
operation*, and past a certain degree that operation is a **FLINT** call —
factorisation over ℤ, or a bivariate resultant. Those are single foreign-function
calls: nothing short of an OS-level kill stops one part-way, and adding checkpoints
around them cannot help. On a degree-62 integrand (`1/(sin³¹x + sin x + 1)`) one
such call measured about 2 s, so a 300 ms budget there returns after roughly that
long. That is the honest floor, and only an **outer process timeout** goes below it —
not `run_with_wall_fallback`, which joins the same call rather than preempting it
([below](#it-does-not-bound-wall-time-for-an-uncooperative-callee)).

(The pure-Rust loops that used to dominate — the ℚ[x] and number-field Euclidean
GCDs — *are* now checkpointed, which is what removed the growth. A per-step check
was also tried on the ℚ long division underneath them and measured no further
improvement, so it was dropped rather than kept for the look of it: it would only
have made `max_steps` count faster for nothing.)

So the guarantee worth relying on is: *the budget is checked between operations, and
one operation on a high-degree integrand can take seconds*. It is not a hard
real-time deadline, and no cooperative mechanism can make it one.

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

### The watchdog runs *while* the call runs

For that example to mean anything, the watchdog thread has to be able to execute
during the call it is trying to cancel. `alkahest.integrate` and `alkahest.limit` —
the two budget-honouring engines — therefore **release the GIL** around their core
call (`py.allow_threads`, the same idiom `simplify_par` uses for its Rayon workers).
Without that the flag was only ever observed if it had been set *before* the call:
the watchdog could not run a single bytecode until the operation it wanted to stop
had already finished, which is the opposite of what a fan-out search loop needs.

Two things that follow, and one that does not:

- **Cancellation is cooperative, not preemptive.** The flag is observed at the
  checkpoints listed above, so the call stops at the next one — not instantly. An
  engine stretch with no checkpoint runs to its end.
- **Other calls still hold the GIL.** Only `integrate` and `limit` release it (plus
  the parallel simplifiers and the compiled-function batch paths, for unrelated
  reasons). `request_cancel()` cannot reach a running Gröbner basis or homotopy
  continuation, because those do not check the budget at all yet.
- **Nothing about pool safety changes.** `ExprPool` is `Send + Sync` and interns
  through a lock-free index; releasing the GIL around a call that holds only a shared
  `&ExprPool` is strictly weaker than the concurrent Rayon access `simplify_par`
  already performs on the same structure.

## Determinism seed

`Budget(seed=...)` doesn't do anything by itself — it makes the seed available via
`alkahest.budget_seed()` (Rust: `alkahest_cas::budget::seed()`) to any RNG-consuming
sampler that chooses to consult it, instead of threading an explicit seed parameter
through every call in a pipeline. Two runs entering `Budget(seed=7)` observe the same
`budget_seed()` at every call site that reads it, so a search loop that seeds its own
sampling from the ambient budget is reproducible run-to-run.

## Budgets and threads

The budget frame is **thread-local**; the cancellation flag is **process-wide**. Every
surprise in this area follows from that pair, so it is worth stating plainly:

- A worker thread does **not** inherit the budget its parent entered. Handing work to a
  `concurrent.futures.ThreadPoolExecutor` yourself runs it unbudgeted unless you
  re-enter the budget inside the worker.
- `alkahest.batch_map` / `batch_map_iter` / the `*_many` helpers do that for you under
  `parallel=True`: the active budget is snapshotted on the calling thread and re-entered
  in each worker task, so a trip is reported as `E-BUDGET-00x` on the item that tripped,
  exactly as it would be sequentially. `wall_ms` stays a single sweep-wide deadline;
  `max_steps` becomes per-item (the Rust step counter is not readable from Python).
  See [Batch and streaming evaluation](./batch.md#combining-with-budgets).
- `alkahest.run_with_wall_fallback` likewise enters its `budget` argument *on the worker
  thread* it spawns, so cooperative call sites there actually observe it.
- `request_cancel()` needs no propagation, and that cuts both ways: it stops every
  in-flight cooperative call in the process, not just the one you had in mind. A single
  candidate's budget trip therefore never sets it — nothing in `batch_map` touches the
  flag.

## The Python-layer wall-clock fallback

Because `simplify` cannot raise through its own return type, `context(budget=...)`
*alone* only bounds it the same way `max_iterations` already does — silently, by
returning early. `alkahest.run_with_wall_fallback` turns that silent truncation into a
raised, coded error: it runs the call on a worker thread (with `budget` entered on that
thread) and raises `BudgetExceededError` (`E-BUDGET-001`) when the call overruns
`wall_ms`.

```python
result = ak.run_with_wall_fallback(ak.simplify, big_expr, budget=ak.Budget(wall_ms=200))
```

### It does not bound wall time for an uncooperative callee

Read this before putting it in a loop. `run_with_wall_fallback` **joins its worker
before the exception propagates**, so it returns control when the callee returns — not
at `wall_ms`. Measured: `run_with_wall_fallback(time.sleep, 3.0, budget=Budget(wall_ms=50))`
raises `E-BUDGET-001` after **3000 ms**. The error message reports the real elapsed time
("returned control after 3000 ms") precisely so this shows up in a log instead of being
inferred later.

For a callee that *does* reach a cooperative checkpoint the wait is short, because the
worker now runs inside the budget and stops on it (`integrate` on a hard integrand:
`wall_ms=300` returns in about 320 ms) — but that is the case where
`context(budget=...)` alone would already have bounded it. The uncooperative case, the
one this function looks like it exists for, is the one it cannot bound.

Why not return at the deadline and let the worker run on? Python cannot kill a thread,
so "return early" means leaking a live thread that still takes the GIL in bursts, still
allocates into the pool, and can only be asked to stop through the **process-wide**
cancel flag — which aborts every unrelated in-flight call, and which nobody can then
clear safely (clearing it before the orphan observes it is a no-op; leaving it set makes
every later cooperative call fail with `E-BUDGET-003`). In a multi-day loop that trades a
bounded stall for unbounded orphan accumulation plus collateral cancellation. Joining is
the lesser evil, so it is what the function does.

**What actually bounds wall time**, in order of preference:

1. `context(budget=...)` around an engine that checks the cooperative budget —
   `integrate` and `limit` today. This is the real mechanism; `run_with_wall_fallback` is
   a reporting shim over it.
2. An **OS-level bound** for anything else: run the work in a subprocess with a timeout,
   or put a process-level watchdog around the loop. Nothing inside one Python process can
   preempt a thread mid-FLINT-call — see
   [what is left, honestly](#how-tightly-wall_ms-binds).

So reach for `run_with_wall_fallback` to get a *raise* out of a cooperatively-budgeted
call that would otherwise hand back a silently truncated answer. Do not reach for it to
contain an unknown callee.

## `ExprPool` never reclaims

`Budget` bounds *time* and *steps*. **Nothing bounds memory**, and the shape of the
memory growth is the single most likely way a multi-day loop dies. This section is as
important as everything above it.

### The mechanism

`ExprPool` is an **append-only** hash-consed arena. It has no `clear`, no `truncate`, no
refcount and no garbage collector; the underlying storage cannot shrink. **The only way
to reclaim interned nodes is to drop the entire pool.** And every `Expr`, `Matrix`,
`Series` and `DerivedResult` holds a *strong* reference to the pool it came from, so
keeping one interesting result alive keeps every node ever interned alive with it — which
is exactly the usage pattern a research loop has.

Measured on this machine, 20 000 `integrate` calls with a distinct integrand each time:

```text
one shared pool for the whole loop ......  1 992 bytes/call, forever, linear
a fresh pool per iteration ..............      0 bytes/call
```

Two properties make this nastier than an ordinary leak:

- **Time stays flat.** Per-call latency does not degrade as the pool grows, so there is
  no early warning — the loop runs at full speed until the OOM killer arrives. Growth is
  O(n) in memory with O(1) time.
- **You cannot measure it from Python.** `ExprPool` exposes no `__len__` and no `stats()`,
  so a loop cannot watch its own footprint and decide to recycle.

Per-call cost depends on the operation. As a rough guide, roughly 200 bytes of resident
memory per interned node, and on the order of 0.8 KB/call for `diff` or `simplify`,
2–3.5 KB for `integrate`, ~8 KB for a `crosscheck.check`, ~12.5 KB for a `series` of
order 6. At one `integrate` per second on one pool that is gigabytes within a day.

### The supported pattern: one pool per problem

```python
import alkahest as ak

for problem in problems:
    pool = ak.ExprPool()             # fresh pool per iteration
    x = pool.symbol("x")
    with ak.context(pool=pool, budget=ak.Budget(wall_ms=500)):
        result = ak.integrate(build(pool, problem), x)
        record(str(result.value))    # keep a *string* / dict, not the Expr
    del pool, x, result              # dropping the pool reclaims everything
```

The critical line is `record(str(result.value))`. Holding the `Expr` (or the
`DerivedResult`, or a `Matrix` derived from it) pins the pool and defeats the whole
scheme. `DerivedResult.to_dict()` / `.to_json()` exist partly for this: they give you a
plain-Python envelope that outlives the pool. Do not carry live `Expr` handles between
iterations of an unattended loop; re-parse or rebuild them in the new pool if you need
them again.

### One operation grows even on identical input

`Matrix.eigenvals()` mints a fresh gensym (`__eigen_lambda_N`) into the pool on **every**
call, so re-asking the same eigenvalue question keeps allocating for no new information —
measured at about 1.9 KB/call on the same 2×2 integer matrix over 20 000 calls, where
`simplify` on identical input is exactly 0. Cache eigenvalue results yourself rather than
recomputing them in a loop. (Every other Python-facing entry point measured is flat on
repeated input.)

### If you enable the LLVM JIT

The `jit` (LLVM) feature leaks a whole LLVM `Context` per compile — a true leak with no
pool to drop, on the error paths as well as the success path. Cranelift (the default
wheel's JIT) is unaffected. Do not compile in a loop under a `+jit` / `+full` build.

## Error codes

| Code | Cause |
|---|---|
| `E-BUDGET-001` | The active budget's wall-clock limit elapsed |
| `E-BUDGET-002` | The active budget's step counter exceeded `max_steps` |
| `E-BUDGET-003` | `request_cancel()` was called and not yet cleared |

All three are `Cause::Resource` in the Rust registry (`alkahest_cas::errors::codes`) —
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
| `run_with_wall_fallback(fn, *args, budget, **kwargs)` | function | Raises `E-BUDGET-001` when `fn` overruns `wall_ms` — after joining its worker, so it does **not** bound wall time for an uncooperative callee ([above](#it-does-not-bound-wall-time-for-an-uncooperative-callee)) |
| `BudgetExceededError` | exception | `E-BUDGET-001..003`, subclass of `AlkahestError` |

On the Rust side (`alkahest_cas::budget`): `Budget`, `enter`, `BudgetGuard`, `check`,
`seed`, `is_active`, `request_cancel`, `clear_cancel`, `is_cancelled`, `BudgetError`.

## See also

- [Autoresearch / agent loops](./search-plumbing.md)
- [Batch and streaming evaluation](./batch.md) — budgets compose with `*_many` /
  `batch_map`, including under `parallel=True`; a trip becomes one failed `BatchItem`
  carrying `E-BUDGET-00x`, not a killed process and not a mathematical verdict
- [Error handling](./errors.md) — `E-BUDGET-*` in the exception hierarchy
- [Claim graphs](./claim-graphs.md) — session-level provenance around budgeted calls
