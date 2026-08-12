# Batch and streaming evaluation

Search loops are embarrassingly parallel *at the candidate level*: try to integrate a
hundred generated integrands, simplify a thousand rewrite targets, differentiate every
entry in a lookup table. Every Alkahest entry point is one-call-one-answer, so today
that fan-out is written by hand at every call site — and the first candidate that raises
aborts the whole batch unless the caller remembers `try/except` around every single call.

`alkahest.batch_map` (and the `*_many` convenience wrappers over `integrate`, `simplify`,
and `diff`) do that fan-out once. They **never raise** for a single bad element — the
exception is caught and turned into a structured `BatchItem` carrying the failing
exception's stable `E-*` [diagnostic code](./errors.md), so a loop can tell "this
candidate has no elementary antiderivative" (a fine, expected answer) from "the whole
batch process crashed".

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

outs = ak.integrate_many([x**2, ak.log(ak.log(x)), ak.sin(x)], x)
for item in outs:
    if item.ok:
        print(item.index, "=>", item.value.value)
    else:
        print(item.index, "FAILED", item.error["code"], item.error["message"])
```

```text
0 => (x^3 * 1/3)
1 FAILED E-INT-001 [E-INT-001] integrate: not implemented: ...
2 => (-1 * cos(x))
```

## Honesty invariant

`batch_map` always returns exactly one `BatchItem` per input, **in input order** —
a batch of 100 items yields a list of 100 items, full stop. Nothing in this module
silently drops a failing candidate; a failure is recorded as `ok=False` with its error,
never as a missing slot.

## `BatchItem`

| Field | Type | Meaning |
| --- | --- | --- |
| `index` | `int` | Position in the *original* input sequence — stable under `parallel=True` and under streaming in completion order |
| `ok` | `bool` | `True` iff the call returned normally |
| `value` | `Any \| None` | The call's return value (often a `DerivedResult`) on success; `None` on failure |
| `error` | `dict \| None` | `{"code", "message", "remediation", "type"}` on failure; `None` on success |
| `elapsed_ms` | `float \| None` | Wall-clock time spent inside the call for this item |

Exactly one of `value` / `error` is populated: `ok=True` implies `error is None`.

`error["code"]` is the raised exception's `.code` when it is an `AlkahestError`-like
exception — including the native error types, which expose the same attribute —
otherwise `alkahest._batch.UNEXPECTED_ERROR_CODE` (`"E-BATCH-001"`), the fallback for a
failure whose exception carries no diagnostic code of its own (e.g. a plain `ValueError`
raised by caller code passed to `batch_map`).

## `batch_map` and `batch_map_iter`

```python
def batch_map(fn, items, *, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]: ...
def batch_map_iter(fn, items, *, parallel=False, max_workers=None, **kwargs) -> Iterator[BatchItem]: ...
```

Both call `fn(item, **kwargs)` once per item. `parallel=True` fans the calls out over a
`concurrent.futures.ThreadPoolExecutor`; some Alkahest hot paths (`integrate`, `limit`,
the parallel simplifiers, NumPy evaluation) release the GIL for their native work, so a
thread pool can genuinely overlap them. For calls that hold the GIL throughout, `parallel=True`
mainly helps when `fn` itself does I/O or otherwise yields the GIL — it never makes
anything *incorrect*, only sometimes not faster.

### Order guarantees

- **`batch_map`** always returns results **in input order**, whether or not
  `parallel=True`. This is the guarantee to reach for when you need
  `zip(items, batch_map(...))` to line up.
- **`batch_map_iter`** documents two different behaviours by design:
  - `parallel=False` streams **in input order** — item *i* is fully computed and
    yielded before item *i + 1* starts.
  - `parallel=True` streams **in completion order**, not input order. This is the whole
    point of streaming under fan-out: a fast failure surfaces immediately instead of
    waiting behind a slow item that happened to be submitted first. Every yielded
    `BatchItem` still carries its original `index`, so a caller that needs input order
    can sort by it, or just use `batch_map`.

```python
# Streaming: react to failures as they arrive, without waiting for the slowest item.
for item in ak.batch_map_iter(ak.simplify, candidates, parallel=True):
    if not item.ok:
        log.warning("candidate %d failed: %s", item.index, item.error["code"])
```

## `integrate_many` / `simplify_many` / `diff_many`

Thin `batch_map` wrappers over the three most common derivation entry points:

```python
def integrate_many(exprs, var, *bounds, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]: ...
def simplify_many(exprs, *, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]: ...
def diff_many(exprs, var, *, parallel=False, max_workers=None, **kwargs) -> list[BatchItem]: ...
```

`integrate_many` accepts optional trailing bounds (`a, b`) for a batch of definite
integrals, exactly like `alkahest.integrate`. `**kwargs` on every helper is forwarded to
the underlying call (e.g. `assumptions=` for `simplify_many`).

```python
outs = ak.simplify_many(candidates, parallel=True)
ok = [o.value for o in outs if o.ok]
failed = [(o.index, o.error) for o in outs if not o.ok]
```

## Never raises — except for real interpreter signals

`batch_map` and `batch_map_iter` catch `Exception`, not `BaseException`: a
`KeyboardInterrupt` or `SystemExit` still propagates and stops the batch, since
swallowing those would make the process unkillable. Everything else — including every
Alkahest `E-*` error and any exception your own `fn` raises — is captured.

## Combining with budgets

Wrap the batch in `context(budget=…)` so each candidate inherits the same
cooperative wall/step limit (and optional seed). A trip surfaces as
`BatchItem(ok=False, error={"code": "E-BUDGET-00x", …})` rather than aborting
the rest of the batch — see [Budgets](./budgets.md).

```python
with ak.context(pool=pool, budget=ak.Budget(wall_ms=50, max_steps=10_000, seed=7)):
    outs = ak.integrate_many(candidates, x, parallel=True)
```

This works under `parallel=True` as well as `parallel=False`, but the two are not
identical field-for-field, because a Rust budget frame lives on a **thread-local**
stack and a worker thread does not inherit its parent's. `batch_map` therefore
snapshots the active budget on the calling thread and re-enters it inside every
worker task:

| Field | `parallel=False` | `parallel=True` |
|---|---|---|
| `wall_ms` | one deadline for the whole sweep (the caller's frame) | one deadline for the whole sweep, captured at the `batch_map` call |
| `max_steps` | one counter for the whole sweep | **per item** — the Rust step counter lives in the frame and is not readable from Python, so each worker counts from zero |
| `seed` | same value everywhere | same value everywhere |

The `wall_ms` deadline is captured when `batch_map` is called, not when
`context(budget=…)` was entered — Python cannot read the frame's start instant — so
a batch launched partway through a budgeted block gets the full `wall_ms` again.
That is one budget's worth of slack for the whole fan-out, not per item.

### A budget trip is not a mathematical verdict

This is the reason the propagation matters more than the speed-up. Before it, a
fanned-out sweep ran completely unbudgeted, and the candidates a sequential sweep
reported as `E-BUDGET-001` came back as `E-INT-001` instead — the integrator's
verdict that *no elementary antiderivative exists*. A research loop records that as
a permanently closed branch, when in truth nothing was decided and the machine
merely ran out of the time it was given. `E-BUDGET-00x` is `Cause::Resource`; keep
the two apart when you interpret a `BatchItem`:

```python
for item in outs:
    if item.ok:
        accept(item.value)
    elif item.error["code"].startswith("E-BUDGET-"):
        requeue(item.index)          # ran out of budget — undecided, try again with more
    else:
        close(item.index, item.error)  # a real verdict about the mathematics
```

### Cancellation across a batch

`request_cancel()` needs no propagation — the flag is process-wide, so every worker
already sees it and a caller can abort a whole in-flight sweep with it (each item
then reports `E-BUDGET-003`). The converse is deliberate: **one item tripping its
budget never cancels its siblings.** `batch_map` never sets the flag itself; the
trip is recorded on the item that tripped, and the rest of the sweep runs out the
shared deadline.

### One pool for the batch, not for the process

A batch shares one `ExprPool` across all its items, which is right — the whole point is
that the items are related. What is *not* right is reusing that pool for the next batch,
and the next: `ExprPool` never reclaims, so a driver that keeps one module-scope pool and
runs `batch_map` in a loop grows linearly and forever at flat latency. Construct the pool
per batch and drop it, and carry `item.value.to_dict(mode="compact")` forward rather than
the `DerivedResult` itself (holding one pins the whole pool). See
[`ExprPool` never reclaims](./budgets.md#exprpool-never-reclaims).

## See also

- [Autoresearch / agent loops](./search-plumbing.md)
- [Budgets, cancellation, and determinism](./budgets.md)
- [Derivation logs — compact envelopes](./derivations.md#machine-parseable-output-to_dict--to_json)
- [Error handling](./errors.md)
