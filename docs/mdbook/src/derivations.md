# Derivation logs

Most transformations in Alkahest return a `DerivedResult` that records the exact sequence of rewrite steps applied. This log is the foundation for both human inspection and Lean proof export.

## DerivedResult

`DerivedResult` is the return type of `diff`, `simplify`, `integrate`, `limit`, `sum_*`, and most other transforming operations. Notable exception: `series` returns a `Series`.

```python
from alkahest import diff, sin

pool = ExprPool()
x = pool.symbol("x")

dr = diff(sin(x**2), x)
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `.value` | `Expr` | The result expression |
| `.steps` | `list[dict]` | Ordered list of rewrite steps |
| `.verification` | `dict` | Evidence status, artifact format, external-check status, and side conditions |
| `.certificate` | `str \| None` | Generated Lean 4 source, when a derivation log, FTC integral, or recognised `Filter.Tendsto` limit can be certified without `sorry` |

### Methods

| Method | Description |
|---|---|
| `.to_dict(mode="full")` | Versioned dict envelope combining `.value`/`.verification`/`.certificate_status`/`.steps`; see [Machine-parseable output](#machine-parseable-output-to_dict--to_json) below |
| `.to_json(mode="full")` | `json.dumps(self.to_dict(mode=mode))` |

## Rewrite steps

Each step in `.steps` is a dict with:

| Key | Value |
|---|---|
| `rule` | Rule name (string) |
| `before` | Expression before the rewrite |
| `after` | Expression after the rewrite |
| `side_conditions` | Side conditions recorded for the rewrite |

```python
for step in dr.steps:
    print(f"  {step['rule']:25s}  {step['before']}  →  {step['after']}")
```

## Side conditions

A side condition is a predicate that must hold for a rewrite to be sound:

- `Positive(x)` — `x` must be positive (e.g. for `sqrt(x²) → x`)
- `NonZero(x)` — `x` must be non-zero (e.g. for `x/x → 1`). For a *symbolic* `x` the rewrite fires and the condition is recorded; for a **literal** zero base it does not fire at all, since `0 · 0⁻¹` has no value ([literal-zero carve-out](./simplification.md#the-literal-zero-carve-out))
- `Integer(n)` — `n` must be an integer (e.g. for some power rules)
- `BranchCut(f, x)` — records that `f` may have a branch cut at `x`

Side conditions propagate into the derivation log as `SideCondition` entries and are aggregated in `dr.verification["side_conditions"]`. A generated Lean source artifact is evidence that can be checked; it is not a claim that the project has checked the artifact with Lean.

For antiderivatives, `exactly_verified` means that the in-kernel symbolic
residual `d/dx(F) - f` simplified to zero. `numerically_checked` means only
that the integration soundness gate found agreement at several floating-point
samples; it is useful evidence, but it is not an exact proof. `lean_checked`
remains reserved for an actual completed external Lean check.

```python
evidence = dr.verification
if evidence["status"] == "certificate_available":
    assert not evidence["externally_verified"]
    lean_source = dr.certificate
    # Invoke a pinned Lean/Mathlib checker before treating this as lean_checked.
```

## Inspecting a derivation

```python
dr = diff(sin(x**2), x)

print(f"Result: {dr.value}")
print(f"Steps ({len(dr.steps)}):")
for step in dr.steps[:5]:
    rule = step['rule']
    before = step['before']
    after = step['after']
    print(f"  [{rule}]: {before} → {after}")
    for condition in step["side_conditions"]:
        print(f"    side condition: {condition}")
```

## DerivationLog overhead

Logging is always on and is cheap — a `Vec<RewriteStep>` appended to during traversal. The benchmark group `log_overhead` in `alkahest-core/benches/alkahest_bench.rs` measures logging cost separately from computation.

For production workloads where you only need `.value`, the steps list is still populated but you can ignore it. There is no way to disable logging in the current API (disabling it would compromise the Lean certificate pipeline).

## Combining logs

When you chain operations, the logs are separate:

```python
simplified = simplify(expr)
derived = diff(simplified.value, x)

# Full derivation: simplify steps first, then diff steps
all_steps = simplified.steps + derived.steps
```

For operations like `integrate` that internally call `simplify`, the log includes the simplification sub-steps interleaved with the integration steps.

## Machine-parseable output: `to_dict` / `to_json`

Agents pay for every character a call returns. `.steps`, `.verification`, and
`.certificate_status` are convenient to poke at interactively, but stitching
them into one payload for logging, RPC, or a context window means writing
that glue yourself, on every call site, forever. `DerivedResult.to_dict()`
and `DerivedResult.to_json()` give you the stitched, versioned envelope
directly:

```python
dr = diff(sin(x**2), x)

full = dr.to_dict()                     # mode="full" is the default
compact = dr.to_dict(mode="compact")    # short keys, token-efficient
json_str = dr.to_json(mode="compact")   # json.dumps(dr.to_dict(mode="compact"))
```

### Envelope shape (`mode="full"`)

```json
{
  "kind": "alkahest.derived_result",
  "schema_version": 1,
  "steps_schema_version": 1,
  "value": "<display string>",
  "verification": { "status": "...", "evidence": "...", "externally_verified": false, "artifact_format": "...", "side_conditions": [...], "method": "..." },
  "certificate_status": { "certifiable": true, "reason": "...", "blocking_steps": [] },
  "steps": [ {"rule": "...", "before": "...", "after": "...", "side_conditions": [...]}, ... ],
  "has_certificate": true
}
```

`verification` and `certificate_status` are exactly the dicts returned by the
`.verification` and `.certificate_status` getters; `steps` is exactly `.steps`.
`kind` is a stable discriminator string — useful when logs or RPC payloads
mix `DerivedResult` envelopes with other structured outputs (e.g. error
envelopes carrying `E-SUBSYSTEM-NNN` codes).

### Schema versions

Two independent version constants, both starting at `1`:

| Constant | Governs |
|---|---|
| `alkahest.RESULT_SCHEMA_VERSION` | The envelope: the set of top-level keys (`kind`, `value`, `verification`, `certificate_status`, `steps`, `has_certificate`, ...) |
| `alkahest.STEPS_SCHEMA_VERSION` | One entry of `steps`: full-mode field names and the compact-mode short-key mapping |

Also available as `DerivedResult.SCHEMA_VERSION` / `DerivedResult.STEPS_SCHEMA_VERSION`
class attributes, and documented alongside the field-name contract in
`alkahest._result_schema` (`STEP_FIELDS`, `STEP_FIELDS_COMPACT`). Either
constant is bumped independently if its shape ever changes, so pinning
`schema_version`/`steps_schema_version` in your own parsing code is safe
across upgrades that don't touch the piece you depend on.

### Compact mode

`mode="compact"` keeps the same top-level envelope shape but shrinks the
biggest token costs:

- **Steps** use short keys — `r` for `rule`, `s` for `side_conditions` — and
  **omit `before`/`after` entirely**. Those two expression strings are
  usually the largest part of a multi-step derivation and the single
  biggest win for token budget. `s` is itself omitted from a step's dict
  when that step has no side conditions (the common case).
- **`verification`** is pruned to `status` and `externally_verified` only.
  These are the two fields that carry the honesty signal — whether the
  result is verified, and whether that verification happened out-of-process
  — so they are never renamed, abbreviated, or dropped in compact mode.
- **`certificate_status`** is pruned to `certifiable` and `reason`; the
  `blocking_steps` diagnostic list (which repeats `before`/`after` text) is
  dropped.
- **No mode ever includes Lean certificate source text.** `has_certificate`
  (bool) plus `certificate_status["reason"]` is enough to know whether a
  certificate exists and, if not, why — without paying for the source. Use
  the `.certificate` getter when you actually need the Lean source.

```python
dr.to_dict(mode="compact")
# {
#   "kind": "alkahest.derived_result",
#   "schema_version": 1,
#   "steps_schema_version": 1,
#   "value": "...",
#   "verification": {"status": "certificate_available", "externally_verified": false},
#   "certificate_status": {"certifiable": true, "reason": "emitted"},
#   "steps": [{"r": "diff_sin"}, {"r": "sqrt_of_square_positive", "s": ["x > 0"]}],
#   "has_certificate": true
# }
```

Prefer `to_dict(mode="compact")` / `to_json(mode="compact")` over reading
`.steps` directly in hot loops — batch derivations, autoresearch search
plumbing, or anywhere you're serialising many `DerivedResult`s and only need
the rule names, side conditions, and verification status rather than full
before/after expression text.

An invalid `mode` (anything other than `"full"`/`"compact"`) raises
`ValueError`.

## Beyond one call

`DerivedResult` is per-call. To accumulate many results into a citable, serialisable,
re-verifiable artifact — a DAG of claims with stable IDs, hypotheses, and certificate
status — see [claim graphs](./claim-graphs.md).

## See also

- [Autoresearch / agent loops](./search-plumbing.md)
- [Batch](./batch.md) — produce many `DerivedResult`s without aborting on one failure
- [Budgets](./budgets.md) — bound the call that produced the derivation
- [Certificate coverage](./certificate-coverage.md) — `certificate_status` in the envelope