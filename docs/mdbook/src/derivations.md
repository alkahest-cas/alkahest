# Derivation logs

Every transformation in Alkahest returns a `DerivedResult` that records the exact sequence of rewrite steps applied. This log is the foundation for both human inspection and Lean proof export.

## DerivedResult

`DerivedResult` is the return type of `diff`, `simplify`, `integrate`, and all top-level operations:

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
| `.certificate` | `str \| None` | Generated Lean 4 source, when a derivation log exists |

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
- `NonZero(x)` — `x` must be non-zero (e.g. for `x/x → 1`)
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
