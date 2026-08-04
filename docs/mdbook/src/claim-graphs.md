# Claim graphs

`DerivedResult` ([derivation logs](./derivations.md)) is a **per-call** object. A research
loop that runs for days produces thousands of them and has nowhere to put them, so its
output degrades into a transcript — and transcripts do not survive context compaction.

`alkahest.research` supplies the session-level artifact that is missing: a **directed
acyclic graph of claims**, serialisable to disk, diffable, re-verifiable, and renderable
into a document a human referee can read.

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

with ak.research.session(title="Worked example", pool=pool, capture=True) as s:
    integrand = x / (x**pool.integer(2) + pool.integer(1))
    definite = ak.integrate(integrand, x, pool.integer(0), pool.integer(1))
    ak.simplify(pool.integer(2) * definite.value - ak.log(pool.integer(2)))

graph = s.graph
graph.save("claims.json")
print(graph.to_markdown())
```

## What a claim carries

| Field | Meaning |
| --- | --- |
| `id` | Content-addressed identifier, `clm_` + 16 hex chars |
| `statement` / `latex` / `kind` | Normalised statement; `"expr"`, `"relation"`, or `"text"` |
| `hypotheses` | Predicates from the governing `Assumptions` |
| `method` | Operation that produced it (`"integrate"`, `"simplify"`, `"conjecture"`, …) |
| `status` / `evidence` / `verification` | Copied **verbatim** from `DerivedResult.verification` |
| `derivation` | `DerivedResult.steps` |
| `certificate` / `certificate_format` | Certificate source, when one was emitted |
| `depends_on` | IDs this claim was derived from |
| `check` | Machine-readable re-verification recipe |
| `provenance` | Operation, arguments, Alkahest version, build features |
| `recorded_at` | ISO-8601 UTC timestamp — **volatile**, excluded from the digest |
| `audit` | Re-verification outcomes appended by `graph.verify()` |

### Stable identity

The ID is `sha256` over a canonical JSON encoding of the **normalised statement**, the
**sorted hypotheses**, and the **method** — nothing else. It deliberately excludes
timestamps, dependency edges, and the library version, so the same claim derived twice in
different sessions receives the same ID and citing across runs is meaningful:

```python
from alkahest.research import claim_id

claim_id("(1/2 * log(2))", ["x > 0"], "integrate")   # 'clm_…', deterministic
```

Expression statements are put into the kernel's normal form (via `simplify`) before
hashing, so `1 + x**2` and `x**2 + 1` produce the same ID. Pass `normalize=False` to
`session()` in very hot loops; IDs then depend on how the expression was built.

Because IDs are content-addressed, re-deriving the same claim is not an error — the stored
claim keeps its status and derivation and merely gains the new dependency edges.

### Hypotheses travel with the claim

This is the whole point of `Assumptions` being first-class. The predicates in the active
assumption context — whether set by `session(assumptions=…)` or by an enclosing
`alkahest.context(assumptions=…)` — are recorded on every claim and are part of its
identity. A claim proved under `x > 0` and the same claim proved unconditionally are
different claims with different IDs.

## Recording

### Automatic capture

`session(capture=True)` wraps the `DerivedResult`-producing functions in the `alkahest`
module namespace, so every result computed inside the block is recorded, and dependency
edges are inferred by looking for previously-recorded values among the *subexpressions* of
each call's arguments.

Automatic capture has a boundary, and it is stated rather than hidden:

- it sees calls made **through the module namespace** — `alkahest.integrate(...)`;
- it does **not** see a name bound before the hooks were installed
  (`from alkahest import integrate`) or methods on objects (`Assumptions.simplify`).

`alkahest.research.captured_operations()` lists exactly what is hooked, and
`session.capture_report()` reports the mode, the hooked operations, the claim count, and
any error raised inside the hook. Hooks are installed once and never removed — removing
them when one session exits would silently disable capture for a session still running on
another thread.

### Explicit recording

For anything outside that boundary, `record()` is one line and always complete:

```python
with ak.research.session(pool=pool) as s:
    result = ak.integrate(integrand, x)
    s.record(result, method="integrate", label="Antiderivative", sources=[integrand])

    # or, equivalently, letting the session make the call:
    s.run(ak.integrate, integrand, x)
```

### Conjectures

`conjecture()` records a claim that is **not** proved. Its status is hard-wired to
`"unverified"`; there is deliberately no parameter to say otherwise.

```python
s.conjecture(
    "-2 * integral(x/(x^2+1), dx, 0, 1) + 1 * log(2) = 0",
    evidence="integer relation found by guess_relation at 60 digits",
)
```

## The honesty invariant

**The recording layer never upgrades a claim's status.**

- A claim's status is whatever `DerivedResult.verification["status"]` said.
- `conjecture()` always produces `"unverified"`.
- `verify()` may only *lower* confidence: a failed re-check sets `"refuted"`; a successful
  one appends an audit entry and promotes nothing.

The renderers follow the same rule. An emitted-but-unchecked Lean certificate is marked
`[CERT ONLY, UNCHECKED]`, never as a proof, and every document opens with the exact
machine-checkable fraction:

> **Machine-checkable subset: 1 of 4 claims (25%).** Only claims marked *verified* were
> checked by a checker. Everything else is recorded evidence and must not be read as proved.

`Claim.machine_checked` is true only for `exactly_verified` and `lean_checked`.

## Querying

```python
graph.by_status("unverified")
graph.machine_checkable()
graph.dependencies(cid)     # direct
graph.ancestors(cid)        # transitive, what it rests on
graph.dependents(cid)       # direct
graph.impact(cid)           # transitive — "what dies if this claim is false?"
graph.roots(), graph.leaves(), graph.summary(), graph.topological_order()
```

`impact()` is the query a loop needs when a lemma turns out to be wrong: it returns every
claim that transitively cited it.

Edges always point at claims already in the graph, so cycles are impossible by
construction; graphs loaded from JSON are topologically checked and raise `CycleError`.

## Serialisation

`to_json()` / `from_json()` round-trip losslessly, with an explicit `schema_version`.
Output uses sorted keys, so a byte diff is a content diff. Reading a document whose
`schema_version` is newer than this build raises `ClaimGraphError` rather than guessing.

```python
graph.save("claims.json")
same = ak.research.ClaimGraph.load("claims.json")

graph.to_json(stable=True)   # drops timestamps: byte-identical across runs
graph.digest()               # sha256 of the stable form
```

Volatile data (`recorded_at`, `started_at`, `finished_at`, audit timestamps) is confined to
explicitly-marked fields and never contributes to identity or to the digest, so two runs of
the same computation produce identical `stable=True` documents and identical digests.

## Rendering

`to_markdown()` emits a document with a verification summary table, then one section per
claim carrying its statement (LaTeX where available), hypotheses, method, dependency
links, derivation table, certificate status, and last re-check.

`to_latex()` emits an `article` (or just the body, with `standalone=False`) in which every
claim carries `\label{clm:<id>}` and dependencies are `\hyperref` links — so a loop that
ran for a week emits a writeup with every claim linked to its derivation and its
certificate status. Both build on `latex()` and the `DerivedResult` machinery rather than
reimplementing them.

## Re-verification

`graph.verify()` walks the graph and re-checks whatever carries a recipe, re-parsing each
claim's recorded text into a **fresh pool** — so a graph loaded from disk can be
revalidated against a newer library build rather than trusted blindly.

| Recipe `kind` | What is re-checked |
| --- | --- |
| `antiderivative` | `simplify(diff(F, x) - f) == 0` |
| `definite_integral` | `simplify(F(b) - F(a) - value) == 0` |
| `derivative` | `simplify(diff(e, x) - value) == 0` |
| `identity` | `simplify(lhs - rhs) == 0` |
| `zero` | `simplify(e) == 0` |
| `numeric_relation` | `abs(sum a_i c_i) <= tolerance` — evidence only |

Outcomes are `ok` (exact), `numeric_ok` (the exact check was inconclusive but a numeric
residual passed — evidence, not a proof), `failed`, `inconclusive`, or `skipped`.

```python
report = graph.verify()
report.ok            # False if anything was refuted
report.summary()     # {'ok': 2, 'numeric_ok': 1, 'skipped': 1}
print(report.to_markdown())
```

## A complete loop

[`examples/pslq_research_loop.py`](https://github.com/alkahest-cas/alkahest/blob/main/examples/pslq_research_loop.py)
runs the full experimental-mathematics loop — high-precision quadrature, `guess_relation`
for the integer relation, a recorded conjecture, symbolic proof, Lean certificate, JSON
round trip, and re-verification — and prints the rendered document.
