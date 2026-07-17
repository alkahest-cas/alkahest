# RFC 0001: Incremental / Merkle expression storage and slotted e-graphs

- **Status:** Design + bounded prototype (not a mainline commitment)
- **Priority:** P2 / exploratory (deprioritized relative to mathematical coverage and Lean parity)
- **Prototype:** `alkahest_core::experimental::merkle_pool`
- **Created:** 2026-07-16

## Summary

Notebook and agent sessions repeatedly rebuild or re-traverse expression DAGs when cells, tools, or assumptions change slightly. This RFC sketches a **content-addressed (Merkle) expression pool** as a complement to today's session-local `ExprPool`, and records a **slotted e-graph** hypothesis for binder-heavy rewriting. Scope for now is design plus a small in-process prototype with unit tests only—no Salsa wiring, no change to default `ExprPool` behavior, and no promise of a production cutover.

## Motivation / problem

Alkahest already hash-conses within a single `ExprPool`: identical `ExprData` shares an `ExprId`, and `pool_persist` can checkpoint the intern table to disk. That solves *intra-process* structural sharing and coarse session restore.

It does **not** yet solve:

1. **Cross-session / cross-process identity.** An `ExprId` is an arena index valid only inside one pool. Reopening a notebook or forking an agent tool call often re-interns from scratch (or remaps IDs), so caches keyed by `ExprId` do not travel.
2. **Fine-grained invalidation.** Editing one equation in a multi-step derivation ideally recomputes only dependent queries. Full-pool checkpoints and whole-graph simplify/diff passes treat the session as mostly monolithic.
3. **Binder rewriting.** Summation indices, integration variables, and quantifiers (`Forall` / `Exists` / `RootSum`) stress vanilla e-graphs; α-equivalent binders do not share classes without an encoding scheme.

Interactive playground and agent workloads amplify (1)–(2): many small edits, repeated simplify/diff/JIT on overlapping subtrees, and a desire for cache hits across tool boundaries.

## Current `ExprPool` (baseline)

| Property | Behavior today |
|----------|----------------|
| Identity | `ExprId(u32)` — index into a lock-free arena |
| Dedup | Hash-cons on `ExprData` (children are `ExprId`s) |
| Sharing | Automatic within one pool |
| Persistence | Opt-in binary checkpoint (`pool_persist`); restore rebuilds index |
| Cross-pool | No stable content address; IDs do not compare across pools |

`ExprId` is already a content key *inside* one session. Merkle storage asks for a **global, reconstructible digest** of the DAG so the same subtree can be looked up without sharing the arena.

## Sketch: content-addressed / Merkle pool

**Idea.** Store nodes keyed by a content hash of `(tag, payload, child_hashes…)` rather than by mutable index. Compound nodes refer to children by hash. The root hash identifies the whole DAG (Merkle-style).

```text
         H(Add, [Hx, Hy])
            /        \
     H(Symbol x)   H(Symbol y)
```

**Vs `ExprPool`:**

| | `ExprPool` | Merkle pool (sketch) |
|--|------------|----------------------|
| Key | arena `ExprId` | content hash |
| Children | `ExprId` | child hashes |
| Equality | pointer/id compare in-pool | hash compare (cross-pool) |
| Mutation | append-only intern | append-only intern by hash |
| Persist | serialize arena order | serialize hash → node map (order-independent) |

**Non-goals for the sketch / prototype:** mmap arenas, Cap'n Proto layouts, IPFS/CID codecs, distributed stores, or replacing `ExprData` heap vectors.

**Bounded prototype (this PR):** `experimental::merkle_pool` implements intern + lookup for a small expression subset (`Symbol`, `Integer`, `Add`, `Mul`, `Pow`, `Func`), with a stable non-crypto digest suitable for tests. It does **not** integrate with Salsa, egglog, or the Python API, and does not alter default `ExprPool`.

## Slotted e-graph binder-rewrite hypothesis

**Hypothesis.** Slotted e-graphs (PLDI 2025) parameterize e-classes by *slots* (bound-variable roles). For Alkahest binders—sums, integrals, quantifiers, `RootSum`—slots could let α-equivalent terms join the same e-class without De Bruijn encoding overhead or name-capture bugs in egglog string round-trips.

**Why only a hypothesis here.** Colored e-graphs already cover a higher-leverage conditional-rewrite path. Binder volume in the current rule set is still modest. Validating slotted e-graphs needs a dedicated spike against `simplify/egraph.rs` and binder-heavy integrate/sum workloads—not a Merkle-pool prerequisite.

**Relation to Merkle storage.** Content hashes compose under binders only if the hash scheme is α-aware (or slots are hashed, not concrete names). The prototype deliberately uses concrete symbol names; α-normalized hashing is a later experiment.

## Incremental computation (Salsa) — deferred

Demand-driven incremental frameworks (e.g. Salsa) could memoize `simplify` / `diff` / `integrate` keyed by content hashes so a notebook cell edit recomputes only dependent queries. That is **explicitly out of scope** for this RFC's prototype: hashing and lookup come first; query graphs and revision tracking are a separate, higher-effort architecture bet (see infra audit: exploratory / very high effort).

## Non-goals

- Replacing or changing default `ExprPool` / `ExprId` semantics
- Full Salsa (or any incremental query) integration
- Production cryptographic CIDs, network distribution, or multi-tenant stores
- Shipping slotted e-graphs in mainline simplify
- Python / PyO3 exposure of the prototype
- Performance claims vs `ExprPool` without measured workloads

## Risks

| Risk | Notes |
|------|--------|
| Hash cost | Per-node hashing can dominate tiny expressions; `ExprId` compare is cheaper in-session |
| Digest stability | Crypto or versioned codecs needed before cross-release cache sharing |
| α-equivalence | Name-based hashes split α-equivalent binders; wrong caches or missed sharing |
| Dual representation | Keeping Merkle + arena pools invites conversion bugs and double memory |
| E-graph mismatch | Egglog still speaks arena/`ExprId` trees; bridging adds round-trips |
| Scope creep | Easy to over-build before notebook/agent metrics justify the complexity |

## Success metrics (for a later prototype / spike)

Treat these as **gates before any mainline commitment**, not as deliverables of this PR:

1. **Cache hit rate:** ≥ 50% subtree hash hits on a recorded demo-playground or agent session replay with single-cell edits (vs cold re-intern).
2. **Invalidation locality:** changing one leaf invalidates only ancestor hashes; unrelated roots remain hits (property tests + a small session fixture).
3. **Parity:** Merkle ↔ `ExprPool` round-trip for the supported node subset preserves structure (bijective on a fixed test corpus).
4. **Overhead budget:** intern+lookup of a 1e4-node random DAG within a small constant factor of `ExprPool::intern` on the same machine (document absolute numbers; do not ship if > ~3× without a compensating cross-session win).
5. **Slotted e-graph spike (optional follow-up):** on a binder suite (nested sums / `RootSum`), show either (a) correct α-join into one e-class or (b) a written reject with measurements—no silent mainline merge.

## Decision

**Deprioritize** full incremental/Merkle storage and slotted e-graphs as product commitments. Land this RFC and the bounded `experimental::merkle_pool` prototype so the design is reviewable and future spikes have a documented baseline. Revisit only if success metrics above are measured on real notebook/agent traces.

## References

- In-tree: `alkahest-core/src/kernel/pool.rs`, `pool_persist.rs`, `simplify/egraph.rs`
- Infra audit notes: content-addressable expressions, Salsa-style incremental computation, slotted e-graphs (exploratory tier)
- Astrolabe-style content identifiers for math fragments (arXiv:2604.10435) — inspirational, not adopted
- Slotted E-Graphs (PLDI 2025) — binder rewriting hypothesis
