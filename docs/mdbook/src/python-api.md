# Python API reference

The Sphinx-generated Python API is published alongside this guide:

**[Open the Python API documentation](https://alkahest-cas.github.io/alkahest/api/)**

It includes `ExprPool`, simplification, calculus, polynomials, numerics, transforms, matrices, ODE/DAE, solvers, codegen, error types, and the [search / workload](https://alkahest-cas.github.io/alkahest/api/api/workload.html) surface (`Budget`, `batch_map`, `DerivedResult.to_dict`, …).

Conceptual chapters for agent-facing plumbing:

| Topic | Guide |
|---|---|
| Budgets, cancellation, seeds | [Budgets](./budgets.md) |
| Batch / streaming fan-out | [Batch](./batch.md) |
| Compact machine-parseable results | [Derivation logs](./derivations.md#machine-parseable-output-to_dict--to_json) |
| Session provenance | [Claim graphs](./claim-graphs.md) |
| Overview | [Autoresearch / agent loops](./search-plumbing.md) |
