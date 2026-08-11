# Python API reference

The Sphinx-generated Python API is published alongside this guide:

**[Open the Python API documentation](https://alkahest-cas.github.io/alkahest/api/)**

It includes `ExprPool`, simplification, calculus, polynomials, numerics, transforms, matrices, ODE/DAE, solvers, codegen, error types, and the [search / workload](https://alkahest-cas.github.io/alkahest/api/api/workload.html) surface (`Budget`, `batch_map`, `DerivedResult.to_dict`, …).

Conceptual chapters for agent-facing plumbing:

| Topic | Guide |
|---|---|
| Budgets, cancellation, seeds, **pool lifetime** | [Budgets](./budgets.md) |
| Batch / streaming fan-out | [Batch](./batch.md) |
| Compact machine-parseable results | [Derivation logs](./derivations.md#machine-parseable-output-to_dict--to_json) |
| Session provenance | [Claim graphs](./claim-graphs.md) |
| Overview | [Autoresearch / agent loops](./search-plumbing.md) |

## Submodules

Not everything lives on the top-level namespace as a function. These are reached as
`alkahest.<name>` and documented in their own chapters:

| Module | What it is | Guide |
|---|---|---|
| `alkahest.ansatz` | Parametric families (`polynomial`, `rational`, `exponential_polynomial`, `linear_combination`, `quadratic_form`) plus `fit`, `enumerate_family`, `certify_nonneg` | [Ansatz families](./ansatz.md) |
| `alkahest.crosscheck` | Differential testing against an external CAS oracle: `check`, `sweep`, `run_frozen_corpus`, `to_sympy`, `register_oracle` | [Cross-CAS testing](./crosscheck.md) |
| `alkahest.smt` | SMT-LIB 2 export and z3/cvc5 bridge: `to_smtlib`, `solve`, `supported`, `solvers` | [SMT bridge](./smt.md) |
| `alkahest.research` | Session claim graphs and provenance | [Claim graphs](./claim-graphs.md) |
| `alkahest.experimental` | Transforms, `dsolve`, asymptotics, `residue`, `Fps`, `to_jax` — may change in a minor release. **Must be imported explicitly** (`from alkahest import experimental as ex`); it is not an attribute of the top-level module until then | [Stability policy](./stability.md) |
| `alkahest.rl` | Verifiable RL environments | [Reinforcement learning](./rl.md) |
| `alkahest.number_theory`, `alkahest.modular`, `alkahest.lattice` | FLINT-backed integer and lattice routines | — |

`alkahest.ansatz`, `alkahest.crosscheck` and `alkahest.smt` are new in 3.8. They are in
`alkahest.__all__` and resolve on attribute access without a separate import, as do
their error classes `AnsatzError`, `CrossCheckError` and `SmtError`.
