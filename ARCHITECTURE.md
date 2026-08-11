# Alkahest architecture

## Crates and layers

| Path | Role |
|------|------|
| `alkahest-core/` | Rust kernel (all math). Add new algorithms here. The Cargo **package** is `alkahest-cas` (published on [crates.io](https://crates.io/crates/alkahest-cas)); external code writes `alkahest_cas::…` and Cargo commands take `-p alkahest-cas`. `alkahest-py` renames the dependency to `alkahest_core` for its own use — that alias is workspace-local. |
| `alkahest-mlir/` | MLIR dialect and lowering passes. Only touch for codegen work. |
| `alkahest-py/` | PyO3 bindings (thin glue). Exposes Rust APIs to Python; add new bindings here when a Rust function needs a Python surface. |
| `python/alkahest/` | Pure-Python layer. Use for Python-only utilities (parsing, pretty-printing, pytrees, context manager). |

**Stack (high level):** Rust kernel → FLINT/Arb (polynomials, ball arithmetic) → vendored egglog + colored e-graphs (simplification) → Cranelift/LLVM JIT + MLIR (native and GPU codegen) → PyO3 → Python.

## Stable vs experimental API

- **Rust stable surface:** `alkahest_cas::stable` re-exports. Adding a function here triggers `cargo semver-checks` in CI — be intentional.
- **Python stable surface:** `alkahest.__all__` in `python/alkahest/__init__.py`. Same rule.
- Experimental / unstable APIs go under `alkahest_cas::experimental` and `alkahest.experimental`.
- `scripts/check_api_freeze.py` enforces this in CI.

## Resource model — read this before writing a long-running loop

`ExprPool` is an **append-only** hash-consed arena (a `boxcar::Vec` of nodes plus a
`DashMap` index). There is no `clear`, no `truncate`, no refcount and no GC: **the only
way to reclaim interned nodes is to drop the whole pool.** Every `Expr`, `Matrix`,
`Series` and `DerivedResult` holds a *strong* reference to its pool, so retaining one
result retains every node ever interned alongside it.

Consequences that matter at the architecture level:

- Growth on a shared pool is **linear and unbounded** — roughly 200 bytes of resident
  memory per interned node — while per-call time stays **flat**. The failure mode is a
  clean OOM with no latency warning beforehand.
- `PyExprPool` exposes no `__len__`/`stats`, so the growth is not observable from Python.
- The supported pattern is therefore **one pool per problem**, dropped when the problem
  is done. This is documented for users in
  [`docs/mdbook/src/budgets.md`](docs/mdbook/src/budgets.md#exprpool-never-reclaims).

Budget state is **thread-local** (`budget::STACK`); the cancellation flag is
**process-wide** (`budget::CANCELLED`). Anything surprising about budgets and threads
follows from that asymmetry.

## Key files

| Path | Purpose |
|------|---------|
| `alkahest-core/src/lib.rs` | Crate root, all re-exports |
| `alkahest-core/src/kernel/mod.rs` | `ExprPool`, `ExprData`, `ExprId` |
| `alkahest-core/src/stable.rs` | Semver-stable public API surface |
| `alkahest-py/src/lib.rs` | All PyO3 `#[pyfunction]` / `#[pyclass]` bindings |
| `python/alkahest/__init__.py` | Python package root and `__all__` |
| `scripts/check_api_freeze.py` | CI guard for stable API surface |

## Directory layout

```
alkahest/
├── alkahest-core/         # Rust kernel (published as the alkahest-cas crate)
│   ├── src/
│   │   ├── kernel/        # hash-consed expression DAG, ExprPool (append-only: see note)
│   │   ├── algebra/       # noncommutative Pauli / Clifford rules
│   │   ├── parse.rs       # Pratt expression parser (parse / ParseError)
│   │   ├── poly/          # UniPoly, MultiPoly, RationalFunction, real-root isolation
│   │   ├── simplify/      # rule engine + e-graph simplification (egglog)
│   │   ├── diff/          # symbolic differentiation
│   │   ├── integrate/     # symbolic integration
│   │   ├── calculus/      # series / limits / Euler–Maclaurin asymptotics
│   │   ├── jit/           # LLVM JIT and interpreter
│   │   ├── ball/          # Arb ball arithmetic
│   │   ├── validated/     # Taylor models, Moore–Skelboe: rigorous bounds over a box
│   │   ├── holonomic/     # creative telescoping (Zeilberger), P-recursive certificates
│   │   ├── budget/        # cooperative wall/step budget + process-wide cancel flag
│   │   ├── logic/         # Formula, SMT-LIB emitter, standalone DPLL
│   │   ├── matrix/        # linear algebra; three-valued zero test (E-LINALG-010/E-MAT-004)
│   │   ├── real/          # CAD real quantifier elimination, SOS/Positivstellensatz
│   │   ├── ode/           # ODE analysis
│   │   ├── dae/           # DAE analysis and index reduction
│   │   ├── diffalg/       # Rosenfeld–Gröbner / differential elimination (groebner)
│   │   ├── solver/        # polynomial solving: Gröbner triangular, regular chains, homotopy
│   │   ├── lean/          # Lean 4 proof certificate export
│   │   ├── errors/        # the E-*-NNN code registry (codes.rs) — every code lives here
│   │   ├── plot/          # SVG polyline + Graphviz DOT renderers (dependency-free)
│   │   └── primitive/     # primitive registration system
│   └── benches/           # criterion benchmarks
├── alkahest-mlir/         # MLIR dialect and lowering passes
├── alkahest-py/           # PyO3 bindings (Rust side)
├── python/alkahest/       # Python package
│   ├── _plot.py           # plotting: plot, plot3d, plot_parametric, plot_implicit, …
│   ├── _transform.py      # trace, grad, jit decorators
│   ├── _pytree.py         # JAX-style pytree flattening
│   ├── _context.py        # context manager and defaults
│   ├── _budget.py         # Budget, request_cancel, run_with_wall_fallback
│   ├── _batch.py          # batch_map / batch_map_iter / *_many (budget-propagating)
│   ├── ansatz.py          # parametric families + fit/certify  (public: alkahest.ansatz)
│   ├── crosscheck.py      # differential testing against an external CAS oracle
│   ├── smt.py             # SMT-LIB export and z3/cvc5 bridge
│   ├── research.py        # claim graphs / session provenance
│   ├── _certificates.py   # certificate coverage ledger, certifiable()
│   └── experimental/      # unstable API surface
│       └── _fastplotlib.py# GPU-accelerated plotting adapter
├── examples/              # runnable end-to-end examples
│   └── rust_quickstart/   # self-contained Cargo project for alkahest-cas
├── tests/                 # Python test suite (pytest + hypothesis)
├── benchmarks/            # Python benchmarks and competitor comparisons
├── fuzz/                  # AFL++ fuzz targets
├── docs/                  # mdBook and Sphinx documentation
├── website/               # landing page (alkahest-cas.github.io)
│   └── src/               # index.html + styles.css source (deployed via CI)
├── alkahest-skill/        # Skill for AI to use alkahest
├── agent-benchmark/       # benchmark for comparing AI use of alkahest vs other CAS
└── scripts/               # CI helpers (API freeze check, error codes)
```
