# Alkahest architecture

## Crates and layers

| Path | Role |
|------|------|
| `alkahest-core/` | Rust kernel (all math). Add new algorithms here. Published on [crates.io](https://crates.io/crates/alkahest-cas) as `alkahest-cas`. |
| `alkahest-mlir/` | MLIR dialect and lowering passes. Only touch for codegen work. |
| `alkahest-py/` | PyO3 bindings (thin glue). Exposes Rust APIs to Python; add new bindings here when a Rust function needs a Python surface. |
| `python/alkahest/` | Pure-Python layer. Use for Python-only utilities (parsing, pretty-printing, pytrees, context manager). |

**Stack (high level):** Rust kernel → FLINT/Arb (polynomials, ball arithmetic) → vendored egglog + colored e-graphs (simplification) → Cranelift/LLVM JIT + MLIR (native and GPU codegen) → PyO3 → Python.

## Stable vs experimental API

- **Rust stable surface:** `alkahest_core::stable` re-exports. Adding a function here triggers `cargo semver-checks` in CI — be intentional.
- **Python stable surface:** `alkahest.__all__` in `python/alkahest/__init__.py`. Same rule.
- Experimental / unstable APIs go under `alkahest_core::experimental` and `alkahest.experimental`.
- `scripts/check_api_freeze.py` enforces this in CI.

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
│   │   ├── kernel/        # hash-consed expression DAG, ExprPool
│   │   ├── algebra/       # noncommutative Pauli / Clifford rules
│   │   ├── parse.rs       # Pratt expression parser (parse / ParseError)
│   │   ├── poly/          # UniPoly, MultiPoly, RationalFunction
│   │   ├── simplify/      # e-graph simplification (egglog)
│   │   ├── diff/          # symbolic differentiation
│   │   ├── integrate/     # symbolic integration
│   │   ├── calculus/      # series / limits
│   │   ├── jit/           # LLVM JIT and interpreter
│   │   ├── ball/          # Arb ball arithmetic
│   │   ├── ode/           # ODE analysis
│   │   ├── dae/           # DAE analysis and index reduction
│   │   ├── diffalg/       # Rosenfeld–Gröbner / differential elimination (groebner)
│   │   ├── solver/        # polynomial solving: Gröbner triangular, regular chains, homotopy
│   │   ├── lean/          # Lean 4 proof certificate export
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
