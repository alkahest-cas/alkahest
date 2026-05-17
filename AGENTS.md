* If any significant changes are made to code run CI and tests
* Follow instructions in [`CONTRIBUTING.md`](CONTRIBUTING.md)
* If asked to fix CI: check CI status using `gh` CLI if available, otherwise check https://github.com/alkahest-cas/alkahest/
* If asked to generate a report: write it to `tmp/`, include at the top — the date, the agent and model writing it (e.g. "Claude Code claude-sonnet-4-6"), and the full git commit hash being reviewed

## Build

```bash
# Build and install the Python extension (required before running Python tests or examples)
pip install maturin
maturin develop --release

# With optional features (groebner required for solve/diophantine/homotopy)
maturin develop --release --features "parallel egraph jit groebner"
```

## Testing

```bash
# Rust unit + proptest + doctest
cargo test --workspace

# Python suite (default excludes @pytest.mark.slow — the long sparse_interp roadmap test)
pytest tests/

# Sparse roadmap stress test (Tier 1b style; needs groebner build — see TESTING.md)
# pytest -m slow tests/test_sparse_interp.py --timeout=0 -v --override-ini="addopts=-v"
#
# Full Python run *including* slow markers (unset pytest.ini filter):
# pytest tests/ --override-ini="addopts=-v"

# Run both
cargo test --workspace && pytest tests/
```

Skipped tests are feature-gated (e.g. `--features groebner`) or oracle tests requiring SymPy/Mathematica. Do not unskip them unless the relevant feature is enabled.

## Architecture

- `alkahest-core/` — Rust kernel (all math). Add new algorithms here.
- `alkahest-mlir/` — MLIR dialect and lowering passes. Only touch for codegen work.
- `alkahest-py/` — PyO3 bindings (thin glue). Exposes Rust APIs to Python; add new bindings here when a Rust function needs a Python surface.
- `python/alkahest/` — Pure-Python layer. Use for Python-only utilities (parsing, pretty-printing, pytrees, context manager).

## Stable vs experimental API

- **Rust stable surface:** `alkahest_core::stable` re-exports. Adding a function here triggers `cargo semver-checks` in CI — be intentional.
- **Python stable surface:** `alkahest.__all__` in `python/alkahest/__init__.py`. Same rule.
- Experimental / unstable APIs go under `alkahest_core::experimental` and `alkahest.experimental`.
- `scripts/check_api_freeze.py` enforces this in CI.

## Adding a new mathematical primitive

See CONTRIBUTING.md for the 6-step registration process. The short version: implement in `alkahest-core/src/primitive/`, register in `PRIMITIVE_REGISTRY`, add PyO3 binding in `alkahest-py/src/lib.rs`, export in `python/alkahest/__init__.py`, add to `__all__`, write tests.

## Error codes

Every error type has a stable `E-SUBSYSTEM-NNN` code. When adding a new error, follow the existing pattern in the relevant `mod.rs` file and add the code to `docs/` if user-facing.

## Key files

| Path | Purpose |
|------|---------|
| `alkahest-core/src/lib.rs` | Crate root, all re-exports |
| `alkahest-core/src/kernel/mod.rs` | `ExprPool`, `ExprData`, `ExprId` |
| `alkahest-core/src/stable.rs` | Semver-stable public API surface |
| `alkahest-py/src/lib.rs` | All PyO3 `#[pyfunction]` / `#[pyclass]` bindings |
| `python/alkahest/__init__.py` | Python package root and `__all__` |
| `scripts/check_api_freeze.py` | CI guard for stable API surface |
