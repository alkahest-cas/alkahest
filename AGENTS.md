* If any significant changes are made to code run CI and tests
* Before pushing Rust code, run `cargo fmt` to ensure formatting passes CI
* Follow instructions in [`CONTRIBUTING.md`](CONTRIBUTING.md)
* If asked to fix CI: check CI status using `gh` CLI if available, otherwise check https://github.com/alkahest-cas/alkahest/
* If asked to generate a report: write it to `temp-alkahest/testing/` or `temp-alkahest/planning/` depending on what label describes the report better, include at the top — the date, the agent and model writing it (e.g. "Claude Code claude-sonnet-4-6"), and the full git commit hash being reviewed
* A `LOCAL-AGENTS.md` file may exist at the repo root — it is untracked by git and contains device-specific instructions (paths, local tooling, machine-specific overrides). If present, follow its instructions in addition to this file.
* If asked to write a report do not include time estimates for how long implementation will take like "~1 week" etc

## Keeping a branch current

**Rebase a pull request onto `origin/main` rather than merging `main` into it,
and re-run CI after you do.** Merging leaves a branch whose checks were green
against a `main` that has since moved; rebasing makes the checks mean what they
appear to mean.

This is not hypothetical. In the 3.11 cycle several pull requests merged on
their own green checks while `main` had moved underneath them, and one of them
merged with **no conflict at all and could not have compiled**: the branch had
widened a function's signature and the other side still called it at the old
arity. Git reported `MERGEABLE` for that happily, because a textual merge cannot
see an arity. Three rules follow:

* A clean merge is not a compiling merge. Rebuild.
* A compiling merge is not a passing one. **Run** the suites — `cargo test
  --workspace` and `pytest tests/` — rather than `cargo check`.
* Resolve conflicts at **item** granularity: whole functions, whole `Case(...)`
  entries, whole changelog bullets. A both-sides-appended conflict usually
  splits mid-item, with the closing brace below the `>>>>>>>` marker and shared
  by both sides; keeping both halves without restoring it produces a file that
  can still parse, still format and still lint while being wrong. Before
  committing, `grep -rn '^<<<<<<<\|^>>>>>>>\|^=======$'` over the tree, and
  stage resolved files **by name** — never `git add -A`.

When several pull requests are in flight over the same files, land them one at a
time, rebasing each onto the new `main` before it merges.

## Demo playground

[`demo-playground/`](demo-playground/) contains an interactive web app for demoing and recording Alkahest. It has a notebook interface, an AI agent chat mode, and a CLI for orchestrating and recording demos as video.

Quick start (requires Node ≥ 18 with pnpm, and Python ≥ 3.9):
```bash
cd demo-playground
cp .env.example web/.env.local   # add API key
pnpm install
pnpm start                        # starts web (port 3000) + Python server (port 8000)
```

See [`demo-playground/README.md`](demo-playground/README.md) for full documentation.

## Build

```bash
# Build and install the Python extension (required before running Python tests or examples)
pip install maturin
maturin develop --release

# With optional features (groebner required for solve/diophantine/homotopy)
maturin develop --release --features "parallel egraph cranelift jit groebner"
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

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for crates, directory layout, stable vs experimental API surfaces, and key files.

## Adding a new mathematical primitive

See CONTRIBUTING.md for the 6-step registration process. The short version: implement in `alkahest-core/src/primitive/`, register in `PRIMITIVE_REGISTRY`, add PyO3 binding in `alkahest-py/src/lib.rs`, export in `python/alkahest/__init__.py`, add to `__all__`, write tests.

## Error codes

Every error type has a stable `E-SUBSYSTEM-NNN` code. When adding a new error, follow the existing pattern in the relevant `mod.rs` file and add the code to `docs/` if user-facing.
