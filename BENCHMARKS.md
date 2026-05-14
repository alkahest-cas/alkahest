# Alkahest Benchmarks

Two complementary suites cover the full stack: **Rust criterion benchmarks**
for precise CPU timing, and a **Python script** that adds PyO3-boundary
overhead and `tracemalloc` peak-heap measurements.

---

## Rust benchmarks (criterion)

Located at `alkahest-core/benches/alkahest_bench.rs`. Uses
[criterion 0.5](https://docs.rs/criterion) with `html_reports`.

### Run

```bash
# Full suite — all groups, HTML report in target/criterion/
cargo bench -p alkahest-core

# One group only
cargo bench -p alkahest-core -- simplify

# Smoke pass (correctness check, no timing)
cargo bench -p alkahest-core -- --test

# Quick pass (3 s per bench instead of 5 s)
cargo bench -p alkahest-core -- --measurement-time 3
```

The HTML report is written to `target/criterion/report/index.html`.

### Benchmark groups

| Group | What it measures |
|---|---|
| `intern` | ExprPool hash-consing throughput — cached symbols, unique integer interning, `build_add3`, structural-sharing verification |
| `simplify` | Simplification engine — `x+0`, constant folding, polynomials degree 1–4, fixpoint-detection on already-simplified expressions |
| `diff` | Symbolic differentiation — polynomials degree 1–4, `sin(x²)`, `exp(sin(x))`, `log(poly)` |
| `unipoly` | FLINT-backed `UniPoly` — `from_symbolic` at degrees 2/4/8, degree-4 multiplication, GCD |
| `multipoly` | Sparse `MultiPoly` — univariate and bivariate `from_symbolic`, bivariate multiplication |
| `memory` | Per-operation heap bytes via a counting `GlobalAlloc`; validates that the pool doesn't grow on a second identical-expression build |
| `log_overhead` | `DerivationLog` step count after `diff` and `simplify` — measures logging cost separate from computation |

### Memory measurement

`alkahest_bench.rs` replaces the default allocator with a counting wrapper
(`CountingAllocator`) that tracks cumulative bytes allocated and number of
`alloc` calls via two `AtomicU64` globals. The `bench_memory` group uses
`iter_custom` to snapshot counters before and after each operation and
passes the deltas through `criterion::black_box` so they appear in
criterion output without being optimised away.

The `memory/hash_consing_second_build` case asserts `pool.len()` is
unchanged after rebuilding an identical expression tree — a regression test
for the intern table's structural-sharing guarantee.

### Comparing baselines

```bash
# Save a baseline
cargo bench -p alkahest-core -- --save-baseline before_change

# Make your change, then compare
cargo bench -p alkahest-core -- --baseline before_change
```

Criterion will print `Performance has regressed` / `improved` for each benchmark.

---

## Python benchmarks

`benchmarks/python_bench.py` measures the full PyO3 call path including
Python object construction, Rust GIL acquisition, and return-value wrapping.

### Dependencies

```bash
pip install hypothesis   # already in dev deps; nothing extra needed
```

### Run

```bash
# Full suite (~30 s)
python benchmarks/python_bench.py

# Quick smoke pass (~2 s)
python benchmarks/python_bench.py --quick
```

### What is measured

Each case reports:

| Column | Meaning |
|---|---|
| `Mean (µs)` | Minimum-of-repeats wall-clock time per iteration in microseconds |
| `Peak (KiB)` | Peak heap allocated by a single call (Python's `tracemalloc`) |
| `Notes` | Operation-specific annotation — step count for `simplify`/`diff`, etc. |

Cases covered:

- **intern** — `symbol()` 100×, `integer()` 100 unique values, hash-consing verify
- **simplify** — `x+0`, const fold, polynomials degree 1–4 (with step counts)
- **diff** — polynomials degree 1–4, `sin(x²)` (with step counts)
- **unipoly** — `from_symbolic` at degree 2/4/8, degree-4 multiplication
- **multipoly** — bivariate `from_symbolic`

### Interpreting results

The `Peak (KiB)` column is the peak of a **single call** measured by
`tracemalloc`. It reflects Python-side allocations; Rust-side heap traffic
from `rug`/FLINT is not visible here (use the Rust `memory` group or
Valgrind for that).

The `steps=N` annotation on `simplify`/`diff` rows gives the number of
`RewriteStep` entries in the returned `DerivedResult.steps` list — a proxy
for derivation-log overhead.

---

## Cross-CAS benchmarks (`benchmarks/cas_comparison.py`)

This driver times the **same symbolic workloads** from `benchmarks/tasks.py`
against **Alkahest** (Rust core + PyO3), **SymPy** (always available if
installed), and — when backends are present — **SageMath**, **Wolfram Engine /
Mathematica** (via `wolframclient`), **SymEngine**, and **Maple**.  Optional
adapters are discovered at runtime; missing systems are skipped and show up as
gaps in the Markdown report.

Each task implements `run_alkahest` and usually `run_sympy`.  Competitor-only
code paths live under `benchmarks/competitors/` as `bench_<task_name>` methods
on `CASAdapter` subclasses.

### Install notes (competitors)

| System | Typical install / env |
|--------|----------------------|
| SymPy | `pip install sympy` (also pulled in by dev deps) |
| SageMath | `pip install sagemath-standard` or system `sage` |
| Mathematica / Wolfram Engine | Install Wolfram Engine; `pip install wolframclient`; optional `WOLFRAM_KERNEL` |
| SymEngine | `pip install symengine` |
| Maple | `maple` on `PATH` |

### Depth / workload controls

Use **`--depth`** to choose how hard each run is: which **problem sizes** are
taken from each task’s `size_params`, optional **extra stress sizes**
(`stress_size_params` on a task, used only for `depth=stress`), and the
**`timeit.repeat` / `timeit.number`** settings.

| Profile | Sizes per task | repeat × number (defaults) |
|---------|----------------|---------------------------|
| `smoke` | smallest only | 1 × 1 |
| `quick` | smallest + largest | 2 × 1 |
| `standard` | full `size_params` | 3 × 1 |
| `deep` | full `size_params` | 5 × 2 |
| `stress` | `size_params` ∪ `stress_size_params` | 7 × 3 |

Override timing only:

```bash
python benchmarks/cas_comparison.py --depth standard --repeat 5 --number 2
```

Override **every** task’s sizes (ignores `--depth` size selection, but keeps timing from `--depth` unless `--repeat` / `--number` are set):

```bash
python benchmarks/cas_comparison.py --sizes 8,16,32
```

### Run

```bash
# After `maturin develop --release` (and `--features groebner` for solve / homotopy tasks)
python benchmarks/cas_comparison.py --depth standard

# SymPy + Alkahest only, one task, smoke depth
python benchmarks/cas_comparison.py --depth smoke --tasks poly_diff --systems alkahest,sympy

# Add Sage, Mathematica, SymEngine, Maple when available (see competitors package)
python benchmarks/cas_comparison.py --depth deep --competitors --systems alkahest,sympy
```

JSONL rows include `depth`, `timeit_repeat`, and `timeit_number` so archived
results stay self-describing.

### Task catalogue (`ALL_TASKS`)

Rough coverage (see `benchmarks/tasks.py` for exact `size_params`):

| Area | Tasks |
|------|--------|
| Calculus | `poly_diff`, `integrate_poly`, `series_expansion`, `limit_computation`, `gradient_nvar` |
| Polynomials | `poly_gcd`, `rational_simplify`, `resultant_poly`, `subresultant_chain`, `factor_univariate_mod_p`, `real_roots_poly`, `horner_form_poly`, `expand_power_simplify` |
| Linear algebra | `jacobian_nxn`, `matrix_det_nxn` |
| Simplification | `trig_identity`, `log_exp_simplify`, `collect_like_terms_mixed` |
| Solvers / decomposition | `solve_circle_line`, `solve_6r_ik`, `numerical_homotopy` |
| Rigorous / fast eval | `ball_sin_cos`, `poly_jit_eval` |
| Interpolation | `sparse_interp_univariate`, `sparse_interp_multivar` |
| Recurrences | `recurrence_solve` |

Some tasks need **optional** Alkahest features (`groebner`, `jit`, …); they
surface as `not_implemented` in the JSONL when the wheel was built without
them.

---

## Profiling beyond benchmarks

### Flame graph (Linux)

```bash
cargo install flamegraph
# Record a 10-second profile of the full bench suite
sudo cargo flamegraph -p alkahest-core --bench alkahest_bench -- --bench
# Open flamegraph.svg in a browser
```

### Valgrind Massif (heap profile)

```bash
cargo build -p alkahest-core --profile bench
valgrind --tool=massif --pages-as-heap=yes \
  ./target/release/deps/alkahest_bench-* --bench simplify
ms_print massif.out.* | head -60
```

### perf stat

```bash
perf stat -e cache-misses,cache-references,instructions \
  cargo bench -p alkahest-core -- simplify 2>&1
```

---

---

## Agent benchmarks (`agent-benchmark/`)

A separate suite that benchmarks **AI agents** solving math problems when equipped
with different CAS skill guides. Where the Rust/Python benchmarks measure raw
library throughput, the agent benchmarks measure skill-driven agent accuracy,
token cost, and task success rate.

### Concept

Each run gives an AI agent (Claude) one of three skill guides:

| Skill | Library | Skill file |
|---|---|---|
| `alkahest` | This library | `alkahest-skill/alkahest.md` |
| `sympy` | SymPy | `agent-benchmark/skills/sympy.md` |
| `mathematica` | Wolfram Engine via `wolframclient` | `agent-benchmark/skills/mathematica.md` |

The agent writes a self-contained Python script, which the harness executes and
checks for correctness. A task is marked correct if the captured `ANSWER:` line
matches the expected value within tolerance.

### Task catalogue

17 tasks across 6 categories (difficulty 1–3):

| Category | Tasks |
|---|---|
| differentiation | `diff_sin_x2`, `diff_poly_leading`, `gradient_sum` |
| integration | `integrate_x2_definite`, `integrate_sin_definite`, `risch_nonelementary` |
| simplification | `trig_identity`, `log_exp_simplify`, `trig_sum_simplify` |
| polynomial | `poly_gcd_eval`, `poly_eval` |
| solving | `solve_circle_line`, `solve_quadratic_count` |
| linear\_algebra | `jacobian_entry`, `matrix_det` |
| numerics | `ball_sin_cos`, `jit_poly_sum` |

### Run

The harness uses [LiteLLM](https://docs.litellm.ai/) so any supported provider
works — set the matching API key and pass a LiteLLM model string.

```bash
# Prerequisites
pip install litellm

# Anthropic (default model: claude-haiku-4-5-20251001)
ANTHROPIC_API_KEY=sk-... python agent-benchmark/run.py

# OpenAI
OPENAI_API_KEY=sk-... python agent-benchmark/run.py --model gpt-4o

# Google Gemini
GEMINI_API_KEY=... python agent-benchmark/run.py --model gemini/gemini-1.5-pro

# Local Ollama (no key needed)
python agent-benchmark/run.py --model ollama/llama3

# Specific skills and difficulty level
python agent-benchmark/run.py --skills alkahest,sympy --difficulty 1

# Preview prompts without calling the API
python agent-benchmark/run.py --dry-run

# Single task, debug mode (prints generated code)
python agent-benchmark/run.py --tasks diff_sin_x2 --debug

# List available tasks or skills
python agent-benchmark/run.py --list-tasks
python agent-benchmark/run.py --list-skills

# Choose model and output paths
python agent-benchmark/run.py --model claude-sonnet-4-6 \
    --output agent-benchmark/results/results.jsonl \
    --report agent-benchmark/results/report.md
```

### Output

`agent-benchmark/results/results.jsonl` — one JSON line per (skill, task) run:

```json
{"skill": "alkahest", "task": "diff_sin_x2", "category": "differentiation",
 "difficulty": 1, "ok": true, "answer_correct": true, "tokens": 512,
 "wall_ms": 3241.0, "model": "claude-haiku-4-5-20251001"}
```

`agent-benchmark/results/report.md` — markdown summary with per-skill accuracy,
full results table, and token usage.

### Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of tasks where `ANSWER:` matches expected |
| `wall_ms` | End-to-end time including API call + code execution |
| Tokens | Total input + output tokens per run (cost proxy) |
| `ok` | Code ran without errors (distinct from answer correctness) |

### Adding a new skill

1. Create `agent-benchmark/skills/<name>.md` following the format of `sympy.md`.
2. Add an entry to `SKILL_PATHS` in `agent-benchmark/harness.py`.
3. Add `run_<name>` methods to `benchmarks/competitors/` if you also want
   the raw timing comparison in the cross-CAS suite.

---

## Nightly deep run (CI)

The CI nightly job (`.github/workflows/ci.yml`) runs the full proptest suite
with `PROPTEST_CASES=100000` and the hypothesis suite with
`HYPOTHESIS_MAX_EXAMPLES=10000`. To reproduce locally:

```bash
PROPTEST_CASES=100000 cargo test --all --release
HYPOTHESIS_MAX_EXAMPLES=10000 python -m pytest tests/test_properties.py -v
```
