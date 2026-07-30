# Alkahest Benchmarks

Two complementary suites cover the full stack: **Rust criterion benchmarks**
for precise CPU timing, and a **Python script** that adds PyO3-boundary
overhead and `tracemalloc` peak-heap measurements.

---

## Rust benchmarks (criterion)

Located at `alkahest-core/benches/alkahest_bench.rs`. Uses
[criterion 0.5](https://docs.rs/criterion) with `html_reports`.

> **Note:** the Cargo package name is `alkahest-cas` (the `-p` flag takes the
> package *name*, not the directory name).

> **Note (Linux):** the bundled `libflint` shared library is embedded in the
> Python wheel.  Running the Rust bench binary directly requires:
> ```bash
> export LD_LIBRARY_PATH="$(python -c \
>   "import alkahest, pathlib; print(next(pathlib.Path(alkahest.__file__).parent.glob('../alkahest.libs')).resolve())"
> ):$LD_LIBRARY_PATH"
> ```
> Or more directly:
> ```bash
> export LD_LIBRARY_PATH=/home/agevorgyan/.local/lib/python3.10/site-packages/alkahest.libs:$LD_LIBRARY_PATH
> ```

### Run

```bash
# Full suite — all groups, HTML report in target/criterion/
cargo bench -p alkahest-cas

# One group only
cargo bench -p alkahest-cas -- simplify

# Smoke pass (correctness check, no timing)
cargo bench -p alkahest-cas -- --test

# Quick pass (3 s per bench instead of 5 s)
cargo bench -p alkahest-cas -- --measurement-time 3
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
cargo bench -p alkahest-cas -- --save-baseline before_change

# Make your change, then compare
cargo bench -p alkahest-cas -- --baseline before_change
```

Criterion will print `Performance has regressed` / `improved` for each benchmark.

### Symbolic GPU harness (CPU baselines + optional GPU comparison)

`alkahest-core/benches/symbolic_gpu_bench.rs` records structured JSON Lines metrics
for candidate GPU kernels. GPU is **not** the default success path — see
[`docs/symbolic-gpu-benchmarks.md`](docs/symbolic-gpu-benchmarks.md) for crossover policy.

```bash
# CPU baseline (CI-friendly)
cargo bench -p alkahest-cas --bench symbolic_gpu_bench -- --nocapture

# Macaulay mod-p reduction + optional GPU timing
ALKAHEST_GPU_BENCH=1 cargo bench -p alkahest-cas --bench symbolic_gpu_bench \
  --features groebner-cuda -- --nocapture
```

Report: `target/symbolic_gpu_bench.jsonl` (override with `SYMBOLIC_GPU_BENCH_OUT`).

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

## Autogenerated local output

Benchmark scripts and agents write local artifacts (JSONL, Markdown reports,
Criterion HTML, ad-hoc logs) under **`temp-alkahest/testing/autogen/`** by
default. That directory is gitignored (via `temp-alkahest/`). Override paths
with each script’s `--output` / `--report` flags.

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
| Mathematica / Wolfram Engine | Install [Wolfram Engine](https://www.wolfram.com/engine/) (free, non-commercial); activate once with `wolframscript -activate`; no extra pip package needed for the primary backend (`wolframscript` subprocess). Optional: `pip install wolframclient` for the socket-session fallback; optional `WOLFRAM_KERNEL` env var. |
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
sudo cargo flamegraph -p alkahest-cas --bench alkahest_bench -- --bench
# Open flamegraph.svg in a browser
```

### Valgrind Massif (heap profile)

```bash
cargo build -p alkahest-cas --profile bench
valgrind --tool=massif --pages-as-heap=yes \
  ./target/release/deps/alkahest_bench-* --bench simplify
ms_print massif.out.* | head -60
```

### perf stat

```bash
perf stat -e cache-misses,cache-references,instructions \
  cargo bench -p alkahest-cas -- simplify 2>&1
```

---

---

## Agent benchmarks (`agent-benchmark/`)

A separate suite that benchmarks **AI agents** solving math problems when
equipped with different CAS skill guides. Where the Rust/Python benchmarks
measure raw library throughput, this measures how well an agent does *with* a
library — and specifically how often the library lets it state a confident wrong
answer.

Full methodology: [`agent-benchmark/README.md`](agent-benchmark/README.md).

### Headline metric: silent-error rate

Raw accuracy saturates. Every mainstream CAS correctly differentiates
`sin(x**2)`, so an accuracy comparison on such problems measures nothing. What
differs is edge behaviour: a library that raises an error gives the agent
something to act on; one that returns a plausible wrong number does not.

The headline number is therefore the share of attempts producing a **confident
but mathematically wrong answer**. Honest refusals ("divergent", "undefined",
"nonelementary") are scored as success on trap tasks, where refusal is the
correct answer.

### Arms

| Arm | Library | Skill file |
|---|---|---|
| `alkahest` | This library, installed from PyPI | `alkahest-skill/alkahest.md` |
| `sympy` | SymPy | `agent-benchmark/skills/sympy.md` |
| `mathematica` | Wolfram Engine via `wolframclient` | `agent-benchmark/skills/mathematica.md` |
| `none` | **Control** — no CAS, stdlib + NumPy | `agent-benchmark/skills/none.md` |

Each arm executes generated code in **its own virtualenv containing only its own
library**, so an arm cannot score points using another arm's CAS. Attempts to do
so are reported as `wrong_library`. The `none` arm is the floor: whatever a CAS
arm scores above it is the value the library adds over plain numerics.

### Task catalogue

19 tasks in five kinds (`agent-benchmark/tasks/catalogue.py`):

| Kind | Count | What it tests |
|---|---|---|
| `control` | 6 | Floor — any working CAS passes |
| `trap` | 7 | A plausible-but-wrong answer is available; refusal is correct |
| `rigor` | 2 | Sound enclosure or extended precision, not a close-looking float |
| `scale` | 3 | Large enough that a slow or recursion-bound implementation fails |
| `certificate` | 1 | Machine-checkable proof; reported separately, excluded from accuracy |

Expected values were verified against both SymPy and alkahest before being
recorded, and the catalogue deliberately includes tasks alkahest fails.

### Run

```bash
pip install -r agent-benchmark/requirements.txt

# One-time: build the isolated per-arm environments (needs network)
python agent-benchmark/run.py --setup-envs

# Anthropic (default model: claude-haiku-4-5-20251001)
ANTHROPIC_API_KEY=sk-... python agent-benchmark/run.py --repeats 5 --temperature 0.7

# Other providers
OPENAI_API_KEY=sk-... python agent-benchmark/run.py --model gpt-4o
GEMINI_API_KEY=...    python agent-benchmark/run.py --model gemini/gemini-2.5-pro

# Subsets
python agent-benchmark/run.py --kinds trap --repeats 5 --temperature 0.7
python agent-benchmark/run.py --max-difficulty 2
python agent-benchmark/run.py --tasks pole_interior_inverse_square --debug

# Preview prompts without calling any API
python agent-benchmark/run.py --dry-run

# Pin the library under test
python agent-benchmark/run.py --setup-envs --alkahest-spec 'alkahest==3.7.0'

# Harness self-test (no API calls, no venvs needed)
python -m pytest agent-benchmark/test_benchmark.py -v
```

### Output

| File | Contents |
|---|---|
| `results/results.jsonl` | One JSON object per run |
| `results/report.md` | Rendered tables |
| `results/provenance.json` | Library versions + `capabilities()`, git SHA, model, skill-file hashes |

Provenance is recorded because version strings alone are not trustworthy: a
local build once reported `3.6.0` while containing part of 3.7.0. Results from a
build that cannot be reconstructed are not usable.

### Metrics

| Metric | Meaning |
|---|---|
| pass@1 | Per-attempt success rate, with a 95% Wilson interval |
| pass@k | Fraction of tasks solved in at least one of k repeats |
| Silent error | Share of checkable attempts that were confidently wrong |
| `exec_ms` | Code execution only — excludes model latency |
| `llm_ms` | Model latency only |
| Prompt / completion tokens | Reported separately; prompt tokens mostly track skill-guide length |

The `alkahest` and `sympy` guides are length- and depth-matched (~1040 vs ~1180
lines, same section coverage), so a measured difference is not a documentation-size
artifact. The `mathematica` guide is not matched — read that arm with the gap in
mind or exclude it.

### Adding a new skill

1. Create `agent-benchmark/skills/<name>.md`.
2. Add a `SkillSpec` to `build_registry()` in `agent-benchmark/envs.py`, listing
   its pip packages and the modules it is allowed to import.
3. Run `python agent-benchmark/run.py --setup-envs --skills <name>`.

### Adding a new task

Add an `AgentTask` to the appropriate list in `agent-benchmark/tasks/catalogue.py`.
Verify the expected value against at least two CAS libraries first, and record
in `rationale` what the task discriminates — `test_benchmark.py` enforces that
every task has one.

---

## Nightly deep run (CI)

The CI nightly job (`.github/workflows/ci.yml`) runs the full proptest suite
with `PROPTEST_CASES=100000` and the hypothesis suite with
`HYPOTHESIS_MAX_EXAMPLES=10000`. To reproduce locally:

```bash
PROPTEST_CASES=100000 cargo test --all --release
HYPOTHESIS_MAX_EXAMPLES=10000 python -m pytest tests/test_properties.py -v
```
