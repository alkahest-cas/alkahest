# Agent benchmark

Measures how well an LLM agent solves mathematical problems when equipped with
different computer algebra systems — and, more importantly, **how often each
library lets the agent state a confident wrong answer**.

## What this measures, and why not accuracy

Raw accuracy is a bad headline metric here. On the problems most benchmarks use
— differentiate `sin(x²)`, integrate `x²`, simplify `sin²+cos²` — every
mainstream CAS is correct, every arm scores near 100%, and the comparison
measures nothing.

What actually differs between libraries, for *agent* use, is behaviour at the
edges: what happens when a problem has no answer, when the antiderivative is not
elementary, when there is a pole inside the interval of integration. A library
that raises an error gives the agent something it can act on. A library that
returns a clean, plausible, wrong number does not — the agent reports it, and
nobody downstream can tell.

So the headline number is the **silent-error rate**: the share of attempts that
produced a confident but mathematically wrong answer.

| Outcome | Meaning | Counts as |
|---|---|---|
| `correct` | Right answer | success |
| `honest_refusal` | Agent said "divergent" / "undefined" / "nonelementary" | success **on trap tasks only** |
| `wrong_answer` | Confident, wrong | **silent error** |
| `no_answer` | Ran, but emitted no parseable `ANSWER:` line | failure |
| `exec_error`, `timeout`, `no_code`, `llm_error` | Mechanical failures | failure |
| `wrong_library` | Tried to import a CAS this arm was not given | failure |

## Design constraints

Four things had to be true before any number here could be shown to anyone:

**1. Arms are isolated.** Each skill gets its own virtualenv containing only its
own library (`envs.py`). An agent given the SymPy guide physically cannot import
alkahest. A static AST check runs first so contamination attempts are reported as
`wrong_library` rather than as a confusing `ImportError`. Without this, every
cell in the results table is uninterpretable.

**2. Results are repeatable and have error bars.** `--repeats N` samples each
cell N times; the report gives pass@1 with a 95% Wilson interval and pass@k.
A single sample per cell cannot distinguish a real difference from noise.

**3. Provenance is captured.** `results/provenance.json` records the exact
library version and `capabilities()` of every arm, the git SHA, the model, the
temperature, and a hash of each skill guide. This matters more than it sounds:
during development a local alkahest build reported version `3.6.0` while
containing some 3.7.0 features and missing others. Numbers from a build nobody
can reconstruct are worthless.

**4. Unavailable arms are excluded, not failed.** The Wolfram arm needs a kernel
most readers do not have. It is reported as unavailable rather than scored as a
wall of failures, which would silently flatter every other arm.

## Task kinds

| Kind | Count | What it tests |
|---|---|---|
| `control` | 6 | Floor. Any working CAS passes. Proves the arms are wired up. |
| `trap` | 7 | A plausible-but-wrong answer is available. Refusal is the correct answer. |
| `rigor` | 2 | Requires a sound enclosure or extended precision, not a close-looking float. |
| `scale` | 3 | Large enough that a slow or recursion-bound implementation fails outright. |
| `certificate` | 1 | Requires emitting a machine-checkable proof. Reported separately. |

Every expected value in `tasks/catalogue.py` was verified against both SymPy and
alkahest before being written down, and every trap and scale task was confirmed
to actually discriminate. Tasks both libraries handle correctly live in
`control`, where they belong.

**The catalogue is not curated to favour alkahest.** `basel_sum` is a task
alkahest currently fails and SymPy passes. All three interior-pole traps are
cases where alkahest returns a confident wrong number and SymPy correctly
reports divergence. A benchmark containing only problems the home library wins
is not evidence, and any reader will spot it.

`certificate` tasks are excluded from headline accuracy: only alkahest can
attempt them, so folding them in would overstate the difference. They are
reported as a capability matrix instead.

## Usage

```bash
pip install -r agent-benchmark/requirements.txt

# One-time: build the isolated per-arm environments (needs network)
python agent-benchmark/run.py --setup-envs

# Run everything
ANTHROPIC_API_KEY=sk-... python agent-benchmark/run.py --repeats 5 --temperature 0.7

# Just the traps
python agent-benchmark/run.py --kinds trap --repeats 5 --temperature 0.7

# A different provider
OPENAI_API_KEY=sk-... python agent-benchmark/run.py --model gpt-4o
GEMINI_API_KEY=...    python agent-benchmark/run.py --model gemini/gemini-2.5-pro

# Preview prompts without spending anything
python agent-benchmark/run.py --dry-run

# Pin the library under test for a reproducible run
python agent-benchmark/run.py --setup-envs --alkahest-spec 'alkahest==3.7.0'
```

Any [LiteLLM](https://docs.litellm.ai/) model string works; set the matching
provider key.

### Arms

| Arm | Library | Skill guide |
|---|---|---|
| `alkahest` | This library, from PyPI | `../alkahest-skill/alkahest.md` |
| `sympy` | SymPy | `skills/sympy.md` |
| `mathematica` | Wolfram Engine via `wolframclient` | `skills/mathematica.md` |
| `none` | **Control:** no CAS, stdlib + NumPy | `skills/none.md` |

The `none` arm is the floor. Whatever a CAS arm scores above it is the value the
library adds over what the model can already do with plain numerics.

### Output

- `results/results.jsonl` — one JSON object per run
- `results/report.md` — rendered tables
- `results/provenance.json` — versions, capabilities, git SHA, skill hashes

## Guide parity

The skill guides are length- and depth-matched, so a measured difference is not
just a documentation-size artifact:

| Guide | Lines | Sections |
|---|---|---|
| `../alkahest-skill/alkahest.md` | ~1040 | 31 |
| `skills/sympy.md` | ~1180 | 24 |
| `skills/mathematica.md` | ~250 | 11 |

Both major guides cover the same ground section-for-section — mental model,
return types, simplification, calculus, solving, polynomials, matrices, ODEs,
transforms, numerics, error handling, performance, and a numbered rules list —
and every code snippet in both was executed before being written down. The SymPy
guide documents that library's genuine strengths (`solveset` domains, adaptive
`evalf`, `lambdify`, the special-function detection recipe) and its real traps,
because a strawman comparison is worthless for the purpose this benchmark exists
to serve.

The Wolfram guide is **not** matched and that arm should be read with the gap in
mind, or excluded.

Prompt-token totals still track guide length, so compare **completion** tokens
for reasoning cost.

## Self-test

`test_benchmark.py` checks the verifiers, code extraction, import policing,
sandbox limits, and report generation without making any API calls:

```bash
python -m pytest agent-benchmark/test_benchmark.py -v
```
