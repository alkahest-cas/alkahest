# Reinforcement learning environments

`alkahest.rl` turns the Alkahest CAS into **verifiable RL environments**: generators
produce tasks, verifiers re-derive correctness (never storing reference answers in
dataset rows), and optional curriculum schedulers advance Risch difficulty tiers.

The package has two layers:

| Layer | Path | Dependencies |
|-------|------|--------------|
| **Core** | `alkahest.rl.core` | Alkahest only — usable from veRL, TRL, OpenRLHF, custom loops |
| **Environments** | `alkahest.rl.envs.*` | Core + optional [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) (Prime Intellect) |

## Install

The RL stack is an **optional extra** on the main PyPI package. It requires **Python
≥ 3.10** because `verifiers` does not support 3.9.

```bash
pip install "alkahest[rl]"
```

This pulls `verifiers` and `datasets`. The integration environment itself ships
inside the `alkahest` wheel — no separate install is required for local use.

From source (development):

```bash
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features egraph
pip install "verifiers>=0.1.5" datasets
```

Build with the `egraph` feature when possible: the integration verifier uses
e-graph simplification as its second verification layer and falls back gracefully
when it is unavailable.

## Quick start — symbolic integration

```python
from alkahest.rl.envs.integration import IntegrationVerifier, load_environment

# Standalone verifier (any trainer)
verifier = IntegrationVerifier()
reward = verifier.verify(
    "x^2",
    {"f_expr": pool_expr, "is_elementary": True, "pool": pool},
)

# Prime Intellect verifiers environment (after pip install "alkahest[rl]")
env = load_environment(
    difficulty_tier=0,
    n_train=1000,
    n_eval=100,
    hard_negative_fraction=0.25,
    adaptive=True,
)
# env.curriculum.record(reward)  # inside your training loop
```

Dataset rows store a parseable **`f_str`** integrand (not live `Expr` objects) so
HuggingFace / Arrow serialization works. The async reward function reconstructs
expressions at scoring time.

### Risch tiers

| Tier | Grammar | Status |
|------|---------|--------|
| 0 | Rational polynomials | Implemented |
| 1 | exp / log towers | Implemented |
| 2 | ℚ(√d) coefficients | Implemented |
| 3 | Rational exponents | Planned |
| 4 | Nested towers | Planned |

Hard negatives (`hard_negative_fraction`) inject integrands certified
**NonElementary** (e.g. `exp(x²)`, `sin(x)/x`) so models learn to refuse honestly.

## Core API

### `BaseGenerator`

Produces `(prompt, metadata)` dicts. **Never** include a `reference_answer` key.

### `BaseVerifier`

```python
def verify(self, completion: str, metadata: dict) -> float:
    """Return reward in [-1, 1]."""
```

### `CurriculumScheduler`

Tracks rolling pass rate at the current tier; advances when `advance_threshold`
(default 0.70) is met over a sliding window (default 256 rewards).

### `Rubric`

Framework-agnostic weighted reward functions (`Rubric.score(**kwargs)`). Prime
Intellect's `vf.Rubric` is used inside `load_environment()` for Hub compatibility.

## veRL adapter

```python
from recipes.verl_integration_reward import compute_score

# ground_truth = row dict from the integration dataset builder
score = compute_score(solution_str, ground_truth)
```

Pass `compute_score` to veRL's `reward_fn` config. The `ground_truth` dict should
include at least `f_str`, `is_elementary`, and fields the verifier expects after
reconstruction (see `IntegrationVerifier.verify`).

## Adding a new domain

1. Create `python/alkahest/rl/envs/<domain>/` mirroring `integration/`.
2. Subclass `BaseGenerator` — emit unsolved tasks + metadata, no reference answer.
3. Subclass `BaseVerifier` — check the model output with Alkahest primitives.
4. Add `load_environment()` returning `vf.SingleTurnEnv` (optional `verifiers` import).
5. Add a Hub `pyproject.toml` + `README.md` if publishing independently.

## Publishing to the Prime Intellect Environments Hub

The integration environment includes Hub metadata at
`python/alkahest/rl/envs/integration/`. See [Hub checklist](#hub-checklist) below.

After install, users run:

```bash
prime env install alkahest/alkahest-symbolic-integration
prime eval run alkahest/alkahest-symbolic-integration -m <model>
```

### Hub checklist

1. **Alkahest on PyPI** — the Hub package depends on `alkahest>=3.5.0` (includes
   `alkahest.rl`). For local development before PyPI catches up, use `maturin develop`
   or a release wheel from GitHub.

2. **Install the Prime CLI** and log in:
   ```bash
   uv tool install prime
   prime login
   ```

3. **Smoke-test locally** from the Hub package directory:
   ```bash
   cd python/alkahest/rl/envs/integration
   pip install -e .                    # pulls verifiers + alkahest + datasets
   python -c "from alkahest.rl.envs.integration.env import load_environment; load_environment(n_train=4, n_eval=2)"
   ```

4. **Push to the Hub** (from the same directory):
   ```bash
   prime env push
   # or: prime env push --team alkahest-cas --auto-bump
   ```
   Hub CI validates the package, `load_environment()` entrypoint, and dependencies.

5. **Run a Hub eval** against a hosted model:
   ```bash
   prime eval run alkahest/alkahest-symbolic-integration -m <model>
   ```

6. **Iterate** — bump `version` in `pyproject.toml` (or use `--auto-bump`) for each
   publish; tag releases in git when verifier tiers or reward logic change.

### Monorepo note

Implementation code lives in the main `alkahest` package; the Hub directory is a
**thin installable manifest** that pins dependencies and declares the
`[tool.verifiers]` entrypoint. This avoids duplicating environment code while still
letting `prime env install` resolve everything from PyPI.

Further reading: [Prime Intellect — Create & Upload Environment](https://docs.primeintellect.ai/tutorials-environments/create),
[Verifiers Environments Hub](https://primeintellect-ai-verifiers.mintlify.app/integrations/environments-hub).
