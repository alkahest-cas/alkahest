# alkahest-symbolic-integration

Symbolic integration RL environment backed by Alkahest's Risch integrator.

Models receive an integrand `f(x)` and must return an elementary antiderivative
`F(x)` with `dF/dx = f`, or the exact phrase **"no elementary form"** when Risch
certifies non-elementarity.

## Install

Requires **Python ≥ 3.10**.

```bash
pip install alkahest-symbolic-integration
```

This installs `alkahest`, `verifiers`, and `datasets`.

## Usage

```python
from alkahest.rl.envs.integration.env import load_environment

env = load_environment(
    difficulty_tier=0,
    n_train=50_000,
    n_eval=2_000,
    hard_negative_fraction=0.25,
    adaptive=True,
)

# Prime Intellect CLI
# prime eval run alkahest-cas/symbolic-integration -m <model>
```

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `difficulty_tier` | `0` | Starting Risch grammar tier (0 = rationals) |
| `hard_negative_fraction` | `0.25` | Fraction of NonElementary integrands |
| `n_train` / `n_eval` | `50_000` / `2_000` | Dataset sizes |
| `seed` | `42` | RNG seed |
| `adaptive` | `True` | Attach `CurriculumScheduler` on `env.curriculum` |

## Reward

`IntegrationVerifier` scores completions in `[-1, 1]`:

- **+1.0** — correct antiderivative (symbolic diff check, optional e-graph, interval spot checks)
- **+0.9** — numerically consistent but not symbolically confirmed
- **+1.0** — honest refusal on a NonElementary integrand
- **−0.2** — unnecessary refusal on an elementary integrand
- **−0.5** — hallucinated antiderivative on NonElementary
- **0.0** — unparseable or wrong

## Development

Source code lives in the main [Alkahest](https://github.com/alkahest-cas/alkahest)
repository under `python/alkahest/rl/`. This directory is the **Environments Hub**
publish root.

```bash
cd python/alkahest/rl/envs/integration
pip install -e .
prime env push --auto-bump
```
