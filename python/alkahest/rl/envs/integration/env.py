from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import alkahest as ak
from alkahest.rl.core.curriculum import CurriculumScheduler
from alkahest.rl.envs.integration.grammar import TIERS, random_elementary
from alkahest.rl.envs.integration.verifier import IntegrationVerifier

if TYPE_CHECKING:
    from datasets import Dataset

try:
    import verifiers as vf
except ImportError:  # pragma: no cover - optional dependency
    vf = None  # type: ignore[assignment]

SYSTEM_PROMPT = """\
You are a symbolic integration assistant. Given an integrand f(x), find an \
antiderivative F(x) such that dF/dx = f(x). Write only the antiderivative, no \
constant of integration. If no elementary antiderivative exists, write exactly: \
"no elementary form"."""

N_TIERS = len(TIERS)


def _known_nonelementary_forms(pool: ak.ExprPool, x: ak.Expr) -> list:
    """Curated corpus of integrands with *no* elementary antiderivative.

    SOUNDNESS INVARIANT (directional issue #3): hard-negative ("nonelementary")
    training rows are labelled ONLY from this hand-verified list — never from the
    engine's own ``integrate()`` verdict. The engine's ``NonElementary`` verdict
    (``E-INT-004``) is a *sound-but-incomplete* oracle: it can emit a FALSE
    ``NonElementary`` on an actually-elementary integrand (see bug B2, fixed in
    #219) and, conversely, it gives up with ``E-INT-001`` ("not implemented") on
    genuinely non-elementary integrands it cannot classify. Trusting either code
    to *generate* labels would poison the "honest refusal" reward. Each entry
    below is a classical, textbook-known non-elementary integrand:

      * exp(x^2)        — Gaussian; antiderivative is erf (non-elementary).
      * sin(x)/x        — sine integral Si(x).
      * exp(x)/x        — exponential integral Ei(x).
      * log(log(x+2))   — logarithmic-integral family; genuinely non-elementary
                          even though the engine only reports E-INT-001 for it.

    The reward (see ``verifier.IntegrationVerifier``) credits an honest refusal
    solely on rows carrying this curated ``is_elementary=False`` label, so a
    refusal is rewarded ONLY when the integrand is provably non-elementary.
    """
    return [
        ak.exp(x * x),
        ak.sin(x) / x,
        ak.exp(x) / x,
        ak.log(ak.log(x + pool.integer(2))),
    ]


def load_environment(
    difficulty_tier: int = 0,
    hard_negative_fraction: float = 0.25,
    n_train: int = 50_000,
    n_eval: int = 2_000,
    seed: int = 42,
    adaptive: bool = True,
) -> Any:
    """Prime Intellect ``verifiers``-compatible entry point.

    Args:
        difficulty_tier: Starting Risch tier (0 = rational functions).
        hard_negative_fraction: Fraction of samples certified NonElementary.
        n_train / n_eval: Dataset sizes.
        seed: RNG seed for reproducibility.
        adaptive: Attach a :class:`CurriculumScheduler` on ``env.curriculum``.
    """
    if vf is None:
        msg = (
            "verifiers is not installed. Install with: pip install 'alkahest[rl]' "
            "or pip install verifiers datasets"
        )
        raise ImportError(msg)

    train_ds = build_dataset(difficulty_tier, hard_negative_fraction, n_train, seed)
    eval_ds = build_dataset(difficulty_tier, hard_negative_fraction, n_eval, seed + 1)

    verifier = IntegrationVerifier()

    async def reward_fn(completion, f_str, is_elementary, **_) -> float:
        pool = ak.ExprPool()
        x = pool.symbol("x")
        f_expr = ak.parse(f_str, pool, {"x": x})
        return verifier.verify(
            completion[-1]["content"],
            {"f_expr": f_expr, "is_elementary": is_elementary, "pool": pool},
        )

    rubric = vf.Rubric(funcs=[reward_fn])

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )

    if adaptive:
        env.curriculum = CurriculumScheduler(n_tiers=N_TIERS)

    return env


def build_dataset(tier: int, neg_frac: float, n: int, seed: int) -> Dataset:
    """Build a HuggingFace dataset of integration tasks."""
    from datasets import Dataset

    rng = random.Random(seed)
    rows = [_make_row(tier, rng.random() < neg_frac, rng) for _ in range(n)]
    return Dataset.from_list(rows)


def _make_row(tier: int, nonelementary: bool, rng: random.Random) -> dict:
    pool = ak.ExprPool()
    x = pool.symbol("x")
    if nonelementary:
        # Labels come from the curated corpus, NOT from the engine's verdict.
        f = rng.choice(_known_nonelementary_forms(pool, x))
        return {
            "prompt": [{"role": "user", "content": f"Find ∫ {ak.unicode_str(f)} dx"}],
            "f_str": str(f),
            "is_elementary": False,
            "tier": tier,
        }

    f_antideriv = random_elementary(pool, tier, rng)
    f_result = ak.diff(f_antideriv, x)
    f = ak.simplify(f_result.value).value
    return {
        "prompt": [{"role": "user", "content": f"Find ∫ {ak.unicode_str(f)} dx"}],
        "f_str": str(f),
        "is_elementary": True,
        "tier": tier,
    }
