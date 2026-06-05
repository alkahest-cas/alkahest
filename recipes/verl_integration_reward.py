"""Drop-in reward function for veRL's custom reward interface."""

from alkahest.rl.envs.integration.verifier import IntegrationVerifier

_V = IntegrationVerifier()


def compute_score(solution_str: str, ground_truth: dict) -> float:
    """Score a model solution against integration task metadata."""
    return _V.verify(solution_str, ground_truth)
