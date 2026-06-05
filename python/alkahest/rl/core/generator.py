from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGenerator(ABC):
    """Produces (prompt, metadata) pairs.

    Never stores the reference answer in the returned dict — the verifier must
    re-derive correctness.
    """

    @abstractmethod
    def sample(self, difficulty: int, rng: Any) -> dict:
        """Return a dict with at least ``{"prompt": str, "difficulty": int}``.

        Domain-specific keys (e.g. ``"f_expr"``, ``"is_elementary"``) go here too,
        but never a ``"reference_answer"`` key.
        """
        ...

    def build_dataset(self, n: int, difficulty: int, seed: int = 0):
        """Convenience: build a HuggingFace Dataset from *n* samples."""
        import random

        from datasets import Dataset

        rng = random.Random(seed)
        return Dataset.from_list([self.sample(difficulty, rng) for _ in range(n)])
