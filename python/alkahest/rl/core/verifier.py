from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    """Checks a model completion against task metadata.

    Never receives the reference answer — must re-derive correctness.
    """

    @abstractmethod
    def verify(self, completion: str, metadata: dict) -> float:
        """Return a scalar reward in [-1, 1]."""
        ...
