from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

RubricFn = Callable[..., float]


@dataclass
class Rubric:
    """Framework-agnostic set of reward functions with optional weights."""

    funcs: list[RubricFn]
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [1.0] * len(self.funcs)
        if len(self.weights) != len(self.funcs):
            msg = f"weights length {len(self.weights)} != funcs length {len(self.funcs)}"
            raise ValueError(msg)

    def score(self, **kwargs) -> float:
        """Weighted sum of sync reward function outputs."""
        total = 0.0
        weight_sum = 0.0
        for fn, w in zip(self.funcs, self.weights):
            if w == 0.0:
                continue
            total += w * float(fn(**kwargs))
            weight_sum += w
        if weight_sum == 0.0:
            return 0.0
        return total / weight_sum
