from __future__ import annotations


class CurriculumScheduler:
    """Tracks per-tier pass rates and advances difficulty when ready.

    Args:
        n_tiers: Total number of difficulty tiers.
        advance_threshold: Pass rate at current tier to trigger advancement.
        window: Rolling window size for pass rate estimation.
    """

    def __init__(
        self,
        n_tiers: int,
        advance_threshold: float = 0.70,
        window: int = 256,
    ) -> None:
        self.n_tiers = n_tiers
        self.advance_threshold = advance_threshold
        self.window = window
        self._current_tier = 0
        self._history: list[float] = []

    @property
    def current_tier(self) -> int:
        return self._current_tier

    def record(self, reward: float) -> None:
        self._history.append(float(reward > 0))
        if len(self._history) > self.window:
            self._history.pop(0)
        if (
            len(self._history) >= self.window // 2
            and self._pass_rate() >= self.advance_threshold
            and self._current_tier < self.n_tiers - 1
        ):
            self._current_tier += 1
            self._history.clear()

    def _pass_rate(self) -> float:
        return sum(self._history) / len(self._history) if self._history else 0.0
