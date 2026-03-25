"""VWAP baseline agent: follows historical volume profile."""
import numpy as np
import pandas as pd
from pathlib import Path
from agents.base_agent import BaseAgent
from simulator.data_classes import ExecutionConfig

BASE_DIR = Path(__file__).resolve().parent.parent
URGENCY_MULTIPLIERS = {0: 0.20, 1: 0.60, 2: 1.00, 3: 1.50, 4: 2.00}


class VWAPAgent(BaseAgent):
    """
    Volume-Weighted Average Price agent.
    Targets execution proportional to the historical volume profile.
    """

    def __init__(self, volume_profile_path: str = None):
        path = Path(volume_profile_path) if volume_profile_path else BASE_DIR / "data" / "volume_profile.parquet"
        try:
            profile = pd.read_parquet(path)
            self._profile = dict(zip(
                profile["minute_of_day"].astype(int),
                profile["fraction"].astype(float)
            ))
        except (FileNotFoundError, Exception):
            self._profile = {i: 1.0 / 1440 for i in range(1440)}

        self._config: ExecutionConfig = None
        self._current_minute: int = 0

    @property
    def name(self) -> str:
        return "VWAP"

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._current_minute = 0

    def set_current_minute(self, minute_of_day: int) -> None:
        self._current_minute = minute_of_day

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        if self._config is None:
            return 2

        # Current minute fraction from volume profile
        current_fraction = self._profile.get(self._current_minute, 1.0 / 1440)

        # Remaining fraction sum from current minute to end of episode
        steps_remaining = max(1, total_steps - time_step)
        remaining_fraction_sum = sum(
            self._profile.get((self._current_minute + i) % 1440, 1.0 / 1440)
            for i in range(steps_remaining)
        )

        if remaining_fraction_sum <= 0:
            return 2

        # Target participation: what fraction of remaining inventory to execute now
        target_fraction = current_fraction / remaining_fraction_sum
        uniform_fraction = 1.0 / steps_remaining

        # Map to urgency: ratio of target to uniform pace
        if uniform_fraction <= 0:
            return 2
        ratio = target_fraction / uniform_fraction

        # Map ratio to nearest urgency level
        return self._ratio_to_urgency(ratio)

    def _ratio_to_urgency(self, ratio: float) -> int:
        """Map participation ratio to nearest urgency level."""
        multipliers = [0.20, 0.60, 1.00, 1.50, 2.00]
        diffs = [abs(ratio - m) for m in multipliers]
        return int(np.argmin(diffs))
