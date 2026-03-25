"""Percentage of Volume (POV) agent: executes a fixed % of real-time market volume."""
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
from agents.base_agent import BaseAgent
from simulator.data_classes import ExecutionConfig

BASE_DIR = Path(__file__).resolve().parent.parent
URGENCY_MULTIPLIERS = [0.20, 0.60, 1.00, 1.50, 2.00]

# Context feature indices
IDX_VOLUME_PARTICIPATION = 2  # volume_participation_rate


class POVAgent(BaseAgent):
    """
    Percentage of Volume (POV) benchmark agent.

    Industry-standard execution benchmark that targets a fixed participation
    rate relative to real-time market volume. The agent adjusts its urgency
    level based on observed volume: more aggressive when market volume is high,
    more passive when volume is low.

    This agent does NOT learn — it follows a deterministic policy based on
    the current volume participation feature and inventory urgency.
    """

    def __init__(
        self,
        config_path: str = None,
        target_participation_rate: float = 0.10,
    ):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)

        self._target_rate = target_participation_rate
        self._config: Optional[ExecutionConfig] = None

    @property
    def name(self) -> str:
        return "POV"

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        if self._config is None or inventory <= 0:
            return 2  # default: on pace

        # --- Volume-based urgency ---
        # Feature index 2 = volume_participation_rate: ratio of recent volume
        # to expected volume. Values > 1 mean higher-than-expected volume.
        vol_participation = context[IDX_VOLUME_PARTICIPATION] if len(context) > IDX_VOLUME_PARTICIPATION else 1.0

        # Scale the target participation rate by observed market volume.
        # When volume is high, we can be more aggressive (market absorbs more).
        # When volume is low, we should be more passive (less liquidity).
        if vol_participation > 1.0:
            # High volume: scale up participation toward aggressive end
            # Linear interpolation: vol_participation 1.0->2.0 maps to rate 1.0->2.0x target
            scale = 1.0 + (vol_participation - 1.0)
            desired_rate = self._target_rate * min(scale, 3.0)  # cap at 3x target
        elif vol_participation < 0.5:
            # Low volume: scale down to be more passive
            # vol_participation 0.0->0.5 maps to rate 0.3->0.7x target
            scale = 0.3 + (vol_participation / 0.5) * 0.4
            desired_rate = self._target_rate * scale
        else:
            # Normal volume range [0.5, 1.0]: moderate scaling
            scale = 0.7 + (vol_participation - 0.5) * 0.6  # 0.7 -> 1.0
            desired_rate = self._target_rate * scale

        # Map desired participation rate to urgency level.
        # Uniform pace = target_rate mapped to urgency 2 (multiplier 1.0).
        # The ratio of desired_rate to target_rate gives us the multiplier.
        multiplier = desired_rate / max(self._target_rate, 1e-9)
        diffs = [abs(multiplier - m) for m in URGENCY_MULTIPLIERS]
        urgency = int(np.argmin(diffs))

        # --- Inventory urgency adjustment ---
        # If we are behind schedule, increase urgency by 1.
        remaining_fraction = max(total_steps - time_step, 1) / max(total_steps, 1)
        inventory_fraction = inventory / max(self._config.total_quantity, 1e-9)
        if inventory_fraction / remaining_fraction > 1.15:
            urgency = min(urgency + 1, 4)

        return urgency
