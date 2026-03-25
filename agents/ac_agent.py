"""Almgren-Chriss optimal trajectory agent."""
import numpy as np
import json
import yaml
from pathlib import Path
from agents.base_agent import BaseAgent
from simulator.data_classes import ExecutionConfig
from simulator.impact_model import compute_ac_trajectory

BASE_DIR = Path(__file__).resolve().parent.parent
URGENCY_MULTIPLIERS = [0.20, 0.60, 1.00, 1.50, 2.00]


class ACAgent(BaseAgent):
    """
    Almgren-Chriss optimal trajectory agent.
    Computes the IS-minimizing trajectory at reset() and follows it deterministically.
    """

    def __init__(self, config_path: str = None, calibration_path: str = None):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        cal_path = Path(calibration_path) if calibration_path else BASE_DIR / "config" / "calibration_params.json"

        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)
        with open(cal_path) as f:
            self._cal = json.load(f)

        self._trajectory: np.ndarray = np.array([])
        self._config: ExecutionConfig = None

    @property
    def name(self) -> str:
        return "AC_Optimal"

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._trajectory = compute_ac_trajectory(
            total_quantity=config.total_quantity,
            time_horizon_seconds=config.time_horizon_seconds,
            num_slices=config.num_slices,
            risk_aversion=config.risk_aversion,
            sigma_per_second=self._cal["sigma_per_second"],
            eta=self._cal["eta"],
            adv=self._cal["adv_btc"],
        )

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        if len(self._trajectory) == 0 or self._config is None:
            return 2

        idx = min(time_step, len(self._trajectory) - 1)
        target_qty = self._trajectory[idx]

        # Compute uniform pace for this step
        steps_remaining = max(1, total_steps - time_step)
        uniform_qty = inventory / steps_remaining

        if uniform_qty <= 0:
            return 2

        ratio = target_qty / uniform_qty
        diffs = [abs(ratio - m) for m in URGENCY_MULTIPLIERS]
        return int(np.argmin(diffs))
