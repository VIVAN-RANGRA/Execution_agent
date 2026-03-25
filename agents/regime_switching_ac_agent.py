"""Regime-Switching Almgren-Chriss agent: two trajectories, switches based on volatility regime."""
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Optional
from agents.base_agent import BaseAgent
from simulator.data_classes import ExecutionConfig
from simulator.impact_model import compute_ac_trajectory

BASE_DIR = Path(__file__).resolve().parent.parent
URGENCY_MULTIPLIERS = [0.20, 0.60, 1.00, 1.50, 2.00]

# Context feature indices
IDX_VOL_RATIO_SHORT_LONG = 9  # vol_ratio_short_long

# Regime parameters
RISK_AVERSION_CALM = 0.05      # Less aggressive in calm markets
RISK_AVERSION_STRESSED = 0.50  # Much more aggressive when volatility spikes
VOL_RATIO_THRESHOLD = 1.5      # Switch to stressed when vol_ratio exceeds this
HYSTERESIS_STEPS = 3           # Minimum steps to stay in stressed mode


class RegimeSwitchACAgent(BaseAgent):
    """
    Regime-Switching Almgren-Chriss agent.

    Maintains two pre-computed AC trajectories — one for calm markets and one
    for stressed (high-volatility) markets — and switches between them based
    on the observed short/long volatility ratio feature.

    Calm trajectory:    risk_aversion = 0.05 (patient, less urgency)
    Stressed trajectory: risk_aversion = 0.50 (aggressive, front-loads execution)

    Hysteresis: once the agent enters stressed mode, it stays there for at
    least 3 consecutive steps before considering a switch back. This prevents
    rapid oscillation between regimes on noisy volatility estimates.

    This agent does NOT learn — it is a simple, interpretable, adaptive
    benchmark that responds to regime changes without any parameter fitting.
    """

    def __init__(self, config_path: str = None, calibration_path: str = None):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        cal_path = Path(calibration_path) if calibration_path else BASE_DIR / "config" / "calibration_params.json"

        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)
        with open(cal_path) as f:
            self._cal = json.load(f)

        self._config: Optional[ExecutionConfig] = None
        self._trajectory_calm: np.ndarray = np.array([])
        self._trajectory_stressed: np.ndarray = np.array([])

        # Regime state
        self._in_stressed: bool = False
        self._stressed_counter: int = 0  # steps spent in stressed mode

    @property
    def name(self) -> str:
        return "RegimeSwitchAC"

    def _compute_trajectory(self, config: ExecutionConfig, risk_aversion: float) -> np.ndarray:
        """Compute an AC trajectory with a specific risk aversion parameter."""
        return compute_ac_trajectory(
            total_quantity=config.total_quantity,
            time_horizon_seconds=config.time_horizon_seconds,
            num_slices=config.num_slices,
            risk_aversion=risk_aversion,
            sigma_per_second=self._cal["sigma_per_second"],
            eta=self._cal["eta"],
            adv=self._cal["adv_btc"],
        )

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config

        # Recompute both trajectories for this episode's config
        self._trajectory_calm = self._compute_trajectory(config, RISK_AVERSION_CALM)
        self._trajectory_stressed = self._compute_trajectory(config, RISK_AVERSION_STRESSED)

        # Reset regime state at the start of each episode
        self._in_stressed = False
        self._stressed_counter = 0

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        if self._config is None or len(self._trajectory_calm) == 0:
            return 2  # default: on pace

        # --- Regime detection with hysteresis ---
        vol_ratio = context[IDX_VOL_RATIO_SHORT_LONG] if len(context) > IDX_VOL_RATIO_SHORT_LONG else 1.0

        if vol_ratio > VOL_RATIO_THRESHOLD:
            # Volatility spike detected: enter or stay in stressed mode
            self._in_stressed = True
            self._stressed_counter = 0  # reset counter on new spike
        elif self._in_stressed:
            # Currently stressed, but vol_ratio is below threshold
            self._stressed_counter += 1
            if self._stressed_counter >= HYSTERESIS_STEPS:
                # Enough calm steps have passed: switch back to calm
                self._in_stressed = False
                self._stressed_counter = 0
            # else: stay in stressed mode (hysteresis)

        # --- Select trajectory based on current regime ---
        trajectory = self._trajectory_stressed if self._in_stressed else self._trajectory_calm

        # --- Map trajectory lot to urgency level ---
        idx = min(time_step, len(trajectory) - 1)
        target_qty = trajectory[idx]

        # Compute uniform pace for comparison
        steps_remaining = max(1, total_steps - time_step)
        uniform_qty = inventory / steps_remaining

        if uniform_qty <= 0:
            return 2

        ratio = target_qty / uniform_qty
        diffs = [abs(ratio - m) for m in URGENCY_MULTIPLIERS]
        return int(np.argmin(diffs))
