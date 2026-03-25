"""Thompson Sampling constrained to +/-1 urgency level of AC's recommendation."""
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
from agents.base_agent import BaseAgent
from agents.ac_agent import ACAgent
from agents.context_adapter import normalize_context
from agents.thompson_agent import ThompsonAgent
from simulator.data_classes import ExecutionConfig

BASE_DIR = Path(__file__).resolve().parent.parent


class ThompsonACHybridAgent(BaseAgent):
    """
    Hybrid agent that combines Thompson Sampling exploration with
    Almgren-Chriss optimal trajectory constraints.

    Thompson samples freely, but the final action is clipped to be within
    +/-1 urgency level of what AC would recommend. This constrains exploration
    to a 'safe zone' around the optimal static schedule, preventing the bandit
    from making dangerously aggressive or passive decisions while still
    allowing meaningful learning.
    """

    def __init__(
        self,
        config_path: str = None,
        calibration_path: str = None,
        prior_variance: float = None,
        warm_start_from_ac: bool = True,
        warm_start_episodes: int = 20,
    ):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)

        # Internal sub-agents
        self._ac_agent = ACAgent(
            config_path=config_path,
            calibration_path=calibration_path,
        )
        self._thompson = ThompsonAgent(
            config_path=config_path,
            prior_variance=prior_variance,
            warm_start_from_ac=warm_start_from_ac,
            warm_start_episodes=warm_start_episodes,
        )

        # Tracking — aggregate
        self.clip_count: int = 0
        self.total_decisions: int = 0
        self.agreement_count: int = 0
        self._episode_count: int = 0
        self._config: Optional[ExecutionConfig] = None
        # Per-episode snapshot for Panel 3 evolution chart
        self.episode_metrics_history: list = []

    @property
    def name(self) -> str:
        return "ThompsonACHybrid"

    @property
    def agreement_rate(self) -> float:
        """Fraction of times Thompson and AC chose the same action."""
        if self.total_decisions == 0:
            return 0.0
        return self.agreement_count / self.total_decisions

    @property
    def clip_rate(self) -> float:
        """Fraction of times Thompson's action was clipped by AC constraint."""
        if self.total_decisions == 0:
            return 0.0
        return self.clip_count / self.total_decisions

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._episode_count += 1
        # Snapshot per-episode metrics before resetting for the next episode
        if self._episode_count > 1:
            self.episode_metrics_history.append({
                "episode": self._episode_count - 1,
                "clip_rate": self.clip_rate,
                "agreement_rate": self.agreement_rate,
            })
        self._ac_agent.reset(config)
        self._thompson.reset(config)

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        x = normalize_context(context, self._thompson._d)

        # Get AC's recommendation (the safe anchor)
        ac_action = self._ac_agent.decide(x, inventory, time_step, total_steps)

        # Get Thompson's recommendation (exploratory)
        ts_action = self._thompson.decide(x, inventory, time_step, total_steps)

        # Clip Thompson to within +/-1 of AC
        lo = max(0, ac_action - 1)
        hi = min(4, ac_action + 1)
        hybrid_action = int(np.clip(ts_action, lo, hi))

        # Track statistics
        self.total_decisions += 1
        if ts_action == ac_action:
            self.agreement_count += 1
        if hybrid_action != ts_action:
            self.clip_count += 1

        return hybrid_action

    def update(self, context: np.ndarray, action: int, reward: float,
               next_context: np.ndarray) -> None:
        # Update the internal Thompson agent so it keeps learning
        # AC agent is deterministic and does not need updates
        self._thompson.update(
            normalize_context(context, self._thompson._d),
            action,
            reward,
            normalize_context(next_context, self._thompson._d),
        )

    def warm_start(self, ac_agent, env) -> None:
        """Delegate warm-start to the internal Thompson agent."""
        self._thompson.warm_start(ac_agent, env)
