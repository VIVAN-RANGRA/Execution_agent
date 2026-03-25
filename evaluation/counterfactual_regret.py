"""Counterfactual regret tracking: measures how much better/worse an agent does vs TWAP on each episode."""
import numpy as np
from typing import List, Dict, Optional


class CounterfactualRegretTracker:
    """
    After each episode, computes what TWAP would have achieved on the exact same episode.
    Maintains running cumulative regret curve and detects degradation.

    Regret_t = sum_{i=1}^{t} (IS_agent_i - IS_twap_i)
    Positive regret = agent is worse than TWAP overall.
    """

    def __init__(self, regret_threshold: float = 50.0, window_size: int = 20):
        self.regret_threshold = regret_threshold
        self.window_size = window_size
        self.agent_is_history: List[float] = []
        self.twap_is_history: List[float] = []
        self.per_episode_regret: List[float] = []
        self.cumulative_regret: List[float] = []
        self._degradation_detected = False

    def record(self, agent_is_bps: float, twap_is_bps: float) -> None:
        """Record one episode's IS for both agent and TWAP baseline."""
        self.agent_is_history.append(agent_is_bps)
        self.twap_is_history.append(twap_is_bps)
        regret = agent_is_bps - twap_is_bps  # positive = agent is worse
        self.per_episode_regret.append(regret)
        cum = self.cumulative_regret[-1] + regret if self.cumulative_regret else regret
        self.cumulative_regret.append(cum)
        self._check_degradation()

    def _check_degradation(self) -> None:
        """Detect if agent has degraded (recent cumulative regret exceeds threshold)."""
        if len(self.cumulative_regret) < self.window_size:
            return
        recent_regret = sum(self.per_episode_regret[-self.window_size:])
        if recent_regret > self.regret_threshold:
            self._degradation_detected = True

    @property
    def is_degraded(self) -> bool:
        return self._degradation_detected

    def get_recommended_alpha(self, current_alpha: float) -> float:
        """
        If degradation detected, recommend reducing exploration.
        If agent is consistently outperforming, allow slightly more exploration.
        """
        if not self.cumulative_regret:
            return current_alpha

        if self._degradation_detected:
            return max(0.1, current_alpha * 0.5)  # halve exploration

        # If doing well recently, slightly increase exploration
        if len(self.per_episode_regret) >= self.window_size:
            recent = self.per_episode_regret[-self.window_size:]
            if np.mean(recent) < -5.0:  # consistently beating TWAP by 5+ bps
                return min(2.0, current_alpha * 1.1)

        return current_alpha

    def get_summary(self) -> Dict:
        """Return summary statistics."""
        if not self.agent_is_history:
            return {}
        return {
            "n_episodes": len(self.agent_is_history),
            "total_cumulative_regret": self.cumulative_regret[-1] if self.cumulative_regret else 0.0,
            "mean_per_episode_regret": float(np.mean(self.per_episode_regret)),
            "std_per_episode_regret": float(np.std(self.per_episode_regret)),
            "degradation_detected": self._degradation_detected,
            "win_rate_vs_twap": float(np.mean([1 if r < 0 else 0 for r in self.per_episode_regret])),
            "best_episode_advantage": float(min(self.per_episode_regret)) if self.per_episode_regret else 0.0,
            "worst_episode_disadvantage": float(max(self.per_episode_regret)) if self.per_episode_regret else 0.0,
        }

    def get_regret_curve(self) -> np.ndarray:
        """Return cumulative regret as numpy array for plotting."""
        return np.array(self.cumulative_regret)

    def get_rolling_regret(self, window: int = 10) -> np.ndarray:
        """Return rolling mean per-episode regret."""
        if len(self.per_episode_regret) < window:
            return np.array(self.per_episode_regret)
        arr = np.array(self.per_episode_regret)
        return np.convolve(arr, np.ones(window) / window, mode='valid')

    def reset(self) -> None:
        """Reset all tracking state."""
        self.agent_is_history.clear()
        self.twap_is_history.clear()
        self.per_episode_regret.clear()
        self.cumulative_regret.clear()
        self._degradation_detected = False
