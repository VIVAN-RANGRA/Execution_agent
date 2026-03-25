"""Abstract base for all bandit agents -- extends BaseAgent with tracking and warm-start."""
from abc import abstractmethod
import numpy as np
from typing import List, Dict, Optional
from agents.base_agent import BaseAgent
from agents.warm_start import BanditRuntimeSettings
from simulator.data_classes import ExecutionConfig


class BanditBase(BaseAgent):
    """
    Shared infrastructure for all bandit agents.

    Provides:
    - Per-action count and reward tracking
    - Weight history snapshots every 10 episodes
    - Generic warm-start from an AC agent
    - Abstract hooks for agent-specific update, snapshot, and weight access
    """

    N_ACTIONS = 5
    MAX_CONTEXT_ABS = 1e6

    def __init__(
        self,
        d_features: int = 12,
        warm_start_settings: Optional[BanditRuntimeSettings] = None,
    ):
        self._d = d_features
        self.action_counts = np.zeros(self.N_ACTIONS, dtype=int)
        self.action_rewards: List[List[float]] = [[] for _ in range(self.N_ACTIONS)]
        self.weight_history: List[Dict] = []
        self._episode_count = 0
        self._config: Optional[ExecutionConfig] = None
        self._warmed_up = False
        self._warm_start_settings = warm_start_settings or BanditRuntimeSettings(
            d_features=d_features,
            warm_start_from_ac=True,
            warm_start_episodes=20,
        )

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._episode_count += 1
        if self._episode_count % 10 == 0:
            self._snapshot_weights()

    def update(self, context: np.ndarray, action: int, reward: float,
               next_context: np.ndarray) -> None:
        safe_context = self._prepare_context(context)
        self._update_model(safe_context, action, reward)
        self.action_counts[action] += 1
        self.action_rewards[action].append(reward)

    def _prepare_context(self, context: np.ndarray) -> np.ndarray:
        """
        Normalize bandit inputs to the configured feature dimension.

        This keeps older 12-d fixtures compatible with newer 18-d agents and
        prevents extreme/invalid values from destabilizing matrix updates.
        """
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=self.MAX_CONTEXT_ABS,
            neginf=-self.MAX_CONTEXT_ABS,
        )
        x = np.clip(x, -self.MAX_CONTEXT_ABS, self.MAX_CONTEXT_ABS)

        if x.shape[0] < self._d:
            x = np.pad(x, (0, self._d - x.shape[0]), mode="constant")
        elif x.shape[0] > self._d:
            x = x[:self._d]

        return x

    @abstractmethod
    def _update_model(self, context: np.ndarray, action: int, reward: float) -> None:
        """Agent-specific weight update."""
        ...

    @abstractmethod
    def _snapshot_weights(self) -> None:
        """Save current weights to weight_history."""
        ...

    @abstractmethod
    def get_weights_matrix(self) -> np.ndarray:
        """Return (N_ACTIONS, d_features) weight matrix for visualization."""
        ...

    def warm_start(self, ac_agent, env, n_episodes: int = 20, seed_offset: int = 10000) -> None:
        """
        Generic warm-start: run AC agent for n_episodes and apply bandit
        updates from the collected (context, action, reward) triples.
        """
        if self._warmed_up or not self._warm_start_settings.warm_start_from_ac:
            return
        n_episodes = int(n_episodes)
        if n_episodes <= 0:
            return
        all_rewards = []
        for ep in range(n_episodes):
            context, info = env.reset(episode_seed=seed_offset + ep)
            ac_agent.reset(env.current_config)
            done = False
            while not done:
                action = ac_agent.decide(context, env.inventory, env.time_step, env.num_slices)
                next_context, reward, done, step_info = env.step(action)
                self.update(context, action, reward, next_context)
                all_rewards.append(reward)
                context = next_context
        self._warmed_up = True
        self._post_warm_start(all_rewards)

    def _post_warm_start(self, all_rewards: List[float]) -> None:
        """Hook for agents that need to do something after warm-start (e.g., Thompson sets sigma_sq)."""
        pass
