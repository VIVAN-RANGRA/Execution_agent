"""LinUCB contextual bandit agent (pure NumPy)."""
import numpy as np
from typing import List
from agents.bandit_base import BanditBase
from agents.exploration_decay import InventoryAwareExploration
from agents.warm_start import resolve_bandit_runtime
from config import load_config
from simulator.data_classes import ExecutionConfig


class LinUCBAgent(BanditBase):
    """
    Linear Upper Confidence Bound contextual bandit.
    State: A_a_inv (d x d), b_a (d,) per action.
    Update: Sherman-Morrison rank-1 update (O(d^2), never inverts matrix).
    Exploration: alpha_explore * sqrt(x^T A_a_inv x).
    AC warm-start: pre-populates weights from 20 AC episodes.
    Inventory-aware exploration: alpha decays as inventory/time depletes.
    """

    def __init__(
        self,
        config_path: str = None,
        calibration_path: str = None,
        alpha_explore: float = None,
        d_features: int = None,
        warm_start_from_ac: bool = None,
        warm_start_episodes: int = None,
    ):
        self._project_config = load_config(config_path)
        runtime = resolve_bandit_runtime(
            self._project_config,
            self._project_config.agents.linucb,
            d_features=d_features,
            warm_start_from_ac=warm_start_from_ac,
            warm_start_episodes=warm_start_episodes,
            default_d_features=self._project_config.agents.linucb.d_features,
        )
        alpha = (
            alpha_explore
            if alpha_explore is not None
            else self._project_config.agents.linucb.alpha_exploration
        )

        # Initialize BanditBase with d_features
        super().__init__(d_features=runtime.d_features, warm_start_settings=runtime)

        self._alpha = alpha
        self._warm_start = runtime.warm_start_from_ac
        self._warm_episodes = runtime.warm_start_episodes

        # Inventory-aware exploration decay
        self._exploration_decay = InventoryAwareExploration(alpha_0=self._alpha)

        # Per-action state (Sherman-Morrison: maintain A_inv, b, theta directly)
        self._A_inv: List[np.ndarray] = [np.eye(self._d) for _ in range(self.N_ACTIONS)]
        self._b: List[np.ndarray] = [np.zeros(self._d) for _ in range(self.N_ACTIONS)]
        self._theta: List[np.ndarray] = [np.zeros(self._d) for _ in range(self.N_ACTIONS)]

    @property
    def name(self) -> str:
        return "LinUCB"

    def reset(self, config: ExecutionConfig) -> None:
        # Delegate to BanditBase for tracking
        super().reset(config)

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        x = self._prepare_context(context)

        # Compute inventory-aware exploration parameter
        inv_fraction = inventory / self._config.total_quantity if self._config else 1.0
        time_fraction = max(0, total_steps - time_step) / total_steps if total_steps > 0 else 1.0
        current_alpha = self._exploration_decay.get_alpha_with_time(inv_fraction, time_fraction)

        p_values = np.zeros(self.N_ACTIONS)
        for a in range(self.N_ACTIONS):
            theta_a = self._theta[a]
            A_inv = self._A_inv[a]
            exploit = theta_a @ x
            explore = current_alpha * np.sqrt(max(0.0, x @ A_inv @ x))
            p_values[a] = exploit + explore
        return int(np.argmax(p_values))

    def _update_model(self, context: np.ndarray, action: int, reward: float) -> None:
        """LinUCB-specific Sherman-Morrison rank-1 update."""
        x = self._prepare_context(context)
        a = action

        # Sherman-Morrison rank-1 update: A_inv_new = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        A_inv = self._A_inv[a]
        u = A_inv @ x  # d-vector
        denom = 1.0 + x @ u
        self._A_inv[a] = A_inv - np.outer(u, u) / denom

        # Update b and theta
        self._b[a] = self._b[a] + reward * x
        self._theta[a] = self._A_inv[a] @ self._b[a]

    def _snapshot_weights(self) -> None:
        """Save current theta vectors to weight_history."""
        self.weight_history.append({
            "episode": self._episode_count,
            "theta": [t.tolist() for t in self._theta],
        })

    def get_weights_matrix(self) -> np.ndarray:
        """Return (N_ACTIONS, d_features) weight matrix for visualization."""
        return np.array([t.copy() for t in self._theta])

    def warm_start(self, ac_agent, env, n_episodes: int = None, seed_offset: int = 10000) -> None:
        """
        Run warm_start_episodes with AC agent, collect (context, action, reward) triples,
        apply LinUCB updates to pre-populate weight vectors.
        """
        episodes = self._warm_episodes if n_episodes is None else int(n_episodes)
        super().warm_start(ac_agent, env, n_episodes=episodes, seed_offset=seed_offset)
