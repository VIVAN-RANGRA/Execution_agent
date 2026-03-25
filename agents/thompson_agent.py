"""Thompson Sampling contextual bandit agent (pure NumPy)."""
import numpy as np
from typing import List
from agents.bandit_base import BanditBase
from agents.warm_start import resolve_bandit_runtime
from config import load_config
from simulator.data_classes import ExecutionConfig


class ThompsonAgent(BanditBase):
    """
    Thompson Sampling contextual bandit using Bayesian linear regression posterior.
    State per action: Lambda_a_inv (d x d), m_a (d,).
    Decision: sample theta_tilde ~ N(mu_a, sigma_sq * Lambda_a_inv), pick argmax(theta_tilde^T x).
    Update: Sherman-Morrison rank-1 update on Lambda_a_inv.
    AC warm-start: pre-populates m_a and estimates sigma_sq from AC episode rewards.
    """

    def __init__(
        self,
        config_path: str = None,
        prior_variance: float = None,
        d_features: int = None,
        warm_start_from_ac: bool = None,
        warm_start_episodes: int = None,
    ):
        self._project_config = load_config(config_path)
        runtime = resolve_bandit_runtime(
            self._project_config,
            self._project_config.agents.thompson,
            d_features=d_features,
            warm_start_from_ac=warm_start_from_ac,
            warm_start_episodes=warm_start_episodes,
            default_d_features=self._project_config.agents.thompson.d_features,
        )
        self._prior_variance = (
            prior_variance
            if prior_variance is not None
            else self._project_config.agents.thompson.prior_variance
        )

        # Initialize BanditBase with d_features
        super().__init__(d_features=runtime.d_features, warm_start_settings=runtime)

        self._warm_start = runtime.warm_start_from_ac
        self._warm_episodes = runtime.warm_start_episodes

        prior_precision = 1.0 / self._prior_variance

        # Per-action state
        self._Lambda_inv: List[np.ndarray] = [
            np.eye(self._d) / prior_precision for _ in range(self.N_ACTIONS)
        ]
        self._m: List[np.ndarray] = [np.zeros(self._d) for _ in range(self.N_ACTIONS)]
        self._sigma_sq: float = 1.0

    @property
    def name(self) -> str:
        return "Thompson"

    def reset(self, config: ExecutionConfig) -> None:
        # Delegate to BanditBase for tracking
        super().reset(config)

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        x = self._prepare_context(context)

        # Inventory-aware exploration decay: scale posterior variance
        inv_fraction = inventory / self._config.total_quantity if self._config else 1.0
        time_fraction = max(0, total_steps - time_step) / total_steps if total_steps > 0 else 1.0
        decay_factor = max(0.05, min(inv_fraction, time_fraction) ** 0.5)

        scores = np.zeros(self.N_ACTIONS)
        for a in range(self.N_ACTIONS):
            Lambda_inv = self._Lambda_inv[a]
            mu_a = Lambda_inv @ self._m[a]
            # Cholesky with jitter for numerical stability
            # Scale covariance by decay_factor to reduce exploration as resources deplete
            sigma_sq = max(self._sigma_sq, 1e-8)
            cov = decay_factor * sigma_sq * (Lambda_inv + 1e-8 * np.eye(self._d))
            # Ensure symmetry
            cov = (cov + cov.T) / 2.0
            try:
                L_a = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L_a = np.linalg.cholesky(cov + 1e-6 * np.eye(self._d))
            z = np.random.standard_normal(self._d)
            theta_tilde = mu_a + L_a @ z
            scores[a] = theta_tilde @ x
        return int(np.argmax(scores))

    def _update_model(self, context: np.ndarray, action: int, reward: float) -> None:
        """Thompson-specific Sherman-Morrison rank-1 update on Lambda_inv."""
        x = self._prepare_context(context)
        a = action

        # Sherman-Morrison rank-1 update on Lambda_inv
        Lambda_inv = self._Lambda_inv[a]
        u = Lambda_inv @ x
        denom = 1.0 + x @ u
        self._Lambda_inv[a] = Lambda_inv - np.outer(u, u) / denom

        # Update m_a
        self._m[a] = self._m[a] + reward * x

    def _snapshot_weights(self) -> None:
        """Save current posterior means to weight_history."""
        means = [
            (self._Lambda_inv[a] @ self._m[a]).tolist()
            for a in range(self.N_ACTIONS)
        ]
        self.weight_history.append({
            "episode": self._episode_count,
            "theta": means,
        })

    def get_weights_matrix(self) -> np.ndarray:
        """Return (N_ACTIONS, d_features) weight matrix (posterior means) for visualization."""
        return np.array([
            self._Lambda_inv[a] @ self._m[a]
            for a in range(self.N_ACTIONS)
        ])

    def _post_warm_start(self, all_rewards: List[float]) -> None:
        """Estimate sigma_sq from warm-start reward variance."""
        if len(all_rewards) > 1:
            self._sigma_sq = float(np.var(all_rewards))
            if self._sigma_sq < 1e-8:
                self._sigma_sq = 1.0

    def warm_start(self, ac_agent, env, n_episodes: int = None, seed_offset: int = 20000) -> None:
        """
        Run warm_start_episodes with AC agent. Collect (context, action, reward) triples.
        Estimate sigma_sq from reward variance. Pre-populate m_a via update rule.
        """
        episodes = self._warm_episodes if n_episodes is None else int(n_episodes)
        super().warm_start(ac_agent, env, n_episodes=episodes, seed_offset=seed_offset)
