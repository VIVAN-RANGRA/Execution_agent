"""Kernel UCB agent — Nyström-approximated RBF kernel with reduced history."""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from agents.base_agent import BaseAgent
from simulator.data_classes import ExecutionConfig

BASE_DIR = Path(__file__).resolve().parent.parent


class KernelUCBAgent(BaseAgent):
    """
    Kernel Upper Confidence Bound contextual bandit (pure NumPy).

    Two speed improvements over the naive implementation:

    1. Reduced history cap (max_history=150 instead of 500).
       The oldest observations are dropped when the cap is hit, keeping
       the per-action buffers small without meaningful accuracy loss on a
       60-slice episode.

    2. Nyström approximation (Nyström, 2001; Williams & Seeger, 2001).
       Instead of inverting the full n×n kernel matrix K (O(n³)), we select
       m << n random inducing points Z and approximate:

           K_nn ≈ K_nm  K_mm⁻¹  K_nm^T

       The Woodbury matrix identity then lets us solve the kernel ridge
       regression system by inverting an m×m matrix (O(m³)) instead of n×n.
       Decision cost drops from O(n·d) to O(m·d).

       Default: m=30, n≤150  →  ~5× faster kernel vectors, ~125× faster solve.

    API is identical to the original implementation.
    """

    N_ACTIONS = 5
    MIN_OBS_FOR_PREDICTION = 3

    def __init__(
        self,
        config_path: str = None,
        kernel_bandwidth: float = None,
        alpha_explore: float = None,
        max_history: int = 150,       # ← reduced from 500
        n_inducing: int = 30,         # ← Nyström inducing points
        regularization: float = 1e-4,
        warm_start_from_ac: bool = True,
        warm_start_episodes: int = 20,
    ):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)

        kernel_cfg = self._cfg.get("agents", {}).get("kernel_ucb", {})
        self._bandwidth  = kernel_bandwidth  if kernel_bandwidth  is not None else kernel_cfg.get("kernel_bandwidth",   1.0)
        self._alpha      = alpha_explore     if alpha_explore      is not None else kernel_cfg.get("alpha_exploration",  1.0)
        self._max_history = max_history
        self._n_inducing  = n_inducing
        self._lambda_reg  = regularization
        self._warm_start  = warm_start_from_ac
        self._warm_episodes = warm_start_episodes

        # Per-action raw observation buffers
        self._contexts: List[List[np.ndarray]] = [[] for _ in range(self.N_ACTIONS)]
        self._rewards:  List[List[float]]       = [[] for _ in range(self.N_ACTIONS)]

        # Nyström cache per action: (Z, K_mm_inv, c)
        #   Z        — inducing points  (m × d)
        #   K_mm_inv — (K_mm + λI)⁻¹   (m × m)
        #   c        — mean coeff vector (m,)  for μ(x) = k_m(x) · c
        self._nystrom_cache: List[Optional[Tuple]] = [None] * self.N_ACTIONS

        # Tracking
        self.action_counts:  np.ndarray       = np.zeros(self.N_ACTIONS, dtype=int)
        self.action_rewards: List[List[float]] = [[] for _ in range(self.N_ACTIONS)]
        self.weight_history: List[Dict]        = []
        self._episode_count: int               = 0
        self._config: Optional[ExecutionConfig] = None
        self._warmed_up: bool = False

    @property
    def name(self) -> str:
        return "KernelUCB"

    # ──────────────────────────────────────────────────────────────────────────
    # Vectorised kernel helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _rbf_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        RBF kernel matrix between rows of A (n×d) and rows of B (m×d).
        Returns (n×m) matrix.  Uses broadcasting — no Python loops.
        """
        # ||a - b||² = ||a||² + ||b||² - 2 a·b
        sq_A = np.sum(A ** 2, axis=1, keepdims=True)   # n×1
        sq_B = np.sum(B ** 2, axis=1, keepdims=True).T  # 1×m
        sq_dist = sq_A + sq_B - 2.0 * (A @ B.T)         # n×m
        sq_dist = np.maximum(sq_dist, 0.0)               # numerical safety
        return np.exp(-sq_dist / (2.0 * self._bandwidth ** 2))

    def _rbf_vector(self, x: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Kernel between single point x (d,) and inducing matrix Z (m×d) → (m,)."""
        diff = x[None, :] - Z                       # m×d
        sq   = np.sum(diff ** 2, axis=1)             # m,
        return np.exp(-sq / (2.0 * self._bandwidth ** 2))

    # ──────────────────────────────────────────────────────────────────────────
    # Nyström fit / cache
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_nystrom(self, action: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit Nyström kernel ridge regression for *action*.

        Returns
        -------
        Z        : (m × d) inducing points
        K_mm_inv : (m × m) regularised inverse of inducing kernel matrix
        c        : (m,)   coefficient s.t.  μ(x) = k_m(x) · c
        """
        X = np.array(self._contexts[action])   # n × d
        r = np.array(self._rewards[action])    # n,
        n = X.shape[0]
        m = min(self._n_inducing, n)           # never ask for more than we have

        # ── select inducing points (random subset, no replacement) ────────────
        idx = np.random.choice(n, size=m, replace=False)
        Z   = X[idx]                           # m × d

        # ── kernel matrices ───────────────────────────────────────────────────
        K_mm = self._rbf_matrix(Z, Z)                    # m × m
        K_nm = self._rbf_matrix(X, Z)                    # n × m

        K_mm_reg = K_mm + self._lambda_reg * np.eye(m)
        try:
            K_mm_inv = np.linalg.inv(K_mm_reg)           # O(m³)
        except np.linalg.LinAlgError:
            K_mm_inv = np.linalg.pinv(K_mm_reg)

        # ── Woodbury solve: (K_nm K_mm_inv K_nm^T + λI) α = r ────────────────
        # M = K_mm + (1/λ) K_nm^T K_nm   (m × m)
        M     = K_mm + (1.0 / self._lambda_reg) * (K_nm.T @ K_nm)
        M_reg = M + self._lambda_reg * np.eye(m)
        try:
            M_inv = np.linalg.inv(M_reg)                 # O(m³)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M_reg)

        # α = (1/λ)(r − K_nm M_inv K_nm^T r / λ)
        KTr   = K_nm.T @ r                               # m,
        alpha = (r - K_nm @ (M_inv @ KTr) / self._lambda_reg) / self._lambda_reg

        # Compact coeff: μ(x) = k_m(x) · c  where c = K_mm_inv K_nm^T α
        c = K_mm_inv @ (K_nm.T @ alpha)                  # m,

        return Z, K_mm_inv, c

    def _get_nystrom(self, action: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached Nyström decomposition, fitting it if stale."""
        if self._nystrom_cache[action] is None:
            self._nystrom_cache[action] = self._fit_nystrom(action)
        return self._nystrom_cache[action]

    # ──────────────────────────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._episode_count += 1

        # Snapshot Nyström coefficients every 10 episodes for Panel 3
        if self._episode_count % 10 == 0:
            mu_vectors = []
            for a in range(self.N_ACTIONS):
                if len(self._contexts[a]) >= self.MIN_OBS_FOR_PREDICTION:
                    _, _, c = self._get_nystrom(a)
                    mu_vectors.append(c.tolist())
                else:
                    mu_vectors.append([])
            self.weight_history.append({
                "episode": self._episode_count,
                "theta":   mu_vectors,
            })

    def decide(
        self,
        context: np.ndarray,
        inventory: float,
        time_step: int,
        total_steps: int,
    ) -> int:
        x = context.astype(np.float64)
        scores = np.zeros(self.N_ACTIONS)

        for a in range(self.N_ACTIONS):
            n_a = len(self._contexts[a])

            if n_a < self.MIN_OBS_FOR_PREDICTION:
                scores[a] = np.random.standard_normal()   # explore
                continue

            Z, K_mm_inv, c = self._get_nystrom(a)        # O(1) if cached

            # k_m(x): kernel between x and inducing points  O(m·d)
            k_m = self._rbf_vector(x, Z)                  # m,

            # Predicted mean  μ(x) = k_m · c               O(m)
            mu_a = float(k_m @ c)

            # Uncertainty proxy  σ²(x) = 1 − k_m^T K_mm_inv k_m
            variance = max(0.0, 1.0 - float(k_m @ (K_mm_inv @ k_m)))
            sigma_a  = np.sqrt(variance)

            scores[a] = mu_a + self._alpha * sigma_a

        return int(np.argmax(scores))

    def update(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        next_context: np.ndarray,
    ) -> None:
        a = action
        x = context.astype(np.float64)

        self._contexts[a].append(x.copy())
        self._rewards[a].append(reward)

        # Keep buffer bounded  (drop oldest)
        if len(self._contexts[a]) > self._max_history:
            self._contexts[a].pop(0)
            self._rewards[a].pop(0)

        # Invalidate Nyström cache for this action
        self._nystrom_cache[a] = None

        self.action_counts[a] += 1
        self.action_rewards[a].append(reward)

    def warm_start(self, ac_agent, env) -> None:
        """Pre-populate buffers from AC episodes to seed the kernel regression."""
        if self._warmed_up:
            return
        for ep in range(self._warm_episodes):
            context, _ = env.reset(episode_seed=40000 + ep)
            ac_agent.reset(env.current_config)
            done = False
            while not done:
                action = ac_agent.decide(context, env.inventory, env.time_step, env.num_slices)
                next_context, reward, done, _ = env.step(action)
                self.update(context, action, reward, next_context)
                context = next_context
        self._warmed_up = True
