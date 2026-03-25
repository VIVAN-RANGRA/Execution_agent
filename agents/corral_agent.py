"""Corral algorithm: optimal online model selection across multiple bandit algorithms."""
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from agents.base_agent import BaseAgent
from agents.twap_agent import TWAPAgent
from agents.ac_agent import ACAgent
from agents.context_adapter import normalize_context
from agents.linucb_agent import LinUCBAgent
from agents.thompson_agent import ThompsonAgent
from simulator.data_classes import ExecutionConfig

BASE_DIR = Path(__file__).resolve().parent.parent


class CorralAgent(BaseAgent):
    """
    Corral online model selection (Agarwal et al., 2017) applied to
    execution scheduling.

    Maintains a probability distribution over N base learners and uses
    importance-weighted rewards with log-barrier updates to adaptively
    concentrate mass on the best-performing agent. Per-agent learning
    rates are adjusted using Corral's key innovation: coupling the
    learning rates to the current probability distribution to achieve
    optimal regret bounds.

    Base learners: TWAP, AC, LinUCB, Thompson.
    All learning sub-agents (LinUCB, Thompson) are updated every step,
    regardless of whether they were selected, to ensure they keep improving.
    """

    def __init__(
        self,
        config_path: str = None,
        calibration_path: str = None,
        gamma: float = None,
        n_episodes: int = None,
    ):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)

        num_slices = self._cfg["execution"]["num_slices"]
        total_episodes = n_episodes if n_episodes is not None else self._cfg["evaluation"]["n_episodes"]

        # Base agents
        self._base_agents: List[BaseAgent] = [
            TWAPAgent(),
            ACAgent(config_path=config_path, calibration_path=calibration_path),
            LinUCBAgent(config_path=config_path, calibration_path=calibration_path),
            ThompsonAgent(config_path=config_path),
        ]
        self._base_agent_names = ["TWAP", "AC_Optimal", "LinUCB", "Thompson"]
        self._n_agents = len(self._base_agents)

        # Indices of learning agents that should be updated every step
        self._learning_indices = [2, 3]  # LinUCB, Thompson

        # Smoothing parameter (default: 1/T where T = total decisions)
        T = max(1, num_slices * total_episodes)
        self._gamma = gamma if gamma is not None else 1.0 / T

        # Probability distribution over agents (uniform initialization)
        self._p: np.ndarray = np.ones(self._n_agents) / self._n_agents

        # Per-agent learning rates (initialized to 1/N)
        self._eta: np.ndarray = np.ones(self._n_agents) / self._n_agents

        # Per-step tracking for update
        self._last_chosen_k: int = 0
        self._last_p_k: float = 0.25

        # Aggregate tracking
        self.selection_history: List[int] = []
        self.probability_history: List[Dict] = []
        self.per_agent_rewards: List[List[float]] = [[] for _ in range(self._n_agents)]
        self._episode_count: int = 0
        self._step_count: int = 0
        self._config: Optional[ExecutionConfig] = None
        self._warmed_up: bool = False

    @property
    def name(self) -> str:
        return "Corral"

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._episode_count += 1
        for agent in self._base_agents:
            agent.reset(config)

        # Snapshot probabilities every 10 episodes
        if self._episode_count % 10 == 0:
            self.probability_history.append({
                "episode": self._episode_count,
                "p": self._p.tolist(),
                "eta": self._eta.tolist(),
            })

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        normalized_context = normalize_context(
            context,
            max(getattr(agent, "_d", np.asarray(context).size) for agent in self._base_agents),
        )

        # Sample a base agent according to distribution p
        k = int(np.random.choice(self._n_agents, p=self._p))

        # Get that agent's action
        action = self._base_agents[k].decide(normalized_context, inventory, time_step, total_steps)

        # Store for update
        self._last_chosen_k = k
        self._last_p_k = self._p[k]

        # Track
        self.selection_history.append(k)
        self._step_count += 1

        return action

    def update(self, context: np.ndarray, action: int, reward: float,
               next_context: np.ndarray) -> None:
        normalized_context = normalize_context(
            context,
            max(getattr(agent, "_d", np.asarray(context).size) for agent in self._base_agents),
        )
        normalized_next = normalize_context(
            next_context,
            max(getattr(agent, "_d", np.asarray(next_context).size) for agent in self._base_agents),
        )
        k = self._last_chosen_k
        p_k = self._last_p_k

        # Track per-agent rewards
        self.per_agent_rewards[k].append(reward)

        # --- Importance-weighted reward ---
        # Clip p_k away from zero for numerical stability
        p_k_safe = max(p_k, 1e-8)
        r_hat = reward / p_k_safe

        # --- Log-barrier probability update ---
        for j in range(self._n_agents):
            if j == k:
                loss_hat = -r_hat  # importance-weighted loss (negative reward)
            else:
                loss_hat = 0.0

            # Multiplicative update with log-barrier
            update_val = -self._eta[j] * loss_hat
            # Clip exponent for numerical stability
            update_val = np.clip(update_val, -20.0, 20.0)
            self._p[j] *= np.exp(update_val)

        # Renormalize
        p_sum = np.sum(self._p)
        if p_sum > 0:
            self._p = self._p / p_sum
        else:
            self._p = np.ones(self._n_agents) / self._n_agents

        # --- Update per-agent learning rates (Corral's key innovation) ---
        # rho = harmonic mean scaling: 1 / sum(1/eta_j)
        inv_eta_sum = np.sum(1.0 / np.maximum(self._eta, 1e-12))
        rho = 1.0 / max(inv_eta_sum, 1e-12)

        for j in range(self._n_agents):
            p_j_safe = max(self._p[j], 1e-12)
            self._eta[j] = min(self._eta[j], rho / p_j_safe)

        # --- Clip probabilities and renormalize ---
        self._p = np.clip(self._p, self._gamma, 1.0)
        p_sum = np.sum(self._p)
        if p_sum > 0:
            self._p = self._p / p_sum
        else:
            self._p = np.ones(self._n_agents) / self._n_agents

        # --- Update ALL learning sub-agents ---
        # This ensures LinUCB and Thompson keep improving even when not selected
        for idx in self._learning_indices:
            self._base_agents[idx].update(normalized_context, action, reward, normalized_next)

    def warm_start(self, ac_agent, env) -> None:
        """Warm-start LinUCB and Thompson sub-agents using AC episodes."""
        if self._warmed_up:
            return

        linucb_agent = self._base_agents[2]
        thompson_agent = self._base_agents[3]

        if hasattr(linucb_agent, 'warm_start'):
            linucb_agent.warm_start(ac_agent, env)
        if hasattr(thompson_agent, 'warm_start'):
            thompson_agent.warm_start(ac_agent, env)

        self._warmed_up = True
