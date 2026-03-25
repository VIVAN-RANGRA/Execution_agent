"""Hierarchical bandit: a top-level LinUCB that selects which sub-agent to delegate each slice to."""
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from agents.base_agent import BaseAgent
from agents.context_adapter import normalize_context
from agents.twap_agent import TWAPAgent
from agents.ac_agent import ACAgent
from agents.linucb_agent import LinUCBAgent
from agents.thompson_agent import ThompsonAgent
from simulator.data_classes import ExecutionConfig

BASE_DIR = Path(__file__).resolve().parent.parent


class MetaAgent(BaseAgent):
    """
    Hierarchical meta-agent using LinUCB at the top level to select among
    sub-agents (TWAP, AC, LinUCB, Thompson) for each execution slice.

    The meta-level action space is agent SELECTION (K=4), not urgency levels.
    Each sub-agent then independently decides the urgency level (0-4).
    Both the meta-level and chosen sub-agent learn simultaneously.
    """

    N_SUB_AGENTS = 4

    def __init__(
        self,
        config_path: str = None,
        calibration_path: str = None,
        alpha_meta: float = 0.5,
        warm_start_episodes: int = 10,
    ):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)

        self._alpha_meta = alpha_meta
        self._warm_episodes = warm_start_episodes
        self._d = self._cfg["agents"]["linucb"]["d_features"]

        # Sub-agents
        self._sub_agents: List[BaseAgent] = [
            TWAPAgent(),
            ACAgent(config_path=config_path, calibration_path=calibration_path),
            LinUCBAgent(config_path=config_path, calibration_path=calibration_path),
            ThompsonAgent(config_path=config_path),
        ]
        self._sub_agent_names = ["TWAP", "AC_Optimal", "LinUCB", "Thompson"]

        # Meta-level LinUCB state (K=4 actions, one per sub-agent)
        self._A_inv: List[np.ndarray] = [np.eye(self._d) for _ in range(self.N_SUB_AGENTS)]
        self._b: List[np.ndarray] = [np.zeros(self._d) for _ in range(self.N_SUB_AGENTS)]
        self._theta: List[np.ndarray] = [np.zeros(self._d) for _ in range(self.N_SUB_AGENTS)]

        # Per-step tracking for update
        self._last_chosen_k: int = 0
        self._last_context: Optional[np.ndarray] = None

        # Aggregate tracking
        self.sub_agent_selection_counts: np.ndarray = np.zeros(self.N_SUB_AGENTS, dtype=int)
        self.sub_agent_selection_history: List[int] = []
        self._episode_count: int = 0
        self._config: Optional[ExecutionConfig] = None
        self._warmed_up: bool = False

    @property
    def name(self) -> str:
        return "MetaAgent"

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._episode_count += 1
        for agent in self._sub_agents:
            agent.reset(config)

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        x = normalize_context(context, self._d)

        # Meta-level: score each sub-agent using LinUCB formula
        scores = np.zeros(self.N_SUB_AGENTS)
        for k in range(self.N_SUB_AGENTS):
            exploit = self._theta[k] @ x
            explore = self._alpha_meta * np.sqrt(max(0.0, x @ self._A_inv[k] @ x))
            scores[k] = exploit + explore

        # Select best sub-agent
        k_star = int(np.argmax(scores))
        self._last_chosen_k = k_star
        self._last_context = x.copy()

        # Delegate to chosen sub-agent
        action = self._sub_agents[k_star].decide(x, inventory, time_step, total_steps)

        # Track selection
        self.sub_agent_selection_counts[k_star] += 1
        self.sub_agent_selection_history.append(k_star)

        return action

    def update(self, context: np.ndarray, action: int, reward: float,
               next_context: np.ndarray) -> None:
        # Update meta-level with (context, chosen_sub_agent_index, reward)
        x = self._last_context if self._last_context is not None else normalize_context(context, self._d)
        k = self._last_chosen_k

        # Sherman-Morrison rank-1 update for meta-level
        A_inv = self._A_inv[k]
        u = A_inv @ x
        denom = 1.0 + x @ u
        self._A_inv[k] = A_inv - np.outer(u, u) / denom
        self._b[k] = self._b[k] + reward * x
        self._theta[k] = self._A_inv[k] @ self._b[k]

        # Also update the chosen sub-agent so it keeps learning
        self._sub_agents[k].update(
            normalize_context(context, self._d),
            action,
            reward,
            normalize_context(next_context, self._d),
        )

    def warm_start(self, ac_agent, env) -> None:
        """
        Run warm_start_episodes with each sub-agent independently, then use
        their average rewards to initialize meta-level weights.
        """
        if self._warmed_up:
            return

        sub_agent_avg_rewards = np.zeros(self.N_SUB_AGENTS)

        for k, agent in enumerate(self._sub_agents):
            episode_rewards = []
            for ep in range(self._warm_episodes):
                context, info = env.reset(episode_seed=30000 + k * 1000 + ep)
                agent.reset(env.current_config)
                done = False
                ep_reward = 0.0
                step_count = 0
                while not done:
                    x = normalize_context(context, self._d)
                    a = agent.decide(x, env.inventory, env.time_step, env.num_slices)
                    next_context, reward, done, step_info = env.step(a)
                    next_x = normalize_context(next_context, self._d)
                    agent.update(x, a, reward, next_x)

                    # Also feed to meta-level for the corresponding sub-agent
                    A_inv = self._A_inv[k]
                    u_vec = A_inv @ x
                    d = 1.0 + x @ u_vec
                    self._A_inv[k] = A_inv - np.outer(u_vec, u_vec) / d
                    self._b[k] = self._b[k] + reward * x
                    self._theta[k] = self._A_inv[k] @ self._b[k]

                    ep_reward += reward
                    step_count += 1
                    context = next_context

                if step_count > 0:
                    episode_rewards.append(ep_reward / step_count)

            if episode_rewards:
                sub_agent_avg_rewards[k] = float(np.mean(episode_rewards))

        # Warm-start LinUCB and Thompson sub-agents via their own warm_start
        # (they may have already been warmed via the loop above, so skip if so)
        linucb_agent = self._sub_agents[2]
        thompson_agent = self._sub_agents[3]
        if hasattr(linucb_agent, '_warmed_up'):
            linucb_agent._warmed_up = True
        if hasattr(thompson_agent, '_warmed_up'):
            thompson_agent._warmed_up = True

        self._warmed_up = True
