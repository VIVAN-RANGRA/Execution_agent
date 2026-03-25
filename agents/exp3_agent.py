"""EXP3 (Exponential-weight algorithm for Exploration and Exploitation) adversarial bandit."""
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from agents.base_agent import BaseAgent
from agents.context_adapter import normalize_context
from simulator.data_classes import ExecutionConfig

BASE_DIR = Path(__file__).resolve().parent.parent


class EXP3Agent(BaseAgent):
    """
    EXP3 adversarial contextual bandit agent (pure NumPy).

    Robust to nonlinear and non-stationary reward-context relationships.
    Maintains context-dependent weight vectors per action, updated with
    importance-weighted exponential updates.

    State per action a:
        weights_a (d,): exponential weight vector in feature space.

    Decision:
        1. Compute context-dependent scores: score_a = weights_a @ x
        2. Apply softmax with temperature
        3. Mix with uniform distribution for exploration
        4. Sample action from mixed distribution

    Update:
        Importance-weighted exponential weight update on the chosen action.

    AC warm-start: pre-populates weights from 20 AC episodes.
    """

    N_ACTIONS = 5

    def __init__(
        self,
        config_path: str = None,
        gamma: float = None,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
        warm_start_from_ac: bool = True,
        warm_start_episodes: int = 20,
    ):
        cfg_path = Path(config_path) if config_path else BASE_DIR / "config" / "default_config.yaml"
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)

        # EXP3-specific config (fall back to defaults if not in config)
        exp3_cfg = self._cfg.get("agents", {}).get("exp3", {})
        self._gamma = gamma if gamma is not None else exp3_cfg.get("gamma", 0.1)
        self._learning_rate = exp3_cfg.get("learning_rate", learning_rate)
        self._temperature = exp3_cfg.get("temperature", temperature)
        self._d = int(exp3_cfg.get("d_features", self._cfg["agents"]["linucb"]["d_features"]))
        self._warm_start = warm_start_from_ac
        self._warm_episodes = warm_start_episodes

        # Per-action weight vectors in feature space
        self._weights: List[np.ndarray] = [np.zeros(self._d) for _ in range(self.N_ACTIONS)]

        # State for importance-weighted update
        self._last_probs: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

        # Tracking
        self.action_counts: np.ndarray = np.zeros(self.N_ACTIONS, dtype=int)
        self.action_rewards: List[List[float]] = [[] for _ in range(self.N_ACTIONS)]
        self.weight_history: List[Dict] = []
        self._episode_count: int = 0

        self._config: Optional[ExecutionConfig] = None
        self._warmed_up: bool = False

    @property
    def name(self) -> str:
        return "EXP3"

    def reset(self, config: ExecutionConfig) -> None:
        self._config = config
        self._episode_count += 1
        self._last_probs = None
        self._last_action = None

        # Snapshot weights every 10 episodes
        if self._episode_count % 10 == 0:
            self.weight_history.append({
                "episode": self._episode_count,
                "weights": [w.tolist() for w in self._weights],
            })

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum()

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        x = normalize_context(context, self._d)

        # Compute context-dependent scores for each action
        scores = np.array([self._weights[a] @ x for a in range(self.N_ACTIONS)])

        # Apply softmax with temperature to get base probabilities
        probs = self._softmax(scores / max(self._temperature, 1e-9))

        # Mix with uniform distribution for guaranteed exploration
        K = self.N_ACTIONS
        mixed_probs = (1.0 - self._gamma) * probs + self._gamma * (1.0 / K)

        # Ensure valid probability distribution (numerical safety)
        mixed_probs = np.maximum(mixed_probs, 1e-9)
        mixed_probs /= mixed_probs.sum()

        # Sample action
        action = int(np.random.choice(K, p=mixed_probs))

        # Store for importance-weighted update
        self._last_probs = mixed_probs.copy()
        self._last_action = action

        return action

    def update(self, context: np.ndarray, action: int, reward: float,
               next_context: np.ndarray) -> None:
        x = normalize_context(context, self._d)
        a = action

        # Get probability of chosen action for importance weighting
        if self._last_probs is not None and self._last_action == a:
            prob_a = self._last_probs[a]
        else:
            # Fallback: recompute (should not normally happen)
            prob_a = 1.0 / self.N_ACTIONS

        # Importance-weighted reward estimate (unbiased)
        prob_a = max(prob_a, 1e-6)  # prevent division by zero
        r_hat = reward / prob_a

        # Exponential weight update for the chosen action only
        self._weights[a] = self._weights[a] + self._learning_rate * r_hat * x

        # Clip weights to prevent overflow in subsequent softmax
        self._weights[a] = np.clip(self._weights[a], -10.0, 10.0)

        # Tracking
        self.action_counts[a] += 1
        self.action_rewards[a].append(reward)

    def warm_start(self, ac_agent, env) -> None:
        """
        Run warm_start_episodes with AC agent, collect (context, action, reward) triples,
        apply EXP3 updates to pre-populate weight vectors.
        """
        if self._warmed_up:
            return
        for ep in range(self._warm_episodes):
            context, info = env.reset(episode_seed=30000 + ep)
            ac_agent.reset(env.current_config)
            done = False
            while not done:
                action = ac_agent.decide(context, env.inventory, env.time_step, env.num_slices)
                next_context, reward, done, step_info = env.step(action)
                # For warm-start, set last_probs to uniform so importance weight = K
                self._last_probs = np.full(self.N_ACTIONS, 1.0 / self.N_ACTIONS)
                self._last_action = action
                self.update(context, action, reward, next_context)
                context = next_context
        self._warmed_up = True
