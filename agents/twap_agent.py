"""TWAP baseline agent: always executes on pace."""
import numpy as np
from agents.base_agent import BaseAgent
from simulator.data_classes import ExecutionConfig


class TWAPAgent(BaseAgent):
    """
    Time-Weighted Average Price agent.
    Always returns action=2 (on pace), ignoring all context.
    Primary benchmark — if the bandit doesn't beat this, something is wrong.
    """

    @property
    def name(self) -> str:
        return "TWAP"

    def reset(self, config: ExecutionConfig) -> None:
        pass

    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        return 2
