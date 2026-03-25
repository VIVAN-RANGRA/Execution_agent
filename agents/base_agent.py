"""Abstract base agent interface."""
from abc import ABC, abstractmethod
import numpy as np
from simulator.data_classes import ExecutionConfig


class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def reset(self, config: ExecutionConfig) -> None:
        """Called at the start of each episode."""
        ...

    @abstractmethod
    def decide(self, context: np.ndarray, inventory: float,
               time_step: int, total_steps: int) -> int:
        """Returns urgency level in {0, 1, 2, 3, 4}."""
        ...

    def update(self, context: np.ndarray, action: int, reward: float,
               next_context: np.ndarray) -> None:
        """No-op for non-learning agents. Bandits override this."""
        pass
