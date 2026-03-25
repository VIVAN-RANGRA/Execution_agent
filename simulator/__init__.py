"""Public simulator package surface."""

from simulator.data_classes import EpisodeResult, ExecutionConfig, Fill, MarketState
from simulator.impact_model import compute_ac_trajectory, permanent_impact, temporary_impact

__all__ = [
    "EpisodeResult",
    "ExecutionConfig",
    "Fill",
    "MarketState",
    "compute_ac_trajectory",
    "permanent_impact",
    "temporary_impact",
    "ExecutionEnv",
]


def __getattr__(name):
    if name == "ExecutionEnv":
        from simulator.execution_env import ExecutionEnv

        return ExecutionEnv
    raise AttributeError(f"module 'simulator' has no attribute {name!r}")
