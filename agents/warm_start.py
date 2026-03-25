"""Shared bandit runtime and warm-start helpers."""
from dataclasses import dataclass
from typing import Any

from config import ProjectConfig


@dataclass(frozen=True)
class BanditRuntimeSettings:
    d_features: int
    warm_start_from_ac: bool
    warm_start_episodes: int


def resolve_bandit_runtime(
    project_config: ProjectConfig,
    agent_section: Any,
    *,
    d_features: int = None,
    warm_start_from_ac: bool = None,
    warm_start_episodes: int = None,
    default_d_features: int = None,
) -> BanditRuntimeSettings:
    """Resolve per-bandit dimensions and warm-start defaults from typed config."""
    configured_d = getattr(agent_section, "d_features", None)
    resolved_d = d_features
    if resolved_d is None:
        resolved_d = configured_d if configured_d is not None else default_d_features
    if resolved_d is None:
        resolved_d = project_config.feature_dimension

    resolved_warm_start = (
        warm_start_from_ac
        if warm_start_from_ac is not None
        else getattr(agent_section, "warm_start_from_ac", True)
    )
    resolved_warm_episodes = (
        warm_start_episodes
        if warm_start_episodes is not None
        else getattr(agent_section, "warm_start_episodes", 20)
    )

    return BanditRuntimeSettings(
        d_features=int(resolved_d),
        warm_start_from_ac=bool(resolved_warm_start),
        warm_start_episodes=max(0, int(resolved_warm_episodes)),
    )
