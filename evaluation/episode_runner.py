"""Runs a single episode for a given agent and environment."""
import numpy as np
from typing import Optional
from simulator.data_classes import EpisodeResult
from evaluation.metrics import compute_episode_metrics


def run_episode(agent, env, episode_seed: Optional[int] = None) -> EpisodeResult:
    """
    Run one full episode.
    Returns EpisodeResult with all metrics computed.
    """
    context, info = env.reset(episode_seed)
    agent.reset(env.current_config)

    while True:
        action = agent.decide(context, env.inventory, env.time_step, env.num_slices)
        next_context, reward, done, step_info = env.step(action)
        agent.update(context, action, reward, next_context)
        context = next_context
        if done:
            break

    return compute_episode_metrics(
        fills=env.fills,
        arrival_price=env.arrival_price,
        agent_name=agent.name,
        episode_seed=episode_seed,
    )
