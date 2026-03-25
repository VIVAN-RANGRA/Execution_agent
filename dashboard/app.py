"""Streamlit dashboard entrypoint for the execution engine."""
from __future__ import annotations

import glob
import importlib
import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # Optional dependency for dashboard rendering
    go = None

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents.agent_registry import benchmark_agents, get_agent, learning_agents, list_agents
from config.config_loader import load_config
from dashboard.components.bandit_weights import (
    build_action_distribution,
    build_feature_importance,
    build_learning_curves,
    build_weight_heatmap,
)
from dashboard.components.comparison_table import (
    build_comparison_table,
    build_colour_legend,
    build_violin_plot,
    build_win_rate_matrix,
)
from dashboard.components.context_heatmap import (
    build_context_heatmap,
    build_is_contribution_chart,
    build_urgency_heatmap,
)
from dashboard.components.trajectory_chart import build_trajectory_chart


st.set_page_config(page_title="Execution Replay Studio", layout="wide")

CONFIG_PATH = BASE_DIR / "config" / "default_config.yaml"
CALIBRATION_PATH = BASE_DIR / "config" / "calibration_params.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
PARSED_AGG_DIR = BASE_DIR / "data" / "parsed" / "aggTrades"

METRIC_LABELS = {
    # Core cost metrics
    "mean_IS_bps": "Mean IS (bps)",
    "std_IS_bps": "Std IS (bps)",
    "median_IS_bps": "Median IS (bps)",
    "p25_IS_bps": "25th Pct IS (bps)",
    "p75_IS_bps": "75th Pct IS (bps)",
    "iqr_IS_bps": "IQR IS (bps)",
    "max_IS_bps": "Worst IS (bps)",
    "min_IS_bps": "Best IS (bps)",
    "information_ratio": "Execution Cost Ratio",
    # Market quality metrics
    "mean_vwap_slippage_bps": "Mean VWAP Slippage (bps)",
    "mean_participation_rate": "Mean Participation Rate",
    # Comparison metrics
    "win_rate_vs_TWAP": "Win Rate vs TWAP",
    "win_rate_vs_AC": "Win Rate vs AC",
    "p_value_vs_TWAP": "P-value vs TWAP",
    # Learning metrics
    "total_cumulative_regret": "Total Cumulative Regret",
    "mean_per_episode_regret": "Mean Per-Episode Regret",
}

METRIC_GUIDANCE = {
    # ── Core cost metrics ────────────────────────────────────────────────────
    "mean_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "Implementation Shortfall (IS) is the signed difference between the achieved execution price "
            "and the decision or arrival benchmark price, expressed in basis points. "
            "In this project, IS is computed from execution VWAP relative to arrival price, so it is a cost measure for buy orders."
        ),
        "reading": (
            "Closer to 0 means the agent stayed close to the arrival benchmark. "
            "Positive values mean worse execution cost for buys, while negative values mean the fills ended up below arrival price. "
            "There is no universal 'good' cutoff: the scale depends on asset liquidity, order size, and horizon. "
            "This is the primary cost metric to minimize."
        ),
    },
    "std_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "Standard deviation of IS across episodes. Measures how unpredictable the execution cost is. "
            "An agent can have a good mean IS but still be risky if outcomes swing wildly."
        ),
        "reading": (
            "Lower means outcomes are more consistent from episode to episode. "
            "Higher means the strategy is less predictable. "
            "Read this alongside Mean IS: a slightly worse average cost may still be preferable if variability is much lower."
        ),
    },
    "median_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "The IS of the middle episode when all are sorted. Unlike the mean, it is not pulled by "
            "extreme outlier episodes, so it better represents a typical execution."
        ),
        "reading": (
            "If median IS is much better than mean IS, a few bad episodes are pulling the average up. "
            "If median and mean are close, outcomes are more balanced. "
            "This is often the best single number for 'typical' execution quality."
        ),
    },
    "p25_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "The 25th percentile of IS — 75% of episodes were more expensive than this. "
            "Represents the performance in favorable conditions."
        ),
        "reading": (
            "This tells you what the agent's better-than-usual episodes look like. "
            "It is useful for understanding upside, but it should not be read as a deployment metric on its own."
        ),
    },
    "p75_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "The 75th percentile of IS — 25% of episodes were more expensive than this. "
            "Represents performance in adverse conditions."
        ),
        "reading": (
            "This is a practical downside-risk summary. "
            "Lower P75 means the agent still behaves acceptably in tougher episodes, while a high P75 means the bad tail starts early."
        ),
    },
    "iqr_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "Interquartile Range: P75 minus P25. The spread of the middle 50% of outcomes, "
            "unaffected by extreme outliers at either end."
        ),
        "reading": (
            "Lower IQR means the middle of the distribution is tight and stable. "
            "Higher IQR means outcomes vary a lot even before you get to the most extreme cases. "
            "IQR is often more robust than standard deviation when outliers are present."
        ),
    },
    "max_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "The worst single-episode IS across the entire evaluation run. "
            "Represents the tail risk: what is the most this strategy has ever cost?"
        ),
        "reading": (
            "This is a tail-risk diagnostic, not a typical-case metric. "
            "Compare it with the mean and P75 to see whether poor performance is occasional or systemic. "
            "Always inspect this before trusting a strategy."
        ),
    },
    "min_IS_bps": {
        "ranking": "lower",
        "direction": "Lower is better (more negative = better best case)",
        "meaning": (
            "The best single-episode IS. Represents the upside: "
            "the most favorable execution the agent has achieved."
        ),
        "reading": (
            "This captures the most favorable episode in the sample. "
            "It is useful for understanding upside, but it should not outweigh the mean, median, and downside metrics."
        ),
    },
    "information_ratio": {
        "ranking": "lower",
        "direction": "More negative is better in this project, but this is a custom cost-style ratio",
        "meaning": (
            "This project uses a custom execution ratio equal to mean IS divided by standard deviation of IS. "
            "That is not the textbook Information Ratio used in portfolio management, where higher excess return over a benchmark divided by tracking error is better. "
            "Here, because IS is a cost for buy orders, a more negative value means lower average cost relative to variability."
        ),
        "reading": (
            "Treat this as a secondary summary only. "
            "Use it to compare agents with similar mean IS but different variability, not as a standalone decision rule. "
            "If you want the standard portfolio-style Information Ratio, the metric definition itself would need to change."
        ),
    },
    # ── Market quality metrics ────────────────────────────────────────────────
    "mean_vwap_slippage_bps": {
        "ranking": "abs_zero",
        "direction": "Closer to zero is better",
        "meaning": (
            "Average difference between the agent’s execution VWAP and the market’s VWAP "
            "over the same window. Measures whether the agent traded in line with the market "
            "or paid a premium relative to what other participants got."
        ),
        "reading": (
            "Closer to zero means achieved price stayed close to the market VWAP benchmark. "
            "For buy orders, positive values mean paying above market VWAP and negative values mean paying below it. "
            "This is a benchmark-relative quality check, not a replacement for IS."
        ),
    },
    "mean_participation_rate": {
        "ranking": "context",
        "direction": "Context dependent",
        "meaning": (
            "Fraction of total market volume that the agent’s fills represent, averaged across episodes. "
            "It describes how aggressive the execution was relative to the market's available volume."
        ),
        "reading": (
            "There is no universal good number. "
            "Lower participation usually means gentler execution but more schedule risk, while higher participation usually means more urgency and potentially more impact. "
            "Interpret it together with IS and the order horizon."
        ),
    },
    # ── Comparison metrics ────────────────────────────────────────────────────
    "win_rate_vs_TWAP": {
        "ranking": "higher",
        "direction": "Higher is better",
        "meaning": (
            "Fraction of episodes where this agent’s IS was lower than TWAP’s IS on the same episode. "
            "TWAP (Time-Weighted Average Price) is the simplest possible benchmark: equal slices every minute."
        ),
        "reading": (
            "Above 0.50 means the agent beat TWAP more often than not. "
            "Below 0.50 means it lost to TWAP more often than it won. "
            "Win rate should be read together with mean IS, because a strategy can win often by tiny amounts but lose badly when it fails."
        ),
    },
    "win_rate_vs_AC": {
        "ranking": "higher",
        "direction": "Higher is better",
        "meaning": (
            "Fraction of episodes where this agent’s IS was lower than the project's AC baseline on the same episode. "
            "This is a project-specific head-to-head comparison, not a universal proof that the strategy is globally optimal."
        ),
        "reading": (
            "Above 0.50 means the agent beat the AC baseline more often than not. "
            "Below 0.50 means the AC baseline won more often. "
            "As with TWAP win rate, frequency of wins should be read together with the size of those wins and losses."
        ),
    },
    "p_value_vs_TWAP": {
        "ranking": "context",
        "direction": "Lower p-value = stronger evidence for the specific one-sided alternative used here",
        "meaning": (
            "The p-value from a one-sided Mann-Whitney U test with alternative='less'. "
            "In this code, it tests whether the agent's IS distribution tends to be lower than TWAP's IS distribution, not merely different."
        ),
        "reading": (
            "A value below 0.05 is commonly treated as evidence that the agent is stochastically better than TWAP under this one-sided test. "
            "A large p-value does not prove the two strategies are the same; it only means the sample does not provide strong evidence for improvement. "
            "This is not an effect-size measure, so always read it with mean IS and win rate. "
            "Also note that the code returns 1.0 when there are fewer than two episodes per side."
        ),
    },
    # ── Learning metrics ─────────────────────────────────────────────────────
    "total_cumulative_regret": {
        "ranking": "lower",
        "direction": "Lower is better",
        "meaning": (
            "Sum of (agent IS - TWAP IS) across all episodes. "
            "A positive value means the agent paid more in total than if you had just used TWAP throughout. "
            "Tracks how much the learning process itself cost."
        ),
        "reading": (
            "Near 0 or negative: the agent has more than paid for its exploration cost. "
            "Moderately positive: the agent is still learning — check if it is trending downward over time. "
            "Large positive and flat: the agent is not learning — check feature normalization and reward signal. "
            "Compare this to the number of episodes: total regret / n_episodes gives a per-episode cost of learning."
        ),
    },
    "mean_per_episode_regret": {
        "ranking": "lower",
        "direction": "Lower (negative is ideal) is better",
        "meaning": (
            "Average (agent IS - TWAP IS) per episode. "
            "Positive means the agent costs more than TWAP per episode on average. "
            "Negative means it is cheaper than TWAP on average — the agent has learned to outperform."
        ),
        "reading": (
            "Negative: the agent is adding value over the naive baseline every episode on average. "
            "Near 0: roughly matching TWAP — no learning benefit yet. "
            "Positive: the agent is consistently worse than doing nothing smart — "
            "investigate exploration rate, feature scaling, or reward sign."
        ),
    },
}

AGENT_GUIDE = {
    "TWAP": {
        "category": "Benchmark",
        "what": "Splits the order evenly across time.",
        "how": "Each slice tries to trade a constant amount, regardless of what the market is doing.",
        "good_for": "Simple baseline, calm markets, and checking whether smarter agents actually add value.",
        "watch": "Can be too rigid when liquidity changes a lot during the execution window.",
    },
    "VWAP": {
        "category": "Benchmark",
        "what": "Matches trading pace to the expected market volume curve.",
        "how": "It trades more when the market is expected to be active and less when expected volume is low.",
        "good_for": "Orders that want to stay close to normal market flow.",
        "watch": "Only as good as the volume profile it follows. If the day behaves differently, it can drift.",
    },
    "AC_Optimal": {
        "category": "Benchmark",
        "what": "Static optimal schedule from the Almgren-Chriss execution model.",
        "how": "Balances impact cost against execution risk using calibrated market-impact parameters and risk aversion.",
        "good_for": "A strong finance baseline when you want a principled cost-risk tradeoff.",
        "watch": "It is not reactive. Once the schedule is set, it does not adapt to new market signals.",
    },
    "POV": {
        "category": "Benchmark",
        "what": "Trades as a fixed percentage of observed market volume.",
        "how": "If the market trades more, it trades more. If the market goes quiet, it slows down too.",
        "good_for": "Participation-style execution that stays tied to current market activity.",
        "watch": "Can fall behind badly in quiet markets if the order still has to finish on time.",
    },
    "RegimeSwitchAC": {
        "category": "Benchmark",
        "what": "Switches between safer and more aggressive AC-style schedules.",
        "how": "A simple volatility regime detector picks a risk-on or risk-off schedule during execution.",
        "good_for": "Interpretable adaptive baseline without full online learning.",
        "watch": "Only adapts through regime switching, so it can miss finer-grained opportunities.",
    },
    "LinUCB": {
        "category": "Contextual Bandit",
        "what": "A linear contextual bandit that uses market features plus an optimism bonus.",
        "how": "It estimates which urgency action should work best for the current context, then adds exploration when uncertainty is high.",
        "good_for": "Fast, interpretable adaptive learning with visible feature weights.",
        "watch": "Assumes the reward relationship is mostly linear in the feature vector.",
    },
    "Thompson": {
        "category": "Contextual Bandit",
        "what": "A Bayesian contextual bandit that samples plausible models before acting.",
        "how": "Instead of always taking the optimistic action, it samples a likely reward model and follows that sample.",
        "good_for": "Adaptive behavior with smoother exploration than simple epsilon-style methods.",
        "watch": "Can still be noisy early on when uncertainty is large.",
    },
    "EXP3": {
        "category": "Contextual Bandit",
        "what": "An adversarial-style bandit that reweights actions based on observed reward.",
        "how": "It boosts the probability of actions that work and downweights those that do not, without needing a linear reward model.",
        "good_for": "More robust when the environment is unstable or the simple linear assumption breaks down.",
        "watch": "Less interpretable than linear models because it focuses on action weighting, not feature coefficients.",
    },
    "KernelUCB": {
        "category": "Contextual Bandit",
        "what": "A nonlinear contextual bandit that compares the current state to similar past states.",
        "how": "It replaces a linear score with a kernel similarity model so it can capture more complex feature interactions.",
        "good_for": "Adaptive execution when market-response patterns are nonlinear.",
        "watch": "Usually heavier and less transparent than linear bandits.",
    },
    "MetaAgent": {
        "category": "Ensemble",
        "what": "A top-level bandit that chooses which sub-agent should act.",
        "how": "Instead of directly choosing urgency, it delegates each slice to one of the available underlying agents.",
        "good_for": "Combining multiple styles of execution without committing to one model everywhere.",
        "watch": "Only works well if the sub-agents are meaningfully different and reasonably strong.",
    },
    "ThompsonACHybrid": {
        "category": "Ensemble",
        "what": "A constrained Thompson agent anchored to the AC schedule.",
        "how": "It uses Thompson sampling for adaptivity, but clips decisions so they stay near the AC recommendation.",
        "good_for": "Safer adaptive execution with a strong static baseline as guardrails.",
        "watch": "May leave some upside on the table because the AC anchor limits how far it can explore.",
    },
    "Corral": {
        "category": "Ensemble",
        "what": "An online model-selection wrapper over multiple agents.",
        "how": "It keeps probability weights over sub-agents and reallocates trust toward those performing better over time.",
        "good_for": "Model selection when you do not know in advance which adaptive agent will work best.",
        "watch": "More moving parts, so it is best interpreted as a controller over agents rather than a single policy.",
    },
}

SPLIT_OPTIONS = {
    "Test": "test",
    "Validation": "val",
    "Train": "train",
}

DEFAULT_AGENT_SHORTLIST = ["TWAP", "VWAP", "AC_Optimal", "LinUCB", "Thompson"]

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _agent_kwargs() -> Dict[str, str]:
    return {
        "config_path": str(CONFIG_PATH),
        "calibration_path": str(CALIBRATION_PATH),
    }


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 0.9rem;
            padding: 0.6rem 0.8rem;
        }
        div[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        .help-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.9rem;
            padding: 0.9rem 1rem;
            background: rgba(255, 255, 255, 0.02);
            margin-bottom: 0.9rem;
        }
        .help-card strong {
            display: block;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "engine_bundle": None,
        "single_episode_data": None,
        "single_episode_meta": None,
        "latest_batch_data": None,
        "latest_batch_path": None,
        "runtime_agents": None,
        "pending_engine_load": None,
        "pending_batch_run": None,
        "saved_batch_selection": None,
        "_pending_saved_batch_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_engine_bundle() -> Dict:
    bundle = st.session_state.get("engine_bundle")
    return bundle if isinstance(bundle, dict) else {}


def _loaded_window_metadata(env) -> Dict[str, Optional[str]]:
    dates = list(getattr(env, "_dates", []) or [])
    return {
        "loaded_days": len(dates),
        "loaded_start": dates[0] if dates else None,
        "loaded_end": dates[-1] if dates else None,
    }


def store_engine_bundle(
    split: str,
    cfg,
    env,
    error: Optional[str],
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_days: Optional[int] = None,
) -> None:
    window_meta = _loaded_window_metadata(env) if env is not None else {
        "loaded_days": 0,
        "loaded_start": None,
        "loaded_end": None,
    }
    st.session_state["engine_bundle"] = {
        "split": split,
        "cfg": cfg,
        "env": env,
        "error": error,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "requested_max_days": max_days,
        **window_meta,
    }


def clear_engine_bundle() -> None:
    st.session_state["engine_bundle"] = None
    st.session_state["runtime_agents"] = None


def engine_ready(bundle: Dict) -> bool:
    return bool(bundle) and bundle.get("env") is not None and not bundle.get("error")


def render_plotly(fig) -> None:
    if go is None:
        st.info("Plotly is not installed in this environment, so interactive charts are unavailable.")
        return
    if fig is None:
        st.info("This chart could not be built.")
        return
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


@st.cache_resource(show_spinner=False)
def load_environment(
    split: str = "test",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_days: Optional[int] = None,
):
    """Load only the execution environment for the requested split."""
    try:
        import simulator.execution_env as execution_env_module

        execution_env_module = importlib.reload(execution_env_module)
        ExecutionEnv = execution_env_module.ExecutionEnv
        cfg = load_config()
        kwargs = {"split": split}
        supported_params = inspect.signature(ExecutionEnv.__init__).parameters
        if "start_date" in supported_params:
            kwargs["start_date"] = start_date
        if "end_date" in supported_params:
            kwargs["end_date"] = end_date
        if "max_days" in supported_params:
            kwargs["max_days"] = max_days

        env = ExecutionEnv(**kwargs)
        return cfg, env, None
    except Exception as exc:
        return None, None, str(exc)


@st.cache_data
def load_results_file(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def available_dates_for_split(split: str) -> List[str]:
    cfg = load_config()
    all_dates = sorted(path.stem for path in PARSED_AGG_DIR.glob("*.parquet")) if PARSED_AGG_DIR.exists() else []
    n = len(all_dates)
    if n == 0:
        return []

    train_end = int(n * cfg.data.train_split)
    val_end = train_end + int(n * cfg.data.val_split)
    if split == "train":
        return all_dates[:train_end]
    if split == "val":
        return all_dates[train_end:val_end]
    return all_dates[val_end:]


def build_agents(selected_agent_names: List[str]) -> Dict[str, object]:
    registry = set(list_agents())
    ordered_names = [name for name in selected_agent_names if name in registry]
    return {name: get_agent(name, **_agent_kwargs()) for name in ordered_names}


@st.cache_resource(show_spinner=False)
def preview_learning_agents(agent_names: tuple[str, ...]) -> Dict[str, object]:
    return build_agents(list(agent_names))


def _parse_timestamp_text(raw_value: Optional[str]) -> Optional[datetime]:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00"))
    except ValueError:
        return None


def batch_agent_names(data: Optional[dict]) -> List[str]:
    if not isinstance(data, dict):
        return []

    names = set()
    for key in ("summary", "is_distributions"):
        section = data.get(key)
        if isinstance(section, dict):
            names.update(name for name in section.keys() if isinstance(name, str))

    completed = data.get("completed_agents")
    if isinstance(completed, list):
        names.update(name for name in completed if isinstance(name, str))

    registry_order = list_agents()
    return [name for name in registry_order if name in names] + sorted(names.difference(registry_order))


def batch_agent_count(data: Optional[dict]) -> int:
    return len(batch_agent_names(data))


def _results_timestamp(path: Optional[str], data: Optional[dict]) -> Optional[datetime]:
    timestamp = _parse_timestamp_text((data or {}).get("timestamp"))
    if timestamp is not None:
        return timestamp
    if path:
        try:
            return datetime.strptime(Path(path).stem, "%Y%m%dT%H%M%S")
        except ValueError:
            return None
    return None


def default_results_path(results_files: List[str]) -> Optional[str]:
    if not results_files:
        return None

    def sort_key(path: str):
        data = load_results_file(path)
        timestamp = _results_timestamp(path, data)
        timestamp_key = timestamp.isoformat() if timestamp is not None else ""
        return (batch_agent_count(data), timestamp_key, path)

    return max(results_files, key=sort_key)


def describe_results_option(path: str) -> str:
    data = load_results_file(path)
    timestamp = _results_timestamp(path, data)
    label = timestamp.strftime("Batch %b %d, %Y %H:%M") if timestamp else Path(path).name

    details = []
    agent_count = batch_agent_count(data)
    if agent_count:
        details.append(f"{agent_count} agents")

    n_episodes = (data or {}).get("n_episodes")
    if isinstance(n_episodes, int):
        details.append(f"{n_episodes} eps")

    if details:
        return f"{label} | {' | '.join(details)}"
    return label


def describe_results_source(path: Optional[str], data: Optional[dict]) -> str:
    parts = []
    timestamp = _results_timestamp(path, data)

    if timestamp is not None:
        parts.append(timestamp.strftime("%b %d %H:%M"))

    agent_count = batch_agent_count(data)
    if agent_count:
        parts.append(f"{agent_count} agents")

    n_episodes = (data or {}).get("n_episodes")
    if isinstance(n_episodes, int):
        parts.append(f"{n_episodes} eps")

    split = (data or {}).get("split")
    if isinstance(split, str) and split:
        parts.append(split.title())

    if not parts and path:
        return Path(path).name
    if not parts:
        return "None"
    return " | ".join(parts)


def describe_window(bundle: Dict) -> str:
    if not bundle:
        return "--"
    loaded_days = bundle.get("loaded_days", 0)
    loaded_start = bundle.get("loaded_start")
    loaded_end = bundle.get("loaded_end")
    if loaded_days and loaded_start and loaded_end:
        if loaded_start == loaded_end:
            return f"{loaded_start} | 1 day"
        return f"{loaded_start} -> {loaded_end} | {loaded_days} days"
    return "--"


def horizon_to_seconds(horizon_str: str) -> int:
    mapping = {"15min": 900, "30min": 1800, "1hr": 3600, "2hr": 7200}
    return mapping.get(horizon_str, 3600)


def run_episode_with_runtime(agent, env, episode_seed: int, risk_aversion: float):
    from evaluation.metrics import compute_episode_metrics

    context, _ = env.reset(episode_seed)
    env.current_config.risk_aversion = float(risk_aversion)
    agent.reset(env.current_config)

    while True:
        action = agent.decide(context, env.inventory, env.time_step, env.num_slices)
        next_context, reward, done, _ = env.step(action)
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


def run_single_episode(env, agents_dict, selected_agent_names, seed, order_size, horizon_s, risk_aversion):
    """Run one episode per selected agent and capture trace data for the charts."""
    results = {}
    if not selected_agent_names:
        return results

    env.total_quantity = order_size
    env.time_horizon_seconds = horizon_s
    env.num_slices = 60

    for agent_name in selected_agent_names:
        agent = agents_dict.get(agent_name)
        if agent is None:
            continue

        context_history = []
        action_history = []

        context, _ = env.reset(episode_seed=seed)
        env.current_config.risk_aversion = float(risk_aversion)
        agent.reset(env.current_config)
        context_history.append(context.copy())

        while True:
            action = agent.decide(context, env.inventory, env.time_step, env.num_slices)
            action_history.append(action)
            next_context, reward, done, _ = env.step(action)
            agent.update(context, action, reward, next_context)
            context = next_context
            context_history.append(context.copy())
            if done:
                break

        results[agent_name] = {
            "fills": list(env.fills),
            "arrival_price": env.arrival_price,
            "context_history": context_history,
            "action_history": action_history,
        }

    return results


def run_dashboard_batch(
    env,
    agents_dict,
    selected_agent_names,
    *,
    n_episodes=20,
    random_seed=42,
    risk_aversion=0.1,
):
    """Run batch evaluation sequentially so all agents are safe to compare."""
    from datetime import datetime

    from evaluation.counterfactual_regret import CounterfactualRegretTracker
    from evaluation.metrics import mann_whitney_p_value, win_rate_vs_baseline

    rng = np.random.RandomState(random_seed)
    seeds = rng.randint(0, int(1e6), n_episodes).tolist()

    all_results = {}
    for agent_name in selected_agent_names:
        agent = agents_dict.get(agent_name)
        if agent is None:
            continue
        episode_results = []
        for seed in seeds:
            result = run_episode_with_runtime(
                agent,
                env,
                episode_seed=int(seed),
                risk_aversion=float(risk_aversion),
            )
            episode_results.append(result)
        all_results[agent_name] = episode_results

    summary = {}
    twap_results = all_results.get("TWAP", [])
    twap_is = [r.implementation_shortfall_bps for r in twap_results]
    twap_is_by_seed = {r.episode_seed: r.implementation_shortfall_bps for r in twap_results}
    regret_trackers = {}

    for agent_name, results in all_results.items():
        if not results:
            continue
        is_values = [r.implementation_shortfall_bps for r in results]
        vwap_slip = [r.vwap_slippage_bps for r in results]
        part_rates = [r.participation_rate for r in results]

        wr_vs_twap = win_rate_vs_baseline(is_values, twap_is) if twap_is and agent_name != "TWAP" else 0.5
        p_val = mann_whitney_p_value(is_values, twap_is) if twap_is and agent_name != "TWAP" else 1.0

        summary[agent_name] = {
            "mean_IS_bps": float(np.mean(is_values)),
            "std_IS_bps": float(np.std(is_values)),
            "median_IS_bps": float(np.median(is_values)),
            "mean_vwap_slippage_bps": float(np.mean(vwap_slip)),
            "mean_participation_rate": float(np.mean(part_rates)),
            "win_rate_vs_TWAP": float(wr_vs_twap),
            "p_value_vs_TWAP": float(p_val),
            "n_episodes": len(results),
            "is_values": is_values,
        }

        if agent_name != "TWAP" and twap_is_by_seed:
            tracker = CounterfactualRegretTracker()
            for r in results:
                twap_val = twap_is_by_seed.get(r.episode_seed)
                if twap_val is not None:
                    tracker.record(r.implementation_shortfall_bps, twap_val)
            regret_trackers[agent_name] = tracker

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_episodes": n_episodes,
        "split": "dashboard",
        "random_seed": random_seed,
        "summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "is_values"}
            for k, v in summary.items()
        },
        "is_distributions": {k: v["is_values"] for k, v in summary.items()},
        "seeds": seeds,
        "regret_tracking": {
            agent_name: tracker.get_summary()
            for agent_name, tracker in regret_trackers.items()
        },
        "regret_curves": {
            agent_name: tracker.get_regret_curve().tolist()
            for agent_name, tracker in regret_trackers.items()
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return output, str(out_path), all_results


def _episode_result_payload(result) -> Dict[str, float]:
    return {
        "implementation_shortfall_bps": float(result.implementation_shortfall_bps),
        "vwap_slippage_bps": float(result.vwap_slippage_bps),
        "participation_rate": float(result.participation_rate),
        "episode_seed": int(result.episode_seed),
    }


def build_dashboard_batch_output(
    all_results: Dict[str, List[Dict[str, float]]],
    *,
    n_episodes: int,
    random_seed: int,
    seeds: List[int],
    split: str = "dashboard",
    failed_agents: Optional[Dict[str, str]] = None,
) -> Dict:
    from evaluation.counterfactual_regret import CounterfactualRegretTracker
    from evaluation.metrics import mann_whitney_p_value, win_rate_vs_baseline

    summary = {}
    twap_results = all_results.get("TWAP", [])
    twap_is = [r["implementation_shortfall_bps"] for r in twap_results]
    twap_is_by_seed = {r["episode_seed"]: r["implementation_shortfall_bps"] for r in twap_results}
    regret_trackers = {}

    for agent_name, results in all_results.items():
        if not results:
            continue

        is_values = [r["implementation_shortfall_bps"] for r in results]
        vwap_slip = [r["vwap_slippage_bps"] for r in results]
        part_rates = [r["participation_rate"] for r in results]

        wr_vs_twap = None
        p_val = None
        if agent_name == "TWAP":
            wr_vs_twap = 0.5
            p_val = 1.0
        elif twap_is:
            wr_vs_twap = float(win_rate_vs_baseline(is_values, twap_is))
            p_val = float(mann_whitney_p_value(is_values, twap_is))

        summary[agent_name] = {
            "mean_IS_bps": float(np.mean(is_values)),
            "std_IS_bps": float(np.std(is_values)),
            "median_IS_bps": float(np.median(is_values)),
            "mean_vwap_slippage_bps": float(np.mean(vwap_slip)),
            "mean_participation_rate": float(np.mean(part_rates)),
            "n_episodes": len(results),
            "is_values": is_values,
        }
        if wr_vs_twap is not None:
            summary[agent_name]["win_rate_vs_TWAP"] = wr_vs_twap
        if p_val is not None:
            summary[agent_name]["p_value_vs_TWAP"] = p_val

        if agent_name != "TWAP" and twap_is_by_seed:
            tracker = CounterfactualRegretTracker()
            for r in results:
                twap_val = twap_is_by_seed.get(r["episode_seed"])
                if twap_val is not None:
                    tracker.record(r["implementation_shortfall_bps"], twap_val)
            regret_trackers[agent_name] = tracker

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_episodes": n_episodes,
        "split": split,
        "random_seed": random_seed,
        "summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "is_values"}
            for k, v in summary.items()
        },
        "is_distributions": {k: v["is_values"] for k, v in summary.items()},
        "seeds": seeds,
        "completed_agents": list(all_results.keys()),
        "failed_agents": failed_agents or {},
        "regret_tracking": {
            agent_name: tracker.get_summary()
            for agent_name, tracker in regret_trackers.items()
        },
        "regret_curves": {
            agent_name: tracker.get_regret_curve().tolist()
            for agent_name, tracker in regret_trackers.items()
        },
    }


def save_dashboard_batch_output(output: Dict) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return str(out_path)


def run_agent_batch_job(
    *,
    agent_name: str,
    split: str,
    start_date: Optional[str],
    end_date: Optional[str],
    max_days: Optional[int],
    seeds: List[int],
    risk_aversion: float,
    order_size: float,
    horizon_s: int,
) -> Dict:
    try:
        import simulator.execution_env as execution_env_module

        execution_env_module = importlib.reload(execution_env_module)
        ExecutionEnv = execution_env_module.ExecutionEnv
        env_kwargs = {"split": split}
        supported_params = inspect.signature(ExecutionEnv.__init__).parameters
        if "start_date" in supported_params:
            env_kwargs["start_date"] = start_date
        if "end_date" in supported_params:
            env_kwargs["end_date"] = end_date
        if "max_days" in supported_params:
            env_kwargs["max_days"] = max_days

        env = ExecutionEnv(**env_kwargs)
        env.total_quantity = float(order_size)
        env.time_horizon_seconds = int(horizon_s)
        env.num_slices = 60

        agent = get_agent(agent_name, **_agent_kwargs())
        episodes = []
        for seed in seeds:
            result = run_episode_with_runtime(
                agent,
                env,
                episode_seed=int(seed),
                risk_aversion=float(risk_aversion),
            )
            episodes.append(_episode_result_payload(result))

        return {
            "agent_name": agent_name,
            "episodes": episodes,
            "agent": agent,
            "error": None,
        }
    except Exception as exc:
        return {
            "agent_name": agent_name,
            "episodes": [],
            "agent": None,
            "error": str(exc),
        }


def merge_regret_into_summary(summary: Dict, regret_tracking: Dict) -> Dict:
    merged = {name: dict(values) for name, values in summary.items()}
    for agent_name, reg in (regret_tracking or {}).items():
        merged.setdefault(agent_name, {})
        merged[agent_name]["total_cumulative_regret"] = reg.get("total_cumulative_regret")
        merged[agent_name]["mean_per_episode_regret"] = reg.get("mean_per_episode_regret")
        merged[agent_name]["degradation_detected"] = reg.get("degradation_detected")
    return merged


def metric_options(summary: Dict) -> List[str]:
    available = []
    for key in METRIC_LABELS:
        if any(isinstance(row.get(key), (int, float, np.number)) for row in summary.values()):
            available.append(key)
    return available


def build_metric_bar_chart(summary: Dict, metric_key: str) -> go.Figure:
    if go is None:
        return None

    agents = []
    values = []
    for agent_name, row in summary.items():
        value = row.get(metric_key)
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float, np.number)):
            agents.append(agent_name)
            values.append(float(value))

    if not agents:
        return go.Figure()

    fig = go.Figure(go.Bar(x=agents, y=values, marker_color="steelblue"))
    fig.update_layout(
        title=METRIC_LABELS.get(metric_key, metric_key),
        xaxis_title="Agent",
        yaxis_title=METRIC_LABELS.get(metric_key, metric_key),
        height=360,
    )
    return fig


def _format_metric_value(metric_key: str, value: Optional[float]) -> str:
    if value is None:
        return "--"
    if metric_key == "p_value_vs_TWAP":
        return f"{float(value):.4f}"
    if metric_key in {"win_rate_vs_TWAP", "win_rate_vs_AC", "mean_participation_rate"}:
        return f"{float(value):.3f}"
    return f"{float(value):.2f}"


def metric_snapshot(summary: Dict, metric_key: str) -> Optional[Dict[str, tuple[str, float]]]:
    ranking = METRIC_GUIDANCE.get(metric_key, {}).get("ranking")
    if ranking not in {"lower", "higher", "abs_zero"}:
        return None

    values = []
    for agent_name, row in summary.items():
        value = row.get(metric_key)
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float, np.number)):
            values.append((agent_name, float(value)))

    if not values:
        return None

    if ranking == "lower":
        best = min(values, key=lambda item: item[1])
        watch = max(values, key=lambda item: item[1])
    elif ranking == "higher":
        best = max(values, key=lambda item: item[1])
        watch = min(values, key=lambda item: item[1])
    else:
        best = min(values, key=lambda item: abs(item[1]))
        watch = max(values, key=lambda item: abs(item[1]))

    return {"best": best, "watch": watch}


def render_metric_guide(metric_key: str, summary: Dict) -> None:
    guide = METRIC_GUIDANCE.get(metric_key)
    if not guide:
        return

    st.subheader("How To Read This Metric")
    expl_col1, expl_col2, expl_col3 = st.columns(3)
    expl_col1.markdown(
        f"""
        <div class="help-card">
            <strong>Direction</strong>
            <div>{guide["direction"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    expl_col2.markdown(
        f"""
        <div class="help-card">
            <strong>What it means</strong>
            <div>{guide["meaning"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    expl_col3.markdown(
        f"""
        <div class="help-card">
            <strong>Rule of thumb</strong>
            <div>{guide["reading"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    snapshot = metric_snapshot(summary, metric_key)
    if snapshot:
        best_agent, best_value = snapshot["best"]
        watch_agent, watch_value = snapshot["watch"]
        st.caption(
            f"Best in this batch: {best_agent} ({_format_metric_value(metric_key, best_value)}) | "
            f"Needs attention: {watch_agent} ({_format_metric_value(metric_key, watch_value)})"
        )


_METRIC_CATEGORIES = {
    "Core Cost": ["mean_IS_bps", "std_IS_bps", "median_IS_bps", "p25_IS_bps", "p75_IS_bps", "iqr_IS_bps", "max_IS_bps", "min_IS_bps", "information_ratio"],
    "Market Quality": ["mean_vwap_slippage_bps", "mean_participation_rate"],
    "Vs Baselines": ["win_rate_vs_TWAP", "win_rate_vs_AC", "p_value_vs_TWAP"],
    "Learning": ["total_cumulative_regret", "mean_per_episode_regret"],
}


def render_metric_dictionary(metric_keys: List[str]) -> None:
    with st.expander("Metric Guide — click to understand what every number means", expanded=False):
        st.markdown(
            """
            **Quick checklist — read in this order:**

            | # | Metric | Target |
            |---|---|---|
            | 1 | Mean IS (bps) | As low as possible. Near 0 = excellent. |
            | 2 | Win Rate vs TWAP | > 0.50 (beat the naive baseline). |
            | 3 | Std IS / IQR IS | Low = consistent, high = unpredictable. |
            | 4 | Information Ratio | More negative = better risk-adjusted cost. |
            | 5 | P-value vs TWAP | < 0.05 = edge is statistically real. |
            | 6 | Worst IS (Max IS) | Tail-risk check before going live. |
            | 7 | Win Rate vs AC | Hardest bar — beats the principled baseline. |

            > **IS in plain English:** If you buy 10 BTC at \\$50,000 arrival price and fill at \\$50,025, IS = +5 bps (costs \\$2,500 extra). Negative IS means you filled *below* arrival — a timing win.
            """
        )

        # Tabs by category — only show tabs that have at least one available metric
        available_categories = {
            cat: [k for k in keys if k in metric_keys and METRIC_GUIDANCE.get(k)]
            for cat, keys in _METRIC_CATEGORIES.items()
        }
        tab_names = [cat for cat, keys in available_categories.items() if keys]
        if not tab_names:
            return

        tabs = st.tabs(tab_names)
        for tab, cat_name in zip(tabs, tab_names):
            with tab:
                for key in available_categories[cat_name]:
                    guide = METRIC_GUIDANCE[key]
                    st.markdown(
                        f"""
                        <div class="help-card">
                            <strong>{METRIC_LABELS.get(key, key)}</strong>
                            &nbsp;<span style="font-size:0.8em;color:#888">{guide["direction"]}</span>
                            <div style="margin-top:4px">{guide["meaning"]}</div>
                            <div style="margin-top:4px;font-style:italic">{guide["reading"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def render_agent_guide() -> None:
    st.header("Agent Guide")
    st.caption("Use this view to understand what each agent is doing before you compare the charts.")

    guide_agents = [name for name in list_agents() if name in AGENT_GUIDE]
    category_filters = ["All", "Benchmark", "Contextual Bandit", "Ensemble"]
    selected_category = st.selectbox(
        "Filter by category",
        options=category_filters,
        index=0,
        help="Benchmarks are fixed baselines, contextual bandits learn from state, and ensembles choose among underlying agents.",
    )

    filtered_agents = [
        name for name in guide_agents
        if selected_category == "All" or AGENT_GUIDE[name]["category"] == selected_category
    ]

    if filtered_agents:
        selected_agent = st.selectbox(
            "Choose an agent",
            options=filtered_agents,
            help="Pick an agent to see a plain-language description of what it does and when it is useful.",
        )
        guide = AGENT_GUIDE[selected_agent]

        top_col1, top_col2 = st.columns([1, 3])
        top_col1.metric("Category", guide["category"])
        top_col2.markdown(
            f"""
            <div class="help-card">
                <strong>{selected_agent}</strong>
                <div><strong>What it is:</strong> {guide["what"]}</div>
                <div><strong>How it works:</strong> {guide["how"]}</div>
                <div><strong>Good for:</strong> {guide["good_for"]}</div>
                <div><strong>Watch out for:</strong> {guide["watch"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("All Agents At A Glance")
    category_order = ["Benchmark", "Contextual Bandit", "Ensemble"]
    available_cats = [c for c in category_order if any(AGENT_GUIDE[n]["category"] == c for n in guide_agents)]
    if available_cats:
        cat_tabs = st.tabs(available_cats)
        for cat_tab, category in zip(cat_tabs, available_cats):
            with cat_tab:
                category_agents = [n for n in guide_agents if AGENT_GUIDE[n]["category"] == category]
                cols = st.columns(2)
                for idx, agent_name in enumerate(category_agents):
                    guide = AGENT_GUIDE[agent_name]
                    cols[idx % 2].markdown(
                        f"""
                        <div class="help-card">
                            <strong>{agent_name}</strong>
                            <div style="margin-top:4px">{guide["what"]}</div>
                            <div style="margin-top:4px"><strong>Works by:</strong> {guide["how"]}</div>
                            <div style="margin-top:4px;color:#888"><strong>Best for:</strong> {guide["good_for"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def _extract_weight_matrix(agent) -> Optional[np.ndarray]:
    if hasattr(agent, "get_weights_matrix"):
        try:
            matrix = np.asarray(agent.get_weights_matrix())
            if matrix.size > 0:
                return matrix
        except Exception:
            pass

    if hasattr(agent, "_thompson") and hasattr(agent._thompson, "get_weights_matrix"):
        try:
            matrix = np.asarray(agent._thompson.get_weights_matrix())
            if matrix.size > 0:
                return matrix
        except Exception:
            pass

    for attr in ("_theta", "_weights"):
        if hasattr(agent, attr):
            matrix = np.asarray(getattr(agent, attr))
            if matrix.ndim == 2 and matrix.size > 0:
                return matrix

    return None

inject_styles()
init_session_state()

st.title("Execution Replay Studio")
st.caption(
    "Compare different strategies for buying a large amount of BTC at the lowest possible cost. "
    "Open a saved result from the sidebar for an instant view, or load the engine and run new simulations."
)

all_agents = list_agents()
default_agents = [name for name in DEFAULT_AGENT_SHORTLIST if name in all_agents]
if not default_agents:
    default_agents = all_agents[: min(5, len(all_agents))]

engine_bundle = current_engine_bundle()

with st.sidebar:
    st.title("Execution Studio")
    st.caption("Simulate and compare trading strategies that execute a large BTC order at minimum cost.")

    st.markdown(
        """
        <div class="help-card">
            <strong>Two ways to get started</strong><br>
            <b>Option A (instant):</b> Pick a saved result below to view previous comparison runs instantly — no loading needed.<br><br>
            <b>Option B (fresh run):</b> Load the engine, configure settings, then click "Compare All" to run new simulations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    results_files = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    selected_results = None
    if results_files:
        preferred_results = default_results_path(results_files)
        # Sync any path staged by a completed batch run (must happen before widget is created)
        pending = st.session_state.get("_pending_saved_batch_path")
        if pending and pending in results_files:
            st.session_state["saved_batch_selection"] = pending
            st.session_state["_pending_saved_batch_path"] = None
        elif st.session_state.get("saved_batch_selection") not in results_files:
            st.session_state["saved_batch_selection"] = preferred_results
        selected_results = st.selectbox(
            "Load a previous result",
            options=results_files,
            key="saved_batch_selection",
            format_func=describe_results_option,
            help="Previous comparison runs are saved automatically. Select one to view it instantly without re-running anything.",
        )
    else:
        st.info("No saved results yet. Run a comparison below to create the first one.")

    st.markdown("---")
    st.subheader("Simulation Engine")
    st.caption("Load market data into memory so you can run fresh simulations. Only needed for 'Compare All' / 'Test Once'.")
    split_label = st.selectbox(
        "Which dataset to use?",
        options=list(SPLIT_OPTIONS.keys()),
        index=0,
        help=(
            "The data is split into three non-overlapping sets. "
            "Train = data the agents may have seen during learning. "
            "Val = tuning/debugging set. "
            "Test = held-out data the agents have never seen — use this for final results."
        ),
    )
    split_value = SPLIT_OPTIONS[split_label]
    split_dates = available_dates_for_split(split_value)

    start_options = ["Auto"] + split_dates
    selected_start = st.selectbox(
        "Start Date (optional)",
        options=start_options,
        index=0,
        help="Leave as 'Auto' to use all available dates in the selected dataset. Only change this if you want to test on a specific date range.",
    )
    start_date_value = None if selected_start == "Auto" else selected_start

    end_candidates = split_dates
    if start_date_value:
        end_candidates = [date for date in split_dates if date >= start_date_value]
    end_options = ["Auto"] + end_candidates
    default_end_index = len(end_options) - 1 if len(end_options) > 1 else 0
    selected_end = st.selectbox(
        "End Date (optional)",
        options=end_options,
        index=default_end_index,
        help="Leave as 'Auto' to use the most recent available date. Only change this to restrict the window.",
    )
    end_date_value = None if selected_end == "Auto" else selected_end

    filtered_dates = split_dates
    if start_date_value:
        filtered_dates = [date for date in filtered_dates if date >= start_date_value]
    if end_date_value:
        filtered_dates = [date for date in filtered_dates if date <= end_date_value]

    limit_loaded_days = st.checkbox(
        "Limit how many days are loaded (saves memory)",
        value=True,
        help="Loading fewer days uses less RAM and starts faster. Recommended unless you need a very large date range.",
    )
    max_days_value = None
    if limit_loaded_days:
        max_allowed_days = max(1, len(filtered_dates)) if filtered_dates else 1
        max_days_value = int(
            st.number_input(
                "Max days to load into memory",
                min_value=1,
                max_value=max_allowed_days,
                value=min(7, max_allowed_days),
                step=1,
                help="The engine will load the most recent N days from the selected window. 7 days is usually enough for reliable test results.",
            )
        )

    if filtered_dates:
        estimated_loaded_days = min(len(filtered_dates), max_days_value) if max_days_value else len(filtered_dates)
        st.caption(
            f"Replay window available: {filtered_dates[0]} -> {filtered_dates[-1]} | "
            f"{len(filtered_dates)} matching day(s), loading up to the most recent {estimated_loaded_days}."
        )
    else:
        st.warning("No parsed dates match this split/date-window selection.")

    load_col, unload_col = st.columns(2)
    load_engine_clicked = load_col.button("Load Engine", type="primary", width="stretch",
        help="Reads market data into memory so fresh simulations can run. Takes a few seconds.")
    unload_engine_clicked = unload_col.button("Unload", width="stretch",
        help="Frees the market data from memory. Do this to recover RAM when you are done running simulations.")

    if load_engine_clicked:
        if not filtered_dates:
            st.error("Cannot load the engine because the selected replay window has no available parsed dates.")
        else:
            st.session_state["pending_engine_load"] = {
                "split": split_value,
                "split_label": split_label,
                "start_date": start_date_value,
                "end_date": end_date_value,
                "max_days": max_days_value,
            }

    if unload_engine_clicked:
        clear_engine_bundle()
        st.session_state["pending_engine_load"] = None
        st.session_state["pending_batch_run"] = None
        engine_bundle = current_engine_bundle()

    if st.session_state.get("pending_engine_load"):
        st.info("Loading market data — check the main panel for progress.")
    elif engine_ready(engine_bundle):
        st.success(f"Ready to simulate ({engine_bundle['split']} data)")
        st.caption(f"Loaded window: {describe_window(engine_bundle)}")
    elif engine_bundle.get("error"):
        st.error(f"Load failed: {engine_bundle['error']}")
    else:
        st.info("Engine not loaded yet. Click 'Load Engine' above to get started.")

    # --- batch_episodes lives OUTSIDE the form so the button label updates immediately ---
    batch_episodes = st.slider(
        "How many test runs? (Batch Size)",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help=(
            "Each 'run' is one simulated order execution. "
            "More runs = more reliable results but takes longer. "
            "Start with 20 to get a quick read, then increase to 50-100 for publication-quality comparisons."
        ),
    )

    with st.form("run_controls"):
        st.subheader("Run Setup")

        selected_agents = st.multiselect(
            "Which trading strategies to compare?",
            options=all_agents,
            default=default_agents,
            help=(
                "Each strategy is a different way of breaking a large order into smaller slices. "
                "TWAP and VWAP are simple baselines. LinUCB / Thompson are the AI agents that learn. "
                "Start with fewer agents for faster runs."
            ),
        )
        st.caption("Tip: start with 2-3 agents to keep things fast, then add more once you like the setup.")

        col_a, col_b = st.columns(2)
        with col_a:
            order_size_btc = st.slider(
                "Order Size (BTC to buy)",
                min_value=1,
                max_value=50,
                value=10,
                help=(
                    "How many Bitcoin to buy in the simulated order. "
                    "Larger orders are harder to execute cheaply — they move the market against you. "
                    "10 BTC is a realistic institutional-sized trade."
                ),
            )
        with col_b:
            time_horizon = st.selectbox(
                "Time to complete the order",
                ["15min", "30min", "1hr", "2hr"],
                index=1,
                help=(
                    "How long the agent has to finish buying all the BTC. "
                    "Shorter = more urgency, likely more market impact. "
                    "Longer = more time to find good prices, but more risk of the price drifting away."
                ),
            )

        col_c, col_d = st.columns(2)
        with col_c:
            episode_seed = st.number_input(
                "Random Seed",
                value=42,
                min_value=0,
                step=1,
                help=(
                    "Controls which random market windows are chosen for testing. "
                    "Changing this number picks different test scenarios. "
                    "Keep it the same across runs to compare agents fairly."
                ),
            )
        with col_d:
            batch_workers = st.slider(
                "Speed (parallel workers)",
                min_value=1,
                max_value=min(4, max(1, len(all_agents))),
                value=min(2, max(1, len(all_agents))),
                step=1,
                help=(
                    "How many agents to test at the same time using multiple CPU cores. "
                    "Higher = faster but uses more of your computer. "
                    "Set to 1 if your machine feels slow."
                ),
            )

        risk_aversion = st.slider(
            "AC Caution Level (how risk-averse the AC-style agents are)",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help=(
                "Only affects the Almgren-Chriss (AC) agent and hybrids based on it. "
                "Low = willing to spread out the order more (cheaper but more timing risk). "
                "High = wants to finish quickly to avoid price drift (safer but more impact cost). "
                "0.10 is a sensible default for most situations."
            ),
        )

        engine_not_ready = not engine_ready(engine_bundle)
        if engine_not_ready:
            st.info("Load the engine above before running simulations.")

        single_col, batch_col = st.columns(2)
        run_single = single_col.form_submit_button(
            "Test Once (1 episode)",
            type="primary",
            width="stretch",
            disabled=engine_not_ready,
        )
        run_batch = batch_col.form_submit_button(
            f"Compare All ({batch_episodes} runs)",
            width="stretch",
            disabled=engine_not_ready,
        )

pending_engine_load = st.session_state.get("pending_engine_load")
if pending_engine_load:
    load_status = st.status("Loading replay engine...", expanded=True)
    load_status.write(
        f"Preparing the {pending_engine_load['split_label'].lower()} replay window. "
        "The dashboard stays visible while market data is being read."
    )
    cfg, env, env_error = load_environment(
        split=pending_engine_load["split"],
        start_date=pending_engine_load["start_date"],
        end_date=pending_engine_load["end_date"],
        max_days=pending_engine_load["max_days"],
    )
    store_engine_bundle(
        pending_engine_load["split"],
        cfg,
        env,
        env_error,
        start_date=pending_engine_load["start_date"],
        end_date=pending_engine_load["end_date"],
        max_days=pending_engine_load["max_days"],
    )
    st.session_state["pending_engine_load"] = None
    engine_bundle = current_engine_bundle()

    if env_error:
        load_status.update(label="Replay engine failed to load", state="error", expanded=True)
        load_status.write(env_error)
    else:
        load_status.update(label="Replay engine ready", state="complete", expanded=False)
        load_status.write(f"Loaded window: {describe_window(engine_bundle)}")

if run_single and engine_ready(engine_bundle):
    horizon_s = horizon_to_seconds(time_horizon)
    runtime_agents = build_agents(selected_agents)
    with st.spinner("Running single-episode simulation..."):
        episode_data = run_single_episode(
            engine_bundle["env"],
            runtime_agents,
            selected_agents,
            seed=int(episode_seed),
            order_size=order_size_btc,
            horizon_s=horizon_s,
            risk_aversion=float(risk_aversion),
        )
    st.session_state["runtime_agents"] = runtime_agents
    st.session_state["single_episode_data"] = episode_data
    st.session_state["single_episode_meta"] = {
        "selected_agents": list(selected_agents),
        "seed": int(episode_seed),
        "order_size_btc": order_size_btc,
        "time_horizon": time_horizon,
        "risk_aversion": float(risk_aversion),
    }

if run_batch and engine_ready(engine_bundle):
    if selected_agents:
        st.session_state["latest_batch_data"] = None
        st.session_state["latest_batch_path"] = None
        st.session_state["pending_batch_run"] = {
            "selected_agents": list(selected_agents),
            "n_episodes": int(batch_episodes),
            "random_seed": int(episode_seed),
            "risk_aversion": float(risk_aversion),
            "order_size_btc": float(order_size_btc),
            "time_horizon": time_horizon,
            "horizon_s": horizon_to_seconds(time_horizon),
            "batch_workers": int(batch_workers),
            "split": engine_bundle.get("split", "test"),
            "start_date": engine_bundle.get("requested_start_date"),
            "end_date": engine_bundle.get("requested_end_date"),
            "max_days": engine_bundle.get("requested_max_days"),
        }

loaded_results_data = load_results_file(selected_results) if selected_results else None
if st.session_state.get("pending_batch_run"):
    active_batch_data = st.session_state.get("latest_batch_data")
else:
    active_batch_data = loaded_results_data or st.session_state.get("latest_batch_data")
active_batch_label = describe_results_source(
    selected_results if loaded_results_data else st.session_state.get("latest_batch_path"),
    active_batch_data,
)
if st.session_state.get("pending_batch_run") and not active_batch_data:
    active_batch_label = "Live run"
active_batch_agents = batch_agent_names(active_batch_data)
saved_batch_mismatch = bool(active_batch_agents) and len(active_batch_agents) < len(selected_agents)

_engine_ok = engine_ready(engine_bundle)
_status_items = [
    ("Engine",        "Loaded" if _engine_ok else "Idle",                          "#22c55e" if _engine_ok else "#f59e0b"),
    ("Split",         engine_bundle.get("split", "—") if engine_bundle else "—",   "#60a5fa"),
    ("Window",        describe_window(engine_bundle),                               "#a78bfa"),
    ("Agents set up", str(len(selected_agents) if selected_agents else 0),          "#f472b6"),
    ("Batch agents",  str(len(active_batch_agents) if active_batch_agents else 0),  "#34d399"),
    ("Last run",      active_batch_label or "—",                                    "#94a3b8"),
]
_badges = "".join(
    f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:18px;'
    f'font-size:0.78rem;color:#cbd5e1">'
    f'<span style="color:{color};font-weight:600">{label}</span>'
    f'<span style="background:#1e293b;color:#e2e8f0;padding:2px 8px;border-radius:9999px;'
    f'font-size:0.75rem;font-weight:500;border:1px solid #334155">{value}</span>'
    f'</span>'
    for label, value, color in _status_items
)
st.markdown(
    f'<div style="padding:8px 4px 4px 0;border-bottom:1px solid #1e293b;margin-bottom:4px">{_badges}</div>',
    unsafe_allow_html=True,
)

st.info("Charts are interactive: hover, zoom, pan, hide traces from the legend, and export images from the Plotly toolbar.")
if saved_batch_mismatch:
    st.caption(
        f"The batch currently being viewed contains {len(active_batch_agents)} agent(s), "
        f"while your current run setup selects {len(selected_agents)}. Run a fresh batch if you want the view to match the current setup."
    )

_tab_single, _tab_batch, _tab_internals, _tab_guide = st.tabs([
    "Single Episode",
    "Batch Comparison",
    "Agent Internals",
    "Agent Guide",
])

with _tab_single:
 if True:
    st.header("Single Episode Execution View")
    st.caption("Use this view for one replayed execution path across the selected agents.")

    if st.session_state.get("single_episode_data"):
        episode_data = st.session_state["single_episode_data"]
        episode_meta = st.session_state.get("single_episode_meta") or {}

        if not engine_ready(engine_bundle):
            st.info("Showing the last completed single-episode run. Load the engine again if you want to generate a new one.")

        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        meta_col1.metric("Agents", len(episode_meta.get("selected_agents", [])))
        meta_col2.metric("Order Size", f"{episode_meta.get('order_size_btc', '--')} BTC")
        meta_col3.metric("Horizon", str(episode_meta.get("time_horizon", "--")))
        meta_col4.metric("Seed", str(episode_meta.get("seed", "--")))

        arrival = list(episode_data.values())[0]["arrival_price"]
        agents_fills = {k: v["fills"] for k, v in episode_data.items()}

        st.subheader("Execution Trajectory")
        render_plotly(build_trajectory_chart(agents_fills, arrival))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Urgency Level Heatmap")
            agents_actions = {k: v["action_history"] for k, v in episode_data.items()}
            render_plotly(build_urgency_heatmap(agents_actions))

        with col2:
            st.subheader("Slice-by-Slice IS Contribution")
            render_plotly(build_is_contribution_chart(agents_fills, arrival))

        first_agent = list(episode_data.keys())[0]
        st.subheader(f"Context Vector Over Time ({first_agent})")
        ctx_hist = episode_data[first_agent]["context_history"]
        render_plotly(build_context_heatmap(ctx_hist))
    elif not engine_ready(engine_bundle):
        st.info("Load the engine from the sidebar first. The dashboard does not auto-load heavy simulation state on startup anymore.")
    else:
        st.info("Choose agents and settings in the sidebar, then click 'Run Single Episode'.")

with _tab_batch:
 if True:
    st.header("Batch Comparison")
    st.caption("Compare saved or freshly generated batch results without auto-running anything in the background.")

    pending_batch_run = st.session_state.get("pending_batch_run")
    if pending_batch_run and engine_ready(engine_bundle):
        selected_batch_agents = pending_batch_run["selected_agents"]
        seeds = np.random.RandomState(pending_batch_run["random_seed"]).randint(
            0, int(1e6), pending_batch_run["n_episodes"]
        ).tolist()
        completed_results: Dict[str, List[Dict[str, float]]] = {}
        runtime_agents: Dict[str, object] = {}
        failed_agents: Dict[str, str] = {}
        total_agents = len(selected_batch_agents)
        worker_count = max(1, min(pending_batch_run["batch_workers"], total_agents))

        st.info(
            f"Running {total_agents} agent(s) with up to {worker_count} worker(s). "
            "Completed agents will appear here as soon as each finishes."
        )
        batch_status = st.status("Batch run in progress...", expanded=True)
        batch_status.write(
            f"Order size: {pending_batch_run['order_size_btc']:.2f} BTC | "
            f"Horizon: {pending_batch_run['time_horizon']} | "
            f"Episodes per agent: {pending_batch_run['n_episodes']}"
        )
        progress_bar = st.progress(0.0)
        completion_caption = st.empty()
        partial_results_slot = st.empty()
        partial_results_slot.info("Waiting for the first agent to finish its batch so live charts can appear.")

        future_to_agent = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dashboard-batch") as executor:
            for agent_name in selected_batch_agents:
                future = executor.submit(
                    run_agent_batch_job,
                    agent_name=agent_name,
                    split=pending_batch_run["split"],
                    start_date=pending_batch_run["start_date"],
                    end_date=pending_batch_run["end_date"],
                    max_days=pending_batch_run["max_days"],
                    seeds=seeds,
                    risk_aversion=pending_batch_run["risk_aversion"],
                    order_size=pending_batch_run["order_size_btc"],
                    horizon_s=pending_batch_run["horizon_s"],
                )
                future_to_agent[future] = agent_name

            for completed_index, future in enumerate(as_completed(future_to_agent), start=1):
                agent_name = future_to_agent[future]
                payload = future.result()
                if payload.get("error"):
                    failed_agents[agent_name] = payload["error"]
                    batch_status.write(f"{agent_name}: failed")
                else:
                    completed_results[agent_name] = payload["episodes"]
                    if payload.get("agent") is not None:
                        runtime_agents[agent_name] = payload["agent"]
                    batch_status.write(f"{agent_name}: complete")

                progress_bar.progress(completed_index / max(total_agents, 1))
                completion_caption.caption(
                    f"Completed: {len(completed_results)}/{total_agents} | "
                    f"Failed: {len(failed_agents)}"
                )

                partial_output = build_dashboard_batch_output(
                    completed_results,
                    n_episodes=pending_batch_run["n_episodes"],
                    random_seed=pending_batch_run["random_seed"],
                    seeds=seeds,
                    split="dashboard",
                    failed_agents=failed_agents,
                )
                partial_summary = merge_regret_into_summary(
                    partial_output.get("summary", {}),
                    partial_output.get("regret_tracking", {}),
                )

                with partial_results_slot.container():
                    if completed_results:
                        st.subheader("Results So Far (updating as each strategy finishes...)")
                        st.caption(
                            "Each strategy completes one at a time. Avg Cost is the quickest first read — lower is better."
                        )
                        render_plotly(build_violin_plot(partial_output.get("is_distributions", {})))
                        render_plotly(build_comparison_table(partial_summary))

        final_output = build_dashboard_batch_output(
            completed_results,
            n_episodes=pending_batch_run["n_episodes"],
            random_seed=pending_batch_run["random_seed"],
            seeds=seeds,
            split="dashboard",
            failed_agents=failed_agents,
        )
        out_path = save_dashboard_batch_output(final_output)
        st.session_state["runtime_agents"] = runtime_agents
        st.session_state["latest_batch_data"] = final_output
        st.session_state["latest_batch_path"] = out_path
        st.session_state["_pending_saved_batch_path"] = out_path
        st.session_state["pending_batch_run"] = None
        active_batch_data = final_output
        active_batch_label = describe_results_source(out_path, final_output)
        active_batch_agents = batch_agent_names(active_batch_data)

        if failed_agents:
            batch_status.update(label="Batch run finished with some agent failures", state="error", expanded=True)
            for failed_agent, failed_error in failed_agents.items():
                batch_status.write(f"{failed_agent}: {failed_error}")
        else:
            batch_status.update(label="Batch run finished", state="complete", expanded=False)

        # Trigger a clean rerun so the final results section renders from saved state
        st.rerun()

    if active_batch_data:
        if active_batch_label:
            st.success(f"Viewing results from: {active_batch_label}")

        summary = merge_regret_into_summary(
            active_batch_data.get("summary", {}),
            active_batch_data.get("regret_tracking", {}),
        )
        is_distributions = active_batch_data.get("is_distributions", {})
        metric_keys = metric_options(summary)

        if active_batch_agents:
            st.caption(f"Agents in this batch: {', '.join(active_batch_agents)}")

        render_metric_dictionary(metric_keys)

        st.subheader("IS Distribution")
        st.caption("Lower on the y-axis is better. Narrower shapes mean more consistent outcomes across episodes.")
        render_plotly(build_violin_plot(is_distributions))

        if metric_keys:
            selected_metric = st.selectbox(
                "Compare Metric",
                options=metric_keys,
                format_func=lambda k: METRIC_LABELS.get(k, k),
            )
            render_metric_guide(selected_metric, summary)
            render_plotly(build_metric_bar_chart(summary, selected_metric))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Win Rate Matrix")
            st.caption("Read each row against each column. Values above 0.50 mean the row agent beat the column agent more often than not.")
            pseudo_results = {}
            for agent_name, is_vals in is_distributions.items():
                pseudo_results[agent_name] = [
                    type("EpisodeResultProxy", (), {"implementation_shortfall_bps": v})()
                    for v in is_vals
                ]
            render_plotly(build_win_rate_matrix(pseudo_results))

        with col2:
            st.subheader("Learning Curves")
            st.caption("Lower curves are better. This is most useful for adaptive agents because it shows whether they improve as episodes accumulate.")
            bandit_is = {k: v for k, v in is_distributions.items() if k in learning_agents()}
            render_plotly(build_learning_curves(bandit_is))

        st.subheader("Strategy Scorecard")
        st.caption(
            "Strategies are ranked by Avg Cost — the cheaper the better. "
            "Green cells are good, yellow is okay, red needs attention. "
            "The 'Quick Verdict' column gives a one-word summary for each strategy."
        )
        legend_col, _ = st.columns([1, 3])
        with legend_col:
            render_plotly(build_colour_legend())
        render_plotly(build_comparison_table(summary))

        if active_batch_data.get("regret_curves") and go is not None:
            st.subheader("Counterfactual Regret")
            st.caption("Near zero is strong. Rising regret means the agent kept losing to the TWAP baseline over time.")
            fig_regret = go.Figure()
            for agent_name, curve in active_batch_data["regret_curves"].items():
                fig_regret.add_trace(
                    go.Scatter(
                        x=list(range(len(curve))),
                        y=curve,
                        name=agent_name,
                        mode="lines",
                    )
                )
            fig_regret.update_layout(
                title="Counterfactual Regret Curves",
                xaxis_title="Episode",
                yaxis_title="Regret (bps)",
                height=360,
            )
            render_plotly(fig_regret)
    else:
        st.info("Select a saved results file from the sidebar or run a new batch after loading the engine.")

with _tab_guide:
 if True:
        render_agent_guide()

with _tab_internals:
 if True:
        st.header("Agent Internals")
        st.caption("Every agent shows the same three panels: What it learned · How it behaves · How it evolved. Run an agent first to see live state instead of a blank preview.")

        # ── Agent type descriptions ───────────────────────────────────────────
        _AGENT_TYPE_DESC = {
            "LinUCB":           ("Linear Contextual Bandit — UCB",  "Maintains a weight vector per urgency action. Picks the action with the highest predicted reward plus an exploration bonus."),
            "Thompson":         ("Linear Contextual Bandit — Thompson Sampling", "Maintains a Bayesian posterior over weight vectors. Samples a plausible model at each step and acts on it."),
            "EXP3":             ("Adversarial Bandit — EXP3",        "Maintains an exponential weight over actions. Does not assume a linear reward model — robust to non-stationary markets."),
            "KernelUCB":        ("Kernel Bandit — RBF UCB",          "Uses past observations directly (no weight vector). Scores actions by similarity to historical contexts where they worked."),
            "MetaAgent":        ("Hierarchical Bandit — Agent Selector", "A top-level LinUCB that decides which sub-agent (TWAP / AC / LinUCB / Thompson) should handle each slice."),
            "Corral":           ("Online Model Selection — Corral",  "Maintains a probability distribution over sub-agents. Routes each slice to a sampled sub-agent and shifts probability toward whoever performs best."),
            "ThompsonACHybrid": ("Constrained Bandit — Thompson + AC Clip", "Runs Thompson Sampling but clips any chosen action to stay within ±1 urgency level of what AC recommends. Exploration is bounded by theory."),
        }

        runtime_agents = st.session_state.get("runtime_agents") or {}
        learning_agent_names = [name for name in list_agents() if name in learning_agents()]
        preview_agents = preview_learning_agents(tuple(learning_agent_names)) if learning_agent_names else {}
        internals_agents = dict(preview_agents)
        internals_agents.update(runtime_agents)

        if not learning_agent_names:
            st.info("No learning agents are available in the current registry.")
        else:
            selected_learning_agent = st.selectbox(
                "Choose Agent to Inspect",
                options=learning_agent_names,
                format_func=lambda name: f"{name}  (live — from last run)" if name in runtime_agents else f"{name}  (preview — run it first for live state)",
            )
            agent = internals_agents[selected_learning_agent]

            # Agent type banner
            if selected_learning_agent in _AGENT_TYPE_DESC:
                type_label, type_desc = _AGENT_TYPE_DESC[selected_learning_agent]
                st.markdown(
                    f"""
                    <div class="help-card">
                        <strong>{selected_learning_agent} — {type_label}</strong><br>
                        {type_desc}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if selected_learning_agent not in runtime_agents:
                st.info("Showing a fresh (untrained) preview. Run this agent in Batch Comparison to see what it actually learned.")

            URGENCY_LABELS = ["Very Passive", "Passive", "On Pace", "Aggressive", "Very Aggressive"]

            # ── Helper: sub-agent names ───────────────────────────────────────
            def _sub_names(ag) -> list:
                for attr in ("_sub_agent_names", "_base_agent_names"):
                    if hasattr(ag, attr):
                        return list(getattr(ag, attr))
                return []

            # ── Helper: build sub-agent usage bar chart ───────────────────────
            def _build_subagent_bar(names, counts, title) -> "go.Figure":
                if go is None:
                    return None
                total = sum(counts) or 1
                pcts = [f"{c/total*100:.1f}%" for c in counts]
                fig = go.Figure(go.Bar(
                    x=names,
                    y=list(counts),
                    text=pcts,
                    textposition="outside",
                    marker_color=["steelblue", "darkorange", "firebrick", "mediumpurple"][:len(names)],
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Sub-Agent",
                    yaxis_title="Times Selected",
                    height=320,
                    margin=dict(t=40, b=40),
                )
                return fig

            # ── Helper: Corral probability evolution line chart ───────────────
            def _build_corral_prob_chart(prob_history, names) -> "go.Figure":
                if go is None or not prob_history:
                    return None
                episodes = [snap["episode"] for snap in prob_history]
                fig = go.Figure()
                n = len(names)
                colors = ["steelblue", "darkorange", "firebrick", "mediumpurple"]
                for i, name in enumerate(names):
                    probs = [snap["p"][i] if i < len(snap["p"]) else 0 for snap in prob_history]
                    fig.add_trace(go.Scatter(
                        x=episodes, y=probs,
                        name=name,
                        mode="lines",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
                fig.update_layout(
                    title="How Corral's Confidence in Each Sub-Agent Changed Over Episodes",
                    xaxis_title="Episode",
                    yaxis_title="Selection Probability",
                    yaxis=dict(range=[0, 1]),
                    height=340,
                    legend=dict(orientation="h", y=-0.2),
                )
                return fig

            # ── Helper: weight norm evolution chart ───────────────────────────
            def _build_weight_evolution(weight_history) -> "go.Figure":
                if go is None or not weight_history:
                    return None
                episodes, norms = [], []
                for snap in weight_history:
                    ep = snap.get("episode", len(episodes))
                    theta = snap.get("theta") or snap.get("weights")
                    if theta is None:
                        continue
                    try:
                        # Handle ragged lists (KernelUCB mu_vectors have varying lengths)
                        flat = []
                        for item in theta:
                            if isinstance(item, (list, np.ndarray)) and len(item) > 0:
                                flat.extend([float(v) for v in item])
                            elif isinstance(item, (int, float, np.floating)):
                                flat.append(float(item))
                        if not flat:
                            continue
                        norms.append(float(np.linalg.norm(flat)))
                        episodes.append(ep)
                    except Exception:
                        continue
                if not norms:
                    return None
                fig = go.Figure(go.Scatter(
                    x=episodes, y=norms,
                    mode="lines+markers",
                    line=dict(color="steelblue", width=2),
                    marker=dict(size=4),
                ))
                fig.update_layout(
                    title="Weight Magnitude Over Training  (growing = agent is learning)",
                    xaxis_title="Episode",
                    yaxis_title="||θ|| (weight vector norm)",
                    height=300,
                    margin=dict(t=40, b=40),
                )
                return fig

            # ── Helper: MetaAgent rolling sub-agent selection chart ───────────
            def _build_meta_selection_chart(selection_history, names, window=10) -> "go.Figure":
                if go is None or len(selection_history) < 2:
                    return None
                n_agents = len(names)
                colors = ["steelblue", "darkorange", "firebrick", "mediumpurple"]
                fig = go.Figure()
                for k, name in enumerate(names):
                    # Rolling share of selections for sub-agent k
                    indicator = [1 if s == k else 0 for s in selection_history]
                    rolling = [
                        sum(indicator[max(0, i - window):i + 1]) / min(i + 1, window)
                        for i in range(len(indicator))
                    ]
                    fig.add_trace(go.Scatter(
                        x=list(range(len(rolling))), y=rolling,
                        name=name, mode="lines",
                        line=dict(color=colors[k % len(colors)], width=2),
                    ))
                fig.update_layout(
                    title=f"MetaAgent — Rolling {window}-Step Selection Share Per Sub-Agent",
                    xaxis_title="Decision Step",
                    yaxis_title="Selection Rate",
                    yaxis=dict(range=[0, 1]),
                    height=320,
                    legend=dict(orientation="h", y=-0.2),
                )
                return fig

            # ── Helper: ThompsonACHybrid clip/agreement rate over episodes ─────
            def _build_hybrid_episode_chart(episode_history) -> "go.Figure":
                if go is None or not episode_history:
                    return None
                eps = [snap["episode"] for snap in episode_history]
                clips = [snap["clip_rate"] for snap in episode_history]
                agrees = [snap["agreement_rate"] for snap in episode_history]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eps, y=clips, name="Clip Rate", mode="lines+markers",
                                         line=dict(color="firebrick", width=2), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=eps, y=agrees, name="Agreement Rate", mode="lines+markers",
                                         line=dict(color="steelblue", width=2), marker=dict(size=4)))
                fig.update_layout(
                    title="ThompsonACHybrid — Clip & Agreement Rate Over Episodes",
                    xaxis_title="Episode",
                    yaxis_title="Rate",
                    yaxis=dict(range=[0, 1]),
                    height=320,
                    legend=dict(orientation="h", y=-0.2),
                )
                return fig

            # ═══════════════════════════════════════════════════════════════════
            # PANEL LAYOUT — same 3 panels for every agent
            # ═══════════════════════════════════════════════════════════════════

            # ── PANEL 1: What it learned ──────────────────────────────────────
            st.markdown("---")
            st.subheader("Panel 1 — What it learned")

            matrix = _extract_weight_matrix(agent)
            sub_names = _sub_names(agent)

            if hasattr(agent, "probability_history") and agent.probability_history:
                # Corral — show current probability distribution as a bar chart
                latest = agent.probability_history[-1]
                p_vals = latest.get("p", [])
                labels = sub_names if sub_names and len(sub_names) == len(p_vals) else [f"Agent {i}" for i in range(len(p_vals))]
                if go is not None and p_vals:
                    fig_p = go.Figure(go.Bar(
                        x=labels,
                        y=p_vals,
                        text=[f"{v:.1%}" for v in p_vals],
                        textposition="outside",
                        marker_color=["steelblue", "darkorange", "firebrick", "mediumpurple"][:len(labels)],
                    ))
                    fig_p.update_layout(
                        title=f"Current Probability Assigned to Each Sub-Agent  (episode {latest.get('episode', '?')})",
                        xaxis_title="Sub-Agent",
                        yaxis_title="Selection Probability",
                        yaxis=dict(range=[0, max(p_vals) * 1.25]),
                        height=320,
                    )
                    render_plotly(fig_p)
                    st.caption("Higher bar = Corral trusts this sub-agent more. Probability shifts toward whoever achieves better execution cost.")

            elif matrix is not None and matrix.size > 0:
                # Weight-based agents — heatmap + feature importance side by side
                wt_col1, wt_col2 = st.columns(2)
                with wt_col1:
                    render_plotly(build_weight_heatmap(matrix, selected_learning_agent))
                    st.caption("Each cell = how much this feature pushes toward this urgency level. Red = pushes toward it, blue = pushes away.")
                with wt_col2:
                    render_plotly(build_feature_importance(matrix, selected_learning_agent))
                    st.caption("Mean absolute weight across all urgency levels. Taller bar = this feature influences decisions more.")

            elif selected_learning_agent == "KernelUCB":
                st.info(
                    "KernelUCB does not use weight vectors. "
                    "It stores past (context, action, reward) observations and scores new decisions by similarity to historical cases. "
                    "The more observations it accumulates, the better its predictions — but there is no single interpretable weight to display."
                )
                if hasattr(agent, "_obs_contexts"):
                    for i, obs in enumerate(agent._obs_contexts):
                        st.caption(f"Action {URGENCY_LABELS[i]}: {len(obs)} stored observations")
            else:
                st.info("Run this agent in a batch to populate its learned state.")

            # ── PANEL 2: How it behaves ───────────────────────────────────────
            st.markdown("---")
            st.subheader("Panel 2 — How it behaves")
            beh_col1, beh_col2 = st.columns(2)

            with beh_col1:
                if hasattr(agent, "action_counts"):
                    counts = np.asarray(agent.action_counts)
                    if counts.sum() > 0:
                        render_plotly(build_action_distribution({selected_learning_agent: counts}))
                        dominant = URGENCY_LABELS[int(np.argmax(counts))]
                        st.caption(f"Most chosen urgency level: **{dominant}**. A healthy agent spreads decisions across levels rather than always picking one.")
                    else:
                        st.info("No decisions recorded yet. Run the agent to populate this chart.")

                elif hasattr(agent, "sub_agent_selection_counts"):
                    # MetaAgent
                    counts = np.asarray(agent.sub_agent_selection_counts)
                    names = sub_names if sub_names else [f"Agent {i}" for i in range(len(counts))]
                    if counts.sum() > 0:
                        render_plotly(_build_subagent_bar(names, counts, "Which Sub-Agent Was Chosen Most Often"))
                        dominant = names[int(np.argmax(counts))] if len(names) > 0 else "?"
                        st.caption(f"Most used sub-agent: **{dominant}**. MetaAgent has learned to trust this agent the most for current market conditions.")
                    else:
                        st.info("No selections recorded yet.")

                elif hasattr(agent, "selection_history") and agent.selection_history:
                    # Corral selection history as bar chart
                    history = np.asarray(agent.selection_history)
                    n_agents = len(sub_names) if sub_names else (max(history) + 1)
                    counts = np.bincount(history, minlength=n_agents)
                    names = sub_names if sub_names else [f"Agent {i}" for i in range(n_agents)]
                    render_plotly(_build_subagent_bar(names, counts, "How Often Corral Routed to Each Sub-Agent"))
                    st.caption(f"Total routing decisions: {len(history)}. Higher bar = Corral chose this sub-agent more often overall.")
                else:
                    st.info("No behavioral data available for this agent type.")

            with beh_col2:
                if hasattr(agent, "clip_rate") and hasattr(agent, "agreement_rate"):
                    st.markdown("**Hybrid Constraint Summary**")
                    m1, m2 = st.columns(2)
                    m1.metric(
                        "Clip Rate",
                        f"{agent.clip_rate:.1%}",
                        help="Fraction of decisions where Thompson's choice was overridden by the AC safety constraint. High = AC is doing a lot of the work.",
                    )
                    m2.metric(
                        "Agreement Rate",
                        f"{agent.agreement_rate:.1%}",
                        help="Fraction of decisions where Thompson naturally agreed with AC's recommendation. High = the bandit has learned to mimic AC.",
                    )
                    st.caption(
                        "Clip Rate + Agreement Rate do not need to sum to 1 — the agent can both agree and be clipped when Thompson's top choice "
                        "matches AC but would have been clipped anyway. A healthy hybrid has a declining clip rate over time as the bandit learns."
                    )

                elif hasattr(agent, "_sub_agents"):
                    # MetaAgent — show action distribution of each sub-agent
                    st.markdown("**Sub-Agent Action Distributions**")
                    sub_agents = getattr(agent, "_sub_agents", [])
                    names = sub_names if sub_names else [f"Agent {i}" for i in range(len(sub_agents))]
                    for sub_name, sub_agent in zip(names, sub_agents):
                        if hasattr(sub_agent, "action_counts") and np.asarray(sub_agent.action_counts).sum() > 0:
                            counts_py = [int(c) for c in sub_agent.action_counts]
                            total_c = sum(counts_py) or 1
                            pct_str = ", ".join(
                                f"{URGENCY_LABELS[i] if i < len(URGENCY_LABELS) else i}: {counts_py[i]} ({counts_py[i]/total_c*100:.0f}%)"
                                for i in range(len(counts_py))
                            )
                            st.caption(f"**{sub_name}** — {pct_str}")
                else:
                    st.info("No additional behavioral metrics for this agent type.")

            # ── PANEL 3: How it evolved ───────────────────────────────────────
            st.markdown("---")
            st.subheader("Panel 3 — How it evolved over training")

            if hasattr(agent, "probability_history") and len(agent.probability_history) >= 1:
                # ── Corral — probability evolution over episodes ──────────────
                p_names = sub_names if sub_names else [f"Agent {i}" for i in range(len(agent.probability_history[0].get("p", [])))]
                render_plotly(_build_corral_prob_chart(agent.probability_history, p_names))
                st.caption(
                    "Lines trending upward = Corral is gaining confidence in that sub-agent. "
                    f"Snapshots: {len(agent.probability_history)}. "
                    "Converged (flat) lines = Corral has settled on a favourite."
                )

            elif hasattr(agent, "sub_agent_selection_history") and len(agent.sub_agent_selection_history) >= 2:
                # ── MetaAgent — rolling selection share ───────────────────────
                m_names = sub_names if sub_names else [f"Agent {i}" for i in range(agent.N_SUB_AGENTS)]
                fig_sel = _build_meta_selection_chart(agent.sub_agent_selection_history, m_names)
                if fig_sel is not None:
                    render_plotly(fig_sel)
                    st.caption(
                        f"Each line shows the 10-step rolling share of decisions routed to that sub-agent. "
                        f"Total decisions tracked: {len(agent.sub_agent_selection_history)}. "
                        "A rising line = MetaAgent has learned to trust that sub-agent more over time."
                    )
                else:
                    st.info("MetaAgent selection history recorded but chart could not be built. Need at least 2 decision steps.")

            elif hasattr(agent, "episode_metrics_history") and agent.episode_metrics_history:
                # ── ThompsonACHybrid — clip/agreement rate per episode ────────
                fig_hyb = _build_hybrid_episode_chart(agent.episode_metrics_history)
                if fig_hyb is not None:
                    render_plotly(fig_hyb)
                    st.caption(
                        "Clip Rate (red) trending down = Thompson is learning to stay within the AC safe zone naturally. "
                        "Agreement Rate (blue) trending up = the bandit is converging to match AC's intuition. "
                        f"Episodes logged: {len(agent.episode_metrics_history)}."
                    )
                else:
                    st.info("Episode metrics recorded but chart could not be built.")

            elif hasattr(agent, "_thompson") and hasattr(agent._thompson, "weight_history") and agent._thompson.weight_history:
                # ── ThompsonACHybrid fallback — delegate to inner Thompson ────
                wh = agent._thompson.weight_history
                fig_evo = _build_weight_evolution(wh)
                if fig_evo is not None:
                    render_plotly(fig_evo)
                    st.caption(
                        f"Showing the inner Thompson Sampling agent's weight magnitude (snapshots: {len(wh)}). "
                        "The ThompsonACHybrid improves through its internal Thompson agent."
                    )

            elif hasattr(agent, "weight_history") and agent.weight_history:
                # ── LinUCB / Thompson / EXP3 / KernelUCB ─────────────────────
                wh = agent.weight_history
                fig_evo = _build_weight_evolution(wh)
                if fig_evo is not None:
                    render_plotly(fig_evo)
                    label = {
                        "LinUCB": "The theta vector norm grows as LinUCB accumulates confident action preferences. Snapshots every 10 episodes.",
                        "Thompson": "Posterior mean magnitude rises as the agent builds stronger beliefs about each action's value.",
                        "EXP3": "EXP3 weight magnitude shows how strongly the agent has committed to particular actions over time.",
                        "KernelUCB": "KernelUCB stores past observations; this shows the total magnitude of the kernel regression weights.",
                    }.get(selected_learning_agent, "The weight vector norm grows as the agent observes more data.")
                    st.caption(f"{label} Snapshots stored: {len(wh)}.")
                else:
                    st.info(f"{len(wh)} weight snapshot(s) stored, but no plottable theta/weights key found in the snapshots.")

            else:
                min_eps = 10
                st.info(
                    f"No training history yet — run at least {min_eps} episodes in **Batch Comparison** to populate this panel. "
                    "The agent saves a snapshot every 10 episodes."
                )
