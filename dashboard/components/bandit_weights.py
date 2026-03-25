"""Plotly bandit learned weights visualization."""
from __future__ import annotations
from typing import List, Dict

import numpy as np

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # Optional dependency for dashboard rendering
    go = None

from features.feature_engineer import FeatureEngineer

URGENCY_LABELS = ["Very Passive", "Passive", "On Pace", "Aggressive", "Very Aggressive"]

AGENT_COLORS = {
    "LinUCB": "firebrick",
    "Thompson": "mediumpurple",
    "EXP3": "sienna",
    "KernelUCB": "slateblue",
    "MetaAgent": "black",
    "ThompsonACHybrid": "indigo",
    "Corral": "darkcyan",
}


def _feature_names(theta_matrix: np.ndarray) -> List[str]:
    if theta_matrix is None or theta_matrix.ndim < 2:
        return FeatureEngineer.FEATURE_NAMES
    width = theta_matrix.shape[1]
    if width <= len(FeatureEngineer.BASE_FEATURE_NAMES):
        return FeatureEngineer.BASE_FEATURE_NAMES[:width]
    return FeatureEngineer.FEATURE_NAMES[:width]


def build_weight_heatmap(theta_matrix: np.ndarray, agent_name: str = "") -> go.Figure:
    if go is None:
        return None

    """
    Heatmap of learned weights.
    Rows: 12 feature names; Columns: 5 urgency levels; Color: weight value.
    theta_matrix shape: (5, 12) — one row per action, one col per feature.
    """
    if theta_matrix is None or theta_matrix.size == 0:
        return go.Figure()

    feature_names = _feature_names(theta_matrix)
    fig = go.Figure(data=go.Heatmap(
        z=theta_matrix.T,  # (12, 5)
        x=URGENCY_LABELS,
        y=feature_names,
        colorscale="RdBu",
        zmid=0,
        text=[[f"{v:.3f}" for v in row] for row in theta_matrix.T],
        texttemplate="%{text}",
        colorbar=dict(title="Weight"),
    ))
    fig.update_layout(
        title=f"{agent_name} Learned Weight Heatmap",
        xaxis_title="Urgency Level",
        yaxis_title="Feature",
        height=450,
    )
    return fig


def build_action_distribution(agents_action_counts: Dict[str, np.ndarray]) -> go.Figure:
    """Stacked bar chart: fraction of decisions at each urgency level per bandit."""
    if go is None:
        return None

    if not agents_action_counts:
        return go.Figure()

    colors = ["#1f77b4", "#aec7e8", "#ffbb78", "#ff7f0e", "#d62728"]
    fig = go.Figure()

    for urgency_idx, label in enumerate(URGENCY_LABELS):
        fractions = []
        agent_names = []
        for agent_name, counts in agents_action_counts.items():
            total = counts.sum()
            fractions.append(counts[urgency_idx] / total if total > 0 else 0)
            agent_names.append(agent_name)
        fig.add_trace(go.Bar(
            name=label,
            x=agent_names,
            y=fractions,
            marker_color=colors[urgency_idx],
        ))

    fig.update_layout(
        barmode="stack",
        title="Action Distribution by Agent",
        yaxis_title="Fraction of Decisions",
        height=350,
    )
    return fig


def build_feature_importance(theta_matrix: np.ndarray, agent_name: str = "") -> go.Figure:
    """Bar chart of feature importance: mean |theta_{a,i}| across actions."""
    if go is None:
        return None

    if theta_matrix is None or theta_matrix.size == 0:
        return go.Figure()

    importance = np.abs(theta_matrix).mean(axis=0)
    sorted_idx = np.argsort(importance)[::-1]
    feature_names = _feature_names(theta_matrix)

    fig = go.Figure(go.Bar(
        x=[feature_names[i] for i in sorted_idx],
        y=[importance[i] for i in sorted_idx],
        marker_color="steelblue",
    ))
    fig.update_layout(
        title=f"{agent_name} Feature Importance (mean |θ|)",
        xaxis_title="Feature",
        yaxis_title="|θ| mean across actions",
        height=350,
    )
    return fig


def build_learning_curves(agents_episode_is: Dict[str, List[float]], window: int = 10) -> go.Figure:
    """Rolling average IS vs episode for LinUCB and Thompson."""
    if go is None:
        return None

    import pandas as pd
    fig = go.Figure()
    for agent_name, is_vals in agents_episode_is.items():
        if not is_vals:
            continue
        rolling = pd.Series(is_vals).rolling(window=window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            y=rolling.tolist(),
            x=list(range(len(rolling))),
            name=f"{agent_name} (rolling {window})",
            line=dict(color=AGENT_COLORS.get(agent_name, "gray"), width=2),
        ))
    fig.update_layout(
        title="Bandit Learning Curves",
        xaxis_title="Episode",
        yaxis_title="Rolling IS (bps)",
        height=350,
    )
    return fig
