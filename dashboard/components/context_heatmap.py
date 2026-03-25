"""Plotly context vector heatmap component."""
from __future__ import annotations
from typing import List

import numpy as np

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # Optional dependency for dashboard rendering
    go = None

from features.feature_engineer import FeatureEngineer


def _feature_names(width: int) -> List[str]:
    if width <= len(FeatureEngineer.BASE_FEATURE_NAMES):
        return FeatureEngineer.BASE_FEATURE_NAMES[:width]
    return FeatureEngineer.FEATURE_NAMES[:width]


def build_context_heatmap(context_history: List[np.ndarray]) -> go.Figure:
    if go is None:
        return None

    """
    Heatmap of the feature vector over time in a single episode.
    X: time steps; Y: feature names; Color: feature value.
    """
    if not context_history:
        return go.Figure()

    matrix = np.array(context_history).T
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(len(context_history))),
        y=_feature_names(matrix.shape[0]),
        colorscale="RdBu",
        zmid=0,
    ))
    fig.update_layout(
        title="Context Vector Over Episode",
        xaxis_title="Time Step",
        yaxis_title="Feature",
        height=400,
    )
    return fig


def build_urgency_heatmap(agents_actions: dict) -> go.Figure:
    if go is None:
        return None

    """
    X: time steps; Y: agents; Color: urgency level (0=blue → 4=red).
    """
    if not agents_actions:
        return go.Figure()

    agents = list(agents_actions.keys())
    max_steps = max(len(v) for v in agents_actions.values())
    matrix = np.full((len(agents), max_steps), np.nan)

    for i, agent in enumerate(agents):
        actions = agents_actions[agent]
        matrix[i, :len(actions)] = actions

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(max_steps)),
        y=agents,
        colorscale="RdYlBu_r",
        zmin=0, zmax=4,
        colorbar=dict(title="Urgency", tickvals=[0, 1, 2, 3, 4],
                      ticktext=["0-Very Passive", "1-Passive", "2-On Pace",
                                "3-Aggressive", "4-Very Aggressive"]),
    ))
    fig.update_layout(
        title="Urgency Level Heatmap",
        xaxis_title="Time Step",
        yaxis_title="Agent",
        height=300,
    )
    return fig


def build_is_contribution_chart(agents_fills: dict, arrival_price: float) -> go.Figure:
    """Grouped bar chart: IS contribution per time step per agent."""
    if go is None:
        return None

    if not agents_fills or arrival_price <= 0:
        return go.Figure()

    colors = {"TWAP": "blue", "VWAP": "green", "AC_Optimal": "orange",
              "LinUCB": "red", "Thompson": "purple"}

    fig = go.Figure()
    for agent_name, fills in agents_fills.items():
        if not fills:
            continue
        total_qty = sum(f.quantity_filled for f in fills)
        if total_qty <= 0:
            continue
        x_vals = [f.slice_index for f in fills]
        contributions = [
            -(f.fill_price - arrival_price) / arrival_price * 10000 * (f.quantity_filled / total_qty)
            for f in fills
        ]
        fig.add_trace(go.Bar(
            x=x_vals, y=contributions,
            name=agent_name,
            marker_color=colors.get(agent_name, "gray"),
            opacity=0.7,
        ))
    fig.update_layout(
        barmode="group",
        title="IS Contribution per Slice",
        xaxis_title="Slice Index",
        yaxis_title="IS Contribution (bps)",
        height=350,
    )
    return fig
