"""Plotly execution trajectory chart component."""
from __future__ import annotations
from typing import List, Dict

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # Optional dependency for dashboard rendering
    go = None
    make_subplots = None

from simulator.data_classes import Fill


AGENT_COLORS = {
    "TWAP": "steelblue",
    "VWAP": "seagreen",
    "AC_Optimal": "darkorange",
    "POV": "teal",
    "RegimeSwitchAC": "crimson",
    "LinUCB": "firebrick",
    "Thompson": "mediumpurple",
    "EXP3": "sienna",
    "KernelUCB": "slateblue",
    "MetaAgent": "black",
    "ThompsonACHybrid": "indigo",
    "Corral": "darkcyan",
}


def build_trajectory_chart(
    agents_fills: Dict[str, List[Fill]],
    arrival_price: float,
    market_mid_prices: List[float] = None,
    market_timestamps: List[int] = None,
) -> go.Figure:
    if go is None or make_subplots is None:
        return None

    """
    Panel 1: Execution Trajectory
    - Primary Y: price (USDT) — market mid-price (gray), arrival price (dotted), fill prices per agent
    - Secondary Y: cumulative inventory filled as % (0→100%)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Market mid-price
    if market_mid_prices and market_timestamps:
        elapsed_min = [(t - market_timestamps[0]) / 60000 for t in market_timestamps]
        fig.add_trace(go.Scatter(
            x=elapsed_min, y=market_mid_prices,
            name="Market Mid", line=dict(color="gray", width=1),
            opacity=0.6,
        ), secondary_y=False)

    # Arrival price horizontal line
    if agents_fills:
        max_min = max(
            (f.slice_index for fills in agents_fills.values() for f in fills),
            default=60
        )
        fig.add_hline(
            y=arrival_price, line_dash="dot", line_color="black",
            annotation_text="Arrival Price", annotation_position="right",
        )

    # Colors per agent
    for agent_name, fills in agents_fills.items():
        if not fills:
            continue
        color = AGENT_COLORS.get(agent_name, "gray")
        x_vals = [f.slice_index for f in fills]
        y_prices = [f.fill_price for f in fills]

        # Fill prices
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_prices,
            name=f"{agent_name} Fill Price",
            mode="markers+lines",
            marker=dict(color=color, size=5),
            line=dict(color=color, width=1, dash="dash"),
        ), secondary_y=False)

        # Cumulative inventory filled %
        total_qty = sum(f.quantity_filled for f in fills)
        cumulative = []
        cum = 0
        for f in fills:
            cum += f.quantity_filled
            cumulative.append(cum / total_qty * 100 if total_qty > 0 else 0)

        fig.add_trace(go.Scatter(
            x=x_vals, y=cumulative,
            name=f"{agent_name} Filled %",
            mode="lines",
            line=dict(color=color, width=1),
            opacity=0.4,
        ), secondary_y=True)

    fig.update_xaxes(title_text="Time Step (minutes)")
    fig.update_yaxes(title_text="Price (USDT)", secondary_y=False)
    fig.update_yaxes(title_text="Inventory Filled (%)", secondary_y=True, range=[0, 110])
    fig.update_layout(title="Execution Trajectory", height=400)
    return fig
