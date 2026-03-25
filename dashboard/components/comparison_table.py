"""Plotly batch metrics comparison table component — noob-friendly redesign."""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None


# ── Colour palette ────────────────────────────────────────────────────────────
AGENT_COLORS = {
    "TWAP":             "steelblue",
    "VWAP":             "seagreen",
    "AC_Optimal":       "darkorange",
    "POV":              "teal",
    "RegimeSwitchAC":   "crimson",
    "LinUCB":           "firebrick",
    "Thompson":         "mediumpurple",
    "EXP3":             "sienna",
    "KernelUCB":        "slateblue",
    "MetaAgent":        "black",
    "ThompsonACHybrid": "indigo",
    "Corral":           "darkcyan",
}

# Cell background helpers
_GREEN  = "#c6efce"   # good
_YELLOW = "#ffeb9c"   # okay
_RED    = "#ffc7ce"   # bad
_GREY   = "#f2f2f2"   # neutral / N/A
_HEADER = "#1a237e"   # dark navy


# ── Verdict logic ─────────────────────────────────────────────────────────────

def _verdict(mean_is: Optional[float], win_rate: Optional[float]) -> tuple[str, str]:
    """Return (label, cell_colour) for the quick-verdict column."""
    if mean_is is None:
        return "?", _GREY

    # Both signals available
    if win_rate is not None:
        if mean_is <= 0 and win_rate >= 0.65:
            return "Excellent", _GREEN
        if mean_is <= 5 and win_rate >= 0.55:
            return "Good", _GREEN
        if win_rate >= 0.50:
            return "Competitive", _YELLOW
        if win_rate >= 0.40:
            return "Weak", _RED
        return "Below baseline", _RED

    # No win-rate (TWAP itself)
    if mean_is <= 0:
        return "Filled below arrival", _GREEN
    if mean_is <= 10:
        return "Low cost", _YELLOW
    return "High cost", _RED


# ── Cell colour helpers ───────────────────────────────────────────────────────

def _colour_is(value: Optional[float]) -> str:
    if value is None:
        return _GREY
    if value <= 0:
        return _GREEN
    if value <= 10:
        return _YELLOW
    if value <= 30:
        return "#ffe0b2"   # light orange
    return _RED


def _colour_std(value: Optional[float]) -> str:
    if value is None:
        return _GREY
    if value <= 5:
        return _GREEN
    if value <= 20:
        return _YELLOW
    return _RED


def _colour_win_rate(value: Optional[float]) -> str:
    if value is None:
        return _GREY
    if value >= 0.65:
        return _GREEN
    if value >= 0.50:
        return _YELLOW
    return _RED


def _colour_pvalue(value: Optional[float]) -> str:
    if value is None:
        return _GREY
    if value <= 0.05:
        return _GREEN
    if value <= 0.10:
        return _YELLOW
    return _RED


def _colour_participation(value: Optional[float]) -> str:
    if value is None:
        return _GREY
    if 0.02 <= value <= 0.10:
        return _GREEN
    if 0.10 < value <= 0.20:
        return _YELLOW
    return _RED


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt(key: str, value) -> str:
    """Format a value as a human-readable string. Always returns a string (never NaN/null)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key == "p_value_vs_TWAP":
        v = float(value)
        if v <= 0.01:
            return f"{v:.4f}  (strong)"
        if v <= 0.05:
            return f"{v:.4f}  (significant)"
        if v <= 0.10:
            return f"{v:.4f}  (weak)"
        return f"{v:.4f}  (not significant)"
    if key in {"win_rate_vs_TWAP", "win_rate_vs_AC", "mean_participation_rate"}:
        return f"{float(value) * 100:.1f}%"
    if key == "n_episodes":
        return str(int(value))
    if key in {"total_cumulative_regret", "mean_per_episode_regret",
               "mean_IS_bps", "std_IS_bps", "median_IS_bps",
               "p25_IS_bps", "p75_IS_bps", "iqr_IS_bps",
               "max_IS_bps", "min_IS_bps", "mean_vwap_slippage_bps",
               "information_ratio"}:
        return f"{float(value):.2f}"
    return str(value)


# ── Main table ────────────────────────────────────────────────────────────────

# Columns shown in the primary (noob-friendly) table
# Each entry: (data_key, plain_header, colour_fn)
_PRIMARY_COLS: List[tuple] = [
    ("mean_IS_bps",             "Avg Cost\n(bps, lower=better)",        _colour_is),
    ("std_IS_bps",              "Consistency\n(bps, lower=better)",      _colour_std),
    ("win_rate_vs_TWAP",        "Beats TWAP\n(target >65%)",             _colour_win_rate),
    ("p_value_vs_TWAP",         "Confidence\n(target <0.05)",            _colour_pvalue),
    ("mean_participation_rate", "Market\nFootprint",                     _colour_participation),
    ("n_episodes",              "Test\nRuns",                            lambda v: _GREY),
]


def build_comparison_table(summary: Dict) -> "go.Figure":
    """
    Noob-friendly summary table.
    - Plain English headers.
    - Colour-coded cells (green = good, yellow = ok, red = bad).
    - Quick Verdict column.
    - Never shows 'null' — missing values always appear as '--'.
    """
    if go is None or not summary:
        return go.Figure()

    # Build rows
    agents: List[str] = list(summary.keys())

    # Rank by mean_IS_bps (lower = better)
    def _sort_key(name: str) -> float:
        v = summary[name].get("mean_IS_bps")
        return float(v) if isinstance(v, (int, float)) and not np.isnan(float(v)) else 9999.0

    agents = sorted(agents, key=_sort_key)

    # Header row
    headers = ["Rank", "Strategy"] + [col[1] for col in _PRIMARY_COLS] + ["Quick\nVerdict"]

    # Data rows + per-cell colours
    col_colours: Dict[str, List[str]] = {h: [] for h in headers}

    rank_vals, agent_vals = [], []
    col_data: Dict[str, List[str]] = {col[1]: [] for col in _PRIMARY_COLS}
    verdict_vals = []

    for rank, agent_name in enumerate(agents, start=1):
        s = summary[agent_name]

        # Safe getters — always return Python None (never pandas NaN, never "null")
        def _get(k):
            v = s.get(k)
            if isinstance(v, float) and np.isnan(v):
                return None
            return v

        mean_is  = _get("mean_IS_bps")
        win_rate = _get("win_rate_vs_TWAP")

        rank_vals.append(str(rank))
        agent_vals.append(agent_name)

        for data_key, plain_header, colour_fn in _PRIMARY_COLS:
            raw = _get(data_key)
            col_data[plain_header].append(_fmt(data_key, raw))
            col_colours[plain_header].append(colour_fn(raw))

        label, vc = _verdict(mean_is, win_rate if agent_name != "TWAP" else None)
        verdict_vals.append(label)
        col_colours["Quick\nVerdict"].append(vc)

        col_colours["Rank"].append(_GREY)
        col_colours["Strategy"].append(_GREY)

    # Assemble column value lists and colour lists
    all_col_vals = (
        [rank_vals, agent_vals]
        + [col_data[col[1]] for col in _PRIMARY_COLS]
        + [verdict_vals]
    )
    all_col_colours = [col_colours[h] for h in headers]

    fig = go.Figure(data=[go.Table(
        columnwidth=[40, 110] + [90] * len(_PRIMARY_COLS) + [110],
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color=_HEADER,
            font=dict(color="white", size=11),
            align="center",
            height=40,
        ),
        cells=dict(
            values=all_col_vals,
            fill_color=all_col_colours,
            align="center",
            font=dict(size=11, color="black"),
            height=30,
        ),
    )])

    fig.update_layout(
        title=dict(
            text="Strategy Performance at a Glance",
            font=dict(size=16),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=max(200, 80 + 32 * len(agents)),
    )
    return fig


# ── Colour legend (separate small figure) ────────────────────────────────────

def build_colour_legend() -> "go.Figure":
    """A tiny legend explaining the green/yellow/red colour scheme."""
    if go is None:
        return go.Figure()

    labels = ["Green = Good", "Yellow = Okay", "Red = Needs attention", "Grey = Not applicable"]
    colours = [_GREEN, _YELLOW, _RED, _GREY]

    fig = go.Figure(data=[go.Table(
        columnwidth=[160],
        header=dict(
            values=["<b>Colour Guide</b>"],
            fill_color=_HEADER,
            font=dict(color="white", size=11),
            align="center",
            height=30,
        ),
        cells=dict(
            values=[labels],
            fill_color=[colours],
            align="center",
            font=dict(size=11, color="black"),
            height=28,
        ),
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=160)
    return fig


# ── Violin plot ───────────────────────────────────────────────────────────────

def build_violin_plot(is_distributions: Dict) -> "go.Figure":
    """Violin plot of IS distributions per agent."""
    if go is None:
        return None

    fig = go.Figure()
    for agent_name, is_vals in is_distributions.items():
        if not is_vals:
            continue
        fig.add_trace(go.Violin(
            y=is_vals,
            name=agent_name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=AGENT_COLORS.get(agent_name, "gray"),
            opacity=0.6,
        ))
    fig.update_layout(
        title="Cost Distribution by Strategy  —  lower = cheaper execution,  negative = filled below arrival price",
        yaxis=dict(
            title=dict(text="Cost (bps)", standoff=10),
            automargin=True,
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        height=420,
    )
    return fig


# ── Win Rate Matrix ───────────────────────────────────────────────────────────

def build_win_rate_matrix(all_results: Dict) -> "go.Figure":
    """Heatmap: cell (i,j) = fraction of episodes where agent i beats agent j."""
    if go is None:
        return None

    agents = list(all_results.keys())
    n = len(agents)
    matrix = np.zeros((n, n))

    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            if i == j:
                matrix[i, j] = 0.5
            else:
                is_i = [r.implementation_shortfall_bps for r in all_results[agent_i]]
                is_j = [r.implementation_shortfall_bps for r in all_results[agent_j]]
                min_len = min(len(is_i), len(is_j))
                if min_len > 0:
                    wins = sum(1 for a, b in zip(is_i[:min_len], is_j[:min_len]) if a < b)
                    matrix[i, j] = wins / min_len

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=agents, y=agents,
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in matrix],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Head-to-Head Win Rate  (row = winner, column = loser — green = row wins more often)",
        height=400,
    )
    return fig
