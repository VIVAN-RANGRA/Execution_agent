"""Evaluation metrics: IS, VWAP slippage, participation rate, timing risk, win rate, Mann-Whitney."""
import numpy as np
from scipy import stats
from typing import List
from simulator.data_classes import Fill, EpisodeResult


def implementation_shortfall_bps(fills: List[Fill], arrival_price: float) -> float:
    """
    IS_bps = (execution_VWAP - arrival_price) / arrival_price * 10000
    Lower is better for buy orders.
    """
    if not fills or arrival_price <= 0:
        return 0.0
    total_qty = sum(f.quantity_filled for f in fills)
    if total_qty <= 0:
        return 0.0
    exec_vwap = sum(f.fill_price * f.quantity_filled for f in fills) / total_qty
    return (exec_vwap - arrival_price) / arrival_price * 10000.0


def vwap_slippage_bps(fills: List[Fill], market_trades_prices: np.ndarray,
                       market_trades_qtys: np.ndarray) -> float:
    """
    VWAP_slippage_bps = (execution_VWAP - market_VWAP) / market_VWAP * 10000
    """
    if not fills:
        return 0.0
    total_qty = sum(f.quantity_filled for f in fills)
    if total_qty <= 0:
        return 0.0
    exec_vwap = sum(f.fill_price * f.quantity_filled for f in fills) / total_qty

    if len(market_trades_prices) == 0 or market_trades_qtys.sum() <= 0:
        return 0.0
    market_vwap = (market_trades_prices * market_trades_qtys).sum() / market_trades_qtys.sum()
    if market_vwap <= 0:
        return 0.0
    return (exec_vwap - market_vwap) / market_vwap * 10000.0


def participation_rate(fills: List[Fill], total_market_volume: float) -> float:
    """
    participation_rate = total_quantity_executed / total_market_volume
    Should be in [0.01, 0.30] for realistic institutional execution.
    """
    if not fills or total_market_volume <= 0:
        return 0.0
    total_qty = sum(f.quantity_filled for f in fills)
    return total_qty / total_market_volume


def timing_risk_bps(fills: List[Fill], arrival_price: float) -> float:
    """
    timing_risk = std(fill_prices) / arrival_price * 10000  [bps]
    """
    if not fills or arrival_price <= 0:
        return 0.0
    prices = np.array([f.fill_price for f in fills])
    return float(np.std(prices) / arrival_price * 10000.0)


def win_rate_vs_baseline(agent_is_list: List[float], baseline_is_list: List[float]) -> float:
    """
    Fraction of episodes where agent_IS < baseline_IS.
    """
    if not agent_is_list or not baseline_is_list:
        return 0.0
    wins = sum(1 for a, b in zip(agent_is_list, baseline_is_list) if a < b)
    return wins / len(agent_is_list)


def mann_whitney_p_value(agent_is_list: List[float], baseline_is_list: List[float]) -> float:
    """
    Mann-Whitney U test (non-parametric) comparing two IS distributions.
    Returns p-value. p < 0.05 = statistically significant improvement.
    """
    if len(agent_is_list) < 2 or len(baseline_is_list) < 2:
        return 1.0
    _, p_value = stats.mannwhitneyu(
        agent_is_list, baseline_is_list, alternative="less"
    )
    return float(p_value)


def compute_episode_metrics(
    fills: List[Fill],
    arrival_price: float,
    agent_name: str,
    episode_seed: int,
    market_trades_prices: np.ndarray = None,
    market_trades_qtys: np.ndarray = None,
    total_market_volume: float = None,
) -> EpisodeResult:
    """Compute all metrics for a single episode and return EpisodeResult."""
    from simulator.data_classes import EpisodeResult

    if market_trades_prices is None:
        market_trades_prices = np.array([f.fill_price for f in fills])
    if market_trades_qtys is None:
        market_trades_qtys = np.array([f.quantity_filled for f in fills])
    if total_market_volume is None:
        total_market_volume = sum(f.quantity_filled for f in fills) * 10  # rough estimate

    is_bps = implementation_shortfall_bps(fills, arrival_price)
    vwap_slip = vwap_slippage_bps(fills, market_trades_prices, market_trades_qtys)
    part_rate = participation_rate(fills, total_market_volume)
    final_inv = 0.0  # filled

    return EpisodeResult(
        fills=fills,
        arrival_price=arrival_price,
        final_inventory=final_inv,
        implementation_shortfall_bps=is_bps,
        vwap_slippage_bps=vwap_slip,
        participation_rate=part_rate,
        agent_name=agent_name,
        episode_seed=episode_seed,
    )
