"""Reward computation functions extracted from ExecutionEnv.step().

These are designed to be swappable: to change the reward design, replace
these functions (or subclass / wrap them) without touching the environment.
"""
import numpy as np
from typing import List, Optional
from simulator.data_classes import Fill


def compute_step_reward(
    fill_price: float,
    arrival_price: float,
    v: float,
    total_quantity: float,
) -> float:
    """
    Per-slice reward: negative Implementation Shortfall contribution in bps,
    weighted by the fraction of total order executed in this slice.

    Parameters
    ----------
    fill_price : float
        The execution price for this slice (USD per BTC).
    arrival_price : float
        The arrival (benchmark) price at episode start (USD per BTC).
    v : float
        Volume executed in this slice (BTC).
    total_quantity : float
        Total order quantity (BTC).

    Returns
    -------
    float
        Reward contribution for this slice.
    """
    if arrival_price <= 0 or total_quantity <= 0:
        return 0.0
    return -(fill_price - arrival_price) / arrival_price * 10000.0 * (v / total_quantity)


def compute_terminal_reward(fills: List[Fill], arrival_price: float,
                            fill_prices: Optional[np.ndarray] = None,
                            fill_quantities: Optional[np.ndarray] = None) -> float:
    """
    Terminal bonus/penalty based on overall execution VWAP vs arrival price.

    Supports two modes:
    1. Legacy: pass fills as List[Fill] (backward compatible)
    2. Vectorized: pass fill_prices and fill_quantities as numpy arrays (fast path)

    Parameters
    ----------
    fills : List[Fill]
        All fills recorded during the episode. Used if arrays not provided.
    arrival_price : float
        The arrival (benchmark) price at episode start (USD per BTC).
    fill_prices : np.ndarray, optional
        Pre-allocated numpy array of fill prices (fast path).
    fill_quantities : np.ndarray, optional
        Pre-allocated numpy array of fill quantities (fast path).

    Returns
    -------
    float
        Terminal reward (negative IS in bps, scaled by 0.1).
    """
    if arrival_price <= 0:
        return 0.0

    # Fast path: use numpy arrays if provided
    if fill_prices is not None and fill_quantities is not None:
        if len(fill_prices) == 0:
            return 0.0
        total_qty = fill_quantities.sum()
        if total_qty <= 0:
            return 0.0
        exec_vwap = (fill_prices * fill_quantities).sum() / total_qty
        is_bps = (exec_vwap - arrival_price) / arrival_price * 10000.0
        return float(-is_bps * 0.1)

    # Legacy path: use Fill objects
    if not fills:
        return 0.0

    total_qty = sum(f.quantity_filled for f in fills)
    if total_qty <= 0:
        return 0.0

    exec_vwap = sum(f.fill_price * f.quantity_filled for f in fills) / total_qty
    is_bps = (exec_vwap - arrival_price) / arrival_price * 10000.0
    return float(-is_bps * 0.1)
