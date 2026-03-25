"""Standalone fill computation extracted from ExecutionEnv.step()."""
from typing import Tuple
from simulator.impact_model import temporary_impact, permanent_impact


def compute_fill(
    v: float,
    mid_price: float,
    spread: float,
    eta: float,
    adv: float,
    alpha: float,
    gamma: float,
    slice_duration_s: float,
    side: str,
    permanent_impact_accumulated: float,
) -> Tuple[float, float, float]:
    """
    Compute fill price, impact cost, and permanent impact delta for a single slice.

    Parameters
    ----------
    v : float
        Volume to execute in this slice (BTC).
    mid_price : float
        Current mid-price including accumulated permanent impact.
    spread : float
        Raw bid-ask spread from the order book (in USD).
    eta : float
        Temporary impact coefficient (calibrated).
    adv : float
        Average daily volume (BTC).
    alpha : float
        Temporary impact exponent (calibrated).
    gamma : float
        Permanent impact coefficient (calibrated).
    slice_duration_s : float
        Duration of one execution slice in seconds.
    side : str
        "buy" or "sell".
    permanent_impact_accumulated : float
        Running sum of permanent price impact so far.

    Returns
    -------
    fill_price : float
        The execution price for this slice (USD per BTC).
    impact_cost_usd : float
        Absolute impact cost for this slice in USD.
    delta_permanent : float
        Signed change to the accumulated permanent impact.
    """
    # Half-spread as a fraction of mid
    spread_half = spread / (2.0 * mid_price) if mid_price > 0 else 0.0

    # Temporary impact fraction
    h = temporary_impact(v, slice_duration_s, eta, adv, alpha)

    # Fill price depends on side
    if side == "buy":
        fill_price = mid_price * (1.0 + spread_half + h)
    else:
        fill_price = mid_price * (1.0 - spread_half - h)

    impact_cost_usd = abs(fill_price - mid_price) * v

    # Permanent impact delta
    delta_s = permanent_impact(v, gamma, adv)
    if side == "buy":
        delta_permanent = delta_s
    else:
        delta_permanent = -delta_s

    return fill_price, impact_cost_usd, delta_permanent
