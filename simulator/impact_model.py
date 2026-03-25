"""Almgren-Chriss temporary and permanent market impact model."""
from collections import OrderedDict

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is optional
    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


@njit(cache=True)
def _temporary_impact_impl(v: float, tau: float, eta: float, adv: float, alpha: float) -> float:
    if tau <= 0.0 or adv <= 0.0 or eta == 0.0:
        return 0.0
    participation = abs(v) / (adv * tau)
    if participation <= 0.0:
        return 0.0
    return eta * (participation ** alpha)


@njit(cache=True)
def _permanent_impact_impl(v: float, gamma: float, adv: float) -> float:
    if adv <= 0.0 or gamma == 0.0:
        return 0.0
    return gamma * (v / adv)


def temporary_impact(v: float, tau: float, eta: float, adv: float, alpha: float) -> float:
    """
    Compute dimensionless temporary price impact fraction.
    h(v, tau) = eta * (|v| / (ADV * tau))^alpha
    """
    return float(_temporary_impact_impl(v, tau, eta, adv, alpha))


def permanent_impact(v: float, gamma: float, adv: float) -> float:
    """
    Compute permanent price shift (in USD) after executing volume v.
    delta_S = gamma * (v / ADV)
    """
    return float(_permanent_impact_impl(v, gamma, adv))


_AC_CACHE_MAX_SIZE = 32
_ac_trajectory_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def _normalize_cache_key(
    total_quantity: float,
    time_horizon_seconds: int,
    num_slices: int,
    risk_aversion: float,
    sigma_per_second: float,
    eta: float,
    adv: float,
) -> tuple:
    return (
        round(float(total_quantity), 12),
        int(time_horizon_seconds),
        int(num_slices),
        round(float(risk_aversion), 12),
        round(float(sigma_per_second), 12),
        round(float(eta), 12),
        round(float(adv), 12),
    )


def compute_ac_trajectory(
    total_quantity: float,
    time_horizon_seconds: int,
    num_slices: int,
    risk_aversion: float,
    sigma_per_second: float,
    eta: float,
    adv: float,
) -> np.ndarray:
    """
    Compute the Almgren-Chriss IS-optimal execution trajectory.
    Returns array of length num_slices summing to total_quantity.

    Results are cached (up to 32 unique parameter combinations) to avoid
    recomputation across episodes with the same config.
    """
    if num_slices <= 0:
        return np.array([], dtype=np.float64)

    cache_key = _normalize_cache_key(
        total_quantity,
        time_horizon_seconds,
        num_slices,
        risk_aversion,
        sigma_per_second,
        eta,
        adv,
    )
    cached = _ac_trajectory_cache.get(cache_key)
    if cached is not None:
        _ac_trajectory_cache.move_to_end(cache_key)
        return cached.copy()

    total_time = float(time_horizon_seconds)
    slices = int(num_slices)
    total_qty = float(total_quantity)
    tau = total_time / slices if slices > 0 else 0.0

    if slices <= 0:
        result = np.array([], dtype=np.float64)
    elif eta <= 0.0 or tau <= 0.0:
        result = np.full(slices, total_qty / slices, dtype=np.float64)
    else:
        kappa = np.sqrt(max(risk_aversion, 0.0) * (sigma_per_second ** 2) / eta)
        kappa_t = min(kappa * total_time, 20.0)
        kappa = (kappa_t / total_time) if total_time > 0 else 0.0

        sinh_kappa_t = np.sinh(kappa_t)
        if sinh_kappa_t == 0.0:
            result = np.full(slices, total_qty / slices, dtype=np.float64)
        else:
            indices = np.arange(slices, dtype=np.float64)
            holdings = total_qty * np.sinh(kappa * (total_time - indices * tau)) / sinh_kappa_t
            total = holdings.sum()
            if total > 0.0:
                result = holdings * (total_qty / total)
            else:
                result = np.full(slices, total_qty / slices, dtype=np.float64)

    if len(_ac_trajectory_cache) >= _AC_CACHE_MAX_SIZE:
        _ac_trajectory_cache.popitem(last=False)
    _ac_trajectory_cache[cache_key] = result.copy()
    return result
