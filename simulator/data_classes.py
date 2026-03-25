"""Data classes: MarketState, ExecutionConfig, Fill, EpisodeResult."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MarketState:
    timestamp_ms: int
    mid_price: float
    bid_price: float
    ask_price: float
    spread: float
    recent_volume_30s: float
    recent_volatility_60s: float
    recent_volatility_300s: float
    ofi_10s: float
    bid_qty: float
    ask_qty: float
    minute_of_day: int


@dataclass
class ExecutionConfig:
    total_quantity: float
    time_horizon_seconds: int
    num_slices: int
    arrival_price: float
    side: str = 'buy'
    risk_aversion: float = 0.1


@dataclass
class Fill:
    timestamp_ms: int
    quantity_filled: float
    fill_price: float
    market_impact_cost: float
    slice_index: int


@dataclass
class EpisodeResult:
    fills: List[Fill]
    arrival_price: float
    final_inventory: float
    implementation_shortfall_bps: float
    vwap_slippage_bps: float
    participation_rate: float
    agent_name: str
    episode_seed: int
