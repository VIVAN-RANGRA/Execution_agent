"""Unit tests for the execution simulator."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.impact_model import temporary_impact, permanent_impact, compute_ac_trajectory
from simulator.data_classes import MarketState, ExecutionConfig, Fill


class TestImpactModel:
    def test_temporary_impact_positive(self):
        h = temporary_impact(v=1.0, tau=60.0, eta=0.01, adv=1000.0, alpha=0.6)
        assert h >= 0.0

    def test_temporary_impact_zero_volume(self):
        h = temporary_impact(v=0.0, tau=60.0, eta=0.01, adv=1000.0, alpha=0.6)
        assert h == 0.0

    def test_permanent_impact_positive(self):
        d = permanent_impact(v=1.0, gamma=0.001, adv=1000.0)
        assert d >= 0.0

    def test_ac_trajectory_sum(self):
        traj = compute_ac_trajectory(10.0, 3600, 60, 0.1, 1e-5, 1e-6, 1000.0)
        assert abs(traj.sum() - 10.0) < 1e-6

    def test_ac_trajectory_length(self):
        traj = compute_ac_trajectory(10.0, 3600, 60, 0.1, 1e-5, 1e-6, 1000.0)
        assert len(traj) == 60

    def test_ac_trajectory_non_negative(self):
        traj = compute_ac_trajectory(10.0, 3600, 60, 0.1, 1e-5, 1e-6, 1000.0)
        assert np.all(traj >= 0)

    def test_fill_price_buy_above_mid(self):
        """Buy fill price must be >= mid price."""
        mid = 50000.0
        h = temporary_impact(1.0, 60.0, 0.01, 1000.0, 0.6)
        spread_half = 5.0 / (2 * mid)
        fill = mid * (1 + spread_half + h)
        assert fill >= mid

    def test_fill_price_sell_below_mid(self):
        """Sell fill price must be <= mid price."""
        mid = 50000.0
        h = temporary_impact(1.0, 60.0, 0.01, 1000.0, 0.6)
        spread_half = 5.0 / (2 * mid)
        fill = mid * (1 - spread_half - h)
        assert fill <= mid


class TestDataClasses:
    def test_market_state_creation(self):
        ms = MarketState(
            timestamp_ms=1000000,
            mid_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            spread=10.0,
            recent_volume_30s=10.0,
            recent_volatility_60s=0.0001,
            recent_volatility_300s=0.0002,
            ofi_10s=0.1,
            bid_qty=5.0,
            ask_qty=4.0,
            minute_of_day=600,
        )
        assert ms.mid_price == 50000.0
        assert ms.minute_of_day == 600

    def test_fill_creation(self):
        f = Fill(timestamp_ms=1000, quantity_filled=1.0, fill_price=50001.0,
                 market_impact_cost=1.0, slice_index=0)
        assert f.quantity_filled == 1.0
